#!/usr/bin/env bash
# Phase 0 on Lambda A100: full Thinkingbox stack + vLLM wiring check.
# Runs INSIDE the project repo on the pod (cloned by bin/run.sh).
# Heavy artifacts -> repo dir on persistent NFS; logs -> phase0_logs/.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
mkdir -p artifacts phase0_logs typesense-data
LOG() { echo "[$(date +%H:%M:%S)] $*" | tee -a phase0_logs/main.log; }

# ---- 1. clone + pin benchmark repos side-by-side ----
PARENT="$(dirname "$ROOT")"
[ -d "$PARENT/thinkingbox" ] || git clone --depth 50 https://github.com/microsoft/thinkingbox.git "$PARENT/thinkingbox"
[ -d "$PARENT/thinkingbox-data" ] || git clone https://github.com/microsoft/thinkingbox-data.git "$PARENT/thinkingbox-data"
git -C "$PARENT/thinkingbox-data" checkout thinkingbox-bench-v1.0
{
  echo "thinkingbox: $(git -C "$PARENT/thinkingbox" rev-parse HEAD)"
  echo "thinkingbox-data: $(git -C "$PARENT/thinkingbox-data" rev-parse HEAD)"
} > artifacts/pinned_commits.txt

# ---- 2. framework env (uv, python 3.12) ----
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
cd "$PARENT/thinkingbox"
uv venv --python 3.12 --allow-existing
export UV_PROJECT_ENVIRONMENT="$PARENT/thinkingbox/.venv" VIRTUAL_ENV="$PARENT/thinkingbox/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
uv sync --group dev
uv pip install --config-settings editable-mode=compat \
  -e ../thinkingbox-data/servers/tb_business_ops_servers_202606
bash scripts/install_typesense.sh
typesense-server --version || true   # no --version flag; banner + rc=1 is expected

# ---- 3. vLLM (bf16 — A100 40GB) ----
uv venv --python 3.12 --allow-existing "$ROOT/vllm-venv"
uv pip install --python "$ROOT/vllm-venv/bin/python" vllm
"$ROOT/vllm-venv/bin/vllm" --version 2>/dev/null || "$ROOT/vllm-venv/bin/vllm" --version

"$ROOT/vllm-venv/bin/vllm" serve Qwen/Qwen3-4B \
  --served-model-name Qwen3-4B --port 8000 --dtype bfloat16 \
  --max-model-len 32768 --enable-auto-tool-choice --tool-call-parser hermes \
  --gpu-memory-utilization 0.85 > "$ROOT/phase0_logs/vllm.log" 2>&1 &
VLLM_PID=$!
for i in $(seq 1 120); do
  curl -s http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && break
  kill -0 $VLLM_PID 2>/dev/null || { LOG "vllm died"; tail -30 "$ROOT/phase0_logs/vllm.log"; exit 1; }
  sleep 5
done
LOG "vLLM up"

# ---- 4. Typesense + Session Proxy ----
TS_DATA="$ROOT/typesense-data"; mkdir -p "$TS_DATA"
typesense-server --data-dir "$TS_DATA" --api-key Fake \
  > "$ROOT/phase0_logs/typesense.log" 2>&1 &
for i in $(seq 1 60); do curl -s http://127.0.0.1:8108/health >/dev/null 2>&1 && break; sleep 2; done
LOG "Typesense up"

export THINKINGBOX_DATA="../thinkingbox-data"
export TB_MCP_START_SERVERS_FILE="../thinkingbox-data/servers/servers.yaml"
export TYPESENSE_API_KEY=Fake
nohup ./scripts/background_tasks.sh > "$ROOT/phase0_logs/background_tasks.log" 2>&1 &
for i in $(seq 1 90); do
  grep -q "All processes are running" "$ROOT/phase0_logs/background_tasks.log" 2>/dev/null && break
  sleep 5
done
grep -q "All processes are running" "$ROOT/phase0_logs/background_tasks.log" \
  || { LOG "services not ready"; tail -20 "$ROOT/phase0_logs/background_tasks.log"; exit 1; }
LOG "Session Proxy + MCP servers up"

# ---- 5. config + wiring check ----
cp "$ROOT/configs/llm_config_template.yaml" "$ROOT/configs/wiring_a100.yaml"
sed -i 's/Qwen3-4B/Qwen3-4B/g' "$ROOT/configs/wiring_a100.yaml"

uv run tb infer -c "$ROOT/configs/wiring_a100.yaml" \
  --dataset ../thinkingbox-data/dataset --agent think \
  --name banking.py:test_get_balance_savings \
  --output "$ROOT/artifacts/wiring_check_output_a100.yaml"
uv run tb pp "$ROOT/artifacts/wiring_check_output_a100.yaml" \
  | tee "$ROOT/artifacts/wiring_check_pretty_a100.txt"

LOG "WIRING CHECK COMPLETE"
kill $VLLM_PID 2>/dev/null || true
echo '{"ok": true}' > "$ROOT/artifacts/phase0_result.json"
