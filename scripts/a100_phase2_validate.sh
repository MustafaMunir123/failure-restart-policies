#!/usr/bin/env bash
# Phase 2 host validation: controller unit tests + end-to-end adapter check.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"; PARENT="$(dirname "$ROOT")"
mkdir -p artifacts phase2_logs
LOG() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$ROOT/phase2_logs/main.log"; }

# ---- venv with our deps (framework env reused from phase-1 NFS state) ----
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
source /lambda/nfs/ic-fs-2/envs/systrem-recovery-modes.env 2>/dev/null || true

# ---- 1. unit + contract tests ----
LOG "running test suite..."
uv run --python 3.12 --with pytest --with pyyaml --with jsonschema \
  pytest tests/ -q 2>&1 | tail -15
[ "${PIPESTATUS[0]}" = 0 ] || { LOG "TESTS FAILED"; exit 1; }
LOG "all tests green"

# ---- 2. framework stack up (same proven chain) ----
[ -d "$PARENT/thinkingbox" ] || { LOG "framework repos missing - run phase1 first"; exit 1; }
git -C "$PARENT/thinkingbox-data" checkout thinkingbox-bench-v1.0
cd "$PARENT/thinkingbox"
export UV_PROJECT_ENVIRONMENT="$PARENT/thinkingbox/.venv" VIRTUAL_ENV="$PARENT/thinkingbox/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
uv sync --group dev
bash scripts/install_typesense.sh

typesense-server --data-dir "$ROOT/typesense-data" --api-key Fake \
  > "$ROOT/phase2_logs/typesense.log" 2>&1 &
for i in $(seq 1 60); do curl -s http://127.0.0.1:8108/health >/dev/null 2>&1 && break; sleep 2; done
export THINKINGBOX_DATA="../thinkingbox-data"
export TB_MCP_START_SERVERS_FILE="../thinkingbox-data/servers/servers.yaml"
export TYPESENSE_API_KEY=Fake
nohup ./scripts/background_tasks.sh > "$ROOT/phase2_logs/background_tasks.log" 2>&1 &
READY=0
for i in $(seq 1 90); do
  grep -q "All processes are running" "$ROOT/phase2_logs/background_tasks.log" 2>/dev/null && { READY=1; break; }
  sleep 5
done
[ "$READY" = 1 ] || { LOG "services not ready"; exit 1; }
LOG "services up"

# ---- 3. vLLM Qwen3-4B bf16 ----
"$ROOT/vllm-venv/bin/vllm" serve Qwen/Qwen3-4B \
  --served-model-name Qwen3-4B --port 8000 --dtype bfloat16 \
  --max-model-len 32768 --enable-auto-tool-choice --tool-call-parser hermes \
  --gpu-memory-utilization 0.85 > "$ROOT/phase2_logs/vllm.log" 2>&1 &
VPID=$!
for i in $(seq 1 150); do
  curl -s http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && break
  kill -0 $VPID 2>/dev/null || { LOG "vllm died"; exit 1; }
  sleep 5
done
LOG "vLLM up"

# ---- 4. e2e validation: orchestrator drives 5 tasks x 4 policies via adapter ----
cat > "$ROOT/artifacts/adapter_e2e_cfg.json" <<JSON
{
  "dataset": "../thinkingbox-data/dataset",
  "base_config": "$ROOT/configs/wiring_a100.yaml",
  "thinkingbox_dir": "$PARENT/thinkingbox",
  "workdir": "$ROOT/artifacts/adapter_work",
  "p2_mechanism": "replay",
  "tasks": [],
  "seeds": [101]
}
JSON
TASKS=$(python3 -c 'import json,os;print(" ".join(json.load(open(os.environ["ROOT"]+"/artifacts/smoke_tasks.json"))))')
python3 - <<EOF
import json, os
cfg = json.load(open("$ROOT/artifacts/adapter_e2e_cfg.json"))
cfg["tasks"] = [{"task_id": t, "domain": t.split(":")[0]} for t in """$TASKS""".split()]
json.dump(cfg, open("$ROOT/artifacts/adapter_e2e_cfg.json", "w"), indent=2)
EOF

uv pip install --python "$VIRTUAL_ENV/bin/python" pyyaml jsonschema pytest
PYTHONPATH="$ROOT" uv run --no-project python - <<'EOF' 2>&1 | tee -a "$ROOT/phase2_logs/e2e.log"
import json, os, sys
sys.path.insert(0, os.environ["ROOT"])
from adapters.thinkingbox_adapter import ThinkingboxAdapter
from orchestration.orchestrator import execute_run

cfg = json.load(open(os.environ["ROOT"] + "/artifacts/adapter_e2e_cfg.json"))
env = ThinkingboxAdapter(cfg)
n_ok = n_total = 0
for task in cfg["tasks"]:
    for policy in ["P0_stop", "P1_blind_restart", "P2_same_session", "P3_restart_note"]:
        rec = execute_run(task["task_id"], task["domain"], "Qwen3-4B", policy,
                          cfg["seeds"][0], env)
        n_total += 1
        n_ok += rec["flags"].get("harness_error") is not True
        print(policy, task["task_id"], "->",
              "harness_error" if rec["flags"].get("harness_error") else "ok",
              "| visible_error:", bool(rec["attempt_1_hit_visible_error"]))
print(f"E2E: {n_ok}/{n_total} runs without harness error")
open(os.environ["ROOT"] + "/artifacts/phase2_result.json", "w").write(
    json.dumps({"e2e_ok": n_ok, "e2e_total": n_total}))
EOF
LOG "PHASE 2 VALIDATION COMPLETE"
kill $VPID 2>/dev/null || true
