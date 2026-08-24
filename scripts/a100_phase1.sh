#!/usr/bin/env bash
# Phase 1 smoke matrix on Lambda A100: feasibility + harness validation.
# 3 sizes (bf16) x 5 tasks x 2 reps thinking-off + Qwen3-4B thinking-on diagnostic.
# Sequential serving. Artifacts -> artifacts/, logs -> phase1_logs/ (persistent NFS).
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"; PARENT="$(dirname "$ROOT")"
mkdir -p artifacts phase1_logs typesense-data
LOG() { echo "[$(date +%H:%M:%S)] $*" | tee -a phase1_logs/main.log; }
FAIL=0

# ---- 1. benchmark repos: clone/pin ----
[ -d "$PARENT/thinkingbox" ] || git clone --depth 50 https://github.com/microsoft/thinkingbox.git "$PARENT/thinkingbox"
[ -d "$PARENT/thinkingbox-data" ] || git clone https://github.com/microsoft/thinkingbox-data.git "$PARENT/thinkingbox-data"
git -C "$PARENT/thinkingbox-data" checkout thinkingbox-bench-v1.0
{
  echo "thinkingbox: $(git -C "$PARENT/thinkingbox" rev-parse HEAD)"
  echo "thinkingbox-data: $(git -C "$PARENT/thinkingbox-data" rev-parse HEAD)"
} > artifacts/pinned_commits.txt

# ---- 2. framework env ----
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

# ---- 3. vLLM env ----
uv venv --python 3.12 --allow-existing "$ROOT/vllm-venv"
uv pip install --python "$ROOT/vllm-venv/bin/python" vllm

# ---- 4. services ----
typesense-server --data-dir "$ROOT/typesense-data" --api-key Fake \
  > "$ROOT/phase1_logs/typesense.log" 2>&1 &
for i in $(seq 1 60); do curl -s http://127.0.0.1:8108/health >/dev/null 2>&1 && break; sleep 2; done
LOG "Typesense up"

export THINKINGBOX_DATA="../thinkingbox-data"
export TB_MCP_START_SERVERS_FILE="../thinkingbox-data/servers/servers.yaml"
export TYPESENSE_API_KEY=Fake
nohup ./scripts/background_tasks.sh > "$ROOT/phase1_logs/background_tasks.log" 2>&1 &
READY=0
for i in $(seq 1 90); do
  grep -q "All processes are running" "$ROOT/phase1_logs/background_tasks.log" 2>/dev/null && { READY=1; break; }
  sleep 5
done
[ "$READY" = 1 ] || { LOG "services not ready"; tail -20 "$ROOT/phase1_logs/background_tasks.log"; exit 1; }
LOG "Session Proxy + MCP servers up"

# ---- 5. smoke task selection (deterministic from canonical testlist) ----
python3 - <<'EOF'
import yaml, json, os
tl = yaml.safe_load(open("../thinkingbox-data/releases/thinkingbox_bench_v1/testlist_thinkingbox_bench_v1.yaml"))
if isinstance(tl, dict):
    tl = tl.get("tests") or tl.get("tasks") or list(tl.values())
names = []
for e in tl:
    n = e if isinstance(e, str) else (e.get("name") or e.get("test") or "")
    if isinstance(n, str) and "." in n and ":" in n:
        names.append(n)
names = sorted(set(names))
picked, seen = [], set()
for n in names:
    f = n.split(":")[0]
    if f not in seen:
        picked.append(n); seen.add(f)
    if len(picked) == 5:
        break
out = os.path.join(os.environ["ROOT"], "artifacts", "smoke_tasks.json")
json.dump(picked, open(out, "w"))
print("SMOKE TASKS:", picked)
EOF
TASKS=$(python3 -c 'import json,os;print("\n".join(json.load(open(os.path.join(os.environ["ROOT"],"artifacts","smoke_tasks.json")))))')
[ -n "$TASKS" ] || { LOG "no tasks selected"; exit 1; }

write_config() { # $1=path $2=served-name $3=thinking(true/false)
  if [ "$3" = true ]; then local R1="true" R2="content"; else R1="false" R2="none"; fi
  cat > "$1" <<CFG
mcp_proxy:
  endpoint_url: "http://127.0.0.1:7111"
  timeout: 300.0

orchestrator:
  type: thinkingbox
  agent_model:
    type: aoai
    deployment: "$2"
    endpoint_url: "http://127.0.0.1:8000/v1/chat/completions"
    credential:
      type: api-key
      api_key: "EMPTY"
    is_reasoning: $R1
    reasoning_source: $R2
    temperature: 0.7
    max_completion_tokens: 4096
    timeout: 120.0

judge_model:
  type: aoai
  deployment: "$2"
  endpoint_url: "http://127.0.0.1:8000/v1/chat/completions"
  credential:
    type: api-key
    api_key: "EMPTY"
  is_reasoning: false
  reasoning_source: none
  temperature: 0.0
  seed: 42
  max_completion_tokens: 128
  timeout: 60.0

judge_type: "legacy"

user_model:
  type: aoai
  deployment: "$2"
  endpoint_url: "http://127.0.0.1:8000/v1/chat/completions"
  credential:
    type: api-key
    api_key: "EMPTY"
  is_reasoning: false
  reasoning_source: none
  temperature: 0.3
  seed: 42
  max_completion_tokens: 512
  timeout: 60.0
CFG
}

serve_and_run() { # $1=label $2=hf_id $3=extra_flags $4=cfg_path $5=reps_per_task $6=tag
  LOG "serving $1 ..."
  # shellcheck disable=SC2086
  "$ROOT/vllm-venv/bin/vllm" serve "$2" \
    --served-model-name "$1" --port 8000 --dtype bfloat16 \
    --max-model-len 32768 --enable-auto-tool-choice --tool-call-parser hermes \
    --gpu-memory-utilization 0.85 $3 \
    > "$ROOT/phase1_logs/vllm_$1.log" 2>&1 &
  local pid=$!
  for i in $(seq 1 150); do
    curl -s http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && break
    kill -0 $pid 2>/dev/null || { LOG "vllm $1 died"; tail -30 "$ROOT/phase1_logs/vllm_$1.log"; FAIL=1; return 1; }
    sleep 5
  done
  LOG "vLLM $1 up"
  while IFS= read -r task; do
    [ -z "$task" ] && continue
    for rep in 0 1; do
      [ "$rep" -ge "$5" ] && break
      safe=$(echo "$task" | tr ':/' '__')
      t0=$(date +%s)
      if uv run tb infer -c "$4" --dataset ../thinkingbox-data/dataset \
           --agent think --name "$task" \
           --output "$ROOT/artifacts/${6}_${safe}_r${rep}.yaml" \
           > "$ROOT/phase1_logs/infer_${6}_${safe}_r${rep}.log" 2>&1; then
        rc=0
      else rc=1; FAIL=1; fi
      dt=$(( $(date +%s) - t0 ))
      echo "{\"model\":\"$1\",\"task\":\"$task\",\"rep\":$rep,\"rc\":$rc,\"wall_s\":$dt}" \
        | tee -a "$ROOT/artifacts/smoke_results.jsonl"
    done
  done <<< "$TASKS"
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
  sleep 10
}

cd "$PARENT/thinkingbox"

# ---- 6. configs, then the matrix (all bf16 on A100) ----
write_config "$ROOT/cfg_17b.yaml"     "Qwen3-1.7B"       false
write_config "$ROOT/cfg_4b.yaml"      "Qwen3-4B"         false
write_config "$ROOT/cfg_8b.yaml"      "Qwen3-8B"         false
write_config "$ROOT/cfg_4bthink.yaml" "Qwen3-4B-think"   true

serve_and_run "Qwen3-1.7B" "Qwen/Qwen3-1.7B" ""                          "$ROOT/cfg_17b.yaml"     2 p1_17b
LOG "--- 1.7B done ---"
serve_and_run "Qwen3-4B"   "Qwen/Qwen3-4B"   ""                          "$ROOT/cfg_4b.yaml"      2 p1_4b
LOG "--- 4B done ---"
serve_and_run "Qwen3-8B"   "Qwen/Qwen3-8B"   ""                          "$ROOT/cfg_8b.yaml"      2 p1_8b
LOG "--- 8B bf16 done; diagnostic next ---"
serve_and_run "Qwen3-4B-think" "Qwen/Qwen3-4B" "--reasoning-parser qwen3" "$ROOT/cfg_4bthink.yaml" 1 p1_diag

LOG "PHASE 1 SMOKE COMPLETE (FAIL=$FAIL)"
echo "{\"ok\": $( [ $FAIL = 0 ] && echo true || echo false), \"see\": \"artifacts/smoke_results.jsonl\"}" > artifacts/phase1_result.json
