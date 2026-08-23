#!/usr/bin/env bash
# Phase 0 wiring check — proves framework + vLLM Qwen3 + tools + checks work end to end.
# Run from thinkingbox/ after bootstrap_host.sh and after vLLM is serving.
set -euo pipefail

BASE_DIR="${1:-$HOME/llm-recovery-mode}"
ART="$BASE_DIR/artifacts"
cd "$BASE_DIR/thinkingbox"
source .venv/bin/activate

MODEL="${MODEL:-Qwen3-4B}"
PORT="${PORT:-8000}"

echo "== [1/3] Starting background services (Typesense + Session Proxy) =="
export THINKINGBOX_DATA="../thinkingbox-data"
export TB_MCP_START_SERVERS_FILE="../thinkingbox-data/servers/servers.yaml"
./scripts/background_tasks.sh   # waits for "All processes are running"

echo "== [2/3] Wiring check: single benchmark task via local vLLM =="
uv run tb infer -c "$BASE_DIR/configs/agent_qwen3_${MODEL}.yaml" \
    --dataset ../thinkingbox-data/dataset --agent think \
    --name banking.py:test_get_balance_savings \
    --output "$ART/wiring_check_output.yaml"

echo "== [3/3] Verifying conversation + assertions =="
uv run tb pp "$ART/wiring_check_output.yaml" | tee "$ART/wiring_check_pretty.txt"

echo ""
echo "WIRING CHECK COMPLETE — inspect $ART/wiring_check_output.yaml"
echo "GO criteria: valid multi-turn conversation, tool calls executed, assertions ran."
