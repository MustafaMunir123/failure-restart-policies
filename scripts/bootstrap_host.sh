#!/usr/bin/env bash
# Phase 0 bootstrap — run ON the A100 Linux host.
# Clones + pins Thinkingbox repos, sets up env, installs Typesense.
set -euo pipefail

BASE_DIR="${1:-$HOME/llm-recovery-mode}"
mkdir -p "$BASE_DIR" && cd "$BASE_DIR"
ART="$BASE_DIR/artifacts"; mkdir -p "$ART"

echo "== [1/6] Cloning repos =="
[ -d thinkingbox ] || git clone https://github.com/microsoft/thinkingbox.git
[ -d thinkingbox-data ] || git clone https://github.com/microsoft/thinkingbox-data.git

echo "== [2/6] Pinning thinkingbox-data to benchmark tag =="
cd thinkingbox-data
git checkout thinkingbox-bench-v1.0
cd ..

echo "== [3/6] Recording pinned commits =="
{
  echo "thinkingbox: $(git -C thinkingbox rev-parse HEAD) ($(git -C thinkingbox rev-parse --short HEAD))"
  echo "thinkingbox-data: $(git -C thinkingbox-data rev-parse HEAD) (tag thinkingbox-bench-v1.0)"
  echo "pinned at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$ART/pinned_commits.txt"
cat "$ART/pinned_commits.txt"

echo "== [4/6] Python env (needs uv; install: curl -LsSf https://astral.sh/uv/install.sh | sh) =="
command -v uv >/dev/null || { echo "uv not found — install it and rerun"; exit 1; }
cd thinkingbox
uv venv --python 3.12
uv sync --group dev
source .venv/bin/activate

echo "== [5/6] Installing benchmark MCP server package =="
uv pip install --config-settings editable-mode=compat \
  -e ../thinkingbox-data/servers/tb_business_ops_servers_202606

echo "== [6/6] Installing Typesense =="
./scripts/install_typesense.sh
typesense-server --version

echo ""
echo "Bootstrap complete. Pinned commits recorded in $ART/pinned_commits.txt"
echo "NEXT (manual): start vLLM with Qwen3, then services via ./scripts/background_tasks.sh,"
echo "then run the wiring check: scripts/wiring_check.sh"
