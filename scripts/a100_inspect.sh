#!/usr/bin/env bash
# Inspect persistent NFS state: pull Phase-1 smoke results + 8B failure cause.
set -uo pipefail
ROOT="/lambda/nfs/ic-fs-2/repos/systrem-recovery-modes"
cd "$ROOT"
echo "===== artifacts/ ====="
ls -la artifacts/ 2>/dev/null
echo "===== smoke_results.jsonl ====="
cat artifacts/smoke_results.jsonl 2>/dev/null
echo "===== phase1 main.log ====="
cat phase1_logs/main.log 2>/dev/null | head -50
echo "===== 8B vllm log (errors) ====="
grep -inE "oom|out of memory|error|valueerror|runtimeerror|no such|download|cuda" \
  phase1_logs/vllm_Qwen3-8B.log 2>/dev/null | grep -vi apiserver | head -25
echo "===== last 30 lines of 8B log ====="
tail -30 phase1_logs/vllm_Qwen3-8B.log 2>/dev/null
tar czf /lambda/nfs/ic-fs-2/p1_logs.tgz phase1_logs artifacts 2>/dev/null
echo "tarball: /lambda/nfs/ic-fs-2/p1_logs.tgz"
