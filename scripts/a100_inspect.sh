#!/usr/bin/env bash
# Inspect #2: get the ACTUAL exception at end of vLLM engine-init tracebacks.
ROOT="/lambda/nfs/ic-fs-2/repos/systrem-recovery-modes"
cd "$ROOT"
for m in Qwen3-1.7B Qwen3-4B Qwen3-8B; do
  echo "########## $m: first non-stacktrace error lines ##########"
  grep -E "ERROR.*Error|ERROR.*error|raise|Exception" "phase1_logs/vllm_$m.log" 2>/dev/null | tail -6
  echo "---------- $m: last 12 lines ----------"
  tail -12 "phase1_logs/vllm_$m.log" 2>/dev/null | grep -v "^\s*File\|^APIServer"
done
echo "########## phase1_logs dir ##########"
ls phase1_logs/ 2>/dev/null
echo "########## any infer logs? ##########"
ls phase1_logs/infer_* 2>/dev/null || echo "none - no task ever ran"
echo "########## disk space ##########"
df -h /lambda/nfs/ic-fs-2 | tail -1
