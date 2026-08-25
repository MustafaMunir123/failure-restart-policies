#!/usr/bin/env bash
# Poll for A100 capacity in us-east-1; launch phase-1 when a slot opens.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
export LAMBDA_API_KEY="$LAMBDA_CLOUD_API_KEY"
ssh-add ~/.ssh/id_ed25519_personal ~/.ssh/lambda_id_ed25519 2>/dev/null

POLL_SECS="${1:-600}"
while true; do
  echo "[$(date +%H:%M:%S)] checking capacity..."
  if python3 infra/lambda/lambda/lambda_cli.py find-capacity --region us-east-1 --prefer gpu_1x_a100_sxm4 gpu_1x_a100 2>&1 | grep -qi a100; then
    echo "[$(date +%H:%M:%S)] CAPACITY FOUND — launching"
    bash infra/lambda/bin/run.sh \
      --repo git@github.com:MustafaMunir123/systrem-recovery-modes.git \
      --branch main \
      --req-file requirements.txt \
      --cmd "bash scripts/a100_phase1.sh" \
      --yes && { echo "[$(date +%H:%M:%S)] run finished"; break; }
  fi
  sleep "$POLL_SECS"
done
