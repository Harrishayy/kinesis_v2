#!/usr/bin/env bash
# Fine-tune SmolVLA (LIBERO checkpoint) on the retargeted human-data episodes.
#   bash scripts/train.sh data/lerobot/ego_bowl outputs/smolvla_ego [extra lerobot_train args]
set -euo pipefail
DATA=${1:-data/lerobot/ego_bowl}
OUT=${2:-outputs/smolvla_ego}
shift 2 || true
REPO_ID=$(python -c "import json;print(json.load(open('$DATA/meta/info.json')).get('repo_id','local/ego_bowl'))" 2>/dev/null || echo local/ego_bowl)
uv run lerobot-train \
  --dataset.repo_id="$REPO_ID" \
  --dataset.root="$DATA" \
  --policy.path=HuggingFaceVLA/smolvla_libero \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir="$OUT" \
  --job_name=smolvla_ego \
  --batch_size=32 \
  --steps=5000 \
  --save_freq=1000 \
  --env_eval_freq=0 \
  --wandb.enable=false \
  "$@"
