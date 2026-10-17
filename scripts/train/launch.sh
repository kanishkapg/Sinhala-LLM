#!/usr/bin/env bash
# Start a run inside tmux, keep a log file, and stop the machine when the run ends or crashes
# (docs/prepare_cpt_training_pipeline.md, section 10.6). Run from the repo root:
#     tmux new -s train
#     bash scripts/train/launch.sh --config scripts/train/config/cpt_easy.json --resume
# All arguments go to train_cpt.py. NO_SHUTDOWN=1 keeps the machine running (smoke test, debugging).
set -uo pipefail

mkdir -p outputs/logs
log="outputs/logs/train-$(date +%Y%m%d-%H%M%S).log"
echo "log: $log"
python scripts/train/train_cpt.py "$@" 2>&1 | tee "$log"
status=${PIPESTATUS[0]}
echo "train_cpt.py exited with status $status (log: $log)"

if [ -z "${NO_SHUTDOWN:-}" ]; then
    # Shutdown behavior = Stop: the EBS disk and the checkpoints stay. One minute to cancel: sudo shutdown -c
    sudo shutdown -h +1
fi
exit "$status"
