#!/usr/bin/env bash
# First-time setup on the AWS g6e.xlarge (docs/prepare_cpt_training_pipeline.md, section 10.5).
# AMI: "Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.x (Ubuntu)". Run from the repo root:
#     bash scripts/train/setup_machine.sh
# Installs into the AMI's PyTorch environment, so its CUDA build of torch is kept.
set -euo pipefail

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
if [ -f /opt/pytorch/bin/activate ]; then
    # shellcheck disable=SC1091
    source /opt/pytorch/bin/activate
    echo "using the AMI environment /opt/pytorch (add 'source /opt/pytorch/bin/activate' to ~/.bashrc)"
fi
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA torch'; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

pip install -r scripts/train/requirements-train.txt
pip install causal-conv1d --no-build-isolation      # compiles for a few minutes

python - <<'EOF'
import importlib, torch, transformers
print("torch", torch.__version__, "| transformers", transformers.__version__)
for mod, fn in [("fla.ops.gated_delta_rule", "chunk_gated_delta_rule"), ("causal_conv1d", "causal_conv1d_fn"),
                ("torchao.optim", "AdamW8bit")]:
    try:
        ok = getattr(importlib.import_module(mod), fn, None) is not None
    except Exception as e:
        ok = f"FAILED ({e})"
    print(f"{mod}.{fn}: {ok}")
EOF

mkdir -p outputs/logs
df -h / | tail -1
cat <<'EOF'

Next:
  hf auth login          # token with write access (epoch copies go to a private repo)
  wandb login
  tmux new -s train      # every run inside tmux
  python scripts/train/gpu_check.py    # kernels + optimizer: RESULT must be "all needed checks passed"
  then README step 2 (smoke test)
EOF
