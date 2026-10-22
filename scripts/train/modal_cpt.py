"""Run the scripts of scripts/train on a Modal L40S (48 GB) instead of the AWS g6e (README, "On Modal").

One-time setup, from the repo root (PowerShell: run `$env:PYTHONUTF8 = "1"` first, for the Sinhala in the logs):

    pip install modal
    modal setup
    modal secret create sinhala-cpt HF_TOKEN=hf_... WANDB_API_KEY=...

Each command starts a fresh L40S container with the image below. The container runs a script of scripts/train from
/repo, the same way as on the g6e. `outputs/` is the Modal Volume `sinhala-cpt-outputs`, so checkpoints, epoch copies and
probe records stay after the container exits. The Hugging Face cache is on the Volume `hf-cache`, so the 4B model
downloads once.

    modal run scripts/train/modal_cpt.py --step check                 # gpu_check.py: kernels + 8-bit AdamW
    modal run --detach scripts/train/modal_cpt.py --step smoke        # README step 2
    modal run --detach scripts/train/modal_cpt.py --step stop25       # README step 3, part 1
    modal run --detach scripts/train/modal_cpt.py --step resume       # part 2 = the full run
    modal run --detach scripts/train/modal_cpt.py --cmd "probe_eval.py --model outputs/cpt-easy-v1/epochs/e1 --out outputs/probes/e1"

--cmd takes any script of scripts/train with its arguments. --detach keeps the run going after the terminal closes.
Download results with `modal volume get sinhala-cpt-outputs /outputs/<path> outputs/`.
"""
import os
import shlex
import subprocess
import sys
from pathlib import Path

import modal

GPU = "L40S"
CFG = "--config scripts/train/config/cpt_easy.json"
STEPS = {
    "check": "gpu_check.py",
    "smoke": f"train_cpt.py {CFG} --smoke",
    "stop25": f"train_cpt.py {CFG} --stop-at-step 25",
    "resume": f"train_cpt.py {CFG} --resume",
}
DATA = "data/processed/cpt/easy"
# The data comes from GitHub at this commit, fetched by Modal's builder: nothing large uploads from your machine
# (a flaky upload of the parquet files stopped the first run). After changing the data: push, then update DATA_REV.
GIT_URL = "https://github.com/kanishkapg/Sinhala-LLM.git"
DATA_REV = "b759c1046bea5498225f05eaacee6d8922d726b0"
DATA_PATHS = [f"{DATA}/packed", f"{DATA}/manifest.json", f"{DATA}/mix/val.jsonl", "data/probes/easy"]

# Local paths only matter on your machine. In the container the image already exists and the mounts below are replayed.
REPO = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/repo")

image = (
    # devel image: nvcc compiles causal-conv1d when no prebuilt wheel matches; CUDA 12.8 = the torch build below
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .pip_install("torch==2.9.1", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("torchao==0.14.1", "ninja", "packaging", "wheel", "setuptools")    # torchao release for torch 2.9
    .pip_install_from_requirements(str(REPO / "scripts/train/requirements-train.txt"))
    .run_commands("pip install causal-conv1d --no-build-isolation")
    .apt_install("git")
    # only the files the scripts read (~30 MB); .git is removed so repro.json records your code commit, not DATA_REV
    .run_commands(f"git clone --filter=blob:none --no-checkout {GIT_URL} /repo",
                  f"cd /repo && git sparse-checkout set --no-cone {' '.join(DATA_PATHS)}"
                  f" && git checkout {DATA_REV} && rm -rf /repo/.git")
    .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1"})
    # the code (~0.6 MB) is mounted from your working tree at start, so code edits need no image rebuild
    .add_local_dir(REPO / "scripts", "/repo/scripts", ignore=["**/__pycache__/**", "**/*.ipynb"])
)

app = modal.App("sinhala-cpt", image=image)
outputs_vol = modal.Volume.from_name("sinhala-cpt-outputs", create_if_missing=True)
cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)


@app.function(gpu=GPU, cpu=4.0, memory=32768, timeout=24 * 3600,
              volumes={"/vol": outputs_vol, "/cache": cache_vol},
              secrets=[modal.Secret.from_name("sinhala-cpt")])
def run(cmd: str, git_commit: str = "", git_dirty: bool = False) -> None:
    out = Path("/vol/outputs")
    out.mkdir(parents=True, exist_ok=True)
    link = Path("/repo/outputs")              # repo-relative paths (outputs/cpt-easy-v1, ...) land on the Volume
    if not link.exists():
        link.symlink_to(out, target_is_directory=True)
    argv = shlex.split(cmd)
    env = os.environ | {"CPT_GIT_COMMIT": git_commit, "CPT_GIT_DIRTY": "1" if git_dirty else "0"}
    print(f"$ python scripts/train/{cmd}", flush=True)
    try:
        status = subprocess.run([sys.executable, f"scripts/train/{argv[0]}", *argv[1:]], cwd="/repo", env=env).returncode
    finally:
        outputs_vol.commit()                  # Volumes also commit in the background; this makes the end state certain
        cache_vol.commit()
    if status:
        raise RuntimeError(f"{argv[0]} exited with status {status}")


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


@app.local_entrypoint()
def main(step: str = "", cmd: str = ""):
    if bool(step) == bool(cmd) or (step and step not in STEPS):
        raise SystemExit(f'give --step {{{",".join(STEPS)}}} or --cmd "<script in scripts/train> <arguments>"')
    cmd = STEPS.get(step, cmd)
    if _git("diff", "--name-only", DATA_REV, "--", *DATA_PATHS):
        raise SystemExit(f"your data differs from DATA_REV {DATA_REV[:7]}, which the image holds: "
                         "commit and push it, then set DATA_REV in modal_cpt.py to the new commit")
    dirty = bool(_git("status", "--porcelain"))
    if dirty:
        print("note: uncommitted changes. The run uses your working tree (recorded as git_dirty in repro.json)")
    run.remote(cmd, _git("rev-parse", "HEAD"), dirty)
