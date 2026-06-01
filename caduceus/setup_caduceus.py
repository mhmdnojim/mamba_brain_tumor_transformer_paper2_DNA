"""
setup_caduceus.py — One-shot setup for the Caduceus baseline.

Run this once per Colab session (or local machine) BEFORE train_caduceus.py:

    python setup_caduceus.py
    python setup_caduceus.py --caduceus_dir /my/path/caduceus-lib

What it does:
    1. pip install required dependencies (einops, mamba-ssm, causal-conv1d)
    2. git clone kuleshov-group/caduceus to --caduceus_dir
    3. Verify imports (CaduceusConfig, Caduceus backbone)
    4. Build a tiny test model and count parameters
    5. Run a single forward pass to confirm GPU/CPU correctness

After this script exits 0, train_caduceus.py is ready to run.
"""

import os
import sys
import argparse
import subprocess
import importlib


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def run(cmd, label=""):
    print(f"  >> {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-600:])
        raise RuntimeError(f"FAILED: {label or cmd[0]}")
    return r.stdout


def pip_install(package):
    print(f"pip install {package} ...")
    run([sys.executable, "-m", "pip", "install", "-q", package], f"pip {package}")
    print(f"  OK: {package}")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — Install Python dependencies
# ─────────────────────────────────────────────────────────────────────────────
REQUIRED_PACKAGES = [
    "einops",
    # mamba-ssm and causal-conv1d require CUDA; skip on CPU-only machines
    # pip install mamba-ssm causal-conv1d  ← run manually on GPU machine if absent
]

# Packages that need CUDA compiled extensions (skip pip install on CPU)
CUDA_PACKAGES = ["mamba-ssm", "causal-conv1d"]


def install_deps():
    print("\n[1/4] Installing Python dependencies ...")

    for pkg in REQUIRED_PACKAGES:
        pip_install(pkg)

    # Check if mamba-ssm is already importable
    for pkg in CUDA_PACKAGES:
        mod = pkg.replace("-", "_")
        if importlib.util.find_spec(mod) is None:
            print(f"  WARNING: {pkg} not found. On a GPU machine run:")
            print(f"    pip install {pkg}")
        else:
            print(f"  OK: {pkg} already installed")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — Clone the Caduceus repo
# ─────────────────────────────────────────────────────────────────────────────
REPO_URL = "https://github.com/kuleshov-group/caduceus.git"


def clone_repo(caduceus_dir: str, force: bool = False):
    print(f"\n[2/4] Cloning Caduceus repo to {caduceus_dir} ...")

    if os.path.isdir(caduceus_dir):
        if force:
            import shutil
            shutil.rmtree(caduceus_dir)
            print(f"  Removed existing dir (--force)")
        else:
            print(f"  Already exists — skipping clone (use --force to re-clone)")
            return

    run(["git", "clone", REPO_URL, caduceus_dir], "git clone caduceus")
    print(f"  Cloned to {caduceus_dir}")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 — Verify imports
# ─────────────────────────────────────────────────────────────────────────────
# Complement map: A=0 ↔ T=3, C=1 ↔ G=2, N/specials → self
DNA_COMPLEMENT_MAP = {0: 3, 1: 2, 2: 1, 3: 0, 4: 4, 5: 5, 6: 6}


def verify_imports(caduceus_dir: str):
    print(f"\n[3/4] Verifying imports from {caduceus_dir} ...")

    if caduceus_dir not in sys.path:
        sys.path.insert(0, caduceus_dir)

    # Clear stale cached modules
    for mod in list(sys.modules.keys()):
        if mod.startswith("caduceus"):
            del sys.modules[mod]

    from caduceus.configuration_caduceus import CaduceusConfig
    from caduceus.modeling_caduceus import Caduceus
    print("  OK: CaduceusConfig imported")
    print("  OK: Caduceus imported")
    return CaduceusConfig, Caduceus


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 — Smoke-test: build model, count params, forward pass
# ─────────────────────────────────────────────────────────────────────────────
def smoke_test(CaduceusConfig, Caduceus):
    print("\n[4/4] Smoke test: build model + forward pass ...")

    import torch
    import torch.nn as nn

    # Build a small model matching our training config
    config = CaduceusConfig(
        d_model=256,
        n_layer=4,
        vocab_size=7,
        rcps=True,
        bidirectional=True,
        bidirectional_strategy="add",
        bidirectional_weight_tie=True,
        complement_map=DNA_COMPLEMENT_MAP,
        pad_vocab_size_multiple=1,
    )
    backbone = Caduceus(config)
    head     = nn.Linear(256, 2)

    n_params = (sum(p.numel() for p in backbone.parameters()) +
                sum(p.numel() for p in head.parameters()))
    print(f"  Param count: {n_params / 1e6:.3f}M  (target ~3.5M)")

    # Forward pass on CPU (no GPU needed for smoke test)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone.to(device)
    head.to(device)
    backbone.eval()
    head.eval()

    batch_ids = torch.randint(0, 7, (2, 512)).to(device)

    with torch.no_grad():
        out = backbone(batch_ids)
        if hasattr(out, "last_hidden_state"):
            hidden = out.last_hidden_state        # (2, 512, 256)
        elif isinstance(out, (tuple, list)):
            hidden = out[0]
        else:
            hidden = out

        pooled = hidden.mean(dim=1)               # (2, 256)
        logits = head(pooled)                     # (2, 2)

    print(f"  Forward pass OK: input {tuple(batch_ids.shape)} "
          f"-> logits {tuple(logits.shape)}")
    print(f"  Device: {device}")

    del backbone, head


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Set up the Caduceus repo for train_caduceus.py")
    parser.add_argument("--caduceus_dir", type=str,
                        default="/content/work/caduceus-lib",
                        help="Directory to clone the Caduceus repo into")
    parser.add_argument("--force", action="store_true",
                        help="Re-clone even if --caduceus_dir already exists")
    args = parser.parse_args()

    print("=" * 55)
    print("  Caduceus setup")
    print(f"  Target dir : {args.caduceus_dir}")
    print("=" * 55)

    install_deps()
    clone_repo(args.caduceus_dir, force=args.force)
    CaduceusConfig, Caduceus = verify_imports(args.caduceus_dir)
    smoke_test(CaduceusConfig, Caduceus)

    print("\n" + "=" * 55)
    print("  Setup complete.")
    print(f"  Run training:")
    print(f"    python train_caduceus.py \\")
    print(f"      --caduceus_dir {args.caduceus_dir} \\")
    print(f"      --save_dir ./checkpoints_caduceus")
    print("=" * 55)


if __name__ == "__main__":
    main()
