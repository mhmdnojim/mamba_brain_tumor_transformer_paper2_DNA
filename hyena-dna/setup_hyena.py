"""
setup_hyena.py — One-shot setup for the HyenaDNA baseline.

Run this once per Colab session (or local machine) BEFORE train_hyena.py:

    python setup_hyena.py
    python setup_hyena.py --hyena_dir /my/path/hyena-dna-lib

What it does:
    1. pip install required dependencies (hydra-core, omegaconf, pytorch-lightning)
    2. git clone HazyResearch/hyena-dna to --hyena_dir
    3. Verify import of HyenaOperator from the cloned repo
    4. Build a small test model and count parameters
    5. Run a single forward pass to confirm GPU/CPU correctness

After this script exits 0, train_hyena.py is ready to run.
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
    "hydra-core",
    "omegaconf",
    "pytorch-lightning",
]


def install_deps():
    print("\n[1/4] Installing Python dependencies ...")
    for pkg in REQUIRED_PACKAGES:
        pip_install(pkg)


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — Clone the HyenaDNA repo
# ─────────────────────────────────────────────────────────────────────────────
REPO_URL = "https://github.com/HazyResearch/hyena-dna.git"


def clone_repo(hyena_dir: str, force: bool = False):
    print(f"\n[2/4] Cloning HyenaDNA repo to {hyena_dir} ...")

    if os.path.isdir(hyena_dir):
        if force:
            import shutil
            shutil.rmtree(hyena_dir)
            print("  Removed existing dir (--force)")
        else:
            print("  Already exists — skipping clone (use --force to re-clone)")
            return

    run(["git", "clone", REPO_URL, hyena_dir], "git clone hyena-dna")
    print(f"  Cloned to {hyena_dir}")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 — Verify HyenaOperator import
# ─────────────────────────────────────────────────────────────────────────────
def verify_imports(hyena_dir: str):
    print(f"\n[3/4] Verifying HyenaOperator import from {hyena_dir} ...")

    if hyena_dir not in sys.path:
        sys.path.insert(0, hyena_dir)

    # Clear stale cached modules
    for mod in list(sys.modules.keys()):
        if mod.startswith("src.") or mod == "src":
            del sys.modules[mod]

    from src.models.sequence.hyena import HyenaOperator
    print("  OK: HyenaOperator imported")
    return HyenaOperator


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 — Smoke test: build model, count params, forward pass
# ─────────────────────────────────────────────────────────────────────────────
def smoke_test(HyenaOperator):
    print("\n[4/4] Smoke test: build model + forward pass ...")

    import torch
    import torch.nn as nn

    D_MODEL      = 256
    N_LAYER      = 4
    L_MAX        = 512
    VOCAB_SIZE   = 7
    FILTER_ORDER = 128
    ORDER        = 2
    MLP_RATIO    = 4
    DROPOUT      = 0.1

    # Replicate HyenaClassifier structure from train_hyena.py
    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = nn.LayerNorm(D_MODEL)
            self.hyena = HyenaOperator(
                d_model=D_MODEL, l_max=L_MAX,
                order=ORDER, filter_order=FILTER_ORDER, dropout=DROPOUT)
            self.norm2 = nn.LayerNorm(D_MODEL)
            self.mlp   = nn.Sequential(
                nn.Linear(D_MODEL, D_MODEL * MLP_RATIO), nn.GELU(),
                nn.Linear(D_MODEL * MLP_RATIO, D_MODEL), nn.Dropout(DROPOUT))

        def forward(self, x):
            x = x + self.hyena(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x

    embed  = nn.Embedding(VOCAB_SIZE, D_MODEL)
    blocks = nn.ModuleList([_Block() for _ in range(N_LAYER)])
    norm   = nn.LayerNorm(D_MODEL)
    head   = nn.Linear(D_MODEL, 2)

    n_params = sum(p.numel() for p in [*embed.parameters(),
                                        *[p for b in blocks for p in b.parameters()],
                                        *norm.parameters(), *head.parameters()])
    print(f"  Param count: {n_params / 1e6:.3f}M  (target ~3.45M)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for m in [embed, *blocks, norm, head]:
        m.to(device)
    for m in [embed, *blocks, norm, head]:
        m.eval()

    batch_ids = torch.randint(0, VOCAB_SIZE, (2, L_MAX)).to(device)

    with torch.no_grad():
        x = embed(batch_ids)
        for blk in blocks:
            x = blk(x)
        x      = norm(x)
        pooled = x.mean(dim=1)
        logits = head(pooled)

    print(f"  Forward pass OK: input {tuple(batch_ids.shape)} "
          f"-> logits {tuple(logits.shape)}")
    print(f"  Device: {device}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Set up the HyenaDNA repo for train_hyena.py")
    parser.add_argument("--hyena_dir", type=str,
                        default="/content/work/hyena-dna-lib",
                        help="Directory to clone the HazyResearch/hyena-dna repo into")
    parser.add_argument("--force", action="store_true",
                        help="Re-clone even if --hyena_dir already exists")
    args = parser.parse_args()

    print("=" * 55)
    print("  HyenaDNA setup")
    print(f"  Target dir : {args.hyena_dir}")
    print("=" * 55)

    install_deps()
    clone_repo(args.hyena_dir, force=args.force)
    HyenaOperator = verify_imports(args.hyena_dir)
    smoke_test(HyenaOperator)

    print("\n" + "=" * 55)
    print("  Setup complete.")
    print(f"  Run training:")
    print(f"    python train_hyena.py \\")
    print(f"      --hyena_dir {args.hyena_dir} \\")
    print(f"      --save_dir ./checkpoints_hyena")
    print("=" * 55)


if __name__ == "__main__":
    main()
