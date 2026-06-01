"""
setup_hybrid.py — One-shot setup for the Hybrid Mamba+Attention baseline.

Run this once per Colab session BEFORE train_hybrid.py:

    python setup_hybrid.py

What it does:
    1. Verify mamba-ssm and causal-conv1d are installed (no repo clone needed —
       the hybrid model uses the same official CUDA kernel as the Mamba baseline)
    2. Verify all other imports (torch, sklearn, transformers)
    3. Build the HybridClassifier (~3.78M params) and count parameters
    4. Run a forward + backward pass to confirm GPU correctness

After this script exits 0, train_hybrid.py is ready to run.
"""

import os
import sys
import argparse
import subprocess
import importlib


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def pip_install(package):
    print(f"pip install {package} ...")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", package],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-600:])
        raise RuntimeError(f"pip install {package} failed")
    print(f"  OK: {package}")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — Verify / install dependencies
# ─────────────────────────────────────────────────────────────────────────────
def check_deps():
    print("\n[1/3] Checking dependencies ...")

    # mamba-ssm needs CUDA compilation — warn if missing, don't silently install
    missing_cuda = []
    for pkg, mod in [("mamba-ssm", "mamba_ssm"), ("causal-conv1d", "causal_conv1d")]:
        if importlib.util.find_spec(mod) is None:
            missing_cuda.append(pkg)
        else:
            print(f"  OK: {pkg}")

    if missing_cuda:
        print(f"\n  WARNING: {', '.join(missing_cuda)} not found.")
        print("  These require CUDA compilation. Install with:")
        print("    pip install mamba-ssm causal-conv1d --no-build-isolation")
        print("  This is the same installation used by the Mamba baseline.")
        raise SystemExit(1)

    # Pure-Python deps — safe to auto-install
    for pkg in ["scikit-learn", "transformers"]:
        pip_install(pkg)


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — Verify imports
# ─────────────────────────────────────────────────────────────────────────────
def verify_imports():
    print("\n[2/3] Verifying imports ...")
    import torch                                          # noqa: F401
    import torch.nn as nn                                 # noqa: F401
    from torch.utils.data import Dataset, DataLoader      # noqa: F401
    from sklearn.metrics import roc_auc_score             # noqa: F401
    from transformers import get_cosine_schedule_with_warmup  # noqa: F401
    from mamba_ssm import Mamba                           # noqa: F401
    print("  OK: torch, sklearn, transformers, mamba_ssm")
    return Mamba


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 — Smoke test: build model, count params, forward + backward
# ─────────────────────────────────────────────────────────────────────────────
D_MODEL    = 256
N_MAMBA    = 8
ATTN_POS   = 4      # 0-indexed → layer 5 of 9
ATTN_HEADS = 8
VOCAB_SIZE = 7
NUM_CLS    = 2


def smoke_test(Mamba):
    print("\n[3/3] Smoke test: build HybridClassifier + forward + backward ...")
    import torch
    import torch.nn as nn

    class MambaLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm  = nn.LayerNorm(D_MODEL)
            self.mamba = Mamba(d_model=D_MODEL, d_state=16, d_conv=4, expand=2)
        def forward(self, x):
            return x + self.mamba(self.norm(x))

    class AttentionLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(D_MODEL)
            self.attn = nn.MultiheadAttention(
                embed_dim=D_MODEL, num_heads=ATTN_HEADS,
                dropout=0.1, batch_first=True)
        def forward(self, x):
            n = self.norm(x)
            o, _ = self.attn(n, n, n, need_weights=False)
            return x + o

    class HybridClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(VOCAB_SIZE, D_MODEL)
            layers = []
            for i in range(N_MAMBA + 1):
                layers.append(AttentionLayer() if i == ATTN_POS else MambaLayer())
            self.layers     = nn.ModuleList(layers)
            self.final_norm = nn.LayerNorm(D_MODEL)
            self.head       = nn.Linear(D_MODEL, NUM_CLS)
        def forward(self, ids):
            h = self.embed(ids)
            for layer in self.layers:
                h = layer(h)
            return self.head(self.final_norm(h).mean(dim=1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = HybridClassifier().to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_mamba = sum(p.numel() for l in model.layers
                  if isinstance(l, MambaLayer) for p in l.parameters())
    n_attn  = sum(p.numel() for l in model.layers
                  if isinstance(l, AttentionLayer) for p in l.parameters())

    print(f"  Total params : {n_total/1e6:.3f}M  (target ~3.78M)")
    print(f"    Mamba ({N_MAMBA} layers)  : {n_mamba/1e6:.3f}M")
    print(f"    Attention (1 layer): {n_attn/1e6:.3f}M")
    print(f"    Other              : {(n_total-n_mamba-n_attn)/1e6:.3f}M")
    assert 3.4e6 < n_total < 4.1e6, f"Param count out of range: {n_total/1e6:.2f}M"

    ids    = torch.randint(0, VOCAB_SIZE, (2, 512)).to(device)
    labels = torch.tensor([0, 1]).to(device)
    logits = model(ids)
    loss   = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()

    print(f"  Forward+backward OK: input {tuple(ids.shape)} "
          f"-> logits {tuple(logits.shape)}")
    print(f"  Loss = {loss.item():.4f}")
    print(f"  Device: {device}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Verify setup for the Hybrid Mamba+Attention baseline")
    parser.parse_args()   # no extra args needed — no repo clone required

    print("=" * 55)
    print("  Hybrid Mamba+Attention setup")
    print("  (no external repo clone required)")
    print("=" * 55)

    check_deps()
    Mamba = verify_imports()
    smoke_test(Mamba)

    print("\n" + "=" * 55)
    print("  Setup complete.")
    print("  Run training:")
    print("    python train_hybrid.py \\")
    print("      --save_dir ./checkpoints_hybrid")
    print("=" * 55)


if __name__ == "__main__":
    main()
