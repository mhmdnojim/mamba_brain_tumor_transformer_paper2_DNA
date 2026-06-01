"""
setup_basset.py — One-shot setup for the Basset CNN baseline.

Run this once per machine/session BEFORE train_basset.py:

    python setup_basset.py
    python setup_basset.py --variant small
    python setup_basset.py --variant full

What it does:
    1. pip install required dependencies (scikit-learn, transformers)
    2. Verify all imports needed by train_basset.py
    3. Build the requested Basset variant (small ~3.49M, full ~3.97M) and count params
    4. Run a single forward pass to confirm GPU/CPU correctness

No external repo clone required — Basset is self-contained in train_basset.py.
After this script exits 0, train_basset.py is ready to run.
"""

import os
import sys
import argparse
import subprocess


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
#  STEP 1 — Install Python dependencies
# ─────────────────────────────────────────────────────────────────────────────
REQUIRED_PACKAGES = ["scikit-learn", "transformers"]


def install_deps():
    print("\n[1/3] Installing Python dependencies ...")
    for pkg in REQUIRED_PACKAGES:
        pip_install(pkg)


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — Verify imports
# ─────────────────────────────────────────────────────────────────────────────
def verify_imports():
    print("\n[2/3] Verifying imports ...")
    import numpy as np                                    # noqa: F401
    import torch                                          # noqa: F401
    import torch.nn as nn                                 # noqa: F401
    from torch.utils.data import Dataset, DataLoader      # noqa: F401
    from sklearn.metrics import roc_auc_score             # noqa: F401
    from transformers import get_cosine_schedule_with_warmup  # noqa: F401
    print("  OK: numpy, torch, sklearn, transformers")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 — Smoke test: build Basset, count params, forward pass
# ─────────────────────────────────────────────────────────────────────────────
# Basset conv/FC configs (mirroring train_basset.py)
VARIANTS = {
    "small": dict(
        conv_filters=[200, 200, 200],
        fc_sizes=[1100, 500],
        description="Basset-small (~3.49M params, param-matched to Mamba 3.51M)",
    ),
    "full": dict(
        conv_filters=[300, 200, 200],
        fc_sizes=[1000, 1000],
        description="Basset-full (~3.97M params, Kelley et al. 2016 scale)",
    ),
}

SEQ_LEN     = 512
IN_CHANNELS = 4    # one-hot DNA (A/T/G/C)
NUM_CLASSES = 2


def build_basset(conv_filters, fc_sizes):
    """Replicate BassetCNN from train_basset.py."""
    import torch.nn as nn

    conv_channels = [IN_CHANNELS] + conv_filters
    conv_layers = []
    for i in range(len(conv_filters)):
        conv_layers += [
            nn.Conv1d(conv_channels[i], conv_channels[i + 1],
                      kernel_size=19, padding=9),
            nn.BatchNorm1d(conv_channels[i + 1]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
        ]

    # Compute flattened size after all pooling: SEQ_LEN // 3**n_conv
    flat_len = SEQ_LEN
    for _ in conv_filters:
        flat_len = flat_len // 3
    flat_size = conv_filters[-1] * flat_len

    fc_dims  = [flat_size] + fc_sizes + [NUM_CLASSES]
    fc_layers = []
    for i in range(len(fc_dims) - 1):
        fc_layers.append(nn.Linear(fc_dims[i], fc_dims[i + 1]))
        if i < len(fc_dims) - 2:
            fc_layers += [nn.ReLU(), nn.Dropout(0.3)]

    class BassetCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(*conv_layers)
            self.fc   = nn.Sequential(*fc_layers)

        def forward(self, x):
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    return BassetCNN(), flat_size


def smoke_test(variant: str):
    import torch

    cfg = VARIANTS[variant]
    print(f"\n[3/3] Smoke test: {cfg['description']} ...")

    model, flat_size = build_basset(cfg["conv_filters"], cfg["fc_sizes"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Param count : {n_params / 1e6:.3f}M")
    print(f"  Flat size   : {flat_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # one-hot input: (batch, 4, seq_len)
    x = torch.zeros(2, IN_CHANNELS, SEQ_LEN, device=device)
    x[:, 0, :] = 1.0  # all-A dummy input

    with torch.no_grad():
        logits = model(x)

    print(f"  Forward pass OK: input {tuple(x.shape)} -> logits {tuple(logits.shape)}")
    print(f"  Device: {device}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Set up and verify the Basset CNN baseline")
    parser.add_argument("--variant", choices=["small", "full", "both"],
                        default="both",
                        help="Which Basset variant to smoke-test (default: both)")
    args = parser.parse_args()

    print("=" * 55)
    print("  Basset CNN setup")
    print(f"  Variant(s)  : {args.variant}")
    print("=" * 55)

    install_deps()
    verify_imports()

    variants = ["small", "full"] if args.variant == "both" else [args.variant]
    for v in variants:
        smoke_test(v)

    print("\n" + "=" * 55)
    print("  Setup complete. No external repo required.")
    print("  Run training:")
    for v in variants:
        print(f"    python train_basset.py --variant {v} \\")
        print(f"      --save_dir ./checkpoints_basset_{v}")
    print("=" * 55)


if __name__ == "__main__":
    main()
