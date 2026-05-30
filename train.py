"""
train.py — Mamba Cancer Gene Detection (Sequence Classification)

Task:
    Input  : DNA sequence  (e.g. "ATCGATCGNNGATC...")
    Output : Cancer (1) or Normal (0)

Data format expected (CSV file):
    sequence,label
    ATCGATCG...,1
    GCTAGCTA...,0

Uses these existing project files:
    mamba_ssm/models/config_mamba.py        → MambaConfig
    mamba_ssm/models/mixer_seq_simple.py    → MixerModel  (backbone, no LM head)
    mamba_ssm/modules/mamba_simple.py       → Mamba SSM block
    mamba_ssm/modules/block.py              → Block (norm + residual)
    mamba_ssm/ops/selective_scan_interface  → CUDA selective scan kernel

Run:
    python train.py --data cancer_genes.csv
    python train.py --data cancer_genes.csv --d_model 256 --n_layer 8
    python train.py --data cancer_genes.csv --resume ./checkpoints/step_500.pt
"""

import os
import math
import time
import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

from transformers import get_cosine_schedule_with_warmup

try:
    from sklearn.metrics import roc_auc_score as _roc_auc
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

# ── Project files (existing in this repo) ────────────────────────────────────
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MixerModel   # backbone only, no LM head
# MixerModel internally loads:
#   modules/mamba_simple.py   → Mamba SSM block
#   modules/block.py          → norm + residual wrapper
#   modules/mlp.py            → GatedMLP (if d_intermediate > 0)
#   ops/selective_scan_interface.py → CUDA kernel
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
#  DNA TOKENIZER  (character-level: A T G C N → integer IDs)
#  No external library needed — DNA has a tiny vocabulary
# ─────────────────────────────────────────────────────────────────────────────
class DNATokenizer:
    """
    Vocabulary:
        0  = [PAD]
        1  = [UNK]
        2  = A
        3  = T
        4  = G
        5  = C
        6  = N   (unknown nucleotide)
    """
    VOCAB = {"[PAD]": 0, "[UNK]": 1, "A": 2, "T": 3, "G": 4, "C": 5, "N": 6}
    PAD_ID = 0
    vocab_size = len(VOCAB)

    def encode(self, sequence: str, max_len: int) -> list:
        ids = [self.VOCAB.get(c.upper(), self.VOCAB["[UNK]"]) for c in sequence]
        ids = ids[:max_len]                                   # truncate
        ids += [self.PAD_ID] * (max_len - len(ids))          # pad
        return ids

    def batch_encode(self, sequences: list, max_len: int) -> torch.Tensor:
        return torch.tensor(
            [self.encode(s, max_len) for s in sequences], dtype=torch.long
        )


# ─────────────────────────────────────────────────────────────────────────────
#  DATASET
#  Expects a CSV with columns: sequence, label
#  label: 1 = cancer gene region, 0 = normal
# ─────────────────────────────────────────────────────────────────────────────
class CancerGeneDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer: DNATokenizer, seq_len: int):
        self.tokenizer = tokenizer
        self.seq_len   = seq_len
        self.sequences = []
        self.labels    = []

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.sequences.append(row["sequence"].strip().upper())
                self.labels.append(int(row["label"]))

        print(f"  Loaded {len(self.sequences)} samples from {csv_path}")
        cancer_count = sum(self.labels)
        print(f"  Cancer: {cancer_count}  Normal: {len(self.labels)-cancer_count}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        input_ids = torch.tensor(
            self.tokenizer.encode(self.sequences[idx], self.seq_len),
            dtype=torch.long,
        )
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return input_ids, label


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL  — Mamba backbone  +  classification head
#
#  MixerModel  (from mixer_seq_simple.py):
#      input_ids (B, L) → hidden_states (B, L, d_model)
#
#  ClassificationHead:
#      mean-pool over L → (B, d_model) → Linear → (B, num_classes)
# ─────────────────────────────────────────────────────────────────────────────
class MambaCancerClassifier(nn.Module):
    def __init__(self, config: MambaConfig, num_classes: int = 2):
        super().__init__()

        # ── Mamba backbone (all existing project files) ───────────────────────
        self.backbone = MixerModel(
            d_model          = config.d_model,
            n_layer          = config.n_layer,
            d_intermediate   = config.d_intermediate,
            vocab_size       = config.vocab_size,
            ssm_cfg          = config.ssm_cfg,
            rms_norm         = config.rms_norm,
            residual_in_fp32 = config.residual_in_fp32,
            fused_add_norm   = config.fused_add_norm,
        )

        # ── Classification head (new — replaces the LM head) ─────────────────
        self.norm       = nn.LayerNorm(config.d_model)
        self.classifier = nn.Linear(config.d_model, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids : (B, L)
        hidden = self.backbone(input_ids)          # (B, L, d_model)
        pooled = hidden.mean(dim=1)                # (B, d_model) — avg over sequence
        pooled = self.norm(pooled)
        logits = self.classifier(pooled)           # (B, num_classes)
        return logits


# ─────────────────────────────────────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(preds: torch.Tensor, labels: torch.Tensor,
                    probs: torch.Tensor | None = None):
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()

    acc       = (tp + tn) / (tp + fp + fn + tn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    result    = {"acc": acc, "precision": precision, "recall": recall, "f1": f1}
    if probs is not None and _HAS_SKLEARN:
        try:
            result["auc"] = _roc_auc(labels.numpy(), probs.numpy())
        except Exception:
            pass
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────
def build_optimizer(model, args):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "bias" in name or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [
            {"params": decay,    "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr, betas=(0.9, 0.95),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  CHECKPOINT
# ─────────────────────────────────────────────────────────────────────────────
def save_checkpoint(path, step, model, optimizer, scheduler, config):
    torch.save({
        "step": step, "config": config,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, path)
    print(f"  [saved] {path}")


def load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["step"]


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Mamba Cancer Gene Classifier")

    # Data
    p.add_argument("--data",       type=str, default="dataset/cancer_genes_matched.csv",
                   help="CSV file with columns: sequence, label (used when no splits_dir)")
    p.add_argument("--splits_dir", type=str, default=None,
                   help="Directory with train.csv/val.csv/test.csv (chromosome-level split). "
                        "Auto-detected at dataset/splits/ if present.")
    p.add_argument("--val_split",  type=float, default=0.1,
                   help="Fraction of data used for validation (random split fallback only)")
    p.add_argument("--seq_len",    type=int, default=512,
                   help="Max DNA sequence length in characters")
    p.add_argument("--num_classes",type=int, default=2,
                   help="2 = binary (cancer/normal). Increase for multi-class.")

    # Model
    p.add_argument("--d_model",        type=int, default=256)
    p.add_argument("--n_layer",        type=int, default=8)
    p.add_argument("--d_intermediate", type=int, default=0)
    p.add_argument("--ssm_layer",      type=str, default="Mamba1",
                   choices=["Mamba1", "Mamba2"])

    # Training
    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--grad_accum",   type=int,   default=2)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int,   default=50)
    p.add_argument("--max_steps",    type=int,   default=1000)
    p.add_argument("--clip_grad",    type=float, default=1.0)

    # Logging / saving
    p.add_argument("--log_every",  type=int, default=20)
    p.add_argument("--save_every", type=int, default=200)
    p.add_argument("--save_dir",   type=str, default="./checkpoints")
    p.add_argument("--resume",     type=str, default=None)

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def ensure_data(data_path: str, seq_len: int):
    """
    If the CSV data file does not exist, automatically run dataset.py
    to download TCGA data and create it.
    """
    if os.path.exists(data_path):
        print(f"[data] Found existing dataset: {data_path}")
        return

    print(f"[data] '{data_path}' not found — running dataset.py to build it...")
    print("[data] This will download from TCGA (GDC API) + Ensembl.")
    print("[data] First run uses demo mode for speed. Use --cancer BRCA for real data.\n")

    # Import dataset.py from the same directory
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(
        "dataset",
        os.path.join(os.path.dirname(__file__), "dataset.py")
    )
    ds_module = importlib.util.load_from_spec(spec)
    spec.loader.exec_module(ds_module)

    # Generate demo data by default (fast, no internet required).
    # To use real TCGA data, run:  python dataset.py --cancer BRCA --out cancer_genes.csv
    ds_module.generate_demo_dataset(out_path=data_path, n_samples=400, seq_len=seq_len)
    print(f"[data] Dataset ready: {data_path}\n")


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  Mamba Cancer Gene Detector")
    print(f"  Device  : {device}")
    if device.type == "cuda":
        print(f"  GPU     : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM    : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"  d_model : {args.d_model}   n_layer : {args.n_layer}")
    print(f"  seq_len : {args.seq_len}   classes : {args.num_classes}")
    print("=" * 60)

    # ── DNA tokenizer (character-level A/T/G/C) ───────────────────────────────
    tokenizer = DNATokenizer()
    print(f"DNA vocabulary size: {tokenizer.vocab_size}")

    # ── Dataset — chromosome split preferred, random split as fallback ────────
    _splits = Path(args.splits_dir) if args.splits_dir else Path("dataset/splits")
    te_ds   = None

    if (_splits / "train.csv").exists():
        print(f"\nChromosome-level split from {_splits}/")
        tr_ds  = CancerGeneDataset(str(_splits / "train.csv"), tokenizer, args.seq_len)
        val_ds = CancerGeneDataset(str(_splits / "val.csv"),   tokenizer, args.seq_len)
        te_ds  = CancerGeneDataset(str(_splits / "test.csv"),  tokenizer, args.seq_len)
        print(f"  Test  : {len(te_ds)} samples (held out until final eval)")
    else:
        print(f"\nWARNING: {_splits}/ not found — falling back to random split.")
        print("  Run Cell 5f in the notebook to build chromosome-level splits.")
        ensure_data(args.data, args.seq_len)
        full_dataset = CancerGeneDataset(args.data, tokenizer, args.seq_len)
        val_size   = max(1, int(len(full_dataset) * args.val_split))
        train_size = len(full_dataset) - val_size
        tr_ds, val_ds = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(tr_ds,  batch_size=args.batch_size,
                              shuffle=True,  drop_last=True,  num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, drop_last=False, num_workers=2)

    print(f"  Train : {len(tr_ds):,} samples  ({len(train_loader)} steps/epoch)")
    print(f"  Val   : {len(val_ds):,} samples")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\nBuilding model...")
    config = MambaConfig(
        d_model          = args.d_model,
        n_layer          = args.n_layer,
        d_intermediate   = args.d_intermediate,
        vocab_size       = tokenizer.vocab_size,   # only 7 tokens (A/T/G/C/N/PAD/UNK)
        ssm_cfg          = {"layer": args.ssm_layer},
        rms_norm         = True,
        residual_in_fp32 = True,
        fused_add_norm   = True,
    )
    model = MambaCancerClassifier(config, num_classes=args.num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params/1e6:.2f}M")
    print(f"  Backbone  : MixerModel  (mixer_seq_simple.py)")
    print(f"  SSM block : {args.ssm_layer}  (mamba_simple.py)")
    print(f"  Head      : Linear({args.d_model} → {args.num_classes})")

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    optimizer = build_optimizer(model, args)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = args.warmup_steps,
        num_training_steps = args.max_steps,
    )
    scaler   = GradScaler()
    loss_fn  = nn.CrossEntropyLoss()

    # ── Resume ────────────────────────────────────────────────────────────────
    global_step = 0
    if args.resume:
        global_step = load_checkpoint(
            args.resume, model, optimizer, scheduler, device
        )
        print(f"  Resumed from step {global_step}")

    os.makedirs(args.save_dir, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    print("\nStarting training...")
    model.train()
    optimizer.zero_grad()

    running_loss = 0.0
    all_preds, all_labels = [], []
    t0 = time.time()

    for epoch in range(99999):
        for input_ids, labels in train_loader:
            if global_step >= args.max_steps:
                break

            input_ids = input_ids.to(device)   # (B, seq_len)
            labels    = labels.to(device)       # (B,)

            # ── Forward ───────────────────────────────────────────────────────
            # Uses: MixerModel → Mamba blocks → selective_scan CUDA kernel
            with autocast(dtype=torch.float16):
                logits = model(input_ids)                        # (B, num_classes)
                loss   = loss_fn(logits, labels) / args.grad_accum

            # ── Backward ──────────────────────────────────────────────────────
            scaler.scale(loss).backward()

            if (global_step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            running_loss += loss.item() * args.grad_accum
            all_preds.append(logits.argmax(dim=-1).detach().cpu())
            all_labels.append(labels.detach().cpu())
            global_step += 1

            # ── Log ───────────────────────────────────────────────────────────
            if global_step % args.log_every == 0:
                avg_loss = running_loss / args.log_every
                preds_t  = torch.cat(all_preds)
                labels_t = torch.cat(all_labels)
                m        = compute_metrics(preds_t, labels_t)
                lr_now   = scheduler.get_last_lr()[0]
                elapsed  = time.time() - t0

                print(
                    f"step {global_step:>5} | "
                    f"loss {avg_loss:.4f} | "
                    f"acc {m['acc']:.3f} | "
                    f"f1 {m['f1']:.3f} | "
                    f"prec {m['precision']:.3f} | "
                    f"rec {m['recall']:.3f} | "
                    f"lr {lr_now:.1e} | "
                    f"{elapsed:.0f}s"
                )
                running_loss = 0.0
                all_preds, all_labels = [], []
                t0 = time.time()

            # ── Validation ────────────────────────────────────────────────────
            if global_step % args.save_every == 0:
                model.eval()
                val_preds, val_labels, val_probs = [], [], []
                val_loss = 0.0

                with torch.no_grad():
                    for v_ids, v_labels in val_loader:
                        v_ids    = v_ids.to(device)
                        v_labels = v_labels.to(device)
                        v_logits = model(v_ids)
                        val_loss += loss_fn(v_logits, v_labels).item()
                        val_probs.append(torch.softmax(v_logits, dim=-1)[:, 1].cpu())
                        val_preds.append(v_logits.argmax(-1).cpu())
                        val_labels.append(v_labels.cpu())

                val_preds  = torch.cat(val_preds)
                val_labels = torch.cat(val_labels)
                val_probs  = torch.cat(val_probs)
                vm = compute_metrics(val_preds, val_labels, probs=val_probs)
                auc_str = f"auc {vm['auc']:.4f} | " if "auc" in vm else ""
                print(
                    f"\n  [VAL] loss {val_loss/len(val_loader):.4f} | "
                    f"{auc_str}"
                    f"acc {vm['acc']:.3f} | f1 {vm['f1']:.3f} | "
                    f"prec {vm['precision']:.3f} | rec {vm['recall']:.3f}\n"
                )

                save_checkpoint(
                    path      = f"{args.save_dir}/step_{global_step}.pt",
                    step      = global_step,
                    model     = model,
                    optimizer = optimizer,
                    scheduler = scheduler,
                    config    = config,
                )
                model.train()

        if global_step >= args.max_steps:
            break

    # ── Final save ────────────────────────────────────────────────────────────
    save_checkpoint(
        path      = f"{args.save_dir}/final.pt",
        step      = global_step,
        model     = model,
        optimizer = optimizer,
        scheduler = scheduler,
        config    = config,
    )
    print("\nTraining complete.")

    # ── Final test evaluation (chromosome-held-out set) ───────────────────────
    if te_ds is not None:
        print("\n" + "=" * 60)
        print("  FINAL TEST SET EVALUATION  (chromosome-held-out)")
        print("=" * 60)
        te_loader = DataLoader(te_ds, batch_size=args.batch_size,
                               shuffle=False, drop_last=False, num_workers=2)
        model.eval()
        te_preds, te_labels, te_probs = [], [], []
        with torch.no_grad():
            for t_ids, t_labels in te_loader:
                t_ids, t_labels = t_ids.to(device), t_labels.to(device)
                t_logits = model(t_ids)
                te_probs.append(torch.softmax(t_logits, dim=-1)[:, 1].cpu())
                te_preds.append(t_logits.argmax(-1).cpu())
                te_labels.append(t_labels.cpu())
        te_preds  = torch.cat(te_preds)
        te_labels = torch.cat(te_labels)
        te_probs  = torch.cat(te_probs)
        tm = compute_metrics(te_preds, te_labels, probs=te_probs)
        print(f"  AUC       : {tm.get('auc', float('nan')):.4f}")
        print(f"  Accuracy  : {tm['acc']:.4f}")
        print(f"  F1        : {tm['f1']:.4f}")
        print(f"  Precision : {tm['precision']:.4f}")
        print(f"  Recall    : {tm['recall']:.4f}")
        print("=" * 60)

    # ── Inference example ─────────────────────────────────────────────────────
    print("\nInference example:")
    model.eval()
    example_seq = "ATCGATCGNNGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
    ids = torch.tensor(
        [tokenizer.encode(example_seq, args.seq_len)], dtype=torch.long
    ).to(device)

    with torch.no_grad():
        logits = model(ids)                              # (1, 2)
        probs  = torch.softmax(logits, dim=-1)
        pred   = logits.argmax(dim=-1).item()

    label_name = "CANCER" if pred == 1 else "NORMAL"
    print(f"  Sequence : {example_seq[:30]}...")
    print(f"  Prediction : {label_name}")
    print(f"  Confidence : Cancer={probs[0,1]:.3f}  Normal={probs[0,0]:.3f}")


if __name__ == "__main__":
    main()
