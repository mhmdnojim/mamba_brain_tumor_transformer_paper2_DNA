"""
train_hyena.py — HyenaDNA baseline (from-scratch) for cancer DNA classification.

Same task, data, and protocol as train.py (Mamba) and train_basset.py (CNN):
    Inputs:  512bp DNA windows, character-level token IDs
    Outputs: cancer (1) / normal (0)
    Splits:  chromosome-disjoint train/val/test from dataset/splits/

Model: stack of Hyena layers (FFT-based long convolutions, from Hyena Hierarchy /
HyenaDNA, Poli et al. 2023, Nguyen et al. 2023). Trained from scratch.

Default config: d_model=256, n_layer=4, filter_order=128, mlp_ratio=4, order=2
                → 3.45M params (matched to Mamba 3.51M).

Requires the HyenaDNA repo cloned to /content/work/hyena-dna/ (or --hyena_dir):
    git clone https://github.com/HazyResearch/hyena-dna.git /content/work/hyena-dna
    pip install hydra-core omegaconf pytorch-lightning

Run:
    python train_hyena.py --save_dir ./checkpoints_hyena
    python train_hyena.py --resume ./checkpoints_hyena/step_5000.pt
"""

import os
import sys
import time
import argparse
import csv
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import roc_auc_score


# ─────────────────────────────────────────────────────────────────────────────
#  HYENA LAYER IMPORT (vendored from hyena-dna repo)
# ─────────────────────────────────────────────────────────────────────────────
def _load_hyena_operator(hyena_dir: str):
    if hyena_dir not in sys.path:
        sys.path.insert(0, hyena_dir)
    # Clear any stale cached imports from a previous failed attempt
    for mod in list(sys.modules.keys()):
        if mod.startswith("src.") or mod == "src":
            del sys.modules[mod]
    try:
        from src.models.sequence.hyena import HyenaOperator
        return HyenaOperator
    except ImportError as e:
        raise ImportError(
            f"Cannot import HyenaOperator from {hyena_dir}/src/models/sequence/hyena.py\n"
            f"Original error: {e}\n"
            "Make sure you have run:\n"
            "    git clone https://github.com/HazyResearch/hyena-dna.git {hyena_dir}\n"
            "    pip install hydra-core omegaconf pytorch-lightning"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  TOKENIZER — character-level DNA (same as train.py for Mamba)
# ─────────────────────────────────────────────────────────────────────────────
class DNATokenizer:
    """A=0, C=1, G=2, T=3, N=4, [PAD]=5, [UNK]=6  (vocab size = 7)"""
    def __init__(self):
        self.vocab = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4, '[PAD]': 5, '[UNK]': 6}
        self.vocab_size = len(self.vocab)

    def encode(self, seq: str, max_length: int) -> list:
        ids = [self.vocab.get(c.upper(), self.vocab['[UNK]']) for c in seq[:max_length]]
        if len(ids) < max_length:
            ids += [self.vocab['[PAD]']] * (max_length - len(ids))
        return ids


# ─────────────────────────────────────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────────────────────────────────────
class HyenaDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer: DNATokenizer, seq_len: int = 512):
        self.tok      = tokenizer
        self.seq_len  = seq_len
        self.sequences = []
        self.labels    = []
        self.chroms    = []

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            # Handle both "chromosome" and legacy "chrom" column names
            chrom_col = "chromosome" if "chromosome" in reader.fieldnames else "chrom"
            for row in reader:
                self.sequences.append(row["sequence"].strip().upper())
                self.labels.append(int(row["label"]))
                self.chroms.append(row.get(chrom_col, ""))

        cancer = sum(self.labels)
        print(f"  {os.path.basename(csv_path):14s}  n={len(self.sequences):>7,}  "
              f"cancer={cancer:>7,}  normal={len(self.labels)-cancer:>7,}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        ids = self.tok.encode(self.sequences[idx], self.seq_len)
        return (torch.tensor(ids, dtype=torch.long),
                torch.tensor(self.labels[idx], dtype=torch.long))


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────────────────────────────────────
class HyenaBlock(nn.Module):
    """Pre-norm Hyena operator + MLP with residual connections."""
    def __init__(self, HyenaOperator, d_model, l_max,
                 order=2, filter_order=128, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.hyena = HyenaOperator(
            d_model=d_model, l_max=l_max,
            order=order, filter_order=filter_order,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp   = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Linear(d_model * mlp_ratio, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, L, D)
        x = x + self.hyena(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class HyenaClassifier(nn.Module):
    """
    HyenaDNA-style binary classifier: embedding → N Hyena blocks → mean-pool → head.

    Default (~3.45M params, matched to Mamba 3.51M):
        vocab_size=7, d_model=256, n_layer=4, l_max=512,
        order=2, filter_order=128, mlp_ratio=4, dropout=0.1
    """
    def __init__(self, HyenaOperator, vocab_size=7, d_model=256, n_layer=4,
                 l_max=512, num_classes=2, order=2, filter_order=128,
                 mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            HyenaBlock(HyenaOperator, d_model, l_max, order,
                       filter_order, mlp_ratio, dropout)
            for _ in range(n_layer)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, ids):
        # ids: (B, L)  →  (B, L, D)  →  (B, D)  →  (B, C)
        x = self.embed(ids)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(preds: torch.Tensor, labels: torch.Tensor) -> dict:
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    acc  = (tp + tn) / (tp + fp + fn + tn + 1e-8)
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    return {"acc": acc, "precision": prec, "recall": rec, "f1": f1}


def compute_auc(logits: torch.Tensor, labels: torch.Tensor) -> float:
    labels_np = labels.cpu().numpy()
    if len(set(labels_np.tolist())) < 2:
        return float("nan")
    probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    return float(roc_auc_score(labels_np, probs))


# ─────────────────────────────────────────────────────────────────────────────
#  OPTIMIZER / CHECKPOINT
# ─────────────────────────────────────────────────────────────────────────────
def build_optimizer(model, args):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "bias" in name or "norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay,    "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
    )


def save_checkpoint(path, step, model, optimizer, scheduler, args):
    torch.save({
        "step": step, "args": vars(args),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, path)
    print(f"  [saved] {path}")


def load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["step"]


# ─────────────────────────────────────────────────────────────────────────────
#  EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(model, loader, device, loss_fn=None):
    model.eval()
    logits_all, labels_all = [], []
    loss_sum, n_batches = 0.0, 0
    with torch.no_grad():
        for ids, y in loader:
            ids = ids.to(device)
            y_dev = y.to(device)
            logits = model(ids)
            if loss_fn is not None:
                loss_sum += loss_fn(logits, y_dev).item()
                n_batches += 1
            logits_all.append(logits.cpu())
            labels_all.append(y)
    avg_loss = loss_sum / max(1, n_batches) if loss_fn else None
    return torch.cat(logits_all), torch.cat(labels_all), avg_loss


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="HyenaDNA baseline (from-scratch) for cancer DNA classification")

    p.add_argument("--hyena_dir",    type=str,
                   default="/content/work/hyena-dna",
                   help="Path to cloned HazyResearch/hyena-dna repo")

    # Data
    p.add_argument("--splits_dir",   type=str,
                   default="/content/drive/MyDrive/Mamba-DNA-1/dataset/splits")
    p.add_argument("--seq_len",      type=int,   default=512)
    p.add_argument("--num_classes",  type=int,   default=2)

    # Model
    p.add_argument("--d_model",      type=int,   default=256)
    p.add_argument("--n_layer",      type=int,   default=4)
    p.add_argument("--order",        type=int,   default=2)
    p.add_argument("--filter_order", type=int,   default=128)
    p.add_argument("--mlp_ratio",    type=int,   default=4)
    p.add_argument("--dropout",      type=float, default=0.1)

    # Training
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--grad_accum",   type=int,   default=1)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int,   default=500)
    p.add_argument("--max_steps",    type=int,   default=30000)
    p.add_argument("--clip_grad",    type=float, default=1.0)

    # Logging / saving
    p.add_argument("--log_every",    type=int,   default=100)
    p.add_argument("--save_every",   type=int,   default=1000)
    p.add_argument("--save_dir",     type=str,   default="./checkpoints_hyena")
    p.add_argument("--resume",       type=str,   default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--results_json", type=str,   default=None)

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    HyenaOperator = _load_hyena_operator(args.hyena_dir)

    print("=" * 60)
    print("  HyenaDNA baseline (from-scratch) — chromosome-disjoint")
    print(f"  Device  : {device}")
    if device.type == "cuda":
        print(f"  GPU     : {torch.cuda.get_device_name(0)}")
    print(f"  d_model : {args.d_model}   n_layer : {args.n_layer}")
    print(f"  order   : {args.order}   filter_order : {args.filter_order}"
          f"   mlp_ratio : {args.mlp_ratio}")
    print(f"  seq_len : {args.seq_len}   classes : {args.num_classes}")
    print("=" * 60)

    # ── Datasets ──────────────────────────────────────────────────────────────
    tokenizer = DNATokenizer()
    print(f"\nDNA vocabulary size: {tokenizer.vocab_size}")

    print(f"\nLoading splits from: {args.splits_dir}")
    train_csv = os.path.join(args.splits_dir, "train.csv")
    val_csv   = os.path.join(args.splits_dir, "val.csv")
    test_csv  = os.path.join(args.splits_dir, "test.csv")
    for f in (train_csv, val_csv, test_csv):
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing split file: {f}")

    train_ds = HyenaDataset(train_csv, tokenizer, args.seq_len)
    val_ds   = HyenaDataset(val_csv,   tokenizer, args.seq_len)
    test_ds  = HyenaDataset(test_csv,  tokenizer, args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              drop_last=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              drop_last=False, num_workers=2, pin_memory=True)
    print(f"  steps/epoch : {len(train_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\nBuilding HyenaClassifier ...")
    model = HyenaClassifier(
        HyenaOperator=HyenaOperator,
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model, n_layer=args.n_layer, l_max=args.seq_len,
        num_classes=args.num_classes, order=args.order,
        filter_order=args.filter_order, mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters : {n_params/1e6:.2f}M")
    print(f"  Input      : token IDs (vocab size {tokenizer.vocab_size}, seq_len {args.seq_len})")

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    optimizer = build_optimizer(model, args)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = args.warmup_steps,
        num_training_steps = args.max_steps,
    )
    scaler  = torch.amp.GradScaler("cuda")
    loss_fn = nn.CrossEntropyLoss()

    os.makedirs(args.save_dir, exist_ok=True)

    # ── Resume ────────────────────────────────────────────────────────────────
    global_step = 0
    if args.resume and os.path.exists(args.resume):
        global_step = load_checkpoint(
            args.resume, model, optimizer, scheduler, device)
        print(f"  Resumed from step {global_step}")

    # ── Training loop ─────────────────────────────────────────────────────────
    print("\nStarting training...")
    model.train()
    optimizer.zero_grad()

    running_loss = 0.0
    all_preds, all_labels_buf = [], []
    t0 = time.time()
    best_val_auc = 0.0
    accum_count  = 0

    for _ in range(99999):
        for ids, y in train_loader:
            if global_step >= args.max_steps:
                break

            ids  = ids.to(device)
            y    = y.to(device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(ids)
                loss   = loss_fn(logits, y) / args.grad_accum

            scaler.scale(loss).backward()
            accum_count += 1

            if accum_count >= args.grad_accum:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                accum_count  = 0
                global_step += 1

                running_loss += loss.item() * args.grad_accum
                all_preds.append(logits.argmax(dim=-1).detach().cpu())
                all_labels_buf.append(y.detach().cpu())

                if global_step % args.log_every == 0:
                    avg_loss = running_loss / args.log_every
                    m        = compute_metrics(torch.cat(all_preds),
                                               torch.cat(all_labels_buf))
                    lr_now   = scheduler.get_last_lr()[0]
                    print(f"step {global_step:>5} | loss {avg_loss:.4f} | "
                          f"acc {m['acc']:.3f} | f1 {m['f1']:.3f} | "
                          f"prec {m['precision']:.3f} | rec {m['recall']:.3f} | "
                          f"lr {lr_now:.1e} | {time.time()-t0:.0f}s")
                    running_loss = 0.0
                    all_preds, all_labels_buf = [], []
                    t0 = time.time()

                if global_step % args.save_every == 0:
                    v_logits, v_labels, v_loss = evaluate(
                        model, val_loader, device, loss_fn)
                    v_auc = compute_auc(v_logits, v_labels)
                    vm    = compute_metrics(v_logits.argmax(-1), v_labels)
                    print(f"\n  [VAL] loss {v_loss:.4f} | auc {v_auc:.4f} | "
                          f"acc {vm['acc']:.3f} | f1 {vm['f1']:.3f} | "
                          f"prec {vm['precision']:.3f} | rec {vm['recall']:.3f}")

                    save_checkpoint(f"{args.save_dir}/step_{global_step}.pt",
                                    global_step, model, optimizer, scheduler, args)

                    if v_auc > best_val_auc:
                        best_val_auc = v_auc
                        save_checkpoint(f"{args.save_dir}/best_val.pt",
                                        global_step, model, optimizer, scheduler, args)
                        print(f"  [BEST VAL AUC] {v_auc:.4f} at step {global_step}\n")
                    else:
                        print()

                    model.train()

        if global_step >= args.max_steps:
            break

    save_checkpoint(f"{args.save_dir}/final.pt",
                    global_step, model, optimizer, scheduler, args)
    print("\nTraining complete.")

    # ── Final test evaluation ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL TEST EVALUATION — HyenaDNA (chr19 + chr22, held out)")
    print("=" * 60)

    best_path = f"{args.save_dir}/best_val.pt"
    best_step = global_step
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        best_step = ckpt["step"]
        print(f"  Loaded best_val.pt from step {best_step} "
              f"(best_val_auc={best_val_auc:.4f})")

    t_logits, t_labels, _ = evaluate(model, test_loader, device)
    test_auc   = compute_auc(t_logits, t_labels)
    test_preds = t_logits.argmax(-1)
    tm = compute_metrics(test_preds, t_labels)

    print(f"\n  Overall test AUC : {test_auc:.4f}")
    print(f"  Acc  : {tm['acc']:.4f}   F1   : {tm['f1']:.4f}")
    print(f"  Prec : {tm['precision']:.4f}   Rec  : {tm['recall']:.4f}")

    # Per-chromosome AUC
    print("\n  Per-chromosome test AUC:")
    test_chroms_arr = np.array(test_ds.chroms)
    test_probs_np   = torch.softmax(t_logits, dim=-1)[:, 1].numpy()
    test_labels_np  = t_labels.numpy()

    per_chrom = {}
    for chrom in sorted(set(test_ds.chroms)):
        mask = test_chroms_arr == chrom
        if mask.sum() > 20 and len(set(test_labels_np[mask].tolist())) > 1:
            c_auc = roc_auc_score(test_labels_np[mask], test_probs_np[mask])
            cf    = test_labels_np[mask].mean()
            print(f"    {chrom}: AUC={c_auc:.4f}  n={int(mask.sum()):>6,}  "
                  f"cancer_frac={cf:.3f}")
            per_chrom[chrom] = {
                "auc": float(c_auc), "n": int(mask.sum()), "cancer_frac": float(cf)}

    print("\n" + "=" * 60)

    if args.results_json:
        results = {
            "model": (f"HyenaDNA from-scratch, d_model={args.d_model}, "
                      f"n_layer={args.n_layer}, order={args.order}, "
                      f"filter_order={args.filter_order}, "
                      f"mlp_ratio={args.mlp_ratio}, {n_params/1e6:.2f}M params"),
            "training_setup": {
                "max_steps": args.max_steps,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "warmup_steps": args.warmup_steps,
                "split": "chromosome-disjoint: train/val/test",
                "test_chroms": ["chr19", "chr22"],
            },
            "best_val_step": best_step,
            "best_val_auc":  round(best_val_auc, 4),
            "test_auc":      round(test_auc, 4),
            "test_acc":      round(tm["acc"], 4),
            "test_f1":       round(tm["f1"], 4),
            "test_precision":round(tm["precision"], 4),
            "test_recall":   round(tm["recall"], 4),
            "per_chromosome": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                                   for kk, vv in v.items()}
                               for k, v in per_chrom.items()},
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.results_json)), exist_ok=True)
        with open(args.results_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved: {args.results_json}\n")


if __name__ == "__main__":
    main()
