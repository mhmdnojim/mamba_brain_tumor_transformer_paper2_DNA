"""
train_basset.py — Basset CNN baseline (Kelley et al. 2016) for cancer DNA classification.

Task & data are identical to train.py (Mamba):
    Inputs:  512 bp DNA windows, one-hot encoded (4 × L)
    Outputs: cancer (1) / normal (0)
    Splits:  chromosome-disjoint train/val/test from dataset/splits/

Two model variants:
    --variant small : reduced filter counts, ~3.5 M params (param-matched to Mamba 3.51M)
    --variant full  : original Kelley et al. 2016 config, ~4.5 M params

Run:
    python train_basset.py --variant small --save_dir ./checkpoints_basset_small
    python train_basset.py --variant full  --save_dir ./checkpoints_basset_full
    python train_basset.py --variant small --resume ./checkpoints_basset_small/step_5000.pt
"""

import os
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
#  DATASET — one-hot encoded DNA (4 × L), identical CSV format to train.py
# ─────────────────────────────────────────────────────────────────────────────
NUC_TO_IDX = {'A': 0, 'T': 1, 'G': 2, 'C': 3}  # N → all-zero column


class BassetDataset(Dataset):
    def __init__(self, csv_path: str, seq_len: int = 512):
        self.seq_len   = seq_len
        self.sequences = []
        self.labels    = []

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.sequences.append(row["sequence"].strip().upper())
                self.labels.append(int(row["label"]))

        cancer = sum(self.labels)
        print(f"  {os.path.basename(csv_path):14s}  n={len(self.sequences):>7,}  "
              f"cancer={cancer:>7,}  normal={len(self.labels)-cancer:>7,}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        x   = np.zeros((4, self.seq_len), dtype=np.float32)
        for i, ch in enumerate(seq[:self.seq_len]):
            j = NUC_TO_IDX.get(ch, -1)
            if j >= 0:
                x[j, i] = 1.0
        return torch.from_numpy(x), torch.tensor(self.labels[idx], dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
#  BASSET MODEL
#
#  full  variant: conv (300, 200, 200), FC (1000, 1000)  → ~4.5 M params
#  small variant: conv (200, 200, 200), FC (1100, 500)   → ~3.5 M params (Mamba-matched)
#
#  Pool schedule: MaxPool(3) → MaxPool(4) → MaxPool(4)
#  seq_len=512: 512 → 170 → 42 → 10  (approximate after padding)
# ─────────────────────────────────────────────────────────────────────────────
class Basset(nn.Module):
    def __init__(self, variant: str = "full", seq_len: int = 512,
                 num_classes: int = 2, dropout: float = 0.3):
        super().__init__()

        if variant == "full":
            c1, c2, c3 = 300, 200, 200
            fc1, fc2   = 1000, 1000
        elif variant == "small":
            c1, c2, c3 = 200, 200, 200
            fc1, fc2   = 1100, 500
        else:
            raise ValueError(f"variant must be 'full' or 'small', got {variant!r}")

        self.conv1 = nn.Conv1d(4,  c1, kernel_size=19, padding=9)
        self.bn1   = nn.BatchNorm1d(c1)
        self.pool1 = nn.MaxPool1d(3)

        self.conv2 = nn.Conv1d(c1, c2, kernel_size=11, padding=5)
        self.bn2   = nn.BatchNorm1d(c2)
        self.pool2 = nn.MaxPool1d(4)

        self.conv3 = nn.Conv1d(c2, c3, kernel_size=7, padding=3)
        self.bn3   = nn.BatchNorm1d(c3)
        self.pool3 = nn.MaxPool1d(4)

        with torch.no_grad():
            dummy       = torch.zeros(1, 4, seq_len)
            flat_dim    = self._conv_forward(dummy).flatten(1).shape[1]

        self.fc1     = nn.Linear(flat_dim, fc1)
        self.fc2     = nn.Linear(fc1, fc2)
        self.head    = nn.Linear(fc2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def _conv_forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv_forward(x).flatten(1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
#  METRICS  (identical to train.py)
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(preds: torch.Tensor, labels: torch.Tensor) -> dict:
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    acc       = (tp + tn) / (tp + fp + fn + tn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    return {"acc": acc, "precision": precision, "recall": recall, "f1": f1}


def compute_auc(logits: torch.Tensor, labels: torch.Tensor) -> float:
    labels_np = labels.cpu().numpy()
    if len(set(labels_np.tolist())) < 2:
        return float("nan")
    probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    return float(roc_auc_score(labels_np, probs))


def evaluate(model, loader, device, loss_fn=None):
    model.eval()
    logits_all, labels_all = [], []
    loss_sum, n_batches = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x      = x.to(device)
            logits = model(x)
            if loss_fn is not None:
                loss_sum  += loss_fn(logits, y.to(device)).item()
                n_batches += 1
            logits_all.append(logits.cpu())
            labels_all.append(y)
    logits_all = torch.cat(logits_all)
    labels_all = torch.cat(labels_all)
    avg_loss   = loss_sum / max(1, n_batches) if loss_fn is not None else None
    return logits_all, labels_all, avg_loss


# ─────────────────────────────────────────────────────────────────────────────
#  OPTIMIZER  (weight-decay groups — BN and bias excluded)
# ─────────────────────────────────────────────────────────────────────────────
def build_optimizer(model, args):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "bias" in name or "bn" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay,    "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  CHECKPOINT
# ─────────────────────────────────────────────────────────────────────────────
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
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Basset CNN baseline for cancer DNA classification")

    p.add_argument("--variant", choices=["full", "small"], default="full",
                   help="full=original Basset (~4.5M), small=param-matched to Mamba (~3.5M)")
    p.add_argument("--splits_dir",   type=str,
                   default="/content/drive/MyDrive/Mamba-DNA-1/dataset/splits")
    p.add_argument("--seq_len",      type=int,   default=512)
    p.add_argument("--num_classes",  type=int,   default=2)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--warmup_steps", type=int,   default=500)
    p.add_argument("--max_steps",    type=int,   default=25000)
    p.add_argument("--clip_grad",    type=float, default=1.0)
    p.add_argument("--dropout",      type=float, default=0.3)
    p.add_argument("--log_every",    type=int,   default=100)
    p.add_argument("--save_every",   type=int,   default=1000)
    p.add_argument("--save_dir",     type=str,   default="./checkpoints_basset")
    p.add_argument("--resume",       type=str,   default=None)
    p.add_argument("--results_json", type=str,
                   default="/content/drive/MyDrive/Mamba-DNA-1/reports/basset_results.json")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(f"  Basset CNN ({args.variant}) — chromosome-disjoint splits")
    print(f"  Device  : {device}")
    if device.type == "cuda":
        print(f"  GPU     : {torch.cuda.get_device_name(0)}")
    print(f"  seq_len : {args.seq_len}   classes : {args.num_classes}")
    print("=" * 60)

    # ── Datasets ─────────────────────────────────────────────────────────────
    print(f"\nLoading splits from: {args.splits_dir}")
    train_csv = os.path.join(args.splits_dir, "train.csv")
    val_csv   = os.path.join(args.splits_dir, "val.csv")
    test_csv  = os.path.join(args.splits_dir, "test.csv")
    for path in (train_csv, val_csv, test_csv):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing split file: {path}")

    train_ds = BassetDataset(train_csv, args.seq_len)
    val_ds   = BassetDataset(val_csv,   args.seq_len)
    test_ds  = BassetDataset(test_csv,  args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              drop_last=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              drop_last=False, num_workers=2, pin_memory=True)
    print(f"  steps/epoch : {len(train_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\nBuilding Basset ({args.variant}) ...")
    model    = Basset(variant=args.variant, seq_len=args.seq_len,
                      num_classes=args.num_classes, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters : {n_params/1e6:.2f}M")
    print(f"  Input      : one-hot (4 × {args.seq_len})")

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    optimizer = build_optimizer(model, args)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = args.warmup_steps,
        num_training_steps = args.max_steps,
    )
    scaler  = torch.amp.GradScaler("cuda")
    loss_fn = nn.CrossEntropyLoss()

    # ── Resume ────────────────────────────────────────────────────────────────
    global_step = 0
    if args.resume and os.path.exists(args.resume):
        global_step = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        print(f"  Resumed from step {global_step}")

    os.makedirs(args.save_dir, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    print("\nStarting training...")
    model.train()
    optimizer.zero_grad()

    running_loss = 0.0
    all_preds, all_labels_buf = [], []
    t0 = time.time()
    best_val_auc = 0.0

    for _ in range(99999):
        for x, y in train_loader:
            if global_step >= args.max_steps:
                break

            x = x.to(device)
            y = y.to(device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(x)
                loss   = loss_fn(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            all_preds.append(logits.argmax(-1).detach().cpu())
            all_labels_buf.append(y.detach().cpu())
            global_step += 1

            if global_step % args.log_every == 0:
                avg_loss = running_loss / args.log_every
                m        = compute_metrics(torch.cat(all_preds), torch.cat(all_labels_buf))
                lr_now   = scheduler.get_last_lr()[0]
                print(f"step {global_step:>5} | loss {avg_loss:.4f} | "
                      f"acc {m['acc']:.3f} | f1 {m['f1']:.3f} | "
                      f"prec {m['precision']:.3f} | rec {m['recall']:.3f} | "
                      f"lr {lr_now:.1e} | {time.time()-t0:.0f}s")
                running_loss = 0.0
                all_preds, all_labels_buf = [], []
                t0 = time.time()

            if global_step % args.save_every == 0:
                v_logits, v_labels, v_loss = evaluate(model, val_loader, device, loss_fn)
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

    save_checkpoint(f"{args.save_dir}/final.pt", global_step, model, optimizer, scheduler, args)
    print("\nTraining complete.")

    # ── Final test evaluation ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  FINAL TEST EVALUATION — Basset-{args.variant} (chr19 + chr22)")
    print("=" * 60)

    best_path = f"{args.save_dir}/best_val.pt"
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"  Loaded best_val.pt from step {ckpt['step']} "
              f"(best_val_auc={best_val_auc:.4f})")

    t_logits, t_labels, _ = evaluate(model, test_loader, device)
    test_auc   = compute_auc(t_logits, t_labels)
    tm         = compute_metrics(t_logits.argmax(-1), t_labels)

    print(f"\n  Overall test AUC : {test_auc:.4f}")
    print(f"  Acc  : {tm['acc']:.4f}   F1   : {tm['f1']:.4f}")
    print(f"  Prec : {tm['precision']:.4f}   Rec  : {tm['recall']:.4f}")

    print("\n  Per-chromosome test AUC:")
    test_chroms_list = []
    with open(test_csv, newline="") as f:
        reader    = csv.DictReader(f)
        chrom_col = "chromosome" if "chromosome" in reader.fieldnames else "chrom"
        for row in reader:
            test_chroms_list.append(row[chrom_col])

    test_chroms_arr = np.array(test_chroms_list)
    test_probs_arr  = torch.softmax(t_logits, dim=-1)[:, 1].numpy()
    test_labels_arr = t_labels.numpy()

    per_chrom = {}
    for chrom in sorted(set(test_chroms_arr.tolist())):
        mask = test_chroms_arr == chrom
        if mask.sum() > 20 and len(set(test_labels_arr[mask].tolist())) > 1:
            chrom_auc = roc_auc_score(test_labels_arr[mask], test_probs_arr[mask])
            cf        = test_labels_arr[mask].mean()
            per_chrom[chrom] = {"auc": round(chrom_auc, 4),
                                "n": int(mask.sum()), "cancer_frac": round(float(cf), 3)}
            print(f"    {chrom}: AUC={chrom_auc:.4f}  n={int(mask.sum()):>6,}  "
                  f"cancer_frac={cf:.3f}")

    print("\n" + "=" * 60)

    # ── Save results JSON ─────────────────────────────────────────────────────
    results = {
        "model": f"Basset-{args.variant}, {n_params/1e6:.2f}M params",
        "variant": args.variant,
        "training_setup": {
            "split": "chromosome-disjoint: train/val/test",
            "test_chroms": ["chr19", "chr22"],
            "max_steps": args.max_steps,
            "best_val_auc": round(best_val_auc, 4),
        },
        "test_results": {
            "overall_auc":       round(test_auc, 4),
            "overall_acc":       round(tm["acc"], 4),
            "overall_f1":        round(tm["f1"], 4),
            "overall_precision": round(tm["precision"], 4),
            "overall_recall":    round(tm["recall"], 4),
            "per_chromosome":    per_chrom,
        },
    }
    os.makedirs(os.path.dirname(args.results_json), exist_ok=True)
    with open(args.results_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: {args.results_json}")


if __name__ == "__main__":
    main()
