"""
cnn_baseline.py — 1-D CNN baseline for cancer somatic-mutation context classification.

Same task, same splits, same metrics as train.py (Mamba), so the comparison is fair:
  - Input  : 512 bp DNA sequence tokenized character-level (A/T/G/C/N → int)
  - Output : Cancer (1) or Normal (0)
  - Splits : chromosome-disjoint train/val/test from --splits_dir

Architecture  : Embedding → 4 × (Conv1d + BN + ReLU + MaxPool) → GlobalAvgPool → MLP
Parameters    : ~3.8 M  (comparable to Mamba 3.51 M)

Run:
    python cnn_baseline.py
    python cnn_baseline.py --max_steps 20000 --save_dir ./checkpoints_cnn
    python cnn_baseline.py --resume ./checkpoints_cnn/step_5000.pt
"""

import os
import time
import argparse
import csv
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score


# ─────────────────────────────────────────────────────────────────────────────
#  TOKENIZER  (identical to train.py — ensures same encoding)
# ─────────────────────────────────────────────────────────────────────────────
class DNATokenizer:
    VOCAB = {"[PAD]": 0, "[UNK]": 1, "A": 2, "T": 3, "G": 4, "C": 5, "N": 6}
    PAD_ID = 0
    vocab_size = len(VOCAB)

    def encode(self, sequence: str, max_len: int) -> list:
        ids = [self.VOCAB.get(c.upper(), self.VOCAB["[UNK]"]) for c in sequence]
        ids = ids[:max_len]
        ids += [self.PAD_ID] * (max_len - len(ids))
        return ids


# ─────────────────────────────────────────────────────────────────────────────
#  DATASET  (identical to train.py)
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
        cancer = sum(self.labels)
        print(f"  {os.path.basename(csv_path):14s}  n={len(self.sequences):>7,}  "
              f"cancer={cancer:>7,}  normal={len(self.labels)-cancer:>7,}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        ids   = torch.tensor(self.tokenizer.encode(self.sequences[idx], self.seq_len),
                             dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return ids, label


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL  — 1-D CNN
#
#  Embedding(7, 128)
#  → Conv1d(128→256, k=8) → BN → ReLU → MaxPool(2)     L: 512 → 256
#  → Conv1d(256→512, k=8) → BN → ReLU → MaxPool(2)     L: 256 → 128
#  → Conv1d(512→512, k=8) → BN → ReLU → MaxPool(2)     L: 128 → 64
#  → Conv1d(512→256, k=4) → BN → ReLU → GlobalAvgPool  L: 64  → 1
#  → Dropout(0.3) → Linear(256→128) → ReLU → Linear(128→2)
#
#  Total params ≈ 3.8 M  (vs Mamba 3.51 M — comparable scale)
# ─────────────────────────────────────────────────────────────────────────────
class CNNCancerClassifier(nn.Module):
    def __init__(self, vocab_size: int = 7, embed_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, 256, kernel_size=8, padding=4), nn.BatchNorm1d(256),
            nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(256, 512, kernel_size=8, padding=4), nn.BatchNorm1d(512),
            nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(512, 512, kernel_size=8, padding=4), nn.BatchNorm1d(512),
            nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(512, 256, kernel_size=4, padding=2), nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x).permute(0, 2, 1)  # (B, E, L)
        x = self.conv(x)                          # (B, 256, L')
        x = x.mean(dim=-1)                        # global average pool
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
#  METRICS
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


# ─────────────────────────────────────────────────────────────────────────────
#  EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(model, loader, device, loss_fn=None):
    model.eval()
    logits_all, labels_all = [], []
    loss_sum, n_batches = 0.0, 0
    with torch.no_grad():
        for ids, labels in loader:
            ids        = ids.to(device)
            labels_dev = labels.to(device)
            logits     = model(ids)
            if loss_fn is not None:
                loss_sum  += loss_fn(logits, labels_dev).item()
                n_batches += 1
            logits_all.append(logits.cpu())
            labels_all.append(labels)
    logits_all = torch.cat(logits_all)
    labels_all = torch.cat(labels_all)
    avg_loss   = loss_sum / max(1, n_batches) if loss_fn is not None else None
    return logits_all, labels_all, avg_loss


# ─────────────────────────────────────────────────────────────────────────────
#  CHECKPOINT
# ─────────────────────────────────────────────────────────────────────────────
def save_checkpoint(path, step, model, optimizer, scheduler):
    torch.save({
        "step": step,
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
    p = argparse.ArgumentParser(description="1-D CNN Cancer Gene Baseline")
    p.add_argument("--splits_dir", type=str,
                   default="/content/drive/MyDrive/Mamba-DNA-1/dataset/splits")
    p.add_argument("--seq_len",      type=int,   default=512)
    p.add_argument("--embed_dim",    type=int,   default=128)
    p.add_argument("--num_classes",  type=int,   default=2)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_steps",    type=int,   default=20000)
    p.add_argument("--warmup_steps", type=int,   default=200)
    p.add_argument("--clip_grad",    type=float, default=1.0)
    p.add_argument("--log_every",    type=int,   default=100)
    p.add_argument("--save_every",   type=int,   default=1000)
    p.add_argument("--save_dir",     type=str,   default="./checkpoints_cnn")
    p.add_argument("--resume",       type=str,   default=None)
    p.add_argument("--results_json", type=str,
                   default="/content/drive/MyDrive/Mamba-DNA-1/reports/cnn_results.json")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  1-D CNN Cancer Gene Baseline (chromosome-disjoint splits)")
    print(f"  Device  : {device}")
    if device.type == "cuda":
        print(f"  GPU     : {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    tokenizer = DNATokenizer()

    # ── Splits ────────────────────────────────────────────────────────────────
    print(f"\nLoading splits from: {args.splits_dir}")
    train_csv = os.path.join(args.splits_dir, "train.csv")
    val_csv   = os.path.join(args.splits_dir, "val.csv")
    test_csv  = os.path.join(args.splits_dir, "test.csv")
    for p in (train_csv, val_csv, test_csv):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing split file: {p}")

    train_ds = CancerGeneDataset(train_csv, tokenizer, args.seq_len)
    val_ds   = CancerGeneDataset(val_csv,   tokenizer, args.seq_len)
    test_ds  = CancerGeneDataset(test_csv,  tokenizer, args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  drop_last=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, drop_last=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, drop_last=False, num_workers=2)

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\nBuilding CNN model...")
    model   = CNNCancerClassifier(tokenizer.vocab_size, args.embed_dim, args.num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters : {n_params/1e6:.2f}M")
    print(f"  Architecture: Embed({tokenizer.vocab_size},{args.embed_dim}) → "
          f"4×Conv1d → GlobalAvgPool → MLP")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    # Cosine decay with linear warmup (manual)
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return max(0.0, 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.amp.GradScaler("cuda")
    loss_fn   = nn.CrossEntropyLoss()

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
        for ids, labels in train_loader:
            if global_step >= args.max_steps:
                break

            ids    = ids.to(device)
            labels = labels.to(device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(ids)
                loss   = loss_fn(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            all_preds.append(logits.argmax(-1).detach().cpu())
            all_labels_buf.append(labels.detach().cpu())
            global_step += 1

            # ── Log ───────────────────────────────────────────────────────────
            if global_step % args.log_every == 0:
                avg_loss = running_loss / args.log_every
                m = compute_metrics(torch.cat(all_preds), torch.cat(all_labels_buf))
                lr_now = scheduler.get_last_lr()[0]
                print(f"step {global_step:>5} | loss {avg_loss:.4f} | "
                      f"acc {m['acc']:.3f} | f1 {m['f1']:.3f} | "
                      f"prec {m['precision']:.3f} | rec {m['recall']:.3f} | "
                      f"lr {lr_now:.1e} | {time.time()-t0:.0f}s")
                running_loss = 0.0
                all_preds, all_labels_buf = [], []
                t0 = time.time()

            # ── Validation ────────────────────────────────────────────────────
            if global_step % args.save_every == 0:
                v_logits, v_labels, v_loss = evaluate(model, val_loader, device, loss_fn)
                v_auc   = compute_auc(v_logits, v_labels)
                v_preds = v_logits.argmax(-1)
                vm      = compute_metrics(v_preds, v_labels)
                print(f"\n  [VAL] loss {v_loss:.4f} | auc {v_auc:.4f} | "
                      f"acc {vm['acc']:.3f} | f1 {vm['f1']:.3f} | "
                      f"prec {vm['precision']:.3f} | rec {vm['recall']:.3f}")

                save_checkpoint(f"{args.save_dir}/step_{global_step}.pt",
                                global_step, model, optimizer, scheduler)

                if v_auc > best_val_auc:
                    best_val_auc = v_auc
                    save_checkpoint(f"{args.save_dir}/best_val.pt",
                                    global_step, model, optimizer, scheduler)
                    print(f"  [BEST VAL AUC] {v_auc:.4f} at step {global_step}\n")
                else:
                    print()

                model.train()

        if global_step >= args.max_steps:
            break

    # ── Final save ────────────────────────────────────────────────────────────
    save_checkpoint(f"{args.save_dir}/final.pt", global_step, model, optimizer, scheduler)
    print("\nTraining complete.")

    # ── Final test evaluation ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL TEST EVALUATION — CNN (chr19 + chr22, held out)")
    print("=" * 60)

    best_path = f"{args.save_dir}/best_val.pt"
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"  Loaded best_val.pt from step {ckpt['step']} "
              f"(best_val_auc={best_val_auc:.4f})")

    t_logits, t_labels, _ = evaluate(model, test_loader, device)
    test_auc   = compute_auc(t_logits, t_labels)
    test_preds = t_logits.argmax(-1)
    tm         = compute_metrics(test_preds, t_labels)

    print(f"\n  Overall test AUC : {test_auc:.4f}")
    print(f"  Acc  : {tm['acc']:.4f}   F1   : {tm['f1']:.4f}")
    print(f"  Prec : {tm['precision']:.4f}   Rec  : {tm['recall']:.4f}")

    # Per-chromosome breakdown
    print("\n  Per-chromosome test AUC:")
    test_chroms_list = []
    with open(test_csv, newline="") as f:
        reader = csv.DictReader(f)
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
            cf = test_labels_arr[mask].mean()
            per_chrom[chrom] = {"auc": round(chrom_auc, 4),
                                "n": int(mask.sum()), "cancer_frac": round(float(cf), 3)}
            print(f"    {chrom}: AUC={chrom_auc:.4f}  n={int(mask.sum()):>6,}  "
                  f"cancer_frac={cf:.3f}")

    print("\n" + "=" * 60)

    # ── Save results JSON ─────────────────────────────────────────────────────
    results = {
        "model": f"1D-CNN, embed_dim={args.embed_dim}, ~{n_params/1e6:.2f}M params",
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

    out_path = args.results_json
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: {out_path}")


if __name__ == "__main__":
    main()
