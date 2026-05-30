"""
train.py — Mamba Cancer Gene Detection (Sequence Classification)

Task:
    Input  : DNA sequence  (e.g. "ATCGATCGNNGATC...")
    Output : Cancer (1) or Normal (0)

Data format expected: chromosome-disjoint pre-built splits at --splits_dir
    train.csv, val.csv, test.csv   each with columns: sequence, label, chromosome

Run:
    python train.py
    python train.py --max_steps 50000 --save_dir ./checkpoints_matched
    python train.py --resume ./checkpoints_matched/step_5000.pt
"""

import os
import math
import time
import argparse
import csv

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import roc_auc_score

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

    def batch_encode(self, sequences: list, max_len: int) -> torch.Tensor:
        return torch.tensor(
            [self.encode(s, max_len) for s in sequences], dtype=torch.long
        )


# ─────────────────────────────────────────────────────────────────────────────
#  DATASET — reads one split CSV (train.csv | val.csv | test.csv)
#  CSV columns: sequence, label, chromosome  (chromosome preserved, not used by model)
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
        input_ids = torch.tensor(
            self.tokenizer.encode(self.sequences[idx], self.seq_len),
            dtype=torch.long,
        )
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return input_ids, label


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL  — Mamba backbone  +  classification head
# ─────────────────────────────────────────────────────────────────────────────
class MambaCancerClassifier(nn.Module):
    def __init__(self, config: MambaConfig, num_classes: int = 2):
        super().__init__()
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
        self.norm       = nn.LayerNorm(config.d_model)
        self.classifier = nn.Linear(config.d_model, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(input_ids)          # (B, L, d_model)
        pooled = hidden.mean(dim=1)                # (B, d_model)
        pooled = self.norm(pooled)
        logits = self.classifier(pooled)           # (B, num_classes)
        return logits


# ─────────────────────────────────────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(preds: torch.Tensor, labels: torch.Tensor):
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
    """Cancer-class probability vs. binary label. Returns NaN if only one class present."""
    labels_np = labels.cpu().numpy()
    if len(set(labels_np.tolist())) < 2:
        return float("nan")
    probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    return float(roc_auc_score(labels_np, probs))


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
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["step"]


# ─────────────────────────────────────────────────────────────────────────────
#  EVALUATION — runs model over a DataLoader, returns logits + labels
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
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Mamba Cancer Gene Classifier")

    # Data — chromosome-disjoint splits (no leakage)
    p.add_argument("--splits_dir", type=str,
                   default="/content/drive/MyDrive/Mamba-DNA-1/dataset/splits",
                   help="Directory containing train.csv / val.csv / test.csv")
    p.add_argument("--seq_len",     type=int, default=512)
    p.add_argument("--num_classes", type=int, default=2)

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
def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  Mamba Cancer Gene Detector (chromosome-disjoint splits)")
    print(f"  Device  : {device}")
    if device.type == "cuda":
        print(f"  GPU     : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM    : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"  d_model : {args.d_model}   n_layer : {args.n_layer}")
    print(f"  seq_len : {args.seq_len}   classes : {args.num_classes}")
    print("=" * 60)

    # ── DNA tokenizer ─────────────────────────────────────────────────────────
    tokenizer = DNATokenizer()
    print(f"DNA vocabulary size: {tokenizer.vocab_size}")

    # ── Chromosome-disjoint datasets ──────────────────────────────────────────
    splits_dir = args.splits_dir
    print(f"\nLoading splits from: {splits_dir}")
    train_csv = os.path.join(splits_dir, "train.csv")
    val_csv   = os.path.join(splits_dir, "val.csv")
    test_csv  = os.path.join(splits_dir, "test.csv")
    for path in (train_csv, val_csv, test_csv):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing split file: {path}")

    train_ds = CancerGeneDataset(train_csv, tokenizer, args.seq_len)
    val_ds   = CancerGeneDataset(val_csv,   tokenizer, args.seq_len)
    test_ds  = CancerGeneDataset(test_csv,  tokenizer, args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  drop_last=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, drop_last=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, drop_last=False, num_workers=2)

    print(f"  steps/epoch : {len(train_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\nBuilding model...")
    config = MambaConfig(
        d_model          = args.d_model,
        n_layer          = args.n_layer,
        d_intermediate   = args.d_intermediate,
        vocab_size       = tokenizer.vocab_size,
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
    scaler  = torch.amp.GradScaler('cuda')
    loss_fn = nn.CrossEntropyLoss()

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
    best_val_auc = [0.0]

    for epoch in range(99999):
        for input_ids, labels in train_loader:
            if global_step >= args.max_steps:
                break

            input_ids = input_ids.to(device)
            labels    = labels.to(device)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = model(input_ids)
                loss   = loss_fn(logits, labels) / args.grad_accum

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
                v_logits, v_labels, v_loss = evaluate(model, val_loader, device, loss_fn)
                v_auc   = compute_auc(v_logits, v_labels)
                v_preds = v_logits.argmax(-1)
                vm      = compute_metrics(v_preds, v_labels)
                print(
                    f"\n  [VAL] loss {v_loss:.4f} | auc {v_auc:.4f} | "
                    f"acc {vm['acc']:.3f} | f1 {vm['f1']:.3f} | "
                    f"prec {vm['precision']:.3f} | rec {vm['recall']:.3f}"
                )

                save_checkpoint(
                    path      = f"{args.save_dir}/step_{global_step}.pt",
                    step      = global_step,
                    model     = model,
                    optimizer = optimizer,
                    scheduler = scheduler,
                    config    = config,
                )

                if v_auc > best_val_auc[0]:
                    best_val_auc[0] = v_auc
                    save_checkpoint(
                        path      = f"{args.save_dir}/best_val.pt",
                        step      = global_step,
                        model     = model,
                        optimizer = optimizer,
                        scheduler = scheduler,
                        config    = config,
                    )
                    print(f"  [BEST VAL AUC] {v_auc:.4f} at step {global_step}\n")
                else:
                    print()

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

    # ── Final test evaluation on held-out chromosomes ─────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL TEST EVALUATION (chr19 + chr22, held out)")
    print("=" * 60)

    best_path = f"{args.save_dir}/best_val.pt"
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"  Loaded best_val.pt from step {ckpt['step']} "
              f"(val_auc={best_val_auc[0]:.4f})")
    else:
        print("  WARNING: best_val.pt not found, using final model state")

    t_logits, t_labels, _ = evaluate(model, test_loader, device, loss_fn=None)
    test_auc   = compute_auc(t_logits, t_labels)
    test_preds = t_logits.argmax(-1)
    tm         = compute_metrics(test_preds, t_labels)

    print(f"\n  Overall test AUC : {test_auc:.4f}")
    print(f"  Acc  : {tm['acc']:.4f}   F1   : {tm['f1']:.4f}")
    print(f"  Prec : {tm['precision']:.4f}   Rec  : {tm['recall']:.4f}")

    # Per-chromosome breakdown — paper defense against chromosome shortcut
    print("\n  Per-chromosome test AUC:")
    test_chroms_list = []
    with open(test_csv, newline="") as f:
        reader = csv.DictReader(f)
        chrom_col = "chromosome" if "chromosome" in reader.fieldnames else "chrom"
        for row in reader:
            test_chroms_list.append(row[chrom_col])

    test_chroms_arr  = np.array(test_chroms_list)
    test_probs_arr   = torch.softmax(t_logits, dim=-1)[:, 1].numpy()
    test_labels_arr  = t_labels.numpy()

    for chrom in sorted(set(test_chroms_arr.tolist())):
        mask = test_chroms_arr == chrom
        if mask.sum() > 20 and len(set(test_labels_arr[mask].tolist())) > 1:
            chrom_auc = roc_auc_score(test_labels_arr[mask], test_probs_arr[mask])
            cf        = test_labels_arr[mask].mean()
            print(f"    {chrom}: AUC={chrom_auc:.4f}  n={int(mask.sum()):>6,}  "
                  f"cancer_frac={cf:.3f}")

    print("\n" + "=" * 60)
    print("  Similar AUC per chromosome confirms the model is not exploiting")
    print("  chromosome identity as a shortcut.")
    print("=" * 60)


if __name__ == "__main__":
    main()
