"""
BENDR fine-tuning on CHB-MIT, using InterpolatedBENDR with Phase 1's
windowed data (18 channels, 512 samples @ 256Hz).

Applies lessons from LaBraM fine-tuning: gradient clipping from the start,
conservative learning rate, class-weighted loss for imbalance, checkpoint
saving per epoch so peak-vs-overfit can be identified after the fact.
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score

sys.path.insert(0, "scripts")
from bendr_loader import build_bendr_datasets
from bendr_utils import load_pretrained_interpolated_bendr
from chbmit_chs_info import get_chbmit_chs_info

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "bendr_checkpoints"
LOG_PATH = "bendr_checkpoints/log.txt"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def evaluate(model, loader, criterion):
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)
            y = y.float().to(DEVICE)

            logits = model(X)[:, 1]  # positive class logit
            loss = criterion(logits, y)
            total_loss += loss.item()
            n_batches += 1

            all_logits.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(y.cpu().numpy())

    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)

    preds = (all_logits >= 0.5).astype(int)
    roc_auc = roc_auc_score(all_labels, all_logits)
    pr_auc = average_precision_score(all_labels, all_logits)
    bal_acc = balanced_accuracy_score(all_labels, preds)

    return {
        "loss": total_loss / n_batches,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "balanced_accuracy": bal_acc,
    }


def main():
    print(f"Using device: {DEVICE}")

    train_dataset, test_dataset = build_bendr_datasets()
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    chs_info = get_chbmit_chs_info()
    model = load_pretrained_interpolated_bendr(chs_info, n_outputs=2)
    model.to(DEVICE)

    # Class-weighted loss for imbalance, same principle as Phase 2's GBM
    train_labels = np.array([train_dataset[i][1] for i in range(0, len(train_dataset), 100)])
    pos_weight_value = (train_labels == 0).sum() / max((train_labels == 1).sum(), 1)
    pos_weight_value = min(pos_weight_value, 10.0)  # cap - 193x was too aggressive
    pos_weight = torch.tensor([pos_weight_value], device=DEVICE)
    print(f"Capped pos_weight: {pos_weight_value:.1f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.05)

    n_epochs = 10
    logs = []

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for step, (X, y) in enumerate(train_loader):
            X = X.to(DEVICE)
            y = y.float().to(DEVICE)

            optimizer.zero_grad()
            logits = model(X)[:, 1]
            loss = criterion(logits, y)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if not torch.isfinite(loss):
                print(f"NaN/Inf loss detected at epoch {epoch} step {step} — stopping.")
                raise RuntimeError("Training diverged")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if step % 500 == 0:
                print(f"Epoch {epoch} step {step}/{len(train_loader)} loss {loss.item():.4f}")

        train_loss = total_loss / n_batches
        eval_stats = evaluate(model, test_loader, criterion)

        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"test_{k}": v for k, v in eval_stats.items()},
        }
        logs.append(log_entry)
        print(json.dumps(log_entry))

        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/checkpoint-{epoch}.pth")

    print("Training complete.")


if __name__ == "__main__":
    main()


#     Automatic origin fit: head of radius 90.4 mm
# Creating RawArray with float64 data, n_channels=20, n_times=18
#     Range : 0 ... 17 =      0.000 ...     0.170 secs
# Ready.
# InterpolatedBENDR loaded cleanly: 157142075 params
# Estimated pos_weight: 193.5
# Epoch 0 step 0/22488 loss 2.8797