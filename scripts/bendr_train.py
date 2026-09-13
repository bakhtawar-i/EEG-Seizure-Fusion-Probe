"""
BENDR fine-tuning on CHB-MIT — frozen backbone, train only final_layer.
This follows braindecode's documented transfer-learning pattern (freeze
pretrained encoder/contextualizer, train only the head) after full
fine-tuning proved unstable (NaN divergence even with gradient clipping
and AMP/GradScaler).
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
CHECKPOINT_DIR = "bendr_checkpoints_frozen"
LOG_PATH = f"{CHECKPOINT_DIR}/log.txt"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

N_EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-3
POS_WEIGHT_CAP = 10.0
CLIP_NORM = 1.0


def evaluate(model, loader, criterion):
    model.eval()
    all_probs = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)
            y = y.float().to(DEVICE)

            with torch.cuda.amp.autocast():
                logits = model(X)[:, 1]
                loss = criterion(logits, y)

            total_loss += loss.item()
            n_batches += 1

            all_probs.append(torch.sigmoid(logits).float().cpu().numpy())
            all_labels.append(y.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    preds = (all_probs >= 0.5).astype(int)

    return {
        "loss": total_loss / n_batches,
        "roc_auc": roc_auc_score(all_labels, all_probs),
        "pr_auc": average_precision_score(all_labels, all_probs),
        "balanced_accuracy": balanced_accuracy_score(all_labels, preds),
    }


def main():
    print(f"Using device: {DEVICE}")

    train_dataset, test_dataset = build_bendr_datasets()
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    chs_info = get_chbmit_chs_info()
    model = load_pretrained_interpolated_bendr(chs_info, n_outputs=2)
    model.to(DEVICE)

    # Freeze everything except the classification head, per braindecode's
    # documented transfer-learning pattern for foundation models
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("final_layer")

    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_names}")
    print(f"Trainable: {trainable_count} / Total: {total_count}")

    train_labels = np.array([train_dataset[i][1] for i in range(0, len(train_dataset), 100)])
    pos_weight_value = (train_labels == 0).sum() / max((train_labels == 1).sum(), 1)
    pos_weight_value = min(pos_weight_value, POS_WEIGHT_CAP)
    pos_weight = torch.tensor([pos_weight_value], device=DEVICE)
    print(f"Capped pos_weight: {pos_weight_value:.1f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=0.05,
    )
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(N_EPOCHS):
        model.train()
        total_loss = 0.0
        n_batches = 0
        skipped_steps = 0

        for step, (X, y) in enumerate(train_loader):
            X = X.to(DEVICE)
            y = y.float().to(DEVICE)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(X)[:, 1]
                loss = criterion(logits, y)

            if not torch.isfinite(loss):
                skipped_steps += 1
                if step % 500 == 0:
                    print(f"Epoch {epoch} step {step}: non-finite loss, skipping this step")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=CLIP_NORM
            )
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1

            if step % 500 == 0:
                print(f"Epoch {epoch} step {step}/{len(train_loader)} loss {loss.item():.4f}")

        train_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch} complete. Skipped steps: {skipped_steps}/{len(train_loader)}")

        eval_stats = evaluate(model, test_loader, criterion)

        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "skipped_steps": skipped_steps,
            **{f"test_{k}": v for k, v in eval_stats.items()},
        }
        print(json.dumps(log_entry))

        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/checkpoint-{epoch}.pth")

    print("Training complete.")


if __name__ == "__main__":
    main()