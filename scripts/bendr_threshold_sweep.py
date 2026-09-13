"""
Threshold sweep for the frozen-backbone BENDR checkpoint, matching the
same sweep methodology used for the Phase 2 GBM baseline.
"""
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score, precision_score, confusion_matrix

sys.path.insert(0, "scripts")
from bendr_loader import build_bendr_datasets
from bendr_utils import load_pretrained_interpolated_bendr
from chbmit_chs_info import get_chbmit_chs_info

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_SEC = 2  # Phase 1 window length, for false-alarms-per-hour calc

CHECKPOINT_PATH = "bendr_checkpoints_frozen/checkpoint-9.pth"  # any epoch works, all near-identical


def main():
    _, test_dataset = build_bendr_datasets()
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    chs_info = get_chbmit_chs_info()
    model = load_pretrained_interpolated_bendr(chs_info, n_outputs=2)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(DEVICE)
            with torch.cuda.amp.autocast():
                logits = model(X)[:, 1]
            probs = torch.sigmoid(logits).float().cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y.numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    total_hours = (len(all_labels) * WINDOW_SEC) / 3600

    print(f"{'Threshold':>10} {'Sensitivity':>12} {'Precision':>10} {'FA/hour':>10} {'TP':>6} {'FP':>6} {'FN':>6}")
    for t in np.arange(0.05, 0.95, 0.05):
        preds = (all_probs >= t).astype(int)
        sens = recall_score(all_labels, preds, zero_division=0)
        prec = precision_score(all_labels, preds, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()
        fa_per_hour = fp / total_hours
        print(f"{t:>10.2f} {sens:>12.4f} {prec:>10.4f} {fa_per_hour:>10.2f} {tp:>6} {fp:>6} {fn:>6}")


if __name__ == "__main__":
    main()