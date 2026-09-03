"""
05_evaluate.py
================
Post-training evaluation for the defense/documentation: computes the exact
numbers and plots a graduation committee will ask for.

Run AFTER 03_dataset_training.py has produced:
    ./best_model.pt
    ./rsna_subset/val_split.csv   (patient-grouped, held-out — see below)

    python 05_evaluate.py

Produces (in ./evaluation_outputs/):
    metrics.json            - weighted log loss + per-class AUROC/AUPRC
    roc_curves.png          - ROC curve per class, overlaid
    pr_curves.png           - Precision-Recall curve per class, overlaid
    confusion_matrices.png  - one 2x2 confusion matrix per class, using
                               the OPTIMAL threshold for that class
    optimal_thresholds.json - per-class cutoff (used by app.py if present)

Why this file matters for the defense:
  - The val set here (val_split.csv) was produced by 03_dataset_training.py
    using GroupKFold on `patient_id`, so every number below reflects
    generalization to PATIENTS the model never saw during training — not
    just unseen slices of an already-seen patient. This is the single most
    important methodological claim to be able to defend.
"""

import json
import logging
from pathlib import Path
from importlib import import_module

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    log_loss,
    multilabel_confusion_matrix,
)

train_mod = import_module("03_dataset_training")
LABEL_COLS = train_mod.LABEL_COLS
RSNA_CLASS_WEIGHTS = train_mod.RSNA_CLASS_WEIGHTS
RSNAHemorrhageDataset = train_mod.RSNAHemorrhageDataset
build_model = train_mod.build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
CHECKPOINT_PATH = "./best_model.pt"
VAL_SPLIT_PATH = "./rsna_subset/val_split.csv"
OUTPUT_DIR = Path("./evaluation_outputs")
IMAGE_SIZE = 256
BATCH_SIZE = 32
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Threshold-selection strategy. For an emergency-triage task like ICH
# detection, missing a real hemorrhage (false negative) is clinically far
# worse than a false alarm (false positive) — so we default to an F-beta
# score with beta=2, which weights recall twice as heavily as precision,
# rather than the beta=1 balance or a flat 0.5 cutoff.
THRESHOLD_METHOD = "fbeta"  # "fbeta" or "youden"
FBETA_BETA = 2.0


@torch.no_grad()
def run_inference(model, loader, device):
    all_probs, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())
    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)


def weighted_multilabel_log_loss(y_true: np.ndarray, y_prob: np.ndarray, weights: np.ndarray) -> float:
    """
    Official RSNA competition metric: per-class log loss, weighted average
    with 'any' at 2x weight, normalized by the sum of weights. Matches the
    training loss formula but computed on probabilities post-hoc (not
    logits), consistent with how the Kaggle leaderboard scored submissions.
    """
    eps = 1e-7
    y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
    per_class_losses = []
    for i in range(y_true.shape[1]):
        per_class_losses.append(log_loss(y_true[:, i], y_prob_clipped[:, i], labels=[0, 1]))
    per_class_losses = np.array(per_class_losses)
    weights = weights.numpy() if torch.is_tensor(weights) else np.array(weights)
    return float((per_class_losses * weights).sum() / weights.sum())


def compute_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray, method: str, beta: float = 2.0) -> float:
    """
    Returns the classification cutoff for one class that maximizes either:
      - Youden's J statistic (TPR - FPR), the classic ROC-based cutoff, or
      - F-beta score (beta>1 biases toward recall — appropriate here since
        missing a hemorrhage is far costlier than a false alarm).
    Falls back to 0.5 if the class has too few positives to fit a curve.
    """
    if y_true.sum() < 2 or y_true.sum() > len(y_true) - 2:
        return 0.5

    if method == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        return float(thresholds[best_idx])

    # F-beta sweep over the precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    precision, recall = precision[:-1], recall[:-1]  # align with `thresholds`
    denom = (beta ** 2 * precision) + recall
    fbeta = np.where(denom > 0, (1 + beta ** 2) * precision * recall / np.where(denom == 0, 1, denom), 0)
    best_idx = np.argmax(fbeta)
    return float(thresholds[best_idx])


def plot_roc_curves(y_true, y_prob, out_path: Path):
    plt.figure(figsize=(7, 7))
    for i, name in enumerate(LABEL_COLS):
        if y_true[:, i].sum() < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
        auc = roc_auc_score(y_true[:, i], y_prob[:, i])
        plt.plot(fpr, tpr, label=f"{name} (AUROC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — RSNA ICH Classifier (held-out patients)")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info("Saved %s", out_path)


def plot_pr_curves(y_true, y_prob, out_path: Path):
    plt.figure(figsize=(7, 7))
    for i, name in enumerate(LABEL_COLS):
        if y_true[:, i].sum() < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_prob[:, i])
        ap = average_precision_score(y_true[:, i], y_prob[:, i])
        plt.plot(recall, precision, label=f"{name} (AUPRC={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves — RSNA ICH Classifier (held-out patients)")
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info("Saved %s", out_path)


def plot_confusion_matrices(y_true, y_pred, out_path: Path):
    cms = multilabel_confusion_matrix(y_true, y_pred)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, (name, cm) in enumerate(zip(LABEL_COLS, cms)):
        ax = axes[i // 3, i % 3]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(name)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticklabels(["True 0", "True 1"])
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                         color="white" if cm[r, c] > cm.max() / 2 else "black")
    fig.suptitle("Multi-Label Confusion Matrices (per-class optimal thresholds)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info("Saved %s", out_path)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    val_path = Path(VAL_SPLIT_PATH)
    if not val_path.exists():
        raise FileNotFoundError(
            f"{VAL_SPLIT_PATH} not found. Run the updated 03_dataset_training.py "
            "first — it now saves the patient-grouped val split to this path "
            "so evaluation always scores the exact same held-out patients."
        )

    val_df = pd.read_csv(val_path)
    log.info("Evaluating on %d held-out slices / %d held-out patients.",
              len(val_df), val_df["patient_id"].nunique())

    val_ds = RSNAHemorrhageDataset(val_df, image_size=IMAGE_SIZE, train=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    model = build_model()
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()

    y_prob, y_true = run_inference(model, val_loader, DEVICE)

    # --- Official weighted log loss ---
    w_log_loss = weighted_multilabel_log_loss(y_true, y_prob, RSNA_CLASS_WEIGHTS)

    # --- Per-class AUROC / AUPRC ---
    per_class_metrics = {}
    for i, name in enumerate(LABEL_COLS):
        if y_true[:, i].sum() < 2:
            per_class_metrics[name] = {"AUROC": None, "AUPRC": None, "note": "too few positives"}
            continue
        auroc = roc_auc_score(y_true[:, i], y_prob[:, i])
        auprc = average_precision_score(y_true[:, i], y_prob[:, i])
        per_class_metrics[name] = {"AUROC": round(float(auroc), 4), "AUPRC": round(float(auprc), 4)}

    # --- Optimal per-class thresholds ---
    thresholds = {}
    for i, name in enumerate(LABEL_COLS):
        t = compute_optimal_threshold(y_true[:, i], y_prob[:, i], method=THRESHOLD_METHOD, beta=FBETA_BETA)
        thresholds[name] = round(t, 4)

    y_pred = np.zeros_like(y_prob)
    for i, name in enumerate(LABEL_COLS):
        y_pred[:, i] = (y_prob[:, i] >= thresholds[name]).astype(int)

    # --- Save everything ---
    metrics_summary = {
        "n_val_slices": len(val_df),
        "n_val_patients": int(val_df["patient_id"].nunique()),
        "weighted_log_loss": round(w_log_loss, 4),
        "threshold_method": f"{THRESHOLD_METHOD}" + (f" (beta={FBETA_BETA})" if THRESHOLD_METHOD == "fbeta" else ""),
        "per_class": per_class_metrics,
    }
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)
    with open(OUTPUT_DIR / "optimal_thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)

    plot_roc_curves(y_true, y_prob, OUTPUT_DIR / "roc_curves.png")
    plot_pr_curves(y_true, y_prob, OUTPUT_DIR / "pr_curves.png")
    plot_confusion_matrices(y_true, y_pred, OUTPUT_DIR / "confusion_matrices.png")

    log.info("=" * 60)
    log.info("Weighted multi-label log loss: %.4f", w_log_loss)
    for name, m in per_class_metrics.items():
        log.info("  %-20s AUROC=%s  AUPRC=%s  threshold=%.3f",
                  name, m.get("AUROC"), m.get("AUPRC"), thresholds[name])
    log.info("All outputs saved to: %s", OUTPUT_DIR.resolve())
    log.info("=" * 60)


if __name__ == "__main__":
    main()
