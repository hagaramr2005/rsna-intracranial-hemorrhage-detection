"""
03_dataset_training.py
=======================
PyTorch Dataset/DataLoader, model, official RSNA weighted BCE loss, and a
mixed-precision train/val loop with early stopping.

Run:
    python 03_dataset_training.py

Expects:
    ./rsna_subset/subset_labels.csv   (from 01_data_curation.py)
    columns: image_id, epidural, intraparenchymal, intraventricular,
             subarachnoid, subdural, any, dicom_path
"""

import os
import time
import logging
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
import albumentations as A
import timm

from importlib import import_module

# 02_preprocessing.py starts with a digit, so it can't be `import`ed with a
# normal dotted name — load it dynamically instead.
prep = import_module("02_preprocessing")
read_dicom_as_hu = prep.read_dicom_as_hu
triple_window_rgb = prep.triple_window_rgb
IMAGENET_MEAN = prep.IMAGENET_MEAN
IMAGENET_STD = prep.IMAGENET_STD

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
LABEL_COLS = [
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
    "any",
]

CSV_PATH = "./rsna_subset/subset_labels.csv"
TRAIN_SPLIT_PATH = "./rsna_subset/train_split.csv"
VAL_SPLIT_PATH = "./rsna_subset/val_split.csv"
N_GROUP_FOLDS = 5  # we train on 1 fold's worth of held-out patients, rest is train
IMAGE_SIZE = 256
BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCHS = 20
LR = 3e-4
WEIGHT_DECAY = 1e-4
BACKBONE = "convnext_tiny"  # swap to 'efficientnet_b0' if VRAM/time constrained
EARLY_STOPPING_PATIENCE = 4
CHECKPOINT_PATH = "./best_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# Official RSNA competition class weights: each of the 5 subtypes gets
# weight 1, "any" gets weight 2 (it's twice as important in the metric).
# Order MUST match LABEL_COLS.
RSNA_CLASS_WEIGHTS = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 2.0])


# --------------------------------------------------------------------------
# AUGMENTATIONS
# --------------------------------------------------------------------------
def build_transforms(image_size: int, train: bool) -> A.Compose:
    """
    Light augmentations that respect cranial anatomy:
      - Horizontal flip only (the skull/brain is roughly left-right
        symmetric — this is standard practice in RSNA-ICH pipelines).
      - NO vertical flip (would invert superior/inferior orientation,
        producing anatomically impossible images).
      - Small rotation (+/-10 deg) to simulate slight head tilt at scan time.
      - Mild brightness/contrast jitter to simulate scanner variability.
    Normalization (mean/std) is already baked into 02_preprocessing.py, so
    here we only operate on the resized uint8 RGB array BEFORE that step.
    """
    if train:
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, border_mode=0, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
                A.Resize(image_size, image_size),
            ]
        )
    return A.Compose([A.Resize(image_size, image_size)])


# --------------------------------------------------------------------------
# DATASET
# --------------------------------------------------------------------------
class RSNAHemorrhageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_size: int = 256, train: bool = True):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.transform = build_transforms(image_size, train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        try:
            hu = read_dicom_as_hu(row["dicom_path"])
            rgb_uint8 = triple_window_rgb(hu)  # (H, W, 3) uint8, native resolution
        except Exception as e:
            # Defensive fallback: if a file is somehow unreadable at train
            # time, return a blank (all-zero) image + zero labels so the
            # DataLoader doesn't crash mid-epoch. Log it so you can inspect.
            log.warning("Failed to read %s (%s) — using blank fallback.", row["dicom_path"], e)
            rgb_uint8 = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        augmented = self.transform(image=rgb_uint8)["image"]

        float_img = augmented.astype(np.float32) / 255.0
        float_img = (float_img - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(float_img.transpose(2, 0, 1)).float()

        labels = torch.tensor(row[LABEL_COLS].values.astype(np.float32))
        return tensor, labels


# --------------------------------------------------------------------------
# MODEL
# --------------------------------------------------------------------------
def build_model(backbone: str = BACKBONE, num_classes: int = len(LABEL_COLS)) -> nn.Module:
    model = timm.create_model(backbone, pretrained=True, num_classes=num_classes)
    return model


# --------------------------------------------------------------------------
# LOSS — official RSNA weighted multi-label BCE
# --------------------------------------------------------------------------
class RSNAWeightedBCELoss(nn.Module):
    """
    Weighted binary cross-entropy across the 6 labels, normalized by the
    sum of weights (matches the official Kaggle competition metric, which
    is a weighted average of per-class log loss with 'any' weighted 2x).
    """

    def __init__(self, class_weights: torch.Tensor = RSNA_CLASS_WEIGHTS):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        per_element_loss = self.bce(logits, targets)  # (B, 6)
        weighted = per_element_loss * self.class_weights.to(logits.device)
        # Normalize by sum of weights so the loss scale doesn't depend on
        # how many classes/batch size you have.
        return weighted.sum() / (self.class_weights.sum() * targets.size(0))


# --------------------------------------------------------------------------
# TRAIN / VAL LOOP
# --------------------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer, scaler, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    n_batches = 0

    torch.set_grad_enabled(train)
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                             dtype=torch.float16, enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, labels)

        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    log.info("Using device: %s", DEVICE)

    df = pd.read_csv(CSV_PATH)
    log.info("Loaded %d slices from %s", len(df), CSV_PATH)

    if "patient_id" not in df.columns:
        raise ValueError(
            "subset_labels.csv has no 'patient_id' column. Re-run "
            "01_data_curation.py (updated version) to regenerate the "
            "subset with PatientID included — training cannot proceed "
            "with a random slice-level split, since slices from the same "
            "patient's CT study would leak across train/val."
        )

    # --- Patient-level split (GroupKFold on patient_id) ---
    # A single CT study contains 30-60 slices of ONE skull. A naive random
    # or 'any'-stratified split (as an earlier draft of this script used)
    # can and will place different slices of the SAME patient into both
    # train and val. The model then partly memorizes that patient's skull
    # shape/anatomy rather than learning to generalize, producing
    # deceptively high validation AUROC that collapses on unseen patients.
    # GroupKFold guarantees every patient_id appears in exactly ONE of
    # train/val, never both.
    gkf = GroupKFold(n_splits=N_GROUP_FOLDS)
    train_idx, val_idx = next(gkf.split(df, groups=df["patient_id"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    overlap = set(train_df["patient_id"]) & set(val_df["patient_id"])
    assert len(overlap) == 0, f"Patient-level leakage detected: {len(overlap)} overlapping patients!"

    log.info(
        "Train: %d slices / %d patients | Val: %d slices / %d patients (0 overlapping patients)",
        len(train_df), train_df["patient_id"].nunique(),
        len(val_df), val_df["patient_id"].nunique(),
    )

    # Persist the exact split so 05_evaluate.py scores the SAME held-out
    # patients the model never saw during training, rather than re-deriving
    # a split that could accidentally diverge.
    train_df.to_csv(TRAIN_SPLIT_PATH, index=False)
    val_df.to_csv(VAL_SPLIT_PATH, index=False)

    train_ds = RSNAHemorrhageDataset(train_df, image_size=IMAGE_SIZE, train=True)
    val_ds = RSNAHemorrhageDataset(val_df, image_size=IMAGE_SIZE, train=False)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    model = build_model().to(DEVICE)
    criterion = RSNAWeightedBCELoss().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss = run_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE, train=True)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, scaler, DEVICE, train=False)

        scheduler.step()
        elapsed = time.time() - t0

        log.info(
            "Epoch %02d/%02d | train_loss=%.4f | val_loss=%.4f | lr=%.2e | %.1fs",
            epoch, EPOCHS, train_loss, val_loss, optimizer.param_groups[0]["lr"], elapsed,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            torch.save(
                {"model_state_dict": best_state, "val_loss": best_val_loss, "backbone": BACKBONE},
                CHECKPOINT_PATH,
            )
            log.info("  -> New best model saved to %s (val_loss=%.4f)", CHECKPOINT_PATH, best_val_loss)
        else:
            patience_counter += 1
            log.info("  -> No improvement (%d/%d)", patience_counter, EARLY_STOPPING_PATIENCE)
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                log.info("Early stopping triggered at epoch %d.", epoch)
                break

    log.info("Training complete. Best val_loss=%.4f. Checkpoint: %s", best_val_loss, CHECKPOINT_PATH)


if __name__ == "__main__":
    main()
