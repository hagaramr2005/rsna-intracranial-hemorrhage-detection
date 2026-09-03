"""
01_data_curation.py
====================
Builds a class-balanced ~25,000-30,000 slice subset from the full RSNA
Intracranial Hemorrhage Detection training set, so a solo student can train
on a single GPU without exhausting disk or VRAM.

Assumes you have already run:
    pip install kaggle pydicom pandas tqdm
    kaggle competitions download -c rsna-intracranial-hemorrhage-detection
    unzip rsna-intracranial-hemorrhage-detection.zip -d ./rsna_raw

Expected raw layout (stage_2 naming, this is the final release of the comp):
    ./rsna_raw/stage_2_train.csv
    ./rsna_raw/stage_2_train/           <- folder of .dcm files
"""

import os
import shutil
import random
import logging
from pathlib import Path

import pandas as pd
import pydicom
from tqdm import tqdm

# --------------------------------------------------------------------------
# CONFIG — edit these paths/knobs for your machine
# --------------------------------------------------------------------------
RAW_ROOT = Path("./rsna_raw")
RAW_CSV = RAW_ROOT / "stage_2_train.csv"
RAW_DICOM_DIR = RAW_ROOT / "stage_2_train"

OUTPUT_DIR = Path("./rsna_subset")
OUTPUT_DICOM_DIR = OUTPUT_DIR / "dicoms"
OUTPUT_CSV = OUTPUT_DIR / "subset_labels.csv"

LABEL_COLS = [
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
    "any",
]

# Total slices you want to end up with. 25-30k is the sweet spot for a
# single consumer GPU (e.g. RTX 3060/4060/T4) at 256-384px with a
# convnext_tiny / efficientnet_b0 backbone.
TARGET_TOTAL = 28000

# Every epidural-positive slice is rare (~0.4% of the full dataset) — we
# keep ALL of them (capped) so the model actually sees enough examples of
# the hardest minority class instead of losing it during random sampling.
EPIDURAL_CAP = 3200

# Fraction of the final subset that should be hemorrhage-positive (any=1).
# The real dataset is ~14% positive; we oversample positives heavily so
# the model isn't drowned in negatives, while still keeping some negatives
# for specificity.
POSITIVE_FRACTION = 0.45

RANDOM_SEED = 42

# A handful of IDs that are publicly known in Kaggle discussions to be
# corrupt / zero-byte / unreadable in the RSNA release. We drop them
# proactively, and ALSO validate every sampled file by opening it, so any
# other corrupt file is caught automatically below.
KNOWN_BAD_IDS = {
    "ID_6431af929",
    "ID_9c0cf5bfb",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)


def load_and_pivot_labels(csv_path: Path) -> pd.DataFrame:
    """
    The raw stage_2_train.csv is LONG format:
        ID                              Label
        ID_xxxxxxxxx_epidural           0
        ID_xxxxxxxxx_intraparenchymal   0
        ...
        ID_xxxxxxxxx_any                1

    We pivot it to WIDE format: one row per slice, one column per subtype.
    """
    log.info("Loading raw label CSV (this file is large, ~2-3M rows)...")
    df = pd.read_csv(csv_path)

    # Known duplicate-row issue in the RSNA release: a very small number of
    # (ID, sub_label) pairs appear twice with identical values. Drop exact
    # duplicates first, keep the first occurrence for any residual clashes.
    df = df.drop_duplicates()

    # Split "ID_xxxxxxxxx_epidural" -> image_id="ID_xxxxxxxxx", subtype="epidural"
    split = df["ID"].str.rsplit("_", n=1, expand=True)
    df["image_id"] = split[0]
    df["subtype"] = split[1]

    wide = df.pivot_table(
        index="image_id", columns="subtype", values="Label", aggfunc="first"
    ).reset_index()

    # Ensure all expected columns exist even if pivot dropped an empty one
    for col in LABEL_COLS:
        if col not in wide.columns:
            wide[col] = 0

    wide[LABEL_COLS] = wide[LABEL_COLS].fillna(0).astype(int)
    wide = wide[["image_id"] + LABEL_COLS]

    # Drop known corrupt IDs
    before = len(wide)
    wide = wide[~wide["image_id"].isin(KNOWN_BAD_IDS)]
    log.info("Dropped %d known-bad IDs.", before - len(wide))

    log.info("Pivoted to %d unique slices.", len(wide))
    return wide


def dicom_path_for(image_id: str) -> Path:
    return RAW_DICOM_DIR / f"{image_id}.dcm"


def read_valid_with_patient_id(path: Path):
    """
    Validation + leakage-prevention in one pass: confirms the DICOM opens
    and decodes cleanly, AND extracts PatientID (tag 0010,0020).

    CRITICAL: PatientID is what lets us do a patient-level train/val split
    later. A single CT study contains 30-60 slices of the SAME patient —
    splitting by slice (SOPInstanceUID) instead of by patient would let
    near-duplicate slices of one skull appear in both train and val,
    inflating validation metrics with memorized anatomy rather than
    genuine generalization. We refuse to lose this field.

    Returns (is_valid: bool, patient_id: str | None).
    """
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=False, force=True)
        _ = ds.pixel_array  # forces the decode, catches truncated/corrupt files
        patient_id = str(getattr(ds, "PatientID", "")).strip()
        if not patient_id:
            # No PatientID tag present — cannot safely group this slice,
            # so treat it as invalid rather than silently risking leakage.
            return False, None
        return True, patient_id
    except Exception:
        return False, None


def build_balanced_subset(df: pd.DataFrame) -> pd.DataFrame:
    random.seed(RANDOM_SEED)

    df = df[df["image_id"].apply(lambda i: dicom_path_for(i).exists())].copy()
    log.info("%d slices have a matching DICOM file on disk.", len(df))

    n_positive_target = int(TARGET_TOTAL * POSITIVE_FRACTION)
    n_negative_target = TARGET_TOTAL - n_positive_target

    # --- Step 1: guarantee epidural coverage (rarest class) ---
    epidural_pool = df[df["epidural"] == 1]
    epidural_take = epidural_pool.sample(
        n=min(EPIDURAL_CAP, len(epidural_pool)), random_state=RANDOM_SEED
    )
    log.info("Kept %d / %d epidural-positive slices.", len(epidural_take), len(epidural_pool))

    chosen_ids = set(epidural_take["image_id"])
    remaining_pos_budget = max(0, n_positive_target - len(epidural_take))

    # --- Step 2: fill remaining positive budget, balanced across the
    # other 4 subtypes (intraparenchymal, intraventricular, subarachnoid,
    # subdural) so no single common subtype (e.g. subdural) dominates.
    other_subtypes = ["intraparenchymal", "intraventricular", "subarachnoid", "subdural"]
    per_class_budget = remaining_pos_budget // len(other_subtypes)

    for subtype in other_subtypes:
        pool = df[(df[subtype] == 1) & (~df["image_id"].isin(chosen_ids))]
        take_n = min(per_class_budget, len(pool))
        taken = pool.sample(n=take_n, random_state=RANDOM_SEED)
        chosen_ids.update(taken["image_id"])
        log.info("Kept %d / %d %s-positive slices.", take_n, len(pool), subtype)

    # If we're still short of the positive target (some subtype pools were
    # smaller than their budget), top up from any remaining any=1 slices.
    positives_so_far = df[df["image_id"].isin(chosen_ids)]
    shortfall = n_positive_target - len(positives_so_far)
    if shortfall > 0:
        pool = df[(df["any"] == 1) & (~df["image_id"].isin(chosen_ids))]
        take_n = min(shortfall, len(pool))
        taken = pool.sample(n=take_n, random_state=RANDOM_SEED)
        chosen_ids.update(taken["image_id"])
        log.info("Topped up %d extra positive slices to hit target.", take_n)

    # --- Step 3: fill negative budget with any=0 slices, sampled randomly ---
    negative_pool = df[(df["any"] == 0) & (~df["image_id"].isin(chosen_ids))]
    negative_take = negative_pool.sample(
        n=min(n_negative_target, len(negative_pool)), random_state=RANDOM_SEED
    )
    chosen_ids.update(negative_take["image_id"])
    log.info("Kept %d negative slices.", len(negative_take))

    subset = df[df["image_id"].isin(chosen_ids)].reset_index(drop=True)
    log.info("Raw subset size before file validation: %d", len(subset))
    return subset


def validate_and_copy(subset: pd.DataFrame) -> pd.DataFrame:
    """
    Opens every sampled DICOM to confirm it decodes cleanly AND to extract
    PatientID (needed downstream for a leakage-free, patient-grouped
    train/val split), then copies valid files into OUTPUT_DICOM_DIR so the
    subset is fully self-contained (you can zip OUTPUT_DIR and move it to
    a Colab/Kaggle GPU instance).
    """
    OUTPUT_DICOM_DIR.mkdir(parents=True, exist_ok=True)
    keep_rows = []
    patient_id_map = {}

    for row in tqdm(subset.itertuples(), total=len(subset), desc="Validating + copying DICOMs"):
        src = dicom_path_for(row.image_id)
        is_valid, patient_id = read_valid_with_patient_id(src)
        if not is_valid:
            continue
        dst = OUTPUT_DICOM_DIR / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        keep_rows.append(row.image_id)
        patient_id_map[row.image_id] = patient_id

    final = subset[subset["image_id"].isin(keep_rows)].reset_index(drop=True)
    final["patient_id"] = final["image_id"].map(patient_id_map)

    n_patients = final["patient_id"].nunique()
    log.info("Final validated subset: %d slices (dropped %d unreadable/missing-PatientID).",
              len(final), len(subset) - len(final))
    log.info("Subset spans %d unique patients (avg %.1f slices/patient).",
              n_patients, len(final) / max(n_patients, 1))
    return final


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    labels_wide = load_and_pivot_labels(RAW_CSV)
    subset = build_balanced_subset(labels_wide)
    final = validate_and_copy(subset)

    final["dicom_path"] = final["image_id"].apply(
        lambda i: str(OUTPUT_DICOM_DIR / f"{i}.dcm")
    )
    final.to_csv(OUTPUT_CSV, index=False)

    log.info("=" * 60)
    log.info("DONE. Subset saved to: %s", OUTPUT_CSV)
    log.info("Total slices: %d", len(final))
    log.info("Class balance:\n%s", final[LABEL_COLS].sum().to_string())
    log.info("Positive rate (any=1): %.2f%%", 100 * final["any"].mean())
    log.info("=" * 60)


if __name__ == "__main__":
    main()
