# RSNA Intracranial Hemorrhage Detection — Solo Capstone Pipeline

End-to-end, single-GPU-friendly pipeline: curated subset -> triple-window
preprocessing -> ConvNeXt-Tiny classifier -> Grad-CAM -> Streamlit demo.

## 0. Setup

```bash
pip install -r requirements.txt

pip install kaggle
kaggle competitions download -c rsna-intracranial-hemorrhage-detection
mkdir -p rsna_raw
unzip rsna-intracranial-hemorrhage-detection.zip -d rsna_raw
# Confirm you end up with:
#   rsna_raw/stage_2_train.csv
#   rsna_raw/stage_2_train/*.dcm
```

## 1. Build the balanced subset

```bash
python 01_data_curation.py
```

Produces `rsna_subset/subset_labels.csv` + `rsna_subset/dicoms/` — a
self-contained ~25-30k slice subset (~45% positive) with every epidural
case kept, other subtypes balanced, and every file validated as a readable
DICOM. This is the only step that touches the full raw dataset, so you can
delete `rsna_raw/` afterward to save disk space if needed.

**Tune these constants at the top of the script** for your disk/VRAM budget:
- `TARGET_TOTAL` — total slice count (default 28,000)
- `POSITIVE_FRACTION` — how hemorrhage-heavy the subset is (default 0.45)
- `EPIDURAL_CAP` — cap on rare epidural slices kept (default 3,200)

## 2. Sanity-check preprocessing (optional)

```bash
python 02_preprocessing.py rsna_subset/dicoms/ID_xxxxxxxxx.dcm
```

Writes `preview_triple_window.png` so you can visually confirm the
brain/blood/bone windows look correct before spending GPU hours training.

## 3. Train

```bash
python 03_dataset_training.py
```

- Backbone: `convnext_tiny` (swap to `efficientnet_b0` in the `BACKBONE`
  constant if you're VRAM/time constrained — both are single-GPU friendly).
- Mixed precision (`torch.autocast` + `GradScaler`), AdamW + cosine LR.
- Official RSNA weighted BCE loss (`any` weighted 2x, others 1x, normalized
  by total weight).
- Early stopping on validation loss (patience=4 by default).
- Best checkpoint saved to `./best_model.pt`.

On a single mid-range GPU (e.g. RTX 3060/4060, or a Colab/Kaggle T4),
expect roughly 3-8 minutes/epoch at 256px with batch size 32 depending on
hardware — adjust `BATCH_SIZE`/`IMAGE_SIZE` down if you hit OOM.

## 4. Explainability (Grad-CAM)

```bash
python 04_gradcam.py rsna_subset/dicoms/ID_xxxxxxxxx.dcm --class_idx 4
```

`--class_idx` indexes into `LABEL_COLS` =
`[epidural, intraparenchymal, intraventricular, subarachnoid, subdural, any]`
(so `4` = subdural, `5` = any). Saves `gradcam_overlay.png` and prints all
6 class probabilities.

## 5. Evaluate (metrics + plots for the defense)

```bash
python 05_evaluate.py
```

Scores the model on the exact patient-grouped held-out validation set
saved by step 3 (`rsna_subset/val_split.csv`) and writes to
`evaluation_outputs/`:
- `metrics.json` — official weighted multi-label log loss + per-class AUROC/AUPRC
- `roc_curves.png`, `pr_curves.png` — overlaid curves for all 6 labels
- `confusion_matrices.png` — one 2x2 matrix per label, using each label's
  optimal threshold (not a flat 0.5)
- `optimal_thresholds.json` — per-class cutoffs, computed via F-beta
  (beta=2, biased toward recall — missing a hemorrhage is worse than a
  false alarm). `app.py` picks this file up automatically if present.

## 6. Deployment demo

```bash
streamlit run app.py
```

Upload any `.dcm` from your subset (or a plain PNG for a quick UI test).
The app runs the identical preprocessing used in training, shows all 6
label probabilities as progress bars (flagged Positive/Negative using the
calibrated thresholds from step 5 if available), and displays the original
brain-window slice next to the Grad-CAM overlay for the strongest
predicted subtype.

## Design notes / why these choices

- **Triple windowing (brain/blood/bone) as RGB** is a well-established
  trick in RSNA-ICH-winning solutions: it lets a standard 3-channel
  ImageNet-pretrained CNN "see" three diagnostically relevant HU ranges at
  once, instead of forcing the network to learn windowing from raw HU.
- **2D slice-level classification, not 3D**, is the right complexity level
  for a solo project: it trains fast on one GPU, avoids the volumetric
  data-loading/memory complexity of 3D CNNs, and is exactly how most
  competitive Kaggle solutions for this dataset were built.
- **ConvNeXt-Tiny / EfficientNet-B0** are chosen over deeper backbones
  (ResNet-101, EfficientNet-B5+) specifically for single-GPU training
  speed — you can iterate in hours, not days.
- **Horizontal-flip-only augmentation** preserves anatomical validity;
  vertical flips or large rotations would create anatomically impossible
  training images and hurt the model.

## Methodology notes for the defense

- **Patient-level split, not slice-level.** A single CT study has 30-60
  slices of one patient. `01_data_curation.py` extracts `PatientID` from
  every DICOM header; `03_dataset_training.py` splits train/val with
  `GroupKFold` on `patient_id`, so no patient's slices appear in both sets.
  The script asserts zero patient overlap before training starts, and
  `05_evaluate.py` always scores the saved, patient-disjoint `val_split.csv`
  — not a re-derived split. This is the single most important claim to be
  able to state and defend clearly: validation AUROC reflects generalization
  to unseen patients, not memorized anatomy from an already-seen skull.
- **Threshold calibration, not a flat 0.5.** `05_evaluate.py` computes a
  per-class optimal cutoff via F-beta (beta=2 by default), which weights
  recall twice as heavily as precision — appropriate for an emergency
  triage task where a missed hemorrhage (false negative) is clinically far
  costlier than a false alarm. Youden's J is also implemented
  (`THRESHOLD_METHOD = "youden"`) if you want to compare the two in your
  write-up.

## Known dataset quirks handled by this pipeline

- Duplicate rows in `stage_2_train.csv` — deduplicated.
- A few known-corrupt DICOM IDs — dropped, plus a full readability
  validation pass over every sampled file (`01_data_curation.py`).
- `MONOCHROME1` inverted photometric interpretation — corrected in HU
  conversion.
- Missing `RescaleSlope`/`RescaleIntercept` tags — defaults applied.
- Extreme padding HU values (e.g. -2000 borders) — clamped before windowing.

## Disclaimer

This is a research/educational classifier for a graduation project, not a
validated clinical device. Do not present its output as a diagnostic tool.
