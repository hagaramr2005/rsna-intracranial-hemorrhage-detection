from __future__ import annotations
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent

RAW_ROOT = PROJECT_ROOT / "rsna_raw"
RAW_CSV = RAW_ROOT / "stage_2_train.csv"
RAW_DICOM_DIR = RAW_ROOT / "stage_2_train"

SUBSET_DIR = PROJECT_ROOT / "rsna_subset"
SUBSET_DICOM_DIR = SUBSET_DIR / "dicoms"
SUBSET_CSV = SUBSET_DIR / "subset_labels.csv"
TRAIN_SPLIT_CSV = SUBSET_DIR / "train_split.csv"
VAL_SPLIT_CSV = SUBSET_DIR / "val_split.csv"

CHECKPOINT_PATH = PROJECT_ROOT / "best_model.pt"

EVAL_OUTPUT_DIR = PROJECT_ROOT / "evaluation_outputs"
THRESHOLDS_PATH = EVAL_OUTPUT_DIR / "optimal_thresholds.json"
METRICS_PATH = EVAL_OUTPUT_DIR / "metrics.json"

LABEL_COLS: list[str] = [
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
    "any",
]
ANY_INDEX = LABEL_COLS.index("any")

RSNA_CLASS_WEIGHTS = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 2.0])

BACKBONE = "convnext_tiny"
NUM_CLASSES = len(LABEL_COLS)
IMAGE_SIZE = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

WINDOW_PRESETS: dict[str, tuple[float, float]] = {
    "Brain Standard (W:80, L:40)": (40, 80),
    "Subdural (W:130, L:75)": (75, 130),
    "Bone (W:2500, L:500)": (500, 2500),
    "Stroke/Ischemia (W:40, L:40)": (40, 40),
}

RANDOM_SEED = 42
