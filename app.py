"""
app.py — NeuroScan AI Streamlit Suite
"""
from __future__ import annotations

import io
import os
import tempfile
import json
from datetime import datetime, timezone

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import timm
import torch
import torch.nn as nn
from fpdf import FPDF
from PIL import Image

from config import (
    ANY_INDEX,
    CHECKPOINT_PATH,
    DEVICE,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    LABEL_COLS as SUBTYPES,
    RANDOM_SEED,
    THRESHOLDS_PATH,
    WINDOW_PRESETS,
)

try:
    from google import genai
except ImportError:
    genai = None

try:
    import pydicom
except ImportError:
    pydicom = None

# Reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True

MEAN = np.array(IMAGENET_MEAN, dtype=np.float32)
STD = np.array(IMAGENET_STD, dtype=np.float32)

def apply_window(hu: np.ndarray, center: float, width: float) -> np.ndarray:
    lower = center - width / 2.0
    upper = center + width / 2.0
    windowed = np.clip(hu, lower, upper)
    return ((windowed - lower) / (upper - lower)).astype(np.float32)

def prepare_model_tensor(hu: np.ndarray, device: torch.device) -> torch.Tensor:
    ch_brain = apply_window(hu, center=40, width=80)
    ch_subdural = apply_window(hu, center=75, width=215)
    ch_bone = apply_window(hu, center=600, width=2800)
    composite = np.stack([ch_brain, ch_subdural, ch_bone], axis=-1)
    resized = cv2.resize(composite, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    
    # تحويل القيم لمجال [0, 1] القياسي ثم تطبيق ImageNet Normalization
    resized = np.clip(resized, 0.0, 1.0)
    norm = (resized - MEAN) / STD
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor.to(device)

def apply_display_window(hu_image: np.ndarray, wl: float, ww: float) -> np.ndarray:
    lower = wl - (ww / 2.0)
    windowed = np.clip((hu_image - lower) / ww, 0.0, 1.0) * 255.0
    return windowed.astype(np.uint8)

def read_scan(file_bytes: bytes, filename: str, wl: float = 40, ww: float = 80):
    clean_name = os.path.splitext(os.path.basename(filename))[0][:12].replace(" ", "_")
    meta = {
        "patient_id": f"PT-{clean_name.upper()}",
        "study_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "slice_thickness": 5.0,
    }

    if filename.lower().endswith(".dcm") and pydicom is not None:
        ds = pydicom.dcmread(io.BytesIO(file_bytes), force=True)
        pixel_array = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        hu = pixel_array * slope + intercept
        
        # تصحيح الـ PhotometricInterpretation إذا كان مقلوباً
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            hu = np.max(hu) - hu

        gray = apply_display_window(hu, wl, ww)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        meta["patient_id"] = str(getattr(ds, "PatientID", meta["patient_id"]))
        meta["study_date"] = str(getattr(ds, "StudyDate", meta["study_date"]))
        meta["slice_thickness"] = float(getattr(ds, "SliceThickness", 5.0))
        return rgb, hu, meta

    np_arr = np.asarray(bytearray(file_bytes), dtype=np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Could not decode image file: {filename}")
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hu = (gray.astype(np.float32) / 255.0) * 1000.0 - 500.0
    return rgb, hu, meta

def predict_with_mc_dropout(model: nn.Module, tensor: torch.Tensor, n_samples: int = 10) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    # تفعيل الـ Dropout فقط للطبقة النهائية دون لمس باقي الشبكة
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()

    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            logits = model(tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            preds.append(probs)

    preds = np.array(preds)
    means = np.mean(preds, axis=0)
    stds = np.std(preds, axis=0)
    return means, stds

class GradCAMPlusPlus:
    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None):
        self.model = model
        self.target_layer = target_layer or self._find_last_conv_layer(model)
        self.gradients = None
        self.activations = None
        self._register_hooks()

    @staticmethod
    def _find_last_conv_layer(model: nn.Module) -> nn.Module:
        last_conv = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        if last_conv is None:
            raise ValueError("No nn.Conv2d layer found.")
        return last_conv

    def _register_hooks(self) -> None:
        def forward_hook(_module, _inputs, output):
            self.activations = output

        def backward_hook(_module, _grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, class_idx: int, out_size: int = 256) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(input_tensor)
        target = output[0, class_idx]
        target.backward(retain_graph=True)

        grads = self.gradients[0].cpu().data.numpy()
        acts = self.activations[0].cpu().data.numpy()

        grads_power_2 = grads**2
        grads_power_3 = grads_power_2 * grads
        sum_acts = np.sum(acts, axis=(1, 2), keepdims=True)
        eps = 1e-7
        aij = grads_power_2 / (2.0 * grads_power_2 + sum_acts * grads_power_3 + eps)
        weights = np.sum(aij * np.maximum(grads, 0), axis=(1, 2))

        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i, :, :]

        cam = np.maximum(cam, 0)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)

        cam = np.where(cam > 0.40, cam, 0)
        return cv2.resize(cam, (out_size, out_size))

@st.cache_resource
def load_system():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"{CHECKPOINT_PATH} not found. Run 03_dataset_training.py first.")
    if not THRESHOLDS_PATH.exists():
        raise FileNotFoundError(f"{THRESHOLDS_PATH} not found. Run 05_evaluate.py first.")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    backbone_name = checkpoint.get("backbone", "efficientnet_b0")
    model = timm.create_model(backbone_name, pretrained=False, num_classes=len(SUBTYPES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE).eval()

    cam_engine = GradCAMPlusPlus(model)

    with open(THRESHOLDS_PATH) as f:
        thresholds = json.load(f)

    return model, cam_engine, thresholds, backbone_name

def compute_subtype_biomarkers(cam_map: np.ndarray, subtype_name: str, is_acute: bool, slice_thickness: float = 5.0, pixel_spacing: float = 0.5) -> dict:
    if not is_acute:
        return {
            "type": "Clear",
            "val_str": "0.0 cm3",
            "val_num": 0.0,
            "metric_name": "Estimated Volume (ABC/2)",
            "dim_str": "0.0 x 0.0 cm",
            "urgency": "Normal",
            "note": "Non-operative / No active focal lesion",
        }

    binary_mask = (cam_map > 0.45).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "type": subtype_name,
            "val_str": "0.0 cm3",
            "val_num": 0.0,
            "metric_name": "Estimated Volume (ABC/2)",
            "dim_str": "0.0 x 0.0 cm",
            "urgency": "Normal",
            "note": "Lesion signal below volumetric threshold",
        }

    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    dim1, dim2 = rect[1]
    scale = (512 / IMAGE_SIZE) * pixel_spacing
    a_mm = max(dim1, dim2) * scale
    b_mm = min(dim1, dim2) * scale

    if subtype_name == "Subarachnoid":
        return {
            "type": "Subarachnoid",
            "val_str": "N/A (Diffuse)",
            "val_num": 0.0,
            "metric_name": "Lesion Quantification",
            "dim_str": f"{round(a_mm / 10.0, 1)} x {round(b_mm / 10.0, 1)} cm",
            "urgency": "Stat CTA Alert",
            "note": "Diffuse sulcal hemorrhage: ABC/2 invalid; CTA angiogram required",
        }
    if subtype_name == "Subdural":
        thickness_mm = round(b_mm, 1)
        is_surgical = thickness_mm > 10.0
        return {
            "type": "Subdural",
            "val_str": f"{thickness_mm} mm",
            "val_num": thickness_mm,
            "metric_name": "Maximal Hematoma Thickness",
            "dim_str": f"Span: {round(a_mm / 10.0, 1)} cm",
            "urgency": "SURGICAL ALERT (>10mm)" if is_surgical else "Sub-surgical (<10mm)",
            "note": "Surgical evacuation indicated if thickness > 10 mm",
        }

    vol_cm3 = round((a_mm * b_mm * slice_thickness) / 2000.0, 2)
    is_surgical = vol_cm3 > 30.0
    return {
        "type": subtype_name,
        "val_str": f"{vol_cm3} cm3",
        "val_num": vol_cm3,
        "metric_name": "Estimated Volume (ABC/2)",
        "dim_str": f"{round(a_mm / 10.0, 1)} x {round(b_mm / 10.0, 1)} cm",
        "urgency": "SURGICAL (>30cm3)" if is_surgical else "Conservative (<30cm3)",
        "note": "Standard ABC/2 focal lesion ellipsoid model",
    }

def estimate_tilt_corrected_midline_shift(hu_slice: np.ndarray, pixel_spacing: tuple[float, float] = (0.5, 0.5)) -> tuple[float, bool]:
    bone_mask = (hu_slice > 300).astype(np.uint8)
    pts = np.argwhere(bone_mask > 0)
    if len(pts) < 100:
        return 0.0, False

    pts_xy = pts[:, [1, 0]].astype(np.float32)
    mean_center, eigenvectors = cv2.PCACompute(pts_xy, mean=np.empty((0)))

    angle_rad = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0])
    angle_deg = np.degrees(angle_rad) - 90.0

    h, w = hu_slice.shape
    rot_mat = cv2.getRotationMatrix2D((float(mean_center[0, 0]), float(mean_center[0, 1])), angle_deg, 1.0)
    aligned_hu = cv2.warpAffine(hu_slice, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderValue=-1000)

    brain_mask = ((aligned_hu >= 15) & (aligned_hu <= 85)).astype(np.uint8)
    moments = cv2.moments(brain_mask)
    if moments["m00"] == 0:
        return 0.0, False

    actual_x = moments["m10"] / moments["m00"]
    midline_x = mean_center[0, 0]

    shift_mm = abs(actual_x - midline_x) * pixel_spacing[0]
    shift_mm = round(float(np.clip(shift_mm, 0.0, 15.0)), 2)
    is_critical = shift_mm >= 5.0
    return shift_mm, is_critical

def export_clinical_pdf(patient_id, study_date, is_acute, any_prob, df_table, impression, bio_val, shift_mm, orig_img_path, fused_img_path) -> str:
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(12, 12, 12)
    pdf.add_page()
    pdf.set_auto_page_break(auto=False)

    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, "NEUROSCAN AI - CLINICAL AUDIT REPORT", ln=True, align="C")

    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 5, "Automated Non-Contrast Head CT Triage & Biomarker Synthesis", ln=True, align="C")
    pdf.ln(3)

    pdf.set_fill_color(248, 250, 252)
    pdf.rect(12, pdf.get_y(), 186, 14, "F")
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(30, 41, 59)
    pdf.set_xy(14, pdf.get_y() + 2)
    pdf.cell(0, 5, f"Patient ID: {patient_id}   |   Study Date: {study_date}   |   Audit: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC", ln=True)

    pdf.set_xy(14, pdf.get_y())
    if is_acute:
        pdf.set_text_color(220, 38, 38)
        status_txt = f"TRIAGE: CRITICAL STAT ({any_prob * 100:.1f}%)"
    elif any_prob >= 0.20:
        pdf.set_text_color(217, 119, 6)
        status_txt = f"TRIAGE: BORDERLINE / EQUIVOCAL ({any_prob * 100:.1f}%)"
    else:
        pdf.set_text_color(16, 185, 129)
        status_txt = f"TRIAGE: ROUTINE / CLEAR ({any_prob * 100:.1f}%)"

    pdf.cell(75, 5, status_txt, ln=False)
    pdf.set_text_color(71, 85, 105)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 5, f"Biomarker: {bio_val}  |  Midline Shift: {shift_mm} mm", ln=True)

    pdf.set_y(pdf.get_y() + 6)
    img_y = pdf.get_y()
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(30, 41, 59)
    pdf.set_xy(12, img_y)
    pdf.cell(90, 5, "Axial Non-Contrast Input", align="C")
    pdf.set_xy(108, img_y)
    pdf.cell(90, 5, "Grad-CAM++ Diagnostic Fusion", align="C")

    pdf.image(orig_img_path, x=26, y=img_y + 6, w=62, h=62)
    pdf.image(fused_img_path, x=122, y=img_y + 6, w=62, h=62)

    table_y = img_y + 72
    pdf.set_y(table_y)

    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(226, 232, 240)
    pdf.set_text_color(15, 23, 42)
    col_w = [50, 46, 45, 45]
    headers = ["Subtype Category", "Model Confidence (+/- Std)", "Operating Threshold", "Triage Decision"]
    for w, h in zip(col_w, headers):
        pdf.cell(w, 6, h, 1, 0, "C", True)
    pdf.ln()

    pdf.set_font("Helvetica", "", 8)
    for _, row in df_table.iterrows():
        pdf.set_text_color(30, 41, 59)
        pdf.cell(col_w[0], 5, str(row["Subtype"]), 1, 0, "L")
        pdf.cell(col_w[1], 5, str(row["Confidence"]), 1, 0, "C")
        pdf.cell(col_w[2], 5, str(row["Threshold"]), 1, 0, "C")
        decision = str(row["Decision"])
        if decision == "POSITIVE":
            pdf.set_text_color(220, 38, 38)
            pdf.set_font("Helvetica", "B", 8)
        else:
            pdf.set_text_color(100, 116, 139)
            pdf.set_font("Helvetica", "", 8)
        pdf.cell(col_w[3], 5, decision, 1, 1, "C")

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 5, "STRUCTURED RADIOLOGICAL IMPRESSION & RECOMMENDATION:", ln=True)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(51, 65, 85)
    pdf.multi_cell(0, 4.2, impression)
    pdf.ln(3)

    pdf.set_font("Helvetica", "I", 7.5)
    pdf.set_text_color(148, 163, 184)
    pdf.multi_cell(
        0, 3.8,
        "REGULATORY DISCLAIMER: NeuroScan AI provides computational decision-support triage. "
        "Quantitative biomarkers and model predictions do not replace a diagnostic radiologist "
        "review. Clinical and surgical management remains solely with the attending physician.",
    )

    tmp_out = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    pdf.output(tmp_out.name)
    return tmp_out.name

# Streamlit App UI
st.set_page_config(page_title="NeuroScan AI | Enterprise CDS Suite", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
<style>
    .metric-card { background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 12px; margin-bottom: 10px; }
    .badge-critical { background-color: #991b1b33; color: #ef4444; border: 1px solid #ef4444; padding: 4px 10px; border-radius: 6px; font-weight: 700; display: inline-block; }
    .badge-equivocal { background-color: #b4530933; color: #f59e0b; border: 1px solid #f59e0b; padding: 4px 10px; border-radius: 6px; font-weight: 700; display: inline-block; }
    .badge-clear { background-color: #065f4633; color: #10b981; border: 1px solid #10b981; padding: 4px 10px; border-radius: 6px; font-weight: 700; display: inline-block; }
    .report-preview { background: #1e293b; border-left: 4px solid #38bdf8; padding: 14px; border-radius: 4px; font-family: monospace; font-size: 13px; line-height: 1.6; }
</style>
""",
    unsafe_allow_html=True,
)

try:
    model, cam_engine, thresholds, active_backbone = load_system()
except Exception as e:
    st.error(f"Clinical Engine Load Error: {e}")
    st.stop()

st.title("NeuroScan AI — Enterprise CDS & Clinical Copilot")
st.caption(f"Commercial-Grade Intracranial Hemorrhage Triage ({active_backbone}) with LLM Clinical Reasoning & Quantitative Biomarkers")

st.sidebar.header("PACS Window Presets")
preset_names = list(WINDOW_PRESETS.keys()) + ["Custom"]
preset = st.sidebar.selectbox("Clinical Preset", preset_names)

if preset in WINDOW_PRESETS:
    wl, ww = WINDOW_PRESETS[preset]
else:
    wl = st.sidebar.slider("Window Level (Center HU)", -200, 1000, 40, 5)
    ww = st.sidebar.slider("Window Width (HU Range)", 20, 3000, 80, 10)

st.sidebar.markdown("---")
enable_uncertainty = st.sidebar.checkbox("Compute Monte Carlo Uncertainty (+/- sigma)", value=True)
cam_opacity = st.sidebar.slider("Diagnostic Fusion Opacity", 0.1, 0.9, 0.45, 0.05)

uploaded_files = st.sidebar.file_uploader("Upload CT Scan(s) [DICOM .dcm / PNG / JPG]", type=["dcm", "png", "jpg", "jpeg"], accept_multiple_files=True)

if not uploaded_files:
    st.info("Upload an axial CT slice or full patient DICOM series from the sidebar to launch analysis.")
    st.stop()

slices_data = []
for f in uploaded_files:
    f_bytes = f.read()
    try:
        rgb, hu, meta = read_scan(f_bytes, f.name, wl, ww)
    except Exception as exc:
        st.warning(f"Skipped {f.name}: could not be read ({exc}).")
        continue

    tensor = prepare_model_tensor(hu, DEVICE)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if enable_uncertainty:
        means, stds = predict_with_mc_dropout(model, tensor, n_samples=10)
    else:
        with torch.no_grad():
            raw_out = model(tensor)
            means = torch.sigmoid(raw_out).cpu().numpy()[0]
        stds = np.zeros_like(means)

    is_acute = means[ANY_INDEX] >= thresholds.get("any", 0.5)

    slices_data.append(
        {
            "name": f.name,
            "rgb": rgb,
            "hu": hu,
            "tensor": tensor,
            "means": means,
            "stds": stds,
            "is_acute": is_acute,
            "any_prob": means[ANY_INDEX],
            "meta": meta,
        }
    )

if not slices_data:
    st.error("None of the uploaded files could be read. Please upload valid DICOM/PNG/JPG scans.")
    st.stop()

if len(slices_data) >= 3:
    for i in range(len(slices_data)):
        if slices_data[i]["is_acute"]:
            prev_acute = slices_data[i - 1]["is_acute"] if i > 0 else False
            next_acute = slices_data[i + 1]["is_acute"] if i < len(slices_data) - 1 else False
            if not prev_acute and not next_acute and slices_data[i]["any_prob"] < 0.65:
                slices_data[i]["is_acute"] = False
                slices_data[i]["consistency_note"] = "Filtered as Single-Slice Noise Artifact"

_, pre_is_crit = estimate_tilt_corrected_midline_shift(slices_data[0]["hu"])

peak_prob = max(s["any_prob"] for s in slices_data)
exam_critical = any(s["is_acute"] for s in slices_data) or pre_is_crit
is_equivocal = (not exam_critical) and (peak_prob >= 0.20)

st.subheader("1. Series Triage & Emergency Worklist Status")
t1, t2, t3, t4 = st.columns(4)
with t1:
    st.markdown("**Emergency Priority**")
    if exam_critical:
        st.markdown('<div class="badge-critical">CRITICAL WORKLIST STAT</div>', unsafe_allow_html=True)
    elif is_equivocal:
        st.markdown('<div class="badge-equivocal">BORDERLINE / EQUIVOCAL</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="badge-clear">ROUTINE / CLEAR</div>', unsafe_allow_html=True)
with t2:
    st.markdown("**Study Patient ID**")
    st.write(slices_data[0]["meta"]["patient_id"])
with t3:
    st.markdown("**Processed Volume**")
    st.write(f"{len(slices_data)} Slice {'(3D Volumetric Audited)' if len(slices_data) > 1 else '(2D Single-Slice Demo - 3D Filter Idle)'}")
with t4:
    st.markdown("**Hemorrhage Probability (Risk Index)**")
    st.write(f"{peak_prob * 100:.1f}%")

st.markdown("---")

active_slice_idx = 0
if len(slices_data) > 1:
    active_slice_idx = st.slider("3D Axial Navigation", 0, len(slices_data) - 1, 0, format="Slice %d")

curr = slices_data[active_slice_idx]

positive_subtypes = [s for s in SUBTYPES if s != "any" and curr["means"][SUBTYPES.index(s)] >= thresholds.get(s, 0.5)]
subtype_means = [curr["means"][SUBTYPES.index(s)] for s in SUBTYPES if s != "any"]
non_any_subtypes = [s for s in SUBTYPES if s != "any"]
top_subtype_idx = int(np.argmax(subtype_means))
top_candidate_name = non_any_subtypes[top_subtype_idx].capitalize()
top_sub_p = subtype_means[top_subtype_idx]

if positive_subtypes:
    top_sub_name = positive_subtypes[0].capitalize()
    cam_target = SUBTYPES.index(positive_subtypes[0])
    is_indeterminate_subtype = False
elif curr["is_acute"]:
    top_sub_name = "Indeterminate / Multi-compartment"
    cam_target = SUBTYPES.index(non_any_subtypes[top_subtype_idx])
    is_indeterminate_subtype = True
else:
    top_sub_name = "None"
    cam_target = ANY_INDEX
    is_indeterminate_subtype = False

cam_map = cam_engine.generate(curr["tensor"], cam_target, out_size=IMAGE_SIZE)
bio = compute_subtype_biomarkers(cam_map, top_sub_name, curr["is_acute"], curr["meta"]["slice_thickness"])
midline_shift_mm, is_critical_shift = estimate_tilt_corrected_midline_shift(curr["hu"])

h_o, w_o, _ = curr["rgb"].shape
cam_full = cv2.resize(cam_map, (w_o, h_o))
heatmap = cv2.applyColorMap(np.uint8(255 * cam_full), cv2.COLORMAP_JET)
heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
overlay = np.uint8((1.0 - cam_opacity) * curr["rgb"] + cam_opacity * heatmap_rgb)

st.subheader("2. Quantitative Subtype-Aware Neuro-Biomarkers")
b1, b2, b3, b4 = st.columns(4)
with b1:
    st.metric(bio["metric_name"], bio["val_str"], delta=bio["urgency"], delta_color="inverse" if "SURGICAL" in bio["urgency"] or "STAT" in bio["urgency"] else "normal")
with b2:
    st.metric("Lesion Diameters (A x B)", bio["dim_str"])
with b3:
    st.metric("Tilt-Corrected Midline Shift", f"{midline_shift_mm} mm", delta="CRITICAL (>5mm)" if is_critical_shift else "Preserved", delta_color="inverse")
with b4:
    st.metric("Active Window Center/Width", f"L:{wl} / W:{ww} HU")

st.caption(f"Clinical Biomarker Rationale: {bio['note']}")

st.subheader(f"3. Explainable Localization & Diagnostic Fusion ({curr['name']})")
c1, c2, c3 = st.columns(3)
with c1:
    st.image(curr["rgb"], caption=f"Axial Scan (Window: {preset})", use_container_width=True)
with c2:
    st.image(heatmap_rgb, caption=f"Grad-CAM++ Focus ({SUBTYPES[cam_target].capitalize()})", use_container_width=True)
with c3:
    st.image(overlay, caption="Diagnostic Fusion (Scan + Heatmap)", use_container_width=True)

with st.expander("Interactive PACS HU Density Probe (Hover / Inspect Pixels)", expanded=False):
    down_hu = cv2.resize(curr["hu"], (128, 128))
    fig_hu = go.Figure(data=go.Heatmap(z=down_hu, colorscale="gray", colorbar=dict(title="HU Value"), hovertemplate="X: %{x}<br>Y: %{y}<br>HU: %{z:.1f}<extra></extra>"))
    fig_hu.update_layout(
        title="Axial HU Density Grid (Hover to inspect regional Hounsfield Units)",
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False, autorange="reversed"),
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig_hu, use_container_width=True)
    st.caption("HU Scale: Clotted Blood (+60 to +85 HU) | Brain Tissue (+30 to +45 HU) | CSF (0 to +15 HU)")

st.subheader("4. Subtype Probability Breakdown vs Calibrated Thresholds")
col_plot, col_table = st.columns([3, 2])

categories = [s.capitalize() for s in non_any_subtypes]
preds = [p * 100 for p in subtype_means]
errors = [curr["stds"][SUBTYPES.index(s)] * 100 for s in non_any_subtypes]
threshs = [thresholds.get(s, 0.5) * 100 for s in non_any_subtypes]

with col_plot:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=categories,
            y=preds,
            name="Model Confidence (%)",
            error_y=dict(type="data", array=errors, visible=enable_uncertainty),
            marker_color=["#ef4444" if p >= t else "#3b82f6" for p, t in zip(preds, threshs)],
            text=[f"{p:.1f}%" for p in preds],
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=categories, y=threshs, name="Operating Threshold (%)",
            mode="lines+markers", line=dict(color="#f59e0b", dash="dash", width=2), marker=dict(size=8, symbol="diamond"),
        )
    )
    fig.update_layout(
        title="Subtype Likelihood & Uncertainty Bounds",
        yaxis=dict(title="Probability (%)", range=[0, 115]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=360,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_table:
    rows = []
    for s in non_any_subtypes:
        idx = SUBTYPES.index(s)
        p = curr["means"][idx]
        sd = curr["stds"][idx]
        th = thresholds.get(s, 0.5)
        rows.append(
            {
                "Subtype": s.capitalize(),
                "Confidence": f"{p * 100:.2f}% +/- {sd * 100:.2f}%" if enable_uncertainty else f"{p * 100:.2f}%",
                "Threshold": f"{th * 100:.1f}%",
                "Decision": "POSITIVE" if p >= th else "NEGATIVE",
            }
        )
    df_table = pd.DataFrame(rows)
    st.dataframe(df_table, use_container_width=True, height=310)

st.subheader("5. Structured Clinical Impression & Official Export")

if midline_shift_mm < 0.5:
    shift_str = "<0.5 mm (Physiological / Preserved)"
else:
    shift_status = "CRITICAL >5mm" if is_critical_shift else "sub-critical"
    shift_str = f"{midline_shift_mm} mm ({shift_status})"

if not curr["is_acute"]:
    if curr["any_prob"] >= 0.20:
        triage_status = "BORDERLINE / EQUIVOCAL"
        rec_action = "URGENT NEURORADIOLOGY OVERREAD ADVISED (Indeterminate attenuation pattern)"
        finding_note = f"Borderline attenuation pattern (Hemorrhage suspicion index: {curr['any_prob'] * 100:.1f}%)."
    else:
        triage_status = "NEGATIVE / LOW RISK"
        rec_action = "Routine emergency worklist."
        finding_note = "Physiological ventricular architecture with no focal hyperdensities."

    clinical_impression = (
        f"FINDINGS: Axial non-contrast CT brain demonstrates {finding_note} "
        f"Tilt-corrected midline shift: {shift_str}. "
        f"IMPRESSION: Non-definitive for overt acute hemorrhage ({triage_status}). "
        f"RECOMMENDATION: {rec_action}"
    )
else:
    urgency_text = "EMERGENT SURGICAL NOTIFICATION" if bio["urgency"].startswith("SURGICAL") or is_critical_shift else "URGENT NEUROLOGICAL READ"
    if is_indeterminate_subtype:
        diag_str = f"Acute intracranial hemorrhage with indeterminate subtype (Leading pattern: {top_candidate_name} {top_sub_p * 100:.1f}%, below independent threshold)"
    else:
        diag_str = f"Acute {top_sub_name} hemorrhage (Subtype threshold exceeded at {top_sub_p * 100:.1f}%)"

    clinical_impression = (
        f"FINDINGS: Hyperdense attenuation pattern identified on axial scan. Global Hemorrhage Risk Index: {curr['any_prob'] * 100:.1f}%. "
        f"Biomarker evaluation: {bio['metric_name']} = {bio['val_str']}. "
        f"Tilt-corrected midline shift: {shift_str}. "
        f"IMPRESSION: {diag_str}. "
        f"RECOMMENDATION: {urgency_text} and urgent neuroradiology overread."
    )

st.markdown(f'<div class="report-preview">{clinical_impression}</div>', unsafe_allow_html=True)
st.write("")

c_alert, c_pdf = st.columns(2)
with c_alert:
    if st.button("Simulate STAT Emergency Webhook / Push Alert", use_container_width=True):
        st.toast(f"STAT Push Alert: {curr['meta']['patient_id']} | Priority STAT | Shift: {midline_shift_mm}mm")
        st.success(f"Emergency Webhook payload sent to On-Call Neurosurgeon: Patient {curr['meta']['patient_id']} | Priority STAT")

with c_pdf:
    if st.button("Export Comprehensive Clinical PDF", use_container_width=True):
        with st.spinner("Generating certified PDF..."):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f1:
                Image.fromarray(curr["rgb"]).save(f1.name)
                orig_p = f1.name
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f2:
                Image.fromarray(overlay).save(f2.name)
                fused_p = f2.name

            pdf_file = export_clinical_pdf(
                curr["meta"]["patient_id"], curr["meta"]["study_date"], curr["is_acute"], curr["any_prob"],
                df_table, clinical_impression, bio["val_str"], midline_shift_mm, orig_p, fused_p,
            )
            with open(pdf_file, "rb") as f:
                st.session_state["cached_pdf"] = f.read()

    if "cached_pdf" in st.session_state:
        st.download_button(
            label="Download Certified PDF Audit",
            data=st.session_state["cached_pdf"],
            file_name=f"NeuroScan_Audit_{curr['meta']['patient_id']}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

st.markdown("---")
st.subheader("6. Rad-Copilot: Autonomous Clinical Reasoning & Case Consultation")
st.caption("Ask questions about this specific scan, surgical implications, or radiological findings.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_question = st.chat_input("Ex: What are the surgical implications of this lesion thickness and midline shift?")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    gemini_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
    if not gemini_key or genai is None:
        response_text = "Clinical AI reasoning engine is currently unavailable. Please verify system environment configuration."
    else:
        try:
            client = genai.Client(api_key=gemini_key)
            context_prompt = f"""
You are a senior neuro-radiologist and AI clinical copilot.
Analyze the current patient case based on these calibrated biomarkers:
- Patient ID: {curr['meta']['patient_id']}
- Acute Hemorrhage Finding: {'POSITIVE (High Risk)' if curr['is_acute'] else ('EQUIVOCAL / BORDERLINE (20-50%)' if curr['any_prob'] >= 0.20 else 'NEGATIVE (Low Risk <20%)')}
- Hemorrhage Probability Index: {curr['any_prob'] * 100:.1f}%
- Prominent Subtype: {top_sub_name} (Confidence: {top_sub_p * 100:.1f}%)
- Quantitative Biomarker: {bio['metric_name']} = {bio['val_str']} ({bio['urgency']})
- Clinical Biomarker Rationale: {bio['note']}
- Tilt-Corrected Midline Shift: {midline_shift_mm} mm (Critical threshold is > 5 mm)
- Clinical Impression: {clinical_impression}

Doctor's Question: {user_question}

Provide a concise, direct, clinical response citing relevant neurosurgical guidelines (e.g. Brain Trauma Foundation). Emphasize that final management is determined by the treating surgeon.
"""
            with st.spinner("Analyzing scan findings & consulting clinical guidelines..."):
                response = client.models.generate_content(model="gemini-2.5-flash", contents=context_prompt)
                response_text = response.text
        except Exception as exc:
            response_text = f"Inference processing error: {exc}"

    with st.chat_message("assistant"):
        st.markdown(response_text)
    st.session_state.messages.append({"role": "assistant", "content": response_text})
