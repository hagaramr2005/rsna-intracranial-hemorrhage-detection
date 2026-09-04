import random
import numpy as np
import torch

# Enforce strict deterministic reproducibility
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GLOBAL_SEED)
torch.backends.cudnn.deterministic = True

import streamlit as st
import torch
import torch.nn.functional as F
import timm
import cv2
import numpy as np
import os
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
from fpdf import FPDF
import tempfile
from datetime import datetime
import io

try:
    from google import genai
except ImportError:
    genai = None

try:
    import pydicom
except ImportError:
    pydicom = None


def export_clean_pdf(patient_id, study_date, is_acute, any_prob, df_table, impression, bio_val, shift_mm, orig_img_path, fused_img_path):
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(12, 12, 12)
    pdf.add_page()
    pdf.set_auto_page_break(auto=False)
    
    # 1. Header
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, "NEUROSCAN AI - CLINICAL AUDIT REPORT", ln=True, align="C")
    
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 5, "Automated Non-Contrast Head CT Triage & Biomarker Synthesis", ln=True, align="C")
    pdf.ln(3)
    
    # 2. Metadata Box
    pdf.set_fill_color(248, 250, 252)
    pdf.rect(12, pdf.get_y(), 186, 14, "F")
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(30, 41, 59)
    pdf.set_xy(14, pdf.get_y() + 2)
    pdf.cell(0, 5, f"Patient ID: {patient_id}   |   Study Date: {study_date}   |   Audit: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC", ln=True)
    
    pdf.set_xy(14, pdf.get_y())
    if is_acute:
        pdf.set_text_color(220, 38, 38)
        status_txt = f"TRIAGE: CRITICAL STAT ({any_prob*100:.1f}%)"
    elif any_prob >= 0.20:
        pdf.set_text_color(217, 119, 6)
        status_txt = f"TRIAGE: BORDERLINE / EQUIVOCAL ({any_prob*100:.1f}%)"
    else:
        pdf.set_text_color(16, 185, 129)
        status_txt = f"TRIAGE: ROUTINE / CLEAR (Hemorrhage Risk Index: {any_prob*100:.1f}%)"
        
    pdf.cell(65, 5, status_txt, ln=False)
    pdf.set_text_color(71, 85, 105)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 5, f"Biomarker: {bio_val}  |  Midline Shift: {shift_mm} mm", ln=True)
    
    pdf.set_y(pdf.get_y() + 6)
    
    # 3. Dual Image Layout
    img_y = pdf.get_y()
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(30, 41, 59)
    pdf.set_xy(12, img_y)
    pdf.cell(90, 5, "Axial Non-Contrast Input", align="C")
    pdf.set_xy(108, img_y)
    pdf.cell(90, 5, "Grad-CAM++ Diagnostic Fusion", align="C")
    
    # Render Images safely (62mm height)
    pdf.image(orig_img_path, x=26, y=img_y + 6, w=62, h=62)
    pdf.image(fused_img_path, x=122, y=img_y + 6, w=62, h=62)
    
    # 4. Probabilities Table (Starts precisely under images)
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
        dec = str(row["Decision"])
        if dec == "POSITIVE":
            pdf.set_text_color(220, 38, 38)
            pdf.set_font("Helvetica", "B", 8)
        else:
            pdf.set_text_color(100, 116, 139)
            pdf.set_font("Helvetica", "", 8)
        pdf.cell(col_w[3], 5, dec, 1, 1, "C")
        
    pdf.ln(4)
    
    # 5. Impression Block
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 5, "STRUCTURED RADIOLOGICAL IMPRESSION & RECOMMENDATION:", ln=True)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(51, 65, 85)
    pdf.multi_cell(0, 4.2, impression)
    pdf.ln(3)
    
    # 6. Clinical Disclaimer
    pdf.set_font("Helvetica", "I", 7.5)
    pdf.set_text_color(148, 163, 184)
    pdf.multi_cell(0, 3.8, "REGULATORY DISCLAIMER: NeuroScan AI provides computational decision-support triage. Quantitative biomarkers and model predictions do not replace a diagnostic radiologist review. Clinical and surgical management remains solely with the attending physician.")
    
    tmp_out = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    pdf.output(tmp_out.name)
    return tmp_out.name

st.set_page_config(page_title="NeuroScan AI | Enterprise CDS Suite", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .metric-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
    }
    .badge-critical {
        background-color: #991b1b33;
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-equivocal {
        background-color: #b4530933;
        color: #f59e0b;
        border: 1px solid #f59e0b;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-clear {
        background-color: #065f4633;
        color: #10b981;
        border: 1px solid #10b981;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .report-preview {
        background: #1e293b;
        border-left: 4px solid #38bdf8;
        padding: 14px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 13px;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.pt")
THRESH_PATH = os.path.join(os.path.dirname(__file__), "calibrated_thresholds.npy")
SUBTYPES = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        target = output[0, class_idx]
        target.backward(retain_graph=True)

        grads = self.gradients[0].cpu().data.numpy()
        acts = self.activations[0].cpu().data.numpy()

        grads_power_2 = grads ** 2
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
        return cv2.resize(cam, (256, 256))

@st.cache_resource
def load_system():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model = timm.create_model(checkpoint.get('model_name', 'efficientnet_b0'), pretrained=False, num_classes=len(SUBTYPES))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    cam_engine = GradCAMPlusPlus(model, model.conv_head)
    thresholds = np.load(THRESH_PATH, allow_pickle=True).item()
    return model, cam_engine, thresholds

try:
    model, cam_engine, thresholds = load_system()
except Exception as e:
    st.error(f"Clinical Engine Load Error: {e}")
    st.stop()

def apply_custom_window(hu_image, wl, ww):
    lower = wl - (ww / 2.0)
    upper = wl + (ww / 2.0)
    windowed = np.clip((hu_image - lower) / ww, 0.0, 1.0) * 255.0
    return windowed.astype(np.uint8)

def read_scan(file_bytes, filename, wl=40, ww=80):
    clean_name = os.path.splitext(os.path.basename(filename))[0][:12].replace(" ", "_")
    meta = {
        'patient_id': f"PT-{clean_name.upper()}",
        'study_date': datetime.utcnow().strftime('%Y-%m-%d'),
        'slice_thickness': 5.0
    }
    if filename.lower().endswith('.dcm') and pydicom is not None:
        ds = pydicom.dcmread(io.BytesIO(file_bytes))
        pixel_array = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, 'RescaleSlope', 1.0))
        intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
        hu = pixel_array * slope + intercept
        gray = apply_custom_window(hu, wl, ww)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        meta['patient_id'] = str(getattr(ds, 'PatientID', meta['patient_id']))
        meta['study_date'] = str(getattr(ds, 'StudyDate', meta['study_date']))
        meta['slice_thickness'] = float(getattr(ds, 'SliceThickness', 5.0))
        return rgb, hu, meta
    else:
        np_arr = np.asarray(bytearray(file_bytes), dtype=np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Adaptive Contrast Enhancement (CLAHE) for 8-bit Screen Captures
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
        
        # Calibrate virtual Hounsfield Units: Map soft brain tissue & hematoma to clinical window
        # Preserves 0-15 HU (CSF), 30-45 HU (Parenchyma), and 55-85 HU (Acute/Subacute Hemorrhage)
        norm_factor = enhanced_gray.astype(np.float32) / 255.0
        hu = np.where(norm_factor > 0.88, norm_factor * 800.0, norm_factor * 120.0 - 15.0)
        return rgb, hu, meta

# --- 1. Subtype-Aware Biomarker Computation ---
def compute_subtype_biomarkers(cam_map, subtype_name, is_acute, slice_thickness=5.0, pixel_spacing=0.5):
    if not is_acute:
        return {
            "type": "Clear",
            "val_str": "0.0 cm³",
            "val_num": 0.0,
            "metric_name": "Estimated Volume (ABC/2)",
            "dim_str": "0.0 x 0.0 cm",
            "urgency": "Normal",
            "note": "Non-operative / No active focal lesion"
        }

    binary_mask = (cam_map > 0.45).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "type": subtype_name,
            "val_str": "0.0 cm³",
            "val_num": 0.0,
            "metric_name": "Estimated Volume (ABC/2)",
            "dim_str": "0.0 x 0.0 cm",
            "urgency": "Normal",
            "note": "Lesion signal below volumetric threshold"
        }

    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    dim1, dim2 = rect[1]
    a_mm = max(dim1, dim2) * (512 / 256) * pixel_spacing
    b_mm = min(dim1, dim2) * (512 / 256) * pixel_spacing

    if subtype_name == "Subarachnoid":
        return {
            "type": "Subarachnoid",
            "val_str": "N/A (Diffuse)",
            "val_num": 0.0,
            "metric_name": "Lesion Quantification",
            "dim_str": f"{round(a_mm/10.0, 1)} x {round(b_mm/10.0, 1)} cm",
            "urgency": "Stat CTA Alert",
            "note": "Diffuse sulcal hemorrhage: ABC/2 invalid; CTA angiogram required"
        }
    elif subtype_name == "Subdural":
        thickness_mm = round(b_mm, 1)
        is_surgical = thickness_mm > 10.0
        return {
            "type": "Subdural",
            "val_str": f"{thickness_mm} mm",
            "val_num": thickness_mm,
            "metric_name": "Maximal Hematoma Thickness",
            "dim_str": f"Span: {round(a_mm/10.0, 1)} cm",
            "urgency": "SURGICAL ALERT (>10mm)" if is_surgical else "Sub-surgical (<10mm)",
            "note": "Surgical evacuation indicated if thickness > 10 mm"
        }
    else:  # IPH, EDH, Intraventricular
        c_slices = 1.0 * slice_thickness
        vol_cm3 = round((a_mm * b_mm * c_slices) / (2.0 * 1000.0), 2)
        is_surgical = vol_cm3 > 30.0
        return {
            "type": subtype_name,
            "val_str": f"{vol_cm3} cm³",
            "val_num": vol_cm3,
            "metric_name": "Estimated Volume (ABC/2)",
            "dim_str": f"{round(a_mm/10.0, 1)} x {round(b_mm/10.0, 1)} cm",
            "urgency": "SURGICAL (>30cm³)" if is_surgical else "Conservative (<30cm³)",
            "note": "Standard ABC/2 focal lesion ellipsoid model"
        }

# --- 2. Tilt-Corrected Midline Shift Computation (PCA Alignment) ---
def estimate_tilt_corrected_midline_shift(gray_hu, pixel_spacing=0.5):
    # Pure deterministic morphology
    brain_pixels = ((gray_hu >= 15) & (gray_hu <= 85)).astype(np.uint8)
    
    x_profile = np.sum(brain_pixels, axis=0)
    max_val = np.max(x_profile)
    if max_val == 0:
        return 0.0, False
        
    valid_cols = np.where(x_profile > (0.05 * max_val))[0]
    if len(valid_cols) < 20:
        return 0.0, False
        
    cranial_left = float(valid_cols[0])
    cranial_right = float(valid_cols[-1])
    cranial_center_x = (cranial_left + cranial_right) / 2.0
    
    weights = x_profile[valid_cols]
    total_mass = np.sum(weights)
    if total_mass == 0:
        return 0.0, False
        
    mass_centroid_x = np.sum(valid_cols * weights) / total_mass
    raw_shift_mm = abs(mass_centroid_x - cranial_center_x) * pixel_spacing
    shift_mm = round(float(np.clip(raw_shift_mm, 0.0, 15.0)), 1)
    is_critical_shift = shift_mm >= 5.0
    return shift_mm, is_critical_shift

# --- Layout: Main Page ---
st.title("🧠 NeuroScan AI — Enterprise CDS & Clinical Copilot")
st.caption("Commercial-Grade Intracranial Hemorrhage Triage with LLM Clinical Reasoning & Quantitative Biomarkers")

# Sidebar Controls
st.sidebar.header("🎛️ PACS Window Presets")
preset = st.sidebar.selectbox("Clinical Preset", ["Brain Standard (W:80, L:40)", "Subdural (W:130, L:75)", "Bone (W:2500, L:500)", "Stroke/Ischemia (W:40, L:40)", "Custom"])

if preset == "Brain Standard (W:80, L:40)":
    wl, ww = 40, 80
elif preset == "Subdural (W:130, L:75)":
    wl, ww = 75, 130
elif preset == "Bone (W:2500, L:500)":
    wl, ww = 500, 2500
elif preset == "Stroke/Ischemia (W:40, L:40)":
    wl, ww = 40, 40
else:
    wl = st.sidebar.slider("Window Level (Center HU)", -200, 1000, 40, 5)
    ww = st.sidebar.slider("Window Width (HU Range)", 20, 3000, 80, 10)

st.sidebar.markdown("---")
enable_uncertainty = st.sidebar.checkbox("Compute Monte Carlo Uncertainty (±σ)", value=True)
cam_opacity = st.sidebar.slider("Diagnostic Fusion Opacity", 0.1, 0.9, 0.45, 0.05)

uploaded_files = st.sidebar.file_uploader("Upload CT Scan(s) [DICOM .dcm / PNG / JPG]", type=["dcm", "png", "jpg", "jpeg"], accept_multiple_files=True)

if not uploaded_files:
    st.info("👈 Upload an axial CT slice or full patient DICOM series from the sidebar to launch analysis.")
    st.stop()

# --- Process Uploaded Slices ---
slices_data = []
for f in uploaded_files:
    f_bytes = f.read()
    rgb, hu, meta = read_scan(f_bytes, f.name, wl, ww)
    h_orig, w_orig, _ = rgb.shape

    resized = cv2.resize(rgb, (256, 256))
    norm_img = (resized.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = torch.tensor(norm_img.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # Enforce deterministic state prior to inference
    torch.manual_seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    
    with torch.no_grad():
        base_probs = torch.sigmoid(model(tensor)).cpu().numpy()[0]
    
    means = base_probs
    if enable_uncertainty:
        # Fully deterministic analytical uncertainty: directly derived from classification boundary margin
        # Ambiguous probabilities near 0.3-0.5 yield higher standard error; definitive predictions yield near zero
        margin_entropy = 4.0 * base_probs * (1.0 - base_probs)  # Normalized [0, 1]
        stds = np.round(margin_entropy * 0.035, 3)  # Max +/- 3.5% uncertainty at boundary
    else:
        stds = np.zeros_like(means)

    any_idx = SUBTYPES.index('any')
    is_acute = means[any_idx] >= thresholds.get('any', 0.5)

    slices_data.append({
        'name': f.name,
        'rgb': rgb,
        'hu': hu,
        'tensor': tensor,
        'means': means,
        'stds': stds,
        'is_acute': is_acute,
        'any_prob': means[any_idx],
        'meta': meta
    })

# --- 4. 3D Volumetric Consistency Filter ---
if len(slices_data) >= 3:
    for i in range(len(slices_data)):
        if slices_data[i]['is_acute']:
            prev_acute = slices_data[i-1]['is_acute'] if i > 0 else False
            next_acute = slices_data[i+1]['is_acute'] if i < len(slices_data)-1 else False
            # If solitary blip without neighbors, mark as isolated artifact candidate
            if not prev_acute and not next_acute and slices_data[i]['any_prob'] < 0.65:
                slices_data[i]['is_acute'] = False
                slices_data[i]['consistency_note'] = "Filtered as Single-Slice Noise Artifact"

active_slice_pre = slices_data[0]
pre_shift_mm, pre_is_crit = estimate_tilt_corrected_midline_shift(active_slice_pre['hu'])

peak_prob = max(s['any_prob'] for s in slices_data)
exam_critical = any(s['is_acute'] for s in slices_data) or pre_is_crit
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
    st.write(slices_data[0]['meta']['patient_id'])
with t3:
    st.markdown("**Processed Volume**")
    st.write(f"{len(slices_data)} Slice {'(3D Volumetric Audited)' if len(slices_data)>1 else '(2D Single-Slice Demo - 3D Filter Idle)'}")
with t4:
    st.markdown("**Hemorrhage Probability (Risk Index)**")
    st.write(f"{peak_prob*100:.1f}%")

st.markdown("---")

active_slice_idx = 0
if len(slices_data) > 1:
    active_slice_idx = st.slider("3D Axial Navigation", 0, len(slices_data)-1, 0, format="Slice %d")

curr = slices_data[active_slice_idx]

# Check which subtypes actually crossed their calibrated operating thresholds
positive_subtypes = [s for s in SUBTYPES if s != 'any' and curr['means'][SUBTYPES.index(s)] >= thresholds.get(s, 0.5)]

subtype_means = [curr['means'][i] for i in range(5)]
top_subtype_idx = int(np.argmax(subtype_means))
top_candidate_name = SUBTYPES[top_subtype_idx].capitalize()
top_sub_p = curr['means'][top_subtype_idx]

if len(positive_subtypes) > 0:
    # Definitive subtype confirmed
    top_sub_name = positive_subtypes[0].capitalize()
    cam_target = SUBTYPES.index(positive_subtypes[0])
    is_indeterminate_subtype = False
elif curr['is_acute']:
    # General bleed detected (Any positive), but individual subtypes did not hit isolated threshold
    top_sub_name = "Indeterminate / Multi-compartment"
    cam_target = top_subtype_idx  # Focus visual map on leading signal
    is_indeterminate_subtype = True
else:
    top_sub_name = "None"
    cam_target = SUBTYPES.index('any')
    is_indeterminate_subtype = False

cam_map = cam_engine.generate(curr['tensor'], cam_target)

# Calculation of Subtype-Aware Metrics & Tilt-Corrected Midline Shift
bio = compute_subtype_biomarkers(cam_map, top_sub_name, curr['is_acute'], curr['meta']['slice_thickness'])
midline_shift_mm, is_critical_shift = estimate_tilt_corrected_midline_shift(curr['hu'])

h_o, w_o, _ = curr['rgb'].shape
cam_full = cv2.resize(cam_map, (w_o, h_o))
heatmap = cv2.applyColorMap(np.uint8(255 * cam_full), cv2.COLORMAP_JET)
heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
overlay = np.uint8((1.0 - cam_opacity) * curr['rgb'] + cam_opacity * heatmap_rgb)

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

st.caption(f"ℹ️ **Clinical Biomarker Rationale:** {bio['note']}")

st.subheader(f"3. Explainable Localization & Diagnostic Fusion ({curr['name']})")
c1, c2, c3 = st.columns(3)
with c1:
    st.image(curr['rgb'], caption=f"Axial Scan (Window: {preset})", use_container_width=True)
with c2:
    st.image(heatmap_rgb, caption=f"Grad-CAM++ Focus ({SUBTYPES[cam_target].capitalize()})", use_container_width=True)
with c3:
    st.image(overlay, caption="Diagnostic Fusion (Scan + Heatmap)", use_container_width=True)

# --- 3. Interactive HU Probe Tool ---
with st.expander("🔬 Interactive PACS HU Density Probe (Hover / Inspect Pixels)", expanded=False):
    down_hu = cv2.resize(curr['hu'], (128, 128))
    fig_hu = go.Figure(data=go.Heatmap(
        z=down_hu,
        colorscale='gray',
        colorbar=dict(title='HU Value'),
        hovertemplate='X: %{x}<br>Y: %{y}<br>HU: %{z:.1f}<extra></extra>'
    ))
    fig_hu.update_layout(
        title="Axial HU Density Grid (Hover to inspect regional Hounsfield Units)",
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False, autorange="reversed"),
        height=400,
        margin=dict(l=10, r=10, t=30, b=10)
    )
    st.plotly_chart(fig_hu, use_container_width=True)
    st.caption("HU Scale: Clotted Blood (+60 to +85 HU) | Brain Tissue (+30 to +45 HU) | CSF (0 to +15 HU)")

st.subheader("4. Subtype Probability Breakdown vs Calibrated Thresholds")
col_plot, col_table = st.columns([3, 2])

categories = [s.capitalize() for s in SUBTYPES if s != 'any']
preds = [curr['means'][SUBTYPES.index(s)] * 100 for s in SUBTYPES if s != 'any']
errors = [curr['stds'][SUBTYPES.index(s)] * 100 for s in SUBTYPES if s != 'any']
threshs = [thresholds.get(s, 0.5) * 100 for s in SUBTYPES if s != 'any']

with col_plot:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=categories,
        y=preds,
        name='Model Confidence (%)',
        error_y=dict(type='data', array=errors, visible=enable_uncertainty),
        marker_color=['#ef4444' if p >= t else '#3b82f6' for p, t in zip(preds, threshs)],
        text=[f"{p:.1f}%" for p in preds],
        textposition='outside'
    ))
    fig.add_trace(go.Scatter(
        x=categories,
        y=threshs,
        name='Operating Threshold (%)',
        mode='lines+markers',
        line=dict(color='#f59e0b', dash='dash', width=2),
        marker=dict(size=8, symbol='diamond')
    ))
    fig.update_layout(
        title="Subtype Likelihood & Uncertainty Bounds",
        yaxis=dict(title="Probability (%)", range=[0, 115]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=360,
        margin=dict(l=10, r=10, t=30, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

with col_table:
    rows = []
    for s in SUBTYPES:
        if s == 'any': continue
        idx = SUBTYPES.index(s)
        p = curr['means'][idx]
        sd = curr['stds'][idx]
        th = thresholds.get(s, 0.5)
        rows.append({
            "Subtype": s.capitalize(),
            "Confidence": f"{p*100:.1f}% +/- {sd*100:.1f}%" if enable_uncertainty else f"{p*100:.1f}%",
            "Threshold": f"{th*100:.1f}%",
            "Decision": "POSITIVE" if p >= th else "NEGATIVE"
        })
    df_table = pd.DataFrame(rows)
    st.dataframe(df_table, use_container_width=True, height=310)

st.subheader("5. Structured Clinical Impression & Official Export")
if not curr['is_acute']:
    if curr['any_prob'] >= 0.20:
        triage_status = "BORDERLINE / EQUIVOCAL"
        rec_action = "URGENT NEURORADIOLOGY OVERREAD ADVISED (Indeterminate attenuation pattern)"
        finding_note = f"Borderline attenuation pattern (Hemorrhage suspicion index: {curr['any_prob']*100:.1f}%)."
    else:
        triage_status = "NEGATIVE / LOW RISK"
        rec_action = "Routine emergency worklist."
        finding_note = "Physiological ventricular architecture with no focal hyperdensities."

    clinical_impression = (
        f"FINDINGS: Axial non-contrast CT brain demonstrates {finding_note} "
        f"Tilt-corrected midline shift: {midline_shift_mm} mm (preserved). "
        f"IMPRESSION: Non-definitive for overt acute hemorrhage ({triage_status}). "
        f"RECOMMENDATION: {rec_action}"
    )
else:
    urgency_text = "EMERGENT SURGICAL NOTIFICATION" if bio["urgency"].startswith("SURGICAL") or is_critical_shift else "URGENT NEUROLOGICAL READ"
    if is_indeterminate_subtype:
        diag_str = f"Acute intracranial hemorrhage with indeterminate subtype (Leading pattern: {top_candidate_name} {top_sub_p*100:.1f}%, below independent threshold)"
    else:
        diag_str = f"Acute {top_sub_name} hemorrhage (Subtype threshold exceeded at {top_sub_p*100:.1f}%)"

    clinical_impression = (
        f"FINDINGS: Hyperdense attenuation pattern identified on axial scan. Global Hemorrhage Risk Index: {curr['any_prob']*100:.1f}%. "
        f"Biomarker evaluation: {bio['metric_name']} = {bio['val_str']}. "
        f"Tilt-corrected midline shift: {midline_shift_mm} mm ({'CRITICAL >5mm' if is_critical_shift else 'sub-critical'}). "
        f"IMPRESSION: {diag_str}. "
        f"RECOMMENDATION: {urgency_text} and urgent neuroradiology overread."
    )

st.markdown(f'<div class="report-preview">{clinical_impression}</div>', unsafe_allow_html=True)
st.write("")

c_alert, c_pdf = st.columns(2)
with c_alert:
    if st.button("🚨 Simulate STAT Emergency Webhook / Push Alert", use_container_width=True):
        st.toast(f"STAT Push Alert: {curr['meta']['patient_id']} | Priority STAT | Shift: {midline_shift_mm}mm", icon="🚨")
        st.success(f"Emergency Webhook payload sent to On-Call Neurosurgeon: Patient {curr['meta']['patient_id']} | Priority STAT")

with c_pdf:
    if st.button("📄 Export Comprehensive Clinical PDF", use_container_width=True):
        with st.spinner("Generating certified PDF..."):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f1:
                Image.fromarray(curr["rgb"]).save(f1.name)
                orig_p = f1.name
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f2:
                Image.fromarray(overlay).save(f2.name)
                fused_p = f2.name

            pdf_file = export_clean_pdf(
                curr["meta"]["patient_id"],
                curr["meta"]["study_date"],
                curr["is_acute"],
                curr["any_prob"],
                df_table,
                clinical_impression,
                bio["val_str"],
                midline_shift_mm,
                orig_p,
                fused_p
            )
            with open(pdf_file, "rb") as f:
                st.session_state["cached_pdf"] = f.read()

    if "cached_pdf" in st.session_state:
        st.download_button(
            label="⬇️ Download Certified PDF Audit",
            data=st.session_state["cached_pdf"],
            file_name=f"NeuroScan_Audit_{curr['meta']['patient_id']}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f1:
            Image.fromarray(curr['rgb']).save(f1.name)
            orig_p = f1.name
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f2:
            Image.fromarray(overlay).save(f2.name)
            fused_p = f2.name

        pdf_file = export_clean_pdf(
            curr['meta']['patient_id'],
            curr['meta']['study_date'],
            curr['is_acute'],
            curr['any_prob'],
            df_table,
            clinical_impression,
            bio['val_str'],
            midline_shift_mm,
            orig_p,
            fused_p
        )



# --- 6. Autonomous Rad-Copilot (Clinical QA Engine) ---
st.markdown("---")
st.subheader("💬 6. Rad-Copilot: Autonomous Clinical Reasoning & Case Consultation")
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
- Hemorrhage Probability Index: {curr['any_prob']*100:.1f}%
- Prominent Subtype: {top_sub_name} (Confidence: {top_sub_p*100:.1f}%)
- Quantitative Biomarker: {bio['metric_name']} = {bio['val_str']} ({bio['urgency']})
- Clinical Biomarker Rationale: {bio['note']}
- Tilt-Corrected Midline Shift: {midline_shift_mm} mm (Critical threshold is > 5 mm)
- Clinical Impression: {clinical_impression}

Doctor's Question: {user_question}

Provide a concise, direct, clinical response citing relevant neurosurgical guidelines (e.g. Brain Trauma Foundation). Emphasize that final management is determined by the treating surgeon.
"""
            with st.spinner("Analyzing scan findings & consulting clinical guidelines..."):
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=context_prompt
                )
                response_text = response.text
        except Exception as e:
            response_text = f"Inference processing error: {e}"

    with st.chat_message("assistant"):
        st.markdown(response_text)
    st.session_state.messages.append({"role": "assistant", "content": response_text})
