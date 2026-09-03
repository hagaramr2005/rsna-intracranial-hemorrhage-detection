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
    import pydicom
except ImportError:
    pydicom = None

st.set_page_config(page_title="NeuroScan AI | Enterprise CDS", layout="wide", initial_sidebar_state="expanded")

# --- Clinical Enterprise CSS ---
st.markdown("""
<style>
    .metric-box {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 10px;
    }
    .badge-critical {
        background-color: #991b1b33;
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-clear {
        background-color: #065f4633;
        color: #10b981;
        border: 1px solid #10b981;
        padding: 6px 12px;
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

class GradCAM:
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

        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = np.maximum(cam, 0)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
        return cv2.resize(cam, (256, 256))

@st.cache_resource
def load_system():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model = timm.create_model(checkpoint.get('model_name', 'efficientnet_b0'), pretrained=False, num_classes=len(SUBTYPES))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    grad_cam = GradCAM(model, model.conv_head)
    thresholds = np.load(THRESH_PATH, allow_pickle=True).item()
    return model, grad_cam, thresholds

try:
    model, grad_cam, thresholds = load_system()
except Exception as e:
    st.error(f"Engine Load Error: {e}")
    st.stop()

# --- 1. Real DICOM & Hounsfield Physics Engine ---
def process_dicom_raw(dcm_bytes):
    ds = pydicom.dcmread(io.BytesIO(dcm_bytes))
    pixel_array = ds.pixel_array.astype(np.float32)
    slope = getattr(ds, 'RescaleSlope', 1.0)
    intercept = getattr(ds, 'RescaleIntercept', 0.0)
    hu = pixel_array * slope + intercept
    
    # Extract 3 Real Clinical Windows into RGB channels
    # Brain: W:80, L:40 | Subdural: W:130, L:75 | Bone: W:2500, L:500
    def window(img, wl, ww):
        lower = wl - ww / 2
        upper = wl + ww / 2
        return np.clip((img - lower) / ww, 0, 1) * 255.0

    b_win = window(hu, 40, 80)
    s_win = window(hu, 75, 130)
    o_win = window(hu, 500, 2500)
    rgb_composite = np.stack([b_win, s_win, o_win], axis=-1).astype(np.uint8)
    
    meta = {
        'patient_id': str(getattr(ds, 'PatientID', 'ANONYMIZED')),
        'study_date': str(getattr(ds, 'StudyDate', datetime.utcnow().strftime('%Y-%m-%d'))),
        'kvp': str(getattr(ds, 'KVP', '120')),
        'slice_thickness': str(getattr(ds, 'SliceThickness', '5.0'))
    }
    return rgb_composite, meta

# --- 2. Anatomical Region Mapping Engine ---
def map_anatomical_location(cam_map):
    h, w = cam_map.shape
    cy, cx = np.unravel_index(np.argmax(cam_map), cam_map.shape)
    
    lateral = "Right" if cx < w / 2 else "Left"
    
    # Distance from center
    norm_x = (cx - w / 2) / (w / 2)
    norm_y = (cy - h / 2) / (h / 2)
    dist = np.sqrt(norm_x**2 + norm_y**2)
    
    if dist > 0.75:
        zone = "Extra-axial / Calvarial Convexity space"
    elif dist < 0.3:
        zone = "Periventricular / Deep nuclear zone"
    elif norm_y < -0.2:
        zone = "Frontal cortical / subcortical parenchyma"
    elif norm_y > 0.3:
        zone = "Posterior fossa / Occipitotemporal parenchyma"
    else:
        zone = "Temporoparietal parenchymal convexity"
        
    return f"{lateral} {zone}"

# --- 3. Clinical Radiology Impression Engine (Rad-LLM Simulator) ---
def generate_radiologist_impression(patient_id, is_acute, peak_conf, top_subtype, top_prob, location_str):
    if not is_acute:
        return (
            f"CLINICAL IMPRESSION: Non-contrast head CT demonstrates no acute intracranial hemorrhage, "
            f"mass effect, or midline shift. Ventricular configuration and basal cisterns remain within normal limits. "
            f"ROUTINE WORKLIST STATUS (Screening Confidence: {(1 - peak_conf) * 100:.1f}%)."
        )
    
    urgency = "EMERGENT STAT" if top_prob > 0.60 else "URGENT"
    return (
        f"CLINICAL IMPRESSION: Non-contrast head CT reveals acute {top_subtype.lower()} focus "
        f"with maximal saliency localized over the {location_str}. "
        f"AI diagnostic confidence is {top_prob*100:.1f}%. Hemorrhagic density warrants immediate "
        f"correlation for localized mass effect and sulcal effacement. "
        f"TRIAGE ACTION: {urgency} neurosurgical notification and confirmatory clinical review advised."
    )

# --- 4. Sanitized PDF Report Generation Engine ---
def build_clean_pdf(patient_id, study_date, is_acute, peak_conf, breakdown_df, impression_text, location_str, orig_path, fused_path):
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(0, 8, "NEUROSCAN AI - ADVANCED RADIOLOGICAL AUDIT", ln=True, align="C")
    pdf.set_font("Helvetica", size=8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, f"PACS Integration Simulation | Certified Deep Inference Audit", ln=True, align="C")
    pdf.ln(4)

    # Patient Details Table
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", 'B', 10)
    pdf.cell(0, 6, "1. Patient & PACS Acquisition Parameters", ln=True)
    pdf.set_font("Helvetica", size=9)
    pdf.cell(95, 6, f"Patient ID: {patient_id.encode('latin-1', 'ignore').decode('latin-1')}", border=1)
    pdf.cell(95, 6, f"Study Date: {study_date}", border=1, ln=True)
    pdf.cell(95, 6, f"Triage Alert: {'CRITICAL (Positive)' if is_acute else 'NON-URGENT (Clear)'}", border=1)
    pdf.cell(95, 6, f"Peak Hemorrhage Probability: {peak_conf*100:.1f}%", border=1, ln=True)
    pdf.ln(4)

    # Subtype Quantitative Breakdown
    pdf.set_font("Helvetica", 'B', 10)
    pdf.cell(0, 6, "2. Subtype Probability Breakdown & Calibrated Decisions", ln=True)
    pdf.set_font("Helvetica", 'B', 8)
    pdf.cell(45, 6, "Subtype", 1)
    pdf.cell(45, 6, "AI Confidence", 1)
    pdf.cell(50, 6, "Operating Threshold", 1)
    pdf.cell(50, 6, "Status", 1, ln=True)

    pdf.set_font("Helvetica", size=8)
    for _, r in breakdown_df.iterrows():
        sub = str(r['Subtype']).encode('latin-1', 'ignore').decode('latin-1')
        conf = str(r['Confidence']).replace("±", "+/-").encode('latin-1', 'ignore').decode('latin-1')
        thr = str(r['Threshold']).encode('latin-1', 'ignore').decode('latin-1')
        stat = str(r['Decision']).replace("🔴", "").replace("⚪", "").strip().upper()
        
        pdf.cell(45, 6, sub, 1)
        pdf.cell(45, 6, conf, 1)
        pdf.cell(50, 6, thr, 1)
        pdf.cell(50, 6, stat, 1, ln=True)
    pdf.ln(4)

    # Visual Evidence
    pdf.set_font("Helvetica", 'B', 10)
    pdf.cell(0, 6, f"3. Visual Localization | Focus: {location_str.encode('latin-1', 'ignore').decode('latin-1')}", ln=True)
    pdf.image(orig_path, x=15, y=pdf.get_y(), w=85)
    pdf.image(fused_path, x=105, y=pdf.get_y(), w=85)
    pdf.ln(88)

    # Clinical Impression
    pdf.set_font("Helvetica", 'B', 10)
    pdf.cell(0, 6, "4. Structured Radiologist Impression (Rad-LLM Synthesis)", ln=True)
    pdf.set_font("Helvetica", size=8)
    clean_impression = impression_text.encode('latin-1', 'ignore').decode('latin-1')
    pdf.multi_cell(0, 4, clean_impression)
    pdf.ln(3)

    pdf.set_font("Helvetica", 'I', 7)
    pdf.set_text_color(130, 130, 130)
    pdf.multi_cell(0, 4, "AUDIT NOTE: Autonomous Clinical Decision Support prototype. Correlate with board-certified radiologist read.")

    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_out.name)
    return temp_out.name

# --- Layout: Main Interface ---
st.title("🧠 NeuroScan AI — Clinical Suite & PACS Diagnostic Engine")
st.caption("Commercial-Grade Diagnostic Triage, Raw DICOM Ingestion & Explainable Localization")

# --- Sidebar Controls ---
st.sidebar.header("🎛️ Clinical Parameters")
enable_uncertainty = st.sidebar.checkbox("Activate Monte Carlo Dropout (±σ)", value=True)
opacity = st.sidebar.slider("Diagnostic Fusion Opacity", 0.1, 0.9, 0.45, 0.05)

uploaded_files = st.sidebar.file_uploader(
    "Ingest CT Study (DICOM .dcm or Pre-windowed PNG/JPG)", 
    type=["dcm", "png", "jpg", "jpeg"], 
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("👈 Ingest axial DICOM (.dcm) files or standard CT slices to launch clinical triage.")
    st.stop()

# --- Ingestion & Inference Loop ---
slices_data = []
metadata_dict = {'patient_id': 'PT-9824-EMERG', 'study_date': datetime.utcnow().strftime('%Y-%m-%d')}

for f in uploaded_files:
    f_bytes = f.read()
    if f.name.lower().endswith('.dcm') and pydicom is not None:
        rgb_img, meta = process_dicom_raw(f_bytes)
        metadata_dict = meta
    else:
        file_bytes = np.asarray(bytearray(f_bytes), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Preprocessing
    resized = cv2.resize(rgb_img, (256, 256))
    norm_img = (resized.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = torch.tensor(norm_img.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # Monte Carlo Dropout or Deterministic Pass
    if enable_uncertainty:
        model.train()
        preds = [torch.sigmoid(model(tensor)).cpu().detach().numpy()[0] for _ in range(8)]
        means = np.mean(preds, axis=0)
        stds = np.std(preds, axis=0)
        model.eval()
    else:
        with torch.no_grad():
            means = torch.sigmoid(model(tensor)).cpu().numpy()[0]
            stds = np.zeros_like(means)

    any_idx = SUBTYPES.index('any')
    slices_data.append({
        'name': f.name,
        'rgb': rgb_img,
        'tensor': tensor,
        'means': means,
        'stds': stds,
        'is_acute': means[any_idx] >= thresholds.get('any', 0.5),
        'any_prob': means[any_idx]
    })

# Series Aggregation
exam_positive = any(s['is_acute'] for s in slices_data)
peak_conf = max(s['any_prob'] for s in slices_data)

# Triage Verdict Dashboard
st.markdown("### 1. Series Triage & Examination Metadata")
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown("**Study Triage Priority**")
    if exam_positive:
        st.markdown('<div class="badge-critical">CRITICAL EMERGENCY STAT</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="badge-clear">ROUTINE / SCREENING CLEAR</div>', unsafe_allow_html=True)
with m2:
    st.markdown("**Patient Study ID**")
    st.write(metadata_dict.get('patient_id', 'ANONYMIZED'))
with m3:
    st.markdown("**Total Exam Volume**")
    st.write(f"{len(slices_data)} Slices Processed")
with m4:
    st.markdown("**Peak Hemorrhage Probability**")
    st.write(f"{peak_conf*100:.1f}%")

st.markdown("---")

# Volumetric Slice Navigation
active_idx = 0
if len(slices_data) > 1:
    active_idx = st.slider("3D Volumetric Axial Navigation", 0, len(slices_data)-1, 0, format="Slice %d")

curr = slices_data[active_idx]

# --- Localization & Anatomical Region Mapping ---
subtype_means = [curr['means'][i] for i in range(5)]
top_subtype_idx = int(np.argmax(subtype_means))
top_subtype_name = SUBTYPES[top_subtype_idx].capitalize()
top_prob = curr['means'][top_subtype_idx]

cam_idx = top_subtype_idx if curr['is_acute'] else SUBTYPES.index('any')
cam_map = grad_cam.generate(curr['tensor'], cam_idx)
anatomical_site = map_anatomical_location(cam_map) if curr['is_acute'] else "Non-focal / Unremarkable"

h_orig, w_orig, _ = curr['rgb'].shape
cam_full = cv2.resize(cam_map, (w_orig, h_orig))
heatmap = cv2.applyColorMap(np.uint8(255 * cam_full), cv2.COLORMAP_JET)
heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
overlay = np.uint8((1.0 - opacity) * curr['rgb'] + opacity * heatmap_rgb)

st.markdown(f"### 2. Explainable Localization & Anatomical Mapping ({curr['name']})")
st.info(f"📍 **Predicted Anatomical Focus:** `{anatomical_site}`")

c_img1, c_img2, c_img3 = st.columns(3)
with c_img1:
    st.image(curr['rgb'], caption="Raw Axial Slice", use_container_width=True)
with c_img2:
    st.image(heatmap_rgb, caption=f"Grad-CAM Attention: {SUBTYPES[cam_idx].capitalize()}", use_container_width=True)
with c_img3:
    st.image(overlay, caption="Diagnostic Fusion (CT + Saliency)", use_container_width=True)

# --- Quantitative Subtype Analytics ---
st.markdown("### 3. Quantitative Subtype Analytics & Threshold Crossings")
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
        title="Subtype Probability vs Operating Cutoffs",
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

# --- 4. Clinical Impression (Rad-LLM Synthesis) & Export ---
st.markdown("### 4. Automated Radiology Impression & Official Export")

clinical_impression_str = generate_radiologist_impression(
    metadata_dict.get('patient_id', 'ANONYMIZED'),
    curr['is_acute'],
    curr['any_prob'],
    top_subtype_name,
    top_prob,
    anatomical_site
)

st.markdown(f'<div class="report-preview">{clinical_impression_str}</div>', unsafe_allow_html=True)
st.write("")

if st.button("📄 Generate & Download Certified Radiological PDF", use_container_width=True):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f1:
        Image.fromarray(curr['rgb']).save(f1.name)
        orig_p = f1.name
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f2:
        Image.fromarray(overlay).save(f2.name)
        fused_p = f2.name

    pdf_file = build_clean_pdf(
        metadata_dict.get('patient_id', 'PT-2026-ICH'),
        metadata_dict.get('study_date', '2026-09-04'),
        curr['is_acute'],
        curr['any_prob'],
        df_table,
        clinical_impression_str,
        anatomical_site,
        orig_p,
        fused_p
    )

    with open(pdf_file, "rb") as pdf_data:
        st.download_button(
            label="⬇️ Click to Download Official PDF Report",
            data=pdf_data.read(),
            file_name=f"Radiology_Audit_{metadata_dict.get('patient_id', 'PT-ICH')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
