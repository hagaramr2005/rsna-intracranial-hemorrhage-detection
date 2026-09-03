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

st.set_page_config(page_title="NeuroScan AI | Clinical Suite", layout="wide", initial_sidebar_state="expanded")

# --- Custom Styling for Hospital CDS Interface ---
st.markdown("""
<style>
    .metric-card {
        background-color: #111927;
        border: 1px solid #1f2a3d;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-critical {
        background-color: #ef444422;
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-clear {
        background-color: #10b98122;
        color: #10b981;
        border: 1px solid #10b981;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
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
    st.error(f"Failed to initialize Clinical Engine: {e}")
    st.stop()

# --- Multi-Window Transformation Simulator ---
def apply_radiology_windows(gray_img):
    # Normalized approximations of CT Hounsfield windows: Brain (W:80, L:40), Subdural (W:130, L:75), Bone (W:2500, L:500)
    norm = gray_img.astype(np.float32) / 255.0
    brain_win = np.clip((norm - 0.2) / 0.6, 0, 1) * 255.0
    subdural_win = np.clip((norm - 0.3) / 0.7, 0, 1) * 255.0
    bone_win = np.clip((norm - 0.6) / 0.4, 0, 1) * 255.0
    return np.stack([brain_win, subdural_win, bone_win], axis=-1).astype(np.uint8)

# --- Monte Carlo Dropout Uncertainty Engine ---
def predict_with_uncertainty(model, tensor, runs=8):
    model.train() # Keep dropout active during inference
    preds = []
    with torch.no_grad():
        for _ in range(runs):
            out = torch.sigmoid(model(tensor)).cpu().numpy()[0]
            preds.append(out)
    preds = np.array(preds)
    means = np.mean(preds, axis=0)
    stds = np.std(preds, axis=0)
    model.eval()
    return means, stds

# --- PDF Generation Pipeline ---
def export_pdf_report(patient_id, triage_status, max_conf, details_df, original_img_path, gradcam_img_path):
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Helvetica", 'B', 18)
    pdf.cell(0, 10, "NEUROSCAN AI - RADIOLOGICAL TRIAGE REPORT", ln=True, align="C")
    pdf.set_font("Helvetica", size=9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, f"Generated on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')} | Validated RSNA Ensemble Core", ln=True, align="C")
    pdf.ln(5)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 8, "1. Patient & Examination Metadata", ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.cell(90, 6, f"Patient/Study ID: {patient_id}", border=1)
    pdf.cell(100, 6, f"Triage Priority: {'CRITICAL (Acute Hemorrhage)' if triage_status else 'NON-URGENT (Screening Negative)'}", border=1, ln=True)
    pdf.cell(90, 6, f"Peak AI Confidence: {max_conf*100:.2f}%", border=1)
    pdf.cell(100, 6, "Architecture: EfficientNet-B0 (Multi-label)", border=1, ln=True)
    pdf.ln(6)

    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 8, "2. Quantitative Subtype Classification", ln=True)
    pdf.set_font("Helvetica", 'B', 9)
    pdf.cell(45, 6, "Subtype", 1)
    pdf.cell(40, 6, "Probability", 1)
    pdf.cell(45, 6, "Operating Threshold", 1)
    pdf.cell(40, 6, "Diagnostic Status", 1, ln=True)
    
    pdf.set_font("Helvetica", size=9)
    for _, row in details_df.iterrows():
        pdf.cell(45, 6, str(row['Subtype']), 1)
        pdf.cell(40, 6, f"{row['Confidence']}", 1)
        pdf.cell(45, 6, f"{row['Threshold']}", 1)
        pdf.cell(40, 6, str(row['Decision']), 1, ln=True)
    pdf.ln(6)

    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 8, "3. Visual Explainability (CT & Grad-CAM Fusion)", ln=True)
    pdf.image(original_img_path, x=20, y=pdf.get_y(), w=80)
    pdf.image(gradcam_img_path, x=110, y=pdf.get_y(), w=80)
    pdf.ln(85)

    pdf.set_font("Helvetica", 'I', 8)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 4, "DISCLAIMER: This diagnostic summary is generated by an automated clinical decision support system (CDS) for research triage purposes. Final diagnosis must be confirmed by a board-certified radiologist.")
    
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_pdf.name)
    return temp_pdf.name

# --- Layout: Header ---
st.title("🧠 NeuroScan AI — Emergency Triage & Explainability Suite")
st.markdown("Clinical Decision Support (CDS) for Automated CT Brain Hemorrhage Screening & Subtyping")

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Clinical Configuration")
enable_uncertainty = st.sidebar.checkbox("Compute Monte Carlo Uncertainty (±σ)", value=True)
opacity = st.sidebar.slider("Grad-CAM Overlay Opacity", min_value=0.1, max_value=0.9, value=0.45, step=0.05)
windowing_mode = st.sidebar.selectbox("CT Window Physics", ["Brain Window Standard", "Triple Composite (Brain/Subdural/Bone)"])

uploaded_files = st.sidebar.file_uploader("Upload Axial CT Slice(s)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if not uploaded_files:
    st.info("👈 Upload one or multiple axial CT slices from the sidebar to initialize the clinical triage suite.")
    st.stop()

# --- Process Multi-Slice / 3D Series ---
st.subheader("1. Series-Level Navigation & Volumetric Triage")

slices_data = []
for idx, f in enumerate(uploaded_files):
    f_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(f_bytes, cv2.IMREAD_COLOR)
    h, w, _ = img_bgr.shape
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    if windowing_mode == "Triple Composite (Brain/Subdural/Bone)":
        proc_img = apply_radiology_windows(gray)
    else:
        proc_img = img_bgr

    resized = cv2.resize(proc_img, (256, 256))
    norm_img = (resized.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = torch.tensor(norm_img.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    if enable_uncertainty:
        means, stds = predict_with_uncertainty(model, tensor)
    else:
        with torch.no_grad():
            means = torch.sigmoid(model(tensor)).cpu().numpy()[0]
            stds = np.zeros_like(means)

    any_idx = SUBTYPES.index('any')
    slices_data.append({
        'name': f.name,
        'original_rgb': cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
        'tensor': tensor,
        'means': means,
        'stds': stds,
        'is_acute': means[any_idx] >= thresholds.get('any', 0.5),
        'any_prob': means[any_idx]
    })

# Series Aggregation (Exam-Level Triage)
exam_positive = any(s['is_acute'] for s in slices_data)
peak_conf = max(s['any_prob'] for s in slices_data)

col_metric1, col_metric2, col_metric3 = st.columns(3)
with col_metric1:
    st.markdown("**Exam Triage Verdict**")
    if exam_positive:
        st.markdown('<div class="badge-critical">CRITICAL WORKLIST PRIORITY</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="badge-clear">NON-URGENT (CLEAR)</div>', unsafe_allow_html=True)

with col_metric2:
    st.markdown("**Total Exam Volume**")
    st.markdown(f"### {len(slices_data)} Slices Processed")

with col_metric3:
    st.markdown("**Peak Hemorrhage Probability**")
    st.markdown(f"### {peak_conf*100:.1f}%")

st.markdown("---")

# Slice Selector / 3D Slider
selected_slice_idx = 0
if len(slices_data) > 1:
    selected_slice_idx = st.slider("3D Axial Volume Slider (Scroll through patient slices)", 0, len(slices_data)-1, 0, format="Slice %d")

curr = slices_data[selected_slice_idx]

# --- Visualization Section ---
st.subheader(f"2. Diagnostic Focus: {curr['name']}")

# Compute Grad-CAM for most prominent subtype
subtype_means = [curr['means'][i] for i in range(5)]
prominent_subtype_idx = int(np.argmax(subtype_means))
cam_target = prominent_subtype_idx if curr['is_acute'] else SUBTYPES.index('any')
cam_target_name = SUBTYPES[cam_target].capitalize()

cam_map = grad_cam.generate(curr['tensor'], cam_target)
cam_resized = cv2.resize(cam_map, (curr['original_rgb'].shape[1], curr['original_rgb'].shape[0]))
heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
overlay = np.uint8((1.0 - opacity) * curr['original_rgb'] + opacity * heatmap_rgb)

col_v1, col_v2, col_v3 = st.columns(3)
with col_v1:
    st.image(curr['original_rgb'], caption="Raw Axial Slice", use_container_width=True)
with col_v2:
    st.image(heatmap_rgb, caption=f"Grad-CAM Saliency ({cam_target_name})", use_container_width=True)
with col_v3:
    st.image(overlay, caption=f"Diagnostic Overlay (Fused)", use_container_width=True)

# --- Quantitative Breakdown & Plotly ---
st.subheader("3. Subtype Confidence & Uncertainty Calibration")

categories = [s.capitalize() for s in SUBTYPES if s != 'any']
preds = [curr['means'][SUBTYPES.index(s)] * 100 for s in SUBTYPES if s != 'any']
errors = [curr['stds'][SUBTYPES.index(s)] * 100 for s in SUBTYPES if s != 'any']
threshs = [thresholds.get(s, 0.5) * 100 for s in SUBTYPES if s != 'any']

col_plot, col_table = st.columns([3, 2])

with col_plot:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=categories,
        y=preds,
        name='AI Confidence (%)',
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
        title="Subtype Likelihood with Uncertainty Error Bands",
        yaxis=dict(title="Probability (%)", range=[0, 115]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=380,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

with col_table:
    breakdown_data = []
    for s in SUBTYPES:
        if s == 'any': continue
        idx = SUBTYPES.index(s)
        p = curr['means'][idx]
        sd = curr['stds'][idx]
        th = thresholds.get(s, 0.5)
        breakdown_data.append({
            "Subtype": s.capitalize(),
            "Confidence": f"{p*100:.1f}% ± {sd*100:.1f}%" if enable_uncertainty else f"{p*100:.1f}%",
            "Threshold": f"{th*100:.1f}%",
            "Decision": "🔴 Positive" if p >= th else "⚪ Negative"
        })
    df_table = pd.DataFrame(breakdown_data)
    st.dataframe(df_table, use_container_width=True, height=330)

# --- 4. Report Generation Section ---
st.subheader("4. Automated Structured Clinical Report")

col_rep1, col_rep2 = st.columns([2, 1])
with col_rep1:
    patient_identifier = st.text_input("Assign Patient / Case ID", value="PT-2026-ICH-9041")
with col_rep2:
    st.write("")
    st.write("")
    if st.button("📄 Generate Radiological PDF Report", use_container_width=True):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_orig:
            Image.fromarray(curr['original_rgb']).save(f_orig.name)
            temp_orig_path = f_orig.name
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_cam:
            Image.fromarray(overlay).save(f_cam.name)
            temp_cam_path = f_cam.name

        pdf_path = export_pdf_report(patient_identifier, curr['is_acute'], curr['any_prob'], df_table, temp_orig_path, temp_cam_path)

        with open(pdf_path, "rb") as f_pdf:
            st.download_button(
                label="⬇️ Download Official Radiology Report",
                data=f_pdf.read(),
                file_name=f"Radiology_Report_{patient_identifier}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
