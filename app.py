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
    meta = {'patient_id': 'PT-EMERG-901', 'study_date': datetime.utcnow().strftime('%Y-%m-%d'), 'slice_thickness': 5.0}
    if filename.lower().endswith('.dcm') and pydicom is not None:
        ds = pydicom.dcmread(io.BytesIO(file_bytes))
        pixel_array = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, 'RescaleSlope', 1.0))
        intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
        hu = pixel_array * slope + intercept
        gray = apply_custom_window(hu, wl, ww)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        meta['patient_id'] = str(getattr(ds, 'PatientID', 'ANONYMIZED'))
        meta['study_date'] = str(getattr(ds, 'StudyDate', meta['study_date']))
        meta['slice_thickness'] = float(getattr(ds, 'SliceThickness', 5.0))
        return rgb, hu, meta
    else:
        np_arr = np.asarray(bytearray(file_bytes), dtype=np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        hu = (gray.astype(np.float32) / 255.0) * 1000.0 - 500.0
        return rgb, hu, meta

def compute_abc2_volume(cam_map, slice_thickness=5.0, pixel_spacing=0.5):
    binary_mask = (cam_map > 0.45).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, 0.0, 0.0
    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    dim1, dim2 = rect[1]
    a_mm = max(dim1, dim2) * (512 / 256) * pixel_spacing
    b_mm = min(dim1, dim2) * (512 / 256) * pixel_spacing
    c_slices = 1.0 * slice_thickness
    volume_cm3 = (a_mm * b_mm * c_slices) / (2.0 * 1000.0)
    return round(volume_cm3, 2), round(a_mm / 10.0, 1), round(b_mm / 10.0, 1)

def estimate_midline_shift(gray_hu):
    h, w = gray_hu.shape
    mid = w // 2
    left_hem = np.mean(gray_hu[:, :mid])
    right_hem = np.mean(gray_hu[:, mid:])
    asymmetry_ratio = abs(left_hem - right_hem) / (max(left_hem, right_hem) + 1e-6)
    shift_mm = round(float(asymmetry_ratio * 12.0), 1)
    is_critical_shift = shift_mm > 5.0
    return shift_mm, is_critical_shift

def export_clean_pdf(patient_id, study_date, is_acute, peak_conf, breakdown_df, impression, vol_cm3, shift_mm, orig_path, fused_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(0, 8, "NEUROSCAN AI - COMPREHENSIVE RADIOLOGY AUDIT", ln=True, align="C")
    pdf.set_font("Helvetica", size=8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, "Automated Quantitative Biomarkers & Clinical Decision Support", ln=True, align="C")
    pdf.ln(4)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", 'B', 10)
    pdf.cell(0, 6, "1. Patient Metadata & Emergency Biomarkers", ln=True)
    pdf.set_font("Helvetica", size=9)
    pdf.cell(95, 6, f"Patient ID: {patient_id}", border=1)
    pdf.cell(95, 6, f"Study Date: {study_date}", border=1, ln=True)
    pdf.cell(95, 6, f"Triage Alert: {'CRITICAL EMERGENCY' if is_acute else 'SCREENING CLEAR'}", border=1)
    pdf.cell(95, 6, f"Estimated Volume: {vol_cm3} cm3 (ABC/2)", border=1, ln=True)
    pdf.cell(95, 6, f"Midline Shift: {shift_mm} mm ({'CRITICAL' if shift_mm > 5.0 else 'NORMAL'})", border=1)
    pdf.cell(95, 6, f"Peak AI Confidence: {peak_conf*100:.1f}%", border=1, ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", 'B', 10)
    pdf.cell(0, 6, "2. Quantitative Subtype Classification", ln=True)
    pdf.set_font("Helvetica", 'B', 8)
    pdf.cell(45, 6, "Subtype", 1)
    pdf.cell(45, 6, "Confidence", 1)
    pdf.cell(50, 6, "Threshold", 1)
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

    pdf.set_font("Helvetica", 'B', 10)
    pdf.cell(0, 6, "3. Visual Explainability (CT Scan + Grad-CAM++ Focus)", ln=True)
    pdf.image(orig_path, x=15, y=pdf.get_y(), w=85)
    pdf.image(fused_path, x=105, y=pdf.get_y(), w=85)
    pdf.ln(88)

    pdf.set_font("Helvetica", 'B', 10)
    pdf.cell(0, 6, "4. Radiologist Clinical Impression", ln=True)
    pdf.set_font("Helvetica", size=8)
    clean_imp = impression.encode('latin-1', 'ignore').decode('latin-1')
    pdf.multi_cell(0, 4, clean_imp)
    pdf.ln(3)
    pdf.set_font("Helvetica", 'I', 7)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 4, "DISCLAIMER: Computational triage aid. Must be corroborated by licensed medical personnel.")

    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_out.name)
    return temp_out.name

# --- Layout: Main Page ---
st.title("🧠 NeuroScan AI — Multimodal CDS & Clinical Copilot")
st.caption("Commercial-Grade Intracranial Hemorrhage Triage with LLM Clinical Reasoning & Quantitative Biomarkers")

# --- Sidebar Controls ---
default_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))

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

peak_prob = max(s['any_prob'] for s in slices_data)
exam_critical = any(s['is_acute'] for s in slices_data)

st.subheader("1. Series Triage & Emergency Worklist Status")
t1, t2, t3, t4 = st.columns(4)
with t1:
    st.markdown("**Emergency Priority**")
    if exam_critical:
        st.markdown('<div class="badge-critical">CRITICAL WORKLIST STAT</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="badge-clear">ROUTINE / CLEAR</div>', unsafe_allow_html=True)
with t2:
    st.markdown("**Study Patient ID**")
    st.write(slices_data[0]['meta']['patient_id'])
with t3:
    st.markdown("**Processed Volume**")
    st.write(f"{len(slices_data)} Slices")
with t4:
    st.markdown("**Peak Hemorrhage Probability**")
    st.write(f"{peak_prob*100:.1f}%")

st.markdown("---")

active_slice_idx = 0
if len(slices_data) > 1:
    active_slice_idx = st.slider("3D Axial Navigation", 0, len(slices_data)-1, 0, format="Slice %d")

curr = slices_data[active_slice_idx]

subtype_means = [curr['means'][i] for i in range(5)]
top_subtype_idx = int(np.argmax(subtype_means))
cam_target = top_subtype_idx if curr['is_acute'] else SUBTYPES.index('any')
cam_map = cam_engine.generate(curr['tensor'], cam_target)

vol_cm3, dim_a, dim_b = compute_abc2_volume(cam_map, curr['meta']['slice_thickness'])
midline_shift_mm, is_critical_shift = estimate_midline_shift(curr['hu'])

h_o, w_o, _ = curr['rgb'].shape
cam_full = cv2.resize(cam_map, (w_o, h_o))
heatmap = cv2.applyColorMap(np.uint8(255 * cam_full), cv2.COLORMAP_JET)
heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
overlay = np.uint8((1.0 - cam_opacity) * curr['rgb'] + cam_opacity * heatmap_rgb)

st.subheader("2. Quantitative Neuro-Biomarkers")
b1, b2, b3, b4 = st.columns(4)
with b1:
    st.metric("Estimated Volume (ABC/2)", f"{vol_cm3} cm³", delta="Surgical Threshold >30cm³" if vol_cm3 > 30 else "Non-operative", delta_color="inverse")
with b2:
    st.metric("Lesion Diameters (A x B)", f"{dim_a} x {dim_b} cm")
with b3:
    st.metric("Midline Shift Deviation", f"{midline_shift_mm} mm", delta="CRITICAL (>5mm)" if is_critical_shift else "Preserved", delta_color="inverse")
with b4:
    st.metric("Active Window Center/Width", f"L:{wl} / W:{ww} HU")

st.subheader(f"3. Explainable Localization & Diagnostic Fusion ({curr['name']})")
c1, c2, c3 = st.columns(3)
with c1:
    st.image(curr['rgb'], caption=f"Axial Scan (Window: {preset})", use_container_width=True)
with c2:
    st.image(heatmap_rgb, caption=f"Grad-CAM++ Focus ({SUBTYPES[cam_target].capitalize()})", use_container_width=True)
with c3:
    st.image(overlay, caption="Diagnostic Fusion (Scan + Heatmap)", use_container_width=True)

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
top_sub_name = SUBTYPES[top_subtype_idx].capitalize()
top_sub_p = curr['means'][top_subtype_idx]

if not curr['is_acute']:
    clinical_impression = (
        f"FINDINGS: Axial non-contrast CT brain demonstrates physiological ventricular architecture. "
        f"No intracranial mass effect or midline shift ({midline_shift_mm} mm). "
        f"IMPRESSION: Negative for acute intracranial hemorrhage (Screening confidence: {(1 - curr['any_prob']) * 100:.1f}%). "
        f"RECOMMENDATION: Routine emergency worklist."
    )
else:
    urgency_text = "EMERGENT SURGICAL NOTIFICATION" if vol_cm3 > 30.0 or is_critical_shift else "URGENT NEUROLOGICAL READ"
    clinical_impression = (
        f"FINDINGS: Acute hyperdense focal lesion identified on axial scan with maximal features consistent with {top_sub_name} hemorrhage. "
        f"Quantitative volumetric computation estimates {vol_cm3} cm3 (dimensions {dim_a} x {dim_b} cm). "
        f"Midline shift calculated at {midline_shift_mm} mm ({'CRITICAL >5mm' if is_critical_shift else 'sub-critical'}). "
        f"IMPRESSION: Acute {top_sub_name} hemorrhage with AI confidence {top_sub_p*100:.1f}%. "
        f"RECOMMENDATION: {urgency_text} and urgent neurosurgical consultation."
    )

st.markdown(f'<div class="report-preview">{clinical_impression}</div>', unsafe_allow_html=True)
st.write("")

c_alert, c_pdf = st.columns(2)
with c_alert:
    if st.button("🚨 Simulate STAT Emergency Webhook / Push Alert", use_container_width=True):
        st.toast(f"STAT Push Alert Dispatched: {curr['meta']['patient_id']} - {top_sub_name} ({vol_cm3} cm³)", icon="🚨")
        st.success(f"Emergency Webhook payload sent to On-Call Neurosurgeon: Patient {curr['meta']['patient_id']} | Priority STAT")

with c_pdf:
    if st.button("📄 Export Comprehensive Clinical PDF", use_container_width=True):
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
            vol_cm3,
            midline_shift_mm,
            orig_p,
            fused_p
        )

        with open(pdf_file, "rb") as pdf_data:
            st.download_button(
                label="⬇️ Download Certified PDF Audit",
                data=pdf_data.read(),
                file_name=f"Comprehensive_Radiology_{curr['meta']['patient_id']}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

# --- 6. Gemini Multimodal Clinical Copilot (NLP Integration) ---
st.markdown("---")
st.subheader("💬 6. Rad-Copilot: Autonomous Clinical Reasoning & Case Consultation")
st.caption("Ask questions about this specific scan, surgical implications, or radiological findings.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_question = st.chat_input("Ex: Does this patient need urgent surgical craniotomy based on ABC/2 and midline shift?")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    gemini_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
    if not gemini_key or genai is None:
        response_text = "⚠️ Clinical AI reasoning engine is currently unavailable. Please verify system environment configuration."
    else:
        try:
            client = genai.Client(api_key=gemini_key)
            context_prompt = f"""
You are a senior neuro-radiologist and AI clinical copilot.
Analyze the current patient case based on these real AI inference biomarkers:
- Patient ID: {curr['meta']['patient_id']}
- Acute Hemorrhage Flag: {curr['is_acute']} (Peak Confidence: {curr['any_prob']*100:.1f}%)
- Prominent Subtype: {top_sub_name} (Confidence: {top_sub_p*100:.1f}%)
- Estimated Volume (ABC/2): {vol_cm3} cm³ (Surgical threshold is > 30 cm³)
- Midline Shift: {midline_shift_mm} mm (Critical shift threshold is > 5 mm)
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
