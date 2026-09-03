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

st.set_page_config(page_title="NeuroScan AI - ICH Triage & Explainability", layout="wide")

MODEL_PATH = "/content/drive/MyDrive/rsna_project/best_model.pt"
THRESH_PATH = "/content/drive/MyDrive/rsna_project/calibrated_thresholds.npy"
SUBTYPES = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Grad-CAM Implementation for EfficientNet-B0 ---
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
        cam = cv2.resize(cam, (256, 256))
        return cam

@st.cache_resource
def load_system():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model = timm.create_model(checkpoint.get('model_name', 'efficientnet_b0'), pretrained=False, num_classes=len(SUBTYPES))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    
    # Target last conv layer in EfficientNet conv_head
    grad_cam = GradCAM(model, model.conv_head)
    thresholds = np.load(THRESH_PATH, allow_pickle=True).item()
    return model, grad_cam, thresholds

st.title("🧠 NeuroScan AI: Acute Intracranial Hemorrhage Triage & Localization")
st.caption("Standalone Clinical Decision Support System | Explainable Multi-label CT Evaluation")

try:
    model, grad_cam, thresholds = load_system()
    st.sidebar.success("Artifacts & Explainability Engine Loaded")
except Exception as e:
    st.sidebar.error(f"Error loading system: {e}")
    st.stop()

uploaded_file = st.file_uploader("Upload Brain CT Scan Slice (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h_orig, w_orig, _ = image_rgb.shape

    # Preprocessing
    resized = cv2.resize(image_bgr, (256, 256))
    norm_img = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm_img = (norm_img - mean) / std
    tensor = torch.tensor(norm_img.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # 1. Forward Pass
    model.eval()
    logits = model(tensor)
    probs = torch.sigmoid(logits).detach().cpu().numpy()[0]

    any_idx = SUBTYPES.index('any')
    any_prob = probs[any_idx]
    any_thresh = thresholds.get('any', 0.5)
    is_positive = any_prob >= any_thresh

    # 2. Clinical Triage Banner
    st.markdown("---")
    if is_positive:
        st.error(f"### ⚠️ CLINICAL ALERT: ACUTE HEMORRHAGE DETECTED (Confidence: {any_prob*100:.1f}%)")
    else:
        st.success(f"### ✅ SCREENING NEGATIVE: NO EVIDENCE OF ACUTE HEMORRHAGE (Confidence: {(1-any_prob)*100:.1f}%)")

    # 3. Visual Localization (Grad-CAM Overlay)
    st.markdown("#### 1. Visual Localization (Model Attention Explanation)")
    
    # Target subtype with highest probability or triage flag
    subtype_probs = [probs[i] for i in range(5)]
    highest_subtype_idx = int(np.argmax(subtype_probs))
    target_cam_idx = highest_subtype_idx if is_positive else any_idx
    target_cam_name = SUBTYPES[target_cam_idx].capitalize()

    cam_map = grad_cam.generate(tensor, target_cam_idx)
    cam_resized = cv2.resize(cam_map, (w_orig, h_orig))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.6 * image_rgb + 0.4 * heatmap)

    col_img1, col_img2, col_img3 = st.columns(3)
    with col_img1:
        st.image(image_rgb, caption="Original CT Slice", use_container_width=True)
    with col_img2:
        st.image(heatmap, caption=f"Model Attention Map ({target_cam_name})", use_container_width=True)
    with col_img3:
        st.image(overlay, caption="Diagnostic Fusion (CT + Grad-CAM Overlay)", use_container_width=True)

    st.caption("> **Clinical Note:** The highlighted region visualizes model saliency/attention guiding classification. It represents localization evidence, not manual radiological segmentation.")

    # 4. Quantitative Subtype Breakdown & Interactive Charts
    st.markdown("#### 2. Subtype Probability Breakdown vs Operating Thresholds")
    
    col_chart, col_table = st.columns([3, 2])

    categories = [s.capitalize() for s in SUBTYPES if s != 'any']
    pred_vals = [probs[SUBTYPES.index(s)] * 100 for s in SUBTYPES if s != 'any']
    thresh_vals = [thresholds.get(s, 0.5) * 100 for s in SUBTYPES if s != 'any']

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=categories,
        y=pred_vals,
        name='Model Confidence (%)',
        marker_color=['#d62728' if p >= t else '#1f77b4' for p, t in zip(pred_vals, thresh_vals)],
        text=[f"{p:.1f}%" for p in pred_vals],
        textposition='outside'
    ))
    fig.add_trace(go.Scatter(
        x=categories,
        y=thresh_vals,
        mode='lines+markers',
        name='Operating Threshold (%)',
        line=dict(color='#ff7f0e', dash='dash', width=2),
        marker=dict(size=8, symbol='diamond')
    ))

    fig.update_layout(
        title="Subtype Likelihood vs Calibrated Decision Thresholds",
        yaxis=dict(title="Probability (%)", range=[0, 105]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
        height=380
    )

    with col_chart:
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        table_records = []
        for s in SUBTYPES:
            if s == 'any':
                continue
            idx = SUBTYPES.index(s)
            p = probs[idx]
            th = thresholds.get(s, 0.5)
            status = "🔴 Positive" if p >= th else "⚪ Negative"
            table_records.append({
                "Subtype": s.capitalize(),
                "Probability": f"{p*100:.2f}%",
                "Threshold": f"{th*100:.2f}%",
                "Status": status
            })
        st.dataframe(pd.DataFrame(table_records), use_container_width=True, height=340)
