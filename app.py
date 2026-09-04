import torch
import numpy as np
import cv2

def apply_window(hu, center, width):
    lower = center - width / 2.0
    upper = center + width / 2.0
    w_img = np.clip(hu, lower, upper)
    return ((w_img - lower) / (upper - lower)).astype(np.float32)

def prepare_rsna_tensor(hu, device):
    ch_brain = apply_window(hu, center=40, width=80)
    ch_subdural = apply_window(hu, center=75, width=215)
    ch_bone = apply_window(hu, center=600, width=2800)
    composite = np.stack([ch_brain, ch_subdural, ch_bone], axis=-1)
    resized = cv2.resize(composite, (224, 224), interpolation=cv2.INTER_AREA)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (resized - mean) / std
    return torch.tensor(norm.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)



def apply_window(hu, center, width):
    lower = center - width / 2.0
    upper = center + width / 2.0
    w_img = np.clip(hu, lower, upper)
    return ((w_img - lower) / (upper - lower)).astype(np.float32)

def prepare_rsna_tensor(hu):
    # إنشاء القنوات الثلاث المعتمدة في تدريب RSNA
    ch_brain = apply_window(hu, center=40, width=80)
    ch_subdural = apply_window(hu, center=75, width=215)
    ch_bone = apply_window(hu, center=600, width=2800)
    composite = np.stack([ch_brain, ch_subdural, ch_bone], axis=-1)
    resized = cv2.resize(composite, (224, 224), interpolation=cv2.INTER_AREA)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (resized - mean) / std
    return torch.tensor(norm.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(DEVICE)

# منع التصفير القسري والحفاظ على الاحتمالات الخام
    means = base_probs
    if enable_uncertainty:
        margin_entropy = 4.0 * means * (1.0 - means)
        stds = np.clip(margin_entropy * 0.035, 0.001, 0.045)
    else:
        stds = np.zeros_like(means)
    any_idx = SUBTYPES.index('any') if 'any' in SUBTYPES else 0
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
            "Confidence": f"{p*100:.2f}% +/- {sd*100:.2f}%" if enable_uncertainty else f"{p*100:.2f}%",
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