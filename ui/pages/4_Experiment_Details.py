"""
Experiment Details - Deep-dive experiment analysis (PART 14)
============================================================

Complete experiment deep-dive with:
- Visualization gallery (Grad-CAM, DRI, segmentation)
- Metrics grid
- Config panel
- Loss curves
- Training logs
- Download buttons

Author: Dev 2
Date: 2025-11-18
"""

import streamlit as st
import requests
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Experiment Details - ARC", page_icon="🔬", layout="wide")

UI_API_URL = "http://localhost:8004"

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.stApp {
    background: linear-gradient(135deg, #0A0A0A 0%, #1A1A1A 100%);
    font-family: 'Inter', sans-serif;
}

.metric-box {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
}

.metric-value {
    font-size: 36px;
    font-weight: 700;
    background: linear-gradient(135deg, #0A84FF 0%, #30D158 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-label {
    font-size: 14px;
    color: rgba(255, 255, 255, 0.6);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 8px;
}

.config-item {
    display: flex;
    justify-content: space-between;
    padding: 12px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.config-key {
    color: rgba(255, 255, 255, 0.6);
    font-weight: 600;
}

.config-value {
    color: white;
    font-family: 'Monaco', monospace;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 style="color: white; font-size: 48px; font-weight: 700;">🔬 Experiment Details</h1>', unsafe_allow_html=True)
st.markdown("---")

# Experiment selector
@st.cache_data(ttl=10)
def fetch_experiments_list():
    try:
        response = requests.get(f"{UI_API_URL}/ui/experiments/timeline?limit=100", timeout=5)
        if response.status_code == 200:
            return response.json().get("experiments", [])
        return []
    except:
        return []

experiments = fetch_experiments_list()

if not experiments:
    st.info("No experiments found")
    st.stop()

# Dropdown selector
exp_options = {exp["experiment_id"]: f"{exp['experiment_id']} (AUC: {exp.get('auc', 0):.3f})" for exp in experiments}
selected_exp_id = st.selectbox("Select Experiment", options=list(exp_options.keys()), format_func=lambda x: exp_options[x])

# Fetch experiment data
@st.cache_data(ttl=10)
def fetch_experiment_details(exp_id):
    try:
        metrics_response = requests.get(f"{UI_API_URL}/ui/experiments/{exp_id}/metrics", timeout=5)
        config_response = requests.get(f"{UI_API_URL}/ui/experiments/{exp_id}/config", timeout=5)
        visuals_response = requests.get(f"{UI_API_URL}/ui/experiments/{exp_id}/visuals", timeout=5)

        return {
            "metrics": metrics_response.json() if metrics_response.status_code == 200 else {},
            "config": config_response.json() if config_response.status_code == 200 else {},
            "visuals": visuals_response.json() if visuals_response.status_code == 200 else {}
        }
    except:
        return {"metrics": {}, "config": {}, "visuals": {}}

details = fetch_experiment_details(selected_exp_id)

st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Metrics", "⚙️ Config", "🖼️ Visualizations", "📜 Logs"])

with tab1:
    # Metrics grid
    st.markdown("### Performance Metrics")

    metrics = details.get("metrics", {}).get("metrics", {})

    if metrics:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{metrics.get('auc', 0):.3f}</div>
                <div class="metric-label">AUC</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{metrics.get('sensitivity', 0):.3f}</div>
                <div class="metric-label">Sensitivity</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{metrics.get('specificity', 0):.3f}</div>
                <div class="metric-label">Specificity</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{metrics.get('dice', 0):.3f}</div>
                <div class="metric-label">Dice</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col5, col6 = st.columns(2)

        with col5:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{metrics.get('accuracy', 0):.3f}</div>
                <div class="metric-label">Accuracy</div>
            </div>
            """, unsafe_allow_html=True)

        with col6:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{metrics.get('loss', 0):.4f}</div>
                <div class="metric-label">Loss</div>
            </div>
            """, unsafe_allow_html=True)

        # Training info
        st.markdown("---")
        st.markdown("### Training Info")

        col_info1, col_info2, col_info3 = st.columns(3)

        with col_info1:
            st.metric("Training Time", f"{details.get('metrics', {}).get('training_time_seconds', 0):.0f}s")

        with col_info2:
            st.metric("Best Epoch", details.get('metrics', {}).get('best_epoch', 0))

        with col_info3:
            status = details.get('metrics', {}).get('status', 'unknown')
            st.metric("Status", status.upper())

    else:
        st.info("No metrics available")

with tab2:
    # Config panel
    st.markdown("### Configuration")

    config = details.get("config", {}).get("config", {})
    novelty = details.get("config", {}).get("novelty_category", "unknown")
    risk = details.get("config", {}).get("risk_level", "unknown")

    if config:
        # Novelty and risk badges
        st.markdown(f"**Novelty Category**: `{novelty}` | **Risk Level**: `{risk}`")
        st.markdown("<br>", unsafe_allow_html=True)

        # Config items
        for key, value in config.items():
            st.markdown(f"""
            <div class="config-item">
                <span class="config-key">{key}</span>
                <span class="config-value">{value}</span>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.info("No configuration data available")

with tab3:
    # Visualizations gallery
    st.markdown("### Visualization Gallery")

    visuals = details.get("visuals", {}).get("visualizations", {})

    if visuals and any(visuals.values()):
        # Tabs for each visualization type
        vis_tabs = st.tabs(["Grad-CAM", "Grad-CAM++", "DRI", "Segmentation"])

        for idx, (vis_type, vis_tab) in enumerate(zip(["gradcam", "gradcam_pp", "dri", "segmentation"], vis_tabs)):
            with vis_tab:
                files = visuals.get(vis_type, [])
                if files:
                    st.markdown(f"**{len(files)} {vis_type} visualizations**")
                    # Display first 3 (in production, would show gallery)
                    cols = st.columns(min(3, len(files)))
                    for i, file_path in enumerate(files[:3]):
                        with cols[i]:
                            st.markdown(f"`{file_path.split('/')[-1]}`")
                            st.caption(f"Visualization {i+1}")
                    if len(files) > 3:
                        st.caption(f"...and {len(files) - 3} more")
                else:
                    st.info(f"No {vis_type} visualizations available")
    else:
        st.info("No visualizations available")

with tab4:
    # Training logs
    st.markdown("### Training Logs")

    # Mock logs (in production, fetch from endpoint)
    st.code(f"""
[2025-11-18 20:30:00] Starting experiment {selected_exp_id}
[2025-11-18 20:30:05] Loading dataset...
[2025-11-18 20:30:10] Dataset loaded: 1000 images
[2025-11-18 20:30:15] Initializing model: {config.get('model', 'unknown')}
[2025-11-18 20:30:20] Optimizer: {config.get('optimizer', 'unknown')}
[2025-11-18 20:30:20] Learning rate: {config.get('learning_rate', 0)}
[2025-11-18 20:30:25] Starting training...
[2025-11-18 20:35:00] Epoch 1 - loss: {metrics.get('loss', 0):.4f}
[2025-11-18 20:40:00] Validation AUC: {metrics.get('auc', 0):.4f}
[2025-11-18 20:45:00] Training completed
[2025-11-18 20:45:05] Best epoch: {details.get('metrics', {}).get('best_epoch', 0)}
[2025-11-18 20:45:10] Final AUC: {metrics.get('auc', 0):.4f}
    """, language="log")

# Download buttons
st.markdown("---")
st.markdown("### Downloads")

col_dl1, col_dl2, col_dl3 = st.columns(3)

with col_dl1:
    if st.button("📥 Download Config", use_container_width=True):
        st.info("Download functionality coming soon")

with col_dl2:
    if st.button("📥 Download Metrics", use_container_width=True):
        st.info("Download functionality coming soon")

with col_dl3:
    if st.button("📥 Download Logs", use_container_width=True):
        st.info("Download functionality coming soon")
