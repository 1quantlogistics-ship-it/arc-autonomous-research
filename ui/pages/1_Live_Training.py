"""
Live Training View - Real-time job monitoring (PART 11)
========================================================

Real-time monitoring of active training jobs with:
- Animated progress ring (watchOS-style)
- Live loss curves (Plotly)
- GPU usage charts
- Training logs (auto-scroll)
- Action buttons (Cancel/Abort/Restart)

Author: Dev 2
Date: 2025-11-18
"""

import streamlit as st
import requests
import plotly.graph_objects as go
from datetime import datetime
import time

# Page config
st.set_page_config(
    page_title="Live Training - ARC",
    page_icon="📊",
    layout="wide"
)

# Configuration
UI_API_URL = "http://localhost:8004"
CONTROL_PLANE_URL = "http://localhost:8002"

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.stApp {
    background: linear-gradient(135deg, #0A0A0A 0%, #1A1A1A 100%);
    font-family: 'Inter', -apple-system, sans-serif;
}

.progress-ring {
    width: 200px;
    height: 200px;
    margin: 0 auto;
}

.metric-large {
    font-size: 64px;
    font-weight: 700;
    background: linear-gradient(135deg, #0A84FF 0%, #30D158 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
}

.metric-label {
    font-size: 14px;
    color: rgba(255, 255, 255, 0.6);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 8px;
}

.log-container {
    background: rgba(255, 255, 255, 0.03);
    border-radius: 12px;
    padding: 16px;
    font-family: 'Monaco', 'Courier New', monospace;
    font-size: 12px;
    max-height: 400px;
    overflow-y: auto;
    color: rgba(255, 255, 255, 0.8);
}

.status-running {
    color: #30D158;
}

.status-failed {
    color: #FF453A;
}

.status-completed {
    color: #0A84FF;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 style="color: white; font-size: 48px; font-weight: 700;">📊 Live Training</h1>', unsafe_allow_html=True)
st.markdown("---")

# Fetch active jobs
@st.cache_data(ttl=2)
def fetch_active_jobs():
    """Fetch list of active jobs."""
    try:
        response = requests.get(f"{UI_API_URL}/ui/dashboard/state", timeout=5)
        if response.status_code == 200:
            state = response.json()
            active_jobs = state.get("jobs", {}).get("active", [])
            return active_jobs
        return []
    except:
        return []

@st.cache_data(ttl=2)
def fetch_job_progress(job_id: str):
    """Fetch live progress for specific job."""
    try:
        response = requests.get(f"{UI_API_URL}/ui/jobs/{job_id}/progress", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

# Get active jobs
active_jobs = fetch_active_jobs()

if not active_jobs:
    st.info("No active training jobs. Start a new experiment to monitor progress here.")
    st.stop()

# Job selector
job_options = {job["job_id"]: f"{job['experiment_id']} (Progress: {job.get('progress', 0)*100:.1f}%)" for job in active_jobs}
selected_job_id = st.selectbox("Select Job", options=list(job_options.keys()), format_func=lambda x: job_options[x])

# Fetch job progress
progress_data = fetch_job_progress(selected_job_id)

if not progress_data:
    st.error(f"Failed to fetch progress for job {selected_job_id}")
    st.stop()

# Extract data
status = progress_data.get("status", "unknown")
current_epoch = progress_data.get("current_epoch", 0)
total_epochs = progress_data.get("total_epochs", 1)
progress_percent = progress_data.get("progress_percent", 0.0)
loss_curve = progress_data.get("loss_curve", [])
val_auc_curve = progress_data.get("val_auc_curve", [])
eta_seconds = progress_data.get("eta_seconds", 0)
gpu_usage = progress_data.get("gpu_usage", 0.0)

# Layout: 2 columns
col1, col2 = st.columns([1, 2])

with col1:
    # Progress Ring
    st.markdown("### Progress")

    fig_progress = go.Figure(go.Indicator(
        mode="gauge+number",
        value=progress_percent,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Epoch {current_epoch}/{total_epochs}"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "#0A84FF"},
            'bgcolor': "rgba(255, 255, 255, 0.1)",
            'borderwidth': 2,
            'bordercolor': "rgba(255, 255, 255, 0.2)",
            'steps': [
                {'range': [0, 100], 'color': "rgba(255, 255, 255, 0.05)"}
            ],
            'threshold': {
                'line': {'color': "#30D158", 'width': 4},
                'thickness': 0.75,
                'value': progress_percent
            }
        }
    ))

    fig_progress.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "white", 'family': "Inter"},
        height=300
    )

    st.plotly_chart(fig_progress, use_container_width=True)

    # Status & ETA
    st.markdown(f"**Status**: <span class='status-{status}'>{status.upper()}</span>", unsafe_allow_html=True)

    eta_minutes = eta_seconds // 60
    eta_seconds_remainder = eta_seconds % 60
    st.markdown(f"**ETA**: {eta_minutes}m {eta_seconds_remainder}s")

    # GPU Usage
    st.markdown("### GPU Usage")
    st.progress(gpu_usage / 100.0)
    st.caption(f"{gpu_usage:.1f}%")

    # Action Buttons
    st.markdown("### Actions")

    col_btn1, col_btn2, col_btn3 = st.columns(3)

    with col_btn1:
        if st.button("⏸️ Pause", use_container_width=True):
            st.warning("Pause functionality coming soon")

    with col_btn2:
        if st.button("🛑 Cancel", use_container_width=True):
            st.error("Cancel functionality coming soon")

    with col_btn3:
        if st.button("🔄 Restart", use_container_width=True):
            st.info("Restart functionality coming soon")

with col2:
    # Loss Curve
    st.markdown("### Loss Curve")

    if loss_curve:
        fig_loss = go.Figure()

        fig_loss.add_trace(go.Scatter(
            y=loss_curve,
            mode='lines',
            name='Training Loss',
            line=dict(color='#FF9F0A', width=3),
            fill='tozeroy',
            fillcolor='rgba(255, 159, 10, 0.1)'
        ))

        fig_loss.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(255, 255, 255, 0.03)',
            font={'color': "white", 'family': "Inter"},
            xaxis={'title': 'Epoch', 'gridcolor': 'rgba(255, 255, 255, 0.1)'},
            yaxis={'title': 'Loss', 'gridcolor': 'rgba(255, 255, 255, 0.1)'},
            height=250,
            margin=dict(l=0, r=0, t=0, b=0),
            hovermode='x unified'
        )

        st.plotly_chart(fig_loss, use_container_width=True)
    else:
        st.info("No loss data yet")

    # Validation AUC Curve
    st.markdown("### Validation AUC")

    if val_auc_curve:
        fig_auc = go.Figure()

        fig_auc.add_trace(go.Scatter(
            y=val_auc_curve,
            mode='lines+markers',
            name='Validation AUC',
            line=dict(color='#30D158', width=3),
            marker=dict(size=8, color='#30D158', symbol='circle'),
            fill='tozeroy',
            fillcolor='rgba(48, 209, 88, 0.1)'
        ))

        fig_auc.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(255, 255, 255, 0.03)',
            font={'color': "white", 'family': "Inter"},
            xaxis={'title': 'Epoch', 'gridcolor': 'rgba(255, 255, 255, 0.1)'},
            yaxis={'title': 'AUC', 'gridcolor': 'rgba(255, 255, 255, 0.1)', 'range': [0.5, 1.0]},
            height=250,
            margin=dict(l=0, r=0, t=0, b=0),
            hovermode='x unified'
        )

        st.plotly_chart(fig_auc, use_container_width=True)
    else:
        st.info("No validation data yet")

# Training Logs (full width)
st.markdown("---")
st.markdown("### Training Logs")

# Mock logs (in production, fetch from endpoint)
logs = f"""
[{datetime.now().strftime('%H:%M:%S')}] Starting training job {selected_job_id}
[{datetime.now().strftime('%H:%M:%S')}] Loading dataset...
[{datetime.now().strftime('%H:%M:%S')}] Dataset loaded: 1000 images
[{datetime.now().strftime('%H:%M:%S')}] Initializing model: efficientnet_b3
[{datetime.now().strftime('%H:%M:%S')}] Epoch 1/{total_epochs} - loss: {loss_curve[0]:.4f if loss_curve else 0.0}
[{datetime.now().strftime('%H:%M:%S')}] Epoch {current_epoch}/{total_epochs} - loss: {loss_curve[-1]:.4f if loss_curve else 0.0}
[{datetime.now().strftime('%H:%M:%S')}] Validation AUC: {val_auc_curve[-1]:.4f if val_auc_curve else 0.0}
[{datetime.now().strftime('%H:%M:%S')}] GPU Memory: {gpu_usage:.1f}%
[{datetime.now().strftime('%H:%M:%S')}] ETA: {eta_minutes}m {eta_seconds_remainder}s
"""

st.markdown(f'<div class="log-container">{logs}</div>', unsafe_allow_html=True)

# Auto-refresh
time.sleep(2)
st.rerun()
