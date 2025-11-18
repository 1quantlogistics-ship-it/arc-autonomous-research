"""
System Health Panel - Comprehensive system monitoring (PART 15)
===============================================================

Complete system health monitoring with:
- GPU grid (per-GPU detailed stats with graphs)
- CPU/RAM/Disk usage graphs
- Job queue timeline
- Experiment throughput
- Alert banners for critical issues

Author: Dev 2
Date: 2025-11-18
"""

import streamlit as st
import requests
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="System Health - ARC", page_icon="⚙️", layout="wide")

UI_API_URL = "http://localhost:8004"

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.stApp {
    background: linear-gradient(135deg, #0A0A0A 0%, #1A1A1A 100%);
    font-family: 'Inter', sans-serif;
}

.health-card {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 24px;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.status-ok {
    color: #30D158;
}

.status-warning {
    color: #FF9F0A;
}

.status-critical {
    color: #FF453A;
}

.alert-banner {
    background: rgba(255, 69, 58, 0.1);
    border-left: 4px solid #FF453A;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 16px 0;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 style="color: white; font-size: 48px; font-weight: 700;">⚙️ System Health</h1>', unsafe_allow_html=True)
st.markdown("---")

# Fetch system health
@st.cache_data(ttl=2)
def fetch_system_health():
    try:
        response = requests.get(f"{UI_API_URL}/ui/system/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {}
    except:
        return {}

@st.cache_data(ttl=2)
def fetch_job_queue():
    try:
        response = requests.get(f"{UI_API_URL}/ui/jobs/queue", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {}
    except:
        return {}

health = fetch_system_health()
job_queue = fetch_job_queue()

# System overview
col_overview1, col_overview2, col_overview3, col_overview4 = st.columns(4)

with col_overview1:
    cpu_percent = health.get("cpu", {}).get("usage_percent", 0)
    cpu_status = "status-ok" if cpu_percent < 70 else ("status-warning" if cpu_percent < 90 else "status-critical")
    st.markdown(f"""
    <div class="health-card">
        <div class="{cpu_status}" style="font-size: 36px; font-weight: 700;">{cpu_percent:.1f}%</div>
        <div style="color: rgba(255, 255, 255, 0.6); margin-top: 8px;">CPU Usage</div>
    </div>
    """, unsafe_allow_html=True)

with col_overview2:
    ram_percent = health.get("ram", {}).get("percent", 0)
    ram_status = "status-ok" if ram_percent < 70 else ("status-warning" if ram_percent < 90 else "status-critical")
    st.markdown(f"""
    <div class="health-card">
        <div class="{ram_status}" style="font-size: 36px; font-weight: 700;">{ram_percent:.1f}%</div>
        <div style="color: rgba(255, 255, 255, 0.6); margin-top: 8px;">RAM Usage</div>
    </div>
    """, unsafe_allow_html=True)

with col_overview3:
    disk_percent = health.get("disk", {}).get("percent", 0)
    disk_status = "status-ok" if disk_percent < 70 else ("status-warning" if disk_percent < 90 else "status-critical")
    st.markdown(f"""
    <div class="health-card">
        <div class="{disk_status}" style="font-size: 36px; font-weight: 700;">{disk_percent:.1f}%</div>
        <div style="color: rgba(255, 255, 255, 0.6); margin-top: 8px;">Disk Usage</div>
    </div>
    """, unsafe_allow_html=True)

with col_overview4:
    uptime_hours = health.get("uptime_seconds", 0) // 3600
    uptime_minutes = (health.get("uptime_seconds", 0) % 3600) // 60
    st.markdown(f"""
    <div class="health-card">
        <div style="font-size: 36px; font-weight: 700; color: white;">{uptime_hours}h {uptime_minutes}m</div>
        <div style="color: rgba(255, 255, 255, 0.6); margin-top: 8px;">Uptime</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Alerts
if cpu_percent > 90 or ram_percent > 90 or disk_percent > 90:
    st.markdown("""
    <div class="alert-banner">
        <strong>⚠️ CRITICAL ALERT</strong><br>
        System resources are critically high. Consider scaling up or clearing disk space.
    </div>
    """, unsafe_allow_html=True)

# GPU Grid
st.markdown("### GPU Health")

gpus = health.get("gpu", [])

if gpus:
    for gpu in gpus:
        gpu_id = gpu.get("id", 0)
        gpu_name = gpu.get("name", "Unknown")
        gpu_usage = gpu.get("usage_percent", 0)
        gpu_memory_used = gpu.get("memory_used_gb", 0)
        gpu_memory_total = gpu.get("memory_total_gb", 1)
        gpu_temp = gpu.get("temp_celsius", 0)

        memory_percent = (gpu_memory_used / gpu_memory_total) * 100 if gpu_memory_total > 0 else 0

        # GPU status
        temp_status = "status-ok" if gpu_temp < 60 else ("status-warning" if gpu_temp < 75 else "status-critical")

        st.markdown(f"#### GPU {gpu_id}: {gpu_name}")

        col_gpu1, col_gpu2, col_gpu3 = st.columns(3)

        with col_gpu1:
            st.metric("Utilization", f"{gpu_usage:.1f}%")
            st.progress(gpu_usage / 100.0)

        with col_gpu2:
            st.metric("Memory", f"{gpu_memory_used:.1f} / {gpu_memory_total:.1f} GB")
            st.progress(memory_percent / 100.0)

        with col_gpu3:
            st.markdown(f"**Temperature**: <span class='{temp_status}'>{gpu_temp}°C</span>", unsafe_allow_html=True)

        # Mock usage graph (in production, show 5-minute history)
        fig_gpu = go.Figure()

        # Mock data
        import numpy as np
        x = list(range(60))
        y = [gpu_usage + np.random.normal(0, 5) for _ in x]

        fig_gpu.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            fill='tozeroy',
            line=dict(color='#0A84FF', width=2),
            fillcolor='rgba(10, 132, 255, 0.1)',
            name='GPU Usage'
        ))

        fig_gpu.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(255, 255, 255, 0.03)',
            font={'color': "white", 'family': "Inter"},
            xaxis={'title': 'Time (seconds ago)', 'gridcolor': 'rgba(255, 255, 255, 0.1)'},
            yaxis={'title': 'Usage %', 'gridcolor': 'rgba(255, 255, 255, 0.1)', 'range': [0, 100]},
            height=200,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False
        )

        st.plotly_chart(fig_gpu, use_container_width=True)

        st.markdown("---")

else:
    st.info("No GPUs detected (CPU-only mode)")

# Job Queue Summary
st.markdown("### Job Queue Status")

jobs = job_queue.get("summary", {})

col_jobs1, col_jobs2, col_jobs3, col_jobs4 = st.columns(4)

with col_jobs1:
    st.metric("Active", jobs.get("active_count", 0))

with col_jobs2:
    st.metric("Queued", jobs.get("queued_count", 0))

with col_jobs3:
    st.metric("Completed Today", jobs.get("completed_today", 0))

with col_jobs4:
    st.metric("Failed Today", jobs.get("failed_today", 0))

# Throughput
st.markdown("### Experiment Throughput")

# Mock throughput data
throughput_hours = [5, 7, 6, 8, 10, 9, 11, 12]  # experiments per hour (last 8 hours)

fig_throughput = go.Figure()

fig_throughput.add_trace(go.Bar(
    x=list(range(len(throughput_hours))),
    y=throughput_hours,
    marker=dict(
        color=throughput_hours,
        colorscale=[[0, '#FF453A'], [0.5, '#FF9F0A'], [1, '#30D158']],
        showscale=False
    ),
    text=throughput_hours,
    textposition='outside'
))

fig_throughput.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(255, 255, 255, 0.03)',
    font={'color': "white", 'family': "Inter"},
    xaxis={'title': 'Hours Ago', 'gridcolor': 'rgba(255, 255, 255, 0.1)'},
    yaxis={'title': 'Experiments/Hour', 'gridcolor': 'rgba(255, 255, 255, 0.1)'},
    height=300,
    margin=dict(l=0, r=0, t=20, b=0)
)

st.plotly_chart(fig_throughput, use_container_width=True)

# Auto-refresh
import time
time.sleep(5)
st.rerun()
