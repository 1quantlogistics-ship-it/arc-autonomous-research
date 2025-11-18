"""
ARC Mission Control Dashboard (PART 10)
========================================

Top-level Apple-style dashboard for ARC autonomous research platform.

Design Philosophy:
- Apple-simple and elegant
- Glowing gradients, soft rounded cards, blurred glass panels
- Large readable typography
- Smooth transitions and animations
- Zero cognitive overload

Components:
1. ARC Status (AUTO/SEMI/IDLE)
2. Active cycle number
3. Best experiment card
4. GPU health indicators
5. Job queue summary
6. Quick actions (start/stop/pause)

Author: Dev 2
Date: 2025-11-18
"""

import streamlit as st
import requests
import time
from datetime import datetime
import plotly.graph_objects as go
from typing import Dict, Any

# ============================================================
# Configuration
# ============================================================

UI_API_URL = "http://localhost:8004"  # State poller endpoint
CONTROL_PLANE_URL = "http://localhost:8002"  # Control plane endpoint

# ============================================================
# Custom CSS - Apple-like Design
# ============================================================

CUSTOM_CSS = """
<style>
/* Global styles */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0A0A0A 0%, #1A1A1A 100%);
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Glass panel card */
.glass-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(20px);
    border-radius: 24px;
    padding: 32px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    transition: all 0.3s ease;
}

.glass-card:hover {
    background: rgba(255, 255, 255, 0.08);
    box-shadow: 0 12px 48px rgba(0, 0, 0, 0.6);
    transform: translateY(-2px);
}

/* Gradient text */
.gradient-text {
    background: linear-gradient(135deg, #0A84FF 0%, #30D158 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 48px;
    font-weight: 700;
    letter-spacing: -1px;
}

/* Status badge */
.status-badge {
    display: inline-block;
    padding: 8px 16px;
    border-radius: 12px;
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

.status-auto {
    background: linear-gradient(135deg, #0A84FF 0%, #30D158 100%);
    color: white;
}

.status-semi {
    background: linear-gradient(135deg, #FF9F0A 0%, #FFD60A 100%);
    color: #0A0A0A;
}

.status-idle {
    background: rgba(255, 255, 255, 0.1);
    color: rgba(255, 255, 255, 0.5);
}

/* Metric card */
.metric-card {
    text-align: center;
    padding: 24px;
}

.metric-value {
    font-size: 48px;
    font-weight: 700;
    color: #FFFFFF;
    line-height: 1;
}

.metric-label {
    font-size: 14px;
    font-weight: 500;
    color: rgba(255, 255, 255, 0.6);
    margin-top: 8px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* GPU bar */
.gpu-bar {
    height: 8px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    overflow: hidden;
    margin: 12px 0;
}

.gpu-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #30D158 0%, #0A84FF 100%);
    border-radius: 4px;
    transition: width 0.5s ease;
}

/* Button */
.arc-button {
    background: linear-gradient(135deg, #0A84FF 0%, #30D158 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 24px;
    font-size: 16px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 16px rgba(10, 132, 255, 0.3);
}

.arc-button:hover {
    box-shadow: 0 6px 24px rgba(10, 132, 255, 0.5);
    transform: translateY(-2px);
}

/* Experiment card */
.experiment-card {
    background: rgba(255, 255, 255, 0.03);
    border-radius: 16px;
    padding: 20px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    margin: 12px 0;
}

.experiment-auc {
    font-size: 32px;
    font-weight: 700;
    background: linear-gradient(135deg, #0A84FF 0%, #30D158 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Section header */
.section-header {
    font-size: 24px;
    font-weight: 600;
    color: #FFFFFF;
    margin-bottom: 16px;
    letter-spacing: -0.5px;
}

/* Pulse animation for active state */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
}

.pulse {
    animation: pulse 2s ease-in-out infinite;
}
</style>
"""

# ============================================================
# Helper Functions
# ============================================================

def fetch_dashboard_state() -> Dict[str, Any]:
    """Fetch aggregated dashboard state from UI State Poller."""
    try:
        response = requests.get(f"{UI_API_URL}/ui/dashboard/state", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return get_fallback_state()
    except Exception as e:
        st.error(f"Failed to fetch dashboard state: {e}")
        return get_fallback_state()

def get_fallback_state() -> Dict[str, Any]:
    """Fallback state if API unavailable."""
    return {
        "system": {
            "cpu": {"usage_percent": 0.0, "cores": 0},
            "ram": {"used_gb": 0.0, "total_gb": 0.0, "percent": 0.0},
            "gpu": [],
            "uptime_seconds": 0
        },
        "jobs": {
            "summary": {
                "active_count": 0,
                "queued_count": 0,
                "completed_today": 0,
                "failed_today": 0
            }
        },
        "timeline": {
            "experiments": [],
            "total_count": 0
        },
        "cognition": {
            "decisions": []
        },
        "meta": {
            "poll_count": 0,
            "last_poll_time": None,
            "error_count": 0
        }
    }

def get_arc_status() -> str:
    """Get ARC operating mode from control plane."""
    try:
        response = requests.get(f"{CONTROL_PLANE_URL}/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("mode", "IDLE")
        else:
            return "IDLE"
    except:
        return "IDLE"

def render_status_badge(status: str):
    """Render status badge with color."""
    status_class = f"status-{status.lower()}"
    return f'<span class="status-badge {status_class}">{status}</span>'

def render_metric_card(value: str, label: str):
    """Render metric card."""
    return f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """

def render_gpu_bar(gpu_id: int, name: str, usage: float, memory_percent: float, temp: int):
    """Render GPU health bar."""
    # Color based on temperature
    if temp < 60:
        color = "#30D158"  # Green
    elif temp < 75:
        color = "#FF9F0A"  # Orange
    else:
        color = "#FF453A"  # Red

    return f"""
    <div style="margin: 16px 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <span style="color: rgba(255, 255, 255, 0.9); font-weight: 600;">GPU {gpu_id}: {name}</span>
            <span style="color: rgba(255, 255, 255, 0.6);">{temp}°C</span>
        </div>
        <div class="gpu-bar">
            <div class="gpu-bar-fill" style="width: {usage}%; background: linear-gradient(90deg, {color} 0%, {color} 100%);"></div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 4px; font-size: 12px;">
            <span style="color: rgba(255, 255, 255, 0.5);">Usage: {usage:.1f}%</span>
            <span style="color: rgba(255, 255, 255, 0.5);">Memory: {memory_percent:.1f}%</span>
        </div>
    </div>
    """

def render_experiment_card(exp: Dict[str, Any]):
    """Render best experiment card."""
    exp_id = exp.get("experiment_id", "N/A")
    auc = exp.get("auc", 0.0)
    timestamp = exp.get("timestamp", "")
    novelty = exp.get("novelty_category", "unknown")

    # Format timestamp
    if timestamp:
        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", ""))
            time_str = dt.strftime("%b %d, %H:%M")
        except:
            time_str = "Unknown"
    else:
        time_str = "Unknown"

    return f"""
    <div class="experiment-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="color: rgba(255, 255, 255, 0.6); font-size: 12px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">
                    {novelty} • {time_str}
                </div>
                <div style="color: rgba(255, 255, 255, 0.9); font-weight: 600; font-size: 14px; margin-bottom: 4px;">
                    {exp_id}
                </div>
            </div>
            <div class="experiment-auc">{auc:.3f}</div>
        </div>
    </div>
    """

# ============================================================
# Main Dashboard
# ============================================================

def main():
    """Render Mission Control Dashboard."""

    # Page config
    st.set_page_config(
        page_title="ARC Mission Control",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Apply custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Header
    st.markdown('<div class="gradient-text">ARC MISSION CONTROL</div>', unsafe_allow_html=True)
    st.markdown("", unsafe_allow_html=True)  # Spacer

    # Fetch state
    state = fetch_dashboard_state()
    arc_status = get_arc_status()

    # Status row
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown(f'<div class="section-header">System Status</div>', unsafe_allow_html=True)
        st.markdown(render_status_badge(arc_status), unsafe_allow_html=True)
        st.markdown(f'<p style="color: rgba(255, 255, 255, 0.6); margin-top: 8px;">Uptime: {state["system"]["uptime_seconds"] // 3600}h {(state["system"]["uptime_seconds"] % 3600) // 60}m</p>', unsafe_allow_html=True)

    with col2:
        st.markdown(f'<div class="section-header">Active Cycle</div>', unsafe_allow_html=True)
        # TODO: Get cycle from control plane
        st.markdown(render_metric_card("—", "Cycle ID"), unsafe_allow_html=True)

    with col3:
        st.markdown(f'<div class="section-header">Last Poll</div>', unsafe_allow_html=True)
        last_poll = state["meta"].get("last_poll_time")
        if last_poll:
            try:
                dt = datetime.fromisoformat(last_poll.replace("Z", ""))
                poll_str = dt.strftime("%H:%M:%S")
            except:
                poll_str = "Unknown"
        else:
            poll_str = "Never"
        st.markdown(f'<p style="color: rgba(255, 255, 255, 0.9); font-size: 18px; font-weight: 600;">{poll_str}</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="color: rgba(255, 255, 255, 0.6);">Poll #{state["meta"]["poll_count"]}</p>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Main content row
    col1, col2 = st.columns([2, 1])

    with col1:
        # GPU Health
        st.markdown('<div class="section-header">GPU Health</div>', unsafe_allow_html=True)

        gpus = state["system"].get("gpu", [])
        if gpus:
            for gpu in gpus:
                gpu_id = gpu.get("id", 0)
                name = gpu.get("name", "Unknown")
                usage = gpu.get("usage_percent", 0.0)
                memory_used = gpu.get("memory_used_gb", 0.0)
                memory_total = gpu.get("memory_total_gb", 1.0)
                memory_percent = (memory_used / memory_total) * 100 if memory_total > 0 else 0
                temp = gpu.get("temp_celsius", 0)

                st.markdown(render_gpu_bar(gpu_id, name, usage, memory_percent, temp), unsafe_allow_html=True)
        else:
            st.markdown('<p style="color: rgba(255, 255, 255, 0.5);">No GPUs detected (CPU-only mode)</p>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Job Queue Summary
        st.markdown('<div class="section-header">Experiment Engine</div>', unsafe_allow_html=True)

        jobs_summary = state["jobs"]["summary"]

        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

        with metric_col1:
            st.markdown(render_metric_card(str(jobs_summary["active_count"]), "Active"), unsafe_allow_html=True)

        with metric_col2:
            st.markdown(render_metric_card(str(jobs_summary["queued_count"]), "Queued"), unsafe_allow_html=True)

        with metric_col3:
            st.markdown(render_metric_card(str(jobs_summary["completed_today"]), "Completed Today"), unsafe_allow_html=True)

        with metric_col4:
            st.markdown(render_metric_card(str(jobs_summary["failed_today"]), "Failed Today"), unsafe_allow_html=True)

    with col2:
        # Best Experiment
        st.markdown('<div class="section-header">Best Experiment</div>', unsafe_allow_html=True)

        experiments = state["timeline"].get("experiments", [])
        if experiments:
            # Find best by AUC
            best_exp = max(experiments, key=lambda x: x.get("auc", 0.0))
            st.markdown(render_experiment_card(best_exp), unsafe_allow_html=True)
        else:
            st.markdown('<p style="color: rgba(255, 255, 255, 0.5);">No experiments yet</p>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # System Resources
        st.markdown('<div class="section-header">System Resources</div>', unsafe_allow_html=True)

        cpu_percent = state["system"]["cpu"]["usage_percent"]
        ram_percent = state["system"]["ram"]["percent"]

        st.markdown(f"""
        <div style="margin: 16px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="color: rgba(255, 255, 255, 0.9); font-weight: 600;">CPU</span>
                <span style="color: rgba(255, 255, 255, 0.6);">{cpu_percent:.1f}%</span>
            </div>
            <div class="gpu-bar">
                <div class="gpu-bar-fill" style="width: {cpu_percent}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="margin: 16px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="color: rgba(255, 255, 255, 0.9); font-weight: 600;">RAM</span>
                <span style="color: rgba(255, 255, 255, 0.6);">{ram_percent:.1f}%</span>
            </div>
            <div class="gpu-bar">
                <div class="gpu-bar-fill" style="width: {ram_percent}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Recent Activity (Agent Cognition Feed Preview)
    st.markdown('<div class="section-header">Recent Activity</div>', unsafe_allow_html=True)

    decisions = state["cognition"].get("decisions", [])[:5]  # Last 5

    if decisions:
        for decision in decisions:
            agent = decision.get("agent", "Unknown")
            message = decision.get("message", "")
            timestamp = decision.get("timestamp", "")

            # Format timestamp
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace("Z", ""))
                    time_str = dt.strftime("%H:%M:%S")
                except:
                    time_str = ""
            else:
                time_str = ""

            st.markdown(f"""
            <div style="background: rgba(255, 255, 255, 0.03); border-radius: 12px; padding: 16px; margin: 8px 0; border-left: 3px solid #0A84FF;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="color: #0A84FF; font-weight: 600; font-size: 14px;">{agent}</span>
                    <span style="color: rgba(255, 255, 255, 0.4); font-size: 12px;">{time_str}</span>
                </div>
                <div style="color: rgba(255, 255, 255, 0.8); font-size: 14px;">{message}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<p style="color: rgba(255, 255, 255, 0.5);">No recent activity</p>', unsafe_allow_html=True)

    # Auto-refresh
    time.sleep(2)
    st.rerun()


if __name__ == "__main__":
    main()
