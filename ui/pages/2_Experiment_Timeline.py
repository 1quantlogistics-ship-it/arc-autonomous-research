"""
Experiment Timeline - Chronological experiment history (PART 12)
================================================================

Beautiful scrollable timeline of all experiments with:
- Horizontal cards (iOS Photos-style)
- Thumbnail visualizations
- AUC badges
- Filter by novelty (exploit/explore/wildcat)
- Sort by timestamp/AUC
- Click → navigate to details

Author: Dev 2
Date: 2025-11-18
"""

import streamlit as st
import requests
from datetime import datetime

st.set_page_config(page_title="Experiment Timeline - ARC", page_icon="📅", layout="wide")

UI_API_URL = "http://localhost:8004"

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.stApp {
    background: linear-gradient(135deg, #0A0A0A 0%, #1A1A1A 100%);
    font-family: 'Inter', sans-serif;
}

.experiment-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(20px);
    border-radius: 16px;
    padding: 20px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    transition: all 0.3s ease;
    cursor: pointer;
}

.experiment-card:hover {
    background: rgba(255, 255, 255, 0.08);
    transform: translateY(-4px);
    box-shadow: 0 12px 48px rgba(0, 0, 0, 0.6);
}

.auc-badge {
    font-size: 32px;
    font-weight: 700;
    background: linear-gradient(135deg, #0A84FF 0%, #30D158 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.novelty-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.novelty-exploit {
    background: #30D158;
    color: #0A0A0A;
}

.novelty-explore {
    background: #0A84FF;
    color: white;
}

.novelty-wildcat {
    background: #FF9F0A;
    color: #0A0A0A;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 style="color: white; font-size: 48px; font-weight: 700;">📅 Experiment Timeline</h1>', unsafe_allow_html=True)
st.markdown("---")

# Fetch experiments
@st.cache_data(ttl=5)
def fetch_experiments(limit=50):
    try:
        response = requests.get(f"{UI_API_URL}/ui/experiments/timeline?limit={limit}", timeout=5)
        if response.status_code == 200:
            return response.json().get("experiments", [])
        return []
    except:
        return []

# Filters
col_filter1, col_filter2, col_filter3 = st.columns([1, 1, 1])

with col_filter1:
    novelty_filter = st.selectbox("Filter by Novelty", ["All", "exploit", "explore", "wildcat"])

with col_filter2:
    sort_by = st.selectbox("Sort by", ["Timestamp (Recent)", "AUC (Highest)", "AUC (Lowest)"])

with col_filter3:
    limit = st.slider("Show experiments", 10, 100, 50, 10)

# Fetch data
experiments = fetch_experiments(limit)

# Apply filters
if novelty_filter != "All":
    experiments = [exp for exp in experiments if exp.get("novelty_category") == novelty_filter]

# Apply sorting
if sort_by == "AUC (Highest)":
    experiments = sorted(experiments, key=lambda x: x.get("auc", 0), reverse=True)
elif sort_by == "AUC (Lowest)":
    experiments = sorted(experiments, key=lambda x: x.get("auc", 0))
# Default: timestamp (already sorted)

st.markdown(f"**Showing {len(experiments)} experiments**")
st.markdown("---")

# Display experiments in grid
if not experiments:
    st.info("No experiments found")
else:
    # 3-column grid
    cols = st.columns(3)

    for idx, exp in enumerate(experiments):
        exp_id = exp.get("experiment_id", "N/A")
        auc = exp.get("auc", 0.0)
        novelty = exp.get("novelty_category", "unknown")
        timestamp = exp.get("timestamp", "")
        status = exp.get("status", "unknown")

        # Format timestamp
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", ""))
                time_str = dt.strftime("%b %d, %H:%M")
            except:
                time_str = "Unknown"
        else:
            time_str = "Unknown"

        # Select column
        col = cols[idx % 3]

        with col:
            # Card
            novelty_class = f"novelty-{novelty}"

            st.markdown(f"""
            <div class="experiment-card">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;">
                    <span class="novelty-badge {novelty_class}">{novelty}</span>
                    <div class="auc-badge">{auc:.3f}</div>
                </div>
                <div style="color: rgba(255, 255, 255, 0.9); font-weight: 600; margin-bottom: 4px;">
                    {exp_id}
                </div>
                <div style="color: rgba(255, 255, 255, 0.5); font-size: 12px;">
                    {time_str} • {status}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Click to view details (would navigate to details page)
            if st.button("View Details", key=f"btn_{exp_id}", use_container_width=True):
                st.info(f"Navigate to details page for {exp_id} (coming in PART 14)")

            st.markdown("<br>", unsafe_allow_html=True)
