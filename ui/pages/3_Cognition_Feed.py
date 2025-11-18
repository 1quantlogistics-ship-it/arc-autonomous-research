"""
Multi-Agent Cognition Feed - iMessage-style agent chat (PART 13)
================================================================

Real-time feed of agent decisions in iMessage style with:
- Chat bubbles (colored by agent)
- Agent avatars (circles with initials)
- Decision messages with metadata
- Expandable details
- Search/filter by agent
- Auto-scroll to latest

Author: Dev 2
Date: 2025-11-18
"""

import streamlit as st
import requests
from datetime import datetime

st.set_page_config(page_title="Cognition Feed - ARC", page_icon="💬", layout="wide")

UI_API_URL = "http://localhost:8004"

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.stApp {
    background: linear-gradient(135deg, #0A0A0A 0%, #1A1A1A 100%);
    font-family: 'Inter', sans-serif;
}

.message-bubble {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 16px 20px;
    margin: 12px 0;
    border-left: 4px solid;
    transition: all 0.3s ease;
}

.message-bubble:hover {
    background: rgba(255, 255, 255, 0.08);
    transform: translateX(4px);
}

.agent-director {
    border-left-color: #0A84FF;
}

.agent-supervisor {
    border-left-color: #FF453A;
}

.agent-architect {
    border-left-color: #30D158;
}

.agent-world-model {
    border-left-color: #FFD60A;
}

.agent-historian {
    border-left-color: #FF9F0A;
}

.agent-avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 16px;
    margin-right: 12px;
}

.agent-name {
    font-weight: 600;
    font-size: 14px;
    color: white;
}

.message-text {
    color: rgba(255, 255, 255, 0.9);
    font-size: 15px;
    line-height: 1.5;
    margin-top: 8px;
}

.message-time {
    font-size: 12px;
    color: rgba(255, 255, 255, 0.4);
    margin-top: 8px;
}

.metadata-box {
    background: rgba(255, 255, 255, 0.03);
    border-radius: 8px;
    padding: 12px;
    margin-top: 12px;
    font-size: 13px;
    color: rgba(255, 255, 255, 0.7);
    font-family: 'Monaco', monospace;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 style="color: white; font-size: 48px; font-weight: 700;">💬 Agent Cognition Feed</h1>', unsafe_allow_html=True)
st.markdown('<p style="color: rgba(255, 255, 255, 0.6); font-size: 16px;">Watch ARC think in real-time</p>', unsafe_allow_html=True)
st.markdown("---")

# Fetch cognition feed
@st.cache_data(ttl=2)
def fetch_cognition_feed(limit=50):
    try:
        response = requests.get(f"{UI_API_URL}/ui/agents/cognition/feed?limit={limit}", timeout=5)
        if response.status_code == 200:
            return response.json().get("decisions", [])
        return []
    except:
        return []

# Filters
col1, col2 = st.columns([3, 1])

with col1:
    agent_filter = st.selectbox(
        "Filter by Agent",
        ["All", "Director", "Supervisor", "Architect", "World-Model", "Historian", "Explorer", "Critic"]
    )

with col2:
    limit = st.slider("Show messages", 10, 100, 50, 10)

# Fetch data
decisions = fetch_cognition_feed(limit)

# Apply filter
if agent_filter != "All":
    decisions = [d for d in decisions if d.get("agent") == agent_filter]

st.markdown(f"**Showing {len(decisions)} decisions**")
st.markdown("---")

# Agent colors and initials
AGENT_COLORS = {
    "Director": "#0A84FF",
    "Supervisor": "#FF453A",
    "Architect": "#30D158",
    "World-Model": "#FFD60A",
    "Historian": "#FF9F0A",
    "Explorer": "#30D158",
    "Critic": "#FF9F0A"
}

AGENT_INITIALS = {
    "Director": "DR",
    "Supervisor": "SV",
    "Architect": "AR",
    "World-Model": "WM",
    "Historian": "HS",
    "Explorer": "EX",
    "Critic": "CR"
}

# Display messages
if not decisions:
    st.info("No agent decisions yet. Start ARC AUTO mode to see cognition feed.")
else:
    for decision in decisions:
        agent = decision.get("agent", "Unknown")
        action = decision.get("action", "")
        message = decision.get("message", "")
        timestamp = decision.get("timestamp", "")
        metadata = decision.get("metadata", {})

        # Format timestamp
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", ""))
                time_str = dt.strftime("%H:%M:%S")
            except:
                time_str = ""
        else:
            time_str = ""

        # Get agent color and initials
        agent_color = AGENT_COLORS.get(agent, "#FFFFFF")
        agent_initials = AGENT_INITIALS.get(agent, "??")

        # Agent class for border color
        agent_class = f"agent-{agent.lower().replace('-', '')}"

        # Render message bubble
        st.markdown(f"""
        <div class="message-bubble {agent_class}">
            <div style="display: flex; align-items: center;">
                <div class="agent-avatar" style="background: {agent_color};">
                    {agent_initials}
                </div>
                <div>
                    <div class="agent-name">{agent}</div>
                    <div class="message-time">{time_str}</div>
                </div>
            </div>
            <div class="message-text">{message}</div>
        </div>
        """, unsafe_allow_html=True)

        # Expandable metadata
        if metadata:
            with st.expander("Show metadata"):
                st.json(metadata)

        st.markdown("<br>", unsafe_allow_html=True)

# Auto-refresh
import time
time.sleep(3)
st.rerun()
