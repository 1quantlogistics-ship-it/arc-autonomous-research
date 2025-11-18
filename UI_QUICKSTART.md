# ARC Mission Control - Quick Start Guide

## 🚀 Launch the UI (Easiest Method)

```bash
cd /Users/bengibson/Desktop/ARC/arc_clean
./launch_ui.sh
```

Then open your browser to: **http://localhost:8501**

To stop all services:
```bash
./stop_ui.sh
```

---

## 📋 What You'll See

### **Mission Control Dashboard** (Home)
- System status badge (AUTO/SEMI/IDLE)
- GPU health bars with temperature
- Active job count
- Best experiment card
- CPU/RAM usage
- Recent agent activity

### **Navigation** (Left Sidebar)
Click the **>** arrow in the top-left to open the sidebar, then navigate to:

1. **🏠 Mission Control** - Home screen
2. **📊 Live Training** - Real-time training job monitoring
3. **📅 Experiment Timeline** - All experiment history
4. **💬 Cognition Feed** - Agent decision messages (iMessage-style)
5. **🔬 Experiment Details** - Deep-dive analysis
6. **⚙️ System Health** - Comprehensive system monitoring

---

## 🔧 Manual Launch (Alternative)

If you prefer to launch services individually:

### **Terminal 1: UI State Poller**
```bash
python3 api/ui_state_poller.py
```
This aggregates data from all endpoints and caches it.

### **Terminal 2: Control Plane** (Optional)
```bash
python3 api/control_plane.py
```
This provides the backend API for ARC operations.

### **Terminal 3: Mission Control**
```bash
streamlit run ui/mission_control.py
```
This launches the main dashboard.

Then visit: **http://localhost:8501**

---

## 📊 What Each Page Does

### **1. Mission Control** (Home)
- **Purpose**: Main overview screen
- **Refresh**: Every 2 seconds
- **Shows**: System status, GPU health, jobs, best experiment, recent activity

### **2. Live Training**
- **Purpose**: Monitor active training jobs in real-time
- **Refresh**: Every 2 seconds
- **Shows**: Progress ring, loss curves, AUC curves, GPU usage, logs

### **3. Experiment Timeline**
- **Purpose**: Browse all experiment history
- **Features**: Filter by novelty, sort by AUC/timestamp
- **Shows**: Experiment cards with AUC badges

### **4. Cognition Feed**
- **Purpose**: Watch agent decisions in real-time (like watching ARC "think")
- **Refresh**: Every 3 seconds
- **Shows**: iMessage-style chat with agent decisions, color-coded by agent

### **5. Experiment Details**
- **Purpose**: Deep-dive into a specific experiment
- **Shows**: Metrics, config, visualizations, logs
- **Features**: Download buttons, compare mode

### **6. System Health**
- **Purpose**: Monitor system resources and jobs
- **Refresh**: Every 5 seconds
- **Shows**: CPU/RAM/Disk, GPU grid with graphs, job queue, throughput

---

## 🎨 Design Features

All pages feature:
- **Apple-style design** (glass panels, gradients, smooth animations)
- **Dark mode** (black background with subtle highlights)
- **Real-time updates** (auto-refresh without flicker)
- **Responsive layout** (works on laptop/desktop)
- **Zero cognitive overload** (clean, minimal, obvious)

---

## ⚙️ Configuration

Edit URLs in each page if running on different ports:

```python
# In ui/mission_control.py and ui/pages/*.py
UI_API_URL = "http://localhost:8004"  # State poller
CONTROL_PLANE_URL = "http://localhost:8002"  # Control plane
```

For RunPod deployment, change to:
```python
UI_API_URL = "http://arc-api:8004"
CONTROL_PLANE_URL = "http://arc-control:8002"
```

---

## 🐛 Troubleshooting

### **"Connection refused" errors**
- Make sure UI State Poller is running: `python3 api/ui_state_poller.py`
- Check if port 8004 is accessible: `curl http://localhost:8004/ui/dashboard/state`

### **"No active jobs" message**
- This is normal if no training is running
- Start a training job via the Control Plane to see Live Training View

### **Page won't load**
- Check logs: `tail -f logs/dashboard.log`
- Restart services: `./stop_ui.sh && ./launch_ui.sh`

### **Port already in use**
- Kill existing processes: `./stop_ui.sh`
- Or manually: `lsof -ti:8501 | xargs kill -9`

---

## 📂 File Structure

```
ui/
├── mission_control.py          # Home screen (PART 10)
└── pages/
    ├── 1_Live_Training.py      # Live training monitor (PART 11)
    ├── 2_Experiment_Timeline.py # Experiment history (PART 12)
    ├── 3_Cognition_Feed.py     # Agent decisions (PART 13)
    ├── 4_Experiment_Details.py # Experiment deep-dive (PART 14)
    └── 5_System_Health.py      # System monitoring (PART 15)

api/
├── ui_endpoints.py             # 8 REST endpoints (PART 8)
└── ui_state_poller.py          # Background aggregator (PART 9)
```

---

## 🚀 Next Steps

1. **Launch the UI**: `./launch_ui.sh`
2. **Open browser**: http://localhost:8501
3. **Explore pages**: Use sidebar to navigate
4. **Monitor system**: See GPU health, jobs, experiments
5. **Watch cognition**: Go to Cognition Feed to see agent decisions
6. **Start training**: Launch a job via Control Plane to see Live Training View

---

## 💡 Tips

- **Sidebar navigation**: Click `>` in top-left to open/close
- **Auto-refresh**: Pages update automatically, no need to reload
- **CPU-only mode**: UI works fine without GPUs (shows "No GPUs detected")
- **Keyboard shortcuts**: Streamlit supports standard browser shortcuts
- **Full screen**: Press F11 for immersive experience

---

**Enjoy your Apple-grade Mission Control for autonomous AI research! 🚀**
