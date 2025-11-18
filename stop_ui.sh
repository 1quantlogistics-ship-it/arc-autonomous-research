#!/bin/bash

# ARC Mission Control Stop Script
# Stops all UI services

echo "🛑 Stopping ARC Mission Control..."

# Kill processes by PID files if they exist
if [ -f /tmp/arc_ui_poller.pid ]; then
    kill $(cat /tmp/arc_ui_poller.pid) 2>/dev/null && echo "✅ UI State Poller stopped"
    rm /tmp/arc_ui_poller.pid
fi

if [ -f /tmp/arc_control_plane.pid ]; then
    kill $(cat /tmp/arc_control_plane.pid) 2>/dev/null && echo "✅ Control Plane stopped"
    rm /tmp/arc_control_plane.pid
fi

if [ -f /tmp/arc_dashboard.pid ]; then
    kill $(cat /tmp/arc_dashboard.pid) 2>/dev/null && echo "✅ Mission Control stopped"
    rm /tmp/arc_dashboard.pid
fi

# Also kill any processes on these ports (cleanup)
lsof -ti:8002 | xargs kill -9 2>/dev/null || true
lsof -ti:8004 | xargs kill -9 2>/dev/null || true
lsof -ti:8501 | xargs kill -9 2>/dev/null || true

echo ""
echo "✅ All ARC services stopped"
