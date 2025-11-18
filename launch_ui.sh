#!/bin/bash

# ARC Mission Control Launch Script
# Starts all necessary services for the UI

echo "🚀 Launching ARC Mission Control..."
echo ""

# Check if we're in the right directory
if [ ! -f "ui/mission_control.py" ]; then
    echo "❌ Error: Please run this script from the ARC root directory"
    exit 1
fi

# Kill any existing processes on these ports
echo "🧹 Cleaning up existing processes..."
lsof -ti:8002 | xargs kill -9 2>/dev/null || true
lsof -ti:8004 | xargs kill -9 2>/dev/null || true
lsof -ti:8501 | xargs kill -9 2>/dev/null || true

sleep 2

echo ""
echo "📡 Starting UI State Poller (port 8004)..."
python3 api/ui_state_poller.py > logs/ui_poller.log 2>&1 &
UI_POLLER_PID=$!

sleep 3

echo "🎛️  Starting Control Plane (port 8002)..."
python3 api/control_plane.py > logs/control_plane.log 2>&1 &
CONTROL_PLANE_PID=$!

sleep 3

echo "🎨 Starting Mission Control Dashboard (port 8501)..."
streamlit run ui/mission_control.py --server.port 8501 > logs/dashboard.log 2>&1 &
DASHBOARD_PID=$!

sleep 5

echo ""
echo "✅ ARC Mission Control is running!"
echo ""
echo "📊 Services:"
echo "   • UI State Poller:  http://localhost:8004  (PID: $UI_POLLER_PID)"
echo "   • Control Plane:    http://localhost:8002  (PID: $CONTROL_PLANE_PID)"
echo "   • Mission Control:  http://localhost:8501  (PID: $DASHBOARD_PID)"
echo ""
echo "🌐 Open your browser to: http://localhost:8501"
echo ""
echo "📜 Logs are in:"
echo "   • logs/ui_poller.log"
echo "   • logs/control_plane.log"
echo "   • logs/dashboard.log"
echo ""
echo "⏹️  To stop all services, run: ./stop_ui.sh"
echo ""

# Save PIDs for cleanup
echo "$UI_POLLER_PID" > /tmp/arc_ui_poller.pid
echo "$CONTROL_PLANE_PID" > /tmp/arc_control_plane.pid
echo "$DASHBOARD_PID" > /tmp/arc_dashboard.pid

# Wait for user to exit
echo "Press Ctrl+C to stop all services..."
trap "echo ''; echo '🛑 Stopping services...'; kill $UI_POLLER_PID $CONTROL_PLANE_PID $DASHBOARD_PID 2>/dev/null; echo '✅ All services stopped'; exit 0" INT

# Keep script running
wait
