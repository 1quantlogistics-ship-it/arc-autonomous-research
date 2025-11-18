#!/bin/bash
################################################################################
# Stop vLLM Server Script
################################################################################

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

PID_FILE="/workspace/arc/vllm.pid"

info "Stopping vLLM server..."

# Check PID file
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    
    if ps -p $PID > /dev/null 2>&1; then
        info "Sending SIGTERM to PID $PID"
        kill $PID
        
        # Wait for graceful shutdown
        WAIT=0
        while ps -p $PID > /dev/null 2>&1 && [ $WAIT -lt 30 ]; do
            echo -n "."
            sleep 1
            WAIT=$((WAIT + 1))
        done
        echo ""
        
        if ps -p $PID > /dev/null 2>&1; then
            warning "Process didn't stop, sending SIGKILL"
            kill -9 $PID
            sleep 2
        fi
        
        if ! ps -p $PID > /dev/null 2>&1; then
            success "vLLM server stopped (PID $PID)"
            rm -f "$PID_FILE"
        else
            error "Failed to stop process $PID"
            exit 1
        fi
    else
        warning "PID $PID not running, cleaning up PID file"
        rm -f "$PID_FILE"
    fi
else
    # Try finding by process name
    PIDS=$(pgrep -f 'vllm.entrypoints.openai.api_server')
    
    if [ -n "$PIDS" ]; then
        warning "No PID file, but found running vLLM processes"
        echo "$PIDS" | while read PID; do
            info "Killing PID $PID"
            kill $PID
        done
        sleep 3
        success "Stopped vLLM processes"
    else
        info "No vLLM server running"
    fi
fi

# Verify port is free
if ss -tlnp 2>/dev/null | grep -q ':8000 '; then
    warning "Port 8000 still in use:"
    ss -tlnp | grep ':8000'
else
    success "Port 8000 is free"
fi

info "Done"
