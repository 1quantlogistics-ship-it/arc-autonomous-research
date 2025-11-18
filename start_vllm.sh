#!/bin/bash
################################################################################
# DeepSeek-R1-Distill-Qwen-14B vLLM Server Startup Script
# RunPod 2x A40 GPU Configuration
# 
# VERIFIED WORKING CONFIGURATION (2025-11-18)
################################################################################

set -e  # Exit on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
WORKSPACE_ROOT="/workspace"
ARC_ROOT="${WORKSPACE_ROOT}/arc"
MODEL_PATH="${WORKSPACE_ROOT}/models/DeepSeek-R1-Distill-Qwen-14B"
VENV_PATH="${ARC_ROOT}/venv"
LOG_FILE="${ARC_ROOT}/vllm.log"
PID_FILE="${ARC_ROOT}/vllm.pid"
PORT=8000
HOST="0.0.0.0"
TENSOR_PARALLEL_SIZE=2

################################################################################
# VERIFIED PACKAGE VERSIONS (DO NOT CHANGE)
################################################################################
# torch==2.1.2+cu121
# vllm==0.3.3
# transformers==4.38.0
# numpy==1.26.4
# sentencepiece==0.2.1
# accelerate==1.11.0
# pydantic==2.12.4
# fastapi==0.121.2
# uvicorn==0.38.0
################################################################################

echo -e "${BLUE}=============================================================================="
echo -e "DeepSeek-R1-Distill-Qwen-14B vLLM Server Startup"
echo -e "=============================================================================="
echo -e "${NC}"

# Function to print colored messages
info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

################################################################################
# PRE-FLIGHT CHECKS
################################################################################

info "Running pre-flight checks..."

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    error "nvidia-smi not found. NVIDIA drivers not installed."
    exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
if [ "$GPU_COUNT" -lt 2 ]; then
    error "Found $GPU_COUNT GPU(s), but 2 GPUs required for tensor parallelism."
    exit 1
fi
success "Found $GPU_COUNT GPUs"

# Display GPU info
info "GPU Configuration:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | nl -v 0

# Check model files
if [ ! -d "$MODEL_PATH" ]; then
    error "Model directory not found: $MODEL_PATH"
    exit 1
fi

SAFETENSORS_COUNT=$(ls "$MODEL_PATH"/*.safetensors 2>/dev/null | wc -l)
if [ "$SAFETENSORS_COUNT" -lt 4 ]; then
    error "Expected 4 safetensors files, found $SAFETENSORS_COUNT"
    exit 1
fi
success "Model files verified ($SAFETENSORS_COUNT safetensors files)"

# Check virtual environment
if [ ! -d "$VENV_PATH" ]; then
    error "Virtual environment not found: $VENV_PATH"
    exit 1
fi
success "Virtual environment found"

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Verify vLLM installation
if ! python -c "import vllm" 2>/dev/null; then
    error "vLLM not installed in virtual environment"
    exit 1
fi

VLLM_VERSION=$(python -c "import vllm; print(vllm.__version__)")
success "vLLM version: $VLLM_VERSION"

################################################################################
# CRITICAL FIX: Patch outlines library for broken pyairports dependency
################################################################################

info "Applying critical patches..."

AIRPORTS_FILE="${VENV_PATH}/lib/python3.10/site-packages/outlines/types/airports.py"

if [ -f "$AIRPORTS_FILE" ]; then
    # Check if already patched
    if grep -q "PATCHED: pyairports package is broken" "$AIRPORTS_FILE"; then
        success "outlines library already patched"
    else
        warning "Patching outlines library to fix pyairports import error"
        
        # Create backup
        cp "$AIRPORTS_FILE" "${AIRPORTS_FILE}.backup"
        
        # Apply patch
        cat > "$AIRPORTS_FILE" << 'PATCH_EOF'
"""Generate valid airport codes."""
from enum import Enum

# PATCHED: pyairports package is broken, using empty list
# from pyairports.airports import AIRPORT_LIST
AIRPORT_LIST = []

AIRPORT_IATA_LIST = list(
    {(airport[3], airport[3]) for airport in AIRPORT_LIST if airport[3] != ""}
)

IATA = Enum("Airport", AIRPORT_IATA_LIST or [("NONE", "NONE")])  # type:ignore
PATCH_EOF
        
        success "outlines library patched successfully"
    fi
else
    warning "airports.py not found - may not be needed"
fi

################################################################################
# CHECK IF ALREADY RUNNING
################################################################################

if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p $OLD_PID > /dev/null 2>&1; then
        warning "vLLM server already running (PID: $OLD_PID)"
        echo ""
        echo "To stop: kill $OLD_PID"
        echo "To restart: kill $OLD_PID && $0"
        echo ""
        exit 0
    else
        warning "Stale PID file found, cleaning up"
        rm -f "$PID_FILE"
    fi
fi

# Check if port is in use
if ss -tlnp | grep -q ":$PORT "; then
    error "Port $PORT already in use"
    ss -tlnp | grep ":$PORT"
    exit 1
fi

################################################################################
# START vLLM SERVER
################################################################################

info "Starting vLLM server..."
echo ""
echo "  Model: $MODEL_PATH"
echo "  Port: $PORT"
echo "  Host: $HOST"
echo "  Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "  Log: $LOG_FILE"
echo ""

# Rotate old logs
if [ -f "$LOG_FILE" ]; then
    mv "$LOG_FILE" "${LOG_FILE}.old"
fi

# Start vLLM server in background
nohup python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --port $PORT \
    --host $HOST \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 256 \
    --dtype auto \
    --kv-cache-dtype auto \
    > "$LOG_FILE" 2>&1 &

VLLM_PID=$!
echo $VLLM_PID > "$PID_FILE"

success "vLLM server started (PID: $VLLM_PID)"

################################################################################
# WAIT FOR SERVER TO BE READY
################################################################################

info "Waiting for server to initialize (this takes ~5-7 minutes)..."
echo ""

WAIT_SECONDS=0
MAX_WAIT=600  # 10 minutes max

while [ $WAIT_SECONDS -lt $MAX_WAIT ]; do
    # Check if process is still running
    if ! ps -p $VLLM_PID > /dev/null 2>&1; then
        error "vLLM process died unexpectedly"
        echo ""
        echo "Last 50 lines of log:"
        tail -50 "$LOG_FILE"
        rm -f "$PID_FILE"
        exit 1
    fi
    
    # Check if server is responding
    if curl -s http://localhost:$PORT/v1/models > /dev/null 2>&1; then
        success "Server is ready!"
        break
    fi
    
    # Show progress every 15 seconds
    if [ $((WAIT_SECONDS % 15)) -eq 0 ] && [ $WAIT_SECONDS -gt 0 ]; then
        echo "  Still loading... (${WAIT_SECONDS}s elapsed)"
        
        # Show GPU memory usage
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | \
            awk '{print "    GPU " $1 ": " $2 " used"}'
    fi
    
    sleep 3
    WAIT_SECONDS=$((WAIT_SECONDS + 3))
done

if [ $WAIT_SECONDS -ge $MAX_WAIT ]; then
    error "Server failed to start within $MAX_WAIT seconds"
    echo ""
    echo "Last 50 lines of log:"
    tail -50 "$LOG_FILE"
    kill $VLLM_PID 2>/dev/null || true
    rm -f "$PID_FILE"
    exit 1
fi

################################################################################
# HEALTH CHECK
################################################################################

echo ""
info "Running health checks..."

# Test API endpoint
MODEL_ID=$(curl -s http://localhost:$PORT/v1/models | python3 -c "import sys, json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "ERROR")

if [ "$MODEL_ID" == "ERROR" ]; then
    error "API health check failed"
    exit 1
fi

success "API endpoint responding"
echo "  Model ID: $MODEL_ID"

# Show GPU utilization
echo ""
info "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv | column -t -s ','

################################################################################
# SUCCESS - DISPLAY CONNECTION INFO
################################################################################

echo ""
echo -e "${GREEN}=============================================================================="
echo -e "vLLM Server Successfully Started"
echo -e "=============================================================================="
echo -e "${NC}"
echo "  PID: $VLLM_PID"
echo "  Log: $LOG_FILE"
echo ""
echo "  API Endpoints:"
echo "    http://localhost:$PORT/v1/models"
echo "    http://localhost:$PORT/v1/completions"
echo "    http://localhost:$PORT/v1/chat/completions"
echo ""
echo "  External Access (via SSH tunnel):"
EXTERNAL_IP=$(hostname -I | awk '{print $1}')
echo "    ssh -L $PORT:localhost:$PORT root@<runpod-ip> -p <port>"
echo ""
echo "  To monitor:"
echo "    tail -f $LOG_FILE"
echo "    watch -n 1 nvidia-smi"
echo ""
echo "  To stop:"
echo "    kill $VLLM_PID"
echo "    # or"
echo "    pkill -f 'vllm.entrypoints.openai.api_server'"
echo ""
echo -e "${GREEN}=============================================================================="
echo -e "${NC}"

# Keep monitoring in background (optional)
# tail -f "$LOG_FILE"
