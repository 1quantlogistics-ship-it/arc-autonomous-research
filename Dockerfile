# ARC - Autonomous Research Collective
# Multi-stage Dockerfile for RunPod deployment with GPU support

# ============================================================================
# Stage 1: Base Image with CUDA Support
# ============================================================================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    curl \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# ============================================================================
# Stage 2: Dependencies Installation
# ============================================================================
FROM base AS dependencies

# Set working directory
WORKDIR /workspace/arc

# Copy requirements files
COPY requirements.txt .
COPY requirements-dev.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install development dependencies (optional - comment out for production)
# RUN pip install --no-cache-dir -r requirements-dev.txt

# ============================================================================
# Stage 3: Application
# ============================================================================
FROM dependencies AS application

# Copy application code
COPY . /workspace/arc/

# Create necessary directories
RUN mkdir -p \
    /workspace/arc/memory \
    /workspace/arc/experiments \
    /workspace/arc/logs \
    /workspace/arc/checkpoints \
    /workspace/arc/snapshots \
    /workspace/arc/workspace/datasets

# Set environment variables for ARC
ENV ARC_HOME=/workspace/arc \
    ARC_ENV=production \
    PYTHONPATH=/workspace/arc:$PYTHONPATH

# Expose ports
EXPOSE 8000 8001 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set entrypoint
ENTRYPOINT ["python"]
CMD ["api/control_plane.py"]
