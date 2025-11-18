#!/bin/bash
# ============================================================================
# ARC RunPod Deployment Script
# ============================================================================
#
# This script prepares and deploys ARC to a RunPod GPU instance.
#
# Usage:
#   ./deploy_runpod.sh [build|run|stop|logs]
#
# Commands:
#   build  - Build Docker image
#   run    - Start ARC services
#   stop   - Stop ARC services
#   logs   - View service logs
#   status - Check service status
# ============================================================================

set -e

# Configuration
IMAGE_NAME="arc-autonomous-research"
IMAGE_TAG="latest"
COMPOSE_FILE="docker-compose.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    log_info "Checking requirements..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker not found. Please install Docker first."
        exit 1
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose not found. Please install Docker Compose first."
        exit 1
    fi

    # Check for NVIDIA GPU
    if ! command -v nvidia-smi &> /dev/null; then
        log_warn "nvidia-smi not found. GPU support may not be available."
    else
        log_info "GPU detected:"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    fi

    log_info "Requirements check passed."
}

build_image() {
    log_info "Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"

    docker build \
        -t ${IMAGE_NAME}:${IMAGE_TAG} \
        -f Dockerfile \
        .

    log_info "Build complete!"
}

run_services() {
    log_info "Starting ARC services..."

    # Create required directories
    mkdir -p memory experiments logs checkpoints snapshots workspace/datasets

    # Copy production env file if not exists
    if [ ! -f .env ]; then
        if [ -f .env.production ]; then
            log_info "Copying .env.production to .env"
            cp .env.production .env
            log_warn "Please edit .env and set your ANTHROPIC_API_KEY"
        else
            log_error ".env.production not found. Cannot create .env"
            exit 1
        fi
    fi

    # Check if API key is set
    if grep -q "your-api-key-here" .env; then
        log_warn "ANTHROPIC_API_KEY not set in .env. Please update before running."
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi

    # Start services
    if command -v docker-compose &> /dev/null; then
        docker-compose -f ${COMPOSE_FILE} up -d
    else
        docker compose -f ${COMPOSE_FILE} up -d
    fi

    log_info "ARC services started!"
    log_info "Control Plane: http://localhost:8000"
    log_info "Dashboard: http://localhost:8501"
    log_info ""
    log_info "View logs with: ./deploy_runpod.sh logs"
}

stop_services() {
    log_info "Stopping ARC services..."

    if command -v docker-compose &> /dev/null; then
        docker-compose -f ${COMPOSE_FILE} down
    else
        docker compose -f ${COMPOSE_FILE} down
    fi

    log_info "Services stopped."
}

view_logs() {
    log_info "Viewing service logs (Ctrl+C to exit)..."

    if command -v docker-compose &> /dev/null; then
        docker-compose -f ${COMPOSE_FILE} logs -f
    else
        docker compose -f ${COMPOSE_FILE} logs -f
    fi
}

check_status() {
    log_info "Checking service status..."

    if command -v docker-compose &> /dev/null; then
        docker-compose -f ${COMPOSE_FILE} ps
    else
        docker compose -f ${COMPOSE_FILE} ps
    fi

    echo ""
    log_info "Health check:"

    # Check Control Plane
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        log_info "✓ Control Plane is healthy"
    else
        log_error "✗ Control Plane is not responding"
    fi
}

# Main script
case "$1" in
    build)
        check_requirements
        build_image
        ;;
    run)
        check_requirements
        run_services
        ;;
    stop)
        stop_services
        ;;
    logs)
        view_logs
        ;;
    status)
        check_status
        ;;
    *)
        echo "Usage: $0 {build|run|stop|logs|status}"
        echo ""
        echo "Commands:"
        echo "  build   - Build Docker image"
        echo "  run     - Start ARC services"
        echo "  stop    - Stop ARC services"
        echo "  logs    - View service logs"
        echo "  status  - Check service status"
        exit 1
        ;;
esac

exit 0
