# RunPod Deployment Guide

This guide explains how to deploy ARC (Autonomous Research Collective) to a RunPod GPU instance.

## Prerequisites

- RunPod account with GPU instance
- Docker and Docker Compose installed on RunPod instance
- NVIDIA GPU with CUDA support
- Anthropic API key

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/your-org/arc-autonomous-research.git
cd arc-autonomous-research
```

### 2. Configure Environment

```bash
# Copy production environment template
cp .env.production .env

# Edit .env and set your API key
nano .env
# Set: ANTHROPIC_API_KEY=your-actual-api-key
```

### 3. Deploy

```bash
# Make deployment script executable
chmod +x deploy_runpod.sh

# Build Docker image
./deploy_runpod.sh build

# Start services
./deploy_runpod.sh run
```

### 4. Verify Deployment

```bash
# Check service status
./deploy_runpod.sh status

# View logs
./deploy_runpod.sh logs
```

### 5. Access Services

- **Control Plane API**: `http://<runpod-ip>:8000`
- **Dashboard**: `http://<runpod-ip>:8501`
- **Health Check**: `http://<runpod-ip>:8000/health`

## Architecture

### Services

1. **Control Plane** (Port 8000)
   - REST API for experiment management
   - Multi-agent orchestration
   - Job scheduling and monitoring

2. **Dashboard** (Port 8501)
   - Streamlit web interface
   - Real-time experiment monitoring
   - Agent performance visualization

### Directory Structure

```
/workspace/arc/
├── memory/           # Protocol memory (persistent)
├── experiments/      # Experiment configs and results
├── logs/             # System logs
├── checkpoints/      # Model checkpoints
├── snapshots/        # World-model snapshots
└── workspace/
    └── datasets/     # Training datasets
```

### Environment Variables

Key configuration variables in `.env`:

```bash
# Core
ARC_HOME=/workspace/arc
ARC_ENV=production
ARC_MODE=SEMI              # SEMI, AUTO, or FULL

# LLM
ANTHROPIC_API_KEY=sk-...
ANTHROPIC_MODEL=claude-sonnet-4-5-20250929

# GPU
CUDA_VISIBLE_DEVICES=0

# Control Plane
MAX_CONCURRENT_JOBS=3
JOB_TIMEOUT_SECONDS=3600
```

## Configuration

### Mode Selection

ARC supports three operational modes:

- **SEMI**: Human approval required for high-risk experiments
- **AUTO**: Automatic approval for low/medium risk
- **FULL**: Fully autonomous (use with caution)

Set mode via environment:
```bash
ARC_MODE=SEMI
```

Or via API:
```bash
curl -X POST http://localhost:8000/mode?mode=SEMI
```

### GPU Configuration

The system automatically detects available GPUs. To specify which GPU(s) to use:

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0

# Multiple GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3
```

### Dataset Setup

Mount datasets in `/workspace/arc/workspace/datasets/`:

```bash
# Example: Copy datasets to workspace
cp -r /path/to/rimone /workspace/arc/workspace/datasets/
cp -r /path/to/refuge /workspace/arc/workspace/datasets/
```

Update dataset paths in experiments if needed.

## Operations

### Starting/Stopping Services

```bash
# Start all services
./deploy_runpod.sh run

# Stop all services
./deploy_runpod.sh stop

# Restart services
./deploy_runpod.sh stop && ./deploy_runpod.sh run
```

### Viewing Logs

```bash
# All services
./deploy_runpod.sh logs

# Specific service
docker-compose logs -f control-plane
docker-compose logs -f dashboard
```

### Monitoring

```bash
# Service health
curl http://localhost:8000/health

# System status
curl http://localhost:8000/status

# Active experiments
curl http://localhost:8000/experiments
```

### Executing Research Cycles

Via API:
```bash
curl -X POST http://localhost:8000/cycle \
  -H "Content-Type: application/json" \
  -d '{
    "objective": "Optimize glaucoma detection on REFUGE dataset",
    "max_experiments": 10
  }'
```

Via Dashboard:
1. Open `http://<runpod-ip>:8501`
2. Navigate to "Execute" tab
3. Set cycle parameters
4. Click "Start Cycle"

## Troubleshooting

### GPU Not Detected

Check GPU availability:
```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

Ensure Docker has GPU access:
```bash
# Install NVIDIA Container Toolkit if missing
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Services Not Starting

Check logs:
```bash
./deploy_runpod.sh logs
```

Verify environment:
```bash
cat .env | grep ANTHROPIC_API_KEY
```

Check port conflicts:
```bash
netstat -tlnp | grep -E ':(8000|8001|8002|8501)'
```

### Memory Issues

Monitor memory usage:
```bash
docker stats
```

Increase memory limits in `docker-compose.yml`:
```yaml
deploy:
  resources:
    limits:
      memory: 32G
```

### Disk Space

Check disk usage:
```bash
df -h /workspace/arc
du -sh /workspace/arc/experiments
du -sh /workspace/arc/checkpoints
```

Clean up old experiments:
```bash
# Backup first!
find /workspace/arc/experiments -mtime +30 -type d -exec rm -rf {} +
```

## Performance Tuning

### Concurrent Jobs

Adjust based on GPU memory:
```bash
# .env
MAX_CONCURRENT_JOBS=3  # For 24GB GPU
MAX_CONCURRENT_JOBS=1  # For 12GB GPU
```

### Batch Size

Configure in experiment configs:
```yaml
training:
  batch_size: 8  # Reduce if OOM errors
```

### Mixed Precision

Enable for faster training (automatically enabled):
```yaml
system:
  mixed_precision: true
```

## Security

### API Authentication

Enable in production:
```bash
# .env
ENABLE_AUTH=true
API_KEY=your-secure-api-key
```

Use in requests:
```bash
curl -H "X-API-Key: your-secure-api-key" http://localhost:8000/status
```

### Network Security

Restrict access to trusted IPs using RunPod firewall rules.

### Secrets Management

Never commit `.env` to version control. Use RunPod's environment variable management or secret stores.

## Monitoring & Observability

### Health Checks

Automated health checks run every 30 seconds:
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
```

### Metrics

Access metrics via dashboard:
- Experiment success rates
- Agent performance
- Resource utilization
- Consensus quality

### Logging

Logs are stored in `/workspace/arc/logs/`:
- `control_plane.log` - Main control plane logs
- `exec_cycle_*.log` - Research cycle execution logs
- `training_*.log` - Training job logs

Configure log level:
```bash
# .env
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
```

## Backup & Recovery

### Critical Data

Backup these directories regularly:
- `/workspace/arc/memory/` - Protocol memory
- `/workspace/arc/experiments/` - Experiment history
- `/workspace/arc/checkpoints/` - Model checkpoints

### Backup Script

```bash
#!/bin/bash
tar -czf arc-backup-$(date +%Y%m%d).tar.gz \
  /workspace/arc/memory \
  /workspace/arc/experiments \
  /workspace/arc/checkpoints
```

### Recovery

```bash
tar -xzf arc-backup-YYYYMMDD.tar.gz -C /
./deploy_runpod.sh run
```

## Upgrading

### Pull Latest Changes

```bash
cd /workspace/arc
git pull origin main
./deploy_runpod.sh stop
./deploy_runpod.sh build
./deploy_runpod.sh run
```

### Database Migrations

Check for migration scripts in `migrations/`:
```bash
ls migrations/*.py
python migrations/migrate_v1_to_v2.py
```

## Support

- **Documentation**: [https://docs.arc-research.ai](https://docs.arc-research.ai)
- **Issues**: [GitHub Issues](https://github.com/your-org/arc-autonomous-research/issues)
- **Community**: [Discord](https://discord.gg/arc-research)

## License

MIT License - See LICENSE file for details.
