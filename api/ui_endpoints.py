"""
UI API Endpoints for ARC Mission Control
=========================================

Silicon Valley-grade REST API for the ARC UI.
Provides 8 specialized endpoints for real-time data aggregation.

Design Philosophy:
- Fast response times (<100ms for most queries)
- Lightweight JSON responses
- Optimized for UI consumption (no raw dumps)
- Real-time compatible (polling-friendly)

Endpoints:
1. GET /ui/system/health → GPU, CPU, RAM, disk, uptime
2. GET /ui/jobs/queue → Active, queued, completed, failed jobs
3. GET /ui/jobs/<id>/progress → Epoch, loss curve, ETA, status
4. GET /ui/experiments/<id>/metrics → All metrics
5. GET /ui/experiments/<id>/visuals → Paths to CAM, DRI, segmentation
6. GET /ui/experiments/<id>/config → Config summary
7. GET /ui/experiments/timeline → All experiments with timestamps
8. GET /ui/agents/cognition/feed → Recent agent decisions

Author: Dev 2
Date: 2025-11-18
"""

import os
import json
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import ARC components
from config import get_settings
from memory_handler import get_memory_handler
from scheduler.job_scheduler import get_job_scheduler
from agents.historian_agent import HistorianAgent

# Initialize
settings = get_settings()
memory = get_memory_handler(settings)
scheduler = get_job_scheduler(settings, max_concurrent_jobs=2)
historian = HistorianAgent(memory_path=str(settings.memory_dir))

# FastAPI app (can be merged into control_plane.py or run standalone)
app = FastAPI(title='ARC UI API', version='1.0.0')


# ============================================================
# ENDPOINT 1: System Health
# ============================================================

@app.get('/ui/system/health')
async def get_system_health() -> Dict[str, Any]:
    """
    Get real-time system health metrics.

    Returns:
        {
            "cpu": {"usage_percent": 45.2, "cores": 8},
            "ram": {"used_gb": 12.3, "total_gb": 32.0, "percent": 38.4},
            "disk": {"used_gb": 250.5, "total_gb": 500.0, "percent": 50.1},
            "gpu": [
                {"id": 0, "name": "A100", "usage_percent": 78.5, "memory_used_gb": 32.1, "memory_total_gb": 40.0, "temp_celsius": 64},
                {"id": 1, "name": "A100", "usage_percent": 82.3, "memory_used_gb": 35.2, "memory_total_gb": 40.0, "temp_celsius": 66}
            ],
            "uptime_seconds": 86400,
            "timestamp": "2025-11-18T21:05:00Z"
        }
    """
    try:
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()

        # RAM metrics
        ram = psutil.virtual_memory()
        ram_used_gb = ram.used / (1024 ** 3)
        ram_total_gb = ram.total / (1024 ** 3)

        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_used_gb = disk.used / (1024 ** 3)
        disk_total_gb = disk.total / (1024 ** 3)

        # GPU metrics (attempt to read from RunPod GPU monitoring if available)
        gpu_metrics = []
        try:
            # Try to get GPU info from scheduler
            gpu_status = scheduler.get_gpu_status()
            for gpu in gpu_status.get("gpus", []):
                gpu_metrics.append({
                    "id": gpu.get("id", 0),
                    "name": gpu.get("name", "Unknown"),
                    "usage_percent": gpu.get("utilization", 0.0),
                    "memory_used_gb": gpu.get("memory_used_gb", 0.0),
                    "memory_total_gb": gpu.get("memory_total_gb", 0.0),
                    "temp_celsius": gpu.get("temperature", 0)
                })
        except Exception:
            # If GPU info unavailable, return empty list
            pass

        # Uptime
        boot_time = psutil.boot_time()
        uptime_seconds = int(time.time() - boot_time)

        return {
            "cpu": {
                "usage_percent": round(cpu_percent, 1),
                "cores": cpu_count
            },
            "ram": {
                "used_gb": round(ram_used_gb, 1),
                "total_gb": round(ram_total_gb, 1),
                "percent": round(ram.percent, 1)
            },
            "disk": {
                "used_gb": round(disk_used_gb, 1),
                "total_gb": round(disk_total_gb, 1),
                "percent": round(disk.percent, 1)
            },
            "gpu": gpu_metrics,
            "uptime_seconds": uptime_seconds,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system health: {str(e)}")


# ============================================================
# ENDPOINT 2: Job Queue Status
# ============================================================

@app.get('/ui/jobs/queue')
async def get_job_queue() -> Dict[str, Any]:
    """
    Get current job queue status.

    Returns:
        {
            "active": [
                {"job_id": "job_001", "experiment_id": "exp_2025_001", "status": "running", "progress": 0.45, "eta_seconds": 120}
            ],
            "queued": [
                {"job_id": "job_002", "experiment_id": "exp_2025_002", "position": 1, "eta_seconds": 300}
            ],
            "completed_recent": [
                {"job_id": "job_000", "experiment_id": "exp_2025_000", "status": "completed", "duration_seconds": 180}
            ],
            "failed_recent": [
                {"job_id": "job_003", "experiment_id": "exp_2025_003", "status": "failed", "error": "GPU OOM"}
            ],
            "summary": {
                "active_count": 1,
                "queued_count": 1,
                "completed_today": 12,
                "failed_today": 1
            }
        }
    """
    try:
        # Get scheduler queue
        queue_status = scheduler.get_queue_status()

        # Format for UI
        active_jobs = []
        for job in queue_status.get("running", []):
            active_jobs.append({
                "job_id": job.get("job_id"),
                "experiment_id": job.get("experiment_id"),
                "status": "running",
                "progress": job.get("progress", 0.0),
                "eta_seconds": job.get("eta_seconds", 0)
            })

        queued_jobs = []
        for idx, job in enumerate(queue_status.get("pending", [])):
            queued_jobs.append({
                "job_id": job.get("job_id"),
                "experiment_id": job.get("experiment_id"),
                "position": idx + 1,
                "eta_seconds": job.get("eta_seconds", 0)
            })

        completed_jobs = []
        for job in queue_status.get("completed", [])[:10]:  # Last 10
            completed_jobs.append({
                "job_id": job.get("job_id"),
                "experiment_id": job.get("experiment_id"),
                "status": "completed",
                "duration_seconds": job.get("duration_seconds", 0)
            })

        failed_jobs = []
        for job in queue_status.get("failed", [])[:10]:  # Last 10
            failed_jobs.append({
                "job_id": job.get("job_id"),
                "experiment_id": job.get("experiment_id"),
                "status": "failed",
                "error": job.get("error", "Unknown error")
            })

        # Summary counts
        completed_today = len([j for j in queue_status.get("completed", []) if _is_today(j.get("completed_at"))])
        failed_today = len([j for j in queue_status.get("failed", []) if _is_today(j.get("failed_at"))])

        return {
            "active": active_jobs,
            "queued": queued_jobs,
            "completed_recent": completed_jobs,
            "failed_recent": failed_jobs,
            "summary": {
                "active_count": len(active_jobs),
                "queued_count": len(queued_jobs),
                "completed_today": completed_today,
                "failed_today": failed_today
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job queue: {str(e)}")


# ============================================================
# ENDPOINT 3: Job Progress (Live Training View)
# ============================================================

@app.get('/ui/jobs/{job_id}/progress')
async def get_job_progress(job_id: str) -> Dict[str, Any]:
    """
    Get live progress for a specific job.

    Returns:
        {
            "job_id": "job_001",
            "experiment_id": "exp_2025_001",
            "status": "running",
            "current_epoch": 15,
            "total_epochs": 30,
            "progress_percent": 50.0,
            "loss_curve": [0.85, 0.72, 0.68, ..., 0.45],
            "val_auc_curve": [0.75, 0.78, 0.81, ..., 0.87],
            "eta_seconds": 120,
            "gpu_usage": 78.5,
            "timestamp": "2025-11-18T21:05:00Z"
        }
    """
    try:
        # Get job status from scheduler
        job_status = scheduler.get_job_status(job_id)

        if not job_status:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        # Read training logs for loss curve (if available)
        loss_curve = []
        val_auc_curve = []

        experiment_id = job_status.get("experiment_id")
        log_path = Path(settings.logs_dir) / f"{experiment_id}_training.log"

        if log_path.exists():
            # Parse training log for metrics (simplified)
            with open(log_path, 'r') as f:
                for line in f:
                    if "Epoch" in line and "loss:" in line:
                        # Extract loss value (example: "Epoch 5/30 - loss: 0.45")
                        try:
                            loss_val = float(line.split("loss:")[1].split()[0])
                            loss_curve.append(loss_val)
                        except:
                            pass
                    if "val_auc:" in line:
                        try:
                            auc_val = float(line.split("val_auc:")[1].split()[0])
                            val_auc_curve.append(auc_val)
                        except:
                            pass

        return {
            "job_id": job_id,
            "experiment_id": experiment_id,
            "status": job_status.get("status", "unknown"),
            "current_epoch": job_status.get("current_epoch", 0),
            "total_epochs": job_status.get("total_epochs", 0),
            "progress_percent": round(job_status.get("progress", 0.0) * 100, 1),
            "loss_curve": loss_curve,
            "val_auc_curve": val_auc_curve,
            "eta_seconds": job_status.get("eta_seconds", 0),
            "gpu_usage": job_status.get("gpu_usage", 0.0),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job progress: {str(e)}")


# ============================================================
# ENDPOINT 4: Experiment Metrics
# ============================================================

@app.get('/ui/experiments/{experiment_id}/metrics')
async def get_experiment_metrics(experiment_id: str) -> Dict[str, Any]:
    """
    Get all metrics for a completed experiment.

    Returns:
        {
            "experiment_id": "exp_2025_001",
            "status": "completed",
            "metrics": {
                "auc": 0.927,
                "sensitivity": 0.91,
                "specificity": 0.94,
                "accuracy": 0.92,
                "dice": 0.88,
                "loss": 0.23
            },
            "training_time_seconds": 180,
            "best_epoch": 25,
            "timestamp": "2025-11-18T20:45:00Z"
        }
    """
    try:
        # Read from Historian
        history = historian.read_memory("training_history.json") or {"experiments": []}

        # Find experiment
        experiment = None
        for exp in history.get("experiments", []):
            if exp.get("experiment_id") == experiment_id:
                experiment = exp
                break

        if not experiment:
            raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

        return {
            "experiment_id": experiment_id,
            "status": experiment.get("status", "unknown"),
            "metrics": experiment.get("metrics", {}),
            "training_time_seconds": experiment.get("duration_seconds", 0),
            "best_epoch": experiment.get("best_epoch", 0),
            "timestamp": experiment.get("completed_at", datetime.utcnow().isoformat() + "Z")
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get experiment metrics: {str(e)}")


# ============================================================
# ENDPOINT 5: Experiment Visualizations
# ============================================================

@app.get('/ui/experiments/{experiment_id}/visuals')
async def get_experiment_visuals(experiment_id: str) -> Dict[str, Any]:
    """
    Get paths to visualization files for an experiment.

    Returns:
        {
            "experiment_id": "exp_2025_001",
            "visualizations": {
                "gradcam": ["/outputs/exp_2025_001/gradcam_sample_1.png", ...],
                "gradcam_pp": ["/outputs/exp_2025_001/gradcam_pp_sample_1.png", ...],
                "dri": ["/outputs/exp_2025_001/dri_sample_1.png", ...],
                "segmentation": ["/outputs/exp_2025_001/seg_sample_1.png", ...]
            },
            "available": true
        }
    """
    try:
        # Look for visualizations directory
        vis_dir = Path(settings.outputs_dir) / experiment_id / "visualizations"

        if not vis_dir.exists():
            return {
                "experiment_id": experiment_id,
                "visualizations": {},
                "available": False
            }

        # Collect visualization paths
        gradcam_files = sorted([str(f) for f in vis_dir.glob("gradcam_*.png")])
        gradcam_pp_files = sorted([str(f) for f in vis_dir.glob("gradcam_pp_*.png")])
        dri_files = sorted([str(f) for f in vis_dir.glob("dri_*.png")])
        seg_files = sorted([str(f) for f in vis_dir.glob("seg_*.png")])

        return {
            "experiment_id": experiment_id,
            "visualizations": {
                "gradcam": gradcam_files,
                "gradcam_pp": gradcam_pp_files,
                "dri": dri_files,
                "segmentation": seg_files
            },
            "available": True
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get experiment visuals: {str(e)}")


# ============================================================
# ENDPOINT 6: Experiment Config
# ============================================================

@app.get('/ui/experiments/{experiment_id}/config')
async def get_experiment_config(experiment_id: str) -> Dict[str, Any]:
    """
    Get configuration summary for an experiment.

    Returns:
        {
            "experiment_id": "exp_2025_001",
            "config": {
                "model": "efficientnet_b3",
                "optimizer": "adam",
                "learning_rate": 0.0001,
                "batch_size": 16,
                "epochs": 30,
                "loss": "focal",
                "dropout": 0.3,
                "input_size": 512
            },
            "novelty_category": "explore",
            "risk_level": "low"
        }
    """
    try:
        # Read from Historian
        history = historian.read_memory("training_history.json") or {"experiments": []}

        # Find experiment
        experiment = None
        for exp in history.get("experiments", []):
            if exp.get("experiment_id") == experiment_id:
                experiment = exp
                break

        if not experiment:
            raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

        return {
            "experiment_id": experiment_id,
            "config": experiment.get("config", {}),
            "novelty_category": experiment.get("proposal_type", "unknown"),
            "risk_level": experiment.get("risk_level", "unknown")
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get experiment config: {str(e)}")


# ============================================================
# ENDPOINT 7: Experiment Timeline
# ============================================================

@app.get('/ui/experiments/timeline')
async def get_experiment_timeline(limit: int = 50) -> Dict[str, Any]:
    """
    Get timeline of all experiments (most recent first).

    Returns:
        {
            "experiments": [
                {
                    "experiment_id": "exp_2025_001",
                    "cycle_id": 23,
                    "status": "completed",
                    "auc": 0.927,
                    "novelty_category": "explore",
                    "timestamp": "2025-11-18T20:45:00Z"
                },
                ...
            ],
            "total_count": 145
        }
    """
    try:
        # Read from Historian
        history = historian.read_memory("training_history.json") or {"experiments": []}

        all_experiments = history.get("experiments", [])

        # Sort by timestamp (most recent first)
        sorted_experiments = sorted(
            all_experiments,
            key=lambda x: x.get("completed_at", ""),
            reverse=True
        )[:limit]

        # Format for UI
        timeline = []
        for exp in sorted_experiments:
            timeline.append({
                "experiment_id": exp.get("experiment_id"),
                "cycle_id": exp.get("cycle_id", 0),
                "status": exp.get("status", "unknown"),
                "auc": exp.get("metrics", {}).get("auc", 0.0),
                "novelty_category": exp.get("proposal_type", "unknown"),
                "timestamp": exp.get("completed_at", "")
            })

        return {
            "experiments": timeline,
            "total_count": len(all_experiments)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get experiment timeline: {str(e)}")


# ============================================================
# ENDPOINT 8: Agent Cognition Feed
# ============================================================

@app.get('/ui/agents/cognition/feed')
async def get_agent_cognition_feed(limit: int = 50) -> Dict[str, Any]:
    """
    Get recent agent decisions and reasoning (iMessage-style feed).

    Returns:
        {
            "decisions": [
                {
                    "timestamp": "2025-11-18T20:45:30Z",
                    "agent": "Director",
                    "action": "strategy_switch",
                    "message": "Detected stagnation. Switching to EXPLORE mode.",
                    "metadata": {"mode": "explore", "reason": "stagnation"}
                },
                {
                    "timestamp": "2025-11-18T20:45:35Z",
                    "agent": "Supervisor",
                    "action": "veto_proposal",
                    "message": "VETOED exp_2025_042: Learning rate 0.5 exceeds safety limit (0.01)",
                    "metadata": {"experiment_id": "exp_2025_042", "violation": "lr_too_high"}
                },
                ...
            ]
        }
    """
    try:
        # Read agent decision logs
        decisions = []

        # Check if agent logs exist
        agent_log_path = Path(settings.memory_dir) / "agent_decisions.json"

        if agent_log_path.exists():
            with open(agent_log_path, 'r') as f:
                agent_logs = json.load(f)

            # Sort by timestamp (most recent first)
            sorted_logs = sorted(
                agent_logs.get("decisions", []),
                key=lambda x: x.get("timestamp", ""),
                reverse=True
            )[:limit]

            decisions = sorted_logs

        return {
            "decisions": decisions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get agent cognition feed: {str(e)}")


# ============================================================
# Helper Functions
# ============================================================

def _is_today(timestamp_str: Optional[str]) -> bool:
    """Check if timestamp is from today."""
    if not timestamp_str:
        return False
    try:
        ts = datetime.fromisoformat(timestamp_str.replace("Z", ""))
        today = datetime.utcnow().date()
        return ts.date() == today
    except:
        return False


# ============================================================
# Main (for standalone testing)
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
