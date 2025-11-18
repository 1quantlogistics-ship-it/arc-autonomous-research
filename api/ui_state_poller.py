"""
UI State Poller for ARC Mission Control
========================================

Background service that aggregates data from all UI endpoints
and provides a single cached `/ui/dashboard/state` endpoint.

Design:
- Polls all 8 UI endpoints every 2-5 seconds
- Caches aggregated state in memory
- Provides single fast endpoint for UI
- Handles endpoint failures gracefully
- Reduces backend load by 8x

Architecture:
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
│   UI        │────▶│  State Poller    │────▶│  8 UI        │
│  Dashboard  │     │  (Aggregator)    │     │  Endpoints   │
└─────────────┘     └──────────────────┘     └──────────────┘
     │ polls once        │ caches                │ polls 8x
     │ every 2s          │ state                 │ every 2s
     └───────────────────┘                       │
                                                 ▼
                                        ┌──────────────────┐
                                        │  Scheduler,      │
                                        │  Historian, GPU  │
                                        └──────────────────┘

Author: Dev 2
Date: 2025-11-18
"""

import asyncio
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

# Import UI endpoints
from api.ui_endpoints import (
    get_system_health,
    get_job_queue,
    get_experiment_timeline,
    get_agent_cognition_feed
)

logger = logging.getLogger(__name__)


class UIStatePoller:
    """
    Background poller that aggregates UI state.

    Polls all UI endpoints at regular intervals and caches
    the aggregated state for fast UI access.
    """

    def __init__(self, poll_interval: float = 2.0):
        """
        Initialize UI state poller.

        Args:
            poll_interval: Seconds between polls (default 2.0)
        """
        self.poll_interval = poll_interval
        self.cached_state: Optional[Dict[str, Any]] = None
        self.last_poll_time: Optional[float] = None
        self.poll_count: int = 0
        self.error_count: int = 0
        self._running: bool = False
        self._task: Optional[asyncio.Task] = None

        logger.info(f"UIStatePoller initialized (poll_interval={poll_interval}s)")

    async def start(self):
        """Start the background polling task."""
        if self._running:
            logger.warning("Poller already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info("UIStatePoller started")

    async def stop(self):
        """Stop the background polling task."""
        if not self._running:
            return

        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info("UIStatePoller stopped")

    async def _poll_loop(self):
        """Main polling loop."""
        logger.info("Poll loop started")

        while self._running:
            try:
                # Poll all endpoints
                await self._poll_all_endpoints()

                # Wait for next poll
                await asyncio.sleep(self.poll_interval)

            except asyncio.CancelledError:
                logger.info("Poll loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in poll loop: {e}", exc_info=True)
                self.error_count += 1
                await asyncio.sleep(self.poll_interval)

    async def _poll_all_endpoints(self):
        """Poll all UI endpoints and aggregate state."""
        start_time = time.time()

        try:
            # Poll all endpoints concurrently
            tasks = [
                self._safe_poll(get_system_health()),
                self._safe_poll(get_job_queue()),
                self._safe_poll(get_experiment_timeline(limit=20)),
                self._safe_poll(get_agent_cognition_feed(limit=20))
            ]

            results = await asyncio.gather(*tasks)

            # Unpack results
            system_health, job_queue, timeline, cognition_feed = results

            # Aggregate state
            self.cached_state = {
                "system": system_health or self._get_fallback_system(),
                "jobs": job_queue or self._get_fallback_jobs(),
                "timeline": timeline or self._get_fallback_timeline(),
                "cognition": cognition_feed or self._get_fallback_cognition(),
                "meta": {
                    "poll_count": self.poll_count,
                    "last_poll_time": datetime.utcnow().isoformat() + "Z",
                    "poll_duration_ms": int((time.time() - start_time) * 1000),
                    "error_count": self.error_count
                }
            }

            self.last_poll_time = time.time()
            self.poll_count += 1

            logger.debug(f"Poll #{self.poll_count} complete (took {self.cached_state['meta']['poll_duration_ms']}ms)")

        except Exception as e:
            logger.error(f"Failed to poll endpoints: {e}", exc_info=True)
            self.error_count += 1

    async def _safe_poll(self, coro) -> Optional[Dict[str, Any]]:
        """
        Safely execute a coroutine with error handling.

        Args:
            coro: Coroutine to execute

        Returns:
            Result dict or None if error
        """
        try:
            result = await coro
            return result
        except Exception as e:
            logger.warning(f"Endpoint poll failed: {e}")
            return None

    def get_cached_state(self) -> Dict[str, Any]:
        """
        Get cached aggregated state.

        Returns:
            Cached state dict or fallback if not available
        """
        if self.cached_state is None:
            return self._get_fallback_state()

        return self.cached_state

    def _get_fallback_state(self) -> Dict[str, Any]:
        """Get fallback state when cache is empty."""
        return {
            "system": self._get_fallback_system(),
            "jobs": self._get_fallback_jobs(),
            "timeline": self._get_fallback_timeline(),
            "cognition": self._get_fallback_cognition(),
            "meta": {
                "poll_count": 0,
                "last_poll_time": None,
                "poll_duration_ms": 0,
                "error_count": 0,
                "status": "initializing"
            }
        }

    def _get_fallback_system(self) -> Dict[str, Any]:
        """Fallback system health."""
        return {
            "cpu": {"usage_percent": 0.0, "cores": 0},
            "ram": {"used_gb": 0.0, "total_gb": 0.0, "percent": 0.0},
            "disk": {"used_gb": 0.0, "total_gb": 0.0, "percent": 0.0},
            "gpu": [],
            "uptime_seconds": 0,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    def _get_fallback_jobs(self) -> Dict[str, Any]:
        """Fallback job queue."""
        return {
            "active": [],
            "queued": [],
            "completed_recent": [],
            "failed_recent": [],
            "summary": {
                "active_count": 0,
                "queued_count": 0,
                "completed_today": 0,
                "failed_today": 0
            }
        }

    def _get_fallback_timeline(self) -> Dict[str, Any]:
        """Fallback experiment timeline."""
        return {
            "experiments": [],
            "total_count": 0
        }

    def _get_fallback_cognition(self) -> Dict[str, Any]:
        """Fallback agent cognition feed."""
        return {
            "decisions": []
        }


# ============================================================
# Global Poller Instance
# ============================================================

_global_poller: Optional[UIStatePoller] = None


def get_ui_state_poller(poll_interval: float = 2.0) -> UIStatePoller:
    """
    Get or create the global UI state poller instance.

    Args:
        poll_interval: Polling interval in seconds

    Returns:
        UIStatePoller instance
    """
    global _global_poller

    if _global_poller is None:
        _global_poller = UIStatePoller(poll_interval=poll_interval)
        logger.info("Created global UI state poller")

    return _global_poller


# ============================================================
# FastAPI Integration (optional)
# ============================================================

from fastapi import FastAPI

app = FastAPI(title='UI State Poller', version='1.0.0')

# Create poller instance
poller = get_ui_state_poller(poll_interval=2.0)


@app.on_event("startup")
async def startup_event():
    """Start poller on app startup."""
    await poller.start()
    logger.info("UI State Poller started on app startup")


@app.on_event("shutdown")
async def shutdown_event():
    """Stop poller on app shutdown."""
    await poller.stop()
    logger.info("UI State Poller stopped on app shutdown")


@app.get('/ui/dashboard/state')
async def get_dashboard_state() -> Dict[str, Any]:
    """
    Get aggregated dashboard state (cached).

    This is the **ONLY** endpoint the UI needs to poll.
    All other endpoints are polled by the background poller.

    Returns:
        {
            "system": {...},     # From /ui/system/health
            "jobs": {...},       # From /ui/jobs/queue
            "timeline": {...},   # From /ui/experiments/timeline
            "cognition": {...},  # From /ui/agents/cognition/feed
            "meta": {
                "poll_count": 142,
                "last_poll_time": "2025-11-18T21:10:00Z",
                "poll_duration_ms": 85,
                "error_count": 0
            }
        }
    """
    return poller.get_cached_state()


@app.get('/ui/poller/status')
async def get_poller_status() -> Dict[str, Any]:
    """
    Get poller status for monitoring.

    Returns:
        {
            "running": true,
            "poll_count": 142,
            "error_count": 0,
            "last_poll_time": "2025-11-18T21:10:00Z",
            "poll_interval": 2.0
        }
    """
    return {
        "running": poller._running,
        "poll_count": poller.poll_count,
        "error_count": poller.error_count,
        "last_poll_time": datetime.fromtimestamp(poller.last_poll_time).isoformat() + "Z" if poller.last_poll_time else None,
        "poll_interval": poller.poll_interval
    }


# ============================================================
# Main (for standalone testing)
# ============================================================

if __name__ == "__main__":
    import uvicorn

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run server
    uvicorn.run(app, host="0.0.0.0", port=8004)
