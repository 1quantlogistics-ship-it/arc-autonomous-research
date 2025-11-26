#!/usr/bin/env python3
"""
ARC Training Cycle Orchestrator - Full cycle with real training execution

Bulletproof Execution Integration (Dev 2):
- ExperimentRegistry for experiment state tracking
- MetricsStreamer for live training metrics
- GPU pre-flight checks (when Dev 1's GPUManager is available)
"""

import os
import json
import requests
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Bulletproof Execution imports (Dev 2)
try:
    from execution.experiment_lifecycle import (
        ExperimentState,
        ExperimentRecord,
        ExperimentRegistry,
        get_experiment_registry,
    )
    from execution.metrics_streamer import (
        MetricsStreamer,
        get_metrics_streamer,
    )
    BULLETPROOF_AVAILABLE = True
except ImportError:
    BULLETPROOF_AVAILABLE = False

# GPU Manager import (Dev 1 - may not exist yet)
try:
    from execution.gpu_manager import GPUManager
    GPU_MANAGER_AVAILABLE = True
except ImportError:
    GPU_MANAGER_AVAILABLE = False

logger = logging.getLogger(__name__)

# Only configure logging if not already configured
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

MEMORY_DIR = '/workspace/arc/memory'
EXPERIMENTS_DIR = '/workspace/arc/experiments'
LLM_ENDPOINT = 'http://localhost:8000/generate'

class TrainingCycleOrchestrator:
    def __init__(self):
        self.memory_dir = MEMORY_DIR
        self.experiments_dir = EXPERIMENTS_DIR
        self.llm_endpoint = LLM_ENDPOINT

        # Bulletproof Execution components (Dev 2)
        self._registry: Optional[ExperimentRegistry] = None
        self._streamer: Optional[MetricsStreamer] = None

        if BULLETPROOF_AVAILABLE:
            try:
                self._registry = get_experiment_registry(
                    storage_path=os.path.join(self.experiments_dir, "registry")
                )
                self._streamer = get_metrics_streamer(poll_interval=1.0)

                # Add callback to update registry when metrics arrive
                self._streamer.add_callback(self._on_metrics_update)

                logger.info("Bulletproof Execution: ExperimentRegistry and MetricsStreamer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Bulletproof Execution components: {e}")

    def _on_metrics_update(
        self,
        experiment_id: str,
        metrics: Dict[str, float],
        step: int,
        timestamp: str,
    ) -> None:
        """Callback when live metrics are received from training."""
        if self._registry:
            self._registry.update_metrics(experiment_id, metrics)
            logger.debug(f"Updated metrics for {experiment_id} at step {step}")
        
    def load_memory(self, filename: str) -> Dict[str, Any]:
        path = os.path.join(self.memory_dir, filename)
        with open(path, 'r') as f:
            return json.load(f)
    
    def save_memory(self, filename: str, data: Dict[str, Any]):
        path = os.path.join(self.memory_dir, filename)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def call_llm(self, prompt: str, max_tokens: int = 2000) -> str:
        try:
            response = requests.post(
                self.llm_endpoint,
                json={'prompt': prompt, 'max_tokens': max_tokens, 'temperature': 0.7},
                timeout=120
            )
            if response.status_code == 200:
                result = response.json()
                return result.get('text', [''])[0]
            return ''
        except Exception as e:
            logger.error(f'LLM call error: {e}')
            return ''
    
    def extract_json(self, response: str) -> Dict[str, Any]:
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
            return {}
        except Exception as e:
            logger.error(f'JSON extraction error: {e}')
            return {}
    
    def historian_update(self, cycle_id: int, new_results: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        logger.info(f'[HISTORIAN] Updating history for cycle {cycle_id}')
        history = self.load_memory('history_summary.json')
        
        if new_results:
            # Update with real results from training
            history['total_experiments'] = history.get('total_experiments', 0) + len(new_results)
            
            # Update best metrics
            for result in new_results:
                auc = result.get('auc')
                sensitivity = result.get('sensitivity')
                specificity = result.get('specificity')
                
                if auc and (history['best_metrics']['auc'] is None or auc > history['best_metrics']['auc']):
                    history['best_metrics']['auc'] = auc
                if sensitivity and (history['best_metrics']['sensitivity'] is None or sensitivity > history['best_metrics']['sensitivity']):
                    history['best_metrics']['sensitivity'] = sensitivity
                if specificity and (history['best_metrics']['specificity'] is None or specificity > history['best_metrics']['specificity']):
                    history['best_metrics']['specificity'] = specificity
            
            # Add to recent experiments
            history['recent_experiments'] = history.get('recent_experiments', [])
            for result in new_results:
                history['recent_experiments'].append({
                    'experiment_id': result.get('experiment_id'),
                    'auc': result.get('auc'),
                    'training_time': result.get('training_time'),
                    'timestamp': result.get('timestamp')
                })
            
            # Keep only last 10
            history['recent_experiments'] = history['recent_experiments'][-10:]
        
        history['total_cycles'] = cycle_id + 1
        
        self.save_memory('history_summary.json', history)
        logger.info(f'[HISTORIAN] Updated with {len(new_results) if new_results else 0} new results')
        return history
    
    def executor_train(self, cycle_id: int, approved: List[str], proposals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info(f'[EXECUTOR] Training {len(approved)} approved experiments')

        results = []
        for exp_id in approved:
            proposal = next((p for p in proposals if p['experiment_id'] == exp_id), None)
            if not proposal:
                continue

            exp_dir = os.path.join(self.experiments_dir, exp_id)
            os.makedirs(exp_dir, exist_ok=True)

            # Register experiment in registry (Bulletproof Execution)
            if self._registry:
                try:
                    record = ExperimentRecord(
                        experiment_id=exp_id,
                        cycle_id=cycle_id,
                        proposal_id=proposal.get('proposal_id', exp_id),
                        config=proposal.get('config_changes', {}),
                        timeout_seconds=300,
                    )
                    self._registry.register(record)
                    self._registry.update_state(exp_id, ExperimentState.QUEUED)
                except ValueError:
                    # Already registered, just update state
                    self._registry.update_state(exp_id, ExperimentState.QUEUED)

            # Setup metrics streaming
            metrics_file = os.path.join(exp_dir, 'metrics.jsonl')
            if self._streamer:
                self._streamer.register_experiment(exp_id, metrics_file)

            # Execute training
            logger.info(f'[EXECUTOR] Launching training for {exp_id}')

            # Update state to RUNNING
            if self._registry:
                self._registry.update_state(exp_id, ExperimentState.RUNNING, pid=os.getpid())

            try:
                cmd = [
                    'python', '/workspace/arc/api/training_stub.py', exp_id
                ]
                result = subprocess.run(
                    cmd,
                    cwd='/workspace/arc/api',
                    capture_output=True,
                    text=True,
                    timeout=300,
                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0'}
                )

                if result.returncode == 0:
                    # Load results
                    results_path = os.path.join(exp_dir, 'results.json')
                    if os.path.exists(results_path):
                        with open(results_path, 'r') as f:
                            metrics = json.load(f)
                        results.append(metrics)
                        logger.info(f'[EXECUTOR] Training complete for {exp_id}: AUC={metrics.get("auc", 0):.4f}')

                        # Update registry with final results
                        if self._registry:
                            self._registry.update_state(
                                exp_id,
                                ExperimentState.COMPLETED,
                                metrics=metrics,
                                exit_code=0,
                            )
                else:
                    logger.error(f'[EXECUTOR] Training failed for {exp_id}: {result.stderr}')
                    if self._registry:
                        self._registry.update_state(
                            exp_id,
                            ExperimentState.FAILED,
                            error_message=result.stderr[:500],
                            exit_code=result.returncode,
                        )

            except subprocess.TimeoutExpired:
                logger.error(f'[EXECUTOR] Training timeout for {exp_id}')
                if self._registry:
                    self._registry.update_state(
                        exp_id,
                        ExperimentState.TIMEOUT,
                        error_message="Training exceeded 300s timeout",
                    )

            except Exception as e:
                logger.error(f'[EXECUTOR] Training error for {exp_id}: {e}')
                if self._registry:
                    self._registry.update_state(
                        exp_id,
                        ExperimentState.CRASHED,
                        error_message=str(e)[:500],
                    )

            finally:
                # Stop streaming for this experiment
                if self._streamer:
                    self._streamer.unregister_experiment(exp_id)

        return results
    
    def run_training_cycle(self, cycle_id: int = 0):
        logger.info(f'===== TRAINING CYCLE {cycle_id} START =====')
        
        # Load proposals and reviews from previous cycle
        proposals_data = self.load_memory('proposals.json')
        reviews_data = self.load_memory('reviews.json')
        
        proposals = proposals_data.get('proposals', [])
        approved = reviews_data.get('approved', [])
        
        if not approved:
            logger.warning('No approved experiments to train')
            return {
                'cycle_id': cycle_id,
                'experiments_run': 0,
                'results': [],
                'message': 'No approved experiments'
            }
        
        # Execute training
        training_results = self.executor_train(cycle_id, approved, proposals)
        
        # Update Historian with real results
        history = self.historian_update(cycle_id, training_results)
        
        # Update system state
        system_state = self.load_memory('system_state.json')
        system_state['last_cycle_timestamp'] = datetime.now().isoformat()
        system_state['status'] = 'training_complete'
        self.save_memory('system_state.json', system_state)
        
        logger.info(f'===== TRAINING CYCLE {cycle_id} COMPLETE =====')
        
        return {
            'cycle_id': cycle_id,
            'experiments_run': len(training_results),
            'results': training_results,
            'best_auc': history['best_metrics']['auc'],
            'timestamp': datetime.now().isoformat()
        }

if __name__ == '__main__':
    import sys
    orchestrator = TrainingCycleOrchestrator()
    cycle_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    result = orchestrator.run_training_cycle(cycle_id)
    print(json.dumps(result, indent=2))
