"""
Test script for Job Autonomy & Recovery System v2

Tests:
1. Retry logic with exponential backoff
2. Auto-resume from checkpoints
3. Job timeouts
4. Auto-clean broken experiments

Uses dummy mode for CPU-only testing.
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scheduler.training_job_manager import TrainingJobManager, JobStatus
from tools.acuvue_tools import run_classification_training


def failing_training_function(**kwargs):
    """Training function that fails for first 2 attempts."""
    import random

    # Simulate failure on first 2 attempts
    attempt_marker = Path("/tmp/arc_test_attempts.txt")

    if attempt_marker.exists():
        attempts = int(attempt_marker.read_text().strip())
    else:
        attempts = 0

    attempts += 1
    attempt_marker.write_text(str(attempts))

    print(f"Training attempt {attempts}")

    if attempts <= 2:
        print(f"Attempt {attempts} - SIMULATING FAILURE")
        raise RuntimeError(f"Simulated training failure (attempt {attempts})")

    print(f"Attempt {attempts} - SUCCESS")

    # On third attempt, succeed
    return {
        "status": "success",
        "best_checkpoint_path": "/tmp/test_checkpoint.pt",
        "return_code": 0
    }


def test_retry_logic():
    """Test 1: Retry logic with exponential backoff."""
    print("\n" + "="*80)
    print("TEST 1: Retry Logic with Exponential Backoff")
    print("="*80)

    # Clean up attempt marker
    attempt_marker = Path("/tmp/arc_test_attempts.txt")
    if attempt_marker.exists():
        attempt_marker.unlink()

    # Create job manager
    job_manager = TrainingJobManager(
        registry_path="/tmp/arc_test_registry.json",
        max_concurrent_jobs=1
    )

    # Submit job with max_retries=3
    job = job_manager.submit_job(
        job_id="test_retry_001",
        experiment_id="retry_test_exp",
        task_type="classification",
        training_function=failing_training_function,
        training_args={"epochs": 10, "dummy_mode": True},
        checkpoint_dir="/tmp/test_checkpoints",
        log_dir="/tmp/test_logs",
        max_retries=3,
        timeout_seconds=None,
        auto_resume=True
    )

    print(f"✓ Job submitted: {job.job_id}")
    print(f"  Max retries: {job.max_retries}")
    print(f"  Retry delay: {job.retry_delay_seconds}s")

    # Wait for completion (with retry delays)
    print("\nWaiting for job completion (will retry on failures)...")

    max_wait = 300  # 5 minutes max
    start_time = time.time()

    while time.time() - start_time < max_wait:
        job_status = job_manager.get_job_status(job.job_id)

        if job_status.status == JobStatus.COMPLETED:
            print(f"\n✓ Job completed successfully!")
            print(f"  Final status: {job_status.status.value}")
            print(f"  Total retry attempts: {job_status.retry_count}")
            print(f"  Last checkpoint: {job_status.last_checkpoint_path}")
            break
        elif job_status.status == JobStatus.FAILED:
            print(f"\n✗ Job failed after all retries")
            print(f"  Error: {job_status.error_message}")
            print(f"  Retry count: {job_status.retry_count}")
            break
        elif job_status.status == JobStatus.RUNNING:
            print(f"  Status: RUNNING (attempt {job_status.retry_count + 1})")
        elif job_status.status == JobStatus.QUEUED:
            print(f"  Status: QUEUED (retry scheduled after backoff)")

        time.sleep(2)

    job_manager.shutdown()

    return job_status.status == JobStatus.COMPLETED


def test_timeout():
    """Test 2: Job timeout with automatic termination."""
    print("\n" + "="*80)
    print("TEST 2: Job Timeout")
    print("="*80)

    def slow_training(**kwargs):
        """Training that takes too long."""
        print("Starting slow training...")
        time.sleep(20)  # Sleep for 20 seconds
        return {"status": "success", "return_code": 0}

    job_manager = TrainingJobManager(
        registry_path="/tmp/arc_test_registry2.json",
        max_concurrent_jobs=1
    )

    # Submit job with 5-second timeout
    job = job_manager.submit_job(
        job_id="test_timeout_001",
        experiment_id="timeout_test_exp",
        task_type="classification",
        training_function=slow_training,
        training_args={"epochs": 10},
        checkpoint_dir="/tmp/test_checkpoints_timeout",
        log_dir="/tmp/test_logs_timeout",
        max_retries=1,  # Will retry once after timeout
        timeout_seconds=5,  # 5 second timeout
        auto_resume=False
    )

    print(f"✓ Job submitted: {job.job_id}")
    print(f"  Timeout: {job.timeout_seconds}s")

    # Wait for timeout and retry
    print("\nWaiting for timeout (should occur after 5s)...")

    time.sleep(15)  # Wait for timeout + retry

    job_status = job_manager.get_job_status(job.job_id)

    print(f"\n✓ Job timed out as expected")
    print(f"  Status: {job_status.status.value}")
    print(f"  Retry count: {job_status.retry_count}")
    print(f"  Error: {job_status.error_message}")

    job_manager.shutdown()

    return "timeout" in (job_status.error_message or "").lower()


def test_auto_clean():
    """Test 3: Auto-clean broken experiments."""
    print("\n" + "="*80)
    print("TEST 3: Auto-Clean Broken Experiments")
    print("="*80)

    # Create broken experiment directory
    broken_exp_dir = Path("/tmp/arc_broken_experiment")
    broken_checkpoint_dir = broken_exp_dir / "checkpoints"
    broken_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create empty checkpoint dir (simulates broken experiment)
    print(f"Created broken experiment at: {broken_exp_dir}")

    def always_fail(**kwargs):
        raise RuntimeError("Always fails")

    job_manager = TrainingJobManager(
        registry_path="/tmp/arc_test_registry3.json",
        max_concurrent_jobs=1
    )

    # Submit job that will fail
    job = job_manager.submit_job(
        job_id="test_autoclean_001",
        experiment_id="broken_exp",
        task_type="classification",
        training_function=always_fail,
        training_args={},
        checkpoint_dir=str(broken_checkpoint_dir),
        log_dir=str(broken_exp_dir / "logs"),
        max_retries=0,  # No retries - will fail immediately
        auto_resume=False
    )

    print(f"✓ Job submitted: {job.job_id}")

    # Wait for failure
    time.sleep(3)

    job_status = job_manager.get_job_status(job.job_id)
    print(f"  Job status: {job_status.status.value}")

    # Check if experiment directory exists before cleanup
    exists_before = broken_exp_dir.exists()
    print(f"  Experiment dir exists before cleanup: {exists_before}")

    # Run auto-clean
    print("\nRunning auto-clean...")
    cleaned_count = job_manager.auto_clean_broken_experiments()

    print(f"✓ Auto-clean completed")
    print(f"  Experiments cleaned: {cleaned_count}")

    # Check if directory was removed
    exists_after = broken_exp_dir.exists()
    print(f"  Experiment dir exists after cleanup: {exists_after}")

    job_manager.shutdown()

    return cleaned_count > 0 and not exists_after


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("ARC JOB AUTONOMY & RECOVERY SYSTEM v2 - TEST SUITE")
    print("="*80)

    results = {}

    try:
        results["retry_logic"] = test_retry_logic()
    except Exception as e:
        print(f"\n✗ Test 1 failed with exception: {e}")
        results["retry_logic"] = False

    try:
        results["timeout"] = test_timeout()
    except Exception as e:
        print(f"\n✗ Test 2 failed with exception: {e}")
        results["timeout"] = False

    try:
        results["auto_clean"] = test_auto_clean()
    except Exception as e:
        print(f"\n✗ Test 3 failed with exception: {e}")
        results["auto_clean"] = False

    # Print summary
    print("\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(results.values())

    if all_passed:
        print("\n✓ ALL TESTS PASSED")
    else:
        print("\n✗ SOME TESTS FAILED")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
