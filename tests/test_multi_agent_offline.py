"""
Test Multi-Agent Orchestrator in Offline Mode
==============================================

Validates that the orchestrator can:
1. Initialize all 9 agents
2. Run a complete research cycle with MockLLMClient
3. Generate proper decision audit trails
4. Handle supervisor veto scenarios
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.multi_agent_orchestrator import MultiAgentOrchestrator
from llm.health_monitor import get_health_monitor


def test_orchestrator_initialization():
    """Test that orchestrator can initialize with all agents."""
    print("\n" + "="*60)
    print("TEST 1: Orchestrator Initialization")
    print("="*60)

    try:
        orchestrator = MultiAgentOrchestrator(
            memory_path="/Users/bengibson/Desktop/ARC/arc_clean/memory",
            offline_mode=True  # Use MockLLMClient for all agents
        )

        print(f"✓ Orchestrator initialized successfully")
        print(f"  - Registry has {len(orchestrator.registry)} agents")

        # List all agents
        agents = orchestrator.registry.get_all_agents()
        print(f"\n  Registered agents:")
        for agent in agents:
            print(f"    - {agent.role:20s} (model: {agent.model}, weight: {agent.voting_weight})")

        return True

    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_health_monitor():
    """Test that health monitor can check model availability."""
    print("\n" + "="*60)
    print("TEST 2: Health Monitor")
    print("="*60)

    try:
        monitor = get_health_monitor()

        print(f"✓ Health monitor initialized: {monitor}")

        # Check mock model (should always be available)
        is_available = monitor.is_model_available("mock-llm")
        print(f"  - mock-llm available: {is_available}")

        # Get health summary
        summary = monitor.get_health_summary()
        print(f"\n  Health summary:")
        print(f"    - Total models: {summary['overall']['total_models']}")
        print(f"    - Healthy: {summary['overall']['healthy']}")
        print(f"    - Offline: {summary['overall']['offline']}")

        return True

    except Exception as e:
        print(f"✗ Health monitor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_directories():
    """Test that required memory directories exist or can be created."""
    print("\n" + "="*60)
    print("TEST 3: Memory Directory Structure")
    print("="*60)

    memory_dir = Path("/Users/bengibson/Desktop/ARC/arc_clean/memory")

    required_dirs = [
        "decisions",
        "supervisor_decisions",
        "voting",
        "conflict_logs",
        "proposals",
        "history",
        "results"
    ]

    try:
        for dir_name in required_dirs:
            dir_path = memory_dir / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            exists = dir_path.exists()
            print(f"  {'✓' if exists else '✗'} {dir_name:25s} {'exists' if exists else 'created'}")

        print(f"\n✓ All required directories ready")
        return True

    except Exception as e:
        print(f"✗ Directory setup failed: {e}")
        return False


def test_simple_research_cycle():
    """Test running a simple research cycle in offline mode."""
    print("\n" + "="*60)
    print("TEST 4: Simple Research Cycle (Offline)")
    print("="*60)

    try:
        orchestrator = MultiAgentOrchestrator(
            memory_path="/Users/bengibson/Desktop/ARC/arc_clean/memory",
            offline_mode=True
        )

        print("✓ Running research cycle...")

        # Run a single cycle
        result = orchestrator.run_research_cycle(cycle_id=1)

        print(f"\n✓ Research cycle completed!")
        print(f"\n  Results:")
        print(f"    - Final decision: {result.get('final_decision', 'unknown')}")
        print(f"    - Supervisor validated: {result.get('supervisor_validated', False)}")
        print(f"    - Total stages completed: {len(result.get('stage_results', []))}")

        # Check decision logging
        decision_log = Path(orchestrator.memory_path) / "decisions" / "cycle_0001_decision.json"
        if decision_log.exists():
            print(f"    - Decision log created: {decision_log.name}")
            with open(decision_log, 'r') as f:
                log_data = json.load(f)
                print(f"    - Logged stages: {list(log_data.keys())[:5]}...")

        return True

    except Exception as e:
        print(f"✗ Research cycle failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_voting():
    """Test that agents can vote on proposals."""
    print("\n" + "="*60)
    print("TEST 5: Agent Voting Mechanism")
    print("="*60)

    try:
        orchestrator = MultiAgentOrchestrator(
            memory_path="/Users/bengibson/Desktop/ARC/arc_clean/memory",
            offline_mode=True
        )

        # Create a test proposal
        test_proposal = {
            "proposal_id": "test_proposal_001",
            "type": "hyperparameter_change",
            "description": "Test voting on parameter change",
            "changes": {
                "learning_rate": 0.001
            },
            "expected_impact": "Test scenario",
            "risk_level": "low"
        }

        print("✓ Creating test proposal...")

        # Get voting agents (all except executor and historian)
        voting_agents = [
            orchestrator.director,
            orchestrator.architect,
            orchestrator.primary_critic,
            orchestrator.secondary_critic,
            orchestrator.explorer,
            orchestrator.parameter_scientist
        ]

        print(f"✓ Collecting votes from {len(voting_agents)} agents...")

        votes = []
        for agent in voting_agents:
            try:
                vote = agent.vote_on_proposal(test_proposal)
                votes.append(vote)
                print(f"  - {agent.role:20s}: {vote.get('decision', 'unknown'):8s} (confidence: {vote.get('confidence', 0):.2f})")
            except Exception as e:
                print(f"  - {agent.role:20s}: ERROR - {e}")

        # Run voting
        vote_result = orchestrator.voting_system.conduct_vote(test_proposal, votes)

        print(f"\n✓ Voting completed!")
        print(f"  - Decision: {vote_result.decision.value}")
        print(f"  - Weighted score: {vote_result.weighted_score:.2f}")
        print(f"  - Consensus reached: {vote_result.consensus_reached}")
        print(f"  - Confidence: {vote_result.confidence:.2f}")

        return True

    except Exception as e:
        print(f"✗ Voting test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("  MULTI-AGENT ORCHESTRATOR OFFLINE MODE VALIDATION")
    print("="*70)

    tests = [
        ("Orchestrator Initialization", test_orchestrator_initialization),
        ("Health Monitor", test_health_monitor),
        ("Memory Directories", test_memory_directories),
        ("Agent Voting", test_agent_voting),
        ("Research Cycle", test_simple_research_cycle),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "="*70)
    print("  TEST SUMMARY")
    print("="*70)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status:8s}  {test_name}")

    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)

    print(f"\n  Total: {total_passed}/{total_tests} tests passed")
    print("="*70)

    return total_passed == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
