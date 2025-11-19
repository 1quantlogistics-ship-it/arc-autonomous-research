"""
CriticAgent: Safety review and constraint enforcement
======================================================

The Critic challenges proposals, detects redundancy, and enforces
safety constraints to prevent risky configurations.
"""

from typing import Dict, Any, List
from agents.base import BaseAgent, AgentCapability
from llm.router import LLMRouter

# Phase E: Architecture grammar validation
try:
    from schemas.architecture_grammar import (
        ArchitectureGrammar, validate_grammar_compatibility
    )
    ARCHITECTURE_GRAMMAR_AVAILABLE = True
except ImportError:
    ARCHITECTURE_GRAMMAR_AVAILABLE = False


class CriticAgent(BaseAgent):
    """
    Safety review agent.

    Responsibilities:
    - Challenge proposals rigorously
    - Detect redundant experiments
    - Enforce safety constraints
    - Flag risky configurations
    """

    def __init__(
        self,
        agent_id: str = "critic_001",
        model: str = "qwen2.5-32b",
        llm_router: LLMRouter = None,
        voting_weight: float = 2.0,
        memory_path: str = "/workspace/arc/memory"
    ):
        """Initialize Critic agent."""
        super().__init__(
            agent_id=agent_id,
            role="critic",
            model=model,
            capabilities=[AgentCapability.SAFETY_REVIEW, AgentCapability.CONSTRAINT_CHECKING],
            voting_weight=voting_weight,
            priority="high",
            offline=False,
            memory_path=memory_path
        )
        self.llm_router = llm_router or LLMRouter(offline_mode=True)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Review experiment proposals for safety.

        Args:
            input_data: Contains proposals to review

        Returns:
            Reviews with approve/reject/revise decisions
        """
        import time
        start_time = time.time()

        try:
            # Read memory
            proposals = self.read_memory("proposals.json")
            history = self.read_memory("history_summary.json")
            constraints = self.read_memory("constraints.json")

            # Build prompt
            prompt = self._build_review_prompt(proposals, history, constraints)

            # Get LLM client
            client = self.llm_router.get_client_for_role(self.role)

            # Generate reviews
            response = client.generate_json(prompt, max_tokens=2500, temperature=0.6)

            # Write to memory
            self.write_memory("reviews.json", response)

            # Track success
            duration_ms = (time.time() - start_time) * 1000
            self._track_task("review_proposals", success=True, duration_ms=duration_ms)

            return response

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._track_task("review_proposals", success=False, duration_ms=duration_ms)
            raise e

    def vote_on_proposal(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Vote on a proposal (safety-focused review).

        Args:
            proposal: Experiment proposal

        Returns:
            Vote decision with safety assessment
        """
        # Read constraints
        constraints = self.read_memory("constraints.json")

        # Get config changes
        config_changes = proposal.get("config_changes", {})

        # Phase E: Validate architecture grammar if present
        if ARCHITECTURE_GRAMMAR_AVAILABLE and "architecture_grammar" in config_changes:
            try:
                # Parse grammar from config
                grammar_dict = config_changes["architecture_grammar"]
                grammar = ArchitectureGrammar(**grammar_dict)

                # Validate compatibility
                is_valid, error_msg = validate_grammar_compatibility(grammar)

                if not is_valid:
                    return {
                        "decision": "reject",
                        "confidence": 0.95,
                        "reasoning": f"Architecture grammar validation failed: {error_msg}",
                        "suggested_changes": "Revise architecture specification to meet constraints"
                    }

            except Exception as e:
                return {
                    "decision": "reject",
                    "confidence": 0.90,
                    "reasoning": f"Invalid architecture grammar: {str(e)}",
                    "suggested_changes": "Fix grammar syntax or validation errors"
                }

        # Check for forbidden parameter ranges
        forbidden_ranges = constraints.get("forbidden_ranges", [])

        for forbidden in forbidden_ranges:
            param = forbidden.get("parameter")
            min_val = forbidden.get("min")
            max_val = forbidden.get("max")

            if param in config_changes:
                value = config_changes[param]
                if min_val <= value <= max_val:
                    return {
                        "decision": "reject",
                        "confidence": 0.95,
                        "reasoning": f"Parameter {param}={value} falls in forbidden range [{min_val}, {max_val}]",
                        "suggested_changes": None
                    }

        # Default: approve with moderate confidence
        return {
            "decision": "approve",
            "confidence": 0.75,
            "reasoning": "No obvious safety concerns detected"
        }

    def _build_review_prompt(
        self,
        proposals: Dict[str, Any],
        history: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> str:
        """Build prompt for safety review."""
        return f"""You are the Critic agent in ARC (Autonomous Research Collective).
Your role is to rigorously review proposals for safety, redundancy, and value.

# Proposals to Review
{proposals}

# Research History (detect redundancy)
{history}

# Safety Constraints (must enforce)
{constraints}

# Your Task
Review each proposal and provide:
1. Decision (approve/reject/revise)
2. Confidence score (0.0-1.0)
3. Detailed reasoning
4. Risk level (low/medium/high/critical)
5. Suggested changes (if revise)

Be CRITICAL. Challenge weak hypotheses, detect redundancy, and flag risks.

Return ONLY a valid JSON object:
{{
  "reviews": [
    {{
      "experiment_id": "exp_XXX",
      "decision": "approve" | "reject" | "revise",
      "confidence": 0.XX,
      "reasoning": "detailed critique",
      "risk_level": "low" | "medium" | "high" | "critical",
      "suggested_changes": {{...}} or null
    }}
  ]
}}"""
