"""
ArchitectAgent: Experiment design and proposal generation
==========================================================

The Architect generates hypothesis-driven experiment proposals
based on Director's strategy and Historian's insights.
"""

from typing import Dict, Any, List, Optional
from agents.base import BaseAgent, AgentCapability
from llm.router import LLMRouter

# Optional world-model integration
try:
    from agents.world_model import get_world_model
    WORLD_MODEL_AVAILABLE = True
except ImportError:
    WORLD_MODEL_AVAILABLE = False


class ArchitectAgent(BaseAgent):
    """
    Experiment design agent.

    Responsibilities:
    - Generate experiment proposals
    - Design configuration changes
    - Predict metric impacts
    - Assign novelty categories
    """

    def __init__(
        self,
        agent_id: str = "architect_001",
        model: str = "deepseek-r1",
        llm_router: LLMRouter = None,
        voting_weight: float = 1.5,
        memory_path: str = "/workspace/arc/memory",
        use_world_model: bool = True
    ):
        """Initialize Architect agent."""
        super().__init__(
            agent_id=agent_id,
            role="architect",
            model=model,
            capabilities=[AgentCapability.PROPOSAL_GENERATION],
            voting_weight=voting_weight,
            priority="medium",
            offline=False,
            memory_path=memory_path
        )
        self.llm_router = llm_router or LLMRouter(offline_mode=True)

        # Initialize world-model for predictive intelligence
        self.world_model = None
        if use_world_model and WORLD_MODEL_AVAILABLE:
            try:
                self.world_model = get_world_model(
                    memory_path=memory_path,
                    target_metric="auc",
                    auto_train=True
                )
            except Exception as e:
                print(f"Warning: Could not initialize world-model: {e}")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate experiment proposals.

        Args:
            input_data: Contains directive, history, constraints

        Returns:
            List of experiment proposals
        """
        import time
        start_time = time.time()

        try:
            # Read memory
            directive = self.read_memory("directive.json")
            history = self.read_memory("history_summary.json")
            constraints = self.read_memory("constraints.json")

            # Build prompt
            prompt = self._build_proposal_prompt(directive, history, constraints)

            # Get LLM client
            client = self.llm_router.get_client_for_role(self.role)

            # Generate proposals
            response = client.generate_json(prompt, max_tokens=3000, temperature=0.8)

            # Write to memory
            self.write_memory("proposals.json", response)

            # Track success
            duration_ms = (time.time() - start_time) * 1000
            self._track_task("generate_proposals", success=True, duration_ms=duration_ms)

            return response

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._track_task("generate_proposals", success=False, duration_ms=duration_ms)
            raise e

    def vote_on_proposal(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Architects generally approve their own proposals.

        Args:
            proposal: Experiment proposal

        Returns:
            Vote decision
        """
        # Architects approve by default (Critics will scrutinize)
        return {
            "decision": "approve",
            "confidence": 0.85,
            "reasoning": "Proposal designed to meet strategic objectives"
        }

    def _build_proposal_prompt(
        self,
        directive: Dict[str, Any],
        history: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> str:
        """Build prompt for proposal generation."""
        return f"""You are the Architect agent in ARC (Autonomous Research Collective).
Your role is to design novel experiments based on strategic direction.

# Strategic Directive
{directive}

# Research History
{history}

# Safety Constraints
{constraints}

# Your Task
Generate {directive.get('novelty_budget', {}).get('exploit', 2) + directive.get('novelty_budget', {}).get('explore', 1)} experiment proposals.

# Architecture Grammar (Phase E)
You can now propose structured architectures using the following grammar:

**Fusion Types** (how to combine visual features + clinical indicators):
- "late": Concatenate CNN embedding + clinical vector → MLP (baseline, safe)
- "film": Clinical indicators modulate CNN feature maps via learned scale/shift (good for small datasets)
- "cross_attention": Clinical indicators as queries, CNN features as keys/values (best for capturing relationships)
- "gated": Learned soft gates weight CNN vs clinical contribution per-sample (adaptive)

**Backbones** (visual feature extraction):
- "efficientnet_b3": Current baseline, efficient CNN
- "convnext_tiny": Modern ConvNet, good balance
- "convnext_small": Larger ConvNeXt variant
- "deit_small": Vision Transformer (requires attention_config)
- "vit_base": Larger ViT (GPU intensive)

**Example Grammar Proposals**:
```json
// FiLM fusion with ConvNeXt (good for exploring beyond baseline)
{{
  "architecture_grammar": {{
    "fusion_type": "film",
    "backbone": "convnext_tiny",
    "pretrained": "medical",
    "fusion_config": {{"num_film_layers": 3, "dropout": 0.1}}
  }}
}}

// Cross-attention with ViT (high-risk, high-reward)
{{
  "architecture_grammar": {{
    "fusion_type": "cross_attention",
    "backbone": "deit_small",
    "pretrained": "imagenet",
    "attention_config": {{"num_heads": 8, "embed_dim": 384, "depth": 12}},
    "fusion_config": {{"dropout": 0.1}}
  }}
}}
```

**Grammar Rules**:
- ViT backbones (deit_small, vit_base) REQUIRE attention_config
- Cross-attention fusion REQUIRES attention_config
- Specify num_film_layers for FiLM fusion (default: 3, max: 5)
- Specify gating_hidden_dim for gated fusion (default: 128)

Each proposal must include:
1. Unique experiment ID
2. Descriptive name
3. Scientific hypothesis
4. Novelty category (exploit/explore/wildcat)
5. Predicted metrics
6. Configuration changes (can include architecture_grammar)
7. Justification

Return ONLY a valid JSON object:
{{
  "proposals": [
    {{
      "experiment_id": "exp_XXX",
      "name": "descriptive_name",
      "hypothesis": "scientific hypothesis",
      "novelty_category": "exploit" | "explore" | "wildcat",
      "predicted_metrics": {{"auc": 0.XX, "sensitivity": 0.XX, "specificity": 0.XX}},
      "config_changes": {{
        "architecture_grammar": {{...}}  // Optional: Use grammar for architecture search
        "learning_rate": 0.001,           // Standard hyperparameters
        "batch_size": 32
      }},
      "justification": "why this experiment is valuable"
    }}
  ]
}}"""

    def filter_proposals_with_predictions(
        self,
        proposals: List[Dict[str, Any]],
        min_predicted_metric: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Filter proposals using world-model predictions.

        Args:
            proposals: List of proposals to filter
            min_predicted_metric: Minimum predicted metric threshold

        Returns:
            Filtered proposals with prediction info added
        """
        if not self.world_model or not self.world_model.is_trained:
            # No filtering if world-model unavailable
            return proposals

        filtered = []
        for proposal in proposals:
            # Get config changes
            config_changes = proposal.get("config_changes", {})

            # Predict outcome
            prediction = self.world_model.predict(config_changes)

            # Add prediction info
            proposal["world_model_prediction"] = {
                "predicted_auc": prediction.mean,
                "uncertainty": prediction.std,
                "confidence": prediction.confidence
            }

            # Filter based on threshold
            if prediction.mean >= min_predicted_metric:
                filtered.append(proposal)
            else:
                # Log filtered proposal
                print(f"Filtered {proposal.get('experiment_id')}: "
                      f"predicted AUC {prediction.mean:.3f} < {min_predicted_metric:.3f}")

        return filtered

    def rank_proposals_by_acquisition(
        self,
        proposals: List[Dict[str, Any]],
        acquisition: str = "ucb"
    ) -> List[Dict[str, Any]]:
        """
        Rank proposals by acquisition function value.

        Args:
            proposals: List of proposals
            acquisition: Acquisition function (ucb, ei, poi)

        Returns:
            Proposals sorted by acquisition value (best first)
        """
        if not self.world_model or not self.world_model.is_trained:
            return proposals

        # Extract configs
        configs = [p.get("config_changes", {}) for p in proposals]

        # Get acquisition values
        suggestions = self.world_model.suggest_next_experiments(
            candidate_configs=configs,
            n_suggestions=len(proposals),
            acquisition=acquisition
        )

        # Sort proposals by acquisition value
        config_to_value = {
            str(config): value
            for config, value in suggestions
        }

        ranked = sorted(
            proposals,
            key=lambda p: config_to_value.get(str(p.get("config_changes", {})), 0),
            reverse=True
        )

        # Add acquisition scores
        for i, proposal in enumerate(ranked):
            config_str = str(proposal.get("config_changes", {}))
            proposal["acquisition_score"] = config_to_value.get(config_str, 0.0)
            proposal["acquisition_rank"] = i + 1

        return ranked
