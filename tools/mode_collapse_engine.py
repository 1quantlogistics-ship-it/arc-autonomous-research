"""
Anti-Mode-Collapse Engine for ARC.

Prevents mode collapse (repetitive proposals, lack of exploration diversity)
by enforcing diversity constraints and triggering exploration interventions.

Integrates with:
- Drift detector for diversity monitoring
- Director for exploration mode interventions
- Proposal system for duplicate rejection
- FDA logging for traceability

Key Features:
- Proposal diversity enforcement (reject near-duplicates)
- Exploration budget management
- Automatic Director mode switching
- Similarity-based duplicate detection
- FDA-compliant logging

Author: ARC Team
Created: 2025-11-18
"""

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple

import numpy as np

from config import get_settings
from tools.dev_logger import get_dev_logger
from tools.drift_detector import get_drift_detector

logger = logging.getLogger(__name__)


@dataclass
class CollapseDetectionResult:
    """Result of mode collapse detection."""
    collapse_detected: bool
    collapse_type: str  # "diversity_collapse", "exploration_deficit", "duplicate_proposals"
    severity: str  # "low", "medium", "high", "critical"
    confidence: float
    details: Dict[str, Any]
    recommended_action: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class InterventionAction:
    """Mode collapse intervention action."""
    action_type: str  # "force_exploration", "reject_duplicate", "increase_diversity", "reset_director"
    success: bool
    details: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class ModeCollapseEngine:
    """
    Anti-mode-collapse engine for ARC multi-agent system.

    Monitors proposal diversity, detects mode collapse patterns, and triggers
    interventions to maintain exploration-exploitation balance.
    """

    def __init__(
        self,
        min_diversity_entropy: float = 2.0,  # bits
        max_duplicate_ratio: float = 0.7,  # 30% duplicates triggers intervention
        exploration_budget: int = 5,  # cycles to allocate for exploration
        similarity_threshold: float = 0.95,  # cosine similarity threshold for duplicates
        history_window: int = 20  # cycles to track
    ):
        """
        Initialize anti-mode-collapse engine.

        Args:
            min_diversity_entropy: Minimum Shannon entropy for healthy diversity
            max_duplicate_ratio: Maximum allowed ratio of duplicate proposals
            exploration_budget: Number of forced exploration cycles
            similarity_threshold: Cosine similarity threshold for duplicate detection
            history_window: Number of cycles to track in history
        """
        self.min_diversity_entropy = min_diversity_entropy
        self.max_duplicate_ratio = max_duplicate_ratio
        self.exploration_budget = exploration_budget
        self.similarity_threshold = similarity_threshold
        self.history_window = history_window

        # Proposal tracking
        self.proposal_history: deque = deque(maxlen=history_window)
        self.proposal_signatures: Set[str] = set()  # All-time signatures for duplicate detection

        # Exploration tracking
        self.exploration_cycles_remaining: int = 0
        self.forced_exploration_active: bool = False

        # Integration
        self.drift_detector = get_drift_detector()
        self.dev_logger = get_dev_logger()

        logger.info(f"ModeCollapseEngine initialized with min_entropy={min_diversity_entropy} bits")

    def detect_mode_collapse(
        self,
        proposals: List[Dict[str, Any]],
        cycle_id: int
    ) -> CollapseDetectionResult:
        """
        Detect mode collapse from proposal set.

        Analyzes proposals for:
        - Low diversity (entropy below threshold)
        - High duplicate ratio
        - Lack of exploration

        Args:
            proposals: List of agent proposals
            cycle_id: Current research cycle ID

        Returns:
            CollapseDetectionResult with detection status and recommended action
        """
        # Track proposals
        self.proposal_history.append({
            "cycle_id": cycle_id,
            "proposals": proposals,
            "count": len(proposals)
        })

        # Check 1: Diversity entropy (via drift detector)
        diversity_result = self.drift_detector.detect_diversity_drift(
            proposal_configs=proposals,
            cycle_id=cycle_id
        )

        # Check 2: Duplicate ratio
        num_duplicates, duplicate_indices = self._detect_duplicates(proposals)
        duplicate_ratio = num_duplicates / len(proposals) if len(proposals) > 0 else 0

        # Check 3: Exploration deficit (consecutive low-diversity cycles)
        exploration_deficit = self._check_exploration_deficit()

        # Determine collapse detection
        collapse_detected = False
        collapse_type = "none"
        severity = "low"
        confidence = 0.0
        recommended_action = "Continue normal operation"

        if diversity_result.drift_detected and diversity_result.severity in ["high", "critical"]:
            collapse_detected = True
            collapse_type = "diversity_collapse"
            severity = diversity_result.severity
            confidence = diversity_result.confidence
            recommended_action = f"Force exploration mode for {self.exploration_budget} cycles"

        elif duplicate_ratio > self.max_duplicate_ratio:
            collapse_detected = True
            collapse_type = "duplicate_proposals"
            severity = "high" if duplicate_ratio > 0.5 else "medium"
            confidence = 0.9
            recommended_action = f"Reject {num_duplicates} duplicate proposals and force diversity"

        elif exploration_deficit:
            collapse_detected = True
            collapse_type = "exploration_deficit"
            severity = "medium"
            confidence = 0.7
            recommended_action = f"Increase exploration budget for {self.exploration_budget} cycles"

        details = {
            "cycle_id": cycle_id,
            "num_proposals": len(proposals),
            "diversity_entropy": diversity_result.details.get("entropy", 0.0),
            "entropy_threshold": self.min_diversity_entropy,
            "duplicate_ratio": float(duplicate_ratio),
            "duplicate_threshold": self.max_duplicate_ratio,
            "num_duplicates": num_duplicates,
            "duplicate_indices": duplicate_indices,
            "exploration_deficit": exploration_deficit,
            "diversity_drift_detected": diversity_result.drift_detected
        }

        # Log to FDA
        if collapse_detected:
            self._log_collapse_to_fda(
                collapse_type=collapse_type,
                severity=severity,
                cycle_id=cycle_id,
                details=details
            )

        return CollapseDetectionResult(
            collapse_detected=collapse_detected,
            collapse_type=collapse_type,
            severity=severity,
            confidence=confidence,
            details=details,
            recommended_action=recommended_action
        )

    def enforce_diversity(
        self,
        proposals: List[Dict[str, Any]],
        cycle_id: int
    ) -> Tuple[List[Dict[str, Any]], InterventionAction]:
        """
        Enforce diversity by filtering duplicate proposals.

        Args:
            proposals: List of agent proposals
            cycle_id: Current research cycle ID

        Returns:
            Tuple of (filtered_proposals, intervention_action)
        """
        num_duplicates, duplicate_indices = self._detect_duplicates(proposals)

        if num_duplicates == 0:
            return proposals, InterventionAction(
                action_type="enforce_diversity",
                success=True,
                details={
                    "num_duplicates": 0,
                    "num_filtered": 0,
                    "original_count": len(proposals),
                    "final_count": len(proposals)
                }
            )

        # Filter duplicates
        filtered_proposals = [
            p for i, p in enumerate(proposals)
            if i not in duplicate_indices
        ]

        logger.info(
            f"Diversity enforcement: filtered {num_duplicates} duplicates "
            f"from {len(proposals)} proposals"
        )

        # Log to FDA
        self.dev_logger.log_risk_event(
            event_type="duplicate_proposals_filtered",
            severity="medium",
            description=f"Filtered {num_duplicates} duplicate proposals",
            cycle_id=cycle_id,
            context={
                "num_duplicates": num_duplicates,
                "duplicate_indices": duplicate_indices,
                "original_count": len(proposals),
                "final_count": len(filtered_proposals)
            }
        )

        return filtered_proposals, InterventionAction(
            action_type="reject_duplicate",
            success=True,
            details={
                "num_duplicates": num_duplicates,
                "num_filtered": num_duplicates,
                "original_count": len(proposals),
                "final_count": len(filtered_proposals),
                "duplicate_indices": duplicate_indices
            }
        )

    def trigger_exploration_mode(
        self,
        cycle_id: int,
        reason: str
    ) -> InterventionAction:
        """
        Trigger forced exploration mode.

        Sets exploration budget and activates exploration flag.

        Args:
            cycle_id: Current research cycle ID
            reason: Reason for triggering exploration

        Returns:
            InterventionAction with exploration activation details
        """
        self.exploration_cycles_remaining = self.exploration_budget
        self.forced_exploration_active = True

        logger.warning(
            f"Forced exploration mode activated for {self.exploration_budget} cycles: {reason}"
        )

        # Log to FDA
        self.dev_logger.log_risk_event(
            event_type="forced_exploration_activated",
            severity="high",
            description=f"Forced exploration mode: {reason}",
            cycle_id=cycle_id,
            context={
                "reason": reason,
                "exploration_budget": self.exploration_budget,
                "cycles_remaining": self.exploration_cycles_remaining
            }
        )

        return InterventionAction(
            action_type="force_exploration",
            success=True,
            details={
                "reason": reason,
                "exploration_budget": self.exploration_budget,
                "cycles_remaining": self.exploration_cycles_remaining
            }
        )

    def should_force_exploration(self) -> bool:
        """
        Check if forced exploration mode is active.

        Decrements exploration cycle counter and deactivates when budget exhausted.

        Returns:
            True if forced exploration should be active for this cycle
        """
        if not self.forced_exploration_active:
            return False

        if self.exploration_cycles_remaining > 0:
            self.exploration_cycles_remaining -= 1

            if self.exploration_cycles_remaining == 0:
                self.forced_exploration_active = False
                logger.info("Forced exploration mode deactivated (budget exhausted)")

            return True

        return False

    def _detect_duplicates(
        self,
        proposals: List[Dict[str, Any]]
    ) -> Tuple[int, List[int]]:
        """
        Detect duplicate proposals using signature-based and similarity-based matching.

        Args:
            proposals: List of agent proposals

        Returns:
            Tuple of (num_duplicates, duplicate_indices)
        """
        duplicate_indices = []

        for i, proposal in enumerate(proposals):
            # Create proposal signature (deterministic hash of config)
            signature = self._create_proposal_signature(proposal)

            # Check for exact duplicate
            if signature in self.proposal_signatures:
                duplicate_indices.append(i)
            else:
                # Check for near-duplicates via cosine similarity
                is_near_duplicate = self._is_near_duplicate(proposal, proposals[:i])

                if is_near_duplicate:
                    duplicate_indices.append(i)
                else:
                    # Not a duplicate - add to signatures
                    self.proposal_signatures.add(signature)

        return len(duplicate_indices), duplicate_indices

    def _create_proposal_signature(self, proposal: Dict[str, Any]) -> str:
        """
        Create deterministic signature for proposal.

        Args:
            proposal: Agent proposal

        Returns:
            Signature string (hash of normalized config)
        """
        # Extract key config parameters
        config_changes = proposal.get("config_changes", {})

        # Normalize to sorted JSON for deterministic hashing
        normalized = json.dumps(config_changes, sort_keys=True)

        return normalized

    def _is_near_duplicate(
        self,
        proposal: Dict[str, Any],
        previous_proposals: List[Dict[str, Any]]
    ) -> bool:
        """
        Check if proposal is near-duplicate of any previous proposal.

        Uses cosine similarity of config vectors.

        Args:
            proposal: Current proposal
            previous_proposals: List of previous proposals to compare against

        Returns:
            True if near-duplicate detected
        """
        if not previous_proposals:
            return False

        proposal_vector = self._proposal_to_vector(proposal)

        for prev_proposal in previous_proposals:
            prev_vector = self._proposal_to_vector(prev_proposal)

            # Compute cosine similarity
            similarity = self._cosine_similarity(proposal_vector, prev_vector)

            if similarity >= self.similarity_threshold:
                return True

        return False

    def _proposal_to_vector(self, proposal: Dict[str, Any]) -> np.ndarray:
        """
        Convert proposal config to numerical vector for similarity comparison.

        Uses hash of serialized config for robust comparison across different
        config structures (architecture_grammar, training params, etc.)

        Args:
            proposal: Agent proposal

        Returns:
            NumPy vector representation
        """
        config_changes = proposal.get("config_changes", {})
        
        # Use multiple hash-based features for robust comparison
        features = []
        
        # Hash the entire config for exact match detection
        config_str = json.dumps(config_changes, sort_keys=True)
        full_hash = hash(config_str) % 10000 / 10000.0
        features.append(full_hash)
        
        # Extract architecture-specific features if present
        arch_grammar = config_changes.get("architecture_grammar", {})
        
        # Backbone hash
        backbone = arch_grammar.get("backbone", config_changes.get("architecture", ""))
        backbone_hash = hash(str(backbone)) % 1000 / 1000.0
        features.append(backbone_hash)
        
        # Fusion type hash
        fusion = arch_grammar.get("fusion_type", "")
        fusion_hash = hash(str(fusion)) % 1000 / 1000.0
        features.append(fusion_hash)
        
        # Pretrained source hash
        pretrained = arch_grammar.get("pretrained", "")
        pretrained_hash = hash(str(pretrained)) % 1000 / 1000.0
        features.append(pretrained_hash)
        
        # Training params (with defaults that vary)
        features.append(config_changes.get("learning_rate", arch_grammar.get("learning_rate", 0.0)))
        features.append(config_changes.get("batch_size", 0))
        features.append(config_changes.get("epochs", 0))
        features.append(config_changes.get("dropout", arch_grammar.get("fusion_config", {}).get("dropout", 0.0)))
        
        # Experiment ID hash for uniqueness
        exp_id = proposal.get("experiment_id", "")
        exp_hash = hash(str(exp_id)) % 1000 / 1000.0
        features.append(exp_hash)

        return np.array(features, dtype=float)

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity in [0, 1]
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _check_exploration_deficit(self) -> bool:
        """
        Check if system has exploration deficit (low diversity over multiple cycles).

        Returns:
            True if exploration deficit detected
        """
        if len(self.proposal_history) < 5:
            return False

        # Check if last 5 cycles all had low diversity
        recent_cycles = list(self.proposal_history)[-5:]

        low_diversity_count = 0

        for cycle_data in recent_cycles:
            proposals = cycle_data.get("proposals", [])

            # Skip if no proposals
            if len(proposals) == 0:
                continue

            # Check diversity via drift detector
            cycle_id = cycle_data.get("cycle_id", 0)
            diversity_result = self.drift_detector.detect_diversity_drift(
                proposal_configs=proposals,
                cycle_id=cycle_id
            )

            if diversity_result.drift_detected:
                low_diversity_count += 1

        # Deficit if 4+ of last 5 cycles had low diversity
        return low_diversity_count >= 4

    def _log_collapse_to_fda(
        self,
        collapse_type: str,
        severity: str,
        cycle_id: int,
        details: Dict[str, Any]
    ) -> None:
        """Log mode collapse detection to FDA development logs."""
        self.dev_logger.log_risk_event(
            event_type=f"mode_collapse_{collapse_type}",
            severity=severity,
            description=f"Mode collapse detected: {collapse_type}",
            cycle_id=cycle_id,
            context=details
        )


# Singleton instance
_mode_collapse_engine_instance: Optional[ModeCollapseEngine] = None


def get_mode_collapse_engine(
    min_diversity_entropy: float = 2.0,
    max_duplicate_ratio: float = 0.7,
    exploration_budget: int = 5
) -> ModeCollapseEngine:
    """
    Get singleton mode collapse engine instance.

    Args:
        min_diversity_entropy: Minimum Shannon entropy for healthy diversity
        max_duplicate_ratio: Maximum allowed duplicate ratio
        exploration_budget: Number of forced exploration cycles

    Returns:
        Global ModeCollapseEngine instance
    """
    global _mode_collapse_engine_instance

    if _mode_collapse_engine_instance is None:
        _mode_collapse_engine_instance = ModeCollapseEngine(
            min_diversity_entropy=min_diversity_entropy,
            max_duplicate_ratio=max_duplicate_ratio,
            exploration_budget=exploration_budget
        )

    return _mode_collapse_engine_instance
