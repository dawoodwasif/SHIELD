"""
Human Feedback Protocol (Section 5.4).

The human operator is modelled as a scripted oracle with access to
ground-truth agent states and attack labels:

  - Latency: 0.5 s  (25 simulator steps at 50 Hz)
  - Interface: binary accept / reject
  - Interventions are event-driven, not continuous.

This module provides both the oracle implementation used during training
and evaluation, and a pattern-based simulator for ablation studies.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Any


# ---------------------------------------------------------------------------
# Oracle constants (Section 5.4)
# ---------------------------------------------------------------------------

ORACLE_LATENCY_S = 0.5      # 0.5 s fixed latency
ORACLE_SIM_STEPS = 25       # 25 steps at 50 Hz = 0.5 s
ORACLE_ERROR_RATE = 0.05    # < 5 % error rate under time pressure (Sec. 6.2)


# ---------------------------------------------------------------------------
# Scripted Oracle  (Section 5.4)
# ---------------------------------------------------------------------------

class HumanOracle:
    """Scripted oracle with ground-truth access for human-in-the-loop evaluation.

    The oracle issues a binary accept/reject response within a fixed 0.5 s
    latency.  It has access to ground-truth attack labels (if provided)
    and otherwise uses the belief vector to make a decision.
    """

    def __init__(
        self,
        latency_s: float = ORACLE_LATENCY_S,
        error_rate: float = ORACLE_ERROR_RATE,
    ):
        self.latency_s = latency_s
        self.error_rate = error_rate

        # Counters for HIC metric (Eq. 25)
        self.total_interventions: int = 0

    def reset(self):
        """Reset counters for a new episode."""
        self.total_interventions = 0

    def respond_to_escalation(
        self,
        node_id: int,
        belief: np.ndarray,
        vacuity: float,
        ground_truth_label: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Process an escalation request and return a binary decision.

        Parameters
        ----------
        node_id : int
            ID of the escalating UAV.
        belief : ndarray (K,)
            Current belief vector of the UAV.
        vacuity : float
            Current vacuity score.
        ground_truth_label : int or None
            If provided (oracle mode), the true semantic class index.

        Returns
        -------
        dict with keys:
            accepted : bool
                True = operator confirms the UAV's current assessment (accept).
                False = operator rejects and provides correction (reject).
            corrected_belief : ndarray or None
                If rejected, the corrected belief vector.  If accepted, None.
        """
        self.total_interventions += 1

        # Determine correctness using ground truth if available
        if ground_truth_label is not None:
            predicted_class = int(np.argmax(belief))
            is_correct = predicted_class == ground_truth_label

            # Oracle may make errors at the specified rate
            if np.random.random() < self.error_rate:
                is_correct = not is_correct

            if is_correct:
                # Accept - current belief is fine
                return {"accepted": True, "corrected_belief": None}
            else:
                # Reject - provide corrected belief
                corrected = self._make_corrected_belief(ground_truth_label, len(belief))
                return {"accepted": False, "corrected_belief": corrected}
        else:
            # No ground truth: use heuristic based on vacuity
            # High vacuity -> likely OOD, accept current best guess with boost
            predicted_class = int(np.argmax(belief))
            if np.random.random() < self.error_rate:
                # Error: provide wrong correction
                wrong_class = np.random.choice(
                    [c for c in range(len(belief)) if c != predicted_class]
                )
                corrected = self._make_corrected_belief(wrong_class, len(belief))
                return {"accepted": False, "corrected_belief": corrected}
            else:
                # Correct: boost confidence in best class
                corrected = self._make_corrected_belief(predicted_class, len(belief))
                return {"accepted": False, "corrected_belief": corrected}

    def respond_to_irs_alert(
        self,
        node_id: int,
        belief: np.ndarray,
        vacuity: float,
        trust_scores: Dict[int, float],
        is_truly_compromised: bool = False,
    ) -> Dict[str, Any]:
        """Binary confirmation for IRS intrusion alert.

        Parameters
        ----------
        is_truly_compromised : bool
            Ground-truth flag for oracle evaluation.

        Returns
        -------
        dict with 'confirmed' : bool and 'corrected_belief' : ndarray or None.
        """
        self.total_interventions += 1

        # Oracle decides based on ground truth
        if np.random.random() < self.error_rate:
            confirmed = not is_truly_compromised  # error
        else:
            confirmed = is_truly_compromised      # correct

        corrected_belief = None
        if not confirmed:
            # False alarm - provide corrected belief (NORMAL state)
            corrected_belief = self._make_corrected_belief(0, len(belief))  # class 0 = NORMAL

        return {"confirmed": confirmed, "corrected_belief": corrected_belief}

    @staticmethod
    def _make_corrected_belief(target_class: int, K: int) -> np.ndarray:
        """Generate a high-confidence corrected belief vector."""
        belief = np.ones(K) * (0.05 / (K - 1))
        belief[target_class] = 0.95
        belief /= belief.sum()
        return belief


# ---------------------------------------------------------------------------
# Human Input Simulator (for ablation with different operator profiles)
# ---------------------------------------------------------------------------

class HumanInputSimulator:
    """Simulates varying human performance for sensitivity studies."""

    PROFILES = {
        "expert":     {"accuracy": 0.95, "latency_s": 0.3},
        "competent":  {"accuracy": 0.90, "latency_s": 0.5},   # default in paper
        "novice":     {"accuracy": 0.70, "latency_s": 1.2},
        "fatigued":   {"accuracy": 0.80, "latency_s": 0.8},
        "distracted": {"accuracy": 0.60, "latency_s": 1.5},
    }

    def __init__(self, profile: str = "competent"):
        if profile not in self.PROFILES:
            raise ValueError(f"Unknown profile '{profile}'. Choose from {list(self.PROFILES)}")
        cfg = self.PROFILES[profile]
        self.oracle = HumanOracle(
            latency_s=cfg["latency_s"],
            error_rate=1.0 - cfg["accuracy"],
        )

    def respond(self, node_id: int, belief: np.ndarray, vacuity: float,
                ground_truth_label: Optional[int] = None) -> Dict[str, Any]:
        return self.oracle.respond_to_escalation(
            node_id, belief, vacuity, ground_truth_label
        )
