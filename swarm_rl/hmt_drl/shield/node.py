"""
SHIELD Node: Core per-agent data structures and logic.

Implements:
  - Evidential Perception (Section 4.1, Eqs. 7-9, Table 13)
  - Trust Update (Section 4.2, Eqs. 10-13, Table 15)
  - Trust-Weighted Fusion (Section 4.3, Eq. 14)
  - Safety Override via KL Divergence (Section 4.3, Eqs. 15-17)
  - Quarantine and Reintegration (Section 4.4)

All parameter defaults match Table 15 of the paper.
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants matching the paper
# ---------------------------------------------------------------------------

# K = 10 semantic classes (Table 16, Appendix A.4)
NUM_SEMANTIC_CLASSES = 10
SECURITY_CLASSES = ["NORMAL", "SUSPICIOUS", "ATTACK", "MALWARE", "INTRUSION"]
ENVIRONMENTAL_CLASSES = ["CLEAR_PATH", "OBSTACLE", "CONGESTED", "EMERGENCY", "UNKNOWN"]
ALL_CLASSES = SECURITY_CLASSES + ENVIRONMENTAL_CLASSES

# SHIELD hyperparameters (Table 15)
DEFAULT_TAU_VAC = 0.4       # Vacuity escalation threshold
DEFAULT_KAPPA = 0.1         # Trust smoothing constant
DEFAULT_BETA = 0.05         # Trust decay rate
DEFAULT_ZETA = 0.2          # Quarantine activation threshold
DEFAULT_EPSILON = 0.3       # KL-divergence override threshold
DEFAULT_DELTA = 0.15        # Consensus consistency radius

# Trust / communication parameters
DEFAULT_TRUST_INIT = 0.5    # Initial trust for new or reintegrated agents
DEFAULT_TIMEOUT = 2.0       # Seconds before decay applies (Delta_t_timeout)

# Reintegration parameters (Section 4.4)
DEFAULT_COOLDOWN_STEPS = 50           # T_q cooldown in timesteps
DEFAULT_REINTEGRATION_CONSISTENT = 10 # m consecutive consistent steps

# IRS parameters
IRS_DETECTION_WINDOW = 5.0  # seconds - detection within 5 s of onset (Eq. 22)

# Simulation timestep
DT = 0.02  # 50 Hz control


# ---------------------------------------------------------------------------
# Message dataclass
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """Inter-agent broadcast message: (s_i, b_i, p_i) per Algorithm 1 line 9."""
    sender_id: int
    state: np.ndarray          # s_i  - position / local state estimate
    belief: np.ndarray         # b_i  - K-dim belief vector
    action_dist: np.ndarray    # p_i  - action distribution (softmax logits)
    timestamp: float           # simulation time
    vacuity: float             # u_i  - for informational purposes only


# ---------------------------------------------------------------------------
# Evidential Perception Network  (Section 4.1, Table 13 adapted for flat obs)
# ---------------------------------------------------------------------------

class EvidentialPerceptionNet(nn.Module):
    """Evidential Neural Network producing Dirichlet concentration parameters.

    Architecture follows Table 13 of the paper conceptually, adapted for the
    flat observation vectors provided by QuadSwarm (the simulator does not
    expose raw depth-camera images).  The output layer uses Softplus to
    guarantee strictly positive concentration parameters (alpha > 0).

    Input  : flat observation vector of dimension ``obs_dim``
    Output : alpha in R^K_{>0}  (Dirichlet concentrations, K = 10)
    """

    def __init__(self, obs_dim: int, num_classes: int = NUM_SEMANTIC_CLASSES):
        super().__init__()
        self.num_classes = num_classes

        # MLP backbone (mirrors the representational capacity of the CNN in
        # Table 13 but operates on flat vectors from QuadSwarm)
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # Evidential head: FC -> ReLU -> FC -> Softplus  (Table 13 last rows)
        self.evidence_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            nn.Softplus(),  # ensures alpha_k > 0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Dirichlet concentration parameters alpha in R^K_{>0}.

        We add 1.0 so that alpha_k >= 1, which yields a well-defined Dirichlet
        distribution and keeps vacuity in (0, 1].
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        features = self.backbone(x)
        alpha = self.evidence_head(features) + 1.0  # shift so alpha_k >= 1
        return alpha


def compute_belief_vacuity(alpha: np.ndarray) -> Tuple[np.ndarray, float]:
    """Compute belief and vacuity from Dirichlet concentrations (Eqs. 8-9).

    Parameters
    ----------
    alpha : ndarray of shape (K,), all > 0
        Dirichlet concentration parameters.

    Returns
    -------
    belief : ndarray of shape (K,)
        b_{i,k} = alpha_{i,k} / sum(alpha_{i,l})   (Eq. 8)
    vacuity : float
        u_i = K / sum(alpha_{i,l})                  (Eq. 9)
    """
    K = len(alpha)
    S = float(np.sum(alpha))
    belief = alpha / S
    vacuity = K / S
    return belief, vacuity


# ---------------------------------------------------------------------------
# Robust Aggregate  (Eq. 10)
# ---------------------------------------------------------------------------

def robust_aggregate(states: List[np.ndarray]) -> np.ndarray:
    """Coordinate-wise median of neighbour states (Eq. 10).

    Using the median prevents circular feedback from trust-weighted consensus
    and is robust to up to f < floor(N/3) Byzantine values per coordinate.
    """
    if len(states) == 0:
        raise ValueError("robust_aggregate requires at least one state vector")
    stacked = np.stack(states, axis=0)          # (num_neighbors, state_dim)
    return np.median(stacked, axis=0)           # (state_dim,)


# ---------------------------------------------------------------------------
# KL Divergence  (Eq. 15)
# ---------------------------------------------------------------------------

def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """One-way KL divergence D_KL(p || q) (Eq. 15).

    Both p and q must be valid probability distributions (non-negative, sum~1).
    """
    p_safe = np.clip(p, eps, None)
    q_safe = np.clip(q, eps, None)
    return float(np.sum(p_safe * np.log(p_safe / q_safe)))


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = logits - np.max(logits)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals)


# ---------------------------------------------------------------------------
# ShieldNode  -  per-agent runtime state for Algorithm 1
# ---------------------------------------------------------------------------

class ShieldNode:
    """Runtime state and logic for a single UAV in the SHIELD framework.

    Implements Algorithm 1 (per-timestep update) and Section 4.4 lifecycle.
    """

    def __init__(
        self,
        node_id: int,
        num_actions: int = 7,
        obs_dim: int = 18,
        # Paper parameters (Table 15)
        tau_vac: float = DEFAULT_TAU_VAC,
        kappa: float = DEFAULT_KAPPA,
        beta: float = DEFAULT_BETA,
        zeta: float = DEFAULT_ZETA,
        epsilon: float = DEFAULT_EPSILON,
        delta: float = DEFAULT_DELTA,
        timeout: float = DEFAULT_TIMEOUT,
        trust_init: float = DEFAULT_TRUST_INIT,
        eta: float = 0.5,  # minimum aggregate trust for fusion fallback
        cooldown_steps: int = DEFAULT_COOLDOWN_STEPS,
        reintegration_m: int = DEFAULT_REINTEGRATION_CONSISTENT,
    ):
        self.node_id = node_id
        self.num_actions = num_actions
        self.K = NUM_SEMANTIC_CLASSES

        # --- SHIELD thresholds (Table 15) ---
        self.tau_vac = tau_vac
        self.kappa = kappa
        self.beta = beta
        self.zeta = zeta
        self.epsilon = epsilon
        self.delta = delta
        self.timeout = timeout
        self.trust_init = trust_init
        self.eta = eta
        self.cooldown_steps = cooldown_steps
        self.reintegration_m = reintegration_m

        # --- Per-agent state ---
        self.state: np.ndarray = np.zeros(3)            # position estimate s_i
        self.belief: np.ndarray = np.ones(self.K) / self.K  # b_i
        self.vacuity: float = 1.0                        # u_i
        self.action_dist: np.ndarray = np.ones(num_actions) / num_actions  # p_i

        # --- Trust graph edges ---
        # Neighbour set N_i(t): maps neighbour_id -> edge data
        self.neighbors: Dict[int, _EdgeState] = {}

        # --- Lifecycle ---
        self.is_quarantined: bool = False
        self.quarantine_step: int = 0            # step at which quarantined
        self._reint_consistent_count: int = 0    # consecutive consistent steps

        # --- Escalation ---
        self.escalation_pending: bool = False

        # --- IRS flags ---
        self.irs_alert_active: bool = False

        # --- ENN (created externally and shared via parameter sharing) ---
        self.enn: Optional[EvidentialPerceptionNet] = None

        # --- Current simulation time (set externally each step) ---
        self.sim_time: float = 0.0
        self.sim_step: int = 0

    # ------------------------------------------------------------------
    # Algorithm 1, Step 1: Evidential Perception (lines 2-7)
    # ------------------------------------------------------------------

    def evidential_perception(self, obs: np.ndarray) -> Tuple[np.ndarray, float]:
        """Run ENN and compute belief / vacuity (Eqs. 7-9).

        If the ENN is not available (e.g. during pure RL training without
        the full SHIELD stack), we fall back to a uniform belief with moderate
        vacuity so that the rest of the pipeline still functions.
        """
        if self.enn is not None:
            with torch.no_grad():
                obs_t = torch.from_numpy(obs.astype(np.float32))
                self.enn.eval()
                alpha = self.enn(obs_t).squeeze(0).numpy()  # (K,)
        else:
            # Fallback: Dirichlet with weak evidence (high vacuity)
            alpha = np.ones(self.K) * 1.5
            # Slightly favour NORMAL + CLEAR_PATH when no ENN is present
            alpha[0] += 0.5  # NORMAL
            alpha[5] += 0.5  # CLEAR_PATH

        self.belief, self.vacuity = compute_belief_vacuity(alpha)
        return self.belief, self.vacuity

    def needs_escalation(self) -> bool:
        """Check vacuity trigger (Algorithm 1 line 4): u_i > tau_vac."""
        return self.vacuity > self.tau_vac

    def apply_human_correction(self, corrected_belief: np.ndarray):
        """Receive corrected belief from operator (Algorithm 1 lines 5-6)."""
        self.belief = corrected_belief.copy()
        self.vacuity = 0.0
        self.escalation_pending = False

    # ------------------------------------------------------------------
    # Algorithm 1, Step 3: Communication (line 9)
    # ------------------------------------------------------------------

    def create_broadcast(self) -> Message:
        """Create broadcast message (s_i, b_i, p_i)."""
        return Message(
            sender_id=self.node_id,
            state=self.state.copy(),
            belief=self.belief.copy(),
            action_dist=self.action_dist.copy(),
            timestamp=self.sim_time,
            vacuity=self.vacuity,
        )

    def receive_messages(self, messages: List[Message]):
        """Ingest messages from neighbours and update last-heard times."""
        for msg in messages:
            nid = msg.sender_id
            if nid == self.node_id:
                continue
            if nid not in self.neighbors:
                # New neighbour - initialise edge
                self.neighbors[nid] = _EdgeState(
                    kappa=self.kappa, trust_init=self.trust_init
                )
            edge = self.neighbors[nid]
            edge.last_msg = msg
            edge.last_heard = self.sim_time

    # ------------------------------------------------------------------
    # Algorithm 1, Step 4: Trust Update (lines 10-22)
    # ------------------------------------------------------------------

    def update_trust(self) -> List[int]:
        """Update trust for every neighbour (Eqs. 10-13, Algorithm 1 lines 10-22).

        Returns list of neighbour IDs that were quarantined.
        """
        if len(self.neighbors) == 0:
            return []

        # Robust reference state (Eq. 10)
        neighbor_states = [
            e.last_msg.state for e in self.neighbors.values()
            if e.last_msg is not None
        ]
        if len(neighbor_states) == 0:
            return []
        s_tilde = robust_aggregate(neighbor_states)

        quarantine_list: List[int] = []

        for nid, edge in list(self.neighbors.items()):
            if edge.last_msg is None:
                continue

            sj = edge.last_msg.state

            # Consistency check (Eq. 11): ||s_j - s_tilde_i|| < delta
            consistent = float(np.linalg.norm(sj - s_tilde)) < self.delta

            # Update counters (Eqs. 11-12)
            edge.C += int(consistent)
            edge.T += 1

            # Trust ratio (Eq. 13 first factor)
            trust_ratio = (edge.C + self.kappa) / (edge.T + self.kappa)

            # Exponential decay - only if timeout exceeded (Section 4.2)
            if (self.sim_time - edge.last_heard) > self.timeout:
                decay = np.exp(-self.beta * (self.sim_time - edge.last_heard))
            else:
                decay = 1.0

            edge.trust = trust_ratio * decay

            # Quarantine check (Algorithm 1 lines 18-21)
            if edge.trust < self.zeta:
                quarantine_list.append(nid)

        # Execute quarantines
        for nid in quarantine_list:
            del self.neighbors[nid]

        return quarantine_list

    # ------------------------------------------------------------------
    # Algorithm 1, Step 5: Trust-Weighted Fusion (lines 23-27)
    # ------------------------------------------------------------------

    def trust_weighted_fusion(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Trust-weighted averaging of neighbour info (Eq. 14).

        Returns fused (s_bar_i, b_bar_i, p_bar_i).
        Falls back to local estimates if sum(tau) < eta  (Algorithm 1 line 26).
        """
        total_trust = 0.0
        weighted_s = np.zeros_like(self.state)
        weighted_b = np.zeros_like(self.belief)
        weighted_p = np.zeros(self.num_actions)

        for edge in self.neighbors.values():
            if edge.last_msg is None:
                continue
            t = edge.trust
            total_trust += t
            weighted_s += t * edge.last_msg.state
            weighted_b += t * edge.last_msg.belief
            # Ensure action_dist has correct size
            ad = edge.last_msg.action_dist
            if len(ad) == self.num_actions:
                weighted_p += t * ad

        if total_trust < self.eta:
            # Fallback to local (Algorithm 1 line 26)
            return self.state.copy(), self.belief.copy(), self.action_dist.copy()

        fused_s = weighted_s / total_trust
        fused_b = weighted_b / total_trust
        fused_p = weighted_p / total_trust
        # Normalise fused action distribution
        fused_p = np.clip(fused_p, 1e-10, None)
        fused_p /= fused_p.sum()
        return fused_s, fused_b, fused_p

    # ------------------------------------------------------------------
    # Algorithm 1, Step 6: Safety Arbitration (lines 28-32)
    # ------------------------------------------------------------------

    def safety_arbitration(
        self, pi_local: np.ndarray, p_bar: np.ndarray
    ) -> Tuple[int, bool]:
        """KL-divergence safety override (Eqs. 15-17, Algorithm 1 lines 28-32).

        Parameters
        ----------
        pi_local : action probability distribution from local policy
        p_bar    : fused consensus distribution

        Returns
        -------
        action : int  - chosen discrete action index
        overridden : bool - whether safety override was triggered
        """
        pi_local_safe = softmax(pi_local) if np.any(pi_local < 0) else pi_local.copy()
        p_bar_safe = softmax(p_bar) if np.any(p_bar < 0) else p_bar.copy()

        # Normalise
        pi_local_safe = np.clip(pi_local_safe, 1e-10, None)
        pi_local_safe /= pi_local_safe.sum()
        p_bar_safe = np.clip(p_bar_safe, 1e-10, None)
        p_bar_safe /= p_bar_safe.sum()

        dkl = kl_divergence(pi_local_safe, p_bar_safe)

        # Override condition (Eq. 16)
        if dkl > self.epsilon or self.vacuity > self.tau_vac:
            # Use consensus argmax (Eq. 17)
            action = int(np.argmax(p_bar_safe))
            return action, True
        else:
            # Sample from local policy
            action = int(np.random.choice(len(pi_local_safe), p=pi_local_safe))
            return action, False

    # ------------------------------------------------------------------
    # Reintegration (Section 4.4)
    # ------------------------------------------------------------------

    def check_reintegration(self, current_step: int) -> bool:
        """Check if a quarantined node can rejoin.

        Requires: cooldown elapsed, low vacuity, m consecutive consistent steps.
        """
        if not self.is_quarantined:
            return False
        if (current_step - self.quarantine_step) < self.cooldown_steps:
            return False
        if self.vacuity > self.tau_vac:
            self._reint_consistent_count = 0
            return False
        return self._reint_consistent_count >= self.reintegration_m

    def record_reintegration_consistency(self, is_consistent: bool):
        """Record one step of consistency for reintegration."""
        if is_consistent:
            self._reint_consistent_count += 1
        else:
            self._reint_consistent_count = 0

    def reintegrate(self):
        """Restore node to active status with initial trust."""
        self.is_quarantined = False
        self._reint_consistent_count = 0


# ---------------------------------------------------------------------------
# Per-edge state
# ---------------------------------------------------------------------------

class _EdgeState:
    """Per-edge (i, j) trust tracking state."""

    __slots__ = ("C", "T", "trust", "last_heard", "last_msg", "kappa")

    def __init__(self, kappa: float = DEFAULT_KAPPA, trust_init: float = DEFAULT_TRUST_INIT):
        self.C: int = 0                    # consistency counter C_ij
        self.T: int = 0                    # total interaction counter T_ij
        self.trust: float = trust_init     # tau_ij
        self.last_heard: float = 0.0       # t^last_ij
        self.last_msg: Optional[Message] = None
        self.kappa: float = kappa
