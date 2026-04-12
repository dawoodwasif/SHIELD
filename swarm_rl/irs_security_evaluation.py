"""
Attack Injector and IRS Security Evaluation (Sections 3.3, 5.2, 5.3).

Implements all six attack classes (A1-A6) with parameterisation from Table 2:

    A1: GPS Spoofing     - delta_max=3.0 m, 5 s on / 20 s off
    A2: Comm. Jamming    - drop prob p=0.5, 3-7 s window
    A3: Byzantine Faults - inject prob p=0.10, continuous
    A4: Replay Attack    - lag Delta_tr=10 s, 5 s on / 25 s off
    A5: Malware Inject.  - policy drift ||Delta_pi|| <= 0.15, continuous
    A6: Network Intrusion- eavesdrop + selective inject, continuous

Evaluation metrics:
    DR  (Eq. 22) - Detection Rate within 5 s of onset
    FPR (Eq. 23) - False Positive Rate
    RT  (Eq. 24) - Recovery Time
    HIC (Eq. 25) - Human Intervention Count
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from collections import deque


# ---------------------------------------------------------------------------
# Simulation constants
# ---------------------------------------------------------------------------

SIM_HZ = 50              # control frequency
DT = 1.0 / SIM_HZ        # 0.02 s per step
GPS_NOISE_STD = 0.3       # sigma_GPS = 0.3 m  (Section 5.1)
IMU_NOISE_STD = 0.01      # sigma_IMU = 0.01 rad/s


# ---------------------------------------------------------------------------
# Attack configuration  (Table 2)
# ---------------------------------------------------------------------------

@dataclass
class AttackConfig:
    # A1: GPS Spoofing
    a1_delta_max: float = 3.0       # metres, ~10 sigma of GPS noise
    a1_on_duration: float = 5.0     # seconds
    a1_off_duration: float = 20.0   # seconds (period ~ 25 s)
    a1_target_count: int = 1        # 1 random UAV

    # A2: Communication Jamming
    a2_drop_prob: float = 0.5
    a2_window_min: float = 3.0      # seconds
    a2_window_max: float = 7.0      # seconds
    a2_target_links: int = 2        # 2 random links

    # A3: Byzantine Faults
    a3_inject_prob: float = 0.10    # probability per message
    a3_target_count: int = 1        # 1 compromised UAV

    # A4: Replay Attack
    a4_lag: float = 10.0            # Delta_tr = 10 s  (500 steps at 50 Hz)
    a4_on_duration: float = 5.0
    a4_off_duration: float = 25.0
    a4_target_count: int = 1

    # A5: Malware Injection
    a5_drift_bound: float = 0.15    # ||Delta_pi|| <= 0.15
    a5_target_count: int = 1

    # A6: Network Intrusion
    a6_target_segments: int = 1     # 1 network segment

    # Constraint from Section 3.2: f < floor(N/3)
    max_compromised: int = 2        # at most f=2 of N=8


# ---------------------------------------------------------------------------
# Per-attack-class state machines
# ---------------------------------------------------------------------------

class _CyclicAttack:
    """On/off cycling for A1, A4."""

    def __init__(self, on_dur: float, off_dur: float):
        self.on_dur = on_dur
        self.off_dur = off_dur
        self.period = on_dur + off_dur
        self._phase_offset = np.random.uniform(0, self.period)

    def is_active(self, sim_time: float) -> bool:
        t = (sim_time + self._phase_offset) % self.period
        return t < self.on_dur


class _WindowAttack:
    """Random-window cycling for A2."""

    def __init__(self, win_min: float, win_max: float):
        self.win_min = win_min
        self.win_max = win_max
        self._next_start: float = np.random.uniform(5.0, 15.0)
        self._window_end: float = 0.0
        self._active = False

    def is_active(self, sim_time: float) -> bool:
        if self._active:
            if sim_time >= self._window_end:
                self._active = False
                self._next_start = sim_time + np.random.uniform(10.0, 20.0)
            return True
        else:
            if sim_time >= self._next_start:
                dur = np.random.uniform(self.win_min, self.win_max)
                self._window_end = sim_time + dur
                self._active = True
                return True
            return False


# ---------------------------------------------------------------------------
# Attack Injector
# ---------------------------------------------------------------------------

class AttackInjector:
    """Injects attacks per Table 2 parameterisation and tracks ground truth.

    Call ``step()`` each timestep to get the current attack state. The
    injector modifies observations, actions, and the message router in place.
    """

    def __init__(self, num_agents: int, config: Optional[AttackConfig] = None):
        self.num_agents = num_agents
        self.cfg = config or AttackConfig()

        # Select victim agents (staggered, non-overlapping per constraint)
        self._assign_victims()

        # Per-attack cycling
        self._a1_cycle = _CyclicAttack(self.cfg.a1_on_duration, self.cfg.a1_off_duration)
        self._a2_window = _WindowAttack(self.cfg.a2_window_min, self.cfg.a2_window_max)
        self._a4_cycle = _CyclicAttack(self.cfg.a4_on_duration, self.cfg.a4_off_duration)

        # Replay buffer for A4 (store last 500 steps of observations)
        self._replay_buffer: Dict[int, deque] = {
            aid: deque(maxlen=int(self.cfg.a4_lag * SIM_HZ))
            for aid in self.a4_targets
        }

        # Ground-truth tracking
        self.active_attacks: Dict[str, bool] = {}  # attack_type -> active this step
        self.attack_onset_times: List[Tuple[str, float, int]] = []  # (type, time, agent)
        self.attack_log: List[Dict] = []

    def _assign_victims(self):
        """Assign victim agents ensuring non-overlap and f < floor(N/3)."""
        all_ids = list(range(self.num_agents))
        np.random.shuffle(all_ids)

        idx = 0
        def pick(n):
            nonlocal idx
            chosen = all_ids[idx:idx+n]
            idx += n
            return chosen

        self.a1_targets = pick(self.cfg.a1_target_count)
        # A2 targets links, not agents directly
        self.a3_targets = pick(self.cfg.a3_target_count)
        self.a4_targets = pick(self.cfg.a4_target_count)
        self.a5_targets = pick(self.cfg.a5_target_count)
        # A6 targets a network segment (subset of agents)
        remaining = all_ids[idx:]
        self.a6_segment = set(remaining[:max(1, len(remaining)//2)])

        # A2: pick random links
        link_agents = list(range(self.num_agents))
        np.random.shuffle(link_agents)
        self.a2_links: List[Tuple[int,int]] = []
        for i in range(0, min(self.cfg.a2_target_links * 2, len(link_agents)), 2):
            if i+1 < len(link_agents):
                self.a2_links.append((link_agents[i], link_agents[i+1]))

    def reset(self):
        """Reset for new episode."""
        self._assign_victims()
        self._a1_cycle = _CyclicAttack(self.cfg.a1_on_duration, self.cfg.a1_off_duration)
        self._a2_window = _WindowAttack(self.cfg.a2_window_min, self.cfg.a2_window_max)
        self._a4_cycle = _CyclicAttack(self.cfg.a4_on_duration, self.cfg.a4_off_duration)
        self._replay_buffer = {
            aid: deque(maxlen=int(self.cfg.a4_lag * SIM_HZ))
            for aid in self.a4_targets
        }
        self.active_attacks = {}
        self.attack_onset_times = []
        self.attack_log = []

    # ------------------------------------------------------------------
    # Per-timestep injection
    # ------------------------------------------------------------------

    def step(
        self,
        sim_time: float,
        observations: Dict[int, np.ndarray],
        actions: Dict[int, np.ndarray],
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Set[tuple], Set[int], Optional[Set[int]]]:
        """Apply all attacks for this timestep.

        Returns
        -------
        corrupted_obs : modified observations
        corrupted_actions : modified actions
        jammed_links : set of (i,j) tuples currently jammed
        byzantine_ids : set of agent IDs broadcasting byzantine
        intrusion_segment : set of agent IDs in intruded segment (or None)
        """
        corrupted_obs = {k: v.copy() for k, v in observations.items()}
        corrupted_actions = {k: v.copy() for k, v in actions.items()}
        jammed_links: Set[tuple] = set()
        byzantine_ids: Set[int] = set()
        intrusion_segment: Optional[Set[int]] = None

        step_attacks: Dict[str, bool] = {}

        # --- A1: GPS Spoofing (Eq. 1) ---
        a1_active = self._a1_cycle.is_active(sim_time)
        step_attacks["A1"] = a1_active
        if a1_active:
            for aid in self.a1_targets:
                if aid in corrupted_obs:
                    delta = np.random.uniform(
                        -self.cfg.a1_delta_max, self.cfg.a1_delta_max, size=3
                    )
                    corrupted_obs[aid][:3] += delta  # corrupt position estimate

        # --- A2: Communication Jamming (Eq. 2) ---
        a2_active = self._a2_window.is_active(sim_time)
        step_attacks["A2"] = a2_active
        if a2_active:
            for link in self.a2_links:
                jammed_links.add(link)

        # --- A3: Byzantine Faults (Eq. 3) ---
        # Always active (continuous); actual injection prob is in message_router
        step_attacks["A3"] = True
        byzantine_ids = set(self.a3_targets)

        # --- A4: Replay Attack (Eq. 4) ---
        a4_active = self._a4_cycle.is_active(sim_time)
        step_attacks["A4"] = a4_active
        for aid in self.a4_targets:
            if aid in corrupted_obs:
                # Always store current obs for future replay
                self._replay_buffer[aid].append(corrupted_obs[aid].copy())
                if a4_active and len(self._replay_buffer[aid]) == self._replay_buffer[aid].maxlen:
                    # Inject stale observation from 10 s ago
                    corrupted_obs[aid] = self._replay_buffer[aid][0].copy()

        # --- A5: Malware Injection (Eq. 5) ---
        step_attacks["A5"] = True  # continuous
        for aid in self.a5_targets:
            if aid in corrupted_actions:
                # Intermittent policy drift bounded by ||Delta_pi|| <= 0.15
                if np.random.random() < 0.3:  # state-dependent activation
                    drift = np.random.uniform(
                        -self.cfg.a5_drift_bound, self.cfg.a5_drift_bound,
                        size=corrupted_actions[aid].shape
                    )
                    # Clip drift magnitude
                    drift_norm = np.linalg.norm(drift)
                    if drift_norm > self.cfg.a5_drift_bound:
                        drift = drift * (self.cfg.a5_drift_bound / drift_norm)
                    corrupted_actions[aid] += drift

        # --- A6: Network Intrusion ---
        step_attacks["A6"] = True  # continuous passive + selective inject
        intrusion_segment = self.a6_segment

        # Log onset transitions
        for atype, is_active in step_attacks.items():
            was_active = self.active_attacks.get(atype, False)
            if is_active and not was_active:
                # New onset
                targets = {
                    "A1": self.a1_targets, "A2": [l[0] for l in self.a2_links],
                    "A3": self.a3_targets, "A4": self.a4_targets,
                    "A5": self.a5_targets, "A6": list(self.a6_segment),
                }.get(atype, [])
                for t in targets:
                    self.attack_onset_times.append((atype, sim_time, t))

        self.active_attacks = step_attacks

        self.attack_log.append({
            "time": sim_time,
            "active": step_attacks.copy(),
        })

        return corrupted_obs, corrupted_actions, jammed_links, byzantine_ids, intrusion_segment

    def get_compromised_agents(self) -> Set[int]:
        """Return all currently compromised agent IDs (ground truth)."""
        compromised = set()
        compromised.update(self.a1_targets)
        compromised.update(self.a3_targets)
        compromised.update(self.a4_targets)
        compromised.update(self.a5_targets)
        return compromised


# ---------------------------------------------------------------------------
# IRS Evaluation Metrics  (Section 5.3, Eqs. 22-25)
# ---------------------------------------------------------------------------

class IRSMetricsTracker:
    """Tracks detection, false positive, recovery, and HIC metrics."""

    def __init__(self, detection_window: float = 5.0):
        self.detection_window = detection_window  # seconds (Eq. 22)

        # Ground-truth attack onsets: list of (attack_type, onset_time, agent_id)
        self.attack_onsets: List[Tuple[str, float, int]] = []

        # Detection events: list of (detection_time, agent_id)
        self.detection_events: List[Tuple[float, int]] = []

        # False positive events
        self.false_positive_count: int = 0
        self.total_benign_steps: int = 0

        # Recovery tracking
        self.recovery_times: List[float] = []

        # Human intervention count
        self.human_interventions: int = 0

    def reset(self):
        self.attack_onsets = []
        self.detection_events = []
        self.false_positive_count = 0
        self.total_benign_steps = 0
        self.recovery_times = []
        self.human_interventions = 0

    def record_attack_onset(self, attack_type: str, sim_time: float, agent_id: int):
        self.attack_onsets.append((attack_type, sim_time, agent_id))

    def record_detection(self, sim_time: float, agent_id: int, is_true_positive: bool):
        if is_true_positive:
            self.detection_events.append((sim_time, agent_id))
        else:
            self.false_positive_count += 1

    def record_benign_step(self, is_flagged: bool):
        self.total_benign_steps += 1
        if is_flagged:
            self.false_positive_count += 1

    def record_recovery(self, recovery_time: float):
        self.recovery_times.append(recovery_time)

    def record_human_intervention(self):
        self.human_interventions += 1

    # ------------------------------------------------------------------
    # Compute final metrics
    # ------------------------------------------------------------------

    def compute_metrics(self) -> Dict[str, float]:
        """Compute DR, FPR, RT, HIC per Eqs. 22-25."""
        # DR (Eq. 22): fraction of attacks detected within window
        total_attacks = len(self.attack_onsets)
        detected_in_window = 0
        for atype, onset, aid in self.attack_onsets:
            for det_time, det_aid in self.detection_events:
                if det_aid == aid and 0 <= (det_time - onset) <= self.detection_window:
                    detected_in_window += 1
                    break

        dr = (detected_in_window / total_attacks * 100) if total_attacks > 0 else 0.0

        # FPR (Eq. 23)
        fpr = (self.false_positive_count / max(1, self.total_benign_steps)) * 100

        # RT (Eq. 24)
        rt = float(np.mean(self.recovery_times)) if self.recovery_times else 0.0

        # HIC (Eq. 25)
        hic = self.human_interventions

        return {
            "detection_rate": dr,
            "false_positive_rate": fpr,
            "recovery_time": rt,
            "human_interventions": hic,
            "total_attacks": total_attacks,
            "detected_attacks": detected_in_window,
        }
