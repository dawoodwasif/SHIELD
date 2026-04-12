"""
SHIELD Gym Wrapper -- implements Algorithm 1 per-timestep update.

This wrapper sits around the QuadSwarm multi-agent environment and executes
the full SHIELD control loop each step:

    1. Evidential Perception  (Section 4.1)
    2. Local Action Proposal  (provided by the RL policy)
    3. Communication          (Section 4.2 via MessageRouter)
    4. Trust Update           (Section 4.2, Eqs. 10-13)
    5. Trust-Weighted Fusion  (Section 4.3, Eq. 14)
    6. Safety Arbitration     (Section 4.3, Eqs. 15-17)
    7. Execution

It does NOT modify rewards -- the RL policy trains on the environment's
native reward signal.  SHIELD's contribution is in action filtering
(safety override) and information fusion, not reward shaping.
"""

import gymnasium as gym
import numpy as np
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from swarm_rl.hmt_drl.shield.node import (
    ShieldNode,
    EvidentialPerceptionNet,
    Message,
    softmax,
    kl_divergence,
    NUM_SEMANTIC_CLASSES,
    DEFAULT_TAU_VAC,
    DEFAULT_KAPPA,
    DEFAULT_BETA,
    DEFAULT_ZETA,
    DEFAULT_EPSILON,
    DEFAULT_DELTA,
    DT,
)
from swarm_rl.hmt_drl.shield.coverage import CoverageManager
from swarm_rl.hmt_drl.shield.human_interface import HumanOracle, ORACLE_LATENCY_S
from swarm_rl.hmt_drl.shield.message_router import MessageRouter
from swarm_rl.irs_security_evaluation import (
    AttackInjector,
    AttackConfig,
    IRSMetricsTracker,
)


class ShieldWrapper(gym.Wrapper):
    """SHIELD wrapper implementing Algorithm 1 around a multi-quadrotor env.

    Parameters
    ----------
    env : gym.Env
        The QuadSwarm multi-agent environment.
    enable_irs : bool
        If True, activate the Intrusion Response System with attack injection.
    attack_config : AttackConfig or None
        Custom attack parameters; defaults to Table 2 values.
    """

    def __init__(
        self,
        env,
        enable_irs: bool = False,
        attack_config: Optional[AttackConfig] = None,
    ):
        super().__init__(env)

        self.num_agents: int = getattr(env, "num_agents", 8)

        # Determine obs / action dimensions from the environment
        obs_space = env.observation_space
        if hasattr(obs_space, "shape"):
            self.obs_dim = obs_space.shape[0]
        else:
            self.obs_dim = 18  # default xyz_vxyz_R_omega

        act_space = env.action_space
        if hasattr(act_space, "n"):
            self.num_actions = act_space.n
        elif hasattr(act_space, "shape"):
            self.num_actions = act_space.shape[0]
        else:
            self.num_actions = 7  # paper default (Table 14)

        # --- SHIELD nodes (one per UAV) ---
        self.nodes: Dict[int, ShieldNode] = {}
        for i in range(self.num_agents):
            self.nodes[i] = ShieldNode(
                node_id=i,
                num_actions=self.num_actions,
                obs_dim=self.obs_dim,
            )

        # --- Shared ENN (parameter sharing across UAVs, Section 5.6) ---
        self.shared_enn = EvidentialPerceptionNet(obs_dim=self.obs_dim)
        for node in self.nodes.values():
            node.enn = self.shared_enn

        # --- Communication ---
        self.router = MessageRouter(nodes=self.nodes)

        # --- Human Oracle (Section 5.4) ---
        self.oracle = HumanOracle()

        # --- Coverage Manager (Section 4.4) ---
        self.coverage = CoverageManager(num_agents=self.num_agents)

        # --- IRS / Attack Injection ---
        self.enable_irs = enable_irs
        self.attack_injector: Optional[AttackInjector] = None
        self.metrics_tracker = IRSMetricsTracker()
        if enable_irs:
            cfg = attack_config or AttackConfig()
            self.attack_injector = AttackInjector(self.num_agents, cfg)

        # --- Simulation clock ---
        self.sim_time: float = 0.0
        self.sim_step: int = 0

        # --- Metric accumulators ---
        self.episode_escalations: int = 0
        self.episode_overrides: int = 0
        self.episode_quarantines: int = 0

        # TensorBoard (optional)
        self._tb_writer = None

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs = result
            info = {}

        # Reset SHIELD state
        self.sim_time = 0.0
        self.sim_step = 0
        self.episode_escalations = 0
        self.episode_overrides = 0
        self.episode_quarantines = 0

        for node in self.nodes.values():
            node.sim_time = 0.0
            node.sim_step = 0
            node.is_quarantined = False
            node.neighbors.clear()
            node.belief = np.ones(NUM_SEMANTIC_CLASSES) / NUM_SEMANTIC_CLASSES
            node.vacuity = 1.0
            node.action_dist = np.ones(self.num_actions) / self.num_actions
            node.escalation_pending = False
            node.irs_alert_active = False

        self.oracle.reset()
        self.coverage.reset(self.num_agents)
        self.metrics_tracker.reset()
        self.router.clear_attacks()

        if self.attack_injector is not None:
            self.attack_injector.reset()

        return obs, info

    def step(self, actions):
        """Execute one SHIELD-augmented timestep (Algorithm 1)."""
        self.sim_time += DT
        self.sim_step += 1

        # Update node clocks
        for node in self.nodes.values():
            node.sim_time = self.sim_time
            node.sim_step = self.sim_step

        # ----------------------------------------------------------------
        # 0. Get current agent positions from the environment
        # ----------------------------------------------------------------
        positions = self._get_positions()

        # ----------------------------------------------------------------
        # 1. Build per-agent observations dict
        # ----------------------------------------------------------------
        # The RL framework provides actions as a flat array or list;
        # we need to distribute to per-agent.
        per_agent_actions = self._to_per_agent(actions)

        # ----------------------------------------------------------------
        # 2. Attack injection (if IRS enabled)
        # ----------------------------------------------------------------
        jammed_links: Set[tuple] = set()
        byzantine_ids: Set[int] = set()
        intrusion_segment: Optional[Set[int]] = None

        if self.attack_injector is not None:
            # Build obs dict from previous step's positions for attack injection
            obs_dict = {i: positions.get(i, np.zeros(3)) for i in range(self.num_agents)}
            act_dict = {i: per_agent_actions.get(i, np.zeros(self.num_actions))
                        for i in range(self.num_agents)}

            _, corrupted_actions, jammed_links, byzantine_ids, intrusion_segment = (
                self.attack_injector.step(self.sim_time, obs_dict, act_dict)
            )

            # Apply malware-corrupted actions (A5)
            for aid, ca in corrupted_actions.items():
                per_agent_actions[aid] = ca

            # Feed attack state to router
            self.router.set_jammed_links(jammed_links)
            self.router.set_byzantine_agents(byzantine_ids)
            self.router.set_intrusion_segment(intrusion_segment)

            # Record onsets for metrics
            for atype, onset_time, aid in self.attack_injector.attack_onset_times:
                self.metrics_tracker.record_attack_onset(atype, onset_time, aid)
            self.attack_injector.attack_onset_times.clear()

        # ----------------------------------------------------------------
        # 3. SHIELD per-agent loop (Algorithm 1)
        # ----------------------------------------------------------------
        # 3a. Evidential perception for each node
        for nid, node in self.nodes.items():
            if node.is_quarantined:
                continue
            obs_i = positions.get(nid, np.zeros(self.obs_dim))
            node.state = obs_i[:3] if len(obs_i) >= 3 else obs_i.copy()
            # Step 1: perception
            node.evidential_perception(obs_i)

            # Step 2: local action proposal
            act_i = per_agent_actions.get(nid, np.zeros(self.num_actions))
            node.action_dist = softmax(act_i) if len(act_i) == self.num_actions else (
                np.ones(self.num_actions) / self.num_actions
            )

        # 3b. Communication (Step 3)
        delivered = self.router.deliver_messages(
            {nid: n.state for nid, n in self.nodes.items() if not n.is_quarantined}
        )
        for nid, msgs in delivered.items():
            if nid in self.nodes and not self.nodes[nid].is_quarantined:
                self.nodes[nid].receive_messages(msgs)

        # 3c. Trust update + quarantine (Step 4)
        all_quarantined: List[int] = []
        for nid, node in self.nodes.items():
            if node.is_quarantined:
                continue
            q_list = node.update_trust()
            if q_list:
                all_quarantined.extend(q_list)

        # Execute quarantines
        for qid in set(all_quarantined):
            if qid in self.nodes and not self.nodes[qid].is_quarantined:
                self.nodes[qid].is_quarantined = True
                self.nodes[qid].quarantine_step = self.sim_step
                self.episode_quarantines += 1

                # Reassign coverage (Section 4.4)
                self.coverage.reassign_coverage(qid)

                # IRS metrics
                if self.attack_injector is not None:
                    is_tp = qid in self.attack_injector.get_compromised_agents()
                    self.metrics_tracker.record_detection(
                        self.sim_time, qid, is_true_positive=is_tp
                    )
                    if not is_tp:
                        self.metrics_tracker.record_benign_step(is_flagged=True)

        # 3d. Trust-weighted fusion + safety arbitration (Steps 5-6)
        final_actions = per_agent_actions.copy()
        for nid, node in self.nodes.items():
            if node.is_quarantined:
                continue

            # Step 5: fusion
            _, _, p_bar = node.trust_weighted_fusion()

            # Step 6: safety arbitration
            pi_local = node.action_dist
            action_idx, overridden = node.safety_arbitration(pi_local, p_bar)

            if overridden:
                self.episode_overrides += 1

            # Vacuity-triggered escalation (Step 1, lines 4-6)
            if node.needs_escalation() and not node.escalation_pending:
                node.escalation_pending = True
                self.episode_escalations += 1

                # Get oracle response
                gt_label = None  # would come from ground-truth if available
                response = self.oracle.respond_to_escalation(
                    nid, node.belief, node.vacuity, gt_label
                )
                if response["corrected_belief"] is not None:
                    node.apply_human_correction(response["corrected_belief"])
                else:
                    node.vacuity = 0.0
                    node.escalation_pending = False

                self.metrics_tracker.record_human_intervention()

            # For discrete action spaces, convert index back; for continuous,
            # use the action as-is (the override doesn't change the action
            # format, only selects argmax of consensus).

        # 3e. Reintegration check (Section 4.4)
        for nid, node in self.nodes.items():
            if node.is_quarantined:
                # Check if reintegration criteria are met
                node.evidential_perception(
                    positions.get(nid, np.zeros(self.obs_dim))
                )
                # Simplified consistency: is vacuity low?
                node.record_reintegration_consistency(node.vacuity <= node.tau_vac)
                if node.check_reintegration(self.sim_step):
                    node.reintegrate()

        # ----------------------------------------------------------------
        # 4. Execute in environment (no action or reward modification)
        # ----------------------------------------------------------------
        result = self.env.step(actions)

        if len(result) == 4:
            obs, rewards, dones, infos = result
            terminated = dones
            truncated = [False] * len(dones) if isinstance(dones, list) else False
        elif len(result) == 5:
            obs, rewards, terminated, truncated, infos = result
        else:
            raise ValueError(f"Unexpected step return format: {len(result)} elements")

        # ----------------------------------------------------------------
        # 5. Attach SHIELD metrics to info
        # ----------------------------------------------------------------
        shield_info = self._collect_metrics()

        if isinstance(infos, list):
            for info in infos:
                if isinstance(info, dict):
                    info["shield"] = shield_info
        elif isinstance(infos, dict):
            infos["shield"] = shield_info

        # Benign-step tracking for FPR
        if self.attack_injector is not None:
            compromised = self.attack_injector.get_compromised_agents()
            for nid in range(self.num_agents):
                if nid not in compromised:
                    flagged = self.nodes[nid].is_quarantined
                    self.metrics_tracker.record_benign_step(is_flagged=flagged)

        return obs, rewards, terminated, truncated, infos

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_positions(self) -> Dict[int, np.ndarray]:
        """Extract 3-D positions from the environment state."""
        positions = {}
        if hasattr(self.env, "pos") and self.env.pos is not None:
            for i in range(min(self.num_agents, len(self.env.pos))):
                positions[i] = self.env.pos[i].copy()
        else:
            # Fallback: zeros
            for i in range(self.num_agents):
                positions[i] = np.zeros(3)
        return positions

    def _to_per_agent(self, actions) -> Dict[int, np.ndarray]:
        """Convert flat/list actions to per-agent dict."""
        result = {}
        if isinstance(actions, dict):
            return {int(k): np.asarray(v, dtype=np.float32) for k, v in actions.items()}
        actions_arr = np.asarray(actions, dtype=np.float32)
        if actions_arr.ndim == 0:
            # Single scalar action
            for i in range(self.num_agents):
                result[i] = actions_arr.reshape(-1)
        elif actions_arr.ndim == 1:
            if len(actions_arr) == self.num_agents:
                for i in range(self.num_agents):
                    result[i] = np.array([actions_arr[i]], dtype=np.float32)
            else:
                for i in range(self.num_agents):
                    result[i] = actions_arr.copy()
        elif actions_arr.ndim == 2:
            for i in range(min(self.num_agents, len(actions_arr))):
                result[i] = actions_arr[i].copy()
        else:
            for i in range(self.num_agents):
                result[i] = np.zeros(self.num_actions, dtype=np.float32)
        return result

    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect per-step SHIELD metrics."""
        active_agents = sum(1 for n in self.nodes.values() if not n.is_quarantined)
        quarantined_agents = self.num_agents - active_agents

        avg_vacuity = float(np.mean([
            n.vacuity for n in self.nodes.values() if not n.is_quarantined
        ])) if active_agents > 0 else 0.0

        avg_trust = 0.0
        trust_count = 0
        for n in self.nodes.values():
            for edge in n.neighbors.values():
                avg_trust += edge.trust
                trust_count += 1
        avg_trust = avg_trust / max(1, trust_count)

        metrics = {
            "sim_time": self.sim_time,
            "sim_step": self.sim_step,
            "active_agents": active_agents,
            "quarantined_agents": quarantined_agents,
            "avg_vacuity": avg_vacuity,
            "avg_trust": avg_trust,
            "episode_escalations": self.episode_escalations,
            "episode_overrides": self.episode_overrides,
            "episode_quarantines": self.episode_quarantines,
            "human_interventions": self.oracle.total_interventions,
        }

        if self.enable_irs:
            metrics["irs"] = self.metrics_tracker.compute_metrics()

        return metrics

    # ------------------------------------------------------------------
    # Public API for external metric queries
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Return comprehensive SHIELD metrics."""
        return self._collect_metrics()

    def get_irs_results(self) -> Dict[str, float]:
        """Return IRS evaluation metrics (Eqs. 22-25)."""
        return self.metrics_tracker.compute_metrics()

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()
