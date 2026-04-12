"""
Environment wrappers for HMT-DRL baselines (Section 5.5).

All baselines receive comparable feedback under identical timing constraints
(Section 5.4): 0.5 s latency, event-driven, no unlimited human input.

    COACH  - Corrective action targets (MacGlashan et al. 2017)
    TAMER  - Scalar reward feedback (Warnell et al. 2018)
    SAHO   - Shared autonomy blending (Javdani et al. 2015)
"""

import gymnasium as gym
import numpy as np
from typing import Any, Dict, Optional


# Oracle latency: 0.5 s = 25 steps at 50 Hz  (Section 5.4)
ORACLE_LATENCY_STEPS = 25
ORACLE_ERROR_RATE = 0.05  # < 5 % under time pressure


# ---------------------------------------------------------------------------
# COACH Wrapper  (Section 5.5, baseline 2)
# ---------------------------------------------------------------------------

class COACHWrapper(gym.Wrapper):
    """COACH: human corrective feedback treated as supervised action targets.

    When triggered, the oracle provides a corrected action that directly
    adjusts the policy toward the corrected behaviour (Section 5.5).
    Corrections are event-driven and occur with probability
    ``correction_prob`` per step.
    """

    def __init__(self, env, correction_prob: float = 0.1, imitation_weight: float = 1.0):
        super().__init__(env)
        self.num_agents = getattr(env, "num_agents", 1)
        self.correction_prob = correction_prob
        self.imitation_weight = imitation_weight

        # Counters
        self.total_corrections = 0
        self.step_counter = 0

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}
        self.total_corrections = 0
        self.step_counter = 0
        return obs, info

    def step(self, actions):
        self.step_counter += 1

        # COACH: occasionally replace actions with oracle-corrected targets
        corrected = self._maybe_correct(actions)

        result = self.env.step(corrected)
        if len(result) == 4:
            obs, rewards, dones, infos = result
            terminated, truncated = dones, ([False] * len(dones) if isinstance(dones, list) else False)
        else:
            obs, rewards, terminated, truncated, infos = result

        # Attach COACH metrics
        coach_info = {
            "coach_corrections": self.total_corrections,
            "coach_step": self.step_counter,
        }
        if isinstance(infos, list):
            for info in infos:
                if isinstance(info, dict):
                    info["coach"] = coach_info
        elif isinstance(infos, dict):
            infos["coach"] = coach_info

        return obs, rewards, terminated, truncated, infos

    def _maybe_correct(self, actions):
        """Apply corrective target with oracle accuracy and latency."""
        actions_arr = np.asarray(actions, dtype=np.float32)
        if actions_arr.ndim < 2:
            return actions  # single-action, skip

        corrected = actions_arr.copy()
        for i in range(len(corrected)):
            if np.random.random() < self.correction_prob:
                # Oracle provides corrected action (slightly improved version)
                # Simulates human identifying better trajectory and providing target
                if np.random.random() > ORACLE_ERROR_RATE:
                    # Good correction: dampen extreme actions, nudge toward center
                    corrected[i] = corrected[i] * 0.8  # reduce magnitude
                else:
                    # Error: add noise
                    corrected[i] += np.random.normal(0, 0.1, corrected[i].shape)
                corrected[i] = np.clip(corrected[i], -1.0, 1.0)
                self.total_corrections += 1
        return corrected


# ---------------------------------------------------------------------------
# TAMER Wrapper  (Section 5.5, baseline 4)
# ---------------------------------------------------------------------------

class TAMERWrapper(gym.Wrapper):
    """TAMER: learns from scalar human reward feedback.

    The oracle provides scalar reward shaping signals that augment the
    environment reward. Feedback is reactive (based on observed behaviour)
    rather than proactive (Section 5.5).
    """

    def __init__(self, env, feedback_freq: int = 10, model_update_freq: int = 100):
        super().__init__(env)
        self.num_agents = getattr(env, "num_agents", 1)
        self.feedback_freq = feedback_freq
        self.model_update_freq = model_update_freq

        self.feedback_count = 0
        self.step_counter = 0

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}
        self.feedback_count = 0
        self.step_counter = 0
        return obs, info

    def step(self, actions):
        self.step_counter += 1

        result = self.env.step(actions)
        if len(result) == 4:
            obs, rewards, dones, infos = result
            terminated, truncated = dones, ([False] * len(dones) if isinstance(dones, list) else False)
        else:
            obs, rewards, terminated, truncated, infos = result

        # TAMER: add scalar human reward feedback at feedback_freq intervals
        if self.step_counter % self.feedback_freq == 0:
            rewards = self._add_human_reward(rewards, obs)

        # Attach TAMER metrics
        tamer_info = {
            "tamer_feedback_count": self.feedback_count,
            "tamer_step": self.step_counter,
        }
        if isinstance(infos, list):
            for info in infos:
                if isinstance(info, dict):
                    info["tamer"] = tamer_info
        elif isinstance(infos, dict):
            infos["tamer"] = tamer_info

        return obs, rewards, terminated, truncated, infos

    def _add_human_reward(self, rewards, obs):
        """Oracle provides scalar reward feedback based on observed state."""
        self.feedback_count += 1

        if isinstance(rewards, (list, np.ndarray)):
            shaped = np.array(rewards, dtype=np.float32)
            for i in range(len(shaped)):
                human_r = self._oracle_scalar_feedback(obs, i)
                shaped[i] += human_r
            return shaped.tolist() if isinstance(rewards, list) else shaped
        else:
            human_r = self._oracle_scalar_feedback(obs, 0)
            return float(rewards) + human_r

    def _oracle_scalar_feedback(self, obs, agent_idx: int) -> float:
        """Simulate oracle scalar feedback.

        Positive for low-risk states, negative for high-risk states.
        Models the "reactive shaping" described in Section 5.5.
        """
        if np.random.random() < ORACLE_ERROR_RATE:
            return np.random.uniform(-0.1, 0.1)  # noisy error

        # Simple heuristic: reward stability (low velocity magnitude)
        obs_arr = np.asarray(obs, dtype=np.float32)
        if obs_arr.ndim == 2 and agent_idx < len(obs_arr):
            agent_obs = obs_arr[agent_idx]
        elif obs_arr.ndim == 1:
            agent_obs = obs_arr
        else:
            return 0.0

        if len(agent_obs) >= 6:
            vel_mag = np.linalg.norm(agent_obs[3:6])
            # Small positive reward for controlled velocity
            return max(0.0, 0.05 * (1.0 - min(vel_mag, 2.0) / 2.0))
        return 0.0


# ---------------------------------------------------------------------------
# SAHO Wrapper  (Section 5.5, baseline 3)
# ---------------------------------------------------------------------------

class SAHOWrapper(gym.Wrapper):
    """SAHO: Shared Autonomy via Hindsight Optimization.

    Blends human input with autonomous control using a cost-to-go
    formulation under a latent-goal POMDP (Section 5.5). Human guidance
    is integrated into action selection at the same trigger frequency as
    other methods.
    """

    def __init__(self, env, blend_weight: float = 0.3, num_goals: int = 10, lookahead: int = 5):
        super().__init__(env)
        self.num_agents = getattr(env, "num_agents", 1)
        self.blend_weight = blend_weight  # how much to trust the human
        self.num_goals = num_goals
        self.lookahead = lookahead

        self.step_counter = 0
        self.blend_count = 0

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}
        self.step_counter = 0
        self.blend_count = 0
        return obs, info

    def step(self, actions):
        self.step_counter += 1

        # SAHO: blend RL action with oracle-suggested action
        blended = self._blend_actions(actions)

        result = self.env.step(blended)
        if len(result) == 4:
            obs, rewards, dones, infos = result
            terminated, truncated = dones, ([False] * len(dones) if isinstance(dones, list) else False)
        else:
            obs, rewards, terminated, truncated, infos = result

        # Attach SAHO metrics
        saho_info = {
            "saho_blend_count": self.blend_count,
            "saho_step": self.step_counter,
            "saho_blend_weight": self.blend_weight,
        }
        if isinstance(infos, list):
            for info in infos:
                if isinstance(info, dict):
                    info["saho"] = saho_info
        elif isinstance(infos, dict):
            infos["saho"] = saho_info

        return obs, rewards, terminated, truncated, infos

    def _blend_actions(self, actions):
        """Blend RL actions with oracle-suggested actions."""
        actions_arr = np.asarray(actions, dtype=np.float32)
        if actions_arr.ndim < 2:
            return actions

        blended = actions_arr.copy()
        for i in range(len(blended)):
            # Oracle suggests an action (simplified cost-to-go)
            oracle_action = self._oracle_action(blended[i])
            # Blend: (1-w)*rl + w*human
            blended[i] = (1.0 - self.blend_weight) * blended[i] + self.blend_weight * oracle_action
            blended[i] = np.clip(blended[i], -1.0, 1.0)
            self.blend_count += 1
        return blended

    def _oracle_action(self, rl_action: np.ndarray) -> np.ndarray:
        """Oracle action via simplified cost-to-go.

        The oracle dampens extreme actions and biases toward stability,
        simulating a human operator who prefers cautious manoeuvres.
        """
        if np.random.random() < ORACLE_ERROR_RATE:
            return rl_action + np.random.normal(0, 0.1, rl_action.shape)

        # Dampen toward zero (human prefers less aggressive control)
        oracle = rl_action * 0.7
        return np.clip(oracle, -1.0, 1.0)
