from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType

# Wrapper for compatibility with gym 0.26
# Mostly copied from gym.EnvCompatability
# Modified since swarm_rl does not have a seed, and is a vectorized env
class QuadEnvCompatibility(gym.Wrapper):
    def __init__(self, env):
        """A wrapper which converts old-style step/reset API to new Gymnasium API.

        Args:
            env: the env to wrap with old step/reset API
        """
        super().__init__(env)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[ObsType, dict]:
        """Resets the environment with new API.

        Args:
            seed: the seed to reset the environment with (ignored for quad envs)
            options: the options to reset the environment with (ignored for quad envs)

        Returns:
            (observation, info)
        """
        # Try new API first, fallback to old API
        try:
            # Check if environment already supports new reset API
            if hasattr(self.env, '_new_reset_api'):
                return self.env.reset(seed=seed, options=options)
            else:
                # Use old reset API
                obs = self.env.reset()
                # Ensure observation is properly formatted
                if isinstance(obs, np.ndarray):
                    obs = obs.astype(np.float32)
                elif isinstance(obs, list):
                    obs = [np.array(o, dtype=np.float32) if isinstance(o, (list, np.ndarray)) else o for o in obs]
                return obs, {}
        except TypeError:
            # Fallback to old reset signature
            obs = self.env.reset()
            if isinstance(obs, np.ndarray):
                obs = obs.astype(np.float32)
            elif isinstance(obs, list):
                obs = [np.array(o, dtype=np.float32) if isinstance(o, (list, np.ndarray)) else o for o in obs]
            return obs, {}

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict]:
        """Steps through the environment with new API.

        Args:
            action: action to step through the environment with

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # Get result from environment
        result = self.env.step(action)
        
        # Handle different return formats
        if len(result) == 5:
            # Already in new format (obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = result
        elif len(result) == 4:
            # Old format (obs, reward, done, info)
            obs, reward, done, info = result
            # Convert done to terminated/truncated
            if isinstance(done, (list, np.ndarray)):
                terminated = done
                truncated = [False] * len(done)
            else:
                terminated = done
                truncated = False
        else:
            raise ValueError(f"Unexpected step return format: {len(result)} elements")

        # Ensure observation is properly formatted
        if isinstance(obs, np.ndarray):
            obs = obs.astype(np.float32)
        elif isinstance(obs, list):
            obs = [np.array(o, dtype=np.float32) if isinstance(o, (list, np.ndarray)) else o for o in obs]

        return obs, reward, terminated, truncated, info

    def render(self) -> Any:
        """Render the environment."""
        return self.env.render()
