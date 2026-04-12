"""
Base class for HMT-DRL approaches
"""

import torch
import numpy as np
from abc import ABC, abstractmethod

class BaseHMTTrainer(ABC):
    """Base class for Human-Machine Teaming DRL trainers."""
    
    def __init__(self, cfg, env, agent):
        self.cfg = cfg
        self.env = env
        self.agent = agent
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.step_count = 0
        
    @abstractmethod
    def get_human_guidance(self, obs, action, reward, done, info):
        """Get human guidance for the current step."""
        pass
    
    @abstractmethod
    def update(self, experience_batch):
        """Update the human model/guidance system."""
        pass
    
    def blend_actions(self, rl_action, human_action, weight):
        """Blend RL and human actions based on weight."""
        if human_action is None:
            return rl_action
        return (1 - weight) * rl_action + weight * human_action
    
    def get_metrics(self):
        """Return training metrics specific to this approach."""
        return {}
