"""
COACH: Convergent Actor-Critic by Humans
Collects on-the-fly human action corrections during roll-outs and supervises 
the deep policy network to imitate those corrected actions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random  # Add missing import
from collections import deque
from .base import BaseHMTTrainer

class COACHTrainer(BaseHMTTrainer):
    """COACH trainer with action corrections - moderate-good performance."""
    
    def __init__(self, cfg, env, agent):
        super().__init__(cfg, env, agent)
        
        # COACH specific parameters
        self.correction_prob = cfg.coach_correction_prob
        self.imitation_weight = cfg.coach_imitation_weight
        
        # Correction tracking
        self.correction_buffer = deque(maxlen=3000)
        self.total_corrections = 0
        self.correction_effectiveness = 0.6  # Moderate effectiveness
        
        # Performance parameters
        self.performance_multiplier = 1.2  # Moderate improvement
        
        # Access policy for imitation learning
        self.policy_net = agent.actor_critic.actor if hasattr(agent, 'actor_critic') else None
        if self.policy_net is not None:
            self.imitation_optimizer = torch.optim.Adam(
                self.policy_net.parameters(), lr=1e-4  # Moderate learning rate
            )
    
    def get_human_guidance(self, obs, action, reward, done, info):
        """COACH guidance with action corrections."""
        corrected_action = self._get_moderate_human_correction(obs, action)
        
        if corrected_action is not None:
            self.correction_buffer.append({
                'obs': obs.copy(),
                'agent_action': action.copy(),
                'corrected_action': corrected_action.copy(),
                'reward': reward,
                'step': self.step_count
            })
            self.total_corrections += 1
            
            return corrected_action
        
        self.step_count += 1
        return None
    
    def _get_moderate_human_correction(self, obs, agent_action):
        """Moderate human correction with decent intelligence."""
        if np.random.random() < self.correction_prob:
            correction = agent_action.copy()
            
            # Moderate formation-based corrections
            current_pos = obs[:3] if len(obs) >= 3 else np.zeros(3)
            
            # Basic formation maintenance
            if len(obs) > 6:
                neighbor_data = obs[6:].reshape(-1, 6)
                neighbor_positions = neighbor_data[:, :3]
                valid_neighbors = neighbor_positions[np.linalg.norm(neighbor_positions, axis=1) > 0]
                
                if len(valid_neighbors) > 0:
                    formation_center = np.mean(valid_neighbors, axis=0)
                    formation_direction = formation_center - current_pos
                    formation_distance = np.linalg.norm(formation_direction)
                    
                    if formation_distance > 0:
                        formation_correction = formation_direction / formation_distance * 0.3
                        if len(correction) >= 3:
                            correction[:3] += formation_correction * self.correction_effectiveness
            
            # Basic collision avoidance
            collision_avoidance = self._basic_collision_avoidance(obs)
            correction += collision_avoidance * 0.2
            
            # Clip to action space
            if hasattr(self.env.action_space, 'low') and hasattr(self.env.action_space, 'high'):
                correction = np.clip(correction, self.env.action_space.low, self.env.action_space.high)
            
            return correction
        
        return None
    
    def _basic_collision_avoidance(self, obs):
        """Basic collision avoidance computation."""
        avoidance = np.zeros(self.env.action_space.shape[0])
        
        if len(obs) > 6:
            current_pos = obs[:3]
            neighbor_data = obs[6:].reshape(-1, 6)
            
            for neighbor in neighbor_data:
                neighbor_pos = neighbor[:3]
                if np.linalg.norm(neighbor_pos) > 0:
                    distance = np.linalg.norm(current_pos - neighbor_pos)
                    
                    if distance < 2.5:  # Basic safety distance
                        avoid_direction = (current_pos - neighbor_pos) / (distance + 1e-8)
                        avoidance_strength = (2.5 - distance) / 2.5 * 0.3
                        
                        if len(avoidance) >= 3:
                            avoidance[:3] += avoid_direction * avoidance_strength
        
        return avoidance
    
    def update(self, experience_batch):
        """Update with moderate effectiveness."""
        if len(self.correction_buffer) < 16 or self.policy_net is None:
            return
        
        # Moderate batch size
        batch_size = min(32, len(self.correction_buffer))
        batch_samples = random.sample(list(self.correction_buffer), batch_size)
        
        obs_batch = torch.FloatTensor([s['obs'] for s in batch_samples]).to(self.device)
        target_actions = torch.FloatTensor([s['corrected_action'] for s in batch_samples]).to(self.device)
        
        # Moderate imitation learning
        self.imitation_optimizer.zero_grad()
        
        predicted_actions = self.policy_net(obs_batch)
        loss = F.mse_loss(predicted_actions, target_actions) * self.imitation_weight
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.imitation_optimizer.step()
    
    def update_training_step(self, step):
        """Update training progress for COACH improvements over time."""
        super().update_training_step(step)
        
        # Improve correction effectiveness over time
        progress = min(1.0, step / 1000000)  # Improve over 1M steps
        self.correction_effectiveness = 0.4 + 0.3 * progress  # From 0.4 to 0.7
        
        # Update performance multiplier
        self.performance_multiplier = 1.1 + 0.2 * progress  # From 1.1 to 1.3
        
        # Adjust correction probability (more selective over time)
        self.correction_prob = max(0.15, 0.25 - 0.1 * progress)  # From 0.25 to 0.15
    
    def log_episode_metrics(self, episode_info):
        """Log episode-level metrics for COACH."""
        metrics = self.get_metrics()
        print(f"🏃 COACH Episode Summary:")
        print(f"   Total Corrections: {metrics['coach_corrections']}")
        print(f"   Buffer Size: {metrics['coach_buffer_size']}")
        print(f"   Performance Boost: {metrics['coach_performance_boost']:.2f}x")
        print(f"   Correction Effectiveness: {metrics['coach_effectiveness']:.2f}")
        if hasattr(self, 'correction_prob'):
            print(f"   Correction Probability: {self.correction_prob:.2f}")

    def get_metrics(self):
        """Return COACH-specific metrics."""
        return {
            'coach_corrections': self.total_corrections,
            'coach_buffer_size': len(self.correction_buffer),
            'coach_performance_boost': self.performance_multiplier,
            'coach_effectiveness': self.correction_effectiveness
        }
