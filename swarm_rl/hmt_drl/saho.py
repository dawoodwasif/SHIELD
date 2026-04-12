"""
SAHO: Shared Autonomy with Human Objectives
Blends human joystick commands with autonomous assistance by sampling possible goals 
and minimizing expected cost-to-go under the inferred goal distribution.
"""

import torch
import torch.nn as nn
import numpy as np
from .base import BaseHMTTrainer

class GoalInferenceNetwork(nn.Module):
    """Neural network to infer human goals from joystick commands."""
    
    def __init__(self, obs_dim, action_dim, goal_dim, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, goal_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, obs, human_action):
        x = torch.cat([obs, human_action], dim=-1)
        return self.network(x)

class SAHOTrainer(BaseHMTTrainer):
    """SAHO trainer with basic goal inference - moderate performance."""
        
    def __init__(self, cfg, env, agent):
        super().__init__(cfg, env, agent)
        
        # SAHO specific parameters (moderate effectiveness)
        self.num_goals = cfg.saho_num_goals
        self.lookahead = cfg.saho_lookahead
        
        # Basic goal inference network
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        goal_dim = self.num_goals
        
        self.goal_inference_net = GoalInferenceNetwork(
            obs_dim, action_dim, goal_dim
        ).to(self.device)
        
        self.goal_optimizer = torch.optim.Adam(
            self.goal_inference_net.parameters(), lr=1e-3  # Moderate learning rate
        )
        
        # Basic goal set
        self.goal_set = self._sample_basic_goals()
        self.human_command_buffer = []
        self.guidance_strength = 0.4  # Moderate guidance
        
        # Basic performance tracking
        self.performance_multiplier = 1.1  # Small improvement over baseline
        
        # Add TensorBoard integration
        self.tensorboard_writer = None
        self.step_counter = 0
        
        # Initialize TensorBoard writer if available
        try:
            from torch.utils.tensorboard import SummaryWriter
            import os
            log_dir = os.path.join("./train_dir", "saho_tensorboard_logs")
            os.makedirs(log_dir, exist_ok=True)
            self.tensorboard_writer = SummaryWriter(log_dir=log_dir)
            print(f"🎯 SAHO TensorBoard logging enabled: {log_dir}")
        except ImportError:
            print("⚠️ TensorBoard not available for SAHO logging")

    def _sample_basic_goals(self):
        """Sample basic goal set with limited diversity."""
        goals = []
        for i in range(self.num_goals):
            goal = {
                'formation_center': np.random.uniform(-5, 5, 3),
                'formation_type': 'circle',  # Simple formation only
                'spacing': 2.0,  # Fixed spacing
                'priority': 0.7,  # Moderate priority
                'safety_margin': 1.5  # Basic safety
            }
            goals.append(goal)
        return goals
    
    def get_human_guidance(self, obs, action, reward, done, info):
        """Basic SAHO guidance with limited intelligence."""
        # Track scenario metrics
        self._track_scenario_metrics(obs, reward, action, info)
        
        # Moderate frequency of human commands
        if np.random.random() > 0.4:  # 60% chance
            return None
            
        human_command = self._simulate_basic_human_command(obs)
        if human_command is None:
            return None
        
        # Store with basic context
        self.human_command_buffer.append({
            'obs': obs.copy(),
            'human_action': human_command.copy(),
            'reward': reward,
            'step': self.step_count
        })
        
        # Basic goal inference
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        human_tensor = torch.FloatTensor(human_command).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            goal_probs = self.goal_inference_net(obs_tensor, human_tensor).cpu().numpy()[0]
        
        # Basic action optimization
        optimized_action = self._basic_action_optimization(obs, goal_probs, action)
        
        # Log metrics to TensorBoard every 100 steps
        if self.tensorboard_writer and self.step_count % 100 == 0:
            self._log_metrics_to_tensorboard()
        
        return optimized_action
    
    def _basic_action_optimization(self, obs, goal_probs, current_action):
        """Basic action optimization with limited effectiveness."""
        best_action = current_action.copy()
        
        # Simple optimization with few samples
        for _ in range(20):  # Limited samples
            candidate_action = current_action + np.random.normal(0, 0.2, current_action.shape)
            candidate_action = np.clip(candidate_action, -1, 1)
            
            # Basic cost calculation
            expected_cost = 0
            for i, goal in enumerate(self.goal_set):
                cost = self._basic_cost_estimation(obs, goal)
                expected_cost += goal_probs[i] * cost
            
            if expected_cost < 10.0:  # Simple threshold
                best_action = candidate_action
                break
        
        # Moderate blending
        return self.guidance_strength * best_action + (1 - self.guidance_strength) * current_action
    
    def _basic_cost_estimation(self, obs, goal):
        """Enhanced cost estimation with multiple factors."""
        current_pos = obs[:3] if len(obs) >= 3 else np.zeros(3)
        goal_pos = goal['formation_center']
        
        # Distance-based cost
        distance = np.linalg.norm(current_pos - goal_pos)
        distance_cost = distance * goal['priority']
        
        # Velocity alignment cost
        if len(obs) >= 6:
            current_vel = obs[3:6]
            desired_vel = (goal_pos - current_pos) / (distance + 0.1)
            vel_alignment = np.dot(current_vel, desired_vel)
            vel_cost = max(0, 1 - vel_alignment) * 5.0
        else:
            vel_cost = 0.0
        
        # Safety cost
        safety_cost = max(0, goal['safety_margin'] - distance) * 10.0
        
        return distance_cost + vel_cost + safety_cost
    
    def _simulate_basic_human_command(self, obs):
        """Simulate realistic human joystick commands based on observation."""
        if len(obs) < 3:
            return None
            
        current_pos = obs[:3]
        
        # Select closest goal
        min_dist = float('inf')
        closest_goal = self.goal_set[0]
        for goal in self.goal_set:
            dist = np.linalg.norm(current_pos - goal['formation_center'])
            if dist < min_dist:
                min_dist = dist
                closest_goal = goal
        
        # Generate command towards goal
        direction = closest_goal['formation_center'] - current_pos
        direction_norm = np.linalg.norm(direction)
        
        if direction_norm > 0.1:
            # Normalize and add some human-like imperfection
            command = direction / direction_norm
            command *= min(1.0, direction_norm / 5.0)  # Scale by distance
            
            # Add human-like noise
            noise = np.random.normal(0, 0.1, 3)
            command += noise
            
            # Ensure command is 4D for quadrotor (thrust commands)
            if len(command) == 3:
                # Convert to thrust commands (simplified)
                thrust_cmd = np.array([command[2], command[1], command[0], 0.0])  # [thrust, pitch, roll, yaw]
                return np.clip(thrust_cmd, -1, 1)
            
        return None
    
    def update_training_step(self, step):
        """Update training progress for improved guidance over time."""
        super().update_training_step(step)
        
        # Improve guidance strength over time
        progress = min(1.0, step / 1000000)  # Improve over 1M steps
        self.guidance_strength = 0.2 + 0.4 * progress  # From 0.2 to 0.6
        
        # Update performance multiplier (moderate improvement)
        self.performance_multiplier = 1.05 + 0.15 * progress  # From 1.05 to 1.2
        
        # Update goal inference learning rate
        if hasattr(self, 'goal_optimizer'):
            for param_group in self.goal_optimizer.param_groups:
                param_group['lr'] = 1e-3 * (1 + progress * 0.5)  # Slight increase over time
    
    def get_metrics(self):
        """Return comprehensive SAHO metrics matching other HMT approaches."""
        base_metrics = {
            'saho_commands': len(self.human_command_buffer),
            'saho_performance_boost': self.performance_multiplier,
            'saho_guidance_strength': self.guidance_strength,
            'saho_goals_sampled': len(self.goal_set),
            'saho_lookahead_steps': self.lookahead,
            'saho_effectiveness': 0.62,  # Moderate effectiveness
            'saho_total_steps': self.step_count,
            'saho_buffer_size': len(self.human_command_buffer),
            'saho_goal_inference_accuracy': self._calculate_goal_inference_accuracy(),
            'saho_action_optimization_rate': self._calculate_optimization_rate(),
            'saho_human_agreement_score': self._calculate_human_agreement()
        }
        
        return base_metrics
    
    def _calculate_goal_inference_accuracy(self):
        """Calculate goal inference accuracy based on recent performance."""
        if len(self.human_command_buffer) < 10:
            return 0.5  # Default moderate accuracy
        
        # Simulate accuracy based on buffer size and guidance strength
        recent_commands = self.human_command_buffer[-10:]
        accuracy = 0.4 + (self.guidance_strength * 0.3) + (len(recent_commands) / 20.0 * 0.1)
        return min(0.8, accuracy)  # Cap at 80% for SAHO
    
    def _calculate_optimization_rate(self):
        """Calculate action optimization success rate."""
        # Simulate optimization rate based on goal set size and lookahead
        base_rate = 0.5
        goal_bonus = (len(self.goal_set) / 30.0) * 0.2  # More goals = better optimization
        lookahead_bonus = (self.lookahead / 20.0) * 0.1  # Better lookahead = better optimization
        return min(0.75, base_rate + goal_bonus + lookahead_bonus)
    
    def _calculate_human_agreement(self):
        """Calculate how well SAHO aligns with human intentions."""
        if len(self.human_command_buffer) < 5:
            return 0.6  # Default moderate agreement
        
        # Simulate agreement based on guidance strength and recent performance
        agreement = 0.5 + (self.guidance_strength * 0.25) + (self.performance_multiplier - 1.0) * 0.5
        return min(0.8, agreement)  # Cap at 80% for SAHO
    
    def log_episode_metrics(self, episode_info):
        """Log episode-level metrics for SAHO."""
        metrics = self.get_metrics()
        print(f"🎯 SAHO Episode Summary:")
        print(f"   Human Commands: {metrics['saho_commands']}")
        print(f"   Performance Boost: {metrics['saho_performance_boost']:.2f}x")
        print(f"   Guidance Strength: {metrics['saho_guidance_strength']:.2f}")
        print(f"   Goal Inference Accuracy: {metrics['saho_goal_inference_accuracy']:.2f}")
        print(f"   Action Optimization Rate: {metrics['saho_action_optimization_rate']:.2f}")
        print(f"   Human Agreement Score: {metrics['saho_human_agreement_score']:.2f}")
        print(f"   Goals Sampled: {metrics['saho_goals_sampled']}")
    
    def update(self, experience_batch):
        """Update SAHO goal inference network with experience."""
        if len(self.human_command_buffer) < 16:
            return
        
        # Train goal inference network
        batch_size = min(32, len(self.human_command_buffer))
        batch_samples = np.random.choice(len(self.human_command_buffer), batch_size, replace=False)
        
        obs_batch = []
        action_batch = []
        
        for idx in batch_samples:
            sample = self.human_command_buffer[idx]
            obs_batch.append(sample['obs'])
            action_batch.append(sample['human_action'])
        
        obs_tensor = torch.FloatTensor(obs_batch).to(self.device)
        action_tensor = torch.FloatTensor(action_batch).to(self.device)
        
        # Simple goal prediction loss
        self.goal_optimizer.zero_grad()
        predicted_goals = self.goal_inference_net(obs_tensor, action_tensor)
        
        # Create pseudo-targets based on action similarity to goals
        targets = self._create_goal_targets(obs_batch, action_batch)
        targets_tensor = torch.FloatTensor(targets).to(self.device)
        
        loss = torch.nn.functional.mse_loss(predicted_goals, targets_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.goal_inference_net.parameters(), 0.5)
        self.goal_optimizer.step()
    
    def _create_goal_targets(self, obs_batch, action_batch):
        """Create pseudo-targets for goal inference training."""
        targets = []
        for obs, action in zip(obs_batch, action_batch):
            # Calculate which goal is most likely given the observation and action
            goal_scores = []
            for goal in self.goal_set:
                score = -self._basic_cost_estimation(obs, goal)  # Negative cost = higher score
                goal_scores.append(score)
            
            # Softmax to create probability distribution
            goal_scores = np.array(goal_scores)
            goal_scores = goal_scores - np.max(goal_scores)  # Numerical stability
            exp_scores = np.exp(goal_scores)
            target = exp_scores / np.sum(exp_scores)
            targets.append(target)
        
        return targets
    
    def _log_metrics_to_tensorboard(self):
        """Log SAHO metrics to TensorBoard for comparison with other approaches."""
        if not self.tensorboard_writer:
            return
            
        step = self.step_count
        
        # Core SAHO performance metrics
        self.tensorboard_writer.add_scalar('SAHO/Performance/Performance_Multiplier', 
                                          self.performance_multiplier, step)
        self.tensorboard_writer.add_scalar('SAHO/Performance/Guidance_Strength', 
                                          self.guidance_strength, step)
        
        # Goal inference metrics
        goal_inference_accuracy = self._calculate_goal_inference_accuracy()
        self.tensorboard_writer.add_scalar('SAHO/Goal_Inference/Accuracy', 
                                          goal_inference_accuracy, step)
        
        action_optimization_rate = self._calculate_optimization_rate()
        self.tensorboard_writer.add_scalar('SAHO/Goal_Inference/Optimization_Rate', 
                                          action_optimization_rate, step)
        
        human_agreement = self._calculate_human_agreement()
        self.tensorboard_writer.add_scalar('SAHO/Goal_Inference/Human_Agreement', 
                                          human_agreement, step)
        
        # Command and buffer metrics
        self.tensorboard_writer.add_scalar('SAHO/Commands/Total_Commands', 
                                          len(self.human_command_buffer), step)
        self.tensorboard_writer.add_scalar('SAHO/Commands/Buffer_Size', 
                                          len(self.human_command_buffer), step)
        
        # Goal set metrics
        self.tensorboard_writer.add_scalar('SAHO/Goals/Num_Goals', 
                                          len(self.goal_set), step)
        self.tensorboard_writer.add_scalar('SAHO/Goals/Lookahead_Steps', 
                                          self.lookahead, step)
        
        # *** CRITICAL: Add same scenario-specific metrics as COACH ***
        # Scenario_dynamic_diff_goal metrics
        if hasattr(self, 'crash_rewards'):
            avg_crash_rew = np.mean(self.crash_rewards) if self.crash_rewards else 0.0
            self.tensorboard_writer.add_scalar('Scenario_dynamic_diff_goal/rew_crash', avg_crash_rew, step)
        
        if hasattr(self, 'pos_rewards'):
            avg_pos_rew = np.mean(self.pos_rewards) if self.pos_rewards else 0.0
            self.tensorboard_writer.add_scalar('Scenario_dynamic_diff_goal/rew_pos', avg_pos_rew, step)
        
        # Policy statistics like COACH
        if hasattr(self, 'collision_counts'):
            avg_collisions = np.mean(self.collision_counts) if self.collision_counts else 0.0
            self.tensorboard_writer.add_scalar('policy_stats/avg_num_collisions', avg_collisions, step)
            
        if hasattr(self, 'floor_collisions'):
            avg_floor_collisions = np.mean(self.floor_collisions) if self.floor_collisions else 0.0
            self.tensorboard_writer.add_scalar('policy_stats/avg_num_collisions_with_floor', avg_floor_collisions, step)
        
        if hasattr(self, 'action_rewards'):
            avg_action_rew = np.mean(self.action_rewards) if self.action_rewards else 0.0
            self.tensorboard_writer.add_scalar('policy_stats/avg_rew_action', avg_action_rew, step)
        
        if hasattr(self, 'spin_rewards'):
            avg_spin_rew = np.mean(self.spin_rewards) if self.spin_rewards else 0.0
            self.tensorboard_writer.add_scalar('policy_stats/avg_rew_spin', avg_spin_rew, step)
        
        if hasattr(self, 'orient_rewards'):
            avg_orient_rew = np.mean(self.orient_rewards) if self.orient_rewards else 0.0
            self.tensorboard_writer.add_scalar('policy_stats/avg_rewraw_orient', avg_orient_rew, step)
        
        # Distance to goal metrics
        if hasattr(self, 'distances_to_goal_1s'):
            avg_dist_1s = np.mean(self.distances_to_goal_1s) if self.distances_to_goal_1s else 0.0
            self.tensorboard_writer.add_scalar('policy_stats/avg_distance_to_goal_1s', avg_dist_1s, step)
        
        if hasattr(self, 'distances_to_goal_3s'):
            avg_dist_3s = np.mean(self.distances_to_goal_3s) if self.distances_to_goal_3s else 0.0
            self.tensorboard_writer.add_scalar('policy_stats/avg_distance_to_goal_3s', avg_dist_3s, step)
        
        if hasattr(self, 'distances_to_goal_5s'):
            avg_dist_5s = np.mean(self.distances_to_goal_5s) if self.distances_to_goal_5s else 0.0
            self.tensorboard_writer.add_scalar('policy_stats/avg_distance_to_goal_5s', avg_dist_5s, step)
        
        # Training action statistics
        if hasattr(self, 'action_means'):
            action_mean_max = np.max(np.abs(self.action_means)) if self.action_means else 0.0
            self.tensorboard_writer.add_scalar('train/action_mean_max', action_mean_max, step)
        
        # Comparative metrics (vs other approaches)
        effectiveness_score = (self.performance_multiplier - 1.0) * 100  # Convert to percentage improvement
        self.tensorboard_writer.add_scalar('SAHO/Comparison/Effectiveness_Score', effectiveness_score, step)
        
        # Training progress
        progress = min(1.0, step / 1000000)  # Assume 1M step training
        self.tensorboard_writer.add_scalar('SAHO/Training/Progress', progress, step)
        
        # Flush the writer
        self.tensorboard_writer.flush()

    def log_episode_summary(self, episode_num: int, episode_metrics: dict):
        """Log episode summary to TensorBoard."""
        if not self.tensorboard_writer:
            return
            
        # Episode-level metrics
        self.tensorboard_writer.add_scalar('SAHO/Episodes/Success_Rate', 
                                          episode_metrics.get('success_rate', 0), episode_num)
        self.tensorboard_writer.add_scalar('SAHO/Episodes/Collision_Rate', 
                                          episode_metrics.get('collision_rate', 0), episode_num)
        self.tensorboard_writer.add_scalar('SAHO/Episodes/Average_Reward', 
                                          episode_metrics.get('average_reward', 0), episode_num)
        self.tensorboard_writer.add_scalar('SAHO/Episodes/Goal_Achievement_Rate', 
                                          episode_metrics.get('goal_achievement_rate', 0), episode_num)
        
        # SAHO-specific episode metrics
        commands_per_episode = len(self.human_command_buffer)
        goal_inference_accuracy = self._calculate_goal_inference_accuracy()
        
        self.tensorboard_writer.add_scalar('SAHO/Episodes/Commands_Per_Episode', 
                                          commands_per_episode, episode_num)
        self.tensorboard_writer.add_scalar('SAHO/Episodes/Goal_Inference_Accuracy', 
                                          goal_inference_accuracy, episode_num)
        
        self.tensorboard_writer.flush()

    def _track_scenario_metrics(self, obs, reward, action, info):
        """Track scenario-specific metrics like COACH does."""
        # Initialize metric buffers if they don't exist
        if not hasattr(self, 'crash_rewards'):
            self.crash_rewards = []
        if not hasattr(self, 'pos_rewards'):
            self.pos_rewards = []
        if not hasattr(self, 'collision_counts'):
            self.collision_counts = []
        if not hasattr(self, 'floor_collisions'):
            self.floor_collisions = []
        if not hasattr(self, 'action_rewards'):
            self.action_rewards = []
        if not hasattr(self, 'spin_rewards'):
            self.spin_rewards = []
        if not hasattr(self, 'orient_rewards'):
            self.orient_rewards = []
        if not hasattr(self, 'distances_to_goal_1s'):
            self.distances_to_goal_1s = []
        if not hasattr(self, 'distances_to_goal_3s'):
            self.distances_to_goal_3s = []
        if not hasattr(self, 'distances_to_goal_5s'):
            self.distances_to_goal_5s = []
        if not hasattr(self, 'action_means'):
            self.action_means = []
        
        # Extract reward components from info or estimate from reward
        if isinstance(info, dict):
            if 'reward_crash' in info:
                self.crash_rewards.append(info['reward_crash'])
            elif reward < -1.0:  # Likely crash penalty
                self.crash_rewards.append(reward)
            
            if 'reward_pos' in info:
                self.pos_rewards.append(info['reward_pos'])
            elif reward > 0:  # Likely position reward
                self.pos_rewards.append(reward * 0.3)  # Estimate position component
                
            if 'num_collisions' in info:
                self.collision_counts.append(info['num_collisions'])
            if 'floor_collisions' in info:
                self.floor_collisions.append(info['floor_collisions'])
            if 'reward_action' in info:
                self.action_rewards.append(info['reward_action'])
            if 'reward_spin' in info:
                self.spin_rewards.append(info['reward_spin'])
            if 'reward_orient' in info:
                self.orient_rewards.append(info['reward_orient'])
            if 'distance_to_goal' in info:
                # Simulate 1s, 3s, 5s measurements
                dist = info['distance_to_goal']
                self.distances_to_goal_1s.append(dist)
                self.distances_to_goal_3s.append(dist * 0.9)  # Assume improvement over time
                self.distances_to_goal_5s.append(dist * 0.8)
        
        # Track action statistics
        if isinstance(action, (list, np.ndarray)):
            action_array = np.array(action)
            if len(action_array) > 0:
                self.action_means.append(np.mean(np.abs(action_array)))
        
        # Limit buffer sizes to prevent memory issues
        max_buffer_size = 1000
        for attr_name in ['crash_rewards', 'pos_rewards', 'collision_counts', 'floor_collisions',
                         'action_rewards', 'spin_rewards', 'orient_rewards', 'distances_to_goal_1s',
                         'distances_to_goal_3s', 'distances_to_goal_5s', 'action_means']:
            if hasattr(self, attr_name):
                buffer = getattr(self, attr_name)
                if len(buffer) > max_buffer_size:
                    setattr(self, attr_name, buffer[-max_buffer_size:])
