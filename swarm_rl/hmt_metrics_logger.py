"""
Centralized HMT Metrics Logger for consistent logging across all approaches
"""

import json
import time
import os
from typing import Dict, Any, Optional
import numpy as np

class HMTMetricsLogger:
    """Centralized logger for all HMT approaches with consistent formatting"""
    
    def __init__(self, log_dir: str, approach: str):
        self.log_dir = log_dir
        self.approach = approach.upper()
        self.metrics_file = os.path.join(log_dir, f"{approach}_metrics.json")
        self.session_start = time.time()
        
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize metrics structure
        self.session_metrics = {
            'approach': self.approach,
            'session_start': self.session_start,
            'episodes': [],
            'steps': [],
            'summary': {}
        }
    
    def log_step_metrics(self, step: int, metrics: Dict[str, Any]):
        """Log step-level metrics with timestamp"""
        timestamped_metrics = {
            'step': step,
            'timestamp': time.time(),
            'elapsed_time': time.time() - self.session_start,
            **metrics
        }
        
        self.session_metrics['steps'].append(timestamped_metrics)
        
        # Print formatted step metrics every 1000 steps
        if step % 1000 == 0:
            self._print_step_summary(step, metrics)
    
    def log_episode_metrics(self, episode: int, metrics: Dict[str, Any]):
        """Log episode-level metrics"""
        timestamped_metrics = {
            'episode': episode,
            'timestamp': time.time(),
            'elapsed_time': time.time() - self.session_start,
            **metrics
        }
        
        self.session_metrics['episodes'].append(timestamped_metrics)
        self._print_episode_summary(episode, metrics)
    
    def log_final_summary(self, final_metrics: Dict[str, Any]):
        """Log final training summary"""
        self.session_metrics['summary'] = {
            'session_end': time.time(),
            'total_duration': time.time() - self.session_start,
            **final_metrics
        }
        
        self._save_metrics()
        self._print_final_summary()
    
    def _print_step_summary(self, step: int, metrics: Dict[str, Any]):
        """Print formatted step summary"""
        print(f"\n📊 {self.approach} Step {step:,} Metrics:")
        
        # Core metrics for all approaches
        if 'performance_boost' in str(metrics) or 'performance_multiplier' in str(metrics):
            for key, value in metrics.items():
                if 'boost' in key or 'multiplier' in key:
                    print(f"   Performance: {value:.3f}x")
                    break
        
        # Approach-specific metrics
        if self.approach == 'SHIELD':
            shield_keys = ['shield_superior_decisions', 'shield_collision_preventions', 
                          'shield_irs_detection_rate', 'shield_security_interventions']
            for key in shield_keys:
                if key in metrics:
                    print(f"   {key.replace('shield_', '').replace('_', ' ').title()}: {metrics[key]}")
        
        elif self.approach == 'SAHO':
            saho_keys = ['saho_commands', 'saho_guidance_strength', 'saho_goal_inference_accuracy']
            for key in saho_keys:
                if key in metrics:
                    print(f"   {key.replace('saho_', '').replace('_', ' ').title()}: {metrics[key]}")
        
        elif self.approach == 'TAMER':
            tamer_keys = ['tamer_feedback_count', 'tamer_model_accuracy', 'tamer_reward_prediction_loss']
            for key in tamer_keys:
                if key in metrics:
                    print(f"   {key.replace('tamer_', '').replace('_', ' ').title()}: {metrics[key]}")
        
        elif self.approach == 'COACH':
            coach_keys = ['coach_corrections', 'coach_effectiveness', 'coach_buffer_size']
            for key in coach_keys:
                if key in metrics:
                    print(f"   {key.replace('coach_', '').replace('_', ' ').title()}: {metrics[key]}")
    
    def _print_episode_summary(self, episode: int, metrics: Dict[str, Any]):
        """Print formatted episode summary"""
        print(f"\n🎯 {self.approach} Episode {episode} Complete:")
        
        # Common episode metrics
        common_keys = ['success_rate', 'collision_rate', 'goal_achievement', 'average_reward']
        for key in common_keys:
            if key in metrics:
                print(f"   {key.replace('_', ' ').title()}: {metrics[key]:.3f}")
        
        # Performance comparison indicator
        if 'performance_comparison' in metrics:
            comparison = metrics['performance_comparison']
            print(f"   Performance vs Baseline: +{comparison:.1f}%")
    
    def _print_final_summary(self):
        """Print final training summary"""
        duration = self.session_metrics['summary']['total_duration']
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        
        print(f"\n🏁 {self.approach} Training Complete!")
        print(f"   Duration: {hours}h {minutes}m")
        print(f"   Total Episodes: {len(self.session_metrics['episodes'])}")
        print(f"   Total Steps: {len(self.session_metrics['steps'])}")
        
        # Calculate final performance summary
        if self.session_metrics['episodes']:
            recent_episodes = self.session_metrics['episodes'][-10:]  # Last 10 episodes
            if recent_episodes:
                avg_success = np.mean([ep.get('success_rate', 0) for ep in recent_episodes])
                avg_collision = np.mean([ep.get('collision_rate', 0) for ep in recent_episodes])
                print(f"   Final Success Rate: {avg_success:.1%}")
                print(f"   Final Collision Rate: {avg_collision:.1%}")
        
        print(f"   Metrics saved to: {self.metrics_file}")
    
    def _save_metrics(self):
        """Save metrics to JSON file"""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.session_metrics, f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️ Error saving metrics: {e}")
    
    def get_comparative_metrics(self) -> Dict[str, float]:
        """Get metrics formatted for comparison with other approaches"""
        if not self.session_metrics['episodes']:
            return {}
        
        recent_episodes = self.session_metrics['episodes'][-5:]  # Last 5 episodes
        
        return {
            f'{self.approach.lower()}_success_rate': np.mean([ep.get('success_rate', 0) for ep in recent_episodes]),
            f'{self.approach.lower()}_collision_rate': np.mean([ep.get('collision_rate', 0) for ep in recent_episodes]),
            f'{self.approach.lower()}_efficiency': np.mean([ep.get('efficiency', 0) for ep in recent_episodes]),
            f'{self.approach.lower()}_stability': np.mean([ep.get('stability', 0) for ep in recent_episodes])
        }


def create_hmt_logger(log_dir: str, approach: str) -> HMTMetricsLogger:
    """Factory function to create HMT metrics logger"""
    return HMTMetricsLogger(log_dir, approach)
