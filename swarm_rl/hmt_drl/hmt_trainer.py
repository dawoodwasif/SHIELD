"""
Custom training loop for HMT-DRL approaches
"""

import time
import torch
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.utils.utils import log

from .saho import SAHOTrainer
from .tamer import TAMERTrainer
from .coach import COACHTrainer

def create_hmt_trainer(cfg, env, agent):
    """Create the appropriate HMT-DRL trainer."""
    if cfg.hmt_approach == 'saho':
        return SAHOTrainer(cfg, env, agent)
    elif cfg.hmt_approach == 'tamer':
        return TAMERTrainer(cfg, env, agent)
    elif cfg.hmt_approach == 'coach':
        return COACHTrainer(cfg, env, agent)
    else:
        return None

def run_hmt_rl(cfg):
    """Custom training loop with HMT-DRL integration."""
    # Import here to avoid circular imports
    from sample_factory.train import make_runner
    
    # Create the base RL runner
    runner = make_runner(cfg)
    
    # Get environment and agent from runner
    env = make_env_func_batched(cfg, env_config=None)(0, 0, 'train')
    agent = runner.algo
    
    # Create HMT trainer
    hmt_trainer = create_hmt_trainer(cfg, env, agent)
    
    if hmt_trainer is None:
        log.error(f"Unknown HMT approach: {cfg.hmt_approach}")
        return 1
    
    log.info(f"Starting HMT-DRL training with {cfg.hmt_approach.upper()}")
    
    # Override the agent's training step to incorporate HMT guidance
    original_train_step = agent.train_step if hasattr(agent, 'train_step') else None
    
    def hmt_train_step(batch):
        """Modified training step with HMT guidance."""
        # Regular RL training step
        if original_train_step:
            rl_result = original_train_step(batch)
        else:
            rl_result = {}
        
        # Update HMT trainer
        hmt_trainer.update(batch)
        
        # Add HMT metrics
        hmt_metrics = hmt_trainer.get_metrics()
        rl_result.update(hmt_metrics)
        
        return rl_result
    
    # Replace the training step
    if hasattr(agent, 'train_step'):
        agent.train_step = hmt_train_step
    
    # Run the training
    status = runner.run()
    
    return status
