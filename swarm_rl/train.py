"""
Main script for training a swarm of quadrotors with SampleFactory and HMT-DRL approaches
"""

import sys

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl

from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env
from swarm_rl.env_wrappers.quadrotor_params import add_quadrotors_env_args, quadrotors_override_defaults
from swarm_rl.models.quad_multi_model import register_models

def add_hmt_drl_args(parser):
    """Add HMT-DRL specific arguments."""
    hmt_group = parser.add_argument_group('HMT-DRL Arguments')
    hmt_group.add_argument(
        '--hmt_approach', 
        type=str, 
        default='none',
        choices=['none', 'saho', 'tamer', 'coach', 'shield'],
        help='HMT-DRL approach to use: none, saho, tamer, coach, or shield'
    )
    hmt_group.add_argument(
        '--hmt_weight', 
        type=float, 
        default=0.1,
        help='Weight for blending HMT guidance with RL (0.0=pure RL, 1.0=pure HMT)'
    )
    hmt_group.add_argument(
        '--hmt_real_human',
        action='store_true',
        help='Use real human input instead of simulated'
    )
    hmt_group.add_argument(
        '--hmt_input_device',
        type=str,
        default='keyboard',
        choices=['keyboard', 'joystick', 'gamepad'],
        help='Input device for real human input'
    )
    
    # SAHO specific
    hmt_group.add_argument('--saho_num_goals', type=int, default=10, help='Number of goals to sample for SAHO')
    hmt_group.add_argument('--saho_lookahead', type=int, default=5, help='Lookahead steps for cost-to-go estimation')
    
    # TAMER specific
    hmt_group.add_argument('--tamer_feedback_freq', type=int, default=10, help='Frequency of human feedback collection')
    hmt_group.add_argument('--tamer_model_update_freq', type=int, default=100, help='Frequency of reward model updates')
    
    # COACH specific
    hmt_group.add_argument('--coach_correction_prob', type=float, default=0.1, help='Probability of human correction')
    hmt_group.add_argument('--coach_imitation_weight', type=float, default=1.0, help='Weight for imitation loss')
    
    # SHIELD specific (Table 15 defaults)
    hmt_group.add_argument('--shield_tau_vac', type=float, default=0.4,
                          help='Vacuity escalation threshold (tau_vac, Table 15)')
    hmt_group.add_argument('--shield_kappa', type=float, default=0.1,
                          help='Trust smoothing constant (kappa, Table 15)')
    hmt_group.add_argument('--shield_beta', type=float, default=0.05,
                          help='Trust decay rate (beta, Table 15)')
    hmt_group.add_argument('--shield_zeta', type=float, default=0.2,
                          help='Quarantine activation threshold (zeta, Table 15)')
    hmt_group.add_argument('--shield_epsilon', type=float, default=0.3,
                          help='KL-divergence override threshold (epsilon, Table 15)')
    hmt_group.add_argument('--shield_delta', type=float, default=0.15,
                          help='Consensus consistency radius (delta, Table 15)')
    hmt_group.add_argument('--shield_enable_irs', action='store_true',
                          help='Enable Intrusion Response System with attack injection')
    hmt_group.add_argument('--shield_human_pattern', type=str, default='competent',
                          choices=['expert', 'competent', 'novice', 'fatigued', 'distracted'],
                          help='Human response pattern for simulation')

    # Legacy aliases (map old names to new ones in code)
    hmt_group.add_argument('--shield_trust_threshold', type=float, default=None,
                          help='(Legacy alias for --shield_zeta)')
    hmt_group.add_argument('--shield_vacuity_threshold', type=float, default=None,
                          help='(Legacy alias for --shield_tau_vac)')
    hmt_group.add_argument('--shield_kl_threshold', type=float, default=None,
                          help='(Legacy alias for --shield_epsilon)')

def register_swarm_components():
    register_env("quadrotor_multi", make_quadrotor_env)
    register_models()

def parse_swarm_cfg(argv=None, evaluation=False):
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    add_quadrotors_env_args(partial_cfg.env, parser)
    add_hmt_drl_args(parser)
    quadrotors_override_defaults(partial_cfg.env, parser)
    final_cfg = parse_full_cfg(parser, argv)

    # Resolve legacy aliases
    if getattr(final_cfg, 'shield_trust_threshold', None) is not None:
        final_cfg.shield_zeta = final_cfg.shield_trust_threshold
    if getattr(final_cfg, 'shield_vacuity_threshold', None) is not None:
        final_cfg.shield_tau_vac = final_cfg.shield_vacuity_threshold
    if getattr(final_cfg, 'shield_kl_threshold', None) is not None:
        final_cfg.shield_epsilon = final_cfg.shield_kl_threshold

    return final_cfg

def main():
    """Script entry point."""
    register_swarm_components()
    cfg = parse_swarm_cfg(evaluation=False)
    
    if cfg.hmt_approach != 'none':
        print(f"Training with HMT-DRL approach: {cfg.hmt_approach.upper()}")
        if cfg.hmt_approach == 'shield':
            print("SHIELD framework enabled with (Table 15):")
            print(f"  tau_vac  = {getattr(cfg, 'shield_tau_vac', 0.4)}")
            print(f"  kappa    = {getattr(cfg, 'shield_kappa', 0.1)}")
            print(f"  beta     = {getattr(cfg, 'shield_beta', 0.05)}")
            print(f"  zeta     = {getattr(cfg, 'shield_zeta', 0.2)}")
            print(f"  epsilon  = {getattr(cfg, 'shield_epsilon', 0.3)}")
            print(f"  delta    = {getattr(cfg, 'shield_delta', 0.15)}")
            print(f"  IRS      = {getattr(cfg, 'shield_enable_irs', False)}")
            print(f"  Operator = {getattr(cfg, 'shield_human_pattern', 'competent')}")
        
        print(f"Human input mode: {'Real' if getattr(cfg, 'hmt_real_human', False) else 'Simulated'}")
        print("HMT guidance applied via environment wrapper")
    else:
        print("Training with standard RL (Baseline DRL)")
    
    # Use standard RL training - SHIELD integration happens in the environment wrapper
    status = run_rl(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())
