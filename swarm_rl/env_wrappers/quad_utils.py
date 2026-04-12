import copy
import gymnasium as gym

import torch
from sample_factory.algo.learning.learner import Learner
from sample_factory.model.actor_critic import create_actor_critic

from gym_art.quadrotor_multi.quad_experience_replay import ExperienceReplayWrapper
from swarm_rl.env_wrappers.compatibility import QuadEnvCompatibility
from swarm_rl.env_wrappers.reward_shaping import DEFAULT_QUAD_REWARD_SHAPING, QuadsRewardShapingWrapper
from swarm_rl.env_wrappers.v_value_map import V_ValueMapWrapper


class AnnealSchedule:
    def __init__(self, coeff_name, final_value, anneal_env_steps):
        self.coeff_name = coeff_name
        self.final_value = final_value
        self.anneal_env_steps = anneal_env_steps


def make_quadrotor_env_multi(cfg, render_mode=None, **kwargs):
    from gym_art.quadrotor_multi.quadrotor_multi import QuadrotorEnvMulti
    quad = 'Crazyflie'
    dyn_randomize_every = dyn_randomization_ratio = None
    raw_control = raw_control_zero_middle = True

    sampler_1 = None
    if dyn_randomization_ratio is not None:
        sampler_1 = dict(type='RelativeSampler', noise_ratio=dyn_randomization_ratio, sampler='normal')

    sense_noise = 'default'
    dynamics_change = dict(noise=dict(thrust_noise_ratio=0.05), damp=dict(vel=0, omega_quadratic=0))

    rew_coeff = DEFAULT_QUAD_REWARD_SHAPING['quad_rewards']
    use_replay_buffer = cfg.replay_buffer_sample_prob > 0.0

    env = QuadrotorEnvMulti(
        num_agents=cfg.quads_num_agents, ep_time=cfg.quads_episode_duration, rew_coeff=rew_coeff,
        obs_repr=cfg.quads_obs_repr,
        # Neighbor
        neighbor_visible_num=cfg.quads_neighbor_visible_num, neighbor_obs_type=cfg.quads_neighbor_obs_type,
        collision_hitbox_radius=cfg.quads_collision_hitbox_radius,
        collision_falloff_radius=cfg.quads_collision_falloff_radius,
        # Obstacle - FIX: Use cfg parameter instead of hardcoded False
        use_obstacles=cfg.quads_use_obstacles, obst_density=cfg.quads_obst_density, obst_size=cfg.quads_obst_size,
        obst_spawn_area=cfg.quads_obst_spawn_area,

        # Aerodynamics
        use_downwash=cfg.quads_use_downwash,
        # Numba Speed Up
        use_numba=cfg.quads_use_numba,
        # Scenarios
        quads_mode=cfg.quads_mode,
        # Room
        room_dims=cfg.quads_room_dims,
        # Replay Buffer
        use_replay_buffer=use_replay_buffer,
        # Rendering
        quads_view_mode=cfg.quads_view_mode, quads_render=cfg.quads_render,
        # Quadrotor Specific (Do Not Change)
        dynamics_params=quad, raw_control=raw_control, raw_control_zero_middle=raw_control_zero_middle,
        dynamics_randomize_every=dyn_randomize_every, dynamics_change=dynamics_change, dyn_sampler_1=sampler_1,
        sense_noise=sense_noise, init_random_state=False,
        # Rendering
        render_mode=render_mode,
    )

    # Fix observation and action spaces to be Gymnasium compatible BEFORE other wrappers
    env = _fix_gymnasium_spaces(env)

    if use_replay_buffer:
        env = ExperienceReplayWrapper(env, cfg.replay_buffer_sample_prob, cfg.quads_obst_density, cfg.quads_obst_size,
                                      cfg.quads_domain_random, cfg.quads_obst_density_random, cfg.quads_obst_size_random,
                                      cfg.quads_obst_density_min, cfg.quads_obst_density_max, cfg.quads_obst_size_min, cfg.quads_obst_size_max)

    reward_shaping = copy.deepcopy(DEFAULT_QUAD_REWARD_SHAPING)

    reward_shaping['quad_rewards']['quadcol_bin'] = cfg.quads_collision_reward
    reward_shaping['quad_rewards']['quadcol_bin_smooth_max'] = cfg.quads_collision_smooth_max_penalty
    reward_shaping['quad_rewards']['quadcol_bin_obst'] = cfg.quads_obst_collision_reward

    # this is annealed by the reward shaping wrapper
    if cfg.anneal_collision_steps > 0:
        reward_shaping['quad_rewards']['quadcol_bin'] = 0.0
        reward_shaping['quad_rewards']['quadcol_bin_smooth_max'] = 0.0
        reward_shaping['quad_rewards']['quadcol_bin_obst'] = 0.0
        annealing = [
            AnnealSchedule('quadcol_bin', cfg.quads_collision_reward, cfg.anneal_collision_steps),
            AnnealSchedule('quadcol_bin_smooth_max', cfg.quads_collision_smooth_max_penalty,
                           cfg.anneal_collision_steps),
            AnnealSchedule('quadcol_bin_obst', cfg.quads_obst_collision_reward, cfg.anneal_collision_steps),
        ]
    else:
        annealing = None

    env = QuadsRewardShapingWrapper(env, reward_shaping_scheme=reward_shaping, annealing=annealing,
                                    with_pbt=cfg.with_pbt)

    # Apply compatibility wrapper at the END to handle step/reset API
    env = QuadEnvCompatibility(env)

    if cfg.visualize_v_value:
        actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
        actor_critic.eval()

        device = torch.device("cpu" if cfg.device == "cpu" else "cuda")
        actor_critic.model_to_device(device)

        policy_id = cfg.policy_index
        name_prefix = dict(latest="checkpoint", best="best")[cfg.load_checkpoint_kind]
        checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(cfg, policy_id), f"{name_prefix}_*")
        checkpoint_dict = Learner.load_checkpoint(checkpoints, device)
        actor_critic.load_state_dict(checkpoint_dict["model"])
        env = V_ValueMapWrapper(env, actor_critic)

    return env


def make_quadrotor_env_single(cfg, render_mode=None, **kwargs):
    """Create single-agent quadrotor environment"""
    from gym_art.quadrotor_multi.quadrotor_multi import QuadrotorEnvMulti
    
    # Use the same base environment but with num_agents=1
    quad = 'Crazyflie'
    dyn_randomize_every = dyn_randomization_ratio = None
    raw_control = raw_control_zero_middle = True

    sampler_1 = None
    if dyn_randomization_ratio is not None:
        sampler_1 = dict(type='RelativeSampler', noise_ratio=dyn_randomization_ratio, sampler='normal')

    sense_noise = 'default'
    dynamics_change = dict(noise=dict(thrust_noise_ratio=0.05), damp=dict(vel=0, omega_quadratic=0))

    rew_coeff = DEFAULT_QUAD_REWARD_SHAPING['quad_rewards']
    use_replay_buffer = cfg.replay_buffer_sample_prob > 0.0

    env = QuadrotorEnvMulti(
        num_agents=1, ep_time=cfg.quads_episode_duration, rew_coeff=rew_coeff,
        obs_repr=cfg.quads_obs_repr,
        # Neighbor (disabled for single agent)
        neighbor_visible_num=0, neighbor_obs_type=cfg.quads_neighbor_obs_type,
        collision_hitbox_radius=cfg.quads_collision_hitbox_radius,
        collision_falloff_radius=cfg.quads_collision_falloff_radius,
        # Obstacle
        use_obstacles=cfg.quads_use_obstacles, obst_density=cfg.quads_obst_density, obst_size=cfg.quads_obst_size,
        obst_spawn_area=cfg.quads_obst_spawn_area,
        # Aerodynamics
        use_downwash=False,  # Not relevant for single agent
        # Numba Speed Up
        use_numba=cfg.quads_use_numba,
        # Scenarios
        quads_mode=cfg.quads_mode,
        # Room
        room_dims=cfg.quads_room_dims,
        # Replay Buffer
        use_replay_buffer=use_replay_buffer,
        # Rendering
        quads_view_mode=cfg.quads_view_mode, quads_render=cfg.quads_render,
        # Quadrotor Specific
        dynamics_params=quad, raw_control=raw_control, raw_control_zero_middle=raw_control_zero_middle,
        dynamics_randomize_every=dyn_randomize_every, dynamics_change=dynamics_change, dyn_sampler_1=sampler_1,
        sense_noise=sense_noise, init_random_state=False,
        # Rendering
        render_mode=render_mode,
    )

    # Fix observation and action spaces to be Gymnasium compatible BEFORE other wrappers
    env = _fix_gymnasium_spaces(env)

    # Apply same wrappers as multi-agent version
    if use_replay_buffer:
        env = ExperienceReplayWrapper(env, cfg.replay_buffer_sample_prob, cfg.quads_obst_density, cfg.quads_obst_size,
                                      cfg.quads_domain_random, cfg.quads_obst_density_random, cfg.quads_obst_size_random,
                                      cfg.quads_obst_density_min, cfg.quads_obst_density_max, cfg.quads_obst_size_min, cfg.quads_obst_size_max)

    reward_shaping = copy.deepcopy(DEFAULT_QUAD_REWARD_SHAPING)
    reward_shaping['quad_rewards']['quadcol_bin'] = cfg.quads_collision_reward
    reward_shaping['quad_rewards']['quadcol_bin_smooth_max'] = cfg.quads_collision_smooth_max_penalty
    reward_shaping['quad_rewards']['quadcol_bin_obst'] = cfg.quads_obst_collision_reward

    if cfg.anneal_collision_steps > 0:
        reward_shaping['quad_rewards']['quadcol_bin'] = 0.0
        reward_shaping['quad_rewards']['quadcol_bin_smooth_max'] = 0.0
        reward_shaping['quad_rewards']['quadcol_bin_obst'] = 0.0
        annealing = [
            AnnealSchedule('quadcol_bin', cfg.quads_collision_reward, cfg.anneal_collision_steps),
            AnnealSchedule('quadcol_bin_smooth_max', cfg.quads_collision_smooth_max_penalty, cfg.anneal_collision_steps),
            AnnealSchedule('quadcol_bin_obst', cfg.quads_obst_collision_reward, cfg.anneal_collision_steps),
        ]
    else:
        annealing = None

    env = QuadsRewardShapingWrapper(env, reward_shaping_scheme=reward_shaping, annealing=annealing, with_pbt=cfg.with_pbt)

    # Apply compatibility wrapper at the END to handle step/reset API
    env = QuadEnvCompatibility(env)

    if cfg.visualize_v_value:
        actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
        actor_critic.eval()
        device = torch.device("cpu" if cfg.device == "cpu" else "cuda")
        actor_critic.model_to_device(device)
        policy_id = cfg.policy_index
        name_prefix = dict(latest="checkpoint", best="best")[cfg.load_checkpoint_kind]
        checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(cfg, policy_id), f"{name_prefix}_*")
        checkpoint_dict = Learner.load_checkpoint(checkpoints, device)
        actor_critic.load_state_dict(checkpoint_dict["model"])
        env = V_ValueMapWrapper(env, actor_critic)

    return env


def make_quadrotor_env(full_env_name, cfg=None, env_config=None, render_mode=None):
    # Fix env_config handling
    if env_config is None:
        env_config = {}
    
    # Create the base environment
    if cfg.quads_num_agents == 1:
        env = make_quadrotor_env_single(cfg, render_mode, **env_config)
    else:
        env = make_quadrotor_env_multi(cfg, render_mode, **env_config)

    # Apply HMT-DRL wrappers based on configuration (Section 5.5)
    hmt_approach = getattr(cfg, 'hmt_approach', 'none')

    if hmt_approach == 'shield':
        from swarm_rl.hmt_drl.shield.shield_wrapper import ShieldWrapper
        from swarm_rl.irs_security_evaluation import AttackConfig

        enable_irs = getattr(cfg, 'shield_enable_irs', False)
        attack_config = AttackConfig() if enable_irs else None

        env = ShieldWrapper(env, enable_irs=enable_irs, attack_config=attack_config)
        print(f"Applied SHIELD wrapper (IRS={'ON' if enable_irs else 'OFF'})")

    elif hmt_approach == 'coach':
        from swarm_rl.hmt_drl.baseline_wrappers import COACHWrapper
        correction_prob = getattr(cfg, 'coach_correction_prob', 0.1)
        imitation_weight = getattr(cfg, 'coach_imitation_weight', 1.0)
        env = COACHWrapper(env, correction_prob=correction_prob, imitation_weight=imitation_weight)
        print(f"Applied COACH wrapper (correction_prob={correction_prob})")

    elif hmt_approach == 'tamer':
        from swarm_rl.hmt_drl.baseline_wrappers import TAMERWrapper
        feedback_freq = getattr(cfg, 'tamer_feedback_freq', 10)
        model_update_freq = getattr(cfg, 'tamer_model_update_freq', 100)
        env = TAMERWrapper(env, feedback_freq=feedback_freq, model_update_freq=model_update_freq)
        print(f"Applied TAMER wrapper (feedback_freq={feedback_freq})")

    elif hmt_approach == 'saho':
        from swarm_rl.hmt_drl.baseline_wrappers import SAHOWrapper
        blend_weight = getattr(cfg, 'hmt_weight', 0.3)
        num_goals = getattr(cfg, 'saho_num_goals', 10)
        lookahead = getattr(cfg, 'saho_lookahead', 5)
        env = SAHOWrapper(env, blend_weight=blend_weight, num_goals=num_goals, lookahead=lookahead)
        print(f"Applied SAHO wrapper (blend_weight={blend_weight})")
    
    return env


def _fix_gymnasium_spaces(env):
    """Fix observation and action spaces to be proper Gymnasium spaces without breaking existing structure."""
    import gymnasium as gym
    import numpy as np
    
    # Fix observation space
    if hasattr(env, 'observation_space') and env.observation_space is not None:
        obs_space = env.observation_space
        if hasattr(obs_space, 'low') and hasattr(obs_space, 'high'):
            # Convert to proper Gymnasium Box space with correct dtype
            low = np.array(obs_space.low, dtype=np.float32)
            high = np.array(obs_space.high, dtype=np.float32)
            shape = low.shape
            env.observation_space = gym.spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)
    
    # Fix action space
    if hasattr(env, 'action_space') and env.action_space is not None:
        act_space = env.action_space
        if hasattr(act_space, 'low') and hasattr(act_space, 'high'):
            # Convert to proper Gymnasium Box space with correct dtype
            low = np.array(act_space.low, dtype=np.float32)
            high = np.array(act_space.high, dtype=np.float32)
            shape = low.shape
            env.action_space = gym.spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)
    
    return env
