# SHIELD: Secure Human-Machine Interaction with Evidential Learning and Dynamic Trust for Drone Swarm Control

This repository contains the implementation of SHIELD, a framework for secure human-swarm teaming that uses evidential vacuity (derived from Dirichlet-based perception) to jointly drive inter-agent trust modulation, intrusion detection, and human escalation within a decentralized control loop.

## Overview

Drone swarms face a dual challenge: human operators cannot effectively supervise more than 10-12 UAVs before situational awareness degrades, yet fully autonomous coordination remains vulnerable to cyber-physical attacks. SHIELD addresses both constraints through a unified uncertainty-driven architecture.

### Key Contributions

1. **Unified uncertainty-driven architecture**: A single evidential vacuity signal governs trust adaptation, intrusion detection, and event-triggered human escalation.
2. **Dynamic trust graph with real-time quarantine**: Decentralized, trust-weighted consensus isolates compromised UAVs without centralized supervision.
3. **Comprehensive adversarial evaluation**: Six cyber-physical attack classes (A1-A6) spanning perception, communication, coordination, software integrity, and network layers.

## Architecture

SHIELD executes the following process at each control timestep (50 Hz):

```
1. Evidential Perception    - ENN produces Dirichlet concentrations -> belief + vacuity
2. Local Action Proposal     - RL policy generates action distribution
3. Communication             - Broadcast (s_i, b_i, p_i) to neighbours within comm range
4. Trust Update              - Robust aggregate, consistency check, trust decay
5. Trust-Weighted Fusion     - Weighted averaging of neighbour info
6. Safety Arbitration        - KL-divergence override if D_KL > epsilon or u_i > tau_vac
7. Execution                 - Execute chosen action
```

## SHIELD Parameters

| Parameter | Symbol  | Default | Description |
|-----------|---------|---------|-------------|
| `--shield_tau_vac` | tau_vac | 0.4 | Vacuity escalation threshold |
| `--shield_kappa`   | kappa   | 0.1 | Trust smoothing constant |
| `--shield_beta`    | beta    | 0.05 | Trust decay rate |
| `--shield_zeta`    | zeta    | 0.2 | Quarantine activation threshold |
| `--shield_epsilon` | epsilon | 0.3 | KL-divergence override threshold |
| `--shield_delta`   | delta   | 0.15 | Consensus consistency radius |

## Attack Classes

| ID | Attack | Parameters |
|----|--------|------------|
| A1 | GPS Spoofing | delta_max=3.0 m, 5 s on / 20 s off |
| A2 | Comm. Jamming | drop prob p=0.5, 3-7 s windows |
| A3 | Byzantine Faults | inject prob p=0.10, continuous |
| A4 | Replay Attack | lag 10 s, 5 s on / 25 s off |
| A5 | Malware Injection | policy drift norm <= 0.15, continuous |
| A6 | Network Intrusion | eavesdrop + selective inject, continuous |

## Project Structure

```
SHIELD/
  swarm_rl/
    train.py                              # Entry point with all CLI arguments
    env_wrappers/
      quad_utils.py                       # Environment factory, applies ShieldWrapper
      quadrotor_params.py                 # Quadrotor-specific CLI arguments
      reward_shaping.py                   # Reward shaping (shared by all methods)
      compatibility.py                    # Gym/Gymnasium compatibility
    models/
      quad_multi_model.py                 # Encoder architectures (attention, deepsets, MLP)
    hmt_drl/
      base.py                            # Base class for HMT-DRL approaches
      coach.py                           # COACH baseline (corrective feedback)
      tamer.py                           # TAMER baseline (scalar reward model)
      saho.py                            # SAHO baseline (shared autonomy)
      hmt_trainer.py                     # HMT trainer factory
      shield/
        __init__.py                      # Package exports
        node.py                          # ShieldNode, ENN, trust logic
        shield_wrapper.py                # Gym wrapper executing full control loop
        coverage.py                      # Max-heap coverage reassignment
        human_interface.py               # Human oracle (0.5 s latency, binary)
        message_router.py                # Range-limited comms with A2/A3/A6 injection
    irs_security_evaluation.py           # AttackInjector (A1-A6) + IRSMetricsTracker
    plot_irs_results.py                  # IRS result plotting
  gym_art/
    quadrotor_multi/                     # QuadSwarm simulator (50 Hz, Gaussian noise)
  train_local_hmt_comparison.sh          # Exp 1: Nominal open-field benchmarking
  train_local_hmt_comparison_obstacles.sh# Exp 1: Nominal dense-field (20% obstacles)
  train_irs_security_evaluation.sh       # Exp 2: Intrusion robustness (6 attack classes)
```

## Installation


Recommended environment:

- Ubuntu 22.04
- Python 3.11.15
- CUDA-capable NVIDIA GPU
- `build-essential`

```bash
sudo apt update
sudo apt install -y build-essential python3.11 python3.11-venv python3.11-dev

git clone https://github.com/dawoodwasif/SHIELD.git
cd SHIELD

python3.11 -m venv shield_env
source shield_env/bin/activate
python -m pip install --upgrade pip setuptools wheel

pip install -e .

# Dependencies
pip install sample-factory==2.0.3 gymnasium numpy torch numba scipy
```

## Training

Use this command to verify the SHIELD artifact without running the full training pipeline:

```bash
python -m swarm_rl.train --env=quadrotor_multi --hmt_approach=shield \
    --quads_num_agents=8 --quads_neighbor_visible_num=0 \
    --train_for_env_steps=100000 --experiment=sanity_check
```

### Experiment 1: Nominal Benchmarking 

Open field (no obstacles), 8 UAVs, all 5 methods:

```bash
bash train_local_hmt_comparison.sh
```

Dense field (20% obstacles), 8 UAVs:

```bash
bash train_local_hmt_comparison_obstacles.sh
```

### Experiment 2: Intrusion Robustness 

All 6 attack classes, dense field, SHIELD with IRS enabled:

```bash
bash train_irs_security_evaluation.sh
```

### Individual Runs

```bash
# Baseline DRL (APPO)
python -m swarm_rl.train --env=quadrotor_multi --hmt_approach=none \
    --quads_num_agents=8 --quads_use_obstacles=True --train_for_env_steps=25000000

# SHIELD (nominal, no attacks)
python -m swarm_rl.train --env=quadrotor_multi --hmt_approach=shield \
    --quads_num_agents=8 --quads_use_obstacles=True --train_for_env_steps=25000000

# SHIELD + IRS (with attack injection)
python -m swarm_rl.train --env=quadrotor_multi --hmt_approach=shield \
    --shield_enable_irs --quads_num_agents=8 --quads_use_obstacles=True \
    --train_for_env_steps=25000000

# COACH baseline
python -m swarm_rl.train --env=quadrotor_multi --hmt_approach=coach \
    --quads_num_agents=8 --train_for_env_steps=25000000

# TAMER baseline
python -m swarm_rl.train --env=quadrotor_multi --hmt_approach=tamer \
    --quads_num_agents=8 --train_for_env_steps=25000000

# SAHO baseline
python -m swarm_rl.train --env=quadrotor_multi --hmt_approach=saho \
    --quads_num_agents=8 --train_for_env_steps=25000000
```

## Evaluation Metrics

### Mission Metrics 
- **CP** (Crash Penalty, Eq. 19): Cumulative collision reward; closer to zero is better.
- **MSR** (Mission Success Rate, Eq. 20): Fraction of agents completing all waypoints without crashing.
- **SR** (Survival Rate, Eq. 21): Fraction of crash-free agents at episode end.

### Intrusion Metrics 

- **DR** (Detection Rate, Eq. 22): Fraction of attacks detected within 5 s of onset.
- **FPR** (False Positive Rate, Eq. 23): Fraction of benign timesteps incorrectly flagged.
- **RT** (Recovery Time, Eq. 24): Time to return within 10% of nominal formation.
- **HIC** (Human Intervention Count, Eq. 25): Number of binary operator confirmations per episode.

## Monitoring

```bash
# TensorBoard
tensorboard --logdir=./train_dir/

# IRS results plot
python -m swarm_rl.plot_irs_results
```


## Baselines

All baselines use identical observation spaces, action spaces, training budget (2.5 x 10^7 steps), and simulation backend (Section 5.5):

| Method | Reference | Key Mechanism |
|--------|-----------|---------------|
| APPO (Baseline DRL) | Petrenko et al. 2020 | Pure deep RL, no human input |
| COACH | MacGlashan et al. 2017 | Corrective action demonstrations |
| SAHO | Javdani et al. 2015 | Shared autonomy via latent-goal POMDP |
| TAMER | Warnell et al. 2018 | Scalar human reward model |
| **SHIELD** | This work | Evidential perception + dynamic trust + IRS |

## Training Configuration (Table 15)

| Parameter | Value |
|-----------|-------|
| Learning rate | 3 x 10^-4 |
| Discount gamma | 0.99 |
| GAE lambda | 0.95 |
| Minibatch size | 64 |
| Epochs/update | 8 |
| Entropy coefficient | 0.01 |
| Value-loss coefficient | 0.5 |
| Total steps | 2.5 x 10^7 |

## Simulation Environment

QuadSwarm multi-quadrotor simulator:
- Arena: 20 x 20 x 10 m
- Control frequency: 50 Hz
- Sensor noise: sigma_GPS = 0.3 m, sigma_IMU = 0.01 rad/s
- Dense field: 20% volume with 0.5 m cubic OctoMap occluders
- Episode length: up to 25,000 steps (500 s)

## Ethical Considerations

This work is defensive in nature. All experiments are conducted entirely in simulation using the open-source QuadSwarm environment. No physical deployments, real communication disruptions, or human subjects are involved. The six attack classes are well-documented in UAV security literature with parameterisations derived from published field studies.


## Acknowledgments

This work builds upon the open-source [Quadswarm](https://github.com/Zhehui-Huang/quad-swarm-rl). We gratefully acknowledge:

> Huang, Zhehui, et al. "Quadswarm: A modular multi-quadrotor simulator for deep reinforcement learning with direct thrust control." arXiv preprint arXiv:2306.09537 (2023).
