#!/bin/bash

# IRS Security Evaluation Script
# Tests SHIELD IRS against various attack scenarios

echo "🛡️ Starting IRS Security Evaluation..."
echo "============================================"
echo "🎯 GOAL: Demonstrate SHIELD IRS superiority against security threats"
echo "📊 Attacks: GPS Spoofing, Jamming, Byzantine, Replay"
echo "📈 Metrics: Detection Rate, False Positive Rate, Recovery Time, Resilience Score"

# Cleanup function
cleanup_irs_dirs() {
    echo "🧹 Cleaning up IRS evaluation directories..."
    rm -rf ./train_dir/irs_*
    mkdir -p ./train_dir
    echo "✅ IRS cleanup completed!"
}

cleanup_irs_dirs

# Timestamp for experiments
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Base parameters for all IRS experiments (IDENTICAL for fair comparison)
BASE_PARAMS="--env=quadrotor_multi --train_for_env_steps=300000000 --algo=APPO --use_rnn=False \
--num_workers=4 --num_envs_per_worker=4 --learning_rate=0.0001 --ppo_clip_value=5.0 \
--nonlinearity=tanh --actor_critic_share_weights=False --policy_initialization=xavier_uniform \
--rollout=128 --batch_size=1024 --normalize_input=False --normalize_returns=False \
--quads_use_numba=True --quads_mode=mix --quads_episode_duration=15.0 \
--quads_obs_repr=xyz_vxyz_R_omega --quads_neighbor_hidden_size=256 \
--quads_collision_hitbox_radius=2.0 --quads_collision_reward=5.0 \
--quads_neighbor_encoder_type=attention --quads_neighbor_visible_num=6 \
--quads_use_obstacles=True --quads_use_downwash=True"

# HMT parameters (IDENTICAL for all HMT approaches)
HMT_PARAMS="--hmt_weight=0.5 \
--saho_num_goals=20 \
--saho_lookahead=10 \
--tamer_feedback_freq=10 \
--tamer_model_update_freq=25 \
--coach_correction_prob=0.25 \
--coach_imitation_weight=1.2 \
--shield_tau_vac=0.4 \
--shield_kappa=0.1 \
--shield_beta=0.05 \
--shield_zeta=0.2 --shield_epsilon=0.3 --shield_delta=0.15 --shield_human_pattern=competent \
--shield_enable_irs"

# Function to run IRS security experiment
run_irs_experiment() {
    local name=$1
    local experiment_name=$2
    local hmt_approach=$3
    local max_retries=2
    local retry_count=0
    
    while [ $retry_count -lt $max_retries ]; do
        echo "🚀 Starting $name (Attempt $((retry_count + 1))/$max_retries)..."
        
        rm -rf "./train_dir/$experiment_name" 2>/dev/null || true
        mkdir -p "./train_dir/$experiment_name"
        
        # Set environment variable to enable IRS evaluation mode
        export IRS_EVALUATION_MODE=true
        export IRS_ATTACK_INTENSITY=0.3
        export IRS_LOG_METRICS=true
        
        # Run experiment with valid parameters only
        if [ "$hmt_approach" = "none" ]; then
            python -m swarm_rl.train $BASE_PARAMS \
                --hmt_approach=none \
                --experiment="$experiment_name" \
                --train_dir="./train_dir/$experiment_name" \
                --experiment_summaries_interval=10 &
        else
            python -m swarm_rl.train $BASE_PARAMS $HMT_PARAMS \
                --hmt_approach="$hmt_approach" \
                --experiment="$experiment_name" \
                --train_dir="./train_dir/$experiment_name" \
                --experiment_summaries_interval=10 &
        fi
        
        local pid=$!
        
        # Wait for completion
        if wait $pid; then
            echo "✅ $name completed successfully!"
            break
        else
            retry_count=$((retry_count + 1))
            echo "⚠️ $name failed (attempt $retry_count/$max_retries). Retrying..."
            sleep 10
        fi
    done
    
    if [ $retry_count -eq $max_retries ]; then
        echo "❌ $name failed after $max_retries attempts"
        return 1
    fi
    
    return 0
}

echo "🛡️ Starting IRS Security Evaluation..."
echo "🚨 Simulating GPS spoofing, jamming, Byzantine, and replay attacks"
echo "📊 Testing all HMT approaches under cyber threats"
echo ""

# Run IRS security evaluation experiments
run_irs_experiment "Pure DRL (Baseline)" "irs-baseline_pure_drl" "none"
run_irs_experiment "SAHO + IRS" "irs-hmt_saho" "saho" 
run_irs_experiment "TAMER + IRS" "irs-hmt_tamer" "tamer"
run_irs_experiment "COACH + IRS" "irs-hmt_coach" "coach"
run_irs_experiment "SHIELD + IRS" "irs-hmt_shield" "shield"

echo ""
echo "IRS SECURITY EVALUATION COMPLETED!"
echo "Results saved in: ./train_dir/irs_*"
echo "View with: tensorboard --logdir=./train_dir/"
echo "Plot with: python -m swarm_rl.plot_irs_results"
