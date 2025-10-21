#!/usr/bin/env bash
set -euo pipefail

# ----------------------------
# Stage-I training (ground-truth observations)
# ----------------------------

ENV_NAME="MPE"
SCENARIO="protected_zone_stage1"

ALGO=${ALGO:-rmappo}          # algorithm: rmappo (HA-RMAPPO), mappo, ippo
DEFENDERS=${DEFENDERS:-2}     # defender UAV count (use 2 for 2v1 case, use 5 for 5v2 case, set 7 for 7v3 case)
INTRUDERS=${INTRUDERS:-1}     # intruder UAV count (use 1 for 2v1 case, use 2 for 5v2 case, set 3 for 7v3 case)
SEED=${SEED:-42}              # random seed for reproducibility

EP_LEN=${EP_LEN:-300}         # episode length (time steps)
NUM_STEPS=${NUM_STEPS:-10000000}  # total environment steps for training
N_ROLLOUT=${N_ROLLOUT:-128}       # number of rollout environments
N_TRAIN_THREADS=${N_TRAIN_THREADS:-16}  # torch training threads
LR=${LR:-5e-4}                # actor learning rate
CRITIC_LR=${CRITIC_LR:-5e-4}  # critic learning rate
ENTROPY_COEF=${ENTROPY_COEF:-0.005}  # entropy regularization coefficient
MAX_GRAD_NORM=${MAX_GRAD_NORM:-0.5}  # gradient clipping norm

RUN_ID=${1:-}         # optional tag appended to experiment name (e.g., seed, trial id)

NUM_AGENTS=$((DEFENDERS + INTRUDERS))
EXP_BASIS="${ALGO}_${DEFENDERS}v${INTRUDERS}"
EXP_NAME="${EXP_BASIS}_stage1_${RUN_ID}"

python scripts/train/train_protected_zone_stage1_bifamily.py \
  --env_name ${ENV_NAME} \
  --scenario_name ${SCENARIO} \
  --algorithm_name ${ALGO} \
  --experiment_name ${EXP_NAME} \
  --num_agents ${NUM_AGENTS} \
  --num_defenders ${DEFENDERS} \
  --num_intruders ${INTRUDERS} \
  --world_r 5.0 \
  --protected_r 0.5 \
  --capture_r 0.2 \
  --defender_max_speed 1.0 \
  --intruder_max_speed 1.0 \
  --episode_length ${EP_LEN} \
  --n_rollout_threads ${N_ROLLOUT} \
  --n_training_threads ${N_TRAIN_THREADS} \
  --num_env_steps ${NUM_STEPS} \
  --lr ${LR} \
  --critic_lr ${CRITIC_LR} \
  --max_grad_norm ${MAX_GRAD_NORM} \
  --entropy_coef ${ENTROPY_COEF} \
  --seed ${SEED}

STAGE1_DIR="results/${ENV_NAME}/${SCENARIO}/${ALGO}/${EXP_NAME}/models"
echo "Stage-1 training finished. Models stored in: ${STAGE1_DIR}"
