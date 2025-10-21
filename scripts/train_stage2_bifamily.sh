#!/usr/bin/env bash
set -euo pipefail

# ----------------------------
# Stage-II training (CPF belief + threat-aware rewards)
# ----------------------------

ENV_NAME="MPE"
SCENARIO="protected_zone_stage2"
STAGE1_SCENARIO="protected_zone_stage1"

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
ENTROPY_COEF=${ENTROPY_COEF:-0.2}  # entropy regularization coefficient

# CPF / AoA settings
CPF_NUM_PARTICLES=${CPF_NUM_PARTICLES:-512}   # particle count for CPF (increase for higher accuracy)
CPF_SIGMA_A=${CPF_SIGMA_A:-0.3}          # process-noise std for acceleration
BEARING_SIGMA0=${BEARING_SIGMA0:-0.02}   # bearing noise (rad) at reference range
BEARING_R0=${BEARING_R0:-0.5}            # reference range for bearing noise scaling

# Threat switching
THREAT_LAMBDA=${THREAT_LAMBDA:-0.05}     # decay rate for threat weight T = exp(-lambda * tau)
THREAT_TAU_MAX=${THREAT_TAU_MAX:-60.0}   # max look-ahead horizon (seconds)
THREAT_TAU_STEP=${THREAT_TAU_STEP:-1.0}  # scan step for earliest collision time
TAU_SWITCH=${TAU_SWITCH:-3.0}            # switch threshold from uncertainty reduction to interception (seconds)

S1_LOGDET_WEIGHT=${S1_LOGDET_WEIGHT:-1.0}  # dense reward weight for Stage-1-style term
S2_DELTA_WEIGHT=${S2_DELTA_WEIGHT:-1.0}    # dense reward weight for Stage-2 interception term

INTRUDER_SENSE_RADIUS=${INTRUDER_SENSE_RADIUS:-1.0}      # intruder KNN sensing radius
INTRUDER_MAX_NEIGHBORS=${INTRUDER_MAX_NEIGHBORS:-2}      # max neighbors for intruder observation (KNN)

RUN_ID=${1:-}         # optional tag appended to experiment name (must match Stage-1 run id)

NUM_AGENTS=$((DEFENDERS + INTRUDERS))
EXP_BASIS="${ALGO}_${DEFENDERS}v${INTRUDERS}"
EXP_NAME="${EXP_BASIS}_stage2_${RUN_ID}"
STAGE1_EXP_NAME="${EXP_BASIS}_stage1_${RUN_ID}"
STAGE1_DIR_DEFAULT="results/${ENV_NAME}/${STAGE1_SCENARIO}/${ALGO}/${STAGE1_EXP_NAME}/models"
STAGE1_DIR=${STAGE1_DIR:-${STAGE1_DIR_DEFAULT}}

if [ ! -d "${STAGE1_DIR}" ]; then
  echo "[ERROR] Stage-1 model directory not found: ${STAGE1_DIR}" >&2
  echo "        Run scripts/train_stage1_bifamily.sh ${RUN_ID} first with matching parameters." >&2
  exit 1
fi

echo "Using Stage-1 models from: ${STAGE1_DIR}"

python scripts/train/train_protected_zone_stage2_bifamily.py \
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
  --intruder_max_speed 0.5 \
  --cpf_num_particles ${CPF_NUM_PARTICLES} \
  --cpf_sigma_a ${CPF_SIGMA_A} \
  --bearing_sigma0 ${BEARING_SIGMA0} \
  --bearing_r0 ${BEARING_R0} \
  --threat_lambda ${THREAT_LAMBDA} \
  --threat_tau_max ${THREAT_TAU_MAX} \
  --threat_tau_step ${THREAT_TAU_STEP} \
  --tau_switch ${TAU_SWITCH} \
  --s1_logdet_weight ${S1_LOGDET_WEIGHT} \
  --s2_delta_weight ${S2_DELTA_WEIGHT} \
  --intruder_sense_radius ${INTRUDER_SENSE_RADIUS} \
  --intruder_max_neighbors ${INTRUDER_MAX_NEIGHBORS} \
  --model_dir ${STAGE1_DIR} \
  --n_rollout_threads ${N_ROLLOUT} \
  --n_training_threads ${N_TRAIN_THREADS} \
  --lr ${LR} \
  --critic_lr ${CRITIC_LR} \
  --entropy_coef ${ENTROPY_COEF} \
  --seed ${SEED} \
  --episode_length ${EP_LEN} \
  --num_env_steps ${NUM_STEPS}

echo "Stage-2 training finished. Results stored under: results/${ENV_NAME}/${SCENARIO}/${ALGO}/${EXP_NAME}"
