#!/usr/bin/env bash
set -euo pipefail

# ----------------------------
# Stage-I rendering helper
# ----------------------------

ALGO=${ALGO:-rmappo}            # rmappo / mappo / ippo
DEFENDERS=${DEFENDERS:-7}       # number of defenders (e.g., 5 for 5v2)
INTRUDERS=${INTRUDERS:-3}       # number of intruders (e.g., 2 for 5v2)
RUN_ID=${RUN_ID:-}              # optional run identifier used during training
EPISODES=${EPISODES:-5}         # number of episodes to render
OUTPUT_DIR=${OUTPUT_DIR:-render_output/stage1}
GIF=${GIF:-true}                # set to false to skip GIF export
PDF=${PDF:-true}                # set to false to skip vector PDF export
GIF_FPS=${GIF_FPS:-12}
DPI=${DPI:-300}
LINEWIDTH=${LINEWIDTH:-2.0}
MARKER_SIZE=${MARKER_SIZE:-6.0}
RESULTS_ROOT=${RESULTS_ROOT:-results/MPE}
MODEL_DIR=${MODEL_DIR:-}

ARGS=(
  --algo ${ALGO}
  --defenders ${DEFENDERS}
  --intruders ${INTRUDERS}
  --episodes ${EPISODES}
  --output_dir "${OUTPUT_DIR}"
  --gif_fps ${GIF_FPS}
  --dpi ${DPI}
  --linewidth ${LINEWIDTH}
  --marker_size ${MARKER_SIZE}
  --results_root "${RESULTS_ROOT}"
)

if [ -n "${RUN_ID}" ]; then
  ARGS+=(--run_id "${RUN_ID}")
fi

if [ -n "${MODEL_DIR}" ]; then
  ARGS+=(--model_dir "${MODEL_DIR}")
fi

if [ "${GIF}" = "true" ]; then
  ARGS+=(--gif)
fi

if [ "${PDF}" = "true" ]; then
  ARGS+=(--pdf)
fi

python scripts/render/render_protected_zone_stage1.py "${ARGS[@]}"
