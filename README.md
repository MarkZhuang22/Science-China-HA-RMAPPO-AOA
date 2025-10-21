# Hybrid Reinforcement Learning Control for UAV Territory Guarding

Code release for the Science China Technological Sciences article  
**“Hybrid Reinforcement Learning Control for UAV Territory Guarding with Cooperative Angle-of-Arrival Localization.”**

The repository contains all components required to reproduce the results in the paper:
- hybrid-action HA‑RMAPPO policies, baselines, and utilities under `algorithms/`, `utils/`, and `runner/`;
- MPE environment extensions with CPF sensing under `envs/`;
- training and rendering pipelines under `scripts/`;
- plotting utilities for algorithm comparison under `scripts/plot_algorithm_comparison.py`;
- example outputs (plots, GIFs) under `plots/` and `render_output/`.

## Repository Map

| Path | Description |
|------|-------------|
| `algorithms/` | PPO, MAPPO, HA-RMAPPO implementations (actor–critic networks, distribution layers, shared utilities). |
| `runner/` | Stage‑1 and Stage‑2 training loops (bi-family hybrid runners with CPF/shaping). |
| `envs/` | Modified MPE scenarios (`protected_zone_stage1`, `protected_zone_stage2`) and wrappers. |
| `utils/` | Shared buffers, device helpers, value normalisation. |
| `scripts/train/*.py` | Stage‑specific Python entry points for HA-RMAPPO training. |
| `scripts/render/*.py` | High-quality PNG/PDF/GIF renderers for Stage‑1/Stage‑2 evaluation. |
| `scripts/*.sh` | Convenience shell scripts for training or rendering with environment variables. |
| `scripts/plot_algorithm_comparison.py` | Generates comparative figures from TensorBoard summaries. |
| `render_output/`, `plots/` | Sample outputs and generated figures. |

## Quick Start

### 1. Environment Setup
```bash
conda create -n ha-rmappo python=3.10
conda activate ha-rmappo
pip install -r requirements.txt
```
(Ensure PyTorch with CUDA/MPS support is installed according to the target platform.)

### 2. Stage‑1 Training (truth-based curriculum start)
Run either the shell wrapper or the Python entry point:

```bash
# shell helper (defaults defined inside script)
bash scripts/train_stage1_bifamily.sh

# or directly
python scripts/train/train_protected_zone_stage1_bifamily.py \
    --scenario_name protected_zone_stage1 \
    --algorithm_name rmappo \
    --num_defenders 5 --num_intruders 2 \
    --experiment_name s1_rmappo
```

### 3. Stage‑2 Training (CPF + threat shaping)
Stage‑2 automatically loads the latest Stage‑1 checkpoint unless `--model_dir` is specified.

```bash
bash scripts/train_stage2_bifamily.sh

# or directly
python scripts/train/train_protected_zone_stage2_bifamily.py \
    --scenario_name protected_zone_stage2 \
    --algorithm_name rmappo \
    --num_defenders 5 --num_intruders 2 \
    --experiment_name s2_rmappo
```

Results (models, logs) are saved under `results/MPE/<scenario>/<algorithm>/<experiment_name>/`.

### 4. Rendering Episodes
Generate publication-quality PNG/PDF/GIF trajectories for either stage.

```bash
# Stage I renderer with optional GIF/PDF outputs
bash scripts/render_stage1.sh

# Stage II renderer (CPF ellipses, threat assignments)
bash scripts/render_stage2.sh

# example direct call
python scripts/render/render_protected_zone_stage2.py \
    --algo rmappo --defenders 5 --intruders 2 \
    --run_id s2_rmappo --gif --pdf
```

Outputs are stored under `render_output/stage1/` or `render_output/stage2/`.

### 5. Plotting Algorithm Comparisons
Assuming TensorBoard summaries (`summary.json` or event files) exist for each method:

```bash
python scripts/plot_algorithm_comparison.py \
    --algorithms rmappo mappo ippo \
    --stage 2 --defenders 5 --intruders 2 \
    --results_root results/MPE \
    --output plots/stage2_5v2_comparison \
    --smooth 50
```
This script exports both PNG and PDF versions for each metric.

### 6. TensorBoard Monitoring
```bash
tensorboard --logdir results/MPE --port 6006
```
Navigate to `http://localhost:6006/` to inspect training curves in real time.

## Reproducing Paper Figures
1. Train Stage‑1 and Stage‑2 models for HA‑RMAPPO, MAPPO, and IPPO.
2. Use `scripts/plot_algorithm_comparison.py` to generate comparison plots (defense success rate, attack success rate, rewards, mean log-det).
3. Render representative trajectories via `render_stage1.sh` and `render_stage2.sh`.

## Citation
If this code is useful, please cite:

```
@article{Zhuang2025HybridAoA,
  title   = {Hybrid Reinforcement Learning Control for UAV Territory Guarding with Cooperative Angle-of-Arrival Localization},
  author  = {Wenjie Zhuang and Longhao Qian and Hugh H.-T. Liu},
  journal = {Science China Technological Sciences},
  year    = {2025}
}
```

## Support
Questions or pull requests are welcome. The repository provides everything needed to reproduce the experiments, figures, and visualisations reported in the paper.
