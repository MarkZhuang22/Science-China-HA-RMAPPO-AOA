#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage-2 training with Bi-Family Runner (learned assignments + velocities)
=========================================================================

Uses protected_zone_stage2 scenario. Trains three policies:
- Defender velocity (Box)
- Defender assignment (Discrete N+1)
- Intruder velocity (Box)

Adds Stage-1/2 dense rewards based on CPF metrics:
  Stage-1: -w_s1 * T_i * logdet(Ppos_i)
  Stage-2: -w_s2 * T_i * delta_{d,i}, delta = max(0, ||mu_i - s_d|| - Rc)/vmax

Usage example:
python scripts/train/train_protected_zone_stage2_bifamily.py \
  --scenario_name protected_zone_stage2 --algorithm_name rmappo \
  --num_defenders 5 --num_intruders 2 --experiment_name s2_bi \
  --cpf_num_particles 256 --cpf_sigma_a 0.5 --bearing_sigma0 0.02 --bearing_r0 0.5 \
  --threat_lambda 0.05 --threat_tau_max 60 --threat_tau_step 1.0 \
  --tau_switch 60 --s1_logdet_weight 1.0 --s2_delta_weight 1.0
"""

import sys
import os
from pathlib import Path
import setproctitle

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import get_config
from envs.mpe.MPE_env import MPEEnv
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv


def make_envs(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "MPE":
                env = MPEEnv(all_args)
            else:
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def parse_args(args, parser):
    # scenario + counts
    parser.add_argument('--scenario_name', type=str, default='protected_zone_stage2')
    parser.add_argument('--num_agents', type=int, default=7)
    parser.add_argument('--num_defenders', type=int, default=5)
    parser.add_argument('--num_intruders', type=int, default=2)

    # geometry
    parser.add_argument('--protected_cx', type=float, default=0.0)
    parser.add_argument('--protected_cy', type=float, default=0.0)
    parser.add_argument('--protected_r', type=float, default=0.5)
    parser.add_argument('--world_r', type=float, default=5.0)
    parser.add_argument('--capture_r', type=float, default=0.2)

    # speeds (ensure defenders not too weak, intruders not too strong)
    parser.add_argument('--defender_max_speed', type=float, default=1.0)
    parser.add_argument('--intruder_max_speed', type=float, default=1.0)

    # AoA + CPF
    parser.add_argument('--bearing_sigma0', type=float, default=0.02)
    parser.add_argument('--bearing_r0', type=float, default=0.5)
    parser.add_argument('--cpf_num_particles', type=int, default=256)
    parser.add_argument('--cpf_sigma_a', type=float, default=0.5)
    parser.add_argument('--cpf_init_pos_std', type=float, default=1.0)
    parser.add_argument('--cpf_init_vel_std', type=float, default=0.5)
    parser.add_argument('--cpf_init_acc_std', type=float, default=0.2)
    parser.add_argument('--cpf_resample_ess_ratio', type=float, default=0.5)

    # threat look-ahead & stage switch
    parser.add_argument('--threat_lambda', type=float, default=0.05)
    parser.add_argument('--threat_tau_max', type=float, default=60.0)
    parser.add_argument('--threat_tau_step', type=float, default=1.0)
    parser.add_argument('--tau_switch', type=float, default=10.0)

    # dense reward weights
    parser.add_argument('--s1_logdet_weight', type=float, default=1.0)
    parser.add_argument('--s2_delta_weight', type=float, default=1.0)

    # stage-1 reward terms (to be applied on top of S2)
    parser.add_argument('--capture_reward', type=float, default=15.0)
    parser.add_argument('--entry_penalty', type=float, default=-15.0)
    parser.add_argument('--time_penalty', type=float, default=-0.01)
    parser.add_argument('--vel_penalty', type=float, default=-0.001)
    parser.add_argument('--distance_reward', type=float, default=0.03)
    parser.add_argument('--formation_reward', type=float, default=0.08)
    parser.add_argument('--threat_weight_reward', type=float, default=0.02)
    parser.add_argument('--intercept_reward', type=float, default=0.05)
    parser.add_argument('--use_stage1_rewards', action='store_true', default=True)

    # intruder sensing
    parser.add_argument('--intruder_sense_radius', type=float, default=2.0)
    parser.add_argument('--intruder_max_neighbors', type=int, default=2)

    return parser.parse_known_args(args)[0]


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    all_args.use_wandb = False

    # algo routing
    if all_args.algorithm_name == "rmappo":
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError

    # device & seeds
    from utils.device_utils import get_device, setup_seeds
    device, device_desc = get_device(all_args)
    print(f"Device: {device_desc}")
    setup_seeds(all_args, device)

    # dirs
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    run_dir = Path(project_root + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    os.makedirs(str(run_dir), exist_ok=True)
    print(f"Results dir: {run_dir}")

    setproctitle.setproctitle(
        f"Stage2-Bi-{all_args.algorithm_name}-{all_args.env_name}-{all_args.experiment_name}@{all_args.user_name}"
    )

    # Auto-load Stage-1 model for defender/intruder policies if available (no hard-coded exp names)
    if getattr(all_args, 'model_dir', None) in (None, ''):
        base_dir = Path(project_root + "/results") / all_args.env_name / "protected_zone_stage1" / all_args.algorithm_name
        if base_dir.exists():
            # find any subdir/*/models with latest mtime
            candidates = []
            for child in base_dir.glob("*/models"):
                if child.is_dir():
                    candidates.append(child)
            if candidates:
                latest = max(candidates, key=lambda p: p.stat().st_mtime)
                all_args.model_dir = str(latest)
                print(f"Auto-load Stage-1 model from: {all_args.model_dir}")
            else:
                print(f"No models/ subdir found under {base_dir}")
        else:
            print(f"Stage-1 base dir not found: {base_dir}")

    envs = make_envs(all_args)
    eval_envs = None
    num_agents = all_args.num_agents
    print(f"Env config: {all_args.num_defenders} defenders + {all_args.num_intruders} intruders = {num_agents} agents")

    config = {
        'all_args': all_args,
        'envs': envs,
        'eval_envs': eval_envs,
        'num_agents': num_agents,
        'device': device,
        'run_dir': run_dir,
    }

    # Hybrid defender runner (Tuple action head)
    from runner.bifamily.mpe_runner_boa_hybrid import BiFamilyRunnerHybrid as Runner
    runner = Runner(config)

    # Auto-tune tau_switch if not explicitly set: ~60% of straight-line time-to-zone
    if not hasattr(all_args, 'tau_switch') or all_args.tau_switch is None:
        t_to_zone = max(all_args.world_r - all_args.protected_r, 0.1) / max(all_args.intruder_max_speed, 1e-6)
        all_args.tau_switch = 0.6 * t_to_zone
        print(f"Auto tau_switch set to {all_args.tau_switch:.2f} s")

    print("=" * 60)
    print("Stage-2 training with bi-family hybrid defender policy")
    print(f"Episodes: {all_args.num_env_steps // all_args.episode_length // all_args.n_rollout_threads}")
    print(f"Episode length: {all_args.episode_length}")
    print(f"Parallel envs: {all_args.n_rollout_threads}")
    print("=" * 60)

    runner.run()

    envs.close()
    try:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()
        print(f"TensorBoard data saved to: {runner.log_dir}")
    except Exception as e:
        print(f"Error saving TensorBoard data: {e}")


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
