#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage-1 training (Bi-Family, hybrid defender actions)
====================================================

Defender policy: Tuple(Box(2), Discrete(N+1)) for (velocity, assignment).
Intruder policy: Box(2) velocity. Scenario: protected_zone_stage1 (truth-based).

Outputs are staged for clean transfer to Stage-2 hybrid runner.
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
    parser.add_argument('--scenario_name', type=str, default='protected_zone_stage1')
    parser.add_argument('--num_agents', type=int, default=7)
    parser.add_argument('--num_defenders', type=int, default=5)
    parser.add_argument('--num_intruders', type=int, default=2)

    # geometry
    parser.add_argument('--protected_cx', type=float, default=0.0)
    parser.add_argument('--protected_cy', type=float, default=0.0)
    parser.add_argument('--protected_r', type=float, default=0.5)
    parser.add_argument('--world_r', type=float, default=5.0)
    parser.add_argument('--capture_r', type=float, default=0.2)

    # speeds (both set to 1.0 per request)
    parser.add_argument('--defender_max_speed', type=float, default=1.0)
    parser.add_argument('--intruder_max_speed', type=float, default=1.0)

    return parser.parse_known_args(args)[0]


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    all_args.use_wandb = False
    all_args.episode_length = 300

    # algorithm
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
        f"Stage1-Bi-{all_args.algorithm_name}-{all_args.env_name}-{all_args.experiment_name}@{all_args.user_name}"
    )

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

    from runner.bifamily.mpe_runner_bifamily import BiFamilyRunnerS1 as Runner
    runner = Runner(config)

    print("=" * 60)
    print("Stage-1 training with bi-family hybrid defender policy")
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

