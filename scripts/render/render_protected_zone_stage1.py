#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""High-quality renderer for Stage-I (ground-truth observations).
Generates publication-ready PNG figures and optional GIF animations.
"""

import argparse
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio.v2 as imageio
import matplotlib

# Headless backend for batch rendering
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
import numpy as np
import torch
from gymnasium import spaces

# Project imports --------------------------------------------------------------
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import get_config
from envs.mpe.MPE_env import MPEEnv
from algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

COLORS = {
    "defender": "#90EE90",
    "defender_traj": "#4CAF50",
    "intruder": "#1E88E5",
    "intruder_traj": "#64B5F6",
    "zone": "#D32F2F",
    "world": "#666666",
    "ellipse": "#555555",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stage-I renderer")
    parser.add_argument("--algo", choices=["rmappo", "mappo", "ippo"], default="rmappo",
                        help="Algorithm label used during training (determines default model path).")
    parser.add_argument("--defenders", type=int, default=5,
                        help="Number of defender UAVs (e.g., 5 for 5v2 baseline).")
    parser.add_argument("--intruders", type=int, default=2,
                        help="Number of intruder UAVs (e.g., 2 for 5v2 baseline).")
    parser.add_argument("--run_id", type=str, default="",
                        help="Optional run identifier matching training scripts (e.g., seed label).")
    parser.add_argument("--model_dir", type=str, default="",
                        help="Optional explicit model directory (overrides automatic resolution).")
    parser.add_argument("--results_root", type=str, default="results/MPE",
                        help="Root directory containing training results (default matches train scripts).")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of evaluation episodes to render.")
    parser.add_argument("--episode_length", type=int, default=300,
                        help="Maximum episode length when rendering.")
    parser.add_argument("--output_dir", type=str, default="render_output/stage1",
                        help="Directory where PNG/GIF files will be saved.")
    parser.add_argument("--gif", action="store_true",
                        help="If set, also export animated GIF alongside static PNG trajectories.")
    parser.add_argument("--gif_fps", type=int, default=12, help="Frames per second for GIF export.")
    parser.add_argument("--pdf", action="store_true",
                        help="Also export vector PDF for publication-quality figures.")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI for PNG export.")
    parser.add_argument("--linewidth", type=float, default=2.0, help="Line width for trajectories.")
    parser.add_argument("--marker_size", type=float, default=6.0, help="Marker size for start/end points.")
    parser.add_argument("--legend_fontsize", type=float, default=10.0,
                        help="Legend font size for rendered figures.")
    parser.add_argument("--label_fontsize", type=float, default=10.0,
                        help="Axis label font size for rendered figures.")
    return parser.parse_args()


def resolve_model_dir(stage: int, args: argparse.Namespace) -> Path:
    if args.model_dir:
        model_path = Path(args.model_dir)
        if model_path.is_dir():
            if (model_path / "models").exists():
                return model_path / "models"
            return model_path
        raise FileNotFoundError(f"Provided model_dir does not exist: {model_path}")

    stage_name = f"protected_zone_stage{stage}"
    exp_name = f"{args.algo}_{args.defenders}v{args.intruders}_stage{stage}"
    if args.run_id:
        exp_name += f"_{args.run_id}"
    base = Path(args.results_root) / stage_name / args.algo
    candidate = base / exp_name / "models"
    if candidate.exists():
        return candidate
    candidate_alt = base / f"{exp_name}_" / "models"
    if candidate_alt.exists():
        return candidate_alt
    raise FileNotFoundError(
        "Couldn't locate trained models at "
        f"{candidate} (also checked {candidate_alt}). Use --model_dir to specify explicitly."
    )


def configure_env(args: argparse.Namespace) -> Tuple[object, object]:
    parser = get_config()
    cfg = parser.parse_args([])
    cfg.env_name = "MPE"
    cfg.scenario_name = "protected_zone_stage1"
    cfg.algorithm_name = args.algo
    cfg.num_defenders = args.defenders
    cfg.num_intruders = args.intruders
    cfg.num_agents = args.defenders + args.intruders
    cfg.episode_length = args.episode_length

    # Geometry (match training defaults)
    cfg.protected_cx = 0.0
    cfg.protected_cy = 0.0
    cfg.protected_r = 0.5
    cfg.world_r = 5.0
    cfg.capture_r = 0.2

    # Motion + reward coefficients (Stage-I defaults)
    cfg.defender_max_speed = 1.0
    cfg.intruder_max_speed = 1.0
    cfg.capture_reward = 15.0
    cfg.entry_penalty = -15.0
    cfg.time_penalty = -0.01
    cfg.vel_penalty = -0.001
    cfg.distance_reward = 0.03
    cfg.formation_reward = 0.08
    cfg.threat_weight_reward = 0.02
    cfg.intercept_reward = 0.05

    # Algorithm-specific toggles (mirror train script)
    if args.algo == "rmappo":
        cfg.use_recurrent_policy = True
        cfg.use_naive_recurrent_policy = False
    elif args.algo == "mappo":
        cfg.use_recurrent_policy = False
        cfg.use_naive_recurrent_policy = False
    elif args.algo == "ippo":
        cfg.use_centralized_V = False
        cfg.use_recurrent_policy = False
        cfg.use_naive_recurrent_policy = False
    else:
        raise ValueError(f"Unsupported algorithm: {args.algo}")

    cfg.hidden_size = 64
    cfg.layer_N = 1
    cfg.recurrent_N = 1
    cfg.use_feature_normalization = True
    cfg.use_orthogonal = True
    cfg.use_ReLU = True
    cfg.use_popart = False
    cfg.use_valuenorm = True
    cfg.use_policy_active_masks = True
    cfg.lr = 5e-4
    cfg.critic_lr = 5e-4
    cfg.opti_eps = 1e-5
    cfg.weight_decay = 0.0
    cfg.gain = 0.01
    cfg.n_rollout_threads = 1
    cfg.use_render = False

    env = MPEEnv(cfg)
    return env, cfg


def build_policies(env, cfg, args, model_dir: Path) -> Tuple[R_MAPPOPolicy, R_MAPPOPolicy]:
    box_action_space = env.action_space[0]
    assign_space = spaces.Discrete(args.intruders + 1)
    hybrid_space = spaces.Tuple([box_action_space, assign_space])

    defender_policy = R_MAPPOPolicy(cfg, env.observation_space[0], env.share_observation_space[0], hybrid_space)
    intruder_policy = R_MAPPOPolicy(cfg, env.observation_space[-1], env.share_observation_space[-1], env.action_space[-1])

    def_paths = [model_dir / "actor.pt", model_dir / "defender_actor.pt"]
    def_critic_paths = [model_dir / "critic.pt", model_dir / "defender_critic.pt"]
    int_paths = [model_dir / "intruder_actor.pt"]
    int_critic_paths = [model_dir / "intruder_critic.pt"]

    def load_weights(policy, actor_paths, critic_paths):
        actor_loaded = False
        for path in actor_paths:
            if path.exists():
                sd = torch.load(path, map_location="cpu")
                current = policy.actor.state_dict()
                filtered = {k: v for k, v in sd.items() if k in current and v.shape == current[k].shape}
                current.update(filtered)
                policy.actor.load_state_dict(current)
                actor_loaded = True
                break
        critic_loaded = False
        for path in critic_paths:
            if path.exists():
                sd = torch.load(path, map_location="cpu")
                current = policy.critic.state_dict()
                filtered = {k: v for k, v in sd.items() if k in current and v.shape == current[k].shape}
                current.update(filtered)
                policy.critic.load_state_dict(current)
                critic_loaded = True
                break
        if not actor_loaded or not critic_loaded:
            raise FileNotFoundError(f"Failed to load actor/critic weights from {model_dir}")
        policy.actor.eval()
        policy.critic.eval()

    load_weights(defender_policy, def_paths, def_critic_paths)
    load_weights(intruder_policy, int_paths, int_critic_paths)
    return defender_policy, intruder_policy


def split_observations(obs, defenders):
    if isinstance(obs, list):
        obs_array = [np.asarray(o, dtype=np.float32) for o in obs]
        obs_stack = np.stack(obs_array, axis=0)
    else:
        obs_stack = np.asarray(obs, dtype=np.float32)
    return obs_stack[:defenders], obs_stack[defenders:]


def collect_episode(env, def_policy, int_policy, args, cfg):
    obs = env.reset()
    defenders = args.defenders
    intruders = args.intruders

    hidden_size = cfg.hidden_size
    recurrent_N = cfg.recurrent_N if getattr(cfg, "use_recurrent_policy", False) else 1

    def_rnn = np.zeros((defenders, recurrent_N, hidden_size), dtype=np.float32)
    int_rnn = np.zeros((intruders, recurrent_N, hidden_size), dtype=np.float32)
    def_masks = np.ones((defenders, 1), dtype=np.float32)
    int_masks = np.ones((intruders, 1), dtype=np.float32)

    trajectories_def = [[] for _ in range(defenders)]
    trajectories_int = [[] for _ in range(intruders)]
    frames: List[Dict[str, np.ndarray]] = []

    step = 0
    episode_done = False
    episode_info: Dict[str, float] = {}

    while not episode_done and step < args.episode_length:
        world = env.world
        def_positions = []
        int_positions = []
        for agent in world.agents:
            pos = agent.state.p_pos.copy()
            if agent.adversary:
                int_positions.append(pos)
            else:
                def_positions.append(pos)
        def_positions = np.asarray(def_positions)
        int_positions = np.asarray(int_positions)

        for idx in range(defenders):
            trajectories_def[idx].append(def_positions[idx].copy())
        for idx in range(intruders):
            trajectories_int[idx].append(int_positions[idx].copy())
        frames.append({"def": def_positions, "intr": int_positions})

        def_obs, int_obs = split_observations(obs, defenders)

        def_actions, def_rnn = def_policy.act(def_obs, def_rnn, def_masks, deterministic=True)
        int_actions, int_rnn = int_policy.act(int_obs, int_rnn, int_masks, deterministic=True)

        def_actions = np.asarray(def_actions.detach().cpu().numpy())
        int_actions = np.asarray(int_actions.detach().cpu().numpy())
        def_cmd = def_actions[:, :2]
        int_cmd = int_actions[:, :2] if int_actions.ndim == 2 else np.stack(int_actions)
        env_actions = np.concatenate([def_cmd, int_cmd], axis=0)

        obs, rewards, dones, infos = env.step(env_actions)
        step += 1
        if isinstance(dones, (list, tuple)):
            episode_done = all(bool(x) for x in dones)
        else:
            episode_done = bool(dones)

        if infos and isinstance(infos[0], dict):
            episode_info = infos[0]

        mask_val = 0.0 if episode_done else 1.0
        def_masks.fill(mask_val)
        int_masks.fill(mask_val)

    statistics = {
        "length": step,
        "defense_success": float(episode_info.get("defense_success", 0.0)),
        "attack_success": float(episode_info.get("attack_success", 0.0)),
        "captures": float(episode_info.get("total_captured", episode_info.get("captures", 0.0))),
    }

    trajectories_def = [np.stack(traj, axis=0) if len(traj) > 0 else np.empty((0, 2)) for traj in trajectories_def]
    trajectories_int = [np.stack(traj, axis=0) if len(traj) > 0 else np.empty((0, 2)) for traj in trajectories_int]

    return {
        "frames": frames,
        "traj_def": trajectories_def,
        "traj_int": trajectories_int,
        "stats": statistics,
    }


def setup_axes(ax, cfg, args):
    world_r = cfg.world_r
    zone_r = cfg.protected_r
    ax.set_aspect("equal")
    margin = 0.2
    ax.set_xlim(-world_r - margin, world_r + margin)
    ax.set_ylim(-world_r - margin, world_r + margin)
    label_size = getattr(args, "label_fontsize", 15)
    tick_size = max(label_size - 3, 1)
    ax.set_xlabel("x [m]", fontsize=label_size)
    ax.set_ylabel("y [m]", fontsize=label_size)
    ax.tick_params(axis='both', labelsize=tick_size)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_facecolor("#f7f7f7")

    world_circle = plt.Circle((0, 0), world_r, edgecolor=COLORS["world"], facecolor="none", linestyle="--", linewidth=1.0, alpha=0.7, label="World boundary")
    zone_circle = plt.Circle((cfg.protected_cx, cfg.protected_cy), zone_r, edgecolor=COLORS["zone"], facecolor="none", linewidth=2.0, label="Protected zone")
    ax.add_patch(world_circle)
    ax.add_patch(zone_circle)


def draw_static_episode(ax, data, cfg, args, episode_idx):
    setup_axes(ax, cfg, args)

    handles = []
    labels = []

    handles.append(patches.Patch(edgecolor=COLORS["zone"], facecolor="none", linewidth=2.0,
                                 label="Protected zone"))
    labels.append("Protected zone")

    def_line = None
    intr_line = None

    for idx, traj in enumerate(data["traj_def"]):
        if traj.size == 0:
            continue
        line, = ax.plot(traj[:, 0], traj[:, 1], color=COLORS["defender_traj"], linewidth=args.linewidth,
                        linestyle="--", alpha=0.9)
        if def_line is None:
            def_line = line
        ax.scatter(traj[0, 0], traj[0, 1], color=COLORS["defender"], s=args.marker_size**2, marker="o")
        ax.scatter(traj[-1, 0], traj[-1, 1], color=COLORS["defender"], s=args.marker_size**2, marker="s")

    for idx, traj in enumerate(data["traj_int"]):
        if traj.size == 0:
            continue
        line, = ax.plot(traj[:, 0], traj[:, 1], color=COLORS["intruder_traj"], linewidth=args.linewidth,
                        linestyle="--", alpha=0.9)
        if intr_line is None:
            intr_line = line
        ax.scatter(traj[0, 0], traj[0, 1], color=COLORS["intruder"], s=args.marker_size**2, marker="o")
        ax.scatter(traj[-1, 0], traj[-1, 1], color=COLORS["intruder"], s=args.marker_size**2, marker="^")

    if def_line is not None:
        def_line.set_label("Defender trajectory")
        handles.append(def_line)
        labels.append("Defender trajectory")
    if intr_line is not None:
        intr_line.set_label("Intruder trajectory")
        handles.append(intr_line)
        labels.append("Intruder trajectory")

    marker_size = max(args.marker_size / 1.5, 4.0)
    def_start_marker = Line2D([], [], marker='o', linestyle='None', markerfacecolor=COLORS["defender"],
                              markeredgecolor=COLORS["defender"], markersize=marker_size)
    def_end_marker = Line2D([], [], marker='s', linestyle='None', markerfacecolor=COLORS["defender"],
                            markeredgecolor=COLORS["defender"], markersize=marker_size)
    intr_start_marker = Line2D([], [], marker='o', linestyle='None', markerfacecolor=COLORS["intruder"],
                               markeredgecolor=COLORS["intruder"], markersize=marker_size)
    intr_end_marker = Line2D([], [], marker='^', linestyle='None', markerfacecolor=COLORS["intruder"],
                             markeredgecolor=COLORS["intruder"], markersize=marker_size)

    handles.append((def_start_marker, def_end_marker))
    labels.append("Defender start/end")
    handles.append((intr_start_marker, intr_end_marker))
    labels.append("Intruder start/end")

    legend_size = getattr(args, "legend_fontsize", 13.0)
    ax.legend(handles=handles, labels=labels, loc="upper right", fontsize=legend_size, frameon=True,
              handler_map={tuple: HandlerTuple(ndivide=None)})


def render_gif(frames: List[Dict[str, np.ndarray]], cfg, args, save_path: Path):
    world_r = cfg.world_r
    zone_r = cfg.protected_r

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    images = []
    for step, frame in enumerate(frames):
        ax.clear()
        setup_axes(ax, cfg, args)

        def_pos = frame["def"]
        int_pos = frame["intr"]
        if def_pos.size:
            ax.scatter(def_pos[:, 0], def_pos[:, 1], color=COLORS["defender"], s=30)
        if int_pos.size:
            ax.scatter(int_pos[:, 0], int_pos[:, 1], color=COLORS["intruder"], s=30)

        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        image = image.reshape(height, width, 4)[..., :3]
        images.append(image.copy())
    plt.close(fig)
    imageio.mimsave(save_path, images, fps=args.gif_fps)


def main():
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    model_dir = resolve_model_dir(stage=1, args=args)
    print(f"Using model directory: {model_dir}")

    env, cfg = configure_env(args)
    def_policy, int_policy = build_policies(env, cfg, args, model_dir)

    for ep in range(args.episodes):
        episode_data = collect_episode(env, def_policy, int_policy, args, cfg)
        png_path = output_root / f"{args.algo}_{args.defenders}v{args.intruders}_stage1_ep{ep + 1:02d}.png"
        fig, ax = plt.subplots(figsize=(6, 6), dpi=args.dpi)
        draw_static_episode(ax, episode_data, cfg, args, ep)
        fig.tight_layout()
        fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
        if args.pdf:
            pdf_path = png_path.with_suffix('.pdf')
            fig.savefig(pdf_path, bbox_inches="tight")
            print(f"Saved vector figure: {pdf_path}")
        plt.close(fig)
        print(f"Saved static figure: {png_path}")

        if args.gif:
            gif_path = output_root / f"{args.algo}_{args.defenders}v{args.intruders}_stage1_ep{ep + 1:02d}.gif"
            render_gif(episode_data["frames"], cfg, args, gif_path)
            print(f"Saved animation: {gif_path}")

    env.close()


if __name__ == "__main__":
    main()
