#!/usr/bin/env python
"""Plot algorithm comparison curves from TensorBoard logs.

Generates publication-quality PNG/PDF figures comparing multiple algorithms
across selected metrics (e.g., defense success rate, rewards, logdet).
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    from tensorboard.backend.event_processing import event_accumulator
    HAVE_TENSORBOARD = True
except ModuleNotFoundError:
    event_accumulator = None
    HAVE_TENSORBOARD = False

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

METRIC_LABELS = {
    "defense_success_rate": "Defense success rate",
    "attack_success_rate": "Attack success rate",
    "avg_defender_reward": "Average defender reward",
    "avg_intruder_reward": "Average intruder reward",
    "pz_ep/logdet_mean_avg": r"Mean log-det (belief)",
    "episode_length": "Episode length",
    "total_captured": "Captured intruders",
}

DEFAULT_METRICS = [
    "defense_success_rate",
    "attack_success_rate",
    "avg_defender_reward",
    "avg_intruder_reward",
    "pz_ep/logdet_mean_avg",
]

DEFAULT_PLOT_CONFIG = {
    "algorithms": ["rmappo", "mappo", "ippo"],
    "metrics": DEFAULT_METRICS,
    "stage": 2,
    "defenders": 2,
    "intruders": 1,
    "run_ids": [""],
    "results_root": "results/MPE",
    "output": "plots/stage{stage}_{defenders}v{intruders}_comparison",
    "smooth": 1,
    "dpi": 300,
    "width": 7.0,
    "height": 9.0,
    "legend_loc": "best",
    "title": "Algorithm comparison",
}

COLOR_MAP = {
    "rmappo": "#1b9e77",
    "mappo": "#d95f02",
    "ippo": "#7570b3",
    "random": "#666666",
}

LINE_WIDTH = {
    "rmappo": 2.5,
    "mappo": 1.8,
    "ippo": 1.2,
    "random": 1.0,
}


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size == 0:
        return values
    window = min(window, values.size)
    cumsum = np.cumsum(np.insert(values, 0, 0.0))
    smoothed = (cumsum[window:] - cumsum[:-window]) / window
    pad = np.full(window - 1, values[0])
    return np.concatenate([pad, smoothed])


def load_scalar(log_dir: Path, tag: str) -> Tuple[np.ndarray, np.ndarray]:
    if not log_dir.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    if HAVE_TENSORBOARD:
        event_files = list(log_dir.glob("events.*"))
        if event_files:
            accumulator = event_accumulator.EventAccumulator(
                str(log_dir),
                size_guidance={event_accumulator.SCALARS: 0},
            )
            accumulator.Reload()

            available = accumulator.Tags().get('scalars', [])
            candidate = None
            if tag in available:
                candidate = tag
            elif f"{tag}/{tag}" in available:
                candidate = f"{tag}/{tag}"
            else:
                for entry in available:
                    if entry.endswith(f"/{tag}") or entry.endswith(tag):
                        candidate = entry
                        break
            if candidate is not None:
                scalar_events = accumulator.Scalars(candidate)
                steps = np.array([event.step for event in scalar_events], dtype=np.float32)
                values = np.array([event.value for event in scalar_events], dtype=np.float32)
                return steps, values

    summary_path = log_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Metric '{tag}' not found and no summary.json present in {log_dir}")

    with summary_path.open() as f:
        summary = json.load(f)

    candidate_key = None
    for key in summary:
        if key.endswith(f"/{tag}") or key.endswith(f"/{tag}/{tag}"):
            candidate_key = key
            break
    if candidate_key is None:
        raise KeyError(f"Metric '{tag}' not present in summary.json at {log_dir}")

    entries = summary[candidate_key]
    steps = np.array([entry[1] for entry in entries], dtype=np.float32)
    values = np.array([entry[2] for entry in entries], dtype=np.float32)
    return steps, values


def resolve_log_dir(results_root: Path, stage: int, algo: str,
                    defenders: int, intruders: int, run_id: str) -> Path:
    stage_name = f"protected_zone_stage{stage}"
    exp_base = f"{algo}_{defenders}v{intruders}_stage{stage}"
    exp_name = f"{exp_base}_{run_id}" if run_id else exp_base

    base = results_root / stage_name / algo
    candidates = [base / exp_name / "logs",
                  base / f"{exp_name}_" / "logs"]

    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "Unable to locate logs for "
        f"stage={stage}, algo={algo}, run='{run_id}'. Checked: "
        + ", ".join(str(c) for c in candidates)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot algorithm comparison curves")
    parser.add_argument("--algorithms", nargs="+", default=DEFAULT_PLOT_CONFIG["algorithms"],
                        help="Algorithms to plot (each must have corresponding logs)")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_PLOT_CONFIG["metrics"],
                        help="TensorBoard scalar tags to plot")
    parser.add_argument("--stage", type=int, default=DEFAULT_PLOT_CONFIG["stage"], choices=[1, 2],
                        help="Training stage to visualize (default: 2)")
    parser.add_argument("--defenders", type=int, default=DEFAULT_PLOT_CONFIG["defenders"],
                        help="Number of defenders (e.g., 5 for 5v2)")
    parser.add_argument("--intruders", type=int, default=DEFAULT_PLOT_CONFIG["intruders"],
                        help="Number of intruders (e.g., 2 for 5v2)")
    parser.add_argument("--run_ids", nargs="*", default=DEFAULT_PLOT_CONFIG["run_ids"],
                        help="Optional run identifiers (one per algorithm or single shared)")
    parser.add_argument("--results_root", type=str, default=DEFAULT_PLOT_CONFIG["results_root"],
                        help="Root directory containing training results")
    parser.add_argument("--output", type=str, default=DEFAULT_PLOT_CONFIG["output"],
                        help="Output file prefix (without extension)")
    parser.add_argument("--smooth", type=int, default=DEFAULT_PLOT_CONFIG["smooth"],
                        help="Moving average window for smoothing (0/1 disables smoothing)")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI for PNG export")
    parser.add_argument("--width", type=float, default=8.0, help="Figure width in inches")
    parser.add_argument("--height", type=float, default=4.5, help="Figure height in inches")
    parser.add_argument("--legend_loc", type=str, default=DEFAULT_PLOT_CONFIG["legend_loc"], help="Matplotlib legend location")
    parser.add_argument("--title", type=str, default=DEFAULT_PLOT_CONFIG["title"], help="Figure title")
    if len(sys.argv) > 1:
        return parser.parse_args()
    return parser.parse_args([])


def ensure_run_ids(algos: List[str], run_ids: List[str]) -> List[str]:
    if len(run_ids) == 1:
        return run_ids * len(algos)
    if len(run_ids) != len(algos):
        raise ValueError("Number of run_ids must be 1 or match number of algorithms")
    return run_ids


def pretty_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def main() -> None:
    args = parse_args()
    run_ids = ensure_run_ids(args.algorithms, args.run_ids)

    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    output_prefix = Path(args.output.format(stage=args.stage,
                                            defenders=args.defenders,
                                            intruders=args.intruders))
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": 16,
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.labelsize": 16,
        "axes.titlesize": 12,
        "legend.fontsize": 12,
        "lines.linewidth": 2.0,
    })

    for metric in args.metrics:
        fig, ax = plt.subplots(figsize=(args.width, args.height))
        ax.set_ylabel(pretty_label(metric))
        ax.set_xlabel("Environment steps")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

        plotted = False
        for algo, run_id in zip(args.algorithms, run_ids):
            try:
                log_dir = resolve_log_dir(results_root, args.stage, algo,
                                           args.defenders, args.intruders, run_id)
            except FileNotFoundError as err:
                print(f"[WARN] {err}")
                continue

            try:
                steps, values = load_scalar(log_dir, metric)
            except (FileNotFoundError, KeyError) as err:
                print(f"[WARN] {err}")
                continue

            if args.smooth > 1:
                values = moving_average(values, args.smooth)

            color = COLOR_MAP.get(algo, None)
            linewidth = LINE_WIDTH.get(algo, 2.0)
            pretty_algo = "HA-RMAPPO" if algo == "rmappo" else algo.upper()
            label = pretty_algo
            if run_id:
                label += f" ({run_id})"

            ax.plot(steps, values, label=label, color=color, linewidth=linewidth)
            plotted = True

        if not plotted:
            plt.close(fig)
            print(f"[WARN] No data available for metric '{metric}', skipping figure.")
            continue

        ax.legend(loc=args.legend_loc, frameon=True)
        fig.tight_layout()

        safe_metric = metric.replace("/", "_")
        png_path = output_prefix.with_name(f"{output_prefix.name}_{safe_metric}.png")
        pdf_path = output_prefix.with_name(f"{output_prefix.name}_{safe_metric}.pdf")
        fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved comparison figure: {png_path}")
        print(f"Saved comparison vector figure: {pdf_path}")


if __name__ == "__main__":
    main()
