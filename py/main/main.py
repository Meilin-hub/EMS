#!/usr/bin/env python3
"""Command-line entry point for the two Yellow River WEF PPO experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from main.config import DEFAULT_SEEDS, ExperimentConfig
    from main.experiments import run_experiments
else:
    from .config import DEFAULT_SEEDS, ExperimentConfig
    from .experiments import run_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PPO/GNN-PPO, then run Experiments 1–4 on WEF observation delays."
    )
    parser.add_argument("--data", type=Path, default=None, help="Input CSV (default: py/filled.csv)")
    parser.add_argument("--result-dir", type=Path, default=None, help="Artifact root (default: result)")
    parser.add_argument("--updates", type=int, default=160, help="Maximum PPO updates per learned model and seed")
    parser.add_argument("--rollout-episodes", type=int, default=4, help="Training years sampled per PPO update")
    parser.add_argument("--update-epochs", type=int, default=6, help="Optimization epochs per rollout")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
        help="Independent training seeds",
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--minimum-updates", type=int, default=60)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument(
        "--terminal-storage-weight", type=float, default=1.0,
        help="Weight on the normalized December carry-over storage deficit",
    )
    parser.add_argument("--water-weight", type=float, default=1.0)
    parser.add_argument("--energy-weight", type=float, default=0.35)
    parser.add_argument("--food-weight", type=float, default=1.0)
    parser.add_argument(
        "--terminal-storage-target-fraction", type=float, default=1.0,
        help="Hard year-end storage target as a fraction of year-start storage",
    )
    parser.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu, mps, cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig()
    if args.data is not None:
        config.data_path = args.data.resolve()
    if args.result_dir is not None:
        config.result_dir = args.result_dir.resolve()
    config.seeds = tuple(args.seeds)
    config.device = args.device
    config.ppo.updates = args.updates
    config.ppo.rollout_episodes = args.rollout_episodes
    config.ppo.update_epochs = args.update_epochs
    config.ppo.hidden_dim = args.hidden_dim
    config.validation.interval_updates = args.validation_interval
    config.validation.minimum_updates = args.minimum_updates
    config.validation.patience_checks = args.early_stopping_patience
    config.reward.water = args.water_weight
    config.reward.energy = args.energy_weight
    config.reward.food = args.food_weight
    config.reward.terminal_storage = args.terminal_storage_weight
    config.physical.terminal_storage_target_fraction = args.terminal_storage_target_fraction
    result = run_experiments(config)
    print("\nExperiment 1 no-delay effects:")
    print(result["current_information_effects"].query("metric == 'cost'").to_string(index=False))
    print("\nExperiment 2 uniform-delay effects:")
    print(result["uniform_delay_overall_effects"].query("metric == 'cost'").to_string(index=False))
    print(f"\nArtifacts written to: {config.result_dir}")


if __name__ == "__main__":
    main()
