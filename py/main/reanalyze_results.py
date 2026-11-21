#!/usr/bin/env python3
"""Rebuild Experiments 1–4 from retained checkpoints and raw Experiment 1/2 data."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .experiments import rebuild_result_artifacts
from .sector_delay import (
    _saved_config,
    run_ppo_sector_delay_experiment,
    run_sector_delay_experiment,
)


def reanalyze(result_dir: Path) -> None:
    result_dir = result_dir.resolve()
    config = _saved_config(result_dir)
    raw_path = result_dir / "raw" / "experiment_1_2_episode_outcomes.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing retained Experiment 1/2 raw outcomes: {raw_path}")
    episodes = pd.read_csv(raw_path)
    rebuild_result_artifacts(config, episodes)
    run_sector_delay_experiment(config)
    run_ppo_sector_delay_experiment(config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=Path("result"))
    args = parser.parse_args()
    reanalyze(args.result_dir)


if __name__ == "__main__":
    main()
