#!/usr/bin/env python3
"""Rebuild fixed-budget tau0/tau5 PPO and GNN-PPO reward learning curves."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .config import ExperimentConfig, seed_everything
from .data import load_wef_data, split_audit
from .environment import YellowRiverWEFEnv
from .experiments import POLICY_VARIANTS
from .model import build_model
from .plots import plot_delay_reward_learning_curves
from .ppo import train_policy
from .sector_delay import _saved_config


DELAY_SCENARIOS = ("tau0", "tau5")
CAPTION_HEADING = "## Training reward — Current and five-month delayed observations"
CAPTION_TEXT = f"""{CAPTION_HEADING}

PPO and GNN-PPO are trained with no delayed observations under a fixed budget of 160 updates (four annual training episodes per update). After every update, the evolving policy is evaluated deterministically on the same 2016–2018 validation years using either no delayed observations (`tau0`) or observations delayed by five months (`tau5`). Curves show the unsmoothed ten-seed mean at every recorded update; colour identifies the policy architecture and line style identifies the delay condition. The delayed curves therefore diagnose observation-delay robustness during learning; they do not represent policies trained under delayed observations.
"""


def _progress(label: str, seed: int):
    def report(update: int, total: int, diagnostics: dict[str, float]) -> None:
        interval = max(1, total // 8)
        if update == 1 or update == total or update % interval == 0:
            print(f"[{label} seed={seed}] {update}/{total}", flush=True)

    return report


def _complete_runs(history: pd.DataFrame, updates: int) -> set[tuple[str, int]]:
    if history.empty:
        return set()
    complete: set[tuple[str, int]] = set()
    for (method, seed), group in history.groupby(["method", "seed"]):
        scenarios = set(group["delay_scenario"])
        if (
            scenarios == set(DELAY_SCENARIOS)
            and int(group["update"].max()) == updates
            and group.groupby("delay_scenario")["update"].nunique().eq(updates).all()
        ):
            complete.add((str(method), int(seed)))
    return complete


def run_delay_reward_curves(
    config: ExperimentConfig, smoothing_window: int = 1
) -> pd.DataFrame:
    """Retrain with a fixed budget and diagnose tau0/tau5 return every update."""

    data = load_wef_data(config.data_path)
    split_audit(data, config.train_years, config.validation_years, config.test_years)
    output_dir = config.result_dir / "training_delay_reward_curves"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "delay_reward_learning_curves_raw.csv"
    if raw_path.exists():
        history = pd.read_csv(raw_path)
    else:
        history = pd.DataFrame()
    complete = _complete_runs(history, config.ppo.updates)

    fixed_budget_validation = replace(
        config.validation,
        patience_checks=config.ppo.updates + 1,
    )
    for seed in config.seeds:
        for variant in POLICY_VARIANTS:
            identity = (variant.label, int(seed))
            if identity in complete:
                print(f"[{variant.label} seed={seed}] retained complete curve", flush=True)
                continue
            if not history.empty:
                history = history[
                    ~((history["method"] == variant.label) & (history["seed"] == seed))
                ].copy()
            seed_everything(seed)
            model = build_model(
                variant.model_name,
                config.feature_dim,
                config.ppo.hidden_dim,
                config.ppo.gnn_layers,
            )
            train_env = YellowRiverWEFEnv(
                data, config.reward, config.physical, seed=seed
            )
            validation_env = YellowRiverWEFEnv(
                data, config.reward, config.physical, seed=500_000 + seed
            )
            training_result = train_policy(
                train_env,
                model,
                config.train_years,
                config.ppo,
                seed,
                validation_env,
                config.validation_years,
                fixed_budget_validation,
                config.device,
                _progress(variant.label, seed),
                diagnostic_delay_scenarios=DELAY_SCENARIOS,
                diagnostic_interval_updates=1,
            )
            frame = pd.DataFrame(training_result.history)
            rows: list[pd.DataFrame] = []
            for delay_scenario in DELAY_SCENARIOS:
                column = f"diagnostic_mean_episode_return_{delay_scenario}"
                subset = frame[["update", column]].rename(
                    columns={column: "mean_episode_return"}
                )
                subset["episodes"] = subset["update"] * config.ppo.rollout_episodes
                subset["method"] = variant.label
                subset["variant_key"] = variant.key
                subset["seed"] = seed
                subset["delay_scenario"] = delay_scenario
                subset["delay_months"] = int(delay_scenario.removeprefix("tau"))
                subset["evaluation_years"] = ";".join(
                    str(year) for year in config.validation_years
                )
                subset["evaluation_policy"] = "deterministic"
                rows.append(subset)
            completed = pd.concat(rows, ignore_index=True)
            history = pd.concat([history, completed], ignore_index=True)
            history = history.sort_values(
                ["method", "seed", "delay_months", "update"]
            ).reset_index(drop=True)
            temporary = raw_path.with_suffix(".tmp.csv")
            history.to_csv(temporary, index=False)
            temporary.replace(raw_path)

    plot_delay_reward_learning_curves(
        history,
        config.result_dir
        / "publication_figures"
        / "training_reward_tau0_tau5_ppo_gnnppo.png",
        output_dir / "delay_reward_learning_curves_summary.csv",
        smoothing_window=smoothing_window,
    )
    caption_path = config.result_dir / "publication_figures" / "FIGURE_CAPTIONS.md"
    existing_caption = (
        caption_path.read_text(encoding="utf-8") if caption_path.exists() else "# Figure captions\n"
    )
    if CAPTION_HEADING not in existing_caption:
        caption_path.write_text(
            existing_caption.rstrip() + "\n\n" + CAPTION_TEXT,
            encoding="utf-8",
        )
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=Path("result"))
    parser.add_argument("--smoothing-window", type=int, default=1)
    args = parser.parse_args()
    run_delay_reward_curves(
        _saved_config(args.result_dir.resolve()),
        smoothing_window=args.smoothing_window,
    )


if __name__ == "__main__":
    main()
