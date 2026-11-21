#!/usr/bin/env python3
"""Build GNN-PPO reward learning curves under uniform tau0--tau5 delays."""

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
from .plots import plot_gnnppo_uniform_delay_learning_curves
from .ppo import train_policy
from .sector_delay import _saved_config


DELAY_SCENARIOS = tuple(f"tau{delay}" for delay in range(6))
METHOD_LABEL = "GNN-PPO"
CAPTION_HEADING = "## Training reward — GNN-PPO under uniform observation delays"
CAPTION_TEXT = f"""{CAPTION_HEADING}

GNN-PPO is trained with no delayed observations under a fixed budget of 160 updates (four annual training episodes per update). After every update, the evolving policy is evaluated deterministically on the same 2016–2018 validation years at each uniform observation delay from `tau0` to `tau5`. Curves show the unsmoothed ten-seed mean at every recorded update. These are delay-diagnostic evaluations of a policy trained with no delayed observations, rather than six separately delay-trained policies.
"""


def _progress(seed: int):
    def report(update: int, total: int, diagnostics: dict[str, float]) -> None:
        interval = max(1, total // 8)
        if update == 1 or update == total or update % interval == 0:
            print(f"[{METHOD_LABEL} seed={seed}] {update}/{total}", flush=True)

    return report


def _complete_seeds(history: pd.DataFrame, updates: int) -> set[int]:
    if history.empty:
        return set()
    complete: set[int] = set()
    for seed, group in history.groupby("seed"):
        scenarios = set(group["delay_scenario"])
        if (
            scenarios == set(DELAY_SCENARIOS)
            and int(group["update"].max()) == updates
            and group.groupby("delay_scenario")["update"].nunique().eq(updates).all()
        ):
            complete.add(int(seed))
    return complete


def run_gnnppo_delay_reward_curves(config: ExperimentConfig) -> pd.DataFrame:
    """Train GNN-PPO once per seed and diagnose all six uniform delay conditions."""

    data = load_wef_data(config.data_path)
    split_audit(data, config.train_years, config.validation_years, config.test_years)
    output_dir = config.result_dir / "training_gnnppo_delay_reward_curves"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "gnnppo_delay_reward_learning_curves_raw.csv"
    history = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()
    complete_seeds = _complete_seeds(history, config.ppo.updates)
    variant = next(item for item in POLICY_VARIANTS if item.label == METHOD_LABEL)
    fixed_budget_validation = replace(
        config.validation,
        patience_checks=config.ppo.updates + 1,
    )

    for seed in config.seeds:
        if int(seed) in complete_seeds:
            print(f"[{METHOD_LABEL} seed={seed}] retained complete curve", flush=True)
            continue
        if not history.empty:
            history = history[history["seed"] != seed].copy()
        seed_everything(seed)
        model = build_model(
            variant.model_name,
            config.feature_dim,
            config.ppo.hidden_dim,
            config.ppo.gnn_layers,
        )
        train_env = YellowRiverWEFEnv(data, config.reward, config.physical, seed=seed)
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
            _progress(seed),
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
            subset["method"] = METHOD_LABEL
            subset["variant_key"] = variant.key
            subset["seed"] = seed
            subset["delay_scenario"] = delay_scenario
            subset["delay_months"] = int(delay_scenario.removeprefix("tau"))
            subset["evaluation_years"] = ";".join(
                str(year) for year in config.validation_years
            )
            subset["evaluation_policy"] = "deterministic"
            rows.append(subset)
        history = pd.concat([history, *rows], ignore_index=True)
        history = history.sort_values(["seed", "delay_months", "update"]).reset_index(
            drop=True
        )
        temporary = raw_path.with_suffix(".tmp.csv")
        history.to_csv(temporary, index=False)
        temporary.replace(raw_path)

    plot_gnnppo_uniform_delay_learning_curves(
        history,
        config.result_dir
        / "publication_figures"
        / "gnnppo_reward_uniform_delay_tau0_tau5.png",
        output_dir / "gnnppo_delay_reward_learning_curves_summary.csv",
    )
    caption_path = config.result_dir / "publication_figures" / "FIGURE_CAPTIONS.md"
    existing_caption = (
        caption_path.read_text(encoding="utf-8")
        if caption_path.exists()
        else "# Figure captions\n"
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
    args = parser.parse_args()
    run_gnnppo_delay_reward_curves(_saved_config(args.result_dir.resolve()))


if __name__ == "__main__":
    main()
