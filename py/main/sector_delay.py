"""Experiments 3 and 4: where stale observations matter in the WEF network.

The GNN-PPO (Experiment 3) and PPO (Experiment 4) policies trained with no delayed observations
checkpoints are each evaluated under the same sector-specific and single-node
delay definitions.  No policy is retrained: the experiments isolate the
operational consequence of *where* observations become stale.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import ExperimentConfig
from .data import load_wef_data, split_audit
from .environment import MAX_DELAY, YellowRiverWEFEnv
from .evaluation import evaluate_controller, model_controller
from .graph import NODE_NAMES, SECTOR_NODE_NAMES
from .model import build_model
from .plots import (
    plot_cascade_heatmap,
    plot_cross_method_cascade_comparison,
    plot_information_location_method_comparison,
    plot_node_vulnerability_figure,
    plot_sector_delay_risk_figure,
)


OUTCOMES = {
    "cost": "Average WEF penalty",
    "water_cost": "Average normalised water penalty",
    "energy_cost": "Average normalised energy penalty",
    "food_cost": "Average normalised food penalty",
}
# These legacy internal keys are retained for compatibility with existing CSV
# artifacts. The aggregate "cost" value is P_avg = -mean_monthly_reward.
EPISODE_OUTCOME_COLUMNS = {
    "cost": "mean_monthly_reward",
    "water_cost": "mean_water_cost_normalized",
    "energy_cost": "mean_energy_cost_normalized",
    "food_cost": "mean_food_cost_normalized",
}
SECTOR_NODE_LOOKUP = {
    node: sector for sector, nodes in SECTOR_NODE_NAMES.items() for node in nodes
}


@dataclass(frozen=True)
class PolicyDelayStudy:
    key: str
    model_name: str
    method_label: str
    experiment_number: int
    directory_name: str
    figure_numbers: tuple[int, int, int]


POLICY_DELAY_STUDIES = {
    "gnn_ppo": PolicyDelayStudy(
        key="gnn_ppo",
        model_name="gnn",
        method_label="GNN-PPO",
        experiment_number=3,
        directory_name="03_sector_specific_delay_gnnppo",
        figure_numbers=(3, 4, 5),
    ),
    "ppo": PolicyDelayStudy(
        key="ppo",
        model_name="ppo",
        method_label="PPO",
        experiment_number=4,
        directory_name="04_sector_specific_delay_ppo",
        figure_numbers=(6, 7, 8),
    ),
}


def delay_definitions() -> pd.DataFrame:
    """Return the complete, auditable set of sector/node delay scenarios.

    ``tau0`` is the no-delay reference.  Sector scenarios delay all
    nodes in one WEF branch while the remaining nodes observe the current
    month.  Node scenarios delay exactly one node.  The physical simulator is
    never delayed.
    """

    rows: list[dict[str, Any]] = [
        {
            "delay_scenario": "tau0",
            "scenario_type": "reference",
            "delayed_sector": "all",
            "delayed_node": "",
            "delay_months": 0,
            "n_delayed_nodes": len(NODE_NAMES),
            "delayed_nodes": ";".join(NODE_NAMES),
            "definition": "All 24 nodes use current-month observations.",
        }
    ]
    for sector, nodes in SECTOR_NODE_NAMES.items():
        for delay in range(1, MAX_DELAY + 1):
            rows.append(
                {
                    "delay_scenario": f"sector_{sector}_tau{delay}",
                    "scenario_type": "sector",
                    "delayed_sector": sector,
                    "delayed_node": "",
                    "delay_months": delay,
                    "n_delayed_nodes": len(nodes),
                    "delayed_nodes": ";".join(nodes),
                    "definition": (
                        f"Only {sector} nodes ({', '.join(nodes)}) use {delay}-month-old "
                        "observations; all other nodes use current-month observations."
                    ),
                }
            )
    for node in NODE_NAMES:
        for delay in range(1, MAX_DELAY + 1):
            rows.append(
                {
                    "delay_scenario": f"node_{node}_tau{delay}",
                    "scenario_type": "node",
                    "delayed_sector": SECTOR_NODE_LOOKUP[node],
                    "delayed_node": node,
                    "delay_months": delay,
                    "n_delayed_nodes": 1,
                    "delayed_nodes": node,
                    "definition": (
                        f"Only node {node} uses a {delay}-month-old observation; the other "
                        "23 nodes use current-month observations."
                    ),
                }
            )
    return pd.DataFrame(rows)


def _load_checkpoint_controller(
    checkpoint_path: Path,
    device: str,
    study: PolicyDelayStudy,
):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Experiment {study.experiment_number} requires the retained {study.method_label} "
            f"checkpoint: {checkpoint_path}"
        )
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch < 2.0 does not accept weights_only.
        payload = torch.load(checkpoint_path, map_location=device)
    if payload.get("model_name") != study.model_name:
        raise ValueError(
            f"Checkpoint model type is not {study.method_label}: {checkpoint_path}"
        )
    model = build_model(
        study.model_name,
        int(payload["feature_dim"]),
        int(payload["hidden_dim"]),
        int(payload["gnn_layers"]),
    )
    model.load_state_dict(payload["state_dict"])
    return model_controller(model, device)


def _hierarchical_ci(values: pd.DataFrame, column: str, seed: int) -> tuple[float, float, float]:
    """Two-way bootstrap over training seeds and repeated held-out years."""

    matrix = values.pivot(index="seed", columns="year", values=column).to_numpy(float)
    rng = np.random.default_rng(seed)
    seed_draws = rng.integers(0, matrix.shape[0], size=(10_000, matrix.shape[0]))
    year_draws = rng.integers(0, matrix.shape[1], size=(10_000, matrix.shape[1]))
    sampled = matrix[seed_draws[:, :, None], year_draws[:, None, :]].mean(axis=(1, 2))
    low, high = np.quantile(sampled, [0.025, 0.975])
    return float(matrix.mean()), float(low), float(high)


def _summarise_changes(changes: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouper: str | list[str] = group_columns[0] if len(group_columns) == 1 else group_columns
    for ordinal, (keys, group) in enumerate(changes.groupby(grouper, sort=False, dropna=False)):
        key_values = (keys,) if len(group_columns) == 1 else tuple(keys)
        identity = dict(zip(group_columns, key_values))
        for outcome, label in OUTCOMES.items():
            change_column = f"{outcome}_change"
            mean, low, high = _hierarchical_ci(
                group, change_column, 20260970 + ordinal * 19 + len(rows)
            )
            rows.append(
                {
                    **identity,
                    "outcome": outcome,
                    "outcome_label": label,
                    "mean_change": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "positive_change_means": "worse than the no-delay reference",
                    "n_training_seeds": int(group["seed"].nunique()),
                    "n_held_out_annual_forcing_sequences": int(group["year"].nunique()),
                    "ci_method": "two-way bootstrap across training seeds and held-out annual forcing sequences (10000 draws)",
                }
            )
    return pd.DataFrame(rows)


def _build_changes(annual: pd.DataFrame) -> pd.DataFrame:
    prepared = annual.copy()
    for outcome, source in EPISODE_OUTCOME_COLUMNS.items():
        prepared[outcome] = -prepared[source] if outcome == "cost" else prepared[source]
    baseline = prepared.loc[
        prepared["delay_scenario"] == "tau0", ["seed", "year", *OUTCOMES]
    ].copy()
    baseline = baseline.rename(columns={outcome: f"{outcome}_tau0" for outcome in OUTCOMES})
    delayed = prepared.loc[prepared["delay_scenario"] != "tau0"].copy()
    changes = delayed.merge(baseline, on=["seed", "year"], validate="many_to_one")
    for outcome in OUTCOMES:
        changes[f"{outcome}_change"] = changes[outcome] - changes[f"{outcome}_tau0"]
    return changes


def _sector_ranking(sector_effects: pd.DataFrame) -> pd.DataFrame:
    cost = sector_effects[sector_effects["outcome"] == "cost"].copy()
    tau5 = cost[cost["delay_months"] == MAX_DELAY].copy().rename(
        columns={
            "mean_change": "tau5_mean_cost_change",
            "ci95_low": "tau5_ci95_low",
            "ci95_high": "tau5_ci95_high",
        }
    )
    earliest = (
        cost[cost["ci95_low"] > 0.0]
        .groupby("delayed_sector", as_index=False)["delay_months"]
        .min()
        .rename(columns={"delay_months": "earliest_positive_ci_delay_months"})
    )
    output = tau5[
        ["delayed_sector", "tau5_mean_cost_change", "tau5_ci95_low", "tau5_ci95_high"]
    ].merge(earliest, on="delayed_sector", how="left")
    output["risk_rank_at_5_months"] = output["tau5_mean_cost_change"].rank(
        ascending=False, method="dense"
    ).astype(int)
    return output.sort_values(["risk_rank_at_5_months", "delayed_sector"]).reset_index(drop=True)


def _update_manifest(
    result_dir: Path,
    definitions: pd.DataFrame,
    annual: pd.DataFrame,
    study: PolicyDelayStudy,
) -> None:
    manifest_path = result_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest[f"experiment_{study.experiment_number}"] = {
        "policy": f"Existing {study.method_label} checkpoints trained with no delayed observations; no retraining.",
        "reference": "tau0: no node has delayed observations.",
        "sector_delay_definition": "sector_<water|energy|food>_tauN delays only that sector; every other node has no delayed observations.",
        "node_delay_definition": "node_<node>_tauN delays exactly one node; every other node has no delayed observations.",
        "delay_months": list(range(1, MAX_DELAY + 1)),
        "n_scenarios_including_reference": int(len(definitions)),
        "n_annual_outcomes": int(len(annual)),
    }
    if (result_dir / POLICY_DELAY_STUDIES["ppo"].directory_name).exists():
        manifest["status"] = "complete_for_experiments_1_to_4"
    else:
        manifest["status"] = "complete_for_experiments_1_to_3"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_results_guide(
    path: Path,
    study: PolicyDelayStudy,
    sector_ranking: pd.DataFrame,
    node_ranking: pd.DataFrame,
    comparison_note: str = "",
) -> None:
    top_sector = sector_ranking.sort_values("risk_rank_at_5_months").iloc[0]
    top_node = node_ranking.sort_values("risk_rank_at_5_months").iloc[0]
    sector_sentence = (
        "没有部门在 1–5 个月范围内出现 95% CI 完全高于 0 的平均 WEF 惩罚恶化；因此部门排序应视为描述性风险筛查，而不是确定的部门脆弱性结论。"
        if sector_ranking["earliest_positive_ci_delay_months"].dropna().empty
        else "至少一个部门在所测试延迟范围内出现 95% CI 完全高于 0 的平均 WEF 惩罚恶化；详见风险表。"
    )
    node_sentence = (
        "该节点的 95% CI 完全高于 0，是当前筛查中最清晰的局部风险信号。"
        if float(top_node["ci95_low"]) > 0.0
        else "该节点的区间仍跨越 0，因此只能作为点估计最高的探索性候选。"
    )
    text = f"""# 实验 {study.experiment_number}：{study.method_label} 的部门与节点信息延迟敏感性

## 设计

本实验不重新训练策略。对每一个已保存的 {study.method_label} 种子，以相同 seed × 测试年下的 `tau0` 作为参照：

- **部门情景**：仅水节点 W1–W9、能源节点 E1–E9 或灌溉节点 F1–F6 延迟 1–5 个月；其他节点保持当月信息。
- **单节点情景**：每次只延迟一个节点 1–5 个月；5 个月结果用于节点风险排序；其他 23 个节点保持当月信息。
- 延迟只作用于策略观测，不延迟物理系统；所有情景共享同一可行性投影器。

平均 WEF 惩罚变化均为“情景 − `tau0`”；正值表示惩罚增加、表现变差。水、能源、灌溉传播矩阵使用**未施加 reward 权重**的平均归一化惩罚分量，因此未把能源分量再乘 0.35。

## 当前结果

- 在 5 个月部门延迟下，按平均 WEF 惩罚点估计排序，最高的是 **{top_sector['delayed_sector']}**（变化 {top_sector['tau5_mean_cost_change']:+.6f}，95% CI [{top_sector['tau5_ci95_low']:+.6f}, {top_sector['tau5_ci95_high']:+.6f}]）。{sector_sentence}
- 单节点 5 个月筛查中，最高风险节点为 **{top_node['delayed_node']}**（{top_node['node_sector']} 部门；变化 {top_node['mean_change']:+.6f}，95% CI [{top_node['ci95_low']:+.6f}, {top_node['ci95_high']:+.6f}]）。{node_sentence}
- 部门和节点的多重比较均为探索性筛查，未作为额外确认性显著性检验。
{comparison_note}
## 文件对应

- `sector_delay_effects.csv`：部门延迟的平均 WEF 惩罚与三类惩罚分量变化。
- `wef_cascade_effects.csv`：源部门到水/能源/灌溉结果分量的传播矩阵数据。
- `sector_risk_ranking.csv`：5 个月部门延迟的描述性排序。
- `node_vulnerability_ranking.csv`：24 个节点的 5 个月单节点延迟筛查。
"""
    path.write_text(text, encoding="utf-8")


def _append_caption_once(caption_path: Path, heading: str, text: str) -> None:
    existing = caption_path.read_text(encoding="utf-8") if caption_path.exists() else ""
    if heading not in existing:
        with caption_path.open("a", encoding="utf-8") as handle:
            handle.write("\n\n" + text.strip() + "\n")


def _figure_filenames(study: PolicyDelayStudy) -> tuple[str, str, str]:
    """Keep the established Experiment 3 names while adding Experiment 4 PNGs."""

    if study.key == "gnn_ppo":
        return (
            "figure_3_sector_specific_delay_risk.png",
            "figure_4_wef_delay_propagation.png",
            "figure_5_single_node_delay_vulnerability.png",
        )
    return (
        "figure_6_sector_specific_delay_risk_ppo.png",
        "figure_7_wef_delay_propagation_ppo.png",
        "figure_8_single_node_delay_vulnerability_ppo.png",
    )


def _write_method_comparison(
    config: ExperimentConfig,
    figures_dir: Path,
    ppo_dir: Path,
) -> None:
    """Create descriptive PPO/GNN-PPO overlays from identically defined scenarios."""

    gnn_dir = config.result_dir / POLICY_DELAY_STUDIES["gnn_ppo"].directory_name
    required = [
        gnn_dir / "sector_delay_effects.csv",
        gnn_dir / "wef_cascade_effects.csv",
        gnn_dir / "node_vulnerability_ranking.csv",
        ppo_dir / "sector_delay_effects.csv",
        ppo_dir / "wef_cascade_effects.csv",
        ppo_dir / "node_vulnerability_ranking.csv",
    ]
    if not all(path.exists() for path in required):
        return
    gnn_sector = pd.read_csv(gnn_dir / "sector_delay_effects.csv")
    ppo_sector = pd.read_csv(ppo_dir / "sector_delay_effects.csv")
    gnn_cascade = pd.read_csv(gnn_dir / "wef_cascade_effects.csv")
    ppo_cascade = pd.read_csv(ppo_dir / "wef_cascade_effects.csv")
    gnn_nodes = pd.read_csv(gnn_dir / "node_vulnerability_ranking.csv")
    ppo_nodes = pd.read_csv(ppo_dir / "node_vulnerability_ranking.csv")
    sector_comparison = ppo_sector.merge(
        gnn_sector,
        on=["delayed_sector", "delay_months", "outcome"],
        suffixes=("_ppo", "_gnn_ppo"),
        validate="one_to_one",
    )
    node_comparison = ppo_nodes.merge(
        gnn_nodes,
        on=["delayed_node", "node_sector", "delay_months", "outcome"],
        suffixes=("_ppo", "_gnn_ppo"),
        validate="one_to_one",
    )
    sector_comparison.to_csv(ppo_dir / "ppo_vs_gnnppo_sector_delay_comparison.csv", index=False)
    node_comparison.to_csv(ppo_dir / "ppo_vs_gnnppo_node_delay_comparison.csv", index=False)
    plot_information_location_method_comparison(
        ppo_sector,
        gnn_sector,
        ppo_nodes,
        gnn_nodes,
        figures_dir / "figure_9_ppo_vs_gnnppo_information_location_comparison.png",
    )
    plot_cross_method_cascade_comparison(
        ppo_cascade,
        gnn_cascade,
        figures_dir / "figure_10_cross_method_wef_delay_propagation.png",
    )
    _append_caption_once(
        figures_dir / "FIGURE_CAPTIONS.md",
        "## Figure 9 — PPO versus GNN-PPO information-location sensitivity",
        """## Figure 9 — PPO versus GNN-PPO information-location sensitivity

Both policy families are evaluated with exactly the same sector-specific and single-node delay definitions. Panel a compares the sector-specific dose response; panel b compares the 24-node five-month point estimates. Each change in average WEF penalty is within-policy relative to that policy's same-seed, same-year `tau0` reference. The cross-architecture display is descriptive and does not assert a new node-level significance test.""",
    )
    _append_caption_once(
        figures_dir / "FIGURE_CAPTIONS.md",
        "## Figure 10 — Cross-method WEF delay propagation",
        """## Figure 10 — Cross-method WEF delay propagation

Rows compare the fixed PPO and GNN-PPO policies, and columns identify the sector whose observations are delayed. Cell values are changes in the unweighted average normalised water, energy and food penalty components relative to the same policy, seed and held-out year under `tau0`; positive values denote deterioration. All panels share one colour scale so that response magnitudes are comparable across policies.""",
    )


def run_information_location_delay_experiment(
    config: ExperimentConfig,
    policy_key: str,
) -> dict[str, pd.DataFrame]:
    """Run one fixed-policy WEF information-location delay experiment."""

    if policy_key not in POLICY_DELAY_STUDIES:
        raise ValueError(f"Unknown policy delay study: {policy_key}")
    study = POLICY_DELAY_STUDIES[policy_key]
    data = load_wef_data(config.data_path)
    split_audit(data, config.train_years, config.validation_years, config.test_years)
    definitions = delay_definitions()
    result_dir = config.result_dir / study.directory_name
    result_dir.mkdir(parents=True, exist_ok=True)
    for old_file in result_dir.glob("*"):
        if old_file.is_file():
            old_file.unlink()
    figures_dir = config.result_dir / "publication_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    filenames = _figure_filenames(study)
    for filename in filenames:
        path = figures_dir / filename
        if path.exists():
            path.unlink()

    annual_frames: list[pd.DataFrame] = []
    scenarios = definitions["delay_scenario"].tolist()
    for seed in config.seeds:
        checkpoint = config.result_dir / "checkpoints" / f"{study.key}_seed{seed}.pt"
        controller = _load_checkpoint_controller(checkpoint, config.device, study)
        print(
            f"[Experiment {study.experiment_number} {study.method_label} seed={seed}] "
            f"evaluating {len(scenarios)} delay scenarios",
            flush=True,
        )
        for scenario in scenarios:
            env = YellowRiverWEFEnv(data, config.reward, config.physical, seed=seed)
            _, episodes = evaluate_controller(
                env,
                controller,
                config.test_years,
                study.method_label,
                seed,
                delay_scenario=scenario,
            )
            episodes["variant_key"] = study.key
            annual_frames.append(episodes)
    annual = pd.concat(annual_frames, ignore_index=True).merge(
        definitions, on="delay_scenario", validate="many_to_one"
    )
    annual = annual.sort_values(
        ["scenario_type", "delayed_sector", "delayed_node", "delay_months", "seed", "year"]
    ).reset_index(drop=True)
    annual.to_csv(result_dir / "annual_episode_outcomes.csv", index=False)
    definitions.to_csv(result_dir / "delay_definitions.csv", index=False)

    changes = _build_changes(annual)
    sector_changes = changes[changes["scenario_type"] == "sector"].copy()
    node_changes = changes[changes["scenario_type"] == "node"].copy()
    sector_effects = _summarise_changes(sector_changes, ["delayed_sector", "delay_months"])
    cascade_effects = sector_effects[
        sector_effects["outcome"].isin(("water_cost", "energy_cost", "food_cost"))
    ].copy()
    node_effects = _summarise_changes(
        node_changes, ["delayed_node", "delayed_sector", "delay_months"]
    ).rename(columns={"delayed_sector": "node_sector"})
    sector_ranking = _sector_ranking(sector_effects)
    node_ranking = node_effects[
        (node_effects["outcome"] == "cost")
        & (node_effects["delay_months"] == MAX_DELAY)
    ].copy()
    node_ranking["risk_rank_at_5_months"] = node_ranking["mean_change"].rank(
        ascending=False, method="dense"
    ).astype(int)
    node_ranking = node_ranking.sort_values("risk_rank_at_5_months").reset_index(drop=True)

    changes.to_csv(result_dir / "annual_outcome_changes_vs_tau0.csv", index=False)
    sector_effects.to_csv(result_dir / "sector_delay_effects.csv", index=False)
    cascade_effects.to_csv(result_dir / "wef_cascade_effects.csv", index=False)
    sector_ranking.to_csv(result_dir / "sector_risk_ranking.csv", index=False)
    node_effects.to_csv(result_dir / "node_delay_effects.csv", index=False)
    node_ranking.to_csv(result_dir / "node_vulnerability_ranking.csv", index=False)

    plot_sector_delay_risk_figure(
        sector_effects,
        sector_ranking,
        figures_dir / filenames[0],
        policy_label=study.method_label,
        experiment_number=study.experiment_number,
    )
    plot_cascade_heatmap(
        cascade_effects,
        figures_dir / filenames[1],
        policy_label=study.method_label,
        experiment_number=study.experiment_number,
    )
    plot_node_vulnerability_figure(
        node_ranking,
        figures_dir / filenames[2],
        policy_label=study.method_label,
        experiment_number=study.experiment_number,
    )
    _append_caption_once(
        figures_dir / "FIGURE_CAPTIONS.md",
        f"## Figure {study.figure_numbers[0]} — Sector-specific {study.method_label} delay risk",
        f"""## Figure {study.figure_numbers[0]} — Sector-specific {study.method_label} delay risk

Markers show the change in average WEF penalty, `Delta P_avg = P_avg_delay - P_avg_tau0`, and the legend identifies the sector whose observations are delayed by 1–5 months. Horizontal intervals are 95% two-way bootstrap confidence intervals across 10 training seeds and three held-out annual forcing sequences. All other nodes have no delayed observations. This is a fixed-policy information-location sensitivity analysis, not a re-training experiment.

## Figure {study.figure_numbers[1]} — {study.method_label} WEF delay propagation

Each panel sets the source of stale information and shows the resulting change in the unweighted average normalised water, energy and food penalty components. Positive values mean deterioration. The three normalised components are displayed before reward weighting, so the 0.35 energy reward coefficient does not conceal a physical branch response.

## Figure {study.figure_numbers[2]} — Single-node {study.method_label} vulnerability screen

Panel a shows all 24 node estimates with 95% two-way bootstrap confidence intervals, panel b groups the mean changes by sector, and panel c enlarges the five highest point estimates. In each scenario, exactly one node receives a five-month-old observation and the other 23 nodes have no delayed observations. The ranking is exploratory and identifies information locations that merit focused monitoring or follow-up stress testing.""",
    )
    comparison_note = ""
    if policy_key == "ppo":
        _write_method_comparison(config, figures_dir, result_dir)
        comparison_note = (
            "\n与 GNN-PPO 的同定义并列结果见 `ppo_vs_gnnppo_sector_delay_comparison.csv`、"
            "`ppo_vs_gnnppo_node_delay_comparison.csv` 以及 Figure 9；这些是描述性并列，"
            "每个算法都相对自身的 `tau0` 参照计算。\n"
        )
    _write_results_guide(
        result_dir / "RESULTS_GUIDE_CN.md",
        study,
        sector_ranking,
        node_ranking,
        comparison_note,
    )
    _update_manifest(config.result_dir, definitions, annual, study)
    from .experiments import refresh_root_results_guide

    refresh_root_results_guide(config)
    return {
        "sector_delay_effects": sector_effects,
        "sector_risk_ranking": sector_ranking,
        "node_vulnerability_ranking": node_ranking,
    }


def run_sector_delay_experiment(config: ExperimentConfig) -> dict[str, pd.DataFrame]:
    """Backward-compatible Experiment 3 GNN-PPO entry point."""

    return run_information_location_delay_experiment(config, "gnn_ppo")


def run_ppo_sector_delay_experiment(config: ExperimentConfig) -> dict[str, pd.DataFrame]:
    """Experiment 4: same information-location screen for PPO checkpoints."""

    return run_information_location_delay_experiment(config, "ppo")


def _saved_config(result_dir: Path) -> ExperimentConfig:
    config = ExperimentConfig(result_dir=result_dir.resolve())
    config_path = result_dir / "experiment_config.json"
    if not config_path.exists():
        return config
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    config.data_path = Path(saved.get("data_path", config.data_path))
    config.train_years = tuple(saved.get("train_years", config.train_years))
    config.validation_years = tuple(saved.get("validation_years", config.validation_years))
    config.test_years = tuple(saved.get("test_years", config.test_years))
    config.seeds = tuple(saved.get("seeds", config.seeds))
    config.device = str(saved.get("device", config.device))
    for section in ("reward", "physical"):
        for key, value in saved.get(section, {}).items():
            if hasattr(getattr(config, section), key):
                setattr(getattr(config, section), key, value)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=Path("result"))
    parser.add_argument(
        "--policy",
        choices=tuple(POLICY_DELAY_STUDIES),
        default="gnn_ppo",
        help="Checkpoint family trained with no delayed observations to evaluate.",
    )
    args = parser.parse_args()
    run_information_location_delay_experiment(_saved_config(args.result_dir.resolve()), args.policy)


if __name__ == "__main__":
    main()
