"""The retained PPO/GNN-PPO WEF experiments.

Experiment 1 compares the two policies with no delayed observations.  Experiment
2 applies the same delay to every one of the 24 WEF nodes.  Experiments 3 and
4 live in :mod:`main.sector_delay`; they reuse the trained GNN-PPO and PPO
checkpoints, respectively, to isolate the risk of delaying selected sectors or
nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import platform

import numpy as np
import pandas as pd
import torch

from .config import ExperimentConfig, seed_everything
from .data import load_wef_data, split_audit
from .environment import MAX_DELAY, YellowRiverWEFEnv
from .evaluation import evaluate_controller, model_controller
from .model import build_model
from .plots import (
    plot_current_information_figure,
    plot_training_reward_convergence,
    plot_uniform_delay_figure,
)
from .ppo import checkpoint_payload, train_policy
from .statistical_analysis import (
    build_analysis_units,
    descriptive_statistics,
    paired_delay_average_effects,
    paired_method_effects,
    paired_seed_differences,
)


@dataclass(frozen=True)
class PolicyVariant:
    key: str
    label: str
    model_name: str
    description: str


POLICY_VARIANTS = (
    PolicyVariant("ppo", "PPO", "ppo", "Conventional PPO trained with no delayed observations."),
    PolicyVariant("gnn_ppo", "GNN-PPO", "gnn", "Heterogeneous graph PPO trained with no delayed observations."),
)
UNIFORM_DELAY_SCENARIOS = tuple(f"tau{delay}" for delay in range(MAX_DELAY + 1))
EVALUATION_DELAY_SCENARIOS = UNIFORM_DELAY_SCENARIOS
METHOD_ORDER = {variant.label: index for index, variant in enumerate(POLICY_VARIANTS)}
DELAY_ORDER = {scenario: index for index, scenario in enumerate(EVALUATION_DELAY_SCENARIOS)}


def _progress(label: str, seed: int):
    def report(update: int, total: int, diagnostics: dict[str, float]) -> None:
        interval = max(1, total // 8)
        if update == 1 or update == total or update % interval == 0:
            validation = diagnostics.get("validation_mean_monthly_reward", float("nan"))
            validation_text = f", val={validation:.3f}" if np.isfinite(validation) else ""
            print(
                f"[{label} seed={seed}] {update:4d}/{total}: "
                f"return={diagnostics['mean_episode_reward']:.3f}{validation_text}",
                flush=True,
            )

    return report


def _annotate(
    monthly: pd.DataFrame, episodes: pd.DataFrame, variant: PolicyVariant
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for frame in (monthly, episodes):
        frame["variant_key"] = variant.key
    return monthly, episodes


def _sort_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    sort_columns: list[str] = []
    temporary: list[str] = []
    if "method" in result:
        result["_method_order"] = result["method"].map(METHOD_ORDER)
        sort_columns.append("_method_order")
        temporary.append("_method_order")
    if "delay_scenario" in result:
        result["_delay_order"] = result["delay_scenario"].map(DELAY_ORDER)
        sort_columns.append("_delay_order")
        temporary.append("_delay_order")
    for column in ("metric", "seed", "year", "month"):
        if column in result:
            sort_columns.append(column)
    if sort_columns:
        result = result.sort_values(sort_columns).reset_index(drop=True)
    return result.drop(columns=temporary)


def _write_comparison_bundle(
    result_dir: Path,
    tables_dir: Path,
    table_prefix: str,
    year_level: pd.DataFrame,
    seed_level: pd.DataFrame,
    summary: pd.DataFrame,
    seed_differences: pd.DataFrame,
    effects: pd.DataFrame,
    overall_effects: pd.DataFrame | None = None,
) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "metrics_by_test_year.csv": year_level,
        "seed_level_metrics.csv": seed_level,
        "summary_statistics.csv": summary,
        "paired_seed_differences.csv": seed_differences,
        "paired_effects.csv": effects,
    }
    if overall_effects is not None:
        outputs["overall_delay_effects.csv"] = overall_effects
    for filename, frame in outputs.items():
        _sort_frame(frame).to_csv(result_dir / filename, index=False)
    tables_dir.mkdir(parents=True, exist_ok=True)
    _sort_frame(summary).to_csv(tables_dir / f"{table_prefix}_descriptive_statistics.csv", index=False)
    _sort_frame(effects).to_csv(tables_dir / f"{table_prefix}_paired_effects.csv", index=False)
    if overall_effects is not None:
        _sort_frame(overall_effects).to_csv(
            tables_dir / f"{table_prefix}_overall_delay_effects.csv", index=False
        )


def _clear_files(directory: Path) -> None:
    if directory.exists():
        for path in directory.iterdir():
            if path.is_file():
                path.unlink()


def _save_result_guide(
    config: ExperimentConfig,
    current_effects: pd.DataFrame,
    uniform_effects: pd.DataFrame,
) -> None:
    current = current_effects.query("metric == 'cost'").iloc[0]
    uniform = uniform_effects.query("metric == 'cost'").iloc[0]
    text = f"""# 实验 1–4 结果阅读指南

本目录仅保留以下研究问题：

1. **实验 1（当前信息）**：PPO 与 GNN-PPO 在全部 24 个节点使用当月观测时，是否存在明显性能差异？
2. **实验 2（统一延迟）**：24 个节点是否同时延迟 0–5 个月时，PPO 与 GNN-PPO 如何变化？
3. **实验 3（GNN-PPO 位置敏感性）**：仅让水、能源或灌溉部门延迟 1–5 个月，或仅延迟一个节点时，GNN-PPO 的平均 WEF 惩罚与水/能/灌溉惩罚分量如何变化？
4. **实验 4（PPO 位置敏感性）**：在与实验 3 完全相同的部门/单节点延迟情景下，PPO 的风险位置与传播模式是否相同？

## 延迟定义（必须按此解释）

- `tauN`：**24 个节点全部**使用 N 个月前的观测。这是实验 2 的统一延迟定义。
- `sector_water_tauN`、`sector_energy_tauN`、`sector_food_tauN`：只有相应部门的节点使用 N 个月前观测，其余节点均使用当月观测。这是实验 3 和 4 的部门延迟定义；三组节点分别为 W1–W9、E1–E9、F1–F6。
- `node_<名称>_tauN`：仅一个节点使用 N 个月前观测，其余 23 个节点使用当月观测。这是实验 3 和 4 的单节点筛查定义。
- 延迟只改变输入策略的**观测信息**，不延迟物理状态演化，也不改变共同的可行性投影器。每个年内第一个月没有使用跨年历史；缓冲区以该年首月观测初始化。

## 可报告的核心结论

- **实验 1：** 当前信息下，GNN-PPO 相对于 PPO 的平均 WEF 惩罚改善为 {float(current['mean_improvement']):+.4f}（正值代表 GNN-PPO 的惩罚更低），95% CI [{float(current['hierarchical_ci95_low']):+.4f}, {float(current['hierarchical_ci95_high']):+.4f}]。这应表述为“未显示明确优势”，而不是未经预设等效界值的统计等效。
- **实验 2：** 统一 1–5 个月延迟平均下，GNN-PPO 相对于 PPO 的平均 WEF 惩罚改善为 {float(uniform['mean_improvement']):+.4f}，95% CI [{float(uniform['hierarchical_ci95_low']):+.4f}, {float(uniform['hierarchical_ci95_high']):+.4f}]。图 2 展示完整剂量响应：重点是 PPO 随信息陈旧而恶化、GNN-PPO 更稳定的模式，而不是“GNN 在任何条件下都更好”。
- **实验 3：** 是固定已训练 GNN-PPO 的观测位置敏感性实验。图 3–5 和 `03_sector_specific_delay_gnnppo/` 记录哪个部门/节点的陈旧信息导致最大的平均 WEF 惩罚风险，以及惩罚分量变化在水、能源、灌溉三条分支中的传播。
- **实验 4：** 将实验 3 的固定策略替换为 PPO，其他情景、训练种子、测试年份和两层 bootstrap 保持一致。图 6–8 与 `04_sector_specific_delay_ppo/` 给出 PPO 的对应结果；图 9 用并列小面板和节点风险散点图比较两种策略，不能将位置敏感性结果外推为重新训练后的因果算法效应。

## 推断边界

- 10 个训练种子是独立的算法重复；2019–2021 是保留的年度强迫序列，含重复气候型输入，因此不是独立流域样本。
- 95% CI 使用训练种子和年度强迫序列的两层 bootstrap；延迟等级是同一策略上的重复条件，不能当作额外独立样本。
- 结果描述“学习策略 + 共同确定性可行性投影器”的组合。投影器的年末库容约束使用当年余下外源序列，因而不应外推为完全因果的实际调度器。
"""
    (config.result_dir / "RESULTS_GUIDE_CN.md").write_text(text, encoding="utf-8")


def refresh_root_results_guide(config: ExperimentConfig) -> None:
    """Refresh the root guide after fixed-policy Experiments 3/4 are available."""

    current_effects_path = config.result_dir / "01_current_information_graph_encoder" / "paired_effects.csv"
    uniform_effects_path = config.result_dir / "02_uniform_delay_robustness" / "overall_delay_effects.csv"
    if not current_effects_path.exists() or not uniform_effects_path.exists():
        return
    _save_result_guide(
        config,
        pd.read_csv(current_effects_path),
        pd.read_csv(uniform_effects_path),
    )


def rebuild_result_artifacts(
    config: ExperimentConfig,
    episodes: pd.DataFrame,
) -> dict[str, Any]:
    """Recompute only the Experiment 1/2 tables and their two PNG figures."""

    result_dir = config.result_dir
    tables_dir = result_dir / "publication_tables"
    figures_dir = result_dir / "publication_figures"
    for directory in (
        result_dir / "01_current_information_graph_encoder",
        result_dir / "02_uniform_delay_robustness",
        tables_dir,
        figures_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
        _clear_files(directory)

    methods = tuple(variant.label for variant in POLICY_VARIANTS)
    current_year, current_seed = build_analysis_units(episodes, methods, ("tau0",))
    current_summary = descriptive_statistics(current_seed)
    current_effects = paired_method_effects(
        current_year, "current_information", "PPO", "GNN-PPO", ("tau0",)
    )
    _write_comparison_bundle(
        result_dir / "01_current_information_graph_encoder",
        tables_dir,
        "table_1_current_information_ppo_vs_gnnppo",
        current_year,
        current_seed,
        current_summary,
        paired_seed_differences(current_seed, "PPO", "GNN-PPO"),
        current_effects,
    )

    delayed_scenarios = UNIFORM_DELAY_SCENARIOS[1:]
    uniform_year, uniform_seed = build_analysis_units(episodes, methods, delayed_scenarios)
    uniform_summary = descriptive_statistics(uniform_seed)
    uniform_effects = paired_method_effects(
        uniform_year, "uniform_delay_robustness", "PPO", "GNN-PPO", delayed_scenarios
    )
    uniform_overall = paired_delay_average_effects(
        uniform_year,
        "uniform_delay_robustness_overall",
        "PPO",
        "GNN-PPO",
        delayed_scenarios,
    )
    _write_comparison_bundle(
        result_dir / "02_uniform_delay_robustness",
        tables_dir,
        "table_2_uniform_delay_ppo_vs_gnnppo",
        uniform_year,
        uniform_seed,
        uniform_summary,
        paired_seed_differences(uniform_seed, "PPO", "GNN-PPO"),
        uniform_effects,
        uniform_overall,
    )

    plot_current_information_figure(
        current_year, figures_dir / "figure_1_current_information_comparison.png"
    )
    all_uniform_year, _ = build_analysis_units(episodes, methods, UNIFORM_DELAY_SCENARIOS)
    plot_uniform_delay_figure(
        all_uniform_year, figures_dir / "figure_2_uniform_delay_robustness.png"
    )
    training_history_path = result_dir / "training" / "training_history.csv"
    if training_history_path.exists():
        plot_training_reward_convergence(
            pd.read_csv(training_history_path),
            figures_dir / "training_mean_episode_reward_tau0.png",
            result_dir / "training" / "training_mean_episode_reward_tau0_summary.csv",
            rollout_episodes=config.ppo.rollout_episodes,
            episode_steps=12,
        )
    captions = """# Figure captions

## Training convergence — Mean episode reward with no delayed observations

PPO and GNN-PPO are trained only with no delayed observations (`tau0`). The horizontal axis reports cumulative environment interactions: each update contains four 12-month episodes, equivalent to 48 environment steps. Curves show the ten-seed mean after a 10-update trailing smooth, and bands show 95% bootstrap confidence intervals across training seeds. To retain a balanced comparison, the plot stops at update 110 (5,280 environment steps), the last update available for every PPO and GNN-PPO seed; later updates are omitted because of early stopping.

## Figure 1 — No-delay comparison

Each paired point is one independently trained seed, averaged across the held-out 2019–2021 annual forcing sequences. The primary endpoint is the average WEF penalty (`P_avg = -mean reward`; lower is better). The figure reports the observed difference and uncertainty; it does not claim statistical equivalence.

## Figure 2 — Uniform observation-delay robustness

`tauN` means that all 24 WEF nodes use observations N months old. Points show the mean reward across 10 independently trained seeds and the 2019–2021 held-out forcing sequences; lines connect the ordered delay levels. The aligned table reports the same means and their observed difference (`GNN-PPO − PPO`), for which positive values favour GNN-PPO. PPO reward deteriorates under uniformly stale information, whereas GNN-PPO remains comparatively stable. Higher reward is better.
"""
    (figures_dir / "FIGURE_CAPTIONS.md").write_text(captions, encoding="utf-8")
    _save_result_guide(config, current_effects, uniform_overall)
    manifest = {
        "status": "complete_for_experiments_1_and_2",
        "study_design": "Experiment 1: no-delay PPO vs GNN-PPO; Experiment 2: uniform 0-5-month delay across all 24 nodes",
        "train_years": list(config.train_years),
        "validation_years": list(config.validation_years),
        "test_years": list(config.test_years),
        "test_forcing_limitation": "2019-2021 are repeated evaluation conditions with repeated-climatology input; they are not independent catchments.",
        "seeds": list(config.seeds),
        "n_training_seeds": len(config.seeds),
        "delay_scenarios": list(UNIFORM_DELAY_SCENARIOS),
        "result_groups": [
            "training",
            "checkpoints",
            "raw/experiment_1_2_*",
            "01_current_information_graph_encoder",
            "02_uniform_delay_robustness",
            "03_sector_specific_delay_gnnppo",
            "04_sector_specific_delay_ppo",
            "publication_tables",
            "publication_figures",
        ],
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    (result_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "current_information_effects": current_effects,
        "uniform_delay_effects": uniform_effects,
        "uniform_delay_overall_effects": uniform_overall,
    }


def run_experiments(config: ExperimentConfig) -> dict[str, Any]:
    """Train PPO/GNN-PPO policies with no delayed observations and run Experiments 1–4."""

    config.result_dir.mkdir(parents=True, exist_ok=True)
    config.save(config.result_dir / "experiment_config.json")
    data = load_wef_data(config.data_path)
    split_audit(data, config.train_years, config.validation_years, config.test_years)
    training_rows: list[pd.DataFrame] = []
    monthly_rows: list[pd.DataFrame] = []
    episode_rows: list[pd.DataFrame] = []
    parameter_rows: list[dict[str, Any]] = []
    checkpoint_dir = config.result_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for variant in POLICY_VARIANTS:
        model = build_model(variant.model_name, config.feature_dim, config.ppo.hidden_dim, config.ppo.gnn_layers)
        parameter_rows.append(
            {
                "method": variant.label,
                "variant_key": variant.key,
                "architecture": "heterogeneous graph PPO" if variant.model_name == "gnn" else "conventional PPO",
                "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
                "description": variant.description,
            }
        )

    for seed in config.seeds:
        for variant in POLICY_VARIANTS:
            seed_everything(seed)
            model = build_model(variant.model_name, config.feature_dim, config.ppo.hidden_dim, config.ppo.gnn_layers)
            train_env = YellowRiverWEFEnv(data, config.reward, config.physical, seed=seed)
            validation_env = YellowRiverWEFEnv(data, config.reward, config.physical, seed=500_000 + seed)
            training_result = train_policy(
                train_env,
                model,
                config.train_years,
                config.ppo,
                seed,
                validation_env,
                config.validation_years,
                config.validation,
                config.device,
                _progress(variant.label, seed),
            )
            history = pd.DataFrame(training_result.history)
            history["method"] = variant.label
            history["variant_key"] = variant.key
            history["seed"] = seed
            history["selected_update"] = training_result.best_update
            history["selected_validation_score"] = training_result.best_validation_score
            history["updates_completed"] = training_result.updates_completed
            history["stopped_early"] = training_result.stopped_early
            training_rows.append(history)
            torch.save(
                checkpoint_payload(
                    model, variant.model_name, seed, config.feature_dim, config.ppo.hidden_dim,
                    config.ppo.gnn_layers, variant.key, training_result.best_update,
                    training_result.best_validation_score, training_result.updates_completed,
                    training_result.stopped_early,
                ),
                checkpoint_dir / f"{variant.key}_seed{seed}.pt",
            )
            controller = model_controller(model, config.device)
            for scenario in EVALUATION_DELAY_SCENARIOS:
                eval_env = YellowRiverWEFEnv(data, config.reward, config.physical, seed=seed)
                monthly, episodes = evaluate_controller(
                    eval_env, controller, config.test_years, variant.label, seed, delay_scenario=scenario
                )
                monthly, episodes = _annotate(monthly, episodes, variant)
                monthly_rows.append(monthly)
                episode_rows.append(episodes)

    training_dir = config.result_dir / "training"
    training_dir.mkdir(parents=True, exist_ok=True)
    training = _sort_frame(pd.concat(training_rows, ignore_index=True))
    training.to_csv(training_dir / "training_history.csv", index=False)
    selection_columns = ["method", "variant_key", "seed", "selected_update", "selected_validation_score", "updates_completed", "stopped_early"]
    training[selection_columns].drop_duplicates().to_csv(training_dir / "selected_checkpoints.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(config.result_dir / "methods_and_parameter_counts.csv", index=False)

    raw_dir = config.result_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    monthly = _sort_frame(pd.concat(monthly_rows, ignore_index=True))
    episodes = _sort_frame(pd.concat(episode_rows, ignore_index=True))
    monthly.to_csv(raw_dir / "experiment_1_2_monthly_trajectories.csv", index=False)
    episodes.to_csv(raw_dir / "experiment_1_2_episode_outcomes.csv", index=False)
    outputs = rebuild_result_artifacts(config, episodes)
    from .sector_delay import run_ppo_sector_delay_experiment, run_sector_delay_experiment

    outputs["experiment_3"] = run_sector_delay_experiment(config)
    outputs["experiment_4"] = run_ppo_sector_delay_experiment(config)
    return outputs
