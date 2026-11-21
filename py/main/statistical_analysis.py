"""Seed-aware descriptive and paired analyses for Experiments 1 and 2."""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd


REPORT_METRICS: dict[str, dict[str, str | bool]] = {
    "cost": {
        "label": "Average WEF penalty",
        "unit": "penalty",
        "higher_is_better": False,
    },
    "ecological_flow_reliability_pct": {
        "label": "Ecological flow reliability",
        "unit": "%",
        "higher_is_better": True,
    },
    "ecological_deficit_total": {
        "label": "Annual ecological-flow deficit volume",
        "unit": "1e8 m3 yr-1",
        "higher_is_better": False,
    },
    "irrigation_satisfaction_ratio_pct": {
        "label": "Irrigation satisfaction ratio",
        "unit": "%",
        "higher_is_better": True,
    },
    "crop_deficit_total": {
        "label": "Annual irrigation water deficit",
        "unit": "1e8 m3 yr-1",
        "higher_is_better": False,
    },
    "energy_demand_satisfaction_ratio_pct": {
        "label": "Energy-demand satisfaction ratio",
        "unit": "%",
        "higher_is_better": True,
    },
    "electricity_import_total": {
        "label": "Electricity import",
        "unit": "GWh yr-1",
        "higher_is_better": False,
    },
}


def _bootstrap_mean_interval(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    indexes = rng.integers(0, len(values), size=(10_000, len(values)))
    sampled = values[indexes].mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return float(low), float(high)


def _hierarchical_bootstrap_interval(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    seed_draws = rng.integers(0, values.shape[0], size=(10_000, values.shape[0]))
    year_draws = rng.integers(0, values.shape[1], size=(10_000, values.shape[1]))
    sampled = values[seed_draws[:, :, None], year_draws[:, None, :]].mean(axis=(1, 2))
    low, high = np.quantile(sampled, [0.025, 0.975])
    return float(low), float(high)


def _exact_sign_permutation_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0 or np.allclose(values, 0.0):
        return 1.0
    observed = abs(float(values.mean()))
    signs = np.asarray(list(product((-1.0, 1.0), repeat=len(values))), dtype=float)
    randomized = np.abs((signs * values.reshape(1, -1)).mean(axis=1))
    return float(np.mean(randomized >= observed - 1.0e-12))


def _paired_effect_size(values: np.ndarray) -> float:
    if len(values) < 2:
        return float("nan")
    standard_deviation = float(np.std(values, ddof=1))
    return 0.0 if standard_deviation <= 1.0e-12 else float(np.mean(values) / standard_deviation)


def _holm_adjust(p_values: np.ndarray) -> np.ndarray:
    output = np.full(len(p_values), np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(p_values))
    ordered = finite[np.argsort(p_values[finite])]
    running = 0.0
    for rank, index in enumerate(ordered):
        running = max(running, (len(ordered) - rank) * float(p_values[index]))
        output[index] = min(1.0, running)
    return output


def _delay_months(scenario: str) -> int:
    return int(str(scenario).removeprefix("tau"))


def add_report_metrics(episodes: pd.DataFrame) -> pd.DataFrame:
    result = episodes.copy()
    # The legacy key "cost" is retained for CSV compatibility. Its value is the
    # average WEF penalty, P_avg = -mean_monthly_reward, so lower remains better.
    result["cost"] = -result["mean_monthly_reward"]
    result["ecological_flow_reliability_pct"] = 100.0 * result["ecological_flow_reliability"]
    result["irrigation_satisfaction_ratio_pct"] = 100.0 * result["irrigation_satisfaction_ratio"]
    result["energy_demand_satisfaction_ratio_pct"] = (
        100.0 * result["energy_demand_satisfaction_ratio"]
    )
    return result


def build_analysis_units(
    episodes: pd.DataFrame,
    methods: tuple[str, ...],
    scenarios: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate annual outcomes to training-seed means for reporting."""

    prepared = add_report_metrics(episodes)
    year_level = prepared[
        prepared["method"].isin(methods) & prepared["delay_scenario"].isin(scenarios)
    ].copy()
    if year_level.empty:
        raise ValueError("No episode records match the requested methods and delay scenarios.")
    required = {"method", "seed", "year", "delay_scenario", *REPORT_METRICS}
    missing = sorted(required.difference(year_level.columns))
    if missing:
        raise ValueError(f"Episode records are missing reporting fields: {missing}")
    metadata = [column for column in ("method", "variant_key", "delay_scenario", "seed") if column in year_level]
    year_level = year_level.groupby([*metadata, "year"], as_index=False)[list(REPORT_METRICS)].mean()
    seed_level = year_level.groupby(metadata, as_index=False)[list(REPORT_METRICS)].mean()
    counts = year_level.groupby(metadata, as_index=False)["year"].nunique().rename(columns={"year": "n_annual_forcing_sequences_averaged"})
    seed_level = seed_level.merge(counts, on=metadata, validate="one_to_one")
    seed_level["analysis_unit"] = "training-seed mean over held-out annual forcing sequences"
    return year_level, seed_level


def descriptive_statistics(
    seed_level: pd.DataFrame,
    group_columns: tuple[str, ...] = ("method", "delay_scenario"),
    bootstrap_seed: int = 20260810,
) -> pd.DataFrame:
    rng = np.random.default_rng(bootstrap_seed)
    rows: list[dict[str, str | float | int]] = []
    grouper: str | list[str] = group_columns[0] if len(group_columns) == 1 else list(group_columns)
    for keys, group in seed_level.groupby(grouper, dropna=False):
        values_for_keys = (keys,) if len(group_columns) == 1 else tuple(keys)
        identity = dict(zip(group_columns, values_for_keys))
        for metric, specification in REPORT_METRICS.items():
            values = group[metric].to_numpy(float)
            mean = float(values.mean())
            sd = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
            low, high = _bootstrap_mean_interval(values, rng)
            rows.append(
                {
                    **identity,
                    "metric": metric,
                    "metric_label": specification["label"],
                    "unit": specification["unit"],
                    "favourable_direction": "higher" if specification["higher_is_better"] else "lower",
                    "n_training_seeds": len(values),
                    "n_annual_forcing_sequences_per_seed": int(group["n_annual_forcing_sequences_averaged"].min()),
                    "mean": mean,
                    "sd": sd,
                    "ci95_low": low,
                    "ci95_high": high,
                    "mean_sd": f"{mean:.4f} ± {sd:.4f}",
                    "mean_ci95": f"{mean:.4f} [{low:.4f}, {high:.4f}]",
                    "ci_method": "percentile bootstrap across training-seed means (10000 draws)",
                }
            )
    return pd.DataFrame(rows)


def _paired_metric_effects(
    year_level: pd.DataFrame,
    comparison: str,
    baseline: str,
    candidate: str,
    scenarios: tuple[str, ...],
    bootstrap_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(bootstrap_seed)
    keys = ["delay_scenario", "seed", "year"]
    left = year_level[year_level["method"] == baseline]
    right = year_level[year_level["method"] == candidate]
    paired = left.merge(right, on=keys, suffixes=("_baseline", "_candidate"), validate="one_to_one")
    rows: list[dict[str, str | float | int]] = []
    for scenario in scenarios:
        group = paired[paired["delay_scenario"] == scenario]
        seeds = sorted(group["seed"].unique())
        years = sorted(group["year"].unique())
        if not seeds or not years:
            raise ValueError(f"Missing paired records for {baseline} and {candidate} under {scenario}.")
        for metric, specification in REPORT_METRICS.items():
            baseline_matrix = group.pivot(index="seed", columns="year", values=f"{metric}_baseline").loc[seeds, years].to_numpy(float)
            candidate_matrix = group.pivot(index="seed", columns="year", values=f"{metric}_candidate").loc[seeds, years].to_numpy(float)
            improvement = candidate_matrix - baseline_matrix if specification["higher_is_better"] else baseline_matrix - candidate_matrix
            seed_improvement = improvement.mean(axis=1)
            low, high = _hierarchical_bootstrap_interval(improvement, rng)
            baseline_mean = float(baseline_matrix.mean())
            candidate_mean = float(candidate_matrix.mean())
            rows.append(
                {
                    "comparison": comparison,
                    "baseline": baseline,
                    "candidate": candidate,
                    "delay_scenario": scenario,
                    "delay_months": _delay_months(scenario)
                    if scenario.startswith("tau") and scenario[3:].isdigit()
                    else float("nan"),
                    "metric": metric,
                    "metric_label": specification["label"],
                    "unit": specification["unit"],
                    "effect_orientation": "positive values favour candidate",
                    "paired_training_seeds": len(seeds),
                    "held_out_annual_forcing_sequences": len(years),
                    "paired_seed_year_records": len(group),
                    "baseline_mean": baseline_mean,
                    "candidate_mean": candidate_mean,
                    "mean_improvement": float(seed_improvement.mean()),
                    "hierarchical_ci95_low": low,
                    "hierarchical_ci95_high": high,
                    "relative_improvement_pct": 100.0 * float(seed_improvement.mean()) / max(abs(baseline_mean), 1.0e-8) if metric == "cost" else float("nan"),
                    "paired_cohens_dz": _paired_effect_size(seed_improvement),
                    "candidate_seed_wins": int(np.count_nonzero(seed_improvement > 0.0)),
                    "candidate_seed_win_rate": float(np.mean(seed_improvement > 0.0)),
                    "exact_p_raw": _exact_sign_permutation_p(seed_improvement),
                    "ci_method": "two-way hierarchical bootstrap over seeds and held-out annual forcing sequences (10000 draws)",
                    "test_method": "two-sided exact paired sign-randomization test on training-seed means",
                }
            )
    result = pd.DataFrame(rows)
    result["exact_p_holm"] = np.nan
    for _, indexes in result.groupby(["comparison", "metric"]).groups.items():
        indexes = list(indexes)
        result.loc[indexes, "exact_p_holm"] = _holm_adjust(result.loc[indexes, "exact_p_raw"].to_numpy(float))
    result["significance_holm"] = np.select(
        [result["exact_p_holm"] < 0.001, result["exact_p_holm"] < 0.01, result["exact_p_holm"] < 0.05],
        ["***", "**", "*"],
        default="ns",
    )
    return result


def paired_method_effects(
    year_level: pd.DataFrame,
    comparison: str,
    baseline: str,
    candidate: str,
    scenarios: tuple[str, ...],
    bootstrap_seed: int = 20260811,
) -> pd.DataFrame:
    return _paired_metric_effects(year_level, comparison, baseline, candidate, scenarios, bootstrap_seed)


def paired_delay_average_effects(
    year_level: pd.DataFrame,
    comparison: str,
    baseline: str,
    candidate: str,
    scenarios: tuple[str, ...],
    bootstrap_seed: int = 20260813,
) -> pd.DataFrame:
    selected = year_level[
        year_level["method"].isin((baseline, candidate)) & year_level["delay_scenario"].isin(scenarios)
    ].copy()
    averaged = selected.groupby(["method", "variant_key", "seed", "year"], as_index=False)[list(REPORT_METRICS)].mean()
    averaged["delay_scenario"] = "tau1_to_tau5_average"
    result = _paired_metric_effects(
        averaged,
        comparison,
        baseline,
        candidate,
        ("tau1_to_tau5_average",),
        bootstrap_seed,
    )
    result["delay_aggregation"] = "mean of tau1--tau5 within each method/seed/year before inference"
    return result


def paired_seed_differences(seed_level: pd.DataFrame, baseline: str, candidate: str) -> pd.DataFrame:
    keys = ["delay_scenario", "seed"]
    left = seed_level[seed_level["method"] == baseline][keys + list(REPORT_METRICS)]
    right = seed_level[seed_level["method"] == candidate][keys + list(REPORT_METRICS)]
    paired = left.merge(right, on=keys, suffixes=("_baseline", "_candidate"), validate="one_to_one")
    rows: list[dict[str, str | float | int]] = []
    for record in paired.itertuples(index=False):
        for metric, specification in REPORT_METRICS.items():
            baseline_value = float(getattr(record, f"{metric}_baseline"))
            candidate_value = float(getattr(record, f"{metric}_candidate"))
            improvement = candidate_value - baseline_value if specification["higher_is_better"] else baseline_value - candidate_value
            rows.append(
                {
                    "baseline": baseline,
                    "candidate": candidate,
                    "delay_scenario": record.delay_scenario,
                    "delay_months": _delay_months(record.delay_scenario),
                    "seed": int(record.seed),
                    "metric": metric,
                    "metric_label": specification["label"],
                    "unit": specification["unit"],
                    "baseline_value": baseline_value,
                    "candidate_value": candidate_value,
                    "candidate_improvement": improvement,
                    "effect_orientation": "positive values favour candidate",
                }
            )
    return pd.DataFrame(rows)
