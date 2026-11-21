"""Publication figures for the retained WEF delay experiments."""

from __future__ import annotations

from math import ceil, floor
from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/wef-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MultipleLocator, ScalarFormatter
import numpy as np
import pandas as pd


PPO_COLOR = "#0072B2"
GNN_COLOR = "#D55E00"
SECTOR_COLORS = {"water": "#0072B2", "energy": "#CC79A7", "food": "#009E73"}
AVERAGE_WEF_PENALTY_LABEL = r"$P_{\mathrm{avg}}$"
AVERAGE_WEF_PENALTY_CHANGE_LABEL = r"$\Delta P_{\mathrm{avg}}$"
METHOD_STYLE = {
    "PPO": {"color": PPO_COLOR, "marker": "o", "linestyle": "-"},
    "GNN-PPO": {"color": GNN_COLOR, "marker": "s", "linestyle": "--"},
}
SECTOR_LABELS = {"water": "Water", "energy": "Energy", "food": "Food (crop)"}


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
            "font.size": 8.0,
            "axes.titlesize": 8.8,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.75,
            "lines.linewidth": 1.35,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        0.012,
        0.988,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        bbox={"boxstyle": "square,pad=0.05", "facecolor": "white", "edgecolor": "none"},
        zorder=20,
    )


def _bootstrap_mean_ci(values: np.ndarray, seed: int, draws: int = 10_000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(values), size=(draws, len(values)))
    sampled = values[indexes].mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def _hierarchical_mean_ci(
    table: pd.DataFrame, value: str, seed: int, draws: int = 10_000
) -> tuple[float, float, float]:
    matrix = table.pivot(index="seed", columns="year", values=value).to_numpy(float)
    rng = np.random.default_rng(seed)
    seed_draws = rng.integers(0, matrix.shape[0], size=(draws, matrix.shape[0]))
    year_draws = rng.integers(0, matrix.shape[1], size=(draws, matrix.shape[1]))
    sampled = matrix[seed_draws[:, :, None], year_draws[:, None, :]].mean(axis=(1, 2))
    low, high = np.quantile(sampled, [0.025, 0.975])
    return float(matrix.mean()), float(low), float(high)


def _save(fig: plt.Figure, path: Path, vector: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600, facecolor="white")
    if vector:
        fig.savefig(path.with_suffix(".pdf"), facecolor="white")
        fig.savefig(path.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def plot_training_reward_convergence(
    training_history: pd.DataFrame,
    path: Path,
    summary_path: Path | None = None,
    smoothing_window: int = 10,
    rollout_episodes: int = 4,
    episode_steps: int = 12,
) -> pd.DataFrame:
    """Plot a balanced tau0 PPO/GNN-PPO learning curve against environment steps.

    ``mean_episode_reward`` is already the mean return of the annual rollout
    episodes collected at one PPO update.  A trailing rolling mean is applied
    within each independently trained seed.  The plotted mean and bootstrap
    interval use only the longest contiguous update range for which all PPO and
    GNN-PPO seeds are present, preventing early-stopping censoring from changing
    the population of runs part-way through the comparison.
    """

    required = {"update", "mean_episode_reward", "method", "seed"}
    missing = sorted(required.difference(training_history.columns))
    if missing:
        raise ValueError(f"Training history is missing required columns: {missing}")
    if smoothing_window < 1:
        raise ValueError("smoothing_window must be at least one update")
    if rollout_episodes < 1 or episode_steps < 1:
        raise ValueError("rollout_episodes and episode_steps must be positive")

    source = training_history.loc[
        training_history["method"].isin(("PPO", "GNN-PPO")),
        ["update", "mean_episode_reward", "method", "seed"],
    ].copy()
    source["update"] = pd.to_numeric(source["update"], errors="coerce")
    source["mean_episode_reward"] = pd.to_numeric(
        source["mean_episode_reward"], errors="coerce"
    )
    source = source.dropna(subset=["update", "mean_episode_reward", "seed"])
    source["update"] = source["update"].astype(int)
    if set(source["method"].unique()) != {"PPO", "GNN-PPO"}:
        raise ValueError("Training convergence requires both PPO and GNN-PPO histories")
    if source.duplicated(["method", "seed", "update"]).any():
        raise ValueError("Training history contains duplicate method/seed/update rows")

    source = source.sort_values(["method", "seed", "update"]).reset_index(drop=True)
    source["rolling_mean_episode_reward"] = source.groupby(
        ["method", "seed"], sort=False
    )["mean_episode_reward"].transform(
        lambda values: values.rolling(smoothing_window, min_periods=1).mean()
    )
    seed_totals = source.groupby("method")["seed"].nunique()
    counts = source.groupby(["update", "method"])["seed"].nunique().unstack("method")
    full_counts = counts.eq(seed_totals, axis="columns").all(axis=1)
    common_end = 0
    for update in sorted(int(value) for value in counts.index):
        if update != common_end + 1 or not bool(full_counts.loc[update]):
            break
        common_end = update
    if common_end == 0:
        raise ValueError("No common update contains every PPO and GNN-PPO training seed")

    balanced = source[source["update"] <= common_end].copy()
    rows: list[dict[str, float | int | str]] = []
    for ordinal, ((method, update), group) in enumerate(
        balanced.groupby(["method", "update"], sort=True)
    ):
        values = group["rolling_mean_episode_reward"].to_numpy(float)
        mean, low, high = _bootstrap_mean_ci(values, 20261001 + ordinal)
        rows.append(
            {
                "method": method,
                "update": int(update),
                "environment_steps": int(update) * rollout_episodes * episode_steps,
                "mean_episode_reward": mean,
                "ci95_low": low,
                "ci95_high": high,
                "n_training_seeds": int(group["seed"].nunique()),
                "smoothing_window_updates": smoothing_window,
                "environment_steps_per_update": rollout_episodes * episode_steps,
                "observation_delay_months": 0,
            }
        )
    summary = pd.DataFrame(rows).sort_values(["method", "update"]).reset_index(drop=True)

    _apply_style()
    fig, axis = plt.subplots(figsize=(3.50, 2.45))
    plotted_values: list[np.ndarray] = []
    for method in ("PPO", "GNN-PPO"):
        style = METHOD_STYLE[method]
        method_summary = summary[summary["method"] == method]
        steps = method_summary["environment_steps"].to_numpy(float) / 1000.0
        mean = method_summary["mean_episode_reward"].to_numpy(float)
        low = method_summary["ci95_low"].to_numpy(float)
        high = method_summary["ci95_high"].to_numpy(float)
        plotted_values.extend([low, high])
        axis.fill_between(
            steps,
            low,
            high,
            color=style["color"],
            alpha=0.16,
            lw=0,
            zorder=1,
        )
        axis.plot(
            steps,
            mean,
            color=style["color"],
            linestyle=style["linestyle"],
            lw=1.45,
            label=method,
            zorder=3,
        )

    values = np.concatenate(plotted_values)
    lower, upper = float(values.min()), float(values.max())
    margin = max(0.08, 0.08 * (upper - lower))
    maximum_steps = common_end * rollout_episodes * episode_steps / 1000.0
    axis.set(
        xlabel=r"Environment steps ($\times 10^3$)",
        ylabel="Mean episode return",
        xlim=(0, maximum_steps),
        ylim=(lower - margin, upper + margin),
    )
    axis.tick_params(direction="out", length=3.0, width=0.75, labelsize=6.5)
    axis.xaxis.label.set_size(7.0)
    axis.yaxis.label.set_size(7.0)
    axis.legend(loc="upper left", frameon=False, fontsize=6.5, handlelength=2.5)
    fig.subplots_adjust(left=0.19, right=0.985, top=0.975, bottom=0.19)
    _save(fig, path, vector=True)
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
    return summary


def plot_delay_reward_learning_curves(
    reward_history: pd.DataFrame,
    path: Path,
    summary_path: Path | None = None,
    smoothing_window: int = 1,
) -> pd.DataFrame:
    """Plot dense, unsmoothed PPO/GNN-PPO returns at tau0 and tau5."""

    required = {
        "method",
        "seed",
        "update",
        "episodes",
        "delay_scenario",
        "mean_episode_return",
    }
    missing = sorted(required.difference(reward_history.columns))
    if missing:
        raise ValueError(f"Delay-reward history is missing required columns: {missing}")
    if smoothing_window < 1:
        raise ValueError("smoothing_window must be at least one update")

    source = reward_history.copy()
    source = source[
        source["method"].isin(("PPO", "GNN-PPO"))
        & source["delay_scenario"].isin(("tau0", "tau5"))
    ].copy()
    source["update"] = pd.to_numeric(source["update"], errors="coerce")
    source["episodes"] = pd.to_numeric(source["episodes"], errors="coerce")
    source["mean_episode_return"] = pd.to_numeric(
        source["mean_episode_return"], errors="coerce"
    )
    source = source.dropna(
        subset=["method", "seed", "update", "episodes", "delay_scenario", "mean_episode_return"]
    )
    if source.duplicated(["method", "seed", "update", "delay_scenario"]).any():
        raise ValueError("Delay-reward history contains duplicate method/seed/update/scenario rows")
    expected = {
        ("PPO", "tau0"),
        ("PPO", "tau5"),
        ("GNN-PPO", "tau0"),
        ("GNN-PPO", "tau5"),
    }
    observed = set(zip(source["method"], source["delay_scenario"]))
    if observed != expected:
        raise ValueError(f"Expected four method-delay curves, found {sorted(observed)}")

    source = source.sort_values(
        ["method", "delay_scenario", "seed", "update"]
    ).reset_index(drop=True)
    source["smoothed_episode_return"] = source.groupby(
        ["method", "delay_scenario", "seed"], sort=False
    )["mean_episode_return"].transform(
        lambda values: values.rolling(smoothing_window, min_periods=1).mean()
    )
    summary = (
        source.groupby(
            ["method", "delay_scenario", "update", "episodes"], as_index=False
        )
        .agg(
            raw_mean_episode_return=("mean_episode_return", "mean"),
            smoothed_mean_episode_return=("smoothed_episode_return", "mean"),
            sd_smoothed_episode_return=("smoothed_episode_return", "std"),
            n_training_seeds=("seed", "nunique"),
        )
        .sort_values(["method", "delay_scenario", "update"])
        .reset_index(drop=True)
    )
    summary["delay_months"] = summary["delay_scenario"].str.removeprefix("tau").astype(int)
    summary["smoothing_window_updates"] = smoothing_window

    # Colour encodes architecture and line style encodes information latency.
    # The redundant encoding remains legible in grayscale and at journal scale.
    curve_styles = {
        ("PPO", "tau0"): {
            "label": r"PPO, $\tau=0$",
            "color": PPO_COLOR,
            "linestyle": "-",
        },
        ("PPO", "tau5"): {
            "label": r"PPO, $\tau=5$",
            "color": PPO_COLOR,
            "linestyle": (0, (4.0, 2.2)),
        },
        ("GNN-PPO", "tau0"): {
            "label": r"GNN-PPO, $\tau=0$",
            "color": GNN_COLOR,
            "linestyle": "-",
        },
        ("GNN-PPO", "tau5"): {
            "label": r"GNN-PPO, $\tau=5$",
            "color": GNN_COLOR,
            "linestyle": (0, (4.0, 2.2)),
        },
    }
    _apply_style()
    fig, axis = plt.subplots(figsize=(7.0, 4.05))
    plotted: list[np.ndarray] = []
    for key in (
        ("PPO", "tau5"),
        ("GNN-PPO", "tau5"),
        ("PPO", "tau0"),
        ("GNN-PPO", "tau0"),
    ):
        method, delay_scenario = key
        style = curve_styles[key]
        curve = summary[
            (summary["method"] == method)
            & (summary["delay_scenario"] == delay_scenario)
        ]
        episodes = curve["episodes"].to_numpy(float)
        raw = curve["raw_mean_episode_return"].to_numpy(float)
        plotted.append(raw)
        axis.plot(
            episodes,
            raw,
            color=style["color"],
            linestyle=style["linestyle"],
            lw=1.40,
            alpha=1.0,
            solid_capstyle="round",
            dash_capstyle="round",
            label=style["label"],
            zorder=2,
        )

    values = np.concatenate(plotted)
    lower, upper = float(values.min()), float(values.max())
    margin = max(0.04, 0.035 * (upper - lower))
    maximum_episodes = int(summary["episodes"].max())
    axis.set(
        xlabel="Training episodes",
        ylabel="Mean validation episode return",
        xlim=(0, maximum_episodes),
        ylim=(lower - margin, upper + margin),
    )
    axis.xaxis.set_major_locator(MultipleLocator(100))
    axis.yaxis.set_major_locator(MultipleLocator(0.5))
    axis.spines["top"].set_visible(True)
    axis.spines["right"].set_visible(True)
    axis.tick_params(direction="out", length=3.4, width=0.8, labelsize=7.4)
    axis.xaxis.label.set_size(8.4)
    axis.yaxis.label.set_size(8.4)
    handles, labels = axis.get_legend_handles_labels()
    handles_by_label = dict(zip(labels, handles))
    legend_keys = (
        ("PPO", "tau0"),
        ("PPO", "tau5"),
        ("GNN-PPO", "tau0"),
        ("GNN-PPO", "tau5"),
    )
    legend_labels = [curve_styles[key]["label"] for key in legend_keys]
    axis.legend(
        [handles_by_label[label] for label in legend_labels],
        legend_labels,
        loc="lower right",
        frameon=False,
        fontsize=7.8,
        handlelength=2.85,
        borderpad=0.15,
        labelspacing=0.36,
    )
    fig.subplots_adjust(left=0.135, right=0.985, top=0.98, bottom=0.165)
    _save(fig, path, vector=False)
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
    return summary


def plot_gnnppo_uniform_delay_learning_curves(
    reward_history: pd.DataFrame,
    path: Path,
    summary_path: Path | None = None,
) -> pd.DataFrame:
    """Plot GNN-PPO validation returns for uniform delays from tau0 to tau5."""

    required = {
        "method",
        "seed",
        "update",
        "episodes",
        "delay_scenario",
        "mean_episode_return",
    }
    missing = sorted(required.difference(reward_history.columns))
    if missing:
        raise ValueError(f"GNN-PPO delay history is missing required columns: {missing}")

    scenarios = tuple(f"tau{delay}" for delay in range(6))
    source = reward_history.loc[
        (reward_history["method"] == "GNN-PPO")
        & reward_history["delay_scenario"].isin(scenarios)
    ].copy()
    for column in ("update", "episodes", "mean_episode_return"):
        source[column] = pd.to_numeric(source[column], errors="coerce")
    source = source.dropna(
        subset=["seed", "update", "episodes", "delay_scenario", "mean_episode_return"]
    )
    if source.duplicated(["seed", "update", "delay_scenario"]).any():
        raise ValueError("GNN-PPO delay history contains duplicate seed/update/scenario rows")
    observed = set(source["delay_scenario"])
    if observed != set(scenarios):
        raise ValueError(f"Expected delay scenarios {scenarios}, found {sorted(observed)}")

    source = source.sort_values(["delay_scenario", "seed", "update"])
    summary = (
        source.groupby(["delay_scenario", "update", "episodes"], as_index=False)
        .agg(
            mean_episode_return=("mean_episode_return", "mean"),
            sd_episode_return=("mean_episode_return", "std"),
            n_training_seeds=("seed", "nunique"),
        )
        .sort_values(["delay_scenario", "update"])
        .reset_index(drop=True)
    )
    summary["delay_months"] = (
        summary["delay_scenario"].str.removeprefix("tau").astype(int)
    )

    # A colour-blind-safe categorical palette retains distinction among the six
    # ordered delay conditions without introducing an uncertainty band.
    delay_colours = {
        "tau0": "#0072B2",
        "tau1": "#56B4E9",
        "tau2": "#009E73",
        "tau3": "#E69F00",
        "tau4": "#D55E00",
        "tau5": "#CC79A7",
    }
    _apply_style()
    fig, axis = plt.subplots(figsize=(7.0, 4.05))
    plotted: list[np.ndarray] = []
    for delay_scenario in scenarios:
        curve = summary[summary["delay_scenario"] == delay_scenario]
        episodes = curve["episodes"].to_numpy(float)
        values = curve["mean_episode_return"].to_numpy(float)
        plotted.append(values)
        delay = int(delay_scenario.removeprefix("tau"))
        axis.plot(
            episodes,
            values,
            color=delay_colours[delay_scenario],
            lw=1.35,
            solid_capstyle="round",
            label=rf"$\tau={delay}$",
            zorder=2 + delay,
        )

    values = np.concatenate(plotted)
    lower, upper = float(values.min()), float(values.max())
    margin = max(0.04, 0.035 * (upper - lower))
    maximum_episodes = int(summary["episodes"].max())
    axis.set(
        xlabel="Training episodes",
        ylabel="Mean validation episode return",
        xlim=(0, maximum_episodes),
        ylim=(lower - margin, upper + margin),
    )
    axis.xaxis.set_major_locator(MultipleLocator(100))
    axis.yaxis.set_major_locator(MultipleLocator(0.5))
    axis.spines["top"].set_visible(True)
    axis.spines["right"].set_visible(True)
    axis.tick_params(direction="out", length=3.4, width=0.8, labelsize=7.4)
    axis.xaxis.label.set_size(8.4)
    axis.yaxis.label.set_size(8.4)
    axis.legend(
        loc="lower right",
        ncol=2,
        frameon=False,
        fontsize=7.4,
        handlelength=2.65,
        columnspacing=1.15,
        borderpad=0.1,
        labelspacing=0.38,
    )
    fig.subplots_adjust(left=0.135, right=0.985, top=0.98, bottom=0.165)
    _save(fig, path, vector=False)
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
    return summary


def plot_current_information_figure(current_year: pd.DataFrame, path: Path) -> None:
    """Experiment 1: preserve every paired seed under the no-delay condition."""

    _apply_style()
    value = "cost"
    by_seed = current_year.groupby(["method", "seed"], as_index=False)[value].mean()
    wide = by_seed.pivot(index="seed", columns="method", values=value).dropna()
    if set(("PPO", "GNN-PPO")).difference(wide.columns):
        raise ValueError("Experiment 1 figure requires PPO and GNN-PPO seed outcomes.")
    differences = wide["GNN-PPO"] - wide["PPO"]
    by_seed_year = current_year.pivot(
        index=["seed", "year"], columns="method", values=value
    ).dropna()
    difference_frame = by_seed_year.reset_index()
    difference_frame["cost_difference"] = (
        difference_frame["GNN-PPO"] - difference_frame["PPO"]
    )
    mean_diff, low_diff, high_diff = _hierarchical_mean_ci(
        difference_frame, "cost_difference", 20260811
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.95), gridspec_kw={"wspace": 0.36})
    for _, row in wide.iterrows():
        axes[0].plot([0, 1], [row["PPO"], row["GNN-PPO"]], color="#BDBDBD", lw=0.75, zorder=1)
    for position, method in enumerate(("PPO", "GNN-PPO")):
        style = METHOD_STYLE[method]
        values = wide[method].to_numpy(float)
        jitter = np.linspace(-0.09, 0.09, len(values))
        axes[0].scatter(
            np.full(len(values), position) + jitter,
            values,
            s=30,
            marker=style["marker"],
            color=style["color"],
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        mean, low, high = _bootstrap_mean_ci(values, 20260832 + position)
        axes[0].errorbar(position, mean, yerr=[[mean - low], [high - mean]], fmt="D", color="#222222", capsize=2.5, markersize=4.2, lw=0.9, zorder=5)
    axes[0].set(
        title="Average WEF penalty with no delayed observations",
        ylabel="Average WEF penalty (-mean reward)\n(lower is better)",
        xticks=[0, 1],
        xticklabels=["PPO", "GNN-PPO"],
        xlim=(-0.32, 1.32),
    )
    axes[0].grid(axis="y", color="#D9D9D9", lw=0.45)

    y = np.arange(len(differences))
    axes[1].axvline(0, color="#555555", lw=0.8, zorder=0)
    axes[1].scatter(differences, y, color="#4D4D4D", s=26, edgecolor="white", linewidth=0.4, zorder=3)
    axes[1].errorbar(mean_diff, len(y) + 0.35, xerr=[[mean_diff - low_diff], [high_diff - mean_diff]], fmt="D", color="#222222", capsize=2.5, markersize=4.3, lw=0.9, zorder=5)
    axes[1].set(
        title="Paired GNN-PPO minus PPO average WEF penalty differences",
        xlabel="Difference in average WEF penalty (negative favours GNN-PPO)",
        yticks=[],
        ylim=(-0.8, len(y) + 1.1),
    )
    axes[1].grid(axis="x", color="#D9D9D9", lw=0.45)
    axes[1].text(
        0.98,
        0.04,
        f"Mean difference = {mean_diff:+.4f}\n95% seed/year bootstrap CI [{low_diff:+.4f}, {high_diff:+.4f}]\nn = {len(differences)} training seeds",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=6.65,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#F4F4F4", "edgecolor": "none"},
    )
    for axis, label in zip(axes, "ab"):
        _panel_label(axis, label)
    fig.suptitle("No-delay performance of PPO and GNN-PPO", y=0.99, fontsize=10.2, fontweight="bold")
    fig.text(
        0.5,
        0.035,
        "Each point is one independently trained seed averaged across the 2019–2021 held-out forcing sequences.\n"
        "Diamonds and bars show means and 95% confidence intervals; the paired difference uses a two-way seed/year bootstrap.",
        ha="center",
        va="bottom",
        fontsize=6.2,
        linespacing=1.12,
    )
    fig.subplots_adjust(left=0.10, right=0.985, top=0.80, bottom=0.23)
    _save(fig, path)


def plot_uniform_delay_figure(all_year: pd.DataFrame, path: Path) -> None:
    """Experiment 2: mean-reward trends integrated with an aligned value table."""

    required = {"method", "delay_scenario", "cost"}
    missing = sorted(required.difference(all_year.columns))
    if missing:
        raise ValueError(f"Uniform-delay data are missing required columns: {missing}")

    source = all_year.loc[
        all_year["method"].isin(("PPO", "GNN-PPO")),
        ["method", "delay_scenario", "cost"],
    ].copy()
    source["delay_months"] = pd.to_numeric(
        source["delay_scenario"].str.removeprefix("tau"), errors="coerce"
    )
    source["cost"] = pd.to_numeric(source["cost"], errors="coerce")
    source = source.dropna(subset=["delay_months", "cost"])
    source["delay_months"] = source["delay_months"].astype(int)
    source = source[source["delay_months"].between(0, 5)]

    mean_reward = (
        source.assign(reward=-source["cost"])
        .groupby(["method", "delay_months"], as_index=False)["reward"]
        .mean()
        .sort_values(["method", "delay_months"])
    )
    expected_delays = set(range(6))
    for method in ("PPO", "GNN-PPO"):
        observed_delays = set(
            mean_reward.loc[mean_reward["method"] == method, "delay_months"]
        )
        if observed_delays != expected_delays:
            raise ValueError(
                f"{method} must contain exactly delays 0-5; found {sorted(observed_delays)}"
            )

    curves = {
        method: mean_reward.loc[mean_reward["method"] == method, "reward"].to_numpy(float)
        for method in ("PPO", "GNN-PPO")
    }
    delays = np.arange(6)
    reward_difference = curves["GNN-PPO"] - curves["PPO"]

    _apply_style()
    fig = plt.figure(figsize=(6.6, 4.15))
    grid = fig.add_gridspec(2, 1, height_ratios=(3.15, 1.0), hspace=0.34)
    axis = fig.add_subplot(grid[0])
    table_axis = fig.add_subplot(grid[1])

    for method in ("PPO", "GNN-PPO"):
        style = METHOD_STYLE[method]
        axis.plot(
            delays,
            curves[method],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=5.0,
            markeredgecolor="white",
            markeredgewidth=0.65,
            lw=1.85,
            solid_capstyle="round",
            label=method,
            zorder=3,
        )

    values = np.concatenate(list(curves.values()))
    lower, upper = float(values.min()), float(values.max())
    margin = max(0.004, 0.12 * (upper - lower))
    axis.set(
        xlabel="Delayed observation(months)",
        ylabel="Reward",
        xlim=(-0.12, 5.12),
        ylim=(lower - margin, upper + margin),
        xticks=range(6),
    )
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
        spine.set_linewidth(0.85)
    axis.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        length=3.6,
        width=0.8,
        color="#333333",
        pad=3.5,
    )
    axis.legend(
        loc="center right",
        bbox_to_anchor=(0.985, 0.56),
        ncol=1,
        frameon=True,
        framealpha=0.94,
        facecolor="white",
        edgecolor="#BDBDBD",
        handlelength=2.4,
        borderpad=0.45,
        labelspacing=0.45,
    )

    table_axis.set_axis_off()
    table_rows = [
        ["PPO", *[f"{value:.4f}" for value in curves["PPO"]]],
        ["GNN-PPO", *[f"{value:.4f}" for value in curves["GNN-PPO"]]],
        [r"$\Delta$ (GNN$-$PPO)", *[f"{value:+.4f}" for value in reward_difference]],
    ]
    table = table_axis.table(
        cellText=table_rows,
        colLabels=["Method", *[str(value) for value in delays]],
        colLoc="center",
        cellLoc="center",
        colWidths=[0.20, *([0.133] * 6)],
        bbox=[0.0, 0.20, 1.0, 0.76],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.9)
    row_label_colors = {1: PPO_COLOR, 2: GNN_COLOR, 3: "#333333"}
    for (row, column), cell in table.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("#8A8A8A")
        cell.PAD = 0.05
        if row == 0:
            cell.visible_edges = "TB"
            cell.set_linewidth(0.75)
            cell.get_text().set_color("#333333")
            cell.get_text().set_fontweight("bold")
        else:
            cell.visible_edges = "" if row == 1 else "B"
            cell.set_linewidth(0.45 if row == 2 else 0.75)
            if column == 0:
                cell.get_text().set_color(row_label_colors[row])
                cell.get_text().set_fontweight("bold")
                cell.get_text().set_ha("left")
            elif row == 2:
                cell.get_text().set_fontweight("bold")
            elif row == 3:
                cell.get_text().set_color("#4D4D4D")
    table_axis.text(
        0.5,
        0.035,
        r"Positive $\Delta$ values indicate a higher observed mean reward for GNN-PPO.",
        transform=table_axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.2,
        color="#606060",
    )

    fig.subplots_adjust(left=0.13, right=0.98, top=0.95, bottom=0.075)
    _save(fig, path)


def plot_sector_delay_risk_figure(
    sector_effects: pd.DataFrame,
    sector_ranking: pd.DataFrame,
    path: Path,
    policy_label: str = "GNN-PPO",
    experiment_number: int = 3,
) -> None:
    """Show all sector-delay estimates in one shared-scale forest plot.

    Delays are ordered conditions rather than a fitted dose-response trend, so
    the estimates are not connected.  Colour and marker shape redundantly encode
    the delayed sector, while the common zero reference makes the uncertainty
    intervals directly comparable.
    """

    _apply_style()
    # Retain the established call signature used by both policy experiments.
    del sector_ranking, policy_label, experiment_number
    required_effect_columns = {
        "delayed_sector",
        "delay_months",
        "outcome",
        "mean_change",
        "ci95_low",
        "ci95_high",
    }
    missing_effect_columns = sorted(required_effect_columns.difference(sector_effects.columns))
    if missing_effect_columns:
        raise ValueError(f"Sector effects are missing required columns: {missing_effect_columns}")

    cost = sector_effects[sector_effects["outcome"] == "cost"].copy()
    cost = cost[cost["delayed_sector"].isin(("water", "energy", "food"))]
    cost = cost.sort_values(["delayed_sector", "delay_months"]).reset_index(drop=True)
    expected_conditions = {(sector, delay) for sector in ("water", "energy", "food") for delay in range(1, 6)}
    observed_conditions = set(zip(cost["delayed_sector"], cost["delay_months"].astype(int)))
    if observed_conditions != expected_conditions:
        missing_conditions = sorted(expected_conditions.difference(observed_conditions))
        extra_conditions = sorted(observed_conditions.difference(expected_conditions))
        raise ValueError(
            "Sector-delay figure requires exactly the 15 sector-by-delay conditions; "
            f"missing={missing_conditions}, extra={extra_conditions}"
        )
    if cost.duplicated(["delayed_sector", "delay_months"]).any():
        raise ValueError("Sector-delay figure contains duplicate sector-by-delay estimates")

    estimate_values = cost[["mean_change", "ci95_low", "ci95_high"]].to_numpy(float)
    if not np.isfinite(estimate_values).all():
        raise ValueError("Sector-delay estimates and confidence limits must all be finite")
    if ((cost["ci95_low"] > cost["mean_change"]) | (cost["mean_change"] > cost["ci95_high"])).any():
        raise ValueError("Every sector-delay mean must lie within its confidence interval")

    all_interval_values = cost[["ci95_low", "ci95_high"]].to_numpy(float)
    max_interval = float(np.nanmax(np.abs(all_interval_values)))
    if not np.isfinite(max_interval) or max_interval <= 0:
        raise ValueError("Sector-delay confidence intervals must contain finite non-zero values")

    # Express the change in the average WEF penalty as Delta P_avg and retain
    # raw values on the ticks. A symmetric, data-adaptive scale prevents either
    # direction from receiving more visual weight.
    raw_tick_step = max_interval / 4.0
    tick_exponent = int(floor(np.log10(raw_tick_step)))
    normalized_step = raw_tick_step / (10.0**tick_exponent)
    if normalized_step <= 1.0:
        nice_step = 1.0
    elif normalized_step <= 2.0:
        nice_step = 2.0
    elif normalized_step <= 5.0:
        nice_step = 5.0
    else:
        nice_step = 10.0
    tick_step = nice_step * (10.0**tick_exponent)
    half_step = tick_step / 2.0
    horizontal_limit = ceil(max_interval * 1.08 / half_step) * half_step
    decimal_places = max(0, -int(floor(np.log10(tick_step))))

    sector_order = ("water", "energy", "food")
    markers = {"water": "o", "energy": "s", "food": "D"}
    vertical_offsets = {"water": -0.20, "energy": 0.0, "food": 0.20}
    # Build at the final single-column size so text is not reduced by LaTeX.
    fig, axis = plt.subplots(figsize=(3.50, 3.25))

    for delay in range(1, 6):
        if delay % 2 == 1:
            axis.axhspan(delay - 0.48, delay + 0.48, color="#F7F8FA", zorder=0)
    axis.axvline(0.0, color="#343A40", lw=0.95, zorder=1)

    for sector in sector_order:
        subset = cost[cost["delayed_sector"] == sector].sort_values("delay_months")
        delays = subset["delay_months"].to_numpy(int)
        mean = subset["mean_change"].to_numpy(float)
        low = subset["ci95_low"].to_numpy(float)
        high = subset["ci95_high"].to_numpy(float)
        y_positions = delays + vertical_offsets[sector]
        axis.errorbar(
            mean,
            y_positions,
            xerr=np.vstack([mean - low, high - mean]),
            fmt=markers[sector],
            color=SECTOR_COLORS[sector],
            ecolor=SECTOR_COLORS[sector],
            markersize=5.8,
            markerfacecolor=SECTOR_COLORS[sector],
            markeredgecolor="white",
            markeredgewidth=0.65,
            elinewidth=1.15,
            capsize=2.8,
            capthick=0.9,
            zorder=3,
            label=SECTOR_LABELS[sector],
        )

    axis.set_xlim(-horizontal_limit, horizontal_limit)
    axis.set_ylim(5.55, 0.45)
    axis.set_yticks(range(1, 6))
    axis.set_xlabel(AVERAGE_WEF_PENALTY_CHANGE_LABEL, labelpad=7, fontsize=9.0)
    axis.set_ylabel("Observation delay (months)", labelpad=8)
    axis.xaxis.set_major_locator(MultipleLocator(tick_step))
    axis.xaxis.set_major_formatter(f"{{x:.{decimal_places}f}}")
    axis.grid(axis="x", color="#E1E5E8", lw=0.50, zorder=0)
    axis.spines["left"].set_visible(False)
    axis.spines["bottom"].set_color("#343A40")
    axis.tick_params(axis="x", direction="out", length=3.0, width=0.7, pad=3)
    axis.tick_params(axis="y", length=0, pad=7)
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.025),
        ncol=3,
        frameon=False,
        fontsize=6.6,
        handlelength=1.3,
        columnspacing=0.9,
        handletextpad=0.4,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(left=0.22, right=0.94, top=0.82, bottom=0.19)
    _save(fig, path, vector=True)


def plot_cascade_heatmap(
    cascade_effects: pd.DataFrame,
    path: Path,
    policy_label: str = "GNN-PPO",
    experiment_number: int = 3,
) -> None:
    """Show how a delayed source sector redistributes normalized WEF penalties."""

    _apply_style()
    # The caller retains these arguments because the same plotting entry point
    # is used for both fixed-policy experiments; the figure itself deliberately
    # avoids an experiment-specific title.
    del policy_label, experiment_number
    required_columns = {
        "delayed_sector",
        "delay_months",
        "outcome",
        "mean_change",
        "ci95_low",
        "ci95_high",
    }
    missing_columns = sorted(required_columns.difference(cascade_effects.columns))
    if missing_columns:
        raise ValueError(f"Cascade effects are missing required columns: {missing_columns}")

    outcome_order = ("water_cost", "energy_cost", "food_cost")
    outcome_labels = ("Water", "Energy", "Food")
    sector_order = ("water", "energy", "food")
    source_labels = {"water": "Water", "energy": "Energy", "food": "Food"}
    cascade = cascade_effects[
        cascade_effects["delayed_sector"].isin(sector_order)
        & cascade_effects["outcome"].isin(outcome_order)
    ].copy()
    expected_conditions = {
        (sector, delay, outcome)
        for sector in sector_order
        for delay in range(1, 6)
        for outcome in outcome_order
    }
    observed_conditions = set(
        zip(
            cascade["delayed_sector"],
            cascade["delay_months"].astype(int),
            cascade["outcome"],
        )
    )
    if observed_conditions != expected_conditions:
        missing_conditions = sorted(expected_conditions.difference(observed_conditions))
        extra_conditions = sorted(observed_conditions.difference(expected_conditions))
        raise ValueError(
            "Cascade heatmap requires exactly the 45 source-delay-outcome conditions; "
            f"missing={missing_conditions}, extra={extra_conditions}"
        )
    if cascade.duplicated(["delayed_sector", "delay_months", "outcome"]).any():
        raise ValueError("Cascade heatmap contains duplicate source-delay-outcome estimates")
    estimate_values = cascade[["mean_change", "ci95_low", "ci95_high"]].to_numpy(float)
    if not np.isfinite(estimate_values).all():
        raise ValueError("Cascade estimates and confidence limits must all be finite")
    if (
        (cascade["ci95_low"] > cascade["mean_change"])
        | (cascade["mean_change"] > cascade["ci95_high"])
    ).any():
        raise ValueError("Every cascade mean must lie within its confidence interval")

    # Colour represents the cell means, so its range is based on those means
    # rather than the wider confidence limits.  A shared power-of-ten unit
    # preserves small structured changes without printing rows of 0.000.
    max_abs = float(np.max(np.abs(cascade["mean_change"].to_numpy(float))))
    if max_abs <= 0:
        raise ValueError("Cascade means must contain at least one non-zero value")
    scale_exponent = int(floor(np.log10(max_abs)))
    scale_factor = 10.0 ** (-scale_exponent)
    scaled_max = max_abs * scale_factor
    display_limit = ceil(scaled_max * 5.0 - 1.0e-12) / 5.0
    norm = TwoSlopeNorm(vmin=-display_limit, vcenter=0.0, vmax=display_limit)

    # Build at the final two-column width so LaTeX does not reduce text size.
    fig = plt.figure(figsize=(7.20, 2.50))
    grid = fig.add_gridspec(
        1,
        4,
        width_ratios=(1.0, 1.0, 1.0, 0.055),
        left=0.080,
        right=0.900,
        top=0.835,
        bottom=0.185,
        wspace=0.17,
    )
    axes: list[plt.Axes] = []
    image = None
    for panel_index, sector in enumerate(sector_order):
        axis = fig.add_subplot(grid[0, panel_index], sharey=axes[0] if axes else None)
        axes.append(axis)
        subset = cascade[cascade["delayed_sector"] == sector]
        matrix = subset.pivot(
            index="delay_months", columns="outcome", values="mean_change"
        ).reindex(index=range(1, 6), columns=outcome_order)
        display_matrix = matrix.to_numpy(float) * scale_factor
        image = axis.pcolormesh(
            np.arange(4) - 0.5,
            np.arange(6) - 0.5,
            display_matrix,
            cmap="RdBu_r",
            norm=norm,
            shading="flat",
            edgecolors="white",
            linewidth=1.0,
            antialiased=True,
        )
        for row_index in range(5):
            for column_index in range(3):
                value = display_matrix[row_index, column_index]
                label = "0.00" if abs(value) < 0.005 else f"{value:+.2f}"
                text_color = "white" if abs(value) >= 0.58 * display_limit else "#202124"
                axis.text(
                    column_index,
                    row_index,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7.0,
                    color=text_color,
                    fontweight="medium",
                )
        axis.set_xticks(range(3), labels=outcome_labels)
        axis.set_yticks(range(5), labels=range(1, 6))
        axis.set_xlim(-0.5, 2.5)
        axis.set_ylim(4.5, -0.5)
        axis.set_title(f"{source_labels[sector]} delayed", pad=8, fontsize=8.7)
        axis.tick_params(axis="both", which="major", length=0, pad=4)
        for spine in axis.spines.values():
            spine.set_visible(False)
        if panel_index > 0:
            axis.tick_params(axis="y", labelleft=False)
        axis.text(
            -0.055,
            1.075,
            "abc"[panel_index],
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=9.2,
            fontweight="bold",
            clip_on=False,
        )

    axes[0].set_ylabel("Observation delay (months)", labelpad=8)
    color_axis = fig.add_subplot(grid[0, 3])
    colorbar = fig.colorbar(
        image,
        cax=color_axis,
        ticks=np.linspace(-display_limit, display_limit, 5),
        format="%.1f",
    )
    # Rasterize only the continuous gradient at 600 dpi to avoid hairline seams
    # between hundreds of vector strips; cells, labels, and annotations remain
    # editable vector elements in PDF/SVG.
    colorbar.solids.set_rasterized(True)
    colorbar.outline.set_linewidth(0.65)
    colorbar.ax.tick_params(labelsize=6.8, width=0.65, length=2.8, pad=3)
    colorbar.set_label(
        rf"Average penalty-component change ($\times 10^{{{scale_exponent}}}$)",
        fontsize=7.2,
        labelpad=8,
    )
    _save(fig, path, vector=True)


def plot_node_vulnerability_figure(
    node_ranking: pd.DataFrame,
    path: Path,
    policy_label: str = "GNN-PPO",
    experiment_number: int = 3,
) -> None:
    """Show the exploratory node-delay screen at overview and detail scales."""

    _apply_style()
    required = {"delayed_node", "node_sector", "mean_change", "ci95_low", "ci95_high"}
    missing = required.difference(node_ranking.columns)
    if missing:
        raise ValueError(f"Node ranking is missing required columns: {sorted(missing)}")

    ranking = node_ranking.sort_values("mean_change", ascending=True).reset_index(drop=True)
    if ranking["delayed_node"].duplicated().any():
        raise ValueError("Each delayed node must appear exactly once in the vulnerability ranking.")
    unknown_sectors = set(ranking["node_sector"]).difference(SECTOR_COLORS)
    if unknown_sectors:
        raise ValueError(f"Unknown node sectors: {sorted(unknown_sectors)}")

    positions = np.arange(len(ranking))
    values = ranking["mean_change"].to_numpy(float)
    low = ranking["ci95_low"].to_numpy(float)
    high = ranking["ci95_high"].to_numpy(float)
    if not np.isfinite(np.concatenate([values, low, high])).all():
        raise ValueError("Node vulnerability estimates and intervals must be finite.")
    if np.any(low > values) or np.any(values > high):
        raise ValueError("Every confidence interval must contain its mean estimate.")

    sector_markers = {"water": "o", "energy": "s", "food": "D"}
    sector_labels = {"water": "Water", "energy": "Energy", "food": "Food"}
    focal_node = "W8" if ranking["delayed_node"].eq("W8").any() else ranking.iloc[-1]["delayed_node"]
    focal_colour = "#B33C2E"
    scale = 1_000.0
    scaled_values = values * scale
    scaled_low = low * scale
    scaled_high = high * scale

    def _x_limits(interval_low: np.ndarray, interval_high: np.ndarray) -> tuple[float, float]:
        left = min(0.0, float(interval_low.min()))
        right = max(0.0, float(interval_high.max()))
        span = max(right - left, 0.1)
        return left - 0.055 * span, right + 0.055 * span

    def _format_effect_axis(axis: plt.Axes, left: float, right: float) -> None:
        axis.set_xlim(left, right)
        axis.axvspan(0, right, color="#FCEDE8", alpha=0.42, zorder=0)
        axis.axvline(0, color="#343A40", lw=0.85, zorder=1)
        axis.set_axisbelow(True)
        axis.grid(axis="x", color="#D8DCE0", lw=0.5)
        axis.tick_params(axis="x", width=0.65, length=3.0, pad=2.5)
        axis.tick_params(axis="y", width=0, length=0, pad=4)
        axis.locator_params(axis="x", nbins=5)
        axis.spines["left"].set_visible(False)
        axis.set_xlabel(r"$\Delta P_{\mathrm{avg}}$ ($\times 10^{-3}$)", labelpad=4)

    def _panel_tag(axis: plt.Axes, label: str) -> None:
        axis.text(
            -0.12,
            1.035,
            label,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            clip_on=False,
        )

    fig = plt.figure(figsize=(7.20, 5.35))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.50, 1.0),
        height_ratios=(1.0, 1.0),
        left=0.078,
        right=0.985,
        bottom=0.135,
        top=0.895,
        wspace=0.34,
        hspace=0.43,
    )
    overview_axis = fig.add_subplot(grid[:, 0])
    matrix_grid = grid[0, 1].subgridspec(2, 1, height_ratios=(1.0, 0.10), hspace=0.26)
    matrix_axis = fig.add_subplot(matrix_grid[0, 0])
    colour_axis = fig.add_subplot(matrix_grid[1, 0])
    detail_axis = fig.add_subplot(grid[1, 1])

    focal_mask = ranking["delayed_node"].eq(focal_node).to_numpy()
    for row in positions[::2]:
        overview_axis.axhspan(row - 0.5, row + 0.5, color="#F5F6F7", zorder=-1)
    if focal_mask.any():
        focal_position = int(positions[focal_mask][0])
        overview_axis.axhspan(
            focal_position - 0.48,
            focal_position + 0.48,
            color="#FBE4DC",
            zorder=0,
        )
    overview_axis.errorbar(
        scaled_values,
        positions,
        xerr=np.vstack([scaled_values - scaled_low, scaled_high - scaled_values]),
        fmt="none",
        ecolor="#6B737A",
        elinewidth=0.78,
        capsize=1.9,
        capthick=0.78,
        zorder=2,
    )
    for sector in ("water", "energy", "food"):
        selected = ranking["node_sector"].eq(sector).to_numpy() & ~focal_mask
        overview_axis.scatter(
            scaled_values[selected],
            positions[selected],
            marker=sector_markers[sector],
            s=36,
            facecolor=SECTOR_COLORS[sector],
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )
    if focal_mask.any():
        overview_axis.errorbar(
            scaled_values[focal_mask],
            positions[focal_mask],
            xerr=np.vstack(
                [
                    scaled_values[focal_mask] - scaled_low[focal_mask],
                    scaled_high[focal_mask] - scaled_values[focal_mask],
                ]
            ),
            fmt="none",
            ecolor=focal_colour,
            elinewidth=1.35,
            capsize=2.6,
            capthick=1.2,
            zorder=4,
        )
        overview_axis.scatter(
            scaled_values[focal_mask],
            positions[focal_mask],
            marker=sector_markers[str(ranking.loc[focal_mask, "node_sector"].iloc[0])],
            s=76,
            facecolor=SECTOR_COLORS[str(ranking.loc[focal_mask, "node_sector"].iloc[0])],
            edgecolor=focal_colour,
            linewidth=1.8,
            zorder=5,
        )
    overview_axis.set_ylim(-0.65, len(ranking) - 0.35)
    overview_axis.set_yticks(positions, labels=ranking["delayed_node"])
    overview_left, overview_right = _x_limits(scaled_low, scaled_high)
    _format_effect_axis(overview_axis, overview_left, overview_right)
    overview_axis.set_title(f"All {len(ranking)} {policy_label} nodes (95% CI)", loc="left", pad=6)
    for label in overview_axis.get_yticklabels():
        if label.get_text() == focal_node:
            label.set_color(focal_colour)
            label.set_fontweight("bold")
    _panel_tag(overview_axis, "a")

    sector_order = ("water", "energy", "food")

    def _node_sort_key(node: object) -> tuple[str, int | str]:
        text = str(node)
        suffix = text[1:]
        return text[:1], int(suffix) if suffix.isdigit() else suffix

    grouped_nodes = {
        sector: sorted(
            ranking.loc[ranking["node_sector"].eq(sector), "delayed_node"].tolist(),
            key=_node_sort_key,
        )
        for sector in sector_order
    }
    matrix_columns = max((len(nodes) for nodes in grouped_nodes.values()), default=1)
    matrix = np.full((len(sector_order), matrix_columns), np.nan, dtype=float)
    matrix_nodes: list[list[str | None]] = [
        [None for _ in range(matrix_columns)] for _ in sector_order
    ]
    value_lookup = ranking.set_index("delayed_node")["mean_change"].mul(scale).to_dict()
    for row_index, sector in enumerate(sector_order):
        for column_index, node in enumerate(grouped_nodes[sector]):
            matrix[row_index, column_index] = float(value_lookup[node])
            matrix_nodes[row_index][column_index] = str(node)
    heat_limit = max(float(np.nanmax(np.abs(matrix))), 0.05)
    colour_map = matplotlib.colormaps["RdBu_r"].copy()
    colour_map.set_bad("#F1F3F4")
    image = matrix_axis.imshow(
        np.ma.masked_invalid(matrix),
        cmap=colour_map,
        norm=TwoSlopeNorm(vmin=-heat_limit, vcenter=0.0, vmax=heat_limit),
        interpolation="nearest",
        aspect="auto",
    )
    matrix_axis.set_yticks(range(len(sector_order)), labels=[sector_labels[s] for s in sector_order])
    matrix_axis.set_xticks([])
    matrix_axis.set_xticks(np.arange(-0.5, matrix_columns, 1.0), minor=True)
    matrix_axis.set_yticks(np.arange(-0.5, len(sector_order), 1.0), minor=True)
    matrix_axis.grid(which="minor", color="white", linewidth=1.35)
    matrix_axis.tick_params(axis="both", which="both", length=0, pad=3)
    for spine in matrix_axis.spines.values():
        spine.set_visible(False)
    for row_index, row_nodes in enumerate(matrix_nodes):
        for column_index, node in enumerate(row_nodes):
            if node is None:
                continue
            value = matrix[row_index, column_index]
            matrix_axis.text(
                column_index,
                row_index,
                node,
                ha="center",
                va="center",
                fontsize=6.0,
                color="white" if abs(value) > 0.46 * heat_limit else "#202428",
                fontweight="bold" if node == focal_node else "normal",
                zorder=3,
            )
            if node == focal_node:
                from matplotlib.patches import Rectangle

                matrix_axis.add_patch(
                    Rectangle(
                        (column_index - 0.48, row_index - 0.48),
                        0.96,
                        0.96,
                        fill=False,
                        edgecolor=focal_colour,
                        linewidth=2.0,
                        zorder=4,
                    )
                )
    matrix_axis.set_title("Sector-grouped mean change", loc="left", pad=6)
    _panel_tag(matrix_axis, "b")
    colour_bar = fig.colorbar(image, cax=colour_axis, orientation="horizontal")
    colour_bar.set_ticks([-heat_limit, 0.0, heat_limit])
    colour_bar.ax.tick_params(labelsize=6.3, width=0.55, length=2.2, pad=1.5)
    colour_bar.outline.set_linewidth(0.55)
    colour_bar.set_label(r"Mean $\Delta P_{\mathrm{avg}}$ ($\times 10^{-3}$)", fontsize=6.8, labelpad=1.5)

    top_count = min(5, len(ranking))
    detail = (
        ranking.nlargest(top_count, "mean_change")
        .sort_values("mean_change", ascending=True)
        .reset_index(drop=True)
    )
    detail_positions = np.arange(len(detail))
    detail_values = detail["mean_change"].to_numpy(float) * scale
    detail_low = detail["ci95_low"].to_numpy(float) * scale
    detail_high = detail["ci95_high"].to_numpy(float) * scale
    detail_focal = detail["delayed_node"].eq(focal_node).to_numpy()
    if detail_focal.any():
        row = int(detail_positions[detail_focal][0])
        detail_axis.axhspan(row - 0.46, row + 0.46, color="#FBE4DC", zorder=0)
    detail_axis.errorbar(
        detail_values,
        detail_positions,
        xerr=np.vstack([detail_values - detail_low, detail_high - detail_values]),
        fmt="none",
        ecolor="#6B737A",
        elinewidth=0.9,
        capsize=2.1,
        capthick=0.85,
        zorder=2,
    )
    for sector in sector_order:
        selected = detail["node_sector"].eq(sector).to_numpy() & ~detail_focal
        detail_axis.scatter(
            detail_values[selected],
            detail_positions[selected],
            marker=sector_markers[sector],
            s=43,
            facecolor=SECTOR_COLORS[sector],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
    if detail_focal.any():
        focal_sector = str(detail.loc[detail_focal, "node_sector"].iloc[0])
        detail_axis.errorbar(
            detail_values[detail_focal],
            detail_positions[detail_focal],
            xerr=np.vstack(
                [
                    detail_values[detail_focal] - detail_low[detail_focal],
                    detail_high[detail_focal] - detail_values[detail_focal],
                ]
            ),
            fmt="none",
            ecolor=focal_colour,
            elinewidth=1.35,
            capsize=2.6,
            capthick=1.2,
            zorder=4,
        )
        detail_axis.scatter(
            detail_values[detail_focal],
            detail_positions[detail_focal],
            marker=sector_markers[focal_sector],
            s=82,
            facecolor=SECTOR_COLORS[focal_sector],
            edgecolor=focal_colour,
            linewidth=1.8,
            zorder=5,
        )
    detail_axis.set_ylim(-0.65, len(detail) - 0.25)
    detail_axis.set_yticks(detail_positions, labels=detail["delayed_node"])
    detail_left, detail_right = _x_limits(detail_low, detail_high)
    _format_effect_axis(detail_axis, detail_left, detail_right)
    detail_axis.set_title(f"Top {top_count} point estimates", loc="left", pad=6)
    for label in detail_axis.get_yticklabels():
        if label.get_text() == focal_node:
            label.set_color(focal_colour)
            label.set_fontweight("bold")
    if detail_focal.any():
        focal_low = float(detail_low[detail_focal][0])
        status = "CI entirely above 0" if focal_low > 0 else "largest point estimate"
        detail_axis.text(
            0.98,
            0.985,
            f"{focal_node} · Hetao diversion: {status}",
            transform=detail_axis.transAxes,
            ha="right",
            va="top",
            fontsize=6.15,
            color="#6D271E",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#D9A093", "linewidth": 0.7},
            zorder=6,
        )
    _panel_tag(detail_axis, "c")

    handles = [
        plt.Line2D(
            [],
            [],
            linestyle="",
            marker=sector_markers[sector],
            markerfacecolor=SECTOR_COLORS[sector],
            markeredgecolor="white",
            markeredgewidth=0.65,
            markersize=7.2,
            label=sector_labels[sector],
        )
        for sector in ("water", "energy", "food")
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.985),
        ncol=3,
        frameon=False,
        columnspacing=1.8,
        handletextpad=0.45,
        borderaxespad=0,
    )
    fig.text(
        0.985,
        0.018,
        "Positive values denote deterioration. Exploratory screen; 95% bootstrap CIs are not multiplicity-adjusted.",
        ha="right",
        va="bottom",
        fontsize=6.55,
        color="#555B61",
    )
    _save(fig, path)


def plot_information_location_method_comparison(
    ppo_sector_effects: pd.DataFrame,
    gnn_sector_effects: pd.DataFrame,
    ppo_node_ranking: pd.DataFrame,
    gnn_node_ranking: pd.DataFrame,
    path: Path,
) -> None:
    """Descriptively compare identical information-location scenarios by policy.

    The top row keeps the source sector as the small-multiple dimension and
    the algorithm as colour/marker, avoiding a six-line panel.  The lower
    scatter preserves each of the 24 node-specific tau5 point estimates; it is
    intentionally a descriptive cross-policy comparison rather than a new
    multiple-testing exercise.
    """

    _apply_style()
    source_frames = {
        "PPO": ppo_sector_effects[ppo_sector_effects["outcome"] == "cost"].copy(),
        "GNN-PPO": gnn_sector_effects[gnn_sector_effects["outcome"] == "cost"].copy(),
    }
    values = np.concatenate(
        [
            frame[["ci95_low", "ci95_high"]].to_numpy(float).ravel()
            for frame in source_frames.values()
        ]
    )
    y_min, y_max = float(values.min()), float(values.max())
    y_margin = max(0.001, 0.10 * (y_max - y_min))
    # A compact physical canvas preserves readable labels when the two
    # publication figures are stacked on one manuscript float page.
    fig = plt.figure(figsize=(6.30, 4.65))
    grid = fig.add_gridspec(
        2,
        3,
        left=0.09,
        right=0.985,
        top=0.86,
        bottom=0.14,
        hspace=0.58,
        wspace=0.30,
        height_ratios=(1.0, 1.16),
    )
    for index, sector in enumerate(("water", "energy", "food")):
        axis = fig.add_subplot(grid[0, index])
        for method in ("PPO", "GNN-PPO"):
            style = METHOD_STYLE[method]
            subset = source_frames[method]
            subset = subset[subset["delayed_sector"] == sector].sort_values("delay_months")
            mean = subset["mean_change"].to_numpy(float)
            low = subset["ci95_low"].to_numpy(float)
            high = subset["ci95_high"].to_numpy(float)
            axis.fill_between(subset["delay_months"], low, high, color=style["color"], alpha=0.10, lw=0)
            axis.errorbar(
                subset["delay_months"],
                mean,
                yerr=np.vstack([mean - low, high - mean]),
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=3.8,
                capsize=1.8,
                lw=1.15,
                label=method,
            )
        axis.axhline(0, color="#555555", lw=0.7, zorder=0)
        axis.set(
            title=f"{SECTOR_LABELS[sector]} observations delayed",
            xlabel="Delay (months)",
            xticks=range(1, 6),
            ylim=(y_min - y_margin, y_max + y_margin),
        )
        if index == 0:
            axis.set_ylabel(r"Change in $P_{\mathrm{avg}}$ vs each policy's $\tau=0$")
            axis.legend(loc="upper left", handlelength=2.0)
        else:
            axis.set_yticklabels([])
        axis.grid(axis="y", color="#D9D9D9", lw=0.45)
        _panel_label(axis, "abc"[index])

    ppo_nodes = ppo_node_ranking[
        ppo_node_ranking["outcome"] == "cost"
    ][["delayed_node", "node_sector", "mean_change"]].rename(
        columns={"mean_change": "ppo_cost_penalty"}
    )
    gnn_nodes = gnn_node_ranking[
        gnn_node_ranking["outcome"] == "cost"
    ][["delayed_node", "node_sector", "mean_change"]].rename(
        columns={"mean_change": "gnn_cost_penalty"}
    )
    nodes = ppo_nodes.merge(
        gnn_nodes,
        on=["delayed_node", "node_sector"],
        validate="one_to_one",
    )
    axis = fig.add_subplot(grid[1, :])
    both = nodes[["gnn_cost_penalty", "ppo_cost_penalty"]].to_numpy(float)
    limit = max(float(np.abs(both).max()) * 1.18, 0.0005)
    axis.plot([-limit, limit], [-limit, limit], color="#777777", ls=":", lw=0.9, zorder=0)
    for sector in ("water", "energy", "food"):
        subset = nodes[nodes["node_sector"] == sector]
        axis.scatter(
            subset["gnn_cost_penalty"],
            subset["ppo_cost_penalty"],
            color=SECTOR_COLORS[sector],
            s=38,
            edgecolor="white",
            linewidth=0.45,
            label=SECTOR_LABELS[sector],
            zorder=3,
        )
    labels = nodes.assign(maximum=nodes[["ppo_cost_penalty", "gnn_cost_penalty"]].max(axis=1))
    for row in labels.nlargest(3, "maximum").itertuples(index=False):
        axis.annotate(
            row.delayed_node,
            (row.gnn_cost_penalty, row.ppo_cost_penalty),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=6.7,
            color="#333333",
        )
    axis.axhline(0, color="#555555", lw=0.7, zorder=0)
    axis.axvline(0, color="#555555", lw=0.7, zorder=0)
    axis.set(
        title="24-node sensitivity at a 5-month delay",
        xlabel=r"GNN-PPO change in $P_{\mathrm{avg}}$ vs its $\tau=0$",
        ylabel=r"PPO change in $P_{\mathrm{avg}}$ vs its $\tau=0$",
        xlim=(-limit, limit),
        ylim=(-limit, limit),
    )
    axis.grid(color="#D9D9D9", lw=0.45)
    axis.legend(
        loc="upper left",
        title="Delayed node sector",
        title_fontsize=7.0,
        frameon=False,
    )
    axis.text(
        0.985,
        0.035,
        "Dotted diagonal: equal point estimates\nLabels: three largest penalties across methods",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.35,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "#F4F4F4", "edgecolor": "none"},
    )
    _panel_label(axis, "d")
    fig.suptitle(
        "Information-location sensitivity under PPO and GNN-PPO",
        y=0.985,
        fontsize=10.0,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.035,
        "Both policies use the same delay definitions, 10 training seeds and held-out years. Top-row intervals are 95% two-way bootstrap CIs.\n"
        "All penalties are within-policy changes relative to the same policy's no-delay reference; the cross-policy comparison is descriptive.",
        ha="center",
        va="bottom",
        fontsize=6.0,
        linespacing=1.12,
    )
    _save(fig, path)


def plot_cross_method_cascade_comparison(
    ppo_cascade_effects: pd.DataFrame,
    gnn_cascade_effects: pd.DataFrame,
    path: Path,
) -> None:
    """Compare WEF propagation under identical sector-specific delays.

    A shared colour scale makes the magnitude difference between policy
    families visible while the cell annotations preserve small GNN-PPO effects.
    """

    _apply_style()
    outcome_order = ("water_cost", "energy_cost", "food_cost")
    outcome_labels = ("Water\n(ecological)", "Energy", "Irrigation")
    method_rows = (
        ("PPO", ppo_cascade_effects),
        ("GNN-PPO", gnn_cascade_effects),
    )
    combined = pd.concat(
        [ppo_cascade_effects, gnn_cascade_effects], ignore_index=True
    )
    max_abs = float(
        np.nanmax(
            np.abs(combined[["mean_change", "ci95_low", "ci95_high"]].to_numpy(float))
        )
    )
    max_abs = max(max_abs, 0.002)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(6.45, 4.27),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.12, "hspace": 0.22},
    )
    image = None
    panel_index = 0
    for row_index, (method, frame) in enumerate(method_rows):
        for column_index, sector in enumerate(("water", "energy", "food")):
            axis = axes[row_index, column_index]
            subset = frame[frame["delayed_sector"] == sector]
            matrix = subset.pivot(
                index="delay_months", columns="outcome", values="mean_change"
            ).reindex(index=range(1, 6), columns=outcome_order)
            image = axis.imshow(
                matrix.to_numpy(float),
                origin="lower",
                aspect="auto",
                cmap="RdBu_r",
                vmin=-max_abs,
                vmax=max_abs,
            )
            for matrix_row, delay in enumerate(range(1, 6)):
                for matrix_column, outcome in enumerate(outcome_order):
                    cell = float(matrix.loc[delay, outcome])
                    text_color = "white" if abs(cell) > 0.58 * max_abs else "#111111"
                    axis.text(
                        matrix_column,
                        matrix_row,
                        f"{cell:+.3f}",
                        ha="center",
                        va="center",
                        fontsize=5.9,
                        color=text_color,
                    )
            if row_index == 0:
                axis.set_title(f"{SECTOR_LABELS[sector]} observations delayed")
            axis.set(
                xticks=range(3),
                xticklabels=outcome_labels,
                yticks=range(5),
                yticklabels=range(1, 6),
            )
            axis.tick_params(axis="x", labelsize=6.3, pad=2)
            _panel_label(axis, "abcdef"[panel_index])
            panel_index += 1
        axes[row_index, 0].set_ylabel(f"{method}\nDelay (months)")

    colorbar = fig.colorbar(image, ax=axes, fraction=0.027, pad=0.025)
    colorbar.set_label(
        "Change in average normalised penalty component vs no-delay condition\n(positive is worse)",
        fontsize=6.8,
    )
    colorbar.ax.tick_params(labelsize=6.2)
    fig.suptitle(
        "Propagation of sector-specific observation delays through WEF outcomes",
        y=0.985,
        fontsize=9.8,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.025,
        "Rows compare fixed PPO and GNN-PPO policies; columns identify the source of stale information.\n"
        "Cell values are unweighted normalised penalty-component changes relative to the same policy, seed and held-out year under the no-delay condition.",
        ha="center",
        va="bottom",
        fontsize=5.8,
        linespacing=1.12,
    )
    fig.subplots_adjust(left=0.095, right=0.89, top=0.86, bottom=0.15)
    _save(fig, path)
