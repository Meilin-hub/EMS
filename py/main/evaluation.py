"""Rule baselines, deterministic evaluation, and physical performance metrics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import torch

from .environment import YellowRiverWEFEnv
from .model import ActorCritic


Controller = Callable[[YellowRiverWEFEnv, np.ndarray], np.ndarray]


def demand_rule(env: YellowRiverWEFEnv, observation: np.ndarray) -> np.ndarray:
    """Transparent rule curve: mid-capacity releases and demand-matching diversions."""

    del observation
    row = env.current_row
    ex = env._exogenous(row)
    rmax1, rmax2, dmax_n, dmax_h = env.physical_action_scales()
    target1 = 0.60 * float(row["W1_total_capacity_reference [1e8 m3]"])
    target2 = 0.60 * float(row["W2_total_capacity_reference [1e8 m3]"])
    release1 = max(0.0, env.state["s1"] + ex["inflow1"] - target1)
    inflow2 = max(0.0, min(release1, rmax1) + ex["g3"])
    release2 = max(0.0, env.state["s2"] + inflow2 - target2)
    requirements = env._crop_requirements(row)
    eta_n = float(row["W7_Ningxia_irrigation_efficiency_eta [-]"])
    eta_h = float(row["W8_Hetao_irrigation_efficiency_eta [-]"])
    diversion_n = requirements[:3].sum() / max(eta_n, 1.0e-6)
    diversion_h = requirements[3:].sum() / max(eta_h, 1.0e-6)
    return np.clip(
        [release1 / rmax1, release2 / rmax2, diversion_n / dmax_n, diversion_h / dmax_h],
        0.0,
        1.0,
    )


def model_controller(model: ActorCritic, device: str = "cpu") -> Controller:
    torch_device = torch.device(device)
    model.to(torch_device)
    model.eval()

    def controller(env: YellowRiverWEFEnv, observation: np.ndarray) -> np.ndarray:
        del env
        with torch.no_grad():
            action, _, _ = model.act(
                torch.as_tensor(observation, dtype=torch.float32, device=torch_device),
                deterministic=True,
            )
        return action.squeeze(0).cpu().numpy()

    return controller


def summarize_episode(records: list[dict[str, Any]]) -> dict[str, float]:
    frame = pd.DataFrame(records)
    crop_requirement = float(frame["crop_requirement"].sum())
    crop_deficit = float(frame["crop_deficit"].sum())
    total_load = float((frame["base_demand"] + frame["pump_demand_N"] + frame["pump_demand_H"]).sum())
    pump_energy_requested = float((frame["pump_demand_N"] + frame["pump_demand_H"]).sum())
    pump_energy_executed = float((frame["pump_executed_N"] + frame["pump_executed_H"]).sum())
    pump_shortfall = float((frame["pump_shortfall_N"] + frame["pump_shortfall_H"]).sum())
    base_unserved = float(frame["unserved_energy"].sum())
    energy_shortfall = pump_shortfall + base_unserved
    local_generation = float(
        (frame["hydro_E1"] + frame["hydro_E2"] + frame["renewable_E3"]).sum()
    )
    served_energy = max(0.0, total_load - energy_shortfall)
    requested_diversion = float((frame["requested_D_N"] + frame["requested_D_H"]).sum())
    delivered_diversion = float((frame["D_N"] + frame["D_H"]).sum())
    final = frame.iloc[-1]
    return {
        "total_reward": float(frame["reward"].sum()),
        "mean_monthly_reward": float(frame["reward"].mean()),
        # These three unit-free components are exported for Experiments 3 and 4. They
        # are calculated before reward weights are applied, so their changes
        # can be compared across the water, energy and irrigation branches of
        # the WEF network without conflating a physical response with the 0.35
        # energy reward weight.
        "mean_water_cost_normalized": float(frame["cost_water_normalized"].mean()),
        "mean_energy_cost_normalized": float(frame["cost_energy_normalized"].mean()),
        "mean_food_cost_normalized": float(frame["cost_food_normalized"].mean()),
        "ecological_flow_reliability": float((frame["Q9"] >= frame["eco_threshold"] - 1.0e-9).mean()),
        "ecological_deficit_total": float(frame["C_W"].sum()),
        "ecological_deficit_normalized": float(frame["cost_water_normalized"].mean()),
        "irrigation_satisfaction_ratio": 1.0 - crop_deficit / max(crop_requirement, 1.0e-8),
        "crop_deficit_total": crop_deficit,
        "electricity_import_total": float(frame["electricity_import_E4"].sum()),
        "pump_shortfall_total": pump_shortfall,
        "unserved_energy_total": base_unserved,
        "energy_self_sufficiency": min(local_generation, served_energy) / max(total_load, 1.0e-8),
        "energy_demand_satisfaction_ratio": served_energy / max(total_load, 1.0e-8),
        "pump_energy_satisfaction_ratio": pump_energy_executed / max(pump_energy_requested, 1.0e-8),
        "pumping_delivery_ratio": delivered_diversion / max(requested_diversion, 1.0e-8),
        "electricity_limited_diversion_total": requested_diversion - delivered_diversion,
        "requested_diversion_total": requested_diversion,
        "delivered_diversion_total": delivered_diversion,
        "pump_energy_requested_total": pump_energy_requested,
        "pump_energy_executed_total": pump_energy_executed,
        "hydropower_total": float((frame["hydro_E1"] + frame["hydro_E2"]).sum()),
        "ningxia_diversion_total": float(frame["D_N"].sum()),
        "hetao_diversion_total": float(frame["D_H"].sum()),
        "mean_initial_storage_fraction": float(final["initial_storage_fraction"]),
        "mean_final_storage_fraction": float(final["mean_storage_fraction"]),
        "storage_recovery_ratio": float(final["storage_recovery_ratio"]),
        "terminal_storage_deficit_normalized": float(final["terminal_storage_deficit_normalized"]),
        "storage_drawdown_from_initial_pp": float(final["storage_drawdown_from_initial_pp"]),
        "storage_reserve_limited_release_total": float(
            frame["storage_reserve_limited_release_total"].sum()
        ),
        "projection_rate": float(frame["projected"].mean()),
        "safety_violations": float(frame["safety_violations"].sum()),
    }


def evaluate_controller(
    env: YellowRiverWEFEnv,
    controller: Controller,
    years: tuple[int, ...],
    method: str,
    seed: int,
    delay_scenario: str = "tau0",
    disturbance: str = "normal",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    for year in years:
        observation = env.reset(
            year,
            delay_scenario=delay_scenario,
            disturbance=disturbance,
        )
        done = False
        while not done:
            action = controller(env, observation)
            observation, _, done, _ = env.step(action)
        records = [
            dict(
                record,
                method=method,
                seed=seed,
            )
            for record in env.records
        ]
        monthly.extend(records)
        episode = summarize_episode(records)
        episode.update(
            method=method,
            seed=seed,
            year=year,
            delay_scenario=delay_scenario,
            disturbance=disturbance,
        )
        episodes.append(episode)
    return pd.DataFrame(monthly), pd.DataFrame(episodes)


def aggregate_metrics(episodes: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    identifier_columns = set(group_columns + ["seed", "year"])
    numeric = [column for column in episodes.select_dtypes(include=[np.number]).columns if column not in identifier_columns]
    grouped = episodes.groupby(group_columns, dropna=False)[numeric].agg(["mean", "std"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    return grouped.reset_index()
