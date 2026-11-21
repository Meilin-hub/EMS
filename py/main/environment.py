"""Monthly Yellow River water-energy-food simulator with safe action projection.

The transition equations follow the manuscript literally where they are fully
specified.  Three implementation choices that the draft leaves implicit are
made explicit here: reservoir overflow is reported as spill; electricity imports
are dispatched minimally after local generation, with base load served before
pumping demand; and only the water volume supported by executed pumping energy
is delivered to the irrigation districts.  These choices are deterministic and
configurable through the input data rather than learned by the policy.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import numpy as np
import pandas as pd

from .config import PhysicalConfig, RewardConfig
from .data import WEFData, clip_feature, safe_ratio
from .graph import GRAPH, NODE_NAMES, SECTOR_NODE_NAMES, node_indexes


MAX_DELAY = 5


@dataclass(frozen=True)
class Disturbance:
    name: str = "normal"
    inflow_factor: float = 1.0
    positive_reach_gain_factor: float = 1.0
    renewable_factor: float = 1.0
    base_demand_factor: float = 1.0
    crop_demand_factor: float = 1.0


DISTURBANCES: dict[str, Disturbance] = {
    "normal": Disturbance(),
    "dry_hot": Disturbance(
        name="dry_hot",
        inflow_factor=0.75,
        positive_reach_gain_factor=0.80,
        renewable_factor=0.80,
        base_demand_factor=1.10,
        crop_demand_factor=1.10,
    ),
    "extreme": Disturbance(
        name="extreme",
        inflow_factor=0.60,
        positive_reach_gain_factor=0.65,
        renewable_factor=0.65,
        base_demand_factor=1.15,
        crop_demand_factor=1.20,
    ),
}


class YellowRiverWEFEnv:
    """A small Gym-like environment without requiring Gym at runtime.

    Policy actions are four unit-interval commands.  They are converted to the
    physical units [R1, R2, D_N, D_H] using the current row's capacities and
    then sequentially projected onto storage, intake, channel, and annual-budget
    constraints.  Every returned observation has shape ``(24, 12)``.
    """

    node_count = 24
    feature_dim = 12
    action_dim = 4

    def __init__(
        self,
        data: WEFData,
        reward: RewardConfig,
        physical: PhysicalConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.data = data
        self.reward_config = reward
        self.physical_config = physical or PhysicalConfig()
        self.rng = np.random.default_rng(seed)
        self.rows: pd.DataFrame | None = None
        self.year = 0
        self.t = 0
        self.delay_name = "tau0"
        self.delays = np.zeros(self.node_count, dtype=np.int64)
        self.disturbance = DISTURBANCES["normal"]
        self.state: dict[str, Any] = {}
        self.history: list[np.ndarray] = []
        self.records: list[dict[str, Any]] = []

    @property
    def current_row(self) -> pd.Series:
        if self.rows is None:
            raise RuntimeError("Call reset() before accessing the environment")
        return self.rows.iloc[min(self.t, len(self.rows) - 1)]

    def _resolve_delays(self, scenario: str) -> np.ndarray:
        """Translate an explicit observation-delay scenario into 24 node lags.

        ``tauN`` is the Experiment 2 definition: every one of the 24 nodes is
        delayed by the same ``N`` months.  Experiments 3 and 4 add two deliberately
        non-uniform definitions: ``sector_<water|energy|food>_tauN`` delays all
        nodes in one named sector while every other node remains current; and
        ``node_<node-name>_tauN`` delays exactly one named node.  In all cases,
        the physical state transition remains current and only the policy's
        observation is delayed.
        """
        if scenario.startswith("tau") and scenario[3:].isdigit():
            delay = int(scenario[3:])
            if 0 <= delay <= MAX_DELAY:
                return np.full(self.node_count, delay, dtype=np.int64)
        sector_match = re.fullmatch(r"sector_(water|energy|food)_tau(\d+)", scenario)
        if sector_match:
            sector, delay_text = sector_match.groups()
            delay = int(delay_text)
            if 0 <= delay <= MAX_DELAY:
                delays = np.zeros(self.node_count, dtype=np.int64)
                delays[list(node_indexes(SECTOR_NODE_NAMES[sector]))] = delay
                return delays
        node_match = re.fullmatch(r"node_([A-Za-z]\d+)_tau(\d+)", scenario)
        if node_match:
            node, delay_text = node_match.groups()
            delay = int(delay_text)
            if node in NODE_NAMES and 0 <= delay <= MAX_DELAY:
                delays = np.zeros(self.node_count, dtype=np.int64)
                delays[NODE_NAMES.index(node)] = delay
                return delays
        raise ValueError(
            "Delay scenario must be tau0--tau5, sector_<water|energy|food>_tau0--tau5, "
            "or node_<WEF-node>_tau0--tau5."
        )

    def reset(
        self,
        year: int,
        delay_scenario: str = "tau0",
        disturbance: str = "normal",
    ) -> np.ndarray:
        if disturbance not in DISTURBANCES:
            raise ValueError(f"Unknown disturbance: {disturbance}")
        self.rows = self.data.year(year)
        self.year = int(year)
        self.t = 0
        self.delay_name = delay_scenario
        self.delays = self._resolve_delays(delay_scenario)
        self.disturbance = DISTURBANCES[disturbance]
        row = self.rows.iloc[0]
        self.state = {
            "s1": float(row["W1_storage_begin_S [1e8 m3]"]),
            "s2": float(row["W2_storage_begin_S [1e8 m3]"]),
            "initial_s1": float(row["W1_storage_begin_S [1e8 m3]"]),
            "initial_s2": float(row["W2_storage_begin_S [1e8 m3]"]),
            "used_n": 0.0,
            "used_h": 0.0,
            "r1": 0.0, "r2": 0.0,
            "river": np.zeros(4, dtype=np.float64),
            "intake_n": 0.0, "intake_h": 0.0, "q9": float(row["W9_Toudaoguai_flow_Q [1e8 m3 month-1]"]),
            "div_n": 0.0, "div_h": 0.0, "effective_n": 0.0, "effective_h": 0.0,
            "crop_alloc": np.zeros(6, dtype=np.float64),
            "crop_deficit": np.zeros(6, dtype=np.float64),
            "generation": np.zeros(2, dtype=np.float64),
            "grid_import": 0.0, "residual": 0.0, "curtailment": 0.0, "unserved": 0.0,
            "pump_demand": np.zeros(2, dtype=np.float64),
            "pump_accepted": np.zeros(2, dtype=np.float64),
            "pump_alloc": np.zeros(2, dtype=np.float64),
            "pump_shortfall": np.zeros(2, dtype=np.float64),
            "pump_exec": np.zeros(2, dtype=np.float64),
            "spill": np.zeros(2, dtype=np.float64),
        }
        self.records = []
        initial = self._true_node_features()
        # Bootstrap the observation buffer without leaking pre-episode years.
        self.history = [initial.copy() for _ in range(MAX_DELAY + 1)]
        return self.observation()

    def observation(self) -> np.ndarray:
        delayed = np.empty((self.node_count, self.feature_dim), dtype=np.float32)
        latest = len(self.history) - 1
        for node in range(self.node_count):
            tau = int(self.delays[node])
            delayed[node] = self.history[max(0, latest - tau)][node]
            # Keep the twelfth channel zero-padded for compatibility with the
            # retained 24 x 12 checkpoints.
            delayed[node, -1] = 0.0
        return delayed

    def _exogenous(self, row: pd.Series) -> dict[str, float]:
        def gain(name: str) -> float:
            value = float(row[name])
            return value * self.disturbance.positive_reach_gain_factor if value > 0 else value

        return {
            "inflow1": float(row["W1_inflow_I [1e8 m3 month-1]"]) * self.disturbance.inflow_factor,
            "g3": gain("W3_net_reach_gain_G [1e8 m3 month-1]"),
            "g4": gain("W4_net_reach_gain_G [1e8 m3 month-1]"),
            "g5": gain("W5_net_reach_gain_G [1e8 m3 month-1]"),
            "g6": gain("W6_net_reach_gain_G [1e8 m3 month-1]"),
            "renewable": float(row["E3_nonhydro_renewable_generation [GWh month-1]"]) * self.disturbance.renewable_factor,
            "base": float(row["E5_base_demand_L_base [GWh month-1]"]) * self.disturbance.base_demand_factor,
        }

    def _crop_requirements(self, row: pd.Series) -> np.ndarray:
        return np.asarray(
            [float(row[f"F{i}_water_requirement_W [1e8 m3 month-1]"]) for i in range(1, 7)],
            dtype=np.float64,
        ) * self.disturbance.crop_demand_factor

    @staticmethod
    def _pump_energy(volume: float, head: float, efficiency: float) -> float:
        # rho*g*(volume*1e8)*head / efficiency, converted from joules to GWh.
        return 0.2725 * max(volume, 0.0) * max(head, 0.0) / max(efficiency, 1.0e-6)

    @staticmethod
    def _pump_volume(energy: float, head: float, efficiency: float) -> float:
        """Invert the pump-energy equation to obtain executable water volume."""

        if head <= 0.0:
            return float("inf")
        return max(energy, 0.0) * max(efficiency, 1.0e-6) / (0.2725 * head)

    def _minimum_recoverable_end_storage(
        self,
        reservoir: int,
        current_index: int,
        target: float,
    ) -> float:
        """Return the viability floor after the current month's transition.

        The backward recursion starts at the explicit terminal target and asks
        how much storage must remain before each future month if all future
        discretionary releases were zero. W1 uses disturbed future inflow. W2
        uses only the guaranteed non-negative natural reach contribution and
        therefore does not assume future upstream policy releases. The forcing
        acts as an annual planning forecast for the safety projection and is not
        added to the learned policy observation.
        """

        if self.rows is None:
            raise RuntimeError("Call reset() before computing a storage floor")
        required = float(target)
        future_rows = self.rows.iloc[current_index + 1 :]
        for _, future in future_rows.iloc[::-1].iterrows():
            if reservoir == 1:
                incoming = (
                    float(future["W1_inflow_I [1e8 m3 month-1]"])
                    * self.disturbance.inflow_factor
                )
                loss = float(future["W1_reservoir_loss_L_res [1e8 m3 month-1]"])
                minimum = float(future["W1_min_storage_Smin [1e8 m3]"])
            elif reservoir == 2:
                gain = float(future["W3_net_reach_gain_G [1e8 m3 month-1]"])
                if gain > 0.0:
                    gain *= self.disturbance.positive_reach_gain_factor
                incoming = max(0.0, gain)
                loss = float(future["W2_reservoir_loss_L_res [1e8 m3 month-1]"])
                minimum = float(future["W2_min_storage_Smin [1e8 m3]"])
            else:
                raise ValueError("reservoir must be 1 or 2")
            required = max(minimum, required - incoming + loss)
        return required

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if self.rows is None or self.t >= len(self.rows):
            raise RuntimeError("Episode is not active; call reset()")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.action_dim,) or not np.isfinite(action).all():
            raise ValueError("Action must be a finite vector with shape (4,)")
        unit_action = np.clip(action, 0.0, 1.0)
        row = self.current_row
        ex = self._exogenous(row)

        r1_cap = float(row["W1_max_release_Rmax [1e8 m3 month-1]"])
        r2_cap = float(row["W2_max_release_Rmax [1e8 m3 month-1]"])
        dn_cap = float(row["W7_Ningxia_channel_capacity_Dmax [1e8 m3 month-1]"])
        dh_cap = float(row["W8_Hetao_channel_capacity_Dmax [1e8 m3 month-1]"])
        raw = unit_action * np.asarray([r1_cap, r2_cap, dn_cap, dh_cap])

        loss1 = float(row["W1_reservoir_loss_L_res [1e8 m3 month-1]"])
        smin1 = float(row["W1_min_storage_Smin [1e8 m3]"])
        smax1 = float(row["W1_total_capacity_reference [1e8 m3]"])
        storage_target_fraction = float(
            np.clip(self.physical_config.terminal_storage_target_fraction, 0.0, 1.0)
        )
        terminal_target_s1 = max(
            smin1, storage_target_fraction * self.state["initial_s1"]
        )
        reserve_floor_s1 = self._minimum_recoverable_end_storage(
            reservoir=1,
            current_index=self.t,
            target=terminal_target_s1,
        )
        physical_feasible_r1 = max(
            0.0, min(r1_cap, self.state["s1"] + ex["inflow1"] - loss1 - smin1)
        )
        feasible_r1 = max(
            0.0,
            min(
                r1_cap,
                self.state["s1"] + ex["inflow1"] - loss1 - reserve_floor_s1,
            ),
        )
        r1 = min(float(raw[0]), feasible_r1)
        reserve_limited_r1 = max(
            0.0, min(float(raw[0]), physical_feasible_r1) - r1
        )
        pre_spill_s1 = self.state["s1"] + ex["inflow1"] - r1 - loss1
        spill1 = max(0.0, pre_spill_s1 - smax1)
        s1_next = float(np.clip(pre_spill_s1 - spill1, smin1, smax1))

        q3 = max(0.0, r1 + ex["g3"])
        loss2 = float(row["W2_reservoir_loss_L_res [1e8 m3 month-1]"])
        smin2 = float(row["W2_min_storage_Smin [1e8 m3]"])
        smax2 = float(row["W2_total_capacity_reference [1e8 m3]"])
        terminal_target_s2 = max(
            smin2, storage_target_fraction * self.state["initial_s2"]
        )
        reserve_floor_s2 = self._minimum_recoverable_end_storage(
            reservoir=2,
            current_index=self.t,
            target=terminal_target_s2,
        )
        physical_feasible_r2 = max(
            0.0, min(r2_cap, self.state["s2"] + q3 - loss2 - smin2)
        )
        feasible_r2 = max(
            0.0,
            min(r2_cap, self.state["s2"] + q3 - loss2 - reserve_floor_s2),
        )
        r2 = min(float(raw[1]), feasible_r2)
        reserve_limited_r2 = max(
            0.0, min(float(raw[1]), physical_feasible_r2) - r2
        )
        pre_spill_s2 = self.state["s2"] + q3 - r2 - loss2
        spill2 = max(0.0, pre_spill_s2 - smax2)
        s2_next = float(np.clip(pre_spill_s2 - spill2, smin2, smax2))

        q_n = max(0.0, r2 + ex["g4"])
        budget_n = float(row["W7_Ningxia_annual_budget_B [1e8 m3 year-1]"])
        remaining_n = max(0.0, budget_n - self.state["used_n"])
        dn_requested = min(float(raw[2]), dn_cap, q_n, remaining_n)
        fn = float(row["W7_Ningxia_return_fraction_f [-]"])
        eta_n = float(row["W7_Ningxia_irrigation_efficiency_eta [-]"])

        # Use the fully requested Ningxia diversion to form a conservative
        # downstream availability bound.  Electricity can only reduce the
        # executed Ningxia diversion, so the final q5 cannot be smaller.
        q5_request_bound = max(0.0, q_n - (1.0 - fn) * dn_requested + ex["g5"])

        budget_h = float(row["W8_Hetao_annual_budget_B [1e8 m3 year-1]"])
        remaining_h = max(0.0, budget_h - self.state["used_h"])
        dh_requested = min(float(raw[3]), dh_cap, q5_request_bound, remaining_h)
        fh = float(row["W8_Hetao_return_fraction_f [-]"])
        eta_h = float(row["W8_Hetao_irrigation_efficiency_eta [-]"])

        beta1 = float(row["E1_hydropower_beta1 [GWh per 1e8 m3]"])
        beta2 = float(row["E2_hydropower_beta2 [GWh per 1e8 m3]"])
        days = float(row.get("fixed_period_days [day]", 30.0))
        e1_cap = float(row["E1_installed_capacity_reference [MW]"]) * 24.0 * days / 1000.0
        e2_cap = float(row["E2_installed_capacity_reference [MW]"]) * 24.0 * days / 1000.0
        e1, e2 = min(beta1 * r1, e1_cap), min(beta2 * r2, e2_cap)

        pump_heads = np.asarray(
            [float(row["E8_Ningxia_pump_head [m]"]), float(row["E9_Hetao_pump_head [m]"])],
            dtype=np.float64,
        )
        pump_efficiencies = np.asarray(
            [float(row["E8_Ningxia_pump_efficiency [-]"]), float(row["E9_Hetao_pump_efficiency [-]"])],
            dtype=np.float64,
        )
        diversion_requested = np.asarray([dn_requested, dh_requested], dtype=np.float64)
        pump_requested = np.asarray(
            [
                self._pump_energy(diversion_requested[k], pump_heads[k], pump_efficiencies[k])
                for k in range(2)
            ],
            dtype=np.float64,
        )
        interface_caps = np.asarray(
            [float(row["E6_Ningxia_interface_capacity [GWh month-1]"]), float(row["E7_Hetao_interface_capacity [GWh month-1]"])],
            dtype=np.float64,
        )
        pump_caps = np.asarray(
            [float(row["E8_Ningxia_pump_capacity_Ebar8 [GWh month-1]"]), float(row["E9_Hetao_pump_capacity_Ebar9 [GWh month-1]"])],
            dtype=np.float64,
        )
        pump_accepted = np.minimum.reduce([pump_requested, interface_caps, pump_caps])
        local = e1 + e2 + ex["renewable"]
        total_demand = ex["base"] + float(pump_accepted.sum())
        import_limit = float(row["E4_import_limit [GWh month-1]"]) * float(row["E4_availability [-]"])
        grid_import = min(import_limit, max(0.0, total_demand - local))
        total_energy = local + grid_import
        unserved = max(0.0, ex["base"] - total_energy)
        pump_available = max(0.0, total_energy - ex["base"])
        if pump_accepted.sum() > 0.0:
            delivery_fraction = min(1.0, pump_available / float(pump_accepted.sum()))
            pump_alloc = delivery_fraction * pump_accepted
        else:
            pump_alloc = np.zeros(2, dtype=np.float64)
        pump_exec = pump_alloc
        pump_capacity_shortfall = np.maximum(0.0, pump_requested - pump_accepted)
        pump_supply_shortfall = np.maximum(0.0, pump_accepted - pump_exec)
        pump_shortfall = np.maximum(0.0, pump_requested - pump_exec)
        residual = max(0.0, total_energy - ex["base"] - float(pump_exec.sum()))
        curtailment = residual

        # Electricity now physically constrains water.  Only the water volume
        # supported by executed pumping energy reaches the districts and crops.
        executable_diversion = np.asarray(
            [
                self._pump_volume(pump_exec[k], pump_heads[k], pump_efficiencies[k])
                for k in range(2)
            ],
            dtype=np.float64,
        )
        dn = min(dn_requested, float(executable_diversion[0]))
        q5 = max(0.0, q_n - (1.0 - fn) * dn + ex["g5"])
        dh = min(dh_requested, float(executable_diversion[1]), q5, remaining_h)
        effective_n = eta_n * dn
        effective_h = eta_h * dh
        q9 = max(0.0, q5 - (1.0 - fh) * dh + ex["g6"])

        requirements = self._crop_requirements(row)
        crop_alloc = np.zeros(6, dtype=np.float64)
        for indices, supply in ((slice(0, 3), effective_n), (slice(3, 6), effective_h)):
            total = float(requirements[indices].sum())
            if total > 0.0:
                crop_alloc[indices] = supply * requirements[indices] / total
        crop_deficit = np.maximum(0.0, requirements - crop_alloc)

        eco_threshold = float(row["W9_ecological_threshold [1e8 m3 month-1]"])
        cw = max(0.0, eco_threshold - q9)
        ce = grid_import + unserved + float(pump_shortfall.sum())
        cf = float(crop_deficit.sum())
        water_cost = safe_ratio(cw, eco_threshold)
        shortage_cost = safe_ratio(
            unserved + float(pump_shortfall.sum()),
            ex["base"] + float(pump_requested.sum()),
        )
        import_cost = safe_ratio(grid_import, max(import_limit, 1.0))
        energy_cost = self.reward_config.import_share * import_cost + (1.0 - self.reward_config.import_share) * shortage_cost
        food_cost = safe_ratio(cf, float(requirements.sum())) if requirements.sum() > 0 else 0.0

        is_terminal = self.t == len(self.rows) - 1
        initial_s1 = float(self.state["initial_s1"])
        initial_s2 = float(self.state["initial_s2"])
        terminal_deficit_1 = max(0.0, initial_s1 - s1_next) / max(initial_s1 - smin1, 1.0e-8)
        terminal_deficit_2 = max(0.0, initial_s2 - s2_next) / max(initial_s2 - smin2, 1.0e-8)
        terminal_storage_cost = (
            0.5 * (terminal_deficit_1 + terminal_deficit_2) if is_terminal else 0.0
        )
        reward = -(
            self.reward_config.water * water_cost
            + self.reward_config.energy * energy_cost
            + self.reward_config.food * food_cost
            + self.reward_config.terminal_storage * terminal_storage_cost
        )

        tolerance = 1.0e-8
        violations = int(r1 < -tolerance or r1 > r1_cap + tolerance)
        violations += int(r2 < -tolerance or r2 > r2_cap + tolerance)
        violations += int(s1_next < smin1 - tolerance or s1_next > smax1 + tolerance)
        violations += int(s2_next < smin2 - tolerance or s2_next > smax2 + tolerance)
        violations += int(s1_next < reserve_floor_s1 - tolerance)
        violations += int(s2_next < reserve_floor_s2 - tolerance)
        violations += int(dn < -tolerance or dn > min(dn_cap, q_n, remaining_n) + tolerance)
        violations += int(dh < -tolerance or dh > min(dh_cap, q5, remaining_h) + tolerance)
        violations += int(self.state["used_n"] + dn > budget_n + tolerance)
        violations += int(self.state["used_h"] + dh > budget_h + tolerance)
        violations += int(float(pump_exec.sum()) > pump_available + tolerance)
        for k, diversion in enumerate((dn, dh)):
            violations += int(pump_exec[k] > pump_accepted[k] + tolerance)
            violations += int(
                self._pump_energy(diversion, pump_heads[k], pump_efficiencies[k])
                > pump_exec[k] + tolerance
            )

        self.state.update(
            s1=s1_next, s2=s2_next, r1=r1, r2=r2,
            river=np.asarray([q3, q_n, q5, q9]), intake_n=q_n, intake_h=q5, q9=q9,
            used_n=self.state["used_n"] + dn, used_h=self.state["used_h"] + dh,
            div_n=dn, div_h=dh, effective_n=effective_n, effective_h=effective_h,
            crop_alloc=crop_alloc, crop_deficit=crop_deficit,
            generation=np.asarray([e1, e2]), grid_import=grid_import, residual=residual,
            curtailment=curtailment, unserved=unserved, pump_demand=pump_requested,
            pump_accepted=pump_accepted, pump_alloc=pump_alloc,
            pump_shortfall=pump_shortfall, pump_exec=pump_exec,
            spill=np.asarray([spill1, spill2]),
        )

        projection_distance = float(np.linalg.norm(raw - np.asarray([r1, r2, dn, dh])))
        requested_diversion_total = float(diversion_requested.sum())
        delivered_diversion_total = dn + dh
        pumping_delivery_ratio = (
            delivered_diversion_total / requested_diversion_total
            if requested_diversion_total > 1.0e-8 else 1.0
        )
        energy_demand = ex["base"] + float(pump_requested.sum())
        energy_demand_satisfaction_ratio = 1.0 - safe_ratio(
            unserved + float(pump_shortfall.sum()), energy_demand
        )
        initial_storage_fraction = 0.5 * (
            initial_s1 / max(smax1, 1.0e-8) + initial_s2 / max(smax2, 1.0e-8)
        )
        current_storage_fraction = 0.5 * (
            s1_next / max(smax1, 1.0e-8) + s2_next / max(smax2, 1.0e-8)
        )
        storage_recovery_ratio = 0.5 * (
            min(1.0, s1_next / max(initial_s1, 1.0e-8))
            + min(1.0, s2_next / max(initial_s2, 1.0e-8))
        )
        record = {
            "year": self.year,
            "month": int(row["month [1-12]"]),
            "year_month": str(row["year_month [YYYY-MM]"]),
            "delay_scenario": self.delay_name,
            "mean_observation_delay": float(self.delays.mean()),
            "max_observation_delay": int(self.delays.max()),
            "disturbance": self.disturbance.name,
            "raw_R1": raw[0], "raw_R2": raw[1], "raw_D_N": raw[2], "raw_D_H": raw[3],
            "requested_D_N": dn_requested, "requested_D_H": dh_requested,
            "R1": r1, "R2": r2, "D_N": dn, "D_H": dh,
            "electricity_limited_D_N": dn_requested - dn,
            "electricity_limited_D_H": dh_requested - dh,
            "S1": s1_next, "S2": s2_next,
            "S1_fraction": s1_next / max(smax1, 1.0e-8),
            "S2_fraction": s2_next / max(smax2, 1.0e-8),
            "initial_storage_fraction": initial_storage_fraction,
            "mean_storage_fraction": current_storage_fraction,
            "storage_drawdown_from_initial_pp": 100.0 * (
                initial_storage_fraction - current_storage_fraction
            ),
            "storage_recovery_ratio": storage_recovery_ratio,
            "terminal_storage_target_fraction": storage_target_fraction,
            "terminal_target_S1": terminal_target_s1,
            "terminal_target_S2": terminal_target_s2,
            "reserve_floor_S1": reserve_floor_s1,
            "reserve_floor_S2": reserve_floor_s2,
            "storage_reserve_limited_R1": reserve_limited_r1,
            "storage_reserve_limited_R2": reserve_limited_r2,
            "storage_reserve_limited_release_total": reserve_limited_r1 + reserve_limited_r2,
            "terminal_storage_deficit_normalized": terminal_storage_cost,
            "spill_1": spill1, "spill_2": spill2,
            "Q_N": q_n, "Q_H": q5, "Q9": q9, "eco_threshold": eco_threshold,
            "effective_N": effective_n, "effective_H": effective_h,
            "hydro_E1": e1, "hydro_E2": e2, "renewable_E3": ex["renewable"],
            "electricity_import_E4": grid_import, "base_demand": ex["base"],
            "unserved_energy": unserved,
            "pump_demand_N": pump_requested[0], "pump_demand_H": pump_requested[1],
            "pump_accepted_N": pump_accepted[0], "pump_accepted_H": pump_accepted[1],
            "pump_executed_N": pump_exec[0], "pump_executed_H": pump_exec[1],
            "pump_capacity_shortfall_N": pump_capacity_shortfall[0],
            "pump_capacity_shortfall_H": pump_capacity_shortfall[1],
            "pump_supply_shortfall_N": pump_supply_shortfall[0],
            "pump_supply_shortfall_H": pump_supply_shortfall[1],
            "pump_shortfall_N": pump_shortfall[0], "pump_shortfall_H": pump_shortfall[1],
            "pumping_delivery_ratio": pumping_delivery_ratio,
            "energy_demand_satisfaction_ratio": energy_demand_satisfaction_ratio,
            "crop_requirement": float(requirements.sum()), "crop_deficit": cf,
            "C_W": cw, "C_E": ce, "C_F": cf, "C_S": terminal_storage_cost,
            "cost_water_normalized": water_cost,
            "cost_energy_normalized": energy_cost,
            "cost_food_normalized": food_cost,
            "cost_terminal_storage_normalized": terminal_storage_cost,
            "reward": reward,
            "projection_distance": projection_distance,
            "projected": float(projection_distance > 1.0e-8),
            "safety_violations": violations,
        }
        for crop in range(6):
            record[f"F{crop + 1}_requirement"] = requirements[crop]
            record[f"F{crop + 1}_allocated"] = crop_alloc[crop]
            record[f"F{crop + 1}_deficit"] = crop_deficit[crop]
        self.records.append(record)

        self.t += 1
        done = self.t >= len(self.rows)
        if not done:
            self.history.append(self._true_node_features())
            next_observation = self.observation()
        else:
            next_observation = self.history[-1].copy()
        return next_observation, float(reward), done, record.copy()

    def _true_node_features(self) -> np.ndarray:
        """Build dimensionless padded node features for all eight entity types."""

        row = self.current_row
        ex = self._exogenous(row)
        features = np.zeros((GRAPH.num_nodes, self.feature_dim), dtype=np.float32)
        month = int(row["month [1-12]"])
        sin_month = np.sin(2.0 * np.pi * (month - 1) / 12.0)
        cos_month = np.cos(2.0 * np.pi * (month - 1) / 12.0)

        def put(node: int, values: list[float]) -> None:
            clean = [clip_feature(v) for v in values[:9]]
            features[node, : len(clean)] = clean
            features[node, 9] = sin_month
            features[node, 10] = cos_month
            # Feature 11 is a zero-padded compatibility channel.

        cap1 = float(row["W1_total_capacity_reference [1e8 m3]"])
        cap2 = float(row["W2_total_capacity_reference [1e8 m3]"])
        rmax1 = float(row["W1_max_release_Rmax [1e8 m3 month-1]"])
        rmax2 = float(row["W2_max_release_Rmax [1e8 m3 month-1]"])
        put(0, [self.state["s1"] / cap1, ex["inflow1"] / rmax1, self.state["r1"] / rmax1,
                float(row["W1_reservoir_loss_L_res [1e8 m3 month-1]"]) / rmax1,
                float(row["W1_min_storage_Smin [1e8 m3]"]) / cap1, self.state["spill"][0] / rmax1,
                float(row["W1_head [m]"]) / 150.0])
        put(1, [self.state["s2"] / cap2, self.state["river"][0] / rmax2, self.state["r2"] / rmax2,
                float(row["W2_reservoir_loss_L_res [1e8 m3 month-1]"]) / rmax2,
                float(row["W2_min_storage_Smin [1e8 m3]"]) / cap2, self.state["spill"][1] / rmax2,
                float(row["W2_head [m]"]) / 150.0])

        gains = [ex["g3"], ex["g4"], ex["g5"], ex["g6"]]
        for offset in range(4):
            put(2 + offset, [self.state["river"][offset] / 40.0, gains[offset] / 15.0])

        req = self._crop_requirements(row)
        for node, district, req_slice in ((6, "n", slice(0, 3)), (7, "h", slice(3, 6))):
            prefix = "W7_Ningxia" if district == "n" else "W8_Hetao"
            dmax = float(row[f"{prefix}_channel_capacity_Dmax [1e8 m3 month-1]"])
            budget = float(row[f"{prefix}_annual_budget_B [1e8 m3 year-1]"])
            used = self.state[f"used_{district}"]
            put(node, [self.state[f"intake_{district}"] / max(dmax, 1.0),
                       self.state[f"div_{district}"] / max(dmax, 1.0),
                       self.state[f"effective_{district}"] / max(req[req_slice].sum(), 1.0),
                       max(0.0, budget - used) / max(budget, 1.0),
                       float(row[f"{prefix}_return_fraction_f [-]"]),
                       float(row[f"{prefix}_irrigation_efficiency_eta [-]"]),
                       req[req_slice].sum() / 15.0])
        threshold = float(row["W9_ecological_threshold [1e8 m3 month-1]"])
        put(8, [self.state["q9"] / max(threshold, 1.0), threshold / 10.0,
                max(0.0, threshold - self.state["q9"]) / max(threshold, 1.0)])

        design = [float(row["E1_design_generation_reference [GWh month-1]"]), float(row["E2_design_generation_reference [GWh month-1]"])]
        beta = [float(row["E1_hydropower_beta1 [GWh per 1e8 m3]"]), float(row["E2_hydropower_beta2 [GWh per 1e8 m3]"])]
        for k in range(2):
            put(9 + k, [self.state["generation"][k] / max(design[k], 1.0), beta[k] / 30.0])
        put(11, [ex["renewable"] / 1000.0, float(row["E3_availability [-]"])])
        import_limit = float(row["E4_import_limit [GWh month-1]"])
        put(12, [self.state["grid_import"] / max(import_limit, 1.0), import_limit / 1000.0,
                 float(row["E4_availability [-]"])])
        put(13, [self.state["residual"] / 1000.0, self.state["curtailment"] / 1000.0,
                 self.state["unserved"] / max(ex["base"], 1.0), ex["base"] / 1000.0])
        interface = [float(row["E6_Ningxia_interface_capacity [GWh month-1]"]), float(row["E7_Hetao_interface_capacity [GWh month-1]"])]
        for k in range(2):
            put(14 + k, [self.state["pump_demand"][k] / max(interface[k], 1.0),
                         self.state["pump_alloc"][k] / max(interface[k], 1.0),
                         self.state["pump_shortfall"][k] / max(interface[k], 1.0)])
        pump_heads = [float(row["E8_Ningxia_pump_head [m]"]), float(row["E9_Hetao_pump_head [m]"])]
        pump_eff = [float(row["E8_Ningxia_pump_efficiency [-]"]), float(row["E9_Hetao_pump_efficiency [-]"])]
        pump_caps = [float(row["E8_Ningxia_pump_capacity_Ebar8 [GWh month-1]"]), float(row["E9_Hetao_pump_capacity_Ebar9 [GWh month-1]"])]
        for k in range(2):
            put(16 + k, [pump_heads[k] / 300.0, pump_eff[k], self.state["pump_exec"][k] / max(pump_caps[k], 1.0),
                         pump_caps[k] / 100.0])

        for crop in range(6):
            area = float(row[f"F{crop + 1}_area_A [ha]"])
            coefficient = float(row[f"F{crop + 1}_crop_coefficient_k [-]"])
            requirement = req[crop]
            put(18 + crop, [area / 300000.0, coefficient / 1.3, requirement / 10.0,
                            self.state["crop_alloc"][crop] / max(requirement, 1.0),
                            self.state["crop_deficit"][crop] / max(requirement, 1.0),
                            0.0 if crop < 3 else 1.0])
        return features

    def physical_action_scales(self) -> np.ndarray:
        row = self.current_row
        return np.asarray(
            [
                row["W1_max_release_Rmax [1e8 m3 month-1]"],
                row["W2_max_release_Rmax [1e8 m3 month-1]"],
                row["W7_Ningxia_channel_capacity_Dmax [1e8 m3 month-1]"],
                row["W8_Hetao_channel_capacity_Dmax [1e8 m3 month-1]"],
            ],
            dtype=np.float64,
        )
