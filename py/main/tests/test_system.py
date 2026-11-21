"""Regression tests for the retained WEF delay experiments."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from main.config import ExperimentConfig
from main.data import WEFData, load_wef_data, split_audit
from main.environment import MAX_DELAY, YellowRiverWEFEnv
from main.graph import GRAPH
from main.model import HeteroGNNActorCritic, build_model
from main.sector_delay import POLICY_DELAY_STUDIES, delay_definitions
from main.statistical_analysis import build_analysis_units, descriptive_statistics, paired_method_effects


class TestWEFSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = ExperimentConfig()
        cls.data = load_wef_data(cls.config.data_path)

    @staticmethod
    def make_outcome(method: str, seed: int, year: int, scenario: str, offset: float) -> dict[str, float | int | str]:
        return {
            "method": method,
            "variant_key": method.lower().replace("-", "_"),
            "delay_scenario": scenario,
            "seed": seed,
            "year": year,
            "mean_monthly_reward": -(0.50 + offset),
            "ecological_flow_reliability": 0.90 - offset / 10.0,
            "ecological_deficit_total": 1.0 + offset,
            "irrigation_satisfaction_ratio": 0.30 - offset / 10.0,
            "crop_deficit_total": 8.0 + offset,
            "energy_demand_satisfaction_ratio": 0.80 - offset / 10.0,
            "electricity_import_total": 100.0 + offset,
            "storage_reserve_limited_release_total": 20.0 + offset,
            "storage_recovery_ratio": 1.0,
            "storage_drawdown_from_initial_pp": 0.0,
            "projection_rate": 1.0,
            "safety_violations": 0.0,
        }

    def test_input_and_graph_contract(self) -> None:
        self.assertEqual(len(self.data.frame), 252)
        self.assertEqual(self.data.years, tuple(range(2001, 2022)))
        self.assertEqual(GRAPH.num_nodes, 24)
        self.assertEqual(GRAPH.num_types, 8)
        self.assertGreater(GRAPH.edge_index.shape[1], 24)
        split_audit(
            self.data,
            self.config.train_years,
            self.config.validation_years,
            self.config.test_years,
        )
        self.assertEqual(len(self.config.seeds), 10)
        self.assertEqual(len(set(self.config.seeds)), 10)

    def test_uniform_sector_and_node_delay_contracts(self) -> None:
        env = YellowRiverWEFEnv(self.data, self.config.reward, self.config.physical, seed=10)
        tau0 = env.reset(2019, delay_scenario="tau0")
        tau5 = env.reset(2019, delay_scenario="tau5")
        self.assertEqual(tau0.shape, (24, 12))
        self.assertEqual(tau5.shape, (24, 12))
        np.testing.assert_allclose(tau0[:, -1], 0.0)
        np.testing.assert_allclose(tau5[:, -1], 0.0)
        self.assertTrue(np.all(env.delays == MAX_DELAY))
        env.reset(2019, delay_scenario="sector_water_tau3")
        self.assertTrue(np.all(env.delays[:9] == 3))
        self.assertTrue(np.all(env.delays[9:] == 0))
        env.reset(2019, delay_scenario="node_E8_tau4")
        self.assertEqual(int(env.delays[16]), 4)
        self.assertEqual(int(env.delays.sum()), 4)
        definitions = delay_definitions()
        self.assertEqual(len(definitions), 136)
        self.assertEqual(int((definitions["scenario_type"] == "sector").sum()), 15)
        self.assertEqual(int((definitions["scenario_type"] == "node").sum()), 120)
        with self.assertRaisesRegex(ValueError, "Delay scenario"):
            env.reset(2019, delay_scenario="mixed")

    def test_location_sensitivity_studies_are_matched_across_algorithms(self) -> None:
        self.assertEqual(set(POLICY_DELAY_STUDIES), {"gnn_ppo", "ppo"})
        gnn = POLICY_DELAY_STUDIES["gnn_ppo"]
        ppo = POLICY_DELAY_STUDIES["ppo"]
        self.assertEqual((gnn.experiment_number, ppo.experiment_number), (3, 4))
        self.assertEqual((gnn.model_name, ppo.model_name), ("gnn", "ppo"))
        self.assertNotEqual(gnn.directory_name, ppo.directory_name)
        self.assertEqual(gnn.figure_numbers, (3, 4, 5))
        self.assertEqual(ppo.figure_numbers, (6, 7, 8))

    def test_publication_statistics_use_training_seed_means(self) -> None:
        rows = []
        for seed in self.config.seeds:
            for year in self.config.test_years:
                rows.append(self.make_outcome("PPO", seed, year, "tau0", 0.05))
                rows.append(self.make_outcome("GNN-PPO", seed, year, "tau0", 0.00))
        years, seeds = build_analysis_units(pd.DataFrame(rows), ("PPO", "GNN-PPO"), ("tau0",))
        self.assertEqual(len(years), 60)
        self.assertEqual(len(seeds), 20)
        summary = descriptive_statistics(seeds)
        self.assertTrue((summary["n_training_seeds"] == 10).all())
        effects = paired_method_effects(years, "test", "PPO", "GNN-PPO", ("tau0",), bootstrap_seed=1)
        cost = effects[effects["metric"] == "cost"].iloc[0]
        self.assertGreater(float(cost["mean_improvement"]), 0.0)
        self.assertEqual(int(cost["paired_training_seeds"]), 10)
        self.assertEqual(int(cost["paired_seed_year_records"]), 30)

    def test_projection_enforces_annual_constraints(self) -> None:
        env = YellowRiverWEFEnv(self.data, self.config.reward, self.config.physical, seed=11)
        env.reset(2020)
        done = False
        while not done:
            _, reward, done, info = env.step(np.ones(4))
            self.assertTrue(np.isfinite(reward))
            self.assertEqual(info["safety_violations"], 0)
            self.assertGreaterEqual(info["S1_fraction"], 0.0)
            self.assertLessEqual(info["S1_fraction"], 1.0)
            self.assertGreaterEqual(info["S2_fraction"], 0.0)
            self.assertLessEqual(info["S2_fraction"], 1.0)
        frame = pd.DataFrame(env.records)
        row = self.data.year(2020).iloc[0]
        self.assertLessEqual(
            frame["D_N"].sum(), row["W7_Ningxia_annual_budget_B [1e8 m3 year-1]"] + 1.0e-8
        )
        self.assertLessEqual(
            frame["D_H"].sum(), row["W8_Hetao_annual_budget_B [1e8 m3 year-1]"] + 1.0e-8
        )

    def test_electricity_physically_limits_delivered_irrigation_water(self) -> None:
        frame = self.data.frame.copy()
        first_month = (frame["year [YYYY]"] == 2019) & (frame["month [1-12]"] == 1)
        frame.loc[first_month, "E3_nonhydro_renewable_generation [GWh month-1]"] = 0.0
        frame.loc[first_month, "E4_import_limit [GWh month-1]"] = 0.0
        stressed_data = WEFData(frame=frame, years=self.data.years)
        env = YellowRiverWEFEnv(stressed_data, self.config.reward, self.config.physical, seed=14)
        env.reset(2019)
        _, _, _, info = env.step(np.asarray([0.0, 0.0, 1.0, 1.0]))
        self.assertGreater(info["requested_D_N"] + info["requested_D_H"], 0.0)
        self.assertAlmostEqual(info["pump_executed_N"] + info["pump_executed_H"], 0.0)
        self.assertAlmostEqual(info["D_N"] + info["D_H"], 0.0)

    def test_terminal_storage_guard_prevents_end_horizon_depletion(self) -> None:
        env = YellowRiverWEFEnv(self.data, self.config.reward, self.config.physical, seed=15)
        env.reset(2019, disturbance="extreme")
        done = False
        while not done:
            _, _, done, info = env.step(np.asarray([1.0, 1.0, 0.0, 0.0]))
        self.assertGreater(sum(r["storage_reserve_limited_release_total"] for r in env.records), 0.0)
        self.assertGreaterEqual(info["S1"] + 1.0e-8, info["terminal_target_S1"])
        self.assertGreaterEqual(info["S2"] + 1.0e-8, info["terminal_target_S2"])

    def test_policy_models_share_observation_contract(self) -> None:
        env = YellowRiverWEFEnv(self.data, self.config.reward, self.config.physical, seed=13)
        observation = torch.as_tensor(env.reset(2019), dtype=torch.float32)
        gnn = HeteroGNNActorCritic(feature_dim=12, hidden_dim=32, layers=2)
        action, log_probability, value = gnn.act(observation)
        self.assertEqual(tuple(action.shape), (1, 4))
        self.assertTrue(torch.all((action > 0) & (action < 1)))
        (-(log_probability.mean() + value.mean())).backward()
        self.assertGreater(
            sum(float(parameter.grad.abs().sum()) for parameter in gnn.parameters() if parameter.grad is not None),
            0.0,
        )
        ppo = build_model("ppo", feature_dim=12, hidden_dim=32, gnn_layers=2)
        ppo_action, _, ppo_value = ppo.act(observation)
        self.assertEqual(tuple(ppo_action.shape), (1, 4))
        self.assertEqual(tuple(ppo_value.shape), (1,))


if __name__ == "__main__":
    unittest.main()
