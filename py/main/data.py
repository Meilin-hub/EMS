"""Input loading and strict schema checks for ``py/filled.csv``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "year [YYYY]", "month [1-12]", "year_month [YYYY-MM]",
    "W1_storage_begin_S [1e8 m3]", "W1_storage_next_Splus [1e8 m3]",
    "W1_inflow_I [1e8 m3 month-1]", "W1_reservoir_loss_L_res [1e8 m3 month-1]",
    "W1_min_storage_Smin [1e8 m3]", "W1_max_release_Rmax [1e8 m3 month-1]",
    "W1_total_capacity_reference [1e8 m3]", "E1_hydropower_beta1 [GWh per 1e8 m3]",
    "W2_storage_begin_S [1e8 m3]", "W2_storage_next_Splus [1e8 m3]",
    "W2_reservoir_loss_L_res [1e8 m3 month-1]", "W2_min_storage_Smin [1e8 m3]",
    "W2_max_release_Rmax [1e8 m3 month-1]", "W2_total_capacity_reference [1e8 m3]",
    "E2_hydropower_beta2 [GWh per 1e8 m3]",
    "W3_net_reach_gain_G [1e8 m3 month-1]", "W4_net_reach_gain_G [1e8 m3 month-1]",
    "W5_net_reach_gain_G [1e8 m3 month-1]", "W6_net_reach_gain_G [1e8 m3 month-1]",
    "W9_ecological_threshold [1e8 m3 month-1]",
    "W7_Ningxia_return_fraction_f [-]", "W7_Ningxia_irrigation_efficiency_eta [-]",
    "W7_Ningxia_channel_capacity_Dmax [1e8 m3 month-1]",
    "W7_Ningxia_annual_budget_B [1e8 m3 year-1]",
    "W8_Hetao_return_fraction_f [-]", "W8_Hetao_irrigation_efficiency_eta [-]",
    "W8_Hetao_channel_capacity_Dmax [1e8 m3 month-1]",
    "W8_Hetao_annual_budget_B [1e8 m3 year-1]",
    "E3_nonhydro_renewable_generation [GWh month-1]", "E3_availability [-]",
    "E4_import_limit [GWh month-1]", "E4_availability [-]",
    "E5_base_demand_L_base [GWh month-1]",
    "E6_Ningxia_interface_capacity [GWh month-1]",
    "E7_Hetao_interface_capacity [GWh month-1]",
    "E8_Ningxia_pump_head [m]", "E8_Ningxia_pump_efficiency [-]",
    "E8_Ningxia_pump_capacity_Ebar8 [GWh month-1]",
    "E9_Hetao_pump_head [m]", "E9_Hetao_pump_efficiency [-]",
    "E9_Hetao_pump_capacity_Ebar9 [GWh month-1]",
)


@dataclass(frozen=True)
class WEFData:
    frame: pd.DataFrame
    years: tuple[int, ...]

    def year(self, year: int) -> pd.DataFrame:
        rows = self.frame[self.frame["year [YYYY]"] == year].copy()
        if len(rows) != 12:
            raise ValueError(f"Year {year} must contain 12 monthly rows; found {len(rows)}")
        return rows.reset_index(drop=True)


def load_wef_data(path: Path) -> WEFData:
    if not path.exists():
        raise FileNotFoundError(f"WEF input not found: {path}")
    frame = pd.read_csv(path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError("Input schema is missing required columns: " + ", ".join(missing_columns))

    numeric_required = [c for c in REQUIRED_COLUMNS if c != "year_month [YYYY-MM]"]
    for column in numeric_required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    bad = [column for column in numeric_required if frame[column].isna().any()]
    if bad:
        raise ValueError("Required physical inputs contain blanks/non-numeric values: " + ", ".join(bad))

    for crop in range(1, 7):
        for suffix in (
            "area_A [ha]", "crop_coefficient_k [-]",
            "water_requirement_W [1e8 m3 month-1]",
        ):
            column = f"F{crop}_{suffix}"
            if column not in frame.columns or pd.to_numeric(frame[column], errors="coerce").isna().any():
                raise ValueError(f"Required crop field is invalid: {column}")
            frame[column] = pd.to_numeric(frame[column], errors="raise")

    frame = frame.sort_values(["year [YYYY]", "month [1-12]"]).reset_index(drop=True)
    years = tuple(int(y) for y in sorted(frame["year [YYYY]"].unique()))
    expected = pd.period_range(f"{years[0]}-01", f"{years[-1]}-12", freq="M").astype(str).tolist()
    actual = frame["year_month [YYYY-MM]"].astype(str).tolist()
    if actual != expected:
        raise ValueError("Input must be one gap-free chronological monthly series")

    bounded = {
        "W7_Ningxia_return_fraction_f [-]": (0.0, 1.0),
        "W8_Hetao_return_fraction_f [-]": (0.0, 1.0),
        "W7_Ningxia_irrigation_efficiency_eta [-]": (0.0, 1.0),
        "W8_Hetao_irrigation_efficiency_eta [-]": (0.0, 1.0),
    }
    for column, (low, high) in bounded.items():
        if not frame[column].between(low, high).all():
            raise ValueError(f"{column} must lie in [{low}, {high}]")
    return WEFData(frame=frame, years=years)


def split_audit(data: WEFData, train: tuple[int, ...], validation: tuple[int, ...], test: tuple[int, ...]) -> None:
    groups = {"train": set(train), "validation": set(validation), "test": set(test)}
    for a, a_years in groups.items():
        unknown = a_years.difference(data.years)
        if unknown:
            raise ValueError(f"{a} years absent from data: {sorted(unknown)}")
        for b, b_years in groups.items():
            if a < b and a_years.intersection(b_years):
                raise ValueError(f"Year leakage between {a} and {b}")


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(abs(denominator), 1.0e-8))


def clip_feature(value: float, limit: float = 5.0) -> float:
    return float(np.clip(np.nan_to_num(value), -limit, limit))

