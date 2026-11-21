#!/usr/bin/env python3
"""Build a literature-informed, audit-ready WEF input CSV without overwriting the source."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "py" / "upper_yellow_river_wef_monthly_inputs_2001_2021.csv"
OUTPUT = ROOT / "py" / "upper_yellow_river_wef_monthly_inputs_2001_2021_filled.csv"
PROVENANCE = ROOT / "py" / "upper_yellow_river_wef_field_provenance.csv"
REPORT = ROOT / "py" / "upper_yellow_river_wef_data_fill_report.md"

HESS_ROOT = Path("/private/tmp/yellow_river_hess_unpacked/Irrig_Dam_YellowRiver_Yin_et_al")
RESCN_ROOT = Path("/private/tmp/rescn_subset")

HESS_ARTICLE = "https://hess.copernicus.org/articles/25/1133/2021/"
HESS_DATA = "https://doi.org/10.5281/zenodo.3979053"
RESCN_ARTICLE = "https://essd.copernicus.org/articles/15/2781/2023/"
RESCN_DATA = "https://doi.org/10.5281/zenodo.7664489"
NIXIA_BULLETINS = "https://slt.nx.gov.cn/xxgk_281/fdzdgknr/gbxx/szygb/"
NIXIA_EFFICIENCY = "https://www.chinawater.com.cn/newscenter/df/nx/201512/t20151228_412242.html"
NIXIA_CAPACITY = "https://slt.nx.gov.cn/slxc/gzdt/slyw/202104/t20210409_2733861.html"
NIXIA_PUMP = "https://www.hongsibu.gov.cn/zjhsb/lyjd/202305/t20230505_4053085.html"
NIXIA_YEARBOOK = "https://www.nx.gov.cn/zwgk/zfxxgk/fdzdgknr/tjxx_40901/tjnj/index.html"
NIXIA_YEARBOOK_MIRROR = "https://nxdata.com.cn/files_nx_pub/html/tjnj/"
HETAO_ARTICLE = "https://www.mdpi.com/2073-4395/15/7/1654"
HETAO_CANAL = "https://www.sciencedirect.com/science/article/pii/S037837742200508X"
LONGYANGXIA_REFERENCE = "https://obor.nea.gov.cn/detail/6852.html"
LONGYANGXIA_HEAD = "https://www.yrcti.edu.cn/slgcxy/info/1057/5306.htm"
ECO_FLOW = "https://doi.org/10.3390/ijerph182010956"


def fnum(value: str | float | int | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "NA":
        return None
    return float(text)


def fmt(value: float | int | None, digits: int = 6) -> str:
    if value is None or not math.isfinite(float(value)):
        return ""
    text = f"{float(value):.{digits}f}".rstrip("0").rstrip(".")
    return "0" if text in {"-0", ""} else text


def lerp(a: float, b: float, ratio: float) -> float:
    return a + (b - a) * ratio


def interpolate_years(anchors: dict[int, float], start: int = 2001, end: int = 2021) -> dict[int, float]:
    years = sorted(anchors)
    result: dict[int, float] = {}
    for year in range(start, end + 1):
        if year in anchors:
            result[year] = anchors[year]
            continue
        left = max((y for y in years if y < year), default=years[0])
        right = min((y for y in years if y > year), default=years[-1])
        if left == right:
            result[year] = anchors[left]
        else:
            result[year] = lerp(anchors[left], anchors[right], (year - left) / (right - left))
    return result


def read_monthly_csv(path: Path, value_key: str) -> tuple[dict[tuple[int, int], float], dict[int, float]]:
    values: dict[tuple[int, int], float] = {}
    by_month: dict[int, list[float]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            year, month = (int(x) for x in row["dates"].split("-")[:2])
            value = fnum(row[value_key])
            if value is None:
                continue
            values[(year, month)] = value
            if 2001 <= year <= 2020:
                by_month[month].append(value)
    climatology = {month: sum(v) / len(v) for month, v in by_month.items()}
    return values, climatology


def read_hess_discharge() -> tuple[list[list[float | None]], dict[int, list[float]]]:
    path = HESS_ROOT / "tab" / "dis.yl.m.hh"
    matrix: list[list[float | None]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        matrix.append([fnum(x) for x in line.split()])
    climatology: dict[int, list[float]] = {}
    for row_index in (22, 16, 20):  # Tangnaihai, Lanzhou, Toudaoguai (1-based 23, 17, 21)
        vals = [[] for _ in range(12)]
        for year in range(2001, 2015):
            for month in range(1, 13):
                idx = (year - 1982) * 12 + month - 1
                value = matrix[row_index][idx]
                if value is not None:
                    vals[month - 1].append(value)
        climatology[row_index] = [sum(x) / len(x) for x in vals]
    return matrix, climatology


def discharge_volume(
    matrix: list[list[float | None]], climatology: dict[int, list[float]], row_index: int, year: int, month: int
) -> float:
    value: float | None = None
    if year <= 2014:
        idx = (year - 1982) * 12 + month - 1
        value = matrix[row_index][idx]
    if value is None:
        value = climatology[row_index][month - 1]
    return value * 30 * 86400 / 1e8


def storage_series(deltas: list[float], lower: float, upper: float) -> tuple[list[float], list[float]]:
    cumulative = [0.0]
    for delta in deltas:
        cumulative.append(cumulative[-1] + delta)
    span_mid = (min(cumulative) + max(cumulative)) / 2
    physical_mid = (lower + upper) / 2
    begins = [physical_mid + cumulative[i] - span_mid for i in range(len(deltas))]
    ends = [physical_mid + cumulative[i + 1] - span_mid for i in range(len(deltas))]
    return begins, ends


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        headers = list(reader.fieldnames or [])
        source_rows = list(reader)
    rows = [dict(row) for row in source_rows]

    before_blank = sum(not str(row[col]).strip() for row in source_rows for col in headers)
    original_by_year = {int(row["year [YYYY]"]): row for row in source_rows if int(row["month [1-12]"]) == 1}

    # Annual Ningxia water data. Existing reported values are preserved; 2009-2017 come from official bulletins.
    nx_water: dict[int, tuple[float, float]] = {}
    for year, row in original_by_year.items():
        gross = fnum(row["W7_Ningxia_annual_diversion_reported [1e8 m3 year-1]"])
        monthly_drain = fnum(row["W7_Ningxia_drainage_return [1e8 m3 month-1]"])
        if gross is not None and monthly_drain is not None:
            nx_water[year] = (gross, monthly_drain * 12)
    nx_water.update(
        {
            2009: (66.057, 30.221),
            2010: (64.599, 33.189),
            2011: (65.238, 34.008),
            2012: (60.890, 33.456),
            2013: (63.300, 33.367),
            2014: (61.613, 31.449),
            2015: (62.032, 31.048),
            2016: (56.090, 28.516),
            2017: (56.668, 29.072),
        }
    )

    nx_eta = interpolate_years(
        {
            2001: 0.38,
            2005: 0.38,
            2009: 0.42,
            2014: 0.47,
            2015: 0.501,
            2016: 0.511,
            2017: 0.524,
            2018: 0.535,
            2019: 0.543,
            2020: 0.551,
            2021: 0.561,
        }
    )

    # Hetao diversion/drainage values were digitized from Figure 6 of the cited article (10^9 m3 -> 10^8 m3).
    he_div_anchors = {
        2002: 49.69,
        2006: 45.63,
        2007: 45.55,
        2008: 43.09,
        2009: 51.80,
        2010: 45.88,
        2011: 47.40,
        2012: 38.37,
        2013: 45.21,
        2014: 44.70,
        2015: 46.05,
        2016: 43.52,
        2017: 44.95,
        2018: 42.92,
        2019: 42.92,
        2021: 40.40,
    }
    he_div = interpolate_years({2001: he_div_anchors[2002], **he_div_anchors})
    he_drain = interpolate_years(
        {
            2001: 2.28,
            2006: 2.28,
            2007: 3.21,
            2008: 3.55,
            2009: 2.70,
            2010: 2.96,
            2011: 2.37,
            2012: 3.30,
            2013: 2.79,
            2014: 3.04,
            2015: 2.45,
            2016: 2.79,
            2017: 2.11,
            2018: 2.96,
            2019: 2.96,
            2021: 2.96,
        }
    )
    he_eta = interpolate_years({2001: 0.40, 2020: 0.467, 2021: 0.47})

    # Statistical-yearbook observations with interpolation/backcasting only where an annual table was not located.
    nx_crop_anchors = {
        2001: (276018.0, 178388.0, 71249.0),
        2005: (276018.0, 178388.0, 71249.0),
        2006: (227440.0, 174516.533333, 81664.066667),
        2008: (204287.0, 208525.0, 80289.0),
        2009: (218455.0, 215078.0, 78243.0),
        2011: (202102.0, 231797.0, 83942.0),
        2013: (148825.0, 262021.0, 82144.0),
        2014: (127469.0, 288756.0, 78044.0),
        2016: (126191.0, 296894.0, 75052.0),
        2018: (128593.0, 310793.0, 78014.0),
        2019: (107773.0, 299777.0, 68054.0),
        2020: (92926.666667, 322726.666667, 60813.333333),
        2021: (67040.0, 367420.0, 50833.333333),
    }
    nx_wheat = interpolate_years({y: v[0] for y, v in nx_crop_anchors.items()})
    nx_maize = interpolate_years({y: v[1] for y, v in nx_crop_anchors.items()})
    nx_rice = interpolate_years({y: v[2] for y, v in nx_crop_anchors.items()})

    he_crop_observed = {
        2000: (241.9, 70.4, 141.0),
        2001: (249.2, 83.8, 153.2),
        2002: (244.3, 93.5, 179.9),
        2003: (132.5, 83.8, 251.6),
        2004: (212.7, 112.9, 207.9),
        2005: (222.5, 114.1, 203.0),
        2006: (193.3, 120.2, 210.3),
        2007: (154.4, 144.5, 211.5),
        2008: (130.1, 144.5, 232.2),
        2009: (164.1, 139.7, 252.8),
        2010: (100.9, 142.1, 279.6),
        2011: (66.9, 143.3, 320.9),
        2012: (97.2, 170.1, 275.9),
        2013: (52.3, 181.0, 325.8),
        2014: (51.1, 177.4, 336.7),
        2015: (49.8, 199.2, 295.4),
        2016: (51.1, 157.9, 314.8),
        2017: (55.9, 187.1, 277.2),
        2018: (54.7, 189.5, 279.6),
        2021: (35.4, 318.4, 238.993333),
    }
    he_wheat = interpolate_years({y: v[0] * 1000 for y, v in he_crop_observed.items() if y >= 2001})
    he_maize = interpolate_years({y: v[1] * 1000 for y, v in he_crop_observed.items() if y >= 2001})
    he_sunflower = interpolate_years({y: v[2] * 1000 for y, v in he_crop_observed.items() if y >= 2001})

    evap1, evap1_clim = read_monthly_csv(RESCN_ROOT / "evap_5992.csv", "evaphs")
    evap2, evap2_clim = read_monthly_csv(RESCN_ROOT / "evap_5089.csv", "evaphs")
    area1, area1_clim = read_monthly_csv(RESCN_ROOT / "swa_5992.csv", "water_area")
    area2, area2_clim = read_monthly_csv(RESCN_ROOT / "swa_5089.csv", "water_area")
    rwsc1, rwsc1_clim = read_monthly_csv(RESCN_ROOT / "rwsc_5992.csv", "SATVariation")

    def res_value(values: dict[tuple[int, int], float], clim: dict[int, float], year: int, month: int) -> float:
        return values.get((year, month), clim[month])

    loss1 = {}
    loss2 = {}
    deltas1 = []
    for year in range(2001, 2022):
        for month in range(1, 13):
            loss1[(year, month)] = res_value(evap1, evap1_clim, year, month) * res_value(area1, area1_clim, year, month) * 30 / 100000
            loss2[(year, month)] = res_value(evap2, evap2_clim, year, month) * res_value(area2, area2_clim, year, month) * 30 / 100000
            deltas1.append(res_value(rwsc1, rwsc1_clim, year, month) * 10)  # km3 -> 10^8 m3

    lyx_delta = [float(x) for x in (HESS_ROOT / "tab" / "dam" / "lyx_mon_8713").read_text().split()]
    ljx_delta = [float(x) for x in (HESS_ROOT / "tab" / "dam" / "ljx_mon_8713").read_text().split()]
    deltas2 = [ljx_delta[(i % 12)] for i in range(252)]
    w1_begin, w1_next = storage_series(deltas1, 53.47, 247.0)
    w2_begin, w2_next = storage_series(deltas2, 15.5, 57.0)

    discharge, discharge_clim = read_hess_discharge()
    # Seasonal W3 proxy from reservoir balances and Lanzhou flow; W4-W6 share an inseparable aggregate residual.
    g3_samples: dict[int, list[float]] = defaultdict(list)
    g_down_samples: dict[int, list[float]] = defaultdict(list)
    for year in range(2001, 2015):
        gross_nx, drain_nx = nx_water[year]
        for month in range(1, 13):
            tang = discharge_volume(discharge, discharge_clim, 22, year, month)
            lanzhou = discharge_volume(discharge, discharge_clim, 16, year, month)
            tdg = discharge_volume(discharge, discharge_clim, 20, year, month)
            release1_proxy = tang - lyx_delta[month - 1] - loss1[(year, month)]
            inflow2_proxy = lanzhou + ljx_delta[month - 1] + loss2[(year, month)]
            g3_samples[month].append(inflow2_proxy - release1_proxy)
            nx_net = (gross_nx - drain_nx) / 12
            he_net = (he_div[year] - he_drain[year]) / 12
            g_down_samples[month].append(tdg - lanzhou + nx_net + he_net)
    g3_clim = {m: sum(v) / len(v) for m, v in g3_samples.items()}
    g_down_clim = {m: sum(v) / len(v) for m, v in g_down_samples.items()}

    solar_values = [
        (fnum(r["Ningxia_solar_radiation [MJ m-2 day-1]"]) + fnum(r["Hetao_solar_radiation [MJ m-2 day-1]"])) / 2
        for r in rows
    ]
    wind_values = [
        (fnum(r["Ningxia_WS2M [m s-1]"]) + fnum(r["Hetao_WS2M [m s-1]"])) / 2
        for r in rows
    ]
    solar_min, solar_max = min(solar_values), max(solar_values)
    wind_min, wind_max = min(wind_values), max(wind_values)

    nx_dmax = 750 * 30 * 86400 / 1e8
    he_dmax = 565 * 30 * 86400 / 1e8
    beta1 = 1000 * 9.81 * 122 * 0.90 * 1e8 / 3.6e12
    beta2 = 1000 * 9.81 * 100 * 0.90 * 1e8 / 3.6e12
    rmax1 = 1280 * 1e6 / (1000 * 9.81 * 122 * 0.90) * 30 * 86400 / 1e8
    rmax2 = 1225 * 1e6 / (1000 * 9.81 * 100 * 0.90) * 30 * 86400 / 1e8
    ebar8 = (1000 * 9.81 * 28 * 300 / 0.75) / 1e6 * 720 / 1000
    ebar9 = (1000 * 9.81 * 565 * 5 / 0.75) / 1e6 * 720 / 1000
    eco_threshold = 125 * 30 * 86400 / 1e8

    for idx, row in enumerate(rows):
        year = int(row["year [YYYY]"])
        month = int(row["month [1-12]"])
        row["record_status [-]"] = "mixed_observed_derived_proxy_algorithm_fields_blank"

        row["W1_storage_begin_S [1e8 m3]"] = fmt(w1_begin[idx])
        row["W1_storage_next_Splus [1e8 m3]"] = fmt(w1_next[idx])
        row["W1_inflow_I [1e8 m3 month-1]"] = fmt(discharge_volume(discharge, discharge_clim, 22, year, month))
        row["W1_reservoir_loss_L_res [1e8 m3 month-1]"] = fmt(loss1[(year, month)])
        row["W1_min_storage_Smin [1e8 m3]"] = fmt(53.47)
        row["W1_max_release_Rmax [1e8 m3 month-1]"] = fmt(rmax1)
        row["W1_head [m]"] = fmt(122)
        row["E1_hydropower_beta1 [GWh per 1e8 m3]"] = fmt(beta1)

        row["W2_storage_begin_S [1e8 m3]"] = fmt(w2_begin[idx])
        row["W2_storage_next_Splus [1e8 m3]"] = fmt(w2_next[idx])
        row["W2_reservoir_loss_L_res [1e8 m3 month-1]"] = fmt(loss2[(year, month)])
        row["W2_min_storage_Smin [1e8 m3]"] = fmt(15.5)
        row["W2_max_release_Rmax [1e8 m3 month-1]"] = fmt(rmax2)
        row["W2_head [m]"] = fmt(100)
        row["E2_hydropower_beta2 [GWh per 1e8 m3]"] = fmt(beta2)

        row["W3_net_reach_gain_G [1e8 m3 month-1]"] = fmt(g3_clim[month])
        for col in (
            "W4_net_reach_gain_G [1e8 m3 month-1]",
            "W5_net_reach_gain_G [1e8 m3 month-1]",
            "W6_net_reach_gain_G [1e8 m3 month-1]",
        ):
            row[col] = fmt(g_down_clim[month] / 3)
        row["W9_Toudaoguai_flow_Q [1e8 m3 month-1]"] = fmt(discharge_volume(discharge, discharge_clim, 20, year, month))
        row["W9_ecological_threshold [1e8 m3 month-1]"] = fmt(eco_threshold)

        nx_gross, nx_drain = nx_water[year]
        nx_gross_m, nx_drain_m = nx_gross / 12, nx_drain / 12
        row["W7_Ningxia_gross_diversion_D [1e8 m3 month-1]"] = fmt(nx_gross_m)
        row["W7_Ningxia_annual_diversion_reported [1e8 m3 year-1]"] = fmt(nx_gross)
        row["W7_Ningxia_drainage_return [1e8 m3 month-1]"] = fmt(nx_drain_m)
        row["W7_Ningxia_net_consumption [1e8 m3 month-1]"] = fmt(nx_gross_m - nx_drain_m)
        row["W7_Ningxia_return_fraction_f [-]"] = fmt(nx_drain / nx_gross)
        row["W7_Ningxia_irrigation_efficiency_eta [-]"] = fmt(nx_eta[year])
        row["W7_Ningxia_field_effective_supply_V [1e8 m3 month-1]"] = fmt(nx_eta[year] * nx_gross_m)
        row["W7_Ningxia_channel_capacity_Dmax [1e8 m3 month-1]"] = fmt(nx_dmax)
        row["W7_Ningxia_annual_budget_B [1e8 m3 year-1]"] = fmt(nx_gross)

        he_gross, he_ret = he_div[year], he_drain[year]
        he_gross_m, he_ret_m = he_gross / 12, he_ret / 12
        row["W8_Hetao_gross_diversion_D [1e8 m3 month-1]"] = fmt(he_gross_m)
        row["W8_Hetao_annual_diversion_reported [1e8 m3 year-1]"] = fmt(he_gross)
        row["W8_Hetao_drainage_return [1e8 m3 month-1]"] = fmt(he_ret_m)
        row["W8_Hetao_net_consumption [1e8 m3 month-1]"] = fmt(he_gross_m - he_ret_m)
        row["W8_Hetao_return_fraction_f [-]"] = fmt(he_ret / he_gross)
        row["W8_Hetao_irrigation_efficiency_eta [-]"] = fmt(he_eta[year])
        row["W8_Hetao_field_effective_supply_V [1e8 m3 month-1]"] = fmt(he_eta[year] * he_gross_m)
        row["W8_Hetao_channel_capacity_Dmax [1e8 m3 month-1]"] = fmt(he_dmax)
        row["W8_Hetao_annual_budget_B [1e8 m3 year-1]"] = fmt(he_gross)

        solar_norm = (solar_values[idx] - solar_min) / (solar_max - solar_min)
        wind_norm = (wind_values[idx] - wind_min) / (wind_max - wind_min)
        renewable_availability = 0.15 + 0.80 * (0.70 * solar_norm + 0.30 * wind_norm)
        row["E3_availability [-]"] = fmt(renewable_availability)
        row["E3_nonhydro_renewable_generation [GWh month-1]"] = fmt(720 * renewable_availability)
        row["E4_import_limit [GWh month-1]"] = fmt(500)
        row["E4_availability [-]"] = fmt(1)
        row["E5_base_demand_L_base [GWh month-1]"] = fmt(1000)
        row["E6_Ningxia_interface_capacity [GWh month-1]"] = fmt(ebar8)
        row["E7_Hetao_interface_capacity [GWh month-1]"] = fmt(ebar9)
        row["E8_Ningxia_pump_head [m]"] = fmt(300)
        row["E8_Ningxia_pump_efficiency [-]"] = fmt(0.75)
        row["E8_Ningxia_pump_capacity_Ebar8 [GWh month-1]"] = fmt(ebar8)
        row["E9_Hetao_pump_head [m]"] = fmt(5)
        row["E9_Hetao_pump_efficiency [-]"] = fmt(0.75)
        row["E9_Hetao_pump_capacity_Ebar9 [GWh month-1]"] = fmt(ebar9)

        area_by_crop = {
            1: nx_wheat[year],
            2: nx_maize[year],
            3: nx_rice[year],
            4: he_wheat[year],
            5: he_maize[year],
            6: he_sunflower[year],
        }
        requirements: dict[int, float] = {}
        for crop in range(1, 7):
            row[f"F{crop}_area_A [ha]"] = fmt(area_by_crop[crop])
            coefficient = fnum(row[f"F{crop}_crop_coefficient_k [-]"]) or 0.0
            if crop <= 3:
                et0 = fnum(row["Ningxia_reference_ET0_E0 [mm month-1]"]) or 0.0
                precip = fnum(row["Ningxia_effective_precipitation_P [mm month-1]"]) or 0.0
            else:
                et0 = fnum(row["Hetao_reference_ET0_E0 [mm month-1]"]) or 0.0
                precip = fnum(row["Hetao_effective_precipitation_P [mm month-1]"]) or 0.0
            requirement = area_by_crop[crop] * max(0.0, coefficient * et0 - precip) * 1e-7
            requirements[crop] = requirement
            row[f"F{crop}_water_requirement_W [1e8 m3 month-1]"] = fmt(requirement)

        for crops, supply in (((1, 2, 3), nx_eta[year] * nx_gross_m), ((4, 5, 6), he_eta[year] * he_gross_m)):
            total = sum(requirements[c] for c in crops)
            for crop in crops:
                allocated = supply * requirements[crop] / total if total > 0 else 0.0
                row[f"F{crop}_allocated_effective_water_Vc [1e8 m3 month-1]"] = fmt(allocated)
                row[f"F{crop}_water_deficit [1e8 m3 month-1]"] = fmt(max(0.0, requirements[crop] - allocated))

        row["Ningxia_water_source_url [-]"] = " | ".join((NIXIA_BULLETINS, NIXIA_EFFICIENCY, NIXIA_CAPACITY, NIXIA_PUMP))
        row["Hetao_water_source_url [-]"] = " | ".join((HETAO_ARTICLE, HETAO_CANAL))
        row["crop_area_source_url [-]"] = " | ".join((NIXIA_YEARBOOK, NIXIA_YEARBOOK_MIRROR, HETAO_ARTICLE))
        row["reservoir_reference_source_url [-]"] = " | ".join((LONGYANGXIA_REFERENCE, LONGYANGXIA_HEAD, HESS_ARTICLE))
        row["additional_reservoir_dataset_url [-]"] = " | ".join((RESCN_DATA, RESCN_ARTICLE, HESS_DATA, HESS_ARTICLE))
        row["transformation_and_quality_notes [-]"] = (
            "Observed and literature-derived values are mixed with explicitly flagged proxies. "
            "Ningxia/Hetao annual water totals are divided uniformly by 12 because monthly district series were unavailable. "
            "Longyangxia storage is a Res-CN anomaly anchored to the physical operating range; Liujiaxia storage uses the HESS monthly regulation climatology. "
            "Reservoir loss contains satellite-derived evaporation only and is therefore a lower bound for L_res. "
            "Tangnaihai/Toudaoguai flows are observed for 2001-2014 and use 2001-2014 monthly climatology for 2015-2021. "
            "W3 is balance-derived; the inseparable Lanzhou-Toudaoguai residual is split equally across W4-W6. "
            "Ningxia crop-area gaps use linear interpolation and 2001-2004 backcasting from 2005; Hetao 2019-2020 uses linear interpolation to 2021. "
            "E3, E4 import capacity and L_base are scenario-scale proxies (1 GW renewable, 500 GWh/month import limit, 1000 GWh/month base load), not observations. "
            "Hetao pump head is a 5 m aggregate engineering proxy; 0.75 pump efficiencies are engineering assumptions. "
            "Executed reservoir releases, W2 inflow, intake-available flows, grid dispatch/allocation, executed pumping, reward components and policy actions remain blank because they are generated by the algorithm/environment."
        )

    # Plain UTF-8 (no BOM) keeps the first header portable across pandas, csv, R, Julia and CLI tools.
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)

    intentionally_blank = {
        "W1_executed_release_R1 [1e8 m3 month-1]": "policy/environment output",
        "E1_generation [GWh month-1]": "derived after executed W1 release",
        "W2_inflow_I [1e8 m3 month-1]": "derived from W1 release and W3 routing",
        "W2_executed_release_R2 [1e8 m3 month-1]": "policy/environment output",
        "E2_generation [GWh month-1]": "derived after executed W2 release",
        "W7_Ningxia_intake_available_Q [1e8 m3 month-1]": "derived from upstream routing",
        "W8_Hetao_intake_available_Q [1e8 m3 month-1]": "derived from upstream routing",
        "raw_policy_R1 [1e8 m3 month-1]": "policy output",
        "raw_policy_R2 [1e8 m3 month-1]": "policy output",
        "raw_policy_D_N [1e8 m3 month-1]": "policy output",
        "raw_policy_D_H [1e8 m3 month-1]": "policy output",
        "E4_grid_import [GWh month-1]": "grid dispatch output",
        "E5_residual_energy [GWh month-1]": "derived after hydropower/import decisions",
        "E5_curtailment [GWh month-1]": "grid-clearing output",
        "E5_unserved_energy [GWh month-1]": "grid-clearing output",
        "E6_Ningxia_accepted_pump_demand [GWh month-1]": "allocation output",
        "E6_Ningxia_allocated_supply [GWh month-1]": "allocation output",
        "E6_Ningxia_shortfall [GWh month-1]": "allocation output",
        "E7_Hetao_accepted_pump_demand [GWh month-1]": "allocation output",
        "E7_Hetao_allocated_supply [GWh month-1]": "allocation output",
        "E7_Hetao_shortfall [GWh month-1]": "allocation output",
        "E8_Ningxia_executed_pumping [GWh month-1]": "environment output",
        "E9_Hetao_executed_pumping [GWh month-1]": "environment output",
        "reward_water_deficit_CW [1e8 m3 month-1]": "reward calculation output",
        "reward_energy_deficit_CE [GWh month-1]": "reward calculation output",
        "reward_food_deficit_CF [1e8 m3 month-1]": "reward calculation output",
        "reward_weight_lambda_W [-]": "experiment hyperparameter",
        "reward_weight_lambda_E [-]": "experiment hyperparameter",
        "reward_weight_lambda_F [-]": "experiment hyperparameter",
        "reward_r [-]": "reward calculation output",
    }

    provenance_rows = [
        ("W1 storage/loss", "derived", "Res-CN storage-change and evaporation series; absolute anomaly centered within physical bounds", "2001-2021", f"{RESCN_DATA} | {RESCN_ARTICLE}", "medium"),
        ("W2 storage/loss", "proxy/derived", "HESS monthly regulation climatology plus Res-CN evaporation", "2001-2021", f"{HESS_DATA} | {RESCN_DATA}", "medium-high"),
        ("W1 inflow and W9 flow", "observed+climatology", "HESS gauge observations through 2014; 2001-2014 monthly climatology afterward", "2001-2021", f"{HESS_ARTICLE} | {HESS_DATA}", "medium after 2014"),
        ("W3-W6 reach gains", "calibrated proxy", "W3 from reservoir balance; downstream aggregate residual split equally across W4-W6", "12-month climatology", f"{HESS_ARTICLE} | {NIXIA_BULLETINS} | {HETAO_ARTICLE}", "high"),
        ("W9 ecological threshold", "literature-derived", "125 m3/s continuous minimum converted to 30-day volume", "all months", ECO_FLOW, "medium"),
        ("Ningxia diversion/drainage", "reported+derived", "Official annual bulletins; totals divided uniformly by 12", "2001-2021", NIXIA_BULLETINS, "low annual / high monthly"),
        ("Ningxia irrigation efficiency", "reported+interpolated", "Published annual anchors with linear interpolation", "2001-2021", NIXIA_EFFICIENCY, "medium"),
        ("Hetao diversion/drainage", "digitized+interpolated", "Digitized Figure 6 annual points; short gaps interpolated", "2001-2021", HETAO_ARTICLE, "medium"),
        ("Hetao irrigation efficiency", "scenario proxy", "Linear 0.40 to reported 0.467/0.47 anchors", "2001-2021", HETAO_ARTICLE, "high"),
        ("Channel capacities", "physical reference", "750 and 565 m3/s converted to 30-day volumes", "all months", f"{NIXIA_CAPACITY} | {HETAO_CANAL}", "medium"),
        ("Reservoir head/beta/Rmax", "physical derived", "Design head, installed capacity and 0.90 turbine efficiency", "all months", f"{LONGYANGXIA_REFERENCE} | {LONGYANGXIA_HEAD} | {HESS_ARTICLE}", "medium"),
        ("Ningxia crop areas", "reported+interpolated", "Statistical-yearbook observations; missing years linearly interpolated/backcast", "2001-2021", f"{NIXIA_YEARBOOK} | {NIXIA_YEARBOOK_MIRROR}", "medium-high for interpolated years"),
        ("Hetao crop areas", "digitized+interpolated", "Digitized Figure 3; 2019-2020 interpolated", "2001-2021", HETAO_ARTICLE, "medium"),
        ("Crop water requirement/allocation", "formula-derived", "TeX equations using area, FAO crop coefficients, ET0, effective precipitation and field supply", "2001-2021", "cas-dc-sample.tex | https://www.fao.org/4/X0490E/x0490e0b.htm", "inherits inputs"),
        ("E3/E4/Lbase", "scenario proxy", "1 GW climate-shaped renewable, 500 GWh/month import limit, 1000 GWh/month base load", "2001-2021", "NASA POWER climate + explicit scenario assumptions", "high; replace for empirical study"),
        ("E8/E9 pump references", "physical+engineering proxy", "Ningxia 300 m/28 m3/s reference; Hetao 5 m aggregate lift proxy; efficiency 0.75", "all months", f"{NIXIA_PUMP} | {HETAO_CANAL}", "high for aggregate Hetao head"),
    ]
    for field, reason in intentionally_blank.items():
        provenance_rows.append((field, "intentionally blank", reason, "all months", "algorithm/environment", "not applicable"))

    with PROVENANCE.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(("field_or_group", "status", "method", "coverage", "source", "uncertainty"))
        writer.writerows(provenance_rows)

    after_blank = sum(not str(row[col]).strip() for row in rows for col in headers)
    blank_fields = [col for col in headers if all(not str(row[col]).strip() for row in rows)]

    # Basic integrity checks.
    assert len(rows) == 252 and len(headers) == 192
    assert len({row["year_month [YYYY-MM]"] for row in rows}) == 252
    for year in range(2001, 2022):
        yr = [r for r in rows if int(r["year [YYYY]"]) == year]
        assert abs(sum(float(r["W7_Ningxia_gross_diversion_D [1e8 m3 month-1]"]) for r in yr) - nx_water[year][0]) < 1e-5
        assert abs(sum(float(r["W8_Hetao_gross_diversion_D [1e8 m3 month-1]"]) for r in yr) - he_div[year]) < 1e-5
    for i in range(251):
        assert abs(float(rows[i]["W1_storage_next_Splus [1e8 m3]"]) - float(rows[i + 1]["W1_storage_begin_S [1e8 m3]"])) < 2e-6
        assert abs(float(rows[i]["W2_storage_next_Splus [1e8 m3]"]) - float(rows[i + 1]["W2_storage_begin_S [1e8 m3]"])) < 2e-6

    report = f"""# Upper Yellow River WEF input data fill report

## Outcome

- Source CSV preserved: `{SOURCE.name}`
- Filled CSV: `{OUTPUT.name}`
- Rows × columns: {len(rows)} × {len(headers)}
- Blank cells before: {before_blank:,}
- Blank cells after: {after_blank:,}
- Newly populated cells: {before_blank - after_blank:,}
- Fully blank columns after filling: {len(blank_fields)} (all are algorithm outputs/hyperparameters listed below)

## What was filled

- Reservoir storage proxies, evaporation losses, physical bounds, heads, release capacities and hydropower conversion coefficients.
- Tangnaihai inflow, Toudaoguai flow and a literature-based ecological threshold.
- Seasonal reach-gain profiles calibrated from gauge/reservoir/district water balances.
- Complete 2001-2021 Ningxia and Hetao annual diversion, drainage, return fraction, irrigation efficiency, effective field supply, capacity and budget inputs.
- Complete crop areas, crop-water requirements, proportional effective-water allocation and crop-water deficits for F1-F6.
- Clearly labeled scenario-scale electricity and pumping reference inputs.

## Intentionally blank columns

""" + "\n".join(f"- `{field}` — {reason}" for field, reason in intentionally_blank.items()) + f"""

## Key limitations

- Annual district water totals are divided uniformly by 12; the CSV does not claim observed monthly irrigation seasonality.
- Longyangxia storage is a satellite storage-change anomaly anchored within physical bounds, not a gauged absolute storage record.
- Liujiaxia absolute storage is a climatological regulation proxy.
- W4-W6 cannot be separately identified from the available Lanzhou and Toudaoguai gauges; their aggregate residual is divided equally.
- E3/E4/Lbase and aggregate pump references are scenario-scale proxies and must be replaced before empirical energy-system inference.
- Digitized figure values and interpolated/backcast crop-area years have higher uncertainty than official annual tables.

## Principal sources

- HESS Yellow River discharge/dam study: {HESS_ARTICLE}
- HESS data archive: {HESS_DATA}
- Res-CN article/data: {RESCN_ARTICLE} ; {RESCN_DATA}
- Ningxia water-resources bulletins: {NIXIA_BULLETINS}
- Ningxia Statistical Yearbooks: {NIXIA_YEARBOOK}
- Hetao annual water/crop figures: {HETAO_ARTICLE}
- Ecological-flow threshold source: {ECO_FLOW}
"""
    REPORT.write_text(report, encoding="utf-8")

    print(f"wrote {OUTPUT}")
    print(f"wrote {PROVENANCE}")
    print(f"wrote {REPORT}")
    print(f"blank cells: {before_blank} -> {after_blank}; fully blank columns: {len(blank_fields)}")


if __name__ == "__main__":
    main()
