# Upper Yellow River WEF input data fill report

## Outcome

- Source CSV preserved: `upper_yellow_river_wef_monthly_inputs_2001_2021.csv`
- Filled CSV: `upper_yellow_river_wef_monthly_inputs_2001_2021_filled.csv`
- Rows × columns: 252 × 192
- Blank cells before: 25,896
- Blank cells after: 7,560
- Newly populated cells: 18,336
- Fully blank columns after filling: 30 (all are algorithm outputs/hyperparameters listed below)

## What was filled

- Reservoir storage proxies, evaporation losses, physical bounds, heads, release capacities and hydropower conversion coefficients.
- Tangnaihai inflow, Toudaoguai flow and a literature-based ecological threshold.
- Seasonal reach-gain profiles calibrated from gauge/reservoir/district water balances.
- Complete 2001-2021 Ningxia and Hetao annual diversion, drainage, return fraction, irrigation efficiency, effective field supply, capacity and budget inputs.
- Complete crop areas, crop-water requirements, proportional effective-water allocation and crop-water deficits for F1-F6.
- Clearly labeled scenario-scale electricity and pumping reference inputs.

## Intentionally blank columns

- `W1_executed_release_R1 [1e8 m3 month-1]` — policy/environment output
- `E1_generation [GWh month-1]` — derived after executed W1 release
- `W2_inflow_I [1e8 m3 month-1]` — derived from W1 release and W3 routing
- `W2_executed_release_R2 [1e8 m3 month-1]` — policy/environment output
- `E2_generation [GWh month-1]` — derived after executed W2 release
- `W7_Ningxia_intake_available_Q [1e8 m3 month-1]` — derived from upstream routing
- `W8_Hetao_intake_available_Q [1e8 m3 month-1]` — derived from upstream routing
- `raw_policy_R1 [1e8 m3 month-1]` — policy output
- `raw_policy_R2 [1e8 m3 month-1]` — policy output
- `raw_policy_D_N [1e8 m3 month-1]` — policy output
- `raw_policy_D_H [1e8 m3 month-1]` — policy output
- `E4_grid_import [GWh month-1]` — grid dispatch output
- `E5_residual_energy [GWh month-1]` — derived after hydropower/import decisions
- `E5_curtailment [GWh month-1]` — grid-clearing output
- `E5_unserved_energy [GWh month-1]` — grid-clearing output
- `E6_Ningxia_accepted_pump_demand [GWh month-1]` — allocation output
- `E6_Ningxia_allocated_supply [GWh month-1]` — allocation output
- `E6_Ningxia_shortfall [GWh month-1]` — allocation output
- `E7_Hetao_accepted_pump_demand [GWh month-1]` — allocation output
- `E7_Hetao_allocated_supply [GWh month-1]` — allocation output
- `E7_Hetao_shortfall [GWh month-1]` — allocation output
- `E8_Ningxia_executed_pumping [GWh month-1]` — environment output
- `E9_Hetao_executed_pumping [GWh month-1]` — environment output
- `reward_water_deficit_CW [1e8 m3 month-1]` — reward calculation output
- `reward_energy_deficit_CE [GWh month-1]` — reward calculation output
- `reward_food_deficit_CF [1e8 m3 month-1]` — reward calculation output
- `reward_weight_lambda_W [-]` — experiment hyperparameter
- `reward_weight_lambda_E [-]` — experiment hyperparameter
- `reward_weight_lambda_F [-]` — experiment hyperparameter
- `reward_r [-]` — reward calculation output

## Key limitations

- Annual district water totals are divided uniformly by 12; the CSV does not claim observed monthly irrigation seasonality.
- Longyangxia storage is a satellite storage-change anomaly anchored within physical bounds, not a gauged absolute storage record.
- Liujiaxia absolute storage is a climatological regulation proxy.
- W4-W6 cannot be separately identified from the available Lanzhou and Toudaoguai gauges; their aggregate residual is divided equally.
- E3/E4/Lbase and aggregate pump references are scenario-scale proxies and must be replaced before empirical energy-system inference.
- Digitized figure values and interpolated/backcast crop-area years have higher uncertainty than official annual tables.

## Principal sources

- HESS Yellow River discharge/dam study: https://hess.copernicus.org/articles/25/1133/2021/
- HESS data archive: https://doi.org/10.5281/zenodo.3979053
- Res-CN article/data: https://essd.copernicus.org/articles/15/2781/2023/ ; https://doi.org/10.5281/zenodo.7664489
- Ningxia water-resources bulletins: https://slt.nx.gov.cn/xxgk_281/fdzdgknr/gbxx/szygb/
- Ningxia Statistical Yearbooks: https://www.nx.gov.cn/zwgk/zfxxgk/fdzdgknr/tjxx_40901/tjnj/index.html
- Hetao annual water/crop figures: https://www.mdpi.com/2073-4395/15/7/1654
- Ecological-flow threshold source: https://doi.org/10.3390/ijerph182010956
