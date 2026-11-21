# PPO and graph-structured PPO for Yellow River WEF operation

This repository retains four connected experiments.

| Experiment | Question | Policy evaluation condition |
|---|---|---|
| 1 | With current information, is GNN-PPO clearly better than PPO? | `tau0`: all 24 nodes use the current month. |
| 2 | Does graph structure help when all information is equally stale? | `tau1`–`tau5`: all 24 nodes use observations 1–5 months old. |
| 3 | Which stale GNN-PPO inputs are most damaging, and how do effects propagate through WEF? | Only one sector, then only one node, is delayed by 1–5 months; every other node remains current. |
| 4 | Does PPO identify the same most damaging stale inputs and propagation pattern? | The identical sector and single-node delay scenarios used in Experiment 3, evaluated with PPO. |

## Exact delay definitions

- `tauN`: uniform delay. Each of W1–W9, E1–E9 and F1–F6 uses a value from `N` months ago.
- `sector_water_tauN`, `sector_energy_tauN`, `sector_food_tauN`: only the selected group is delayed. The groups are W1–W9 (water), E1–E9 (energy) and F1–F6 (irrigation/food).
- `node_<name>_tauN`: only the named node is delayed; the remaining 23 nodes use current observations.

Delays are applied to policy observations only. Physical transitions and the common feasibility projection remain current. For the first `N` months of a year, the delayed buffer is initialized with the first-month observation; no information is borrowed from the preceding year.

Experiments 3 and 4 reuse the current-information-trained GNN-PPO and PPO checkpoints, respectively; neither retrains a policy. They are matched sensitivity/propagation analyses, not new training runs.

## Run

Train the two policies and run Experiments 1–4:

```bash
MPLCONFIGDIR=/tmp/wef-matplotlib PYTHONPATH=py python3 -m main.main --result-dir result
```

Re-run a single location-sensitivity experiment from retained checkpoints without retraining:

```bash
MPLCONFIGDIR=/tmp/wef-matplotlib PYTHONPATH=py python3 -m main.sector_delay --result-dir result
MPLCONFIGDIR=/tmp/wef-matplotlib PYTHONPATH=py python3 -m main.sector_delay --result-dir result --policy ppo
```

## Retained results

- `01_current_information_graph_encoder/`: Experiment 1 seed-level outcomes and paired comparison.
- `02_uniform_delay_robustness/`: Experiment 2 `tau1`–`tau5` outcomes and the pre-specified average-delay comparison.
- `03_sector_specific_delay_gnnppo/`: Experiment 3 definitions, annual outcomes, sector cascade effects and node-risk ranking.
- `04_sector_specific_delay_ppo/`: Experiment 4 definitions, annual outcomes, sector cascade effects, node-risk ranking, and matched PPO–GNN-PPO comparison tables.
- `publication_figures/`: nine high-resolution PNG figures only.
- `raw/`: Experiment 1/2 raw annual and monthly records needed to rebuild their results.

The primary outcome is the average WEF penalty, `P_avg = -mean_monthly_reward` (lower is better). The internal `cost` key is retained for compatibility with existing CSV files. The three penalty components in Experiments 3 and 4 are shown before reward weighting to expose water, energy and irrigation propagation separately.
