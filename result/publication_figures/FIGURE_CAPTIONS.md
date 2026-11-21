# Figure captions

## Training reward — Current and five-month delayed observations

PPO and GNN-PPO are trained with no delayed observations under a fixed budget of 160 updates (four annual training episodes per update). After every update, the evolving policy is evaluated deterministically on the same 2016–2018 validation years using either no delayed observations (`tau0`) or observations delayed by five months (`tau5`). Curves show the unsmoothed ten-seed mean at every recorded update; colour identifies the policy architecture and line style identifies the delay condition. The delayed curves therefore diagnose observation-delay robustness during learning; they do not represent policies trained under delayed observations.

## Training convergence — Mean episode reward with no delayed observations

PPO and GNN-PPO are trained only with no delayed observations (`tau0`). The horizontal axis reports cumulative environment interactions: each update contains four 12-month episodes, equivalent to 48 environment steps. Curves show the ten-seed mean after a 10-update trailing smooth, and bands show 95% bootstrap confidence intervals across training seeds. To retain a balanced comparison, the plot stops at update 110 (5,280 environment steps), the last update available for every PPO and GNN-PPO seed; later updates are omitted because of early stopping.

## Figure 1 — No-delay comparison

Each paired point is one independently trained seed, averaged across the held-out 2019–2021 annual forcing sequences. The primary endpoint is the average WEF penalty (`P_avg = -mean reward`; lower is better). The figure reports the observed difference and uncertainty; it does not claim statistical equivalence.

## Figure 2 — Uniform observation-delay robustness

`tauN` means that all 24 WEF nodes use observations N months old. Points show the mean reward across 10 independently trained seeds and the 2019–2021 held-out forcing sequences; lines connect the ordered delay levels. The aligned table reports the same means and their observed difference (`GNN-PPO − PPO`), for which positive values favour GNN-PPO. PPO reward deteriorates under uniformly stale information, whereas GNN-PPO remains comparatively stable. Higher reward is better.


## Figure 3 — Sector-specific GNN-PPO delay risk

Markers show the change in average WEF penalty, `Delta P_avg = P_avg_delay - P_avg_tau0`, and the legend identifies the sector whose observations are delayed by 1–5 months. Horizontal intervals are 95% two-way bootstrap confidence intervals across 10 training seeds and three held-out annual forcing sequences. All other nodes have no delayed observations. This is a fixed-policy information-location sensitivity analysis, not a re-training experiment.

## Figure 4 — WEF delay propagation

Each panel sets the source of stale information and shows the resulting change in the unweighted average normalised water, energy and food penalty components. Positive values mean deterioration. Cell annotations use the power-of-ten multiplier shown on the shared colour bar. The three normalised components are displayed before reward weighting, so the 0.35 energy reward coefficient does not conceal a physical branch response.

## Figure 5 — Single-node GNN-PPO vulnerability screen

Panel a shows all 24 node estimates with 95% two-way bootstrap confidence intervals, panel b groups the mean changes by sector, and panel c enlarges the five highest point estimates. In each scenario, exactly one node receives a five-month-old observation and the other 23 nodes have no delayed observations. Positive values indicate deterioration. The ranking is exploratory and identifies information locations that merit focused monitoring or follow-up stress testing.


## Figure 6 — Sector-specific PPO delay risk

Only the named sector is delayed by 1–5 months; all other nodes have no delayed observations. Changes in average WEF penalty are relative to the same checkpoint, seed and held-out year under `tau0`. This is a fixed-policy information-location sensitivity analysis, not a re-training experiment.

## Figure 7 — PPO WEF delay propagation

Each panel sets the source of stale information and shows the resulting change in the unweighted average normalised water, energy and food penalty components. Positive values mean deterioration. Cell annotations use the power-of-ten multiplier shown on the shared colour bar. The three normalised components are displayed before reward weighting, so the 0.35 energy reward coefficient does not conceal a physical branch response.

## Figure 8 — Single-node PPO vulnerability screen

For each row, exactly one of the 24 nodes receives a five-month-old observation and all other nodes are current. The ranking is exploratory and identifies information locations that merit focused monitoring or follow-up stress testing.


## Figure 9 — PPO versus GNN-PPO information-location sensitivity

Both policy families are evaluated with exactly the same sector-specific and single-node delay definitions. Panel a compares the sector-specific dose response; panel b compares the 24-node five-month point estimates. Each change in average WEF penalty is within-policy relative to that policy's same-seed, same-year `tau0` reference. The cross-architecture display is descriptive and does not assert a new node-level significance test.

## Training reward — GNN-PPO under uniform observation delays

GNN-PPO is trained with no delayed observations under a fixed budget of 160 updates (four annual training episodes per update). After every update, the evolving policy is evaluated deterministically on the same 2016–2018 validation years at each uniform observation delay from `tau0` to `tau5`. Curves show the unsmoothed ten-seed mean at every recorded update. These are delay-diagnostic evaluations of a policy trained with no delayed observations, rather than six separately delay-trained policies.
