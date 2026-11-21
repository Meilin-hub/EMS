# Statistical analysis and independence

## Units of inference

- Independent algorithmic repetitions: 10 training seeds.
- Held-out annual forcing sequences: 2019–2021. They are repeated evaluation conditions with a documented repeated-flow-climatology limitation, not independent catchments.
- Uniform delays `tau0`–`tau5` are repeated conditions applied to the same policy seeds, not additional samples.

## Analysis plan

- The primary reported objective is `cost = -mean_monthly_reward` (lower is better).
- Descriptive values are seed means after averaging across held-out annual forcing sequences. Pairwise 95% intervals use a two-way bootstrap that resamples seeds and annual forcing sequences. Exact sign-randomization p values are calculated on seed means.
- Per-delay tests are Holm corrected within each metric across `tau1`–`tau5`. The pre-specified `tau1`–`tau5` average is one test per metric.
- The maximum-cost-across-`tau1`–`tau5` endpoint is exploratory. Its two exported endpoints (cost and penalty versus `tau0`) receive a separate Holm correction.
- `table_s3_seed_label_free_permutation_sensitivity.csv` gives a label-permutation sensitivity analysis that does not assume identical numeric seed labels create true cross-architecture pairs. Its three primary-cost endpoints are Holm adjusted together.

## Scope and review risks

PPO and GNN-PPO have different trainable parameter counts, recorded in `methods_and_parameter_counts.csv`; an observed difference cannot be attributed uniquely to graph structure. A parameter-matched MLP or graph-edge ablation would require additional training. The deterministic feasibility projector has perfect within-year forcing knowledge for its terminal-storage guard, so the delay study evaluates policies under a common external supervisor rather than a fully causal operational controller.
