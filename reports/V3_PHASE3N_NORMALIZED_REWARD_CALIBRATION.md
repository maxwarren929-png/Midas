# V3 Phase 3N — Normalized Reward Semantics and Calibration

Phase 3N implements and calibrates unit-safe turnover and drawdown reward
semantics. The two-seed gate failed, so the planned five-seed promotion was not
run. Development remained confined to training and validation; the test split
was not inspected.

## Versioned reward contract

Historical `risk_adjusted_return` behavior is retained exactly for experiment
and checkpoint reproducibility. A new explicit
`normalized_risk_adjusted_return` kind uses:

```text
drawdown_basis = max(current_maximum_drawdown - previous_maximum_drawdown, 0)
turnover_basis = daily_trade_notional / previous_net_worth
reward = log_return
       - drawdown_penalty_weight * drawdown_basis
       - turnover_penalty_weight * turnover_basis
       - other configured penalties
```

This formulation charges a historical drawdown only when it worsens and makes
turnover independent of account denomination. `RewardBreakdown` records both
bases and their realized penalties. Compact diagnostics aggregate all four
values and expose the new reward kind through a controlled override.

Regressions cover account-scale invariance, no repeated charge for unchanged
drawdown, finite-difference penalty slopes, and the complete engine path from
portfolio turnover and prior drawdown state into committed decision traces.

## Two-seed calibration

The predeclared candidate approximately matched Phase 3M's reward pressure in
the new units:

```text
drawdown_penalty_weight=0.15
turnover_penalty_weight=0.0004
seeds=41,42 population=10 steps=1259
seed_derivation=splitmix_v2 optimizer=momentum learning_rate=3e-4
gradient_clip=1.0
```

Across 25,180 training decisions, the drawdown basis summed to 3.8984 and the
turnover basis to 1,768.3702. Their penalties summed to 0.5848 and 0.7073,
respectively, or 0.00005132 combined per decision. This was 76% of the legacy
Phase 3M pressure over the same two seeds: comparable, finite, and far below
reward clipping.

| Training measurement, seeds 41–42 | Unpenalized | Legacy penalties | Normalized penalties |
| --- | ---: | ---: | ---: |
| PPO updates | 6,306 | 6,306 | 6,304 |
| Deaths | 15 | 13 | 11 |
| Mean aggregate lifetime turnover per seed | 9,944,776 | 9,828,240 | 9,773,415 |
| Mean completed-life maximum drawdown | 12.147% | 12.432% | 12.579% |

Training turnover fell 1.7% from the unpenalized baseline and deaths fell, but
drawdown did not improve.

## Matched validation gate

Each normalized checkpoint was evaluated with its trained lifetime parameters
and with those parameters reset to its exact final genomes. The two matching
Phase 3L unpenalized and Phase 3M legacy-penalty seeds are included for context.

| Validation measurement | Normalized final genome | Normalized PPO | Legacy PPO | Unpenalized PPO |
| --- | ---: | ---: | ---: | ---: |
| Mean return | -2.727% | -3.051% | -2.972% | -2.404% |
| Mean within-seed median return | -2.444% | -2.771% | -2.573% | -1.980% |
| Mean maximum drawdown | 4.195% | 4.961% | 4.853% | 4.321% |
| Mean population turnover | 26,032.92 | 32,653.53 | 33,497.22 | 27,433.98 |
| Mean population fees | 13.02 | 16.33 | 16.75 | 13.72 |

Normalized PPO beat its matched genome on mean return in one of two seeds and
on drawdown or turnover in zero. It lost to unpenalized PPO on return,
drawdown, and turnover in both seeds. Compared with the legacy penalty, it
slightly reduced turnover but did not improve return or drawdown.

## Decision

The candidate fails the promotion gate and is not advanced to five seeds or
made a default. The implementation remains available as a correctly scaled,
versioned research objective, while all historical reward behavior remains
reproducible.

The result suggests that units were a real correctness problem but not the
main learning bottleneck. Incremental maximum drawdown is sparse, and PPO's
per-rollout advantage normalization may attenuate reward components that are
small or nearly common across transitions. Before another reward sweep, the
next audit should measure component variance and covariance with normalized
advantages and policy updates. A direct risk-aware critic/auxiliary target or
constrained objective should be considered only after that measurement.

Raw summaries and checkpoints are in `reports/phase3n/`.
