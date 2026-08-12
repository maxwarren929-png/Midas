# V3 Phase 3M — Drawdown and Turnover Reward Comparison

Phase 3M repeats the Phase 3L full-horizon momentum protocol with only the
reward objective changed. Development remains confined to training and
validation; the test split was not inspected.

## Frozen treatment

The baseline reward used pure log return. The treatment uses
`risk_adjusted_return` with these predeclared weights:

```text
drawdown_penalty_weight=0.0005
turnover_penalty_weight=0.00000005
```

Turnover enters the reward as raw daily trade notional, while maximum drawdown
is a fraction. Phase 3L averaged $781.26 turnover per agent-day, 11.56%
terminal maximum drawdown, and approximately 0.000331 absolute daily log
return. The selected weights targeted a combined penalty near 20% of that
return scale without clipping or dominating it.

All other training controls match Phase 3L:

```text
config=configs/prompt6/combined-ppo.toml
split=training steps=1259 population=10 seeds=41,42,43,44,45
seed_derivation=splitmix_v2 gradient_clip=1.0
optimizer=momentum learning_rate=3e-4
```

The five runs completed 15,769 PPO updates and 38 deaths. Peak resident memory
was 53,836 KiB. Diagnostics record the effective reward configuration and the
aggregate reward decomposition. Across 50,360 decisions, drawdown penalty was
1.4740 and turnover penalty was 1.9685, or 0.00006836 combined per decision.
That is 20.7% of the calibration return scale, confirming that the treatment
was active and neither a no-op nor a clipped, unit-mismatched penalty.

## Validation comparison

Each penalized checkpoint was evaluated on validation both with trained
lifetime parameters and with those parameters reset to its exact final
genomes. The unpenalized Phase 3L trained result is the treatment comparator.

| Measurement | Penalized final genome | Penalized PPO | Unpenalized PPO | Penalized vs unpenalized |
| --- | ---: | ---: | ---: | ---: |
| Across-seed mean return | -1.967% | -2.218% | -2.158% | -0.060 pp |
| Mean within-seed median return | -1.805% | -1.873% | -1.546% | -0.327 pp |
| Mean maximum drawdown | 3.816% | 4.531% | 4.534% | -0.003 pp |
| Mean population turnover | 25,158.70 | 31,140.13 | 28,279.06 | +10.1% |
| Mean population fees | 12.58 | 15.57 | 14.14 | +10.1% |

Against its matched final genomes, penalized PPO won mean return in two of five
seeds and median-agent return in three of five. It improved drawdown and
turnover in zero of five seeds. Against unpenalized PPO, the treatment won mean
return in one seed, median return in one, drawdown in two, and turnover in one.

The direct treatment comparison includes any treatment-induced differences in
replacement and final population, as it should for an end-to-end intervention.
It is not a same-genome causal estimate. The within-checkpoint genome control
isolates lifetime-learning value for each penalized final population.

## Decision

These weights are rejected as defaults. They produced no meaningful drawdown
improvement, increased validation turnover, and slightly reduced return. The
training populations also had essentially unchanged turnover (9.832 million
versus 9.836 million mean aggregate turnover per seed), despite the verified
penalty pressure.

Blindly increasing the coefficients is not the preferred next step. Maximum
drawdown is a persistent path-level quantity charged again on every later day,
and raw-notional turnover makes its coefficient dependent on account size. A
better isolated experiment should first make turnover dimensionless (for
example, daily turnover divided by pre-trade net worth) and use an incremental
drawdown cost or recovery-aware risk term. Those semantics should receive
finite-difference/unit regressions before another five-seed run.

Raw summaries and checkpoints are in `reports/phase3m/`.
