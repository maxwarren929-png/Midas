# V3 Phase 3J — Paired Multi-Seed Optimizer Comparison

Phase 3J compares retained momentum against the update-scale-calibrated AdamW
candidate across five disjoint SplitMix V2 seed ensembles. This is short
synthetic training evidence, not a profitability claim.

## Protocol

Each seed launched momentum and AdamW concurrently. All other inputs were
identical:

```text
config=configs/prompt6/combined-ppo.toml
split=training steps=127 population=10 seeds=41,42,43,44,45
seed_derivation=splitmix_v2 gradient_clip=1.0 weight_decay=0.01
momentum_learning_rate=3e-4 adamw_learning_rate=3.5e-6
```

SplitMix V2 makes the adjacent roots independent at the agent-seed level, so
all five pairs are retained.

## Paired results

| Seed | Momentum mean return | AdamW mean return | AdamW delta | Momentum median | AdamW median | Drawdown delta | Turnover delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 41 | 1.881% | 2.003% | +0.121 pp | 1.778% | 1.623% | +0.102 pp | +0.255% |
| 42 | 2.522% | 2.338% | -0.184 pp | 2.446% | 1.986% | -0.043 pp | -0.412% |
| 43 | 2.019% | 2.201% | +0.181 pp | 2.038% | 2.005% | -0.057 pp | +1.212% |
| 44 | 1.398% | 1.562% | +0.164 pp | 1.160% | 1.421% | +0.060 pp | +0.194% |
| 45 | 1.841% | 2.124% | +0.283 pp | 1.517% | 1.653% | -0.014 pp | -0.430% |

AdamW won paired mean return in four of five seeds. Its mean paired advantage
was `0.1132` percentage points, its median paired advantage was `0.1640`
percentage points, and the paired-difference sample standard deviation was
`0.1764` percentage points. AdamW won within-population median return in only
two of five seeds; averaged across seeds, median return was 0.0503 percentage
points below momentum.

## Aggregate mechanics

| Measurement | Momentum | AdamW `3.5e-6` |
| --- | ---: | ---: |
| Across-seed mean return | 1.932% | 2.045% |
| Mean within-seed median return | 1.788% | 1.737% |
| Mean maximum drawdown | 1.527% | 1.537% |
| Mean turnover | 1,211,322.67 | 1,213,312.87 |
| Mean fees | 605.66 | 606.66 |
| Mean relative update | `1.46958e-5` | `1.45926e-5` |
| Gradient clipping frequency | 69.84% | 69.75% |
| Mean absolute approximate KL | `1.16388e-5` | `1.36219e-6` |
| Hidden parameter delta | 0.0554 | 0.0955 |
| Recurrent parameter delta | 0.0352 | 0.0355 |
| Actor parameter delta | 0.0697 | 0.0121 |
| Critic parameter delta | 0.000577 | 0.00811 |

All ten runs completed 310 PPO updates with zero invalid actions, policy
failures, deaths, or policy-ratio clipping. The calibration held across seeds:
AdamW's mean relative update was only 0.70% below momentum. AdamW again
redistributed movement strongly toward hidden and critic parameters and away
from the actor head. The seed-41 momentum and AdamW trace hashes exactly match
the retained Phase 3H and Phase 3I controls respectively.

## Decision

Retain momentum as the default. Calibrated AdamW is mechanically healthy and
promising, but five short training pairs do not establish a robust advantage:
the mean-return win coexists with a lower average median agent return, a wide
paired spread, and no material drawdown or cost improvement.

The next optimizer evidence gate should use longer-horizon or held-out data
with the rates frozen at `3e-4` and `3.5e-6`. Do not tune either optimizer on
that evaluation slice.
