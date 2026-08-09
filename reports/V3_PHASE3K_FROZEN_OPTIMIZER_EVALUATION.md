# V3 Phase 3K — Frozen Optimizer Evaluation

Phase 3K evaluates the five paired Phase 3J training checkpoints without
learning on the validation and test splits. Rates were frozen at momentum
`3e-4` and AdamW `3.5e-6`; neither optimizer was tuned from held-out results.

## Evaluation contract

The existing checkpoint-evaluation path restores immutable genomes and trained
lifetime parameters, resets portfolios and recurrent state, and disables
learning, evolution, replacement, and automatic checkpoint writes. Evaluation
used fresh deterministic SplitMix V2 evaluation streams.

The diagnostics CLI now accepts `--evaluate-checkpoint`, allowing the exact
training overrides and configuration hash to be reconstructed without
hand-edited configuration copies. Every held-out run reported zero PPO updates,
invalid actions, policy failures, and deaths.

| Split | Dates | Steps |
| --- | --- | ---: |
| Validation | 2014-11-03 to 2015-11-13 | 269 |
| Test | 2015-11-16 to 2016-11-25 | 269 |

## Validation results

| Seed | Momentum mean return | AdamW mean return | AdamW delta | Median delta | Drawdown delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 41 | -2.948% | -2.961% | -0.012 pp | -0.259 pp | -0.091 pp |
| 42 | -2.346% | -2.131% | +0.215 pp | +0.654 pp | -0.415 pp |
| 43 | -0.187% | 0.241% | +0.428 pp | +0.526 pp | +0.128 pp |
| 44 | -1.699% | -1.341% | +0.358 pp | -0.204 pp | -0.064 pp |
| 45 | -0.542% | -0.531% | +0.012 pp | +0.309 pp | -0.358 pp |

AdamW won mean return and drawdown in four of five validation pairs. Its mean
return was -1.344% versus momentum's -1.544%, its mean paired advantage was
0.200 percentage points, and its average median-agent advantage was 0.205
percentage points. It reduced mean drawdown by 0.160 percentage points and
aggregate turnover by about 2.8%. Both optimizers lost money on average in this
regime, so the result was relative robustness rather than profitability.

## Untouched test results

| Seed | Momentum mean return | AdamW mean return | AdamW delta | Median delta | Drawdown delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 41 | -1.479% | -1.318% | +0.161 pp | -0.274 pp | +0.053 pp |
| 42 | 1.951% | 1.878% | -0.072 pp | -0.967 pp | +0.141 pp |
| 43 | 0.528% | 0.024% | -0.504 pp | -0.609 pp | +0.498 pp |
| 44 | 0.142% | 0.754% | +0.612 pp | +0.510 pp | -0.131 pp |
| 45 | -1.796% | -1.845% | -0.049 pp | -0.464 pp | -0.076 pp |

On the untouched test split, AdamW won mean return in only two of five pairs.
Its across-seed mean was -0.101% versus momentum's -0.131%, only a 0.030
percentage-point paired advantage with a 0.406 percentage-point paired sample
standard deviation. AdamW's average median-agent return was 0.361 percentage
points worse, and its mean drawdown was 0.097 percentage points worse.

## Decision

Retain momentum `3e-4` as the default optimizer. AdamW `3.5e-6` is correctly
implemented, deterministic, scale calibrated, and mechanically healthy, but
its training and validation advantage did not generalize robustly to the
untouched test split. AdamW remains an explicit experimental option; no further
rate tuning should use these validation or test intervals.

The broader result is also cautionary: neither optimizer demonstrated stable
profitability across regimes. Future intelligence work should focus on the
learning signal, observations, objectives, and evaluation protocol rather than
continuing optimizer selection on this dataset.
