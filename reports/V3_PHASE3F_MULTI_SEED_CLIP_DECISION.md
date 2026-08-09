# V3 Phase 3F — Multi-Seed Gradient-Clip Decision

Phase 3F evaluates global gradient clipping thresholds 1.0 and 2.0 across five
paired 10-agent, 127-step runs. Each pair changes only the clipping threshold.

## Seed protocol correction

The initially proposed adjacent seeds 41–45 are not independent population
ensembles in the current implementation. Initial genome and policy seeds are
derived with `simulation.seed ^ agent_id`. For a 10-agent population, seeds 42
and 43 produce the same set of derived values in a different order; seeds 44
and 45 do the same.

Those runs were completed but excluded from the decision. The valid protocol
uses seeds 41, 73, 105, 137, and 169. Their derived genome-seed sets have zero
pairwise overlap for agent IDs 0–9.

This is a reproducibility finding, not merely an experimental inconvenience.
Future work should replace raw XOR derivation with a versioned deterministic
mixing function before relying on adjacent seed sweeps.

## Aggregate results

| Across-seed measurement | Clip 1.0 | Clip 2.0 |
| --- | ---: | ---: |
| Mean return | 1.988% | 2.091% |
| Median return | 2.254% | 2.209% |
| Return sample deviation | 0.457 pp | 0.491 pp |
| Mean turnover | 121,579.53 | 121,923.19 |
| Mean total fees | 607.90 | 609.62 |
| Mean maximum drawdown | 1.516% | 1.537% |
| Gradient clipping frequency | 70.87% | 22.63% |
| Relative parameter update | 0.00001495 | 0.00002274 |
| Mean absolute aggregate KL | 0.0000137 | 0.0000200 |
| Mean peak RSS | 46,193 KiB | 46,091 KiB |
| Mean concurrent wall time | 35.32 s | 35.62 s |
| Invalid actions / deaths | 0 / 0 | 0 / 0 |
| PPO policy clip fraction | 0 | 0 |

Clip 2.0 raises mean return by 0.1025 percentage points, but lowers the median,
increases variability, and produces small increases in turnover, fees, and
drawdown. It wins three of five return pairs rather than showing a consistent
advantage.

## Paired economic differences

Values below are clip 2.0 minus clip 1.0.

| Seed | Return | Turnover | Fees | Mean drawdown |
| ---: | ---: | ---: | ---: | ---: |
| 41 | -0.153 pp | +1,282.83 | +6.41 | -0.012 pp |
| 73 | +0.192 pp | -136.08 | -0.68 | +0.006 pp |
| 105 | +0.184 pp | +124.68 | +0.62 | +0.074 pp |
| 137 | -0.036 pp | -93.22 | -0.47 | +0.019 pp |
| 169 | +0.325 pp | +540.10 | +2.70 | +0.017 pp |

The paired direction is mixed. Seed 169 supplies the largest improvement,
while seed 41 supplies the largest deterioration.

## Temporal gradient behavior

The table averages first-half and second-half trace summaries across the five
valid seeds.

| Measurement | Clip 1 first | Clip 1 second | Clip 2 first | Clip 2 second |
| --- | ---: | ---: | ---: | ---: |
| Gradient clipping | 64.39% | 77.35% | 15.11% | 30.15% |
| GRU update gate | 0.00161 | 0.00228 | 0.00163 | 0.00232 |
| GRU reset gate | 0.00299 | 0.01035 | 0.00299 | 0.01039 |
| GRU candidate | 0.58008 | 0.68246 | 0.58111 | 0.68532 |
| Relative update | 0.00001466 | 0.00001523 | 0.00002113 | 0.00002436 |
| Mean absolute per-update KL | 0.000124 | 0.000172 | 0.000191 | 0.000289 |

Both thresholds preserve increasing gate gradients. Clip 2.0 permits larger
updates and KL, especially later in training, without reaching PPO ratio
clipping or invalid behavior.

## Decision

Retain the configured global clipping threshold of 1.0. Clip 2.0 is
mechanically safe, but the predefined adoption condition required consistent
economic improvement without material cost or drawdown degradation. The five
valid pairs do not meet that condition.

Do not begin AdamW against an ambiguous seed protocol. First introduce a
versioned deterministic seed mixer for agent genome, policy, and learning
streams, add regressions proving adjacent experiment seeds produce distinct
ensembles, and recapture the affected baseline. AdamW can then be compared
against the retained clip-1 momentum baseline.
