# V3 Phase 3D — Paired Population Gradient Baseline

Phase 3D compares 10 and 20 agents under the same 127-step combined-senses PPO
protocol. Both processes ran concurrently on the same eight-way host, matching
the intended two-run workload for a 16 GiB machine.

## Protocol

- Config: `configs/prompt6/combined-ppo.toml`
- Seed: 41
- Split: training
- Horizon: 127 committed steps
- Populations: 10 and 20
- Architecture, PPO settings, data, costs, and action distribution: identical
- Outputs: report-only traces, JSON summaries, and format-19 checkpoints

The diagnostics CLI gained explicit `--steps`, `--population-size`, `--output`,
and `--checkpoint` options. It also records wall time and Linux peak RSS in the
summary. These options modify only the diagnostic invocation; the underlying
simulation configuration and checkpoint contract are unchanged.

## Aggregate results

| Measurement | 10 agents | 20 agents |
| --- | ---: | ---: |
| PPO updates | 310 | 620 |
| Internal optimizer steps | 2,480 | 4,960 |
| Neural actions | 1,270 | 2,540 |
| Invalid actions | 0 | 0 |
| Deaths | 0 | 0 |
| Peak RSS | 46,668 KiB | 51,748 KiB |
| Contended wall time | 36.28 s | 64.71 s |
| Checkpoint size | 1,318,653 B | 2,620,729 B |
| Trace rows | 1,580 | 3,160 |

Doubling population added only 5,080 KiB of peak RSS, but approximately doubled
persistent checkpoint and report volume. Wall time rose by 78%; it is a
concurrent-throughput measurement rather than an isolated scaling benchmark.

## Gradient health

| Mean measurement | 10 agents | 20 agents |
| --- | ---: | ---: |
| Global norm before clipping | 1.58453 | 1.59621 |
| Global norm after clipping | 0.92056 | 0.92123 |
| Gradient clipping frequency | 72.58% | 73.15% |
| Hidden layer 1 | 0.64596 | 0.66379 |
| Hidden layer 2 | 0.41842 | 0.41699 |
| Complete recurrent | 0.67247 | 0.66013 |
| GRU update gate | 0.00193 | 0.00205 |
| GRU reset gate | 0.00662 | 0.00729 |
| GRU candidate | 0.67241 | 0.66006 |
| Actor | 1.14687 | 1.15514 |
| Critic | 0.00382 | 0.00359 |
| Relative parameter update | 0.0000152 | 0.0000150 |
| PPO policy clip fraction | 0 | 0 |

Population size did not materially change normalized gradient behavior. The
global gradient clip activates frequently, even though the policy-ratio clip
does not activate and approximate KL remains near zero.

## Temporal comparison

| Measurement | 10 first half | 10 second half | 20 first half | 20 second half |
| --- | ---: | ---: | ---: | ---: |
| Pre-clip norm | 1.35067 | 1.81839 | 1.36319 | 1.82923 |
| Gradient clipping | 66.13% | 79.03% | 66.29% | 80.00% |
| Update gate | 0.00163 | 0.00223 | 0.00170 | 0.00239 |
| Reset gate | 0.00317 | 0.01008 | 0.00321 | 0.01138 |
| Candidate | 0.60000 | 0.74483 | 0.58976 | 0.73037 |
| Relative update | 0.0000148 | 0.0000156 | 0.0000147 | 0.0000152 |

The small GRU gates are not frozen: update gradients increase and reset
gradients more than triple in the second half. At the same time, the global
norm and clipping frequency rise materially.

## Paired determinism

The 20-agent traces were filtered to agents 0–9 and canonicalized. Their hash
matches the complete canonicalized 10-agent traces exactly:

`226b01f2a42b474dc12a78943116341d02f85caeece452e3631479f368e62399`

Whole checkpoints necessarily differ because population state is part of the
checkpoint:

- 10 agents: `4aecbd564febf670c4678c457e366658362f2632aed449fdfc903916b5cc9ad3`
- 20 agents: `9413dfe70dba9c8d08ff84987cf1cdad3426c96ef2f8298e7ae38b6e819ffa62`

## Decision

Twenty agents are safe within 16 GiB but do not improve normalized gradient
health enough to justify doubling experiment volume for optimizer tuning. Use
10 agents for the next controlled comparison.

The next isolated change should compare `gradient_clip = 1.0` with `2.0` using
the same 10-agent, 127-step protocol. The goal is to determine whether less
aggressive clipping improves gate learning without causing excessive KL,
policy-ratio clipping, invalid actions, or unstable relative updates. AdamW
should remain deferred until that result is available.
