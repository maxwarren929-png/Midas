# V3 Phase 3E — Global Gradient-Clip Experiment

Phase 3E compares the current global gradient clipping threshold of 1.0 with a
relaxed threshold of 2.0. Both 10-agent processes ran concurrently using the
same seed, initial genomes, data, architecture, PPO settings, and 127-step
horizon.

## Aggregate learning results

| Measurement | Clip 1.0 | Clip 2.0 |
| --- | ---: | ---: |
| PPO updates | 310 | 310 |
| Internal optimizer steps | 2,480 | 2,480 |
| Gradient clipping frequency | 72.58% | 23.43% |
| Global norm before clipping | 1.58453 | 1.58610 |
| Global norm after clipping | 0.92056 | 1.37172 |
| Relative parameter update | 0.0000152 | 0.0000232 |
| Approximate KL | -0.0000050 | -0.0000237 |
| PPO policy clip fraction | 0 | 0 |
| Invalid actions | 0 | 0 |
| Deaths | 0 | 0 |

The threshold of 2.0 allows roughly 52.5% larger relative updates while
retaining extremely small KL and no policy-ratio clipping.

## Component gradients and lifetime deltas

Component gradient norms are measured before clipping, so similar values are
expected when the threshold alone changes.

| Measurement | Clip 1.0 | Clip 2.0 |
| --- | ---: | ---: |
| Hidden layer norms | 0.64596, 0.41842 | 0.64771, 0.41732 |
| Complete recurrent norm | 0.67247 | 0.67246 |
| GRU update gate | 0.00193 | 0.00197 |
| GRU reset gate | 0.00662 | 0.00663 |
| GRU candidate | 0.67241 | 0.67240 |
| Actor norm | 1.14687 | 1.14883 |
| Critic norm | 0.00382 | 0.00383 |
| Hidden lifetime delta | 0.06021 | 0.09884 |
| Recurrent lifetime delta | 0.03577 | 0.05820 |
| Actor lifetime delta | 0.06351 | 0.09301 |
| Critic lifetime delta | 0.000613 | 0.000766 |

Clip 2.0 does not alter the raw gradient distribution, but it applies more of
that signal to every trainable component.

## Temporal behavior

| Measurement | Clip 1 first | Clip 1 second | Clip 2 first | Clip 2 second |
| --- | ---: | ---: | ---: | ---: |
| Pre-clip norm | 1.35067 | 1.81839 | 1.35234 | 1.81985 |
| Post-clip norm | 0.90115 | 0.93997 | 1.25884 | 1.48461 |
| Gradient clipping | 66.13% | 79.03% | 15.24% | 31.61% |
| Update gate | 0.00163 | 0.00223 | 0.00166 | 0.00229 |
| Reset gate | 0.00317 | 0.01008 | 0.00315 | 0.01012 |
| Candidate | 0.60000 | 0.74483 | 0.60101 | 0.74379 |
| Relative update | 0.0000148 | 0.0000156 | 0.0000212 | 0.0000251 |

The relaxed threshold continues to clip more often later in training, but no
longer suppresses most optimizer minibatches.

## Resource and economic observations

| Measurement | Clip 1.0 | Clip 2.0 |
| --- | ---: | ---: |
| Peak RSS | 46,492 KiB | 46,676 KiB |
| Concurrent wall time | 33.79 s | 35.07 s |
| Mean ending net worth | 10,236.23 | 10,220.94 |
| Mean return | 2.362% | 2.209% |
| Mean turnover | 122,863.10 | 124,145.93 |
| Total fees | 614.32 | 620.73 |

The economic difference is small and comes from one short synthetic seed. It
does not establish that either threshold is superior. Clip 2.0 slightly
increased trading and costs in this run.

## Artifacts and deterministic control

- Clip 1 checkpoint:
  `4aecbd564febf670c4678c457e366658362f2632aed449fdfc903916b5cc9ad3`
- Clip 2 checkpoint:
  `d5de876f87747b278483b6c73d08b9fef581a9612f881f0d4f33752dcc0edfe3`

The clip-1 checkpoint exactly matches the independently produced Phase 3D
10-agent checkpoint. This confirms deterministic reproduction across the two
experiment stages.

## Decision

Clip 2.0 passes the mechanical stability gate and is a credible candidate. It
should not replace 1.0 based on one seed because the larger updates did not
improve the observed economic result.

Run a paired multi-seed comparison next, retaining 10 agents and 127 steps. The
selection criteria should prioritize invalid-action rate, KL, PPO clipping,
turnover/costs, drawdown, and consistency of returns—not return alone. AdamW
remains deferred until the clipping threshold is selected.

Phase 3F completed that comparison using disjoint seed ensembles and retained
clip 1.0; see `reports/V3_PHASE3F_MULTI_SEED_CLIP_DECISION.md`.
