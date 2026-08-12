# V3 Phase 3O — PPO Advantage-Signal Audit

Phase 3O measures how the normalized drawdown and turnover reward components
reach PPO advantages. It changes no training inputs, sample ordering, loss, or
gradient. Development remains confined to training and validation evidence;
the test split was not inspected.

## Instrumentation

Each PPO rollout now carries a parallel report-only reward decomposition. The
raw-return, drawdown-cost, turnover-cost, and residual-cost streams are passed
through the reward-contribution portion of the same GAE chronology and terminal
masks as the real update. For each actor-eligible update, diagnostics record:

- pre- and post-normalization advantage mean and standard deviation;
- normalization scale, sign flips, and rank inversions;
- each component signal's standard deviation;
- covariance and correlation with the pre-normalization advantage;
- projection onto the pre-normalization advantage direction.

The actual transition and checkpoint formats are unchanged. The parallel
component buffer is transactionally forked but excluded from checkpoints. The
first update after restoring a non-empty partial rollout is explicitly marked
incomplete; fresh and committed-boundary runs are complete.

Regressions compare instrumented and control learners and require identical
normalized samples, deterministic shuffle order, snapshots, and committed
state. A hand-calculated rollout verifies component propagation, normalization,
sign flips, and rank preservation. The simulation regression requires complete
diagnostics on real PPO updates.

## Replay protocol

Seeds 41 and 42 replay the exact Phase 3N training treatment concurrently:

```text
population=10 steps=1259 seed_derivation=splitmix_v2
optimizer=momentum learning_rate=3e-4 gradient_clip=1.0
reward=normalized_risk_adjusted_return
drawdown_penalty_weight=0.15 turnover_penalty_weight=0.0004
```

Both replayed `SimulationReport` objects and every pre-existing mechanics,
learning, and evolution diagnostic hash exactly match Phase 3N. Only the new
advantage-signal section differs, proving the instrumentation did not change
behavior.

## Measurements

All 6,304 PPO updates were complete, covering 25,180 actor samples.

| Signal | Measurement |
| --- | ---: |
| Mean pre-normalization advantage standard deviation | 0.003664 |
| Mean post-normalization advantage standard deviation | 0.999801 |
| Mean normalization scale | 453.82x |
| Advantage sign-flip fraction after centering | 33.31% |
| Rank inversions | 0 |

Normalization is a positive affine transform within each update. The observed
zero rank inversions confirms that it does not erase the penalty by changing
transition order, although centering changes which samples have positive and
negative policy-gradient coefficients.

| Reward contribution | Signal standard deviation | Fraction of advantage SD | Correlation with advantage | Projection on advantage |
| --- | ---: | ---: | ---: | ---: |
| Raw return | 0.002695 | 73.55% | +0.624 | +53.21% |
| Drawdown cost | 0.0000262 | 0.716% | +0.055 | +0.444% |
| Turnover cost | 0.0000283 | 0.773% | -0.081 | -0.026% |

The penalty streams survive GAE and normalization, but they are tiny relative
to the total advantage variation and barely project onto its direction. The
turnover projection is slightly negative, meaning its realized variation is
weakly opposed to the total advantage direction in this replay rather than
providing a useful independent control signal.

## Decision

Advantage normalization is not the primary defect, so disabling it is not
justified. Increasing scalar penalty weights is also not selected: the earlier
calibrations already showed worse validation behavior, and larger coefficients
would mix more cost noise into the same critic target without establishing
useful actor-gradient alignment.

The next isolated audit should calculate counterfactual actor-gradient vectors
for raw return, drawdown cost, and turnover cost without applying them. Their
norms and cosine similarity with the actual PPO actor gradient will distinguish
weak magnitude from destructive gradient conflict. If the costs have coherent
but underweighted gradients, a separate cost-advantage/Lagrangian PPO objective
is warranted. If their gradients are noisy or opposed, the cost definitions or
credit horizon must change first.

Raw summaries are in `reports/phase3o/`.
