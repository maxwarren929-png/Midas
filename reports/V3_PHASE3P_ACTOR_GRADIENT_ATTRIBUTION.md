# V3 Phase 3P — Counterfactual Actor-Gradient Attribution

Phase 3P measures the parameter-space actor gradients implied by each reward
component without applying them. It follows Phase 3O's finding that normalized
risk costs survive GAE but barely project onto the scalar advantage.

## Instrumentation boundary

At every real PPO optimization substep, the audit reuses the exact:

- current policy ratio and actual PPO clipping gate;
- Beta action log-probability derivatives;
- recurrent representation produced for the real sample;
- actor-eligible mask and minibatch divisor;
- component advantages centered and scaled by the real update's advantage
  normalization.

It constructs counterfactual raw-return, drawdown-cost, turnover-cost, and
residual-cost gradients for the actor head before global gradient clipping and
optimizer state. Entropy and critic derivatives are excluded so the comparison
isolates the PPO policy gradient. The vectors are reduced to norms and cosine
similarities, recorded in report traces, and never applied.

No transition or checkpoint format changes. A geometry regression requires a
sole raw-return component to reproduce the exact actual policy-gradient norm
and direction. Learner snapshots, sample order, actual advantages, losses, and
updates remain unchanged.

## Replay

Seeds 41 and 42 replay the Phase 3N/3O normalized-reward protocol:

```text
population=10 steps=1259 seed_derivation=splitmix_v2
optimizer=momentum learning_rate=3e-4 gradient_clip=1.0
reward=normalized_risk_adjusted_return
drawdown_penalty_weight=0.15 turnover_penalty_weight=0.0004
```

All 6,304 PPO updates and 25,180 actor samples produced gradient attribution.
Both `SimulationReport` hashes and every earlier mechanics, learning, and
evolution diagnostic hash exactly match Phase 3O.

## Results

| Actor-head gradient | Mean norm | Norm / actual | Cosine with actual |
| --- | ---: | ---: | ---: |
| Actual PPO policy gradient | 1.3278 | 100% | 1.000 |
| Raw-return component | 1.0331 | 77.81% | +0.594 |
| Drawdown-cost component | 0.00972 | 0.732% | +0.050 |
| Turnover-cost component | 0.01790 | 1.348% | -0.081 |

Pairwise component geometry:

| Pair | Mean cosine |
| --- | ---: |
| Raw return / drawdown cost | +0.075 |
| Raw return / turnover cost | -0.0002 |
| Drawdown cost / turnover cost | +0.077 |

The cost vectors are non-zero and nearly orthogonal to the return vector. The
turnover vector weakly conflicts with the actual scalar PPO direction. Their
small norms are therefore not merely a reporting-scale artifact: scalar PPO
gives these economically distinct directions almost no control authority.
The pattern is consistent across both seeds.

## Decision

The evidence supports a constrained objective with independently scaled cost
advantages. It does not support another scalar-weight sweep, disabling
advantage normalization, or increasing model capacity.

Implementation should be staged:

1. Introduce an opt-in turnover-constrained PPO mode first. Turnover is dense,
   dimensionless, and has the clearer conflicting gradient.
2. Give turnover its own cost return/advantage normalization and cost critic,
   leaving the return critic target uncontaminated.
3. Use a non-negative checkpointed Lagrange multiplier updated toward an
   explicit turnover budget; include bounds and deterministic dual updates.
4. Keep drawdown report-only initially. Its incremental signal is sparse and
   should not force a second critic/head migration until turnover-constrained
   PPO passes mechanical and two-seed gates.
5. Compare against exact matched final-genome and unconstrained controls before
   any five-seed promotion.

This is a modest objective/head extension, not a larger general network. It
will require explicit architecture and checkpoint compatibility work because a
cost critic is authoritative trainable state.

Raw summaries are in `reports/phase3p/`.
