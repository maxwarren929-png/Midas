# Implementation Status

## Prompt 1 implemented

- Cargo workspace with `midas-core` and `midas-cli`, pinned by `Cargo.lock`.
- Strict typed TOML with unknown-field rejection, bounded numeric validation, derived dimensions, and chronological non-overlapping splits.
- Strong `AssetId`, `AgentId`, and `ExperimentId` types.
- Polars CSV/Parquet ingestion into validated long-form `MarketBar` data, union trading calendar, latest-close fallback, and content identity.
- Explicit corporate-action modes, provider-adjustment rejection, discontinuity detection, tested split accounting, and forward-only point-in-time split features.
- Versioned schema-driven observations with stable hashes, configured ordering and dimensions, and correct unrealised-return features.
- Close `t` observation to open `t+1` execution, authoritative shared action validation, long-only masking/capping/fallback, and immutable cost-aware proportional execution planning.
- Independent portfolios; commission, spread, slippage, turnover, minimum trade, survival liabilities, close valuation, bankruptcy, death, and replacement events.
- Typed events and event-driven performance metrics including return, annualisation, volatility, drawdown, Sharpe-like and Sortino-like values, turnover, costs, trades, survival, and bankruptcy.
- Policy, neural-model/factory, agent-factory, action-validator, data-source, feature, observation-builder, execution-planner/applier, event-sink, learning, and replacement traits.
- Cash, seeded-random, equal-weight, and configured-benchmark buy-and-hold policies.
- Burn CPU actor-critic inference behind framework-neutral neural interfaces, persistent recurrent memory, configured evolvable-trait bounds, stable genome matrices, and complete architecture/schema validation.
- Explicit `NoLearning` and explicitly non-evolutionary random-genome replacement.
- Checkpoint format v2 with lifecycle, complete observation schema and architecture, policies/genomes/lifetime memory, portfolios, metrics, deterministic domain events, counters, and accessed-through date. Operational events are deliberately excluded.
- CLI validation, inspection, simulation, resume, and benchmark commands plus offline synthetic data.
- Twenty-five focused regressions covering all seven repaired findings, CSV/Parquet, property invariants, mock and Burn neural models, replacement consistency, lifecycle idempotence, and checkpoint-resume domain-event equivalence.

## Known Prompt 1 limitations

- CPU `NdArray` is the only configured Burn backend implemented in this build; the policy boundary and stable genome do not require engine redesign for another backend.
- Metrics do not yet calculate a meaningful trade win rate because lot-level realised P&L history is not retained.
- Random/sequential training episode selection and walk-forward window generation are not exposed by the current CLI; simulations run the complete configured chronological split.
- Forced final liquidation is not implemented; final holdings are valued at the last close, which is the specification default.
- Detailed event persistence is in-memory; additional sinks can implement durable output.
- The synthetic dataset is intentionally small and does not meet the recommended research demonstration universe or eliminate survivorship bias.
- Agent inference is deterministic and independent but currently sequential; stable indexed parallel inference is reserved as a performance extension.
- Explicit corporate actions currently support splits and cash dividends supplied through a separate CSV; embedded provider-specific action encodings are not inferred.

## Reserved exactly for Prompt 2

- PPO training, transition and rollout buffers, GAE, clipped policy objectives, critic/entropy losses, optimiser updates/state, gradient clipping and rollback, reward normalisation, and all lifetime parameter updates.
- Two-parent selection, crossover, mutation, reproductive fitness, genetic diversity metrics, ancestry/family trees, and stateful evolutionary RNG streams.
- Lamarckian inheritance; the genome remains unchanged by lifetime state.

No Prompt 1 component is labelled PPO or evolutionary reproduction.
