# Implementation Status

## Prompt 1 implemented

- Cargo workspace with `midas-core` and `midas-cli`, pinned by `Cargo.lock`.
- Strict typed TOML with unknown-field rejection, bounded numeric validation, derived dimensions, and chronological non-overlapping splits.
- Strong `AssetId`, `AgentId`, and `ExperimentId` types.
- Polars CSV/Parquet ingestion into validated long-form `MarketBar` data, union trading calendar, latest-close fallback, and content identity.
- Versioned configurable backward-only feature pipeline with leakage tests.
- Close `t` observation to open `t+1` execution, long-only allocations, cash fallback, next-day unavailable-order rejection, fractional shares, sells-before-buys, and proportional cash limiting.
- Independent portfolios; commission, spread, slippage, turnover, minimum trade, survival liabilities, close valuation, bankruptcy, death, and replacement events.
- Typed events and event-driven performance metrics including return, annualisation, volatility, drawdown, Sharpe-like and Sortino-like values, turnover, costs, trades, survival, and bankruptcy.
- Policy, data-source, feature, execution, event-sink, learning, and replacement traits.
- Cash, seeded-random, equal-weight, and configured-benchmark buy-and-hold policies.
- Configurable Burn CPU actor-critic inference with runtime-derived input/output sizes, configurable dense layout and activation, deterministic masking/capping, persistent recurrent memory, stable genome matrices, and separate lifetime state.
- Explicit `NoLearning` and explicitly non-evolutionary random-genome replacement.
- Versioned deterministic checkpoints with configuration, dataset, feature-schema, policies/genomes/lifetime memory, portfolios, metrics, events, counters, and accessed-through date.
- CLI validation, inspection, simulation, resume, and benchmark commands plus offline synthetic data.
- Unit, integration, property, leakage, Parquet, timing, portfolio-isolation, Burn determinism/memory, and checkpoint-resume tests.

## Known Prompt 1 limitations

- CPU `NdArray` is the only configured Burn backend implemented in this build; the policy boundary and stable genome do not require engine redesign for another backend.
- Metrics do not yet calculate a meaningful trade win rate because lot-level realised P&L history is not retained.
- Random/sequential training episode selection and walk-forward window generation are not exposed by the current CLI; simulations run the complete configured chronological split.
- Forced final liquidation is not implemented; final holdings are valued at the last close, which is the specification default.
- Detailed event persistence is in-memory; additional sinks can implement durable output.
- The synthetic dataset is intentionally small and does not meet the recommended research demonstration universe or eliminate survivorship bias.
- Agent inference is deterministic and independent but currently sequential; stable indexed parallel inference is reserved as a performance extension.

## Reserved exactly for Prompt 2

- PPO training, transition and rollout buffers, GAE, clipped policy objectives, critic/entropy losses, optimiser updates/state, gradient clipping and rollback, reward normalisation, and all lifetime parameter updates.
- Two-parent selection, crossover, mutation, reproductive fitness, genetic diversity metrics, ancestry/family trees, and stateful evolutionary RNG streams.
- Lamarckian inheritance; the genome remains unchanged by lifetime state.

No Prompt 1 component is labelled PPO or evolutionary reproduction.
