# Implementation Status

## Prompt 1.75 verification and hardening

The Prompt 1.75 audit and regression pass is implemented in the core market, execution, observation, brain, agent, replacement, simulation, configuration, metrics, and integration-test modules. The current foundation is verified rather than advertised beyond its scope.

### Final policy-failure correctness repair

`SimulationEngine::step` now forks mutable agent and policy state into a per-step transaction. Events, metrics, counters, portfolios, corporate actions, replacement strategy state, and the step boundary are committed only after the transaction succeeds. `StopExperiment` enters `Failed { reason }` and records only diagnostic failure metadata; `KillAgent` commits immediate policy-failure death at the observation close without fallback or next-day effects. Recoverable failures have the single `PolicyFailure` then `ActionFallback` sequence. Checkpoint format 4 records the failed lifecycle without serializing a half-applied step. The 43-test foundation suite includes failure injection, atomicity, replay, counter, event-order, and exactly-once split/dividend regressions.

- Corporate actions have one explicit convention: raw opens/closes remain execution and valuation prices; `FeaturePriceSeries` applies only split ratios dated on or before the requested feature date; dividends stay out of the feature price and credit cash exactly once at the beginning of the next-open execution date. Same-day actions are ordered split then dividend. Split, reverse-split, multiple-split, dividend, discontinuity-rejection, and future-leakage regressions cover the contract.
- Backend/device selection is authoritative through `BackendConfig::compute_device` and `AgentBuildContext`. Only the implemented CPU Burn `NdArray` `default` device is accepted; unsupported device names fail configuration validation. Checkpoint format 4 records the backend identity and rejects incompatible restores.
- Burn remains private to the neural adapter. `NeuralModel`/`NeuralModelFactory` support the mock external model and Burn model through the same agent/simulation path. Burn runtime tensors are built once per model build/restore and reused; immutable `BrainGenome`, mutable lifetime parameters, and recurrent runtime state are separate.
- Policy and observation failures preserve their original diagnostics in `DomainEvent::PolicyFailure`, increment per-agent total/consecutive counters, and follow configurable continue, kill-agent, or stop-experiment behavior before cash fallback. The shared observation builder validates dimensions and normalization instead of panicking. The step transaction buffers all domain events and mutable agent/replacement state until commit; stop failures enter a persistent failed lifecycle without advancing the boundary, while fatal agent failures die before next-day actions and never emit a fallback.
- Long-only action validation covers finite/non-negative proposals, availability masks, concentration limits, cash redistribution, tiny caps, all-unavailable markets, and numerical edge cases. Execution planning includes costs before proportional scaling, uses stable asset IDs, converges minimum-trade filtering deterministically, and the applier recomputes trade economics before atomic mutation.
- Checkpoint tests compare uninterrupted/resumed domain histories, portfolios, agent states, policy blueprints, serialized completed state, lifecycle idempotence, observation schema, corporate-action data, backend compatibility, and deterministic failure replay. Operational checkpoint events remain intentionally outside deterministic domain history; a failed checkpoint may contain failure metadata but always retains the last committed simulation state.
- Replacement contexts expose read-only living-agent summaries (net worth, age, drawdown, birth date, parent IDs, policy blueprint, and borrowed optional genome references) plus architecture/schema/device/seed material. A deterministic selection strategy test spawns through `StandardAgentFactory`; two-parent selection and reproduction remain deferred.
- The primary integration suite now contains 43 executable regressions, including property, permutation, policy-failure, atomicity, replay, and exactly-once corporate-action tests. No network or external data dependency is required.

Files changed in this hardening pass include `crates/midas-core/src/{agent,brain,replacement,simulation}.rs`, `crates/midas-core/tests/foundation.rs`, `README.md`, `docs/EXTENDING.md`, and this status document.

Known remaining limitations are deliberate: CPU `NdArray` is the only neural backend; memory is a simple recurrent layer, not a GRU; corporate actions are explicit split/dividend records and are not inferred from providers; event persistence is in-memory; and there is no PPO, lifetime optimizer state, mutation, crossover, two-parent reproduction, reproductive fitness, diversity, ancestry, or Lamarckian inheritance. Those items remain reserved for Prompt 2.

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
- Checkpoint format v4 with lifecycle (including persistent failure reason), complete observation schema and architecture, policies/genomes/lifetime memory, portfolios, metrics, deterministic domain events, counters, backend identity, and accessed-through date. Operational events are deliberately excluded from deterministic equivalence.
- CLI validation, inspection, simulation, resume, and benchmark commands plus offline synthetic data.
- The Prompt 1 foundation suite began with twenty-five regressions; Prompt 1.75 expands it to 43 focused regressions covering corporate-action timing, backend validation, policy diagnostics, execution permutations, schema edge cases, mock and Burn neural models, replacement readiness, lifecycle idempotence, atomic failed steps, and checkpoint equivalence.

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
