# Midas

Midas is an offline, deterministic, multi-asset artificial-life market simulator written in Rust. The current build includes canonical data, point-in-time observations, close-to-next-open execution, independent portfolios, event-driven metrics, Burn inference/training behind a framework-neutral neural interface, simple recurrent memory, state-dependent frozen-encoder PPO head updates, deterministic checkpoints, and configurable two-parent genome evolution. It remains an experiment harness, not evidence of profitable trading.

The bundled data is synthetic and exists only for offline validation. It is not research evidence. A fixed modern asset universe also carries survivorship bias; real studies must document and address that limitation.

## Quick start

```bash
cargo run -p midas-cli -- validate-config --config configs/default.toml
cargo run -p midas-cli -- inspect-data --config configs/test.toml
cargo run -p midas-cli -- simulate --config configs/test.toml --split training
cargo run -p midas-cli -- benchmark --config configs/test.toml
cargo run -p midas-cli -- evolution-report --config configs/prompt2.toml --split training
cargo run -p midas-cli -- observations schema --config configs/default.toml
```

Save a complete final checkpoint with `--checkpoint checkpoints/run.bin`. The `resume` command accepts checkpoints captured at any deterministic step boundary:

```bash
cargo run -p midas-cli -- resume --config configs/test.toml --checkpoint checkpoints/run.bin
```

## Timing and data contract

Data uses a long-form canonical schema: `date, asset, open, high, low, close, adjusted_close, volume`. CSV and Parquet are loaded through Polars. `adjusted_close` must equal raw close within the configured tolerance; arbitrary provider back-adjustment is rejected and never used for features. In `reject` corporate-action mode, declared actions and likely undeclared discontinuities are fatal. In `explicit_events` mode, canonical split and cash-dividend records may be supplied separately. Raw closes are the execution and valuation source of truth. A point-in-time feature series multiplies each raw close by only split ratios whose event date is on or before that feature date; dividends are never folded into prices. At the beginning of `t+1`, before opening orders, each declared event on that date is applied once: splits multiply quantity and divide average entry price, while dividends credit cash by the held quantity times the declared amount. Holdings are valued at raw closes after the event. If a split and dividend share a date, the canonical order is split first and dividend second (the dividend amount is per post-split share).

An optional `data.corporate_actions_path` CSV uses `date,asset,kind,value`, where `kind` is `split` or `cash_dividend`. Split `value` is the positive new-shares-per-old-share ratio; dividend `value` is non-negative cash per share.

At close on day `t`, a policy sees features and availability known through `t`; its proposal is validated centrally and executes at open on `t+1`. Missing `t+1` bars reject that asset's order. Holdings in unavailable assets use the latest known close for valuation. Each step is transactional: policy state, neural memory, portfolios, counters, events, metrics, replacement state, RNG state, and `next_step` are committed together only after the full step succeeds.

Feature rows are calculated from current and earlier bars only. The modular observation registry has read-only market, portfolio, and agent-state providers that publish named raw values. An optional `rolling_market_stats` provider adds past-only raw-close momentum, volatility, trend, relative-strength, and market-summary senses only when explicitly selected in `[observations]`; the default schema remains Prompt 5-equivalent. A complete versioned schema, rather than provider registration order, maps those names to neural indices and applies the configured normalisation. Its stable hash is part of `BrainArchitecture` and checkpoint compatibility. Use `midas observations list`, `schema`, `describe <name>`, or `validate` with `--config` for offline introspection. Checkpoint format version 13 persists replacement-strategy state and safely reads format 12; format 11 remains incompatible because its positional observation-schema identity cannot establish compatibility with the provider registry. An enabled rolling schema also rejects a default-schema checkpoint rather than silently misaligning inputs.

## Architecture

`midas-core` contains the domain model and has no CLI dependency. The simulation engine composes these boundaries:

- `MarketDataSource`: load a canonical `MarketDataset`.
- `FeaturePipeline`: publish a versioned schema and legal feature rows.
- `ObservationProvider` and `ObservationRegistry`: publish named, typed read-only sensor values; `ObservationSchema` is the sole index/normalisation compiler.
- `Policy`: consume a typed observation and emit an untrusted `ProposedAction`.
- `ActionValidator`: authoritatively mask, normalise, cap, or fall back before execution.
- `ExecutionPlanner` and `ExecutionApplier`: create an immutable cost-aware proportional plan, then apply it atomically.
- `NeuralModel` and `NeuralModelFactory`: isolate Burn from policy and simulation interfaces; the current implementation is CPU `NdArray` only.
- `AgentFactory`: build both initial and replacement agents from `AgentSpawnRequest`.
- `ReplacementStrategy`: use read-only replacement context to choose a policy blueprint and optional genome.
- `EventSink`: consume deterministic `DomainEvent` values; operational events are separate.
- `LearningStrategy`: lifetime-learning boundary with `NoLearning` and clipped PPO implementations. Rollouts, GAE, shape-complete optimizer state, and learning updates are checkpointed.
- `EvolutionState`, parent selection, crossover, mutation, fitness, ancestry, and diversity helpers: deterministic genome-only inheritance by default. Lifetime-trained parameters and recurrent memory are never inherited.

The stable `BrainGenome` stores explicit matrix shapes, values, configured bounded traits, and complete architecture metadata. A neural policy keeps the immutable genome separate from mutable lifetime parameters, optimizer state, and simple recurrent runtime memory. Burn caches runtime tensors when a model is built or restored; serialisation and simulation interfaces do not expose Burn tensor types. Neural PPO now uses an **independent bounded Beta impulse distribution**: the actor emits two raw values per asset, which become positive Beta shapes through `(softplus(raw) + minimum_shape) / temperature`; each active asset samples a latent `u` and maps it to the incremental trade impulse `2u - 1`. The impulse is scaled by `[brain.trading].maximum_trade_fraction` of current net worth and executed against the persistent portfolio. Cash is not an action dimension. Unavailable assets are omitted from the latent vector and receive exact zero impulses. The latent vector is the PPO action, so its exact joint log probability is the sum of Beta log densities and its entropy is the sum of Beta entropies; no post-hoc clipping or fabricated probability is used. Stable asset order is explicit and remains a deterministic ordering choice for this version. Deterministic evaluation uses Beta means without consuming RNG or decaying temperature. The frozen hidden/recurrent encoder supplies a state representation to trainable actor and critic dense output heads, so head weights learn state-dependent trade impulses. Sequence minibatches preserve recurrent timestep order and are shuffled deterministically by sequence. Policy failures preserve diagnostics in `DomainEvent::PolicyFailure` and follow the configured failure limits before any cash fallback is used. A recoverable incident emits exactly `PolicyFailure` then `ActionFallback`, executes the fallback economically, and records one masked chronological rollout entry with the configured invalid-action penalty. Actor and critic eligibility are independent; inference and observation failures never fabricate probabilities or values. `KillAgent` kills the agent during the decision phase, emits `PolicyFailure` then `AgentDied`, and applies no next-day actions, trades, costs, age, or reward. `StopExperiment` discards the in-progress step, enters `SimulationLifecycle::Failed { reason }`, and preserves the last committed checkpoint boundary; repeated `run` and `run_steps` calls return the same failure without mutation. A valid validated action alone resets consecutive failures; total failures reset only for a newly spawned replacement. Built-in cash, equal-weight, and buy-and-hold benchmarks are exempt from neural concentration caps; buy-and-hold invests once then holds. At death, fitness is recorded and the evolutionary replacement strategy selects eligible parents, applies mating distance as a constraint on the configured fitness selector (with explicit relaxation events), crosses and mutates their birth genomes, assigns new genome/lineage identities, and spawns fresh lifetime state.

See [Observation Architecture](docs/OBSERVATIONS.md), [Extending Midas](docs/EXTENDING.md), and [Implementation Status](IMPLEMENTATION_STATUS.md) for subsystem-specific extension steps and the exact implementation boundary.

### Prompt 3B diagnostics

The long-horizon diagnostic suite is preserved in [reports/PROMPT3B_DIAGNOSTIC_EXPERIMENT.md](reports/PROMPT3B_DIAGNOSTIC_EXPERIMENT.md). It uses a frozen 1,800-date synthetic market, chronological training/validation/test splits, 10 paired primary seeds, and report-only JSONL/CSV traces. The traces decompose reward and execution costs, measure turnover and concentration, record asset-order positions, and expose PPO KL, clipping, entropy, gradients, and head deltas. The measured short-rollout PPO baseline paid substantially higher trading costs than static control; a conservative update schedule is the next isolated correction to validate, while no permanent turnover penalty is enabled. Real PPO-plus-evolution runs exercised deaths and two-parent reproduction, but did not establish evolutionary improvement. These are mechanical diagnostics, not profitability or intelligence claims.

## Data layout

Raw source files belong in `data/raw/` and should be treated as immutable. Reproducibly processed Parquet files belong in `data/processed/`. Assets, dimensions, dates, costs, population, capital, seeds, architecture, paths, policy, backend selection, and replacement selection are configuration values or derived runtime metadata.

## Verification

```bash
cargo fmt --all --check
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```
