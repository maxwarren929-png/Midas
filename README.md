# Midas

Midas is an offline, deterministic, multi-asset artificial-life market simulator written in Rust. This repository implements the Prompt 1 foundation: canonical data, leakage-safe observations, close-to-next-open execution, independent portfolios, event-driven metrics, real Burn inference, recurrent lifetime memory, and deterministic checkpoints. Lifetime reinforcement learning and evolutionary reproduction are intentionally deferred.

The bundled data is synthetic and exists only for offline validation. It is not research evidence. A fixed modern asset universe also carries survivorship bias; real studies must document and address that limitation.

## Quick start

```bash
cargo run -p midas-cli -- validate-config --config configs/default.toml
cargo run -p midas-cli -- inspect-data --config configs/test.toml
cargo run -p midas-cli -- simulate --config configs/test.toml --split training
cargo run -p midas-cli -- benchmark --config configs/test.toml
```

Save a complete final checkpoint with `--checkpoint checkpoints/run.bin`. The `resume` command accepts checkpoints captured at any deterministic step boundary:

```bash
cargo run -p midas-cli -- resume --config configs/test.toml --checkpoint checkpoints/run.bin
```

## Timing and data contract

Data uses a long-form canonical schema: `date, asset, open, high, low, close, adjusted_close, volume`. CSV and Parquet are loaded through Polars. `adjusted_close` must equal raw close within the configured tolerance; arbitrary provider back-adjustment is rejected and never used for features. In `reject` corporate-action mode, declared actions and likely undeclared discontinuities are fatal. In `explicit_events` mode, canonical split and cash-dividend records may be supplied separately. Splits adjust quantities and entry prices before next-open execution, and features build a forward-only point-in-time split series without rewriting prior observations.

An optional `data.corporate_actions_path` CSV uses `date,asset,kind,value`, where `kind` is `split` or `cash_dividend`. Split `value` is the positive new-shares-per-old-share ratio; dividend `value` is non-negative cash per share.

At close on day `t`, a policy sees features and availability known through `t`; its proposal is validated centrally and executes at open on `t+1`. Missing `t+1` bars reject that asset's order. Holdings in unavailable assets use the latest known close for valuation.

Feature rows are calculated from current and earlier bars only. A complete versioned observation schema orders market, portfolio, and agent-state features and derives the neural input dimension. Its stable hash is part of `BrainArchitecture` and checkpoint compatibility. Checkpoint format version 2 rejects older layouts explicitly.

## Architecture

`midas-core` contains the domain model and has no CLI dependency. The simulation engine composes these boundaries:

- `MarketDataSource`: load a canonical `MarketDataset`.
- `FeaturePipeline`: publish a versioned schema and legal feature rows.
- `Policy`: consume a typed observation and emit an untrusted `ProposedAllocation`.
- `ActionValidator`: authoritatively mask, normalise, cap, or fall back before execution.
- `ExecutionPlanner` and `ExecutionApplier`: create an immutable cost-aware proportional plan, then apply it atomically.
- `NeuralModel` and `NeuralModelFactory`: isolate Burn from policy and simulation interfaces.
- `AgentFactory`: build both initial and replacement agents from `AgentSpawnRequest`.
- `ReplacementStrategy`: use read-only replacement context to choose a policy blueprint and optional genome.
- `EventSink`: consume deterministic `DomainEvent` values; operational events are separate.
- `LearningStrategy`: lifetime-learning boundary; Prompt 1 provides only `NoLearning`.

The stable `BrainGenome` stores explicit matrix shapes, values, configured bounded traits, and complete architecture metadata. `LifetimeBrain` owns a copy plus recurrent memory and counters. Burn implements `NeuralModel`; serialisation and simulation interfaces do not expose Burn tensor types.

See [Extending Midas](docs/EXTENDING.md) for subsystem-specific extension steps and [Implementation Status](IMPLEMENTATION_STATUS.md) for the exact Prompt 1 boundary.

## Data layout

Raw source files belong in `data/raw/` and should be treated as immutable. Reproducibly processed Parquet files belong in `data/processed/`. Assets, dimensions, dates, costs, population, capital, seeds, architecture, paths, policy, backend selection, and replacement selection are configuration values or derived runtime metadata.

## Verification

```bash
cargo fmt --all --check
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```
