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

Data uses a long-form canonical schema: `date, asset, open, high, low, close, adjusted_close, volume`. CSV and Parquet are loaded through Polars. Execution uses unadjusted prices; backward-looking return features use adjusted close. At close on day `t`, a policy sees features and availability known through `t`; its target allocation is executed at open on `t+1`. Missing `t+1` bars reject that asset's order. Holdings in unavailable assets use the latest known close for valuation.

Feature rows are calculated from current and earlier bars only. Configuration splits are chronological and non-overlapping. The feature schema has a version and identity hash; checkpoint compatibility includes this identity, the dataset content identity, configuration hash, and checkpoint version.

## Architecture

`midas-core` contains the domain model and has no CLI dependency. The simulation engine composes these boundaries:

- `MarketDataSource`: load a canonical `MarketDataset`.
- `FeaturePipeline`: publish a versioned schema and legal feature rows.
- `Policy`: consume a fixed observation and return target allocations.
- `ExecutionModel`: apply allocation targets and return transparent trades/rejections.
- `EventSink`: consume typed immutable simulation events.
- `ReplacementStrategy`: supply a replacement genome without owning market logic.
- `LearningStrategy`: lifetime-learning boundary; Prompt 1 provides only `NoLearning`.

The stable `BrainGenome` stores explicit matrix shapes, values, bounded traits, and format/architecture versions. `LifetimeBrain` owns a copy plus recurrent memory and counters. Burn is used for inference behind `Policy`; serialisation does not rely on Burn's private record format.

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
