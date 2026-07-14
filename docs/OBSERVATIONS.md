# Observation Architecture

Midas observations are compiled from named read-only sensors. A provider is not a feature pipeline, policy, reward, execution model, or checkpoint participant. It receives an immutable `ObservationContext` at close `t` and publishes finite raw values by name. The registry owns registration, duplicate detection, metadata, and deterministic provider execution; the schema owns every neural-input index.

## Built-in providers

The default schema registers three providers:

- `market`: the configured point-in-time market feature for each asset.
- `portfolio`: cash fraction, per-asset allocation, unrealised return, net-worth ratio, drawdown, one-step portfolio return, and turnover ratio when selected by the schema.
- `agent_state`: normalised age when selected by the schema.

Every published scalar has a stable name such as `market.return1.asset_0`, `portfolio.cash_fraction`, or `portfolio.unrealized_return.asset_1`. Metadata includes its provider, numeric value type, description, units, scalar or fixed-vector shape, optional source-order hint, suggested normalisation, schema version, and tags. Asset indices deliberately follow the validated runtime asset order, which prevents provider registration order from affecting an input vector.

## Schema and normalisation

`[observation]` remains the compact backwards-compatible source for the default feature groups. `[observations]` chooses the built-in registry and providers:

```toml
[observations]
schema = "default"
providers = ["market", "portfolio", "agent_state"]

[observations.normalizations."portfolio.cash_fraction"]
kind = "min_max"
minimum = 0.0
maximum = 1.0
```

The compiler expands these into `ObservationSchemaEntry` records containing an explicit index, name, provider, component, and normalisation rule. Supported rules are `none`, `ratio_clipped`, `min_max`, `z_score`, `log`, `signed_log`, `percentage`, and `custom` (finite affine scale and offset). Providers publish raw values; only compilation normalises them. Existing default clipped-ratio semantics are retained exactly.

The schema rejects duplicate names or indices, gaps in index order, provider/source disagreement, missing providers, unreferenced published metadata, non-finite values, shape mismatches, invalid normalisation parameters, and unknown normalisation overrides. Increment `[observation].version` whenever names, dimensions, ordering, providers, or normalisation semantics change.

## Introspection

```bash
cargo run -p midas-cli -- observations list --config configs/default.toml
cargo run -p midas-cli -- observations schema --config configs/default.toml
cargo run -p midas-cli -- observations describe portfolio.cash_fraction --config configs/default.toml
cargo run -p midas-cli -- observations validate --config configs/default.toml
```

The schema output is the authoritative input map. It is suitable for recording experimental inputs without exposing neural framework values.

## Compatibility and future providers

The schema hash includes its version, entries, providers, and normalisation rules. It is embedded in `BrainArchitecture` and checkpoint compatibility. Checkpoint format 12 rejects format 11 because that format predates the registry identity; no positional-schema migration is attempted.

Future indicators, external data, or evolutionary sensor selection must add providers or explicit schema entries through this boundary. They must not append positional fields in `SimulationEngine`, modify PPO input handling, or make a provider mutate simulation state. Future genome-controlled sensor subsets can be represented as a selected schema without changing the policy or learning interfaces.
