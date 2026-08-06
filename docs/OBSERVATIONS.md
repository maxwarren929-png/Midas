# Observation Architecture

Midas observations are compiled from named read-only sensors. A provider is not a feature pipeline, policy, reward, execution model, or checkpoint participant. It receives an immutable `ObservationContext` at close `t` and publishes finite raw values by name. The registry owns registration, duplicate detection, metadata, and deterministic provider execution; the schema owns every neural-input index.

## Module layout

`crates/midas-core/src/observation.rs` is a compatibility facade: public
imports remain available through `midas_core::observation::*`. Implementation
is split into `normalization.rs` (rules and conversion), `schema.rs` (source
types, configuration compilation, validation, and identity), `contracts.rs`
(provider contracts and runtime values), `registry.rs` (generic registration),
and `builder.rs` (scratch-backed vector compilation). Built-in wiring and the
market, portfolio, agent-state, and rolling providers live under
`observation/providers/`. Moving these modules did not change Prompt 5/6
observation names, entry order, schema identities, or checkpoint format.

## Built-in providers

The default schema registers three providers:

- `market`: the configured point-in-time market feature for each asset.
- `portfolio`: cash fraction, per-asset allocation, unrealised return, net-worth ratio, drawdown, one-step portfolio return, and turnover ratio when selected by the schema.
- `agent_state`: normalised age when selected by the schema.

`agent_awareness` is an additional opt-in provider for V2 self-awareness:
recent clipped reward, an EMA reward mean, EMA reward volatility, exploration
temperature relative to birth, daily survival cost relative to net worth, and
the sign of short- versus long-term reward EMAs. These values are persisted in
the agent state and are only added to the neural vector when explicitly listed
in `[observation].features` and `[observations].providers`.

`rolling_market_stats` is an opt-in fourth provider introduced in Prompt 6. It
precomputes immutable, past-only values from the canonical point-in-time raw
close series and publishes them by decision date. It has no dependency on PPO,
execution, evolution, rewards, or mutable simulation state. The default schema
does not register it, so Prompt 5 inputs and checkpoint identity remain
unchanged unless it is explicitly enabled.

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

## Rolling market statistics

Enable the provider explicitly and increment the compact observation schema
version whenever its entries change:

```toml
[observation]
version = 3

[observations]
schema = "default"
providers = ["market", "portfolio", "agent_state", "rolling_market_stats"]

[observations.rolling_market_stats]
enabled = true
return_windows = [20, 60]
volatility_windows = [20, 60]
sma_windows = [60]
relative_strength_windows = [20, 60]
market_summary = true
market_summary_window = 20
minimum_history = "zero_fill"
```

The provider emits named raw scalars such as `rolling_return_60.SPY`,
`rolling_volatility_20.AAA`, `price_to_sma_60.BBB`,
`relative_strength_20.CCC`, `market_mean_return_20`,
`market_dispersion_20`, and `market_breadth_20`. Asset symbols and window
lengths are part of each stable name. Per-name schema normalisation overrides
continue to apply after publication.

At close `t`, using only bars at or before `t`, the formulas are:

- `rolling_return_w = close_t / close_(t-w) - 1`.
- `rolling_volatility_w` is the population standard deviation of the most
  recent `w` simple close-to-close returns.
- `price_to_sma_w = close_t / mean(close_(t-w+1)..close_t) - 1`.
- `relative_strength_w` is an asset's rolling return minus the mean rolling
  return of currently tradable assets.
- market mean return and volatility are means across currently tradable assets;
  dispersion is the population standard deviation of their rolling returns;
  breadth is the fraction with strictly positive rolling return.

Cash is not an asset dimension for these metrics. Untradable assets publish
zero for asset-scoped rolling values and are excluded from cross-sectional
summaries. When history is too short, `zero_fill` publishes exactly `0.0`.
Zero therefore means unavailable lookback after normalisation, not a future
estimate. No metric reads adjusted closes or bars after `t`.

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

The schema hash includes its version, entries, providers, and normalisation rules. It is embedded in `BrainArchitecture` and checkpoint compatibility. Baseline checkpoint format 13 safely reads format 12 and legacy format-13 agent layouts, while awareness schemas use format 14 because they persist additional agent state. Format 11 remains incompatible because it predates the registry identity; no positional-schema migration is attempted. Prompt 6 did not change the format-12 observation layout. An enabled rolling or awareness provider has a different schema/configuration identity and is rejected on restore against a default-schema checkpoint.

Future indicators, external data, or evolutionary sensor selection must add providers or explicit schema entries through this boundary. They must not append positional fields in `SimulationEngine`, modify PPO input handling, or make a provider mutate simulation state. Future genome-controlled sensor subsets can be represented as a selected schema without changing the policy or learning interfaces.

V2 awareness schemas use checkpoint format 14 because their persistent agent
statistics extend the serialized state. Baseline schemas retain format 13;
format-12 and legacy format-13 checkpoints are migrated with zeroed awareness
statistics when loaded under a compatible baseline configuration.
