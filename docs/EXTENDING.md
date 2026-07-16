# Extending Midas

## Configuration compatibility

Configuration architecture changes must preserve the compact
`serde_json::to_vec(Config)` bytes used by `Config::hash`, including struct
field order, `BTreeMap` ordering, defaults, `Option` encoding, serde names, and
the default `rolling_market_stats` omission. The regression snapshots under
`crates/midas-core/tests/fixtures/config/` and the shipped-config hash manifest
under `crates/midas-cli/tests/fixtures/` are fixed compatibility inputs, not
auto-updating golden files. Update them only for an intentional configuration
format or behavior change.

Run `cargo test -p midas-core --test config_regression` for serialization,
validation, rolling, and version-12 checkpoint/report compatibility. The CLI
integration test validates every TOML under `configs/`, requires exact stdout
and empty stderr, and fails when a config is added without a recorded hash. To
exercise one release build rather than Cargo's test binary, build once and run:

```bash
cargo build --release -p midas-cli
MIDAS_CONFIG_REGRESSION_BIN=target/release/midas-cli \
  cargo test -p midas-cli --test config_regression
```

These tests exist so configuration modules can be moved mechanically without
silently changing hashes, defaults, validation precedence, checkpoint
acceptance, CLI output, or deterministic reports.

## Market data and corporate actions

Implement `MarketDataSource` and construct `MarketDataset` through validated constructors. Provider-adjusted history is not accepted as point-in-time data. Raw opens/closes are used for execution and valuation. `FeaturePriceSeries` may apply only split ratios known on or before the requested date; cash dividends remain explicit cash events. New corporate actions belong in the canonical `CorporateAction` enum and need an explicit timeline rule, portfolio accounting test, and future-safety test. The current timeline applies events at the start of the next-open execution date, before planning; same-day splits are ordered before dividends.

## Features and observations

`FeaturePipeline` produces market-only rows. `FeaturePriceSeries` is the point-in-time price contract for feature calculations. `ObservationProvider` implementations are read-only sensors: they publish raw named values to `ObservationSink` and must not depend on policies, PPO, execution, rewards, evolution, checkpoints, or mutable simulation state. `ObservationRegistry` rejects duplicate provider or observation names and compiles providers in deterministic name order. `ObservationSchema` expands the configured legacy feature groups into explicit named entries, owns index order, dimensions, normalisation rules, and the stable hash; provider registration order is never an input-order contract.

The public facade is `crates/midas-core/src/observation.rs`. Add schema/source
definitions in `observation/schema.rs`, normalisation behavior in
`observation/normalization.rs`, shared provider contracts in
`observation/contracts.rs`, generic registration behavior in
`observation/registry.rs`, and concrete built-in sensors under
`observation/providers/`. `SchemaObservationBuilder` in
`observation/builder.rs` remains the only component that compiles named raw
values into the neural vector. Keep provider factories under `providers/` so
the generic registry never imports concrete sensors.

To add a future sensor, implement `ObservationProvider`, return complete `ObservationMetadata` (name, provider, description, units, shape, normalisation, version, tags), and register it when constructing a registry. Add explicit schema entries by name and validate them with `SchemaObservationBuilder::with_registry`; no provider may publish an unreferenced value and no schema entry may be missing. Providers should emit raw finite values whenever practical. The schema compiler supports `none`, clipped ratio, min/max, z-score, log, signed-log, percentage, and finite custom affine normalisation. Per-name overrides live under `[observations.normalizations]`; changing dimensions, names, order, provider membership, or normalisation semantics requires an intentional `[observation].version` increment. The portfolio state (cash, holdings, allocations, unrealised returns, and cost basis where configured) remains part of the observation because neural actions are incremental. See [Observation Architecture](OBSERVATIONS.md) for the full contract and CLI introspection commands.

`RollingMarketStatsProvider` is the reference market-sensor implementation. It receives immutable `MarketDataset` data only at builder construction, precomputes a date-keyed raw-value table from `PointInTimeFeaturePriceSeries`, and then publishes one scalar for each explicit schema entry. It never reads future bars, adjusted closes, or mutable agent state. Use `[observations.rolling_market_stats]` to select positive rolling windows; empty metric-window lists disable a group, and `minimum_history = "zero_fill"` emits `0.0` until the lookback exists. Its formulas and stable names are documented in [Observation Architecture](OBSERVATIONS.md). New rolling-like sensors should follow the same pattern: raw source values, metadata-rich names, schema-owned normalisation, deterministic precomputation or a bounded past-only cache, and an identity-changing schema entry. Do not add direct neural input fields to the engine.

## Actions and policies

Policies implement `propose_action`, snapshot/reset state, and return a `PolicyBlueprint`. Proposals are untrusted. `ActionValidator` is the only authority for length, finiteness, long-only rules, masking, normalisation, fallback, concentration, and the configured neural trade-fraction ceiling. Its context distinguishes an optional policy/genome allocation cap, an optional trade-fraction cap, and an optional universal market cap. Neural policies emit `ProposedTradeImpulses` in `[-1,1]`; validation masks unavailable assets and preserves valid impulses. Benchmarks continue to emit target allocations and do not inherit neural caps. `ExecutionPlanner::plan_action` dispatches validated allocations or incremental impulses. Impulses are sized as `impulse * maximum_trade_fraction * current close net worth`; sells are planned before buys, and only broker affordability/holding constraints can scale or reject a request. Buy-and-hold invests once at the initial decision and then passively holds; equal-weight rebalances its available assets without a neural cap.

## Neural implementations

Implement `NeuralModel` and `NeuralModelFactory`. A model owns recurrent state and maps `NeuralInput` to raw actor-head parameters/value. Burn is one adapter and currently supports only CPU `NdArray` `default`. Runtime tensors are cached at model construction/restoration; `runtime_parameter_builds` is available for diagnostics. Stable snapshots use `NeuralModelState`; adding another serialisable implementation requires a distinct state kind and checkpoint tests. The simulation engine must remain free of ML-framework types. The immutable `BrainGenome`, mutable lifetime parameters, recurrent runtime state, and optimizer state are separate concepts. `PolicyEvaluation` carries the sampled impulse action, exact latent log probability, entropy, and value estimate. Burn's trainable adapter uses one encoded-state path and updates complete actor/critic dense output heads through `NeuralModel::train_batch`; hidden and recurrent layers remain frozen. Another framework can use the same boundary.

The public brain facade remains `crates/midas-core/src/brain.rs`; external
imports must continue to use `midas_core::brain::*`. Add architecture contracts
in `brain/architecture.rs`, inherited parameters and validation in
`brain/genome.rs`, checkpointed runtime/policy state in `brain/state.rs`, and
framework-neutral model traits in `brain/model.rs`. Keep Burn tensors and
cached runtime conversion inside `brain/burn_runtime.rs`; frozen-encoder PPO
head-update math belongs in `brain/head_training.rs`. New bounded action
distributions belong in `brain/impulse_beta.rs` or a sibling module, while
built-in benchmark policies belong in `brain/benchmarks.rs`. Do not move
serialized fields, reorder enum variants, or bypass the facade when extending
these modules: those layouts are checkpoint-sensitive.

### PPO and rewards

`LearningStrategy` owns bounded `RolloutBuffer` state. `PpoLearning` computes GAE with configured `gamma` and `gae_lambda`, uses forked next-observation critic estimates for continuing neural episodes, normalizes advantages when enabled, preserves recurrent sequence order, and requests a clipped objective update at rollout length or episode end. A successful update is committed only after the model returns finite metrics; failed updates remain inside the step transaction. `RewardFunction` separates environment reward from reproductive fitness; `LogReturn` and `RiskAdjustedReturn` are available.

For a neural agent, the inherited learning-rate and entropy traits initialize that agent's PPO update configuration, while action-temperature decay is mutable runtime state and is checkpointed with the policy snapshot. The immutable birth genome is not rewritten by these lifetime updates.

The action distribution is `BoundedImpulseBeta`: the actor emits two raw values per asset. For each active asset, `(softplus(raw) + minimum_shape) / temperature` is clamped to the configured shape range and parameterises an independent Beta latent. A stochastic latent `u` maps to the impulse `2u - 1`; the configured trading fraction, not a simplex cap, determines the notional at execution. Unavailable assets are excluded from the latent vector and receive exact zero impulses. The latent vector is the PPO action, so log probability is the sum of exact Beta log densities and entropy is the sum of Beta entropies. Deterministic evaluation uses the Beta means, does not consume RNG, and does not mutate temperature. There is no post-hoc portfolio clipping or fabricated probability. Stable asset order is retained only as a reproducible latent/RNG ordering choice. PPO differentiates Beta log-probability and entropy through the softplus/temperature transform and output-head weights using the frozen recurrent representation. Full recurrent backpropagation through time remains future work.

## Agent construction and replacement

Both initial population and death replacement use `AgentFactory::spawn`. A replacement strategy receives `ReplacementContext` with dead state and blueprint, living summaries, configuration, architecture, observation schema, and deterministic seed material. It returns `AgentSpawnRequest`; it must not construct a policy. Strategies with mutable selection/RNG state should implement `ReplacementStrategy::fork`; the simulation swaps the fork only when the containing step commits. `EvolutionaryReplacementStrategy` uses tournament or rank-weighted selection, configurable eligibility, deterministic crossover and bounded mutation, and explicit random/clone/abort fallbacks. Benchmark blueprints remain non-learning and non-evolutionary unless configuration explicitly chooses otherwise.

### Evolution and ancestry

`GenomeId`, `LineageId`, and `AgentId` are distinct. `EvolutionState` stores flat `GenomeRecord` ancestry, completed-life fitness summaries, deterministic evolutionary RNG state, and population metrics. Default `GenomeOnly` inheritance crosses birth genomes only; lifetime-trained weights, optimizer state, recurrent memory, and rollouts are reset at birth. Genetic distance is symmetric and supports fitness sharing and minimum mating distance. The mating distance filters the second-parent pool while retaining tournament/rank-weighted selection pressure. If no qualifying pair exists, the same configured selector chooses a constrained fallback and emits `DiversityConstraintRelaxed`; it is never replaced by an unlabelled uniform pair draw. Reset mutation independently samples every weight and bias and resets selected values to zero, reporting weight and bias counts separately.

The default reproductive fitness normalizes and bounds log return, survival duration, drawdown, and turnover components before applying configured weights. It is separate from the environment reward used by PPO.

Evolutionary births emit parent-selection, crossover, mutation-report, lineage, fitness, and fallback domain events without embedding parameter arrays in the history. A mutation report can legitimately contain zero changes when configured mutation probabilities or sigmas produce no perturbation.

## Execution

`ExecutionPlanner` computes target-allocation rebalances or persistent impulse trades. For impulses it computes sell fills first, then applies a common affordability scale to eligible buys after fixed and variable costs and convergent minimum-trade filtering. `ExecutionApplier` independently recomputes and validates every trade, checks stable `AssetId` lookups, and applies the immutable plan atomically. Add permutation tests for any new cost model.

## Events and checkpoints

### Simulation module layout

The public simulation facade remains `crates/midas-core/src/simulation.rs`.
Prompt 7B.1 moved only low-risk public contracts behind that facade:
`simulation/types.rs` contains `CHECKPOINT_VERSION`, `DatasetSplit`,
`SimulationLifecycle`, and `DeathCause`; `simulation/events.rs` contains
`DomainEvent` and `OperationalEvent`; and `simulation/diagnostics.rs` contains
`DecisionTrace`, `PpoUpdateTrace`, and `ReportTrace`. Continue importing these
items through `midas_core::simulation::*`.

Prompt 7B.2 moved the private checkpoint envelope, agent snapshots,
serialization, deserialization, and compatibility checks into
`simulation/checkpoint.rs`. Public checkpoint, resume, and frozen-evaluation
methods remain on `SimulationEngine`; checkpoint candidate timing and engine
orchestration remain in the facade. The checkpoint module is an implementation
detail and must not become a downstream import path. Do not reorder serialized
fields or enum variants while extending these contracts: checkpoint format
remains 12 and existing version-12 bytes remain compatible.

Prompt 7B.3 moved the `SimulationEngine` state shell, constructors, founder
setup, inspection APIs, configuration hooks, simple event emission, and
read-only valuation-price lookup into private `simulation/engine.rs`. The
facade continues to export `midas_core::simulation::SimulationEngine`; its
public API and public metrics/event/trace fields are unchanged. Run and daily
step orchestration, transactions, checkpoint and evaluation methods, learning,
replacement lifecycle, and report construction remain in `simulation.rs`.
This shell split does not change checkpoint format 12 or any simulation order.

Prompt 7B.4 moved lifecycle and loop control into private
`simulation/runner.rs`, and moved the step transaction buffer, agent fork,
abort state, commit, and rollback-by-discard boundary into private
`simulation/transaction.rs`. Automatic checkpoint candidates are still written
before authoritative mutation; transaction state then commits in the original
order, `next_step` advances, and only then is the operational checkpoint event
appended. Daily decision and economic logic remains in `simulation.rs`.
`SimulationEngine::run` and `run_steps` remain available through the existing
facade, and checkpoint format remains 12.

Prompt 7B.5 moved observation-to-policy decision handling, validation,
fallback/failure behavior, daily agent dispatch, and decision traces into
private `simulation/decision.rs`. The unchanged corporate-action, execution,
valuation, survival-cost, drawdown, reward, and trade-summary pipeline now
lives in private `simulation/advance.rs`. Their original daily ordering, event
and trace timing, transaction semantics, deterministic replay, public
`SimulationEngine` API, and checkpoint format 12 are preserved. Learning,
replacement, checkpoint/evaluation, and reporting implementation remains in
`simulation.rs` for later staged splits.

Prompt 7B.6 moved bootstrap evaluation, transition and rollout recording,
eligibility/masking, episode completion, PPO update application, learning
events, and PPO traces into private `simulation/learning_bridge.rs`. Completed
life and fitness handling, replacement invocation, reproduction, genome and
lineage bookkeeping, and evolution events now live in private
`simulation/death_replacement.rs`. Generic learning, evolution, and
replacement algorithms remain in their original modules. PPO, death,
replacement, RNG, event, and trace ordering are unchanged, as are the public
API, deterministic replay, and checkpoint format 12. Checkpoint/evaluation and
reporting remain in `simulation.rs` for Prompt 7B.7.

Prompt 7B.7 completes the modular simulation architecture. Private
`simulation/evaluation.rs` owns frozen checkpoint evaluation and evaluation
reset/setup, while private `simulation/report.rs` owns report construction and
the unchanged serialized `SimulationReport` contract. Public checkpoint
save/resume methods now live beside checkpoint I/O and compatibility checks in
private `simulation/checkpoint.rs`. The top-level `simulation.rs` is a thin
compatibility facade, so existing `midas_core::simulation::*` imports remain
valid. Checkpoint format 12, deterministic replay, evaluation behavior, report
bytes, event/trace ordering, and all public APIs are unchanged.

`DomainEvent` is deterministic and checkpointed. `OperationalEvent` records effects such as successful checkpoint writes and is not part of simulation equivalence. `SimulationLifecycle` makes completion idempotent. Checkpoint format 12 records the modular observation-schema identity alongside backend identity, the bounded-impulse-Beta distribution identity, separated neural/runtime/optimizer state, full actor/critic head momentum, unfinished masked rollouts with latent impulse samples, deterministic learning state, evolution registries, life records, IDs, impulse diagnostics, mutation diagnostics, and a persistent failed lifecycle. Format 11 and formats 6 through 10 are rejected because their positional observation, metrics, optimizer/rollout, or action-distribution layouts cannot be migrated safely; format or schema changes require a version bump and explicit rejection of incompatible files. Policy failures emit diagnostics, count per agent, and follow `policy_failures.behavior` after configured consecutive/total limits.

Every simulation step is a transaction. The engine forks mutable policy/neural state, buffers domain events, and commits agents, replacement state, counters, metrics, and `next_step` only after the complete close-to-next-open path succeeds. A `StopExperiment` limit discards that transaction and enters `SimulationLifecycle::Failed { reason }`; it does not apply corporate actions, trades, costs, valuation, aging, or automatic checkpoint writes. Repeated calls after failure return the stored reason without mutation. The failure event is diagnostic metadata; the checkpoint boundary remains the previous committed step. Completed runs are likewise idempotent.

Failure event sequences are deliberately strict. A recoverable incident emits one `PolicyFailure` followed by one `ActionFallback`, then executes the cash allocation normally. A fatal `KillAgent` incident emits `PolicyFailure` followed by `AgentDied` with `DeathCause::PolicyFailure`; no fallback or next-day event is executed for that agent. Consecutive failures increment on every unusable decision and reset only after a valid, non-fallback action has passed validation. Total failures never reset during a lifetime; a replacement starts both counters at zero. Corporate actions retain the order observe close `t`, decide/fail, advance survivors, split then dividend at `t+1`, execute opening trades, and value the close.
Failure event sequences are deliberately strict. A recoverable incident emits one `PolicyFailure` followed by one `ActionFallback`, then executes the cash allocation normally and records one masked rollout timestep. Critic samples may use a valid value from a successful evaluation; actor samples are masked whenever the executed fallback does not correspond to the sampled policy action. Inference and observation failures carry chronology and penalties but no fabricated probability, value, or observation. GAE uses discounted environment returns across masked boundaries. A fatal `KillAgent` incident emits `PolicyFailure` followed by `AgentDied` with `DeathCause::PolicyFailure`; no fallback or next-day event is executed for that agent. Consecutive failures increment on every unusable decision and reset only after a valid, non-fallback action has passed validation. Total failures never reset during a lifetime; a replacement starts both counters at zero. Corporate actions retain the order observe close `t`, decide/fail, advance survivors, split then dividend at `t+1`, execute opening trades, and value the close.

## Determinism and limitations

Learning, selection, crossover, mutation, and replacement derive separate deterministic ChaCha8 streams from checkpointed seed material, so adding one evolutionary operation does not silently reorder the others. PPO is currently frozen-encoder head training: actor/critic weights and biases update, while hidden/recurrent weights remain fixed. This is state-dependent output-head learning, not full-network or full recurrent PPO. Continuing-value bootstrap fork failures are fatal transactional errors rather than zero-value fallbacks. Only the CPU Burn NdArray backend is implemented. No GPU, distributed training, species clustering, topology evolution, or Lamarckian inheritance mode is advertised.

## Long-horizon diagnostics

Prompt 3B keeps diagnostics separate from simulation semantics. `RewardBreakdown` must preserve the exact scalar returned by the configured reward while exposing raw return, drawdown, turnover, invalid-action, and bankruptcy components. The simulation transaction commits `DecisionTrace` and `PpoUpdateTrace` together with the economic state; a failed step commits neither. The `diagnostics` binary can write JSONL traces, and `experiments/prompt3b/analyze_traces.py` converts them to compact CSV for offline analysis. Do not put backend tensors or unbounded histories into domain events or checkpoints.

The frozen protocol uses a deterministic multi-regime synthetic market with separate training, validation, and test ranges. Conditions must share data, costs, survival pressure, architecture, population, and seeds; only the named learning/evolution or diagnostic ablation fields may differ. Preserve `reports/PROMPT3B_DIAGNOSTIC_EXPERIMENT.md` and its machine-readable tables when adding a new condition. Interpret turnover, concentration, ordering, PPO update intensity, and reward-scale results separately before selecting a correction. A turnover penalty is not a default and must be calibrated in the units of the observed reward.

The current suite found extreme PPO turnover/costs and measurable stable stick-breaking order bias. Episode rotation is not a safe fix because it remaps latent decisions and increases trading. A conservative rollout/update schedule is a hypothesis for the next isolated experiment, not a validated general improvement. Evolution is mechanically exercised by PPO-plus-evolution runs, but reproduction alone is not evidence of adaptation. All evaluation runs must be frozen and must not overwrite training checkpoints; `SimulationEngine::evaluate_checkpoint` disables automatic checkpoint cadence after validating the original checkpoint identity.
