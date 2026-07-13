# Extending Midas

## Market data and corporate actions

Implement `MarketDataSource` and construct `MarketDataset` through validated constructors. Provider-adjusted history is not accepted as point-in-time data. Raw opens/closes are used for execution and valuation. `FeaturePriceSeries` may apply only split ratios known on or before the requested date; cash dividends remain explicit cash events. New corporate actions belong in the canonical `CorporateAction` enum and need an explicit timeline rule, portfolio accounting test, and future-safety test. The current timeline applies events at the start of the next-open execution date, before planning; same-day splits are ordered before dividends.

## Features and observations

`FeaturePipeline` produces market-only rows. `FeaturePriceSeries` is the point-in-time price contract for feature calculations. `ObservationSchema` expands configured market, portfolio, and agent-state specifications, owns ordering, dimensions, normalization rules, and the stable hash. Add new values in `SchemaObservationBuilder`; never append positional fields in the engine. Any semantic or ordering change requires a schema-version increment.

## Actions and policies

Policies implement `propose_action`, snapshot/reset state, and return a `PolicyBlueprint`. Proposals are untrusted. `ActionValidator` is the only authority for length, finiteness, long-only rules, masking, normalisation, fallback, and concentration. Execution accepts only `ValidatedAllocation`.

## Neural implementations

Implement `NeuralModel` and `NeuralModelFactory`. A model owns recurrent state and maps `NeuralInput` to logits/value. Burn is one adapter and currently supports only CPU `NdArray` `default`. Runtime tensors are cached at model construction/restoration; `runtime_parameter_builds` is available for diagnostics. Stable snapshots use `NeuralModelState`; adding another serialisable implementation requires a distinct state kind and checkpoint tests. The simulation engine must remain free of ML-framework types. The immutable `BrainGenome`, mutable lifetime parameters, and recurrent runtime state are separate concepts; optimizer state is reserved for Prompt 2.

## Agent construction and replacement

Both initial population and death replacement use `AgentFactory::spawn`. A replacement strategy receives `ReplacementContext` with dead state and blueprint, living summaries, configuration, architecture, observation schema, and deterministic seed material. It returns `AgentSpawnRequest`; it must not construct a policy. Prompt 1 preserves benchmark blueprints and randomises only neural genomes. Parent selection and reproduction remain deferred.

## Execution

`ExecutionPlanner` computes sells, eligible buys, full fixed/variable costs, a common affordability scale, and convergent minimum-trade filtering without mutating a portfolio. `ExecutionApplier` independently recomputes and validates every trade, checks stable `AssetId` lookups, and applies the immutable plan atomically. Add permutation tests for any new cost model.

## Events and checkpoints

`DomainEvent` is deterministic and checkpointed. `OperationalEvent` records effects such as successful checkpoint writes and is not part of simulation equivalence. `SimulationLifecycle` makes completion idempotent. Checkpoint format 3 records backend identity and the separated neural runtime state; format or schema changes require a version bump and explicit rejection of incompatible files. Policy failures emit diagnostics, count per agent, and follow `policy_failures.behavior` after configured consecutive/total limits.

## Learning

Prompt 1 uses `NoLearning`; it is not PPO. Prompt 2 may add rollout ownership, GAE, clipped losses, optimiser state, gradient validation/rollback, terminal handling, and checkpointed learning RNG through the existing policy/model boundary.
