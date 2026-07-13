# Extending Midas

## Market data and corporate actions

Implement `MarketDataSource` and construct `MarketDataset` through validated constructors. Provider-adjusted history is not accepted as point-in-time data. Raw opens/closes are used for execution and valuation. `FeaturePriceSeries` may apply only split ratios known on or before the requested date; cash dividends remain explicit cash events. New corporate actions belong in the canonical `CorporateAction` enum and need an explicit timeline rule, portfolio accounting test, and future-safety test. The current timeline applies events at the start of the next-open execution date, before planning; same-day splits are ordered before dividends.

## Features and observations

`FeaturePipeline` produces market-only rows. `FeaturePriceSeries` is the point-in-time price contract for feature calculations. `ObservationSchema` expands configured market, portfolio, and agent-state specifications, owns ordering, dimensions, normalization rules, and the stable hash. Add new values in `SchemaObservationBuilder`; never append positional fields in the engine. Any semantic or ordering change requires a schema-version increment.

## Actions and policies

Policies implement `propose_action`, snapshot/reset state, and return a `PolicyBlueprint`. Proposals are untrusted. `ActionValidator` is the only authority for length, finiteness, long-only rules, masking, normalisation, fallback, and concentration. Execution accepts only `ValidatedAllocation`.

## Neural implementations

Implement `NeuralModel` and `NeuralModelFactory`. A model owns recurrent state and maps `NeuralInput` to logits/value. Burn is one adapter and currently supports only CPU `NdArray` `default`. Runtime tensors are cached at model construction/restoration; `runtime_parameter_builds` is available for diagnostics. Stable snapshots use `NeuralModelState`; adding another serialisable implementation requires a distinct state kind and checkpoint tests. The simulation engine must remain free of ML-framework types. The immutable `BrainGenome`, mutable lifetime parameters, recurrent runtime state, and optimizer state are separate concepts. `PolicyEvaluation` carries the sampled categorical action, log probability, entropy, and value estimate. Burn's trainable adapter updates cached actor/critic heads through `NeuralModel::train_batch`; another framework can use the same boundary.

### PPO and rewards

`LearningStrategy` owns bounded `RolloutBuffer` state. `PpoLearning` computes GAE with configured `gamma` and `gae_lambda`, uses forked next-observation critic estimates for continuing neural episodes, normalizes advantages when enabled, preserves recurrent sequence order, and requests a clipped objective update at rollout length or episode end. A successful update is committed only after the model returns finite metrics; failed updates remain inside the step transaction. `RewardFunction` separates environment reward from reproductive fitness; `LogReturn` and `RiskAdjustedReturn` are available.

For a neural agent, the inherited learning-rate and entropy traits initialize that agent's PPO update configuration, while action-temperature decay is mutable runtime state and is checkpointed with the policy snapshot. The immutable birth genome is not rewritten by these lifetime updates.

The action distribution is categorical over cash and each currently tradable asset. The cash category means a fully cash allocation. An asset category means exactly the configured maximum allocation in that asset and the remainder in cash, so concentration constraints hold by construction. Deterministic evaluation takes the masked argmax and maps it to that same legal allocation; stochastic PPO evaluation samples the categorical distribution and stores its exact log probability.

## Agent construction and replacement

Both initial population and death replacement use `AgentFactory::spawn`. A replacement strategy receives `ReplacementContext` with dead state and blueprint, living summaries, configuration, architecture, observation schema, and deterministic seed material. It returns `AgentSpawnRequest`; it must not construct a policy. Strategies with mutable selection/RNG state should implement `ReplacementStrategy::fork`; the simulation swaps the fork only when the containing step commits. `EvolutionaryReplacementStrategy` uses tournament or rank-weighted selection, configurable eligibility, deterministic crossover and bounded mutation, and explicit random/clone/abort fallbacks. Benchmark blueprints remain non-learning and non-evolutionary unless configuration explicitly chooses otherwise.

### Evolution and ancestry

`GenomeId`, `LineageId`, and `AgentId` are distinct. `EvolutionState` stores flat `GenomeRecord` ancestry, completed-life fitness summaries, deterministic evolutionary RNG state, and population metrics. Default `GenomeOnly` inheritance crosses birth genomes only; lifetime-trained weights, optimizer state, recurrent memory, and rollouts are reset at birth. Genetic distance is symmetric and supports fitness sharing and minimum mating distance.

The default reproductive fitness normalizes and bounds log return, survival duration, drawdown, and turnover components before applying configured weights. It is separate from the environment reward used by PPO.

Evolutionary births emit parent-selection, crossover, mutation-report, lineage, fitness, and fallback domain events without embedding parameter arrays in the history. A mutation report can legitimately contain zero changes when configured mutation probabilities or sigmas produce no perturbation.

## Execution

`ExecutionPlanner` computes sells, eligible buys, full fixed/variable costs, a common affordability scale, and convergent minimum-trade filtering without mutating a portfolio. `ExecutionApplier` independently recomputes and validates every trade, checks stable `AssetId` lookups, and applies the immutable plan atomically. Add permutation tests for any new cost model.

## Events and checkpoints

`DomainEvent` is deterministic and checkpointed. `OperationalEvent` records effects such as successful checkpoint writes and is not part of simulation equivalence. `SimulationLifecycle` makes completion idempotent. Checkpoint format 6 records backend identity, separated neural/runtime/optimizer state, unfinished rollouts, deterministic learning state, evolution registries, life records, IDs, mutation diagnostics, and a persistent failed lifecycle; format or schema changes require a version bump and explicit rejection of incompatible files. Policy failures emit diagnostics, count per agent, and follow `policy_failures.behavior` after configured consecutive/total limits.

Every simulation step is a transaction. The engine forks mutable policy/neural state, buffers domain events, and commits agents, replacement state, counters, metrics, and `next_step` only after the complete close-to-next-open path succeeds. A `StopExperiment` limit discards that transaction and enters `SimulationLifecycle::Failed { reason }`; it does not apply corporate actions, trades, costs, valuation, aging, or automatic checkpoint writes. Repeated calls after failure return the stored reason without mutation. The failure event is diagnostic metadata; the checkpoint boundary remains the previous committed step. Completed runs are likewise idempotent.

Failure event sequences are deliberately strict. A recoverable incident emits one `PolicyFailure` followed by one `ActionFallback`, then executes the cash allocation normally. A fatal `KillAgent` incident emits `PolicyFailure` followed by `AgentDied` with `DeathCause::PolicyFailure`; no fallback or next-day event is executed for that agent. Consecutive failures increment on every unusable decision and reset only after a valid, non-fallback action has passed validation. Total failures never reset during a lifetime; a replacement starts both counters at zero. Corporate actions retain the order observe close `t`, decide/fail, advance survivors, split then dividend at `t+1`, execute opening trades, and value the close.

## Determinism and limitations

Learning, selection, crossover, mutation, and replacement derive separate deterministic ChaCha8 streams from checkpointed seed material, so adding one evolutionary operation does not silently reorder the others. PPO currently updates the Burn actor and critic heads while hidden/recurrent weights remain fixed; this is a deliberate bounded implementation choice, not a claim of full-network research training. Only the CPU Burn NdArray backend is implemented. No GPU, distributed training, species clustering, topology evolution, or Lamarckian inheritance mode is advertised.
