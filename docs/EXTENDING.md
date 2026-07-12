# Extending Midas

## Data sources

Implement `MarketDataSource` and return `MarketDataset::new`. Provider parsing stays outside the engine; canonical validation, typed asset IDs, ordering, and identity hashing remain centralized. Add malformed-input and identity tests.

## Feature pipelines

Implement `FeaturePipeline`, publish a new schema version whenever ordering or meaning changes, and derive vector size from the schema. Each row must be reproducible from data no later than its observation date. Add an extreme-future-value leakage test and document adjusted-price use.

## Policies and brains

Implement `Policy::act` and `Policy::snapshot`. Outputs contain cash plus one allocation per runtime asset. The execution layer remains authoritative. Stable brain changes require new genome and architecture versions plus explicit checkpoint compatibility handling.

## Execution models

Implement `ExecutionModel::rebalance` without querying future market state. Preserve sells-before-buys, non-negative cash and quantities, minimum-trade handling, unavailable-asset rejection, and separate cost fields. Add accounting and monotonic-cost property tests.

## Events and metrics

Add immutable variants to `SimulationEvent`. Metrics implement `EventSink` and must derive results only from events. A new sink can stream JSON, persist records, or aggregate diagnostics without modifying the daily loop.

## Replacement

Implement `ReplacementStrategy`. Prompt 1's strategy is deliberately named `NonEvolutionaryRandomGenomeReplacement`: it has no selection, parents, crossover, mutation, ancestry, or reproductive fitness. Prompt 2 should add those concepts behind this boundary and add stable RNG state to checkpoints for any stateful strategy.

## Learning

Implement `LearningStrategy` only when rollout and update semantics are complete. Prompt 1 uses `NoLearning`; it is never called PPO. PPO will require typed transitions, rollout ownership, GAE, clipped actor loss, critic loss, entropy, AdamW state, gradient validation/rollback, terminal handling, and checkpointed optimiser/RNG/buffer state.

## Configuration and CLI

Add typed enums and `deny_unknown_fields` configuration structures, validate before I/O, and route CLI construction through a factory. Do not branch inside the simulation timeline for provider- or model-specific behavior.
