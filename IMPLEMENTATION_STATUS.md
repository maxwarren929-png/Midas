# Implementation Status

## Prompt 2.5 experimental correctness repair

The pre-experiment correctness pass is implemented. Checkpoint format is now 8 because rollout entries carry explicit masked-timestep and actor/critic eligibility state, while action validation records separate policy and universal-market caps.

- Recoverable policy failures execute the fallback once, retain its realized reward and penalty, and append one chronological rollout entry. Successful evaluations may retain a critic sample, but fallback actions are never given a fabricated actor probability; inference and observation failures remain masked with no invented model outputs. Masked GAE uses discounted environment returns across missing-value boundaries so failure penalties affect neighboring eligible samples. Metrics count masked, actor-ineligible, critic-ineligible, and penalty totals.
- Tournament and rank-weighted parent selectors now apply the minimum mating distance while selecting both parents. If no distant second parent exists, the same selector chooses the best available constrained fallback and a deterministic `DiversityConstraintRelaxed` event and metric record the relaxation; uniform random pair replacement is gone.
- Action validation distinguishes an optional policy cap from an optional universal market cap. Neural and random policies use their configured cap; cash, equal-weight, and buy-and-hold benchmarks do not inherit it. Buy-and-hold makes one fully invested initial purchase and then holds, while equal-weight can remain fully diversified subject only to execution economics.
- Deterministic/diagnostic/bootstrapped policy evaluation no longer decays exploration temperature. Reset mutation independently covers every weight and bias and reports both reset counts.

Regression coverage is now 25 core unit tests, 48 foundation tests, and 5 Prompt 2 integration tests (78 tests, no failures). Full recurrent BPTT, alternate distributions, GPU, and Lamarckian inheritance remain outside scope.

Files changed in this pass: `src/lib.rs`, `src/action.rs`, `src/agent.rs`, `src/brain.rs`, `src/evolution.rs`, `src/learning.rs`, `src/metrics.rs`, `src/replacement.rs`, `src/simulation.rs`, the foundation and Prompt 2 integration tests, `README.md`, `docs/EXTENDING.md`, and this status document. Verification passed with `cargo fmt --all --check`, workspace check, Clippy with `-D warnings`, the full workspace test suite, all focused recovery/selection/benchmark/temperature/mutation/checkpoint/atomic filters, and the release CLI validation, simulation, benchmark, and evolution-report commands. The synthetic Prompt 2 run completed 7 steps with 6 PPO updates and no deaths; this is a correctness smoke test, not an experiment claim.

## Prompt 2.25 verification and hardening

State-dependent frozen-encoder PPO head learning is implemented and tested. Burn now uses one shared encoded-state path for inference and training. The hidden and recurrent layers remain frozen, while actor and critic dense output weights and biases are lifetime parameters updated by PPO. Gradient accumulation uses the outer product of the encoded representation and the categorical logit/value gradients; entropy, clipping, joint gradient clipping, finite checks, and shape-complete momentum state are included. The birth genome remains immutable.

- `OptimizerState` now contains separate actor/critic `DenseOptimizerState` values with weight and bias momentum vectors sized from the architecture. Restore validates all lengths and finiteness.
- The action distribution is explicitly `SingleAssetCappedCategorical`: cash or one tradable asset at the configured cap plus cash remainder, not a general multi-asset simplex.
- Continuing-value bootstrap fork failures return structured errors and abort the transactional step; zero is used only for non-PPO or terminal paths. Failed lifecycle errors are idempotent without diagnostic double-wrapping.
- Checkpoint format was 7 for the full-head repair and is now 8 after the masked-rollout and benchmark-cap schema change. Formats 6 and 7 are rejected with explicit incompatibility messages.
- Regression coverage includes exact dense outer-product gradients, clipped-gradient suppression, actor/critic head updates with frozen encoders, two-state action/value learning, malformed optimizer restore, and transactional bootstrap-fork failure. The Prompt 2 integration test verifies full-head updates and immutable genomes.

Current test totals are 21 core unit tests, 45 foundation regressions, and 4 Prompt 2 integration tests (70 tests, no failures). Only CPU Burn `NdArray` is implemented. Full recurrent BPTT, hidden-layer training, alternate allocation distributions, GPU, distributed learning, topology evolution, and Lamarckian inheritance remain outside this prompt.

## Prompt 2: lifetime learning and evolution

The Prompt 2 implementation is present and verified at the current foundation boundary.

- `learning.rs` defines typed transitions, bounded rollouts, terminal/truncated GAE with continuing-state bootstrap estimates, advantage normalization, clipped PPO losses, deterministic recurrent sequence shuffling, update metrics, and checkpointable `NoLearning`/`PpoLearning` strategies.
- `brain.rs` exposes framework-neutral policy evaluation and trainable-model hooks. The single-asset capped categorical action distribution is over cash and tradable assets; an asset draw maps to the configured maximum allocation and cash remainder, so sampled and deterministic actions are legal and stochastic samples have an exact categorical log probability. Burn caches runtime tensors and applies finite clipped actor/value updates with shape-complete momentum optimizer state. Actor and critic output weights and biases update; hidden/recurrent weights remain fixed by design.
- `reward.rs` separates configurable log-return and risk-adjusted environment rewards from reproductive fitness.
- `evolution.rs` provides stable `GenomeId`/`LineageId`, flat ancestry records, deterministic tournament/rank selection, uniform/layer blend crossover, bounded Gaussian mutation, fitness, genetic distance, fitness sharing, and checkpointable evolutionary metrics/RNG.
- `EvolutionaryReplacementStrategy` selects two living parents when eligible, crosses and mutates birth genomes, emits deterministic mutation reports (including zero-change reports), applies explicit random/clone/abort fallbacks, and spawns a fresh policy, optimizer, recurrent state, rollout, and failure counters through `AgentFactory`. Benchmarks remain non-learning and non-evolutionary unless explicitly configured.
- `SimulationEngine` transactions fork learning and evolution state with agents; PPO updates, life records, replacement, IDs, events, and metrics commit together. Checkpoint format 8 includes learning snapshots, full-head optimizer state, masked unfinished rollouts, evolution registries, life records, IDs, RNG, mutation diagnostics, and metrics.
- CLI includes the offline `evolution-report` command. `configs/prompt2.toml` is a short deterministic PPO/evolution example.
- Prompt 2 coverage includes the Prompt 2.25 tests listed above plus deterministic PPO checkpoint resumption, committed automatic checkpoints, and two-parent replacement integration tests.

Known Prompt 2 limitations are explicit: only CPU Burn NdArray is implemented; PPO is frozen-encoder actor/critic head training rather than hidden/recurrent or full-network optimization; recurrent optimization uses ordered sequence minibatches but does not provide full framework autodiff through recurrent layers; diversity protection is a bounded sharing/distance mechanism rather than species clustering; no GPU, distributed training, topology evolution, Lamarckian inheritance, or profitability claim is provided.

## Prompt 1.75 verification and hardening

The Prompt 1.75 audit and regression pass is implemented in the core market, execution, observation, brain, agent, replacement, simulation, configuration, metrics, and integration-test modules. Its transactional guarantees remain the base for Prompt 2.

### Final policy-failure correctness repair

`SimulationEngine::step` now forks mutable agent and policy state into a per-step transaction. Events, metrics, counters, portfolios, corporate actions, replacement strategy state, and the step boundary are committed only after the transaction succeeds. `StopExperiment` enters `Failed { reason }` and records only diagnostic failure metadata; `KillAgent` commits immediate policy-failure death at the observation close without fallback or next-day effects. Recoverable failures have the single `PolicyFailure` then `ActionFallback` sequence and one masked learning timestep. Checkpoint format 8 extends the failed-lifecycle record with masked learning, full-head optimizer, and evolution state without serializing a half-applied step. The foundation regressions remain covered alongside Prompt 2 tests.

- Corporate actions have one explicit convention: raw opens/closes remain execution and valuation prices; `FeaturePriceSeries` applies only split ratios dated on or before the requested feature date; dividends stay out of the feature price and credit cash exactly once at the beginning of the next-open execution date. Same-day actions are ordered split then dividend. Split, reverse-split, multiple-split, dividend, discontinuity-rejection, and future-leakage regressions cover the contract.
- Backend/device selection is authoritative through `BackendConfig::compute_device` and `AgentBuildContext`. Only the implemented CPU Burn `NdArray` `default` device is accepted; unsupported device names fail configuration validation. Checkpoint format 8 records the backend identity and rejects incompatible restores.
- Burn remains private to the neural adapter. `NeuralModel`/`NeuralModelFactory` support the mock external model and Burn model through the same agent/simulation path. Burn runtime tensors are built once per model build/restore and reused; immutable `BrainGenome`, mutable lifetime parameters, and recurrent runtime state are separate.
- Policy and observation failures preserve their original diagnostics in `DomainEvent::PolicyFailure`, increment per-agent total/consecutive counters, and follow configurable continue, kill-agent, or stop-experiment behavior before cash fallback. The shared observation builder validates dimensions and normalization instead of panicking. The step transaction buffers all domain events and mutable agent/replacement state until commit; stop failures enter a persistent failed lifecycle without advancing the boundary, while fatal agent failures die before next-day actions and never emit a fallback.
- Long-only action validation covers finite/non-negative proposals, availability masks, concentration limits, cash redistribution, tiny caps, all-unavailable markets, and numerical edge cases. Execution planning includes costs before proportional scaling, uses stable asset IDs, converges minimum-trade filtering deterministically, and the applier recomputes trade economics before atomic mutation.
- Checkpoint tests compare uninterrupted/resumed domain histories, portfolios, agent states, policy blueprints, serialized completed state, lifecycle idempotence, observation schema, corporate-action data, backend compatibility, and deterministic failure replay. Operational checkpoint events remain intentionally outside deterministic domain history; a failed checkpoint may contain failure metadata but always retains the last committed simulation state.
- Replacement contexts expose read-only living-agent summaries (net worth, age, drawdown, birth date, parent IDs, policy blueprint, and borrowed optional genome references) plus architecture/schema/device/seed material. Prompt 2 adds deterministic two-parent selection and reproduction through the same factory boundary.
- The primary foundation integration suite now contains 48 executable regressions, including property, permutation, policy-failure, atomicity, replay, and exactly-once corporate-action tests. No network or external data dependency is required.

Files changed in this hardening pass include `crates/midas-core/src/{agent,brain,replacement,simulation}.rs`, `crates/midas-core/tests/foundation.rs`, `README.md`, `docs/EXTENDING.md`, and this status document.

Known remaining foundation limitations are deliberate: CPU `NdArray` is the only neural backend; memory is a simple recurrent layer, not a GRU; corporate actions are explicit split/dividend records and are not inferred from providers; and event persistence is in-memory. Prompt 2 adds bounded PPO head updates and genome-only evolution; full-network recurrent autodiff and advanced evolutionary research remain future work.

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
- Checkpoint format v8 with lifecycle (including persistent failure reason), complete observation schema and architecture, policies/genomes/lifetime memory, full actor/critic head PPO optimizer and masked rollout state, evolution registries, portfolios, metrics, deterministic domain events, counters, backend identity, mutation diagnostics, and accessed-through date. Operational events are deliberately excluded from deterministic equivalence. Formats 6 and 7 are rejected because their optimizer/rollout layouts do not cover the current state.
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

## Reserved for future research prompts

- Full-network/recurrent autodiff training, alternative valid allocation distributions, GPU backends, distributed optimization, and advanced recurrent PPO variants.
- Species clustering, topology evolution, richer novelty metrics, and optional Lamarckian inheritance modes.
- Real-money execution, causal performance claims, and profitability conclusions.

Prompt 2 components are explicitly labelled PPO and evolutionary reproduction only where implementation and tests cover them.
