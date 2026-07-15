# Implementation Status

## Prompt 6.5: conservative observation facade split

The former 2,777-line observation module is now a small
`midas_core::observation` compatibility facade over focused private modules for
normalisation, schema compilation/identity, provider contracts, registry,
builder, and built-in providers. Concrete provider construction moved out of
the generic registry into the private provider factory. Public observation
imports, serialized schemas, observation names/order, schema hashes, rolling
formulas, zero-fill behavior, and checkpoint format 12 are unchanged. Baseline
and post-refactor CLI validation matched all six default/Prompt 6 schema
dimensions and hashes. No simulation, PPO, reward, evolution, execution, or
market-generation behavior changed.

## Prompt 6: first modular market senses

Prompt 6 adds an opt-in `rolling_market_stats` observation provider without changing PPO, rewards, evolution, execution, the incremental-trade policy, or the Prompt 5 default schema. The provider precomputes immutable, point-in-time raw-close statistics by decision date and publishes named raw scalars through the existing registry/schema boundary.

- Supported per-asset metrics are rolling simple returns (`5`, `20`, `60`, and `120` bars by default), population volatility of simple daily returns, price-to-SMA distance, and cross-sectional relative strength. Market summaries provide mean return, mean volatility, dispersion, and breadth across currently tradable assets. The configured schema controls selected windows and all normalisation; no scaling is hidden in the provider.
- `zero_fill` is the implemented missing-history contract. Insufficient history and unavailable asset-specific values produce finite `0.0`; unavailable assets are excluded from cross-sectional summaries. Every calculation uses bars at or before close `t` and the split-safe `PointInTimeFeaturePriceSeries`; dividends and provider-adjusted prices are not folded into these inputs.
- Default configuration serialisation omits the disabled provider settings, retaining the Prompt 5 v12 default configuration/schema identity. Enabling senses changes the explicit schema hash and architecture input dimension; default and rolling checkpoints therefore reject each other cleanly. The serialized checkpoint layout remains format 12.
- `configs/prompt6/` contains frozen Prompt 3B/P4-derived default, momentum, volatility/trend, combined, and combined-plus-evolution schema conditions. `experiments/prompt6/generate_configs.sh` regenerates them mechanically from the frozen seed-41 bases.

Rolling-provider unit and integration coverage now verifies default-vector preservation, formulas, deterministic zero fill, unavailable-asset handling, normalisation ownership, schema dimension/hash changes, future-price leakage, metadata, configuration validation, and default-versus-rolling checkpoint rejection. Controlled ten-seed provider ablation results are recorded in `reports/PROMPT6_MARKET_SENSES_EXPERIMENT.md`: combined senses improved the frozen validation/test comparison against default PPO, while the two narrower sensor groups did not hold their validation gains on test. Verification passed with `cargo fmt --all --check`, workspace check, Clippy with `-D warnings`, and `cargo test --workspace --all-features` (**108 passed, 0 failed, 0 ignored**). These results are research instrumentation, not market-performance evidence.

## Prompt 5: modular observation architecture

Prompt 5 replaces the positional observation-builder implementation with a versioned registry of read-only sensor providers while preserving Prompt 4's default vector values and trading behaviour.

- `ObservationProvider` publishes named raw values through a registry-owned sink. The built-in `market`, `portfolio`, and `agent_state` providers are independent of PPO, neural models, execution, rewards, evolution, and checkpoints. Metadata records name, provider, description, units, shape, normalisation, version, and tags.
- `ObservationSchema` now compiles each neural input into an explicit index/name/provider/source/component/normalisation entry. Registry registration order cannot alter the input vector. The default schema remains the existing market, portfolio, and agent-state ordering; the `modular_default_builder_matches_prompt4_positional_vector` regression compares it exactly with the retired positional contract.
- `[observations]` selects the implemented default registry and provider set. Optional per-name `[observations.normalizations]` overrides support none, clipped ratio, min/max, z-score, log, signed log, percentage, and finite custom affine rules. Invalid names, duplicate metadata or indices, missing/unused registrations, dimensions, non-finite values, and invalid normalisation fail before simulation steps begin.
- `midas observations list`, `schema`, `describe`, and `validate` provide deterministic offline introspection. The registry reuses per-builder scratch buffers instead of creating a provider-to-neural-vector path on every call.
- Checkpoint format is now 12. Format 11 is rejected because it cannot prove compatibility with the registry-backed schema identity; no unsafe positional migration is provided. Schema hash remains part of the brain architecture and checkpoint validation.

Prompt 5 deliberately adds no technical indicators, reward changes, PPO changes, evolution changes, or policy/action changes. Future providers can be added through the registry/schema boundary; evolutionary sensor selection is not yet implemented. [Default Observation Equivalence](reports/PROMPT5_OBSERVATION_EQUIVALENCE.md) records an exact Prompt 4 versus Prompt 5 economic-report comparison for the preserved seed-41 control run. Verification passed with `cargo fmt --all --check`, workspace check, Clippy with `-D warnings`, `cargo test --workspace --all-features` (**102 passed, 0 failed, 0 ignored**), observation and checkpoint filters, release config validation, observation CLI introspection, simulation, benchmark, and an exact partial-checkpoint/resume report comparison.

## Prompt 4: persistent incremental trade policy

Prompt 4 replaces the neural target-portfolio action with persistent incremental trade impulses. The actor still emits two bounded-Beta shape values per asset, but the latent action is now mapped to one impulse in `[-1, 1]` per tradable asset (`2u - 1`). The configured `[brain.trading].maximum_trade_fraction` scales each impulse by current close net worth. Cash is implicit; existing holdings persist between decisions.

- `ProposedAction`/`ValidatedAction` carry either benchmark allocations or neural `TradeImpulses`. The central validator checks dimensions, finite bounds, and availability without applying allocation-cap semantics to neural trades.
- `ExecutionPlanner::plan_action` performs sells first, then fixed/variable-cost-aware common scaling of buys, minimum-trade filtering, and deterministic stable-ID ordering. `ExecutionApplier` remains atomic and rejects negative cash, holdings, leverage, and tampered economics. Impossible requests are broker constraints, not policy failures.
- PPO continues to use the exact independent Beta latent log probability and entropy; deterministic evaluation uses Beta means and does not consume RNG or decay temperature. Full actor/critic head training, frozen encoder, reward, and evolution are unchanged.
- Decision traces now include requested/executed impulses, notionals, scaling, rejected notional, cash and holdings before/after, and impulse magnitude/saturation metrics. Portfolio allocation metrics describe the persistent post-execution holdings.
- Checkpoint format was 11 for Prompt 4 and is now 12 after Prompt 5's schema-identity change. Format 10 and formats 6 through 11 are rejected because their action samples, rollout/policy snapshots, metrics, or observation layouts are incompatible; no unsafe migration is attempted. Birth genomes remain immutable and replacement lifetime state is fresh.

Prompt 4's controlled rerun is recorded in [Prompt 4 Persistent Incremental Trade Experiment](reports/PROMPT4_INCREMENTAL_TRADE_EXPERIMENT.md). It is a mechanical action-interface change, not evidence of profitability or intelligence. Known limitations are the independent per-asset Beta impulse factorisation, stable latent ordering, and frozen hidden/recurrent encoder.

## Prompt 3B: long-horizon diagnosis and adaptive reward experiment

Prompt 3B adds report-only observability and a frozen, long-horizon experimental protocol. The implementation commit tested for the suite is `c692101`. The generated market contains 1,800 business dates, four assets, 1,259 training steps, and frozen validation/test splits; its SHA-256 is recorded in `reports/PROMPT3B_DIAGNOSTIC_EXPERIMENT.md`.

- `RewardBreakdown`, transactional `DecisionTrace`, and `PpoUpdateTrace` expose return components, costs, turnover, allocations, asset-order positions, PPO KL/clipping, gradients, and head deltas without changing the reward scalar or checkpointed simulation state. The diagnostics binary writes compact JSONL/CSV traces.
- Forty primary CLI runs (control, PPO, evolution-only, and PPO+evolution; seeds 41-50) and eighteen three-seed ablations completed. Checkpoint/resume matched for representative runs. Recoverable-failure learning and benchmark exposure regressions remain covered by tests.
- PPO underperformance is primarily associated with extreme turnover and execution costs: the representative PPO trace had about 6,257 turnover units and $75,623 execution cost per agent-day versus about 40 and $488 for control. Stable order bias is measurable; naive episode rotation increased turnover and worsened results. Short-rollout PPO over-updating is supported as a contributing factor because a conservative schedule improved all three paired ablation seeds, but it is not yet a final selected policy.
- Real PPO+evolution CLI runs produced deaths, two-parent reproduction, crossover, mutation, ancestry, and generation depth. Evolution did not show consistent performance improvement. No permanent turnover or concentration penalty was added; the turnover ablation was unit-mismatched and clipped heavily.
- A correctness issue found during the protocol was fixed: frozen evaluation now disables automatic checkpoint writes after validating the training checkpoint identity, so evaluation cannot overwrite a training checkpoint. This is covered by `evaluation_does_not_overwrite_training_checkpoint`.

The complete results, raw-output index, exact test totals, limitations, and next-experiment recommendation are in [Prompt 3B Diagnostic Experiment](reports/PROMPT3B_DIAGNOSTIC_EXPERIMENT.md). This suite does not support claims of profitability or intelligence. The next experiment should validate the conservative PPO schedule at full seed count and calibrate any turnover ablation in compatible units before combining changes.

## Prompt 3A: multi-asset portfolio policy distribution

Implemented the transition from the restricted single-asset categorical action space to a mathematically valid bounded stick-breaking portfolio distribution. The actor output is now `2 * asset_count`: one raw alpha and beta shape value per asset. For each tradable asset in stable order, the transformed shapes parameterise an independent Beta latent; the sampled latent is mapped to `min(maximum_allocation, remaining) * u`, and cash receives the residual. This supports simultaneous positive allocations to multiple assets while guaranteeing finite, non-negative, simplex-valid actions and exact zero for unavailable assets.

- `ActionDistributionConfig` is typed and validated (`bounded_stick_breaking`, minimum/maximum shape, latent epsilon, stable asset order). The distribution identity is embedded in `BrainArchitecture`, configuration hashes, and checkpoint compatibility.
- `PolicyActionSample` stores active asset indices, latent samples, final allocation, cap, exact joint log probability, and entropy. PPO treats the latent vector as the action; no post-hoc cap clipping or unchanged-probability renormalisation is used.
- Burn training recomputes the exact Beta log density and entropy. Actor gradients include digamma/trigamma derivatives, softplus and temperature chain rules, PPO clipping, entropy regularisation, and joint gradient clipping. Critic training remains unchanged. Hidden and recurrent parameters stay frozen; birth genomes remain immutable.
- Deterministic evaluation uses Beta means without consuming RNG or decaying exploration temperature. The central action validator confirms sampled allocations without changing their semantics beyond numerical tolerance.
- Checkpoint format is now 9. Formats 6, 7, and 8 are rejected explicitly because their optimizer, rollout, or categorical action-sample layouts cannot be migrated safely. Evolution crossover, mutation, genome distance, and genome-only inheritance use the expanded actor matrices through architecture validation.
- Regression coverage includes the formerly failing two-asset allocation test, masks, caps from `0.01` through `1.0`, no-tradable cash fallback, deterministic RNG behavior, exact log-probability recomputation, finite-difference Beta gradients, full-head PPO updates, frozen encoders, multi-asset execution, and checkpoint rejection.

The previous controlled experiment suite was rerun with the same synthetic data, seeds, survival pressure, costs, reward, fitness, and horizon after implementation. Results are recorded in `reports/PROMPT3A_CONTROLLED_EXPERIMENT.md`; they remain short-horizon CPU evidence rather than profitability evidence. The stable sequential stick-breaking order is a known ordering bias, and full recurrent BPTT, alternate action distributions, GPU execution, and turnover-penalty experiments remain outside this prompt.

## Prompt 2.5 experimental correctness repair (historical; superseded by Prompt 3A checkpoint v9)

The pre-experiment correctness pass is implemented. Checkpoint format is now 8 because rollout entries carry explicit masked-timestep and actor/critic eligibility state, while action validation records separate policy and universal-market caps.

- Recoverable policy failures execute the fallback once, retain its realized reward and penalty, and append one chronological rollout entry. Successful evaluations may retain a critic sample, but fallback actions are never given a fabricated actor probability; inference and observation failures remain masked with no invented model outputs. Masked GAE uses discounted environment returns across missing-value boundaries so failure penalties affect neighboring eligible samples. Metrics count masked, actor-ineligible, critic-ineligible, and penalty totals.
- Tournament and rank-weighted parent selectors now apply the minimum mating distance while selecting both parents. If no distant second parent exists, the same selector chooses the best available constrained fallback and a deterministic `DiversityConstraintRelaxed` event and metric record the relaxation; uniform random pair replacement is gone.
- Action validation distinguishes an optional policy cap from an optional universal market cap. Neural and random policies use their configured cap; cash, equal-weight, and buy-and-hold benchmarks do not inherit it. Buy-and-hold makes one fully invested initial purchase and then holds, while equal-weight can remain fully diversified subject only to execution economics.
- Deterministic/diagnostic/bootstrapped policy evaluation no longer decays exploration temperature. Reset mutation independently covers every weight and bias and reports both reset counts.

Regression coverage is now 25 core unit tests, 48 foundation tests, and 5 Prompt 2 integration tests (78 tests, no failures). Full recurrent BPTT, alternate distributions, GPU, and Lamarckian inheritance remain outside scope.

Files changed in this pass: `src/lib.rs`, `src/action.rs`, `src/agent.rs`, `src/brain.rs`, `src/evolution.rs`, `src/learning.rs`, `src/metrics.rs`, `src/replacement.rs`, `src/simulation.rs`, the foundation and Prompt 2 integration tests, `README.md`, `docs/EXTENDING.md`, and this status document. Verification passed with `cargo fmt --all --check`, workspace check, Clippy with `-D warnings`, the full workspace test suite, all focused recovery/selection/benchmark/temperature/mutation/checkpoint/atomic filters, and the release CLI validation, simulation, benchmark, and evolution-report commands. The synthetic Prompt 2 run completed 7 steps with 6 PPO updates and no deaths; this is a correctness smoke test, not an experiment claim.

## Prompt 2.25 verification and hardening (historical action-space description; superseded by Prompt 3A)

State-dependent frozen-encoder PPO head learning is implemented and tested. Burn now uses one shared encoded-state path for inference and training. The hidden and recurrent layers remain frozen, while actor and critic dense output weights and biases are lifetime parameters updated by PPO. Gradient accumulation uses the outer product of the encoded representation and the categorical logit/value gradients; entropy, clipping, joint gradient clipping, finite checks, and shape-complete momentum state are included. The birth genome remains immutable.

- `OptimizerState` now contains separate actor/critic `DenseOptimizerState` values with weight and bias momentum vectors sized from the architecture. Restore validates all lengths and finiteness.
- The action distribution is explicitly `SingleAssetCappedCategorical`: cash or one tradable asset at the configured cap plus cash remainder, not a general multi-asset simplex.
- Continuing-value bootstrap fork failures return structured errors and abort the transactional step; zero is used only for non-PPO or terminal paths. Failed lifecycle errors are idempotent without diagnostic double-wrapping.
- Checkpoint format was 7 for the full-head repair and is now 8 after the masked-rollout and benchmark-cap schema change. Formats 6 and 7 are rejected with explicit incompatibility messages.
- Regression coverage includes exact dense outer-product gradients, clipped-gradient suppression, actor/critic head updates with frozen encoders, two-state action/value learning, malformed optimizer restore, and transactional bootstrap-fork failure. The Prompt 2 integration test verifies full-head updates and immutable genomes.

Current test totals are 21 core unit tests, 45 foundation regressions, and 4 Prompt 2 integration tests (70 tests, no failures). Only CPU Burn `NdArray` is implemented. Full recurrent BPTT, hidden-layer training, alternate allocation distributions, GPU, distributed learning, topology evolution, and Lamarckian inheritance remain outside this prompt.

## Prompt 2: lifetime learning and evolution (historical action-space description; superseded by Prompt 3A)

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
