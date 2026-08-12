# Implementation Status

## V3 Phase 3P: counterfactual actor-gradient attribution

Burn PPO now emits report-only actor-head gradient attribution for raw return,
drawdown cost, turnover cost, and residual cost. Counterfactual vectors reuse
the real ratio, clipping decision, action derivatives, recurrent
representation, normalized component advantages, and minibatch divisor but are
never clipped globally, optimized, or applied. A geometry regression locks the
exact sole-component equivalence; checkpoint formats and training behavior are
unchanged.

Seeds 41–42 reproduced all Phase 3O behavior hashes exactly. Drawdown and
turnover gradients were 0.732% and 1.348% of actual policy-gradient norm, with
cosines +0.050 and -0.081 to the actual update. Both were nearly orthogonal to
raw return. The next implementation is an opt-in turnover-constrained PPO mode
with its own cost critic, independently normalized cost advantage, and
checkpointed bounded Lagrange multiplier. Full evidence is in
`reports/V3_PHASE3P_ACTOR_GRADIENT_ATTRIBUTION.md`.

## V3 Phase 3O: PPO advantage-signal audit

PPO maintains a checkpoint-excluded, transactionally forked reward-component
buffer for report-only attribution. Each update records pre/post-normalization
advantage scale, sign changes, rank inversions, and the standard deviation,
correlation, and projection of raw-return, drawdown-cost, turnover-cost, and
residual-cost signals. Restored partial rollouts explicitly mark their first
diagnostic update incomplete; training state and checkpoint formats are
unchanged.

Seeds 41–42 reproduced the Phase 3N economic and existing diagnostic hashes
exactly. Across 6,304 complete updates, normalization caused zero rank
inversions, while drawdown and turnover signals were only 0.716% and 0.773% of
pre-normalization advantage standard deviation. Their projections on the
advantage direction were +0.444% and -0.026%. Normalization is therefore not
the primary bottleneck; counterfactual actor-gradient alignment is next. Full
evidence is in `reports/V3_PHASE3O_ADVANTAGE_SIGNAL_AUDIT.md`.

## V3 Phase 3N: normalized reward semantics and calibration

The opt-in `normalized_risk_adjusted_return` charges only increases in running
maximum drawdown and divides daily trade notional by previous net worth before
applying turnover sensitivity. Historical `risk_adjusted_return` semantics are
unchanged. Reward breakdowns and compact diagnostics expose the normalized
bases and realized penalty totals. Scale-invariance, unchanged-drawdown,
finite-difference slope, and full simulation-path regressions pass.

A predeclared two-seed calibration used drawdown weight `0.15` and turnover
weight `0.0004`. It modestly reduced training turnover but lost to unpenalized
PPO on validation return, drawdown, and turnover in both seeds. The candidate
failed its promotion gate, so it was not run across five seeds or enabled by
default. Full evidence is in
`reports/V3_PHASE3N_NORMALIZED_REWARD_CALIBRATION.md`.

## V3 Phase 3L: full-horizon learning value audit

Five 10-agent momentum populations trained for the full 1,259-step training
split, producing 15,768 PPO updates. Their frozen validation performance was
compared with exact matched controls that preserve each checkpoint's final
genomes but reset all lifetime weights to birth parameters.

- PPO changed mean return by -0.050 percentage points and won two of five
  matched pairs. It improved average median-agent return by 0.306 points, but
  increased turnover and fees by 22.6% and worsened drawdown by 0.804 points,
  losing drawdown in every seed.
- A regression locks the new genome-baseline evaluation contract. Diagnostics
  also supports controlled policy and bankruptcy-threshold overrides.
- Complete-horizon validation benchmarks ranged from -17.187% for random to
  75.624% for the configured 20-day momentum policy. PPO-trained neural agents
  averaged -2.158%, showing that usable point-in-time signal exists but is not
  being learned effectively.
- The active reward is pure log return with zero drawdown and turnover
  penalties, matching the two dimensions that deteriorated under PPO.

The next isolated experiment is reward alignment, not optimizer or architecture
growth. Full evidence is in
`reports/V3_PHASE3L_FULL_HORIZON_LEARNING_AUDIT.md`.

## V3 Phase 3K: frozen held-out optimizer evaluation

The five paired Phase 3J checkpoints were evaluated with learning disabled on
269-step validation and test splits. Diagnostics now exposes the existing
frozen checkpoint-evaluation path while reconstructing the exact overridden
training configuration hash.

- Validation favored calibrated AdamW: it won mean return and drawdown in four
  of five pairs, improved mean return by 0.200 percentage points, reduced mean
  drawdown by 0.160 points, and reduced aggregate turnover by about 2.8%.
- The untouched test split did not confirm the advantage. AdamW won mean return
  in two of five pairs; its mean advantage shrank to 0.030 percentage points,
  while median-agent return was 0.361 points worse and drawdown 0.097 points
  worse.
- Every evaluation had zero learning updates, invalid actions, failures, or
  deaths. Rates remained frozen at momentum `3e-4` and AdamW `3.5e-6`.

Momentum remains the default. AdamW remains an experimental option because its
short training/validation advantage did not generalize robustly to test. Full
evidence is in `reports/V3_PHASE3K_FROZEN_OPTIMIZER_EVALUATION.md`.

## V3 Phase 3J: paired multi-seed optimizer comparison

Momentum `3e-4` and calibrated AdamW `3.5e-6` were compared across five paired
10-agent, 127-step runs using disjoint SplitMix V2 seeds 41–45.

- AdamW won mean return in four of five pairs. Its across-seed mean was 2.045%
  versus momentum's 1.932%, a mean paired advantage of 0.113 percentage points.
- The paired-difference sample standard deviation was 0.176 percentage points,
  and AdamW won within-population median return in only two of five seeds. Its
  average median return was 0.050 percentage points lower.
- Mean drawdown, turnover, fees, clipping frequency, and global relative update
  were nearly matched. All ten runs had zero invalid actions, failures, deaths,
  or policy-ratio clipping.
- Adaptive preconditioning still redistributed updates: AdamW moved hidden and
  critic parameters more and the actor head less than momentum.

Momentum remains the default. AdamW is retained at `3.5e-6` for a frozen-rate,
longer-horizon or held-out comparison. Full evidence is in
`reports/V3_PHASE3J_MULTI_SEED_OPTIMIZER_COMPARISON.md`.

## V3 Phase 3I: AdamW learning-rate calibration

AdamW was calibrated against the retained SplitMix V2 momentum baseline using
four 10-agent, 127-step candidates. The diagnostics runner now supports a
fixed `--learning-rate` override across both root PPO configuration and the
genome trait; a regression prevents the trait from silently superseding the
requested experiment value.

- AdamW `3.5e-6` produced a mean relative update of `1.43999e-5`, within 2.1%
  of momentum's `1.47088e-5`. Rates `3e-6`, `4e-6`, and `1e-5` were 16.0%
  below, 11.6% above, and 179.3% above the momentum scale respectively.
- All four candidates completed 310 PPO updates with zero invalid actions,
  policy failures, deaths, or policy-ratio clipping.
- Equal global update scale did not imply equal component movement: calibrated
  AdamW moved hidden and critic parameters more, but actor parameters less,
  than momentum.

Momentum remains the default. AdamW `3.5e-6` is the retained candidate for a
paired multi-seed optimizer comparison. Full evidence is in
`reports/V3_PHASE3I_ADAMW_RATE_CALIBRATION.md`.

## V3 Phase 3H: fully checkpointed AdamW

AdamW is implemented across the hidden encoder, packed GRU, actor, and critic.
It uses clipped gradients, bias-corrected first and second moments, configurable
epsilon, and decoupled weight decay on weights but not biases. Momentum remains
the serialization-compatible default; AdamW is selected through the optional
top-level `[optimizer]` configuration.

- AdamW moment tensors are shape/finiteness validated and serialized in a new
  neural-state variant. Baseline/awareness checkpoint formats are now 21/22,
  with formats 12–20 still readable.
- Numerical coverage reconstructs first-step weight and bias updates from the
  saved moments. Malformed state is rejected, and checkpoint/resume matches an
  uninterrupted AdamW run.
- The prior momentum baseline remains exact: the new trace hash is unchanged,
  and checkpoint bytes differ only in the four-byte format version.
- A paired 10-agent, 127-step V2 run completed with zero invalid actions and
  deaths. AdamW raised mean return by 0.105 percentage points and reduced
  turnover/fees by 5.72%, but increased mean drawdown by 0.214 percentage
  points.
- At the inherited `0.0003` learning rate, AdamW relative updates were about
  86 times larger than momentum, with correspondingly large hidden, recurrent,
  and head deltas. This setting is mechanically stable but not calibrated.

Momentum remains the default. The next optimizer experiment should tune AdamW
learning rate to the retained update scale before any multi-seed adoption
decision.

## V3 Phase 3G: versioned domain-separated seed derivation

All simulation-owned random streams now pass through a single versioned seed
derivation API. Existing configurations default to `legacy_xor`, which is
omitted from serialization so retained JSON hashes and old checkpoints remain
compatible. New experiments opt into `splitmix_v2`; that explicit choice is
part of configuration identity and checkpoint validation.

- V2 separates initial genome, policy, PPO learning, evolution, replacement,
  selection, crossover, mutation, and evaluation streams with stable tags and
  SplitMix64 finalization.
- Adjacent roots 41–45 produce disjoint initial-genome seed sets for agents
  0–9, eliminating the Phase 3F reordered-ensemble alias.
- A V2 checkpoint/resume regression matches uninterrupted domain history and
  agent state. Legacy configuration snapshots and version-12 migration tests
  remain exact; checkpoint formats stay at 19/20.
- Two concurrent 10-agent, 127-step V2 captures produced byte-identical
  checkpoints (`d4cf38b5...ecc15ad`) and traces (`d014e369...33bb9`), with zero
  invalid actions or deaths and roughly 46 MiB peak RSS each.

The V2 seed-41 clip-1 baseline returned 1.881% on average with 66.77% gradient
clipping. Because the new seed protocol intentionally changes the population,
it replaces rather than extends the legacy seed-41 artifact. The next isolated
experiment is a paired V2 momentum-versus-AdamW comparison.

## V3 Phase 3F: disjoint multi-seed clipping decision

The clip-1/clip-2 comparison was extended to five paired 10-agent runs using
disjoint seed ensembles 41, 73, 105, 137, and 169.

- An initial adjacent-seed protocol was rejected during analysis: initial
  genome and policy seeds use `simulation.seed ^ agent_id`, so adjacent seeds
  can produce the same 10 seed values in a different order. Seeds 42/43 and
  44/45 were therefore excluded rather than counted as independent evidence.
- Clip 2.0 won mean return in three of five valid pairs. Its across-seed mean
  return was 2.091% versus 1.988% for clip 1.0, but its median was lower
  (2.209% versus 2.254%) and sample variability was higher.
- Clip 2.0 increased average turnover and fees by 0.28% and mean drawdown by
  0.0206 percentage points. These are small changes, but there was no
  consistent economic improvement to offset them.
- Mechanically, clip 2.0 remained healthy across every seed: zero deaths,
  invalid actions, and PPO policy-ratio clipping; mean absolute KL remained
  below `0.000020`; gradient clipping fell from 70.9% to 22.6%; and relative
  updates increased by about 52.2%.
- Peak RSS and concurrent runtime were unchanged in practical terms at about
  45 MiB per process and 35 seconds per paired run.

The configured default remains clip 1.0. Before an AdamW comparison, seed
derivation should be hardened with a deterministic mixing function so ordinary
adjacent experiment seeds produce distinct agent ensembles. That change must
be versioned because it intentionally changes seeded baselines.

## V3 Phase 3E: global gradient-clip comparison

The 10-agent, 127-step Phase 3D protocol was repeated concurrently with global
gradient clipping thresholds 1.0 and 2.0. All other inputs were identical.

- Raising the threshold reduced clipping frequency from 72.6% to 23.4% and
  increased mean relative parameter-update size from `1.52e-5` to `2.32e-5`.
- PPO remained mechanically stable: both runs had zero invalid actions, zero
  deaths, zero policy-ratio clipping, and approximate KL near zero.
- Pre-clip hidden, recurrent, GRU-gate, actor, and critic norms were nearly
  unchanged. The relaxed threshold changes how much gradient is applied, not
  the raw backpropagated signal.
- Hidden, recurrent, actor, and critic lifetime deltas all increased under 2.0,
  while memory use and runtime remained effectively unchanged.
- The single seed did not show a performance benefit: clip 2.0 ended about
  0.15% lower in mean net worth and generated about 1.04% more turnover and
  fees. This short synthetic result is not sufficient to select a default.
- The clip-1 checkpoint exactly reproduced the Phase 3D 10-agent artifact,
  SHA-256 `4aecbd56...cc9ad3`, providing an independent deterministic control.

Clip 2.0 passes the stability gate but remains experimental. The next step is a
paired multi-seed clip-1/clip-2 comparison before either changing the default or
introducing AdamW.

## V3 Phase 3D: paired population gradient baseline

The retained 127-step combined-senses protocol was run concurrently with 10
and 20 agents using the same seed, architecture, learning configuration, and
market data.

- Both runs completed with zero deaths and zero invalid actions. The 10-agent
  run produced 310 PPO updates and the 20-agent run produced 620.
- Peak RSS was 46,668 KiB and 51,748 KiB respectively. Contended wall time was
  36.3 and 64.7 seconds on the reported eight-way host.
- Mean pre-clip norms were 1.585 and 1.596; gradient clipping occurred in 72.6%
  and 73.1% of optimizer minibatches. PPO policy clipping remained zero and
  approximate KL remained near zero.
- The candidate path remained the largest GRU component, but the reset-gate
  norm increased more than threefold between the first and second halves in
  both runs. Update-gate norms also increased. The gates are small, not frozen.
- Canonically filtered traces for shared agents 0–9 were byte-identical across
  populations, SHA-256 `226b01f2...62399`. Population size therefore did not
  perturb common-agent execution or learning.

The 20-agent run doubled checkpoint and trace volume while adding only about
5 MiB peak RSS. It did not materially change normalized gradient health. The
next isolated experiment should retain the 10-agent population and compare the
current global gradient clip of 1.0 against 2.0 before considering AdamW.

## V3 Phase 3C: gradient health telemetry

Burn PPO updates now measure gradient and parameter-update health without
changing optimizer behavior or authoritative checkpoint state.

- Report-only PPO traces record the global norm before and after clipping,
  per-hidden-layer norms, the complete recurrent norm, separate GRU update,
  reset, and candidate norms, actor and critic norms, gradient-clipping
  frequency, and update-to-parameter norm ratio.
- Multi-epoch/minibatch updates aggregate every value deterministically. GRU
  gate norms are absent rather than fabricated for legacy dense-recurrent
  checkpoints.
- `midas-diagnostics` publishes these measurements under
  `learning.gradient_health`. Report traces remain excluded from checkpoints,
  so formats 19/20 and legacy resume compatibility are unchanged.
- A seven-step, three-agent Prompt 2 smoke completed six PPO updates. Gradient
  clipping occurred in 41.7% of optimizer minibatches; mean pre/post-clip norms
  were 1.037/0.808, and mean relative update norm was `1.20e-5`.
- In that smoke, the GRU candidate gradient (0.442) dominated the update and
  reset gates (0.00253 and 0.00138). This is a diagnostic signal from a very
  short synthetic run, not yet evidence that gate learning is persistently
  vanishing.

The next controlled experiment should measure these values over the retained
127-step baseline and multiple seeds before deciding whether to tune learning
rates or replace momentum with AdamW.

## V3 Phase 3B: standard GRU replacement

New neural genomes now use a standard update/reset/candidate GRU for inference
and truncated backpropagation through time. The three gate parameter blocks are
packed into the existing evolvable recurrent `DenseGenome`, so mutation,
crossover, genetic distance, immutable birth genomes, and lifetime optimizer
state continue to cover the complete recurrent cell.

- Inference and Burn autodiff use the same equations, including reset-gated
  hidden contributions to the candidate state.
- Hand-calculated inference and finite-difference regressions cover update,
  reset, candidate-input, and candidate-hidden parameters.
- Genome format 4 identifies the packed GRU layout. Legacy genome format 3
  retains the original dense-SiLU recurrence exactly; resumed legacy
  populations also create legacy-layout replacements.
- Mixed recurrent layouts are rejected by crossover and genetic-distance
  operations rather than silently truncating parameter vectors.
- Baseline checkpoints now use format 19 and awareness checkpoints use format
  20. Formats 12–18 remain readable; a retained format-17 BPTT checkpoint
  completed validation evaluation successfully.
- Two independent 127-step seed-41 runs completed 248 PPO updates and 1,016
  neural actions with zero invalid actions. Both produced the same
  1,055,827-byte checkpoint, SHA-256 `2175abd6...ecf3a78`.

Phase 3C subsequently added optimizer and per-component gradient observability.
Controlled larger-architecture and longer-window experiments remain future
work.

## V3 Phase 3A: truncated recurrent backpropagation

Burn now unrolls the existing simple recurrent memory for each configured PPO
sequence and backpropagates through time to hidden, recurrent, actor, and critic
parameters.

- Rollout assembly marks every recurrent sequence start explicitly. Eligible
  fragments are capped by `sequence_length`, shuffled only as complete
  sequences, and kept whole when packed into minibatches.
- Masked or ineligible environment timesteps create hard sequence boundaries;
  temporal gradients cannot bridge missing model states.
- Each sequence begins from its stored pre-action memory as a detached initial
  condition. Later timesteps consume graph-produced memory, allowing temporal
  credit assignment within the truncated window.
- Recurrent weight and bias momentum is persistent in `BpttOptimizerState`.
  Birth genomes remain immutable. Formats 15/16 migrate with zero recurrent
  momentum; current baseline and awareness checkpoints use formats 17/18.
- A focused regression applies loss only at timestep two and proves that the
  earlier observation receives a gradient only when both timesteps share a
  sequence. Parallel checkpoint/resume integration remains exact.
- Two independent 127-step seed-41 runs completed 248 PPO updates with zero
  invalid actions and produced the same 1,004,137-byte checkpoint, SHA-256
  `4f96ccc7...8156bb1`. Debug runtime was 21.5–22.3 seconds.

This section records the Phase 3A boundary. Phase 3B subsequently replaced the
cell for new genomes with a standard GRU while preserving the legacy execution
path for older checkpoints. Increasing truncation windows and adding per-layer
gradient telemetry remain future stages.

## V3 Phase 2: trainable hidden encoder

Burn autodiff now spans raw observations, every configured feed-forward hidden
layer and activation, the fixed recurrent projection, and both PPO heads.

- Hidden weights and biases are optimized with the same clipped momentum rule
  as the actor and critic. Their momentum is persisted in
  `TrainableEncoderOptimizerState`.
- Recurrent weights and sampled recurrent states remain fixed boundaries, but
  the recurrent projection remains inside the graph so gradients reach the
  hidden encoder. Birth genomes remain immutable.
- New Burn neural snapshots retain the legacy frozen-encoder enum layout and
  add a trainable-encoder variant. Formats 13/14 remain readable, with hidden
  momentum initialized to zero; new baseline and awareness checkpoints use
  formats 15 and 16.
- NdArray autodiff graph construction/backward is serialized within a process
  because parallel Burn graphs caused small schedule-dependent differences.
  Inference remains concurrent. Parallel integration tests now preserve exact
  checkpoint/resume determinism.
- Two independent 127-step seed-41 runs completed 248 PPO updates with zero
  invalid actions and produced the same 996,825-byte checkpoint, SHA-256
  `8c3fc8f6...142a9f8a`. Each debug run took approximately 13.7 seconds.

Phase 3A subsequently added recurrent-parameter training and truncated
backpropagation through time, and Phase 3B added GRU memory. Larger
architectures and AdamW remain future V3 stages.

## V3 Phase 1: frozen-encoder Burn autodiff integration

Burn training now uses an explicit `Autodiff<NdArray>` backend for actor and
critic head gradients while the existing `NdArray` tensors remain the inference
runtime. The hand-built dense outer-product accumulator has been removed.

- The bounded-Beta distribution still computes its exact log-probability and
  entropy derivatives at the logits boundary; Burn 0.18 has no differentiable
  lgamma/digamma tensor operations. Burn owns the complete vector-Jacobian
  product from those logits through both dense heads.
- Hidden and simple-recurrent lifetime parameters remain frozen and the genome
  remains immutable. Existing momentum updates, optimizer serialization,
  checkpoint format, public neural interfaces, and inference behavior are
  unchanged.
- Focused tests prove the Burn graph matches exact dense outer products, actor
  and critic learning remain state-dependent, clipped gradients remain zero,
  and encoder/genome parameters stay unchanged.
- A 127-step combined-senses seed-41 replay produced 248 PPO updates with no
  deaths or invalid actions. It differed from the captured manual-gradient
  baseline by at most approximately `1.8e-9` net worth and repeated with a
  byte-identical Phase 1 checkpoint. Debug runtime increased from 5.94 to 7.42
  seconds and sampled peak RSS from 43.59 to 45.39 MiB.
- The complete 1,259-step training split completed 2,522 PPO updates with zero
  invalid actions in 83.09 seconds and 69.51 MiB sampled peak RSS. Resume from
  the step-1,152 checkpoint reproduced the uninterrupted result, and an
  independent replay produced the same 4,977,651-byte final checkpoint with
  SHA-256 `ce33d416...a09fa374`.

Phase 2 subsequently extended this graph through the feed-forward hidden
encoder while retaining a fixed recurrent boundary.

## V2: self-awareness and evolvable traits

The first V2 implementation is complete and opt-in through
`configs/v2-evolvable.toml`.

- The observation registry now supports six named agent-awareness signals:
  recent reward, rolling reward mean, rolling reward volatility, exploration
  level, survival-cost ratio, and reward trend. Their values are published by
  a dedicated provider and enter the vector only when the provider and schema
  features are configured.
- `AgentState` persists reward EMAs and trend state. Neural policies expose
  current and birth temperatures through the policy boundary, so the provider
  remains independent of policy implementation details.
- All seven genome traits are enabled in the V2 config. Learning rate mutation
  uses log-space noise; the other traits use bounded mutation. Neural trade
  impulses honor inherited maximum allocation, and risk-adjusted reward uses
  inherited drawdown/turnover sensitivities when they are enabled.
- Current awareness checkpoints use format 20 and baseline checkpoints use
  format 19. Formats 12–18 remain readable; missing hidden or recurrent
  optimizer momentum is initialized to zero during migration.

The V2 configuration has been validated and exercised through a one-step PPO
evolution run, including a format-14 checkpoint write. Experimental comparison
results from the six-condition protocol remain to be collected.

## Prompt 9D: stateful replacement checkpointing

Checkpoint format 13 adds a versioned `ReplacementStrategyState` containing a
stable strategy identity and opaque deterministic payload. Automatic
checkpoints snapshot the transaction-local replacement fork before authoritative
commit, so strategy state and the population transition share one checkpoint
boundary.

- `ReplacementStrategy::checkpoint_state` is required for checkpoint-safe
  custom strategies; its default fails explicitly instead of silently dropping
  state. Built-in strategies emit validated stateless version-1 snapshots.
- `ReplacementStrategyFactory` restores strategy state during resume and frozen
  evaluation. The engine verifies that the restored strategy re-snapshots to
  the exact stored identity, version, and payload. New factory-aware resume and
  evaluation entry points preserve the existing built-in APIs.
- Format-12 checkpoints are decoded through their original envelope and safely
  reconstruct the configured built-in strategy. They retain identical reports
  and are written in the current format only when explicitly re-saved; older
  incompatible formats remain rejected.
- A stateful alternating-strategy regression proves byte-identical uninterrupted
  and checkpoint-resumed state, including the automatic pre-commit checkpoint
  path. Configuration hashes and report serialization are unchanged.

## Prompt 9C: replacement context narrowing

`ReplacementContext` no longer exposes the root serialized `Config` to
replacement strategies. It now carries explicit replacement and evolution
DTOs, evolvable brain traits, architecture, compute device, observation schema,
deterministic seed material, living summaries, completed lives, and the genome
registry.

- Built-in replacement behavior, selection inputs, mutation and crossover
  configuration, RNG derivation, spawn requests, and transactional strategy
  forking are unchanged.
- Learning and complete agent construction remain owned by `AgentFactory`
  after the strategy returns an `AgentSpawnRequest`; replacement strategies do
  not receive unrelated market, execution, reward, checkpoint, or simulation
  configuration.
- An architecture regression prevents `replacement.rs` from regaining a root
  `Config` dependency. Configuration serialization and hashes, checkpoint
  format 12, checkpoint bytes, and deterministic reports remain unchanged.

## Prompt 9B: replacement population semantics

Evolutionary parent selection now uses the configured reproductive-fitness
function for every eligible living agent instead of treating raw net worth as
fitness. `LivingAgentSummary` exposes that finite selection score explicitly so
replacement strategies receive the same fitness contract for living and
completed lives.

- A death batch freezes its eligible parent IDs before any replacement is
  spawned. Newborn agents cannot become parents for later deaths in the same
  simulation step, while replacement slots and event order remain unchanged.
- `active_lineages` now counts distinct lineages in the current living
  population at initialization and after population transitions, rather than
  all lineages ever recorded in the genome registry.
- Existing version-12 checkpoints still resume and re-save byte-identically;
  no stored checkpoint field, serialization layout, RNG stream, or checkpoint
  version changed. Replacement-context narrowing and custom strategy
  checkpoint state remain later work.

## Prompt 9A: replacement correctness contracts

Replacement strategies now expose an additive typed evolutionary report for
reproduction, clone fallback, and random fallback outcomes. Simulation
accounting consumes that report instead of inferring crossover, mutation, and
fallback semantics from free-form spawn-reason text.

- Random fallbacks carry no parent ancestry and increment only the random
  fallback metric. Clone fallbacks retain one eligible parent and inherit that
  parent's lineage. Clone-best fallback respects configured age and net-worth
  eligibility.
- Replacement requests are validated before agent construction and ancestry
  allocation. Unknown parents, incompatible policy/genome combinations,
  missing parent genomes or identities, and invalid mating-distance metadata
  fail transactionally.
- Existing strategy implementations remain source-compatible because the new
  report method has a default. Checkpoint format remains 12.

## Prompt 8C.2: backend device construction ownership

Compute-device construction and validation now belong to the brain domain.
Configuration maps `BackendConfig` into a private brain-owned option contract;
configuration validation, simulation setup, and checkpoint resume use that
contract directly.

- Public `BackendConfig::compute_device` remains compatible through the isolated
  private `config/backend_compat.rs` adapter.
- Backend validation remains the final ordered configuration validation step.
  Device error text, backend identity, configuration serialization and hashes,
  checkpoint format 12, and deterministic behavior are unchanged.

## Prompt 8C.3: configuration dependency audit

The current configuration dependency graph, visibility boundary, accepted leaf
DTO consumption, remaining root-`Config` consumers, and construction edges are
recorded in `docs/CONFIG_ARCHITECTURE.md`. The audit distinguishes composition
and compatibility contracts from candidates for later narrowing.

- A regression test now prevents root `Config` from returning to observation
  schema, builder, registry, normalization, or provider code and confirms the
  established public adapters remain isolated in `observation/config_compat.rs`.
- No production visibility or ownership changes were justified independently
  of the skipped backend/brain construction stage. Backend device construction,
  replacement context, market loading, and agent construction remain explicitly
  documented future work.

## Prompt 8C.1: observation configuration contract narrowing

Observation schema compilation and built-in provider construction now consume
private observation-owned option views containing only their required fields.
Configuration owns conversion from root `Config`; simulation construction,
checkpoint resume, and configuration validation use the narrow contracts.

- Public `ObservationSchema::{from_config, try_from_config}` and
  `SchemaObservationBuilder::from_config` remain compatible through an isolated
  private adapter module.
- Core schema and builder implementations no longer import or accept root
  `Config`. They still consume stable leaf configuration DTOs; broader DTO
  ownership and backend/brain construction cleanup remain later work.
- Schema identities, provider and entry ordering, normalization behavior,
  configuration hashes, public APIs, deterministic replay, and checkpoint
  format 12 are unchanged.

## Prompt 8B.2: domain configuration validation decomposition

Private domain-local validation now lives beside its configuration DTOs for
market/features/corporate actions, simulation/policy failures, execution,
brain, observation, splits, learning, reward, and evolution. Private
`config/validation.rs` remains the explicit ordered coordinator for
`Config::validate`.

- Mixed simulation/execution economic checks and observation schema compilation
  with provider cross-checking remain in the coordinator. Backend/device
  validation remains the final step.
- Public APIs, validation sequence, first-error precedence and text,
  serialization, configuration hashes, deterministic behavior, and checkpoint
  format 12 are unchanged. Configuration dependency reversal has intentionally
  not been performed.

## Prompt 8B.1: ordered configuration validation extraction

The complete ordered `Config::validate` implementation now lives in private
`config/validation.rs`. This is a module-boundary extraction only: the method
signature, validation sequence, observation-schema construction position,
backend-last check, first-error precedence, and error text are unchanged.

- `config.rs` remains the public compatibility facade and still owns root
  `Config`, loading, and compact-JSON hashing.
- Validation has not been delegated into domain modules, and dependency
  direction between configuration, brain, and observation remains unchanged.
  Serialization, configuration hashes, deterministic behavior, and checkpoint
  format 12 remain unchanged.

## Prompt 8A.2: high-coupling configuration facade split

Stage 2 of the mechanical configuration modularization is complete. Brain,
observation, evolution, replacement, and date/split DTOs, defaults, and local
helpers now live in private child modules under `config/`. Together with Stage
1, every domain configuration DTO is now owned by a child module while
`config.rs` preserves all established public re-exports.

- Root `Config` field order, loading, compact-JSON hashing, and the complete
  ordered validation body remain in `config.rs`.
- Configuration serialization and hashes, rolling-default omission, validation
  precedence and text, public `midas_core::config::*` imports, and checkpoint
  format 12 are unchanged. Dependency-direction cleanup between configuration,
  brain, and observation has intentionally not been performed yet.

## Prompt 8A.1: low-risk configuration facade split

Stage 1 of the configuration modularization is complete. Backend, checkpoint,
execution, learning, market, reward, and simulation configuration DTOs,
defaults, and local helpers now live in private child modules under `config/`.
`config.rs` publicly re-exports the established types, so existing
`midas_core::config::*` imports remain valid.

- Root `Config` loading, hashing, validation, and cross-domain coordination are
  unchanged and remain in `config.rs`.
- Brain, observation, evolution, replacement, date, and split configuration
  remain in the facade for the later high-coupling stage. Serialization,
  defaults, validation behavior, configuration hashes, and checkpoint format 12
  are unchanged.

## Prompt 8A.0: configuration regression snapshots

A permanent configuration-compatibility suite now records the exact compact
JSON bytes and BLAKE3 hashes for the default, test, Prompt 6 combined-PPO, and
Prompt 6 combined-PPO/evolution configurations. It also locks root field and
`BTreeMap` order, `Option` encoding, default expansion, required sections,
validation-error precedence and text, and the special omitted-versus-enabled
`rolling_market_stats` representation.

- A CLI manifest records the exact validation stdout and hash for every shipped
  TOML under `configs/`; the integration test also requires empty stderr and an
  exact match between the manifest and the files on disk.
- The suite resumes and evaluates a tracked version-12 checkpoint, compares
  exact report JSON, proves explicit re-saving is byte-identical, and verifies
  that the source checkpoint is never modified.
- These tests are the compatibility boundary for the planned Prompt 8
  configuration architecture work. No production configuration, simulation,
  serialization, hash, CLI, or checkpoint behavior changed.

## Prompt 7B.7: final modular simulation facade

The staged simulation refactor is complete. `simulation.rs` is now a thin
compatibility facade that declares private implementation modules and
re-exports the established public engine, report, event, trace, lifecycle, and
checkpoint-version contracts.

- Frozen checkpoint evaluation and evaluation-only reset/setup now live in
  private `simulation/evaluation.rs`. Report construction and the unchanged
  serialized `SimulationReport` contract now live in private
  `simulation/report.rs`.
- Public checkpoint save/resume methods now sit with the existing private
  checkpoint envelope and compatibility helpers in `simulation/checkpoint.rs`.
  Public APIs, report bytes, deterministic replay, and checkpoint format 12
  remain unchanged.

## Prompt 7B.6: simulation learning and death/replacement split

The simulation-to-learning integration now lives in private
`simulation/learning_bridge.rs`, including bootstrap value estimation,
transition recording, eligibility and masking, episode completion, PPO update
application, learning events, and PPO traces. Private
`simulation/death_replacement.rs` now owns completed-life and fitness handling,
replacement invocation, reproduction, genome/lineage bookkeeping, and related
evolution events.

- Generic learning, evolution, and replacement algorithms remain in their
  existing modules; these child modules contain only simulation integration.
- PPO timing and contents, death/replacement ordering, RNG and event/trace
  ordering, deterministic replay, public APIs, and checkpoint format 12 are
  unchanged. Checkpoint/resume, frozen evaluation, and report construction
  remain in `simulation.rs` for Prompt 7B.7.

## Prompt 7B.5: simulation decision and economic advance split

The daily policy-decision pipeline now lives in private
`simulation/decision.rs`, including observation and policy evaluation, action
validation, policy-failure handling, fallback behavior, daily agent dispatch,
and committed decision-trace construction. Private `simulation/advance.rs` now
owns the unchanged economic advancement path: corporate actions, execution,
valuation, survival costs, drawdown, rewards, and trade summaries.

- The close-to-next-open order, event and trace timing, transaction boundary,
  reward and execution semantics, and deterministic replay are unchanged.
- Learning/PPO integration, death and replacement handling, checkpoint and
  evaluation methods, and report construction remain in `simulation.rs`.
  Public APIs and checkpoint format 12 are unchanged.

## Prompt 7B.4: simulation runner and transaction split

The simulation lifecycle runner now lives in private `simulation/runner.rs`,
including `run`, `run_steps`, step dispatch, failure transitions, and final
learning flush. Private `simulation/transaction.rs` now owns the step
transaction buffer, abort type, agent-state fork, authoritative commit, and
rollback-by-discard boundary.

- Automatic checkpoints are still built and written from the completed
  transaction candidate before authoritative state mutation; agent, evolution,
  metrics, events, traces, and `next_step` then commit in the original order,
  followed by the operational checkpoint event.
- Daily policy decisions, economic advancement, learning, death, replacement,
  checkpoint public methods, frozen evaluation, and reporting remain in
  `simulation.rs`. Public APIs and checkpoint format 12 are unchanged.

## Prompt 7B.3: simulation engine shell split

The stable `SimulationEngine` state, founder-population construction, public
inspection methods, event-sink and replacement hooks, event emission, and
read-only valuation-price lookup now live in the private
`simulation/engine.rs` module. The compatibility facade still re-exports the
type as `midas_core::simulation::SimulationEngine`, including its existing
public metrics and event/trace fields.

- The run loop, daily-step orchestration, transactions, learning, replacement,
  checkpoint public methods, frozen evaluation, and report construction remain
  in `simulation.rs` for later staged splits.
- Checkpoint format remains 12. Founder seeds and ordering, schema and
  architecture identities, event timing, checkpoint bytes, deterministic
  replay, and all public method signatures are unchanged.

## Prompt 7B.2: private simulation checkpoint module

The staged `midas_core::simulation` split now keeps its checkpoint envelope,
agent snapshots, state-to-checkpoint construction, bincode file I/O, and
compatibility validation in the private `simulation/checkpoint.rs` module.
The public `SimulationEngine` checkpoint, resume, and frozen-evaluation methods
remain unchanged on the simulation facade, and checkpoint creation timing and
simulation orchestration remain in `simulation.rs`.

- Checkpoint format remains 12. Serialized struct field order, derives,
  validation order, compatibility checks, and error text are preserved.
- Existing `midas_core::simulation::*` imports remain compatible. No simulation
  ordering, transaction, learning, replacement, evolution, report, or frozen
  evaluation behavior changed.

## Prompt 7B.1: simulation types, events, and diagnostics split

The staged `midas_core::simulation` facade split has begun without moving the
engine or changing simulation behavior. `simulation/types.rs` now owns the
checkpoint-version constant, dataset split, lifecycle, and death-cause
contracts; `simulation/events.rs` owns deterministic domain and operational
events; and `simulation/diagnostics.rs` owns report-only decision and PPO
traces. The public `midas_core::simulation::*` imports remain unchanged through
facade re-exports.

- Checkpoint format remains 12. The checkpoint envelope, serialized field and
  enum layouts, and all simulation methods remain in `simulation.rs` for this
  stage.
- Transaction boundaries, event timing, diagnostics timing, PPO, execution,
  rewards, evolution, replacement, observations, and configuration behavior
  are unchanged.

## Prompt 7A: brain facade split

The former 2,766-line `midas_core::brain` module is now a compatibility
facade over focused private implementation modules. Public imports remain
unchanged through `midas_core::brain::*`; `brain.rs` only declares private
children and re-exports the established contracts.

- `brain/architecture.rs`, `genome.rs`, and `state.rs` contain the stable
  architecture, immutable genome, and checkpointed lifetime/runtime/optimizer
  state respectively. Their derives, fields, and enum variants are unchanged.
- `brain/model.rs` contains framework-neutral model/device contracts;
  `brain/burn_runtime.rs` contains the CPU Burn adapter and its cached tensors;
  `brain/head_training.rs` contains the existing frozen-encoder PPO head
  update math.
- `brain/policy.rs`, `impulse_beta.rs`, and `benchmarks.rs` isolate policy
  snapshots, bounded impulse-distribution logic, and benchmark policies.
  This is a mechanical relocation only: sampling, temperature timing, PPO,
  evolution, execution, observations, and checkpoint semantics are unchanged.
- Checkpoint format remains 12. Existing v12 checkpoints continue to use the
  same bincode field and enum layouts; no migration or architecture/schema
  identity change is introduced.

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
- PPO continues to use the exact independent Beta latent log probability and entropy; deterministic evaluation uses Beta means and does not consume RNG or decay temperature. At the Prompt 4 stage, actor/critic head training, the frozen encoder, reward, and evolution were unchanged; V3 now trains the feed-forward hidden encoder.
- Decision traces now include requested/executed impulses, notionals, scaling, rejected notional, cash and holdings before/after, and impulse magnitude/saturation metrics. Portfolio allocation metrics describe the persistent post-execution holdings.
- Checkpoint format was 11 for Prompt 4 and is now 12 after Prompt 5's schema-identity change. Format 10 and formats 6 through 11 are rejected because their action samples, rollout/policy snapshots, metrics, or observation layouts are incompatible; no unsafe migration is attempted. Birth genomes remain immutable and replacement lifetime state is fresh.

Prompt 4's controlled rerun is recorded in [Prompt 4 Persistent Incremental Trade Experiment](reports/PROMPT4_INCREMENTAL_TRADE_EXPERIMENT.md). It is a mechanical action-interface change, not evidence of profitability or intelligence. Known limitations are the independent per-asset Beta impulse factorisation, stable latent ordering, and frozen hidden/recurrent encoder.

## Prompt 3B: long-horizon diagnosis and adaptive reward experiment

### V3 Phase 3M: calibrated reward-penalty comparison

Diagnostics can override and record reward kind, drawdown weight, and turnover
weight without editing the base configuration. Their compact summaries now
include aggregate raw-return, drawdown-penalty, turnover-penalty, and unclipped
reward totals from the committed decision traces. A five-seed full-horizon
comparison used drawdown weight `0.0005` and raw-notional turnover weight
`5e-8`, changing no other Phase 3L control. The treatment was active at 20.7%
of the calibration daily-return scale, but penalized PPO averaged -2.218%
validation return, 4.531% drawdown, and 31,140 population turnover versus
-2.158%, 4.534%, and 28,279 unpenalized. These penalty weights are therefore
not defaults. Details and raw artifacts are in
[V3 Phase 3M](reports/V3_PHASE3M_PENALIZED_REWARD_COMPARISON.md).

V3 Phase 3Q implements opt-in turnover-constrained PPO with a separate cost
critic, independently normalized cost advantage, and projected checkpointed
dual variable. Existing PPO remains one-critic and legacy checkpoints/config
hashes remain readable. Mechanical gates and all 164 workspace tests pass.
Two two-seed training/validation calibrations failed the primary held-out
turnover gate, including a stronger controller whose mean multiplier reached
3.01. The mode remains experimental, drawdown remains diagnostic-only, no
five-seed promotion was run, and the test split was untouched. Results are in
[V3 Phase 3Q](reports/V3_PHASE3Q_TURNOVER_CONSTRAINED_PPO.md).

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

The previous controlled experiment suite was rerun with the same synthetic data, seeds, survival pressure, costs, reward, fitness, and horizon after implementation. Results are recorded in `reports/PROMPT3A_CONTROLLED_EXPERIMENT.md`; they remain short-horizon CPU evidence rather than profitability evidence. The stable sequential stick-breaking order is a known ordering bias. Recurrent BPTT was outside Prompt 3A and was added later in V3 Phase 3A; alternate action distributions, GPU execution, and turnover-penalty experiments remain future work.

## Prompt 2.5 experimental correctness repair (historical; superseded by Prompt 3A checkpoint v9)

The pre-experiment correctness pass is implemented. Checkpoint format is now 8 because rollout entries carry explicit masked-timestep and actor/critic eligibility state, while action validation records separate policy and universal-market caps.

- Recoverable policy failures execute the fallback once, retain its realized reward and penalty, and append one chronological rollout entry. Successful evaluations may retain a critic sample, but fallback actions are never given a fabricated actor probability; inference and observation failures remain masked with no invented model outputs. Masked GAE uses discounted environment returns across missing-value boundaries so failure penalties affect neighboring eligible samples. Metrics count masked, actor-ineligible, critic-ineligible, and penalty totals.
- Tournament and rank-weighted parent selectors now apply the minimum mating distance while selecting both parents. If no distant second parent exists, the same selector chooses the best available constrained fallback and a deterministic `DiversityConstraintRelaxed` event and metric record the relaxation; uniform random pair replacement is gone.
- Action validation distinguishes an optional policy cap from an optional universal market cap. Neural and random policies use their configured cap; cash, equal-weight, and buy-and-hold benchmarks do not inherit it. Buy-and-hold makes one fully invested initial purchase and then holds, while equal-weight can remain fully diversified subject only to execution economics.
- Deterministic/diagnostic/bootstrapped policy evaluation no longer decays exploration temperature. Reset mutation independently covers every weight and bias and reports both reset counts.

At that stage, regression coverage was 25 core unit tests, 48 foundation tests,
and 5 Prompt 2 integration tests (78 tests, no failures). Recurrent BPTT was
added later in V3 Phase 3A; alternate distributions, GPU, and Lamarckian
inheritance remain outside scope.

Files changed in this pass: `src/lib.rs`, `src/action.rs`, `src/agent.rs`, `src/brain.rs`, `src/evolution.rs`, `src/learning.rs`, `src/metrics.rs`, `src/replacement.rs`, `src/simulation.rs`, the foundation and Prompt 2 integration tests, `README.md`, `docs/EXTENDING.md`, and this status document. Verification passed with `cargo fmt --all --check`, workspace check, Clippy with `-D warnings`, the full workspace test suite, all focused recovery/selection/benchmark/temperature/mutation/checkpoint/atomic filters, and the release CLI validation, simulation, benchmark, and evolution-report commands. The synthetic Prompt 2 run completed 7 steps with 6 PPO updates and no deaths; this is a correctness smoke test, not an experiment claim.

## Prompt 2.25 verification and hardening (historical action-space description; superseded by Prompt 3A)

State-dependent frozen-encoder PPO head learning is implemented and tested. Burn now uses one shared encoded-state path for inference and training. The hidden and recurrent layers remain frozen, while actor and critic dense output weights and biases are lifetime parameters updated by PPO. Gradient accumulation uses the outer product of the encoded representation and the categorical logit/value gradients; entropy, clipping, joint gradient clipping, finite checks, and shape-complete momentum state are included. The birth genome remains immutable.

- `OptimizerState` now contains separate actor/critic `DenseOptimizerState` values with weight and bias momentum vectors sized from the architecture. Restore validates all lengths and finiteness.
- The action distribution is explicitly `SingleAssetCappedCategorical`: cash or one tradable asset at the configured cap plus cash remainder, not a general multi-asset simplex.
- Continuing-value bootstrap fork failures return structured errors and abort the transactional step; zero is used only for non-PPO or terminal paths. Failed lifecycle errors are idempotent without diagnostic double-wrapping.
- Checkpoint format was 7 for the full-head repair and is now 8 after the masked-rollout and benchmark-cap schema change. Formats 6 and 7 are rejected with explicit incompatibility messages.
- Regression coverage includes exact dense outer-product gradients, clipped-gradient suppression, actor/critic head updates with frozen encoders, two-state action/value learning, malformed optimizer restore, and transactional bootstrap-fork failure. The Prompt 2 integration test verifies full-head updates and immutable genomes.

At that stage, test totals were 21 core unit tests, 45 foundation regressions,
and 4 Prompt 2 integration tests (70 tests, no failures). Hidden-layer training
and recurrent BPTT were added later in V3; alternate allocation distributions,
GPU, distributed learning, topology evolution, and Lamarckian inheritance
remain unimplemented.

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

Known remaining limitations are explicit: only CPU Burn NdArray is implemented;
PPO trains hidden, GRU, and head parameters through truncated sequences.
Diversity protection is a
bounded sharing/distance mechanism rather than species clustering; no GPU,
distributed training, topology evolution, Lamarckian inheritance, or
profitability claim is provided.

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

Known remaining foundation limitations are deliberate: CPU `NdArray` is the only neural backend; corporate actions are explicit split/dividend records and are not inferred from providers; and event persistence is in-memory. Later V3 phases added full-network recurrent autodiff and GRU memory; advanced evolutionary research remains future work.

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
