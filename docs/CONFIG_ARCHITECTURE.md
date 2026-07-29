# Configuration Architecture

## Current boundary

`crates/midas-core/src/config.rs` is the public compatibility facade. It owns
the serialized root `Config`, loading, compact-JSON hashing, private module
declarations, and public re-exports. Domain DTOs, defaults, and domain-local
validation live in private files under `config/`. The explicit cross-domain
validation order lives in `config/validation.rs`.

Configuration field order, serde attributes, defaults, validation precedence,
and error text are compatibility contracts. `Config::hash` is the BLAKE3 hash
of `serde_json::to_vec(self)`. Physical module placement is not part of the
hash, but declaration order and serialization behavior are.

## Dependency classification

Dependencies on leaf configuration DTOs are accepted when a domain consumes
typed options without owning configuration loading or hashing. Examples are:

- brain algorithms consuming `BrainConfig`, `LearningConfig`, and action
  distribution DTOs;
- observation schema/providers consuming feature, normalization, and rolling
  observation DTOs;
- execution, learning, evolution, replacement, and reward algorithms consuming
  their respective DTOs.

Root `Config` dependencies require stronger justification. Current production
uses are classified as follows:

| Consumer | Reason | Status |
| --- | --- | --- |
| simulation engine/checkpoint/evaluation | Composition root and authoritative run state | Acceptable |
| observation compatibility adapter | Preserves public `from_config` APIs | Isolated compatibility debt |
| market data source and feature pipeline | Existing public loading/construction contract | Candidate for later narrowing |
| agent construction and blueprint selection | Cross-domain composition input | Candidate for later narrowing |
| replacement context | Existing strategy contract exposes full experiment configuration | High-value later narrowing |
| `DatasetSplit::range` | Convenience lookup into root split configuration | Low-priority cleanup |

Observation schema compilation and built-in builder construction do not consume
root `Config`. Configuration creates private `ObservationSchemaOptions` and
`ObservationBuilderOptions` views; core observation code consumes those views.
The public whole-config entry points delegate through
`observation/config_compat.rs` only.

## Construction dependencies

Ordered configuration validation compiles an observation schema at the
established point in validation. This protects schema/provider consistency and
must not be reordered.

Backend runtime construction is brain-owned. Configuration maps
`BackendConfig` into a private `ComputeDeviceOptions` contract, and brain
constructs and validates `ComputeDevice`. Configuration validation, simulation
setup, and checkpoint resume use that constructor directly. The established
public `BackendConfig::compute_device` API delegates through the isolated
`config/backend_compat.rs` adapter.

`BrainArchitecture::from_config` already accepts `BrainConfig` plus explicit
schema hash, observation size, and asset count. It is a narrow domain-owned
compiler and does not require root `Config`.

## Visibility audit

- Configuration child modules remain private.
- Established DTOs remain publicly re-exported from `midas_core::config`.
- Domain-local validators use `pub(super)`.
- Observation option views and `Config` conversion helpers use `pub(crate)`
  because configuration, simulation, checkpoint, and observation modules share
  them internally.
- Observation compatibility methods remain public because they are established
  API.
- Brain device option views and construction use `pub(crate)` for configuration
  mapping and composition-root callers. The configuration compatibility method
  remains public because it is established API.

No visibility can be narrowed further without breaking existing cross-module
or public contracts. No new public configuration items are needed.

## Duplication and ownership

The observation compatibility adapter delegates to the same narrow compiler
used by production callers; it does not duplicate schema or provider logic.
Domain validation has one implementation per DTO group, and the coordinator
only retains mixed simulation/execution checks and observation schema/provider
cross-checks.

Root `Config` remains owned by `SimulationEngine` because checkpoints, reports,
replacement, and runtime construction require a stable experiment definition.
Changing that ownership would be a simulation contract redesign, not
configuration cleanup.

## Next priorities

1. Narrow `ReplacementContext` to explicit evolution, replacement, brain, and
   learning inputs before adding replacement strategies.
2. Evaluate narrow market loading and feature-pipeline option views before
   adding data providers or feature configuration.
3. Consider narrowing agent construction inputs only alongside a dedicated
   agent-factory compatibility plan.

Every stage must keep the configuration regression snapshots, version-12
checkpoint comparisons, schema and architecture identities, deterministic
reports, and benchmark output unchanged.
