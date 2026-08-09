# V3 Phase 3G — Versioned Seed Derivation

Phase 3G replaces ad hoc XOR seed construction with one versioned,
domain-separated derivation API. Existing configurations remain on
`legacy_xor`; new experiments may opt into `splitmix_v2` through
`simulation.seed_derivation` or the diagnostics override.

## Compatibility contract

- An omitted `simulation.seed_derivation` selects `legacy_xor` and is omitted
  again during serialization. The four retained configuration JSON/hash
  snapshots are unchanged.
- Legacy formulas for initial genomes, policies, PPO learning, evolution, and
  replacement substreams are preserved exactly.
- An explicit `splitmix_v2` value is serialized and therefore changes the
  configuration hash. A V2 checkpoint cannot be resumed accidentally with a
  legacy configuration, or vice versa.
- At the Phase 3G boundary, checkpoint formats remained 19 for the baseline
  observation schema and 20 for awareness. Phase 3H subsequently introduced
  AdamW state and raised current formats to 21/22 while retaining 12–20 reads.

V2 derives independent streams for initial genomes, policies, PPO learning,
evolution, replacement entropy, replacement genomes, selection, crossover,
mutation, evaluation policies, and evaluation evolution. Its mixing sequence
uses stable domain tags and SplitMix64 finalization over the root, subject, and
optional entropy instead of raw XOR composition.

## Regression evidence

- Identical inputs derive identical values, while changing the domain or agent
  changes the result.
- Initial-genome seed sets for adjacent roots 41–45 and agent IDs 0–9 have zero
  pairwise overlap. This directly covers the Phase 3F alias failure.
- A V2 partial run resumed from checkpoint matches its uninterrupted run in
  domain history and final agent state.
- The complete core suite passes 148 tests, including unchanged legacy config
  hashes and version-12 checkpoint migration/evaluation coverage.

## Recaptured baseline

Two identical runs were launched concurrently with:

```text
config=configs/prompt6/combined-ppo.toml
split=training steps=127 population=10 seed=41
seed_derivation=splitmix_v2 gradient_clip=1.0
```

Both completed with zero invalid actions and zero deaths. Their authoritative
checkpoints are byte-identical at SHA-256
`d4cf38b5b62f44e7ac1c8576ab3d8f9e8898fc24713a10ced5bf16425ecc15ad`.
Their JSONL traces are also byte-identical at SHA-256
`d014e36921112d08fbb5a039fd249ff0437a10c86ae42df225b9cfab3cd33bb9`.
The summaries differ only in measured process resources.

| Measurement | SplitMix V2 baseline |
| --- | ---: |
| Mean return | 1.881% |
| Median return | 1.778% |
| Ending population value | 101,881.38 |
| Total turnover | 1,232,312.55 |
| Total fees | 616.16 |
| Mean maximum drawdown | 1.527% |
| PPO updates / optimizer steps | 310 / 2,480 |
| Gradient clipping frequency | 66.77% |
| Mean pre-clip global norm | 1.491 |
| Mean relative update | 0.00001471 |
| Peak RSS, concurrent runs | 46,952 / 47,248 KiB |
| Wall time, concurrent runs | 38.03 / 38.00 s |

The prior legacy seed-41 clip-1 run returned 2.362% on average. The V2 result
is not directly comparable as an optimizer outcome because V2 intentionally
creates a different population and stochastic stream. This recapture is the
new reference artifact for optimizer experiments.

## Decision

Adopt `splitmix_v2` for new controlled experiments while retaining
`legacy_xor` as the compatibility default. The next isolated experiment can
now compare momentum against AdamW using paired V2 seeds without the adjacent
seed alias confound.
