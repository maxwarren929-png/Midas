# V3 Phase 1 — Frozen-Encoder Burn Autodiff

Phase 1 replaces manual dense actor/critic gradient accumulation with Burn's
`Autodiff<NdArray>` computation graph. The inference backend, architecture,
PPO objective, momentum update rule, optimizer state, and checkpoint format are
unchanged. Hidden and recurrent parameters remain frozen.

## Objective boundary

Burn 0.18 does not provide differentiable lgamma/digamma tensor operations.
The bounded-Beta implementation therefore continues to calculate its exact
log-probability and entropy derivatives at the actor logits. Those derivatives
become the upstream vector for a scalar surrogate objective; Burn backpropagates
that vector through the actor and critic dense heads. This is the same exact
vector-Jacobian product previously accumulated by hand and is suitable for
extending the graph through the encoder in Phase 2.

## Baseline comparison

Both runs use `configs/prompt6/combined-ppo.toml`, seed 41, eight agents, the
training split, and a 127-step boundary in the Cargo debug profile.

| Measurement | Manual baseline | Phase 1 autodiff |
|---|---:|---:|
| PPO updates | 248 | 248 |
| Neural actions | 1,016 | 1,016 |
| Deaths | 0 | 0 |
| Invalid actions | 0 | 0 |
| Mean ending net worth | 10,130.2926 | 10,130.2926 |
| Timed runtime | 5.943 s | 7.423 s |
| Sampled peak RSS | 43.59 MiB | 45.39 MiB |
| Checkpoint size | 879,409 bytes | 879,409 bytes |

Seven per-agent reports are bit-identical. Agent 6 ending net worth changes
from `10046.55521063353` to `10046.555210631692`, an absolute difference of
approximately `1.8e-9`. This is within the Phase 1 floating-point tolerance.

Two independent Phase 1 runs produced the same checkpoint SHA-256:

`2bdbabff271f1cc44a272cee125bb7439ca05ab50015ae81f5daa47236b170c1`

The verified checkpoint is saved at
`checkpoints/baselines/v3-phase1-autodiff-seed-41-step127.bin`.

The original manual-gradient checkpoint remains preserved at
`checkpoints/baselines/frozen-combined-ppo-seed-41-step127.bin`.

## Full-split validation gate

The complete 1,259-step training split was then run with the same seed-41
configuration. It completed 2,522 PPO updates and 10,072 neural actions, with
15 births, seven deaths, and zero invalid actions. In the Cargo debug profile,
the run took 83.086 seconds and reached a sampled peak RSS of 71,180 KiB
(69.51 MiB). Its final checkpoint is 4,977,651 bytes.

Resuming the periodic step-1,152 checkpoint reproduced every final report
field from the uninterrupted run. A second independent full run produced a
byte-identical final checkpoint. Both final checkpoints have SHA-256:

`ce33d4163b6df6757acf37d4b4f57f8e85521dc975c98fcd92b86743a09fa374`

The retained checkpoints are:

- `checkpoints/baselines/v3-phase1-full-seed-41.bin`
- `checkpoints/baselines/v3-phase1-full-seed-41-replay.bin`
- `checkpoints/baselines/v3-phase1-full-seed-41-auto.bin` (step 1,152)

The historical manual-gradient full report at
`reports/prompt6/combined-ppo-seed-41.txt` has the same steps, PPO updates,
actions, births, deaths, invalid-action count, and lifecycle. Its largest
per-agent ending-net-worth difference is approximately `6.78e-5` (agent 5).
Its `active_lineages = 15` value uses the older all-recorded-lineages meaning;
the current report's value of eight counts distinct lineages in the living
population, a later reporting correction unrelated to autodiff.

This completes the Phase 1 long-horizon, deterministic-replay, and
checkpoint/resume gate. Phase 2 can proceed without changing the captured
Phase 1 reference artifacts.

## Verification

- Exact Burn actor/critic vector-Jacobian product regression passes.
- All 52 core unit tests pass.
- All 83 core integration/configuration tests pass.
- Core doc tests and workspace `cargo check --all-targets` pass.
- The workspace-wide test command has one unrelated local-tree failure: ignored
  generated `configs/prompt9/runs/**` files are discovered by the CLI shipped
  config manifest test. Those experiment artifacts were preserved.
