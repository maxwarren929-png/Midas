# V3 Phase 3A — Truncated Backpropagation Through Time

Phase 3A trains the existing simple recurrent cell by unrolling Burn's autodiff
graph across each configured PPO sequence.

## Temporal boundary

- Every `PpoTrainingSample` records whether it starts a recurrent sequence.
- Sequences are capped by `learning.sequence_length`, shuffled as indivisible
  units, and packed into minibatches without splitting them.
- A sequence starts from the recurrent memory captured before its first action.
  That initial state is detached. Each later timestep consumes the preceding
  graph-produced state, so loss gradients flow backward within the sequence.
- Masked policy failures and other ineligible transitions form hard boundaries
  instead of allowing eligible samples on either side to be stitched together.
- Hidden, recurrent, actor, and critic gradients share one global clipping norm
  and the existing momentum update rule. Birth genomes remain immutable.

The current recurrence remains a dense SiLU cell. This is genuine truncated
BPTT, but it is not yet a GRU and does not carry gradients beyond the configured
sequence window.

## Checkpoints and migration

`BpttOptimizerState` persists recurrent weight and bias momentum. New neural
snapshots use `NeuralModelState::BurnBptt`; the earlier Burn variants retain
their original binary layouts. Formats 13–16 remain readable, with any missing
hidden or recurrent momentum initialized to zero. New baseline checkpoints use
format 17 and awareness checkpoints use format 18.

A retained format-15 Phase 2 checkpoint was loaded by the Phase 3A runtime and
completed a frozen validation evaluation successfully.

## Verification

- A temporal-credit regression applies loss only at timestep two. The earlier
  observation receives a hidden gradient when both steps share a sequence and
  receives none when timestep two starts a new sequence.
- A masked-transition regression proves the rollout builder emits two distinct
  recurrent fragments rather than stitching across the missing model state.
- Parallel full-simulation regressions preserve exact uninterrupted versus
  checkpoint/resumed state.

Two independent debug-profile runs used
`configs/prompt6/combined-ppo.toml`, seed 41, eight agents, and 127 training
steps. Both completed 248 PPO updates and 1,016 neural actions with zero deaths
or invalid actions. Runtime was 21.5–22.3 seconds.

Both 1,004,137-byte checkpoints have SHA-256:

`4f96ccc7ef236dfae907102baedd2bb85c71cc06072e3c39e75030d7d8156bb1`

The retained artifacts are:

- `checkpoints/baselines/v3-bptt-seed-41-step127.bin`
- `checkpoints/baselines/v3-bptt-seed-41-step127-replay.bin`

Phase 3B subsequently replaced the recurrence for new genomes with a standard
GRU. This report intentionally preserves the measured Phase 3A boundary; see
`reports/V3_PHASE3B_GRU.md` for the replacement and compatibility results.
