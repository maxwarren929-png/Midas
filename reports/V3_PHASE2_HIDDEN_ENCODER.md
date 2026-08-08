# V3 Phase 2 — Trainable Hidden Encoder

Phase 2 extends the Burn autodiff graph from raw observations through every
configured feed-forward hidden layer, the fixed recurrent projection, and both
actor and critic heads.

## Training boundary

- Hidden weights and biases, actor parameters, and critic parameters are
  trainable.
- Recurrent weights, recurrent bias, sampled recurrent states, and birth
  genomes remain frozen.
- The recurrent projection is part of the graph, so its fixed Jacobian carries
  actor and critic gradients back into the hidden encoder.
- The bounded-Beta log-probability and entropy derivatives remain the exact
  upstream boundary because Burn 0.18 lacks differentiable lgamma/digamma
  tensor operations.
- Global gradient clipping covers hidden and head gradients together. All
  trainable dense layers use the existing momentum update rule.

## Checkpoints and determinism

Hidden momentum is stored in a new `TrainableEncoderOptimizerState`. The old
`NeuralModelState::Burn` layout is retained for migration and a new
`BurnTrainableEncoder` variant is written. Formats 13 and 14 remain readable;
their hidden momentum starts at zero. New baseline checkpoints use format 15
and awareness checkpoints use format 16.

Concurrent NdArray autodiff graphs in one process produced small
schedule-dependent differences during initial testing. Graph
construction/backward is therefore guarded by a process-local mutex. Inference
is not locked. Two consecutive parallel integration runs then reproduced
checkpoint/resume state exactly.

## 127-step smoke gate

Two independent debug-profile runs used
`configs/prompt6/combined-ppo.toml`, seed 41, eight agents, and 127 training
steps. Both completed 248 PPO updates and 1,016 neural actions with zero deaths
or invalid actions. Runtime was approximately 13.7 seconds per run.

Both 996,825-byte checkpoints have SHA-256:

`8c3fc8f6ec13f41dca2cfe4a3c9762cb8e6eb1b9a3ead9c1c5681ecd142a9f8a`

The retained artifacts are:

- `checkpoints/baselines/v3-phase2-hidden-seed-41-step127.bin`
- `checkpoints/baselines/v3-phase2-hidden-seed-41-step127-replay.bin`

Unit and integration regressions prove hidden parameters change, recurrent and
genome parameters do not, malformed hidden optimizer state is rejected, legacy
frozen-encoder state migrates, and checkpoint/resume remains exact.

This report captures the Phase 2 boundary. Phase 3A subsequently made the
simple recurrent parameters trainable through truncated BPTT; see
`reports/V3_PHASE3A_BPTT.md`.
