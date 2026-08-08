# V3 Phase 3B — Standard GRU Replacement

Phase 3B replaces the dense-SiLU recurrent cell for new agents with a standard
gated recurrent unit and keeps the complete cell inside Burn's truncated
autodiff graph.

## Cell and parameter layout

For hidden input `x`, previous memory `h`, update gate `z`, reset gate `r`, and
candidate `n`, inference and training both compute:

```text
z = sigmoid(xW_xz + hW_hz + b_z)
r = sigmoid(xW_xr + hW_hr + b_r)
n = tanh(xW_xn + (r * h)W_hn + b_n)
h' = z * h + (1 - z) * n
```

Genome format 4 packs update, reset, and candidate columns into the recurrent
`DenseGenome`. Keeping this evolvable container means mutation, crossover,
genetic distance, lifetime momentum, and checkpoint persistence cover every
gate without adding a parallel parameter system.

## Compatibility

Legacy genome format 3 still executes its original dense-SiLU recurrence.
Checkpoint formats 12–18 remain readable, while new baseline and awareness
checkpoints use formats 19 and 20. Replacement genomes inherit the dead
agent's layout, which keeps resumed legacy populations homogeneous. Crossover
and genetic-distance operations reject mixed layouts explicitly.

A retained format-17 Phase 3A checkpoint completed a 269-step validation
evaluation with zero invalid actions under the new binary.

## Verification

- A hand-calculated regression matches GRU inference state exactly.
- Packed Burn gradients match finite differences for update, reset,
  candidate-input, and candidate-hidden weights.
- Parallel checkpoint/resume integration remains exact for new GRU genomes.
- The focused brain and evolution suites pass.

Two independent debug-profile runs used
`configs/prompt6/combined-ppo.toml`, seed 41, eight agents, and 127 training
steps. Both completed 248 PPO updates and 1,016 neural actions with zero deaths
or invalid actions. Each run took approximately 22 seconds.

Both 1,055,827-byte format-19 checkpoints have SHA-256:

`2175abd6ad8b1a04a1d7a7e5727444574324108d97159ecf446a7f3b8ecf3a78`

The retained artifacts are:

- `checkpoints/baselines/v3-gru-seed-41-step127.bin`
- `checkpoints/baselines/v3-gru-seed-41-step127-replay.bin`
