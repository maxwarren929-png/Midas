# V3 Phase 3H — AdamW Implementation

Phase 3H adds a fully checkpointed AdamW optimizer for every trainable neural
parameter group: hidden encoder layers, packed GRU, actor, and critic.
Momentum remains the compatibility default; AdamW is an explicit opt-in.

## Optimizer contract

AdamW applies the globally clipped PPO gradient using:

- first-moment coefficient `beta1 = 0.9`;
- second-moment coefficient `beta2 = 0.999`;
- denominator epsilon `1e-8`;
- decoupled weight decay `0.01` on matrix weights only;
- no decay on biases;
- step-wise bias correction for both moments.

Configuration is authoritative through the optional top-level `[optimizer]`
section. Omitting it preserves the exact prior configuration JSON/hash and
momentum behavior. New AdamW checkpoints serialize complete first and second
moments in a distinct neural-state variant. Baseline checkpoint format is 21
and awareness format is 22; formats 12–20 remain readable.

```toml
[optimizer]
kind = "adamw"
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
weight_decay = 0.01
```

## Regression evidence

- A numerical first-step regression reconstructs AdamW updates from the saved
  moments, including bias correction, weight decay, and the absence of bias
  decay.
- Malformed second-moment shapes are rejected during restore.
- An AdamW checkpoint/resume run matches its uninterrupted domain history and
  final agent state.
- Legacy configuration snapshots remain exact. A new momentum checkpoint is
  byte-identical to the Phase 3G checkpoint after excluding only the four-byte
  format version, and its trace hash is unchanged.

## Controlled optimizer comparison

Momentum and AdamW were run concurrently with the same Phase 3G protocol:

```text
config=configs/prompt6/combined-ppo.toml
split=training steps=127 population=10 seed=41
seed_derivation=splitmix_v2 gradient_clip=1.0
learning_rate=0.0003
```

| Measurement | Momentum | AdamW |
| --- | ---: | ---: |
| Mean return | 1.881% | 1.986% |
| Median return | 1.778% | 1.762% |
| Ending population value | 101,881.38 | 101,986.13 |
| Total turnover | 1,232,312.55 | 1,161,814.12 |
| Total fees | 616.16 | 580.91 |
| Mean maximum drawdown | 1.527% | 1.740% |
| Gradient clipping frequency | 66.77% | 70.44% |
| Mean pre-clip global norm | 1.491 | 1.536 |
| Mean relative update | 0.00001471 | 0.00126435 |
| Mean approximate KL | -0.0000152 | -0.0006294 |
| Hidden parameter delta | 0.0545 | 8.2914 |
| Recurrent parameter delta | 0.0401 | 3.3097 |
| Actor parameter delta | 0.0778 | 1.2221 |
| Critic parameter delta | 0.000526 | 0.3203 |
| Peak RSS | 46,468 KiB | 47,268 KiB |
| Checkpoint size | 1,316,693 B | 1,481,401 B |
| Invalid actions / deaths | 0 / 0 | 0 / 0 |

The momentum trace retains SHA-256
`d014e36921112d08fbb5a039fd249ff0437a10c86ae42df225b9cfab3cd33bb9`.
The AdamW trace SHA-256 is
`a9ca1fe90895f43e4f6433cace6e250c4b20222aa9880821a63c0201051ce04a`.

## Decision

AdamW is mechanically correct and deterministic, but the inherited learning
rate is not a calibrated comparison. Its mean relative update is about 86
times the momentum baseline, and its parameter deltas and drawdown are much
larger. Retain momentum as the default. The next experiment should calibrate a
lower AdamW learning rate against the Phase 3G update scale before running a
multi-seed optimizer decision.
