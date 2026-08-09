# V3 Phase 3I — AdamW Learning-Rate Calibration

Phase 3I calibrates AdamW against the retained Phase 3G momentum update scale.
It is a mechanical scale comparison, not a profitability or optimizer-adoption
decision.

## Protocol

All candidates used the same controlled run:

```text
config=configs/prompt6/combined-ppo.toml
split=training steps=127 population=10 seed=41
seed_derivation=splitmix_v2 gradient_clip=1.0
optimizer=adamw weight_decay=0.01
```

Candidates were run two processes at a time. The diagnostics CLI gained a
`--learning-rate` override that fixes both the root PPO rate and the genome
learning-rate trait. An initial pair was rejected and overwritten after it
revealed that changing only the root rate was silently superseded by the
genome trait; both invalid traces had exactly reproduced the old `0.0003`
AdamW trace. A regression now locks both layers of the override.

## Results

| Optimizer / rate | Mean relative update | vs momentum | Mean return | Median return | Mean max drawdown | Turnover | Fees |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Momentum `3e-4` | `1.47088e-5` | baseline | 1.881% | 1.778% | 1.527% | 1,232,312.55 | 616.16 |
| AdamW `3e-6` | `1.23529e-5` | -16.0% | 2.009% | 1.640% | 1.625% | 1,235,369.53 | 617.68 |
| AdamW `3.5e-6` | `1.43999e-5` | -2.1% | 2.003% | 1.623% | 1.629% | 1,235,459.53 | 617.73 |
| AdamW `4e-6` | `1.64122e-5` | +11.6% | 2.042% | 1.623% | 1.640% | 1,235,038.09 | 617.52 |
| AdamW `1e-5` | `4.10872e-5` | +179.3% | 2.002% | 1.677% | 1.550% | 1,229,041.09 | 614.52 |

Every candidate completed 310 PPO updates with zero invalid actions, policy
failures, deaths, or policy-ratio clipping. The selected `3.5e-6` run had mean
absolute approximate KL below `6.5e-7` and 66.94% gradient clipping.

Matching the global update norm does not make the optimizers locally
equivalent. At `3.5e-6`, AdamW produced hidden, recurrent, actor, and critic
parameter deltas of `0.0968`, `0.0377`, `0.0125`, and `0.00762`; momentum
produced `0.0545`, `0.0401`, `0.0778`, and `0.000526`. AdamW's adaptive
preconditioning redistributes movement substantially across parameter groups.

## Decision

Retain momentum as the default. Use `3.5e-6` as the calibrated AdamW candidate
for the next paired multi-seed optimizer comparison because its global update
scale is within 2.1% of momentum. Do not interpret the small single-seed return
differences as optimizer evidence.

The selected checkpoint SHA-256 is
`4380a938f13c0dae382a23bb87ae92d280686b977912733e98b2bad0fe8d497d` and its
trace SHA-256 is
`59e3efb8abf91d253f608b66091dd236a9355cd636440439c361b5d1066ae4be`.
