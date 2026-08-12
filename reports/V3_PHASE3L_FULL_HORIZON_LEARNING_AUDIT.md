# V3 Phase 3L — Full-Horizon Learning Value Audit

Phase 3L tests whether full-horizon PPO adds value over the same networks before
lifetime learning and over simple policies. Development uses validation only;
the already-inspected test split is not reused.

## Protocol

Five momentum populations trained across the complete training interval:

```text
config=configs/prompt6/combined-ppo.toml
split=training steps=1259 population=10 seeds=41,42,43,44,45
seed_derivation=splitmix_v2 gradient_clip=1.0
optimizer=momentum learning_rate=3e-4
```

The runs completed 15,768 PPO updates. They also recorded 38 deaths and random
non-evolutionary replacements, leaving ten active agents per checkpoint. This
made a simple comparison with the original seed-zero population causally
invalid: the final population did not always contain the original genomes.

The authoritative control therefore evaluates each final checkpoint twice on
validation:

1. with its trained lifetime hidden, GRU, actor, and critic parameters;
2. with those lifetime parameters reset to the same final agent's immutable
   genome, preserving replacement and survivorship outcomes.

Both paths reset portfolios and recurrent state and disable learning,
replacement, and evolution. A regression verifies that every baseline tensor
equals its checkpoint genome. The diagnostics CLI now exposes this as
`--genome-baseline`.

## Matched learning value

| Measurement | Final-genome baseline | PPO-trained lifetime | PPO delta |
| --- | ---: | ---: | ---: |
| Across-seed mean return | -2.108% | -2.158% | -0.050 pp |
| Mean within-seed median return | -1.852% | -1.546% | +0.306 pp |
| Mean maximum drawdown | 3.730% | 4.534% | +0.804 pp |
| Mean turnover | 23,064.98 | 28,279.06 | +22.6% |
| Mean fees | 11.53 | 14.14 | +22.6% |

PPO won paired mean return in two of five seeds, median-agent return in three of
five, and drawdown in zero of five. Its mean-return effect was essentially
flat, but it consistently increased risk and trading. The initially captured
seed-zero control averaged -1.316%; it is retained only as supplementary
population evidence and is not used for the learning-effect decision because
of training-time replacement.

## Complete-horizon policy benchmarks

Benchmarks ran for all 269 validation steps with the bankruptcy threshold set
to zero, preventing replacement from truncating or rebasing policy histories.
Deterministic policies repeat across seeds; random uses all five streams.

| Policy | Mean return | Mean max drawdown | Mean turnover | Mean fees |
| --- | ---: | ---: | ---: | ---: |
| Cash | -1.345% | 1.345% | 0.00 | 0.00 |
| Equal weight | -3.143% | 8.654% | 234,489.68 | 117.24 |
| Buy and hold | 33.436% | 22.819% | 99,880.14 | 49.94 |
| Random | -17.187% | 17.456% | 12,506,145.08 | 6,253.07 |
| 20-day momentum | 75.624% | 7.627% | 5,177,002.72 | 2,588.50 |
| PPO-trained neural | -2.158% | 4.534% | 28,279.06 | 14.14 |

The benchmark winner is costly and regime-specific, so its return is not a
direct target for neural imitation. It nevertheless proves that strong
point-in-time signal exists in this validation regime while the learned policy
does not exploit it.

## Diagnosis and decision

More training does not currently create robust learning value. PPO slightly
improves the typical final agent but fails to improve population mean return,
raises turnover, and worsens drawdown in every matched seed.

The immediate problem is objective alignment, not optimizer choice or model
capacity. The experiment configuration rewards pure log return while setting
both `drawdown_penalty_weight` and `turnover_penalty_weight` to zero. The
observed behavior is consistent with that omission: PPO is not directly
penalized for the two dimensions that deteriorate most.

Momentum remains the default, and architecture growth is deferred. The next
isolated phase should compare small, predeclared reward penalties using the
matched-genome validation control. The test split must remain out of that
selection loop.
