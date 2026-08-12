# V3 Phase 3Q — Turnover-Constrained PPO

Phase 3Q implements the turnover-first constrained objective selected by the
Phase 3P gradient audit. The implementation passed its mechanical gate, but
both two-seed calibrations failed the economic promotion gate. Constrained PPO
therefore remains opt-in and is not advanced to five seeds.

## Implementation

An optional top-level `turnover_constraint` configuration enables:

- a second critic output trained only on discounted normalized-turnover cost;
- a cost advantage formed from that return and its own critic baseline, then
  normalized independently of reward advantage;
- actor advantage `reward_advantage - multiplier * turnover_cost_advantage`;
- projected dual ascent after a successfully committed rollout update;
- a non-negative bounded multiplier serialized in a distinct constrained-PPO
  learning state under checkpoint format 23; and
- report telemetry for cost-critic loss/gradient, mean rollout cost, and the
  active multiplier.

Turnover cost is always daily traded notional divided by previous net worth,
independent of the scalar reward kind. Drawdown remains diagnostic-only.
Ordinary PPO retains a one-output critic and unchanged config/checkpoint bytes.
Two-output genome initialization preserves the exact hidden, GRU, actor, and
reward-critic random streams and appends only the new cost-critic column.

Mechanical regressions prove that the cost head receives gradients, the dual
changes the actor gradient, dual projection occurs only at commit, constrained
state round-trips, and legacy version-12/version-22 checkpoints still migrate.
The full workspace has 164 passing tests; strict Clippy, formatting, and diff
checks pass.

## Two-seed calibration

Both candidates used the Phase 3L full-horizon protocol on training and
validation only:

```text
population=10 steps=1259 seeds=41,42 seed_derivation=splitmix_v2
optimizer=momentum learning_rate=3e-4 gradient_clip=1.0
reward=log_return drawdown_penalty=0 turnover_penalty=0
cost_critic_coefficient=0.5 initial_multiplier=0 maximum_multiplier=10
```

The first candidate used a 7% daily budget and dual learning rate 0.1. Its
mean multiplier was only 0.091 across PPO substeps; training turnover rose
0.86% over the paired unconstrained runs. Validation turnover rose in both
seeds.

The tighter calibration used a 5% budget and dual learning rate 1.0. Its mean
multiplier increased to 3.01 and seed 42 training turnover fell 0.76%, proving
the controller was active, but seed 41 turnover rose 3.16%. Validation again
failed to reduce turnover in either seed and return/drawdown deteriorated.

| Validation, mean across seeds | Unconstrained PPO | 7% / 0.1 | 5% / 1.0 |
| --- | ---: | ---: | ---: |
| Mean return | -2.404% | -1.803% | -3.241% |
| Median-agent return | -1.980% | -1.906% | -3.148% |
| Maximum drawdown | 4.321% | 4.156% | 5.040% |
| Population turnover | 27,433.98 | 33,969.30 | 32,353.33 |
| Fees | 13.72 | 16.98 | 16.18 |

The looser candidate improved return and drawdown over the two-seed
unconstrained mean, but it missed the primary turnover objective by 23.8%.
The tighter candidate missed turnover by 17.9% and materially harmed return
and drawdown. Neither passes a turnover-constrained promotion gate.

## Decision

Keep the implementation as research infrastructure, with no default config and
no five-seed promotion. More multiplier pressure is not supported: the tighter
run already produced a large cost coefficient without out-of-sample turnover
control. The likely next investigation is action/execution-level turnover
control (for example, a no-trade region or explicit trade-rate cap), where the
desired behavior is enforced directly instead of learned through a noisy
long-horizon policy gradient.

Raw summaries and checkpoints are in `reports/phase3q/`. The test split was not
used.
