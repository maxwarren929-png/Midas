# V3 Phase 3C — Gradient Health Telemetry

Phase 3C measures whether PPO learning signals reach each trainable component
before changing the optimizer itself.

## Measurements

Every report-only `PpoUpdateTrace` now records:

- global gradient norm before and after clipping;
- one norm for each feed-forward hidden layer;
- the complete recurrent gradient norm;
- separate update-gate, reset-gate, and candidate GRU norms;
- actor and critic gradient norms;
- the fraction of internal optimizer minibatches that triggered global
  gradient clipping; and
- the L2 norm of the actual momentum update divided by the L2 norm of all
  lifetime parameters.

Component norms are measured before global clipping. Relative update size is
measured from the clipped, momentum-filtered update that is actually applied.
Multi-epoch and minibatch results are averaged deterministically. Legacy
dense-recurrent agents report no GRU gate values.

These fields live only in training metrics and report traces. Report traces are
not authoritative simulation state and are excluded from checkpoints, so the
format remains baseline 19 / awareness 20 and deterministic resume behavior is
unchanged.

## Initial smoke

`midas-diagnostics --config configs/prompt2.toml --split training` completed a
seven-step synthetic run with three agents and six PPO updates. It reported:

| Measurement | Mean |
| --- | ---: |
| Global norm before clipping | 1.03692 |
| Global norm after clipping | 0.80798 |
| Gradient clipping frequency | 41.67% |
| Hidden layer norms | 0.16952, 0.19990 |
| Complete recurrent norm | 0.44222 |
| GRU update-gate norm | 0.00253 |
| GRU reset-gate norm | 0.00138 |
| GRU candidate norm | 0.44221 |
| Actor norm | 0.89225 |
| Critic norm | 0.00263 |
| Relative parameter update | 0.0000120 |

The candidate path dominated both gates in this tiny run. The clipping rate is
also high enough to warrant measurement on the 127-step retained protocol.
Neither observation alone justifies an optimizer change: the run is deliberately
small and synthetic.

## Decision gate

Run the retained 127-step configuration across several seeds and compare:

1. gate norms over time, especially whether update/reset remain near zero;
2. clipping frequency and the pre/post-clip gap;
3. relative update magnitude by training stage; and
4. PPO KL, policy clipping, and performance diagnostics alongside gradient
   health.

Only then choose between learning-rate/clip tuning and an AdamW experiment.
