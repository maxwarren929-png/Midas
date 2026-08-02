# Prompt 9 Screening Results

Screening used the frozen protocol at `reports/PROMPT9_EXPERIMENT_PROTOCOL.md`:
24 development markets, eight regime families, three walk-forward folds, seeds
41-45, seven frozen conditions, and validation only.

## Completion and Reliability

- Planned tasks: 3,240 (720 training, 2,520 evaluation).
- Completed evaluations: 2,520.
- Unresolved failures: 0.
- Six benchmark processes initially returned exit code 1 with empty stderr and
  succeeded unchanged on resumable retry. All attempts remained observable in
  runner output; the final manifest contains one valid row per planned task.
- The interrupted-run audit reconstructed all 3,240 manifest rows from valid
  outputs without changing completed raw result bytes.
- No return was inspected until all screening tasks completed and aggregation ran.

## Aggregate Validation Results

| Condition | Mean return | Median return | Profitable | Beat cash | Beat equal weight | Beat momentum | Mean drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|
| Combined PPO | -0.9174% | -0.9157% | 12.78% | 56.11% | 42.78% | 72.22% | 1.5452% |
| Combined PPO + evolution | -0.9255% | -0.9515% | 13.61% | 54.44% | 42.78% | 72.22% | 1.5490% |
| Cash | -0.9950% | -0.9950% | 0.00% | 0.00% | 43.06% | 72.22% | 0.9950% |
| Equal weight | +2.2415% | -0.1426% | 48.61% | 56.94% | 0.00% | 73.61% | 9.1749% |
| Buy and hold | +2.0108% | -1.6830% | 43.06% | 47.22% | 50.00% | 70.83% | 11.9859% |
| Momentum | -1.5346% | -5.5531% | 27.78% | 27.78% | 26.39% | 0.00% | 12.3691% |
| Random | -5.2386% | -6.2038% | 10.28% | 12.22% | 5.28% | 38.89% | 9.7978% |

Returns include configured survival and transaction costs. Neural results are
run-level means across the eight frozen evaluation agents.

## Paired Screening Comparisons

| Strategy | Comparator | Mean difference | Median difference | Wins / losses / ties |
|---|---|---:|---:|---:|
| Combined PPO | Cash | +0.0776% | +0.0793% | 202 / 158 / 0 |
| Combined PPO | Equal weight | -3.1588% | -0.7466% | 154 / 206 / 0 |
| Combined PPO | Momentum | +0.6173% | +4.8141% | 260 / 100 / 0 |
| PPO + evolution | PPO | -0.0081% | 0.0000% | 153 / 162 / 45 |
| PPO + evolution | Cash | +0.0695% | +0.0435% | 196 / 164 / 0 |
| PPO + evolution | Equal weight | -3.1670% | -0.7409% | 154 / 206 / 0 |
| PPO + evolution | Momentum | +0.6092% | +4.6983% | 260 / 100 / 0 |

## Screening Interpretation

The validation evidence does not establish profitability. Median neural returns
are negative in all eight regime families, fewer than 14% of neural runs are
profitable, and neither neural candidate beats equal weight in a majority of
paired runs. Momentum performs poorly after its high turnover and configured
costs; beating it is not sufficient evidence of an edge. Evolution does not
improve PPO materially in this screening sample.

This result causes no configuration, seed, market, selection, or protocol change.
Both preregistered neural candidates proceed unchanged to confirmatory test
evaluation, as required by the frozen protocol.
