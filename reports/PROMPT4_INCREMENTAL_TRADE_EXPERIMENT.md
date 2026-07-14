# Prompt 4 Persistent Incremental Trade Experiment

## Executive Summary

The neural action interface now emits persistent incremental trade impulses rather than target portfolios. All 40 frozen Prompt 3B condition/seed runs (control, PPO, evolution-only, PPO plus evolution; seeds 41-50) completed with checkpoint format 11, and every validation/test replay completed. The action-path regression suite and checkpoint-resume comparison passed.

On this frozen synthetic market, incremental trades reduced test turnover substantially relative to the Prompt 3B target-allocation implementation. PPO test mean net worth was above the new static neural control, but results remain preliminary and do not establish profitability or generalisation.

## Commit and Dataset

Implementation revision used for the rerun: `c5dee48f33b23d89a0c46ff532ecf1d4c7cfaaa5`.
Dataset: `data/prompt3b/market.csv`, SHA-256 `59daa371f0f9f6a00128d99d66c8d86842f4b4bfef52df3f33105f65fe67ae28`, 1,800 business dates, 1,259 training steps, 269 validation steps, and 269 test steps. The market, costs, survival pressure, architecture, reward, and seeds were unchanged from Prompt 3B.

## Action Semantics

Each active asset has an independent Beta latent `u`; the PPO action is the latent vector and the executed impulse is `2u - 1`. The configured `maximum_trade_fraction = 0.05` scales the impulse by current close net worth. Cash is implicit and holdings persist. The validator masks unavailable assets and enforces the configured fraction; execution sells first, then cost-aware proportionally scales buys. Beta log probability and entropy are exact, and deterministic evaluation uses Beta means without consuming RNG or decaying temperature.

## Mechanical Results

| Condition | Seeds | Train NW | Validation NW | Test NW | Test turnover | Test costs | PPO updates | Deaths | Births |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 10 | 10213.76 | 9871.60 | 9858.22 | 1729.52 | 0.86 | 0.0 | 3.9 | 11.9 |
| ppo | 10 | 10485.86 | 9824.46 | 9905.10 | 2506.41 | 1.25 | 2522.8 | 7.4 | 15.4 |
| evolution-only | 10 | 10212.15 | 9877.84 | 9851.87 | 1528.48 | 0.76 | 0.0 | 3.7 | 11.7 |
| ppo-evolution | 10 | 10660.73 | 9844.80 | 9926.38 | 2378.06 | 1.19 | 2523.0 | 7.0 | 15.0 |

Mean test turnover fell from the Prompt 3B values of 17,461 (control), 17,681 (PPO), 17,416 (evolution-only), and 17,536 (PPO plus evolution) to 1,730, 2,506, 1,528, and 2,378 respectively. That is an approximately 90%, 86%, 91%, and 86% reduction. The new PPO policy had mean absolute impulse 0.463, mean active positions 3.61, mean concentration 0.496, and mean cash allocation 0.198 in training.

PPO beat the paired static control on 6/10 test seeds; PPO plus evolution beat it on 8/10. The mean test differences were +46.89 and +68.16 net-worth units respectively. These paired results are encouraging for the action-interface objective but are not a profitability claim, and evolution is not isolated from the changed action semantics in this comparison.

## Prompt 3B Comparison

| Condition | Prompt 3B test NW | Prompt 4 test NW | Prompt 3B turnover | Prompt 4 turnover |
| --- | ---: | ---: | ---: | ---: |
| control | 9260.26 | 9858.22 | 17461 | 1729.52 |
| PPO | 9267.50 | 9905.10 | 17681 | 2506.41 |
| evolution-only | 9258.11 | 9851.87 | 17416 | 1528.48 |
| PPO + evolution | 9274.80 | 9926.38 | 17536 | 2378.06 |

The historical and current runs use the same frozen market and protocol; the difference is the intended action-interface change plus the required checkpoint/validator compatibility updates. The large turnover reduction is therefore consistent with persistent holdings, but performance changes should not be attributed to PPO alone.

## PPO and Evolution Evidence

Seed 42 PPO diagnostics: 2521 updates, actor delta 0.248659, critic delta 0.003905, hidden delta 0.0, recurrent delta 0.0, birth genomes unchanged `True`, mean KL 7.186e-06, clip fraction 0.000.
Seed 42 PPO plus evolution diagnostics: 5 deaths, 5 reproductions, 5 crossovers, 5 mutation reports, maximum generation 1, and 13 active lineages. Birth lifetime state is built fresh from birth genomes; learned parent state is not inherited.

## Benchmarks

The test-split benchmark command produced:

```text
Cash: ending_net_worth=9865.5000
Random: ending_net_worth=8940.4721
EqualWeight: ending_net_worth=9746.7361
BuyAndHold: ending_net_worth=8943.4865
```

Buy-and-hold, equal-weight, and cash continue to use their documented benchmark semantics and are not constrained by neural concentration caps. The neural conditions exceed equal-weight and buy-and-hold in this particular test replay, while PPO is only modestly above cash; this is not evidence of robust trading skill.

## Checkpoint Determinism

For PPO seed 42, an uninterrupted run and a 600-step checkpoint followed by resume produced byte-identical JSON reports. The full test suite also compares domain histories, policy/lifetime state, optimizer state, rollouts, and evolution state. Checkpoint format 11 rejects format 10 because the new impulse metrics changed the serialized state layout; no unsafe migration is attempted.

## Verification

- `cargo fmt --all --check` passed.
- `cargo check --workspace --all-targets` passed.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings` passed.
- `cargo test --workspace --all-features` passed: 33 core unit tests, 54 foundation regressions, and 7 Prompt 2 integration tests; **94 passed, 0 failed, 0 ignored**.
- Release CLI validation, 40 training runs, 80 validation/test replays, benchmark, diagnostics, and checkpoint-resume comparison passed.

## Limitations

The action distribution factorises independent per-asset Beta impulses and retains stable asset ordering. Hidden/recurrent layers remain frozen; full recurrent BPTT, turnover/reward redesign, GPU support, and Lamarckian inheritance remain outside scope. The experiment uses one synthetic market and ten seeds. No profitability or intelligence claim is supported.

Raw run logs and checkpoints are kept under `reports/prompt4/` and `checkpoints/prompt4/`; compact tables are `condition-summary.csv`, `per-seed-results.csv`, and `run-manifest.csv`.
