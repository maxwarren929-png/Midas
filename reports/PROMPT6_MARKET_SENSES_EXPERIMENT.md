# Prompt 6 Market Senses Experiment

## Summary

Prompt 6 adds optional, past-only rolling market sensors through the Prompt 5
registry without changing PPO, reward, evolution, execution, incremental
trading, or the default observation vector. The default schema reproduced the
Prompt 5/P4 seed-41 control report exactly after excluding only the intentional
schema identity field: SHA-256
`3fcacff5362a9ed29121d37c7a56001191d9b1e0615b4cb4156d4e348f465577`.

On the frozen ten-seed Prompt 3B/P4 protocol, the combined senses schema
improved both validation and test mean population net worth over default PPO.
Momentum-only and volatility/trend-only schemas improved validation but lost on
test, so they are classified as overfitting for this market. This is one
synthetic market, not evidence of profitability or a general trading edge.

## Frozen Protocol

- Implementation commit: `4daef959ec07bd9413e7bc296b768e3301b40549`.
- Dataset: `data/prompt3b/market.csv`, SHA-256
  `59daa371f0f9f6a00128d99d66c8d86842f4b4bfef52df3f33105f65fe67ae28`.
- Four assets, 1,259 training, 269 validation, and 269 test steps.
- Seeds: 41--50. Costs, survival pressure, architecture, action policy,
  PPO settings, rewards, and evolution settings were inherited unchanged from
  the frozen Prompt 4 protocol.
- Training used the training split only. `evaluate` restores each checkpoint,
  resets evaluation runtime state, and freezes learning/evolution for the
  validation and test splits.

## Schemas

| Schema | Provider groups | Input dimension | Schema hash |
| --- | --- | ---: | --- |
| Default | Prompt 5 market, portfolio, agent state | 54 | `95af73d3a1e48b89c35af1b1da1eb1b9786fcb414121b003fe2043a4506b091b` |
| Momentum | return 20/60, relative strength 20/60 | 70 | `29e0e95ff637b1230ea91ddd54f1780e33f48bb09a018ede20ff7677b149069e` |
| Volatility/trend | volatility 20/60, SMA 60, summaries | 70 | `c28b97a7eff573bff766ca0c65d2eb1121c0e73fad088c71a717c514f42ec894` |
| Combined | return 5/20/60/120, volatility 20/60, SMA 20/60, relative strength 20/60, summaries | 98 | `ceca739b4fac89b487a2a872e5d973632fb6e4b4e605e84d34cf8e17d64a316e` |

All new raw values were finite in provider tests and runtime simulations.
Insufficient history uses deterministic `0.0` zero fill; the first available
lookback is never backfilled from future data. The combined schema has 44
additional inputs; its planned early-lookback zero-fill count over all 1,800
dates is 1,852 of 79,200 raw rolling values (2.34%), excluding incidental real
zeros. Schema normalisation is applied only after raw publication.

## Results

| Condition | Seeds | Train NW | Validation NW | Test NW | Test DD | Test turnover | Test costs | PPO updates | Deaths | Births |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Default control | 10 | 10213.76 | 9871.60 | 9858.22 | 0.0192 | 1729.52 | 0.86 | 0.0 | 3.9 | 11.9 |
| Default PPO | 10 | 10485.86 | 9824.46 | 9905.10 | 0.0198 | 2506.41 | 1.25 | 2522.8 | 7.4 | 15.4 |
| Momentum PPO | 10 | 10935.10 | 9873.27 | 9819.41 | 0.0254 | 1860.92 | 0.93 | 2522.7 | 5.3 | 13.3 |
| Volatility/trend PPO | 10 | 10590.73 | 9886.22 | 9863.22 | 0.0228 | 1797.88 | 0.90 | 2523.8 | 7.2 | 15.2 |
| Combined PPO | 10 | 10323.04 | 9902.97 | 9989.73 | 0.0208 | 2137.05 | 1.07 | 2523.1 | 8.1 | 16.1 |
| Combined PPO + evolution | 10 | 10328.26 | 9868.06 | 10021.44 | 0.0194 | 2427.73 | 1.21 | 2523.4 | 8.2 | 16.2 |

The combined PPO schema beat default PPO on validation in 7/10 paired seeds
and on test in 7/10, with mean test net worth higher by 84.63. The
combined-plus-evolution condition was higher by 116.34 on test and also won
7/10 paired seeds, but this experiment does not isolate a causal evolutionary
improvement. Momentum-only won 7/10 validation seeds but only 2/10 test seeds;
volatility/trend won 7/10 validation and 4/10 test seeds. Those two ablations
therefore do not meet the stated provider-help criterion.

## Benchmarks and Determinism

Test-split benchmark net worth: cash 9865.50, random 8936.06, equal-weight
9746.74, and buy-and-hold 8943.49. Combined PPO's mean test value (9989.73)
is above these single frozen-protocol benchmark results, but variance and the
single synthetic market preclude a performance claim.

The rolling seed-41 checkpoint resumed to byte-identical report JSON. The
checkpoint regression suite also verifies partial checkpoint/resume equivalence
and explicit rejection when the default and rolling schema hashes differ.

## Limitations

The provider uses raw close-derived indicators only; it supplies no news,
macro, fundamentals, or future labels. Short-window return 20 and volatility
20 intentionally overlap existing default market features in the controlled
ablations so the effect of registry-selected raw senses and cross-sectional
context can be measured; future schemas should avoid redundant entries unless
the hypothesis requires them. A second independently generated market was not
run because the existing Prompt 3B generator is fixed to its frozen output
path/seed and changing it would alter a separate generation subsystem outside
Prompt 6 scope.

Raw reports, generated per-seed configs, checkpoints, and compact CSV tables
are retained under `reports/prompt6/`, `configs/prompt6/runs/`, and
`checkpoints/prompt6/`.
