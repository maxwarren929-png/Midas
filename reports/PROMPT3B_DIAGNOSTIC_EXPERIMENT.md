# Prompt 3B Diagnostic Experiment

## Executive Summary

Prompt 3B was run against implementation commit `c692101` on a frozen 1,800-session, four-asset synthetic market. Forty primary runs (four conditions, seeds 41-50) and eighteen three-seed ablation runs completed without a failed lifecycle. PPO was operational, but its current short-rollout schedule generated roughly 6,200 units of turnover per agent-day versus roughly 40 for the static control. Representative PPO execution costs were `$75,623` versus `$488` for control, and PPO raw log return was negative. This is strong evidence that turnover and execution costs are the dominant immediate mechanism behind the shortfall.

Stable stick-breaking order bias was also confirmed: the first two order positions received about 0.249 and 0.252 mean allocation, while later positions received about 0.207 and 0.133. Rotating the order made named-asset allocations more symmetric, but caused destructive daily remapping and approximately 35x higher turnover. The rotation result is a diagnostic, not a selected fix.

The selected smallest correction is the conservative PPO schedule (`rollout_length=64`, `minibatch_size=32`, `sequence_length=16`, `optimisation_epochs=2`). On the paired seeds 41-43 it improved test mean net worth over current PPO on all three seeds and reduced PPO update intensity. It is not yet a claim of general improvement: the conservative ablation has three seeds, and it remains below the buy-and-hold benchmark on the test regime.

## Environment and Commit

Recorded environment:

```text
HEAD before diagnostics: 06d9b55aff221b51089a6070ddb06263550fc787
Implementation commit tested: c692101
rustc 1.96.1
cargo 1.96.1
Linux x86_64, kernel 7.0.13-arch1-1
```

The initial full gate passed with 33 core unit tests, 49 foundation tests, and 7 Prompt 2 integration tests (89 tests before the checkpoint regression was added). The final gate after the fix passed 33 core, 50 foundation, and 7 Prompt 2 tests: **90 tests, 90 passed, 0 failed, 0 ignored**.

## Dataset and Frozen Protocol

`data/prompt3b/market.csv` contains 7,200 OHLCV rows: 1,800 business dates and `SPY`, `AAA`, `BBB`, `CCC`. Regime labels are kept separately in `data/prompt3b/regimes.csv` and are not observed by policies. The market SHA-256 is:

```text
59daa371f0f9f6a00128d99d66c8d86842f4b4bfef52df3f33105f65fe67ae28
```

Splits are chronological and frozen: training 1,260 dates (1,259 steps), validation 270 dates (269 steps), and test 270 dates (269 steps). Population size is 8, starting cash is 10,000, costs and architecture are shared, and seeds are 41-50. Survival calibration changed the generated market once to add cross-sectional regime variation and then set the common bankruptcy boundary to 8,950. Seed 42 produced one founder death out of eight (12.5%) in the no-learning evolutionary calibration; the pressure was then frozen for all conditions.

The config diff is in `reports/prompt3b/config-diff.txt`. Primary conditions differ only in learning enablement and evolution enablement. Conservative and diagnostic ablations explicitly change only their named schedule, trait initializer, order mode, or reward coefficient.

## Instrumentation

`RewardBreakdown` reports raw log return, drawdown, turnover, invalid-action, bankruptcy, unclipped, and clipped components while preserving the scalar reward exactly. Transactionally committed `DecisionTrace` and `PpoUpdateTrace` values expose costs, turnover, allocations, active positions, latents, order position, entropy, log probability, KL, clipping, gradients, and head deltas. JSONL traces are converted to compact CSV by `experiments/prompt3b/analyze_traces.py`.

Recoverable-failure behavior was verified by the focused regression test in `reports/prompt3b/failure-learning-test.txt`: one chronological masked entry is retained with its penalty and no fabricated actor probability; the policy-failure diagnostics test also passed.

## Primary Results

Values below are means across the available seeds; `control`, `ppo`, `evolution-only`, and `ppo-evolution` each use 10 seeds. Net worth is the population mean at the end of the frozen test split.

| Condition | Seeds | Validation NW | Test NW | Test drawdown | Test turnover | PPO updates | Deaths | Reproductions | Max generation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 10 | 10013.08 | 9260.26 | 0.09377 | 17461 | 0 | 1.0 | 0.0 | 0 |
| PPO | 10 | 10011.79 | 9267.50 | 0.09320 | 17681 | 2546.7 | 70.5 | 0.0 | 0 |
| evolution-only | 10 | 10011.09 | 9258.11 | 0.09393 | 17416 | 0 | 1.0 | 1.0 | 1 |
| PPO + evolution | 10 | 10009.70 | 9274.80 | 0.09265 | 17536 | 2547.7 | 70.9 | 70.9 | 40 |

The complete machine-readable table is `reports/prompt3b/condition-summary.csv`; per-seed values are in `reports/prompt3b/per-seed-results.csv`.

Benchmarks on the same frozen market (seed-independent policy definitions) were:

| Split | Cash | Random | Equal weight | Buy-and-hold |
| --- | ---: | ---: | ---: | ---: |
| Training | 9370.50 | 8862.96 | 8899.70 | 18995.29 |
| Validation | 9865.50 | 8948.91 | 9685.66 | 8943.40 |
| Test | 9865.50 | 8940.47 | 9746.74 | 8943.49 |

Neural conditions did not beat equal weight or cash on the test split. The benchmark outputs are preserved as `reports/prompt3b/benchmark-*.txt`.

## PPO Diagnosis

Representative seed 42 traces show:

| Condition | Mean turnover/step | Execution cost | Raw log return sum | Mean concentration | Mean active positions |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 40.4 | 488 | 0.081 | 0.252 | 4.0 |
| PPO | 6256.9 | 75,623 | -8.202 | 0.397 | 4.0 |
| PPO + evolution | 6302.3 | 76,172 | -8.270 | 0.400 | 4.0 |
| conservative PPO | 6235.0 | 75,358 | -8.029 | 0.397 | 4.0 |
| rotated order | 6769.4 | 81,817 | -8.641 | 0.397 | 4.0 |

The PPO schedule produced 2,548 updates in 1,259 steps for seed 42. Mean approximate KL was `1.16e-6`, clip fraction was zero, and gradients were finite. This is not a numerical explosion; it is repeated small head updates on very short rollouts. The conservative schedule produced 198 updates, smaller head deltas, and a lower mean KL. Paired test means for seeds 41, 42, and 43 were:

```text
current PPO       9265.19, 9260.81, 9265.43
conservative PPO  9278.14, 9267.73, 9277.25
```

Actor and critic deltas were non-zero; hidden and recurrent deltas remained zero and birth genomes matched the policy snapshots. The exact PPO diagnostics are in `reports/prompt3b/ppo-diagnostics.csv`.

## Turnover, Reward, and Concentration

The cost gap is much larger than the small test-net-worth gap because both controls and PPO eventually encounter the same falling test regime, while PPO pays repeated spread/slippage. PPO's reward mean/std for seed 42 was `-0.00786/0.0841`; control was `-0.000091/0.0106`. No baseline reward samples were clipped. Mild and moderate turnover coefficients produced penalties of approximately 31,400 and 62,800 over the trace; moderate clipping occurred on about 66% of samples. Neither penalty ablation reduced turnover or improved the three-seed test result, so no permanent turnover penalty was added.

Stable order bias is measurable and named-asset dependent. In stable order, mean allocations by position were approximately `[0.249, 0.252, 0.207, 0.133]`; in the rotated diagnostic the position pattern persisted but named assets became approximately equal (`0.208-0.213`). Rotating every episode nevertheless remapped latent decisions onto different assets and produced very high turnover and poor validation/test performance. A symmetric ordering redesign is deferred rather than guessed.

## Evolution

Evolution was exercised in real CLI runs. `ppo-evolution` produced 67-75 deaths and the same number of two-parent crossovers/mutations per seed, with no random fallback births. Seed 42 recorded 79 genome records, 71 completed lives, mean generation depth 15.67, maximum depth 34, 79 active lineages, and finite fitness variance. Evolution-only had only one death/reproduction per seed and maximum generation depth 1, so it is a useful low-pressure control but not evidence of adaptation. The full table is `reports/prompt3b/evolution-diagnostics.csv`; the seed-42 registry is preserved in `reports/prompt3b/ppo-evolution-seed-42-evolution-report.json`.

Evolution did not show consistent improvement over PPO or static control across the 10 paired seeds. It is mechanically operational, not demonstrated as beneficial.

## Checkpoint Determinism

Representative uninterrupted versus partial-checkpoint/resumed reports matched exactly for control, PPO, evolution-only, PPO+evolution, conservative PPO, and rotated-order seed 42. Results are recorded in `reports/prompt3b/checkpoint-equivalence.txt`. The repository regression suite also checks full domain-state equivalence. A correctness bug found during this protocol was fixed: frozen evaluation could overwrite a training checkpoint through automatic cadence. Evaluation now disables automatic checkpoint writes after validating the original training hash; `evaluation_does_not_overwrite_training_checkpoint` covers it.

## Verdicts

- **Mechanically valid:** yes. All 40 primary CLI runs and their validation/test evaluations completed; checkpoint resume matched; accounting and benchmark exposure were valid.
- **PPO learning operational:** yes. Actor/critic heads changed, the encoder/genome stayed fixed, losses and gradients were finite, and state-dependent traces were produced.
- **Evolution operational:** yes. Real CLI deaths, two-parent ancestry, crossover, mutation, fresh genome IDs, and generation depth were observed.
- **Turnover/cost mechanism confirmed:** yes. PPO turnover and execution costs were orders of magnitude above control and coincide with negative raw return.
- **Ordering bias detected:** yes. Stable position allocation is systematic; naive rotation is harmful because it remaps decisions every step.
- **PPO over-updating:** supported as a contributing factor. The conservative schedule reduced update count and improved all three paired test seeds, but it is not yet a 10-seed selected correction.
- **Reward mismatch:** turnover-penalty scaling is demonstrably too large for the observed turnover units; the ablation clipped heavily and did not improve behavior. No permanent reward change was made.
- **Improvement detected:** no general improvement claim. PPO+evolution and PPO are not consistently superior to static control or benchmarks. The conservative schedule is the recommended next controlled experiment, not a validated final policy.

## Remaining Uncertainty and Next Step

The main unresolved question is whether conservative training remains better over 10-20 paired seeds and a second frozen market seed. The next experiment should run the conservative schedule at full seed count, add a units-calibrated *isolated* turnover ablation only after fixing its scale, and test a non-remapping symmetric order policy. No profitability or intelligence claim is supported by this suite.
