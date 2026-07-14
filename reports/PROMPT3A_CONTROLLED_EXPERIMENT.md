# Prompt 3A Controlled Experiment

## Executive Summary

Prompt 3A replaces the neural single-asset categorical action with a bounded
stick-breaking portfolio distribution. Five seeds (41--45) were run for each
of four conditions on the offline synthetic training split. All 20 runs reached
`Completed`; checkpoint/resume was exact for seed 41 in every condition.

The new policy produced multi-asset portfolios in every neural action observed
in this short run: the multi-asset action fraction was `1.0` and the mean number
of active asset positions was `3.0`. PPO was operational: the actor and critic
heads changed while the frozen encoder remained unchanged and the birth genome
matched the policy. The frozen 16-step market did not produce deaths, so the
evolution-enabled CLI conditions did not exercise reproduction; evolution
compatibility is covered by the repository tests and the existing direct
replacement tests. No performance improvement over the static control was
detected: PPO finished below control on all five seeds.

## Commit and Environment

The final implementation and experiment artifact commit is
`aabb0ef320c979db8dc58268d4aec0145ce2db23`. The experiments used:

* Rust `1.96.1`, Cargo `1.96.1`
* Linux `7.0.13-arch1-1`, x86_64
* Burn `NdArray` CPU backend (`cpu_ndarray:default`)
* `data/raw/synthetic.csv`, training split 2020-01-02 through 2020-01-27
* Five seeds: 41, 42, 43, 44, 45

## Action Distribution

The actor emits two values per asset. For each tradable asset in stable asset
order, the values are transformed as:

```text
alpha = clamp((softplus(raw_alpha) + minimum_shape) / temperature)
beta  = clamp((softplus(raw_beta)  + minimum_shape) / temperature)
u     ~ Beta(alpha, beta)
weight = min(maximum_allocation, remaining) * u
```

Cash receives the remaining weight. Unavailable assets are omitted from the
active distribution and expanded back as exact zeroes. The PPO action is the
vector of latent `u` values, so its log probability is the sum of the exact
Beta log densities; no post-hoc clipping or renormalisation changes the action
identity. Entropy is the sum of Beta differential entropies. Deterministic
evaluation uses Beta means and does not consume RNG or decay temperature.

This is a bounded, ordered stick-breaking distribution rather than a general
exchangeable Dirichlet. Stable asset order is intentional for this version and
is a remaining source of ordering bias.

## Conditions

| Condition | Learning | Evolution |
|---|---|---|
| control | `no_learning` | disabled |
| ppo | `ppo` | disabled |
| evolution-only | `no_learning` | enabled, genome-only |
| ppo-evolution | `ppo` | enabled, genome-only |

The four TOMLs differ in exactly two fields: `learning.implementation` and
`evolution.enabled`. See [`prompt3a/config-diff.txt`](prompt3a/config-diff.txt).

## Results

Values are means across the five seed-level runs unless noted otherwise.

| Condition | Mean final NW | Median final NW | Mean return | Mean drawdown | Survival | PPO updates | Deaths | Reproductions | Max generation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 10236.66 | 10236.76 | 0.02367 | 0.000396 | 1.000 | 0 | 0 | 0 | 0 |
| ppo | 10166.51 | 10171.46 | 0.01665 | 0.000637 | 1.000 | 64 | 0 | 0 | 0 |
| evolution-only | 10236.66 | 10236.76 | 0.02367 | 0.000396 | 1.000 | 0 | 0 | 0 | 0 |
| ppo-evolution | 10166.51 | 10171.46 | 0.01665 | 0.000637 | 1.000 | 64 | 0 | 0 | 0 |

PPO versus control was `-70.15` mean final net worth, with zero of five
seeds beating control. The action distribution therefore increased portfolio
expressiveness, but this run does not show a return improvement.

| Condition | Multi-asset action fraction | Mean active positions | Mean concentration | Mean cash |
|---|---:|---:|---:|---:|
| control | 1.000 | 3.000 | 0.25188 | 0.25008 |
| ppo | 1.000 | 3.000 | 0.39233 | 0.26025 |
| evolution-only | 1.000 | 3.000 | 0.25188 | 0.25008 |
| ppo-evolution | 1.000 | 3.000 | 0.39233 | 0.26025 |

PPO diagnostics averaged 512 optimizer steps, actor delta `0.03442`, critic
delta `0.001096`, hidden delta `0`, recurrent delta `0`, and birth-genome
match `true`. Mean policy loss was `-0.000122`, value loss `0.000440`, and
gradient norm `0.75355`. Entropy is differential entropy and averaged
`-0.06447`; a negative differential entropy is valid for continuous densities.

## Benchmark Comparison

Training-split benchmark means were:

| Benchmark | Ending net worth |
|---|---:|
| Cash | 9968.00 |
| Random | 10177.92 |
| EqualWeight | 10328.33 |
| BuyAndHold | 10334.30 |

Control was `+268.66` versus cash, `-91.67` versus equal-weight, and `-97.64`
versus buy-and-hold. PPO was `+198.51`, `-161.82`, and `-167.79` respectively.
The benchmark exposure tests continue to assert fully invested equal-weight
and buy-and-hold semantics without neural concentration caps.

## Failure and Determinism Checks

The controlled market runs produced no policy failures, so failure-learning
coverage remains in deterministic injected tests. Those tests verify penalised
chronological masked transitions, actor/critic eligibility, exact fallback
event counts, and GAE continuity. A partial run at step 8 followed by resume
matched the uninterrupted report exactly for all four conditions at seed 41;
see [`prompt3a/checkpoint-resume-equivalence.txt`](prompt3a/checkpoint-resume-equivalence.txt).

Raw simulations, diagnostics, evolution reports, benchmarks, checkpoints, and
CSV aggregations are retained under [`prompt3a/`](prompt3a/). The complete
repository test suite and CLI verification were rerun after the implementation.

## Verdict

* **Mechanically valid:** Yes for the tested runs. All 20 runs completed and
  resume equivalence was exact.
* **PPO operational:** Yes. Full actor/critic head updates and multi-asset
  actions are measured; the encoder and birth genomes stayed fixed.
* **Evolution operational in this CLI suite:** Not demonstrated by these
  frozen runs because no agent died. Evolution architecture and child shape
  compatibility remain covered by unit/integration tests.
* **Improvement detected:** No. PPO was below control on all five seeds and
  below equal-weight and buy-and-hold on the mean.
* **Benchmark trustworthiness:** Exposure semantics are tested; results are
  short synthetic-horizon references, not profitability evidence.

## Limitations and Next Experiment

The run is CPU-only, short, synthetic, and has stable-order stick-breaking
bias. There is no turnover penalty, no full recurrent BPTT, and no claim of
profitability. A follow-up should use a longer or deliberately survival-
pressured horizon to exercise two-parent reproduction, while keeping the
multi-asset distribution and benchmark definitions fixed.
