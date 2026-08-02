# Prompt 9 Multi-Market Walk-Forward Experiment Protocol

Status: frozen before screening, confirmatory test, or holdout evaluation.

### Recorded Protocol Correction

During the required interrupted-run resume check after screening outputs had been
created, an incomplete final CSV row caused the runner's manifest rewrite to sort
a null task key and truncate the manifest. Raw reports, checkpoints, market data,
and simulation results were not affected. A regression test and atomic,
incomplete-row-tolerant manifest rewrite were added in a separate correctness
commit. Existing raw outputs are retained because the worker and all simulation
inputs are unchanged; the affected manifest is rebuilt from provenance-matched
outputs. The implementation commit below remains the simulation/generator/worker
identity embedded in every run, while the final report will identify the runner
correctness commit separately.

## Implementation

- Implementation commit: `23aba2a1bd720304f47f8899409fd6e48562161b`
- Checkpoint format: 13; version-12 migration compatibility remains covered by regression tests.
- Generator: `experiments/prompt9/generate_market.py`, version 1.
- Runner: `experiments/prompt9/run_experiment.py`.
- Aggregator: `experiments/prompt9/aggregate.py`.
- Market definitions: `configs/prompt9/generator/*.json`.
- Condition definitions: `configs/prompt9/conditions/*.toml`.
- Market, condition, and fold manifests: `data/prompt9/manifests/`.

No neural architecture, PPO formula, evolution algorithm, reward, observation,
execution, or checkpoint semantic is changed for this experiment. A deterministic
momentum benchmark and experiment-only generation/orchestration/reporting tooling
are the only new behavioral capabilities.

## Market Suite

The generator uses a local `random.Random(seed)` instance, business-day dates,
log returns, configurable common/idiosyncratic shocks, correlation, mean
reversion, volatility clustering, cross-sectional drift offsets, rotating
leadership, and volume response. Each specification contains its generation
timestamp. Reproduction from one specification is byte-identical at any output
path. Regime labels and boundaries are written only to separate metadata and are
never market columns or observations.

All markets contain 1,600 business dates, 4 assets (`SPY`, `AAA`, `BBB`, `CCC`),
6,400 rows, and run from 2010-01-04 through 2016-02-19. The suite has three
independent seeds in each development family:

1. Uptrend: positive drift, moderate volatility, negative pullback shocks, and asset dispersion.
2. Downtrend: negative drift, moderate volatility, positive relief shocks, and asset dispersion.
3. Sideways: zero drift with anchor mean reversion and repeated reversals.
4. High volatility: large moves, strong volatility clustering, and drawdown shocks.
5. Low volatility: small moves whose signal is comparable to trading costs.
6. Regime switching: low-volatility uptrend, high-volatility downtrend, sideways recovery, renewed trend.
7. Cross-sectional dispersion: rotating unequal asset drift leadership.
8. Correlated stress: high correlation, clustered volatility, and common adverse shocks.

Development generator seeds are 90101-90103 through 90801-90803 by family.
The untouched holdout uses seeds 99001-99004 and four new markets spanning
uptrend, sideways, regime-switching, and correlated-stress families. Holdout
dataset hashes are frozen in `data/prompt9/manifests/markets.csv` before any
holdout evaluation. Losing markets or seeds will not be regenerated or removed.

## Walk-Forward Folds

Each fold uses non-overlapping training, validation, and test windows internally.
Later folds expand the training window chronologically; no earlier training run
uses later validation or test rows.

| Fold | Training | Validation | Test |
|---|---|---|---|
| 1 | 2010-01-04 to 2012-04-20 | 2012-04-23 to 2013-01-25 | 2013-01-28 to 2013-11-01 |
| 2 | 2010-01-04 to 2013-06-14 | 2013-06-17 to 2014-03-21 | 2014-03-24 to 2014-12-26 |
| 3 | 2010-01-04 to 2014-08-08 | 2014-08-11 to 2015-05-15 | 2015-05-18 to 2016-02-19 |

Development screening and confirmation use all three folds. Untouched holdout
uses fold 3 only and is evaluated as one predeclared matrix.

## Frozen Conditions

Neural candidates use the existing Prompt 6 combined observation schema and
current settings without tuning:

1. `combined-ppo`: combined observations, PPO, evolution disabled.
2. `combined-ppo-evolution`: identical PPO and observations, evolution enabled.

Each neural run has population 8. A run-level result is the arithmetic mean of
the eight frozen evaluation agent results, matching prior Prompt 6 reporting;
failed/dead agents remain included.

Benchmarks use population 1, identical data, execution costs, survival cost,
starting cash, and split dates:

3. Cash.
4. Equal weight over currently tradable assets.
5. Buy and hold of the configured benchmark asset (`SPY`).
6. Seeded random allocation.
7. Cross-sectional momentum.

Momentum reads only each asset's current point-in-time normalized `return20`
observation, chooses the greatest strictly positive signal among tradable assets,
uses stable lower-index tie breaking, allocates 100% to that asset, and otherwise
holds cash. It re-evaluates daily and pays the normal costs. No clean mean-
reversion policy existed before protocol freeze; the prompt's priority rule is
therefore followed by including momentum rather than adding a second new policy.

## Checkpoint and Selection Rule

For every neural market/fold/seed, training uses only the training split. The
deterministic checkpoint at the completed training boundary is the selected
checkpoint. There is no best-of-epoch or best-seed selection. Validation results
will not select parameters, seeds, markets, or checkpoints.

Validation is used only for preregistered screening diagnostics: detecting
systemic failures, verifying the runner and metrics, and describing performance.
Both frozen neural candidates proceed to confirmation regardless of whether
their validation return is favorable. A technical failure rate above 5%, any
leakage failure, or nondeterministic reproduction stops the experiment for a
separately documented correctness fix; performance cannot trigger a redesign.

Frozen evaluation resets evaluation runtime and portfolio state through the
existing `SimulationEngine::evaluate_checkpoint` path. Learning, mutation,
replacement, and checkpoint overwrites are disabled by that path.

## Stages and Seeds

- Screening: seeds 41-45, all 24 development markets, all 3 folds, validation only.
- Confirmation: seeds 41-50, both frozen neural candidates, all benchmarks, all 24 development markets, all 3 folds, test only.
- Untouched holdout: seeds 41-50, all frozen conditions, 4 holdout markets, fold 3, test only, run once after confirmation.

Benchmarks are repeated at matching seeds so random is properly paired and every
comparison has a shared market/fold/seed key. Deterministic benchmarks may produce
identical repeated results; those repetitions are retained rather than collapsed.

Planned task counts (training is a separate task for neural conditions):

| Stage | Training tasks | Evaluation tasks | Total |
|---|---:|---:|---:|
| Screening | 720 | 2,520 | 3,240 |
| Confirmation | 1,440 | 5,040 | 6,480 |
| Holdout | 80 | 280 | 360 |
| Total | 2,240 | 7,840 | 10,080 |

A measured release run of one 1,200-step neural training plus 200-step validation
took approximately 5.7 seconds locally. The complete requested multi-fold design
is therefore retained. Runner parallelism is bounded at 4 jobs and per-task
timeout is 1,800 seconds.

## Recorded Metrics

For each condition/market/fold/seed/split the compact result records starting and
ending net worth, profit, total and annualized return, drawdown, daily-return
volatility, Sharpe-like and Sortino-like measures, turnover, transaction costs,
trade count, profitable status, paired benchmark wins, deaths, replacements,
reproduction and PPO update counts, final positions, mean cash allocation,
concentration, policy failures, checkpoint SHA-256, dataset identity/SHA-256,
schema identity, config hash, trace count, mutation/crossover counts, lifecycle,
and failure reason.

Annualization uses the simulator's 252-business-day convention. Sharpe-like uses
daily arithmetic returns, zero risk-free rate, and square-root-252 scaling.
Sortino-like uses downside daily returns under the same convention. These are
synthetic diagnostic ratios, not claims about deployable risk-adjusted returns.

Aggregates include mean, median, population standard deviation, interquartile
range, best/worst, profitable percentage, percentages beating required
benchmarks, mean/worst drawdown, mean transaction cost, and mean turnover by
strategy, regime, market, fold/seed source rows, all markets, and holdout.

## Paired Analysis

Return differences are paired by identical market, fold, split, and seed for:

- Combined PPO versus cash, equal weight, and momentum.
- Combined PPO plus evolution versus Combined PPO, cash, equal weight, and momentum.

Reports include mean and median paired difference, wins/losses/ties, paired win
percentage, a deterministic 2,000-resample percentile confidence interval for
the mean, and an exact two-sided paired sign test. These statistics are
exploratory and cannot by themselves establish deployable profitability.

## Frozen Success Criteria

The result is classified as at least promising only if all are satisfied:

1. Positive median development test return across independent markets.
2. At least 60% of neural test runs are profitable.
3. Beats cash in at least 60% of paired test runs.
4. Beats equal weight in at least 60% of paired test runs.
5. Beats momentum in more than 50% of paired test runs.
6. Positive aggregate paired advantage after all configured costs.
7. No single market contributes more than 35% of aggregate neural profit.
8. Positive median result in at least five of eight development regime families.
9. Positive median return on the untouched holdout.
10. Drawdown is not materially worse than benchmarks without compensating return.

For criterion 10, "materially worse" means aggregate mean maximum drawdown more
than 25% above the comparator while aggregate mean return is not higher. A
stronger conclusion additionally requires repeated holdout wins over rule-based
benchmarks, seed stability, and no dominant market/regime. Otherwise the report
must state that Midas has not demonstrated a general trading edge.

## Failure, Leakage, and Overfitting Controls

Every attempted task remains in the run manifest. Failed runs are not removed or
imputed; reasons appear in `failures.csv`. A stage is invalid if completed-output
provenance does not match implementation commit, config, dataset, and checkpoint
hashes.

Before final reporting, tests must confirm that market and rolling observations
use only current/historical rows, future corporate actions are unavailable,
splits are isolated, frozen evaluation resets runtime without learning/evolution,
regime labels/boundaries are absent from observations, momentum consumes only its
current observation, generator reruns are byte-identical, interrupted runner
execution resumes without changing completed output, and aggregation is byte-
identical for identical inputs.

No final test or holdout result may cause seed removal, market regeneration,
reward/observation/PPO/evolution changes, benchmark changes, or report omission.
If a correctness defect blocks execution, it receives a failing regression test,
a separate disclosed commit, affected baselines are rerun, and this protocol gets
an explicit dated amendment preserving the original text.

## Output and Retention

Tracked artifacts include generator specifications, manifests with dataset
hashes, this protocol, compact aggregate CSVs, reproducibility hashes, and final
reports. Reproducible 18 MB market CSVs, generated run configs, approximately
4.6 MB checkpoints, and raw/log files are locally retained but gitignored to
avoid repository growth. They can be reconstructed from the tracked inputs.
