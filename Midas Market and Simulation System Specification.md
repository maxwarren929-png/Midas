# Midas Market and Simulation System Specification

**Version 1.0**

## 1. Purpose

Midas is a Rust-based artificial-life and machine-learning experiment in which a population of neural-network agents trades using historical real-world stock-market data.

The Market and Simulation System is responsible for providing the world in which these agents operate. It must:
- Load and process historical market data.
- Present agents only with information available at that simulated moment.
- Execute simulated trades consistently.
- Track cash, holdings, costs, and net worth.
- Detect bankruptcy and agent death.
- Prevent future-data leakage.
- Support deterministic and reproducible experiments.
- Produce trustworthy performance measurements.

This system does not decide how agents think, learn, reproduce, or mutate. Those behaviours are defined in the Agent Brain and Evolution specifications.

Midas V1 is strictly a research simulator. It must not place real trades or connect to a brokerage account.

## 2. Design Principles

### 2.1 Correctness before realism

The simulator must prioritise correct accounting and repeatable behaviour before adding advanced market mechanics. A simple market model that is fully tested is preferable to a realistic-looking model with hidden errors.

### 2.2 No future information

An agent may only observe information that would have been available when its decision was made. No calculation, data adjustment, feature, or portfolio action may expose future prices or future company information.

### 2.3 Deterministic execution

Given the same:
- Dataset
- Configuration
- Agent states
- Random seed
- Program version

the simulation must produce the same results.

### 2.4 Shared and fair market conditions

All agents in the same experiment must experience the same market timeline and the same underlying asset prices. Agents may have different decisions, portfolios, learning states, and transaction costs caused by their own actions.

### 2.5 Price-taking agents

In V1, agents are too small to affect the real market. Their trades do not alter historical prices or the prices observed by other agents. Market impact may be added in a future version.

## 3. Technology Requirements

The system must be implemented in Rust.

Recommended components:
- Polars: market-data loading, validation, transformation, and feature preparation.
- Serde: configuration, metadata, and checkpoint serialisation.
- Rayon: safe parallel processing where determinism can be preserved.
- Parquet: primary processed-data storage format.
- CSV: supported as an import and export format.
- Tracing: structured logs and diagnostics.
- Anyhow or thiserror: error handling.

Exact crate versions must be pinned through Cargo.lock.

The simulation core must not depend directly on one market-data provider. Provider-specific download code must be isolated from the internal data format.

## 4. Scope of Version 1

Midas V1 must support:
- Historical daily market candles.
- Multiple stocks and exchange-traded funds.
- Long-only trading.
- Buying, selling, and holding.
- Fractional shares.
- Cash balances.
- Transaction fees.
- Bid-ask spread approximation.
- Slippage.
- Bankruptcy and death.
- Training, validation, and testing periods.
- Buy-and-hold and cash benchmarks.
- Deterministic simulation seeds.
- Local offline datasets.
- Parallel agent decision evaluation.

Midas V1 must not support:
- Real-money trading.
- Brokerage integration.
- Intraday candles.
- Options.
- Futures.
- Short selling.
- Margin borrowing.
- Leverage.
- Interest-bearing debt.
- Agent-driven price movement.
- Limit orders.
- Stop-loss orders.
- After-hours trading.
- Dividend reinvestment instructions chosen by agents.
- Fundamental or news data.

These may be introduced in later versions.

## 5. Market Dataset

### 5.1 Required candle fields

Every asset must contain daily records with:

```
timestamp
open
high
low
close
adjusted_close
volume
```

Optional metadata may include:

```
ticker
exchange
currency
asset_type
data_source
download_timestamp
```

All numeric prices must be positive finite values. Volume must be zero or positive. Each asset may have at most one candle for a given trading date.

### 5.2 Canonical internal schema

Processed data must be converted into a canonical format:

```rust
struct MarketBar {
    date: TradingDate,
    asset_id: AssetId,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    adjusted_close: f64,
    volume: f64,
    tradable: bool,
}
```

The exact date and identifier types may differ, but they must be strongly typed.

Currency values used for portfolio accounting must use f64 in V1, with strict finite-value checks and tolerance-based tests. A future version may replace floating-point accounting with fixed-point decimal arithmetic.

### 5.3 Adjusted and unadjusted prices

Raw trading execution must use unadjusted market prices. Adjusted prices may be used for historical return calculations and features, provided they do not introduce future knowledge. Stock splits and dividends must be handled consistently. The system must not mix adjusted holdings with unadjusted execution prices.

### 5.4 Data storage

The data pipeline must maintain two separate forms:

```
data/raw/
data/processed/
```

Raw files must be treated as immutable source records. Processed files must be generated reproducibly from raw files. Parquet is the preferred processed format because it is compact and fast to load.

### 5.5 Initial asset universe

The default V1 demonstration dataset should contain approximately 10-20 liquid assets from multiple market sectors and asset categories.

A reasonable example universe is:

```
SPY
QQQ
IWM
AAPL
MSFT
JPM
XOM
JNJ
KO
GLD
```

The exact tickers must be configurable. No asset ticker may be hard-coded into the simulation engine.

### 5.6 Survivorship bias warning

A dataset containing only companies that remain successful today creates survivorship bias. V1 may use a fixed modern asset universe for development, but all reports must clearly state this limitation. The final research version should eventually support historically accurate asset membership and delisted securities.

## 6. Data Validation

Before an experiment begins, the loader must validate:
- Dates are ordered.
- Dates are unique per asset.
- Prices are finite.
- Prices are positive.
- low <= open <= high.
- low <= close <= high.
- Volume is non-negative.
- Required assets exist.
- Required date ranges are available.
- There are no duplicate rows.
- There are no impossible percentage values.
- No future row is included in an earlier observation.

Invalid data must cause a clear error rather than being silently accepted.

Missing trading days may be valid because of weekends, public holidays, suspensions, or asset-specific gaps. Missing price values must not be automatically filled with future prices.

## 7. Master Trading Calendar

Midas must create a master ordered sequence of trading dates.

For V1, the recommended calendar is based on the union of all valid dates across the configured asset universe.

For each date and asset, the system must track whether the asset is tradable. An asset may be unavailable because:
- It was not yet listed.
- It was delisted.
- Its data is missing.
- Trading was suspended.
- The market record is invalid.

Unavailable assets must not be purchasable or sellable on that date. Existing holdings in a temporarily unavailable asset remain in the portfolio and are valued using the latest previously known valid close, while being marked untradable. This fallback must be visible in diagnostics.

## 8. Simulation Timeline

One simulation step represents one trading day.

The standard daily sequence must be:

1. Begin trading day.
2. Reveal the permitted observation.
3. Agents choose actions.
4. Validate requested actions.
5. Convert actions into orders.
6. Execute orders using the configured execution price.
7. Apply fees, spread, and slippage.
8. Update cash and holdings.
9. Advance to the daily closing valuation.
10. Calculate new portfolio net worth.
11. Calculate rewards and metrics.
12. Check for bankruptcy.
13. Record logs and checkpoints.
14. Advance to the next trading day.

The sequence must be fixed and documented because changing the order may change results.

## 9. Observation Timing

The simulator must explicitly define when agents make decisions.

Midas V1 will use the following model:

> At the end of trading day t, the agent observes data available through the close of day t and selects target portfolio positions to be executed at the open of trading day t + 1.

This avoids allowing an agent to observe a closing price and trade at that same closing price.

The agent must never:
- Observe day t + 1 prices before acting.
- Trade using the day t + 1 closing price.
- Use features calculated from future rows.
- See future corporate events.
- See future asset-universe membership.

The first tradable action therefore occurs only after sufficient historical observations have been collected.

## 10. Agent Actions

The brain specification will define the exact neural outputs. The market system must accept normalised target allocations.

For N assets, an agent action contains:

```
cash allocation
asset 1 allocation
asset 2 allocation
...
asset N allocation
```

All allocations must be finite and non-negative. The total target allocation must equal or be normalised to:

```
1.0
```

Because V1 is long-only and has no leverage:

```
0.0 <= allocation <= 1.0
```

The system must support holding cash.

Example:

```
Cash: 0.20
SPY: 0.35
AAPL: 0.25
GLD: 0.20
```

Invalid outputs such as NaN, infinity, or all-zero allocations must activate a safe fallback. The default fallback is:

```
100% cash
```

The invalid action must also be logged.

## 11. Portfolio State

Each living agent must have an independent portfolio:

```rust
struct Portfolio {
    cash: f64,
    positions: Vec<Position>,
    net_worth: f64,
    previous_net_worth: f64,
    cumulative_fees: f64,
    cumulative_slippage: f64,
}
```

Each position must track at least:

```rust
struct Position {
    asset_id: AssetId,
    quantity: f64,
    average_entry_price: f64,
}
```

Fractional shares are permitted in V1. All quantities must remain non-negative. No agent may hold a negative cash balance after order execution.

## 12. Starting Capital

Every newly created agent begins with a configurable amount of simulated cash.

Default:

```
$10,000
```

A newborn agent begins with:
- Full starting capital in cash.
- No holdings.
- No pending orders.
- No inherited money.
- No inherited portfolio positions.
- No inherited transaction history.

The starting value must be equal for all births unless an experiment explicitly tests another rule.

## 13. Order Generation

The execution engine converts target allocations into the required changes in holdings.

For each asset:

```
target position value - current position value = requested trade value
```

Small trades below a configurable minimum threshold may be ignored to prevent meaningless continuous rebalancing.

Recommended default:

```
minimum trade value = $1.00
```

The system must sell assets before buying new assets so that proceeds can fund purchases. If available cash is insufficient after costs, buy orders must be reduced proportionally. The engine must never allow cash to become negative.

## 14. Execution Price

All V1 orders are market orders executed at the next trading day's open.

Base execution price:

```
next_day_open
```

The effective buy price must be increased by spread and slippage. The effective sell price must be decreased by spread and slippage.

Example:

```
buy price = open × (1 + half_spread + slippage)
sell price = open × (1 - half_spread - slippage)
```

All adjustments must be configurable.

If the next-day opening price is missing or invalid, the trade must not execute. The order must be rejected or postponed according to one fixed policy.

V1 policy: Reject the order for that date and leave the existing portfolio unchanged for that asset.

## 15. Trading Costs

### 15.1 Commission

Each executed trade must incur a configurable commission.

Default model:

```
commission = fixed fee + percentage fee
```

Recommended defaults:

```
fixed fee: $0.00
percentage fee: 0.05%
```

### 15.2 Bid-ask spread

Spread must be modelled as an execution disadvantage.

Recommended default:

```
total spread: 0.10%
half-spread per execution side: 0.05%
```

### 15.3 Slippage

Slippage must be modelled separately from spread.

Recommended V1 model:

```
base slippage + volume-dependent slippage
```

For the first implementation, a fixed slippage rate is sufficient.

Recommended default:

```
0.02%
```

A future version may calculate slippage from order size and market volume.

### 15.4 Cost transparency

The engine must track separately:
- Commission paid.
- Spread cost.
- Slippage cost.
- Total turnover.
- Total traded value.

Costs must not be merged invisibly into portfolio returns.

## 16. Daily Survival Cost

Agents that remain entirely in cash could otherwise survive indefinitely without participating. Midas V1 therefore includes an optional daily survival cost.

Default:

```
$2.00 per simulated trading day
```

This cost is deducted from cash after daily valuation. If cash is insufficient, assets must not be automatically sold merely to pay the cost. Instead, the unpaid amount reduces net worth through a survival-cost liability for that step.

For V1 simplicity, the implementation may directly deduct the cost from total portfolio value during the bankruptcy calculation.

The survival cost must be configurable and may be disabled for comparison experiments.

## 17. Net Worth

An agent's net worth is:

```
cash
+ sum of current market value of all holdings
- unpaid costs or liabilities
```

Because V1 has no borrowing or margin, normal liabilities should be limited to outstanding simulation costs. Holdings are valued at the current day's closing price. Net worth must be recalculated after every simulation day. The simulator must reject NaN, infinity, and impossible negative position values.

## 18. Bankruptcy and Death

An agent dies when:

```
net_worth <= bankruptcy_threshold
```

Default threshold:

```
$0.01
```

Using a small positive threshold avoids floating-point edge cases around exact zero.

A dead agent:
- Cannot trade.
- Cannot receive future rewards.
- Cannot continue lifetime learning.
- Cannot be selected as a parent after death.
- Has its final state recorded.
- Is passed to the Evolution System for replacement.

The market system reports the death event but does not decide the parents or construct the offspring.

Death records must include:

```
agent identifier
birth date
death date
age in trading days
starting net worth
final net worth
maximum net worth
total return
maximum drawdown
fees paid
cause of death
```

## 19. End-of-Period Handling

An agent that survives until the end of an experiment does not automatically die.

At the end of the timeline:
- All holdings are valued at the final closing price.
- Final performance metrics are calculated.
- Positions do not need to be forcibly liquidated unless configured.
- Remaining agents are marked as survivors.
- The experiment report distinguishes bankruptcy deaths from timeline completion.

Forced liquidation may be enabled for exact cash-based comparisons, but it must include all normal transaction costs.

## 20. Dataset Splits

Every experiment must define separate chronological periods:

```
training
validation
test
```

Example only:

```
Training:   2005-2018
Validation: 2019-2021
Test:       2022-2025
```

Exact dates are configuration values.

Rules:
- Training data may influence neural learning and evolutionary selection.
- Validation data may influence configuration choices.
- Test data must remain untouched until final evaluation.
- Test performance must not influence training, mutation, reward design, or hyperparameter selection.

The system must reject overlapping split dates.

## 21. Walk-Forward Evaluation

Midas should support walk-forward experiments.

Example:

```
Train: 2005-2012    Validate: 2013    Test: 2014
Train: 2006-2013    Validate: 2014    Test: 2015
```

This allows evaluation across different market regimes without repeatedly testing on one fixed period. Walk-forward windows must be generated from configuration, not hard-coded.

## 22. Episode Selection

During training, agents may encounter different historical episodes.

An episode is a continuous date range from the training dataset.

Episode selection may be:
- Sequential.
- Randomised.
- Regime-balanced.
- Curriculum-based.

V1 should support sequential and seeded-random episode selection. Random episode selection must use a deterministic experiment seed. All agents being compared during one episode should receive the same market period unless an experiment explicitly tests otherwise.

## 23. Feature Preparation Boundary

The market system may calculate general market features, such as:
- Percentage returns.
- Log returns.
- Moving averages.
- Volatility.
- Volume changes.
- Distance from rolling highs.
- Market-relative performance.

However, the exact observation vector is defined in the Agent Brain specification.

Every rolling calculation must use only the current and previous rows. For example, a 20-day moving average at day t may use:

```
t-19 through t
```

It must not use any row after day t.

Features must be calculated separately within each chronological split or with carefully controlled historical context. Validation and test normalisation statistics must come only from training data.

## 24. Normalisation

Raw prices must not be passed directly to neural networks without transformation.

The data system must support normalised features.

Permitted methods include:
- Percentage change.
- Log return.
- Ratio to moving average.
- Training-set z-score.
- Bounded clipping.
- Rolling volatility scaling.

Any fitted normalisation parameters must be derived only from the training period.

The following are forbidden:
- Normalising the entire dataset before splitting.
- Using future maximum or minimum prices.
- Using test-period means or standard deviations.
- Scaling with values calculated after the observation date.

## 25. Future-Data Leakage Protection

The system must contain explicit automated tests against data leakage.

Required protections include:
1. Observations at day t cannot access rows after t.
2. Orders based on day t information execute no earlier than day t + 1.
3. Rolling features use backward-looking windows only.
4. Training normalisers never fit on validation or test data.
5. Test-period results cannot alter training state.
6. Future asset availability is not exposed.
7. Missing values are not filled using future rows.
8. Adjusted-price handling is documented and consistent.
9. Random episode selection never crosses split boundaries.
10. Checkpoints identify which data periods were accessed.

Any violation must fail loudly in tests or runtime validation.

## 26. Benchmark Strategies

Every experiment report must compare agents against basic benchmarks.

### 26.1 Cash benchmark

Hold 100% cash for the entire period while paying the same applicable survival costs.

### 26.2 Equal-weight buy-and-hold

Divide starting capital equally across all available assets and hold them.

### 26.3 Market-index buy-and-hold

Invest all capital in the configured benchmark index or ETF, such as SPY.

### 26.4 Random policy

Generate valid random target allocations using the same execution and cost rules.

Benchmark strategies must use the same:
- Starting capital.
- Date range.
- Tradability rules.
- Fees.
- Spread.
- Slippage.
- Survival costs.

## 27. Performance Metrics

The simulator must calculate at least:

```
starting net worth
ending net worth
total return
annualised return
daily return series
volatility
maximum drawdown
Sharpe-like ratio
Sortino-like ratio
turnover
number of trades
win-rate where meaningful
total fees
total spread cost
total slippage cost
survival duration
bankruptcy status
```

The system must not treat raw profit as the only measure of success. Risk-free-rate assumptions must be configurable and clearly stated. Metrics must avoid division by zero and invalid floating-point results.

## 28. Determinism

Every experiment must have a master random seed.

Derived seeds should be generated for:
- Episode selection.
- Agent exploration.
- Trade slippage if stochastic.
- Population initialisation.
- Reproduction.
- Mutation.

The same simulation must not depend on thread scheduling.

Parallel processing must follow this pattern:
1. Create immutable observation.
2. Evaluate agent decisions in parallel.
3. Store decisions in indexed buffers.
4. Execute decisions in deterministic agent order.
5. Apply results.

No agent may modify shared market state during parallel decision evaluation.

## 29. Configuration

The simulation must be configurable from a human-readable file such as TOML.

Example categories:

```toml
[market]
assets = ["SPY", "QQQ", "AAPL"]
benchmark_asset = "SPY"

[simulation]
starting_cash = 10000.0
bankruptcy_threshold = 0.01
daily_survival_cost = 2.0
minimum_trade_value = 1.0
seed = 42

[execution]
commission_fixed = 0.0
commission_rate = 0.0005
spread_rate = 0.001
slippage_rate = 0.0002

[data]
processed_path = "data/processed/market.parquet"

[splits]
training_start = "2005-01-01"
training_end = "2018-12-31"
validation_start = "2019-01-01"
validation_end = "2021-12-31"
test_start = "2022-01-01"
test_end = "2025-12-31"
```

All configuration values must be validated before simulation. Invalid values must produce readable errors.

## 30. Logging and Records

The simulator must support structured logs at multiple levels.

**Minimal experiment log:**
```
experiment identifier
configuration hash
program version
dataset hash
random seed
start time
end time
number of agents
number of market days
number of deaths
number of births
final population statistics
benchmark results
```

**Optional detailed logs:**
```
daily agent net worth
daily allocations
executed trades
rejected trades
fees
slippage
death events
invalid neural outputs
missing market data
```

Detailed logging must be optional because logging every agent every day may become extremely large.

## 31. Checkpoints

The simulation must support saving and restoring checkpoints.

A checkpoint must include:
- Current trading date.
- Experiment configuration.
- Dataset identity or hash.
- Random-number-generator states.
- Population identifiers.
- Portfolio states.
- Accumulated metrics.
- Pending evolution events.
- Relevant simulation counters.

Agent brains and genomes are defined by the other specifications but must be included in complete experiment checkpoints.

Loading a checkpoint with a different dataset or incompatible configuration must produce an error.

## 32. Error Handling

The system must never silently continue after serious corruption.

**Fatal errors include:**
- Invalid prices.
- Missing required configuration.
- Negative holdings.
- Non-finite cash.
- Non-finite net worth.
- Duplicate market dates.
- Overlapping data splits.
- Incompatible checkpoint.
- Failed accounting invariant.

**Recoverable events include:**
- An unavailable asset.
- A rejected trade.
- An invalid agent action replaced with the cash fallback.
- A missing optional log destination.

Recoverable events must still be recorded.

## 33. Required Accounting Invariants

After every execution step, the following must hold:

```
cash >= 0
position quantity >= 0
all values are finite
total buy spending <= available cash
fees are non-negative
slippage costs are non-negative
net worth matches independently recalculated value
no trade occurs in an unavailable asset
no agent uses another agent's holdings
```

An invariant failure must stop the affected experiment.

## 34. Testing Requirements

### 34.1 Unit tests

Required tests include:
- Buying shares reduces cash correctly.
- Selling shares increases cash correctly.
- Fees are charged correctly.
- Spread worsens execution correctly.
- Slippage worsens execution correctly.
- Fractional shares work.
- Insufficient-cash orders are reduced.
- Negative allocations are rejected.
- NaN outputs use the safe fallback.
- Unavailable assets cannot be traded.
- Bankruptcy occurs at the configured threshold.
- Net worth is calculated correctly.
- Survival costs are applied correctly.

### 34.2 Property tests

Property-based tests should verify:
- Cash never becomes negative.
- Holdings never become negative.
- Fees never increase wealth.
- Slippage never improves an execution price.
- An agent cannot create money through immediate buy-sell loops.
- Identical seeds produce identical results.
- Different agent portfolios remain isolated.

### 34.3 Integration tests

Required scenarios:
1. One asset with constant price.
2. One asset with steadily rising price.
3. One asset with steadily falling price.
4. Multiple assets with missing dates.
5. An agent remaining entirely in cash.
6. An agent continuously trading until costs cause bankruptcy.
7. A complete small training-to-test experiment.
8. Save checkpoint, reload, and confirm identical continuation.

### 34.4 Leakage tests

Create synthetic price data where future values are deliberately extreme.

Confirm that changing a future price does not alter:
- Earlier observations.
- Earlier features.
- Earlier actions.
- Earlier rewards.
- Earlier portfolio values.

## 35. Performance Requirements

Midas must be able to simulate at least:

```
500 agents
10 assets
20 years of daily data
```

without architectural changes. The design should later scale to thousands of agents.

Market data for the current step should be shared rather than copied separately for every agent. Agent decisions may be evaluated in parallel. Trade execution and accounting must remain deterministic.

Performance optimisations must not weaken correctness checks without an explicit release-mode policy.

## 36. Suggested Rust Module Structure

```
src/
├── market/
│   ├── mod.rs
│   ├── bar.rs
│   ├── calendar.rs
│   ├── dataset.rs
│   ├── features.rs
│   ├── loader.rs
│   └── validation.rs
├── simulation/
│   ├── mod.rs
│   ├── engine.rs
│   ├── timeline.rs
│   ├── observation.rs
│   └── checkpoint.rs
├── execution/
│   ├── mod.rs
│   ├── order.rs
│   ├── pricing.rs
│   ├── costs.rs
│   └── rebalance.rs
├── portfolio/
│   ├── mod.rs
│   ├── position.rs
│   ├── portfolio.rs
│   └── accounting.rs
├── metrics/
│   ├── mod.rs
│   ├── returns.rs
│   ├── drawdown.rs
│   └── benchmark.rs
└── config/
    ├── mod.rs
    └── validation.rs
```

The exact structure may evolve, but market data, execution, portfolio accounting, and simulation control must remain logically separated.

## 37. Version 1 Completion Criteria

The Market and Simulation System is complete when:
1. Historical daily data loads from local CSV or Parquet files.
2. Data validation detects malformed records.
3. Training, validation, and test periods are separated.
4. Agents receive only legal historical observations.
5. Target allocations convert into valid trades.
6. Trades execute at the next day's open.
7. Fees, spread, and slippage are applied.
8. Portfolios remain internally consistent.
9. Net worth is calculated daily.
10. Agents die when reaching the bankruptcy threshold.
11. Death events are passed to the Evolution System.
12. Benchmarks run under identical rules.
13. Checkpoints save and restore correctly.
14. Identical seeds reproduce identical outcomes.
15. Required unit, integration, property, and leakage tests pass.
16. The simulator runs from configuration without source-code edits.

## 38. Future Extensions

The following features are explicitly deferred:
- Intraday market data.
- Limit-order-book simulation.
- Agent-driven market impact.
- Realistic liquidity curves.
- Short selling.
- Margin and leverage.
- Interest rates.
- Taxes.
- Dividends as explicit cash payments.
- Delistings and bankrupt real companies.
- Historical index membership.
- Multiple currencies.
- Foreign-exchange conversion.
- Fundamental data.
- News and sentiment.
- Brokerage integration.
- Live paper trading.
- Real-money execution.

These features must not be partially added to V1 unless all accounting and testing implications are addressed.

## 39. Final System Definition

The Midas Market and Simulation System is:

> A deterministic Rust-based historical market environment that exposes legally timed observations, executes long-only simulated trades with realistic costs, maintains isolated agent portfolios, detects bankruptcy, and produces reproducible performance results without future-data leakage.

Its primary responsibility is not to make agents profitable. Its responsibility is to make every profit, loss, survival, and death trustworthy.
