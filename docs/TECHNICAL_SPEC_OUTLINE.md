# Midas Technical Specification & Architecture Outline

This document defines the complete technical architecture and system specification for **Midas**, an offline, deterministic, multi-asset artificial-life market simulator written in Rust.

---

## 1. High-Level Architecture & Domain Boundaries

Midas separates domain logic from command-line interface execution and experiment management across two primary workspace crates:
- **`midas-core`**: Contains the authoritative simulation engine, data pipelines, domain models, learning algorithms, and evolutionary biology mechanics. No dependencies on CLI or terminal frameworks.
- **`midas-cli`**: Serves as the interactive test harness, running calibration runs, benchmarks, diagnostics, walk-forward pipelines, and partial-step verification.

```
+-----------------------------------------------------------------------+
|                              midas-cli                                |
|   [validate-config] [simulate] [benchmark] [evaluate] [diagnostics]   |
+----------------------------------+------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|                              midas-core                               |
|                                                                       |
|  +---------------------+  +--------------------+  +----------------+  |
|  | Market & Data Core  |  | Brain & Learning   |  | Evolution & AL |  |
|  | (Polars, Execution, |  | (Burn Neural API,  |  | (2-Parent Mating|  |
|  |  Point-in-Time)     |  |  Beta Impulse, PPO)|  |  Genome Crossover|  |
|  +---------------------+  +--------------------+  +----------------+  |
+-----------------------------------------------------------------------+
```

---

## 2. Market & Simulation System

### 2.1 Canonical Data Pipeline & Schema
- **Long-Form Canonical Schema**: Loaded via Polars (`CSV` / `Parquet`) with strictly required fields: `date, asset, open, high, low, close, adjusted_close, volume`.
- **Valuation Source of Truth**: Raw unadjusted `close` prices serve as the sole source of truth for portfolio valuation and order execution. Arbitrary provider back-adjustment is rejected.
- **Corporate Action Processing**: In `explicit_events` mode, point-in-time split ratios and cash dividends are applied at the beginning of day `t+1` prior to market open:
  - **Splits**: Multiply share counts and divide average entry prices without altering gross equity.
  - **Dividends**: Direct cash credits to agent portfolios based on pre-open holdings.

### 2.2 Timing, Execution, and Transactionality
- **Close-to-Next-Open Execution**: At the close of day `t`, an agent observes point-in-time features known up to `t`. Proposed actions are validated and executed at open on `t+1`.
- **Accounting & Costs**: Supports fractional shares, cash tracking, bid-ask spread approximations, and proportional trade slippage.
- **Transactional Step Commit**: Each simulation step is atomic. Policy parameters, recurrent runtime memory, portfolios, metrics, counter IDs, replacement state, and RNG state are committed together only upon full step success.
- **Agent Mortality & Bankruptcy**: Configurable bankruptcy net-worth boundaries. Upon crossing below threshold, `KillAgent` terminates the agent, emits domain diagnostics (`AgentDied`), applies no next-day trades, and frees the population slot for reproduction.

---

## 3. Agent Brain & Reinforcement Learning System

### 3.1 Dual-Brain Ontology: Genome vs. Lifetime
Every Midas agent possesses two structurally decoupled representations of its cognitive and behavioral traits:
1. **Genome Brain (`BrainGenome`)**: An immutable genetic sequence defining matrix shapes, initial neural network parameters, bounded behavior traits (risk preference, exploration parameters, learning rate), and stable metadata hash. Created exclusively through evolutionary birth.
2. **Lifetime Brain (`LifetimeBrain`)**: Initiated at birth as an exact copy of the genome. Mutates during life via continuous trial-and-error Reinforcement Learning (PPO) and stores runtime recurrent sequences. Discarded upon death (non-Lamarckian inheritance by default).

### 3.2 Burn Neural Runtime & Impulse Action Mechanics
- **Framework Isolation**: Behind a framework-neutral interface, models are compiled using the [Burn](https://burn.dev/) framework (currently CPU `NdArray` backend).
- **Independent Bounded Beta Impulse Distribution**:
  - Rather than outputting arbitrary target allocation ratios that suffer from order bias, the actor head emits two raw parameters per asset.
  - Raw outputs transform into positive Beta distribution shapes: $\alpha, \beta = (\text{softplus}(\text{raw}) + \text{minimum\_shape}) / \text{temperature}$.
  - For each active asset, a latent $u \sim \text{Beta}(\alpha, \beta)$ maps directly to an incremental trade impulse: $\Delta = 2u - 1$.
  - Impulses are scaled by `maximum_trade_fraction` of current net worth and executed directly against existing portfolio holdings.

### 3.3 Lifetime Learning Strategy (Clipped PPO)
- **Frozen Encoder & Trainable Heads**: The hidden/recurrent encoder layers remain frozen during lifetime learning to maintain a stable state representation. Only dense output actor/critic heads undergo parameter updates.
- **GAE & Rollout Buffers**: Computes Generalized Advantage Estimation (GAE) across deterministic chronologically ordered sequence minibatches.
- **Recoverable Failure Learning**: Policy inference errors or invalid outputs emit `PolicyFailure` and `ActionFallback` events, recording one masked chronological rollout entry with an invalid-action penalty without corrupting probability calculations.

---

## 4. Modular Observation Framework

The observation pipeline aggregates multidimensional sensor streams into fixed, normalized input vectors before feeding them to the neural architecture:
- **Observation Registry & Providers**: A modular plug-in registry hosting read-only sensor groups:
  - `market_features`: Price returns, volume dynamics, moving averages.
  - `portfolio_features`: Current net worth, active exposure, cash concentration.
  - `agent_state`: Internal age, consecutive failure counters, current reward self-awareness.
  - `rolling_market_stats` & `agent_awareness`: Optional rolling window indicators (momentum, trends, volatility) with strict zero-fill for insufficient lookbacks.
- **Versioned Schema Hashes**: An explicit `ObservationSchema` maps provider string names to tensor indices and normalizes outputs. The compiled SHA-256 schema hash is baked into the checkpoint and `BrainArchitecture`, preventing silent observation misalignment on resume.

---

## 5. Evolution, Reproduction & Artificial Life

When an agent succumbs to mortality (bankruptcy or maximum lifespan):
1. **Fitness Evaluation**: Final completed-life economic performance is recorded in evolutionary registries.
2. **Parent Selection & Mating Constraints**: Eligible living parents are selected using configured fitness weighting (e.g., tournament or fitness proportional), enforced by a genetic mating distance constraint (with recorded relaxation fallback events if population diversity drops).
3. **Crossover & Mutation**: Two-parent crossover merges birth genomes, followed by bounded parameter mutations to initial neural matrix weights and behavioral traits.
4. **Lineage Tracking**: Assigns new unique genome IDs and lineage ancestors, spawning a completely fresh lifetime state with zeroed recurrent memory.

---

## 6. Built-In Benchmark References

Midas embeds standardized, non-learning, non-evolutionary baseline controls alongside agent populations within the exact same simulated environment:
- **`Cash`**: 100% cash preservation; zero market exposure or turnover.
- **`EqualWeight`**: Rebalances equally across all tradable assets daily without neural concentration constraints.
- **`BuyAndHold`**: Invests total starting net worth across active assets once at initial market open, subsequently holding without turnover or rebalancing costs.
- **`Random`**: Samples uniform random portfolio trade proposals to benchmark unskilled market interaction.
