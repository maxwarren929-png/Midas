# Philosophy, Design Ethos & Goal of Midas

This document outlines the philosophical principles, engineering tenets, and overarching scientific ambitions driving **Midas**, a deterministic artificial-life and reinforcement-learning market simulation harness written in Rust.

---

## 1. Core Mission & Vision

Midas is designed to answer a fundamental scientific question in artificial intelligence and financial economics:

> **Can populations of neural network agents, interacting within an evolutionary marketplace and adapting via reinforcement learning, develop robust behavioral strategies without reliance on human heuristic biases?**

Crucially, Midas is **not** built to be a profitable trading bot, nor is it a commercial algorithmic execution platform. It is an offline, experimental testbed and simulated sandbox—an artificial world designed for reproducible, controlled experimentation at the intersection of quantitative finance, Darwinian artificial life, and deep reinforcement learning.

---

## 2. Guiding Design Principles

Midas strictly enforces five non-negotiable architectural tenets across its entire codebase:

### I. Correctness Before Realism
In complex simulation systems, attempting to model highly intricate financial realisms (such as order-book queue position, high-frequency latency arbitrage, or endogenous market impact) often obscures foundational implementation bugs. Midas embraces clean, provable simplicity over fragile realism:
* Proportional transaction fee calculations and slippage curves are applied uniformly.
* Strong type accounting and invariant checks guarantee that an agent cannot create cash out of nothing or hold negative equity.
* A verified simple model is universally preferred over an elaborate abstraction containing unmeasurable error states.

### II. Zero Future-Information Leakage
In historical market backtests, data leakage (survivorship bias, look-ahead bias, or incorporating future corporate splits into past training prices) destroys scientific validity.
* An agent may only observe point-in-time state that would have been physically known at that exact historical second.
* Raw closing prices act as the absolute source of truth for execution and portfolio equity valuation; arbitrary commercial adjusted prices are explicitly rejected for trading logic.
* Feature sensors (e.g., rolling averages, volatility, momentum) rely strictly on backward-looking windows with deterministic zero-filling for insufficient early history.

### III. Absolute Determinism & Transactional Verification
Scientific reproducibility demands that any simulation can be duplicated byte-for-byte anywhere in the world.
* Given an identical configuration TOML, canonical dataset, random seed, and schema hash, Midas guarantees **exact byte-level reproducibility** across every metric and parameter delta.
* Simulation steps execute as **atomic transactions**: policy inference, memory updates, portfolio fills, event logs, and Random Number Generator (RNG) states commit simultaneously only when the step fully succeeds. If interrupted, checkpoints allow flawless resumption with zero state corruption.

### IV. Shared & Fair Market Environment
All agents within an experimental generation experience an identical chronological universe of market prices. While agents diverge in their inherited genomes, lifetime memories, and portfolio balances, the environmental pressures (fees, bankruptcy thresholds, survival costs) apply identically and equitably.

### V. Price-Taking Agents (V1 Scope)
In Version 1, individual agents act as atomistic price-takers. Their portfolio trade proposals do not alter historical market clearing prices or impact other agents' sensor data. Endogenous price discovery and market impact remain purposefully deferred to future research phases.

---

## 3. The Biological Concept: Evolution vs. Experience

Midas constructs an explicit philosophical and ontological separation between **Darwinian evolution** and **individual lifetime adaptation**:

```
+-------------------------------------------------------------------------+
|                  THE DUAL-BRAIN COGNITIVE PARADIGM                      |
|                                                                         |
|   +-----------------------+              +--------------------------+   |
|   |   GENOME BRAIN        |    Spawns    |   LIFETIME BRAIN         |   |
|   |   (Innate Instincts)  | -----------> |   (Empirical Experience) |   |
|   |                       |              |                          |   |
|   | - Immutable at birth  |              | - Mutates daily via PPO  |   |
|   | - Trait boundaries    |              | - Recurrent short memory |   |
|   | - Passed to offspring |              | - Discarded upon death   |   |
|   +-----------------------+              +--------------------------+   |
+-------------------------------------------------------------------------+
```

* **Evolution Creates Instincts**: Evolutionary algorithms (parent fitness selection, distance-constrained mating, genetic crossover, and trait mutation) operate across multi-generational time scales. They embed structural starting weights, risk tolerance parameters, and learning rate boundaries—representing innate biological reflexes at birth.
* **Reinforcement Learning Creates Experience**: Once born, an agent navigates real-time price dynamics using Proximal Policy Optimization (PPO). The agent adjusts its *Lifetime Brain* weights based on financial reward signals and GAE rollouts.
* **Non-Lamarckian Inheritance by Default**: When an agent dies from bankruptcy or old age, its learned empirical experience (the lifetime network weights and memory state) is permanently erased. Offspring inherit only combinations of their parents' underlying *birth genomes*. This ensures the population discovers architectures intrinsically capable of adaptation, rather than overfitting a fragile set of memorized historical price patterns.

---

## 4. Scientific Integrity & The Anti-Hype Ethos

In an industry often saturated with hyperbole surrounding Artificial Intelligence and quantitative algorithmic performance, Midas commits to brutal intellectual transparency:

1. **Explicit Reporting of Limitations & Bias**: Every experimental report explicitly documents structural shortfalls—such as survivorship bias in modern liquid asset selection (e.g., using current S&P 500 components over past decades) or synthetic horizon limitations.
2. **Separating Mechanics from Intelligence**: Demonstrating that PPO head weights update cleanly or that evolutionary crossover successfully spawns descendants represents a *mechanical verification*, never a claim of generalized trading skill.
3. **Embracing Negative Experimental Findings**: As demonstrated in the **Prompt 9 Walk-Forward Screening**, when rigorous out-of-sample testing across 2,520 evaluation tasks revealed that neural PPO agents underperformed simple Equal-Weight portfolios and failed to generate positive median alpha across diverse regime shifts, the result was celebrated as proof of protocol rigor. Nothing was massaged, data-mined, or obfuscated to manufacture false profitability.

---

## 5. Summary Goal

The ultimate goal of Midas is to establish a rock-solid, fully verifiable artificial market laboratory where cognitive architectures, evolutionary selection pressures, and lifelong reinforcement learning can be isolated, scrutinized, and tested with unwavering scientific rigor.
