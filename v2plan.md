# Midas V2 Plan — Self-Awareness & Evolvable Traits

## Goal

Close two spec gaps that are low engineering cost but high potential impact:

1. **Agent self-awareness observations** — give agents signal about their own performance trajectory
2. **Evolvable traits** — let evolution tune per-agent behavioral parameters instead of using identical fixed values

## Implementation status

Implemented in the current working tree through the opt-in
`configs/v2-evolvable.toml` configuration. The awareness provider, persistent
reward statistics, policy temperature contract, trait mutation behavior, and
trait-sensitive reward/trading paths are wired and tested. Awareness runs use
checkpoint format 14; existing baseline configurations retain format 13 and
legacy checkpoints are migrated with zeroed awareness state. The six-condition
experiment protocol below is still an experiment plan, not completed result
data.

Both work within the existing frozen-encoder architecture and require no
changes to the neural network or PPO head-training code. Awareness state does
require a versioned checkpoint envelope; the implementation keeps baseline
format 13 writes stable and uses format 14 for awareness schemas.

---

## Feature 1: Agent Self-Awareness Observations

### Motivation

The Brain spec (§10) calls for 7 agent-state features. Only `normalized_age` is implemented. The agent currently has no signal for whether its decisions are working, whether its performance is improving or declining, or how aggressively it's exploring. This is the simplest observation gap to close and a prerequisite for any future emotion-drive system.

### New Observation Signals

| Signal | Formula | What it tells the agent |
|--------|---------|------------------------|
| `recent_reward` | Last step's clipped reward | "Was my most recent decision good or bad?" |
| `rolling_mean_reward` | EMA of recent rewards (α=0.05, ~20-step lookback) | "Am I generally profitable or losing?" |
| `rolling_reward_volatility` | EMA of squared deviation from mean | "Are my results consistent or chaotic?" |
| `exploration_level` | Current temperature / initial temperature | "How much am I exploring vs exploiting?" |
| `survival_cost_ratio` | Daily survival cost / current net worth | "How much pressure am I under from costs?" |
| `reward_trend` | Sign of (short EMA - long EMA of reward) | "Are things getting better or worse?" |

### Implementation

**New file: `crates/midas-core/src/observation/providers/agent_awareness.rs`**

```rust
pub struct AgentAwarenessProvider {
    metadata: Vec<ObservationMetadata>,
    outputs: Vec<(String, AwarenessFeatureSpec)>,
}

impl ObservationProvider for AgentAwarenessProvider {
    fn observe(&self, context: &ObservationContext, sink: &mut ObservationSink) -> Result<()> {
        // Publish each configured awareness signal from context fields
    }
}
```

**Changes to `ObservationContext` (`contracts.rs`):**

Add fields the provider needs to read:

```rust
pub struct ObservationContext<'a> {
    // ... existing fields ...
    pub recent_reward: f64,              // NEW: last step's reward
    pub rolling_mean_reward: f64,        // NEW: EMA of rewards
    pub rolling_reward_volatility: f64,  // NEW: EMA of reward variance
    pub current_temperature: f32,        // NEW: current action temperature
    pub initial_temperature: f32,        // NEW: birth temperature
    pub daily_survival_cost: f64,        // NEW: from config
}
```

**Changes to `AgentState` (`agent.rs`):**

Add rolling statistics that persist across steps:

```rust
pub struct AgentState {
    // ... existing fields ...
    pub reward_ema: f64,              // NEW: exponential moving average
    pub reward_variance_ema: f64,     // NEW: EMA of squared deviation
    pub last_reward: f64,             // NEW: most recent reward
}
```

**Changes to simulation step (`advance.rs` / `runner.rs`):**

After computing reward each step, update the agent's rolling stats before the next observation:

```rust
let alpha = 0.05; // ~20-step lookback
agent.state.last_reward = reward;
agent.state.reward_ema = alpha * reward + (1.0 - alpha) * agent.state.reward_ema;
let deviation = (reward - agent.state.reward_ema).powi(2);
agent.state.reward_variance_ema = alpha * deviation + (1.0 - alpha) * agent.state.reward_variance_ema;
```

**Config additions (`v2-evolvable.toml`):**

```toml
[observation]
features = [
    # ... existing features ...
    "recent_reward",
    "rolling_mean_reward",
    "rolling_reward_volatility",
    "exploration_level",
    "survival_cost_ratio",
    "reward_trend",
]

[observations]
providers = ["market", "portfolio", "agent_state", "agent_awareness"]
```

**Schema changes:**

- New `AgentAwarenessFeatureSpec` enum variants in `schema.rs`
- New provider name constant `AGENT_AWARENESS_PROVIDER`
- Observation schema hash changes → checkpoint format bump to 14

### Files Changed

| File | Change |
|------|--------|
| `src/observation/providers/agent_awareness.rs` | **New file** — provider implementation |
| `src/observation/providers/mod.rs` | Register new provider |
| `src/observation/schema.rs` | Add awareness feature specs and names |
| `src/observation/contracts.rs` | Extend `ObservationContext` with new fields |
| `src/agent.rs` | Add rolling stats to `AgentState` |
| `src/simulation/advance.rs` | Update rolling stats after reward computation |
| `src/simulation/decision.rs` | Pass new context fields when building observations |
| `configs/v2-evolvable.toml` | Opt in to the new features and provider |

**Estimated effort:** ~300 lines of new code, ~50 lines of changes to existing files.

### Testing

1. **Unit test:** Provider publishes correct values for known context
2. **Unit test:** Rolling EMA converges to expected value on constant reward stream
3. **Unit test:** Exploration level is 1.0 at birth, decays with temperature
4. **Integration test:** Schema hash changes when provider is added
5. **Regression test:** Existing configs without the provider continue to work (provider is opt-in)

---

## Feature 2: Evolvable Traits

### Motivation

The infrastructure for evolvable traits is fully built — genome storage, crossover, mutation, trait bounds, validation. But every trait has `evolvable = false` and `mutation_scale = 0.0`. Evolution currently operates only on random neural network weights, which (as Prompt 9 showed) produces near-zero impact. Enabling trait evolution gives evolution something meaningful and immediately functional to select on.

### What Gets Turned On

All seven traits become evolvable with configured mutation scales:

| Trait | Default | Mutation Scale | Bounds | Effect |
|-------|---------|---------------|--------|--------|
| `learning_rate` | 0.0003 | 0.3 (log-scale) | 0.00001 – 0.01 | Fast vs slow learners |
| `entropy_coefficient` | 0.01 | 0.2 | 0.0 – 0.1 | High vs low exploration pressure |
| `action_temperature` | 1.0 | 0.15 | 0.1 – 3.0 | Conservative vs aggressive trading |
| `temperature_decay` | 0.999 | 0.05 | 0.9 – 1.0 | Fast vs slow exploration reduction |
| `drawdown_sensitivity` | 0.1 | 0.3 | 0.0 – 2.0 | Risk-tolerant vs risk-averse |
| `turnover_sensitivity` | 0.01 | 0.3 | 0.0 – 1.0 | Frequent vs infrequent traders |
| `maximum_allocation` | 0.5 | 0.15 | 0.01 – 1.0 | Concentrated vs diversified |

### Implementation

The existing genome and replacement infrastructure is enabled through config.
Two small runtime integrations make the traits effective: neural trade caps
use inherited `maximum_allocation`, and risk-adjusted reward can consume the
inherited drawdown/turnover sensitivities. Learning rate mutation uses the
planned log-space noise; the remaining traits use bounded additive noise.

**New config file: `configs/v2-evolvable.toml`**

```toml
[brain.traits.learning_rate]
initial = 0.0003
minimum = 0.00001
maximum = 0.01
mutation_scale = 0.3
evolvable = true

[brain.traits.entropy_coefficient]
initial = 0.01
minimum = 0.0
maximum = 0.1
mutation_scale = 0.2
evolvable = true

[brain.traits.action_temperature]
initial = 1.0
minimum = 0.1
maximum = 3.0
mutation_scale = 0.15
evolvable = true

[brain.traits.temperature_decay]
initial = 0.999
minimum = 0.9
maximum = 1.0
mutation_scale = 0.05
evolvable = true

[brain.traits.drawdown_sensitivity]
initial = 0.1
minimum = 0.0
maximum = 2.0
mutation_scale = 0.3
evolvable = true

[brain.traits.turnover_sensitivity]
initial = 0.01
minimum = 0.0
maximum = 1.0
mutation_scale = 0.3
evolvable = true

[brain.traits.maximum_allocation]
initial = 0.5
minimum = 0.01
maximum = 1.0
mutation_scale = 0.15
evolvable = true
```

### Dependencies

- **Requires evolutionary reproduction** — in the current implementation this
  is selected by `evolution.enabled = true`; the configured replacement enum
  remains `non_evolutionary_random_genome` because the engine selects its
  stateful evolutionary strategy from the evolution switch.
- **Use `risk_adjusted_return` to exercise risk traits** — log-return runs keep
  the baseline reward path, while risk-adjusted runs consume inherited
  drawdown and turnover sensitivities.

### What To Watch For

- **Trait convergence:** Do traits converge to similar values across the population (selection pressure) or stay diverse (neutral drift)?
- **Trait-performance correlation:** Do agents with certain trait profiles (e.g., low learning rate + high drawdown sensitivity) consistently survive longer?
- **Reporting:** The evolution report already records genome traits. Add trait statistics (mean, std, min, max per trait) to the experiment summary.

### Testing

1. **Experiment:** Run a 4-condition comparison (control / PPO / PPO+evo-weights-only / PPO+evo-traits) on the Prompt 3B market to isolate the effect of trait evolution
2. **Diagnostic:** After 1,259 steps, dump population trait distributions and check for convergence vs initial uniform distribution
3. **Regression:** Verify that `evolvable = false` configs produce identical results to current baseline

---

## Combined V2 Experiment Protocol

### Conditions (6 total)

| Condition | Learning | Evolution | Traits Evolvable | Self-Awareness |
|-----------|----------|-----------|-----------------|----------------|
| control | none | off | n/a | off |
| ppo-baseline | PPO | off | n/a | off |
| ppo-aware | PPO | off | n/a | **on** |
| ppo-evo-baseline | PPO | on | off | off |
| ppo-evo-traits | PPO | on | **on** | off |
| ppo-evo-full-v2 | PPO | on | **on** | **on** |

### Market

Use the Prompt 3B frozen 1,800-date synthetic market for direct comparison with existing results.

### Seeds

10 seeds (41–50) per condition = 60 runs total.

### Success Criteria

1. `ppo-aware` > `ppo-baseline` on mean test NW (self-awareness helps)
2. `ppo-evo-traits` > `ppo-evo-baseline` on mean test NW (trait evolution helps)
3. `ppo-evo-full-v2` > all others (combined effect)
4. Trait distributions in evolutionary conditions show convergence (evidence of selection pressure)

### Estimated Runtime

At current scale (8 agents, ~3K params, CPU): ~2 minutes per run × 60 = **~2 hours total**. Laptop-friendly.

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Self-awareness features overfit on training split | Medium | Compare val vs test improvement; use clipped/normalized values |
| Trait evolution produces degenerate populations (all converge to extremes) | Medium | Bounds are already hard-clamped; monitor trait distributions |
| Schema hash change breaks existing checkpoints | Certain | New checkpoint format 14; old checkpoints won't resume (expected) |
| Increased observation vector size hurts frozen encoder | Low | Only 6 new scalars; observation size goes from ~54 to ~60 |
