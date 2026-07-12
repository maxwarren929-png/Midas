# Midas Agent Brain and Learning System Specification

**Version 1.0**

## 1. Purpose

The Agent Brain and Learning System defines how each Midas agent:
- Receives market and portfolio information.
- Processes information using a neural network.
- Chooses target portfolio allocations.
- Stores short-term internal memory.
- Explores alternative actions.
- Learns during its lifetime through reinforcement learning.
- Separates inherited traits from learned experience.
- Exposes its genetic state to the Evolution System.
- Saves and restores its complete state.

The system must be implemented in Rust using the Burn machine-learning framework.

This specification does not define:
- Historical market-data processing.
- Trade execution.
- Portfolio accounting.
- Bankruptcy calculations.
- Parent selection.
- Reproduction.
- Crossover.
- Population-level mutation.

Those responsibilities belong to the Market and Simulation Specification and the Evolution, Reproduction and Experiment Specification.

## 2. Core Concept

Every agent has two related but separate brain states:

### 2.1 Genome brain

The genome brain contains the inherited starting parameters of the agent.

It includes:
- Initial neural-network parameters.
- Inherited behavioural traits.
- Learning hyperparameters.
- Exploration settings.
- Memory settings.
- Risk preferences.

The genome brain is created at birth through two-parent reproduction.

### 2.2 Lifetime brain

The lifetime brain begins as a copy of the genome brain. During the agent's life, reinforcement learning may modify its parameters.

The lifetime brain:
- Makes trading decisions.
- Learns from rewards.
- Stores temporary memory.
- Is discarded when the agent dies, unless an experiment explicitly enables learned-weight inheritance.

The distinction must be represented clearly in code.

```rust
struct AgentBrainState {
    genome: BrainGenome,
    lifetime: LifetimeBrain,
}
```

The exact types may differ, but inherited and lifetime parameters must not be silently merged.

## 3. Design Principles

### 3.1 Evolution creates instincts

Evolution determines the agent's initial brain and inherited traits. These inherited properties represent strategies or predispositions available at birth.

### 3.2 Reinforcement learning creates experience

Lifetime learning allows the agent to adapt to market conditions encountered after birth. Learning must not directly modify the stored genome in V1.

### 3.3 Fixed architecture for Version 1

Every V1 agent must use the same neural-network architecture.

Agents may differ in:
- Weights.
- Biases.
- Learning rate.
- Exploration level.
- Risk-related traits.
- Memory behaviour.

Agents may not differ in:
- Number of inputs.
- Number of outputs.
- Number of layers.
- Tensor shapes.
- Hidden-layer dimensions.

A fixed architecture makes:
- Two-parent crossover safer.
- Batched execution easier.
- GPU processing easier.
- Checkpointing more reliable.
- Testing simpler.

Architecture evolution is deferred.

### 3.4 Valid actions over unrestricted outputs

The neural network must not directly create arbitrary trade orders. It produces scores that are transformed into valid long-only portfolio allocations.

### 3.5 Learning must not violate simulation rules

The brain may suggest actions, but the market system remains authoritative. The brain cannot:
- Create cash.
- Hold negative quantities.
- Trade unavailable assets.
- Access future prices.
- Bypass fees.
- Prevent bankruptcy.

## 4. Technology Requirements

The brain system must use:
- Rust.
- Burn.
- Serde-compatible stable genome representations.
- Deterministic seeded randomness.
- A configurable CPU-compatible backend.
- Optional GPU acceleration.

The implementation must isolate Burn-specific model objects from the stable serialised genome format where practical. This reduces the risk of saved experiments becoming unusable after a Burn API or record-format change.

The system should provide conversion functions similar to:

```rust
fn build_lifetime_brain(
    genome: &BrainGenome,
    device: &Device,
) -> Result<LifetimeBrain>;

fn extract_parameters(
    brain: &LifetimeBrain,
) -> Result<BrainParameters>;
```

## 5. Version 1 Scope

Midas V1 must support:
- A fixed feed-forward actor-critic neural architecture.
- A small recurrent or memory component.
- Continuous portfolio-allocation outputs.
- Long-only actions.
- Lifetime reinforcement learning.
- Policy-gradient-based optimisation.
- Exploration.
- Configurable learning rates.
- Gradient clipping.
- Reward normalisation.
- Batched transition learning.
- Deterministic evaluation mode.
- Brain checkpoints.
- Genome and lifetime separation.

Midas V1 must not support:
- Evolving network topology.
- Transformer architectures.
- Language-model inputs.
- Raw news or text processing.
- Intraday recurrent sequences.
- Multi-GPU distributed training.
- Direct real-money execution.
- Online learning from live brokerage accounts.
- Arbitrary code generation by agents.
- Unlimited memory.
- Inheritance of the full lifetime brain by default.

## 6. Observation Structure

The Market and Simulation System produces a legal observation for each agent. The brain must receive a fixed-size observation vector.

The complete observation contains:

```
market features
portfolio features
agent-state features
memory features
```

All observations must use finite, normalised values. No raw unbounded value should enter the network without scaling or clipping.

## 7. Market Features

For every configured asset, V1 should include a fixed feature set.

Recommended per-asset features:

```
1-day return
5-day return
20-day return
price / 20-day moving average
price / 50-day moving average
20-day realised volatility
volume change
volume / rolling average volume
distance from 20-day high
tradability flag
```

That gives approximately 10 features per asset. For 10 assets:

```
10 assets × 10 features = 100 market inputs
```

The exact set must be configurable but fixed for an experiment. All feature definitions must come from the Market and Simulation System and must obey leakage-prevention rules. The brain must not independently query raw future data.

## 8. Global Market Features

The observation may include a small set of global features shared across all assets.

Recommended V1 features:

```
benchmark 1-day return
benchmark 20-day return
benchmark volatility
average asset return
average asset volatility
fraction of assets above moving average
current episode progress
```

These features may help the agent recognise broad market regimes. Episode progress must not reveal an absolute future date in a way that allows simple date memorisation. It may be represented as a bounded relative value from the beginning of the current episode.

## 9. Portfolio Features

Each agent must receive information about its own portfolio.

Recommended portfolio inputs:

```
cash allocation
current allocation per asset
unrealised return per asset
portfolio 1-day return
portfolio 20-day return
current drawdown
distance from bankruptcy threshold
recent turnover
recent transaction-cost ratio
```

Absolute dollar wealth should not normally be used directly. Prefer relative values such as:

```
net worth / starting capital
cash / net worth
drawdown percentage
```

This prevents the brain from behaving differently merely because one agent has existed longer or has more nominal dollars.

## 10. Agent-State Features

The observation may include:

```
normalised age
recent reward
rolling average reward
rolling reward volatility
recent learning loss
exploration level
survival-cost burden
```

The brain must not receive ancestry identifiers or unique agent IDs as normal inputs. Those values could encourage meaningless memorisation.

## 11. Memory System

Midas V1 should use a small learned recurrent state.

Recommended implementation:
- One GRU layer, or
- A fixed-size recurrent hidden-state update implemented through Burn.

Recommended hidden memory size:

```
32 values
```

The memory state allows an agent to recognise temporal patterns not fully represented by rolling indicators.

The memory state:
- Begins as zeros at birth.
- Updates once per market observation.
- Persists across trading days.
- Is reset at the beginning of a new independent episode unless continuation is explicitly intended.
- Is not part of the inherited genome.
- Is saved in checkpoints.
- Is discarded at death.

The network must not receive an unbounded history of past observations.

## 12. Recommended Neural Architecture

Midas V1 should use an actor-critic model.

A recommended structure is:

```
Observation vector
       ↓
Input normalisation
       ↓
Dense layer: 256 units
       ↓
Layer normalisation
       ↓
Activation
       ↓
Dense layer: 128 units
       ↓
Activation
       ↓
GRU or recurrent memory: 32 units
       ↓
Shared latent representation
    ────────────────
   ↓                ↓
Actor head      Critic head
   ↓                ↓
Allocation logits  State value
```

Recommended activation:

```
GELU, SiLU, or ReLU
```

One activation must be selected and fixed in the configuration.

The critic outputs one scalar estimate of expected future reward. The actor outputs:

```
N asset logits
+ 1 cash logit
```

For 10 assets: **11 actor outputs**

## 13. Action Transformation

Actor outputs must be transformed into valid target allocations.

Use a softmax transformation:

```
allocation_i = softmax(logit_i)
```

This guarantees:
- `allocation_i >= 0`
- `sum of allocations = 1`

The action therefore automatically satisfies:
- Long-only trading.
- No leverage.
- No negative cash allocation.
- A valid total portfolio weight.

Unavailable assets must be masked before softmax. A masked asset must receive an effectively impossible logit. After masking, allocations must be renormalised.

If all assets are unavailable, the result must be:

```
100% cash
```

## 14. Maximum Position Constraint

Although softmax produces valid allocations, an agent could allocate nearly all wealth to one asset. Midas should support a configurable maximum asset allocation.

Recommended default:

```
maximum allocation per asset = 50%
```

Any excess must be redistributed across:
- Other available assets, and
- Cash.

The exact redistribution algorithm must be deterministic. Experiments may disable this limit. The selected limit must be reported because it materially affects strategy and survival.

## 15. Action Frequency

The brain produces one target allocation per trading day. The network may choose to maintain its current portfolio by outputting allocations close to its existing allocations.

The brain does not directly choose:
- Number of shares.
- Exact order sequencing.
- Commission.
- Execution price.

Those are handled by the market system.

## 16. Stochastic and Deterministic Modes

The actor must support two decision modes.

### 16.1 Training mode

During lifetime learning, actions may be stochastic. Exploration may be produced through:
- Sampling from a probability distribution around allocation logits.
- Adding controlled noise to logits.
- A Dirichlet allocation distribution.
- Temperature-scaled action sampling.

The chosen mechanism must produce valid allocations.

### 16.2 Evaluation mode

During validation and final testing:
- Exploration must be disabled.
- The policy must act deterministically.
- Lifetime learning should normally be frozen unless the experiment explicitly evaluates online adaptation.
- Random sampling must not affect actions.

This prevents lucky exploration from being mistaken for a reliable strategy.

## 17. Recommended Reinforcement-Learning Algorithm

Midas V1 should use an actor-critic algorithm.

The recommended initial implementation is:

```
Proximal Policy Optimisation with continuous portfolio-allocation actions.
```

PPO is recommended because it is:
- More stable than basic policy gradients.
- Suitable for actor-critic learning.
- Compatible with batched experience.
- Common enough to test and reason about.
- Able to support controlled policy updates.

A simpler advantage actor-critic implementation may be used for the first compiling milestone, but the final V1 target should be PPO.

## 18. Transition Records

For every learning step, the agent should record a transition containing:

```rust
struct Transition {
    observation: Observation,
    memory_state: MemoryState,
    action: AllocationAction,
    log_probability: f32,
    value_estimate: f32,
    reward: f32,
    next_observation: Observation,
    next_memory_state: MemoryState,
    done: bool,
}
```

The exact representation may differ. Stored transitions must contain all values required to reproduce the learning update. Transitions must not include direct mutable references to simulation state.

## 19. Reward Definition

The reward must encourage sustainable growth rather than reckless one-time profit.

The base reward should use change in logarithmic net worth:

```
base reward = ln(current net worth / previous net worth)
```

This is preferred over raw dollar change because it:
- Scales across different wealth levels.
- Treats percentage gains consistently.
- Strongly penalises approaching zero.
- Makes rewards more numerically manageable.

The final reward may include additional penalties:

```
reward = log net-worth return
         - drawdown penalty
         - turnover penalty
         - risk penalty
         - invalid-action penalty
```

Transaction costs must not be subtracted twice. Because costs already reduce net worth, they are already reflected in the base reward. A separate turnover penalty may still be used to discourage unstable overtrading, but it must be clearly documented.

## 20. Recommended Reward Components

### 20.1 Net-worth return

Primary component:

```
log(current net worth / previous net worth)
```

### 20.2 Drawdown penalty

Apply an additional penalty when current drawdown exceeds a configurable threshold.

Example:

```
drawdown penalty weight = 0.1
```

### 20.3 Turnover penalty

Penalise unnecessary portfolio rebalancing.

Example:

```
turnover penalty = turnover × configured weight
```

### 20.4 Concentration penalty

Optionally penalise extreme concentration. This should be disabled when the maximum-position rule already enforces sufficient diversification.

### 20.5 Invalid-action penalty

If the network produces:
- NaN.
- Infinity.
- Invalid logits.
- An impossible distribution.

the market system uses the safe cash fallback. The learning system should also assign a configurable negative reward.

### 20.6 Bankruptcy terminal penalty

When an agent dies, apply a terminal penalty. Recommended:

```
-1.0
```

The exact value must be configurable and scaled consistently with normal rewards.

## 21. Reward Clipping and Normalisation

Rewards may be normalised using statistics collected from the current training experience.

Permitted techniques include:
- Running mean and variance.
- Advantage normalisation.
- Bounded clipping.

Normalisation must not use future validation or test results.

Recommended reward clipping:

```
[-5.0, 5.0]
```

Aggressive clipping should be avoided because it may remove meaningful differences between severe losses and small losses. The raw reward must still be recorded for research analysis.

## 22. Advantage Estimation

PPO should use Generalised Advantage Estimation.

Required configurable values:

```
discount factor, gamma
GAE factor, lambda
```

Recommended initial values:

```
gamma = 0.99
lambda = 0.95
```

These values are defaults, not permanent truths.

Advantage calculations must correctly handle:
- Normal transitions.
- Episode endings.
- Bankruptcy.
- Checkpoint boundaries.
- Truncated learning batches.

## 23. PPO Objective

The actor loss must use a clipped policy objective. The critic loss must compare predicted and observed returns.

The total loss should include:

```
actor policy loss
+ critic value loss
+ entropy bonus
```

Configurable parameters should include:

```
PPO clipping range
critic loss coefficient
entropy coefficient
number of optimisation epochs
minibatch size
```

Recommended initial values:

```
PPO clip range = 0.2
critic coefficient = 0.5
entropy coefficient = 0.01
optimisation epochs = 4
```

All values must live in configuration rather than being scattered through the code.

## 24. Exploration

Exploration allows agents to discover strategies beyond their current deterministic preference.

Inherited exploration traits may include:

```
initial action temperature
minimum action temperature
temperature decay
entropy preference
noise scale
```

Exploration must generally decline with age or experience, but this decline may itself be controlled by inherited genes.

Exploration must never:
- Create invalid allocations.
- Bypass asset masks.
- Create leverage.
- Alter market data.
- Access another agent's state.

## 25. Learning Frequency

The agent should not update its neural network after every single market day in an inefficient one-sample step.

Recommended V1 design:

```
collect 64–256 transitions
then perform a PPO update
```

Possible default:

```
rollout length = 128 trading steps
```

For agents that die before filling the buffer:
- Mark the final transition as terminal.
- Perform an update using the available transitions if the minimum batch size is reached.
- Otherwise discard or combine according to one documented policy.

V1 recommendation: Perform a final smaller update when at least 16 transitions exist.

## 26. Optimiser

Use an optimiser supported by Burn.

Recommended initial optimiser:

```
AdamW
```

Configurable values:

```
learning rate
weight decay
epsilon
gradient clipping threshold
```

Recommended initial learning rate:

```
0.0003
```

Each agent may inherit a slightly different learning rate. Inherited learning rates must be bounded to a safe range.

Example:

```
minimum = 0.00001
maximum = 0.01
```

## 27. Gradient Safety

The training system must protect against unstable updates.

Required protections:
- Reject non-finite losses.
- Reject non-finite gradients.
- Apply global gradient-norm clipping.
- Validate parameters after an update.
- Restore the previous valid parameters if an update corrupts the model.
- Log failed updates.

Recommended gradient norm limit:

```
1.0
```

A failed learning update must not immediately crash the entire experiment unless repeated failures indicate systemic corruption.

## 28. Brain Parameters

The fixed neural-network parameters include:
- Dense-layer weights.
- Dense-layer biases.
- Recurrent-layer parameters.
- Layer-normalisation parameters.
- Actor-head parameters.
- Critic-head parameters.

The genome must contain a stable serialisable representation of all inherited model parameters.

Example conceptual structure:

```rust
struct BrainGenome {
    architecture_version: u32,
    parameters: BrainParameterGenome,
    traits: LearningTraits,
}
```

The stable representation should not depend unnecessarily on private Burn internals.

## 29. Inherited Learning Traits

Each genome should include bounded inherited traits.

Recommended traits:

```
learning rate
discount factor
GAE lambda
entropy coefficient
action temperature
temperature decay
risk-penalty sensitivity
drawdown-penalty sensitivity
turnover-penalty sensitivity
memory-decay setting
maximum target concentration
```

Not every reinforcement-learning setting needs to evolve. Some global values should remain fixed for fair comparisons.

For V1, recommended evolvable traits are:

```
learning rate
entropy coefficient
action temperature
temperature decay
drawdown sensitivity
turnover sensitivity
maximum target concentration
```

Recommended globally fixed traits are:

```
PPO clip range
rollout length
optimisation epochs
network dimensions
```

## 30. Trait Bounds

All inherited traits must have hard bounds.

Example:

```
learning rate:          0.00001 to 0.01
entropy coefficient:    0.0 to 0.1
action temperature:     0.1 to 3.0
temperature decay:      0.90 to 1.0
drawdown sensitivity:   0.0 to 2.0
turnover sensitivity:   0.0 to 1.0
maximum allocation:     0.1 to 1.0
```

Mutation or crossover results outside the allowed range must be clamped or rejected according to one deterministic rule.

## 31. Birth Initialisation

At birth:
1. The Evolution System creates a genome.
2. The brain system validates the genome.
3. The lifetime model is built from the genome parameters.
4. Optimiser state is initialised.
5. Memory state is set to zero.
6. Exploration state is initialised from inherited traits.
7. Transition buffers are empty.
8. Lifetime counters are zero.
9. The agent enters training mode.

A malformed genome must not enter the simulation.

## 32. Learned-Weight Inheritance

Midas V1 uses Darwinian inheritance by default. The child inherits from the parents' genomes, not their fully learned lifetime brains.

Therefore:

```
parent lifetime learning
does not directly overwrite
parent reproductive genome
```

This creates a clean distinction:
- Evolution discovers good starting brains.
- Reinforcement learning discovers useful lifetime adaptations.

A future experiment may enable Lamarckian inheritance, where some lifetime weights are written back into reproductive material. That feature must be explicitly disabled by default.

## 33. Evaluation Behaviour

The brain must support a frozen evaluation mode.

In frozen mode:
- No optimiser updates occur.
- No exploration noise is added.
- Dropout, if used, is disabled.
- Actions are deterministic.
- Memory may still update during the episode.
- Model parameters remain unchanged.
- Reward may still be measured.
- Loss values may be calculated for diagnostics without applying gradients.

Validation and final test results must state whether lifetime adaptation was enabled.

The default final-test mode is:

```
deterministic policy
frozen weights
active recurrent memory
```

## 34. Batched Agent Execution

Because all agents use the same architecture, observations should eventually be processed in batches.

Conceptual input shape:

```
[batch_size, observation_size]
```

Recurrent state shape:

```
[batch_size, memory_size]
```

Batched execution should support:
- CPU backend.
- GPU backend.
- Deterministic result ordering.
- Dead-agent masking.
- Different agent parameters.

Because each agent has distinct parameters, naive batching across agents may be difficult. V1 may execute agents independently while batching transitions within each agent. A future optimisation may group genetically identical or compatible networks. Correctness takes priority over GPU utilisation.

## 35. Parallelism Rules

Agent inference may run in parallel. However:
- Each agent owns its lifetime model.
- Each agent owns its optimiser.
- Each agent owns its memory.
- Each agent owns its transition buffer.
- No agent may mutate another agent's brain.
- Results must be collected in stable agent order.

Learning updates may also run in parallel when each brain is independently owned. Thread scheduling must not change random sampling. Each agent must use its own derived random seed.

## 36. Brain Checkpoints

A complete brain checkpoint must include:

```
agent identifier
architecture version
genome parameters
lifetime parameters
optimiser state
recurrent memory state
exploration state
transition buffer
learning counters
random-number-generator state
recent losses
recent rewards
```

Loading a checkpoint must verify:
- Correct architecture version.
- Matching observation size.
- Matching asset count.
- Compatible Burn model structure.
- Finite parameters.
- Valid trait ranges.

Incompatible checkpoints must fail with a clear error.

## 37. Brain Versioning

The brain format must contain an explicit version number.

Example:

```
architecture_version = 1
genome_format_version = 1
checkpoint_format_version = 1
```

Changing layer sizes or parameter ordering requires a new architecture version. The program must not silently load parameters into a differently shaped network.

## 38. Invalid Observation Handling

The market system should already validate observations. The brain must still protect itself against:
- NaN.
- Infinity.
- Incorrect vector length.
- Missing recurrent state.
- Invalid tradability masks.

Default behaviour:
1. Reject the observation.
2. Produce a safe all-cash action.
3. Skip the learning transition.
4. Log the error.
5. Continue only if the simulation considers the error recoverable.

Repeated invalid observations should stop the experiment.

## 39. Invalid Output Handling

After inference, validate:
- All logits are finite.
- Critic value is finite.
- Memory state is finite.
- Allocation transformation succeeds.
- Allocation sum is approximately one.

If validation fails:

```
action = 100% cash
```

The failed inference must be logged. The transition should include an invalid-action flag so the reward system can apply a penalty.

## 40. Brain Metrics

Each agent should track:

```
number of inference steps
number of learning updates
mean actor loss
mean critic loss
mean entropy
mean total loss
mean gradient norm
failed updates
invalid actions
current exploration temperature
mean predicted value
mean realised return
mean advantage
```

These values help distinguish:
- Good trading behaviour.
- Unstable learning.
- Excessive exploration.
- Critic failure.
- Vanishing or exploding gradients.

## 41. Population-Level Learning Metrics

The system should expose aggregate values such as:

```
average policy entropy
average learning rate
average actor loss
average critic loss
fraction of invalid actions
fraction of failed updates
mean genome-to-lifetime parameter distance
```

Genome-to-lifetime distance measures how far an agent's learned brain has moved from its inherited starting point.

This may reveal whether:
- Evolution is producing useful initial brains.
- Agents require major lifetime adaptation.
- Learning is damaging inherited strategies.

## 42. Determinism

Given the same:
- Genome.
- Observation sequence.
- Random seed.
- Burn backend.
- Device type.
- Configuration.
- Program version.

the brain should produce reproducible behaviour.

Exact floating-point identity across CPU and GPU backends is not guaranteed. Therefore, experiment records must include:

```
backend
device
precision
Burn version
compiler version
```

Determinism tests should compare runs on the same backend and device class.

## 43. Numerical Precision

Midas V1 should use:

```
f32
```

for neural-network tensors. Portfolio accounting remains defined separately by the market system.

Observation values should be converted safely from market f64 values into bounded f32 values. Conversions must reject non-finite or out-of-range values. Mixed-precision training is deferred.

## 44. Suggested Rust Module Structure

```
src/
├── brain/
│   ├── mod.rs
│   ├── model.rs
│   ├── actor.rs
│   ├── critic.rs
│   ├── recurrent.rs
│   ├── genome.rs
│   ├── traits.rs
│   ├── observation.rs
│   ├── action.rs
│   ├── inference.rs
│   ├── checkpoint.rs
│   └── validation.rs
├── learning/
│   ├── mod.rs
│   ├── transition.rs
│   ├── rollout.rs
│   ├── reward.rs
│   ├── advantage.rs
│   ├── ppo.rs
│   ├── optimiser.rs
│   └── metrics.rs
└── backend/
    ├── mod.rs
    ├── cpu.rs
    └── gpu.rs
```

The exact structure may change, but the following must remain separate:
- Model architecture.
- Stable genome representation.
- Inference.
- Reinforcement learning.
- Reward calculation.
- Checkpointing.
- Validation.

## 45. Configuration

Example brain configuration:

```toml
[brain]
architecture_version = 1
hidden_size_1 = 256
hidden_size_2 = 128
memory_size = 32
activation = "gelu"
maximum_allocation = 0.50

[learning]
algorithm = "ppo"
learning_rate = 0.0003
gamma = 0.99
gae_lambda = 0.95
ppo_clip = 0.20
critic_coefficient = 0.50
entropy_coefficient = 0.01
gradient_clip = 1.0
rollout_length = 128
minimum_terminal_batch = 16
optimisation_epochs = 4
reward_clip_min = -5.0
reward_clip_max = 5.0

[exploration]
initial_temperature = 1.0
minimum_temperature = 0.1
temperature_decay = 0.999

[reward]
drawdown_penalty_weight = 0.10
turnover_penalty_weight = 0.01
invalid_action_penalty = 0.05
bankruptcy_penalty = 1.0

[backend]
kind = "cpu"
device = "default"
```

Any field controlled by inherited traits must have clear precedence over global defaults.

## 46. Required Unit Tests

Required tests include:
- Correct observation-vector size.
- Correct actor-output size.
- Softmax allocations sum to one.
- Allocations are non-negative.
- Unavailable assets receive zero allocation.
- All unavailable assets produce all-cash allocation.
- Maximum-allocation limit is respected.
- Identical genome and seed produce identical actions.
- Deterministic evaluation disables exploration.
- Training mode enables exploration.
- Memory begins at zero.
- Memory changes after valid observations.
- Memory resets correctly.
- Invalid observations produce all-cash actions.
- Invalid model outputs produce all-cash actions.
- Genome parameters correctly initialise lifetime parameters.
- Lifetime learning does not mutate the genome.
- Frozen evaluation does not change weights.

## 47. Reinforcement-Learning Tests

Required tests include:
- Positive returns produce higher rewards than equal negative returns.
- Bankruptcy produces a terminal transition.
- Transaction costs affect reward through net-worth change.
- Advantage calculation matches known synthetic examples.
- PPO clipping limits excessive policy updates.
- Critic loss decreases on a simple learnable dataset.
- Entropy bonus increases exploration pressure.
- Gradient clipping limits gradient norms.
- Non-finite losses are rejected.
- Failed updates restore valid parameters.
- Rollout buffers clear correctly after updates.
- Truncated episodes bootstrap correctly.
- Terminal episodes do not bootstrap beyond death.

## 48. Integration Tests

Required scenarios:

### 48.1 Constant market

All prices remain constant. Expected behaviour:
- Trading costs harm frequent traders.
- Cash-preserving strategies survive longer.
- Learning does not invent free profit.

### 48.2 Rising market

One asset rises steadily. Expected behaviour:
- A learnable policy can increase allocation to that asset.
- The actor and critic losses remain finite.
- The trained policy outperforms random behaviour in a controlled test.

### 48.3 Falling market

All assets decline. Expected behaviour:
- The agent can learn to prefer cash.
- Long-only constraints remain intact.
- No short positions appear.

### 48.4 Regime change

The market rises and then falls. Expected behaviour:
- Recurrent memory updates.
- Lifetime learning can alter behaviour.
- No future regime information appears in earlier actions.

### 48.5 Bankruptcy

A strategy repeatedly trades badly until reaching the death threshold. Expected behaviour:
- The final transition is terminal.
- The bankruptcy penalty is applied.
- Learning stops after death.
- The dead brain no longer produces actions.

## 49. Property Tests

Property-based tests should verify:
- Every valid action sums approximately to one.
- No valid action contains a negative allocation.
- Masked assets never receive allocation.
- Finite observations produce finite outputs under valid parameters.
- Learning never changes tensor shapes.
- Genome parameter shapes remain fixed.
- The lifetime model cannot alter stored genome values.
- Gradient clipping never increases an already excessive norm.
- Exploration never creates an invalid portfolio.
- Checkpoint save and load preserve action output within tolerance.

## 50. Performance Requirements

The brain system should initially support:

```
500 independent agents
one daily inference per agent
10 assets
20 years of historical data
```

The implementation must avoid:
- Rebuilding models every day.
- Reallocating large tensors unnecessarily.
- Copying the complete observation history.
- Saving checkpoints every step.
- Performing one optimiser update for each individual transition.

Profiling must distinguish:

```
inference time
learning time
data-conversion time
checkpoint time
```

## 51. Debugging Modes

The system should provide:

### 51.1 Inference-only mode

Runs the brain without learning. Useful for testing evolution and market accounting.

### 51.2 Learning-disabled mode

Uses fixed lifetime weights but retains memory and exploration.

### 51.3 Deterministic mode

Disables all exploration and learning.

### 51.4 Single-agent trace mode

Logs:
- Observation summary.
- Actor logits.
- Final allocations.
- Critic value.
- Reward.
- Advantage.
- Loss.
- Gradient norm.
- Memory summary.

This mode is for debugging and should not be enabled across a large population by default.

## 52. Version 1 Completion Criteria

The Agent Brain and Learning System is complete when:
1. A fixed Burn actor-critic model can be created from a genome.
2. The model receives the complete fixed-size observation vector.
3. The model outputs valid cash and asset allocations.
4. Tradability masks are enforced.
5. Long-only and no-leverage constraints are guaranteed.
6. Recurrent memory persists correctly.
7. Training and evaluation modes behave differently.
8. PPO updates the lifetime brain.
9. The genome remains unchanged during lifetime learning.
10. Rewards use relative net-worth change.
11. Bankruptcy produces a terminal learning event.
12. Invalid observations and outputs safely fall back to cash.
13. Brain checkpoints save and restore correctly.
14. Learning metrics are recorded.
15. Required unit, integration, property, and RL tests pass.
16. The system runs on a CPU backend.
17. A GPU backend can be enabled without redesigning the model.
18. The brain integrates with the Market and Simulation System through stable interfaces.

## 53. Future Extensions

The following are deferred:
- Evolving neural architecture.
- Attention mechanisms.
- Transformer-based market models.
- Multiple timescale recurrent modules.
- Intraday observations.
- Raw order-book inputs.
- News and text processing.
- Fundamental-data encoders.
- Communication between agents.
- Agents observing other agents.
- Shared population critics.
- Centralised multi-agent training.
- Offline pretraining.
- Imitation learning.
- Meta-learning.
- Learned reward models.
- Full Lamarckian inheritance.
- Neuromodulated plasticity.
- Continual live-market adaptation.
- Multi-GPU training.
- Mixed-precision training.

These features must not be partially introduced into V1 without updating crossover, checkpointing, determinism, and evaluation rules.

## 54. Final System Definition

The Midas Agent Brain and Learning System is:

> A fixed-architecture Burn neural actor-critic that transforms legally timed market observations into valid long-only portfolio allocations, maintains recurrent lifetime memory, learns through reinforcement learning, and keeps inherited genome parameters separate from acquired experience.

Its purpose is not simply to predict tomorrow's stock price. Its purpose is to create agents capable of inheriting strategies, adapting during life, acting under risk, and producing behaviour that evolution can select.
