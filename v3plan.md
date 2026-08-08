# Midas V3 Plan — Unfrozen Encoder

## Goal

Remove the single biggest architectural bottleneck in Midas: the frozen hidden/recurrent encoder. Currently only the actor and critic output heads (~1K params) are trained by PPO. The hidden layers and recurrent memory (~2K params at current scale, ~100K at spec scale) are permanently locked at their random initialization. This means the agent's learned representation of the market is literally a random projection.

V3 unfreezes the encoder so the entire network learns end-to-end via backpropagation.

**This is the highest-impact change identified in the codebase.**

## Implementation status

Phase 1 is implemented through a staged frozen-encoder autodiff boundary. The
runtime now defines separate Burn `NdArray` inference and
`Autodiff<NdArray>` training backends. Exact bounded-Beta PPO derivatives are
injected at the actor-logit boundary because Burn 0.18 does not expose
differentiable lgamma/digamma tensor operations; Burn performs the full dense
actor/critic vector-Jacobian product. Existing momentum state, checkpoint
format, inference tensors, and frozen hidden/recurrent parameters are retained.

The Phase 1 frozen-encoder boundary passed both short and complete-split
determinism gates. Phase 2 now moves raw observations and all feed-forward
hidden layers into the autodiff graph. Hidden parameters use persistent clipped
momentum updates; gradients pass through the recurrent projection, while its
weights and recurrent sample states remain frozen. Legacy checkpoints migrate
with zeroed hidden momentum, and parallel graph execution is serialized to
retain exact checkpoint/resume determinism.

Phase 3 is implemented. Phase 3A trained the legacy recurrent cell through
explicitly bounded, masked-safe sequences using truncated backpropagation
through time. Phase 3B replaced it for new genomes with a standard GRU while
retaining exact legacy-checkpoint execution. Longer truncation-window
experiments remain future work.

---

## Current Architecture (Frozen)

```
Observation (54–98 inputs)
    │
    ▼
┌─────────────────────────┐
│  Hidden Layer 1 [32]    │ ◄── FROZEN (random weights from birth)
│  GELU activation        │
├─────────────────────────┤
│  Hidden Layer 2 [16]    │ ◄── FROZEN (random weights from birth)
│  GELU activation        │
├─────────────────────────┤
│  Recurrent [8]          │ ◄── FROZEN (random weights from birth)
│  tanh activation        │     Memory state updates but weights don't
└─────────┬───────────────┘
          │ representation (8 floats)
    ┌─────┴─────┐
    ▼           ▼
┌────────┐ ┌────────┐
│ Actor  │ │ Critic │ ◄── TRAINABLE (PPO updates these)
│ head   │ │ head   │
└────────┘ └────────┘
```

The representation feeding the actor/critic is a fixed random function of the input. The heads can only learn a linear transform of this random representation — severely limiting what the agent can learn.

---

## Target Architecture (Unfrozen)

```
Observation (54–98 inputs)
    │
    ▼
┌─────────────────────────┐
│  Hidden Layer 1 [128]   │ ◄── TRAINABLE (PPO + backprop)
│  LayerNorm + GELU       │
├─────────────────────────┤
│  Hidden Layer 2 [64]    │ ◄── TRAINABLE (PPO + backprop)
│  GELU activation        │
├─────────────────────────┤
│  GRU [32]               │ ◄── TRAINABLE (PPO + BPTT)
│  gated memory updates   │
└─────────┬───────────────┘
          │ representation (32 floats)
    ┌─────┴─────┐
    ▼           ▼
┌────────┐ ┌────────┐
│ Actor  │ │ Critic │ ◄── TRAINABLE (PPO updates these)
│ head   │ │ head   │
└────────┘ └────────┘

All layers receive gradients. The network learns its own
market representation instead of using random projections.
```

Changes from current to target:
- Hidden layers scale up: [32, 16] → [128, 64] (closer to spec's [256, 128])
- Memory scales up: 8 → 32 (matches spec recommendation)
- Simple recurrent → GRU (gated memory for selective retention)
- Add LayerNorm after first hidden layer (stabilizes training)
- All layers receive gradients via backpropagation

---

## Implementation Strategy

### Option A: Extend Hand-Rolled Gradients (not recommended)

The current `head_training.rs` computes gradients manually for the actor/critic dense layers. We could extend this to all layers:

- **Dense backprop**: Straightforward chain rule through GELU activation
- **Recurrent backprop-through-time (BPTT)**: Significantly more complex — must unroll the recurrent layer across the rollout's `sequence_length` steps, accumulate gradients through time, and handle truncation

**Pros**: No new dependencies, stays within current architecture
**Cons**: Error-prone, hard to maintain, no autodiff safety net, GRU backprop by hand is ~500 lines of careful math

### Option B: Use Burn's Autodiff Backend (recommended)

Replace the hand-rolled training with Burn's actual automatic differentiation:

1. Switch from `NdArray` backend to `Autodiff<NdArray>` for training
2. Build the model using Burn's module system so parameters are tracked
3. Use Burn's `backward()` to compute gradients automatically
4. Keep the stable `BrainGenome` ↔ Burn conversion layer (already exists)

**Pros**: Correct by construction, GRU/attention/any layer "just works", easier to add future architecture changes
**Cons**: Requires rearchitecting `BurnNeuralModel`, Burn's autodiff adds memory overhead, must validate determinism

### Recommended: Option B with staged rollout

---

## Implementation Phases

### Phase 1: Burn Autodiff Integration (~1 week)

**Goal**: Replace hand-rolled gradient computation with Burn's autodiff while keeping the current architecture unchanged (still [32, 16] + recurrent 8, still head-only training initially).

**Changes to `burn_runtime.rs`:**

```rust
// Current: uses NdArray backend only
type Backend = NdArray<f32>;

// New: uses Autodiff<NdArray> for training, NdArray for inference
type TrainBackend = Autodiff<NdArray<f32>>;
type InferBackend = NdArray<f32>;
```

**Changes to `head_training.rs` → `training.rs`:**

Replace the entire manual gradient computation with:

```rust
fn train_step(
    model: &mut BurnModel<TrainBackend>,
    batch: &PpoTrainingBatch,
    config: &LearningConfig,
) -> Result<TrainingMetrics> {
    // Forward pass (Burn tracks computation graph)
    let outputs = model.forward_training(batch);

    // Compute PPO loss (same formula, now on autodiff tensors)
    let loss = ppo_loss(outputs, batch, config);

    // Backward pass (Burn computes all gradients automatically)
    let gradients = loss.backward();

    // Apply gradients with optimizer
    model = optimizer.step(learning_rate, model, gradients);

    Ok(metrics)
}
```

**Validation**: Run existing test suite. All 108+ tests must pass. Results must be identical to hand-rolled implementation (within floating-point tolerance) when training heads only.

### Phase 2: Unfreeze Hidden Layers (~3 days)

**Status**: Implemented for every feed-forward hidden layer. The fixed
recurrent projection is inside the graph but its parameters and temporal state
remain frozen until Phase 3.

**Goal**: Allow gradients to flow through all dense hidden layers.

**Change**: Remove the gradient detachment / `no_grad` barrier between the encoder and heads. In the current hand-rolled system, this is implicit (gradients are only computed for heads). With Burn autodiff, it means simply including hidden layer parameters in the optimizer's parameter set.

```rust
// Current: only head params in optimizer
let params = model.actor_params().chain(model.critic_params());

// New: all dense params in optimizer
let params = model.hidden_params()
    .chain(model.actor_params())
    .chain(model.critic_params());
```

**Risk**: Gradient flow through the entire network can destabilize training. Mitigations:
- Gradient clipping (already implemented, threshold=1.0)
- Persistent momentum state for every hidden layer (implemented)
- Exact checkpoint/resume and independent replay gates (implemented)
- LayerNorm after first hidden layer (optional future hardening)
- Lower learning rate for encoder vs heads (optional differential LR)
- Report gradient norms per layer (future observability)

### Phase 3: GRU + Backprop Through Time (~1 week)

**Status**: Implemented. New genome-format-4 agents use a packed standard GRU
in both inference and Burn autodiff. Genome-format-3 checkpoints retain their
original dense-SiLU recurrence.

**Goal**: Replace the simple recurrent layer with a GRU and enable temporal gradient flow.

**GRU advantages over simple recurrent**:
- **Gating mechanism**: Selective memory retention (forget irrelevant states, remember important ones)
- **Gradient flow**: Reset and update gates prevent vanishing gradients through long sequences
- **Proven**: Standard architecture for sequential decision-making in RL

**BPTT implementation**:

The PPO rollout already stores transitions in chronological sequence with `sequence_length = 16`. BPTT unrolls the GRU across these sequences:

```rust
fn forward_sequence(
    model: &BurnModel<TrainBackend>,
    sequence: &[Transition],  // 16 chronological steps
) -> Vec<(NeuralOutput, GruState)> {
    let mut state = sequence[0].recurrent_state;
    let mut outputs = Vec::new();
    for transition in sequence {
        let (output, new_state) = model.forward_with_state(
            &transition.observation,
            &state,
        );
        outputs.push((output, new_state.clone()));
        state = new_state;
    }
    outputs
    // Burn autodiff tracks the entire unrolled computation graph
    // backward() will flow gradients through all 16 steps
}
```

**Truncated BPTT**: Gradients flow through at most `sequence_length` steps (16 by default). This bounds memory usage and prevents vanishing gradients over very long horizons.

### Phase 4: Scale Up Architecture (~2 days)

**Goal**: Increase network capacity toward spec recommendations.

**Staged scaling**:

| Stage | Hidden Layers | Memory | Total Params | Notes |
|-------|:------------:|:------:|:------------:|-------|
| Current | [32, 16] | 8 | ~3K | Frozen encoder, too small |
| V3-small | [64, 32] | 16 | ~12K | First unfrozen test |
| V3-medium | [128, 64] | 32 | ~45K | Good balance for CPU |
| V3-spec | [256, 128] | 32 | ~105K | Full spec compliance |

Start with V3-small to validate that unfreezing works, then scale up. Each size increase is a config change:

```toml
[brain]
hidden_layers = [128, 64]
memory_size = 32
```

### Phase 5: Optimizer Upgrade — AdamW (~2 days)

**Goal**: Replace hand-rolled momentum SGD with AdamW (spec requirement).

With Burn's autodiff, this is straightforward — Burn provides `AdamW` as a built-in optimizer:

```rust
use burn::optim::AdamWConfig;

let optimizer = AdamWConfig::new()
    .with_learning_rate(config.learning_rate)
    .with_weight_decay(1e-4)
    .init();
```

**AdamW advantages over momentum SGD**:
- Per-parameter adaptive learning rates (important when unfreezing — encoder and heads may need different rates)
- Decoupled weight decay (regularization)
- Better convergence on non-stationary objectives (which RL inherently is)

---

## Impact on Existing Systems

### Genome Format

The `BrainGenome` already stores all layer weights — hidden, recurrent, actor, critic. **No genome format change needed.** The genome is the same; we're just allowing lifetime modifications to propagate to all layers instead of only the heads.

The `LifetimeParameters` struct already contains all layers too. Currently hidden/recurrent lifetime params are always identical to birth genome (because they never receive gradient updates). After V3, they'll diverge.

### Evolution

**Genome-only inheritance still works.** Offspring receive the parent's birth genome (random-init weights), not the parent's trained lifetime weights. V3 doesn't change this — it just means lifetime learning is more powerful for each individual agent.

**Lamarckian inheritance becomes much more interesting** after V3 because the encoder now contains learned representations worth inheriting. But that's a V4 concern.

### Checkpoint Format

Lifetime parameters already include hidden/recurrent layers. Checkpoint save/restore will capture the trained encoder state automatically. **No checkpoint format change needed** for unfreezing alone.

The GRU recurrent layout change produced genome format 4 and checkpoint formats
19/20. Formats 13–18 remain readable through the legacy recurrent runtime.
If network dimensions change → **architecture version bump** → new checkpoint format.

### Observation Schema

**No change.** The observation vector is the same; only the network processing it changes.

### Experiment Comparability

Results from V3 are **not directly comparable** to V2/V1 results because:
- Different network capacity (different parameter count)
- Different optimization (full backprop vs head-only)
- Different optimizer (AdamW vs SGD)

New baseline experiments must be run.

---

## V3 Experiment Protocol

### Conditions (5 total)

| Condition | Encoder | Architecture | Optimizer | Learning |
|-----------|---------|:------------:|-----------|----------|
| control | n/a | [32, 16] | n/a | none |
| v1-frozen | frozen | [32, 16] | SGD | PPO heads only |
| v3-unfrozen-small | **unfrozen** | [64, 32] | AdamW | PPO full |
| v3-unfrozen-medium | **unfrozen** | [128, 64] | AdamW | PPO full |
| v3-unfrozen-gru | **unfrozen** | [128, 64] + GRU(32) | AdamW | PPO + BPTT |

### Market

Prompt 3B frozen 1,800-date synthetic market (for comparison with V1/V2 baselines).

### Seeds

10 seeds (41–50) per condition = 50 runs.

### Success Criteria

1. `v3-unfrozen-small` > `v1-frozen` (unfreezing helps even at small scale)
2. `v3-unfrozen-medium` > `v3-unfrozen-small` (more capacity helps)
3. `v3-unfrozen-gru` > `v3-unfrozen-medium` (GRU memory helps)
4. Encoder layer gradient norms are non-trivial (gradients actually reach early layers)
5. Learned representations show structure (PCA/t-SNE of hidden activations cluster by market regime)

### Performance Budget

| Config | Est. params | Est. single-run time (CPU) | 50-run suite |
|--------|:----------:|:--------------------------:|:------------:|
| v1-frozen [32,16] | ~3K | ~2 min | ~1.5 hr |
| v3-small [64,32] | ~12K | ~5 min | ~4 hr |
| v3-medium [128,64] | ~45K | ~15 min | ~12 hr |
| v3-gru [128,64]+GRU(32) | ~50K | ~20 min | ~16 hr |

All feasible on a laptop. The medium/GRU configs would benefit from running overnight.

---

## Files Changed

| File | Phase | Change |
|------|:-----:|--------|
| `brain/burn_runtime.rs` | 1 | Dual backend (Autodiff for training, NdArray for inference) |
| `brain/head_training.rs` → `brain/training.rs` | 1 | Replace hand-rolled gradients with Burn autodiff |
| `brain/model.rs` | 1 | Update `NeuralModel` trait for autodiff compatibility |
| `brain/state.rs` | 1 | Add optimizer state for all layers (not just heads) |
| `brain/burn_runtime.rs` | 2 | Include hidden params in optimizer parameter set |
| `brain/burn_runtime.rs` | 3 | Replace simple recurrent with GRU module |
| `brain/state.rs` | 3 | GRU state layout (3× gates) |
| `brain/genome.rs` | 3 | GRU genome representation |
| `config/brain.rs` | 4 | Validate larger architectures |
| `brain/training.rs` | 5 | Integrate Burn's AdamW optimizer |
| `brain/state.rs` | 5 | AdamW optimizer state (first+second moment per param) |
| `configs/v3-*.toml` | 4 | New architecture configs |

**Estimated total effort**: ~3 weeks across all 5 phases.

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Burn autodiff introduces non-determinism | Medium | High | Pin Burn version, validate determinism with seed tests |
| Full backprop destabilizes PPO | High | Medium | Gradient clipping, differential LR, LayerNorm, monitor KL divergence |
| GRU BPTT memory usage at scale | Low | Medium | Truncated BPTT (16 steps), gradient checkpointing if needed |
| Training too slow on CPU for experiments | Medium | Medium | Start with V3-small, run larger configs overnight |
| Burn API breaking changes | Low | High | Pin exact crate version in Cargo.lock |
| Regression in existing test suite | Medium | High | Run all 108+ tests after each phase, checkpoint compatibility tests |

---

## Dependencies

- **V2 should ship first.** Self-awareness observations and evolvable traits work with the frozen encoder and establish new baselines. V3 builds on those baselines.
- **Burn autodiff familiarity.** The current codebase uses Burn only for tensor operations, not its training infrastructure. V3 requires understanding Burn's module system, autodiff backend, and optimizer APIs.

---

## What V3 Unlocks

Once the encoder is unfrozen, several previously-useless or low-impact features become high-impact:

| Feature | With frozen encoder | With unfrozen encoder |
|---------|-------------------|-----------------------|
| Evolution | Near-zero impact (Prompt 9) | Meaningful — evolved weights in learned layers carry information |
| Evolvable traits (V2) | Moderate — traits modulate but brain can't adapt | High — traits + learned representations compound |
| Self-awareness (V2) | Moderate — useful signals through random projection | High — network can learn to use its own trajectory |
| Lamarckian inheritance | Pointless — nothing worth inheriting | Transformative — offspring start with trained representations |
| Larger networks | Pointless — more random params | Essential — more capacity for learned features |
| GRU memory | Pointless — gating on random representations | Valuable — selective memory of learned market patterns |
| Emotion drives | Useful via Tier 2 bypass | Useful via both Tier 1 (learned) and Tier 2 (modulation) |
