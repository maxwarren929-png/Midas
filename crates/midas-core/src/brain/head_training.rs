use super::{
    burn_runtime::{BurnNeuralModel, TrainBackend, split_gru_parameters},
    impulse_beta::evaluate_incremental_trade_sample,
};
use crate::{
    MidasError, Result,
    config::{Activation, LearningConfig, OptimizerConfig, OptimizerKind},
};
use burn::tensor::{Tensor, TensorData, activation};
use std::{ops::Range, sync::Mutex};

// Burn's NdArray autodiff graph bookkeeping is not deterministic when several
// graphs backpropagate concurrently in one process. Simulation updates are
// normally sequential, but tests and embedders may run engines in parallel.
// Serialize only graph construction/backward; inference remains concurrent.
static AUTODIFF_GRAPH_LOCK: Mutex<()> = Mutex::new(());

pub(super) fn train_burn_encoder(
    model: &mut BurnNeuralModel,
    batch: &crate::learning::PpoTrainingBatch,
    config: &LearningConfig,
    optimizer: &OptimizerConfig,
) -> Result<crate::learning::TrainingMetrics> {
    let saved_runtime = model.runtime.clone();
    let saved_lifetime = model.lifetime.clone();
    let saved_optimizer = model.optimizer.clone();
    let saved_adamw_optimizer = model.adamw_optimizer.clone();
    let saved_cached = model.cached.clone();
    let mut aggregate = crate::learning::TrainingMetrics::empty();
    let epochs = config.optimisation_epochs.max(1);
    let minibatches = recurrent_minibatch_ranges(&batch.samples, config.minibatch_size.max(1))?;
    let mut updates = 0usize;
    let result = (|| {
        for _ in 0..epochs {
            for range in &minibatches {
                let minibatch = crate::learning::PpoTrainingBatch {
                    samples: batch.samples[range.clone()].to_vec(),
                    sequence_length: batch.sequence_length,
                };
                let metrics = train_burn_encoder_inner(model, &minibatch, config, optimizer)?;
                aggregate.policy_loss += metrics.policy_loss;
                aggregate.value_loss += metrics.value_loss;
                aggregate.entropy += metrics.entropy;
                aggregate.total_loss += metrics.total_loss;
                aggregate.gradient_norm += metrics.gradient_norm;
                aggregate.pre_clip_gradient_norm += metrics.pre_clip_gradient_norm;
                if aggregate.hidden_gradient_norms.is_empty() {
                    aggregate.hidden_gradient_norms =
                        vec![0.0; metrics.hidden_gradient_norms.len()];
                }
                if aggregate.hidden_gradient_norms.len() != metrics.hidden_gradient_norms.len() {
                    return Err(MidasError::Policy(
                        "inconsistent hidden gradient telemetry layout".into(),
                    ));
                }
                for (total, value) in aggregate
                    .hidden_gradient_norms
                    .iter_mut()
                    .zip(&metrics.hidden_gradient_norms)
                {
                    *total += value;
                }
                aggregate.recurrent_gradient_norm += metrics.recurrent_gradient_norm;
                accumulate_optional_norm(
                    &mut aggregate.gru_update_gradient_norm,
                    metrics.gru_update_gradient_norm,
                );
                accumulate_optional_norm(
                    &mut aggregate.gru_reset_gradient_norm,
                    metrics.gru_reset_gradient_norm,
                );
                accumulate_optional_norm(
                    &mut aggregate.gru_candidate_gradient_norm,
                    metrics.gru_candidate_gradient_norm,
                );
                aggregate.actor_gradient_norm += metrics.actor_gradient_norm;
                aggregate.critic_gradient_norm += metrics.critic_gradient_norm;
                aggregate.gradient_clip_fraction += metrics.gradient_clip_fraction;
                aggregate.relative_update_norm += metrics.relative_update_norm;
                aggregate.approximate_kl += metrics.approximate_kl;
                aggregate.clip_fraction += metrics.clip_fraction;
                aggregate.update_index = metrics.update_index;
                updates += 1;
            }
        }
        if updates > 0 {
            aggregate.rollout_size = batch.samples.len();
            let divisor = updates as f32;
            aggregate.policy_loss /= divisor;
            aggregate.value_loss /= divisor;
            aggregate.entropy /= divisor;
            aggregate.total_loss /= divisor;
            aggregate.gradient_norm /= divisor;
            aggregate.pre_clip_gradient_norm /= divisor;
            for value in &mut aggregate.hidden_gradient_norms {
                *value /= divisor;
            }
            aggregate.recurrent_gradient_norm /= divisor;
            divide_optional_norm(&mut aggregate.gru_update_gradient_norm, divisor);
            divide_optional_norm(&mut aggregate.gru_reset_gradient_norm, divisor);
            divide_optional_norm(&mut aggregate.gru_candidate_gradient_norm, divisor);
            aggregate.actor_gradient_norm /= divisor;
            aggregate.critic_gradient_norm /= divisor;
            aggregate.gradient_clip_fraction /= divisor;
            aggregate.relative_update_norm /= divisor;
            aggregate.approximate_kl /= divisor;
            aggregate.clip_fraction /= divisor;
        }
        Ok(aggregate)
    })();
    model.runtime = saved_runtime;
    if result.is_err() {
        model.lifetime = saved_lifetime;
        model.optimizer = saved_optimizer;
        model.adamw_optimizer = saved_adamw_optimizer;
        model.cached = saved_cached;
    }
    result
}

fn accumulate_optional_norm(total: &mut Option<f32>, value: Option<f32>) {
    if let Some(value) = value {
        *total.get_or_insert(0.0) += value;
    }
}

fn divide_optional_norm(value: &mut Option<f32>, divisor: f32) {
    if let Some(value) = value {
        *value /= divisor;
    }
}

fn recurrent_minibatch_ranges(
    samples: &[crate::learning::PpoTrainingSample],
    target_size: usize,
) -> Result<Vec<Range<usize>>> {
    if samples.is_empty() {
        return Ok(Vec::new());
    }
    if !samples[0].sequence_start {
        return Err(MidasError::Policy(
            "PPO batch does not start at a recurrent sequence boundary".into(),
        ));
    }
    let mut starts = samples
        .iter()
        .enumerate()
        .filter_map(|(index, sample)| sample.sequence_start.then_some(index))
        .collect::<Vec<_>>();
    starts.push(samples.len());
    let mut minibatches = Vec::new();
    let mut batch_start = starts[0];
    let mut batch_size = 0usize;
    for window in starts.windows(2) {
        let sequence_start = window[0];
        let sequence_end = window[1];
        let sequence_size = sequence_end - sequence_start;
        if batch_size > 0 && batch_size + sequence_size > target_size {
            minibatches.push(batch_start..sequence_start);
            batch_start = sequence_start;
            batch_size = 0;
        }
        batch_size += sequence_size;
    }
    if batch_start < samples.len() {
        minibatches.push(batch_start..samples.len());
    }
    Ok(minibatches)
}

/// Applies a PPO update through truncated recurrent sequences. Each explicit
/// sequence start supplies a detached initial memory; subsequent steps use the
/// graph-produced memory so recurrent gradients flow backward within, but
/// never across, sequence boundaries.
fn train_burn_encoder_inner(
    model: &mut BurnNeuralModel,
    batch: &crate::learning::PpoTrainingBatch,
    config: &LearningConfig,
    optimizer: &OptimizerConfig,
) -> Result<crate::learning::TrainingMetrics> {
    if batch.samples.is_empty() {
        return Ok(crate::learning::TrainingMetrics::empty());
    }
    if config.learning_rate <= 0.0
        || !config.learning_rate.is_finite()
        || config.ppo_clip < 0.0
        || !config.ppo_clip.is_finite()
        || !config.gradient_clip.is_finite()
        || config.gradient_clip <= 0.0
        || config.critic_coefficient < 0.0
        || !config.critic_coefficient.is_finite()
        || config.entropy_coefficient < 0.0
        || !config.entropy_coefficient.is_finite()
    {
        return Err(MidasError::Config(
            "invalid PPO update configuration".into(),
        ));
    }
    let actor_inputs = model.architecture.memory.size;
    let actor_outputs = model.architecture.actor_output_size;
    let critic_outputs = model.architecture.critic_output_size;
    if critic_outputs != 1 {
        return Err(MidasError::Policy(
            "PPO encoder training requires a one-output critic".into(),
        ));
    }
    let mut observations =
        Vec::with_capacity(batch.samples.len() * model.architecture.observation_size);
    let mut recurrent_memories = Vec::with_capacity(batch.samples.len() * actor_inputs);
    let mut sequence_starts = Vec::with_capacity(batch.samples.len());
    let mut actor_output_gradients = Vec::with_capacity(batch.samples.len() * actor_outputs);
    let mut critic_output_gradients = Vec::with_capacity(batch.samples.len() * critic_outputs);
    let mut policy_loss = 0.0_f32;
    let mut value_loss = 0.0_f32;
    let mut entropy = 0.0_f32;
    let mut total_loss = 0.0_f32;
    let mut actor_count = 0usize;
    let mut critic_count = 0usize;
    let mut approximate_kl = 0.0_f32;
    let mut clipped_samples = 0usize;

    let mut recurrent_state = None;
    for sample in &batch.samples {
        if sample.recurrent_state.memory.len() != model.architecture.memory.size {
            return Err(MidasError::Policy(
                "PPO sample does not match neural architecture".into(),
            ));
        }
        if sample.sequence_start {
            recurrent_state = Some(sample.recurrent_state.clone());
        }
        let current_state = recurrent_state.as_ref().ok_or_else(|| {
            MidasError::Policy("PPO sample crosses an unmarked recurrent boundary".into())
        })?;
        let encoded = model.encode_observation(&sample.observation, current_state)?;
        let output = model.evaluate_representation(&encoded.representation)?;
        recurrent_state = Some(encoded.next_runtime_state);
        observations.extend_from_slice(&sample.observation.values);
        recurrent_memories.extend_from_slice(&sample.recurrent_state.memory);
        sequence_starts.push(sample.sequence_start);
        let mut actor_output_gradient = vec![0.0_f32; actor_outputs];
        if let Some(action_sample) = sample.action_sample.as_ref() {
            let old_log_probability = sample.old_log_probability.ok_or_else(|| {
                MidasError::Policy("actor sample is missing its old log probability".into())
            })?;
            let advantage = sample
                .advantage
                .ok_or_else(|| MidasError::Policy("actor sample is missing advantage".into()))?;
            let temperature = sample
                .temperature
                .ok_or_else(|| MidasError::Policy("actor sample is missing temperature".into()))?;
            let distribution = evaluate_incremental_trade_sample(
                &output.logits,
                &sample.tradable,
                action_sample,
                temperature,
                &model.architecture.action_distribution,
            )?;
            let new_log_probability = distribution.log_probability;
            let sample_entropy = distribution.entropy;
            let target_return = sample.return_value.unwrap_or(output.value);
            let loss = crate::learning::ppo_loss(
                old_log_probability,
                new_log_probability,
                advantage,
                output.value,
                target_return,
                sample_entropy,
                config.ppo_clip,
                config.critic_coefficient,
                config.entropy_coefficient,
            )?;
            policy_loss += loss.policy_loss;
            entropy += sample_entropy;
            total_loss += loss.policy_loss - config.entropy_coefficient * sample_entropy;
            actor_count += 1;

            let ratio = loss.ratio;
            let unclipped = ratio * advantage;
            let clipped = ratio.clamp(1.0 - config.ppo_clip, 1.0 + config.ppo_clip) * advantage;
            let clipped_active = if advantage >= 0.0 {
                clipped < unclipped
            } else {
                clipped > unclipped
            };
            approximate_kl += old_log_probability - new_log_probability;
            if clipped_active {
                clipped_samples += 1;
            }
            if !clipped_active {
                let coefficient = ratio * advantage;
                for decision in &distribution.decisions {
                    actor_output_gradient[decision.actor_alpha_index] +=
                        -coefficient * decision.d_log_probability_d_raw_alpha;
                    actor_output_gradient[decision.actor_beta_index] +=
                        -coefficient * decision.d_log_probability_d_raw_beta;
                }
            }
            // The entropy derivative is the exact derivative of each Beta
            // factor, chained through the bounded softplus shape transform.
            if config.entropy_coefficient > 0.0 {
                for decision in &distribution.decisions {
                    actor_output_gradient[decision.actor_alpha_index] +=
                        -config.entropy_coefficient * decision.d_entropy_d_raw_alpha;
                    actor_output_gradient[decision.actor_beta_index] +=
                        -config.entropy_coefficient * decision.d_entropy_d_raw_beta;
                }
            }
        }
        actor_output_gradients.extend_from_slice(&actor_output_gradient);
        let mut critic_output_gradient = vec![0.0_f32; critic_outputs];
        if let Some(target_return) = sample.return_value {
            critic_output_gradient[0] =
                2.0 * config.critic_coefficient * (output.value - target_return);
            let critic_loss = (output.value - target_return).powi(2);
            if !critic_loss.is_finite() {
                return Err(MidasError::Policy("non-finite critic loss".into()));
            }
            value_loss += critic_loss;
            total_loss += config.critic_coefficient * critic_loss;
            critic_count += 1;
        }
        critic_output_gradients.extend_from_slice(&critic_output_gradient);
    }

    let actor_divisor = actor_count.max(1) as f32;
    let critic_divisor = critic_count.max(1) as f32;
    for gradient in &mut actor_output_gradients {
        *gradient /= actor_divisor;
        if !gradient.is_finite() {
            return Err(MidasError::Policy("non-finite PPO gradient".into()));
        }
    }
    for gradient in &mut critic_output_gradients {
        *gradient /= critic_divisor;
        if !gradient.is_finite() {
            return Err(MidasError::Policy("non-finite PPO gradient".into()));
        }
    }
    let BpttGradients {
        mut hidden,
        mut recurrent,
        mut actor_weight_gradient,
        mut actor_bias_gradient,
        mut critic_weight_gradient,
        mut critic_bias_gradient,
    } = autodiff_bptt_gradients(
        model,
        batch.samples.len(),
        observations,
        recurrent_memories,
        sequence_starts,
        actor_output_gradients,
        critic_output_gradients,
    )?;
    let hidden_gradient_norms = hidden.iter().map(dense_gradient_norm).collect::<Vec<_>>();
    let recurrent_gradient_norm = dense_gradient_norm(&recurrent);
    let actor_gradient_norm = vector_pair_norm(&actor_weight_gradient, &actor_bias_gradient);
    let critic_gradient_norm = vector_pair_norm(&critic_weight_gradient, &critic_bias_gradient);
    let gru_gate_norms = gru_gate_gradient_norms(
        &recurrent,
        model.lifetime.recurrent.inputs,
        model.architecture.memory.size,
    );
    let pre_clip_gradient_norm = hidden
        .iter()
        .flat_map(|gradient| {
            gradient
                .weight_gradient
                .iter()
                .chain(&gradient.bias_gradient)
        })
        .chain(&recurrent.weight_gradient)
        .chain(&recurrent.bias_gradient)
        .chain(&actor_weight_gradient)
        .chain(&actor_bias_gradient)
        .chain(&critic_weight_gradient)
        .chain(&critic_bias_gradient)
        .map(|gradient| gradient * gradient)
        .sum::<f32>()
        .sqrt();
    if !pre_clip_gradient_norm.is_finite() {
        return Err(MidasError::Policy("non-finite PPO gradient norm".into()));
    }
    let gradient_clip_applied =
        config.gradient_clip > 0.0 && pre_clip_gradient_norm > config.gradient_clip;
    let gradient_norm = if gradient_clip_applied {
        let scale = config.gradient_clip / pre_clip_gradient_norm;
        for gradient in &mut hidden {
            for value in gradient
                .weight_gradient
                .iter_mut()
                .chain(gradient.bias_gradient.iter_mut())
            {
                *value *= scale;
            }
        }
        for value in recurrent
            .weight_gradient
            .iter_mut()
            .chain(recurrent.bias_gradient.iter_mut())
        {
            *value *= scale;
        }
        for gradient in actor_weight_gradient
            .iter_mut()
            .chain(actor_bias_gradient.iter_mut())
            .chain(critic_weight_gradient.iter_mut())
            .chain(critic_bias_gradient.iter_mut())
        {
            *gradient *= scale;
        }
        config.gradient_clip
    } else {
        pre_clip_gradient_norm
    };

    let gradients = BpttGradients {
        hidden,
        recurrent,
        actor_weight_gradient,
        actor_bias_gradient,
        critic_weight_gradient,
        critic_bias_gradient,
    };
    let (next_lifetime, relative_update_norm, update_index) = match optimizer.kind {
        OptimizerKind::Momentum => apply_momentum(model, &gradients, config)?,
        OptimizerKind::AdamW => apply_adamw(model, &gradients, config, optimizer)?,
    };
    if !relative_update_norm.is_finite() {
        return Err(MidasError::Policy(
            "non-finite relative parameter update norm".into(),
        ));
    }
    next_lifetime.validate(&model.architecture)?;
    let device = Default::default();
    model.cached = model.cached.with_updated_trainable(&next_lifetime, &device);
    model.lifetime = next_lifetime;
    Ok(crate::learning::TrainingMetrics {
        rollout_size: batch.samples.len(),
        policy_loss: policy_loss / actor_divisor,
        value_loss: value_loss / critic_divisor,
        entropy: entropy / actor_divisor,
        total_loss: total_loss / (actor_count.max(critic_count).max(1) as f32),
        gradient_norm,
        pre_clip_gradient_norm,
        hidden_gradient_norms,
        recurrent_gradient_norm,
        gru_update_gradient_norm: gru_gate_norms.map(|norms| norms[0]),
        gru_reset_gradient_norm: gru_gate_norms.map(|norms| norms[1]),
        gru_candidate_gradient_norm: gru_gate_norms.map(|norms| norms[2]),
        actor_gradient_norm,
        critic_gradient_norm,
        gradient_clip_fraction: if gradient_clip_applied { 1.0 } else { 0.0 },
        relative_update_norm,
        approximate_kl: approximate_kl / actor_divisor,
        clip_fraction: clipped_samples as f32 / actor_divisor,
        update_index,
    })
}

pub(super) struct DenseGradients {
    pub(super) weight_gradient: Vec<f32>,
    pub(super) bias_gradient: Vec<f32>,
}

pub(super) struct BpttGradients {
    pub(super) hidden: Vec<DenseGradients>,
    pub(super) recurrent: DenseGradients,
    pub(super) actor_weight_gradient: Vec<f32>,
    pub(super) actor_bias_gradient: Vec<f32>,
    pub(super) critic_weight_gradient: Vec<f32>,
    pub(super) critic_bias_gradient: Vec<f32>,
}

fn vector_pair_norm(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .chain(right)
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
}

fn dense_gradient_norm(gradient: &DenseGradients) -> f32 {
    vector_pair_norm(&gradient.weight_gradient, &gradient.bias_gradient)
}

fn gru_gate_gradient_norms(
    gradient: &DenseGradients,
    recurrent_inputs: usize,
    memory_size: usize,
) -> Option<[f32; 3]> {
    let recurrent_outputs = memory_size.checked_mul(3)?;
    if gradient.weight_gradient.len() != recurrent_inputs.checked_mul(recurrent_outputs)?
        || gradient.bias_gradient.len() != recurrent_outputs
    {
        return None;
    }
    let mut squared = [0.0_f32; 3];
    for row in gradient.weight_gradient.chunks_exact(recurrent_outputs) {
        for (gate, total) in squared.iter_mut().enumerate() {
            let start = gate * memory_size;
            *total += row[start..start + memory_size]
                .iter()
                .map(|value| value * value)
                .sum::<f32>();
        }
    }
    for (gate, total) in squared.iter_mut().enumerate() {
        let start = gate * memory_size;
        *total += gradient.bias_gradient[start..start + memory_size]
            .iter()
            .map(|value| value * value)
            .sum::<f32>();
    }
    Some(squared.map(f32::sqrt))
}

fn apply_momentum(
    model: &mut BurnNeuralModel,
    gradients: &BpttGradients,
    config: &LearningConfig,
) -> Result<(super::LifetimeParameters, f32, u64)> {
    if model.adamw_optimizer.is_some() {
        return Err(MidasError::Policy(
            "cannot change a trained neural model from AdamW to momentum".into(),
        ));
    }
    let mut next_optimizer = model.optimizer.clone();
    next_optimizer.step = next_optimizer.step.saturating_add(1);
    let momentum = 0.9_f32;
    for (state, gradient) in next_optimizer.hidden.iter_mut().zip(&gradients.hidden) {
        update_momentum_dense(state, gradient, momentum);
    }
    update_momentum_dense(
        &mut next_optimizer.recurrent,
        &gradients.recurrent,
        momentum,
    );
    update_momentum_values(
        &mut next_optimizer.actor.weight_momentum,
        &gradients.actor_weight_gradient,
        momentum,
    );
    update_momentum_values(
        &mut next_optimizer.actor.bias_momentum,
        &gradients.actor_bias_gradient,
        momentum,
    );
    update_momentum_values(
        &mut next_optimizer.critic.weight_momentum,
        &gradients.critic_weight_gradient,
        momentum,
    );
    update_momentum_values(
        &mut next_optimizer.critic.bias_momentum,
        &gradients.critic_bias_gradient,
        momentum,
    );
    let relative_update_norm =
        relative_update_norm(&model.lifetime, &next_optimizer, config.learning_rate);
    let mut next_lifetime = model.lifetime.clone();
    for (layer, state) in next_lifetime.hidden.iter_mut().zip(&next_optimizer.hidden) {
        apply_momentum_dense(layer, state, config.learning_rate);
    }
    apply_momentum_dense(
        &mut next_lifetime.recurrent,
        &next_optimizer.recurrent,
        config.learning_rate,
    );
    apply_momentum_dense(
        &mut next_lifetime.actor,
        &next_optimizer.actor,
        config.learning_rate,
    );
    apply_momentum_dense(
        &mut next_lifetime.critic,
        &next_optimizer.critic,
        config.learning_rate,
    );
    next_optimizer.validate(&model.architecture, next_lifetime.recurrent.outputs)?;
    let step = next_optimizer.step;
    model.optimizer = next_optimizer;
    Ok((next_lifetime, relative_update_norm, step))
}

fn update_momentum_dense(
    state: &mut super::DenseOptimizerState,
    gradient: &DenseGradients,
    momentum: f32,
) {
    update_momentum_values(
        &mut state.weight_momentum,
        &gradient.weight_gradient,
        momentum,
    );
    update_momentum_values(&mut state.bias_momentum, &gradient.bias_gradient, momentum);
}

fn update_momentum_values(state: &mut [f32], gradient: &[f32], momentum: f32) {
    for (state, gradient) in state.iter_mut().zip(gradient) {
        *state = momentum * *state + (1.0 - momentum) * gradient;
    }
}

fn apply_momentum_dense(
    layer: &mut super::DenseGenome,
    state: &super::DenseOptimizerState,
    learning_rate: f32,
) {
    for (parameter, update) in layer.weights.iter_mut().zip(&state.weight_momentum) {
        *parameter -= learning_rate * update;
    }
    for (parameter, update) in layer.bias.iter_mut().zip(&state.bias_momentum) {
        *parameter -= learning_rate * update;
    }
}

fn apply_adamw(
    model: &mut BurnNeuralModel,
    gradients: &BpttGradients,
    config: &LearningConfig,
    optimizer_config: &OptimizerConfig,
) -> Result<(super::LifetimeParameters, f32, u64)> {
    if model.adamw_optimizer.is_none() && model.optimizer.step != 0 {
        return Err(MidasError::Policy(
            "cannot change a trained neural model from momentum to AdamW".into(),
        ));
    }
    let recurrent_outputs = model.lifetime.recurrent.outputs;
    let mut optimizer = model.adamw_optimizer.clone().unwrap_or_else(|| {
        super::AdamWOptimizerState::empty(&model.architecture, recurrent_outputs)
    });
    optimizer.step = optimizer.step.saturating_add(1);
    let step = optimizer.step;
    let bias_correction1 = 1.0 - optimizer_config.beta1.powf(step as f32);
    let bias_correction2 = 1.0 - optimizer_config.beta2.powf(step as f32);
    let mut next_lifetime = model.lifetime.clone();
    let mut parameter_squared = 0.0_f32;
    let mut update_squared = 0.0_f32;
    for ((layer, state), gradient) in next_lifetime
        .hidden
        .iter_mut()
        .zip(&mut optimizer.hidden)
        .zip(&gradients.hidden)
    {
        apply_adamw_dense(
            layer,
            state,
            gradient,
            config,
            optimizer_config,
            bias_correction1,
            bias_correction2,
            &mut parameter_squared,
            &mut update_squared,
        );
    }
    apply_adamw_dense(
        &mut next_lifetime.recurrent,
        &mut optimizer.recurrent,
        &gradients.recurrent,
        config,
        optimizer_config,
        bias_correction1,
        bias_correction2,
        &mut parameter_squared,
        &mut update_squared,
    );
    let actor = DenseGradients {
        weight_gradient: gradients.actor_weight_gradient.clone(),
        bias_gradient: gradients.actor_bias_gradient.clone(),
    };
    apply_adamw_dense(
        &mut next_lifetime.actor,
        &mut optimizer.actor,
        &actor,
        config,
        optimizer_config,
        bias_correction1,
        bias_correction2,
        &mut parameter_squared,
        &mut update_squared,
    );
    let critic = DenseGradients {
        weight_gradient: gradients.critic_weight_gradient.clone(),
        bias_gradient: gradients.critic_bias_gradient.clone(),
    };
    apply_adamw_dense(
        &mut next_lifetime.critic,
        &mut optimizer.critic,
        &critic,
        config,
        optimizer_config,
        bias_correction1,
        bias_correction2,
        &mut parameter_squared,
        &mut update_squared,
    );
    optimizer.validate(&model.architecture, recurrent_outputs)?;
    let relative_update_norm = update_squared.sqrt() / parameter_squared.sqrt().max(f32::EPSILON);
    model.adamw_optimizer = Some(optimizer);
    Ok((next_lifetime, relative_update_norm, step))
}

#[allow(clippy::too_many_arguments)]
fn apply_adamw_dense(
    layer: &mut super::DenseGenome,
    state: &mut super::AdamWDenseState,
    gradient: &DenseGradients,
    config: &LearningConfig,
    optimizer: &OptimizerConfig,
    bias_correction1: f32,
    bias_correction2: f32,
    parameter_squared: &mut f32,
    update_squared: &mut f32,
) {
    for (((parameter, first), second), gradient) in layer
        .weights
        .iter_mut()
        .zip(&mut state.first_weight_moment)
        .zip(&mut state.second_weight_moment)
        .zip(&gradient.weight_gradient)
    {
        *parameter_squared += *parameter * *parameter;
        *first = optimizer.beta1 * *first + (1.0 - optimizer.beta1) * gradient;
        *second = optimizer.beta2 * *second + (1.0 - optimizer.beta2) * gradient * gradient;
        let adaptive =
            (*first / bias_correction1) / ((*second / bias_correction2).sqrt() + optimizer.epsilon);
        let update = adaptive + optimizer.weight_decay * *parameter;
        let delta = config.learning_rate * update;
        *update_squared += delta * delta;
        *parameter -= delta;
    }
    for (((parameter, first), second), gradient) in layer
        .bias
        .iter_mut()
        .zip(&mut state.first_bias_moment)
        .zip(&mut state.second_bias_moment)
        .zip(&gradient.bias_gradient)
    {
        *parameter_squared += *parameter * *parameter;
        *first = optimizer.beta1 * *first + (1.0 - optimizer.beta1) * gradient;
        *second = optimizer.beta2 * *second + (1.0 - optimizer.beta2) * gradient * gradient;
        let update =
            (*first / bias_correction1) / ((*second / bias_correction2).sqrt() + optimizer.epsilon);
        let delta = config.learning_rate * update;
        *update_squared += delta * delta;
        *parameter -= delta;
    }
}

fn relative_update_norm(
    lifetime: &super::LifetimeParameters,
    optimizer: &super::BpttOptimizerState,
    learning_rate: f32,
) -> f32 {
    let mut parameter_squared = 0.0_f32;
    let mut update_squared = 0.0_f32;
    let mut accumulate = |layer: &super::DenseGenome, state: &super::DenseOptimizerState| {
        parameter_squared += layer
            .weights
            .iter()
            .chain(&layer.bias)
            .map(|value| value * value)
            .sum::<f32>();
        update_squared += state
            .weight_momentum
            .iter()
            .chain(&state.bias_momentum)
            .map(|value| {
                let update = learning_rate * value;
                update * update
            })
            .sum::<f32>();
    };
    for (layer, state) in lifetime.hidden.iter().zip(&optimizer.hidden) {
        accumulate(layer, state);
    }
    accumulate(&lifetime.recurrent, &optimizer.recurrent);
    accumulate(&lifetime.actor, &optimizer.actor);
    accumulate(&lifetime.critic, &optimizer.critic);
    update_squared.sqrt() / parameter_squared.sqrt().max(f32::EPSILON)
}

enum TrainRecurrentParameters {
    Simple(Box<TrainSimpleRecurrentParameters>),
    Gru(Box<TrainGruParameters>),
}

struct TrainSimpleRecurrentParameters {
    weights: Tensor<TrainBackend, 2>,
    bias: Tensor<TrainBackend, 2>,
}

struct TrainGruParameters {
    update_weights: Tensor<TrainBackend, 2>,
    update_bias: Tensor<TrainBackend, 2>,
    reset_weights: Tensor<TrainBackend, 2>,
    reset_bias: Tensor<TrainBackend, 2>,
    candidate_input_weights: Tensor<TrainBackend, 2>,
    candidate_hidden_weights: Tensor<TrainBackend, 2>,
    candidate_bias: Tensor<TrainBackend, 2>,
    hidden_size: usize,
    memory_size: usize,
}

pub(super) fn autodiff_bptt_gradients(
    model: &BurnNeuralModel,
    sample_count: usize,
    observations: Vec<f32>,
    recurrent_memories: Vec<f32>,
    sequence_starts: Vec<bool>,
    actor_output_gradients: Vec<f32>,
    critic_output_gradients: Vec<f32>,
) -> Result<BpttGradients> {
    let _graph_guard = AUTODIFF_GRAPH_LOCK
        .lock()
        .map_err(|_| MidasError::Policy("Burn autodiff graph lock is poisoned".into()))?;
    let actor_inputs = model.architecture.memory.size;
    let observation_inputs = model.architecture.observation_size;
    let actor_outputs = model.architecture.actor_output_size;
    let critic_outputs = model.architecture.critic_output_size;
    if sample_count == 0
        || observations.len() != sample_count * observation_inputs
        || recurrent_memories.len() != sample_count * actor_inputs
        || sequence_starts.len() != sample_count
        || !sequence_starts.first().copied().unwrap_or(false)
        || actor_output_gradients.len() != sample_count * actor_outputs
        || critic_output_gradients.len() != sample_count * critic_outputs
        || observations
            .iter()
            .chain(&recurrent_memories)
            .chain(&actor_output_gradients)
            .chain(&critic_output_gradients)
            .any(|value| !value.is_finite())
    {
        return Err(MidasError::Policy(
            "autodiff BPTT batch dimensions, boundaries, or finiteness mismatch".into(),
        ));
    }

    let device = Default::default();
    let mut hidden_tensors = Vec::with_capacity(model.lifetime.hidden.len());
    for layer in &model.lifetime.hidden {
        let weights = Tensor::<TrainBackend, 2>::from_data(
            TensorData::new(layer.weights.clone(), [layer.inputs, layer.outputs]),
            &device,
        )
        .require_grad();
        let bias = Tensor::<TrainBackend, 2>::from_data(
            TensorData::new(layer.bias.clone(), [1, layer.outputs]),
            &device,
        )
        .require_grad();
        hidden_tensors.push((weights, bias));
    }
    let recurrent = &model.lifetime.recurrent;
    let recurrent_parameters = if recurrent.outputs == actor_inputs {
        TrainRecurrentParameters::Simple(Box::new(TrainSimpleRecurrentParameters {
            weights: Tensor::<TrainBackend, 2>::from_data(
                TensorData::new(
                    recurrent.weights.clone(),
                    [recurrent.inputs, recurrent.outputs],
                ),
                &device,
            )
            .require_grad(),
            bias: Tensor::<TrainBackend, 2>::from_data(
                TensorData::new(recurrent.bias.clone(), [1, recurrent.outputs]),
                &device,
            )
            .require_grad(),
        }))
    } else {
        let hidden_size = recurrent.inputs - actor_inputs;
        let parts = split_gru_parameters(recurrent, hidden_size, actor_inputs);
        TrainRecurrentParameters::Gru(Box::new(TrainGruParameters {
            update_weights: train_parameter(
                parts.update_weights,
                [recurrent.inputs, actor_inputs],
                &device,
            ),
            update_bias: train_parameter(parts.update_bias, [1, actor_inputs], &device),
            reset_weights: train_parameter(
                parts.reset_weights,
                [recurrent.inputs, actor_inputs],
                &device,
            ),
            reset_bias: train_parameter(parts.reset_bias, [1, actor_inputs], &device),
            candidate_input_weights: train_parameter(
                parts.candidate_input_weights,
                [hidden_size, actor_inputs],
                &device,
            ),
            candidate_hidden_weights: train_parameter(
                parts.candidate_hidden_weights,
                [actor_inputs, actor_inputs],
                &device,
            ),
            candidate_bias: train_parameter(parts.candidate_bias, [1, actor_inputs], &device),
            hidden_size,
            memory_size: actor_inputs,
        }))
    };
    let mut memory = None;
    let mut representations = Vec::with_capacity(sample_count);
    for (index, sequence_start) in sequence_starts.iter().copied().enumerate() {
        if sequence_start {
            let start = index * actor_inputs;
            memory = Some(Tensor::<TrainBackend, 2>::from_data(
                TensorData::new(
                    recurrent_memories[start..start + actor_inputs].to_vec(),
                    [1, actor_inputs],
                ),
                &device,
            ));
        }
        let current_memory = memory.clone().ok_or_else(|| {
            MidasError::Policy("autodiff BPTT sequence has no initial memory".into())
        })?;
        let observation_start = index * observation_inputs;
        let mut encoded = Tensor::<TrainBackend, 2>::from_data(
            TensorData::new(
                observations[observation_start..observation_start + observation_inputs].to_vec(),
                [1, observation_inputs],
            ),
            &device,
        );
        for (weights, bias) in &hidden_tensors {
            encoded = activate_train(
                encoded.matmul(weights.clone()) + bias.clone(),
                model.architecture.activation,
            );
        }
        let representation = recurrent_train_step(encoded, current_memory, &recurrent_parameters);
        memory = Some(representation.clone());
        representations.push(representation);
    }
    let representation = Tensor::cat(representations, 0);
    let actor_weights = Tensor::<TrainBackend, 2>::from_data(
        TensorData::new(
            model.lifetime.actor.weights.clone(),
            [actor_inputs, actor_outputs],
        ),
        &device,
    )
    .require_grad();
    let actor_bias = Tensor::<TrainBackend, 2>::from_data(
        TensorData::new(model.lifetime.actor.bias.clone(), [1, actor_outputs]),
        &device,
    )
    .require_grad();
    let critic_weights = Tensor::<TrainBackend, 2>::from_data(
        TensorData::new(
            model.lifetime.critic.weights.clone(),
            [actor_inputs, critic_outputs],
        ),
        &device,
    )
    .require_grad();
    let critic_bias = Tensor::<TrainBackend, 2>::from_data(
        TensorData::new(model.lifetime.critic.bias.clone(), [1, critic_outputs]),
        &device,
    )
    .require_grad();

    let actor_upstream = Tensor::<TrainBackend, 2>::from_data(
        TensorData::new(actor_output_gradients, [sample_count, actor_outputs]),
        &device,
    );
    let critic_upstream = Tensor::<TrainBackend, 2>::from_data(
        TensorData::new(critic_output_gradients, [sample_count, critic_outputs]),
        &device,
    );
    let actor_output = representation.clone().matmul(actor_weights.clone()) + actor_bias.clone();
    let critic_output = representation.matmul(critic_weights.clone()) + critic_bias.clone();
    let surrogate_objective =
        (actor_output * actor_upstream).sum() + (critic_output * critic_upstream).sum();
    let gradients = surrogate_objective.backward();

    let hidden = hidden_tensors
        .iter()
        .enumerate()
        .map(|(index, (weights, bias))| {
            Ok(DenseGradients {
                weight_gradient: gradient_vec_or_zeros(
                    weights,
                    &gradients,
                    model.lifetime.hidden[index].weights.len(),
                    &format!("hidden layer {index} weights"),
                )?,
                bias_gradient: gradient_vec_or_zeros(
                    bias,
                    &gradients,
                    model.lifetime.hidden[index].bias.len(),
                    &format!("hidden layer {index} bias"),
                )?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let result = BpttGradients {
        hidden,
        recurrent: recurrent_gradients(
            &recurrent_parameters,
            &gradients,
            &model.lifetime.recurrent,
        )?,
        actor_weight_gradient: gradient_vec_or_zeros(
            &actor_weights,
            &gradients,
            model.lifetime.actor.weights.len(),
            "actor weights",
        )?,
        actor_bias_gradient: gradient_vec_or_zeros(
            &actor_bias,
            &gradients,
            model.lifetime.actor.bias.len(),
            "actor bias",
        )?,
        critic_weight_gradient: gradient_vec_or_zeros(
            &critic_weights,
            &gradients,
            model.lifetime.critic.weights.len(),
            "critic weights",
        )?,
        critic_bias_gradient: gradient_vec_or_zeros(
            &critic_bias,
            &gradients,
            model.lifetime.critic.bias.len(),
            "critic bias",
        )?,
    };
    if result
        .hidden
        .iter()
        .flat_map(|gradient| {
            gradient
                .weight_gradient
                .iter()
                .chain(&gradient.bias_gradient)
        })
        .chain(&result.recurrent.weight_gradient)
        .chain(&result.recurrent.bias_gradient)
        .chain(&result.actor_weight_gradient)
        .chain(&result.actor_bias_gradient)
        .chain(&result.critic_weight_gradient)
        .chain(&result.critic_bias_gradient)
        .any(|value| !value.is_finite())
    {
        return Err(MidasError::Policy(
            "Burn autodiff produced a non-finite PPO BPTT gradient".into(),
        ));
    }
    Ok(result)
}

fn activate_train(
    x: Tensor<TrainBackend, 2>,
    activation_kind: Activation,
) -> Tensor<TrainBackend, 2> {
    match activation_kind {
        Activation::Relu => activation::relu(x),
        Activation::Gelu => activation::gelu(x),
        Activation::Silu => activation::silu(x),
    }
}

fn train_parameter(
    values: Vec<f32>,
    shape: [usize; 2],
    device: &<TrainBackend as burn::tensor::backend::Backend>::Device,
) -> Tensor<TrainBackend, 2> {
    Tensor::from_data(TensorData::new(values, shape), device).require_grad()
}

fn recurrent_train_step(
    input: Tensor<TrainBackend, 2>,
    memory: Tensor<TrainBackend, 2>,
    parameters: &TrainRecurrentParameters,
) -> Tensor<TrainBackend, 2> {
    match parameters {
        TrainRecurrentParameters::Simple(simple) => activation::silu(
            Tensor::cat(vec![input, memory], 1).matmul(simple.weights.clone())
                + simple.bias.clone(),
        ),
        TrainRecurrentParameters::Gru(gru) => {
            let combined = Tensor::cat(vec![input.clone(), memory.clone()], 1);
            let update = activation::sigmoid(
                combined.clone().matmul(gru.update_weights.clone()) + gru.update_bias.clone(),
            );
            let reset = activation::sigmoid(
                combined.matmul(gru.reset_weights.clone()) + gru.reset_bias.clone(),
            );
            let candidate = (input.matmul(gru.candidate_input_weights.clone())
                + (reset * memory.clone()).matmul(gru.candidate_hidden_weights.clone())
                + gru.candidate_bias.clone())
            .tanh();
            update.clone() * memory + (update * -1.0 + 1.0) * candidate
        }
    }
}

fn recurrent_gradients(
    parameters: &TrainRecurrentParameters,
    gradients: &<TrainBackend as burn::tensor::backend::AutodiffBackend>::Gradients,
    recurrent: &super::DenseGenome,
) -> Result<DenseGradients> {
    match parameters {
        TrainRecurrentParameters::Simple(simple) => Ok(DenseGradients {
            weight_gradient: gradient_vec_or_zeros(
                &simple.weights,
                gradients,
                recurrent.weights.len(),
                "recurrent weights",
            )?,
            bias_gradient: gradient_vec_or_zeros(
                &simple.bias,
                gradients,
                recurrent.bias.len(),
                "recurrent bias",
            )?,
        }),
        TrainRecurrentParameters::Gru(gru) => {
            let update_weight_gradient = gradient_vec_or_zeros(
                &gru.update_weights,
                gradients,
                recurrent.inputs * gru.memory_size,
                "GRU update weights",
            )?;
            let reset_weight_gradient = gradient_vec_or_zeros(
                &gru.reset_weights,
                gradients,
                recurrent.inputs * gru.memory_size,
                "GRU reset weights",
            )?;
            let candidate_input_gradient = gradient_vec_or_zeros(
                &gru.candidate_input_weights,
                gradients,
                gru.hidden_size * gru.memory_size,
                "GRU candidate input weights",
            )?;
            let candidate_hidden_gradient = gradient_vec_or_zeros(
                &gru.candidate_hidden_weights,
                gradients,
                gru.memory_size * gru.memory_size,
                "GRU candidate hidden weights",
            )?;
            let mut weight_gradient = vec![0.0; recurrent.weights.len()];
            for row in 0..recurrent.inputs {
                let packed_start = row * recurrent.outputs;
                let gate_start = row * gru.memory_size;
                weight_gradient[packed_start..packed_start + gru.memory_size].copy_from_slice(
                    &update_weight_gradient[gate_start..gate_start + gru.memory_size],
                );
                weight_gradient[packed_start + gru.memory_size..packed_start + gru.memory_size * 2]
                    .copy_from_slice(
                        &reset_weight_gradient[gate_start..gate_start + gru.memory_size],
                    );
                let candidate_source = if row < gru.hidden_size {
                    &candidate_input_gradient
                        [row * gru.memory_size..row * gru.memory_size + gru.memory_size]
                } else {
                    let memory_row = row - gru.hidden_size;
                    &candidate_hidden_gradient[memory_row * gru.memory_size
                        ..memory_row * gru.memory_size + gru.memory_size]
                };
                weight_gradient
                    [packed_start + gru.memory_size * 2..packed_start + gru.memory_size * 3]
                    .copy_from_slice(candidate_source);
            }
            let mut bias_gradient = gradient_vec_or_zeros(
                &gru.update_bias,
                gradients,
                gru.memory_size,
                "GRU update bias",
            )?;
            bias_gradient.extend(gradient_vec_or_zeros(
                &gru.reset_bias,
                gradients,
                gru.memory_size,
                "GRU reset bias",
            )?);
            bias_gradient.extend(gradient_vec_or_zeros(
                &gru.candidate_bias,
                gradients,
                gru.memory_size,
                "GRU candidate bias",
            )?);
            Ok(DenseGradients {
                weight_gradient,
                bias_gradient,
            })
        }
    }
}

fn gradient_vec_or_zeros(
    tensor: &Tensor<TrainBackend, 2>,
    gradients: &<TrainBackend as burn::tensor::backend::AutodiffBackend>::Gradients,
    length: usize,
    label: &str,
) -> Result<Vec<f32>> {
    match tensor.grad(gradients) {
        Some(gradient) => gradient
            .into_data()
            .to_vec::<f32>()
            .map_err(|error| MidasError::Policy(format!("invalid {label} gradient: {error:?}"))),
        None => Ok(vec![0.0; length]),
    }
}
