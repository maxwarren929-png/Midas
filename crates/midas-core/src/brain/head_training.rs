use super::{burn_runtime::BurnNeuralModel, impulse_beta::evaluate_incremental_trade_sample};
use crate::{MidasError, Result, config::LearningConfig};

pub(super) fn train_burn_heads(
    model: &mut BurnNeuralModel,
    batch: &crate::learning::PpoTrainingBatch,
    config: &LearningConfig,
) -> Result<crate::learning::TrainingMetrics> {
    let saved_runtime = model.runtime.clone();
    let saved_lifetime = model.lifetime.clone();
    let saved_optimizer = model.optimizer.clone();
    let saved_cached = model.cached.clone();
    let mut aggregate = crate::learning::TrainingMetrics::empty();
    let epochs = config.optimisation_epochs.max(1);
    let chunk_size = config.minibatch_size.max(config.sequence_length).max(1);
    let mut updates = 0usize;
    let result = (|| {
        for _ in 0..epochs {
            for samples in batch.samples.chunks(chunk_size) {
                let minibatch = crate::learning::PpoTrainingBatch {
                    samples: samples.to_vec(),
                    sequence_length: batch.sequence_length,
                };
                let metrics = train_burn_heads_inner(model, &minibatch, config)?;
                aggregate.policy_loss += metrics.policy_loss;
                aggregate.value_loss += metrics.value_loss;
                aggregate.entropy += metrics.entropy;
                aggregate.total_loss += metrics.total_loss;
                aggregate.gradient_norm += metrics.gradient_norm;
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
            aggregate.approximate_kl /= divisor;
            aggregate.clip_fraction /= divisor;
        }
        Ok(aggregate)
    })();
    model.runtime = saved_runtime;
    if result.is_err() {
        model.lifetime = saved_lifetime;
        model.optimizer = saved_optimizer;
        model.cached = saved_cached;
    }
    result
}

/// Applies a PPO update to the actor and critic heads. Hidden and recurrent
/// layers remain part of the stable runtime representation for now; updating
/// the heads still provides a genuine clipped policy/value optimization while
/// keeping the adapter independent of Burn's optimizer APIs.
fn train_burn_heads_inner(
    model: &mut BurnNeuralModel,
    batch: &crate::learning::PpoTrainingBatch,
    config: &LearningConfig,
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
    let critic_inputs = model.architecture.memory.size;
    let critic_outputs = model.architecture.critic_output_size;
    if critic_outputs != 1 {
        return Err(MidasError::Policy(
            "PPO head training requires a one-output critic".into(),
        ));
    }
    let mut actor_weight_gradient = vec![0.0_f32; actor_inputs * actor_outputs];
    let mut actor_bias_gradient = vec![0.0_f32; actor_outputs];
    let mut critic_weight_gradient = vec![0.0_f32; critic_inputs * critic_outputs];
    let mut critic_bias_gradient = vec![0.0_f32; critic_outputs];
    let mut policy_loss = 0.0_f32;
    let mut value_loss = 0.0_f32;
    let mut entropy = 0.0_f32;
    let mut total_loss = 0.0_f32;
    let mut actor_count = 0usize;
    let mut critic_count = 0usize;
    let mut approximate_kl = 0.0_f32;
    let mut clipped_samples = 0usize;

    for sample in &batch.samples {
        if sample.recurrent_state.memory.len() != model.architecture.memory.size {
            return Err(MidasError::Policy(
                "PPO sample does not match neural architecture".into(),
            ));
        }
        let encoded = model.encode_observation(&sample.observation, &sample.recurrent_state)?;
        let output = model.evaluate_representation(&encoded.representation)?;
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
            let mut actor_output_gradient = vec![0.0_f32; actor_outputs];
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
            accumulate_dense_gradient(
                &encoded.representation,
                &actor_output_gradient,
                &mut actor_weight_gradient,
                &mut actor_bias_gradient,
            )?;
        }
        if let Some(target_return) = sample.return_value {
            let critic_output_gradient =
                [2.0 * config.critic_coefficient * (output.value - target_return)];
            let critic_loss = (output.value - target_return).powi(2);
            if !critic_loss.is_finite() {
                return Err(MidasError::Policy("non-finite critic loss".into()));
            }
            value_loss += critic_loss;
            total_loss += config.critic_coefficient * critic_loss;
            critic_count += 1;
            accumulate_dense_gradient(
                &encoded.representation,
                &critic_output_gradient,
                &mut critic_weight_gradient,
                &mut critic_bias_gradient,
            )?;
        }
    }

    let actor_divisor = actor_count.max(1) as f32;
    let critic_divisor = critic_count.max(1) as f32;
    for gradient in actor_weight_gradient
        .iter_mut()
        .chain(actor_bias_gradient.iter_mut())
    {
        *gradient /= actor_divisor;
        if !gradient.is_finite() {
            return Err(MidasError::Policy("non-finite PPO gradient".into()));
        }
    }
    for gradient in critic_weight_gradient
        .iter_mut()
        .chain(critic_bias_gradient.iter_mut())
    {
        *gradient /= critic_divisor;
        if !gradient.is_finite() {
            return Err(MidasError::Policy("non-finite PPO gradient".into()));
        }
    }
    let mut gradient_norm = actor_weight_gradient
        .iter()
        .chain(&actor_bias_gradient)
        .chain(&critic_weight_gradient)
        .chain(&critic_bias_gradient)
        .map(|gradient| gradient * gradient)
        .sum::<f32>()
        .sqrt();
    if !gradient_norm.is_finite() {
        return Err(MidasError::Policy("non-finite PPO gradient norm".into()));
    }
    if config.gradient_clip > 0.0 && gradient_norm > config.gradient_clip {
        let scale = config.gradient_clip / gradient_norm;
        for gradient in actor_weight_gradient
            .iter_mut()
            .chain(actor_bias_gradient.iter_mut())
            .chain(critic_weight_gradient.iter_mut())
            .chain(critic_bias_gradient.iter_mut())
        {
            *gradient *= scale;
        }
        gradient_norm = config.gradient_clip;
    }

    let mut next_optimizer = model.optimizer.clone();
    next_optimizer.step = next_optimizer.step.saturating_add(1);
    let momentum = 0.9_f32;
    for (state, gradient) in next_optimizer
        .actor
        .weight_momentum
        .iter_mut()
        .zip(&actor_weight_gradient)
    {
        *state = momentum * *state + (1.0 - momentum) * gradient;
    }
    for (state, gradient) in next_optimizer
        .actor
        .bias_momentum
        .iter_mut()
        .zip(&actor_bias_gradient)
    {
        *state = momentum * *state + (1.0 - momentum) * gradient;
    }
    for (state, gradient) in next_optimizer
        .critic
        .weight_momentum
        .iter_mut()
        .zip(&critic_weight_gradient)
    {
        *state = momentum * *state + (1.0 - momentum) * gradient;
    }
    for (state, gradient) in next_optimizer
        .critic
        .bias_momentum
        .iter_mut()
        .zip(&critic_bias_gradient)
    {
        *state = momentum * *state + (1.0 - momentum) * gradient;
    }
    let mut next_lifetime = model.lifetime.clone();
    for (weight, update) in next_lifetime
        .actor
        .weights
        .iter_mut()
        .zip(&next_optimizer.actor.weight_momentum)
    {
        *weight -= config.learning_rate * update;
    }
    for (bias, update) in next_lifetime
        .actor
        .bias
        .iter_mut()
        .zip(&next_optimizer.actor.bias_momentum)
    {
        *bias -= config.learning_rate * update;
    }
    for (weight, update) in next_lifetime
        .critic
        .weights
        .iter_mut()
        .zip(&next_optimizer.critic.weight_momentum)
    {
        *weight -= config.learning_rate * update;
    }
    for (bias, update) in next_lifetime
        .critic
        .bias
        .iter_mut()
        .zip(&next_optimizer.critic.bias_momentum)
    {
        *bias -= config.learning_rate * update;
    }
    next_lifetime.validate(&model.architecture)?;
    next_optimizer.validate(&model.architecture)?;
    let device = Default::default();
    model.cached = model.cached.with_updated_heads(&next_lifetime, &device);
    model.lifetime = next_lifetime;
    model.optimizer = next_optimizer;
    Ok(crate::learning::TrainingMetrics {
        rollout_size: batch.samples.len(),
        policy_loss: policy_loss / actor_divisor,
        value_loss: value_loss / critic_divisor,
        entropy: entropy / actor_divisor,
        total_loss: total_loss / (actor_count.max(critic_count).max(1) as f32),
        gradient_norm,
        approximate_kl: approximate_kl / actor_divisor,
        clip_fraction: clipped_samples as f32 / actor_divisor,
        update_index: model.optimizer.step,
    })
}

pub(super) fn accumulate_dense_gradient(
    representation: &[f32],
    output_gradient: &[f32],
    weight_gradient: &mut [f32],
    bias_gradient: &mut [f32],
) -> Result<()> {
    if weight_gradient.len() != representation.len() * output_gradient.len()
        || bias_gradient.len() != output_gradient.len()
    {
        return Err(MidasError::Policy(
            "dense gradient dimensions do not match head dimensions".into(),
        ));
    }
    for (output, gradient) in output_gradient.iter().enumerate() {
        if !gradient.is_finite() {
            return Err(MidasError::Policy(
                "non-finite dense output gradient".into(),
            ));
        }
        bias_gradient[output] += *gradient;
        for (input, value) in representation.iter().enumerate() {
            if !value.is_finite() {
                return Err(MidasError::Policy("non-finite dense representation".into()));
            }
            weight_gradient[input * output_gradient.len() + output] += value * gradient;
        }
    }
    if weight_gradient
        .iter()
        .chain(bias_gradient.iter())
        .any(|value| !value.is_finite())
    {
        return Err(MidasError::Policy("non-finite dense gradient".into()));
    }
    Ok(())
}
