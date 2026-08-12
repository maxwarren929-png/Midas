use super::{NeuralOutput, PolicyActionSample, PolicyEvaluation};
use crate::{
    MidasError, Result,
    action::{ProposedAction, ProposedTradeImpulses},
    config::{ActionDistributionConfig, ActionDistributionKind, AssetOrder},
};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Beta, Distribution};

/// An independent bounded Beta impulse distribution.  The PPO action is the
/// vector of latent Beta samples.  Each active asset receives the deterministic
/// impulse `2u - 1`; the broker then interprets that impulse as an incremental
/// trade.  Keeping the latent sample as the PPO action makes the joint density
/// the exact sum of the independent Beta densities (the affine mapping to an
/// impulse is deterministic and is not post-processed by the validator).
#[derive(Clone, Debug)]
pub(super) struct BetaDecision {
    pub(super) actor_alpha_index: usize,
    pub(super) actor_beta_index: usize,
    pub(super) alpha: f64,
    pub(super) beta: f64,
    pub(super) latent: f64,
    pub(super) d_log_probability_d_raw_alpha: f32,
    pub(super) d_log_probability_d_raw_beta: f32,
    pub(super) d_entropy_d_raw_alpha: f32,
    pub(super) d_entropy_d_raw_beta: f32,
}

#[derive(Clone, Debug)]
pub(super) struct ImpulseBetaEvaluation {
    pub(super) log_probability: f32,
    pub(super) entropy: f32,
    pub(super) decisions: Vec<BetaDecision>,
}

#[cfg(test)]
pub(super) fn bounded_impulse_beta_evaluation(
    output: NeuralOutput,
    tradable: &[bool],
    maximum_trade_fraction: f64,
    stochastic: bool,
    temperature: f32,
    rng: &mut ChaCha8Rng,
    config: &ActionDistributionConfig,
) -> Result<PolicyEvaluation> {
    bounded_impulse_beta_evaluation_with_order(
        output,
        tradable,
        maximum_trade_fraction,
        stochastic,
        temperature,
        rng,
        config,
        0,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn bounded_impulse_beta_evaluation_with_order(
    output: NeuralOutput,
    tradable: &[bool],
    maximum_trade_fraction: f64,
    stochastic: bool,
    temperature: f32,
    rng: &mut ChaCha8Rng,
    config: &ActionDistributionConfig,
    order_offset: usize,
) -> Result<PolicyEvaluation> {
    validate_distribution_inputs(
        &output,
        tradable,
        maximum_trade_fraction,
        temperature,
        config,
    )?;
    let active_asset_indices = ordered_asset_indices(tradable, config, order_offset)?;
    let latents = if stochastic {
        sample_latents(
            &output.logits,
            &active_asset_indices,
            temperature,
            config,
            rng,
        )?
    } else {
        deterministic_latents(&output.logits, &active_asset_indices, temperature, config)?
    };
    let (impulses, decisions) = evaluate_impulses(
        &output.logits,
        tradable,
        &active_asset_indices,
        &latents,
        temperature,
        config,
    )?;
    let log_probability = decisions
        .iter()
        .map(|decision| beta_log_probability(decision.alpha, decision.beta, decision.latent))
        .sum::<Result<f64>>()? as f32;
    let entropy = decisions
        .iter()
        .map(|decision| beta_entropy(decision.alpha, decision.beta))
        .sum::<Result<f64>>()? as f32;
    let proposed = ProposedTradeImpulses::new(impulses.clone(), maximum_trade_fraction);
    Ok(PolicyEvaluation {
        proposed_action: ProposedAction::TradeImpulses(proposed),
        action_sample: stochastic.then_some(PolicyActionSample {
            active_asset_indices,
            latent_samples: latents.iter().map(|value| *value as f32).collect(),
            trade_impulses: impulses.iter().map(|value| *value as f32).collect(),
            maximum_trade_fraction: maximum_trade_fraction as f32,
            log_probability,
            entropy,
        }),
        log_probability: stochastic.then_some(log_probability),
        entropy,
        value_estimate: output.value,
        raw_logits: output.logits,
        temperature,
    })
}

fn validate_distribution_inputs(
    output: &NeuralOutput,
    tradable: &[bool],
    maximum_trade_fraction: f64,
    temperature: f32,
    config: &ActionDistributionConfig,
) -> Result<()> {
    if !matches!(config.kind, ActionDistributionKind::BoundedImpulseBeta)
        || !matches!(
            config.asset_order,
            AssetOrder::Stable | AssetOrder::EpisodeRotated
        )
        || output.logits.len() != tradable.len() * 2
        || !output.value.is_finite()
        || output.logits.iter().any(|value| !value.is_finite())
        || !temperature.is_finite()
        || temperature <= 0.0
        || !maximum_trade_fraction.is_finite()
        || !(0.0..=1.0).contains(&maximum_trade_fraction)
        || maximum_trade_fraction == 0.0
        || !config.minimum_shape.is_finite()
        || !config.maximum_shape.is_finite()
        || config.minimum_shape <= 0.0
        || config.maximum_shape <= config.minimum_shape
        || !config.latent_epsilon.is_finite()
        || !(0.0..0.5).contains(&config.latent_epsilon)
        || config.latent_epsilon == 0.0
    {
        return Err(MidasError::Policy(
            "invalid bounded impulse Beta output or action bounds".into(),
        ));
    }
    Ok(())
}

fn active_asset_indices(tradable: &[bool]) -> Result<Vec<usize>> {
    Ok(tradable
        .iter()
        .enumerate()
        .filter_map(|(index, is_tradable)| is_tradable.then_some(index))
        .collect())
}

fn ordered_asset_indices(
    tradable: &[bool],
    config: &ActionDistributionConfig,
    order_offset: usize,
) -> Result<Vec<usize>> {
    let mut indices = active_asset_indices(tradable)?;
    if matches!(config.asset_order, AssetOrder::EpisodeRotated) && !indices.is_empty() {
        let length = indices.len();
        indices.rotate_left(order_offset % length);
    }
    Ok(indices)
}

fn sample_latents(
    logits: &[f32],
    active_asset_indices: &[usize],
    temperature: f32,
    config: &ActionDistributionConfig,
    rng: &mut ChaCha8Rng,
) -> Result<Vec<f64>> {
    active_asset_indices
        .iter()
        .map(|asset_index| {
            let (alpha, beta, _, _) = beta_shapes(
                logits[asset_index * 2],
                logits[asset_index * 2 + 1],
                temperature,
                config,
            )?;
            let distribution = Beta::new(alpha, beta).map_err(|error| {
                MidasError::Policy(format!("invalid Beta action parameters: {error}"))
            })?;
            let sample = distribution.sample(rng).clamp(
                config.latent_epsilon as f64,
                1.0 - config.latent_epsilon as f64,
            );
            if !sample.is_finite() {
                return Err(MidasError::Policy(
                    "non-finite bounded impulse sample".into(),
                ));
            }
            Ok(sample)
        })
        .collect()
}

fn deterministic_latents(
    logits: &[f32],
    active_asset_indices: &[usize],
    temperature: f32,
    config: &ActionDistributionConfig,
) -> Result<Vec<f64>> {
    active_asset_indices
        .iter()
        .map(|asset_index| {
            let (alpha, beta, _, _) = beta_shapes(
                logits[asset_index * 2],
                logits[asset_index * 2 + 1],
                temperature,
                config,
            )?;
            Ok((alpha / (alpha + beta)).clamp(
                config.latent_epsilon as f64,
                1.0 - config.latent_epsilon as f64,
            ))
        })
        .collect()
}

fn evaluate_impulses(
    logits: &[f32],
    tradable: &[bool],
    active_asset_indices: &[usize],
    latents: &[f64],
    temperature: f32,
    config: &ActionDistributionConfig,
) -> Result<(Vec<f64>, Vec<BetaDecision>)> {
    if active_asset_indices.len() != latents.len() || logits.len() != tradable.len() * 2 {
        return Err(MidasError::Policy(
            "bounded impulse Beta latent dimensions mismatch".into(),
        ));
    }
    let mut impulses = vec![0.0; tradable.len()];
    let mut decisions = Vec::with_capacity(active_asset_indices.len());
    for (position, asset_index) in active_asset_indices.iter().copied().enumerate() {
        let latent = latents[position];
        if !latent.is_finite()
            || latent < config.latent_epsilon as f64
            || latent > 1.0 - config.latent_epsilon as f64
        {
            return Err(MidasError::Policy(
                "bounded impulse Beta latent is outside the supported interior".into(),
            ));
        }
        let (alpha, beta, d_alpha_d_raw, d_beta_d_raw) = beta_shapes(
            logits[asset_index * 2],
            logits[asset_index * 2 + 1],
            temperature,
            config,
        )?;
        let impulse = 2.0 * latent - 1.0;
        if !impulse.is_finite() || !(-1.0..=1.0).contains(&impulse) {
            return Err(MidasError::Policy(
                "bounded impulse Beta action is invalid".into(),
            ));
        }
        impulses[asset_index] = impulse;
        let log_gradient_alpha = beta_log_probability_derivative_alpha(alpha, beta, latent)?;
        let log_gradient_beta = beta_log_probability_derivative_beta(alpha, beta, latent)?;
        let entropy_gradient_alpha = beta_entropy_derivative_alpha(alpha, beta)?;
        let entropy_gradient_beta = beta_entropy_derivative_beta(alpha, beta)?;
        decisions.push(BetaDecision {
            actor_alpha_index: asset_index * 2,
            actor_beta_index: asset_index * 2 + 1,
            alpha,
            beta,
            latent,
            d_log_probability_d_raw_alpha: (log_gradient_alpha * d_alpha_d_raw) as f32,
            d_log_probability_d_raw_beta: (log_gradient_beta * d_beta_d_raw) as f32,
            d_entropy_d_raw_alpha: (entropy_gradient_alpha * d_alpha_d_raw) as f32,
            d_entropy_d_raw_beta: (entropy_gradient_beta * d_beta_d_raw) as f32,
        });
    }
    if impulses
        .iter()
        .any(|impulse| !impulse.is_finite() || !(-1.0..=1.0).contains(impulse))
    {
        return Err(MidasError::Policy(
            "bounded impulse Beta action failed validation".into(),
        ));
    }
    Ok((impulses, decisions))
}

pub(super) fn evaluate_incremental_trade_sample(
    logits: &[f32],
    tradable: &[bool],
    sample: &PolicyActionSample,
    temperature: f32,
    config: &ActionDistributionConfig,
) -> Result<ImpulseBetaEvaluation> {
    validate_distribution_inputs(
        &NeuralOutput {
            logits: logits.to_vec(),
            value: 0.0,
            turnover_cost_value: None,
        },
        tradable,
        sample.maximum_trade_fraction as f64,
        temperature,
        config,
    )?;
    let expected_indices = sample.active_asset_indices.clone();
    let mut sorted_indices = expected_indices.clone();
    sorted_indices.sort_unstable();
    let canonical_indices = active_asset_indices(tradable)?;
    if sorted_indices != canonical_indices
        || sample.latent_samples.len() != expected_indices.len()
        || sample.trade_impulses.len() != tradable.len()
    {
        return Err(MidasError::Policy(
            "PPO portfolio action sample dimensions or mask mismatch".into(),
        ));
    }
    let latents = sample
        .latent_samples
        .iter()
        .map(|value| *value as f64)
        .collect::<Vec<_>>();
    let (impulses, decisions) = evaluate_impulses(
        logits,
        tradable,
        &expected_indices,
        &latents,
        temperature,
        config,
    )?;
    if impulses
        .iter()
        .zip(&sample.trade_impulses)
        .any(|(actual, stored)| !stored.is_finite() || (*actual - *stored as f64).abs() > 1e-5)
    {
        return Err(MidasError::Policy(
            "PPO incremental trade sample does not match its latent action".into(),
        ));
    }
    let log_probability = decisions
        .iter()
        .map(|decision| beta_log_probability(decision.alpha, decision.beta, decision.latent))
        .sum::<Result<f64>>()? as f32;
    let entropy = decisions
        .iter()
        .map(|decision| beta_entropy(decision.alpha, decision.beta))
        .sum::<Result<f64>>()? as f32;
    if !log_probability.is_finite() || !entropy.is_finite() {
        return Err(MidasError::Policy(
            "non-finite bounded impulse Beta PPO density".into(),
        ));
    }
    Ok(ImpulseBetaEvaluation {
        log_probability,
        entropy,
        decisions,
    })
}

#[cfg(test)]
pub(super) fn evaluate_impulse_beta_sample(
    logits: &[f32],
    tradable: &[bool],
    sample: &PolicyActionSample,
    temperature: f32,
    config: &ActionDistributionConfig,
) -> Result<ImpulseBetaEvaluation> {
    evaluate_incremental_trade_sample(logits, tradable, sample, temperature, config)
}

pub(super) fn beta_shapes(
    raw_alpha: f32,
    raw_beta: f32,
    temperature: f32,
    config: &ActionDistributionConfig,
) -> Result<(f64, f64, f64, f64)> {
    let shape = |raw: f32| -> Result<(f64, f64)> {
        if !raw.is_finite() {
            return Err(MidasError::Policy("non-finite Beta shape logit".into()));
        }
        let raw = raw as f64;
        let base = softplus(raw) + config.minimum_shape as f64;
        let unclamped = base / temperature as f64;
        let shape = unclamped.clamp(config.minimum_shape as f64, config.maximum_shape as f64);
        let derivative =
            if unclamped > config.minimum_shape as f64 && unclamped < config.maximum_shape as f64 {
                sigmoid(raw) / temperature as f64
            } else {
                0.0
            };
        if !shape.is_finite() || shape <= 0.0 || !derivative.is_finite() {
            return Err(MidasError::Policy("invalid bounded Beta shape".into()));
        }
        Ok((shape, derivative))
    };
    let (alpha, d_alpha) = shape(raw_alpha)?;
    let (beta, d_beta) = shape(raw_beta)?;
    Ok((alpha, beta, d_alpha, d_beta))
}

fn softplus(value: f64) -> f64 {
    if value > 20.0 {
        value
    } else {
        (1.0 + value.exp()).ln()
    }
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

pub(super) fn beta_log_probability(alpha: f64, beta: f64, latent: f64) -> Result<f64> {
    if !alpha.is_finite()
        || !beta.is_finite()
        || alpha <= 0.0
        || beta <= 0.0
        || !latent.is_finite()
        || !(0.0..1.0).contains(&latent)
    {
        return Err(MidasError::Policy(
            "invalid Beta log-probability input".into(),
        ));
    }
    let value = (alpha - 1.0) * latent.ln() + (beta - 1.0) * (1.0 - latent).ln()
        - beta_log_normalizer(alpha, beta);
    if !value.is_finite() {
        return Err(MidasError::Policy("non-finite Beta log probability".into()));
    }
    Ok(value)
}

fn beta_log_normalizer(alpha: f64, beta: f64) -> f64 {
    libm::lgamma(alpha) + libm::lgamma(beta) - libm::lgamma(alpha + beta)
}

pub(super) fn beta_log_probability_derivative_alpha(
    alpha: f64,
    beta: f64,
    latent: f64,
) -> Result<f64> {
    let value = latent.ln() - digamma(alpha) + digamma(alpha + beta);
    if value.is_finite() {
        Ok(value)
    } else {
        Err(MidasError::Policy(
            "non-finite Beta alpha log-probability gradient".into(),
        ))
    }
}

fn beta_log_probability_derivative_beta(alpha: f64, beta: f64, latent: f64) -> Result<f64> {
    let value = (1.0 - latent).ln() - digamma(beta) + digamma(alpha + beta);
    if value.is_finite() {
        Ok(value)
    } else {
        Err(MidasError::Policy(
            "non-finite Beta beta log-probability gradient".into(),
        ))
    }
}

pub(super) fn beta_entropy(alpha: f64, beta: f64) -> Result<f64> {
    let sum = alpha + beta;
    let value = beta_log_normalizer(alpha, beta)
        - (alpha - 1.0) * digamma(alpha)
        - (beta - 1.0) * digamma(beta)
        + (sum - 2.0) * digamma(sum);
    if value.is_finite() {
        Ok(value)
    } else {
        Err(MidasError::Policy("non-finite Beta entropy".into()))
    }
}

pub(super) fn beta_entropy_derivative_alpha(alpha: f64, beta: f64) -> Result<f64> {
    let value = -(alpha - 1.0) * trigamma(alpha) + (alpha + beta - 2.0) * trigamma(alpha + beta);
    if value.is_finite() {
        Ok(value)
    } else {
        Err(MidasError::Policy(
            "non-finite Beta entropy gradient".into(),
        ))
    }
}

fn beta_entropy_derivative_beta(alpha: f64, beta: f64) -> Result<f64> {
    let value = -(beta - 1.0) * trigamma(beta) + (alpha + beta - 2.0) * trigamma(alpha + beta);
    if value.is_finite() {
        Ok(value)
    } else {
        Err(MidasError::Policy(
            "non-finite Beta entropy gradient".into(),
        ))
    }
}

// Recurrence plus an asymptotic expansion gives stable digamma/trigamma
// values for the bounded positive shapes used by the action distribution.
fn digamma(mut value: f64) -> f64 {
    let mut result = 0.0;
    while value < 8.0 {
        result -= 1.0 / value;
        value += 1.0;
    }
    let inverse = 1.0 / value;
    let inverse_squared = inverse * inverse;
    result + value.ln()
        - 0.5 * inverse
        - inverse_squared
            * (1.0 / 12.0
                - inverse_squared
                    * (1.0 / 120.0
                        - inverse_squared
                            * (1.0 / 252.0
                                - inverse_squared * (1.0 / 240.0 - inverse_squared * 1.0 / 132.0))))
}

fn trigamma(mut value: f64) -> f64 {
    let mut result = 0.0;
    while value < 8.0 {
        result += 1.0 / (value * value);
        value += 1.0;
    }
    let inverse = 1.0 / value;
    let inverse_squared = inverse * inverse;
    result + inverse + 0.5 * inverse_squared + inverse_squared * inverse / 6.0
        - inverse_squared * inverse_squared * inverse / 30.0
        + inverse_squared.powi(3) * inverse / 42.0
        - inverse_squared.powi(4) * inverse / 30.0
}
