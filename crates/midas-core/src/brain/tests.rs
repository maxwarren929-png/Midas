use super::burn_runtime::BurnRuntimeParameters;
use super::head_training::accumulate_dense_gradient;
use super::impulse_beta::{
    beta_entropy, beta_entropy_derivative_alpha, beta_log_probability,
    beta_log_probability_derivative_alpha, beta_shapes, bounded_impulse_beta_evaluation,
    evaluate_impulse_beta_sample,
};
use super::*;
use crate::{
    action::{ActionValidationContext, ActionValidator, LongOnlyActionValidator, ProposedAction},
    config::{
        ActionDistributionConfig, Activation, EvolvableTraitsConfig, LearningConfig, LearningKind,
    },
    learning::{PpoTrainingBatch, PpoTrainingSample},
    observation::Observation,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn architecture() -> BrainArchitecture {
    BrainArchitecture {
        version: 1,
        observation_schema_hash: "schema".into(),
        observation_size: 2,
        asset_count: 1,
        hidden_layers: vec![2],
        memory: MemoryArchitecture { size: 1 },
        actor_output_size: 2,
        critic_output_size: 1,
        activation: Activation::Relu,
        action_distribution: ActionDistributionConfig::default(),
    }
}

fn learning_config() -> LearningConfig {
    LearningConfig {
        implementation: LearningKind::Ppo,
        learning_rate: 0.01,
        gamma: 0.99,
        gae_lambda: 0.95,
        ppo_clip: 0.2,
        critic_coefficient: 0.5,
        entropy_coefficient: 0.01,
        gradient_clip: 1.0,
        rollout_length: 1,
        minibatch_size: 1,
        sequence_length: 1,
        normalize_advantages: false,
        optimisation_epochs: 1,
    }
}

fn ppo_sample(
    values: Vec<f32>,
    action_index: usize,
    advantage: f32,
    return_value: f32,
    old_log_probability: f32,
) -> PpoTrainingSample {
    PpoTrainingSample {
        observation: Observation {
            schema_hash: "schema".into(),
            values,
        },
        action_sample: Some(PolicyActionSample {
            active_asset_indices: vec![0],
            latent_samples: vec![if action_index == 0 { 0.25 } else { 0.75 }],
            trade_impulses: vec![if action_index == 0 { -0.5 } else { 0.5 }],
            maximum_trade_fraction: 0.05,
            log_probability: old_log_probability,
            entropy: 0.0,
        }),
        old_log_probability: Some(old_log_probability),
        advantage: Some(advantage),
        return_value: Some(return_value),
        value_estimate: Some(0.0),
        recurrent_state: NeuralRuntimeState {
            memory: vec![0.0],
            inference_steps: 0,
        },
        tradable: vec![true],
        temperature: Some(1.0),
        invalid_action: false,
    }
}

fn deterministic_head_model() -> BurnNeuralModel {
    let mut architecture = architecture();
    architecture.activation = Activation::Silu;
    let genome = BrainGenome::random(&architecture, &EvolvableTraitsConfig::default(), 31).unwrap();
    let lifetime = LifetimeParameters {
        hidden: vec![DenseGenome {
            inputs: 2,
            outputs: 2,
            weights: vec![1.0, 0.0, 0.0, 1.0],
            bias: vec![0.0, 0.0],
        }],
        recurrent: DenseGenome {
            inputs: 3,
            outputs: 1,
            weights: vec![1.0, 0.0, 0.0],
            bias: vec![0.0],
        },
        actor: DenseGenome {
            inputs: 1,
            outputs: 2,
            weights: vec![0.0, 0.0],
            bias: vec![0.0, 0.0],
        },
        critic: DenseGenome {
            inputs: 1,
            outputs: 1,
            weights: vec![0.0],
            bias: vec![0.0],
        },
    };
    lifetime.validate(&architecture).unwrap();
    let burn_device = Default::default();
    BurnNeuralModel {
        architecture,
        cached: BurnRuntimeParameters::from_lifetime(&lifetime, &burn_device),
        lifetime,
        runtime: NeuralRuntimeState {
            memory: vec![0.0],
            inference_steps: 0,
        },
        optimizer: OptimizerState::empty(&genome.architecture),
    }
}

#[test]
fn burn_training_changes_lifetime_parameters_without_changing_genome() {
    let architecture = architecture();
    let traits = EvolvableTraitsConfig::default();
    let genome = BrainGenome::random(&architecture, &traits, 7).unwrap();
    let before_genome = genome.clone();
    let device = ComputeDevice::CpuNdArray {
        device: "default".into(),
    };
    let mut model = BurnNeuralModelFactory
        .build(&genome, &architecture, &device)
        .unwrap();
    let before = model.lifetime_parameters().unwrap();
    let batch = PpoTrainingBatch {
        samples: vec![PpoTrainingSample {
            observation: Observation {
                schema_hash: "schema".into(),
                values: vec![0.2, -0.1],
            },
            action_sample: Some(PolicyActionSample {
                active_asset_indices: vec![0],
                latent_samples: vec![0.5],
                trade_impulses: vec![0.0],
                maximum_trade_fraction: 0.05,
                log_probability: -std::f32::consts::LN_2,
                entropy: 0.0,
            }),
            old_log_probability: Some(-std::f32::consts::LN_2),
            advantage: Some(1.0),
            return_value: Some(1.0),
            value_estimate: Some(0.0),
            recurrent_state: NeuralRuntimeState {
                memory: vec![0.0],
                inference_steps: 0,
            },
            tradable: vec![true],
            temperature: Some(1.0),
            invalid_action: false,
        }],
        sequence_length: 1,
    };
    let metrics = model.train_batch(&batch, &learning_config()).unwrap();
    assert!(metrics.rollout_size > 0);
    assert!(metrics.gradient_norm <= 1.0 + 1e-6);
    assert_ne!(before, model.lifetime_parameters().unwrap());
    assert_eq!(before_genome, genome);
}

#[test]
fn ppo_head_learning_changes_actor_weights_regression() {
    let mut model = deterministic_head_model();
    let before = model.lifetime_parameters().unwrap();
    let batch = PpoTrainingBatch {
        samples: vec![ppo_sample(vec![0.8, 0.3], 1, 1.0, 0.5, -0.7)],
        sequence_length: 1,
    };

    model.train_batch(&batch, &learning_config()).unwrap();
    let after = model.lifetime_parameters().unwrap();
    assert_ne!(
        before.actor.weights, after.actor.weights,
        "PPO must update the actor output weight matrix, not only its bias"
    );
}

#[test]
fn full_head_training_updates_heads_but_freezes_encoder_and_genome() {
    let architecture = architecture();
    let genome = BrainGenome::random(&architecture, &EvolvableTraitsConfig::default(), 41).unwrap();
    let before_genome = genome.clone();
    let mut model = deterministic_head_model();
    let before = model.lifetime_parameters().unwrap();
    let builds_before = model.runtime_parameter_builds();
    let batch = PpoTrainingBatch {
        samples: vec![ppo_sample(vec![0.6, -0.2], 1, 1.0, 2.0, -0.7)],
        sequence_length: 1,
    };
    model.train_batch(&batch, &learning_config()).unwrap();
    let after = model.lifetime_parameters().unwrap();
    assert_ne!(before.actor.weights, after.actor.weights);
    assert_ne!(before.actor.bias, after.actor.bias);
    assert_ne!(before.critic.weights, after.critic.weights);
    assert_ne!(before.critic.bias, after.critic.bias);
    assert_eq!(before.hidden, after.hidden);
    assert_eq!(before.recurrent, after.recurrent);
    assert_eq!(before_genome, genome);

    let optimizer = model.optimizer_state().unwrap();
    assert_eq!(
        optimizer.actor.weight_momentum.len(),
        architecture.memory.size * architecture.actor_output_size
    );
    assert_eq!(
        optimizer.actor.bias_momentum.len(),
        architecture.actor_output_size
    );
    assert_eq!(
        optimizer.critic.weight_momentum.len(),
        architecture.memory.size * architecture.critic_output_size
    );
    assert_eq!(
        optimizer.critic.bias_momentum.len(),
        architecture.critic_output_size
    );
    assert_eq!(model.runtime_parameter_builds(), builds_before + 1);
}

#[test]
fn deterministic_evaluation_does_not_decay_temperature() {
    let architecture = architecture();
    let mut genome =
        BrainGenome::random(&architecture, &EvolvableTraitsConfig::default(), 52).unwrap();
    genome.traits.action_temperature = 1.0;
    genome.traits.temperature_decay = 0.5;
    let device = ComputeDevice::CpuNdArray {
        device: "default".into(),
    };
    let mut policy = NeuralPolicy::new_with_seed(
        genome.clone(),
        BurnNeuralModelFactory
            .build(&genome, &architecture, &device)
            .unwrap(),
        7,
    );
    let observation = Observation {
        schema_hash: "schema".into(),
        values: vec![0.1, -0.2],
    };
    let context = PolicyContext {
        tradable: &[true],
        current_allocations: &[1.0, 0.0],
    };
    let before = match policy.snapshot().unwrap() {
        PolicySnapshot::Neural { temperature, .. } => temperature,
        _ => unreachable!(),
    };
    policy
        .evaluate_action(&observation, &context, false, 1.0)
        .unwrap();
    let after_deterministic = match policy.snapshot().unwrap() {
        PolicySnapshot::Neural { temperature, .. } => temperature,
        _ => unreachable!(),
    };
    assert_eq!(before, after_deterministic);
    policy
        .evaluate_action(&observation, &context, true, 1.0)
        .unwrap();
    let after_stochastic = match policy.snapshot().unwrap() {
        PolicySnapshot::Neural { temperature, .. } => temperature,
        _ => unreachable!(),
    };
    assert!((after_stochastic - before * 0.5).abs() < 1e-6);
    policy
        .evaluate_action(&observation, &context, false, 1.0)
        .unwrap();
    let after_bootstrap = match policy.snapshot().unwrap() {
        PolicySnapshot::Neural { temperature, .. } => temperature,
        _ => unreachable!(),
    };
    assert_eq!(after_stochastic, after_bootstrap);
}

#[test]
fn dense_gradient_matches_outer_product_exactly() {
    let representation = [2.0_f32, -1.0];
    let output_gradient = [0.5_f32, -0.25];
    let mut weight_gradient = vec![0.0; 4];
    let mut bias_gradient = vec![0.0; 2];
    accumulate_dense_gradient(
        &representation,
        &output_gradient,
        &mut weight_gradient,
        &mut bias_gradient,
    )
    .unwrap();
    assert_eq!(weight_gradient, vec![1.0, -0.5, -0.5, 0.25]);
    assert_eq!(bias_gradient, vec![0.5, -0.25]);
}

#[test]
fn clipped_policy_gradient_suppresses_head_update_but_keeps_objective_valid() {
    let mut model = deterministic_head_model();
    let before = model.lifetime_parameters().unwrap();
    let mut config = learning_config();
    config.critic_coefficient = 0.0;
    config.entropy_coefficient = 0.0;
    let sample = ppo_sample(vec![1.0, 0.0], 1, 1.0, 0.0, -2.0 * std::f32::consts::LN_2);
    let metrics = model
        .train_batch(
            &PpoTrainingBatch {
                samples: vec![sample],
                sequence_length: 1,
            },
            &config,
        )
        .unwrap();
    let after = model.lifetime_parameters().unwrap();
    assert_eq!(before.actor, after.actor);
    assert_eq!(before.critic, after.critic);
    assert_eq!(metrics.gradient_norm, 0.0);
}

#[test]
fn actor_learning_is_state_dependent_for_two_encoded_states() {
    let mut model = deterministic_head_model();
    let mut config = learning_config();
    config.learning_rate = 0.4;
    config.ppo_clip = 10.0;
    config.critic_coefficient = 0.0;
    config.entropy_coefficient = 0.0;
    config.gradient_clip = 100.0;
    config.optimisation_epochs = 8;
    config.minibatch_size = 2;
    let batch = PpoTrainingBatch {
        samples: vec![
            ppo_sample(vec![1.0, 0.0], 1, 1.0, 0.0, -std::f32::consts::LN_2),
            ppo_sample(vec![-1.0, 0.0], 0, 1.0, 0.0, -std::f32::consts::LN_2),
        ],
        sequence_length: 1,
    };
    model.train_batch(&batch, &config).unwrap();
    let zero = NeuralRuntimeState {
        memory: vec![0.0],
        inference_steps: 0,
    };
    let state_a = model
        .encode_observation(
            &Observation {
                schema_hash: "schema".into(),
                values: vec![1.0, 0.0],
            },
            &zero,
        )
        .unwrap();
    let state_b = model
        .encode_observation(
            &Observation {
                schema_hash: "schema".into(),
                values: vec![-1.0, 0.0],
            },
            &zero,
        )
        .unwrap();
    let output_a = model
        .evaluate_representation(&state_a.representation)
        .unwrap();
    let output_b = model
        .evaluate_representation(&state_b.representation)
        .unwrap();
    let evaluation_a = bounded_impulse_beta_evaluation(
        output_a,
        &[true],
        1.0,
        false,
        1.0,
        &mut ChaCha8Rng::seed_from_u64(1),
        &model.architecture.action_distribution,
    )
    .unwrap();
    let evaluation_b = bounded_impulse_beta_evaluation(
        output_b,
        &[true],
        1.0,
        false,
        1.0,
        &mut ChaCha8Rng::seed_from_u64(1),
        &model.architecture.action_distribution,
    )
    .unwrap();
    let impulse_a = match evaluation_a.proposed_action {
        ProposedAction::TradeImpulses(value) => value.impulses[0],
        ProposedAction::Allocation(_) => panic!("expected impulse action"),
    };
    let impulse_b = match evaluation_b.proposed_action {
        ProposedAction::TradeImpulses(value) => value.impulses[0],
        ProposedAction::Allocation(_) => panic!("expected impulse action"),
    };
    assert!(impulse_a > impulse_b);
    assert!(impulse_a > 0.0);
    assert!(impulse_b < 0.0);
}

#[test]
fn critic_learning_is_state_dependent_for_two_encoded_states() {
    let mut model = deterministic_head_model();
    let mut config = learning_config();
    config.learning_rate = 0.4;
    config.ppo_clip = 10.0;
    config.critic_coefficient = 1.0;
    config.entropy_coefficient = 0.0;
    config.gradient_clip = 100.0;
    config.optimisation_epochs = 8;
    config.minibatch_size = 2;
    let batch = PpoTrainingBatch {
        samples: vec![
            ppo_sample(vec![1.0, 0.0], 0, 0.0, 2.0, -std::f32::consts::LN_2),
            ppo_sample(vec![-1.0, 0.0], 0, 0.0, -2.0, -std::f32::consts::LN_2),
        ],
        sequence_length: 1,
    };
    model.train_batch(&batch, &config).unwrap();
    let zero = NeuralRuntimeState {
        memory: vec![0.0],
        inference_steps: 0,
    };
    let state_a = model
        .encode_observation(
            &Observation {
                schema_hash: "schema".into(),
                values: vec![1.0, 0.0],
            },
            &zero,
        )
        .unwrap();
    let state_b = model
        .encode_observation(
            &Observation {
                schema_hash: "schema".into(),
                values: vec![-1.0, 0.0],
            },
            &zero,
        )
        .unwrap();
    let value_a = model
        .evaluate_representation(&state_a.representation)
        .unwrap()
        .value;
    let value_b = model
        .evaluate_representation(&state_b.representation)
        .unwrap()
        .value;
    assert!(value_a > value_b);
}

#[test]
fn malformed_full_head_optimizer_state_is_rejected_on_restore() {
    let architecture = architecture();
    let genome = BrainGenome::random(&architecture, &EvolvableTraitsConfig::default(), 51).unwrap();
    let device = ComputeDevice::CpuNdArray {
        device: "default".into(),
    };
    let model = BurnNeuralModelFactory
        .build(&genome, &architecture, &device)
        .unwrap();
    let base_state = model.snapshot_state().unwrap();
    for invalid in 0..4 {
        let NeuralModelState::Burn {
            parameters,
            runtime,
            mut optimizer,
        } = base_state.clone()
        else {
            panic!("expected Burn state")
        };
        match invalid {
            0 => {
                optimizer.actor.weight_momentum.pop();
            }
            1 => {
                optimizer.actor.bias_momentum.pop();
            }
            2 => {
                optimizer.critic.weight_momentum.pop();
            }
            _ => {
                optimizer.critic.bias_momentum[0] = f32::NAN;
            }
        }
        let error = match BurnNeuralModelFactory.restore(
            &genome,
            &architecture,
            NeuralModelState::Burn {
                parameters,
                runtime,
                optimizer,
            },
            &device,
        ) {
            Ok(_) => panic!("shape-invalid optimizer state was accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("optimizer"));
    }
}

#[test]
fn genome_validation_rejects_hidden_layout_changes() {
    let architecture = architecture();
    let mut genome =
        BrainGenome::random(&architecture, &EvolvableTraitsConfig::default(), 17).unwrap();
    genome.hidden.pop();
    assert!(genome.validate(&architecture).is_err());
}

#[test]
fn bounded_impulse_beta_is_masked_and_seed_reproducible() {
    let output = NeuralOutput {
        logits: vec![0.0, 2.0, 0.0, 1.0],
        value: 0.0,
    };
    let mut deterministic_rng = ChaCha8Rng::seed_from_u64(1);
    let deterministic = bounded_impulse_beta_evaluation(
        output.clone(),
        &[true, false],
        0.5,
        false,
        1.0,
        &mut deterministic_rng,
        &ActionDistributionConfig::default(),
    )
    .unwrap();
    let impulses = match deterministic.proposed_action {
        ProposedAction::TradeImpulses(value) => value.impulses,
        ProposedAction::Allocation(_) => panic!("expected impulse action"),
    };
    assert_eq!(impulses.len(), 2);
    assert_eq!(impulses[1], 0.0);
    let mut left_rng = ChaCha8Rng::seed_from_u64(5);
    let mut right_rng = ChaCha8Rng::seed_from_u64(5);
    let left = bounded_impulse_beta_evaluation(
        output.clone(),
        &[true, true],
        0.5,
        true,
        1.0,
        &mut left_rng,
        &ActionDistributionConfig::default(),
    )
    .unwrap();
    let right = bounded_impulse_beta_evaluation(
        output,
        &[true, true],
        0.5,
        true,
        1.0,
        &mut right_rng,
        &ActionDistributionConfig::default(),
    )
    .unwrap();
    assert_eq!(left, right);
}

#[test]
fn neural_policy_can_request_two_asset_impulses() {
    let output = NeuralOutput {
        logits: vec![0.0, 0.0, 0.0, 0.0],
        value: 0.0,
    };
    let mut rng = ChaCha8Rng::seed_from_u64(9);
    let evaluation = bounded_impulse_beta_evaluation(
        output,
        &[true, true],
        0.5,
        false,
        1.0,
        &mut rng,
        &ActionDistributionConfig::default(),
    )
    .unwrap();
    let impulses = match evaluation.proposed_action {
        ProposedAction::TradeImpulses(value) => value.impulses,
        ProposedAction::Allocation(_) => panic!("expected impulse action"),
    };
    assert!(impulses.iter().all(|value| (-1.0..=1.0).contains(value)));
    assert_eq!(impulses.len(), 2);
}

#[test]
fn bounded_impulse_beta_distribution_is_valid_across_masks_caps_and_seeds() {
    for (tradable, cap) in [
        (vec![true, true, true], 1.0),
        (vec![true, false, true], 0.5),
        (vec![false, false, false], 0.25),
        (vec![true, true, true], 0.01),
    ] {
        for seed in 0..32_u64 {
            let output = NeuralOutput {
                logits: vec![0.3, -0.2, 1.0, -1.0, -0.7, 0.8],
                value: 0.0,
            };
            let evaluation = bounded_impulse_beta_evaluation(
                output,
                &tradable,
                cap,
                true,
                1.0,
                &mut ChaCha8Rng::seed_from_u64(seed),
                &ActionDistributionConfig::default(),
            )
            .unwrap();
            let impulses = match evaluation.proposed_action {
                ProposedAction::TradeImpulses(value) => value.impulses,
                ProposedAction::Allocation(_) => panic!("expected impulse action"),
            };
            assert_eq!(impulses.len(), 3);
            assert!(
                impulses
                    .iter()
                    .all(|value| value.is_finite() && (-1.0..=1.0).contains(value))
            );
            for (index, is_tradable) in tradable.iter().enumerate() {
                if !is_tradable {
                    assert_eq!(impulses[index], 0.0);
                }
            }
        }
    }
}

#[test]
fn bounded_impulse_beta_log_probability_recomputes_exactly() {
    let output = NeuralOutput {
        logits: vec![0.3, -0.2, 1.0, -1.0, -0.7, 0.8],
        value: 0.0,
    };
    let config = ActionDistributionConfig::default();
    let evaluation = bounded_impulse_beta_evaluation(
        output.clone(),
        &[true, false, true],
        0.5,
        true,
        1.0,
        &mut ChaCha8Rng::seed_from_u64(12),
        &config,
    )
    .unwrap();
    let sample = evaluation.action_sample.as_ref().unwrap();
    let recomputed =
        evaluate_impulse_beta_sample(&output.logits, &[true, false, true], sample, 1.0, &config)
            .unwrap();
    assert!((sample.log_probability - recomputed.log_probability).abs() < 1e-6);
    assert!((sample.entropy - recomputed.entropy).abs() < 1e-6);
}

#[test]
fn deterministic_impulse_evaluation_does_not_consume_rng() {
    let output = NeuralOutput {
        logits: vec![0.0, 0.0, 0.0, 0.0],
        value: 0.0,
    };
    let mut rng = ChaCha8Rng::seed_from_u64(77);
    let mut expected = rng.clone();
    bounded_impulse_beta_evaluation(
        output,
        &[true, true],
        0.5,
        false,
        1.0,
        &mut rng,
        &ActionDistributionConfig::default(),
    )
    .unwrap();
    assert_eq!(rng.r#gen::<u64>(), expected.r#gen::<u64>());
}

#[test]
fn sampled_valid_impulses_survive_central_validation_unchanged() {
    let evaluation = bounded_impulse_beta_evaluation(
        NeuralOutput {
            logits: vec![0.2, -0.3, 0.4, 0.1],
            value: 0.0,
        },
        &[true, true],
        0.5,
        true,
        1.0,
        &mut ChaCha8Rng::seed_from_u64(3),
        &ActionDistributionConfig::default(),
    )
    .unwrap();
    let proposed = evaluation.proposed_action.clone();
    let validated = LongOnlyActionValidator
        .validate_action(
            proposed.clone(),
            &ActionValidationContext {
                asset_count: 2,
                tradable: &[true, true],
                policy_maximum_asset_allocation: None,
                policy_maximum_trade_fraction: None,
                market_maximum_asset_allocation: None,
            },
        )
        .unwrap();
    match (proposed, validated) {
        (ProposedAction::TradeImpulses(a), crate::action::ValidatedAction::TradeImpulses(b)) => {
            assert_eq!(a.impulses, b.impulses());
        }
        _ => panic!("expected impulse validation"),
    }
}

#[test]
fn beta_log_probability_gradient_matches_finite_difference() {
    let config = ActionDistributionConfig::default();
    let raw_alpha = 0.35_f32;
    let raw_beta = -0.4_f32;
    let latent = 0.37_f64;
    let (alpha, beta, d_alpha, _) = beta_shapes(raw_alpha, raw_beta, 1.0, &config).unwrap();
    let analytic = beta_log_probability_derivative_alpha(alpha, beta, latent).unwrap() * d_alpha;
    let step = 1e-4_f32;
    let shifted = |raw: f32| {
        let (a, b, _, _) = beta_shapes(raw, raw_beta, 1.0, &config).unwrap();
        beta_log_probability(a, b, latent).unwrap()
    };
    let finite_difference =
        (shifted(raw_alpha + step) - shifted(raw_alpha - step)) / (2.0 * step as f64);
    assert!((analytic - finite_difference).abs() < 2e-3);
}

#[test]
fn beta_entropy_gradient_matches_finite_difference() {
    let config = ActionDistributionConfig::default();
    let raw_alpha = 0.35_f32;
    let raw_beta = -0.4_f32;
    let (alpha, beta, d_alpha, _) = beta_shapes(raw_alpha, raw_beta, 1.0, &config).unwrap();
    let analytic = beta_entropy_derivative_alpha(alpha, beta).unwrap() * d_alpha;
    let step = 1e-4_f32;
    let shifted = |raw: f32| {
        let (a, b, _, _) = beta_shapes(raw, raw_beta, 1.0, &config).unwrap();
        beta_entropy(a, b).unwrap()
    };
    let finite_difference =
        (shifted(raw_alpha + step) - shifted(raw_alpha - step)) / (2.0 * step as f64);
    assert!((analytic - finite_difference).abs() < 2e-3);
}
