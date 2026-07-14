use crate::{
    MidasError, PolicyFailureCategory, Result,
    action::ValidatedAllocation,
    brain::{NeuralRuntimeState, PolicyActionSample},
    config::LearningConfig,
    observation::Observation,
};
use chrono::NaiveDate;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Transition {
    pub observation: Option<Observation>,
    pub action: ValidatedAllocation,
    pub action_sample: Option<PolicyActionSample>,
    pub raw_policy_output: Option<Vec<f32>>,
    pub log_probability: Option<f32>,
    pub temperature: Option<f32>,
    pub value_estimate: Option<f32>,
    pub reward: f32,
    pub next_value: Option<f32>,
    pub done: bool,
    pub truncated: bool,
    pub recurrent_state_before: Option<NeuralRuntimeState>,
    pub recurrent_state_after: Option<NeuralRuntimeState>,
    pub date: NaiveDate,
    pub tradable: Vec<bool>,
    pub invalid_action: bool,
    pub failure_category: Option<PolicyFailureCategory>,
    pub actor_eligible: bool,
    pub critic_eligible: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PpoTrainingSample {
    pub observation: Observation,
    pub action_index: Option<usize>,
    pub old_log_probability: Option<f32>,
    pub advantage: Option<f32>,
    pub return_value: Option<f32>,
    pub value_estimate: Option<f32>,
    pub recurrent_state: NeuralRuntimeState,
    pub tradable: Vec<bool>,
    pub temperature: Option<f32>,
    pub invalid_action: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PpoTrainingBatch {
    pub samples: Vec<PpoTrainingSample>,
    pub sequence_length: usize,
}
impl PpoTrainingBatch {
    pub fn len(&self) -> usize {
        self.samples.len()
    }
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum LearningUpdate {
    None,
    Ppo(PpoTrainingBatch),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TrainingMetrics {
    pub rollout_size: usize,
    pub policy_loss: f32,
    pub value_loss: f32,
    pub entropy: f32,
    pub total_loss: f32,
    pub gradient_norm: f32,
    pub update_index: u64,
}
impl TrainingMetrics {
    pub fn empty() -> Self {
        Self {
            rollout_size: 0,
            policy_loss: 0.0,
            value_loss: 0.0,
            entropy: 0.0,
            total_loss: 0.0,
            gradient_norm: 0.0,
            update_index: 0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EpisodeEndContext {
    pub date: NaiveDate,
    pub final_value: f32,
    pub truncated: bool,
}

pub trait LearningStrategy: Send {
    fn observe_transition(&mut self, transition: Transition) -> Result<LearningUpdate>;
    fn end_episode(&mut self, context: &EpisodeEndContext) -> Result<LearningUpdate>;
    fn commit_update(&mut self) -> Result<()>;
    fn snapshot(&self) -> Result<LearningStrategyState>;
    fn fork(&self) -> Result<Box<dyn LearningStrategy>>;
    fn name(&self) -> &'static str;
    fn training_config(&self) -> Option<LearningConfig> {
        None
    }
}

pub struct NoLearning;
impl LearningStrategy for NoLearning {
    fn observe_transition(&mut self, _: Transition) -> Result<LearningUpdate> {
        Ok(LearningUpdate::None)
    }
    fn end_episode(&mut self, _: &EpisodeEndContext) -> Result<LearningUpdate> {
        Ok(LearningUpdate::None)
    }
    fn commit_update(&mut self) -> Result<()> {
        Ok(())
    }
    fn snapshot(&self) -> Result<LearningStrategyState> {
        Ok(LearningStrategyState::NoLearning)
    }
    fn fork(&self) -> Result<Box<dyn LearningStrategy>> {
        Ok(Box::new(Self))
    }
    fn name(&self) -> &'static str {
        "no_learning"
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RolloutBuffer {
    capacity: usize,
    transitions: Vec<Transition>,
}
impl RolloutBuffer {
    pub fn new(capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(MidasError::Config(
                "rollout capacity must be positive".into(),
            ));
        }
        Ok(Self {
            capacity,
            transitions: Vec::with_capacity(capacity),
        })
    }
    pub fn len(&self) -> usize {
        self.transitions.len()
    }
    pub fn is_empty(&self) -> bool {
        self.transitions.is_empty()
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn transitions(&self) -> &[Transition] {
        &self.transitions
    }
    pub fn push(&mut self, transition: Transition) -> Result<()> {
        if self.transitions.len() >= self.capacity {
            return Err(MidasError::Policy(
                "rollout buffer capacity exceeded".into(),
            ));
        }
        if !transition.reward.is_finite()
            || transition
                .value_estimate
                .is_some_and(|value| !value.is_finite())
            || transition
                .next_value
                .is_some_and(|value| !value.is_finite())
            || transition
                .log_probability
                .is_some_and(|value| !value.is_finite())
            || transition
                .temperature
                .is_some_and(|value| !value.is_finite())
        {
            return Err(MidasError::Policy(
                "rollout transition is non-finite".into(),
            ));
        }
        self.transitions.push(transition);
        Ok(())
    }
    pub fn clear(&mut self) {
        self.transitions.clear();
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[allow(clippy::large_enum_variant)]
pub enum LearningStrategyState {
    NoLearning,
    Ppo {
        config: LearningConfig,
        rollout: RolloutBuffer,
        rng: ChaCha8Rng,
        update_index: u64,
        pending_update: bool,
    },
}

pub struct PpoLearning {
    config: LearningConfig,
    rollout: RolloutBuffer,
    rng: ChaCha8Rng,
    update_index: u64,
    pending_update: bool,
}
impl PpoLearning {
    pub fn new(config: LearningConfig, seed: u64) -> Result<Self> {
        if config.implementation != crate::config::LearningKind::Ppo {
            return Err(MidasError::Config(
                "PpoLearning requires learning.implementation = ppo".into(),
            ));
        }
        if !config.learning_rate.is_finite()
            || config.learning_rate <= 0.0
            || !config.gamma.is_finite()
            || !(0.0..=1.0).contains(&config.gamma)
            || !config.gae_lambda.is_finite()
            || !(0.0..=1.0).contains(&config.gae_lambda)
            || !config.ppo_clip.is_finite()
            || config.ppo_clip < 0.0
            || !config.critic_coefficient.is_finite()
            || config.critic_coefficient < 0.0
            || !config.entropy_coefficient.is_finite()
            || config.entropy_coefficient < 0.0
            || !config.gradient_clip.is_finite()
            || config.gradient_clip <= 0.0
            || config.rollout_length == 0
            || config.minibatch_size == 0
            || config.minibatch_size > config.rollout_length
            || config.sequence_length == 0
            || config.sequence_length > config.rollout_length
            || config.optimisation_epochs == 0
        {
            return Err(MidasError::Config(
                "invalid PPO learning configuration".into(),
            ));
        }
        Ok(Self {
            rollout: RolloutBuffer::new(config.rollout_length)?,
            config,
            rng: ChaCha8Rng::seed_from_u64(seed),
            update_index: 0,
            pending_update: false,
        })
    }
    pub fn config(&self) -> &LearningConfig {
        &self.config
    }
    pub fn rollout(&self) -> &RolloutBuffer {
        &self.rollout
    }
    pub fn update_index(&self) -> u64 {
        self.update_index
    }
    fn make_update(&mut self) -> Result<LearningUpdate> {
        let transitions = self.rollout.transitions();
        if transitions.is_empty() {
            return Ok(LearningUpdate::None);
        }
        let (advantages, returns) =
            generalized_advantages_masked(transitions, self.config.gamma, self.config.gae_lambda)?;
        let mut samples = transitions
            .iter()
            .zip(advantages)
            .zip(returns)
            .filter_map(|((transition, advantage), return_value)| {
                let observation = transition.observation.clone()?;
                let recurrent_state = transition.recurrent_state_before.clone()?;
                let actor_sample = transition
                    .actor_eligible
                    .then_some(())
                    .and(transition.action_sample.as_ref())
                    .and_then(|sample| {
                        transition
                            .log_probability
                            .zip(transition.temperature)
                            .zip(advantage)
                            .map(|((old_log_probability, temperature), advantage)| {
                                (
                                    sample.action_index,
                                    old_log_probability,
                                    temperature,
                                    advantage,
                                )
                            })
                    });
                let critic_sample = transition
                    .critic_eligible
                    .then_some(())
                    .and(transition.value_estimate.zip(return_value));
                if actor_sample.is_none() && critic_sample.is_none() {
                    return None;
                }
                Some(PpoTrainingSample {
                    observation,
                    action_index: actor_sample.map(|sample| sample.0),
                    old_log_probability: actor_sample.map(|sample| sample.1),
                    advantage: actor_sample.map(|sample| sample.3),
                    return_value: critic_sample.map(|sample| sample.1),
                    value_estimate: critic_sample.map(|sample| sample.0),
                    recurrent_state,
                    tradable: transition.tradable.clone(),
                    temperature: actor_sample.map(|sample| sample.2),
                    invalid_action: transition.invalid_action,
                })
            })
            .collect::<Vec<_>>();
        if samples.is_empty() {
            self.pending_update = true;
            return Ok(LearningUpdate::None);
        }
        if self.config.normalize_advantages {
            let actor_count = samples
                .iter()
                .filter(|sample| sample.advantage.is_some())
                .count();
            let mean = samples
                .iter()
                .filter_map(|sample| sample.advantage)
                .sum::<f32>()
                / actor_count.max(1) as f32;
            let variance = samples
                .iter()
                .filter_map(|sample| sample.advantage)
                .map(|advantage| (advantage - mean).powi(2))
                .sum::<f32>()
                / actor_count.max(1) as f32;
            let standard_deviation = variance.sqrt().max(1e-8);
            for sample in &mut samples {
                if let Some(advantage) = sample.advantage {
                    sample.advantage = Some((advantage - mean) / standard_deviation);
                }
            }
        }
        // Shuffle complete recurrent sequences, never individual timesteps.
        let sequence = self.config.sequence_length.max(1);
        let chunk_count = samples.len().div_ceil(sequence);
        let mut order = (0..chunk_count).collect::<Vec<_>>();
        for index in (1..order.len()).rev() {
            let swap = self.rng.gen_range(0..=index);
            order.swap(index, swap);
        }
        let shuffled = if transitions
            .iter()
            .any(|transition| !transition.actor_eligible || !transition.critic_eligible)
        {
            // Masked environment steps are intentionally absent from model
            // tensors but remain in the rollout chronology. Keep eligible
            // samples in source order rather than stitching across a masked
            // recurrent boundary.
            samples
        } else {
            order
                .into_iter()
                .flat_map(|chunk| {
                    samples
                        .iter()
                        .skip(chunk * sequence)
                        .take(sequence)
                        .cloned()
                        .collect::<Vec<_>>()
                })
                .collect()
        };
        self.pending_update = true;
        Ok(LearningUpdate::Ppo(PpoTrainingBatch {
            samples: shuffled,
            sequence_length: self.config.sequence_length,
        }))
    }
}
impl LearningStrategy for PpoLearning {
    fn observe_transition(&mut self, transition: Transition) -> Result<LearningUpdate> {
        if self.pending_update {
            return Err(MidasError::Policy(
                "previous PPO update must be committed before observing more transitions".into(),
            ));
        }
        self.rollout.push(transition)?;
        if self.rollout.len() >= self.config.rollout_length {
            self.make_update()
        } else {
            Ok(LearningUpdate::None)
        }
    }
    fn end_episode(&mut self, _: &EpisodeEndContext) -> Result<LearningUpdate> {
        if self.pending_update {
            return Err(MidasError::Policy(
                "previous PPO update must be committed before ending the episode".into(),
            ));
        }
        self.make_update()
    }
    fn commit_update(&mut self) -> Result<()> {
        if self.pending_update {
            self.rollout.clear();
            self.pending_update = false;
            self.update_index = self.update_index.saturating_add(1);
        }
        Ok(())
    }
    fn snapshot(&self) -> Result<LearningStrategyState> {
        Ok(LearningStrategyState::Ppo {
            config: self.config.clone(),
            rollout: self.rollout.clone(),
            rng: self.rng.clone(),
            update_index: self.update_index,
            pending_update: self.pending_update,
        })
    }
    fn fork(&self) -> Result<Box<dyn LearningStrategy>> {
        Ok(Box::new(Self {
            config: self.config.clone(),
            rollout: self.rollout.clone(),
            rng: self.rng.clone(),
            update_index: self.update_index,
            pending_update: self.pending_update,
        }))
    }
    fn name(&self) -> &'static str {
        "ppo"
    }
    fn training_config(&self) -> Option<LearningConfig> {
        Some(self.config.clone())
    }
}

pub fn learning_from_state(state: LearningStrategyState) -> Result<Box<dyn LearningStrategy>> {
    match state {
        LearningStrategyState::NoLearning => Ok(Box::new(NoLearning)),
        LearningStrategyState::Ppo {
            config,
            rollout,
            rng,
            update_index,
            pending_update,
        } => Ok(Box::new(PpoLearning {
            config,
            rollout,
            rng,
            update_index,
            pending_update,
        })),
    }
}

pub fn generalized_advantages(
    transitions: &[Transition],
    gamma: f32,
    gae_lambda: f32,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let (advantages, returns) = generalized_advantages_masked(transitions, gamma, gae_lambda)?;
    if advantages.iter().any(Option::is_none) || returns.iter().any(Option::is_none) {
        return Err(MidasError::Policy(
            "masked transition requires generalized_advantages_masked".into(),
        ));
    }
    Ok((
        advantages.into_iter().map(Option::unwrap).collect(),
        returns.into_iter().map(Option::unwrap).collect(),
    ))
}

/// Computes GAE over the complete environment chronology. When every
/// timestep has a critic estimate, this is the standard bootstrapped GAE.
/// If a failure timestep has no trustworthy model value, the critic path uses
/// discounted environment returns across the masked entry instead of inventing
/// a zero value; masked entries themselves remain ineligible for critic loss.
pub type MaskedGae = (Vec<Option<f32>>, Vec<Option<f32>>);

pub fn generalized_advantages_masked(
    transitions: &[Transition],
    gamma: f32,
    gae_lambda: f32,
) -> Result<MaskedGae> {
    if !gamma.is_finite()
        || !gae_lambda.is_finite()
        || !(0.0..=1.0).contains(&gamma)
        || !(0.0..=1.0).contains(&gae_lambda)
    {
        return Err(MidasError::Config("invalid GAE parameters".into()));
    }
    let fully_valued = transitions.iter().all(|transition| {
        transition.critic_eligible
            && transition.value_estimate.is_some()
            && transition.next_value.is_some()
    });
    if !fully_valued {
        let mut returns = vec![None; transitions.len()];
        let mut discounted = 0.0_f32;
        for index in (0..transitions.len()).rev() {
            let transition = &transitions[index];
            if transition.done && !transition.truncated {
                discounted = transition.reward;
            } else {
                discounted = transition.reward + gamma * discounted;
            }
            if transition.critic_eligible {
                let Some(value) = transition.value_estimate else {
                    continue;
                };
                let advantage = discounted - value;
                if !advantage.is_finite() || !discounted.is_finite() {
                    return Err(MidasError::Policy(
                        "masked GAE produced non-finite values".into(),
                    ));
                }
                returns[index] = Some(discounted);
                // The advantage is stored in the parallel vector below.
                // Keeping this branch explicit makes masked critic samples
                // distinguishable from actor-ineligible entries.
                let _ = advantage;
            }
        }
        let advantages = transitions
            .iter()
            .enumerate()
            .map(|(index, transition)| {
                transition
                    .value_estimate
                    .zip(returns[index])
                    .map(|(value, target)| target - value)
            })
            .collect::<Vec<_>>();
        return Ok((advantages, returns));
    }
    let mut advantages = vec![0.0; transitions.len()];
    let mut next_gae = 0.0;
    for index in (0..transitions.len()).rev() {
        let transition = &transitions[index];
        let bootstrap = !transition.done || transition.truncated;
        let mask = if bootstrap { 1.0 } else { 0.0 };
        let delta = transition.reward
            + gamma
                * transition
                    .next_value
                    .expect("fully valued transition has next value")
                * mask
            - transition
                .value_estimate
                .expect("fully valued transition has value");
        next_gae = delta + gamma * gae_lambda * mask * next_gae;
        advantages[index] = next_gae;
    }
    let returns = advantages
        .iter()
        .zip(transitions)
        .map(|(advantage, transition)| {
            advantage
                + transition
                    .value_estimate
                    .expect("fully valued transition has value")
        })
        .collect::<Vec<_>>();
    if advantages
        .iter()
        .chain(&returns)
        .any(|value| !value.is_finite())
    {
        return Err(MidasError::Policy("GAE produced non-finite values".into()));
    }
    Ok((
        advantages.into_iter().map(Some).collect(),
        returns.into_iter().map(Some).collect(),
    ))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PpoLoss {
    pub policy_loss: f32,
    pub value_loss: f32,
    pub entropy_bonus: f32,
    pub total_loss: f32,
    pub ratio: f32,
}
#[allow(clippy::too_many_arguments)]
pub fn ppo_loss(
    old_log_probability: f32,
    new_log_probability: f32,
    advantage: f32,
    value: f32,
    target_return: f32,
    entropy: f32,
    clip: f32,
    critic_coefficient: f32,
    entropy_coefficient: f32,
) -> Result<PpoLoss> {
    let ratio = (new_log_probability - old_log_probability).exp();
    if [ratio, advantage, value, target_return, entropy]
        .iter()
        .any(|v| !v.is_finite())
        || !clip.is_finite()
        || clip < 0.0
        || !critic_coefficient.is_finite()
        || critic_coefficient < 0.0
        || !entropy_coefficient.is_finite()
        || entropy_coefficient < 0.0
    {
        return Err(MidasError::Policy("non-finite PPO objective input".into()));
    }
    let clipped_ratio = ratio.clamp(1.0 - clip, 1.0 + clip);
    let unclipped = ratio * advantage;
    let clipped = clipped_ratio * advantage;
    let surrogate = if advantage >= 0.0 {
        unclipped.min(clipped)
    } else {
        unclipped.max(clipped)
    };
    let policy_loss = -surrogate;
    let value_loss = (value - target_return).powi(2);
    let total_loss = policy_loss + critic_coefficient * value_loss - entropy_coefficient * entropy;
    if !total_loss.is_finite() {
        return Err(MidasError::Policy("non-finite PPO loss".into()));
    }
    Ok(PpoLoss {
        policy_loss,
        value_loss,
        entropy_bonus: entropy,
        total_loss,
        ratio,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        action::{
            ActionValidationContext, ActionValidator, LongOnlyActionValidator, ProposedAllocation,
        },
        brain::NeuralRuntimeState,
    };

    fn transition(reward: f32, value: f32, next_value: f32, done: bool) -> Transition {
        let action = LongOnlyActionValidator
            .validate(
                ProposedAllocation::rebalance(vec![1.0, 0.0]),
                &ActionValidationContext {
                    asset_count: 1,
                    tradable: &[true],
                    policy_maximum_asset_allocation: Some(1.0),
                    market_maximum_asset_allocation: None,
                },
            )
            .unwrap();
        Transition {
            observation: Some(Observation {
                schema_hash: "schema".into(),
                values: vec![0.0],
            }),
            action,
            action_sample: Some(PolicyActionSample {
                action_index: 0,
                log_probability: -std::f32::consts::LN_2,
                entropy: std::f32::consts::LN_2,
            }),
            raw_policy_output: Some(vec![0.0, 0.0]),
            log_probability: Some(-std::f32::consts::LN_2),
            temperature: Some(1.0),
            value_estimate: Some(value),
            reward,
            next_value: Some(next_value),
            done,
            truncated: false,
            recurrent_state_before: Some(NeuralRuntimeState {
                memory: Vec::new(),
                inference_steps: 0,
            }),
            recurrent_state_after: Some(NeuralRuntimeState {
                memory: Vec::new(),
                inference_steps: 1,
            }),
            date: NaiveDate::from_ymd_opt(2020, 1, 1).unwrap(),
            tradable: vec![true],
            invalid_action: false,
            failure_category: None,
            actor_eligible: true,
            critic_eligible: true,
        }
    }

    #[test]
    fn gae_matches_terminal_hand_calculation() {
        let transitions = vec![
            transition(1.0, 0.0, 0.5, false),
            transition(2.0, 0.0, 0.0, true),
        ];
        let (advantages, returns) = generalized_advantages(&transitions, 0.9, 0.8).unwrap();
        assert!((advantages[1] - 2.0).abs() < 1e-6);
        assert!((advantages[0] - (1.0 + 0.9 * 0.5 + 0.9 * 0.8 * 2.0)).abs() < 1e-6);
        assert_eq!(returns, advantages);
    }

    #[test]
    fn masked_failure_reward_remains_in_chronological_returns() {
        let mut masked = transition(0.0, 0.0, 0.0, false);
        masked.observation = None;
        masked.action_sample = None;
        masked.value_estimate = None;
        masked.next_value = None;
        masked.recurrent_state_before = None;
        masked.recurrent_state_after = None;
        masked.actor_eligible = false;
        masked.critic_eligible = false;
        masked.invalid_action = true;
        masked.reward = -2.0;
        let transitions = vec![
            transition(1.0, 0.0, 0.0, false),
            masked,
            transition(3.0, 0.0, 0.0, true),
        ];
        let (advantages, returns) = generalized_advantages_masked(&transitions, 0.9, 0.95).unwrap();
        assert!((advantages[0].unwrap() - 1.63).abs() < 1e-6);
        assert!(advantages[1].is_none());
        assert!((returns[0].unwrap() - 1.63).abs() < 1e-6);
        assert_eq!(returns[1], None);
        assert_eq!(returns[2], Some(3.0));
    }

    #[test]
    fn ppo_clipping_handles_both_advantage_signs() {
        let positive = ppo_loss(0.0, (2.0f32).ln(), 1.0, 0.0, 0.0, 0.0, 0.2, 0.5, 0.01).unwrap();
        let negative = ppo_loss(0.0, (2.0f32).ln(), -1.0, 0.0, 0.0, 0.0, 0.2, 0.5, 0.01).unwrap();
        assert!((positive.policy_loss + 1.2).abs() < 1e-6);
        assert!((negative.policy_loss - 1.2).abs() < 1e-6);
    }

    #[test]
    fn ppo_loss_rejects_non_finite_hyperparameters_and_includes_regularizers() {
        let loss = ppo_loss(0.0, 0.0, 0.5, 0.0, 1.0, 0.25, 0.2, 0.5, 0.1).unwrap();
        assert!((loss.value_loss - 1.0).abs() < 1e-6);
        assert!((loss.entropy_bonus - 0.25).abs() < 1e-6);
        assert!(ppo_loss(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, f32::NAN, 0.5, 0.1).is_err());
    }

    #[test]
    fn rollout_update_is_bounded_and_commit_clears_it() {
        let mut config = LearningConfig {
            implementation: crate::config::LearningKind::Ppo,
            learning_rate: 0.001,
            gamma: 0.99,
            gae_lambda: 0.95,
            ppo_clip: 0.2,
            critic_coefficient: 0.5,
            entropy_coefficient: 0.01,
            gradient_clip: 1.0,
            rollout_length: 2,
            minibatch_size: 1,
            sequence_length: 1,
            normalize_advantages: true,
            optimisation_epochs: 1,
        };
        let mut learner = PpoLearning::new(config.clone(), 7).unwrap();
        assert!(matches!(
            learner
                .observe_transition(transition(1.0, 0.0, 0.0, false))
                .unwrap(),
            LearningUpdate::None
        ));
        let update = learner
            .observe_transition(transition(1.0, 0.0, 0.0, true))
            .unwrap();
        assert!(matches!(update, LearningUpdate::Ppo(_)));
        learner.commit_update().unwrap();
        assert!(learner.rollout().is_empty());
        config.rollout_length = 0;
        assert!(PpoLearning::new(config, 1).is_err());
    }
}
