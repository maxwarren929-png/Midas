use crate::{
    MidasError, PolicyFailureCategory, Result,
    action::ValidatedAction,
    brain::{NeuralRuntimeState, PolicyActionSample},
    config::{LearningConfig, OptimizerConfig},
    observation::Observation,
    reward::RewardBreakdown,
};
use chrono::NaiveDate;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Transition {
    pub observation: Option<Observation>,
    pub action: ValidatedAction,
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
    /// True when this sample starts an independently unrolled recurrent
    /// sequence. Gradients never cross this boundary.
    pub sequence_start: bool,
    pub observation: Observation,
    pub action_sample: Option<PolicyActionSample>,
    pub old_log_probability: Option<f32>,
    pub advantage: Option<f32>,
    pub return_value: Option<f32>,
    pub value_estimate: Option<f32>,
    pub recurrent_state: NeuralRuntimeState,
    pub tradable: Vec<bool>,
    pub temperature: Option<f32>,
    pub invalid_action: bool,
    /// Report-only centered component contributions in the same normalized
    /// units as `advantage`. Training ignores this field.
    pub counterfactual_advantages: Option<CounterfactualAdvantages>,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct CounterfactualAdvantages {
    pub raw_return: f32,
    pub drawdown_cost: f32,
    pub turnover_cost: f32,
    pub other_cost: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PpoTrainingBatch {
    pub samples: Vec<PpoTrainingSample>,
    pub sequence_length: usize,
    /// Report-only attribution computed before samples are shuffled.
    pub advantage_diagnostics: Option<Box<PpoAdvantageDiagnostics>>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct AdvantageComponentDiagnostics {
    pub mean: f64,
    pub standard_deviation: f64,
    pub covariance_with_pre_normalization_advantage: f64,
    pub correlation_with_pre_normalization_advantage: f64,
    pub projection_on_pre_normalization_advantage: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct PpoAdvantageDiagnostics {
    /// False only for the first update after restoring a non-empty rollout,
    /// because report-only reward decomposition is excluded from checkpoints.
    pub complete: bool,
    pub actor_sample_count: usize,
    pub pre_normalization_mean: f64,
    pub pre_normalization_standard_deviation: f64,
    pub post_normalization_mean: f64,
    pub post_normalization_standard_deviation: f64,
    pub normalization_scale: f64,
    pub sign_flip_fraction: f64,
    pub rank_inversions: u64,
    pub raw_return: AdvantageComponentDiagnostics,
    pub drawdown_cost: AdvantageComponentDiagnostics,
    pub turnover_cost: AdvantageComponentDiagnostics,
    pub other_cost: AdvantageComponentDiagnostics,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ActorGradientComponentDiagnostics {
    pub norm: f64,
    pub cosine_with_actual_policy_gradient: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ActorGradientAttribution {
    pub actual_policy_gradient_norm: f64,
    pub raw_return: ActorGradientComponentDiagnostics,
    pub drawdown_cost: ActorGradientComponentDiagnostics,
    pub turnover_cost: ActorGradientComponentDiagnostics,
    pub other_cost: ActorGradientComponentDiagnostics,
    pub raw_return_drawdown_cosine: f64,
    pub raw_return_turnover_cosine: f64,
    pub drawdown_turnover_cosine: f64,
}

impl ActorGradientAttribution {
    pub fn accumulate(&mut self, other: &Self) {
        self.actual_policy_gradient_norm += other.actual_policy_gradient_norm;
        self.raw_return.norm += other.raw_return.norm;
        self.raw_return.cosine_with_actual_policy_gradient +=
            other.raw_return.cosine_with_actual_policy_gradient;
        self.drawdown_cost.norm += other.drawdown_cost.norm;
        self.drawdown_cost.cosine_with_actual_policy_gradient +=
            other.drawdown_cost.cosine_with_actual_policy_gradient;
        self.turnover_cost.norm += other.turnover_cost.norm;
        self.turnover_cost.cosine_with_actual_policy_gradient +=
            other.turnover_cost.cosine_with_actual_policy_gradient;
        self.other_cost.norm += other.other_cost.norm;
        self.other_cost.cosine_with_actual_policy_gradient +=
            other.other_cost.cosine_with_actual_policy_gradient;
        self.raw_return_drawdown_cosine += other.raw_return_drawdown_cosine;
        self.raw_return_turnover_cosine += other.raw_return_turnover_cosine;
        self.drawdown_turnover_cosine += other.drawdown_turnover_cosine;
    }

    pub fn divide(&mut self, divisor: f64) {
        self.actual_policy_gradient_norm /= divisor;
        self.raw_return.norm /= divisor;
        self.raw_return.cosine_with_actual_policy_gradient /= divisor;
        self.drawdown_cost.norm /= divisor;
        self.drawdown_cost.cosine_with_actual_policy_gradient /= divisor;
        self.turnover_cost.norm /= divisor;
        self.turnover_cost.cosine_with_actual_policy_gradient /= divisor;
        self.other_cost.norm /= divisor;
        self.other_cost.cosine_with_actual_policy_gradient /= divisor;
        self.raw_return_drawdown_cosine /= divisor;
        self.raw_return_turnover_cosine /= divisor;
        self.drawdown_turnover_cosine /= divisor;
    }
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
    pub pre_clip_gradient_norm: f32,
    pub hidden_gradient_norms: Vec<f32>,
    pub recurrent_gradient_norm: f32,
    pub gru_update_gradient_norm: Option<f32>,
    pub gru_reset_gradient_norm: Option<f32>,
    pub gru_candidate_gradient_norm: Option<f32>,
    pub actor_gradient_norm: f32,
    pub critic_gradient_norm: f32,
    pub gradient_clip_fraction: f32,
    pub relative_update_norm: f32,
    pub approximate_kl: f32,
    pub clip_fraction: f32,
    pub update_index: u64,
    pub actor_gradient_attribution: Option<ActorGradientAttribution>,
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
            pre_clip_gradient_norm: 0.0,
            hidden_gradient_norms: Vec::new(),
            recurrent_gradient_norm: 0.0,
            gru_update_gradient_norm: None,
            gru_reset_gradient_norm: None,
            gru_candidate_gradient_norm: None,
            actor_gradient_norm: 0.0,
            critic_gradient_norm: 0.0,
            gradient_clip_fraction: 0.0,
            relative_update_norm: 0.0,
            approximate_kl: 0.0,
            clip_fraction: 0.0,
            update_index: 0,
            actor_gradient_attribution: None,
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
    fn observe_transition_with_diagnostics(
        &mut self,
        transition: Transition,
        _reward_breakdown: RewardBreakdown,
    ) -> Result<LearningUpdate> {
        self.observe_transition(transition)
    }
    fn end_episode(&mut self, context: &EpisodeEndContext) -> Result<LearningUpdate>;
    fn commit_update(&mut self) -> Result<()>;
    fn snapshot(&self) -> Result<LearningStrategyState>;
    fn fork(&self) -> Result<Box<dyn LearningStrategy>>;
    fn name(&self) -> &'static str;
    fn training_config(&self) -> Option<LearningConfig> {
        None
    }
    fn training_optimizer(&self) -> Option<OptimizerConfig> {
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
        validate_transition(&transition)?;
        self.transitions.push(transition);
        Ok(())
    }
    pub fn clear(&mut self) {
        self.transitions.clear();
    }
}

fn validate_transition(transition: &Transition) -> Result<()> {
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
    Ok(())
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
    optimizer: OptimizerConfig,
    rollout: RolloutBuffer,
    rng: ChaCha8Rng,
    update_index: u64,
    pending_update: bool,
    diagnostic_rewards: Vec<RewardBreakdown>,
    diagnostic_rewards_complete: bool,
}
impl PpoLearning {
    pub fn new(config: LearningConfig, seed: u64) -> Result<Self> {
        Self::new_with_optimizer(config, OptimizerConfig::default(), seed)
    }
    pub fn new_with_optimizer(
        config: LearningConfig,
        optimizer: OptimizerConfig,
        seed: u64,
    ) -> Result<Self> {
        optimizer.validate()?;
        if config.implementation != crate::config::LearningKind::Ppo {
            return Err(MidasError::Config(
                "PpoLearning requires learning.implementation = ppo".into(),
            ));
        }
        config.validate()?;
        let rollout_length = config.rollout_length;
        Ok(Self {
            rollout: RolloutBuffer::new(rollout_length)?,
            config,
            optimizer,
            rng: ChaCha8Rng::seed_from_u64(seed),
            update_index: 0,
            pending_update: false,
            diagnostic_rewards: Vec::with_capacity(rollout_length),
            diagnostic_rewards_complete: true,
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
        let pre_normalization_advantages = advantages.clone();
        let mut samples = transitions
            .iter()
            .enumerate()
            .zip(advantages)
            .zip(returns)
            .filter_map(|(((source_index, transition), advantage), return_value)| {
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
                                (sample.clone(), old_log_probability, temperature, advantage)
                            })
                    });
                let critic_sample = transition
                    .critic_eligible
                    .then_some(())
                    .and(transition.value_estimate.zip(return_value));
                if actor_sample.is_none() && critic_sample.is_none() {
                    return None;
                }
                Some((
                    source_index,
                    PpoTrainingSample {
                        sequence_start: false,
                        observation,
                        action_sample: actor_sample.as_ref().map(|sample| sample.0.clone()),
                        old_log_probability: actor_sample.as_ref().map(|sample| sample.1),
                        advantage: actor_sample.as_ref().map(|sample| sample.3),
                        return_value: critic_sample.map(|sample| sample.1),
                        value_estimate: critic_sample.map(|sample| sample.0),
                        recurrent_state,
                        tradable: transition.tradable.clone(),
                        temperature: actor_sample.as_ref().map(|sample| sample.2),
                        invalid_action: transition.invalid_action,
                        counterfactual_advantages: None,
                    },
                ))
            })
            .collect::<Vec<_>>();
        if samples.is_empty() {
            self.pending_update = true;
            return Ok(LearningUpdate::None);
        }
        let mut advantage_normalization_scale = 1.0_f32;
        if self.config.normalize_advantages {
            let actor_count = samples
                .iter()
                .filter(|(_, sample)| sample.advantage.is_some())
                .count();
            let mean = samples
                .iter()
                .filter_map(|(_, sample)| sample.advantage)
                .sum::<f32>()
                / actor_count.max(1) as f32;
            let variance = samples
                .iter()
                .filter_map(|(_, sample)| sample.advantage)
                .map(|advantage| (advantage - mean).powi(2))
                .sum::<f32>()
                / actor_count.max(1) as f32;
            let standard_deviation = variance.sqrt().max(1e-8);
            advantage_normalization_scale = standard_deviation.recip();
            for (_, sample) in &mut samples {
                if let Some(advantage) = sample.advantage {
                    sample.advantage = Some((advantage - mean) / standard_deviation);
                }
            }
        }
        let (advantage_diagnostics, counterfactual_advantages) = build_advantage_diagnostics(
            transitions,
            &self.diagnostic_rewards,
            self.diagnostic_rewards_complete,
            &pre_normalization_advantages,
            &samples,
            self.config.gamma,
            self.config.gae_lambda,
            self.config.normalize_advantages,
            advantage_normalization_scale,
        )?;
        for (source_index, sample) in &mut samples {
            sample.counterfactual_advantages = counterfactual_advantages[*source_index];
        }
        // Build explicit recurrent sequences before shuffling. A masked source
        // transition creates an index gap and therefore a hard BPTT boundary.
        let maximum_sequence = self.config.sequence_length.max(1);
        let mut sequences = Vec::<Vec<PpoTrainingSample>>::new();
        let mut previous_source = None;
        for (source_index, mut sample) in samples {
            let starts_sequence = previous_source.is_none_or(|previous| {
                source_index != previous + 1
                    || sequences
                        .last()
                        .is_none_or(|sequence| sequence.len() >= maximum_sequence)
            });
            sample.sequence_start = starts_sequence;
            if starts_sequence {
                sequences.push(vec![sample]);
            } else if let Some(sequence) = sequences.last_mut() {
                sequence.push(sample);
            }
            previous_source = Some(source_index);
        }
        let mut order = (0..sequences.len()).collect::<Vec<_>>();
        for index in (1..order.len()).rev() {
            let swap = self.rng.gen_range(0..=index);
            order.swap(index, swap);
        }
        let shuffled = order
            .into_iter()
            .flat_map(|index| sequences[index].clone())
            .collect();
        self.pending_update = true;
        Ok(LearningUpdate::Ppo(PpoTrainingBatch {
            samples: shuffled,
            sequence_length: self.config.sequence_length,
            advantage_diagnostics: Some(Box::new(advantage_diagnostics)),
        }))
    }

    fn observe_transition_internal(
        &mut self,
        transition: Transition,
        reward_breakdown: RewardBreakdown,
        diagnostics_complete: bool,
    ) -> Result<LearningUpdate> {
        if self.pending_update {
            return Err(MidasError::Policy(
                "previous PPO update must be committed before observing more transitions".into(),
            ));
        }
        let previous_complete = self.diagnostic_rewards_complete;
        self.diagnostic_rewards.push(reward_breakdown);
        self.diagnostic_rewards_complete &= diagnostics_complete;
        if let Err(error) = self.rollout.push(transition) {
            self.diagnostic_rewards.pop();
            self.diagnostic_rewards_complete = previous_complete;
            return Err(error);
        }
        if self.rollout.len() >= self.config.rollout_length {
            self.make_update()
        } else {
            Ok(LearningUpdate::None)
        }
    }
}
impl LearningStrategy for PpoLearning {
    fn observe_transition(&mut self, transition: Transition) -> Result<LearningUpdate> {
        self.observe_transition_internal(transition, RewardBreakdown::default(), false)
    }
    fn observe_transition_with_diagnostics(
        &mut self,
        transition: Transition,
        reward_breakdown: RewardBreakdown,
    ) -> Result<LearningUpdate> {
        self.observe_transition_internal(transition, reward_breakdown, true)
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
            self.diagnostic_rewards.clear();
            self.diagnostic_rewards_complete = true;
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
            optimizer: self.optimizer.clone(),
            rollout: self.rollout.clone(),
            rng: self.rng.clone(),
            update_index: self.update_index,
            pending_update: self.pending_update,
            diagnostic_rewards: self.diagnostic_rewards.clone(),
            diagnostic_rewards_complete: self.diagnostic_rewards_complete,
        }))
    }
    fn name(&self) -> &'static str {
        "ppo"
    }
    fn training_config(&self) -> Option<LearningConfig> {
        Some(self.config.clone())
    }
    fn training_optimizer(&self) -> Option<OptimizerConfig> {
        Some(self.optimizer.clone())
    }
}

pub fn learning_from_state(
    state: LearningStrategyState,
    optimizer: &crate::config::OptimizerConfig,
) -> Result<Box<dyn LearningStrategy>> {
    match state {
        LearningStrategyState::NoLearning => Ok(Box::new(NoLearning)),
        LearningStrategyState::Ppo {
            config,
            rollout,
            rng,
            update_index,
            pending_update,
        } => {
            optimizer.validate()?;
            if config.implementation != crate::config::LearningKind::Ppo {
                return Err(MidasError::Checkpoint(
                    "restored PPO state has a non-PPO implementation".into(),
                ));
            }
            config.validate()?;
            if rollout.capacity != config.rollout_length
                || rollout.transitions.len() > rollout.capacity
            {
                return Err(MidasError::Checkpoint(
                    "restored PPO rollout capacity does not match its configuration".into(),
                ));
            }
            for transition in &rollout.transitions {
                validate_transition(transition)?;
            }
            if pending_update && rollout.transitions.is_empty() {
                return Err(MidasError::Checkpoint(
                    "restored PPO update is pending with an empty rollout".into(),
                ));
            }
            Ok(Box::new(PpoLearning {
                config,
                optimizer: optimizer.clone(),
                diagnostic_rewards: vec![RewardBreakdown::default(); rollout.len()],
                diagnostic_rewards_complete: rollout.is_empty(),
                rollout,
                rng,
                update_index,
                pending_update,
            }))
        }
    }
}

fn reward_signal_returns(
    transitions: &[Transition],
    rewards: &[f64],
    gamma: f32,
    gae_lambda: f32,
) -> Result<Vec<f64>> {
    if transitions.len() != rewards.len() || rewards.iter().any(|value| !value.is_finite()) {
        return Err(MidasError::Policy(
            "advantage diagnostics have invalid reward components".into(),
        ));
    }
    let fully_valued = transitions.iter().all(|transition| {
        transition.critic_eligible
            && transition.value_estimate.is_some()
            && transition.next_value.is_some()
    });
    let discount = f64::from(if fully_valued {
        gamma * gae_lambda
    } else {
        gamma
    });
    let mut signals = vec![0.0; transitions.len()];
    let mut next = 0.0;
    for index in (0..transitions.len()).rev() {
        let transition = &transitions[index];
        if transition.done && !transition.truncated {
            next = rewards[index];
        } else {
            next = rewards[index] + discount * next;
        }
        signals[index] = next;
    }
    Ok(signals)
}

fn population_moments(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    (mean, variance.max(0.0).sqrt())
}

fn component_diagnostics(
    component: &[f64],
    pre_normalization_advantages: &[f64],
) -> AdvantageComponentDiagnostics {
    let (mean, standard_deviation) = population_moments(component);
    let (advantage_mean, advantage_standard_deviation) =
        population_moments(pre_normalization_advantages);
    let covariance = if component.is_empty() {
        0.0
    } else {
        component
            .iter()
            .zip(pre_normalization_advantages)
            .map(|(value, advantage)| (value - mean) * (advantage - advantage_mean))
            .sum::<f64>()
            / component.len() as f64
    };
    let correlation = if standard_deviation > 0.0 && advantage_standard_deviation > 0.0 {
        covariance / (standard_deviation * advantage_standard_deviation)
    } else {
        0.0
    };
    let advantage_variance = advantage_standard_deviation.powi(2);
    AdvantageComponentDiagnostics {
        mean,
        standard_deviation,
        covariance_with_pre_normalization_advantage: covariance,
        correlation_with_pre_normalization_advantage: correlation,
        projection_on_pre_normalization_advantage: if advantage_variance > 0.0 {
            covariance / advantage_variance
        } else {
            0.0
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn build_advantage_diagnostics(
    transitions: &[Transition],
    breakdowns: &[RewardBreakdown],
    complete: bool,
    pre_normalization_advantages: &[Option<f32>],
    samples: &[(usize, PpoTrainingSample)],
    gamma: f32,
    gae_lambda: f32,
    center_components: bool,
    normalization_scale: f32,
) -> Result<(
    PpoAdvantageDiagnostics,
    Vec<Option<CounterfactualAdvantages>>,
)> {
    if transitions.len() != breakdowns.len()
        || transitions.len() != pre_normalization_advantages.len()
    {
        return Err(MidasError::Policy(
            "advantage diagnostics are not aligned with the rollout".into(),
        ));
    }
    let raw_rewards = breakdowns
        .iter()
        .map(|breakdown| breakdown.raw_log_return)
        .collect::<Vec<_>>();
    let drawdown_rewards = breakdowns
        .iter()
        .map(|breakdown| -breakdown.drawdown_penalty)
        .collect::<Vec<_>>();
    let turnover_rewards = breakdowns
        .iter()
        .map(|breakdown| -breakdown.turnover_penalty)
        .collect::<Vec<_>>();
    let other_rewards = breakdowns
        .iter()
        .map(|breakdown| {
            f64::from(breakdown.clipped_total) - breakdown.raw_log_return
                + breakdown.drawdown_penalty
                + breakdown.turnover_penalty
        })
        .collect::<Vec<_>>();
    let raw_signals = reward_signal_returns(transitions, &raw_rewards, gamma, gae_lambda)?;
    let drawdown_signals =
        reward_signal_returns(transitions, &drawdown_rewards, gamma, gae_lambda)?;
    let turnover_signals =
        reward_signal_returns(transitions, &turnover_rewards, gamma, gae_lambda)?;
    let other_signals = reward_signal_returns(transitions, &other_rewards, gamma, gae_lambda)?;

    let mut indices = Vec::new();
    let mut pre = Vec::new();
    let mut post = Vec::new();
    for (source_index, sample) in samples {
        let Some(post_advantage) = sample.advantage else {
            continue;
        };
        let Some(pre_advantage) = pre_normalization_advantages[*source_index] else {
            continue;
        };
        indices.push(*source_index);
        pre.push(f64::from(pre_advantage));
        post.push(f64::from(post_advantage));
    }
    let select = |signals: &[f64]| {
        indices
            .iter()
            .map(|index| signals[*index])
            .collect::<Vec<_>>()
    };
    let (pre_mean, pre_standard_deviation) = population_moments(&pre);
    let (post_mean, post_standard_deviation) = population_moments(&post);
    let sign_flips = pre
        .iter()
        .zip(&post)
        .filter(|(before, after)| (**before >= 0.0) != (**after >= 0.0))
        .count();
    let mut rank_inversions = 0_u64;
    for left in 0..pre.len() {
        for right in left + 1..pre.len() {
            if (pre[left] - pre[right]) * (post[left] - post[right]) < 0.0 {
                rank_inversions = rank_inversions.saturating_add(1);
            }
        }
    }
    let raw = select(&raw_signals);
    let drawdown = select(&drawdown_signals);
    let turnover = select(&turnover_signals);
    let other = select(&other_signals);
    let (raw_mean, _) = population_moments(&raw);
    let (drawdown_mean, _) = population_moments(&drawdown);
    let (turnover_mean, _) = population_moments(&turnover);
    let (other_mean, _) = population_moments(&other);
    let mut counterfactual = vec![None; transitions.len()];
    if complete {
        let scale_component = |value: f64, mean: f64| {
            let centered = if center_components {
                value - mean
            } else {
                value
            };
            (centered * f64::from(normalization_scale)) as f32
        };
        for (position, source_index) in indices.iter().enumerate() {
            let values = CounterfactualAdvantages {
                raw_return: scale_component(raw[position], raw_mean),
                drawdown_cost: scale_component(drawdown[position], drawdown_mean),
                turnover_cost: scale_component(turnover[position], turnover_mean),
                other_cost: scale_component(other[position], other_mean),
            };
            if [
                values.raw_return,
                values.drawdown_cost,
                values.turnover_cost,
                values.other_cost,
            ]
            .iter()
            .any(|value| !value.is_finite())
            {
                return Err(MidasError::Policy(
                    "counterfactual advantage is non-finite".into(),
                ));
            }
            counterfactual[*source_index] = Some(values);
        }
    }
    let diagnostics = PpoAdvantageDiagnostics {
        complete,
        actor_sample_count: pre.len(),
        pre_normalization_mean: pre_mean,
        pre_normalization_standard_deviation: pre_standard_deviation,
        post_normalization_mean: post_mean,
        post_normalization_standard_deviation: post_standard_deviation,
        normalization_scale: if pre_standard_deviation > 0.0 {
            post_standard_deviation / pre_standard_deviation
        } else {
            0.0
        },
        sign_flip_fraction: if pre.is_empty() {
            0.0
        } else {
            sign_flips as f64 / pre.len() as f64
        },
        rank_inversions,
        raw_return: component_diagnostics(&raw, &pre),
        drawdown_cost: component_diagnostics(&drawdown, &pre),
        turnover_cost: component_diagnostics(&turnover, &pre),
        other_cost: component_diagnostics(&other, &pre),
    };
    Ok((diagnostics, counterfactual))
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
        let mut discounted = transitions
            .last()
            .filter(|transition| !transition.done || transition.truncated)
            .and_then(|transition| transition.next_value)
            .unwrap_or(0.0);
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
                    policy_maximum_trade_fraction: None,
                    market_maximum_asset_allocation: None,
                },
            )
            .unwrap();
        Transition {
            observation: Some(Observation {
                schema_hash: "schema".into(),
                values: vec![0.0],
            }),
            action: crate::action::ValidatedAction::Allocation(action),
            action_sample: Some(PolicyActionSample {
                active_asset_indices: vec![0],
                latent_samples: vec![0.5],
                trade_impulses: vec![0.0],
                maximum_trade_fraction: 0.05,
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
    fn masked_continuing_rollout_preserves_boundary_bootstrap() {
        let mut masked = transition(-2.0, 0.0, 0.0, false);
        masked.observation = None;
        masked.action_sample = None;
        masked.value_estimate = None;
        masked.next_value = None;
        masked.recurrent_state_before = None;
        masked.recurrent_state_after = None;
        masked.actor_eligible = false;
        masked.critic_eligible = false;
        masked.invalid_action = true;
        let transitions = vec![
            transition(1.0, 0.0, 0.0, false),
            masked,
            transition(3.0, 0.0, 5.0, false),
        ];

        let (advantages, returns) = generalized_advantages_masked(&transitions, 0.9, 0.95).unwrap();

        assert!((advantages[0].unwrap() - 5.275).abs() < 1e-6);
        assert!(advantages[1].is_none());
        assert!((advantages[2].unwrap() - 7.5).abs() < 1e-6);
        assert_eq!(returns, advantages);
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

    #[test]
    fn reward_component_diagnostics_do_not_change_ppo_batch_or_state() {
        let config = LearningConfig {
            implementation: crate::config::LearningKind::Ppo,
            learning_rate: 0.001,
            gamma: 1.0,
            gae_lambda: 1.0,
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
        let mut control = PpoLearning::new(config.clone(), 23).unwrap();
        let mut instrumented = PpoLearning::new(config, 23).unwrap();
        let breakdowns = [
            RewardBreakdown {
                raw_log_return: 0.01,
                drawdown_penalty: 0.001,
                turnover_penalty: 0.002,
                unclipped_total: 0.007,
                clipped_total: 0.007,
                ..RewardBreakdown::default()
            },
            RewardBreakdown {
                raw_log_return: 0.02,
                turnover_penalty: 0.003,
                unclipped_total: 0.017,
                clipped_total: 0.017,
                ..RewardBreakdown::default()
            },
        ];

        let first = transition(0.007, 0.0, 0.0, false);
        let second = transition(0.017, 0.0, 0.0, true);
        assert!(matches!(
            control.observe_transition(first.clone()).unwrap(),
            LearningUpdate::None
        ));
        assert!(matches!(
            instrumented
                .observe_transition_with_diagnostics(first, breakdowns[0])
                .unwrap(),
            LearningUpdate::None
        ));
        let LearningUpdate::Ppo(control_batch) =
            control.observe_transition(second.clone()).unwrap()
        else {
            panic!("control did not produce a PPO batch")
        };
        let LearningUpdate::Ppo(instrumented_batch) = instrumented
            .observe_transition_with_diagnostics(second, breakdowns[1])
            .unwrap()
        else {
            panic!("instrumented learner did not produce a PPO batch")
        };
        let mut control_samples = control_batch.samples.clone();
        let mut instrumented_samples = instrumented_batch.samples.clone();
        for sample in &mut control_samples {
            sample.counterfactual_advantages = None;
        }
        for sample in &mut instrumented_samples {
            sample.counterfactual_advantages = None;
        }
        assert_eq!(control_samples, instrumented_samples);
        assert_eq!(
            control_batch.sequence_length,
            instrumented_batch.sequence_length
        );
        assert_eq!(
            control.snapshot().unwrap(),
            instrumented.snapshot().unwrap()
        );

        let diagnostics = instrumented_batch.advantage_diagnostics.unwrap();
        assert!(diagnostics.complete);
        assert_eq!(diagnostics.actor_sample_count, 2);
        assert!((diagnostics.pre_normalization_mean - 0.0205).abs() < 1e-6);
        assert!((diagnostics.pre_normalization_standard_deviation - 0.0035).abs() < 1e-6);
        assert!(diagnostics.post_normalization_mean.abs() < 1e-6);
        assert!((diagnostics.post_normalization_standard_deviation - 1.0).abs() < 1e-6);
        assert!((diagnostics.sign_flip_fraction - 0.5).abs() < 1e-12);
        assert_eq!(diagnostics.rank_inversions, 0);
        assert!((diagnostics.raw_return.standard_deviation - 0.005).abs() < 1e-12);
        assert!((diagnostics.drawdown_cost.standard_deviation - 0.0005).abs() < 1e-12);
        assert!((diagnostics.turnover_cost.standard_deviation - 0.001).abs() < 1e-12);
        for sample in &instrumented_batch.samples {
            let counterfactual = sample.counterfactual_advantages.unwrap();
            let component_sum = counterfactual.raw_return
                + counterfactual.drawdown_cost
                + counterfactual.turnover_cost
                + counterfactual.other_cost;
            assert!((component_sum - sample.advantage.unwrap()).abs() < 1e-5);
        }

        control.commit_update().unwrap();
        instrumented.commit_update().unwrap();
        assert_eq!(
            control.snapshot().unwrap(),
            instrumented.snapshot().unwrap()
        );
    }

    #[test]
    fn restored_ppo_state_is_validated_before_use() {
        let config = LearningConfig {
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
        let learner = PpoLearning::new(config, 7).unwrap();
        let mut invalid_config = learner.snapshot().unwrap();
        let LearningStrategyState::Ppo { config, .. } = &mut invalid_config else {
            unreachable!()
        };
        config.learning_rate = f32::NAN;
        assert!(learning_from_state(invalid_config, &Default::default()).is_err());

        let mut invalid_rollout = learner.snapshot().unwrap();
        let LearningStrategyState::Ppo { rollout, .. } = &mut invalid_rollout else {
            unreachable!()
        };
        rollout.capacity += 1;
        let error = match learning_from_state(invalid_rollout, &Default::default()) {
            Ok(_) => panic!("mismatched PPO rollout capacity was accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("rollout capacity"));
    }

    #[test]
    fn restored_partial_rollout_marks_first_signal_diagnostic_incomplete() {
        let config = LearningConfig {
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
        let mut learner = PpoLearning::new(config, 31).unwrap();
        learner
            .observe_transition_with_diagnostics(
                transition(0.01, 0.0, 0.0, false),
                RewardBreakdown {
                    raw_log_return: 0.01,
                    unclipped_total: 0.01,
                    clipped_total: 0.01,
                    ..RewardBreakdown::default()
                },
            )
            .unwrap();
        let mut restored = learning_from_state(learner.snapshot().unwrap(), &Default::default())
            .expect("restore partial rollout");
        let LearningUpdate::Ppo(batch) = restored
            .observe_transition_with_diagnostics(
                transition(0.02, 0.0, 0.0, true),
                RewardBreakdown {
                    raw_log_return: 0.02,
                    unclipped_total: 0.02,
                    clipped_total: 0.02,
                    ..RewardBreakdown::default()
                },
            )
            .unwrap()
        else {
            panic!("restored rollout did not update")
        };
        assert!(!batch.advantage_diagnostics.unwrap().complete);
    }

    #[test]
    fn masked_timestep_creates_an_explicit_recurrent_sequence_boundary() {
        let config = LearningConfig {
            implementation: crate::config::LearningKind::Ppo,
            learning_rate: 0.001,
            gamma: 0.99,
            gae_lambda: 0.95,
            ppo_clip: 0.2,
            critic_coefficient: 0.5,
            entropy_coefficient: 0.01,
            gradient_clip: 1.0,
            rollout_length: 4,
            minibatch_size: 2,
            sequence_length: 2,
            normalize_advantages: false,
            optimisation_epochs: 1,
        };
        let mut learner = PpoLearning::new(config, 19).unwrap();
        assert!(matches!(
            learner
                .observe_transition(transition(1.0, 0.0, 0.0, false))
                .unwrap(),
            LearningUpdate::None
        ));
        let mut masked = transition(-1.0, 0.0, 0.0, false);
        masked.observation = None;
        masked.action_sample = None;
        masked.value_estimate = None;
        masked.next_value = None;
        masked.recurrent_state_before = None;
        masked.recurrent_state_after = None;
        masked.actor_eligible = false;
        masked.critic_eligible = false;
        assert!(matches!(
            learner.observe_transition(masked).unwrap(),
            LearningUpdate::None
        ));
        assert!(matches!(
            learner
                .observe_transition(transition(2.0, 0.0, 0.0, false))
                .unwrap(),
            LearningUpdate::None
        ));
        let LearningUpdate::Ppo(batch) = learner
            .observe_transition(transition(3.0, 0.0, 0.0, true))
            .unwrap()
        else {
            panic!("expected PPO update")
        };
        assert_eq!(batch.samples.len(), 3);
        assert_eq!(
            batch
                .samples
                .iter()
                .filter(|sample| sample.sequence_start)
                .count(),
            2
        );
        assert!(batch.samples[0].sequence_start);
        let mut current_length = 0usize;
        let mut maximum_length = 0usize;
        for sample in &batch.samples {
            if sample.sequence_start {
                current_length = 0;
            }
            current_length += 1;
            maximum_length = maximum_length.max(current_length);
        }
        assert!(maximum_length <= 2);
    }
}
