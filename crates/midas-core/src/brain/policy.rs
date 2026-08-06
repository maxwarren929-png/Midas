use super::impulse_beta::bounded_impulse_beta_evaluation_with_order;
use super::{
    BrainGenome, LifetimeParameters, NeuralInput, NeuralModel, NeuralModelState,
    NeuralRuntimeState, PolicyEvaluation,
};
use crate::{
    MidasError, Result, action::ProposedAction, config::LearningConfig, observation::Observation,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, Serialize, Deserialize, PartialEq)]
pub enum PolicyBlueprint {
    Cash,
    Random,
    EqualWeight,
    BuyAndHold {
        target_asset: usize,
    },
    Neural,
    Momentum {
        signal_indices: Vec<usize>,
        minimum_signal_bits: u32,
    },
}
pub struct PolicyContext<'a> {
    pub tradable: &'a [bool],
    pub current_allocations: &'a [f64],
}
pub trait Policy: Send {
    fn propose_action(
        &mut self,
        _: &Observation,
        context: &PolicyContext<'_>,
    ) -> Result<ProposedAction>;
    fn snapshot(&self) -> Result<PolicySnapshot>;
    fn reset_episode_state(&mut self);
    fn policy_blueprint(&self) -> PolicyBlueprint;
    /// Returns the current and birth action temperatures used to describe
    /// exploration to an optional agent-awareness provider.
    fn temperature_state(&self) -> (f32, f32) {
        (1.0, 1.0)
    }
    fn evaluate_action(
        &mut self,
        observation: &Observation,
        context: &PolicyContext<'_>,
        _stochastic: bool,
        _maximum_asset_allocation: f64,
    ) -> Result<PolicyEvaluation> {
        let proposed = self.propose_action(observation, context)?;
        Ok(PolicyEvaluation {
            proposed_action: proposed,
            action_sample: None,
            log_probability: None,
            entropy: 0.0,
            value_estimate: 0.0,
            raw_logits: Vec::new(),
            temperature: 1.0,
        })
    }
    fn apply_learning(
        &mut self,
        _batch: &crate::learning::PpoTrainingBatch,
        _config: &LearningConfig,
    ) -> Result<crate::learning::TrainingMetrics> {
        Err(MidasError::Policy(
            "policy does not expose trainable lifetime parameters".into(),
        ))
    }
    fn runtime_state(&self) -> Option<NeuralRuntimeState> {
        None
    }
    fn lifetime_parameters(&self) -> Result<Option<LifetimeParameters>> {
        Ok(None)
    }
    /// Fork policy state for a transactional simulation step.
    fn fork(&self) -> Result<Box<dyn Policy>> {
        Err(MidasError::Policy(
            "policy does not support transactional forking".into(),
        ))
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[allow(clippy::large_enum_variant)]
pub enum PolicySnapshot {
    Cash,
    Random {
        rng: ChaCha8Rng,
    },
    EqualWeight,
    BuyAndHold {
        initialized: bool,
        target_asset: usize,
    },
    Neural {
        genome: BrainGenome,
        state: Box<NeuralModelState>,
        rng: ChaCha8Rng,
        temperature: f32,
        maximum_trade_fraction: f64,
    },
    Momentum {
        signal_indices: Vec<usize>,
        minimum_signal: f32,
    },
}

pub struct NeuralPolicy {
    genome: BrainGenome,
    model: Box<dyn NeuralModel>,
    rng: ChaCha8Rng,
    temperature: f32,
    maximum_trade_fraction: f64,
}
impl NeuralPolicy {
    pub fn new(genome: BrainGenome, model: Box<dyn NeuralModel>) -> Self {
        Self::new_with_seed_and_trade_fraction(genome, model, 0, 0.05)
    }
    pub fn new_with_seed(genome: BrainGenome, model: Box<dyn NeuralModel>, seed: u64) -> Self {
        Self::new_with_seed_and_trade_fraction(genome, model, seed, 0.05)
    }
    pub fn new_with_seed_and_trade_fraction(
        genome: BrainGenome,
        model: Box<dyn NeuralModel>,
        seed: u64,
        maximum_trade_fraction: f64,
    ) -> Self {
        let temperature = genome.traits.action_temperature;
        Self {
            genome,
            model,
            rng: ChaCha8Rng::seed_from_u64(seed),
            temperature,
            maximum_trade_fraction,
        }
    }
    pub(crate) fn with_rng(
        genome: BrainGenome,
        model: Box<dyn NeuralModel>,
        rng: ChaCha8Rng,
        temperature: f32,
        maximum_trade_fraction: f64,
    ) -> Self {
        Self {
            genome,
            model,
            rng,
            temperature,
            maximum_trade_fraction,
        }
    }
}
impl Policy for NeuralPolicy {
    fn propose_action(&mut self, o: &Observation, c: &PolicyContext<'_>) -> Result<ProposedAction> {
        Ok(self
            .evaluate_action(o, c, false, self.maximum_trade_fraction)?
            .proposed_action)
    }
    fn evaluate_action(
        &mut self,
        o: &Observation,
        c: &PolicyContext<'_>,
        stochastic: bool,
        maximum_trade_fraction: f64,
    ) -> Result<PolicyEvaluation> {
        let order_offset = self
            .model
            .runtime_state()
            .map(|state| state.inference_steps as usize)
            .unwrap_or(0);
        let out = self.model.forward(&NeuralInput { observation: o })?;
        let evaluation = bounded_impulse_beta_evaluation_with_order(
            out,
            c.tradable,
            maximum_trade_fraction,
            stochastic,
            self.temperature,
            &mut self.rng,
            &self.model.action_distribution(),
            order_offset,
        )?;
        // Exploration is part of stochastic acting only. Deterministic
        // evaluation and critic bootstrap must be side-effect free so merely
        // asking for a diagnostic action cannot consume the temperature
        // schedule.
        if stochastic && evaluation.action_sample.is_some() {
            let decay = self.genome.traits.temperature_decay;
            if !decay.is_finite() || decay < 0.0 {
                return Err(MidasError::Policy(
                    "invalid action-temperature decay".into(),
                ));
            }
            self.temperature = (self.temperature * decay).max(f32::MIN_POSITIVE);
        }
        Ok(evaluation)
    }
    fn apply_learning(
        &mut self,
        batch: &crate::learning::PpoTrainingBatch,
        config: &LearningConfig,
    ) -> Result<crate::learning::TrainingMetrics> {
        self.model.train_batch(batch, config)
    }
    fn runtime_state(&self) -> Option<NeuralRuntimeState> {
        self.model.runtime_state()
    }
    fn lifetime_parameters(&self) -> Result<Option<LifetimeParameters>> {
        self.model.lifetime_parameters().map(Some)
    }
    fn snapshot(&self) -> Result<PolicySnapshot> {
        Ok(PolicySnapshot::Neural {
            genome: self.genome.clone(),
            state: Box::new(self.model.snapshot_state()?),
            rng: self.rng.clone(),
            temperature: self.temperature,
            maximum_trade_fraction: self.maximum_trade_fraction,
        })
    }
    fn reset_episode_state(&mut self) {
        self.model.reset_memory();
        self.temperature = self.genome.traits.action_temperature;
    }
    fn policy_blueprint(&self) -> PolicyBlueprint {
        PolicyBlueprint::Neural
    }
    fn temperature_state(&self) -> (f32, f32) {
        (self.temperature, self.genome.traits.action_temperature)
    }
    fn fork(&self) -> Result<Box<dyn Policy>> {
        Ok(Box::new(Self {
            genome: self.genome.clone(),
            model: self.model.fork()?,
            rng: self.rng.clone(),
            temperature: self.temperature,
            maximum_trade_fraction: self.maximum_trade_fraction,
        }))
    }
}
