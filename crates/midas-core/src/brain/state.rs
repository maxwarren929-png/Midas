use super::genome::validate_dense;
use super::{BrainArchitecture, BrainGenome, DenseGenome};
use crate::{MidasError, Result, action::ProposedAction, observation::Observation};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LifetimeParameters {
    pub hidden: Vec<DenseGenome>,
    pub recurrent: DenseGenome,
    pub actor: DenseGenome,
    pub critic: DenseGenome,
}
impl LifetimeParameters {
    pub(super) fn from_genome(genome: &BrainGenome) -> Self {
        Self {
            hidden: genome.hidden.clone(),
            recurrent: genome.recurrent.clone(),
            actor: genome.actor.clone(),
            critic: genome.critic.clone(),
        }
    }
    pub(super) fn validate(&self, architecture: &BrainArchitecture) -> Result<()> {
        if self.hidden.len() != architecture.hidden_layers.len()
            || self
                .hidden
                .iter()
                .zip(&architecture.hidden_layers)
                .any(|(layer, expected_outputs)| layer.outputs != *expected_outputs)
        {
            return Err(MidasError::Policy(
                "lifetime hidden-layer layout mismatch".into(),
            ));
        }
        let mut expected = architecture.observation_size;
        for layer in &self.hidden {
            validate_dense(layer, expected)?;
            expected = layer.outputs;
        }
        validate_dense(&self.recurrent, expected + architecture.memory.size)?;
        validate_dense(&self.actor, architecture.memory.size)?;
        validate_dense(&self.critic, architecture.memory.size)?;
        if !matches!(
            self.recurrent.outputs,
            outputs if outputs == architecture.memory.size || outputs == architecture.memory.size * 3
        ) || self.actor.outputs != architecture.actor_output_size
            || self.critic.outputs != architecture.critic_output_size
        {
            return Err(MidasError::Policy(
                "lifetime parameter output dimensions mismatch".into(),
            ));
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct NeuralRuntimeState {
    pub memory: Vec<f32>,
    pub inference_steps: u64,
}
#[derive(Clone, Debug)]
pub struct NeuralInput<'a> {
    pub observation: &'a Observation,
}
#[derive(Clone, Debug, PartialEq)]
pub struct NeuralOutput {
    pub logits: Vec<f32>,
    pub value: f32,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PolicyActionSample {
    pub active_asset_indices: Vec<usize>,
    pub latent_samples: Vec<f32>,
    pub trade_impulses: Vec<f32>,
    pub maximum_trade_fraction: f32,
    pub log_probability: f32,
    pub entropy: f32,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PolicyEvaluation {
    pub proposed_action: ProposedAction,
    pub action_sample: Option<PolicyActionSample>,
    pub log_probability: Option<f32>,
    pub entropy: f32,
    pub value_estimate: f32,
    pub raw_logits: Vec<f32>,
    pub temperature: f32,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum NeuralModelState {
    /// Legacy frozen-encoder Burn state retained for checkpoint migration.
    Burn {
        parameters: Box<LifetimeParameters>,
        runtime: NeuralRuntimeState,
        optimizer: OptimizerState,
    },
    External {
        kind: String,
        state: Vec<f32>,
    },
    BurnTrainableEncoder {
        parameters: Box<LifetimeParameters>,
        runtime: NeuralRuntimeState,
        optimizer: TrainableEncoderOptimizerState,
    },
    BurnBptt {
        parameters: Box<LifetimeParameters>,
        runtime: NeuralRuntimeState,
        optimizer: BpttOptimizerState,
    },
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DenseOptimizerState {
    pub weight_momentum: Vec<f32>,
    pub bias_momentum: Vec<f32>,
}
impl DenseOptimizerState {
    pub(super) fn empty(inputs: usize, outputs: usize) -> Self {
        Self {
            weight_momentum: vec![0.0; inputs * outputs],
            bias_momentum: vec![0.0; outputs],
        }
    }

    pub(super) fn validate(&self, inputs: usize, outputs: usize, label: &str) -> Result<()> {
        if self.weight_momentum.len() != inputs * outputs
            || self.bias_momentum.len() != outputs
            || self
                .weight_momentum
                .iter()
                .chain(&self.bias_momentum)
                .any(|value| !value.is_finite())
        {
            return Err(MidasError::Checkpoint(format!(
                "neural optimizer {label} state shape or finiteness mismatch"
            )));
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct OptimizerState {
    pub step: u64,
    pub actor: DenseOptimizerState,
    pub critic: DenseOptimizerState,
}
impl OptimizerState {
    pub fn empty(architecture: &BrainArchitecture) -> Self {
        Self {
            step: 0,
            actor: DenseOptimizerState::empty(
                architecture.memory.size,
                architecture.actor_output_size,
            ),
            critic: DenseOptimizerState::empty(
                architecture.memory.size,
                architecture.critic_output_size,
            ),
        }
    }

    pub(super) fn validate(&self, architecture: &BrainArchitecture) -> Result<()> {
        self.actor.validate(
            architecture.memory.size,
            architecture.actor_output_size,
            "actor",
        )?;
        self.critic.validate(
            architecture.memory.size,
            architecture.critic_output_size,
            "critic",
        )?;
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TrainableEncoderOptimizerState {
    pub step: u64,
    pub hidden: Vec<DenseOptimizerState>,
    pub actor: DenseOptimizerState,
    pub critic: DenseOptimizerState,
}
impl TrainableEncoderOptimizerState {
    pub fn empty(architecture: &BrainArchitecture) -> Self {
        let mut inputs = architecture.observation_size;
        let hidden = architecture
            .hidden_layers
            .iter()
            .map(|outputs| {
                let state = DenseOptimizerState::empty(inputs, *outputs);
                inputs = *outputs;
                state
            })
            .collect();
        Self {
            step: 0,
            hidden,
            actor: DenseOptimizerState::empty(
                architecture.memory.size,
                architecture.actor_output_size,
            ),
            critic: DenseOptimizerState::empty(
                architecture.memory.size,
                architecture.critic_output_size,
            ),
        }
    }

    pub(super) fn from_frozen_encoder(
        legacy: OptimizerState,
        architecture: &BrainArchitecture,
    ) -> Result<Self> {
        legacy.validate(architecture)?;
        let mut state = Self::empty(architecture);
        state.step = legacy.step;
        state.actor = legacy.actor;
        state.critic = legacy.critic;
        Ok(state)
    }

    pub(super) fn validate(&self, architecture: &BrainArchitecture) -> Result<()> {
        validate_encoder_optimizer_parts(&self.hidden, &self.actor, &self.critic, architecture)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct BpttOptimizerState {
    pub step: u64,
    pub hidden: Vec<DenseOptimizerState>,
    pub recurrent: DenseOptimizerState,
    pub actor: DenseOptimizerState,
    pub critic: DenseOptimizerState,
}
impl BpttOptimizerState {
    pub fn empty(architecture: &BrainArchitecture, recurrent_outputs: usize) -> Self {
        let encoder = TrainableEncoderOptimizerState::empty(architecture);
        let recurrent_inputs = architecture
            .hidden_layers
            .last()
            .copied()
            .unwrap_or(architecture.observation_size)
            + architecture.memory.size;
        Self {
            step: encoder.step,
            hidden: encoder.hidden,
            recurrent: DenseOptimizerState::empty(recurrent_inputs, recurrent_outputs),
            actor: encoder.actor,
            critic: encoder.critic,
        }
    }

    pub(super) fn from_frozen_encoder(
        legacy: OptimizerState,
        architecture: &BrainArchitecture,
    ) -> Result<Self> {
        let encoder = TrainableEncoderOptimizerState::from_frozen_encoder(legacy, architecture)?;
        Self::from_trainable_encoder(encoder, architecture)
    }

    pub(super) fn from_trainable_encoder(
        encoder: TrainableEncoderOptimizerState,
        architecture: &BrainArchitecture,
    ) -> Result<Self> {
        encoder.validate(architecture)?;
        let mut state = Self::empty(architecture, architecture.memory.size);
        state.step = encoder.step;
        state.hidden = encoder.hidden;
        state.actor = encoder.actor;
        state.critic = encoder.critic;
        Ok(state)
    }

    pub(super) fn validate(
        &self,
        architecture: &BrainArchitecture,
        recurrent_outputs: usize,
    ) -> Result<()> {
        validate_encoder_optimizer_parts(&self.hidden, &self.actor, &self.critic, architecture)?;
        let recurrent_inputs = architecture
            .hidden_layers
            .last()
            .copied()
            .unwrap_or(architecture.observation_size)
            + architecture.memory.size;
        self.recurrent
            .validate(recurrent_inputs, recurrent_outputs, "recurrent")
    }
}

fn validate_encoder_optimizer_parts(
    hidden: &[DenseOptimizerState],
    actor: &DenseOptimizerState,
    critic: &DenseOptimizerState,
    architecture: &BrainArchitecture,
) -> Result<()> {
    if hidden.len() != architecture.hidden_layers.len() {
        return Err(MidasError::Checkpoint(
            "neural optimizer hidden-layer state count mismatch".into(),
        ));
    }
    let mut inputs = architecture.observation_size;
    for (index, (state, outputs)) in hidden.iter().zip(&architecture.hidden_layers).enumerate() {
        state.validate(inputs, *outputs, &format!("hidden layer {index}"))?;
        inputs = *outputs;
    }
    actor.validate(
        architecture.memory.size,
        architecture.actor_output_size,
        "actor",
    )?;
    critic.validate(
        architecture.memory.size,
        architecture.critic_output_size,
        "critic",
    )?;
    Ok(())
}
