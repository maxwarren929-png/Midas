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
        if self.recurrent.outputs != architecture.memory.size
            || self.actor.outputs != architecture.actor_output_size
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
    Burn {
        parameters: Box<LifetimeParameters>,
        runtime: NeuralRuntimeState,
        optimizer: OptimizerState,
    },
    External {
        kind: String,
        state: Vec<f32>,
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
