use crate::{MidasError, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LearningConfig {
    pub implementation: LearningKind,
    pub learning_rate: f32,
    pub gamma: f32,
    pub gae_lambda: f32,
    pub ppo_clip: f32,
    pub critic_coefficient: f32,
    pub entropy_coefficient: f32,
    pub gradient_clip: f32,
    pub rollout_length: usize,
    #[serde(default = "default_minibatch_size")]
    pub minibatch_size: usize,
    #[serde(default = "default_sequence_length")]
    pub sequence_length: usize,
    #[serde(default = "default_true")]
    pub normalize_advantages: bool,
    pub optimisation_epochs: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OptimizerConfig {
    #[serde(default)]
    pub kind: OptimizerKind,
    #[serde(default = "default_beta1")]
    pub beta1: f32,
    #[serde(default = "default_beta2")]
    pub beta2: f32,
    #[serde(default = "default_epsilon")]
    pub epsilon: f32,
    #[serde(default = "default_weight_decay")]
    pub weight_decay: f32,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            kind: OptimizerKind::Momentum,
            beta1: default_beta1(),
            beta2: default_beta2(),
            epsilon: default_epsilon(),
            weight_decay: default_weight_decay(),
        }
    }
}

impl OptimizerConfig {
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if !self.beta1.is_finite()
            || !(0.0..1.0).contains(&self.beta1)
            || !self.beta2.is_finite()
            || !(0.0..1.0).contains(&self.beta2)
            || !self.epsilon.is_finite()
            || self.epsilon <= 0.0
            || !self.weight_decay.is_finite()
            || self.weight_decay < 0.0
        {
            return Err(MidasError::Config(
                "learning optimizer configuration values are invalid".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerKind {
    #[default]
    Momentum,
    #[serde(rename = "adamw")]
    AdamW,
}

fn default_beta1() -> f32 {
    0.9
}
fn default_beta2() -> f32 {
    0.999
}
fn default_epsilon() -> f32 {
    1.0e-8
}
fn default_weight_decay() -> f32 {
    0.01
}

fn default_minibatch_size() -> usize {
    32
}
fn default_sequence_length() -> usize {
    16
}
fn default_true() -> bool {
    true
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LearningKind {
    NoLearning,
    Ppo,
}

impl LearningConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        if !self.learning_rate.is_finite()
            || self.learning_rate <= 0.0
            || !self.gamma.is_finite()
            || !(0.0..=1.0).contains(&self.gamma)
            || !self.gae_lambda.is_finite()
            || !(0.0..=1.0).contains(&self.gae_lambda)
            || !self.ppo_clip.is_finite()
            || self.ppo_clip < 0.0
            || !self.critic_coefficient.is_finite()
            || self.critic_coefficient < 0.0
            || !self.entropy_coefficient.is_finite()
            || self.entropy_coefficient < 0.0
            || !self.gradient_clip.is_finite()
            || self.gradient_clip <= 0.0
            || self.rollout_length == 0
            || self.minibatch_size == 0
            || self.minibatch_size > self.rollout_length
            || self.sequence_length == 0
            || self.sequence_length > self.rollout_length
            || self.optimisation_epochs == 0
        {
            return Err(MidasError::Config(
                "learning configuration values are invalid".into(),
            ));
        }
        Ok(())
    }
}
