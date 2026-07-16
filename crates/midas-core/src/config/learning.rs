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
