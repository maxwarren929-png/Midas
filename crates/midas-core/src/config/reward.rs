use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RewardConfig {
    #[serde(default)]
    pub kind: RewardKind,
    #[serde(default)]
    pub drawdown_penalty_weight: f32,
    #[serde(default)]
    pub turnover_penalty_weight: f32,
    #[serde(default)]
    pub invalid_action_penalty: f32,
    #[serde(default = "default_bankruptcy_penalty")]
    pub bankruptcy_penalty: f32,
    #[serde(default = "default_reward_clip_min")]
    pub clip_min: f32,
    #[serde(default = "default_reward_clip_max")]
    pub clip_max: f32,
}
fn default_bankruptcy_penalty() -> f32 {
    1.0
}
fn default_reward_clip_min() -> f32 {
    -5.0
}
fn default_reward_clip_max() -> f32 {
    5.0
}
impl Default for RewardConfig {
    fn default() -> Self {
        Self {
            kind: RewardKind::LogReturn,
            drawdown_penalty_weight: 0.0,
            turnover_penalty_weight: 0.0,
            invalid_action_penalty: 0.05,
            bankruptcy_penalty: 1.0,
            clip_min: -5.0,
            clip_max: 5.0,
        }
    }
}
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RewardKind {
    #[default]
    LogReturn,
    RiskAdjustedReturn,
}
