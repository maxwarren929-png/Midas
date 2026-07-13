use crate::{
    MidasError, Result,
    config::{RewardConfig, RewardKind},
};

#[derive(Clone, Copy, Debug)]
pub struct RewardContext {
    pub previous_net_worth: f64,
    pub current_net_worth: f64,
    pub maximum_drawdown: f64,
    pub turnover: f64,
    pub bankrupt: bool,
    pub invalid_action: bool,
}

pub trait RewardFunction: Send + Sync {
    fn reward(&self, context: &RewardContext) -> Result<f32>;
    fn name(&self) -> &'static str;
}

#[derive(Clone, Debug)]
pub struct LogReturnReward {
    pub config: RewardConfig,
}

impl RewardFunction for LogReturnReward {
    fn reward(&self, context: &RewardContext) -> Result<f32> {
        let base = if context.previous_net_worth > 0.0 && context.current_net_worth > 0.0 {
            (context.current_net_worth / context.previous_net_worth).ln()
        } else {
            0.0
        };
        finish_reward(base, &self.config, context)
    }
    fn name(&self) -> &'static str {
        "log_return"
    }
}

#[derive(Clone, Debug)]
pub struct RiskAdjustedReturnReward {
    pub config: RewardConfig,
}

impl RewardFunction for RiskAdjustedReturnReward {
    fn reward(&self, context: &RewardContext) -> Result<f32> {
        let base = if context.previous_net_worth > 0.0 && context.current_net_worth > 0.0 {
            (context.current_net_worth / context.previous_net_worth).ln()
        } else {
            0.0
        };
        let adjusted = base
            - self.config.drawdown_penalty_weight as f64 * context.maximum_drawdown
            - self.config.turnover_penalty_weight as f64 * context.turnover;
        finish_reward(adjusted, &self.config, context)
    }
    fn name(&self) -> &'static str {
        "risk_adjusted_return"
    }
}

fn finish_reward(value: f64, config: &RewardConfig, context: &RewardContext) -> Result<f32> {
    if !value.is_finite() {
        return Err(MidasError::Policy("reward is non-finite".into()));
    }
    let mut value = value;
    if context.invalid_action {
        value -= config.invalid_action_penalty as f64;
    }
    if context.bankrupt {
        value -= config.bankruptcy_penalty as f64;
    }
    value = value.clamp(config.clip_min as f64, config.clip_max as f64);
    if !value.is_finite() {
        return Err(MidasError::Policy("clipped reward is non-finite".into()));
    }
    Ok(value as f32)
}

pub fn configured_reward(config: &RewardConfig) -> Box<dyn RewardFunction> {
    match config.kind {
        RewardKind::LogReturn => Box::new(LogReturnReward {
            config: config.clone(),
        }),
        RewardKind::RiskAdjustedReturn => Box::new(RiskAdjustedReturnReward {
            config: config.clone(),
        }),
    }
}
