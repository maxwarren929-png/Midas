use crate::{
    MidasError, Result,
    config::{RewardConfig, RewardKind},
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug)]
pub struct RewardContext {
    pub previous_net_worth: f64,
    pub current_net_worth: f64,
    pub previous_maximum_drawdown: f64,
    pub maximum_drawdown: f64,
    pub turnover: f64,
    pub bankrupt: bool,
    pub invalid_action: bool,
    /// Optional genome-controlled sensitivities. `None` preserves the
    /// experiment-level reward configuration for baseline runs.
    pub drawdown_sensitivity: Option<f64>,
    pub turnover_sensitivity: Option<f64>,
}

/// A reportable decomposition of the scalar environment reward. The fields
/// are deliberately computed from the same inputs as `reward`, so diagnostics
/// cannot drift from the value used by learning.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct RewardBreakdown {
    pub raw_log_return: f64,
    /// Quantity multiplied by the configured drawdown sensitivity.
    pub drawdown_basis: f64,
    /// Quantity multiplied by the configured turnover sensitivity.
    pub turnover_basis: f64,
    pub drawdown_penalty: f64,
    pub turnover_penalty: f64,
    pub invalid_action_penalty: f64,
    pub bankruptcy_penalty: f64,
    pub unclipped_total: f64,
    pub clipped_total: f32,
}

pub trait RewardFunction: Send + Sync {
    fn reward(&self, context: &RewardContext) -> Result<f32>;
    fn breakdown(&self, context: &RewardContext) -> Result<RewardBreakdown> {
        let value = self.reward(context)?;
        Ok(RewardBreakdown {
            unclipped_total: f64::from(value),
            clipped_total: value,
            ..RewardBreakdown::default()
        })
    }
    fn name(&self) -> &'static str;
}

#[derive(Clone, Debug)]
pub struct LogReturnReward {
    pub config: RewardConfig,
}

impl RewardFunction for LogReturnReward {
    fn reward(&self, context: &RewardContext) -> Result<f32> {
        Ok(self.breakdown(context)?.clipped_total)
    }
    fn breakdown(&self, context: &RewardContext) -> Result<RewardBreakdown> {
        finish_breakdown(log_return(context), &self.config, context)
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
        Ok(self.breakdown(context)?.clipped_total)
    }
    fn breakdown(&self, context: &RewardContext) -> Result<RewardBreakdown> {
        let raw_log_return = log_return(context);
        let drawdown_penalty = context
            .drawdown_sensitivity
            .unwrap_or(self.config.drawdown_penalty_weight as f64)
            * context.maximum_drawdown;
        let turnover_penalty = context
            .turnover_sensitivity
            .unwrap_or(self.config.turnover_penalty_weight as f64)
            * context.turnover;
        finish_breakdown_with_components(
            raw_log_return,
            context.maximum_drawdown,
            context.turnover,
            drawdown_penalty,
            turnover_penalty,
            &self.config,
            context,
        )
    }
    fn name(&self) -> &'static str {
        "risk_adjusted_return"
    }
}

#[derive(Clone, Debug)]
pub struct NormalizedRiskAdjustedReturnReward {
    pub config: RewardConfig,
}

impl RewardFunction for NormalizedRiskAdjustedReturnReward {
    fn reward(&self, context: &RewardContext) -> Result<f32> {
        Ok(self.breakdown(context)?.clipped_total)
    }

    fn breakdown(&self, context: &RewardContext) -> Result<RewardBreakdown> {
        let raw_log_return = log_return(context);
        let drawdown_basis =
            (context.maximum_drawdown - context.previous_maximum_drawdown).max(0.0);
        let turnover_basis = if context.previous_net_worth > 0.0 {
            context.turnover / context.previous_net_worth
        } else {
            0.0
        };
        let drawdown_penalty = context
            .drawdown_sensitivity
            .unwrap_or(self.config.drawdown_penalty_weight as f64)
            * drawdown_basis;
        let turnover_penalty = context
            .turnover_sensitivity
            .unwrap_or(self.config.turnover_penalty_weight as f64)
            * turnover_basis;
        finish_breakdown_with_components(
            raw_log_return,
            drawdown_basis,
            turnover_basis,
            drawdown_penalty,
            turnover_penalty,
            &self.config,
            context,
        )
    }

    fn name(&self) -> &'static str {
        "normalized_risk_adjusted_return"
    }
}

fn log_return(context: &RewardContext) -> f64 {
    if context.previous_net_worth > 0.0 && context.current_net_worth > 0.0 {
        (context.current_net_worth / context.previous_net_worth).ln()
    } else {
        0.0
    }
}

fn finish_breakdown(
    raw_log_return: f64,
    config: &RewardConfig,
    context: &RewardContext,
) -> Result<RewardBreakdown> {
    finish_breakdown_with_components(raw_log_return, 0.0, 0.0, 0.0, 0.0, config, context)
}

fn finish_breakdown_with_components(
    raw_log_return: f64,
    drawdown_basis: f64,
    turnover_basis: f64,
    drawdown_penalty: f64,
    turnover_penalty: f64,
    config: &RewardConfig,
    context: &RewardContext,
) -> Result<RewardBreakdown> {
    if !raw_log_return.is_finite()
        || !drawdown_basis.is_finite()
        || !turnover_basis.is_finite()
        || drawdown_basis < 0.0
        || turnover_basis < 0.0
        || !drawdown_penalty.is_finite()
        || !turnover_penalty.is_finite()
    {
        return Err(MidasError::Policy(
            "reward basis or penalty is invalid".into(),
        ));
    }
    let invalid_action_penalty = if context.invalid_action {
        config.invalid_action_penalty as f64
    } else {
        0.0
    };
    let bankruptcy_penalty = if context.bankrupt {
        config.bankruptcy_penalty as f64
    } else {
        0.0
    };
    let unclipped_total = raw_log_return
        - drawdown_penalty
        - turnover_penalty
        - invalid_action_penalty
        - bankruptcy_penalty;
    let clipped_total = unclipped_total.clamp(config.clip_min as f64, config.clip_max as f64);
    if !unclipped_total.is_finite() || !clipped_total.is_finite() {
        return Err(MidasError::Policy("clipped reward is non-finite".into()));
    }
    Ok(RewardBreakdown {
        raw_log_return,
        drawdown_basis,
        turnover_basis,
        drawdown_penalty,
        turnover_penalty,
        invalid_action_penalty,
        bankruptcy_penalty,
        unclipped_total,
        clipped_total: clipped_total as f32,
    })
}

pub fn configured_reward(config: &RewardConfig) -> Box<dyn RewardFunction> {
    match config.kind {
        RewardKind::LogReturn => Box::new(LogReturnReward {
            config: config.clone(),
        }),
        RewardKind::RiskAdjustedReturn => Box::new(RiskAdjustedReturnReward {
            config: config.clone(),
        }),
        RewardKind::NormalizedRiskAdjustedReturn => Box::new(NormalizedRiskAdjustedReturnReward {
            config: config.clone(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context() -> RewardContext {
        RewardContext {
            previous_net_worth: 100.0,
            current_net_worth: 101.0,
            previous_maximum_drawdown: 0.08,
            maximum_drawdown: 0.1,
            turnover: 2.0,
            bankrupt: false,
            invalid_action: true,
            drawdown_sensitivity: None,
            turnover_sensitivity: None,
        }
    }

    #[test]
    fn reward_breakdown_matches_scalar_for_both_reward_kinds() {
        let mut config = RewardConfig {
            drawdown_penalty_weight: 0.2,
            turnover_penalty_weight: 0.01,
            ..RewardConfig::default()
        };
        for kind in [
            RewardKind::LogReturn,
            RewardKind::RiskAdjustedReturn,
            RewardKind::NormalizedRiskAdjustedReturn,
        ] {
            config.kind = kind;
            let reward = configured_reward(&config);
            let scalar = reward.reward(&context()).unwrap();
            let breakdown = reward.breakdown(&context()).unwrap();
            assert_eq!(scalar, breakdown.clipped_total);
            assert!(
                (breakdown
                    .unclipped_total
                    .clamp(config.clip_min as f64, config.clip_max as f64,)
                    as f32
                    - scalar)
                    .abs()
                    < f32::EPSILON
            );
            if kind == RewardKind::LogReturn {
                assert_eq!(breakdown.drawdown_penalty, 0.0);
                assert_eq!(breakdown.turnover_penalty, 0.0);
            } else {
                assert!(breakdown.drawdown_penalty > 0.0);
                assert!(breakdown.turnover_penalty > 0.0);
            }
            assert!(breakdown.invalid_action_penalty > 0.0);
        }
    }

    #[test]
    fn genome_sensitivities_override_global_risk_penalty_weights_when_present() {
        let config = RewardConfig {
            kind: RewardKind::RiskAdjustedReturn,
            drawdown_penalty_weight: 0.0,
            turnover_penalty_weight: 0.0,
            ..RewardConfig::default()
        };
        let reward = configured_reward(&config);
        let mut context = context();
        context.drawdown_sensitivity = Some(0.5);
        context.turnover_sensitivity = Some(0.25);
        let breakdown = reward.breakdown(&context).unwrap();
        assert!((breakdown.drawdown_penalty - 0.05).abs() < 1e-12);
        assert!((breakdown.turnover_penalty - 0.5).abs() < 1e-12);
    }

    #[test]
    fn normalized_reward_is_scale_invariant_and_does_not_recharge_old_drawdown() {
        let config = RewardConfig {
            kind: RewardKind::NormalizedRiskAdjustedReturn,
            drawdown_penalty_weight: 0.5,
            turnover_penalty_weight: 0.25,
            ..RewardConfig::default()
        };
        let reward = configured_reward(&config);
        let base = context();
        let breakdown = reward.breakdown(&base).unwrap();
        assert!((breakdown.drawdown_basis - 0.02).abs() < 1e-12);
        assert!((breakdown.turnover_basis - 0.02).abs() < 1e-12);
        assert!((breakdown.drawdown_penalty - 0.01).abs() < 1e-12);
        assert!((breakdown.turnover_penalty - 0.005).abs() < 1e-12);

        let mut scaled = base;
        scaled.previous_net_worth *= 10.0;
        scaled.current_net_worth *= 10.0;
        scaled.turnover *= 10.0;
        assert_eq!(breakdown, reward.breakdown(&scaled).unwrap());

        let mut unchanged_drawdown = base;
        unchanged_drawdown.previous_maximum_drawdown = unchanged_drawdown.maximum_drawdown;
        let unchanged = reward.breakdown(&unchanged_drawdown).unwrap();
        assert_eq!(unchanged.drawdown_basis, 0.0);
        assert_eq!(unchanged.drawdown_penalty, 0.0);
    }

    #[test]
    fn normalized_penalty_slopes_match_weights_by_finite_difference() {
        let config = RewardConfig {
            kind: RewardKind::NormalizedRiskAdjustedReturn,
            drawdown_penalty_weight: 0.5,
            turnover_penalty_weight: 0.25,
            ..RewardConfig::default()
        };
        let reward = configured_reward(&config);
        let base = context();
        let baseline = reward.breakdown(&base).unwrap().unclipped_total;
        let epsilon = 1e-6;

        let mut more_drawdown = base;
        more_drawdown.maximum_drawdown += epsilon;
        let drawdown_slope =
            (reward.breakdown(&more_drawdown).unwrap().unclipped_total - baseline) / epsilon;
        assert!((drawdown_slope + 0.5).abs() < 1e-8);

        let mut more_turnover = base;
        more_turnover.turnover += epsilon;
        let turnover_slope =
            (reward.breakdown(&more_turnover).unwrap().unclipped_total - baseline) / epsilon;
        assert!((turnover_slope + 0.25 / base.previous_net_worth).abs() < 1e-8);
    }
}
