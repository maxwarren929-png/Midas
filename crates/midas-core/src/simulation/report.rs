use crate::AgentId;
use serde::Serialize;

use super::{DatasetSplit, SimulationEngine, SimulationLifecycle};

impl SimulationEngine {
    pub fn report(&self) -> SimulationReport {
        let learning_implementation = match self.config.learning.implementation {
            crate::config::LearningKind::NoLearning => "no_learning",
            crate::config::LearningKind::Ppo => "ppo",
        };
        SimulationReport {
            split: self.split,
            steps: self.next_step,
            agents: self
                .metrics
                .agents
                .iter()
                .map(|(id, m)| (*id, m.summary(self.config.simulation.starting_cash, 0.0)))
                .collect(),
            deaths: self.metrics.deaths,
            births: self.metrics.births,
            invalid_actions: self.metrics.invalid_actions,
            dataset_identity: self.data.identity.clone(),
            observation_schema_identity: self
                .observation_schema
                .identity(self.data.assets.len())
                .expect("serializable schema"),
            replacement_strategy: self.replacement.name().into(),
            learning_implementation: learning_implementation.into(),
            learning_updates: self.metrics.learning_updates,
            evolutionary_births: self.metrics.reproductions,
            neural_action_count: self.metrics.neural_action_count,
            neural_multi_asset_action_fraction: if self.metrics.neural_action_count > 0 {
                self.metrics.neural_multi_asset_actions as f64
                    / self.metrics.neural_action_count as f64
            } else {
                0.0
            },
            neural_mean_active_positions: if self.metrics.neural_action_count > 0 {
                self.metrics.neural_active_position_total / self.metrics.neural_action_count as f64
            } else {
                0.0
            },
            neural_mean_concentration: if self.metrics.neural_action_count > 0 {
                self.metrics.neural_concentration_total / self.metrics.neural_action_count as f64
            } else {
                0.0
            },
            neural_mean_cash_allocation: if self.metrics.neural_action_count > 0 {
                self.metrics.neural_cash_allocation_total / self.metrics.neural_action_count as f64
            } else {
                0.0
            },
            neural_mean_abs_trade_impulse: if self.metrics.neural_impulse_count > 0 {
                self.metrics.neural_abs_impulse_total / self.metrics.neural_impulse_count as f64
            } else {
                0.0
            },
            neural_small_trade_impulse_fraction: if self.metrics.neural_impulse_count > 0 {
                self.metrics.neural_small_impulse_count as f64
                    / self.metrics.neural_impulse_count as f64
            } else {
                0.0
            },
            neural_saturated_trade_impulse_fraction: if self.metrics.neural_impulse_count > 0 {
                self.metrics.neural_saturated_impulse_count as f64
                    / self.metrics.neural_impulse_count as f64
            } else {
                0.0
            },
            active_lineages: self.evolution.metrics.active_lineages,
            lifecycle: self.lifecycle.clone(),
            failure_reason: match &self.lifecycle {
                SimulationLifecycle::Failed { reason } => Some(reason.clone()),
                _ => None,
            },
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct SimulationReport {
    pub split: DatasetSplit,
    pub steps: usize,
    pub agents: Vec<(AgentId, crate::metrics::PerformanceSummary)>,
    pub deaths: u64,
    pub births: u64,
    pub invalid_actions: u64,
    pub dataset_identity: String,
    pub observation_schema_identity: String,
    pub replacement_strategy: String,
    pub learning_implementation: String,
    pub learning_updates: u64,
    pub evolutionary_births: u64,
    pub neural_action_count: u64,
    pub neural_multi_asset_action_fraction: f64,
    pub neural_mean_active_positions: f64,
    pub neural_mean_concentration: f64,
    pub neural_mean_cash_allocation: f64,
    pub neural_mean_abs_trade_impulse: f64,
    pub neural_small_trade_impulse_fraction: f64,
    pub neural_saturated_trade_impulse_fraction: f64,
    pub active_lineages: u64,
    pub lifecycle: SimulationLifecycle,
    pub failure_reason: Option<String>,
}
