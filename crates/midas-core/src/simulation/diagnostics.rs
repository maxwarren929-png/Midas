use crate::{AgentId, PolicyFailureCategory, reward::RewardBreakdown};
use chrono::NaiveDate;
use serde::{Deserialize, Serialize};

/// Report-only per-decision data. It is committed transactionally, but is not
/// authoritative simulation state and is intentionally excluded from
/// checkpoints so diagnostics do not alter resume compatibility.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DecisionTrace {
    pub observation_date: NaiveDate,
    pub execution_date: NaiveDate,
    pub agent_id: AgentId,
    pub genome_id: Option<crate::evolution::GenomeId>,
    pub generation_depth: Option<u32>,
    pub policy: crate::brain::PolicyBlueprint,
    pub previous_net_worth: f64,
    pub current_net_worth: f64,
    pub reward: f32,
    pub reward_breakdown: RewardBreakdown,
    pub transaction_cost: f64,
    pub commission_cost: f64,
    pub spread_cost: f64,
    pub slippage_cost: f64,
    pub turnover: f64,
    pub trade_count: usize,
    pub buy_notional: f64,
    pub sell_notional: f64,
    pub current_allocations: Vec<f64>,
    pub action_allocations: Vec<f64>,
    pub active_asset_indices: Vec<usize>,
    pub latent_samples: Vec<f32>,
    pub requested_trade_impulses: Vec<f64>,
    pub executed_trade_impulses: Vec<f64>,
    pub trade_scaling_factor: f64,
    pub requested_buy_notional: f64,
    pub requested_sell_notional: f64,
    pub executed_buy_notional: f64,
    pub executed_sell_notional: f64,
    pub rejected_notional: f64,
    pub cash_before: f64,
    pub cash_after: f64,
    pub holdings_before: Vec<f64>,
    pub holdings_after: Vec<f64>,
    pub order_position_allocations: Vec<f64>,
    pub cash_allocation: f64,
    pub maximum_concentration: f64,
    pub active_positions: usize,
    pub entropy: f32,
    pub old_log_probability: Option<f32>,
    pub value_estimate: Option<f32>,
    pub temperature: Option<f32>,
    pub invalid_action: bool,
    pub failure_category: Option<PolicyFailureCategory>,
    pub actor_eligible: bool,
    pub critic_eligible: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PpoUpdateTrace {
    pub agent_id: AgentId,
    pub date: NaiveDate,
    pub rollout_size: usize,
    pub policy_loss: f32,
    pub value_loss: f32,
    pub entropy: f32,
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
    pub actor_delta_norm: f64,
    pub critic_delta_norm: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[allow(clippy::large_enum_variant)]
pub enum ReportTrace {
    Decision(DecisionTrace),
    PpoUpdate(PpoUpdateTrace),
}
