use crate::{
    AgentId, AssetId, PolicyFailureCategory, execution::PlannedTrade, market::CorporateAction,
};
use chrono::NaiveDate;
use serde::{Deserialize, Serialize};

use super::DeathCause;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum DomainEvent {
    DayStarted {
        date: NaiveDate,
        step: usize,
    },
    ObservationRevealed {
        date: NaiveDate,
    },
    AgentBorn {
        agent_id: AgentId,
        date: NaiveDate,
    },
    ActionFallback {
        agent_id: AgentId,
        date: NaiveDate,
    },
    LearningUpdate {
        agent_id: AgentId,
        date: NaiveDate,
        rollout_size: usize,
        policy_loss: f32,
        value_loss: f32,
        entropy: f32,
        gradient_norm: f32,
    },
    LearningTransitionMasked {
        agent_id: AgentId,
        date: NaiveDate,
        actor_eligible: bool,
        critic_eligible: bool,
        penalty: f32,
    },
    FitnessAssigned {
        agent_id: AgentId,
        genome_id: Option<crate::evolution::GenomeId>,
        fitness: f64,
    },
    ParentsSelected {
        child_agent_id: AgentId,
        parent_agent_ids: Vec<AgentId>,
    },
    DiversityConstraintRelaxed {
        first_parent_agent_id: AgentId,
        requested_minimum_distance: f64,
        selected_distance: f64,
    },
    GenomeCrossed {
        child_agent_id: AgentId,
        method: crate::config::CrossoverMethod,
    },
    GenomeMutated {
        child_agent_id: AgentId,
        weight_changes: u64,
        bias_changes: u64,
        trait_changes: u64,
        reset_changes: u64,
        weight_resets: u64,
        bias_resets: u64,
    },
    EvolutionFallback {
        reason: String,
    },
    LineageCreated {
        lineage_id: crate::evolution::LineageId,
        parent_lineages: Vec<crate::evolution::LineageId>,
    },
    PolicyFailure {
        agent_id: AgentId,
        date: NaiveDate,
        policy_kind: crate::brain::PolicyBlueprint,
        category: PolicyFailureCategory,
        message: String,
    },
    CorporateActionApplied {
        agent_id: AgentId,
        date: NaiveDate,
        action: CorporateAction,
    },
    TradeExecuted {
        agent_id: AgentId,
        date: NaiveDate,
        trade: PlannedTrade,
    },
    TradeRejected {
        agent_id: AgentId,
        date: NaiveDate,
        asset_id: AssetId,
    },
    Valuation {
        agent_id: AgentId,
        date: NaiveDate,
        net_worth: f64,
        previous: f64,
        reward: f32,
    },
    AgentDied {
        agent_id: AgentId,
        birth_date: NaiveDate,
        death_date: NaiveDate,
        age: u64,
        starting_net_worth: f64,
        final_net_worth: f64,
        maximum_net_worth: f64,
        cause: DeathCause,
    },
    TimelineCompleted {
        date: NaiveDate,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum OperationalEvent {
    CheckpointSaved { date: NaiveDate, path: String },
}
