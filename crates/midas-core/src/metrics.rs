use crate::{AgentId, execution::PlannedTrade, simulation::DomainEvent};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub trait EventSink: Send {
    fn on_event(&mut self, event: &DomainEvent);
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct AgentMetrics {
    pub daily_net_worth: Vec<f64>,
    pub daily_returns: Vec<f64>,
    pub trades: u64,
    pub turnover: f64,
    pub commission: f64,
    pub spread_cost: f64,
    pub slippage_cost: f64,
    pub bankrupt: bool,
    pub survival_days: u64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct MetricsCollector {
    pub agents: BTreeMap<AgentId, AgentMetrics>,
    pub invalid_actions: u64,
    pub rejected_trades: u64,
    pub policy_failures: u64,
    pub deaths: u64,
    pub births: u64,
    pub learning_updates: u64,
    pub fitness_assignments: u64,
    pub reproductions: u64,
    pub evolution_fallbacks: u64,
    pub masked_failure_transitions: u64,
    pub actor_ineligible_transitions: u64,
    pub critic_ineligible_transitions: u64,
    pub invalid_action_penalty_total: f64,
}
impl EventSink for MetricsCollector {
    fn on_event(&mut self, event: &DomainEvent) {
        match event {
            DomainEvent::AgentBorn { agent_id, .. } => {
                self.births += 1;
                self.agents.entry(*agent_id).or_default();
            }
            DomainEvent::Valuation {
                agent_id,
                net_worth,
                previous,
                ..
            } => {
                let m = self.agents.entry(*agent_id).or_default();
                m.daily_net_worth.push(*net_worth);
                m.daily_returns.push(if *previous > 0.0 {
                    net_worth / previous - 1.0
                } else {
                    0.0
                });
                m.survival_days += 1;
            }
            DomainEvent::TradeExecuted {
                agent_id, trade, ..
            } => add_trade(self.agents.entry(*agent_id).or_default(), trade),
            DomainEvent::ActionFallback { .. } => self.invalid_actions += 1,
            DomainEvent::PolicyFailure { .. } => self.policy_failures += 1,
            DomainEvent::LearningUpdate { .. } => self.learning_updates += 1,
            DomainEvent::FitnessAssigned { .. } => self.fitness_assignments += 1,
            DomainEvent::ParentsSelected { .. } => self.reproductions += 1,
            DomainEvent::EvolutionFallback { .. } => self.evolution_fallbacks += 1,
            DomainEvent::LearningTransitionMasked {
                actor_eligible,
                critic_eligible,
                penalty,
                ..
            } => {
                self.masked_failure_transitions += 1;
                if !actor_eligible {
                    self.actor_ineligible_transitions += 1;
                }
                if !critic_eligible {
                    self.critic_ineligible_transitions += 1;
                }
                self.invalid_action_penalty_total += *penalty as f64;
            }
            DomainEvent::TradeRejected { .. } => self.rejected_trades += 1,
            DomainEvent::AgentDied {
                agent_id, cause, ..
            } => {
                self.deaths += 1;
                self.agents.entry(*agent_id).or_default().bankrupt =
                    matches!(cause, crate::simulation::DeathCause::Bankruptcy)
            }
            _ => {}
        }
    }
}
fn add_trade(m: &mut AgentMetrics, t: &PlannedTrade) {
    m.trades += 1;
    m.turnover += t.notional;
    m.commission += t.commission;
    m.spread_cost += t.spread_cost;
    m.slippage_cost += t.slippage_cost;
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub starting_net_worth: f64,
    pub ending_net_worth: f64,
    pub total_return: f64,
    pub annualised_return: f64,
    pub volatility: f64,
    pub maximum_drawdown: f64,
    pub sharpe_like: f64,
    pub sortino_like: f64,
    pub turnover: f64,
    pub trades: u64,
    pub total_fees: f64,
    pub spread_cost: f64,
    pub slippage_cost: f64,
    pub survival_days: u64,
    pub bankrupt: bool,
}
impl AgentMetrics {
    pub fn summary(&self, starting: f64, risk_free: f64) -> PerformanceSummary {
        let ending = self.daily_net_worth.last().copied().unwrap_or(starting);
        let n = self.daily_returns.len() as f64;
        let mean = if n > 0.0 {
            self.daily_returns.iter().sum::<f64>() / n
        } else {
            0.0
        };
        let variance = if n > 0.0 {
            self.daily_returns
                .iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f64>()
                / n
        } else {
            0.0
        };
        let vol = variance.sqrt() * (252.0f64).sqrt();
        let annual = if n > 0.0 {
            (ending / starting).max(0.0).powf(252.0 / n) - 1.0
        } else {
            0.0
        };
        let downside = (if n > 0.0 {
            self.daily_returns
                .iter()
                .filter(|r| **r < 0.0)
                .map(|r| r * r)
                .sum::<f64>()
                / n
        } else {
            0.0
        })
        .sqrt()
            * (252.0f64).sqrt();
        let mut peak = starting;
        let mut dd: f64 = 0.0;
        for v in &self.daily_net_worth {
            peak = peak.max(*v);
            if peak > 0.0 {
                dd = dd.max(1.0 - v / peak)
            }
        }
        PerformanceSummary {
            starting_net_worth: starting,
            ending_net_worth: ending,
            total_return: ending / starting - 1.0,
            annualised_return: annual,
            volatility: vol,
            maximum_drawdown: dd,
            sharpe_like: if vol > 0.0 {
                (annual - risk_free) / vol
            } else {
                0.0
            },
            sortino_like: if downside > 0.0 {
                (annual - risk_free) / downside
            } else {
                0.0
            },
            turnover: self.turnover,
            trades: self.trades,
            total_fees: self.commission,
            spread_cost: self.spread_cost,
            slippage_cost: self.slippage_cost,
            survival_days: self.survival_days,
            bankrupt: self.bankrupt,
        }
    }
}
