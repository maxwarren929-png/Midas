use crate::{
    AgentId, Result,
    action::ValidatedAction,
    agent::Agent,
    execution::ExecutionSnapshot,
    market::CorporateAction,
    reward::{RewardBreakdown, RewardContext},
};
use chrono::NaiveDate;

use super::{DeathCause, DomainEvent, SimulationEngine};

pub(super) struct TradeSummary {
    pub(super) transaction_cost: f64,
    pub(super) commission_cost: f64,
    pub(super) spread_cost: f64,
    pub(super) slippage_cost: f64,
    pub(super) turnover: f64,
    pub(super) trade_count: usize,
    pub(super) buy_notional: f64,
    pub(super) sell_notional: f64,
    pub(super) requested_trade_impulses: Vec<f64>,
    pub(super) executed_trade_impulses: Vec<f64>,
    pub(super) trade_scaling_factor: f64,
    pub(super) requested_buy_notional: f64,
    pub(super) requested_sell_notional: f64,
    pub(super) rejected_notional: f64,
    pub(super) cash_before: f64,
    pub(super) cash_after: f64,
    pub(super) holdings_before: Vec<f64>,
    pub(super) holdings_after: Vec<f64>,
}
pub(super) struct AdvanceResult {
    pub(super) events: Vec<DomainEvent>,
    pub(super) death: Option<(AgentId, NaiveDate)>,
    pub(super) reward: f32,
    pub(super) reward_breakdown: RewardBreakdown,
    pub(super) trade: TradeSummary,
    pub(super) previous_net_worth: f64,
    pub(super) current_net_worth: f64,
}

impl SimulationEngine {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn advance_agent(
        &self,
        agent: &mut Agent,
        validated: &ValidatedAction,
        _observation_date: NaiveDate,
        execution_date: NaiveDate,
        current_prices: &[Option<f64>],
        next_prices: &[Option<f64>],
        bars: &[Option<&crate::market::MarketBar>],
        actions: &[CorporateAction],
        invalid_action: bool,
    ) -> Result<AdvanceResult> {
        let mut events = Vec::new();
        let turnover_before = agent.state.portfolio.cumulative_turnover;
        let cash_before = agent.state.portfolio.cash;
        let holdings_before = agent
            .state
            .portfolio
            .positions
            .iter()
            .map(|position| position.quantity)
            .collect::<Vec<_>>();
        let current_worth = agent.state.portfolio.value(current_prices)?;
        // Actions take effect at the next open, in the canonical split then
        // dividend order supplied by MarketDataset.
        for action in actions {
            agent.state.portfolio.apply_corporate_action(action)?;
            events.push(DomainEvent::CorporateActionApplied {
                agent_id: agent.state.id,
                date: execution_date,
                action: action.clone(),
            });
        }
        let plan = self.planner.plan_action(
            &agent.state.portfolio,
            validated,
            &ExecutionSnapshot {
                bars,
                valuation_prices: current_prices,
            },
            self.config.simulation.minimum_trade_value,
        )?;
        let report = self.applier.apply(&mut agent.state.portfolio, &plan)?;
        let mut requested_trade_impulses = Vec::new();
        let mut executed_trade_impulses = Vec::new();
        if let ValidatedAction::TradeImpulses(impulses) = validated {
            requested_trade_impulses = impulses.impulses().to_vec();
            executed_trade_impulses = vec![0.0; impulses.impulses().len()];
            let denominator = current_worth * impulses.maximum_trade_fraction();
            if denominator > 0.0 && denominator.is_finite() {
                for trade in &report.trades {
                    if let Some(index) = agent
                        .state
                        .portfolio
                        .positions
                        .iter()
                        .position(|position| position.asset_id == trade.asset_id)
                    {
                        let sign = if matches!(trade.side, crate::execution::TradeSide::Buy) {
                            1.0
                        } else {
                            -1.0
                        };
                        executed_trade_impulses[index] += sign * trade.notional / denominator;
                    }
                }
            }
        }
        let commission_cost = report
            .trades
            .iter()
            .map(|trade| trade.commission)
            .sum::<f64>();
        let spread_cost = report
            .trades
            .iter()
            .map(|trade| trade.spread_cost)
            .sum::<f64>();
        let slippage_cost = report
            .trades
            .iter()
            .map(|trade| trade.slippage_cost)
            .sum::<f64>();
        let mut trade_summary = TradeSummary {
            transaction_cost: commission_cost + spread_cost + slippage_cost,
            commission_cost,
            spread_cost,
            slippage_cost,
            turnover: 0.0,
            trade_count: report.trades.len(),
            buy_notional: report
                .trades
                .iter()
                .filter(|trade| matches!(trade.side, crate::execution::TradeSide::Buy))
                .map(|trade| trade.notional)
                .sum(),
            sell_notional: report
                .trades
                .iter()
                .filter(|trade| matches!(trade.side, crate::execution::TradeSide::Sell))
                .map(|trade| trade.notional)
                .sum(),
            requested_trade_impulses,
            executed_trade_impulses,
            trade_scaling_factor: plan.affordability_scale,
            requested_buy_notional: plan.requested_buy_notional,
            requested_sell_notional: plan.requested_sell_notional,
            rejected_notional: plan.rejected_notional,
            cash_before,
            cash_after: agent.state.portfolio.cash,
            holdings_before,
            holdings_after: agent
                .state
                .portfolio
                .positions
                .iter()
                .map(|position| position.quantity)
                .collect(),
        };
        for trade in report.trades {
            events.push(DomainEvent::TradeExecuted {
                agent_id: agent.state.id,
                date: execution_date,
                trade,
            });
        }
        for asset_id in report.rejected_assets {
            events.push(DomainEvent::TradeRejected {
                agent_id: agent.state.id,
                date: execution_date,
                asset_id,
            });
        }
        let previous = agent.state.portfolio.net_worth;
        agent.state.portfolio.previous_net_worth = previous;
        agent.state.portfolio.net_worth = agent.state.portfolio.value(next_prices)?;
        agent
            .state
            .portfolio
            .apply_survival_cost(self.config.simulation.daily_survival_cost);
        let current = agent.state.portfolio.net_worth;
        let bankrupt = current <= self.config.simulation.bankruptcy_threshold;
        agent.state.age += 1;
        let previous_maximum_drawdown = agent.state.maximum_drawdown;
        agent.state.maximum_net_worth = agent.state.maximum_net_worth.max(current);
        if agent.state.maximum_net_worth > 0.0 {
            agent.state.maximum_drawdown = agent
                .state
                .maximum_drawdown
                .max(1.0 - current / agent.state.maximum_net_worth);
        }
        let use_genome_sensitivities = agent.state.genome.is_some()
            && (self.config.brain.traits.drawdown_sensitivity.evolvable
                || self.config.brain.traits.turnover_sensitivity.evolvable);
        let genome_sensitivities = agent.state.genome.as_ref().map(|genome| {
            (
                f64::from(genome.traits.drawdown_sensitivity),
                f64::from(genome.traits.turnover_sensitivity),
            )
        });
        let reward_context = RewardContext {
            previous_net_worth: previous,
            current_net_worth: current,
            previous_maximum_drawdown,
            maximum_drawdown: agent.state.maximum_drawdown,
            turnover: (agent.state.portfolio.cumulative_turnover - turnover_before).max(0.0),
            bankrupt,
            invalid_action,
            drawdown_sensitivity: use_genome_sensitivities
                .then(|| genome_sensitivities.map(|values| values.0))
                .flatten(),
            turnover_sensitivity: use_genome_sensitivities
                .then(|| genome_sensitivities.map(|values| values.1))
                .flatten(),
        };
        let reward_breakdown = self.reward_function.breakdown(&reward_context)?;
        let reward = reward_breakdown.clipped_total;
        agent.state.update_reward_statistics(reward);
        trade_summary.turnover = reward_context.turnover;
        trade_summary.cash_after = agent.state.portfolio.cash;
        trade_summary.holdings_after = agent
            .state
            .portfolio
            .positions
            .iter()
            .map(|position| position.quantity)
            .collect();
        events.push(DomainEvent::Valuation {
            agent_id: agent.state.id,
            date: execution_date,
            net_worth: current,
            previous,
            reward,
        });
        if bankrupt {
            agent.state.alive = false;
            events.push(DomainEvent::AgentDied {
                agent_id: agent.state.id,
                birth_date: agent.state.birth_date,
                death_date: execution_date,
                age: agent.state.age,
                starting_net_worth: self.config.simulation.starting_cash,
                final_net_worth: current,
                maximum_net_worth: agent.state.maximum_net_worth,
                cause: DeathCause::Bankruptcy,
            });
            return Ok(AdvanceResult {
                events,
                death: Some((agent.state.id, execution_date)),
                reward,
                reward_breakdown,
                trade: trade_summary,
                previous_net_worth: previous,
                current_net_worth: current,
            });
        }
        Ok(AdvanceResult {
            events,
            death: None,
            reward,
            reward_breakdown,
            trade: trade_summary,
            previous_net_worth: previous,
            current_net_worth: current,
        })
    }
}
