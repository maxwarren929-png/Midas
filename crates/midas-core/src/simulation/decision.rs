use crate::{
    AssetId, MidasError, PolicyFailureCategory, Result,
    action::{ActionValidationContext, ProposedAction, ProposedAllocation, ValidatedAction},
    agent::Agent,
    brain::PolicyContext,
    config::{LearningKind, PolicyFailureBehavior},
    market::FeatureRow,
    observation::ObservationContext,
};
use chrono::NaiveDate;

use super::{
    DeathCause, DecisionTrace, DomainEvent, ReportTrace, SimulationEngine,
    advance::AdvanceResult,
    transaction::{StepAbort, StepTransaction},
};

#[derive(Clone, Debug)]
struct PolicyFailureIncident {
    category: PolicyFailureCategory,
    message: String,
}
enum PolicyDecisionOutcome {
    Valid {
        validated: ValidatedAction,
        observation: crate::observation::Observation,
        evaluation: crate::brain::PolicyEvaluation,
        runtime_before: Option<crate::brain::NeuralRuntimeState>,
    },
    RecoverableFailure {
        incident: PolicyFailureIncident,
        fallback: ValidatedAction,
        observation: Option<crate::observation::Observation>,
        evaluation: Option<crate::brain::PolicyEvaluation>,
        runtime_before: Option<crate::brain::NeuralRuntimeState>,
    },
    FatalAgentFailure {
        incident: PolicyFailureIncident,
    },
    FatalExperimentFailure {
        incident: PolicyFailureIncident,
    },
}

fn learning_eligibility(
    observation: &Option<crate::observation::Observation>,
    evaluation: &Option<crate::brain::PolicyEvaluation>,
    runtime_before: &Option<crate::brain::NeuralRuntimeState>,
    invalid_action: bool,
) -> (bool, bool) {
    let action_sample = evaluation
        .as_ref()
        .and_then(|evaluation| evaluation.action_sample.as_ref());
    let log_probability = evaluation
        .as_ref()
        .and_then(|evaluation| evaluation.log_probability);
    let temperature = evaluation.as_ref().map(|evaluation| evaluation.temperature);
    let value_estimate = evaluation
        .as_ref()
        .map(|evaluation| evaluation.value_estimate)
        .filter(|value| value.is_finite());
    let actor_eligible = !invalid_action
        && observation.is_some()
        && runtime_before.is_some()
        && action_sample.is_some()
        && log_probability.is_some()
        && temperature.is_some();
    let critic_eligible =
        observation.is_some() && runtime_before.is_some() && value_estimate.is_some();
    (actor_eligible, critic_eligible)
}

impl SimulationEngine {
    pub(super) fn execute_step(
        &mut self,
        transaction: &mut StepTransaction,
    ) -> std::result::Result<(), StepAbort> {
        let t = self.next_step;
        let observation_date = self.features[t].date;
        let execution_date = self.features[t + 1].date;
        let current_prices = self.valuation_prices(observation_date);
        let next_prices = self.valuation_prices(execution_date);
        let bars = (0..self.data.assets.len())
            .map(|i| self.data.bar(execution_date, AssetId(i as u64)))
            .collect::<Vec<_>>();
        let tradable = self.features[t].tradable.clone();
        let next_tradable = self.features[t + 1].tradable.clone();
        let market = self.features[t].clone();
        let actions = self
            .data
            .corporate_actions_on(execution_date)
            .cloned()
            .collect::<Vec<_>>();
        transaction.events.push(DomainEvent::DayStarted {
            date: execution_date,
            step: t,
        });
        transaction.events.push(DomainEvent::ObservationRevealed {
            date: observation_date,
        });

        let mut deaths = Vec::new();
        for agent in &mut transaction.agents {
            if !agent.state.alive {
                continue;
            }
            let allocations = agent.state.portfolio.allocations(&current_prices)?;
            let maximum_asset_allocation = agent
                .state
                .genome
                .as_ref()
                .map(|genome| genome.traits.maximum_allocation as f64)
                .unwrap_or(self.config.brain.traits.maximum_allocation.initial as f64);
            let policy_cap = match agent.policy.policy_blueprint() {
                crate::brain::PolicyBlueprint::Random => Some(maximum_asset_allocation),
                crate::brain::PolicyBlueprint::Neural => None,
                crate::brain::PolicyBlueprint::Cash
                | crate::brain::PolicyBlueprint::EqualWeight
                | crate::brain::PolicyBlueprint::BuyAndHold { .. }
                | crate::brain::PolicyBlueprint::Momentum { .. } => None,
            };
            let action_context = ActionValidationContext {
                asset_count: self.data.assets.len(),
                tradable: &tradable,
                policy_maximum_asset_allocation: policy_cap,
                policy_maximum_trade_fraction: matches!(
                    agent.policy.policy_blueprint(),
                    crate::brain::PolicyBlueprint::Neural
                )
                .then_some(self.neural_maximum_trade_fraction(agent)),
                market_maximum_asset_allocation: None,
            };
            let outcome = self.decide_policy(
                agent,
                &market,
                &current_prices,
                &tradable,
                &allocations,
                &action_context,
            )?;
            match outcome {
                PolicyDecisionOutcome::Valid {
                    validated,
                    observation,
                    evaluation,
                    runtime_before,
                } => {
                    if matches!(
                        agent.policy.policy_blueprint(),
                        crate::brain::PolicyBlueprint::Neural
                    ) && let ValidatedAction::Allocation(allocation) = &validated
                    {
                        transaction.allocation_stats.record(allocation.weights())?;
                    }
                    let trace_observation = observation.clone();
                    let trace_evaluation = evaluation.clone();
                    let trace_runtime_before = runtime_before.clone();
                    let mut action_allocations = match &validated {
                        ValidatedAction::Allocation(allocation) => allocation.weights().to_vec(),
                        ValidatedAction::TradeImpulses(_) => allocations.clone(),
                    };
                    let advanced = self.advance_agent(
                        agent,
                        &validated,
                        observation_date,
                        execution_date,
                        &current_prices,
                        &next_prices,
                        &bars,
                        &actions,
                        false,
                    )?;
                    if matches!(
                        agent.policy.policy_blueprint(),
                        crate::brain::PolicyBlueprint::Neural
                    ) {
                        action_allocations = agent.state.portfolio.allocations(&next_prices)?;
                        transaction.allocation_stats.record(&action_allocations)?;
                        if let ValidatedAction::TradeImpulses(impulses) = &validated {
                            transaction
                                .allocation_stats
                                .record_impulses(impulses.impulses())?;
                        }
                    }
                    transaction.events.extend(advanced.events.clone());
                    let next_value = if advanced.death.is_none()
                        && self.config.learning.implementation == LearningKind::Ppo
                    {
                        Some(self.next_value_estimate(
                            agent,
                            t + 1,
                            &next_prices,
                            &next_tradable,
                        )?)
                    } else {
                        None
                    };
                    self.record_learning(
                        agent,
                        Some(observation),
                        Some(evaluation),
                        validated,
                        advanced.reward,
                        next_value,
                        advanced.death.is_some(),
                        observation_date,
                        &tradable,
                        runtime_before,
                        false,
                        None,
                        &mut transaction.events,
                        &mut transaction.traces,
                    )?;
                    let (actor_eligible, critic_eligible) = learning_eligibility(
                        &Some(trace_observation.clone()),
                        &Some(trace_evaluation.clone()),
                        &trace_runtime_before,
                        false,
                    );
                    transaction
                        .traces
                        .push(ReportTrace::Decision(self.decision_trace(
                            agent,
                            observation_date,
                            execution_date,
                            &allocations,
                            &action_allocations,
                            Some(&trace_evaluation),
                            &advanced,
                            false,
                            None,
                            actor_eligible,
                            critic_eligible,
                        )));
                    if let Some(death) = advanced.death {
                        deaths.push(death);
                    }
                }
                PolicyDecisionOutcome::RecoverableFailure {
                    incident,
                    fallback,
                    observation,
                    evaluation,
                    runtime_before,
                } => {
                    if matches!(
                        agent.policy.policy_blueprint(),
                        crate::brain::PolicyBlueprint::Neural
                    ) && let ValidatedAction::Allocation(allocation) = &fallback
                    {
                        transaction.allocation_stats.record(allocation.weights())?;
                    }
                    let failure_event =
                        self.policy_failure_event(agent, observation_date, incident.clone());
                    transaction.events.push(failure_event);
                    transaction.events.push(DomainEvent::ActionFallback {
                        agent_id: agent.state.id,
                        date: observation_date,
                    });
                    let mut action_allocations = match &fallback {
                        ValidatedAction::Allocation(allocation) => allocation.weights().to_vec(),
                        ValidatedAction::TradeImpulses(_) => allocations.clone(),
                    };
                    let trace_observation = observation.clone();
                    let trace_evaluation = evaluation.clone();
                    let trace_runtime_before = runtime_before.clone();
                    let advanced = self.advance_agent(
                        agent,
                        &fallback,
                        observation_date,
                        execution_date,
                        &current_prices,
                        &next_prices,
                        &bars,
                        &actions,
                        true,
                    )?;
                    if matches!(
                        agent.policy.policy_blueprint(),
                        crate::brain::PolicyBlueprint::Neural
                    ) {
                        action_allocations = agent.state.portfolio.allocations(&next_prices)?;
                        transaction.allocation_stats.record(&action_allocations)?;
                        if let ValidatedAction::TradeImpulses(impulses) = &fallback {
                            transaction
                                .allocation_stats
                                .record_impulses(impulses.impulses())?;
                        }
                    }
                    transaction.events.extend(advanced.events.clone());
                    let next_value = if advanced.death.is_none()
                        && evaluation.is_some()
                        && self.config.learning.implementation == LearningKind::Ppo
                    {
                        Some(self.next_value_estimate(
                            agent,
                            t + 1,
                            &next_prices,
                            &next_tradable,
                        )?)
                    } else {
                        None
                    };
                    self.record_learning(
                        agent,
                        observation,
                        evaluation,
                        fallback,
                        advanced.reward,
                        next_value,
                        advanced.death.is_some(),
                        observation_date,
                        &tradable,
                        runtime_before,
                        true,
                        Some(incident.category.clone()),
                        &mut transaction.events,
                        &mut transaction.traces,
                    )?;
                    let (actor_eligible, critic_eligible) = learning_eligibility(
                        &trace_observation,
                        &trace_evaluation,
                        &trace_runtime_before,
                        true,
                    );
                    transaction
                        .traces
                        .push(ReportTrace::Decision(self.decision_trace(
                            agent,
                            observation_date,
                            execution_date,
                            &allocations,
                            &action_allocations,
                            trace_evaluation.as_ref(),
                            &advanced,
                            true,
                            Some(&incident.category),
                            actor_eligible,
                            critic_eligible,
                        )));
                    if let Some(death) = advanced.death {
                        deaths.push(death);
                    }
                }
                PolicyDecisionOutcome::FatalAgentFailure { incident } => {
                    let final_net_worth = agent.state.portfolio.value(&current_prices)?;
                    agent.state.portfolio.previous_net_worth = agent.state.portfolio.net_worth;
                    agent.state.portfolio.net_worth = final_net_worth;
                    agent.state.maximum_net_worth =
                        agent.state.maximum_net_worth.max(final_net_worth);
                    if agent.state.maximum_net_worth > 0.0 {
                        agent.state.maximum_drawdown = agent
                            .state
                            .maximum_drawdown
                            .max(1.0 - final_net_worth / agent.state.maximum_net_worth);
                    }
                    agent.state.alive = false;
                    self.finish_learning_episode(
                        agent,
                        observation_date,
                        final_net_worth,
                        false,
                        &mut transaction.events,
                        &mut transaction.traces,
                    )?;
                    transaction.events.push(self.policy_failure_event(
                        agent,
                        observation_date,
                        incident,
                    ));
                    transaction.events.push(DomainEvent::AgentDied {
                        agent_id: agent.state.id,
                        birth_date: agent.state.birth_date,
                        death_date: observation_date,
                        age: agent.state.age,
                        starting_net_worth: self.config.simulation.starting_cash,
                        final_net_worth,
                        maximum_net_worth: agent.state.maximum_net_worth,
                        cause: DeathCause::PolicyFailure,
                    });
                    deaths.push((agent.state.id, observation_date));
                }
                PolicyDecisionOutcome::FatalExperimentFailure { incident } => {
                    let event =
                        self.policy_failure_event(agent, observation_date, incident.clone());
                    return Err(StepAbort::FatalExperiment {
                        event,
                        reason: format!(
                            "policy failure limit reached for agent {} at {}: {}",
                            agent.state.id, observation_date, incident.message
                        ),
                    });
                }
            }
        }
        let eligible_parent_ids = transaction
            .agents
            .iter()
            .filter(|agent| agent.state.alive)
            .map(|agent| agent.state.id)
            .collect::<std::collections::BTreeSet<_>>();
        for (id, date) in deaths {
            self.handle_death_in_transaction(transaction, id, date, &eligible_parent_ids)?;
        }
        Ok(())
    }

    pub(super) fn neural_maximum_trade_fraction(&self, agent: &Agent) -> f64 {
        let configured = self.config.brain.trading.maximum_trade_fraction;
        agent
            .state
            .genome
            .as_ref()
            .map(|genome| configured.min(genome.traits.maximum_allocation as f64))
            .unwrap_or(configured)
    }

    pub(super) fn observation_context<'a>(
        &self,
        agent: &'a Agent,
        market: &'a FeatureRow,
        prices: &'a [Option<f64>],
    ) -> ObservationContext<'a> {
        let (current_temperature, initial_temperature) = agent.policy.temperature_state();
        ObservationContext {
            market,
            portfolio: &agent.state.portfolio,
            prices,
            starting_net_worth: self.config.simulation.starting_cash,
            maximum_net_worth: agent.state.maximum_net_worth,
            age: agent.state.age,
            episode_steps: self.features.len() - 1,
            recent_reward: agent.state.last_reward,
            rolling_mean_reward: agent.state.reward_ema,
            rolling_reward_volatility: agent.state.reward_variance_ema,
            reward_trend: agent.state.reward_trend,
            current_temperature,
            initial_temperature,
            daily_survival_cost: self.config.simulation.daily_survival_cost,
        }
    }

    fn decide_policy(
        &self,
        agent: &mut Agent,
        market: &FeatureRow,
        prices: &[Option<f64>],
        tradable: &[bool],
        allocations: &[f64],
        action_context: &ActionValidationContext<'_>,
    ) -> Result<PolicyDecisionOutcome> {
        let observation = match self
            .observation_builder
            .build(&self.observation_context(agent, market, prices))
        {
            Ok(observation) => observation,
            Err(error) => {
                let incident = PolicyFailureIncident {
                    category: PolicyFailureCategory::InvalidObservation,
                    message: error.to_string(),
                };
                return self.failure_outcome(
                    agent,
                    incident,
                    action_context,
                    None,
                    None,
                    None,
                    None,
                );
            }
        };
        let stochastic = matches!(
            self.config.learning.implementation,
            crate::config::LearningKind::Ppo
        ) && matches!(
            agent.policy.policy_blueprint(),
            crate::brain::PolicyBlueprint::Neural
        );
        let evaluation_cap = if matches!(
            agent.policy.policy_blueprint(),
            crate::brain::PolicyBlueprint::Neural
        ) {
            action_context
                .policy_maximum_trade_fraction
                .unwrap_or(self.config.brain.trading.maximum_trade_fraction)
        } else {
            action_context
                .policy_maximum_asset_allocation
                .or(action_context.market_maximum_asset_allocation)
                .unwrap_or(1.0)
        };
        let runtime_before = agent.policy.runtime_state();
        let evaluation = match agent.policy.evaluate_action(
            &observation,
            &PolicyContext {
                tradable,
                current_allocations: allocations,
            },
            stochastic,
            evaluation_cap,
        ) {
            Ok(evaluation) => evaluation,
            Err(error) => {
                let incident = PolicyFailureIncident {
                    category: classify_policy_failure(&error),
                    message: error.to_string(),
                };
                return self.failure_outcome(
                    agent,
                    incident,
                    action_context,
                    None,
                    None,
                    None,
                    runtime_before,
                );
            }
        };
        let validated = match self
            .validator
            .validate_action(evaluation.proposed_action.clone(), action_context)
        {
            Ok(validated) => validated,
            Err(error) => {
                let incident = PolicyFailureIncident {
                    category: PolicyFailureCategory::InvalidOutput,
                    message: error.to_string(),
                };
                return self.failure_outcome(
                    agent,
                    incident,
                    action_context,
                    None,
                    Some(observation),
                    Some(evaluation),
                    runtime_before,
                );
            }
        };
        if matches!(&validated, ValidatedAction::Allocation(value) if value.used_fallback()) {
            let incident = PolicyFailureIncident {
                category: PolicyFailureCategory::InvalidOutput,
                message: match &validated {
                    ValidatedAction::Allocation(value) => value
                        .fallback_reason()
                        .unwrap_or("invalid policy output")
                        .to_string(),
                    ValidatedAction::TradeImpulses(_) => "invalid policy output".into(),
                },
            };
            self.failure_outcome(
                agent,
                incident,
                action_context,
                Some(validated),
                Some(observation),
                Some(evaluation),
                runtime_before,
            )
        } else {
            agent.state.consecutive_policy_failures = 0;
            Ok(PolicyDecisionOutcome::Valid {
                validated,
                observation,
                evaluation,
                runtime_before,
            })
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn failure_outcome(
        &self,
        agent: &mut Agent,
        incident: PolicyFailureIncident,
        action_context: &ActionValidationContext<'_>,
        fallback: Option<ValidatedAction>,
        observation: Option<crate::observation::Observation>,
        evaluation: Option<crate::brain::PolicyEvaluation>,
        runtime_before: Option<crate::brain::NeuralRuntimeState>,
    ) -> Result<PolicyDecisionOutcome> {
        agent.state.total_policy_failures = agent.state.total_policy_failures.saturating_add(1);
        agent.state.consecutive_policy_failures =
            agent.state.consecutive_policy_failures.saturating_add(1);
        let limits_reached = agent.state.consecutive_policy_failures
            >= self.config.policy_failures.maximum_consecutive
            || agent.state.total_policy_failures >= self.config.policy_failures.maximum_total;
        match self.config.policy_failures.behavior {
            PolicyFailureBehavior::StopExperiment if limits_reached => {
                Ok(PolicyDecisionOutcome::FatalExperimentFailure { incident })
            }
            PolicyFailureBehavior::KillAgent if limits_reached => {
                Ok(PolicyDecisionOutcome::FatalAgentFailure { incident })
            }
            _ => {
                let fallback = match fallback {
                    Some(value) => value,
                    None => self.validator.validate_action(
                        ProposedAction::Allocation(ProposedAllocation::rebalance(vec![
                            f64::NAN;
                            action_context.asset_count
                                + 1
                        ])),
                        action_context,
                    )?,
                };
                if !matches!(&fallback, ValidatedAction::Allocation(value) if value.used_fallback())
                {
                    return Err(MidasError::Policy(
                        "cash fallback did not produce a validated fallback allocation".into(),
                    ));
                }
                Ok(PolicyDecisionOutcome::RecoverableFailure {
                    incident,
                    fallback,
                    observation,
                    evaluation,
                    runtime_before,
                })
            }
        }
    }

    fn policy_failure_event(
        &self,
        agent: &Agent,
        date: NaiveDate,
        incident: PolicyFailureIncident,
    ) -> DomainEvent {
        DomainEvent::PolicyFailure {
            agent_id: agent.state.id,
            date,
            policy_kind: agent.policy.policy_blueprint(),
            category: incident.category,
            message: incident.message,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn decision_trace(
        &self,
        agent: &Agent,
        observation_date: NaiveDate,
        execution_date: NaiveDate,
        current_allocations: &[f64],
        action_allocations: &[f64],
        evaluation: Option<&crate::brain::PolicyEvaluation>,
        advanced: &AdvanceResult,
        invalid_action: bool,
        failure_category: Option<&PolicyFailureCategory>,
        actor_eligible: bool,
        critic_eligible: bool,
    ) -> DecisionTrace {
        let sample = evaluation.and_then(|evaluation| evaluation.action_sample.as_ref());
        let active_asset_indices = sample
            .map(|sample| sample.active_asset_indices.clone())
            .unwrap_or_default();
        let latent_samples = sample
            .map(|sample| sample.latent_samples.clone())
            .unwrap_or_default();
        let order_position_allocations = active_asset_indices
            .iter()
            .map(|asset_index| {
                action_allocations
                    .get(asset_index + 1)
                    .copied()
                    .unwrap_or(0.0)
            })
            .collect::<Vec<_>>();
        let cash_allocation = action_allocations.first().copied().unwrap_or(0.0);
        let maximum_concentration = action_allocations
            .iter()
            .skip(1)
            .copied()
            .fold(0.0_f64, f64::max);
        let active_positions = action_allocations
            .iter()
            .skip(1)
            .filter(|weight| **weight > 1e-8)
            .count();
        let generation_depth = agent
            .state
            .genome_id
            .and_then(|genome_id| self.evolution.genome_registry.get(&genome_id))
            .map(|record| record.generation_depth);
        DecisionTrace {
            observation_date,
            execution_date,
            agent_id: agent.state.id,
            genome_id: agent.state.genome_id,
            generation_depth,
            policy: agent.policy.policy_blueprint(),
            previous_net_worth: advanced.previous_net_worth,
            current_net_worth: advanced.current_net_worth,
            reward: advanced.reward,
            reward_breakdown: advanced.reward_breakdown,
            transaction_cost: advanced.trade.transaction_cost,
            commission_cost: advanced.trade.commission_cost,
            spread_cost: advanced.trade.spread_cost,
            slippage_cost: advanced.trade.slippage_cost,
            turnover: advanced.trade.turnover,
            trade_count: advanced.trade.trade_count,
            buy_notional: advanced.trade.buy_notional,
            sell_notional: advanced.trade.sell_notional,
            current_allocations: current_allocations.to_vec(),
            action_allocations: action_allocations.to_vec(),
            active_asset_indices,
            latent_samples,
            requested_trade_impulses: advanced.trade.requested_trade_impulses.clone(),
            executed_trade_impulses: advanced.trade.executed_trade_impulses.clone(),
            trade_scaling_factor: advanced.trade.trade_scaling_factor,
            requested_buy_notional: advanced.trade.requested_buy_notional,
            requested_sell_notional: advanced.trade.requested_sell_notional,
            executed_buy_notional: advanced.trade.buy_notional,
            executed_sell_notional: advanced.trade.sell_notional,
            rejected_notional: advanced.trade.rejected_notional,
            cash_before: advanced.trade.cash_before,
            cash_after: advanced.trade.cash_after,
            holdings_before: advanced.trade.holdings_before.clone(),
            holdings_after: advanced.trade.holdings_after.clone(),
            order_position_allocations,
            cash_allocation,
            maximum_concentration,
            active_positions,
            entropy: evaluation
                .map(|evaluation| evaluation.entropy)
                .unwrap_or(0.0),
            old_log_probability: evaluation.and_then(|evaluation| evaluation.log_probability),
            value_estimate: evaluation
                .map(|evaluation| evaluation.value_estimate)
                .filter(|value| value.is_finite()),
            temperature: evaluation.map(|evaluation| evaluation.temperature),
            invalid_action,
            failure_category: failure_category.cloned(),
            actor_eligible,
            critic_eligible,
        }
    }
}

fn classify_policy_failure(error: &MidasError) -> PolicyFailureCategory {
    let message = error.to_string().to_ascii_lowercase();
    if message.contains("schema") {
        PolicyFailureCategory::SchemaMismatch
    } else if message.contains("observation") {
        PolicyFailureCategory::InvalidObservation
    } else if message.contains("non-finite") || message.contains("finite") {
        PolicyFailureCategory::NonFiniteState
    } else if message.contains("output") || message.contains("logit") {
        PolicyFailureCategory::InvalidOutput
    } else {
        PolicyFailureCategory::InferenceFailure
    }
}
