use crate::{
    MidasError, PolicyFailureCategory, Result,
    action::ValidatedAction,
    agent::Agent,
    brain::PolicyContext,
    learning::{EpisodeEndContext, LearningUpdate, Transition},
    reward::RewardBreakdown,
};
use chrono::NaiveDate;

use super::{DomainEvent, PpoUpdateTrace, ReportTrace, SimulationEngine};

fn dense_parameter_delta(
    before: &crate::brain::DenseGenome,
    after: &crate::brain::DenseGenome,
) -> f64 {
    before
        .weights
        .iter()
        .zip(&after.weights)
        .chain(before.bias.iter().zip(&after.bias))
        .map(|(left, right)| f64::from(*left - *right).powi(2))
        .sum::<f64>()
        .sqrt()
}

impl SimulationEngine {
    pub(super) fn next_value_estimate(
        &self,
        agent: &Agent,
        step: usize,
        prices: &[Option<f64>],
        tradable: &[bool],
    ) -> Result<f32> {
        if !matches!(
            agent.policy.policy_blueprint(),
            crate::brain::PolicyBlueprint::Neural
        ) {
            return Ok(0.0);
        }
        let mut policy = agent.policy.fork().map_err(|error| {
            MidasError::Policy(format!("next-value policy fork failed: {error}"))
        })?;
        let market = self
            .features
            .get(step)
            .ok_or_else(|| MidasError::Data("next-value feature row is missing".into()))?;
        let observation = self
            .observation_builder
            .build(&self.observation_context(agent, market, prices))?;
        let allocations = agent.state.portfolio.allocations(prices)?;
        let maximum_asset_allocation = self.neural_maximum_trade_fraction(agent);
        let evaluation = policy.evaluate_action(
            &observation,
            &PolicyContext {
                tradable,
                current_allocations: &allocations,
            },
            false,
            maximum_asset_allocation,
        )?;
        if evaluation.value_estimate.is_finite() {
            Ok(evaluation.value_estimate)
        } else {
            Err(MidasError::Policy(
                "next-value estimate is non-finite".into(),
            ))
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_learning(
        &self,
        agent: &mut Agent,
        observation: Option<crate::observation::Observation>,
        evaluation: Option<crate::brain::PolicyEvaluation>,
        action: ValidatedAction,
        reward: f32,
        reward_breakdown: RewardBreakdown,
        next_value: Option<f32>,
        done: bool,
        date: NaiveDate,
        tradable: &[bool],
        runtime_before: Option<crate::brain::NeuralRuntimeState>,
        invalid_action: bool,
        failure_category: Option<PolicyFailureCategory>,
        events: &mut Vec<DomainEvent>,
        traces: &mut Vec<ReportTrace>,
    ) -> Result<()> {
        let after = agent.policy.runtime_state();
        let action_sample = evaluation
            .as_ref()
            .and_then(|evaluation| evaluation.action_sample.clone());
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
        let transition = Transition {
            observation,
            action,
            action_sample,
            raw_policy_output: evaluation.map(|evaluation| evaluation.raw_logits),
            log_probability,
            temperature,
            value_estimate,
            reward,
            next_value,
            done,
            truncated: false,
            recurrent_state_before: runtime_before,
            recurrent_state_after: after,
            date,
            tradable: tradable.to_vec(),
            invalid_action,
            failure_category,
            actor_eligible,
            critic_eligible,
        };
        if invalid_action {
            events.push(DomainEvent::LearningTransitionMasked {
                agent_id: agent.state.id,
                date,
                actor_eligible,
                critic_eligible,
                penalty: self.config.reward.invalid_action_penalty,
            });
        }
        let update = agent
            .learning
            .observe_transition_with_diagnostics(transition, reward_breakdown)?;
        self.apply_learning_update(agent, update, date, events, traces)?;
        if done {
            self.finish_learning_episode(
                agent,
                date,
                agent.state.portfolio.net_worth,
                false,
                events,
                traces,
            )?;
        }
        Ok(())
    }

    pub(super) fn finish_learning_episode(
        &self,
        agent: &mut Agent,
        date: NaiveDate,
        final_value: f64,
        truncated: bool,
        events: &mut Vec<DomainEvent>,
        traces: &mut Vec<ReportTrace>,
    ) -> Result<()> {
        let update = agent.learning.end_episode(&EpisodeEndContext {
            date,
            final_value: final_value as f32,
            truncated,
        })?;
        self.apply_learning_update(agent, update, date, events, traces)
    }

    fn apply_learning_update(
        &self,
        agent: &mut Agent,
        update: LearningUpdate,
        date: NaiveDate,
        events: &mut Vec<DomainEvent>,
        traces: &mut Vec<ReportTrace>,
    ) -> Result<()> {
        if let LearningUpdate::Ppo(batch) = update {
            let advantage_diagnostics = batch.advantage_diagnostics.as_deref().cloned();
            let learning_config = agent
                .learning
                .training_config()
                .unwrap_or_else(|| self.config.learning.clone());
            let optimizer = agent
                .learning
                .training_optimizer()
                .unwrap_or_else(|| self.config.optimizer.clone());
            let lifetime_before = agent.policy.lifetime_parameters()?;
            let metrics = agent
                .policy
                .apply_learning(&batch, &learning_config, &optimizer)?;
            let lifetime_after = agent.policy.lifetime_parameters()?;
            agent.learning.commit_update()?;
            let (actor_delta_norm, critic_delta_norm) = match (lifetime_before, lifetime_after) {
                (Some(before), Some(after)) => (
                    dense_parameter_delta(&before.actor, &after.actor),
                    dense_parameter_delta(&before.critic, &after.critic),
                ),
                _ => (0.0, 0.0),
            };
            traces.push(ReportTrace::PpoUpdate(PpoUpdateTrace {
                agent_id: agent.state.id,
                date,
                rollout_size: metrics.rollout_size,
                policy_loss: metrics.policy_loss,
                value_loss: metrics.value_loss,
                entropy: metrics.entropy,
                gradient_norm: metrics.gradient_norm,
                pre_clip_gradient_norm: metrics.pre_clip_gradient_norm,
                hidden_gradient_norms: metrics.hidden_gradient_norms,
                recurrent_gradient_norm: metrics.recurrent_gradient_norm,
                gru_update_gradient_norm: metrics.gru_update_gradient_norm,
                gru_reset_gradient_norm: metrics.gru_reset_gradient_norm,
                gru_candidate_gradient_norm: metrics.gru_candidate_gradient_norm,
                actor_gradient_norm: metrics.actor_gradient_norm,
                critic_gradient_norm: metrics.critic_gradient_norm,
                gradient_clip_fraction: metrics.gradient_clip_fraction,
                relative_update_norm: metrics.relative_update_norm,
                approximate_kl: metrics.approximate_kl,
                clip_fraction: metrics.clip_fraction,
                actor_delta_norm,
                critic_delta_norm,
                advantage_diagnostics,
                actor_gradient_attribution: metrics.actor_gradient_attribution,
            }));
            events.push(DomainEvent::LearningUpdate {
                agent_id: agent.state.id,
                date,
                rollout_size: metrics.rollout_size,
                policy_loss: metrics.policy_loss,
                value_loss: metrics.value_loss,
                entropy: metrics.entropy,
                gradient_norm: metrics.gradient_norm,
            });
        } else {
            agent.learning.commit_update()?;
        }
        Ok(())
    }
}
