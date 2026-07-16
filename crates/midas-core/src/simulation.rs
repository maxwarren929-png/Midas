pub use crate::PolicyFailureCategory;

mod advance;
mod checkpoint;
mod decision;
mod diagnostics;
mod engine;
mod events;
mod runner;
mod transaction;
mod types;

pub use diagnostics::{DecisionTrace, PpoUpdateTrace, ReportTrace};
pub use engine::SimulationEngine;
pub use events::{DomainEvent, OperationalEvent};
pub use types::{CHECKPOINT_VERSION, DatasetSplit, DeathCause, SimulationLifecycle};

use transaction::StepTransaction;

use crate::{
    AgentId, MidasError, Result,
    action::{LongOnlyActionValidator, ValidatedAction},
    agent::{Agent, AgentBuildContext, AgentFactory, StandardAgentFactory},
    brain::{BrainArchitecture, NeuralModelState, PolicyContext, PolicySnapshot},
    config::{Config, DeathHandling, LearningKind},
    evolution::{CompletedLifeSummary, DefaultFitnessFunction, EvolutionState},
    execution::{CostExecutionEngine, Portfolio},
    learning::{EpisodeEndContext, LearningStrategyState, LearningUpdate, Transition},
    market::{BackwardFeaturePipeline, FeaturePipeline, MarketDataset},
    metrics::{EventSink, MetricsCollector},
    observation::{ObservationContext, ObservationSchema, SchemaObservationBuilder},
    replacement::{
        LivingAgentSummary, NonEvolutionaryRandomGenomeReplacement, ReplacementContext,
        ReplacementStrategy,
    },
    reward::configured_reward,
};
use chrono::NaiveDate;
use serde::Serialize;
use std::path::Path;

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
    fn next_value_estimate(
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
        let observation = self.observation_builder.build(&ObservationContext {
            market,
            portfolio: &agent.state.portfolio,
            prices,
            starting_net_worth: self.config.simulation.starting_cash,
            maximum_net_worth: agent.state.maximum_net_worth,
            age: agent.state.age,
            episode_steps: self.features.len() - 1,
        })?;
        let allocations = agent.state.portfolio.allocations(prices)?;
        let maximum_asset_allocation = self.config.brain.trading.maximum_trade_fraction;
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
    fn record_learning(
        &self,
        agent: &mut Agent,
        observation: Option<crate::observation::Observation>,
        evaluation: Option<crate::brain::PolicyEvaluation>,
        action: ValidatedAction,
        reward: f32,
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
        let update = agent.learning.observe_transition(transition)?;
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

    fn finish_learning_episode(
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
            let learning_config = agent
                .learning
                .training_config()
                .unwrap_or_else(|| self.config.learning.clone());
            let lifetime_before = agent.policy.lifetime_parameters()?;
            let metrics = agent.policy.apply_learning(&batch, &learning_config)?;
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
                approximate_kl: metrics.approximate_kl,
                clip_fraction: metrics.clip_fraction,
                actor_delta_norm,
                critic_delta_norm,
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

    fn handle_death_in_transaction(
        &mut self,
        transaction: &mut StepTransaction,
        id: AgentId,
        date: NaiveDate,
    ) -> Result<()> {
        let dead_index = transaction
            .agents
            .iter()
            .position(|a| a.state.id == id)
            .ok_or_else(|| MidasError::Accounting("dead agent missing".into()))?;
        let dead_snapshot = transaction.agents[dead_index].state.clone();
        let birth_step = self
            .features
            .iter()
            .position(|feature| feature.date == dead_snapshot.birth_date)
            .unwrap_or(0);
        let life = CompletedLifeSummary {
            agent_id: id,
            genome_id: dead_snapshot.genome_id,
            birth_step,
            death_step: self.next_step,
            age: dead_snapshot.age,
            starting_net_worth: self.config.simulation.starting_cash,
            final_net_worth: dead_snapshot.portfolio.net_worth,
            maximum_drawdown: dead_snapshot.maximum_drawdown,
            turnover: dead_snapshot.portfolio.cumulative_turnover,
            policy_failures: dead_snapshot.total_policy_failures,
            fitness: None,
        };
        let fitness_function = DefaultFitnessFunction::from_config(&self.config.evolution.fitness);
        let fitness = transaction
            .evolution
            .record_life_with(life, &fitness_function)?;
        transaction.events.push(DomainEvent::FitnessAssigned {
            agent_id: id,
            genome_id: dead_snapshot.genome_id,
            fitness: fitness.0,
        });
        let living_genomes = transaction
            .agents
            .iter()
            .filter(|candidate| candidate.state.alive)
            .filter_map(|candidate| candidate.state.genome.as_ref())
            .collect::<Vec<_>>();
        transaction.evolution.update_diversity(&living_genomes)?;
        match self.config.simulation.death_handling {
            DeathHandling::LeaveVacant => Ok(()),
            DeathHandling::EndAgentOnly => {
                transaction.agents.retain(|a| a.state.id != id);
                Ok(())
            }
            DeathHandling::Replace => {
                let slot = transaction
                    .agents
                    .iter()
                    .position(|a| a.state.id == id)
                    .ok_or_else(|| MidasError::Accounting("dead agent missing".into()))?;
                let dead_state = transaction.agents[slot].state.clone();
                let dead_policy = transaction.agents[slot].policy.policy_blueprint();
                let new_id = AgentId(transaction.next_agent_id);
                transaction.next_agent_id += 1;
                let mut replacement = match transaction.replacement.take() {
                    Some(replacement) => replacement,
                    None => self.replacement.fork()?,
                };
                let request = {
                    let living = transaction
                        .agents
                        .iter()
                        .filter(|a| a.state.alive)
                        .map(|a| LivingAgentSummary {
                            id: a.state.id,
                            net_worth: a.state.portfolio.net_worth,
                            age: a.state.age,
                            policy: a.policy.policy_blueprint(),
                            genome: a.state.genome.as_ref(),
                            maximum_drawdown: a.state.maximum_drawdown,
                            birth_date: a.state.birth_date,
                            parent_ids: a.state.parent_ids.clone(),
                        })
                        .collect::<Vec<_>>();
                    replacement.create_replacement(&ReplacementContext {
                        dead_agent: &dead_state,
                        dead_policy,
                        living_population: &living,
                        config: &self.config,
                        architecture: &self.architecture,
                        compute_device: &self.compute_device,
                        observation_schema: &self.observation_schema,
                        deterministic_seed: transaction.evolution.rng.next_u64()
                            ^ self.config.simulation.seed
                            ^ new_id.0.wrapping_mul(0x9e3779b97f4a7c15),
                        completed_lives: &transaction.evolution.completed_lives,
                        genome_registry: &transaction.evolution.genome_registry,
                    })?
                };
                let mutation_report = replacement.mutation_report();
                let parent_ids = request.parent_ids.clone();
                let first_parent_id = parent_ids.first().copied().unwrap_or(AgentId(0));
                let request_reason = request.metadata.reason.clone();
                let diversity_relaxed = request.metadata.diversity_constraint_relaxed;
                let requested_mating_distance = request.metadata.requested_mating_distance;
                let selected_mating_distance = request.metadata.selected_mating_distance;
                let child_genome = request.genome.clone();
                transaction.replacement = Some(replacement);
                let mut agent = self.factory.spawn(
                    request,
                    &AgentBuildContext {
                        agent_id: new_id,
                        birth_date: date,
                        config: &self.config,
                        architecture: &self.architecture,
                        compute_device: &self.compute_device,
                    },
                )?;
                if let Some(genome) = child_genome.as_ref() {
                    let parent_genome_ids = parent_ids
                        .iter()
                        .filter_map(|parent_id| {
                            transaction
                                .agents
                                .iter()
                                .find(|candidate| candidate.state.id == *parent_id)
                                .and_then(|candidate| candidate.state.genome_id)
                        })
                        .collect::<Vec<_>>();
                    let parent_lineages = parent_genome_ids
                        .iter()
                        .filter_map(|parent_id| {
                            transaction.evolution.genome_registry.get(parent_id)
                        })
                        .map(|record| record.lineage_id)
                        .collect::<Vec<_>>();
                    let shared_lineage = if parent_lineages.len() >= 2
                        && parent_lineages.windows(2).all(|pair| pair[0] == pair[1])
                    {
                        parent_lineages.first().copied()
                    } else {
                        None
                    };
                    let (genome_id, lineage_id) =
                        transaction.evolution.allocate_genome_with_reason(
                            genome,
                            parent_genome_ids,
                            self.next_step,
                            shared_lineage,
                            request_reason.clone(),
                        )?;
                    agent.state.genome_id = Some(genome_id);
                    agent.state.lineage_id = Some(lineage_id);
                    if parent_lineages.len() >= 2
                        && parent_lineages.windows(2).any(|pair| pair[0] != pair[1])
                    {
                        transaction.events.push(DomainEvent::LineageCreated {
                            lineage_id,
                            parent_lineages,
                        });
                    }
                    if !parent_ids.is_empty() {
                        transaction.evolution.metrics.reproduction_count = transaction
                            .evolution
                            .metrics
                            .reproduction_count
                            .saturating_add(1);
                        transaction.events.push(DomainEvent::ParentsSelected {
                            child_agent_id: new_id,
                            parent_agent_ids: parent_ids,
                        });
                        if diversity_relaxed {
                            transaction
                                .evolution
                                .metrics
                                .diversity_constraint_relaxations = transaction
                                .evolution
                                .metrics
                                .diversity_constraint_relaxations
                                .saturating_add(1);
                            transaction
                                .events
                                .push(DomainEvent::DiversityConstraintRelaxed {
                                    first_parent_agent_id: first_parent_id,
                                    requested_minimum_distance: requested_mating_distance,
                                    selected_distance: selected_mating_distance,
                                });
                        }
                    }
                    if request_reason.starts_with("evolution_two_parent:") {
                        transaction.evolution.metrics.crossover_count = transaction
                            .evolution
                            .metrics
                            .crossover_count
                            .saturating_add(1);
                        transaction.evolution.metrics.mutation_count = transaction
                            .evolution
                            .metrics
                            .mutation_count
                            .saturating_add(1);
                        transaction.events.push(DomainEvent::GenomeCrossed {
                            child_agent_id: new_id,
                            method: self.config.evolution.crossover.method,
                        });
                        let report = mutation_report.unwrap_or_default();
                        transaction.events.push(DomainEvent::GenomeMutated {
                            child_agent_id: new_id,
                            weight_changes: report.weight_changes,
                            bias_changes: report.bias_changes,
                            trait_changes: report.trait_changes,
                            reset_changes: report.reset_changes,
                            weight_resets: report.weight_resets,
                            bias_resets: report.bias_resets,
                        });
                    } else if request_reason.contains("fallback") {
                        transaction.evolution.metrics.random_fallback_births = transaction
                            .evolution
                            .metrics
                            .random_fallback_births
                            .saturating_add(1);
                        transaction.events.push(DomainEvent::EvolutionFallback {
                            reason: request_reason,
                        });
                    }
                }
                transaction.agents[slot] = agent;
                let living_genomes = transaction
                    .agents
                    .iter()
                    .filter(|candidate| candidate.state.alive)
                    .filter_map(|candidate| candidate.state.genome.as_ref())
                    .collect::<Vec<_>>();
                transaction.evolution.update_diversity(&living_genomes)?;
                transaction.events.push(DomainEvent::AgentBorn {
                    agent_id: new_id,
                    date,
                });
                Ok(())
            }
        }
    }
    pub fn save_checkpoint(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let checkpoint = checkpoint::from_state(
            &self.config,
            &self.data,
            &self.observation_schema,
            &self.architecture,
            self.split,
            &self.features,
            &self.compute_device,
            self.next_step,
            self.next_agent_id,
            &self.agents,
            self.metrics.clone(),
            self.domain_events.clone(),
            self.evolution.clone(),
            self.lifecycle.clone(),
        )?;
        checkpoint::write(path.as_ref(), &checkpoint)?;
        let date = self
            .features
            .get(self.next_step)
            .map(|f| f.date)
            .unwrap_or_else(|| self.features.last().expect("features").date);
        self.operational_events
            .push(OperationalEvent::CheckpointSaved {
                date,
                path: path.as_ref().display().to_string(),
            });
        Ok(())
    }
    pub fn resume(config: Config, data: MarketDataset, path: impl AsRef<Path>) -> Result<Self> {
        Self::resume_with_factory(
            config,
            data,
            path,
            Box::new(StandardAgentFactory::default()),
        )
    }
    pub fn resume_with_factory(
        config: Config,
        mut data: MarketDataset,
        path: impl AsRef<Path>,
        factory: Box<dyn AgentFactory>,
    ) -> Result<Self> {
        let cp = checkpoint::load_for_resume(path)?;
        checkpoint::validate_dataset(&cp, &data.identity)?;
        data.rebuild_index();
        let pipeline = BackwardFeaturePipeline::from_config(&config);
        let dates = data.split_dates(cp.split.range(&config));
        let features = pipeline.generate(&data, &dates)?;
        let observation_schema = ObservationSchema::from_config(&config);
        let schema_hash = observation_schema.identity(data.assets.len())?;
        checkpoint::validate_schema(&cp, &schema_hash)?;
        let compute_device = config.backend.compute_device()?;
        checkpoint::validate_backend(&cp, &compute_device.identity())?;
        checkpoint::validate_resume_config(&cp, &config.hash()?)?;
        let architecture = BrainArchitecture::from_config(
            &config.brain,
            schema_hash,
            observation_schema.dimension(data.assets.len()),
            data.assets.len(),
        );
        checkpoint::validate_architecture(&cp, &architecture)?;
        checkpoint::validate_feature_count(features.len())?;
        let observation_builder = SchemaObservationBuilder::from_config(
            observation_schema.clone(),
            &config,
            Some(&data),
        )?;
        let birth = features[0].date;
        let agents = cp
            .agents
            .into_iter()
            .map(|a| {
                let agent_id = a.state.id;
                let mut agent = factory.restore(
                    a.state,
                    a.policy,
                    &AgentBuildContext {
                        agent_id,
                        birth_date: birth,
                        config: &config,
                        architecture: &architecture,
                        compute_device: &compute_device,
                    },
                )?;
                agent.restore_learning(a.learning)?;
                Ok(agent)
            })
            .collect::<Result<_>>()?;
        let costs = CostExecutionEngine {
            costs: config.execution.clone(),
        };
        let reward_function = configured_reward(&config.reward);
        let replacement: Box<dyn ReplacementStrategy> = if config.evolution.enabled {
            Box::new(crate::replacement::EvolutionaryReplacementStrategy::default())
        } else {
            Box::new(NonEvolutionaryRandomGenomeReplacement)
        };
        Ok(Self {
            config,
            data,
            split: cp.split,
            features,
            observation_schema,
            observation_builder: Box::new(observation_builder),
            architecture,
            compute_device,
            agents,
            planner: Box::new(costs.clone()),
            applier: Box::new(costs),
            validator: Box::new(LongOnlyActionValidator),
            factory,
            replacement,
            reward_function,
            evolution: cp.evolution,
            metrics: cp.metrics,
            domain_events: cp.domain_events,
            operational_events: Vec::new(),
            report_traces: Vec::new(),
            event_sinks: Vec::new(),
            next_step: cp.next_step,
            next_agent_id: cp.next_agent_id,
            lifecycle: cp.lifecycle,
        })
    }

    /// Evaluate the lifetime parameters stored in a completed training
    /// checkpoint on a separate split. Evaluation starts fresh portfolios and
    /// recurrent state, and disables learning, replacement, and evolution.
    /// The checkpoint's immutable genomes and trained lifetime model state are
    /// restored, while optimizer/rollout state is intentionally not resumed.
    pub fn evaluate_checkpoint(
        config: Config,
        data: MarketDataset,
        path: impl AsRef<Path>,
        split: DatasetSplit,
    ) -> Result<Self> {
        let cp = checkpoint::load_for_evaluation(path)?;
        checkpoint::validate_dataset(&cp, &data.identity)?;
        checkpoint::validate_evaluation_config(&cp, &config.hash()?)?;

        let mut evaluation_config = config;
        evaluation_config.learning.implementation = LearningKind::NoLearning;
        evaluation_config.evolution.enabled = false;
        evaluation_config.simulation.death_handling = DeathHandling::LeaveVacant;
        // Evaluation is a read-only replay of a training checkpoint. It must
        // never overwrite that checkpoint with a different split or config
        // hash through the automatic cadence.
        evaluation_config.checkpoint.every_steps = 0;
        let mut engine = Self::new(evaluation_config, data, split)?;
        let schema_hash = engine
            .observation_schema
            .identity(engine.data.assets.len())?;
        checkpoint::validate_schema(&cp, &schema_hash)?;
        checkpoint::validate_architecture(&cp, &engine.architecture)?;
        checkpoint::validate_backend(&cp, &engine.compute_device.identity())?;
        let birth_date = engine.features[0].date;
        let factory = std::mem::replace(
            &mut engine.factory,
            Box::new(StandardAgentFactory::default()),
        );
        let mut restored = Vec::with_capacity(cp.agents.len());
        for checkpoint_agent in cp.agents {
            let agent_id = checkpoint_agent.state.id;
            let mut state = checkpoint_agent.state;
            state.birth_date = birth_date;
            state.age = 0;
            state.portfolio = Portfolio::new(
                engine.config.simulation.starting_cash,
                engine.architecture.asset_count,
            );
            state.maximum_net_worth = engine.config.simulation.starting_cash;
            state.maximum_drawdown = 0.0;
            state.alive = true;
            state.total_policy_failures = 0;
            state.consecutive_policy_failures = 0;
            let snapshot = reset_evaluation_snapshot(
                checkpoint_agent.policy,
                engine.config.brain.trading.maximum_trade_fraction,
            )?;
            let mut agent = factory.restore(
                state,
                snapshot,
                &AgentBuildContext {
                    agent_id,
                    birth_date,
                    config: &engine.config,
                    architecture: &engine.architecture,
                    compute_device: &engine.compute_device,
                },
            )?;
            agent.restore_learning(LearningStrategyState::NoLearning)?;
            restored.push(agent);
        }
        engine.factory = factory;
        engine.agents = restored;
        engine.metrics = MetricsCollector::default();
        engine.domain_events.clear();
        engine.evolution = EvolutionState::new(engine.config.simulation.seed ^ 0x243f6a8885a308d3);
        engine.next_agent_id = engine
            .agents
            .iter()
            .map(|agent| agent.state.id.0)
            .max()
            .unwrap_or(0)
            .saturating_add(1);
        let agent_ids = engine
            .agents
            .iter()
            .map(|agent| agent.state.id)
            .collect::<Vec<_>>();
        for agent_id in agent_ids {
            let event = DomainEvent::AgentBorn {
                agent_id,
                date: birth_date,
            };
            engine.metrics.on_event(&event);
            engine.domain_events.push(event);
        }
        engine.lifecycle = SimulationLifecycle::Initialized;
        Ok(engine)
    }
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

fn reset_evaluation_snapshot(
    snapshot: PolicySnapshot,
    maximum_trade_fraction: f64,
) -> Result<PolicySnapshot> {
    match snapshot {
        PolicySnapshot::Neural {
            genome, state, rng, ..
        } => {
            let state = match *state {
                NeuralModelState::Burn {
                    parameters,
                    mut runtime,
                    optimizer,
                } => {
                    runtime.memory.fill(0.0);
                    runtime.inference_steps = 0;
                    NeuralModelState::Burn {
                        parameters,
                        runtime,
                        optimizer,
                    }
                }
                NeuralModelState::External { kind, mut state } => {
                    state.fill(0.0);
                    NeuralModelState::External { kind, state }
                }
            };
            Ok(PolicySnapshot::Neural {
                temperature: genome.traits.action_temperature,
                genome,
                state: Box::new(state),
                rng,
                maximum_trade_fraction,
            })
        }
        other => Ok(other),
    }
}
