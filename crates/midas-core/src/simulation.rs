pub use crate::PolicyFailureCategory;

mod advance;
mod checkpoint;
mod death_replacement;
mod decision;
mod diagnostics;
mod engine;
mod events;
mod learning_bridge;
mod runner;
mod transaction;
mod types;

pub use diagnostics::{DecisionTrace, PpoUpdateTrace, ReportTrace};
pub use engine::SimulationEngine;
pub use events::{DomainEvent, OperationalEvent};
pub use types::{CHECKPOINT_VERSION, DatasetSplit, DeathCause, SimulationLifecycle};

use crate::{
    AgentId, Result,
    action::LongOnlyActionValidator,
    agent::{AgentBuildContext, AgentFactory, StandardAgentFactory},
    brain::{BrainArchitecture, NeuralModelState, PolicySnapshot},
    config::{Config, DeathHandling, LearningKind},
    evolution::EvolutionState,
    execution::{CostExecutionEngine, Portfolio},
    learning::LearningStrategyState,
    market::{BackwardFeaturePipeline, FeaturePipeline, MarketDataset},
    metrics::{EventSink, MetricsCollector},
    observation::{ObservationSchema, SchemaObservationBuilder},
    replacement::{NonEvolutionaryRandomGenomeReplacement, ReplacementStrategy},
    reward::configured_reward,
};
use serde::Serialize;
use std::path::Path;

impl SimulationEngine {
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
