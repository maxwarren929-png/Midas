use crate::{
    MidasError, Result,
    agent::{AgentBuildContext, StandardAgentFactory},
    brain::{LifetimeParameters, NeuralModelState, PolicySnapshot},
    config::{Config, DeathHandling, LearningKind},
    evolution::EvolutionState,
    execution::Portfolio,
    learning::LearningStrategyState,
    market::MarketDataset,
    metrics::{EventSink, MetricsCollector},
    replacement::{BuiltInReplacementStrategyFactory, ReplacementStrategyFactory},
    seed::SeedDomain,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::path::Path;

use super::{DatasetSplit, DomainEvent, SimulationEngine, SimulationLifecycle, checkpoint};

impl SimulationEngine {
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
        Self::evaluate_checkpoint_with_options(
            config,
            data,
            path,
            split,
            Box::new(BuiltInReplacementStrategyFactory),
            false,
        )
    }

    /// Evaluate the checkpoint's final genomes with all lifetime learning
    /// removed. This preserves selection/replacement outcomes while providing
    /// a matched birth-parameter control for the trained lifetime weights.
    pub fn evaluate_checkpoint_genome_baseline(
        config: Config,
        data: MarketDataset,
        path: impl AsRef<Path>,
        split: DatasetSplit,
    ) -> Result<Self> {
        Self::evaluate_checkpoint_with_options(
            config,
            data,
            path,
            split,
            Box::new(BuiltInReplacementStrategyFactory),
            true,
        )
    }

    pub fn evaluate_checkpoint_with_replacement_factory(
        config: Config,
        data: MarketDataset,
        path: impl AsRef<Path>,
        split: DatasetSplit,
        replacement_factory: Box<dyn ReplacementStrategyFactory>,
    ) -> Result<Self> {
        Self::evaluate_checkpoint_with_options(
            config,
            data,
            path,
            split,
            replacement_factory,
            false,
        )
    }

    fn evaluate_checkpoint_with_options(
        config: Config,
        data: MarketDataset,
        path: impl AsRef<Path>,
        split: DatasetSplit,
        replacement_factory: Box<dyn ReplacementStrategyFactory>,
        reset_lifetime_to_genome: bool,
    ) -> Result<Self> {
        let cp = checkpoint::load_for_evaluation(path)?;
        checkpoint::validate_dataset(&cp, &data.identity)?;
        checkpoint::validate_evaluation_config(&cp, &config.hash()?)?;
        checkpoint::restore_replacement(&cp, &config, replacement_factory.as_ref())?;

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
                engine
                    .config
                    .simulation
                    .seed_derivation
                    .then_evaluation_seed(engine.config.simulation.seed, agent_id.0),
                reset_lifetime_to_genome,
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
            agent.restore_learning(LearningStrategyState::NoLearning, &engine.config.optimizer)?;
            restored.push(agent);
        }
        engine.factory = factory;
        engine.agents = restored;
        engine.metrics = MetricsCollector::default();
        engine.domain_events.clear();
        engine.evolution = EvolutionState::new(engine.config.simulation.seed_derivation.derive(
            engine.config.simulation.seed,
            SeedDomain::EvaluationEvolution,
            0,
        ));
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
}

fn reset_evaluation_snapshot(
    snapshot: PolicySnapshot,
    maximum_trade_fraction: f64,
    evaluation_seed: Option<u64>,
    reset_lifetime_to_genome: bool,
) -> Result<PolicySnapshot> {
    match snapshot {
        PolicySnapshot::Neural {
            genome, state, rng, ..
        } => {
            if reset_lifetime_to_genome
                && matches!(state.as_ref(), NeuralModelState::External { .. })
            {
                return Err(MidasError::Policy(
                    "external neural state has no genome-parameter baseline".into(),
                ));
            }
            let genome_parameters = reset_lifetime_to_genome
                .then(|| Box::new(LifetimeParameters::from_genome(&genome)));
            let state = match *state {
                NeuralModelState::Burn {
                    parameters,
                    mut runtime,
                    optimizer,
                } => {
                    runtime.memory.fill(0.0);
                    runtime.inference_steps = 0;
                    NeuralModelState::Burn {
                        parameters: genome_parameters.clone().unwrap_or(parameters),
                        runtime,
                        optimizer,
                    }
                }
                NeuralModelState::BurnTrainableEncoder {
                    parameters,
                    mut runtime,
                    optimizer,
                } => {
                    runtime.memory.fill(0.0);
                    runtime.inference_steps = 0;
                    NeuralModelState::BurnTrainableEncoder {
                        parameters: genome_parameters.clone().unwrap_or(parameters),
                        runtime,
                        optimizer,
                    }
                }
                NeuralModelState::BurnBptt {
                    parameters,
                    mut runtime,
                    optimizer,
                } => {
                    runtime.memory.fill(0.0);
                    runtime.inference_steps = 0;
                    NeuralModelState::BurnBptt {
                        parameters: genome_parameters.clone().unwrap_or(parameters),
                        runtime,
                        optimizer,
                    }
                }
                NeuralModelState::BurnAdamW {
                    parameters,
                    mut runtime,
                    optimizer,
                } => {
                    runtime.memory.fill(0.0);
                    runtime.inference_steps = 0;
                    NeuralModelState::BurnAdamW {
                        parameters: genome_parameters.clone().unwrap_or(parameters),
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
                rng: evaluation_seed.map_or(rng, ChaCha8Rng::seed_from_u64),
                maximum_trade_fraction,
            })
        }
        other => Ok(other),
    }
}
