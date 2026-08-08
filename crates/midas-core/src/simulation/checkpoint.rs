use crate::{
    MidasError, Result,
    action::LongOnlyActionValidator,
    agent::{Agent, AgentBuildContext, AgentFactory, StandardAgentFactory},
    brain::{BrainArchitecture, ComputeDevice, PolicySnapshot},
    config::Config,
    evolution::{EvolutionState, GenomeId, LineageId},
    execution::CostExecutionEngine,
    market::{BackwardFeaturePipeline, FeaturePipeline, FeatureRow, MarketDataset},
    metrics::MetricsCollector,
    observation::{ObservationSchema, SchemaObservationBuilder},
    replacement::{
        BuiltInReplacementStrategyFactory, NonEvolutionaryRandomGenomeReplacement,
        ReplacementStrategy, ReplacementStrategyFactory, ReplacementStrategyState,
    },
    reward::configured_reward,
};
use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

use super::{
    AWARENESS_CHECKPOINT_VERSION, CHECKPOINT_VERSION, DatasetSplit, DomainEvent, OperationalEvent,
    SimulationEngine, SimulationLifecycle,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct AgentCheckpoint {
    pub(super) state: crate::agent::AgentState,
    pub(super) policy: PolicySnapshot,
    pub(super) learning: crate::learning::LearningStrategyState,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct Checkpoint {
    pub(super) version: u32,
    pub(super) config_hash: String,
    pub(super) dataset_identity: String,
    pub(super) observation_schema_hash: String,
    pub(super) architecture: BrainArchitecture,
    pub(super) split: DatasetSplit,
    pub(super) next_step: usize,
    pub(super) next_agent_id: u64,
    pub(super) lifecycle: SimulationLifecycle,
    pub(super) agents: Vec<AgentCheckpoint>,
    pub(super) metrics: MetricsCollector,
    pub(super) domain_events: Vec<DomainEvent>,
    pub(super) accessed_through: Option<NaiveDate>,
    pub(super) backend_identity: String,
    pub(super) evolution: EvolutionState,
    pub(super) replacement: Option<ReplacementStrategyState>,
}

/// Agent state as it appeared in format 12/13. Keep this layout separate from
/// the current state so old bincode checkpoints can be migrated without
/// relying on serde defaults, which bincode does not use for missing tuple
/// fields.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct AgentStateLegacy {
    id: crate::AgentId,
    birth_date: NaiveDate,
    age: u64,
    portfolio: crate::execution::Portfolio,
    maximum_net_worth: f64,
    maximum_drawdown: f64,
    alive: bool,
    genome: Option<crate::brain::BrainGenome>,
    #[serde(default)]
    genome_id: Option<GenomeId>,
    #[serde(default)]
    lineage_id: Option<LineageId>,
    parent_ids: Vec<crate::AgentId>,
    total_policy_failures: u32,
    consecutive_policy_failures: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct AgentCheckpointLegacy {
    state: AgentStateLegacy,
    policy: PolicySnapshot,
    learning: crate::learning::LearningStrategyState,
}

impl From<AgentStateLegacy> for crate::agent::AgentState {
    fn from(state: AgentStateLegacy) -> Self {
        Self {
            id: state.id,
            birth_date: state.birth_date,
            age: state.age,
            portfolio: state.portfolio,
            maximum_net_worth: state.maximum_net_worth,
            maximum_drawdown: state.maximum_drawdown,
            alive: state.alive,
            genome: state.genome,
            genome_id: state.genome_id,
            lineage_id: state.lineage_id,
            parent_ids: state.parent_ids,
            total_policy_failures: state.total_policy_failures,
            consecutive_policy_failures: state.consecutive_policy_failures,
            reward_ema: 0.0,
            reward_variance_ema: 0.0,
            last_reward: 0.0,
            short_reward_ema: 0.0,
            long_reward_ema: 0.0,
            reward_trend: 0.0,
        }
    }
}

impl From<AgentCheckpointLegacy> for AgentCheckpoint {
    fn from(checkpoint: AgentCheckpointLegacy) -> Self {
        Self {
            state: checkpoint.state.into(),
            policy: checkpoint.policy,
            learning: checkpoint.learning,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CheckpointV12Legacy {
    version: u32,
    config_hash: String,
    dataset_identity: String,
    observation_schema_hash: String,
    architecture: BrainArchitecture,
    split: DatasetSplit,
    next_step: usize,
    next_agent_id: u64,
    lifecycle: SimulationLifecycle,
    agents: Vec<AgentCheckpointLegacy>,
    metrics: MetricsCollector,
    domain_events: Vec<DomainEvent>,
    accessed_through: Option<NaiveDate>,
    backend_identity: String,
    evolution: EvolutionState,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CheckpointV13Legacy {
    version: u32,
    config_hash: String,
    dataset_identity: String,
    observation_schema_hash: String,
    architecture: BrainArchitecture,
    split: DatasetSplit,
    next_step: usize,
    next_agent_id: u64,
    lifecycle: SimulationLifecycle,
    agents: Vec<AgentCheckpointLegacy>,
    metrics: MetricsCollector,
    domain_events: Vec<DomainEvent>,
    accessed_through: Option<NaiveDate>,
    backend_identity: String,
    evolution: EvolutionState,
    replacement: Option<ReplacementStrategyState>,
}

fn migrate_agents(agents: Vec<AgentCheckpointLegacy>) -> Vec<AgentCheckpoint> {
    agents.into_iter().map(AgentCheckpoint::from).collect()
}

impl From<CheckpointV12Legacy> for Checkpoint {
    fn from(checkpoint: CheckpointV12Legacy) -> Self {
        Self {
            version: checkpoint.version,
            config_hash: checkpoint.config_hash,
            dataset_identity: checkpoint.dataset_identity,
            observation_schema_hash: checkpoint.observation_schema_hash,
            architecture: checkpoint.architecture,
            split: checkpoint.split,
            next_step: checkpoint.next_step,
            next_agent_id: checkpoint.next_agent_id,
            lifecycle: checkpoint.lifecycle,
            agents: migrate_agents(checkpoint.agents),
            metrics: checkpoint.metrics,
            domain_events: checkpoint.domain_events,
            accessed_through: checkpoint.accessed_through,
            backend_identity: checkpoint.backend_identity,
            evolution: checkpoint.evolution,
            replacement: None,
        }
    }
}

impl From<CheckpointV13Legacy> for Checkpoint {
    fn from(checkpoint: CheckpointV13Legacy) -> Self {
        Self {
            version: checkpoint.version,
            config_hash: checkpoint.config_hash,
            dataset_identity: checkpoint.dataset_identity,
            observation_schema_hash: checkpoint.observation_schema_hash,
            architecture: checkpoint.architecture,
            split: checkpoint.split,
            next_step: checkpoint.next_step,
            next_agent_id: checkpoint.next_agent_id,
            lifecycle: checkpoint.lifecycle,
            agents: migrate_agents(checkpoint.agents),
            metrics: checkpoint.metrics,
            domain_events: checkpoint.domain_events,
            accessed_through: checkpoint.accessed_through,
            backend_identity: checkpoint.backend_identity,
            evolution: checkpoint.evolution,
            replacement: checkpoint.replacement,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn from_state(
    config: &Config,
    data: &MarketDataset,
    observation_schema: &ObservationSchema,
    architecture: &BrainArchitecture,
    split: DatasetSplit,
    features: &[FeatureRow],
    compute_device: &ComputeDevice,
    next_step: usize,
    next_agent_id: u64,
    agents: &[Agent],
    metrics: MetricsCollector,
    domain_events: Vec<DomainEvent>,
    evolution: EvolutionState,
    replacement: &dyn ReplacementStrategy,
    lifecycle: SimulationLifecycle,
) -> Result<Checkpoint> {
    let replacement = replacement.checkpoint_state()?;
    replacement.validate()?;
    Ok(Checkpoint {
        version: if observation_schema
            .providers
            .iter()
            .any(|provider| provider == crate::observation::AGENT_AWARENESS_PROVIDER)
        {
            AWARENESS_CHECKPOINT_VERSION
        } else {
            CHECKPOINT_VERSION
        },
        config_hash: config.hash()?,
        dataset_identity: data.identity.clone(),
        observation_schema_hash: observation_schema.identity(data.assets.len())?,
        architecture: architecture.clone(),
        split,
        next_step,
        next_agent_id,
        lifecycle,
        agents: agents
            .iter()
            .map(|agent| {
                Ok(AgentCheckpoint {
                    state: agent.state.clone(),
                    policy: agent.policy.snapshot()?,
                    learning: agent.learning_snapshot()?,
                })
            })
            .collect::<Result<_>>()?,
        metrics,
        domain_events,
        accessed_through: features.get(next_step).map(|feature| feature.date),
        backend_identity: compute_device.identity(),
        evolution,
        replacement: Some(replacement),
    })
}

pub(super) fn write(path: impl AsRef<Path>, checkpoint: &Checkpoint) -> Result<()> {
    if let Some(parent) = path.as_ref().parent() {
        fs::create_dir_all(parent)?
    }
    fs::write(path.as_ref(), bincode::serialize(checkpoint)?)?;
    Ok(())
}

pub(super) fn load_for_resume(path: impl AsRef<Path>) -> Result<Checkpoint> {
    let bytes = fs::read(path)?;
    let checkpoint = deserialize_checkpoint(&bytes, "resume")?;
    validate_resume_version(&checkpoint)?;
    Ok(checkpoint)
}

pub(super) fn load_for_evaluation(path: impl AsRef<Path>) -> Result<Checkpoint> {
    let bytes = fs::read(path)?;
    let checkpoint = deserialize_checkpoint(&bytes, "evaluation")?;
    validate_evaluation_version(&checkpoint)?;
    Ok(checkpoint)
}

fn deserialize_checkpoint(bytes: &[u8], operation: &str) -> Result<Checkpoint> {
    let version = bytes
        .get(..std::mem::size_of::<u32>())
        .and_then(|bytes| bytes.try_into().ok())
        .map(u32::from_le_bytes)
        .ok_or_else(|| {
            MidasError::Checkpoint(format!(
                "incompatible checkpoint for {operation}; missing format version"
            ))
        })?;
    if version == 12 {
        return bincode::deserialize::<CheckpointV12Legacy>(bytes)
            .map(Checkpoint::from)
            .map_err(|error| {
                MidasError::Checkpoint(format!(
                    "incompatible version-12 checkpoint for {operation}: {error}"
                ))
            });
    }
    if version == 13 {
        return match bincode::deserialize::<Checkpoint>(bytes) {
            Ok(checkpoint) => Ok(checkpoint),
            Err(current_error) => bincode::deserialize::<CheckpointV13Legacy>(bytes)
                .map(Checkpoint::from)
                .map_err(|legacy_error| {
                    MidasError::Checkpoint(format!(
                        "incompatible version-13 checkpoint for {operation}: {current_error}; legacy decode failed: {legacy_error}"
                    ))
                }),
        };
    }
    bincode::deserialize::<Checkpoint>(bytes).map_err(|error| {
        MidasError::Checkpoint(format!(
            "incompatible checkpoint for {operation}; expected format versions {CHECKPOINT_VERSION} or {AWARENESS_CHECKPOINT_VERSION}: {error}"
        ))
    })
}

fn validate_resume_version(checkpoint: &Checkpoint) -> Result<()> {
    if !matches!(
        checkpoint.version,
        CHECKPOINT_VERSION | AWARENESS_CHECKPOINT_VERSION | 12 | 13 | 14 | 15 | 16 | 17 | 18
    ) {
        let detail = if checkpoint.version == 11 {
            "; checkpoint 11 predates the modular observation registry and cannot be migrated safely"
        } else if checkpoint.version == 6 {
            "; checkpoint 6 contains bias-only optimizer state and cannot be migrated safely"
        } else if checkpoint.version == 8 {
            "; checkpoint 8 contains single-asset categorical action samples and cannot be migrated safely"
        } else {
            ""
        };
        return Err(MidasError::Checkpoint(format!(
            "checkpoint version {} is incompatible with {CHECKPOINT_VERSION}{detail}",
            checkpoint.version,
        )));
    }
    Ok(())
}

fn validate_evaluation_version(checkpoint: &Checkpoint) -> Result<()> {
    if !matches!(
        checkpoint.version,
        CHECKPOINT_VERSION | AWARENESS_CHECKPOINT_VERSION | 12 | 13 | 14 | 15 | 16 | 17 | 18
    ) {
        return Err(MidasError::Checkpoint(format!(
            "checkpoint version {} is incompatible with {CHECKPOINT_VERSION}; observation-registry and neural action distribution layouts are not migrated",
            checkpoint.version
        )));
    }
    Ok(())
}

pub(super) fn validate_dataset(checkpoint: &Checkpoint, identity: &str) -> Result<()> {
    if checkpoint.dataset_identity != identity {
        return Err(MidasError::Checkpoint("dataset identity mismatch".into()));
    }
    Ok(())
}

pub(super) fn validate_schema(checkpoint: &Checkpoint, schema_hash: &str) -> Result<()> {
    if checkpoint.observation_schema_hash != schema_hash {
        return Err(MidasError::Checkpoint("observation schema mismatch".into()));
    }
    Ok(())
}

pub(super) fn validate_backend(checkpoint: &Checkpoint, identity: &str) -> Result<()> {
    if checkpoint.backend_identity != identity {
        return Err(MidasError::Checkpoint(
            "checkpoint backend is incompatible".into(),
        ));
    }
    Ok(())
}

pub(super) fn validate_resume_config(checkpoint: &Checkpoint, hash: &str) -> Result<()> {
    if checkpoint.config_hash != hash {
        return Err(MidasError::Checkpoint("configuration mismatch".into()));
    }
    Ok(())
}

pub(super) fn validate_evaluation_config(checkpoint: &Checkpoint, hash: &str) -> Result<()> {
    if checkpoint.config_hash != hash {
        return Err(MidasError::Checkpoint(
            "evaluation configuration mismatch".into(),
        ));
    }
    Ok(())
}

pub(super) fn validate_architecture(
    checkpoint: &Checkpoint,
    architecture: &BrainArchitecture,
) -> Result<()> {
    if checkpoint.architecture != *architecture {
        return Err(MidasError::Checkpoint("brain architecture mismatch".into()));
    }
    Ok(())
}

pub(super) fn validate_feature_count(count: usize) -> Result<()> {
    if count < 2 {
        return Err(MidasError::Checkpoint(
            "checkpoint split needs at least two trading dates".into(),
        ));
    }
    Ok(())
}

pub(super) fn restore_replacement(
    checkpoint: &Checkpoint,
    config: &Config,
    factory: &dyn ReplacementStrategyFactory,
) -> Result<Box<dyn ReplacementStrategy>> {
    let Some(state) = checkpoint.replacement.as_ref() else {
        if checkpoint.version == 12 {
            return Ok(configured_replacement(config));
        }
        return Err(MidasError::Checkpoint(
            "checkpoint replacement strategy state is missing".into(),
        ));
    };
    state.validate()?;
    let replacement = factory.restore(state)?;
    let restored_state = replacement.checkpoint_state()?;
    if restored_state != *state {
        return Err(MidasError::Checkpoint(format!(
            "restored replacement strategy state does not match checkpoint for {}",
            state.identity
        )));
    }
    Ok(replacement)
}

fn configured_replacement(config: &Config) -> Box<dyn ReplacementStrategy> {
    if config.evolution.enabled {
        Box::new(crate::replacement::EvolutionaryReplacementStrategy::default())
    } else {
        Box::new(NonEvolutionaryRandomGenomeReplacement)
    }
}

impl SimulationEngine {
    pub fn save_checkpoint(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let checkpoint = from_state(
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
            self.replacement.as_ref(),
            self.lifecycle.clone(),
        )?;
        write(path.as_ref(), &checkpoint)?;
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
        Self::resume_with_factories(
            config,
            data,
            path,
            Box::new(StandardAgentFactory::default()),
            Box::new(BuiltInReplacementStrategyFactory),
        )
    }
    pub fn resume_with_factory(
        config: Config,
        data: MarketDataset,
        path: impl AsRef<Path>,
        factory: Box<dyn AgentFactory>,
    ) -> Result<Self> {
        Self::resume_with_factories(
            config,
            data,
            path,
            factory,
            Box::new(BuiltInReplacementStrategyFactory),
        )
    }

    pub fn resume_with_replacement_factory(
        config: Config,
        data: MarketDataset,
        path: impl AsRef<Path>,
        replacement_factory: Box<dyn ReplacementStrategyFactory>,
    ) -> Result<Self> {
        Self::resume_with_factories(
            config,
            data,
            path,
            Box::new(StandardAgentFactory::default()),
            replacement_factory,
        )
    }

    pub fn resume_with_factories(
        config: Config,
        mut data: MarketDataset,
        path: impl AsRef<Path>,
        factory: Box<dyn AgentFactory>,
        replacement_factory: Box<dyn ReplacementStrategyFactory>,
    ) -> Result<Self> {
        let cp = load_for_resume(path)?;
        validate_dataset(&cp, &data.identity)?;
        data.rebuild_index();
        let pipeline = BackwardFeaturePipeline::from_config(&config);
        let dates = data.split_dates(cp.split.range(&config));
        let features = pipeline.generate(&data, &dates)?;
        let observation_schema =
            ObservationSchema::from_options(config.observation_schema_options());
        let schema_hash = observation_schema.identity(data.assets.len())?;
        validate_schema(&cp, &schema_hash)?;
        let compute_device = ComputeDevice::from_options(config.backend.compute_device_options())?;
        validate_backend(&cp, &compute_device.identity())?;
        validate_resume_config(&cp, &config.hash()?)?;
        let architecture = BrainArchitecture::from_config(
            &config.brain,
            schema_hash,
            observation_schema.dimension(data.assets.len()),
            data.assets.len(),
        );
        validate_architecture(&cp, &architecture)?;
        validate_feature_count(features.len())?;
        let replacement = restore_replacement(&cp, &config, replacement_factory.as_ref())?;
        let observation_builder = SchemaObservationBuilder::from_options(
            observation_schema.clone(),
            config.observation_builder_options(),
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
}
