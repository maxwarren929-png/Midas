use crate::{
    MidasError, Result,
    action::LongOnlyActionValidator,
    agent::{Agent, AgentBuildContext, AgentFactory, StandardAgentFactory},
    brain::{BrainArchitecture, ComputeDevice, PolicySnapshot},
    config::Config,
    evolution::EvolutionState,
    execution::CostExecutionEngine,
    market::{BackwardFeaturePipeline, FeaturePipeline, FeatureRow, MarketDataset},
    metrics::MetricsCollector,
    observation::{ObservationSchema, SchemaObservationBuilder},
    replacement::{NonEvolutionaryRandomGenomeReplacement, ReplacementStrategy},
    reward::configured_reward,
};
use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

use super::{
    CHECKPOINT_VERSION, DatasetSplit, DomainEvent, OperationalEvent, SimulationEngine,
    SimulationLifecycle,
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
    lifecycle: SimulationLifecycle,
) -> Result<Checkpoint> {
    Ok(Checkpoint {
        version: CHECKPOINT_VERSION,
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
    let checkpoint: Checkpoint = bincode::deserialize(&bytes).map_err(|error| {
        MidasError::Checkpoint(format!(
            "incompatible checkpoint; expected format version {CHECKPOINT_VERSION}; older files use an incompatible neural action-sample layout: {error}"
        ))
    })?;
    validate_resume_version(&checkpoint)?;
    Ok(checkpoint)
}

pub(super) fn load_for_evaluation(path: impl AsRef<Path>) -> Result<Checkpoint> {
    let bytes = fs::read(path)?;
    let checkpoint: Checkpoint = bincode::deserialize(&bytes).map_err(|error| {
        MidasError::Checkpoint(format!(
            "incompatible checkpoint; expected format version {CHECKPOINT_VERSION}; older files use an incompatible neural action-sample or metrics layout: {error}"
        ))
    })?;
    validate_evaluation_version(&checkpoint)?;
    Ok(checkpoint)
}

fn validate_resume_version(checkpoint: &Checkpoint) -> Result<()> {
    if checkpoint.version != CHECKPOINT_VERSION {
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
    if checkpoint.version != CHECKPOINT_VERSION {
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
        let cp = load_for_resume(path)?;
        validate_dataset(&cp, &data.identity)?;
        data.rebuild_index();
        let pipeline = BackwardFeaturePipeline::from_config(&config);
        let dates = data.split_dates(cp.split.range(&config));
        let features = pipeline.generate(&data, &dates)?;
        let observation_schema = ObservationSchema::from_config(&config);
        let schema_hash = observation_schema.identity(data.assets.len())?;
        validate_schema(&cp, &schema_hash)?;
        let compute_device = config.backend.compute_device()?;
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
}
