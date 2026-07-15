use crate::{
    MidasError, Result,
    agent::Agent,
    brain::{BrainArchitecture, ComputeDevice, PolicySnapshot},
    config::Config,
    evolution::EvolutionState,
    market::{FeatureRow, MarketDataset},
    metrics::MetricsCollector,
    observation::ObservationSchema,
};
use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

use super::{CHECKPOINT_VERSION, DatasetSplit, DomainEvent, SimulationLifecycle};

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
