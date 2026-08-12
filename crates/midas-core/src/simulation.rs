pub use crate::PolicyFailureCategory;

mod advance;
mod checkpoint;
mod death_replacement;
mod decision;
mod diagnostics;
mod engine;
mod evaluation;
mod events;
mod learning_bridge;
mod report;
mod runner;
mod transaction;
mod types;

pub use diagnostics::{DecisionTrace, PpoUpdateTrace, ReportTrace};
pub use engine::SimulationEngine;
pub use events::{DomainEvent, OperationalEvent};
pub use report::SimulationReport;
pub use types::{
    AWARENESS_CHECKPOINT_VERSION, CHECKPOINT_VERSION, CONSTRAINED_PPO_CHECKPOINT_VERSION,
    DatasetSplit, DeathCause, SimulationLifecycle,
};
