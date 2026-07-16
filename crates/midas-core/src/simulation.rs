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
pub use types::{CHECKPOINT_VERSION, DatasetSplit, DeathCause, SimulationLifecycle};
