use crate::config::{Config, DateRange};
use serde::{Deserialize, Serialize};

// Version 11 replaces target allocations with persistent incremental trade
// impulses and persists their diagnostics in the checkpointed metrics state.
// Older checkpoints contain incompatible action samples or state layouts and
// cannot be migrated safely.
/// Format 12 records the modular observation-registry schema identity.
pub const CHECKPOINT_VERSION: u32 = 12;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum DatasetSplit {
    Training,
    Validation,
    Test,
}

impl DatasetSplit {
    pub fn range<'a>(&self, c: &'a Config) -> &'a DateRange {
        match self {
            Self::Training => &c.splits.training,
            Self::Validation => &c.splits.validation,
            Self::Test => &c.splits.test,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum SimulationLifecycle {
    Initialized,
    Running,
    Completed,
    Failed { reason: String },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum DeathCause {
    Bankruptcy,
    PolicyFailure,
}
