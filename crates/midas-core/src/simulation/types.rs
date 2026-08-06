use crate::config::{Config, DateRange};
use serde::{Deserialize, Serialize};

// Version 11 replaces target allocations with persistent incremental trade
// impulses. Version 12 records the modular observation-registry schema and can
// be migrated because built-in replacement strategies had no persisted state.
// Earlier checkpoints contain incompatible action samples or state layouts.
/// Format 13 records replacement-strategy identity, version, and state.
pub const CHECKPOINT_VERSION: u32 = 13;
/// Format 14 adds persistent agent self-awareness statistics. Baseline
/// configurations continue to write format 13 for compatibility.
pub const AWARENESS_CHECKPOINT_VERSION: u32 = 14;

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
