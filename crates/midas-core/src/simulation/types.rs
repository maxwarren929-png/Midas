use crate::config::{Config, DateRange};
use serde::{Deserialize, Serialize};

// Version 11 replaces target allocations with persistent incremental trade
// impulses. Version 12 records the modular observation-registry schema and can
// be migrated because built-in replacement strategies had no persisted state.
// Earlier checkpoints contain incompatible action samples or state layouts.
/// Format 21 adds serialized AdamW first/second moments for explicitly opted-in
/// runs. Formats 12 through 20 remain readable.
pub const CHECKPOINT_VERSION: u32 = 21;
/// Format 22 combines AdamW-capable neural state with persistent agent
/// self-awareness statistics.
pub const AWARENESS_CHECKPOINT_VERSION: u32 = 22;

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
