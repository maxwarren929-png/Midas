use crate::{MidasError, Result};
use chrono::NaiveDate;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DateRange {
    pub start: NaiveDate,
    pub end: NaiveDate,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SplitConfig {
    pub training: DateRange,
    pub validation: DateRange,
    pub test: DateRange,
}

impl SplitConfig {
    pub(super) fn validate(&self) -> Result<()> {
        let bad = |message: &str| Err(MidasError::Config(message.into()));
        for range in [&self.training, &self.validation, &self.test] {
            if range.start > range.end {
                return bad("split start must not follow split end");
            }
        }
        if self.training.end >= self.validation.start || self.validation.end >= self.test.start {
            return bad("splits must be chronological and non-overlapping");
        }
        Ok(())
    }
}
