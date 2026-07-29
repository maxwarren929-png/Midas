use crate::{MidasError, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionConfig {
    pub commission_fixed: f64,
    pub commission_rate: f64,
    pub spread_rate: f64,
    pub slippage_rate: f64,
}

impl ExecutionConfig {
    pub(super) fn validate_rate_bounds(&self) -> Result<()> {
        let bad = |message: &str| Err(MidasError::Config(message.into()));
        if self.spread_rate >= 2.0 {
            return bad("spread_rate must be less than 2");
        }
        if self.slippage_rate >= 1.0 {
            return bad("slippage_rate must be less than 1");
        }
        if self.spread_rate / 2.0 + self.slippage_rate >= 1.0 {
            return bad("spread_rate / 2 + slippage_rate must be less than 1");
        }
        Ok(())
    }
}
