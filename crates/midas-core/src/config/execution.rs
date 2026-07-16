use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionConfig {
    pub commission_fixed: f64,
    pub commission_rate: f64,
    pub spread_rate: f64,
    pub slippage_rate: f64,
}
