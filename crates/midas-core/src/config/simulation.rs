use crate::{MidasError, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SimulationConfig {
    pub population_size: usize,
    pub starting_cash: f64,
    pub bankruptcy_threshold: f64,
    pub daily_survival_cost: f64,
    pub minimum_trade_value: f64,
    pub seed: u64,
    #[serde(default)]
    pub forced_liquidation: bool,
    #[serde(default)]
    pub death_handling: DeathHandling,
}

impl SimulationConfig {
    pub(super) fn validate_population(&self) -> Result<()> {
        if self.population_size == 0 {
            return Err(MidasError::Config(
                "population_size must be positive".into(),
            ));
        }
        Ok(())
    }

    pub(super) fn validate_cash_threshold(&self) -> Result<()> {
        if self.starting_cash <= self.bankruptcy_threshold {
            return Err(MidasError::Config(
                "starting_cash must exceed bankruptcy_threshold".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeathHandling {
    #[default]
    Replace,
    LeaveVacant,
    EndAgentOnly,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyFailureBehavior {
    Continue,
    #[default]
    KillAgent,
    StopExperiment,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyFailureConfig {
    pub maximum_consecutive: u32,
    pub maximum_total: u32,
    pub behavior: PolicyFailureBehavior,
}
impl Default for PolicyFailureConfig {
    fn default() -> Self {
        Self {
            maximum_consecutive: 3,
            maximum_total: 20,
            behavior: PolicyFailureBehavior::KillAgent,
        }
    }
}

impl PolicyFailureConfig {
    pub(super) fn validate(&self) -> Result<()> {
        if self.maximum_consecutive == 0 || self.maximum_total == 0 {
            return Err(MidasError::Config(
                "policy failure limits must be positive".into(),
            ));
        }
        Ok(())
    }
}
