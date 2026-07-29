use super::{Config, ExecutionConfig, SimulationConfig};
use crate::{MidasError, Result};

impl Config {
    pub fn validate(&self) -> Result<()> {
        self.market.validate()?;
        self.simulation.validate_population()?;
        validate_economic_values(&self.simulation, &self.execution)?;
        self.simulation.validate_cash_threshold()?;
        self.execution.validate_rate_bounds()?;
        self.features.validate()?;
        self.brain.validate()?;
        self.observation.validate()?;
        self.observations.validate()?;
        validate_observation_schema(self)?;
        self.corporate_actions.validate()?;
        self.policy_failures.validate()?;
        self.splits.validate()?;
        self.learning.validate()?;
        self.reward.validate()?;
        self.evolution.validate()?;
        crate::brain::ComputeDevice::from_options(self.backend.compute_device_options())?;
        Ok(())
    }
}

fn validate_economic_values(
    simulation: &SimulationConfig,
    execution: &ExecutionConfig,
) -> Result<()> {
    for (name, value) in [
        ("starting_cash", simulation.starting_cash),
        ("bankruptcy_threshold", simulation.bankruptcy_threshold),
        ("daily_survival_cost", simulation.daily_survival_cost),
        ("minimum_trade_value", simulation.minimum_trade_value),
        ("commission_fixed", execution.commission_fixed),
        ("commission_rate", execution.commission_rate),
        ("spread_rate", execution.spread_rate),
        ("slippage_rate", execution.slippage_rate),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(MidasError::Config(format!(
                "{name} must be finite and non-negative"
            )));
        }
    }
    Ok(())
}

fn validate_observation_schema(config: &Config) -> Result<()> {
    let observation_schema = crate::observation::ObservationSchema::try_from_options(
        config.observation_schema_options(),
    )?;
    let used_observation_providers = observation_schema
        .entries
        .iter()
        .map(|entry| entry.provider.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    let configured_observation_providers = config
        .observations
        .providers
        .iter()
        .map(String::as_str)
        .collect::<std::collections::BTreeSet<_>>();
    if used_observation_providers != configured_observation_providers {
        return Err(MidasError::Config(
            "configured observation providers must each have schema entries".into(),
        ));
    }
    Ok(())
}
