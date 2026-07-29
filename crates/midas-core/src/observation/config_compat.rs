use crate::{Result, config::Config, market::MarketDataset};

use super::{ObservationSchema, SchemaObservationBuilder};

impl ObservationSchema {
    pub fn from_config(config: &Config) -> Self {
        Self::from_options(config.observation_schema_options())
    }

    pub fn try_from_config(config: &Config) -> Result<Self> {
        Self::try_from_options(config.observation_schema_options())
    }
}

impl SchemaObservationBuilder {
    /// Constructs built-in providers from typed configuration. Market-derived
    /// sensors are precomputed from immutable raw data, while the default
    /// Prompt 5 provider set remains allocation-free at construction.
    pub fn from_config(
        schema: ObservationSchema,
        config: &Config,
        market_data: Option<&MarketDataset>,
    ) -> Result<Self> {
        Self::from_options(schema, config.observation_builder_options(), market_data)
    }
}
