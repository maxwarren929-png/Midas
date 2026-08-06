mod agent_awareness;
mod agent_state;
mod market;
mod portfolio;
pub(crate) mod rolling_market_stats;

use crate::{
    Result,
    config::{FeatureKind, RollingMarketStatsConfig},
    market::MarketDataset,
};

use self::{
    agent_awareness::AgentAwarenessProvider, agent_state::AgentStateObservationProvider,
    market::MarketObservationProvider, portfolio::PortfolioObservationProvider,
    rolling_market_stats::RollingMarketStatsProvider,
};
use super::{
    contracts::{ObservationMetadata, ObservationShape, ObservationValueType},
    registry::ObservationRegistry,
    schema::{
        AGENT_AWARENESS_PROVIDER, AGENT_STATE_PROVIDER, MARKET_PROVIDER, ObservationSchema,
        PORTFOLIO_PROVIDER, ROLLING_MARKET_STATS_PROVIDER,
    },
};

pub(crate) fn built_in_registry(
    schema: &ObservationSchema,
    market_features: &[FeatureKind],
    asset_count: usize,
    rolling_config: &RollingMarketStatsConfig,
    market_data: Option<&MarketDataset>,
) -> Result<ObservationRegistry> {
    let mut registry = ObservationRegistry::new();
    for provider in &schema.providers {
        let used = schema
            .entries
            .iter()
            .any(|entry| entry.provider == *provider);
        match provider.as_str() {
            MARKET_PROVIDER if used => registry.register(Box::new(
                MarketObservationProvider::new(schema, market_features, asset_count)?,
            ))?,
            PORTFOLIO_PROVIDER if used => registry.register(Box::new(
                PortfolioObservationProvider::new(schema, asset_count)?,
            ))?,
            AGENT_STATE_PROVIDER if used => {
                registry.register(Box::new(AgentStateObservationProvider::new(schema)?))?
            }
            AGENT_AWARENESS_PROVIDER if used => {
                registry.register(Box::new(AgentAwarenessProvider::new(schema)?))?
            }
            ROLLING_MARKET_STATS_PROVIDER if used => registry.register(Box::new(
                RollingMarketStatsProvider::new(schema, asset_count, rolling_config, market_data)?,
            ))?,
            MARKET_PROVIDER
            | PORTFOLIO_PROVIDER
            | AGENT_STATE_PROVIDER
            | AGENT_AWARENESS_PROVIDER
            | ROLLING_MARKET_STATS_PROVIDER => {}
            name => {
                return Err(crate::MidasError::Config(format!(
                    "configured observation provider {name} is not registered"
                )));
            }
        }
    }
    registry.validate_schema(schema, asset_count)?;
    Ok(registry)
}

pub(crate) fn metadata_for_entries(
    schema: &ObservationSchema,
    provider: &str,
    description_prefix: &str,
    units: &str,
) -> Vec<ObservationMetadata> {
    schema
        .entries
        .iter()
        .filter(|entry| entry.provider == provider)
        .map(|entry| ObservationMetadata {
            name: entry.name.clone(),
            provider: provider.into(),
            description: format!("{description_prefix}: {}", entry.name),
            units: units.into(),
            value_type: ObservationValueType::Float64,
            shape: ObservationShape::Scalar,
            ordering_hint: Some(entry.index),
            normalization: entry.normalization.clone(),
            version: schema.version,
            tags: vec![provider.into()],
        })
        .collect()
}
