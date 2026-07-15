use crate::{
    Result,
    config::{Config, FeatureKind, RollingMarketStatsConfig},
    market::MarketDataset,
};
use std::{collections::BTreeMap, sync::Mutex};

use super::{
    contracts::{Observation, ObservationContext, ObservationMetadata, ObservationSink},
    normalization::{NormalizationRule, normalize},
    providers::built_in_registry,
    registry::ObservationRegistry,
    schema::ObservationSchema,
};

#[cfg(test)]
use super::schema::{AgentStateFeatureSpec, ObservationFeatureSpec, PortfolioFeatureSpec};

#[derive(Default)]
struct ObservationScratch {
    values: Vec<Vec<f64>>,
    present: Vec<bool>,
}

impl ObservationScratch {
    fn new(metadata: &[&ObservationMetadata]) -> Self {
        Self {
            values: metadata
                .iter()
                .map(|item| Vec::with_capacity(item.shape.dimension()))
                .collect(),
            present: vec![false; metadata.len()],
        }
    }

    fn reset(&mut self) {
        self.present.fill(false);
    }
}

#[derive(Clone)]
struct CompiledEntry {
    index: usize,
    slot: usize,
    component: usize,
    normalization: NormalizationRule,
    name: String,
}

pub trait ObservationBuilder: Send + Sync {
    fn schema(&self) -> &ObservationSchema;
    fn build(&self, context: &ObservationContext<'_>) -> Result<Observation>;
}

pub struct SchemaObservationBuilder {
    schema: ObservationSchema,
    registry: ObservationRegistry,
    entries: Vec<CompiledEntry>,
    slots: BTreeMap<String, usize>,
    dimensions: Vec<usize>,
    asset_count: usize,
    schema_hash: String,
    scratch: Mutex<ObservationScratch>,
}

impl SchemaObservationBuilder {
    pub fn new(
        schema: ObservationSchema,
        market_features: Vec<FeatureKind>,
        asset_count: usize,
    ) -> Result<Self> {
        Self::from_parts(
            schema,
            market_features,
            asset_count,
            &RollingMarketStatsConfig::default(),
            None,
        )
    }

    /// Constructs built-in providers from typed configuration. Market-derived
    /// sensors are precomputed from immutable raw data, while the default
    /// Prompt 5 provider set remains allocation-free at construction.
    pub fn from_config(
        schema: ObservationSchema,
        config: &Config,
        market_data: Option<&MarketDataset>,
    ) -> Result<Self> {
        Self::from_parts(
            schema,
            config.features.features.clone(),
            config.market.assets.len(),
            &config.observations.rolling_market_stats,
            market_data,
        )
    }

    fn from_parts(
        schema: ObservationSchema,
        market_features: Vec<FeatureKind>,
        asset_count: usize,
        rolling_config: &RollingMarketStatsConfig,
        market_data: Option<&MarketDataset>,
    ) -> Result<Self> {
        if market_features.is_empty() {
            return Err(crate::MidasError::Config(
                "observation market feature list must not be empty".into(),
            ));
        }
        let registry = built_in_registry(
            &schema,
            &market_features,
            asset_count,
            rolling_config,
            market_data,
        )?;
        Self::with_registry(schema, registry, asset_count)
    }

    pub fn with_registry(
        schema: ObservationSchema,
        registry: ObservationRegistry,
        asset_count: usize,
    ) -> Result<Self> {
        registry.validate_schema(&schema, asset_count)?;
        let metadata = registry.metadata();
        let slots = metadata
            .iter()
            .enumerate()
            .map(|(slot, item)| (item.name.clone(), slot))
            .collect::<BTreeMap<_, _>>();
        let dimensions = metadata
            .iter()
            .map(|item| item.shape.dimension())
            .collect::<Vec<_>>();
        let scratch = ObservationScratch::new(&metadata);
        drop(metadata);
        let mut entries = schema
            .entries
            .iter()
            .map(|entry| {
                Ok(CompiledEntry {
                    index: entry.index,
                    slot: *slots.get(&entry.name).ok_or_else(|| {
                        crate::MidasError::Config(format!(
                            "observation {} is missing from registry",
                            entry.name
                        ))
                    })?,
                    component: entry.component,
                    normalization: entry.normalization.clone(),
                    name: entry.name.clone(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        entries.sort_by_key(|entry| entry.index);
        let schema_hash = schema.identity(asset_count)?;
        Ok(Self {
            schema,
            registry,
            entries,
            slots,
            dimensions,
            asset_count,
            schema_hash,
            scratch: Mutex::new(scratch),
        })
    }

    pub fn registry(&self) -> &ObservationRegistry {
        &self.registry
    }
}

impl ObservationBuilder for SchemaObservationBuilder {
    fn schema(&self) -> &ObservationSchema {
        &self.schema
    }

    fn build(&self, context: &ObservationContext<'_>) -> Result<Observation> {
        validate_context(context, self.asset_count)?;
        let mut scratch = self.scratch.lock().map_err(|_| {
            crate::MidasError::Policy("observation scratch buffer lock is poisoned".into())
        })?;
        scratch.reset();
        {
            let ObservationScratch { values, present } = &mut *scratch;
            let mut sink = ObservationSink {
                slots: &self.slots,
                dimensions: &self.dimensions,
                values,
                present,
            };
            for provider in self.registry.providers() {
                provider.observe(context, &mut sink)?;
            }
        }
        let mut values = vec![0.0_f32; self.entries.len()];
        for entry in &self.entries {
            if !scratch.present[entry.slot] {
                return Err(crate::MidasError::Policy(format!(
                    "observation {} was not published",
                    entry.name
                )));
            }
            let raw = scratch.values[entry.slot]
                .get(entry.component)
                .copied()
                .ok_or_else(|| {
                    crate::MidasError::Policy(format!(
                        "observation {} is missing component {}",
                        entry.name, entry.component
                    ))
                })?;
            values[entry.index] = normalize(raw, &entry.normalization)?;
        }
        if values.len() != self.schema.dimension(self.asset_count)
            || values.iter().any(|value| !value.is_finite())
        {
            return Err(crate::MidasError::Policy(
                "observation registry produced an invalid vector".into(),
            ));
        }
        Ok(Observation {
            schema_hash: self.schema_hash.clone(),
            values,
        })
    }
}

fn validate_context(context: &ObservationContext<'_>, asset_count: usize) -> Result<()> {
    if context.market.tradable.len() != asset_count
        || context.portfolio.positions.len() != asset_count
        || context.prices.len() != asset_count
        || !context.starting_net_worth.is_finite()
        || context.starting_net_worth <= 0.0
        || !context.maximum_net_worth.is_finite()
        || context.maximum_net_worth <= 0.0
    {
        return Err(crate::MidasError::Policy(
            "invalid observation context dimensions or bounds".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn legacy_default_values(
    schema: &ObservationSchema,
    market_features: &[FeatureKind],
    asset_count: usize,
    context: &ObservationContext<'_>,
) -> Result<Vec<f32>> {
    validate_context(context, asset_count)?;
    if context.market.values.len() != asset_count * market_features.len() {
        return Err(crate::MidasError::Policy(
            "invalid legacy feature dimension".into(),
        ));
    }
    let allocations = context.portfolio.allocations(context.prices)?;
    let mut values = Vec::with_capacity(schema.entries.len());
    for spec in &schema.features {
        match spec {
            ObservationFeatureSpec::Market {
                feature,
                normalization,
            } => {
                let feature_index = market_features
                    .iter()
                    .position(|candidate| candidate == feature)
                    .ok_or_else(|| crate::MidasError::Policy("missing legacy feature".into()))?;
                for asset in 0..asset_count {
                    values.push(normalize(
                        context.market.values[asset * market_features.len() + feature_index] as f64,
                        normalization,
                    )?);
                }
            }
            ObservationFeatureSpec::Portfolio {
                feature,
                normalization,
            } => match feature {
                PortfolioFeatureSpec::CashAllocation => {
                    values.push(normalize(allocations[0], normalization)?)
                }
                PortfolioFeatureSpec::AssetAllocation => {
                    for value in allocations.iter().skip(1) {
                        values.push(normalize(*value, normalization)?);
                    }
                }
                PortfolioFeatureSpec::UnrealizedReturn => {
                    for (position, price) in context.portfolio.positions.iter().zip(context.prices)
                    {
                        let value = if position.quantity > 0.0 && position.average_entry_price > 0.0
                        {
                            price.unwrap_or(position.average_entry_price)
                                / position.average_entry_price
                                - 1.0
                        } else {
                            0.0
                        };
                        values.push(normalize(value, normalization)?);
                    }
                }
                PortfolioFeatureSpec::NetWorthToStarting => values.push(normalize(
                    context.portfolio.net_worth / context.starting_net_worth,
                    normalization,
                )?),
                PortfolioFeatureSpec::CurrentDrawdown => values.push(normalize(
                    1.0 - context.portfolio.net_worth / context.maximum_net_worth,
                    normalization,
                )?),
                PortfolioFeatureSpec::PortfolioReturn1 => values.push(normalize(
                    context.portfolio.net_worth / context.portfolio.previous_net_worth.max(1e-9)
                        - 1.0,
                    normalization,
                )?),
                PortfolioFeatureSpec::TurnoverToNetWorth => values.push(normalize(
                    context.portfolio.cumulative_turnover / context.portfolio.net_worth.max(1e-9),
                    normalization,
                )?),
            },
            ObservationFeatureSpec::AgentState {
                feature: AgentStateFeatureSpec::NormalizedAge,
                normalization,
            } => {
                values.push(normalize(
                    context.age as f64 / context.episode_steps.max(1) as f64,
                    normalization,
                )?);
            }
        }
    }
    Ok(values)
}
