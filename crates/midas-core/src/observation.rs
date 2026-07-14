//! Versioned, named observations assembled from read-only sensor providers.
//!
//! Providers publish raw named values.  [`ObservationSchema`] owns the only
//! mapping from those names to neural-input indices and applies normalisation.
//! This keeps feature sources independent from policy, learning, and execution.

use crate::{
    Result,
    config::{
        Config, FeatureKind, ObservationFeatureKind, ObservationNormalizationConfig,
        ObservationNormalizationKind,
    },
    execution::Portfolio,
    market::FeatureRow,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Mutex,
};

const MARKET_PROVIDER: &str = "market";
const PORTFOLIO_PROVIDER: &str = "portfolio";
const AGENT_STATE_PROVIDER: &str = "agent_state";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum NormalizationRule {
    None,
    /// Historical Prompt 4 behaviour: clamp an already ratio-like value.
    RatioClipped {
        minimum: f64,
        maximum: f64,
    },
    MinMax {
        minimum: f64,
        maximum: f64,
    },
    ZScore {
        mean: f64,
        standard_deviation: f64,
    },
    Log,
    SignedLog,
    Percentage {
        scale: f64,
    },
    /// An explicit finite affine transform for research-only schema entries.
    Custom {
        scale: f64,
        offset: f64,
    },
}

impl NormalizationRule {
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::None | Self::Log | Self::SignedLog => Ok(()),
            Self::RatioClipped { minimum, maximum } | Self::MinMax { minimum, maximum } => {
                if minimum.is_finite() && maximum.is_finite() && minimum < maximum {
                    Ok(())
                } else {
                    Err(crate::MidasError::Config(
                        "observation normalization bounds are invalid".into(),
                    ))
                }
            }
            Self::ZScore {
                mean,
                standard_deviation,
            } => {
                if mean.is_finite() && standard_deviation.is_finite() && *standard_deviation > 0.0 {
                    Ok(())
                } else {
                    Err(crate::MidasError::Config(
                        "observation z-score normalization is invalid".into(),
                    ))
                }
            }
            Self::Percentage { scale } => {
                if scale.is_finite() && *scale != 0.0 {
                    Ok(())
                } else {
                    Err(crate::MidasError::Config(
                        "observation percentage normalization scale is invalid".into(),
                    ))
                }
            }
            Self::Custom { scale, offset } => {
                if scale.is_finite() && offset.is_finite() {
                    Ok(())
                } else {
                    Err(crate::MidasError::Config(
                        "observation custom normalization is invalid".into(),
                    ))
                }
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum PortfolioFeatureSpec {
    CashAllocation,
    AssetAllocation,
    UnrealizedReturn,
    NetWorthToStarting,
    CurrentDrawdown,
    PortfolioReturn1,
    TurnoverToNetWorth,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum AgentStateFeatureSpec {
    NormalizedAge,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ObservationFeatureSpec {
    Market {
        feature: FeatureKind,
        normalization: NormalizationRule,
    },
    Portfolio {
        feature: PortfolioFeatureSpec,
        normalization: NormalizationRule,
    },
    AgentState {
        feature: AgentStateFeatureSpec,
        normalization: NormalizationRule,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ObservationSource {
    Market {
        feature: FeatureKind,
        asset_index: usize,
    },
    Portfolio {
        feature: PortfolioFeatureSpec,
        asset_index: Option<usize>,
    },
    AgentState {
        feature: AgentStateFeatureSpec,
    },
}

impl ObservationSource {
    fn provider(&self) -> &'static str {
        match self {
            Self::Market { .. } => MARKET_PROVIDER,
            Self::Portfolio { .. } => PORTFOLIO_PROVIDER,
            Self::AgentState { .. } => AGENT_STATE_PROVIDER,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ObservationSchemaEntry {
    pub index: usize,
    pub name: String,
    pub provider: String,
    pub source: Option<ObservationSource>,
    pub component: usize,
    pub normalization: NormalizationRule,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ObservationSchema {
    pub version: u32,
    pub market_schema_version: u32,
    /// Kept as a compact, human-readable source specification.
    pub features: Vec<ObservationFeatureSpec>,
    /// The authoritative flattened neural-input mapping.
    pub entries: Vec<ObservationSchemaEntry>,
    /// Required provider names. Their registration order has no semantic effect.
    pub providers: Vec<String>,
}

impl ObservationSchema {
    pub fn from_config(config: &Config) -> Self {
        Self::try_from_config(config).expect("configuration was validated before schema creation")
    }

    pub fn try_from_config(config: &Config) -> Result<Self> {
        let mut features = Vec::new();
        for configured in &config.observation.features {
            match configured {
                ObservationFeatureKind::Market => {
                    for feature in &config.features.features {
                        features.push(ObservationFeatureSpec::Market {
                            feature: *feature,
                            normalization: NormalizationRule::RatioClipped {
                                minimum: -config.features.clip,
                                maximum: config.features.clip,
                            },
                        });
                    }
                }
                ObservationFeatureKind::CashAllocation => {
                    features.push(portfolio(PortfolioFeatureSpec::CashAllocation, 0.0, 1.0))
                }
                ObservationFeatureKind::AssetAllocation => {
                    features.push(portfolio(PortfolioFeatureSpec::AssetAllocation, 0.0, 1.0))
                }
                ObservationFeatureKind::UnrealizedReturn => features.push(portfolio(
                    PortfolioFeatureSpec::UnrealizedReturn,
                    -10.0,
                    10.0,
                )),
                ObservationFeatureKind::NetWorthToStarting => features.push(portfolio(
                    PortfolioFeatureSpec::NetWorthToStarting,
                    -10.0,
                    10.0,
                )),
                ObservationFeatureKind::CurrentDrawdown => {
                    features.push(portfolio(PortfolioFeatureSpec::CurrentDrawdown, 0.0, 1.0))
                }
                ObservationFeatureKind::PortfolioReturn1 => features.push(portfolio(
                    PortfolioFeatureSpec::PortfolioReturn1,
                    -10.0,
                    10.0,
                )),
                ObservationFeatureKind::TurnoverToNetWorth => features.push(portfolio(
                    PortfolioFeatureSpec::TurnoverToNetWorth,
                    0.0,
                    10.0,
                )),
                ObservationFeatureKind::NormalizedAge => {
                    features.push(ObservationFeatureSpec::AgentState {
                        feature: AgentStateFeatureSpec::NormalizedAge,
                        normalization: NormalizationRule::RatioClipped {
                            minimum: 0.0,
                            maximum: 1.0,
                        },
                    })
                }
            }
        }
        let mut entries = Vec::new();
        for spec in &features {
            match spec {
                ObservationFeatureSpec::Market {
                    feature,
                    normalization,
                } => {
                    for asset_index in 0..config.market.assets.len() {
                        push_entry(
                            &mut entries,
                            market_name(*feature, asset_index),
                            ObservationSource::Market {
                                feature: *feature,
                                asset_index,
                            },
                            apply_override(
                                normalization,
                                &market_name(*feature, asset_index),
                                &config.observations.normalizations,
                            )?,
                        );
                    }
                }
                ObservationFeatureSpec::Portfolio {
                    feature,
                    normalization,
                } => match feature {
                    PortfolioFeatureSpec::AssetAllocation
                    | PortfolioFeatureSpec::UnrealizedReturn => {
                        for asset_index in 0..config.market.assets.len() {
                            let name = portfolio_asset_name(feature, asset_index);
                            push_entry(
                                &mut entries,
                                name.clone(),
                                ObservationSource::Portfolio {
                                    feature: feature.clone(),
                                    asset_index: Some(asset_index),
                                },
                                apply_override(
                                    normalization,
                                    &name,
                                    &config.observations.normalizations,
                                )?,
                            );
                        }
                    }
                    _ => {
                        let name = portfolio_scalar_name(feature);
                        push_entry(
                            &mut entries,
                            name.clone(),
                            ObservationSource::Portfolio {
                                feature: feature.clone(),
                                asset_index: None,
                            },
                            apply_override(
                                normalization,
                                &name,
                                &config.observations.normalizations,
                            )?,
                        );
                    }
                },
                ObservationFeatureSpec::AgentState {
                    feature,
                    normalization,
                } => {
                    let name = agent_state_name(feature);
                    push_entry(
                        &mut entries,
                        name.clone(),
                        ObservationSource::AgentState {
                            feature: feature.clone(),
                        },
                        apply_override(normalization, &name, &config.observations.normalizations)?,
                    );
                }
            }
        }
        let schema = Self {
            version: config.observation.version,
            market_schema_version: config.features.schema_version,
            features,
            entries,
            providers: config.observations.providers.clone(),
        };
        schema.validate(config.market.assets.len())?;
        for name in config.observations.normalizations.keys() {
            if !schema.entries.iter().any(|entry| &entry.name == name) {
                return Err(crate::MidasError::Config(format!(
                    "observation normalization references unknown observation {name}"
                )));
            }
        }
        Ok(schema)
    }

    pub fn dimension(&self, _asset_count: usize) -> usize {
        self.entries.len()
    }

    pub fn validate(&self, asset_count: usize) -> Result<()> {
        if self.version == 0 || self.market_schema_version == 0 || asset_count == 0 {
            return Err(crate::MidasError::Config(
                "observation schema dimensions and versions must be positive".into(),
            ));
        }
        if self.entries.is_empty() || self.providers.is_empty() {
            return Err(crate::MidasError::Config(
                "observation schema entries and providers must be non-empty".into(),
            ));
        }
        let mut names = BTreeSet::new();
        let mut indices = BTreeSet::new();
        let configured_providers: BTreeSet<_> = self.providers.iter().map(String::as_str).collect();
        if configured_providers.len() != self.providers.len()
            || configured_providers
                .iter()
                .any(|name| name.trim().is_empty())
        {
            return Err(crate::MidasError::Config(
                "observation providers must be unique non-empty names".into(),
            ));
        }
        for entry in &self.entries {
            if entry.name.trim().is_empty()
                || !names.insert(entry.name.as_str())
                || !indices.insert(entry.index)
                || entry.index >= self.entries.len()
            {
                return Err(crate::MidasError::Config(
                    "observation schema contains duplicate or invalid entries".into(),
                ));
            }
            if !configured_providers.contains(entry.provider.as_str()) {
                return Err(crate::MidasError::Config(format!(
                    "observation {} requires missing provider {}",
                    entry.name, entry.provider
                )));
            }
            if entry
                .source
                .as_ref()
                .is_some_and(|source| source.provider() != entry.provider)
            {
                return Err(crate::MidasError::Config(format!(
                    "observation {} source and provider disagree",
                    entry.name
                )));
            }
            entry.normalization.validate()?;
        }
        if indices.iter().copied().collect::<Vec<_>>()
            != (0..self.entries.len()).collect::<Vec<_>>()
        {
            return Err(crate::MidasError::Config(
                "observation schema indices must be contiguous from zero".into(),
            ));
        }
        Ok(())
    }

    pub fn identity(&self, asset_count: usize) -> Result<String> {
        self.validate(asset_count)?;
        Ok(blake3::hash(&serde_json::to_vec(&(self, asset_count))?)
            .to_hex()
            .to_string())
    }
}

fn push_entry(
    entries: &mut Vec<ObservationSchemaEntry>,
    name: String,
    source: ObservationSource,
    normalization: NormalizationRule,
) {
    entries.push(ObservationSchemaEntry {
        index: entries.len(),
        provider: source.provider().into(),
        name,
        source: Some(source),
        component: 0,
        normalization,
    });
}

fn market_name(feature: FeatureKind, asset_index: usize) -> String {
    format!("market.{}.asset_{asset_index}", feature_name(feature))
}

fn portfolio_asset_name(feature: &PortfolioFeatureSpec, asset_index: usize) -> String {
    format!(
        "portfolio.{}.asset_{asset_index}",
        portfolio_feature_name(feature)
    )
}

fn portfolio_scalar_name(feature: &PortfolioFeatureSpec) -> String {
    format!("portfolio.{}", portfolio_feature_name(feature))
}

fn agent_state_name(feature: &AgentStateFeatureSpec) -> String {
    match feature {
        AgentStateFeatureSpec::NormalizedAge => "agent.normalized_age".into(),
    }
}

fn feature_name(feature: FeatureKind) -> &'static str {
    match feature {
        FeatureKind::Return1 => "return1",
        FeatureKind::Return5 => "return5",
        FeatureKind::Return20 => "return20",
        FeatureKind::PriceToMa20 => "price_to_ma20",
        FeatureKind::PriceToMa50 => "price_to_ma50",
        FeatureKind::Volatility20 => "volatility20",
        FeatureKind::VolumeChange => "volume_change",
        FeatureKind::VolumeToMa20 => "volume_to_ma20",
        FeatureKind::DistanceHigh20 => "distance_high20",
        FeatureKind::Tradable => "tradable",
    }
}

fn portfolio_feature_name(feature: &PortfolioFeatureSpec) -> &'static str {
    match feature {
        PortfolioFeatureSpec::CashAllocation => "cash_fraction",
        PortfolioFeatureSpec::AssetAllocation => "allocation_fraction",
        PortfolioFeatureSpec::UnrealizedReturn => "unrealized_return",
        PortfolioFeatureSpec::NetWorthToStarting => "net_worth_to_starting",
        PortfolioFeatureSpec::CurrentDrawdown => "current_drawdown",
        PortfolioFeatureSpec::PortfolioReturn1 => "portfolio_return1",
        PortfolioFeatureSpec::TurnoverToNetWorth => "turnover_to_net_worth",
    }
}

fn portfolio(feature: PortfolioFeatureSpec, minimum: f64, maximum: f64) -> ObservationFeatureSpec {
    ObservationFeatureSpec::Portfolio {
        feature,
        normalization: NormalizationRule::RatioClipped { minimum, maximum },
    }
}

fn apply_override(
    default: &NormalizationRule,
    name: &str,
    overrides: &BTreeMap<String, ObservationNormalizationConfig>,
) -> Result<NormalizationRule> {
    match overrides.get(name) {
        Some(config) => normalization_from_config(config),
        None => Ok(default.clone()),
    }
}

fn normalization_from_config(config: &ObservationNormalizationConfig) -> Result<NormalizationRule> {
    config.validate()?;
    let missing = |field: &str| {
        crate::MidasError::Config(format!("observation normalization is missing {field}"))
    };
    Ok(match config.kind {
        ObservationNormalizationKind::None => NormalizationRule::None,
        ObservationNormalizationKind::RatioClipped => NormalizationRule::RatioClipped {
            minimum: config.minimum.ok_or_else(|| missing("minimum"))?,
            maximum: config.maximum.ok_or_else(|| missing("maximum"))?,
        },
        ObservationNormalizationKind::MinMax => NormalizationRule::MinMax {
            minimum: config.minimum.ok_or_else(|| missing("minimum"))?,
            maximum: config.maximum.ok_or_else(|| missing("maximum"))?,
        },
        ObservationNormalizationKind::ZScore => NormalizationRule::ZScore {
            mean: config.mean.ok_or_else(|| missing("mean"))?,
            standard_deviation: config
                .standard_deviation
                .ok_or_else(|| missing("standard_deviation"))?,
        },
        ObservationNormalizationKind::Log => NormalizationRule::Log,
        ObservationNormalizationKind::SignedLog => NormalizationRule::SignedLog,
        ObservationNormalizationKind::Percentage => NormalizationRule::Percentage {
            scale: config.scale.ok_or_else(|| missing("scale"))?,
        },
        ObservationNormalizationKind::Custom => NormalizationRule::Custom {
            scale: config.scale.ok_or_else(|| missing("scale"))?,
            offset: config.offset.ok_or_else(|| missing("offset"))?,
        },
    })
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Observation {
    pub schema_hash: String,
    pub values: Vec<f32>,
}

/// Immutable data available to providers at close `t`.  It deliberately has
/// no reference to policies, learning, execution planners, or mutable state.
pub struct ObservationContext<'a> {
    pub market: &'a FeatureRow,
    pub portfolio: &'a Portfolio,
    pub prices: &'a [Option<f64>],
    pub starting_net_worth: f64,
    pub maximum_net_worth: f64,
    pub age: u64,
    pub episode_steps: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ObservationShape {
    Scalar,
    Fixed { dimensions: usize },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ObservationValueType {
    Float64,
}

impl ObservationShape {
    fn dimension(&self) -> usize {
        match self {
            Self::Scalar => 1,
            Self::Fixed { dimensions } => *dimensions,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ObservationMetadata {
    pub name: String,
    pub provider: String,
    pub description: String,
    pub units: String,
    pub value_type: ObservationValueType,
    pub shape: ObservationShape,
    /// A provider may advertise a stable source-order hint. The schema index,
    /// not this hint, is always authoritative for neural input order.
    pub ordering_hint: Option<usize>,
    pub normalization: NormalizationRule,
    pub version: u32,
    pub tags: Vec<String>,
}

impl ObservationMetadata {
    fn validate(&self) -> Result<()> {
        if self.name.trim().is_empty()
            || self.provider.trim().is_empty()
            || self.description.trim().is_empty()
            || self.units.trim().is_empty()
            || self.version == 0
            || self.shape.dimension() == 0
        {
            return Err(crate::MidasError::Config(
                "observation metadata is incomplete".into(),
            ));
        }
        self.normalization.validate()
    }
}

/// A sensor provider.  Providers publish raw named values to a registry-owned
/// sink and cannot write the neural vector directly.
pub trait ObservationProvider: Send + Sync {
    fn provider_name(&self) -> &str;
    fn metadata(&self) -> Vec<ObservationMetadata>;
    fn observe(
        &self,
        context: &ObservationContext<'_>,
        sink: &mut ObservationSink<'_>,
    ) -> Result<()>;
}

pub struct ObservationSink<'a> {
    slots: &'a BTreeMap<String, usize>,
    dimensions: &'a [usize],
    values: &'a mut [Vec<f64>],
    present: &'a mut [bool],
}

impl ObservationSink<'_> {
    pub fn publish_scalar(&mut self, name: &str, value: f64) -> Result<()> {
        self.publish(name, &[value])
    }

    pub fn publish(&mut self, name: &str, values: &[f64]) -> Result<()> {
        let slot = self.slots.get(name).copied().ok_or_else(|| {
            crate::MidasError::Policy(format!(
                "provider published unregistered observation {name}"
            ))
        })?;
        if self.present[slot] {
            return Err(crate::MidasError::Policy(format!(
                "observation {name} was published more than once"
            )));
        }
        if values.len() != self.dimensions[slot] || values.iter().any(|value| !value.is_finite()) {
            return Err(crate::MidasError::Policy(format!(
                "observation {name} has invalid dimensions or non-finite values"
            )));
        }
        self.values[slot].clear();
        self.values[slot].extend_from_slice(values);
        self.present[slot] = true;
        Ok(())
    }
}

pub struct ObservationRegistry {
    providers: BTreeMap<String, Box<dyn ObservationProvider>>,
    metadata: BTreeMap<String, ObservationMetadata>,
}

impl ObservationRegistry {
    pub fn new() -> Self {
        Self {
            providers: BTreeMap::new(),
            metadata: BTreeMap::new(),
        }
    }

    pub fn register(&mut self, provider: Box<dyn ObservationProvider>) -> Result<()> {
        let provider_name = provider.provider_name().to_owned();
        if provider_name.trim().is_empty() || self.providers.contains_key(&provider_name) {
            return Err(crate::MidasError::Config(format!(
                "duplicate or empty observation provider {provider_name}"
            )));
        }
        let metadata = provider.metadata();
        if metadata.is_empty() {
            return Err(crate::MidasError::Config(format!(
                "observation provider {provider_name} does not publish observations"
            )));
        }
        for item in &metadata {
            item.validate()?;
            if item.provider != provider_name || self.metadata.contains_key(&item.name) {
                return Err(crate::MidasError::Config(format!(
                    "duplicate observation name {}",
                    item.name
                )));
            }
        }
        for item in metadata {
            self.metadata.insert(item.name.clone(), item);
        }
        self.providers.insert(provider_name, provider);
        Ok(())
    }

    pub fn metadata(&self) -> Vec<&ObservationMetadata> {
        self.metadata.values().collect()
    }

    pub fn metadata_by_name(&self, name: &str) -> Option<&ObservationMetadata> {
        self.metadata.get(name)
    }

    pub fn validate_schema(&self, schema: &ObservationSchema, asset_count: usize) -> Result<()> {
        schema.validate(asset_count)?;
        let expected_providers: BTreeSet<_> = schema
            .entries
            .iter()
            .map(|entry| entry.provider.as_str())
            .collect();
        let registered_providers: BTreeSet<_> = self.providers.keys().map(String::as_str).collect();
        if expected_providers != registered_providers {
            return Err(crate::MidasError::Config(
                "configured observation providers do not match registry".into(),
            ));
        }
        let schema_names: BTreeSet<_> = schema
            .entries
            .iter()
            .map(|entry| entry.name.as_str())
            .collect();
        let metadata_names: BTreeSet<_> = self.metadata.keys().map(String::as_str).collect();
        if schema_names != metadata_names {
            let missing = schema_names.difference(&metadata_names).next().copied();
            let unused = metadata_names.difference(&schema_names).next().copied();
            return Err(crate::MidasError::Config(format!(
                "observation schema and registry differ; missing={missing:?}, unused={unused:?}"
            )));
        }
        for entry in &schema.entries {
            let metadata = self
                .metadata
                .get(&entry.name)
                .expect("set equality checked");
            if metadata.provider != entry.provider || metadata.shape.dimension() <= entry.component
            {
                return Err(crate::MidasError::Config(format!(
                    "observation metadata does not match schema entry {}",
                    entry.name
                )));
            }
            if metadata
                .ordering_hint
                .is_some_and(|index| index != entry.index)
            {
                return Err(crate::MidasError::Config(format!(
                    "observation metadata ordering hint disagrees with schema entry {}",
                    entry.name
                )));
            }
        }
        Ok(())
    }

    fn built_in(
        schema: &ObservationSchema,
        market_features: &[FeatureKind],
        asset_count: usize,
    ) -> Result<Self> {
        let mut registry = Self::new();
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
                MARKET_PROVIDER | PORTFOLIO_PROVIDER | AGENT_STATE_PROVIDER => {}
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
}

impl Default for ObservationRegistry {
    fn default() -> Self {
        Self::new()
    }
}

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
        if market_features.is_empty() {
            return Err(crate::MidasError::Config(
                "observation market feature list must not be empty".into(),
            ));
        }
        let registry = ObservationRegistry::built_in(&schema, &market_features, asset_count)?;
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
            for provider in self.registry.providers.values() {
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

struct MarketObservationProvider {
    metadata: Vec<ObservationMetadata>,
    outputs: Vec<(String, FeatureKind, usize)>,
    market_features: Vec<FeatureKind>,
    asset_count: usize,
}

impl MarketObservationProvider {
    fn new(
        schema: &ObservationSchema,
        market_features: &[FeatureKind],
        asset_count: usize,
    ) -> Result<Self> {
        let mut outputs = Vec::new();
        for entry in schema
            .entries
            .iter()
            .filter(|entry| entry.provider == MARKET_PROVIDER)
        {
            let Some(ObservationSource::Market {
                feature,
                asset_index,
            }) = entry.source.as_ref()
            else {
                return Err(crate::MidasError::Config(format!(
                    "market observation {} has no market source",
                    entry.name
                )));
            };
            if *asset_index >= asset_count || !market_features.contains(feature) {
                return Err(crate::MidasError::Config(format!(
                    "market observation {} is unavailable",
                    entry.name
                )));
            }
            outputs.push((entry.name.clone(), *feature, *asset_index));
        }
        Ok(Self {
            metadata: metadata_for_entries(schema, MARKET_PROVIDER, "market feature", "ratio"),
            outputs,
            market_features: market_features.to_vec(),
            asset_count,
        })
    }
}

impl ObservationProvider for MarketObservationProvider {
    fn provider_name(&self) -> &str {
        MARKET_PROVIDER
    }

    fn metadata(&self) -> Vec<ObservationMetadata> {
        self.metadata.clone()
    }

    fn observe(
        &self,
        context: &ObservationContext<'_>,
        sink: &mut ObservationSink<'_>,
    ) -> Result<()> {
        if context.market.values.len() != self.asset_count * self.market_features.len() {
            return Err(crate::MidasError::Policy(
                "market observation values have an invalid dimension".into(),
            ));
        }
        for (name, feature, asset_index) in &self.outputs {
            let feature_index = self
                .market_features
                .iter()
                .position(|candidate| candidate == feature)
                .ok_or_else(|| {
                    crate::MidasError::Policy(format!(
                        "market feature {feature:?} is unavailable for observation"
                    ))
                })?;
            sink.publish_scalar(
                name,
                context.market.values[asset_index * self.market_features.len() + feature_index]
                    as f64,
            )?;
        }
        Ok(())
    }
}

struct PortfolioObservationProvider {
    metadata: Vec<ObservationMetadata>,
    outputs: Vec<(String, PortfolioFeatureSpec, Option<usize>)>,
}

impl PortfolioObservationProvider {
    fn new(schema: &ObservationSchema, asset_count: usize) -> Result<Self> {
        let mut outputs = Vec::new();
        for entry in schema
            .entries
            .iter()
            .filter(|entry| entry.provider == PORTFOLIO_PROVIDER)
        {
            let Some(ObservationSource::Portfolio {
                feature,
                asset_index,
            }) = entry.source.as_ref()
            else {
                return Err(crate::MidasError::Config(format!(
                    "portfolio observation {} has no portfolio source",
                    entry.name
                )));
            };
            if asset_index.is_some_and(|index| index >= asset_count) {
                return Err(crate::MidasError::Config(format!(
                    "portfolio observation {} has invalid asset index",
                    entry.name
                )));
            }
            outputs.push((entry.name.clone(), feature.clone(), *asset_index));
        }
        Ok(Self {
            metadata: metadata_for_entries(schema, PORTFOLIO_PROVIDER, "portfolio state", "ratio"),
            outputs,
        })
    }
}

impl ObservationProvider for PortfolioObservationProvider {
    fn provider_name(&self) -> &str {
        PORTFOLIO_PROVIDER
    }

    fn metadata(&self) -> Vec<ObservationMetadata> {
        self.metadata.clone()
    }

    fn observe(
        &self,
        context: &ObservationContext<'_>,
        sink: &mut ObservationSink<'_>,
    ) -> Result<()> {
        let allocations = context.portfolio.allocations(context.prices)?;
        for (name, feature, asset_index) in &self.outputs {
            let value = match (feature, asset_index) {
                (PortfolioFeatureSpec::CashAllocation, None) => allocations[0],
                (PortfolioFeatureSpec::AssetAllocation, Some(index)) => allocations[index + 1],
                (PortfolioFeatureSpec::UnrealizedReturn, Some(index)) => {
                    let position = &context.portfolio.positions[*index];
                    if position.quantity > 0.0 && position.average_entry_price > 0.0 {
                        context.prices[*index].unwrap_or(position.average_entry_price)
                            / position.average_entry_price
                            - 1.0
                    } else {
                        0.0
                    }
                }
                (PortfolioFeatureSpec::NetWorthToStarting, None) => {
                    context.portfolio.net_worth / context.starting_net_worth
                }
                (PortfolioFeatureSpec::CurrentDrawdown, None) => {
                    1.0 - context.portfolio.net_worth / context.maximum_net_worth
                }
                (PortfolioFeatureSpec::PortfolioReturn1, None) => {
                    context.portfolio.net_worth / context.portfolio.previous_net_worth.max(1e-9)
                        - 1.0
                }
                (PortfolioFeatureSpec::TurnoverToNetWorth, None) => {
                    context.portfolio.cumulative_turnover / context.portfolio.net_worth.max(1e-9)
                }
                _ => {
                    return Err(crate::MidasError::Policy(format!(
                        "portfolio observation {name} has incompatible source"
                    )));
                }
            };
            sink.publish_scalar(name, value)?;
        }
        Ok(())
    }
}

struct AgentStateObservationProvider {
    metadata: Vec<ObservationMetadata>,
    outputs: Vec<(String, AgentStateFeatureSpec)>,
}

impl AgentStateObservationProvider {
    fn new(schema: &ObservationSchema) -> Result<Self> {
        let mut outputs = Vec::new();
        for entry in schema
            .entries
            .iter()
            .filter(|entry| entry.provider == AGENT_STATE_PROVIDER)
        {
            let Some(ObservationSource::AgentState { feature }) = entry.source.as_ref() else {
                return Err(crate::MidasError::Config(format!(
                    "agent-state observation {} has no agent-state source",
                    entry.name
                )));
            };
            outputs.push((entry.name.clone(), feature.clone()));
        }
        Ok(Self {
            metadata: metadata_for_entries(schema, AGENT_STATE_PROVIDER, "agent state", "ratio"),
            outputs,
        })
    }
}

impl ObservationProvider for AgentStateObservationProvider {
    fn provider_name(&self) -> &str {
        AGENT_STATE_PROVIDER
    }

    fn metadata(&self) -> Vec<ObservationMetadata> {
        self.metadata.clone()
    }

    fn observe(
        &self,
        context: &ObservationContext<'_>,
        sink: &mut ObservationSink<'_>,
    ) -> Result<()> {
        for (name, feature) in &self.outputs {
            let value = match feature {
                AgentStateFeatureSpec::NormalizedAge => {
                    context.age as f64 / context.episode_steps.max(1) as f64
                }
            };
            sink.publish_scalar(name, value)?;
        }
        Ok(())
    }
}

fn metadata_for_entries(
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

fn normalize(value: f64, rule: &NormalizationRule) -> Result<f32> {
    if !value.is_finite() {
        return Err(crate::MidasError::Policy(
            "observation feature is non-finite".into(),
        ));
    }
    let normalized = match rule {
        NormalizationRule::None => value,
        NormalizationRule::RatioClipped { minimum, maximum } => value.clamp(*minimum, *maximum),
        NormalizationRule::MinMax { minimum, maximum } => {
            ((value - minimum) / (maximum - minimum)).clamp(0.0, 1.0)
        }
        NormalizationRule::ZScore {
            mean,
            standard_deviation,
        } => (value - mean) / standard_deviation,
        NormalizationRule::Log => {
            if value <= 0.0 {
                return Err(crate::MidasError::Policy(
                    "log-normalized observation must be positive".into(),
                ));
            }
            value.ln()
        }
        NormalizationRule::SignedLog => value.signum() * value.abs().ln_1p(),
        NormalizationRule::Percentage { scale } => value * scale,
        NormalizationRule::Custom { scale, offset } => value * scale + offset,
    };
    let normalized = normalized as f32;
    if normalized.is_finite() {
        Ok(normalized)
    } else {
        Err(crate::MidasError::Policy(
            "observation feature conversion is non-finite".into(),
        ))
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market::FeatureRow;
    use std::path::PathBuf;

    struct StaticProvider {
        name: String,
        metadata: Vec<ObservationMetadata>,
        values: Vec<(String, Vec<f64>)>,
    }

    impl ObservationProvider for StaticProvider {
        fn provider_name(&self) -> &str {
            &self.name
        }

        fn metadata(&self) -> Vec<ObservationMetadata> {
            self.metadata.clone()
        }

        fn observe(
            &self,
            _context: &ObservationContext<'_>,
            sink: &mut ObservationSink<'_>,
        ) -> Result<()> {
            for (name, values) in &self.values {
                sink.publish(name, values)?;
            }
            Ok(())
        }
    }

    fn metadata(name: &str, provider: &str, shape: ObservationShape) -> ObservationMetadata {
        ObservationMetadata {
            name: name.into(),
            provider: provider.into(),
            description: format!("test observation {name}"),
            units: "unit".into(),
            value_type: ObservationValueType::Float64,
            shape,
            ordering_hint: None,
            normalization: NormalizationRule::None,
            version: 1,
            tags: vec!["test".into()],
        }
    }

    fn context() -> (FeatureRow, Portfolio, Vec<Option<f64>>) {
        (
            FeatureRow {
                date: chrono::NaiveDate::from_ymd_opt(2020, 1, 1).unwrap(),
                values: vec![0.0],
                tradable: vec![true],
            },
            Portfolio::new(100.0, 1),
            vec![Some(10.0)],
        )
    }

    fn custom_schema(
        entries: Vec<ObservationSchemaEntry>,
        providers: Vec<&str>,
    ) -> ObservationSchema {
        ObservationSchema {
            version: 1,
            market_schema_version: 1,
            features: Vec::new(),
            entries,
            providers: providers.into_iter().map(str::to_owned).collect(),
        }
    }

    #[test]
    fn modular_default_builder_matches_prompt4_positional_vector() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let config = Config::load(root.join("configs/test.toml")).unwrap();
        let schema = ObservationSchema::from_config(&config);
        let builder = SchemaObservationBuilder::new(
            schema.clone(),
            config.features.features.clone(),
            config.market.assets.len(),
        )
        .unwrap();
        let mut portfolio = Portfolio::new(25.0, config.market.assets.len());
        portfolio.positions[0].quantity = 2.0;
        portfolio.positions[0].average_entry_price = 10.0;
        portfolio.net_worth = 90.0;
        portfolio.previous_net_worth = 80.0;
        portfolio.cumulative_turnover = 12.0;
        let market = FeatureRow {
            date: chrono::NaiveDate::from_ymd_opt(2020, 1, 1).unwrap(),
            values: (0..config.market.assets.len() * config.features.features.len())
                .map(|index| index as f32 / 10.0 - 1.0)
                .collect(),
            tradable: vec![true; config.market.assets.len()],
        };
        let prices = vec![Some(15.0); config.market.assets.len()];
        let context = ObservationContext {
            market: &market,
            portfolio: &portfolio,
            prices: &prices,
            starting_net_worth: 100.0,
            maximum_net_worth: 120.0,
            age: 3,
            episode_steps: 12,
        };
        let expected = legacy_default_values(
            &schema,
            &config.features.features,
            config.market.assets.len(),
            &context,
        )
        .unwrap();
        assert_eq!(builder.build(&context).unwrap().values, expected);
    }

    #[test]
    fn duplicate_registration_and_missing_provider_are_rejected() {
        let mut registry = ObservationRegistry::new();
        registry
            .register(Box::new(StaticProvider {
                name: "first".into(),
                metadata: vec![metadata("duplicate", "first", ObservationShape::Scalar)],
                values: vec![("duplicate".into(), vec![1.0])],
            }))
            .unwrap();
        let error = registry
            .register(Box::new(StaticProvider {
                name: "second".into(),
                metadata: vec![metadata("duplicate", "second", ObservationShape::Scalar)],
                values: vec![("duplicate".into(), vec![2.0])],
            }))
            .unwrap_err();
        assert!(error.to_string().contains("duplicate observation name"));

        let schema = custom_schema(
            vec![ObservationSchemaEntry {
                index: 0,
                name: "missing".into(),
                provider: "missing".into(),
                source: None,
                component: 0,
                normalization: NormalizationRule::None,
            }],
            vec!["missing"],
        );
        assert!(SchemaObservationBuilder::new(schema, vec![FeatureKind::Return1], 1).is_err());

        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut config = Config::load(root.join("configs/test.toml")).unwrap();
        config.observations.providers = vec![PORTFOLIO_PROVIDER.into()];
        assert!(config.validate().is_err());
    }

    #[test]
    fn compilation_is_independent_of_provider_registration_order() {
        let schema = custom_schema(
            vec![
                ObservationSchemaEntry {
                    index: 0,
                    name: "beta.value".into(),
                    provider: "beta".into(),
                    source: None,
                    component: 0,
                    normalization: NormalizationRule::None,
                },
                ObservationSchemaEntry {
                    index: 1,
                    name: "alpha.value".into(),
                    provider: "alpha".into(),
                    source: None,
                    component: 0,
                    normalization: NormalizationRule::None,
                },
            ],
            vec!["alpha", "beta"],
        );
        let provider = |name: &str, value: f64| StaticProvider {
            name: name.into(),
            metadata: vec![metadata(
                &format!("{name}.value"),
                name,
                ObservationShape::Scalar,
            )],
            values: vec![(format!("{name}.value"), vec![value])],
        };
        let mut first = ObservationRegistry::new();
        first.register(Box::new(provider("alpha", 1.0))).unwrap();
        first.register(Box::new(provider("beta", 2.0))).unwrap();
        let mut second = ObservationRegistry::new();
        second.register(Box::new(provider("beta", 2.0))).unwrap();
        second.register(Box::new(provider("alpha", 1.0))).unwrap();
        let first = SchemaObservationBuilder::with_registry(schema.clone(), first, 1).unwrap();
        let second = SchemaObservationBuilder::with_registry(schema, second, 1).unwrap();
        let (market, portfolio, prices) = context();
        let context = ObservationContext {
            market: &market,
            portfolio: &portfolio,
            prices: &prices,
            starting_net_worth: 100.0,
            maximum_net_worth: 100.0,
            age: 0,
            episode_steps: 1,
        };
        assert_eq!(first.build(&context).unwrap().values, vec![2.0, 1.0]);
        assert_eq!(
            first.build(&context).unwrap(),
            second.build(&context).unwrap()
        );
    }

    #[test]
    fn normalization_rules_and_schema_versions_are_explicit() {
        assert_eq!(
            normalize(
                2.0,
                &NormalizationRule::MinMax {
                    minimum: 0.0,
                    maximum: 4.0
                }
            )
            .unwrap(),
            0.5
        );
        assert_eq!(
            normalize(-3.0, &NormalizationRule::SignedLog).unwrap(),
            -(4.0_f64.ln() as f32)
        );
        assert_eq!(
            normalize(0.25, &NormalizationRule::Percentage { scale: 100.0 }).unwrap(),
            25.0
        );
        assert_eq!(
            normalize(
                3.0,
                &NormalizationRule::ZScore {
                    mean: 1.0,
                    standard_deviation: 2.0,
                },
            )
            .unwrap(),
            1.0
        );
        assert_eq!(
            normalize(
                3.0,
                &NormalizationRule::Custom {
                    scale: 2.0,
                    offset: -1.0,
                },
            )
            .unwrap(),
            5.0
        );
        assert!(normalize(0.0, &NormalizationRule::Log).is_err());
        let entry = ObservationSchemaEntry {
            index: 0,
            name: "value".into(),
            provider: "provider".into(),
            source: None,
            component: 0,
            normalization: NormalizationRule::None,
        };
        let first = custom_schema(vec![entry.clone()], vec!["provider"]);
        let mut second = custom_schema(vec![entry], vec!["provider"]);
        second.version = 2;
        assert_ne!(first.identity(1).unwrap(), second.identity(1).unwrap());

        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut config = Config::load(root.join("configs/test.toml")).unwrap();
        config.observations.normalizations.insert(
            "portfolio.cash_fraction".into(),
            ObservationNormalizationConfig {
                kind: ObservationNormalizationKind::MinMax,
                minimum: Some(0.0),
                maximum: Some(2.0),
                mean: None,
                standard_deviation: None,
                scale: None,
                offset: None,
            },
        );
        let schema = ObservationSchema::try_from_config(&config).unwrap();
        assert_eq!(
            schema
                .entries
                .iter()
                .find(|entry| entry.name == "portfolio.cash_fraction")
                .unwrap()
                .normalization,
            NormalizationRule::MinMax {
                minimum: 0.0,
                maximum: 2.0
            }
        );
    }

    #[test]
    fn schema_rejects_duplicate_indices_and_unused_registry_metadata() {
        let duplicate_indices = custom_schema(
            vec![
                ObservationSchemaEntry {
                    index: 0,
                    name: "one".into(),
                    provider: "provider".into(),
                    source: None,
                    component: 0,
                    normalization: NormalizationRule::None,
                },
                ObservationSchemaEntry {
                    index: 0,
                    name: "two".into(),
                    provider: "provider".into(),
                    source: None,
                    component: 0,
                    normalization: NormalizationRule::None,
                },
            ],
            vec!["provider"],
        );
        assert!(duplicate_indices.validate(1).is_err());

        let schema = custom_schema(
            vec![ObservationSchemaEntry {
                index: 0,
                name: "one".into(),
                provider: "provider".into(),
                source: None,
                component: 0,
                normalization: NormalizationRule::None,
            }],
            vec!["provider"],
        );
        let mut registry = ObservationRegistry::new();
        registry
            .register(Box::new(StaticProvider {
                name: "provider".into(),
                metadata: vec![
                    metadata("one", "provider", ObservationShape::Scalar),
                    metadata("unused", "provider", ObservationShape::Scalar),
                ],
                values: vec![("one".into(), vec![1.0]), ("unused".into(), vec![2.0])],
            }))
            .unwrap();
        assert!(SchemaObservationBuilder::with_registry(schema, registry, 1).is_err());
    }

    #[test]
    fn dimension_mismatch_and_non_finite_provider_output_fail_loudly() {
        let schema = custom_schema(
            vec![ObservationSchemaEntry {
                index: 0,
                name: "vector".into(),
                provider: "provider".into(),
                source: None,
                component: 0,
                normalization: NormalizationRule::None,
            }],
            vec!["provider"],
        );
        let mut registry = ObservationRegistry::new();
        registry
            .register(Box::new(StaticProvider {
                name: "provider".into(),
                metadata: vec![metadata(
                    "vector",
                    "provider",
                    ObservationShape::Fixed { dimensions: 2 },
                )],
                values: vec![("vector".into(), vec![1.0])],
            }))
            .unwrap();
        let builder = SchemaObservationBuilder::with_registry(schema, registry, 1).unwrap();
        let (market, portfolio, prices) = context();
        let context = ObservationContext {
            market: &market,
            portfolio: &portfolio,
            prices: &prices,
            starting_net_worth: 100.0,
            maximum_net_worth: 100.0,
            age: 0,
            episode_steps: 1,
        };
        assert!(builder.build(&context).is_err());

        let schema = custom_schema(
            vec![ObservationSchemaEntry {
                index: 0,
                name: "vector".into(),
                provider: "provider".into(),
                source: None,
                component: 1,
                normalization: NormalizationRule::None,
            }],
            vec!["provider"],
        );
        let mut registry = ObservationRegistry::new();
        registry
            .register(Box::new(StaticProvider {
                name: "provider".into(),
                metadata: vec![metadata(
                    "vector",
                    "provider",
                    ObservationShape::Fixed { dimensions: 2 },
                )],
                values: vec![("vector".into(), vec![1.0, 2.0])],
            }))
            .unwrap();
        let builder = SchemaObservationBuilder::with_registry(schema, registry, 1).unwrap();
        assert_eq!(builder.build(&context).unwrap().values, vec![2.0]);

        let schema = custom_schema(
            vec![ObservationSchemaEntry {
                index: 0,
                name: "finite".into(),
                provider: "finite_provider".into(),
                source: None,
                component: 0,
                normalization: NormalizationRule::None,
            }],
            vec!["finite_provider"],
        );
        let mut registry = ObservationRegistry::new();
        registry
            .register(Box::new(StaticProvider {
                name: "finite_provider".into(),
                metadata: vec![metadata(
                    "finite",
                    "finite_provider",
                    ObservationShape::Scalar,
                )],
                values: vec![("finite".into(), vec![f64::NAN])],
            }))
            .unwrap();
        let builder = SchemaObservationBuilder::with_registry(schema, registry, 1).unwrap();
        assert!(builder.build(&context).is_err());
    }
}
