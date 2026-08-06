use crate::{
    Result,
    config::{
        FeatureKind, ObservationFeatureKind, ObservationNormalizationConfig,
        RollingMarketStatsConfig,
    },
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use super::{
    normalization::{NormalizationRule, apply_override},
    options::ObservationSchemaOptions,
};

pub(crate) const MARKET_PROVIDER: &str = "market";
pub(crate) const PORTFOLIO_PROVIDER: &str = "portfolio";
pub(crate) const AGENT_STATE_PROVIDER: &str = "agent_state";
pub(crate) const AGENT_AWARENESS_PROVIDER: &str = "agent_awareness";
pub(crate) const ROLLING_MARKET_STATS_PROVIDER: &str = "rolling_market_stats";

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
pub enum AgentAwarenessFeatureSpec {
    RecentReward,
    RollingMeanReward,
    RollingRewardVolatility,
    ExplorationLevel,
    SurvivalCostRatio,
    RewardTrend,
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
    AgentAwareness {
        feature: AgentAwarenessFeatureSpec,
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
    AgentAwareness {
        feature: AgentAwarenessFeatureSpec,
    },
    RollingMarketStats {
        metric: RollingMarketMetric,
        asset_index: Option<usize>,
        window: usize,
    },
}

/// Past-only raw-close market statistics published by the rolling-market
/// provider. Values are normalised by the schema, never by the provider.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RollingMarketMetric {
    RollingReturn,
    RollingVolatility,
    PriceToSma,
    RelativeStrength,
    MarketMeanReturn,
    MarketMeanVolatility,
    MarketDispersion,
    MarketBreadth,
}

impl ObservationSource {
    pub(crate) fn provider(&self) -> &'static str {
        match self {
            Self::Market { .. } => MARKET_PROVIDER,
            Self::Portfolio { .. } => PORTFOLIO_PROVIDER,
            Self::AgentState { .. } => AGENT_STATE_PROVIDER,
            Self::AgentAwareness { .. } => AGENT_AWARENESS_PROVIDER,
            Self::RollingMarketStats { .. } => ROLLING_MARKET_STATS_PROVIDER,
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
    pub(crate) fn from_options(options: ObservationSchemaOptions<'_>) -> Self {
        Self::try_from_options(options).expect("configuration was validated before schema creation")
    }

    pub(crate) fn try_from_options(options: ObservationSchemaOptions<'_>) -> Result<Self> {
        options.rolling_market_stats.validate()?;
        let mut features = Vec::new();
        for configured in options.observation_features {
            match configured {
                ObservationFeatureKind::Market => {
                    for feature in options.market_features {
                        features.push(ObservationFeatureSpec::Market {
                            feature: *feature,
                            normalization: NormalizationRule::RatioClipped {
                                minimum: -options.market_clip,
                                maximum: options.market_clip,
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
                ObservationFeatureKind::RecentReward => features.push(agent_awareness(
                    AgentAwarenessFeatureSpec::RecentReward,
                    -5.0,
                    5.0,
                )),
                ObservationFeatureKind::RollingMeanReward => features.push(agent_awareness(
                    AgentAwarenessFeatureSpec::RollingMeanReward,
                    -5.0,
                    5.0,
                )),
                ObservationFeatureKind::RollingRewardVolatility => features.push(agent_awareness(
                    AgentAwarenessFeatureSpec::RollingRewardVolatility,
                    0.0,
                    25.0,
                )),
                ObservationFeatureKind::ExplorationLevel => features.push(agent_awareness(
                    AgentAwarenessFeatureSpec::ExplorationLevel,
                    0.0,
                    1.0,
                )),
                ObservationFeatureKind::SurvivalCostRatio => features.push(agent_awareness(
                    AgentAwarenessFeatureSpec::SurvivalCostRatio,
                    0.0,
                    10.0,
                )),
                ObservationFeatureKind::RewardTrend => features.push(agent_awareness(
                    AgentAwarenessFeatureSpec::RewardTrend,
                    -1.0,
                    1.0,
                )),
            }
        }
        let mut entries = Vec::new();
        for spec in &features {
            match spec {
                ObservationFeatureSpec::Market {
                    feature,
                    normalization,
                } => {
                    for asset_index in 0..options.assets.len() {
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
                                options.normalizations,
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
                        for asset_index in 0..options.assets.len() {
                            let name = portfolio_asset_name(feature, asset_index);
                            push_entry(
                                &mut entries,
                                name.clone(),
                                ObservationSource::Portfolio {
                                    feature: feature.clone(),
                                    asset_index: Some(asset_index),
                                },
                                apply_override(normalization, &name, options.normalizations)?,
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
                            apply_override(normalization, &name, options.normalizations)?,
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
                        apply_override(normalization, &name, options.normalizations)?,
                    );
                }
                ObservationFeatureSpec::AgentAwareness {
                    feature,
                    normalization,
                } => {
                    let name = agent_awareness_name(feature);
                    push_entry(
                        &mut entries,
                        name.clone(),
                        ObservationSource::AgentAwareness {
                            feature: feature.clone(),
                        },
                        apply_override(normalization, &name, options.normalizations)?,
                    );
                }
            }
        }
        if options.rolling_market_stats.enabled {
            append_rolling_market_entries(
                &mut entries,
                options.assets,
                options.rolling_market_stats,
                options.normalizations,
            )?;
        }
        let schema = Self {
            version: options.observation_version,
            market_schema_version: options.market_schema_version,
            features,
            entries,
            providers: options.providers.to_vec(),
        };
        schema.validate(options.assets.len())?;
        for name in options.normalizations.keys() {
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

fn append_rolling_market_entries(
    entries: &mut Vec<ObservationSchemaEntry>,
    assets: &[String],
    config: &RollingMarketStatsConfig,
    overrides: &BTreeMap<String, ObservationNormalizationConfig>,
) -> Result<()> {
    let ratio = NormalizationRule::RatioClipped {
        minimum: -10.0,
        maximum: 10.0,
    };
    let non_negative = NormalizationRule::RatioClipped {
        minimum: 0.0,
        maximum: 10.0,
    };
    let fraction = NormalizationRule::RatioClipped {
        minimum: 0.0,
        maximum: 1.0,
    };
    for (asset_index, asset) in assets.iter().enumerate() {
        for &window in &config.return_windows {
            push_rolling_entry(
                entries,
                format!("rolling_return_{window}.{asset}"),
                RollingMarketMetric::RollingReturn,
                Some(asset_index),
                window,
                &ratio,
                overrides,
            )?;
        }
        for &window in &config.volatility_windows {
            push_rolling_entry(
                entries,
                format!("rolling_volatility_{window}.{asset}"),
                RollingMarketMetric::RollingVolatility,
                Some(asset_index),
                window,
                &non_negative,
                overrides,
            )?;
        }
        for &window in &config.sma_windows {
            push_rolling_entry(
                entries,
                format!("price_to_sma_{window}.{asset}"),
                RollingMarketMetric::PriceToSma,
                Some(asset_index),
                window,
                &ratio,
                overrides,
            )?;
        }
        for &window in &config.relative_strength_windows {
            push_rolling_entry(
                entries,
                format!("relative_strength_{window}.{asset}"),
                RollingMarketMetric::RelativeStrength,
                Some(asset_index),
                window,
                &ratio,
                overrides,
            )?;
        }
    }
    if config.market_summary {
        let window = config.market_summary_window;
        for (name, metric, normalization) in [
            (
                format!("market_mean_return_{window}"),
                RollingMarketMetric::MarketMeanReturn,
                &ratio,
            ),
            (
                format!("market_mean_volatility_{window}"),
                RollingMarketMetric::MarketMeanVolatility,
                &non_negative,
            ),
            (
                format!("market_dispersion_{window}"),
                RollingMarketMetric::MarketDispersion,
                &non_negative,
            ),
            (
                format!("market_breadth_{window}"),
                RollingMarketMetric::MarketBreadth,
                &fraction,
            ),
        ] {
            push_rolling_entry(
                entries,
                name,
                metric,
                None,
                window,
                normalization,
                overrides,
            )?;
        }
    }
    Ok(())
}

fn push_rolling_entry(
    entries: &mut Vec<ObservationSchemaEntry>,
    name: String,
    metric: RollingMarketMetric,
    asset_index: Option<usize>,
    window: usize,
    default: &NormalizationRule,
    overrides: &BTreeMap<String, ObservationNormalizationConfig>,
) -> Result<()> {
    let normalization = apply_override(default, &name, overrides)?;
    push_entry(
        entries,
        name,
        ObservationSource::RollingMarketStats {
            metric,
            asset_index,
            window,
        },
        normalization,
    );
    Ok(())
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

fn agent_awareness_name(feature: &AgentAwarenessFeatureSpec) -> String {
    format!("agent.{}", agent_awareness_feature_name(feature))
}

fn agent_awareness_feature_name(feature: &AgentAwarenessFeatureSpec) -> &'static str {
    match feature {
        AgentAwarenessFeatureSpec::RecentReward => "recent_reward",
        AgentAwarenessFeatureSpec::RollingMeanReward => "rolling_mean_reward",
        AgentAwarenessFeatureSpec::RollingRewardVolatility => "rolling_reward_volatility",
        AgentAwarenessFeatureSpec::ExplorationLevel => "exploration_level",
        AgentAwarenessFeatureSpec::SurvivalCostRatio => "survival_cost_ratio",
        AgentAwarenessFeatureSpec::RewardTrend => "reward_trend",
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

fn agent_awareness(
    feature: AgentAwarenessFeatureSpec,
    minimum: f64,
    maximum: f64,
) -> ObservationFeatureSpec {
    ObservationFeatureSpec::AgentAwareness {
        feature,
        normalization: NormalizationRule::RatioClipped { minimum, maximum },
    }
}
