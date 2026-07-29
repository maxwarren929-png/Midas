use super::Config;
use crate::{
    MidasError, Result,
    observation::{ObservationBuilderOptions, ObservationSchemaOptions},
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

impl Config {
    pub(crate) fn observation_schema_options(&self) -> ObservationSchemaOptions<'_> {
        ObservationSchemaOptions {
            assets: &self.market.assets,
            market_schema_version: self.features.schema_version,
            market_features: &self.features.features,
            market_clip: self.features.clip,
            observation_version: self.observation.version,
            observation_features: &self.observation.features,
            providers: &self.observations.providers,
            normalizations: &self.observations.normalizations,
            rolling_market_stats: &self.observations.rolling_market_stats,
        }
    }

    pub(crate) fn observation_builder_options(&self) -> ObservationBuilderOptions<'_> {
        ObservationBuilderOptions {
            market_features: &self.features.features,
            asset_count: self.market.assets.len(),
            rolling_market_stats: &self.observations.rolling_market_stats,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationFeatureKind {
    Market,
    CashAllocation,
    AssetAllocation,
    UnrealizedReturn,
    NetWorthToStarting,
    CurrentDrawdown,
    PortfolioReturn1,
    TurnoverToNetWorth,
    NormalizedAge,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationConfig {
    pub version: u32,
    pub features: Vec<ObservationFeatureKind>,
}

impl Default for ObservationConfig {
    fn default() -> Self {
        Self {
            version: 1,
            features: vec![
                ObservationFeatureKind::Market,
                ObservationFeatureKind::CashAllocation,
                ObservationFeatureKind::AssetAllocation,
                ObservationFeatureKind::UnrealizedReturn,
                ObservationFeatureKind::NetWorthToStarting,
                ObservationFeatureKind::CurrentDrawdown,
                ObservationFeatureKind::PortfolioReturn1,
                ObservationFeatureKind::TurnoverToNetWorth,
                ObservationFeatureKind::NormalizedAge,
            ],
        }
    }
}

impl ObservationConfig {
    pub(super) fn validate(&self) -> Result<()> {
        if self.version == 0 || self.features.is_empty() {
            return Err(MidasError::Config(
                "observation schema version and features must be non-empty".into(),
            ));
        }
        Ok(())
    }
}

/// Registry selection and per-name normalisation overrides.  The singular
/// `observation` table remains the backwards-compatible compact schema source;
/// this table controls which sensor providers are registered.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationRegistryConfig {
    #[serde(default = "default_observation_schema_name")]
    pub schema: String,
    #[serde(default = "default_observation_providers")]
    pub providers: Vec<String>,
    #[serde(default)]
    pub normalizations: BTreeMap<String, ObservationNormalizationConfig>,
    /// Optional past-only market sensors.  This field is omitted from the
    /// configuration hash while disabled so unchanged Prompt 5 configurations
    /// remain compatible with their v12 checkpoints.
    #[serde(default, skip_serializing_if = "RollingMarketStatsConfig::is_default")]
    pub rolling_market_stats: RollingMarketStatsConfig,
}

fn default_observation_schema_name() -> String {
    "default".into()
}

fn default_observation_providers() -> Vec<String> {
    vec!["market".into(), "portfolio".into(), "agent_state".into()]
}

impl Default for ObservationRegistryConfig {
    fn default() -> Self {
        Self {
            schema: default_observation_schema_name(),
            providers: default_observation_providers(),
            normalizations: BTreeMap::new(),
            rolling_market_stats: RollingMarketStatsConfig::default(),
        }
    }
}

impl ObservationRegistryConfig {
    pub(super) fn validate(&self) -> Result<()> {
        let bad = |message: &str| Err(MidasError::Config(message.into()));
        if self.schema.trim().is_empty() || self.providers.is_empty() {
            return bad("observation registry schema and providers must be non-empty");
        }
        if self.schema != "default" {
            return bad(
                "only the built-in observation schema named default is currently supported",
            );
        }
        let mut observation_providers = self.providers.clone();
        observation_providers.sort();
        observation_providers.dedup();
        if observation_providers.len() != self.providers.len()
            || observation_providers
                .iter()
                .any(|provider| provider.trim().is_empty())
        {
            return bad("observation registry providers must be unique non-empty names");
        }
        for normalization in self.normalizations.values() {
            normalization.validate()?;
        }
        self.rolling_market_stats.validate()?;
        Ok(())
    }
}

/// How a rolling sensor represents an unavailable lookback. Prompt 6 supports
/// zero fill only: it is deterministic, finite, and does not consume future
/// bars. Future schemas may add validity-mask observations explicitly.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RollingMissingHistory {
    #[default]
    ZeroFill,
}

/// Opt-in configuration for the `rolling_market_stats` observation provider.
/// Providers emit raw values; all scaling remains schema-owned.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RollingMarketStatsConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_rolling_return_windows")]
    pub return_windows: Vec<usize>,
    #[serde(default = "default_rolling_volatility_windows")]
    pub volatility_windows: Vec<usize>,
    #[serde(default = "default_rolling_sma_windows")]
    pub sma_windows: Vec<usize>,
    #[serde(default = "default_rolling_relative_strength_windows")]
    pub relative_strength_windows: Vec<usize>,
    #[serde(default = "default_rolling_market_summary")]
    pub market_summary: bool,
    #[serde(default = "default_rolling_market_summary_window")]
    pub market_summary_window: usize,
    #[serde(default)]
    pub minimum_history: RollingMissingHistory,
}

fn default_rolling_return_windows() -> Vec<usize> {
    vec![5, 20, 60, 120]
}

fn default_rolling_volatility_windows() -> Vec<usize> {
    vec![20, 60]
}

fn default_rolling_sma_windows() -> Vec<usize> {
    vec![20, 60]
}

fn default_rolling_relative_strength_windows() -> Vec<usize> {
    vec![20, 60]
}

fn default_rolling_market_summary() -> bool {
    true
}

fn default_rolling_market_summary_window() -> usize {
    20
}

impl Default for RollingMarketStatsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            return_windows: default_rolling_return_windows(),
            volatility_windows: default_rolling_volatility_windows(),
            sma_windows: default_rolling_sma_windows(),
            relative_strength_windows: default_rolling_relative_strength_windows(),
            market_summary: default_rolling_market_summary(),
            market_summary_window: default_rolling_market_summary_window(),
            minimum_history: RollingMissingHistory::ZeroFill,
        }
    }
}

impl RollingMarketStatsConfig {
    pub fn is_default(config: &Self) -> bool {
        config == &Self::default()
    }

    pub fn validate(&self) -> Result<()> {
        let groups = [
            ("return_windows", &self.return_windows),
            ("volatility_windows", &self.volatility_windows),
            ("sma_windows", &self.sma_windows),
            ("relative_strength_windows", &self.relative_strength_windows),
        ];
        let mut count = usize::from(self.market_summary);
        for (name, windows) in groups {
            if windows.contains(&0) {
                return Err(MidasError::Config(format!(
                    "observations.rolling_market_stats.{name} must contain positive windows"
                )));
            }
            let unique = windows
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>();
            if unique.len() != windows.len() {
                return Err(MidasError::Config(format!(
                    "observations.rolling_market_stats.{name} must not contain duplicate windows"
                )));
            }
            count += windows.len();
        }
        if self.market_summary_window == 0 {
            return Err(MidasError::Config(
                "observations.rolling_market_stats.market_summary_window must be positive".into(),
            ));
        }
        if !self.enabled {
            return Ok(());
        }
        if count == 0 {
            return Err(MidasError::Config(
                "rolling_market_stats must enable at least one metric".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationNormalizationKind {
    None,
    RatioClipped,
    MinMax,
    ZScore,
    Log,
    SignedLog,
    Percentage,
    Custom,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationNormalizationConfig {
    pub kind: ObservationNormalizationKind,
    #[serde(default)]
    pub minimum: Option<f64>,
    #[serde(default)]
    pub maximum: Option<f64>,
    #[serde(default)]
    pub mean: Option<f64>,
    #[serde(default)]
    pub standard_deviation: Option<f64>,
    #[serde(default)]
    pub scale: Option<f64>,
    #[serde(default)]
    pub offset: Option<f64>,
}

impl ObservationNormalizationConfig {
    pub fn validate(&self) -> Result<()> {
        let all_finite = [
            self.minimum,
            self.maximum,
            self.mean,
            self.standard_deviation,
            self.scale,
            self.offset,
        ]
        .into_iter()
        .flatten()
        .all(f64::is_finite);
        if !all_finite {
            return Err(MidasError::Config(
                "observation normalization values must be finite".into(),
            ));
        }
        let valid = match self.kind {
            ObservationNormalizationKind::None
            | ObservationNormalizationKind::Log
            | ObservationNormalizationKind::SignedLog => true,
            ObservationNormalizationKind::RatioClipped | ObservationNormalizationKind::MinMax => {
                self.minimum
                    .zip(self.maximum)
                    .is_some_and(|(minimum, maximum)| minimum < maximum)
            }
            ObservationNormalizationKind::ZScore => self
                .mean
                .zip(self.standard_deviation)
                .is_some_and(|(_, standard_deviation)| standard_deviation > 0.0),
            ObservationNormalizationKind::Percentage => {
                self.scale.is_some_and(|scale| scale != 0.0)
            }
            ObservationNormalizationKind::Custom => self.scale.is_some() && self.offset.is_some(),
        };
        if valid {
            Ok(())
        } else {
            Err(MidasError::Config(
                "observation normalization configuration is incomplete or invalid".into(),
            ))
        }
    }
}
