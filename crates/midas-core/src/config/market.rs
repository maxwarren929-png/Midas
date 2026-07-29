use crate::{MidasError, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MarketConfig {
    pub assets: Vec<String>,
    pub benchmark_asset: String,
}

impl MarketConfig {
    pub(super) fn validate(&self) -> Result<()> {
        let bad = |message: &str| Err(MidasError::Config(message.into()));
        if self.assets.is_empty() {
            return bad("market.assets must not be empty");
        }
        let mut assets = self.assets.clone();
        assets.sort();
        assets.dedup();
        if assets.len() != self.assets.len() {
            return bad("market.assets contains duplicates");
        }
        if !self.assets.contains(&self.benchmark_asset) {
            return bad("benchmark_asset must be in market.assets");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DataConfig {
    pub path: String,
    #[serde(default)]
    pub format: DataFormat,
    #[serde(default)]
    pub corporate_actions_path: Option<String>,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataFormat {
    #[default]
    Csv,
    Parquet,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeatureConfig {
    pub schema_version: u32,
    pub features: Vec<FeatureKind>,
    pub clip: f64,
}

impl FeatureConfig {
    pub(super) fn validate(&self) -> Result<()> {
        if self.schema_version == 0
            || self.features.is_empty()
            || !self.clip.is_finite()
            || self.clip <= 0.0
        {
            return Err(MidasError::Config(
                "feature schema version, feature list, and clip must be valid".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureKind {
    Return1,
    Return5,
    Return20,
    PriceToMa20,
    PriceToMa50,
    Volatility20,
    VolumeChange,
    VolumeToMa20,
    DistanceHigh20,
    Tradable,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorporateActionMode {
    #[default]
    Reject,
    ExplicitEvents,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorporateActionConfig {
    pub mode: CorporateActionMode,
    pub discontinuity_threshold: f64,
    pub adjusted_price_tolerance: f64,
}

impl Default for CorporateActionConfig {
    fn default() -> Self {
        Self {
            mode: CorporateActionMode::Reject,
            discontinuity_threshold: 0.35,
            adjusted_price_tolerance: 1e-6,
        }
    }
}

impl CorporateActionConfig {
    pub(super) fn validate(&self) -> Result<()> {
        if self.discontinuity_threshold <= 0.0
            || self.discontinuity_threshold >= 1.0
            || !self.discontinuity_threshold.is_finite()
            || self.adjusted_price_tolerance < 0.0
            || !self.adjusted_price_tolerance.is_finite()
        {
            return Err(MidasError::Config(
                "corporate action thresholds are invalid".into(),
            ));
        }
        Ok(())
    }
}
