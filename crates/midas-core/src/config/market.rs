use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MarketConfig {
    pub assets: Vec<String>,
    pub benchmark_asset: String,
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
