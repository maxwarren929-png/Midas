use crate::config::{
    FeatureKind, ObservationFeatureKind, ObservationNormalizationConfig, RollingMarketStatsConfig,
};
use std::collections::BTreeMap;

pub(crate) struct ObservationSchemaOptions<'a> {
    pub(crate) assets: &'a [String],
    pub(crate) market_schema_version: u32,
    pub(crate) market_features: &'a [FeatureKind],
    pub(crate) market_clip: f64,
    pub(crate) observation_version: u32,
    pub(crate) observation_features: &'a [ObservationFeatureKind],
    pub(crate) providers: &'a [String],
    pub(crate) normalizations: &'a BTreeMap<String, ObservationNormalizationConfig>,
    pub(crate) rolling_market_stats: &'a RollingMarketStatsConfig,
}

pub(crate) struct ObservationBuilderOptions<'a> {
    pub(crate) market_features: &'a [FeatureKind],
    pub(crate) asset_count: usize,
    pub(crate) rolling_market_stats: &'a RollingMarketStatsConfig,
}
