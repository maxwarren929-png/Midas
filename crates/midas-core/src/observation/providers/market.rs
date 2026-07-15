use crate::{Result, config::FeatureKind};

use super::super::{
    contracts::{ObservationContext, ObservationMetadata, ObservationProvider, ObservationSink},
    schema::{MARKET_PROVIDER, ObservationSchema, ObservationSource},
};
use super::metadata_for_entries;

pub(super) struct MarketObservationProvider {
    metadata: Vec<ObservationMetadata>,
    outputs: Vec<(String, FeatureKind, usize)>,
    market_features: Vec<FeatureKind>,
    asset_count: usize,
}

impl MarketObservationProvider {
    pub(super) fn new(
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
