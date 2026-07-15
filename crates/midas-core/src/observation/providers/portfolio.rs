use crate::Result;

use super::super::{
    contracts::{ObservationContext, ObservationMetadata, ObservationProvider, ObservationSink},
    schema::{ObservationSchema, ObservationSource, PORTFOLIO_PROVIDER, PortfolioFeatureSpec},
};
use super::metadata_for_entries;

pub(super) struct PortfolioObservationProvider {
    metadata: Vec<ObservationMetadata>,
    outputs: Vec<(String, PortfolioFeatureSpec, Option<usize>)>,
}

impl PortfolioObservationProvider {
    pub(super) fn new(schema: &ObservationSchema, asset_count: usize) -> Result<Self> {
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
