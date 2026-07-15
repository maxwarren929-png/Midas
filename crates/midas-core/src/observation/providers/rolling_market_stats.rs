use crate::{
    AssetId, Result,
    config::RollingMarketStatsConfig,
    market::{FeaturePriceSeries, MarketDataset, PointInTimeFeaturePriceSeries},
};
use chrono::NaiveDate;
use std::collections::BTreeMap;

use super::super::{
    contracts::{
        ObservationContext, ObservationMetadata, ObservationProvider, ObservationShape,
        ObservationSink, ObservationValueType,
    },
    schema::{
        ObservationSchema, ObservationSource, ROLLING_MARKET_STATS_PROVIDER, RollingMarketMetric,
    },
};

#[derive(Clone, Copy)]
pub(crate) struct RollingOutput {
    pub(crate) metric: RollingMarketMetric,
    pub(crate) asset_index: Option<usize>,
    pub(crate) window: usize,
}

/// Past-only market statistics calculated once from the canonical feature-price
/// series. A table is used rather than re-scanning history while agents act so
/// providers remain read-only and compilation remains inexpensive.
pub(super) struct RollingMarketStatsProvider {
    metadata: Vec<ObservationMetadata>,
    outputs: Vec<(String, RollingOutput)>,
    values_by_date: Option<BTreeMap<NaiveDate, Vec<f64>>>,
}

impl RollingMarketStatsProvider {
    pub(super) fn new(
        schema: &ObservationSchema,
        asset_count: usize,
        config: &RollingMarketStatsConfig,
        market_data: Option<&MarketDataset>,
    ) -> Result<Self> {
        if !config.enabled {
            return Err(crate::MidasError::Config(
                "rolling_market_stats provider is configured but disabled".into(),
            ));
        }
        let mut outputs = Vec::new();
        for entry in schema
            .entries
            .iter()
            .filter(|entry| entry.provider == ROLLING_MARKET_STATS_PROVIDER)
        {
            let Some(ObservationSource::RollingMarketStats {
                metric,
                asset_index,
                window,
            }) = entry.source.as_ref()
            else {
                return Err(crate::MidasError::Config(format!(
                    "rolling observation {} has no rolling source",
                    entry.name
                )));
            };
            if *window == 0 || asset_index.is_some_and(|index| index >= asset_count) {
                return Err(crate::MidasError::Config(format!(
                    "rolling observation {} has an invalid source",
                    entry.name
                )));
            }
            let expects_asset = matches!(
                metric,
                RollingMarketMetric::RollingReturn
                    | RollingMarketMetric::RollingVolatility
                    | RollingMarketMetric::PriceToSma
                    | RollingMarketMetric::RelativeStrength
            );
            if expects_asset != asset_index.is_some() {
                return Err(crate::MidasError::Config(format!(
                    "rolling observation {} has incompatible metric dimensions",
                    entry.name
                )));
            }
            outputs.push((
                entry.name.clone(),
                RollingOutput {
                    metric: *metric,
                    asset_index: *asset_index,
                    window: *window,
                },
            ));
        }
        let values_by_date = market_data
            .map(|data| rolling_values_by_date(data, &outputs, asset_count))
            .transpose()?;
        Ok(Self {
            metadata: rolling_metadata_for_entries(schema),
            outputs,
            values_by_date,
        })
    }
}

impl ObservationProvider for RollingMarketStatsProvider {
    fn provider_name(&self) -> &str {
        ROLLING_MARKET_STATS_PROVIDER
    }

    fn metadata(&self) -> Vec<ObservationMetadata> {
        self.metadata.clone()
    }

    fn observe(
        &self,
        context: &ObservationContext<'_>,
        sink: &mut ObservationSink<'_>,
    ) -> Result<()> {
        let values = self
            .values_by_date
            .as_ref()
            .ok_or_else(|| {
                crate::MidasError::Policy(
                    "rolling market observations require immutable market data".into(),
                )
            })?
            .get(&context.market.date)
            .ok_or_else(|| {
                crate::MidasError::Policy(format!(
                    "rolling market observations are unavailable for {}",
                    context.market.date
                ))
            })?;
        if values.len() != self.outputs.len() {
            return Err(crate::MidasError::Policy(
                "rolling market observation table has invalid dimensions".into(),
            ));
        }
        for ((name, _), value) in self.outputs.iter().zip(values) {
            sink.publish_scalar(name, *value)?;
        }
        Ok(())
    }
}

pub(crate) fn rolling_values_by_date(
    data: &MarketDataset,
    outputs: &[(String, RollingOutput)],
    asset_count: usize,
) -> Result<BTreeMap<NaiveDate, Vec<f64>>> {
    if data.assets.len() != asset_count {
        return Err(crate::MidasError::Data(
            "rolling market provider asset count does not match market data".into(),
        ));
    }
    let price_series = PointInTimeFeaturePriceSeries::new(data);
    let mut histories = vec![Vec::new(); asset_count];
    let mut table = BTreeMap::new();
    for date in &data.dates {
        let mut live = vec![false; asset_count];
        for asset_index in 0..asset_count {
            let asset = AssetId(asset_index as u64);
            if let Some(bar) = data.bar(*date, asset) {
                let price = price_series.price(asset, *date)?.unwrap_or(bar.close);
                if !price.is_finite() || price <= 0.0 {
                    return Err(crate::MidasError::Data(format!(
                        "rolling market provider received invalid price for asset {asset_index} on {date}"
                    )));
                }
                histories[asset_index].push(price);
                live[asset_index] = bar.tradable;
            }
        }
        let values = outputs
            .iter()
            .map(|(_, output)| rolling_value(output, &histories, &live))
            .collect::<Vec<_>>();
        if values.iter().any(|value| !value.is_finite()) {
            return Err(crate::MidasError::Data(format!(
                "rolling market provider produced a non-finite value on {date}"
            )));
        }
        table.insert(*date, values);
    }
    Ok(table)
}

fn rolling_value(output: &RollingOutput, histories: &[Vec<f64>], live: &[bool]) -> f64 {
    let asset_value = |calculation: fn(&[f64], usize) -> Option<f64>| {
        let index = output.asset_index.expect("validated asset metric");
        if live[index] {
            calculation(&histories[index], output.window).unwrap_or(0.0)
        } else {
            0.0
        }
    };
    match output.metric {
        RollingMarketMetric::RollingReturn => asset_value(rolling_return),
        RollingMarketMetric::RollingVolatility => asset_value(rolling_volatility),
        RollingMarketMetric::PriceToSma => asset_value(price_to_sma),
        RollingMarketMetric::RelativeStrength => {
            let index = output.asset_index.expect("validated asset metric");
            if !live[index] {
                return 0.0;
            }
            let Some(value) = rolling_return(&histories[index], output.window) else {
                return 0.0;
            };
            let returns = live
                .iter()
                .enumerate()
                .filter_map(|(asset, is_live)| {
                    is_live.then(|| rolling_return(&histories[asset], output.window))?
                })
                .collect::<Vec<_>>();
            if returns.is_empty() {
                0.0
            } else {
                value - mean(&returns)
            }
        }
        RollingMarketMetric::MarketMeanReturn => {
            market_values(histories, live, output.window, rolling_return)
                .map_or(0.0, |values| mean(&values))
        }
        RollingMarketMetric::MarketMeanVolatility => {
            market_values(histories, live, output.window, rolling_volatility)
                .map_or(0.0, |values| mean(&values))
        }
        RollingMarketMetric::MarketDispersion => {
            market_values(histories, live, output.window, rolling_return)
                .map_or(0.0, |values| population_standard_deviation(&values))
        }
        RollingMarketMetric::MarketBreadth => {
            let values = market_values(histories, live, output.window, rolling_return);
            values.map_or(0.0, |items| {
                items.iter().filter(|value| **value > 0.0).count() as f64 / items.len() as f64
            })
        }
    }
}

fn market_values(
    histories: &[Vec<f64>],
    live: &[bool],
    window: usize,
    calculation: fn(&[f64], usize) -> Option<f64>,
) -> Option<Vec<f64>> {
    let values = live
        .iter()
        .enumerate()
        .filter_map(|(asset, is_live)| is_live.then(|| calculation(&histories[asset], window))?)
        .collect::<Vec<_>>();
    (!values.is_empty()).then_some(values)
}

/// Simple close-to-close return from the current close to the close `window`
/// observed bars earlier. `None` indicates insufficient history, never a
/// future-data fallback.
fn rolling_return(history: &[f64], window: usize) -> Option<f64> {
    if history.len() <= window {
        return None;
    }
    Some(history.last()? / history[history.len() - 1 - window] - 1.0)
}

/// Population standard deviation of the last `window` simple daily returns.
fn rolling_volatility(history: &[f64], window: usize) -> Option<f64> {
    if history.len() <= window {
        return None;
    }
    let start = history.len() - window - 1;
    let returns = history[start..]
        .windows(2)
        .map(|pair| pair[1] / pair[0] - 1.0)
        .collect::<Vec<_>>();
    Some(population_standard_deviation(&returns))
}

fn price_to_sma(history: &[f64], window: usize) -> Option<f64> {
    if history.len() < window {
        return None;
    }
    let start = history.len() - window;
    let average = mean(&history[start..]);
    if average <= 0.0 {
        return None;
    }
    Some(history.last()? / average - 1.0)
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn population_standard_deviation(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let average = mean(values);
    (values
        .iter()
        .map(|value| (value - average).powi(2))
        .sum::<f64>()
        / values.len() as f64)
        .sqrt()
}

fn rolling_metadata_for_entries(schema: &ObservationSchema) -> Vec<ObservationMetadata> {
    schema
        .entries
        .iter()
        .filter(|entry| entry.provider == ROLLING_MARKET_STATS_PROVIDER)
        .map(|entry| {
            let (metric, asset, window) = match entry.source.as_ref() {
                Some(ObservationSource::RollingMarketStats {
                    metric,
                    asset_index,
                    window,
                }) => (*metric, *asset_index, *window),
                _ => unreachable!("rolling provider validated schema source"),
            };
            let (description, units, tags) = rolling_metadata(metric, asset, window);
            ObservationMetadata {
                name: entry.name.clone(),
                provider: ROLLING_MARKET_STATS_PROVIDER.into(),
                description,
                units,
                value_type: ObservationValueType::Float64,
                shape: ObservationShape::Scalar,
                ordering_hint: Some(entry.index),
                normalization: entry.normalization.clone(),
                version: schema.version,
                tags,
            }
        })
        .collect()
}

fn rolling_metadata(
    metric: RollingMarketMetric,
    asset: Option<usize>,
    window: usize,
) -> (String, String, Vec<String>) {
    let asset_tag = asset.map(|index| format!("asset_{index}"));
    let (description, units, mut tags) = match metric {
        RollingMarketMetric::RollingReturn => (
            format!("close_t / close_(t-{window}) - 1 using point-in-time raw closes"),
            "fraction".into(),
            vec!["market".into(), "rolling".into(), "momentum".into()],
        ),
        RollingMarketMetric::RollingVolatility => (
            format!("population standard deviation of {window} past simple daily returns"),
            "daily_return_stddev".into(),
            vec!["market".into(), "rolling".into(), "volatility".into()],
        ),
        RollingMarketMetric::PriceToSma => (
            format!("close_t / past_only_sma_{window} - 1"),
            "fraction".into(),
            vec!["market".into(), "rolling".into(), "trend".into()],
        ),
        RollingMarketMetric::RelativeStrength => (
            format!(
                "asset rolling return minus cross-asset mean rolling return over {window} bars"
            ),
            "fraction".into(),
            vec!["market".into(), "rolling".into(), "cross_sectional".into()],
        ),
        RollingMarketMetric::MarketMeanReturn => (
            format!("mean rolling return across tradable assets over {window} bars"),
            "fraction".into(),
            vec!["market".into(), "rolling".into(), "summary".into()],
        ),
        RollingMarketMetric::MarketMeanVolatility => (
            format!("mean rolling volatility across tradable assets over {window} bars"),
            "daily_return_stddev".into(),
            vec![
                "market".into(),
                "rolling".into(),
                "summary".into(),
                "volatility".into(),
            ],
        ),
        RollingMarketMetric::MarketDispersion => (
            format!(
                "cross-asset population standard deviation of rolling returns over {window} bars"
            ),
            "fraction".into(),
            vec![
                "market".into(),
                "rolling".into(),
                "summary".into(),
                "cross_sectional".into(),
            ],
        ),
        RollingMarketMetric::MarketBreadth => (
            format!("fraction of tradable assets with positive rolling return over {window} bars"),
            "fraction".into(),
            vec!["market".into(), "rolling".into(), "summary".into()],
        ),
    };
    if let Some(asset_tag) = asset_tag {
        tags.push(asset_tag);
    }
    (description, units, tags)
}
