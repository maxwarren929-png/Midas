use super::*;
use super::{
    builder::legacy_default_values,
    normalization::normalize,
    providers::rolling_market_stats::{RollingOutput, rolling_values_by_date},
    schema::ROLLING_MARKET_STATS_PROVIDER,
};
use crate::{
    AssetId, Result,
    config::{Config, FeatureKind, ObservationNormalizationConfig, ObservationNormalizationKind},
    execution::Portfolio,
    market::{FeatureRow, MarketBar, MarketDataset},
};
use chrono::{Duration, NaiveDate};
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
            date: NaiveDate::from_ymd_opt(2020, 1, 1).unwrap(),
            values: vec![0.0],
            tradable: vec![true],
        },
        Portfolio::new(100.0, 1),
        vec![Some(10.0)],
    )
}

fn custom_schema(entries: Vec<ObservationSchemaEntry>, providers: Vec<&str>) -> ObservationSchema {
    ObservationSchema {
        version: 1,
        market_schema_version: 1,
        features: Vec::new(),
        entries,
        providers: providers.into_iter().map(str::to_owned).collect(),
    }
}

fn rolling_dataset(prices: &[Vec<f64>]) -> MarketDataset {
    assert!(!prices.is_empty());
    let asset_count = prices.len();
    let steps = prices[0].len();
    assert!(prices.iter().all(|series| series.len() == steps));
    let start = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
    let bars = prices
        .iter()
        .enumerate()
        .flat_map(|(asset_index, series)| {
            series
                .iter()
                .enumerate()
                .map(move |(step, close)| MarketBar {
                    date: start + Duration::days(step as i64),
                    asset_id: AssetId(asset_index as u64),
                    open: *close,
                    high: *close,
                    low: *close,
                    close: *close,
                    adjusted_close: *close,
                    volume: 1_000.0,
                    tradable: true,
                })
        })
        .collect();
    MarketDataset::from_test_bars(
        (0..asset_count).map(|index| format!("A{index}")).collect(),
        bars,
    )
}

fn rolling_config() -> Config {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut config = Config::load(root.join("configs/test.toml")).unwrap();
    config.observation.version = 3;
    config
        .observations
        .providers
        .push(ROLLING_MARKET_STATS_PROVIDER.into());
    config.observations.rolling_market_stats.enabled = true;
    config.observations.rolling_market_stats.return_windows = vec![2];
    config.observations.rolling_market_stats.volatility_windows = vec![2];
    config.observations.rolling_market_stats.sma_windows = vec![2];
    config
        .observations
        .rolling_market_stats
        .relative_strength_windows = vec![2];
    config.observations.rolling_market_stats.market_summary = true;
    config
        .observations
        .rolling_market_stats
        .market_summary_window = 2;
    config
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
        date: NaiveDate::from_ymd_opt(2020, 1, 1).unwrap(),
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
        recent_reward: 0.0,
        rolling_mean_reward: 0.0,
        rolling_reward_volatility: 0.0,
        reward_trend: 0.0,
        current_temperature: 1.0,
        initial_temperature: 1.0,
        daily_survival_cost: 0.0,
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
fn rolling_market_stats_are_past_only_and_zero_fill_missing_history() {
    let data = rolling_dataset(&[vec![100.0, 100.0, 110.0], vec![100.0, 100.0, 90.0]]);
    let outputs = vec![
        (
            "rolling_return_2.A0".into(),
            RollingOutput {
                metric: RollingMarketMetric::RollingReturn,
                asset_index: Some(0),
                window: 2,
            },
        ),
        (
            "rolling_volatility_2.A0".into(),
            RollingOutput {
                metric: RollingMarketMetric::RollingVolatility,
                asset_index: Some(0),
                window: 2,
            },
        ),
        (
            "price_to_sma_2.A0".into(),
            RollingOutput {
                metric: RollingMarketMetric::PriceToSma,
                asset_index: Some(0),
                window: 2,
            },
        ),
        (
            "relative_strength_2.A0".into(),
            RollingOutput {
                metric: RollingMarketMetric::RelativeStrength,
                asset_index: Some(0),
                window: 2,
            },
        ),
        (
            "market_mean_return_2".into(),
            RollingOutput {
                metric: RollingMarketMetric::MarketMeanReturn,
                asset_index: None,
                window: 2,
            },
        ),
        (
            "market_dispersion_2".into(),
            RollingOutput {
                metric: RollingMarketMetric::MarketDispersion,
                asset_index: None,
                window: 2,
            },
        ),
        (
            "market_breadth_2".into(),
            RollingOutput {
                metric: RollingMarketMetric::MarketBreadth,
                asset_index: None,
                window: 2,
            },
        ),
    ];
    let table = rolling_values_by_date(&data, &outputs, 2).unwrap();
    let first = table.get(&data.dates[0]).unwrap();
    assert_eq!(first, &vec![0.0; outputs.len()]);
    let last = table.get(&data.dates[2]).unwrap();
    assert!((last[0] - 0.1).abs() < 1e-12);
    assert!((last[1] - 0.05).abs() < 1e-12);
    assert!((last[2] - (110.0 / 105.0 - 1.0)).abs() < 1e-12);
    assert!((last[3] - 0.1).abs() < 1e-12);
    assert!(last[4].abs() < 1e-12);
    assert!((last[5] - 0.1).abs() < 1e-12);
    assert!((last[6] - 0.5).abs() < 1e-12);
    assert!(last.iter().all(|value| value.is_finite()));
}

#[test]
fn rolling_market_stats_do_not_leak_future_prices() {
    let baseline = rolling_dataset(&[
        vec![100.0, 102.0, 104.0, 106.0, 108.0],
        vec![100.0, 99.0, 98.0, 97.0, 96.0],
    ]);
    let outputs = vec![
        (
            "return".into(),
            RollingOutput {
                metric: RollingMarketMetric::RollingReturn,
                asset_index: Some(0),
                window: 2,
            },
        ),
        (
            "relative".into(),
            RollingOutput {
                metric: RollingMarketMetric::RelativeStrength,
                asset_index: Some(0),
                window: 2,
            },
        ),
        (
            "volatility".into(),
            RollingOutput {
                metric: RollingMarketMetric::RollingVolatility,
                asset_index: Some(0),
                window: 2,
            },
        ),
        (
            "trend".into(),
            RollingOutput {
                metric: RollingMarketMetric::PriceToSma,
                asset_index: Some(0),
                window: 2,
            },
        ),
        (
            "mean_return".into(),
            RollingOutput {
                metric: RollingMarketMetric::MarketMeanReturn,
                asset_index: None,
                window: 2,
            },
        ),
        (
            "mean_volatility".into(),
            RollingOutput {
                metric: RollingMarketMetric::MarketMeanVolatility,
                asset_index: None,
                window: 2,
            },
        ),
        (
            "dispersion".into(),
            RollingOutput {
                metric: RollingMarketMetric::MarketDispersion,
                asset_index: None,
                window: 2,
            },
        ),
        (
            "breadth".into(),
            RollingOutput {
                metric: RollingMarketMetric::MarketBreadth,
                asset_index: None,
                window: 2,
            },
        ),
    ];
    let target = baseline.dates[2];
    let expected = rolling_values_by_date(&baseline, &outputs, 2)
        .unwrap()
        .get(&target)
        .cloned()
        .unwrap();
    let mut changed = baseline.clone();
    for bar in changed.bars.iter_mut().filter(|bar| bar.date > target) {
        bar.close *= 100.0;
        bar.adjusted_close = bar.close;
    }
    let actual = rolling_values_by_date(&changed, &outputs, 2)
        .unwrap()
        .get(&target)
        .cloned()
        .unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn rolling_market_stats_exclude_untradable_assets_without_non_finite_values() {
    let mut data = rolling_dataset(&[vec![100.0, 110.0, 121.0], vec![100.0, 90.0, 81.0]]);
    let final_date = data.dates[2];
    data.bars
        .iter_mut()
        .find(|bar| bar.date == final_date && bar.asset_id == AssetId(1))
        .unwrap()
        .tradable = false;
    let outputs = vec![
        (
            "asset_return".into(),
            RollingOutput {
                metric: RollingMarketMetric::RollingReturn,
                asset_index: Some(1),
                window: 2,
            },
        ),
        (
            "mean".into(),
            RollingOutput {
                metric: RollingMarketMetric::MarketMeanReturn,
                asset_index: None,
                window: 2,
            },
        ),
        (
            "breadth".into(),
            RollingOutput {
                metric: RollingMarketMetric::MarketBreadth,
                asset_index: None,
                window: 2,
            },
        ),
    ];
    let values = rolling_values_by_date(&data, &outputs, 2)
        .unwrap()
        .get(&final_date)
        .cloned()
        .unwrap();
    assert_eq!(values[0], 0.0);
    assert!((values[1] - 0.21).abs() < 1e-12);
    assert_eq!(values[2], 1.0);
    assert!(values.iter().all(|value| value.is_finite()));
}

#[test]
fn rolling_provider_is_opt_in_and_schema_identity_is_dimensioned() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let default_config = Config::load(root.join("configs/test.toml")).unwrap();
    let default_schema = ObservationSchema::try_from_config(&default_config).unwrap();
    let configured = rolling_config();
    configured.validate().unwrap();
    let schema = ObservationSchema::try_from_config(&configured).unwrap();
    let assets = configured.market.assets.len();
    let expected_added = assets * 4 + 4;
    assert_eq!(
        schema.dimension(assets),
        default_schema.dimension(assets) + expected_added
    );
    assert_ne!(
        schema.identity(assets).unwrap(),
        default_schema.identity(assets).unwrap()
    );
    assert!(
        schema
            .entries
            .iter()
            .any(|entry| entry.name == "rolling_return_2.SPY")
    );
    assert!(
        schema
            .entries
            .iter()
            .any(|entry| entry.name == "market_breadth_2")
    );
    let serialized = serde_json::to_value(&default_config).unwrap();
    assert!(
        serialized["observations"]
            .get("rolling_market_stats")
            .is_none()
    );

    let mut invalid = configured.clone();
    invalid.observations.rolling_market_stats.return_windows = vec![0];
    assert!(invalid.validate().is_err());
    let mut duplicate = configured;
    duplicate.observations.rolling_market_stats.return_windows = vec![2, 2];
    assert!(duplicate.validate().is_err());
    let mut disabled_invalid = default_config;
    disabled_invalid
        .observations
        .rolling_market_stats
        .return_windows = vec![0];
    assert!(disabled_invalid.validate().is_err());
}

#[test]
fn agent_awareness_provider_publishes_named_self_state_and_changes_schema_identity() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut config = Config::load(root.join("configs/test.toml")).unwrap();
    let baseline = ObservationSchema::try_from_config(&config).unwrap();
    config.observation.version = 3;
    config.observation.features.extend([
        crate::config::ObservationFeatureKind::RecentReward,
        crate::config::ObservationFeatureKind::RollingMeanReward,
        crate::config::ObservationFeatureKind::RollingRewardVolatility,
        crate::config::ObservationFeatureKind::ExplorationLevel,
        crate::config::ObservationFeatureKind::SurvivalCostRatio,
        crate::config::ObservationFeatureKind::RewardTrend,
    ]);
    config.observations.providers.push("agent_awareness".into());
    config.validate().unwrap();
    let schema = ObservationSchema::try_from_config(&config).unwrap();
    assert_ne!(
        schema.identity(config.market.assets.len()).unwrap(),
        baseline.identity(config.market.assets.len()).unwrap()
    );
    assert_eq!(
        schema
            .entries
            .iter()
            .filter(|entry| entry.provider == "agent_awareness")
            .count(),
        6
    );

    let builder = SchemaObservationBuilder::from_config(schema.clone(), &config, None).unwrap();
    let market = FeatureRow {
        date: NaiveDate::from_ymd_opt(2020, 1, 1).unwrap(),
        values: vec![0.0; config.market.assets.len() * config.features.features.len()],
        tradable: vec![true; config.market.assets.len()],
    };
    let portfolio = Portfolio::new(100.0, config.market.assets.len());
    let prices = vec![Some(10.0); config.market.assets.len()];
    let observation = builder
        .build(&ObservationContext {
            market: &market,
            portfolio: &portfolio,
            prices: &prices,
            starting_net_worth: 100.0,
            maximum_net_worth: 100.0,
            age: 0,
            episode_steps: 1,
            recent_reward: -0.25,
            rolling_mean_reward: 0.10,
            rolling_reward_volatility: 0.04,
            reward_trend: -1.0,
            current_temperature: 0.5,
            initial_temperature: 1.0,
            daily_survival_cost: 2.0,
        })
        .unwrap();
    for (name, expected) in [
        ("agent.recent_reward", -0.25),
        ("agent.rolling_mean_reward", 0.10),
        ("agent.rolling_reward_volatility", 0.04),
        ("agent.exploration_level", 0.5),
        ("agent.survival_cost_ratio", 0.02),
        ("agent.reward_trend", -1.0),
    ] {
        let index = schema
            .entries
            .iter()
            .find(|entry| entry.name == name)
            .unwrap()
            .index;
        assert!(
            (f64::from(observation.values[index]) - expected).abs() < 1e-6,
            "{name}"
        );
    }
    assert!(
        builder
            .registry()
            .metadata_by_name("agent.recent_reward")
            .unwrap()
            .tags
            .iter()
            .any(|tag| tag == "agent_awareness")
    );
}

#[test]
fn rolling_provider_compiles_schema_order_and_exposes_metadata() {
    let mut config = rolling_config();
    config.observations.normalizations.insert(
        "rolling_return_2.SPY".into(),
        ObservationNormalizationConfig {
            kind: ObservationNormalizationKind::Percentage,
            minimum: None,
            maximum: None,
            mean: None,
            standard_deviation: None,
            scale: Some(100.0),
            offset: None,
        },
    );
    let schema = ObservationSchema::try_from_config(&config).unwrap();
    let data = rolling_dataset(&[
        vec![100.0, 100.0, 110.0],
        vec![100.0, 100.0, 90.0],
        vec![100.0, 100.0, 100.0],
    ]);
    let builder =
        SchemaObservationBuilder::from_config(schema.clone(), &config, Some(&data)).unwrap();
    let market = FeatureRow {
        date: data.dates[2],
        values: vec![0.0; config.market.assets.len() * config.features.features.len()],
        tradable: vec![true; config.market.assets.len()],
    };
    let portfolio = Portfolio::new(100.0, config.market.assets.len());
    let prices = vec![Some(100.0); config.market.assets.len()];
    let observation = builder
        .build(&ObservationContext {
            market: &market,
            portfolio: &portfolio,
            prices: &prices,
            starting_net_worth: 100.0,
            maximum_net_worth: 100.0,
            age: 0,
            episode_steps: 1,
            recent_reward: 0.0,
            rolling_mean_reward: 0.0,
            rolling_reward_volatility: 0.0,
            reward_trend: 0.0,
            current_temperature: 1.0,
            initial_temperature: 1.0,
            daily_survival_cost: 0.0,
        })
        .unwrap();
    assert_eq!(
        observation.values.len(),
        schema.dimension(config.market.assets.len())
    );
    let index = schema
        .entries
        .iter()
        .find(|entry| entry.name == "rolling_return_2.SPY")
        .unwrap()
        .index;
    assert!((observation.values[index] - 10.0).abs() < 1e-6);
    let metadata = builder
        .registry()
        .metadata_by_name("rolling_return_2.SPY")
        .unwrap();
    assert_eq!(metadata.provider, ROLLING_MARKET_STATS_PROVIDER);
    assert!(metadata.tags.iter().any(|tag| tag == "momentum"));
    assert_eq!(
        metadata.normalization,
        NormalizationRule::Percentage { scale: 100.0 }
    );
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
    config.observations.providers = vec!["portfolio".into()];
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
        recent_reward: 0.0,
        rolling_mean_reward: 0.0,
        rolling_reward_volatility: 0.0,
        reward_trend: 0.0,
        current_temperature: 1.0,
        initial_temperature: 1.0,
        daily_survival_cost: 0.0,
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
            }
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
            }
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
        recent_reward: 0.0,
        rolling_mean_reward: 0.0,
        rolling_reward_volatility: 0.0,
        reward_trend: 0.0,
        current_temperature: 1.0,
        initial_temperature: 1.0,
        daily_survival_cost: 0.0,
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
