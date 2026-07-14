use chrono::NaiveDate;
use midas_core::{
    AgentId, AssetId, MidasError,
    action::{
        ActionValidationContext, ActionValidator, LongOnlyActionValidator, ProposedAllocation,
    },
    agent::{
        AgentBuildContext, AgentFactory, AgentSpawnRequest, SpawnMetadata, StandardAgentFactory,
    },
    brain::{
        BrainArchitecture, BrainGenome, BurnNeuralModelFactory, ComputeDevice, NeuralInput,
        NeuralModel, NeuralModelFactory, NeuralModelState, NeuralOutput, PolicyBlueprint,
    },
    config::{
        Config, CorporateActionConfig, CorporateActionMode, DeathHandling, LearningKind,
        ObservationFeatureKind, PolicyFailureBehavior, PolicyKind,
    },
    execution::{
        CostExecutionEngine, ExecutionApplier, ExecutionPlanner, ExecutionSnapshot, Portfolio,
        TradeSide,
    },
    market::{
        BackwardFeaturePipeline, CorporateAction, FeaturePipeline, FeaturePriceSeries, MarketBar,
        MarketDataSource, MarketDataset, PointInTimeFeaturePriceSeries, PolarsDataSource,
        write_parquet,
    },
    observation::{
        Observation, ObservationBuilder, ObservationContext, ObservationSchema,
        SchemaObservationBuilder,
    },
    replacement::{LivingAgentSummary, ReplacementContext, ReplacementStrategy},
    simulation::{DatasetSplit, DomainEvent, SimulationEngine, SimulationLifecycle},
};
use proptest::prelude::*;
use std::{fs, path::PathBuf};

fn config() -> Config {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut c = Config::load(root.join("configs/test.toml")).unwrap();
    c.data.path = root.join("data/raw/synthetic.csv").to_string_lossy().into();
    c
}
fn date(s: &str) -> NaiveDate {
    NaiveDate::parse_from_str(s, "%Y-%m-%d").unwrap()
}
fn bar(asset: u64, day: &str, open: f64) -> MarketBar {
    MarketBar {
        date: date(day),
        asset_id: AssetId(asset),
        open,
        high: open + 1.0,
        low: open - 1.0,
        close: open,
        adjusted_close: open,
        volume: 1000.0,
        tradable: true,
    }
}
fn loaded(c: &Config) -> MarketDataset {
    PolarsDataSource.load(c).unwrap()
}
fn validator(
    weights: Vec<f64>,
    tradable: &[bool],
    max: f64,
) -> midas_core::action::ValidatedAllocation {
    LongOnlyActionValidator
        .validate(
            ProposedAllocation::rebalance(weights),
            &ActionValidationContext {
                asset_count: tradable.len(),
                tradable,
                policy_maximum_asset_allocation: Some(max),
                market_maximum_asset_allocation: None,
            },
        )
        .unwrap()
}

#[test]
fn csv_and_parquet_round_trip() {
    let mut c = config();
    let data = loaded(&c);
    assert_eq!(data.bars.len(), 72);
    let file = tempfile::NamedTempFile::new().unwrap();
    write_parquet(&data, file.path()).unwrap();
    c.data.path = file.path().display().to_string();
    c.data.format = midas_core::config::DataFormat::Parquet;
    let again = loaded(&c);
    assert_eq!(data.bars, again.bars);
}

#[test]
fn split_preserves_value_and_entry_price() {
    let mut p = Portfolio::new(0.0, 1);
    p.positions[0].quantity = 10.0;
    p.positions[0].average_entry_price = 100.0;
    let before = p.value(&[Some(100.0)]).unwrap();
    p.apply_corporate_action(&CorporateAction::Split {
        asset_id: AssetId(0),
        date: date("2020-01-02"),
        ratio: 2.0,
    })
    .unwrap();
    assert_eq!(p.positions[0].quantity, 20.0);
    assert_eq!(p.positions[0].average_entry_price, 50.0);
    assert_eq!(before, p.value(&[Some(50.0)]).unwrap());
}

#[test]
fn reverse_and_multiple_splits_compose_without_feature_discontinuity() {
    let config = CorporateActionConfig {
        mode: CorporateActionMode::ExplicitEvents,
        ..Default::default()
    };
    let bars = vec![
        bar(0, "2020-01-01", 100.0),
        bar(0, "2020-01-02", 50.0),
        bar(0, "2020-01-03", 25.0),
    ];
    let actions = vec![
        CorporateAction::Split {
            asset_id: AssetId(0),
            date: date("2020-01-02"),
            ratio: 2.0,
        },
        CorporateAction::Split {
            asset_id: AssetId(0),
            date: date("2020-01-03"),
            ratio: 2.0,
        },
    ];
    let dataset =
        MarketDataset::new_with_actions(vec!["A".into()], bars, actions, &config).unwrap();
    let series = PointInTimeFeaturePriceSeries::new(&dataset);
    assert_eq!(
        series.price(AssetId(0), date("2020-01-01")).unwrap(),
        Some(100.0)
    );
    assert_eq!(
        series.price(AssetId(0), date("2020-01-02")).unwrap(),
        Some(100.0)
    );
    assert_eq!(
        series.price(AssetId(0), date("2020-01-03")).unwrap(),
        Some(100.0)
    );
    let mut portfolio = Portfolio::new(0.0, 1);
    portfolio.positions[0].quantity = 100.0;
    portfolio.positions[0].average_entry_price = 10.0;
    portfolio
        .apply_corporate_action(&CorporateAction::Split {
            asset_id: AssetId(0),
            date: date("2020-01-02"),
            ratio: 0.2,
        })
        .unwrap();
    assert_eq!(portfolio.positions[0].quantity, 20.0);
    assert_eq!(portfolio.positions[0].average_entry_price, 50.0);
    assert_eq!(portfolio.value(&[Some(50.0)]).unwrap(), 1000.0);
}

#[test]
fn dividend_is_credited_once_on_declared_event() {
    let action = CorporateAction::CashDividend {
        asset_id: AssetId(0),
        date: date("2020-01-02"),
        amount_per_share: 1.25,
    };
    let mut portfolio = Portfolio::new(10.0, 1);
    portfolio.positions[0].quantity = 4.0;
    portfolio.apply_corporate_action(&action).unwrap();
    assert_eq!(portfolio.cash, 15.0);
    assert_eq!(portfolio.cash, 15.0);
    assert_eq!(portfolio.positions[0].quantity, 4.0);
}

#[test]
fn same_day_split_and_dividend_follow_documented_order() {
    let corporate = CorporateActionConfig {
        mode: CorporateActionMode::ExplicitEvents,
        ..Default::default()
    };
    let actions = vec![
        CorporateAction::CashDividend {
            asset_id: AssetId(0),
            date: date("2020-01-02"),
            amount_per_share: 1.0,
        },
        CorporateAction::Split {
            asset_id: AssetId(0),
            date: date("2020-01-02"),
            ratio: 2.0,
        },
    ];
    let dataset = MarketDataset::new_with_actions(
        vec!["A".into()],
        vec![bar(0, "2020-01-01", 100.0), bar(0, "2020-01-02", 50.0)],
        actions,
        &corporate,
    )
    .unwrap();
    let mut portfolio = Portfolio::new(0.0, 1);
    portfolio.positions[0].quantity = 10.0;
    portfolio.positions[0].average_entry_price = 100.0;
    for action in &dataset.corporate_actions {
        portfolio.apply_corporate_action(action).unwrap();
    }
    assert_eq!(portfolio.positions[0].quantity, 20.0);
    assert_eq!(portfolio.cash, 20.0);
}

#[test]
fn reject_mode_refuses_actions_adjustments_and_discontinuities() {
    let reject = CorporateActionConfig::default();
    let bars = vec![bar(0, "2020-01-01", 100.0), bar(0, "2020-01-02", 50.0)];
    let action = CorporateAction::Split {
        asset_id: AssetId(0),
        date: date("2020-01-02"),
        ratio: 2.0,
    };
    assert!(
        MarketDataset::new_with_actions(vec!["A".into()], bars.clone(), vec![action], &reject)
            .is_err()
    );
    assert!(MarketDataset::new_with_actions(vec!["A".into()], bars, vec![], &reject).is_err());
    let mut adjusted = bar(0, "2020-01-01", 100.0);
    adjusted.adjusted_close = 50.0;
    assert!(MarketDataset::new(vec!["A".into()], vec![adjusted]).is_err());
}

#[test]
fn split_features_are_point_in_time_and_future_safe() {
    let mut c = config();
    c.market.assets = vec!["A".into()];
    c.market.benchmark_asset = "A".into();
    c.corporate_actions.mode = CorporateActionMode::ExplicitEvents;
    let bars = vec![
        bar(0, "2020-01-01", 100.0),
        bar(0, "2020-01-02", 101.0),
        bar(0, "2020-01-03", 50.5),
        bar(0, "2020-01-04", 51.0),
    ];
    let action = CorporateAction::Split {
        asset_id: AssetId(0),
        date: date("2020-01-03"),
        ratio: 2.0,
    };
    let with = MarketDataset::new_with_actions(
        vec!["A".into()],
        bars.clone(),
        vec![action.clone()],
        &c.corporate_actions,
    )
    .unwrap();
    let without = MarketDataset::new_with_actions(
        vec!["A".into()],
        bars[..2].to_vec(),
        vec![],
        &c.corporate_actions,
    )
    .unwrap();
    let pipeline = BackwardFeaturePipeline::from_config(&c);
    let early = without.dates.clone();
    let a = pipeline.generate(&with, &early).unwrap();
    let b = pipeline.generate(&without, &early).unwrap();
    assert_eq!(
        a.iter().map(|r| &r.values).collect::<Vec<_>>(),
        b.iter().map(|r| &r.values).collect::<Vec<_>>()
    );
    let rows = pipeline.generate(&with, &with.dates).unwrap();
    let return_index = c
        .features
        .features
        .iter()
        .position(|f| *f == midas_core::config::FeatureKind::Return1)
        .unwrap();
    assert!((rows[2].values[return_index]).abs() < 1e-6);
}

#[test]
fn execution_rate_validation_rejects_invalid_economics() {
    for mutate in [0, 1, 2] {
        let mut c = config();
        match mutate {
            0 => c.execution.spread_rate = 2.0,
            1 => c.execution.slippage_rate = 1.0,
            _ => {
                c.execution.spread_rate = 1.8;
                c.execution.slippage_rate = 0.11
            }
        }
        assert!(c.validate().is_err());
    }
    let mut c = config();
    c.execution.spread_rate = 2.0;
    let engine = CostExecutionEngine { costs: c.execution };
    let p = Portfolio::new(100.0, 1);
    let b = bar(0, "2020-01-01", 10.0);
    let action = validator(vec![0.0, 1.0], &[true], 1.0);
    assert!(
        engine
            .plan(
                &p,
                &action,
                &ExecutionSnapshot {
                    bars: &[Some(&b)],
                    valuation_prices: &[Some(10.0)]
                },
                1.0
            )
            .is_err()
    );
}

#[test]
fn planned_trade_values_are_positive_and_commission_non_negative() {
    let c = config();
    let engine = CostExecutionEngine { costs: c.execution };
    let p = Portfolio::new(100.0, 1);
    let b = bar(0, "2020-01-01", 10.0);
    let action = validator(vec![0.0, 1.0], &[true], 1.0);
    let plan = engine
        .plan(
            &p,
            &action,
            &ExecutionSnapshot {
                bars: &[Some(&b)],
                valuation_prices: &[Some(10.0)],
            },
            1.0,
        )
        .unwrap();
    assert!(plan.trades.iter().all(|t| t.execution_price > 0.0
        && t.quantity > 0.0
        && t.commission >= 0.0
        && t.spread_cost >= 0.0
        && t.slippage_cost >= 0.0));
}

#[test]
fn action_validator_masks_caps_and_falls_back() {
    let valid = validator(vec![0.0, 0.9, 0.1], &[true, false], 0.4);
    assert!(valid.weights()[1] <= 0.4 + 1e-9);
    assert_eq!(valid.weights()[2], 0.0);
    assert!((valid.weights().iter().sum::<f64>() - 1.0).abs() < 1e-9);
    let fallback = validator(vec![f64::NAN, 1.0, 0.0], &[true, true], 0.4);
    assert!(fallback.used_fallback());
    assert_eq!(fallback.weights(), &[1.0, 0.0, 0.0]);
}

#[test]
fn action_validator_handles_edge_cases_without_drift_or_looping() {
    let proposals = [
        vec![0.0, 1e300, 1e-300, 5.0, 0.0, 9.0],
        vec![0.0; 6],
        vec![f64::NAN; 6],
        vec![f64::INFINITY, 1.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, -1.0, 1.0, 0.0, 0.0, 0.0],
    ];
    for maximum in [1.0, 0.01, 0.0001] {
        for (tradable, proposal) in [
            (vec![true, true, true, true, true], proposals[0].clone()),
            (
                vec![false, false, false, false, false],
                proposals[1].clone(),
            ),
            (vec![true, false, true, false, true], proposals[2].clone()),
            (vec![false, true, false, true, false], proposals[3].clone()),
            (vec![true, true, false, false, true], proposals[4].clone()),
        ] {
            let value = validator(proposal, &tradable, maximum);
            assert!(value.weights().iter().all(|weight| weight.is_finite()));
            assert!((value.weights().iter().sum::<f64>() - 1.0).abs() < 1e-9);
            assert!(
                value
                    .weights()
                    .iter()
                    .skip(1)
                    .zip(&tradable)
                    .all(|(weight, available)| {
                        (!*available && *weight == 0.0) || *weight <= maximum + 1e-9
                    })
            );
        }
    }
    let one = validator(vec![0.0, 10.0, 1.0], &[false, true], 0.0001);
    assert_eq!(one.weights()[1], 0.0);
    assert!(one.weights()[2] <= 0.0001 + 1e-9);
    let many = vec![true; 127];
    let mut proposal = vec![0.0; many.len() + 1];
    proposal[1..].fill(1.0);
    let value = validator(proposal, &many, 0.003);
    assert!((value.weights().iter().sum::<f64>() - 1.0).abs() < 1e-9);
    assert!(
        value
            .weights()
            .iter()
            .skip(1)
            .all(|weight| *weight <= 0.003 + 1e-9)
    );
}

#[test]
fn random_and_malicious_proposals_cannot_bypass_concentration() {
    let mut c = config();
    c.brain.traits.maximum_allocation.initial = 0.25;
    for seed in 0..100 {
        let mut policy = midas_core::brain::RandomPolicy::new(seed);
        let observation = Observation {
            schema_hash: "x".into(),
            values: vec![],
        };
        let proposal = midas_core::brain::Policy::propose_action(
            &mut policy,
            &observation,
            &midas_core::brain::PolicyContext {
                tradable: &[true, true, true],
                current_allocations: &[1.0, 0.0, 0.0, 0.0],
            },
        )
        .unwrap();
        let valid = LongOnlyActionValidator
            .validate(
                proposal,
                &ActionValidationContext {
                    asset_count: 3,
                    tradable: &[true, true, true],
                    policy_maximum_asset_allocation: Some(0.25),
                    market_maximum_asset_allocation: None,
                },
            )
            .unwrap();
        assert!(valid.weights().iter().skip(1).all(|v| *v <= 0.25 + 1e-9));
    }
    let valid = validator(vec![0.0, 100.0, 0.0, 0.0], &[true, true, true], 0.25);
    assert!(valid.weights().iter().skip(1).all(|v| *v <= 0.25 + 1e-9));
}

#[test]
fn proportional_planning_includes_costs_and_is_order_invariant() {
    let mut c = config();
    c.execution.commission_fixed = 2.0;
    c.execution.commission_rate = 0.01;
    c.execution.spread_rate = 0.02;
    c.execution.slippage_rate = 0.01;
    let engine = CostExecutionEngine { costs: c.execution };
    let p = Portfolio::new(100.0, 2);
    let a = bar(0, "2020-01-01", 10.0);
    let b = bar(1, "2020-01-01", 20.0);
    let action = validator(vec![0.0, 0.5, 0.5], &[true, true], 1.0);
    let plan = engine
        .plan(
            &p,
            &action,
            &ExecutionSnapshot {
                bars: &[Some(&a), Some(&b)],
                valuation_prices: &[Some(10.0), Some(20.0)],
            },
            1.0,
        )
        .unwrap();
    let buys = plan
        .trades
        .iter()
        .filter(|t| t.side == TradeSide::Buy)
        .collect::<Vec<_>>();
    assert_eq!(buys.len(), 2);
    assert!((buys[0].notional / 50.0 - plan.affordability_scale).abs() < 1e-9);
    assert!((buys[1].notional / 50.0 - plan.affordability_scale).abs() < 1e-9);
    let mut reversed_a = b.clone();
    reversed_a.asset_id = AssetId(0);
    let mut reversed_b = a.clone();
    reversed_b.asset_id = AssetId(1);
    let reversed = engine
        .plan(
            &p,
            &action,
            &ExecutionSnapshot {
                bars: &[Some(&reversed_a), Some(&reversed_b)],
                valuation_prices: &[Some(20.0), Some(10.0)],
            },
            1.0,
        )
        .unwrap();
    assert!((plan.affordability_scale - reversed.affordability_scale).abs() < 1e-12);
    assert_eq!(
        plan.trades.iter().map(|t| t.notional).sum::<f64>(),
        reversed.trades.iter().map(|t| t.notional).sum::<f64>()
    );
}

#[test]
fn execution_uses_stable_asset_ids_and_rejects_tampered_plans() {
    let mut c = config();
    c.execution.commission_fixed = 1.0;
    c.execution.commission_rate = 0.01;
    c.execution.spread_rate = 0.02;
    c.execution.slippage_rate = 0.01;
    let engine = CostExecutionEngine { costs: c.execution };
    let mut first = Portfolio::new(100.0, 2);
    first.positions[0].asset_id = AssetId(10);
    first.positions[1].asset_id = AssetId(20);
    let first_a = MarketBar {
        asset_id: AssetId(10),
        ..bar(10, "2020-01-01", 10.0)
    };
    let first_b = MarketBar {
        asset_id: AssetId(20),
        ..bar(20, "2020-01-01", 20.0)
    };
    let first_bars = [Some(&first_a), Some(&first_b)];
    let first_prices = [Some(10.0), Some(20.0)];
    let action = validator(vec![0.0, 0.5, 0.5], &[true, true], 1.0);
    let first_plan = engine
        .plan(
            &first,
            &action,
            &ExecutionSnapshot {
                bars: &first_bars,
                valuation_prices: &first_prices,
            },
            0.0,
        )
        .unwrap();
    let mut second = Portfolio::new(100.0, 2);
    second.positions.swap(0, 1);
    second.positions[0].asset_id = AssetId(20);
    second.positions[1].asset_id = AssetId(10);
    let second_bars = [Some(&first_b), Some(&first_a)];
    let second_prices = [Some(20.0), Some(10.0)];
    let second_action = validator(vec![0.0, 0.5, 0.5], &[true, true], 1.0);
    let second_plan = engine
        .plan(
            &second,
            &second_action,
            &ExecutionSnapshot {
                bars: &second_bars,
                valuation_prices: &second_prices,
            },
            0.0,
        )
        .unwrap();
    let by_id = |plan: &midas_core::execution::ExecutionPlan| {
        plan.trades
            .iter()
            .map(|trade| (trade.asset_id, (trade.quantity, trade.notional)))
            .collect::<std::collections::BTreeMap<_, _>>()
    };
    assert_eq!(by_id(&first_plan), by_id(&second_plan));
    let first_report = engine.apply(&mut first, &first_plan).unwrap();
    assert!(first.cash >= 0.0);
    assert!(
        first_report
            .trades
            .iter()
            .all(|trade| trade.asset_id.0 >= 10)
    );
    let mut tampered = first_plan.clone();
    tampered.trades[0].commission += 1.0;
    assert!(
        engine
            .apply(&mut Portfolio::new(100.0, 2), &tampered)
            .is_err()
    );
    let unavailable_bars = [Some(&first_a), None];
    let mut unavailable_portfolio = Portfolio::new(100.0, 2);
    unavailable_portfolio.positions[0].asset_id = AssetId(10);
    unavailable_portfolio.positions[1].asset_id = AssetId(20);
    let unavailable = engine
        .plan(
            &unavailable_portfolio,
            &action,
            &ExecutionSnapshot {
                bars: &unavailable_bars,
                valuation_prices: &first_prices,
            },
            0.0,
        )
        .unwrap();
    assert!(unavailable.rejected_assets.contains(&AssetId(20)));
    assert!(
        unavailable
            .trades
            .iter()
            .all(|trade| trade.asset_id != AssetId(20))
    );
}

#[test]
fn minimum_trade_filter_stabilizes_and_apply_is_atomic() {
    let mut c = config();
    c.execution.commission_fixed = 10.0;
    let engine = CostExecutionEngine { costs: c.execution };
    let mut p = Portfolio::new(30.0, 2);
    let a = bar(0, "2020-01-01", 10.0);
    let b = bar(1, "2020-01-01", 10.0);
    let action = validator(vec![0.0, 0.9, 0.1], &[true, true], 1.0);
    let plan = engine
        .plan(
            &p,
            &action,
            &ExecutionSnapshot {
                bars: &[Some(&a), Some(&b)],
                valuation_prices: &[Some(10.0), Some(10.0)],
            },
            5.0,
        )
        .unwrap();
    assert_eq!(
        plan.trades
            .iter()
            .filter(|t| t.side == TradeSide::Buy)
            .count(),
        1
    );
    engine.apply(&mut p, &plan).unwrap();
    assert!(p.cash >= 0.0);

    let mut expensive = config();
    expensive.execution.commission_fixed = 100.0;
    let expensive_engine = CostExecutionEngine {
        costs: expensive.execution,
    };
    let no_trade = expensive_engine
        .plan(
            &Portfolio::new(30.0, 1),
            &validator(vec![0.0, 1.0], &[true], 1.0),
            &ExecutionSnapshot {
                bars: &[Some(&a)],
                valuation_prices: &[Some(10.0)],
            },
            0.0,
        )
        .unwrap();
    assert!(no_trade.trades.is_empty());
    assert_eq!(no_trade.affordability_scale, 0.0);
}

#[test]
fn observation_schema_drives_dimension_order_and_unrealized_return() {
    let mut c = config();
    c.market.assets = vec!["A".into()];
    c.observation.features = vec![
        ObservationFeatureKind::UnrealizedReturn,
        ObservationFeatureKind::CashAllocation,
    ];
    let schema = ObservationSchema::from_config(&c);
    assert_eq!(schema.dimension(1), 2);
    let builder = SchemaObservationBuilder::new(schema, c.features.features.clone(), 1).unwrap();
    let mut p = Portfolio::new(50.0, 1);
    p.positions[0].quantity = 2.0;
    p.positions[0].average_entry_price = 10.0;
    p.net_worth = 80.0;
    let market = midas_core::market::FeatureRow {
        date: date("2020-01-01"),
        values: vec![0.0; c.features.features.len()],
        tradable: vec![true],
    };
    let context = ObservationContext {
        market: &market,
        portfolio: &p,
        prices: &[Some(15.0)],
        starting_net_worth: 80.0,
        maximum_net_worth: 80.0,
        age: 0,
        episode_steps: 1,
    };
    let observation = builder.build(&context).unwrap();
    assert!((observation.values[0] - 0.5).abs() < 1e-6);
    assert!((observation.values[1] - 0.625).abs() < 1e-6);
    c.observation.features.reverse();
    let reversed = SchemaObservationBuilder::new(
        ObservationSchema::from_config(&c),
        c.features.features.clone(),
        1,
    )
    .unwrap()
    .build(&context)
    .unwrap();
    assert_eq!(reversed.values, vec![0.625, 0.5]);
}

fn bankrupt_config(policy: PolicyKind, handling: DeathHandling) -> Config {
    let mut c = config();
    c.brain.policy = policy;
    c.simulation.population_size = 1;
    c.simulation.starting_cash = 1.0;
    c.simulation.bankruptcy_threshold = 0.5;
    c.simulation.daily_survival_cost = 2.0;
    c.simulation.death_handling = handling;
    c
}
#[test]
fn replacements_preserve_every_policy_blueprint() {
    for policy in [
        PolicyKind::Cash,
        PolicyKind::Random,
        PolicyKind::EqualWeight,
        PolicyKind::BuyAndHold,
        PolicyKind::BurnActorCritic,
    ] {
        let c = bankrupt_config(policy, DeathHandling::Replace);
        let data = loaded(&c);
        let mut engine = SimulationEngine::new(c, data, DatasetSplit::Training).unwrap();
        let initial = engine.agent_blueprints().next().unwrap().1;
        engine.run_steps(1).unwrap();
        let replacement = engine.agent_blueprints().next().unwrap().1;
        assert_eq!(initial, replacement);
        assert_eq!(engine.metrics.births, 2);
    }
}
#[test]
fn configured_no_replacement_is_respected() {
    for handling in [DeathHandling::LeaveVacant, DeathHandling::EndAgentOnly] {
        let c = bankrupt_config(PolicyKind::Cash, handling);
        let data = loaded(&c);
        let mut engine = SimulationEngine::new(c, data, DatasetSplit::Training).unwrap();
        engine.run_steps(1).unwrap();
        assert_eq!(engine.metrics.births, 1);
        if handling == DeathHandling::EndAgentOnly {
            assert_eq!(engine.agents().count(), 0)
        } else {
            assert_eq!(engine.agents().count(), 1);
            assert!(!engine.agents().next().unwrap().alive);
        }
    }
}

struct SelectSpecificLiving;
impl ReplacementStrategy for SelectSpecificLiving {
    fn create_replacement(
        &mut self,
        context: &ReplacementContext<'_>,
    ) -> midas_core::Result<AgentSpawnRequest> {
        let living = context
            .living_population
            .iter()
            .find(|summary| summary.id == AgentId(7))
            .ok_or_else(|| MidasError::Policy("selected living agent is missing".into()))?;
        Ok(AgentSpawnRequest {
            policy_blueprint: living.policy.clone(),
            genome: living.genome.cloned(),
            parent_ids: vec![living.id],
            metadata: SpawnMetadata {
                reason: "test_deterministic_selection".into(),
                ..SpawnMetadata::default()
            },
        })
    }

    fn name(&self) -> &'static str {
        "test_specific_living"
    }
}

#[test]
fn replacement_context_supports_deterministic_read_only_selection() {
    let c = config();
    let data = loaded(&c);
    let engine = SimulationEngine::new(c.clone(), data, DatasetSplit::Training).unwrap();
    let state = engine.agents().next().unwrap().clone();
    let architecture = BrainArchitecture::from_config(
        &c.brain,
        engine
            .observation_schema()
            .identity(c.market.assets.len())
            .unwrap(),
        engine.observation_schema().dimension(c.market.assets.len()),
        c.market.assets.len(),
    );
    let device = c.backend.compute_device().unwrap();
    let living = vec![LivingAgentSummary {
        id: AgentId(7),
        net_worth: state.portfolio.net_worth,
        age: state.age,
        policy: PolicyBlueprint::Neural,
        genome: state.genome.as_ref(),
        maximum_drawdown: state.maximum_drawdown,
        birth_date: state.birth_date,
        parent_ids: state.parent_ids.clone(),
    }];
    let schema = engine.observation_schema().clone();
    let mut strategy = SelectSpecificLiving;
    let request = strategy
        .create_replacement(&ReplacementContext {
            dead_agent: &state,
            dead_policy: PolicyBlueprint::Neural,
            living_population: &living,
            config: &c,
            architecture: &architecture,
            compute_device: &device,
            observation_schema: &schema,
            deterministic_seed: 7,
            completed_lives: &[],
            genome_registry: &std::collections::BTreeMap::new(),
        })
        .unwrap();
    assert_eq!(request.parent_ids, vec![AgentId(7)]);
    assert_eq!(request.genome.as_ref(), living[0].genome);
    let factory = StandardAgentFactory::default();
    let spawned = factory
        .spawn(
            request,
            &AgentBuildContext {
                agent_id: AgentId(99),
                birth_date: state.birth_date,
                config: &c,
                architecture: &architecture,
                compute_device: &device,
            },
        )
        .unwrap();
    assert_eq!(spawned.policy.policy_blueprint(), PolicyBlueprint::Neural);
    assert_eq!(spawned.state.parent_ids, vec![AgentId(7)]);
}

#[test]
fn checkpoint_domain_history_and_completion_are_deterministic() {
    let c = config();
    let data = loaded(&c);
    let mut full = SimulationEngine::new(c.clone(), data.clone(), DatasetSplit::Training).unwrap();
    full.run().unwrap();
    let mut partial =
        SimulationEngine::new(c.clone(), data.clone(), DatasetSplit::Training).unwrap();
    partial.run_steps(3).unwrap();
    let file = tempfile::NamedTempFile::new().unwrap();
    partial.save_checkpoint(file.path()).unwrap();
    assert_eq!(partial.operational_events.len(), 1);
    let mut resumed = SimulationEngine::resume(c.clone(), data.clone(), file.path()).unwrap();
    resumed.run().unwrap();
    assert_eq!(full.domain_events, resumed.domain_events);
    assert_eq!(
        full.agents().collect::<Vec<_>>(),
        resumed.agents().collect::<Vec<_>>()
    );
    assert_eq!(
        full.agent_blueprints().collect::<Vec<_>>(),
        resumed.agent_blueprints().collect::<Vec<_>>()
    );
    let count = resumed.domain_events.len();
    resumed.run().unwrap();
    assert_eq!(count, resumed.domain_events.len());
    let completed = tempfile::NamedTempFile::new().unwrap();
    resumed.save_checkpoint(completed.path()).unwrap();
    let mut restored = SimulationEngine::resume(c, data, completed.path()).unwrap();
    assert_eq!(restored.lifecycle(), SimulationLifecycle::Completed);
    let count = restored.domain_events.len();
    restored.run().unwrap();
    assert_eq!(count, restored.domain_events.len());
    let full_file = tempfile::NamedTempFile::new().unwrap();
    let resumed_file = tempfile::NamedTempFile::new().unwrap();
    full.save_checkpoint(full_file.path()).unwrap();
    resumed.save_checkpoint(resumed_file.path()).unwrap();
    assert_eq!(
        std::fs::read(full_file.path()).unwrap(),
        std::fs::read(resumed_file.path()).unwrap()
    );
}

#[test]
fn checkpoint_rejects_observation_schema_change() {
    let c = config();
    let data = loaded(&c);
    let mut engine =
        SimulationEngine::new(c.clone(), data.clone(), DatasetSplit::Training).unwrap();
    engine.run_steps(1).unwrap();
    let file = tempfile::NamedTempFile::new().unwrap();
    engine.save_checkpoint(file.path()).unwrap();
    let mut changed = c;
    changed.observation.features.reverse();
    let error = match SimulationEngine::resume(changed, data, file.path()) {
        Ok(_) => panic!("schema mismatch accepted"),
        Err(e) => e,
    };
    assert!(error.to_string().contains("observation schema"));
}

#[test]
fn checkpoint_rejects_brain_architecture_change() {
    let c = config();
    let data = loaded(&c);
    let mut engine =
        SimulationEngine::new(c.clone(), data.clone(), DatasetSplit::Training).unwrap();
    engine.run_steps(1).unwrap();
    let file = tempfile::NamedTempFile::new().unwrap();
    engine.save_checkpoint(file.path()).unwrap();
    let mut changed = c;
    changed.brain.hidden_layers = vec![8, 8];
    let error = match SimulationEngine::resume(changed, data, file.path()) {
        Ok(_) => panic!("architecture mismatch accepted"),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains("configuration") || error.to_string().contains("architecture")
    );
}

#[test]
fn checkpoint_rejects_corporate_action_data_change() {
    let mut c = config();
    c.corporate_actions.mode = CorporateActionMode::ExplicitEvents;
    let source = loaded(&c);
    let action = CorporateAction::CashDividend {
        asset_id: AssetId(0),
        date: date("2020-01-03"),
        amount_per_share: 0.25,
    };
    let with_action = MarketDataset::new_with_actions(
        source.assets.clone(),
        source.bars.clone(),
        vec![action],
        &c.corporate_actions,
    )
    .unwrap();
    let mut engine = SimulationEngine::new(c.clone(), with_action, DatasetSplit::Training).unwrap();
    engine.run_steps(1).unwrap();
    let file = tempfile::NamedTempFile::new().unwrap();
    engine.save_checkpoint(file.path()).unwrap();
    let without_action = MarketDataset::new_with_actions(
        source.assets,
        source.bars,
        Vec::new(),
        &c.corporate_actions,
    )
    .unwrap();
    let error = match SimulationEngine::resume(c, without_action, file.path()) {
        Ok(_) => panic!("corporate-action data mismatch accepted"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("dataset identity"));
}

#[test]
fn backend_identity_is_configuration_driven_and_checkpoint_checked() {
    let mut c = config();
    c.backend.device = "default".into();
    assert_eq!(
        c.backend.compute_device().unwrap().identity(),
        "cpu_ndarray:default"
    );
    let data = loaded(&c);
    let mut engine =
        SimulationEngine::new(c.clone(), data.clone(), DatasetSplit::Training).unwrap();
    engine.run_steps(1).unwrap();
    let file = tempfile::NamedTempFile::new().unwrap();
    engine.save_checkpoint(file.path()).unwrap();
    let mut changed = c;
    changed.backend.device = "other-device".into();
    let error = match SimulationEngine::resume(changed, data, file.path()) {
        Ok(_) => panic!("backend mismatch accepted"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("backend") || error.to_string().contains("device"));
    let mut unsupported = config();
    unsupported.backend.device = "unsupported".into();
    let error = unsupported.validate().unwrap_err();
    assert!(error.to_string().contains("unsupported CPU NdArray device"));
}

#[test]
fn checkpoint_v6_is_rejected_as_bias_only_optimizer_layout() {
    let c = config();
    let data = loaded(&c);
    let mut engine =
        SimulationEngine::new(c.clone(), data.clone(), DatasetSplit::Training).unwrap();
    engine.run_steps(1).unwrap();
    let file = tempfile::NamedTempFile::new().unwrap();
    engine.save_checkpoint(file.path()).unwrap();
    let mut bytes = fs::read(file.path()).unwrap();
    assert_eq!(bytes.first().copied(), Some(9));
    bytes[0] = 6;
    fs::write(file.path(), bytes).unwrap();
    let error = match SimulationEngine::resume(c, data, file.path()) {
        Ok(_) => panic!("checkpoint 6 was accepted"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("bias-only optimizer"));
}

#[test]
fn checkpoint_v8_is_rejected_as_single_asset_action_layout() {
    let c = config();
    let data = loaded(&c);
    let mut engine =
        SimulationEngine::new(c.clone(), data.clone(), DatasetSplit::Training).unwrap();
    engine.run_steps(1).unwrap();
    let file = tempfile::NamedTempFile::new().unwrap();
    engine.save_checkpoint(file.path()).unwrap();
    let mut bytes = fs::read(file.path()).unwrap();
    assert_eq!(bytes.first().copied(), Some(9));
    bytes[0] = 8;
    fs::write(file.path(), bytes).unwrap();
    let error = match SimulationEngine::resume(c, data, file.path()) {
        Ok(_) => panic!("checkpoint 8 was accepted"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("single-asset categorical"));
}

#[test]
fn evaluation_does_not_overwrite_training_checkpoint() {
    let mut c = config();
    c.learning.implementation = LearningKind::NoLearning;
    c.evolution.enabled = false;
    let data = loaded(&c);
    let checkpoint = tempfile::NamedTempFile::new().unwrap();
    let mut training =
        SimulationEngine::new(c.clone(), data.clone(), DatasetSplit::Training).unwrap();
    training.run().unwrap();
    training.save_checkpoint(checkpoint.path()).unwrap();
    let before = fs::read(checkpoint.path()).unwrap();

    let mut evaluation =
        SimulationEngine::evaluate_checkpoint(c, data, checkpoint.path(), DatasetSplit::Validation)
            .unwrap();
    evaluation.run().unwrap();
    assert_eq!(before, fs::read(checkpoint.path()).unwrap());
}

struct MockModel {
    schema: String,
    size: usize,
    memory: f32,
    fail: bool,
}
impl NeuralModel for MockModel {
    fn forward(&mut self, input: &NeuralInput<'_>) -> midas_core::Result<NeuralOutput> {
        assert_eq!(input.observation.schema_hash, self.schema);
        if self.fail {
            return Err(MidasError::Policy(
                "non-finite neural output injected".into(),
            ));
        }
        self.memory += 1.0;
        Ok(NeuralOutput {
            logits: vec![0.0; self.size],
            value: self.memory,
        })
    }
    fn reset_memory(&mut self) {
        self.memory = 0.0
    }
    fn snapshot_state(&self) -> midas_core::Result<NeuralModelState> {
        Ok(NeuralModelState::External {
            kind: "mock".into(),
            state: vec![self.memory],
        })
    }
    fn fork(&self) -> midas_core::Result<Box<dyn NeuralModel>> {
        Ok(Box::new(Self {
            schema: self.schema.clone(),
            size: self.size,
            memory: self.memory,
            fail: self.fail,
        }))
    }
}
struct MockFactory {
    fail: bool,
}
impl NeuralModelFactory for MockFactory {
    fn build(
        &self,
        _: &BrainGenome,
        a: &BrainArchitecture,
        device: &ComputeDevice,
    ) -> midas_core::Result<Box<dyn NeuralModel>> {
        assert_eq!(device.identity(), "cpu_ndarray:default");
        Ok(Box::new(MockModel {
            schema: a.observation_schema_hash.clone(),
            size: a.actor_output_size,
            memory: 0.0,
            fail: self.fail,
        }))
    }
    fn restore(
        &self,
        _: &BrainGenome,
        a: &BrainArchitecture,
        state: NeuralModelState,
        device: &ComputeDevice,
    ) -> midas_core::Result<Box<dyn NeuralModel>> {
        assert_eq!(device.identity(), "cpu_ndarray:default");
        let NeuralModelState::External { state, .. } = state else {
            return Err(MidasError::Policy("wrong mock state".into()));
        };
        Ok(Box::new(MockModel {
            schema: a.observation_schema_hash.clone(),
            size: a.actor_output_size,
            memory: state[0],
            fail: self.fail,
        }))
    }
}

struct BootstrapForkFailureModel {
    schema: String,
    size: usize,
    fork_count: u32,
}
impl NeuralModel for BootstrapForkFailureModel {
    fn forward(&mut self, input: &NeuralInput<'_>) -> midas_core::Result<NeuralOutput> {
        assert_eq!(input.observation.schema_hash, self.schema);
        Ok(NeuralOutput {
            logits: vec![0.0; self.size],
            value: 0.0,
        })
    }
    fn reset_memory(&mut self) {}
    fn snapshot_state(&self) -> midas_core::Result<NeuralModelState> {
        Ok(NeuralModelState::External {
            kind: "bootstrap-fork-failure".into(),
            state: vec![self.fork_count as f32],
        })
    }
    fn fork(&self) -> midas_core::Result<Box<dyn NeuralModel>> {
        if self.fork_count >= 1 {
            return Err(MidasError::Policy(
                "injected next-value fork failure".into(),
            ));
        }
        Ok(Box::new(Self {
            schema: self.schema.clone(),
            size: self.size,
            fork_count: self.fork_count + 1,
        }))
    }
}
struct BootstrapForkFailureFactory;
impl NeuralModelFactory for BootstrapForkFailureFactory {
    fn build(
        &self,
        _: &BrainGenome,
        a: &BrainArchitecture,
        _: &ComputeDevice,
    ) -> midas_core::Result<Box<dyn NeuralModel>> {
        Ok(Box::new(BootstrapForkFailureModel {
            schema: a.observation_schema_hash.clone(),
            size: a.actor_output_size,
            fork_count: 0,
        }))
    }
    fn restore(
        &self,
        _: &BrainGenome,
        a: &BrainArchitecture,
        state: NeuralModelState,
        _: &ComputeDevice,
    ) -> midas_core::Result<Box<dyn NeuralModel>> {
        let NeuralModelState::External { state, .. } = state else {
            return Err(MidasError::Policy("wrong bootstrap failure state".into()));
        };
        Ok(Box::new(BootstrapForkFailureModel {
            schema: a.observation_schema_hash.clone(),
            size: a.actor_output_size,
            fork_count: state.first().copied().unwrap_or(0.0) as u32,
        }))
    }
}

#[derive(Clone)]
struct SequencedFactory {
    fail_calls: Vec<u32>,
    invalid_calls: Vec<u32>,
}
struct SequencedModel {
    schema: String,
    size: usize,
    memory: f32,
    calls: u32,
    fail_calls: Vec<u32>,
    invalid_calls: Vec<u32>,
}
impl NeuralModel for SequencedModel {
    fn forward(&mut self, input: &NeuralInput<'_>) -> midas_core::Result<NeuralOutput> {
        assert_eq!(input.observation.schema_hash, self.schema);
        self.calls += 1;
        if self.fail_calls.contains(&self.calls) {
            return Err(MidasError::Policy(format!(
                "sequenced inference failure at call {}",
                self.calls
            )));
        }
        if self.invalid_calls.contains(&self.calls) {
            return Ok(NeuralOutput {
                logits: vec![f32::NAN; self.size],
                value: f32::NAN,
            });
        }
        self.memory += 1.0;
        Ok(NeuralOutput {
            logits: vec![0.0; self.size],
            value: self.memory,
        })
    }
    fn reset_memory(&mut self) {
        self.memory = 0.0;
    }
    fn snapshot_state(&self) -> midas_core::Result<NeuralModelState> {
        Ok(NeuralModelState::External {
            kind: "sequenced".into(),
            state: vec![self.memory, self.calls as f32],
        })
    }
    fn fork(&self) -> midas_core::Result<Box<dyn NeuralModel>> {
        Ok(Box::new(Self {
            schema: self.schema.clone(),
            size: self.size,
            memory: self.memory,
            calls: self.calls,
            fail_calls: self.fail_calls.clone(),
            invalid_calls: self.invalid_calls.clone(),
        }))
    }
}
impl NeuralModelFactory for SequencedFactory {
    fn build(
        &self,
        _: &BrainGenome,
        a: &BrainArchitecture,
        device: &ComputeDevice,
    ) -> midas_core::Result<Box<dyn NeuralModel>> {
        assert_eq!(device.identity(), "cpu_ndarray:default");
        Ok(Box::new(SequencedModel {
            schema: a.observation_schema_hash.clone(),
            size: a.actor_output_size,
            memory: 0.0,
            calls: 0,
            fail_calls: self.fail_calls.clone(),
            invalid_calls: self.invalid_calls.clone(),
        }))
    }
    fn restore(
        &self,
        _: &BrainGenome,
        a: &BrainArchitecture,
        state: NeuralModelState,
        device: &ComputeDevice,
    ) -> midas_core::Result<Box<dyn NeuralModel>> {
        assert_eq!(device.identity(), "cpu_ndarray:default");
        let NeuralModelState::External { state, .. } = state else {
            return Err(MidasError::Policy("wrong sequenced model state".into()));
        };
        Ok(Box::new(SequencedModel {
            schema: a.observation_schema_hash.clone(),
            size: a.actor_output_size,
            memory: state.first().copied().unwrap_or(0.0),
            calls: state.get(1).copied().unwrap_or(0.0) as u32,
            fail_calls: self.fail_calls.clone(),
            invalid_calls: self.invalid_calls.clone(),
        }))
    }
}

fn data_with_actions(c: &mut Config, actions: Vec<CorporateAction>) -> MarketDataset {
    c.corporate_actions.mode = CorporateActionMode::ExplicitEvents;
    let source = loaded(c);
    MarketDataset::new_with_actions(source.assets, source.bars, actions, &c.corporate_actions)
        .unwrap()
}

fn sequenced_engine(
    mut c: Config,
    actions: Vec<CorporateAction>,
    factory: SequencedFactory,
) -> SimulationEngine {
    c.simulation.population_size = 1;
    let data = data_with_actions(&mut c, actions);
    SimulationEngine::new_with_factory(
        c,
        data,
        DatasetSplit::Training,
        Box::new(StandardAgentFactory::with_neural_factory(Box::new(factory))),
    )
    .unwrap()
}
#[test]
fn simulation_runs_mock_and_burn_neural_models() {
    let c = config();
    let data = loaded(&c);
    let factory = StandardAgentFactory::with_neural_factory(Box::new(MockFactory { fail: false }));
    let mut mock = SimulationEngine::new_with_factory(
        c.clone(),
        data.clone(),
        DatasetSplit::Training,
        Box::new(factory),
    )
    .unwrap();
    mock.run_steps(2).unwrap();
    assert!(mock.metrics.agents.values().all(|m| m.survival_days == 2));
    let mut burn = SimulationEngine::new(c, data, DatasetSplit::Training).unwrap();
    burn.run_steps(2).unwrap();
    assert!(burn.metrics.agents.values().all(|m| m.survival_days == 2));
    assert!(!include_str!("../src/simulation.rs").contains("burn::"));
}

#[test]
fn continuing_bootstrap_fork_failure_is_transactional_and_not_zero() {
    let mut c = config();
    c.simulation.population_size = 1;
    c.learning.implementation = LearningKind::Ppo;
    c.learning.rollout_length = 1;
    c.learning.minibatch_size = 1;
    c.learning.sequence_length = 1;
    c.learning.optimisation_epochs = 1;
    let data = loaded(&c);
    let factory = StandardAgentFactory::with_neural_factory(Box::new(BootstrapForkFailureFactory));
    let mut engine =
        SimulationEngine::new_with_factory(c, data, DatasetSplit::Training, Box::new(factory))
            .unwrap();
    let before_agents = engine.agents().cloned().collect::<Vec<_>>();
    let before_events = engine.domain_events.clone();
    let error = engine.run_steps(1).unwrap_err();
    assert!(error.to_string().contains("next-value policy fork failed"));
    assert!(matches!(
        engine.lifecycle(),
        SimulationLifecycle::Failed { .. }
    ));
    assert_eq!(engine.report().steps, 0);
    assert_eq!(engine.agents().cloned().collect::<Vec<_>>(), before_agents);
    assert_eq!(engine.domain_events, before_events);
    let repeated = engine.run_steps(1).unwrap_err();
    assert_eq!(repeated.to_string(), error.to_string());
    assert_eq!(engine.agents().cloned().collect::<Vec<_>>(), before_agents);
    assert_eq!(engine.domain_events, before_events);
}

#[test]
fn policy_failures_emit_diagnostics_and_obey_limits() {
    let mut c = config();
    c.simulation.population_size = 1;
    c.policy_failures.maximum_consecutive = 1;
    c.policy_failures.maximum_total = 1;
    c.policy_failures.behavior = PolicyFailureBehavior::KillAgent;
    let data = loaded(&c);
    let factory = StandardAgentFactory::with_neural_factory(Box::new(MockFactory { fail: true }));
    let mut engine =
        SimulationEngine::new_with_factory(c, data, DatasetSplit::Training, Box::new(factory))
            .unwrap();
    engine.run_steps(1).unwrap();
    assert_eq!(engine.metrics.policy_failures, 1);
    assert_eq!(engine.metrics.births, 2);
    assert!(engine.domain_events.iter().any(|event| matches!(event, DomainEvent::PolicyFailure { category: midas_core::simulation::PolicyFailureCategory::NonFiniteState, message, .. } if message.contains("injected"))));
    let mut c = config();
    c.simulation.population_size = 1;
    c.policy_failures.maximum_consecutive = 1;
    c.policy_failures.maximum_total = 2;
    c.policy_failures.behavior = PolicyFailureBehavior::Continue;
    let data = loaded(&c);
    let factory = StandardAgentFactory::with_neural_factory(Box::new(MockFactory { fail: true }));
    let mut engine =
        SimulationEngine::new_with_factory(c, data, DatasetSplit::Training, Box::new(factory))
            .unwrap();
    engine.run_steps(2).unwrap();
    assert_eq!(engine.metrics.policy_failures, 2);
    assert_eq!(engine.metrics.births, 1);
}

#[test]
fn stop_on_policy_failure_preserves_original_failure_event() {
    let mut c = config();
    c.simulation.population_size = 1;
    c.policy_failures.maximum_consecutive = 1;
    c.policy_failures.maximum_total = 1;
    c.policy_failures.behavior = PolicyFailureBehavior::StopExperiment;
    let data = loaded(&c);
    let factory = StandardAgentFactory::with_neural_factory(Box::new(MockFactory { fail: true }));
    let mut engine =
        SimulationEngine::new_with_factory(c, data, DatasetSplit::Training, Box::new(factory))
            .unwrap();
    assert!(engine.run_steps(1).is_err());
    assert!(engine.domain_events.iter().any(|event| matches!(event, DomainEvent::PolicyFailure { message, .. } if message.contains("injected"))));
}

#[test]
fn recoverable_policy_failure_has_one_fallback_and_advances_normally() {
    let mut c = config();
    c.policy_failures.maximum_consecutive = 3;
    c.policy_failures.maximum_total = 20;
    c.policy_failures.behavior = PolicyFailureBehavior::Continue;
    let mut engine = sequenced_engine(
        c,
        vec![CorporateAction::Split {
            asset_id: AssetId(0),
            date: date("2020-01-03"),
            ratio: 2.0,
        }],
        SequencedFactory {
            fail_calls: vec![1],
            invalid_calls: vec![],
        },
    );
    engine.run_steps(1).unwrap();
    let failures = engine
        .domain_events
        .iter()
        .filter(|event| matches!(event, DomainEvent::PolicyFailure { .. }))
        .count();
    let fallbacks = engine
        .domain_events
        .iter()
        .filter(|event| matches!(event, DomainEvent::ActionFallback { .. }))
        .count();
    assert_eq!(failures, 1);
    assert_eq!(fallbacks, 1);
    assert!(engine.domain_events.iter().any(|event| matches!(
        event,
        DomainEvent::CorporateActionApplied { date: event_date, .. }
            if *event_date == date("2020-01-03")
    )));
    assert!(engine.domain_events.iter().any(|event| matches!(
        event,
        DomainEvent::Valuation { date: event_date, .. } if *event_date == date("2020-01-03")
    )));
    let state = engine.agents().next().unwrap();
    assert_eq!(state.total_policy_failures, 1);
    assert_eq!(state.consecutive_policy_failures, 1);
}

#[test]
fn recoverable_invalid_output_creates_penalized_chronological_rollout_entry() {
    let mut c = config();
    c.learning.implementation = LearningKind::Ppo;
    c.learning.rollout_length = 4;
    c.learning.minibatch_size = 2;
    c.learning.sequence_length = 2;
    c.policy_failures.maximum_consecutive = 3;
    c.policy_failures.maximum_total = 20;
    c.policy_failures.behavior = PolicyFailureBehavior::Continue;
    let mut engine = sequenced_engine(
        c,
        Vec::new(),
        SequencedFactory {
            fail_calls: vec![],
            invalid_calls: vec![1],
        },
    );
    let agent_id = engine.agents().next().unwrap().id;
    engine.run_steps(1).unwrap();
    let midas_core::learning::LearningStrategyState::Ppo { rollout, .. } =
        engine.agent_learning_snapshot(agent_id).unwrap()
    else {
        panic!("expected PPO learning state")
    };
    assert_eq!(rollout.len(), 1);
    let transition = &rollout.transitions()[0];
    assert!(transition.reward < 0.0);
    assert!(transition.invalid_action);
    assert_eq!(transition.action_sample, None);
    assert!(matches!(
        transition.failure_category.clone(),
        Some(midas_core::PolicyFailureCategory::InvalidOutput)
            | Some(midas_core::PolicyFailureCategory::InferenceFailure)
            | Some(midas_core::PolicyFailureCategory::NonFiniteState)
    ));
    assert!(!transition.actor_eligible);
    assert!(!transition.critic_eligible);
    assert!(transition.observation.is_none());
    assert_eq!(engine.metrics.masked_failure_transitions, 1);
    assert_eq!(engine.metrics.actor_ineligible_transitions, 1);
    assert_eq!(engine.metrics.critic_ineligible_transitions, 1);
    assert!(engine.metrics.invalid_action_penalty_total > 0.0);
}

#[test]
fn valid_action_resets_only_consecutive_policy_failures() {
    let mut c = config();
    c.policy_failures.maximum_consecutive = 5;
    c.policy_failures.maximum_total = 20;
    c.policy_failures.behavior = PolicyFailureBehavior::Continue;
    let mut engine = sequenced_engine(
        c,
        Vec::new(),
        SequencedFactory {
            fail_calls: vec![1, 3],
            invalid_calls: vec![],
        },
    );
    engine.run_steps(1).unwrap();
    let first = engine.agents().next().unwrap().clone();
    engine.run_steps(1).unwrap();
    let second = engine.agents().next().unwrap().clone();
    engine.run_steps(1).unwrap();
    let third = engine.agents().next().unwrap().clone();
    assert_eq!(
        (
            first.total_policy_failures,
            first.consecutive_policy_failures
        ),
        (1, 1)
    );
    assert_eq!(
        (
            second.total_policy_failures,
            second.consecutive_policy_failures
        ),
        (1, 0)
    );
    assert_eq!(
        (
            third.total_policy_failures,
            third.consecutive_policy_failures
        ),
        (2, 1)
    );
}

#[test]
fn invalid_policy_output_is_one_recoverable_failure() {
    let mut c = config();
    c.policy_failures.maximum_consecutive = 3;
    c.policy_failures.maximum_total = 20;
    c.policy_failures.behavior = PolicyFailureBehavior::Continue;
    let mut engine = sequenced_engine(
        c,
        Vec::new(),
        SequencedFactory {
            fail_calls: vec![],
            invalid_calls: vec![1],
        },
    );
    engine.run_steps(1).unwrap();
    let policy_events = engine
        .domain_events
        .iter()
        .filter_map(|event| match event {
            DomainEvent::PolicyFailure { category, .. } => Some(category),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(policy_events.len(), 1);
    assert!(matches!(
        policy_events[0],
        midas_core::simulation::PolicyFailureCategory::InvalidOutput
    ));
    assert_eq!(
        engine
            .domain_events
            .iter()
            .filter(|event| matches!(event, DomainEvent::ActionFallback { .. }))
            .count(),
        1
    );
    let state = engine.agents().next().unwrap();
    assert_eq!(state.total_policy_failures, 1);
    assert_eq!(state.consecutive_policy_failures, 1);
}

#[test]
fn stop_experiment_discards_the_entire_failed_step_and_is_idempotent() {
    let checkpoint = tempfile::NamedTempFile::new().unwrap();
    let mut c = config();
    c.policy_failures.maximum_consecutive = 1;
    c.policy_failures.maximum_total = 1;
    c.policy_failures.behavior = PolicyFailureBehavior::StopExperiment;
    c.checkpoint.every_steps = 1;
    c.checkpoint.path = checkpoint.path().display().to_string();
    let mut engine = sequenced_engine(
        c,
        vec![
            CorporateAction::Split {
                asset_id: AssetId(0),
                date: date("2020-01-06"),
                ratio: 2.0,
            },
            CorporateAction::CashDividend {
                asset_id: AssetId(0),
                date: date("2020-01-06"),
                amount_per_share: 1.0,
            },
        ],
        SequencedFactory {
            fail_calls: vec![2],
            invalid_calls: vec![],
        },
    );
    engine.run_steps(1).unwrap();
    let good_agents = engine.agents().cloned().collect::<Vec<_>>();
    let good_events = engine.domain_events.clone();
    let good_metrics = engine.metrics.clone();
    let good_steps = engine.report().steps;
    let checkpoint_bytes = fs::read(checkpoint.path()).unwrap();
    let error = engine.run_steps(1).unwrap_err();
    assert!(error.to_string().contains("sequenced inference failure"));
    assert!(matches!(
        engine.lifecycle(),
        SimulationLifecycle::Failed { .. }
    ));
    assert_eq!(engine.agents().cloned().collect::<Vec<_>>(), good_agents);
    assert_eq!(engine.report().steps, good_steps);
    assert_eq!(fs::read(checkpoint.path()).unwrap(), checkpoint_bytes);
    assert_eq!(engine.metrics.invalid_actions, good_metrics.invalid_actions);
    assert_eq!(engine.metrics.deaths, good_metrics.deaths);
    assert_eq!(
        engine
            .domain_events
            .iter()
            .filter(|event| matches!(event, DomainEvent::ActionFallback { .. }))
            .count(),
        0
    );
    assert!(!engine.domain_events.iter().any(|event| matches!(
        event,
        DomainEvent::CorporateActionApplied { date: event_date, .. }
            if *event_date == date("2020-01-06")
    )));
    assert!(!engine.domain_events.iter().any(|event| matches!(
        event,
        DomainEvent::Valuation { date: event_date, .. }
            if *event_date == date("2020-01-06")
    )));
    let events_after_failure = engine.domain_events.clone();
    let metrics_after_failure = engine.metrics.clone();
    let repeated = engine.run().unwrap_err();
    assert_eq!(repeated.to_string(), error.to_string());
    assert_eq!(engine.domain_events, events_after_failure);
    assert_eq!(engine.metrics, metrics_after_failure);
    let expected_reason = error
        .to_string()
        .strip_prefix("policy error: ")
        .unwrap()
        .to_string();
    assert_eq!(
        engine.report().failure_reason.as_deref(),
        Some(expected_reason.as_str())
    );
    assert_eq!(good_events.len() + 1, engine.domain_events.len());
}

#[test]
fn fatal_agent_policy_failure_dies_before_next_day_actions_without_fallback() {
    let mut c = config();
    c.policy_failures.maximum_consecutive = 1;
    c.policy_failures.maximum_total = 1;
    c.policy_failures.behavior = PolicyFailureBehavior::KillAgent;
    let mut engine = sequenced_engine(
        c,
        vec![
            CorporateAction::Split {
                asset_id: AssetId(0),
                date: date("2020-01-06"),
                ratio: 2.0,
            },
            CorporateAction::CashDividend {
                asset_id: AssetId(0),
                date: date("2020-01-06"),
                amount_per_share: 1.0,
            },
        ],
        SequencedFactory {
            fail_calls: vec![2],
            invalid_calls: vec![],
        },
    );
    engine.run_steps(1).unwrap();
    let before_failure = engine.agents().next().unwrap().clone();
    let expected_final = before_failure
        .portfolio
        .value(&[Some(320.95), Some(50.50), Some(79.80)])
        .unwrap();
    engine.run_steps(1).unwrap();
    let policy_failure_index = engine
        .domain_events
        .iter()
        .position(|event| matches!(event, DomainEvent::PolicyFailure { .. }))
        .unwrap();
    let death_index = engine
        .domain_events
        .iter()
        .position(|event| {
            matches!(
                event,
                DomainEvent::AgentDied {
                    cause: midas_core::simulation::DeathCause::PolicyFailure,
                    ..
                }
            )
        })
        .unwrap();
    assert!(policy_failure_index < death_index);
    assert_eq!(
        engine
            .domain_events
            .iter()
            .filter(|event| matches!(event, DomainEvent::ActionFallback { .. }))
            .count(),
        0
    );
    assert!(!engine.domain_events.iter().any(|event| matches!(
        event,
        DomainEvent::CorporateActionApplied {
            agent_id: AgentId(0),
            date: event_date,
            ..
        } if *event_date == date("2020-01-06")
    )));
    assert!(!engine.domain_events.iter().any(|event| matches!(
        event,
        DomainEvent::Valuation {
            agent_id: AgentId(0),
            date: event_date,
            ..
        } if *event_date == date("2020-01-06")
    )));
    let death = engine
        .domain_events
        .iter()
        .find_map(|event| match event {
            DomainEvent::AgentDied {
                agent_id: AgentId(0),
                final_net_worth,
                age,
                ..
            } => Some((*final_net_worth, *age)),
            _ => None,
        })
        .unwrap();
    assert!((death.0 - expected_final).abs() < 1e-8);
    assert_eq!(death.1, before_failure.age);
    let replacement = engine
        .agents()
        .find(|state| state.id == AgentId(1))
        .unwrap();
    assert_eq!(replacement.total_policy_failures, 0);
    assert_eq!(replacement.consecutive_policy_failures, 0);
}

#[test]
fn stop_experiment_failure_replays_identically_from_last_good_checkpoint() {
    let file = tempfile::NamedTempFile::new().unwrap();
    let mut c = config();
    c.policy_failures.maximum_consecutive = 1;
    c.policy_failures.maximum_total = 1;
    c.policy_failures.behavior = PolicyFailureBehavior::StopExperiment;
    let data = data_with_actions(&mut c, Vec::new());
    let factory = || {
        Box::new(StandardAgentFactory::with_neural_factory(Box::new(
            SequencedFactory {
                fail_calls: vec![2],
                invalid_calls: vec![],
            },
        ))) as Box<dyn AgentFactory>
    };
    let mut first = SimulationEngine::new_with_factory(
        c.clone(),
        data.clone(),
        DatasetSplit::Training,
        factory(),
    )
    .unwrap();
    first.run_steps(1).unwrap();
    first.save_checkpoint(file.path()).unwrap();
    let first_error = first.run_steps(1).unwrap_err().to_string();
    let first_events = first.domain_events.clone();
    let first_states = first.agents().cloned().collect::<Vec<_>>();
    let mut resumed =
        SimulationEngine::resume_with_factory(c, data, file.path(), factory()).unwrap();
    let resumed_error = resumed.run_steps(1).unwrap_err().to_string();
    assert_eq!(first_error, resumed_error);
    assert_eq!(first.lifecycle(), resumed.lifecycle());
    assert_eq!(first.domain_events, resumed.domain_events);
    assert_eq!(first_events, resumed.domain_events);
    assert_eq!(first_states, resumed.agents().cloned().collect::<Vec<_>>());
    assert_eq!(first.report().steps, resumed.report().steps);
}

#[test]
fn engine_corporate_actions_apply_once_across_resume_and_repeat() {
    let checkpoint = tempfile::NamedTempFile::new().unwrap();
    let mut c = config();
    c.brain.policy = PolicyKind::BuyAndHold;
    c.simulation.population_size = 1;
    let data = data_with_actions(
        &mut c,
        vec![
            CorporateAction::Split {
                asset_id: AssetId(0),
                date: date("2020-01-06"),
                ratio: 2.0,
            },
            CorporateAction::CashDividend {
                asset_id: AssetId(0),
                date: date("2020-01-06"),
                amount_per_share: 1.0,
            },
        ],
    );
    let mut engine =
        SimulationEngine::new(c.clone(), data.clone(), DatasetSplit::Training).unwrap();
    engine.run_steps(1).unwrap();
    let before_event = engine.agents().next().unwrap().clone();
    engine.run_steps(1).unwrap();
    let after_event = engine.agents().next().unwrap().clone();
    assert!(after_event.portfolio.positions[0].quantity > 0.0);
    assert!(
        (after_event.portfolio.positions[0].quantity
            - 2.0 * before_event.portfolio.positions[0].quantity)
            .abs()
            < 1e-8
    );
    assert!(
        (after_event.portfolio.cash
            - before_event.portfolio.cash
            - 2.0 * before_event.portfolio.positions[0].quantity
            + 2.0)
            .abs()
            < 1e-8
    );
    assert_eq!(
        engine
            .domain_events
            .iter()
            .filter(|event| matches!(event, DomainEvent::CorporateActionApplied { .. }))
            .count(),
        2
    );
    engine.save_checkpoint(checkpoint.path()).unwrap();
    let mut resumed = SimulationEngine::resume(c, data, checkpoint.path()).unwrap();
    resumed.run().unwrap();
    let completed = resumed.agents().next().unwrap().portfolio.clone();
    resumed.run().unwrap();
    assert_eq!(resumed.agents().next().unwrap().portfolio, completed);
    assert_eq!(
        resumed
            .domain_events
            .iter()
            .filter(|event| matches!(event, DomainEvent::CorporateActionApplied { .. }))
            .count(),
        2
    );
}

#[test]
fn neural_memory_resets_and_genome_validates_full_schema() {
    let architecture = BrainArchitecture {
        version: 1,
        observation_schema_hash: "one".into(),
        observation_size: 2,
        asset_count: 1,
        hidden_layers: vec![2],
        memory: midas_core::brain::MemoryArchitecture { size: 2 },
        actor_output_size: 2,
        critic_output_size: 1,
        activation: midas_core::config::Activation::Relu,
        action_distribution: midas_core::config::ActionDistributionConfig::default(),
    };
    let c = config();
    let genome = BrainGenome::random(&architecture, &c.brain.traits, 1).unwrap();
    let mut changed = architecture.clone();
    changed.observation_schema_hash = "two".into();
    assert!(genome.validate(&changed).is_err());
    let mut invalid_traits = genome.clone();
    invalid_traits.traits.maximum_allocation = 2.0;
    assert!(
        invalid_traits
            .validate_with_traits(&architecture, &c.brain.traits)
            .is_err()
    );
    let mut model = MockModel {
        schema: "one".into(),
        size: 2,
        memory: 5.0,
        fail: false,
    };
    model.reset_memory();
    let NeuralModelState::External { state, .. } = model.snapshot_state().unwrap() else {
        panic!()
    };
    assert_eq!(state[0], 0.0);
}

#[test]
fn no_hardcoded_observation_formula_remains() {
    let source = include_str!("../src/simulation.rs");
    assert!(!source.contains("assets * 2 + 5"));
    assert!(!source.contains("market + assets"));
}

#[test]
fn earlier_observations_do_not_change_with_extreme_future_price() {
    let c = config();
    let pipeline = BackwardFeaturePipeline::from_config(&c);
    let a = (0..5)
        .map(|i| bar(0, &format!("2020-01-0{}", i + 1), 10.0 + i as f64))
        .collect::<Vec<_>>();
    let mut b = a.clone();
    b[4] = bar(0, "2020-01-05", 10_000.0);
    let ca = CorporateActionConfig {
        mode: CorporateActionMode::ExplicitEvents,
        discontinuity_threshold: 0.99,
        ..Default::default()
    };
    let da = MarketDataset::new_with_actions(vec!["A".into()], a, vec![], &ca).unwrap();
    let db = MarketDataset::new_with_actions(
        vec!["A".into()],
        b,
        vec![CorporateAction::Split {
            asset_id: AssetId(0),
            date: date("2020-01-05"),
            ratio: 0.0014,
        }],
        &ca,
    )
    .unwrap();
    let dates = da.dates[..4].to_vec();
    assert_eq!(
        pipeline
            .generate(&da, &dates)
            .unwrap()
            .iter()
            .map(|r| r.values.clone())
            .collect::<Vec<_>>(),
        pipeline
            .generate(&db, &dates)
            .unwrap()
            .iter()
            .map(|r| r.values.clone())
            .collect::<Vec<_>>()
    );
}

#[test]
fn malformed_ohlc_and_duplicates_are_rejected() {
    assert!(
        MarketDataset::new(
            vec!["A".into()],
            vec![MarketBar {
                high: 5.0,
                ..bar(0, "2020-01-01", 10.0)
            }]
        )
        .is_err()
    );
    let value = bar(0, "2020-01-01", 10.0);
    assert!(MarketDataset::new(vec!["A".into()], vec![value.clone(), value]).is_err());
}

proptest! {
    #[test]
    fn planned_and_applied_allocations_never_make_balances_negative(weight in 0.0f64..1.0) {
        let c = config(); let engine = CostExecutionEngine { costs: c.execution }; let mut portfolio = Portfolio::new(1000.0, 1); let market_bar = bar(0, "2020-01-01", 10.0);
        let action = validator(vec![1.0 - weight, weight], &[true], 1.0);
        let plan = engine.plan(&portfolio, &action, &ExecutionSnapshot { bars: &[Some(&market_bar)], valuation_prices: &[Some(10.0)] }, 0.0).unwrap();
        engine.apply(&mut portfolio, &plan).unwrap();
        prop_assert!(portfolio.cash >= 0.0); prop_assert!(portfolio.positions[0].quantity >= 0.0);
    }
}

#[test]
fn observation_hash_includes_market_version_and_normalization() {
    let c = config();
    let original = ObservationSchema::from_config(&c)
        .identity(c.market.assets.len())
        .unwrap();
    let mut changed = c.clone();
    changed.features.clip *= 2.0;
    assert_ne!(
        original,
        ObservationSchema::from_config(&changed)
            .identity(changed.market.assets.len())
            .unwrap()
    );
    changed = c.clone();
    changed.features.schema_version += 1;
    assert_ne!(
        original,
        ObservationSchema::from_config(&changed)
            .identity(changed.market.assets.len())
            .unwrap()
    );
}

#[test]
fn independent_agents_and_buy_hold_behavior_are_preserved() {
    let mut c = config();
    c.brain.policy = PolicyKind::Random;
    c.simulation.population_size = 2;
    let data = loaded(&c);
    let mut engine = SimulationEngine::new(c, data, DatasetSplit::Training).unwrap();
    let report = engine.run().unwrap();
    assert_eq!(report.agents.len(), 2);
    assert_ne!(report.agents[0].0, report.agents[1].0);
    let mut c = config();
    c.brain.policy = PolicyKind::BuyAndHold;
    c.simulation.population_size = 1;
    let data = loaded(&c);
    let mut engine = SimulationEngine::new(c, data, DatasetSplit::Test).unwrap();
    let report = engine.run().unwrap();
    assert!(report.agents[0].1.trades > 0);
    assert!(report.agents[0].1.ending_net_worth > 10_000.0);
}

#[test]
fn buy_and_hold_is_fully_invested_even_when_neural_cap_is_half() {
    let mut c = config();
    c.brain.policy = PolicyKind::BuyAndHold;
    c.brain.traits.maximum_allocation.initial = 0.5;
    c.simulation.population_size = 1;
    let data = loaded(&c);
    let valuation_date = data.dates[1];
    let prices = (0..data.assets.len())
        .map(|asset| {
            data.bar(valuation_date, AssetId(asset as u64))
                .map(|bar| bar.close)
        })
        .collect::<Vec<_>>();
    let mut engine = SimulationEngine::new(c, data, DatasetSplit::Training).unwrap();
    engine.run_steps(1).unwrap();
    let state = engine.agents().next().unwrap();
    let allocations = state.portfolio.allocations(&prices).unwrap();
    assert!(
        allocations[1] > 0.9,
        "buy-and-hold should target full initial exposure, got {allocations:?}"
    );
}

#[test]
fn equal_weight_benchmark_ignores_neural_concentration_cap() {
    let mut c = config();
    c.brain.policy = PolicyKind::EqualWeight;
    c.brain.traits.maximum_allocation.initial = 0.25;
    c.simulation.population_size = 1;
    let data = loaded(&c);
    let valuation_date = data.dates[1];
    let prices = (0..data.assets.len())
        .map(|asset| {
            data.bar(valuation_date, AssetId(asset as u64))
                .map(|bar| bar.close)
        })
        .collect::<Vec<_>>();
    let mut engine = SimulationEngine::new(c, data, DatasetSplit::Training).unwrap();
    engine.run_steps(1).unwrap();
    let allocations = engine
        .agents()
        .next()
        .unwrap()
        .portfolio
        .allocations(&prices)
        .unwrap();
    assert!(allocations[1..].iter().all(|weight| *weight > 0.30));
    assert!(allocations[0] < 0.1);
}

#[test]
fn burn_model_is_deterministic_and_memory_is_checkpointed() {
    let c = config();
    let schema = ObservationSchema::from_config(&c);
    let hash = schema.identity(c.market.assets.len()).unwrap();
    let architecture = BrainArchitecture::from_config(
        &c.brain,
        hash.clone(),
        schema.dimension(c.market.assets.len()),
        c.market.assets.len(),
    );
    let genome = BrainGenome::random(&architecture, &c.brain.traits, 7).unwrap();
    let genome_before = genome.clone();
    let factory = BurnNeuralModelFactory;
    let mut first = factory
        .build(
            &genome,
            &architecture,
            &ComputeDevice::CpuNdArray {
                device: "default".into(),
            },
        )
        .unwrap();
    let mut second = factory
        .build(
            &genome,
            &architecture,
            &ComputeDevice::CpuNdArray {
                device: "default".into(),
            },
        )
        .unwrap();
    let observation = Observation {
        schema_hash: hash,
        values: vec![0.1; architecture.observation_size],
    };
    let a = first
        .forward(&NeuralInput {
            observation: &observation,
        })
        .unwrap();
    let b = second
        .forward(&NeuralInput {
            observation: &observation,
        })
        .unwrap();
    assert_eq!(a, b);
    assert_eq!(
        first.snapshot_state().unwrap(),
        second.snapshot_state().unwrap()
    );
    first.reset_memory();
    assert_ne!(
        first.snapshot_state().unwrap(),
        second.snapshot_state().unwrap()
    );
    assert_eq!(first.runtime_parameter_builds(), 1);
    assert_eq!(second.runtime_parameter_builds(), 1);
    for _ in 0..32 {
        first
            .forward(&NeuralInput {
                observation: &observation,
            })
            .unwrap();
    }
    assert_eq!(genome, genome_before);
    assert_eq!(first.runtime_parameter_builds(), 1);
}

#[test]
fn survival_liability_remains_in_net_worth() {
    let mut portfolio = Portfolio::new(1.0, 1);
    portfolio.apply_survival_cost(2.0);
    assert_eq!(portfolio.cash, 0.0);
    assert_eq!(portfolio.liabilities, 1.0);
    assert_eq!(portfolio.value(&[Some(10.0)]).unwrap(), -1.0);
}
