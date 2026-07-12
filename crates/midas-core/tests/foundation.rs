use chrono::NaiveDate;
use midas_core::{
    AssetId, MidasError,
    action::{
        ActionValidationContext, ActionValidator, LongOnlyActionValidator, ProposedAllocation,
    },
    agent::StandardAgentFactory,
    brain::{
        BrainArchitecture, BrainGenome, BurnNeuralModelFactory, ComputeDevice, NeuralInput,
        NeuralModel, NeuralModelFactory, NeuralModelState, NeuralOutput,
    },
    config::{
        Config, CorporateActionConfig, CorporateActionMode, DeathHandling, ObservationFeatureKind,
        PolicyKind,
    },
    execution::{
        CostExecutionEngine, ExecutionApplier, ExecutionPlanner, ExecutionSnapshot, Portfolio,
        TradeSide,
    },
    market::{
        BackwardFeaturePipeline, CorporateAction, FeaturePipeline, MarketBar, MarketDataSource,
        MarketDataset, PolarsDataSource, write_parquet,
    },
    observation::{
        Observation, ObservationBuilder, ObservationContext, ObservationSchema,
        SchemaObservationBuilder,
    },
    simulation::{DatasetSplit, SimulationEngine, SimulationLifecycle},
};
use proptest::prelude::*;
use std::path::PathBuf;

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
                maximum_asset_allocation: max,
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
                    maximum_asset_allocation: 0.25,
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

struct MockModel {
    schema: String,
    size: usize,
    memory: f32,
}
impl NeuralModel for MockModel {
    fn forward(&mut self, input: &NeuralInput<'_>) -> midas_core::Result<NeuralOutput> {
        assert_eq!(input.observation.schema_hash, self.schema);
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
}
struct MockFactory;
impl NeuralModelFactory for MockFactory {
    fn build(
        &self,
        _: &BrainGenome,
        a: &BrainArchitecture,
        _: &ComputeDevice,
    ) -> midas_core::Result<Box<dyn NeuralModel>> {
        Ok(Box::new(MockModel {
            schema: a.observation_schema_hash.clone(),
            size: a.actor_output_size,
            memory: 0.0,
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
            return Err(MidasError::Policy("wrong mock state".into()));
        };
        Ok(Box::new(MockModel {
            schema: a.observation_schema_hash.clone(),
            size: a.actor_output_size,
            memory: state[0],
        }))
    }
}
#[test]
fn simulation_runs_mock_and_burn_neural_models() {
    let c = config();
    let data = loaded(&c);
    let factory = StandardAgentFactory::with_neural_factory(Box::new(MockFactory));
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
    let factory = BurnNeuralModelFactory;
    let mut first = factory
        .build(&genome, &architecture, &ComputeDevice::Cpu)
        .unwrap();
    let mut second = factory
        .build(&genome, &architecture, &ComputeDevice::Cpu)
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
}

#[test]
fn survival_liability_remains_in_net_worth() {
    let mut portfolio = Portfolio::new(1.0, 1);
    portfolio.apply_survival_cost(2.0);
    assert_eq!(portfolio.cash, 0.0);
    assert_eq!(portfolio.liabilities, 1.0);
    assert_eq!(portfolio.value(&[Some(10.0)]).unwrap(), -1.0);
}
