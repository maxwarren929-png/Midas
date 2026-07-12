use chrono::NaiveDate;
use midas_core::{
    AssetId,
    brain::{BrainGenome, BurnActorCriticPolicy, Policy, PolicyInput},
    config::Config,
    execution::{CostExecutionModel, ExecutionModel, Portfolio},
    market::{
        BackwardFeaturePipeline, FeaturePipeline, MarketBar, MarketDataSource, MarketDataset,
        PolarsDataSource, write_parquet,
    },
    simulation::{DatasetSplit, SimulationEngine, SimulationEvent},
};
use proptest::prelude::*;
use std::path::PathBuf;

fn config() -> Config {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut value = Config::load(root.join("configs/test.toml")).unwrap();
    value.data.path = root.join("data/raw/synthetic.csv").to_string_lossy().into();
    value
}

fn bar(date: &str, open: f64) -> MarketBar {
    MarketBar {
        date: NaiveDate::parse_from_str(date, "%Y-%m-%d").unwrap(),
        asset_id: AssetId(0),
        open,
        high: open + 1.0,
        low: open - 1.0,
        close: open + 0.25,
        adjusted_close: open + 0.25,
        volume: 1000.0,
        tradable: true,
    }
}

#[test]
fn config_and_csv_are_valid() {
    let c = config();
    c.validate().unwrap();
    let d = PolarsDataSource.load(&c).unwrap();
    assert_eq!(d.assets.len(), 3);
    assert_eq!(d.dates.len(), 24);
}

#[test]
fn parquet_round_trip_uses_the_canonical_schema() {
    let mut c = config();
    let data = PolarsDataSource.load(&c).unwrap();
    let file = tempfile::NamedTempFile::new().unwrap();
    write_parquet(&data, file.path()).unwrap();
    c.data.path = file.path().to_string_lossy().into();
    c.data.format = midas_core::config::DataFormat::Parquet;
    let loaded = PolarsDataSource.load(&c).unwrap();
    assert_eq!(loaded.bars, data.bars);
    assert_eq!(loaded.identity, data.identity);
}

#[test]
fn rejects_bad_ohlc_and_duplicates() {
    assert!(
        MarketDataset::new(
            vec!["A".into()],
            vec![MarketBar {
                high: 5.0,
                ..bar("2020-01-01", 10.0)
            }]
        )
        .is_err()
    );
    let b = bar("2020-01-01", 10.0);
    assert!(MarketDataset::new(vec!["A".into()], vec![b.clone(), b]).is_err());
}

#[test]
fn features_are_backward_only() {
    let c = config();
    let pipeline = BackwardFeaturePipeline::from_config(&c);
    let a = (0..5)
        .map(|i| bar(&format!("2020-01-0{}", i + 1), 10.0 + i as f64))
        .collect::<Vec<_>>();
    let mut b = a.clone();
    b[4] = bar("2020-01-05", 10_000.0);
    let da = MarketDataset::new(vec!["A".into()], a).unwrap();
    let db = MarketDataset::new(vec!["A".into()], b).unwrap();
    let early = da.dates[..4].to_vec();
    let fa = pipeline
        .generate(&da, &early)
        .unwrap()
        .into_iter()
        .map(|r| r.values)
        .collect::<Vec<_>>();
    let fb = pipeline
        .generate(&db, &early)
        .unwrap()
        .into_iter()
        .map(|r| r.values)
        .collect::<Vec<_>>();
    assert_eq!(fa, fb);
}

#[test]
fn burn_inference_is_deterministic_and_stateful() {
    let c = config();
    let obs = c.market.assets.len() * c.features.features.len() + c.market.assets.len() * 2 + 5;
    let genome = BrainGenome::random(&c.brain, obs, c.market.assets.len(), 7).unwrap();
    let original = genome.clone();
    let mut p1 = BurnActorCriticPolicy::new(genome.clone()).unwrap();
    let mut p2 = BurnActorCriticPolicy::new(genome).unwrap();
    let values = vec![0.1; obs];
    let mask = vec![true; c.market.assets.len()];
    let allocations = vec![1.0, 0.0, 0.0, 0.0];
    let a = p1
        .act(PolicyInput {
            observation: &values,
            tradable: &mask,
            current_allocations: &allocations,
        })
        .unwrap();
    let b = p2
        .act(PolicyInput {
            observation: &values,
            tradable: &mask,
            current_allocations: &allocations,
        })
        .unwrap();
    assert_eq!(a, b);
    assert!((a.allocations.iter().sum::<f64>() - 1.0).abs() < 1e-8);
    assert!(a.allocations.iter().all(|x| *x >= 0.0));
    assert_ne!(p1.lifetime.memory, vec![0.0; c.brain.memory_size]);
    assert_eq!(p1.genome, original);
}

#[test]
fn invalid_observation_falls_back_to_cash() {
    let c = config();
    let g = BrainGenome::random(&c.brain, 10, 1, 1).unwrap();
    let mut p = BurnActorCriticPolicy::new(g).unwrap();
    let out = p
        .act(PolicyInput {
            observation: &[f32::NAN; 10],
            tradable: &[true],
            current_allocations: &[1.0, 0.0],
        })
        .unwrap();
    assert_eq!(out.allocations, vec![1.0, 0.0]);
    assert!(!out.valid);
}

#[test]
fn execution_charges_costs_and_keeps_invariants() {
    let c = config();
    let model = CostExecutionModel { costs: c.execution };
    let mut p = Portfolio::new(1000.0, 1);
    let b = bar("2020-01-02", 10.0);
    let report = model
        .rebalance(&mut p, &[0.0, 1.0], &[Some(&b)], &[Some(10.0)], 1.0)
        .unwrap();
    assert_eq!(report.trades.len(), 1);
    assert!(p.cash >= 0.0);
    assert!(p.positions[0].quantity > 0.0);
    assert!(p.cumulative_commission > 0.0);
    assert!(p.cumulative_spread > 0.0);
    assert!(p.cumulative_slippage > 0.0);
}

#[test]
fn survival_cost_becomes_a_persistent_liability_and_can_bankrupt() {
    let mut p = Portfolio::new(1.0, 1);
    p.apply_survival_cost(2.0);
    assert_eq!(p.cash, 0.0);
    assert_eq!(p.liabilities, 1.0);
    assert_eq!(p.value(&[Some(10.0)]).unwrap(), -1.0);
}

proptest! {
    #[test]
    fn allocation_execution_never_makes_cash_or_holdings_negative(weight in 0.0f64..1.0) {
        let c = config(); let model = CostExecutionModel { costs: c.execution }; let mut p = Portfolio::new(1000.0, 1); let b = bar("2020-01-02", 10.0);
        model.rebalance(&mut p, &[1.0 - weight, weight], &[Some(&b)], &[Some(10.0)], 0.0).unwrap();
        prop_assert!(p.cash >= 0.0); prop_assert!(p.positions[0].quantity >= 0.0);
    }
}

#[test]
fn timing_and_checkpoint_resume_are_deterministic() {
    let c = config();
    let d = PolarsDataSource.load(&c).unwrap();
    let mut full = SimulationEngine::new(c.clone(), d.clone(), DatasetSplit::Training).unwrap();
    let full_report = full.run().unwrap();
    let mut partial = SimulationEngine::new(c.clone(), d.clone(), DatasetSplit::Training).unwrap();
    partial.run_steps(3).unwrap();
    let file = tempfile::NamedTempFile::new().unwrap();
    partial.save_checkpoint(file.path()).unwrap();
    let mut resumed = SimulationEngine::resume(c, d, file.path()).unwrap();
    let resumed_report = resumed.run().unwrap();
    assert_eq!(full_report.steps, resumed_report.steps);
    assert_eq!(full_report.agents, resumed_report.agents);
    let revealed = full
        .events
        .iter()
        .find_map(|e| {
            if let SimulationEvent::ObservationRevealed { date } = e {
                Some(*date)
            } else {
                None
            }
        })
        .unwrap();
    let traded = full
        .events
        .iter()
        .find_map(|e| {
            if let SimulationEvent::TradeExecuted { date, .. } = e {
                Some(*date)
            } else {
                None
            }
        })
        .unwrap();
    assert!(traded > revealed);
}

#[test]
fn independent_agents_have_independent_metrics() {
    let mut c = config();
    c.brain.policy = midas_core::config::PolicyKind::Random;
    c.simulation.population_size = 2;
    let d = PolarsDataSource.load(&c).unwrap();
    let mut engine = SimulationEngine::new(c, d, DatasetSplit::Training).unwrap();
    let report = engine.run().unwrap();
    assert_eq!(report.agents.len(), 2);
    assert_ne!(report.agents[0].0, report.agents[1].0);
}

#[test]
fn buy_and_hold_benchmark_enters_the_configured_index() {
    let mut c = config();
    c.brain.policy = midas_core::config::PolicyKind::BuyAndHold;
    c.simulation.population_size = 1;
    let d = PolarsDataSource.load(&c).unwrap();
    let mut engine = SimulationEngine::new(c, d, DatasetSplit::Test).unwrap();
    let report = engine.run().unwrap();
    assert!(report.agents[0].1.trades > 0);
    assert!(report.agents[0].1.ending_net_worth > 10_000.0);
}
