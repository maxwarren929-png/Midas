use super::*;

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
