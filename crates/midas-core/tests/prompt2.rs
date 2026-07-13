use midas_core::{
    agent::{AgentBuildContext, AgentFactory, StandardAgentFactory},
    brain::PolicyBlueprint,
    config::Config,
    market::{MarketDataSource, PolarsDataSource},
    replacement::{
        EvolutionaryReplacementStrategy, LivingAgentSummary, ReplacementContext,
        ReplacementStrategy,
    },
    simulation::{DatasetSplit, SimulationEngine, SimulationLifecycle},
};
use std::path::PathBuf;

fn prompt2_config() -> Config {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut config = Config::load(root.join("configs/prompt2.toml")).unwrap();
    config.data.path = root.join("data/raw/synthetic.csv").to_string_lossy().into();
    config
}

#[test]
fn ppo_learning_and_checkpoint_resume_are_deterministic() {
    let config = prompt2_config();
    let data = PolarsDataSource.load(&config).unwrap();
    let mut uninterrupted =
        SimulationEngine::new(config.clone(), data.clone(), DatasetSplit::Training).unwrap();
    let uninterrupted_report = uninterrupted.run().unwrap();
    assert_eq!(uninterrupted_report.learning_implementation, "ppo");
    assert!(uninterrupted_report.learning_updates > 0);

    let mut partial =
        SimulationEngine::new(config.clone(), data.clone(), DatasetSplit::Training).unwrap();
    partial.run_steps(3).unwrap();
    let checkpoint = tempfile::NamedTempFile::new().unwrap();
    partial.save_checkpoint(checkpoint.path()).unwrap();
    let mut resumed = SimulationEngine::resume(config, data, checkpoint.path()).unwrap();
    let resumed_report = resumed.run().unwrap();
    assert_eq!(uninterrupted_report.steps, resumed_report.steps);
    assert_eq!(
        uninterrupted_report.learning_updates,
        resumed_report.learning_updates
    );
    assert_eq!(uninterrupted.domain_events, resumed.domain_events);
    assert_eq!(
        uninterrupted.agents().collect::<Vec<_>>(),
        resumed.agents().collect::<Vec<_>>()
    );
    assert_eq!(resumed.lifecycle(), SimulationLifecycle::Completed);
    let before = resumed.domain_events.len();
    resumed.run().unwrap();
    assert_eq!(before, resumed.domain_events.len());
}

#[test]
fn automatic_checkpoint_is_written_from_committed_candidate_state() {
    let mut config = prompt2_config();
    let directory = tempfile::tempdir().unwrap();
    config.checkpoint.every_steps = 1;
    config.checkpoint.path = directory.path().join("automatic.bin").display().to_string();
    let data = PolarsDataSource.load(&config).unwrap();
    let mut engine = SimulationEngine::new(config, data, DatasetSplit::Training).unwrap();
    engine.run_steps(2).unwrap();
    assert!(directory.path().join("automatic.bin").exists());
    assert_eq!(engine.lifecycle(), SimulationLifecycle::Running);
    assert!(
        engine
            .operational_events
            .iter()
            .filter(|event| matches!(
                event,
                midas_core::simulation::OperationalEvent::CheckpointSaved { .. }
            ))
            .count()
            >= 2
    );
}

#[test]
fn evolutionary_replacement_selects_two_parents_and_reports_mutation() {
    let config = prompt2_config();
    let data = PolarsDataSource.load(&config).unwrap();
    let engine = SimulationEngine::new(config.clone(), data, DatasetSplit::Training).unwrap();
    let dead = engine.agents().next().unwrap().clone();
    let living = engine
        .agents()
        .map(|state| LivingAgentSummary {
            id: state.id,
            net_worth: state.portfolio.net_worth,
            age: state.age,
            policy: PolicyBlueprint::Neural,
            genome: state.genome.as_ref(),
            maximum_drawdown: state.maximum_drawdown,
            birth_date: state.birth_date,
            parent_ids: state.parent_ids.clone(),
        })
        .collect::<Vec<_>>();
    let architecture = midas_core::brain::BrainArchitecture::from_config(
        &config.brain,
        engine
            .observation_schema()
            .identity(config.market.assets.len())
            .unwrap(),
        engine
            .observation_schema()
            .dimension(config.market.assets.len()),
        config.market.assets.len(),
    );
    let device = config.backend.compute_device().unwrap();
    let schema = engine.observation_schema().clone();
    let mut strategy = EvolutionaryReplacementStrategy::default();
    let request = strategy
        .create_replacement(&ReplacementContext {
            dead_agent: &dead,
            dead_policy: PolicyBlueprint::Neural,
            living_population: &living,
            config: &config,
            architecture: &architecture,
            compute_device: &device,
            observation_schema: &schema,
            deterministic_seed: 9,
            completed_lives: &[],
            genome_registry: &std::collections::BTreeMap::new(),
        })
        .unwrap();
    assert_eq!(request.parent_ids.len(), 2);
    assert!(strategy.mutation_report().is_some());
    let factory = StandardAgentFactory::default();
    let child = factory
        .spawn(
            request,
            &AgentBuildContext {
                agent_id: midas_core::AgentId(99),
                birth_date: dead.birth_date,
                config: &config,
                architecture: &architecture,
                compute_device: &device,
            },
        )
        .unwrap();
    assert_eq!(child.policy.policy_blueprint(), PolicyBlueprint::Neural);
    assert_eq!(child.state.total_policy_failures, 0);
    assert_eq!(child.state.consecutive_policy_failures, 0);
}
