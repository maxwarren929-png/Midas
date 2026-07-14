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
    for state in uninterrupted.agents() {
        assert_eq!(
            uninterrupted.agent_lifetime_parameters(state.id).unwrap(),
            resumed.agent_lifetime_parameters(state.id).unwrap()
        );
        assert_eq!(
            uninterrupted.agent_policy_snapshot(state.id).unwrap(),
            resumed.agent_policy_snapshot(state.id).unwrap()
        );
    }
    assert_eq!(resumed.lifecycle(), SimulationLifecycle::Completed);
    let before = resumed.domain_events.len();
    resumed.run().unwrap();
    assert_eq!(before, resumed.domain_events.len());
}

#[test]
fn prompt2_ppo_updates_full_heads_with_a_frozen_encoder() {
    let config = prompt2_config();
    let data = PolarsDataSource.load(&config).unwrap();
    let mut engine = SimulationEngine::new(config, data, DatasetSplit::Training).unwrap();
    let agent_id = engine.agents().next().unwrap().id;
    let before = engine
        .agent_lifetime_parameters(agent_id)
        .unwrap()
        .expect("neural agent lifetime parameters");
    let birth_genome = engine
        .agents()
        .find(|state| state.id == agent_id)
        .unwrap()
        .genome
        .clone()
        .unwrap();

    engine.run_steps(4).unwrap();

    let after = engine
        .agent_lifetime_parameters(agent_id)
        .unwrap()
        .expect("neural agent lifetime parameters");
    assert_ne!(before.actor.weights, after.actor.weights);
    assert_ne!(before.actor.bias, after.actor.bias);
    assert_ne!(before.critic.weights, after.critic.weights);
    assert_ne!(before.critic.bias, after.critic.bias);
    assert_eq!(before.hidden, after.hidden);
    assert_eq!(before.recurrent, after.recurrent);
    assert_eq!(
        birth_genome,
        engine
            .agents()
            .find(|state| state.id == agent_id)
            .unwrap()
            .genome
            .clone()
            .unwrap()
    );
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

#[test]
fn mating_distance_preserves_configured_selection_pressure() {
    let mut unrestricted = prompt2_config();
    unrestricted.evolution.diversity.minimum_mating_distance = 0.0;
    unrestricted.evolution.selection.tournament_size = 3;
    unrestricted.simulation.population_size = 4;
    let data = PolarsDataSource.load(&unrestricted).unwrap();
    let engine = SimulationEngine::new(unrestricted.clone(), data, DatasetSplit::Training).unwrap();
    let dead = engine.agents().next().unwrap().clone();
    let living = engine
        .agents()
        .skip(1)
        .enumerate()
        .map(|(index, state)| LivingAgentSummary {
            id: state.id,
            net_worth: 10_000.0 + (index as f64) * 10.0,
            age: state.age,
            policy: PolicyBlueprint::Neural,
            genome: state.genome.as_ref(),
            maximum_drawdown: state.maximum_drawdown,
            birth_date: state.birth_date,
            parent_ids: state.parent_ids.clone(),
        })
        .collect::<Vec<_>>();
    let schema = engine.observation_schema().clone();
    let architecture = midas_core::brain::BrainArchitecture::from_config(
        &unrestricted.brain,
        schema.identity(unrestricted.market.assets.len()).unwrap(),
        schema.dimension(unrestricted.market.assets.len()),
        unrestricted.market.assets.len(),
    );
    let device = unrestricted.backend.compute_device().unwrap();
    let registry = std::collections::BTreeMap::new();
    let baseline_context = ReplacementContext {
        dead_agent: &dead,
        dead_policy: PolicyBlueprint::Neural,
        living_population: &living,
        config: &unrestricted,
        architecture: &architecture,
        compute_device: &device,
        observation_schema: &schema,
        deterministic_seed: 19,
        completed_lives: &[],
        genome_registry: &registry,
    };
    let mut baseline = EvolutionaryReplacementStrategy::default();
    let baseline_request = baseline.create_replacement(&baseline_context).unwrap();

    let mut constrained = unrestricted.clone();
    constrained.evolution.diversity.minimum_mating_distance = 0.0001;
    let constrained_context = ReplacementContext {
        dead_agent: &dead,
        dead_policy: PolicyBlueprint::Neural,
        living_population: &living,
        config: &constrained,
        architecture: &architecture,
        compute_device: &device,
        observation_schema: &schema,
        deterministic_seed: 19,
        completed_lives: &[],
        genome_registry: &registry,
    };
    let mut constrained_strategy = EvolutionaryReplacementStrategy::default();
    let constrained_request = constrained_strategy
        .create_replacement(&constrained_context)
        .unwrap();

    assert_eq!(
        baseline_request.parent_ids, constrained_request.parent_ids,
        "mating distance must filter the second-parent pool without replacing selector pressure"
    );
}
