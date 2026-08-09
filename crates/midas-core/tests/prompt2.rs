use midas_core::{
    agent::{AgentBuildContext, AgentFactory, StandardAgentFactory},
    brain::PolicyBlueprint,
    config::Config,
    evolution::FitnessScore,
    market::{MarketDataSource, PolarsDataSource},
    replacement::{
        EvolutionReplacementOutcome, EvolutionaryReplacementStrategy, LivingAgentSummary,
        ReplacementContext, ReplacementStrategy,
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
    assert_eq!(
        uninterrupted.domain_events.len(),
        resumed.domain_events.len()
    );
    for (index, (left, right)) in uninterrupted
        .domain_events
        .iter()
        .zip(&resumed.domain_events)
        .enumerate()
    {
        assert_eq!(left, right, "domain event mismatch at index {index}");
    }
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
fn frozen_evaluation_reuses_trained_checkpoint_without_learning() {
    let config = prompt2_config();
    let data = PolarsDataSource.load(&config).unwrap();
    let mut training =
        SimulationEngine::new(config.clone(), data.clone(), DatasetSplit::Training).unwrap();
    training.run().unwrap();
    let checkpoint = tempfile::NamedTempFile::new().unwrap();
    training.save_checkpoint(checkpoint.path()).unwrap();

    let mut evaluation = SimulationEngine::evaluate_checkpoint(
        config,
        data,
        checkpoint.path(),
        DatasetSplit::Validation,
    )
    .unwrap();
    let report = evaluation.run().unwrap();
    assert_eq!(report.split, DatasetSplit::Validation);
    assert_eq!(report.learning_implementation, "no_learning");
    assert_eq!(report.learning_updates, 0);
    assert_eq!(report.lifecycle, SimulationLifecycle::Completed);
}

#[test]
fn prompt2_ppo_updates_hidden_recurrent_and_head_parameters() {
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
    assert_ne!(before.hidden, after.hidden);
    assert_ne!(before.recurrent, after.recurrent);
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
fn prompt3a_multi_asset_portfolios_and_resume_are_deterministic() {
    let config = prompt2_config();
    let data = PolarsDataSource.load(&config).unwrap();
    let mut uninterrupted =
        SimulationEngine::new(config.clone(), data.clone(), DatasetSplit::Training).unwrap();
    let first_agent = uninterrupted.agents().next().unwrap().id;
    let before = uninterrupted
        .agent_lifetime_parameters(first_agent)
        .unwrap()
        .unwrap();
    let report = uninterrupted.run().unwrap();

    assert_eq!(report.lifecycle, SimulationLifecycle::Completed);
    assert!(report.neural_action_count > 0);
    assert!(report.neural_multi_asset_action_fraction > 0.0);
    assert!(report.neural_mean_active_positions > 1.0);
    assert_ne!(
        before.actor.weights,
        uninterrupted
            .agent_lifetime_parameters(first_agent)
            .unwrap()
            .unwrap()
            .actor
            .weights
    );

    let mut partial =
        SimulationEngine::new(config.clone(), data.clone(), DatasetSplit::Training).unwrap();
    partial.run_steps(8).unwrap();
    let checkpoint = tempfile::NamedTempFile::new().unwrap();
    partial.save_checkpoint(checkpoint.path()).unwrap();
    let mut resumed = SimulationEngine::resume(config, data, checkpoint.path()).unwrap();
    let resumed_report = resumed.run().unwrap();

    assert_eq!(report.steps, resumed_report.steps);
    assert_eq!(
        report.neural_multi_asset_action_fraction,
        resumed_report.neural_multi_asset_action_fraction
    );
    assert_eq!(
        report.neural_mean_active_positions,
        resumed_report.neural_mean_active_positions
    );
    assert_eq!(
        uninterrupted.domain_events.len(),
        resumed.domain_events.len()
    );
    for (index, (left, right)) in uninterrupted
        .domain_events
        .iter()
        .zip(&resumed.domain_events)
        .enumerate()
    {
        assert_eq!(left, right, "domain event mismatch at index {index}");
    }
    assert_eq!(
        uninterrupted.agents().collect::<Vec<_>>(),
        resumed.agents().collect::<Vec<_>>()
    );
    assert_eq!(
        uninterrupted
            .agent_lifetime_parameters(first_agent)
            .unwrap(),
        resumed.agent_lifetime_parameters(first_agent).unwrap()
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
            selection_fitness: FitnessScore(state.portfolio.net_worth),
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
            replacement: &config.replacement,
            evolution: &config.evolution,
            evolvable_traits: &config.brain.traits,
            architecture: &architecture,
            compute_device: &device,
            observation_schema: &schema,
            deterministic_seed: 9,
            seed_derivation: Default::default(),
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
fn evolutionary_fallbacks_report_typed_outcomes_and_correct_parentage() {
    let mut config = prompt2_config();
    let data = PolarsDataSource.load(&config).unwrap();
    let engine = SimulationEngine::new(config.clone(), data, DatasetSplit::Training).unwrap();
    let states = engine.agents().collect::<Vec<_>>();
    let dead = states[0].clone();
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
    let registry = std::collections::BTreeMap::new();

    config.evolution.fallback = midas_core::config::EvolutionFallback::CloneBestEligible;
    let one_living = vec![LivingAgentSummary {
        id: states[1].id,
        net_worth: states[1].portfolio.net_worth,
        selection_fitness: FitnessScore(states[1].portfolio.net_worth),
        age: states[1].age,
        policy: PolicyBlueprint::Neural,
        genome: states[1].genome.as_ref(),
        maximum_drawdown: states[1].maximum_drawdown,
        birth_date: states[1].birth_date,
        parent_ids: states[1].parent_ids.clone(),
    }];
    let mut clone_strategy = EvolutionaryReplacementStrategy::default();
    let clone_request = clone_strategy
        .create_replacement(&ReplacementContext {
            dead_agent: &dead,
            dead_policy: PolicyBlueprint::Neural,
            living_population: &one_living,
            replacement: &config.replacement,
            evolution: &config.evolution,
            evolvable_traits: &config.brain.traits,
            architecture: &architecture,
            compute_device: &device,
            observation_schema: &schema,
            deterministic_seed: 23,
            seed_derivation: Default::default(),
            completed_lives: &[],
            genome_registry: &registry,
        })
        .unwrap();
    assert_eq!(clone_request.parent_ids, vec![states[1].id]);
    let clone_report = clone_strategy.evolution_report().unwrap();
    assert_eq!(
        clone_report.outcome,
        EvolutionReplacementOutcome::CloneFallback
    );
    assert!(!clone_report.crossover);
    assert!(clone_report.mutation.is_none());

    config.evolution.selection.minimum_age = 1;
    let mut ineligible_clone_strategy = EvolutionaryReplacementStrategy::default();
    assert_eq!(
        ineligible_clone_strategy
            .create_replacement(&ReplacementContext {
                dead_agent: &dead,
                dead_policy: PolicyBlueprint::Neural,
                living_population: &one_living,
                replacement: &config.replacement,
                evolution: &config.evolution,
                evolvable_traits: &config.brain.traits,
                architecture: &architecture,
                compute_device: &device,
                observation_schema: &schema,
                deterministic_seed: 23,
                seed_derivation: Default::default(),
                completed_lives: &[],
                genome_registry: &registry,
            })
            .unwrap_err()
            .to_string(),
        "policy error: fewer than two eligible parents"
    );

    config.evolution.selection.minimum_age = 0;
    config.evolution.fallback = midas_core::config::EvolutionFallback::RandomGenome;
    config.evolution.mutation.weight_probability = 0.0;
    config.evolution.mutation.bias_probability = 0.0;
    config.evolution.mutation.reset_probability = 0.0;
    config.evolution.mutation.trait_probability = 0.0;
    config.brain.traits.learning_rate.minimum = 0.5;
    config.brain.traits.learning_rate.maximum = 1.0;
    let two_living = vec![
        LivingAgentSummary {
            id: states[1].id,
            net_worth: states[1].portfolio.net_worth,
            selection_fitness: FitnessScore(states[1].portfolio.net_worth),
            age: states[1].age,
            policy: PolicyBlueprint::Neural,
            genome: states[1].genome.as_ref(),
            maximum_drawdown: states[1].maximum_drawdown,
            birth_date: states[1].birth_date,
            parent_ids: states[1].parent_ids.clone(),
        },
        LivingAgentSummary {
            id: states[2].id,
            net_worth: states[2].portfolio.net_worth,
            selection_fitness: FitnessScore(states[2].portfolio.net_worth),
            age: states[2].age,
            policy: PolicyBlueprint::Neural,
            genome: states[2].genome.as_ref(),
            maximum_drawdown: states[2].maximum_drawdown,
            birth_date: states[2].birth_date,
            parent_ids: states[2].parent_ids.clone(),
        },
    ];
    let mut random_strategy = EvolutionaryReplacementStrategy::default();
    let random_request = random_strategy
        .create_replacement(&ReplacementContext {
            dead_agent: &dead,
            dead_policy: PolicyBlueprint::Neural,
            living_population: &two_living,
            replacement: &config.replacement,
            evolution: &config.evolution,
            evolvable_traits: &config.brain.traits,
            architecture: &architecture,
            compute_device: &device,
            observation_schema: &schema,
            deterministic_seed: 23,
            seed_derivation: Default::default(),
            completed_lives: &[],
            genome_registry: &registry,
        })
        .unwrap();
    assert!(random_request.parent_ids.is_empty());
    assert!(
        random_request
            .metadata
            .reason
            .starts_with("mutation_failed:")
    );
    let random_report = random_strategy.evolution_report().unwrap();
    assert_eq!(
        random_report.outcome,
        EvolutionReplacementOutcome::RandomFallback
    );
    assert!(!random_report.crossover);
    assert!(random_report.mutation.is_none());
}

#[test]
fn evolutionary_replacement_uses_selection_fitness_not_raw_net_worth() {
    let mut config = prompt2_config();
    let data = PolarsDataSource.load(&config).unwrap();
    let engine = SimulationEngine::new(config.clone(), data, DatasetSplit::Training).unwrap();
    config.evolution.fallback = midas_core::config::EvolutionFallback::CloneBestEligible;
    config.evolution.diversity.fitness_sharing = false;
    config.evolution.diversity.minimum_mating_distance = f32::NAN;
    let states = engine.agents().collect::<Vec<_>>();
    let dead = states[0].clone();
    let living = vec![
        LivingAgentSummary {
            id: states[1].id,
            net_worth: 20_000.0,
            selection_fitness: FitnessScore(-1.0),
            age: states[1].age,
            policy: PolicyBlueprint::Neural,
            genome: states[1].genome.as_ref(),
            maximum_drawdown: states[1].maximum_drawdown,
            birth_date: states[1].birth_date,
            parent_ids: states[1].parent_ids.clone(),
        },
        LivingAgentSummary {
            id: states[2].id,
            net_worth: 1_000.0,
            selection_fitness: FitnessScore(1.0),
            age: states[2].age,
            policy: PolicyBlueprint::Neural,
            genome: states[2].genome.as_ref(),
            maximum_drawdown: states[2].maximum_drawdown,
            birth_date: states[2].birth_date,
            parent_ids: states[2].parent_ids.clone(),
        },
    ];
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
    let registry = std::collections::BTreeMap::new();
    let mut strategy = EvolutionaryReplacementStrategy::default();

    let request = strategy
        .create_replacement(&ReplacementContext {
            dead_agent: &dead,
            dead_policy: PolicyBlueprint::Neural,
            living_population: &living,
            replacement: &config.replacement,
            evolution: &config.evolution,
            evolvable_traits: &config.brain.traits,
            architecture: &architecture,
            compute_device: &device,
            observation_schema: &schema,
            deterministic_seed: 31,
            seed_derivation: Default::default(),
            completed_lives: &[],
            genome_registry: &registry,
        })
        .unwrap();

    assert_eq!(request.parent_ids, vec![states[2].id]);
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
            selection_fitness: FitnessScore(10_000.0 + (index as f64) * 10.0),
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
        replacement: &unrestricted.replacement,
        evolution: &unrestricted.evolution,
        evolvable_traits: &unrestricted.brain.traits,
        architecture: &architecture,
        compute_device: &device,
        observation_schema: &schema,
        deterministic_seed: 19,
        seed_derivation: Default::default(),
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
        replacement: &constrained.replacement,
        evolution: &constrained.evolution,
        evolvable_traits: &constrained.brain.traits,
        architecture: &architecture,
        compute_device: &device,
        observation_schema: &schema,
        deterministic_seed: 19,
        seed_derivation: Default::default(),
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
