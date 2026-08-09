use super::*;

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
fn random_evolution_fallback_emits_event_and_only_random_metric() {
    let mut c = bankrupt_config(PolicyKind::BurnActorCritic, DeathHandling::Replace);
    c.evolution.enabled = true;
    c.evolution.fallback = EvolutionFallback::RandomGenome;
    let data = loaded(&c);
    let mut engine = SimulationEngine::new(c, data, DatasetSplit::Training).unwrap();

    engine.run_steps(1).unwrap();

    let evolution = engine.evolution_state();
    assert_eq!(evolution.metrics.random_fallback_births, 1);
    assert_eq!(evolution.metrics.crossover_count, 0);
    assert_eq!(evolution.metrics.mutation_count, 0);
    assert_eq!(evolution.metrics.reproduction_count, 0);
    assert!(engine.domain_events.iter().any(|event| matches!(
        event,
        DomainEvent::EvolutionFallback { reason }
            if reason == "evolution_random_genome_fallback"
    )));
}

#[test]
fn simultaneous_deaths_use_frozen_parent_pool_and_current_lineages() {
    let mut c = bankrupt_config(PolicyKind::BurnActorCritic, DeathHandling::Replace);
    c.simulation.population_size = 3;
    c.evolution.enabled = true;
    c.evolution.fallback = EvolutionFallback::RandomGenome;
    let data = loaded(&c);
    let mut engine = SimulationEngine::new(c, data, DatasetSplit::Training).unwrap();

    engine.run_steps(1).unwrap();

    let evolution = engine.evolution_state();
    assert_eq!(evolution.metrics.random_fallback_births, 3);
    assert_eq!(evolution.metrics.reproduction_count, 0);
    assert_eq!(evolution.metrics.crossover_count, 0);
    assert_eq!(evolution.metrics.mutation_count, 0);
    assert_eq!(evolution.metrics.active_lineages, 3);
    assert_eq!(evolution.genome_registry.len(), 6);
    assert_eq!(engine.report().active_lineages, 3);
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

#[derive(Clone)]
struct AlternatingReplacement {
    replacements: u64,
}

impl ReplacementStrategy for AlternatingReplacement {
    fn create_replacement(
        &mut self,
        _context: &ReplacementContext<'_>,
    ) -> midas_core::Result<AgentSpawnRequest> {
        self.replacements += 1;
        let policy_blueprint = if self.replacements.is_multiple_of(2) {
            PolicyBlueprint::Random
        } else {
            PolicyBlueprint::Cash
        };
        Ok(AgentSpawnRequest {
            policy_blueprint,
            genome: None,
            parent_ids: Vec::new(),
            metadata: SpawnMetadata {
                reason: format!("alternating_replacement:{}", self.replacements),
                ..SpawnMetadata::default()
            },
        })
    }

    fn name(&self) -> &'static str {
        "test_alternating_replacement"
    }

    fn fork(&self) -> midas_core::Result<Box<dyn ReplacementStrategy>> {
        Ok(Box::new(self.clone()))
    }

    fn checkpoint_state(&self) -> midas_core::Result<ReplacementStrategyState> {
        Ok(ReplacementStrategyState {
            identity: self.name().into(),
            version: 1,
            payload: self.replacements.to_le_bytes().to_vec(),
        })
    }
}

struct AlternatingReplacementFactory;

impl ReplacementStrategyFactory for AlternatingReplacementFactory {
    fn restore(
        &self,
        state: &ReplacementStrategyState,
    ) -> midas_core::Result<Box<dyn ReplacementStrategy>> {
        state.validate()?;
        if state.identity != "test_alternating_replacement" || state.version != 1 {
            return Err(MidasError::Checkpoint(
                "incompatible alternating replacement state".into(),
            ));
        }
        let payload: [u8; 8] = state.payload.as_slice().try_into().map_err(|_| {
            MidasError::Checkpoint("invalid alternating replacement payload".into())
        })?;
        Ok(Box::new(AlternatingReplacement {
            replacements: u64::from_le_bytes(payload),
        }))
    }
}

#[test]
fn stateful_replacement_is_checkpointed_before_commit_and_resumes_deterministically() {
    let directory = tempfile::tempdir().unwrap();
    let automatic = directory.path().join("automatic.bin");
    let full_checkpoint = directory.path().join("full.bin");
    let resumed_checkpoint = directory.path().join("resumed.bin");
    let mut c = bankrupt_config(PolicyKind::Cash, DeathHandling::Replace);
    c.checkpoint.every_steps = 1;
    c.checkpoint.path = automatic.to_string_lossy().into_owned();
    let data = loaded(&c);

    let mut full = SimulationEngine::new(c.clone(), data.clone(), DatasetSplit::Training).unwrap();
    full.set_replacement_strategy(Box::new(AlternatingReplacement { replacements: 0 }));
    full.run_steps(2).unwrap();
    full.save_checkpoint(&full_checkpoint).unwrap();

    let mut partial =
        SimulationEngine::new(c.clone(), data.clone(), DatasetSplit::Training).unwrap();
    partial.set_replacement_strategy(Box::new(AlternatingReplacement { replacements: 0 }));
    partial.run_steps(1).unwrap();
    assert_eq!(
        u32::from_le_bytes(fs::read(&automatic).unwrap()[..4].try_into().unwrap()),
        CHECKPOINT_VERSION
    );
    assert_eq!(
        SimulationEngine::resume(c.clone(), data.clone(), &automatic)
            .err()
            .unwrap()
            .to_string(),
        "checkpoint error: replacement strategy test_alternating_replacement is not available from the built-in factory"
    );

    let mut resumed = SimulationEngine::resume_with_replacement_factory(
        c.clone(),
        data.clone(),
        &automatic,
        Box::new(AlternatingReplacementFactory),
    )
    .unwrap();
    resumed.run_steps(1).unwrap();
    resumed.save_checkpoint(&resumed_checkpoint).unwrap();

    assert_eq!(
        fs::read(full_checkpoint).unwrap(),
        fs::read(&resumed_checkpoint).unwrap()
    );
    assert_eq!(
        full.agent_blueprints().collect::<Vec<_>>(),
        resumed.agent_blueprints().collect::<Vec<_>>()
    );
    assert_eq!(full.domain_events, resumed.domain_events);
    assert_eq!(full.metrics, resumed.metrics);

    SimulationEngine::evaluate_checkpoint_with_replacement_factory(
        c,
        data,
        resumed_checkpoint,
        DatasetSplit::Validation,
        Box::new(AlternatingReplacementFactory),
    )
    .unwrap();
}

#[test]
fn checkpoint_rejects_a_strategy_without_state_support() {
    let c = config();
    let data = loaded(&c);
    let mut engine = SimulationEngine::new(c, data, DatasetSplit::Training).unwrap();
    engine.set_replacement_strategy(Box::new(SelectSpecificLiving));
    let checkpoint = tempfile::NamedTempFile::new().unwrap();

    assert_eq!(
        engine
            .save_checkpoint(checkpoint.path())
            .unwrap_err()
            .to_string(),
        "checkpoint error: replacement strategy test_specific_living does not support checkpoint state"
    );
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
        selection_fitness: FitnessScore(state.portfolio.net_worth),
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
            replacement: &c.replacement,
            evolution: &c.evolution,
            evolvable_traits: &c.brain.traits,
            architecture: &architecture,
            compute_device: &device,
            observation_schema: &schema,
            deterministic_seed: 7,
            seed_derivation: Default::default(),
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
