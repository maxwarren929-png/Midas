use crate::{
    AgentId, AssetId, MidasError, Result,
    action::{ActionValidator, LongOnlyActionValidator},
    agent::{
        Agent, AgentBuildContext, AgentFactory, AgentSpawnRequest, SpawnMetadata,
        StandardAgentFactory, configured_blueprint,
    },
    brain::{BrainArchitecture, BrainGenome, ComputeDevice, PolicySnapshot},
    config::Config,
    evolution::EvolutionState,
    execution::{CostExecutionEngine, ExecutionApplier, ExecutionPlanner},
    market::{BackwardFeaturePipeline, FeaturePipeline, FeatureRow, MarketDataset},
    metrics::{EventSink, MetricsCollector},
    observation::{ObservationBuilder, ObservationSchema, SchemaObservationBuilder},
    replacement::{NonEvolutionaryRandomGenomeReplacement, ReplacementStrategy},
    reward::{RewardFunction, configured_reward},
    seed::SeedDomain,
};
use chrono::NaiveDate;

use super::{DatasetSplit, DomainEvent, OperationalEvent, ReportTrace, SimulationLifecycle};

pub struct SimulationEngine {
    pub(super) config: Config,
    pub(super) data: MarketDataset,
    pub(super) split: DatasetSplit,
    pub(super) features: Vec<FeatureRow>,
    pub(super) observation_schema: ObservationSchema,
    pub(super) observation_builder: Box<dyn ObservationBuilder>,
    pub(super) architecture: BrainArchitecture,
    pub(super) compute_device: ComputeDevice,
    pub(super) agents: Vec<Agent>,
    pub(super) planner: Box<dyn ExecutionPlanner>,
    pub(super) applier: Box<dyn ExecutionApplier>,
    pub(super) validator: Box<dyn ActionValidator>,
    pub(super) factory: Box<dyn AgentFactory>,
    pub(super) replacement: Box<dyn ReplacementStrategy>,
    pub(super) reward_function: Box<dyn RewardFunction>,
    pub(super) evolution: EvolutionState,
    pub metrics: MetricsCollector,
    pub domain_events: Vec<DomainEvent>,
    pub operational_events: Vec<OperationalEvent>,
    pub report_traces: Vec<ReportTrace>,
    pub(super) event_sinks: Vec<Box<dyn EventSink>>,
    pub(super) next_step: usize,
    pub(super) next_agent_id: u64,
    pub(super) lifecycle: SimulationLifecycle,
}

impl SimulationEngine {
    pub fn new(config: Config, data: MarketDataset, split: DatasetSplit) -> Result<Self> {
        Self::new_with_factory(
            config,
            data,
            split,
            Box::new(StandardAgentFactory::default()),
        )
    }

    pub fn new_with_factory(
        config: Config,
        data: MarketDataset,
        split: DatasetSplit,
        factory: Box<dyn AgentFactory>,
    ) -> Result<Self> {
        config.validate()?;
        let compute_device = ComputeDevice::from_options(config.backend.compute_device_options())?;
        let pipeline = BackwardFeaturePipeline::from_config(&config);
        let dates = data.split_dates(split.range(&config));
        if dates.len() < 2 {
            return Err(MidasError::Data(
                "selected split needs at least two trading dates".into(),
            ));
        }
        let features = pipeline.generate(&data, &dates)?;
        let observation_schema =
            ObservationSchema::from_options(config.observation_schema_options());
        let schema_hash = observation_schema.identity(data.assets.len())?;
        let observation_builder = SchemaObservationBuilder::from_options(
            observation_schema.clone(),
            config.observation_builder_options(),
            Some(&data),
        )?;
        let mut architecture = BrainArchitecture::from_config(
            &config.brain,
            schema_hash,
            observation_schema.dimension(data.assets.len()),
            data.assets.len(),
        );
        if config.turnover_constraint.is_some() {
            // Output zero remains the reward critic. Output one is an
            // independently trained turnover-cost critic.
            architecture.critic_output_size = 2;
        }
        let first = features[0].date;
        let blueprint = configured_blueprint(&config)?;
        let mut agents = Vec::new();
        let seed_derivation = config.simulation.seed_derivation;
        let root_seed = config.simulation.seed;
        let mut evolution =
            EvolutionState::new(seed_derivation.derive(root_seed, SeedDomain::Evolution, 0));
        let mut metrics = MetricsCollector::default();
        let mut domain_events = Vec::new();
        for i in 0..config.simulation.population_size {
            let id = AgentId(i as u64);
            let genome = if matches!(blueprint, crate::brain::PolicyBlueprint::Neural) {
                Some(BrainGenome::random(
                    &architecture,
                    &config.brain.traits,
                    seed_derivation.derive(root_seed, SeedDomain::InitialGenome, id.0),
                )?)
            } else {
                None
            };
            let request = AgentSpawnRequest {
                policy_blueprint: blueprint.clone(),
                genome,
                parent_ids: Vec::new(),
                metadata: SpawnMetadata {
                    reason: "initial_population".into(),
                    ..SpawnMetadata::default()
                },
            };
            let mut agent = factory.spawn(
                request,
                &AgentBuildContext {
                    agent_id: id,
                    birth_date: first,
                    config: &config,
                    architecture: &architecture,
                    compute_device: &compute_device,
                },
            )?;
            if let Some(genome) = agent.state.genome.as_ref() {
                let (genome_id, lineage_id) = evolution.allocate_genome_with_reason(
                    genome,
                    Vec::new(),
                    0,
                    None,
                    "founder",
                )?;
                agent.state.genome_id = Some(genome_id);
                agent.state.lineage_id = Some(lineage_id);
            }
            let event = DomainEvent::AgentBorn {
                agent_id: id,
                date: first,
            };
            metrics.on_event(&event);
            domain_events.push(event);
            agents.push(agent);
        }
        let initial_genomes = agents
            .iter()
            .filter_map(|agent| agent.state.genome.as_ref())
            .collect::<Vec<_>>();
        evolution.update_diversity(&initial_genomes)?;
        let initial_lineages = agents
            .iter()
            .filter(|agent| agent.state.alive)
            .filter_map(|agent| agent.state.lineage_id)
            .collect::<Vec<_>>();
        evolution.update_active_lineages(&initial_lineages);
        let costs = CostExecutionEngine {
            costs: config.execution.clone(),
        };
        let reward_function = configured_reward(&config.reward);
        let replacement: Box<dyn ReplacementStrategy> = if config.evolution.enabled {
            Box::new(crate::replacement::EvolutionaryReplacementStrategy::default())
        } else {
            Box::new(NonEvolutionaryRandomGenomeReplacement)
        };
        let next_agent_id = config.simulation.population_size as u64;
        Ok(Self {
            config,
            data,
            split,
            features,
            observation_schema,
            observation_builder: Box::new(observation_builder),
            architecture,
            compute_device,
            agents,
            planner: Box::new(costs.clone()),
            applier: Box::new(costs),
            validator: Box::new(LongOnlyActionValidator),
            factory,
            replacement,
            reward_function,
            evolution,
            metrics,
            domain_events,
            operational_events: Vec::new(),
            report_traces: Vec::new(),
            event_sinks: Vec::new(),
            next_step: 0,
            next_agent_id,
            lifecycle: SimulationLifecycle::Initialized,
        })
    }

    pub fn lifecycle(&self) -> SimulationLifecycle {
        self.lifecycle.clone()
    }

    pub fn agents(&self) -> impl Iterator<Item = &crate::agent::AgentState> {
        self.agents.iter().map(|a| &a.state)
    }

    pub fn agent_blueprints(
        &self,
    ) -> impl Iterator<Item = (AgentId, crate::brain::PolicyBlueprint)> + '_ {
        self.agents
            .iter()
            .map(|agent| (agent.state.id, agent.policy.policy_blueprint()))
    }

    pub fn agent_lifetime_parameters(
        &self,
        agent_id: AgentId,
    ) -> Result<Option<crate::brain::LifetimeParameters>> {
        let agent = self
            .agents
            .iter()
            .find(|agent| agent.state.id == agent_id)
            .ok_or_else(|| MidasError::Policy(format!("agent {agent_id:?} is not present")))?;
        agent.policy.lifetime_parameters()
    }

    pub fn agent_policy_snapshot(&self, agent_id: AgentId) -> Result<PolicySnapshot> {
        let agent = self
            .agents
            .iter()
            .find(|agent| agent.state.id == agent_id)
            .ok_or_else(|| MidasError::Policy(format!("agent {agent_id:?} is not present")))?;
        agent.policy.snapshot()
    }

    pub fn agent_learning_snapshot(
        &self,
        agent_id: AgentId,
    ) -> Result<crate::learning::LearningStrategyState> {
        let agent = self
            .agents
            .iter()
            .find(|agent| agent.state.id == agent_id)
            .ok_or_else(|| MidasError::Policy(format!("agent {agent_id:?} is not present")))?;
        agent.learning_snapshot()
    }

    pub fn observation_schema(&self) -> &ObservationSchema {
        &self.observation_schema
    }

    pub fn evolution_state(&self) -> &EvolutionState {
        &self.evolution
    }

    pub fn add_event_sink(&mut self, sink: Box<dyn EventSink>) {
        self.event_sinks.push(sink)
    }

    pub fn set_replacement_strategy(&mut self, r: Box<dyn ReplacementStrategy>) {
        self.replacement = r
    }

    pub(super) fn valuation_prices(&self, date: NaiveDate) -> Vec<Option<f64>> {
        (0..self.data.assets.len())
            .map(|i| self.data.last_close(date, AssetId(i as u64)))
            .collect()
    }

    pub(super) fn emit(&mut self, event: DomainEvent) {
        self.metrics.on_event(&event);
        for sink in &mut self.event_sinks {
            sink.on_event(&event)
        }
        self.domain_events.push(event)
    }
}
