pub use crate::PolicyFailureCategory;
use crate::{
    AgentId, AssetId, MidasError, Result,
    action::{
        ActionValidationContext, ActionValidator, LongOnlyActionValidator, ProposedAllocation,
        ValidatedAllocation,
    },
    agent::{
        Agent, AgentBuildContext, AgentFactory, AgentSpawnRequest, SpawnMetadata,
        StandardAgentFactory, configured_blueprint,
    },
    brain::{
        BrainArchitecture, BrainGenome, ComputeDevice, NeuralModelState, PolicyContext,
        PolicySnapshot,
    },
    config::{Config, DateRange, DeathHandling, LearningKind, PolicyFailureBehavior},
    evolution::{CompletedLifeSummary, DefaultFitnessFunction, EvolutionState},
    execution::{
        CostExecutionEngine, ExecutionApplier, ExecutionPlanner, ExecutionSnapshot, PlannedTrade,
        Portfolio,
    },
    learning::{EpisodeEndContext, LearningStrategyState, LearningUpdate, Transition},
    market::{
        BackwardFeaturePipeline, CorporateAction, FeaturePipeline, FeatureRow, MarketDataset,
    },
    metrics::{EventSink, MetricsCollector},
    observation::{
        ObservationBuilder, ObservationContext, ObservationSchema, SchemaObservationBuilder,
    },
    replacement::{
        LivingAgentSummary, NonEvolutionaryRandomGenomeReplacement, ReplacementContext,
        ReplacementStrategy,
    },
    reward::{RewardContext, RewardFunction, configured_reward},
};
use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};
// Version 8 adds masked chronological rollout entries and benchmark-aware
// action-validation state. Checkpoint 7 cannot be safely migrated because
// its transition eligibility and allocation-cap semantics are absent.
pub const CHECKPOINT_VERSION: u32 = 8;
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum DatasetSplit {
    Training,
    Validation,
    Test,
}
impl DatasetSplit {
    pub fn range<'a>(&self, c: &'a Config) -> &'a DateRange {
        match self {
            Self::Training => &c.splits.training,
            Self::Validation => &c.splits.validation,
            Self::Test => &c.splits.test,
        }
    }
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum SimulationLifecycle {
    Initialized,
    Running,
    Completed,
    Failed { reason: String },
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum DeathCause {
    Bankruptcy,
    PolicyFailure,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum DomainEvent {
    DayStarted {
        date: NaiveDate,
        step: usize,
    },
    ObservationRevealed {
        date: NaiveDate,
    },
    AgentBorn {
        agent_id: AgentId,
        date: NaiveDate,
    },
    ActionFallback {
        agent_id: AgentId,
        date: NaiveDate,
    },
    LearningUpdate {
        agent_id: AgentId,
        date: NaiveDate,
        rollout_size: usize,
        policy_loss: f32,
        value_loss: f32,
        entropy: f32,
        gradient_norm: f32,
    },
    LearningTransitionMasked {
        agent_id: AgentId,
        date: NaiveDate,
        actor_eligible: bool,
        critic_eligible: bool,
        penalty: f32,
    },
    FitnessAssigned {
        agent_id: AgentId,
        genome_id: Option<crate::evolution::GenomeId>,
        fitness: f64,
    },
    ParentsSelected {
        child_agent_id: AgentId,
        parent_agent_ids: Vec<AgentId>,
    },
    DiversityConstraintRelaxed {
        first_parent_agent_id: AgentId,
        requested_minimum_distance: f64,
        selected_distance: f64,
    },
    GenomeCrossed {
        child_agent_id: AgentId,
        method: crate::config::CrossoverMethod,
    },
    GenomeMutated {
        child_agent_id: AgentId,
        weight_changes: u64,
        bias_changes: u64,
        trait_changes: u64,
        reset_changes: u64,
        weight_resets: u64,
        bias_resets: u64,
    },
    EvolutionFallback {
        reason: String,
    },
    LineageCreated {
        lineage_id: crate::evolution::LineageId,
        parent_lineages: Vec<crate::evolution::LineageId>,
    },
    PolicyFailure {
        agent_id: AgentId,
        date: NaiveDate,
        policy_kind: crate::brain::PolicyBlueprint,
        category: PolicyFailureCategory,
        message: String,
    },
    CorporateActionApplied {
        agent_id: AgentId,
        date: NaiveDate,
        action: CorporateAction,
    },
    TradeExecuted {
        agent_id: AgentId,
        date: NaiveDate,
        trade: PlannedTrade,
    },
    TradeRejected {
        agent_id: AgentId,
        date: NaiveDate,
        asset_id: AssetId,
    },
    Valuation {
        agent_id: AgentId,
        date: NaiveDate,
        net_worth: f64,
        previous: f64,
        reward: f32,
    },
    AgentDied {
        agent_id: AgentId,
        birth_date: NaiveDate,
        death_date: NaiveDate,
        age: u64,
        starting_net_worth: f64,
        final_net_worth: f64,
        maximum_net_worth: f64,
        cause: DeathCause,
    },
    TimelineCompleted {
        date: NaiveDate,
    },
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum OperationalEvent {
    CheckpointSaved { date: NaiveDate, path: String },
}
#[derive(Clone, Debug, Serialize, Deserialize)]
struct AgentCheckpoint {
    state: crate::agent::AgentState,
    policy: PolicySnapshot,
    learning: crate::learning::LearningStrategyState,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Checkpoint {
    version: u32,
    config_hash: String,
    dataset_identity: String,
    observation_schema_hash: String,
    architecture: BrainArchitecture,
    split: DatasetSplit,
    next_step: usize,
    next_agent_id: u64,
    lifecycle: SimulationLifecycle,
    agents: Vec<AgentCheckpoint>,
    metrics: MetricsCollector,
    domain_events: Vec<DomainEvent>,
    accessed_through: Option<NaiveDate>,
    backend_identity: String,
    evolution: EvolutionState,
}

pub struct SimulationEngine {
    config: Config,
    data: MarketDataset,
    split: DatasetSplit,
    features: Vec<FeatureRow>,
    observation_schema: ObservationSchema,
    observation_builder: Box<dyn ObservationBuilder>,
    architecture: BrainArchitecture,
    compute_device: ComputeDevice,
    agents: Vec<Agent>,
    planner: Box<dyn ExecutionPlanner>,
    applier: Box<dyn ExecutionApplier>,
    validator: Box<dyn ActionValidator>,
    factory: Box<dyn AgentFactory>,
    replacement: Box<dyn ReplacementStrategy>,
    reward_function: Box<dyn RewardFunction>,
    evolution: EvolutionState,
    pub metrics: MetricsCollector,
    pub domain_events: Vec<DomainEvent>,
    pub operational_events: Vec<OperationalEvent>,
    event_sinks: Vec<Box<dyn EventSink>>,
    next_step: usize,
    next_agent_id: u64,
    lifecycle: SimulationLifecycle,
}
struct StepTransaction {
    agents: Vec<Agent>,
    next_agent_id: u64,
    events: Vec<DomainEvent>,
    replacement: Option<Box<dyn ReplacementStrategy>>,
    evolution: EvolutionState,
}
type AdvanceResult = (Vec<DomainEvent>, Option<(AgentId, NaiveDate)>, f32);
#[derive(Clone, Debug)]
struct PolicyFailureIncident {
    category: PolicyFailureCategory,
    message: String,
}
enum PolicyDecisionOutcome {
    Valid {
        validated: crate::action::ValidatedAllocation,
        observation: crate::observation::Observation,
        evaluation: crate::brain::PolicyEvaluation,
        runtime_before: Option<crate::brain::NeuralRuntimeState>,
    },
    RecoverableFailure {
        incident: PolicyFailureIncident,
        fallback: crate::action::ValidatedAllocation,
        observation: Option<crate::observation::Observation>,
        evaluation: Option<crate::brain::PolicyEvaluation>,
        runtime_before: Option<crate::brain::NeuralRuntimeState>,
    },
    FatalAgentFailure {
        incident: PolicyFailureIncident,
    },
    FatalExperimentFailure {
        incident: PolicyFailureIncident,
    },
}
enum StepAbort {
    Error(MidasError),
    FatalExperiment { event: DomainEvent, reason: String },
}
impl From<MidasError> for StepAbort {
    fn from(error: MidasError) -> Self {
        Self::Error(error)
    }
}
impl SimulationEngine {
    fn lifecycle_error(reason: &str) -> MidasError {
        if let Some(message) = reason.strip_prefix("configuration error: ") {
            MidasError::Config(message.into())
        } else if let Some(message) = reason.strip_prefix("market data error: ") {
            MidasError::Data(message.into())
        } else if let Some(message) = reason.strip_prefix("accounting invariant failed: ") {
            MidasError::Accounting(message.into())
        } else if let Some(message) = reason.strip_prefix("policy error: ") {
            MidasError::Policy(message.into())
        } else if let Some(message) = reason.strip_prefix("checkpoint error: ") {
            MidasError::Checkpoint(message.into())
        } else {
            MidasError::Policy(reason.into())
        }
    }

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
        let compute_device = config.backend.compute_device()?;
        let pipeline = BackwardFeaturePipeline::from_config(&config);
        let dates = data.split_dates(split.range(&config));
        if dates.len() < 2 {
            return Err(MidasError::Data(
                "selected split needs at least two trading dates".into(),
            ));
        }
        let features = pipeline.generate(&data, &dates)?;
        let observation_schema = ObservationSchema::from_config(&config);
        let schema_hash = observation_schema.identity(data.assets.len())?;
        let observation_builder = SchemaObservationBuilder::new(
            observation_schema.clone(),
            config.features.features.clone(),
            data.assets.len(),
        )?;
        let architecture = BrainArchitecture::from_config(
            &config.brain,
            schema_hash,
            observation_schema.dimension(data.assets.len()),
            data.assets.len(),
        );
        let first = features[0].date;
        let blueprint = configured_blueprint(&config)?;
        let mut agents = Vec::new();
        let mut evolution = EvolutionState::new(config.simulation.seed ^ 0x243f6a8885a308d3);
        let mut metrics = MetricsCollector::default();
        let mut domain_events = Vec::new();
        for i in 0..config.simulation.population_size {
            let id = AgentId(i as u64);
            let genome = if matches!(blueprint, crate::brain::PolicyBlueprint::Neural) {
                Some(BrainGenome::random(
                    &architecture,
                    &config.brain.traits,
                    config.simulation.seed ^ id.0,
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
    pub fn run(&mut self) -> Result<SimulationReport> {
        match &self.lifecycle {
            SimulationLifecycle::Completed => return Ok(self.report()),
            SimulationLifecycle::Failed { reason } => {
                return Err(Self::lifecycle_error(reason));
            }
            SimulationLifecycle::Initialized | SimulationLifecycle::Running => {}
        }
        self.lifecycle = SimulationLifecycle::Running;
        while self.next_step + 1 < self.features.len() {
            self.step()?;
        }
        if let Err(error) = self.flush_learning() {
            self.lifecycle = SimulationLifecycle::Failed {
                reason: error.to_string(),
            };
            return Err(error);
        }
        self.lifecycle = SimulationLifecycle::Completed;
        let date = self.features.last().expect("validated features").date;
        self.emit(DomainEvent::TimelineCompleted { date });
        Ok(self.report())
    }
    pub fn run_steps(&mut self, count: usize) -> Result<()> {
        match &self.lifecycle {
            SimulationLifecycle::Completed => return Ok(()),
            SimulationLifecycle::Failed { reason } => {
                return Err(Self::lifecycle_error(reason));
            }
            SimulationLifecycle::Initialized | SimulationLifecycle::Running => {}
        }
        self.lifecycle = SimulationLifecycle::Running;
        for _ in 0..count {
            if self.next_step + 1 >= self.features.len() {
                break;
            }
            self.step()?;
        }
        Ok(())
    }
    fn step(&mut self) -> Result<()> {
        let agents = match self.fork_agents() {
            Ok(agents) => agents,
            Err(error) => {
                let reason = error.to_string();
                self.lifecycle = SimulationLifecycle::Failed {
                    reason: reason.clone(),
                };
                return Err(error);
            }
        };
        let transaction = StepTransaction {
            agents,
            next_agent_id: self.next_agent_id,
            events: Vec::new(),
            replacement: None,
            evolution: self.evolution.clone(),
        };
        let mut transaction = transaction;
        match self.execute_step(&mut transaction) {
            Ok(()) => match self.commit_step(transaction) {
                Ok(()) => Ok(()),
                Err(error) => {
                    self.lifecycle = SimulationLifecycle::Failed {
                        reason: error.to_string(),
                    };
                    Err(error)
                }
            },
            Err(StepAbort::Error(error)) => {
                let reason = error.to_string();
                self.lifecycle = SimulationLifecycle::Failed {
                    reason: reason.clone(),
                };
                Err(error)
            }
            Err(StepAbort::FatalExperiment { event, reason }) => {
                self.emit(event);
                self.lifecycle = SimulationLifecycle::Failed {
                    reason: reason.clone(),
                };
                Err(MidasError::Policy(reason))
            }
        }
    }

    fn flush_learning(&mut self) -> Result<()> {
        let mut agents = self.fork_agents()?;
        let mut events = Vec::new();
        let date = self.features.last().expect("validated features").date;
        for agent in &mut agents {
            if agent.state.alive {
                self.finish_learning_episode(
                    agent,
                    date,
                    agent.state.portfolio.net_worth,
                    true,
                    &mut events,
                )?;
            }
        }
        self.agents = agents;
        for event in events {
            self.emit(event);
        }
        Ok(())
    }

    fn fork_agents(&self) -> Result<Vec<Agent>> {
        self.agents
            .iter()
            .map(|agent| match agent.fork() {
                Ok(forked) => Ok(forked),
                Err(fork_error) => {
                    let snapshot = agent.policy.snapshot().map_err(|restore_error| {
                        MidasError::Policy(format!(
                            "transactional policy fork failed: {fork_error}; snapshot failed: {restore_error}"
                        ))
                    })?;
                    let learning = agent.learning_snapshot()?;
                    let mut restored = self.factory
                        .restore(
                            agent.state.clone(),
                            snapshot,
                            &AgentBuildContext {
                                agent_id: agent.state.id,
                                birth_date: agent.state.birth_date,
                                config: &self.config,
                                architecture: &self.architecture,
                                compute_device: &self.compute_device,
                            },
                        )
                        .map_err(|restore_error| {
                            MidasError::Policy(format!(
                                "transactional policy fork failed: {fork_error}; restore failed: {restore_error}"
                            ))
                        })?;
                    restored.restore_learning(learning)?;
                    Ok(restored)
                }
            })
            .collect()
    }

    fn commit_step(&mut self, transaction: StepTransaction) -> Result<()> {
        let checkpoint_due = self.config.checkpoint.every_steps > 0
            && (self.next_step + 1).is_multiple_of(self.config.checkpoint.every_steps);
        let checkpoint_path = self.config.checkpoint.path.clone();
        if checkpoint_due {
            let mut candidate_metrics = self.metrics.clone();
            for event in &transaction.events {
                candidate_metrics.on_event(event);
            }
            let mut candidate_events = self.domain_events.clone();
            candidate_events.extend(transaction.events.clone());
            let checkpoint = self.checkpoint_from_state(
                self.next_step + 1,
                transaction.next_agent_id,
                &transaction.agents,
                candidate_metrics,
                candidate_events,
                transaction.evolution.clone(),
                self.lifecycle.clone(),
            )?;
            self.write_checkpoint_file(&checkpoint_path, &checkpoint)?;
        }
        let StepTransaction {
            agents,
            next_agent_id,
            events,
            replacement,
            evolution,
        } = transaction;
        self.agents = agents;
        self.next_agent_id = next_agent_id;
        self.evolution = evolution;
        if let Some(replacement) = replacement {
            self.replacement = replacement;
        }
        for event in events {
            self.emit(event);
        }
        self.next_step += 1;
        if checkpoint_due {
            let date = self
                .features
                .get(self.next_step)
                .map(|f| f.date)
                .unwrap_or_else(|| self.features.last().expect("features").date);
            self.operational_events
                .push(OperationalEvent::CheckpointSaved {
                    date,
                    path: checkpoint_path,
                });
        }
        Ok(())
    }

    fn execute_step(
        &mut self,
        transaction: &mut StepTransaction,
    ) -> std::result::Result<(), StepAbort> {
        let t = self.next_step;
        let observation_date = self.features[t].date;
        let execution_date = self.features[t + 1].date;
        let current_prices = self.valuation_prices(observation_date);
        let next_prices = self.valuation_prices(execution_date);
        let bars = (0..self.data.assets.len())
            .map(|i| self.data.bar(execution_date, AssetId(i as u64)))
            .collect::<Vec<_>>();
        let tradable = self.features[t].tradable.clone();
        let next_tradable = self.features[t + 1].tradable.clone();
        let market = self.features[t].clone();
        let actions = self
            .data
            .corporate_actions_on(execution_date)
            .cloned()
            .collect::<Vec<_>>();
        transaction.events.push(DomainEvent::DayStarted {
            date: execution_date,
            step: t,
        });
        transaction.events.push(DomainEvent::ObservationRevealed {
            date: observation_date,
        });

        let mut deaths = Vec::new();
        for agent in &mut transaction.agents {
            if !agent.state.alive {
                continue;
            }
            let allocations = agent.state.portfolio.allocations(&current_prices)?;
            let maximum_asset_allocation = agent
                .state
                .genome
                .as_ref()
                .map(|genome| genome.traits.maximum_allocation as f64)
                .unwrap_or(self.config.brain.traits.maximum_allocation.initial as f64);
            let policy_cap = match agent.policy.policy_blueprint() {
                crate::brain::PolicyBlueprint::Neural | crate::brain::PolicyBlueprint::Random => {
                    Some(maximum_asset_allocation)
                }
                crate::brain::PolicyBlueprint::Cash
                | crate::brain::PolicyBlueprint::EqualWeight
                | crate::brain::PolicyBlueprint::BuyAndHold { .. } => None,
            };
            let action_context = ActionValidationContext {
                asset_count: self.data.assets.len(),
                tradable: &tradable,
                policy_maximum_asset_allocation: policy_cap,
                market_maximum_asset_allocation: None,
            };
            let outcome = self.decide_policy(
                agent,
                &market,
                &current_prices,
                &tradable,
                &allocations,
                &action_context,
            )?;
            match outcome {
                PolicyDecisionOutcome::Valid {
                    validated,
                    observation,
                    evaluation,
                    runtime_before,
                } => {
                    let (events, death, reward) = self.advance_agent(
                        agent,
                        &validated,
                        observation_date,
                        execution_date,
                        &current_prices,
                        &next_prices,
                        &bars,
                        &actions,
                        false,
                    )?;
                    transaction.events.extend(events);
                    let next_value = if death.is_none()
                        && self.config.learning.implementation == LearningKind::Ppo
                    {
                        Some(self.next_value_estimate(
                            agent,
                            t + 1,
                            &next_prices,
                            &next_tradable,
                        )?)
                    } else {
                        None
                    };
                    self.record_learning(
                        agent,
                        Some(observation),
                        Some(evaluation),
                        validated,
                        reward,
                        next_value,
                        death.is_some(),
                        observation_date,
                        &tradable,
                        runtime_before,
                        false,
                        None,
                        &mut transaction.events,
                    )?;
                    if let Some(death) = death {
                        deaths.push(death);
                    }
                }
                PolicyDecisionOutcome::RecoverableFailure {
                    incident,
                    fallback,
                    observation,
                    evaluation,
                    runtime_before,
                } => {
                    let failure_event =
                        self.policy_failure_event(agent, observation_date, incident.clone());
                    transaction.events.push(failure_event);
                    transaction.events.push(DomainEvent::ActionFallback {
                        agent_id: agent.state.id,
                        date: observation_date,
                    });
                    let (events, death, reward) = self.advance_agent(
                        agent,
                        &fallback,
                        observation_date,
                        execution_date,
                        &current_prices,
                        &next_prices,
                        &bars,
                        &actions,
                        true,
                    )?;
                    transaction.events.extend(events);
                    let next_value = if death.is_none()
                        && evaluation.is_some()
                        && self.config.learning.implementation == LearningKind::Ppo
                    {
                        Some(self.next_value_estimate(
                            agent,
                            t + 1,
                            &next_prices,
                            &next_tradable,
                        )?)
                    } else {
                        None
                    };
                    self.record_learning(
                        agent,
                        observation,
                        evaluation,
                        fallback,
                        reward,
                        next_value,
                        death.is_some(),
                        observation_date,
                        &tradable,
                        runtime_before,
                        true,
                        Some(incident.category.clone()),
                        &mut transaction.events,
                    )?;
                    if let Some(death) = death {
                        deaths.push(death);
                    }
                }
                PolicyDecisionOutcome::FatalAgentFailure { incident } => {
                    let final_net_worth = agent.state.portfolio.value(&current_prices)?;
                    agent.state.portfolio.previous_net_worth = agent.state.portfolio.net_worth;
                    agent.state.portfolio.net_worth = final_net_worth;
                    agent.state.maximum_net_worth =
                        agent.state.maximum_net_worth.max(final_net_worth);
                    if agent.state.maximum_net_worth > 0.0 {
                        agent.state.maximum_drawdown = agent
                            .state
                            .maximum_drawdown
                            .max(1.0 - final_net_worth / agent.state.maximum_net_worth);
                    }
                    agent.state.alive = false;
                    self.finish_learning_episode(
                        agent,
                        observation_date,
                        final_net_worth,
                        false,
                        &mut transaction.events,
                    )?;
                    transaction.events.push(self.policy_failure_event(
                        agent,
                        observation_date,
                        incident,
                    ));
                    transaction.events.push(DomainEvent::AgentDied {
                        agent_id: agent.state.id,
                        birth_date: agent.state.birth_date,
                        death_date: observation_date,
                        age: agent.state.age,
                        starting_net_worth: self.config.simulation.starting_cash,
                        final_net_worth,
                        maximum_net_worth: agent.state.maximum_net_worth,
                        cause: DeathCause::PolicyFailure,
                    });
                    deaths.push((agent.state.id, observation_date));
                }
                PolicyDecisionOutcome::FatalExperimentFailure { incident } => {
                    let event =
                        self.policy_failure_event(agent, observation_date, incident.clone());
                    return Err(StepAbort::FatalExperiment {
                        event,
                        reason: format!(
                            "policy failure limit reached for agent {} at {}: {}",
                            agent.state.id, observation_date, incident.message
                        ),
                    });
                }
            }
        }
        for (id, date) in deaths {
            self.handle_death_in_transaction(transaction, id, date)?;
        }
        Ok(())
    }

    fn decide_policy(
        &self,
        agent: &mut Agent,
        market: &FeatureRow,
        prices: &[Option<f64>],
        tradable: &[bool],
        allocations: &[f64],
        action_context: &ActionValidationContext<'_>,
    ) -> Result<PolicyDecisionOutcome> {
        let observation = match self.observation_builder.build(&ObservationContext {
            market,
            portfolio: &agent.state.portfolio,
            prices,
            starting_net_worth: self.config.simulation.starting_cash,
            maximum_net_worth: agent.state.maximum_net_worth,
            age: agent.state.age,
            episode_steps: self.features.len() - 1,
        }) {
            Ok(observation) => observation,
            Err(error) => {
                let incident = PolicyFailureIncident {
                    category: PolicyFailureCategory::InvalidObservation,
                    message: error.to_string(),
                };
                return self.failure_outcome(
                    agent,
                    incident,
                    action_context,
                    None,
                    None,
                    None,
                    None,
                );
            }
        };
        let stochastic = matches!(
            self.config.learning.implementation,
            crate::config::LearningKind::Ppo
        ) && matches!(
            agent.policy.policy_blueprint(),
            crate::brain::PolicyBlueprint::Neural
        );
        let evaluation_cap = action_context
            .policy_maximum_asset_allocation
            .or(action_context.market_maximum_asset_allocation)
            .unwrap_or(1.0);
        let runtime_before = agent.policy.runtime_state();
        let evaluation = match agent.policy.evaluate_action(
            &observation,
            &PolicyContext {
                tradable,
                current_allocations: allocations,
            },
            stochastic,
            evaluation_cap,
        ) {
            Ok(evaluation) => evaluation,
            Err(error) => {
                let incident = PolicyFailureIncident {
                    category: classify_policy_failure(&error),
                    message: error.to_string(),
                };
                return self.failure_outcome(
                    agent,
                    incident,
                    action_context,
                    None,
                    None,
                    None,
                    runtime_before,
                );
            }
        };
        let validated = match self
            .validator
            .validate(evaluation.proposed_allocation.clone(), action_context)
        {
            Ok(validated) => validated,
            Err(error) => {
                let incident = PolicyFailureIncident {
                    category: PolicyFailureCategory::InvalidOutput,
                    message: error.to_string(),
                };
                return self.failure_outcome(
                    agent,
                    incident,
                    action_context,
                    None,
                    Some(observation),
                    Some(evaluation),
                    runtime_before,
                );
            }
        };
        if validated.used_fallback() {
            let incident = PolicyFailureIncident {
                category: PolicyFailureCategory::InvalidOutput,
                message: validated
                    .fallback_reason()
                    .unwrap_or("invalid policy output")
                    .to_string(),
            };
            self.failure_outcome(
                agent,
                incident,
                action_context,
                Some(validated),
                Some(observation),
                Some(evaluation),
                runtime_before,
            )
        } else {
            agent.state.consecutive_policy_failures = 0;
            Ok(PolicyDecisionOutcome::Valid {
                validated,
                observation,
                evaluation,
                runtime_before,
            })
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn failure_outcome(
        &self,
        agent: &mut Agent,
        incident: PolicyFailureIncident,
        action_context: &ActionValidationContext<'_>,
        fallback: Option<ValidatedAllocation>,
        observation: Option<crate::observation::Observation>,
        evaluation: Option<crate::brain::PolicyEvaluation>,
        runtime_before: Option<crate::brain::NeuralRuntimeState>,
    ) -> Result<PolicyDecisionOutcome> {
        agent.state.total_policy_failures = agent.state.total_policy_failures.saturating_add(1);
        agent.state.consecutive_policy_failures =
            agent.state.consecutive_policy_failures.saturating_add(1);
        let limits_reached = agent.state.consecutive_policy_failures
            >= self.config.policy_failures.maximum_consecutive
            || agent.state.total_policy_failures >= self.config.policy_failures.maximum_total;
        match self.config.policy_failures.behavior {
            PolicyFailureBehavior::StopExperiment if limits_reached => {
                Ok(PolicyDecisionOutcome::FatalExperimentFailure { incident })
            }
            PolicyFailureBehavior::KillAgent if limits_reached => {
                Ok(PolicyDecisionOutcome::FatalAgentFailure { incident })
            }
            _ => {
                let fallback = match fallback {
                    Some(value) => value,
                    None => self.validator.validate(
                        ProposedAllocation::rebalance(vec![
                            f64::NAN;
                            action_context.asset_count + 1
                        ]),
                        action_context,
                    )?,
                };
                if !fallback.used_fallback() {
                    return Err(MidasError::Policy(
                        "cash fallback did not produce a validated fallback allocation".into(),
                    ));
                }
                Ok(PolicyDecisionOutcome::RecoverableFailure {
                    incident,
                    fallback,
                    observation,
                    evaluation,
                    runtime_before,
                })
            }
        }
    }

    fn policy_failure_event(
        &self,
        agent: &Agent,
        date: NaiveDate,
        incident: PolicyFailureIncident,
    ) -> DomainEvent {
        DomainEvent::PolicyFailure {
            agent_id: agent.state.id,
            date,
            policy_kind: agent.policy.policy_blueprint(),
            category: incident.category,
            message: incident.message,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn advance_agent(
        &self,
        agent: &mut Agent,
        validated: &ValidatedAllocation,
        _observation_date: NaiveDate,
        execution_date: NaiveDate,
        current_prices: &[Option<f64>],
        next_prices: &[Option<f64>],
        bars: &[Option<&crate::market::MarketBar>],
        actions: &[CorporateAction],
        invalid_action: bool,
    ) -> Result<AdvanceResult> {
        let mut events = Vec::new();
        let turnover_before = agent.state.portfolio.cumulative_turnover;
        // Actions take effect at the next open, in the canonical split then
        // dividend order supplied by MarketDataset.
        for action in actions {
            agent.state.portfolio.apply_corporate_action(action)?;
            events.push(DomainEvent::CorporateActionApplied {
                agent_id: agent.state.id,
                date: execution_date,
                action: action.clone(),
            });
        }
        let plan = self.planner.plan(
            &agent.state.portfolio,
            validated,
            &ExecutionSnapshot {
                bars,
                valuation_prices: current_prices,
            },
            self.config.simulation.minimum_trade_value,
        )?;
        let report = self.applier.apply(&mut agent.state.portfolio, &plan)?;
        for trade in report.trades {
            events.push(DomainEvent::TradeExecuted {
                agent_id: agent.state.id,
                date: execution_date,
                trade,
            });
        }
        for asset_id in report.rejected_assets {
            events.push(DomainEvent::TradeRejected {
                agent_id: agent.state.id,
                date: execution_date,
                asset_id,
            });
        }
        let previous = agent.state.portfolio.net_worth;
        agent.state.portfolio.previous_net_worth = previous;
        agent.state.portfolio.net_worth = agent.state.portfolio.value(next_prices)?;
        agent
            .state
            .portfolio
            .apply_survival_cost(self.config.simulation.daily_survival_cost);
        let current = agent.state.portfolio.net_worth;
        let bankrupt = current <= self.config.simulation.bankruptcy_threshold;
        agent.state.age += 1;
        agent.state.maximum_net_worth = agent.state.maximum_net_worth.max(current);
        if agent.state.maximum_net_worth > 0.0 {
            agent.state.maximum_drawdown = agent
                .state
                .maximum_drawdown
                .max(1.0 - current / agent.state.maximum_net_worth);
        }
        let reward = self.reward_function.reward(&RewardContext {
            previous_net_worth: previous,
            current_net_worth: current,
            maximum_drawdown: agent.state.maximum_drawdown,
            turnover: (agent.state.portfolio.cumulative_turnover - turnover_before).max(0.0),
            bankrupt,
            invalid_action,
        })?;
        events.push(DomainEvent::Valuation {
            agent_id: agent.state.id,
            date: execution_date,
            net_worth: current,
            previous,
            reward,
        });
        if bankrupt {
            agent.state.alive = false;
            events.push(DomainEvent::AgentDied {
                agent_id: agent.state.id,
                birth_date: agent.state.birth_date,
                death_date: execution_date,
                age: agent.state.age,
                starting_net_worth: self.config.simulation.starting_cash,
                final_net_worth: current,
                maximum_net_worth: agent.state.maximum_net_worth,
                cause: DeathCause::Bankruptcy,
            });
            return Ok((events, Some((agent.state.id, execution_date)), reward));
        }
        Ok((events, None, reward))
    }

    fn next_value_estimate(
        &self,
        agent: &Agent,
        step: usize,
        prices: &[Option<f64>],
        tradable: &[bool],
    ) -> Result<f32> {
        if !matches!(
            agent.policy.policy_blueprint(),
            crate::brain::PolicyBlueprint::Neural
        ) {
            return Ok(0.0);
        }
        let mut policy = agent.policy.fork().map_err(|error| {
            MidasError::Policy(format!("next-value policy fork failed: {error}"))
        })?;
        let market = self
            .features
            .get(step)
            .ok_or_else(|| MidasError::Data("next-value feature row is missing".into()))?;
        let observation = self.observation_builder.build(&ObservationContext {
            market,
            portfolio: &agent.state.portfolio,
            prices,
            starting_net_worth: self.config.simulation.starting_cash,
            maximum_net_worth: agent.state.maximum_net_worth,
            age: agent.state.age,
            episode_steps: self.features.len() - 1,
        })?;
        let allocations = agent.state.portfolio.allocations(prices)?;
        let maximum_asset_allocation = agent
            .state
            .genome
            .as_ref()
            .map(|genome| genome.traits.maximum_allocation as f64)
            .unwrap_or(self.config.brain.traits.maximum_allocation.initial as f64);
        let evaluation = policy.evaluate_action(
            &observation,
            &PolicyContext {
                tradable,
                current_allocations: &allocations,
            },
            false,
            maximum_asset_allocation,
        )?;
        if evaluation.value_estimate.is_finite() {
            Ok(evaluation.value_estimate)
        } else {
            Err(MidasError::Policy(
                "next-value estimate is non-finite".into(),
            ))
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn record_learning(
        &self,
        agent: &mut Agent,
        observation: Option<crate::observation::Observation>,
        evaluation: Option<crate::brain::PolicyEvaluation>,
        action: ValidatedAllocation,
        reward: f32,
        next_value: Option<f32>,
        done: bool,
        date: NaiveDate,
        tradable: &[bool],
        runtime_before: Option<crate::brain::NeuralRuntimeState>,
        invalid_action: bool,
        failure_category: Option<PolicyFailureCategory>,
        events: &mut Vec<DomainEvent>,
    ) -> Result<()> {
        let after = agent.policy.runtime_state();
        let action_sample = evaluation
            .as_ref()
            .and_then(|evaluation| evaluation.action_sample.clone());
        let log_probability = evaluation
            .as_ref()
            .and_then(|evaluation| evaluation.log_probability);
        let temperature = evaluation.as_ref().map(|evaluation| evaluation.temperature);
        let value_estimate = evaluation
            .as_ref()
            .map(|evaluation| evaluation.value_estimate)
            .filter(|value| value.is_finite());
        let actor_eligible = !invalid_action
            && observation.is_some()
            && runtime_before.is_some()
            && action_sample.is_some()
            && log_probability.is_some()
            && temperature.is_some();
        let critic_eligible =
            observation.is_some() && runtime_before.is_some() && value_estimate.is_some();
        let transition = Transition {
            observation,
            action,
            action_sample,
            raw_policy_output: evaluation.map(|evaluation| evaluation.raw_logits),
            log_probability,
            temperature,
            value_estimate,
            reward,
            next_value,
            done,
            truncated: false,
            recurrent_state_before: runtime_before,
            recurrent_state_after: after,
            date,
            tradable: tradable.to_vec(),
            invalid_action,
            failure_category,
            actor_eligible,
            critic_eligible,
        };
        if invalid_action {
            events.push(DomainEvent::LearningTransitionMasked {
                agent_id: agent.state.id,
                date,
                actor_eligible,
                critic_eligible,
                penalty: self.config.reward.invalid_action_penalty,
            });
        }
        let update = agent.learning.observe_transition(transition)?;
        self.apply_learning_update(agent, update, date, events)?;
        if done {
            self.finish_learning_episode(
                agent,
                date,
                agent.state.portfolio.net_worth,
                false,
                events,
            )?;
        }
        Ok(())
    }

    fn finish_learning_episode(
        &self,
        agent: &mut Agent,
        date: NaiveDate,
        final_value: f64,
        truncated: bool,
        events: &mut Vec<DomainEvent>,
    ) -> Result<()> {
        let update = agent.learning.end_episode(&EpisodeEndContext {
            date,
            final_value: final_value as f32,
            truncated,
        })?;
        self.apply_learning_update(agent, update, date, events)
    }

    fn apply_learning_update(
        &self,
        agent: &mut Agent,
        update: LearningUpdate,
        date: NaiveDate,
        events: &mut Vec<DomainEvent>,
    ) -> Result<()> {
        if let LearningUpdate::Ppo(batch) = update {
            let learning_config = agent
                .learning
                .training_config()
                .unwrap_or_else(|| self.config.learning.clone());
            let metrics = agent.policy.apply_learning(&batch, &learning_config)?;
            agent.learning.commit_update()?;
            events.push(DomainEvent::LearningUpdate {
                agent_id: agent.state.id,
                date,
                rollout_size: metrics.rollout_size,
                policy_loss: metrics.policy_loss,
                value_loss: metrics.value_loss,
                entropy: metrics.entropy,
                gradient_norm: metrics.gradient_norm,
            });
        } else {
            agent.learning.commit_update()?;
        }
        Ok(())
    }

    fn handle_death_in_transaction(
        &mut self,
        transaction: &mut StepTransaction,
        id: AgentId,
        date: NaiveDate,
    ) -> Result<()> {
        let dead_index = transaction
            .agents
            .iter()
            .position(|a| a.state.id == id)
            .ok_or_else(|| MidasError::Accounting("dead agent missing".into()))?;
        let dead_snapshot = transaction.agents[dead_index].state.clone();
        let birth_step = self
            .features
            .iter()
            .position(|feature| feature.date == dead_snapshot.birth_date)
            .unwrap_or(0);
        let life = CompletedLifeSummary {
            agent_id: id,
            genome_id: dead_snapshot.genome_id,
            birth_step,
            death_step: self.next_step,
            age: dead_snapshot.age,
            starting_net_worth: self.config.simulation.starting_cash,
            final_net_worth: dead_snapshot.portfolio.net_worth,
            maximum_drawdown: dead_snapshot.maximum_drawdown,
            turnover: dead_snapshot.portfolio.cumulative_turnover,
            policy_failures: dead_snapshot.total_policy_failures,
            fitness: None,
        };
        let fitness_function = DefaultFitnessFunction::from_config(&self.config.evolution.fitness);
        let fitness = transaction
            .evolution
            .record_life_with(life, &fitness_function)?;
        transaction.events.push(DomainEvent::FitnessAssigned {
            agent_id: id,
            genome_id: dead_snapshot.genome_id,
            fitness: fitness.0,
        });
        let living_genomes = transaction
            .agents
            .iter()
            .filter(|candidate| candidate.state.alive)
            .filter_map(|candidate| candidate.state.genome.as_ref())
            .collect::<Vec<_>>();
        transaction.evolution.update_diversity(&living_genomes)?;
        match self.config.simulation.death_handling {
            DeathHandling::LeaveVacant => Ok(()),
            DeathHandling::EndAgentOnly => {
                transaction.agents.retain(|a| a.state.id != id);
                Ok(())
            }
            DeathHandling::Replace => {
                let slot = transaction
                    .agents
                    .iter()
                    .position(|a| a.state.id == id)
                    .ok_or_else(|| MidasError::Accounting("dead agent missing".into()))?;
                let dead_state = transaction.agents[slot].state.clone();
                let dead_policy = transaction.agents[slot].policy.policy_blueprint();
                let new_id = AgentId(transaction.next_agent_id);
                transaction.next_agent_id += 1;
                let mut replacement = match transaction.replacement.take() {
                    Some(replacement) => replacement,
                    None => self.replacement.fork()?,
                };
                let request = {
                    let living = transaction
                        .agents
                        .iter()
                        .filter(|a| a.state.alive)
                        .map(|a| LivingAgentSummary {
                            id: a.state.id,
                            net_worth: a.state.portfolio.net_worth,
                            age: a.state.age,
                            policy: a.policy.policy_blueprint(),
                            genome: a.state.genome.as_ref(),
                            maximum_drawdown: a.state.maximum_drawdown,
                            birth_date: a.state.birth_date,
                            parent_ids: a.state.parent_ids.clone(),
                        })
                        .collect::<Vec<_>>();
                    replacement.create_replacement(&ReplacementContext {
                        dead_agent: &dead_state,
                        dead_policy,
                        living_population: &living,
                        config: &self.config,
                        architecture: &self.architecture,
                        compute_device: &self.compute_device,
                        observation_schema: &self.observation_schema,
                        deterministic_seed: transaction.evolution.rng.next_u64()
                            ^ self.config.simulation.seed
                            ^ new_id.0.wrapping_mul(0x9e3779b97f4a7c15),
                        completed_lives: &transaction.evolution.completed_lives,
                        genome_registry: &transaction.evolution.genome_registry,
                    })?
                };
                let mutation_report = replacement.mutation_report();
                let parent_ids = request.parent_ids.clone();
                let first_parent_id = parent_ids.first().copied().unwrap_or(AgentId(0));
                let request_reason = request.metadata.reason.clone();
                let diversity_relaxed = request.metadata.diversity_constraint_relaxed;
                let requested_mating_distance = request.metadata.requested_mating_distance;
                let selected_mating_distance = request.metadata.selected_mating_distance;
                let child_genome = request.genome.clone();
                transaction.replacement = Some(replacement);
                let mut agent = self.factory.spawn(
                    request,
                    &AgentBuildContext {
                        agent_id: new_id,
                        birth_date: date,
                        config: &self.config,
                        architecture: &self.architecture,
                        compute_device: &self.compute_device,
                    },
                )?;
                if let Some(genome) = child_genome.as_ref() {
                    let parent_genome_ids = parent_ids
                        .iter()
                        .filter_map(|parent_id| {
                            transaction
                                .agents
                                .iter()
                                .find(|candidate| candidate.state.id == *parent_id)
                                .and_then(|candidate| candidate.state.genome_id)
                        })
                        .collect::<Vec<_>>();
                    let parent_lineages = parent_genome_ids
                        .iter()
                        .filter_map(|parent_id| {
                            transaction.evolution.genome_registry.get(parent_id)
                        })
                        .map(|record| record.lineage_id)
                        .collect::<Vec<_>>();
                    let shared_lineage = if parent_lineages.len() >= 2
                        && parent_lineages.windows(2).all(|pair| pair[0] == pair[1])
                    {
                        parent_lineages.first().copied()
                    } else {
                        None
                    };
                    let (genome_id, lineage_id) =
                        transaction.evolution.allocate_genome_with_reason(
                            genome,
                            parent_genome_ids,
                            self.next_step,
                            shared_lineage,
                            request_reason.clone(),
                        )?;
                    agent.state.genome_id = Some(genome_id);
                    agent.state.lineage_id = Some(lineage_id);
                    if parent_lineages.len() >= 2
                        && parent_lineages.windows(2).any(|pair| pair[0] != pair[1])
                    {
                        transaction.events.push(DomainEvent::LineageCreated {
                            lineage_id,
                            parent_lineages,
                        });
                    }
                    if !parent_ids.is_empty() {
                        transaction.evolution.metrics.reproduction_count = transaction
                            .evolution
                            .metrics
                            .reproduction_count
                            .saturating_add(1);
                        transaction.events.push(DomainEvent::ParentsSelected {
                            child_agent_id: new_id,
                            parent_agent_ids: parent_ids,
                        });
                        if diversity_relaxed {
                            transaction
                                .evolution
                                .metrics
                                .diversity_constraint_relaxations = transaction
                                .evolution
                                .metrics
                                .diversity_constraint_relaxations
                                .saturating_add(1);
                            transaction
                                .events
                                .push(DomainEvent::DiversityConstraintRelaxed {
                                    first_parent_agent_id: first_parent_id,
                                    requested_minimum_distance: requested_mating_distance,
                                    selected_distance: selected_mating_distance,
                                });
                        }
                    }
                    if request_reason.starts_with("evolution_two_parent:") {
                        transaction.evolution.metrics.crossover_count = transaction
                            .evolution
                            .metrics
                            .crossover_count
                            .saturating_add(1);
                        transaction.evolution.metrics.mutation_count = transaction
                            .evolution
                            .metrics
                            .mutation_count
                            .saturating_add(1);
                        transaction.events.push(DomainEvent::GenomeCrossed {
                            child_agent_id: new_id,
                            method: self.config.evolution.crossover.method,
                        });
                        let report = mutation_report.unwrap_or_default();
                        transaction.events.push(DomainEvent::GenomeMutated {
                            child_agent_id: new_id,
                            weight_changes: report.weight_changes,
                            bias_changes: report.bias_changes,
                            trait_changes: report.trait_changes,
                            reset_changes: report.reset_changes,
                            weight_resets: report.weight_resets,
                            bias_resets: report.bias_resets,
                        });
                    } else if request_reason.contains("fallback") {
                        transaction.evolution.metrics.random_fallback_births = transaction
                            .evolution
                            .metrics
                            .random_fallback_births
                            .saturating_add(1);
                        transaction.events.push(DomainEvent::EvolutionFallback {
                            reason: request_reason,
                        });
                    }
                }
                transaction.agents[slot] = agent;
                let living_genomes = transaction
                    .agents
                    .iter()
                    .filter(|candidate| candidate.state.alive)
                    .filter_map(|candidate| candidate.state.genome.as_ref())
                    .collect::<Vec<_>>();
                transaction.evolution.update_diversity(&living_genomes)?;
                transaction.events.push(DomainEvent::AgentBorn {
                    agent_id: new_id,
                    date,
                });
                Ok(())
            }
        }
    }
    fn valuation_prices(&self, date: NaiveDate) -> Vec<Option<f64>> {
        (0..self.data.assets.len())
            .map(|i| self.data.last_close(date, AssetId(i as u64)))
            .collect()
    }
    fn emit(&mut self, event: DomainEvent) {
        self.metrics.on_event(&event);
        for sink in &mut self.event_sinks {
            sink.on_event(&event)
        }
        self.domain_events.push(event)
    }

    #[allow(clippy::too_many_arguments)]
    fn checkpoint_from_state(
        &self,
        next_step: usize,
        next_agent_id: u64,
        agents: &[Agent],
        metrics: MetricsCollector,
        domain_events: Vec<DomainEvent>,
        evolution: EvolutionState,
        lifecycle: SimulationLifecycle,
    ) -> Result<Checkpoint> {
        Ok(Checkpoint {
            version: CHECKPOINT_VERSION,
            config_hash: self.config.hash()?,
            dataset_identity: self.data.identity.clone(),
            observation_schema_hash: self.observation_schema.identity(self.data.assets.len())?,
            architecture: self.architecture.clone(),
            split: self.split,
            next_step,
            next_agent_id,
            lifecycle,
            agents: agents
                .iter()
                .map(|agent| {
                    Ok(AgentCheckpoint {
                        state: agent.state.clone(),
                        policy: agent.policy.snapshot()?,
                        learning: agent.learning_snapshot()?,
                    })
                })
                .collect::<Result<_>>()?,
            metrics,
            domain_events,
            accessed_through: self.features.get(next_step).map(|f| f.date),
            backend_identity: self.compute_device.identity(),
            evolution,
        })
    }

    fn write_checkpoint_file(&self, path: impl AsRef<Path>, checkpoint: &Checkpoint) -> Result<()> {
        if let Some(parent) = path.as_ref().parent() {
            fs::create_dir_all(parent)?
        }
        fs::write(path.as_ref(), bincode::serialize(checkpoint)?)?;
        Ok(())
    }

    pub fn save_checkpoint(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let checkpoint = self.checkpoint_from_state(
            self.next_step,
            self.next_agent_id,
            &self.agents,
            self.metrics.clone(),
            self.domain_events.clone(),
            self.evolution.clone(),
            self.lifecycle.clone(),
        )?;
        self.write_checkpoint_file(path.as_ref(), &checkpoint)?;
        let date = self
            .features
            .get(self.next_step)
            .map(|f| f.date)
            .unwrap_or_else(|| self.features.last().expect("features").date);
        self.operational_events
            .push(OperationalEvent::CheckpointSaved {
                date,
                path: path.as_ref().display().to_string(),
            });
        Ok(())
    }
    pub fn resume(config: Config, data: MarketDataset, path: impl AsRef<Path>) -> Result<Self> {
        Self::resume_with_factory(
            config,
            data,
            path,
            Box::new(StandardAgentFactory::default()),
        )
    }
    pub fn resume_with_factory(
        config: Config,
        mut data: MarketDataset,
        path: impl AsRef<Path>,
        factory: Box<dyn AgentFactory>,
    ) -> Result<Self> {
        let bytes = fs::read(path)?;
        let cp: Checkpoint = bincode::deserialize(&bytes).map_err(|_| {
            MidasError::Checkpoint(format!(
                "incompatible checkpoint; expected format version {CHECKPOINT_VERSION}; older files omit masked rollout eligibility and benchmark allocation semantics"
            ))
        })?;
        if cp.version != CHECKPOINT_VERSION {
            let detail = if cp.version == 6 {
                "; checkpoint 6 contains bias-only optimizer state and cannot be migrated safely"
            } else {
                ""
            };
            return Err(MidasError::Checkpoint(format!(
                "checkpoint version {} is incompatible with {CHECKPOINT_VERSION}{detail}",
                cp.version,
            )));
        }
        if cp.dataset_identity != data.identity {
            return Err(MidasError::Checkpoint("dataset identity mismatch".into()));
        }
        data.rebuild_index();
        let pipeline = BackwardFeaturePipeline::from_config(&config);
        let dates = data.split_dates(cp.split.range(&config));
        let features = pipeline.generate(&data, &dates)?;
        let observation_schema = ObservationSchema::from_config(&config);
        let schema_hash = observation_schema.identity(data.assets.len())?;
        if cp.observation_schema_hash != schema_hash {
            return Err(MidasError::Checkpoint("observation schema mismatch".into()));
        }
        let compute_device = config.backend.compute_device()?;
        if cp.backend_identity != compute_device.identity() {
            return Err(MidasError::Checkpoint(
                "checkpoint backend is incompatible".into(),
            ));
        }
        if cp.config_hash != config.hash()? {
            return Err(MidasError::Checkpoint("configuration mismatch".into()));
        }
        let architecture = BrainArchitecture::from_config(
            &config.brain,
            schema_hash,
            observation_schema.dimension(data.assets.len()),
            data.assets.len(),
        );
        if cp.architecture != architecture {
            return Err(MidasError::Checkpoint("brain architecture mismatch".into()));
        }
        if features.len() < 2 {
            return Err(MidasError::Checkpoint(
                "checkpoint split needs at least two trading dates".into(),
            ));
        }
        let observation_builder = SchemaObservationBuilder::new(
            observation_schema.clone(),
            config.features.features.clone(),
            data.assets.len(),
        )?;
        let birth = features[0].date;
        let agents = cp
            .agents
            .into_iter()
            .map(|a| {
                let agent_id = a.state.id;
                let mut agent = factory.restore(
                    a.state,
                    a.policy,
                    &AgentBuildContext {
                        agent_id,
                        birth_date: birth,
                        config: &config,
                        architecture: &architecture,
                        compute_device: &compute_device,
                    },
                )?;
                agent.restore_learning(a.learning)?;
                Ok(agent)
            })
            .collect::<Result<_>>()?;
        let costs = CostExecutionEngine {
            costs: config.execution.clone(),
        };
        let reward_function = configured_reward(&config.reward);
        let replacement: Box<dyn ReplacementStrategy> = if config.evolution.enabled {
            Box::new(crate::replacement::EvolutionaryReplacementStrategy::default())
        } else {
            Box::new(NonEvolutionaryRandomGenomeReplacement)
        };
        Ok(Self {
            config,
            data,
            split: cp.split,
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
            evolution: cp.evolution,
            metrics: cp.metrics,
            domain_events: cp.domain_events,
            operational_events: Vec::new(),
            event_sinks: Vec::new(),
            next_step: cp.next_step,
            next_agent_id: cp.next_agent_id,
            lifecycle: cp.lifecycle,
        })
    }

    /// Evaluate the lifetime parameters stored in a completed training
    /// checkpoint on a separate split. Evaluation starts fresh portfolios and
    /// recurrent state, and disables learning, replacement, and evolution.
    /// The checkpoint's immutable genomes and trained lifetime model state are
    /// restored, while optimizer/rollout state is intentionally not resumed.
    pub fn evaluate_checkpoint(
        config: Config,
        data: MarketDataset,
        path: impl AsRef<Path>,
        split: DatasetSplit,
    ) -> Result<Self> {
        let bytes = fs::read(path)?;
        let cp: Checkpoint = bincode::deserialize(&bytes).map_err(|_| {
            MidasError::Checkpoint(format!(
                "incompatible checkpoint; expected format version {CHECKPOINT_VERSION}"
            ))
        })?;
        if cp.version != CHECKPOINT_VERSION {
            return Err(MidasError::Checkpoint(format!(
                "checkpoint version {} is incompatible with {CHECKPOINT_VERSION}",
                cp.version
            )));
        }
        if cp.dataset_identity != data.identity {
            return Err(MidasError::Checkpoint("dataset identity mismatch".into()));
        }
        if cp.config_hash != config.hash()? {
            return Err(MidasError::Checkpoint(
                "evaluation configuration mismatch".into(),
            ));
        }

        let mut evaluation_config = config;
        evaluation_config.learning.implementation = LearningKind::NoLearning;
        evaluation_config.evolution.enabled = false;
        evaluation_config.simulation.death_handling = DeathHandling::LeaveVacant;
        let mut engine = Self::new(evaluation_config, data, split)?;
        let schema_hash = engine
            .observation_schema
            .identity(engine.data.assets.len())?;
        if cp.observation_schema_hash != schema_hash {
            return Err(MidasError::Checkpoint("observation schema mismatch".into()));
        }
        if cp.architecture != engine.architecture {
            return Err(MidasError::Checkpoint("brain architecture mismatch".into()));
        }
        if cp.backend_identity != engine.compute_device.identity() {
            return Err(MidasError::Checkpoint(
                "checkpoint backend is incompatible".into(),
            ));
        }
        let birth_date = engine.features[0].date;
        let factory = std::mem::replace(
            &mut engine.factory,
            Box::new(StandardAgentFactory::default()),
        );
        let mut restored = Vec::with_capacity(cp.agents.len());
        for checkpoint_agent in cp.agents {
            let agent_id = checkpoint_agent.state.id;
            let mut state = checkpoint_agent.state;
            state.birth_date = birth_date;
            state.age = 0;
            state.portfolio = Portfolio::new(
                engine.config.simulation.starting_cash,
                engine.architecture.asset_count,
            );
            state.maximum_net_worth = engine.config.simulation.starting_cash;
            state.maximum_drawdown = 0.0;
            state.alive = true;
            state.total_policy_failures = 0;
            state.consecutive_policy_failures = 0;
            let snapshot = reset_evaluation_snapshot(checkpoint_agent.policy)?;
            let mut agent = factory.restore(
                state,
                snapshot,
                &AgentBuildContext {
                    agent_id,
                    birth_date,
                    config: &engine.config,
                    architecture: &engine.architecture,
                    compute_device: &engine.compute_device,
                },
            )?;
            agent.restore_learning(LearningStrategyState::NoLearning)?;
            restored.push(agent);
        }
        engine.factory = factory;
        engine.agents = restored;
        engine.metrics = MetricsCollector::default();
        engine.domain_events.clear();
        engine.evolution = EvolutionState::new(engine.config.simulation.seed ^ 0x243f6a8885a308d3);
        engine.next_agent_id = engine
            .agents
            .iter()
            .map(|agent| agent.state.id.0)
            .max()
            .unwrap_or(0)
            .saturating_add(1);
        let agent_ids = engine
            .agents
            .iter()
            .map(|agent| agent.state.id)
            .collect::<Vec<_>>();
        for agent_id in agent_ids {
            let event = DomainEvent::AgentBorn {
                agent_id,
                date: birth_date,
            };
            engine.metrics.on_event(&event);
            engine.domain_events.push(event);
        }
        engine.lifecycle = SimulationLifecycle::Initialized;
        Ok(engine)
    }
    pub fn report(&self) -> SimulationReport {
        let learning_implementation = match self.config.learning.implementation {
            crate::config::LearningKind::NoLearning => "no_learning",
            crate::config::LearningKind::Ppo => "ppo",
        };
        SimulationReport {
            split: self.split,
            steps: self.next_step,
            agents: self
                .metrics
                .agents
                .iter()
                .map(|(id, m)| (*id, m.summary(self.config.simulation.starting_cash, 0.0)))
                .collect(),
            deaths: self.metrics.deaths,
            births: self.metrics.births,
            invalid_actions: self.metrics.invalid_actions,
            dataset_identity: self.data.identity.clone(),
            observation_schema_identity: self
                .observation_schema
                .identity(self.data.assets.len())
                .expect("serializable schema"),
            replacement_strategy: self.replacement.name().into(),
            learning_implementation: learning_implementation.into(),
            learning_updates: self.metrics.learning_updates,
            evolutionary_births: self.metrics.reproductions,
            active_lineages: self.evolution.metrics.active_lineages,
            lifecycle: self.lifecycle.clone(),
            failure_reason: match &self.lifecycle {
                SimulationLifecycle::Failed { reason } => Some(reason.clone()),
                _ => None,
            },
        }
    }
}
#[derive(Clone, Debug, Serialize)]
pub struct SimulationReport {
    pub split: DatasetSplit,
    pub steps: usize,
    pub agents: Vec<(AgentId, crate::metrics::PerformanceSummary)>,
    pub deaths: u64,
    pub births: u64,
    pub invalid_actions: u64,
    pub dataset_identity: String,
    pub observation_schema_identity: String,
    pub replacement_strategy: String,
    pub learning_implementation: String,
    pub learning_updates: u64,
    pub evolutionary_births: u64,
    pub active_lineages: u64,
    pub lifecycle: SimulationLifecycle,
    pub failure_reason: Option<String>,
}

fn reset_evaluation_snapshot(snapshot: PolicySnapshot) -> Result<PolicySnapshot> {
    match snapshot {
        PolicySnapshot::Neural {
            genome, state, rng, ..
        } => {
            let state = match *state {
                NeuralModelState::Burn {
                    parameters,
                    mut runtime,
                    optimizer,
                } => {
                    runtime.memory.fill(0.0);
                    runtime.inference_steps = 0;
                    NeuralModelState::Burn {
                        parameters,
                        runtime,
                        optimizer,
                    }
                }
                NeuralModelState::External { kind, mut state } => {
                    state.fill(0.0);
                    NeuralModelState::External { kind, state }
                }
            };
            Ok(PolicySnapshot::Neural {
                temperature: genome.traits.action_temperature,
                genome,
                state: Box::new(state),
                rng,
            })
        }
        other => Ok(other),
    }
}

fn classify_policy_failure(error: &MidasError) -> PolicyFailureCategory {
    let message = error.to_string().to_ascii_lowercase();
    if message.contains("schema") {
        PolicyFailureCategory::SchemaMismatch
    } else if message.contains("observation") {
        PolicyFailureCategory::InvalidObservation
    } else if message.contains("non-finite") || message.contains("finite") {
        PolicyFailureCategory::NonFiniteState
    } else if message.contains("output") || message.contains("logit") {
        PolicyFailureCategory::InvalidOutput
    } else {
        PolicyFailureCategory::InferenceFailure
    }
}
