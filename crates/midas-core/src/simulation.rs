use crate::{
    AgentId, AssetId, MidasError, Result,
    action::{
        ActionValidationContext, ActionValidator, LongOnlyActionValidator, ProposedAllocation,
    },
    agent::{
        Agent, AgentBuildContext, AgentFactory, AgentSpawnRequest, SpawnMetadata,
        StandardAgentFactory, configured_blueprint,
    },
    brain::{BrainArchitecture, BrainGenome, ComputeDevice, PolicyContext, PolicySnapshot},
    config::{Config, DateRange, DeathHandling, PolicyFailureBehavior},
    execution::{
        CostExecutionEngine, ExecutionApplier, ExecutionPlanner, ExecutionSnapshot, PlannedTrade,
    },
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
};
use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};
// Version 3 includes backend identity and the separated neural lifetime/runtime state.
pub const CHECKPOINT_VERSION: u32 = 3;
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
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum SimulationLifecycle {
    Initialized,
    Running,
    Completed,
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
pub enum PolicyFailureCategory {
    InvalidObservation,
    InferenceFailure,
    InvalidOutput,
    NonFiniteState,
    SchemaMismatch,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum OperationalEvent {
    CheckpointSaved { date: NaiveDate, path: String },
}
#[derive(Clone, Debug, Serialize, Deserialize)]
struct AgentCheckpoint {
    state: crate::agent::AgentState,
    policy: PolicySnapshot,
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
    pub metrics: MetricsCollector,
    pub domain_events: Vec<DomainEvent>,
    pub operational_events: Vec<OperationalEvent>,
    event_sinks: Vec<Box<dyn EventSink>>,
    next_step: usize,
    next_agent_id: u64,
    lifecycle: SimulationLifecycle,
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
                },
            };
            let agent = factory.spawn(
                request,
                &AgentBuildContext {
                    agent_id: id,
                    birth_date: first,
                    config: &config,
                    architecture: &architecture,
                    compute_device: &compute_device,
                },
            )?;
            let event = DomainEvent::AgentBorn {
                agent_id: id,
                date: first,
            };
            metrics.on_event(&event);
            domain_events.push(event);
            agents.push(agent);
        }
        let costs = CostExecutionEngine {
            costs: config.execution.clone(),
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
            replacement: Box::new(NonEvolutionaryRandomGenomeReplacement),
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
        self.lifecycle
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
    pub fn observation_schema(&self) -> &ObservationSchema {
        &self.observation_schema
    }
    pub fn add_event_sink(&mut self, sink: Box<dyn EventSink>) {
        self.event_sinks.push(sink)
    }
    pub fn set_replacement_strategy(&mut self, r: Box<dyn ReplacementStrategy>) {
        self.replacement = r
    }
    pub fn run(&mut self) -> Result<SimulationReport> {
        if self.lifecycle == SimulationLifecycle::Completed {
            return Ok(self.report());
        }
        self.lifecycle = SimulationLifecycle::Running;
        while self.next_step + 1 < self.features.len() {
            self.step()?;
        }
        self.lifecycle = SimulationLifecycle::Completed;
        let date = self.features.last().expect("validated features").date;
        self.emit(DomainEvent::TimelineCompleted { date });
        Ok(self.report())
    }
    pub fn run_steps(&mut self, count: usize) -> Result<()> {
        if self.lifecycle == SimulationLifecycle::Completed {
            return Ok(());
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
        let t = self.next_step;
        let observation_date = self.features[t].date;
        let execution_date = self.features[t + 1].date;
        self.emit(DomainEvent::DayStarted {
            date: execution_date,
            step: t,
        });
        self.emit(DomainEvent::ObservationRevealed {
            date: observation_date,
        });
        let current_prices = self.valuation_prices(observation_date);
        let next_prices = self.valuation_prices(execution_date);
        let bars = (0..self.data.assets.len())
            .map(|i| self.data.bar(execution_date, AssetId(i as u64)))
            .collect::<Vec<_>>();
        let tradable = self.features[t].tradable.clone();
        let market = self.features[t].clone();
        let actions = self
            .data
            .corporate_actions_on(execution_date)
            .cloned()
            .collect::<Vec<_>>();
        let mut deaths = Vec::new();
        let mut pending = Vec::new();
        let mut stop_experiment = false;
        for agent in &mut self.agents {
            if !agent.state.alive {
                continue;
            }
            let allocations = agent.state.portfolio.allocations(&current_prices)?;
            let mut policy_failure_kill = false;
            let mut policy_error_recorded = false;
            let policy_result = match self.observation_builder.build(&ObservationContext {
                market: &market,
                portfolio: &agent.state.portfolio,
                prices: &current_prices,
                starting_net_worth: self.config.simulation.starting_cash,
                maximum_net_worth: agent.state.maximum_net_worth,
                age: agent.state.age,
                episode_steps: self.features.len() - 1,
            }) {
                Ok(observation) => agent.policy.propose_action(
                    &observation,
                    &PolicyContext {
                        tradable: &tradable,
                        current_allocations: &allocations,
                    },
                ),
                Err(error) => Err(MidasError::Policy(format!("invalid observation: {error}"))),
            };
            let proposal = match policy_result {
                Ok(p) => {
                    agent.state.consecutive_policy_failures = 0;
                    p
                }
                Err(error) => {
                    policy_error_recorded = true;
                    agent.state.total_policy_failures =
                        agent.state.total_policy_failures.saturating_add(1);
                    agent.state.consecutive_policy_failures =
                        agent.state.consecutive_policy_failures.saturating_add(1);
                    let category = classify_policy_failure(&error);
                    pending.push(DomainEvent::PolicyFailure {
                        agent_id: agent.state.id,
                        date: observation_date,
                        policy_kind: agent.policy.policy_blueprint(),
                        category,
                        message: error.to_string(),
                    });
                    pending.push(DomainEvent::ActionFallback {
                        agent_id: agent.state.id,
                        date: observation_date,
                    });
                    let limits_reached = agent.state.consecutive_policy_failures
                        >= self.config.policy_failures.maximum_consecutive
                        || agent.state.total_policy_failures
                            >= self.config.policy_failures.maximum_total;
                    match self.config.policy_failures.behavior {
                        PolicyFailureBehavior::Continue => {}
                        PolicyFailureBehavior::KillAgent if limits_reached => {
                            policy_failure_kill = true
                        }
                        PolicyFailureBehavior::StopExperiment if limits_reached => {
                            stop_experiment = true
                        }
                        _ => {}
                    }
                    ProposedAllocation::rebalance(vec![f64::NAN; self.data.assets.len() + 1])
                }
            };
            if policy_failure_kill {
                agent.state.alive = false;
            }
            let maximum_asset_allocation = agent
                .state
                .genome
                .as_ref()
                .map(|genome| genome.traits.maximum_allocation as f64)
                .unwrap_or(self.config.brain.traits.maximum_allocation.initial as f64);
            let validated = self.validator.validate(
                proposal,
                &ActionValidationContext {
                    asset_count: self.data.assets.len(),
                    tradable: &tradable,
                    maximum_asset_allocation,
                },
            )?;
            if validated.used_fallback() {
                pending.push(DomainEvent::ActionFallback {
                    agent_id: agent.state.id,
                    date: observation_date,
                });
                if !policy_error_recorded {
                    agent.state.total_policy_failures =
                        agent.state.total_policy_failures.saturating_add(1);
                    agent.state.consecutive_policy_failures =
                        agent.state.consecutive_policy_failures.saturating_add(1);
                    pending.push(DomainEvent::PolicyFailure {
                        agent_id: agent.state.id,
                        date: observation_date,
                        policy_kind: agent.policy.policy_blueprint(),
                        category: PolicyFailureCategory::InvalidOutput,
                        message: validated
                            .fallback_reason()
                            .unwrap_or("invalid policy output")
                            .into(),
                    });
                    let limits_reached = agent.state.consecutive_policy_failures
                        >= self.config.policy_failures.maximum_consecutive
                        || agent.state.total_policy_failures
                            >= self.config.policy_failures.maximum_total;
                    match self.config.policy_failures.behavior {
                        PolicyFailureBehavior::Continue => {}
                        PolicyFailureBehavior::KillAgent if limits_reached => {
                            policy_failure_kill = true
                        }
                        PolicyFailureBehavior::StopExperiment if limits_reached => {
                            stop_experiment = true
                        }
                        _ => {}
                    }
                }
            }
            // Corporate actions are effective at the start of execution_date:
            // splits adjust holdings before opening orders and dividends credit
            // cash once before the planner evaluates those orders.
            for action in &actions {
                agent.state.portfolio.apply_corporate_action(action)?;
                pending.push(DomainEvent::CorporateActionApplied {
                    agent_id: agent.state.id,
                    date: execution_date,
                    action: action.clone(),
                });
            }
            let plan = self.planner.plan(
                &agent.state.portfolio,
                &validated,
                &ExecutionSnapshot {
                    bars: &bars,
                    valuation_prices: &current_prices,
                },
                self.config.simulation.minimum_trade_value,
            )?;
            let report = self.applier.apply(&mut agent.state.portfolio, &plan)?;
            for trade in report.trades {
                pending.push(DomainEvent::TradeExecuted {
                    agent_id: agent.state.id,
                    date: execution_date,
                    trade,
                });
            }
            for asset_id in report.rejected_assets {
                pending.push(DomainEvent::TradeRejected {
                    agent_id: agent.state.id,
                    date: execution_date,
                    asset_id,
                });
            }
            let previous = agent.state.portfolio.net_worth;
            agent.state.portfolio.previous_net_worth = previous;
            agent.state.portfolio.net_worth = agent.state.portfolio.value(&next_prices)?;
            agent
                .state
                .portfolio
                .apply_survival_cost(self.config.simulation.daily_survival_cost);
            let current = agent.state.portfolio.net_worth;
            let reward = if previous > 0.0 && current > 0.0 {
                (current / previous).ln() as f32
            } else {
                -1.0
            };
            agent.state.age += 1;
            agent.state.maximum_net_worth = agent.state.maximum_net_worth.max(current);
            if agent.state.maximum_net_worth > 0.0 {
                agent.state.maximum_drawdown = agent
                    .state
                    .maximum_drawdown
                    .max(1.0 - current / agent.state.maximum_net_worth);
            }
            pending.push(DomainEvent::Valuation {
                agent_id: agent.state.id,
                date: execution_date,
                net_worth: current,
                previous,
                reward,
            });
            if current <= self.config.simulation.bankruptcy_threshold || policy_failure_kill {
                agent.state.alive = false;
                deaths.push(agent.state.id);
                pending.push(DomainEvent::AgentDied {
                    agent_id: agent.state.id,
                    birth_date: agent.state.birth_date,
                    death_date: execution_date,
                    age: agent.state.age,
                    starting_net_worth: self.config.simulation.starting_cash,
                    final_net_worth: current,
                    maximum_net_worth: agent.state.maximum_net_worth,
                    cause: if policy_failure_kill {
                        DeathCause::PolicyFailure
                    } else {
                        DeathCause::Bankruptcy
                    },
                });
            }
        }
        for event in pending {
            self.emit(event)
        }
        if stop_experiment {
            return Err(MidasError::Policy(
                "policy failure limit reached; stopping experiment".into(),
            ));
        }
        for id in deaths {
            self.handle_death(id, execution_date)?
        }
        self.next_step += 1;
        if self.config.checkpoint.every_steps > 0
            && self
                .next_step
                .is_multiple_of(self.config.checkpoint.every_steps)
        {
            let path = self.config.checkpoint.path.clone();
            self.save_checkpoint(&path)?;
        }
        Ok(())
    }
    fn handle_death(&mut self, id: AgentId, date: NaiveDate) -> Result<()> {
        match self.config.simulation.death_handling {
            DeathHandling::LeaveVacant => Ok(()),
            DeathHandling::EndAgentOnly => {
                self.agents.retain(|a| a.state.id != id);
                Ok(())
            }
            DeathHandling::Replace => {
                let slot = self
                    .agents
                    .iter()
                    .position(|a| a.state.id == id)
                    .ok_or_else(|| MidasError::Accounting("dead agent missing".into()))?;
                let dead_state = self.agents[slot].state.clone();
                let dead_policy = self.agents[slot].policy.policy_blueprint();
                let new_id = AgentId(self.next_agent_id);
                self.next_agent_id += 1;
                let request = {
                    let living = self
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
                    self.replacement.create_replacement(&ReplacementContext {
                        dead_agent: &dead_state,
                        dead_policy,
                        living_population: &living,
                        config: &self.config,
                        architecture: &self.architecture,
                        compute_device: &self.compute_device,
                        observation_schema: &self.observation_schema,
                        deterministic_seed: self.config.simulation.seed
                            ^ new_id.0.wrapping_mul(0x9e3779b97f4a7c15),
                    })?
                };
                let agent = self.factory.spawn(
                    request,
                    &AgentBuildContext {
                        agent_id: new_id,
                        birth_date: date,
                        config: &self.config,
                        architecture: &self.architecture,
                        compute_device: &self.compute_device,
                    },
                )?;
                self.agents[slot] = agent;
                self.emit(DomainEvent::AgentBorn {
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
    pub fn save_checkpoint(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let checkpoint = Checkpoint {
            version: CHECKPOINT_VERSION,
            config_hash: self.config.hash()?,
            dataset_identity: self.data.identity.clone(),
            observation_schema_hash: self.observation_schema.identity(self.data.assets.len())?,
            architecture: self.architecture.clone(),
            split: self.split,
            next_step: self.next_step,
            next_agent_id: self.next_agent_id,
            lifecycle: self.lifecycle,
            agents: self
                .agents
                .iter()
                .map(|a| {
                    Ok(AgentCheckpoint {
                        state: a.state.clone(),
                        policy: a.policy.snapshot()?,
                    })
                })
                .collect::<Result<_>>()?,
            metrics: self.metrics.clone(),
            domain_events: self.domain_events.clone(),
            accessed_through: self.features.get(self.next_step).map(|f| f.date),
            backend_identity: self.compute_device.identity(),
        };
        if let Some(parent) = path.as_ref().parent() {
            fs::create_dir_all(parent)?
        }
        fs::write(path.as_ref(), bincode::serialize(&checkpoint)?)?;
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
    pub fn resume(config: Config, mut data: MarketDataset, path: impl AsRef<Path>) -> Result<Self> {
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
        let observation_builder = SchemaObservationBuilder::new(
            observation_schema.clone(),
            config.features.features.clone(),
            data.assets.len(),
        )?;
        let factory: Box<dyn AgentFactory> = Box::new(StandardAgentFactory::default());
        let birth = features[0].date;
        let agents = cp
            .agents
            .into_iter()
            .map(|a| {
                factory.restore(
                    a.state,
                    a.policy,
                    &AgentBuildContext {
                        agent_id: AgentId(0),
                        birth_date: birth,
                        config: &config,
                        architecture: &architecture,
                        compute_device: &compute_device,
                    },
                )
            })
            .collect::<Result<_>>()?;
        let costs = CostExecutionEngine {
            costs: config.execution.clone(),
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
            replacement: Box::new(NonEvolutionaryRandomGenomeReplacement),
            metrics: cp.metrics,
            domain_events: cp.domain_events,
            operational_events: Vec::new(),
            event_sinks: Vec::new(),
            next_step: cp.next_step,
            next_agent_id: cp.next_agent_id,
            lifecycle: cp.lifecycle,
        })
    }
    pub fn report(&self) -> SimulationReport {
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
            learning_implementation: "no_learning".into(),
            lifecycle: self.lifecycle,
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
    pub lifecycle: SimulationLifecycle,
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
