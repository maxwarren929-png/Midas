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
// Version 4 includes failed lifecycle metadata and transactional step semantics.
pub const CHECKPOINT_VERSION: u32 = 4;
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
struct StepTransaction {
    agents: Vec<Agent>,
    next_agent_id: u64,
    events: Vec<DomainEvent>,
    replacement: Option<Box<dyn ReplacementStrategy>>,
}
type AdvanceResult = (Vec<DomainEvent>, Option<(AgentId, NaiveDate)>);
#[derive(Clone, Debug)]
struct PolicyFailureIncident {
    category: PolicyFailureCategory,
    message: String,
}
enum PolicyDecisionOutcome {
    Valid(crate::action::ValidatedAllocation),
    RecoverableFailure {
        incident: PolicyFailureIncident,
        fallback: crate::action::ValidatedAllocation,
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
        match &self.lifecycle {
            SimulationLifecycle::Completed => return Ok(self.report()),
            SimulationLifecycle::Failed { reason } => {
                return Err(MidasError::Policy(reason.clone()));
            }
            SimulationLifecycle::Initialized | SimulationLifecycle::Running => {}
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
        match &self.lifecycle {
            SimulationLifecycle::Completed => return Ok(()),
            SimulationLifecycle::Failed { reason } => {
                return Err(MidasError::Policy(reason.clone()));
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
        };
        let mut transaction = transaction;
        match self.execute_step(&mut transaction) {
            Ok(()) => self.commit_step(transaction),
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
                    self.factory
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
                        })
                }
            })
            .collect()
    }

    fn commit_step(&mut self, transaction: StepTransaction) -> Result<()> {
        let StepTransaction {
            agents,
            next_agent_id,
            events,
            replacement,
        } = transaction;
        self.agents = agents;
        self.next_agent_id = next_agent_id;
        if let Some(replacement) = replacement {
            self.replacement = replacement;
        }
        for event in events {
            self.emit(event);
        }
        self.next_step += 1;
        if self.config.checkpoint.every_steps > 0
            && self
                .next_step
                .is_multiple_of(self.config.checkpoint.every_steps)
        {
            let path = self.config.checkpoint.path.clone();
            self.save_checkpoint(&path).inspect_err(|error| {
                let reason = error.to_string();
                self.lifecycle = SimulationLifecycle::Failed {
                    reason: reason.clone(),
                };
            })?;
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
            let action_context = ActionValidationContext {
                asset_count: self.data.assets.len(),
                tradable: &tradable,
                maximum_asset_allocation,
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
                PolicyDecisionOutcome::Valid(validated) => {
                    let (events, death) = self.advance_agent(
                        agent,
                        &validated,
                        observation_date,
                        execution_date,
                        &current_prices,
                        &next_prices,
                        &bars,
                        &actions,
                    )?;
                    transaction.events.extend(events);
                    if let Some(death) = death {
                        deaths.push(death);
                    }
                }
                PolicyDecisionOutcome::RecoverableFailure { incident, fallback } => {
                    let failure_event =
                        self.policy_failure_event(agent, observation_date, incident.clone());
                    transaction.events.push(failure_event);
                    transaction.events.push(DomainEvent::ActionFallback {
                        agent_id: agent.state.id,
                        date: observation_date,
                    });
                    let (events, death) = self.advance_agent(
                        agent,
                        &fallback,
                        observation_date,
                        execution_date,
                        &current_prices,
                        &next_prices,
                        &bars,
                        &actions,
                    )?;
                    transaction.events.extend(events);
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
        let observation = self.observation_builder.build(&ObservationContext {
            market,
            portfolio: &agent.state.portfolio,
            prices,
            starting_net_worth: self.config.simulation.starting_cash,
            maximum_net_worth: agent.state.maximum_net_worth,
            age: agent.state.age,
            episode_steps: self.features.len() - 1,
        });
        let proposal = match observation {
            Ok(observation) => agent.policy.propose_action(
                &observation,
                &PolicyContext {
                    tradable,
                    current_allocations: allocations,
                },
            ),
            Err(error) => Err(MidasError::Policy(format!("invalid observation: {error}"))),
        };
        match proposal {
            Ok(proposal) => {
                let validated = self.validator.validate(proposal, action_context)?;
                if validated.used_fallback() {
                    let incident = PolicyFailureIncident {
                        category: PolicyFailureCategory::InvalidOutput,
                        message: validated
                            .fallback_reason()
                            .unwrap_or("invalid policy output")
                            .to_string(),
                    };
                    self.failure_outcome(agent, incident, action_context, Some(validated))
                } else {
                    agent.state.consecutive_policy_failures = 0;
                    Ok(PolicyDecisionOutcome::Valid(validated))
                }
            }
            Err(error) => {
                let incident = PolicyFailureIncident {
                    category: classify_policy_failure(&error),
                    message: error.to_string(),
                };
                self.failure_outcome(agent, incident, action_context, None)
            }
        }
    }

    fn failure_outcome(
        &self,
        agent: &mut Agent,
        incident: PolicyFailureIncident,
        action_context: &ActionValidationContext<'_>,
        fallback: Option<ValidatedAllocation>,
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
                Ok(PolicyDecisionOutcome::RecoverableFailure { incident, fallback })
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
    ) -> Result<AdvanceResult> {
        let mut events = Vec::new();
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
        events.push(DomainEvent::Valuation {
            agent_id: agent.state.id,
            date: execution_date,
            net_worth: current,
            previous,
            reward,
        });
        if current <= self.config.simulation.bankruptcy_threshold {
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
            return Ok((events, Some((agent.state.id, execution_date))));
        }
        Ok((events, None))
    }

    fn handle_death_in_transaction(
        &mut self,
        transaction: &mut StepTransaction,
        id: AgentId,
        date: NaiveDate,
    ) -> Result<()> {
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
                        deterministic_seed: self.config.simulation.seed
                            ^ new_id.0.wrapping_mul(0x9e3779b97f4a7c15),
                    })?
                };
                transaction.replacement = Some(replacement);
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
                transaction.agents[slot] = agent;
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
            lifecycle: self.lifecycle.clone(),
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
    pub lifecycle: SimulationLifecycle,
    pub failure_reason: Option<String>,
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
