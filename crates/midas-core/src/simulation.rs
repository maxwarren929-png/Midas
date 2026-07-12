use crate::{
    AgentId, AssetId, MidasError, Result,
    brain::{
        BrainGenome, BurnActorCriticPolicy, BuyAndHoldPolicy, CashPolicy, EqualWeightPolicy,
        Policy, PolicyInput, PolicySnapshot, RandomPolicy,
    },
    config::{Config, DateRange, PolicyKind},
    execution::{CostExecutionModel, ExecutionModel, Portfolio, Trade},
    market::{BackwardFeaturePipeline, FeaturePipeline, FeatureRow, MarketDataset},
    metrics::{EventSink, MetricsCollector},
};
use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

pub const CHECKPOINT_VERSION: u32 = 1;

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

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum DeathCause {
    Bankruptcy,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum SimulationEvent {
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
    TradeExecuted {
        agent_id: AgentId,
        date: NaiveDate,
        trade: Trade,
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
    CheckpointSaved {
        date: NaiveDate,
        path: String,
    },
    TimelineCompleted {
        date: NaiveDate,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AgentState {
    pub id: AgentId,
    pub birth_date: NaiveDate,
    pub age: u64,
    pub portfolio: Portfolio,
    pub maximum_net_worth: f64,
    pub alive: bool,
    pub genome: Option<BrainGenome>,
}
struct Agent {
    state: AgentState,
    policy: Box<dyn Policy>,
}

pub trait ReplacementStrategy: Send {
    fn replacement_genome(
        &mut self,
        config: &Config,
        observation_size: usize,
        asset_count: usize,
        agent_id: AgentId,
    ) -> Result<BrainGenome>;
    fn name(&self) -> &'static str;
}
pub struct NonEvolutionaryRandomGenomeReplacement;
impl ReplacementStrategy for NonEvolutionaryRandomGenomeReplacement {
    fn replacement_genome(
        &mut self,
        c: &Config,
        o: usize,
        a: usize,
        id: AgentId,
    ) -> Result<BrainGenome> {
        BrainGenome::random(
            &c.brain,
            o,
            a,
            c.simulation.seed ^ id.0.wrapping_mul(0x9e3779b97f4a7c15),
        )
    }
    fn name(&self) -> &'static str {
        "non_evolutionary_random_genome"
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct AgentCheckpoint {
    state: AgentState,
    policy: PolicySnapshot,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Checkpoint {
    version: u32,
    config_hash: String,
    dataset_identity: String,
    feature_schema_identity: String,
    split: DatasetSplit,
    next_step: usize,
    next_agent_id: u64,
    agents: Vec<AgentCheckpoint>,
    metrics: MetricsCollector,
    events: Vec<SimulationEvent>,
    accessed_through: Option<NaiveDate>,
}

pub struct SimulationEngine {
    config: Config,
    data: MarketDataset,
    split: DatasetSplit,
    features: Vec<FeatureRow>,
    feature_schema_identity: String,
    agents: Vec<Agent>,
    execution: Box<dyn ExecutionModel>,
    replacement: Box<dyn ReplacementStrategy>,
    event_sinks: Vec<Box<dyn EventSink>>,
    pub metrics: MetricsCollector,
    pub events: Vec<SimulationEvent>,
    next_step: usize,
    next_agent_id: u64,
}

impl SimulationEngine {
    pub fn new(config: Config, data: MarketDataset, split: DatasetSplit) -> Result<Self> {
        let pipeline = BackwardFeaturePipeline::from_config(&config);
        let dates = data.split_dates(split.range(&config));
        if dates.len() < 2 {
            return Err(MidasError::Data(
                "selected split needs at least two trading dates".into(),
            ));
        }
        let features = pipeline.generate(&data, &dates)?;
        let schema_id = pipeline.schema().identity()?;
        let observation_size =
            observation_size(pipeline.schema().market_size(), config.market.assets.len());
        let first = features[0].date;
        let mut agents = Vec::new();
        let mut metrics = MetricsCollector::default();
        let mut events = Vec::new();
        for i in 0..config.simulation.population_size {
            let id = AgentId(i as u64);
            let (policy, genome) = new_policy(&config, observation_size, id)?;
            let state = AgentState {
                id,
                birth_date: first,
                age: 0,
                portfolio: Portfolio::new(
                    config.simulation.starting_cash,
                    config.market.assets.len(),
                ),
                maximum_net_worth: config.simulation.starting_cash,
                alive: true,
                genome,
            };
            let event = SimulationEvent::AgentBorn {
                agent_id: id,
                date: first,
            };
            metrics.on_event(&event);
            events.push(event);
            agents.push(Agent { state, policy });
        }
        let next_agent_id = config.simulation.population_size as u64;
        Ok(Self {
            execution: Box::new(CostExecutionModel {
                costs: config.execution.clone(),
            }),
            replacement: Box::new(NonEvolutionaryRandomGenomeReplacement),
            event_sinks: Vec::new(),
            config,
            data,
            split,
            features,
            feature_schema_identity: schema_id,
            agents,
            metrics,
            events,
            next_step: 0,
            next_agent_id,
        })
    }

    pub fn run(&mut self) -> Result<SimulationReport> {
        while self.next_step + 1 < self.features.len() {
            self.step()?;
        }
        let date = self.features.last().expect("validated feature rows").date;
        self.emit(SimulationEvent::TimelineCompleted { date });
        Ok(self.report())
    }
    pub fn set_execution_model(&mut self, execution: Box<dyn ExecutionModel>) {
        self.execution = execution;
    }
    pub fn set_replacement_strategy(&mut self, replacement: Box<dyn ReplacementStrategy>) {
        self.replacement = replacement;
    }
    pub fn add_event_sink(&mut self, sink: Box<dyn EventSink>) {
        self.event_sinks.push(sink);
    }
    pub fn run_steps(&mut self, count: usize) -> Result<()> {
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
        self.emit(SimulationEvent::DayStarted {
            date: execution_date,
            step: t,
        });
        self.emit(SimulationEvent::ObservationRevealed {
            date: observation_date,
        });
        let current_closes = self.valuation_prices(observation_date);
        let next_closes = self.valuation_prices(execution_date);
        let bars = (0..self.data.assets.len())
            .map(|i| self.data.bar(execution_date, AssetId(i as u64)))
            .collect::<Vec<_>>();
        let market_values = self.features[t].values.clone();
        let tradable = self.features[t].tradable.clone();
        let total_steps = self.features.len() - 1;
        let mut deaths = Vec::new();
        let mut pending_events = Vec::new();
        for agent in &mut self.agents {
            if !agent.state.alive {
                continue;
            }
            let allocations = agent.state.portfolio.allocations(&current_closes)?;
            let obs = build_observation(&market_values, &agent.state, &allocations, total_steps);
            let output = agent.policy.act(PolicyInput {
                observation: &obs,
                tradable: &tradable,
                current_allocations: &allocations,
            })?;
            if !output.valid {
                pending_events.push(SimulationEvent::ActionFallback {
                    agent_id: agent.state.id,
                    date: observation_date,
                });
            }
            let report = if output.rebalance {
                self.execution.rebalance(
                    &mut agent.state.portfolio,
                    &output.allocations,
                    &bars,
                    &current_closes,
                    self.config.simulation.minimum_trade_value,
                )?
            } else {
                Default::default()
            };
            if report.invalid_action {
                pending_events.push(SimulationEvent::ActionFallback {
                    agent_id: agent.state.id,
                    date: observation_date,
                });
            }
            for trade in report.trades {
                pending_events.push(SimulationEvent::TradeExecuted {
                    agent_id: agent.state.id,
                    date: execution_date,
                    trade,
                })
            }
            for asset_id in report.rejected_assets {
                pending_events.push(SimulationEvent::TradeRejected {
                    agent_id: agent.state.id,
                    date: execution_date,
                    asset_id,
                })
            }
            let previous = agent.state.portfolio.net_worth;
            agent.state.portfolio.previous_net_worth = previous;
            agent.state.portfolio.net_worth = agent.state.portfolio.value(&next_closes)?;
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
            pending_events.push(SimulationEvent::Valuation {
                agent_id: agent.state.id,
                date: execution_date,
                net_worth: current,
                previous,
                reward,
            });
            if current <= self.config.simulation.bankruptcy_threshold {
                agent.state.alive = false;
                deaths.push(agent.state.id);
                pending_events.push(SimulationEvent::AgentDied {
                    agent_id: agent.state.id,
                    birth_date: agent.state.birth_date,
                    death_date: execution_date,
                    age: agent.state.age,
                    starting_net_worth: self.config.simulation.starting_cash,
                    final_net_worth: current,
                    maximum_net_worth: agent.state.maximum_net_worth,
                    cause: DeathCause::Bankruptcy,
                });
            }
        }
        for event in pending_events {
            self.emit(event)
        }
        for id in deaths {
            self.replace(id, execution_date)?
        }
        self.next_step += 1;
        if self.config.checkpoint.every_steps > 0
            && self
                .next_step
                .is_multiple_of(self.config.checkpoint.every_steps)
        {
            let path = self.config.checkpoint.path.clone();
            self.save_checkpoint(&path)?;
            self.emit(SimulationEvent::CheckpointSaved {
                date: execution_date,
                path,
            });
        }
        Ok(())
    }
    fn replace(&mut self, id: AgentId, date: NaiveDate) -> Result<()> {
        let slot = self
            .agents
            .iter()
            .position(|a| a.state.id == id)
            .ok_or_else(|| MidasError::Accounting("dead agent missing".into()))?;
        let new_id = AgentId(self.next_agent_id);
        self.next_agent_id += 1;
        let obs = self.features[0].values.len() + self.data.assets.len() * 2 + 5;
        let genome = self.replacement.replacement_genome(
            &self.config,
            obs,
            self.data.assets.len(),
            new_id,
        )?;
        let policy = BurnActorCriticPolicy::new(genome.clone())?;
        let state = AgentState {
            id: new_id,
            birth_date: date,
            age: 0,
            portfolio: Portfolio::new(self.config.simulation.starting_cash, self.data.assets.len()),
            maximum_net_worth: self.config.simulation.starting_cash,
            alive: true,
            genome: Some(genome),
        };
        self.agents[slot] = Agent {
            state,
            policy: Box::new(policy),
        };
        self.emit(SimulationEvent::AgentBorn {
            agent_id: new_id,
            date,
        });
        Ok(())
    }
    fn valuation_prices(&self, date: NaiveDate) -> Vec<Option<f64>> {
        (0..self.data.assets.len())
            .map(|i| self.data.last_close(date, AssetId(i as u64)))
            .collect()
    }
    fn emit(&mut self, event: SimulationEvent) {
        self.metrics.on_event(&event);
        for sink in &mut self.event_sinks {
            sink.on_event(&event);
        }
        self.events.push(event)
    }
    pub fn save_checkpoint(&self, path: impl AsRef<Path>) -> Result<()> {
        let checkpoint = Checkpoint {
            version: CHECKPOINT_VERSION,
            config_hash: self.config.hash()?,
            dataset_identity: self.data.identity.clone(),
            feature_schema_identity: self.feature_schema_identity.clone(),
            split: self.split,
            next_step: self.next_step,
            next_agent_id: self.next_agent_id,
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
            events: self.events.clone(),
            accessed_through: self.features.get(self.next_step).map(|f| f.date),
        };
        if let Some(parent) = path.as_ref().parent() {
            fs::create_dir_all(parent)?
        }
        fs::write(path, bincode::serialize(&checkpoint)?)?;
        Ok(())
    }
    pub fn resume(config: Config, mut data: MarketDataset, path: impl AsRef<Path>) -> Result<Self> {
        let cp: Checkpoint = bincode::deserialize(&fs::read(path)?)?;
        if cp.version != CHECKPOINT_VERSION {
            return Err(MidasError::Checkpoint("checkpoint version mismatch".into()));
        }
        if cp.config_hash != config.hash()? || cp.dataset_identity != data.identity {
            return Err(MidasError::Checkpoint(
                "configuration or dataset identity mismatch".into(),
            ));
        }
        data.rebuild_index();
        let pipeline = BackwardFeaturePipeline::from_config(&config);
        if cp.feature_schema_identity != pipeline.schema().identity()? {
            return Err(MidasError::Checkpoint("feature schema mismatch".into()));
        }
        let dates = data.split_dates(cp.split.range(&config));
        let features = pipeline.generate(&data, &dates)?;
        let agents = cp
            .agents
            .into_iter()
            .map(|a| {
                Ok(Agent {
                    state: a.state,
                    policy: restore_policy(a.policy)?,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            execution: Box::new(CostExecutionModel {
                costs: config.execution.clone(),
            }),
            replacement: Box::new(NonEvolutionaryRandomGenomeReplacement),
            event_sinks: Vec::new(),
            config,
            data,
            split: cp.split,
            features,
            feature_schema_identity: cp.feature_schema_identity,
            agents,
            metrics: cp.metrics,
            events: cp.events,
            next_step: cp.next_step,
            next_agent_id: cp.next_agent_id,
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
            feature_schema_identity: self.feature_schema_identity.clone(),
            replacement_strategy: self.replacement.name().into(),
            learning_implementation: "no_learning".into(),
        }
    }
}

fn observation_size(market: usize, assets: usize) -> usize {
    market + assets * 2 + 5
}
fn build_observation(
    market: &[f32],
    state: &AgentState,
    allocations: &[f64],
    total_steps: usize,
) -> Vec<f32> {
    let mut o = market.to_vec();
    o.push(allocations[0] as f32);
    o.extend(allocations.iter().skip(1).map(|v| *v as f32));
    o.extend(state.portfolio.positions.iter().map(|p| {
        if p.average_entry_price > 0.0 {
            ((p.quantity * p.average_entry_price) / state.portfolio.net_worth.max(1e-9))
                .clamp(-10.0, 10.0) as f32
        } else {
            0.0
        }
    }));
    o.push((state.portfolio.net_worth / state.maximum_net_worth.max(1e-9)).clamp(0.0, 1.0) as f32);
    o.push(
        (state.portfolio.net_worth / state.portfolio.previous_net_worth.max(1e-9) - 1.0)
            .clamp(-10.0, 10.0) as f32,
    );
    o.push(
        (state.portfolio.cumulative_turnover / state.portfolio.net_worth.max(1e-9)).clamp(0.0, 10.0)
            as f32,
    );
    o.push((state.age as f64 / total_steps.max(1) as f64).clamp(0.0, 1.0) as f32);
    o
}
fn new_policy(
    c: &Config,
    obs: usize,
    id: AgentId,
) -> Result<(Box<dyn Policy>, Option<BrainGenome>)> {
    Ok(match c.brain.policy {
        PolicyKind::Cash => (Box::new(CashPolicy), None),
        PolicyKind::Random => (Box::new(RandomPolicy::new(c.simulation.seed ^ id.0)), None),
        PolicyKind::EqualWeight => (Box::new(EqualWeightPolicy), None),
        PolicyKind::BuyAndHold => {
            let target = c
                .market
                .assets
                .iter()
                .position(|a| a == &c.market.benchmark_asset)
                .ok_or_else(|| MidasError::Config("benchmark asset missing".into()))?;
            (Box::new(BuyAndHoldPolicy::new(target)), None)
        }
        PolicyKind::BurnActorCritic => {
            let g = BrainGenome::random(
                &c.brain,
                obs,
                c.market.assets.len(),
                c.simulation.seed ^ id.0,
            )?;
            (Box::new(BurnActorCriticPolicy::new(g.clone())?), Some(g))
        }
    })
}
fn restore_policy(s: PolicySnapshot) -> Result<Box<dyn Policy>> {
    Ok(match s {
        PolicySnapshot::Cash => Box::new(CashPolicy),
        PolicySnapshot::Random { rng } => Box::new(RandomPolicy { rng }),
        PolicySnapshot::EqualWeight => Box::new(EqualWeightPolicy),
        PolicySnapshot::BuyAndHold {
            initialized,
            target_asset,
        } => Box::new(BuyAndHoldPolicy {
            initialized,
            target_asset,
        }),
        PolicySnapshot::Burn { genome, lifetime } => Box::new(BurnActorCriticPolicy {
            genome,
            lifetime: *lifetime,
        }),
    })
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
    pub feature_schema_identity: String,
    pub replacement_strategy: String,
    pub learning_implementation: String,
}
