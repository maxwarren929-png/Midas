use crate::{
    AgentId, MidasError, Result,
    brain::{
        BrainArchitecture, BrainGenome, BurnNeuralModelFactory, BuyAndHoldPolicy, CashPolicy,
        ComputeDevice, EqualWeightPolicy, NeuralModelFactory, NeuralPolicy, Policy,
        PolicyBlueprint, PolicySnapshot, RandomPolicy,
    },
    config::Config,
    evolution::{GenomeId, LineageId},
    execution::Portfolio,
    learning::{
        LearningStrategy, LearningStrategyState, NoLearning, PpoLearning, learning_from_state,
    },
};
use chrono::NaiveDate;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AgentState {
    pub id: AgentId,
    pub birth_date: NaiveDate,
    pub age: u64,
    pub portfolio: Portfolio,
    pub maximum_net_worth: f64,
    pub maximum_drawdown: f64,
    pub alive: bool,
    pub genome: Option<BrainGenome>,
    #[serde(default)]
    pub genome_id: Option<GenomeId>,
    #[serde(default)]
    pub lineage_id: Option<LineageId>,
    pub parent_ids: Vec<AgentId>,
    pub total_policy_failures: u32,
    pub consecutive_policy_failures: u32,
}
pub struct Agent {
    pub state: AgentState,
    pub policy: Box<dyn Policy>,
    pub learning: Box<dyn LearningStrategy>,
}
impl Agent {
    pub fn fork(&self) -> Result<Self> {
        Ok(Self {
            state: self.state.clone(),
            policy: self.policy.fork()?,
            learning: self.learning.fork()?,
        })
    }
    pub fn learning_snapshot(&self) -> Result<LearningStrategyState> {
        self.learning.snapshot()
    }
    pub fn restore_learning(&mut self, state: LearningStrategyState) -> Result<()> {
        self.learning = learning_from_state(state)?;
        Ok(())
    }
}
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct SpawnMetadata {
    pub reason: String,
    #[serde(default)]
    pub diversity_constraint_relaxed: bool,
    #[serde(default)]
    pub requested_mating_distance: f64,
    #[serde(default)]
    pub selected_mating_distance: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AgentSpawnRequest {
    pub policy_blueprint: PolicyBlueprint,
    pub genome: Option<BrainGenome>,
    pub parent_ids: Vec<AgentId>,
    pub metadata: SpawnMetadata,
}
pub struct AgentBuildContext<'a> {
    pub agent_id: AgentId,
    pub birth_date: NaiveDate,
    pub config: &'a Config,
    pub architecture: &'a BrainArchitecture,
    pub compute_device: &'a ComputeDevice,
}
pub trait AgentFactory: Send + Sync {
    fn spawn(&self, request: AgentSpawnRequest, context: &AgentBuildContext<'_>) -> Result<Agent>;
    fn restore(
        &self,
        state: AgentState,
        snapshot: PolicySnapshot,
        context: &AgentBuildContext<'_>,
    ) -> Result<Agent>;
}
pub struct StandardAgentFactory {
    neural: Box<dyn NeuralModelFactory>,
}
impl Default for StandardAgentFactory {
    fn default() -> Self {
        Self {
            neural: Box::new(BurnNeuralModelFactory),
        }
    }
}
impl StandardAgentFactory {
    pub fn with_neural_factory(neural: Box<dyn NeuralModelFactory>) -> Self {
        Self { neural }
    }
    fn build_policy(
        &self,
        blueprint: &PolicyBlueprint,
        genome: Option<&BrainGenome>,
        a: &BrainArchitecture,
        seed: u64,
        device: &ComputeDevice,
    ) -> Result<Box<dyn Policy>> {
        Ok(match blueprint {
            PolicyBlueprint::Cash => Box::new(CashPolicy),
            PolicyBlueprint::Random => Box::new(RandomPolicy::new(seed)),
            PolicyBlueprint::EqualWeight => Box::new(EqualWeightPolicy),
            PolicyBlueprint::BuyAndHold { target_asset } => {
                Box::new(BuyAndHoldPolicy::new(*target_asset))
            }
            PolicyBlueprint::Neural => {
                let g = genome
                    .ok_or_else(|| MidasError::Policy("neural spawn requires genome".into()))?;
                Box::new(NeuralPolicy::new_with_seed(
                    g.clone(),
                    self.neural.build(g, a, device)?,
                    seed,
                ))
            }
        })
    }
    fn build_learning(
        &self,
        blueprint: &PolicyBlueprint,
        genome: Option<&BrainGenome>,
        config: &Config,
        seed: u64,
    ) -> Result<Box<dyn LearningStrategy>> {
        if !matches!(blueprint, PolicyBlueprint::Neural)
            || config.learning.implementation == crate::config::LearningKind::NoLearning
        {
            return Ok(Box::new(NoLearning));
        }
        let mut learning_config = config.learning.clone();
        if let Some(genome) = genome {
            learning_config.learning_rate = genome.traits.learning_rate;
            learning_config.entropy_coefficient = genome.traits.entropy_coefficient;
        }
        Ok(Box::new(PpoLearning::new(learning_config, seed)?))
    }
}
impl AgentFactory for StandardAgentFactory {
    fn spawn(&self, r: AgentSpawnRequest, c: &AgentBuildContext<'_>) -> Result<Agent> {
        if let Some(genome) = &r.genome {
            genome.validate_with_traits(c.architecture, &c.config.brain.traits)?;
        }
        let policy = self.build_policy(
            &r.policy_blueprint,
            r.genome.as_ref(),
            c.architecture,
            c.config.simulation.seed ^ c.agent_id.0,
            c.compute_device,
        )?;
        let learning = self.build_learning(
            &r.policy_blueprint,
            r.genome.as_ref(),
            c.config,
            c.config.simulation.seed ^ c.agent_id.0.wrapping_add(0x517cc1b727220a95),
        )?;
        Ok(Agent {
            state: AgentState {
                id: c.agent_id,
                birth_date: c.birth_date,
                age: 0,
                portfolio: Portfolio::new(
                    c.config.simulation.starting_cash,
                    c.architecture.asset_count,
                ),
                maximum_net_worth: c.config.simulation.starting_cash,
                maximum_drawdown: 0.0,
                alive: true,
                genome: r.genome,
                genome_id: None,
                lineage_id: None,
                parent_ids: r.parent_ids,
                total_policy_failures: 0,
                consecutive_policy_failures: 0,
            },
            policy,
            learning,
        })
    }
    fn restore(
        &self,
        state: AgentState,
        s: PolicySnapshot,
        c: &AgentBuildContext<'_>,
    ) -> Result<Agent> {
        if let Some(genome) = &state.genome {
            genome.validate_with_traits(c.architecture, &c.config.brain.traits)?;
        }
        let policy: Box<dyn Policy> = match s {
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
            PolicySnapshot::Neural {
                genome,
                state: model_state,
                rng,
                temperature,
            } => {
                if !temperature.is_finite() || temperature <= 0.0 {
                    return Err(MidasError::Checkpoint(
                        "checkpoint action temperature is invalid".into(),
                    ));
                }
                genome.validate_with_traits(c.architecture, &c.config.brain.traits)?;
                let state_genome = state.genome.as_ref().ok_or_else(|| {
                    MidasError::Checkpoint("neural state is missing genome".into())
                })?;
                if state_genome != &genome {
                    return Err(MidasError::Checkpoint(
                        "checkpoint neural genome does not match agent state".into(),
                    ));
                }
                let model =
                    self.neural
                        .restore(&genome, c.architecture, *model_state, c.compute_device)?;
                Box::new(NeuralPolicy::with_rng(genome, model, rng, temperature))
            }
        };
        let learning = self.build_learning(
            &policy.policy_blueprint(),
            state.genome.as_ref(),
            c.config,
            c.config.simulation.seed ^ state.id.0.wrapping_add(0x517cc1b727220a95),
        )?;
        Ok(Agent {
            state,
            policy,
            learning,
        })
    }
}

pub fn configured_blueprint(config: &Config) -> Result<PolicyBlueprint> {
    Ok(match config.brain.policy {
        crate::config::PolicyKind::Cash => PolicyBlueprint::Cash,
        crate::config::PolicyKind::Random => PolicyBlueprint::Random,
        crate::config::PolicyKind::EqualWeight => PolicyBlueprint::EqualWeight,
        crate::config::PolicyKind::BuyAndHold => PolicyBlueprint::BuyAndHold {
            target_asset: config
                .market
                .assets
                .iter()
                .position(|a| a == &config.market.benchmark_asset)
                .ok_or_else(|| MidasError::Config("benchmark asset missing".into()))?,
        },
        crate::config::PolicyKind::BurnActorCritic => PolicyBlueprint::Neural,
    })
}
