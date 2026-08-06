use crate::{
    AgentId, MidasError, Result,
    brain::{
        BrainArchitecture, BrainGenome, BurnNeuralModelFactory, BuyAndHoldPolicy, CashPolicy,
        ComputeDevice, EqualWeightPolicy, MomentumPolicy, NeuralModelFactory, NeuralPolicy, Policy,
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

pub const REWARD_EMA_ALPHA: f64 = 0.05;
const SHORT_REWARD_EMA_ALPHA: f64 = 0.20;

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
    #[serde(default)]
    pub reward_ema: f64,
    #[serde(default)]
    pub reward_variance_ema: f64,
    #[serde(default)]
    pub last_reward: f64,
    #[serde(default)]
    pub short_reward_ema: f64,
    #[serde(default)]
    pub long_reward_ema: f64,
    #[serde(default)]
    pub reward_trend: f64,
}
impl AgentState {
    pub fn update_reward_statistics(&mut self, reward: f32) {
        let reward = f64::from(reward);
        self.last_reward = reward;
        self.reward_ema = REWARD_EMA_ALPHA * reward + (1.0 - REWARD_EMA_ALPHA) * self.reward_ema;
        let deviation = (reward - self.reward_ema).powi(2);
        self.reward_variance_ema =
            REWARD_EMA_ALPHA * deviation + (1.0 - REWARD_EMA_ALPHA) * self.reward_variance_ema;
        self.short_reward_ema = SHORT_REWARD_EMA_ALPHA * reward
            + (1.0 - SHORT_REWARD_EMA_ALPHA) * self.short_reward_ema;
        self.long_reward_ema =
            REWARD_EMA_ALPHA * reward + (1.0 - REWARD_EMA_ALPHA) * self.long_reward_ema;
        self.reward_trend = (self.short_reward_ema - self.long_reward_ema).signum();
    }
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
        maximum_trade_fraction: f64,
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
                Box::new(NeuralPolicy::new_with_seed_and_trade_fraction(
                    g.clone(),
                    self.neural.build(g, a, device)?,
                    seed,
                    maximum_trade_fraction,
                ))
            }
            PolicyBlueprint::Momentum {
                signal_indices,
                minimum_signal_bits,
            } => Box::new(MomentumPolicy::new(
                signal_indices.clone(),
                f32::from_bits(*minimum_signal_bits),
            )),
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
            c.config.brain.trading.maximum_trade_fraction,
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
                reward_ema: 0.0,
                reward_variance_ema: 0.0,
                last_reward: 0.0,
                short_reward_ema: 0.0,
                long_reward_ema: 0.0,
                reward_trend: 0.0,
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
                maximum_trade_fraction,
            } => {
                if !temperature.is_finite() || temperature <= 0.0 {
                    return Err(MidasError::Checkpoint(
                        "checkpoint action temperature is invalid".into(),
                    ));
                }
                if !maximum_trade_fraction.is_finite()
                    || maximum_trade_fraction <= 0.0
                    || maximum_trade_fraction > 1.0
                    || (maximum_trade_fraction - c.config.brain.trading.maximum_trade_fraction)
                        .abs()
                        > 1e-12
                {
                    return Err(MidasError::Checkpoint(
                        "checkpoint incremental trade fraction is incompatible with configuration"
                            .into(),
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
                Box::new(NeuralPolicy::with_rng(
                    genome,
                    model,
                    rng,
                    temperature,
                    maximum_trade_fraction,
                ))
            }
            PolicySnapshot::Momentum {
                signal_indices,
                minimum_signal,
            } => Box::new(MomentumPolicy::new(signal_indices, minimum_signal)),
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
        crate::config::PolicyKind::Momentum => {
            let schema = crate::observation::ObservationSchema::try_from_options(
                config.observation_schema_options(),
            )?;
            let signal_indices = (0..config.market.assets.len())
                .map(|asset_index| {
                    schema
                        .entries
                        .iter()
                        .find(|entry| {
                            matches!(
                                entry.source,
                                Some(crate::observation::ObservationSource::Market {
                                    feature,
                                    asset_index: source_asset,
                                }) if feature == config.brain.momentum.signal_feature
                                    && source_asset == asset_index
                            )
                        })
                        .map(|entry| entry.index)
                        .ok_or_else(|| {
                            MidasError::Config(format!(
                                "momentum signal {:?} is missing for asset {}",
                                config.brain.momentum.signal_feature,
                                config.market.assets[asset_index]
                            ))
                        })
                })
                .collect::<Result<Vec<_>>>()?;
            PolicyBlueprint::Momentum {
                signal_indices,
                minimum_signal_bits: config.brain.momentum.minimum_signal.to_bits(),
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reward_statistics_track_recent_mean_variance_and_trend() {
        let mut state = AgentState {
            id: crate::AgentId(1),
            birth_date: NaiveDate::from_ymd_opt(2020, 1, 1).unwrap(),
            age: 0,
            portfolio: Portfolio::new(100.0, 1),
            maximum_net_worth: 100.0,
            maximum_drawdown: 0.0,
            alive: true,
            genome: None,
            genome_id: None,
            lineage_id: None,
            parent_ids: Vec::new(),
            total_policy_failures: 0,
            consecutive_policy_failures: 0,
            reward_ema: 0.0,
            reward_variance_ema: 0.0,
            last_reward: 0.0,
            short_reward_ema: 0.0,
            long_reward_ema: 0.0,
            reward_trend: 0.0,
        };
        for _ in 0..200 {
            state.update_reward_statistics(1.0);
        }
        assert_eq!(state.last_reward, 1.0);
        assert!((state.reward_ema - 1.0).abs() < 1e-4);
        assert!(state.reward_variance_ema < 1e-4);
        assert_eq!(state.reward_trend, 1.0);

        state.update_reward_statistics(-1.0);
        assert_eq!(state.last_reward, -1.0);
        assert_eq!(state.reward_trend, -1.0);
    }
}
