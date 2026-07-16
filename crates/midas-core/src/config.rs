use crate::{MidasError, Result};
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

mod backend;
mod brain;
mod checkpoint;
mod evolution;
mod execution;
mod learning;
mod market;
mod observation;
mod replacement;
mod reward;
mod simulation;
mod splits;

pub use backend::{BackendConfig, BackendKind};
pub use brain::{
    ActionDistributionConfig, ActionDistributionKind, Activation, AssetOrder, BrainConfig,
    EvolvableTraitConfig, EvolvableTraitsConfig, PolicyKind, TradingConfig,
};
pub use checkpoint::CheckpointConfig;
pub use evolution::{
    CrossoverMethod, EvolutionConfig, EvolutionCrossoverConfig, EvolutionDiversityConfig,
    EvolutionFallback, EvolutionFitnessConfig, EvolutionMutationConfig, EvolutionSelectionConfig,
    InheritanceMode, SelectionMethod,
};
pub use execution::ExecutionConfig;
pub use learning::{LearningConfig, LearningKind};
pub use market::{
    CorporateActionConfig, CorporateActionMode, DataConfig, DataFormat, FeatureConfig, FeatureKind,
    MarketConfig,
};
pub use observation::{
    ObservationConfig, ObservationFeatureKind, ObservationNormalizationConfig,
    ObservationNormalizationKind, ObservationRegistryConfig, RollingMarketStatsConfig,
    RollingMissingHistory,
};
pub use replacement::{ReplacementConfig, ReplacementKind};
pub use reward::{RewardConfig, RewardKind};
pub use simulation::{DeathHandling, PolicyFailureBehavior, PolicyFailureConfig, SimulationConfig};
pub use splits::{DateRange, SplitConfig};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub market: MarketConfig,
    pub data: DataConfig,
    pub splits: SplitConfig,
    pub simulation: SimulationConfig,
    pub execution: ExecutionConfig,
    pub features: FeatureConfig,
    #[serde(default)]
    pub observation: ObservationConfig,
    #[serde(default)]
    pub observations: ObservationRegistryConfig,
    #[serde(default)]
    pub corporate_actions: CorporateActionConfig,
    #[serde(default)]
    pub policy_failures: PolicyFailureConfig,
    pub brain: BrainConfig,
    pub learning: LearningConfig,
    pub backend: BackendConfig,
    pub replacement: ReplacementConfig,
    #[serde(default)]
    pub checkpoint: CheckpointConfig,
    #[serde(default)]
    pub reward: RewardConfig,
    #[serde(default)]
    pub evolution: EvolutionConfig,
}

impl Config {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let value: Self = toml::from_str(&fs::read_to_string(path)?)?;
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<()> {
        let bad = |message: &str| Err(MidasError::Config(message.into()));
        if self.market.assets.is_empty() {
            return bad("market.assets must not be empty");
        }
        let mut assets = self.market.assets.clone();
        assets.sort();
        assets.dedup();
        if assets.len() != self.market.assets.len() {
            return bad("market.assets contains duplicates");
        }
        if !self.market.assets.contains(&self.market.benchmark_asset) {
            return bad("benchmark_asset must be in market.assets");
        }
        if self.simulation.population_size == 0 {
            return bad("population_size must be positive");
        }
        for (name, value) in [
            ("starting_cash", self.simulation.starting_cash),
            ("bankruptcy_threshold", self.simulation.bankruptcy_threshold),
            ("daily_survival_cost", self.simulation.daily_survival_cost),
            ("minimum_trade_value", self.simulation.minimum_trade_value),
            ("commission_fixed", self.execution.commission_fixed),
            ("commission_rate", self.execution.commission_rate),
            ("spread_rate", self.execution.spread_rate),
            ("slippage_rate", self.execution.slippage_rate),
        ] {
            if !value.is_finite() || value < 0.0 {
                return bad(&format!("{name} must be finite and non-negative"));
            }
        }
        if self.simulation.starting_cash <= self.simulation.bankruptcy_threshold {
            return bad("starting_cash must exceed bankruptcy_threshold");
        }
        if self.execution.spread_rate >= 2.0 {
            return bad("spread_rate must be less than 2");
        }
        if self.execution.slippage_rate >= 1.0 {
            return bad("slippage_rate must be less than 1");
        }
        if self.execution.spread_rate / 2.0 + self.execution.slippage_rate >= 1.0 {
            return bad("spread_rate / 2 + slippage_rate must be less than 1");
        }
        if self.features.schema_version == 0
            || self.features.features.is_empty()
            || !self.features.clip.is_finite()
            || self.features.clip <= 0.0
        {
            return bad("feature schema version, feature list, and clip must be valid");
        }
        if self.brain.architecture_version == 0
            || self.brain.hidden_layers.is_empty()
            || self.brain.hidden_layers.contains(&0)
            || self.brain.memory_size == 0
        {
            return bad("brain architecture dimensions must be positive");
        }
        if !matches!(
            self.brain.action_distribution.kind,
            ActionDistributionKind::BoundedImpulseBeta
        ) || !self.brain.action_distribution.minimum_shape.is_finite()
            || !self.brain.action_distribution.maximum_shape.is_finite()
            || !self.brain.action_distribution.latent_epsilon.is_finite()
            || self.brain.action_distribution.minimum_shape <= 0.0
            || self.brain.action_distribution.maximum_shape
                <= self.brain.action_distribution.minimum_shape
            || !(0.0..0.5).contains(&self.brain.action_distribution.latent_epsilon)
            || self.brain.action_distribution.latent_epsilon == 0.0
        {
            return bad("bounded impulse Beta action distribution parameters are invalid");
        }
        if !self.brain.trading.maximum_trade_fraction.is_finite()
            || self.brain.trading.maximum_trade_fraction <= 0.0
            || self.brain.trading.maximum_trade_fraction > 1.0
        {
            return bad("brain.trading.maximum_trade_fraction must be in (0, 1]");
        }
        for (name, value) in [
            ("learning_rate", &self.brain.traits.learning_rate),
            (
                "entropy_coefficient",
                &self.brain.traits.entropy_coefficient,
            ),
            ("action_temperature", &self.brain.traits.action_temperature),
            ("temperature_decay", &self.brain.traits.temperature_decay),
            (
                "drawdown_sensitivity",
                &self.brain.traits.drawdown_sensitivity,
            ),
            (
                "turnover_sensitivity",
                &self.brain.traits.turnover_sensitivity,
            ),
            ("maximum_allocation", &self.brain.traits.maximum_allocation),
        ] {
            value.validate(name)?;
        }
        if self.brain.traits.maximum_allocation.minimum <= 0.0
            || self.brain.traits.maximum_allocation.maximum > 1.0
        {
            return bad("maximum allocation trait bounds must be within (0, 1]");
        }
        if self.brain.traits.learning_rate.minimum <= 0.0
            || self.brain.traits.action_temperature.minimum <= 0.0
            || self.brain.traits.temperature_decay.minimum < 0.0
            || self.brain.traits.temperature_decay.maximum > 1.0
        {
            return bad("learning and action-temperature trait bounds are invalid");
        }
        if self.observation.version == 0 || self.observation.features.is_empty() {
            return bad("observation schema version and features must be non-empty");
        }
        if self.observations.schema.trim().is_empty() || self.observations.providers.is_empty() {
            return bad("observation registry schema and providers must be non-empty");
        }
        if self.observations.schema != "default" {
            return bad(
                "only the built-in observation schema named default is currently supported",
            );
        }
        let mut observation_providers = self.observations.providers.clone();
        observation_providers.sort();
        observation_providers.dedup();
        if observation_providers.len() != self.observations.providers.len()
            || observation_providers
                .iter()
                .any(|provider| provider.trim().is_empty())
        {
            return bad("observation registry providers must be unique non-empty names");
        }
        for normalization in self.observations.normalizations.values() {
            normalization.validate()?;
        }
        self.observations.rolling_market_stats.validate()?;
        let observation_schema = crate::observation::ObservationSchema::try_from_config(self)?;
        let used_observation_providers = observation_schema
            .entries
            .iter()
            .map(|entry| entry.provider.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        let configured_observation_providers = self
            .observations
            .providers
            .iter()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>();
        if used_observation_providers != configured_observation_providers {
            return bad("configured observation providers must each have schema entries");
        }
        if self.corporate_actions.discontinuity_threshold <= 0.0
            || self.corporate_actions.discontinuity_threshold >= 1.0
            || !self.corporate_actions.discontinuity_threshold.is_finite()
            || self.corporate_actions.adjusted_price_tolerance < 0.0
            || !self.corporate_actions.adjusted_price_tolerance.is_finite()
        {
            return bad("corporate action thresholds are invalid");
        }
        if self.policy_failures.maximum_consecutive == 0 || self.policy_failures.maximum_total == 0
        {
            return bad("policy failure limits must be positive");
        }
        for range in [
            &self.splits.training,
            &self.splits.validation,
            &self.splits.test,
        ] {
            if range.start > range.end {
                return bad("split start must not follow split end");
            }
        }
        if self.splits.training.end >= self.splits.validation.start
            || self.splits.validation.end >= self.splits.test.start
        {
            return bad("splits must be chronological and non-overlapping");
        }
        if !self.learning.learning_rate.is_finite()
            || self.learning.learning_rate <= 0.0
            || !self.learning.gamma.is_finite()
            || !(0.0..=1.0).contains(&self.learning.gamma)
            || !self.learning.gae_lambda.is_finite()
            || !(0.0..=1.0).contains(&self.learning.gae_lambda)
            || !self.learning.ppo_clip.is_finite()
            || self.learning.ppo_clip < 0.0
            || !self.learning.critic_coefficient.is_finite()
            || self.learning.critic_coefficient < 0.0
            || !self.learning.entropy_coefficient.is_finite()
            || self.learning.entropy_coefficient < 0.0
            || !self.learning.gradient_clip.is_finite()
            || self.learning.gradient_clip <= 0.0
            || self.learning.rollout_length == 0
            || self.learning.minibatch_size == 0
            || self.learning.minibatch_size > self.learning.rollout_length
            || self.learning.sequence_length == 0
            || self.learning.sequence_length > self.learning.rollout_length
            || self.learning.optimisation_epochs == 0
        {
            return bad("learning configuration values are invalid");
        }
        for (name, value) in [
            (
                "reward.drawdown_penalty_weight",
                self.reward.drawdown_penalty_weight,
            ),
            (
                "reward.turnover_penalty_weight",
                self.reward.turnover_penalty_weight,
            ),
            (
                "reward.invalid_action_penalty",
                self.reward.invalid_action_penalty,
            ),
            ("reward.bankruptcy_penalty", self.reward.bankruptcy_penalty),
        ] {
            if !value.is_finite() || value < 0.0 {
                return bad(&format!("{name} must be finite and non-negative"));
            }
        }
        if !self.reward.clip_min.is_finite() || !self.reward.clip_max.is_finite() {
            return bad("reward clip bounds must be finite");
        }
        if self.reward.clip_min >= self.reward.clip_max {
            return bad("reward.clip_min must be less than clip_max");
        }
        let mutation = &self.evolution.mutation;
        for (name, value) in [
            ("weight_probability", mutation.weight_probability),
            ("bias_probability", mutation.bias_probability),
            ("reset_probability", mutation.reset_probability),
            ("trait_probability", mutation.trait_probability),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return bad(&format!("evolution.mutation.{name} must be within [0, 1]"));
            }
        }
        for (name, value) in [
            ("weight_sigma", mutation.weight_sigma),
            ("bias_sigma", mutation.bias_sigma),
        ] {
            if !value.is_finite() || value < 0.0 {
                return bad(&format!("evolution.mutation.{name} must be non-negative"));
            }
        }
        if self.evolution.selection.tournament_size == 0
            || !self.evolution.selection.minimum_net_worth.is_finite()
            || self.evolution.selection.minimum_net_worth < 0.0
            || !self.evolution.diversity.sharing_radius.is_finite()
            || self.evolution.diversity.sharing_radius < 0.0
            || !self.evolution.diversity.minimum_mating_distance.is_finite()
            || self.evolution.diversity.minimum_mating_distance < 0.0
        {
            return bad("evolution selection or diversity configuration is invalid");
        }
        for (name, value) in [
            (
                "evolution.fitness.return_weight",
                self.evolution.fitness.return_weight,
            ),
            (
                "evolution.fitness.survival_weight",
                self.evolution.fitness.survival_weight,
            ),
            (
                "evolution.fitness.drawdown_weight",
                self.evolution.fitness.drawdown_weight,
            ),
            (
                "evolution.fitness.turnover_weight",
                self.evolution.fitness.turnover_weight,
            ),
        ] {
            if !value.is_finite() || value < 0.0 {
                return bad(&format!("{name} must be finite and non-negative"));
            }
        }
        self.backend.compute_device()?;
        Ok(())
    }

    pub fn hash(&self) -> Result<String> {
        Ok(blake3::hash(&serde_json::to_vec(self)?)
            .to_hex()
            .to_string())
    }
}
