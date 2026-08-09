use crate::Result;
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

mod backend;
mod backend_compat;
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
mod validation;

pub use crate::seed::SeedDerivation;
pub use backend::{BackendConfig, BackendKind};
pub use brain::{
    ActionDistributionConfig, ActionDistributionKind, Activation, AssetOrder, BrainConfig,
    EvolvableTraitConfig, EvolvableTraitsConfig, MomentumConfig, PolicyKind, TradingConfig,
};
pub use checkpoint::CheckpointConfig;
pub use evolution::{
    CrossoverMethod, EvolutionConfig, EvolutionCrossoverConfig, EvolutionDiversityConfig,
    EvolutionFallback, EvolutionFitnessConfig, EvolutionMutationConfig, EvolutionSelectionConfig,
    InheritanceMode, SelectionMethod,
};
pub use execution::ExecutionConfig;
pub use learning::{LearningConfig, LearningKind, OptimizerConfig, OptimizerKind};
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
    #[serde(default, skip_serializing_if = "OptimizerConfig::is_default")]
    pub optimizer: OptimizerConfig,
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

    pub fn hash(&self) -> Result<String> {
        Ok(blake3::hash(&serde_json::to_vec(self)?)
            .to_hex()
            .to_string())
    }
}
