use crate::{MidasError, Result};
use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

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
    pub corporate_actions: CorporateActionConfig,
    pub brain: BrainConfig,
    pub learning: LearningConfig,
    pub backend: BackendConfig,
    pub replacement: ReplacementConfig,
    #[serde(default)]
    pub checkpoint: CheckpointConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MarketConfig {
    pub assets: Vec<String>,
    pub benchmark_asset: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DataConfig {
    pub path: String,
    #[serde(default)]
    pub format: DataFormat,
    #[serde(default)]
    pub corporate_actions_path: Option<String>,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataFormat {
    #[default]
    Csv,
    Parquet,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DateRange {
    pub start: NaiveDate,
    pub end: NaiveDate,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SplitConfig {
    pub training: DateRange,
    pub validation: DateRange,
    pub test: DateRange,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SimulationConfig {
    pub population_size: usize,
    pub starting_cash: f64,
    pub bankruptcy_threshold: f64,
    pub daily_survival_cost: f64,
    pub minimum_trade_value: f64,
    pub seed: u64,
    #[serde(default)]
    pub forced_liquidation: bool,
    #[serde(default)]
    pub death_handling: DeathHandling,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionConfig {
    pub commission_fixed: f64,
    pub commission_rate: f64,
    pub spread_rate: f64,
    pub slippage_rate: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeatureConfig {
    pub schema_version: u32,
    pub features: Vec<FeatureKind>,
    pub clip: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureKind {
    Return1,
    Return5,
    Return20,
    PriceToMa20,
    PriceToMa50,
    Volatility20,
    VolumeChange,
    VolumeToMa20,
    DistanceHigh20,
    Tradable,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BrainConfig {
    pub architecture_version: u32,
    pub hidden_layers: Vec<usize>,
    pub memory_size: usize,
    #[serde(default = "default_activation")]
    pub activation: Activation,
    #[serde(default = "default_policy")]
    pub policy: PolicyKind,
    #[serde(default)]
    pub traits: EvolvableTraitsConfig,
}
fn default_activation() -> Activation {
    Activation::Gelu
}
fn default_policy() -> PolicyKind {
    PolicyKind::BurnActorCritic
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Activation {
    Relu,
    Gelu,
    Silu,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyKind {
    Cash,
    Random,
    EqualWeight,
    BuyAndHold,
    BurnActorCritic,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeathHandling {
    #[default]
    Replace,
    LeaveVacant,
    EndAgentOnly,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvolvableTraitConfig {
    pub initial: f32,
    pub minimum: f32,
    pub maximum: f32,
    pub mutation_scale: f32,
    pub evolvable: bool,
}

impl EvolvableTraitConfig {
    fn validate(&self, name: &str) -> Result<()> {
        if !self.initial.is_finite()
            || !self.minimum.is_finite()
            || !self.maximum.is_finite()
            || !self.mutation_scale.is_finite()
            || self.minimum > self.initial
            || self.initial > self.maximum
            || self.mutation_scale < 0.0
        {
            return Err(MidasError::Config(format!(
                "invalid evolvable trait {name}"
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvolvableTraitsConfig {
    pub learning_rate: EvolvableTraitConfig,
    pub entropy_coefficient: EvolvableTraitConfig,
    pub action_temperature: EvolvableTraitConfig,
    pub temperature_decay: EvolvableTraitConfig,
    pub drawdown_sensitivity: EvolvableTraitConfig,
    pub turnover_sensitivity: EvolvableTraitConfig,
    pub maximum_allocation: EvolvableTraitConfig,
}

impl Default for EvolvableTraitsConfig {
    fn default() -> Self {
        let fixed = |initial, minimum, maximum| EvolvableTraitConfig {
            initial,
            minimum,
            maximum,
            mutation_scale: 0.0,
            evolvable: false,
        };
        Self {
            learning_rate: fixed(0.0003, 0.00001, 0.01),
            entropy_coefficient: fixed(0.01, 0.0, 0.1),
            action_temperature: fixed(1.0, 0.1, 3.0),
            temperature_decay: fixed(0.999, 0.9, 1.0),
            drawdown_sensitivity: fixed(0.1, 0.0, 2.0),
            turnover_sensitivity: fixed(0.01, 0.0, 1.0),
            maximum_allocation: fixed(0.5, 0.01, 1.0),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationFeatureKind {
    Market,
    CashAllocation,
    AssetAllocation,
    UnrealizedReturn,
    NetWorthToStarting,
    CurrentDrawdown,
    PortfolioReturn1,
    TurnoverToNetWorth,
    NormalizedAge,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationConfig {
    pub version: u32,
    pub features: Vec<ObservationFeatureKind>,
}

impl Default for ObservationConfig {
    fn default() -> Self {
        Self {
            version: 1,
            features: vec![
                ObservationFeatureKind::Market,
                ObservationFeatureKind::CashAllocation,
                ObservationFeatureKind::AssetAllocation,
                ObservationFeatureKind::UnrealizedReturn,
                ObservationFeatureKind::NetWorthToStarting,
                ObservationFeatureKind::CurrentDrawdown,
                ObservationFeatureKind::PortfolioReturn1,
                ObservationFeatureKind::TurnoverToNetWorth,
                ObservationFeatureKind::NormalizedAge,
            ],
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorporateActionMode {
    #[default]
    Reject,
    ExplicitEvents,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorporateActionConfig {
    pub mode: CorporateActionMode,
    pub discontinuity_threshold: f64,
    pub adjusted_price_tolerance: f64,
}

impl Default for CorporateActionConfig {
    fn default() -> Self {
        Self {
            mode: CorporateActionMode::Reject,
            discontinuity_threshold: 0.35,
            adjusted_price_tolerance: 1e-6,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LearningConfig {
    pub implementation: LearningKind,
    pub learning_rate: f32,
    pub gamma: f32,
    pub gae_lambda: f32,
    pub ppo_clip: f32,
    pub critic_coefficient: f32,
    pub entropy_coefficient: f32,
    pub gradient_clip: f32,
    pub rollout_length: usize,
    pub optimisation_epochs: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LearningKind {
    NoLearning,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackendConfig {
    pub kind: BackendKind,
    pub device: String,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    Cpu,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplacementConfig {
    pub strategy: ReplacementKind,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplacementKind {
    NonEvolutionaryRandomGenome,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointConfig {
    pub path: String,
    pub every_steps: usize,
}
impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            path: "checkpoints/latest.bin".into(),
            every_steps: 0,
        }
    }
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
        if self.observation.version == 0 || self.observation.features.is_empty() {
            return bad("observation schema version and features must be non-empty");
        }
        if self.corporate_actions.discontinuity_threshold <= 0.0
            || self.corporate_actions.discontinuity_threshold >= 1.0
            || !self.corporate_actions.discontinuity_threshold.is_finite()
            || self.corporate_actions.adjusted_price_tolerance < 0.0
            || !self.corporate_actions.adjusted_price_tolerance.is_finite()
        {
            return bad("corporate action thresholds are invalid");
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
        if self.learning.implementation != LearningKind::NoLearning {
            return bad("Prompt 1 supports only explicit no_learning");
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
            || self.learning.optimisation_epochs == 0
        {
            return bad("learning configuration values are invalid");
        }
        if self.backend.device.trim().is_empty() {
            return bad("backend.device must not be empty");
        }
        Ok(())
    }

    pub fn hash(&self) -> Result<String> {
        Ok(blake3::hash(&serde_json::to_vec(self)?)
            .to_hex()
            .to_string())
    }
}
