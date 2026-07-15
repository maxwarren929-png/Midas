use crate::{MidasError, Result};
use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs, path::Path};

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
    #[serde(default)]
    pub action_distribution: ActionDistributionConfig,
    #[serde(default)]
    pub trading: TradingConfig,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionDistributionKind {
    BoundedImpulseBeta,
    #[serde(rename = "bounded_stick_breaking")]
    BoundedStickBreaking,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssetOrder {
    Stable,
    EpisodeRotated,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActionDistributionConfig {
    #[serde(default = "default_action_distribution_kind")]
    pub kind: ActionDistributionKind,
    #[serde(default = "default_minimum_shape")]
    pub minimum_shape: f32,
    #[serde(default = "default_maximum_shape")]
    pub maximum_shape: f32,
    #[serde(default = "default_latent_epsilon")]
    pub latent_epsilon: f32,
    #[serde(default = "default_asset_order")]
    pub asset_order: AssetOrder,
}

fn default_action_distribution_kind() -> ActionDistributionKind {
    ActionDistributionKind::BoundedImpulseBeta
}
fn default_minimum_shape() -> f32 {
    0.1
}
fn default_maximum_shape() -> f32 {
    100.0
}
fn default_latent_epsilon() -> f32 {
    1e-6
}
fn default_asset_order() -> AssetOrder {
    AssetOrder::Stable
}

impl Default for ActionDistributionConfig {
    fn default() -> Self {
        Self {
            kind: default_action_distribution_kind(),
            minimum_shape: default_minimum_shape(),
            maximum_shape: default_maximum_shape(),
            latent_epsilon: default_latent_epsilon(),
            asset_order: default_asset_order(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TradingConfig {
    #[serde(default = "default_maximum_trade_fraction")]
    pub maximum_trade_fraction: f64,
}

fn default_maximum_trade_fraction() -> f64 {
    0.05
}

impl Default for TradingConfig {
    fn default() -> Self {
        Self {
            maximum_trade_fraction: default_maximum_trade_fraction(),
        }
    }
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

/// Registry selection and per-name normalisation overrides.  The singular
/// `observation` table remains the backwards-compatible compact schema source;
/// this table controls which sensor providers are registered.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationRegistryConfig {
    #[serde(default = "default_observation_schema_name")]
    pub schema: String,
    #[serde(default = "default_observation_providers")]
    pub providers: Vec<String>,
    #[serde(default)]
    pub normalizations: BTreeMap<String, ObservationNormalizationConfig>,
    /// Optional past-only market sensors.  This field is omitted from the
    /// configuration hash while disabled so unchanged Prompt 5 configurations
    /// remain compatible with their v12 checkpoints.
    #[serde(default, skip_serializing_if = "RollingMarketStatsConfig::is_default")]
    pub rolling_market_stats: RollingMarketStatsConfig,
}

fn default_observation_schema_name() -> String {
    "default".into()
}

fn default_observation_providers() -> Vec<String> {
    vec!["market".into(), "portfolio".into(), "agent_state".into()]
}

impl Default for ObservationRegistryConfig {
    fn default() -> Self {
        Self {
            schema: default_observation_schema_name(),
            providers: default_observation_providers(),
            normalizations: BTreeMap::new(),
            rolling_market_stats: RollingMarketStatsConfig::default(),
        }
    }
}

/// How a rolling sensor represents an unavailable lookback. Prompt 6 supports
/// zero fill only: it is deterministic, finite, and does not consume future
/// bars. Future schemas may add validity-mask observations explicitly.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RollingMissingHistory {
    #[default]
    ZeroFill,
}

/// Opt-in configuration for the `rolling_market_stats` observation provider.
/// Providers emit raw values; all scaling remains schema-owned.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RollingMarketStatsConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_rolling_return_windows")]
    pub return_windows: Vec<usize>,
    #[serde(default = "default_rolling_volatility_windows")]
    pub volatility_windows: Vec<usize>,
    #[serde(default = "default_rolling_sma_windows")]
    pub sma_windows: Vec<usize>,
    #[serde(default = "default_rolling_relative_strength_windows")]
    pub relative_strength_windows: Vec<usize>,
    #[serde(default = "default_rolling_market_summary")]
    pub market_summary: bool,
    #[serde(default = "default_rolling_market_summary_window")]
    pub market_summary_window: usize,
    #[serde(default)]
    pub minimum_history: RollingMissingHistory,
}

fn default_rolling_return_windows() -> Vec<usize> {
    vec![5, 20, 60, 120]
}

fn default_rolling_volatility_windows() -> Vec<usize> {
    vec![20, 60]
}

fn default_rolling_sma_windows() -> Vec<usize> {
    vec![20, 60]
}

fn default_rolling_relative_strength_windows() -> Vec<usize> {
    vec![20, 60]
}

fn default_rolling_market_summary() -> bool {
    true
}

fn default_rolling_market_summary_window() -> usize {
    20
}

impl Default for RollingMarketStatsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            return_windows: default_rolling_return_windows(),
            volatility_windows: default_rolling_volatility_windows(),
            sma_windows: default_rolling_sma_windows(),
            relative_strength_windows: default_rolling_relative_strength_windows(),
            market_summary: default_rolling_market_summary(),
            market_summary_window: default_rolling_market_summary_window(),
            minimum_history: RollingMissingHistory::ZeroFill,
        }
    }
}

impl RollingMarketStatsConfig {
    pub fn is_default(config: &Self) -> bool {
        config == &Self::default()
    }

    pub fn validate(&self) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        let groups = [
            ("return_windows", &self.return_windows),
            ("volatility_windows", &self.volatility_windows),
            ("sma_windows", &self.sma_windows),
            ("relative_strength_windows", &self.relative_strength_windows),
        ];
        let mut count = usize::from(self.market_summary);
        for (name, windows) in groups {
            if windows.contains(&0) {
                return Err(MidasError::Config(format!(
                    "observations.rolling_market_stats.{name} must contain positive windows"
                )));
            }
            let unique = windows
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>();
            if unique.len() != windows.len() {
                return Err(MidasError::Config(format!(
                    "observations.rolling_market_stats.{name} must not contain duplicate windows"
                )));
            }
            count += windows.len();
        }
        if self.market_summary_window == 0 {
            return Err(MidasError::Config(
                "observations.rolling_market_stats.market_summary_window must be positive".into(),
            ));
        }
        if count == 0 {
            return Err(MidasError::Config(
                "rolling_market_stats must enable at least one metric".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationNormalizationKind {
    None,
    RatioClipped,
    MinMax,
    ZScore,
    Log,
    SignedLog,
    Percentage,
    Custom,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationNormalizationConfig {
    pub kind: ObservationNormalizationKind,
    #[serde(default)]
    pub minimum: Option<f64>,
    #[serde(default)]
    pub maximum: Option<f64>,
    #[serde(default)]
    pub mean: Option<f64>,
    #[serde(default)]
    pub standard_deviation: Option<f64>,
    #[serde(default)]
    pub scale: Option<f64>,
    #[serde(default)]
    pub offset: Option<f64>,
}

impl ObservationNormalizationConfig {
    pub fn validate(&self) -> Result<()> {
        let all_finite = [
            self.minimum,
            self.maximum,
            self.mean,
            self.standard_deviation,
            self.scale,
            self.offset,
        ]
        .into_iter()
        .flatten()
        .all(f64::is_finite);
        if !all_finite {
            return Err(MidasError::Config(
                "observation normalization values must be finite".into(),
            ));
        }
        let valid = match self.kind {
            ObservationNormalizationKind::None
            | ObservationNormalizationKind::Log
            | ObservationNormalizationKind::SignedLog => true,
            ObservationNormalizationKind::RatioClipped | ObservationNormalizationKind::MinMax => {
                self.minimum
                    .zip(self.maximum)
                    .is_some_and(|(minimum, maximum)| minimum < maximum)
            }
            ObservationNormalizationKind::ZScore => self
                .mean
                .zip(self.standard_deviation)
                .is_some_and(|(_, standard_deviation)| standard_deviation > 0.0),
            ObservationNormalizationKind::Percentage => {
                self.scale.is_some_and(|scale| scale != 0.0)
            }
            ObservationNormalizationKind::Custom => self.scale.is_some() && self.offset.is_some(),
        };
        if valid {
            Ok(())
        } else {
            Err(MidasError::Config(
                "observation normalization configuration is incomplete or invalid".into(),
            ))
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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyFailureBehavior {
    Continue,
    #[default]
    KillAgent,
    StopExperiment,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyFailureConfig {
    pub maximum_consecutive: u32,
    pub maximum_total: u32,
    pub behavior: PolicyFailureBehavior,
}
impl Default for PolicyFailureConfig {
    fn default() -> Self {
        Self {
            maximum_consecutive: 3,
            maximum_total: 20,
            behavior: PolicyFailureBehavior::KillAgent,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
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
    #[serde(default = "default_minibatch_size")]
    pub minibatch_size: usize,
    #[serde(default = "default_sequence_length")]
    pub sequence_length: usize,
    #[serde(default = "default_true")]
    pub normalize_advantages: bool,
    pub optimisation_epochs: usize,
}

fn default_minibatch_size() -> usize {
    32
}
fn default_sequence_length() -> usize {
    16
}
fn default_true() -> bool {
    true
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LearningKind {
    NoLearning,
    Ppo,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RewardConfig {
    #[serde(default)]
    pub kind: RewardKind,
    #[serde(default)]
    pub drawdown_penalty_weight: f32,
    #[serde(default)]
    pub turnover_penalty_weight: f32,
    #[serde(default)]
    pub invalid_action_penalty: f32,
    #[serde(default = "default_bankruptcy_penalty")]
    pub bankruptcy_penalty: f32,
    #[serde(default = "default_reward_clip_min")]
    pub clip_min: f32,
    #[serde(default = "default_reward_clip_max")]
    pub clip_max: f32,
}
fn default_bankruptcy_penalty() -> f32 {
    1.0
}
fn default_reward_clip_min() -> f32 {
    -5.0
}
fn default_reward_clip_max() -> f32 {
    5.0
}
impl Default for RewardConfig {
    fn default() -> Self {
        Self {
            kind: RewardKind::LogReturn,
            drawdown_penalty_weight: 0.0,
            turnover_penalty_weight: 0.0,
            invalid_action_penalty: 0.05,
            bankruptcy_penalty: 1.0,
            clip_min: -5.0,
            clip_max: 5.0,
        }
    }
}
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RewardKind {
    #[default]
    LogReturn,
    RiskAdjustedReturn,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvolutionConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub fallback: EvolutionFallback,
    #[serde(default)]
    pub inheritance_mode: InheritanceMode,
    #[serde(default)]
    pub selection: EvolutionSelectionConfig,
    #[serde(default)]
    pub crossover: EvolutionCrossoverConfig,
    #[serde(default)]
    pub mutation: EvolutionMutationConfig,
    #[serde(default)]
    pub diversity: EvolutionDiversityConfig,
    #[serde(default)]
    pub fitness: EvolutionFitnessConfig,
}
impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            fallback: EvolutionFallback::RandomGenome,
            inheritance_mode: InheritanceMode::GenomeOnly,
            selection: EvolutionSelectionConfig::default(),
            crossover: EvolutionCrossoverConfig::default(),
            mutation: EvolutionMutationConfig::default(),
            diversity: EvolutionDiversityConfig::default(),
            fitness: EvolutionFitnessConfig::default(),
        }
    }
}
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvolutionFallback {
    #[default]
    RandomGenome,
    CloneBestEligible,
    AbortExperiment,
}
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InheritanceMode {
    #[default]
    GenomeOnly,
}
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionMethod {
    #[default]
    Tournament,
    RankWeighted,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvolutionSelectionConfig {
    #[serde(default)]
    pub method: SelectionMethod,
    #[serde(default = "default_tournament_size")]
    pub tournament_size: usize,
    #[serde(default)]
    pub minimum_age: u64,
    #[serde(default)]
    pub allow_self_parenting: bool,
    #[serde(default)]
    pub minimum_net_worth: f64,
}
fn default_tournament_size() -> usize {
    4
}
impl Default for EvolutionSelectionConfig {
    fn default() -> Self {
        Self {
            method: SelectionMethod::Tournament,
            tournament_size: 4,
            minimum_age: 0,
            allow_self_parenting: false,
            minimum_net_worth: 0.0,
        }
    }
}
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CrossoverMethod {
    #[default]
    UniformElementWise,
    LayerWiseBlend,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvolutionCrossoverConfig {
    #[serde(default)]
    pub method: CrossoverMethod,
}
impl Default for EvolutionCrossoverConfig {
    fn default() -> Self {
        Self {
            method: CrossoverMethod::UniformElementWise,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvolutionMutationConfig {
    #[serde(default = "default_weight_probability")]
    pub weight_probability: f32,
    #[serde(default = "default_weight_sigma")]
    pub weight_sigma: f32,
    #[serde(default = "default_bias_probability")]
    pub bias_probability: f32,
    #[serde(default = "default_bias_sigma")]
    pub bias_sigma: f32,
    #[serde(default)]
    pub reset_probability: f32,
    #[serde(default = "default_trait_probability")]
    pub trait_probability: f32,
}
fn default_weight_probability() -> f32 {
    0.02
}
fn default_weight_sigma() -> f32 {
    0.05
}
fn default_bias_probability() -> f32 {
    0.02
}
fn default_bias_sigma() -> f32 {
    0.02
}
fn default_trait_probability() -> f32 {
    0.05
}
impl Default for EvolutionMutationConfig {
    fn default() -> Self {
        Self {
            weight_probability: 0.02,
            weight_sigma: 0.05,
            bias_probability: 0.02,
            bias_sigma: 0.02,
            reset_probability: 0.001,
            trait_probability: 0.05,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvolutionDiversityConfig {
    #[serde(default)]
    pub fitness_sharing: bool,
    #[serde(default = "default_sharing_radius")]
    pub sharing_radius: f32,
    #[serde(default)]
    pub minimum_mating_distance: f32,
}
fn default_sharing_radius() -> f32 {
    0.15
}
impl Default for EvolutionDiversityConfig {
    fn default() -> Self {
        Self {
            fitness_sharing: false,
            sharing_radius: 0.15,
            minimum_mating_distance: 0.0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvolutionFitnessConfig {
    #[serde(default = "default_fitness_return_weight")]
    pub return_weight: f64,
    #[serde(default = "default_fitness_survival_weight")]
    pub survival_weight: f64,
    #[serde(default = "default_fitness_drawdown_weight")]
    pub drawdown_weight: f64,
    #[serde(default = "default_fitness_turnover_weight")]
    pub turnover_weight: f64,
}
fn default_fitness_return_weight() -> f64 {
    1.0
}
fn default_fitness_survival_weight() -> f64 {
    0.01
}
fn default_fitness_drawdown_weight() -> f64 {
    0.5
}
fn default_fitness_turnover_weight() -> f64 {
    0.001
}
impl Default for EvolutionFitnessConfig {
    fn default() -> Self {
        Self {
            return_weight: 1.0,
            survival_weight: 0.01,
            drawdown_weight: 0.5,
            turnover_weight: 0.001,
        }
    }
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
    #[serde(alias = "cpu")]
    Cpu,
}

impl BackendConfig {
    pub fn compute_device(&self) -> Result<crate::brain::ComputeDevice> {
        if self.device.trim().is_empty() {
            return Err(MidasError::Config(
                "backend.device must not be empty".into(),
            ));
        }
        let device = match self.kind {
            BackendKind::Cpu => crate::brain::ComputeDevice::CpuNdArray {
                device: self.device.clone(),
            },
        };
        device.validate()?;
        Ok(device)
    }
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
