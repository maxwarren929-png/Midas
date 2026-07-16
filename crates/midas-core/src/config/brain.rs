use crate::{MidasError, Result};
use serde::{Deserialize, Serialize};

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
    pub(super) fn validate(&self, name: &str) -> Result<()> {
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
