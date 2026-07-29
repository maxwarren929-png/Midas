use crate::{MidasError, Result};
use serde::{Deserialize, Serialize};

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

impl EvolutionConfig {
    pub(super) fn validate(&self) -> Result<()> {
        let bad = |message: &str| Err(MidasError::Config(message.into()));
        let mutation = &self.mutation;
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
        if self.selection.tournament_size == 0
            || !self.selection.minimum_net_worth.is_finite()
            || self.selection.minimum_net_worth < 0.0
            || !self.diversity.sharing_radius.is_finite()
            || self.diversity.sharing_radius < 0.0
            || !self.diversity.minimum_mating_distance.is_finite()
            || self.diversity.minimum_mating_distance < 0.0
        {
            return bad("evolution selection or diversity configuration is invalid");
        }
        for (name, value) in [
            (
                "evolution.fitness.return_weight",
                self.fitness.return_weight,
            ),
            (
                "evolution.fitness.survival_weight",
                self.fitness.survival_weight,
            ),
            (
                "evolution.fitness.drawdown_weight",
                self.fitness.drawdown_weight,
            ),
            (
                "evolution.fitness.turnover_weight",
                self.fitness.turnover_weight,
            ),
        ] {
            if !value.is_finite() || value < 0.0 {
                return bad(&format!("{name} must be finite and non-negative"));
            }
        }
        Ok(())
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
