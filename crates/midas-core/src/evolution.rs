use crate::{
    AgentId, MidasError, Result,
    brain::{BrainGenome, DenseGenome, LearningTraits},
    config::{
        CrossoverMethod, EvolutionCrossoverConfig, EvolutionMutationConfig,
        EvolutionSelectionConfig, EvolvableTraitsConfig,
    },
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct GenomeId(pub u64);
impl fmt::Display for GenomeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct LineageId(pub u64);
impl fmt::Display for LineageId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GenomeRecord {
    pub genome_id: GenomeId,
    pub parent_ids: Vec<GenomeId>,
    pub lineage_id: LineageId,
    pub birth_step: usize,
    pub generation_depth: u32,
    pub genome_hash: String,
    pub creation_reason: String,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct FitnessScore(pub f64);
impl FitnessScore {
    pub fn finite(value: f64) -> Result<Self> {
        if value.is_finite() {
            Ok(Self(value))
        } else {
            Err(MidasError::Policy("fitness is non-finite".into()))
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CompletedLifeSummary {
    pub agent_id: AgentId,
    pub genome_id: Option<GenomeId>,
    pub birth_step: usize,
    pub death_step: usize,
    pub age: u64,
    pub starting_net_worth: f64,
    pub final_net_worth: f64,
    pub maximum_drawdown: f64,
    pub turnover: f64,
    pub policy_failures: u32,
    pub fitness: Option<FitnessScore>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EvolutionMetrics {
    pub birth_count: u64,
    pub death_count: u64,
    pub reproduction_count: u64,
    pub random_fallback_births: u64,
    pub crossover_count: u64,
    pub mutation_count: u64,
    pub rejected_parent_pairs: u64,
    pub fitness_values: Vec<f64>,
    pub mean_fitness: f64,
    pub maximum_fitness: f64,
    pub median_fitness: f64,
    pub fitness_variance: f64,
    pub mean_lifespan: f64,
    pub mean_final_net_worth: f64,
    pub mean_maximum_drawdown: f64,
    pub mean_genetic_distance: f64,
    pub minimum_genetic_distance: f64,
    pub active_lineages: u64,
    pub mean_generation_depth: f64,
    pub maximum_generation_depth: u32,
}
impl Default for EvolutionMetrics {
    fn default() -> Self {
        Self {
            birth_count: 0,
            death_count: 0,
            reproduction_count: 0,
            random_fallback_births: 0,
            crossover_count: 0,
            mutation_count: 0,
            rejected_parent_pairs: 0,
            fitness_values: Vec::new(),
            mean_fitness: 0.0,
            maximum_fitness: 0.0,
            median_fitness: 0.0,
            fitness_variance: 0.0,
            mean_lifespan: 0.0,
            mean_final_net_worth: 0.0,
            mean_maximum_drawdown: 0.0,
            mean_genetic_distance: 0.0,
            minimum_genetic_distance: 0.0,
            active_lineages: 0,
            mean_generation_depth: 0.0,
            maximum_generation_depth: 0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EvolutionState {
    pub next_genome_id: u64,
    pub next_lineage_id: u64,
    pub genome_registry: BTreeMap<GenomeId, GenomeRecord>,
    pub completed_lives: Vec<CompletedLifeSummary>,
    pub rng: EvolutionRng,
    pub metrics: EvolutionMetrics,
}
impl Default for EvolutionState {
    fn default() -> Self {
        Self {
            next_genome_id: 0,
            next_lineage_id: 0,
            genome_registry: BTreeMap::new(),
            completed_lives: Vec::new(),
            rng: EvolutionRng::seed(0),
            metrics: EvolutionMetrics::default(),
        }
    }
}
impl EvolutionState {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: EvolutionRng::seed(seed),
            ..Self::default()
        }
    }
    pub fn allocate_genome(
        &mut self,
        genome: &BrainGenome,
        parent_ids: Vec<GenomeId>,
        birth_step: usize,
        lineage_id: Option<LineageId>,
    ) -> Result<(GenomeId, LineageId)> {
        self.allocate_genome_with_reason(genome, parent_ids, birth_step, lineage_id, "unspecified")
    }
    pub fn allocate_genome_with_reason(
        &mut self,
        genome: &BrainGenome,
        parent_ids: Vec<GenomeId>,
        birth_step: usize,
        lineage_id: Option<LineageId>,
        creation_reason: impl Into<String>,
    ) -> Result<(GenomeId, LineageId)> {
        let id = GenomeId(self.next_genome_id);
        self.next_genome_id = self
            .next_genome_id
            .checked_add(1)
            .ok_or_else(|| MidasError::Policy("genome id exhausted".into()))?;
        let lineage = lineage_id.unwrap_or_else(|| {
            let next = LineageId(self.next_lineage_id);
            self.next_lineage_id = self.next_lineage_id.saturating_add(1);
            next
        });
        let generation_depth = parent_ids
            .iter()
            .filter_map(|parent| self.genome_registry.get(parent))
            .map(|record| record.generation_depth)
            .max()
            .unwrap_or(0)
            .saturating_add(u32::from(!parent_ids.is_empty()));
        self.genome_registry.insert(
            id,
            GenomeRecord {
                genome_id: id,
                parent_ids,
                lineage_id: lineage,
                birth_step,
                generation_depth,
                genome_hash: genome_hash(genome)?,
                creation_reason: creation_reason.into(),
            },
        );
        self.metrics.birth_count = self.metrics.birth_count.saturating_add(1);
        self.metrics.active_lineages = self
            .genome_registry
            .values()
            .map(|r| r.lineage_id)
            .collect::<std::collections::BTreeSet<_>>()
            .len() as u64;
        self.metrics.maximum_generation_depth =
            self.metrics.maximum_generation_depth.max(generation_depth);
        self.metrics.mean_generation_depth = self
            .genome_registry
            .values()
            .map(|record| record.generation_depth as f64)
            .sum::<f64>()
            / self.genome_registry.len() as f64;
        Ok((id, lineage))
    }
    pub fn record_life(&mut self, life: CompletedLifeSummary) -> Result<FitnessScore> {
        self.record_life_with(life, &DefaultFitnessFunction::default())
    }
    pub fn record_life_with(
        &mut self,
        mut life: CompletedLifeSummary,
        fitness_function: &dyn FitnessFunction,
    ) -> Result<FitnessScore> {
        let fitness = fitness_function.evaluate(&life)?;
        life.fitness = Some(fitness);
        self.metrics.death_count = self.metrics.death_count.saturating_add(1);
        self.metrics.fitness_values.push(fitness.0);
        self.completed_lives.push(life);
        let count = self.completed_lives.len() as f64;
        self.metrics.mean_fitness = self.metrics.fitness_values.iter().sum::<f64>() / count;
        self.metrics.maximum_fitness = self
            .metrics
            .fitness_values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let mut sorted = self.metrics.fitness_values.clone();
        sorted.sort_by(f64::total_cmp);
        self.metrics.median_fitness = sorted[sorted.len() / 2];
        self.metrics.fitness_variance = self
            .metrics
            .fitness_values
            .iter()
            .map(|value| (value - self.metrics.mean_fitness).powi(2))
            .sum::<f64>()
            / count;
        self.metrics.mean_lifespan = self
            .completed_lives
            .iter()
            .map(|life| life.age as f64)
            .sum::<f64>()
            / count;
        self.metrics.mean_final_net_worth = self
            .completed_lives
            .iter()
            .map(|life| life.final_net_worth)
            .sum::<f64>()
            / count;
        self.metrics.mean_maximum_drawdown = self
            .completed_lives
            .iter()
            .map(|life| life.maximum_drawdown)
            .sum::<f64>()
            / count;
        Ok(fitness)
    }

    pub fn update_diversity(&mut self, genomes: &[&BrainGenome]) -> Result<()> {
        let mut distances = Vec::new();
        for (index, left) in genomes.iter().enumerate() {
            for right in genomes.iter().skip(index + 1) {
                distances.push(genome_distance(left, right, 1.0, 1.0)?);
            }
        }
        if distances.is_empty() {
            self.metrics.mean_genetic_distance = 0.0;
            self.metrics.minimum_genetic_distance = 0.0;
        } else {
            self.metrics.mean_genetic_distance =
                distances.iter().sum::<f64>() / distances.len() as f64;
            self.metrics.minimum_genetic_distance =
                distances.iter().copied().fold(f64::INFINITY, f64::min);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EvolutionRng(pub ChaCha8Rng);
impl EvolutionRng {
    pub fn seed(seed: u64) -> Self {
        Self(ChaCha8Rng::seed_from_u64(seed))
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0.r#gen()
    }
    pub fn choose_index(&mut self, length: usize) -> Result<usize> {
        if length == 0 {
            return Err(MidasError::Policy(
                "cannot choose from empty population".into(),
            ));
        }
        Ok(self.0.gen_range(0..length))
    }
}

pub fn genome_hash(genome: &BrainGenome) -> Result<String> {
    let bytes = bincode::serialize(genome)?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ParentCandidate {
    pub agent_id: AgentId,
    pub genome_id: Option<GenomeId>,
    pub genome: BrainGenome,
    pub fitness: FitnessScore,
    pub age: u64,
    pub net_worth: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct SelectionPopulation {
    pub candidates: Vec<ParentCandidate>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SelectedParents {
    pub first: ParentCandidate,
    pub second: ParentCandidate,
}
pub trait ParentSelector: Send {
    fn select_parents(
        &mut self,
        population: &SelectionPopulation,
        rng: &mut EvolutionRng,
    ) -> Result<SelectedParents>;
    fn name(&self) -> &'static str;
    fn fork(&self) -> Box<dyn ParentSelector>;
}
#[derive(Clone, Debug)]
pub struct TournamentSelection {
    pub config: EvolutionSelectionConfig,
}
impl ParentSelector for TournamentSelection {
    fn select_parents(
        &mut self,
        population: &SelectionPopulation,
        rng: &mut EvolutionRng,
    ) -> Result<SelectedParents> {
        eligible_pair(
            population,
            &self.config,
            |pool, rng| {
                let size = self.config.tournament_size.min(pool.len()).max(1);
                let mut best = rng.choose_index(pool.len())?;
                for _ in 1..size {
                    let index = rng.choose_index(pool.len())?;
                    if pool[index].fitness.0 > pool[best].fitness.0 {
                        best = index;
                    }
                }
                Ok(best)
            },
            rng,
        )
    }
    fn name(&self) -> &'static str {
        "tournament"
    }
    fn fork(&self) -> Box<dyn ParentSelector> {
        Box::new(self.clone())
    }
}
#[derive(Clone, Debug)]
pub struct RankWeightedSelection {
    pub config: EvolutionSelectionConfig,
}
impl ParentSelector for RankWeightedSelection {
    fn select_parents(
        &mut self,
        population: &SelectionPopulation,
        rng: &mut EvolutionRng,
    ) -> Result<SelectedParents> {
        eligible_pair(
            population,
            &self.config,
            |pool, rng| {
                let mut order = (0..pool.len()).collect::<Vec<_>>();
                order.sort_by(|a, b| pool[*b].fitness.0.total_cmp(&pool[*a].fitness.0));
                let total = (pool.len() * (pool.len() + 1) / 2) as f64;
                let mut draw = rng.0.gen_range(0.0..total);
                for (rank, index) in order.iter().enumerate() {
                    let weight = (pool.len() - rank) as f64;
                    if draw < weight {
                        return Ok(*index);
                    }
                    draw -= weight;
                }
                Ok(*order.last().unwrap_or(&0))
            },
            rng,
        )
    }
    fn name(&self) -> &'static str {
        "rank_weighted"
    }
    fn fork(&self) -> Box<dyn ParentSelector> {
        Box::new(self.clone())
    }
}
fn eligible_pair<F>(
    population: &SelectionPopulation,
    config: &EvolutionSelectionConfig,
    pick: F,
    rng: &mut EvolutionRng,
) -> Result<SelectedParents>
where
    F: Fn(&[ParentCandidate], &mut EvolutionRng) -> Result<usize>,
{
    let pool = population
        .candidates
        .iter()
        .filter(|candidate| {
            candidate.age >= config.minimum_age
                && candidate.net_worth >= config.minimum_net_worth
                && candidate.fitness.0.is_finite()
        })
        .cloned()
        .collect::<Vec<_>>();
    if pool.len() < 2 && !config.allow_self_parenting {
        return Err(MidasError::Policy("fewer than two eligible parents".into()));
    }
    if pool.is_empty() {
        return Err(MidasError::Policy("no eligible parents".into()));
    }
    let first_index = pick(&pool, rng)?;
    let second_index = if config.allow_self_parenting {
        pick(&pool, rng)?
    } else {
        let reduced = pool
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != first_index)
            .map(|(_, c)| c.clone())
            .collect::<Vec<_>>();
        let index = pick(&reduced, rng)?;
        // The reduced vector owns the same candidate values; use it directly.
        return Ok(SelectedParents {
            first: pool[first_index].clone(),
            second: reduced[index].clone(),
        });
    };
    Ok(SelectedParents {
        first: pool[first_index].clone(),
        second: pool[second_index].clone(),
    })
}

pub fn crossover(
    a: &BrainGenome,
    b: &BrainGenome,
    config: &EvolutionCrossoverConfig,
    rng: &mut EvolutionRng,
) -> Result<BrainGenome> {
    if a.architecture != b.architecture {
        return Err(MidasError::Policy(
            "cannot crossover incompatible architectures".into(),
        ));
    }
    let mut child = a.clone();
    match config.method {
        CrossoverMethod::UniformElementWise => {
            for (left, right) in all_layers_mut(&mut child).into_iter().zip(all_layers(b)) {
                uniform_layer(left, right, rng);
            }
            child.traits = crossover_traits(&a.traits, &b.traits, rng);
        }
        CrossoverMethod::LayerWiseBlend => {
            for (left, right) in all_layers_mut(&mut child).into_iter().zip(all_layers(b)) {
                let alpha = rng.0.gen_range(0.0..=1.0) as f32;
                blend_layer(left, right, alpha);
            }
            child.traits = blend_traits(&a.traits, &b.traits, 0.5);
        }
    }
    child.validate(&a.architecture)?;
    Ok(child)
}
fn all_layers(genome: &BrainGenome) -> Vec<&DenseGenome> {
    genome
        .hidden
        .iter()
        .chain(std::iter::once(&genome.recurrent))
        .chain(std::iter::once(&genome.actor))
        .chain(std::iter::once(&genome.critic))
        .collect()
}
fn all_layers_mut(genome: &mut BrainGenome) -> Vec<&mut DenseGenome> {
    let mut result = Vec::with_capacity(genome.hidden.len() + 3);
    result.extend(genome.hidden.iter_mut());
    result.push(&mut genome.recurrent);
    result.push(&mut genome.actor);
    result.push(&mut genome.critic);
    result
}
fn uniform_layer(left: &mut DenseGenome, right: &DenseGenome, rng: &mut EvolutionRng) {
    for (a, b) in left.weights.iter_mut().zip(&right.weights) {
        if rng.0.r#gen() {
            *a = *b;
        }
    }
    for (a, b) in left.bias.iter_mut().zip(&right.bias) {
        if rng.0.r#gen() {
            *a = *b;
        }
    }
}
fn blend_layer(left: &mut DenseGenome, right: &DenseGenome, alpha: f32) {
    for (a, b) in left.weights.iter_mut().zip(&right.weights) {
        *a = alpha * *a + (1.0 - alpha) * *b;
    }
    for (a, b) in left.bias.iter_mut().zip(&right.bias) {
        *a = alpha * *a + (1.0 - alpha) * *b;
    }
}
fn crossover_traits(
    a: &LearningTraits,
    b: &LearningTraits,
    rng: &mut EvolutionRng,
) -> LearningTraits {
    if rng.0.r#gen() { a.clone() } else { b.clone() }
}
fn blend_traits(a: &LearningTraits, b: &LearningTraits, alpha: f32) -> LearningTraits {
    LearningTraits {
        learning_rate: alpha * a.learning_rate + (1.0 - alpha) * b.learning_rate,
        entropy_coefficient: alpha * a.entropy_coefficient + (1.0 - alpha) * b.entropy_coefficient,
        action_temperature: alpha * a.action_temperature + (1.0 - alpha) * b.action_temperature,
        temperature_decay: alpha * a.temperature_decay + (1.0 - alpha) * b.temperature_decay,
        drawdown_sensitivity: alpha * a.drawdown_sensitivity
            + (1.0 - alpha) * b.drawdown_sensitivity,
        turnover_sensitivity: alpha * a.turnover_sensitivity
            + (1.0 - alpha) * b.turnover_sensitivity,
        maximum_allocation: alpha * a.maximum_allocation + (1.0 - alpha) * b.maximum_allocation,
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct MutationReport {
    pub weight_changes: u64,
    pub bias_changes: u64,
    pub trait_changes: u64,
    pub reset_changes: u64,
}
#[derive(Clone, Debug)]
pub struct MutationContext<'a> {
    pub traits: &'a EvolvableTraitsConfig,
}
pub trait GenomeMutator: Send + Sync {
    fn mutate(
        &self,
        genome: &mut BrainGenome,
        context: &MutationContext<'_>,
        rng: &mut EvolutionRng,
    ) -> Result<MutationReport>;
}
#[derive(Clone, Debug)]
pub struct DefaultGenomeMutator {
    pub config: EvolutionMutationConfig,
}
impl GenomeMutator for DefaultGenomeMutator {
    fn mutate(
        &self,
        genome: &mut BrainGenome,
        context: &MutationContext<'_>,
        rng: &mut EvolutionRng,
    ) -> Result<MutationReport> {
        let mut report = MutationReport::default();
        for layer in all_layers_mut(genome) {
            if self.config.reset_probability > 0.0
                && rng.0.gen_bool(self.config.reset_probability as f64)
            {
                report.reset_changes +=
                    layer.bias.iter().filter(|value| **value != 0.0).count() as u64;
                layer.bias.fill(0.0);
            }
            for value in &mut layer.weights {
                if rng.0.gen_bool(self.config.weight_probability as f64) {
                    let before = *value;
                    *value += normal(rng) * self.config.weight_sigma;
                    if *value != before {
                        report.weight_changes += 1;
                    }
                }
            }
            for value in &mut layer.bias {
                if rng.0.gen_bool(self.config.bias_probability as f64) {
                    let before = *value;
                    *value += normal(rng) * self.config.bias_sigma;
                    if *value != before {
                        report.bias_changes += 1;
                    }
                }
            }
        }
        if rng.0.gen_bool(self.config.trait_probability as f64) {
            let mut apply = |value: &mut f32, bounds: &crate::config::EvolvableTraitConfig| {
                if bounds.evolvable {
                    let before = *value;
                    *value = (*value + normal(rng) * bounds.mutation_scale)
                        .clamp(bounds.minimum, bounds.maximum);
                    if *value != before {
                        report.trait_changes += 1;
                    }
                }
            };
            apply(
                &mut genome.traits.learning_rate,
                &context.traits.learning_rate,
            );
            apply(
                &mut genome.traits.entropy_coefficient,
                &context.traits.entropy_coefficient,
            );
            apply(
                &mut genome.traits.action_temperature,
                &context.traits.action_temperature,
            );
            apply(
                &mut genome.traits.temperature_decay,
                &context.traits.temperature_decay,
            );
            apply(
                &mut genome.traits.drawdown_sensitivity,
                &context.traits.drawdown_sensitivity,
            );
            apply(
                &mut genome.traits.turnover_sensitivity,
                &context.traits.turnover_sensitivity,
            );
            apply(
                &mut genome.traits.maximum_allocation,
                &context.traits.maximum_allocation,
            );
        }
        genome.validate_with_traits(&genome.architecture, context.traits)?;
        Ok(report)
    }
}
fn normal(rng: &mut EvolutionRng) -> f32 {
    let u1 = rng.0.gen_range(f64::MIN_POSITIVE..1.0);
    let u2 = rng.0.gen_range(0.0..1.0);
    ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
}

pub fn genome_distance(
    a: &BrainGenome,
    b: &BrainGenome,
    weight_component: f64,
    trait_component: f64,
) -> Result<f64> {
    if a.architecture != b.architecture {
        return Err(MidasError::Policy(
            "cannot compare incompatible genomes".into(),
        ));
    }
    let left = all_layers(a);
    let right = all_layers(b);
    let mut sum = 0.0;
    let mut count = 0usize;
    for (x, y) in left.into_iter().zip(right) {
        for (u, v) in x
            .weights
            .iter()
            .chain(&x.bias)
            .zip(y.weights.iter().chain(&y.bias))
        {
            sum += (*u as f64 - *v as f64).abs();
            count += 1;
        }
    }
    let parameter = if count == 0 { 0.0 } else { sum / count as f64 };
    let trait_distance = [
        (a.traits.learning_rate, b.traits.learning_rate),
        (a.traits.entropy_coefficient, b.traits.entropy_coefficient),
        (a.traits.action_temperature, b.traits.action_temperature),
        (a.traits.temperature_decay, b.traits.temperature_decay),
        (a.traits.drawdown_sensitivity, b.traits.drawdown_sensitivity),
        (a.traits.turnover_sensitivity, b.traits.turnover_sensitivity),
        (a.traits.maximum_allocation, b.traits.maximum_allocation),
    ]
    .iter()
    .map(|(x, y)| (*x as f64 - *y as f64).abs())
    .sum::<f64>()
        / 7.0;
    let value = weight_component * parameter + trait_component * trait_distance;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(MidasError::Policy("genome distance is non-finite".into()))
    }
}

pub trait FitnessFunction: Send + Sync {
    fn evaluate(&self, life: &CompletedLifeSummary) -> Result<FitnessScore>;
}
#[derive(Clone, Copy, Debug)]
pub struct DefaultFitnessFunction {
    pub return_weight: f64,
    pub survival_weight: f64,
    pub drawdown_weight: f64,
    pub turnover_weight: f64,
}
impl Default for DefaultFitnessFunction {
    fn default() -> Self {
        Self {
            return_weight: 1.0,
            survival_weight: 0.01,
            drawdown_weight: 0.5,
            turnover_weight: 0.001,
        }
    }
}
impl DefaultFitnessFunction {
    pub fn from_config(config: &crate::config::EvolutionFitnessConfig) -> Self {
        Self {
            return_weight: config.return_weight,
            survival_weight: config.survival_weight,
            drawdown_weight: config.drawdown_weight,
            turnover_weight: config.turnover_weight,
        }
    }
}
impl FitnessFunction for DefaultFitnessFunction {
    fn evaluate(&self, life: &CompletedLifeSummary) -> Result<FitnessScore> {
        if !life.starting_net_worth.is_finite()
            || life.starting_net_worth <= 0.0
            || !life.final_net_worth.is_finite()
            || !life.maximum_drawdown.is_finite()
            || !life.turnover.is_finite()
        {
            return Err(MidasError::Policy(
                "invalid life metrics for fitness".into(),
            ));
        }
        let return_value = (life.final_net_worth.max(1e-9) / life.starting_net_worth).ln();
        let normalized_return = (return_value / 10.0).clamp(-1.0, 1.0);
        let normalized_survival = (life.age as f64 / 252.0).clamp(0.0, 1.0);
        let normalized_drawdown = life.maximum_drawdown.clamp(0.0, 1.0);
        let normalized_turnover = (life.turnover / life.starting_net_worth).clamp(0.0, 10.0) / 10.0;
        let value = self.return_weight * normalized_return
            + self.survival_weight * normalized_survival
            - self.drawdown_weight * normalized_drawdown
            - self.turnover_weight * normalized_turnover;
        FitnessScore::finite(value)
    }
}

pub fn shared_fitness(
    score: FitnessScore,
    genome: &BrainGenome,
    population: &[BrainGenome],
    radius: f64,
) -> Result<FitnessScore> {
    if radius <= 0.0 || !radius.is_finite() {
        return Ok(score);
    }
    let mut sharing = 0.0;
    for other in population {
        let distance = genome_distance(genome, other, 1.0, 1.0)?;
        if distance < radius {
            sharing += 1.0 - distance / radius;
        }
    }
    FitnessScore::finite(score.0 / sharing.max(1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        brain::{BrainArchitecture, MemoryArchitecture},
        config::{Activation, EvolutionCrossoverConfig, EvolutionMutationConfig},
    };

    fn architecture() -> BrainArchitecture {
        BrainArchitecture {
            version: 1,
            observation_schema_hash: "schema".into(),
            observation_size: 2,
            asset_count: 1,
            hidden_layers: vec![2],
            memory: MemoryArchitecture { size: 1 },
            actor_output_size: 2,
            critic_output_size: 1,
            activation: Activation::Relu,
        }
    }

    #[test]
    fn distance_is_symmetric_and_zero_for_identical_genomes() {
        let a = BrainGenome::random(&architecture(), &EvolvableTraitsConfig::default(), 1).unwrap();
        let b = a.clone();
        assert_eq!(genome_distance(&a, &b, 1.0, 1.0).unwrap(), 0.0);
        let c = BrainGenome::random(&architecture(), &EvolvableTraitsConfig::default(), 2).unwrap();
        assert!(
            (genome_distance(&a, &c, 1.0, 1.0).unwrap()
                - genome_distance(&c, &a, 1.0, 1.0).unwrap())
            .abs()
                < 1e-12
        );
    }

    #[test]
    fn crossover_and_disabled_mutation_preserve_validity() {
        let a = BrainGenome::random(&architecture(), &EvolvableTraitsConfig::default(), 1).unwrap();
        let b = BrainGenome::random(&architecture(), &EvolvableTraitsConfig::default(), 2).unwrap();
        let mut rng = EvolutionRng::seed(3);
        let child = crossover(&a, &b, &EvolutionCrossoverConfig::default(), &mut rng).unwrap();
        child.validate(&architecture()).unwrap();
        let before = child.clone();
        let mutator = DefaultGenomeMutator {
            config: EvolutionMutationConfig {
                weight_probability: 0.0,
                bias_probability: 0.0,
                reset_probability: 0.0,
                trait_probability: 0.0,
                ..EvolutionMutationConfig::default()
            },
        };
        let mut unchanged = child.clone();
        let report = mutator
            .mutate(
                &mut unchanged,
                &MutationContext {
                    traits: &EvolvableTraitsConfig::default(),
                },
                &mut rng,
            )
            .unwrap();
        assert_eq!(report, MutationReport::default());
        assert_eq!(before, unchanged);
        assert_eq!(before.architecture, child.architecture);
        assert_eq!(a.architecture, b.architecture);
    }

    #[test]
    fn tournament_selection_is_deterministic_under_seed() {
        let a = BrainGenome::random(&architecture(), &EvolvableTraitsConfig::default(), 1).unwrap();
        let b = BrainGenome::random(&architecture(), &EvolvableTraitsConfig::default(), 2).unwrap();
        let population = SelectionPopulation {
            candidates: vec![
                ParentCandidate {
                    agent_id: AgentId(1),
                    genome_id: None,
                    genome: a.clone(),
                    fitness: FitnessScore(1.0),
                    age: 2,
                    net_worth: 10.0,
                },
                ParentCandidate {
                    agent_id: AgentId(2),
                    genome_id: None,
                    genome: b.clone(),
                    fitness: FitnessScore(2.0),
                    age: 2,
                    net_worth: 10.0,
                },
            ],
        };
        let config = EvolutionSelectionConfig {
            tournament_size: 2,
            ..EvolutionSelectionConfig::default()
        };
        let mut first = TournamentSelection {
            config: config.clone(),
        };
        let mut second = TournamentSelection { config };
        let mut first_rng = EvolutionRng::seed(44);
        let mut second_rng = EvolutionRng::seed(44);
        assert_eq!(
            first.select_parents(&population, &mut first_rng).unwrap(),
            second.select_parents(&population, &mut second_rng).unwrap()
        );
    }

    #[test]
    fn seeded_mutation_repeats_exactly() {
        let base =
            BrainGenome::random(&architecture(), &EvolvableTraitsConfig::default(), 3).unwrap();
        let mutation = DefaultGenomeMutator {
            config: EvolutionMutationConfig {
                weight_probability: 1.0,
                weight_sigma: 0.01,
                bias_probability: 1.0,
                bias_sigma: 0.01,
                ..EvolutionMutationConfig::default()
            },
        };
        let bounds = EvolvableTraitsConfig::default();
        let mut left = base.clone();
        let mut right = base;
        let mut left_rng = EvolutionRng::seed(9);
        let mut right_rng = EvolutionRng::seed(9);
        mutation
            .mutate(
                &mut left,
                &MutationContext { traits: &bounds },
                &mut left_rng,
            )
            .unwrap();
        mutation
            .mutate(
                &mut right,
                &MutationContext { traits: &bounds },
                &mut right_rng,
            )
            .unwrap();
        assert_eq!(left, right);
    }

    #[test]
    fn first_completed_life_updates_finite_population_metrics() {
        let mut state = EvolutionState::default();
        let fitness = state
            .record_life(CompletedLifeSummary {
                agent_id: AgentId(1),
                genome_id: None,
                birth_step: 0,
                death_step: 3,
                age: 3,
                starting_net_worth: 100.0,
                final_net_worth: 105.0,
                maximum_drawdown: 0.1,
                turnover: 2.0,
                policy_failures: 0,
                fitness: None,
            })
            .unwrap();
        assert!(fitness.0.is_finite());
        assert_eq!(state.completed_lives.len(), 1);
        assert!(state.metrics.mean_fitness.is_finite());
        assert!(state.metrics.fitness_variance.is_finite());
        assert!((state.metrics.mean_lifespan - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn diversity_metrics_are_finite_and_deterministic() {
        let first =
            BrainGenome::random(&architecture(), &EvolvableTraitsConfig::default(), 11).unwrap();
        let second =
            BrainGenome::random(&architecture(), &EvolvableTraitsConfig::default(), 12).unwrap();
        let mut state = EvolutionState::default();
        state.update_diversity(&[&first, &second]).unwrap();
        assert!(state.metrics.mean_genetic_distance.is_finite());
        assert!(state.metrics.minimum_genetic_distance.is_finite());
        assert!(state.metrics.mean_genetic_distance > 0.0);
        assert_eq!(
            state.metrics.mean_genetic_distance,
            state.metrics.minimum_genetic_distance
        );
    }

    #[test]
    fn ancestry_registry_records_parent_generation_and_reason() {
        let architecture = architecture();
        let bounds = EvolvableTraitsConfig::default();
        let parent_a = BrainGenome::random(&architecture, &bounds, 21).unwrap();
        let parent_b = BrainGenome::random(&architecture, &bounds, 22).unwrap();
        let child = crossover(
            &parent_a,
            &parent_b,
            &EvolutionCrossoverConfig::default(),
            &mut EvolutionRng::seed(23),
        )
        .unwrap();
        let mut state = EvolutionState::default();
        let (first, _) = state
            .allocate_genome_with_reason(&parent_a, Vec::new(), 0, None, "founder")
            .unwrap();
        let (second, _) = state
            .allocate_genome_with_reason(&parent_b, Vec::new(), 0, None, "founder")
            .unwrap();
        let (child_id, _) = state
            .allocate_genome_with_reason(
                &child,
                vec![first, second],
                4,
                None,
                "evolution_two_parent",
            )
            .unwrap();
        let record = state.genome_registry.get(&child_id).unwrap();
        assert_eq!(record.parent_ids, vec![first, second]);
        assert_eq!(record.generation_depth, 1);
        assert_eq!(record.birth_step, 4);
        assert_eq!(record.creation_reason, "evolution_two_parent");
    }
}
