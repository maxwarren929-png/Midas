use crate::{
    AgentId, Result,
    agent::{AgentSpawnRequest, AgentState, SpawnMetadata},
    brain::{BrainArchitecture, BrainGenome, ComputeDevice, PolicyBlueprint},
    config::Config,
    evolution::{
        CompletedLifeSummary, DefaultGenomeMutator, EvolutionRng, FitnessScore, GenomeMutator,
        GenomeRecord, MutationContext, MutationReport, ParentCandidate, ParentSelector,
        RankWeightedSelection, SelectionPopulation, TournamentSelection, crossover,
    },
    observation::ObservationSchema,
};

#[derive(Clone, Debug)]
pub struct LivingAgentSummary<'a> {
    pub id: AgentId,
    pub net_worth: f64,
    pub selection_fitness: FitnessScore,
    pub age: u64,
    pub policy: PolicyBlueprint,
    pub genome: Option<&'a BrainGenome>,
    pub maximum_drawdown: f64,
    pub birth_date: chrono::NaiveDate,
    pub parent_ids: Vec<AgentId>,
}
pub struct ReplacementContext<'a> {
    pub dead_agent: &'a AgentState,
    pub dead_policy: PolicyBlueprint,
    pub living_population: &'a [LivingAgentSummary<'a>],
    pub config: &'a Config,
    pub architecture: &'a BrainArchitecture,
    pub compute_device: &'a ComputeDevice,
    pub observation_schema: &'a ObservationSchema,
    pub deterministic_seed: u64,
    pub completed_lives: &'a [CompletedLifeSummary],
    pub genome_registry: &'a std::collections::BTreeMap<crate::evolution::GenomeId, GenomeRecord>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvolutionReplacementOutcome {
    Reproduction,
    CloneFallback,
    RandomFallback,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EvolutionReplacementReport {
    pub outcome: EvolutionReplacementOutcome,
    pub crossover: bool,
    pub mutation: Option<MutationReport>,
}

pub trait ReplacementStrategy: Send {
    fn create_replacement(&mut self, context: &ReplacementContext<'_>)
    -> Result<AgentSpawnRequest>;
    fn name(&self) -> &'static str;
    /// Returns the mutation report produced by the most recent request, when
    /// the strategy has evolutionary mutation diagnostics to expose.
    fn mutation_report(&self) -> Option<MutationReport> {
        None
    }
    /// Returns structured evolutionary diagnostics for the most recent
    /// replacement request. Implementations that do not perform evolution may
    /// retain the default.
    fn evolution_report(&self) -> Option<EvolutionReplacementReport> {
        None
    }
    /// Clone strategy state for a transactional simulation step. Stateful
    /// strategies must override this before they can replace agents.
    fn fork(&self) -> Result<Box<dyn ReplacementStrategy>> {
        Err(crate::MidasError::Policy(
            "replacement strategy does not support transactional forking".into(),
        ))
    }
}
pub struct NonEvolutionaryRandomGenomeReplacement;
impl ReplacementStrategy for NonEvolutionaryRandomGenomeReplacement {
    fn create_replacement(&mut self, c: &ReplacementContext<'_>) -> Result<AgentSpawnRequest> {
        let genome = if c.dead_policy == PolicyBlueprint::Neural {
            Some(BrainGenome::random(
                c.architecture,
                &c.config.brain.traits,
                c.deterministic_seed,
            )?)
        } else {
            None
        };
        Ok(AgentSpawnRequest {
            policy_blueprint: c.dead_policy.clone(),
            genome,
            parent_ids: Vec::new(),
            metadata: SpawnMetadata {
                reason: "non_evolutionary_replacement".into(),
                ..SpawnMetadata::default()
            },
        })
    }
    fn name(&self) -> &'static str {
        "non_evolutionary_random_genome"
    }
    fn fork(&self) -> Result<Box<dyn ReplacementStrategy>> {
        Ok(Box::new(Self))
    }
}

#[derive(Clone, Default)]
pub struct EvolutionaryReplacementStrategy {
    last_report: Option<EvolutionReplacementReport>,
}
impl ReplacementStrategy for EvolutionaryReplacementStrategy {
    fn create_replacement(&mut self, c: &ReplacementContext<'_>) -> Result<AgentSpawnRequest> {
        self.last_report = None;
        if c.dead_policy != PolicyBlueprint::Neural {
            return Ok(AgentSpawnRequest {
                policy_blueprint: c.dead_policy.clone(),
                genome: None,
                parent_ids: Vec::new(),
                metadata: SpawnMetadata {
                    reason: "evolution_non_neural_policy".into(),
                    ..SpawnMetadata::default()
                },
            });
        }
        let candidates = c
            .living_population
            .iter()
            .filter_map(|summary| {
                let genome = summary.genome?.clone();
                let fitness = FitnessScore::finite(summary.selection_fitness.0).ok()?;
                Some(ParentCandidate {
                    agent_id: summary.id,
                    genome_id: None,
                    genome,
                    fitness,
                    age: summary.age,
                    net_worth: summary.net_worth,
                })
            })
            .collect::<Vec<_>>();
        let mut candidates = candidates;
        if c.config.evolution.diversity.fitness_sharing {
            let genomes = candidates
                .iter()
                .map(|candidate| candidate.genome.clone())
                .collect::<Vec<_>>();
            for candidate in &mut candidates {
                candidate.fitness = crate::evolution::shared_fitness(
                    candidate.fitness,
                    &candidate.genome,
                    &genomes,
                    c.config.evolution.diversity.sharing_radius as f64,
                )?;
            }
        }
        let population = SelectionPopulation { candidates };
        let stream_seed = c.deterministic_seed ^ 0x9e3779b97f4a7c15;
        let mut selection_rng = EvolutionRng::seed(stream_seed ^ 0x53454c454354_u64);
        let mut crossover_rng = EvolutionRng::seed(stream_seed ^ 0x43524f53534f_u64);
        let mut mutation_rng = EvolutionRng::seed(stream_seed ^ 0x4d5554415445_u64);
        let selection_config = c.config.evolution.selection.clone();
        let mut selector: Box<dyn ParentSelector> = match selection_config.method {
            crate::config::SelectionMethod::Tournament => Box::new(TournamentSelection {
                config: selection_config.clone(),
            }),
            crate::config::SelectionMethod::RankWeighted => Box::new(RankWeightedSelection {
                config: selection_config.clone(),
            }),
        };
        let minimum_distance = c.config.evolution.diversity.minimum_mating_distance as f64;
        let parents = match selector.select_parents_with_constraints(
            &population,
            &crate::evolution::PairConstraints { minimum_distance },
            &mut selection_rng,
        ) {
            Ok(parents) => parents,
            Err(error) => {
                let (request, outcome) = match c.config.evolution.fallback {
                    crate::config::EvolutionFallback::AbortExperiment => Err(error),
                    crate::config::EvolutionFallback::CloneBestEligible => {
                        let best = population
                            .candidates
                            .iter()
                            .filter(|candidate| {
                                candidate.age >= selection_config.minimum_age
                                    && candidate.net_worth >= selection_config.minimum_net_worth
                            })
                            .max_by(|a, b| a.fitness.0.total_cmp(&b.fitness.0))
                            .ok_or(error)?;
                        Ok((
                            AgentSpawnRequest {
                                policy_blueprint: PolicyBlueprint::Neural,
                                genome: Some(best.genome.clone()),
                                parent_ids: vec![best.agent_id],
                                metadata: SpawnMetadata {
                                    reason: "evolution_clone_best_fallback".into(),
                                    ..SpawnMetadata::default()
                                },
                            },
                            EvolutionReplacementOutcome::CloneFallback,
                        ))
                    }
                    crate::config::EvolutionFallback::RandomGenome => Ok((
                        AgentSpawnRequest {
                            policy_blueprint: PolicyBlueprint::Neural,
                            genome: Some(BrainGenome::random(
                                c.architecture,
                                &c.config.brain.traits,
                                c.deterministic_seed,
                            )?),
                            parent_ids: Vec::new(),
                            metadata: SpawnMetadata {
                                reason: "evolution_random_genome_fallback".into(),
                                ..SpawnMetadata::default()
                            },
                        },
                        EvolutionReplacementOutcome::RandomFallback,
                    )),
                }?;
                self.last_report = Some(EvolutionReplacementReport {
                    outcome,
                    crossover: false,
                    mutation: None,
                });
                return Ok(request);
            }
        };
        let mut fallback_reason = None;
        let mut fallback_outcome = None;
        let mut crossover_applied = true;
        let mut child = match crossover(
            &parents.first.genome,
            &parents.second.genome,
            &c.config.evolution.crossover,
            &mut crossover_rng,
        ) {
            Ok(child) => child,
            Err(error) => match c.config.evolution.fallback {
                crate::config::EvolutionFallback::AbortExperiment => return Err(error),
                crate::config::EvolutionFallback::CloneBestEligible => {
                    fallback_reason = Some(format!("crossover_failed:{error}"));
                    fallback_outcome = Some(EvolutionReplacementOutcome::CloneFallback);
                    crossover_applied = false;
                    parents.first.genome.clone()
                }
                crate::config::EvolutionFallback::RandomGenome => {
                    fallback_reason = Some(format!("crossover_failed:{error}"));
                    fallback_outcome = Some(EvolutionReplacementOutcome::RandomFallback);
                    crossover_applied = false;
                    BrainGenome::random(
                        c.architecture,
                        &c.config.brain.traits,
                        c.deterministic_seed,
                    )?
                }
            },
        };
        let mutator = DefaultGenomeMutator {
            config: c.config.evolution.mutation.clone(),
        };
        let mutation_report = match mutator.mutate(
            &mut child,
            &MutationContext {
                traits: &c.config.brain.traits,
            },
            &mut mutation_rng,
        ) {
            Ok(report) => Some(report),
            Err(error) => {
                match c.config.evolution.fallback {
                    crate::config::EvolutionFallback::AbortExperiment => return Err(error),
                    crate::config::EvolutionFallback::CloneBestEligible => {
                        fallback_reason = Some(format!("mutation_failed:{error}"));
                        fallback_outcome = Some(EvolutionReplacementOutcome::CloneFallback);
                        crossover_applied = false;
                        child = parents.first.genome.clone()
                    }
                    crate::config::EvolutionFallback::RandomGenome => {
                        fallback_reason = Some(format!("mutation_failed:{error}"));
                        fallback_outcome = Some(EvolutionReplacementOutcome::RandomFallback);
                        crossover_applied = false;
                        child = BrainGenome::random(
                            c.architecture,
                            &c.config.brain.traits,
                            c.deterministic_seed,
                        )?
                    }
                }
                None
            }
        };
        let outcome = fallback_outcome.unwrap_or(EvolutionReplacementOutcome::Reproduction);
        self.last_report = Some(EvolutionReplacementReport {
            outcome,
            crossover: crossover_applied,
            mutation: mutation_report,
        });
        let reason = fallback_reason.unwrap_or_else(|| {
            format!(
                "evolution_two_parent:{}+{}",
                parents.first.agent_id, parents.second.agent_id
            )
        });
        Ok(AgentSpawnRequest {
            policy_blueprint: PolicyBlueprint::Neural,
            genome: Some(child),
            parent_ids: match outcome {
                EvolutionReplacementOutcome::Reproduction => {
                    vec![parents.first.agent_id, parents.second.agent_id]
                }
                EvolutionReplacementOutcome::CloneFallback => vec![parents.first.agent_id],
                EvolutionReplacementOutcome::RandomFallback => Vec::new(),
            },
            metadata: SpawnMetadata {
                reason,
                diversity_constraint_relaxed: parents.distance_relaxed,
                requested_mating_distance: minimum_distance,
                selected_mating_distance: parents.selected_distance,
            },
        })
    }
    fn name(&self) -> &'static str {
        "evolutionary_two_parent"
    }
    fn mutation_report(&self) -> Option<MutationReport> {
        self.last_report
            .as_ref()
            .and_then(|report| report.mutation.clone())
    }
    fn evolution_report(&self) -> Option<EvolutionReplacementReport> {
        self.last_report.clone()
    }
    fn fork(&self) -> Result<Box<dyn ReplacementStrategy>> {
        Ok(Box::new(self.clone()))
    }
}
