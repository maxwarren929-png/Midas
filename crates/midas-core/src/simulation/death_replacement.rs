use crate::{
    AgentId, MidasError, Result,
    agent::{AgentBuildContext, AgentSpawnRequest},
    brain::PolicyBlueprint,
    config::DeathHandling,
    evolution::{CompletedLifeSummary, DefaultFitnessFunction},
    replacement::{EvolutionReplacementOutcome, LivingAgentSummary, ReplacementContext},
};
use chrono::NaiveDate;

use super::{DomainEvent, SimulationEngine, transaction::StepTransaction};

fn validate_replacement_request(
    request: &AgentSpawnRequest,
    living: &[LivingAgentSummary<'_>],
) -> Result<()> {
    match (&request.policy_blueprint, &request.genome) {
        (PolicyBlueprint::Neural, None) => {
            return Err(MidasError::Policy(
                "neural replacement requires a genome".into(),
            ));
        }
        (PolicyBlueprint::Neural, Some(_)) | (_, None) => {}
        (_, Some(_)) => {
            return Err(MidasError::Policy(
                "non-neural replacement must not carry a genome".into(),
            ));
        }
    }
    for parent_id in &request.parent_ids {
        let parent = living
            .iter()
            .find(|candidate| candidate.id == *parent_id)
            .ok_or_else(|| {
                MidasError::Policy(format!(
                    "replacement parent {parent_id} is not in the living population"
                ))
            })?;
        if request.genome.is_some() && parent.genome.is_none() {
            return Err(MidasError::Policy(format!(
                "replacement parent {parent_id} has no genome"
            )));
        }
    }
    for (name, value) in [
        (
            "requested mating distance",
            request.metadata.requested_mating_distance,
        ),
        (
            "selected mating distance",
            request.metadata.selected_mating_distance,
        ),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(MidasError::Policy(format!(
                "replacement {name} must be finite and non-negative"
            )));
        }
    }
    Ok(())
}

fn inherited_lineage(
    parent_lineages: &[crate::evolution::LineageId],
) -> Option<crate::evolution::LineageId> {
    let first = parent_lineages.first().copied()?;
    parent_lineages
        .iter()
        .all(|lineage| *lineage == first)
        .then_some(first)
}

impl SimulationEngine {
    pub(super) fn handle_death_in_transaction(
        &mut self,
        transaction: &mut StepTransaction,
        id: AgentId,
        date: NaiveDate,
    ) -> Result<()> {
        let dead_index = transaction
            .agents
            .iter()
            .position(|a| a.state.id == id)
            .ok_or_else(|| MidasError::Accounting("dead agent missing".into()))?;
        let dead_snapshot = transaction.agents[dead_index].state.clone();
        let birth_step = self
            .features
            .iter()
            .position(|feature| feature.date == dead_snapshot.birth_date)
            .unwrap_or(0);
        let life = CompletedLifeSummary {
            agent_id: id,
            genome_id: dead_snapshot.genome_id,
            birth_step,
            death_step: self.next_step,
            age: dead_snapshot.age,
            starting_net_worth: self.config.simulation.starting_cash,
            final_net_worth: dead_snapshot.portfolio.net_worth,
            maximum_drawdown: dead_snapshot.maximum_drawdown,
            turnover: dead_snapshot.portfolio.cumulative_turnover,
            policy_failures: dead_snapshot.total_policy_failures,
            fitness: None,
        };
        let fitness_function = DefaultFitnessFunction::from_config(&self.config.evolution.fitness);
        let fitness = transaction
            .evolution
            .record_life_with(life, &fitness_function)?;
        transaction.events.push(DomainEvent::FitnessAssigned {
            agent_id: id,
            genome_id: dead_snapshot.genome_id,
            fitness: fitness.0,
        });
        let living_genomes = transaction
            .agents
            .iter()
            .filter(|candidate| candidate.state.alive)
            .filter_map(|candidate| candidate.state.genome.as_ref())
            .collect::<Vec<_>>();
        transaction.evolution.update_diversity(&living_genomes)?;
        match self.config.simulation.death_handling {
            DeathHandling::LeaveVacant => Ok(()),
            DeathHandling::EndAgentOnly => {
                transaction.agents.retain(|a| a.state.id != id);
                Ok(())
            }
            DeathHandling::Replace => {
                let slot = transaction
                    .agents
                    .iter()
                    .position(|a| a.state.id == id)
                    .ok_or_else(|| MidasError::Accounting("dead agent missing".into()))?;
                let dead_state = transaction.agents[slot].state.clone();
                let dead_policy = transaction.agents[slot].policy.policy_blueprint();
                let new_id = AgentId(transaction.next_agent_id);
                transaction.next_agent_id += 1;
                let mut replacement = match transaction.replacement.take() {
                    Some(replacement) => replacement,
                    None => self.replacement.fork()?,
                };
                let (request, evolution_report) = {
                    let living = transaction
                        .agents
                        .iter()
                        .filter(|a| a.state.alive)
                        .map(|a| LivingAgentSummary {
                            id: a.state.id,
                            net_worth: a.state.portfolio.net_worth,
                            age: a.state.age,
                            policy: a.policy.policy_blueprint(),
                            genome: a.state.genome.as_ref(),
                            maximum_drawdown: a.state.maximum_drawdown,
                            birth_date: a.state.birth_date,
                            parent_ids: a.state.parent_ids.clone(),
                        })
                        .collect::<Vec<_>>();
                    let request = replacement.create_replacement(&ReplacementContext {
                        dead_agent: &dead_state,
                        dead_policy,
                        living_population: &living,
                        config: &self.config,
                        architecture: &self.architecture,
                        compute_device: &self.compute_device,
                        observation_schema: &self.observation_schema,
                        deterministic_seed: transaction.evolution.rng.next_u64()
                            ^ self.config.simulation.seed
                            ^ new_id.0.wrapping_mul(0x9e3779b97f4a7c15),
                        completed_lives: &transaction.evolution.completed_lives,
                        genome_registry: &transaction.evolution.genome_registry,
                    })?;
                    validate_replacement_request(&request, &living)?;
                    (request, replacement.evolution_report())
                };
                let parent_ids = request.parent_ids.clone();
                let first_parent_id = parent_ids.first().copied().unwrap_or(AgentId(0));
                let request_reason = request.metadata.reason.clone();
                let diversity_relaxed = request.metadata.diversity_constraint_relaxed;
                let requested_mating_distance = request.metadata.requested_mating_distance;
                let selected_mating_distance = request.metadata.selected_mating_distance;
                let child_genome = request.genome.clone();
                transaction.replacement = Some(replacement);
                let mut agent = self.factory.spawn(
                    request,
                    &AgentBuildContext {
                        agent_id: new_id,
                        birth_date: date,
                        config: &self.config,
                        architecture: &self.architecture,
                        compute_device: &self.compute_device,
                    },
                )?;
                if let Some(genome) = child_genome.as_ref() {
                    let parent_genome_ids = parent_ids
                        .iter()
                        .map(|parent_id| {
                            let parent = transaction
                                .agents
                                .iter()
                                .find(|candidate| candidate.state.id == *parent_id)
                                .ok_or_else(|| {
                                    MidasError::Policy(format!(
                                        "replacement parent {parent_id} disappeared before ancestry allocation"
                                    ))
                                })?;
                            parent.state.genome_id.ok_or_else(|| {
                                MidasError::Policy(format!(
                                    "replacement parent {parent_id} has no genome identity"
                                ))
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let parent_lineages = parent_genome_ids
                        .iter()
                        .filter_map(|parent_id| {
                            transaction.evolution.genome_registry.get(parent_id)
                        })
                        .map(|record| record.lineage_id)
                        .collect::<Vec<_>>();
                    let shared_lineage = inherited_lineage(&parent_lineages);
                    let (genome_id, lineage_id) =
                        transaction.evolution.allocate_genome_with_reason(
                            genome,
                            parent_genome_ids,
                            self.next_step,
                            shared_lineage,
                            request_reason.clone(),
                        )?;
                    agent.state.genome_id = Some(genome_id);
                    agent.state.lineage_id = Some(lineage_id);
                    if parent_lineages.len() >= 2
                        && parent_lineages.windows(2).any(|pair| pair[0] != pair[1])
                    {
                        transaction.events.push(DomainEvent::LineageCreated {
                            lineage_id,
                            parent_lineages,
                        });
                    }
                    if !parent_ids.is_empty() {
                        transaction.evolution.metrics.reproduction_count = transaction
                            .evolution
                            .metrics
                            .reproduction_count
                            .saturating_add(1);
                        transaction.events.push(DomainEvent::ParentsSelected {
                            child_agent_id: new_id,
                            parent_agent_ids: parent_ids,
                        });
                        if diversity_relaxed {
                            transaction
                                .evolution
                                .metrics
                                .diversity_constraint_relaxations = transaction
                                .evolution
                                .metrics
                                .diversity_constraint_relaxations
                                .saturating_add(1);
                            transaction
                                .events
                                .push(DomainEvent::DiversityConstraintRelaxed {
                                    first_parent_agent_id: first_parent_id,
                                    requested_minimum_distance: requested_mating_distance,
                                    selected_distance: selected_mating_distance,
                                });
                        }
                    }
                    if let Some(report) = evolution_report {
                        if report.crossover {
                            transaction.evolution.metrics.crossover_count = transaction
                                .evolution
                                .metrics
                                .crossover_count
                                .saturating_add(1);
                            transaction.events.push(DomainEvent::GenomeCrossed {
                                child_agent_id: new_id,
                                method: self.config.evolution.crossover.method,
                            });
                        }
                        if let Some(mutation) = report.mutation {
                            transaction.evolution.metrics.mutation_count = transaction
                                .evolution
                                .metrics
                                .mutation_count
                                .saturating_add(1);
                            transaction.events.push(DomainEvent::GenomeMutated {
                                child_agent_id: new_id,
                                weight_changes: mutation.weight_changes,
                                bias_changes: mutation.bias_changes,
                                trait_changes: mutation.trait_changes,
                                reset_changes: mutation.reset_changes,
                                weight_resets: mutation.weight_resets,
                                bias_resets: mutation.bias_resets,
                            });
                        }
                        match report.outcome {
                            EvolutionReplacementOutcome::Reproduction => {}
                            EvolutionReplacementOutcome::CloneFallback => {
                                transaction.events.push(DomainEvent::EvolutionFallback {
                                    reason: request_reason,
                                });
                            }
                            EvolutionReplacementOutcome::RandomFallback => {
                                transaction.evolution.metrics.random_fallback_births = transaction
                                    .evolution
                                    .metrics
                                    .random_fallback_births
                                    .saturating_add(1);
                                transaction.events.push(DomainEvent::EvolutionFallback {
                                    reason: request_reason,
                                });
                            }
                        }
                    }
                }
                transaction.agents[slot] = agent;
                let living_genomes = transaction
                    .agents
                    .iter()
                    .filter(|candidate| candidate.state.alive)
                    .filter_map(|candidate| candidate.state.genome.as_ref())
                    .collect::<Vec<_>>();
                transaction.evolution.update_diversity(&living_genomes)?;
                transaction.events.push(DomainEvent::AgentBorn {
                    agent_id: new_id,
                    date,
                });
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{inherited_lineage, validate_replacement_request};
    use crate::{
        AgentId,
        agent::{AgentSpawnRequest, SpawnMetadata},
        brain::PolicyBlueprint,
        evolution::LineageId,
    };

    #[test]
    fn one_parent_clone_inherits_its_lineage() {
        assert_eq!(inherited_lineage(&[LineageId(7)]), Some(LineageId(7)));
        assert_eq!(
            inherited_lineage(&[LineageId(7), LineageId(7)]),
            Some(LineageId(7))
        );
        assert_eq!(inherited_lineage(&[LineageId(7), LineageId(8)]), None);
        assert_eq!(inherited_lineage(&[]), None);
    }

    #[test]
    fn replacement_request_rejects_unknown_parents() {
        let request = AgentSpawnRequest {
            policy_blueprint: PolicyBlueprint::Cash,
            genome: None,
            parent_ids: vec![AgentId(42)],
            metadata: SpawnMetadata::default(),
        };
        assert_eq!(
            validate_replacement_request(&request, &[])
                .unwrap_err()
                .to_string(),
            "policy error: replacement parent 42 is not in the living population"
        );
    }
}
