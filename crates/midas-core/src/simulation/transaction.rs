use crate::{
    MidasError, Result,
    agent::{Agent, AgentBuildContext},
    evolution::EvolutionState,
    metrics::{AllocationStats, EventSink},
    replacement::ReplacementStrategy,
};

use super::{DomainEvent, OperationalEvent, ReportTrace, SimulationEngine, checkpoint};

pub(super) struct StepTransaction {
    pub(super) agents: Vec<Agent>,
    pub(super) next_agent_id: u64,
    pub(super) events: Vec<DomainEvent>,
    pub(super) replacement: Option<Box<dyn ReplacementStrategy>>,
    pub(super) evolution: EvolutionState,
    pub(super) allocation_stats: AllocationStats,
    pub(super) traces: Vec<ReportTrace>,
}

pub(super) enum StepAbort {
    Error(MidasError),
    FatalExperiment { event: DomainEvent, reason: String },
}

impl From<MidasError> for StepAbort {
    fn from(error: MidasError) -> Self {
        Self::Error(error)
    }
}

impl SimulationEngine {
    pub(super) fn fork_agents(&self) -> Result<Vec<Agent>> {
        self.agents
            .iter()
            .map(|agent| match agent.fork() {
                Ok(forked) => Ok(forked),
                Err(fork_error) => {
                    let snapshot = agent.policy.snapshot().map_err(|restore_error| {
                        MidasError::Policy(format!(
                            "transactional policy fork failed: {fork_error}; snapshot failed: {restore_error}"
                        ))
                    })?;
                    let learning = agent.learning_snapshot()?;
                    let mut restored = self.factory
                        .restore(
                            agent.state.clone(),
                            snapshot,
                            &AgentBuildContext {
                                agent_id: agent.state.id,
                                birth_date: agent.state.birth_date,
                                config: &self.config,
                                architecture: &self.architecture,
                                compute_device: &self.compute_device,
                            },
                        )
                        .map_err(|restore_error| {
                            MidasError::Policy(format!(
                                "transactional policy fork failed: {fork_error}; restore failed: {restore_error}"
                            ))
                        })?;
                    restored.restore_learning(learning)?;
                    Ok(restored)
                }
            })
            .collect()
    }

    pub(super) fn commit_step(&mut self, transaction: StepTransaction) -> Result<()> {
        let checkpoint_due = self.config.checkpoint.every_steps > 0
            && (self.next_step + 1).is_multiple_of(self.config.checkpoint.every_steps);
        let checkpoint_path = self.config.checkpoint.path.clone();
        if checkpoint_due {
            let mut candidate_metrics = self.metrics.clone();
            for event in &transaction.events {
                candidate_metrics.on_event(event);
            }
            candidate_metrics.record_neural_allocations(&transaction.allocation_stats);
            let mut candidate_events = self.domain_events.clone();
            candidate_events.extend(transaction.events.clone());
            let checkpoint = checkpoint::from_state(
                &self.config,
                &self.data,
                &self.observation_schema,
                &self.architecture,
                self.split,
                &self.features,
                &self.compute_device,
                self.next_step + 1,
                transaction.next_agent_id,
                &transaction.agents,
                candidate_metrics,
                candidate_events,
                transaction.evolution.clone(),
                self.lifecycle.clone(),
            )?;
            checkpoint::write(&checkpoint_path, &checkpoint)?;
        }
        let StepTransaction {
            agents,
            next_agent_id,
            events,
            replacement,
            evolution,
            allocation_stats,
            traces,
        } = transaction;
        self.agents = agents;
        self.next_agent_id = next_agent_id;
        self.evolution = evolution;
        self.metrics.record_neural_allocations(&allocation_stats);
        self.report_traces.extend(traces);
        if let Some(replacement) = replacement {
            self.replacement = replacement;
        }
        for event in events {
            self.emit(event);
        }
        self.next_step += 1;
        if checkpoint_due {
            let date = self
                .features
                .get(self.next_step)
                .map(|f| f.date)
                .unwrap_or_else(|| self.features.last().expect("features").date);
            self.operational_events
                .push(OperationalEvent::CheckpointSaved {
                    date,
                    path: checkpoint_path,
                });
        }
        Ok(())
    }
}
