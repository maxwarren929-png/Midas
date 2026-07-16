use crate::{MidasError, Result, metrics::AllocationStats};

use super::{
    DomainEvent, SimulationEngine, SimulationLifecycle, SimulationReport,
    transaction::{StepAbort, StepTransaction},
};

impl SimulationEngine {
    fn lifecycle_error(reason: &str) -> MidasError {
        if let Some(message) = reason.strip_prefix("configuration error: ") {
            MidasError::Config(message.into())
        } else if let Some(message) = reason.strip_prefix("market data error: ") {
            MidasError::Data(message.into())
        } else if let Some(message) = reason.strip_prefix("accounting invariant failed: ") {
            MidasError::Accounting(message.into())
        } else if let Some(message) = reason.strip_prefix("policy error: ") {
            MidasError::Policy(message.into())
        } else if let Some(message) = reason.strip_prefix("checkpoint error: ") {
            MidasError::Checkpoint(message.into())
        } else {
            MidasError::Policy(reason.into())
        }
    }

    pub fn run(&mut self) -> Result<SimulationReport> {
        match &self.lifecycle {
            SimulationLifecycle::Completed => return Ok(self.report()),
            SimulationLifecycle::Failed { reason } => {
                return Err(Self::lifecycle_error(reason));
            }
            SimulationLifecycle::Initialized | SimulationLifecycle::Running => {}
        }
        self.lifecycle = SimulationLifecycle::Running;
        while self.next_step + 1 < self.features.len() {
            self.step()?;
        }
        if let Err(error) = self.flush_learning() {
            self.lifecycle = SimulationLifecycle::Failed {
                reason: error.to_string(),
            };
            return Err(error);
        }
        self.lifecycle = SimulationLifecycle::Completed;
        let date = self.features.last().expect("validated features").date;
        self.emit(DomainEvent::TimelineCompleted { date });
        Ok(self.report())
    }

    pub fn run_steps(&mut self, count: usize) -> Result<()> {
        match &self.lifecycle {
            SimulationLifecycle::Completed => return Ok(()),
            SimulationLifecycle::Failed { reason } => {
                return Err(Self::lifecycle_error(reason));
            }
            SimulationLifecycle::Initialized | SimulationLifecycle::Running => {}
        }
        self.lifecycle = SimulationLifecycle::Running;
        for _ in 0..count {
            if self.next_step + 1 >= self.features.len() {
                break;
            }
            self.step()?;
        }
        Ok(())
    }

    fn step(&mut self) -> Result<()> {
        let agents = match self.fork_agents() {
            Ok(agents) => agents,
            Err(error) => {
                let reason = error.to_string();
                self.lifecycle = SimulationLifecycle::Failed {
                    reason: reason.clone(),
                };
                return Err(error);
            }
        };
        let transaction = StepTransaction {
            agents,
            next_agent_id: self.next_agent_id,
            events: Vec::new(),
            replacement: None,
            evolution: self.evolution.clone(),
            allocation_stats: AllocationStats::default(),
            traces: Vec::new(),
        };
        let mut transaction = transaction;
        match self.execute_step(&mut transaction) {
            Ok(()) => match self.commit_step(transaction) {
                Ok(()) => Ok(()),
                Err(error) => {
                    self.lifecycle = SimulationLifecycle::Failed {
                        reason: error.to_string(),
                    };
                    Err(error)
                }
            },
            Err(StepAbort::Error(error)) => {
                let reason = error.to_string();
                self.lifecycle = SimulationLifecycle::Failed {
                    reason: reason.clone(),
                };
                Err(error)
            }
            Err(StepAbort::FatalExperiment { event, reason }) => {
                self.emit(event);
                self.lifecycle = SimulationLifecycle::Failed {
                    reason: reason.clone(),
                };
                Err(MidasError::Policy(reason))
            }
        }
    }

    fn flush_learning(&mut self) -> Result<()> {
        let mut agents = self.fork_agents()?;
        let mut events = Vec::new();
        let mut traces = Vec::new();
        let date = self.features.last().expect("validated features").date;
        for agent in &mut agents {
            if agent.state.alive {
                self.finish_learning_episode(
                    agent,
                    date,
                    agent.state.portfolio.net_worth,
                    true,
                    &mut events,
                    &mut traces,
                )?;
            }
        }
        self.agents = agents;
        for event in events {
            self.emit(event);
        }
        self.report_traces.extend(traces);
        Ok(())
    }
}
