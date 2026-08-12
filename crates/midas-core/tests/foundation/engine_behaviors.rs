use super::*;

struct MockModel {
    schema: String,
    size: usize,
    memory: f32,
    fail: bool,
}
impl NeuralModel for MockModel {
    fn forward(&mut self, input: &NeuralInput<'_>) -> midas_core::Result<NeuralOutput> {
        assert_eq!(input.observation.schema_hash, self.schema);
        if self.fail {
            return Err(MidasError::Policy(
                "non-finite neural output injected".into(),
            ));
        }
        self.memory += 1.0;
        Ok(NeuralOutput {
            logits: vec![0.0; self.size],
            value: self.memory,
        })
    }
    fn reset_memory(&mut self) {
        self.memory = 0.0
    }
    fn snapshot_state(&self) -> midas_core::Result<NeuralModelState> {
        Ok(NeuralModelState::External {
            kind: "mock".into(),
            state: vec![self.memory],
        })
    }
    fn fork(&self) -> midas_core::Result<Box<dyn NeuralModel>> {
        Ok(Box::new(Self {
            schema: self.schema.clone(),
            size: self.size,
            memory: self.memory,
            fail: self.fail,
        }))
    }
}
struct MockFactory {
    fail: bool,
}
impl NeuralModelFactory for MockFactory {
    fn build(
        &self,
        _: &BrainGenome,
        a: &BrainArchitecture,
        device: &ComputeDevice,
    ) -> midas_core::Result<Box<dyn NeuralModel>> {
        assert_eq!(device.identity(), "cpu_ndarray:default");
        Ok(Box::new(MockModel {
            schema: a.observation_schema_hash.clone(),
            size: a.actor_output_size,
            memory: 0.0,
            fail: self.fail,
        }))
    }
    fn restore(
        &self,
        _: &BrainGenome,
        a: &BrainArchitecture,
        state: NeuralModelState,
        device: &ComputeDevice,
    ) -> midas_core::Result<Box<dyn NeuralModel>> {
        assert_eq!(device.identity(), "cpu_ndarray:default");
        let NeuralModelState::External { state, .. } = state else {
            return Err(MidasError::Policy("wrong mock state".into()));
        };
        Ok(Box::new(MockModel {
            schema: a.observation_schema_hash.clone(),
            size: a.actor_output_size,
            memory: state[0],
            fail: self.fail,
        }))
    }
}

struct BootstrapForkFailureModel {
    schema: String,
    size: usize,
    fork_count: u32,
}
impl NeuralModel for BootstrapForkFailureModel {
    fn forward(&mut self, input: &NeuralInput<'_>) -> midas_core::Result<NeuralOutput> {
        assert_eq!(input.observation.schema_hash, self.schema);
        Ok(NeuralOutput {
            logits: vec![0.0; self.size],
            value: 0.0,
        })
    }
    fn reset_memory(&mut self) {}
    fn snapshot_state(&self) -> midas_core::Result<NeuralModelState> {
        Ok(NeuralModelState::External {
            kind: "bootstrap-fork-failure".into(),
            state: vec![self.fork_count as f32],
        })
    }
    fn fork(&self) -> midas_core::Result<Box<dyn NeuralModel>> {
        if self.fork_count >= 1 {
            return Err(MidasError::Policy(
                "injected next-value fork failure".into(),
            ));
        }
        Ok(Box::new(Self {
            schema: self.schema.clone(),
            size: self.size,
            fork_count: self.fork_count + 1,
        }))
    }
}
struct BootstrapForkFailureFactory;
impl NeuralModelFactory for BootstrapForkFailureFactory {
    fn build(
        &self,
        _: &BrainGenome,
        a: &BrainArchitecture,
        _: &ComputeDevice,
    ) -> midas_core::Result<Box<dyn NeuralModel>> {
        Ok(Box::new(BootstrapForkFailureModel {
            schema: a.observation_schema_hash.clone(),
            size: a.actor_output_size,
            fork_count: 0,
        }))
    }
    fn restore(
        &self,
        _: &BrainGenome,
        a: &BrainArchitecture,
        state: NeuralModelState,
        _: &ComputeDevice,
    ) -> midas_core::Result<Box<dyn NeuralModel>> {
        let NeuralModelState::External { state, .. } = state else {
            return Err(MidasError::Policy("wrong bootstrap failure state".into()));
        };
        Ok(Box::new(BootstrapForkFailureModel {
            schema: a.observation_schema_hash.clone(),
            size: a.actor_output_size,
            fork_count: state.first().copied().unwrap_or(0.0) as u32,
        }))
    }
}

#[derive(Clone)]
struct SequencedFactory {
    fail_calls: Vec<u32>,
    invalid_calls: Vec<u32>,
}
struct SequencedModel {
    schema: String,
    size: usize,
    memory: f32,
    calls: u32,
    fail_calls: Vec<u32>,
    invalid_calls: Vec<u32>,
}
impl NeuralModel for SequencedModel {
    fn forward(&mut self, input: &NeuralInput<'_>) -> midas_core::Result<NeuralOutput> {
        assert_eq!(input.observation.schema_hash, self.schema);
        self.calls += 1;
        if self.fail_calls.contains(&self.calls) {
            return Err(MidasError::Policy(format!(
                "sequenced inference failure at call {}",
                self.calls
            )));
        }
        if self.invalid_calls.contains(&self.calls) {
            return Ok(NeuralOutput {
                logits: vec![f32::NAN; self.size],
                value: f32::NAN,
            });
        }
        self.memory += 1.0;
        Ok(NeuralOutput {
            logits: vec![0.0; self.size],
            value: self.memory,
        })
    }
    fn reset_memory(&mut self) {
        self.memory = 0.0;
    }
    fn snapshot_state(&self) -> midas_core::Result<NeuralModelState> {
        Ok(NeuralModelState::External {
            kind: "sequenced".into(),
            state: vec![self.memory, self.calls as f32],
        })
    }
    fn fork(&self) -> midas_core::Result<Box<dyn NeuralModel>> {
        Ok(Box::new(Self {
            schema: self.schema.clone(),
            size: self.size,
            memory: self.memory,
            calls: self.calls,
            fail_calls: self.fail_calls.clone(),
            invalid_calls: self.invalid_calls.clone(),
        }))
    }
}
impl NeuralModelFactory for SequencedFactory {
    fn build(
        &self,
        _: &BrainGenome,
        a: &BrainArchitecture,
        device: &ComputeDevice,
    ) -> midas_core::Result<Box<dyn NeuralModel>> {
        assert_eq!(device.identity(), "cpu_ndarray:default");
        Ok(Box::new(SequencedModel {
            schema: a.observation_schema_hash.clone(),
            size: a.actor_output_size,
            memory: 0.0,
            calls: 0,
            fail_calls: self.fail_calls.clone(),
            invalid_calls: self.invalid_calls.clone(),
        }))
    }
    fn restore(
        &self,
        _: &BrainGenome,
        a: &BrainArchitecture,
        state: NeuralModelState,
        device: &ComputeDevice,
    ) -> midas_core::Result<Box<dyn NeuralModel>> {
        assert_eq!(device.identity(), "cpu_ndarray:default");
        let NeuralModelState::External { state, .. } = state else {
            return Err(MidasError::Policy("wrong sequenced model state".into()));
        };
        Ok(Box::new(SequencedModel {
            schema: a.observation_schema_hash.clone(),
            size: a.actor_output_size,
            memory: state.first().copied().unwrap_or(0.0),
            calls: state.get(1).copied().unwrap_or(0.0) as u32,
            fail_calls: self.fail_calls.clone(),
            invalid_calls: self.invalid_calls.clone(),
        }))
    }
}

fn data_with_actions(c: &mut Config, actions: Vec<CorporateAction>) -> MarketDataset {
    c.corporate_actions.mode = CorporateActionMode::ExplicitEvents;
    let source = loaded(c);
    MarketDataset::new_with_actions(source.assets, source.bars, actions, &c.corporate_actions)
        .unwrap()
}

fn sequenced_engine(
    mut c: Config,
    actions: Vec<CorporateAction>,
    factory: SequencedFactory,
) -> SimulationEngine {
    c.simulation.population_size = 1;
    let data = data_with_actions(&mut c, actions);
    SimulationEngine::new_with_factory(
        c,
        data,
        DatasetSplit::Training,
        Box::new(StandardAgentFactory::with_neural_factory(Box::new(factory))),
    )
    .unwrap()
}
#[test]
fn simulation_runs_mock_and_burn_neural_models() {
    let c = config();
    let data = loaded(&c);
    let factory = StandardAgentFactory::with_neural_factory(Box::new(MockFactory { fail: false }));
    let mut mock = SimulationEngine::new_with_factory(
        c.clone(),
        data.clone(),
        DatasetSplit::Training,
        Box::new(factory),
    )
    .unwrap();
    mock.run_steps(2).unwrap();
    assert!(mock.metrics.agents.values().all(|m| m.survival_days == 2));
    let mut burn = SimulationEngine::new(c, data, DatasetSplit::Training).unwrap();
    burn.run_steps(2).unwrap();
    assert!(burn.metrics.agents.values().all(|m| m.survival_days == 2));
    assert!(!include_str!("../../src/simulation.rs").contains("burn::"));
}

#[test]
fn normalized_risk_reward_uses_engine_drawdown_increments_and_turnover_fraction() {
    let mut c = config();
    c.simulation.population_size = 1;
    c.simulation.bankruptcy_threshold = 0.0;
    c.brain.policy = PolicyKind::EqualWeight;
    c.reward.kind = RewardKind::NormalizedRiskAdjustedReturn;
    c.reward.drawdown_penalty_weight = 0.5;
    c.reward.turnover_penalty_weight = 0.25;
    let data = loaded(&c);
    let mut engine = SimulationEngine::new(c, data, DatasetSplit::Training).unwrap();
    let report = engine.run().unwrap();
    let decisions = engine
        .report_traces
        .iter()
        .filter_map(|trace| match trace {
            ReportTrace::Decision(decision) => Some(decision),
            ReportTrace::PpoUpdate(_) => None,
        })
        .collect::<Vec<_>>();

    assert!(!decisions.is_empty());
    assert!(
        decisions
            .iter()
            .any(|trace| trace.reward_breakdown.turnover_basis > 0.0)
    );
    for trace in &decisions {
        let expected_turnover_basis = trace.turnover / trace.previous_net_worth;
        assert!((trace.reward_breakdown.turnover_basis - expected_turnover_basis).abs() < 1e-12);
        assert!(
            (trace.reward_breakdown.turnover_penalty - 0.25 * expected_turnover_basis).abs()
                < 1e-12
        );
        assert!(
            (trace.reward_breakdown.drawdown_penalty - 0.5 * trace.reward_breakdown.drawdown_basis)
                .abs()
                < 1e-12
        );
    }
    let drawdown_increments = decisions
        .iter()
        .map(|trace| trace.reward_breakdown.drawdown_basis)
        .sum::<f64>();
    assert!((drawdown_increments - report.agents[0].1.maximum_drawdown).abs() < 1e-12);
}

#[test]
fn continuing_bootstrap_fork_failure_is_transactional_and_not_zero() {
    let mut c = config();
    c.simulation.population_size = 1;
    c.learning.implementation = LearningKind::Ppo;
    c.learning.rollout_length = 1;
    c.learning.minibatch_size = 1;
    c.learning.sequence_length = 1;
    c.learning.optimisation_epochs = 1;
    let data = loaded(&c);
    let factory = StandardAgentFactory::with_neural_factory(Box::new(BootstrapForkFailureFactory));
    let mut engine =
        SimulationEngine::new_with_factory(c, data, DatasetSplit::Training, Box::new(factory))
            .unwrap();
    let before_agents = engine.agents().cloned().collect::<Vec<_>>();
    let before_events = engine.domain_events.clone();
    let error = engine.run_steps(1).unwrap_err();
    assert!(error.to_string().contains("next-value policy fork failed"));
    assert!(matches!(
        engine.lifecycle(),
        SimulationLifecycle::Failed { .. }
    ));
    assert_eq!(engine.report().steps, 0);
    assert_eq!(engine.agents().cloned().collect::<Vec<_>>(), before_agents);
    assert_eq!(engine.domain_events, before_events);
    let repeated = engine.run_steps(1).unwrap_err();
    assert_eq!(repeated.to_string(), error.to_string());
    assert_eq!(engine.agents().cloned().collect::<Vec<_>>(), before_agents);
    assert_eq!(engine.domain_events, before_events);
}

#[test]
fn policy_failures_emit_diagnostics_and_obey_limits() {
    let mut c = config();
    c.simulation.population_size = 1;
    c.policy_failures.maximum_consecutive = 1;
    c.policy_failures.maximum_total = 1;
    c.policy_failures.behavior = PolicyFailureBehavior::KillAgent;
    let data = loaded(&c);
    let factory = StandardAgentFactory::with_neural_factory(Box::new(MockFactory { fail: true }));
    let mut engine =
        SimulationEngine::new_with_factory(c, data, DatasetSplit::Training, Box::new(factory))
            .unwrap();
    engine.run_steps(1).unwrap();
    assert_eq!(engine.metrics.policy_failures, 1);
    assert_eq!(engine.metrics.births, 2);
    assert!(engine.domain_events.iter().any(|event| matches!(event, DomainEvent::PolicyFailure { category: midas_core::simulation::PolicyFailureCategory::NonFiniteState, message, .. } if message.contains("injected"))));
    let mut c = config();
    c.simulation.population_size = 1;
    c.policy_failures.maximum_consecutive = 1;
    c.policy_failures.maximum_total = 2;
    c.policy_failures.behavior = PolicyFailureBehavior::Continue;
    let data = loaded(&c);
    let factory = StandardAgentFactory::with_neural_factory(Box::new(MockFactory { fail: true }));
    let mut engine =
        SimulationEngine::new_with_factory(c, data, DatasetSplit::Training, Box::new(factory))
            .unwrap();
    engine.run_steps(2).unwrap();
    assert_eq!(engine.metrics.policy_failures, 2);
    assert_eq!(engine.metrics.births, 1);
}

#[test]
fn stop_on_policy_failure_preserves_original_failure_event() {
    let mut c = config();
    c.simulation.population_size = 1;
    c.policy_failures.maximum_consecutive = 1;
    c.policy_failures.maximum_total = 1;
    c.policy_failures.behavior = PolicyFailureBehavior::StopExperiment;
    let data = loaded(&c);
    let factory = StandardAgentFactory::with_neural_factory(Box::new(MockFactory { fail: true }));
    let mut engine =
        SimulationEngine::new_with_factory(c, data, DatasetSplit::Training, Box::new(factory))
            .unwrap();
    assert!(engine.run_steps(1).is_err());
    assert!(engine.domain_events.iter().any(|event| matches!(event, DomainEvent::PolicyFailure { message, .. } if message.contains("injected"))));
}

#[test]
fn recoverable_policy_failure_has_one_fallback_and_advances_normally() {
    let mut c = config();
    c.policy_failures.maximum_consecutive = 3;
    c.policy_failures.maximum_total = 20;
    c.policy_failures.behavior = PolicyFailureBehavior::Continue;
    let mut engine = sequenced_engine(
        c,
        vec![CorporateAction::Split {
            asset_id: AssetId(0),
            date: date("2020-01-03"),
            ratio: 2.0,
        }],
        SequencedFactory {
            fail_calls: vec![1],
            invalid_calls: vec![],
        },
    );
    engine.run_steps(1).unwrap();
    let failures = engine
        .domain_events
        .iter()
        .filter(|event| matches!(event, DomainEvent::PolicyFailure { .. }))
        .count();
    let fallbacks = engine
        .domain_events
        .iter()
        .filter(|event| matches!(event, DomainEvent::ActionFallback { .. }))
        .count();
    assert_eq!(failures, 1);
    assert_eq!(fallbacks, 1);
    assert!(engine.domain_events.iter().any(|event| matches!(
        event,
        DomainEvent::CorporateActionApplied { date: event_date, .. }
            if *event_date == date("2020-01-03")
    )));
    assert!(engine.domain_events.iter().any(|event| matches!(
        event,
        DomainEvent::Valuation { date: event_date, .. } if *event_date == date("2020-01-03")
    )));
    let state = engine.agents().next().unwrap();
    assert_eq!(state.total_policy_failures, 1);
    assert_eq!(state.consecutive_policy_failures, 1);
}

#[test]
fn recoverable_invalid_output_creates_penalized_chronological_rollout_entry() {
    let mut c = config();
    c.learning.implementation = LearningKind::Ppo;
    c.learning.rollout_length = 4;
    c.learning.minibatch_size = 2;
    c.learning.sequence_length = 2;
    c.policy_failures.maximum_consecutive = 3;
    c.policy_failures.maximum_total = 20;
    c.policy_failures.behavior = PolicyFailureBehavior::Continue;
    let mut engine = sequenced_engine(
        c,
        Vec::new(),
        SequencedFactory {
            fail_calls: vec![],
            invalid_calls: vec![1],
        },
    );
    let agent_id = engine.agents().next().unwrap().id;
    engine.run_steps(1).unwrap();
    let midas_core::learning::LearningStrategyState::Ppo { rollout, .. } =
        engine.agent_learning_snapshot(agent_id).unwrap()
    else {
        panic!("expected PPO learning state")
    };
    assert_eq!(rollout.len(), 1);
    let transition = &rollout.transitions()[0];
    assert!(transition.reward < 0.0);
    assert!(transition.invalid_action);
    assert_eq!(transition.action_sample, None);
    assert!(matches!(
        transition.failure_category.clone(),
        Some(midas_core::PolicyFailureCategory::InvalidOutput)
            | Some(midas_core::PolicyFailureCategory::InferenceFailure)
            | Some(midas_core::PolicyFailureCategory::NonFiniteState)
    ));
    assert!(!transition.actor_eligible);
    assert!(!transition.critic_eligible);
    assert!(transition.observation.is_none());
    assert_eq!(engine.metrics.masked_failure_transitions, 1);
    assert_eq!(engine.metrics.actor_ineligible_transitions, 1);
    assert_eq!(engine.metrics.critic_ineligible_transitions, 1);
    assert!(engine.metrics.invalid_action_penalty_total > 0.0);
}

#[test]
fn valid_action_resets_only_consecutive_policy_failures() {
    let mut c = config();
    c.policy_failures.maximum_consecutive = 5;
    c.policy_failures.maximum_total = 20;
    c.policy_failures.behavior = PolicyFailureBehavior::Continue;
    let mut engine = sequenced_engine(
        c,
        Vec::new(),
        SequencedFactory {
            fail_calls: vec![1, 3],
            invalid_calls: vec![],
        },
    );
    engine.run_steps(1).unwrap();
    let first = engine.agents().next().unwrap().clone();
    engine.run_steps(1).unwrap();
    let second = engine.agents().next().unwrap().clone();
    engine.run_steps(1).unwrap();
    let third = engine.agents().next().unwrap().clone();
    assert_eq!(
        (
            first.total_policy_failures,
            first.consecutive_policy_failures
        ),
        (1, 1)
    );
    assert_eq!(
        (
            second.total_policy_failures,
            second.consecutive_policy_failures
        ),
        (1, 0)
    );
    assert_eq!(
        (
            third.total_policy_failures,
            third.consecutive_policy_failures
        ),
        (2, 1)
    );
}

#[test]
fn invalid_policy_output_is_one_recoverable_failure() {
    let mut c = config();
    c.policy_failures.maximum_consecutive = 3;
    c.policy_failures.maximum_total = 20;
    c.policy_failures.behavior = PolicyFailureBehavior::Continue;
    let mut engine = sequenced_engine(
        c,
        Vec::new(),
        SequencedFactory {
            fail_calls: vec![],
            invalid_calls: vec![1],
        },
    );
    engine.run_steps(1).unwrap();
    let policy_events = engine
        .domain_events
        .iter()
        .filter_map(|event| match event {
            DomainEvent::PolicyFailure { category, .. } => Some(category),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(policy_events.len(), 1);
    assert!(matches!(
        policy_events[0],
        midas_core::simulation::PolicyFailureCategory::InvalidOutput
    ));
    assert_eq!(
        engine
            .domain_events
            .iter()
            .filter(|event| matches!(event, DomainEvent::ActionFallback { .. }))
            .count(),
        1
    );
    let state = engine.agents().next().unwrap();
    assert_eq!(state.total_policy_failures, 1);
    assert_eq!(state.consecutive_policy_failures, 1);
}

#[test]
fn stop_experiment_discards_the_entire_failed_step_and_is_idempotent() {
    let checkpoint = tempfile::NamedTempFile::new().unwrap();
    let mut c = config();
    c.policy_failures.maximum_consecutive = 1;
    c.policy_failures.maximum_total = 1;
    c.policy_failures.behavior = PolicyFailureBehavior::StopExperiment;
    c.checkpoint.every_steps = 1;
    c.checkpoint.path = checkpoint.path().display().to_string();
    let mut engine = sequenced_engine(
        c,
        vec![
            CorporateAction::Split {
                asset_id: AssetId(0),
                date: date("2020-01-06"),
                ratio: 2.0,
            },
            CorporateAction::CashDividend {
                asset_id: AssetId(0),
                date: date("2020-01-06"),
                amount_per_share: 1.0,
            },
        ],
        SequencedFactory {
            fail_calls: vec![2],
            invalid_calls: vec![],
        },
    );
    engine.run_steps(1).unwrap();
    let good_agents = engine.agents().cloned().collect::<Vec<_>>();
    let good_events = engine.domain_events.clone();
    let good_metrics = engine.metrics.clone();
    let good_steps = engine.report().steps;
    let checkpoint_bytes = fs::read(checkpoint.path()).unwrap();
    let error = engine.run_steps(1).unwrap_err();
    assert!(error.to_string().contains("sequenced inference failure"));
    assert!(matches!(
        engine.lifecycle(),
        SimulationLifecycle::Failed { .. }
    ));
    assert_eq!(engine.agents().cloned().collect::<Vec<_>>(), good_agents);
    assert_eq!(engine.report().steps, good_steps);
    assert_eq!(fs::read(checkpoint.path()).unwrap(), checkpoint_bytes);
    assert_eq!(engine.metrics.invalid_actions, good_metrics.invalid_actions);
    assert_eq!(engine.metrics.deaths, good_metrics.deaths);
    assert_eq!(
        engine
            .domain_events
            .iter()
            .filter(|event| matches!(event, DomainEvent::ActionFallback { .. }))
            .count(),
        0
    );
    assert!(!engine.domain_events.iter().any(|event| matches!(
        event,
        DomainEvent::CorporateActionApplied { date: event_date, .. }
            if *event_date == date("2020-01-06")
    )));
    assert!(!engine.domain_events.iter().any(|event| matches!(
        event,
        DomainEvent::Valuation { date: event_date, .. }
            if *event_date == date("2020-01-06")
    )));
    let events_after_failure = engine.domain_events.clone();
    let metrics_after_failure = engine.metrics.clone();
    let repeated = engine.run().unwrap_err();
    assert_eq!(repeated.to_string(), error.to_string());
    assert_eq!(engine.domain_events, events_after_failure);
    assert_eq!(engine.metrics, metrics_after_failure);
    let expected_reason = error
        .to_string()
        .strip_prefix("policy error: ")
        .unwrap()
        .to_string();
    assert_eq!(
        engine.report().failure_reason.as_deref(),
        Some(expected_reason.as_str())
    );
    assert_eq!(good_events.len() + 1, engine.domain_events.len());
}

#[test]
fn fatal_agent_policy_failure_dies_before_next_day_actions_without_fallback() {
    let mut c = config();
    c.policy_failures.maximum_consecutive = 1;
    c.policy_failures.maximum_total = 1;
    c.policy_failures.behavior = PolicyFailureBehavior::KillAgent;
    let mut engine = sequenced_engine(
        c,
        vec![
            CorporateAction::Split {
                asset_id: AssetId(0),
                date: date("2020-01-06"),
                ratio: 2.0,
            },
            CorporateAction::CashDividend {
                asset_id: AssetId(0),
                date: date("2020-01-06"),
                amount_per_share: 1.0,
            },
        ],
        SequencedFactory {
            fail_calls: vec![2],
            invalid_calls: vec![],
        },
    );
    engine.run_steps(1).unwrap();
    let before_failure = engine.agents().next().unwrap().clone();
    let expected_final = before_failure
        .portfolio
        .value(&[Some(320.95), Some(50.50), Some(79.80)])
        .unwrap();
    engine.run_steps(1).unwrap();
    let policy_failure_index = engine
        .domain_events
        .iter()
        .position(|event| matches!(event, DomainEvent::PolicyFailure { .. }))
        .unwrap();
    let death_index = engine
        .domain_events
        .iter()
        .position(|event| {
            matches!(
                event,
                DomainEvent::AgentDied {
                    cause: midas_core::simulation::DeathCause::PolicyFailure,
                    ..
                }
            )
        })
        .unwrap();
    assert!(policy_failure_index < death_index);
    assert_eq!(
        engine
            .domain_events
            .iter()
            .filter(|event| matches!(event, DomainEvent::ActionFallback { .. }))
            .count(),
        0
    );
    assert!(!engine.domain_events.iter().any(|event| matches!(
        event,
        DomainEvent::CorporateActionApplied {
            agent_id: AgentId(0),
            date: event_date,
            ..
        } if *event_date == date("2020-01-06")
    )));
    assert!(!engine.domain_events.iter().any(|event| matches!(
        event,
        DomainEvent::Valuation {
            agent_id: AgentId(0),
            date: event_date,
            ..
        } if *event_date == date("2020-01-06")
    )));
    let death = engine
        .domain_events
        .iter()
        .find_map(|event| match event {
            DomainEvent::AgentDied {
                agent_id: AgentId(0),
                final_net_worth,
                age,
                ..
            } => Some((*final_net_worth, *age)),
            _ => None,
        })
        .unwrap();
    assert!((death.0 - expected_final).abs() < 1e-8);
    assert_eq!(death.1, before_failure.age);
    let replacement = engine
        .agents()
        .find(|state| state.id == AgentId(1))
        .unwrap();
    assert_eq!(replacement.total_policy_failures, 0);
    assert_eq!(replacement.consecutive_policy_failures, 0);
}

#[test]
fn stop_experiment_failure_replays_identically_from_last_good_checkpoint() {
    let file = tempfile::NamedTempFile::new().unwrap();
    let mut c = config();
    c.policy_failures.maximum_consecutive = 1;
    c.policy_failures.maximum_total = 1;
    c.policy_failures.behavior = PolicyFailureBehavior::StopExperiment;
    let data = data_with_actions(&mut c, Vec::new());
    let factory = || {
        Box::new(StandardAgentFactory::with_neural_factory(Box::new(
            SequencedFactory {
                fail_calls: vec![2],
                invalid_calls: vec![],
            },
        ))) as Box<dyn AgentFactory>
    };
    let mut first = SimulationEngine::new_with_factory(
        c.clone(),
        data.clone(),
        DatasetSplit::Training,
        factory(),
    )
    .unwrap();
    first.run_steps(1).unwrap();
    first.save_checkpoint(file.path()).unwrap();
    let first_error = first.run_steps(1).unwrap_err().to_string();
    let first_events = first.domain_events.clone();
    let first_states = first.agents().cloned().collect::<Vec<_>>();
    let mut resumed =
        SimulationEngine::resume_with_factory(c, data, file.path(), factory()).unwrap();
    let resumed_error = resumed.run_steps(1).unwrap_err().to_string();
    assert_eq!(first_error, resumed_error);
    assert_eq!(first.lifecycle(), resumed.lifecycle());
    assert_eq!(first.domain_events, resumed.domain_events);
    assert_eq!(first_events, resumed.domain_events);
    assert_eq!(first_states, resumed.agents().cloned().collect::<Vec<_>>());
    assert_eq!(first.report().steps, resumed.report().steps);
}

#[test]
fn engine_corporate_actions_apply_once_across_resume_and_repeat() {
    let checkpoint = tempfile::NamedTempFile::new().unwrap();
    let mut c = config();
    c.brain.policy = PolicyKind::BuyAndHold;
    c.simulation.population_size = 1;
    let data = data_with_actions(
        &mut c,
        vec![
            CorporateAction::Split {
                asset_id: AssetId(0),
                date: date("2020-01-06"),
                ratio: 2.0,
            },
            CorporateAction::CashDividend {
                asset_id: AssetId(0),
                date: date("2020-01-06"),
                amount_per_share: 1.0,
            },
        ],
    );
    let mut engine =
        SimulationEngine::new(c.clone(), data.clone(), DatasetSplit::Training).unwrap();
    engine.run_steps(1).unwrap();
    let before_event = engine.agents().next().unwrap().clone();
    engine.run_steps(1).unwrap();
    let after_event = engine.agents().next().unwrap().clone();
    assert!(after_event.portfolio.positions[0].quantity > 0.0);
    assert!(
        (after_event.portfolio.positions[0].quantity
            - 2.0 * before_event.portfolio.positions[0].quantity)
            .abs()
            < 1e-8
    );
    assert!(
        (after_event.portfolio.cash
            - before_event.portfolio.cash
            - 2.0 * before_event.portfolio.positions[0].quantity
            + 2.0)
            .abs()
            < 1e-8
    );
    assert_eq!(
        engine
            .domain_events
            .iter()
            .filter(|event| matches!(event, DomainEvent::CorporateActionApplied { .. }))
            .count(),
        2
    );
    engine.save_checkpoint(checkpoint.path()).unwrap();
    let mut resumed = SimulationEngine::resume(c, data, checkpoint.path()).unwrap();
    resumed.run().unwrap();
    let completed = resumed.agents().next().unwrap().portfolio.clone();
    resumed.run().unwrap();
    assert_eq!(resumed.agents().next().unwrap().portfolio, completed);
    assert_eq!(
        resumed
            .domain_events
            .iter()
            .filter(|event| matches!(event, DomainEvent::CorporateActionApplied { .. }))
            .count(),
        2
    );
}

#[test]
fn neural_memory_resets_and_genome_validates_full_schema() {
    let architecture = BrainArchitecture {
        version: 1,
        observation_schema_hash: "one".into(),
        observation_size: 2,
        asset_count: 1,
        hidden_layers: vec![2],
        memory: midas_core::brain::MemoryArchitecture { size: 2 },
        actor_output_size: 2,
        critic_output_size: 1,
        activation: midas_core::config::Activation::Relu,
        action_distribution: midas_core::config::ActionDistributionConfig::default(),
    };
    let c = config();
    let genome = BrainGenome::random(&architecture, &c.brain.traits, 1).unwrap();
    let mut changed = architecture.clone();
    changed.observation_schema_hash = "two".into();
    assert!(genome.validate(&changed).is_err());
    let mut invalid_traits = genome.clone();
    invalid_traits.traits.maximum_allocation = 2.0;
    assert!(
        invalid_traits
            .validate_with_traits(&architecture, &c.brain.traits)
            .is_err()
    );
    let mut model = MockModel {
        schema: "one".into(),
        size: 2,
        memory: 5.0,
        fail: false,
    };
    model.reset_memory();
    let NeuralModelState::External { state, .. } = model.snapshot_state().unwrap() else {
        panic!()
    };
    assert_eq!(state[0], 0.0);
}
