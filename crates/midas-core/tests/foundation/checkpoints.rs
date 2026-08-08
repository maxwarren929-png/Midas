use super::*;

#[test]
fn checkpoint_domain_history_and_completion_are_deterministic() {
    let c = config();
    let data = loaded(&c);
    let mut full = SimulationEngine::new(c.clone(), data.clone(), DatasetSplit::Training).unwrap();
    full.run().unwrap();
    let mut partial =
        SimulationEngine::new(c.clone(), data.clone(), DatasetSplit::Training).unwrap();
    partial.run_steps(3).unwrap();
    let file = tempfile::NamedTempFile::new().unwrap();
    partial.save_checkpoint(file.path()).unwrap();
    assert_eq!(partial.operational_events.len(), 1);
    let mut resumed = SimulationEngine::resume(c.clone(), data.clone(), file.path()).unwrap();
    resumed.run().unwrap();
    assert_eq!(full.domain_events, resumed.domain_events);
    assert_eq!(
        full.agents().collect::<Vec<_>>(),
        resumed.agents().collect::<Vec<_>>()
    );
    assert_eq!(
        full.agent_blueprints().collect::<Vec<_>>(),
        resumed.agent_blueprints().collect::<Vec<_>>()
    );
    let count = resumed.domain_events.len();
    resumed.run().unwrap();
    assert_eq!(count, resumed.domain_events.len());
    let completed = tempfile::NamedTempFile::new().unwrap();
    resumed.save_checkpoint(completed.path()).unwrap();
    let mut restored = SimulationEngine::resume(c, data, completed.path()).unwrap();
    assert_eq!(restored.lifecycle(), SimulationLifecycle::Completed);
    let count = restored.domain_events.len();
    restored.run().unwrap();
    assert_eq!(count, restored.domain_events.len());
    let full_file = tempfile::NamedTempFile::new().unwrap();
    let resumed_file = tempfile::NamedTempFile::new().unwrap();
    full.save_checkpoint(full_file.path()).unwrap();
    resumed.save_checkpoint(resumed_file.path()).unwrap();
    assert_eq!(
        std::fs::read(full_file.path()).unwrap(),
        std::fs::read(resumed_file.path()).unwrap()
    );
}

#[test]
fn checkpoint_rejects_observation_schema_change() {
    let c = config();
    let data = loaded(&c);
    let mut engine =
        SimulationEngine::new(c.clone(), data.clone(), DatasetSplit::Training).unwrap();
    engine.run_steps(1).unwrap();
    let file = tempfile::NamedTempFile::new().unwrap();
    engine.save_checkpoint(file.path()).unwrap();
    let mut changed = c;
    changed.observation.features.reverse();
    let error = match SimulationEngine::resume(changed, data, file.path()) {
        Ok(_) => panic!("schema mismatch accepted"),
        Err(e) => e,
    };
    assert!(error.to_string().contains("observation schema"));
}

#[test]
fn checkpoint_rejects_opt_in_rolling_observation_schema() {
    let c = config();
    let data = loaded(&c);
    let mut engine =
        SimulationEngine::new(c.clone(), data.clone(), DatasetSplit::Training).unwrap();
    engine.run_steps(1).unwrap();
    let file = tempfile::NamedTempFile::new().unwrap();
    engine.save_checkpoint(file.path()).unwrap();

    let mut changed = c;
    changed.observation.version += 1;
    changed
        .observations
        .providers
        .push("rolling_market_stats".into());
    changed.observations.rolling_market_stats.enabled = true;
    changed.observations.rolling_market_stats.return_windows = vec![5];
    changed
        .observations
        .rolling_market_stats
        .volatility_windows
        .clear();
    changed
        .observations
        .rolling_market_stats
        .sma_windows
        .clear();
    changed
        .observations
        .rolling_market_stats
        .relative_strength_windows
        .clear();
    changed.observations.rolling_market_stats.market_summary = false;
    let error = match SimulationEngine::resume(changed, data, file.path()) {
        Ok(_) => panic!("rolling observation schema mismatch accepted"),
        Err(e) => e,
    };
    assert!(error.to_string().contains("observation schema"));
}

#[test]
fn checkpoint_rejects_brain_architecture_change() {
    let c = config();
    let data = loaded(&c);
    let mut engine =
        SimulationEngine::new(c.clone(), data.clone(), DatasetSplit::Training).unwrap();
    engine.run_steps(1).unwrap();
    let file = tempfile::NamedTempFile::new().unwrap();
    engine.save_checkpoint(file.path()).unwrap();
    let mut changed = c;
    changed.brain.hidden_layers = vec![8, 8];
    let error = match SimulationEngine::resume(changed, data, file.path()) {
        Ok(_) => panic!("architecture mismatch accepted"),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains("configuration") || error.to_string().contains("architecture")
    );
}

#[test]
fn checkpoint_rejects_corporate_action_data_change() {
    let mut c = config();
    c.corporate_actions.mode = CorporateActionMode::ExplicitEvents;
    let source = loaded(&c);
    let action = CorporateAction::CashDividend {
        asset_id: AssetId(0),
        date: date("2020-01-03"),
        amount_per_share: 0.25,
    };
    let with_action = MarketDataset::new_with_actions(
        source.assets.clone(),
        source.bars.clone(),
        vec![action],
        &c.corporate_actions,
    )
    .unwrap();
    let mut engine = SimulationEngine::new(c.clone(), with_action, DatasetSplit::Training).unwrap();
    engine.run_steps(1).unwrap();
    let file = tempfile::NamedTempFile::new().unwrap();
    engine.save_checkpoint(file.path()).unwrap();
    let without_action = MarketDataset::new_with_actions(
        source.assets,
        source.bars,
        Vec::new(),
        &c.corporate_actions,
    )
    .unwrap();
    let error = match SimulationEngine::resume(c, without_action, file.path()) {
        Ok(_) => panic!("corporate-action data mismatch accepted"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("dataset identity"));
}

#[test]
fn backend_identity_is_configuration_driven_and_checkpoint_checked() {
    let mut c = config();
    c.backend.device = "default".into();
    assert_eq!(
        c.backend.compute_device().unwrap().identity(),
        "cpu_ndarray:default"
    );
    let data = loaded(&c);
    let mut engine =
        SimulationEngine::new(c.clone(), data.clone(), DatasetSplit::Training).unwrap();
    engine.run_steps(1).unwrap();
    let file = tempfile::NamedTempFile::new().unwrap();
    engine.save_checkpoint(file.path()).unwrap();
    let mut changed = c;
    changed.backend.device = "other-device".into();
    let error = match SimulationEngine::resume(changed, data, file.path()) {
        Ok(_) => panic!("backend mismatch accepted"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("backend") || error.to_string().contains("device"));
    let mut unsupported = config();
    unsupported.backend.device = "unsupported".into();
    let error = unsupported.validate().unwrap_err();
    assert!(error.to_string().contains("unsupported CPU NdArray device"));
}

#[test]
fn checkpoint_v6_is_rejected_as_bias_only_optimizer_layout() {
    let c = config();
    let data = loaded(&c);
    let mut engine =
        SimulationEngine::new(c.clone(), data.clone(), DatasetSplit::Training).unwrap();
    engine.run_steps(1).unwrap();
    let file = tempfile::NamedTempFile::new().unwrap();
    engine.save_checkpoint(file.path()).unwrap();
    let mut bytes = fs::read(file.path()).unwrap();
    assert_eq!(bytes.first().copied(), Some(CHECKPOINT_VERSION as u8));
    bytes[0] = 6;
    fs::write(file.path(), bytes).unwrap();
    let error = match SimulationEngine::resume(c, data, file.path()) {
        Ok(_) => panic!("checkpoint 6 was accepted"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("bias-only optimizer"));
}

#[test]
fn checkpoint_v8_is_rejected_as_single_asset_action_layout() {
    let c = config();
    let data = loaded(&c);
    let mut engine =
        SimulationEngine::new(c.clone(), data.clone(), DatasetSplit::Training).unwrap();
    engine.run_steps(1).unwrap();
    let file = tempfile::NamedTempFile::new().unwrap();
    engine.save_checkpoint(file.path()).unwrap();
    let mut bytes = fs::read(file.path()).unwrap();
    assert_eq!(bytes.first().copied(), Some(CHECKPOINT_VERSION as u8));
    bytes[0] = 8;
    fs::write(file.path(), bytes).unwrap();
    let error = match SimulationEngine::resume(c, data, file.path()) {
        Ok(_) => panic!("checkpoint 8 was accepted"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("single-asset categorical"));
}

#[test]
fn checkpoint_v10_is_rejected_as_pre_impulse_metrics_layout() {
    let c = config();
    let data = loaded(&c);
    let mut engine =
        SimulationEngine::new(c.clone(), data.clone(), DatasetSplit::Training).unwrap();
    engine.run_steps(1).unwrap();
    let file = tempfile::NamedTempFile::new().unwrap();
    engine.save_checkpoint(file.path()).unwrap();
    let mut bytes = fs::read(file.path()).unwrap();
    assert_eq!(bytes.first().copied(), Some(CHECKPOINT_VERSION as u8));
    bytes[0] = 10;
    fs::write(file.path(), bytes).unwrap();
    let error = match SimulationEngine::resume(c, data, file.path()) {
        Ok(_) => panic!("checkpoint 10 was accepted"),
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains(&format!("incompatible with {CHECKPOINT_VERSION}"))
    );
}

#[test]
fn checkpoint_v11_is_rejected_as_pre_registry_observation_layout() {
    let c = config();
    let data = loaded(&c);
    let mut engine =
        SimulationEngine::new(c.clone(), data.clone(), DatasetSplit::Training).unwrap();
    engine.run_steps(1).unwrap();
    let file = tempfile::NamedTempFile::new().unwrap();
    engine.save_checkpoint(file.path()).unwrap();
    let mut bytes = fs::read(file.path()).unwrap();
    assert_eq!(bytes.first().copied(), Some(CHECKPOINT_VERSION as u8));
    bytes[0] = 11;
    fs::write(file.path(), bytes).unwrap();
    let error = match SimulationEngine::resume(c, data, file.path()) {
        Ok(_) => panic!("checkpoint 11 was accepted"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("modular observation registry"));
}

#[test]
fn evaluation_does_not_overwrite_training_checkpoint() {
    let mut c = config();
    c.learning.implementation = LearningKind::NoLearning;
    c.evolution.enabled = false;
    let data = loaded(&c);
    let checkpoint = tempfile::NamedTempFile::new().unwrap();
    let mut training =
        SimulationEngine::new(c.clone(), data.clone(), DatasetSplit::Training).unwrap();
    training.run().unwrap();
    training.save_checkpoint(checkpoint.path()).unwrap();
    let before = fs::read(checkpoint.path()).unwrap();

    let mut evaluation =
        SimulationEngine::evaluate_checkpoint(c, data, checkpoint.path(), DatasetSplit::Validation)
            .unwrap();
    evaluation.run().unwrap();
    assert_eq!(before, fs::read(checkpoint.path()).unwrap());
}
