use midas_core::{
    config::{
        Config, ObservationNormalizationConfig, ObservationNormalizationKind,
        RollingMarketStatsConfig,
    },
    market::{MarketDataSource, PolarsDataSource},
    simulation::{DatasetSplit, SimulationEngine},
};
use std::{collections::BTreeMap, fs, path::PathBuf};

const CONFIG_CASES: [(&str, &str, &[u8], &str); 4] = [
    (
        "default",
        "configs/default.toml",
        include_bytes!("fixtures/config/default.json"),
        "29cef930c9b07c77bbce136e44e3f0af3f0a5744a5ce519f520ce0afd63b6d28",
    ),
    (
        "combined-ppo",
        "configs/prompt6/combined-ppo.toml",
        include_bytes!("fixtures/config/combined-ppo.json"),
        "d72abfff021f7f37c1659e626a82c9f05fe937f5418e31994e28cccf55c9e1bd",
    ),
    (
        "combined-ppo-evolution",
        "configs/prompt6/combined-ppo-evolution.toml",
        include_bytes!("fixtures/config/combined-ppo-evolution.json"),
        "c309aa6984f8393af45060c9857a45012ec82955738b8df3fb7b23638e13a5c0",
    ),
    (
        "test",
        "configs/test.toml",
        include_bytes!("fixtures/config/test.json"),
        "fe428dbba4a0cf174433f504226d74787882309ca1b4439c7feb88cf8b297bde",
    ),
];

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn load(relative: &str) -> Config {
    Config::load(root().join(relative)).unwrap()
}

fn load_data(config: &Config) -> midas_core::market::MarketDataset {
    let mut data_config = config.clone();
    data_config.data.path = root()
        .join(&data_config.data.path)
        .to_string_lossy()
        .into_owned();
    if let Some(path) = &mut data_config.data.corporate_actions_path {
        *path = root().join(&*path).to_string_lossy().into_owned();
    }
    PolarsDataSource.load(&data_config).unwrap()
}

fn remove_table(source: &str, name: &str) -> String {
    let header = format!("[{name}]");
    let mut removing = false;
    source
        .lines()
        .filter(|line| {
            if line.trim() == header {
                removing = true;
                return false;
            }
            if removing && line.trim_start().starts_with('[') {
                removing = false;
            }
            !removing
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn representative_config_json_and_hash_snapshots_are_exact() {
    for (name, path, expected_json, expected_hash) in CONFIG_CASES {
        let config = load(path);
        let actual_json = serde_json::to_vec(&config).unwrap();
        assert_eq!(
            actual_json, expected_json,
            "JSON snapshot changed for {name}"
        );
        assert_eq!(
            config.hash().unwrap(),
            expected_hash,
            "hash changed for {name}"
        );
        assert_eq!(
            blake3::hash(&actual_json).to_hex().as_str(),
            expected_hash,
            "Config::hash no longer matches the serialized bytes for {name}"
        );
    }
}

#[test]
fn serialization_preserves_field_map_and_option_ordering() {
    let mut config = load("configs/default.toml");
    let rule = || ObservationNormalizationConfig {
        kind: ObservationNormalizationKind::None,
        minimum: None,
        maximum: None,
        mean: None,
        standard_deviation: None,
        scale: None,
        offset: None,
    };
    config.observations.normalizations =
        BTreeMap::from([("zeta".to_owned(), rule()), ("alpha".to_owned(), rule())]);
    let json = serde_json::to_string(&config).unwrap();

    let ordered_root_fields = [
        "\"market\"",
        "\"data\"",
        "\"splits\"",
        "\"simulation\"",
        "\"execution\"",
        "\"features\"",
        "\"observation\"",
        "\"observations\"",
        "\"corporate_actions\"",
        "\"policy_failures\"",
        "\"brain\"",
        "\"learning\"",
        "\"backend\"",
        "\"replacement\"",
        "\"checkpoint\"",
        "\"reward\"",
        "\"evolution\"",
    ];
    let positions = ordered_root_fields.map(|field| json.find(field).unwrap());
    assert!(positions.windows(2).all(|pair| pair[0] < pair[1]));
    assert!(json.find("\"alpha\"").unwrap() < json.find("\"zeta\"").unwrap());
    assert!(json.contains("\"corporate_actions_path\":null"));
}

#[test]
fn omitted_root_sections_expand_to_the_exact_snapshot() {
    let mut source = fs::read_to_string(root().join("configs/default.toml")).unwrap();
    for table in [
        "observation",
        "observations",
        "corporate_actions",
        "policy_failures",
        "checkpoint",
    ] {
        source = remove_table(&source, table);
    }
    let config: Config = toml::from_str(&source).unwrap();
    config.validate().unwrap();
    assert_eq!(
        serde_json::to_vec(&config).unwrap(),
        include_bytes!("fixtures/config/omitted-root-defaults.json")
    );

    let value = serde_json::to_value(config).unwrap();
    assert_eq!(value["observation"]["version"], 1);
    assert_eq!(value["reward"]["kind"], "log_return");
    assert_eq!(value["evolution"]["enabled"], false);
    assert_eq!(value["checkpoint"]["every_steps"], 0);
    assert_eq!(value["corporate_actions"]["mode"], "reject");
    assert_eq!(value["policy_failures"]["behavior"], "kill_agent");
}

#[test]
fn learning_and_replacement_sections_remain_required() {
    let mut value: serde_json::Value =
        serde_json::from_slice(include_bytes!("fixtures/config/default.json")).unwrap();
    value.as_object_mut().unwrap().remove("learning");
    let error = serde_json::from_value::<Config>(value).unwrap_err();
    assert_eq!(error.to_string(), "missing field `learning`");

    let mut value: serde_json::Value =
        serde_json::from_slice(include_bytes!("fixtures/config/default.json")).unwrap();
    value.as_object_mut().unwrap().remove("replacement");
    let error = serde_json::from_value::<Config>(value).unwrap_err();
    assert_eq!(error.to_string(), "missing field `replacement`");
}

#[test]
fn validation_returns_the_first_error_with_exact_text() {
    let mut config = load("configs/default.toml");
    config.market.assets.push(config.market.assets[0].clone());
    config.learning.learning_rate = 0.0;
    assert_eq!(
        config.validate().unwrap_err().to_string(),
        "configuration error: market.assets contains duplicates"
    );

    let mut config = load("configs/default.toml");
    config.learning.ppo_clip = -1.0;
    config.reward.clip_min = config.reward.clip_max;
    assert_eq!(
        config.validate().unwrap_err().to_string(),
        "configuration error: learning configuration values are invalid"
    );

    let mut config = load("configs/default.toml");
    config.observations.providers = vec!["market".into(), "market".into()];
    config.backend.device = "unsupported".into();
    assert_eq!(
        config.validate().unwrap_err().to_string(),
        "configuration error: observation registry providers must be unique non-empty names"
    );

    let mut config = load("configs/default.toml");
    config.splits.training.start = config.splits.training.end.succ_opt().unwrap();
    config.evolution.mutation.weight_probability = 2.0;
    assert_eq!(
        config.validate().unwrap_err().to_string(),
        "configuration error: split start must not follow split end"
    );

    let mut config = load("configs/default.toml");
    config.execution.spread_rate = 2.0;
    config.execution.slippage_rate = 1.0;
    assert_eq!(
        config.validate().unwrap_err().to_string(),
        "configuration error: spread_rate must be less than 2"
    );
}

#[test]
fn backend_device_compatibility_preserves_identity_and_error_text() {
    let mut config = load("configs/default.toml");
    assert_eq!(
        config.backend.compute_device().unwrap().identity(),
        "cpu_ndarray:default"
    );

    config.backend.device.clear();
    assert_eq!(
        config.backend.compute_device().unwrap_err().to_string(),
        "configuration error: backend.device must not be empty"
    );

    config.backend.device = "unsupported".into();
    assert_eq!(
        config.backend.compute_device().unwrap_err().to_string(),
        "configuration error: unsupported CPU NdArray device 'unsupported'; only 'default' is implemented"
    );
}

#[test]
fn default_rolling_config_is_omitted_from_serialization() {
    let mut config = load("configs/default.toml");
    config.observations.rolling_market_stats = RollingMarketStatsConfig::default();
    let json = serde_json::to_string(&config).unwrap();
    assert!(!json.contains("rolling_market_stats"));
    assert_eq!(
        config.hash().unwrap(),
        "29cef930c9b07c77bbce136e44e3f0af3f0a5744a5ce519f520ce0afd63b6d28"
    );
}

#[test]
fn nondefault_rolling_config_is_serialized_exactly() {
    let config = load("configs/prompt6/combined-ppo.toml");
    let json = serde_json::to_vec(&config).unwrap();
    assert_eq!(json, include_bytes!("fixtures/config/combined-ppo.json"));
    let value: serde_json::Value = serde_json::from_slice(&json).unwrap();
    assert_eq!(
        value["observations"]["rolling_market_stats"]["enabled"],
        true
    );
    assert_eq!(
        value["observations"]["rolling_market_stats"]["return_windows"],
        serde_json::json!([5, 20, 60, 120])
    );
}

#[test]
fn version_12_checkpoint_resume_migrates_to_version_13_without_report_drift() {
    let checkpoint = root().join("checkpoints/prompt6/default-control-seed-41.bin");
    let before = fs::read(&checkpoint).unwrap();
    assert_eq!(before.len(), 2_431_176);
    assert_eq!(
        blake3::hash(&before).to_hex().as_str(),
        "3e6536d20758751ed8123a3c111ed9a554ed87daf6cb2b86f3739b5b36388b33"
    );

    let config = load("configs/prompt6/runs/default-control-seed-41.toml");
    let data = load_data(&config);
    let mut engine = SimulationEngine::resume(config, data, &checkpoint).unwrap();
    let report = engine.run().unwrap();
    assert_eq!(
        serde_json::to_vec(&report).unwrap(),
        include_bytes!("fixtures/config/checkpoint-resume-report.json")
    );

    let saved = tempfile::NamedTempFile::new().unwrap();
    engine.save_checkpoint(saved.path()).unwrap();
    let migrated = fs::read(saved.path()).unwrap();
    assert_ne!(migrated, before);
    assert_eq!(u32::from_le_bytes(migrated[..4].try_into().unwrap()), 13);
    assert_eq!(fs::read(checkpoint).unwrap(), before);

    let config = load("configs/prompt6/runs/default-control-seed-41.toml");
    let data = load_data(&config);
    let mut restored = SimulationEngine::resume(config, data, saved.path()).unwrap();
    assert_eq!(
        serde_json::to_vec(&restored.run().unwrap()).unwrap(),
        serde_json::to_vec(&report).unwrap()
    );
}

#[test]
fn version_12_checkpoint_evaluation_is_byte_and_report_compatible() {
    let checkpoint = root().join("checkpoints/prompt6/default-control-seed-41.bin");
    let before = fs::read(&checkpoint).unwrap();
    let config = load("configs/prompt6/runs/default-control-seed-41.toml");
    let data = load_data(&config);
    let mut engine =
        SimulationEngine::evaluate_checkpoint(config, data, &checkpoint, DatasetSplit::Validation)
            .unwrap();
    let report = engine.run().unwrap();
    assert_eq!(
        serde_json::to_vec(&report).unwrap(),
        include_bytes!("fixtures/config/checkpoint-evaluation-report.json")
    );
    assert_eq!(fs::read(checkpoint).unwrap(), before);
}
