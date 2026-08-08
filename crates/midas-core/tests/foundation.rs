use chrono::NaiveDate;
use midas_core::{
    AgentId, AssetId, MidasError,
    action::{
        ActionValidationContext, ActionValidator, LongOnlyActionValidator, ProposedAction,
        ProposedAllocation, ProposedTradeImpulses,
    },
    agent::{
        AgentBuildContext, AgentFactory, AgentSpawnRequest, SpawnMetadata, StandardAgentFactory,
    },
    brain::{
        BrainArchitecture, BrainGenome, BurnNeuralModelFactory, ComputeDevice, NeuralInput,
        NeuralModel, NeuralModelFactory, NeuralModelState, NeuralOutput, PolicyBlueprint,
    },
    config::{
        Config, CorporateActionConfig, CorporateActionMode, DeathHandling, EvolutionFallback,
        LearningKind, ObservationFeatureKind, PolicyFailureBehavior, PolicyKind,
    },
    evolution::FitnessScore,
    execution::{
        CostExecutionEngine, ExecutionApplier, ExecutionPlanner, ExecutionSnapshot, Portfolio,
        TradeSide,
    },
    market::{
        BackwardFeaturePipeline, CorporateAction, FeaturePipeline, FeaturePriceSeries, MarketBar,
        MarketDataSource, MarketDataset, PointInTimeFeaturePriceSeries, PolarsDataSource,
        write_parquet,
    },
    observation::{
        Observation, ObservationBuilder, ObservationContext, ObservationSchema,
        SchemaObservationBuilder,
    },
    replacement::{
        LivingAgentSummary, ReplacementContext, ReplacementStrategy, ReplacementStrategyFactory,
        ReplacementStrategyState,
    },
    simulation::{
        CHECKPOINT_VERSION, DatasetSplit, DomainEvent, SimulationEngine, SimulationLifecycle,
    },
};
use proptest::prelude::*;
use std::{fs, path::PathBuf};

fn config() -> Config {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut c = Config::load(root.join("configs/test.toml")).unwrap();
    c.data.path = root.join("data/raw/synthetic.csv").to_string_lossy().into();
    c
}
fn date(s: &str) -> NaiveDate {
    NaiveDate::parse_from_str(s, "%Y-%m-%d").unwrap()
}
fn bar(asset: u64, day: &str, open: f64) -> MarketBar {
    MarketBar {
        date: date(day),
        asset_id: AssetId(asset),
        open,
        high: open + 1.0,
        low: open - 1.0,
        close: open,
        adjusted_close: open,
        volume: 1000.0,
        tradable: true,
    }
}
fn loaded(c: &Config) -> MarketDataset {
    PolarsDataSource.load(c).unwrap()
}
fn validator(
    weights: Vec<f64>,
    tradable: &[bool],
    max: f64,
) -> midas_core::action::ValidatedAllocation {
    LongOnlyActionValidator
        .validate(
            ProposedAllocation::rebalance(weights),
            &ActionValidationContext {
                asset_count: tradable.len(),
                tradable,
                policy_maximum_asset_allocation: Some(max),
                policy_maximum_trade_fraction: None,
                market_maximum_asset_allocation: None,
            },
        )
        .unwrap()
}

// Keep the foundation regressions in one integration-test crate so shared
// fixtures compile once, while each behavioral area remains independently
// navigable. Child modules intentionally import this small shared prelude.
#[path = "foundation/action_execution.rs"]
mod action_execution;
#[path = "foundation/checkpoints.rs"]
mod checkpoints;
#[path = "foundation/engine_behaviors.rs"]
mod engine_behaviors;
#[path = "foundation/market_actions.rs"]
mod market_actions;
#[path = "foundation/observation_market.rs"]
mod observation_market;
#[path = "foundation/observation_schema.rs"]
mod observation_schema;
#[path = "foundation/policies_accounting.rs"]
mod policies_accounting;
#[path = "foundation/replacement_evolution.rs"]
mod replacement_evolution;
