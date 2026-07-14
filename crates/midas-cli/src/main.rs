use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use midas_core::{
    config::{Config, LearningKind, PolicyKind},
    market::{MarketDataSource, PolarsDataSource},
    observation::{ObservationBuilder, ObservationSchema, SchemaObservationBuilder},
    simulation::{DatasetSplit, SimulationEngine},
};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "midas",
    version,
    about = "Offline artificial-life market simulator"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    ValidateConfig {
        #[arg(long)]
        config: PathBuf,
    },
    InspectData {
        #[arg(long)]
        config: PathBuf,
    },
    Simulate {
        #[arg(long)]
        config: PathBuf,
        #[arg(long, value_enum)]
        split: SplitArg,
        #[arg(long)]
        checkpoint: Option<PathBuf>,
        /// Run at most this many steps before optionally saving a checkpoint.
        /// Omitting it preserves the normal run-to-completion behavior.
        #[arg(long)]
        steps: Option<usize>,
    },
    Resume {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        checkpoint: PathBuf,
    },
    Evaluate {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long, value_enum)]
        split: SplitArg,
    },
    Benchmark {
        #[arg(long)]
        config: PathBuf,
        #[arg(long, value_enum, default_value = "test")]
        split: SplitArg,
    },
    EvolutionReport {
        #[arg(long)]
        config: PathBuf,
        #[arg(long, value_enum, default_value = "test")]
        split: SplitArg,
    },
    Observations {
        #[command(subcommand)]
        command: ObservationCommand,
    },
}

#[derive(Subcommand)]
enum ObservationCommand {
    /// List named observations published by the configured providers.
    List {
        #[arg(long)]
        config: PathBuf,
    },
    /// Show the authoritative neural-input index mapping.
    Schema {
        #[arg(long)]
        config: PathBuf,
    },
    /// Describe one named observation.
    Describe {
        #[arg(long)]
        config: PathBuf,
        name: String,
    },
    /// Validate provider registration, schema compilation, and normalisation.
    Validate {
        #[arg(long)]
        config: PathBuf,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum SplitArg {
    Training,
    Validation,
    Test,
}
impl From<SplitArg> for DatasetSplit {
    fn from(value: SplitArg) -> Self {
        match value {
            SplitArg::Training => Self::Training,
            SplitArg::Validation => Self::Validation,
            SplitArg::Test => Self::Test,
        }
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let cli = Cli::parse();
    match cli.command {
        Command::ValidateConfig { config } => {
            let c = Config::load(&config)?;
            println!("valid configuration {}", c.hash()?);
        }
        Command::InspectData { config } => {
            let c = Config::load(&config)?;
            let d = PolarsDataSource.load(&c)?;
            println!("{}", serde_json::to_string_pretty(&d.inspect())?);
        }
        Command::Simulate {
            config,
            split,
            checkpoint,
            steps,
        } => {
            let c = Config::load(&config)?;
            let d = PolarsDataSource.load(&c)?;
            let mut engine = SimulationEngine::new(c, d, split.into())?;
            let report = if let Some(steps) = steps {
                engine.run_steps(steps)?;
                engine.report()
            } else {
                engine.run()?
            };
            if let Some(path) = checkpoint {
                engine.save_checkpoint(path)?
            }
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::Resume { config, checkpoint } => {
            let c = Config::load(&config)?;
            let d = PolarsDataSource.load(&c)?;
            let mut engine = SimulationEngine::resume(c, d, &checkpoint)
                .with_context(|| format!("cannot resume {}", checkpoint.display()))?;
            println!("{}", serde_json::to_string_pretty(&engine.run()?)?);
        }
        Command::Evaluate {
            config,
            checkpoint,
            split,
        } => {
            let c = Config::load(&config)?;
            let d = PolarsDataSource.load(&c)?;
            let mut engine = SimulationEngine::evaluate_checkpoint(c, d, &checkpoint, split.into())
                .with_context(|| format!("cannot evaluate {}", checkpoint.display()))?;
            println!("{}", serde_json::to_string_pretty(&engine.run()?)?);
        }
        Command::Benchmark { config, split } => {
            let base = Config::load(&config)?;
            let data = PolarsDataSource.load(&base)?;
            for policy in [
                PolicyKind::Cash,
                PolicyKind::Random,
                PolicyKind::EqualWeight,
                PolicyKind::BuyAndHold,
            ] {
                let mut c = base.clone();
                c.brain.policy = policy;
                c.learning.implementation = LearningKind::NoLearning;
                c.evolution.enabled = false;
                let name = format!("{policy:?}");
                let mut engine = SimulationEngine::new(c, data.clone(), split.into())?;
                let report = engine.run()?;
                let ending = report
                    .agents
                    .first()
                    .map(|x| x.1.ending_net_worth)
                    .unwrap_or_default();
                println!("{name}: ending_net_worth={ending:.4}");
            }
        }
        Command::EvolutionReport { config, split } => {
            let c = Config::load(&config)?;
            let d = PolarsDataSource.load(&c)?;
            let mut engine = SimulationEngine::new(c, d, split.into())?;
            let report = engine.run()?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "simulation": report,
                    "evolution": engine.evolution_state(),
                }))?
            );
        }
        Command::Observations { command } => match command {
            ObservationCommand::List { config } => {
                let (_, schema, builder) = observation_builder(&config)?;
                let metadata = builder.registry().metadata();
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "schema_version": schema.version,
                        "providers": schema.providers,
                        "observations": metadata,
                    }))?
                );
            }
            ObservationCommand::Schema { config } => {
                let (_, schema, _) = observation_builder(&config)?;
                println!("{}", serde_json::to_string_pretty(&schema)?);
            }
            ObservationCommand::Describe { config, name } => {
                let (_, _, builder) = observation_builder(&config)?;
                let metadata = builder.registry().metadata_by_name(&name).ok_or_else(|| {
                    anyhow::anyhow!("configured observation {name} does not exist")
                })?;
                println!("{}", serde_json::to_string_pretty(metadata)?);
            }
            ObservationCommand::Validate { config } => {
                let (asset_count, schema, builder) = observation_builder(&config)?;
                println!(
                    "valid observation schema version={} dimension={} hash={}",
                    schema.version,
                    builder.schema().dimension(asset_count),
                    schema.identity(asset_count)?
                );
            }
        },
    }
    Ok(())
}

fn observation_builder(
    path: &PathBuf,
) -> Result<(usize, ObservationSchema, SchemaObservationBuilder)> {
    let config = Config::load(path)?;
    let asset_count = config.market.assets.len();
    let schema = ObservationSchema::from_config(&config);
    let builder = SchemaObservationBuilder::new(
        schema.clone(),
        config.features.features.clone(),
        asset_count,
    )?;
    Ok((asset_count, schema, builder))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observations_subcommands_parse_and_expose_schema_metadata() {
        let cli = Cli::try_parse_from([
            "midas",
            "observations",
            "describe",
            "--config",
            "configs/test.toml",
            "portfolio.cash_fraction",
        ])
        .unwrap();
        assert!(matches!(cli.command, Command::Observations { .. }));
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let (_, schema, builder) = observation_builder(&root.join("configs/test.toml")).unwrap();
        assert_eq!(schema.entries.len(), schema.dimension(3));
        let metadata = builder
            .registry()
            .metadata_by_name("portfolio.cash_fraction")
            .unwrap();
        assert_eq!(metadata.provider, "portfolio");
        assert_eq!(metadata.units, "ratio");
    }
}
