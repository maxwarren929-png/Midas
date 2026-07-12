use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use midas_core::{
    config::{Config, PolicyKind},
    market::{MarketDataSource, PolarsDataSource},
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
    },
    Resume {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        checkpoint: PathBuf,
    },
    Benchmark {
        #[arg(long)]
        config: PathBuf,
        #[arg(long, value_enum, default_value = "test")]
        split: SplitArg,
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
        } => {
            let c = Config::load(&config)?;
            let d = PolarsDataSource.load(&c)?;
            let mut engine = SimulationEngine::new(c, d, split.into())?;
            let report = engine.run()?;
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
    }
    Ok(())
}
