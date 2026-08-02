use anyhow::{Context, Result, anyhow};
use clap::{Parser, ValueEnum};
use midas_core::{
    config::Config,
    market::{MarketDataSource, PolarsDataSource},
    simulation::{DatasetSplit, SimulationEngine},
};
use serde_json::json;
use std::path::PathBuf;

#[derive(Parser)]
#[command(about = "Structured simulation worker for reproducible experiment runners")]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long, value_enum)]
    mode: Mode,
    #[arg(long, value_enum)]
    split: SplitArg,
    #[arg(long)]
    checkpoint: Option<PathBuf>,
}

#[derive(Clone, Copy, ValueEnum)]
enum Mode {
    Train,
    Evaluate,
    Simulate,
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
    let args = Args::parse();
    let config = Config::load(&args.config)?;
    let config_hash = config.hash()?;
    let data = PolarsDataSource.load(&config)?;
    let split = args.split.into();
    let mut engine = match args.mode {
        Mode::Train | Mode::Simulate => SimulationEngine::new(config, data, split)?,
        Mode::Evaluate => {
            let checkpoint = args
                .checkpoint
                .as_ref()
                .ok_or_else(|| anyhow!("evaluate mode requires --checkpoint"))?;
            SimulationEngine::evaluate_checkpoint(config, data, checkpoint, split)
                .with_context(|| format!("cannot evaluate {}", checkpoint.display()))?
        }
    };
    let report = engine.run()?;
    if matches!(args.mode, Mode::Train) {
        let checkpoint = args
            .checkpoint
            .as_ref()
            .ok_or_else(|| anyhow!("train mode requires --checkpoint"))?;
        engine.save_checkpoint(checkpoint)?;
    }
    let agents = engine.agents().cloned().collect::<Vec<_>>();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "report": report,
            "metrics": engine.metrics,
            "agents": agents,
            "evolution": engine.evolution_state(),
            "config_hash": config_hash,
        }))?
    );
    Ok(())
}
