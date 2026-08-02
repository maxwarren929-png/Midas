use anyhow::{Context, Result, anyhow};
use clap::{Parser, ValueEnum};
use midas_core::{
    config::Config,
    market::{MarketDataSource, PolarsDataSource},
    simulation::{DatasetSplit, ReportTrace, SimulationEngine},
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
    let agents = engine
        .agents()
        .map(|agent| {
            json!({
                "id": agent.id,
                "alive": agent.alive,
                "age": agent.age,
                "portfolio": agent.portfolio,
                "total_policy_failures": agent.total_policy_failures,
                "consecutive_policy_failures": agent.consecutive_policy_failures,
                "genome_id": agent.genome_id,
                "lineage_id": agent.lineage_id,
            })
        })
        .collect::<Vec<_>>();
    let evolution = engine.evolution_state();
    let metrics = &engine.metrics;
    let mut decision_count = 0usize;
    let mut cash_allocation_total = 0.0;
    let mut concentration_total = 0.0;
    let mut active_positions_total = 0usize;
    for trace in &engine.report_traces {
        if let ReportTrace::Decision(decision) = trace {
            decision_count += 1;
            cash_allocation_total += decision.cash_allocation;
            concentration_total += decision.maximum_concentration;
            active_positions_total += decision.active_positions;
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "report": report,
            "metrics": {
                "invalid_actions": metrics.invalid_actions,
                "rejected_trades": metrics.rejected_trades,
                "policy_failures": metrics.policy_failures,
                "deaths": metrics.deaths,
                "births": metrics.births,
                "learning_updates": metrics.learning_updates,
                "fitness_assignments": metrics.fitness_assignments,
                "reproductions": metrics.reproductions,
                "evolution_fallbacks": metrics.evolution_fallbacks,
                "masked_failure_transitions": metrics.masked_failure_transitions,
                "actor_ineligible_transitions": metrics.actor_ineligible_transitions,
                "critic_ineligible_transitions": metrics.critic_ineligible_transitions,
                "invalid_action_penalty_total": metrics.invalid_action_penalty_total,
            },
            "agents": agents,
            "evolution": {
                "metrics": evolution.metrics,
                "next_genome_id": evolution.next_genome_id,
                "next_lineage_id": evolution.next_lineage_id,
            },
            "diagnostics": {
                "report_trace_count": engine.report_traces.len(),
                "decision_count": decision_count,
                "mean_cash_allocation": if decision_count == 0 { 0.0 } else { cash_allocation_total / decision_count as f64 },
                "mean_concentration": if decision_count == 0 { 0.0 } else { concentration_total / decision_count as f64 },
                "mean_active_positions": if decision_count == 0 { 0.0 } else { active_positions_total as f64 / decision_count as f64 },
            },
            "config_hash": config_hash,
        }))?
    );
    Ok(())
}
