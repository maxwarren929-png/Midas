use anyhow::Result;
use clap::{Parser, ValueEnum};
use midas_core::{
    brain::{DenseGenome, NeuralModelState, PolicySnapshot},
    config::Config,
    learning::{LearningStrategyState, TrainingMetrics},
    market::{MarketDataSource, PolarsDataSource},
    simulation::{DatasetSplit, DomainEvent, SimulationEngine},
};
use serde_json::json;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "midas-diagnostics", about = "Offline experiment diagnostics")]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long, value_enum)]
    split: SplitArg,
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

fn dense_delta(lifetime: &DenseGenome, genome: &DenseGenome) -> f64 {
    lifetime
        .weights
        .iter()
        .zip(&genome.weights)
        .chain(lifetime.bias.iter().zip(&genome.bias))
        .map(|(a, b)| f64::from((*a - *b).abs()).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn main() -> Result<()> {
    let args = Args::parse();
    let config = Config::load(&args.config)?;
    let data = PolarsDataSource.load(&config)?;
    let mut engine = SimulationEngine::new(config, data, args.split.into())?;
    let report = engine.run()?;

    let mut updates = Vec::<TrainingMetrics>::new();
    let mut masked = 0_u64;
    let mut crossover = 0_u64;
    let mut mutation = 0_u64;
    for event in &engine.domain_events {
        match event {
            DomainEvent::LearningUpdate {
                rollout_size,
                policy_loss,
                value_loss,
                entropy,
                gradient_norm,
                ..
            } => updates.push(TrainingMetrics {
                rollout_size: *rollout_size,
                policy_loss: *policy_loss,
                value_loss: *value_loss,
                entropy: *entropy,
                total_loss: 0.0,
                gradient_norm: *gradient_norm,
                update_index: updates.len() as u64,
            }),
            DomainEvent::LearningTransitionMasked { .. } => masked += 1,
            DomainEvent::GenomeCrossed { .. } => crossover += 1,
            DomainEvent::GenomeMutated { .. } => mutation += 1,
            _ => {}
        }
    }

    let mut actor_delta = 0.0_f64;
    let mut critic_delta = 0.0_f64;
    let mut hidden_delta = 0.0_f64;
    let mut recurrent_delta = 0.0_f64;
    let mut optimizer_steps = 0_u64;
    let mut rollout_entries = 0_u64;
    let mut birth_genome_matches = true;
    let mut neural_agents = 0_u64;
    let mut temperatures = Vec::new();
    for state in engine.agents() {
        let Some(genome) = state.genome.as_ref() else {
            continue;
        };
        neural_agents += 1;
        let Some(lifetime) = engine.agent_lifetime_parameters(state.id)? else {
            continue;
        };
        actor_delta += dense_delta(&lifetime.actor, &genome.actor);
        critic_delta += dense_delta(&lifetime.critic, &genome.critic);
        for (lifetime_layer, genome_layer) in lifetime.hidden.iter().zip(&genome.hidden) {
            hidden_delta += dense_delta(lifetime_layer, genome_layer);
        }
        recurrent_delta += dense_delta(&lifetime.recurrent, &genome.recurrent);
        match engine.agent_policy_snapshot(state.id)? {
            PolicySnapshot::Neural {
                genome: snapshot_genome,
                state: model_state,
                temperature,
                ..
            } => {
                birth_genome_matches &= snapshot_genome == *genome;
                temperatures.push(temperature);
                if let NeuralModelState::Burn { optimizer, .. } = *model_state {
                    optimizer_steps = optimizer_steps.saturating_add(optimizer.step);
                }
            }
            _ => birth_genome_matches = false,
        }
        if let LearningStrategyState::Ppo { rollout, .. } =
            engine.agent_learning_snapshot(state.id)?
        {
            rollout_entries = rollout_entries.saturating_add(rollout.len() as u64);
        }
    }
    let mean = |f: fn(&TrainingMetrics) -> f64| {
        if updates.is_empty() {
            0.0
        } else {
            updates.iter().map(f).sum::<f64>() / updates.len() as f64
        }
    };
    let temperature = if temperatures.is_empty() {
        0.0
    } else {
        temperatures.iter().map(|v| f64::from(*v)).sum::<f64>() / temperatures.len() as f64
    };

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "report": report,
            "mechanics": {
                "policy_failures": engine.metrics.policy_failures,
                "recoverable_fallbacks": engine.metrics.invalid_actions,
                "masked_rollout_entries": masked,
                "actor_ineligible_entries": engine.metrics.actor_ineligible_transitions,
                "critic_ineligible_entries": engine.metrics.critic_ineligible_transitions,
                "invalid_action_penalty_total": engine.metrics.invalid_action_penalty_total,
                "deaths": engine.metrics.deaths,
                "births": engine.metrics.births,
                "reproductions": engine.metrics.reproductions,
                "evolution_fallbacks": engine.metrics.evolution_fallbacks,
                "neural_action_count": engine.metrics.neural_action_count,
                "neural_multi_asset_action_fraction": if engine.metrics.neural_action_count > 0 {
                    engine.metrics.neural_multi_asset_actions as f64
                        / engine.metrics.neural_action_count as f64
                } else {
                    0.0
                },
                "neural_mean_active_positions": if engine.metrics.neural_action_count > 0 {
                    engine.metrics.neural_active_position_total
                        / engine.metrics.neural_action_count as f64
                } else {
                    0.0
                },
                "neural_mean_concentration": if engine.metrics.neural_action_count > 0 {
                    engine.metrics.neural_concentration_total
                        / engine.metrics.neural_action_count as f64
                } else {
                    0.0
                },
                "neural_mean_cash_allocation": if engine.metrics.neural_action_count > 0 {
                    engine.metrics.neural_cash_allocation_total
                        / engine.metrics.neural_action_count as f64
                } else {
                    0.0
                },
            },
            "learning": {
                "ppo_updates": updates.len(),
                "optimizer_steps": optimizer_steps,
                "rollout_entries_at_end": rollout_entries,
                "mean_policy_loss": mean(|m| f64::from(m.policy_loss)),
                "mean_value_loss": mean(|m| f64::from(m.value_loss)),
                "mean_entropy": mean(|m| f64::from(m.entropy)),
                "mean_gradient_norm": mean(|m| f64::from(m.gradient_norm)),
                "actor_head_delta_norm": actor_delta,
                "critic_head_delta_norm": critic_delta,
                "hidden_delta_norm": hidden_delta,
                "recurrent_delta_norm": recurrent_delta,
                "birth_genome_matches_policy": birth_genome_matches,
                "neural_agents_inspected": neural_agents,
                "mean_temperature": temperature,
            },
            "evolution": {
                "crossovers_observed": crossover,
                "mutations_observed": mutation,
                "maximum_generation_depth": engine.evolution_state().metrics.maximum_generation_depth,
                "active_lineages": engine.evolution_state().metrics.active_lineages,
                "mean_genetic_distance": engine.evolution_state().metrics.mean_genetic_distance,
                "minimum_genetic_distance": engine.evolution_state().metrics.minimum_genetic_distance,
            }
        }))?
    );
    Ok(())
}
