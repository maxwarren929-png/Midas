use anyhow::Result;
use clap::{Parser, ValueEnum};
use midas_core::{
    brain::{DenseGenome, NeuralModelState, PolicySnapshot},
    config::{
        Config, OptimizerKind, PolicyKind, RewardKind, SeedDerivation, TurnoverConstraintConfig,
    },
    learning::{
        ActorGradientAttribution, LearningStrategyState, PpoAdvantageDiagnostics, TrainingMetrics,
    },
    market::{MarketDataSource, PolarsDataSource},
    simulation::{DatasetSplit, DomainEvent, PpoUpdateTrace, ReportTrace, SimulationEngine},
};
use serde_json::json;
use std::{fs, io::Write, path::PathBuf, time::Instant};

#[derive(Parser)]
#[command(name = "midas-diagnostics", about = "Offline experiment diagnostics")]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long, value_enum)]
    split: SplitArg,
    /// Optional directory for transactionally committed decision/PPO traces.
    #[arg(long)]
    trace_dir: Option<PathBuf>,
    /// Stop after this many committed simulation steps instead of completing the split.
    #[arg(long)]
    steps: Option<usize>,
    /// Override the configured population size for a controlled diagnostic run.
    #[arg(long)]
    population_size: Option<usize>,
    /// Override the bankruptcy threshold for a controlled diagnostic run.
    #[arg(long)]
    bankruptcy_threshold: Option<f64>,
    /// Override the configured deterministic simulation seed.
    #[arg(long)]
    seed: Option<u64>,
    /// Override deterministic seed derivation while preserving legacy defaults.
    #[arg(long, value_enum)]
    seed_derivation: Option<SeedDerivationArg>,
    /// Override the configured global gradient clipping norm.
    #[arg(long)]
    gradient_clip: Option<f32>,
    /// Override the configured PPO learning rate.
    #[arg(long)]
    learning_rate: Option<f32>,
    /// Override the lifetime neural optimizer.
    #[arg(long, value_enum)]
    optimizer: Option<OptimizerArg>,
    /// Override AdamW decoupled weight decay.
    #[arg(long)]
    weight_decay: Option<f32>,
    /// Override the population policy for a controlled benchmark run.
    #[arg(long, value_enum)]
    policy: Option<PolicyArg>,
    /// Override the reward function for a controlled diagnostic run.
    #[arg(long, value_enum)]
    reward_kind: Option<RewardArg>,
    /// Override the maximum-drawdown penalty weight.
    #[arg(long)]
    drawdown_penalty_weight: Option<f32>,
    /// Override the raw-notional turnover penalty weight.
    #[arg(long)]
    turnover_penalty_weight: Option<f32>,
    /// Enable constrained PPO with this mean daily turnover/net-worth budget.
    #[arg(long)]
    turnover_budget: Option<f32>,
    /// Projected dual-ascent learning rate (requires --turnover-budget).
    #[arg(long, requires = "turnover_budget", default_value_t = 0.01)]
    turnover_dual_learning_rate: f32,
    /// Initial Lagrange multiplier (requires --turnover-budget).
    #[arg(long, requires = "turnover_budget", default_value_t = 0.0)]
    turnover_initial_multiplier: f32,
    /// Maximum projected Lagrange multiplier (requires --turnover-budget).
    #[arg(long, requires = "turnover_budget", default_value_t = 10.0)]
    turnover_maximum_multiplier: f32,
    /// Turnover-cost critic loss coefficient (requires --turnover-budget).
    #[arg(long, requires = "turnover_budget", default_value_t = 0.5)]
    turnover_critic_coefficient: f32,
    /// Write the JSON summary to this path instead of standard output.
    #[arg(long)]
    output: Option<PathBuf>,
    /// Optionally save the authoritative state reached by this diagnostic run.
    #[arg(long)]
    checkpoint: Option<PathBuf>,
    /// Evaluate a training checkpoint with learning disabled instead of starting a new run.
    #[arg(long)]
    evaluate_checkpoint: Option<PathBuf>,
    /// Reset lifetime weights to the checkpoint genomes during evaluation.
    #[arg(long, requires = "evaluate_checkpoint")]
    genome_baseline: bool,
}

#[derive(Clone, Copy, ValueEnum)]
enum SplitArg {
    Training,
    Validation,
    Test,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum SeedDerivationArg {
    LegacyXor,
    SplitmixV2,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum OptimizerArg {
    Momentum,
    Adamw,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum PolicyArg {
    Cash,
    Random,
    EqualWeight,
    BuyAndHold,
    Momentum,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum RewardArg {
    #[value(name = "log-return")]
    Log,
    #[value(name = "risk-adjusted-return")]
    RiskAdjusted,
    #[value(name = "normalized-risk-adjusted-return")]
    NormalizedRiskAdjusted,
}

impl From<RewardArg> for RewardKind {
    fn from(value: RewardArg) -> Self {
        match value {
            RewardArg::Log => Self::LogReturn,
            RewardArg::RiskAdjusted => Self::RiskAdjustedReturn,
            RewardArg::NormalizedRiskAdjusted => Self::NormalizedRiskAdjustedReturn,
        }
    }
}

impl From<PolicyArg> for PolicyKind {
    fn from(value: PolicyArg) -> Self {
        match value {
            PolicyArg::Cash => Self::Cash,
            PolicyArg::Random => Self::Random,
            PolicyArg::EqualWeight => Self::EqualWeight,
            PolicyArg::BuyAndHold => Self::BuyAndHold,
            PolicyArg::Momentum => Self::Momentum,
        }
    }
}

impl From<OptimizerArg> for OptimizerKind {
    fn from(value: OptimizerArg) -> Self {
        match value {
            OptimizerArg::Momentum => Self::Momentum,
            OptimizerArg::Adamw => Self::AdamW,
        }
    }
}

impl From<SeedDerivationArg> for SeedDerivation {
    fn from(value: SeedDerivationArg) -> Self {
        match value {
            SeedDerivationArg::LegacyXor => Self::LegacyXor,
            SeedDerivationArg::SplitmixV2 => Self::SplitmixV2,
        }
    }
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

fn linux_peak_rss_kib() -> Option<u64> {
    fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find_map(|line| {
            let value = line.strip_prefix("VmHWM:")?.trim();
            value.strip_suffix(" kB")?.trim().parse().ok()
        })
}

fn force_learning_rate(config: &mut Config, learning_rate: f32) {
    config.learning.learning_rate = learning_rate;
    let trait_config = &mut config.brain.traits.learning_rate;
    trait_config.initial = learning_rate;
    trait_config.minimum = learning_rate;
    trait_config.maximum = learning_rate;
    trait_config.mutation_scale = 0.0;
    trait_config.evolvable = false;
}

fn main() -> Result<()> {
    let started = Instant::now();
    let args = Args::parse();
    let mut config = Config::load(&args.config)?;
    if let Some(population_size) = args.population_size {
        config.simulation.population_size = population_size;
    }
    if let Some(bankruptcy_threshold) = args.bankruptcy_threshold {
        config.simulation.bankruptcy_threshold = bankruptcy_threshold;
    }
    if let Some(seed) = args.seed {
        config.simulation.seed = seed;
    }
    if let Some(seed_derivation) = args.seed_derivation {
        config.simulation.seed_derivation = seed_derivation.into();
    }
    if let Some(gradient_clip) = args.gradient_clip {
        config.learning.gradient_clip = gradient_clip;
    }
    if let Some(learning_rate) = args.learning_rate {
        force_learning_rate(&mut config, learning_rate);
    }
    if let Some(optimizer) = args.optimizer {
        config.optimizer.kind = optimizer.into();
    }
    if let Some(weight_decay) = args.weight_decay {
        config.optimizer.weight_decay = weight_decay;
    }
    if let Some(policy) = args.policy {
        config.brain.policy = policy.into();
    }
    if let Some(reward_kind) = args.reward_kind {
        config.reward.kind = reward_kind.into();
    }
    if let Some(weight) = args.drawdown_penalty_weight {
        config.reward.drawdown_penalty_weight = weight;
    }
    if let Some(weight) = args.turnover_penalty_weight {
        config.reward.turnover_penalty_weight = weight;
    }
    if let Some(budget) = args.turnover_budget {
        config.turnover_constraint = Some(TurnoverConstraintConfig {
            budget,
            dual_learning_rate: args.turnover_dual_learning_rate,
            initial_multiplier: args.turnover_initial_multiplier,
            maximum_multiplier: args.turnover_maximum_multiplier,
            critic_coefficient: args.turnover_critic_coefficient,
        });
    }
    let experiment = json!({
        "requested_steps": args.steps,
        "population_size": config.simulation.population_size,
        "bankruptcy_threshold": config.simulation.bankruptcy_threshold,
        "seed": config.simulation.seed,
        "seed_derivation": config.simulation.seed_derivation,
        "gradient_clip": config.learning.gradient_clip,
        "learning_rate": config.learning.learning_rate,
        "optimizer": config.optimizer.kind,
        "weight_decay": config.optimizer.weight_decay,
        "policy": config.brain.policy,
        "reward_kind": config.reward.kind,
        "drawdown_penalty_weight": config.reward.drawdown_penalty_weight,
        "turnover_penalty_weight": config.reward.turnover_penalty_weight,
        "turnover_constraint": config.turnover_constraint,
        "evaluation_checkpoint": args.evaluate_checkpoint,
        "genome_baseline": args.genome_baseline,
    });
    let data = PolarsDataSource.load(&config)?;
    let mut engine = if let Some(checkpoint) = &args.evaluate_checkpoint {
        if args.genome_baseline {
            SimulationEngine::evaluate_checkpoint_genome_baseline(
                config,
                data,
                checkpoint,
                args.split.into(),
            )?
        } else {
            SimulationEngine::evaluate_checkpoint(config, data, checkpoint, args.split.into())?
        }
    } else {
        SimulationEngine::new(config, data, args.split.into())?
    };
    let report = if let Some(steps) = args.steps {
        engine.run_steps(steps)?;
        engine.report()
    } else {
        engine.run()?
    };

    if let Some(checkpoint) = &args.checkpoint {
        if let Some(parent) = checkpoint
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)?;
        }
        engine.save_checkpoint(checkpoint)?;
    }

    if let Some(trace_dir) = &args.trace_dir {
        fs::create_dir_all(trace_dir)?;
        let path = trace_dir.join("traces.jsonl");
        let mut file = fs::File::create(path)?;
        for trace in &engine.report_traces {
            serde_json::to_writer(&mut file, trace)?;
            file.write_all(b"\n")?;
        }
    }

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
                pre_clip_gradient_norm: *gradient_norm,
                hidden_gradient_norms: Vec::new(),
                recurrent_gradient_norm: 0.0,
                gru_update_gradient_norm: None,
                gru_reset_gradient_norm: None,
                gru_candidate_gradient_norm: None,
                actor_gradient_norm: 0.0,
                critic_gradient_norm: 0.0,
                turnover_critic_loss: 0.0,
                turnover_critic_gradient_norm: 0.0,
                turnover_constraint_multiplier: None,
                mean_turnover_cost: None,
                gradient_clip_fraction: 0.0,
                relative_update_norm: 0.0,
                approximate_kl: 0.0,
                clip_fraction: 0.0,
                update_index: updates.len() as u64,
                actor_gradient_attribution: None,
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
    let ppo_traces = engine
        .report_traces
        .iter()
        .filter_map(|trace| match trace {
            ReportTrace::PpoUpdate(update) => Some(update),
            ReportTrace::Decision(_) => None,
        })
        .collect::<Vec<_>>();
    let decision_traces = engine
        .report_traces
        .iter()
        .filter_map(|trace| match trace {
            ReportTrace::Decision(decision) => Some(decision),
            ReportTrace::PpoUpdate(_) => None,
        })
        .collect::<Vec<_>>();
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
                let optimizer_step = match *model_state {
                    NeuralModelState::Burn { optimizer, .. } => optimizer.step,
                    NeuralModelState::BurnTrainableEncoder { optimizer, .. } => optimizer.step,
                    NeuralModelState::BurnBptt { optimizer, .. } => optimizer.step,
                    NeuralModelState::BurnAdamW { optimizer, .. } => optimizer.step,
                    NeuralModelState::External { .. } => 0,
                };
                optimizer_steps = optimizer_steps.saturating_add(optimizer_step);
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
    let mean_ppo = |f: fn(&PpoUpdateTrace) -> f64| {
        if ppo_traces.is_empty() {
            0.0
        } else {
            ppo_traces.iter().map(|trace| f(trace)).sum::<f64>() / ppo_traces.len() as f64
        }
    };
    let hidden_layer_count = ppo_traces
        .iter()
        .map(|trace| trace.hidden_gradient_norms.len())
        .max()
        .unwrap_or(0);
    let mean_hidden_gradient_norms = (0..hidden_layer_count)
        .map(|layer| {
            if ppo_traces.is_empty() {
                0.0
            } else {
                ppo_traces
                    .iter()
                    .map(|trace| {
                        trace
                            .hidden_gradient_norms
                            .get(layer)
                            .copied()
                            .map_or(0.0, f64::from)
                    })
                    .sum::<f64>()
                    / ppo_traces.len() as f64
            }
        })
        .collect::<Vec<_>>();
    let mean_optional_ppo = |f: fn(&PpoUpdateTrace) -> Option<f32>| {
        let values = ppo_traces
            .iter()
            .filter_map(|trace| f(trace).map(f64::from))
            .collect::<Vec<_>>();
        if values.is_empty() {
            None
        } else {
            Some(values.iter().sum::<f64>() / values.len() as f64)
        }
    };
    let advantage_diagnostics = ppo_traces
        .iter()
        .filter_map(|trace| trace.advantage_diagnostics.as_ref())
        .collect::<Vec<_>>();
    let advantage_actor_samples = advantage_diagnostics
        .iter()
        .map(|diagnostics| diagnostics.actor_sample_count)
        .sum::<usize>();
    let weighted_advantage = |f: fn(&PpoAdvantageDiagnostics) -> f64| {
        if advantage_actor_samples == 0 {
            0.0
        } else {
            advantage_diagnostics
                .iter()
                .map(|diagnostics| f(diagnostics) * diagnostics.actor_sample_count as f64)
                .sum::<f64>()
                / advantage_actor_samples as f64
        }
    };
    let gradient_attributions = ppo_traces
        .iter()
        .filter_map(|trace| {
            trace
                .actor_gradient_attribution
                .as_ref()
                .map(|attribution| {
                    (
                        attribution,
                        trace
                            .advantage_diagnostics
                            .as_ref()
                            .map_or(0, |diagnostics| diagnostics.actor_sample_count),
                    )
                })
        })
        .collect::<Vec<_>>();
    let gradient_actor_samples = gradient_attributions
        .iter()
        .map(|(_, samples)| *samples)
        .sum::<usize>();
    let weighted_gradient = |f: fn(&ActorGradientAttribution) -> f64| {
        if gradient_actor_samples == 0 {
            0.0
        } else {
            gradient_attributions
                .iter()
                .map(|(attribution, samples)| f(attribution) * *samples as f64)
                .sum::<f64>()
                / gradient_actor_samples as f64
        }
    };

    let summary = serde_json::to_string_pretty(&json!({
            "experiment": experiment,
            "resources": {
                "wall_seconds": started.elapsed().as_secs_f64(),
                "peak_rss_kib": linux_peak_rss_kib(),
                "available_parallelism": std::thread::available_parallelism().ok().map(|count| count.get()),
            },
        "report": report,
        "mechanics": {
            "policy_failures": engine.metrics.policy_failures,
            "recoverable_fallbacks": engine.metrics.invalid_actions,
            "masked_rollout_entries": masked,
            "actor_ineligible_entries": engine.metrics.actor_ineligible_transitions,
            "critic_ineligible_entries": engine.metrics.critic_ineligible_transitions,
            "invalid_action_penalty_total": engine.metrics.invalid_action_penalty_total,
            "reward_decisions": decision_traces.len(),
            "raw_log_return_total": decision_traces.iter().map(|trace| trace.reward_breakdown.raw_log_return).sum::<f64>(),
            "drawdown_basis_total": decision_traces.iter().map(|trace| trace.reward_breakdown.drawdown_basis).sum::<f64>(),
            "turnover_basis_total": decision_traces.iter().map(|trace| trace.reward_breakdown.turnover_basis).sum::<f64>(),
            "drawdown_penalty_total": decision_traces.iter().map(|trace| trace.reward_breakdown.drawdown_penalty).sum::<f64>(),
            "turnover_penalty_total": decision_traces.iter().map(|trace| trace.reward_breakdown.turnover_penalty).sum::<f64>(),
            "unclipped_reward_total": decision_traces.iter().map(|trace| trace.reward_breakdown.unclipped_total).sum::<f64>(),
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
            "mean_abs_trade_impulse": if engine.metrics.neural_impulse_count > 0 {
                engine.metrics.neural_abs_impulse_total / engine.metrics.neural_impulse_count as f64
            } else { 0.0 },
            "small_trade_impulse_fraction": if engine.metrics.neural_impulse_count > 0 {
                engine.metrics.neural_small_impulse_count as f64 / engine.metrics.neural_impulse_count as f64
            } else { 0.0 },
            "saturated_trade_impulse_fraction": if engine.metrics.neural_impulse_count > 0 {
                engine.metrics.neural_saturated_impulse_count as f64 / engine.metrics.neural_impulse_count as f64
            } else { 0.0 },
        },
        "learning": {
            "ppo_updates": updates.len(),
            "optimizer_steps": optimizer_steps,
            "rollout_entries_at_end": rollout_entries,
            "mean_policy_loss": mean(|m| f64::from(m.policy_loss)),
            "mean_value_loss": mean(|m| f64::from(m.value_loss)),
            "mean_entropy": mean(|m| f64::from(m.entropy)),
            "mean_gradient_norm": mean(|m| f64::from(m.gradient_norm)),
            "advantage_signal": {
                "updates_observed": advantage_diagnostics.len(),
                "complete_updates": advantage_diagnostics.iter().filter(|diagnostics| diagnostics.complete).count(),
                "actor_samples": advantage_actor_samples,
                "weighted_pre_normalization_mean": weighted_advantage(|diagnostics| diagnostics.pre_normalization_mean),
                "weighted_pre_normalization_standard_deviation": weighted_advantage(|diagnostics| diagnostics.pre_normalization_standard_deviation),
                "weighted_post_normalization_mean": weighted_advantage(|diagnostics| diagnostics.post_normalization_mean),
                "weighted_post_normalization_standard_deviation": weighted_advantage(|diagnostics| diagnostics.post_normalization_standard_deviation),
                "weighted_normalization_scale": weighted_advantage(|diagnostics| diagnostics.normalization_scale),
                "weighted_sign_flip_fraction": weighted_advantage(|diagnostics| diagnostics.sign_flip_fraction),
                "rank_inversions": advantage_diagnostics.iter().map(|diagnostics| diagnostics.rank_inversions).sum::<u64>(),
                "raw_return": {
                    "weighted_standard_deviation": weighted_advantage(|diagnostics| diagnostics.raw_return.standard_deviation),
                    "weighted_correlation": weighted_advantage(|diagnostics| diagnostics.raw_return.correlation_with_pre_normalization_advantage),
                    "weighted_projection": weighted_advantage(|diagnostics| diagnostics.raw_return.projection_on_pre_normalization_advantage),
                },
                "drawdown_cost": {
                    "weighted_standard_deviation": weighted_advantage(|diagnostics| diagnostics.drawdown_cost.standard_deviation),
                    "weighted_correlation": weighted_advantage(|diagnostics| diagnostics.drawdown_cost.correlation_with_pre_normalization_advantage),
                    "weighted_projection": weighted_advantage(|diagnostics| diagnostics.drawdown_cost.projection_on_pre_normalization_advantage),
                },
                "turnover_cost": {
                    "weighted_standard_deviation": weighted_advantage(|diagnostics| diagnostics.turnover_cost.standard_deviation),
                    "weighted_correlation": weighted_advantage(|diagnostics| diagnostics.turnover_cost.correlation_with_pre_normalization_advantage),
                    "weighted_projection": weighted_advantage(|diagnostics| diagnostics.turnover_cost.projection_on_pre_normalization_advantage),
                },
                "other_cost": {
                    "weighted_standard_deviation": weighted_advantage(|diagnostics| diagnostics.other_cost.standard_deviation),
                    "weighted_correlation": weighted_advantage(|diagnostics| diagnostics.other_cost.correlation_with_pre_normalization_advantage),
                    "weighted_projection": weighted_advantage(|diagnostics| diagnostics.other_cost.projection_on_pre_normalization_advantage),
                },
            },
            "actor_gradient_signal": {
                "updates_observed": gradient_attributions.len(),
                "actor_samples": gradient_actor_samples,
                "weighted_actual_policy_gradient_norm": weighted_gradient(|attribution| attribution.actual_policy_gradient_norm),
                "raw_return": {
                    "weighted_norm": weighted_gradient(|attribution| attribution.raw_return.norm),
                    "weighted_cosine_with_actual": weighted_gradient(|attribution| attribution.raw_return.cosine_with_actual_policy_gradient),
                },
                "drawdown_cost": {
                    "weighted_norm": weighted_gradient(|attribution| attribution.drawdown_cost.norm),
                    "weighted_cosine_with_actual": weighted_gradient(|attribution| attribution.drawdown_cost.cosine_with_actual_policy_gradient),
                },
                "turnover_cost": {
                    "weighted_norm": weighted_gradient(|attribution| attribution.turnover_cost.norm),
                    "weighted_cosine_with_actual": weighted_gradient(|attribution| attribution.turnover_cost.cosine_with_actual_policy_gradient),
                },
                "other_cost": {
                    "weighted_norm": weighted_gradient(|attribution| attribution.other_cost.norm),
                    "weighted_cosine_with_actual": weighted_gradient(|attribution| attribution.other_cost.cosine_with_actual_policy_gradient),
                },
                "weighted_raw_return_drawdown_cosine": weighted_gradient(|attribution| attribution.raw_return_drawdown_cosine),
                "weighted_raw_return_turnover_cosine": weighted_gradient(|attribution| attribution.raw_return_turnover_cosine),
                "weighted_drawdown_turnover_cosine": weighted_gradient(|attribution| attribution.drawdown_turnover_cosine),
            },
            "gradient_health": {
                "mean_pre_clip_global_norm": mean_ppo(|trace| f64::from(trace.pre_clip_gradient_norm)),
                "mean_post_clip_global_norm": mean_ppo(|trace| f64::from(trace.gradient_norm)),
                "mean_hidden_layer_norms": mean_hidden_gradient_norms,
                "mean_recurrent_norm": mean_ppo(|trace| f64::from(trace.recurrent_gradient_norm)),
                "mean_gru_update_gate_norm": mean_optional_ppo(|trace| trace.gru_update_gradient_norm),
                "mean_gru_reset_gate_norm": mean_optional_ppo(|trace| trace.gru_reset_gradient_norm),
                "mean_gru_candidate_norm": mean_optional_ppo(|trace| trace.gru_candidate_gradient_norm),
                "mean_actor_norm": mean_ppo(|trace| f64::from(trace.actor_gradient_norm)),
                "mean_critic_norm": mean_ppo(|trace| f64::from(trace.critic_gradient_norm)),
                "mean_turnover_critic_loss": mean_ppo(|trace| f64::from(trace.turnover_critic_loss)),
                "mean_turnover_critic_norm": mean_ppo(|trace| f64::from(trace.turnover_critic_gradient_norm)),
                "mean_turnover_constraint_multiplier": mean_optional_ppo(|trace| trace.turnover_constraint_multiplier),
                "mean_turnover_cost": mean_optional_ppo(|trace| trace.mean_turnover_cost),
                "gradient_clip_frequency": mean_ppo(|trace| f64::from(trace.gradient_clip_fraction)),
                "mean_relative_update_norm": mean_ppo(|trace| f64::from(trace.relative_update_norm)),
            },
            "mean_approximate_kl": if ppo_traces.is_empty() { 0.0 } else {
                ppo_traces.iter().map(|trace| f64::from(trace.approximate_kl)).sum::<f64>()
                    / ppo_traces.len() as f64
            },
            "mean_clip_fraction": if ppo_traces.is_empty() { 0.0 } else {
                ppo_traces.iter().map(|trace| f64::from(trace.clip_fraction)).sum::<f64>()
                    / ppo_traces.len() as f64
            },
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
    }))?;
    if let Some(output) = &args.output {
        if let Some(parent) = output
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)?;
        }
        fs::write(output, summary)?;
    } else {
        println!("{summary}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn controlled_run_overrides_parse_together() {
        let args = Args::try_parse_from([
            "diagnostics",
            "--config",
            "configs/prompt6/combined-ppo.toml",
            "--split",
            "training",
            "--steps",
            "127",
            "--population-size",
            "10",
            "--bankruptcy-threshold",
            "0",
            "--seed",
            "43",
            "--seed-derivation",
            "splitmix-v2",
            "--gradient-clip",
            "2.0",
            "--learning-rate",
            "0.000003",
            "--optimizer",
            "adamw",
            "--weight-decay",
            "0.01",
            "--policy",
            "equal-weight",
            "--reward-kind",
            "risk-adjusted-return",
            "--drawdown-penalty-weight",
            "0.0005",
            "--turnover-penalty-weight",
            "0.00000005",
            "--turnover-budget",
            "0.02",
            "--turnover-dual-learning-rate",
            "0.1",
            "--turnover-initial-multiplier",
            "0.25",
            "--turnover-maximum-multiplier",
            "5.0",
            "--turnover-critic-coefficient",
            "0.75",
            "--output",
            "summary.json",
            "--checkpoint",
            "state.bin",
            "--evaluate-checkpoint",
            "trained.bin",
            "--genome-baseline",
        ])
        .unwrap();

        assert!(matches!(args.split, SplitArg::Training));
        assert_eq!(args.steps, Some(127));
        assert_eq!(args.population_size, Some(10));
        assert_eq!(args.bankruptcy_threshold, Some(0.0));
        assert_eq!(args.seed, Some(43));
        assert!(matches!(
            args.seed_derivation,
            Some(SeedDerivationArg::SplitmixV2)
        ));
        assert_eq!(args.gradient_clip, Some(2.0));
        assert_eq!(args.learning_rate, Some(0.000003));
        assert!(matches!(args.optimizer, Some(OptimizerArg::Adamw)));
        assert_eq!(args.weight_decay, Some(0.01));
        assert!(matches!(args.policy, Some(PolicyArg::EqualWeight)));
        assert!(matches!(args.reward_kind, Some(RewardArg::RiskAdjusted)));
        assert_eq!(args.drawdown_penalty_weight, Some(0.0005));
        assert_eq!(args.turnover_penalty_weight, Some(0.00000005));
        assert_eq!(args.turnover_budget, Some(0.02));
        assert_eq!(args.turnover_dual_learning_rate, 0.1);
        assert_eq!(args.turnover_initial_multiplier, 0.25);
        assert_eq!(args.turnover_maximum_multiplier, 5.0);
        assert_eq!(args.turnover_critic_coefficient, 0.75);
        assert_eq!(args.output, Some(PathBuf::from("summary.json")));
        assert_eq!(args.checkpoint, Some(PathBuf::from("state.bin")));
        assert_eq!(args.evaluate_checkpoint, Some(PathBuf::from("trained.bin")));
        assert!(args.genome_baseline);
    }

    #[test]
    fn learning_rate_override_controls_root_and_genome_trait() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut config = Config::load(root.join("configs/prompt6/combined-ppo.toml")).unwrap();
        force_learning_rate(&mut config, 0.000003);

        assert_eq!(config.learning.learning_rate, 0.000003);
        assert_eq!(config.brain.traits.learning_rate.initial, 0.000003);
        assert_eq!(config.brain.traits.learning_rate.minimum, 0.000003);
        assert_eq!(config.brain.traits.learning_rate.maximum, 0.000003);
        assert!(!config.brain.traits.learning_rate.evolvable);
        config.validate().unwrap();
    }
}
