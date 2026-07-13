use crate::{
    MidasError, Result,
    action::ProposedAllocation,
    config::{Activation, BrainConfig, EvolvableTraitsConfig, LearningConfig},
    observation::Observation,
};
use burn::{
    backend::NdArray,
    tensor::{Tensor, TensorData, activation},
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

type CpuBackend = NdArray<f32>;
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MemoryArchitecture {
    pub size: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct BrainArchitecture {
    pub version: u32,
    pub observation_schema_hash: String,
    pub observation_size: usize,
    pub asset_count: usize,
    pub hidden_layers: Vec<usize>,
    pub memory: MemoryArchitecture,
    pub actor_output_size: usize,
    pub critic_output_size: usize,
    pub activation: Activation,
}
impl BrainArchitecture {
    pub fn from_config(
        c: &BrainConfig,
        schema_hash: String,
        observation_size: usize,
        asset_count: usize,
    ) -> Self {
        Self {
            version: c.architecture_version,
            observation_schema_hash: schema_hash,
            observation_size,
            asset_count,
            hidden_layers: c.hidden_layers.clone(),
            memory: MemoryArchitecture {
                size: c.memory_size,
            },
            actor_output_size: asset_count + 1,
            critic_output_size: 1,
            activation: c.activation,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DenseGenome {
    pub inputs: usize,
    pub outputs: usize,
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LearningTraits {
    pub learning_rate: f32,
    pub entropy_coefficient: f32,
    pub action_temperature: f32,
    pub temperature_decay: f32,
    pub drawdown_sensitivity: f32,
    pub turnover_sensitivity: f32,
    pub maximum_allocation: f32,
}
impl LearningTraits {
    fn from_config(c: &EvolvableTraitsConfig) -> Self {
        Self {
            learning_rate: c.learning_rate.initial,
            entropy_coefficient: c.entropy_coefficient.initial,
            action_temperature: c.action_temperature.initial,
            temperature_decay: c.temperature_decay.initial,
            drawdown_sensitivity: c.drawdown_sensitivity.initial,
            turnover_sensitivity: c.turnover_sensitivity.initial,
            maximum_allocation: c.maximum_allocation.initial,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct BrainGenome {
    pub genome_format_version: u32,
    pub architecture: BrainArchitecture,
    pub hidden: Vec<DenseGenome>,
    pub recurrent: DenseGenome,
    pub actor: DenseGenome,
    pub critic: DenseGenome,
    pub traits: LearningTraits,
}
impl BrainGenome {
    pub fn random(
        a: &BrainArchitecture,
        traits: &EvolvableTraitsConfig,
        seed: u64,
    ) -> Result<Self> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut input = a.observation_size;
        let mut hidden = Vec::new();
        for &output in &a.hidden_layers {
            hidden.push(random_dense(input, output, &mut rng));
            input = output;
        }
        let recurrent = random_dense(input + a.memory.size, a.memory.size, &mut rng);
        let actor = random_dense(a.memory.size, a.actor_output_size, &mut rng);
        let critic = random_dense(a.memory.size, a.critic_output_size, &mut rng);
        let g = Self {
            genome_format_version: 2,
            architecture: a.clone(),
            hidden,
            recurrent,
            actor,
            critic,
            traits: LearningTraits::from_config(traits),
        };
        g.validate(a)?;
        Ok(g)
    }
    pub fn validate(&self, a: &BrainArchitecture) -> Result<()> {
        if self.genome_format_version != 2 || &self.architecture != a {
            return Err(MidasError::Policy(
                "genome architecture or observation schema mismatch".into(),
            ));
        }
        if self.hidden.len() != a.hidden_layers.len()
            || self
                .hidden
                .iter()
                .zip(&a.hidden_layers)
                .any(|(layer, expected_outputs)| layer.outputs != *expected_outputs)
        {
            return Err(MidasError::Policy(
                "genome hidden-layer layout mismatch".into(),
            ));
        }
        let mut expected = a.observation_size;
        for l in &self.hidden {
            validate_dense(l, expected)?;
            expected = l.outputs;
        }
        validate_dense(&self.recurrent, expected + a.memory.size)?;
        validate_dense(&self.actor, a.memory.size)?;
        validate_dense(&self.critic, a.memory.size)?;
        if self.recurrent.outputs != a.memory.size
            || self.actor.outputs != a.actor_output_size
            || self.critic.outputs != a.critic_output_size
        {
            return Err(MidasError::Policy(
                "genome output dimensions mismatch".into(),
            ));
        }
        Ok(())
    }
    pub fn validate_with_traits(
        &self,
        architecture: &BrainArchitecture,
        bounds: &EvolvableTraitsConfig,
    ) -> Result<()> {
        self.validate(architecture)?;
        for (name, value, configured) in [
            (
                "learning_rate",
                self.traits.learning_rate,
                &bounds.learning_rate,
            ),
            (
                "entropy_coefficient",
                self.traits.entropy_coefficient,
                &bounds.entropy_coefficient,
            ),
            (
                "action_temperature",
                self.traits.action_temperature,
                &bounds.action_temperature,
            ),
            (
                "temperature_decay",
                self.traits.temperature_decay,
                &bounds.temperature_decay,
            ),
            (
                "drawdown_sensitivity",
                self.traits.drawdown_sensitivity,
                &bounds.drawdown_sensitivity,
            ),
            (
                "turnover_sensitivity",
                self.traits.turnover_sensitivity,
                &bounds.turnover_sensitivity,
            ),
            (
                "maximum_allocation",
                self.traits.maximum_allocation,
                &bounds.maximum_allocation,
            ),
        ] {
            if !value.is_finite() || value < configured.minimum || value > configured.maximum {
                return Err(MidasError::Policy(format!(
                    "genome trait {name} is outside configured bounds"
                )));
            }
        }
        Ok(())
    }
}
fn validate_dense(l: &DenseGenome, inputs: usize) -> Result<()> {
    if l.inputs != inputs
        || l.outputs == 0
        || l.weights.len() != l.inputs * l.outputs
        || l.bias.len() != l.outputs
        || l.weights.iter().chain(&l.bias).any(|v| !v.is_finite())
    {
        Err(MidasError::Policy("invalid dense parameters".into()))
    } else {
        Ok(())
    }
}
fn random_dense(inputs: usize, outputs: usize, r: &mut ChaCha8Rng) -> DenseGenome {
    let scale = (2.0 / inputs as f32).sqrt();
    DenseGenome {
        inputs,
        outputs,
        weights: (0..inputs * outputs)
            .map(|_| r.gen_range(-scale..scale))
            .collect(),
        bias: vec![0.0; outputs],
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LifetimeParameters {
    pub hidden: Vec<DenseGenome>,
    pub recurrent: DenseGenome,
    pub actor: DenseGenome,
    pub critic: DenseGenome,
}
impl LifetimeParameters {
    fn from_genome(genome: &BrainGenome) -> Self {
        Self {
            hidden: genome.hidden.clone(),
            recurrent: genome.recurrent.clone(),
            actor: genome.actor.clone(),
            critic: genome.critic.clone(),
        }
    }
    fn validate(&self, architecture: &BrainArchitecture) -> Result<()> {
        if self.hidden.len() != architecture.hidden_layers.len()
            || self
                .hidden
                .iter()
                .zip(&architecture.hidden_layers)
                .any(|(layer, expected_outputs)| layer.outputs != *expected_outputs)
        {
            return Err(MidasError::Policy(
                "lifetime hidden-layer layout mismatch".into(),
            ));
        }
        let mut expected = architecture.observation_size;
        for layer in &self.hidden {
            validate_dense(layer, expected)?;
            expected = layer.outputs;
        }
        validate_dense(&self.recurrent, expected + architecture.memory.size)?;
        validate_dense(&self.actor, architecture.memory.size)?;
        validate_dense(&self.critic, architecture.memory.size)?;
        if self.recurrent.outputs != architecture.memory.size
            || self.actor.outputs != architecture.actor_output_size
            || self.critic.outputs != architecture.critic_output_size
        {
            return Err(MidasError::Policy(
                "lifetime parameter output dimensions mismatch".into(),
            ));
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct NeuralRuntimeState {
    pub memory: Vec<f32>,
    pub inference_steps: u64,
}
#[derive(Clone, Debug)]
pub struct NeuralInput<'a> {
    pub observation: &'a Observation,
}
#[derive(Clone, Debug, PartialEq)]
pub struct NeuralOutput {
    pub logits: Vec<f32>,
    pub value: f32,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PolicyActionSample {
    pub action_index: usize,
    pub log_probability: f32,
    pub entropy: f32,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PolicyEvaluation {
    pub proposed_allocation: ProposedAllocation,
    pub action_sample: Option<PolicyActionSample>,
    pub log_probability: Option<f32>,
    pub entropy: f32,
    pub value_estimate: f32,
    pub raw_logits: Vec<f32>,
    pub temperature: f32,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum NeuralModelState {
    Burn {
        parameters: Box<LifetimeParameters>,
        runtime: NeuralRuntimeState,
        optimizer: OptimizerState,
    },
    External {
        kind: String,
        state: Vec<f32>,
    },
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct OptimizerState {
    pub step: u64,
    pub actor_momentum: Vec<f32>,
    pub critic_momentum: Vec<f32>,
}
impl OptimizerState {
    pub fn empty(architecture: &BrainArchitecture) -> Self {
        Self {
            step: 0,
            actor_momentum: vec![0.0; architecture.actor_output_size],
            critic_momentum: vec![0.0; architecture.critic_output_size],
        }
    }
}
pub trait NeuralModel: Send {
    fn forward(&mut self, input: &NeuralInput<'_>) -> Result<NeuralOutput>;
    fn reset_memory(&mut self);
    fn snapshot_state(&self) -> Result<NeuralModelState>;
    /// Fork runtime state for a transactional simulation step. Framework
    /// adapters should clone cached parameters rather than rebuilding them.
    fn fork(&self) -> Result<Box<dyn NeuralModel>> {
        Err(MidasError::Policy(
            "neural model does not support transactional forking".into(),
        ))
    }
    fn runtime_parameter_builds(&self) -> u64 {
        0
    }
    fn train_batch(
        &mut self,
        _batch: &crate::learning::PpoTrainingBatch,
        _config: &LearningConfig,
    ) -> Result<crate::learning::TrainingMetrics> {
        Err(MidasError::Policy(
            "neural model is inference-only and cannot train".into(),
        ))
    }
    fn optimizer_state(&self) -> Result<OptimizerState> {
        Err(MidasError::Policy(
            "neural model has no optimizer state".into(),
        ))
    }
    fn lifetime_parameters(&self) -> Result<LifetimeParameters> {
        Err(MidasError::Policy(
            "neural model does not expose lifetime parameters".into(),
        ))
    }
    fn runtime_state(&self) -> Option<NeuralRuntimeState> {
        None
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComputeDevice {
    CpuNdArray { device: String },
}
impl ComputeDevice {
    pub const CPU_NDARRAY_DEFAULT: &'static str = "default";

    pub fn validate(&self) -> Result<()> {
        match self {
            Self::CpuNdArray { device } if device == Self::CPU_NDARRAY_DEFAULT => Ok(()),
            Self::CpuNdArray { device } => Err(MidasError::Config(format!(
                "unsupported CPU NdArray device '{device}'; only '{}' is implemented",
                Self::CPU_NDARRAY_DEFAULT
            ))),
        }
    }

    pub fn identity(&self) -> String {
        match self {
            Self::CpuNdArray { device } => format!("cpu_ndarray:{device}"),
        }
    }
}
pub trait NeuralModelFactory: Send + Sync {
    fn build(
        &self,
        genome: &BrainGenome,
        architecture: &BrainArchitecture,
        device: &ComputeDevice,
    ) -> Result<Box<dyn NeuralModel>>;
    fn restore(
        &self,
        genome: &BrainGenome,
        architecture: &BrainArchitecture,
        state: NeuralModelState,
        device: &ComputeDevice,
    ) -> Result<Box<dyn NeuralModel>>;
}

pub struct BurnNeuralModel {
    architecture: BrainArchitecture,
    lifetime: LifetimeParameters,
    runtime: NeuralRuntimeState,
    cached: BurnRuntimeParameters,
    optimizer: OptimizerState,
}
#[derive(Clone)]
struct BurnRuntimeDense {
    weights: Tensor<CpuBackend, 2>,
    bias: Tensor<CpuBackend, 2>,
}
#[derive(Clone)]
struct BurnRuntimeParameters {
    hidden: Vec<BurnRuntimeDense>,
    recurrent: BurnRuntimeDense,
    actor: BurnRuntimeDense,
    critic: BurnRuntimeDense,
    builds: u64,
}
impl BurnRuntimeParameters {
    fn from_lifetime(
        lifetime: &LifetimeParameters,
        d: &<CpuBackend as burn::tensor::backend::Backend>::Device,
    ) -> Self {
        Self {
            hidden: lifetime
                .hidden
                .iter()
                .map(|l| runtime_dense(l, d))
                .collect(),
            recurrent: runtime_dense(&lifetime.recurrent, d),
            actor: runtime_dense(&lifetime.actor, d),
            critic: runtime_dense(&lifetime.critic, d),
            builds: 1,
        }
    }
}
fn runtime_dense(
    l: &DenseGenome,
    d: &<CpuBackend as burn::tensor::backend::Backend>::Device,
) -> BurnRuntimeDense {
    BurnRuntimeDense {
        weights: Tensor::from_data(TensorData::new(l.weights.clone(), [l.inputs, l.outputs]), d),
        bias: Tensor::from_data(TensorData::new(l.bias.clone(), [1, l.outputs]), d),
    }
}
pub struct BurnNeuralModelFactory;
impl NeuralModelFactory for BurnNeuralModelFactory {
    fn build(
        &self,
        g: &BrainGenome,
        a: &BrainArchitecture,
        device: &ComputeDevice,
    ) -> Result<Box<dyn NeuralModel>> {
        device.validate()?;
        g.validate(a)?;
        let burn_device = Default::default();
        let lifetime = LifetimeParameters::from_genome(g);
        lifetime.validate(a)?;
        Ok(Box::new(BurnNeuralModel {
            architecture: a.clone(),
            cached: BurnRuntimeParameters::from_lifetime(&lifetime, &burn_device),
            lifetime,
            runtime: NeuralRuntimeState {
                memory: vec![0.0; a.memory.size],
                inference_steps: 0,
            },
            optimizer: OptimizerState::empty(a),
        }))
    }
    fn restore(
        &self,
        g: &BrainGenome,
        a: &BrainArchitecture,
        state: NeuralModelState,
        device: &ComputeDevice,
    ) -> Result<Box<dyn NeuralModel>> {
        device.validate().map_err(|error| {
            MidasError::Checkpoint(format!("checkpoint backend is unsupported: {error}"))
        })?;
        g.validate(a)?;
        let NeuralModelState::Burn {
            parameters,
            runtime,
            optimizer,
        } = state
        else {
            return Err(MidasError::Checkpoint("expected Burn neural state".into()));
        };
        let lifetime = *parameters;
        lifetime.validate(a)?;
        if runtime.memory.len() != a.memory.size {
            return Err(MidasError::Checkpoint(
                "neural runtime memory size mismatch".into(),
            ));
        }
        if optimizer.actor_momentum.len() != a.actor_output_size
            || optimizer.critic_momentum.len() != a.critic_output_size
            || optimizer
                .actor_momentum
                .iter()
                .chain(&optimizer.critic_momentum)
                .any(|value| !value.is_finite())
        {
            return Err(MidasError::Checkpoint(
                "neural optimizer state shape or finiteness mismatch".into(),
            ));
        }
        let burn_device = Default::default();
        Ok(Box::new(BurnNeuralModel {
            architecture: a.clone(),
            cached: BurnRuntimeParameters::from_lifetime(&lifetime, &burn_device),
            lifetime,
            runtime,
            optimizer,
        }))
    }
}
impl NeuralModel for BurnNeuralModel {
    fn forward(&mut self, input: &NeuralInput<'_>) -> Result<NeuralOutput> {
        if input.observation.schema_hash != self.architecture.observation_schema_hash
            || input.observation.values.len() != self.architecture.observation_size
            || input.observation.values.iter().any(|v| !v.is_finite())
        {
            return Err(MidasError::Policy("invalid neural observation".into()));
        }
        let device = Default::default();
        let mut x = Tensor::<CpuBackend, 2>::from_data(
            TensorData::new(
                input.observation.values.clone(),
                [1, self.architecture.observation_size],
            ),
            &device,
        );
        for l in &self.cached.hidden {
            x = activate(dense_runtime(x, l), self.architecture.activation);
        }
        let memory = Tensor::<CpuBackend, 2>::from_data(
            TensorData::new(
                self.runtime.memory.clone(),
                [1, self.architecture.memory.size],
            ),
            &device,
        );
        let next = activation::silu(dense_runtime(
            Tensor::cat(vec![x, memory], 1),
            &self.cached.recurrent,
        ));
        let logits = to_vec(dense_runtime(next.clone(), &self.cached.actor))?;
        let value = to_vec(dense_runtime(next.clone(), &self.cached.critic))?[0];
        self.runtime.memory = to_vec(next)?;
        self.runtime.inference_steps += 1;
        if logits.iter().any(|v| !v.is_finite())
            || !value.is_finite()
            || self.runtime.memory.iter().any(|v| !v.is_finite())
        {
            return Err(MidasError::Policy("non-finite neural output".into()));
        }
        Ok(NeuralOutput { logits, value })
    }
    fn reset_memory(&mut self) {
        self.runtime.memory.fill(0.0)
    }
    fn snapshot_state(&self) -> Result<NeuralModelState> {
        Ok(NeuralModelState::Burn {
            parameters: Box::new(self.lifetime.clone()),
            runtime: self.runtime.clone(),
            optimizer: self.optimizer.clone(),
        })
    }
    fn fork(&self) -> Result<Box<dyn NeuralModel>> {
        Ok(Box::new(Self {
            architecture: self.architecture.clone(),
            lifetime: self.lifetime.clone(),
            runtime: self.runtime.clone(),
            cached: self.cached.clone(),
            optimizer: self.optimizer.clone(),
        }))
    }
    fn runtime_parameter_builds(&self) -> u64 {
        self.cached.builds
    }
    fn train_batch(
        &mut self,
        batch: &crate::learning::PpoTrainingBatch,
        config: &LearningConfig,
    ) -> Result<crate::learning::TrainingMetrics> {
        train_burn_heads(self, batch, config)
    }
    fn optimizer_state(&self) -> Result<OptimizerState> {
        Ok(self.optimizer.clone())
    }
    fn lifetime_parameters(&self) -> Result<LifetimeParameters> {
        Ok(self.lifetime.clone())
    }
    fn runtime_state(&self) -> Option<NeuralRuntimeState> {
        Some(self.runtime.clone())
    }
}
fn dense_runtime(x: Tensor<CpuBackend, 2>, l: &BurnRuntimeDense) -> Tensor<CpuBackend, 2> {
    x.matmul(l.weights.clone()) + l.bias.clone()
}
fn activate(x: Tensor<CpuBackend, 2>, a: Activation) -> Tensor<CpuBackend, 2> {
    match a {
        Activation::Relu => activation::relu(x),
        Activation::Gelu => activation::gelu(x),
        Activation::Silu => activation::silu(x),
    }
}
fn to_vec(x: Tensor<CpuBackend, 2>) -> Result<Vec<f32>> {
    x.into_data()
        .to_vec::<f32>()
        .map_err(|e| MidasError::Policy(format!("{e:?}")))
}

fn train_burn_heads(
    model: &mut BurnNeuralModel,
    batch: &crate::learning::PpoTrainingBatch,
    config: &LearningConfig,
) -> Result<crate::learning::TrainingMetrics> {
    let saved_runtime = model.runtime.clone();
    let saved_lifetime = model.lifetime.clone();
    let saved_optimizer = model.optimizer.clone();
    let saved_cached = model.cached.clone();
    let mut aggregate = crate::learning::TrainingMetrics::empty();
    let epochs = config.optimisation_epochs.max(1);
    let chunk_size = config.minibatch_size.max(config.sequence_length).max(1);
    let mut updates = 0usize;
    let result = (|| {
        for _ in 0..epochs {
            for samples in batch.samples.chunks(chunk_size) {
                let minibatch = crate::learning::PpoTrainingBatch {
                    samples: samples.to_vec(),
                    sequence_length: batch.sequence_length,
                };
                let metrics = train_burn_heads_inner(model, &minibatch, config)?;
                aggregate.policy_loss += metrics.policy_loss;
                aggregate.value_loss += metrics.value_loss;
                aggregate.entropy += metrics.entropy;
                aggregate.total_loss += metrics.total_loss;
                aggregate.gradient_norm += metrics.gradient_norm;
                aggregate.update_index = metrics.update_index;
                updates += 1;
            }
        }
        if updates > 0 {
            aggregate.rollout_size = batch.samples.len();
            let divisor = updates as f32;
            aggregate.policy_loss /= divisor;
            aggregate.value_loss /= divisor;
            aggregate.entropy /= divisor;
            aggregate.total_loss /= divisor;
            aggregate.gradient_norm /= divisor;
        }
        Ok(aggregate)
    })();
    model.runtime = saved_runtime;
    if result.is_err() {
        model.lifetime = saved_lifetime;
        model.optimizer = saved_optimizer;
        model.cached = saved_cached;
    }
    result
}

/// Applies a PPO update to the actor and critic heads. Hidden and recurrent
/// layers remain part of the stable runtime representation for now; updating
/// the heads still provides a genuine clipped policy/value optimization while
/// keeping the adapter independent of Burn's optimizer APIs.
fn train_burn_heads_inner(
    model: &mut BurnNeuralModel,
    batch: &crate::learning::PpoTrainingBatch,
    config: &LearningConfig,
) -> Result<crate::learning::TrainingMetrics> {
    if batch.samples.is_empty() {
        return Ok(crate::learning::TrainingMetrics::empty());
    }
    if config.learning_rate <= 0.0
        || !config.learning_rate.is_finite()
        || config.ppo_clip < 0.0
        || !config.ppo_clip.is_finite()
        || !config.gradient_clip.is_finite()
        || config.gradient_clip <= 0.0
        || config.critic_coefficient < 0.0
        || !config.critic_coefficient.is_finite()
        || config.entropy_coefficient < 0.0
        || !config.entropy_coefficient.is_finite()
    {
        return Err(MidasError::Config(
            "invalid PPO update configuration".into(),
        ));
    }
    let mut actor_gradient = vec![0.0_f32; model.architecture.actor_output_size];
    let mut critic_gradient = vec![0.0_f32; model.architecture.critic_output_size];
    let mut policy_loss = 0.0_f32;
    let mut value_loss = 0.0_f32;
    let mut entropy = 0.0_f32;
    let mut total_loss = 0.0_f32;

    for sample in &batch.samples {
        if sample.action_index >= model.architecture.actor_output_size
            || sample.recurrent_state.memory.len() != model.architecture.memory.size
        {
            return Err(MidasError::Policy(
                "PPO sample does not match neural architecture".into(),
            ));
        }
        model.runtime = sample.recurrent_state.clone();
        let output = model.forward(&NeuralInput {
            observation: &sample.observation,
        })?;
        let probabilities = categorical_probabilities(
            &output.logits,
            &sample.tradable,
            model.architecture.asset_count,
            sample.temperature,
        )?;
        let new_log_probability = probabilities[sample.action_index].ln();
        let sample_entropy = probabilities
            .iter()
            .filter(|probability| **probability > 0.0)
            .map(|probability| -probability * probability.ln())
            .sum::<f64>() as f32;
        let loss = crate::learning::ppo_loss(
            sample.old_log_probability,
            new_log_probability as f32,
            sample.advantage,
            output.value,
            sample.return_value,
            sample_entropy,
            config.ppo_clip,
            config.critic_coefficient,
            config.entropy_coefficient,
        )?;
        policy_loss += loss.policy_loss;
        value_loss += loss.value_loss;
        entropy += sample_entropy;
        total_loss += loss.total_loss;

        let ratio = loss.ratio;
        let unclipped = ratio * sample.advantage;
        let clipped = ratio.clamp(1.0 - config.ppo_clip, 1.0 + config.ppo_clip) * sample.advantage;
        let clipped_active = if sample.advantage >= 0.0 {
            clipped < unclipped
        } else {
            clipped > unclipped
        };
        if !clipped_active {
            let coefficient = ratio * sample.advantage;
            for (index, probability) in probabilities.iter().enumerate() {
                actor_gradient[index] += coefficient
                    * (*probability as f32
                        - if index == sample.action_index {
                            1.0
                        } else {
                            0.0
                        });
            }
        }
        // The entropy derivative is the exact derivative of the categorical
        // entropy with respect to logits; it is added to the actor gradient.
        if config.entropy_coefficient > 0.0 {
            for (index, probability) in probabilities.iter().enumerate() {
                if *probability > 0.0 {
                    actor_gradient[index] += config.entropy_coefficient
                        * (*probability as f32)
                        * (probability.ln() as f32 + sample_entropy);
                }
            }
        }
        critic_gradient[0] +=
            2.0 * config.critic_coefficient * (output.value - sample.return_value);
    }

    let count = batch.samples.len() as f32;
    for gradient in actor_gradient.iter_mut().chain(critic_gradient.iter_mut()) {
        *gradient /= count;
        if !gradient.is_finite() {
            return Err(MidasError::Policy("non-finite PPO gradient".into()));
        }
    }
    let mut gradient_norm = actor_gradient
        .iter()
        .chain(&critic_gradient)
        .map(|gradient| gradient * gradient)
        .sum::<f32>()
        .sqrt();
    if !gradient_norm.is_finite() {
        return Err(MidasError::Policy("non-finite PPO gradient norm".into()));
    }
    if config.gradient_clip > 0.0 && gradient_norm > config.gradient_clip {
        let scale = config.gradient_clip / gradient_norm;
        for gradient in actor_gradient.iter_mut().chain(critic_gradient.iter_mut()) {
            *gradient *= scale;
        }
        gradient_norm = config.gradient_clip;
    }

    let mut next_optimizer = model.optimizer.clone();
    next_optimizer.step = next_optimizer.step.saturating_add(1);
    let momentum = 0.9_f32;
    for (index, gradient) in actor_gradient.iter().enumerate() {
        next_optimizer.actor_momentum[index] =
            momentum * next_optimizer.actor_momentum[index] + (1.0 - momentum) * gradient;
    }
    for (index, gradient) in critic_gradient.iter().enumerate() {
        next_optimizer.critic_momentum[index] =
            momentum * next_optimizer.critic_momentum[index] + (1.0 - momentum) * gradient;
    }
    let mut next_lifetime = model.lifetime.clone();
    for (bias, update) in next_lifetime
        .actor
        .bias
        .iter_mut()
        .zip(&next_optimizer.actor_momentum)
    {
        *bias -= config.learning_rate * update;
    }
    for (bias, update) in next_lifetime
        .critic
        .bias
        .iter_mut()
        .zip(&next_optimizer.critic_momentum)
    {
        *bias -= config.learning_rate * update;
    }
    next_lifetime.validate(&model.architecture)?;
    if next_lifetime
        .actor
        .bias
        .iter()
        .chain(&next_lifetime.critic.bias)
        .any(|value| !value.is_finite())
    {
        return Err(MidasError::Policy("non-finite PPO parameters".into()));
    }
    let device = Default::default();
    model.cached = BurnRuntimeParameters::from_lifetime(&next_lifetime, &device);
    model.lifetime = next_lifetime;
    model.optimizer = next_optimizer;
    Ok(crate::learning::TrainingMetrics {
        rollout_size: batch.samples.len(),
        policy_loss: policy_loss / count,
        value_loss: value_loss / count,
        entropy: entropy / count,
        total_loss: total_loss / count,
        gradient_norm,
        update_index: model.optimizer.step,
    })
}

#[derive(Clone, Debug, Eq, Serialize, Deserialize, PartialEq)]
pub enum PolicyBlueprint {
    Cash,
    Random,
    EqualWeight,
    BuyAndHold { target_asset: usize },
    Neural,
}
pub struct PolicyContext<'a> {
    pub tradable: &'a [bool],
    pub current_allocations: &'a [f64],
}
pub trait Policy: Send {
    fn propose_action(
        &mut self,
        _: &Observation,
        context: &PolicyContext<'_>,
    ) -> Result<ProposedAllocation>;
    fn snapshot(&self) -> Result<PolicySnapshot>;
    fn reset_episode_state(&mut self);
    fn policy_blueprint(&self) -> PolicyBlueprint;
    fn evaluate_action(
        &mut self,
        observation: &Observation,
        context: &PolicyContext<'_>,
        _stochastic: bool,
        _maximum_asset_allocation: f64,
    ) -> Result<PolicyEvaluation> {
        let proposed = self.propose_action(observation, context)?;
        Ok(PolicyEvaluation {
            proposed_allocation: proposed,
            action_sample: None,
            log_probability: None,
            entropy: 0.0,
            value_estimate: 0.0,
            raw_logits: Vec::new(),
            temperature: 1.0,
        })
    }
    fn apply_learning(
        &mut self,
        _batch: &crate::learning::PpoTrainingBatch,
        _config: &LearningConfig,
    ) -> Result<crate::learning::TrainingMetrics> {
        Err(MidasError::Policy(
            "policy does not expose trainable lifetime parameters".into(),
        ))
    }
    fn runtime_state(&self) -> Option<NeuralRuntimeState> {
        None
    }
    fn lifetime_parameters(&self) -> Result<Option<LifetimeParameters>> {
        Ok(None)
    }
    /// Fork policy state for a transactional simulation step.
    fn fork(&self) -> Result<Box<dyn Policy>> {
        Err(MidasError::Policy(
            "policy does not support transactional forking".into(),
        ))
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[allow(clippy::large_enum_variant)]
pub enum PolicySnapshot {
    Cash,
    Random {
        rng: ChaCha8Rng,
    },
    EqualWeight,
    BuyAndHold {
        initialized: bool,
        target_asset: usize,
    },
    Neural {
        genome: BrainGenome,
        state: Box<NeuralModelState>,
        rng: ChaCha8Rng,
        temperature: f32,
    },
}

pub struct NeuralPolicy {
    genome: BrainGenome,
    model: Box<dyn NeuralModel>,
    rng: ChaCha8Rng,
    temperature: f32,
}
impl NeuralPolicy {
    pub fn new(genome: BrainGenome, model: Box<dyn NeuralModel>) -> Self {
        Self::new_with_seed(genome, model, 0)
    }
    pub fn new_with_seed(genome: BrainGenome, model: Box<dyn NeuralModel>, seed: u64) -> Self {
        let temperature = genome.traits.action_temperature;
        Self {
            genome,
            model,
            rng: ChaCha8Rng::seed_from_u64(seed),
            temperature,
        }
    }
    pub(crate) fn with_rng(
        genome: BrainGenome,
        model: Box<dyn NeuralModel>,
        rng: ChaCha8Rng,
        temperature: f32,
    ) -> Self {
        Self {
            genome,
            model,
            rng,
            temperature,
        }
    }
}
impl Policy for NeuralPolicy {
    fn propose_action(
        &mut self,
        o: &Observation,
        c: &PolicyContext<'_>,
    ) -> Result<ProposedAllocation> {
        Ok(self.evaluate_action(o, c, false, 1.0)?.proposed_allocation)
    }
    fn evaluate_action(
        &mut self,
        o: &Observation,
        c: &PolicyContext<'_>,
        stochastic: bool,
        maximum_asset_allocation: f64,
    ) -> Result<PolicyEvaluation> {
        let out = self.model.forward(&NeuralInput { observation: o })?;
        let evaluation = categorical_evaluation(
            out,
            c.tradable,
            maximum_asset_allocation,
            stochastic,
            self.temperature,
            &mut self.rng,
        )?;
        let decay = self.genome.traits.temperature_decay;
        if !decay.is_finite() || decay < 0.0 {
            return Err(MidasError::Policy(
                "invalid action-temperature decay".into(),
            ));
        }
        self.temperature = (self.temperature * decay).max(f32::MIN_POSITIVE);
        Ok(evaluation)
    }
    fn apply_learning(
        &mut self,
        batch: &crate::learning::PpoTrainingBatch,
        config: &LearningConfig,
    ) -> Result<crate::learning::TrainingMetrics> {
        self.model.train_batch(batch, config)
    }
    fn runtime_state(&self) -> Option<NeuralRuntimeState> {
        self.model.runtime_state()
    }
    fn lifetime_parameters(&self) -> Result<Option<LifetimeParameters>> {
        self.model.lifetime_parameters().map(Some)
    }
    fn snapshot(&self) -> Result<PolicySnapshot> {
        Ok(PolicySnapshot::Neural {
            genome: self.genome.clone(),
            state: Box::new(self.model.snapshot_state()?),
            rng: self.rng.clone(),
            temperature: self.temperature,
        })
    }
    fn reset_episode_state(&mut self) {
        self.model.reset_memory();
        self.temperature = self.genome.traits.action_temperature;
    }
    fn policy_blueprint(&self) -> PolicyBlueprint {
        PolicyBlueprint::Neural
    }
    fn fork(&self) -> Result<Box<dyn Policy>> {
        Ok(Box::new(Self {
            genome: self.genome.clone(),
            model: self.model.fork()?,
            rng: self.rng.clone(),
            temperature: self.temperature,
        }))
    }
}

/// The training distribution is categorical over cash and each tradable
/// asset. An asset draw maps to a legal allocation of `maximum_allocation` in
/// that asset and the remainder in cash; a cash draw is all cash. This keeps
/// the sampled action and its probability in the same constrained action
/// space, unlike post-hoc clipping of a continuous simplex sample.
fn categorical_evaluation(
    output: NeuralOutput,
    tradable: &[bool],
    maximum_asset_allocation: f64,
    stochastic: bool,
    temperature: f32,
    rng: &mut ChaCha8Rng,
) -> Result<PolicyEvaluation> {
    if output.logits.len() != tradable.len() + 1
        || !output.value.is_finite()
        || !temperature.is_finite()
        || temperature <= 0.0
        || !maximum_asset_allocation.is_finite()
        || !(0.0..=1.0).contains(&maximum_asset_allocation)
    {
        return Err(MidasError::Policy(
            "invalid categorical policy output or action bounds".into(),
        ));
    }
    let probabilities =
        categorical_probabilities(&output.logits, tradable, tradable.len(), temperature)?;
    let entropy = probabilities
        .iter()
        .filter(|probability| **probability > 0.0)
        .map(|probability| -probability * probability.ln())
        .sum::<f64>() as f32;
    let selected = if stochastic {
        let draw = rng.r#gen::<f64>();
        let mut cumulative = 0.0;
        probabilities
            .iter()
            .position(|probability| {
                cumulative += *probability;
                draw < cumulative
            })
            .unwrap_or(0)
    } else {
        probabilities
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index)
            .unwrap_or(0)
    };
    let mut weights = vec![0.0; tradable.len() + 1];
    if selected == 0 {
        weights[0] = 1.0;
    } else {
        weights[0] = 1.0 - maximum_asset_allocation;
        weights[selected] = maximum_asset_allocation;
    }
    let proposed = ProposedAllocation::rebalance(weights);
    let log_probability = probabilities[selected].max(f64::MIN_POSITIVE).ln() as f32;
    Ok(PolicyEvaluation {
        proposed_allocation: proposed,
        action_sample: stochastic.then_some(PolicyActionSample {
            action_index: selected,
            log_probability,
            entropy,
        }),
        log_probability: stochastic.then_some(log_probability),
        entropy,
        value_estimate: output.value,
        raw_logits: output.logits,
        temperature,
    })
}

fn categorical_probabilities(
    logits: &[f32],
    tradable: &[bool],
    asset_count: usize,
    temperature: f32,
) -> Result<Vec<f64>> {
    if logits.len() != asset_count + 1 || tradable.len() != asset_count {
        return Err(MidasError::Policy(
            "categorical logits and tradability dimensions mismatch".into(),
        ));
    }
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(MidasError::Policy(
            "categorical temperature is invalid".into(),
        ));
    }
    let scaled = logits
        .iter()
        .enumerate()
        .map(|(index, value)| {
            if index > 0 && !tradable[index - 1] {
                f64::NEG_INFINITY
            } else {
                (*value as f64) / temperature as f64
            }
        })
        .collect::<Vec<_>>();
    if scaled.iter().any(|value| value.is_nan()) {
        return Err(MidasError::Policy("non-finite categorical logits".into()));
    }
    let peak = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut probabilities = scaled
        .iter()
        .map(|value| {
            if value.is_finite() {
                (*value - peak).exp()
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    let sum = probabilities.iter().sum::<f64>();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(MidasError::Policy(
            "categorical probabilities are invalid".into(),
        ));
    }
    for probability in &mut probabilities {
        *probability /= sum;
    }
    Ok(probabilities)
}

pub struct CashPolicy;
impl Policy for CashPolicy {
    fn propose_action(
        &mut self,
        _observation: &Observation,
        c: &PolicyContext<'_>,
    ) -> Result<ProposedAllocation> {
        let mut w = vec![0.0; c.tradable.len() + 1];
        w[0] = 1.0;
        Ok(ProposedAllocation::rebalance(w))
    }
    fn snapshot(&self) -> Result<PolicySnapshot> {
        Ok(PolicySnapshot::Cash)
    }
    fn reset_episode_state(&mut self) {}
    fn policy_blueprint(&self) -> PolicyBlueprint {
        PolicyBlueprint::Cash
    }
    fn fork(&self) -> Result<Box<dyn Policy>> {
        Ok(Box::new(CashPolicy))
    }
}
pub struct EqualWeightPolicy;
impl Policy for EqualWeightPolicy {
    fn propose_action(
        &mut self,
        observation: &Observation,
        c: &PolicyContext<'_>,
    ) -> Result<ProposedAllocation> {
        let n = c.tradable.iter().filter(|v| **v).count();
        if n == 0 {
            return CashPolicy.propose_action(observation, c);
        }
        let mut w = vec![0.0; c.tradable.len() + 1];
        for (i, t) in c.tradable.iter().enumerate() {
            if *t {
                w[i + 1] = 1.0 / n as f64
            }
        }
        Ok(ProposedAllocation::rebalance(w))
    }
    fn snapshot(&self) -> Result<PolicySnapshot> {
        Ok(PolicySnapshot::EqualWeight)
    }
    fn reset_episode_state(&mut self) {}
    fn policy_blueprint(&self) -> PolicyBlueprint {
        PolicyBlueprint::EqualWeight
    }
    fn fork(&self) -> Result<Box<dyn Policy>> {
        Ok(Box::new(EqualWeightPolicy))
    }
}
pub struct RandomPolicy {
    pub(crate) rng: ChaCha8Rng,
}
impl RandomPolicy {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }
}
impl Policy for RandomPolicy {
    fn propose_action(
        &mut self,
        _: &Observation,
        c: &PolicyContext<'_>,
    ) -> Result<ProposedAllocation> {
        Ok(ProposedAllocation::rebalance(
            (0..=c.tradable.len())
                .map(|_| self.rng.gen_range(0.0001..1.0))
                .collect(),
        ))
    }
    fn snapshot(&self) -> Result<PolicySnapshot> {
        Ok(PolicySnapshot::Random {
            rng: self.rng.clone(),
        })
    }
    fn reset_episode_state(&mut self) {}
    fn policy_blueprint(&self) -> PolicyBlueprint {
        PolicyBlueprint::Random
    }
    fn fork(&self) -> Result<Box<dyn Policy>> {
        Ok(Box::new(Self {
            rng: self.rng.clone(),
        }))
    }
}
pub struct BuyAndHoldPolicy {
    pub(crate) initialized: bool,
    pub(crate) target_asset: usize,
}
impl BuyAndHoldPolicy {
    pub fn new(target_asset: usize) -> Self {
        Self {
            initialized: false,
            target_asset,
        }
    }
}
impl Policy for BuyAndHoldPolicy {
    fn propose_action(
        &mut self,
        _: &Observation,
        c: &PolicyContext<'_>,
    ) -> Result<ProposedAllocation> {
        if self.initialized {
            return Ok(ProposedAllocation::hold(c.current_allocations.to_vec()));
        }
        self.initialized = true;
        let mut w = vec![0.0; c.tradable.len() + 1];
        if c.tradable.get(self.target_asset).copied().unwrap_or(false) {
            w[self.target_asset + 1] = 1.0
        } else {
            w[0] = 1.0
        }
        Ok(ProposedAllocation::rebalance(w))
    }
    fn snapshot(&self) -> Result<PolicySnapshot> {
        Ok(PolicySnapshot::BuyAndHold {
            initialized: self.initialized,
            target_asset: self.target_asset,
        })
    }
    fn reset_episode_state(&mut self) {
        self.initialized = false
    }
    fn policy_blueprint(&self) -> PolicyBlueprint {
        PolicyBlueprint::BuyAndHold {
            target_asset: self.target_asset,
        }
    }
    fn fork(&self) -> Result<Box<dyn Policy>> {
        Ok(Box::new(Self {
            initialized: self.initialized,
            target_asset: self.target_asset,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::{Activation, EvolvableTraitsConfig, LearningConfig, LearningKind},
        learning::{PpoTrainingBatch, PpoTrainingSample},
    };

    fn architecture() -> BrainArchitecture {
        BrainArchitecture {
            version: 1,
            observation_schema_hash: "schema".into(),
            observation_size: 2,
            asset_count: 1,
            hidden_layers: vec![2],
            memory: MemoryArchitecture { size: 1 },
            actor_output_size: 2,
            critic_output_size: 1,
            activation: Activation::Relu,
        }
    }

    fn learning_config() -> LearningConfig {
        LearningConfig {
            implementation: LearningKind::Ppo,
            learning_rate: 0.01,
            gamma: 0.99,
            gae_lambda: 0.95,
            ppo_clip: 0.2,
            critic_coefficient: 0.5,
            entropy_coefficient: 0.01,
            gradient_clip: 1.0,
            rollout_length: 1,
            minibatch_size: 1,
            sequence_length: 1,
            normalize_advantages: false,
            optimisation_epochs: 1,
        }
    }

    #[test]
    fn burn_training_changes_lifetime_parameters_without_changing_genome() {
        let architecture = architecture();
        let traits = EvolvableTraitsConfig::default();
        let genome = BrainGenome::random(&architecture, &traits, 7).unwrap();
        let before_genome = genome.clone();
        let device = ComputeDevice::CpuNdArray {
            device: "default".into(),
        };
        let mut model = BurnNeuralModelFactory
            .build(&genome, &architecture, &device)
            .unwrap();
        let before = model.lifetime_parameters().unwrap();
        let batch = PpoTrainingBatch {
            samples: vec![PpoTrainingSample {
                observation: Observation {
                    schema_hash: "schema".into(),
                    values: vec![0.2, -0.1],
                },
                action_index: 0,
                old_log_probability: -std::f32::consts::LN_2,
                advantage: 1.0,
                return_value: 1.0,
                value_estimate: 0.0,
                recurrent_state: NeuralRuntimeState {
                    memory: vec![0.0],
                    inference_steps: 0,
                },
                tradable: vec![true],
                temperature: 1.0,
            }],
            sequence_length: 1,
        };
        let metrics = model.train_batch(&batch, &learning_config()).unwrap();
        assert!(metrics.rollout_size > 0);
        assert!(metrics.gradient_norm <= 1.0 + 1e-6);
        assert_ne!(before, model.lifetime_parameters().unwrap());
        assert_eq!(before_genome, genome);
    }

    #[test]
    fn genome_validation_rejects_hidden_layout_changes() {
        let architecture = architecture();
        let mut genome =
            BrainGenome::random(&architecture, &EvolvableTraitsConfig::default(), 17).unwrap();
        genome.hidden.pop();
        assert!(genome.validate(&architecture).is_err());
    }

    #[test]
    fn categorical_policy_is_masked_and_seed_reproducible() {
        let output = NeuralOutput {
            logits: vec![0.0, 2.0, 1.0],
            value: 0.0,
        };
        let mut deterministic_rng = ChaCha8Rng::seed_from_u64(1);
        let deterministic = categorical_evaluation(
            output.clone(),
            &[true, false],
            0.5,
            false,
            1.0,
            &mut deterministic_rng,
        )
        .unwrap();
        assert!(
            (deterministic
                .proposed_allocation
                .weights()
                .iter()
                .sum::<f64>()
                - 1.0)
                .abs()
                < 1e-9
        );
        assert_eq!(deterministic.proposed_allocation.weights()[2], 0.0);
        let mut left_rng = ChaCha8Rng::seed_from_u64(5);
        let mut right_rng = ChaCha8Rng::seed_from_u64(5);
        let left =
            categorical_evaluation(output.clone(), &[true, true], 0.5, true, 1.0, &mut left_rng)
                .unwrap();
        let right =
            categorical_evaluation(output, &[true, true], 0.5, true, 1.0, &mut right_rng).unwrap();
        assert_eq!(left, right);
    }
}
