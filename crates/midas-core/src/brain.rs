use crate::{
    MidasError, Result,
    action::ProposedAllocation,
    config::{
        ActionDistributionConfig, ActionDistributionKind, Activation, AssetOrder, BrainConfig,
        EvolvableTraitsConfig, LearningConfig,
    },
    observation::Observation,
};
use burn::{
    backend::NdArray,
    tensor::{Tensor, TensorData, activation},
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Beta, Distribution};
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
    pub action_distribution: ActionDistributionConfig,
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
            actor_output_size: asset_count * 2,
            critic_output_size: 1,
            activation: c.activation,
            action_distribution: c.action_distribution.clone(),
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
            genome_format_version: 3,
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
        if self.genome_format_version != 3 || &self.architecture != a {
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
    pub active_asset_indices: Vec<usize>,
    pub latent_samples: Vec<f32>,
    pub allocation: Vec<f64>,
    pub maximum_asset_allocation: f32,
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
pub struct DenseOptimizerState {
    pub weight_momentum: Vec<f32>,
    pub bias_momentum: Vec<f32>,
}
impl DenseOptimizerState {
    fn empty(inputs: usize, outputs: usize) -> Self {
        Self {
            weight_momentum: vec![0.0; inputs * outputs],
            bias_momentum: vec![0.0; outputs],
        }
    }

    fn validate(&self, inputs: usize, outputs: usize, label: &str) -> Result<()> {
        if self.weight_momentum.len() != inputs * outputs
            || self.bias_momentum.len() != outputs
            || self
                .weight_momentum
                .iter()
                .chain(&self.bias_momentum)
                .any(|value| !value.is_finite())
        {
            return Err(MidasError::Checkpoint(format!(
                "neural optimizer {label} state shape or finiteness mismatch"
            )));
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct OptimizerState {
    pub step: u64,
    pub actor: DenseOptimizerState,
    pub critic: DenseOptimizerState,
}
impl OptimizerState {
    pub fn empty(architecture: &BrainArchitecture) -> Self {
        Self {
            step: 0,
            actor: DenseOptimizerState::empty(
                architecture.memory.size,
                architecture.actor_output_size,
            ),
            critic: DenseOptimizerState::empty(
                architecture.memory.size,
                architecture.critic_output_size,
            ),
        }
    }

    fn validate(&self, architecture: &BrainArchitecture) -> Result<()> {
        self.actor.validate(
            architecture.memory.size,
            architecture.actor_output_size,
            "actor",
        )?;
        self.critic.validate(
            architecture.memory.size,
            architecture.critic_output_size,
            "critic",
        )?;
        Ok(())
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
    fn action_distribution(&self) -> ActionDistributionConfig {
        ActionDistributionConfig::default()
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

    fn with_updated_heads(
        &self,
        lifetime: &LifetimeParameters,
        d: &<CpuBackend as burn::tensor::backend::Backend>::Device,
    ) -> Self {
        Self {
            hidden: self.hidden.clone(),
            recurrent: self.recurrent.clone(),
            actor: runtime_dense(&lifetime.actor, d),
            critic: runtime_dense(&lifetime.critic, d),
            builds: self.builds.saturating_add(1),
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
        if runtime.memory.len() != a.memory.size
            || runtime.memory.iter().any(|value| !value.is_finite())
        {
            return Err(MidasError::Checkpoint(
                "neural runtime memory size or finiteness mismatch".into(),
            ));
        }
        optimizer.validate(a)?;
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
struct EncodedState {
    representation: Vec<f32>,
    next_runtime_state: NeuralRuntimeState,
}
impl NeuralModel for BurnNeuralModel {
    fn forward(&mut self, input: &NeuralInput<'_>) -> Result<NeuralOutput> {
        let encoded = self.encode_observation(input.observation, &self.runtime)?;
        let output = self.evaluate_representation(&encoded.representation)?;
        self.runtime = encoded.next_runtime_state;
        Ok(output)
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
    fn action_distribution(&self) -> ActionDistributionConfig {
        self.architecture.action_distribution.clone()
    }
}
impl BurnNeuralModel {
    fn encode_observation(
        &self,
        observation: &Observation,
        recurrent_state: &NeuralRuntimeState,
    ) -> Result<EncodedState> {
        if observation.schema_hash != self.architecture.observation_schema_hash
            || observation.values.len() != self.architecture.observation_size
            || observation.values.iter().any(|value| !value.is_finite())
            || recurrent_state.memory.len() != self.architecture.memory.size
            || recurrent_state
                .memory
                .iter()
                .any(|value| !value.is_finite())
        {
            return Err(MidasError::Policy(
                "invalid neural observation or recurrent state".into(),
            ));
        }
        let device = Default::default();
        let mut x = Tensor::<CpuBackend, 2>::from_data(
            TensorData::new(
                observation.values.clone(),
                [1, self.architecture.observation_size],
            ),
            &device,
        );
        for l in &self.cached.hidden {
            x = activate(dense_runtime(x, l), self.architecture.activation);
        }
        let memory = Tensor::<CpuBackend, 2>::from_data(
            TensorData::new(
                recurrent_state.memory.clone(),
                [1, self.architecture.memory.size],
            ),
            &device,
        );
        let next = activation::silu(dense_runtime(
            Tensor::cat(vec![x, memory], 1),
            &self.cached.recurrent,
        ));
        let representation = to_vec(next)?;
        if representation.len() != self.architecture.memory.size
            || representation.iter().any(|value| !value.is_finite())
        {
            return Err(MidasError::Policy(
                "non-finite or incorrectly sized recurrent representation".into(),
            ));
        }
        Ok(EncodedState {
            representation: representation.clone(),
            next_runtime_state: NeuralRuntimeState {
                memory: representation.clone(),
                inference_steps: recurrent_state.inference_steps.saturating_add(1),
            },
        })
    }

    fn evaluate_representation(&self, representation: &[f32]) -> Result<NeuralOutput> {
        if representation.len() != self.architecture.memory.size
            || representation.iter().any(|value| !value.is_finite())
        {
            return Err(MidasError::Policy(
                "invalid encoded neural representation".into(),
            ));
        }
        let device = Default::default();
        let state = Tensor::<CpuBackend, 2>::from_data(
            TensorData::new(representation.to_vec(), [1, self.architecture.memory.size]),
            &device,
        );
        let logits = to_vec(dense_runtime(state.clone(), &self.cached.actor))?;
        let critic = to_vec(dense_runtime(state, &self.cached.critic))?;
        if critic.len() != self.architecture.critic_output_size
            || self.architecture.critic_output_size != 1
        {
            return Err(MidasError::Policy(
                "critic output must contain exactly one value".into(),
            ));
        }
        let value = critic[0];
        if logits.iter().any(|v| !v.is_finite()) || !value.is_finite() {
            return Err(MidasError::Policy("non-finite neural output".into()));
        }
        Ok(NeuralOutput { logits, value })
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
                aggregate.approximate_kl += metrics.approximate_kl;
                aggregate.clip_fraction += metrics.clip_fraction;
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
            aggregate.approximate_kl /= divisor;
            aggregate.clip_fraction /= divisor;
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
    let actor_inputs = model.architecture.memory.size;
    let actor_outputs = model.architecture.actor_output_size;
    let critic_inputs = model.architecture.memory.size;
    let critic_outputs = model.architecture.critic_output_size;
    if critic_outputs != 1 {
        return Err(MidasError::Policy(
            "PPO head training requires a one-output critic".into(),
        ));
    }
    let mut actor_weight_gradient = vec![0.0_f32; actor_inputs * actor_outputs];
    let mut actor_bias_gradient = vec![0.0_f32; actor_outputs];
    let mut critic_weight_gradient = vec![0.0_f32; critic_inputs * critic_outputs];
    let mut critic_bias_gradient = vec![0.0_f32; critic_outputs];
    let mut policy_loss = 0.0_f32;
    let mut value_loss = 0.0_f32;
    let mut entropy = 0.0_f32;
    let mut total_loss = 0.0_f32;
    let mut actor_count = 0usize;
    let mut critic_count = 0usize;
    let mut approximate_kl = 0.0_f32;
    let mut clipped_samples = 0usize;

    for sample in &batch.samples {
        if sample.recurrent_state.memory.len() != model.architecture.memory.size {
            return Err(MidasError::Policy(
                "PPO sample does not match neural architecture".into(),
            ));
        }
        let encoded = model.encode_observation(&sample.observation, &sample.recurrent_state)?;
        let output = model.evaluate_representation(&encoded.representation)?;
        if let Some(action_sample) = sample.action_sample.as_ref() {
            let old_log_probability = sample.old_log_probability.ok_or_else(|| {
                MidasError::Policy("actor sample is missing its old log probability".into())
            })?;
            let advantage = sample
                .advantage
                .ok_or_else(|| MidasError::Policy("actor sample is missing advantage".into()))?;
            let temperature = sample
                .temperature
                .ok_or_else(|| MidasError::Policy("actor sample is missing temperature".into()))?;
            let distribution = evaluate_bounded_stick_breaking_sample(
                &output.logits,
                &sample.tradable,
                action_sample,
                temperature,
                &model.architecture.action_distribution,
            )?;
            let new_log_probability = distribution.log_probability;
            let sample_entropy = distribution.entropy;
            let target_return = sample.return_value.unwrap_or(output.value);
            let loss = crate::learning::ppo_loss(
                old_log_probability,
                new_log_probability,
                advantage,
                output.value,
                target_return,
                sample_entropy,
                config.ppo_clip,
                config.critic_coefficient,
                config.entropy_coefficient,
            )?;
            policy_loss += loss.policy_loss;
            entropy += sample_entropy;
            total_loss += loss.policy_loss - config.entropy_coefficient * sample_entropy;
            actor_count += 1;

            let ratio = loss.ratio;
            let unclipped = ratio * advantage;
            let clipped = ratio.clamp(1.0 - config.ppo_clip, 1.0 + config.ppo_clip) * advantage;
            let clipped_active = if advantage >= 0.0 {
                clipped < unclipped
            } else {
                clipped > unclipped
            };
            approximate_kl += old_log_probability - new_log_probability;
            if clipped_active {
                clipped_samples += 1;
            }
            let mut actor_output_gradient = vec![0.0_f32; actor_outputs];
            if !clipped_active {
                let coefficient = ratio * advantage;
                for decision in &distribution.decisions {
                    actor_output_gradient[decision.actor_alpha_index] +=
                        -coefficient * decision.d_log_probability_d_raw_alpha;
                    actor_output_gradient[decision.actor_beta_index] +=
                        -coefficient * decision.d_log_probability_d_raw_beta;
                }
            }
            // The entropy derivative is the exact derivative of each Beta
            // factor, chained through the bounded softplus shape transform.
            if config.entropy_coefficient > 0.0 {
                for decision in &distribution.decisions {
                    actor_output_gradient[decision.actor_alpha_index] +=
                        -config.entropy_coefficient * decision.d_entropy_d_raw_alpha;
                    actor_output_gradient[decision.actor_beta_index] +=
                        -config.entropy_coefficient * decision.d_entropy_d_raw_beta;
                }
            }
            accumulate_dense_gradient(
                &encoded.representation,
                &actor_output_gradient,
                &mut actor_weight_gradient,
                &mut actor_bias_gradient,
            )?;
        }
        if let Some(target_return) = sample.return_value {
            let critic_output_gradient =
                [2.0 * config.critic_coefficient * (output.value - target_return)];
            let critic_loss = (output.value - target_return).powi(2);
            if !critic_loss.is_finite() {
                return Err(MidasError::Policy("non-finite critic loss".into()));
            }
            value_loss += critic_loss;
            total_loss += config.critic_coefficient * critic_loss;
            critic_count += 1;
            accumulate_dense_gradient(
                &encoded.representation,
                &critic_output_gradient,
                &mut critic_weight_gradient,
                &mut critic_bias_gradient,
            )?;
        }
    }

    let actor_divisor = actor_count.max(1) as f32;
    let critic_divisor = critic_count.max(1) as f32;
    for gradient in actor_weight_gradient
        .iter_mut()
        .chain(actor_bias_gradient.iter_mut())
    {
        *gradient /= actor_divisor;
        if !gradient.is_finite() {
            return Err(MidasError::Policy("non-finite PPO gradient".into()));
        }
    }
    for gradient in critic_weight_gradient
        .iter_mut()
        .chain(critic_bias_gradient.iter_mut())
    {
        *gradient /= critic_divisor;
        if !gradient.is_finite() {
            return Err(MidasError::Policy("non-finite PPO gradient".into()));
        }
    }
    let mut gradient_norm = actor_weight_gradient
        .iter()
        .chain(&actor_bias_gradient)
        .chain(&critic_weight_gradient)
        .chain(&critic_bias_gradient)
        .map(|gradient| gradient * gradient)
        .sum::<f32>()
        .sqrt();
    if !gradient_norm.is_finite() {
        return Err(MidasError::Policy("non-finite PPO gradient norm".into()));
    }
    if config.gradient_clip > 0.0 && gradient_norm > config.gradient_clip {
        let scale = config.gradient_clip / gradient_norm;
        for gradient in actor_weight_gradient
            .iter_mut()
            .chain(actor_bias_gradient.iter_mut())
            .chain(critic_weight_gradient.iter_mut())
            .chain(critic_bias_gradient.iter_mut())
        {
            *gradient *= scale;
        }
        gradient_norm = config.gradient_clip;
    }

    let mut next_optimizer = model.optimizer.clone();
    next_optimizer.step = next_optimizer.step.saturating_add(1);
    let momentum = 0.9_f32;
    for (state, gradient) in next_optimizer
        .actor
        .weight_momentum
        .iter_mut()
        .zip(&actor_weight_gradient)
    {
        *state = momentum * *state + (1.0 - momentum) * gradient;
    }
    for (state, gradient) in next_optimizer
        .actor
        .bias_momentum
        .iter_mut()
        .zip(&actor_bias_gradient)
    {
        *state = momentum * *state + (1.0 - momentum) * gradient;
    }
    for (state, gradient) in next_optimizer
        .critic
        .weight_momentum
        .iter_mut()
        .zip(&critic_weight_gradient)
    {
        *state = momentum * *state + (1.0 - momentum) * gradient;
    }
    for (state, gradient) in next_optimizer
        .critic
        .bias_momentum
        .iter_mut()
        .zip(&critic_bias_gradient)
    {
        *state = momentum * *state + (1.0 - momentum) * gradient;
    }
    let mut next_lifetime = model.lifetime.clone();
    for (weight, update) in next_lifetime
        .actor
        .weights
        .iter_mut()
        .zip(&next_optimizer.actor.weight_momentum)
    {
        *weight -= config.learning_rate * update;
    }
    for (bias, update) in next_lifetime
        .actor
        .bias
        .iter_mut()
        .zip(&next_optimizer.actor.bias_momentum)
    {
        *bias -= config.learning_rate * update;
    }
    for (weight, update) in next_lifetime
        .critic
        .weights
        .iter_mut()
        .zip(&next_optimizer.critic.weight_momentum)
    {
        *weight -= config.learning_rate * update;
    }
    for (bias, update) in next_lifetime
        .critic
        .bias
        .iter_mut()
        .zip(&next_optimizer.critic.bias_momentum)
    {
        *bias -= config.learning_rate * update;
    }
    next_lifetime.validate(&model.architecture)?;
    next_optimizer.validate(&model.architecture)?;
    let device = Default::default();
    model.cached = model.cached.with_updated_heads(&next_lifetime, &device);
    model.lifetime = next_lifetime;
    model.optimizer = next_optimizer;
    Ok(crate::learning::TrainingMetrics {
        rollout_size: batch.samples.len(),
        policy_loss: policy_loss / actor_divisor,
        value_loss: value_loss / critic_divisor,
        entropy: entropy / actor_divisor,
        total_loss: total_loss / (actor_count.max(critic_count).max(1) as f32),
        gradient_norm,
        approximate_kl: approximate_kl / actor_divisor,
        clip_fraction: clipped_samples as f32 / actor_divisor,
        update_index: model.optimizer.step,
    })
}

fn accumulate_dense_gradient(
    representation: &[f32],
    output_gradient: &[f32],
    weight_gradient: &mut [f32],
    bias_gradient: &mut [f32],
) -> Result<()> {
    if weight_gradient.len() != representation.len() * output_gradient.len()
        || bias_gradient.len() != output_gradient.len()
    {
        return Err(MidasError::Policy(
            "dense gradient dimensions do not match head dimensions".into(),
        ));
    }
    for (output, gradient) in output_gradient.iter().enumerate() {
        if !gradient.is_finite() {
            return Err(MidasError::Policy(
                "non-finite dense output gradient".into(),
            ));
        }
        bias_gradient[output] += *gradient;
        for (input, value) in representation.iter().enumerate() {
            if !value.is_finite() {
                return Err(MidasError::Policy("non-finite dense representation".into()));
            }
            weight_gradient[input * output_gradient.len() + output] += value * gradient;
        }
    }
    if weight_gradient
        .iter()
        .chain(bias_gradient.iter())
        .any(|value| !value.is_finite())
    {
        return Err(MidasError::Policy("non-finite dense gradient".into()));
    }
    Ok(())
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
        let order_offset = self
            .model
            .runtime_state()
            .map(|state| state.inference_steps as usize)
            .unwrap_or(0);
        let out = self.model.forward(&NeuralInput { observation: o })?;
        let evaluation = bounded_stick_breaking_evaluation_with_order(
            out,
            c.tradable,
            maximum_asset_allocation,
            stochastic,
            self.temperature,
            &mut self.rng,
            &self.model.action_distribution(),
            order_offset,
        )?;
        // Exploration is part of stochastic acting only. Deterministic
        // evaluation and critic bootstrap must be side-effect free so merely
        // asking for a diagnostic action cannot consume the temperature
        // schedule.
        if stochastic && evaluation.action_sample.is_some() {
            let decay = self.genome.traits.temperature_decay;
            if !decay.is_finite() || decay < 0.0 {
                return Err(MidasError::Policy(
                    "invalid action-temperature decay".into(),
                ));
            }
            self.temperature = (self.temperature * decay).max(f32::MIN_POSITIVE);
        }
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

/// A bounded stick-breaking portfolio distribution. The PPO action is the
/// vector of latent Beta samples, not the final portfolio weights. For each
/// tradable asset in stable asset order, `weight = cap * u`; cash receives the
/// residual. This makes the concentration cap part of the action mapping and
/// leaves the joint log probability as the sum of the exact Beta log densities.
#[derive(Clone, Debug)]
struct BetaDecision {
    actor_alpha_index: usize,
    actor_beta_index: usize,
    alpha: f64,
    beta: f64,
    latent: f64,
    d_log_probability_d_raw_alpha: f32,
    d_log_probability_d_raw_beta: f32,
    d_entropy_d_raw_alpha: f32,
    d_entropy_d_raw_beta: f32,
}

#[derive(Clone, Debug)]
struct StickBreakingEvaluation {
    log_probability: f32,
    entropy: f32,
    decisions: Vec<BetaDecision>,
}

#[cfg(test)]
fn bounded_stick_breaking_evaluation(
    output: NeuralOutput,
    tradable: &[bool],
    maximum_asset_allocation: f64,
    stochastic: bool,
    temperature: f32,
    rng: &mut ChaCha8Rng,
    config: &ActionDistributionConfig,
) -> Result<PolicyEvaluation> {
    bounded_stick_breaking_evaluation_with_order(
        output,
        tradable,
        maximum_asset_allocation,
        stochastic,
        temperature,
        rng,
        config,
        0,
    )
}

#[allow(clippy::too_many_arguments)]
fn bounded_stick_breaking_evaluation_with_order(
    output: NeuralOutput,
    tradable: &[bool],
    maximum_asset_allocation: f64,
    stochastic: bool,
    temperature: f32,
    rng: &mut ChaCha8Rng,
    config: &ActionDistributionConfig,
    order_offset: usize,
) -> Result<PolicyEvaluation> {
    validate_distribution_inputs(
        &output,
        tradable,
        maximum_asset_allocation,
        temperature,
        config,
    )?;
    let active_asset_indices = ordered_asset_indices(tradable, config, order_offset)?;
    let latents = if stochastic {
        sample_latents(
            &output.logits,
            &active_asset_indices,
            temperature,
            config,
            rng,
        )?
    } else {
        deterministic_latents(&output.logits, &active_asset_indices, temperature, config)?
    };
    let (weights, decisions) = evaluate_latents(
        &output.logits,
        tradable,
        &active_asset_indices,
        &latents,
        maximum_asset_allocation,
        temperature,
        config,
    )?;
    let log_probability = decisions
        .iter()
        .map(|decision| beta_log_probability(decision.alpha, decision.beta, decision.latent))
        .sum::<Result<f64>>()? as f32;
    let entropy = decisions
        .iter()
        .map(|decision| beta_entropy(decision.alpha, decision.beta))
        .sum::<Result<f64>>()? as f32;
    let proposed = ProposedAllocation::rebalance(weights.clone());
    Ok(PolicyEvaluation {
        proposed_allocation: proposed,
        action_sample: stochastic.then_some(PolicyActionSample {
            active_asset_indices,
            latent_samples: latents.iter().map(|value| *value as f32).collect(),
            allocation: weights,
            maximum_asset_allocation: maximum_asset_allocation as f32,
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

fn validate_distribution_inputs(
    output: &NeuralOutput,
    tradable: &[bool],
    maximum_asset_allocation: f64,
    temperature: f32,
    config: &ActionDistributionConfig,
) -> Result<()> {
    if !matches!(config.kind, ActionDistributionKind::BoundedStickBreaking)
        || !matches!(
            config.asset_order,
            AssetOrder::Stable | AssetOrder::EpisodeRotated
        )
        || output.logits.len() != tradable.len() * 2
        || !output.value.is_finite()
        || output.logits.iter().any(|value| !value.is_finite())
        || !temperature.is_finite()
        || temperature <= 0.0
        || !maximum_asset_allocation.is_finite()
        || !(0.0..=1.0).contains(&maximum_asset_allocation)
        || maximum_asset_allocation == 0.0
        || !config.minimum_shape.is_finite()
        || !config.maximum_shape.is_finite()
        || config.minimum_shape <= 0.0
        || config.maximum_shape <= config.minimum_shape
        || !config.latent_epsilon.is_finite()
        || !(0.0..0.5).contains(&config.latent_epsilon)
        || config.latent_epsilon == 0.0
    {
        return Err(MidasError::Policy(
            "invalid bounded stick-breaking output or action bounds".into(),
        ));
    }
    Ok(())
}

fn active_asset_indices(tradable: &[bool]) -> Result<Vec<usize>> {
    Ok(tradable
        .iter()
        .enumerate()
        .filter_map(|(index, is_tradable)| is_tradable.then_some(index))
        .collect())
}

fn ordered_asset_indices(
    tradable: &[bool],
    config: &ActionDistributionConfig,
    order_offset: usize,
) -> Result<Vec<usize>> {
    let mut indices = active_asset_indices(tradable)?;
    if matches!(config.asset_order, AssetOrder::EpisodeRotated) && !indices.is_empty() {
        let length = indices.len();
        indices.rotate_left(order_offset % length);
    }
    Ok(indices)
}

fn sample_latents(
    logits: &[f32],
    active_asset_indices: &[usize],
    temperature: f32,
    config: &ActionDistributionConfig,
    rng: &mut ChaCha8Rng,
) -> Result<Vec<f64>> {
    active_asset_indices
        .iter()
        .map(|asset_index| {
            let (alpha, beta, _, _) = beta_shapes(
                logits[asset_index * 2],
                logits[asset_index * 2 + 1],
                temperature,
                config,
            )?;
            let distribution = Beta::new(alpha, beta).map_err(|error| {
                MidasError::Policy(format!("invalid Beta action parameters: {error}"))
            })?;
            let sample = distribution.sample(rng).clamp(
                config.latent_epsilon as f64,
                1.0 - config.latent_epsilon as f64,
            );
            if !sample.is_finite() {
                return Err(MidasError::Policy(
                    "non-finite bounded stick-breaking sample".into(),
                ));
            }
            Ok(sample)
        })
        .collect()
}

fn deterministic_latents(
    logits: &[f32],
    active_asset_indices: &[usize],
    temperature: f32,
    config: &ActionDistributionConfig,
) -> Result<Vec<f64>> {
    active_asset_indices
        .iter()
        .map(|asset_index| {
            let (alpha, beta, _, _) = beta_shapes(
                logits[asset_index * 2],
                logits[asset_index * 2 + 1],
                temperature,
                config,
            )?;
            Ok((alpha / (alpha + beta)).clamp(
                config.latent_epsilon as f64,
                1.0 - config.latent_epsilon as f64,
            ))
        })
        .collect()
}

fn evaluate_latents(
    logits: &[f32],
    tradable: &[bool],
    active_asset_indices: &[usize],
    latents: &[f64],
    maximum_asset_allocation: f64,
    temperature: f32,
    config: &ActionDistributionConfig,
) -> Result<(Vec<f64>, Vec<BetaDecision>)> {
    if active_asset_indices.len() != latents.len() || logits.len() != tradable.len() * 2 {
        return Err(MidasError::Policy(
            "bounded stick-breaking latent dimensions mismatch".into(),
        ));
    }
    let mut weights = vec![0.0; tradable.len() + 1];
    let mut remaining = 1.0_f64;
    let mut decisions = Vec::with_capacity(active_asset_indices.len());
    for (position, asset_index) in active_asset_indices.iter().copied().enumerate() {
        let latent = latents[position];
        if !latent.is_finite()
            || latent < config.latent_epsilon as f64
            || latent > 1.0 - config.latent_epsilon as f64
        {
            return Err(MidasError::Policy(
                "bounded stick-breaking latent is outside the supported interior".into(),
            ));
        }
        let (alpha, beta, d_alpha_d_raw, d_beta_d_raw) = beta_shapes(
            logits[asset_index * 2],
            logits[asset_index * 2 + 1],
            temperature,
            config,
        )?;
        let cap = maximum_asset_allocation.min(remaining);
        let weight = cap * latent;
        if !weight.is_finite() || weight < 0.0 || weight > cap + 1e-12 {
            return Err(MidasError::Policy(
                "bounded stick-breaking allocation is invalid".into(),
            ));
        }
        weights[asset_index + 1] = weight;
        remaining -= weight;
        let log_gradient_alpha = beta_log_probability_derivative_alpha(alpha, beta, latent)?;
        let log_gradient_beta = beta_log_probability_derivative_beta(alpha, beta, latent)?;
        let entropy_gradient_alpha = beta_entropy_derivative_alpha(alpha, beta)?;
        let entropy_gradient_beta = beta_entropy_derivative_beta(alpha, beta)?;
        decisions.push(BetaDecision {
            actor_alpha_index: asset_index * 2,
            actor_beta_index: asset_index * 2 + 1,
            alpha,
            beta,
            latent,
            d_log_probability_d_raw_alpha: (log_gradient_alpha * d_alpha_d_raw) as f32,
            d_log_probability_d_raw_beta: (log_gradient_beta * d_beta_d_raw) as f32,
            d_entropy_d_raw_alpha: (entropy_gradient_alpha * d_alpha_d_raw) as f32,
            d_entropy_d_raw_beta: (entropy_gradient_beta * d_beta_d_raw) as f32,
        });
    }
    weights[0] = remaining;
    let total = weights.iter().sum::<f64>();
    if !total.is_finite() || total <= 0.0 {
        return Err(MidasError::Policy(
            "bounded stick-breaking weights are non-finite".into(),
        ));
    }
    let drift = 1.0 - total;
    if drift.abs() > 1e-10 {
        weights[0] += drift;
    }
    if weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight < 0.0)
        || (weights.iter().sum::<f64>() - 1.0).abs() > 1e-9
    {
        return Err(MidasError::Policy(
            "bounded stick-breaking weights failed simplex validation".into(),
        ));
    }
    Ok((weights, decisions))
}

fn evaluate_bounded_stick_breaking_sample(
    logits: &[f32],
    tradable: &[bool],
    sample: &PolicyActionSample,
    temperature: f32,
    config: &ActionDistributionConfig,
) -> Result<StickBreakingEvaluation> {
    validate_distribution_inputs(
        &NeuralOutput {
            logits: logits.to_vec(),
            value: 0.0,
        },
        tradable,
        sample.maximum_asset_allocation as f64,
        temperature,
        config,
    )?;
    let expected_indices = sample.active_asset_indices.clone();
    let mut sorted_indices = expected_indices.clone();
    sorted_indices.sort_unstable();
    let canonical_indices = active_asset_indices(tradable)?;
    if sorted_indices != canonical_indices
        || sample.latent_samples.len() != expected_indices.len()
        || sample.allocation.len() != tradable.len() + 1
    {
        return Err(MidasError::Policy(
            "PPO portfolio action sample dimensions or mask mismatch".into(),
        ));
    }
    let latents = sample
        .latent_samples
        .iter()
        .map(|value| *value as f64)
        .collect::<Vec<_>>();
    let (allocation, decisions) = evaluate_latents(
        logits,
        tradable,
        &expected_indices,
        &latents,
        sample.maximum_asset_allocation as f64,
        temperature,
        config,
    )?;
    if allocation
        .iter()
        .zip(&sample.allocation)
        .any(|(actual, stored)| !stored.is_finite() || (actual - stored).abs() > 1e-5)
    {
        return Err(MidasError::Policy(
            "PPO portfolio action sample allocation does not match its latent action".into(),
        ));
    }
    let log_probability = decisions
        .iter()
        .map(|decision| beta_log_probability(decision.alpha, decision.beta, decision.latent))
        .sum::<Result<f64>>()? as f32;
    let entropy = decisions
        .iter()
        .map(|decision| beta_entropy(decision.alpha, decision.beta))
        .sum::<Result<f64>>()? as f32;
    if !log_probability.is_finite() || !entropy.is_finite() {
        return Err(MidasError::Policy(
            "non-finite bounded stick-breaking PPO density".into(),
        ));
    }
    Ok(StickBreakingEvaluation {
        log_probability,
        entropy,
        decisions,
    })
}

fn beta_shapes(
    raw_alpha: f32,
    raw_beta: f32,
    temperature: f32,
    config: &ActionDistributionConfig,
) -> Result<(f64, f64, f64, f64)> {
    let shape = |raw: f32| -> Result<(f64, f64)> {
        if !raw.is_finite() {
            return Err(MidasError::Policy("non-finite Beta shape logit".into()));
        }
        let raw = raw as f64;
        let base = softplus(raw) + config.minimum_shape as f64;
        let unclamped = base / temperature as f64;
        let shape = unclamped.clamp(config.minimum_shape as f64, config.maximum_shape as f64);
        let derivative =
            if unclamped > config.minimum_shape as f64 && unclamped < config.maximum_shape as f64 {
                sigmoid(raw) / temperature as f64
            } else {
                0.0
            };
        if !shape.is_finite() || shape <= 0.0 || !derivative.is_finite() {
            return Err(MidasError::Policy("invalid bounded Beta shape".into()));
        }
        Ok((shape, derivative))
    };
    let (alpha, d_alpha) = shape(raw_alpha)?;
    let (beta, d_beta) = shape(raw_beta)?;
    Ok((alpha, beta, d_alpha, d_beta))
}

fn softplus(value: f64) -> f64 {
    if value > 20.0 {
        value
    } else {
        (1.0 + value.exp()).ln()
    }
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn beta_log_probability(alpha: f64, beta: f64, latent: f64) -> Result<f64> {
    if !alpha.is_finite()
        || !beta.is_finite()
        || alpha <= 0.0
        || beta <= 0.0
        || !latent.is_finite()
        || !(0.0..1.0).contains(&latent)
    {
        return Err(MidasError::Policy(
            "invalid Beta log-probability input".into(),
        ));
    }
    let value = (alpha - 1.0) * latent.ln() + (beta - 1.0) * (1.0 - latent).ln()
        - beta_log_normalizer(alpha, beta);
    if !value.is_finite() {
        return Err(MidasError::Policy("non-finite Beta log probability".into()));
    }
    Ok(value)
}

fn beta_log_normalizer(alpha: f64, beta: f64) -> f64 {
    libm::lgamma(alpha) + libm::lgamma(beta) - libm::lgamma(alpha + beta)
}

fn beta_log_probability_derivative_alpha(alpha: f64, beta: f64, latent: f64) -> Result<f64> {
    let value = latent.ln() - digamma(alpha) + digamma(alpha + beta);
    if value.is_finite() {
        Ok(value)
    } else {
        Err(MidasError::Policy(
            "non-finite Beta alpha log-probability gradient".into(),
        ))
    }
}

fn beta_log_probability_derivative_beta(alpha: f64, beta: f64, latent: f64) -> Result<f64> {
    let value = (1.0 - latent).ln() - digamma(beta) + digamma(alpha + beta);
    if value.is_finite() {
        Ok(value)
    } else {
        Err(MidasError::Policy(
            "non-finite Beta beta log-probability gradient".into(),
        ))
    }
}

fn beta_entropy(alpha: f64, beta: f64) -> Result<f64> {
    let sum = alpha + beta;
    let value = beta_log_normalizer(alpha, beta)
        - (alpha - 1.0) * digamma(alpha)
        - (beta - 1.0) * digamma(beta)
        + (sum - 2.0) * digamma(sum);
    if value.is_finite() {
        Ok(value)
    } else {
        Err(MidasError::Policy("non-finite Beta entropy".into()))
    }
}

fn beta_entropy_derivative_alpha(alpha: f64, beta: f64) -> Result<f64> {
    let value = -(alpha - 1.0) * trigamma(alpha) + (alpha + beta - 2.0) * trigamma(alpha + beta);
    if value.is_finite() {
        Ok(value)
    } else {
        Err(MidasError::Policy(
            "non-finite Beta entropy gradient".into(),
        ))
    }
}

fn beta_entropy_derivative_beta(alpha: f64, beta: f64) -> Result<f64> {
    let value = -(beta - 1.0) * trigamma(beta) + (alpha + beta - 2.0) * trigamma(alpha + beta);
    if value.is_finite() {
        Ok(value)
    } else {
        Err(MidasError::Policy(
            "non-finite Beta entropy gradient".into(),
        ))
    }
}

// Recurrence plus an asymptotic expansion gives stable digamma/trigamma
// values for the bounded positive shapes used by the action distribution.
fn digamma(mut value: f64) -> f64 {
    let mut result = 0.0;
    while value < 8.0 {
        result -= 1.0 / value;
        value += 1.0;
    }
    let inverse = 1.0 / value;
    let inverse_squared = inverse * inverse;
    result + value.ln()
        - 0.5 * inverse
        - inverse_squared
            * (1.0 / 12.0
                - inverse_squared
                    * (1.0 / 120.0
                        - inverse_squared
                            * (1.0 / 252.0
                                - inverse_squared * (1.0 / 240.0 - inverse_squared * 1.0 / 132.0))))
}

fn trigamma(mut value: f64) -> f64 {
    let mut result = 0.0;
    while value < 8.0 {
        result += 1.0 / (value * value);
        value += 1.0;
    }
    let inverse = 1.0 / value;
    let inverse_squared = inverse * inverse;
    result + inverse + 0.5 * inverse_squared + inverse_squared * inverse / 6.0
        - inverse_squared * inverse_squared * inverse / 30.0
        + inverse_squared.powi(3) * inverse / 42.0
        - inverse_squared.powi(4) * inverse / 30.0
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
        action::{ActionValidationContext, ActionValidator, LongOnlyActionValidator},
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
            action_distribution: ActionDistributionConfig::default(),
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

    fn ppo_sample(
        values: Vec<f32>,
        action_index: usize,
        advantage: f32,
        return_value: f32,
        old_log_probability: f32,
    ) -> PpoTrainingSample {
        PpoTrainingSample {
            observation: Observation {
                schema_hash: "schema".into(),
                values,
            },
            action_sample: Some(PolicyActionSample {
                active_asset_indices: vec![0],
                latent_samples: vec![if action_index == 0 { 0.25 } else { 0.75 }],
                allocation: vec![
                    if action_index == 0 { 0.75 } else { 0.25 },
                    if action_index == 0 { 0.25 } else { 0.75 },
                ],
                maximum_asset_allocation: 1.0,
                log_probability: old_log_probability,
                entropy: 0.0,
            }),
            old_log_probability: Some(old_log_probability),
            advantage: Some(advantage),
            return_value: Some(return_value),
            value_estimate: Some(0.0),
            recurrent_state: NeuralRuntimeState {
                memory: vec![0.0],
                inference_steps: 0,
            },
            tradable: vec![true],
            temperature: Some(1.0),
            invalid_action: false,
        }
    }

    fn deterministic_head_model() -> BurnNeuralModel {
        let mut architecture = architecture();
        architecture.activation = Activation::Silu;
        let genome =
            BrainGenome::random(&architecture, &EvolvableTraitsConfig::default(), 31).unwrap();
        let lifetime = LifetimeParameters {
            hidden: vec![DenseGenome {
                inputs: 2,
                outputs: 2,
                weights: vec![1.0, 0.0, 0.0, 1.0],
                bias: vec![0.0, 0.0],
            }],
            recurrent: DenseGenome {
                inputs: 3,
                outputs: 1,
                weights: vec![1.0, 0.0, 0.0],
                bias: vec![0.0],
            },
            actor: DenseGenome {
                inputs: 1,
                outputs: 2,
                weights: vec![0.0, 0.0],
                bias: vec![0.0, 0.0],
            },
            critic: DenseGenome {
                inputs: 1,
                outputs: 1,
                weights: vec![0.0],
                bias: vec![0.0],
            },
        };
        lifetime.validate(&architecture).unwrap();
        let burn_device = Default::default();
        BurnNeuralModel {
            architecture,
            cached: BurnRuntimeParameters::from_lifetime(&lifetime, &burn_device),
            lifetime,
            runtime: NeuralRuntimeState {
                memory: vec![0.0],
                inference_steps: 0,
            },
            optimizer: OptimizerState::empty(&genome.architecture),
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
                action_sample: Some(PolicyActionSample {
                    active_asset_indices: vec![0],
                    latent_samples: vec![0.5],
                    allocation: vec![0.5, 0.5],
                    maximum_asset_allocation: 1.0,
                    log_probability: -std::f32::consts::LN_2,
                    entropy: 0.0,
                }),
                old_log_probability: Some(-std::f32::consts::LN_2),
                advantage: Some(1.0),
                return_value: Some(1.0),
                value_estimate: Some(0.0),
                recurrent_state: NeuralRuntimeState {
                    memory: vec![0.0],
                    inference_steps: 0,
                },
                tradable: vec![true],
                temperature: Some(1.0),
                invalid_action: false,
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
    fn ppo_head_learning_changes_actor_weights_regression() {
        let mut model = deterministic_head_model();
        let before = model.lifetime_parameters().unwrap();
        let batch = PpoTrainingBatch {
            samples: vec![ppo_sample(vec![0.8, 0.3], 1, 1.0, 0.5, -0.7)],
            sequence_length: 1,
        };

        model.train_batch(&batch, &learning_config()).unwrap();
        let after = model.lifetime_parameters().unwrap();
        assert_ne!(
            before.actor.weights, after.actor.weights,
            "PPO must update the actor output weight matrix, not only its bias"
        );
    }

    #[test]
    fn full_head_training_updates_heads_but_freezes_encoder_and_genome() {
        let architecture = architecture();
        let genome =
            BrainGenome::random(&architecture, &EvolvableTraitsConfig::default(), 41).unwrap();
        let before_genome = genome.clone();
        let mut model = deterministic_head_model();
        let before = model.lifetime_parameters().unwrap();
        let builds_before = model.runtime_parameter_builds();
        let batch = PpoTrainingBatch {
            samples: vec![ppo_sample(vec![0.6, -0.2], 1, 1.0, 2.0, -0.7)],
            sequence_length: 1,
        };
        model.train_batch(&batch, &learning_config()).unwrap();
        let after = model.lifetime_parameters().unwrap();
        assert_ne!(before.actor.weights, after.actor.weights);
        assert_ne!(before.actor.bias, after.actor.bias);
        assert_ne!(before.critic.weights, after.critic.weights);
        assert_ne!(before.critic.bias, after.critic.bias);
        assert_eq!(before.hidden, after.hidden);
        assert_eq!(before.recurrent, after.recurrent);
        assert_eq!(before_genome, genome);

        let optimizer = model.optimizer_state().unwrap();
        assert_eq!(
            optimizer.actor.weight_momentum.len(),
            architecture.memory.size * architecture.actor_output_size
        );
        assert_eq!(
            optimizer.actor.bias_momentum.len(),
            architecture.actor_output_size
        );
        assert_eq!(
            optimizer.critic.weight_momentum.len(),
            architecture.memory.size * architecture.critic_output_size
        );
        assert_eq!(
            optimizer.critic.bias_momentum.len(),
            architecture.critic_output_size
        );
        assert_eq!(model.runtime_parameter_builds(), builds_before + 1);
    }

    #[test]
    fn deterministic_evaluation_does_not_decay_temperature() {
        let architecture = architecture();
        let mut genome =
            BrainGenome::random(&architecture, &EvolvableTraitsConfig::default(), 52).unwrap();
        genome.traits.action_temperature = 1.0;
        genome.traits.temperature_decay = 0.5;
        let device = ComputeDevice::CpuNdArray {
            device: "default".into(),
        };
        let mut policy = NeuralPolicy::new_with_seed(
            genome.clone(),
            BurnNeuralModelFactory
                .build(&genome, &architecture, &device)
                .unwrap(),
            7,
        );
        let observation = Observation {
            schema_hash: "schema".into(),
            values: vec![0.1, -0.2],
        };
        let context = PolicyContext {
            tradable: &[true],
            current_allocations: &[1.0, 0.0],
        };
        let before = match policy.snapshot().unwrap() {
            PolicySnapshot::Neural { temperature, .. } => temperature,
            _ => unreachable!(),
        };
        policy
            .evaluate_action(&observation, &context, false, 1.0)
            .unwrap();
        let after_deterministic = match policy.snapshot().unwrap() {
            PolicySnapshot::Neural { temperature, .. } => temperature,
            _ => unreachable!(),
        };
        assert_eq!(before, after_deterministic);
        policy
            .evaluate_action(&observation, &context, true, 1.0)
            .unwrap();
        let after_stochastic = match policy.snapshot().unwrap() {
            PolicySnapshot::Neural { temperature, .. } => temperature,
            _ => unreachable!(),
        };
        assert!((after_stochastic - before * 0.5).abs() < 1e-6);
        policy
            .evaluate_action(&observation, &context, false, 1.0)
            .unwrap();
        let after_bootstrap = match policy.snapshot().unwrap() {
            PolicySnapshot::Neural { temperature, .. } => temperature,
            _ => unreachable!(),
        };
        assert_eq!(after_stochastic, after_bootstrap);
    }

    #[test]
    fn dense_gradient_matches_outer_product_exactly() {
        let representation = [2.0_f32, -1.0];
        let output_gradient = [0.5_f32, -0.25];
        let mut weight_gradient = vec![0.0; 4];
        let mut bias_gradient = vec![0.0; 2];
        accumulate_dense_gradient(
            &representation,
            &output_gradient,
            &mut weight_gradient,
            &mut bias_gradient,
        )
        .unwrap();
        assert_eq!(weight_gradient, vec![1.0, -0.5, -0.5, 0.25]);
        assert_eq!(bias_gradient, vec![0.5, -0.25]);
    }

    #[test]
    fn clipped_policy_gradient_suppresses_head_update_but_keeps_objective_valid() {
        let mut model = deterministic_head_model();
        let before = model.lifetime_parameters().unwrap();
        let mut config = learning_config();
        config.critic_coefficient = 0.0;
        config.entropy_coefficient = 0.0;
        let sample = ppo_sample(vec![1.0, 0.0], 1, 1.0, 0.0, -2.0 * std::f32::consts::LN_2);
        let metrics = model
            .train_batch(
                &PpoTrainingBatch {
                    samples: vec![sample],
                    sequence_length: 1,
                },
                &config,
            )
            .unwrap();
        let after = model.lifetime_parameters().unwrap();
        assert_eq!(before.actor, after.actor);
        assert_eq!(before.critic, after.critic);
        assert_eq!(metrics.gradient_norm, 0.0);
    }

    #[test]
    fn actor_learning_is_state_dependent_for_two_encoded_states() {
        let mut model = deterministic_head_model();
        let mut config = learning_config();
        config.learning_rate = 0.4;
        config.ppo_clip = 10.0;
        config.critic_coefficient = 0.0;
        config.entropy_coefficient = 0.0;
        config.gradient_clip = 100.0;
        config.optimisation_epochs = 8;
        config.minibatch_size = 2;
        let batch = PpoTrainingBatch {
            samples: vec![
                ppo_sample(vec![1.0, 0.0], 1, 1.0, 0.0, -std::f32::consts::LN_2),
                ppo_sample(vec![-1.0, 0.0], 0, 1.0, 0.0, -std::f32::consts::LN_2),
            ],
            sequence_length: 1,
        };
        model.train_batch(&batch, &config).unwrap();
        let zero = NeuralRuntimeState {
            memory: vec![0.0],
            inference_steps: 0,
        };
        let state_a = model
            .encode_observation(
                &Observation {
                    schema_hash: "schema".into(),
                    values: vec![1.0, 0.0],
                },
                &zero,
            )
            .unwrap();
        let state_b = model
            .encode_observation(
                &Observation {
                    schema_hash: "schema".into(),
                    values: vec![-1.0, 0.0],
                },
                &zero,
            )
            .unwrap();
        let output_a = model
            .evaluate_representation(&state_a.representation)
            .unwrap();
        let output_b = model
            .evaluate_representation(&state_b.representation)
            .unwrap();
        let evaluation_a = bounded_stick_breaking_evaluation(
            output_a,
            &[true],
            1.0,
            false,
            1.0,
            &mut ChaCha8Rng::seed_from_u64(1),
            &model.architecture.action_distribution,
        )
        .unwrap();
        let evaluation_b = bounded_stick_breaking_evaluation(
            output_b,
            &[true],
            1.0,
            false,
            1.0,
            &mut ChaCha8Rng::seed_from_u64(1),
            &model.architecture.action_distribution,
        )
        .unwrap();
        assert!(
            evaluation_a.proposed_allocation.weights()[1]
                > evaluation_b.proposed_allocation.weights()[1]
        );
        assert!(
            evaluation_a.proposed_allocation.weights()[1]
                > 1.0 - evaluation_a.proposed_allocation.weights()[1]
        );
        assert!(
            evaluation_b.proposed_allocation.weights()[0]
                > evaluation_b.proposed_allocation.weights()[1]
        );
    }

    #[test]
    fn critic_learning_is_state_dependent_for_two_encoded_states() {
        let mut model = deterministic_head_model();
        let mut config = learning_config();
        config.learning_rate = 0.4;
        config.ppo_clip = 10.0;
        config.critic_coefficient = 1.0;
        config.entropy_coefficient = 0.0;
        config.gradient_clip = 100.0;
        config.optimisation_epochs = 8;
        config.minibatch_size = 2;
        let batch = PpoTrainingBatch {
            samples: vec![
                ppo_sample(vec![1.0, 0.0], 0, 0.0, 2.0, -std::f32::consts::LN_2),
                ppo_sample(vec![-1.0, 0.0], 0, 0.0, -2.0, -std::f32::consts::LN_2),
            ],
            sequence_length: 1,
        };
        model.train_batch(&batch, &config).unwrap();
        let zero = NeuralRuntimeState {
            memory: vec![0.0],
            inference_steps: 0,
        };
        let state_a = model
            .encode_observation(
                &Observation {
                    schema_hash: "schema".into(),
                    values: vec![1.0, 0.0],
                },
                &zero,
            )
            .unwrap();
        let state_b = model
            .encode_observation(
                &Observation {
                    schema_hash: "schema".into(),
                    values: vec![-1.0, 0.0],
                },
                &zero,
            )
            .unwrap();
        let value_a = model
            .evaluate_representation(&state_a.representation)
            .unwrap()
            .value;
        let value_b = model
            .evaluate_representation(&state_b.representation)
            .unwrap()
            .value;
        assert!(value_a > value_b);
    }

    #[test]
    fn malformed_full_head_optimizer_state_is_rejected_on_restore() {
        let architecture = architecture();
        let genome =
            BrainGenome::random(&architecture, &EvolvableTraitsConfig::default(), 51).unwrap();
        let device = ComputeDevice::CpuNdArray {
            device: "default".into(),
        };
        let model = BurnNeuralModelFactory
            .build(&genome, &architecture, &device)
            .unwrap();
        let base_state = model.snapshot_state().unwrap();
        for invalid in 0..4 {
            let NeuralModelState::Burn {
                parameters,
                runtime,
                mut optimizer,
            } = base_state.clone()
            else {
                panic!("expected Burn state")
            };
            match invalid {
                0 => {
                    optimizer.actor.weight_momentum.pop();
                }
                1 => {
                    optimizer.actor.bias_momentum.pop();
                }
                2 => {
                    optimizer.critic.weight_momentum.pop();
                }
                _ => {
                    optimizer.critic.bias_momentum[0] = f32::NAN;
                }
            }
            let error = match BurnNeuralModelFactory.restore(
                &genome,
                &architecture,
                NeuralModelState::Burn {
                    parameters,
                    runtime,
                    optimizer,
                },
                &device,
            ) {
                Ok(_) => panic!("shape-invalid optimizer state was accepted"),
                Err(error) => error,
            };
            assert!(error.to_string().contains("optimizer"));
        }
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
    fn bounded_stick_breaking_is_masked_and_seed_reproducible() {
        let output = NeuralOutput {
            logits: vec![0.0, 2.0, 0.0, 1.0],
            value: 0.0,
        };
        let mut deterministic_rng = ChaCha8Rng::seed_from_u64(1);
        let deterministic = bounded_stick_breaking_evaluation(
            output.clone(),
            &[true, false],
            0.5,
            false,
            1.0,
            &mut deterministic_rng,
            &ActionDistributionConfig::default(),
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
        let left = bounded_stick_breaking_evaluation(
            output.clone(),
            &[true, true],
            0.5,
            true,
            1.0,
            &mut left_rng,
            &ActionDistributionConfig::default(),
        )
        .unwrap();
        let right = bounded_stick_breaking_evaluation(
            output,
            &[true, true],
            0.5,
            true,
            1.0,
            &mut right_rng,
            &ActionDistributionConfig::default(),
        )
        .unwrap();
        assert_eq!(left, right);
    }

    #[test]
    fn neural_policy_can_allocate_positive_weight_to_two_assets() {
        let output = NeuralOutput {
            logits: vec![0.0, 0.0, 0.0, 0.0],
            value: 0.0,
        };
        let mut rng = ChaCha8Rng::seed_from_u64(9);
        let evaluation = bounded_stick_breaking_evaluation(
            output,
            &[true, true],
            0.5,
            false,
            1.0,
            &mut rng,
            &ActionDistributionConfig::default(),
        )
        .unwrap();
        let weights = evaluation.proposed_allocation.weights();
        assert!(weights[1] > 0.0, "first asset must receive positive weight");
        assert!(
            weights[2] > 0.0,
            "second asset must receive positive weight"
        );
    }

    #[test]
    fn bounded_stick_breaking_distribution_is_valid_across_masks_caps_and_seeds() {
        for (tradable, cap) in [
            (vec![true, true, true], 1.0),
            (vec![true, false, true], 0.5),
            (vec![false, false, false], 0.25),
            (vec![true, true, true], 0.01),
        ] {
            for seed in 0..32_u64 {
                let output = NeuralOutput {
                    logits: vec![0.3, -0.2, 1.0, -1.0, -0.7, 0.8],
                    value: 0.0,
                };
                let evaluation = bounded_stick_breaking_evaluation(
                    output,
                    &tradable,
                    cap,
                    true,
                    1.0,
                    &mut ChaCha8Rng::seed_from_u64(seed),
                    &ActionDistributionConfig::default(),
                )
                .unwrap();
                let weights = evaluation.proposed_allocation.weights();
                assert_eq!(weights.len(), 4);
                assert!(
                    weights
                        .iter()
                        .all(|weight| weight.is_finite() && *weight >= 0.0)
                );
                assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-9);
                for (index, is_tradable) in tradable.iter().enumerate() {
                    if !is_tradable {
                        assert_eq!(weights[index + 1], 0.0);
                    } else {
                        assert!(weights[index + 1] <= cap + 1e-9);
                    }
                }
            }
        }
    }

    #[test]
    fn bounded_stick_breaking_log_probability_recomputes_exactly() {
        let output = NeuralOutput {
            logits: vec![0.3, -0.2, 1.0, -1.0, -0.7, 0.8],
            value: 0.0,
        };
        let config = ActionDistributionConfig::default();
        let evaluation = bounded_stick_breaking_evaluation(
            output.clone(),
            &[true, false, true],
            0.5,
            true,
            1.0,
            &mut ChaCha8Rng::seed_from_u64(12),
            &config,
        )
        .unwrap();
        let sample = evaluation.action_sample.as_ref().unwrap();
        let recomputed = evaluate_bounded_stick_breaking_sample(
            &output.logits,
            &[true, false, true],
            sample,
            1.0,
            &config,
        )
        .unwrap();
        assert!((sample.log_probability - recomputed.log_probability).abs() < 1e-6);
        assert!((sample.entropy - recomputed.entropy).abs() < 1e-6);
    }

    #[test]
    fn deterministic_stick_breaking_evaluation_does_not_consume_rng() {
        let output = NeuralOutput {
            logits: vec![0.0, 0.0, 0.0, 0.0],
            value: 0.0,
        };
        let mut rng = ChaCha8Rng::seed_from_u64(77);
        let mut expected = rng.clone();
        bounded_stick_breaking_evaluation(
            output,
            &[true, true],
            0.5,
            false,
            1.0,
            &mut rng,
            &ActionDistributionConfig::default(),
        )
        .unwrap();
        assert_eq!(rng.r#gen::<u64>(), expected.r#gen::<u64>());
    }

    #[test]
    fn sampled_valid_allocation_survives_central_validation_unchanged() {
        let evaluation = bounded_stick_breaking_evaluation(
            NeuralOutput {
                logits: vec![0.2, -0.3, 0.4, 0.1],
                value: 0.0,
            },
            &[true, true],
            0.5,
            true,
            1.0,
            &mut ChaCha8Rng::seed_from_u64(3),
            &ActionDistributionConfig::default(),
        )
        .unwrap();
        let proposed = evaluation.proposed_allocation.clone();
        let validated = LongOnlyActionValidator
            .validate(
                proposed,
                &ActionValidationContext {
                    asset_count: 2,
                    tradable: &[true, true],
                    policy_maximum_asset_allocation: Some(0.5),
                    market_maximum_asset_allocation: None,
                },
            )
            .unwrap();
        assert!(
            validated
                .weights()
                .iter()
                .zip(evaluation.proposed_allocation.weights())
                .all(|(actual, proposed)| (actual - proposed).abs() < 1e-9)
        );
    }

    #[test]
    fn beta_log_probability_gradient_matches_finite_difference() {
        let config = ActionDistributionConfig::default();
        let raw_alpha = 0.35_f32;
        let raw_beta = -0.4_f32;
        let latent = 0.37_f64;
        let (alpha, beta, d_alpha, _) = beta_shapes(raw_alpha, raw_beta, 1.0, &config).unwrap();
        let analytic =
            beta_log_probability_derivative_alpha(alpha, beta, latent).unwrap() * d_alpha;
        let step = 1e-4_f32;
        let shifted = |raw: f32| {
            let (a, b, _, _) = beta_shapes(raw, raw_beta, 1.0, &config).unwrap();
            beta_log_probability(a, b, latent).unwrap()
        };
        let finite_difference =
            (shifted(raw_alpha + step) - shifted(raw_alpha - step)) / (2.0 * step as f64);
        assert!((analytic - finite_difference).abs() < 2e-3);
    }

    #[test]
    fn beta_entropy_gradient_matches_finite_difference() {
        let config = ActionDistributionConfig::default();
        let raw_alpha = 0.35_f32;
        let raw_beta = -0.4_f32;
        let (alpha, beta, d_alpha, _) = beta_shapes(raw_alpha, raw_beta, 1.0, &config).unwrap();
        let analytic = beta_entropy_derivative_alpha(alpha, beta).unwrap() * d_alpha;
        let step = 1e-4_f32;
        let shifted = |raw: f32| {
            let (a, b, _, _) = beta_shapes(raw, raw_beta, 1.0, &config).unwrap();
            beta_entropy(a, b).unwrap()
        };
        let finite_difference =
            (shifted(raw_alpha + step) - shifted(raw_alpha - step)) / (2.0 * step as f64);
        assert!((analytic - finite_difference).abs() < 2e-3);
    }
}
