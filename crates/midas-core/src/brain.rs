use crate::{
    MidasError, Result,
    action::ProposedAllocation,
    config::{Activation, BrainConfig, EvolvableTraitsConfig},
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
pub enum NeuralModelState {
    Burn {
        parameters: Box<LifetimeParameters>,
        runtime: NeuralRuntimeState,
    },
    External {
        kind: String,
        state: Vec<f32>,
    },
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
        let burn_device = Default::default();
        Ok(Box::new(BurnNeuralModel {
            architecture: a.clone(),
            cached: BurnRuntimeParameters::from_lifetime(&lifetime, &burn_device),
            lifetime,
            runtime,
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
        })
    }
    fn fork(&self) -> Result<Box<dyn NeuralModel>> {
        Ok(Box::new(Self {
            architecture: self.architecture.clone(),
            lifetime: self.lifetime.clone(),
            runtime: self.runtime.clone(),
            cached: self.cached.clone(),
        }))
    }
    fn runtime_parameter_builds(&self) -> u64 {
        self.cached.builds
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
    /// Fork policy state for a transactional simulation step.
    fn fork(&self) -> Result<Box<dyn Policy>> {
        Err(MidasError::Policy(
            "policy does not support transactional forking".into(),
        ))
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
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
    },
}

pub struct NeuralPolicy {
    genome: BrainGenome,
    model: Box<dyn NeuralModel>,
}
impl NeuralPolicy {
    pub fn new(genome: BrainGenome, model: Box<dyn NeuralModel>) -> Self {
        Self { genome, model }
    }
}
impl Policy for NeuralPolicy {
    fn propose_action(
        &mut self,
        o: &Observation,
        _: &PolicyContext<'_>,
    ) -> Result<ProposedAllocation> {
        let out = self.model.forward(&NeuralInput { observation: o })?;
        let peak = out.logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut weights = out
            .logits
            .iter()
            .map(|v| ((*v - peak) as f64).exp())
            .collect::<Vec<_>>();
        let sum = weights.iter().sum::<f64>();
        for v in &mut weights {
            *v /= sum
        }
        Ok(ProposedAllocation::rebalance(weights))
    }
    fn snapshot(&self) -> Result<PolicySnapshot> {
        Ok(PolicySnapshot::Neural {
            genome: self.genome.clone(),
            state: Box::new(self.model.snapshot_state()?),
        })
    }
    fn reset_episode_state(&mut self) {
        self.model.reset_memory()
    }
    fn policy_blueprint(&self) -> PolicyBlueprint {
        PolicyBlueprint::Neural
    }
    fn fork(&self) -> Result<Box<dyn Policy>> {
        Ok(Box::new(Self {
            genome: self.genome.clone(),
            model: self.model.fork()?,
        }))
    }
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

pub trait LearningStrategy: Send {
    fn name(&self) -> &'static str;
    fn observe(&mut self, _reward: f32, _terminal: bool) -> Result<()>;
}
pub struct NoLearning;
impl LearningStrategy for NoLearning {
    fn name(&self) -> &'static str {
        "no_learning"
    }
    fn observe(&mut self, _: f32, _: bool) -> Result<()> {
        Ok(())
    }
}
