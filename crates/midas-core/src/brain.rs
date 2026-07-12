use crate::{
    MidasError, Result,
    config::{Activation, BrainConfig},
};
use burn::{
    backend::NdArray,
    tensor::{Tensor, TensorData, activation},
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

pub type CpuBackend = NdArray<f32>;

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

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct BrainGenome {
    pub genome_format_version: u32,
    pub architecture_version: u32,
    pub observation_size: usize,
    pub asset_count: usize,
    pub hidden: Vec<DenseGenome>,
    pub recurrent: DenseGenome,
    pub actor: DenseGenome,
    pub critic: DenseGenome,
    pub traits: LearningTraits,
    pub activation: Activation,
}

impl BrainGenome {
    pub fn random(
        config: &BrainConfig,
        observation_size: usize,
        asset_count: usize,
        seed: u64,
    ) -> Result<Self> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut input = observation_size;
        let mut hidden = Vec::new();
        for &output in &config.hidden_layers {
            hidden.push(random_dense(input, output, &mut rng));
            input = output;
        }
        let recurrent = random_dense(input + config.memory_size, config.memory_size, &mut rng);
        let actor = random_dense(config.memory_size, asset_count + 1, &mut rng);
        let critic = random_dense(config.memory_size, 1, &mut rng);
        let genome = Self {
            genome_format_version: 1,
            architecture_version: config.architecture_version,
            observation_size,
            asset_count,
            hidden,
            recurrent,
            actor,
            critic,
            traits: LearningTraits {
                learning_rate: 0.0003,
                entropy_coefficient: 0.01,
                action_temperature: 1.0,
                temperature_decay: 0.999,
                drawdown_sensitivity: 0.1,
                turnover_sensitivity: 0.01,
                maximum_allocation: config.maximum_allocation as f32,
            },
            activation: config.activation,
        };
        genome.validate()?;
        Ok(genome)
    }
    pub fn validate(&self) -> Result<()> {
        if self.genome_format_version != 1
            || self.architecture_version == 0
            || self.observation_size == 0
            || self.asset_count == 0
        {
            return Err(MidasError::Policy(
                "invalid genome version or dimensions".into(),
            ));
        }
        let mut expected = self.observation_size;
        for layer in &self.hidden {
            validate_dense(layer, expected)?;
            expected = layer.outputs;
        }
        if self.recurrent.inputs <= expected {
            return Err(MidasError::Policy(
                "recurrent layer lacks memory inputs".into(),
            ));
        }
        validate_dense(&self.recurrent, self.recurrent.inputs)?;
        let memory = self.recurrent.inputs - expected;
        if self.recurrent.outputs != memory {
            return Err(MidasError::Policy(
                "recurrent output must equal memory size".into(),
            ));
        }
        validate_dense(&self.actor, memory)?;
        validate_dense(&self.critic, memory)?;
        if self.actor.outputs != self.asset_count + 1 || self.critic.outputs != 1 {
            return Err(MidasError::Policy(
                "actor/critic output dimensions mismatch".into(),
            ));
        }
        Ok(())
    }
}

fn validate_dense(layer: &DenseGenome, inputs: usize) -> Result<()> {
    if layer.inputs != inputs
        || layer.outputs == 0
        || layer.weights.len() != layer.inputs * layer.outputs
        || layer.bias.len() != layer.outputs
        || layer
            .weights
            .iter()
            .chain(&layer.bias)
            .any(|v| !v.is_finite())
    {
        Err(MidasError::Policy(
            "invalid dense parameter shape or value".into(),
        ))
    } else {
        Ok(())
    }
}
fn random_dense(inputs: usize, outputs: usize, rng: &mut ChaCha8Rng) -> DenseGenome {
    let scale = (2.0 / inputs as f32).sqrt();
    DenseGenome {
        inputs,
        outputs,
        weights: (0..inputs * outputs)
            .map(|_| rng.gen_range(-scale..scale))
            .collect(),
        bias: vec![0.0; outputs],
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LifetimeBrain {
    pub parameters: BrainGenome,
    pub memory: Vec<f32>,
    pub inference_steps: u64,
}

#[derive(Clone, Debug)]
pub struct PolicyInput<'a> {
    pub observation: &'a [f32],
    pub tradable: &'a [bool],
    pub current_allocations: &'a [f64],
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PolicyOutput {
    pub allocations: Vec<f64>,
    pub value: f32,
    pub valid: bool,
    pub rebalance: bool,
}

pub trait Policy: Send {
    fn act(&mut self, input: PolicyInput<'_>) -> Result<PolicyOutput>;
    fn snapshot(&self) -> Result<PolicySnapshot>;
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
    Burn {
        genome: BrainGenome,
        lifetime: Box<LifetimeBrain>,
    },
}

pub struct BurnActorCriticPolicy {
    pub genome: BrainGenome,
    pub lifetime: LifetimeBrain,
}
impl BurnActorCriticPolicy {
    pub fn new(genome: BrainGenome) -> Result<Self> {
        genome.validate()?;
        let memory = vec![0.0; genome.recurrent.outputs];
        Ok(Self {
            lifetime: LifetimeBrain {
                parameters: genome.clone(),
                memory,
                inference_steps: 0,
            },
            genome,
        })
    }
}

impl Policy for BurnActorCriticPolicy {
    fn act(&mut self, input: PolicyInput<'_>) -> Result<PolicyOutput> {
        if input.observation.len() != self.genome.observation_size
            || input.tradable.len() != self.genome.asset_count
            || input.observation.iter().any(|v| !v.is_finite())
        {
            return Ok(cash(self.genome.asset_count, false));
        }
        let device = Default::default();
        let mut x = Tensor::<CpuBackend, 2>::from_data(
            TensorData::new(input.observation.to_vec(), [1, input.observation.len()]),
            &device,
        );
        for layer in &self.lifetime.parameters.hidden {
            x = activate(dense(x, layer, &device), self.genome.activation);
        }
        let memory = Tensor::<CpuBackend, 2>::from_data(
            TensorData::new(
                self.lifetime.memory.clone(),
                [1, self.lifetime.memory.len()],
            ),
            &device,
        );
        let joined = Tensor::cat(vec![x, memory], 1);
        let next = activate(
            dense(joined, &self.lifetime.parameters.recurrent, &device),
            Activation::Silu,
        );
        let logits = dense(next.clone(), &self.lifetime.parameters.actor, &device)
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| MidasError::Policy(format!("{e:?}")))?;
        let value = dense(next.clone(), &self.lifetime.parameters.critic, &device)
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| MidasError::Policy(format!("{e:?}")))?[0];
        self.lifetime.memory = next
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| MidasError::Policy(format!("{e:?}")))?;
        self.lifetime.inference_steps += 1;
        if logits.iter().any(|v| !v.is_finite())
            || !value.is_finite()
            || self.lifetime.memory.iter().any(|v| !v.is_finite())
        {
            return Ok(cash(self.genome.asset_count, false));
        }
        Ok(PolicyOutput {
            allocations: masked_softmax(
                &logits,
                input.tradable,
                self.genome.traits.maximum_allocation as f64,
            ),
            value,
            valid: true,
            rebalance: true,
        })
    }
    fn snapshot(&self) -> Result<PolicySnapshot> {
        Ok(PolicySnapshot::Burn {
            genome: self.genome.clone(),
            lifetime: Box::new(self.lifetime.clone()),
        })
    }
}

fn dense(
    x: Tensor<CpuBackend, 2>,
    layer: &DenseGenome,
    device: &<CpuBackend as burn::tensor::backend::Backend>::Device,
) -> Tensor<CpuBackend, 2> {
    let w = Tensor::from_data(
        TensorData::new(layer.weights.clone(), [layer.inputs, layer.outputs]),
        device,
    );
    let b = Tensor::from_data(
        TensorData::new(layer.bias.clone(), [1, layer.outputs]),
        device,
    );
    x.matmul(w) + b
}
fn activate(x: Tensor<CpuBackend, 2>, kind: Activation) -> Tensor<CpuBackend, 2> {
    match kind {
        Activation::Relu => activation::relu(x),
        Activation::Gelu => activation::gelu(x),
        Activation::Silu => activation::silu(x),
    }
}

fn masked_softmax(logits: &[f32], tradable: &[bool], max: f64) -> Vec<f64> {
    if tradable.iter().all(|v| !*v) {
        return cash(tradable.len(), true).allocations;
    }
    let mut allowed = vec![true];
    allowed.extend_from_slice(tradable);
    let peak = logits
        .iter()
        .zip(&allowed)
        .filter(|(_, a)| **a)
        .map(|(v, _)| *v)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut out = logits
        .iter()
        .zip(&allowed)
        .map(|(v, a)| if *a { ((*v - peak) as f64).exp() } else { 0.0 })
        .collect::<Vec<_>>();
    let sum = out.iter().sum::<f64>();
    for v in &mut out {
        *v /= sum
    }
    for _ in 0..out.len() {
        let mut excess = 0.0;
        let mut open = 0;
        for i in 1..out.len() {
            if out[i] > max {
                excess += out[i] - max;
                out[i] = max
            } else if allowed[i] {
                open += 1
            }
        }
        if excess <= 1e-12 {
            break;
        }
        if open == 0 {
            out[0] += excess
        } else {
            let cash_share = excess / (open + 1) as f64;
            out[0] += cash_share;
            for i in 1..out.len() {
                if allowed[i] && out[i] < max {
                    out[i] += cash_share
                }
            }
        }
    }
    let sum = out.iter().sum::<f64>();
    for v in &mut out {
        *v /= sum
    }
    out
}

fn cash(assets: usize, valid: bool) -> PolicyOutput {
    let mut allocations = vec![0.0; assets + 1];
    allocations[0] = 1.0;
    PolicyOutput {
        allocations,
        value: 0.0,
        valid,
        rebalance: true,
    }
}
pub struct CashPolicy;
impl Policy for CashPolicy {
    fn act(&mut self, input: PolicyInput<'_>) -> Result<PolicyOutput> {
        Ok(cash(input.tradable.len(), true))
    }
    fn snapshot(&self) -> Result<PolicySnapshot> {
        Ok(PolicySnapshot::Cash)
    }
}
pub struct EqualWeightPolicy;
impl Policy for EqualWeightPolicy {
    fn act(&mut self, input: PolicyInput<'_>) -> Result<PolicyOutput> {
        let n = input.tradable.iter().filter(|v| **v).count();
        if n == 0 {
            return Ok(cash(input.tradable.len(), true));
        }
        let mut a = vec![0.0; input.tradable.len() + 1];
        for (i, t) in input.tradable.iter().enumerate() {
            if *t {
                a[i + 1] = 1.0 / n as f64
            }
        }
        Ok(PolicyOutput {
            allocations: a,
            value: 0.0,
            valid: true,
            rebalance: true,
        })
    }
    fn snapshot(&self) -> Result<PolicySnapshot> {
        Ok(PolicySnapshot::EqualWeight)
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
    fn act(&mut self, input: PolicyInput<'_>) -> Result<PolicyOutput> {
        let mut x = vec![self.rng.gen_range(0.0001..1.0)];
        x.extend(input.tradable.iter().map(|t| {
            if *t {
                self.rng.gen_range(0.0001..1.0)
            } else {
                0.0
            }
        }));
        let s = x.iter().sum::<f64>();
        for v in &mut x {
            *v /= s
        }
        Ok(PolicyOutput {
            allocations: x,
            value: 0.0,
            valid: true,
            rebalance: true,
        })
    }
    fn snapshot(&self) -> Result<PolicySnapshot> {
        Ok(PolicySnapshot::Random {
            rng: self.rng.clone(),
        })
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
    fn act(&mut self, input: PolicyInput<'_>) -> Result<PolicyOutput> {
        if self.initialized {
            return Ok(PolicyOutput {
                allocations: input.current_allocations.to_vec(),
                value: 0.0,
                valid: true,
                rebalance: false,
            });
        }
        self.initialized = true;
        if input
            .tradable
            .get(self.target_asset)
            .copied()
            .unwrap_or(false)
        {
            let mut a = vec![0.0; input.tradable.len() + 1];
            a[self.target_asset + 1] = 1.0;
            Ok(PolicyOutput {
                allocations: a,
                value: 0.0,
                valid: true,
                rebalance: true,
            })
        } else {
            Ok(cash(input.tradable.len(), true))
        }
    }
    fn snapshot(&self) -> Result<PolicySnapshot> {
        Ok(PolicySnapshot::BuyAndHold {
            initialized: self.initialized,
            target_asset: self.target_asset,
        })
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
    fn observe(&mut self, _reward: f32, _terminal: bool) -> Result<()> {
        Ok(())
    }
}
