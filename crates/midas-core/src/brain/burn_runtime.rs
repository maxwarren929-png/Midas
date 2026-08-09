use super::{
    AdamWOptimizerState, BpttOptimizerState, BrainArchitecture, BrainGenome, ComputeDevice,
    DenseGenome, LifetimeParameters, NeuralInput, NeuralModel, NeuralModelFactory,
    NeuralModelState, NeuralOutput, NeuralRuntimeState,
};
use crate::{
    MidasError, Result,
    config::{ActionDistributionConfig, Activation, LearningConfig, OptimizerConfig},
    observation::Observation,
};
use burn::{
    backend::{Autodiff, NdArray},
    tensor::{Tensor, TensorData, activation},
};

pub(super) type InferBackend = NdArray<f32>;
pub(super) type TrainBackend = Autodiff<InferBackend>;

pub struct BurnNeuralModel {
    pub(super) architecture: BrainArchitecture,
    pub(super) lifetime: LifetimeParameters,
    pub(super) runtime: NeuralRuntimeState,
    pub(super) cached: BurnRuntimeParameters,
    pub(super) optimizer: BpttOptimizerState,
    pub(super) adamw_optimizer: Option<AdamWOptimizerState>,
}
#[derive(Clone)]
struct BurnRuntimeDense {
    weights: Tensor<InferBackend, 2>,
    bias: Tensor<InferBackend, 2>,
}
#[derive(Clone)]
enum BurnRuntimeRecurrent {
    Simple(Box<BurnRuntimeDense>),
    Gru(Box<BurnRuntimeGru>),
}
#[derive(Clone)]
struct BurnRuntimeGru {
    update_weights: Tensor<InferBackend, 2>,
    update_bias: Tensor<InferBackend, 2>,
    reset_weights: Tensor<InferBackend, 2>,
    reset_bias: Tensor<InferBackend, 2>,
    candidate_input_weights: Tensor<InferBackend, 2>,
    candidate_hidden_weights: Tensor<InferBackend, 2>,
    candidate_bias: Tensor<InferBackend, 2>,
}
pub(super) struct GruParameters {
    pub(super) update_weights: Vec<f32>,
    pub(super) update_bias: Vec<f32>,
    pub(super) reset_weights: Vec<f32>,
    pub(super) reset_bias: Vec<f32>,
    pub(super) candidate_input_weights: Vec<f32>,
    pub(super) candidate_hidden_weights: Vec<f32>,
    pub(super) candidate_bias: Vec<f32>,
}
#[derive(Clone)]
pub(super) struct BurnRuntimeParameters {
    hidden: Vec<BurnRuntimeDense>,
    recurrent: BurnRuntimeRecurrent,
    actor: BurnRuntimeDense,
    critic: BurnRuntimeDense,
    builds: u64,
}
impl BurnRuntimeParameters {
    pub(super) fn from_lifetime(
        lifetime: &LifetimeParameters,
        d: &<InferBackend as burn::tensor::backend::Backend>::Device,
    ) -> Self {
        Self {
            hidden: lifetime
                .hidden
                .iter()
                .map(|l| runtime_dense(l, d))
                .collect(),
            recurrent: runtime_recurrent(lifetime, d),
            actor: runtime_dense(&lifetime.actor, d),
            critic: runtime_dense(&lifetime.critic, d),
            builds: 1,
        }
    }

    pub(super) fn with_updated_trainable(
        &self,
        lifetime: &LifetimeParameters,
        d: &<InferBackend as burn::tensor::backend::Backend>::Device,
    ) -> Self {
        Self {
            hidden: lifetime
                .hidden
                .iter()
                .map(|layer| runtime_dense(layer, d))
                .collect(),
            recurrent: runtime_recurrent(lifetime, d),
            actor: runtime_dense(&lifetime.actor, d),
            critic: runtime_dense(&lifetime.critic, d),
            builds: self.builds.saturating_add(1),
        }
    }
}
fn runtime_dense(
    l: &DenseGenome,
    d: &<InferBackend as burn::tensor::backend::Backend>::Device,
) -> BurnRuntimeDense {
    BurnRuntimeDense {
        weights: Tensor::from_data(TensorData::new(l.weights.clone(), [l.inputs, l.outputs]), d),
        bias: Tensor::from_data(TensorData::new(l.bias.clone(), [1, l.outputs]), d),
    }
}
fn runtime_recurrent(
    lifetime: &LifetimeParameters,
    d: &<InferBackend as burn::tensor::backend::Backend>::Device,
) -> BurnRuntimeRecurrent {
    let recurrent = &lifetime.recurrent;
    let memory_size = lifetime.actor.inputs;
    if recurrent.outputs == memory_size {
        return BurnRuntimeRecurrent::Simple(Box::new(runtime_dense(recurrent, d)));
    }
    let hidden_size = recurrent.inputs - memory_size;
    let parts = split_gru_parameters(recurrent, hidden_size, memory_size);
    BurnRuntimeRecurrent::Gru(Box::new(BurnRuntimeGru {
        update_weights: Tensor::from_data(
            TensorData::new(parts.update_weights, [recurrent.inputs, memory_size]),
            d,
        ),
        update_bias: Tensor::from_data(TensorData::new(parts.update_bias, [1, memory_size]), d),
        reset_weights: Tensor::from_data(
            TensorData::new(parts.reset_weights, [recurrent.inputs, memory_size]),
            d,
        ),
        reset_bias: Tensor::from_data(TensorData::new(parts.reset_bias, [1, memory_size]), d),
        candidate_input_weights: Tensor::from_data(
            TensorData::new(parts.candidate_input_weights, [hidden_size, memory_size]),
            d,
        ),
        candidate_hidden_weights: Tensor::from_data(
            TensorData::new(parts.candidate_hidden_weights, [memory_size, memory_size]),
            d,
        ),
        candidate_bias: Tensor::from_data(
            TensorData::new(parts.candidate_bias, [1, memory_size]),
            d,
        ),
    }))
}

pub(super) fn split_gru_parameters(
    recurrent: &DenseGenome,
    hidden_size: usize,
    memory_size: usize,
) -> GruParameters {
    let gate_weights = |gate: usize| {
        (0..recurrent.inputs)
            .flat_map(|row| {
                let start = row * recurrent.outputs + gate * memory_size;
                recurrent.weights[start..start + memory_size]
                    .iter()
                    .copied()
            })
            .collect::<Vec<_>>()
    };
    let candidate = gate_weights(2);
    GruParameters {
        update_weights: gate_weights(0),
        update_bias: recurrent.bias[..memory_size].to_vec(),
        reset_weights: gate_weights(1),
        reset_bias: recurrent.bias[memory_size..memory_size * 2].to_vec(),
        candidate_input_weights: candidate[..hidden_size * memory_size].to_vec(),
        candidate_hidden_weights: candidate[hidden_size * memory_size..].to_vec(),
        candidate_bias: recurrent.bias[memory_size * 2..].to_vec(),
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
        let recurrent_outputs = lifetime.recurrent.outputs;
        Ok(Box::new(BurnNeuralModel {
            architecture: a.clone(),
            cached: BurnRuntimeParameters::from_lifetime(&lifetime, &burn_device),
            lifetime,
            runtime: NeuralRuntimeState {
                memory: vec![0.0; a.memory.size],
                inference_steps: 0,
            },
            optimizer: BpttOptimizerState::empty(a, recurrent_outputs),
            adamw_optimizer: None,
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
        let (parameters, runtime, optimizer, adamw_optimizer) = match state {
            NeuralModelState::Burn {
                parameters,
                runtime,
                optimizer,
            } => (
                parameters,
                runtime,
                BpttOptimizerState::from_frozen_encoder(optimizer, a)?,
                None,
            ),
            NeuralModelState::BurnTrainableEncoder {
                parameters,
                runtime,
                optimizer,
            } => (
                parameters,
                runtime,
                BpttOptimizerState::from_trainable_encoder(optimizer, a)?,
                None,
            ),
            NeuralModelState::BurnBptt {
                parameters,
                runtime,
                optimizer,
            } => (parameters, runtime, optimizer, None),
            NeuralModelState::BurnAdamW {
                parameters,
                runtime,
                optimizer,
            } => {
                let recurrent_outputs = parameters.recurrent.outputs;
                optimizer.validate(a, recurrent_outputs)?;
                (
                    parameters,
                    runtime,
                    BpttOptimizerState::empty(a, recurrent_outputs),
                    Some(optimizer),
                )
            }
            NeuralModelState::External { .. } => {
                return Err(MidasError::Checkpoint("expected Burn neural state".into()));
            }
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
        optimizer.validate(a, lifetime.recurrent.outputs)?;
        let burn_device = Default::default();
        Ok(Box::new(BurnNeuralModel {
            architecture: a.clone(),
            cached: BurnRuntimeParameters::from_lifetime(&lifetime, &burn_device),
            lifetime,
            runtime,
            optimizer,
            adamw_optimizer,
        }))
    }
}
pub(super) struct EncodedState {
    pub(super) representation: Vec<f32>,
    pub(super) next_runtime_state: NeuralRuntimeState,
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
        Ok(if let Some(optimizer) = &self.adamw_optimizer {
            NeuralModelState::BurnAdamW {
                parameters: Box::new(self.lifetime.clone()),
                runtime: self.runtime.clone(),
                optimizer: optimizer.clone(),
            }
        } else {
            NeuralModelState::BurnBptt {
                parameters: Box::new(self.lifetime.clone()),
                runtime: self.runtime.clone(),
                optimizer: self.optimizer.clone(),
            }
        })
    }
    fn fork(&self) -> Result<Box<dyn NeuralModel>> {
        Ok(Box::new(Self {
            architecture: self.architecture.clone(),
            lifetime: self.lifetime.clone(),
            runtime: self.runtime.clone(),
            cached: self.cached.clone(),
            optimizer: self.optimizer.clone(),
            adamw_optimizer: self.adamw_optimizer.clone(),
        }))
    }
    fn runtime_parameter_builds(&self) -> u64 {
        self.cached.builds
    }
    fn train_batch(
        &mut self,
        batch: &crate::learning::PpoTrainingBatch,
        config: &LearningConfig,
        optimizer: &OptimizerConfig,
    ) -> Result<crate::learning::TrainingMetrics> {
        super::head_training::train_burn_encoder(self, batch, config, optimizer)
    }
    fn optimizer_state(&self) -> Result<BpttOptimizerState> {
        if self.adamw_optimizer.is_some() {
            return Err(MidasError::Policy(
                "AdamW state is not representable as momentum optimizer state".into(),
            ));
        }
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
    pub(super) fn encode_observation(
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
        let mut x = Tensor::<InferBackend, 2>::from_data(
            TensorData::new(
                observation.values.clone(),
                [1, self.architecture.observation_size],
            ),
            &device,
        );
        for l in &self.cached.hidden {
            x = activate(dense_runtime(x, l), self.architecture.activation);
        }
        let memory = Tensor::<InferBackend, 2>::from_data(
            TensorData::new(
                recurrent_state.memory.clone(),
                [1, self.architecture.memory.size],
            ),
            &device,
        );
        let next = recurrent_runtime_step(x, memory, &self.cached.recurrent);
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

    pub(super) fn evaluate_representation(&self, representation: &[f32]) -> Result<NeuralOutput> {
        if representation.len() != self.architecture.memory.size
            || representation.iter().any(|value| !value.is_finite())
        {
            return Err(MidasError::Policy(
                "invalid encoded neural representation".into(),
            ));
        }
        let device = Default::default();
        let state = Tensor::<InferBackend, 2>::from_data(
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
fn dense_runtime(x: Tensor<InferBackend, 2>, l: &BurnRuntimeDense) -> Tensor<InferBackend, 2> {
    x.matmul(l.weights.clone()) + l.bias.clone()
}
fn recurrent_runtime_step(
    input: Tensor<InferBackend, 2>,
    memory: Tensor<InferBackend, 2>,
    recurrent: &BurnRuntimeRecurrent,
) -> Tensor<InferBackend, 2> {
    match recurrent {
        BurnRuntimeRecurrent::Simple(layer) => {
            activation::silu(dense_runtime(Tensor::cat(vec![input, memory], 1), layer))
        }
        BurnRuntimeRecurrent::Gru(gru) => {
            let combined = Tensor::cat(vec![input.clone(), memory.clone()], 1);
            let update = activation::sigmoid(
                combined.clone().matmul(gru.update_weights.clone()) + gru.update_bias.clone(),
            );
            let reset = activation::sigmoid(
                combined.matmul(gru.reset_weights.clone()) + gru.reset_bias.clone(),
            );
            let candidate = (input.matmul(gru.candidate_input_weights.clone())
                + (reset * memory.clone()).matmul(gru.candidate_hidden_weights.clone())
                + gru.candidate_bias.clone())
            .tanh();
            update.clone() * memory + (update * -1.0 + 1.0) * candidate
        }
    }
}
fn activate(x: Tensor<InferBackend, 2>, a: Activation) -> Tensor<InferBackend, 2> {
    match a {
        Activation::Relu => activation::relu(x),
        Activation::Gelu => activation::gelu(x),
        Activation::Silu => activation::silu(x),
    }
}
fn to_vec(x: Tensor<InferBackend, 2>) -> Result<Vec<f32>> {
    x.into_data()
        .to_vec::<f32>()
        .map_err(|e| MidasError::Policy(format!("{e:?}")))
}
