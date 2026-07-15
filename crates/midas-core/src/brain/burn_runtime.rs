use super::{
    BrainArchitecture, BrainGenome, ComputeDevice, DenseGenome, LifetimeParameters, NeuralInput,
    NeuralModel, NeuralModelFactory, NeuralModelState, NeuralOutput, NeuralRuntimeState,
    OptimizerState,
};
use crate::{
    MidasError, Result,
    config::{ActionDistributionConfig, Activation, LearningConfig},
    observation::Observation,
};
use burn::{
    backend::NdArray,
    tensor::{Tensor, TensorData, activation},
};

type CpuBackend = NdArray<f32>;

pub struct BurnNeuralModel {
    pub(super) architecture: BrainArchitecture,
    pub(super) lifetime: LifetimeParameters,
    pub(super) runtime: NeuralRuntimeState,
    pub(super) cached: BurnRuntimeParameters,
    pub(super) optimizer: OptimizerState,
}
#[derive(Clone)]
struct BurnRuntimeDense {
    weights: Tensor<CpuBackend, 2>,
    bias: Tensor<CpuBackend, 2>,
}
#[derive(Clone)]
pub(super) struct BurnRuntimeParameters {
    hidden: Vec<BurnRuntimeDense>,
    recurrent: BurnRuntimeDense,
    actor: BurnRuntimeDense,
    critic: BurnRuntimeDense,
    builds: u64,
}
impl BurnRuntimeParameters {
    pub(super) fn from_lifetime(
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

    pub(super) fn with_updated_heads(
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
        super::head_training::train_burn_heads(self, batch, config)
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

    pub(super) fn evaluate_representation(&self, representation: &[f32]) -> Result<NeuralOutput> {
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
