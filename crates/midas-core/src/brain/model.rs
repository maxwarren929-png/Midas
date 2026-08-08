use super::{
    BpttOptimizerState, BrainArchitecture, BrainGenome, LifetimeParameters, NeuralInput,
    NeuralModelState, NeuralOutput, NeuralRuntimeState,
};
use crate::{
    MidasError, Result,
    config::{ActionDistributionConfig, LearningConfig},
};

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
    fn optimizer_state(&self) -> Result<BpttOptimizerState> {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ComputeBackend {
    CpuNdArray,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ComputeDeviceOptions<'a> {
    pub(crate) backend: ComputeBackend,
    pub(crate) device: &'a str,
}

impl ComputeDevice {
    pub const CPU_NDARRAY_DEFAULT: &'static str = "default";

    pub(crate) fn from_options(options: ComputeDeviceOptions<'_>) -> Result<Self> {
        if options.device.trim().is_empty() {
            return Err(MidasError::Config(
                "backend.device must not be empty".into(),
            ));
        }
        let device = match options.backend {
            ComputeBackend::CpuNdArray => Self::CpuNdArray {
                device: options.device.to_owned(),
            },
        };
        device.validate()?;
        Ok(device)
    }

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
