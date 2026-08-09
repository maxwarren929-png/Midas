//! Neural-policy contracts, trainable encoder runtime, and built-in policies.
//!
//! This facade preserves the stable `midas_core::brain::*` API while keeping
//! implementation details in focused private modules.

mod architecture;
mod benchmarks;
mod burn_runtime;
mod genome;
mod head_training;
mod impulse_beta;
mod model;
mod policy;
mod state;

#[cfg(test)]
mod tests;

pub use architecture::{BrainArchitecture, MemoryArchitecture};
pub use benchmarks::{
    BuyAndHoldPolicy, CashPolicy, EqualWeightPolicy, MomentumPolicy, RandomPolicy,
};
pub use burn_runtime::{BurnNeuralModel, BurnNeuralModelFactory};
pub use genome::{BrainGenome, DenseGenome, LearningTraits};
pub(crate) use model::{ComputeBackend, ComputeDeviceOptions};
pub use model::{ComputeDevice, NeuralModel, NeuralModelFactory};
pub use policy::{NeuralPolicy, Policy, PolicyBlueprint, PolicyContext, PolicySnapshot};
pub use state::{
    AdamWDenseState, AdamWOptimizerState, BpttOptimizerState, DenseOptimizerState,
    LifetimeParameters, NeuralInput, NeuralModelState, NeuralOutput, NeuralRuntimeState,
    OptimizerState, PolicyActionSample, PolicyEvaluation, TrainableEncoderOptimizerState,
};
