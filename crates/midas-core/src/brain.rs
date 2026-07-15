//! Neural-policy contracts, frozen-encoder runtime, and built-in policies.
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
pub use benchmarks::{BuyAndHoldPolicy, CashPolicy, EqualWeightPolicy, RandomPolicy};
pub use burn_runtime::{BurnNeuralModel, BurnNeuralModelFactory};
pub use genome::{BrainGenome, DenseGenome, LearningTraits};
pub use model::{ComputeDevice, NeuralModel, NeuralModelFactory};
pub use policy::{NeuralPolicy, Policy, PolicyBlueprint, PolicyContext, PolicySnapshot};
pub use state::{
    DenseOptimizerState, LifetimeParameters, NeuralInput, NeuralModelState, NeuralOutput,
    NeuralRuntimeState, OptimizerState, PolicyActionSample, PolicyEvaluation,
};
