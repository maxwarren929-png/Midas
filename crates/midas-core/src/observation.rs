//! Versioned, named observations assembled from read-only sensor providers.
//!
//! Providers publish raw named values. [`ObservationSchema`] owns the only
//! mapping from those names to neural-input indices and applies normalisation.
//! This facade preserves the public `midas_core::observation` API while the
//! implementation is separated into schema, registry, builder, and provider
//! modules.

mod builder;
mod contracts;
mod normalization;
mod providers;
mod registry;
mod schema;

pub use builder::{ObservationBuilder, SchemaObservationBuilder};
pub use contracts::{
    Observation, ObservationContext, ObservationMetadata, ObservationProvider, ObservationShape,
    ObservationSink, ObservationValueType,
};
pub use normalization::NormalizationRule;
pub use registry::ObservationRegistry;
pub use schema::{
    AgentStateFeatureSpec, ObservationFeatureSpec, ObservationSchema, ObservationSchemaEntry,
    ObservationSource, PortfolioFeatureSpec, RollingMarketMetric,
};

#[cfg(test)]
mod tests;
