//! Versioned, named observations assembled from read-only sensor providers.
//!
//! Providers publish raw named values. [`ObservationSchema`] owns the only
//! mapping from those names to neural-input indices and applies normalisation.
//! This facade preserves the public `midas_core::observation` API while the
//! implementation is separated into schema, registry, builder, and provider
//! modules.

mod builder;
mod config_compat;
mod contracts;
mod normalization;
mod options;
mod providers;
mod registry;
mod schema;

pub use builder::{ObservationBuilder, SchemaObservationBuilder};
pub use contracts::{
    Observation, ObservationContext, ObservationMetadata, ObservationProvider, ObservationShape,
    ObservationSink, ObservationValueType,
};
pub use normalization::NormalizationRule;
pub(crate) use options::{ObservationBuilderOptions, ObservationSchemaOptions};
pub use registry::ObservationRegistry;
pub(crate) use schema::AGENT_AWARENESS_PROVIDER;
pub use schema::{
    AgentAwarenessFeatureSpec, AgentStateFeatureSpec, ObservationFeatureSpec, ObservationSchema,
    ObservationSchemaEntry, ObservationSource, PortfolioFeatureSpec, RollingMarketMetric,
};

#[cfg(test)]
mod tests;
