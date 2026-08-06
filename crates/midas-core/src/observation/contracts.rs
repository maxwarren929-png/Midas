use crate::{Result, execution::Portfolio, market::FeatureRow};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use super::normalization::NormalizationRule;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Observation {
    pub schema_hash: String,
    pub values: Vec<f32>,
}

/// Immutable data available to providers at close `t`. It deliberately has
/// no reference to policies, learning, execution planners, or mutable state.
pub struct ObservationContext<'a> {
    pub market: &'a FeatureRow,
    pub portfolio: &'a Portfolio,
    pub prices: &'a [Option<f64>],
    pub starting_net_worth: f64,
    pub maximum_net_worth: f64,
    pub age: u64,
    pub episode_steps: usize,
    pub recent_reward: f64,
    pub rolling_mean_reward: f64,
    pub rolling_reward_volatility: f64,
    pub reward_trend: f64,
    pub current_temperature: f32,
    pub initial_temperature: f32,
    pub daily_survival_cost: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ObservationShape {
    Scalar,
    Fixed { dimensions: usize },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ObservationValueType {
    Float64,
}

impl ObservationShape {
    pub(crate) fn dimension(&self) -> usize {
        match self {
            Self::Scalar => 1,
            Self::Fixed { dimensions } => *dimensions,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ObservationMetadata {
    pub name: String,
    pub provider: String,
    pub description: String,
    pub units: String,
    pub value_type: ObservationValueType,
    pub shape: ObservationShape,
    /// A provider may advertise a stable source-order hint. The schema index,
    /// not this hint, is always authoritative for neural input order.
    pub ordering_hint: Option<usize>,
    pub normalization: NormalizationRule,
    pub version: u32,
    pub tags: Vec<String>,
}

impl ObservationMetadata {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.name.trim().is_empty()
            || self.provider.trim().is_empty()
            || self.description.trim().is_empty()
            || self.units.trim().is_empty()
            || self.version == 0
            || self.shape.dimension() == 0
        {
            return Err(crate::MidasError::Config(
                "observation metadata is incomplete".into(),
            ));
        }
        self.normalization.validate()
    }
}

/// A sensor provider. Providers publish raw named values to a registry-owned
/// sink and cannot write the neural vector directly.
pub trait ObservationProvider: Send + Sync {
    fn provider_name(&self) -> &str;
    fn metadata(&self) -> Vec<ObservationMetadata>;
    fn observe(
        &self,
        context: &ObservationContext<'_>,
        sink: &mut ObservationSink<'_>,
    ) -> Result<()>;
}

pub struct ObservationSink<'a> {
    pub(crate) slots: &'a BTreeMap<String, usize>,
    pub(crate) dimensions: &'a [usize],
    pub(crate) values: &'a mut [Vec<f64>],
    pub(crate) present: &'a mut [bool],
}

impl ObservationSink<'_> {
    pub fn publish_scalar(&mut self, name: &str, value: f64) -> Result<()> {
        self.publish(name, &[value])
    }

    pub fn publish(&mut self, name: &str, values: &[f64]) -> Result<()> {
        let slot = self.slots.get(name).copied().ok_or_else(|| {
            crate::MidasError::Policy(format!(
                "provider published unregistered observation {name}"
            ))
        })?;
        if self.present[slot] {
            return Err(crate::MidasError::Policy(format!(
                "observation {name} was published more than once"
            )));
        }
        if values.len() != self.dimensions[slot] || values.iter().any(|value| !value.is_finite()) {
            return Err(crate::MidasError::Policy(format!(
                "observation {name} has invalid dimensions or non-finite values"
            )));
        }
        self.values[slot].clear();
        self.values[slot].extend_from_slice(values);
        self.present[slot] = true;
        Ok(())
    }
}
