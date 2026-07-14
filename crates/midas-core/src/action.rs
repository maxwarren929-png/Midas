use crate::{MidasError, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ProposedAllocation {
    pub weights: Vec<f64>,
    pub rebalance: bool,
}

impl ProposedAllocation {
    pub fn rebalance(weights: Vec<f64>) -> Self {
        Self {
            weights,
            rebalance: true,
        }
    }
    pub fn hold(weights: Vec<f64>) -> Self {
        Self {
            weights,
            rebalance: false,
        }
    }
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ValidatedAllocation {
    weights: Vec<f64>,
    rebalance: bool,
    used_fallback: bool,
    fallback_reason: Option<String>,
}

impl ValidatedAllocation {
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }
    pub fn rebalance(&self) -> bool {
        self.rebalance
    }
    pub fn used_fallback(&self) -> bool {
        self.used_fallback
    }
    pub fn fallback_reason(&self) -> Option<&str> {
        self.fallback_reason.as_deref()
    }
}

pub struct ActionValidationContext<'a> {
    pub asset_count: usize,
    pub tradable: &'a [bool],
    /// A cap supplied by the policy or its inherited genome. Benchmarks leave
    /// this unset because their documented allocations are not neural-policy
    /// proposals.
    pub policy_maximum_asset_allocation: Option<f64>,
    /// An explicit market-wide cap, when an experiment chooses to impose one
    /// on every policy including benchmarks.
    pub market_maximum_asset_allocation: Option<f64>,
}

pub trait ActionValidator: Send + Sync {
    fn validate(
        &self,
        proposed: ProposedAllocation,
        context: &ActionValidationContext<'_>,
    ) -> Result<ValidatedAllocation>;
}

#[derive(Default)]
pub struct LongOnlyActionValidator;

impl LongOnlyActionValidator {
    fn cash(asset_count: usize, used_fallback: bool) -> ValidatedAllocation {
        let mut weights = vec![0.0; asset_count + 1];
        weights[0] = 1.0;
        ValidatedAllocation {
            weights,
            rebalance: true,
            used_fallback,
            fallback_reason: used_fallback
                .then_some("invalid or unavailable allocation proposal".into()),
        }
    }
}

impl ActionValidator for LongOnlyActionValidator {
    fn validate(
        &self,
        proposed: ProposedAllocation,
        context: &ActionValidationContext<'_>,
    ) -> Result<ValidatedAllocation> {
        if context.tradable.len() != context.asset_count {
            return Err(MidasError::Config(
                "invalid action-validation context".into(),
            ));
        }
        let cap = match (
            context.policy_maximum_asset_allocation,
            context.market_maximum_asset_allocation,
        ) {
            (Some(policy), Some(market)) => Some(policy.min(market)),
            (Some(policy), None) => Some(policy),
            (None, Some(market)) => Some(market),
            (None, None) => None,
        };
        if cap.is_some_and(|value| {
            !value.is_finite() || !(0.0..=1.0).contains(&value) || value == 0.0
        }) {
            return Err(MidasError::Config(
                "invalid action-validation allocation cap".into(),
            ));
        }
        if proposed.weights.len() != context.asset_count + 1
            || proposed.weights.iter().any(|v| !v.is_finite() || *v < 0.0)
        {
            let mut value = Self::cash(context.asset_count, true);
            value.fallback_reason =
                Some("proposal length, finiteness, or sign validation failed".into());
            return Ok(value);
        }
        let mut weights = proposed.weights;
        for (index, tradable) in context.tradable.iter().enumerate() {
            if !tradable {
                weights[index + 1] = 0.0;
            }
        }
        let sum = weights.iter().sum::<f64>();
        if !sum.is_finite() || sum <= 0.0 {
            let mut value = Self::cash(context.asset_count, true);
            value.fallback_reason = Some("proposal sum was zero or non-finite".into());
            return Ok(value);
        }
        for weight in &mut weights {
            *weight /= sum;
        }
        if let Some(max) = cap {
            loop {
                let mut excess = 0.0;
                for weight in weights.iter_mut().skip(1) {
                    if *weight > max {
                        excess += *weight - max;
                        *weight = max;
                    }
                }
                if excess <= 1e-12 {
                    break;
                }
                let eligible = (1..weights.len())
                    .filter(|i| context.tradable[*i - 1] && weights[*i] < max - 1e-12)
                    .collect::<Vec<_>>();
                if eligible.is_empty() {
                    weights[0] += excess;
                    break;
                }
                let share = excess / (eligible.len() + 1) as f64;
                weights[0] += share;
                for index in eligible {
                    weights[index] += share;
                }
            }
        }
        let sum = weights.iter().sum::<f64>();
        for weight in &mut weights {
            *weight /= sum;
        }
        if let Some(max) = cap {
            let mut rounding_excess = 0.0;
            for weight in weights.iter_mut().skip(1) {
                if *weight > max {
                    rounding_excess += *weight - max;
                    *weight = max;
                }
            }
            weights[0] += rounding_excess;
        }
        if weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
            || (weights.iter().sum::<f64>() - 1.0).abs() > 1e-9
            || cap.is_some_and(|max| weights.iter().skip(1).any(|weight| *weight > max))
        {
            return Err(MidasError::Policy(
                "concentration enforcement failed".into(),
            ));
        }
        Ok(ValidatedAllocation {
            weights,
            rebalance: proposed.rebalance,
            used_fallback: false,
            fallback_reason: None,
        })
    }
}
