use super::{Policy, PolicyBlueprint, PolicyContext, PolicySnapshot};
use crate::{
    Result,
    action::{ProposedAction, ProposedAllocation},
    observation::Observation,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

pub struct CashPolicy;
impl Policy for CashPolicy {
    fn propose_action(
        &mut self,
        _observation: &Observation,
        c: &PolicyContext<'_>,
    ) -> Result<ProposedAction> {
        let mut w = vec![0.0; c.tradable.len() + 1];
        w[0] = 1.0;
        Ok(ProposedAction::Allocation(ProposedAllocation::rebalance(w)))
    }
    fn snapshot(&self) -> Result<PolicySnapshot> {
        Ok(PolicySnapshot::Cash)
    }
    fn reset_episode_state(&mut self) {}
    fn policy_blueprint(&self) -> PolicyBlueprint {
        PolicyBlueprint::Cash
    }
    fn fork(&self) -> Result<Box<dyn Policy>> {
        Ok(Box::new(CashPolicy))
    }
}
pub struct EqualWeightPolicy;
impl Policy for EqualWeightPolicy {
    fn propose_action(
        &mut self,
        observation: &Observation,
        c: &PolicyContext<'_>,
    ) -> Result<ProposedAction> {
        let n = c.tradable.iter().filter(|v| **v).count();
        if n == 0 {
            return CashPolicy.propose_action(observation, c);
        }
        let mut w = vec![0.0; c.tradable.len() + 1];
        for (i, t) in c.tradable.iter().enumerate() {
            if *t {
                w[i + 1] = 1.0 / n as f64
            }
        }
        Ok(ProposedAction::Allocation(ProposedAllocation::rebalance(w)))
    }
    fn snapshot(&self) -> Result<PolicySnapshot> {
        Ok(PolicySnapshot::EqualWeight)
    }
    fn reset_episode_state(&mut self) {}
    fn policy_blueprint(&self) -> PolicyBlueprint {
        PolicyBlueprint::EqualWeight
    }
    fn fork(&self) -> Result<Box<dyn Policy>> {
        Ok(Box::new(EqualWeightPolicy))
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
    fn propose_action(&mut self, _: &Observation, c: &PolicyContext<'_>) -> Result<ProposedAction> {
        Ok(ProposedAction::Allocation(ProposedAllocation::rebalance(
            (0..=c.tradable.len())
                .map(|_| self.rng.gen_range(0.0001..1.0))
                .collect(),
        )))
    }
    fn snapshot(&self) -> Result<PolicySnapshot> {
        Ok(PolicySnapshot::Random {
            rng: self.rng.clone(),
        })
    }
    fn reset_episode_state(&mut self) {}
    fn policy_blueprint(&self) -> PolicyBlueprint {
        PolicyBlueprint::Random
    }
    fn fork(&self) -> Result<Box<dyn Policy>> {
        Ok(Box::new(Self {
            rng: self.rng.clone(),
        }))
    }
}
pub struct BuyAndHoldPolicy {
    pub(crate) initialized: bool,
    pub(crate) target_asset: usize,
}

pub struct MomentumPolicy {
    pub(crate) signal_indices: Vec<usize>,
    pub(crate) minimum_signal: f32,
}

impl MomentumPolicy {
    pub fn new(signal_indices: Vec<usize>, minimum_signal: f32) -> Self {
        Self {
            signal_indices,
            minimum_signal,
        }
    }
}

impl Policy for MomentumPolicy {
    fn propose_action(
        &mut self,
        observation: &Observation,
        c: &PolicyContext<'_>,
    ) -> Result<ProposedAction> {
        let selected = self
            .signal_indices
            .iter()
            .copied()
            .enumerate()
            .filter(|(asset_index, _)| c.tradable.get(*asset_index).copied().unwrap_or(false))
            .filter_map(|(asset_index, observation_index)| {
                observation
                    .values
                    .get(observation_index)
                    .copied()
                    .filter(|signal| signal.is_finite() && *signal > self.minimum_signal)
                    .map(|signal| (asset_index, signal))
            })
            .max_by(|left, right| {
                left.1
                    .total_cmp(&right.1)
                    .then_with(|| right.0.cmp(&left.0))
            })
            .map(|(asset_index, _)| asset_index);

        let mut weights = vec![0.0; c.tradable.len() + 1];
        if let Some(asset_index) = selected {
            weights[asset_index + 1] = 1.0;
        } else {
            weights[0] = 1.0;
        }
        Ok(ProposedAction::Allocation(ProposedAllocation::rebalance(
            weights,
        )))
    }

    fn snapshot(&self) -> Result<PolicySnapshot> {
        Ok(PolicySnapshot::Momentum {
            signal_indices: self.signal_indices.clone(),
            minimum_signal: self.minimum_signal,
        })
    }

    fn reset_episode_state(&mut self) {}

    fn policy_blueprint(&self) -> PolicyBlueprint {
        PolicyBlueprint::Momentum {
            signal_indices: self.signal_indices.clone(),
            minimum_signal_bits: self.minimum_signal.to_bits(),
        }
    }

    fn fork(&self) -> Result<Box<dyn Policy>> {
        Ok(Box::new(Self::new(
            self.signal_indices.clone(),
            self.minimum_signal,
        )))
    }
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
    fn propose_action(&mut self, _: &Observation, c: &PolicyContext<'_>) -> Result<ProposedAction> {
        if self.initialized {
            return Ok(ProposedAction::Allocation(ProposedAllocation::hold(
                c.current_allocations.to_vec(),
            )));
        }
        self.initialized = true;
        let mut w = vec![0.0; c.tradable.len() + 1];
        if c.tradable.get(self.target_asset).copied().unwrap_or(false) {
            w[self.target_asset + 1] = 1.0
        } else {
            w[0] = 1.0
        }
        Ok(ProposedAction::Allocation(ProposedAllocation::rebalance(w)))
    }
    fn snapshot(&self) -> Result<PolicySnapshot> {
        Ok(PolicySnapshot::BuyAndHold {
            initialized: self.initialized,
            target_asset: self.target_asset,
        })
    }
    fn reset_episode_state(&mut self) {
        self.initialized = false
    }
    fn policy_blueprint(&self) -> PolicyBlueprint {
        PolicyBlueprint::BuyAndHold {
            target_asset: self.target_asset,
        }
    }
    fn fork(&self) -> Result<Box<dyn Policy>> {
        Ok(Box::new(Self {
            initialized: self.initialized,
            target_asset: self.target_asset,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weights(action: ProposedAction) -> Vec<f64> {
        match action {
            ProposedAction::Allocation(allocation) => allocation.weights,
            ProposedAction::TradeImpulses(_) => panic!("benchmark returned trade impulses"),
        }
    }

    #[test]
    fn momentum_uses_only_configured_current_observation_values() {
        let observation = Observation {
            schema_hash: "schema".into(),
            values: vec![0.2, 99.0, 0.4, -99.0, 0.4],
        };
        let current = [1.0, 0.0, 0.0, 0.0];
        let mut policy = MomentumPolicy::new(vec![0, 2, 4], 0.0);

        let tied = policy
            .propose_action(
                &observation,
                &PolicyContext {
                    tradable: &[true, true, true],
                    current_allocations: &current,
                },
            )
            .unwrap();
        assert_eq!(weights(tied), vec![0.0, 0.0, 1.0, 0.0]);

        let unavailable_leader = policy
            .propose_action(
                &observation,
                &PolicyContext {
                    tradable: &[true, false, true],
                    current_allocations: &current,
                },
            )
            .unwrap();
        assert_eq!(weights(unavailable_leader), vec![0.0, 0.0, 0.0, 1.0]);

        policy.minimum_signal = 0.5;
        let below_threshold = policy
            .propose_action(
                &observation,
                &PolicyContext {
                    tradable: &[true, true, true],
                    current_allocations: &current,
                },
            )
            .unwrap();
        assert_eq!(weights(below_threshold), vec![1.0, 0.0, 0.0, 0.0]);
    }
}
