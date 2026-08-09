use serde::{Deserialize, Serialize};

const GOLDEN_RATIO: u64 = 0x9e37_79b9_7f4a_7c15;
const LEARNING_OFFSET: u64 = 0x517c_c1b7_2722_0a95;
const EVOLUTION_OFFSET: u64 = 0x243f_6a88_85a3_08d3;

/// Versioned deterministic seed derivation for independent simulation streams.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SeedDerivation {
    /// Preserve the original XOR formulas for existing configurations.
    #[default]
    LegacyXor,
    /// Mix the root, domain, subject, and optional entropy independently.
    SplitmixV2,
}

impl SeedDerivation {
    pub fn is_legacy(&self) -> bool {
        *self == Self::LegacyXor
    }

    pub fn derive(self, root: u64, domain: SeedDomain, subject: u64) -> u64 {
        self.derive_with_entropy(root, domain, subject, 0)
    }

    pub fn derive_with_entropy(
        self,
        root: u64,
        domain: SeedDomain,
        subject: u64,
        entropy: u64,
    ) -> u64 {
        match self {
            Self::LegacyXor => domain.legacy(root, subject, entropy),
            Self::SplitmixV2 => {
                let root = splitmix64(root ^ 0x4d49_4441_535f_5632);
                let domain = splitmix64(root ^ domain.tag());
                let subject = splitmix64(domain ^ subject);
                splitmix64(subject ^ entropy)
            }
        }
    }

    /// Legacy evaluation continues from its serialized policy RNG; V2 starts a
    /// dedicated evaluation stream independent of training RNG consumption.
    pub fn then_evaluation_seed(self, root: u64, subject: u64) -> Option<u64> {
        (self == Self::SplitmixV2).then(|| self.derive(root, SeedDomain::EvaluationPolicy, subject))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SeedDomain {
    InitialGenome,
    Policy,
    Learning,
    Evolution,
    Replacement,
    ReplacementGenome,
    Selection,
    Crossover,
    Mutation,
    EvaluationPolicy,
    EvaluationEvolution,
}

impl SeedDomain {
    const fn tag(self) -> u64 {
        match self {
            Self::InitialGenome => 0x494e_4954_4745_4e4f,
            Self::Policy => 0x504f_4c49_4359_0001,
            Self::Learning => 0x4c45_4152_4e49_4e47,
            Self::Evolution => 0x4556_4f4c_5554_494f,
            Self::Replacement => 0x5245_504c_4143_4501,
            Self::ReplacementGenome => 0x5245_504c_4745_4e4f,
            Self::Selection => 0x5345_4c45_4354_0001,
            Self::Crossover => 0x4352_4f53_534f_5645,
            Self::Mutation => 0x4d55_5441_5445_0001,
            Self::EvaluationPolicy => 0x4556_414c_504f_4c49,
            Self::EvaluationEvolution => 0x4556_414c_4556_4f4c,
        }
    }

    const fn legacy(self, root: u64, subject: u64, entropy: u64) -> u64 {
        match self {
            Self::InitialGenome | Self::Policy => root ^ subject,
            Self::Learning => root ^ subject.wrapping_add(LEARNING_OFFSET),
            Self::Evolution | Self::EvaluationEvolution => root ^ EVOLUTION_OFFSET,
            Self::Replacement => entropy ^ root ^ subject.wrapping_mul(GOLDEN_RATIO),
            Self::ReplacementGenome => root,
            Self::Selection => root ^ GOLDEN_RATIO ^ 0x5345_4c45_4354,
            Self::Crossover => root ^ GOLDEN_RATIO ^ 0x4352_4f53_534f,
            Self::Mutation => root ^ GOLDEN_RATIO ^ 0x4d55_5441_5445,
            // Legacy evaluation retained the RNG serialized in the policy snapshot.
            Self::EvaluationPolicy => root ^ subject,
        }
    }
}

fn splitmix64(value: u64) -> u64 {
    let mut value = value.wrapping_add(GOLDEN_RATIO);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn legacy_derivation_preserves_existing_formulas() {
        let root = 41;
        let id = 7;
        let entropy = 99;
        assert_eq!(
            SeedDerivation::LegacyXor.derive(root, SeedDomain::Policy, id),
            root ^ id
        );
        assert_eq!(
            SeedDerivation::LegacyXor.derive(root, SeedDomain::Learning, id),
            root ^ id.wrapping_add(LEARNING_OFFSET)
        );
        assert_eq!(
            SeedDerivation::LegacyXor.derive_with_entropy(
                root,
                SeedDomain::Replacement,
                id,
                entropy
            ),
            entropy ^ root ^ id.wrapping_mul(GOLDEN_RATIO)
        );
    }

    #[test]
    fn splitmix_v2_is_deterministic_and_domain_separated() {
        let version = SeedDerivation::SplitmixV2;
        let policy = version.derive(41, SeedDomain::Policy, 7);
        assert_eq!(policy, version.derive(41, SeedDomain::Policy, 7));
        assert_ne!(policy, version.derive(41, SeedDomain::Learning, 7));
        assert_ne!(policy, version.derive(41, SeedDomain::Policy, 8));
    }

    #[test]
    fn splitmix_v2_adjacent_roots_do_not_alias_agent_seed_sets() {
        let version = SeedDerivation::SplitmixV2;
        let sets = (41..=45)
            .map(|root| {
                (0..10)
                    .map(|id| version.derive(root, SeedDomain::InitialGenome, id))
                    .collect::<BTreeSet<_>>()
            })
            .collect::<Vec<_>>();
        for left in 0..sets.len() {
            for right in left + 1..sets.len() {
                assert!(sets[left].is_disjoint(&sets[right]));
            }
        }
    }
}
