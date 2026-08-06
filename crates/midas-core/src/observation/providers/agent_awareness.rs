use crate::Result;

use super::super::{
    contracts::{ObservationContext, ObservationMetadata, ObservationProvider, ObservationSink},
    schema::{
        AGENT_AWARENESS_PROVIDER, AgentAwarenessFeatureSpec, ObservationSchema, ObservationSource,
    },
};
use super::metadata_for_entries;

pub(super) struct AgentAwarenessProvider {
    metadata: Vec<ObservationMetadata>,
    outputs: Vec<(String, AgentAwarenessFeatureSpec)>,
}

impl AgentAwarenessProvider {
    pub(super) fn new(schema: &ObservationSchema) -> Result<Self> {
        let mut outputs = Vec::new();
        for entry in schema
            .entries
            .iter()
            .filter(|entry| entry.provider == AGENT_AWARENESS_PROVIDER)
        {
            let Some(ObservationSource::AgentAwareness { feature }) = entry.source.as_ref() else {
                return Err(crate::MidasError::Config(format!(
                    "agent-awareness observation {} has no agent-awareness source",
                    entry.name
                )));
            };
            outputs.push((entry.name.clone(), feature.clone()));
        }
        Ok(Self {
            metadata: metadata_for_entries(
                schema,
                AGENT_AWARENESS_PROVIDER,
                "agent self-awareness",
                "ratio",
            ),
            outputs,
        })
    }
}

impl ObservationProvider for AgentAwarenessProvider {
    fn provider_name(&self) -> &str {
        AGENT_AWARENESS_PROVIDER
    }

    fn metadata(&self) -> Vec<ObservationMetadata> {
        self.metadata.clone()
    }

    fn observe(
        &self,
        context: &ObservationContext<'_>,
        sink: &mut ObservationSink<'_>,
    ) -> Result<()> {
        for (name, feature) in &self.outputs {
            let value = match feature {
                AgentAwarenessFeatureSpec::RecentReward => context.recent_reward,
                AgentAwarenessFeatureSpec::RollingMeanReward => context.rolling_mean_reward,
                AgentAwarenessFeatureSpec::RollingRewardVolatility => {
                    context.rolling_reward_volatility
                }
                AgentAwarenessFeatureSpec::ExplorationLevel => {
                    if context.initial_temperature > 0.0 {
                        f64::from(context.current_temperature / context.initial_temperature)
                    } else {
                        1.0
                    }
                }
                AgentAwarenessFeatureSpec::SurvivalCostRatio => {
                    context.daily_survival_cost / context.portfolio.net_worth.max(1e-9)
                }
                AgentAwarenessFeatureSpec::RewardTrend => context.reward_trend,
            };
            sink.publish_scalar(name, value)?;
        }
        Ok(())
    }
}
