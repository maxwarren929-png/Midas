use crate::Result;

use super::super::{
    contracts::{ObservationContext, ObservationMetadata, ObservationProvider, ObservationSink},
    schema::{AGENT_STATE_PROVIDER, AgentStateFeatureSpec, ObservationSchema, ObservationSource},
};
use super::metadata_for_entries;

pub(super) struct AgentStateObservationProvider {
    metadata: Vec<ObservationMetadata>,
    outputs: Vec<(String, AgentStateFeatureSpec)>,
}

impl AgentStateObservationProvider {
    pub(super) fn new(schema: &ObservationSchema) -> Result<Self> {
        let mut outputs = Vec::new();
        for entry in schema
            .entries
            .iter()
            .filter(|entry| entry.provider == AGENT_STATE_PROVIDER)
        {
            let Some(ObservationSource::AgentState { feature }) = entry.source.as_ref() else {
                return Err(crate::MidasError::Config(format!(
                    "agent-state observation {} has no agent-state source",
                    entry.name
                )));
            };
            outputs.push((entry.name.clone(), feature.clone()));
        }
        Ok(Self {
            metadata: metadata_for_entries(schema, AGENT_STATE_PROVIDER, "agent state", "ratio"),
            outputs,
        })
    }
}

impl ObservationProvider for AgentStateObservationProvider {
    fn provider_name(&self) -> &str {
        AGENT_STATE_PROVIDER
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
                AgentStateFeatureSpec::NormalizedAge => {
                    context.age as f64 / context.episode_steps.max(1) as f64
                }
            };
            sink.publish_scalar(name, value)?;
        }
        Ok(())
    }
}
