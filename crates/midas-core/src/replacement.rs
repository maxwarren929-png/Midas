use crate::{
    AgentId, Result,
    agent::{AgentSpawnRequest, AgentState, SpawnMetadata},
    brain::{BrainArchitecture, BrainGenome, ComputeDevice, PolicyBlueprint},
    config::Config,
    observation::ObservationSchema,
};

#[derive(Clone, Debug)]
pub struct LivingAgentSummary<'a> {
    pub id: AgentId,
    pub net_worth: f64,
    pub age: u64,
    pub policy: PolicyBlueprint,
    pub genome: Option<&'a BrainGenome>,
    pub maximum_drawdown: f64,
    pub birth_date: chrono::NaiveDate,
    pub parent_ids: Vec<AgentId>,
}
pub struct ReplacementContext<'a> {
    pub dead_agent: &'a AgentState,
    pub dead_policy: PolicyBlueprint,
    pub living_population: &'a [LivingAgentSummary<'a>],
    pub config: &'a Config,
    pub architecture: &'a BrainArchitecture,
    pub compute_device: &'a ComputeDevice,
    pub observation_schema: &'a ObservationSchema,
    pub deterministic_seed: u64,
}
pub trait ReplacementStrategy: Send {
    fn create_replacement(&mut self, context: &ReplacementContext<'_>)
    -> Result<AgentSpawnRequest>;
    fn name(&self) -> &'static str;
}
pub struct NonEvolutionaryRandomGenomeReplacement;
impl ReplacementStrategy for NonEvolutionaryRandomGenomeReplacement {
    fn create_replacement(&mut self, c: &ReplacementContext<'_>) -> Result<AgentSpawnRequest> {
        let genome = if c.dead_policy == PolicyBlueprint::Neural {
            Some(BrainGenome::random(
                c.architecture,
                &c.config.brain.traits,
                c.deterministic_seed,
            )?)
        } else {
            None
        };
        Ok(AgentSpawnRequest {
            policy_blueprint: c.dead_policy.clone(),
            genome,
            parent_ids: Vec::new(),
            metadata: SpawnMetadata {
                reason: "non_evolutionary_replacement".into(),
            },
        })
    }
    fn name(&self) -> &'static str {
        "non_evolutionary_random_genome"
    }
}
