use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointConfig {
    pub path: String,
    pub every_steps: usize,
}
impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            path: "checkpoints/latest.bin".into(),
            every_steps: 0,
        }
    }
}
