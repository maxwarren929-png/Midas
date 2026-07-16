use crate::{MidasError, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackendConfig {
    pub kind: BackendKind,
    pub device: String,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    #[serde(alias = "cpu")]
    Cpu,
}

impl BackendConfig {
    pub fn compute_device(&self) -> Result<crate::brain::ComputeDevice> {
        if self.device.trim().is_empty() {
            return Err(MidasError::Config(
                "backend.device must not be empty".into(),
            ));
        }
        let device = match self.kind {
            BackendKind::Cpu => crate::brain::ComputeDevice::CpuNdArray {
                device: self.device.clone(),
            },
        };
        device.validate()?;
        Ok(device)
    }
}
