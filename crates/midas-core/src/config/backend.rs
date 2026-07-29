use crate::brain::{ComputeBackend, ComputeDeviceOptions};
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
    pub(crate) fn compute_device_options(&self) -> ComputeDeviceOptions<'_> {
        ComputeDeviceOptions {
            backend: match self.kind {
                BackendKind::Cpu => ComputeBackend::CpuNdArray,
            },
            device: &self.device,
        }
    }
}
