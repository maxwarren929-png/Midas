use super::BackendConfig;
use crate::{Result, brain::ComputeDevice};

impl BackendConfig {
    pub fn compute_device(&self) -> Result<ComputeDevice> {
        ComputeDevice::from_options(self.compute_device_options())
    }
}
