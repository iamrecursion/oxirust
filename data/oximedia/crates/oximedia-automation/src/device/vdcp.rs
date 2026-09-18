//! VDCP (Video Disk Control Protocol) device implementation.

use crate::protocol::vdcp::VdcpProtocol;
use crate::{AutomationError, Result};
use tracing::info;

/// VDCP device controller.
pub struct VdcpDevice {
    port: String,
    protocol: Option<VdcpProtocol>,
}

impl VdcpDevice {
    /// Create a new VDCP device.
    pub async fn new(port: &str) -> Result<Self> {
        info!("Creating VDCP device on port: {}", port);

        Ok(Self {
            port: port.to_string(),
            protocol: None,
        })
    }

    /// Connect to the device.
    pub async fn connect(&mut self) -> Result<()> {
        info!("Connecting to VDCP device on {}", self.port);

        let protocol = VdcpProtocol::new(&self.port).await?;
        self.protocol = Some(protocol);

        Ok(())
    }

    /// Disconnect from the device.
    pub async fn disconnect(&mut self) -> Result<()> {
        info!("Disconnecting from VDCP device on {}", self.port);

        if let Some(ref mut protocol) = self.protocol {
            protocol.close().await?;
        }
        self.protocol = None;

        Ok(())
    }

    /// Send play command.
    pub async fn play(&mut self) -> Result<()> {
        if let Some(ref mut protocol) = self.protocol {
            protocol.send_play().await
        } else {
            Err(AutomationError::DeviceControl(
                "Device not connected".to_string(),
            ))
        }
    }

    /// Send stop command.
    pub async fn stop(&mut self) -> Result<()> {
        if let Some(ref mut protocol) = self.protocol {
            protocol.send_stop().await
        } else {
            Err(AutomationError::DeviceControl(
                "Device not connected".to_string(),
            ))
        }
    }

    /// Send cue command.
    pub async fn cue(&mut self, timecode: &str) -> Result<()> {
        if let Some(ref mut protocol) = self.protocol {
            protocol.send_cue(timecode).await
        } else {
            Err(AutomationError::DeviceControl(
                "Device not connected".to_string(),
            ))
        }
    }

    /// Get device status.
    pub async fn status(&mut self) -> Result<String> {
        if let Some(ref mut protocol) = self.protocol {
            protocol.get_status().await
        } else {
            Err(AutomationError::DeviceControl(
                "Device not connected".to_string(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_vdcp_device_creation() {
        let device = VdcpDevice::new("/dev/ttyS0").await;
        assert!(device.is_ok());
    }
}
