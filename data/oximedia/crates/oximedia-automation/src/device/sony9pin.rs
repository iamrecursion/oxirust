//! Sony 9-pin RS-422 VTR control implementation.

use crate::protocol::sony::SonyProtocol;
use crate::{AutomationError, Result};
use tracing::info;

/// Sony 9-pin device controller.
pub struct Sony9PinDevice {
    port: String,
    protocol: Option<SonyProtocol>,
}

impl Sony9PinDevice {
    /// Create a new Sony 9-pin device.
    pub async fn new(port: &str) -> Result<Self> {
        info!("Creating Sony 9-pin device on port: {}", port);

        Ok(Self {
            port: port.to_string(),
            protocol: None,
        })
    }

    /// Connect to the device.
    pub async fn connect(&mut self) -> Result<()> {
        info!("Connecting to Sony 9-pin device on {}", self.port);

        let protocol = SonyProtocol::new(&self.port).await?;
        self.protocol = Some(protocol);

        Ok(())
    }

    /// Disconnect from the device.
    pub async fn disconnect(&mut self) -> Result<()> {
        info!("Disconnecting from Sony 9-pin device on {}", self.port);

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

    /// Send record command.
    pub async fn record(&mut self) -> Result<()> {
        if let Some(ref mut protocol) = self.protocol {
            protocol.send_record().await
        } else {
            Err(AutomationError::DeviceControl(
                "Device not connected".to_string(),
            ))
        }
    }

    /// Send fast forward command.
    pub async fn fast_forward(&mut self) -> Result<()> {
        if let Some(ref mut protocol) = self.protocol {
            protocol.send_fast_forward().await
        } else {
            Err(AutomationError::DeviceControl(
                "Device not connected".to_string(),
            ))
        }
    }

    /// Send rewind command.
    pub async fn rewind(&mut self) -> Result<()> {
        if let Some(ref mut protocol) = self.protocol {
            protocol.send_rewind().await
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
    async fn test_sony9pin_device_creation() {
        let device = Sony9PinDevice::new("/dev/ttyS0").await;
        assert!(device.is_ok());
    }
}
