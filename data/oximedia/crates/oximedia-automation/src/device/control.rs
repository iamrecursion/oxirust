//! Device control abstraction layer.

use crate::device::{gpi::GpiDevice, gpo::GpoDevice, sony9pin::Sony9PinDevice, vdcp::VdcpDevice};
use crate::{AutomationError, Result};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

/// Device type enumeration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeviceType {
    /// VDCP video disk control
    Vdcp {
        /// Serial port path
        port: String,
    },
    /// Sony 9-pin RS-422 control
    Sony9Pin {
        /// Serial port path
        port: String,
    },
    /// General Purpose Input
    Gpi {
        /// GPIO port path
        port: String,
    },
    /// General Purpose Output
    Gpo {
        /// GPIO port path
        port: String,
    },
}

impl DeviceType {
    /// Get device type name.
    pub fn name(&self) -> &str {
        match self {
            DeviceType::Vdcp { .. } => "vdcp",
            DeviceType::Sony9Pin { .. } => "sony9pin",
            DeviceType::Gpi { .. } => "gpi",
            DeviceType::Gpo { .. } => "gpo",
        }
    }
}

/// Device state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceState {
    /// Device is disconnected
    Disconnected,
    /// Device is connecting
    Connecting,
    /// Device is connected and ready
    Connected,
    /// Device is in error state
    Error,
}

/// Device controller abstraction.
pub struct DeviceController {
    device_type: DeviceType,
    state: DeviceState,
    vdcp: Option<VdcpDevice>,
    sony9pin: Option<Sony9PinDevice>,
    gpi: Option<GpiDevice>,
    gpo: Option<GpoDevice>,
}

impl DeviceController {
    /// Create a new device controller.
    pub async fn new(device_type: DeviceType) -> Result<Self> {
        info!("Creating device controller: {:?}", device_type);

        let mut controller = Self {
            device_type: device_type.clone(),
            state: DeviceState::Disconnected,
            vdcp: None,
            sony9pin: None,
            gpi: None,
            gpo: None,
        };

        // Initialize the appropriate device
        match device_type {
            DeviceType::Vdcp { ref port } => {
                controller.vdcp = Some(VdcpDevice::new(port).await?);
            }
            DeviceType::Sony9Pin { ref port } => {
                controller.sony9pin = Some(Sony9PinDevice::new(port).await?);
            }
            DeviceType::Gpi { ref port } => {
                controller.gpi = Some(GpiDevice::new(port).await?);
            }
            DeviceType::Gpo { ref port } => {
                controller.gpo = Some(GpoDevice::new(port).await?);
            }
        }

        Ok(controller)
    }

    /// Initialize the device.
    pub async fn initialize(&mut self) -> Result<()> {
        info!("Initializing device: {:?}", self.device_type);

        self.state = DeviceState::Connecting;

        let result = match &mut self.vdcp {
            Some(device) => device.connect().await,
            None => match &mut self.sony9pin {
                Some(device) => device.connect().await,
                None => match &mut self.gpi {
                    Some(device) => device.connect().await,
                    None => match &mut self.gpo {
                        Some(device) => device.connect().await,
                        None => Err(AutomationError::DeviceControl(
                            "No device initialized".to_string(),
                        )),
                    },
                },
            },
        };

        match result {
            Ok(()) => {
                self.state = DeviceState::Connected;
                info!("Device initialized successfully");
                Ok(())
            }
            Err(e) => {
                self.state = DeviceState::Error;
                warn!("Device initialization failed: {}", e);
                Err(e)
            }
        }
    }

    /// Release the device.
    pub async fn release(&mut self) -> Result<()> {
        info!("Releasing device: {:?}", self.device_type);

        if let Some(ref mut device) = self.vdcp {
            device.disconnect().await?;
        } else if let Some(ref mut device) = self.sony9pin {
            device.disconnect().await?;
        } else if let Some(ref mut device) = self.gpi {
            device.disconnect().await?;
        } else if let Some(ref mut device) = self.gpo {
            device.disconnect().await?;
        }

        self.state = DeviceState::Disconnected;
        Ok(())
    }

    /// Get device state.
    pub fn state(&self) -> DeviceState {
        self.state
    }

    /// Get device type.
    pub fn device_type(&self) -> &DeviceType {
        &self.device_type
    }

    /// Send play command (VTR devices).
    pub async fn play(&mut self) -> Result<()> {
        if let Some(ref mut device) = self.vdcp {
            device.play().await
        } else if let Some(ref mut device) = self.sony9pin {
            device.play().await
        } else {
            Err(AutomationError::DeviceControl(
                "Device does not support play".to_string(),
            ))
        }
    }

    /// Send stop command (VTR devices).
    pub async fn stop(&mut self) -> Result<()> {
        if let Some(ref mut device) = self.vdcp {
            device.stop().await
        } else if let Some(ref mut device) = self.sony9pin {
            device.stop().await
        } else {
            Err(AutomationError::DeviceControl(
                "Device does not support stop".to_string(),
            ))
        }
    }

    /// Trigger output (GPO devices).
    pub async fn trigger(&mut self, pin: u8) -> Result<()> {
        if let Some(ref mut device) = self.gpo {
            device.trigger(pin).await
        } else {
            Err(AutomationError::DeviceControl(
                "Device does not support trigger".to_string(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_type_name() {
        let device = DeviceType::Vdcp {
            port: "/dev/ttyS0".to_string(),
        };
        assert_eq!(device.name(), "vdcp");

        let device = DeviceType::Sony9Pin {
            port: "/dev/ttyS1".to_string(),
        };
        assert_eq!(device.name(), "sony9pin");
    }

    #[tokio::test]
    async fn test_device_controller_creation() {
        let device_type = DeviceType::Vdcp {
            port: "/dev/ttyS0".to_string(),
        };
        let controller = DeviceController::new(device_type).await;
        assert!(controller.is_ok());
    }
}
