//! GPI (General Purpose Input) device implementation.

use crate::Result;
use tokio::sync::mpsc;
use tracing::info;

/// GPI event type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpiEvent {
    /// Pin triggered high
    High(u8),
    /// Pin triggered low
    Low(u8),
}

/// GPI device controller.
pub struct GpiDevice {
    port: String,
    event_tx: Option<mpsc::UnboundedSender<GpiEvent>>,
}

impl GpiDevice {
    /// Create a new GPI device.
    pub async fn new(port: &str) -> Result<Self> {
        info!("Creating GPI device on port: {}", port);

        Ok(Self {
            port: port.to_string(),
            event_tx: None,
        })
    }

    /// Connect to the device.
    pub async fn connect(&mut self) -> Result<()> {
        info!("Connecting to GPI device on {}", self.port);

        let (tx, _rx) = mpsc::unbounded_channel();
        self.event_tx = Some(tx);

        // In a real implementation, this would open the hardware port
        // and start monitoring for input events

        Ok(())
    }

    /// Disconnect from the device.
    pub async fn disconnect(&mut self) -> Result<()> {
        info!("Disconnecting from GPI device on {}", self.port);

        self.event_tx = None;

        Ok(())
    }

    /// Subscribe to GPI events.
    pub fn subscribe(&self) -> mpsc::UnboundedReceiver<GpiEvent> {
        let (_tx, rx) = mpsc::unbounded_channel();
        // In a real implementation, this would register the subscriber
        rx
    }

    /// Read current pin state.
    pub async fn read_pin(&self, _pin: u8) -> Result<bool> {
        // In a real implementation, this would read the actual hardware state
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_gpi_device_creation() {
        let device = GpiDevice::new("/dev/gpio0").await;
        assert!(device.is_ok());
    }

    #[tokio::test]
    async fn test_gpi_event_equality() {
        let event1 = GpiEvent::High(1);
        let event2 = GpiEvent::High(1);
        let event3 = GpiEvent::Low(1);

        assert_eq!(event1, event2);
        assert_ne!(event1, event3);
    }
}
