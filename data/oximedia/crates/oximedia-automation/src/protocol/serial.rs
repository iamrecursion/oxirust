//! Serial port communication abstraction.

use crate::Result;
use tracing::{debug, info};

/// Serial port abstraction for device communication.
pub struct SerialPort {
    port_name: String,
    baud_rate: u32,
    #[allow(dead_code)]
    mock: bool,
}

impl SerialPort {
    /// Create a new serial port.
    pub fn new(port: &str, baud_rate: u32) -> Result<Self> {
        info!("Opening serial port: {} at {} baud", port, baud_rate);

        // In a real implementation, this would open the actual serial port
        // using tokio-serial

        Ok(Self {
            port_name: port.to_string(),
            baud_rate,
            mock: false,
        })
    }

    /// Create a mock serial port for testing.
    #[cfg(test)]
    pub fn mock() -> Self {
        Self {
            port_name: "/dev/null".to_string(),
            baud_rate: 38400,
            mock: true,
        }
    }

    /// Write data to the serial port.
    pub fn write(&mut self, data: &[u8]) -> Result<()> {
        debug!(
            "Writing {} bytes to serial port {}",
            data.len(),
            self.port_name
        );

        // In a real implementation, this would write to the actual serial port

        Ok(())
    }

    /// Read data from the serial port.
    #[allow(dead_code)]
    pub fn read(&mut self, _buffer: &mut [u8]) -> Result<usize> {
        debug!("Reading from serial port {}", self.port_name);

        // In a real implementation, this would read from the actual serial port

        Ok(0)
    }

    /// Close the serial port.
    pub fn close(&mut self) -> Result<()> {
        info!("Closing serial port {}", self.port_name);

        // In a real implementation, this would close the actual serial port

        Ok(())
    }

    /// Get port name.
    #[allow(dead_code)]
    pub fn port_name(&self) -> &str {
        &self.port_name
    }

    /// Get baud rate.
    #[allow(dead_code)]
    pub fn baud_rate(&self) -> u32 {
        self.baud_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serial_port_creation() {
        let port = SerialPort::new("/dev/ttyS0", 38400);
        assert!(port.is_ok());
    }

    #[test]
    fn test_serial_port_mock() {
        let port = SerialPort::mock();
        assert_eq!(port.baud_rate, 38400);
    }

    #[test]
    fn test_serial_port_write() {
        let mut port = SerialPort::mock();
        let data = vec![0x01, 0x02, 0x03];
        assert!(port.write(&data).is_ok());
    }
}
