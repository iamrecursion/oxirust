//! CRC calculation for FLAC.
//!
//! FLAC uses:
//! - CRC-8 for frame headers
//! - CRC-16 for complete frames

#![forbid(unsafe_code)]

/// CRC-8 polynomial (0x07).
const CRC8_POLYNOMIAL: u8 = 0x07;

/// CRC-16 polynomial (0x8005).
const CRC16_POLYNOMIAL: u16 = 0x8005;

/// CRC-8 lookup table.
static CRC8_TABLE: [u8; 256] = generate_crc8_table();

/// CRC-16 lookup table.
static CRC16_TABLE: [u16; 256] = generate_crc16_table();

/// Generate CRC-8 lookup table at compile time.
const fn generate_crc8_table() -> [u8; 256] {
    let mut table = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u8;
        let mut j = 0;
        while j < 8 {
            if (crc & 0x80) != 0 {
                crc = (crc << 1) ^ CRC8_POLYNOMIAL;
            } else {
                crc <<= 1;
            }
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

/// Generate CRC-16 lookup table at compile time.
const fn generate_crc16_table() -> [u16; 256] {
    let mut table = [0u16; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = (i as u16) << 8;
        let mut j = 0;
        while j < 8 {
            if (crc & 0x8000) != 0 {
                crc = (crc << 1) ^ CRC16_POLYNOMIAL;
            } else {
                crc <<= 1;
            }
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

/// Calculate CRC-8 of data.
#[must_use]
pub fn crc8(data: &[u8]) -> u8 {
    let mut crc = 0u8;
    for &byte in data {
        crc = CRC8_TABLE[usize::from(crc ^ byte)];
    }
    crc
}

/// Calculate CRC-16 of data.
#[must_use]
pub fn crc16(data: &[u8]) -> u16 {
    let mut crc = 0u16;
    for &byte in data {
        let index = ((crc >> 8) as u8) ^ byte;
        crc = (crc << 8) ^ CRC16_TABLE[usize::from(index)];
    }
    crc
}

/// CRC-8 calculator for incremental updates.
#[derive(Debug, Clone, Default)]
pub struct Crc8 {
    crc: u8,
}

impl Crc8 {
    /// Create new CRC-8 calculator.
    #[must_use]
    pub const fn new() -> Self {
        Self { crc: 0 }
    }

    /// Update CRC with data.
    pub fn update(&mut self, data: &[u8]) {
        for &byte in data {
            self.crc = CRC8_TABLE[usize::from(self.crc ^ byte)];
        }
    }

    /// Update CRC with single byte.
    pub fn update_byte(&mut self, byte: u8) {
        self.crc = CRC8_TABLE[usize::from(self.crc ^ byte)];
    }

    /// Get current CRC value.
    #[must_use]
    pub const fn value(&self) -> u8 {
        self.crc
    }

    /// Reset CRC to initial value.
    pub fn reset(&mut self) {
        self.crc = 0;
    }
}

/// CRC-16 calculator for incremental updates.
#[derive(Debug, Clone, Default)]
pub struct Crc16 {
    crc: u16,
}

impl Crc16 {
    /// Create new CRC-16 calculator.
    #[must_use]
    pub const fn new() -> Self {
        Self { crc: 0 }
    }

    /// Update CRC with data.
    pub fn update(&mut self, data: &[u8]) {
        for &byte in data {
            let index = ((self.crc >> 8) as u8) ^ byte;
            self.crc = (self.crc << 8) ^ CRC16_TABLE[usize::from(index)];
        }
    }

    /// Update CRC with single byte.
    pub fn update_byte(&mut self, byte: u8) {
        let index = ((self.crc >> 8) as u8) ^ byte;
        self.crc = (self.crc << 8) ^ CRC16_TABLE[usize::from(index)];
    }

    /// Get current CRC value.
    #[must_use]
    pub const fn value(&self) -> u16 {
        self.crc
    }

    /// Reset CRC to initial value.
    pub fn reset(&mut self) {
        self.crc = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crc8() {
        let data = b"Hello, World!";
        let crc = crc8(data);
        assert_ne!(crc, 0);

        // Test incremental calculation
        let mut calc = Crc8::new();
        calc.update(data);
        assert_eq!(calc.value(), crc);
    }

    #[test]
    fn test_crc8_empty() {
        assert_eq!(crc8(&[]), 0);
    }

    #[test]
    fn test_crc16() {
        let data = b"Hello, World!";
        let crc = crc16(data);
        assert_ne!(crc, 0);

        // Test incremental calculation
        let mut calc = Crc16::new();
        calc.update(data);
        assert_eq!(calc.value(), crc);
    }

    #[test]
    fn test_crc16_empty() {
        assert_eq!(crc16(&[]), 0);
    }

    #[test]
    fn test_crc8_incremental() {
        let data = b"Hello, World!";
        let mut calc = Crc8::new();
        for &byte in data {
            calc.update_byte(byte);
        }
        assert_eq!(calc.value(), crc8(data));
    }

    #[test]
    fn test_crc16_incremental() {
        let data = b"Hello, World!";
        let mut calc = Crc16::new();
        for &byte in data {
            calc.update_byte(byte);
        }
        assert_eq!(calc.value(), crc16(data));
    }

    #[test]
    fn test_crc8_reset() {
        let mut calc = Crc8::new();
        calc.update(b"test");
        calc.reset();
        assert_eq!(calc.value(), 0);
    }

    #[test]
    fn test_crc16_reset() {
        let mut calc = Crc16::new();
        calc.update(b"test");
        calc.reset();
        assert_eq!(calc.value(), 0);
    }
}
