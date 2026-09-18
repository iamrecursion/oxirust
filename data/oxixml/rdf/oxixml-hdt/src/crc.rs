//! The three CRC variants used throughout the HDT binary format.
//!
//! The [HDT binary format specification](http://www.rdfhdt.org/hdt-binary-format/) mandates
//! exactly three checksums, each guarding a different granularity of data:
//!
//! * **CRC8-CCITT** (polynomial `x^8 + x^2 + x + 1`, hex `0x07`) — guards the small
//!   [`ControlInformation`](crate::control_info::ControlInformation)-adjacent metadata
//!   preambles (e.g. the `count`/`length` fields in front of a dictionary section or bitmap).
//!   Non-reflected, no initial or final XOR. This is the standard **CRC-8/SMBUS** variant.
//! * **CRC16-ANSI** (polynomial `x^16 + x^15 + x^2 + 1`, hex `0x8005`) — guards every
//!   [`ControlInformation`](crate::control_info::ControlInformation) block. Reflected input and
//!   output, no initial or final XOR. This is the standard **CRC-16/ARC** variant.
//! * **CRC32C** (Castagnoli polynomial, hex `0x1EDC6F41`) — guards the large payload buffers
//!   (dictionary string blocks, bitmap bit arrays, ID arrays). Reflected input and output,
//!   `0xFFFFFFFF` initial value, `0xFFFFFFFF` final XOR. This is the standard **CRC-32C**
//!   (a.k.a. CRC-32/ISCSI) variant, matching the reference `hdt-cpp` implementation exactly.
//!
//! All three are implemented here as simple table-free bit-at-a-time updates (clear and
//! auditable rather than maximally fast — HDT files are read/written in bulk, not
//! byte-at-a-time hot loops) and are validated against the well-known "check value" vectors
//! for the ASCII string `"123456789"` published by the
//! [CRC Catalogue](https://reveng.sourceforge.io/crc-catalogue/all.htm).

/// Streaming CRC-8/SMBUS (a.k.a. CRC-8-CCITT with polynomial `0x07`).
///
/// Non-reflected, initial value `0`, no final XOR.
#[derive(Debug, Clone, Copy, Default)]
pub struct Crc8 {
    state: u8,
}

impl Crc8 {
    /// Creates a new CRC-8 accumulator in its initial state.
    #[inline]
    pub const fn new() -> Self {
        Self { state: 0 }
    }

    /// Folds `data` into the running checksum.
    pub fn update(&mut self, data: &[u8]) {
        let mut crc = self.state;
        for &byte in data {
            crc ^= byte;
            for _ in 0..8 {
                crc = if crc & 0x80 != 0 {
                    (crc << 1) ^ 0x07
                } else {
                    crc << 1
                };
            }
        }
        self.state = crc;
    }

    /// Returns the checksum accumulated so far.
    #[inline]
    pub const fn finalize(self) -> u8 {
        self.state
    }
}

/// Computes the CRC-8/SMBUS checksum of `data` in one call.
#[inline]
pub fn crc8(data: &[u8]) -> u8 {
    let mut crc = Crc8::new();
    crc.update(data);
    crc.finalize()
}

/// Streaming CRC-16/ARC (a.k.a. CRC-16-ANSI with polynomial `0x8005`).
///
/// Reflected input and output, initial value `0`, no final XOR.
#[derive(Debug, Clone, Copy, Default)]
pub struct Crc16 {
    state: u16,
}

impl Crc16 {
    /// Creates a new CRC-16 accumulator in its initial state.
    #[inline]
    pub const fn new() -> Self {
        Self { state: 0 }
    }

    /// Folds `data` into the running checksum.
    pub fn update(&mut self, data: &[u8]) {
        let mut crc = self.state;
        for &byte in data {
            crc ^= u16::from(byte);
            for _ in 0..8 {
                crc = if crc & 1 != 0 {
                    (crc >> 1) ^ 0xA001
                } else {
                    crc >> 1
                };
            }
        }
        self.state = crc;
    }

    /// Returns the checksum accumulated so far.
    #[inline]
    pub const fn finalize(self) -> u16 {
        self.state
    }
}

/// Computes the CRC-16/ARC checksum of `data` in one call.
#[inline]
pub fn crc16(data: &[u8]) -> u16 {
    let mut crc = Crc16::new();
    crc.update(data);
    crc.finalize()
}

/// Streaming CRC-32C (Castagnoli).
///
/// Reflected input and output, initial value `0xFFFFFFFF`, final XOR `0xFFFFFFFF`.
#[derive(Debug, Clone, Copy)]
pub struct Crc32c {
    state: u32,
}

impl Default for Crc32c {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Crc32c {
    /// Creates a new CRC-32C accumulator in its initial state.
    #[inline]
    pub const fn new() -> Self {
        Self { state: 0xFFFF_FFFF }
    }

    /// Folds `data` into the running checksum.
    pub fn update(&mut self, data: &[u8]) {
        let mut crc = self.state;
        for &byte in data {
            crc ^= u32::from(byte);
            for _ in 0..8 {
                crc = if crc & 1 != 0 {
                    (crc >> 1) ^ 0x82F6_3B78
                } else {
                    crc >> 1
                };
            }
        }
        self.state = crc;
    }

    /// Returns the checksum accumulated so far, with the final XOR applied.
    #[inline]
    pub const fn finalize(self) -> u32 {
        self.state ^ 0xFFFF_FFFF
    }
}

/// Computes the CRC-32C checksum of `data` in one call.
#[inline]
pub fn crc32c(data: &[u8]) -> u32 {
    let mut crc = Crc32c::new();
    crc.update(data);
    crc.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    // The de-facto standard "check value" vectors: the CRC of the ASCII string
    // "123456789", as published by the CRC RevEng catalogue for each named variant.

    #[test]
    fn crc8_smbus_check_value() {
        assert_eq!(crc8(b"123456789"), 0xF4);
    }

    #[test]
    fn crc16_arc_check_value() {
        assert_eq!(crc16(b"123456789"), 0xBB3D);
    }

    #[test]
    fn crc32c_check_value() {
        assert_eq!(crc32c(b"123456789"), 0xE306_9283);
    }

    #[test]
    fn crc8_empty_is_zero() {
        assert_eq!(crc8(b""), 0);
    }

    #[test]
    fn crc16_empty_is_zero() {
        assert_eq!(crc16(b""), 0);
    }

    #[test]
    fn crc32c_empty_is_zero() {
        assert_eq!(crc32c(b""), 0);
    }

    #[test]
    fn streaming_matches_one_shot() {
        let data = b"the quick brown fox jumps over the lazy dog";
        let mut crc = Crc32c::new();
        crc.update(&data[..10]);
        crc.update(&data[10..]);
        assert_eq!(crc.finalize(), crc32c(data));
    }
}
