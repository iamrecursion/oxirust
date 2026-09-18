//! Bitstream reading and writing for H.263.
//!
//! This module provides bit-level I/O operations for parsing and
//! generating H.263 bitstreams.

use crate::CodecError;

/// Bitstream reader for H.263 data.
///
/// Reads bits from a byte buffer, MSB first.
pub struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u8,
}

impl<'a> BitReader<'a> {
    /// Create a new bit reader.
    ///
    /// # Arguments
    ///
    /// * `data` - Input byte buffer
    #[must_use]
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    /// Read a single bit.
    ///
    /// # Errors
    ///
    /// Returns error if end of buffer reached.
    pub fn read_bit(&mut self) -> Result<bool, CodecError> {
        if self.byte_pos >= self.data.len() {
            return Err(CodecError::InvalidData(
                "Unexpected end of bitstream".into(),
            ));
        }

        let byte = self.data[self.byte_pos];
        let bit = (byte >> (7 - self.bit_pos)) & 1;

        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }

        Ok(bit != 0)
    }

    /// Read multiple bits as an unsigned integer.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of bits to read (1-32)
    ///
    /// # Errors
    ///
    /// Returns error if not enough bits available.
    pub fn read_bits(&mut self, n: u8) -> Result<u32, CodecError> {
        if n == 0 || n > 32 {
            return Err(CodecError::InvalidData(format!("Invalid bit count: {n}")));
        }

        let mut value = 0u32;
        for _ in 0..n {
            value = (value << 1) | u32::from(self.read_bit()?);
        }

        Ok(value)
    }

    /// Read a signed integer.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of bits to read (1-32)
    ///
    /// # Errors
    ///
    /// Returns error if not enough bits available.
    pub fn read_signed_bits(&mut self, n: u8) -> Result<i32, CodecError> {
        let value = self.read_bits(n)?;

        // Sign extension
        let sign_bit = 1u32 << (n - 1);
        if (value & sign_bit) != 0 {
            // Negative number
            let mask = (1u32 << n) - 1;
            Ok((value | !mask) as i32)
        } else {
            Ok(value as i32)
        }
    }

    /// Peek at the next n bits without consuming them.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of bits to peek (1-32)
    ///
    /// # Errors
    ///
    /// Returns error if not enough bits available.
    pub fn peek_bits(&self, n: u8) -> Result<u32, CodecError> {
        if n == 0 || n > 32 {
            return Err(CodecError::InvalidData(format!("Invalid bit count: {n}")));
        }

        let mut value = 0u32;
        let mut byte_pos = self.byte_pos;
        let mut bit_pos = self.bit_pos;

        for _ in 0..n {
            if byte_pos >= self.data.len() {
                return Err(CodecError::InvalidData(
                    "Unexpected end of bitstream".into(),
                ));
            }

            let byte = self.data[byte_pos];
            let bit = (byte >> (7 - bit_pos)) & 1;
            value = (value << 1) | u32::from(bit);

            bit_pos += 1;
            if bit_pos == 8 {
                bit_pos = 0;
                byte_pos += 1;
            }
        }

        Ok(value)
    }

    /// Skip n bits.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of bits to skip
    ///
    /// # Errors
    ///
    /// Returns error if not enough bits available.
    pub fn skip_bits(&mut self, n: usize) -> Result<(), CodecError> {
        for _ in 0..n {
            self.read_bit()?;
        }
        Ok(())
    }

    /// Align to next byte boundary.
    pub fn byte_align(&mut self) {
        if self.bit_pos != 0 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
    }

    /// Get current bit position.
    #[must_use]
    pub fn bit_position(&self) -> usize {
        self.byte_pos * 8 + self.bit_pos as usize
    }

    /// Get number of bits remaining.
    #[must_use]
    pub fn bits_remaining(&self) -> usize {
        (self.data.len() - self.byte_pos) * 8 - self.bit_pos as usize
    }

    /// Check if at end of buffer.
    #[must_use]
    pub fn is_eof(&self) -> bool {
        self.byte_pos >= self.data.len()
    }

    /// Search for the next start code (0x0000 01xx).
    ///
    /// # Returns
    ///
    /// Position of start code, or None if not found.
    pub fn find_start_code(&mut self) -> Option<usize> {
        self.byte_align();

        while self.byte_pos + 2 < self.data.len() {
            if self.data[self.byte_pos] == 0x00
                && self.data[self.byte_pos + 1] == 0x00
                && (self.data[self.byte_pos + 2] & 0x80) == 0x80
            {
                return Some(self.byte_pos);
            }
            self.byte_pos += 1;
        }

        None
    }

    /// Read VLC code up to max_bits.
    ///
    /// # Arguments
    ///
    /// * `max_bits` - Maximum number of bits to read
    ///
    /// # Errors
    ///
    /// Returns error if not enough bits available.
    pub fn read_vlc(&mut self, max_bits: u8) -> Result<(u32, u8), CodecError> {
        for bits in 1..=max_bits {
            let code = self.peek_bits(bits)?;
            // Caller will validate the code
            self.skip_bits(bits as usize)?;
            return Ok((code, bits));
        }

        Err(CodecError::InvalidData("VLC code too long".into()))
    }
}

/// Bitstream writer for H.263 data.
///
/// Writes bits to a byte buffer, MSB first.
pub struct BitWriter {
    data: Vec<u8>,
    bit_pos: u8,
}

impl BitWriter {
    /// Create a new bit writer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            bit_pos: 0,
        }
    }

    /// Create a new bit writer with capacity.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Initial capacity in bytes
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            bit_pos: 0,
        }
    }

    /// Write a single bit.
    ///
    /// # Arguments
    ///
    /// * `bit` - Bit value (true = 1, false = 0)
    pub fn write_bit(&mut self, bit: bool) {
        if self.bit_pos == 0 {
            self.data.push(0);
        }

        if bit {
            let last_idx = self.data.len() - 1;
            self.data[last_idx] |= 1 << (7 - self.bit_pos);
        }

        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.bit_pos = 0;
        }
    }

    /// Write multiple bits from an unsigned integer.
    ///
    /// # Arguments
    ///
    /// * `value` - Value to write
    /// * `n` - Number of bits to write (1-32)
    pub fn write_bits(&mut self, value: u32, n: u8) {
        if n == 0 || n > 32 {
            return;
        }

        for i in (0..n).rev() {
            let bit = (value >> i) & 1;
            self.write_bit(bit != 0);
        }
    }

    /// Write a signed integer.
    ///
    /// # Arguments
    ///
    /// * `value` - Signed value to write
    /// * `n` - Number of bits to write (1-32)
    pub fn write_signed_bits(&mut self, value: i32, n: u8) {
        let mask = (1u32 << n) - 1;
        let unsigned = (value as u32) & mask;
        self.write_bits(unsigned, n);
    }

    /// Align to next byte boundary with zero padding.
    pub fn byte_align(&mut self) {
        while self.bit_pos != 0 {
            self.write_bit(false);
        }
    }

    /// Get the current data.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Consume the writer and return the data.
    #[must_use]
    pub fn into_vec(mut self) -> Vec<u8> {
        if self.bit_pos != 0 {
            self.byte_align();
        }
        self.data
    }

    /// Get current bit position.
    #[must_use]
    pub fn bit_position(&self) -> usize {
        self.data.len() * 8 + self.bit_pos as usize
    }

    /// Write a VLC code.
    ///
    /// # Arguments
    ///
    /// * `code` - VLC code value
    /// * `bits` - Number of bits in the code
    pub fn write_vlc(&mut self, code: u32, bits: u8) {
        self.write_bits(code, bits);
    }

    /// Write stuffing bits (zero padding to byte boundary).
    pub fn write_stuffing(&mut self) {
        self.byte_align();
    }

    /// Write a start code.
    ///
    /// # Arguments
    ///
    /// * `code` - Start code value (0x00-0xFF)
    pub fn write_start_code(&mut self, code: u8) {
        self.byte_align();
        self.data.extend_from_slice(&[0x00, 0x00, code]);
    }
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Exponential-Golomb code reader/writer utilities.
pub struct ExpGolomb;

impl ExpGolomb {
    /// Read unsigned Exp-Golomb code.
    ///
    /// # Arguments
    ///
    /// * `reader` - Bit reader
    ///
    /// # Errors
    ///
    /// Returns error if invalid code.
    pub fn read_ue(reader: &mut BitReader<'_>) -> Result<u32, CodecError> {
        let mut leading_zeros = 0;

        while !reader.read_bit()? {
            leading_zeros += 1;
            if leading_zeros > 31 {
                return Err(CodecError::InvalidData("Invalid Exp-Golomb code".into()));
            }
        }

        if leading_zeros == 0 {
            return Ok(0);
        }

        let value = reader.read_bits(leading_zeros)?;
        Ok((1 << leading_zeros) - 1 + value)
    }

    /// Read signed Exp-Golomb code.
    ///
    /// # Arguments
    ///
    /// * `reader` - Bit reader
    ///
    /// # Errors
    ///
    /// Returns error if invalid code.
    pub fn read_se(reader: &mut BitReader<'_>) -> Result<i32, CodecError> {
        let value = Self::read_ue(reader)?;
        if value == 0 {
            return Ok(0);
        }

        let sign = if (value & 1) != 0 { 1 } else { -1 };
        Ok(sign * ((value + 1) / 2) as i32)
    }

    /// Write unsigned Exp-Golomb code.
    ///
    /// # Arguments
    ///
    /// * `writer` - Bit writer
    /// * `value` - Value to encode
    pub fn write_ue(writer: &mut BitWriter, value: u32) {
        if value == 0 {
            writer.write_bit(true);
            return;
        }

        let bits = 32 - (value + 1).leading_zeros();
        let leading_zeros = bits - 1;

        // Write leading zeros
        for _ in 0..leading_zeros {
            writer.write_bit(false);
        }

        // Write 1 bit
        writer.write_bit(true);

        // Write remaining bits
        if leading_zeros > 0 {
            let remainder = value + 1 - (1 << leading_zeros);
            writer.write_bits(remainder, leading_zeros as u8);
        }
    }

    /// Write signed Exp-Golomb code.
    ///
    /// # Arguments
    ///
    /// * `writer` - Bit writer
    /// * `value` - Signed value to encode
    pub fn write_se(writer: &mut BitWriter, value: i32) {
        if value == 0 {
            Self::write_ue(writer, 0);
            return;
        }

        let abs_value = value.unsigned_abs();
        let code = if value > 0 {
            2 * abs_value - 1
        } else {
            2 * abs_value
        };

        Self::write_ue(writer, code);
    }
}

/// Utilities for stuffing and emulation prevention.
pub struct StuffingHelper;

impl StuffingHelper {
    /// Check if byte sequence needs emulation prevention.
    ///
    /// Detects 0x000000-0x000003 patterns that could be confused with start codes.
    #[must_use]
    pub fn needs_emulation_prevention(data: &[u8], pos: usize) -> bool {
        if pos < 2 {
            return false;
        }

        data[pos - 2] == 0x00 && data[pos - 1] == 0x00 && data[pos] <= 0x03
    }

    /// Add emulation prevention bytes.
    ///
    /// Inserts 0x03 byte after 0x0000 sequences to prevent start code emulation.
    #[must_use]
    pub fn add_emulation_prevention(data: &[u8]) -> Vec<u8> {
        let mut result = Vec::with_capacity(data.len() + data.len() / 100);
        let mut zero_count = 0;

        for &byte in data {
            if zero_count == 2 && byte <= 0x03 {
                result.push(0x03); // Emulation prevention byte
                zero_count = 0;
            }

            result.push(byte);

            if byte == 0x00 {
                zero_count += 1;
            } else {
                zero_count = 0;
            }
        }

        result
    }

    /// Remove emulation prevention bytes.
    #[must_use]
    pub fn remove_emulation_prevention(data: &[u8]) -> Vec<u8> {
        let mut result = Vec::with_capacity(data.len());
        let mut i = 0;

        while i < data.len() {
            if i + 2 < data.len() && data[i] == 0x00 && data[i + 1] == 0x00 && data[i + 2] == 0x03 {
                // Found emulation prevention byte
                result.push(0x00);
                result.push(0x00);
                i += 3; // Skip the 0x03 byte
            } else {
                result.push(data[i]);
                i += 1;
            }
        }

        result
    }
}
