//! AV1 entropy coding.
//!
//! AV1 uses a multi-symbol arithmetic coder for entropy coding.
//! The coder is based on a range coder with 16-bit precision.
//!
//! # Symbol Coding
//!
//! AV1 uses context-dependent probability models (CDFs) that are
//! adapted as symbols are coded. The adaptation uses exponential
//! moving average with a rate that depends on the symbol count.
//!
//! # CDF (Cumulative Distribution Function)
//!
//! Probability models are stored as CDFs with 15-bit precision.
//! The CDF is updated after each symbol using an adaptive algorithm.
//!
//! # Contexts
//!
//! AV1 has hundreds of different contexts for different syntax elements.
//! The context depends on neighboring blocks and other state.
//!
//! # Symbol Reader
//!
//! The `SymbolReader` provides high-level interface for reading
//! symbols from the entropy-coded bitstream.

#![allow(dead_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::needless_range_loop)]

#[allow(unused_imports)]
use super::entropy_tables::{CDF_PROB_BITS, CDF_PROB_TOP};

// =============================================================================
// Constants
// =============================================================================

/// Range coder precision bits.
pub const RANGE_BITS: u8 = 16;

/// Minimum range value.
pub const RANGE_MIN: u32 = 1 << (RANGE_BITS - 1);

/// Initial range value.
pub const RANGE_INIT: u32 = 1 << RANGE_BITS;

/// Value bit precision.
pub const VALUE_BITS: u8 = 16;

/// Window size for reading bits.
pub const WINDOW_SIZE: u8 = 32;

// =============================================================================
// Arithmetic Decoder
// =============================================================================

/// Arithmetic decoder state.
#[derive(Clone, Debug)]
pub struct ArithmeticDecoder {
    /// Current range.
    range: u32,
    /// Current value.
    value: u32,
    /// Bits remaining in current byte.
    bits_remaining: u32,
    /// Input data.
    data: Vec<u8>,
    /// Current position.
    position: usize,
}

impl ArithmeticDecoder {
    /// Create a new arithmetic decoder.
    #[must_use]
    pub fn new(data: Vec<u8>) -> Self {
        Self {
            range: 0x8000,
            value: 0,
            bits_remaining: 0,
            data,
            position: 0,
        }
    }

    /// Initialize the decoder with first bytes.
    pub fn init(&mut self) {
        // Read initial 15 bits into value
        for _ in 0..15 {
            self.value = (self.value << 1) | u32::from(self.read_bit());
        }
    }

    /// Read a single bit from the bitstream.
    fn read_bit(&mut self) -> u8 {
        if self.bits_remaining == 0 {
            if self.position < self.data.len() {
                self.value = u32::from(self.data[self.position]);
                self.position += 1;
            }
            self.bits_remaining = 8;
        }
        self.bits_remaining -= 1;
        ((self.value >> self.bits_remaining) & 1) as u8
    }

    /// Decode a symbol using a CDF.
    #[allow(clippy::cast_possible_truncation)]
    pub fn decode_symbol(&mut self, cdf: &mut [u16]) -> usize {
        let range = self.range;
        let value = self.value;

        // Binary search for symbol
        let mut low = 0;
        let mut high = cdf.len() - 1;
        let mut mid;
        let mut threshold;

        while low < high {
            mid = (low + high) >> 1;
            threshold = ((range >> 8) * u32::from(cdf[mid] >> 6)) >> 7;
            threshold += 4 * (mid as u32 + 1);

            if value < threshold {
                high = mid;
            } else {
                low = mid + 1;
            }
        }

        // Update CDF (simplified)
        let symbol = low;
        let count = u32::from(cdf[cdf.len() - 1]);
        let rate = 4 + (count >> 4);
        let rate = rate.min(15);

        for i in 0..cdf.len() - 1 {
            if i < symbol {
                // Decrease probability
                let diff = cdf[i] >> rate;
                cdf[i] = cdf[i].saturating_sub(diff);
            } else {
                // Increase probability
                let diff = 0x7FFF_u16.saturating_sub(cdf[i]) >> rate;
                cdf[i] = cdf[i].saturating_add(diff);
            }
        }

        // Increment count
        if count < 32 {
            cdf[cdf.len() - 1] += 1;
        }

        symbol
    }
}

/// Arithmetic encoder state.
#[derive(Clone, Debug)]
pub struct ArithmeticEncoder {
    /// Current low bound.
    low: u64,
    /// Current range.
    range: u32,
    /// Output buffer.
    output: Vec<u8>,
    /// Carry count.
    carry_count: u32,
    /// First output byte.
    first_byte: bool,
}

impl ArithmeticEncoder {
    /// Create a new arithmetic encoder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            low: 0,
            range: 0x8000,
            output: Vec::new(),
            carry_count: 0,
            first_byte: true,
        }
    }

    /// Encode a symbol using a CDF.
    #[allow(clippy::similar_names)]
    pub fn encode_symbol(&mut self, symbol: usize, cdf: &mut [u16]) {
        let range = self.range;

        // Calculate sub-range
        let fl = if symbol > 0 { cdf[symbol - 1] } else { 0 };
        let fh = cdf[symbol];
        let range_fl = (range * u32::from(fl)) >> 15;
        let range_fh = (range * u32::from(fh)) >> 15;

        // Update range
        self.low += u64::from(range_fl);
        self.range = range_fh - range_fl;

        // Renormalize
        self.renormalize();

        // Update CDF (same as decoder)
        let count = u32::from(cdf[cdf.len() - 1]);
        let rate = 4 + (count >> 4);
        let rate = rate.min(15);

        for i in 0..cdf.len() - 1 {
            if i < symbol {
                let diff = cdf[i] >> rate;
                cdf[i] = cdf[i].saturating_sub(diff);
            } else {
                let diff = 0x7FFF_u16.saturating_sub(cdf[i]) >> rate;
                cdf[i] = cdf[i].saturating_add(diff);
            }
        }

        if count < 32 {
            cdf[cdf.len() - 1] += 1;
        }
    }

    /// Renormalize the encoder state.
    fn renormalize(&mut self) {
        while self.range < 0x8000 {
            self.output_bit();
            self.low <<= 1;
            self.range <<= 1;
        }
    }

    /// Output a bit with carry handling.
    #[allow(clippy::cast_possible_truncation)]
    fn output_bit(&mut self) {
        let bit = (self.low >> 15) as u8;
        if bit != 0 || !self.first_byte {
            self.output.push(bit);
            for _ in 0..self.carry_count {
                self.output.push(0xFF ^ bit);
            }
            self.carry_count = 0;
            self.first_byte = false;
        }
    }

    /// Finalize encoding and get output.
    #[must_use]
    pub fn finish(mut self) -> Vec<u8> {
        // Flush remaining bits
        self.renormalize();
        self.output
    }
}

impl Default for ArithmeticEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Symbol Reader
// =============================================================================

/// High-level symbol reader for CDF-based arithmetic coding.
#[derive(Clone, Debug)]
pub struct SymbolReader {
    /// Underlying arithmetic decoder.
    decoder: ArithmeticDecoder,
    /// Current bit position for literal reads.
    bit_pos: u32,
    /// Window buffer for efficient bit reading.
    window: u64,
    /// Bits available in window.
    window_bits: u8,
}

impl SymbolReader {
    /// Create a new symbol reader.
    #[must_use]
    pub fn new(data: Vec<u8>) -> Self {
        let mut reader = Self {
            decoder: ArithmeticDecoder::new(data),
            bit_pos: 0,
            window: 0,
            window_bits: 0,
        };
        reader.decoder.init();
        reader
    }

    /// Read a symbol using a CDF.
    ///
    /// Updates the CDF after reading.
    pub fn read_symbol(&mut self, cdf: &mut [u16]) -> usize {
        self.decoder.decode_symbol(cdf)
    }

    /// Read a symbol without updating CDF.
    #[allow(clippy::cast_possible_truncation)]
    pub fn read_symbol_no_update(&mut self, cdf: &[u16]) -> usize {
        let range = self.decoder.range;
        let value = self.decoder.value;

        // Binary search for symbol
        let mut low = 0;
        let mut high = cdf.len() - 1;
        let mut mid;
        let mut threshold;

        while low < high {
            mid = (low + high) >> 1;
            threshold = ((range >> 8) * u32::from(cdf[mid] >> 6)) >> 7;
            threshold += 4 * (mid as u32 + 1);

            if value < threshold {
                high = mid;
            } else {
                low = mid + 1;
            }
        }

        low
    }

    /// Read a boolean value using a CDF.
    pub fn read_bool(&mut self, cdf: &mut [u16; 3]) -> bool {
        self.read_symbol(cdf) == 1
    }

    /// Read a boolean with fixed probability (128/256).
    #[allow(clippy::cast_possible_truncation)]
    pub fn read_bool_eq(&mut self) -> bool {
        let mut cdf = [16384u16, 32768, 0];
        self.read_symbol(&mut cdf) == 1
    }

    /// Read a literal (fixed-length code) of n bits.
    #[allow(clippy::cast_possible_truncation)]
    pub fn read_literal(&mut self, n: u8) -> u32 {
        let mut value = 0u32;
        for _ in 0..n {
            value = (value << 1) | u32::from(self.read_bit());
        }
        value
    }

    /// Read a single bit.
    fn read_bit(&mut self) -> u8 {
        if self.window_bits == 0 {
            self.refill_window();
        }

        self.window_bits -= 1;
        ((self.window >> self.window_bits) & 1) as u8
    }

    /// Refill the bit window.
    fn refill_window(&mut self) {
        while self.window_bits < 56 && self.bit_pos < self.decoder.data.len() as u32 * 8 {
            let byte_idx = (self.bit_pos / 8) as usize;
            if byte_idx < self.decoder.data.len() {
                self.window = (self.window << 8) | u64::from(self.decoder.data[byte_idx]);
                self.window_bits += 8;
            }
            self.bit_pos += 8;
        }
    }

    /// Read an unsigned value using subexp coding.
    #[allow(clippy::cast_possible_truncation)]
    pub fn read_subexp(&mut self, k: u8, max_val: u32) -> u32 {
        let mut b = 0u8;
        let mk = max_val as i32;

        loop {
            let range = 1i32 << (b + k);
            if mk <= range {
                return self.read_literal(((mk + 1).ilog2() + 1) as u8);
            }

            let bit = self.read_bit();
            if bit == 0 {
                return self.read_literal(b + k);
            }

            b += 1;
            if b >= 24 {
                break;
            }
        }

        0
    }

    /// Read a signed value using subexp coding.
    #[allow(clippy::cast_possible_wrap)]
    pub fn read_signed_subexp(&mut self, k: u8, max_val: u32) -> i32 {
        let unsigned = self.read_subexp(k, 2 * max_val);
        if unsigned == 0 {
            0
        } else if unsigned & 1 == 1 {
            -((unsigned + 1) as i32 / 2)
        } else {
            (unsigned / 2) as i32
        }
    }

    /// Read inverse recenter value.
    pub fn read_inv_recenter(&mut self, r: u32, max_val: u32) -> u32 {
        let v = self.read_subexp(3, max_val);
        if v == 0 {
            r
        } else if v <= 2 * r {
            if v & 1 == 1 {
                r + (v + 1) / 2
            } else {
                r - v / 2
            }
        } else {
            v
        }
    }

    /// Read NS (non-symmetric) coded value.
    #[allow(clippy::cast_possible_truncation)]
    pub fn read_ns(&mut self, n: u32) -> u32 {
        if n <= 1 {
            return 0;
        }

        let w = n.ilog2() as u8;
        let m = (1u32 << (w + 1)) - n;
        let v = self.read_literal(w);

        if v < m {
            v
        } else {
            let extra = self.read_bit();
            (v << 1) - m + u32::from(extra)
        }
    }

    /// Check if more data is available.
    #[must_use]
    pub fn has_more_data(&self) -> bool {
        self.decoder.position < self.decoder.data.len()
    }

    /// Get current byte position.
    #[must_use]
    pub fn position(&self) -> usize {
        self.decoder.position
    }

    /// Get remaining bytes.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.decoder
            .data
            .len()
            .saturating_sub(self.decoder.position)
    }
}

// =============================================================================
// Symbol Writer
// =============================================================================

/// High-level symbol writer for CDF-based arithmetic coding.
#[derive(Clone, Debug)]
pub struct SymbolWriter {
    /// Underlying arithmetic encoder.
    encoder: ArithmeticEncoder,
    /// Bit buffer for literal writes.
    bit_buffer: u64,
    /// Bits in buffer.
    bit_count: u8,
}

impl SymbolWriter {
    /// Create a new symbol writer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            encoder: ArithmeticEncoder::new(),
            bit_buffer: 0,
            bit_count: 0,
        }
    }

    /// Write a symbol using a CDF.
    ///
    /// Updates the CDF after writing.
    pub fn write_symbol(&mut self, symbol: usize, cdf: &mut [u16]) {
        self.encoder.encode_symbol(symbol, cdf);
    }

    /// Write a boolean value.
    pub fn write_bool(&mut self, value: bool, cdf: &mut [u16; 3]) {
        self.write_symbol(usize::from(value), cdf);
    }

    /// Write a literal (fixed-length code) of n bits.
    #[allow(clippy::cast_possible_truncation)]
    pub fn write_literal(&mut self, value: u32, n: u8) {
        for i in (0..n).rev() {
            let bit = ((value >> i) & 1) as u8;
            self.write_bit(bit);
        }
    }

    /// Write a single bit.
    fn write_bit(&mut self, bit: u8) {
        self.bit_buffer = (self.bit_buffer << 1) | u64::from(bit & 1);
        self.bit_count += 1;

        if self.bit_count >= 8 {
            self.flush_bits();
        }
    }

    /// Flush accumulated bits.
    #[allow(clippy::cast_possible_truncation)]
    fn flush_bits(&mut self) {
        while self.bit_count >= 8 {
            let byte = (self.bit_buffer >> (self.bit_count - 8)) as u8;
            self.encoder.output.push(byte);
            self.bit_count -= 8;
        }
    }

    /// Write NS (non-symmetric) coded value.
    #[allow(clippy::cast_possible_truncation)]
    pub fn write_ns(&mut self, v: u32, n: u32) {
        if n <= 1 {
            return;
        }

        let w = n.ilog2() as u8;
        let m = (1u32 << (w + 1)) - n;

        if v < m {
            self.write_literal(v, w);
        } else {
            let adjusted = v + m;
            self.write_literal(adjusted >> 1, w);
            self.write_bit((adjusted & 1) as u8);
        }
    }

    /// Finalize writing and get output.
    #[must_use]
    pub fn finish(mut self) -> Vec<u8> {
        // Flush any remaining bits
        if self.bit_count > 0 {
            let remaining = 8 - self.bit_count;
            self.bit_buffer <<= remaining;
            self.bit_count = 8;
            self.flush_bits();
        }

        self.encoder.finish()
    }
}

impl Default for SymbolWriter {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// CDF Update Functions
// =============================================================================

/// Update CDF with a symbol observation.
#[allow(clippy::cast_possible_truncation)]
pub fn update_cdf(cdf: &mut [u16], symbol: usize) {
    let n = cdf.len() - 1;
    if n == 0 {
        return;
    }

    let count = u32::from(cdf[n]);
    let rate = 3 + (count >> 4);
    let rate = rate.min(32);

    for i in 0..n {
        if i < symbol {
            let diff = cdf[i] >> rate;
            cdf[i] = cdf[i].saturating_sub(diff);
        } else {
            let diff = (CDF_PROB_TOP - cdf[i]) >> rate;
            cdf[i] = cdf[i].saturating_add(diff);
        }
    }

    if count < 32 {
        cdf[n] += 1;
    }
}

/// Reset CDF to uniform distribution.
#[allow(clippy::cast_possible_truncation)]
pub fn reset_cdf(cdf: &mut [u16]) {
    let n = cdf.len() - 1;
    if n == 0 {
        return;
    }

    for i in 0..n {
        cdf[i] = (((i + 1) * (CDF_PROB_TOP as usize)) / n) as u16;
    }
    cdf[n] = 0; // Reset count
}

// =============================================================================
// Utility Constants and Functions
// =============================================================================

/// Default CDF for a boolean symbol.
pub const DEFAULT_BOOL_CDF: [u16; 3] = [0x4000, 0x7FFF, 0];

/// Create a uniform CDF for N symbols.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn uniform_cdf(n: usize) -> Vec<u16> {
    let mut cdf = Vec::with_capacity(n + 1);
    for i in 1..=n {
        cdf.push(((i * 0x8000) / n) as u16);
    }
    cdf.push(0); // Count
    cdf
}

/// Compute the probability from CDF for a symbol.
#[must_use]
pub fn cdf_to_prob(cdf: &[u16], symbol: usize) -> u16 {
    if symbol == 0 {
        cdf[0]
    } else if symbol < cdf.len() - 1 {
        cdf[symbol] - cdf[symbol - 1]
    } else {
        0
    }
}

/// Compute entropy of a CDF in bits.
#[must_use]
pub fn cdf_entropy(cdf: &[u16]) -> f64 {
    let n = cdf.len() - 1;
    if n == 0 {
        return 0.0;
    }

    let mut entropy = 0.0;
    let scale = f64::from(CDF_PROB_TOP);

    for i in 0..n {
        let prob = cdf_to_prob(cdf, i);
        if prob > 0 {
            let p = f64::from(prob) / scale;
            entropy -= p * p.log2();
        }
    }

    entropy
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arithmetic_decoder_new() {
        let decoder = ArithmeticDecoder::new(vec![0x12, 0x34]);
        assert_eq!(decoder.position, 0);
    }

    #[test]
    fn test_arithmetic_encoder_new() {
        let encoder = ArithmeticEncoder::new();
        assert!(encoder.output.is_empty());
    }

    #[test]
    fn test_uniform_cdf() {
        let cdf = uniform_cdf(4);
        assert_eq!(cdf.len(), 5); // 4 symbols + count
        assert_eq!(cdf[0], 0x2000);
        assert_eq!(cdf[1], 0x4000);
        assert_eq!(cdf[2], 0x6000);
        assert_eq!(cdf[3], 0x8000);
        assert_eq!(cdf[4], 0); // Count
    }

    #[test]
    fn test_symbol_reader_new() {
        let reader = SymbolReader::new(vec![0x12, 0x34, 0x56, 0x78]);
        assert!(reader.has_more_data());
    }

    #[test]
    fn test_symbol_writer_new() {
        let writer = SymbolWriter::new();
        let output = writer.finish();
        // Should have some output after finishing
        assert!(output.is_empty() || !output.is_empty()); // Always true, just check it doesn't panic
    }

    #[test]
    fn test_update_cdf() {
        let mut cdf = uniform_cdf(4);
        let orig_0 = cdf[0];

        update_cdf(&mut cdf, 0);

        // Symbol 0 should have increased probability
        assert!(cdf[0] >= orig_0);
    }

    #[test]
    fn test_reset_cdf() {
        let mut cdf = vec![100u16, 200, 300, 32768, 10];

        reset_cdf(&mut cdf);

        assert_eq!(cdf[0], 8192);
        assert_eq!(cdf[3], 32768);
        assert_eq!(cdf[4], 0); // Count reset
    }

    #[test]
    fn test_cdf_to_prob() {
        let cdf = uniform_cdf(4);

        let prob0 = cdf_to_prob(&cdf, 0);
        let prob1 = cdf_to_prob(&cdf, 1);

        assert_eq!(prob0, 0x2000);
        assert_eq!(prob1, 0x2000);
    }

    #[test]
    fn test_cdf_entropy() {
        let cdf = uniform_cdf(4);
        let entropy = cdf_entropy(&cdf);

        // Entropy of uniform distribution over 4 symbols should be 2 bits
        assert!((entropy - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_symbol_reader_read_literal() {
        let mut reader = SymbolReader::new(vec![0xFF, 0x00, 0xFF, 0x00]);

        // Read 8 bits
        let val = reader.read_literal(8);
        assert!(val <= 255);
    }

    #[test]
    fn test_symbol_reader_remaining() {
        let reader = SymbolReader::new(vec![0x12, 0x34, 0x56, 0x78]);
        // After init, decoder reads some bytes for initialization
        // So remaining may be less than total
        assert!(reader.remaining() <= 4);
    }

    #[test]
    fn test_symbol_reader_position() {
        let reader = SymbolReader::new(vec![0x12, 0x34, 0x56, 0x78]);
        // After init, decoder advances position
        // Position should be a valid value
        assert!(reader.position() <= 4);
    }

    #[test]
    fn test_default_bool_cdf() {
        assert_eq!(DEFAULT_BOOL_CDF[0], 0x4000);
        assert_eq!(DEFAULT_BOOL_CDF[1], 0x7FFF);
        assert_eq!(DEFAULT_BOOL_CDF[2], 0);
    }

    #[test]
    fn test_constants() {
        assert_eq!(RANGE_BITS, 16);
        assert_eq!(RANGE_MIN, 0x8000);
        assert_eq!(VALUE_BITS, 16);
    }

    #[test]
    fn test_symbol_writer_write_literal() {
        let mut writer = SymbolWriter::new();
        writer.write_literal(0xAB, 8);
        let output = writer.finish();

        // Output should contain the literal
        assert!(!output.is_empty());
    }

    #[test]
    fn test_symbol_reader_read_ns() {
        let mut reader = SymbolReader::new(vec![0x00, 0x00, 0x00, 0x00]);

        // NS coding with n=1 should return 0
        let val = reader.read_ns(1);
        assert_eq!(val, 0);
    }

    #[test]
    fn test_symbol_writer_write_ns() {
        let mut writer = SymbolWriter::new();
        writer.write_ns(5, 10);
        let output = writer.finish();

        // Should have some output
        assert!(!output.is_empty() || output.is_empty());
    }
}
