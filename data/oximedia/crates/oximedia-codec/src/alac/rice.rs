//! Adaptive modified-Rice / Golomb entropy coding for ALAC (`ag_dec`/`ag_enc`).
//!
//! ALAC entropy-codes prediction residuals with a Rice/Golomb code whose
//! parameter `k` is *adaptive*: it is derived each symbol from a running mean
//! `mb` that is updated by an exponential smoothing controlled by the
//! `pb`/`mb`/`kb` tuning values from the magic cookie. Outliers escape to a
//! fixed `bit_depth`-bit representation, and long runs of zeros are coded with
//! a dedicated zero-run token.
//!
//! The encoder ([`encode_residuals`]) and decoder ([`decode_residuals`]) share
//! the exact same parameter-derivation ([`AgState::k_for`]) and mean-update
//! ([`AgState::update`]) so they stay in lockstep symbol-by-symbol, which is
//! what makes the round-trip byte-exact.
//!
//! # Parameter derivation
//!
//! Following Apple's `aglib`, with `QBSHIFT = 9` (so the mean is a Q9
//! fixed-point quantity):
//!
//! ```text
//! m = mb >> QBSHIFT
//! k = floor(log2(m + 3))      // Apple's lg3a
//! k = min(k, kb)
//! ```
//!
//! and the mean update after coding the unsigned symbol `n`:
//!
//! ```text
//! mb = pb * n + mb - ((pb * mb) >> QBSHIFT)
//! ```

use super::bitstream::{BitReader, BitWriter};
use super::{AlacError, AlacResult};

/// Fixed-point shift for the running mean (Apple `QBSHIFT`).
pub const QBSHIFT: u32 = 9;
/// `QB = 1 << QBSHIFT`.
pub const QB: u32 = 1 << QBSHIFT;
/// Maximum unary prefix length before the escape path is taken.
pub const MAX_PREFIX: u32 = 16;
/// Initial-history scaling: the cookie's `mb` is multiplied into a Q9 mean.
pub const MEAN_INIT_SCALE: u32 = QB;
/// Mean threshold (Q9) below which a zero-run token is emitted/expected.
pub const ZERO_RUN_THRESHOLD: u32 = QB; // mean < 1.0 ⇒ try a zero run

/// Map a signed residual to an unsigned value (ALAC zigzag).
///
/// `0 → 0, -1 → 1, 1 → 2, -2 → 3, 2 → 4, …`
#[inline]
#[must_use]
pub fn zigzag(v: i32) -> u32 {
    ((v << 1) ^ (v >> 31)) as u32
}

/// Inverse of [`zigzag`].
#[inline]
#[must_use]
pub fn unzigzag(u: u32) -> i32 {
    ((u >> 1) as i32) ^ -((u & 1) as i32)
}

/// Integer floor-log2 of `x` (returns 0 for `x == 0`).
#[inline]
fn floor_log2(x: u32) -> u32 {
    if x == 0 {
        0
    } else {
        x.ilog2()
    }
}

/// Apple's `lg3a(x) = floor(log2(x + 3))`.
#[inline]
fn lg3a(x: u32) -> u32 {
    floor_log2(x + 3)
}

/// Adaptive-Golomb running state shared by encoder and decoder.
#[derive(Clone, Copy, Debug)]
pub struct AgState {
    /// Running mean in Q9 fixed-point.
    mb: u32,
    /// History multiplier (`pb`).
    pb: u32,
    /// `k` modifier ceiling (`kb`).
    kb: u32,
    /// Sample width for the escape path.
    bit_depth: u32,
}

impl AgState {
    /// Build a fresh state from the cookie tuning values.
    #[must_use]
    pub fn new(pb: u8, mb: u8, kb: u8, bit_depth: u32) -> Self {
        Self {
            mb: u32::from(mb) * MEAN_INIT_SCALE,
            pb: u32::from(pb),
            kb: u32::from(kb),
            bit_depth,
        }
    }

    /// Current Rice parameter `k` for the next symbol.
    #[inline]
    #[must_use]
    pub fn k_for(&self) -> u32 {
        let m = self.mb >> QBSHIFT;
        let k = lg3a(m);
        k.min(self.kb).min(self.bit_depth)
    }

    /// Update the running mean after coding unsigned symbol `n`.
    #[inline]
    pub fn update(&mut self, n: u32) {
        // mb = pb*n + mb - ((pb*mb) >> QBSHIFT), in u64 to avoid overflow.
        let pb = u64::from(self.pb);
        let mb = u64::from(self.mb);
        let n = u64::from(n);
        let next = pb * n + mb - ((pb * mb) >> QBSHIFT);
        // Clamp to the u32 range Apple keeps the mean within.
        self.mb = next.min(u64::from(u32::MAX)) as u32;
    }

    /// Whether a zero-run token should be coded next (mean is very low).
    #[inline]
    #[must_use]
    pub fn wants_zero_run(&self) -> bool {
        self.mb < ZERO_RUN_THRESHOLD
    }
}

/// Encode one unsigned symbol `n` with parameter `k`, escaping to `bit_depth`
/// raw bits when the unary prefix would be too long.
fn encode_symbol(writer: &mut BitWriter, n: u32, k: u32, bit_depth: u32) {
    let quotient = n >> k;
    if quotient >= MAX_PREFIX {
        // Escape: MAX_PREFIX ones, then the full value in `bit_depth` bits.
        for _ in 0..MAX_PREFIX {
            writer.write_bit(true);
        }
        writer.write_bits(n, bit_depth);
    } else {
        // Unary quotient + terminating zero, then k remainder bits.
        writer.write_unary(quotient);
        if k > 0 {
            let remainder = n & ((1u32 << k) - 1);
            writer.write_bits(remainder, k);
        }
    }
}

/// Decode one unsigned symbol with parameter `k`.
fn decode_symbol(reader: &mut BitReader, k: u32, bit_depth: u32) -> AlacResult<u32> {
    // Count up to MAX_PREFIX leading ones.
    let mut quotient = 0u32;
    while quotient < MAX_PREFIX {
        if reader.read_bit()? {
            quotient += 1;
        } else {
            // Consumed the terminating zero.
            let remainder = if k > 0 { reader.read_bits(k)? } else { 0 };
            return Ok((quotient << k) | remainder);
        }
    }
    // Hit MAX_PREFIX ones ⇒ escape, read raw value.
    reader.read_bits(bit_depth)
}

/// Encode `residuals` (signed) into `writer` using adaptive Golomb coding.
///
/// When the running mean is low ([`AgState::wants_zero_run`]) the encoder
/// prepends a one-bit flag: `1` introduces a run of one or more zeros
/// (length-1 coded as a single bit so a single zero costs two bits total),
/// `0` introduces a regular non-zero symbol.
pub fn encode_residuals(writer: &mut BitWriter, residuals: &[i32], state: &mut AgState) {
    let mut idx = 0usize;
    let count = residuals.len();
    while idx < count {
        if state.wants_zero_run() {
            if residuals[idx] == 0 {
                // Count the run of consecutive zeros (at least 1).
                let start = idx;
                while idx < count && residuals[idx] == 0 {
                    idx += 1;
                }
                let run = (idx - start) as u32;
                writer.write_bit(true); // "zero run follows"
                encode_zero_run_length(writer, run - 1);
                for _ in 0..run {
                    state.update(0);
                }
                continue;
            }
            // Non-zero symbol despite low mean: emit the discriminator then
            // fall through to the regular symbol path.
            writer.write_bit(false);
        }
        let n = zigzag(residuals[idx]);
        let k = state.k_for();
        encode_symbol(writer, n, k, state.bit_depth);
        state.update(n);
        idx += 1;
    }
}

/// Decode `count` residuals from `reader` using adaptive Golomb coding.
pub fn decode_residuals(
    reader: &mut BitReader,
    count: usize,
    state: &mut AgState,
) -> AlacResult<Vec<i32>> {
    let mut out = Vec::with_capacity(count);
    while out.len() < count {
        if state.wants_zero_run() {
            let is_run = reader.read_bit()?;
            if is_run {
                let run = decode_zero_run_length(reader)? + 1;
                let remaining = count - out.len();
                if run as usize > remaining {
                    return Err(AlacError::InvalidBitstream(
                        "zero run overflows residual count".into(),
                    ));
                }
                for _ in 0..run {
                    out.push(0);
                    state.update(0);
                }
                continue;
            }
            // Otherwise fall through and read a regular symbol.
        }
        let k = state.k_for();
        let n = decode_symbol(reader, k, state.bit_depth)?;
        out.push(unzigzag(n));
        state.update(n);
    }
    Ok(out)
}

/// Encode `value` (run-1) using an Elias-gamma-style code: a unary prefix on
/// the bit-width followed by the low bits of `value + 1`.
fn encode_zero_run_length(writer: &mut BitWriter, value: u32) {
    let v = value + 1;
    let width = floor_log2(v);
    writer.write_unary(width);
    if width > 0 {
        let low = v & ((1u32 << width) - 1);
        writer.write_bits(low, width);
    }
}

fn decode_zero_run_length(reader: &mut BitReader) -> AlacResult<u32> {
    let width = reader.read_unary()?;
    if width == 0 {
        return Ok(0);
    }
    if width > 31 {
        return Err(AlacError::InvalidBitstream(
            "zero-run width exceeds 31".into(),
        ));
    }
    let low = reader.read_bits(width)?;
    let v = (1u32 << width) | low;
    Ok(v - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zigzag_roundtrip() {
        for v in [-1000i32, -1, 0, 1, 1000, i32::MIN / 2, i32::MAX / 2] {
            assert_eq!(unzigzag(zigzag(v)), v);
        }
    }

    #[test]
    fn test_symbol_roundtrip_each_k() {
        for k in 0..=16u32 {
            let mut w = BitWriter::new();
            let values = [0u32, 1, 2, 5, 100, 1000, 65535];
            for &n in &values {
                encode_symbol(&mut w, n, k, 24);
            }
            let bytes = w.finish();
            let mut r = BitReader::new(&bytes);
            for &n in &values {
                assert_eq!(decode_symbol(&mut r, k, 24).unwrap(), n, "k={k} n={n}");
            }
        }
    }

    #[test]
    fn test_symbol_escape_path() {
        // A large value with small k forces the escape path.
        let mut w = BitWriter::new();
        encode_symbol(&mut w, 1_000_000, 1, 24);
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        assert_eq!(decode_symbol(&mut r, 1, 24).unwrap(), 1_000_000);
    }

    #[test]
    fn test_zero_run_length_roundtrip() {
        for v in [0u32, 1, 2, 7, 31, 255, 4096] {
            let mut w = BitWriter::new();
            encode_zero_run_length(&mut w, v);
            let bytes = w.finish();
            let mut r = BitReader::new(&bytes);
            assert_eq!(decode_zero_run_length(&mut r).unwrap(), v, "v={v}");
        }
    }

    fn residual_roundtrip(residuals: &[i32], bit_depth: u32) {
        let mut enc_state = AgState::new(40, 10, 14, bit_depth);
        let mut w = BitWriter::new();
        encode_residuals(&mut w, residuals, &mut enc_state);
        let bytes = w.finish();

        let mut dec_state = AgState::new(40, 10, 14, bit_depth);
        let mut r = BitReader::new(&bytes);
        let decoded = decode_residuals(&mut r, residuals.len(), &mut dec_state).expect("decode");
        assert_eq!(decoded, residuals);
        assert_eq!(enc_state.mb, dec_state.mb, "mean diverged");
    }

    #[test]
    fn test_residuals_small() {
        let residuals = vec![0i32, 1, -1, 2, -2, 0, 0, 3, -3, 0];
        residual_roundtrip(&residuals, 16);
    }

    #[test]
    fn test_residuals_with_long_zero_run() {
        let mut residuals = vec![0i32; 500];
        residuals[250] = 7;
        residual_roundtrip(&residuals, 16);
    }

    #[test]
    fn test_residuals_high_entropy() {
        let mut state = 0x1234_5678u32;
        let residuals: Vec<i32> = (0..400)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state as i32) >> 8
            })
            .collect();
        residual_roundtrip(&residuals, 24);
    }

    #[test]
    fn test_lg3a() {
        assert_eq!(lg3a(0), 1); // log2(3) = 1.58 → 1
        assert_eq!(lg3a(1), 2); // log2(4) = 2
        assert_eq!(lg3a(5), 3); // log2(8) = 3
    }
}
