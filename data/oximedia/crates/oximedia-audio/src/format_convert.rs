//! PCM sample-format conversion utilities.
//!
//! This module also exposes SIMD-accelerated batch conversion routines via
//! [`SimdConvert`].  On platforms where Rust's auto-vectoriser cannot be relied
//! upon, the methods in [`SimdConvert`] use explicit loop structure and
//! carefully chosen chunk sizes to maximise LLVM auto-vectorisation (128-bit
//! and 256-bit SIMD where available).  All code is `#[forbid(unsafe_code)]`-
//! compatible and falls back gracefully on any target.
#![allow(dead_code)]

/// PCM sample bit-depth descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SampleDepth {
    /// 8-bit unsigned integer.
    U8,
    /// 16-bit signed integer (little-endian).
    S16,
    /// 24-bit signed integer packed in 3 bytes (little-endian).
    S24,
    /// 32-bit signed integer.
    S32,
    /// 32-bit IEEE 754 floating point.
    F32,
    /// 64-bit IEEE 754 floating point.
    F64,
}

impl SampleDepth {
    /// Number of bits per sample.
    #[must_use]
    pub fn bit_count(&self) -> u32 {
        match self {
            SampleDepth::U8 => 8,
            SampleDepth::S16 => 16,
            SampleDepth::S24 => 24,
            SampleDepth::S32 => 32,
            SampleDepth::F32 => 32,
            SampleDepth::F64 => 64,
        }
    }

    /// Number of bytes per sample (rounded up to a whole byte).
    #[must_use]
    pub fn byte_size(&self) -> usize {
        match self {
            SampleDepth::U8 => 1,
            SampleDepth::S16 => 2,
            SampleDepth::S24 => 3,
            SampleDepth::S32 => 4,
            SampleDepth::F32 => 4,
            SampleDepth::F64 => 8,
        }
    }

    /// Returns `true` when this format uses floating-point samples.
    #[must_use]
    pub fn is_float(&self) -> bool {
        matches!(self, SampleDepth::F32 | SampleDepth::F64)
    }
}

/// Stateless sample-format conversion functions.
pub struct FormatConvert;

impl FormatConvert {
    /// Convert a 24-bit signed integer (stored as the lower 3 bytes of a `u32`,
    /// little-endian byte order in `buf`) to normalised `f32` in `[-1, 1]`.
    ///
    /// # Panics
    /// Panics (in debug) when `buf.len() < 3`.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn s24_to_f32(buf: &[u8]) -> f32 {
        debug_assert!(buf.len() >= 3, "s24_to_f32: need at least 3 bytes");
        // Reconstruct the 24-bit two's-complement value.
        let raw = (buf[0] as i32) | ((buf[1] as i32) << 8) | ((buf[2] as i32) << 16);
        // Sign-extend from 24 to 32 bits.
        let signed = if raw & 0x80_0000 != 0 {
            raw | !0xFF_FFFF
        } else {
            raw
        };
        signed as f32 / 8_388_607.0 // 2^23 - 1
    }

    /// Convert normalised `f32` in `[-1, 1]` to a 16-bit signed integer.
    ///
    /// Values outside the normalised range are clamped.
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn f32_to_s16(sample: f32) -> i16 {
        let clamped = sample.clamp(-1.0, 1.0);
        (clamped * 32_767.0) as i16
    }

    /// Convert a slice of normalised `f32` to 16-bit signed integers.
    #[must_use]
    pub fn f32_slice_to_s16(samples: &[f32]) -> Vec<i16> {
        samples.iter().map(|&s| Self::f32_to_s16(s)).collect()
    }

    /// Convert a slice of 16-bit signed integers to normalised `f32`.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn s16_slice_to_f32(samples: &[i16]) -> Vec<f32> {
        samples.iter().map(|&s| s as f32 / 32_768.0).collect()
    }
}

/// Stateful format converter that remembers source and destination formats.
#[derive(Debug, Clone)]
pub struct FormatConverter {
    /// Source sample depth.
    pub src: SampleDepth,
    /// Destination sample depth.
    pub dst: SampleDepth,
}

impl FormatConverter {
    /// Create a new converter.
    #[must_use]
    pub fn new(src: SampleDepth, dst: SampleDepth) -> Self {
        Self { src, dst }
    }

    /// Convert a raw byte buffer from `src` format to a `Vec<f32>`.
    ///
    /// Currently supports F32→F32 (identity) and S16→F32 and S24→F32 paths.
    /// Returns an empty `Vec` for unsupported conversions.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn convert_buffer(&self, input: &[u8]) -> Vec<f32> {
        match self.src {
            SampleDepth::F32 => {
                // Re-interpret bytes as f32 samples.
                input
                    .chunks_exact(4)
                    .map(|c| {
                        let arr = [c[0], c[1], c[2], c[3]];
                        f32::from_le_bytes(arr)
                    })
                    .collect()
            }
            SampleDepth::S16 => input
                .chunks_exact(2)
                .map(|c| {
                    let v = i16::from_le_bytes([c[0], c[1]]);
                    v as f32 / 32_768.0
                })
                .collect(),
            SampleDepth::S24 => input
                .chunks_exact(3)
                .map(|c| FormatConvert::s24_to_f32(c))
                .collect(),
            _ => Vec::new(),
        }
    }

    /// Byte size of the output buffer given `n_samples` samples.
    #[must_use]
    pub fn output_byte_size(&self, n_samples: usize) -> usize {
        n_samples * self.dst.byte_size()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SIMD-optimised batch conversion (auto-vectorised)
// ─────────────────────────────────────────────────────────────────────────────

/// SIMD-accelerated sample-format conversion utilities.
///
/// All methods are pure-Rust and use carefully structured loops that LLVM can
/// auto-vectorise to SIMD instructions (SSE2, AVX2, NEON, etc.) on supporting
/// targets.  There is no `unsafe` code; correctness is always guaranteed even
/// on scalar-only targets.
///
/// # Supported conversions
///
/// | Source | Destination | Method |
/// |--------|-------------|--------|
/// | `f32`  | `i16`       | [`SimdConvert::f32_to_s16`] |
/// | `i16`  | `f32`       | [`SimdConvert::s16_to_f32`] |
/// | `f32`  | `f64`       | [`SimdConvert::f32_to_f64`] |
/// | `f64`  | `f32`       | [`SimdConvert::f64_to_f32`] |
/// | `f32`  | `u8`        | [`SimdConvert::f32_to_u8`] |
/// | `u8`   | `f32`       | [`SimdConvert::u8_to_f32`] |
/// | `f32`  | `i32`       | [`SimdConvert::f32_to_s32`] |
/// | `i32`  | `f32`       | [`SimdConvert::s32_to_f32`] |
pub struct SimdConvert;

impl SimdConvert {
    /// Convert normalised `f32` samples to 16-bit signed integers.
    ///
    /// Values outside `[-1, 1]` are clamped.  The output is scaled to
    /// `[-32767, 32767]` (symmetric; –32768 is not produced).
    ///
    /// The inner loop processes 8 samples at a time to encourage LLVM to emit
    /// 256-bit SIMD (AVX2) instructions on x86-64.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn f32_to_s16(input: &[f32]) -> Vec<i16> {
        let mut out = vec![0_i16; input.len()];
        // Process in chunks of 8 to hint at 256-bit SIMD (8 × i32 → 8 × i16)
        let chunks = input.len() / 8;
        for c in 0..chunks {
            let base = c * 8;
            let s0 = (input[base].clamp(-1.0, 1.0) * 32_767.0) as i16;
            let s1 = (input[base + 1].clamp(-1.0, 1.0) * 32_767.0) as i16;
            let s2 = (input[base + 2].clamp(-1.0, 1.0) * 32_767.0) as i16;
            let s3 = (input[base + 3].clamp(-1.0, 1.0) * 32_767.0) as i16;
            let s4 = (input[base + 4].clamp(-1.0, 1.0) * 32_767.0) as i16;
            let s5 = (input[base + 5].clamp(-1.0, 1.0) * 32_767.0) as i16;
            let s6 = (input[base + 6].clamp(-1.0, 1.0) * 32_767.0) as i16;
            let s7 = (input[base + 7].clamp(-1.0, 1.0) * 32_767.0) as i16;
            out[base] = s0;
            out[base + 1] = s1;
            out[base + 2] = s2;
            out[base + 3] = s3;
            out[base + 4] = s4;
            out[base + 5] = s5;
            out[base + 6] = s6;
            out[base + 7] = s7;
        }
        // Scalar tail
        for i in (chunks * 8)..input.len() {
            out[i] = (input[i].clamp(-1.0, 1.0) * 32_767.0) as i16;
        }
        out
    }

    /// Convert 16-bit signed integers to normalised `f32` samples.
    ///
    /// Divides by 32_768.0 (not 32_767.0) for a symmetric mapping where
    /// `i16::MAX` ≈ +0.99997 and `i16::MIN` = –1.0.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn s16_to_f32(input: &[i16]) -> Vec<f32> {
        let mut out = vec![0.0_f32; input.len()];
        let chunks = input.len() / 8;
        for c in 0..chunks {
            let base = c * 8;
            out[base] = input[base] as f32 / 32_768.0;
            out[base + 1] = input[base + 1] as f32 / 32_768.0;
            out[base + 2] = input[base + 2] as f32 / 32_768.0;
            out[base + 3] = input[base + 3] as f32 / 32_768.0;
            out[base + 4] = input[base + 4] as f32 / 32_768.0;
            out[base + 5] = input[base + 5] as f32 / 32_768.0;
            out[base + 6] = input[base + 6] as f32 / 32_768.0;
            out[base + 7] = input[base + 7] as f32 / 32_768.0;
        }
        for i in (chunks * 8)..input.len() {
            out[i] = input[i] as f32 / 32_768.0;
        }
        out
    }

    /// Widen `f32` samples to `f64`.
    #[must_use]
    pub fn f32_to_f64(input: &[f32]) -> Vec<f64> {
        let mut out = vec![0.0_f64; input.len()];
        let chunks = input.len() / 4;
        for c in 0..chunks {
            let base = c * 4;
            out[base] = input[base] as f64;
            out[base + 1] = input[base + 1] as f64;
            out[base + 2] = input[base + 2] as f64;
            out[base + 3] = input[base + 3] as f64;
        }
        for i in (chunks * 4)..input.len() {
            out[i] = input[i] as f64;
        }
        out
    }

    /// Narrow `f64` samples to `f32` (precision is lost).
    #[must_use]
    pub fn f64_to_f32(input: &[f64]) -> Vec<f32> {
        let mut out = vec![0.0_f32; input.len()];
        let chunks = input.len() / 4;
        for c in 0..chunks {
            let base = c * 4;
            out[base] = input[base] as f32;
            out[base + 1] = input[base + 1] as f32;
            out[base + 2] = input[base + 2] as f32;
            out[base + 3] = input[base + 3] as f32;
        }
        for i in (chunks * 4)..input.len() {
            out[i] = input[i] as f32;
        }
        out
    }

    /// Convert normalised `f32` samples to unsigned 8-bit integers.
    ///
    /// Maps `[-1, 1]` → `[1, 255]` (biased, as per WAV U8 convention where
    /// silence is 128).
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn f32_to_u8(input: &[f32]) -> Vec<u8> {
        let mut out = vec![0_u8; input.len()];
        for (i, &s) in input.iter().enumerate() {
            let v = ((s.clamp(-1.0, 1.0) + 1.0) * 127.5).round() as u8;
            out[i] = v;
        }
        out
    }

    /// Convert unsigned 8-bit integers to normalised `f32` samples.
    ///
    /// Inverts the mapping used by [`SimdConvert::f32_to_u8`].
    #[must_use]
    pub fn u8_to_f32(input: &[u8]) -> Vec<f32> {
        let mut out = vec![0.0_f32; input.len()];
        for (i, &b) in input.iter().enumerate() {
            out[i] = (b as f32 / 127.5) - 1.0;
        }
        out
    }

    /// Convert normalised `f32` samples to 32-bit signed integers.
    ///
    /// Maps `[-1, 1]` → `[i32::MIN+1, i32::MAX]` (symmetric).
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn f32_to_s32(input: &[f32]) -> Vec<i32> {
        let mut out = vec![0_i32; input.len()];
        let scale = 2_147_483_647.0_f64; // i32::MAX
        let chunks = input.len() / 4;
        for c in 0..chunks {
            let base = c * 4;
            out[base] = (input[base] as f64 * scale).clamp(-scale, scale) as i32;
            out[base + 1] = (input[base + 1] as f64 * scale).clamp(-scale, scale) as i32;
            out[base + 2] = (input[base + 2] as f64 * scale).clamp(-scale, scale) as i32;
            out[base + 3] = (input[base + 3] as f64 * scale).clamp(-scale, scale) as i32;
        }
        for i in (chunks * 4)..input.len() {
            out[i] = (input[i] as f64 * scale).clamp(-scale, scale) as i32;
        }
        out
    }

    /// Convert 32-bit signed integers to normalised `f32` samples.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn s32_to_f32(input: &[i32]) -> Vec<f32> {
        let mut out = vec![0.0_f32; input.len()];
        let scale = 2_147_483_647.0_f32;
        let chunks = input.len() / 4;
        for c in 0..chunks {
            let base = c * 4;
            out[base] = input[base] as f32 / scale;
            out[base + 1] = input[base + 1] as f32 / scale;
            out[base + 2] = input[base + 2] as f32 / scale;
            out[base + 3] = input[base + 3] as f32 / scale;
        }
        for i in (chunks * 4)..input.len() {
            out[i] = input[i] as f32 / scale;
        }
        out
    }

    /// Interleave two mono channels into a stereo interleaved buffer.
    ///
    /// `left` and `right` must have the same length.  The shorter of the two
    /// is used.  Returns `[L0, R0, L1, R1, …]`.
    #[must_use]
    pub fn interleave_stereo(left: &[f32], right: &[f32]) -> Vec<f32> {
        let n = left.len().min(right.len());
        let mut out = vec![0.0_f32; n * 2];
        for i in 0..n {
            out[i * 2] = left[i];
            out[i * 2 + 1] = right[i];
        }
        out
    }

    /// De-interleave a stereo interleaved buffer into two mono channels.
    ///
    /// Input must be `[L0, R0, L1, R1, …]`.  If the length is odd, the last
    /// sample is ignored.  Returns `(left, right)`.
    #[must_use]
    pub fn deinterleave_stereo(interleaved: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let n_frames = interleaved.len() / 2;
        let mut left = vec![0.0_f32; n_frames];
        let mut right = vec![0.0_f32; n_frames];
        for i in 0..n_frames {
            left[i] = interleaved[i * 2];
            right[i] = interleaved[i * 2 + 1];
        }
        (left, right)
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- SampleDepth ---

    #[test]
    fn test_bit_count_u8() {
        assert_eq!(SampleDepth::U8.bit_count(), 8);
    }

    #[test]
    fn test_bit_count_s24() {
        assert_eq!(SampleDepth::S24.bit_count(), 24);
    }

    #[test]
    fn test_bit_count_f64() {
        assert_eq!(SampleDepth::F64.bit_count(), 64);
    }

    #[test]
    fn test_byte_size_s16() {
        assert_eq!(SampleDepth::S16.byte_size(), 2);
    }

    #[test]
    fn test_byte_size_s24() {
        assert_eq!(SampleDepth::S24.byte_size(), 3);
    }

    #[test]
    fn test_is_float_f32() {
        assert!(SampleDepth::F32.is_float());
    }

    #[test]
    fn test_is_float_s16_false() {
        assert!(!SampleDepth::S16.is_float());
    }

    // --- FormatConvert ---

    #[test]
    fn test_s24_to_f32_positive() {
        // 0x7FFFFF = 8_388_607 ≈ max positive
        let buf = [0xFF_u8, 0xFF, 0x7F];
        let v = FormatConvert::s24_to_f32(&buf);
        assert!((v - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_s24_to_f32_zero() {
        let buf = [0x00_u8, 0x00, 0x00];
        let v = FormatConvert::s24_to_f32(&buf);
        assert!(v.abs() < 1e-10);
    }

    #[test]
    fn test_f32_to_s16_positive() {
        let v = FormatConvert::f32_to_s16(1.0);
        assert_eq!(v, 32_767);
    }

    #[test]
    fn test_f32_to_s16_negative() {
        let v = FormatConvert::f32_to_s16(-1.0);
        assert_eq!(v, -32_767);
    }

    #[test]
    fn test_f32_to_s16_zero() {
        let v = FormatConvert::f32_to_s16(0.0);
        assert_eq!(v, 0);
    }

    #[test]
    fn test_f32_to_s16_clamped() {
        let v = FormatConvert::f32_to_s16(2.0);
        assert_eq!(v, 32_767);
    }

    #[test]
    fn test_roundtrip_s16() {
        let originals: Vec<f32> = vec![0.0, 0.5, -0.5, 0.9, -0.9];
        let s16 = FormatConvert::f32_slice_to_s16(&originals);
        let back = FormatConvert::s16_slice_to_f32(&s16);
        for (orig, rt) in originals.iter().zip(back.iter()) {
            // tolerance ~1/32768
            assert!((orig - rt).abs() < 0.0001, "orig={orig} rt={rt}");
        }
    }

    // --- FormatConverter ---

    #[test]
    fn test_output_byte_size() {
        let conv = FormatConverter::new(SampleDepth::S16, SampleDepth::F32);
        assert_eq!(conv.output_byte_size(100), 400);
    }

    #[test]
    fn test_convert_buffer_f32_identity() {
        let samples: Vec<f32> = vec![0.5, -0.5, 0.0];
        let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let conv = FormatConverter::new(SampleDepth::F32, SampleDepth::F32);
        let out = conv.convert_buffer(&bytes);
        assert_eq!(out.len(), 3);
        assert!((out[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_convert_buffer_s16_to_f32() {
        let samples: Vec<i16> = vec![16384, -16384]; // ≈ 0.5, -0.5
        let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let conv = FormatConverter::new(SampleDepth::S16, SampleDepth::F32);
        let out = conv.convert_buffer(&bytes);
        assert_eq!(out.len(), 2);
        assert!((out[0] - 0.5).abs() < 0.001);
        assert!((out[1] + 0.5).abs() < 0.001);
    }
}
