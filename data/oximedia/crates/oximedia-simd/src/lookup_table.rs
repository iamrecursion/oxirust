//! Lookup table (LUT) primitives for fast pixel transformations.
//!
//! Provides fixed-size LUTs for 8-bit and 16-bit components, interpolated
//! entries, and a convenience gamma LUT builder.  All lookups are O(1).

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// A single LUT entry that supports linear interpolation between adjacent values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LutEntry {
    /// The output value stored at this LUT index.
    pub value: f32,
}

impl LutEntry {
    /// Create a new `LutEntry`.
    #[must_use]
    pub const fn new(value: f32) -> Self {
        Self { value }
    }

    /// Interpolate linearly between `self` and `other` by fraction `t` (0.0 – 1.0).
    #[must_use]
    pub fn interpolated_value(&self, other: &Self, t: f32) -> f32 {
        self.value + (other.value - self.value) * t.clamp(0.0, 1.0)
    }
}

/// A 256-entry LUT for 8-bit component transforms.
///
/// Backed by a plain array for cache-friendly random access.
#[derive(Debug, Clone)]
pub struct Lut256 {
    table: [f32; 256],
}

impl Lut256 {
    /// Create a new `Lut256` with all entries set to zero.
    #[must_use]
    pub fn new() -> Self {
        Self { table: [0.0; 256] }
    }

    /// Create a `Lut256` from a precomputed array.
    #[must_use]
    pub fn from_array(table: [f32; 256]) -> Self {
        Self { table }
    }

    /// Create a `Lut256` where entry `i` equals `i / 255.0` (identity / linear).
    #[must_use]
    pub fn identity() -> Self {
        let mut table = [0.0_f32; 256];
        for (i, v) in table.iter_mut().enumerate() {
            *v = i as f32 / 255.0;
        }
        Self { table }
    }

    /// Look up the output value for an 8-bit input.
    #[must_use]
    #[inline]
    pub fn lookup(&self, index: u8) -> f32 {
        self.table[usize::from(index)]
    }

    /// Set the value at a given index.
    pub fn set(&mut self, index: u8, value: f32) {
        self.table[usize::from(index)] = value;
    }

    /// Apply this LUT to every sample in `slice`, writing results to `dst`.
    ///
    /// Input samples are clamped to `[0.0, 1.0]` and quantised to 8-bit before lookup.
    ///
    /// # Panics
    ///
    /// Panics if `slice.len() != dst.len()`.
    pub fn apply_to_slice(&self, slice: &[f32], dst: &mut [f32]) {
        assert_eq!(
            slice.len(),
            dst.len(),
            "slice and dst must have the same length"
        );
        for (&s, d) in slice.iter().zip(dst.iter_mut()) {
            let idx = (s.clamp(0.0, 1.0) * 255.0).round() as u8;
            *d = self.lookup(idx);
        }
    }

    /// Apply this LUT in-place to a slice of `u8` values, returning a new `Vec<f32>`.
    #[must_use]
    pub fn apply_to_u8(&self, input: &[u8]) -> Vec<f32> {
        input.iter().map(|&b| self.lookup(b)).collect()
    }
}

impl Default for Lut256 {
    fn default() -> Self {
        Self::new()
    }
}

/// A 65 536-entry LUT for 16-bit component transforms.
#[derive(Debug, Clone)]
pub struct Lut65536 {
    table: Vec<f32>,
}

impl Lut65536 {
    /// Create a new `Lut65536` with all entries set to zero.
    #[must_use]
    pub fn new() -> Self {
        Self {
            table: vec![0.0; 65536],
        }
    }

    /// Create a `Lut65536` where entry `i` equals `i / 65535.0` (identity / linear).
    #[must_use]
    pub fn identity() -> Self {
        let table: Vec<f32> = (0..=65535_u32).map(|i| i as f32 / 65535.0).collect();
        Self { table }
    }

    /// Look up the output value for a 16-bit input.
    #[must_use]
    #[inline]
    pub fn lookup_16bit(&self, index: u16) -> f32 {
        self.table[usize::from(index)]
    }

    /// Set the value at a given 16-bit index.
    pub fn set(&mut self, index: u16, value: f32) {
        self.table[usize::from(index)] = value;
    }

    /// Apply this LUT to a slice of `u16` values.
    #[must_use]
    pub fn apply_to_u16(&self, input: &[u16]) -> Vec<f32> {
        input.iter().map(|&v| self.lookup_16bit(v)).collect()
    }
}

impl Default for Lut65536 {
    fn default() -> Self {
        Self::new()
    }
}

/// Precomputed gamma correction LUT (8-bit input → linear float output).
///
/// Converts from gamma-encoded `u8` values to linear light `f32` values.
#[derive(Debug, Clone)]
pub struct GammaLut {
    lut: Lut256,
    gamma: f32,
}

impl GammaLut {
    /// Build a `GammaLut` for the given gamma exponent.
    ///
    /// Each entry `i` stores `(i / 255) ^ gamma`.
    #[must_use]
    pub fn from_gamma(gamma: f32) -> Self {
        let mut table = [0.0_f32; 256];
        for (i, v) in table.iter_mut().enumerate() {
            let normalized = i as f32 / 255.0;
            *v = normalized.powf(gamma);
        }
        Self {
            lut: Lut256::from_array(table),
            gamma,
        }
    }

    /// Build a sRGB-to-linear LUT using the IEC 61966-2-1 piecewise formula.
    #[must_use]
    pub fn srgb_to_linear() -> Self {
        let mut table = [0.0_f32; 256];
        for (i, v) in table.iter_mut().enumerate() {
            let s = i as f32 / 255.0;
            *v = if s <= 0.040_45 {
                s / 12.92
            } else {
                ((s + 0.055) / 1.055).powf(2.4)
            };
        }
        Self {
            lut: Lut256::from_array(table),
            gamma: 2.2, // approximate
        }
    }

    /// Return the gamma exponent this LUT was built with.
    #[must_use]
    pub fn gamma(&self) -> f32 {
        self.gamma
    }

    /// Apply gamma correction to a single `u8` value.
    #[must_use]
    #[inline]
    pub fn apply(&self, value: u8) -> f32 {
        self.lut.lookup(value)
    }

    /// Apply gamma correction to a slice of `u8` values.
    #[must_use]
    pub fn apply_slice(&self, input: &[u8]) -> Vec<f32> {
        self.lut.apply_to_u8(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lut_entry_interpolate_at_zero() {
        let a = LutEntry::new(0.0);
        let b = LutEntry::new(1.0);
        assert!((a.interpolated_value(&b, 0.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_lut_entry_interpolate_at_one() {
        let a = LutEntry::new(0.0);
        let b = LutEntry::new(1.0);
        assert!((a.interpolated_value(&b, 1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_lut_entry_interpolate_midpoint() {
        let a = LutEntry::new(0.0);
        let b = LutEntry::new(2.0);
        assert!((a.interpolated_value(&b, 0.5) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_lut_entry_interpolate_clamped() {
        let a = LutEntry::new(0.0);
        let b = LutEntry::new(1.0);
        // t > 1.0 should be clamped to 1.0
        assert!((a.interpolated_value(&b, 2.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_lut256_identity() {
        let lut = Lut256::identity();
        assert!((lut.lookup(0) - 0.0).abs() < 1e-6);
        assert!((lut.lookup(255) - 1.0).abs() < 1e-6);
        assert!((lut.lookup(128) - 128.0 / 255.0).abs() < 1e-5);
    }

    #[test]
    fn test_lut256_set_and_lookup() {
        let mut lut = Lut256::new();
        lut.set(42, 0.75);
        assert!((lut.lookup(42) - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_lut256_apply_to_slice() {
        let lut = Lut256::identity();
        let input = vec![0.0_f32, 0.5, 1.0];
        let mut output = vec![0.0_f32; 3];
        lut.apply_to_slice(&input, &mut output);
        // 0.0 → index 0 → 0/255, 1.0 → index 255 → 1.0
        assert!((output[0] - 0.0).abs() < 1e-5);
        assert!((output[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_lut256_apply_to_u8() {
        let lut = Lut256::identity();
        let input = vec![0u8, 255];
        let out = lut.apply_to_u8(&input);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_lut65536_identity_endpoints() {
        let lut = Lut65536::identity();
        assert!((lut.lookup_16bit(0) - 0.0).abs() < 1e-7);
        assert!((lut.lookup_16bit(65535) - 1.0).abs() < 1e-7);
    }

    #[test]
    fn test_lut65536_set_and_lookup() {
        let mut lut = Lut65536::new();
        lut.set(1000, 0.5);
        assert!((lut.lookup_16bit(1000) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_lut65536_apply_to_u16() {
        let lut = Lut65536::identity();
        let input = vec![0u16, 65535];
        let out = lut.apply_to_u16(&input);
        assert!((out[0] - 0.0).abs() < 1e-7);
        assert!((out[1] - 1.0).abs() < 1e-7);
    }

    #[test]
    fn test_gamma_lut_from_gamma_1() {
        // gamma=1.0 → identity
        let lut = GammaLut::from_gamma(1.0);
        assert!((lut.apply(0) - 0.0).abs() < 1e-6);
        assert!((lut.apply(255) - 1.0).abs() < 1e-6);
        assert!((lut.apply(128) - 128.0 / 255.0).abs() < 1e-5);
    }

    #[test]
    fn test_gamma_lut_gamma_value() {
        let lut = GammaLut::from_gamma(2.2);
        assert!((lut.gamma() - 2.2).abs() < 1e-6);
    }

    #[test]
    fn test_gamma_lut_srgb_black_white() {
        let lut = GammaLut::srgb_to_linear();
        assert!((lut.apply(0) - 0.0).abs() < 1e-6);
        assert!((lut.apply(255) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_gamma_lut_apply_slice() {
        let lut = GammaLut::from_gamma(2.0);
        let out = lut.apply_slice(&[0, 255]);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[1] - 1.0).abs() < 1e-5);
    }
}
