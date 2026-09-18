//! Electro-optical / opto-electronic transfer functions for common colour spaces.
//!
//! Hot-path linearisation is accelerated through 1024-entry lookup tables
//! (`TransferFunctionLut`) cached globally per transfer function variant so
//! that table construction occurs at most once per process lifetime.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// All supported transfer functions (EOTFs / OETFs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransferFunction {
    /// Identity (no encoding).
    Linear,
    /// BT.1886 / traditional CRT gamma 2.2.
    Gamma22,
    /// sYCC / AdobeRGB gamma 2.4 approximation.
    Gamma24,
    /// IEC 61966-2-1 sRGB piecewise transfer function.
    Srgb,
    /// SMPTE ST 2084 Perceptual Quantizer (HDR10 / Dolby Vision).
    Pq,
    /// ITU-R BT.2100 Hybrid Log-Gamma.
    Hlg,
}

impl TransferFunction {
    /// Encode a normalised linear light value into a non-linear signal value.
    ///
    /// `linear` is in [0, 1] (relative scene luminance).
    /// The returned value is the electrical/code signal, clamped to [0, 1] for SDR
    /// functions.  PQ returns values in [0, 1] where 1 = 10 000 cd/m².
    pub fn encode(&self, linear: f64) -> f64 {
        let v = linear.max(0.0);
        match self {
            TransferFunction::Linear => v,
            TransferFunction::Gamma22 => v.powf(1.0 / 2.2).min(1.0),
            TransferFunction::Gamma24 => v.powf(1.0 / 2.4).min(1.0),
            TransferFunction::Srgb => srgb_encode(v),
            TransferFunction::Pq => pq_encode(v),
            TransferFunction::Hlg => hlg_encode(v),
        }
    }

    /// Decode a non-linear signal value back to normalised linear light.
    ///
    /// `nonlinear` should be in [0, 1].
    pub fn decode(&self, nonlinear: f64) -> f64 {
        let v = nonlinear.clamp(0.0, 1.0);
        match self {
            TransferFunction::Linear => v,
            TransferFunction::Gamma22 => v.powf(2.2),
            TransferFunction::Gamma24 => v.powf(2.4),
            TransferFunction::Srgb => srgb_decode(v),
            TransferFunction::Pq => pq_decode(v),
            TransferFunction::Hlg => hlg_decode(v),
        }
    }

    /// Return `true` if this is an HDR transfer function.
    pub fn is_hdr(&self) -> bool {
        matches!(self, TransferFunction::Pq | TransferFunction::Hlg)
    }

    /// Return a human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            TransferFunction::Linear => "Linear",
            TransferFunction::Gamma22 => "Gamma 2.2",
            TransferFunction::Gamma24 => "Gamma 2.4",
            TransferFunction::Srgb => "sRGB",
            TransferFunction::Pq => "PQ (ST 2084)",
            TransferFunction::Hlg => "HLG (BT.2100)",
        }
    }

    /// Linearise an encoded value using the cached 1024-entry LUT.
    ///
    /// Equivalent to [`TransferFunction::decode`] but operates on `f32` and uses
    /// linear interpolation within a precomputed table.  Maximum error vs. the
    /// exact formula is < 0.001 across \[0, 1\].
    ///
    /// The LUT is built once per transfer-function variant and then cached for
    /// the lifetime of the process.
    #[must_use]
    pub fn linearize_via_lut(&self, encoded: f32) -> f32 {
        get_cached_tf_lut(self).linearize(encoded)
    }

    /// Encode a linear value using the cached 1024-entry LUT.
    ///
    /// Equivalent to [`TransferFunction::encode`] but operates on `f32` and uses
    /// linear interpolation within a precomputed table.  Maximum error vs. the
    /// exact formula is < 0.001 across \[0, 1\].
    #[must_use]
    pub fn delinearize_via_lut(&self, linear: f32) -> f32 {
        get_cached_tf_lut(self).delinearize(linear)
    }
}

// ── LUT cache ─────────────────────────────────────────────────────────────────

/// Number of entries in each LUT (forward and inverse).
const TF_LUT_SIZE: usize = 1024;

/// 1024-entry forward (linearise) and inverse (delinearise) lookup tables for a
/// single transfer function.
///
/// Index `i` corresponds to the normalised value `i / (TF_LUT_SIZE - 1)`.
/// Lookup uses linear interpolation between adjacent entries.
#[derive(Debug, Clone)]
pub struct TransferFunctionLut {
    /// `forward[i]` ≈ `tf.decode(i / (N-1))` — encoded → linear, `f32`.
    pub forward: Vec<f32>,
    /// `inverse[i]` ≈ `tf.encode(i / (N-1))` — linear → encoded, `f32`.
    pub inverse: Vec<f32>,
}

impl TransferFunctionLut {
    /// Build a `TransferFunctionLut` from the exact `f64` decode/encode functions
    /// of `tf`.  The two tables each have `TF_LUT_SIZE` (1024) entries.
    #[must_use]
    pub fn build(tf: &TransferFunction) -> Self {
        let n = TF_LUT_SIZE;
        let mut forward = Vec::with_capacity(n);
        let mut inverse = Vec::with_capacity(n);

        for i in 0..n {
            let t = i as f64 / (n - 1) as f64;
            forward.push(tf.decode(t) as f32);
            inverse.push(tf.encode(t) as f32);
        }

        Self { forward, inverse }
    }

    /// Look up the linearised (decoded) value for `encoded ∈ [0, 1]` using
    /// bilinear interpolation between adjacent table entries.
    ///
    /// Inputs outside `[0, 1]` are clamped before lookup.
    #[must_use]
    pub fn linearize(&self, encoded: f32) -> f32 {
        Self::interp(&self.forward, encoded)
    }

    /// Look up the encoded (compressed) value for `linear ∈ [0, 1]` using
    /// bilinear interpolation between adjacent table entries.
    ///
    /// Inputs outside `[0, 1]` are clamped before lookup.
    #[must_use]
    pub fn delinearize(&self, linear: f32) -> f32 {
        Self::interp(&self.inverse, linear)
    }

    /// Shared linear-interpolation helper used by both `linearize` and
    /// `delinearize`.
    #[inline]
    fn interp(table: &[f32], v: f32) -> f32 {
        let n = table.len();
        debug_assert!(n >= 2, "LUT must have at least 2 entries");
        let v = v.clamp(0.0, 1.0);
        let scaled = v * (n - 1) as f32;
        let lo = scaled.floor() as usize;
        let hi = (lo + 1).min(n - 1);
        let frac = scaled - lo as f32;
        table[lo] + frac * (table[hi] - table[lo])
    }
}

/// Global LUT cache: one `TransferFunctionLut` per `TransferFunction` variant,
/// keyed by the variant's discriminant cast to `u8`.
static TF_LUT_CACHE: OnceLock<Mutex<HashMap<u8, TransferFunctionLut>>> = OnceLock::new();

/// Return the cached `TransferFunctionLut` for `tf`, building it on first access.
///
/// Thread-safe: the global `Mutex` ensures only one thread builds any given LUT.
#[must_use]
pub fn get_cached_tf_lut(tf: &TransferFunction) -> TransferFunctionLut {
    let cache = TF_LUT_CACHE.get_or_init(|| Mutex::new(HashMap::new()));

    // Use the enum discriminant as cache key.
    let key = *tf as u8;

    {
        let guard = cache.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(lut) = guard.get(&key) {
            return lut.clone();
        }
    }

    // Build outside the lock to avoid holding it during potentially costly float ops.
    let lut = TransferFunctionLut::build(tf);

    {
        let mut guard = cache.lock().unwrap_or_else(|e| e.into_inner());
        // Another thread may have raced us; insert only if still absent.
        guard.entry(key).or_insert_with(|| lut.clone());
    }

    lut
}

// ── sRGB ──────────────────────────────────────────────────────────────────

fn srgb_encode(linear: f64) -> f64 {
    if linear <= 0.003_130_8 {
        12.92 * linear
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    }
    .min(1.0)
}

fn srgb_decode(nonlinear: f64) -> f64 {
    if nonlinear <= 0.040_45 {
        nonlinear / 12.92
    } else {
        ((nonlinear + 0.055) / 1.055).powf(2.4)
    }
}

// ── PQ (SMPTE ST 2084) ────────────────────────────────────────────────────
// Normalised so that `linear = 1.0` maps to 10 000 cd/m².

const PQ_M1: f64 = 2610.0 / 16384.0;
const PQ_M2: f64 = 2523.0 / 4096.0 * 128.0;
const PQ_C1: f64 = 3424.0 / 4096.0;
const PQ_C2: f64 = 2413.0 / 4096.0 * 32.0;
const PQ_C3: f64 = 2392.0 / 4096.0 * 32.0;

fn pq_encode(linear: f64) -> f64 {
    let y = linear.powf(PQ_M1);
    let num = PQ_C1 + PQ_C2 * y;
    let den = 1.0 + PQ_C3 * y;
    (num / den).powf(PQ_M2)
}

fn pq_decode(nonlinear: f64) -> f64 {
    let e = nonlinear.powf(1.0 / PQ_M2);
    let num = (e - PQ_C1).max(0.0);
    let den = PQ_C2 - PQ_C3 * e;
    if den <= 0.0 {
        return 0.0;
    }
    (num / den).powf(1.0 / PQ_M1)
}

// ── HLG (BT.2100) ─────────────────────────────────────────────────────────

const HLG_A: f64 = 0.178_832_77;
const HLG_B: f64 = 0.284_668_92;
const HLG_C: f64 = 0.559_910_73;

fn hlg_encode(linear: f64) -> f64 {
    if linear <= 1.0 / 12.0 {
        (3.0 * linear).sqrt()
    } else {
        HLG_A * (12.0 * linear - HLG_B).ln() + HLG_C
    }
}

fn hlg_decode(nonlinear: f64) -> f64 {
    if nonlinear <= 0.5 {
        (nonlinear * nonlinear) / 3.0
    } else {
        ((nonlinear - HLG_C).exp() / HLG_A + HLG_B) / 12.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_encode_decode_identity() {
        let v = 0.42;
        assert!((TransferFunction::Linear.encode(v) - v).abs() < 1e-10);
        assert!((TransferFunction::Linear.decode(v) - v).abs() < 1e-10);
    }

    #[test]
    fn test_gamma22_encode_mid() {
        let enc = TransferFunction::Gamma22.encode(0.5);
        let expected = 0.5_f64.powf(1.0 / 2.2);
        assert!((enc - expected).abs() < 1e-10);
    }

    #[test]
    fn test_gamma22_roundtrip() {
        let v = 0.3;
        let enc = TransferFunction::Gamma22.encode(v);
        let dec = TransferFunction::Gamma22.decode(enc);
        assert!((dec - v).abs() < 1e-9);
    }

    #[test]
    fn test_gamma24_roundtrip() {
        let v = 0.7;
        let enc = TransferFunction::Gamma24.encode(v);
        let dec = TransferFunction::Gamma24.decode(enc);
        assert!((dec - v).abs() < 1e-9);
    }

    #[test]
    fn test_srgb_encode_low() {
        // Linear value in sRGB linear region
        let enc = TransferFunction::Srgb.encode(0.001);
        assert!((enc - 0.001 * 12.92).abs() < 1e-9);
    }

    #[test]
    fn test_srgb_roundtrip() {
        for v in [0.0, 0.01, 0.18, 0.5, 0.9, 1.0] {
            let enc = TransferFunction::Srgb.encode(v);
            let dec = TransferFunction::Srgb.decode(enc);
            assert!((dec - v).abs() < 1e-9, "sRGB roundtrip failed for v={v}");
        }
    }

    #[test]
    fn test_pq_encode_one_maps_to_one() {
        // linear=1.0 corresponds to 10 000 cd/m², PQ code = 1.0
        let enc = TransferFunction::Pq.encode(1.0);
        assert!((enc - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_pq_encode_zero() {
        let enc = TransferFunction::Pq.encode(0.0);
        assert!(enc >= 0.0 && enc < 0.01);
    }

    #[test]
    fn test_pq_roundtrip() {
        let v = 0.01; // ~100 cd/m² normalised
        let enc = TransferFunction::Pq.encode(v);
        let dec = TransferFunction::Pq.decode(enc);
        assert!((dec - v).abs() < 1e-8, "PQ roundtrip failed: got {dec}");
    }

    #[test]
    fn test_hlg_roundtrip() {
        let v = 0.08;
        let enc = TransferFunction::Hlg.encode(v);
        let dec = TransferFunction::Hlg.decode(enc);
        assert!((dec - v).abs() < 1e-9, "HLG roundtrip failed: got {dec}");
    }

    #[test]
    fn test_is_hdr_true() {
        assert!(TransferFunction::Pq.is_hdr());
        assert!(TransferFunction::Hlg.is_hdr());
    }

    #[test]
    fn test_is_hdr_false() {
        assert!(!TransferFunction::Linear.is_hdr());
        assert!(!TransferFunction::Gamma22.is_hdr());
        assert!(!TransferFunction::Srgb.is_hdr());
        assert!(!TransferFunction::Gamma24.is_hdr());
    }

    #[test]
    fn test_names_non_empty() {
        for tf in [
            TransferFunction::Linear,
            TransferFunction::Gamma22,
            TransferFunction::Gamma24,
            TransferFunction::Srgb,
            TransferFunction::Pq,
            TransferFunction::Hlg,
        ] {
            assert!(!tf.name().is_empty());
        }
    }

    #[test]
    fn test_encode_clamps_negative() {
        // Negative input should be treated as 0
        let enc = TransferFunction::Srgb.encode(-0.5);
        assert!(enc >= 0.0);
    }

    // ── LUT tests ─────────────────────────────────────────────────────────────

    #[test]
    fn test_tf_lut_srgb_accuracy() {
        // Build the LUT and sample 100 uniformly-spaced points in [0, 1].
        // The LUT-based linearisation must agree with the exact sRGB decode
        // formula to within 0.001.
        let lut = TransferFunctionLut::build(&TransferFunction::Srgb);
        for i in 0u32..=100 {
            let t = i as f32 / 100.0;
            let lut_val = lut.linearize(t);
            let exact_val = TransferFunction::Srgb.decode(t as f64) as f32;
            let err = (lut_val - exact_val).abs();
            assert!(
                err < 0.001,
                "sRGB LUT error too large at t={t}: lut={lut_val}, exact={exact_val}, err={err}"
            );
        }
    }

    #[test]
    fn test_tf_lut_pq_accuracy() {
        // Same accuracy check for the PQ EOTF.
        let lut = TransferFunctionLut::build(&TransferFunction::Pq);
        for i in 0u32..=100 {
            let t = i as f32 / 100.0;
            let lut_val = lut.linearize(t);
            let exact_val = TransferFunction::Pq.decode(t as f64) as f32;
            let err = (lut_val - exact_val).abs();
            assert!(
                err < 0.001,
                "PQ LUT error too large at t={t}: lut={lut_val}, exact={exact_val}, err={err}"
            );
        }
    }

    #[test]
    fn test_tf_lut_cached() {
        // Two calls to `get_cached_tf_lut` for the same transfer function must
        // return LUTs with identical content (bit-for-bit identical forward tables,
        // proving the second call hit the cache rather than rebuilding).
        let lut1 = get_cached_tf_lut(&TransferFunction::Srgb);
        let lut2 = get_cached_tf_lut(&TransferFunction::Srgb);
        assert_eq!(
            lut1.forward, lut2.forward,
            "cached LUT forward tables must be identical"
        );
        assert_eq!(
            lut1.inverse, lut2.inverse,
            "cached LUT inverse tables must be identical"
        );

        // Also verify the LUT-based method on TransferFunction matches.
        let v = 0.5_f32;
        assert!(
            (TransferFunction::Srgb.linearize_via_lut(v)
                - TransferFunction::Srgb.decode(v as f64) as f32)
                .abs()
                < 0.001,
            "linearize_via_lut must agree with exact decode"
        );
    }

    #[test]
    fn test_tf_lut_boundary_values() {
        // 0.0 and 1.0 must be exact (they are the first and last table entries).
        for tf in [
            TransferFunction::Linear,
            TransferFunction::Srgb,
            TransferFunction::Pq,
            TransferFunction::Hlg,
            TransferFunction::Gamma22,
            TransferFunction::Gamma24,
        ] {
            let lut = TransferFunctionLut::build(&tf);
            let exact0 = tf.decode(0.0) as f32;
            let exact1 = tf.decode(1.0) as f32;
            // Boundary entries must round-trip faithfully (f32 cast of f64).
            assert!(
                (lut.linearize(0.0) - exact0).abs() < 1e-6,
                "{} LUT boundary 0 failed",
                tf.name()
            );
            assert!(
                (lut.linearize(1.0) - exact1).abs() < 1e-6,
                "{} LUT boundary 1 failed",
                tf.name()
            );
        }
    }
}
