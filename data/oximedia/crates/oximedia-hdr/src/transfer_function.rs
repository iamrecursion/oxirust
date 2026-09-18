//! Transfer functions for HDR video: SMPTE ST 2084 (PQ), HLG, and SDR gamma.
//!
//! Reference luminance for PQ: 10 000 cd/m² (nits).
//! Reference luminance for HLG: 1 000 cd/m².
//! Reference luminance for SDR: 100 cd/m².

use crate::{HdrError, Result};

// ── PQ (SMPTE ST 2084) constants ─────────────────────────────────────────────
/// m1 = 2610 / 16384
const M1: f64 = 0.159_301_757_812_5;
/// m2 = 2523/32 × 128 = 78.84375
const M2: f64 = 78.843_75;
/// c1 = 3424 / 4096
const C1: f64 = 0.835_937_5;
/// c2 = 2413 / 128
const C2: f64 = 18.851_562_5;
/// c3 = 2392 / 128
const C3: f64 = 18.687_5;

// ── HLG constants ─────────────────────────────────────────────────────────────
const HLG_A: f64 = 0.178_832_77;
const HLG_B: f64 = 0.284_668_92;
const HLG_C: f64 = 0.559_910_73;
const HLG_THRESHOLD: f64 = 1.0 / 12.0; // scene linear threshold

/// SMPTE ST 2084 PQ OETF: scene-linear (normalised to 10 000 nits) → PQ signal [0, 1].
///
/// # Errors
/// Returns `HdrError::InvalidLuminance` if `lin` is negative.
pub fn pq_oetf(lin: f64) -> Result<f64> {
    if lin < 0.0 {
        return Err(HdrError::InvalidLuminance(lin as f32));
    }
    let y = lin.max(0.0).powf(M1);
    let num = C1 + C2 * y;
    let den = 1.0 + C3 * y;
    Ok((num / den).powf(M2))
}

/// SMPTE ST 2084 PQ EOTF: PQ signal [0, 1] → scene-linear (multiply by 10 000 for nits).
///
/// # Errors
/// Returns `HdrError::InvalidLuminance` if `pq` is outside [0, 1].
pub fn pq_eotf(pq: f64) -> Result<f64> {
    if !(0.0..=1.0).contains(&pq) {
        return Err(HdrError::InvalidLuminance(pq as f32));
    }
    let v = pq.max(0.0).powf(1.0 / M2);
    let num = (v - C1).max(0.0);
    let den = C2 - C3 * v;
    if den <= 0.0 {
        return Ok(1.0);
    }
    Ok((num / den).powf(1.0 / M1))
}

/// HLG OETF: scene-linear [0, 1] → HLG signal [0, 1].
///
/// Reference: ARIB STD-B67.
///
/// # Errors
/// Returns `HdrError::InvalidLuminance` if `lin` is negative.
pub fn hlg_oetf(lin: f64) -> Result<f64> {
    if lin < 0.0 {
        return Err(HdrError::InvalidLuminance(lin as f32));
    }
    if lin <= HLG_THRESHOLD {
        Ok((3.0 * lin).sqrt())
    } else {
        Ok(HLG_A * (12.0 * lin - HLG_B).ln() + HLG_C)
    }
}

/// HLG EOTF: HLG signal [0, 1] → scene-linear [0, 1].
///
/// # Errors
/// Returns `HdrError::InvalidLuminance` if `hlg` is outside [0, 1].
pub fn hlg_eotf(hlg: f64) -> Result<f64> {
    if !(0.0..=1.0).contains(&hlg) {
        return Err(HdrError::InvalidLuminance(hlg as f32));
    }
    // Threshold in the signal domain: hlg_oetf(1/12) = sqrt(3 * 1/12) = 0.5
    const THRESHOLD_SIGNAL: f64 = 0.5;
    if hlg <= THRESHOLD_SIGNAL {
        Ok((hlg * hlg) / 3.0)
    } else {
        Ok((((hlg - HLG_C) / HLG_A).exp() + HLG_B) / 12.0)
    }
}

/// SDR gamma EOTF (gamma 2.2): encoded signal → scene-linear.
pub fn sdr_gamma(v: f64) -> f64 {
    v.max(0.0).powf(2.2)
}

/// SDR gamma OETF (gamma 1/2.2): scene-linear → encoded signal.
pub fn sdr_gamma_inv(v: f64) -> f64 {
    v.max(0.0).powf(1.0 / 2.2)
}

// ── TransferFunction enum ─────────────────────────────────────────────────────

/// Transfer function selection for HDR / SDR video.
#[derive(Debug, Clone, PartialEq)]
pub enum TransferFunction {
    /// SMPTE ST 2084 Perceptual Quantizer (PQ / HDR10 / HDR10+).
    Pq,
    /// ARIB STD-B67 Hybrid Log-Gamma (HLG).
    Hlg,
    /// Power-law SDR gamma with configurable exponent.
    SdrGamma(f32),
    /// Linear light, no transfer function applied.
    Linear,
}

impl TransferFunction {
    /// Convert an encoded signal value to scene-linear light.
    ///
    /// For `Pq` the linear output is normalised to 1.0 = 10 000 nits.
    ///
    /// # Errors
    /// Propagates `HdrError::InvalidLuminance` for out-of-range inputs.
    pub fn to_linear(&self, v: f64) -> Result<f64> {
        match self {
            TransferFunction::Pq => pq_eotf(v),
            TransferFunction::Hlg => hlg_eotf(v),
            TransferFunction::SdrGamma(g) => Ok(v.max(0.0).powf(f64::from(*g))),
            TransferFunction::Linear => Ok(v),
        }
    }

    /// Convert scene-linear light to an encoded signal value.
    ///
    /// For `Pq` the linear input should be normalised to 1.0 = 10 000 nits.
    ///
    /// # Errors
    /// Propagates `HdrError::InvalidLuminance` for negative inputs.
    pub fn from_linear(&self, v: f64) -> Result<f64> {
        match self {
            TransferFunction::Pq => pq_oetf(v),
            TransferFunction::Hlg => hlg_oetf(v),
            TransferFunction::SdrGamma(g) => Ok(v.max(0.0).powf(1.0 / f64::from(*g))),
            TransferFunction::Linear => Ok(v),
        }
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            TransferFunction::Pq => "PQ (SMPTE ST 2084)",
            TransferFunction::Hlg => "HLG (ARIB STD-B67)",
            TransferFunction::SdrGamma(_) => "SDR Gamma",
            TransferFunction::Linear => "Linear",
        }
    }

    /// Peak luminance in nits for this transfer function.
    pub fn peak_luminance_nits(&self) -> f64 {
        match self {
            TransferFunction::Pq => 10_000.0,
            TransferFunction::Hlg => 1_000.0,
            TransferFunction::SdrGamma(_) => 100.0,
            TransferFunction::Linear => 100.0,
        }
    }
}

// ── Batch SIMD-friendly PQ processing ────────────────────────────────────────

/// Branchless PQ EOTF for batch / LUT use (clamps out-of-range inputs).
#[inline]
pub fn pq_eotf_fast(pq: f64) -> f64 {
    let pq = pq.clamp(0.0, 1.0);
    let v = pq.powf(1.0 / M2);
    let num = (v - C1).max(0.0);
    let den = (C2 - C3 * v).max(1e-12);
    (num / den).powf(1.0 / M1)
}

/// Branchless PQ OETF for batch / LUT use (clamps negative inputs).
#[inline]
pub fn pq_oetf_fast(lin: f64) -> f64 {
    let lin = lin.max(0.0);
    let y = lin.powf(M1);
    let num = C1 + C2 * y;
    let den = 1.0 + C3 * y;
    (num / den).powf(M2)
}

/// Branchless HLG EOTF for batch / LUT use (clamps out-of-range inputs).
#[inline]
fn hlg_eotf_fast(hlg: f64) -> f64 {
    let hlg = hlg.clamp(0.0, 1.0);
    const THRESHOLD_SIGNAL: f64 = 0.5;
    if hlg <= THRESHOLD_SIGNAL {
        (hlg * hlg) / 3.0
    } else {
        (((hlg - HLG_C) / HLG_A).exp() + HLG_B) / 12.0
    }
}

/// SIMD-friendly batch PQ EOTF: convert a slice of PQ signal values to linear light.
///
/// All inputs are clamped to [0, 1].  Outputs are normalised to 1.0 = 10 000 nits.
/// The loop is structured for auto-vectorisation (no branches on element values).
///
/// # Errors
/// Returns [`crate::HdrError::ToneMappingError`] if `pq_signals` and `out` differ in length.
pub fn pq_eotf_batch(pq_signals: &[f32], out: &mut [f32]) -> crate::Result<()> {
    if pq_signals.len() != out.len() {
        return Err(crate::HdrError::ToneMappingError(format!(
            "pq_eotf_batch: input length {} != output length {}",
            pq_signals.len(),
            out.len()
        )));
    }
    for (i, &pq) in pq_signals.iter().enumerate() {
        out[i] = pq_eotf_fast(f64::from(pq)) as f32;
    }
    Ok(())
}

/// SIMD-friendly batch PQ OETF: convert linear light values to PQ signal.
///
/// Negative inputs are clamped to 0.  Inputs should be normalised so that
/// 1.0 = 10 000 nits.
///
/// # Errors
/// Returns [`crate::HdrError::ToneMappingError`] if `linear_values` and `out` differ in length.
pub fn pq_oetf_batch(linear_values: &[f32], out: &mut [f32]) -> crate::Result<()> {
    if linear_values.len() != out.len() {
        return Err(crate::HdrError::ToneMappingError(format!(
            "pq_oetf_batch: input length {} != output length {}",
            linear_values.len(),
            out.len()
        )));
    }
    for (i, &lin) in linear_values.iter().enumerate() {
        out[i] = pq_oetf_fast(f64::from(lin)) as f32;
    }
    Ok(())
}

// ── LUT-based fast paths ──────────────────────────────────────────────────────

/// Pre-computed look-up table for the PQ EOTF (PQ signal → linear light).
///
/// Provides ~0.03% max relative error with linear interpolation between
/// 4 096 uniformly-spaced entries spanning [0, 1].
#[derive(Debug, Clone)]
pub struct PqEotfLut {
    table: Vec<f32>,
    n_minus_1: f32,
}

impl PqEotfLut {
    /// Build the LUT with `n_entries` uniformly spaced samples.
    ///
    /// Typical value: 4096 (≈0.03% max error).
    pub fn new(n_entries: usize) -> Self {
        let n = n_entries.max(2);
        let table: Vec<f32> = (0..n)
            .map(|i| {
                let pq = i as f64 / (n - 1) as f64;
                pq_eotf_fast(pq) as f32
            })
            .collect();
        Self {
            table,
            n_minus_1: (n - 1) as f32,
        }
    }

    /// Evaluate the EOTF for a single PQ signal value via linear interpolation.
    pub fn eval(&self, pq: f32) -> f32 {
        let pq = pq.clamp(0.0, 1.0);
        let pos = pq * self.n_minus_1;
        let lo = pos.floor() as usize;
        let hi = (lo + 1).min(self.table.len() - 1);
        let frac = pos - lo as f32;
        self.table[lo] + frac * (self.table[hi] - self.table[lo])
    }

    /// Evaluate the EOTF for a batch of PQ signals into `out`.
    pub fn eval_batch(&self, pq_signals: &[f32], out: &mut [f32]) {
        let len = pq_signals.len().min(out.len());
        for i in 0..len {
            out[i] = self.eval(pq_signals[i]);
        }
    }
}

/// Pre-computed look-up table for the PQ OETF (linear light → PQ signal).
#[derive(Debug, Clone)]
pub struct PqOetfLut {
    table: Vec<f32>,
    n_minus_1: f32,
}

impl PqOetfLut {
    /// Build the LUT with `n_entries` uniformly spaced samples over [0, 1] linear.
    pub fn new(n_entries: usize) -> Self {
        let n = n_entries.max(2);
        let table: Vec<f32> = (0..n)
            .map(|i| {
                let lin = i as f64 / (n - 1) as f64;
                pq_oetf_fast(lin) as f32
            })
            .collect();
        Self {
            table,
            n_minus_1: (n - 1) as f32,
        }
    }

    /// Evaluate the OETF for a single linear light value via linear interpolation.
    pub fn eval(&self, lin: f32) -> f32 {
        let lin = lin.clamp(0.0, 1.0);
        let pos = lin * self.n_minus_1;
        let lo = pos.floor() as usize;
        let hi = (lo + 1).min(self.table.len() - 1);
        let frac = pos - lo as f32;
        self.table[lo] + frac * (self.table[hi] - self.table[lo])
    }

    /// Evaluate the OETF for a batch of linear values into `out`.
    pub fn eval_batch(&self, linear_values: &[f32], out: &mut [f32]) {
        let len = linear_values.len().min(out.len());
        for i in 0..len {
            out[i] = self.eval(linear_values[i]);
        }
    }
}

/// Pre-computed look-up table for the HLG EOTF (HLG signal → linear light).
#[derive(Debug, Clone)]
pub struct HlgEotfLut {
    table: Vec<f32>,
    n_minus_1: f32,
}

impl HlgEotfLut {
    /// Build the LUT with `n_entries` uniformly spaced samples over [0, 1].
    pub fn new(n_entries: usize) -> Self {
        let n = n_entries.max(2);
        let table: Vec<f32> = (0..n)
            .map(|i| {
                let hlg = i as f64 / (n - 1) as f64;
                hlg_eotf_fast(hlg) as f32
            })
            .collect();
        Self {
            table,
            n_minus_1: (n - 1) as f32,
        }
    }

    /// Evaluate the HLG EOTF via linear interpolation.
    pub fn eval(&self, hlg: f32) -> f32 {
        let hlg = hlg.clamp(0.0, 1.0);
        let pos = hlg * self.n_minus_1;
        let lo = pos.floor() as usize;
        let hi = (lo + 1).min(self.table.len() - 1);
        let frac = pos - lo as f32;
        self.table[lo] + frac * (self.table[hi] - self.table[lo])
    }

    /// Evaluate the HLG EOTF for a batch of signals into `out`.
    pub fn eval_batch(&self, hlg_signals: &[f32], out: &mut [f32]) {
        let len = hlg_signals.len().min(out.len());
        for i in 0..len {
            out[i] = self.eval(hlg_signals[i]);
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-6;
    const LOOSE_EPSILON: f64 = 1e-4;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    // 1. PQ round-trip at various luminance levels
    #[test]
    fn test_pq_round_trip_mid() {
        let lin = 0.5_f64;
        let encoded = pq_oetf(lin).expect("pq_oetf");
        let decoded = pq_eotf(encoded).expect("pq_eotf");
        assert!(
            approx_eq(lin, decoded, EPSILON),
            "PQ round-trip mid: {lin} vs {decoded}"
        );
    }

    #[test]
    fn test_pq_round_trip_zero() {
        let encoded = pq_oetf(0.0).expect("pq_oetf 0");
        let decoded = pq_eotf(encoded).expect("pq_eotf 0");
        assert!(
            approx_eq(0.0, decoded, EPSILON),
            "PQ round-trip 0: {decoded}"
        );
    }

    #[test]
    fn test_pq_round_trip_one() {
        let encoded = pq_oetf(1.0).expect("pq_oetf 1");
        assert!(
            approx_eq(encoded, 1.0, EPSILON),
            "PQ(1.0) should encode to 1.0, got {encoded}"
        );
        let decoded = pq_eotf(encoded).expect("pq_eotf 1");
        assert!(
            approx_eq(1.0, decoded, EPSILON),
            "PQ round-trip 1.0: {decoded}"
        );
    }

    // 2. PQ reference: signal ≈ 0.5081 corresponds to 100 nits (normalised to 10 000)
    //    i.e. pq_eotf(0.5081) ≈ 0.01 (= 100/10000)
    #[test]
    fn test_pq_reference_100_nits() {
        // 100 nits / 10000 = 0.01 linear
        let signal = pq_oetf(0.01).expect("pq_oetf 100nits");
        // Published reference: ~0.5081
        assert!(
            approx_eq(signal, 0.5081, 1e-3),
            "PQ(100 nits) signal expected ~0.5081, got {signal}"
        );
        let lin = pq_eotf(signal).expect("pq_eotf 100nits");
        assert!(
            approx_eq(lin, 0.01, LOOSE_EPSILON),
            "PQ EOTF(0.5081) expected 0.01, got {lin}"
        );
    }

    // 3. PQ edge case: negative input should error
    #[test]
    fn test_pq_oetf_negative_error() {
        assert!(pq_oetf(-0.1).is_err(), "pq_oetf(-0.1) should error");
    }

    #[test]
    fn test_pq_eotf_out_of_range_error() {
        assert!(pq_eotf(1.1).is_err(), "pq_eotf(1.1) should error");
        assert!(pq_eotf(-0.1).is_err(), "pq_eotf(-0.1) should error");
    }

    // 4. HLG round-trip
    #[test]
    fn test_hlg_round_trip_low() {
        let lin = 0.05_f64;
        let enc = hlg_oetf(lin).expect("hlg_oetf low");
        let dec = hlg_eotf(enc).expect("hlg_eotf low");
        assert!(
            approx_eq(lin, dec, LOOSE_EPSILON),
            "HLG round-trip low: {lin} vs {dec}"
        );
    }

    #[test]
    fn test_hlg_round_trip_high() {
        let lin = 0.5_f64;
        let enc = hlg_oetf(lin).expect("hlg_oetf high");
        let dec = hlg_eotf(enc).expect("hlg_eotf high");
        assert!(
            approx_eq(lin, dec, LOOSE_EPSILON),
            "HLG round-trip high: {lin} vs {dec}"
        );
    }

    #[test]
    fn test_hlg_oetf_zero() {
        let enc = hlg_oetf(0.0).expect("hlg_oetf 0");
        assert!(approx_eq(enc, 0.0, EPSILON), "HLG OETF(0) = {enc}");
    }

    #[test]
    fn test_hlg_eotf_zero() {
        let dec = hlg_eotf(0.0).expect("hlg_eotf 0");
        assert!(approx_eq(dec, 0.0, EPSILON), "HLG EOTF(0) = {dec}");
    }

    #[test]
    fn test_hlg_negative_error() {
        assert!(hlg_oetf(-0.01).is_err());
        assert!(hlg_eotf(-0.01).is_err());
    }

    // 5. SDR gamma
    #[test]
    fn test_sdr_gamma_round_trip() {
        let v = 0.5_f64;
        let lin = sdr_gamma(v);
        let enc = sdr_gamma_inv(lin);
        assert!(
            approx_eq(v, enc, EPSILON),
            "SDR gamma round-trip: {v} vs {enc}"
        );
    }

    // 6. TransferFunction enum
    #[test]
    fn test_transfer_function_pq_name() {
        assert_eq!(TransferFunction::Pq.name(), "PQ (SMPTE ST 2084)");
    }

    #[test]
    fn test_transfer_function_peak_luminance() {
        assert!((TransferFunction::Pq.peak_luminance_nits() - 10_000.0).abs() < 1.0);
        assert!((TransferFunction::Hlg.peak_luminance_nits() - 1_000.0).abs() < 1.0);
        assert!((TransferFunction::SdrGamma(2.2).peak_luminance_nits() - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_transfer_function_to_from_linear_pq() {
        let tf = TransferFunction::Pq;
        let lin = 0.3_f64;
        let enc = tf.from_linear(lin).expect("from_linear pq");
        let dec = tf.to_linear(enc).expect("to_linear pq");
        assert!(
            approx_eq(lin, dec, EPSILON),
            "TF PQ round-trip: {lin} vs {dec}"
        );
    }

    #[test]
    fn test_transfer_function_linear_passthrough() {
        let tf = TransferFunction::Linear;
        let v = 0.7_f64;
        let enc = tf.from_linear(v).expect("from_linear linear");
        let dec = tf.to_linear(enc).expect("to_linear linear");
        assert!(approx_eq(v, dec, EPSILON));
    }

    // ── BT.2100 reference values ──────────────────────────────────────────────
    // Values from ITU-R BT.2100-2 Table 4 (computed from the PQ function).

    // 10 nits / 10000 = 0.001 linear → PQ signal ≈ 0.3021
    #[test]
    fn test_pq_oetf_bt2100_ref_10_nits() {
        let signal = pq_oetf(0.001).expect("pq_oetf 10nits");
        assert!(
            approx_eq(signal, 0.3021, 5e-3),
            "PQ(10 nits) signal expected ~0.3021, got {signal}"
        );
    }

    // 1000 nits / 10000 = 0.10 linear → PQ signal ≈ 0.7523
    #[test]
    fn test_pq_oetf_bt2100_ref_1000_nits() {
        let signal = pq_oetf(0.10).expect("pq_oetf 1000nits");
        assert!(
            approx_eq(signal, 0.7523, 5e-3),
            "PQ(1000 nits) signal expected ~0.7523, got {signal}"
        );
    }

    // Inverse: PQ signal ≈ 0.7523 → 0.10 linear (1000 nits / 10000)
    #[test]
    fn test_pq_eotf_bt2100_ref_1000_nits() {
        let signal = pq_oetf(0.10).expect("pq_oetf 1000nits for eotf test");
        let lin = pq_eotf(signal).expect("pq_eotf 1000nits");
        assert!(
            approx_eq(lin, 0.10, 1e-4),
            "PQ EOTF({signal}) expected 0.10, got {lin}"
        );
    }

    // ── Monotonicity tests ────────────────────────────────────────────────────

    #[test]
    fn test_pq_oetf_monotonic() {
        let mut prev = pq_oetf(0.0).expect("pq_oetf 0");
        for i in 1..=100 {
            let lin = i as f64 / 100.0;
            let cur = pq_oetf(lin).expect("pq_oetf mono");
            assert!(
                cur >= prev - 1e-10,
                "pq_oetf not monotonic at {lin}: {cur} < {prev}"
            );
            prev = cur;
        }
    }

    #[test]
    fn test_pq_eotf_monotonic() {
        let mut prev = pq_eotf(0.0).expect("pq_eotf 0");
        for i in 1..=100 {
            let pq = i as f64 / 100.0;
            let cur = pq_eotf(pq).expect("pq_eotf mono");
            assert!(
                cur >= prev - 1e-10,
                "pq_eotf not monotonic at {pq}: {cur} < {prev}"
            );
            prev = cur;
        }
    }

    // ── Batch PQ EOTF/OETF ───────────────────────────────────────────────────

    #[test]
    fn test_pq_eotf_batch_round_trip() {
        let input: Vec<f32> = (0..=10).map(|i| i as f32 / 10.0).collect();
        let mut encoded = vec![0.0f32; input.len()];
        let mut decoded = vec![0.0f32; input.len()];
        pq_oetf_batch(&input, &mut encoded).expect("pq_oetf_batch");
        pq_eotf_batch(&encoded, &mut decoded).expect("pq_eotf_batch");
        for (i, (&orig, &dec)) in input.iter().zip(decoded.iter()).enumerate() {
            assert!(
                (orig - dec).abs() < 1e-5,
                "batch round-trip at index {i}: {orig} vs {dec}"
            );
        }
    }

    #[test]
    fn test_pq_eotf_batch_length_mismatch() {
        let input = vec![0.5f32; 5];
        let mut out = vec![0.0f32; 4]; // wrong length
        assert!(pq_eotf_batch(&input, &mut out).is_err());
    }

    #[test]
    fn test_pq_oetf_batch_length_mismatch() {
        let input = vec![0.5f32; 5];
        let mut out = vec![0.0f32; 6]; // wrong length
        assert!(pq_oetf_batch(&input, &mut out).is_err());
    }

    // ── PqEotfLut ─────────────────────────────────────────────────────────────

    #[test]
    fn test_pq_eotf_lut_accuracy() {
        let lut = PqEotfLut::new(4096);
        for i in 0..=100 {
            let pq = i as f64 / 100.0;
            let exact = pq_eotf_fast(pq) as f32;
            let approx = lut.eval(pq as f32);
            let rel_err = if exact.abs() > 1e-6 {
                ((approx - exact) / exact).abs()
            } else {
                (approx - exact).abs()
            };
            assert!(
                rel_err < 5e-4,
                "PqEotfLut error at pq={pq}: exact={exact} approx={approx} rel_err={rel_err}"
            );
        }
    }

    #[test]
    fn test_pq_eotf_lut_monotonic() {
        let lut = PqEotfLut::new(4096);
        let mut prev = lut.eval(0.0);
        for i in 1..=100 {
            let pq = i as f32 / 100.0;
            let cur = lut.eval(pq);
            assert!(
                cur >= prev - 1e-6,
                "PqEotfLut not monotonic at {pq}: {cur} < {prev}"
            );
            prev = cur;
        }
    }

    // ── PqOetfLut ─────────────────────────────────────────────────────────────

    #[test]
    fn test_pq_oetf_lut_accuracy() {
        let lut = PqOetfLut::new(4096);
        for i in 0..=100 {
            let lin = i as f64 / 100.0;
            let exact = pq_oetf_fast(lin) as f32;
            let approx = lut.eval(lin as f32);
            let rel_err = if exact.abs() > 1e-6 {
                ((approx - exact) / exact).abs()
            } else {
                (approx - exact).abs()
            };
            assert!(
                rel_err < 5e-4,
                "PqOetfLut error at lin={lin}: exact={exact} approx={approx} rel_err={rel_err}"
            );
        }
    }

    // ── HlgEotfLut ────────────────────────────────────────────────────────────

    #[test]
    fn test_hlg_eotf_lut_accuracy() {
        let lut = HlgEotfLut::new(4096);
        for i in 0..=100 {
            let hlg = i as f64 / 100.0;
            let exact = hlg_eotf_fast(hlg) as f32;
            let approx = lut.eval(hlg as f32);
            let rel_err = if exact.abs() > 1e-6 {
                ((approx - exact) / exact).abs()
            } else {
                (approx - exact).abs()
            };
            assert!(
                rel_err < 5e-4,
                "HlgEotfLut error at hlg={hlg}: exact={exact} approx={approx} rel_err={rel_err}"
            );
        }
    }

    #[test]
    fn test_hlg_eotf_lut_eval_batch() {
        let lut = HlgEotfLut::new(4096);
        let input: Vec<f32> = (0..=10).map(|i| i as f32 / 10.0).collect();
        let mut out = vec![0.0f32; input.len()];
        lut.eval_batch(&input, &mut out);
        for &v in &out {
            assert!(
                (0.0..=1.0).contains(&v),
                "HlgEotfLut output {v} out of range"
            );
        }
    }
}
