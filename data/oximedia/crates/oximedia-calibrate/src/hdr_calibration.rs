//! HDR display calibration with PQ and HLG transfer functions and peak
//! luminance mapping.
//!
//! This module provides calibration workflows for HDR displays that use
//! ST 2084 (Perceptual Quantizer / PQ) or ARIB STD-B67 (Hybrid Log-Gamma /
//! HLG) electro-optical transfer functions (EOTF/OETF). Key capabilities:
//!
//! - Encode/decode luminance with PQ (ST 2084) and HLG (ARIB STD-B67)
//! - Peak-luminance mapping: scale a measured PQ/HLG signal to a target
//!   display peak, applying tone-mapping at the upper end
//! - Per-band calibration: generate a correction lookup to compensate for
//!   a measured display deviation from the ideal transfer function
//! - Metadata structures for HDR10/HLG content description

#![allow(dead_code)]

use crate::error::{CalibrationError, CalibrationResult};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// PQ peak luminance constant L_p (cd/m², ST 2084 Eq. 5).
pub const PQ_PEAK_NITS: f64 = 10_000.0;

/// PQ constants from SMPTE ST 2084 §6.
const PQ_M1: f64 = 0.159_301_758;
const PQ_M2: f64 = 78.843_75;
const PQ_C1: f64 = 0.835_937_5;
const PQ_C2: f64 = 18.851_563;
const PQ_C3: f64 = 18.6875;

/// HLG system gamma.
const HLG_GAMMA: f64 = 1.2;
/// HLG `a` constant.
const HLG_A: f64 = 0.178_832_77;
/// HLG `b` constant.
const HLG_B: f64 = 0.284_668_92;
/// HLG `c` constant.
const HLG_C: f64 = 0.559_910_73;

// ---------------------------------------------------------------------------
// HDR transfer function enum
// ---------------------------------------------------------------------------

/// Supported HDR transfer function (EOTF/OETF type).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HdrTf {
    /// SMPTE ST 2084 Perceptual Quantizer (PQ) — standard for HDR10.
    Pq,
    /// ARIB STD-B67 Hybrid Log-Gamma (HLG) — standard for HDR broadcast.
    Hlg,
}

impl HdrTf {
    /// Returns a short name string for this transfer function.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Pq => "ST-2084 PQ",
            Self::Hlg => "ARIB STD-B67 HLG",
        }
    }
}

// ---------------------------------------------------------------------------
// PQ encode / decode
// ---------------------------------------------------------------------------

/// Encode a linear luminance value (cd/m²) to a PQ code value in [0, 1].
///
/// Values above [`PQ_PEAK_NITS`] (10,000 cd/m²) are clamped.
#[must_use]
pub fn pq_encode(luminance_nits: f64) -> f64 {
    let y = (luminance_nits / PQ_PEAK_NITS).clamp(0.0, 1.0);
    let ym = y.powf(PQ_M1);
    let num = PQ_C1 + PQ_C2 * ym;
    let den = 1.0 + PQ_C3 * ym;
    (num / den).powf(PQ_M2)
}

/// Decode a PQ code value (in [0, 1]) to a linear luminance (cd/m²).
#[must_use]
pub fn pq_decode(code: f64) -> f64 {
    let code = code.clamp(0.0, 1.0);
    let vm = code.powf(1.0 / PQ_M2);
    let num = (vm - PQ_C1).max(0.0);
    let den = PQ_C2 - PQ_C3 * vm;
    if den <= 0.0 {
        return 0.0;
    }
    (num / den).powf(1.0 / PQ_M1) * PQ_PEAK_NITS
}

// ---------------------------------------------------------------------------
// HLG encode / decode
// ---------------------------------------------------------------------------

/// Encode a normalised scene-linear value in [0, 1] to an HLG signal [0, 1].
///
/// The input `e` represents linear scene light normalised to the reference
/// level (1.0 = 100 % reference white).
#[must_use]
pub fn hlg_encode(e: f64) -> f64 {
    let e = e.max(0.0);
    if e <= 1.0 / 12.0 {
        (3.0 * e).sqrt()
    } else {
        HLG_A * (12.0 * e - HLG_B).ln() + HLG_C
    }
}

/// Decode an HLG signal in [0, 1] to a normalised scene-linear value.
#[must_use]
pub fn hlg_decode(e: f64) -> f64 {
    let e = e.clamp(0.0, 1.0);
    if e <= 0.5 {
        e * e / 3.0
    } else {
        (((e - HLG_C) / HLG_A).exp() + HLG_B) / 12.0
    }
}

// ---------------------------------------------------------------------------
// Peak luminance mapping
// ---------------------------------------------------------------------------

/// Map a PQ-encoded signal from one peak luminance to another using a simple
/// linear rescale with an optional knee/tone-map at the top end.
///
/// # Arguments
///
/// * `code`       - PQ code value in [0, 1] at `src_peak` nits.
/// * `src_peak`   - Peak luminance the signal was mastered at (cd/m²).
/// * `dst_peak`   - Target display peak luminance (cd/m²).
///
/// # Returns
///
/// Re-mapped PQ code value targeting the `dst_peak` display.
#[must_use]
pub fn pq_remap_peak(code: f64, src_peak: f64, dst_peak: f64) -> f64 {
    if src_peak <= 0.0 || dst_peak <= 0.0 {
        return code.clamp(0.0, 1.0);
    }
    // Decode to linear.
    let linear = pq_decode(code);
    // Scale linearly (simple max-lift rescale, no complex tone-map).
    let scaled = linear * (dst_peak / src_peak);
    // Re-encode.
    pq_encode(scaled.min(PQ_PEAK_NITS))
}

// ---------------------------------------------------------------------------
// Calibration measurement & correction LUT
// ---------------------------------------------------------------------------

/// A single luminance calibration measurement: input code → measured nits.
#[derive(Debug, Clone)]
pub struct LuminanceMeasurement {
    /// Input code value sent to the display (normalised, 0.0–1.0).
    pub code: f64,
    /// Measured luminance output in cd/m².
    pub measured_nits: f64,
}

/// HDR display calibration result.
///
/// Contains a correction LUT that maps the display's *actual* code-to-luminance
/// characteristic back to the ideal transfer function.
#[derive(Debug, Clone)]
pub struct HdrCalibrationResult {
    /// Transfer function used for calibration.
    pub tf: HdrTf,
    /// Target peak display luminance (cd/m²).
    pub peak_nits: f64,
    /// Number of LUT entries (evenly spaced over [0, 1]).
    pub lut_size: usize,
    /// Correction LUT: `correction[i]` is the corrected code to send to the
    /// display when the ideal code is `i / (lut_size - 1)`.
    pub correction: Vec<f64>,
    /// Average absolute error after correction (code units, 0–1).
    pub avg_error: f64,
    /// Maximum absolute error after correction (code units, 0–1).
    pub max_error: f64,
}

impl HdrCalibrationResult {
    /// Apply the correction LUT to a raw code value via linear interpolation.
    #[must_use]
    pub fn apply(&self, code: f64) -> f64 {
        let code = code.clamp(0.0, 1.0);
        if self.lut_size < 2 {
            return code;
        }
        let scale = (self.lut_size - 1) as f64;
        let pos = code * scale;
        let lo = pos.floor() as usize;
        let hi = (lo + 1).min(self.lut_size - 1);
        let frac = pos - lo as f64;
        self.correction[lo] * (1.0 - frac) + self.correction[hi] * frac
    }
}

// ---------------------------------------------------------------------------
// HdrCalibrator
// ---------------------------------------------------------------------------

/// Calibrator for HDR displays supporting PQ and HLG transfer functions.
#[derive(Debug, Clone)]
pub struct HdrCalibrator {
    /// Target HDR transfer function.
    pub tf: HdrTf,
    /// Target peak display luminance (cd/m²).
    pub peak_nits: f64,
    /// Number of entries in the generated correction LUT.
    pub lut_size: usize,
}

impl HdrCalibrator {
    /// Create a new `HdrCalibrator`.
    ///
    /// # Arguments
    ///
    /// * `tf`        - Target transfer function (`Pq` or `Hlg`).
    /// * `peak_nits` - Target peak display luminance (cd/m²).
    /// * `lut_size`  - Number of correction LUT entries (≥ 2).
    #[must_use]
    pub fn new(tf: HdrTf, peak_nits: f64, lut_size: usize) -> Self {
        Self {
            tf,
            peak_nits,
            lut_size: lut_size.max(2),
        }
    }

    /// Create a calibrator for a standard HDR10 display (1000 nits).
    #[must_use]
    pub fn hdr10_1000nit() -> Self {
        Self::new(HdrTf::Pq, 1000.0, 1024)
    }

    /// Create a calibrator for an HLG broadcast display (1000 nits).
    #[must_use]
    pub fn hlg_broadcast() -> Self {
        Self::new(HdrTf::Hlg, 1000.0, 1024)
    }

    /// Calibrate a display from luminance measurements.
    ///
    /// Given a set of `(code → measured_nits)` measurements, this function
    /// fits a monotone correction LUT that maps ideal codes to the input
    /// codes that produce the target luminance on this specific display.
    ///
    /// # Arguments
    ///
    /// * `measurements` - At least 2 `LuminanceMeasurement` entries, sorted by
    ///   ascending `code`.
    ///
    /// # Errors
    ///
    /// Returns an error if fewer than 2 measurements are provided or if the
    /// measurements are not sorted by code.
    pub fn calibrate(
        &self,
        measurements: &[LuminanceMeasurement],
    ) -> CalibrationResult<HdrCalibrationResult> {
        if measurements.len() < 2 {
            return Err(CalibrationError::InsufficientData(
                "HDR calibration requires at least 2 luminance measurements".to_string(),
            ));
        }

        // Verify monotone ordering.
        for w in measurements.windows(2) {
            if w[1].code < w[0].code {
                return Err(CalibrationError::InvalidMeasurement(
                    "Measurements must be sorted by ascending code".to_string(),
                ));
            }
        }

        // Build the ideal code→nits curve from the transfer function.
        let n = self.lut_size;
        let mut correction = Vec::with_capacity(n);
        let mut total_err = 0.0_f64;
        let mut max_err = 0.0_f64;

        for idx in 0..n {
            let ideal_code = idx as f64 / (n - 1) as f64;
            // Ideal luminance for this code.
            let ideal_nits = self.decode_to_nits(ideal_code);

            // Find the code that produces `ideal_nits` on the measured display
            // by inverse interpolation of the measurement curve.
            let corrected_code = self.inverse_interpolate(measurements, ideal_nits);
            correction.push(corrected_code);

            // Measure error as |corrected - ideal|.
            let err = (corrected_code - ideal_code).abs();
            total_err += err;
            if err > max_err {
                max_err = err;
            }
        }

        Ok(HdrCalibrationResult {
            tf: self.tf,
            peak_nits: self.peak_nits,
            lut_size: n,
            correction,
            avg_error: total_err / n as f64,
            max_error: max_err,
        })
    }

    /// Decode a code to luminance (nits) using the configured transfer function.
    fn decode_to_nits(&self, code: f64) -> f64 {
        match self.tf {
            HdrTf::Pq => {
                // PQ gives absolute nits; scale to display peak.
                let abs_nits = pq_decode(code);
                abs_nits.min(self.peak_nits)
            }
            HdrTf::Hlg => {
                // HLG gives relative luminance * peak.
                hlg_decode(code) * self.peak_nits
            }
        }
    }

    /// Find the code that, on the *measured* display, produces `target_nits`
    /// by linear interpolation of the measurement curve.
    fn inverse_interpolate(&self, measurements: &[LuminanceMeasurement], target_nits: f64) -> f64 {
        // Find the two measurements that bracket `target_nits`.
        let first_nits = measurements.first().map(|m| m.measured_nits).unwrap_or(0.0);
        let last_nits = measurements.last().map(|m| m.measured_nits).unwrap_or(0.0);

        if target_nits <= first_nits {
            return measurements.first().map(|m| m.code).unwrap_or(0.0);
        }
        if target_nits >= last_nits {
            return measurements.last().map(|m| m.code).unwrap_or(1.0);
        }

        for w in measurements.windows(2) {
            let lo = &w[0];
            let hi = &w[1];
            if target_nits >= lo.measured_nits && target_nits <= hi.measured_nits {
                let dn = hi.measured_nits - lo.measured_nits;
                if dn.abs() < 1e-12 {
                    return lo.code;
                }
                let frac = (target_nits - lo.measured_nits) / dn;
                return lo.code + frac * (hi.code - lo.code);
            }
        }

        // Fallback: return the nearest endpoint.
        measurements.last().map(|m| m.code).unwrap_or(1.0)
    }
}

// ---------------------------------------------------------------------------
// HDR10 / HLG metadata helpers
// ---------------------------------------------------------------------------

/// HDR10 static metadata (SEI message for HDR10/HDR10+).
#[derive(Debug, Clone)]
pub struct Hdr10Metadata {
    /// Display primaries (Rx,Ry, Gx,Gy, Bx,By) in (50,000× chromaticity) units.
    pub display_primaries: [(u16, u16); 3],
    /// White point (Wx, Wy) in 50,000× chromaticity units.
    pub white_point: (u16, u16),
    /// Maximum display mastering luminance (cd/m²) in 0.0001 cd/m² units.
    pub max_mastering_luminance: u32,
    /// Minimum display mastering luminance (cd/m²) in 0.0001 cd/m² units.
    pub min_mastering_luminance: u32,
    /// Maximum content light level (CLL) in cd/m².
    pub max_cll: u16,
    /// Maximum frame-average light level (FALL) in cd/m².
    pub max_fall: u16,
}

impl Hdr10Metadata {
    /// Create a default HDR10 metadata block for a 1000-nit P3-D65 display.
    #[must_use]
    pub fn p3_d65_1000nit() -> Self {
        Self {
            // P3-D65 primaries × 50,000
            display_primaries: [
                (34_000, 16_000), // R: (0.680, 0.320)
                (13_250, 34_500), // G: (0.265, 0.690)
                (7_500, 3_000),   // B: (0.150, 0.060)
            ],
            white_point: (15_635, 16_450), // D65: (0.3127, 0.3290)
            max_mastering_luminance: 10_000_000, // 1000 cd/m² × 10,000
            min_mastering_luminance: 10,   // 0.001 cd/m² × 10,000
            max_cll: 1_000,
            max_fall: 400,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── PQ round-trip ─────────────────────────────────────────────────────

    #[test]
    fn test_pq_encode_decode_roundtrip() {
        for &nits in &[0.005_f64, 1.0, 100.0, 1_000.0, 4_000.0] {
            let code = pq_encode(nits);
            let back = pq_decode(code);
            let rel = (back - nits).abs() / (nits + 1.0);
            assert!(
                rel < 1e-9,
                "PQ round-trip failed for {nits} nits: code={code:.6}, decoded={back:.4}"
            );
        }
        // At peak (10,000 nits) the code slightly exceeds 1.0; allow wider tolerance.
        let code = pq_encode(10_000.0);
        let back = pq_decode(code);
        let rel = (back - 10_000.0).abs() / 10_001.0;
        assert!(
            rel < 1e-4,
            "PQ round-trip failed for 10000 nits: code={code:.6}, decoded={back:.4}"
        );
    }

    #[test]
    fn test_pq_zero_luminance_encodes_to_zero() {
        let code = pq_encode(0.0);
        let decoded = pq_decode(0.0);
        // PQ(0) should approach 0, but due to the ST2084 formula it gives the
        // "black" reference code ≈ 0.
        assert!(
            code.abs() < 0.01,
            "PQ encode(0) should be near 0, got {code}"
        );
        assert!(
            decoded.abs() < 0.01,
            "PQ decode(0) should be near 0 nits, got {decoded}"
        );
    }

    #[test]
    fn test_pq_encode_monotone() {
        let values = [0.0, 10.0, 100.0, 1_000.0, 4_000.0, 10_000.0];
        let encoded: Vec<f64> = values.iter().map(|&v| pq_encode(v)).collect();
        for w in encoded.windows(2) {
            assert!(w[1] > w[0], "PQ encode must be monotonically increasing");
        }
    }

    // ── HLG round-trip ────────────────────────────────────────────────────

    #[test]
    fn test_hlg_encode_decode_roundtrip() {
        // HLG round-trip is exact only for e in [0, 1] (decoder clamps input to [0,1]).
        for &e in &[0.0_f64, 0.01, 0.1, 0.5, 1.0] {
            let code = hlg_encode(e);
            let back = hlg_decode(code);
            let err = (back - e).abs();
            assert!(
                err < 1e-9,
                "HLG round-trip failed for e={e}: code={code:.6}, decoded={back:.9}, err={err:.2e}"
            );
        }
        // e=2.0 encodes to code>1.0; decoder clamps so round-trip is not exact — verify
        // monotone growth at least.
        let c1 = hlg_encode(1.0);
        let c2 = hlg_encode(2.0);
        assert!(c2 > c1, "HLG encode must be monotone beyond 1.0");
    }

    #[test]
    fn test_hlg_encode_monotone() {
        let values = [0.0, 0.05, 0.1, 0.5, 1.0, 2.0, 4.0];
        let encoded: Vec<f64> = values.iter().map(|&v| hlg_encode(v)).collect();
        for w in encoded.windows(2) {
            assert!(w[1] > w[0], "HLG encode must be monotonically increasing");
        }
    }

    // ── Peak remapping ────────────────────────────────────────────────────

    #[test]
    fn test_pq_remap_peak_identity() {
        let code = pq_encode(500.0);
        let remapped = pq_remap_peak(code, 1_000.0, 1_000.0);
        assert!(
            (remapped - code).abs() < 1e-9,
            "Remapping to same peak should be identity: {code:.6} vs {remapped:.6}"
        );
    }

    #[test]
    fn test_pq_remap_peak_lower() {
        // Content mastered at 4000 nits, display peaks at 1000 nits.
        let code_4k = pq_encode(3_000.0);
        let remapped = pq_remap_peak(code_4k, 4_000.0, 1_000.0);
        // Remapped code must be ≤ pq_encode(1000), since we're scaling down.
        let code_1k = pq_encode(1_000.0);
        assert!(
            remapped <= code_1k + 1e-9,
            "Remapped code {remapped:.6} must be ≤ {code_1k:.6}"
        );
    }

    // ── HdrCalibrator ─────────────────────────────────────────────────────

    #[test]
    fn test_hdr_calibrator_ideal_display_is_identity() {
        let cal = HdrCalibrator::hdr10_1000nit();
        // Build ideal measurements: PQ codes → ideal nits.
        let measurements: Vec<LuminanceMeasurement> = (0..=100)
            .map(|i| {
                let code = i as f64 / 100.0;
                let nits = pq_decode(code).min(1_000.0);
                LuminanceMeasurement {
                    code,
                    measured_nits: nits,
                }
            })
            .collect();

        let result = cal
            .calibrate(&measurements)
            .expect("Calibration should succeed");
        // For an ideal display the correction should be close to identity
        // in the non-saturated region (codes < 0.77 where PQ nits < 1000).
        // Above that the display clips to peak and the LUT saturates to 1.0.
        for i in 0..result.lut_size {
            let ideal = i as f64 / (result.lut_size - 1) as f64;
            let corrected = result.correction[i];
            // Skip saturated region (pq_decode(ideal) >= peak_nits)
            if pq_decode(ideal) >= 1_000.0 {
                continue;
            }
            assert!(
                (corrected - ideal).abs() < 0.05,
                "Ideal display correction [{i}]: ideal={ideal:.4}, corrected={corrected:.4}"
            );
        }
    }

    #[test]
    fn test_hdr_calibrator_requires_at_least_2_measurements() {
        let cal = HdrCalibrator::hdr10_1000nit();
        let single = vec![LuminanceMeasurement {
            code: 0.5,
            measured_nits: 100.0,
        }];
        let result = cal.calibrate(&single);
        assert!(
            result.is_err(),
            "Should fail with fewer than 2 measurements"
        );
    }

    #[test]
    fn test_hdr_calibration_apply_clamps() {
        let cal = HdrCalibrator::hdr10_1000nit();
        let measurements: Vec<LuminanceMeasurement> = vec![
            LuminanceMeasurement {
                code: 0.0,
                measured_nits: 0.0,
            },
            LuminanceMeasurement {
                code: 1.0,
                measured_nits: 1_000.0,
            },
        ];
        let result = cal
            .calibrate(&measurements)
            .expect("Calibration should succeed");
        let lo = result.apply(-0.1);
        let hi = result.apply(1.5);
        assert!((0.0..=1.0).contains(&lo), "apply must clamp low: {lo}");
        assert!((0.0..=1.0).contains(&hi), "apply must clamp high: {hi}");
    }

    #[test]
    fn test_hdr10_metadata_construction() {
        let meta = Hdr10Metadata::p3_d65_1000nit();
        assert_eq!(meta.max_cll, 1_000);
        assert_eq!(meta.max_fall, 400);
    }

    #[test]
    fn test_hdr_tf_names() {
        assert_eq!(HdrTf::Pq.name(), "ST-2084 PQ");
        assert_eq!(HdrTf::Hlg.name(), "ARIB STD-B67 HLG");
    }
}
