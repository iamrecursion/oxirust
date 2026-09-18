//! ICtCp colour space conversions per ITU-R BT.2100.
//!
//! ICtCp is a perceptual colour space designed for HDR and wide-colour-gamut
//! content.  It separates intensity (I, the PQ-encoded luminance) from
//! chroma tritan-axis (Ct, blue-yellow) and protan-axis (Cp, red-green)
//! components.
//!
//! The conversion pipeline is:
//! 1. Linear RGB (BT.2020) → LMS via a 3×3 cross-talk matrix
//! 2. LMS → PQ-encoded L'M'S' (per-channel PQ OETF)
//! 3. L'M'S' → ICtCp via a second 3×3 matrix
//!
//! All operations are invertible.  See ITU-R BT.2100-2 Note 5 for the
//! matrix coefficient values.

use crate::transfer_function::{pq_eotf, pq_oetf};
use crate::{HdrError, Result};

// ── LMS matrix (Rec.2020 linear RGB → LMS) ──────────────────────────────────

/// BT.2100 / SMPTE ST 2084 matrix: Rec.2020 linear RGB → LMS.
///
/// Row-major 3×3: each row is [R_coeff, G_coeff, B_coeff].
const RGB_TO_LMS: [[f64; 3]; 3] = [
    [1688.0 / 4096.0, 2146.0 / 4096.0, 262.0 / 4096.0],
    [683.0 / 4096.0, 2951.0 / 4096.0, 462.0 / 4096.0],
    [99.0 / 4096.0, 309.0 / 4096.0, 3688.0 / 4096.0],
];

/// Inverse of `RGB_TO_LMS` (LMS → Rec.2020 linear RGB), computed analytically.
const LMS_TO_RGB: [[f64; 3]; 3] = [
    [3.43661, -2.50646, 0.06985],
    [-0.79133, 1.98360, -0.19228],
    [-0.02598, -0.09898, 1.12497],
];

// ── ICtCp matrix (L'M'S' → ICtCp) ──────────────────────────────────────────

/// BT.2100 matrix: L'M'S' (PQ-encoded) → ICtCp.
const LMS_PRIME_TO_ICTCP: [[f64; 3]; 3] = [
    [0.5, 0.5, 0.0],
    [6610.0 / 4096.0, -13613.0 / 4096.0, 7003.0 / 4096.0],
    [17933.0 / 4096.0, -17390.0 / 4096.0, -543.0 / 4096.0],
];

/// Inverse of `LMS_PRIME_TO_ICTCP` (ICtCp → L'M'S').
const ICTCP_TO_LMS_PRIME: [[f64; 3]; 3] = [
    [1.0, 0.008609037037932, 0.111029625003026],
    [1.0, -0.008609037037932, -0.111029625003026],
    [1.0, 0.560031971642994, -0.320627174987319],
];

// ── Low-level helpers ────────────────────────────────────────────────────────

/// Multiply a 3×3 matrix (row-major) by a column vector, producing a 3-vector.
#[inline]
fn mat3_vec_mul(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

// ── Forward conversion: Rec.2020 linear RGB → ICtCp ──────────────────────────

/// Convert a Rec.2020 linear-light RGB triplet to ICtCp.
///
/// Input `r`, `g`, `b` are scene-linear (normalised to 1.0 = 10 000 nits,
/// the PQ absolute peak luminance).
///
/// # Errors
/// Returns `HdrError::ToneMappingError` if any channel is negative or if
/// the PQ OETF fails for any LMS value.
pub fn rgb_to_ictcp(r: f64, g: f64, b: f64) -> Result<(f64, f64, f64)> {
    if r < 0.0 || g < 0.0 || b < 0.0 {
        return Err(HdrError::ToneMappingError(format!(
            "ICtCp input must be non-negative: ({r}, {g}, {b})"
        )));
    }

    // Step 1: Rec.2020 linear RGB → LMS
    let lms = mat3_vec_mul(&RGB_TO_LMS, [r, g, b]);

    // Step 2: Per-channel PQ OETF (LMS → L'M'S')
    let l_prime = pq_oetf(lms[0].max(0.0))?;
    let m_prime = pq_oetf(lms[1].max(0.0))?;
    let s_prime = pq_oetf(lms[2].max(0.0))?;

    // Step 3: L'M'S' → ICtCp
    let ictcp = mat3_vec_mul(&LMS_PRIME_TO_ICTCP, [l_prime, m_prime, s_prime]);

    Ok((ictcp[0], ictcp[1], ictcp[2]))
}

// ── Inverse conversion: ICtCp → Rec.2020 linear RGB ──────────────────────────

/// Convert an ICtCp triplet back to Rec.2020 linear-light RGB.
///
/// Output is normalised to 1.0 = 10 000 nits.
///
/// # Errors
/// Returns `HdrError::ToneMappingError` if the PQ EOTF fails for any L'M'S' value.
pub fn ictcp_to_rgb(ic: f64, ct: f64, cp: f64) -> Result<(f64, f64, f64)> {
    // Step 1: ICtCp → L'M'S'
    let lms_prime = mat3_vec_mul(&ICTCP_TO_LMS_PRIME, [ic, ct, cp]);

    // Step 2: Per-channel PQ EOTF (L'M'S' → LMS)
    let lms = [
        pq_eotf(lms_prime[0].clamp(0.0, 1.0))?,
        pq_eotf(lms_prime[1].clamp(0.0, 1.0))?,
        pq_eotf(lms_prime[2].clamp(0.0, 1.0))?,
    ];

    // Step 3: LMS → Rec.2020 linear RGB
    let rgb = mat3_vec_mul(&LMS_TO_RGB, lms);

    Ok((rgb[0].max(0.0), rgb[1].max(0.0), rgb[2].max(0.0)))
}

// ── ICtCpFrame ────────────────────────────────────────────────────────────────

/// Per-frame ICtCp conversion utilities.
pub struct ICtCpFrame;

impl ICtCpFrame {
    /// Convert an interleaved Rec.2020 linear-light RGB frame to ICtCp.
    ///
    /// # Arguments
    /// - `pixels`: interleaved linear-light RGB values, normalised to 1.0 = 10 000 nits
    ///   (length must be divisible by 3)
    ///
    /// # Returns
    /// Interleaved `[I, Ct, Cp]` values in the same layout as the input.
    ///
    /// # Errors
    /// Returns `HdrError::ToneMappingError` if the pixel buffer length is not
    /// divisible by 3 or if any channel value is negative.
    pub fn rgb_to_ictcp_frame(pixels: &[f64]) -> Result<Vec<f64>> {
        if !pixels.len().is_multiple_of(3) {
            return Err(HdrError::ToneMappingError(format!(
                "pixel buffer length {} is not divisible by 3",
                pixels.len()
            )));
        }
        let mut out = Vec::with_capacity(pixels.len());
        for chunk in pixels.chunks_exact(3) {
            let (ic, ct, cp) = rgb_to_ictcp(chunk[0], chunk[1], chunk[2])?;
            out.push(ic);
            out.push(ct);
            out.push(cp);
        }
        Ok(out)
    }

    /// Convert an interleaved ICtCp frame back to Rec.2020 linear-light RGB.
    ///
    /// # Errors
    /// Returns `HdrError::ToneMappingError` if the buffer length is not divisible by 3.
    pub fn ictcp_to_rgb_frame(pixels: &[f64]) -> Result<Vec<f64>> {
        if !pixels.len().is_multiple_of(3) {
            return Err(HdrError::ToneMappingError(format!(
                "pixel buffer length {} is not divisible by 3",
                pixels.len()
            )));
        }
        let mut out = Vec::with_capacity(pixels.len());
        for chunk in pixels.chunks_exact(3) {
            let (r, g, b) = ictcp_to_rgb(chunk[0], chunk[1], chunk[2])?;
            out.push(r);
            out.push(g);
            out.push(b);
        }
        Ok(out)
    }

    /// Compute the delta-ICtCp perceptual colour difference between two ICtCp triplets.
    ///
    /// Delta-ICtCp is defined in ITU-R BT.2124 as:
    ///
    ///   ΔICtCp = sqrt(ΔI² + Δ(0.5·Ct)² + ΔCp²)
    ///
    /// A value of 1.0 corresponds to approximately one JND (just-noticeable difference).
    pub fn delta_ictcp(i1: f64, ct1: f64, cp1: f64, i2: f64, ct2: f64, cp2: f64) -> f64 {
        let di = i1 - i2;
        let dct = 0.5 * (ct1 - ct2);
        let dcp = cp1 - cp2;
        (di * di + dct * dct + dcp * dcp).sqrt()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    const EPS: f64 = 1e-5;
    const LOOSE: f64 = 1e-3;

    // 1. Black maps to (0, 0, 0) in ICtCp
    #[test]
    fn test_black_rgb_to_ictcp() {
        let (ic, ct, cp) = rgb_to_ictcp(0.0, 0.0, 0.0).expect("black");
        // Black in linear → PQ OETF of 0 → L'M'S'=(0,0,0) but PQ OETF(0)≠0 in general;
        // the I component will be non-zero due to PQ encoding offset.
        // Ct and Cp (chroma) should both be zero for a neutral colour.
        assert!(approx(ct, 0.0, EPS), "Ct for black: {ct}");
        assert!(approx(cp, 0.0, EPS), "Cp for black: {cp}");
        assert!(ic >= 0.0, "I for black must be non-negative: {ic}");
    }

    // 2. Neutral grey: Ct and Cp are zero
    #[test]
    fn test_neutral_grey_chroma_zero() {
        // A neutral grey at 50% of PQ range (equal RGB)
        let v = 0.005; // ≈ 50 nits / 10000 nits
        let (_, ct, cp) = rgb_to_ictcp(v, v, v).expect("grey");
        assert!(approx(ct, 0.0, EPS), "neutral grey Ct: {ct}");
        assert!(approx(cp, 0.0, EPS), "neutral grey Cp: {cp}");
    }

    // 3. Round-trip: RGB → ICtCp → RGB ≈ identity
    #[test]
    fn test_rgb_ictcp_round_trip() {
        let (r_in, g_in, b_in) = (0.01, 0.005, 0.002); // ≈ 100, 50, 20 nits
        let (ic, ct, cp) = rgb_to_ictcp(r_in, g_in, b_in).expect("forward");
        let (r_out, g_out, b_out) = ictcp_to_rgb(ic, ct, cp).expect("inverse");
        assert!(
            approx(r_in, r_out, LOOSE),
            "R round-trip: {r_in} vs {r_out}"
        );
        assert!(
            approx(g_in, g_out, LOOSE),
            "G round-trip: {g_in} vs {g_out}"
        );
        assert!(
            approx(b_in, b_out, LOOSE),
            "B round-trip: {b_in} vs {b_out}"
        );
    }

    // 4. I component is monotonically increasing with luminance
    #[test]
    fn test_i_monotonic_with_luminance() {
        let mut prev_i = f64::NEG_INFINITY;
        for i in 1..=10 {
            let v = (i as f64) * 0.001; // 0.001 to 0.01
            let (ic, _, _) = rgb_to_ictcp(v, v, v).expect("grey series");
            assert!(ic > prev_i, "I not monotonic at v={v}: {ic} < {prev_i}");
            prev_i = ic;
        }
    }

    // 5. Negative input returns error
    #[test]
    fn test_negative_rgb_error() {
        assert!(rgb_to_ictcp(-0.01, 0.0, 0.0).is_err());
    }

    // 6. Frame conversion: empty buffer returns Ok
    #[test]
    fn test_rgb_to_ictcp_empty_frame() {
        let result = ICtCpFrame::rgb_to_ictcp_frame(&[]).expect("empty");
        assert!(result.is_empty());
    }

    // 7. Frame conversion: invalid length returns error
    #[test]
    fn test_rgb_to_ictcp_invalid_length() {
        assert!(ICtCpFrame::rgb_to_ictcp_frame(&[0.0, 0.0]).is_err());
    }

    // 8. Frame round-trip: RGB → ICtCp → RGB
    #[test]
    fn test_frame_round_trip() {
        let frame = vec![0.01f64, 0.005, 0.002, 0.008, 0.008, 0.008];
        let ictcp = ICtCpFrame::rgb_to_ictcp_frame(&frame).expect("rgb_to_ictcp_frame");
        assert_eq!(ictcp.len(), 6);
        let back = ICtCpFrame::ictcp_to_rgb_frame(&ictcp).expect("ictcp_to_rgb_frame");
        assert_eq!(back.len(), 6);
        for i in 0..6 {
            assert!(
                approx(frame[i], back[i], LOOSE),
                "frame round-trip at [{}]: {} vs {}",
                i,
                frame[i],
                back[i]
            );
        }
    }

    // 9. Delta-ICtCp is zero for identical pixels
    #[test]
    fn test_delta_ictcp_zero_for_same() {
        let (ic, ct, cp) = rgb_to_ictcp(0.005, 0.003, 0.001).expect("pixel");
        let delta = ICtCpFrame::delta_ictcp(ic, ct, cp, ic, ct, cp);
        assert!(approx(delta, 0.0, EPS), "delta for same: {delta}");
    }

    // 10. Delta-ICtCp is positive for different pixels
    #[test]
    fn test_delta_ictcp_positive_for_different() {
        let (i1, ct1, cp1) = rgb_to_ictcp(0.005, 0.003, 0.001).expect("p1");
        let (i2, ct2, cp2) = rgb_to_ictcp(0.008, 0.003, 0.001).expect("p2");
        let delta = ICtCpFrame::delta_ictcp(i1, ct1, cp1, i2, ct2, cp2);
        assert!(delta > 0.0, "delta should be positive: {delta}");
    }

    // 11. White point (equal RGB at 1.0) has Ct=Cp=0
    #[test]
    fn test_white_chroma_zero() {
        let (_, ct, cp) = rgb_to_ictcp(1.0, 1.0, 1.0).expect("white");
        assert!(approx(ct, 0.0, EPS), "white Ct: {ct}");
        assert!(approx(cp, 0.0, EPS), "white Cp: {cp}");
    }

    // 12. ICtCp frame with invalid inverse length returns error
    #[test]
    fn test_ictcp_to_rgb_invalid_length() {
        assert!(ICtCpFrame::ictcp_to_rgb_frame(&[0.5, 0.0]).is_err());
    }

    // 13. Chroma components separate R from G from B
    #[test]
    fn test_red_has_nonzero_chroma() {
        // A red-dominant pixel should have non-zero Cp (red-green axis)
        let (_, _, cp_red) = rgb_to_ictcp(0.01, 0.001, 0.001).expect("red");
        let (_, _, cp_grey) = rgb_to_ictcp(0.01, 0.01, 0.01).expect("grey");
        assert!(
            (cp_red - cp_grey).abs() > 1e-4,
            "Red should have non-zero Cp vs grey: cp_red={cp_red}, cp_grey={cp_grey}"
        );
    }

    // 14. Luminance scaling: doubling brightness increases I
    #[test]
    fn test_luminance_scaling_increases_i() {
        let (i_low, _, _) = rgb_to_ictcp(0.001, 0.001, 0.001).expect("low");
        let (i_high, _, _) = rgb_to_ictcp(0.01, 0.01, 0.01).expect("high");
        assert!(
            i_high > i_low,
            "Higher luminance should have higher I: {i_low} vs {i_high}"
        );
    }
}
