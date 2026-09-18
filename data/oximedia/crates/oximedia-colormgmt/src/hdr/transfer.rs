//! HDR transfer functions: PQ (ST.2084) and HLG (Hybrid Log-Gamma).
//!
//! These implement the EOTF (Electro-Optical Transfer Function) and OETF
//! (Opto-Electronic Transfer Function) for HDR content per ITU-R BT.2100.
//!
//! # PQ (Perceptual Quantizer) – SMPTE ST 2084
//!
//! PQ maps a signal [0, 1] to an absolute luminance of 0–10,000 cd/m².
//! It is used in HDR10 and Dolby Vision.
//!
//! # HLG (Hybrid Log-Gamma) – ITU-R BT.2100
//!
//! HLG uses a combined linear and logarithmic curve. Unlike PQ it is relative
//! (scene-light) rather than absolute, making it backward-compatible with SDR.

#[allow(dead_code)]
/// Peak reference luminance assumed by the PQ standard (cd/m²).
pub const PQ_PEAK_NITS: f64 = 10_000.0;

/// Nominal peak luminance for HDR10 mastering displays (cd/m²).
pub const HDR10_PEAK_NITS: f64 = 1_000.0;

/// Nominal peak luminance for SDR displays (cd/m²).
pub const SDR_PEAK_NITS: f64 = 100.0;

// --------------------------------------------------------------------------
// PQ constants (SMPTE ST 2084)
// --------------------------------------------------------------------------
const PQ_M1: f64 = 2610.0 / 16384.0; // 0.159302
const PQ_M2: f64 = 2523.0 / 4096.0 * 128.0; // 78.8438
const PQ_C1: f64 = 3424.0 / 4096.0; // 0.835938
const PQ_C2: f64 = 2413.0 / 4096.0 * 32.0; // 18.8516
const PQ_C3: f64 = 2392.0 / 4096.0 * 32.0; // 18.6875

// --------------------------------------------------------------------------
// HLG constants (ITU-R BT.2100)
// --------------------------------------------------------------------------
const HLG_A: f64 = 0.178_832_77;
const HLG_B: f64 = 0.284_668_92;
const HLG_C: f64 = 0.559_910_73;

// --------------------------------------------------------------------------
// PQ transfer functions
// --------------------------------------------------------------------------

/// Apply the PQ EOTF: convert normalized PQ signal [0, 1] to linear light.
///
/// Returns linear light normalized so that `1.0 = PQ_PEAK_NITS` (10,000 cd/m²).
///
/// # Arguments
///
/// * `e` – PQ-encoded signal in [0, 1]
///
/// # Returns
///
/// Linear light in [0, 1] where `1.0` represents 10,000 nits.
#[must_use]
pub fn pq_eotf_normalized(e: f64) -> f64 {
    let e = e.clamp(0.0, 1.0);
    let e_m2 = e.powf(1.0 / PQ_M2);
    let num = (e_m2 - PQ_C1).max(0.0);
    let den = PQ_C2 - PQ_C3 * e_m2;
    if den.abs() < 1e-15 {
        return 0.0;
    }
    (num / den).powf(1.0 / PQ_M1)
}

/// Apply the PQ EOTF and convert to absolute nits.
///
/// # Arguments
///
/// * `e` – PQ-encoded signal in [0, 1]
///
/// # Returns
///
/// Absolute luminance in cd/m² (nits), in the range `[0, 10_000]`.
#[must_use]
pub fn pq_eotf_nits(e: f64) -> f64 {
    pq_eotf_normalized(e) * PQ_PEAK_NITS
}

/// Apply the inverse PQ EOTF (OETF): convert normalized linear light to PQ signal.
///
/// # Arguments
///
/// * `y` – Linear light normalized to `[0, 1]` where `1.0 = 10,000 nits`
///
/// # Returns
///
/// PQ-encoded signal in [0, 1].
#[must_use]
pub fn pq_oetf_normalized(y: f64) -> f64 {
    let y = y.clamp(0.0, 1.0);
    let y_m1 = y.powf(PQ_M1);
    let num = PQ_C1 + PQ_C2 * y_m1;
    let den = 1.0 + PQ_C3 * y_m1;
    (num / den).powf(PQ_M2)
}

/// Apply the inverse PQ EOTF from nits to PQ signal.
///
/// # Arguments
///
/// * `nits` – Absolute luminance in cd/m², clamped to `[0, 10_000]`
///
/// # Returns
///
/// PQ-encoded signal in [0, 1].
#[must_use]
pub fn pq_oetf_nits(nits: f64) -> f64 {
    pq_oetf_normalized((nits / PQ_PEAK_NITS).clamp(0.0, 1.0))
}

/// Apply PQ EOTF to an RGB triplet, returning normalized linear values.
///
/// # Arguments
///
/// * `pq` – PQ-encoded RGB in [0, 1]
///
/// # Returns
///
/// Linear RGB normalized so that `1.0 = 10,000 nits`.
#[must_use]
pub fn pq_eotf_rgb(pq: [f64; 3]) -> [f64; 3] {
    [
        pq_eotf_normalized(pq[0]),
        pq_eotf_normalized(pq[1]),
        pq_eotf_normalized(pq[2]),
    ]
}

/// Apply inverse PQ EOTF to a normalized linear RGB triplet.
///
/// # Arguments
///
/// * `linear` – Linear RGB in [0, 1] where `1.0 = 10,000 nits`
///
/// # Returns
///
/// PQ-encoded RGB in [0, 1].
#[must_use]
pub fn pq_oetf_rgb(linear: [f64; 3]) -> [f64; 3] {
    [
        pq_oetf_normalized(linear[0]),
        pq_oetf_normalized(linear[1]),
        pq_oetf_normalized(linear[2]),
    ]
}

// --------------------------------------------------------------------------
// HLG transfer functions (ITU-R BT.2100)
// --------------------------------------------------------------------------

/// Apply the HLG OETF (scene light to HLG signal).
///
/// Converts normalized scene-linear light `[0, 1]` to HLG encoded signal `[0, 1]`.
/// The HLG OETF is used for camera encoding.
///
/// # Arguments
///
/// * `e` – Normalized scene linear light in [0, 1]
///
/// # Returns
///
/// HLG-encoded signal in [0, 1].
#[must_use]
pub fn hlg_oetf(e: f64) -> f64 {
    let e = e.clamp(0.0, 1.0);
    if e <= 1.0 / 12.0 {
        (3.0 * e).sqrt()
    } else {
        HLG_A * (12.0 * e - HLG_B).ln() + HLG_C
    }
}

/// Apply the HLG EOTF (HLG signal to display light).
///
/// Converts HLG encoded signal `[0, 1]` to normalized display-linear light `[0, 1]`.
///
/// # Arguments
///
/// * `e` – HLG-encoded signal in [0, 1]
///
/// # Returns
///
/// Normalized linear display light in [0, 1].
#[must_use]
pub fn hlg_eotf(e: f64) -> f64 {
    let e = e.clamp(0.0, 1.0);
    if e <= 0.5 {
        (e * e) / 3.0
    } else {
        ((e - HLG_C) / HLG_A).exp() / 12.0 + HLG_B / 12.0
    }
}

/// Apply HLG OETF to an RGB triplet.
///
/// # Arguments
///
/// * `linear` – Scene-linear RGB in [0, 1]
///
/// # Returns
///
/// HLG-encoded RGB in [0, 1].
#[must_use]
pub fn hlg_oetf_rgb(linear: [f64; 3]) -> [f64; 3] {
    [hlg_oetf(linear[0]), hlg_oetf(linear[1]), hlg_oetf(linear[2])]
}

/// Apply HLG EOTF to an RGB triplet.
///
/// # Arguments
///
/// * `hlg` – HLG-encoded RGB in [0, 1]
///
/// # Returns
///
/// Normalized linear display RGB in [0, 1].
#[must_use]
pub fn hlg_eotf_rgb(hlg: [f64; 3]) -> [f64; 3] {
    [hlg_eotf(hlg[0]), hlg_eotf(hlg[1]), hlg_eotf(hlg[2])]
}

/// Apply the extended (scene-referred) HLG EOTF with the system gamma OOTF.
///
/// Per ITU-R BT.2100 Table 5, the full HLG OOTF applies a `γ` exponent to luminance.
/// `γ = 1.2 + 0.42 * log10(L_W / 1000)`, where `L_W` is the nominal peak display white.
///
/// # Arguments
///
/// * `hlg` – HLG-encoded RGB in [0, 1]
/// * `peak_display_nits` – Nominal peak luminance of the display in cd/m²
///
/// # Returns
///
/// Linear display-referred RGB in [0, 1].
#[must_use]
pub fn hlg_eotf_with_ootf(hlg: [f64; 3], peak_display_nits: f64) -> [f64; 3] {
    // Step 1: Apply HLG EOTF per channel
    let linear = hlg_eotf_rgb(hlg);

    // Step 2: Compute scene luminance using BT.2100 luminance coefficients
    let luma = 0.2627 * linear[0] + 0.6780 * linear[1] + 0.0593 * linear[2];

    // Step 3: Compute system gamma γ = 1.2 + 0.42 * log10(L_W / 1000)
    let l_w = peak_display_nits.max(1.0);
    let gamma = 1.2 + 0.42 * (l_w / 1000.0).log10();

    // Step 4: Apply OOTF: E_D = α * E_S^(γ-1) * E_S
    // where α = L_W / reference (simplified here to scale only via luma exponent)
    let ootf_scale = if luma > 1e-10 {
        luma.powf(gamma - 1.0)
    } else {
        1.0
    };

    [
        (linear[0] * ootf_scale).clamp(0.0, 1.0),
        (linear[1] * ootf_scale).clamp(0.0, 1.0),
        (linear[2] * ootf_scale).clamp(0.0, 1.0),
    ]
}

// --------------------------------------------------------------------------
// BT.709 transfer functions
// --------------------------------------------------------------------------

/// Apply BT.709 EOTF (gamma to linear).
///
/// # Arguments
///
/// * `e` – BT.709-encoded signal in [0, 1]
///
/// # Returns
///
/// Linear light in [0, 1].
#[must_use]
pub fn bt709_eotf(e: f64) -> f64 {
    const ALPHA: f64 = 1.099_296_826_809_44;
    const BETA: f64 = 0.018_053_968_510_807;
    const GAMMA: f64 = 1.0 / 0.45;

    let e = e.clamp(0.0, 1.0);
    if e < BETA * 4.5 {
        e / 4.5
    } else {
        ((e + (ALPHA - 1.0)) / ALPHA).powf(GAMMA)
    }
}

/// Apply BT.709 OETF (linear to gamma).
///
/// # Arguments
///
/// * `y` – Linear light in [0, 1]
///
/// # Returns
///
/// BT.709-encoded signal in [0, 1].
#[must_use]
pub fn bt709_oetf(y: f64) -> f64 {
    const ALPHA: f64 = 1.099_296_826_809_44;
    const BETA: f64 = 0.018_053_968_510_807;
    const GAMMA: f64 = 0.45;

    let y = y.clamp(0.0, 1.0);
    if y < BETA {
        4.5 * y
    } else {
        ALPHA * y.powf(GAMMA) - (ALPHA - 1.0)
    }
}

/// Apply BT.709 EOTF to an RGB triplet.
#[must_use]
pub fn bt709_eotf_rgb(encoded: [f64; 3]) -> [f64; 3] {
    [bt709_eotf(encoded[0]), bt709_eotf(encoded[1]), bt709_eotf(encoded[2])]
}

/// Apply BT.709 OETF to a linear RGB triplet.
#[must_use]
pub fn bt709_oetf_rgb(linear: [f64; 3]) -> [f64; 3] {
    [bt709_oetf(linear[0]), bt709_oetf(linear[1]), bt709_oetf(linear[2])]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pq_eotf_normalized_clamp() {
        // Signal 0 → 0 nits
        assert!((pq_eotf_normalized(0.0)).abs() < 1e-10);
        // Signal 1 → 1.0 normalized (10,000 nits)
        assert!((pq_eotf_normalized(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_pq_roundtrip() {
        for &v in &[0.001, 0.01, 0.1, 0.5, 0.9] {
            let encoded = pq_oetf_normalized(v);
            let decoded = pq_eotf_normalized(encoded);
            assert!(
                (decoded - v).abs() < 1e-9,
                "PQ roundtrip failed for {v}: got {decoded}"
            );
        }
    }

    #[test]
    fn test_pq_nits_roundtrip() {
        for &nits in &[0.0, 1.0, 100.0, 1000.0, 4000.0, 10000.0] {
            let sig = pq_oetf_nits(nits);
            let back = pq_eotf_nits(sig);
            assert!(
                (back - nits).abs() < 0.1,
                "PQ nits roundtrip failed for {nits}: got {back}"
            );
        }
    }

    #[test]
    fn test_pq_rgb_roundtrip() {
        let orig = [0.2, 0.5, 0.8];
        let enc = pq_oetf_rgb(orig);
        let dec = pq_eotf_rgb(enc);
        for i in 0..3 {
            assert!((dec[i] - orig[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn test_hlg_eotf_oetf_roundtrip() {
        for &v in &[0.001, 0.01, 0.1, 0.3, 0.5, 0.8] {
            let encoded = hlg_oetf(v);
            let decoded = hlg_eotf(encoded);
            assert!(
                (decoded - v).abs() < 1e-9,
                "HLG roundtrip failed for {v}: got {decoded}"
            );
        }
    }

    #[test]
    fn test_hlg_rgb_roundtrip() {
        let orig = [0.2, 0.5, 0.8];
        let enc = hlg_oetf_rgb(orig);
        let dec = hlg_eotf_rgb(enc);
        for i in 0..3 {
            assert!((dec[i] - orig[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn test_hlg_eotf_with_ootf_range() {
        let hlg = [0.5, 0.6, 0.3];
        let out = hlg_eotf_with_ootf(hlg, 1000.0);
        for v in out {
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn test_bt709_roundtrip() {
        for &v in &[0.01, 0.1, 0.5, 0.9] {
            let enc = bt709_oetf(v);
            let dec = bt709_eotf(enc);
            assert!((dec - v).abs() < 1e-9, "BT.709 roundtrip failed for {v}: got {dec}");
        }
    }
}
