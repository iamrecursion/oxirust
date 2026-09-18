//! Color difference (Delta E) calculations.
//!
//! ## Performance notes
//!
//! `delta_e_2000_batch` operates on `f32` Lab triples and uses:
//!
//! 1. A precomputed 360-entry sin/cos table (`TRIG_TABLE`) that eliminates
//!    per-pair `sin`/`cos` calls for the hue-angle terms in CIEDE2000.
//! 2. Rayon parallel iteration (when the `rayon` feature is enabled) so that
//!    large batches scale linearly with available CPU cores.

use std::sync::OnceLock;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::xyz::Lab;

// ── Precomputed trig table ─────────────────────────────────────────────────────

/// Number of entries in the precomputed sin/cos table.
const DELTA_E_TRIG_TABLE_SIZE: usize = 360;

/// Precomputed sin/cos table for integer degree values 0..360.
///
/// `TRIG_TABLE[i]` = `(sin(i°), cos(i°))` for i in 0..360.
static TRIG_TABLE: OnceLock<Box<[(f32, f32); DELTA_E_TRIG_TABLE_SIZE]>> = OnceLock::new();

/// Return a reference to the global 360-entry sin/cos table, initialising it on
/// the first call.
fn get_trig_table() -> &'static [(f32, f32); DELTA_E_TRIG_TABLE_SIZE] {
    TRIG_TABLE.get_or_init(|| {
        let mut table = Box::new([(0.0f32, 0.0f32); DELTA_E_TRIG_TABLE_SIZE]);
        for i in 0..DELTA_E_TRIG_TABLE_SIZE {
            let angle_rad = (i as f32) * std::f32::consts::PI / 180.0;
            table[i] = (angle_rad.sin(), angle_rad.cos());
        }
        table
    })
}

/// Look up `(sin(angle), cos(angle))` for any `angle_deg` using linear
/// interpolation between adjacent 1-degree table entries.
///
/// Linear interpolation reduces the worst-case error from O(step) ≈ 0.017
/// to O(step²) ≈ 0.00015, which is well within the CIEDE2000 accuracy budget.
#[inline]
fn trig_lookup(angle_deg: f32) -> (f32, f32) {
    let tbl = get_trig_table();
    let n = DELTA_E_TRIG_TABLE_SIZE;

    // Reduce to [0, 360).
    let a = angle_deg.rem_euclid(360.0);
    let lo = (a as usize) % n;
    let hi = (lo + 1) % n; // wraps 359 -> 0
    let frac = a - a.floor();

    let (s0, c0) = tbl[lo];
    let (s1, c1) = tbl[hi];
    (s0 + frac * (s1 - s0), c0 + frac * (c1 - c0))
}

// ── CIEDE2000 f32 kernel (trig-table accelerated) ────────────────────────────

/// Compute the CIEDE2000 G factor from the mean raw chroma `c_bar`.
///
/// `G = 0.5 * (1 - sqrt(c_bar^7 / (c_bar^7 + 25^7)))`
#[inline]
fn de2000_g_factor(c_bar: f32) -> f32 {
    let c7 = c_bar.powi(7);
    0.5 * (1.0 - (c7 / (c7 + 25.0_f32.powi(7))).sqrt())
}

/// Compute h' in degrees \[0, 360\) for the adjusted a' and original b.
#[inline]
fn de2000_h_prime(b: f32, a_prime: f32) -> f32 {
    if b == 0.0 && a_prime == 0.0 {
        0.0
    } else {
        let mut h = b.atan2(a_prime).to_degrees();
        if h < 0.0 {
            h += 360.0;
        }
        h
    }
}

/// Compute the mean hue angle H̄' in degrees, taking wrap-around into account.
#[inline]
fn de2000_h_bar(c1p: f32, c2p: f32, h1p: f32, h2p: f32) -> f32 {
    if c1p * c2p == 0.0 {
        h1p + h2p
    } else if (h1p - h2p).abs() <= 180.0 {
        (h1p + h2p) * 0.5
    } else if h1p + h2p < 360.0 {
        (h1p + h2p + 360.0) * 0.5
    } else {
        (h1p + h2p - 360.0) * 0.5
    }
}

/// Compute the T factor from the mean hue H̄' (in degrees) using the
/// precomputed trig table.
#[inline]
fn de2000_t_factor(h_bar_p: f32) -> f32 {
    let (_, cos_hbar_m30) = trig_lookup(h_bar_p - 30.0);
    let (_, cos_2hbar) = trig_lookup(2.0 * h_bar_p);
    let (_, cos_3hbar_p6) = trig_lookup(3.0 * h_bar_p + 6.0);
    let (_, cos_4hbar_m63) = trig_lookup(4.0 * h_bar_p - 63.0);
    1.0 - 0.17 * cos_hbar_m30 + 0.24 * cos_2hbar + 0.32 * cos_3hbar_p6 - 0.20 * cos_4hbar_m63
}

/// Full CIEDE2000 implementation operating in `f32` with trig-table
/// acceleration for the hue angle terms.
///
/// The formula follows Sharma, Wu & Dalal (2005) with the same structure as
/// the `f64` reference in `lib.rs`.  Accuracy relative to the `f64` reference
/// is typically < 0.01 DeltaE units (dominated by f32 precision, not the trig
/// table).
#[must_use]
fn delta_e_2000_f32(l1: f32, a1: f32, b1: f32, l2: f32, a2: f32, b2: f32) -> f32 {
    let c1 = (a1 * a1 + b1 * b1).sqrt();
    let c2 = (a2 * a2 + b2 * b2).sqrt();
    let g = de2000_g_factor((c1 + c2) * 0.5);

    // Adjusted a', C'
    let a1p = a1 * (1.0 + g);
    let a2p = a2 * (1.0 + g);
    let c1p = (a1p * a1p + b1 * b1).sqrt();
    let c2p = (a2p * a2p + b2 * b2).sqrt();

    // h' (in degrees)
    let h1p = de2000_h_prime(b1, a1p);
    let h2p = de2000_h_prime(b2, a2p);

    // DeltaL', DeltaC', Deltah'
    let delta_lp = l2 - l1;
    let delta_cp = c2p - c1p;
    let delta_hp = if c1p * c2p == 0.0 {
        0.0
    } else {
        let raw = h2p - h1p;
        if raw.abs() <= 180.0 {
            raw
        } else if raw > 180.0 {
            raw - 360.0
        } else {
            raw + 360.0
        }
    };

    // DeltaH' (uses trig table for the half-angle sin)
    let (sin_dhp_half, _) = trig_lookup(delta_hp * 0.5);
    let delta_big_hp = 2.0 * (c1p * c2p).sqrt() * sin_dhp_half;

    // L-bar', C-bar', H-bar'
    let l_bar = (l1 + l2) * 0.5;
    let c_bar_p = (c1p + c2p) * 0.5;
    let h_bar_p = de2000_h_bar(c1p, c2p, h1p, h2p);

    // T and weighting functions
    let t = de2000_t_factor(h_bar_p);
    let l50 = l_bar - 50.0;
    let s_l = 1.0 + (0.015 * l50 * l50) / (20.0 + l50 * l50).sqrt();
    let s_c = 1.0 + 0.045 * c_bar_p;
    let s_h = 1.0 + 0.015 * c_bar_p * t;

    // R_T (uses trig table for the rotation-term sin)
    let delta_theta = 30.0 * (-(((h_bar_p - 275.0) / 25.0).powi(2))).exp();
    let c7p = c_bar_p.powi(7);
    let r_c = 2.0 * (c7p / (c7p + 25.0_f32.powi(7))).sqrt();
    let (sin_2dt, _) = trig_lookup(2.0 * delta_theta);
    let r_t = -r_c * sin_2dt;

    // Final DeltaE
    let tl = delta_lp / s_l;
    let tc = delta_cp / s_c;
    let th = delta_big_hp / s_h;
    (tl * tl + tc * tc + th * th + r_t * tc * th).sqrt()
}

// ── Public batch API ──────────────────────────────────────────────────────────

/// Compute ΔE₂₀₀₀ for N corresponding pairs simultaneously.
///
/// This function accepts cache-friendly `[L*, a*, b*]` arrays and returns a
/// `Vec<f32>` of CIEDE2000 values.  When the `rayon` feature is enabled the
/// pairs are evaluated in parallel across all available CPU cores.
///
/// # Arguments
///
/// * `reference` - Slice of reference Lab colors as `[L, a, b]` triples.
/// * `sample`    - Slice of sample Lab colors (must be the same length).
///
/// # Panics
///
/// Panics if `reference.len() != sample.len()`.
#[must_use]
pub fn delta_e_2000_batch(reference: &[[f32; 3]], sample: &[[f32; 3]]) -> Vec<f32> {
    assert_eq!(
        reference.len(),
        sample.len(),
        "reference and sample slices must have the same length"
    );

    #[cfg(feature = "rayon")]
    {
        reference
            .par_iter()
            .zip(sample.par_iter())
            .map(|(r, s)| delta_e_2000_f32(r[0], r[1], r[2], s[0], s[1], s[2]))
            .collect()
    }

    #[cfg(not(feature = "rayon"))]
    {
        reference
            .iter()
            .zip(sample.iter())
            .map(|(r, s)| delta_e_2000_f32(r[0], r[1], r[2], s[0], s[1], s[2]))
            .collect()
    }
}

/// Calculates CIE Delta E 1976 (CIE76).
///
/// Simple Euclidean distance in Lab* space.
///
/// # Arguments
///
/// * `lab1` - First color
/// * `lab2` - Second color
///
/// # Returns
///
/// Delta E value. < 1 imperceptible, 1-2 just noticeable, 2-10 noticeable, > 10 very different.
#[must_use]
pub fn cie76(lab1: &Lab, lab2: &Lab) -> f64 {
    let dl = lab1.l - lab2.l;
    let da = lab1.a - lab2.a;
    let db = lab1.b - lab2.b;
    (dl * dl + da * da + db * db).sqrt()
}

/// Calculates CIE Delta E 1994 (CIE94).
///
/// Improved formula accounting for perceptual non-uniformity.
///
/// # Arguments
///
/// * `lab1` - First color (reference)
/// * `lab2` - Second color (sample)
/// * `kl` - Weighting factor for lightness (typically 1.0)
/// * `kc` - Weighting factor for chroma (typically 1.0)
/// * `kh` - Weighting factor for hue (typically 1.0)
#[must_use]
pub fn cie94(lab1: &Lab, lab2: &Lab, kl: f64, kc: f64, kh: f64) -> f64 {
    let k1 = 0.045;
    let k2 = 0.015;

    let dl = lab1.l - lab2.l;
    let da = lab1.a - lab2.a;
    let db = lab1.b - lab2.b;

    let c1 = (lab1.a * lab1.a + lab1.b * lab1.b).sqrt();
    let c2 = (lab2.a * lab2.a + lab2.b * lab2.b).sqrt();
    let dc = c1 - c2;

    let dh_sq = da * da + db * db - dc * dc;
    let dh = if dh_sq > 0.0 { dh_sq.sqrt() } else { 0.0 };

    let sl = 1.0;
    let sc = 1.0 + k1 * c1;
    let sh = 1.0 + k2 * c1;

    let term_l = dl / (kl * sl);
    let term_c = dc / (kc * sc);
    let term_h = dh / (kh * sh);

    (term_l * term_l + term_c * term_c + term_h * term_h).sqrt()
}

/// Calculates CIE Delta E 2000 (CIEDE2000).
///
/// Most sophisticated formula, industry standard for color difference.
///
/// # Arguments
///
/// * `lab1` - First color
/// * `lab2` - Second color
#[must_use]
pub fn cie2000(lab1: &Lab, lab2: &Lab) -> f64 {
    crate::delta_e::delta_e_2000(lab1, lab2)
}

/// Calculates CMC l:c Delta E.
///
/// Used in textile industry.
///
/// # Arguments
///
/// * `lab1` - First color (reference)
/// * `lab2` - Second color (sample)
/// * `l` - Lightness factor (typically 2.0 for acceptability, 1.0 for perceptibility)
/// * `c` - Chroma factor (typically 1.0)
#[must_use]
pub fn cmc(lab1: &Lab, lab2: &Lab, l: f64, c: f64) -> f64 {
    let dl = lab1.l - lab2.l;
    let da = lab1.a - lab2.a;
    let db = lab1.b - lab2.b;

    let c1 = (lab1.a * lab1.a + lab1.b * lab1.b).sqrt();
    let c2 = (lab2.a * lab2.a + lab2.b * lab2.b).sqrt();
    let dc = c1 - c2;

    let dh_sq = da * da + db * db - dc * dc;
    let dh = if dh_sq > 0.0 { dh_sq.sqrt() } else { 0.0 };

    let h1 = lab1.b.atan2(lab1.a).to_degrees();
    let h1 = if h1 < 0.0 { h1 + 360.0 } else { h1 };

    let sl = if lab1.l < 16.0 {
        0.511
    } else {
        (0.040975 * lab1.l) / (1.0 + 0.01765 * lab1.l)
    };

    let sc = (0.0638 * c1) / (1.0 + 0.0131 * c1) + 0.638;

    let f = (c1.powi(4)) / ((c1.powi(4)) + 1900.0).sqrt();

    let t = if (164.0..=345.0).contains(&h1) {
        0.56 + (0.2 * ((h1 + 168.0) * std::f64::consts::PI / 180.0).cos()).abs()
    } else {
        0.36 + (0.4 * ((h1 + 35.0) * std::f64::consts::PI / 180.0).cos()).abs()
    };

    let sh = sc * (f * t + 1.0 - f);

    ((dl / (l * sl)).powi(2) + (dc / (c * sc)).powi(2) + (dh / sh).powi(2)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cie76_same_color() {
        let lab = Lab::new(50.0, 10.0, 20.0);
        assert_eq!(cie76(&lab, &lab), 0.0);
    }

    #[test]
    fn test_cie76_different_colors() {
        let lab1 = Lab::new(50.0, 10.0, 20.0);
        let lab2 = Lab::new(60.0, 10.0, 20.0);
        assert!((cie76(&lab1, &lab2) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_cie94() {
        let lab1 = Lab::new(50.0, 10.0, 20.0);
        let lab2 = Lab::new(50.0, 10.0, 20.0);
        assert_eq!(cie94(&lab1, &lab2, 1.0, 1.0, 1.0), 0.0);
    }

    #[test]
    fn test_cmc() {
        let lab1 = Lab::new(50.0, 10.0, 20.0);
        let lab2 = Lab::new(50.0, 10.0, 20.0);
        assert_eq!(cmc(&lab1, &lab2, 2.0, 1.0), 0.0);
    }

    // ── Trig-table tests ──────────────────────────────────────────────────────

    #[test]
    fn test_delta_e_2000_trig_accuracy() {
        // Compare the interpolated trig table lookup against exact f32 sin/cos
        // for 100 uniformly-spaced angles in [0, 360).  Linear interpolation
        // reduces worst-case error from O(step) to O(step^2) ≈ 0.0002, so the
        // threshold is set to 0.001.
        for i in 0u32..100 {
            let angle_deg = i as f32 * 3.6; // step = 3.6 deg -> covers full circle
            let (sin_tbl, cos_tbl) = trig_lookup(angle_deg);
            let angle_rad = angle_deg * std::f32::consts::PI / 180.0;
            let sin_exact = angle_rad.sin();
            let cos_exact = angle_rad.cos();
            let sin_err = (sin_tbl - sin_exact).abs();
            let cos_err = (cos_tbl - cos_exact).abs();
            assert!(
                sin_err < 0.001,
                "sin error too large at {angle_deg} deg: table={sin_tbl}, exact={sin_exact}, err={sin_err}"
            );
            assert!(
                cos_err < 0.001,
                "cos error too large at {angle_deg} deg: table={cos_tbl}, exact={cos_exact}, err={cos_err}"
            );
        }
    }

    #[test]
    fn test_delta_e_2000_batch_consistency() {
        // Generate 100 Lab pairs deterministically and verify that the
        // trig-table-based f32 kernel produces values near the f64 reference.
        // With linear interpolation, trig error is O(step^2) ≈ 0.0002, so the
        // dominant error source is f32 vs f64 precision; threshold is 0.05.
        use crate::delta_e::delta_e_2000 as de2000_ref;

        for i in 0u32..100 {
            let fi = i as f64;
            let r_arr = [
                (fi * 0.97 % 100.0) as f32,
                ((fi * 1.31 % 128.0) - 64.0) as f32,
                ((fi * 0.73 % 128.0) - 64.0) as f32,
            ];
            let s_arr = [
                ((fi * 1.07 + 5.0) % 100.0) as f32,
                ((fi * 0.61 % 128.0) - 64.0) as f32,
                ((fi * 1.19 % 128.0) - 64.0) as f32,
            ];

            let de_f32 =
                delta_e_2000_f32(r_arr[0], r_arr[1], r_arr[2], s_arr[0], s_arr[1], s_arr[2]);
            let de_ref = de2000_ref(
                &Lab::new(r_arr[0] as f64, r_arr[1] as f64, r_arr[2] as f64),
                &Lab::new(s_arr[0] as f64, s_arr[1] as f64, s_arr[2] as f64),
            ) as f32;
            let err = (de_f32 - de_ref).abs();
            assert!(
                err < 0.05,
                "f32 kernel vs f64 ref deviation too large at i={i}: f32={de_f32}, ref={de_ref}, err={err}"
            );
        }
    }

    // ── Batch API tests ───────────────────────────────────────────────────────

    #[test]
    fn test_delta_e_2000_batch_matches_scalar() {
        // Verify that delta_e_2000_batch returns the same values as calling
        // delta_e_2000_f32 element-wise for 50 pairs.
        let refs: Vec<[f32; 3]> = (0u32..50)
            .map(|i| {
                let fi = i as f32;
                [
                    fi * 2.0 % 100.0,
                    (fi * 1.5 % 128.0) - 64.0,
                    (fi * 0.9 % 128.0) - 64.0,
                ]
            })
            .collect();

        let samps: Vec<[f32; 3]> = (0u32..50)
            .map(|i| {
                let fi = i as f32;
                [
                    (fi * 1.3 + 3.0) % 100.0,
                    ((fi * 0.7 + 10.0) % 128.0) - 64.0,
                    ((fi * 1.1 + 5.0) % 128.0) - 64.0,
                ]
            })
            .collect();

        let batch_result = delta_e_2000_batch(&refs, &samps);

        assert_eq!(batch_result.len(), 50);

        for (idx, ((r, s), &batch_val)) in refs
            .iter()
            .zip(samps.iter())
            .zip(batch_result.iter())
            .enumerate()
        {
            let scalar_val = delta_e_2000_f32(r[0], r[1], r[2], s[0], s[1], s[2]);
            assert!(
                (batch_val - scalar_val).abs() < 1e-6,
                "batch[{idx}]={batch_val} != scalar={scalar_val}"
            );
        }
    }

    #[test]
    fn test_delta_e_2000_batch_same_color_is_zero() {
        // Identical colors must yield ΔE = 0.
        let color = [50.0_f32, 20.0, -30.0];
        let refs = vec![color; 10];
        let samps = vec![color; 10];
        let result = delta_e_2000_batch(&refs, &samps);
        for (i, v) in result.iter().enumerate() {
            assert!(
                v.abs() < 1e-5,
                "ΔE for identical colors[{i}] must be 0, got {v}"
            );
        }
    }
}
