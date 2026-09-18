//! SILK LP analysis helpers: autocorrelation, Levinson-Durbin, LPC→NLSF
//! conversion, and NLSF stage-1/stage-2 quantisation (RFC 6716 §4.2.7.5).
//!
//! These functions are internal to the SILK encoder.  They are declared
//! `pub(super)` so that `silk_encoder.rs` and the other sibling modules can
//! reach them without leaking implementation detail outside the `opus` module.

use crate::{CodecError, CodecResult};

use super::silk_decoder::{nlsf_to_lpc, stabilise_nlsf, SilkBandwidth};
use super::silk_tables as t;

// ---------------------------------------------------------------------------
// Public-facing type used by the encoder coordinator
// ---------------------------------------------------------------------------

/// Outcome of the encoder's NLSF analysis + quantisation step.
pub(super) struct NlsfDecision {
    /// Stage-1 codebook index.
    pub i1: usize,
    /// Stage-2 residual indices (decoder values, range `[-NLSF_EXT_MAX, +NLSF_EXT_MAX]`).
    pub res_idx: Vec<i32>,
    /// Reconstructed NLSF (Q15) — what the decoder will see.
    pub nlsf_q15: Vec<i16>,
}

/// Maximum absolute decoder-domain residual value we are willing to encode.
/// The base symbol covers `[-4, 4]`; values past `4` consume an `NLSF_EXT`
/// extension byte each, so we cap the search at the value the geometric tail
/// can comfortably encode (the decoder accepts arbitrarily many `NLSF_EXT`
/// increments, but the iCDF tail probability drops sharply past ±7).
pub(super) const NLSF_RES_MAX: i32 = 10;

// ---------------------------------------------------------------------------
// LP analysis: autocorrelation + Levinson-Durbin
// ---------------------------------------------------------------------------

/// Computes the unnormalised autocorrelation of `x` up to `max_lag`, with
/// a Hann window applied to the input.  The Hann window suppresses spectral
/// leakage and prevents the autocorrelation matrix from becoming rank-deficient
/// on tonal input (which would yield a degenerate Levinson-Durbin LPC).
pub(super) fn autocorrelation(x: &[f32], max_lag: usize) -> Vec<f64> {
    let n = x.len();
    let mut windowed = Vec::with_capacity(n);
    if n > 1 {
        let scale = std::f64::consts::PI / (n as f64);
        for (i, &s) in x.iter().enumerate() {
            let w = 0.5 - 0.5 * (2.0 * scale * (i as f64 + 0.5)).cos();
            windowed.push(f64::from(s) * w);
        }
    } else {
        windowed.extend(x.iter().map(|&s| f64::from(s)));
    }
    let mut r = vec![0.0f64; max_lag + 1];
    for lag in 0..=max_lag {
        let mut s = 0.0f64;
        for i in lag..n {
            s += windowed[i] * windowed[i - lag];
        }
        r[lag] = s;
    }
    // Lag windowing: multiplies r[k] by exp(-(2π*lag_step*k)²/2).
    let lag_step = 1.0e-4f64;
    for (k, slot) in r.iter_mut().enumerate() {
        let arg = 2.0 * std::f64::consts::PI * lag_step * (k as f64);
        *slot *= (-0.5 * arg * arg).exp();
    }
    // White-noise floor so degenerate cases (silence, pure tone) yield a
    // well-conditioned matrix.
    if r[0] > 0.0 {
        r[0] *= 1.000_1;
    } else {
        r[0] = 1e-12;
    }
    r
}

/// Solves the order-`p` Yule-Walker system via Levinson-Durbin recursion.
/// Returns the LPC coefficients `a[0..p]` (with `a[0]` implicitly 1).
/// A bandwidth-expansion factor `chirp = 0.99` is applied on the way out to
/// push the poles slightly inside the unit circle.
pub(super) fn levinson_durbin(r: &[f64], p: usize) -> Vec<f64> {
    if p == 0 {
        return Vec::new();
    }
    let mut a = vec![0.0f64; p];
    let mut prev = vec![0.0f64; p];
    let mut e = if r[0] > 0.0 { r[0] } else { 1e-30 };
    for i in 0..p {
        let mut k = r[i + 1];
        for j in 0..i {
            k -= a[j] * r[i - j];
        }
        let k = if e.abs() > 1e-30 { k / e } else { 0.0 };
        let k = k.clamp(-0.999, 0.999);
        prev[..i].copy_from_slice(&a[..i]);
        for j in 0..i {
            a[j] = prev[j] - k * prev[i - 1 - j];
        }
        a[i] = k;
        e *= 1.0 - k * k;
        if e <= 0.0 {
            e = 1e-30;
        }
    }
    // Bandwidth expansion (chirp): multiplies a[k] by chirp^(k+1).
    let chirp = 0.99f64;
    let mut factor = chirp;
    for ak in a.iter_mut() {
        *ak *= factor;
        factor *= chirp;
    }
    a
}

// ---------------------------------------------------------------------------
// NLSF stage-1 + stage-2 quantisation
// ---------------------------------------------------------------------------

/// Picks the stage-1 NLSF codebook entry whose vector best matches the LPC
/// analysis of `samples` under the weighted-L2 metric.
fn quantise_nlsf_stage1(samples: &[f32], bw: SilkBandwidth) -> CodecResult<usize> {
    let order = bw.lpc_order();
    let r = autocorrelation(samples, order);
    let a = levinson_durbin(&r, order);
    let nlsf_target = lpc_to_nlsf(&a, order);

    let mut best_idx = 0usize;
    let mut best_dist = f64::INFINITY;
    let num_entries = if bw.is_wideband() {
        t::NLSF_CB1_WB_Q8.len()
    } else {
        t::NLSF_CB1_NB_MB_Q8.len()
    };
    for i in 0..num_entries {
        let mut dist = 0.0f64;
        for k in 0..order {
            let (cb_q8, wght_q9) = if bw.is_wideband() {
                (
                    i32::from(t::NLSF_CB1_WB_Q8[i][k]),
                    i32::from(t::NLSF_CB1_WGHT_WB_Q9[i][k]),
                )
            } else {
                (
                    i32::from(t::NLSF_CB1_NB_MB_Q8[i][k]),
                    i32::from(t::NLSF_CB1_WGHT_NB_MB_Q9[i][k]),
                )
            };
            let cb_nlsf = f64::from(cb_q8) / 256.0;
            let diff = nlsf_target[k] - cb_nlsf;
            dist += diff * diff * f64::from(wght_q9);
        }
        if dist < best_dist {
            best_dist = dist;
            best_idx = i;
        }
    }
    Ok(best_idx)
}

/// Runs the encoder NLSF analysis + stage-1 + stage-2 quantisation.  The
/// returned [`NlsfDecision`] holds the stage-1 index, the chosen stage-2
/// residual indices (per coefficient), and the reconstructed Q15 NLSF the
/// decoder will see.
pub(super) fn quantise_nlsf_full(samples: &[f32], bw: SilkBandwidth) -> CodecResult<NlsfDecision> {
    let order = bw.lpc_order();
    let i1 = quantise_nlsf_stage1(samples, bw)?;

    let r = autocorrelation(samples, order);
    let a = levinson_durbin(&r, order);
    let mut nlsf_target_q15 = vec![0i32; order];
    {
        let nlsf_target = lpc_to_nlsf(&a, order);
        for (k, &v) in nlsf_target.iter().enumerate().take(order) {
            nlsf_target_q15[k] = (v * 32768.0).round().clamp(0.0, 32767.0) as i32;
        }
    }

    let (cb_row, wght_row, select_row, pred_row0, pred_row1, _cb2_icdf_set) = if bw.is_wideband() {
        (
            &t::NLSF_CB1_WB_Q8[i1][..],
            &t::NLSF_CB1_WGHT_WB_Q9[i1][..],
            &t::NLSF_CB2_SELECT_WB[i1 * (order / 2)..i1 * (order / 2) + order / 2],
            &t::NLSF_PRED_WB_Q8[0][..],
            &t::NLSF_PRED_WB_Q8[1][..],
            &t::NLSF_CB2_ICDF_WB[..],
        )
    } else {
        (
            &t::NLSF_CB1_NB_MB_Q8[i1][..],
            &t::NLSF_CB1_WGHT_NB_MB_Q9[i1][..],
            &t::NLSF_CB2_SELECT_NB_MB[i1 * (order / 2)..i1 * (order / 2) + order / 2],
            &t::NLSF_PRED_NB_MB_Q8[0][..],
            &t::NLSF_PRED_NB_MB_Q8[1][..],
            &t::NLSF_CB2_ICDF_NB_MB[..],
        )
    };

    let mut pred_q8 = vec![0i32; order];
    for pair in 0..order / 2 {
        let entry = select_row[pair];
        let even = 2 * pair;
        let odd = even + 1;
        if even < order - 1 {
            pred_q8[even] = if entry & 0x01 != 0 {
                i32::from(pred_row1[even])
            } else {
                i32::from(pred_row0[even])
            };
        }
        if odd < order - 1 {
            pred_q8[odd] = if (entry >> 4) & 0x01 != 0 {
                i32::from(pred_row1[odd])
            } else {
                i32::from(pred_row0[odd])
            };
        }
    }

    let qstep = if bw.is_wideband() {
        t::NLSF_QSTEP_WB
    } else {
        t::NLSF_QSTEP_NB_MB
    };

    let mut target_q10 = vec![0i32; order];
    for k in 0..order {
        let cb_q8 = i32::from(cb_row[k]);
        let wght_q9 = i32::from(wght_row[k]);
        let diff_q15 = nlsf_target_q15[k] - (cb_q8 << 7);
        let v_q24 = i64::from(diff_q15) * i64::from(wght_q9);
        target_q10[k] = (v_q24 >> 14) as i32;
    }

    let mut res_idx = vec![0i32; order];
    let mut out_q10 = 0i32;
    let mut residual_q10 = vec![0i32; order];
    for coeff in (0..order).rev() {
        let pred_q10 = (out_q10 * pred_q8[coeff]) >> 8;
        let target_step = target_q10[coeff] - pred_q10;
        let r_est = if target_step > 0 {
            ((i64::from(target_step) * 65536 / i64::from(qstep)) + 102 + 512) >> 10
        } else if target_step < 0 {
            ((i64::from(target_step) * 65536 / i64::from(qstep)) - 102 - 512) >> 10
        } else {
            0
        };
        let mut best_r = 0i32;
        let mut best_err = i64::MAX;
        let mut best_out = pred_q10;
        for cand_off in -2..=2 {
            let cand_r = ((r_est as i32) + cand_off).clamp(-NLSF_RES_MAX, NLSF_RES_MAX);
            let raw = match cand_r.cmp(&0) {
                std::cmp::Ordering::Greater => (cand_r << 10) - 102,
                std::cmp::Ordering::Less => (cand_r << 10) + 102,
                std::cmp::Ordering::Equal => 0,
            };
            let step = ((i64::from(raw) * i64::from(qstep)) >> 16) as i32;
            let cand_out = pred_q10 + step;
            let err = (i64::from(target_q10[coeff]) - i64::from(cand_out)).abs();
            if err < best_err {
                best_err = err;
                best_r = cand_r;
                best_out = cand_out;
            }
        }
        res_idx[coeff] = best_r;
        out_q10 = best_out;
        residual_q10[coeff] = best_out;
    }

    let mut nlsf_q15 = vec![0i32; order];
    for coeff in 0..order {
        let cb_q8 = i32::from(cb_row[coeff]);
        let wght_q9 = i32::from(wght_row[coeff]);
        let add = if wght_q9 != 0 {
            (residual_q10[coeff] << 14) / wght_q9
        } else {
            0
        };
        nlsf_q15[coeff] = (add + (cb_q8 << 7)).clamp(0, 32767);
    }
    let min_spacing: &[i16] = if bw.is_wideband() {
        &t::NLSF_DELTA_MIN_WB_Q15
    } else {
        &t::NLSF_DELTA_MIN_NB_MB_Q15
    };
    stabilise_nlsf(&mut nlsf_q15, min_spacing, order);
    let nlsf_q15_out: Vec<i16> = nlsf_q15.iter().map(|&v| v as i16).collect();

    Ok(NlsfDecision {
        i1,
        res_idx,
        nlsf_q15: nlsf_q15_out,
    })
}

/// Approximates the LPC → NLSF conversion by scanning the unit circle and
/// finding roots of the symmetric and antisymmetric polynomials
/// `P(z), Q(z) = A(z) ± z^-(p+1) A(z^-1)`.  The result is `order`
/// normalised frequencies in `[0, 1)` (each LSP θ_k / π).
pub(super) fn lpc_to_nlsf(a: &[f64], order: usize) -> Vec<f64> {
    if order == 0 {
        return Vec::new();
    }
    let mut a_full = vec![0.0f64; order + 1];
    a_full[0] = 1.0;
    for k in 0..order {
        a_full[k + 1] = -a[k];
    }
    let polylen = order + 2;
    let mut p = vec![0.0f64; polylen];
    let mut q = vec![0.0f64; polylen];
    p[0] = a_full[0];
    p[order + 1] = a_full[0];
    q[0] = a_full[0];
    q[order + 1] = -a_full[0];
    for k in 1..=order {
        p[k] = a_full[k] + a_full[order + 1 - k];
        q[k] = a_full[k] - a_full[order + 1 - k];
    }
    let eval_mag_sq = |poly: &[f64], theta: f64| -> f64 {
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for (k, &c) in poly.iter().enumerate() {
            re += c * (k as f64 * theta).cos();
            im -= c * (k as f64 * theta).sin();
        }
        re * re + im * im
    };
    let pi = std::f64::consts::PI;
    let steps = 4096;
    let mut p_vals = Vec::with_capacity(steps);
    let mut q_vals = Vec::with_capacity(steps);
    for i in 0..steps {
        let theta = pi * (i as f64) / ((steps - 1) as f64);
        p_vals.push((theta, eval_mag_sq(&p, theta)));
        q_vals.push((theta, eval_mag_sq(&q, theta)));
    }
    let mut roots: Vec<f64> = Vec::with_capacity(order + 4);
    let p_peak = p_vals.iter().fold(0.0f64, |a, &(_, v)| a.max(v));
    let q_peak = q_vals.iter().fold(0.0f64, |a, &(_, v)| a.max(v));
    let p_thresh = (p_peak * 1e-2).max(1e-12);
    let q_thresh = (q_peak * 1e-2).max(1e-12);
    for i in 1..(steps - 1) {
        let (theta, v) = p_vals[i];
        if theta <= 1e-9 || theta >= pi - 1e-9 {
            continue;
        }
        if v < p_vals[i - 1].1 && v < p_vals[i + 1].1 && v < p_thresh {
            let v0 = p_vals[i - 1].1;
            let v1 = v;
            let v2 = p_vals[i + 1].1;
            let denom = v0 - 2.0 * v1 + v2;
            let frac = if denom.abs() > 1e-30 {
                0.5 * (v0 - v2) / denom
            } else {
                0.0
            };
            let step = pi / ((steps - 1) as f64);
            roots.push(theta + frac * step);
        }
    }
    for i in 1..(steps - 1) {
        let (theta, v) = q_vals[i];
        if theta <= 1e-9 || theta >= pi - 1e-9 {
            continue;
        }
        if v < q_vals[i - 1].1 && v < q_vals[i + 1].1 && v < q_thresh {
            let v0 = q_vals[i - 1].1;
            let v1 = v;
            let v2 = q_vals[i + 1].1;
            let denom = v0 - 2.0 * v1 + v2;
            let frac = if denom.abs() > 1e-30 {
                0.5 * (v0 - v2) / denom
            } else {
                0.0
            };
            let step = pi / ((steps - 1) as f64);
            roots.push(theta + frac * step);
        }
    }
    roots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    while roots.len() < order {
        let i = roots.len();
        roots.push(pi * ((i + 1) as f64) / (order as f64 + 1.0));
    }
    roots.truncate(order);
    roots.iter().map(|&r| (r / pi).clamp(0.0, 1.0)).collect()
}
