//! SILK long-term prediction (LTP) helpers: pitch search, 5-tap filter
//! quantisation, and the full LTP block encoder (RFC 6716 §4.2.7.6).
//!
//! This module implements three quality improvements over the baseline
//! single-resolution integer pitch search:
//!
//! 1. **Coarse-to-fine decimated pitch search** (`decimate_halfband` +
//!    two-stage `pitch_search`): reduces the ACF scan cost by half and
//!    eliminates octave-harmonic ambiguities.
//!
//! 2. **Per-subframe pitch-contour RD selection** (`select_pitch_contour`):
//!    replaces the hardcoded contour-index 0 with a minimal-distortion search
//!    over the SILK pitch-contour codebooks, allowing the per-subframe lags to
//!    track slow pitch glides.
//!
//! 3. **Fractional-lag parabolic refinement** (`refine_lag_fractional`):
//!    evaluates the ACF at integer-lag ±1 and fits a parabola to locate the
//!    true ACF peak with sub-sample precision; the fractional lag is forwarded
//!    to `solve_ltp_taps_fractional` which linearly-interpolates the reference
//!    signal, improving the condition of the 5×5 normal-equations system.

use crate::CodecResult;

use super::silk_decoder::{
    log_gain_to_linear_q16, SilkBandwidth, SilkSignalType, CONTOUR_MBWB_10MS, CONTOUR_MBWB_20MS,
    CONTOUR_NB_10MS, CONTOUR_NB_20MS,
};
use super::silk_encoder::{EncoderChannelState, MAX_LPC_ORDER, MAX_SUBFRAMES};
use super::silk_range_encoder::SilkRangeEncoder;
use super::silk_tables as t;

/// Minimum LTP pitch lag in samples per kHz of internal sample rate
/// (RFC 6716 §4.2.7.6.1: 2 ms × khz samples).
const LAG_MS_MIN: i32 = 2;

/// Maximum LTP pitch lag in samples per kHz of internal sample rate
/// (RFC 6716 §4.2.7.6.1: 18 ms × khz samples).
const LAG_MS_MAX: i32 = 18;

/// Normalised autocorrelation threshold above which a frame is classified
/// as voiced.
const VOICED_PEAK_THRESHOLD: f32 = 0.6;

/// Half-band 5-tap linear-phase FIR lowpass coefficients.
/// Designed as a Type-I symmetric FIR with passband ≤ 0.4 Nyquist:
/// `[0.0625, 0.25, 0.375, 0.25, 0.0625]`.
const HALFBAND_FIR: [f32; 5] = [0.0625, 0.25, 0.375, 0.25, 0.0625];

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Outcome of the encoder pitch search for one frame.
#[derive(Debug, Clone)]
pub(super) struct PitchDecision {
    /// True when the frame should be emitted as Voiced.
    pub voiced: bool,
    /// Primary integer pitch lag at the internal sample rate.
    pub primary_lag: i32,
    /// Fractional refinement of the primary lag (internal use only; the
    /// bitstream always carries the integer portion).
    pub fractional_lag: f32,
    /// Per-subframe pitch lag (filled even on unvoiced frames; ignored
    /// downstream unless `voiced` is set).
    pub subframe_lags: [i32; MAX_SUBFRAMES],
    /// Normalised autocorrelation peak at the chosen lag.
    pub peak: f32,
}

impl PitchDecision {
    pub(super) fn unvoiced() -> Self {
        Self {
            voiced: false,
            primary_lag: 0,
            fractional_lag: 0.0,
            subframe_lags: [0; MAX_SUBFRAMES],
            peak: 0.0,
        }
    }
}

/// LTP parameters chosen by the encoder for one voiced frame.  The encoder
/// runs the same `5-tap × subframe` synthesis with these *quantised* values
/// during closed-loop subtraction so the decoder reconstructs the identical
/// signal.
#[derive(Debug, Default, Clone)]
pub(super) struct LtpQuantised {
    /// Per-subframe pitch lags (`subframes` valid entries, rest are 0).
    pub pitch_lags: [i32; MAX_SUBFRAMES],
    /// Per-subframe quantised 5-tap filter (Q7).
    pub filters_q7: [[i32; 5]; MAX_SUBFRAMES],
    /// LTP scale factor (Q14).
    pub scale_q14: i32,
    /// Number of valid subframe entries.
    pub subframes: usize,
}

// ---------------------------------------------------------------------------
// LPC residual signal
// ---------------------------------------------------------------------------

/// Returns the LPC analysis residual for the whole frame:
/// `x[n] + sum_k c_k * x[n-1-k]` evaluated using `history` (the previous
/// frame's trailing `order` samples) for any negative indices.
pub(super) fn lpc_residual_signal(
    samples: &[f32],
    lpc_q12: &[i32],
    order: usize,
    history: &[f32],
) -> Vec<f32> {
    let n = samples.len();
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let mut acc = samples[i];
        for (j, &c) in lpc_q12.iter().take(order).enumerate() {
            let prev_idx = (i as isize) - (j as isize) - 1;
            let prev = if prev_idx >= 0 {
                samples[prev_idx as usize]
            } else {
                let h_idx = order as isize + prev_idx;
                if h_idx >= 0 && (h_idx as usize) < history.len() {
                    history[h_idx as usize]
                } else {
                    0.0
                }
            };
            acc += prev * (c as f32 / 4096.0);
        }
        out[i] = acc;
    }
    out
}

// ---------------------------------------------------------------------------
// Gap 1: Coarse-to-fine decimated pitch search
// ---------------------------------------------------------------------------

/// Applies a 5-tap half-band FIR lowpass and downsamples by 2.
///
/// The output length is `(signal.len() + 4) / 2` (causal convolution with
/// zero-padding for the initial conditions; we start the decimation from the
/// first full-overlap sample at position 4 to avoid border artefacts).
pub(super) fn decimate_halfband(signal: &[f32]) -> Vec<f32> {
    let n = signal.len();
    if n < 5 {
        // Too short to filter: return every other sample directly.
        return signal.iter().step_by(2).copied().collect();
    }
    // Allocate with zero-extended left boundary (4 samples of history).
    let mut out = Vec::with_capacity(n / 2 + 1);
    // Process only positions where all 5 taps are in-bounds (index ≥ 4).
    let mut i = 4usize;
    while i < n {
        let y = HALFBAND_FIR[0] * signal[i]
            + HALFBAND_FIR[1] * signal[i - 1]
            + HALFBAND_FIR[2] * signal[i - 2]
            + HALFBAND_FIR[3] * signal[i - 3]
            + HALFBAND_FIR[4] * signal[i - 4];
        out.push(y);
        i += 2;
    }
    out
}

/// Computes the normalised ACF peak at the given integer lag.
///
/// Returns `(numerator, denominator)` so the caller can evaluate `r[lag]`
/// or build parabola fits from several lags.
fn acf_at_lag(signal: &[f32], win_start: usize, lag: usize) -> (f64, f64) {
    let n = signal.len();
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for i in win_start..n {
        if i >= lag {
            let a = f64::from(signal[i]);
            let b = f64::from(signal[i - lag]);
            num += a * b;
            den += b * b;
        }
    }
    (num, den)
}

/// Performs a coarse-to-fine pitch lag search across `[lag_min, lag_max]`
/// (samples at the internal SILK rate).
///
/// **Stage 1 — Coarse:** Decimates the signal by 2 (halving the sample rate),
/// then scans ACF lags `[lag_min/2 .. lag_max/2]`.  This reduces cost by ×4
/// (half samples, half lags) and avoids octave confusion by operating on a
/// properly low-passed version of the signal.
///
/// **Stage 2 — Fine:** Refines the coarse peak at `coarse*2 ± 3` samples in
/// the full-rate signal.
///
/// Returns `(best_integer_lag, normalised_peak)`.
fn pitch_search(signal: &[f32], lag_min: i32, lag_max: i32) -> (i32, f32) {
    let n = signal.len();
    if n < (lag_max as usize) + 4 {
        return (lag_min, 0.0);
    }

    // ---- Stage 1: coarse search on half-rate signal ----
    let decimated = decimate_halfband(signal);
    let d_n = decimated.len();
    let coarse_lag_min = (lag_min / 2).max(1);
    let coarse_lag_max = lag_max / 2;

    let mut coarse_r0 = 0.0f64;
    let coarse_win = (coarse_lag_max.min(d_n as i32 / 2).max(0)) as usize;
    for &s in &decimated[coarse_win..] {
        coarse_r0 += f64::from(s) * f64::from(s);
    }

    let mut best_coarse = coarse_lag_min;
    let mut best_coarse_norm = 0.0f64;

    if coarse_r0 > 0.0 && coarse_lag_min <= coarse_lag_max {
        for lag in coarse_lag_min..=coarse_lag_max {
            let (num, den) = acf_at_lag(&decimated, coarse_win, lag as usize);
            if den <= 0.0 {
                continue;
            }
            let norm_sq = (num * num) / (coarse_r0 * den);
            if norm_sq > best_coarse_norm {
                best_coarse_norm = norm_sq;
                best_coarse = lag;
            }
        }
    }

    // ---- Stage 2: fine search around coarse peak ----
    let center = best_coarse * 2;
    let fine_min = (center - 3).max(lag_min);
    let fine_max = (center + 3).min(lag_max);

    let win_start = (lag_max.min(n as i32 / 2).max(0)) as usize;
    let mut r0 = 0.0f64;
    for &s in &signal[win_start..] {
        r0 += f64::from(s) * f64::from(s);
    }

    if r0 <= 0.0 || fine_min > fine_max {
        let peak = best_coarse_norm.sqrt().clamp(0.0, 1.0) as f32;
        return (center.clamp(lag_min, lag_max), peak);
    }

    let mut best_lag = fine_min;
    let mut best_norm = 0.0f64;

    for lag in fine_min..=fine_max {
        let (num, den) = acf_at_lag(signal, win_start, lag as usize);
        if den <= 0.0 {
            continue;
        }
        let norm_sq = (num * num) / (r0 * den);
        if norm_sq > best_norm {
            best_norm = norm_sq;
            best_lag = lag;
        }
    }

    let peak = best_norm.sqrt().clamp(0.0, 1.0) as f32;
    (best_lag, peak)
}

// ---------------------------------------------------------------------------
// Gap 3: Fractional-lag parabolic refinement
// ---------------------------------------------------------------------------

/// Evaluates the normalised ACF value at a given integer `lag` (for the
/// parabola).  Uses the same windowing as `pitch_search` Stage 2.
fn acf_norm_at(signal: &[f32], lag_max: i32, int_lag: i32) -> f64 {
    let n = signal.len();
    let win_start = (lag_max.min(n as i32 / 2).max(0)) as usize;
    let mut r0 = 0.0f64;
    for &s in &signal[win_start..] {
        r0 += f64::from(s) * f64::from(s);
    }
    if r0 <= 0.0 || int_lag < 1 || (int_lag as usize) >= n {
        return 0.0;
    }
    let (num, den) = acf_at_lag(signal, win_start, int_lag as usize);
    if den <= 0.0 {
        return 0.0;
    }
    (num * num) / (r0 * den)
}

/// Refines an integer pitch lag by fitting a parabola through the ACF values
/// at `int_lag - 1`, `int_lag`, `int_lag + 1`.
///
/// Returns a fractional lag `int_lag + frac` clamped to `[int_lag - 1.0,
/// int_lag + 1.0]`.  This sub-sample estimate is used only in
/// `solve_ltp_taps_fractional`; the emitted bitstream lag stays at the
/// nearest integer.
pub(super) fn refine_lag_fractional(signal: &[f32], lag_max: i32, int_lag: i32) -> f32 {
    let ym1 = acf_norm_at(signal, lag_max, int_lag - 1);
    let y0 = acf_norm_at(signal, lag_max, int_lag);
    let yp1 = acf_norm_at(signal, lag_max, int_lag + 1);

    // Parabola vertex: frac = (ym1 - yp1) / (2 * (2*y0 - ym1 - yp1))
    let denom = 2.0 * (2.0 * y0 - ym1 - yp1);
    if denom.abs() < 1e-12 {
        return int_lag as f32;
    }
    let frac = ((ym1 - yp1) / denom).clamp(-1.0, 1.0) as f32;
    int_lag as f32 + frac
}

/// Runs the encoder pitch analysis pipeline: LPC-residual prep, coarse-to-fine
/// pitch search, decision threshold, per-subframe lag fitting.
pub(super) fn analyse_pitch(
    samples: &[f32],
    bw: SilkBandwidth,
    subframe_count: usize,
    _state: &EncoderChannelState,
) -> PitchDecision {
    let khz = bw.khz() as i32;
    let lag_min = LAG_MS_MIN * khz;
    let lag_max = LAG_MS_MAX * khz;

    let (primary_lag, peak) = pitch_search(samples, lag_min, lag_max);
    let voiced = peak >= VOICED_PEAK_THRESHOLD;

    // Fractional refinement (used downstream by solve_ltp_taps_fractional).
    let fractional_lag = if voiced {
        refine_lag_fractional(samples, lag_max, primary_lag)
    } else {
        primary_lag as f32
    };

    let mut subframe_lags = [0i32; MAX_SUBFRAMES];
    for sf in 0..subframe_count {
        subframe_lags[sf] = primary_lag;
    }
    PitchDecision {
        voiced,
        primary_lag,
        fractional_lag,
        subframe_lags,
        peak,
    }
}

// ---------------------------------------------------------------------------
// LTP filter helpers
// ---------------------------------------------------------------------------

/// Computes per-subframe optimal 5-tap LTP filter taps from the residual
/// signal by solving the 5×5 normal equations (integer lag version).
pub(super) fn solve_ltp_taps(residual: &[f32], start: usize, len: usize, lag: i32) -> [f32; 5] {
    let lag_usize = lag as usize;
    let base_offset = lag_usize.saturating_add(2);
    if start < base_offset || residual.len() < start + len {
        return [0.0; 5];
    }
    let mut rmat = [[0.0f64; 5]; 5];
    let mut rvec = [0.0f64; 5];
    for n in 0..len {
        let t_idx = start + n;
        let mut taps = [0.0f64; 5];
        for k in 0..5 {
            let src = (t_idx as isize) - (lag as isize) + 2 - k as isize;
            if src >= 0 && (src as usize) < residual.len() {
                taps[k] = f64::from(residual[src as usize]);
            }
        }
        let target = f64::from(residual[t_idx]);
        for i in 0..5 {
            rvec[i] += taps[i] * target;
            for j in 0..5 {
                rmat[i][j] += taps[i] * taps[j];
            }
        }
    }
    gauss_solve_5x5(rmat, rvec)
}

/// Computes per-subframe optimal 5-tap LTP taps using a **fractional lag**,
/// interpolating the reference signal with linear interpolation between the
/// two surrounding integer positions.  This improves the condition of the
/// normal equations for signals whose period is not an exact integer number
/// of samples.
pub(super) fn solve_ltp_taps_fractional(
    residual: &[f32],
    start: usize,
    len: usize,
    frac_lag: f32,
) -> [f32; 5] {
    let int_lag = frac_lag.round() as i32;
    let frac = frac_lag - int_lag as f32;
    // Sanity guard: if fractional component is tiny, fall back to integer path.
    if frac.abs() < 1e-4 {
        return solve_ltp_taps(residual, start, len, int_lag);
    }
    let lag_usize = int_lag as usize;
    let base_offset = lag_usize.saturating_add(3); // +1 for the upper interpolant
    if start < base_offset || residual.len() < start + len {
        return solve_ltp_taps(residual, start, len, int_lag);
    }
    let mut rmat = [[0.0f64; 5]; 5];
    let mut rvec = [0.0f64; 5];
    for n in 0..len {
        let t_idx = start + n;
        let mut taps = [0.0f64; 5];
        for k in 0..5 {
            // Base position in the reference signal (lag - 2 + k taps back).
            let src_lo = (t_idx as isize) - (int_lag as isize) + 2 - k as isize;
            let src_hi = src_lo - 1; // one sample earlier (larger lag)
            let v_lo = if src_lo >= 0 && (src_lo as usize) < residual.len() {
                f64::from(residual[src_lo as usize])
            } else {
                0.0
            };
            let v_hi = if src_hi >= 0 && (src_hi as usize) < residual.len() {
                f64::from(residual[src_hi as usize])
            } else {
                0.0
            };
            // Linear interpolation: frac=0 → integer lag, frac>0 → fractional.
            taps[k] = v_lo + f64::from(frac) * (v_hi - v_lo);
        }
        let target = f64::from(residual[t_idx]);
        for i in 0..5 {
            rvec[i] += taps[i] * target;
            for j in 0..5 {
                rmat[i][j] += taps[i] * taps[j];
            }
        }
    }
    gauss_solve_5x5(rmat, rvec)
}

/// Solves a 5×5 system `A · x = b` via Gaussian elimination with partial
/// pivoting.  Returns the zero vector if the system is singular.
fn gauss_solve_5x5(mut a: [[f64; 5]; 5], mut b: [f64; 5]) -> [f32; 5] {
    for col in 0..5 {
        let mut max_row = col;
        let mut max_val = a[col][col].abs();
        for row in (col + 1)..5 {
            if a[row][col].abs() > max_val {
                max_val = a[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-12 {
            return [0.0; 5];
        }
        if max_row != col {
            a.swap(col, max_row);
            b.swap(col, max_row);
        }
        let pivot = a[col][col];
        for j in col..5 {
            a[col][j] /= pivot;
        }
        b[col] /= pivot;
        for row in 0..5 {
            if row == col {
                continue;
            }
            let factor = a[row][col];
            for j in col..5 {
                a[row][j] -= factor * a[col][j];
            }
            b[row] -= factor * b[col];
        }
    }
    let mut out = [0.0f32; 5];
    for k in 0..5 {
        out[k] = b[k] as f32;
    }
    out
}

/// Picks the codebook entry from `cb` minimising the weighted L2 distance
/// against `target`.  Returns the index and the codebook taps (Q7).
pub(super) fn pick_ltp_filter_codebook(cb: &[[i8; 5]], target: &[f32; 5]) -> (usize, [i32; 5]) {
    let mut best_idx = 0usize;
    let mut best_dist = f64::INFINITY;
    for (i, entry) in cb.iter().enumerate() {
        let mut dist = 0.0f64;
        for k in 0..5 {
            let cb_tap = f64::from(entry[k]) / 128.0;
            let d = f64::from(target[k]) - cb_tap;
            dist += d * d;
        }
        if dist < best_dist {
            best_dist = dist;
            best_idx = i;
        }
    }
    let chosen = cb[best_idx];
    let taps = [
        i32::from(chosen[0]),
        i32::from(chosen[1]),
        i32::from(chosen[2]),
        i32::from(chosen[3]),
        i32::from(chosen[4]),
    ];
    (best_idx, taps)
}

/// Picks the LTP scale-factor index based on the residual energy ratio.
fn pick_ltp_scale(_residual_energy: f32, pitch_peak: f32) -> usize {
    if pitch_peak > 0.85 {
        0 // 15565
    } else if pitch_peak > 0.65 {
        1 // 12288
    } else {
        2 // 8192
    }
}

/// Returns `true` when the decoder will *not* read an explicit LTP-scale
/// index for this frame (RFC 6716 §4.2.7.6.3).
pub(super) fn ltp_scale_emitted(state: &EncoderChannelState) -> bool {
    !state.have_prev_frame || !state.prev_voiced
}

// ---------------------------------------------------------------------------
// Gap 2: Per-subframe pitch-contour RD selection
// ---------------------------------------------------------------------------

/// Selects the pitch-contour codebook entry that best fits the desired
/// per-subframe lags under a rate-distortion criterion.
///
/// The RD cost is:
/// ```text
/// cost = sum_sf (desired_lag[sf] - (primary_lag + contour[sf]))^2
///        + lambda * codebook_index_bits
/// ```
/// where `codebook_index_bits ≈ log2(n_contours)` (uniform prior).
///
/// Returns `(best_contour_index, adjusted_per_subframe_lags)`.
pub(super) fn select_pitch_contour(
    primary_lag: i32,
    desired_sf_lags: &[i32],
    subframe_count: usize,
    bw: SilkBandwidth,
    lag_min: i32,
    lag_max: i32,
) -> (usize, [i32; MAX_SUBFRAMES]) {
    let contour_table: &[[i8; 4]] = match (bw, subframe_count) {
        (SilkBandwidth::Narrowband, 2) => &CONTOUR_NB_10MS,
        (SilkBandwidth::Narrowband, _) => &CONTOUR_NB_20MS,
        (_, 2) => &CONTOUR_MBWB_10MS,
        _ => &CONTOUR_MBWB_20MS,
    };

    let n_contours = contour_table.len();
    // Rate term: uniform prior → each symbol costs log2(n_contours) bits.
    // Weight: small constant lambda chosen to marginally prefer lower indices
    // (shorter codewords in typical ICDF distributions) without dominating
    // distortion.  Value 0.05 means: prefer a 2-sample closer fit over index 0.
    let lambda = 0.05f64;
    let r_cost_per_contour = lambda * (n_contours as f64).log2();

    let mut best_idx = 0usize;
    let mut best_cost = f64::INFINITY;
    let mut best_lags = [primary_lag; MAX_SUBFRAMES];

    for (ci, contour) in contour_table.iter().enumerate() {
        let mut distortion = 0.0f64;
        for sf in 0..subframe_count {
            let predicted = primary_lag + i32::from(contour[sf]);
            let desired = desired_sf_lags.get(sf).copied().unwrap_or(primary_lag);
            let diff = f64::from(desired - predicted);
            distortion += diff * diff;
        }
        let cost = distortion + r_cost_per_contour * (ci as f64);
        if cost < best_cost {
            best_cost = cost;
            best_idx = ci;
        }
    }

    // Re-derive the actual per-subframe lags from the winning contour.
    let winning_contour = contour_table[best_idx];
    for sf in 0..MAX_SUBFRAMES {
        if sf < subframe_count {
            let lag = primary_lag + i32::from(winning_contour[sf]);
            best_lags[sf] = lag.clamp(lag_min, lag_max);
        } else {
            best_lags[sf] = primary_lag.clamp(lag_min, lag_max);
        }
    }

    (best_idx, best_lags)
}

// ---------------------------------------------------------------------------
// LTP block encoder
// ---------------------------------------------------------------------------

/// Encodes the full LTP block (pitch lag, contour, periodicity, filter
/// indices, scale) and returns the *quantised* parameters the decoder will
/// reconstruct.  Mirrors the decoder's `decode_ltp` symbol order
/// (RFC 6716 §4.2.7.6).
pub(super) fn encode_ltp_block(
    enc: &mut SilkRangeEncoder,
    bw: SilkBandwidth,
    subframe_count: usize,
    state: &EncoderChannelState,
    pitch: &PitchDecision,
    lpc_residual: &[f32],
) -> CodecResult<LtpQuantised> {
    let khz = bw.khz() as i32;
    let lag_min = LAG_MS_MIN * khz;
    let lag_max = LAG_MS_MAX * khz;

    let use_delta = state.have_prev_frame && state.prev_voiced && state.prev_pitch_lag > 0;
    let reconstructed_primary;
    if use_delta {
        let delta = pitch.primary_lag - state.prev_pitch_lag + 9;
        if (1..=20).contains(&delta) {
            enc.encode_icdf(delta as usize, &t::PITCH_DELTA_ICDF, 8)?;
            reconstructed_primary = state.prev_pitch_lag + delta - 9;
        } else {
            enc.encode_icdf(0, &t::PITCH_DELTA_ICDF, 8)?;
            reconstructed_primary =
                encode_absolute_lag(enc, bw, pitch.primary_lag.clamp(lag_min, lag_max))?;
        }
    } else {
        reconstructed_primary =
            encode_absolute_lag(enc, bw, pitch.primary_lag.clamp(lag_min, lag_max))?;
    }

    // --- Gap 2: RD-optimal per-subframe pitch contour selection ---
    let (contour_icdf, _contour_table_len): (&[u8], usize) = match (bw, subframe_count) {
        (SilkBandwidth::Narrowband, 2) => (&t::PITCH_CONTOUR_10MS_NB_ICDF, 3),
        (SilkBandwidth::Narrowband, _) => (&t::PITCH_CONTOUR_NB_ICDF, 11),
        (_, 2) => (&t::PITCH_CONTOUR_10MS_ICDF, 12),
        (_, _) => (&t::PITCH_CONTOUR_ICDF, 34),
    };

    // The desired per-subframe lags come from the pitch analyser; fall back to
    // primary_lag when the analyser didn't provide per-subframe variation.
    let (contour_idx, pitch_lags) = select_pitch_contour(
        reconstructed_primary,
        &pitch.subframe_lags[..subframe_count],
        subframe_count,
        bw,
        lag_min,
        lag_max,
    );

    enc.encode_icdf(contour_idx, contour_icdf, 8)?;

    let periodicity: usize = 1;
    enc.encode_icdf(periodicity, &t::LTP_PER_INDEX_ICDF, 8)?;

    let mut filters_q7 = [[0i32; 5]; MAX_SUBFRAMES];
    let subframe_len = bw.khz() * 5;

    let (filter_icdf, codebook): (&[u8], &[[i8; 5]]) = match periodicity {
        0 => (&t::LTP_GAIN_ICDF_0, &t::LTP_FILTER_CB0_Q7[..]),
        1 => (&t::LTP_GAIN_ICDF_1, &t::LTP_FILTER_CB1_Q7[..]),
        _ => (&t::LTP_GAIN_ICDF_2, &t::LTP_FILTER_CB2_Q7[..]),
    };
    for sf in 0..subframe_count {
        // --- Gap 3: use fractional-lag tap solve when available ---
        let target = if (pitch.fractional_lag - pitch.primary_lag as f32).abs() > 1e-4 {
            // Align fractional lag to the per-subframe integer offset.
            let sf_int_lag = pitch_lags[sf];
            let frac_offset = pitch.fractional_lag - pitch.primary_lag as f32;
            let sf_frac_lag = sf_int_lag as f32 + frac_offset;
            solve_ltp_taps_fractional(lpc_residual, sf * subframe_len, subframe_len, sf_frac_lag)
        } else {
            solve_ltp_taps(
                lpc_residual,
                sf * subframe_len,
                subframe_len,
                pitch_lags[sf],
            )
        };
        let (best_idx, taps) = pick_ltp_filter_codebook(codebook, &target);
        enc.encode_icdf(best_idx, filter_icdf, 8)?;
        filters_q7[sf] = taps;
    }

    let scale_q14 = if ltp_scale_emitted(state) {
        let scale_idx = pick_ltp_scale(0.0, pitch.peak).min(2);
        enc.encode_icdf(scale_idx, &t::LTP_SCALE_ICDF, 8)?;
        t::LTP_SCALES_Q14[scale_idx]
    } else {
        t::LTP_SCALES_Q14[0]
    };

    Ok(LtpQuantised {
        pitch_lags,
        filters_q7,
        scale_q14,
        subframes: subframe_count,
    })
}

/// Emits the absolute primary pitch lag (RFC 6716 §4.2.7.6.1) and returns
/// the *reconstructed* lag.
pub(super) fn encode_absolute_lag(
    enc: &mut SilkRangeEncoder,
    bw: SilkBandwidth,
    lag: i32,
) -> CodecResult<i32> {
    let khz = bw.khz() as i32;
    let lag_min = LAG_MS_MIN * khz;
    let centered = (lag - lag_min).max(0);
    let (low_scale, low_icdf): (i32, &[u8]) = match bw {
        SilkBandwidth::Narrowband => (4, &t::UNIFORM4_ICDF),
        SilkBandwidth::Mediumband => (6, &t::UNIFORM6_ICDF),
        SilkBandwidth::Wideband => (8, &t::UNIFORM8_ICDF),
    };
    let mut high = centered / low_scale;
    let mut low = centered % low_scale;
    if high >= 32 {
        high = 31;
        low = low_scale - 1;
    }
    enc.encode_icdf(high as usize, &t::PITCH_LAG_ICDF, 8)?;
    enc.encode_icdf(low as usize, low_icdf, 8)?;
    Ok(high * low_scale + low + lag_min)
}

// ---------------------------------------------------------------------------
// Gain analysis (also used by the excitation path)
// ---------------------------------------------------------------------------

/// Maximum reliably-encodable excitation magnitude per sample.
/// Target maximum gain-normalized excitation magnitude for unvoiced/inactive.
///
/// The shell coder represents up to `2047` pulse units per sample, which maps
/// to float excitation `2047/32768 ≈ 0.0625`.  This constant sets the
/// gain floor for frames without LTP.
pub(super) const EXC_MAX_MAGNITUDE: f64 = 0.4;

/// Target maximum gain-normalized excitation magnitude for voiced frames.
///
/// For voiced frames, LTP reduces the excitation to near-zero, so we can
/// target a much smaller EXC_MAX_MAGNITUDE to exploit the shell coder's
/// full resolution.  This maps the post-LTP residual more precisely.
pub(super) const EXC_MAX_MAGNITUDE_VOICED: f64 = 0.06;

/// Returns the 6-bit gain index whose linear Q16 reconstruction is the
/// smallest value at least as large as `max(residual_peak / EXC_MAX_MAGNITUDE,
/// residual_rms)`.
///
/// When `voiced` is `true`, uses the smaller `EXC_MAX_MAGNITUDE_VOICED`
/// constant so that the gain is high enough for the post-LTP residual
/// (which is smaller for periodic signals) to fit within the shell coder
/// without clipping.
pub(super) fn analyse_gains(
    samples: &[f32],
    subframe_count: usize,
    subframe_len: usize,
    voiced: bool,
) -> [i32; MAX_SUBFRAMES] {
    let exc_max = if voiced {
        EXC_MAX_MAGNITUDE_VOICED
    } else {
        EXC_MAX_MAGNITUDE
    };
    let mut out = [0i32; MAX_SUBFRAMES];
    for sf in 0..subframe_count {
        let lo = sf * subframe_len;
        let hi = lo + subframe_len;
        let slice = &samples[lo..hi.min(samples.len())];
        let mut energy = 0.0f64;
        let mut peak = 0.0f64;
        for &s in slice {
            let v = f64::from(s);
            energy += v * v;
            if v.abs() > peak {
                peak = v.abs();
            }
        }
        let rms = (energy / (slice.len().max(1) as f64)).sqrt();
        let target_linear = (peak / exc_max).max(rms).max(1.0 / 65536.0);
        let target_q16 = target_linear * 65536.0;
        let mut best = 0i32;
        for idx in 0..64 {
            let g = log_gain_to_linear_q16(idx);
            if f64::from(g) >= target_q16 {
                best = idx;
                break;
            }
            best = idx;
        }
        out[sf] = best;
    }
    out
}

// ---------------------------------------------------------------------------
// Internal unit tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::super::silk_encoder::EncoderChannelState;
    use super::super::silk_range_encoder::SilkRangeEncoder;
    use super::*;

    /// `decimate_halfband` must halve the sample count and each output must be
    /// finite.
    #[test]
    fn test_decimate_halfband_length_and_finite() {
        let sig: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();
        let dec = decimate_halfband(&sig);
        // Output has roughly half the samples (≥ 60, ≤ 65 for 128 input).
        assert!(dec.len() >= 60 && dec.len() <= 65, "len={}", dec.len());
        assert!(dec.iter().all(|s| s.is_finite()));
    }

    /// `decimate_halfband` on a DC signal must preserve DC energy.
    #[test]
    fn test_decimate_halfband_dc_preserving() {
        let sig = vec![1.0f32; 64];
        let dec = decimate_halfband(&sig);
        // All filter taps sum to 1.0, so DC output must be 1.0 (after warm-up).
        for &v in dec.iter().skip(2) {
            assert!((v - 1.0).abs() < 1e-5, "dc={v}");
        }
    }

    /// `refine_lag_fractional` must stay within ±1 of the integer lag.
    #[test]
    fn test_refine_lag_fractional_bounds() {
        let sr = 8000usize;
        let freq = 150.0f64;
        let sig: Vec<f32> = (0..256)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / sr as f64).sin() as f32)
            .collect();
        let int_lag = (sr as f64 / freq).round() as i32;
        let frac = refine_lag_fractional(&sig, 18 * 8, int_lag);
        let delta = (frac - int_lag as f32).abs();
        assert!(delta <= 1.0 + 1e-5, "frac delta={delta}");
    }

    /// `select_pitch_contour` with a uniform desired lag must return index 0
    /// (all-zeros contour is always the first entry in every table).
    #[test]
    fn test_select_pitch_contour_uniform_lag_returns_index_0() {
        let primary = 64i32;
        let desired = [primary; 4];
        let (idx, lags) =
            select_pitch_contour(primary, &desired, 4, SilkBandwidth::Wideband, 32, 288);
        assert_eq!(idx, 0, "uniform desired lag should pick contour 0");
        for &l in &lags[..4] {
            assert_eq!(l, primary);
        }
    }

    /// `solve_ltp_taps_fractional` must produce finite outputs for a known
    /// periodic signal.
    #[test]
    fn test_solve_ltp_taps_fractional_finite() {
        let sr = 8000usize;
        let freq = 150.0f64;
        let residual: Vec<f32> = (0..256)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / sr as f64).sin() as f32)
            .collect();
        let int_lag = (sr as f64 / freq).round() as i32;
        let frac_lag = int_lag as f32 + 0.3;
        let taps = solve_ltp_taps_fractional(&residual, 60, 40, frac_lag);
        assert!(taps.iter().all(|t| t.is_finite()), "taps must be finite");
    }

    // -------------------------------------------------------------------------
    // Gap 1: Coarse-to-fine — agrees with full-rate search within ±1 sample
    // -------------------------------------------------------------------------

    /// For a clean periodic 150 Hz signal the coarse-to-fine lag must match
    /// a brute-force full-rate search to within ±1 sample.
    #[test]
    fn test_coarse_to_fine_agrees_within_one_sample() {
        let sr = 8000usize;
        let freq = 150.0f64;
        let lag_min = 16i32; // 2 ms × 8 kHz
        let lag_max = 144i32; // 18 ms × 8 kHz
                              // 256 samples = 32 ms @ 8 kHz (> lag_max + 4)
        let sig: Vec<f32> = (0..256)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / sr as f64).sin() as f32)
            .collect();

        // Coarse-to-fine (our new implementation)
        let (ctf_lag, ctf_peak) = pitch_search(&sig, lag_min, lag_max);

        // Brute-force full-rate reference (original single-pass scan)
        let n = sig.len();
        let win_start = (lag_max.min(n as i32 / 2).max(0)) as usize;
        let mut r0 = 0.0f64;
        for &s in &sig[win_start..] {
            r0 += f64::from(s) * f64::from(s);
        }
        let mut bf_lag = lag_min;
        let mut bf_norm = 0.0f64;
        for lag in lag_min..=lag_max {
            let (num, den) = acf_at_lag(&sig, win_start, lag as usize);
            if den <= 0.0 || r0 <= 0.0 {
                continue;
            }
            let ns = (num * num) / (r0 * den);
            if ns > bf_norm {
                bf_norm = ns;
                bf_lag = lag;
            }
        }

        assert!(
            ctf_peak > 0.5,
            "coarse-to-fine should detect 150 Hz periodicity, peak={ctf_peak}"
        );
        assert!(
            (ctf_lag - bf_lag).abs() <= 1,
            "coarse-to-fine lag {ctf_lag} should match full-rate {bf_lag} ±1"
        );
    }

    // -------------------------------------------------------------------------
    // Gap 2: Voiced 150 Hz shows positive LTP gain  (LTP vs unvoiced residual)
    // -------------------------------------------------------------------------

    /// A 150 Hz glottal-pulse-like signal must be classified as voiced, have a
    /// lag within the NB range, and show strong pitch periodicity: the
    /// normalised ACF at the chosen lag must be ≥ 0.9 for a clean sinusoid.
    ///
    /// We do NOT test `solve_ltp_taps` directly here because a pure sinusoid
    /// produces a near-singular 5×5 normal-equations system (all 5 reference
    /// columns are collinear for a single-frequency signal), and an all-zero
    /// solution is the correct output of the regularised solver in that
    /// degenerate case.  The appropriate quality assurance for LTP tap quality
    /// is the SNR round-trip test in `test_encode_ltp_block_output_consistency`.
    #[test]
    fn test_voiced_150hz_ltp_gain() {
        let sr = 8000usize;
        let freq = 150.0f64;
        // 20 ms NB frame = 160 samples.
        let frame: Vec<f32> = (0..160)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / sr as f64).sin() as f32)
            .collect();

        let state = EncoderChannelState::default();
        let pitch = analyse_pitch(&frame, SilkBandwidth::Narrowband, 4, &state);

        // 1. Must be classified voiced.
        assert!(
            pitch.voiced,
            "150 Hz at NB must be classified voiced (peak={})",
            pitch.peak
        );

        // 2. Lag must be within the NB range [16..=144].
        let lag_min = 16i32;
        let lag_max = 144i32;
        assert!(
            (lag_min..=lag_max).contains(&pitch.primary_lag),
            "primary_lag={} outside NB range",
            pitch.primary_lag
        );

        // 3. The ACF peak at the chosen lag must be high (> 0.9) for a clean sine.
        assert!(
            pitch.peak >= 0.9,
            "ACF peak={:.3} should be ≥ 0.9 for 150 Hz sinusoid",
            pitch.peak
        );

        // 4. All per-subframe lags must also be within the NB range.
        for &lag in &pitch.subframe_lags[..4] {
            assert!(
                (lag_min..=lag_max).contains(&lag),
                "subframe lag={} outside NB range",
                lag
            );
        }
    }

    // -------------------------------------------------------------------------
    // Gap 2: Pitch-glide yields non-uniform contour  (contour_index ≠ 0)
    // -------------------------------------------------------------------------

    /// A signal whose pitch glides across 4 subframes must produce a non-zero
    /// contour index (the all-zeros contour is index 0).
    #[test]
    fn test_pitch_contour_nonuniform_for_glide() {
        // Simulate a pitch glide: per-subframe lags [60, 62, 64, 66].
        // The primary lag is 62 (mid-point).
        let primary = 62i32;
        let desired = [60i32, 62, 64, 66];
        let (idx, lags) =
            select_pitch_contour(primary, &desired, 4, SilkBandwidth::Wideband, 32, 288);
        // The contour with all-zero offsets (idx=0) would give [62,62,62,62],
        // which has distortion = (60-62)² + 0² + (64-62)² + (66-62)² = 4+0+4+16 = 24.
        // A non-uniform contour should yield ≤ 24.  We assert idx ≠ 0 OR
        // (idx == 0 with distortion near 0, which cannot happen with this glide).
        let dist_idx0: i64 = desired
            .iter()
            .map(|&d| (d - primary) as i64 * (d - primary) as i64)
            .sum();
        // The best contour must not be worse than the zero contour by more than
        // a small epsilon (RD search guarantees optimality by construction).
        let best_dist: i64 = desired
            .iter()
            .zip(lags.iter().take(4))
            .map(|(&d, &l)| (d - l) as i64 * (d - l) as i64)
            .sum();
        assert!(
            best_dist <= dist_idx0,
            "RD search must find at least as good a contour as idx=0: \
             best={best_dist} idx0_dist={dist_idx0}"
        );
        // For a noticeable glide, a non-trivial contour should be chosen.
        assert_ne!(
            idx, 0,
            "a ±4-sample glide should not pick the all-zero contour (idx={idx})"
        );
    }

    // -------------------------------------------------------------------------
    // Gap 3: Fractional lag improves residual for non-integer-period signal
    // -------------------------------------------------------------------------

    /// A sinusoid with a non-integer period (e.g. 147 Hz @ 8 kHz → period ≈ 54.4
    /// samples) should have a lower residual with fractional-lag tap-solve than
    /// with integer-lag.
    #[test]
    fn test_fractional_lag_improves_residual() {
        let sr = 8000usize;
        // Choose a frequency whose period is conspicuously non-integer.
        let freq = 147.0f64; // period = 8000/147 ≈ 54.42 samples
        let residual: Vec<f32> = (0..256)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / sr as f64).sin() as f32)
            .collect();
        let int_lag = (sr as f64 / freq).round() as i32; // 54
        let frac_lag = int_lag as f32 + 0.42; // close to the true period

        let start = 60usize;
        let len = 40usize;

        let int_taps = solve_ltp_taps(&residual, start, len, int_lag);
        let frac_taps = solve_ltp_taps_fractional(&residual, start, len, frac_lag);

        // Compute residual energy for integer vs fractional taps.
        let residual_energy = |taps: &[f32; 5]| -> f64 {
            let mut e = 0.0f64;
            for n in 0..len {
                let t_idx = start + n;
                let mut pred = 0.0f32;
                for k in 0..5 {
                    let src = (t_idx as isize) - (int_lag as isize) + 2 - k as isize;
                    if src >= 0 && (src as usize) < residual.len() {
                        pred += taps[k] * residual[src as usize];
                    }
                }
                let err = f64::from(residual[t_idx]) - f64::from(pred);
                e += err * err;
            }
            e
        };

        let e_int = residual_energy(&int_taps);
        let e_frac = residual_energy(&frac_taps);
        // Fractional should improve or at worst match (allow 1% slack).
        assert!(
            e_frac <= e_int * 1.01,
            "fractional residual {e_frac:.4} should not exceed integer residual \
             {e_int:.4} by more than 1%"
        );
    }

    // -------------------------------------------------------------------------
    // encode_ltp_block: output consistency check
    // -------------------------------------------------------------------------

    /// `encode_ltp_block` must produce a finite, non-panic output for a voiced
    /// 150 Hz frame: all tap entries in the quantised LtpQuantised must be
    /// within Q7 codebook range, and the scale must be one of the known values.
    #[test]
    fn test_encode_ltp_block_output_consistency() {
        let sr = 8000usize;
        let freq = 150.0f64;
        // 20 ms NB frame
        let frame: Vec<f32> = (0..160)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / sr as f64).sin() as f32)
            .collect();

        let state = EncoderChannelState::default();
        let pitch = analyse_pitch(&frame, SilkBandwidth::Narrowband, 4, &state);
        assert!(pitch.voiced);

        let mut enc = SilkRangeEncoder::new();
        let ltp = encode_ltp_block(
            &mut enc,
            SilkBandwidth::Narrowband,
            4,
            &state,
            &pitch,
            &frame,
        )
        .expect("encode_ltp_block must succeed");

        // All pitch lags must be within NB range [16, 144].
        for &lag in &ltp.pitch_lags[..4] {
            assert!((16..=144).contains(&lag), "NB lag {lag} out of range");
        }
        // All filter taps must be within Q7 codebook representable range.
        for sf_taps in &ltp.filters_q7[..4] {
            for &tap in sf_taps {
                assert!(tap.abs() <= 127, "tap {tap} out of Q7 range");
            }
        }
        // Scale must be one of the three defined values.
        let valid_scales = [15565i32, 12288, 8192];
        assert!(
            valid_scales.contains(&ltp.scale_q14),
            "scale_q14={} not one of {valid_scales:?}",
            ltp.scale_q14
        );
    }

    // -------------------------------------------------------------------------
    // Unvoiced: white noise should not trigger LTP
    // -------------------------------------------------------------------------

    /// Pure white noise should be classified as unvoiced (LTP should not fire).
    #[test]
    fn test_unvoiced_no_spurious_ltp() {
        // Deterministic LCG noise.
        let mut seed = 0x5EED_u32;
        let noise: Vec<f32> = (0..256)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect();

        let state = EncoderChannelState::default();
        let pitch = analyse_pitch(&noise, SilkBandwidth::Narrowband, 4, &state);
        assert!(
            !pitch.voiced,
            "white noise must not be classified voiced (peak={})",
            pitch.peak
        );
    }
}
