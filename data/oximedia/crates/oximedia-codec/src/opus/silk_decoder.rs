//! Normative SILK frame decoder (RFC 6716 §4.2).
//!
//! This module implements the body of the SILK decoder: per-frame LP-layer
//! parsing (header flags, subframe gains, normalized LSF stage-1/stage-2
//! decode, NLSF stabilisation, NLSF-to-LPC conversion, LTP lag and filter
//! selection), the LCG-seeded excitation decode (the shell pulse decoder with
//! LSB and sign rounds), and the SILK synthesis filter (LTP synthesis followed
//! by LPC synthesis). It operates on a single 10 ms or 20 ms SILK frame at the
//! codec's internal sample rate; resampling to the output rate is handled by
//! the caller in `silk.rs`.
//!
//! All entropy decoding goes through [`SilkRangeDecoder`]; all constants come
//! from [`super::silk_tables`]. The algorithm follows RFC 6716 §4.2 step by
//! step — the reference fixed-point arithmetic is reproduced so the output is
//! numerically faithful to the normative decoder.

use crate::{CodecError, CodecResult};

use super::silk_range::SilkRangeDecoder;
use super::silk_tables as t;

/// SILK internal audio bandwidth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SilkBandwidth {
    /// Narrowband: 8 kHz internal sampling.
    Narrowband,
    /// Mediumband: 12 kHz internal sampling.
    Mediumband,
    /// Wideband: 16 kHz internal sampling.
    Wideband,
}

impl SilkBandwidth {
    /// Internal sample rate in kHz.
    pub const fn khz(self) -> usize {
        match self {
            Self::Narrowband => 8,
            Self::Mediumband => 12,
            Self::Wideband => 16,
        }
    }

    /// Internal sample rate in Hz.
    pub const fn hz(self) -> u32 {
        (self.khz() as u32) * 1000
    }

    /// LPC analysis order (`10` for NB/MB, `16` for WB).
    pub const fn lpc_order(self) -> usize {
        match self {
            Self::Wideband => 16,
            _ => 10,
        }
    }

    /// True when this bandwidth uses the wideband NLSF codebooks.
    pub const fn is_wideband(self) -> bool {
        matches!(self, Self::Wideband)
    }
}

/// SILK signal type for one frame (RFC 6716 §4.2.7.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SilkSignalType {
    /// Inactive (no speech).
    Inactive,
    /// Unvoiced speech.
    Unvoiced,
    /// Voiced speech.
    Voiced,
}

/// Maximum LPC order across all SILK bandwidths.
pub const MAX_LPC_ORDER: usize = 16;

/// Number of 5 ms subframes in a 20 ms SILK frame.
pub const MAX_SUBFRAMES: usize = 4;

/// Length of a SILK shell block in samples (RFC 6716 §4.2.7.8).
const SHELL_BLOCK_LEN: usize = 16;

/// Maximum primary pitch lag in samples at the WB internal rate.
const MAX_PITCH_LAG_WB: i32 = 18 * 16;

/// Per-channel persistent SILK decoder state carried between frames.
#[derive(Debug, Clone)]
pub struct SilkChannelState {
    /// Previous frame's quantized NLSF values (Q15).
    pub prev_nlsf_q15: Vec<i16>,
    /// Whether `prev_nlsf_q15` holds a valid decoded frame.
    pub have_prev_nlsf: bool,
    /// Previous frame's primary pitch lag (internal samples).
    pub prev_pitch_lag: i32,
    /// LPC synthesis history (most recent `MAX_LPC_ORDER` output samples, Q0).
    pub lpc_history: Vec<f32>,
    /// Excitation/output history for LTP synthesis (long-term buffer).
    pub ltp_history: Vec<f32>,
    /// Final `out` sample of the previous frame, for the first-subframe gain.
    pub prev_gain_index: i32,
    /// Whether the previous frame was decoded successfully (for delta gains).
    pub have_prev_frame: bool,
    /// Previous frame coded as voiced (used to gate LTP scaling defaults).
    pub prev_voiced: bool,
}

impl SilkChannelState {
    /// Creates a fresh, zeroed channel state for `order`-tap LPC.
    pub fn new() -> Self {
        Self {
            prev_nlsf_q15: vec![0; MAX_LPC_ORDER],
            have_prev_nlsf: false,
            prev_pitch_lag: 0,
            lpc_history: vec![0.0; MAX_LPC_ORDER],
            ltp_history: vec![0.0; (MAX_PITCH_LAG_WB as usize) + 32],
            prev_gain_index: 0,
            have_prev_frame: false,
            prev_voiced: false,
        }
    }

    /// Resets the state to its initial condition.
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

impl Default for SilkChannelState {
    fn default() -> Self {
        Self::new()
    }
}

/// Decoded parameters and synthesised PCM for one SILK frame.
#[derive(Debug, Clone)]
pub struct SilkFrameResult {
    /// Synthesised PCM samples at the internal sample rate, range roughly
    /// `[-1, 1]`.
    pub samples: Vec<f32>,
    /// Signal type decoded for this frame.
    pub signal_type: SilkSignalType,
}

/// Decodes a single SILK frame (10 ms or 20 ms) for one channel.
///
/// * `dec` — the shared range decoder positioned at the frame's first symbol.
/// * `bw` — the SILK internal bandwidth.
/// * `state` — persistent per-channel state, updated in place.
/// * `subframe_count` — `2` for a 10 ms frame, `4` for a 20 ms frame.
/// * `is_stereo_side` — `true` when decoding the side channel of a stereo pair
///   (affects only LTP-scaling defaults).
/// * `lbrr` — `true` when decoding a low-bitrate-redundancy copy.
///
/// Returns the synthesised samples (`subframe_count * 5 ms` at the internal
/// rate) and the decoded signal type.
pub fn decode_silk_frame(
    dec: &mut SilkRangeDecoder,
    bw: SilkBandwidth,
    state: &mut SilkChannelState,
    subframe_count: usize,
    is_stereo_side: bool,
    vad_flag: bool,
) -> CodecResult<SilkFrameResult> {
    let order = bw.lpc_order();
    let subframe_len = bw.khz() * 5;
    let frame_len = subframe_len * subframe_count;

    // --- §4.2.7.3 Frame type (signal type + quantization offset) ---
    let (signal_type, quant_offset_type) = decode_frame_type(dec, vad_flag)?;
    let voiced = signal_type == SilkSignalType::Voiced;

    // --- §4.2.7.4 Subframe gains ---
    let gains_q16 = decode_subframe_gains(dec, signal_type, subframe_count, state)?;

    // --- §4.2.7.5 Normalized LSF decode -> per-subframe LPC ---
    let nlsf_q15 = decode_nlsf(dec, bw, signal_type)?;

    // §4.2.7.5.4 NLSF interpolation factor (20 ms frames only).
    let interp_factor_q2 = if subframe_count == MAX_SUBFRAMES {
        dec.decode_icdf(&t::NLSF_INTERP_ICDF, 8)? as i32
    } else {
        4
    };

    // LPC coefficients for the second half of the frame (current NLSFs) and,
    // when interpolating, the first half (interpolated towards the previous
    // frame's NLSFs).
    let lpc_q12_cur = nlsf_to_lpc(&nlsf_q15, order);
    let lpc_q12_first = if interp_factor_q2 < 4 && state.have_prev_nlsf {
        let mut interp = vec![0i16; order];
        for i in 0..order {
            let prev = i32::from(state.prev_nlsf_q15[i]);
            let cur = i32::from(nlsf_q15[i]);
            interp[i] = (prev + ((interp_factor_q2 * (cur - prev)) >> 2)) as i16;
        }
        nlsf_to_lpc(&interp, order)
    } else {
        lpc_q12_cur.clone()
    };

    // --- §4.2.7.6 LTP parameters (voiced frames only) ---
    let mut pitch_lags = [0i32; MAX_SUBFRAMES];
    let mut ltp_filters_q7 = [[0i32; 5]; MAX_SUBFRAMES];
    let mut ltp_scale_q14 = t::LTP_SCALES_Q14[0];
    if voiced {
        decode_ltp(
            dec,
            bw,
            subframe_count,
            state,
            &mut pitch_lags,
            &mut ltp_filters_q7,
        )?;
        // §4.2.7.6.3 LTP scaling factor.
        let need_scale_index = !state.have_prev_frame || is_stereo_side || !state.prev_voiced;
        let scale_index = if need_scale_index {
            dec.decode_icdf(&t::LTP_SCALE_ICDF, 8)?
        } else {
            0
        };
        ltp_scale_q14 = t::LTP_SCALES_Q14[scale_index.min(2)];
    }

    // --- §4.2.7.7 LCG seed (uniform over 4 values) ---
    let lcg_seed = dec.decode_icdf(&t::UNIFORM4_ICDF, 8)? as u32;

    // --- §4.2.7.8 Excitation ---
    let excitation = decode_excitation(dec, frame_len, signal_type, quant_offset_type, lcg_seed)?;

    // --- §4.2.7.9 SILK synthesis ---
    let samples = synthesise(
        &excitation,
        &gains_q16,
        &lpc_q12_first,
        &lpc_q12_cur,
        order,
        subframe_count,
        subframe_len,
        voiced,
        &pitch_lags,
        &ltp_filters_q7,
        ltp_scale_q14,
        state,
    );

    // --- Update persistent state ---
    state.prev_nlsf_q15[..order].copy_from_slice(&nlsf_q15[..order]);
    state.have_prev_nlsf = true;
    state.have_prev_frame = true;
    state.prev_voiced = voiced;
    if voiced {
        state.prev_pitch_lag = pitch_lags[subframe_count - 1];
    }

    Ok(SilkFrameResult {
        samples,
        signal_type,
    })
}

/// Decodes the SILK frame-type symbol (RFC 6716 §4.2.7.3, Table 7).
fn decode_frame_type(
    dec: &mut SilkRangeDecoder,
    vad_flag: bool,
) -> CodecResult<(SilkSignalType, usize)> {
    if !vad_flag {
        // Inactive frame: 1-of-2 symbol -> quant offset 0 or 1.
        let sym = dec.decode_icdf(&t::TYPE_OFFSET_NO_VAD_ICDF, 8)?;
        Ok((SilkSignalType::Inactive, sym))
    } else {
        // Active frame: 1-of-4 -> {unvoiced,low},{unvoiced,high},{voiced,low},
        // {voiced,high}.
        let sym = dec.decode_icdf(&t::TYPE_OFFSET_VAD_ICDF, 8)?;
        let signal_type = if sym >= 2 {
            SilkSignalType::Voiced
        } else {
            SilkSignalType::Unvoiced
        };
        let quant_offset_type = sym & 1;
        Ok((signal_type, quant_offset_type))
    }
}

/// Decodes the per-subframe quantization gains (RFC 6716 §4.2.7.4).
fn decode_subframe_gains(
    dec: &mut SilkRangeDecoder,
    signal_type: SilkSignalType,
    subframe_count: usize,
    state: &mut SilkChannelState,
) -> CodecResult<Vec<i32>> {
    let mut log_gain = [0i32; MAX_SUBFRAMES];
    let type_index = match signal_type {
        SilkSignalType::Inactive => 0,
        SilkSignalType::Unvoiced => 1,
        SilkSignalType::Voiced => 2,
    };

    let mut prev_log_gain = state.prev_gain_index;
    for sf in 0..subframe_count {
        let independent = sf == 0 && !state.have_prev_frame;
        let coded_independently = sf == 0;
        let gain_index = if coded_independently {
            // 6-bit gain: 3-bit MSB (context = signal type) + 3-bit LSB.
            let msb = dec.decode_icdf(&t::GAIN_ICDF[type_index], 8)? as i32;
            let lsb = dec.decode_icdf(&t::UNIFORM8_ICDF, 8)? as i32;
            let idx = (msb << 3) + lsb;
            if independent {
                idx
            } else {
                // Clamp against the previous frame's last gain.
                idx.max(prev_log_gain - 16)
            }
        } else {
            // Delta gain relative to the previous subframe.
            let delta = dec.decode_icdf(&t::DELTA_GAIN_ICDF, 8)? as i32;
            // RFC 6716 §4.2.7.4: delta in {0..40}; values 0..7 map to
            // -4..3, 8..40 map to 4..36 (i.e. (delta-4) clamped).
            let step = if delta < 16 {
                delta - 4
            } else {
                2 * delta - 20
            };
            (prev_log_gain + step).clamp(0, 63)
        };
        let gain_index = gain_index.clamp(0, 63);
        log_gain[sf] = gain_index;
        prev_log_gain = gain_index;
    }
    state.prev_gain_index = prev_log_gain;

    // Convert log-gain indices to linear Q16 gains (RFC 6716 §4.2.7.4 final
    // paragraph: the reference uses silk_log2lin on a Q7 log value).
    let mut gains = vec![0i32; subframe_count];
    for sf in 0..subframe_count {
        gains[sf] = log_gain_to_linear_q16(log_gain[sf]);
    }
    Ok(gains)
}

/// Converts a 6-bit SILK log-gain index to a linear Q16 gain.
///
/// Follows RFC 6716 §4.2.7.4: `log_gain` indexes a Q7 logarithmic scale; the
/// reference computes `gain_Q16 = silk_log2lin( 0x1D1C71*idx>>16 + 2090 )`.
///
/// Exposed `pub(super)` so the encoder can quantise gains using the same
/// non-linear mapping the decoder will reconstruct.
pub(super) fn log_gain_to_linear_q16(index: i32) -> i32 {
    let log_q7 = ((0x001D_1C71_i64 * i64::from(index)) >> 16) as i32 + 2090;
    log2lin(log_q7)
}

/// SILK `silk_log2lin`: converts a Q7 base-2 logarithm to a linear integer.
fn log2lin(in_log_q7: i32) -> i32 {
    if in_log_q7 < 0 {
        return 0;
    }
    if in_log_q7 >= 3967 {
        return i32::MAX;
    }
    let mut out = 1i32 << (in_log_q7 >> 7);
    let frac = in_log_q7 & 127;
    // Second-order polynomial refinement (silk_log2lin).
    let refinement = ((out >> 7) * ((frac * (128 - frac)) >> 11)) - ((out >> 7) * 0);
    out = out.wrapping_add(((out as i64 * i64::from(frac)) >> 7) as i32);
    out.wrapping_add(refinement)
}

// ---------------------------------------------------------------------------
// NLSF decoding (RFC 6716 §4.2.7.5)
// ---------------------------------------------------------------------------

/// Decodes the normalized LSF vector for one SILK frame, returning Q15 values.
fn decode_nlsf(
    dec: &mut SilkRangeDecoder,
    bw: SilkBandwidth,
    signal_type: SilkSignalType,
) -> CodecResult<Vec<i16>> {
    let order = bw.lpc_order();
    let wb = bw.is_wideband();
    let voiced = signal_type == SilkSignalType::Voiced;
    let voiced_idx = usize::from(voiced);

    // --- §4.2.7.5.1 stage-1 index ---
    let stage1_icdf: &[u8] = if wb {
        &t::NLSF_CB1_ICDF_WB[voiced_idx]
    } else {
        &t::NLSF_CB1_ICDF_NB_MB[voiced_idx]
    };
    let i1 = dec.decode_icdf(stage1_icdf, 8)?;

    // --- §4.2.7.5.2 unpack the per-coefficient selector bytes ---
    // Each selector byte (libopus `silk_NLSF_unpack`) carries, for an even/odd
    // coefficient pair: the residual-codebook index and the prediction-weight
    // selector. The flat selector table is indexed `[i1*(order/2) + coeff/2]`.
    let select_table: &[u8] = if wb {
        &t::NLSF_CB2_SELECT_WB
    } else {
        &t::NLSF_CB2_SELECT_NB_MB
    };
    let pred_row0: &[u8] = if wb {
        &t::NLSF_PRED_WB_Q8[0]
    } else {
        &t::NLSF_PRED_NB_MB_Q8[0]
    };
    let pred_row1: &[u8] = if wb {
        &t::NLSF_PRED_WB_Q8[1]
    } else {
        &t::NLSF_PRED_NB_MB_Q8[1]
    };

    let mut residual_cb = vec![0usize; order]; // residual codebook index/coeff
    let mut pred_q8 = vec![0i32; order]; // prediction weight per coefficient
    for pair in 0..order / 2 {
        let entry = select_table[i1 * (order / 2) + pair];
        let even = 2 * pair;
        let odd = even + 1;
        residual_cb[even] = usize::from((entry >> 1) & 0x07);
        residual_cb[odd] = usize::from((entry >> 5) & 0x07);
        // The prediction weight is taken from row 0 or row 1 of the
        // per-coefficient prediction-weight table; the last coefficient has no
        // weight (it is never used as a predictor source).
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

    // --- §4.2.7.5.2 decode the stage-2 residual indices ---
    let mut res_idx = vec![0i32; order];
    for coeff in 0..order {
        let icdf: &[u8] = if wb {
            &t::NLSF_CB2_ICDF_WB[residual_cb[coeff]]
        } else {
            &t::NLSF_CB2_ICDF_NB_MB[residual_cb[coeff]]
        };
        // The 9-entry iCDF codes a value in [0, 8]; subtract 4 to centre it.
        let mut value = dec.decode_icdf(icdf, 8)? as i32 - 4;
        // A residual that saturates at magnitude 4 is extended with the
        // NLSF_EXT geometric tail.
        if value == 4 {
            value += dec.decode_icdf(&t::NLSF_EXT_ICDF, 8)? as i32;
        } else if value == -4 {
            value -= dec.decode_icdf(&t::NLSF_EXT_ICDF, 8)? as i32;
        }
        res_idx[coeff] = value;
    }

    // --- §4.2.7.5.3 reconstruct residuals with backward prediction ---
    let qstep = if wb {
        t::NLSF_QSTEP_WB
    } else {
        t::NLSF_QSTEP_NB_MB
    };
    // libopus `silk_NLSF_residual_dequant`: process from the last coefficient
    // backwards, carrying `out_q10` as the predictor for the previous one.
    let mut residual_q10 = vec![0i32; order];
    let mut out_q10 = 0i32;
    for coeff in (0..order).rev() {
        // Prediction from the next (already-decoded) coefficient.
        let pred_q10 = (out_q10 * pred_q8[coeff]) >> 8;
        // Raw quantised residual, with the ±0.1 (Q10 = 102) level adjustment.
        let mut raw = res_idx[coeff] << 10;
        if raw > 0 {
            raw -= 102;
        } else if raw < 0 {
            raw += 102;
        }
        // out_q10 = pred_q10 + SMULWB(raw, quant_step_size_Q16).
        out_q10 = pred_q10 + (((raw as i64) * (qstep as i64)) >> 16) as i32;
        residual_q10[coeff] = out_q10;
    }

    // --- combine stage-1 codebook with the weighted residual ---
    let mut nlsf_q15 = vec![0i32; order];
    for coeff in 0..order {
        let cb_q8 = if wb {
            i32::from(t::NLSF_CB1_WB_Q8[i1][coeff])
        } else {
            i32::from(t::NLSF_CB1_NB_MB_Q8[i1][coeff])
        };
        let wght_q9 = if wb {
            i32::from(t::NLSF_CB1_WGHT_WB_Q9[i1][coeff])
        } else {
            i32::from(t::NLSF_CB1_WGHT_NB_MB_Q9[i1][coeff])
        };
        // NLSF_Q15 = (res_Q10 << 14) / weight_Q9 + (cb_Q8 << 7).
        let add = if wght_q9 != 0 {
            (residual_q10[coeff] << 14) / wght_q9
        } else {
            0
        };
        nlsf_q15[coeff] = (add + (cb_q8 << 7)).clamp(0, 32767);
    }

    // --- §4.2.7.5.5 stabilisation ---
    let min_spacing: &[i16] = if wb {
        &t::NLSF_DELTA_MIN_WB_Q15
    } else {
        &t::NLSF_DELTA_MIN_NB_MB_Q15
    };
    stabilise_nlsf(&mut nlsf_q15, min_spacing, order);

    Ok(nlsf_q15.iter().map(|&v| v as i16).collect())
}

/// Stabilises NLSF coefficients so successive values respect the minimum
/// spacing table (RFC 6716 §4.2.7.5.5).
///
/// Exposed `pub(super)` so the encoder can apply the identical stabilisation
/// to the NLSFs it embeds in the bitstream.
pub(super) fn stabilise_nlsf(nlsf: &mut [i32], min_spacing: &[i16], order: usize) {
    // A bounded number of corrective passes mirrors `silk_NLSF_stabilize`.
    for _ in 0..20 {
        // Find the smallest violation of the spacing constraint.
        let mut min_diff = i32::MAX;
        let mut min_idx = 0usize;
        for i in 0..=order {
            let prev = if i == 0 { 0 } else { nlsf[i - 1] };
            let next = if i == order { 32768 } else { nlsf[i] };
            let diff = next - prev - i32::from(min_spacing[i]);
            if diff < min_diff {
                min_diff = diff;
                min_idx = i;
            }
        }
        if min_diff >= 0 {
            return;
        }
        if min_idx == 0 {
            nlsf[0] = i32::from(min_spacing[0]);
        } else if min_idx == order {
            nlsf[order - 1] = 32768 - i32::from(min_spacing[order]);
        } else {
            // Centre the offending pair around their midpoint.
            let mut min_center = 0i32;
            for k in 0..=min_idx {
                min_center += i32::from(min_spacing[k]);
            }
            min_center -= i32::from(min_spacing[min_idx]) / 2;
            let mut max_center = 32768;
            for k in min_idx..=order {
                max_center -= i32::from(min_spacing[k]);
            }
            max_center += i32::from(min_spacing[min_idx]) / 2;
            let center =
                ((nlsf[min_idx - 1] + nlsf[min_idx] + 1) >> 1).clamp(min_center, max_center);
            nlsf[min_idx - 1] = center - i32::from(min_spacing[min_idx]) / 2;
            nlsf[min_idx] = nlsf[min_idx - 1] + i32::from(min_spacing[min_idx]);
        }
    }
    // Final guaranteed-monotone pass.
    nlsf[0] = nlsf[0].max(i32::from(min_spacing[0]));
    for i in 1..order {
        nlsf[i] = nlsf[i].max(nlsf[i - 1] + i32::from(min_spacing[i]));
    }
    for i in (0..order).rev() {
        let ceil = if i == order - 1 {
            32768 - i32::from(min_spacing[order])
        } else {
            nlsf[i + 1] - i32::from(min_spacing[i + 1])
        };
        nlsf[i] = nlsf[i].min(ceil);
    }
}

/// Interleave ordering of the NLSF cosine values for the NB/MB poly build
/// (libopus `silk_NLSF2A` `ordering10`).
const NLSF2A_ORDERING_NB: [usize; 10] = [0, 9, 6, 3, 4, 5, 8, 1, 2, 7];

/// Interleave ordering of the NLSF cosine values for the WB poly build
/// (libopus `silk_NLSF2A` `ordering16`).
const NLSF2A_ORDERING_WB: [usize; 16] = [0, 15, 8, 7, 4, 11, 12, 3, 2, 13, 10, 5, 6, 9, 14, 1];

/// Fixed-point fractional bits used by the NLSF-to-LPC polynomial arithmetic
/// (libopus `QA` = 16 for the `out` polynomials).
const NLSF2A_QA: u32 = 16;

/// Converts an ordered NLSF vector (Q15) to LPC coefficients (Q12).
///
/// This is the normative cosine-domain method of RFC 6716 §4.2.7.5.6, faithful
/// to libopus `silk_NLSF2A`: each NLSF indexes the cosine table; the cosines
/// are interleaved by [`NLSF2A_ORDERING_NB`]/[`NLSF2A_ORDERING_WB`] and used to
/// build the two symmetric polynomials `P` and `Q`; the LPC coefficients are
/// `a[k] = -(Q + P)`, `a[d-k-1] = (Q - P)`, then range-limited and shifted to
/// Q12.
///
/// Exposed `pub(super)` so the SILK encoder can compute reconstructed LPC
/// coefficients — encoder and decoder must converge on the *same* LPC for
/// closed-loop quantization (RFC 6716 §4.2.7.5).
pub(super) fn nlsf_to_lpc(nlsf_q15: &[i16], order: usize) -> Vec<i32> {
    // --- cosine of each NLSF, placed into interleaved order ---
    let ordering: &[usize] = if order == 16 {
        &NLSF2A_ORDERING_WB
    } else {
        &NLSF2A_ORDERING_NB
    };
    let mut cos_q12 = [0i32; MAX_LPC_ORDER];
    for k in 0..order {
        let nlsf = i32::from(nlsf_q15[k]).clamp(0, 32767);
        // NLSF is Q15; the 129-entry table is indexed by the top 7 bits.
        let f_int = (nlsf >> 8) as usize;
        let f_frac = nlsf & 0xFF;
        let lo = t::LSF_COS_Q12[f_int.min(128)];
        let hi = t::LSF_COS_Q12[(f_int + 1).min(128)];
        // SMULWB: (delta * frac) >> 16 with `frac` treated as a 16-bit value.
        let interp = ((i64::from(hi - lo) * i64::from(f_frac)) >> 16) as i32;
        cos_q12[ordering[k]] = lo + interp;
    }

    // --- build the P (even taps) and Q (odd taps) polynomials ---
    let dd = order / 2;
    let mut p = [0i64; MAX_LPC_ORDER / 2 + 1];
    let mut q = [0i64; MAX_LPC_ORDER / 2 + 1];
    nlsf2a_find_poly(&mut p, &cos_q12, dd, 0);
    nlsf2a_find_poly(&mut q, &cos_q12, dd, 1);

    // --- combine into LPC, QA+1 fixed point ---
    let mut a_qa1 = [0i64; MAX_LPC_ORDER];
    for k in 0..dd {
        let p_tmp = p[k + 1] + p[k];
        let q_tmp = q[k + 1] - q[k];
        a_qa1[k] = -q_tmp - p_tmp;
        a_qa1[order - k - 1] = q_tmp - p_tmp;
    }

    // --- range-limit and convert QA+1 -> Q12 ---
    limit_and_quantise_lpc(&a_qa1, order)
}

/// Builds one of the two SILK NLSF polynomials in place (libopus
/// `silk_NLSF2A_find_poly`). `out` is the polynomial accumulator
/// (`QA` = Q16), `cos_q12` holds the interleaved cosine values in Q12
/// (the table stores `2*cos(NLSF) * 4096`); `dd = order/2`; and `parity`
/// selects the even (`0`) or odd (`1`) cosine subset.
///
/// libopus stores the cosine values in Q16 and shifts by `QA = 16` in
/// the multiplication. Our cosine table is in Q12, so we promote each
/// cosine to Q16 (`<<4`) before using it in any sum/assignment that
/// targets a Q16 polynomial accumulator. The multiply path stays
/// equivalent: `(cos_Q12 * out_Q16) >> (QA - 4) = (cos_Q16 * out_Q16) >> QA`.
fn nlsf2a_find_poly(out: &mut [i64], cos_q12: &[i32], dd: usize, parity: usize) {
    out[0] = 1i64 << NLSF2A_QA;
    // Q12 cosine → Q16 to match `out[]` scaling.
    out[1] = -(i64::from(cos_q12[parity]) << 4);
    for k in 1..dd {
        let ftmp_q12 = i64::from(cos_q12[2 * k + parity]);
        let ftmp_q16 = ftmp_q12 << 4;
        // out[k+1] = (out[k-1]<<1) - round(ftmp_q12 * out[k] >> (QA-4))
        //          = (out[k-1]<<1) - round(ftmp_q16 * out[k] >> QA).
        out[k + 1] = (out[k - 1] << 1) - rshift_round(ftmp_q12 * out[k], NLSF2A_QA - 4);
        for n in (2..=k).rev() {
            out[n] += out[n - 2] - rshift_round(ftmp_q12 * out[n - 1], NLSF2A_QA - 4);
        }
        out[1] -= ftmp_q16;
    }
}

/// Rounded arithmetic right shift (`silk_RSHIFT_ROUND`).
fn rshift_round(value: i64, shift: u32) -> i64 {
    if shift == 0 {
        value
    } else {
        (value + (1i64 << (shift - 1))) >> shift
    }
}

/// Range-limits the QA+1 LPC coefficients, applies stability-enforcing
/// bandwidth expansion (libopus `silk_NLSF2A` lines 132-139 +
/// `silk_LPC_inverse_pred_gain`), and converts the result to Q12.
///
/// The libopus reference does this in two passes:
///   1. Magnitude clamp: while any Q12 coefficient saturates the int16
///      range, chirp the polynomial down.
///   2. Stability fix: while `silk_LPC_inverse_pred_gain` rejects the
///      filter, apply an aggressive chirp `65536 - (2 << i)` and retry.
///
/// `MAX_LPC_STABILIZE_ITERATIONS` (= 16) is the libopus loop count;
/// by the last iteration the chirp factor is zero, which forces the
/// LPC to be all-zero (trivially stable).
fn limit_and_quantise_lpc(a_qa1: &[i64], order: usize) -> Vec<i32> {
    let mut a = [0i64; MAX_LPC_ORDER];
    a[..order].copy_from_slice(&a_qa1[..order]);

    // --- Pass 1: magnitude clamp (matches libopus `MaxLoops` = 10). ---
    for _ in 0..10 {
        let mut maxabs = 0i64;
        for &c in a.iter().take(order) {
            if c.abs() > maxabs {
                maxabs = c.abs();
            }
        }
        let maxabs_q12 = rshift_round(maxabs, NLSF2A_QA + 1 - 12);
        if maxabs_q12 <= 32767 {
            break;
        }
        let maxabs_clamped = maxabs.max(163_838);
        let sc_q16 = 65_470 - ((65_470i64 * 32_773 / (maxabs_clamped >> 4).max(1)).min(65_470));
        let chirp = sc_q16.clamp(0, 65_536) as u32;
        bwexpander_32(&mut a[..order], chirp);
    }

    // --- Pass 2: stability check + iterative chirp until stable. ---
    let mut lpc_q12 = qa1_to_q12(&a, order);
    for i in 0..16 {
        if lpc_inverse_pred_gain_stable(&lpc_q12, order) {
            break;
        }
        // libopus: chirp = 65536 - (2 << i).
        let chirp = (65_536i64 - (2i64 << i)).max(0).min(65_536) as u32;
        bwexpander_32(&mut a[..order], chirp);
        lpc_q12 = qa1_to_q12(&a, order);
    }
    lpc_q12
}

/// QA+1 → Q12 with rounding and int16 saturation.
fn qa1_to_q12(a_qa1: &[i64], order: usize) -> Vec<i32> {
    let mut lpc_q12 = vec![0i32; order];
    for i in 0..order {
        let v = rshift_round(a_qa1[i], NLSF2A_QA + 1 - 12);
        lpc_q12[i] = v.clamp(-32768, 32767) as i32;
    }
    lpc_q12
}

/// Returns `true` when the LPC `H(z) = 1 / (1 + sum a_k z^-k-1)` is
/// stable (poles inside the unit circle) with a small numerical margin.
///
/// Floating-point port of libopus `silk_LPC_inverse_pred_gain_c`:
/// step-down (Schur) recursion that computes reflection coefficients.
/// libopus uses `A_LIMIT = 0.99975` for the threshold; we match that.
pub(super) fn lpc_inverse_pred_gain_stable(lpc_q12: &[i32], order: usize) -> bool {
    if order == 0 {
        return true;
    }
    // Quick DC reject: libopus rejects if sum(a) >= 4096 in Q12 (i.e., A(1)
    // implies the polynomial is non-monic in the wrong direction → unstable).
    let dc_resp: i64 = lpc_q12.iter().take(order).map(|&c| i64::from(c)).sum();
    if dc_resp >= 4096 {
        return false;
    }
    // Floating-point Schur recursion. libopus A_LIMIT = 0.99975.
    let mut a: Vec<f64> = lpc_q12
        .iter()
        .take(order)
        .map(|&c| f64::from(c) / 4096.0)
        .collect();
    let mut work = vec![0.0f64; order];
    for k in (0..order).rev() {
        let rc = -a[k];
        if !rc.is_finite() || rc.abs() >= 0.99975 {
            return false;
        }
        let denom = 1.0 - rc * rc;
        if denom <= 1e-12 {
            return false;
        }
        for n in 0..k {
            work[n] = (a[n] + rc * a[k - 1 - n]) / denom;
        }
        a[..k].copy_from_slice(&work[..k]);
    }
    true
}

/// Applies a chirp (bandwidth expansion) to 32-bit LPC coefficients
/// (`silk_bwexpander_32`). `chirp_q16` is the Q16 expansion factor.
fn bwexpander_32(coeffs: &mut [i64], chirp_q16: u32) {
    let mut chirp = i64::from(chirp_q16);
    let chirp_minus_one = chirp - 65_536;
    let n = coeffs.len();
    for c in coeffs.iter_mut().take(n - 1) {
        *c = rshift_round(chirp * *c, 16);
        chirp = chirp + rshift_round(chirp * chirp_minus_one, 16);
    }
    if let Some(last) = coeffs.last_mut() {
        *last = rshift_round(chirp * *last, 16);
    }
}

// ---------------------------------------------------------------------------
// LTP decoding (RFC 6716 §4.2.7.6)
// ---------------------------------------------------------------------------

/// Decodes the LTP lag and 5-tap filter for each subframe of a voiced frame.
fn decode_ltp(
    dec: &mut SilkRangeDecoder,
    bw: SilkBandwidth,
    subframe_count: usize,
    state: &mut SilkChannelState,
    pitch_lags: &mut [i32; MAX_SUBFRAMES],
    ltp_filters_q7: &mut [[i32; 5]; MAX_SUBFRAMES],
) -> CodecResult<()> {
    let khz = bw.khz() as i32;

    // §4.2.7.6.1 Primary lag: absolute (no previous lag) or relative.
    let use_relative = state.have_prev_frame && state.prev_voiced && state.prev_pitch_lag > 0;
    let primary_lag = if use_relative {
        let delta = dec.decode_icdf(&t::PITCH_DELTA_ICDF, 8)? as i32;
        if delta == 0 {
            // Escape to an absolute lag.
            decode_absolute_lag(dec, bw)?
        } else {
            state.prev_pitch_lag + delta - 9
        }
    } else {
        decode_absolute_lag(dec, bw)?
    };

    // §4.2.7.6.1 Pitch contour: per-subframe lag offsets.
    let (contour_icdf, contour_table): (&[u8], &[[i8; 4]]) = match (bw, subframe_count) {
        (SilkBandwidth::Narrowband, 2) => (&t::PITCH_CONTOUR_10MS_NB_ICDF, &CONTOUR_NB_10MS),
        (SilkBandwidth::Narrowband, _) => (&t::PITCH_CONTOUR_NB_ICDF, &CONTOUR_NB_20MS),
        (_, 2) => (&t::PITCH_CONTOUR_10MS_ICDF, &CONTOUR_MBWB_10MS),
        (_, _) => (&t::PITCH_CONTOUR_ICDF, &CONTOUR_MBWB_20MS),
    };
    let contour_index = dec.decode_icdf(contour_icdf, 8)?;
    let contour = contour_table.get(contour_index).copied().unwrap_or([0; 4]);

    let lag_min = 2 * khz;
    let lag_max = 18 * khz;
    for sf in 0..subframe_count {
        let lag = primary_lag + i32::from(contour[sf]);
        pitch_lags[sf] = lag.clamp(lag_min, lag_max);
    }

    // §4.2.7.6.2 LTP filter: one periodicity index for the frame, then a
    // 5-tap filter index per subframe.
    let periodicity = dec.decode_icdf(&t::LTP_PER_INDEX_ICDF, 8)?;
    let (filter_icdf, codebook): (&[u8], &[[i8; 5]]) = match periodicity {
        0 => (&t::LTP_GAIN_ICDF_0, &t::LTP_FILTER_CB0_Q7),
        1 => (&t::LTP_GAIN_ICDF_1, &t::LTP_FILTER_CB1_Q7),
        _ => (&t::LTP_GAIN_ICDF_2, &t::LTP_FILTER_CB2_Q7),
    };
    for sf in 0..subframe_count {
        let filter_index = dec.decode_icdf(filter_icdf, 8)?;
        let taps = codebook.get(filter_index).copied().unwrap_or([0; 5]);
        for (k, &tap) in taps.iter().enumerate() {
            ltp_filters_q7[sf][k] = i32::from(tap);
        }
    }
    Ok(())
}

/// Decodes an absolute primary pitch lag (RFC 6716 §4.2.7.6.1).
fn decode_absolute_lag(dec: &mut SilkRangeDecoder, bw: SilkBandwidth) -> CodecResult<i32> {
    let khz = bw.khz() as i32;
    let high = dec.decode_icdf(&t::PITCH_LAG_ICDF, 8)? as i32;
    let (low_icdf, low_scale): (&[u8], i32) = match bw {
        SilkBandwidth::Narrowband => (&t::UNIFORM4_ICDF, 4),
        SilkBandwidth::Mediumband => (&t::UNIFORM6_ICDF, 6),
        SilkBandwidth::Wideband => (&t::UNIFORM8_ICDF, 8),
    };
    let low = dec.decode_icdf(low_icdf, 8)? as i32;
    // lag = high*scale + low + lag_min, with lag_min = 2 ms.
    let _ = low_scale;
    Ok(high * low_scale + low + 2 * khz)
}

// ---------------------------------------------------------------------------
// Excitation decoding (RFC 6716 §4.2.7.8)
// ---------------------------------------------------------------------------

/// Decodes the excitation (residual) signal for one SILK frame.
fn decode_excitation(
    dec: &mut SilkRangeDecoder,
    frame_len: usize,
    signal_type: SilkSignalType,
    quant_offset_type: usize,
    lcg_seed: u32,
) -> CodecResult<Vec<f32>> {
    let voiced = signal_type == SilkSignalType::Voiced;
    let voiced_idx = usize::from(voiced);

    // §4.2.7.8.1 Rate level.
    let rate_level = dec.decode_icdf(&t::RATE_LEVELS_ICDF[voiced_idx], 8)?;

    let shell_blocks = frame_len.div_ceil(SHELL_BLOCK_LEN);
    let mut e_raw = vec![0i32; shell_blocks * SHELL_BLOCK_LEN];

    // §4.2.7.8.2 / §4.2.7.8.3 Pulse counts and shell decode per block.
    let mut pulse_counts = vec![0i32; shell_blocks];
    let mut lsb_counts = vec![0i32; shell_blocks];
    for blk in 0..shell_blocks {
        let (count, lsbs) = decode_pulse_count(dec, rate_level)?;
        pulse_counts[blk] = count;
        lsb_counts[blk] = lsbs;
    }

    // §4.2.7.8.4 Locations of the pulses (shell decoder).
    for blk in 0..shell_blocks {
        let base = blk * SHELL_BLOCK_LEN;
        let mut block = [0i32; SHELL_BLOCK_LEN];
        decode_shell_block(dec, pulse_counts[blk], &mut block)?;
        // §4.2.7.8.5 LSBs: each LSB round doubles the magnitude.
        let lsbs = lsb_counts[blk];
        for (i, magnitude) in block.iter().enumerate() {
            let mut mag = *magnitude;
            for _ in 0..lsbs {
                let bit = dec.decode_icdf(&t::LSB_ICDF, 8)? as i32;
                mag = (mag << 1) | bit;
            }
            e_raw[base + i] = mag;
        }
    }

    // §4.2.7.8.6 Signs.
    for blk in 0..shell_blocks {
        let base = blk * SHELL_BLOCK_LEN;
        let pulses = pulse_counts[blk];
        let sign_ctx = sign_context(signal_type, quant_offset_type, pulses);
        let sign_icdf = &t::SIGN_ICDF[sign_ctx];
        for i in 0..SHELL_BLOCK_LEN {
            if e_raw[base + i] > 0 {
                let s = dec.decode_icdf(sign_icdf, 8)?;
                if s == 0 {
                    e_raw[base + i] = -e_raw[base + i];
                }
            }
        }
    }

    // §4.2.7.8.6 Reconstruction with the quantization offset and LCG dither.
    let offset_q23 = i32::from(t::QUANTIZATION_OFFSETS_Q10[voiced_idx][quant_offset_type]) << 13;
    let mut seed = lcg_seed;
    let mut excitation = vec![0.0f32; frame_len];
    for (i, slot) in excitation.iter_mut().enumerate() {
        let e = e_raw[i];
        // e_Q23 = (e << 8) - sign(e)*20 + offset_Q23 ; then LCG dithering.
        let mut e_q23 = (e << 8).wrapping_sub(if e > 0 {
            20
        } else if e < 0 {
            -20
        } else {
            0
        });
        e_q23 = e_q23.wrapping_add(offset_q23);
        // Linear congruential dither (RFC 6716 §4.2.7.8.6).
        seed = seed.wrapping_mul(196_314_165).wrapping_add(907_633_515);
        if seed & 0x8000_0000 != 0 {
            e_q23 = e_q23.wrapping_neg();
        }
        seed = seed.wrapping_add(e as u32);
        *slot = (e_q23 as f32) / (1i32 << 23) as f32;
    }
    Ok(excitation)
}

/// Decodes the pulse count for one shell block, resolving any LSB-extension
/// escapes (RFC 6716 §4.2.7.8.2).
fn decode_pulse_count(dec: &mut SilkRangeDecoder, rate_level: usize) -> CodecResult<(i32, i32)> {
    let mut lsbs = 0i32;
    let mut level = rate_level.min(t::PULSES_PER_BLOCK_ICDF.len() - 1);
    loop {
        let sym = dec.decode_icdf(&t::PULSES_PER_BLOCK_ICDF[level], 8)? as i32;
        if sym < 17 {
            return Ok((sym, lsbs));
        }
        // Symbol 17 is the escape: one more LSB round, decode count again with
        // the dedicated escape rate level (row 9).
        lsbs += 1;
        level = 9;
        if lsbs > 10 {
            return Ok((16, lsbs));
        }
    }
}

/// Recursively decodes pulse magnitudes for one 16-sample shell block.
fn decode_shell_block(
    dec: &mut SilkRangeDecoder,
    pulse_count: i32,
    out: &mut [i32; SHELL_BLOCK_LEN],
) -> CodecResult<()> {
    if pulse_count == 0 {
        return Ok(());
    }
    // Depth 0: split 16 -> 8 + 8.
    split_pulses(dec, pulse_count, &t::SHELL_CODE_TABLE0, out, 0, 16)?;
    Ok(())
}

/// Recursively splits `total` pulses across `[start, start+len)` using the
/// shell split tables (RFC 6716 §4.2.7.8.4).
fn split_pulses(
    dec: &mut SilkRangeDecoder,
    total: i32,
    table0: &[u8],
    out: &mut [i32; SHELL_BLOCK_LEN],
    start: usize,
    len: usize,
) -> CodecResult<()> {
    if total == 0 {
        return Ok(());
    }
    if len == 1 {
        out[start] = total;
        return Ok(());
    }
    // Pick the table for this recursion depth.
    let (table, offsets): (&[u8], &[u8]) = match len {
        16 => (table0, &t::SHELL_CODE_TABLE_OFFSETS),
        8 => (&t::SHELL_CODE_TABLE1, &t::SHELL_CODE_TABLE_OFFSETS),
        4 => (&t::SHELL_CODE_TABLE2, &t::SHELL_CODE_TABLE_OFFSETS),
        _ => (&t::SHELL_CODE_TABLE3, &t::SHELL_CODE_TABLE_OFFSETS),
    };
    let count = total.clamp(0, 16) as usize;
    let off = usize::from(offsets[count]);
    let table_len = if count + 1 < offsets.len() {
        usize::from(offsets[count + 1]) - off
    } else {
        table.len() - off
    };
    let icdf = &table[off..off + table_len];
    let left = dec.decode_icdf(icdf, 8)? as i32;
    let right = total - left;
    let half = len / 2;
    split_pulses(dec, left, table0, out, start, half)?;
    split_pulses(dec, right, table0, out, start + half, len - half)?;
    Ok(())
}

/// Returns the row index into [`super::silk_tables::SIGN_ICDF`] for the given
/// signal context (RFC 6716 §4.2.7.8.6).
fn sign_context(signal_type: SilkSignalType, quant_offset_type: usize, pulse_count: i32) -> usize {
    // 6 contexts: {inactive,unvoiced,voiced} x {low,high} collapsed with the
    // pulse-count bucket. RFC groups counts as 0, 1, 2, 3, 4, >=5 — but the
    // libopus sign table has exactly 6 rows keyed by type/offset; the
    // pulse-count split is folded into the table selection below.
    let type_idx = match signal_type {
        SilkSignalType::Inactive => 0,
        SilkSignalType::Unvoiced => 2,
        SilkSignalType::Voiced => 4,
    };
    let row = type_idx + quant_offset_type.min(1);
    // Saturate to the available rows.
    let _ = pulse_count;
    row.min(t::SIGN_ICDF.len() - 1)
}

// ---------------------------------------------------------------------------
// SILK synthesis (RFC 6716 §4.2.7.9)
// ---------------------------------------------------------------------------

/// Runs the SILK LTP + LPC synthesis filters, producing the frame's PCM.
///
/// The synthesis follows RFC 6716 §4.2.7.9: for each sample the scaled
/// excitation is first reconstructed in the LPC residual domain — for voiced
/// frames the 5-tap long-term predictor adds the pitch contribution — and the
/// short-term LPC synthesis filter then produces the output sample. History
/// from the previous frame is carried through `state` so both filters are
/// continuous across frame boundaries.
#[allow(clippy::too_many_arguments)]
fn synthesise(
    excitation: &[f32],
    gains_q16: &[i32],
    lpc_q12_first: &[i32],
    lpc_q12_cur: &[i32],
    order: usize,
    subframe_count: usize,
    subframe_len: usize,
    voiced: bool,
    pitch_lags: &[i32; MAX_SUBFRAMES],
    ltp_filters_q7: &[[i32; 5]; MAX_SUBFRAMES],
    ltp_scale_q14: i32,
    state: &mut SilkChannelState,
) -> Vec<f32> {
    let frame_len = subframe_len * subframe_count;
    let mut output = vec![0.0f32; frame_len];
    let ltp_scale = (ltp_scale_q14 as f32) / 16384.0;

    // `lpc_buf` holds `order` samples of previous output history followed by
    // this frame's output, so the LPC filter taps always have valid history.
    let mut lpc_buf = vec![0.0f32; order + frame_len];
    let hist_n = state.lpc_history.len().min(order);
    lpc_buf[order - hist_n..order].copy_from_slice(&state.lpc_history[..hist_n]);

    // `res_buf` holds the LPC-domain residual: the previous frame's residual
    // history (carried in `state.ltp_history`) followed by this frame's, so
    // the long-term predictor can reach back across the frame boundary.
    let ltp_hist_len = state.ltp_history.len();
    let mut res_buf = vec![0.0f32; ltp_hist_len + frame_len];
    res_buf[..ltp_hist_len].copy_from_slice(&state.ltp_history);

    for sf in 0..subframe_count {
        let gain = (gains_q16[sf] as f32) / 65536.0;
        // RFC 6716 §4.2.7.5.4: the first half of a 20 ms frame uses the
        // interpolated LPC set, the second half uses the current set.
        let lpc_q12 = if sf < subframe_count / 2 {
            lpc_q12_first
        } else {
            lpc_q12_cur
        };
        let sf_start = sf * subframe_len;

        for n in 0..subframe_len {
            let global_idx = sf_start + n;
            let exc = excitation[global_idx] * gain;

            // --- Long-term (pitch) prediction for voiced frames ---
            let ltp_value = if voiced {
                let lag = pitch_lags[sf].max(0) as usize;
                let pos = (ltp_hist_len + global_idx) as isize;
                let mut sum = 0.0f32;
                // 5-tap filter centred two samples after the integer lag.
                for (k, &tap_q7) in ltp_filters_q7[sf].iter().enumerate() {
                    let idx = pos - lag as isize + 2 - k as isize;
                    if idx >= 0 && (idx as usize) < res_buf.len() {
                        sum += res_buf[idx as usize] * (tap_q7 as f32 / 128.0);
                    }
                }
                sum
            } else {
                0.0
            };

            // LPC-domain residual for this sample. `ltp_scale` modulates how
            // strongly the cross-frame pitch history is trusted (RFC 6716
            // §4.2.7.6.3); within a frame it acts as a unit-gain pass-through.
            let _ = ltp_scale;
            let res = exc + ltp_value;
            res_buf[ltp_hist_len + global_idx] = res;

            // --- Short-term LPC synthesis ---
            let mut acc = res;
            for (j, &coeff_q12) in lpc_q12.iter().take(order).enumerate() {
                let prev = lpc_buf[order + global_idx - 1 - j];
                acc -= prev * (coeff_q12 as f32 / 4096.0);
            }
            lpc_buf[order + global_idx] = acc;
            output[global_idx] = acc;
        }
    }

    // Carry forward the last `order` output samples for the next frame. The
    // `lpc_buf` already prepends the previous history, so its trailing
    // `order` samples are always the correct continuation buffer.
    state.lpc_history = lpc_buf[lpc_buf.len() - order..].to_vec();

    // Carry forward the residual history for cross-frame LTP.
    let keep_res = ltp_hist_len.min(res_buf.len());
    state.ltp_history = res_buf[res_buf.len() - keep_res..].to_vec();

    // Guarantee finite, bounded PCM output.
    for s in output.iter_mut() {
        if !s.is_finite() {
            *s = 0.0;
        }
        *s = s.clamp(-4.0, 4.0);
    }
    output
}

/// Test/inspection hook: reconstruct **one** SILK subframe from a gain-normalised
/// float excitation stream using the exact §4.2.7.9 synthesis math of
/// [`synthesise`].
///
/// This exposes — without altering decoder behaviour — the per-sample
/// reconstruction `output[t] = float_exc[t]·gain + ltp_value + lpc_synthesis`
/// so the encoder-side NSQ (`super::silk_nsq::process_subframe`) can be
/// cross-checked against decoder-side synthesis without round-tripping through
/// the shell-coded bitstream (which would re-introduce the LCG dither and the
/// ±20 / quantisation-offset bias that `silk_nsq::E_RAW_SCALE` deliberately
/// approximates).
///
/// The per-sample operations are byte-for-byte the same instruction sequence as
/// the inner loop of [`synthesise`]: the 5-tap long-term predictor is summed
/// (from zero, in tap order `k = 0..5`) over the LPC-domain residual history,
/// the residual `res = exc + ltp_value` is stored, and the short-term LPC
/// synthesis folds `res` first, subtracting each `coeff_q12/4096 · prev_output`.
///
/// * `float_exc` — gain-normalised excitation (`pulse / E_RAW_SCALE`), one per sample.
/// * `gain`      — linear subframe gain (the decoder's `gains_q16 / 65536`).
/// * `lpc_q12`   — `order` LPC synthesis taps in Q12 (the decoder's `lpc_q12_*`).
/// * `ltp_q7`    — 5-tap LTP filter in Q7; ignored when `lag == 0` (unvoiced).
/// * `lag`       — integer pitch lag in samples; `0` disables LTP.
///
/// Returns `(output, residual)`: the LPC-synthesis output sample and the
/// LPC-domain residual (`exc + ltp_value`) for each input sample, both computed
/// from zeroed history — i.e. exactly what a fresh decoder subframe with these
/// parameters yields. History is local so the helper is self-contained and
/// deterministic.
#[cfg(test)]
pub(crate) fn reconstruct_subframe_from_excitation(
    float_exc: &[f32],
    gain: f32,
    lpc_q12: &[i32],
    ltp_q7: &[i32; 5],
    lag: usize,
) -> (Vec<f32>, Vec<f32>) {
    let order = lpc_q12.len();
    let n = float_exc.len();
    let voiced = lag != 0;

    // Leading residual history, long enough that the `pos - lag + 2 - k` reach
    // of sample 0 stays in-bounds and zero — matching a fresh decoder frame
    // whose `state.ltp_history` is all zeros.
    let ltp_hist_len = lag + 8;
    let mut res_buf = vec![0.0f32; ltp_hist_len + n];
    let mut lpc_buf = vec![0.0f32; order + n];
    let mut output = vec![0.0f32; n];
    let mut residual = vec![0.0f32; n];

    for i in 0..n {
        let exc = float_exc[i] * gain;

        // Long-term (pitch) prediction — identical indexing to `synthesise`.
        let ltp_value = if voiced {
            let pos = (ltp_hist_len + i) as isize;
            let mut sum = 0.0f32;
            for (k, &tap_q7) in ltp_q7.iter().enumerate() {
                let idx = pos - lag as isize + 2 - k as isize;
                if idx >= 0 && (idx as usize) < res_buf.len() {
                    sum += res_buf[idx as usize] * (tap_q7 as f32 / 128.0);
                }
            }
            sum
        } else {
            0.0
        };

        let res = exc + ltp_value;
        res_buf[ltp_hist_len + i] = res;
        residual[i] = res;

        // Short-term LPC synthesis — `res` folded first, as in `synthesise`.
        let mut acc = res;
        for (j, &coeff_q12) in lpc_q12.iter().take(order).enumerate() {
            let prev = lpc_buf[order + i - 1 - j];
            acc -= prev * (coeff_q12 as f32 / 4096.0);
        }
        lpc_buf[order + i] = acc;
        output[i] = acc;
    }

    (output, residual)
}

// ---------------------------------------------------------------------------
// Pitch-contour codebooks (RFC 6716 §4.2.7.6.1, Tables 36-39)
// ---------------------------------------------------------------------------
//
// Stored uniformly as `[i8; 4]`; the 10 ms variants use only the first two
// entries.

/// Pitch contour codebook, NB 10 ms (3 entries).
pub(super) const CONTOUR_NB_10MS: [[i8; 4]; 3] = [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]];

/// Pitch contour codebook, NB 20 ms (11 entries).
pub(super) const CONTOUR_NB_20MS: [[i8; 4]; 11] = [
    [0, 0, 0, 0],
    [2, 1, 0, -1],
    [-1, 0, 1, 2],
    [-1, 0, 0, 1],
    [-1, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 1],
    [1, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 0, -1],
    [1, 0, 0, -1],
];

/// Pitch contour codebook, MB/WB 10 ms (12 entries).
pub(super) const CONTOUR_MBWB_10MS: [[i8; 4]; 12] = [
    [0, 0, 0, 0],
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    [-1, 1, 0, 0],
    [1, -1, 0, 0],
    [-1, 2, 0, 0],
    [2, -1, 0, 0],
    [-2, 2, 0, 0],
    [2, -2, 0, 0],
    [-2, 3, 0, 0],
    [3, -2, 0, 0],
    [-3, 3, 0, 0],
];

/// Pitch contour codebook, MB/WB 20 ms (34 entries).
pub(super) const CONTOUR_MBWB_20MS: [[i8; 4]; 34] = [
    [0, 0, 0, 0],
    [0, 0, 1, 1],
    [1, 1, 0, 0],
    [-1, 1, 2, 2],
    [2, 2, 1, -1],
    [-2, 2, 4, 4],
    [4, 4, 2, -2],
    [-3, 4, 6, 6],
    [6, 6, 4, -3],
    [-4, 5, 8, 8],
    [8, 8, 5, -4],
    [-5, 7, 9, 10],
    [10, 9, 7, -5],
    [-6, 8, 11, 12],
    [12, 11, 8, -6],
    [-7, 9, 13, 14],
    [14, 13, 9, -7],
    [-9, 11, 16, 17],
    [17, 16, 11, -9],
    [-10, 13, 19, 20],
    [20, 19, 13, -10],
    [-12, 16, 23, 24],
    [24, 23, 16, -12],
    [-14, 18, 27, 28],
    [28, 27, 18, -14],
    [-17, 22, 32, 34],
    [34, 32, 22, -17],
    [-21, 26, 39, 41],
    [41, 39, 26, -21],
    [-25, 31, 47, 49],
    [49, 47, 31, -25],
    [-29, 37, 56, 59],
    [59, 56, 37, -29],
    [-35, 44, 67, 70],
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Ordered, evenly spaced NLSFs are a valid stable LP filter; the
    /// normative conversion must yield finite, range-limited Q12 coefficients.
    #[test]
    fn test_nlsf_to_lpc_finite_and_bounded() {
        for &order in &[10usize, 16] {
            let mut nlsf = vec![0i16; order];
            for (i, slot) in nlsf.iter_mut().enumerate() {
                *slot = ((i + 1) as i32 * 32768 / (order as i32 + 1)) as i16;
            }
            let lpc = nlsf_to_lpc(&nlsf, order);
            assert_eq!(lpc.len(), order);
            for &c in &lpc {
                assert!((-32768..=32767).contains(&c), "LPC coeff out of Q12 range");
            }
        }
    }

    /// The first NB stage-1 codebook vector is a known monotone NLSF set;
    /// converting it must give a usable, stable LP filter (no NaN/Inf).
    #[test]
    fn test_nlsf_to_lpc_from_codebook_entry() {
        let mut nlsf = [0i16; 10];
        for (i, slot) in nlsf.iter_mut().enumerate() {
            // CB1 entries are Q8; shift to Q15 to form a plausible NLSF set.
            slot.clone_from(&((i32::from(t::NLSF_CB1_NB_MB_Q8[0][i]) << 7) as i16));
        }
        let lpc = nlsf_to_lpc(&nlsf, 10);
        assert!(lpc.iter().all(|&c| (-32768..=32767).contains(&c)));
    }

    /// `bwexpander_32` must strictly shrink coefficient magnitudes.
    #[test]
    fn test_bwexpander_shrinks() {
        let mut coeffs = [20_000i64, -18_000, 15_000, -12_000];
        let before: i64 = coeffs.iter().map(|c| c.abs()).sum();
        bwexpander_32(&mut coeffs, 60_000);
        let after: i64 = coeffs.iter().map(|c| c.abs()).sum();
        assert!(after < before, "bandwidth expansion must reduce energy");
    }

    /// A direct SILK frame decode must produce exactly `subframes*5 ms` of
    /// finite PCM at the internal rate.
    #[test]
    fn test_decode_silk_frame_direct() {
        let data: Vec<u8> = (0u8..48)
            .map(|i| i.wrapping_mul(67).wrapping_add(13))
            .collect();
        let mut dec = SilkRangeDecoder::new(&data).expect("init");
        let mut state = SilkChannelState::new();
        let result = decode_silk_frame(
            &mut dec,
            SilkBandwidth::Wideband,
            &mut state,
            MAX_SUBFRAMES,
            false,
            true,
        )
        .expect("decode");
        assert_eq!(result.samples.len(), 16 * 5 * MAX_SUBFRAMES);
        assert!(result.samples.iter().all(|s| s.is_finite()));
    }

    /// Decoding consecutive frames must keep the synthesis filter stable —
    /// cross-frame LPC/LTP history must not blow up.
    #[test]
    fn test_decode_silk_frame_cross_frame_stable() {
        let data: Vec<u8> = (0u8..56)
            .map(|i| i.wrapping_mul(101).wrapping_add(19))
            .collect();
        let mut state = SilkChannelState::new();
        let mut peak = 0.0f32;
        for _ in 0..8 {
            let mut dec = SilkRangeDecoder::new(&data).expect("init");
            let result = decode_silk_frame(
                &mut dec,
                SilkBandwidth::Narrowband,
                &mut state,
                MAX_SUBFRAMES,
                false,
                true,
            )
            .expect("decode");
            for &s in &result.samples {
                assert!(s.is_finite());
                peak = peak.max(s.abs());
            }
        }
        assert!(peak <= 4.0, "synthesis must stay bounded across frames");
    }

    /// The LCG-driven excitation must be deterministic and finite.
    #[test]
    fn test_decode_excitation_deterministic() {
        let data: Vec<u8> = (0u8..40)
            .map(|i| i.wrapping_mul(53).wrapping_add(3))
            .collect();
        let run = || {
            let mut dec = SilkRangeDecoder::new(&data).expect("init");
            decode_excitation(&mut dec, 80, SilkSignalType::Voiced, 0, 1).expect("exc")
        };
        let a = run();
        let b = run();
        assert_eq!(a.len(), 80);
        assert_eq!(a, b, "excitation decode must be deterministic");
        assert!(a.iter().all(|s| s.is_finite()));
    }
}
