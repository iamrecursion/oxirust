//! SILK excitation analysis and bitstream encoding (RFC 6716 §4.2.7.8).
//!
//! This module handles the open-loop (non-NSQ) excitation path and the shell-
//! coded pulse bitstream.  The NSQ closed-loop path lives in [`super::silk_nsq`]
//! and replaces `compute_excitation` when enabled.

use crate::CodecResult;

use super::silk_decoder::{log_gain_to_linear_q16, SilkBandwidth, SilkSignalType};
use super::silk_encoder::{EncoderChannelState, MAX_SUBFRAMES};
use super::silk_ltp::{lpc_residual_signal, LtpQuantised, EXC_MAX_MAGNITUDE};
use super::silk_range_encoder::SilkRangeEncoder;
use super::silk_tables as t;

/// Shell-block length in samples.
const SHELL_BLOCK_LEN: usize = 16;

// ---------------------------------------------------------------------------
// Frame-type / gain / NLSF helpers used by the encoder coordinator
// ---------------------------------------------------------------------------

pub(super) fn signal_type_index(signal_type: SilkSignalType) -> usize {
    match signal_type {
        SilkSignalType::Inactive => 0,
        SilkSignalType::Unvoiced => 1,
        SilkSignalType::Voiced => 2,
    }
}

/// Emits the frame-type symbol (RFC 6716 §4.2.7.3).
pub(super) fn encode_frame_type(
    enc: &mut SilkRangeEncoder,
    vad_flag: bool,
    signal_type: SilkSignalType,
    quant_offset_type: usize,
) -> CodecResult<()> {
    if !vad_flag {
        enc.encode_icdf(quant_offset_type & 1, &t::TYPE_OFFSET_NO_VAD_ICDF, 8)?;
    } else {
        let voiced = signal_type == SilkSignalType::Voiced;
        let sym = if voiced {
            2 + (quant_offset_type & 1)
        } else {
            quant_offset_type & 1
        };
        enc.encode_icdf(sym, &t::TYPE_OFFSET_VAD_ICDF, 8)?;
    }
    Ok(())
}

/// Emits the per-subframe gain indices (RFC 6716 §4.2.7.4) and updates
/// `gain_index` in place to the *reconstructed* values the decoder will see.
pub(super) fn encode_gains(
    enc: &mut SilkRangeEncoder,
    gain_index: &mut [i32],
    type_index: usize,
    have_prev_frame: bool,
    prev_gain_index: i32,
) -> CodecResult<()> {
    let mut prev = prev_gain_index;
    for sf in 0..gain_index.len() {
        let g = gain_index[sf];
        if sf == 0 {
            let g_clamped = if have_prev_frame {
                g.max(prev - 16).max(0)
            } else {
                g
            }
            .clamp(0, 63);
            let msb = (g_clamped >> 3) as usize;
            let lsb = (g_clamped & 7) as usize;
            enc.encode_icdf(msb, &t::GAIN_ICDF[type_index], 8)?;
            enc.encode_icdf(lsb, &t::UNIFORM8_ICDF, 8)?;
            prev = g_clamped;
            gain_index[sf] = g_clamped;
        } else {
            let step = g - prev;
            let delta = if step < 12 {
                (step + 4).clamp(0, 15)
            } else {
                ((step + 20) / 2).clamp(16, 40)
            };
            enc.encode_icdf(delta as usize, &t::DELTA_GAIN_ICDF, 8)?;
            let reconstructed_step = if delta < 16 {
                delta - 4
            } else {
                2 * delta - 20
            };
            let new_index = (prev + reconstructed_step).clamp(0, 63);
            prev = new_index;
            gain_index[sf] = new_index;
        }
    }
    Ok(())
}

/// Emits the NLSF stage-1 codebook index plus the chosen stage-2 residual
/// indices (RFC 6716 §4.2.7.5.1 / §4.2.7.5.2).
pub(super) fn encode_nlsf_full(
    enc: &mut SilkRangeEncoder,
    bw: SilkBandwidth,
    signal_type: SilkSignalType,
    decision: &super::silk_lpc::NlsfDecision,
) -> CodecResult<()> {
    let order = bw.lpc_order();
    let voiced = signal_type == SilkSignalType::Voiced;
    let voiced_idx = usize::from(voiced);
    let i1 = decision.i1;

    let stage1_icdf: &[u8] = if bw.is_wideband() {
        &t::NLSF_CB1_ICDF_WB[voiced_idx]
    } else {
        &t::NLSF_CB1_ICDF_NB_MB[voiced_idx]
    };
    enc.encode_icdf(i1, stage1_icdf, 8)?;

    let select_table: &[u8] = if bw.is_wideband() {
        &t::NLSF_CB2_SELECT_WB
    } else {
        &t::NLSF_CB2_SELECT_NB_MB
    };
    let mut residual_cb = vec![0usize; order];
    for pair in 0..order / 2 {
        let entry = select_table[i1 * (order / 2) + pair];
        let even = 2 * pair;
        let odd = even + 1;
        residual_cb[even] = usize::from((entry >> 1) & 0x07);
        residual_cb[odd] = usize::from((entry >> 5) & 0x07);
    }

    let nlsf_ext_max = (t::NLSF_EXT_ICDF.len() - 1) as i32;
    for coeff in 0..order {
        let icdf: &[u8] = if bw.is_wideband() {
            &t::NLSF_CB2_ICDF_WB[residual_cb[coeff]]
        } else {
            &t::NLSF_CB2_ICDF_NB_MB[residual_cb[coeff]]
        };
        let value = decision.res_idx[coeff];
        if value >= 4 {
            enc.encode_icdf(8, icdf, 8)?;
            let ext = (value - 4).clamp(0, nlsf_ext_max) as usize;
            enc.encode_icdf(ext, &t::NLSF_EXT_ICDF, 8)?;
        } else if value <= -4 {
            enc.encode_icdf(0, icdf, 8)?;
            let ext = (-value - 4).clamp(0, nlsf_ext_max) as usize;
            enc.encode_icdf(ext, &t::NLSF_EXT_ICDF, 8)?;
        } else {
            let sym = (value + 4) as usize;
            enc.encode_icdf(sym, icdf, 8)?;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Excitation analysis (open-loop, pre-NSQ path)
// ---------------------------------------------------------------------------

/// Computes the LPC residual signal and gain-normalises it to `[-1, 1]`.
///
/// This is the **open-loop** excitation path.  When NSQ is wired in,
/// [`compute_excitation_nsq`] in `silk_nsq.rs` replaces this function for the
/// actual encoder output; this function is still used by the gain analysis step.
#[allow(clippy::too_many_arguments)]
pub(super) fn compute_excitation(
    samples: &[f32],
    lpc_q12: &[i32],
    order: usize,
    gain_index: &[i32; MAX_SUBFRAMES],
    subframe_count: usize,
    subframe_len: usize,
    voiced: bool,
    ltp: &LtpQuantised,
    history: &[f32],
) -> Vec<f32> {
    let frame_len = subframe_count * subframe_len;
    let mut out = vec![0.0f32; frame_len];
    let lpc_residual = lpc_residual_signal(samples, lpc_q12, order, history);
    for sf in 0..subframe_count {
        let gain = log_gain_to_linear_q16(gain_index[sf]) as f32 / 65536.0;
        let inv_gain = if gain.abs() > 1e-9 { 1.0 / gain } else { 0.0 };
        for n in 0..subframe_len {
            let idx = sf * subframe_len + n;
            let mut res = lpc_residual[idx];
            if voiced {
                let lag = ltp.pitch_lags[sf].max(1) as isize;
                let mut ltp_pred = 0.0f32;
                for k in 0..5usize {
                    let src = idx as isize - lag + 2 - k as isize;
                    if src >= 0 && (src as usize) < lpc_residual.len() {
                        ltp_pred +=
                            lpc_residual[src as usize] * (ltp.filters_q7[sf][k] as f32 / 128.0);
                    }
                }
                res -= ltp_pred;
            }
            out[idx] = (res * inv_gain).clamp(-1.0, 1.0);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Shell-coded excitation bitstream emission (RFC 6716 §4.2.7.8)
// ---------------------------------------------------------------------------

/// Quantises the gain-normalised excitation into shell pulses and emits the
/// full §4.2.7.8 symbol sequence (rate level, pulse counts, shell blocks,
/// LSBs, signs).
pub(super) fn encode_excitation(
    enc: &mut SilkRangeEncoder,
    excitation: &[f32],
    signal_type: SilkSignalType,
    quant_offset_type: usize,
    lcg_seed: u32,
) -> CodecResult<()> {
    let voiced = signal_type == SilkSignalType::Voiced;
    let voiced_idx = usize::from(voiced);

    // Rate level 6: gives non-zero-pulse shell blocks the dominant probability.
    let rate_level: usize = 6;
    enc.encode_icdf(rate_level, &t::RATE_LEVELS_ICDF[voiced_idx], 8)?;

    let frame_len = excitation.len();
    let shell_blocks = frame_len.div_ceil(SHELL_BLOCK_LEN);

    let offset_q23 = i32::from(t::QUANTIZATION_OFFSETS_Q10[voiced_idx][quant_offset_type]) << 13;
    let mut seed = lcg_seed;
    let mut e_raw = vec![0i32; shell_blocks * SHELL_BLOCK_LEN];
    for (i, &x) in excitation.iter().enumerate() {
        seed = seed.wrapping_mul(196_314_165).wrapping_add(907_633_515);
        let will_flip = (seed & 0x8000_0000) != 0;
        let target_q23_f = f64::from(x) * f64::from(1i32 << 23);
        let pre_flip_q23 = if will_flip {
            -target_q23_f
        } else {
            target_q23_f
        };
        let e_q23_centered = pre_flip_q23 - f64::from(offset_q23);
        let mag = (e_q23_centered.abs() / 256.0).round() as i64;
        let signed_e = if e_q23_centered >= 0.0 { mag } else { -mag };
        let clamped = signed_e.clamp(-2047, 2047) as i32;
        e_raw[i] = clamped;
        seed = seed.wrapping_add(clamped as u32);
    }

    let mut pulse_counts = vec![0i32; shell_blocks];
    let mut lsb_counts = vec![0i32; shell_blocks];
    let mut shell_mags = vec![[0i32; SHELL_BLOCK_LEN]; shell_blocks];
    for blk in 0..shell_blocks {
        let base = blk * SHELL_BLOCK_LEN;
        let mut lsbs = 0i32;
        let mut total;
        loop {
            total = 0i32;
            for i in 0..SHELL_BLOCK_LEN {
                total += e_raw[base + i].abs() >> lsbs;
            }
            if total <= 16 || lsbs >= 16 {
                break;
            }
            lsbs += 1;
        }
        lsb_counts[blk] = lsbs;
        for i in 0..SHELL_BLOCK_LEN {
            shell_mags[blk][i] = e_raw[base + i].abs() >> lsbs;
        }
        if total > 16 {
            let scale_num = 16i64;
            let scale_den = i64::from(total);
            let mut new_total = 0i32;
            for i in 0..SHELL_BLOCK_LEN {
                let m = ((i64::from(shell_mags[blk][i]) * scale_num) / scale_den) as i32;
                shell_mags[blk][i] = m;
                new_total += m;
            }
            total = new_total;
        }
        pulse_counts[blk] = total.min(16);
    }

    // --- §4.2.7.8.2 emit pulse counts ---
    for blk in 0..shell_blocks {
        let pulses = pulse_counts[blk];
        let lsbs = lsb_counts[blk];
        let mut level = rate_level.min(t::PULSES_PER_BLOCK_ICDF.len() - 1);
        let mut remaining_lsbs = lsbs;
        while remaining_lsbs > 0 {
            enc.encode_icdf(17, &t::PULSES_PER_BLOCK_ICDF[level], 8)?;
            remaining_lsbs -= 1;
            level = 9;
        }
        let pc = pulses.clamp(0, 16) as usize;
        enc.encode_icdf(pc, &t::PULSES_PER_BLOCK_ICDF[level], 8)?;
    }

    // --- §4.2.7.8.4 / §4.2.7.8.5 shell-encode the pulse magnitudes ---
    for blk in 0..shell_blocks {
        let pulses = pulse_counts[blk] as i32;
        if pulses > 0 {
            split_pulses_encode(enc, &shell_mags[blk], 0, SHELL_BLOCK_LEN, pulses, 16)?;
        }
        let lsbs = lsb_counts[blk] as u32;
        if lsbs > 0 {
            for i in 0..SHELL_BLOCK_LEN {
                let abs_val = e_raw[blk * SHELL_BLOCK_LEN + i].unsigned_abs() as u32;
                for k in (0..lsbs).rev() {
                    let bit = (abs_val >> k) & 1;
                    enc.encode_icdf(bit as usize, &t::LSB_ICDF, 8)?;
                }
            }
        }
    }

    // --- §4.2.7.8.6 signs ---
    for blk in 0..shell_blocks {
        let sign_ctx = sign_context(signal_type, quant_offset_type, pulse_counts[blk]);
        let sign_icdf: &[u8] = &t::SIGN_ICDF[sign_ctx];
        for i in 0..SHELL_BLOCK_LEN {
            let m = e_raw[blk * SHELL_BLOCK_LEN + i];
            if m != 0 {
                let s = if m > 0 { 1 } else { 0 };
                enc.encode_icdf(s, sign_icdf, 8)?;
            }
        }
    }

    Ok(())
}

/// Recursively splits `total` pulses across `[start, start+len)` using the
/// shell pulse-split tables, emitting the matching iCDF symbol per recursion
/// depth (mirror of `split_pulses` in the decoder).
fn split_pulses_encode(
    enc: &mut SilkRangeEncoder,
    mags: &[i32; SHELL_BLOCK_LEN],
    start: usize,
    len: usize,
    total: i32,
    depth_len: usize,
) -> CodecResult<()> {
    if total == 0 || len == 1 {
        return Ok(());
    }
    let half = len / 2;
    let mut left_sum = 0i32;
    for i in 0..half {
        left_sum += mags[start + i];
    }
    let right_sum = total - left_sum;
    let (table, offsets): (&[u8], &[u8]) = match depth_len {
        16 => (&t::SHELL_CODE_TABLE0, &t::SHELL_CODE_TABLE_OFFSETS),
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
    let left = left_sum.clamp(0, total) as usize;
    enc.encode_icdf(left, icdf, 8)?;
    split_pulses_encode(enc, mags, start, half, left_sum, half)?;
    split_pulses_encode(enc, mags, start + half, len - half, right_sum, half)?;
    Ok(())
}

/// Returns the row index into [`super::silk_tables::SIGN_ICDF`] for the
/// given signal context.  Matches the decoder's `sign_context`.
pub(super) fn sign_context(
    signal_type: SilkSignalType,
    quant_offset_type: usize,
    _pulse_count: i32,
) -> usize {
    let type_idx = match signal_type {
        SilkSignalType::Inactive => 0,
        SilkSignalType::Unvoiced => 2,
        SilkSignalType::Voiced => 4,
    };
    let row = type_idx + quant_offset_type.min(1);
    row.min(t::SIGN_ICDF.len() - 1)
}
