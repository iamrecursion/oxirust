//! SILK frame encoder (RFC 6716 §4.2) — coordinator module.
//!
//! This module owns the top-level encode pipeline for a single SILK frame:
//! pre-emphasis, LP analysis, gain/NLSF/LTP blocks, and excitation encoding.
//! The heavy-lifting sub-functions live in the sibling modules:
//!
//! * [`super::silk_lpc`] — Levinson-Durbin, NLSF quantisation.
//! * [`super::silk_ltp`] — Pitch search, LTP block encoder, gain analysis.
//! * [`super::silk_excitation`] — Open-loop excitation, shell-coded emission.
//! * [`super::silk_nsq`] — Closed-loop NSQ excitation (replaces open-loop).

use crate::{CodecError, CodecResult};

use super::silk_decoder::{nlsf_to_lpc, SilkBandwidth, SilkSignalType};
use super::silk_excitation::{
    compute_excitation, encode_excitation, encode_frame_type, encode_gains, encode_nlsf_full,
    signal_type_index,
};
use super::silk_lpc::quantise_nlsf_full;
use super::silk_ltp::{analyse_gains, analyse_pitch, encode_ltp_block, lpc_residual_signal};
use super::silk_nsq::{process_subframe, NsqState};
use super::silk_range_encoder::SilkRangeEncoder;
use super::silk_tables as t;

/// Maximum LPC order across all SILK bandwidths.
pub const MAX_LPC_ORDER: usize = 16;

/// Maximum subframes per SILK frame.
pub const MAX_SUBFRAMES: usize = 4;

/// Pre-emphasis state carried across encoder calls (RFC 6716 §4.2.7.1).
#[derive(Debug, Default, Clone)]
pub(super) struct PreEmphasisState {
    /// Last sample seen by the pre-emphasis filter.
    pub(super) last_sample: f32,
}

impl PreEmphasisState {
    /// Applies the per-sample pre-emphasis filter `y[n] = x[n] - 0.5*x[n-1]`.
    fn apply(&mut self, input: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(input.len());
        let mut prev = self.last_sample;
        for &s in input {
            out.push(s - 0.5 * prev);
            prev = s;
        }
        self.last_sample = prev;
        out
    }
}

/// Persistent SILK encoder channel state.
#[derive(Debug, Default, Clone)]
pub struct EncoderChannelState {
    /// Pre-emphasis filter state.
    pub(super) pre: PreEmphasisState,
    /// Previous-frame quantised NLSF (Q15), for delta interpolation.
    pub prev_nlsf_q15: Vec<i16>,
    /// True after the first frame has been encoded.
    pub have_prev_frame: bool,
    /// Last frame's last gain index.
    pub prev_gain_index: i32,
    /// Previous frame was emitted as voiced.
    pub prev_voiced: bool,
    /// Previous frame's primary pitch lag.
    pub prev_pitch_lag: i32,
    /// Trailing `MAX_LPC_ORDER` samples from the previous frame, used as
    /// LPC history when computing the forward (residual) filter on the
    /// current frame.
    pub lpc_history: Vec<f32>,
    /// NSQ state, one per subframe budget (lazy init on first voiced frame).
    pub nsq_state: Option<NsqState>,
}

/// Encodes one SILK frame for one channel into the provided range encoder.
///
/// Returns the signal type written.
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_silk_frame(
    enc: &mut SilkRangeEncoder,
    bw: SilkBandwidth,
    state: &mut EncoderChannelState,
    input: &[f32],
    subframe_count: usize,
    vad_flag: bool,
) -> CodecResult<SilkSignalType> {
    let order = bw.lpc_order();
    let subframe_len = bw.khz() * 5;
    let frame_len = subframe_len * subframe_count;
    if input.len() < frame_len {
        return Err(CodecError::InvalidData(format!(
            "SILK encoder needs {frame_len} samples, got {}",
            input.len()
        )));
    }

    // --- §4.2.7.1 Pre-emphasis ---
    let _ = state.pre.apply(&input[..frame_len]);
    let preemph = input[..frame_len].to_vec();

    // --- §4.2.7.5 LP analysis -> NLSF -> stage-1 + stage-2 ---
    let nlsf_decision = quantise_nlsf_full(&preemph, bw)?;
    let lpc_q12 = nlsf_to_lpc(&nlsf_decision.nlsf_q15, order);

    // --- §4.2.7.6 Pitch search ---
    let pitch_decision = if vad_flag {
        analyse_pitch(&preemph, bw, subframe_count, state)
    } else {
        super::silk_ltp::PitchDecision::unvoiced()
    };
    let voiced = vad_flag && pitch_decision.voiced;

    // --- §4.2.7.4 Gain analysis ---
    let prev_history: Vec<f32> = if state.lpc_history.len() == order {
        state.lpc_history.clone()
    } else {
        vec![0.0; order]
    };
    let lpc_residual_full = lpc_residual_signal(&preemph, &lpc_q12, order, &prev_history);
    // For voiced frames compute the post-LTP residual so the gain accurately
    // reflects what the excitation budget needs to cover (the LTP substantially
    // reduces the residual for periodic signals).
    let gain_residual = if voiced {
        let lag = pitch_decision.primary_lag.max(1) as usize;
        let n = lpc_residual_full.len();
        let mut post_ltp = lpc_residual_full.clone();
        // Compute the optimal 1-tap LTP gain at the primary lag.
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for i in lag..n {
            let a = f64::from(lpc_residual_full[i]);
            let b = f64::from(lpc_residual_full[i - lag]);
            num += a * b;
            den += b * b;
        }
        let optimal_gain = if den > 1e-12 {
            (num / den).clamp(0.0, 0.99)
        } else {
            0.0
        };
        // The SILK LTP codebook (periodicity 1) can represent gains up to ~0.88
        // (best codebook entry for pure tones ≈ [0,0,101/128,0,0] = 0.79).
        // Cap the effective gain at 0.75 to account for codebook quantization,
        // then clamp to [0.1, 0.75] to avoid over/under-predicting.
        let effective_gain = (optimal_gain * 0.80).clamp(0.05, 0.85) as f32;
        if effective_gain > 0.05 {
            for i in lag..n {
                let pred = lpc_residual_full[i - lag] * effective_gain;
                post_ltp[i] = lpc_residual_full[i] - pred;
            }
        }
        post_ltp
    } else {
        lpc_residual_full.clone()
    };
    let gain_index = analyse_gains(&gain_residual, subframe_count, subframe_len, voiced);

    // --- §4.2.7.3 Frame type ---
    let signal_type = if !vad_flag {
        SilkSignalType::Inactive
    } else if voiced {
        SilkSignalType::Voiced
    } else {
        SilkSignalType::Unvoiced
    };
    let quant_offset_type: usize = 0;
    encode_frame_type(enc, vad_flag, signal_type, quant_offset_type)?;

    // --- §4.2.7.4 Gains ---
    let type_index = signal_type_index(signal_type);
    let mut gain_index_recon = gain_index;
    encode_gains(
        enc,
        &mut gain_index_recon[..subframe_count],
        type_index,
        state.have_prev_frame,
        state.prev_gain_index,
    )?;
    state.prev_gain_index = gain_index_recon[subframe_count - 1];

    // --- §4.2.7.5 NLSF stage-1 + stage-2 residuals ---
    encode_nlsf_full(enc, bw, signal_type, &nlsf_decision)?;

    // --- §4.2.7.5.4 Interpolation factor (20 ms frames only) ---
    if subframe_count == MAX_SUBFRAMES {
        enc.encode_icdf(4, &t::NLSF_INTERP_ICDF, 8)?;
    }

    // --- §4.2.7.6 LTP block (voiced frames only) ---
    let mut ltp_quantised = super::silk_ltp::LtpQuantised::default();
    if voiced {
        ltp_quantised = encode_ltp_block(
            enc,
            bw,
            subframe_count,
            state,
            &pitch_decision,
            &lpc_residual_full,
        )?;
    }

    // --- §4.2.7.7 LCG seed ---
    let lcg_seed: u32 = 0;
    enc.encode_icdf(lcg_seed as usize, &t::UNIFORM4_ICDF, 8)?;

    // --- §4.2.7.8 Excitation (NSQ closed-loop path) ---
    // Build or re-use the NSQ state.
    if state.nsq_state.is_none() {
        let ltp_max_lag = 288usize; // max WB pitch lag
        let mut new_nsq = NsqState::new(order, ltp_max_lag);
        // Warm-up: initialise slpc from the encoder's LPC history so the
        // closed-loop prediction starts from a coherent state rather than zeros.
        let hist_len = state.lpc_history.len().min(order);
        for k in 0..hist_len {
            // lpc_history[order-1] = oldest, [order-1-k] = k+1 steps ago.
            new_nsq.slpc[k] = state.lpc_history[order - 1 - k];
        }
        state.nsq_state = Some(new_nsq);
    }
    let nsq = state.nsq_state.as_mut().expect("just initialised");

    // Derive LTP filter for NSQ (5-tap, may be zero for unvoiced).
    let ltp_coeffs: [f32; 5] = if voiced {
        // Use the first subframe's quantised taps (same for all subframes
        // in this encoder; correct enough for NSQ shaping).
        let q7 = ltp_quantised.filters_q7[0];
        [
            q7[0] as f32 / 128.0,
            q7[1] as f32 / 128.0,
            q7[2] as f32 / 128.0,
            q7[3] as f32 / 128.0,
            q7[4] as f32 / 128.0,
        ]
    } else {
        [0.0; 5]
    };

    // Convert Q12 LPC to f32.
    let lpc_f32: Vec<f32> = lpc_q12.iter().map(|&c| c as f32 / 4096.0).collect();

    // NSQ produces the gain-normalised excitation (one concatenated slice
    // for all subframes); the existing shell-coder is unmodified.
    let mut nsq_excitation = Vec::with_capacity(frame_len);
    for sf in 0..subframe_count {
        let sf_start = sf * subframe_len;
        let sf_end = sf_start + subframe_len;
        let sf_slice = &preemph[sf_start..sf_end];
        let gain =
            super::silk_decoder::log_gain_to_linear_q16(gain_index_recon[sf]) as f32 / 65536.0;
        let ltp_lag = if voiced {
            ltp_quantised.pitch_lags[sf] as usize
        } else {
            0
        };
        // NSQ returns gain-normalised float excitation in [-1, 1] directly.
        let subframe_exc = process_subframe(
            sf_slice,
            &lpc_f32,
            &ltp_coeffs,
            ltp_lag,
            gain,
            signal_type,
            nsq,
        );
        nsq_excitation.extend(subframe_exc);
    }
    encode_excitation(
        enc,
        &nsq_excitation,
        signal_type,
        quant_offset_type,
        lcg_seed,
    )?;

    // --- Update persistent state ---
    state.prev_nlsf_q15 = nlsf_decision.nlsf_q15.clone();
    state.have_prev_frame = true;
    state.prev_voiced = voiced;
    if voiced {
        state.prev_pitch_lag = ltp_quantised.pitch_lags[subframe_count - 1];
    }
    let history_start = frame_len.saturating_sub(order);
    state.lpc_history = preemph[history_start..frame_len].to_vec();
    if state.lpc_history.len() < order {
        let pad = order - state.lpc_history.len();
        let mut padded = vec![0.0f32; pad];
        padded.extend_from_slice(&state.lpc_history);
        state.lpc_history = padded;
    }
    Ok(signal_type)
}

#[cfg(test)]
mod tests {
    use super::super::silk_excitation::sign_context;
    use super::super::silk_lpc::{levinson_durbin, quantise_nlsf_full, NLSF_RES_MAX};
    use super::super::silk_ltp::analyse_gains;
    use super::super::silk_range::SilkRangeDecoder;
    use super::*;
    use crate::opus::silk_decoder::{SilkBandwidth, SilkSignalType};

    /// Reproduce the exact header symbol sequence the SILK decoder expects
    /// (VAD flag, LBRR flag, frame type) and decode it back.
    #[test]
    fn test_header_transcript_roundtrip() {
        let mut enc = SilkRangeEncoder::new();
        enc.encode_bit_logp(true, 1).expect("vad");
        enc.encode_bit_logp(false, 1).expect("lbrr");
        enc.encode_icdf(0, &t::TYPE_OFFSET_VAD_ICDF, 8)
            .expect("type");
        for _ in 0..16 {
            enc.encode_bit_logp(false, 1).expect("pad");
        }
        let bytes = enc.finish().expect("finish");

        let mut dec = SilkRangeDecoder::new(&bytes).expect("dec");
        let vad = dec.decode_bit_logp(1).expect("decode vad");
        let lbrr = dec.decode_bit_logp(1).expect("decode lbrr");
        let frame_type = dec
            .decode_icdf(&t::TYPE_OFFSET_VAD_ICDF, 8)
            .expect("decode type");
        assert!(vad);
        assert!(!lbrr);
        assert_eq!(frame_type, 0);
    }

    /// Levinson-Durbin must reproduce the reflection coefficients of a known
    /// autocorrelation (here a simple AR(1) with k = 0.7).
    #[test]
    fn test_levinson_ar1() {
        let r: Vec<f64> = (0..=4).map(|k| 0.7f64.powi(k as i32)).collect();
        let a = levinson_durbin(&r, 4);
        assert!((a[0] - 0.7).abs() < 0.05, "a[0] = {}", a[0]);
        for &v in &a[1..] {
            assert!(v.abs() < 0.2);
        }
    }

    /// Stage-1 quantisation must return a valid codebook index.
    #[test]
    fn test_stage1_quantise_in_range() {
        use super::super::silk_lpc::lpc_to_nlsf;
        let samples: Vec<f32> = (0..160)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin() * 0.5)
            .collect();
        let i1 = {
            let bw = SilkBandwidth::Wideband;
            let order = bw.lpc_order();
            use super::super::silk_lpc::autocorrelation;
            let r = autocorrelation(&samples, order);
            let _a = levinson_durbin(&r, order);
            // Just ensure quantise_nlsf_full doesn't panic.
            quantise_nlsf_full(&samples, bw).expect("ok").i1
        };
        assert!(i1 < 32);
    }

    /// Gain analysis returns valid 6-bit indices.
    #[test]
    fn test_analyse_gains_in_range() {
        let samples = vec![0.5f32; 320];
        let g = analyse_gains(&samples, 4, 80, false);
        for &v in &g {
            assert!((0..64).contains(&v));
        }
    }

    /// The reconstructed NLSF from `quantise_nlsf_full` must be
    /// monotone-increasing and lie within `[0, 32767]`.
    #[test]
    fn test_nlsf_full_monotone_for_tone() {
        let samples: Vec<f32> = (0..320)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin() * 0.5)
            .collect();
        let decision = quantise_nlsf_full(&samples, SilkBandwidth::Wideband).expect("ok");
        for (i, &v) in decision.nlsf_q15.iter().enumerate() {
            assert!(
                (0..=32767).contains(&i32::from(v)),
                "out of range at {i}: {v}"
            );
            if i > 0 {
                assert!(
                    v > decision.nlsf_q15[i - 1],
                    "NLSF must be strictly increasing: idx {} got {} prev {}",
                    i,
                    v,
                    decision.nlsf_q15[i - 1],
                );
            }
        }
        let lpc = nlsf_to_lpc(&decision.nlsf_q15, 16);
        for (i, &c) in lpc.iter().enumerate() {
            assert!(
                (-32768..=32767).contains(&c),
                "LPC coeff {i} = {c} out of Q12 range"
            );
        }
    }

    /// Stage-2 residuals must lie within the encodable range.
    #[test]
    fn test_nlsf_residuals_in_encodable_range() {
        let samples: Vec<f32> = (0..320)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin() * 0.5)
            .collect();
        let decision = quantise_nlsf_full(&samples, SilkBandwidth::Wideband).expect("ok");
        for &r in &decision.res_idx {
            assert!(
                (-NLSF_RES_MAX..=NLSF_RES_MAX).contains(&r),
                "residual {r} outside encodable range",
            );
        }
    }

    /// Diagnostic: end-to-end round trip through the public SILK encoder and
    /// decoder.  Measure peak/rms/SNR.
    #[test]
    fn test_silk_full_pipeline_diagnostic() {
        use super::super::packet::OpusBandwidth;
        use super::super::silk::{SilkDecoder, SilkEncoder};
        const SR: u32 = 16000;
        const FRAME: usize = 320;
        let mut encoder = SilkEncoder::new(SR, 1, OpusBandwidth::Wideband);
        let mut decoder = SilkDecoder::new(SR, 1, OpusBandwidth::Wideband);

        let silence = vec![0.0f32; FRAME];
        let mut buf = vec![0u8; 4096];
        let mut out = vec![0.0f32; FRAME];
        for _ in 0..6 {
            let _ = encoder.encode(&silence, &mut buf, FRAME);
        }
        let input: Vec<f32> = (0..FRAME * 8)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / (SR as f32)).sin() * 0.5)
            .collect();
        for k in 0..8 {
            let slice = &input[k * FRAME..(k + 1) * FRAME];
            let n = encoder.encode(slice, &mut buf, FRAME).expect("enc");
            decoder.decode(&buf[..n], &mut out, FRAME).expect("dec");
            let mut sig_e = 0.0f64;
            let mut err_e = 0.0f64;
            let mut max_err = 0.0f32;
            for i in 0..FRAME {
                let s = f64::from(slice[i]);
                let r = f64::from(out[i]);
                sig_e += s * s;
                err_e += (s - r) * (s - r);
                max_err = max_err.max((slice[i] - out[i]).abs());
            }
            let snr = if err_e > 1e-12 {
                10.0 * (sig_e / err_e).log10()
            } else {
                120.0
            };
            println!("frame {k}: bytes={n} max_err={max_err:.4} SNR={snr:.2}dB");
        }
    }

    /// Round trip: when the encoder emits a chosen `NlsfDecision`, the decoder
    /// must reconstruct the *identical* NLSF Q15 values.
    #[test]
    fn test_nlsf_encode_decode_roundtrip() {
        use super::super::silk_excitation::encode_nlsf_full;

        let samples: Vec<f32> = (0..320)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin() * 0.5)
            .collect();
        let bw = SilkBandwidth::Wideband;
        let order = bw.lpc_order();
        let decision = quantise_nlsf_full(&samples, bw).expect("ok");

        let mut enc = SilkRangeEncoder::new();
        enc.encode_bit_logp(true, 1).expect("vad");
        enc.encode_bit_logp(false, 1).expect("lbrr");
        enc.encode_icdf(0, &t::TYPE_OFFSET_VAD_ICDF, 8).expect("ft");
        encode_nlsf_full(&mut enc, bw, SilkSignalType::Unvoiced, &decision).expect("enc nlsf");
        let bytes = enc.finish().expect("finish");

        let mut dec = SilkRangeDecoder::new(&bytes).expect("dec init");
        let _vad = dec.decode_bit_logp(1).expect("vad");
        let _lbrr = dec.decode_bit_logp(1).expect("lbrr");
        let _frame_type = dec.decode_icdf(&t::TYPE_OFFSET_VAD_ICDF, 8).expect("ft");

        let voiced_idx = 0;
        let stage1_icdf = &t::NLSF_CB1_ICDF_WB[voiced_idx];
        let i1 = dec.decode_icdf(stage1_icdf, 8).expect("dec i1");
        assert_eq!(i1, decision.i1, "i1 mismatch");

        let select_table = &t::NLSF_CB2_SELECT_WB;
        let mut residual_cb = vec![0usize; order];
        for pair in 0..order / 2 {
            let entry = select_table[i1 * (order / 2) + pair];
            let even = 2 * pair;
            let odd = even + 1;
            residual_cb[even] = usize::from((entry >> 1) & 0x07);
            residual_cb[odd] = usize::from((entry >> 5) & 0x07);
        }

        let mut decoded_values = vec![0i32; order];
        for coeff in 0..order {
            let icdf = &t::NLSF_CB2_ICDF_WB[residual_cb[coeff]];
            let mut value = dec.decode_icdf(icdf, 8).expect("res") as i32 - 4;
            if value == 4 {
                value += dec.decode_icdf(&t::NLSF_EXT_ICDF, 8).expect("ext+") as i32;
            } else if value == -4 {
                value -= dec.decode_icdf(&t::NLSF_EXT_ICDF, 8).expect("ext-") as i32;
            }
            decoded_values[coeff] = value;
        }

        for coeff in 0..order {
            assert_eq!(
                decoded_values[coeff], decision.res_idx[coeff],
                "residual mismatch at coeff {coeff}: encoder picked {} but decoder reads {}",
                decision.res_idx[coeff], decoded_values[coeff],
            );
        }
    }

    /// Sign context never indexes out of the 6-row table.
    #[test]
    fn test_sign_context_bounded() {
        for st in [
            SilkSignalType::Inactive,
            SilkSignalType::Unvoiced,
            SilkSignalType::Voiced,
        ] {
            for off in 0..2 {
                let row = sign_context(st, off, 5);
                assert!(row < 6);
            }
        }
    }
}
