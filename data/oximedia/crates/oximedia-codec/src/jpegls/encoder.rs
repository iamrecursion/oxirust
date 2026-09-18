//! JPEG-LS encoder top-level (ISO 14495-1, LOCO-I forward path).
//!
//! ## Pipeline
//!
//! ```text
//! Raw sample planes
//!   → write_headers()   — SOI/SOF55/[LSE]/SOS markers
//!   → encode_scan()     — LOCO-I predictor + Golomb-Rice entropy encode
//!   → EOI               — terminate the stream
//! ```
//!
//! The encoder is the exact forward inverse of
//! [`super::decoder`]: a stream produced here decodes losslessly (NEAR = 0) or
//! within `±NEAR` (NEAR > 0) back to the source samples via
//! [`JpegLsDecoder`](super::decoder::JpegLsDecoder).
//!
//! ## Supported features
//!
//! | Feature                       | Status |
//! |-------------------------------|--------|
//! | Lossless (NEAR = 0)           | Full   |
//! | Near-lossless (NEAR > 0)      | Full   |
//! | ILV = 0 (non-interleaved)     | Full   |
//! | ILV = 1 (line-interleaved)    | Full   |
//! | ILV = 2 (sample-interleaved)  | Full   |
//! | RUN mode §A.7 (ILV 0/1)       | Full   |
//! | RUN mode ILV 2                | Suspended per ISO (regular-only) |
//!
//! ## Entropy-coding model
//!
//! The encoder emits both regular-mode §A.6 codes and the §A.7 RUN-mode
//! length tokens.  Flat / near-flat regions trigger RUN entry exactly
//! where the decoder will: when the raw gradients `(D−B, B−C, C−A)` are
//! within `±NEAR`.  The encoder then writes one `1` bit per full
//! `(1 << J[run_index])`-sized token, optionally one `0` bit plus a
//! `J[run_index]`-bit residual length for an in-line interruption, and
//! finally the breaking sample under context 365 (`Ra == Rb`) or 366
//! (`Ra != Rb`).  At end-of-line a single `1` bit signals "remaining
//! samples to EOL match the run value".

use super::context::{context_index, update_context, ContextState, NUM_TOTAL_CONTEXTS};
use super::golomb::{compute_limit, compute_qbpp, map_error_lossless, map_error_near};
use super::golomb_write::{encode_golomb_unsigned_limited, BitWriter};
use super::marker_write::{write_eoi, write_sof55, write_soi, write_sos, FrameComponent};
use super::markers::JlsPresetParams;
use super::predictor::{predict, quantize_gradient};
use super::run_mode::{
    bump_run_index, decrement_run_index, enter_run_near, j_for, run_termination_ctx, threshold_for,
    RunState,
};
use super::{JlsError, JlsResult};

/// Configuration for a [`JpegLsEncoder`].
#[derive(Debug, Clone)]
pub struct JpegLsEncoderConfig {
    /// Image width in samples (must be > 0).
    pub width: u32,
    /// Image height in lines (must be > 0).
    pub height: u32,
    /// Number of colour components (1 = greyscale, ≥1 supported).
    pub components: u8,
    /// Bits per sample (1–16).
    pub bit_depth: u8,
    /// NEAR parameter: 0 = lossless, > 0 = near-lossless (max error `± near`).
    pub near: u8,
    /// Interleave mode: 0 = non-interleaved, 1 = line-interleaved,
    /// 2 = sample-interleaved.
    pub interleave: u8,
}

impl JpegLsEncoderConfig {
    /// Create a lossless single-scan greyscale configuration.
    #[must_use]
    pub fn greyscale(width: u32, height: u32, bit_depth: u8) -> Self {
        Self {
            width,
            height,
            components: 1,
            bit_depth,
            near: 0,
            interleave: 0,
        }
    }

    /// Create a lossless multi-component configuration with the given interleave.
    #[must_use]
    pub fn multicomponent(
        width: u32,
        height: u32,
        components: u8,
        bit_depth: u8,
        interleave: u8,
    ) -> Self {
        Self {
            width,
            height,
            components,
            bit_depth,
            near: 0,
            interleave,
        }
    }

    /// Set the NEAR parameter (consuming builder).
    #[must_use]
    pub fn with_near(mut self, near: u8) -> Self {
        self.near = near;
        self
    }
}

/// JPEG-LS encoder (LOCO-I forward path, regular + RUN modes).
pub struct JpegLsEncoder {
    config: JpegLsEncoderConfig,
    presets: JlsPresetParams,
}

impl JpegLsEncoder {
    /// Create a new encoder from a [`JpegLsEncoderConfig`].
    ///
    /// # Errors
    ///
    /// Returns [`JlsError::Unsupported`] when the configuration is invalid:
    /// zero dimensions, zero components, a bit depth outside `1..=16`, an
    /// unknown ILV mode, or a NEAR value too large for the bit depth.
    pub fn new(config: JpegLsEncoderConfig) -> JlsResult<Self> {
        if config.width == 0 || config.height == 0 {
            return Err(JlsError::Unsupported("zero image dimension".to_string()));
        }
        if config.components == 0 {
            return Err(JlsError::Unsupported("zero components".to_string()));
        }
        if config.bit_depth == 0 || config.bit_depth > 16 {
            return Err(JlsError::Unsupported(format!(
                "bit depth {} out of range 1..=16",
                config.bit_depth
            )));
        }
        if config.interleave > 2 {
            return Err(JlsError::Unsupported(format!(
                "ILV mode {} is not defined in ISO 14495-1",
                config.interleave
            )));
        }
        if config.interleave != 0 && config.components < 2 {
            return Err(JlsError::Unsupported(format!(
                "ILV={} requires ≥2 components, got {}",
                config.interleave, config.components
            )));
        }
        let presets = JlsPresetParams::default_for_precision(config.bit_depth);
        if config.near as i32 > presets.max_val as i32 {
            return Err(JlsError::Unsupported(format!(
                "NEAR {} exceeds MaxVal {}",
                config.near, presets.max_val
            )));
        }
        Ok(Self { config, presets })
    }

    /// Borrow the active configuration.
    #[must_use]
    pub fn config(&self) -> &JpegLsEncoderConfig {
        &self.config
    }

    /// Borrow the preset parameters (MaxVal/T1/T2/T3/Reset) in effect.
    #[must_use]
    pub fn presets(&self) -> &JlsPresetParams {
        &self.presets
    }

    /// Encode planar sample data into a complete JPEG-LS byte stream.
    ///
    /// `planes` must contain exactly `config.components` slices, each holding
    /// `width * height` row-major samples (low `bit_depth` bits used). The
    /// returned `Vec<u8>` begins with SOI and ends with EOI and decodes via
    /// [`JpegLsDecoder`](super::decoder::JpegLsDecoder).
    ///
    /// # Errors
    ///
    /// Returns [`JlsError::Unsupported`] when the number of planes or the
    /// length of any plane does not match the configured geometry.
    pub fn encode_planes(&self, planes: &[&[u16]]) -> JlsResult<Vec<u8>> {
        let nc = self.config.components as usize;
        if planes.len() != nc {
            return Err(JlsError::Unsupported(format!(
                "expected {} planes, got {}",
                nc,
                planes.len()
            )));
        }
        let w = self.config.width as usize;
        let h = self.config.height as usize;
        let expected = w * h;
        for (i, plane) in planes.iter().enumerate() {
            if plane.len() != expected {
                return Err(JlsError::Unsupported(format!(
                    "plane {i} length {} != width*height {}",
                    plane.len(),
                    expected
                )));
            }
        }

        let mut out = Vec::new();
        self.write_headers(&mut out);
        let scan = self.encode_scan(planes)?;
        out.extend_from_slice(&scan);
        write_eoi(&mut out);
        Ok(out)
    }

    /// Convenience wrapper for the single-component (greyscale) case.
    ///
    /// `samples` holds `width * height` row-major samples. Equivalent to
    /// [`encode_planes`](Self::encode_planes) with a one-element plane slice.
    ///
    /// # Errors
    ///
    /// Returns [`JlsError::Unsupported`] when the encoder is not configured for
    /// a single component or `samples` has the wrong length.
    pub fn encode_greyscale(&self, samples: &[u16]) -> JlsResult<Vec<u8>> {
        if self.config.components != 1 {
            return Err(JlsError::Unsupported(format!(
                "encode_greyscale requires 1 component, configured {}",
                self.config.components
            )));
        }
        self.encode_planes(&[samples])
    }

    /// Encode 8-bit planar input given as `u8` slices (convenience helper).
    ///
    /// Each input slice is widened to `u16` before encoding. Mirrors
    /// [`encode_planes`](Self::encode_planes) otherwise.
    ///
    /// # Errors
    ///
    /// Propagates the same errors as [`encode_planes`](Self::encode_planes).
    pub fn encode_planes_u8(&self, planes: &[&[u8]]) -> JlsResult<Vec<u8>> {
        let widened: Vec<Vec<u16>> = planes
            .iter()
            .map(|p| p.iter().map(|&v| v as u16).collect())
            .collect();
        let refs: Vec<&[u16]> = widened.iter().map(Vec::as_slice).collect();
        self.encode_planes(&refs)
    }

    // ── Header emission ──────────────────────────────────────────────────────

    /// Emit SOI, SOF55, an optional LSE preset marker, and SOS.
    fn write_headers(&self, out: &mut Vec<u8>) {
        write_soi(out);

        let comps: Vec<FrameComponent> = (1..=self.config.components)
            .map(FrameComponent::standard)
            .collect();
        write_sof55(
            out,
            self.config.bit_depth,
            self.config.height as u16,
            self.config.width as u16,
            &comps,
        );

        // The decoder reconstructs default presets from the precision when no
        // LSE marker is present, and our `presets` are exactly those defaults,
        // so no LSE marker is required. (write_lse_preset is available for the
        // custom-preset case and is exercised by marker_write's unit tests.)

        let ids: Vec<u8> = (1..=self.config.components).collect();
        write_sos(out, &ids, self.config.near, self.config.interleave, 0);
    }

    // ── Scan encoding ────────────────────────────────────────────────────────

    /// Encode all sample planes into compressed scan bytes (with byte-stuffing).
    fn encode_scan(&self, planes: &[&[u16]]) -> JlsResult<Vec<u8>> {
        let w = self.config.width as usize;
        let h = self.config.height as usize;
        let nc = self.config.components as usize;

        let params = ScanParams {
            max_val: self.presets.max_val as i32,
            near: self.config.near as i32,
            reset: self.presets.reset as i32,
            limit: compute_limit(self.presets.max_val as i32),
            qbpp: compute_qbpp(self.presets.max_val as i32),
            t1: self.presets.t1,
            t2: self.presets.t2,
            t3: self.presets.t3,
            w,
        };

        // Per-component reconstruction buffers — predictions must reference the
        // values the decoder will reconstruct, not the (possibly lossier) source.
        let mut recon: Vec<Vec<u16>> = (0..nc).map(|_| vec![0u16; w * h]).collect();
        let mut all_ctx: Vec<Vec<ContextState>> = (0..nc)
            .map(|_| vec![ContextState::default(); NUM_TOTAL_CONTEXTS])
            .collect();

        let mut writer = BitWriter::with_capacity(w * h * nc / 2 + 16);

        match self.config.interleave {
            // ── ILV = 0: encode each plane fully before the next.  RUN mode active.
            0 => {
                let mut run_states: Vec<RunState> = (0..nc).map(|_| RunState::new()).collect();
                for comp in 0..nc {
                    let ctx_states = &mut all_ctx[comp];
                    let run_state = &mut run_states[comp];
                    for row in 0..h {
                        encode_row_with_run_mode(
                            planes[comp],
                            &mut recon[comp],
                            ctx_states,
                            run_state,
                            &mut writer,
                            row,
                            &params,
                        );
                    }
                }
            }

            // ── ILV = 1: one full row per component, cycling by row.  RUN active.
            1 => {
                let mut run_states: Vec<RunState> = (0..nc).map(|_| RunState::new()).collect();
                for row in 0..h {
                    for comp in 0..nc {
                        let ctx_states = &mut all_ctx[comp];
                        let run_state = &mut run_states[comp];
                        encode_row_with_run_mode(
                            planes[comp],
                            &mut recon[comp],
                            ctx_states,
                            run_state,
                            &mut writer,
                            row,
                            &params,
                        );
                    }
                }
            }

            // ── ILV = 2: one sample per component per raster position.
            //    RUN mode is suspended per ISO §F.1 — every sample uses the
            //    regular §A.6 path.
            2 => {
                for row in 0..h {
                    for col in 0..w {
                        for comp in 0..nc {
                            let ctx_states = &mut all_ctx[comp];
                            let sample = planes[comp][row * w + col];
                            let (a, b, c, d) = fetch_neighbours(&recon[comp], row, col, w);
                            let rx = encode_pixel_regular(
                                ctx_states,
                                &mut writer,
                                a,
                                b,
                                c,
                                d,
                                sample,
                                &params,
                            );
                            recon[comp][row * w + col] = rx;
                        }
                    }
                }
            }

            other => {
                return Err(JlsError::Unsupported(format!(
                    "ILV mode {other} is not defined in ISO 14495-1"
                )));
            }
        }

        Ok(writer.finish())
    }
}

/// Immutable per-scan parameters shared by every per-pixel call.
struct ScanParams {
    max_val: i32,
    near: i32,
    reset: i32,
    limit: i32,
    qbpp: u8,
    t1: i32,
    t2: i32,
    t3: i32,
    w: usize,
}

/// Causal-neighbour fetch, identical to the decoder's
/// [`super::decoder::fetch_neighbours`](super::decoder).  Replicated
/// here to avoid making the decoder helper public.
#[inline]
fn fetch_neighbours(recon: &[u16], row: usize, col: usize, w: usize) -> (i32, i32, i32, i32) {
    let a = if col > 0 {
        recon[row * w + col - 1] as i32
    } else if row > 0 {
        recon[(row - 1) * w] as i32
    } else {
        0
    };

    let b = if row > 0 {
        recon[(row - 1) * w + col] as i32
    } else {
        a
    };

    let c = if row > 0 && col > 0 {
        recon[(row - 1) * w + col - 1] as i32
    } else if row > 0 {
        recon[(row - 1) * w] as i32
    } else {
        0
    };

    let d = if row > 0 && col + 1 < w {
        recon[(row - 1) * w + col + 1] as i32
    } else {
        b
    };

    (a, b, c, d)
}

/// Encode one row of one component with RUN-mode dispatch enabled.
///
/// Mirrors `decode_row_with_run_mode` in [`super::decoder`]: at each
/// column the RUN-entry test is applied to the raw gradients; on entry
/// [`encode_run_mode`] consumes consecutive matching source samples and
/// emits the corresponding length tokens; otherwise the regular §A.6
/// path emits one Golomb-Rice residual.
fn encode_row_with_run_mode(
    source: &[u16],
    recon: &mut [u16],
    ctx_states: &mut [ContextState],
    run_state: &mut RunState,
    writer: &mut BitWriter,
    row: usize,
    p: &ScanParams,
) {
    let w = p.w;
    run_state.reset_at_line_start();
    let mut col = 0usize;
    while col < w {
        let (a, b, c, d) = fetch_neighbours(recon, row, col, w);
        let d1 = d - b;
        let d2 = b - c;
        let d3 = c - a;

        if enter_run_near(d1, d2, d3, p.near) {
            // RUN entry: scan source forward from `col` while it matches
            // `runval = Ra` (lossless) or stays within `±NEAR`.
            col = encode_run_mode(source, recon, ctx_states, run_state, writer, row, col, a, p);
        } else {
            let sample = source[row * w + col];
            let rx = encode_pixel_regular(ctx_states, writer, a, b, c, d, sample, p);
            recon[row * w + col] = rx;
            col += 1;
        }
    }
}

/// RUN-mode forward (§A.7.2).  Returns the new column position.
///
/// `runval = Ra = a`.  The encoder counts how many consecutive source
/// samples starting at `(row, col)` stay within `±NEAR` of `runval`,
/// then emits the matching token sequence and (if the run is
/// interrupted in-line) the termination sample under context 365 or
/// 366.  Reconstructed samples are written into `recon` so the
/// decoder's view of `Ra` etc. for downstream pixels matches.
#[allow(clippy::too_many_arguments)]
fn encode_run_mode(
    source: &[u16],
    recon: &mut [u16],
    ctx_states: &mut [ContextState],
    run_state: &mut RunState,
    writer: &mut BitWriter,
    row: usize,
    start_col: usize,
    ra: i32,
    p: &ScanParams,
) -> usize {
    let w = p.w;
    let runval = ra;
    let near_bound = p.near;

    // Step 1: determine the run length — the number of consecutive
    // samples from `start_col` that stay within `±NEAR` of `runval`.
    let mut max_run = 0usize;
    while start_col + max_run < w {
        let s = source[row * w + start_col + max_run] as i32;
        if (s - runval).abs() <= near_bound {
            max_run += 1;
        } else {
            break;
        }
    }

    let mut col = start_col;
    let mut remaining = max_run;

    // Phase 1: emit full-length tokens of length `1 << J[run_index]` for
    // as long as both the source matches and the row has room.
    loop {
        let thr = threshold_for(run_state.run_index);
        if thr <= 0 {
            break;
        }
        let thr_usize = thr as usize;
        if col + thr_usize > w {
            // A full token would extend past EOL — fall to phase 2.
            break;
        }
        if remaining < thr_usize {
            // Not enough matching samples left for a full token — phase 2.
            break;
        }
        // Emit a full token.
        writer.write_bit(1);
        // Reconstruct: each filled sample takes value `runval` (clamped to range).
        for k in 0..thr_usize {
            recon[row * w + col + k] = runval.clamp(0, p.max_val) as u16;
        }
        col += thr_usize;
        remaining -= thr_usize;
        bump_run_index(run_state);
        if col == w {
            // Exactly EOL after a full token — no further bits.
            return col;
        }
    }

    // Phase 2: either EOL with a residual fragment, or interruption at
    // the non-matching source sample.
    if col + remaining == w {
        // EOL with partial fragment — emit a single `1` bit.
        // Reconstruct the remaining matching samples.
        for k in 0..remaining {
            recon[row * w + col + k] = runval.clamp(0, p.max_val) as u16;
        }
        writer.write_bit(1);
        col + remaining
    } else {
        // Interruption: write `0`, residual length in J[run_index] bits,
        // then the termination sample.
        writer.write_bit(0);
        let j_bits = j_for(run_state.run_index) as u8;
        if j_bits > 0 {
            writer.write_bits(remaining as u32, j_bits);
        }
        for k in 0..remaining {
            recon[row * w + col + k] = runval.clamp(0, p.max_val) as u16;
        }
        col += remaining;
        // Termination sample at `col`.
        let term_source = source[row * w + col];
        let term_rx = encode_run_termination_sample(
            recon,
            ctx_states,
            writer,
            row,
            col,
            runval,
            term_source,
            p,
        );
        recon[row * w + col] = term_rx;
        col += 1;
        decrement_run_index(run_state);
        col
    }
}

/// Encode the RUN-interruption (termination) sample at `(row, col)`.
///
/// Uses one of two special contexts (365 / 366) with the predictor fixed
/// at `PX = Ra = runval`.  Symmetric counterpart of
/// [`super::decoder::decode_run_termination_sample`](super::decoder).
#[allow(clippy::too_many_arguments)]
fn encode_run_termination_sample(
    recon: &[u16],
    ctx_states: &mut [ContextState],
    writer: &mut BitWriter,
    row: usize,
    col: usize,
    runval: i32,
    sample: u16,
    p: &ScanParams,
) -> u16 {
    let rb = if row > 0 {
        recon[(row - 1) * p.w + col] as i32
    } else {
        runval
    };

    let ctx_idx = run_termination_ctx(runval, rb);
    let state = &mut ctx_states[ctx_idx];

    let sign = if runval == rb {
        1i32
    } else if rb > runval {
        1i32
    } else {
        -1i32
    };

    let predicted = runval;
    let corrected_px = (predicted - sign * state.cx).clamp(0, p.max_val);

    let k = state.k.max(0);
    let sample_i = sample as i32;

    if p.near == 0 {
        let err = (sample_i - corrected_px) * sign;
        let e_mapped = map_error_lossless(err);
        encode_golomb_unsigned_limited(writer, e_mapped, k, p.limit, p.qbpp);

        let rx = (corrected_px + err * sign).clamp(0, p.max_val) as u16;
        update_context(state, err, p.near, p.reset, p.max_val);
        rx
    } else {
        let q_step = 2 * p.near + 1;
        let err_raw = (sample_i - corrected_px) * sign;
        let err_q = if err_raw >= 0 {
            (err_raw + p.near) / q_step
        } else {
            (err_raw - p.near) / q_step
        };
        let e_mapped = map_error_near(err_q, p.near, p.max_val);
        encode_golomb_unsigned_limited(writer, e_mapped, k, p.limit, p.qbpp);

        let dequant = err_q * q_step * sign;
        let rx = (corrected_px + dequant).clamp(0, p.max_val) as u16;
        update_context(state, err_q.abs(), p.near, p.reset, p.max_val);
        rx
    }
}

/// Encode one sample for one component using the regular §A.6 path and
/// return its reconstructed value.
///
/// This is the exact forward inverse of `decode_pixel_regular` in
/// [`super::decoder`]: it derives the identical causal neighbourhood,
/// context index, bias-corrected prediction, and adaptive Golomb order
/// `k`, then maps and writes the residual the decoder will read back.
#[allow(clippy::too_many_arguments)]
fn encode_pixel_regular(
    ctx_states: &mut [ContextState],
    writer: &mut BitWriter,
    a: i32,
    b: i32,
    c: i32,
    d: i32,
    sample: u16,
    p: &ScanParams,
) -> u16 {
    let d1 = d - b;
    let d2 = b - c;
    let d3 = c - a;
    let q1 = quantize_gradient(d1, p.t1, p.t2, p.t3);
    let q2 = quantize_gradient(d2, p.t1, p.t2, p.t3);
    let q3 = quantize_gradient(d3, p.t1, p.t2, p.t3);

    let (ctx_idx, sign) = context_index(q1, q2, q3);
    let state = &mut ctx_states[ctx_idx];

    let px = predict(a, b, c);
    let corrected_px = (px - sign * state.cx).clamp(0, p.max_val);

    let k = state.k.max(0);
    let sample_i = sample as i32;

    if p.near == 0 {
        let err = (sample_i - corrected_px) * sign;
        let e_mapped = map_error_lossless(err);
        encode_golomb_unsigned_limited(writer, e_mapped, k, p.limit, p.qbpp);

        let rx = (corrected_px + err * sign).clamp(0, p.max_val) as u16;
        update_context(state, err, p.near, p.reset, p.max_val);
        rx
    } else {
        let q_step = 2 * p.near + 1;
        let err_raw = (sample_i - corrected_px) * sign;
        let err_q = if err_raw >= 0 {
            (err_raw + p.near) / q_step
        } else {
            (err_raw - p.near) / q_step
        };
        let e_mapped = map_error_near(err_q, p.near, p.max_val);
        encode_golomb_unsigned_limited(writer, e_mapped, k, p.limit, p.qbpp);

        let dequant = err_q * q_step * sign;
        let rx = (corrected_px + dequant).clamp(0, p.max_val) as u16;
        update_context(state, err_q.abs(), p.near, p.reset, p.max_val);
        rx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_zero_dimension() {
        let cfg = JpegLsEncoderConfig::greyscale(0, 4, 8);
        assert!(JpegLsEncoder::new(cfg).is_err());
    }

    #[test]
    fn rejects_bad_bit_depth() {
        let cfg = JpegLsEncoderConfig::greyscale(4, 4, 17);
        assert!(JpegLsEncoder::new(cfg).is_err());
        let cfg0 = JpegLsEncoderConfig::greyscale(4, 4, 0);
        assert!(JpegLsEncoder::new(cfg0).is_err());
    }

    #[test]
    fn rejects_interleave_with_single_component() {
        let cfg = JpegLsEncoderConfig {
            width: 4,
            height: 4,
            components: 1,
            bit_depth: 8,
            near: 0,
            interleave: 1,
        };
        assert!(JpegLsEncoder::new(cfg).is_err());
    }

    #[test]
    fn rejects_plane_count_mismatch() {
        let enc = JpegLsEncoder::new(JpegLsEncoderConfig::greyscale(2, 2, 8)).expect("config ok");
        let a = [0u16; 4];
        let b = [0u16; 4];
        // Two planes for a 1-component encoder → error.
        assert!(enc.encode_planes(&[&a, &b]).is_err());
    }

    #[test]
    fn rejects_plane_length_mismatch() {
        let enc = JpegLsEncoder::new(JpegLsEncoderConfig::greyscale(4, 4, 8)).expect("config ok");
        let short = [0u16; 3];
        assert!(enc.encode_greyscale(&short).is_err());
    }

    #[test]
    fn produces_soi_and_eoi_framing() {
        let enc = JpegLsEncoder::new(JpegLsEncoderConfig::greyscale(2, 2, 8)).expect("config ok");
        let samples = [10u16, 20, 30, 40];
        let bytes = enc.encode_greyscale(&samples).expect("encode ok");
        assert_eq!(&bytes[0..2], &[0xFF, 0xD8], "must start with SOI");
        assert_eq!(
            &bytes[bytes.len() - 2..],
            &[0xFF, 0xD9],
            "must end with EOI"
        );
    }
}
