//! JPEG-LS decoder top-level (ISO 14495-1).
//!
//! ## Pipeline
//!
//! ```text
//! Input bytes
//!   → parse_headers()        — SOI/SOF55/LSE/SOS markers
//!   → decode_scan()          — LOCO-I predictor + Golomb-Rice entropy decode
//!   → DecodedImage           — row-major u16 planes, one Vec per component
//! ```
//!
//! ## Supported features
//!
//! | Feature                  | Status    |
//! |--------------------------|-----------|
//! | Lossless (NEAR = 0)      | Full      |
//! | Near-lossless (NEAR > 0) | Full      |
//! | ILV = 0 (non-interleaved)| Full      |
//! | ILV = 1 (line-interleaved)| Full     |
//! | ILV = 2 (sample-interleaved)| Full   |
//! | RUN mode (§A.7) ILV 0/1  | Full      |
//! | RUN mode ILV 2           | Suspended per ISO (regular-only) |
//!
//! ## RUN-mode dispatch
//!
//! At each pixel position the decoder computes the raw gradient triple
//! `(d1, d2, d3) = (D−B, B−C, C−A)` and tests
//! [`enter_run_near`].  When the test
//! passes (lossless: all gradients zero; near-lossless: all within
//! `±NEAR`), control transfers to `decode_run_mode`: a sequence of
//! run-length tokens drawn from `J[]` is read until either end-of-line
//! is reached or a non-matching sample terminates the run.  The
//! terminating sample is decoded with one of two RUN-interruption
//! contexts (365 for `Ra == Rb`, 366 otherwise).  Outside RUN mode the
//! existing §A.6 regular path is used unchanged.

use super::context::{context_index, update_context, ContextState, NUM_TOTAL_CONTEXTS};
use super::golomb::{
    compute_limit, compute_qbpp, decode_golomb_unsigned_limited, map_error_near,
    unmap_error_lossless, unmap_error_near, BitReader,
};
use super::markers::{parse_headers, JlsHeaders};
use super::predictor::{predict, quantize_gradient};
use super::run_mode::{
    bump_run_index, decrement_run_index, enter_run_near, j_for, run_termination_ctx, threshold_for,
    RunState,
};
use super::{JlsError, JlsResult};

/// A JPEG-LS decoded image.
///
/// Each component is stored as a flat row-major `Vec<u16>` with
/// `width * height` entries. For 8-bit images only the low 8 bits are used.
#[derive(Debug, Clone)]
pub struct DecodedImage {
    /// Width of the image in samples.
    pub width: u32,
    /// Height of the image in lines.
    pub height: u32,
    /// Number of colour components (1 = greyscale, 3 = colour).
    pub num_components: u8,
    /// Bits per sample (1–16).
    pub precision: u8,
    /// One `Vec<u16>` per component in row-major order.
    pub samples: Vec<Vec<u16>>,
}

/// JPEG-LS lossless decoder.
pub struct JpegLsDecoder;

impl JpegLsDecoder {
    /// Create a new decoder instance.
    pub fn new() -> Self {
        Self
    }

    /// Return `true` if `data` looks like a JPEG-LS stream.
    ///
    /// Checks for SOI (0xFFD8) followed immediately by SOF55 (0xFFF7).
    pub fn is_jpegls(data: &[u8]) -> bool {
        data.len() >= 4 && data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF && data[3] == 0xF7
    }

    /// Decode a JPEG-LS byte stream to a [`DecodedImage`].
    pub fn decode(data: &[u8]) -> JlsResult<DecodedImage> {
        let headers = parse_headers(data)?;
        let scan_data = &data[headers.scan_data_start..];
        decode_scan(scan_data, &headers)
    }
}

impl Default for JpegLsDecoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Shared per-scan decode parameters.  Bundling these lets the
/// regular and RUN paths share one signature and helps the compiler
/// inline the per-pixel hot path.
struct ScanDecodeParams {
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

// ─── Causal-neighbour fetch ──────────────────────────────────────────────────

/// Return `(a, b, c, d)` — the JPEG-LS causal neighbourhood at `(row, col)`.
///
/// `out` is the per-component reconstruction buffer (row-major).  The four
/// values follow ISO 14495-1 §A.1: `a` is the left neighbour, `b` is the
/// top neighbour, `c` is the top-left corner and `d` is the top-right
/// corner.  Edge-of-image rules replicate the nearest valid sample.
#[inline]
fn fetch_neighbours(out: &[u16], row: usize, col: usize, w: usize) -> (i32, i32, i32, i32) {
    let a = if col > 0 {
        out[row * w + col - 1] as i32
    } else if row > 0 {
        out[(row - 1) * w] as i32
    } else {
        0
    };

    let b = if row > 0 {
        out[(row - 1) * w + col] as i32
    } else {
        a
    };

    let c = if row > 0 && col > 0 {
        out[(row - 1) * w + col - 1] as i32
    } else if row > 0 {
        out[(row - 1) * w] as i32
    } else {
        0
    };

    let d = if row > 0 && col + 1 < w {
        out[(row - 1) * w + col + 1] as i32
    } else {
        b
    };

    (a, b, c, d)
}

// ─── Per-pixel LOCO-I regular-mode decode ────────────────────────────────────

/// Decode a single sample with the regular §A.6 path given its
/// already-fetched causal neighbourhood.
///
/// Returns the reconstructed sample value clamped to `[0, max_val]`.
///
/// ### Near-lossless (NEAR > 0)
///
/// The quantisation step is `q_step = 2 * near + 1`.  The encoder divides the
/// sign-normalised error by `q_step` (truncating toward zero) and maps the
/// resulting integer with the near-lossless mapping.  The decoder unmaps
/// with [`unmap_error_near`] and multiplies back by `q_step` before
/// adding to the bias-corrected prediction.  The reconstructed value is
/// therefore the original sample rounded to the nearest multiple of
/// `q_step` within `NEAR` of the true value.
fn decode_pixel_regular(
    ctx_states: &mut [ContextState],
    reader: &mut BitReader<'_>,
    a: i32,
    b: i32,
    c: i32,
    d: i32,
    p: &ScanDecodeParams,
) -> JlsResult<u16> {
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
    let e_mapped =
        decode_golomb_unsigned_limited(reader, k, p.limit, p.qbpp).ok_or(JlsError::Truncated {
            context: "scan data",
        })?;

    let (err_q, rx) = if p.near == 0 {
        let err_abs = unmap_error_lossless(e_mapped);
        let err = err_abs * sign;
        let reconstructed = (corrected_px + err).clamp(0, p.max_val) as u16;
        (err_abs, reconstructed)
    } else {
        let q_step = 2 * p.near + 1;
        let err_q_signed = unmap_error_near(e_mapped, p.near, p.max_val);
        let err = err_q_signed * q_step * sign;
        let reconstructed = (corrected_px + err).clamp(0, p.max_val) as u16;
        (err_q_signed.abs(), reconstructed)
    };

    update_context(state, err_q, p.near, p.reset, p.max_val);

    Ok(rx)
}

// ─── RUN-mode decode (§A.7.3) ────────────────────────────────────────────────

/// Decode RUN mode starting at column `start_col` in row `row` for one
/// component.  Returns the new column position after the run plus its
/// (optional) termination sample have been written into `samples`.
///
/// The caller has already verified the RUN-entry test on `(d1, d2, d3)`
/// for this pixel; this function consumes one or more bits from
/// `reader`, fills consecutive samples in `samples` with the run value
/// `Ra`, and — when the run is interrupted before EOL — decodes the
/// breaking sample using one of the two RUN-interruption contexts.
#[allow(clippy::too_many_arguments)]
fn decode_run_mode(
    samples: &mut [u16],
    ctx_states: &mut [ContextState],
    run_state: &mut RunState,
    reader: &mut BitReader<'_>,
    row: usize,
    start_col: usize,
    ra: i32,
    p: &ScanDecodeParams,
) -> JlsResult<usize> {
    let w = p.w;
    let runval = ra;
    let mut col = start_col;

    // Phase 1: read full-length tokens.  Each `1` bit consumes
    // `1 << J[run_index]` samples (all filled with `runval`).  When the
    // next token would extend past the row, fall through to phase 2.
    loop {
        let thr = threshold_for(run_state.run_index);
        if thr <= 0 {
            // Defensive: run_index out of range — treat as no more tokens.
            break;
        }
        if col + thr as usize > w {
            // Cannot fit a full token before EOL — read the single
            // trailing bit below.
            break;
        }
        let bit = reader.read_bit().ok_or(JlsError::Truncated {
            context: "run mode token",
        })?;
        if bit == 1 {
            // Full token: fill `thr` samples with `runval`.
            for k in 0..thr as usize {
                samples[row * w + col + k] = runval as u16;
            }
            col += thr as usize;
            bump_run_index(run_state);
            if col == w {
                // Exactly EOL after a full token — RUN ends with no extra bits.
                return Ok(col);
            }
            // Continue: try to read another full token at the next index.
            continue;
        } else {
            // `0` bit ⇒ interruption.  Read the residual length in
            // `J[run_index]` bits and the termination sample.
            let j_bits = j_for(run_state.run_index) as u8;
            let residual = if j_bits == 0 {
                0i32
            } else {
                reader.read_bits(j_bits).ok_or(JlsError::Truncated {
                    context: "run residual length",
                })? as i32
            };
            // Defend against a malformed run residual that would write past the
            // end of the row (out-of-bounds write). In RUN-interruption mode the
            // breaking (termination) sample sits at `col + residual`, so that
            // index must be strictly inside the row *before* any sample is
            // filled — checking after the write loop is too late.
            let residual = residual as usize;
            if col + residual >= w {
                return Err(JlsError::Truncated {
                    context: "run residual overflows row",
                });
            }
            for k in 0..residual {
                samples[row * w + col + k] = runval as u16;
            }
            col += residual;
            // Decode the breaking sample with the RUN-interruption context.
            let term_sample =
                decode_run_termination_sample(samples, ctx_states, reader, row, col, runval, p)?;
            samples[row * w + col] = term_sample;
            col += 1;
            decrement_run_index(run_state);
            return Ok(col);
        }
    }

    // Phase 2: cannot fit another full token; read one trailing bit.
    let bit = reader.read_bit().ok_or(JlsError::Truncated {
        context: "run trailing bit",
    })?;
    if bit == 1 {
        // EOL-fill: remaining samples up to `w` all match runval.
        while col < w {
            samples[row * w + col] = runval as u16;
            col += 1;
        }
        // RUN ends at EOL.
        Ok(col)
    } else {
        // Interruption with residual + termination sample.
        let j_bits = j_for(run_state.run_index) as u8;
        let residual = if j_bits == 0 {
            0i32
        } else {
            reader.read_bits(j_bits).ok_or(JlsError::Truncated {
                context: "run residual length",
            })? as i32
        };
        // Same out-of-bounds defence as the phase-1 interruption branch: the
        // termination sample lands at `col + residual`, so bound the run length
        // before writing any samples to the row buffer.
        let residual = residual as usize;
        if col + residual >= w {
            return Err(JlsError::Truncated {
                context: "run residual overflows row",
            });
        }
        for k in 0..residual {
            samples[row * w + col + k] = runval as u16;
        }
        col += residual;
        let term_sample =
            decode_run_termination_sample(samples, ctx_states, reader, row, col, runval, p)?;
        samples[row * w + col] = term_sample;
        col += 1;
        decrement_run_index(run_state);
        Ok(col)
    }
}

/// Decode the run-interruption (termination) sample at `(row, col)`.
///
/// The breaking sample uses one of two special contexts (365 or 366) and
/// a special predictor: `PX = Ra = runval`.  The sign-normalised error
/// is mapped through the same lossless / near-lossless mapping that the
/// encoder used.  The context's adaptive `(B, N, k, Cx)` state is then
/// updated identically to the regular path.
fn decode_run_termination_sample(
    samples: &[u16],
    ctx_states: &mut [ContextState],
    reader: &mut BitReader<'_>,
    row: usize,
    col: usize,
    runval: i32,
    p: &ScanDecodeParams,
) -> JlsResult<u16> {
    // `Rb` is the top neighbour of the termination sample (NOT runval).
    let rb = if row > 0 {
        samples[(row - 1) * p.w + col] as i32
    } else {
        // At row 0 the "top neighbour" is replicated from `a` — which is
        // also `runval`.
        runval
    };

    let ctx_idx = run_termination_ctx(runval, rb);
    let state = &mut ctx_states[ctx_idx];

    // RIType: 0 when Ra == Rb, 1 otherwise.  When RIType = 1 the sign is
    // normalised in the direction of (Rb - Ra) so the residual is in the
    // upper half of the range.  When RIType = 0 the sign is fixed to +1.
    let sign = if runval == rb {
        1i32
    } else if rb > runval {
        1i32
    } else {
        -1i32
    };

    // Predicted value is `Ra` (the run value); bias-correct as usual.
    let predicted = runval;
    let corrected_px = (predicted - sign * state.cx).clamp(0, p.max_val);

    let k = state.k.max(0);
    let e_mapped =
        decode_golomb_unsigned_limited(reader, k, p.limit, p.qbpp).ok_or(JlsError::Truncated {
            context: "run termination sample",
        })?;

    let (err_q, rx) = if p.near == 0 {
        let err_abs = unmap_error_lossless(e_mapped);
        let err = err_abs * sign;
        let reconstructed = (corrected_px + err).clamp(0, p.max_val) as u16;
        (err_abs, reconstructed)
    } else {
        let q_step = 2 * p.near + 1;
        let err_q_signed = unmap_error_near(e_mapped, p.near, p.max_val);
        let err = err_q_signed * q_step * sign;
        let reconstructed = (corrected_px + err).clamp(0, p.max_val) as u16;
        (err_q_signed.abs(), reconstructed)
    };

    update_context(state, err_q, p.near, p.reset, p.max_val);

    Ok(rx)
}

// ─── Top-level scan decoder ──────────────────────────────────────────────────

/// Decode one row of one component with RUN-mode dispatch enabled.
///
/// Iterates `col` from `0` to `w`, computing the causal neighbourhood at
/// each position.  When the RUN-entry test passes the row advances by
/// the entire run length (plus its termination sample, if any);
/// otherwise the regular per-pixel decode advances by one.
#[allow(clippy::too_many_arguments)]
fn decode_row_with_run_mode(
    samples: &mut [u16],
    ctx_states: &mut [ContextState],
    run_state: &mut RunState,
    reader: &mut BitReader<'_>,
    row: usize,
    p: &ScanDecodeParams,
) -> JlsResult<()> {
    let w = p.w;
    run_state.reset_at_line_start();
    let mut col = 0usize;
    while col < w {
        let (a, b, c, d) = fetch_neighbours(samples, row, col, w);
        let d1 = d - b;
        let d2 = b - c;
        let d3 = c - a;

        if enter_run_near(d1, d2, d3, p.near) {
            // RUN entry: fill from `col` with `Ra` until interruption or EOL.
            col = decode_run_mode(samples, ctx_states, run_state, reader, row, col, a, p)?;
        } else {
            let rx = decode_pixel_regular(ctx_states, reader, a, b, c, d, p)?;
            samples[row * w + col] = rx;
            col += 1;
        }
    }
    Ok(())
}

/// Decode the compressed scan data into raw sample planes.
fn decode_scan(scan_data: &[u8], headers: &JlsHeaders) -> JlsResult<DecodedImage> {
    let w = headers.frame.width as usize;
    let h = headers.frame.height as usize;
    let nc = headers.frame.num_components as usize;
    let precision = headers.frame.precision;
    let max_val = headers.presets.max_val as i32;
    let near = headers.scan.near as i32;
    let ilv = headers.scan.ilv;

    // Defend against an allocation bomb: `w`, `h` and `nc` come straight from
    // the frame header and drive a `vec![0u16; w*h]` allocation per component
    // below. Reject impossibly large / overflowing declarations up front.
    crate::limits::checked_dims(w, h, nc, 2).map_err(JlsError::Unsupported)?;

    let t1 = headers.presets.t1;
    let t2 = headers.presets.t2;
    let t3 = headers.presets.t3;
    let reset = headers.presets.reset as i32;
    let limit = compute_limit(max_val);
    let qbpp = compute_qbpp(max_val);

    let mut reader = BitReader::new(scan_data);
    let mut all_samples: Vec<Vec<u16>> = (0..nc).map(|_| vec![0u16; w * h]).collect();

    // Each component holds 367 context-state entries: 365 regular + 2
    // RUN-interruption (see [`NUM_TOTAL_CONTEXTS`]).
    let mut all_ctx: Vec<Vec<ContextState>> = (0..nc)
        .map(|_| vec![ContextState::default(); NUM_TOTAL_CONTEXTS])
        .collect();

    let params = ScanDecodeParams {
        max_val,
        near,
        reset,
        limit,
        qbpp,
        t1,
        t2,
        t3,
        w,
    };

    match ilv {
        // ── ILV = 0: non-interleaved — full plane per component ───────────────
        // ILV = 1: line-interleaved — RUN mode active per-component, per-line.
        0 => {
            let mut run_states: Vec<RunState> = (0..nc).map(|_| RunState::new()).collect();
            for comp in 0..nc {
                let ctx_states = &mut all_ctx[comp];
                let run_state = &mut run_states[comp];
                for row in 0..h {
                    decode_row_with_run_mode(
                        &mut all_samples[comp],
                        ctx_states,
                        run_state,
                        &mut reader,
                        row,
                        &params,
                    )?;
                }
            }
        }

        1 => {
            let mut run_states: Vec<RunState> = (0..nc).map(|_| RunState::new()).collect();
            for row in 0..h {
                for comp in 0..nc {
                    let ctx_states = &mut all_ctx[comp];
                    let run_state = &mut run_states[comp];
                    decode_row_with_run_mode(
                        &mut all_samples[comp],
                        ctx_states,
                        run_state,
                        &mut reader,
                        row,
                        &params,
                    )?;
                }
            }
        }

        // ── ILV = 2: sample-interleaved.  Per ISO 14495-1 §F.1 RUN mode is
        //    suspended for ILV = 2 (the per-sample interleaving prevents
        //    the run from coding multiple samples atomically).  Each
        //    component sample is decoded with the regular §A.6 path.
        2 => {
            for row in 0..h {
                for col in 0..w {
                    for comp in 0..nc {
                        let ctx_states = &mut all_ctx[comp];
                        let (a, b, c, d) = fetch_neighbours(&all_samples[comp], row, col, w);
                        let rx =
                            decode_pixel_regular(ctx_states, &mut reader, a, b, c, d, &params)?;
                        all_samples[comp][row * w + col] = rx;
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

    Ok(DecodedImage {
        width: w as u32,
        height: h as u32,
        num_components: nc as u8,
        precision,
        samples: all_samples,
    })
}

// ─── Re-export for use in tests ──────────────────────────────────────────────

/// Re-export the near-lossless map function so that the inline test encoder
/// can use it without duplicating the formula.
pub use super::golomb::map_error_near as golomb_map_error_near;

#[cfg(test)]
mod run_mode_regression_tests {
    use super::*;

    /// Regression for the RUN-mode out-of-bounds write: a run interruption whose
    /// residual length would extend past the end of the row must return a clean
    /// `Err`, never write past the row buffer.
    ///
    /// Setup: a 12-pixel-wide row (row 0), RUN entered at column 10 with
    /// `run_index = 7` (`J[7] = 2`, threshold `1 << 2 = 4`). Because
    /// `10 + 4 > 12`, no full token fits, so the decoder takes the phase-2
    /// trailing-bit path. Byte `0x60 = 0b0110_0000` (read MSB-first) supplies a
    /// `0` trailing bit (run interruption) followed by residual bits `11` = 3,
    /// giving `10 + 3 = 13 >= 12`. Before the fix this wrote `samples[12]`,
    /// one past the 12-element row.
    #[test]
    fn run_residual_past_row_end_errors_not_oob() {
        let w = 12usize;
        let mut samples = vec![0u16; w]; // single row, row = 0
        let mut ctx = vec![ContextState::default(); NUM_TOTAL_CONTEXTS];
        let mut run_state = RunState {
            run_index: 7, // J[7] = 2 → residual is 2 bits, threshold 4
            run_value: 0,
        };
        let scan = [0x60u8];
        let mut reader = BitReader::new(&scan);
        let params = ScanDecodeParams {
            max_val: 255,
            near: 0,
            reset: 64,
            limit: 47,
            qbpp: 8,
            t1: 3,
            t2: 7,
            t3: 21,
            w,
        };

        // Confirm the crafted configuration exercises the phase-2 residual path.
        assert_eq!(threshold_for(7), 4);
        assert_eq!(j_for(7), 2);

        let result = decode_run_mode(
            &mut samples,
            &mut ctx,
            &mut run_state,
            &mut reader,
            0,  // row
            10, // start_col
            0,  // ra (run value)
            &params,
        );
        assert!(
            result.is_err(),
            "a residual overrunning the row must error, got {result:?}"
        );
    }
}
