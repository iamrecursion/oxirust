// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! JPEG XS marker constants and codestream header parser.
//!
//! JPEG XS (ISO/IEC 21122-1:2019) uses JPEG-style two-byte markers with the
//! high byte always `0xFF`. The marker space `0xFF10`–`0xFF1F` is reserved
//! for JPEG XS; `0xFF10` is Start Of Codestream (SOC) and `0xFF11` is End Of
//! Codestream (EOC).
//!
//! # Codestream structure
//!
//! ```text
//! FF 10 (SOC)
//! FF 12 (PIH) — Picture Header
//! FF 13 (CDT) — Component Table
//! FF 14 (WGT) — Weights Table          [optional]
//! FF 15 (NLT) — Nonlinear Transform    [optional]
//! FF 17 (CWD) — Codeword Mapping       [optional]
//! [FF 19 (SLH) slice header + slice data] × N_slices
//! FF 11 (EOC)
//! ```

use super::{JxsError, JxsResult};

// ── Marker constants ──────────────────────────────────────────────────────────

/// Start Of Codestream marker.
pub const SOC: u16 = 0xFF10;
/// End Of Codestream marker.
pub const EOC: u16 = 0xFF11;
/// Picture Header marker.
pub const PIH: u16 = 0xFF12;
/// Component Table marker.
pub const CDT: u16 = 0xFF13;
/// Weights Table marker.
pub const WGT: u16 = 0xFF14;
/// Nonlinear Transform marker.
pub const NLT: u16 = 0xFF15;
/// Codeword Mapping marker.
pub const CWD: u16 = 0xFF17;
/// Slice Header marker.
pub const SLH: u16 = 0xFF19;
/// Compressed Image Segment marker (body follows SLH).
pub const CAP: u16 = 0xFF50;

// ── Profile constants ─────────────────────────────────────────────────────────

/// JPEG XS Main profile code (ISO 21122-2 §A.3.1).
pub const PROFILE_MAIN: u16 = 0x1500;
/// JPEG XS Light profile code.
pub const PROFILE_LIGHT: u16 = 0x1100;
/// JPEG XS High profile code.
pub const PROFILE_HIGH: u16 = 0x2500;

// ── Header structures ─────────────────────────────────────────────────────────

/// JPEG XS Picture Header parameters (PIH marker payload).
#[derive(Debug, Clone)]
pub struct PicHeader {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels (number of lines).
    pub height: u32,
    /// Height of each slice in lines.
    pub slice_height: u32,
    /// Number of image components (colour planes).
    pub num_components: u8,
    /// Sample bit depth (1–16 bits per sample).
    pub bit_depth: u8,
    /// Total codestream length in bytes (0 = undefined / VBR).
    pub codestream_len: u32,
    /// Profile identifier.
    pub profile: u16,
    /// Level identifier.
    pub level: u16,
}

/// Per-component descriptor from the CDT marker.
#[derive(Debug, Clone)]
pub struct ComponentDesc {
    /// Component bit depth (may override `PicHeader::bit_depth`).
    pub bit_depth: u8,
    /// Horizontal chroma subsampling factor (1 = no subsampling).
    pub sx: u8,
    /// Vertical chroma subsampling factor (1 = no subsampling).
    pub sy: u8,
}

/// Slice-level header (SLH marker payload).
#[derive(Debug, Clone)]
pub struct SliceHeader {
    /// Quantisation step size for this slice (scaled by `Fq`).
    pub qp: u16,
    /// Byte offset from the start of the codestream to this slice's data.
    pub data_offset: usize,
    /// Length in bytes of the slice data (compressed payload).
    pub data_len: usize,
}

/// Aggregated JPEG XS headers parsed from the codestream preamble.
#[derive(Debug, Clone)]
pub struct JxsHeaders {
    /// Picture header parameters.
    pub pih: PicHeader,
    /// Per-component descriptors (length == `pih.num_components`).
    pub components: Vec<ComponentDesc>,
    /// Quantisation weights per subband (from WGT marker, if present).
    pub weights: Vec<u16>,
    /// Parsed slice headers (from all SLH markers).
    pub slices: Vec<SliceHeader>,
    /// True if an NLT marker was present in the codestream.
    pub has_nlt: bool,
    /// Raw NLT marker payload bytes (5 bytes: `Tnlt` + `T1`×2 + `T2`×2) when an NLT
    /// marker is present, `None` otherwise. Use `parse_nlt_payload` to decode.
    pub nlt_payload: Option<Vec<u8>>,
}

// ── Low-level byte helpers ────────────────────────────────────────────────────

/// Read a big-endian `u16` from `data[pos..]` without bounds confusion.
///
/// # Errors
/// `JxsError::TruncatedStream` if `pos + 2 > data.len()`.
fn read_u16(data: &[u8], pos: usize) -> JxsResult<u16> {
    if pos + 2 > data.len() {
        return Err(JxsError::TruncatedStream {
            need: pos + 2,
            have: data.len(),
        });
    }
    Ok(u16::from_be_bytes([data[pos], data[pos + 1]]))
}

/// Read a big-endian `u32` from `data[pos..]`.
///
/// # Errors
/// `JxsError::TruncatedStream` if `pos + 4 > data.len()`.
fn read_u32(data: &[u8], pos: usize) -> JxsResult<u32> {
    if pos + 4 > data.len() {
        return Err(JxsError::TruncatedStream {
            need: pos + 4,
            have: data.len(),
        });
    }
    Ok(u32::from_be_bytes([
        data[pos],
        data[pos + 1],
        data[pos + 2],
        data[pos + 3],
    ]))
}

/// Read a 3-byte big-endian unsigned integer.
fn read_u24(data: &[u8], pos: usize) -> JxsResult<u32> {
    if pos + 3 > data.len() {
        return Err(JxsError::TruncatedStream {
            need: pos + 3,
            have: data.len(),
        });
    }
    Ok(u32::from(data[pos]) << 16 | u32::from(data[pos + 1]) << 8 | u32::from(data[pos + 2]))
}

// ── PIH parser ────────────────────────────────────────────────────────────────

/// Parse the PIH (Picture Header) marker payload.
///
/// `payload` is the marker body starting immediately after the 2-byte length
/// field (i.e. the first byte of the actual PIH data).
///
/// ISO 21122-1 §A.4 defines the following sequential fields:
/// - `Lcod`  3 bytes — total codestream length
/// - `Ppih`  2 bytes — profile
/// - `Plev`  2 bytes — level
/// - `Wf`    2 bytes — frame width  (Lc)
/// - `Hf`    2 bytes — frame height (Lr)
/// - `Cw`    2 bytes — codegroup width (not used here)
/// - `Hsl`   2 bytes — slice height
/// - `Nc`    1 byte  — number of components
/// - `Ng`    1 byte  — ganging factor
/// - `Ss`    1 byte  — sample size (bit depth)
/// - `Bw`    1 byte  — extended Bw field
/// - `Fq`    1 byte  — fractional bits in QP
/// - `Br`    3 bytes — target bitrate (0 = VBR)
/// - `Fsl`   1 byte  — slice fraction
/// - `Ppoc`  1 byte  — progressive order
/// - `Cpih`  1 byte  — coding mode bitmask
fn parse_pih_payload(payload: &[u8]) -> JxsResult<PicHeader> {
    // Minimum payload size for the mandatory fields:
    // 3+2+2+2+2+2+2+1+1+1+1+1+3+1+1+1 = 26 bytes
    if payload.len() < 26 {
        return Err(JxsError::InvalidHeader(format!(
            "PIH payload too short: {} < 26 bytes",
            payload.len()
        )));
    }

    let codestream_len = read_u24(payload, 0)?;
    let profile = read_u16(payload, 3)?;
    let level = read_u16(payload, 5)?;
    let width = u32::from(read_u16(payload, 7)?);
    let height = u32::from(read_u16(payload, 9)?);
    // Cw (codegroup width) at offset 11 — not needed for basic decode
    let slice_height = u32::from(read_u16(payload, 13)?);
    let num_components = payload[15];
    // Ng at offset 16 — ganging, skip
    let bit_depth = payload[17];
    // Bw at 18, Fq at 19, Br at 20..22, Fsl at 23, Ppoc at 24, Cpih at 25

    if num_components == 0 {
        return Err(JxsError::InvalidHeader(
            "PIH Nc=0: at least one component required".to_string(),
        ));
    }
    if bit_depth == 0 || bit_depth > 16 {
        return Err(JxsError::InvalidHeader(format!(
            "PIH Ss={bit_depth}: bit depth must be 1–16"
        )));
    }
    if width == 0 || height == 0 {
        return Err(JxsError::InvalidHeader(format!(
            "PIH frame size {width}x{height}: dimensions must be non-zero"
        )));
    }
    let effective_slice_height = if slice_height == 0 {
        height
    } else {
        slice_height
    };

    Ok(PicHeader {
        width,
        height,
        slice_height: effective_slice_height,
        num_components,
        bit_depth,
        codestream_len,
        profile,
        level,
    })
}

// ── CDT parser ────────────────────────────────────────────────────────────────

/// Parse the CDT (Component Table) marker payload.
///
/// Layout per component (3 bytes each):
/// - `Bdc[c]` 1 byte — component bit depth
/// - `Sx[c]`  1 byte — horizontal subsampling factor (0 = same as Ss)
/// - `Sy[c]`  1 byte — vertical subsampling factor
fn parse_cdt_payload(payload: &[u8], num_components: u8) -> JxsResult<Vec<ComponentDesc>> {
    let nc = num_components as usize;
    if payload.len() < nc * 3 {
        return Err(JxsError::InvalidHeader(format!(
            "CDT payload too short for {nc} components: {} < {} bytes",
            payload.len(),
            nc * 3
        )));
    }
    let mut components = Vec::with_capacity(nc);
    for c in 0..nc {
        let base = c * 3;
        let bit_depth = payload[base];
        let sx = if payload[base + 1] == 0 {
            1
        } else {
            payload[base + 1]
        };
        let sy = if payload[base + 2] == 0 {
            1
        } else {
            payload[base + 2]
        };
        components.push(ComponentDesc { bit_depth, sx, sy });
    }
    Ok(components)
}

// ── WGT parser ────────────────────────────────────────────────────────────────

/// Parse the WGT (Weights Table) marker payload.
///
/// Each weight is a `u16` big-endian value. The number of weights equals the
/// number of subbands (determined by the profile / decomposition level).
/// We parse all available u16 pairs without hard-coding the count.
fn parse_wgt_payload(payload: &[u8]) -> Vec<u16> {
    let num_weights = payload.len() / 2;
    let mut weights = Vec::with_capacity(num_weights);
    for i in 0..num_weights {
        let w = u16::from_be_bytes([payload[i * 2], payload[i * 2 + 1]]);
        weights.push(w);
    }
    weights
}

// ── SLH parser ────────────────────────────────────────────────────────────────

/// Parse one SLH (Slice Header) marker payload.
///
/// Layout (ISO 21122-1 §A.7):
/// - `Lslh`  2 bytes — slice header length (including these 2 bytes)
/// - `Qpih`  2 bytes — quantisation step (global QP for this slice)
/// - `Bslh`  variable — per-band rate targets (not parsed here)
///
/// `data_pos` is the byte offset in the original codestream where the
/// compressed slice data begins (immediately after the SLH marker+payload).
fn parse_slh_payload(
    payload: &[u8],
    data_offset: usize,
    data_len: usize,
) -> JxsResult<SliceHeader> {
    // Minimum: 2 bytes of Qpih (Lslh is already consumed before this call)
    if payload.len() < 2 {
        return Err(JxsError::InvalidHeader(format!(
            "SLH payload too short: {} < 2 bytes",
            payload.len()
        )));
    }
    let qp = read_u16(payload, 0)?;
    Ok(SliceHeader {
        qp,
        data_offset,
        data_len,
    })
}

// ── Master header parser ──────────────────────────────────────────────────────

/// Parse all JPEG XS codestream headers.
///
/// Reads sequentially from `data[0]` until the first slice data segment or
/// until EOC is encountered. Returns `(JxsHeaders, header_end_offset)` where
/// `header_end_offset` is the byte index in `data` immediately after the last
/// header marker (i.e. the start of the first slice's compressed data).
///
/// # Errors
/// Returns `JxsError::InvalidMarker` if SOC is not present, or structural
/// errors on malformed marker sequences.
pub fn parse_headers(data: &[u8]) -> JxsResult<(JxsHeaders, usize)> {
    // ── 1. Verify SOC ────────────────────────────────────────────────────────
    if data.len() < 2 {
        return Err(JxsError::TruncatedStream {
            need: 2,
            have: data.len(),
        });
    }
    let first_marker = read_u16(data, 0)?;
    if first_marker != SOC {
        return Err(JxsError::InvalidMarker {
            expected: SOC,
            got: first_marker,
        });
    }

    let mut pos = 2usize; // skip SOC (no payload)
    let mut pih_opt: Option<PicHeader> = None;
    let mut components: Vec<ComponentDesc> = Vec::new();
    let mut weights: Vec<u16> = Vec::new();
    let mut slices: Vec<SliceHeader> = Vec::new();
    let mut has_nlt = false;
    let mut nlt_payload: Option<Vec<u8>> = None;

    loop {
        if pos + 2 > data.len() {
            return Err(JxsError::TruncatedStream {
                need: pos + 2,
                have: data.len(),
            });
        }
        let marker = read_u16(data, pos)?;
        pos += 2;

        match marker {
            EOC => {
                // End of codestream — no payload
                break;
            }
            PIH => {
                // PIH: 2-byte length Lp (includes the 2 length bytes), then payload
                let lp = read_u16(data, pos)? as usize;
                if lp < 2 {
                    return Err(JxsError::InvalidHeader(format!(
                        "PIH Lp={lp} too small (minimum 2)"
                    )));
                }
                let payload_len = lp - 2;
                pos += 2;
                if pos + payload_len > data.len() {
                    return Err(JxsError::TruncatedStream {
                        need: pos + payload_len,
                        have: data.len(),
                    });
                }
                let payload = &data[pos..pos + payload_len];
                pih_opt = Some(parse_pih_payload(payload)?);
                pos += payload_len;
            }
            CDT => {
                let lp = read_u16(data, pos)? as usize;
                let payload_len = lp.saturating_sub(2);
                pos += 2;
                if pos + payload_len > data.len() {
                    return Err(JxsError::TruncatedStream {
                        need: pos + payload_len,
                        have: data.len(),
                    });
                }
                let payload = &data[pos..pos + payload_len];
                let nc = pih_opt.as_ref().map(|p| p.num_components).unwrap_or(0);
                if nc > 0 {
                    components = parse_cdt_payload(payload, nc)?;
                }
                pos += payload_len;
            }
            WGT => {
                let lp = read_u16(data, pos)? as usize;
                let payload_len = lp.saturating_sub(2);
                pos += 2;
                if pos + payload_len > data.len() {
                    return Err(JxsError::TruncatedStream {
                        need: pos + payload_len,
                        have: data.len(),
                    });
                }
                weights = parse_wgt_payload(&data[pos..pos + payload_len]);
                pos += payload_len;
            }
            NLT => {
                let lp = read_u16(data, pos)? as usize;
                let payload_len = lp.saturating_sub(2);
                pos += 2;
                if pos + payload_len > data.len() {
                    return Err(JxsError::TruncatedStream {
                        need: pos + payload_len,
                        have: data.len(),
                    });
                }
                // Capture the raw NLT payload bytes so the decoder can call
                // `parse_nlt_payload` without re-scanning the codestream.
                nlt_payload = Some(data[pos..pos + payload_len].to_vec());
                pos += payload_len;
                has_nlt = true;
            }
            SLH => {
                // SLH: 2-byte Lslh (includes these 2 bytes), then Qpih (2 bytes), then band info
                let lslh = read_u16(data, pos)? as usize;
                if lslh < 2 {
                    return Err(JxsError::InvalidHeader(format!(
                        "SLH Lslh={lslh} too small"
                    )));
                }
                let payload_len = lslh - 2;
                pos += 2;
                if pos + payload_len > data.len() {
                    return Err(JxsError::TruncatedStream {
                        need: pos + payload_len,
                        have: data.len(),
                    });
                }
                let payload = &data[pos..pos + payload_len];
                // Compressed slice data immediately follows SLH — we'll compute its length
                // later once we know the next marker; for now record where it starts.
                let data_start = pos + payload_len;
                // Determine the slice data length.
                //
                // JPEG XS entropy payloads are not byte-stuffed, so the compressed
                // bytes may legitimately contain `0xFF`-prefixed patterns that look
                // like markers. To avoid mistaking such a coefficient byte for the
                // real terminator, prefer the trailing EOC marker as the slice end:
                // a single-slice codestream is always `… SLH | slice data | EOC`, so
                // the slice data runs from `data_start` up to the final EOC. If the
                // codestream does not end in EOC (e.g. a malformed or multi-segment
                // stream), fall back to scanning for the next marker.
                let slice_data_len = slice_data_len_to_trailing_eoc(data, data_start)
                    .unwrap_or_else(|| find_next_marker_offset(data, data_start));
                let slh = parse_slh_payload(payload, data_start, slice_data_len)?;
                slices.push(slh);
                pos += payload_len;
                // Skip over the slice data — it will be parsed by the entropy decoder later.
                pos += slice_data_len;
            }
            // Unknown / ignored markers — skip by reading their length field
            _ if (marker & 0xFF00) == 0xFF00 => {
                if pos + 2 > data.len() {
                    return Err(JxsError::TruncatedStream {
                        need: pos + 2,
                        have: data.len(),
                    });
                }
                let lp = read_u16(data, pos)? as usize;
                let payload_len = lp.saturating_sub(2);
                pos += 2 + payload_len;
            }
            _ => {
                return Err(JxsError::InvalidMarker {
                    expected: 0xFF00,
                    got: marker,
                });
            }
        }
    }

    let pih = pih_opt.ok_or_else(|| JxsError::InvalidHeader("missing PIH marker".to_string()))?;

    // Fill in default component descriptors if CDT was absent.
    if components.is_empty() {
        components = (0..pih.num_components)
            .map(|_| ComponentDesc {
                bit_depth: pih.bit_depth,
                sx: 1,
                sy: 1,
            })
            .collect();
    }

    Ok((
        JxsHeaders {
            pih,
            components,
            weights,
            slices,
            has_nlt,
            nlt_payload,
        },
        pos,
    ))
}

/// Compute the slice-data length assuming the codestream terminates with an
/// `EOC` marker (`0xFF11`) as its final two bytes.
///
/// JPEG XS entropy payloads are **not** byte-stuffed, so the compressed slice
/// bytes may contain `0xFF`-prefixed patterns that resemble markers. For the
/// single-slice codestream layout produced by the encoder
/// (`… SLH | slice data | EOC`), the slice data is exactly the bytes from
/// `start` up to the trailing `EOC`. This function returns that length, or
/// `None` if `data` does not end in `EOC` (so the caller can fall back to a
/// forward marker scan).
fn slice_data_len_to_trailing_eoc(data: &[u8], start: usize) -> Option<usize> {
    let n = data.len();
    if n < 2 || start > n - 2 {
        return None;
    }
    if data[n - 2] == 0xFF && data[n - 1] == (EOC & 0x00FF) as u8 {
        Some((n - 2) - start)
    } else {
        None
    }
}

/// Scan forward from `start` to find the next `0xFF` marker prefix, returning
/// the length of the data segment before it (i.e. the slice data length).
///
/// Returns `data.len() - start` if no marker is found (data extends to end).
fn find_next_marker_offset(data: &[u8], start: usize) -> usize {
    let mut i = start;
    while i + 1 < data.len() {
        if data[i] == 0xFF && data[i + 1] != 0x00 {
            return i - start;
        }
        i += 1;
    }
    data.len() - start
}

// ── Build a minimal synthetic codestream for tests ────────────────────────────

/// Build a minimal JPEG XS codestream that can be parsed by `parse_headers`.
///
/// Produces: SOC + PIH + CDT + EOC. No slice data. Useful for unit tests.
pub fn build_test_codestream(
    width: u16,
    height: u16,
    slice_height: u16,
    num_components: u8,
    bit_depth: u8,
) -> Vec<u8> {
    let mut buf = Vec::new();

    // SOC
    buf.extend_from_slice(&SOC.to_be_bytes());

    // PIH — 26 bytes of payload (Lp = 28, i.e. 2 length bytes + 26 payload bytes)
    buf.extend_from_slice(&PIH.to_be_bytes());
    let lp: u16 = 28; // 2 (Lp self) + 26 payload bytes
    buf.extend_from_slice(&lp.to_be_bytes());
    // Lcod  (3 bytes): 0 = VBR
    buf.extend_from_slice(&[0x00, 0x00, 0x00]);
    // Ppih  (2 bytes): MAIN profile
    buf.extend_from_slice(&PROFILE_MAIN.to_be_bytes());
    // Plev  (2 bytes): level 0
    buf.extend_from_slice(&0x0000u16.to_be_bytes());
    // Wf    (2 bytes): frame width
    buf.extend_from_slice(&width.to_be_bytes());
    // Hf    (2 bytes): frame height
    buf.extend_from_slice(&height.to_be_bytes());
    // Cw    (2 bytes): codegroup width = frame width (no tiling)
    buf.extend_from_slice(&width.to_be_bytes());
    // Hsl   (2 bytes): slice height
    buf.extend_from_slice(&slice_height.to_be_bytes());
    // Nc    (1 byte): number of components
    buf.push(num_components);
    // Ng    (1 byte): no ganging
    buf.push(0x00);
    // Ss    (1 byte): sample bit depth
    buf.push(bit_depth);
    // Bw    (1 byte): extended bitwidth = 0
    buf.push(0x00);
    // Fq    (1 byte): fractional QP bits = 0
    buf.push(0x00);
    // Br    (3 bytes): 0 = VBR
    buf.extend_from_slice(&[0x00, 0x00, 0x00]);
    // Fsl   (1 byte)
    buf.push(0x00);
    // Ppoc  (1 byte)
    buf.push(0x00);
    // Cpih  (1 byte)
    buf.push(0x00);

    // CDT — 3 bytes per component
    buf.extend_from_slice(&CDT.to_be_bytes());
    let cdt_payload_len = num_components as usize * 3;
    let cdt_lp = (cdt_payload_len + 2) as u16;
    buf.extend_from_slice(&cdt_lp.to_be_bytes());
    for _c in 0..num_components {
        buf.push(bit_depth); // Bdc
        buf.push(0x01); // Sx = 1
        buf.push(0x01); // Sy = 1
    }

    // EOC
    buf.extend_from_slice(&EOC.to_be_bytes());

    buf
}

/// Build a minimal JPEG XS codestream containing an NLT marker with the given
/// quadratic parameters. Produces: SOC + PIH + CDT + NLT + EOC. No slice data.
///
/// The NLT payload is 5 bytes: `Tnlt` (1) + `T1` (2, big-endian) + `T2` (2, big-endian).
pub fn build_test_codestream_with_nlt(
    width: u16,
    height: u16,
    slice_height: u16,
    num_components: u8,
    bit_depth: u8,
    t1: u16,
    t2: u16,
) -> Vec<u8> {
    // Start with the base codestream, then insert NLT before EOC.
    let mut base = build_test_codestream(width, height, slice_height, num_components, bit_depth);
    // Remove the trailing EOC (last 2 bytes).
    let eoc_pos = base.len() - 2;
    base.truncate(eoc_pos);

    // NLT marker — Lnlt = 7 (2 length bytes + 5 payload bytes)
    base.extend_from_slice(&NLT.to_be_bytes());
    let lnlt: u16 = 7;
    base.extend_from_slice(&lnlt.to_be_bytes());
    // Tnlt = 0 (quadratic)
    base.push(0x00);
    // T1 (2 bytes)
    base.extend_from_slice(&t1.to_be_bytes());
    // T2 (2 bytes)
    base.extend_from_slice(&t2.to_be_bytes());

    // Re-append EOC
    base.extend_from_slice(&EOC.to_be_bytes());
    base
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soc_marker_value() {
        assert_eq!(SOC, 0xFF10);
    }

    #[test]
    fn eoc_marker_value() {
        assert_eq!(EOC, 0xFF11);
    }

    #[test]
    fn pih_marker_value() {
        assert_eq!(PIH, 0xFF12);
    }

    #[test]
    fn parse_minimal_test_codestream() {
        let data = build_test_codestream(640, 480, 16, 3, 8);
        let (headers, _end) = parse_headers(&data).expect("parse_headers");
        assert_eq!(headers.pih.width, 640);
        assert_eq!(headers.pih.height, 480);
        assert_eq!(headers.pih.slice_height, 16);
        assert_eq!(headers.pih.num_components, 3);
        assert_eq!(headers.pih.bit_depth, 8);
        assert_eq!(headers.components.len(), 3);
        assert_eq!(headers.components[0].sx, 1);
        assert_eq!(headers.components[0].sy, 1);
        assert!(!headers.has_nlt);
        assert!(headers.nlt_payload.is_none());
    }

    #[test]
    fn parse_codestream_with_nlt_marker_captures_payload() {
        let data = build_test_codestream_with_nlt(64, 64, 8, 1, 8, 64, 192);
        let (headers, _end) = parse_headers(&data).expect("parse_headers");
        assert!(headers.has_nlt);
        assert!(headers.nlt_payload.is_some());
        let payload = headers.nlt_payload.unwrap();
        // Tnlt = 0 (quadratic)
        assert_eq!(payload[0], 0x00);
        // T1 = 64 = 0x0040
        assert_eq!(u16::from_be_bytes([payload[1], payload[2]]), 64);
        // T2 = 192 = 0x00C0
        assert_eq!(u16::from_be_bytes([payload[3], payload[4]]), 192);
    }

    #[test]
    fn parse_codestream_without_nlt_has_no_payload() {
        let data = build_test_codestream(32, 32, 8, 1, 8);
        let (headers, _) = parse_headers(&data).expect("parse_headers");
        assert!(!headers.has_nlt);
        assert!(headers.nlt_payload.is_none());
    }

    #[test]
    fn parse_no_soc_returns_error() {
        let data = [0x00u8, 0x00, 0x00, 0x00];
        let result = parse_headers(&data);
        assert!(result.is_err());
        if let Err(JxsError::InvalidMarker { expected, got }) = result {
            assert_eq!(expected, SOC);
            assert_eq!(got, 0x0000);
        } else {
            panic!("expected InvalidMarker error");
        }
    }

    #[test]
    fn parse_truncated_after_soc_returns_error() {
        let data = [0xFFu8, 0x10]; // just SOC
        let result = parse_headers(&data);
        // After SOC we need at least another marker — should get TruncatedStream
        assert!(result.is_err());
    }

    #[test]
    fn build_test_codestream_single_component_10bit() {
        let data = build_test_codestream(1920, 1080, 32, 1, 10);
        let (headers, _) = parse_headers(&data).expect("parse");
        assert_eq!(headers.pih.width, 1920);
        assert_eq!(headers.pih.height, 1080);
        assert_eq!(headers.pih.bit_depth, 10);
        assert_eq!(headers.pih.num_components, 1);
    }

    #[test]
    fn find_next_marker_at_start() {
        let data = [0xFFu8, 0x11, 0x00, 0x00]; // EOC at start
        let len = find_next_marker_offset(&data, 0);
        assert_eq!(len, 0);
    }

    #[test]
    fn find_next_marker_not_found_returns_full_length() {
        let data = [0x00u8, 0x01, 0x02, 0x03];
        let len = find_next_marker_offset(&data, 0);
        assert_eq!(len, 4);
    }
}
