//! JPEG-LS marker parsing (ISO 14495-1 Annex C).
//!
//! JPEG-LS reuses the JPEG framing format (SOI/EOI/markers) but defines its
//! own SOF55 (0xFFF7) and LSE (0xFFF8) markers for LOCO-I parameters.

use super::{JlsError, JlsResult};

/// Start Of Image marker (shared with baseline JPEG).
pub const SOI: u16 = 0xFFD8;
/// End Of Image marker.
pub const EOI: u16 = 0xFFD9;
/// Start Of Frame for JPEG-LS (not the standard JPEG SOF markers).
pub const SOF55: u16 = 0xFFF7;
/// JPEG-LS Preset Parameters extension marker.
pub const LSE: u16 = 0xFFF8;
/// Start Of Scan marker.
pub const SOS: u16 = 0xFFDA;
/// Define Number of Lines marker.
pub const DNL: u16 = 0xFFDC;
/// Define Restart Interval marker.
pub const DRI: u16 = 0xFFDD;

/// Frame parameters extracted from the SOF55 marker.
#[derive(Debug, Clone)]
pub struct FrameParams {
    /// Bits per sample (1–16). Default is 8.
    pub precision: u8,
    /// Image height in lines.
    pub height: u16,
    /// Image width in samples per line.
    pub width: u16,
    /// Number of components (1 = greyscale, 3 = colour).
    pub num_components: u8,
    /// Per-component descriptor list (length = `num_components`).
    pub components: Vec<ComponentSpec>,
}

/// Single-component descriptor within a SOF55 marker.
#[derive(Debug, Clone)]
pub struct ComponentSpec {
    /// Component identifier byte.
    pub id: u8,
    /// Horizontal sampling factor (high nibble of the HiVi byte).
    pub h_factor: u8,
    /// Vertical sampling factor (low nibble of the HiVi byte).
    pub v_factor: u8,
    /// Quantisation table index (unused in JPEG-LS; always 0).
    pub quant_table_idx: u8,
}

/// JPEG-LS preset parameters (from LSE marker or SPLS defaults).
#[derive(Debug, Clone)]
pub struct JlsPresetParams {
    /// MaxVal — maximum sample value. Default = `(1 << precision) - 1`.
    pub max_val: u16,
    /// T1 threshold for context quantisation.
    pub t1: i32,
    /// T2 threshold for context quantisation.
    pub t2: i32,
    /// T3 threshold for context quantisation.
    pub t3: i32,
    /// Reset interval Rt for adaptive statistics. Default = 64.
    pub reset: u16,
}

impl JlsPresetParams {
    /// Compute default preset parameters per ISO 14495-1 Annex C.2 / Table C.1.
    #[must_use]
    pub fn default_for_precision(precision: u8) -> Self {
        // Defend against `1u32 << precision` overflow (shift >= 32 panics):
        // JPEG-LS sample precision is 2..=16, so clamp. A larger value is
        // malformed and the clamp never affects a well-formed stream.
        let precision = precision.min(16);
        let max_val = (1u32 << precision) - 1;
        let (t1, t2, t3) = if max_val >= 128 {
            let factor = (max_val as i32 + 127) / 256;
            let t1 = (factor * 1 + 10).max(2).min(max_val as i32 + 1);
            let t2 = (factor * 4 + 11).max(3).min(max_val as i32 + 1);
            let t3 = (factor * 17 + 12).max(4).min(max_val as i32 + 1);
            (t1, t2, t3)
        } else {
            let mv = max_val as i32;
            ((mv / 4).max(2), (mv / 2).max(3), (3 * mv / 4).max(4))
        };
        Self {
            max_val: max_val as u16,
            t1,
            t2,
            t3,
            reset: 64,
        }
    }
}

/// Scan header extracted from the SOS marker.
#[derive(Debug, Clone)]
pub struct ScanHeader {
    /// NEAR parameter: 0 = lossless, >0 = near-lossless.
    pub near: u8,
    /// ILV interleave mode: 0 = non-interleaved, 1 = line, 2 = sample.
    pub ilv: u8,
    /// Pt point transform (shift right before encoding). Usually 0.
    pub pt: u8,
    /// Component identifiers included in this scan.
    pub component_ids: Vec<u8>,
}

/// All parsed headers required by the decoder, plus the offset into the raw
/// input where compressed scan data begins.
#[derive(Debug, Clone)]
pub struct JlsHeaders {
    /// Frame geometry and component layout.
    pub frame: FrameParams,
    /// Golomb/context preset parameters.
    pub presets: JlsPresetParams,
    /// Scan-level parameters (NEAR, ILV, Pt).
    pub scan: ScanHeader,
    /// Byte offset within the original input slice where scan data starts.
    pub scan_data_start: usize,
}

/// Read a big-endian `u16` from `data` at `pos`, advancing `pos` by 2.
fn read_u16(data: &[u8], pos: &mut usize) -> JlsResult<u16> {
    if *pos + 2 > data.len() {
        return Err(JlsError::Truncated {
            context: "marker read_u16",
        });
    }
    let v = u16::from_be_bytes([data[*pos], data[*pos + 1]]);
    *pos += 2;
    Ok(v)
}

/// Read a single byte from `data` at `pos`, advancing `pos` by 1.
fn read_u8(data: &[u8], pos: &mut usize) -> JlsResult<u8> {
    if *pos >= data.len() {
        return Err(JlsError::Truncated {
            context: "marker read_u8",
        });
    }
    let v = data[*pos];
    *pos += 1;
    Ok(v)
}

/// Parse the SOF55 frame header.
fn parse_sof55(data: &[u8], pos: &mut usize, len: usize) -> JlsResult<FrameParams> {
    let start = *pos;
    let precision = read_u8(data, pos)?;
    let height = read_u16(data, pos)?;
    let width = read_u16(data, pos)?;
    let num_components = read_u8(data, pos)?;

    let mut components = Vec::with_capacity(num_components as usize);
    for _ in 0..num_components {
        let id = read_u8(data, pos)?;
        let hv = read_u8(data, pos)?;
        let tq = read_u8(data, pos)?;
        components.push(ComponentSpec {
            id,
            h_factor: hv >> 4,
            v_factor: hv & 0x0F,
            quant_table_idx: tq,
        });
    }

    // Verify we consumed exactly `len - 2` bytes (marker length includes the 2-byte
    // length field itself but parse_sof55 is called after it was read).
    let consumed = *pos - start;
    if consumed != len - 2 {
        *pos = start + len - 2;
    }

    Ok(FrameParams {
        precision,
        height,
        width,
        num_components,
        components,
    })
}

/// Parse the LSE preset-parameters marker (Id = 1).
fn parse_lse(data: &[u8], pos: &mut usize, len: usize) -> JlsResult<Option<JlsPresetParams>> {
    if len < 2 + 11 {
        *pos += len - 2;
        return Ok(None);
    }
    let id = read_u8(data, pos)?;
    if id != 1 {
        *pos += len - 3;
        return Ok(None);
    }
    let max_val = read_u16(data, pos)?;
    let t1 = read_u16(data, pos)? as i32;
    let t2 = read_u16(data, pos)? as i32;
    let t3 = read_u16(data, pos)? as i32;
    let reset = read_u16(data, pos)?;
    Ok(Some(JlsPresetParams {
        max_val,
        t1,
        t2,
        t3,
        reset,
    }))
}

/// Parse the SOS scan header.
fn parse_sos(data: &[u8], pos: &mut usize, len: usize) -> JlsResult<ScanHeader> {
    let start = *pos;
    let ns = read_u8(data, pos)?;
    let mut component_ids = Vec::with_capacity(ns as usize);
    for _ in 0..ns {
        let cs = read_u8(data, pos)?;
        let _td_ta = read_u8(data, pos)?; // not used in JPEG-LS
        component_ids.push(cs);
    }
    // Per ISO 14495-1: Ss = NEAR, Se = 0 (unused), Ah = ILV, Al = Pt
    let near = read_u8(data, pos)?;
    let ilv = read_u8(data, pos)?;
    let pt = read_u8(data, pos)?;

    let consumed = *pos - start;
    if consumed < len - 2 {
        *pos = start + len - 2;
    }

    Ok(ScanHeader {
        near,
        ilv,
        pt,
        component_ids,
    })
}

/// Inner marker-walk loop, separated to avoid the `unused_assignments` lint
/// on the outer mutable `Option` accumulators.
fn walk_markers(data: &[u8], start_pos: usize) -> JlsResult<JlsHeaders> {
    let mut pos = start_pos;
    let mut frame_opt: Option<FrameParams> = None;
    let mut presets_opt: Option<JlsPresetParams> = None;

    let scan = loop {
        if pos + 2 > data.len() {
            return Err(JlsError::Truncated {
                context: "marker scan",
            });
        }
        if data[pos] != 0xFF {
            return Err(JlsError::InvalidMarker(0xFF00 | data[pos] as u16));
        }
        let marker_byte = data[pos + 1];
        let marker = 0xFF00u16 | marker_byte as u16;
        pos += 2;

        // Markers with no length field
        if marker_byte == 0x00 || marker_byte == 0xFF {
            continue;
        }
        // SOI/EOI have no length field
        if marker == SOI || marker == EOI {
            continue;
        }

        // Read length (includes the 2-byte length field itself)
        if pos + 2 > data.len() {
            return Err(JlsError::Truncated {
                context: "marker length",
            });
        }
        let seg_len = read_u16(data, &mut pos)? as usize;
        if seg_len < 2 {
            return Err(JlsError::InvalidMarker(marker));
        }

        let seg_start = pos;
        let seg_data_len = seg_len - 2; // bytes after the length field

        match marker {
            SOF55 => {
                if data.len() < pos + seg_data_len {
                    return Err(JlsError::Truncated { context: "SOF55" });
                }
                frame_opt = Some(parse_sof55(data, &mut pos, seg_len)?);
            }
            LSE => {
                if data.len() < pos + seg_data_len {
                    return Err(JlsError::Truncated { context: "LSE" });
                }
                if let Some(p) = parse_lse(data, &mut pos, seg_len)? {
                    presets_opt = Some(p);
                }
            }
            SOS => {
                if data.len() < pos + seg_data_len {
                    return Err(JlsError::Truncated { context: "SOS" });
                }
                break parse_sos(data, &mut pos, seg_len)?;
            }
            _ => {
                // Skip unknown / irrelevant markers (APP0–APPn, COM, DRI, etc.)
                pos = seg_start + seg_data_len;
            }
        }

        // Guard against infinite loops on malformed input
        if pos >= data.len() {
            return Err(JlsError::Truncated {
                context: "marker walk",
            });
        }
    };

    let frame = frame_opt.ok_or(JlsError::NotJpegLs)?;
    let presets =
        presets_opt.unwrap_or_else(|| JlsPresetParams::default_for_precision(frame.precision));

    Ok(JlsHeaders {
        presets,
        scan,
        scan_data_start: pos,
        frame,
    })
}

/// Parse all JPEG-LS headers from `data`.
///
/// Returns a [`JlsHeaders`] struct containing all frame/scan parameters and
/// the byte offset where compressed scan data starts.
pub fn parse_headers(data: &[u8]) -> JlsResult<JlsHeaders> {
    if data.len() < 4 {
        return Err(JlsError::NotJpegLs);
    }

    // Must begin with SOI
    let soi = u16::from_be_bytes([data[0], data[1]]);
    if soi != SOI {
        return Err(JlsError::NotJpegLs);
    }

    // The third and fourth bytes must be 0xFF + SOF55 byte (0xF7) or another
    // marker. We check during the walk below that SOF55 is present.

    walk_markers(data, 2usize)
}
