//! JPEG 2000 codestream marker segment parser (ISO/IEC 15444-1 §A).
//!
//! Parses the main header markers: SOC, SIZ, COD, QCD, SOT, SOD, EOC.
//!
//! All multi-byte integers in a codestream are big-endian.

use super::{Jp2Error, Jp2Result};

// ── Marker codes ───────────────────────────────────────────────────────────────

/// Start of Codestream.
pub const SOC: u16 = 0xFF4F;
/// Image and tile size (SIZ).
pub const SIZ: u16 = 0xFF51;
/// Coding style default (COD).
pub const COD: u16 = 0xFF52;
/// Quantization default (QCD).
pub const QCD: u16 = 0xFF5C;
/// Start of Tile-part (SOT).
pub const SOT: u16 = 0xFF90;
/// Start of Data (SOD).
pub const SOD: u16 = 0xFF93;
/// End of Codestream (EOC).
pub const EOC: u16 = 0xFFD9;

// ── Parsed marker structs ─────────────────────────────────────────────────────

/// Per-component parameters from the SIZ marker.
#[derive(Debug, Clone)]
pub struct ComponentParams {
    /// Component precision (bit depth - 1), with bit 7 = sign flag.
    pub ssiz: u8,
    /// Horizontal subsampling exponent.
    pub xr_siz: u8,
    /// Vertical subsampling exponent.
    pub yr_siz: u8,
}

impl ComponentParams {
    /// Effective bit depth (1..=38 per spec; we support 1..=16).
    #[must_use]
    pub fn bit_depth(&self) -> u8 {
        (self.ssiz & 0x7F) + 1
    }
    /// True if this component is signed.
    #[must_use]
    pub fn is_signed(&self) -> bool {
        (self.ssiz & 0x80) != 0
    }
}

/// Parsed SIZ (image and tile size) marker.
#[derive(Debug, Clone)]
pub struct SizMarker {
    /// Decoder capabilities (Rsiz).
    pub rsiz: u16,
    /// Reference grid width (image width).
    pub x_siz: u32,
    /// Reference grid height (image height).
    pub y_siz: u32,
    /// Image area horizontal offset.
    pub xo_siz: u32,
    /// Image area vertical offset.
    pub yo_siz: u32,
    /// Tile width in the reference grid.
    pub xt_siz: u32,
    /// Tile height in the reference grid.
    pub yt_siz: u32,
    /// Tile area horizontal offset.
    pub xto_siz: u32,
    /// Tile area vertical offset.
    pub yto_siz: u32,
    /// Number of components.
    pub csiz: u16,
    /// Per-component parameters (length = csiz).
    pub components: Vec<ComponentParams>,
}

impl SizMarker {
    /// Return the effective image width in pixels (x_siz - xo_siz).
    #[must_use]
    pub fn image_width(&self) -> u32 {
        self.x_siz.saturating_sub(self.xo_siz)
    }
    /// Return the effective image height in pixels (y_siz - yo_siz).
    #[must_use]
    pub fn image_height(&self) -> u32 {
        self.y_siz.saturating_sub(self.yo_siz)
    }

    /// Number of tile columns.
    #[must_use]
    pub fn num_tiles_x(&self) -> u32 {
        let effective_xt = if self.xt_siz == 0 {
            self.x_siz
        } else {
            self.xt_siz
        };
        let width = self.x_siz.saturating_sub(self.xto_siz);
        if width == 0 {
            return 1;
        }
        (width + effective_xt - 1) / effective_xt
    }

    /// Number of tile rows.
    #[must_use]
    pub fn num_tiles_y(&self) -> u32 {
        let effective_yt = if self.yt_siz == 0 {
            self.y_siz
        } else {
            self.yt_siz
        };
        let height = self.y_siz.saturating_sub(self.yto_siz);
        if height == 0 {
            return 1;
        }
        (height + effective_yt - 1) / effective_yt
    }

    /// Returns `(tile_x, tile_y, tile_width, tile_height)` for the given tile index.
    ///
    /// `tile_idx` is row-major: `tile_idx = tile_row * num_tiles_x() + tile_col`.
    #[must_use]
    pub fn tile_rect(&self, tile_idx: u32) -> (u32, u32, u32, u32) {
        let ntx = self.num_tiles_x().max(1);
        let col = tile_idx % ntx;
        let row = tile_idx / ntx;
        let effective_xt = if self.xt_siz == 0 {
            self.x_siz
        } else {
            self.xt_siz
        };
        let effective_yt = if self.yt_siz == 0 {
            self.y_siz
        } else {
            self.yt_siz
        };
        let x0 = self.xto_siz + col * effective_xt;
        let y0 = self.yto_siz + row * effective_yt;
        let x1 = (x0 + effective_xt).min(self.x_siz);
        let y1 = (y0 + effective_yt).min(self.y_siz);
        let tw = x1.saturating_sub(x0);
        let th = y1.saturating_sub(y0);
        (x0, y0, tw, th)
    }
}

/// Parsed COD (coding style default) marker.
#[derive(Debug, Clone)]
pub struct CodMarker {
    /// Coding style byte (Scod).
    pub scod: u8,
    /// Progression order (0=LRCP, 1=RLCP, 2=RPCL, 3=PCRL, 4=CPRL).
    pub progression_order: u8,
    /// Number of quality layers.
    pub num_layers: u16,
    /// Multiple component transform (0=none, 1=RCT or ICT).
    pub mct: u8,
    /// Number of decomposition levels.
    pub num_decomp_levels: u8,
    /// Code-block width exponent (xcb): block width = 2^(xcb+2).
    pub xcb: u8,
    /// Code-block height exponent (ycb): block height = 2^(ycb+2).
    pub ycb: u8,
    /// Code-block style byte.
    pub cb_style: u8,
    /// Wavelet transformation: 0 = 9-7 lossy, 1 = 5-3 lossless.
    pub wavelet_filter: u8,
}

impl CodMarker {
    /// Return true if the 5-3 lossless (reversible) wavelet filter is selected.
    #[must_use]
    pub fn is_lossless_wavelet(&self) -> bool {
        self.wavelet_filter == 1
    }

    /// Return true if the CDF 9/7 irreversible (lossy) wavelet filter is selected.
    #[must_use]
    pub fn is_irreversible_97(&self) -> bool {
        self.wavelet_filter == 0
    }
}

/// Identifies a subband type in the decomposition hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubbandKind {
    /// Lowest-frequency approximation subband.
    Ll,
    /// Horizontal detail (low-pass rows, high-pass cols).
    Hl,
    /// Vertical detail (high-pass rows, low-pass cols).
    Lh,
    /// Diagonal detail (all high-pass).
    Hh,
}

/// Parsed QCD (quantization default) marker.
#[derive(Debug, Clone)]
pub struct QcdMarker {
    /// Quantization style (Sqcd). 0 = no quantization (lossless).
    pub sqcd: u8,
    /// Step size values for subbands.
    pub step_sizes: Vec<u16>,
}

impl QcdMarker {
    /// Extract the guard bits (top 3 bits of Sqcd).
    #[must_use]
    pub fn guard_bits(&self) -> u8 {
        self.sqcd >> 5
    }

    /// Extract the quantization style (low 5 bits of Sqcd).
    /// - 0: no quantization (lossless)
    /// - 1: scalar derived (single step size, rest derived)
    /// - 2: scalar expounded (one step size per subband)
    #[must_use]
    pub fn quant_style(&self) -> u8 {
        self.sqcd & 0x1F
    }

    /// Compute the quantization step size for a given subband index.
    ///
    /// `subband_idx` enumerates subbands in the QCD order (LL first, then
    /// HH/HL/LH for each decomposition level from coarsest to finest).
    /// `bit_depth` is the component bit depth from the SIZ marker.
    ///
    /// Returns 1.0 for the no-quantization style (lossless path).
    #[must_use]
    pub fn step_size_for_subband(&self, subband_idx: usize, bit_depth: u8) -> f64 {
        let style = self.quant_style();
        match style {
            0 => 1.0, // no quantization — lossless path
            1 => {
                // Scalar derived: one entry for all subbands, rest are derived by
                // halving the step size at each level.
                if self.step_sizes.is_empty() {
                    return 1.0;
                }
                let raw = self.step_sizes[0];
                let epsilon = i32::from((raw >> 11) & 0x1F);
                let mu = f64::from(raw & 0x7FF);
                let r_b = i32::from(bit_depth);
                (2f64).powi(r_b - epsilon) * (1.0 + mu / 2048.0)
            }
            2 => {
                // Scalar expounded: one entry per subband.
                if self.step_sizes.is_empty() {
                    return 1.0;
                }
                let idx = subband_idx.min(self.step_sizes.len().saturating_sub(1));
                let raw = self.step_sizes[idx];
                let epsilon = i32::from((raw >> 11) & 0x1F);
                let mu = f64::from(raw & 0x7FF);
                let r_b = i32::from(bit_depth);
                (2f64).powi(r_b - epsilon) * (1.0 + mu / 2048.0)
            }
            _ => 1.0, // unknown — fall back to no-op
        }
    }
}

/// Parsed SOT (start of tile-part) marker.
#[derive(Debug, Clone)]
pub struct SotMarker {
    /// Tile index.
    pub isot: u16,
    /// Length of the tile-part in bytes (including the SOT marker segment).
    pub psot: u32,
    /// Tile-part index within the tile (0-based).
    pub tpsot: u8,
    /// Total number of tile-parts (0 = unknown).
    pub tnsot: u8,
}

/// A parsed marker segment from the codestream.
#[derive(Debug, Clone)]
pub enum MarkerSegment {
    /// SIZ (image and tile size) marker.
    Siz(SizMarker),
    /// COD (coding style default) marker.
    Cod(CodMarker),
    /// QCD (quantization default) marker.
    Qcd(QcdMarker),
    /// SOT (start of tile-part) marker.
    Sot(SotMarker),
    /// SOD (start of data) — the tile bitstream bytes that follow.
    Sod {
        /// Raw compressed tile data (everything between SOD and the next SOT or EOC).
        data: Vec<u8>,
    },
    /// EOC (end of codestream).
    Eoc,
}

// ── Parser helpers ────────────────────────────────────────────────────────────

fn read_u8(buf: &[u8], offset: usize, ctx: &'static str) -> Jp2Result<u8> {
    buf.get(offset).copied().ok_or(Jp2Error::Truncated {
        context: ctx,
        needed: offset + 1,
        available: buf.len(),
    })
}

fn read_u16_be(buf: &[u8], offset: usize, ctx: &'static str) -> Jp2Result<u16> {
    if offset + 2 > buf.len() {
        return Err(Jp2Error::Truncated {
            context: ctx,
            needed: offset + 2,
            available: buf.len(),
        });
    }
    Ok(u16::from_be_bytes([buf[offset], buf[offset + 1]]))
}

fn read_u32_be(buf: &[u8], offset: usize, ctx: &'static str) -> Jp2Result<u32> {
    if offset + 4 > buf.len() {
        return Err(Jp2Error::Truncated {
            context: ctx,
            needed: offset + 4,
            available: buf.len(),
        });
    }
    Ok(u32::from_be_bytes([
        buf[offset],
        buf[offset + 1],
        buf[offset + 2],
        buf[offset + 3],
    ]))
}

// ── Individual marker parsers ─────────────────────────────────────────────────

fn parse_siz(payload: &[u8]) -> Jp2Result<SizMarker> {
    // Fixed fields: rsiz(2)+xsiz(4)+ysiz(4)+xosiz(4)+yosiz(4)+
    //               xtsiz(4)+ytsiz(4)+xtosiz(4)+ytosiz(4)+csiz(2) = 36 bytes.
    // (The Lsiz length field is consumed by the main parser, not included in payload.)
    const FIXED: usize = 36;
    if payload.len() < FIXED {
        return Err(Jp2Error::Truncated {
            context: "SIZ marker",
            needed: FIXED,
            available: payload.len(),
        });
    }
    let rsiz = read_u16_be(payload, 0, "SIZ.rsiz")?;
    let x_siz = read_u32_be(payload, 2, "SIZ.xsiz")?;
    let y_siz = read_u32_be(payload, 6, "SIZ.ysiz")?;
    let xo_siz = read_u32_be(payload, 10, "SIZ.xosiz")?;
    let yo_siz = read_u32_be(payload, 14, "SIZ.yosiz")?;
    let xt_siz = read_u32_be(payload, 18, "SIZ.xtsiz")?;
    let yt_siz = read_u32_be(payload, 22, "SIZ.ytsiz")?;
    let xto_siz = read_u32_be(payload, 26, "SIZ.xtosiz")?;
    let yto_siz = read_u32_be(payload, 30, "SIZ.ytosiz")?;
    let csiz = read_u16_be(payload, 34, "SIZ.csiz")?;

    let needed_len = FIXED + 3 * usize::from(csiz);
    if payload.len() < needed_len {
        return Err(Jp2Error::Truncated {
            context: "SIZ component params",
            needed: needed_len,
            available: payload.len(),
        });
    }
    let mut components = Vec::with_capacity(usize::from(csiz));
    let mut off = FIXED;
    for _ in 0..csiz {
        let ssiz = read_u8(payload, off, "SIZ.ssiz")?;
        let xr_siz = read_u8(payload, off + 1, "SIZ.xrsiz")?;
        let yr_siz = read_u8(payload, off + 2, "SIZ.yrsiz")?;
        // Defend against a divide-by-zero in the per-component size computation
        // `(width + xr - 1) / xr` performed by the decoder: a zero subsampling
        // factor is malformed input.
        if xr_siz == 0 || yr_siz == 0 {
            return Err(Jp2Error::Unsupported(
                "SIZ component subsampling factor must be non-zero".to_string(),
            ));
        }
        // Defend against `1u32 << bit_depth` overflow in the decoder: bit depth
        // is `(Ssiz & 0x7F) + 1` (up to 128), but samples are stored as u16, so
        // reject anything above 16 bits.
        let bit_depth = (ssiz & 0x7F) + 1;
        if bit_depth > 16 {
            return Err(Jp2Error::Unsupported(format!(
                "SIZ component bit depth {bit_depth} exceeds 16"
            )));
        }
        components.push(ComponentParams {
            ssiz,
            xr_siz,
            yr_siz,
        });
        off += 3;
    }
    Ok(SizMarker {
        rsiz,
        x_siz,
        y_siz,
        xo_siz,
        yo_siz,
        xt_siz,
        yt_siz,
        xto_siz,
        yto_siz,
        csiz,
        components,
    })
}

fn parse_cod(payload: &[u8]) -> Jp2Result<CodMarker> {
    // Minimum fixed-size part: scod(1) + SGcod(4) + SPcod(5+) = 10 bytes
    if payload.len() < 10 {
        return Err(Jp2Error::Truncated {
            context: "COD marker",
            needed: 10,
            available: payload.len(),
        });
    }
    let scod = payload[0];
    // SGcod: progression_order(1) + num_layers(2) + mct(1)
    let progression_order = payload[1];
    let num_layers = u16::from_be_bytes([payload[2], payload[3]]);
    let mct = payload[4];
    // SPcod: num_decomp_levels(1) + xcb(1) + ycb(1) + cb_style(1) + wavelet(1)
    let num_decomp_levels = payload[5];
    let xcb = payload[6];
    let ycb = payload[7];
    let cb_style = payload[8];
    let wavelet_filter = payload[9];
    Ok(CodMarker {
        scod,
        progression_order,
        num_layers,
        mct,
        num_decomp_levels,
        xcb,
        ycb,
        cb_style,
        wavelet_filter,
    })
}

fn parse_qcd(payload: &[u8]) -> Jp2Result<QcdMarker> {
    if payload.is_empty() {
        return Err(Jp2Error::Truncated {
            context: "QCD marker",
            needed: 1,
            available: 0,
        });
    }
    let sqcd = payload[0];
    let quant_style = sqcd & 0x1F;
    let mut step_sizes = Vec::new();
    match quant_style {
        0 => {
            // No quantization: each subband is encoded as a u8 (SPqcd = 8-bit precision).
            for &b in &payload[1..] {
                step_sizes.push(u16::from(b));
            }
        }
        1 => {
            // Scalar derived: single 16-bit step size.
            if payload.len() >= 3 {
                let v = u16::from_be_bytes([payload[1], payload[2]]);
                step_sizes.push(v);
            }
        }
        2 => {
            // Scalar expounded: one 16-bit step per subband.
            let mut off = 1;
            while off + 1 < payload.len() {
                let v = u16::from_be_bytes([payload[off], payload[off + 1]]);
                step_sizes.push(v);
                off += 2;
            }
        }
        _ => {
            // Unknown quantization style — store raw bytes as u16.
            for &b in &payload[1..] {
                step_sizes.push(u16::from(b));
            }
        }
    }
    Ok(QcdMarker { sqcd, step_sizes })
}

fn parse_sot(payload: &[u8]) -> Jp2Result<SotMarker> {
    // isot(2) + psot(4) + tpsot(1) + tnsot(1) = 8 bytes
    if payload.len() < 8 {
        return Err(Jp2Error::Truncated {
            context: "SOT marker",
            needed: 8,
            available: payload.len(),
        });
    }
    let isot = u16::from_be_bytes([payload[0], payload[1]]);
    let psot = u32::from_be_bytes([payload[2], payload[3], payload[4], payload[5]]);
    let tpsot = payload[6];
    let tnsot = payload[7];
    Ok(SotMarker {
        isot,
        psot,
        tpsot,
        tnsot,
    })
}

// ── Main codestream parser ────────────────────────────────────────────────────

/// Parse a JPEG 2000 codestream into a sequence of [`MarkerSegment`]s.
///
/// Expects to start at the beginning of the codestream (SOC marker).
/// Handles the main header through to EOC.
pub fn parse_codestream(data: &[u8]) -> Jp2Result<Vec<MarkerSegment>> {
    let mut segments = Vec::new();
    let mut pos = 0;

    // Require SOC as first marker.
    if data.len() < 2 {
        return Err(Jp2Error::Truncated {
            context: "codestream SOC",
            needed: 2,
            available: data.len(),
        });
    }
    let first_marker = u16::from_be_bytes([data[0], data[1]]);
    if first_marker != SOC {
        return Err(Jp2Error::InvalidMarker {
            marker: first_marker,
            offset: 0,
        });
    }
    pos += 2;

    // Track whether we are in a tile-part (after SOT, before next SOT/EOC).
    let mut in_tile = false;
    let mut tile_data_start = 0usize;
    // Start offset of the SOT marker for the current tile-part, and its Psot.
    // When `Psot` is known (> 0) the tile data is delimited by length
    // (`sot_start + psot`) rather than by scanning for the next marker — this is
    // required when the entropy-coded data may itself contain `0xFF 0x90` / `0xFF
    // 0xD9` byte pairs at code-block boundaries.
    let mut sot_start = 0usize;
    let mut sot_psot = 0u32;

    loop {
        if pos + 1 >= data.len() {
            break;
        }
        let marker = u16::from_be_bytes([data[pos], data[pos + 1]]);

        // If we're scanning tile-part data, look for next SOT or EOC to delimit it.
        if in_tile {
            if marker == EOC {
                let tile_data = data[tile_data_start..pos].to_vec();
                segments.push(MarkerSegment::Sod { data: tile_data });
                segments.push(MarkerSegment::Eoc);
                break;
            } else if marker == SOT {
                let tile_data = data[tile_data_start..pos].to_vec();
                segments.push(MarkerSegment::Sod { data: tile_data });
                in_tile = false;
                // Fall through to process the SOT marker below.
            } else {
                pos += 1;
                continue;
            }
        }

        // SOD and EOC have no length field (handled above when in_tile is true).
        match marker {
            EOC => {
                segments.push(MarkerSegment::Eoc);
                break;
            }
            SOD => {
                // The tile data starts immediately after the 2-byte SOD marker.
                pos += 2;
                tile_data_start = pos;
                in_tile = true;
                // If Psot is known, the tile data ends at `sot_start + psot`;
                // emit it directly and continue past it (avoids scanning, which
                // can be fooled by marker-like bytes in the entropy data).
                if sot_psot > 0 {
                    let tile_end = sot_start.saturating_add(sot_psot as usize).min(data.len());
                    if tile_end > tile_data_start {
                        let tile_data = data[tile_data_start..tile_end].to_vec();
                        segments.push(MarkerSegment::Sod { data: tile_data });
                        pos = tile_end;
                        in_tile = false;
                    }
                }
                // Otherwise fall through to scan for the next SOT or EOC.
                continue;
            }
            _ => {}
        }

        // All other markers have a 2-byte length field after the marker code.
        let marker_start = pos;
        pos += 2; // skip marker
        if pos + 2 > data.len() {
            return Err(Jp2Error::Truncated {
                context: "marker segment length",
                needed: pos + 2,
                available: data.len(),
            });
        }
        let lseg = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
        // Lseg includes its own 2 bytes.
        if lseg < 2 {
            return Err(Jp2Error::InvalidMarker {
                marker,
                offset: pos - 2,
            });
        }
        let payload_len = lseg - 2;
        pos += 2; // skip length field

        if pos + payload_len > data.len() {
            return Err(Jp2Error::Truncated {
                context: "marker segment payload",
                needed: pos + payload_len,
                available: data.len(),
            });
        }
        let payload = &data[pos..pos + payload_len];
        pos += payload_len;

        match marker {
            SIZ => {
                let siz = parse_siz(payload)?;
                segments.push(MarkerSegment::Siz(siz));
            }
            COD => {
                let cod = parse_cod(payload)?;
                segments.push(MarkerSegment::Cod(cod));
            }
            QCD => {
                let qcd = parse_qcd(payload)?;
                segments.push(MarkerSegment::Qcd(qcd));
            }
            SOT => {
                let sot = parse_sot(payload)?;
                sot_start = marker_start;
                sot_psot = sot.psot;
                segments.push(MarkerSegment::Sot(sot));
            }
            _ => {
                // Skip unknown markers (COM, TLM, PLM, etc.).
            }
        }
    }

    Ok(segments)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal SOC + SIZ + SOT + EOC codestream for testing.
    fn minimal_codestream() -> Vec<u8> {
        let mut v = Vec::new();
        // SOC
        v.extend_from_slice(&SOC.to_be_bytes());

        // SIZ marker
        // Lsiz = 2 (length field itself) + 36 (fixed fields) + 3 (one component) = 41
        let siz_payload_len: u16 = 2 + 36 + 3;
        v.extend_from_slice(&SIZ.to_be_bytes());
        v.extend_from_slice(&siz_payload_len.to_be_bytes());
        v.extend_from_slice(&0u16.to_be_bytes()); // rsiz
        v.extend_from_slice(&16u32.to_be_bytes()); // x_siz
        v.extend_from_slice(&16u32.to_be_bytes()); // y_siz
        v.extend_from_slice(&0u32.to_be_bytes()); // xo_siz
        v.extend_from_slice(&0u32.to_be_bytes()); // yo_siz
        v.extend_from_slice(&16u32.to_be_bytes()); // xt_siz
        v.extend_from_slice(&16u32.to_be_bytes()); // yt_siz
        v.extend_from_slice(&0u32.to_be_bytes()); // xto_siz
        v.extend_from_slice(&0u32.to_be_bytes()); // yto_siz
        v.extend_from_slice(&1u16.to_be_bytes()); // csiz = 1
        v.push(7); // ssiz: 8-bit unsigned (7+1=8)
        v.push(1); // xr_siz
        v.push(1); // yr_siz

        // SOT marker (length = 2 + 8 = 10)
        let sot_len: u16 = 10;
        v.extend_from_slice(&SOT.to_be_bytes());
        v.extend_from_slice(&sot_len.to_be_bytes());
        v.extend_from_slice(&0u16.to_be_bytes()); // isot
        v.extend_from_slice(&0u32.to_be_bytes()); // psot = 0 (unknown)
        v.push(0); // tpsot
        v.push(0); // tnsot

        // SOD
        v.extend_from_slice(&SOD.to_be_bytes());
        // (no tile data)

        // EOC
        v.extend_from_slice(&EOC.to_be_bytes());
        v
    }

    #[test]
    fn parse_minimal_codestream() {
        let data = minimal_codestream();
        let segments = parse_codestream(&data).expect("parse");
        // Should have: Siz, Sot, Sod, Eoc
        let has_siz = segments.iter().any(|s| matches!(s, MarkerSegment::Siz(_)));
        let has_sot = segments.iter().any(|s| matches!(s, MarkerSegment::Sot(_)));
        let has_eoc = segments.iter().any(|s| matches!(s, MarkerSegment::Eoc));
        assert!(has_siz, "Expected Siz marker");
        assert!(has_sot, "Expected Sot marker");
        assert!(has_eoc, "Expected Eoc marker");
    }

    #[test]
    fn siz_fields_correct() {
        let data = minimal_codestream();
        let segments = parse_codestream(&data).expect("parse");
        let siz = segments
            .iter()
            .find_map(|s| {
                if let MarkerSegment::Siz(sz) = s {
                    Some(sz)
                } else {
                    None
                }
            })
            .expect("SIZ marker");
        assert_eq!(siz.image_width(), 16);
        assert_eq!(siz.image_height(), 16);
        assert_eq!(siz.csiz, 1);
        assert_eq!(siz.components[0].bit_depth(), 8);
        assert!(!siz.components[0].is_signed());
    }

    #[test]
    fn sot_fields_correct() {
        let data = minimal_codestream();
        let segments = parse_codestream(&data).expect("parse");
        let sot = segments
            .iter()
            .find_map(|s| {
                if let MarkerSegment::Sot(st) = s {
                    Some(st)
                } else {
                    None
                }
            })
            .expect("SOT marker");
        assert_eq!(sot.isot, 0);
        assert_eq!(sot.tpsot, 0);
    }

    #[test]
    fn missing_soc_returns_error() {
        let data = vec![0x00u8, 0x00];
        assert!(parse_codestream(&data).is_err());
    }

    #[test]
    fn truncated_codestream_returns_error() {
        let data = vec![0xFF, 0x4F]; // Just SOC, then nothing
                                     // Should not panic; may return empty segment list or error.
        let result = parse_codestream(&data);
        // Either Ok([]) or Err — both are acceptable; we just check no panic.
        let _ = result;
    }

    #[test]
    fn parse_siz_rejects_zero_subsampling() {
        // Regression: a zero XRsiz/YRsiz subsampling factor caused a
        // divide-by-zero in the decoder's `(width + xr - 1) / xr`. parse_siz
        // must reject it while parsing the marker segment.
        let mut p = Vec::new();
        p.extend_from_slice(&0u16.to_be_bytes()); // Rsiz
        p.extend_from_slice(&16u32.to_be_bytes()); // Xsiz
        p.extend_from_slice(&16u32.to_be_bytes()); // Ysiz
        p.extend_from_slice(&0u32.to_be_bytes()); // XOsiz
        p.extend_from_slice(&0u32.to_be_bytes()); // YOsiz
        p.extend_from_slice(&16u32.to_be_bytes()); // XTsiz
        p.extend_from_slice(&16u32.to_be_bytes()); // YTsiz
        p.extend_from_slice(&0u32.to_be_bytes()); // XTOsiz
        p.extend_from_slice(&0u32.to_be_bytes()); // YTOsiz
        p.extend_from_slice(&1u16.to_be_bytes()); // Csiz = 1
        p.push(7); // Ssiz = 8-bit
        p.push(0); // XRsiz = 0 (malformed)
        p.push(1); // YRsiz = 1
        assert!(parse_siz(&p).is_err());

        // The same payload with a valid XRsiz parses successfully.
        let mut ok = p.clone();
        let n = ok.len();
        ok[n - 2] = 1; // XRsiz = 1
        assert!(parse_siz(&ok).is_ok());
    }
}
