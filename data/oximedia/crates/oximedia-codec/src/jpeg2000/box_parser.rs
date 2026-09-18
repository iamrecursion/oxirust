//! JP2 (JPEG 2000) ISOBMFF container box parser.
//!
//! Parses the box hierarchy of a `.jp2` file and extracts the J2K codestream.
//!
//! ## JP2 file structure
//!
//! ```text
//! [jP  ] Signature box (12 bytes, magic 0x0D0A870A)
//! [ftyp] File type box
//! [jp2h] JP2 Header box (superbox)
//!   [ihdr] Image header
//!   [colr] Colour specification
//! [jp2c] Contiguous Codestream box — contains the raw J2K codestream
//! ```
//!
//! All JP2 box lengths and integers are big-endian.

use super::{Jp2Error, Jp2Result};

/// JP2 colour space identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Jp2ColorSpace {
    /// sRGB (enumCS = 16).
    Srgb,
    /// Greyscale (enumCS = 17).
    Greyscale,
    /// YCC (enumCS = 18).
    Ycc,
    /// Other/unknown colour space.
    Other(u32),
}

impl Jp2ColorSpace {
    fn from_enum_cs(v: u32) -> Self {
        match v {
            16 => Self::Srgb,
            17 => Self::Greyscale,
            18 => Self::Ycc,
            other => Self::Other(other),
        }
    }
}

/// Parsed JP2 main header fields (extracted from the `jp2h` superbox).
#[derive(Debug, Clone)]
pub struct Jp2Header {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Number of image components.
    pub num_components: u16,
    /// Effective bit depth (1..=16).
    pub bit_depth: u8,
    /// True if samples are signed.
    pub is_signed: bool,
    /// Colour space.
    pub color_space: Jp2ColorSpace,
}

// ── Box layout constants ──────────────────────────────────────────────────────

const BOX_SIGNATURE: u32 = 0x6A50_2020; // 'jP  '
const BOX_FTYP: u32 = 0x6674_7970; // 'ftyp'
const BOX_JP2H: u32 = 0x6A70_3268; // 'jp2h'
const BOX_IHDR: u32 = 0x6968_6472; // 'ihdr'
const BOX_COLR: u32 = 0x636F_6C72; // 'colr'
const BOX_JP2C: u32 = 0x6A70_3263; // 'jp2c'

const JP2_MAGIC: [u8; 4] = [0x0D, 0x0A, 0x87, 0x0A];

// ── Box iterator ─────────────────────────────────────────────────────────────

/// A single ISOBMFF box.
struct Box<'a> {
    box_type: u32,
    payload: &'a [u8],
}

/// Iterate over top-level boxes in `data`.
fn next_box(data: &[u8], offset: usize) -> Jp2Result<Option<(Box<'_>, usize)>> {
    if offset >= data.len() {
        return Ok(None);
    }
    let remaining = &data[offset..];
    if remaining.len() < 8 {
        return Err(Jp2Error::Truncated {
            context: "JP2 box header",
            needed: offset + 8,
            available: data.len(),
        });
    }
    let box_len =
        u32::from_be_bytes([remaining[0], remaining[1], remaining[2], remaining[3]]) as usize;
    let box_type = u32::from_be_bytes([remaining[4], remaining[5], remaining[6], remaining[7]]);

    // box_len = 0 means "extends to end of file".
    // box_len = 1 means 64-bit extended length (we don't support files > 4 GiB).
    let (total_box_len, payload_start) = if box_len == 1 {
        // 64-bit length: next 8 bytes.
        if remaining.len() < 16 {
            return Err(Jp2Error::Truncated {
                context: "JP2 box extended length",
                needed: offset + 16,
                available: data.len(),
            });
        }
        let hi =
            u32::from_be_bytes([remaining[8], remaining[9], remaining[10], remaining[11]]) as usize;
        let lo = u32::from_be_bytes([remaining[12], remaining[13], remaining[14], remaining[15]])
            as usize;
        if hi != 0 {
            return Err(Jp2Error::Unsupported(
                "JP2 box larger than 4 GiB".to_string(),
            ));
        }
        (lo, 16)
    } else if box_len == 0 {
        // Extends to end of file.
        (remaining.len(), 8)
    } else {
        (box_len, 8)
    };

    if total_box_len < payload_start {
        return Err(Jp2Error::Truncated {
            context: "JP2 box total length",
            needed: offset + total_box_len,
            available: data.len(),
        });
    }
    if offset + total_box_len > data.len() {
        return Err(Jp2Error::Truncated {
            context: "JP2 box payload",
            needed: offset + total_box_len,
            available: data.len(),
        });
    }

    let payload = &remaining[payload_start..total_box_len];
    let next_offset = offset + total_box_len;
    Ok(Some((Box { box_type, payload }, next_offset)))
}

// ── Individual box parsers ────────────────────────────────────────────────────

fn parse_ihdr(payload: &[u8]) -> Jp2Result<(u32, u32, u16, u8, bool)> {
    // height(4) + width(4) + nc(2) + bpc(1) + c(1) + unkc(1) + ipr(1) = 14 bytes
    if payload.len() < 14 {
        return Err(Jp2Error::Truncated {
            context: "ihdr box",
            needed: 14,
            available: payload.len(),
        });
    }
    let height = u32::from_be_bytes([payload[0], payload[1], payload[2], payload[3]]);
    let width = u32::from_be_bytes([payload[4], payload[5], payload[6], payload[7]]);
    let nc = u16::from_be_bytes([payload[8], payload[9]]);
    let bpc = payload[10]; // bits-per-component - 1, bit7 = sign
    let _compression_type = payload[11]; // must be 7
                                         // unkc = payload[12], ipr = payload[13]
    let is_signed = (bpc & 0x80) != 0;
    let bit_depth = (bpc & 0x7F) + 1;
    Ok((height, width, nc, bit_depth, is_signed))
}

fn parse_colr(payload: &[u8]) -> Jp2Result<Jp2ColorSpace> {
    if payload.is_empty() {
        return Err(Jp2Error::Truncated {
            context: "colr box",
            needed: 1,
            available: 0,
        });
    }
    let meth = payload[0];
    if meth == 1 {
        // Enumerated colourspace.
        if payload.len() < 7 {
            return Err(Jp2Error::Truncated {
                context: "colr enumCS",
                needed: 7,
                available: payload.len(),
            });
        }
        // prec(1) approx(1) enumCS(4)
        let enum_cs = u32::from_be_bytes([payload[3], payload[4], payload[5], payload[6]]);
        Ok(Jp2ColorSpace::from_enum_cs(enum_cs))
    } else {
        // ICC profile or other — return Other.
        Ok(Jp2ColorSpace::Other(0))
    }
}

fn parse_jp2h(payload: &[u8]) -> Jp2Result<(Jp2Header, Option<Jp2ColorSpace>)> {
    let mut height = 0u32;
    let mut width = 0u32;
    let mut num_components = 0u16;
    let mut bit_depth = 8u8;
    let mut is_signed = false;
    let mut color_space = None;
    let mut found_ihdr = false;

    let mut off = 0;
    loop {
        match next_box(payload, off)? {
            None => break,
            Some((b, next)) => {
                match b.box_type {
                    BOX_IHDR => {
                        let (h, w, nc, bd, sgn) = parse_ihdr(b.payload)?;
                        height = h;
                        width = w;
                        num_components = nc;
                        bit_depth = bd;
                        is_signed = sgn;
                        found_ihdr = true;
                    }
                    BOX_COLR => {
                        color_space = Some(parse_colr(b.payload)?);
                    }
                    _ => {} // skip unknown sub-boxes
                }
                off = next;
            }
        }
    }

    if !found_ihdr {
        return Err(Jp2Error::Unsupported(
            "JP2 file missing required ihdr box in jp2h".to_string(),
        ));
    }

    let cs = color_space.unwrap_or(if num_components == 1 {
        Jp2ColorSpace::Greyscale
    } else {
        Jp2ColorSpace::Srgb
    });

    Ok((
        Jp2Header {
            width,
            height,
            num_components,
            bit_depth,
            is_signed,
            color_space: cs,
        },
        None,
    ))
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Parse a JP2 container file.
///
/// Validates the signature box, parses the `jp2h` superbox for header fields,
/// and returns the codestream bytes from the `jp2c` box.
///
/// # Returns
///
/// `(header, codestream_bytes)` where `codestream_bytes` is a slice into the
/// original `data` buffer.
pub fn parse_jp2(data: &[u8]) -> Jp2Result<(Jp2Header, &[u8])> {
    let mut off = 0;

    // 1. Validate signature box (must be first).
    let (sig_box, next) = next_box(data, off)?.ok_or(Jp2Error::InvalidSignature)?;
    off = next;
    if sig_box.box_type != BOX_SIGNATURE {
        return Err(Jp2Error::InvalidSignature);
    }
    // Signature box payload must start with the magic bytes.
    if sig_box.payload.len() < 4 || &sig_box.payload[0..4] != JP2_MAGIC {
        return Err(Jp2Error::InvalidSignature);
    }

    let mut header: Option<Jp2Header> = None;
    let mut codestream: Option<&[u8]> = None;

    loop {
        match next_box(data, off)? {
            None => break,
            Some((b, next)) => {
                match b.box_type {
                    BOX_FTYP => {} // skip
                    BOX_JP2H => {
                        let (hdr, _) = parse_jp2h(b.payload)?;
                        header = Some(hdr);
                    }
                    BOX_JP2C => {
                        codestream = Some(b.payload);
                    }
                    _ => {} // skip unknown top-level boxes
                }
                off = next;
                // Early exit once we have both.
                if header.is_some() && codestream.is_some() {
                    break;
                }
            }
        }
    }

    let hdr =
        header.ok_or_else(|| Jp2Error::Unsupported("JP2 file missing jp2h box".to_string()))?;
    let cs =
        codestream.ok_or_else(|| Jp2Error::Unsupported("JP2 file missing jp2c box".to_string()))?;
    Ok((hdr, cs))
}

/// Detect whether `data` looks like a JP2 container (vs a raw J2K codestream).
///
/// Returns `true` if the first box is the JP2 signature box with the correct
/// four-character code `'jP  '` (0x6A502020).
#[must_use]
pub fn is_jp2_container(data: &[u8]) -> bool {
    if data.len() < 12 {
        return false;
    }
    let box_type = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
    box_type == BOX_SIGNATURE
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid JP2 file in memory.
    fn make_minimal_jp2(width: u32, height: u32, num_comp: u16, codestream: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();

        // --- Signature box ---
        // Length = 4 (len) + 4 (type) + 4 (magic) = 12
        out.extend_from_slice(&12u32.to_be_bytes()); // LBox
        out.extend_from_slice(&BOX_SIGNATURE.to_be_bytes()); // TBox
        out.extend_from_slice(&JP2_MAGIC); // DBox

        // --- ftyp box ---
        // Length = 4+4+4+4 = 16: LBox + TBox + brand + minV
        out.extend_from_slice(&20u32.to_be_bytes());
        out.extend_from_slice(&BOX_FTYP.to_be_bytes());
        out.extend_from_slice(b"jp2 "); // brand
        out.extend_from_slice(&0u32.to_be_bytes()); // minor version
        out.extend_from_slice(b"jp2 "); // compat

        // --- jp2h superbox ---
        // ihdr: 4+4+14 = 22 bytes total
        // colr: 4+4+7 = 15 bytes total
        // jp2h total = 8 + 22 + 15 = 45
        let mut jp2h_payload = Vec::new();

        // ihdr box payload (14 bytes)
        let ihdr_len: u32 = 8 + 14;
        jp2h_payload.extend_from_slice(&ihdr_len.to_be_bytes());
        jp2h_payload.extend_from_slice(&BOX_IHDR.to_be_bytes());
        jp2h_payload.extend_from_slice(&height.to_be_bytes());
        jp2h_payload.extend_from_slice(&width.to_be_bytes());
        jp2h_payload.extend_from_slice(&num_comp.to_be_bytes());
        jp2h_payload.push(7); // bpc = 8-bit unsigned
        jp2h_payload.push(7); // compression type
        jp2h_payload.push(0); // unkc
        jp2h_payload.push(0); // ipr

        // colr box payload (7 bytes)
        let colr_len: u32 = 8 + 7;
        jp2h_payload.extend_from_slice(&colr_len.to_be_bytes());
        jp2h_payload.extend_from_slice(&BOX_COLR.to_be_bytes());
        jp2h_payload.push(1); // meth = enumerated colourspace
        jp2h_payload.push(0); // prec
        jp2h_payload.push(0); // approx
        jp2h_payload.extend_from_slice(&17u32.to_be_bytes()); // enumCS = 17 (greyscale)

        let jp2h_total_len: u32 = 8 + jp2h_payload.len() as u32;
        out.extend_from_slice(&jp2h_total_len.to_be_bytes());
        out.extend_from_slice(&BOX_JP2H.to_be_bytes());
        out.extend_from_slice(&jp2h_payload);

        // --- jp2c box ---
        let jp2c_total_len: u32 = 8 + codestream.len() as u32;
        out.extend_from_slice(&jp2c_total_len.to_be_bytes());
        out.extend_from_slice(&BOX_JP2C.to_be_bytes());
        out.extend_from_slice(codestream);

        out
    }

    #[test]
    fn parse_minimal_jp2_header() {
        let dummy_codestream = [0xFF, 0x4F, 0xFF, 0xD9]; // SOC + EOC
        let jp2 = make_minimal_jp2(32, 64, 1, &dummy_codestream);
        let (hdr, cs) = parse_jp2(&jp2).expect("parse_jp2");
        assert_eq!(hdr.width, 32);
        assert_eq!(hdr.height, 64);
        assert_eq!(hdr.num_components, 1);
        assert_eq!(hdr.bit_depth, 8);
        assert!(!hdr.is_signed);
        assert_eq!(hdr.color_space, Jp2ColorSpace::Greyscale);
        assert_eq!(cs, &dummy_codestream[..]);
    }

    #[test]
    fn is_jp2_container_detects_signature() {
        let dummy = make_minimal_jp2(1, 1, 1, &[0xFF, 0x4F, 0xFF, 0xD9]);
        assert!(is_jp2_container(&dummy));
    }

    #[test]
    fn is_jp2_container_rejects_j2k() {
        // Raw J2K starts with 0xFF4F (SOC)
        let j2k = [
            0xFF, 0x4F, 0xFF, 0x51, 0x00, 0x00, 0xFF, 0xD9, 0x00, 0x00, 0x00, 0x00,
        ];
        assert!(!is_jp2_container(&j2k));
    }

    #[test]
    fn invalid_signature_returns_error() {
        let bad = vec![0x00u8; 32];
        assert!(parse_jp2(&bad).is_err());
    }

    #[test]
    fn jp2_color_space_from_enum() {
        assert_eq!(Jp2ColorSpace::from_enum_cs(16), Jp2ColorSpace::Srgb);
        assert_eq!(Jp2ColorSpace::from_enum_cs(17), Jp2ColorSpace::Greyscale);
        assert_eq!(Jp2ColorSpace::from_enum_cs(18), Jp2ColorSpace::Ycc);
        assert!(matches!(
            Jp2ColorSpace::from_enum_cs(99),
            Jp2ColorSpace::Other(99)
        ));
    }
}
