//! ICC profile metadata reader for v2/v4/v5 binary profiles.
//!
//! Parses the ICC profile header (128 bytes) and tag table as defined by
//! ICC.1:2022 (v4) and the iccMAX (v5) specification. Provides two entry
//! points:
//!
//! - [`read_icc_metadata`] — full header + tag table extraction
//! - [`read_icc_description`] — extracts the human-readable `desc` tag string
//!
//! # Binary Layout (ICC header)
//!
//! | Bytes  | Field                         |
//! |--------|-------------------------------|
//! | 0–3    | Profile size (u32 BE)         |
//! | 4–7    | Preferred CMM                 |
//! | 8–11   | Profile version               |
//! | 12–15  | Device class                  |
//! | 16–19  | Data colour space             |
//! | 20–23  | Profile Connection Space (PCS)|
//! | 24–35  | Date/time created             |
//! | 36–39  | Signature `'acsp'`            |
//! | 40–43  | Primary platform              |
//! | 44–47  | Profile flags                 |
//! | 48–51  | Device manufacturer           |
//! | 52–55  | Device model                  |
//! | 56–63  | Device attributes             |
//! | 64–67  | Rendering intent              |
//! | 68–79  | Profile illuminant (XYZ)      |
//! | 80–83  | Profile creator               |
//! | 84–99  | Profile ID (MD5)              |
//! | 128+   | Tag count + tag table         |

#![allow(clippy::cast_precision_loss)]

use crate::error::{ColorError, Result};
use crate::icc::RenderingIntent;

// ── Enumerations ──────────────────────────────────────────────────────────────

/// ICC device class (profile class) as per ICC spec section 7.2.5.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IccDeviceClass {
    /// Input device (camera, scanner).  Tag: `scnr`
    Input,
    /// Display device (monitor).  Tag: `mntr`
    Display,
    /// Output device (printer).  Tag: `prtr`
    Output,
    /// Color space conversion profile.  Tag: `spac`
    ColorSpace,
    /// Abstract profile.  Tag: `abst`
    Abstract,
    /// Named colour profile.  Tag: `nmcl`
    Named,
    /// Device link profile.  Tag: `link`
    DeviceLink,
    /// Unknown device class.
    Unknown([u8; 4]),
}

impl IccDeviceClass {
    /// Parse from a 4-byte ICC device-class field.
    #[must_use]
    pub fn from_bytes(b: [u8; 4]) -> Self {
        match &b {
            b"scnr" => Self::Input,
            b"mntr" => Self::Display,
            b"prtr" => Self::Output,
            b"spac" => Self::ColorSpace,
            b"abst" => Self::Abstract,
            b"nmcl" => Self::Named,
            b"link" => Self::DeviceLink,
            _ => Self::Unknown(b),
        }
    }
}

/// ICC colour space signature as per ICC spec section 7.2.6.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IccColorSpace {
    /// CIE XYZ.
    XYZ,
    /// CIE L*a*b* (D50).
    Lab,
    /// CIE Luv.
    Luv,
    /// YCbCr.
    YCbCr,
    /// CIE Yxy.
    Yxy,
    /// RGB.
    RGB,
    /// Greyscale.
    Gray,
    /// HSV.
    HSV,
    /// HLS.
    HLS,
    /// CMYK.
    CMYK,
    /// CMY.
    CMY,
    /// 2-channel device colour space (nCLR where n=2).
    CLR2,
    /// 3-channel device colour space.
    CLR3,
    /// 4-channel device colour space.
    CLR4,
    /// 5-channel device colour space.
    CLR5,
    /// 6-channel device colour space.
    CLR6,
    /// 7-channel device colour space.
    CLR7,
    /// 8-channel device colour space.
    CLR8,
    /// 9-channel device colour space.
    CLR9,
    /// 10-channel device colour space.
    CLR10,
    /// 11-channel device colour space.
    CLR11,
    /// 12-channel device colour space.
    CLR12,
    /// 13-channel device colour space.
    CLR13,
    /// 14-channel device colour space.
    CLR14,
    /// 15-channel device colour space.
    CLR15,
    /// Unknown or proprietary colour space.
    Unknown([u8; 4]),
}

impl IccColorSpace {
    /// Parse from a 4-byte ICC colour-space field.
    #[must_use]
    pub fn from_bytes(b: [u8; 4]) -> Self {
        match &b {
            b"XYZ " => Self::XYZ,
            b"Lab " => Self::Lab,
            b"Luv " => Self::Luv,
            b"YCbr" => Self::YCbCr,
            b"Yxy " => Self::Yxy,
            b"RGB " => Self::RGB,
            b"GRAY" => Self::Gray,
            b"HSV " => Self::HSV,
            b"HLS " => Self::HLS,
            b"CMYK" => Self::CMYK,
            b"CMY " => Self::CMY,
            b"2CLR" => Self::CLR2,
            b"3CLR" => Self::CLR3,
            b"4CLR" => Self::CLR4,
            b"5CLR" => Self::CLR5,
            b"6CLR" => Self::CLR6,
            b"7CLR" => Self::CLR7,
            b"8CLR" => Self::CLR8,
            b"9CLR" => Self::CLR9,
            b"ACLR" => Self::CLR10,
            b"BCLR" => Self::CLR11,
            b"CCLR" => Self::CLR12,
            b"DCLR" => Self::CLR13,
            b"ECLR" => Self::CLR14,
            b"FCLR" => Self::CLR15,
            _ => Self::Unknown(b),
        }
    }
}

// ── Version ───────────────────────────────────────────────────────────────────

/// ICC profile version number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IccVersion {
    /// Major version (2, 4, or 5).
    pub major: u8,
    /// Minor version (BCD upper nibble of byte 9).
    pub minor: u8,
    /// Sub-minor version (BCD lower nibble of byte 9).
    pub sub: u8,
}

impl IccVersion {
    /// Parse from the 4-byte version field (bytes 8–11 of the ICC header).
    ///
    /// Per spec: byte 8 = major, byte 9 = minor(hi nibble) + sub(lo nibble),
    /// bytes 10–11 are reserved zeros.
    #[must_use]
    pub fn from_bytes(b: [u8; 4]) -> Self {
        Self {
            major: b[0],
            minor: b[1] >> 4,
            sub: b[1] & 0x0F,
        }
    }
}

impl std::fmt::Display for IccVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.sub)
    }
}

// ── Tag table entry ───────────────────────────────────────────────────────────

/// A single entry in the ICC tag table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IccTag {
    /// 4-byte tag signature (e.g. `b"desc"`, `b"cprt"`).
    pub sig: [u8; 4],
    /// Byte offset from the start of the profile to the tag data.
    pub offset: u32,
    /// Size of the tag data in bytes.
    pub size: u32,
}

// ── Profile metadata ──────────────────────────────────────────────────────────

/// Full metadata extracted from an ICC profile header and tag table.
///
/// Does not apply any colour transforms — it is a pure structural read.
#[derive(Debug, Clone)]
pub struct IccProfileMetadata {
    /// Total profile size in bytes (from header field 0–3).
    pub profile_size: u32,
    /// Preferred CMM type (4-byte tag, bytes 4–7).
    pub preferred_cmm: [u8; 4],
    /// Profile version number.
    pub version: IccVersion,
    /// Device class (Input, Display, Output, etc.).
    pub device_class: IccDeviceClass,
    /// Data colour space (RGB, CMYK, Lab, etc.).
    pub color_space: IccColorSpace,
    /// Profile Connection Space (usually XYZ or Lab).
    pub pcs: IccColorSpace,
    /// ISO 8601-style creation date/time string (`"YYYY-MM-DD HH:MM:SS"`).
    pub creation_datetime: String,
    /// Profile file signature — always `b"acsp"` for valid profiles.
    pub signature: [u8; 4],
    /// Primary platform tag (e.g. `b"APPL"`, `b"MSFT"`).
    pub platform: [u8; 4],
    /// Profile flags bitfield (bytes 44–47).
    pub flags: u32,
    /// Device manufacturer tag (bytes 48–51).
    pub manufacturer: [u8; 4],
    /// Device model tag (bytes 52–55).
    pub model: [u8; 4],
    /// Rendering intent (bytes 64–67, lower 16 bits).
    pub rendering_intent: RenderingIntent,
    /// Profile illuminant as CIE XYZ (bytes 68–79, three s15Fixed16 values).
    pub illuminant: [f64; 3],
    /// All tags listed in the profile tag table.
    pub tags: Vec<IccTag>,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Reads and validates an ICC profile header and tag table.
///
/// The profile data must be at least 128 bytes.  The signature field at
/// bytes 36–39 must equal `b"acsp"`.
///
/// # Errors
///
/// Returns [`ColorError::IccProfile`] when:
/// - `data.len() < 128`
/// - The signature bytes 36–39 are not `b"acsp"`
/// - The tag table extends beyond the provided data
pub fn read_icc_metadata(data: &[u8]) -> Result<IccProfileMetadata> {
    if data.len() < 128 {
        return Err(ColorError::IccProfile(format!(
            "Profile data too short: {} bytes (minimum 128 required)",
            data.len()
        )));
    }

    // Validate 'acsp' signature at bytes 36–39
    let signature: [u8; 4] = [data[36], data[37], data[38], data[39]];
    if &signature != b"acsp" {
        return Err(ColorError::IccProfile(format!(
            "Invalid ICC signature: {:?} (expected b\"acsp\")",
            signature
        )));
    }

    let profile_size = read_u32_be(data, 0);
    let preferred_cmm: [u8; 4] = [data[4], data[5], data[6], data[7]];
    let version = IccVersion::from_bytes([data[8], data[9], data[10], data[11]]);
    let device_class = IccDeviceClass::from_bytes([data[12], data[13], data[14], data[15]]);
    let color_space = IccColorSpace::from_bytes([data[16], data[17], data[18], data[19]]);
    let pcs = IccColorSpace::from_bytes([data[20], data[21], data[22], data[23]]);
    let creation_datetime = parse_datetime(data, 24);
    let platform: [u8; 4] = [data[40], data[41], data[42], data[43]];
    let flags = read_u32_be(data, 44);
    let manufacturer: [u8; 4] = [data[48], data[49], data[50], data[51]];
    let model: [u8; 4] = [data[52], data[53], data[54], data[55]];

    // Rendering intent: lower 16 bits of the u32 at bytes 64–67
    let ri_raw = read_u32_be(data, 64) & 0xFFFF;
    let rendering_intent = match ri_raw {
        0 => RenderingIntent::Perceptual,
        1 => RenderingIntent::RelativeColorimetric,
        2 => RenderingIntent::Saturation,
        3 => RenderingIntent::AbsoluteColorimetric,
        _ => RenderingIntent::Perceptual,
    };

    // Illuminant XYZ: three s15Fixed16Number values at bytes 68–79
    let illuminant = [
        read_s15fixed16(data, 68),
        read_s15fixed16(data, 72),
        read_s15fixed16(data, 76),
    ];

    // Tag table starts at byte 128
    let tags = if data.len() >= 132 {
        parse_tag_table(data)?
    } else {
        Vec::new()
    };

    Ok(IccProfileMetadata {
        profile_size,
        preferred_cmm,
        version,
        device_class,
        color_space,
        pcs,
        creation_datetime,
        signature,
        platform,
        flags,
        manufacturer,
        model,
        rendering_intent,
        illuminant,
        tags,
    })
}

/// Extracts the human-readable profile description from the `desc` tag.
///
/// Supports both the legacy `descType` (ASCII) and the modern
/// `multiLocalizedUnicodeType` (`mluc`) formats used in ICC v4 profiles.
///
/// Returns an empty string if the `desc` tag is absent or cannot be decoded.
///
/// # Errors
///
/// Returns [`ColorError::IccProfile`] if the binary structure is invalid
/// (e.g. tag offset points outside the data buffer).
pub fn read_icc_description(data: &[u8]) -> Result<String> {
    let meta = read_icc_metadata(data)?;

    // Find the 'desc' tag
    let desc_tag = meta.tags.iter().find(|t| &t.sig == b"desc");
    let Some(tag) = desc_tag else {
        return Ok(String::new());
    };

    let start = tag.offset as usize;
    let end = start.saturating_add(tag.size as usize);

    if end > data.len() || start + 8 > data.len() {
        return Err(ColorError::IccProfile(format!(
            "desc tag extends beyond profile data (offset={}, size={})",
            tag.offset, tag.size
        )));
    }

    let tag_data = &data[start..end];

    if tag_data.len() < 4 {
        return Ok(String::new());
    }

    let type_sig = &tag_data[0..4];

    if type_sig == b"desc" {
        // Legacy textDescriptionType: count at [8..12], ASCII at [12..12+count]
        if tag_data.len() < 12 {
            return Ok(String::new());
        }
        let count =
            u32::from_be_bytes([tag_data[8], tag_data[9], tag_data[10], tag_data[11]]) as usize;
        let end_of_text = (12 + count).min(tag_data.len());
        let raw = &tag_data[12..end_of_text];
        let text = String::from_utf8_lossy(raw);
        Ok(text.trim_end_matches('\0').to_string())
    } else if type_sig == b"mluc" {
        // multiLocalizedUnicodeType
        // Header: type sig [0..4], reserved [4..8], record count [8..12], record size [12..16]
        // Each record: lang [0..2], country [2..4], length u32 [4..8], offset u32 [8..12]
        parse_mluc_description(tag_data)
    } else {
        Ok(String::new())
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Parse the tag table at byte 128 of an ICC profile.
fn parse_tag_table(data: &[u8]) -> Result<Vec<IccTag>> {
    if data.len() < 132 {
        return Ok(Vec::new());
    }
    let tag_count = read_u32_be(data, 128) as usize;

    // Each tag entry is 12 bytes; table starts at byte 132
    let table_end = 132 + tag_count * 12;
    if table_end > data.len() {
        return Err(ColorError::IccProfile(format!(
            "Tag table ({tag_count} tags) extends beyond data length {}",
            data.len()
        )));
    }

    let mut tags = Vec::with_capacity(tag_count);
    for i in 0..tag_count {
        let base = 132 + i * 12;
        let sig: [u8; 4] = [data[base], data[base + 1], data[base + 2], data[base + 3]];
        let offset = read_u32_be(data, base + 4);
        let size = read_u32_be(data, base + 8);
        tags.push(IccTag { sig, offset, size });
    }
    Ok(tags)
}

/// Read a big-endian u32 at `offset`.
fn read_u32_be(data: &[u8], offset: usize) -> u32 {
    if offset + 4 > data.len() {
        return 0;
    }
    u32::from_be_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ])
}

/// Read an `s15Fixed16Number` (signed 32-bit fixed-point, 1/65536 units) at `offset`.
fn read_s15fixed16(data: &[u8], offset: usize) -> f64 {
    if offset + 4 > data.len() {
        return 0.0;
    }
    let raw = i32::from_be_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]);
    f64::from(raw) / 65536.0
}

/// Parse the 12-byte ICC creation date/time field (six BE u16 values).
fn parse_datetime(data: &[u8], offset: usize) -> String {
    if offset + 12 > data.len() {
        return String::from("1970-01-01 00:00:00");
    }
    let year = read_u16_be(data, offset);
    let month = read_u16_be(data, offset + 2);
    let day = read_u16_be(data, offset + 4);
    let hour = read_u16_be(data, offset + 6);
    let min = read_u16_be(data, offset + 8);
    let sec = read_u16_be(data, offset + 10);
    format!("{year:04}-{month:02}-{day:02} {hour:02}:{min:02}:{sec:02}")
}

/// Read a big-endian u16 at `offset`.
fn read_u16_be(data: &[u8], offset: usize) -> u16 {
    if offset + 2 > data.len() {
        return 0;
    }
    u16::from_be_bytes([data[offset], data[offset + 1]])
}

/// Decode the first record of an `mluc` (multiLocalizedUnicode) tag as a
/// UTF-16BE string.
fn parse_mluc_description(tag_data: &[u8]) -> Result<String> {
    // mluc layout:
    //   [0..4]  type sig 'mluc'
    //   [4..8]  reserved
    //   [8..12] record count
    //   [12..16] record size (should be 12)
    //   [16..]  records: lang[2], country[2], length u32, offset u32
    if tag_data.len() < 20 {
        return Ok(String::new());
    }
    let record_count =
        u32::from_be_bytes([tag_data[8], tag_data[9], tag_data[10], tag_data[11]]) as usize;
    if record_count == 0 || tag_data.len() < 28 {
        return Ok(String::new());
    }
    // First record at offset 16
    let str_length =
        u32::from_be_bytes([tag_data[20], tag_data[21], tag_data[22], tag_data[23]]) as usize;
    let str_offset =
        u32::from_be_bytes([tag_data[24], tag_data[25], tag_data[26], tag_data[27]]) as usize;

    // str_offset is relative to the start of the tag_data
    let str_end = str_offset.saturating_add(str_length);
    if str_end > tag_data.len() || str_length % 2 != 0 {
        return Ok(String::new());
    }

    let utf16_bytes = &tag_data[str_offset..str_end];
    let u16_chars: Vec<u16> = utf16_bytes
        .chunks_exact(2)
        .map(|c| u16::from_be_bytes([c[0], c[1]]))
        .collect();

    Ok(String::from_utf16_lossy(&u16_chars)
        .trim_end_matches('\0')
        .to_string())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid ICC profile in memory.
    ///
    /// The profile has the required 'acsp' signature and can optionally carry
    /// a `desc` tag (ASCII `descType` format).
    fn build_minimal_profile(
        major: u8,
        device_class: &[u8; 4],
        color_space: &[u8; 4],
        pcs: &[u8; 4],
        rendering_intent: u32,
        desc_text: Option<&str>,
    ) -> Vec<u8> {
        // We build a profile large enough for header + tag table + optional desc tag
        let mut data = vec![0u8; 400];

        // Profile size (will be updated after we know total size)
        let profile_size: u32 = data.len() as u32;
        data[0..4].copy_from_slice(&profile_size.to_be_bytes());

        // Preferred CMM: none (zeros)

        // Version: major.0.0
        data[8] = major;
        data[9] = 0x40; // minor=4, sub=0 (like ICC v2.4.0 or v4.4.0)

        // Device class
        data[12..16].copy_from_slice(device_class);
        // Color space
        data[16..20].copy_from_slice(color_space);
        // PCS
        data[20..24].copy_from_slice(pcs);

        // Creation date: 2024-01-15 12:30:45
        let datetime: [(usize, u16); 6] = [
            (24, 2024), // year
            (26, 1),    // month
            (28, 15),   // day
            (30, 12),   // hour
            (32, 30),   // minute
            (34, 45),   // second
        ];
        for (off, val) in datetime {
            data[off..off + 2].copy_from_slice(&val.to_be_bytes());
        }

        // Signature 'acsp'
        data[36..40].copy_from_slice(b"acsp");

        // Platform: 'APPL'
        data[40..44].copy_from_slice(b"APPL");

        // Rendering intent
        data[64..68].copy_from_slice(&rendering_intent.to_be_bytes());

        // Illuminant: D50 = (0.9642, 1.0000, 0.8249) as s15Fixed16
        let d50_x: i32 = (0.9642 * 65536.0) as i32;
        let d50_y: i32 = (1.0000 * 65536.0) as i32;
        let d50_z: i32 = (0.8249 * 65536.0) as i32;
        data[68..72].copy_from_slice(&d50_x.to_be_bytes());
        data[72..76].copy_from_slice(&d50_y.to_be_bytes());
        data[76..80].copy_from_slice(&d50_z.to_be_bytes());

        if let Some(text) = desc_text {
            // Build a desc tag at offset 200
            let tag_data_offset: u32 = 200;
            let desc_bytes = text.as_bytes();
            let desc_count = desc_bytes.len() as u32 + 1; // include NUL terminator
            let tag_size = 12 + desc_count;

            // Tag count = 1
            data[128..132].copy_from_slice(&1u32.to_be_bytes());
            // Tag entry: sig='desc', offset=200, size=tag_size
            data[132..136].copy_from_slice(b"desc");
            data[136..140].copy_from_slice(&tag_data_offset.to_be_bytes());
            data[140..144].copy_from_slice(&tag_size.to_be_bytes());

            // Write tag data at offset 200
            let base = tag_data_offset as usize;
            data[base..base + 4].copy_from_slice(b"desc");
            // reserved [4..8] = 0
            // count at [8..12]
            data[base + 8..base + 12].copy_from_slice(&desc_count.to_be_bytes());
            // ASCII text at [12..]
            let text_start = base + 12;
            let text_end = text_start + desc_bytes.len();
            data[text_start..text_end].copy_from_slice(desc_bytes);
            // NUL terminator already zero from vec initialisation
        } else {
            // Tag count = 0
            data[128..132].copy_from_slice(&0u32.to_be_bytes());
        }

        data
    }

    #[test]
    fn test_read_icc_metadata_minimal() {
        let data = build_minimal_profile(4, b"mntr", b"RGB ", b"XYZ ", 0, None);
        let meta = read_icc_metadata(&data).expect("should parse valid profile");
        assert_eq!(meta.signature, *b"acsp");
        assert_eq!(meta.version.major, 4);
        assert!(matches!(meta.device_class, IccDeviceClass::Display));
        assert!(matches!(meta.color_space, IccColorSpace::RGB));
        assert!(matches!(meta.pcs, IccColorSpace::XYZ));
    }

    #[test]
    fn test_read_icc_metadata_invalid_signature() {
        let mut data = build_minimal_profile(4, b"mntr", b"RGB ", b"XYZ ", 0, None);
        // Corrupt signature
        data[36] = b'X';
        let result = read_icc_metadata(&data);
        assert!(result.is_err(), "should fail on invalid signature");
    }

    #[test]
    fn test_read_icc_metadata_too_small() {
        let data = vec![0u8; 100];
        let result = read_icc_metadata(&data);
        assert!(result.is_err(), "should fail on data < 128 bytes");
    }

    #[test]
    fn test_icc_device_class_display() {
        let dc = IccDeviceClass::from_bytes(*b"mntr");
        assert!(matches!(dc, IccDeviceClass::Display));
    }

    #[test]
    fn test_icc_device_class_input() {
        let dc = IccDeviceClass::from_bytes(*b"scnr");
        assert!(matches!(dc, IccDeviceClass::Input));
    }

    #[test]
    fn test_icc_device_class_output() {
        let dc = IccDeviceClass::from_bytes(*b"prtr");
        assert!(matches!(dc, IccDeviceClass::Output));
    }

    #[test]
    fn test_icc_device_class_unknown() {
        let dc = IccDeviceClass::from_bytes(*b"xxxx");
        assert!(matches!(dc, IccDeviceClass::Unknown(_)));
    }

    #[test]
    fn test_icc_color_space_rgb() {
        let cs = IccColorSpace::from_bytes(*b"RGB ");
        assert!(matches!(cs, IccColorSpace::RGB));
    }

    #[test]
    fn test_icc_color_space_cmyk() {
        let cs = IccColorSpace::from_bytes(*b"CMYK");
        assert!(matches!(cs, IccColorSpace::CMYK));
    }

    #[test]
    fn test_icc_color_space_gray() {
        let cs = IccColorSpace::from_bytes(*b"GRAY");
        assert!(matches!(cs, IccColorSpace::Gray));
    }

    #[test]
    fn test_icc_color_space_xyz() {
        let cs = IccColorSpace::from_bytes(*b"XYZ ");
        assert!(matches!(cs, IccColorSpace::XYZ));
    }

    #[test]
    fn test_icc_color_space_lab() {
        let cs = IccColorSpace::from_bytes(*b"Lab ");
        assert!(matches!(cs, IccColorSpace::Lab));
    }

    #[test]
    fn test_icc_rendering_intent_perceptual() {
        let data = build_minimal_profile(4, b"mntr", b"RGB ", b"XYZ ", 0, None);
        let meta = read_icc_metadata(&data).expect("should parse");
        assert!(matches!(meta.rendering_intent, RenderingIntent::Perceptual));
    }

    #[test]
    fn test_icc_rendering_intent_relative_colorimetric() {
        let data = build_minimal_profile(4, b"mntr", b"RGB ", b"XYZ ", 1, None);
        let meta = read_icc_metadata(&data).expect("should parse");
        assert!(
            matches!(meta.rendering_intent, RenderingIntent::RelativeColorimetric),
            "expected RelativeColorimetric, got {:?}",
            meta.rendering_intent
        );
    }

    #[test]
    fn test_icc_rendering_intent_saturation() {
        let data = build_minimal_profile(4, b"mntr", b"RGB ", b"XYZ ", 2, None);
        let meta = read_icc_metadata(&data).expect("should parse");
        assert!(matches!(meta.rendering_intent, RenderingIntent::Saturation));
    }

    #[test]
    fn test_icc_rendering_intent_absolute_colorimetric() {
        let data = build_minimal_profile(4, b"mntr", b"RGB ", b"XYZ ", 3, None);
        let meta = read_icc_metadata(&data).expect("should parse");
        assert!(matches!(
            meta.rendering_intent,
            RenderingIntent::AbsoluteColorimetric
        ));
    }

    #[test]
    fn test_icc_illuminant_d50() {
        let data = build_minimal_profile(4, b"mntr", b"RGB ", b"XYZ ", 0, None);
        let meta = read_icc_metadata(&data).expect("should parse");
        // D50 illuminant should be approximately (0.9642, 1.0000, 0.8249)
        assert!(
            (meta.illuminant[0] - 0.9642).abs() < 0.001,
            "illuminant X: {}",
            meta.illuminant[0]
        );
        assert!(
            (meta.illuminant[1] - 1.0).abs() < 0.001,
            "illuminant Y: {}",
            meta.illuminant[1]
        );
        assert!(
            (meta.illuminant[2] - 0.8249).abs() < 0.001,
            "illuminant Z: {}",
            meta.illuminant[2]
        );
    }

    #[test]
    fn test_icc_version_parsing_v2() {
        let data = build_minimal_profile(2, b"mntr", b"RGB ", b"XYZ ", 0, None);
        let meta = read_icc_metadata(&data).expect("should parse");
        assert_eq!(meta.version.major, 2);
        assert_eq!(meta.version.minor, 4);
    }

    #[test]
    fn test_icc_version_parsing_v4() {
        let data = build_minimal_profile(4, b"mntr", b"RGB ", b"XYZ ", 0, None);
        let meta = read_icc_metadata(&data).expect("should parse");
        assert_eq!(meta.version.major, 4);
    }

    #[test]
    fn test_icc_version_display() {
        let v = IccVersion {
            major: 4,
            minor: 4,
            sub: 0,
        };
        assert_eq!(v.to_string(), "4.4.0");
    }

    #[test]
    fn test_read_icc_description_ascii() {
        let data =
            build_minimal_profile(4, b"mntr", b"RGB ", b"XYZ ", 0, Some("sRGB IEC61966-2.1"));
        let desc = read_icc_description(&data).expect("should read description");
        assert_eq!(desc, "sRGB IEC61966-2.1");
    }

    #[test]
    fn test_read_icc_description_empty_when_no_tag() {
        let data = build_minimal_profile(4, b"mntr", b"RGB ", b"XYZ ", 0, None);
        let desc = read_icc_description(&data).expect("should return empty string");
        assert_eq!(desc, "", "no desc tag should return empty string");
    }

    #[test]
    fn test_icc_tag_table_empty() {
        let data = build_minimal_profile(4, b"mntr", b"RGB ", b"XYZ ", 0, None);
        let meta = read_icc_metadata(&data).expect("should parse");
        assert_eq!(meta.tags.len(), 0, "no tags expected in minimal profile");
    }

    #[test]
    fn test_icc_tag_table_with_desc() {
        let data = build_minimal_profile(4, b"mntr", b"RGB ", b"XYZ ", 0, Some("Test Profile"));
        let meta = read_icc_metadata(&data).expect("should parse");
        assert_eq!(meta.tags.len(), 1);
        assert_eq!(&meta.tags[0].sig, b"desc");
    }

    #[test]
    fn test_icc_creation_datetime() {
        let data = build_minimal_profile(4, b"mntr", b"RGB ", b"XYZ ", 0, None);
        let meta = read_icc_metadata(&data).expect("should parse");
        assert_eq!(meta.creation_datetime, "2024-01-15 12:30:45");
    }

    #[test]
    fn test_icc_platform_appl() {
        let data = build_minimal_profile(4, b"mntr", b"RGB ", b"XYZ ", 0, None);
        let meta = read_icc_metadata(&data).expect("should parse");
        assert_eq!(&meta.platform, b"APPL");
    }

    #[test]
    fn test_read_icc_description_adobe_rgb() {
        let data = build_minimal_profile(4, b"mntr", b"RGB ", b"XYZ ", 1, Some("Adobe RGB (1998)"));
        let desc = read_icc_description(&data).expect("should read description");
        assert_eq!(desc, "Adobe RGB (1998)");
    }

    #[test]
    fn test_icc_profile_size_field() {
        let data = build_minimal_profile(4, b"mntr", b"RGB ", b"XYZ ", 0, None);
        let meta = read_icc_metadata(&data).expect("should parse");
        assert_eq!(meta.profile_size, 400);
    }
}
