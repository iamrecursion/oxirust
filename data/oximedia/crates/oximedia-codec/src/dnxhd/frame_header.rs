//! DNxHD / VC-3 frame header parsing (SMPTE ST 2019-1).
//!
//! A DNxHD frame starts with a fixed 40-byte header containing the
//! compression ID (CID), frame dimensions, chroma format, and bit depth.
//! Following the header is a slice size table (4 bytes per slice, big-endian
//! u32 giving the byte count of each compressed slice), then the slice data.
//!
//! # Frame layout
//!
//! ```text
//! Offset   Size   Field
//!  0       4      Magic: 0x00 0x00 0x02 0x80
//!  4       4      Frame marker: 0x00 0x00 0x00 0x01
//!  8       4      CID (Compression ID, big-endian u32)
//! 12       2      Frame width (big-endian u16)
//! 14       2      Frame height (big-endian u16)
//! 16       2      Active lines / height (same as height for progressive)
//! 18       1      Reserved
//! 19       1      Chroma subsampling marker (0x58 = 4:2:2, 0x48 = 4:4:4)
//! 20       2      Bit-depth marker (0x5814 = 8-bit, 0x58A4 = 10-bit)
//! 22       2      Number of slices (big-endian u16)
//! 24       2      Macroblock width in units (big-endian u16)
//! 26       remaining: slice size table + slice data
//! ```

use super::DecodeError;

/// DNxHD compression profile, derived from the CID field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DnxhdProfile {
    /// CID 1237 — 1440×1080, 8-bit 4:2:2, ~145 Mbit/s.
    Dnxhd145,
    /// CID 1238 — 1920×1080, 8-bit 4:2:2, ~220 Mbit/s.
    Dnxhd220,
    /// CID 1235 — 1920×1080, 10-bit 4:2:2, ~220 Mbit/s.
    Dnxhd220x,
    /// CID 1241 — 1440×1080, 10-bit 4:2:2, ~145 Mbit/s.
    Dnxhd145x,
    /// CID 1242 — 1280×720, 8-bit 4:2:2, ~100 Mbit/s.
    Dnxhd100,
    /// CID 1243 — 1280×720, 8-bit 4:2:2, ~60 Mbit/s.
    Dnxhd60,
    /// Any other CID not recognized by this implementation.
    Unknown(u32),
}

impl DnxhdProfile {
    /// Resolve a CID to a profile.
    fn from_cid(cid: u32) -> Self {
        match cid {
            1237 => Self::Dnxhd145,
            1238 => Self::Dnxhd220,
            1235 => Self::Dnxhd220x,
            1241 => Self::Dnxhd145x,
            1242 => Self::Dnxhd100,
            1243 => Self::Dnxhd60,
            other => Self::Unknown(other),
        }
    }

    /// True if this profile uses 10-bit output.
    pub fn is_10bit(self) -> bool {
        matches!(self, Self::Dnxhd220x | Self::Dnxhd145x)
    }
}

impl std::fmt::Display for DnxhdProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dnxhd145 => write!(f, "DNxHD 145 (CID 1237)"),
            Self::Dnxhd220 => write!(f, "DNxHD 220 (CID 1238)"),
            Self::Dnxhd220x => write!(f, "DNxHD 220x (CID 1235)"),
            Self::Dnxhd145x => write!(f, "DNxHD 145x (CID 1241)"),
            Self::Dnxhd100 => write!(f, "DNxHD 100 (CID 1242)"),
            Self::Dnxhd60 => write!(f, "DNxHD 60 (CID 1243)"),
            Self::Unknown(c) => write!(f, "Unknown CID {c}"),
        }
    }
}

/// Parsed DNxHD frame header.
#[derive(Debug, Clone)]
pub struct FrameHeader {
    /// Compression ID.
    pub cid: u32,
    /// Decoded profile.
    pub profile: DnxhdProfile,
    /// Luma width in pixels.
    pub width: u16,
    /// Luma height in pixels.
    pub height: u16,
    /// Chroma subsampling marker byte.
    pub chroma_format: u8,
    /// Bits per pixel: 8 or 10.
    pub bits_per_pixel: u8,
    /// Number of compressed slices in this frame.
    pub num_slices: u16,
    /// Number of macroblock columns (width / 16, rounded up).
    pub mb_width: u16,
}

/// The 4-byte frame magic value.
pub(crate) const FRAME_MAGIC: [u8; 4] = [0x00, 0x00, 0x02, 0x80];
/// The 4-byte frame-type marker following the magic.
const FRAME_MARKER: [u8; 4] = [0x00, 0x00, 0x00, 0x01];
/// Minimum header bytes we must have before reading the slice table.
const HEADER_MIN_LEN: usize = 26;

/// Parse a DNxHD frame header from the start of `data`.
///
/// Returns `(header, bytes_consumed)` where `bytes_consumed` is the number
/// of bytes up to (and including) the slice size table start.  The caller
/// should advance past `bytes_consumed` to reach the slice data.
pub fn parse_frame_header(data: &[u8]) -> Result<(FrameHeader, usize), DecodeError> {
    if data.len() < HEADER_MIN_LEN {
        return Err(DecodeError::BufferTooSmall {
            need: HEADER_MIN_LEN,
            have: data.len(),
        });
    }

    // Magic check.
    if data[0..4] != FRAME_MAGIC {
        return Err(DecodeError::InvalidMagic);
    }
    if data[4..8] != FRAME_MARKER {
        return Err(DecodeError::InvalidMagic);
    }

    let cid = u32::from_be_bytes([data[8], data[9], data[10], data[11]]);
    let profile = DnxhdProfile::from_cid(cid);

    let width = u16::from_be_bytes([data[12], data[13]]);
    let height = u16::from_be_bytes([data[14], data[15]]);
    // active lines at [16..18] — we ignore, using height.

    let chroma_format = data[19]; // 0x58 = 4:2:2, 0x48 = 4:4:4
    let bpp_marker = u16::from_be_bytes([data[20], data[21]]);

    let bits_per_pixel: u8 = match bpp_marker {
        0x5814 => 8,
        0x58A4 => 10,
        // Fall back to profile knowledge when marker is non-standard.
        _ if profile.is_10bit() => 10,
        _ => 8,
    };

    let num_slices = u16::from_be_bytes([data[22], data[23]]);
    let mb_width = u16::from_be_bytes([data[24], data[25]]);

    let header = FrameHeader {
        cid,
        profile,
        width,
        height,
        chroma_format,
        bits_per_pixel,
        num_slices,
        mb_width,
    };

    // Consumed = 26-byte fixed header; slice size table follows immediately.
    Ok((header, HEADER_MIN_LEN))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Build a minimal 40-byte DNxHD header for CID 1237 (DNxHD 145).
    pub fn make_test_header(cid: u32, width: u16, height: u16, bpp_marker: u16) -> Vec<u8> {
        let mut h = vec![0u8; 40];
        // Magic + marker
        h[0..4].copy_from_slice(&FRAME_MAGIC);
        h[4..8].copy_from_slice(&FRAME_MARKER);
        // CID
        h[8..12].copy_from_slice(&cid.to_be_bytes());
        // Dimensions
        h[12..14].copy_from_slice(&width.to_be_bytes());
        h[14..16].copy_from_slice(&height.to_be_bytes());
        h[16..18].copy_from_slice(&height.to_be_bytes()); // active lines = height
                                                          // h[18] reserved
        h[19] = 0x58; // 4:2:2
        h[20..22].copy_from_slice(&bpp_marker.to_be_bytes());
        // num_slices = height / 16 (one slice per 16 lines)
        let ns = (height / 16).max(1);
        h[22..24].copy_from_slice(&ns.to_be_bytes());
        // mb_width = width / 16
        let mbw = (width / 16).max(1);
        h[24..26].copy_from_slice(&mbw.to_be_bytes());
        h
    }

    #[test]
    fn parse_dnxhd145_header() {
        let data = make_test_header(1237, 1440, 1080, 0x5814);
        let (hdr, consumed) = parse_frame_header(&data).unwrap();
        assert_eq!(hdr.profile, DnxhdProfile::Dnxhd145);
        assert_eq!(hdr.cid, 1237);
        assert_eq!(hdr.width, 1440);
        assert_eq!(hdr.height, 1080);
        assert_eq!(hdr.bits_per_pixel, 8);
        assert_eq!(hdr.chroma_format, 0x58);
        assert_eq!(consumed, HEADER_MIN_LEN);
    }

    #[test]
    fn parse_dnxhd220_header() {
        let data = make_test_header(1238, 1920, 1080, 0x5814);
        let (hdr, _) = parse_frame_header(&data).unwrap();
        assert_eq!(hdr.profile, DnxhdProfile::Dnxhd220);
        assert_eq!(hdr.bits_per_pixel, 8);
    }

    #[test]
    fn parse_dnxhd220x_10bit() {
        let data = make_test_header(1235, 1920, 1080, 0x58A4);
        let (hdr, _) = parse_frame_header(&data).unwrap();
        assert_eq!(hdr.profile, DnxhdProfile::Dnxhd220x);
        assert_eq!(hdr.bits_per_pixel, 10);
    }

    #[test]
    fn parse_dnxhd145x_10bit() {
        let data = make_test_header(1241, 1440, 1080, 0x58A4);
        let (hdr, _) = parse_frame_header(&data).unwrap();
        assert_eq!(hdr.profile, DnxhdProfile::Dnxhd145x);
        assert_eq!(hdr.bits_per_pixel, 10);
    }

    #[test]
    fn bad_magic_errors() {
        let mut data = make_test_header(1237, 1440, 1080, 0x5814);
        data[0] = 0xFF;
        assert!(matches!(
            parse_frame_header(&data),
            Err(DecodeError::InvalidMagic)
        ));
    }

    #[test]
    fn short_buffer_errors() {
        let data = [0u8; 10];
        assert!(matches!(
            parse_frame_header(&data),
            Err(DecodeError::BufferTooSmall { .. })
        ));
    }

    #[test]
    fn unknown_cid_is_unknown_variant() {
        let data = make_test_header(9999, 1920, 1080, 0x5814);
        let (hdr, _) = parse_frame_header(&data).unwrap();
        assert!(matches!(hdr.profile, DnxhdProfile::Unknown(9999)));
    }

    #[test]
    fn dnxhd_profile_is_10bit_correct() {
        assert!(DnxhdProfile::Dnxhd220x.is_10bit());
        assert!(DnxhdProfile::Dnxhd145x.is_10bit());
        assert!(!DnxhdProfile::Dnxhd145.is_10bit());
        assert!(!DnxhdProfile::Dnxhd220.is_10bit());
    }
}
