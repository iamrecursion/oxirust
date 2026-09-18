#![allow(dead_code)]
//! EXIF tag parsing utilities for extracting structured data from raw EXIF byte streams.
//!
//! Provides low-level byte-oriented EXIF parsing including:
//! - TIFF header detection and byte order negotiation
//! - IFD (Image File Directory) traversal
//! - Standard EXIF tag extraction (camera model, exposure, GPS, etc.)
//! - Rational number handling for focal length, aperture, etc.

use std::collections::HashMap;
use std::fmt;

/// Byte order used in the EXIF data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteOrder {
    /// Little-endian (Intel byte order, marker `II`)
    LittleEndian,
    /// Big-endian (Motorola byte order, marker `MM`)
    BigEndian,
}

impl fmt::Display for ByteOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LittleEndian => write!(f, "Little-Endian (II)"),
            Self::BigEndian => write!(f, "Big-Endian (MM)"),
        }
    }
}

/// A rational number stored as numerator/denominator (common in EXIF).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rational {
    /// Numerator of the rational value.
    pub numerator: u32,
    /// Denominator of the rational value.
    pub denominator: u32,
}

impl Rational {
    /// Create a new rational number.
    pub fn new(numerator: u32, denominator: u32) -> Self {
        Self {
            numerator,
            denominator,
        }
    }

    /// Convert rational to floating-point.
    #[allow(clippy::cast_precision_loss)]
    pub fn to_f64(self) -> f64 {
        if self.denominator == 0 {
            return 0.0;
        }
        self.numerator as f64 / self.denominator as f64
    }

    /// Simplify the rational number by dividing by GCD.
    pub fn simplify(self) -> Self {
        let g = gcd(self.numerator, self.denominator);
        if g == 0 {
            return self;
        }
        Self {
            numerator: self.numerator / g,
            denominator: self.denominator / g,
        }
    }

    /// Check if this rational represents an integer value.
    pub fn is_integer(&self) -> bool {
        self.denominator != 0 && self.numerator % self.denominator == 0
    }
}

impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.denominator == 1 {
            write!(f, "{}", self.numerator)
        } else {
            write!(f, "{}/{}", self.numerator, self.denominator)
        }
    }
}

/// Compute GCD using Euclid's algorithm.
fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// A signed rational number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SRational {
    /// Numerator of the signed rational value.
    pub numerator: i32,
    /// Denominator of the signed rational value.
    pub denominator: i32,
}

impl SRational {
    /// Create a new signed rational number.
    pub fn new(numerator: i32, denominator: i32) -> Self {
        Self {
            numerator,
            denominator,
        }
    }

    /// Convert signed rational to floating-point.
    #[allow(clippy::cast_precision_loss)]
    pub fn to_f64(self) -> f64 {
        if self.denominator == 0 {
            return 0.0;
        }
        self.numerator as f64 / self.denominator as f64
    }
}

impl fmt::Display for SRational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.denominator == 1 {
            write!(f, "{}", self.numerator)
        } else {
            write!(f, "{}/{}", self.numerator, self.denominator)
        }
    }
}

/// EXIF data type identifiers (from the TIFF specification).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExifDataType {
    /// 8-bit unsigned integer.
    Byte,
    /// ASCII string (7-bit).
    Ascii,
    /// 16-bit unsigned integer.
    Short,
    /// 32-bit unsigned integer.
    Long,
    /// Two LONGs: numerator, denominator.
    Rational,
    /// 8-bit signed integer.
    SByte,
    /// Undefined byte sequence.
    Undefined,
    /// 16-bit signed integer.
    SShort,
    /// 32-bit signed integer.
    SLong,
    /// Two SLONGs: signed numerator, signed denominator.
    SRational,
    /// 32-bit IEEE float.
    Float,
    /// 64-bit IEEE double.
    Double,
    /// Unknown or vendor-specific data type.
    Unknown(u16),
}

impl ExifDataType {
    /// Parse a data type from the TIFF type code.
    pub fn from_code(code: u16) -> Self {
        match code {
            1 => Self::Byte,
            2 => Self::Ascii,
            3 => Self::Short,
            4 => Self::Long,
            5 => Self::Rational,
            6 => Self::SByte,
            7 => Self::Undefined,
            8 => Self::SShort,
            9 => Self::SLong,
            10 => Self::SRational,
            11 => Self::Float,
            12 => Self::Double,
            other => Self::Unknown(other),
        }
    }

    /// Get the size in bytes for one element of this data type.
    pub fn element_size(&self) -> usize {
        match self {
            Self::Byte | Self::SByte | Self::Ascii | Self::Undefined => 1,
            Self::Short | Self::SShort => 2,
            Self::Long | Self::SLong | Self::Float => 4,
            Self::Rational | Self::SRational | Self::Double => 8,
            Self::Unknown(_) => 1,
        }
    }

    /// Convert back to the TIFF type code.
    pub fn to_code(&self) -> u16 {
        match self {
            Self::Byte => 1,
            Self::Ascii => 2,
            Self::Short => 3,
            Self::Long => 4,
            Self::Rational => 5,
            Self::SByte => 6,
            Self::Undefined => 7,
            Self::SShort => 8,
            Self::SLong => 9,
            Self::SRational => 10,
            Self::Float => 11,
            Self::Double => 12,
            Self::Unknown(c) => *c,
        }
    }
}

/// A single parsed EXIF tag entry.
#[derive(Debug, Clone)]
pub struct ExifTag {
    /// Tag ID (e.g., 0x010F for Make).
    pub tag_id: u16,
    /// Data type of the tag.
    pub data_type: ExifDataType,
    /// Number of values.
    pub count: u32,
    /// Raw value bytes.
    pub raw_value: Vec<u8>,
}

impl ExifTag {
    /// Create a new EXIF tag entry.
    pub fn new(tag_id: u16, data_type: ExifDataType, count: u32, raw_value: Vec<u8>) -> Self {
        Self {
            tag_id,
            data_type,
            count,
            raw_value,
        }
    }

    /// Try to interpret the tag value as an ASCII string.
    pub fn as_ascii(&self) -> Option<String> {
        if self.data_type != ExifDataType::Ascii {
            return None;
        }
        let bytes: Vec<u8> = self
            .raw_value
            .iter()
            .copied()
            .take_while(|&b| b != 0)
            .collect();
        String::from_utf8(bytes).ok()
    }

    /// Try to interpret the tag value as a u16 (SHORT).
    pub fn as_short(&self) -> Option<u16> {
        if self.raw_value.len() < 2 {
            return None;
        }
        Some(u16::from_le_bytes([self.raw_value[0], self.raw_value[1]]))
    }

    /// Try to interpret the tag value as a u32 (LONG).
    pub fn as_long(&self) -> Option<u32> {
        if self.raw_value.len() < 4 {
            return None;
        }
        Some(u32::from_le_bytes([
            self.raw_value[0],
            self.raw_value[1],
            self.raw_value[2],
            self.raw_value[3],
        ]))
    }

    /// Try to interpret the tag value as a Rational.
    pub fn as_rational(&self) -> Option<Rational> {
        if self.raw_value.len() < 8 {
            return None;
        }
        let num = u32::from_le_bytes([
            self.raw_value[0],
            self.raw_value[1],
            self.raw_value[2],
            self.raw_value[3],
        ]);
        let den = u32::from_le_bytes([
            self.raw_value[4],
            self.raw_value[5],
            self.raw_value[6],
            self.raw_value[7],
        ]);
        Some(Rational::new(num, den))
    }

    /// Compute total byte length of the value data.
    pub fn value_byte_length(&self) -> usize {
        self.data_type.element_size() * self.count as usize
    }
}

/// Standard well-known EXIF tag IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WellKnownTag {
    /// Camera manufacturer (0x010F).
    Make,
    /// Camera model (0x0110).
    Model,
    /// Image orientation (0x0112).
    Orientation,
    /// X resolution (0x011A).
    XResolution,
    /// Y resolution (0x011B).
    YResolution,
    /// Exposure time in seconds (0x829A).
    ExposureTime,
    /// F-number / aperture (0x829D).
    FNumber,
    /// ISO speed ratings (0x8827).
    IsoSpeed,
    /// Date/time original (0x9003).
    DateTimeOriginal,
    /// Focal length in mm (0x920A).
    FocalLength,
    /// Image width (0xA002).
    PixelXDimension,
    /// Image height (0xA003).
    PixelYDimension,
    /// GPS latitude reference (N/S) (0x0001).
    GpsLatitudeRef,
    /// GPS latitude (0x0002).
    GpsLatitude,
    /// GPS longitude reference (E/W) (0x0003).
    GpsLongitudeRef,
    /// GPS longitude (0x0004).
    GpsLongitude,
}

impl WellKnownTag {
    /// Get the numeric tag ID.
    pub fn tag_id(self) -> u16 {
        match self {
            Self::Make => 0x010F,
            Self::Model => 0x0110,
            Self::Orientation => 0x0112,
            Self::XResolution => 0x011A,
            Self::YResolution => 0x011B,
            Self::ExposureTime => 0x829A,
            Self::FNumber => 0x829D,
            Self::IsoSpeed => 0x8827,
            Self::DateTimeOriginal => 0x9003,
            Self::FocalLength => 0x920A,
            Self::PixelXDimension => 0xA002,
            Self::PixelYDimension => 0xA003,
            Self::GpsLatitudeRef => 0x0001,
            Self::GpsLatitude => 0x0002,
            Self::GpsLongitudeRef => 0x0003,
            Self::GpsLongitude => 0x0004,
        }
    }

    /// Try to match a tag ID to a well-known tag.
    pub fn from_tag_id(id: u16) -> Option<Self> {
        match id {
            0x010F => Some(Self::Make),
            0x0110 => Some(Self::Model),
            0x0112 => Some(Self::Orientation),
            0x011A => Some(Self::XResolution),
            0x011B => Some(Self::YResolution),
            0x829A => Some(Self::ExposureTime),
            0x829D => Some(Self::FNumber),
            0x8827 => Some(Self::IsoSpeed),
            0x9003 => Some(Self::DateTimeOriginal),
            0x920A => Some(Self::FocalLength),
            0xA002 => Some(Self::PixelXDimension),
            0xA003 => Some(Self::PixelYDimension),
            0x0001 => Some(Self::GpsLatitudeRef),
            0x0002 => Some(Self::GpsLatitude),
            0x0003 => Some(Self::GpsLongitudeRef),
            0x0004 => Some(Self::GpsLongitude),
            _ => None,
        }
    }

    /// Human-readable name for this tag.
    pub fn display_name(self) -> &'static str {
        match self {
            Self::Make => "Make",
            Self::Model => "Model",
            Self::Orientation => "Orientation",
            Self::XResolution => "X Resolution",
            Self::YResolution => "Y Resolution",
            Self::ExposureTime => "Exposure Time",
            Self::FNumber => "F-Number",
            Self::IsoSpeed => "ISO Speed",
            Self::DateTimeOriginal => "Date/Time Original",
            Self::FocalLength => "Focal Length",
            Self::PixelXDimension => "Pixel X Dimension",
            Self::PixelYDimension => "Pixel Y Dimension",
            Self::GpsLatitudeRef => "GPS Latitude Ref",
            Self::GpsLatitude => "GPS Latitude",
            Self::GpsLongitudeRef => "GPS Longitude Ref",
            Self::GpsLongitude => "GPS Longitude",
        }
    }
}

/// Result of EXIF header detection.
#[derive(Debug, Clone)]
pub struct TiffHeader {
    /// Detected byte order.
    pub byte_order: ByteOrder,
    /// Offset to the first IFD (from start of TIFF data).
    pub ifd0_offset: u32,
    /// Whether the magic number (42) was valid.
    pub valid: bool,
}

/// Detect the TIFF header from raw bytes.
///
/// Expects at least 8 bytes: 2 byte-order marker, 2 magic (0x002A), 4 IFD0 offset.
pub fn detect_tiff_header(data: &[u8]) -> Option<TiffHeader> {
    if data.len() < 8 {
        return None;
    }

    let byte_order = match (data[0], data[1]) {
        (b'I', b'I') => ByteOrder::LittleEndian,
        (b'M', b'M') => ByteOrder::BigEndian,
        _ => return None,
    };

    let magic = read_u16(data, 2, byte_order);
    let valid = magic == 42;

    let ifd0_offset = read_u32(data, 4, byte_order);

    Some(TiffHeader {
        byte_order,
        ifd0_offset,
        valid,
    })
}

/// Read a u16 from `data` at `offset` with the given byte order.
fn read_u16(data: &[u8], offset: usize, order: ByteOrder) -> u16 {
    if offset + 2 > data.len() {
        return 0;
    }
    match order {
        ByteOrder::LittleEndian => u16::from_le_bytes([data[offset], data[offset + 1]]),
        ByteOrder::BigEndian => u16::from_be_bytes([data[offset], data[offset + 1]]),
    }
}

/// Read a u32 from `data` at `offset` with the given byte order.
fn read_u32(data: &[u8], offset: usize, order: ByteOrder) -> u32 {
    if offset + 4 > data.len() {
        return 0;
    }
    match order {
        ByteOrder::LittleEndian => u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]),
        ByteOrder::BigEndian => u32::from_be_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]),
    }
}

/// Parsed collection of EXIF tags from one IFD.
#[derive(Debug, Clone)]
pub struct ParsedIfd {
    /// The IFD index (0 = IFD0, 1 = IFD1, ...).
    pub index: usize,
    /// Tags within this IFD.
    pub tags: Vec<ExifTag>,
    /// Offset to the next IFD (0 if none).
    pub next_ifd_offset: u32,
}

impl ParsedIfd {
    /// Create a new empty IFD.
    pub fn new(index: usize) -> Self {
        Self {
            index,
            tags: Vec::new(),
            next_ifd_offset: 0,
        }
    }

    /// Find a tag by its tag ID.
    pub fn find_tag(&self, tag_id: u16) -> Option<&ExifTag> {
        self.tags.iter().find(|t| t.tag_id == tag_id)
    }

    /// Get tag count.
    pub fn tag_count(&self) -> usize {
        self.tags.len()
    }

    /// Check if the IFD has a specific tag.
    pub fn has_tag(&self, tag_id: u16) -> bool {
        self.tags.iter().any(|t| t.tag_id == tag_id)
    }
}

/// Full parsed EXIF data structure.
#[derive(Debug, Clone)]
pub struct ParsedExif {
    /// Byte order of the source data.
    pub byte_order: ByteOrder,
    /// Parsed IFDs.
    pub ifds: Vec<ParsedIfd>,
    /// Convenience map: tag_id -> (ifd_index, tag_index).
    tag_map: HashMap<u16, (usize, usize)>,
}

impl ParsedExif {
    /// Create a new parsed EXIF structure.
    pub fn new(byte_order: ByteOrder) -> Self {
        Self {
            byte_order,
            ifds: Vec::new(),
            tag_map: HashMap::new(),
        }
    }

    /// Add an IFD and index its tags.
    pub fn add_ifd(&mut self, ifd: ParsedIfd) {
        let ifd_idx = self.ifds.len();
        for (tag_idx, tag) in ifd.tags.iter().enumerate() {
            self.tag_map.insert(tag.tag_id, (ifd_idx, tag_idx));
        }
        self.ifds.push(ifd);
    }

    /// Look up a tag across all IFDs.
    pub fn find_tag(&self, tag_id: u16) -> Option<&ExifTag> {
        if let Some(&(ifd_idx, tag_idx)) = self.tag_map.get(&tag_id) {
            self.ifds.get(ifd_idx).and_then(|ifd| ifd.tags.get(tag_idx))
        } else {
            None
        }
    }

    /// Total number of tags across all IFDs.
    pub fn total_tags(&self) -> usize {
        self.ifds.iter().map(|ifd| ifd.tags.len()).sum()
    }

    /// Get the camera make string if present.
    pub fn camera_make(&self) -> Option<String> {
        self.find_tag(WellKnownTag::Make.tag_id())
            .and_then(|t| t.as_ascii())
    }

    /// Get the camera model string if present.
    pub fn camera_model(&self) -> Option<String> {
        self.find_tag(WellKnownTag::Model.tag_id())
            .and_then(|t| t.as_ascii())
    }

    /// Get the orientation value if present.
    pub fn orientation(&self) -> Option<u16> {
        self.find_tag(WellKnownTag::Orientation.tag_id())
            .and_then(|t| t.as_short())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rational_new_and_display() {
        let r = Rational::new(1, 100);
        assert_eq!(r.numerator, 1);
        assert_eq!(r.denominator, 100);
        assert_eq!(r.to_string(), "1/100");
    }

    #[test]
    fn test_rational_to_f64() {
        let r = Rational::new(3, 4);
        let val = r.to_f64();
        assert!((val - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_rational_zero_denominator() {
        let r = Rational::new(5, 0);
        assert_eq!(r.to_f64(), 0.0);
    }

    #[test]
    fn test_rational_simplify() {
        let r = Rational::new(6, 4).simplify();
        assert_eq!(r.numerator, 3);
        assert_eq!(r.denominator, 2);
    }

    #[test]
    fn test_rational_is_integer() {
        assert!(Rational::new(10, 5).is_integer());
        assert!(!Rational::new(10, 3).is_integer());
        assert!(!Rational::new(1, 0).is_integer());
    }

    #[test]
    fn test_rational_display_whole_number() {
        let r = Rational::new(42, 1);
        assert_eq!(r.to_string(), "42");
    }

    #[test]
    fn test_srational_to_f64() {
        let sr = SRational::new(-3, 4);
        let val = sr.to_f64();
        assert!((val - (-0.75)).abs() < 1e-10);
    }

    #[test]
    fn test_srational_zero_denominator() {
        let sr = SRational::new(-5, 0);
        assert_eq!(sr.to_f64(), 0.0);
    }

    #[test]
    fn test_exif_data_type_roundtrip() {
        for code in 1u16..=12 {
            let dt = ExifDataType::from_code(code);
            assert_eq!(dt.to_code(), code);
        }
    }

    #[test]
    fn test_exif_data_type_element_size() {
        assert_eq!(ExifDataType::Byte.element_size(), 1);
        assert_eq!(ExifDataType::Short.element_size(), 2);
        assert_eq!(ExifDataType::Long.element_size(), 4);
        assert_eq!(ExifDataType::Rational.element_size(), 8);
        assert_eq!(ExifDataType::Double.element_size(), 8);
        assert_eq!(ExifDataType::Unknown(99).element_size(), 1);
    }

    #[test]
    fn test_exif_tag_as_ascii() {
        let tag = ExifTag::new(0x010F, ExifDataType::Ascii, 6, b"Canon\0".to_vec());
        assert_eq!(tag.as_ascii(), Some("Canon".to_string()));
    }

    #[test]
    fn test_exif_tag_as_short() {
        let tag = ExifTag::new(0x0112, ExifDataType::Short, 1, vec![1, 0]);
        assert_eq!(tag.as_short(), Some(1));
    }

    #[test]
    fn test_exif_tag_as_long() {
        let tag = ExifTag::new(0xA002, ExifDataType::Long, 1, vec![0x00, 0x10, 0x00, 0x00]);
        assert_eq!(tag.as_long(), Some(4096));
    }

    #[test]
    fn test_exif_tag_as_rational() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&125u32.to_le_bytes());
        let tag = ExifTag::new(0x829A, ExifDataType::Rational, 1, bytes);
        let rat = tag.as_rational().expect("should succeed in test");
        assert_eq!(rat.numerator, 1);
        assert_eq!(rat.denominator, 125);
    }

    #[test]
    fn test_exif_tag_value_byte_length() {
        let tag = ExifTag::new(0x829A, ExifDataType::Rational, 3, vec![0; 24]);
        assert_eq!(tag.value_byte_length(), 24);
    }

    #[test]
    fn test_well_known_tag_roundtrip() {
        let tags = [
            WellKnownTag::Make,
            WellKnownTag::Model,
            WellKnownTag::Orientation,
            WellKnownTag::ExposureTime,
            WellKnownTag::FocalLength,
        ];
        for tag in &tags {
            let id = tag.tag_id();
            assert_eq!(WellKnownTag::from_tag_id(id), Some(*tag));
        }
    }

    #[test]
    fn test_well_known_tag_unknown_id() {
        assert_eq!(WellKnownTag::from_tag_id(0xFFFF), None);
    }

    #[test]
    fn test_well_known_tag_display_name() {
        assert_eq!(WellKnownTag::Make.display_name(), "Make");
        assert_eq!(WellKnownTag::FocalLength.display_name(), "Focal Length");
    }

    #[test]
    fn test_detect_tiff_header_little_endian() {
        let mut data = vec![b'I', b'I', 0x2A, 0x00, 0x08, 0x00, 0x00, 0x00];
        let header = detect_tiff_header(&data).expect("should succeed in test");
        assert_eq!(header.byte_order, ByteOrder::LittleEndian);
        assert!(header.valid);
        assert_eq!(header.ifd0_offset, 8);
        // suppress unused mut warning
        data.push(0);
    }

    #[test]
    fn test_detect_tiff_header_big_endian() {
        let data = vec![b'M', b'M', 0x00, 0x2A, 0x00, 0x00, 0x00, 0x08];
        let header = detect_tiff_header(&data).expect("should succeed in test");
        assert_eq!(header.byte_order, ByteOrder::BigEndian);
        assert!(header.valid);
        assert_eq!(header.ifd0_offset, 8);
    }

    #[test]
    fn test_detect_tiff_header_too_short() {
        let data = vec![b'I', b'I', 0x2A];
        assert!(detect_tiff_header(&data).is_none());
    }

    #[test]
    fn test_detect_tiff_header_invalid_marker() {
        let data = vec![b'X', b'Y', 0x2A, 0x00, 0x08, 0x00, 0x00, 0x00];
        assert!(detect_tiff_header(&data).is_none());
    }

    #[test]
    fn test_parsed_ifd_operations() {
        let mut ifd = ParsedIfd::new(0);
        assert_eq!(ifd.tag_count(), 0);
        assert!(!ifd.has_tag(0x010F));

        ifd.tags.push(ExifTag::new(
            0x010F,
            ExifDataType::Ascii,
            5,
            b"Nikon\0".to_vec(),
        ));
        assert_eq!(ifd.tag_count(), 1);
        assert!(ifd.has_tag(0x010F));
        assert!(!ifd.has_tag(0x0110));

        let found = ifd.find_tag(0x010F).expect("should succeed in test");
        assert_eq!(found.as_ascii(), Some("Nikon".to_string()));
    }

    #[test]
    fn test_parsed_exif_find_tag() {
        let mut exif = ParsedExif::new(ByteOrder::LittleEndian);
        let mut ifd = ParsedIfd::new(0);
        ifd.tags.push(ExifTag::new(
            0x010F,
            ExifDataType::Ascii,
            6,
            b"Canon\0".to_vec(),
        ));
        ifd.tags.push(ExifTag::new(
            0x0110,
            ExifDataType::Ascii,
            9,
            b"EOS R5\0\0\0".to_vec(),
        ));
        exif.add_ifd(ifd);

        assert_eq!(exif.total_tags(), 2);
        assert_eq!(exif.camera_make(), Some("Canon".to_string()));
        assert_eq!(exif.camera_model(), Some("EOS R5".to_string()));
    }

    #[test]
    fn test_byte_order_display() {
        assert_eq!(ByteOrder::LittleEndian.to_string(), "Little-Endian (II)");
        assert_eq!(ByteOrder::BigEndian.to_string(), "Big-Endian (MM)");
    }

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(12, 8), 4);
        assert_eq!(gcd(7, 13), 1);
        assert_eq!(gcd(0, 5), 5);
        assert_eq!(gcd(10, 0), 10);
    }
}
