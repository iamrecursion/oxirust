//! EXIF (Exchangeable Image File Format) metadata parsing and writing support.
//!
//! EXIF metadata is commonly used in JPEG and TIFF images.
//!
//! # Format
//!
//! EXIF uses TIFF structure with IFDs (Image File Directories) containing tags.
//!
//! # Common Tags
//!
//! - **0x010F**: Make (camera manufacturer)
//! - **0x0110**: Model (camera model)
//! - **0x0132**: DateTime
//! - **0x013B**: Artist
//! - **0x8298**: Copyright
//! - **0x9003**: DateTimeOriginal
//! - **0x9004**: DateTimeDigitized
//!
//! # Makernote Support
//!
//! Manufacturer-specific Makernote tags are parsed for:
//! - **Canon** (identified by Make = "Canon"): uses standard TIFF IFD layout
//! - **Nikon** (Type 3, "Nikon\0" header): TIFF-in-TIFF sub-block
//! - **Sony** (SonyDSC header): raw IFD without nested TIFF header

use crate::{Error, Metadata, MetadataFormat, MetadataValue};
use std::io::{Cursor, Read, Seek, SeekFrom};

/// EXIF byte order marker (little-endian)
const EXIF_LE: &[u8; 2] = b"II";

/// EXIF byte order marker (big-endian)
const EXIF_BE: &[u8; 2] = b"MM";

/// TIFF magic number
const TIFF_MAGIC: u16 = 0x002A;

/// EXIF tag: Makernote (0x927C)
const TAG_MAKERNOTE: u16 = 0x927C;

/// EXIF tag: Make (camera manufacturer) used to identify Makernote dialect
const TAG_MAKE: u16 = 0x010F;

/// Byte order for reading multi-byte values
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteOrder {
    LittleEndian,
    BigEndian,
}

/// EXIF tag types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
enum TagType {
    Byte = 1,
    Ascii = 2,
    Short = 3,
    Long = 4,
    Rational = 5,
    Undefined = 7,
    SLong = 9,
    SRational = 10,
}

impl TagType {
    fn from_u16(value: u16) -> Option<Self> {
        match value {
            1 => Some(Self::Byte),
            2 => Some(Self::Ascii),
            3 => Some(Self::Short),
            4 => Some(Self::Long),
            5 => Some(Self::Rational),
            7 => Some(Self::Undefined),
            9 => Some(Self::SLong),
            10 => Some(Self::SRational),
            _ => None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Makernote dialect detection and parsing
// ─────────────────────────────────────────────────────────────────────────────

/// Camera manufacturer dialect for Makernote parsing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MakernoteDialect {
    /// Canon Makernote — standard IFD directly in the Makernote value.
    Canon,
    /// Nikon Type 3 — "Nikon\0\x02\x10\x00\x00" header + mini-TIFF block.
    Nikon,
    /// Sony DSC — "SONY DSC \0\0\0" header + raw IFD.
    Sony,
    /// Unknown / unsupported manufacturer.
    Unknown,
}

impl MakernoteDialect {
    /// Detect the dialect from the Make string and Makernote data prefix.
    #[must_use]
    pub fn detect(make: &str, data: &[u8]) -> Self {
        let make_upper = make.to_uppercase();
        if make_upper.contains("CANON") {
            return Self::Canon;
        }
        if make_upper.contains("NIKON") {
            // Nikon Type 3 starts with "Nikon\0"
            if data.len() >= 6 && &data[..6] == b"Nikon\0" {
                return Self::Nikon;
            }
        }
        if make_upper.contains("SONY") {
            // Sony Makernote starts with "SONY DSC " or just uses raw IFD
            return Self::Sony;
        }
        Self::Unknown
    }
}

/// A decoded Makernote tag entry.
#[derive(Debug, Clone, PartialEq)]
pub struct MakernoteTag {
    /// Raw tag id (manufacturer-specific).
    pub tag_id: u16,
    /// Human-readable name (may be empty for unknown tags).
    pub name: String,
    /// Decoded value.
    pub value: MetadataValue,
}

/// Parse Canon Makernote from raw bytes.
///
/// Canon uses a plain TIFF IFD starting at byte 0 of the Makernote value.
fn parse_canon_makernote(data: &[u8], byte_order: ByteOrder) -> Vec<MakernoteTag> {
    let mut out = Vec::new();
    if data.len() < 2 {
        return out;
    }

    let count = match read_u16_raw(data, 0, byte_order) {
        Some(n) => n as usize,
        None => return out,
    };

    for i in 0..count {
        let entry_off = 2 + i * 12;
        if entry_off + 12 > data.len() {
            break;
        }

        let tag_id = match read_u16_raw(data, entry_off, byte_order) {
            Some(v) => v,
            None => continue,
        };
        let type_id = match read_u16_raw(data, entry_off + 2, byte_order) {
            Some(v) => v,
            None => continue,
        };
        let count_n = match read_u32_raw(data, entry_off + 4, byte_order) {
            Some(v) => v as usize,
            None => continue,
        };

        let value = decode_ifd_entry_value(data, type_id, count_n, entry_off + 8, byte_order);
        if let Some(v) = value {
            out.push(MakernoteTag {
                tag_id,
                name: canon_tag_name(tag_id),
                value: v,
            });
        }
    }

    out
}

/// Parse Nikon Type 3 Makernote from raw bytes.
///
/// Layout: "Nikon\0\x02" (7 bytes) + byte-order (2) + magic (2) + ifd_offset (4) + IFD
fn parse_nikon_makernote(data: &[u8]) -> Vec<MakernoteTag> {
    let mut out = Vec::new();

    // Nikon type 3: "Nikon\0" then a version byte then a mini-TIFF
    // Minimum: 6 (header) + 1 (version) + 8 (TIFF header) = 15 bytes
    if data.len() < 15 {
        return out;
    }
    if &data[..6] != b"Nikon\0" {
        return out;
    }

    // The mini-TIFF block starts at offset 10 (after "Nikon\0" + 4-byte version/padding)
    let tiff_start = 10;
    if data.len() < tiff_start + 8 {
        return out;
    }

    let tiff_data = &data[tiff_start..];

    let byte_order = if tiff_data.len() >= 2 && &tiff_data[..2] == b"II" {
        ByteOrder::LittleEndian
    } else if tiff_data.len() >= 2 && &tiff_data[..2] == b"MM" {
        ByteOrder::BigEndian
    } else {
        return out;
    };

    // Skip byte-order (2) + magic (2) = 4 bytes, then read IFD offset (4 bytes)
    let ifd_offset = match read_u32_raw(tiff_data, 4, byte_order) {
        Some(v) => v as usize,
        None => return out,
    };

    if ifd_offset + 2 > tiff_data.len() {
        return out;
    }

    let count = match read_u16_raw(tiff_data, ifd_offset, byte_order) {
        Some(n) => n as usize,
        None => return out,
    };

    for i in 0..count {
        let entry_off = ifd_offset + 2 + i * 12;
        if entry_off + 12 > tiff_data.len() {
            break;
        }

        let tag_id = match read_u16_raw(tiff_data, entry_off, byte_order) {
            Some(v) => v,
            None => continue,
        };
        let type_id = match read_u16_raw(tiff_data, entry_off + 2, byte_order) {
            Some(v) => v,
            None => continue,
        };
        let count_n = match read_u32_raw(tiff_data, entry_off + 4, byte_order) {
            Some(v) => v as usize,
            None => continue,
        };

        let value = decode_ifd_entry_value(tiff_data, type_id, count_n, entry_off + 8, byte_order);
        if let Some(v) = value {
            out.push(MakernoteTag {
                tag_id,
                name: nikon_tag_name(tag_id),
                value: v,
            });
        }
    }

    out
}

/// Parse Sony Makernote from raw bytes.
///
/// Sony Makernotes begin with "SONY DSC \0\0\0" (12 bytes) followed by a
/// standard IFD (no embedded TIFF header).  We use the parent's byte order.
fn parse_sony_makernote(data: &[u8], byte_order: ByteOrder) -> Vec<MakernoteTag> {
    let mut out = Vec::new();

    // Sony header: "SONY DSC \0\0\0" (12 bytes) or raw IFD
    let ifd_start = if data.len() >= 12 && &data[..9] == b"SONY DSC " {
        12usize
    } else {
        0usize
    };

    if data.len() < ifd_start + 2 {
        return out;
    }

    let count = match read_u16_raw(data, ifd_start, byte_order) {
        Some(n) => n as usize,
        None => return out,
    };

    for i in 0..count {
        let entry_off = ifd_start + 2 + i * 12;
        if entry_off + 12 > data.len() {
            break;
        }

        let tag_id = match read_u16_raw(data, entry_off, byte_order) {
            Some(v) => v,
            None => continue,
        };
        let type_id = match read_u16_raw(data, entry_off + 2, byte_order) {
            Some(v) => v,
            None => continue,
        };
        let count_n = match read_u32_raw(data, entry_off + 4, byte_order) {
            Some(v) => v as usize,
            None => continue,
        };

        let value = decode_ifd_entry_value(data, type_id, count_n, entry_off + 8, byte_order);
        if let Some(v) = value {
            out.push(MakernoteTag {
                tag_id,
                name: sony_tag_name(tag_id),
                value: v,
            });
        }
    }

    out
}

/// Decode a single IFD entry value using inline or offset data.
///
/// Returns `None` for unsupported or malformed entries.
fn decode_ifd_entry_value(
    data: &[u8],
    type_id: u16,
    count: usize,
    value_field_off: usize,
    byte_order: ByteOrder,
) -> Option<MetadataValue> {
    let tag_type = TagType::from_u16(type_id)?;

    match tag_type {
        TagType::Ascii => {
            // Inline if total size <= 4, else value field contains offset
            let byte_count = count;
            let val_off = if byte_count <= 4 {
                value_field_off
            } else {
                read_u32_raw(data, value_field_off, byte_order)? as usize
            };
            if val_off + byte_count > data.len() {
                return None;
            }
            let raw = &data[val_off..val_off + byte_count];
            // Strip null terminators
            let s: Vec<u8> = raw.iter().copied().take_while(|&b| b != 0).collect();
            let text = String::from_utf8(s).ok()?;
            Some(MetadataValue::Text(text))
        }
        TagType::Short => {
            // 1 Short = 2 bytes; if count == 1 the value is inline in the first 2 bytes
            if count == 1 {
                let v = read_u16_raw(data, value_field_off, byte_order)?;
                Some(MetadataValue::Integer(i64::from(v)))
            } else {
                // Multiple shorts — return the first one
                let v = read_u16_raw(data, value_field_off, byte_order)?;
                Some(MetadataValue::Integer(i64::from(v)))
            }
        }
        TagType::Long => {
            if count == 1 {
                let v = read_u32_raw(data, value_field_off, byte_order)?;
                Some(MetadataValue::Integer(i64::from(v)))
            } else {
                let v = read_u32_raw(data, value_field_off, byte_order)?;
                Some(MetadataValue::Integer(i64::from(v)))
            }
        }
        TagType::Rational => {
            // Rational = two u32 (numerator / denominator). Always at offset.
            let off = read_u32_raw(data, value_field_off, byte_order)? as usize;
            if off + 8 > data.len() {
                return None;
            }
            let num = read_u32_raw(data, off, byte_order)? as f64;
            let den = read_u32_raw(data, off + 4, byte_order)? as f64;
            if den == 0.0 {
                return None;
            }
            Some(MetadataValue::Float(num / den))
        }
        TagType::Byte | TagType::Undefined => {
            let byte_count = count;
            let val_off = if byte_count <= 4 {
                value_field_off
            } else {
                read_u32_raw(data, value_field_off, byte_order)? as usize
            };
            if val_off + byte_count > data.len() {
                return None;
            }
            Some(MetadataValue::Binary(
                data[val_off..val_off + byte_count].to_vec(),
            ))
        }
        TagType::SLong | TagType::SRational => {
            // Treat signed long as integer
            if count == 1 {
                let v = read_u32_raw(data, value_field_off, byte_order)? as i32;
                Some(MetadataValue::Integer(i64::from(v)))
            } else {
                None
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Manufacturer tag name tables
// ─────────────────────────────────────────────────────────────────────────────

fn canon_tag_name(tag: u16) -> String {
    let name = match tag {
        0x0001 => "Canon.CameraSettings",
        0x0002 => "Canon.FocalLength",
        0x0003 => "Canon.FlashInfo",
        0x0004 => "Canon.ShotInfo",
        0x0005 => "Canon.Panorama",
        0x0006 => "Canon.ImageType",
        0x0007 => "Canon.FirmwareVersion",
        0x0008 => "Canon.FileNumber",
        0x0009 => "Canon.OwnerName",
        0x000C => "Canon.SerialNumber",
        0x000D => "Canon.CameraInfo",
        0x000E => "Canon.FileLength",
        0x000F => "Canon.CustomFunctions",
        0x0010 => "Canon.ModelID",
        0x0012 => "Canon.PictureInfo",
        0x0013 => "Canon.ThumbnailImageValidArea",
        0x0015 => "Canon.SerialNumberFormat",
        0x001A => "Canon.SuperMacro",
        0x001C => "Canon.DateStampMode",
        0x001D => "Canon.MyColors",
        0x001E => "Canon.FirmwareRevision",
        0x0023 => "Canon.Categories",
        0x0026 => "Canon.FaceDetect1",
        0x0027 => "Canon.FaceDetect2",
        0x0028 => "Canon.AFInfo",
        0x0029 => "Canon.ContrastInfo",
        0x002F => "Canon.FaceDetect3",
        0x0035 => "Canon.TimeInfo",
        0x0038 => "Canon.BatteryType",
        0x003C => "Canon.AFInfo2",
        0x0081 => "Canon.RawDataOffset",
        0x0083 => "Canon.OriginalDecisionDataOffset",
        0x0090 => "Canon.CustomFunctions1D",
        0x0091 => "Canon.PersonalFunctions",
        0x0092 => "Canon.PersonalFunctionValues",
        0x0093 => "Canon.FileInfo",
        0x0094 => "Canon.AFPointsInFocus1D",
        0x0095 => "Canon.LensModel",
        0x0096 => "Canon.InternalSerialNumber",
        0x0097 => "Canon.DustRemovalData",
        0x0099 => "Canon.CustomFunctions2",
        0x009A => "Canon.AspectInfo",
        0x00A0 => "Canon.ProcessingInfo",
        0x00AA => "Canon.MeasuredColor",
        0x00AE => "Canon.ColorTemperature",
        0x00B0 => "Canon.CanonFlags",
        0x00B1 => "Canon.ModifiedInfo",
        0x00B2 => "Canon.ToneCurveTable",
        0x00B3 => "Canon.SharpnessTable",
        0x00B4 => "Canon.SharpnessFreqTable",
        0x00B5 => "Canon.WhiteBalanceTable",
        0x00B6 => "Canon.ColorBalance",
        0x00B7 => "Canon.MeasuredColor2",
        0x00B9 => "Canon.ColorTemp2",
        0x00D0 => "Canon.VignettingCorr",
        0x00D1 => "Canon.VignettingCorr2",
        0x00D2 => "Canon.LightingOpt",
        0x00D3 => "Canon.LensInfo",
        0x00D4 => "Canon.AmbienceInfo",
        0x00D5 => "Canon.MultiExp",
        0x00D6 => "Canon.FilterInfo",
        0x00D7 => "Canon.HDRInfo",
        0x00DA => "Canon.AFConfig",
        0x4013 => "Canon.AFMicroAdj",
        0x4015 => "Canon.VignettingCorr3",
        0x4016 => "Canon.STMacro",
        0x4018 => "Canon.LensInfo2",
        0x4019 => "Canon.WhiteBalance",
        0x4021 => "Canon.MyMenu",
        _ => "",
    };
    if name.is_empty() {
        format!("Canon.Tag_{tag:04X}")
    } else {
        name.to_string()
    }
}

fn nikon_tag_name(tag: u16) -> String {
    let name = match tag {
        0x0001 => "Nikon.MakernoteVersion",
        0x0002 => "Nikon.ISO",
        0x0003 => "Nikon.ColorMode",
        0x0004 => "Nikon.Quality",
        0x0005 => "Nikon.WhiteBalance",
        0x0006 => "Nikon.Sharpness",
        0x0007 => "Nikon.FocusMode",
        0x0008 => "Nikon.FlashSetting",
        0x0009 => "Nikon.FlashType",
        0x000B => "Nikon.WhiteBalanceFineTune",
        0x000C => "Nikon.WB_RBLevels",
        0x000D => "Nikon.ProgramShift",
        0x000E => "Nikon.ExposureDifference",
        0x000F => "Nikon.ISOSelection",
        0x0010 => "Nikon.DataDump",
        0x0011 => "Nikon.PreviewIFD",
        0x0012 => "Nikon.FlashExposureComp",
        0x0013 => "Nikon.ISOSetting",
        0x0016 => "Nikon.ImageBoundary",
        0x0017 => "Nikon.ExternalFlashExposureComp",
        0x0018 => "Nikon.FlashExposureBracketValue",
        0x0019 => "Nikon.ExposureBracketValue",
        0x001A => "Nikon.ImageProcessing",
        0x001B => "Nikon.CropHiSpeed",
        0x001C => "Nikon.ExposureTuning",
        0x001D => "Nikon.SerialNumber",
        0x001E => "Nikon.ColorSpace",
        0x001F => "Nikon.VRInfo",
        0x0020 => "Nikon.ImageAuthentication",
        0x0022 => "Nikon.ActiveD-Lighting",
        0x0023 => "Nikon.PictureControlData",
        0x0024 => "Nikon.WorldTime",
        0x0025 => "Nikon.ISOInfo",
        0x002A => "Nikon.VignetteControl",
        0x002B => "Nikon.DistortInfo",
        0x0034 => "Nikon.ShotInfoD3",
        0x0035 => "Nikon.ShotInfoD300",
        0x003D => "Nikon.Unknown0x003D",
        0x0080 => "Nikon.ImageAdjustment",
        0x0081 => "Nikon.ToneComp",
        0x0082 => "Nikon.AuxiliaryLens",
        0x0083 => "Nikon.LensType",
        0x0084 => "Nikon.Lens",
        0x0085 => "Nikon.ManualFocusDistance",
        0x0086 => "Nikon.DigitalZoom",
        0x0087 => "Nikon.FlashMode",
        0x0088 => "Nikon.AFInfo",
        0x0089 => "Nikon.ShootingMode",
        0x008A => "Nikon.AutoBracketRelease",
        0x008B => "Nikon.LensFStops",
        0x008C => "Nikon.ContrastCurve",
        0x008D => "Nikon.ColorHue",
        0x008F => "Nikon.SceneMode",
        0x0090 => "Nikon.LightSource",
        0x0091 => "Nikon.ShotInfo",
        0x0092 => "Nikon.HueAdjustment",
        0x0093 => "Nikon.NEFCompression",
        0x0094 => "Nikon.Saturation",
        0x0095 => "Nikon.NoiseReduction",
        0x0096 => "Nikon.LinearizationTable",
        0x0097 => "Nikon.ColorBalance",
        0x0098 => "Nikon.LensData",
        0x0099 => "Nikon.RawImageCenter",
        0x009A => "Nikon.SensorPixelSize",
        0x009C => "Nikon.Scene Assist",
        0x009E => "Nikon.RetouchHistory",
        0x00A0 => "Nikon.SerialNumber2",
        0x00A2 => "Nikon.ImageDataSize",
        0x00A5 => "Nikon.ImageCount",
        0x00A6 => "Nikon.DeletedImageCount",
        0x00A7 => "Nikon.ShutterCount",
        0x00A8 => "Nikon.FlashInfo",
        0x00A9 => "Nikon.ImageOptimization",
        0x00AA => "Nikon.Saturation2",
        0x00AB => "Nikon.VariProgram",
        0x00AC => "Nikon.ImageStabilization",
        0x00AD => "Nikon.AFResponse",
        0x00B0 => "Nikon.MultiExposure",
        0x00B1 => "Nikon.HighISONoiseReduction",
        0x00B3 => "Nikon.ToningEffect",
        0x00B6 => "Nikon.PowerUpTime",
        0x00B7 => "Nikon.AFInfo2",
        0x00B8 => "Nikon.FileInfo",
        0x00B9 => "Nikon.AFTune",
        0x00BD => "Nikon.PictureControlData2",
        _ => "",
    };
    if name.is_empty() {
        format!("Nikon.Tag_{tag:04X}")
    } else {
        name.to_string()
    }
}

fn sony_tag_name(tag: u16) -> String {
    let name = match tag {
        0x0102 => "Sony.Quality",
        0x0104 => "Sony.FlashExposureComp",
        0x0105 => "Sony.Teleconverter",
        0x0112 => "Sony.WhiteBalanceFineTune",
        0x0114 => "Sony.CameraSettings",
        0x0115 => "Sony.WhiteBalance",
        0x0116 => "Sony.ExtraInfo",
        0x0E00 => "Sony.PrintIM",
        0x1000 => "Sony.MultiBurstMode",
        0x1001 => "Sony.MultiBurstImageWidth",
        0x1002 => "Sony.MultiBurstImageHeight",
        0x1003 => "Sony.Panorama",
        0x2001 => "Sony.PreviewImage",
        0x2002 => "Sony.Rating",
        0x2004 => "Sony.Contrast",
        0x2005 => "Sony.Saturation",
        0x2006 => "Sony.Sharpness",
        0x2007 => "Sony.Brightness",
        0x2008 => "Sony.LongExposureNoiseReduction",
        0x2009 => "Sony.HighISONoiseReduction2",
        0x200A => "Sony.HDR",
        0x200B => "Sony.MultiFrameNoiseReduction",
        0x200E => "Sony.PictureEffect",
        0x200F => "Sony.SoftSkinEffect",
        0x2011 => "Sony.VignettingCorrection",
        0x2012 => "Sony.LateralChromaticAberration",
        0x2013 => "Sony.DistortionCorrection",
        0x2014 => "Sony.WBShiftAB_GM",
        0x2016 => "Sony.AutoPortraitFramed",
        0x201B => "Sony.FocusMode",
        0x201C => "Sony.AFAreaModeSetting",
        0x201D => "Sony.FlexibleSpotPosition",
        0x201E => "Sony.AFPointSelected",
        0x3000 => "Sony.ShotInfo",
        0xB000 => "Sony.FileFormat",
        0xB001 => "Sony.SonyModelID",
        0xB020 => "Sony.ColorReproduction",
        0xB021 => "Sony.ColorTemperature",
        0xB022 => "Sony.ColorCompensationFilter",
        0xB023 => "Sony.SceneRecognitionType",
        0xB024 => "Sony.SceneRecognitionType2",
        0xB025 => "Sony.DynamicRangeOptimizer",
        0xB026 => "Sony.ImageStabilization",
        0xB027 => "Sony.LensType",
        0xB028 => "Sony.MinoltaMakerNote",
        0xB029 => "Sony.ColorMode",
        0xB02A => "Sony.LensSpec",
        0xB02B => "Sony.FullImageSize",
        0xB02C => "Sony.PreviewImageSize",
        0xB040 => "Sony.Macro",
        0xB041 => "Sony.ExposureMode",
        0xB042 => "Sony.FocusMode2",
        0xB043 => "Sony.AFMode",
        0xB044 => "Sony.AFIlluminator",
        0xB047 => "Sony.JPEGQuality",
        0xB048 => "Sony.FlashLevel",
        0xB049 => "Sony.ReleaseMode",
        0xB04A => "Sony.SequenceNumber",
        0xB04B => "Sony.AntiBlur",
        0xB04E => "Sony.FocusMode3",
        0xB04F => "Sony.DynamicRangeOptimizer2",
        0xB052 => "Sony.IntelligentAuto",
        0xB054 => "Sony.WhiteBalance2",
        _ => "",
    };
    if name.is_empty() {
        format!("Sony.Tag_{tag:04X}")
    } else {
        name.to_string()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Low-level byte helpers (no-std-cursor variants)
// ─────────────────────────────────────────────────────────────────────────────

fn read_u16_raw(data: &[u8], off: usize, byte_order: ByteOrder) -> Option<u16> {
    if off + 2 > data.len() {
        return None;
    }
    Some(match byte_order {
        ByteOrder::LittleEndian => u16::from_le_bytes([data[off], data[off + 1]]),
        ByteOrder::BigEndian => u16::from_be_bytes([data[off], data[off + 1]]),
    })
}

fn read_u32_raw(data: &[u8], off: usize, byte_order: ByteOrder) -> Option<u32> {
    if off + 4 > data.len() {
        return None;
    }
    Some(match byte_order {
        ByteOrder::LittleEndian => {
            u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
        }
        ByteOrder::BigEndian => {
            u32::from_be_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
        }
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Public Makernote parsing API
// ─────────────────────────────────────────────────────────────────────────────

/// Parse Makernote tags from raw Makernote bytes, given the camera Make string
/// and the byte order of the parent EXIF IFD.
///
/// Returns an empty `Vec` for unsupported or malformed Makernotes.
#[must_use]
pub fn parse_makernote(
    makernote_bytes: &[u8],
    make: &str,
    byte_order: ByteOrder,
) -> Vec<MakernoteTag> {
    match MakernoteDialect::detect(make, makernote_bytes) {
        MakernoteDialect::Canon => parse_canon_makernote(makernote_bytes, byte_order),
        MakernoteDialect::Nikon => parse_nikon_makernote(makernote_bytes),
        MakernoteDialect::Sony => parse_sony_makernote(makernote_bytes, byte_order),
        MakernoteDialect::Unknown => Vec::new(),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Core EXIF parse / write
// ─────────────────────────────────────────────────────────────────────────────

/// Parse EXIF metadata from data.
///
/// Makernote tags are parsed (when the manufacturer is recognised) and
/// inserted with their manufacturer-prefixed names, e.g.
/// `"Canon.CameraSettings"`, `"Nikon.ISO"`, `"Sony.ColorMode"`.
///
/// # Errors
///
/// Returns an error if the data is not valid EXIF.
#[allow(clippy::too_many_lines)]
pub fn parse(data: &[u8]) -> Result<Metadata, Error> {
    if data.len() < 8 {
        return Err(Error::ParseError("Data too short for EXIF".to_string()));
    }

    // Determine byte order
    let byte_order = if &data[0..2] == EXIF_LE {
        ByteOrder::LittleEndian
    } else if &data[0..2] == EXIF_BE {
        ByteOrder::BigEndian
    } else {
        return Err(Error::ParseError(
            "Invalid EXIF byte order marker".to_string(),
        ));
    };

    let mut cursor = Cursor::new(data);
    cursor.set_position(2);

    // Read TIFF magic number
    let magic = read_u16(&mut cursor, byte_order)?;
    if magic != TIFF_MAGIC {
        return Err(Error::ParseError("Invalid TIFF magic number".to_string()));
    }

    // Read offset to first IFD
    let ifd_offset = read_u32(&mut cursor, byte_order)?;

    let mut metadata = Metadata::new(MetadataFormat::Exif);

    // Parse IFD — collect raw entries including Makernote
    let makernote_bytes =
        parse_ifd_with_makernote(&mut cursor, ifd_offset as u64, byte_order, &mut metadata)?;

    // If we captured a Makernote, try to parse it
    if let Some((makernote_data, make)) = makernote_bytes {
        let mn_tags = parse_makernote(&makernote_data, &make, byte_order);
        for tag in mn_tags {
            metadata.insert(tag.name, tag.value);
        }
    }

    Ok(metadata)
}

/// Parse an IFD (Image File Directory), returning any Makernote data found.
///
/// Returns `Ok(Some((makernote_bytes, make_string)))` when both Make and
/// Makernote are present, `Ok(None)` otherwise.
fn parse_ifd_with_makernote(
    cursor: &mut Cursor<&[u8]>,
    offset: u64,
    byte_order: ByteOrder,
    metadata: &mut Metadata,
) -> Result<Option<(Vec<u8>, String)>, Error> {
    cursor
        .seek(SeekFrom::Start(offset))
        .map_err(|e| Error::ParseError(format!("Failed to seek to IFD: {e}")))?;

    let entry_count = read_u16(cursor, byte_order)?;

    // We need two passes: collect all entries first to resolve offset values,
    // then handle Makernote with the accumulated Make string.
    let mut make_value: Option<String> = None;
    let mut makernote_data: Option<Vec<u8>> = None;
    let mut saved_pos = cursor.position();

    for _ in 0..entry_count {
        cursor
            .seek(SeekFrom::Start(saved_pos))
            .map_err(|e| Error::ParseError(format!("Seek error: {e}")))?;

        let tag = read_u16(cursor, byte_order)?;
        let tag_type = read_u16(cursor, byte_order)?;
        let count = read_u32(cursor, byte_order)?;
        let value_offset = read_u32(cursor, byte_order)?;

        saved_pos = cursor.position();

        if tag == TAG_MAKERNOTE {
            // Makernote is always stored at offset (Undefined type, large count)
            let data_slice = cursor.get_ref();
            let off = value_offset as usize;
            let len = count as usize;
            if off + len <= data_slice.len() {
                makernote_data = Some(data_slice[off..off + len].to_vec());
            }
            continue;
        }

        // Parse tag value
        if let Some(value) =
            parse_tag_value(cursor, tag, tag_type, count, value_offset, byte_order)?
        {
            let tag_name = get_tag_name(tag);
            if tag == TAG_MAKE {
                if let MetadataValue::Text(ref s) = value {
                    make_value = Some(s.clone());
                }
            }
            metadata.insert(tag_name, value);
        }
    }

    if let (Some(mn), Some(make)) = (makernote_data, make_value) {
        Ok(Some((mn, make)))
    } else {
        Ok(None)
    }
}

/// Parse an IFD (Image File Directory) — simple version without Makernote.
fn parse_ifd(
    cursor: &mut Cursor<&[u8]>,
    offset: u64,
    byte_order: ByteOrder,
    metadata: &mut Metadata,
) -> Result<(), Error> {
    cursor
        .seek(SeekFrom::Start(offset))
        .map_err(|e| Error::ParseError(format!("Failed to seek to IFD: {e}")))?;

    let entry_count = read_u16(cursor, byte_order)?;

    for _ in 0..entry_count {
        let tag = read_u16(cursor, byte_order)?;
        let tag_type = read_u16(cursor, byte_order)?;
        let count = read_u32(cursor, byte_order)?;
        let value_offset = read_u32(cursor, byte_order)?;

        if let Some(value) =
            parse_tag_value(cursor, tag, tag_type, count, value_offset, byte_order)?
        {
            let tag_name = get_tag_name(tag);
            metadata.insert(tag_name, value);
        }
    }

    Ok(())
}

/// Parse a tag value.
#[allow(clippy::too_many_arguments)]
fn parse_tag_value(
    cursor: &mut Cursor<&[u8]>,
    _tag: u16,
    tag_type: u16,
    count: u32,
    value_offset: u32,
    _byte_order: ByteOrder,
) -> Result<Option<MetadataValue>, Error> {
    let tag_type = TagType::from_u16(tag_type);

    match tag_type {
        Some(TagType::Ascii) => {
            // ASCII string
            let current_pos = cursor.position();

            let value_size = count;
            let value_pos = if value_size <= 4 {
                current_pos - 4 // Inline value
            } else {
                value_offset as u64 // Value at offset
            };

            cursor
                .seek(SeekFrom::Start(value_pos))
                .map_err(|e| Error::ParseError(format!("Failed to seek to value: {e}")))?;

            let mut value_bytes = vec![0u8; count as usize];
            cursor
                .read_exact(&mut value_bytes)
                .map_err(|e| Error::ParseError(format!("Failed to read value: {e}")))?;

            while let Some(&0) = value_bytes.last() {
                value_bytes.pop();
            }

            let text = String::from_utf8(value_bytes)
                .map_err(|e| Error::EncodingError(format!("Invalid UTF-8 in EXIF tag: {e}")))?;

            cursor.seek(SeekFrom::Start(current_pos))?;

            Ok(Some(MetadataValue::Text(text)))
        }
        Some(TagType::Short) => {
            let value = if count == 1 {
                u32::from(value_offset >> 16)
            } else {
                value_offset
            };
            Ok(Some(MetadataValue::Integer(i64::from(value))))
        }
        Some(TagType::Long) => Ok(Some(MetadataValue::Integer(i64::from(value_offset)))),
        _ => Ok(None),
    }
}

/// Get tag name from tag ID.
fn get_tag_name(tag: u16) -> String {
    match tag {
        0x010F => "Make".to_string(),
        0x0110 => "Model".to_string(),
        0x0132 => "DateTime".to_string(),
        0x013B => "Artist".to_string(),
        0x8298 => "Copyright".to_string(),
        0x9003 => "DateTimeOriginal".to_string(),
        0x9004 => "DateTimeDigitized".to_string(),
        0x010E => "ImageDescription".to_string(),
        0x0131 => "Software".to_string(),
        0x927C => "Makernote".to_string(),
        _ => format!("Tag_{tag:04X}"),
    }
}

/// Write EXIF metadata to data.
///
/// # Errors
///
/// Returns an error if writing fails.
pub fn write(metadata: &Metadata) -> Result<Vec<u8>, Error> {
    let mut result = Vec::new();

    result.extend_from_slice(EXIF_LE);
    result.extend_from_slice(&TIFF_MAGIC.to_le_bytes());
    result.extend_from_slice(&8_u32.to_le_bytes());

    let text_fields: Vec<_> = metadata
        .fields()
        .iter()
        .filter(|(_, v)| matches!(v, MetadataValue::Text(_)))
        .collect();

    result.extend_from_slice(&(text_fields.len() as u16).to_le_bytes());

    let mut value_offset = 8 + 2 + (text_fields.len() * 12) + 4;

    for (key, value) in &text_fields {
        let tag = get_tag_id(key);
        let text = value.as_text().unwrap_or("");

        result.extend_from_slice(&tag.to_le_bytes());
        result.extend_from_slice(&(TagType::Ascii as u16).to_le_bytes());

        let count = text.len() + 1;
        result.extend_from_slice(&(count as u32).to_le_bytes());

        if count <= 4 {
            let mut inline_value = [0u8; 4];
            inline_value[..text.len()].copy_from_slice(text.as_bytes());
            result.extend_from_slice(&inline_value);
        } else {
            result.extend_from_slice(&(value_offset as u32).to_le_bytes());
            value_offset += count;
        }
    }

    result.extend_from_slice(&0_u32.to_le_bytes());

    for (_, value) in &text_fields {
        let text = value.as_text().unwrap_or("");
        if text.len() + 1 > 4 {
            result.extend_from_slice(text.as_bytes());
            result.push(0);
        }
    }

    Ok(result)
}

/// Get tag ID from tag name.
fn get_tag_id(name: &str) -> u16 {
    match name {
        "Make" => 0x010F,
        "Model" => 0x0110,
        "DateTime" => 0x0132,
        "Artist" => 0x013B,
        "Copyright" => 0x8298,
        "DateTimeOriginal" => 0x9003,
        "DateTimeDigitized" => 0x9004,
        "ImageDescription" => 0x010E,
        "Software" => 0x0131,
        _ => 0xFFFF,
    }
}

/// Read a 16-bit unsigned integer.
fn read_u16(cursor: &mut Cursor<&[u8]>, byte_order: ByteOrder) -> Result<u16, Error> {
    let mut bytes = [0u8; 2];
    cursor
        .read_exact(&mut bytes)
        .map_err(|e| Error::ParseError(format!("Failed to read u16: {e}")))?;

    Ok(match byte_order {
        ByteOrder::LittleEndian => u16::from_le_bytes(bytes),
        ByteOrder::BigEndian => u16::from_be_bytes(bytes),
    })
}

/// Read a 32-bit unsigned integer.
fn read_u32(cursor: &mut Cursor<&[u8]>, byte_order: ByteOrder) -> Result<u32, Error> {
    let mut bytes = [0u8; 4];
    cursor
        .read_exact(&mut bytes)
        .map_err(|e| Error::ParseError(format!("Failed to read u32: {e}")))?;

    Ok(match byte_order {
        ByteOrder::LittleEndian => u32::from_le_bytes(bytes),
        ByteOrder::BigEndian => u32::from_be_bytes(bytes),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exif_round_trip() {
        let mut metadata = Metadata::new(MetadataFormat::Exif);

        metadata.insert(
            "Artist".to_string(),
            MetadataValue::Text("Test Artist".to_string()),
        );
        metadata.insert(
            "Copyright".to_string(),
            MetadataValue::Text("Copyright 2024".to_string()),
        );

        let data = write(&metadata).expect("Write failed");
        let parsed = parse(&data).expect("Parse failed");

        assert_eq!(
            parsed.get("Artist").and_then(|v| v.as_text()),
            Some("Test Artist")
        );
        assert_eq!(
            parsed.get("Copyright").and_then(|v| v.as_text()),
            Some("Copyright 2024")
        );
    }

    #[test]
    fn test_get_tag_name() {
        assert_eq!(get_tag_name(0x010F), "Make");
        assert_eq!(get_tag_name(0x0110), "Model");
        assert_eq!(get_tag_name(0x013B), "Artist");
        assert_eq!(get_tag_name(0x8298), "Copyright");
        assert_eq!(get_tag_name(0x927C), "Makernote");
    }

    #[test]
    fn test_get_tag_id() {
        assert_eq!(get_tag_id("Make"), 0x010F);
        assert_eq!(get_tag_id("Model"), 0x0110);
        assert_eq!(get_tag_id("Artist"), 0x013B);
        assert_eq!(get_tag_id("Copyright"), 0x8298);
    }

    // ─── Makernote dialect detection ───────────────────────────────────────

    #[test]
    fn test_makernote_dialect_detect_canon() {
        let d = MakernoteDialect::detect("Canon EOS R5", &[]);
        assert_eq!(d, MakernoteDialect::Canon);
    }

    #[test]
    fn test_makernote_dialect_detect_nikon() {
        let header = b"Nikon\0\x02\x10\x00\x00";
        let d = MakernoteDialect::detect("NIKON CORPORATION", header);
        assert_eq!(d, MakernoteDialect::Nikon);
    }

    #[test]
    fn test_makernote_dialect_detect_sony() {
        let d = MakernoteDialect::detect("SONY", &[]);
        assert_eq!(d, MakernoteDialect::Sony);
    }

    #[test]
    fn test_makernote_dialect_detect_unknown() {
        let d = MakernoteDialect::detect("Fujifilm", &[]);
        assert_eq!(d, MakernoteDialect::Unknown);
    }

    // ─── Canon Makernote parsing ───────────────────────────────────────────

    #[test]
    fn test_canon_makernote_minimal_empty() {
        // A Canon Makernote with 0 entries: just a 2-byte count
        let data: &[u8] = &[0x00, 0x00]; // LE count = 0
        let tags = parse_canon_makernote(data, ByteOrder::LittleEndian);
        assert!(tags.is_empty());
    }

    #[test]
    fn test_canon_makernote_one_ascii_entry_inline() {
        // Build a synthetic Canon Makernote (LE) with one ASCII entry (≤4 bytes)
        // Count(2) + Entry(12):
        //   tag(2) = 0x0006 (ImageType), type(2) = 2 (ASCII), count(4) = 3,
        //   value(4) = b"SL\0\0" (inline)
        let mut data = Vec::new();
        data.extend_from_slice(&1u16.to_le_bytes()); // count
        data.extend_from_slice(&0x0006u16.to_le_bytes()); // tag: ImageType
        data.extend_from_slice(&2u16.to_le_bytes()); // type: ASCII
        data.extend_from_slice(&3u32.to_le_bytes()); // count = 3 chars incl. null
        data.extend_from_slice(b"SL\0\0"); // inline value (4 bytes)

        let tags = parse_canon_makernote(&data, ByteOrder::LittleEndian);
        assert_eq!(tags.len(), 1);
        assert_eq!(tags[0].tag_id, 0x0006);
        assert_eq!(tags[0].name, "Canon.ImageType");
        // "SL" without null terminator
        assert_eq!(tags[0].value, MetadataValue::Text("SL".to_string()));
    }

    #[test]
    fn test_canon_makernote_short_entry() {
        // Build a Canon Makernote with one Short entry (LE)
        let mut data = Vec::new();
        data.extend_from_slice(&1u16.to_le_bytes()); // count
        data.extend_from_slice(&0x000Cu16.to_le_bytes()); // tag: SerialNumber
        data.extend_from_slice(&3u16.to_le_bytes()); // type: Short
        data.extend_from_slice(&1u32.to_le_bytes()); // count = 1
                                                     // For Short with count=1, value is stored at value_field_off
                                                     // value_field_off in our synthetic data = 2 + 12 (past the count + entry)
                                                     // Actually in decode_ifd_entry_value for Short count==1:
                                                     // read_u16_raw(data, value_field_off, ...) where value_field_off = entry_off + 8
                                                     // entry_off = 2, so value_field_off = 10
                                                     // We need data[10..12] = 0x0042
        data.extend_from_slice(&0x0042u16.to_le_bytes()); // inline short value = 0x0042
        data.extend_from_slice(&0x0000u16.to_le_bytes()); // padding

        let tags = parse_canon_makernote(&data, ByteOrder::LittleEndian);
        assert_eq!(tags.len(), 1);
        assert_eq!(tags[0].tag_id, 0x000C);
        assert_eq!(tags[0].name, "Canon.SerialNumber");
        assert_eq!(tags[0].value, MetadataValue::Integer(0x0042));
    }

    // ─── Nikon Makernote parsing ───────────────────────────────────────────

    #[test]
    fn test_nikon_makernote_too_short() {
        let tags = parse_nikon_makernote(b"Nikon\0");
        assert!(tags.is_empty());
    }

    #[test]
    fn test_nikon_makernote_bad_header() {
        let data = vec![0u8; 30];
        // No "Nikon\0" header
        let tags = parse_nikon_makernote(&data);
        assert!(tags.is_empty());
    }

    #[test]
    fn test_nikon_makernote_minimal_zero_entries() {
        // "Nikon\0" + 4 bytes padding + LE TIFF header + IFD with 0 entries
        // tiff_start = 10
        // tiff_data: b"II" + 42(2) + offset=8(4) + count=0(2)
        let mut data = Vec::new();
        data.extend_from_slice(b"Nikon\0"); // 6 bytes
        data.extend_from_slice(&[0x02, 0x10, 0x00, 0x00]); // 4 bytes padding/version
                                                           // mini-TIFF: LE byte order + magic + IFD offset
        data.extend_from_slice(b"II"); // 2 bytes
        data.extend_from_slice(&42u16.to_le_bytes()); // 2 bytes magic
        data.extend_from_slice(&8u32.to_le_bytes()); // 4 bytes: IFD at offset 8
                                                     // IFD: count = 0
        data.extend_from_slice(&0u16.to_le_bytes()); // 2 bytes

        let tags = parse_nikon_makernote(&data);
        assert!(tags.is_empty());
    }

    // ─── Sony Makernote parsing ────────────────────────────────────────────

    #[test]
    fn test_sony_makernote_empty_data() {
        let tags = parse_sony_makernote(&[], ByteOrder::LittleEndian);
        assert!(tags.is_empty());
    }

    #[test]
    fn test_sony_makernote_zero_entries_no_header() {
        let data: &[u8] = &[0x00, 0x00]; // count = 0, no Sony header
        let tags = parse_sony_makernote(data, ByteOrder::LittleEndian);
        assert!(tags.is_empty());
    }

    #[test]
    fn test_sony_makernote_zero_entries_with_header() {
        // "SONY DSC \0\0\0" (12 bytes) + count=0 (2 bytes)
        let mut data = Vec::new();
        data.extend_from_slice(b"SONY DSC \0\0\0"); // 12 bytes
        data.extend_from_slice(&0u16.to_le_bytes()); // count = 0
        let tags = parse_sony_makernote(&data, ByteOrder::LittleEndian);
        assert!(tags.is_empty());
    }

    // ─── Tag name tables ───────────────────────────────────────────────────

    #[test]
    fn test_canon_tag_name_known() {
        assert_eq!(canon_tag_name(0x0006), "Canon.ImageType");
        assert_eq!(canon_tag_name(0x000C), "Canon.SerialNumber");
        assert_eq!(canon_tag_name(0x0095), "Canon.LensModel");
    }

    #[test]
    fn test_canon_tag_name_unknown() {
        let n = canon_tag_name(0xDEAD);
        assert!(n.starts_with("Canon.Tag_"));
    }

    #[test]
    fn test_nikon_tag_name_known() {
        assert_eq!(nikon_tag_name(0x001D), "Nikon.SerialNumber");
        assert_eq!(nikon_tag_name(0x0084), "Nikon.Lens");
    }

    #[test]
    fn test_nikon_tag_name_unknown() {
        let n = nikon_tag_name(0xDEAD);
        assert!(n.starts_with("Nikon.Tag_"));
    }

    #[test]
    fn test_sony_tag_name_known() {
        assert_eq!(sony_tag_name(0xB027), "Sony.LensType");
        assert_eq!(sony_tag_name(0x0102), "Sony.Quality");
    }

    #[test]
    fn test_sony_tag_name_unknown() {
        let n = sony_tag_name(0xDEAD);
        assert!(n.starts_with("Sony.Tag_"));
    }

    // ─── parse_makernote public API ────────────────────────────────────────

    #[test]
    fn test_parse_makernote_unknown_make_returns_empty() {
        let result = parse_makernote(b"\x00\x00", "Unknown Maker", ByteOrder::LittleEndian);
        assert!(result.is_empty());
    }

    #[test]
    fn test_parse_makernote_canon_dispatches() {
        // A Canon Makernote with 0 entries should return empty vec (not panic)
        let data: &[u8] = &[0x00, 0x00];
        let result = parse_makernote(data, "Canon EOS 5D", ByteOrder::LittleEndian);
        assert!(result.is_empty());
    }

    // ─── decode_ifd_entry_value ────────────────────────────────────────────

    #[test]
    fn test_decode_rational_value() {
        // Rational at offset 0 of a synthetic buffer: num=1, den=100 (for 1/100s)
        // The value_field_off points to a 4-byte offset, then at that offset: num(4)+den(4)
        // We'll arrange: value_field_off=0 holds offset=4, then at 4: num=1, den=100
        let mut buf = Vec::new();
        buf.extend_from_slice(&4u32.to_le_bytes()); // offset to rational data
        buf.extend_from_slice(&1u32.to_le_bytes()); // numerator
        buf.extend_from_slice(&100u32.to_le_bytes()); // denominator

        let result =
            decode_ifd_entry_value(&buf, 5 /* Rational */, 1, 0, ByteOrder::LittleEndian);
        assert!(result.is_some());
        if let Some(MetadataValue::Float(f)) = result {
            assert!((f - 0.01).abs() < 1e-9);
        } else {
            panic!("Expected Float");
        }
    }

    #[test]
    fn test_decode_rational_zero_denominator() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&4u32.to_le_bytes()); // offset
        buf.extend_from_slice(&1u32.to_le_bytes()); // numerator
        buf.extend_from_slice(&0u32.to_le_bytes()); // denominator = 0

        let result = decode_ifd_entry_value(&buf, 5, 1, 0, ByteOrder::LittleEndian);
        assert!(result.is_none());
    }

    #[test]
    fn test_read_u16_raw_little_endian() {
        let data = [0x34, 0x12];
        assert_eq!(
            read_u16_raw(&data, 0, ByteOrder::LittleEndian),
            Some(0x1234)
        );
    }

    #[test]
    fn test_read_u16_raw_big_endian() {
        let data = [0x12, 0x34];
        assert_eq!(read_u16_raw(&data, 0, ByteOrder::BigEndian), Some(0x1234));
    }

    #[test]
    fn test_read_u16_raw_out_of_bounds() {
        let data = [0x12];
        assert_eq!(read_u16_raw(&data, 0, ByteOrder::LittleEndian), None);
    }

    #[test]
    fn test_read_u32_raw_little_endian() {
        let data = [0x78, 0x56, 0x34, 0x12];
        assert_eq!(
            read_u32_raw(&data, 0, ByteOrder::LittleEndian),
            Some(0x12345678)
        );
    }
}
