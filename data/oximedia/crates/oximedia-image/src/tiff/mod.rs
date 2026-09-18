//! TIFF (Tagged Image File Format) support.
//!
//! Implements TIFF format with comprehensive compression and color space support.
//!
//! # Features
//!
//! - Multiple compression methods (None, LZW, `PackBits`, Deflate, JPEG)
//! - Photometric interpretations (RGB, CMYK, YCbCr, CIE Lab)
//! - All bit depths (1-64 bit)
//! - Multi-page TIFF
//! - `BigTIFF` (64-bit offsets)
//! - Tiled TIFF
//!
//! # Example
//!
//! ```no_run
//! use oximedia_image::tiff;
//! use std::path::Path;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let frame = tiff::read_tiff(Path::new("image.tif"), 1)?;
//! println!("TIFF frame: {}x{}", frame.width, frame.height);
//! # Ok(())
//! # }
//! ```

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::similar_names)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::field_reassign_with_default)]

use crate::error::{ImageError, ImageResult};
use crate::{ColorSpace, Endian, ImageData, ImageFrame, PixelType};
use byteorder::{BigEndian, LittleEndian, ReadBytesExt, WriteBytesExt};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

mod ccitt;
mod jpeg_compression;

/// TIFF magic number (big-endian).
const TIFF_MAGIC_BE: u16 = 0x4D4D; // "MM"

/// TIFF magic number (little-endian).
const TIFF_MAGIC_LE: u16 = 0x4949; // "II"

/// TIFF version (42).
const TIFF_VERSION: u16 = 42;

/// `BigTIFF` version (43).
const BIGTIFF_VERSION: u16 = 43;

/// TIFF tag types.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TiffTag {
    /// Image width.
    ImageWidth = 256,
    /// Image height.
    ImageLength = 257,
    /// Bits per sample.
    BitsPerSample = 258,
    /// Compression method.
    Compression = 259,
    /// Photometric interpretation.
    PhotometricInterpretation = 262,
    /// Image description.
    ImageDescription = 270,
    /// Strip offsets.
    StripOffsets = 273,
    /// Samples per pixel.
    SamplesPerPixel = 277,
    /// Rows per strip.
    RowsPerStrip = 278,
    /// Strip byte counts.
    StripByteCounts = 279,
    /// X resolution.
    XResolution = 282,
    /// Y resolution.
    YResolution = 283,
    /// Planar configuration.
    PlanarConfiguration = 284,
    /// Resolution unit.
    ResolutionUnit = 296,
    /// Software.
    Software = 305,
    /// `DateTime`.
    DateTime = 306,
    /// Predictor.
    Predictor = 317,
    /// Tile width.
    TileWidth = 322,
    /// Tile length.
    TileLength = 323,
    /// Tile offsets.
    TileOffsets = 324,
    /// Tile byte counts.
    TileByteCounts = 325,
    /// Sample format.
    SampleFormat = 339,
    /// Extra samples.
    ExtraSamples = 338,
}

/// TIFF data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TiffDataType {
    /// Unsigned 8-bit.
    Byte = 1,
    /// ASCII string.
    Ascii = 2,
    /// Unsigned 16-bit.
    Short = 3,
    /// Unsigned 32-bit.
    Long = 4,
    /// Rational (two longs).
    Rational = 5,
    /// Signed 8-bit.
    SByte = 6,
    /// Undefined.
    Undefined = 7,
    /// Signed 16-bit.
    SShort = 8,
    /// Signed 32-bit.
    SLong = 9,
    /// Signed rational.
    SRational = 10,
    /// 32-bit float.
    Float = 11,
    /// 64-bit double.
    Double = 12,
}

impl TiffDataType {
    fn from_u16(value: u16) -> ImageResult<Self> {
        match value {
            1 => Ok(Self::Byte),
            2 => Ok(Self::Ascii),
            3 => Ok(Self::Short),
            4 => Ok(Self::Long),
            5 => Ok(Self::Rational),
            6 => Ok(Self::SByte),
            7 => Ok(Self::Undefined),
            8 => Ok(Self::SShort),
            9 => Ok(Self::SLong),
            10 => Ok(Self::SRational),
            11 => Ok(Self::Float),
            12 => Ok(Self::Double),
            _ => Err(ImageError::invalid_format("Invalid TIFF data type")),
        }
    }

    const fn size(&self) -> usize {
        match self {
            Self::Byte | Self::SByte | Self::Ascii | Self::Undefined => 1,
            Self::Short | Self::SShort => 2,
            Self::Long | Self::SLong | Self::Float => 4,
            Self::Rational | Self::SRational | Self::Double => 8,
        }
    }
}

/// TIFF compression types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TiffCompression {
    /// No compression.
    None = 1,
    /// CCITT modified Huffman RLE.
    CcittRle = 2,
    /// CCITT Group 3 fax.
    CcittFax3 = 3,
    /// CCITT Group 4 fax.
    CcittFax4 = 4,
    /// LZW compression.
    Lzw = 5,
    /// JPEG (old-style).
    JpegOld = 6,
    /// JPEG.
    Jpeg = 7,
    /// Deflate (Adobe).
    DeflateAdobe = 8,
    /// Deflate (PKZIP).
    Deflate = 32946,
    /// `PackBits` (Macintosh RLE).
    PackBits = 32773,
}

impl TiffCompression {
    fn from_u16(value: u16) -> ImageResult<Self> {
        match value {
            1 => Ok(Self::None),
            2 => Ok(Self::CcittRle),
            3 => Ok(Self::CcittFax3),
            4 => Ok(Self::CcittFax4),
            5 => Ok(Self::Lzw),
            6 => Ok(Self::JpegOld),
            7 => Ok(Self::Jpeg),
            8 => Ok(Self::DeflateAdobe),
            32946 => Ok(Self::Deflate),
            32773 => Ok(Self::PackBits),
            _ => Err(ImageError::unsupported(format!(
                "TIFF compression: {value}"
            ))),
        }
    }
}

/// Photometric interpretation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhotometricInterpretation {
    /// Min is white.
    WhiteIsZero = 0,
    /// Min is black.
    BlackIsZero = 1,
    /// RGB.
    Rgb = 2,
    /// Palette color.
    Palette = 3,
    /// Transparency mask.
    TransparencyMask = 4,
    /// CMYK.
    Cmyk = 5,
    /// YCbCr.
    YCbCr = 6,
    /// CIE Lab.
    CieLab = 8,
}

impl PhotometricInterpretation {
    fn from_u16(value: u16) -> ImageResult<Self> {
        match value {
            0 => Ok(Self::WhiteIsZero),
            1 => Ok(Self::BlackIsZero),
            2 => Ok(Self::Rgb),
            3 => Ok(Self::Palette),
            4 => Ok(Self::TransparencyMask),
            5 => Ok(Self::Cmyk),
            6 => Ok(Self::YCbCr),
            8 => Ok(Self::CieLab),
            _ => Err(ImageError::invalid_format(
                "Invalid photometric interpretation",
            )),
        }
    }
}

/// TIFF IFD (Image File Directory) entry.
#[derive(Debug, Clone)]
pub struct IfdEntry {
    /// Tag identifier.
    pub tag: u16,
    /// Data type.
    pub data_type: TiffDataType,
    /// Number of values.
    pub count: u32,
    /// Value or offset.
    pub value_offset: u32,
}

/// TIFF image information.
#[derive(Debug, Clone)]
pub struct TiffInfo {
    /// Image width.
    pub width: u32,
    /// Image height.
    pub height: u32,
    /// Bits per sample.
    pub bits_per_sample: Vec<u16>,
    /// Compression method.
    pub compression: TiffCompression,
    /// Photometric interpretation.
    pub photometric: PhotometricInterpretation,
    /// Samples per pixel.
    pub samples_per_pixel: u16,
    /// Rows per strip.
    pub rows_per_strip: u32,
    /// Strip offsets.
    pub strip_offsets: Vec<u32>,
    /// Strip byte counts.
    pub strip_byte_counts: Vec<u32>,
    /// Planar configuration (1 = chunky, 2 = planar).
    pub planar_config: u16,
    /// Sample format (1 = uint, 2 = int, 3 = float).
    pub sample_format: Vec<u16>,
    /// Is tiled.
    pub is_tiled: bool,
    /// Tile width.
    pub tile_width: u32,
    /// Tile height.
    pub tile_height: u32,
    /// Tile offsets.
    pub tile_offsets: Vec<u32>,
    /// Tile byte counts.
    pub tile_byte_counts: Vec<u32>,
    /// `JPEGTables` (tag 347): abbreviated JPEG datastream with shared
    /// quantization/Huffman tables, used by "new-style" JPEG compression.
    pub jpeg_tables: Option<Vec<u8>>,
    /// JPEG encode quality (1-100), used only on the write path.
    pub jpeg_quality: u8,
    /// Metadata.
    pub metadata: HashMap<String, String>,
    /// When `true`, IFD tags not consumed by the standard reader are preserved
    /// in `extra_ifd_entries` and re-emitted verbatim on the write path.
    pub preserve_unknown_tags: bool,
    /// IFD entries for non-standard (unknown) tags, keyed by tag number.
    /// Populated during reading when `preserve_unknown_tags` is `true`.
    /// Written to the IFD after all standard tags on the write path.
    pub extra_ifd_entries: HashMap<u16, IfdEntry>,
}

impl Default for TiffInfo {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            bits_per_sample: vec![8],
            compression: TiffCompression::None,
            photometric: PhotometricInterpretation::Rgb,
            samples_per_pixel: 1,
            rows_per_strip: 0,
            strip_offsets: Vec::new(),
            strip_byte_counts: Vec::new(),
            planar_config: 1,
            sample_format: vec![1],
            is_tiled: false,
            tile_width: 0,
            tile_height: 0,
            tile_offsets: Vec::new(),
            tile_byte_counts: Vec::new(),
            jpeg_tables: None,
            jpeg_quality: 90,
            metadata: HashMap::new(),
            preserve_unknown_tags: true,
            extra_ifd_entries: HashMap::new(),
        }
    }
}

impl TiffInfo {
    /// Insert a custom IFD entry, overwriting any existing entry with the same tag.
    pub fn set_tag(&mut self, tag: u16, entry: IfdEntry) {
        self.extra_ifd_entries.insert(tag, entry);
    }

    /// Retrieve a reference to a custom IFD entry by tag number.
    #[must_use]
    pub fn get_tag(&self, tag: u16) -> Option<&IfdEntry> {
        self.extra_ifd_entries.get(&tag)
    }
}

/// Reads a TIFF file.
///
/// # Arguments
///
/// * `path` - Path to the TIFF file
/// * `frame_number` - Frame number for the image
///
/// # Errors
///
/// Returns an error if the file cannot be read or is invalid.
pub fn read_tiff(path: &Path, frame_number: u32) -> ImageResult<ImageFrame> {
    let mut file = File::open(path)?;

    // Read and validate header
    let byte_order = file.read_u16::<BigEndian>()?;
    let endian = match byte_order {
        TIFF_MAGIC_BE => Endian::Big,
        TIFF_MAGIC_LE => Endian::Little,
        _ => return Err(ImageError::invalid_format("Invalid TIFF magic number")),
    };

    // Read version
    let version = match endian {
        Endian::Big => file.read_u16::<BigEndian>()?,
        Endian::Little => file.read_u16::<LittleEndian>()?,
    };

    if version != TIFF_VERSION && version != BIGTIFF_VERSION {
        return Err(ImageError::invalid_format("Invalid TIFF version"));
    }

    let is_bigtiff = version == BIGTIFF_VERSION;

    // Read first IFD offset
    let ifd_offset = if is_bigtiff {
        match endian {
            Endian::Big => file.read_u64::<BigEndian>()?,
            Endian::Little => file.read_u64::<LittleEndian>()?,
        }
    } else {
        match endian {
            Endian::Big => u64::from(file.read_u32::<BigEndian>()?),
            Endian::Little => u64::from(file.read_u32::<LittleEndian>()?),
        }
    };

    // Read IFD
    let info = read_ifd(&mut file, ifd_offset, endian)?;

    // Determine pixel type and color space
    let pixel_type = determine_pixel_type(&info)?;
    let color_space = determine_color_space(&info);
    let components = info.samples_per_pixel as u8;

    // Read image data
    let data = if info.is_tiled {
        read_tiled_image(&mut file, &info, endian)?
    } else {
        read_stripped_image(&mut file, &info, endian)?
    };

    let mut frame = ImageFrame::new(
        frame_number,
        info.width,
        info.height,
        pixel_type,
        components,
        color_space,
        ImageData::interleaved(data),
    );

    // Add metadata
    for (key, value) in &info.metadata {
        frame.add_metadata(key.clone(), value.clone());
    }
    frame.add_metadata("compression".to_string(), format!("{:?}", info.compression));
    frame.add_metadata("photometric".to_string(), format!("{:?}", info.photometric));

    Ok(frame)
}

/// Writes a TIFF file.
///
/// # Arguments
///
/// * `path` - Output path
/// * `frame` - Image frame to write
/// * `compression` - Compression method
///
/// # Errors
///
/// Returns an error if the file cannot be written.
pub fn write_tiff(
    path: &Path,
    frame: &ImageFrame,
    compression: TiffCompression,
) -> ImageResult<()> {
    let mut file = File::create(path)?;
    let endian = Endian::native();

    // Write header
    match endian {
        Endian::Big => {
            file.write_u16::<BigEndian>(TIFF_MAGIC_BE)?;
            file.write_u16::<BigEndian>(TIFF_VERSION)?;
            file.write_u32::<BigEndian>(8)?; // IFD offset
        }
        Endian::Little => {
            file.write_u16::<LittleEndian>(TIFF_MAGIC_LE)?;
            file.write_u16::<LittleEndian>(TIFF_VERSION)?;
            file.write_u32::<LittleEndian>(8)?; // IFD offset
        }
    }

    // Create TIFF info
    let info = create_tiff_info(frame, compression)?;

    // Write image data first to get offsets
    let data_offset = file.stream_position()? as u32;
    let Some(data) = frame.data.as_slice() else {
        return Err(ImageError::unsupported(
            "Planar data not supported for TIFF yet",
        ));
    };

    let compressed_data = compress_image_data(data, &info)?;
    file.write_all(&compressed_data)?;

    // Write IFD
    write_ifd(
        &mut file,
        &info,
        data_offset,
        compressed_data.len() as u32,
        endian,
    )?;

    Ok(())
}

fn read_ifd(file: &mut File, offset: u64, endian: Endian) -> ImageResult<TiffInfo> {
    file.seek(SeekFrom::Start(offset))?;

    let entry_count = match endian {
        Endian::Big => file.read_u16::<BigEndian>()?,
        Endian::Little => file.read_u16::<LittleEndian>()?,
    } as usize;

    let mut info = TiffInfo::default();

    for _ in 0..entry_count {
        let tag = match endian {
            Endian::Big => file.read_u16::<BigEndian>()?,
            Endian::Little => file.read_u16::<LittleEndian>()?,
        };

        let type_val = match endian {
            Endian::Big => file.read_u16::<BigEndian>()?,
            Endian::Little => file.read_u16::<LittleEndian>()?,
        };
        let data_type = TiffDataType::from_u16(type_val)?;

        let count = match endian {
            Endian::Big => file.read_u32::<BigEndian>()?,
            Endian::Little => file.read_u32::<LittleEndian>()?,
        };

        let value_offset = match endian {
            Endian::Big => file.read_u32::<BigEndian>()?,
            Endian::Little => file.read_u32::<LittleEndian>()?,
        };

        // Parse tag
        match tag {
            256 => info.width = read_tag_value_u32(file, data_type, count, value_offset, endian)?,
            257 => info.height = read_tag_value_u32(file, data_type, count, value_offset, endian)?,
            258 => {
                info.bits_per_sample =
                    read_tag_values_u16(file, data_type, count, value_offset, endian)?
            }
            259 => {
                let comp = read_tag_value_u16(file, data_type, count, value_offset, endian)?;
                info.compression = TiffCompression::from_u16(comp)?;
            }
            262 => {
                let photo = read_tag_value_u16(file, data_type, count, value_offset, endian)?;
                info.photometric = PhotometricInterpretation::from_u16(photo)?;
            }
            270 => {
                let desc = read_tag_string(file, count, value_offset)?;
                info.metadata.insert("description".to_string(), desc);
            }
            273 => {
                info.strip_offsets =
                    read_tag_values_u32(file, data_type, count, value_offset, endian)?
            }
            277 => {
                info.samples_per_pixel =
                    read_tag_value_u16(file, data_type, count, value_offset, endian)?
            }
            278 => {
                info.rows_per_strip =
                    read_tag_value_u32(file, data_type, count, value_offset, endian)?
            }
            279 => {
                info.strip_byte_counts =
                    read_tag_values_u32(file, data_type, count, value_offset, endian)?
            }
            284 => {
                info.planar_config =
                    read_tag_value_u16(file, data_type, count, value_offset, endian)?
            }
            305 => {
                let software = read_tag_string(file, count, value_offset)?;
                info.metadata.insert("software".to_string(), software);
            }
            306 => {
                let datetime = read_tag_string(file, count, value_offset)?;
                info.metadata.insert("datetime".to_string(), datetime);
            }
            322 => {
                info.tile_width = read_tag_value_u32(file, data_type, count, value_offset, endian)?;
                info.is_tiled = true;
            }
            323 => {
                info.tile_height =
                    read_tag_value_u32(file, data_type, count, value_offset, endian)?;
                info.is_tiled = true;
            }
            324 => {
                info.tile_offsets =
                    read_tag_values_u32(file, data_type, count, value_offset, endian)?;
                info.is_tiled = true;
            }
            325 => {
                info.tile_byte_counts =
                    read_tag_values_u32(file, data_type, count, value_offset, endian)?;
                info.is_tiled = true;
            }
            339 => {
                info.sample_format =
                    read_tag_values_u16(file, data_type, count, value_offset, endian)?
            }
            347 => {
                // JPEGTables: abbreviated JPEG datastream with shared tables.
                info.jpeg_tables = Some(read_tag_bytes(file, count, value_offset)?);
            }
            _ => {
                // Preserve unknown/non-standard tags when requested.
                if info.preserve_unknown_tags {
                    let entry = IfdEntry {
                        tag,
                        data_type,
                        count,
                        value_offset,
                    };
                    info.extra_ifd_entries.insert(tag, entry);
                }
            }
        }
    }

    // Set default rows per strip if not specified
    if info.rows_per_strip == 0 {
        info.rows_per_strip = info.height;
    }

    Ok(info)
}

fn read_tag_value_u16(
    file: &mut File,
    data_type: TiffDataType,
    count: u32,
    value_offset: u32,
    endian: Endian,
) -> ImageResult<u16> {
    if count != 1 {
        return Err(ImageError::invalid_format("Expected single value"));
    }

    let total_size = data_type.size() * count as usize;

    if total_size <= 4 {
        // Value is stored in offset field
        Ok((value_offset & 0xFFFF) as u16)
    } else {
        let saved_pos = file.stream_position()?;
        file.seek(SeekFrom::Start(u64::from(value_offset)))?;

        let value = match endian {
            Endian::Big => file.read_u16::<BigEndian>()?,
            Endian::Little => file.read_u16::<LittleEndian>()?,
        };

        file.seek(SeekFrom::Start(saved_pos))?;
        Ok(value)
    }
}

fn read_tag_value_u32(
    file: &mut File,
    data_type: TiffDataType,
    count: u32,
    value_offset: u32,
    endian: Endian,
) -> ImageResult<u32> {
    if count != 1 {
        return Err(ImageError::invalid_format("Expected single value"));
    }

    let total_size = data_type.size() * count as usize;

    if total_size <= 4 {
        Ok(value_offset)
    } else {
        let saved_pos = file.stream_position()?;
        file.seek(SeekFrom::Start(u64::from(value_offset)))?;

        let value = match endian {
            Endian::Big => file.read_u32::<BigEndian>()?,
            Endian::Little => file.read_u32::<LittleEndian>()?,
        };

        file.seek(SeekFrom::Start(saved_pos))?;
        Ok(value)
    }
}

fn read_tag_values_u16(
    file: &mut File,
    data_type: TiffDataType,
    count: u32,
    value_offset: u32,
    endian: Endian,
) -> ImageResult<Vec<u16>> {
    let total_size = data_type.size() * count as usize;
    let mut values = Vec::new();

    if total_size <= 4 {
        // Values are stored in offset field
        for i in 0..count {
            let shift = (count - 1 - i) * 16;
            values.push(((value_offset >> shift) & 0xFFFF) as u16);
        }
    } else {
        let saved_pos = file.stream_position()?;
        file.seek(SeekFrom::Start(u64::from(value_offset)))?;

        for _ in 0..count {
            values.push(match endian {
                Endian::Big => file.read_u16::<BigEndian>()?,
                Endian::Little => file.read_u16::<LittleEndian>()?,
            });
        }

        file.seek(SeekFrom::Start(saved_pos))?;
    }

    Ok(values)
}

fn read_tag_values_u32(
    file: &mut File,
    data_type: TiffDataType,
    count: u32,
    value_offset: u32,
    endian: Endian,
) -> ImageResult<Vec<u32>> {
    let total_size = data_type.size() * count as usize;
    let mut values = Vec::new();

    if total_size <= 4 && count == 1 {
        values.push(value_offset);
    } else {
        let saved_pos = file.stream_position()?;
        file.seek(SeekFrom::Start(u64::from(value_offset)))?;

        for _ in 0..count {
            values.push(match endian {
                Endian::Big => file.read_u32::<BigEndian>()?,
                Endian::Little => file.read_u32::<LittleEndian>()?,
            });
        }

        file.seek(SeekFrom::Start(saved_pos))?;
    }

    Ok(values)
}

fn read_tag_string(file: &mut File, count: u32, value_offset: u32) -> ImageResult<String> {
    let saved_pos = file.stream_position()?;

    if count <= 4 {
        // String is in the offset field
        let bytes = value_offset.to_le_bytes();
        Ok(String::from_utf8_lossy(&bytes[..count.min(4) as usize])
            .trim_end_matches('\0')
            .to_string())
    } else {
        // Out-of-line string: bound-check the offset so a malformed tag does
        // not abort the whole IFD parse with an EOF error.
        let file_len = file.seek(SeekFrom::End(0))?;
        file.seek(SeekFrom::Start(saved_pos))?;
        if u64::from(value_offset) >= file_len {
            return Ok(String::new());
        }
        let available = (file_len - u64::from(value_offset)) as usize;
        let read_len = (count as usize).min(available);

        file.seek(SeekFrom::Start(u64::from(value_offset)))?;
        let mut bytes = vec![0u8; read_len];
        file.read_exact(&mut bytes)?;
        file.seek(SeekFrom::Start(saved_pos))?;

        Ok(String::from_utf8_lossy(&bytes)
            .trim_end_matches('\0')
            .to_string())
    }
}

/// Read a raw byte array tag value (used for `JPEGTables`).
fn read_tag_bytes(file: &mut File, count: u32, value_offset: u32) -> ImageResult<Vec<u8>> {
    let saved_pos = file.stream_position()?;
    let result = if count <= 4 {
        // Bytes are packed into the value/offset field (little-endian layout).
        value_offset.to_le_bytes()[..count as usize].to_vec()
    } else {
        // Out-of-line payload: bound-check the offset against the file length.
        let file_len = file.seek(SeekFrom::End(0))?;
        file.seek(SeekFrom::Start(saved_pos))?;
        if u64::from(value_offset) >= file_len {
            return Ok(Vec::new());
        }
        let available = (file_len - u64::from(value_offset)) as usize;
        let read_len = (count as usize).min(available);
        file.seek(SeekFrom::Start(u64::from(value_offset)))?;
        let mut bytes = vec![0u8; read_len];
        file.read_exact(&mut bytes)?;
        bytes
    };
    file.seek(SeekFrom::Start(saved_pos))?;
    Ok(result)
}

fn read_stripped_image(file: &mut File, info: &TiffInfo, _endian: Endian) -> ImageResult<Vec<u8>> {
    let bits_total: u16 = info.bits_per_sample.iter().sum();
    // Bilevel imagery (1 bit/pixel) keeps a packed row layout; multi-byte
    // imagery uses whole bytes per pixel.
    let row_bytes = if bits_total <= 8 {
        (info.width as usize)
            .saturating_mul(bits_total as usize)
            .div_ceil(8)
    } else {
        (info.width as usize) * (bits_total / 8) as usize
    };
    let total_size = row_bytes * (info.height as usize);
    let mut output = vec![0u8; total_size];

    let mut output_offset = 0;
    let rows_per_strip = info.rows_per_strip.max(1) as usize;
    let height = info.height as usize;

    for (strip_idx, (strip_offset, strip_bytes)) in info
        .strip_offsets
        .iter()
        .zip(&info.strip_byte_counts)
        .enumerate()
    {
        file.seek(SeekFrom::Start(u64::from(*strip_offset)))?;

        let mut strip_data = vec![0u8; *strip_bytes as usize];
        file.read_exact(&mut strip_data)?;

        // The last strip may hold fewer than rows_per_strip rows.
        let rows_in_strip = rows_per_strip.min(height.saturating_sub(strip_idx * rows_per_strip));

        // Decompress if needed
        let decompressed = decompress_strip(&strip_data, info, rows_in_strip)?;

        let copy_len = decompressed.len().min(total_size - output_offset);
        if copy_len > 0 {
            output[output_offset..output_offset + copy_len]
                .copy_from_slice(&decompressed[..copy_len]);
            output_offset += copy_len;
        }
    }

    Ok(output)
}

fn read_tiled_image(file: &mut File, info: &TiffInfo, endian: Endian) -> ImageResult<Vec<u8>> {
    let width = info.width as usize;
    let height = info.height as usize;
    let tile_width = info.tile_width as usize;
    let tile_height = info.tile_height as usize;

    if tile_width == 0 || tile_height == 0 {
        return Err(ImageError::invalid_format(
            "Tiled TIFF has zero tile dimension",
        ));
    }

    let bits_total: u16 = info.bits_per_sample.iter().sum();
    let bytes_per_pixel = (bits_total / 8) as usize;

    if bytes_per_pixel == 0 {
        return Err(ImageError::invalid_format(
            "Tiled TIFF: zero bytes per pixel",
        ));
    }

    let total_size = width * height * bytes_per_pixel;
    let mut output = vec![0u8; total_size];

    // Compute tile grid dimensions (ceiling division)
    let tiles_across = width.div_ceil(tile_width);
    let tiles_down = height.div_ceil(tile_height);

    let tile_count = tiles_across * tiles_down;
    if info.tile_offsets.len() < tile_count || info.tile_byte_counts.len() < tile_count {
        return Err(ImageError::invalid_format(
            "Tiled TIFF: insufficient tile offset/bytecount entries",
        ));
    }

    for tile_row in 0..tiles_down {
        for tile_col in 0..tiles_across {
            let tile_idx = tile_row * tiles_across + tile_col;

            let tile_offset = u64::from(info.tile_offsets[tile_idx]);
            let tile_bytes = info.tile_byte_counts[tile_idx] as usize;

            file.seek(SeekFrom::Start(tile_offset))?;

            let mut compressed = vec![0u8; tile_bytes];
            file.read_exact(&mut compressed)?;

            // A tile always carries tile_height rows in its coded stream.
            let decompressed = decompress_strip(&compressed, info, tile_height)?;

            // Compute tile position in the output image
            let x_start = tile_col * tile_width;
            let y_start = tile_row * tile_height;

            // Actual tile size may be smaller at image edges
            let actual_tile_w = tile_width.min(width.saturating_sub(x_start));
            let actual_tile_h = tile_height.min(height.saturating_sub(y_start));

            // The tile data is laid out as full tile_width rows in the compressed stream.
            // We copy only actual_tile_w pixels per row, skipping any padding bytes.
            let src_row_stride = tile_width * bytes_per_pixel;
            let dst_row_stride = width * bytes_per_pixel;
            let copy_len = actual_tile_w * bytes_per_pixel;

            for tile_y in 0..actual_tile_h {
                let src_offset = tile_y * src_row_stride;
                let dst_offset = (y_start + tile_y) * dst_row_stride + x_start * bytes_per_pixel;

                if src_offset + copy_len > decompressed.len() {
                    break;
                }
                if dst_offset + copy_len > output.len() {
                    break;
                }

                output[dst_offset..dst_offset + copy_len]
                    .copy_from_slice(&decompressed[src_offset..src_offset + copy_len]);
            }
        }
    }

    // Endian-swap 16-bit samples when data is stored in big-endian and we are on little-endian
    // (mirrors the behaviour expected for multi-byte pixel types)
    let bits_first = info.bits_per_sample.first().copied().unwrap_or(8);
    if bits_first == 16 && endian == Endian::Big {
        for chunk in output.chunks_exact_mut(2) {
            chunk.swap(0, 1);
        }
    }

    Ok(output)
}

/// Decompress a single TIFF strip (or tile).
///
/// `rows_in_strip` is the number of image rows the strip's coded stream
/// represents — needed by the CCITT fax codecs, which carry no intrinsic
/// row count.
fn decompress_strip(data: &[u8], info: &TiffInfo, rows_in_strip: usize) -> ImageResult<Vec<u8>> {
    match info.compression {
        TiffCompression::None => Ok(data.to_vec()),
        TiffCompression::Lzw => decompress_lzw(data),
        TiffCompression::PackBits => decompress_packbits(data),
        TiffCompression::Deflate | TiffCompression::DeflateAdobe => decompress_deflate(data),
        TiffCompression::CcittRle | TiffCompression::CcittFax3 | TiffCompression::CcittFax4 => {
            decompress_ccitt(data, info, rows_in_strip)
        }
        TiffCompression::Jpeg => {
            jpeg_compression::decode_jpeg_strip(data, info.jpeg_tables.as_deref())
        }
        TiffCompression::JpegOld => Err(ImageError::unsupported(
            "TIFF old-style JPEG (compression 6) is obsolete and not supported",
        )),
    }
}

/// Decompress a CCITT-coded bilevel strip into a packed 1-bpp bitmap.
fn decompress_ccitt(data: &[u8], info: &TiffInfo, rows_in_strip: usize) -> ImageResult<Vec<u8>> {
    let bits_total: u16 = info.bits_per_sample.iter().sum();
    if bits_total != 1 {
        return Err(ImageError::unsupported(
            "CCITT compression requires 1-bit (bilevel) imagery",
        ));
    }
    // CCITT-in-tiled-TIFF would code each tile at tile_width; the common case
    // is stripped imagery coded at the full image width.
    let width = if info.is_tiled {
        info.tile_width as usize
    } else {
        info.width as usize
    };
    // TIFF default photometric for CCITT is WhiteIsZero: a 0 bit is white.
    let white_is_zero = matches!(
        info.photometric,
        PhotometricInterpretation::WhiteIsZero | PhotometricInterpretation::TransparencyMask
    );
    match info.compression {
        TiffCompression::CcittRle => {
            ccitt::decode_ccitt_rle(data, width, rows_in_strip, white_is_zero)
        }
        TiffCompression::CcittFax3 => {
            ccitt::decode_ccitt_fax3(data, width, rows_in_strip, white_is_zero)
        }
        TiffCompression::CcittFax4 => {
            ccitt::decode_ccitt_fax4(data, width, rows_in_strip, white_is_zero)
        }
        _ => unreachable!("decompress_ccitt called with non-CCITT compression"),
    }
}

fn decompress_lzw(data: &[u8]) -> ImageResult<Vec<u8>> {
    use weezl::{decode::Decoder, BitOrder};

    let mut decoder = Decoder::new(BitOrder::Msb, 8);
    let result = decoder
        .decode(data)
        .map_err(|e| ImageError::Compression(format!("LZW decompression failed: {e:?}")))?;

    Ok(result)
}

fn decompress_packbits(data: &[u8]) -> ImageResult<Vec<u8>> {
    let mut output = Vec::new();
    let mut i = 0;

    while i < data.len() {
        let header = data[i] as i8;
        i += 1;

        if header >= 0 {
            // Copy next (header + 1) bytes literally
            let count = (header + 1) as usize;
            if i + count > data.len() {
                break;
            }
            output.extend_from_slice(&data[i..i + count]);
            i += count;
        } else if header != -128 {
            // Repeat next byte (-header + 1) times
            let count = (-header + 1) as usize;
            if i >= data.len() {
                break;
            }
            let byte = data[i];
            i += 1;
            output.extend(std::iter::repeat_n(byte, count));
        }
    }

    Ok(output)
}

fn decompress_deflate(data: &[u8]) -> ImageResult<Vec<u8>> {
    use oxiarc_deflate::ZlibStreamDecoder;

    let mut decoder = ZlibStreamDecoder::new(data);
    let mut output = Vec::new();
    decoder
        .read_to_end(&mut output)
        .map_err(|e| ImageError::Compression(format!("Deflate decompression failed: {e}")))?;

    Ok(output)
}

fn determine_pixel_type(info: &TiffInfo) -> ImageResult<PixelType> {
    let bits = info.bits_per_sample.first().copied().unwrap_or(8);
    let sample_format = info.sample_format.first().copied().unwrap_or(1);

    match (bits, sample_format) {
        // Bilevel (1-bit) and sub-byte greyscale imagery is delivered as a
        // packed bitmap carried in U8 storage — each byte holds 8 pixels.
        (1..=8, 1) => Ok(PixelType::U8),
        (16, 1) => Ok(PixelType::U16),
        (32, 1) => Ok(PixelType::U32),
        (16, 3) => Ok(PixelType::F16),
        (32, 3) => Ok(PixelType::F32),
        _ => Err(ImageError::unsupported(format!(
            "Bit depth {bits} with format {sample_format}"
        ))),
    }
}

fn determine_color_space(info: &TiffInfo) -> ColorSpace {
    match info.photometric {
        PhotometricInterpretation::Rgb => ColorSpace::LinearRgb,
        PhotometricInterpretation::BlackIsZero | PhotometricInterpretation::WhiteIsZero => {
            ColorSpace::Luma
        }
        PhotometricInterpretation::Cmyk => ColorSpace::Cmyk,
        PhotometricInterpretation::YCbCr => ColorSpace::YCbCr,
        _ => ColorSpace::LinearRgb,
    }
}

fn create_tiff_info(frame: &ImageFrame, compression: TiffCompression) -> ImageResult<TiffInfo> {
    let mut info = TiffInfo::default();

    info.width = frame.width;
    info.height = frame.height;
    info.samples_per_pixel = u16::from(frame.components);
    info.compression = compression;

    // Set bits per sample based on pixel type
    let bits = u16::from(frame.pixel_type.bit_depth());
    info.bits_per_sample = vec![bits; frame.components as usize];

    // Set sample format
    let format = if frame.pixel_type.is_float() { 3 } else { 1 };
    info.sample_format = vec![format; frame.components as usize];

    // Set photometric interpretation
    info.photometric = match (frame.components, &frame.color_space) {
        (1, _) => PhotometricInterpretation::BlackIsZero,
        (3 | 4, ColorSpace::LinearRgb) => PhotometricInterpretation::Rgb,
        (3 | 4, ColorSpace::YCbCr) => PhotometricInterpretation::YCbCr,
        (4, ColorSpace::Cmyk) => PhotometricInterpretation::Cmyk,
        _ => PhotometricInterpretation::Rgb,
    };

    info.rows_per_strip = frame.height;
    info.planar_config = 1; // Chunky format

    // For JPEG ("new-style") compression, emit a JPEGTables tag carrying the
    // shared quantization/Huffman tables. The strips remain self-contained, so
    // the tag is informational, but a strict TIFF reader expects to see it.
    if compression == TiffCompression::Jpeg {
        let components = if frame.components == 1 { 1u8 } else { 3 };
        info.jpeg_tables = Some(jpeg_compression::build_jpeg_tables(
            components,
            info.jpeg_quality,
        )?);
    }

    Ok(info)
}

fn compress_image_data(data: &[u8], info: &TiffInfo) -> ImageResult<Vec<u8>> {
    match info.compression {
        TiffCompression::None => Ok(data.to_vec()),
        TiffCompression::Lzw => compress_lzw(data),
        TiffCompression::PackBits => compress_packbits(data),
        TiffCompression::Deflate | TiffCompression::DeflateAdobe => compress_deflate(data),
        TiffCompression::CcittRle | TiffCompression::CcittFax3 | TiffCompression::CcittFax4 => {
            compress_ccitt(data, info)
        }
        TiffCompression::Jpeg => compress_jpeg(data, info),
        TiffCompression::JpegOld => Err(ImageError::unsupported(
            "TIFF old-style JPEG (compression 6) is obsolete; never produced on the write path",
        )),
    }
}

/// Compress a packed 1-bpp bitmap with a CCITT fax codec.
fn compress_ccitt(data: &[u8], info: &TiffInfo) -> ImageResult<Vec<u8>> {
    let bits_total: u16 = info.bits_per_sample.iter().sum();
    if bits_total != 1 {
        return Err(ImageError::unsupported(
            "CCITT compression requires 1-bit (bilevel) imagery",
        ));
    }
    let width = info.width as usize;
    let rows = info.height as usize;
    let white_is_zero = matches!(
        info.photometric,
        PhotometricInterpretation::WhiteIsZero | PhotometricInterpretation::TransparencyMask
    );
    let encoded = match info.compression {
        TiffCompression::CcittRle => ccitt::encode_ccitt_rle(data, width, rows, white_is_zero),
        TiffCompression::CcittFax3 => ccitt::encode_ccitt_fax3(data, width, rows, white_is_zero),
        TiffCompression::CcittFax4 => ccitt::encode_ccitt_fax4(data, width, rows, white_is_zero),
        _ => unreachable!("compress_ccitt called with non-CCITT compression"),
    };
    Ok(encoded)
}

/// Compress interleaved 8-bit pixel data as a baseline JPEG datastream.
fn compress_jpeg(data: &[u8], info: &TiffInfo) -> ImageResult<Vec<u8>> {
    let bits_total: u16 = info.bits_per_sample.iter().sum();
    let components = info.samples_per_pixel.max(1) as u8;
    if bits_total != u16::from(components) * 8 {
        return Err(ImageError::unsupported(
            "JPEG compression requires 8-bit components",
        ));
    }
    jpeg_compression::encode_jpeg_strip(
        data,
        info.width,
        info.height,
        components,
        info.jpeg_quality,
    )
}

fn compress_lzw(data: &[u8]) -> ImageResult<Vec<u8>> {
    use weezl::{encode::Encoder, BitOrder};

    let mut encoder = Encoder::new(BitOrder::Msb, 8);
    encoder
        .encode(data)
        .map_err(|e| ImageError::Compression(format!("LZW compression failed: {e:?}")))
}

fn compress_packbits(data: &[u8]) -> ImageResult<Vec<u8>> {
    let mut output = Vec::new();
    let mut i = 0;

    while i < data.len() {
        // Find run length
        let run_len = count_run_length(&data[i..]);

        if run_len >= 3 {
            // Encode as run
            let count = run_len.min(128);
            output.push((-(count as i8) + 1) as u8);
            output.push(data[i]);
            i += count;
        } else {
            // Encode as literal
            let lit_len = find_literal_length(&data[i..]);
            let count = lit_len.min(128);
            output.push((count - 1) as u8);
            output.extend_from_slice(&data[i..i + count]);
            i += count;
        }
    }

    Ok(output)
}

fn count_run_length(data: &[u8]) -> usize {
    if data.is_empty() {
        return 0;
    }

    let first = data[0];
    data.iter().take_while(|&&b| b == first).count()
}

fn find_literal_length(data: &[u8]) -> usize {
    let mut len = 1;

    while len < data.len() {
        let run = count_run_length(&data[len..]);
        if run >= 3 {
            break;
        }
        len += 1;
    }

    len
}

fn compress_deflate(data: &[u8]) -> ImageResult<Vec<u8>> {
    use oxiarc_deflate::ZlibStreamEncoder;

    let mut encoder = ZlibStreamEncoder::new(Vec::new(), 6);
    encoder
        .write_all(data)
        .map_err(|e| ImageError::Compression(format!("Deflate compression failed: {e}")))?;

    encoder
        .finish()
        .map_err(|e| ImageError::Compression(format!("Deflate compression failed: {e}")))
}

/// Pack 1 or 2 `Short` values into the 4-byte value field of an IFD entry.
///
/// The matching reader places value `i` at bit-shift `(count - 1 - i) * 16`.
/// So a single value occupies the low 16 bits; a pair packs as
/// `(v0 << 16) | v1`. This keeps `write_tag` output consistent with both
/// `read_tag_value_u16` and `read_tag_values_u16`.
fn pack_inline_shorts(values: &[u16]) -> u32 {
    let count = values.len().max(1) as u32;
    let mut field = 0u32;
    for (i, &v) in values.iter().enumerate() {
        let shift = (count - 1 - i as u32) * 16;
        field |= u32::from(v) << shift;
    }
    field
}

#[allow(clippy::too_many_arguments)]
fn write_ifd(
    file: &mut File,
    info: &TiffInfo,
    data_offset: u32,
    data_size: u32,
    endian: Endian,
) -> ImageResult<()> {
    let ifd_offset = file.stream_position()?;

    // A `Short` array fits inline only when 2*N <= 4, i.e. N <= 2. Wider
    // BitsPerSample / SampleFormat arrays must be stored out-of-line after the
    // IFD. The JPEGTables datastream is likewise stored out-of-line.
    let samples = info.samples_per_pixel.max(1) as usize;
    let bps: Vec<u16> = {
        let mut v = info.bits_per_sample.clone();
        v.resize(samples, *info.bits_per_sample.first().unwrap_or(&8));
        v
    };
    let sfmt: Vec<u16> = {
        let mut v = info.sample_format.clone();
        v.resize(samples, *info.sample_format.first().unwrap_or(&1));
        v
    };
    let bps_inline = samples <= 2;
    let sfmt_inline = samples <= 2;
    let has_jpeg_tables = info.jpeg_tables.as_ref().is_some_and(|t| !t.is_empty());

    // Collect extra inline tags (only those whose value fits in 4 bytes).
    // Sort by tag number so the IFD remains in ascending order.
    let mut extra_tags: Vec<&IfdEntry> = info
        .extra_ifd_entries
        .values()
        .filter(|e| {
            let size = e.data_type.size() * e.count as usize;
            size <= 4
        })
        .collect();
    extra_tags.sort_by_key(|e| e.tag);
    let extra_count = extra_tags.len() as u16;

    // 12 base tags (256, 257, 258, 259, 262, 273, 277, 278, 279, 284, 305,
    // 339); the JPEGTables tag (347) adds a 13th when present; extra inline
    // tags add further entries.
    let tag_count: u16 = (if has_jpeg_tables { 13 } else { 12 }) + extra_count;

    // The IFD occupies: 2-byte count + tag_count*12-byte entries + 4-byte
    // next-IFD pointer. Out-of-line data blocks follow directly after, in a
    // fixed order: BitsPerSample, SampleFormat, JPEGTables.
    let ifd_end = ifd_offset + 2 + u64::from(tag_count) * 12 + 4;
    let mut cursor = ifd_end;
    let bps_offset = cursor;
    if !bps_inline {
        cursor += (bps.len() * 2) as u64;
    }
    let sfmt_offset = cursor;
    if !sfmt_inline {
        cursor += (sfmt.len() * 2) as u64;
    }
    let jpeg_tables_offset = cursor;

    match endian {
        Endian::Big => file.write_u16::<BigEndian>(tag_count)?,
        Endian::Little => file.write_u16::<LittleEndian>(tag_count)?,
    }

    // Tags must be emitted in ascending tag-number order.
    write_tag(file, 256, TiffDataType::Long, 1, info.width, endian)?;
    write_tag(file, 257, TiffDataType::Long, 1, info.height, endian)?;

    // 258 BitsPerSample
    if bps_inline {
        write_tag(
            file,
            258,
            TiffDataType::Short,
            bps.len() as u32,
            pack_inline_shorts(&bps),
            endian,
        )?;
    } else {
        write_tag(
            file,
            258,
            TiffDataType::Short,
            bps.len() as u32,
            bps_offset as u32,
            endian,
        )?;
    }

    write_tag(
        file,
        259,
        TiffDataType::Short,
        1,
        pack_inline_shorts(&[info.compression as u16]),
        endian,
    )?;
    write_tag(
        file,
        262,
        TiffDataType::Short,
        1,
        pack_inline_shorts(&[info.photometric as u16]),
        endian,
    )?;
    write_tag(file, 273, TiffDataType::Long, 1, data_offset, endian)?;
    write_tag(
        file,
        277,
        TiffDataType::Short,
        1,
        pack_inline_shorts(&[info.samples_per_pixel]),
        endian,
    )?;
    write_tag(
        file,
        278,
        TiffDataType::Long,
        1,
        info.rows_per_strip,
        endian,
    )?;
    write_tag(file, 279, TiffDataType::Long, 1, data_size, endian)?;
    write_tag(
        file,
        284,
        TiffDataType::Short,
        1,
        pack_inline_shorts(&[info.planar_config]),
        endian,
    )?;

    // 305 Software (always 4 bytes, inline).
    write_software_tag(file, endian)?;

    // 339 SampleFormat
    if sfmt_inline {
        write_tag(
            file,
            339,
            TiffDataType::Short,
            sfmt.len() as u32,
            pack_inline_shorts(&sfmt),
            endian,
        )?;
    } else {
        write_tag(
            file,
            339,
            TiffDataType::Short,
            sfmt.len() as u32,
            sfmt_offset as u32,
            endian,
        )?;
    }

    // 347 JPEGTables (out-of-line abbreviated JPEG datastream).
    if has_jpeg_tables {
        let tables = info
            .jpeg_tables
            .as_ref()
            .ok_or_else(|| ImageError::invalid_format("JPEGTables expected"))?;
        write_tag(
            file,
            347,
            TiffDataType::Undefined,
            tables.len() as u32,
            jpeg_tables_offset as u32,
            endian,
        )?;
    }

    // Extra inline tags (unknown/non-standard), sorted by tag number.
    // Only inline-fitting entries (size <= 4 bytes) are emitted.
    for entry in &extra_tags {
        write_tag(
            file,
            entry.tag,
            entry.data_type,
            entry.count,
            entry.value_offset,
            endian,
        )?;
    }

    // Next IFD offset (0 = no more IFDs).
    match endian {
        Endian::Big => file.write_u32::<BigEndian>(0)?,
        Endian::Little => file.write_u32::<LittleEndian>(0)?,
    }

    // Out-of-line data blocks, in the order their offsets were reserved.
    debug_assert_eq!(file.stream_position()?, ifd_end);
    if !bps_inline {
        for &v in &bps {
            match endian {
                Endian::Big => file.write_u16::<BigEndian>(v)?,
                Endian::Little => file.write_u16::<LittleEndian>(v)?,
            }
        }
    }
    if !sfmt_inline {
        for &v in &sfmt {
            match endian {
                Endian::Big => file.write_u16::<BigEndian>(v)?,
                Endian::Little => file.write_u16::<LittleEndian>(v)?,
            }
        }
    }
    if has_jpeg_tables {
        let tables = info
            .jpeg_tables
            .as_ref()
            .ok_or_else(|| ImageError::invalid_format("JPEGTables expected"))?;
        file.write_all(tables)?;
    }

    // Update IFD offset in the file header.
    file.seek(SeekFrom::Start(4))?;
    match endian {
        Endian::Big => file.write_u32::<BigEndian>(ifd_offset as u32)?,
        Endian::Little => file.write_u32::<LittleEndian>(ifd_offset as u32)?,
    }

    Ok(())
}

fn write_tag(
    file: &mut File,
    tag: u16,
    data_type: TiffDataType,
    count: u32,
    value: u32,
    endian: Endian,
) -> ImageResult<()> {
    match endian {
        Endian::Big => {
            file.write_u16::<BigEndian>(tag)?;
            file.write_u16::<BigEndian>(data_type as u16)?;
            file.write_u32::<BigEndian>(count)?;
            file.write_u32::<BigEndian>(value)?;
        }
        Endian::Little => {
            file.write_u16::<LittleEndian>(tag)?;
            file.write_u16::<LittleEndian>(data_type as u16)?;
            file.write_u32::<LittleEndian>(count)?;
            file.write_u32::<LittleEndian>(value)?;
        }
    }

    Ok(())
}

fn write_software_tag(file: &mut File, endian: Endian) -> ImageResult<()> {
    // The Software (305) tag is an ASCII string. To keep the IFD fixed-layout
    // (no out-of-line data for this tag), the value must fit in the 4-byte
    // value field: a 3-character NUL-terminated identifier. `count` must equal
    // the stored length so the reader takes the inline path.
    let software = b"Oxi\0";
    let tag = 305u16;
    let data_type = TiffDataType::Ascii;
    let count = software.len() as u32;

    // Store the 4 bytes in the value field (little-endian byte order).
    let value = u32::from_le_bytes([
        software[0],
        software.get(1).copied().unwrap_or(0),
        software.get(2).copied().unwrap_or(0),
        software.get(3).copied().unwrap_or(0),
    ]);

    match endian {
        Endian::Big => {
            file.write_u16::<BigEndian>(tag)?;
            file.write_u16::<BigEndian>(data_type as u16)?;
            file.write_u32::<BigEndian>(count)?;
            file.write_u32::<BigEndian>(value)?;
        }
        Endian::Little => {
            file.write_u16::<LittleEndian>(tag)?;
            file.write_u16::<LittleEndian>(data_type as u16)?;
            file.write_u32::<LittleEndian>(count)?;
            file.write_u32::<LittleEndian>(value)?;
        }
    }

    Ok(())
}

// ===========================================================================
// Tests (moved to tests.rs to keep this file under 2000 lines)
// ===========================================================================

#[cfg(test)]
mod tests;
