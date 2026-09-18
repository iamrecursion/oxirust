//! DPX (Digital Picture Exchange) format support.
//!
//! Implements SMPTE 268M-2003 v2.0 standard for professional cinema and VFX workflows.
//!
//! # Features
//!
//! - All pixel formats (RGB, RGBA, YCbCr, Luma)
//! - All bit depths (8, 10, 12, 16-bit)
//! - Packed and filled packing modes
//! - Big-endian and little-endian support
//! - Comprehensive metadata (camera, film, TV, user data)
//!
//! # Example
//!
//! ```no_run
//! use oximedia_image::dpx;
//! use std::path::Path;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let frame = dpx::read_dpx(Path::new("frame.dpx"), 1)?;
//! println!("DPX frame: {}x{}", frame.width, frame.height);
//! # Ok(())
//! # }
//! ```

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::similar_names)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::match_same_arms)]

use crate::error::{ImageError, ImageResult};
use crate::{ColorSpace, Endian, ImageData, ImageFrame, PixelType};
use byteorder::{BigEndian, LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

/// DPX magic number (big-endian).
const DPX_MAGIC_BE: u32 = 0x5344_5058; // "SDPX"

/// DPX magic number (little-endian).
const DPX_MAGIC_LE: u32 = 0x5850_4453; // "XPDS"

/// 10-bit packing method for DPX format.
///
/// SMPTE 268M defines two packing methods for 10-bit data in 32-bit words:
/// - **Method A**: 3 components packed into one 32-bit word, MSB-aligned, 2-bit padding at LSB.
///   Layout: `[C0(10) | C1(10) | C2(10) | pad(2)]`
/// - **Method B**: 3 components packed into one 32-bit word, LSB-aligned, 2-bit padding at MSB.
///   Layout: `[pad(2) | C0(10) | C1(10) | C2(10)]`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackingMethod {
    /// Method A: MSB-aligned, 2-bit padding at LSB end.
    MethodA,
    /// Method B: LSB-aligned, 2-bit padding at MSB end.
    MethodB,
}

impl Default for PackingMethod {
    fn default() -> Self {
        Self::MethodA
    }
}

/// DPX file header (2048 bytes total).
#[derive(Debug, Clone, Default)]
pub struct DpxHeader {
    /// File information header.
    pub file: FileInfo,
    /// Image information header.
    pub image: ImageInfo,
    /// Image orientation header.
    pub orientation: OrientationInfo,
    /// Motion picture film information.
    pub film: FilmInfo,
    /// Television information.
    pub television: TelevisionInfo,
}

/// File information header (768 bytes).
#[derive(Debug, Clone)]
pub struct FileInfo {
    /// Magic number (SDPX or XPDS).
    pub magic: u32,
    /// Offset to image data in bytes.
    pub image_offset: u32,
    /// Version string.
    pub version: String,
    /// Total file size in bytes.
    pub file_size: u32,
    /// Ditto key (0 = same as previous frame, 1 = new).
    pub ditto_key: u32,
    /// Generic section header length.
    pub generic_size: u32,
    /// Industry specific header length.
    pub industry_size: u32,
    /// User-defined data length.
    pub user_size: u32,
    /// Image filename.
    pub filename: String,
    /// Creation timestamp.
    pub timestamp: String,
    /// Creator software.
    pub creator: String,
    /// Project name.
    pub project: String,
    /// Copyright notice.
    pub copyright: String,
    /// Encryption key (0xFFFFFFFF = unencrypted).
    pub encrypt_key: u32,
}

/// Image information header (640 bytes).
#[derive(Debug, Clone)]
pub struct ImageInfo {
    /// Image orientation (0 = left-to-right, top-to-bottom).
    pub orientation: u16,
    /// Number of image elements (1-8).
    pub element_count: u16,
    /// Pixels per line.
    pub width: u32,
    /// Lines per image.
    pub height: u32,
    /// Image elements.
    pub elements: Vec<ImageElement>,
}

/// Image element (72 bytes per element).
#[derive(Debug, Clone)]
pub struct ImageElement {
    /// Data sign (0 = unsigned, 1 = signed).
    pub data_sign: u32,
    /// Reference low data code value.
    pub ref_low_data: u32,
    /// Reference low quantity.
    pub ref_low_quantity: f32,
    /// Reference high data code value.
    pub ref_high_data: u32,
    /// Reference high quantity.
    pub ref_high_quantity: f32,
    /// Descriptor (pixel format).
    pub descriptor: u8,
    /// Transfer characteristic (gamma).
    pub transfer: u8,
    /// Colorimetric specification.
    pub colorimetric: u8,
    /// Bit depth (8, 10, 12, 16).
    pub bit_depth: u8,
    /// Packing (0 = packed, 1 = filled to 32-bit).
    pub packing: u16,
    /// Encoding (0 = none, 1 = RLE).
    pub encoding: u16,
    /// Data offset.
    pub data_offset: u32,
    /// End-of-line padding.
    pub eol_padding: u32,
    /// End-of-image padding.
    pub eoi_padding: u32,
    /// Description.
    pub description: String,
}

/// Image orientation header (256 bytes).
#[derive(Debug, Clone)]
pub struct OrientationInfo {
    /// X offset.
    pub x_offset: u32,
    /// Y offset.
    pub y_offset: u32,
    /// X center.
    pub x_center: f32,
    /// Y center.
    pub y_center: f32,
    /// X original size.
    pub x_original_size: u32,
    /// Y original size.
    pub y_original_size: u32,
    /// Source image filename.
    pub source_filename: String,
    /// Source image timestamp.
    pub source_timestamp: String,
    /// Input device name.
    pub input_device: String,
    /// Input device serial number.
    pub input_serial: String,
}

/// Motion picture film information (256 bytes).
#[derive(Debug, Clone)]
pub struct FilmInfo {
    /// Film manufacturing ID.
    pub film_mfg_id: String,
    /// Film type.
    pub film_type: String,
    /// Offset in perfs.
    pub offset: String,
    /// Prefix.
    pub prefix: String,
    /// Count.
    pub count: String,
    /// Format (e.g., "Academy").
    pub format: String,
    /// Frame position in sequence.
    pub frame_position: u32,
    /// Sequence length.
    pub sequence_length: u32,
    /// Held count.
    pub held_count: u32,
    /// Frame rate.
    pub frame_rate: f32,
    /// Shutter angle.
    pub shutter_angle: f32,
    /// Frame identification.
    pub frame_id: String,
    /// Slate information.
    pub slate_info: String,
}

/// Television information (128 bytes).
#[derive(Debug, Clone)]
pub struct TelevisionInfo {
    /// Time code.
    pub time_code: u32,
    /// User bits.
    pub user_bits: u32,
    /// Interlace (0 = progressive).
    pub interlace: u8,
    /// Field number.
    pub field_number: u8,
    /// Video signal standard.
    pub video_signal: u8,
    /// Horizontal sampling rate (Hz).
    pub horizontal_sample_rate: f32,
    /// Vertical sampling rate (Hz).
    pub vertical_sample_rate: f32,
    /// Frame rate.
    pub frame_rate: f32,
    /// Time offset.
    pub time_offset: f32,
    /// Gamma value.
    pub gamma: f32,
    /// Black level code value.
    pub black_level: f32,
    /// Black gain.
    pub black_gain: f32,
    /// Breakpoint.
    pub breakpoint: f32,
    /// White level.
    pub white_level: f32,
    /// Integration times.
    pub integration_times: f32,
}

impl Default for FileInfo {
    fn default() -> Self {
        Self {
            magic: DPX_MAGIC_BE,
            image_offset: 2048,
            version: "V2.0".to_string(),
            file_size: 0,
            ditto_key: 1,
            generic_size: 1664,
            industry_size: 384,
            user_size: 0,
            filename: String::new(),
            timestamp: String::new(),
            creator: "OxiMedia".to_string(),
            project: String::new(),
            copyright: String::new(),
            encrypt_key: 0xFFFF_FFFF,
        }
    }
}

impl Default for ImageInfo {
    fn default() -> Self {
        Self {
            orientation: 0,
            element_count: 1,
            width: 0,
            height: 0,
            elements: vec![ImageElement::default()],
        }
    }
}

impl Default for ImageElement {
    fn default() -> Self {
        Self {
            data_sign: 0,
            ref_low_data: 0,
            ref_low_quantity: 0.0,
            ref_high_data: 1023,
            ref_high_quantity: 2.046,
            descriptor: 50,  // RGB
            transfer: 2,     // Linear
            colorimetric: 1, // Printing density
            bit_depth: 10,
            packing: 1,
            encoding: 0,
            data_offset: 0,
            eol_padding: 0,
            eoi_padding: 0,
            description: String::new(),
        }
    }
}

impl Default for OrientationInfo {
    fn default() -> Self {
        Self {
            x_offset: 0,
            y_offset: 0,
            x_center: 0.0,
            y_center: 0.0,
            x_original_size: 0,
            y_original_size: 0,
            source_filename: String::new(),
            source_timestamp: String::new(),
            input_device: String::new(),
            input_serial: String::new(),
        }
    }
}

impl Default for FilmInfo {
    fn default() -> Self {
        Self {
            film_mfg_id: String::new(),
            film_type: String::new(),
            offset: String::new(),
            prefix: String::new(),
            count: String::new(),
            format: String::new(),
            frame_position: 0,
            sequence_length: 0,
            held_count: 0,
            frame_rate: 24.0,
            shutter_angle: 180.0,
            frame_id: String::new(),
            slate_info: String::new(),
        }
    }
}

impl Default for TelevisionInfo {
    fn default() -> Self {
        Self {
            time_code: 0,
            user_bits: 0,
            interlace: 0,
            field_number: 0,
            video_signal: 0,
            horizontal_sample_rate: 0.0,
            vertical_sample_rate: 0.0,
            frame_rate: 24.0,
            time_offset: 0.0,
            gamma: 2.2,
            black_level: 0.0,
            black_gain: 0.0,
            breakpoint: 0.0,
            white_level: 1.0,
            integration_times: 0.0,
        }
    }
}

/// Reads a DPX file.
///
/// # Arguments
///
/// * `path` - Path to the DPX file
/// * `frame_number` - Frame number for the image
///
/// # Errors
///
/// Returns an error if the file cannot be read or is invalid.
pub fn read_dpx(path: &Path, frame_number: u32) -> ImageResult<ImageFrame> {
    let mut file = File::open(path)?;

    // Read magic number to determine endianness
    let magic = file.read_u32::<BigEndian>()?;
    let endian = match magic {
        DPX_MAGIC_BE => Endian::Big,
        DPX_MAGIC_LE => Endian::Little,
        _ => return Err(ImageError::invalid_format("Invalid DPX magic number")),
    };

    // Seek back to start
    file.seek(SeekFrom::Start(0))?;

    // Read header
    let header = read_header(&mut file, endian)?;

    // Defend against an allocation bomb: the DPX header carries 32-bit
    // width/height that feed every `width*height*components` pixel-buffer
    // allocation in the readers below. Cap them once, here, against the shared
    // decode ceiling so a tiny header cannot request a multi-GB buffer.
    crate::limits::check_dimensions(header.image.width as usize, header.image.height as usize)
        .map_err(ImageError::InvalidFormat)?;

    // Validate
    if header.image.element_count == 0 || header.image.element_count > 8 {
        return Err(ImageError::invalid_format("Invalid element count"));
    }

    let element = &header.image.elements[0];

    // Determine pixel type and color space
    let pixel_type = match element.bit_depth {
        8 => PixelType::U8,
        10 => PixelType::U10,
        12 => PixelType::U12,
        16 => PixelType::U16,
        _ => {
            return Err(ImageError::unsupported(format!(
                "Unsupported bit depth: {}",
                element.bit_depth
            )))
        }
    };

    let (components, color_space) = match element.descriptor {
        1 => (1, ColorSpace::Luma),       // Luma
        6 => (3, ColorSpace::Luma),       // Luma (legacy)
        50 => (3, ColorSpace::LinearRgb), // RGB
        51 => (4, ColorSpace::LinearRgb), // RGBA
        52 => (4, ColorSpace::LinearRgb), // ABGR
        100 => (3, ColorSpace::YCbCr),    // CbYCrY (4:2:2)
        102 => (3, ColorSpace::YCbCr),    // CbYACrYA (4:2:2:4)
        _ => {
            return Err(ImageError::unsupported(format!(
                "Unsupported descriptor: {}",
                element.descriptor
            )))
        }
    };

    // Read image data
    file.seek(SeekFrom::Start(u64::from(header.file.image_offset)))?;

    let data = if element.packing == 0 {
        // Packed data
        read_packed_data(&mut file, &header, element, endian)?
    } else {
        // Filled data (each component in 32-bit word)
        read_filled_data(&mut file, &header, element, endian)?
    };

    let mut frame = ImageFrame::new(
        frame_number,
        header.image.width,
        header.image.height,
        pixel_type,
        components,
        color_space,
        ImageData::interleaved(data),
    );

    // Add metadata
    if !header.file.filename.is_empty() {
        frame.add_metadata("filename".to_string(), header.file.filename);
    }
    if !header.file.creator.is_empty() {
        frame.add_metadata("creator".to_string(), header.file.creator);
    }
    if !header.orientation.input_device.is_empty() {
        frame.add_metadata("camera".to_string(), header.orientation.input_device);
    }
    frame.add_metadata("transfer".to_string(), format!("{}", element.transfer));
    frame.add_metadata("bit_depth".to_string(), format!("{}", element.bit_depth));

    Ok(frame)
}

/// Writes a DPX file.
///
/// # Arguments
///
/// * `path` - Output path
/// * `frame` - Image frame to write
/// * `endian` - Byte order (big or little endian)
///
/// # Errors
///
/// Returns an error if the file cannot be written.
pub fn write_dpx(path: &Path, frame: &ImageFrame, endian: Endian) -> ImageResult<()> {
    let mut header = DpxHeader::default();

    // Set file info
    header.file.magic = match endian {
        Endian::Big => DPX_MAGIC_BE,
        Endian::Little => DPX_MAGIC_LE,
    };

    if let Some(name) = path.file_name() {
        header.file.filename = name.to_string_lossy().to_string();
    }

    // Set image info
    header.image.width = frame.width;
    header.image.height = frame.height;

    let mut element = ImageElement::default();
    element.bit_depth = frame.pixel_type.bit_depth();
    element.descriptor = match frame.components {
        1 => 1,  // Luma
        3 => 50, // RGB
        4 => 51, // RGBA
        _ => {
            return Err(ImageError::InvalidPixelFormat(
                "Unsupported component count".to_string(),
            ))
        }
    };

    // Set reference values based on bit depth
    element.ref_high_data = match element.bit_depth {
        8 => 255,
        10 => 1023,
        12 => 4095,
        16 => 65535,
        _ => 1023,
    };

    header.image.elements = vec![element.clone()];

    // Calculate file size
    let pixel_count = (frame.width * frame.height) as usize;
    let bytes_per_pixel = (frame.components as usize) * frame.pixel_type.bytes_per_component();
    let image_size = pixel_count * bytes_per_pixel;
    header.file.file_size = (header.file.image_offset as usize + image_size) as u32;

    // Write file
    let mut file = File::create(path)?;
    write_header(&mut file, &header, endian)?;

    // Write image data
    if let Some(data) = frame.data.as_slice() {
        if element.packing == 0 {
            // Write packed data
            write_packed_data(&mut file, data, &element, frame.width, frame.height, endian)?;
        } else {
            // Write filled data
            write_filled_data(&mut file, data, &element, frame.width, frame.height, endian)?;
        }
    } else {
        return Err(ImageError::unsupported("Planar data not supported for DPX"));
    }

    Ok(())
}

fn read_header(file: &mut File, endian: Endian) -> ImageResult<DpxHeader> {
    let mut header = DpxHeader::default();

    match endian {
        Endian::Big => read_header_be(file, &mut header)?,
        Endian::Little => read_header_le(file, &mut header)?,
    }

    Ok(header)
}

#[allow(clippy::too_many_lines)]
fn read_header_be(file: &mut File, header: &mut DpxHeader) -> ImageResult<()> {
    // File information (768 bytes)
    header.file.magic = file.read_u32::<BigEndian>()?;
    header.file.image_offset = file.read_u32::<BigEndian>()?;

    let mut version_buf = [0u8; 8];
    file.read_exact(&mut version_buf)?;
    header.file.version = String::from_utf8_lossy(&version_buf)
        .trim_end_matches('\0')
        .to_string();

    header.file.file_size = file.read_u32::<BigEndian>()?;
    header.file.ditto_key = file.read_u32::<BigEndian>()?;
    header.file.generic_size = file.read_u32::<BigEndian>()?;
    header.file.industry_size = file.read_u32::<BigEndian>()?;
    header.file.user_size = file.read_u32::<BigEndian>()?;

    let mut filename_buf = [0u8; 100];
    file.read_exact(&mut filename_buf)?;
    header.file.filename = String::from_utf8_lossy(&filename_buf)
        .trim_end_matches('\0')
        .to_string();

    let mut timestamp_buf = [0u8; 24];
    file.read_exact(&mut timestamp_buf)?;
    header.file.timestamp = String::from_utf8_lossy(&timestamp_buf)
        .trim_end_matches('\0')
        .to_string();

    let mut creator_buf = [0u8; 100];
    file.read_exact(&mut creator_buf)?;
    header.file.creator = String::from_utf8_lossy(&creator_buf)
        .trim_end_matches('\0')
        .to_string();

    let mut project_buf = [0u8; 200];
    file.read_exact(&mut project_buf)?;
    header.file.project = String::from_utf8_lossy(&project_buf)
        .trim_end_matches('\0')
        .to_string();

    let mut copyright_buf = [0u8; 200];
    file.read_exact(&mut copyright_buf)?;
    header.file.copyright = String::from_utf8_lossy(&copyright_buf)
        .trim_end_matches('\0')
        .to_string();

    header.file.encrypt_key = file.read_u32::<BigEndian>()?;

    // Skip reserved (104 bytes)
    file.seek(SeekFrom::Current(104))?;

    // Image information (640 bytes)
    header.image.orientation = file.read_u16::<BigEndian>()?;
    header.image.element_count = file.read_u16::<BigEndian>()?;
    header.image.width = file.read_u32::<BigEndian>()?;
    header.image.height = file.read_u32::<BigEndian>()?;

    // Read image elements (up to 8)
    header.image.elements.clear();
    for _ in 0..header.image.element_count {
        let mut element = ImageElement::default();
        element.data_sign = file.read_u32::<BigEndian>()?;
        element.ref_low_data = file.read_u32::<BigEndian>()?;
        element.ref_low_quantity = file.read_f32::<BigEndian>()?;
        element.ref_high_data = file.read_u32::<BigEndian>()?;
        element.ref_high_quantity = file.read_f32::<BigEndian>()?;
        element.descriptor = file.read_u8()?;
        element.transfer = file.read_u8()?;
        element.colorimetric = file.read_u8()?;
        element.bit_depth = file.read_u8()?;
        element.packing = file.read_u16::<BigEndian>()?;
        element.encoding = file.read_u16::<BigEndian>()?;
        element.data_offset = file.read_u32::<BigEndian>()?;
        element.eol_padding = file.read_u32::<BigEndian>()?;
        element.eoi_padding = file.read_u32::<BigEndian>()?;

        let mut desc_buf = [0u8; 32];
        file.read_exact(&mut desc_buf)?;
        element.description = String::from_utf8_lossy(&desc_buf)
            .trim_end_matches('\0')
            .to_string();

        header.image.elements.push(element);
    }

    // Skip remaining element slots
    let elements_to_skip = 8 - i64::from(header.image.element_count);
    file.seek(SeekFrom::Current(elements_to_skip * 72))?;

    // Skip reserved (52 bytes)
    file.seek(SeekFrom::Current(52))?;

    // Orientation information (256 bytes)
    header.orientation.x_offset = file.read_u32::<BigEndian>()?;
    header.orientation.y_offset = file.read_u32::<BigEndian>()?;
    header.orientation.x_center = file.read_f32::<BigEndian>()?;
    header.orientation.y_center = file.read_f32::<BigEndian>()?;
    header.orientation.x_original_size = file.read_u32::<BigEndian>()?;
    header.orientation.y_original_size = file.read_u32::<BigEndian>()?;

    let mut source_filename_buf = [0u8; 100];
    file.read_exact(&mut source_filename_buf)?;
    header.orientation.source_filename = String::from_utf8_lossy(&source_filename_buf)
        .trim_end_matches('\0')
        .to_string();

    let mut source_timestamp_buf = [0u8; 24];
    file.read_exact(&mut source_timestamp_buf)?;
    header.orientation.source_timestamp = String::from_utf8_lossy(&source_timestamp_buf)
        .trim_end_matches('\0')
        .to_string();

    let mut input_device_buf = [0u8; 32];
    file.read_exact(&mut input_device_buf)?;
    header.orientation.input_device = String::from_utf8_lossy(&input_device_buf)
        .trim_end_matches('\0')
        .to_string();

    let mut input_serial_buf = [0u8; 32];
    file.read_exact(&mut input_serial_buf)?;
    header.orientation.input_serial = String::from_utf8_lossy(&input_serial_buf)
        .trim_end_matches('\0')
        .to_string();

    // Skip reserved (64 bytes)
    file.seek(SeekFrom::Current(64))?;

    // Film information (256 bytes)
    let mut film_mfg_id_buf = [0u8; 2];
    file.read_exact(&mut film_mfg_id_buf)?;
    header.film.film_mfg_id = String::from_utf8_lossy(&film_mfg_id_buf)
        .trim_end_matches('\0')
        .to_string();

    let mut film_type_buf = [0u8; 2];
    file.read_exact(&mut film_type_buf)?;
    header.film.film_type = String::from_utf8_lossy(&film_type_buf)
        .trim_end_matches('\0')
        .to_string();

    let mut offset_buf = [0u8; 2];
    file.read_exact(&mut offset_buf)?;
    header.film.offset = String::from_utf8_lossy(&offset_buf)
        .trim_end_matches('\0')
        .to_string();

    let mut prefix_buf = [0u8; 6];
    file.read_exact(&mut prefix_buf)?;
    header.film.prefix = String::from_utf8_lossy(&prefix_buf)
        .trim_end_matches('\0')
        .to_string();

    let mut count_buf = [0u8; 4];
    file.read_exact(&mut count_buf)?;
    header.film.count = String::from_utf8_lossy(&count_buf)
        .trim_end_matches('\0')
        .to_string();

    let mut format_buf = [0u8; 32];
    file.read_exact(&mut format_buf)?;
    header.film.format = String::from_utf8_lossy(&format_buf)
        .trim_end_matches('\0')
        .to_string();

    header.film.frame_position = file.read_u32::<BigEndian>()?;
    header.film.sequence_length = file.read_u32::<BigEndian>()?;
    header.film.held_count = file.read_u32::<BigEndian>()?;
    header.film.frame_rate = file.read_f32::<BigEndian>()?;
    header.film.shutter_angle = file.read_f32::<BigEndian>()?;

    let mut frame_id_buf = [0u8; 32];
    file.read_exact(&mut frame_id_buf)?;
    header.film.frame_id = String::from_utf8_lossy(&frame_id_buf)
        .trim_end_matches('\0')
        .to_string();

    let mut slate_info_buf = [0u8; 100];
    file.read_exact(&mut slate_info_buf)?;
    header.film.slate_info = String::from_utf8_lossy(&slate_info_buf)
        .trim_end_matches('\0')
        .to_string();

    // Skip reserved (56 bytes)
    file.seek(SeekFrom::Current(56))?;

    // Television information (128 bytes)
    header.television.time_code = file.read_u32::<BigEndian>()?;
    header.television.user_bits = file.read_u32::<BigEndian>()?;
    header.television.interlace = file.read_u8()?;
    header.television.field_number = file.read_u8()?;
    header.television.video_signal = file.read_u8()?;

    // Skip padding (1 byte)
    file.seek(SeekFrom::Current(1))?;

    header.television.horizontal_sample_rate = file.read_f32::<BigEndian>()?;
    header.television.vertical_sample_rate = file.read_f32::<BigEndian>()?;
    header.television.frame_rate = file.read_f32::<BigEndian>()?;
    header.television.time_offset = file.read_f32::<BigEndian>()?;
    header.television.gamma = file.read_f32::<BigEndian>()?;
    header.television.black_level = file.read_f32::<BigEndian>()?;
    header.television.black_gain = file.read_f32::<BigEndian>()?;
    header.television.breakpoint = file.read_f32::<BigEndian>()?;
    header.television.white_level = file.read_f32::<BigEndian>()?;
    header.television.integration_times = file.read_f32::<BigEndian>()?;

    // Skip reserved (76 bytes)
    file.seek(SeekFrom::Current(76))?;

    Ok(())
}

#[allow(clippy::too_many_lines)]
fn read_header_le(file: &mut File, header: &mut DpxHeader) -> ImageResult<()> {
    // File information (768 bytes)
    header.file.magic = file.read_u32::<LittleEndian>()?;
    header.file.image_offset = file.read_u32::<LittleEndian>()?;

    let mut version_buf = [0u8; 8];
    file.read_exact(&mut version_buf)?;
    header.file.version = String::from_utf8_lossy(&version_buf)
        .trim_end_matches('\0')
        .to_string();

    header.file.file_size = file.read_u32::<LittleEndian>()?;
    header.file.ditto_key = file.read_u32::<LittleEndian>()?;
    header.file.generic_size = file.read_u32::<LittleEndian>()?;
    header.file.industry_size = file.read_u32::<LittleEndian>()?;
    header.file.user_size = file.read_u32::<LittleEndian>()?;

    let mut filename_buf = [0u8; 100];
    file.read_exact(&mut filename_buf)?;
    header.file.filename = String::from_utf8_lossy(&filename_buf)
        .trim_end_matches('\0')
        .to_string();

    let mut timestamp_buf = [0u8; 24];
    file.read_exact(&mut timestamp_buf)?;
    header.file.timestamp = String::from_utf8_lossy(&timestamp_buf)
        .trim_end_matches('\0')
        .to_string();

    let mut creator_buf = [0u8; 100];
    file.read_exact(&mut creator_buf)?;
    header.file.creator = String::from_utf8_lossy(&creator_buf)
        .trim_end_matches('\0')
        .to_string();

    let mut project_buf = [0u8; 200];
    file.read_exact(&mut project_buf)?;
    header.file.project = String::from_utf8_lossy(&project_buf)
        .trim_end_matches('\0')
        .to_string();

    let mut copyright_buf = [0u8; 200];
    file.read_exact(&mut copyright_buf)?;
    header.file.copyright = String::from_utf8_lossy(&copyright_buf)
        .trim_end_matches('\0')
        .to_string();

    header.file.encrypt_key = file.read_u32::<LittleEndian>()?;

    // Skip reserved (104 bytes)
    file.seek(SeekFrom::Current(104))?;

    // Image information (640 bytes)
    header.image.orientation = file.read_u16::<LittleEndian>()?;
    header.image.element_count = file.read_u16::<LittleEndian>()?;
    header.image.width = file.read_u32::<LittleEndian>()?;
    header.image.height = file.read_u32::<LittleEndian>()?;

    // Read image elements (up to 8)
    header.image.elements.clear();
    for _ in 0..header.image.element_count {
        let mut element = ImageElement::default();
        element.data_sign = file.read_u32::<LittleEndian>()?;
        element.ref_low_data = file.read_u32::<LittleEndian>()?;
        element.ref_low_quantity = file.read_f32::<LittleEndian>()?;
        element.ref_high_data = file.read_u32::<LittleEndian>()?;
        element.ref_high_quantity = file.read_f32::<LittleEndian>()?;
        element.descriptor = file.read_u8()?;
        element.transfer = file.read_u8()?;
        element.colorimetric = file.read_u8()?;
        element.bit_depth = file.read_u8()?;
        element.packing = file.read_u16::<LittleEndian>()?;
        element.encoding = file.read_u16::<LittleEndian>()?;
        element.data_offset = file.read_u32::<LittleEndian>()?;
        element.eol_padding = file.read_u32::<LittleEndian>()?;
        element.eoi_padding = file.read_u32::<LittleEndian>()?;

        let mut desc_buf = [0u8; 32];
        file.read_exact(&mut desc_buf)?;
        element.description = String::from_utf8_lossy(&desc_buf)
            .trim_end_matches('\0')
            .to_string();

        header.image.elements.push(element);
    }

    // Skip remaining element slots
    let elements_to_skip = 8 - i64::from(header.image.element_count);
    file.seek(SeekFrom::Current(elements_to_skip * 72))?;

    // Skip reserved (52 bytes)
    file.seek(SeekFrom::Current(52))?;

    // Orientation information (256 bytes)
    header.orientation.x_offset = file.read_u32::<LittleEndian>()?;
    header.orientation.y_offset = file.read_u32::<LittleEndian>()?;
    header.orientation.x_center = file.read_f32::<LittleEndian>()?;
    header.orientation.y_center = file.read_f32::<LittleEndian>()?;
    header.orientation.x_original_size = file.read_u32::<LittleEndian>()?;
    header.orientation.y_original_size = file.read_u32::<LittleEndian>()?;

    let mut source_filename_buf = [0u8; 100];
    file.read_exact(&mut source_filename_buf)?;
    header.orientation.source_filename = String::from_utf8_lossy(&source_filename_buf)
        .trim_end_matches('\0')
        .to_string();

    let mut source_timestamp_buf = [0u8; 24];
    file.read_exact(&mut source_timestamp_buf)?;
    header.orientation.source_timestamp = String::from_utf8_lossy(&source_timestamp_buf)
        .trim_end_matches('\0')
        .to_string();

    let mut input_device_buf = [0u8; 32];
    file.read_exact(&mut input_device_buf)?;
    header.orientation.input_device = String::from_utf8_lossy(&input_device_buf)
        .trim_end_matches('\0')
        .to_string();

    let mut input_serial_buf = [0u8; 32];
    file.read_exact(&mut input_serial_buf)?;
    header.orientation.input_serial = String::from_utf8_lossy(&input_serial_buf)
        .trim_end_matches('\0')
        .to_string();

    // Skip reserved (64 bytes)
    file.seek(SeekFrom::Current(64))?;

    // Film information (256 bytes)
    let mut film_mfg_id_buf = [0u8; 2];
    file.read_exact(&mut film_mfg_id_buf)?;
    header.film.film_mfg_id = String::from_utf8_lossy(&film_mfg_id_buf)
        .trim_end_matches('\0')
        .to_string();

    let mut film_type_buf = [0u8; 2];
    file.read_exact(&mut film_type_buf)?;
    header.film.film_type = String::from_utf8_lossy(&film_type_buf)
        .trim_end_matches('\0')
        .to_string();

    let mut offset_buf = [0u8; 2];
    file.read_exact(&mut offset_buf)?;
    header.film.offset = String::from_utf8_lossy(&offset_buf)
        .trim_end_matches('\0')
        .to_string();

    let mut prefix_buf = [0u8; 6];
    file.read_exact(&mut prefix_buf)?;
    header.film.prefix = String::from_utf8_lossy(&prefix_buf)
        .trim_end_matches('\0')
        .to_string();

    let mut count_buf = [0u8; 4];
    file.read_exact(&mut count_buf)?;
    header.film.count = String::from_utf8_lossy(&count_buf)
        .trim_end_matches('\0')
        .to_string();

    let mut format_buf = [0u8; 32];
    file.read_exact(&mut format_buf)?;
    header.film.format = String::from_utf8_lossy(&format_buf)
        .trim_end_matches('\0')
        .to_string();

    header.film.frame_position = file.read_u32::<LittleEndian>()?;
    header.film.sequence_length = file.read_u32::<LittleEndian>()?;
    header.film.held_count = file.read_u32::<LittleEndian>()?;
    header.film.frame_rate = file.read_f32::<LittleEndian>()?;
    header.film.shutter_angle = file.read_f32::<LittleEndian>()?;

    let mut frame_id_buf = [0u8; 32];
    file.read_exact(&mut frame_id_buf)?;
    header.film.frame_id = String::from_utf8_lossy(&frame_id_buf)
        .trim_end_matches('\0')
        .to_string();

    let mut slate_info_buf = [0u8; 100];
    file.read_exact(&mut slate_info_buf)?;
    header.film.slate_info = String::from_utf8_lossy(&slate_info_buf)
        .trim_end_matches('\0')
        .to_string();

    // Skip reserved (56 bytes)
    file.seek(SeekFrom::Current(56))?;

    // Television information (128 bytes)
    header.television.time_code = file.read_u32::<LittleEndian>()?;
    header.television.user_bits = file.read_u32::<LittleEndian>()?;
    header.television.interlace = file.read_u8()?;
    header.television.field_number = file.read_u8()?;
    header.television.video_signal = file.read_u8()?;

    // Skip padding (1 byte)
    file.seek(SeekFrom::Current(1))?;

    header.television.horizontal_sample_rate = file.read_f32::<LittleEndian>()?;
    header.television.vertical_sample_rate = file.read_f32::<LittleEndian>()?;
    header.television.frame_rate = file.read_f32::<LittleEndian>()?;
    header.television.time_offset = file.read_f32::<LittleEndian>()?;
    header.television.gamma = file.read_f32::<LittleEndian>()?;
    header.television.black_level = file.read_f32::<LittleEndian>()?;
    header.television.black_gain = file.read_f32::<LittleEndian>()?;
    header.television.breakpoint = file.read_f32::<LittleEndian>()?;
    header.television.white_level = file.read_f32::<LittleEndian>()?;
    header.television.integration_times = file.read_f32::<LittleEndian>()?;

    // Skip reserved (76 bytes)
    file.seek(SeekFrom::Current(76))?;

    Ok(())
}

fn read_packed_data(
    file: &mut File,
    header: &DpxHeader,
    element: &ImageElement,
    endian: Endian,
) -> ImageResult<Vec<u8>> {
    let width = header.image.width as usize;
    let height = header.image.height as usize;
    let components = match element.descriptor {
        1 | 6 => 1,
        50 => 3,
        51 | 52 => 4,
        100 | 102 => 3,
        _ => 3,
    };

    match element.bit_depth {
        8 => {
            let size = width * height * components;
            let mut data = vec![0u8; size];
            file.read_exact(&mut data)?;
            Ok(data)
        }
        10 => {
            // 10-bit packed: 3 components per 32-bit word
            let pixel_count = width * height * components;
            let words_needed = pixel_count.div_ceil(3);
            let mut raw = vec![0u8; words_needed * 4];
            file.read_exact(&mut raw)?;

            let method = determine_packing_method(element);
            unpack_10bit_packed(&raw, pixel_count, method, endian)
        }
        12 => {
            // For 12-bit, use bitstream unpacking
            let packed_size = (width * height * components * 12).div_ceil(8);
            let mut packed_data = vec![0u8; packed_size];
            file.read_exact(&mut packed_data)?;

            unpack_12bit(&packed_data, width * height * components)
        }
        16 => {
            let size = width * height * components * 2;
            let mut data = vec![0u8; size];
            file.read_exact(&mut data)?;
            Ok(data)
        }
        _ => Err(ImageError::unsupported(format!(
            "Unsupported bit depth: {}",
            element.bit_depth
        ))),
    }
}

/// Determines the packing method from the element's packing field.
/// packing == 0 is Method A (MSB-aligned), packing == 5 is Method B (LSB-aligned).
fn determine_packing_method(element: &ImageElement) -> PackingMethod {
    if element.packing == 5 {
        PackingMethod::MethodB
    } else {
        PackingMethod::MethodA
    }
}

/// Unpacks 10-bit packed data using Method A or Method B.
///
/// Method A: 3 components packed MSB-first in 32-bit word.
///   `[C0(bits 31-22) | C1(bits 21-12) | C2(bits 11-2) | pad(bits 1-0)]`
///
/// Method B: 3 components packed LSB-first in 32-bit word.
///   `[pad(bits 31-30) | C0(bits 29-20) | C1(bits 19-10) | C2(bits 9-0)]`
pub fn unpack_10bit_packed(
    packed: &[u8],
    count: usize,
    method: PackingMethod,
    endian: Endian,
) -> ImageResult<Vec<u8>> {
    let mut unpacked = vec![0u8; count * 2];
    let words = packed.len() / 4;
    let mut out_idx = 0;

    for w in 0..words {
        if out_idx >= count {
            break;
        }
        let offset = w * 4;
        let word = match endian {
            Endian::Big => u32::from_be_bytes([
                packed[offset],
                packed[offset + 1],
                packed[offset + 2],
                packed[offset + 3],
            ]),
            Endian::Little => u32::from_le_bytes([
                packed[offset],
                packed[offset + 1],
                packed[offset + 2],
                packed[offset + 3],
            ]),
        };

        let (c0, c1, c2) = match method {
            PackingMethod::MethodA => {
                // MSB-aligned: [C0(31..22) | C1(21..12) | C2(11..2) | pad(1..0)]
                let v0 = ((word >> 22) & 0x3FF) as u16;
                let v1 = ((word >> 12) & 0x3FF) as u16;
                let v2 = ((word >> 2) & 0x3FF) as u16;
                (v0, v1, v2)
            }
            PackingMethod::MethodB => {
                // LSB-aligned: [pad(31..30) | C0(29..20) | C1(19..10) | C2(9..0)]
                let v0 = ((word >> 20) & 0x3FF) as u16;
                let v1 = ((word >> 10) & 0x3FF) as u16;
                let v2 = (word & 0x3FF) as u16;
                (v0, v1, v2)
            }
        };

        for val in [c0, c1, c2] {
            if out_idx >= count {
                break;
            }
            unpacked[out_idx * 2] = (val >> 8) as u8;
            unpacked[out_idx * 2 + 1] = (val & 0xFF) as u8;
            out_idx += 1;
        }
    }

    Ok(unpacked)
}

/// Packs 10-bit data into 32-bit words using Method A or Method B.
///
/// Input: 16-bit big-endian values (only lower 10 bits used).
/// Output: packed 32-bit words.
pub fn pack_10bit(
    data: &[u8],
    count: usize,
    method: PackingMethod,
    endian: Endian,
) -> ImageResult<Vec<u8>> {
    let words_needed = count.div_ceil(3);
    let mut output = vec![0u8; words_needed * 4];

    let mut in_idx = 0;
    for w in 0..words_needed {
        let v0 = if in_idx < count {
            let val = u16::from_be_bytes([
                data.get(in_idx * 2).copied().unwrap_or(0),
                data.get(in_idx * 2 + 1).copied().unwrap_or(0),
            ]);
            in_idx += 1;
            u32::from(val & 0x3FF)
        } else {
            0
        };
        let v1 = if in_idx < count {
            let val = u16::from_be_bytes([
                data.get(in_idx * 2).copied().unwrap_or(0),
                data.get(in_idx * 2 + 1).copied().unwrap_or(0),
            ]);
            in_idx += 1;
            u32::from(val & 0x3FF)
        } else {
            0
        };
        let v2 = if in_idx < count {
            let val = u16::from_be_bytes([
                data.get(in_idx * 2).copied().unwrap_or(0),
                data.get(in_idx * 2 + 1).copied().unwrap_or(0),
            ]);
            in_idx += 1;
            u32::from(val & 0x3FF)
        } else {
            0
        };

        let word = match method {
            PackingMethod::MethodA => (v0 << 22) | (v1 << 12) | (v2 << 2),
            PackingMethod::MethodB => (v0 << 20) | (v1 << 10) | v2,
        };

        let offset = w * 4;
        let bytes = match endian {
            Endian::Big => word.to_be_bytes(),
            Endian::Little => word.to_le_bytes(),
        };
        output[offset..offset + 4].copy_from_slice(&bytes);
    }

    Ok(output)
}

fn read_filled_data(
    file: &mut File,
    header: &DpxHeader,
    element: &ImageElement,
    endian: Endian,
) -> ImageResult<Vec<u8>> {
    let width = header.image.width as usize;
    let height = header.image.height as usize;
    let components = match element.descriptor {
        1 | 6 => 1,
        50 => 3,
        51 | 52 => 4,
        100 | 102 => 3,
        _ => 3,
    };

    let pixel_count = width * height * components;
    let mut data = vec![0u8; pixel_count * 2]; // 16-bit output

    for i in 0..pixel_count {
        let value = match endian {
            Endian::Big => file.read_u32::<BigEndian>()?,
            Endian::Little => file.read_u32::<LittleEndian>()?,
        };

        // Extract the relevant bits based on bit depth
        let shifted = value >> (32 - element.bit_depth);
        let value_16 = (shifted & 0xFFFF) as u16;

        data[i * 2] = (value_16 >> 8) as u8;
        data[i * 2 + 1] = (value_16 & 0xFF) as u8;
    }

    Ok(data)
}

/// Unpacks 12-bit bitstream data to 16-bit values.
fn unpack_12bit(packed: &[u8], count: usize) -> ImageResult<Vec<u8>> {
    let mut unpacked = vec![0u8; count * 2];
    let mut bit_pos = 0usize;

    for i in 0..count {
        let byte_pos = bit_pos / 8;
        let bit_offset = bit_pos % 8;

        if byte_pos + 1 >= packed.len() {
            break;
        }

        let value = if bit_offset <= 4 {
            let val = (u16::from(packed[byte_pos]) << (8 + bit_offset as u16))
                | (u16::from(packed[byte_pos + 1]) << (bit_offset as u16));
            (val >> 4) & 0xFFF
        } else {
            if byte_pos + 2 >= packed.len() {
                break;
            }
            let val = (u32::from(packed[byte_pos]) << 16)
                | (u32::from(packed[byte_pos + 1]) << 8)
                | u32::from(packed[byte_pos + 2]);
            ((val >> (12 - bit_offset)) & 0xFFF) as u16
        };

        unpacked[i * 2] = (value >> 8) as u8;
        unpacked[i * 2 + 1] = (value & 0xFF) as u8;
        bit_pos += 12;
    }

    Ok(unpacked)
}

#[allow(clippy::too_many_arguments)]
fn write_header(file: &mut File, header: &DpxHeader, endian: Endian) -> ImageResult<()> {
    match endian {
        Endian::Big => write_header_be(file, header)?,
        Endian::Little => write_header_le(file, header)?,
    }

    // Pad to 2048 bytes
    let current_pos = file.stream_position()?;
    if current_pos < 2048 {
        let padding = vec![0u8; (2048 - current_pos) as usize];
        file.write_all(&padding)?;
    }

    Ok(())
}

#[allow(clippy::too_many_lines)]
fn write_header_be(file: &mut File, header: &DpxHeader) -> ImageResult<()> {
    // File information
    file.write_u32::<BigEndian>(header.file.magic)?;
    file.write_u32::<BigEndian>(header.file.image_offset)?;

    let mut version_buf = [0u8; 8];
    version_buf[..header.file.version.len().min(8)]
        .copy_from_slice(&header.file.version.as_bytes()[..header.file.version.len().min(8)]);
    file.write_all(&version_buf)?;

    file.write_u32::<BigEndian>(header.file.file_size)?;
    file.write_u32::<BigEndian>(header.file.ditto_key)?;
    file.write_u32::<BigEndian>(header.file.generic_size)?;
    file.write_u32::<BigEndian>(header.file.industry_size)?;
    file.write_u32::<BigEndian>(header.file.user_size)?;

    let mut filename_buf = [0u8; 100];
    filename_buf[..header.file.filename.len().min(100)]
        .copy_from_slice(&header.file.filename.as_bytes()[..header.file.filename.len().min(100)]);
    file.write_all(&filename_buf)?;

    let mut timestamp_buf = [0u8; 24];
    timestamp_buf[..header.file.timestamp.len().min(24)]
        .copy_from_slice(&header.file.timestamp.as_bytes()[..header.file.timestamp.len().min(24)]);
    file.write_all(&timestamp_buf)?;

    let mut creator_buf = [0u8; 100];
    creator_buf[..header.file.creator.len().min(100)]
        .copy_from_slice(&header.file.creator.as_bytes()[..header.file.creator.len().min(100)]);
    file.write_all(&creator_buf)?;

    let mut project_buf = [0u8; 200];
    project_buf[..header.file.project.len().min(200)]
        .copy_from_slice(&header.file.project.as_bytes()[..header.file.project.len().min(200)]);
    file.write_all(&project_buf)?;

    let mut copyright_buf = [0u8; 200];
    copyright_buf[..header.file.copyright.len().min(200)]
        .copy_from_slice(&header.file.copyright.as_bytes()[..header.file.copyright.len().min(200)]);
    file.write_all(&copyright_buf)?;

    file.write_u32::<BigEndian>(header.file.encrypt_key)?;

    // Reserved (104 bytes)
    file.write_all(&[0u8; 104])?;

    // Image information
    file.write_u16::<BigEndian>(header.image.orientation)?;
    file.write_u16::<BigEndian>(header.image.element_count)?;
    file.write_u32::<BigEndian>(header.image.width)?;
    file.write_u32::<BigEndian>(header.image.height)?;

    // Write image elements
    for element in &header.image.elements {
        file.write_u32::<BigEndian>(element.data_sign)?;
        file.write_u32::<BigEndian>(element.ref_low_data)?;
        file.write_f32::<BigEndian>(element.ref_low_quantity)?;
        file.write_u32::<BigEndian>(element.ref_high_data)?;
        file.write_f32::<BigEndian>(element.ref_high_quantity)?;
        file.write_u8(element.descriptor)?;
        file.write_u8(element.transfer)?;
        file.write_u8(element.colorimetric)?;
        file.write_u8(element.bit_depth)?;
        file.write_u16::<BigEndian>(element.packing)?;
        file.write_u16::<BigEndian>(element.encoding)?;
        file.write_u32::<BigEndian>(element.data_offset)?;
        file.write_u32::<BigEndian>(element.eol_padding)?;
        file.write_u32::<BigEndian>(element.eoi_padding)?;

        let mut desc_buf = [0u8; 32];
        desc_buf[..element.description.len().min(32)]
            .copy_from_slice(&element.description.as_bytes()[..element.description.len().min(32)]);
        file.write_all(&desc_buf)?;
    }

    // Pad remaining element slots
    let elements_to_pad = 8 - header.image.element_count as usize;
    file.write_all(&vec![0u8; elements_to_pad * 72])?;

    // Reserved (52 bytes)
    file.write_all(&[0u8; 52])?;

    // Orientation information
    file.write_u32::<BigEndian>(header.orientation.x_offset)?;
    file.write_u32::<BigEndian>(header.orientation.y_offset)?;
    file.write_f32::<BigEndian>(header.orientation.x_center)?;
    file.write_f32::<BigEndian>(header.orientation.y_center)?;
    file.write_u32::<BigEndian>(header.orientation.x_original_size)?;
    file.write_u32::<BigEndian>(header.orientation.y_original_size)?;

    let mut source_filename_buf = [0u8; 100];
    source_filename_buf[..header.orientation.source_filename.len().min(100)].copy_from_slice(
        &header.orientation.source_filename.as_bytes()
            [..header.orientation.source_filename.len().min(100)],
    );
    file.write_all(&source_filename_buf)?;

    let mut source_timestamp_buf = [0u8; 24];
    source_timestamp_buf[..header.orientation.source_timestamp.len().min(24)].copy_from_slice(
        &header.orientation.source_timestamp.as_bytes()
            [..header.orientation.source_timestamp.len().min(24)],
    );
    file.write_all(&source_timestamp_buf)?;

    let mut input_device_buf = [0u8; 32];
    input_device_buf[..header.orientation.input_device.len().min(32)].copy_from_slice(
        &header.orientation.input_device.as_bytes()
            [..header.orientation.input_device.len().min(32)],
    );
    file.write_all(&input_device_buf)?;

    let mut input_serial_buf = [0u8; 32];
    input_serial_buf[..header.orientation.input_serial.len().min(32)].copy_from_slice(
        &header.orientation.input_serial.as_bytes()
            [..header.orientation.input_serial.len().min(32)],
    );
    file.write_all(&input_serial_buf)?;

    // Reserved (64 bytes)
    file.write_all(&[0u8; 64])?;

    // Film information
    let mut film_mfg_id_buf = [0u8; 2];
    film_mfg_id_buf[..header.film.film_mfg_id.len().min(2)].copy_from_slice(
        &header.film.film_mfg_id.as_bytes()[..header.film.film_mfg_id.len().min(2)],
    );
    file.write_all(&film_mfg_id_buf)?;

    let mut film_type_buf = [0u8; 2];
    film_type_buf[..header.film.film_type.len().min(2)]
        .copy_from_slice(&header.film.film_type.as_bytes()[..header.film.film_type.len().min(2)]);
    file.write_all(&film_type_buf)?;

    let mut offset_buf = [0u8; 2];
    offset_buf[..header.film.offset.len().min(2)]
        .copy_from_slice(&header.film.offset.as_bytes()[..header.film.offset.len().min(2)]);
    file.write_all(&offset_buf)?;

    let mut prefix_buf = [0u8; 6];
    prefix_buf[..header.film.prefix.len().min(6)]
        .copy_from_slice(&header.film.prefix.as_bytes()[..header.film.prefix.len().min(6)]);
    file.write_all(&prefix_buf)?;

    let mut count_buf = [0u8; 4];
    count_buf[..header.film.count.len().min(4)]
        .copy_from_slice(&header.film.count.as_bytes()[..header.film.count.len().min(4)]);
    file.write_all(&count_buf)?;

    let mut format_buf = [0u8; 32];
    format_buf[..header.film.format.len().min(32)]
        .copy_from_slice(&header.film.format.as_bytes()[..header.film.format.len().min(32)]);
    file.write_all(&format_buf)?;

    file.write_u32::<BigEndian>(header.film.frame_position)?;
    file.write_u32::<BigEndian>(header.film.sequence_length)?;
    file.write_u32::<BigEndian>(header.film.held_count)?;
    file.write_f32::<BigEndian>(header.film.frame_rate)?;
    file.write_f32::<BigEndian>(header.film.shutter_angle)?;

    let mut frame_id_buf = [0u8; 32];
    frame_id_buf[..header.film.frame_id.len().min(32)]
        .copy_from_slice(&header.film.frame_id.as_bytes()[..header.film.frame_id.len().min(32)]);
    file.write_all(&frame_id_buf)?;

    let mut slate_info_buf = [0u8; 100];
    slate_info_buf[..header.film.slate_info.len().min(100)].copy_from_slice(
        &header.film.slate_info.as_bytes()[..header.film.slate_info.len().min(100)],
    );
    file.write_all(&slate_info_buf)?;

    // Reserved (56 bytes)
    file.write_all(&[0u8; 56])?;

    // Television information
    file.write_u32::<BigEndian>(header.television.time_code)?;
    file.write_u32::<BigEndian>(header.television.user_bits)?;
    file.write_u8(header.television.interlace)?;
    file.write_u8(header.television.field_number)?;
    file.write_u8(header.television.video_signal)?;
    file.write_u8(0)?; // Padding

    file.write_f32::<BigEndian>(header.television.horizontal_sample_rate)?;
    file.write_f32::<BigEndian>(header.television.vertical_sample_rate)?;
    file.write_f32::<BigEndian>(header.television.frame_rate)?;
    file.write_f32::<BigEndian>(header.television.time_offset)?;
    file.write_f32::<BigEndian>(header.television.gamma)?;
    file.write_f32::<BigEndian>(header.television.black_level)?;
    file.write_f32::<BigEndian>(header.television.black_gain)?;
    file.write_f32::<BigEndian>(header.television.breakpoint)?;
    file.write_f32::<BigEndian>(header.television.white_level)?;
    file.write_f32::<BigEndian>(header.television.integration_times)?;

    // Reserved (76 bytes)
    file.write_all(&[0u8; 76])?;

    Ok(())
}

#[allow(clippy::too_many_lines)]
fn write_header_le(file: &mut File, header: &DpxHeader) -> ImageResult<()> {
    // File information
    file.write_u32::<LittleEndian>(header.file.magic)?;
    file.write_u32::<LittleEndian>(header.file.image_offset)?;

    let mut version_buf = [0u8; 8];
    version_buf[..header.file.version.len().min(8)]
        .copy_from_slice(&header.file.version.as_bytes()[..header.file.version.len().min(8)]);
    file.write_all(&version_buf)?;

    file.write_u32::<LittleEndian>(header.file.file_size)?;
    file.write_u32::<LittleEndian>(header.file.ditto_key)?;
    file.write_u32::<LittleEndian>(header.file.generic_size)?;
    file.write_u32::<LittleEndian>(header.file.industry_size)?;
    file.write_u32::<LittleEndian>(header.file.user_size)?;

    let mut filename_buf = [0u8; 100];
    filename_buf[..header.file.filename.len().min(100)]
        .copy_from_slice(&header.file.filename.as_bytes()[..header.file.filename.len().min(100)]);
    file.write_all(&filename_buf)?;

    let mut timestamp_buf = [0u8; 24];
    timestamp_buf[..header.file.timestamp.len().min(24)]
        .copy_from_slice(&header.file.timestamp.as_bytes()[..header.file.timestamp.len().min(24)]);
    file.write_all(&timestamp_buf)?;

    let mut creator_buf = [0u8; 100];
    creator_buf[..header.file.creator.len().min(100)]
        .copy_from_slice(&header.file.creator.as_bytes()[..header.file.creator.len().min(100)]);
    file.write_all(&creator_buf)?;

    let mut project_buf = [0u8; 200];
    project_buf[..header.file.project.len().min(200)]
        .copy_from_slice(&header.file.project.as_bytes()[..header.file.project.len().min(200)]);
    file.write_all(&project_buf)?;

    let mut copyright_buf = [0u8; 200];
    copyright_buf[..header.file.copyright.len().min(200)]
        .copy_from_slice(&header.file.copyright.as_bytes()[..header.file.copyright.len().min(200)]);
    file.write_all(&copyright_buf)?;

    file.write_u32::<LittleEndian>(header.file.encrypt_key)?;

    // Reserved (104 bytes)
    file.write_all(&[0u8; 104])?;

    // Image information
    file.write_u16::<LittleEndian>(header.image.orientation)?;
    file.write_u16::<LittleEndian>(header.image.element_count)?;
    file.write_u32::<LittleEndian>(header.image.width)?;
    file.write_u32::<LittleEndian>(header.image.height)?;

    // Write image elements
    for element in &header.image.elements {
        file.write_u32::<LittleEndian>(element.data_sign)?;
        file.write_u32::<LittleEndian>(element.ref_low_data)?;
        file.write_f32::<LittleEndian>(element.ref_low_quantity)?;
        file.write_u32::<LittleEndian>(element.ref_high_data)?;
        file.write_f32::<LittleEndian>(element.ref_high_quantity)?;
        file.write_u8(element.descriptor)?;
        file.write_u8(element.transfer)?;
        file.write_u8(element.colorimetric)?;
        file.write_u8(element.bit_depth)?;
        file.write_u16::<LittleEndian>(element.packing)?;
        file.write_u16::<LittleEndian>(element.encoding)?;
        file.write_u32::<LittleEndian>(element.data_offset)?;
        file.write_u32::<LittleEndian>(element.eol_padding)?;
        file.write_u32::<LittleEndian>(element.eoi_padding)?;

        let mut desc_buf = [0u8; 32];
        desc_buf[..element.description.len().min(32)]
            .copy_from_slice(&element.description.as_bytes()[..element.description.len().min(32)]);
        file.write_all(&desc_buf)?;
    }

    // Pad remaining element slots
    let elements_to_pad = 8 - header.image.element_count as usize;
    file.write_all(&vec![0u8; elements_to_pad * 72])?;

    // Reserved (52 bytes)
    file.write_all(&[0u8; 52])?;

    // Orientation information
    file.write_u32::<LittleEndian>(header.orientation.x_offset)?;
    file.write_u32::<LittleEndian>(header.orientation.y_offset)?;
    file.write_f32::<LittleEndian>(header.orientation.x_center)?;
    file.write_f32::<LittleEndian>(header.orientation.y_center)?;
    file.write_u32::<LittleEndian>(header.orientation.x_original_size)?;
    file.write_u32::<LittleEndian>(header.orientation.y_original_size)?;

    let mut source_filename_buf = [0u8; 100];
    source_filename_buf[..header.orientation.source_filename.len().min(100)].copy_from_slice(
        &header.orientation.source_filename.as_bytes()
            [..header.orientation.source_filename.len().min(100)],
    );
    file.write_all(&source_filename_buf)?;

    let mut source_timestamp_buf = [0u8; 24];
    source_timestamp_buf[..header.orientation.source_timestamp.len().min(24)].copy_from_slice(
        &header.orientation.source_timestamp.as_bytes()
            [..header.orientation.source_timestamp.len().min(24)],
    );
    file.write_all(&source_timestamp_buf)?;

    let mut input_device_buf = [0u8; 32];
    input_device_buf[..header.orientation.input_device.len().min(32)].copy_from_slice(
        &header.orientation.input_device.as_bytes()
            [..header.orientation.input_device.len().min(32)],
    );
    file.write_all(&input_device_buf)?;

    let mut input_serial_buf = [0u8; 32];
    input_serial_buf[..header.orientation.input_serial.len().min(32)].copy_from_slice(
        &header.orientation.input_serial.as_bytes()
            [..header.orientation.input_serial.len().min(32)],
    );
    file.write_all(&input_serial_buf)?;

    // Reserved (64 bytes)
    file.write_all(&[0u8; 64])?;

    // Film information
    let mut film_mfg_id_buf = [0u8; 2];
    film_mfg_id_buf[..header.film.film_mfg_id.len().min(2)].copy_from_slice(
        &header.film.film_mfg_id.as_bytes()[..header.film.film_mfg_id.len().min(2)],
    );
    file.write_all(&film_mfg_id_buf)?;

    let mut film_type_buf = [0u8; 2];
    film_type_buf[..header.film.film_type.len().min(2)]
        .copy_from_slice(&header.film.film_type.as_bytes()[..header.film.film_type.len().min(2)]);
    file.write_all(&film_type_buf)?;

    let mut offset_buf = [0u8; 2];
    offset_buf[..header.film.offset.len().min(2)]
        .copy_from_slice(&header.film.offset.as_bytes()[..header.film.offset.len().min(2)]);
    file.write_all(&offset_buf)?;

    let mut prefix_buf = [0u8; 6];
    prefix_buf[..header.film.prefix.len().min(6)]
        .copy_from_slice(&header.film.prefix.as_bytes()[..header.film.prefix.len().min(6)]);
    file.write_all(&prefix_buf)?;

    let mut count_buf = [0u8; 4];
    count_buf[..header.film.count.len().min(4)]
        .copy_from_slice(&header.film.count.as_bytes()[..header.film.count.len().min(4)]);
    file.write_all(&count_buf)?;

    let mut format_buf = [0u8; 32];
    format_buf[..header.film.format.len().min(32)]
        .copy_from_slice(&header.film.format.as_bytes()[..header.film.format.len().min(32)]);
    file.write_all(&format_buf)?;

    file.write_u32::<LittleEndian>(header.film.frame_position)?;
    file.write_u32::<LittleEndian>(header.film.sequence_length)?;
    file.write_u32::<LittleEndian>(header.film.held_count)?;
    file.write_f32::<LittleEndian>(header.film.frame_rate)?;
    file.write_f32::<LittleEndian>(header.film.shutter_angle)?;

    let mut frame_id_buf = [0u8; 32];
    frame_id_buf[..header.film.frame_id.len().min(32)]
        .copy_from_slice(&header.film.frame_id.as_bytes()[..header.film.frame_id.len().min(32)]);
    file.write_all(&frame_id_buf)?;

    let mut slate_info_buf = [0u8; 100];
    slate_info_buf[..header.film.slate_info.len().min(100)].copy_from_slice(
        &header.film.slate_info.as_bytes()[..header.film.slate_info.len().min(100)],
    );
    file.write_all(&slate_info_buf)?;

    // Reserved (56 bytes)
    file.write_all(&[0u8; 56])?;

    // Television information
    file.write_u32::<LittleEndian>(header.television.time_code)?;
    file.write_u32::<LittleEndian>(header.television.user_bits)?;
    file.write_u8(header.television.interlace)?;
    file.write_u8(header.television.field_number)?;
    file.write_u8(header.television.video_signal)?;
    file.write_u8(0)?; // Padding

    file.write_f32::<LittleEndian>(header.television.horizontal_sample_rate)?;
    file.write_f32::<LittleEndian>(header.television.vertical_sample_rate)?;
    file.write_f32::<LittleEndian>(header.television.frame_rate)?;
    file.write_f32::<LittleEndian>(header.television.time_offset)?;
    file.write_f32::<LittleEndian>(header.television.gamma)?;
    file.write_f32::<LittleEndian>(header.television.black_level)?;
    file.write_f32::<LittleEndian>(header.television.black_gain)?;
    file.write_f32::<LittleEndian>(header.television.breakpoint)?;
    file.write_f32::<LittleEndian>(header.television.white_level)?;
    file.write_f32::<LittleEndian>(header.television.integration_times)?;

    // Reserved (76 bytes)
    file.write_all(&[0u8; 76])?;

    Ok(())
}

fn write_packed_data(
    file: &mut File,
    data: &[u8],
    element: &ImageElement,
    width: u32,
    height: u32,
    endian: Endian,
) -> ImageResult<()> {
    let components = match element.descriptor {
        1 | 6 => 1usize,
        50 => 3,
        51 | 52 => 4,
        100 | 102 => 3,
        _ => 3,
    };
    let pixel_count = (width as usize) * (height as usize) * components;

    match element.bit_depth {
        8 => {
            let size = pixel_count;
            if data.len() < size {
                return Err(ImageError::invalid_format(
                    "Insufficient data for packed write",
                ));
            }
            file.write_all(&data[..size])?;
        }
        10 => {
            let method = determine_packing_method(element);
            let packed = pack_10bit(data, pixel_count, method, endian)?;
            file.write_all(&packed)?;
        }
        16 => {
            let size = pixel_count * 2;
            if data.len() < size {
                return Err(ImageError::invalid_format(
                    "Insufficient data for packed write",
                ));
            }
            file.write_all(&data[..size])?;
        }
        _ => {
            return Err(ImageError::unsupported(format!(
                "Packed write for bit depth {} not supported",
                element.bit_depth
            )));
        }
    }

    Ok(())
}

fn write_filled_data(
    file: &mut File,
    data: &[u8],
    element: &ImageElement,
    _width: u32,
    _height: u32,
    endian: Endian,
) -> ImageResult<()> {
    // Write filled data (each component in 32-bit word)
    let shift = 32 - element.bit_depth;

    for chunk in data.chunks(2) {
        if chunk.len() == 2 {
            let value = u32::from(chunk[0]) << 8 | u32::from(chunk[1]);
            let shifted = value << shift;

            match endian {
                Endian::Big => file.write_u32::<BigEndian>(shifted)?,
                Endian::Little => file.write_u32::<LittleEndian>(shifted)?,
            }
        }
    }

    Ok(())
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_dpx_rejects_oversized_dimensions() {
        // Regression: the DPX header carries 32-bit width/height that size every
        // pixel-buffer allocation. A header declaring enormous dimensions must
        // be rejected (allocation bomb) rather than requesting multiple GB.
        let mut buf = vec![0u8; 4096];
        // Magic "SDPX" (big-endian) selects the big-endian header reader.
        buf[0..4].copy_from_slice(&DPX_MAGIC_BE.to_be_bytes());
        // width @ offset 772, height @ offset 776 (big-endian), both enormous.
        buf[772..776].copy_from_slice(&0xFFFF_FFFFu32.to_be_bytes());
        buf[776..780].copy_from_slice(&0xFFFF_FFFFu32.to_be_bytes());

        let path = std::env::temp_dir().join("oximedia_dpx_oversized_test.dpx");
        std::fs::write(&path, &buf).expect("write temp dpx");
        let result = read_dpx(&path, 0);
        let _ = std::fs::remove_file(&path);
        assert!(result.is_err(), "oversized DPX dimensions must be rejected");
    }

    // --- 10-bit packing Method A ---

    #[test]
    fn test_pack_unpack_10bit_method_a_roundtrip() {
        // 3 components: 100, 512, 1023
        let values: Vec<u16> = vec![100, 512, 1023];
        let mut data = Vec::with_capacity(values.len() * 2);
        for v in &values {
            data.extend_from_slice(&v.to_be_bytes());
        }

        let packed =
            pack_10bit(&data, 3, PackingMethod::MethodA, Endian::Big).expect("pack should succeed");
        assert_eq!(packed.len(), 4); // 1 word for 3 components

        let unpacked = unpack_10bit_packed(&packed, 3, PackingMethod::MethodA, Endian::Big)
            .expect("unpack should succeed");

        for (i, &expected) in values.iter().enumerate() {
            let got = u16::from_be_bytes([unpacked[i * 2], unpacked[i * 2 + 1]]);
            assert_eq!(got, expected, "component {i} mismatch");
        }
    }

    #[test]
    fn test_pack_unpack_10bit_method_b_roundtrip() {
        let values: Vec<u16> = vec![0, 511, 1023];
        let mut data = Vec::with_capacity(values.len() * 2);
        for v in &values {
            data.extend_from_slice(&v.to_be_bytes());
        }

        let packed =
            pack_10bit(&data, 3, PackingMethod::MethodB, Endian::Big).expect("pack should succeed");
        let unpacked = unpack_10bit_packed(&packed, 3, PackingMethod::MethodB, Endian::Big)
            .expect("unpack should succeed");

        for (i, &expected) in values.iter().enumerate() {
            let got = u16::from_be_bytes([unpacked[i * 2], unpacked[i * 2 + 1]]);
            assert_eq!(got, expected, "component {i} mismatch");
        }
    }

    #[test]
    fn test_pack_10bit_method_a_bit_layout() {
        // Method A: [C0(31..22) | C1(21..12) | C2(11..2) | pad(1..0)]
        let values: Vec<u16> = vec![0x3FF, 0x000, 0x155]; // all-ones, zero, pattern
        let mut data = Vec::new();
        for v in &values {
            data.extend_from_slice(&v.to_be_bytes());
        }

        let packed =
            pack_10bit(&data, 3, PackingMethod::MethodA, Endian::Big).expect("pack should succeed");
        let word = u32::from_be_bytes([packed[0], packed[1], packed[2], packed[3]]);

        assert_eq!((word >> 22) & 0x3FF, 0x3FF);
        assert_eq!((word >> 12) & 0x3FF, 0x000);
        assert_eq!((word >> 2) & 0x3FF, 0x155);
        assert_eq!(word & 0x3, 0); // padding bits
    }

    #[test]
    fn test_pack_10bit_method_b_bit_layout() {
        // Method B: [pad(31..30) | C0(29..20) | C1(19..10) | C2(9..0)]
        let values: Vec<u16> = vec![0x3FF, 0x000, 0x155];
        let mut data = Vec::new();
        for v in &values {
            data.extend_from_slice(&v.to_be_bytes());
        }

        let packed =
            pack_10bit(&data, 3, PackingMethod::MethodB, Endian::Big).expect("pack should succeed");
        let word = u32::from_be_bytes([packed[0], packed[1], packed[2], packed[3]]);

        assert_eq!((word >> 30) & 0x3, 0); // padding bits
        assert_eq!((word >> 20) & 0x3FF, 0x3FF);
        assert_eq!((word >> 10) & 0x3FF, 0x000);
        assert_eq!(word & 0x3FF, 0x155);
    }

    #[test]
    fn test_pack_unpack_10bit_little_endian() {
        let values: Vec<u16> = vec![300, 600, 900];
        let mut data = Vec::new();
        for v in &values {
            data.extend_from_slice(&v.to_be_bytes());
        }

        let packed = pack_10bit(&data, 3, PackingMethod::MethodA, Endian::Little)
            .expect("pack should succeed");
        let unpacked = unpack_10bit_packed(&packed, 3, PackingMethod::MethodA, Endian::Little)
            .expect("unpack should succeed");

        for (i, &expected) in values.iter().enumerate() {
            let got = u16::from_be_bytes([unpacked[i * 2], unpacked[i * 2 + 1]]);
            assert_eq!(got, expected, "LE component {i} mismatch");
        }
    }

    #[test]
    fn test_pack_unpack_10bit_multiple_words() {
        // 9 components = 3 words
        let values: Vec<u16> = vec![1, 2, 3, 100, 200, 300, 500, 700, 1000];
        let mut data = Vec::new();
        for v in &values {
            data.extend_from_slice(&v.to_be_bytes());
        }

        for method in [PackingMethod::MethodA, PackingMethod::MethodB] {
            let packed = pack_10bit(&data, 9, method, Endian::Big).expect("pack should succeed");
            assert_eq!(packed.len(), 12); // 3 words * 4 bytes

            let unpacked = unpack_10bit_packed(&packed, 9, method, Endian::Big)
                .expect("unpack should succeed");

            for (i, &expected) in values.iter().enumerate() {
                let got = u16::from_be_bytes([unpacked[i * 2], unpacked[i * 2 + 1]]);
                assert_eq!(got, expected, "{method:?} component {i} mismatch");
            }
        }
    }

    #[test]
    fn test_pack_10bit_partial_word() {
        // 2 components (not multiple of 3) - last word has padding
        let values: Vec<u16> = vec![400, 800];
        let mut data = Vec::new();
        for v in &values {
            data.extend_from_slice(&v.to_be_bytes());
        }

        let packed =
            pack_10bit(&data, 2, PackingMethod::MethodA, Endian::Big).expect("pack should succeed");
        let unpacked = unpack_10bit_packed(&packed, 2, PackingMethod::MethodA, Endian::Big)
            .expect("unpack should succeed");

        for (i, &expected) in values.iter().enumerate() {
            let got = u16::from_be_bytes([unpacked[i * 2], unpacked[i * 2 + 1]]);
            assert_eq!(got, expected, "partial word component {i} mismatch");
        }
    }

    #[test]
    fn test_packing_method_default() {
        assert_eq!(PackingMethod::default(), PackingMethod::MethodA);
    }

    #[test]
    fn test_determine_packing_method() {
        let mut elem = ImageElement::default();
        elem.packing = 0;
        assert_eq!(determine_packing_method(&elem), PackingMethod::MethodA);

        elem.packing = 5;
        assert_eq!(determine_packing_method(&elem), PackingMethod::MethodB);
    }

    #[test]
    fn test_pack_10bit_all_zeros() {
        let data = vec![0u8; 6]; // 3 zero components
        let packed =
            pack_10bit(&data, 3, PackingMethod::MethodA, Endian::Big).expect("pack should succeed");
        assert!(packed.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_pack_10bit_all_max() {
        let values: Vec<u16> = vec![1023, 1023, 1023];
        let mut data = Vec::new();
        for v in &values {
            data.extend_from_slice(&v.to_be_bytes());
        }

        let packed =
            pack_10bit(&data, 3, PackingMethod::MethodA, Endian::Big).expect("pack should succeed");
        let word = u32::from_be_bytes([packed[0], packed[1], packed[2], packed[3]]);
        // All component bits should be set, padding 0
        assert_eq!(word, 0xFFFF_FFFC);
    }
}
