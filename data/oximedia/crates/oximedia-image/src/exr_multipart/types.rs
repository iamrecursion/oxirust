//! Type definitions for OpenEXR 2.0 multi-part format.
//!
//! Contains channel types, bounding boxes, compression codecs, part types,
//! and the per-part data structure.

use crate::error::{ImageError, ImageResult};

// ── Channel type ──────────────────────────────────────────────────────────────

/// Data type for a single channel sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExrChannelType {
    /// 16-bit IEEE 754 half-float.
    Half,
    /// 32-bit IEEE 754 single-precision float.
    Float,
    /// 32-bit unsigned integer.
    Uint,
}

impl ExrChannelType {
    /// Bytes per sample.
    #[must_use]
    pub const fn bytes_per_sample(self) -> usize {
        match self {
            Self::Half => 2,
            Self::Float | Self::Uint => 4,
        }
    }

    /// EXR wire encoding (0=uint, 1=half, 2=float).
    #[must_use]
    pub const fn wire_code(self) -> u32 {
        match self {
            Self::Uint => 0,
            Self::Half => 1,
            Self::Float => 2,
        }
    }

    /// Parse from EXR wire code.
    pub fn from_wire_code(code: u32) -> ImageResult<Self> {
        match code {
            0 => Ok(Self::Uint),
            1 => Ok(Self::Half),
            2 => Ok(Self::Float),
            _ => Err(ImageError::invalid_format(format!(
                "Unknown EXR channel type code {code}"
            ))),
        }
    }
}

// ── Channel descriptor ────────────────────────────────────────────────────────

/// Describes one channel within an EXR part.
#[derive(Debug, Clone, PartialEq)]
pub struct ExrChannel {
    /// Channel name (e.g. `"R"`, `"G"`, `"B"`, `"A"`, `"Z"`, `"N.x"`).
    pub name: String,
    /// Sample data type.
    pub channel_type: ExrChannelType,
    /// Sub-sampling in X (1 = full resolution).
    pub x_sampling: u32,
    /// Sub-sampling in Y.
    pub y_sampling: u32,
    /// Whether the channel is stored linearly (non-perceptual).
    pub linear: bool,
}

impl ExrChannel {
    /// Create a full-resolution float channel.
    #[must_use]
    pub fn float(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            channel_type: ExrChannelType::Float,
            x_sampling: 1,
            y_sampling: 1,
            linear: true,
        }
    }

    /// Create a full-resolution half channel.
    #[must_use]
    pub fn half(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            channel_type: ExrChannelType::Half,
            x_sampling: 1,
            y_sampling: 1,
            linear: true,
        }
    }
}

// ── Bounding box (box2i) ──────────────────────────────────────────────────────

/// An axis-aligned integer rectangle used for data and display windows.
///
/// Coordinates are inclusive: a 1×1 image at origin has `x_min=0, y_min=0,
/// x_max=0, y_max=0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExrBox2i {
    /// Minimum X coordinate (inclusive).
    pub x_min: i32,
    /// Minimum Y coordinate (inclusive).
    pub y_min: i32,
    /// Maximum X coordinate (inclusive).
    pub x_max: i32,
    /// Maximum Y coordinate (inclusive).
    pub y_max: i32,
}

impl ExrBox2i {
    /// Width of the box (number of columns).
    #[must_use]
    pub fn width(&self) -> u32 {
        if self.x_max < self.x_min {
            0
        } else {
            (self.x_max - self.x_min + 1) as u32
        }
    }

    /// Height of the box (number of rows).
    #[must_use]
    pub fn height(&self) -> u32 {
        if self.y_max < self.y_min {
            0
        } else {
            (self.y_max - self.y_min + 1) as u32
        }
    }

    /// Total pixel count.
    #[must_use]
    pub fn pixel_count(&self) -> usize {
        self.width() as usize * self.height() as usize
    }

    /// Construct a zero-origin box for the given dimensions.
    ///
    /// A zero dimension yields `x_max < x_min` (empty box).
    #[must_use]
    pub fn from_dims(width: u32, height: u32) -> Self {
        if width == 0 || height == 0 {
            // Empty / degenerate box: x_max < x_min signals an empty region.
            return Self {
                x_min: 0,
                y_min: 0,
                x_max: -1,
                y_max: -1,
            };
        }
        Self {
            x_min: 0,
            y_min: 0,
            x_max: (width - 1) as i32,
            y_max: (height - 1) as i32,
        }
    }
}

// ── Compression ───────────────────────────────────────────────────────────────

/// EXR compression method codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExrCompression {
    /// No compression; raw pixel bytes.
    None,
    /// Run-length encoding.
    Rle,
    /// Deflate/zlib — one scanline per block.
    ZipSingle,
    /// Deflate/zlib — sixteen scanlines per block.
    Zip,
    /// PIZ (Haar wavelet + entropy coder).
    Piz,
    /// PXR24 (lossy 24-bit).
    Pxr24,
    /// B44 (lossy 4×4 half-float blocks).
    B44,
    /// B44A (B44 with flat-field optimization).
    B44a,
    /// DWAA (lossy DCT, 32-line blocks).
    Dwaa,
    /// DWAB (lossy DCT, 256-line blocks).
    Dwab,
}

impl ExrCompression {
    /// Number of scanlines grouped into one chunk block.
    #[must_use]
    pub const fn scanlines_per_block(self) -> u32 {
        match self {
            Self::None | Self::Rle | Self::ZipSingle => 1,
            Self::Zip | Self::Pxr24 => 16,
            Self::Piz | Self::B44 | Self::B44a | Self::Dwaa => 32,
            Self::Dwab => 256,
        }
    }

    /// EXR wire code (single byte in the header).
    #[must_use]
    pub const fn wire_code(self) -> u8 {
        match self {
            Self::None => 0,
            Self::Rle => 1,
            Self::ZipSingle => 2,
            Self::Zip => 3,
            Self::Piz => 4,
            Self::Pxr24 => 5,
            Self::B44 => 6,
            Self::B44a => 7,
            Self::Dwaa => 8,
            Self::Dwab => 9,
        }
    }

    /// Parse from EXR wire code.
    pub fn from_wire_code(code: u8) -> ImageResult<Self> {
        match code {
            0 => Ok(Self::None),
            1 => Ok(Self::Rle),
            2 => Ok(Self::ZipSingle),
            3 => Ok(Self::Zip),
            4 => Ok(Self::Piz),
            5 => Ok(Self::Pxr24),
            6 => Ok(Self::B44),
            7 => Ok(Self::B44a),
            8 => Ok(Self::Dwaa),
            9 => Ok(Self::Dwab),
            _ => Err(ImageError::invalid_format(format!(
                "Unknown EXR compression code {code}"
            ))),
        }
    }
}

// ── Part type ─────────────────────────────────────────────────────────────────

/// The storage type of a part in a multi-part EXR file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExrPartType {
    /// Scanline (row-by-row) image — the most common type.
    ScanlineImage,
    /// Tiled image (rectangular tile blocks).
    TiledImage,
    /// Deep scanline image (variable number of samples per pixel).
    DeepScanline,
    /// Deep tiled image.
    DeepTile,
}

impl ExrPartType {
    /// EXR header string representation.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ScanlineImage => "scanlineimage",
            Self::TiledImage => "tiledimage",
            Self::DeepScanline => "deepscanline",
            Self::DeepTile => "deeptile",
        }
    }

    /// Parse from EXR header string.
    pub fn from_str(s: &str) -> ImageResult<Self> {
        match s {
            "scanlineimage" => Ok(Self::ScanlineImage),
            "tiledimage" => Ok(Self::TiledImage),
            "deepscanline" => Ok(Self::DeepScanline),
            "deeptile" => Ok(Self::DeepTile),
            other => Err(ImageError::invalid_format(format!(
                "Unknown EXR part type '{other}'"
            ))),
        }
    }
}

// ── ExrPart ───────────────────────────────────────────────────────────────────

/// A single part (layer) within a multi-part EXR document.
///
/// `pixels` stores channel samples in row-major, channel-interleaved order:
/// `[ch0_px0, ch1_px0, …, chN_px0, ch0_px1, ch1_px1, …]`.
///
/// For an RGBA image of size *W × H*, `pixels.len()` is `W × H × 4`.
#[derive(Debug, Clone)]
pub struct ExrPart {
    /// Part / layer name (must be unique within the document).
    pub name: String,
    /// Storage type of this part.
    pub part_type: ExrPartType,
    /// Channel descriptors in display order.
    pub channels: Vec<ExrChannel>,
    /// Actual pixel data bounding box.
    pub data_window: ExrBox2i,
    /// Display (view) bounding box.
    pub display_window: ExrBox2i,
    /// Compression used for this part.
    pub compression: ExrCompression,
    /// Pixel data — channel-interleaved `f32` samples, row-major.
    ///
    /// Length must equal `width × height × channels.len()`.
    pub pixels: Vec<f32>,
    /// Width derived from `data_window`.
    pub width: u32,
    /// Height derived from `data_window`.
    pub height: u32,
}

impl ExrPart {
    /// Create a new part with zero-initialised pixels.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        part_type: ExrPartType,
        channels: Vec<ExrChannel>,
        data_window: ExrBox2i,
        display_window: ExrBox2i,
        compression: ExrCompression,
    ) -> Self {
        let width = data_window.width();
        let height = data_window.height();
        let pixel_count = width as usize * height as usize * channels.len();
        Self {
            name: name.into(),
            part_type,
            channels,
            data_window,
            display_window,
            compression,
            pixels: vec![0.0_f32; pixel_count],
            width,
            height,
        }
    }

    /// Validate internal consistency.
    pub fn validate(&self) -> ImageResult<()> {
        let expected = self.width as usize * self.height as usize * self.channels.len();
        if self.pixels.len() != expected {
            return Err(ImageError::invalid_format(format!(
                "Part '{}': pixel buffer has {} samples, expected {} \
                 ({}×{}×{} channels)",
                self.name,
                self.pixels.len(),
                expected,
                self.width,
                self.height,
                self.channels.len()
            )));
        }
        if self.width != self.data_window.width() {
            return Err(ImageError::invalid_format(format!(
                "Part '{}': width {} does not match data_window width {}",
                self.name,
                self.width,
                self.data_window.width()
            )));
        }
        if self.height != self.data_window.height() {
            return Err(ImageError::invalid_format(format!(
                "Part '{}': height {} does not match data_window height {}",
                self.name,
                self.height,
                self.data_window.height()
            )));
        }
        Ok(())
    }

    /// Number of channels.
    #[must_use]
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    /// Index of a channel by name.
    #[must_use]
    pub fn channel_index(&self, name: &str) -> Option<usize> {
        self.channels.iter().position(|c| c.name == name)
    }

    /// Get pixel value at `(x, y)` for channel `ch_idx`.
    pub fn get_sample(&self, x: u32, y: u32, ch_idx: usize) -> ImageResult<f32> {
        if x >= self.width || y >= self.height {
            return Err(ImageError::InvalidDimensions(x, y));
        }
        if ch_idx >= self.channels.len() {
            return Err(ImageError::invalid_format(format!(
                "Channel index {ch_idx} out of range (part has {} channels)",
                self.channels.len()
            )));
        }
        let stride = self.channels.len();
        let idx = (y as usize * self.width as usize + x as usize) * stride + ch_idx;
        Ok(self.pixels[idx])
    }

    /// Set pixel value at `(x, y)` for channel `ch_idx`.
    pub fn set_sample(&mut self, x: u32, y: u32, ch_idx: usize, value: f32) -> ImageResult<()> {
        if x >= self.width || y >= self.height {
            return Err(ImageError::InvalidDimensions(x, y));
        }
        if ch_idx >= self.channels.len() {
            return Err(ImageError::invalid_format(format!(
                "Channel index {ch_idx} out of range"
            )));
        }
        let stride = self.channels.len();
        let idx = (y as usize * self.width as usize + x as usize) * stride + ch_idx;
        self.pixels[idx] = value;
        Ok(())
    }
}
