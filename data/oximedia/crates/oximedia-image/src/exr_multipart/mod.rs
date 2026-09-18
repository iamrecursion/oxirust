//! OpenEXR 2.0 multi-part format support.
//!
//! Implements reading and writing of OpenEXR 2.0 multi-part files as specified
//! by the Weta Digital / AMPAS OpenEXR 2.0 specification.  Each independent
//! image (part) has its own header, chunk offset table, and pixel data blocks.
//!
//! # Multi-Part Format Overview
//!
//! A multi-part EXR file is distinguished by version bit 12 (`0x1000`):
//!
//! ```text
//! [magic: u32][version: u32]          ← version has bit 12 set for multi-part
//! [header 0] [NUL]                    ← per-part header, NUL-terminated
//! [header 1] [NUL]
//!   …
//! [NUL]                               ← empty header = end of headers
//! [chunk_offset_table_0: N0 × u64]
//! [chunk_offset_table_1: N1 × u64]
//!   …
//! [chunk_0_0] [chunk_0_1] … [chunk_1_0] …
//! ```
//!
//! Each scanline chunk in a multi-part file is prefixed with a 4-byte
//! `part_number` (i32 LE) before the `y_coordinate` (i32 LE) and
//! `pixel_data_size` (u32 LE).
//!
//! # Example
//!
//! ```rust
//! use oximedia_image::exr_multipart::{
//!     ExrBox2i, ExrChannel, ExrChannelType, ExrCompression, ExrPart, ExrPartType, MultiPartExr,
//! };
//!
//! let window = ExrBox2i { x_min: 0, y_min: 0, x_max: 3, y_max: 3 };
//! let mut part = ExrPart {
//!     name: "beauty".to_string(),
//!     part_type: ExrPartType::ScanlineImage,
//!     channels: vec![
//!         ExrChannel { name: "R".to_string(), channel_type: ExrChannelType::Float,
//!                      x_sampling: 1, y_sampling: 1, linear: true },
//!     ],
//!     data_window: window,
//!     display_window: window,
//!     compression: ExrCompression::None,
//!     pixels: vec![0.0f32; 16],   // 4×4 × 1 channel
//!     width: 4,
//!     height: 4,
//! };
//! let doc = MultiPartExr { parts: vec![part] };
//! let bytes = doc.to_bytes().expect("serialise");
//! let roundtrip = MultiPartExr::from_bytes(&bytes).expect("deserialise");
//! assert_eq!(roundtrip.parts.len(), 1);
//! ```

pub mod parse;
pub mod types;
pub mod write;

// Re-export public types at the module level for backward compatibility.
pub use types::{ExrBox2i, ExrChannel, ExrChannelType, ExrCompression, ExrPart, ExrPartType};

use crate::error::{ImageError, ImageResult};
use std::io::Cursor;

// ── EXR constants ─────────────────────────────────────────────────────────────

/// OpenEXR magic number (unchanged from 1.x).
pub(crate) const EXR_MAGIC: u32 = 20000630;

/// EXR version number field (byte 0).
pub(crate) const EXR_VERSION: u8 = 2;

/// Version flag indicating multi-part file (bit 12 of the 32-bit version word).
pub(crate) const VERSION_FLAG_MULTIPART: u32 = 0x1000;

/// Version flag indicating tiled storage (bit 9).
pub(crate) const VERSION_FLAG_TILED: u32 = 0x0200;

/// Version flag indicating deep image (bit 11).
pub(crate) const VERSION_FLAG_DEEP: u32 = 0x0800;

// ── MultiPartExr ──────────────────────────────────────────────────────────────

/// A complete OpenEXR 2.0 multi-part document.
///
/// Contains one or more [`ExrPart`]s, each with independent channels,
/// data windows, and compression.
#[derive(Debug)]
pub struct MultiPartExr {
    /// All parts in declaration order.
    pub parts: Vec<ExrPart>,
}

impl MultiPartExr {
    /// Create an empty document.
    #[must_use]
    pub fn new() -> Self {
        Self { parts: Vec::new() }
    }

    /// Add a part.
    pub fn add_part(&mut self, part: ExrPart) {
        self.parts.push(part);
    }

    /// Get a part by name (immutable).
    #[must_use]
    pub fn part_by_name(&self, name: &str) -> Option<&ExrPart> {
        self.parts.iter().find(|p| p.name == name)
    }

    /// Get a part by name (mutable).
    #[must_use]
    pub fn part_by_name_mut(&mut self, name: &str) -> Option<&mut ExrPart> {
        self.parts.iter_mut().find(|p| p.name == name)
    }

    /// Number of parts.
    #[must_use]
    pub fn part_count(&self) -> usize {
        self.parts.len()
    }

    /// Validate all parts.
    pub fn validate(&self) -> ImageResult<()> {
        if self.parts.is_empty() {
            return Err(ImageError::invalid_format("MultiPartExr has no parts"));
        }
        for part in &self.parts {
            part.validate()?;
        }
        // Duplicate name check
        let mut names: Vec<&str> = self.parts.iter().map(|p| p.name.as_str()).collect();
        names.sort_unstable();
        for win in names.windows(2) {
            if win[0] == win[1] {
                return Err(ImageError::invalid_format(format!(
                    "Duplicate part name '{}'",
                    win[0]
                )));
            }
        }
        Ok(())
    }

    // ── Deserialisation ───────────────────────────────────────────────────────

    /// Parse a multi-part EXR file from a byte slice.
    ///
    /// # Errors
    ///
    /// Returns [`ImageError::InvalidFormat`] for malformed data,
    /// [`ImageError::Unsupported`] for non-scanline-image pixel data
    /// (metadata for Tiled/Deep parts is still parsed; only pixels are skipped).
    pub fn from_bytes(data: &[u8]) -> ImageResult<Self> {
        let mut cur = Cursor::new(data);
        parse::read_multipart_exr(&mut cur)
    }

    /// Check whether a byte slice is a (single-part OR multi-part) EXR file
    /// without fully parsing it.
    ///
    /// Returns `true` if the magic bytes match.
    #[must_use]
    pub fn is_exr(data: &[u8]) -> bool {
        data.len() >= 4 && {
            let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
            magic == EXR_MAGIC
        }
    }

    /// Returns `true` if the byte slice has the multi-part version flag set.
    ///
    /// # Errors
    ///
    /// Returns an error if the data is too short or the magic is wrong.
    pub fn is_multipart_bytes(data: &[u8]) -> ImageResult<bool> {
        if data.len() < 8 {
            return Err(ImageError::invalid_format(
                "Buffer too short to be an EXR file",
            ));
        }
        let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        if magic != EXR_MAGIC {
            return Err(ImageError::invalid_format("Not an EXR file (bad magic)"));
        }
        let version_word = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        let flags = version_word >> 8;
        Ok((flags & VERSION_FLAG_MULTIPART) != 0)
    }

    // ── Serialisation ─────────────────────────────────────────────────────────

    /// Serialise the document to a byte vector.
    ///
    /// Only [`ExrPartType::ScanlineImage`] parts are supported for pixel
    /// output.  Calling this on a document containing Tiled, DeepScanline,
    /// or DeepTile parts returns [`ImageError::Unsupported`].
    ///
    /// # Errors
    ///
    /// Returns an error if any part has an invalid configuration or if a
    /// non-scanline part type is encountered.
    pub fn to_bytes(&self) -> ImageResult<Vec<u8>> {
        self.validate()?;
        for part in &self.parts {
            if part.part_type != ExrPartType::ScanlineImage {
                return Err(ImageError::Unsupported(format!(
                    "Part '{}' has type '{}' — only ScanlineImage is \
                     supported for writing",
                    part.name,
                    part.part_type.as_str()
                )));
            }
        }
        write::write_multipart_exr(self)
    }
}

impl Default for MultiPartExr {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    // All tests are in tests.rs; include them here so they run as part of this module.
    include!("tests.rs");
}
