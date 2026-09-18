//! Memory-mapped reader for DPX and EXR image sequences.
//!
//! Uses [`memmap2`] to map image files directly into virtual address space,
//! avoiding an explicit `read()` syscall and allowing the OS page cache to
//! satisfy accesses without copying data into user-space buffers.
//!
//! This is especially beneficial for large DPX/EXR frames (≥ 4 MB) read from
//! NVMe or networked storage, where eliminating the copy overhead improves
//! throughput.
//!
//! # Safety
//!
//! [`memmap2::Mmap`] is technically `unsafe` because the memory mapping can be
//! invalidated if another process truncates the file while it is mapped.
//! In practice this never occurs for read-only image files, but callers that
//! cannot tolerate even theoretical UB should use the fallback
//! [`MmapImageReader::open_buffered`] instead.
//!
//! # Example
//!
//! ```no_run
//! use std::path::Path;
//! use oximedia_image::mmap_reader::{MmapImageReader, ImageSequenceFormat};
//!
//! # fn example() -> Result<(), oximedia_image::ImageError> {
//! let reader = MmapImageReader::open(
//!     Path::new("/tmp/frame0001.dpx"),
//!     ImageSequenceFormat::Dpx,
//! )?;
//! let bytes = reader.as_bytes();
//! println!("File size: {} bytes", bytes.len());
//! # Ok(())
//! # }
//! ```

// memmap2::Mmap::map() is inherently unsafe (external process truncation is
// the only risk for read-only image files).  The buffered fallback
// (open_buffered) is fully safe.
#![allow(unsafe_code)]

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

use crate::error::{ImageError, ImageResult};

// ── Format ───────────────────────────────────────────────────────────────────

/// The image sequence format handled by this reader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageSequenceFormat {
    /// SMPTE 268M Digital Picture Exchange format.
    Dpx,
    /// ILM OpenEXR high-dynamic-range format.
    Exr,
}

impl ImageSequenceFormat {
    /// Returns the canonical file extension (lowercase, no dot).
    #[must_use]
    pub const fn extension(self) -> &'static str {
        match self {
            Self::Dpx => "dpx",
            Self::Exr => "exr",
        }
    }

    /// Detect format from the first 4 magic bytes of a file, or return `None`.
    #[must_use]
    pub fn from_magic(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 4 {
            return None;
        }
        // DPX magic: 0x53445058 ('SDPX') big-endian or 0x58504453 ('XPDS') little-endian
        if &bytes[0..4] == b"SDPX" || &bytes[0..4] == b"XPDS" {
            return Some(Self::Dpx);
        }
        // OpenEXR magic: 0x762F3101 (little-endian)
        if bytes[0] == 0x76 && bytes[1] == 0x2F && bytes[2] == 0x31 && bytes[3] == 0x01 {
            return Some(Self::Exr);
        }
        None
    }
}

// ── Reader ───────────────────────────────────────────────────────────────────

/// A memory-mapped reader for a single DPX or EXR image file.
///
/// The file is mapped read-only into virtual address space via
/// [`memmap2::Mmap`]. Accessing [`MmapImageReader::as_bytes`] returns a slice
/// directly backed by the mapping — no heap copy is made.
pub struct MmapImageReader {
    /// Keeps the file handle alive for the lifetime of the mapping.
    _file: File,
    /// The actual memory mapping.
    mmap: Mmap,
    /// The detected or user-supplied image format.
    format: ImageSequenceFormat,
}

impl MmapImageReader {
    /// Open `path` and map it into memory.
    ///
    /// # Errors
    ///
    /// Returns [`ImageError::Io`] if the file cannot be opened, or if
    /// `memmap2` fails to create the mapping (e.g., empty file, permission
    /// denied).
    pub fn open(path: &Path, format: ImageSequenceFormat) -> ImageResult<Self> {
        let file = File::open(path)?;
        // SAFETY: The file is opened read-only and we keep `_file` alive for
        // as long as `mmap` exists, preventing the mapping from being
        // invalidated by a close.  The file is never truncated by this
        // process.
        let mmap = unsafe { Mmap::map(&file) }?;
        Ok(Self {
            _file: file,
            mmap,
            format,
        })
    }

    /// Open `path`, detect the format from the file magic bytes, and map it.
    ///
    /// # Errors
    ///
    /// Returns [`ImageError::Io`] on I/O failures or
    /// [`ImageError::InvalidFormat`] when the magic bytes are not recognised.
    pub fn open_detect(path: &Path) -> ImageResult<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file) }?;
        let format =
            ImageSequenceFormat::from_magic(&mmap[..mmap.len().min(4)]).ok_or_else(|| {
                ImageError::invalid_format("Unrecognised magic bytes — expected DPX or EXR")
            })?;
        Ok(Self {
            _file: file,
            mmap,
            format,
        })
    }

    /// Open `path` by reading all bytes into a `Vec<u8>` (non-mmap fallback).
    ///
    /// Use this when memory mapping is undesirable (e.g., on platforms where
    /// `mmap` semantics are unreliable, or in sandboxed environments).
    ///
    /// # Errors
    ///
    /// Returns [`ImageError::Io`] on I/O failures.
    pub fn open_buffered(
        path: &Path,
        format: ImageSequenceFormat,
    ) -> ImageResult<BufferedImageReader> {
        let data = std::fs::read(path)?;
        Ok(BufferedImageReader { data, format })
    }

    /// Returns the raw bytes of the mapped file.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.mmap
    }

    /// Returns the image sequence format.
    #[must_use]
    pub fn format(&self) -> ImageSequenceFormat {
        self.format
    }

    /// Returns the total size of the mapped file in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.mmap.len()
    }

    /// Returns `true` when the mapped file is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
    }

    /// Decode the frame header and return the raw pixel bytes.
    ///
    /// Currently this returns all bytes after the format-specific header offset
    /// as a `Vec<u8>`.  Full pixel decoding (colour space conversion, bit
    /// depth expansion) is delegated to the respective format modules.
    ///
    /// # Errors
    ///
    /// Returns [`ImageError::InvalidFormat`] if the file is too short to
    /// contain a valid header, or [`ImageError::Unsupported`] for format
    /// features not yet implemented.
    pub fn decode_frame(&self) -> ImageResult<Vec<u8>> {
        let bytes = self.as_bytes();
        match self.format {
            ImageSequenceFormat::Dpx => decode_dpx_raw(bytes),
            ImageSequenceFormat::Exr => decode_exr_raw(bytes),
        }
    }
}

// ── Buffered fallback ─────────────────────────────────────────────────────────

/// A non-mmap reader that holds the file content in a heap-allocated buffer.
///
/// Created by [`MmapImageReader::open_buffered`].
pub struct BufferedImageReader {
    data: Vec<u8>,
    format: ImageSequenceFormat,
}

impl BufferedImageReader {
    /// Returns the raw file bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Returns the image sequence format.
    #[must_use]
    pub fn format(&self) -> ImageSequenceFormat {
        self.format
    }

    /// Returns the size of the buffered data in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` when the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Decode the frame header and return the raw pixel bytes.
    ///
    /// # Errors
    ///
    /// Returns [`ImageError::InvalidFormat`] if the data is too short.
    pub fn decode_frame(&self) -> ImageResult<Vec<u8>> {
        match self.format {
            ImageSequenceFormat::Dpx => decode_dpx_raw(&self.data),
            ImageSequenceFormat::Exr => decode_exr_raw(&self.data),
        }
    }
}

// ── Raw header decoders ───────────────────────────────────────────────────────

/// Minimal DPX raw decoder: validates magic and returns pixel data bytes.
///
/// DPX file layout (SMPTE 268M-2003):
/// - Bytes 0–3: magic `SDPX` (BE) or `XPDS` (LE)
/// - Bytes 4–7: image data offset (u32, endian-matched)
/// - Remaining bytes from offset onward: pixel data
fn decode_dpx_raw(bytes: &[u8]) -> ImageResult<Vec<u8>> {
    if bytes.len() < 8 {
        return Err(ImageError::invalid_format("DPX file too short for header"));
    }
    let is_be = &bytes[0..4] == b"SDPX";
    let is_le = &bytes[0..4] == b"XPDS";
    if !is_be && !is_le {
        return Err(ImageError::invalid_format("Not a DPX file (bad magic)"));
    }
    let offset = if is_be {
        u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize
    } else {
        u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize
    };
    if offset > bytes.len() {
        return Err(ImageError::invalid_format(
            "DPX image data offset exceeds file length",
        ));
    }
    Ok(bytes[offset..].to_vec())
}

/// Minimal EXR raw decoder: validates magic and returns bytes after the magic.
///
/// Full EXR channel/tile decoding is handled by `crate::exr`.
fn decode_exr_raw(bytes: &[u8]) -> ImageResult<Vec<u8>> {
    if bytes.len() < 4 {
        return Err(ImageError::invalid_format("EXR file too short for header"));
    }
    if !(bytes[0] == 0x76 && bytes[1] == 0x2F && bytes[2] == 0x31 && bytes[3] == 0x01) {
        return Err(ImageError::invalid_format("Not an EXR file (bad magic)"));
    }
    // Return everything after the 4-byte magic as raw payload
    Ok(bytes[4..].to_vec())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Write bytes to a temp file and return its path.
    fn temp_file_with_bytes(bytes: &[u8]) -> (tempfile::NamedTempFile, std::path::PathBuf) {
        let mut f = tempfile::NamedTempFile::new().expect("temp file");
        f.write_all(bytes).expect("write");
        f.flush().expect("flush");
        let path = f.path().to_path_buf();
        (f, path)
    }

    #[test]
    fn test_mmap_reader_opens() {
        let data = b"hello world this is a test file for mmap";
        let (_f, path) = temp_file_with_bytes(data);
        let reader =
            MmapImageReader::open(&path, ImageSequenceFormat::Dpx).expect("open should succeed");
        assert_eq!(reader.len(), data.len(), "mapped length should match file");
    }

    #[test]
    fn test_mmap_reader_bytes_accessible() {
        // DPX BE magic + 4-byte offset
        let mut data = Vec::new();
        data.extend_from_slice(b"SDPX"); // magic
        data.extend_from_slice(&8u32.to_be_bytes()); // offset = 8 (points right after header)
        data.extend_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD]); // dummy pixels
        let (_f, path) = temp_file_with_bytes(&data);
        let reader = MmapImageReader::open(&path, ImageSequenceFormat::Dpx).expect("open");
        let bytes = reader.as_bytes();
        // First 4 bytes must be the DPX magic
        assert_eq!(&bytes[0..4], b"SDPX");
    }

    #[test]
    fn test_mmap_reader_decode_dpx() {
        let mut data = Vec::new();
        data.extend_from_slice(b"SDPX");
        data.extend_from_slice(&8u32.to_be_bytes());
        data.extend_from_slice(&[10, 20, 30, 40]);
        let (_f, path) = temp_file_with_bytes(&data);
        let reader = MmapImageReader::open(&path, ImageSequenceFormat::Dpx).expect("open");
        let pixels = reader.decode_frame().expect("decode");
        assert_eq!(pixels, vec![10, 20, 30, 40]);
    }

    #[test]
    fn test_mmap_reader_decode_exr() {
        let mut data = Vec::new();
        data.extend_from_slice(&[0x76, 0x2F, 0x31, 0x01]); // EXR magic
        data.extend_from_slice(&[1, 2, 3, 4, 5]);
        let (_f, path) = temp_file_with_bytes(&data);
        let reader = MmapImageReader::open(&path, ImageSequenceFormat::Exr).expect("open");
        let raw = reader.decode_frame().expect("decode");
        assert_eq!(raw, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_mmap_reader_detect_dpx() {
        let mut data = vec![0u8; 12];
        data[0..4].copy_from_slice(b"SDPX");
        data[4..8].copy_from_slice(&8u32.to_be_bytes());
        let (_f, path) = temp_file_with_bytes(&data);
        let reader = MmapImageReader::open_detect(&path).expect("detect");
        assert_eq!(reader.format(), ImageSequenceFormat::Dpx);
    }

    #[test]
    fn test_mmap_reader_detect_exr() {
        let mut data = vec![0u8; 8];
        data[0..4].copy_from_slice(&[0x76, 0x2F, 0x31, 0x01]);
        let (_f, path) = temp_file_with_bytes(&data);
        let reader = MmapImageReader::open_detect(&path).expect("detect");
        assert_eq!(reader.format(), ImageSequenceFormat::Exr);
    }

    #[test]
    fn test_mmap_reader_detect_unknown_format() {
        let data = b"THIS_IS_NOT_A_KNOWN_FORMAT_XYZ";
        let (_f, path) = temp_file_with_bytes(data);
        assert!(MmapImageReader::open_detect(&path).is_err());
    }

    #[test]
    fn test_buffered_reader_bytes_accessible() {
        let mut data = Vec::new();
        data.extend_from_slice(b"SDPX");
        data.extend_from_slice(&8u32.to_be_bytes());
        data.extend_from_slice(&[9, 8, 7, 6]);
        let (_f, path) = temp_file_with_bytes(&data);
        let reader = MmapImageReader::open_buffered(&path, ImageSequenceFormat::Dpx).expect("open");
        assert_eq!(reader.len(), data.len());
        assert_eq!(&reader.as_bytes()[0..4], b"SDPX");
    }

    #[test]
    fn test_format_extension() {
        assert_eq!(ImageSequenceFormat::Dpx.extension(), "dpx");
        assert_eq!(ImageSequenceFormat::Exr.extension(), "exr");
    }

    #[test]
    fn test_image_sequence_format_from_magic_dpx_be() {
        assert_eq!(
            ImageSequenceFormat::from_magic(b"SDPX"),
            Some(ImageSequenceFormat::Dpx)
        );
    }

    #[test]
    fn test_image_sequence_format_from_magic_dpx_le() {
        assert_eq!(
            ImageSequenceFormat::from_magic(b"XPDS"),
            Some(ImageSequenceFormat::Dpx)
        );
    }

    #[test]
    fn test_image_sequence_format_from_magic_exr() {
        assert_eq!(
            ImageSequenceFormat::from_magic(&[0x76, 0x2F, 0x31, 0x01]),
            Some(ImageSequenceFormat::Exr)
        );
    }

    #[test]
    fn test_image_sequence_format_from_magic_unknown() {
        assert_eq!(ImageSequenceFormat::from_magic(b"JPEG"), None);
        assert_eq!(ImageSequenceFormat::from_magic(&[]), None);
    }
}
