//! File format V2 for chunked out-of-core arrays.
//!
//! This module defines the file format for storing large arrays on disk
//! with efficient chunk-based access. Format V2 supports:
//! - Incremental chunk writing without full materialization
//! - Direct chunk seeking and reading
//! - Optional compression per chunk
//! - Backward compatibility with V1 format

use crate::error::{CoreError, CoreResult, ErrorContext, ErrorLocation};
use serde::{Deserialize, Serialize};
use std::io::{Read, Seek, SeekFrom, Write};

/// Magic bytes for SciRS2 out-of-core array files (V2 format)
pub const MAGIC_BYTES_V2: &[u8; 4] = b"SCI2";

/// Magic bytes for V1 format detection (oxicode serialized arrays)
/// V1 format doesn't have explicit magic bytes, but we can detect it
/// by checking if oxicode deserialization succeeds
pub const MAGIC_BYTES_V1_MARKER: u8 = 0x91; // Oxicode array marker

/// File format version
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FormatVersion {
    /// Version 1: Full array serialization (legacy)
    V1,
    /// Version 2: Chunked serialization with index
    V2,
}

/// Compression algorithm for chunks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionType {
    /// No compression
    None,
    /// LZ4 compression (fast)
    Lz4,
    /// Zstd compression (balanced)
    Zstd,
    /// Snappy compression (very fast)
    Snappy,
}

/// Header for out-of-core array file (V2 format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutOfCoreHeaderV2 {
    /// Magic bytes to identify file format
    pub magic: [u8; 4],

    /// Format version
    pub version: u16,

    /// Array shape
    pub shape: Vec<usize>,

    /// Total number of elements
    pub total_elements: usize,

    /// Element size in bytes
    pub element_size: usize,

    /// Number of chunks
    pub num_chunks: usize,

    /// Compression type used for chunks
    pub compression: CompressionType,

    /// Offset to chunk index (from start of file)
    pub chunk_index_offset: u64,

    /// Reserved bytes for future use (using Vec for serde compatibility)
    pub reserved: Vec<u8>,
}

impl OutOfCoreHeaderV2 {
    /// Create a new header with default values
    pub fn new(
        shape: Vec<usize>,
        element_size: usize,
        num_chunks: usize,
        compression: CompressionType,
    ) -> Self {
        let total_elements = shape.iter().product();

        Self {
            magic: *MAGIC_BYTES_V2,
            version: 2,
            shape,
            total_elements,
            element_size,
            num_chunks,
            compression,
            chunk_index_offset: 0, // Will be set after chunks are written
            reserved: vec![0; 64],
        }
    }

    /// Serialize header to bytes
    pub fn to_bytes(&self) -> CoreResult<Vec<u8>> {
        use oxicode::{config, serde as oxicode_serde};

        let cfg = config::standard();
        oxicode_serde::encode_to_vec(self, cfg).map_err(|e| {
            CoreError::IoError(
                ErrorContext::new(format!("Failed to serialize header: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })
    }

    /// Deserialize header from bytes
    pub fn from_bytes(bytes: &[u8]) -> CoreResult<Self> {
        use oxicode::{config, serde as oxicode_serde};

        let cfg = config::standard();
        let (header, _len): (Self, usize) = oxicode_serde::decode_owned_from_slice(bytes, cfg)
            .map_err(|e| {
                CoreError::IoError(
                    ErrorContext::new(format!("Failed to deserialize header: {e}"))
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;

        Ok(header)
    }

    /// Validate header integrity
    pub fn validate(&self) -> CoreResult<()> {
        // Check magic bytes
        if &self.magic != MAGIC_BYTES_V2 {
            return Err(CoreError::ValidationError(
                ErrorContext::new(format!(
                    "Invalid magic bytes: expected {:?}, got {:?}",
                    MAGIC_BYTES_V2, self.magic
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // Check version
        if self.version != 2 {
            return Err(CoreError::ValidationError(
                ErrorContext::new(format!("Unsupported version: {}", self.version))
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // Check shape
        if self.shape.is_empty() {
            return Err(CoreError::ValidationError(
                ErrorContext::new("Shape cannot be empty".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // Check total elements
        let computed_total: usize = self.shape.iter().product();
        if computed_total != self.total_elements {
            return Err(CoreError::ValidationError(
                ErrorContext::new(format!(
                    "Total elements mismatch: computed {computed_total}, stored {}",
                    self.total_elements
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // Check element size
        if self.element_size == 0 {
            return Err(CoreError::ValidationError(
                ErrorContext::new("Element size cannot be zero".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // Check num_chunks
        if self.num_chunks == 0 {
            return Err(CoreError::ValidationError(
                ErrorContext::new("Number of chunks cannot be zero".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        Ok(())
    }
}

/// Entry in the chunk index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkIndexEntry {
    /// Chunk number (0-based)
    pub chunk_id: usize,

    /// Offset from start of file to chunk data
    pub offset: u64,

    /// Size of chunk data in bytes (uncompressed)
    pub size: usize,

    /// Size of chunk data in bytes (compressed), or 0 if not compressed
    pub compressed_size: usize,

    /// Number of elements in this chunk
    pub num_elements: usize,
}

impl ChunkIndexEntry {
    /// Create a new chunk index entry
    pub fn new(chunk_id: usize, offset: u64, size: usize, num_elements: usize) -> Self {
        Self {
            chunk_id,
            offset,
            size,
            compressed_size: 0, // No compression by default
            num_elements,
        }
    }

    /// Set compression info
    pub fn with_compression(mut self, compressed_size: usize) -> Self {
        self.compressed_size = compressed_size;
        self
    }

    /// Check if chunk is compressed
    pub const fn is_compressed(&self) -> bool {
        self.compressed_size > 0
    }

    /// Get the actual size to read from disk
    pub const fn disk_size(&self) -> usize {
        if self.is_compressed() {
            self.compressed_size
        } else {
            self.size
        }
    }
}

/// Chunk index for efficient seeking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkIndex {
    /// All chunk entries
    pub entries: Vec<ChunkIndexEntry>,
}

impl ChunkIndex {
    /// Create a new empty chunk index
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Add a chunk entry
    pub fn add_entry(&mut self, entry: ChunkIndexEntry) {
        self.entries.push(entry);
    }

    /// Get chunk entry by ID
    pub fn get_entry(&self, chunk_id: usize) -> Option<&ChunkIndexEntry> {
        self.entries.get(chunk_id)
    }

    /// Get number of chunks
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Serialize index to bytes
    pub fn to_bytes(&self) -> CoreResult<Vec<u8>> {
        use oxicode::{config, serde as oxicode_serde};

        let cfg = config::standard();
        oxicode_serde::encode_to_vec(self, cfg).map_err(|e| {
            CoreError::IoError(
                ErrorContext::new(format!("Failed to serialize chunk index: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })
    }

    /// Deserialize index from bytes
    pub fn from_bytes(bytes: &[u8]) -> CoreResult<Self> {
        use oxicode::{config, serde as oxicode_serde};

        let cfg = config::standard();
        let (index, _len): (Self, usize) = oxicode_serde::decode_owned_from_slice(bytes, cfg)
            .map_err(|e| {
                CoreError::IoError(
                    ErrorContext::new(format!("Failed to deserialize chunk index: {e}"))
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;

        Ok(index)
    }
}

impl Default for ChunkIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Detect file format version
pub fn detect_format_version<R: Read + Seek>(reader: &mut R) -> CoreResult<FormatVersion> {
    // Save current position
    let original_pos = reader.stream_position().map_err(|e| {
        CoreError::IoError(
            ErrorContext::new(format!("Failed to get stream position: {e}"))
                .with_location(ErrorLocation::new(file!(), line!())),
        )
    })?;

    // Read first 4 bytes
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic).map_err(|e| {
        CoreError::IoError(
            ErrorContext::new(format!("Failed to read magic bytes: {e}"))
                .with_location(ErrorLocation::new(file!(), line!())),
        )
    })?;

    // Restore position
    reader.seek(SeekFrom::Start(original_pos)).map_err(|e| {
        CoreError::IoError(
            ErrorContext::new(format!("Failed to restore stream position: {e}"))
                .with_location(ErrorLocation::new(file!(), line!())),
        )
    })?;

    // Check magic bytes
    if &magic == MAGIC_BYTES_V2 {
        Ok(FormatVersion::V2)
    } else {
        // Assume V1 format (no magic bytes)
        Ok(FormatVersion::V1)
    }
}

/// Fixed header size (padded to ensure consistent size)
pub const HEADER_FIXED_SIZE: usize = 256;

/// Write header to file with fixed size padding
pub fn write_header<W: Write>(writer: &mut W, header: &OutOfCoreHeaderV2) -> CoreResult<usize> {
    let header_bytes = header.to_bytes()?;

    if header_bytes.len() > HEADER_FIXED_SIZE - 4 {
        return Err(CoreError::IoError(
            ErrorContext::new(format!(
                "Header too large: {} bytes (max {})",
                header_bytes.len(),
                HEADER_FIXED_SIZE - 4
            ))
            .with_location(ErrorLocation::new(file!(), line!())),
        ));
    }

    // Write header size first (4 bytes)
    let size_bytes = (header_bytes.len() as u32).to_le_bytes();
    writer.write_all(&size_bytes).map_err(|e| {
        CoreError::IoError(
            ErrorContext::new(format!("Failed to write header size: {e}"))
                .with_location(ErrorLocation::new(file!(), line!())),
        )
    })?;

    // Write header data
    writer.write_all(&header_bytes).map_err(|e| {
        CoreError::IoError(
            ErrorContext::new(format!("Failed to write header: {e}"))
                .with_location(ErrorLocation::new(file!(), line!())),
        )
    })?;

    // Pad to fixed size
    let padding_size = HEADER_FIXED_SIZE - 4 - header_bytes.len();
    let padding = vec![0u8; padding_size];
    writer.write_all(&padding).map_err(|e| {
        CoreError::IoError(
            ErrorContext::new(format!("Failed to write header padding: {e}"))
                .with_location(ErrorLocation::new(file!(), line!())),
        )
    })?;

    Ok(HEADER_FIXED_SIZE)
}

/// Read header from file (with padding skip)
pub fn read_header<R: Read>(reader: &mut R) -> CoreResult<OutOfCoreHeaderV2> {
    // Read header size first (4 bytes)
    let mut size_bytes = [0u8; 4];
    reader.read_exact(&mut size_bytes).map_err(|e| {
        CoreError::IoError(
            ErrorContext::new(format!("Failed to read header size: {e}"))
                .with_location(ErrorLocation::new(file!(), line!())),
        )
    })?;

    let header_size = u32::from_le_bytes(size_bytes) as usize;

    // Read header data
    let mut header_bytes = vec![0u8; header_size];
    reader.read_exact(&mut header_bytes).map_err(|e| {
        CoreError::IoError(
            ErrorContext::new(format!("Failed to read header data: {e}"))
                .with_location(ErrorLocation::new(file!(), line!())),
        )
    })?;

    // Skip padding to reach fixed header size
    let padding_size = HEADER_FIXED_SIZE - 4 - header_size;
    let mut padding = vec![0u8; padding_size];
    reader.read_exact(&mut padding).map_err(|e| {
        CoreError::IoError(
            ErrorContext::new(format!("Failed to skip header padding: {e}"))
                .with_location(ErrorLocation::new(file!(), line!())),
        )
    })?;

    // Deserialize and validate
    let header = OutOfCoreHeaderV2::from_bytes(&header_bytes)?;
    header.validate()?;

    Ok(header)
}

/// Write chunk index to file
pub fn write_chunk_index<W: Write>(writer: &mut W, index: &ChunkIndex) -> CoreResult<u64> {
    let index_bytes = index.to_bytes()?;

    // Write index size first (4 bytes)
    let size_bytes = (index_bytes.len() as u32).to_le_bytes();
    writer.write_all(&size_bytes).map_err(|e| {
        CoreError::IoError(
            ErrorContext::new(format!("Failed to write chunk index size: {e}"))
                .with_location(ErrorLocation::new(file!(), line!())),
        )
    })?;

    // Write index data
    writer.write_all(&index_bytes).map_err(|e| {
        CoreError::IoError(
            ErrorContext::new(format!("Failed to write chunk index: {e}"))
                .with_location(ErrorLocation::new(file!(), line!())),
        )
    })?;

    Ok((4 + index_bytes.len()) as u64)
}

/// Read chunk index from file
pub fn read_chunk_index<R: Read + Seek>(reader: &mut R, offset: u64) -> CoreResult<ChunkIndex> {
    // Seek to chunk index offset
    reader.seek(SeekFrom::Start(offset)).map_err(|e| {
        CoreError::IoError(
            ErrorContext::new(format!("Failed to seek to chunk index: {e}"))
                .with_location(ErrorLocation::new(file!(), line!())),
        )
    })?;

    // Read index size first (4 bytes)
    let mut size_bytes = [0u8; 4];
    reader.read_exact(&mut size_bytes).map_err(|e| {
        CoreError::IoError(
            ErrorContext::new(format!("Failed to read chunk index size: {e}"))
                .with_location(ErrorLocation::new(file!(), line!())),
        )
    })?;

    let index_size = u32::from_le_bytes(size_bytes) as usize;

    // Read index data
    let mut index_bytes = vec![0u8; index_size];
    reader.read_exact(&mut index_bytes).map_err(|e| {
        CoreError::IoError(
            ErrorContext::new(format!("Failed to read chunk index data: {e}"))
                .with_location(ErrorLocation::new(file!(), line!())),
        )
    })?;

    ChunkIndex::from_bytes(&index_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_header_creation_and_validation() {
        let header = OutOfCoreHeaderV2::new(vec![100, 200], 8, 10, CompressionType::None);

        assert_eq!(header.magic, *MAGIC_BYTES_V2);
        assert_eq!(header.version, 2);
        assert_eq!(header.shape, vec![100, 200]);
        assert_eq!(header.total_elements, 20000);
        assert_eq!(header.element_size, 8);
        assert_eq!(header.num_chunks, 10);
        assert!(header.validate().is_ok());
    }

    #[test]
    fn test_header_serialization() {
        let header = OutOfCoreHeaderV2::new(vec![100, 200], 8, 10, CompressionType::Lz4);

        let bytes = header.to_bytes().expect("Serialization failed");
        let deserialized = OutOfCoreHeaderV2::from_bytes(&bytes).expect("Deserialization failed");

        assert_eq!(header.magic, deserialized.magic);
        assert_eq!(header.version, deserialized.version);
        assert_eq!(header.shape, deserialized.shape);
        assert_eq!(header.total_elements, deserialized.total_elements);
    }

    #[test]
    fn test_chunk_index_entry() {
        let entry = ChunkIndexEntry::new(0, 1024, 8192, 1000);

        assert_eq!(entry.chunk_id, 0);
        assert_eq!(entry.offset, 1024);
        assert_eq!(entry.size, 8192);
        assert_eq!(entry.num_elements, 1000);
        assert!(!entry.is_compressed());
        assert_eq!(entry.disk_size(), 8192);

        let compressed = entry.with_compression(4096);
        assert!(compressed.is_compressed());
        assert_eq!(compressed.disk_size(), 4096);
    }

    #[test]
    fn test_chunk_index() {
        let mut index = ChunkIndex::new();

        assert!(index.is_empty());
        assert_eq!(index.len(), 0);

        index.add_entry(ChunkIndexEntry::new(0, 1024, 8192, 1000));
        index.add_entry(ChunkIndexEntry::new(1, 9216, 8192, 1000));

        assert!(!index.is_empty());
        assert_eq!(index.len(), 2);

        let entry = index.get_entry(0).expect("Entry not found");
        assert_eq!(entry.chunk_id, 0);
        assert_eq!(entry.offset, 1024);
    }

    #[test]
    fn test_chunk_index_serialization() {
        let mut index = ChunkIndex::new();
        index.add_entry(ChunkIndexEntry::new(0, 1024, 8192, 1000));
        index.add_entry(ChunkIndexEntry::new(1, 9216, 8192, 1000));

        let bytes = index.to_bytes().expect("Serialization failed");
        let deserialized = ChunkIndex::from_bytes(&bytes).expect("Deserialization failed");

        assert_eq!(index.len(), deserialized.len());
        assert_eq!(
            index.get_entry(0).expect("Entry not found").offset,
            deserialized.get_entry(0).expect("Entry not found").offset
        );
    }

    #[test]
    fn test_format_detection() {
        // Test V2 format detection
        let mut v2_data = Vec::new();
        v2_data.extend_from_slice(MAGIC_BYTES_V2);
        v2_data.extend_from_slice(&[0; 100]); // Dummy data

        let mut cursor = Cursor::new(v2_data);
        let version = detect_format_version(&mut cursor).expect("Detection failed");
        assert_eq!(version, FormatVersion::V2);

        // Test V1 format detection (no magic bytes)
        let v1_data = vec![MAGIC_BYTES_V1_MARKER, 0, 0, 0];
        let mut cursor = Cursor::new(v1_data);
        let version = detect_format_version(&mut cursor).expect("Detection failed");
        assert_eq!(version, FormatVersion::V1);
    }
}
