//! Cache state serialization and restoration.
//!
//! Provides a binary serializer / deserializer so a cache can be persisted
//! to disk on shutdown and quickly restored on startup — avoiding a cold-start
//! penalty where all traffic misses until the cache is re-warmed organically.
//!
//! # Wire format
//!
//! The on-disk format is a simple length-prefixed binary encoding; no external
//! serialization dependency is needed.
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │ MAGIC   8 bytes   "OXICACHE"                        │
//! │ VERSION 2 bytes   u16 LE = 1                        │
//! │ FLAGS   2 bytes   u16 LE (reserved, must be 0)      │
//! │ N       4 bytes   u32 LE number of entries          │
//! │ ── per entry ──────────────────────────────────────  │
//! │   key_len   4 bytes  u32 LE                         │
//! │   key       key_len bytes UTF-8                     │
//! │   val_len   4 bytes  u32 LE                         │
//! │   value     val_len bytes raw bytes                 │
//! │   priority  4 bytes  u32 LE                         │
//! │   ttl_secs  8 bytes  u64 LE (0 = no TTL)           │
//! └─────────────────────────────────────────────────────┘
//! ```
//!
//! The format is intentionally simple and forward-compatible: unknown flags
//! and trailing data are ignored on read.

use std::io::{self, Read, Write};
use thiserror::Error;

// ── Magic + version ───────────────────────────────────────────────────────────

const MAGIC: &[u8; 8] = b"OXICACHE";
const FORMAT_VERSION: u16 = 1;

// ── Errors ────────────────────────────────────────────────────────────────────

/// Errors returned by serialization / deserialization functions.
#[derive(Debug, Error)]
pub enum SerializeError {
    /// An I/O error occurred while reading or writing.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// The magic header did not match the expected value.
    #[error("invalid magic header: expected 'OXICACHE', got {0:?}")]
    InvalidMagic([u8; 8]),

    /// The format version is not supported.
    #[error("unsupported format version {0}; expected {FORMAT_VERSION}")]
    UnsupportedVersion(u16),

    /// A key could not be decoded as valid UTF-8.
    #[error("key at entry {0} is not valid UTF-8: {1}")]
    InvalidKeyUtf8(usize, std::string::FromUtf8Error),

    /// An entry's data length field exceeds the configured safety limit.
    #[error("entry {index} value length {actual} exceeds safety limit {limit}")]
    ValueTooLarge {
        /// Entry index in the stream.
        index: usize,
        /// Value length reported in the stream.
        actual: u32,
        /// Configured safety limit.
        limit: u32,
    },
}

// ── CacheRecord ───────────────────────────────────────────────────────────────

/// A single key-value record with optional metadata.
///
/// This is the unit of serialization: a collection of `CacheRecord`s
/// represents a complete cache snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheRecord {
    /// Cache key.
    pub key: String,
    /// Raw value bytes.
    pub value: Vec<u8>,
    /// Optional TTL in seconds (0 means no TTL / immortal).
    pub ttl_secs: u64,
    /// Priority tag (higher = more important to keep on restore).
    pub priority: u32,
}

impl CacheRecord {
    /// Create a new `CacheRecord` with the given key and value, no TTL,
    /// and default priority 0.
    pub fn new(key: impl Into<String>, value: Vec<u8>) -> Self {
        Self {
            key: key.into(),
            value,
            ttl_secs: 0,
            priority: 0,
        }
    }

    /// Set a TTL hint (seconds).  A value of `0` means no TTL.
    pub fn with_ttl(mut self, ttl_secs: u64) -> Self {
        self.ttl_secs = ttl_secs;
        self
    }

    /// Set the priority tag.
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }
}

// ── Serializer ────────────────────────────────────────────────────────────────

/// Write a collection of [`CacheRecord`]s to `writer` in the OXICACHE binary
/// format.
///
/// # Errors
///
/// Returns [`SerializeError::Io`] on any I/O failure.
pub fn serialize<W: Write>(writer: &mut W, records: &[CacheRecord]) -> Result<(), SerializeError> {
    // Magic + version + flags.
    writer.write_all(MAGIC)?;
    writer.write_all(&FORMAT_VERSION.to_le_bytes())?;
    let flags: u16 = 0;
    writer.write_all(&flags.to_le_bytes())?;

    // Entry count.
    let n = records.len() as u32;
    writer.write_all(&n.to_le_bytes())?;

    for rec in records {
        let key_bytes = rec.key.as_bytes();
        writer.write_all(&(key_bytes.len() as u32).to_le_bytes())?;
        writer.write_all(key_bytes)?;

        writer.write_all(&(rec.value.len() as u32).to_le_bytes())?;
        writer.write_all(&rec.value)?;

        writer.write_all(&rec.priority.to_le_bytes())?;
        writer.write_all(&rec.ttl_secs.to_le_bytes())?;
    }

    Ok(())
}

// ── Deserializer ──────────────────────────────────────────────────────────────

/// Configuration for the deserializer.
#[derive(Debug, Clone)]
pub struct DeserializeConfig {
    /// Maximum allowed value size in bytes.  Records whose value exceeds this
    /// limit are rejected with [`SerializeError::ValueTooLarge`].
    ///
    /// Default: 512 MiB.
    pub max_value_bytes: u32,
    /// Maximum number of records to restore.  Additional records in the
    /// stream are silently discarded.
    ///
    /// Default: `u32::MAX` (no limit).
    pub max_records: u32,
}

impl Default for DeserializeConfig {
    fn default() -> Self {
        Self {
            max_value_bytes: 512 * 1024 * 1024, // 512 MiB
            max_records: u32::MAX,
        }
    }
}

/// Read [`CacheRecord`]s from `reader` using the default [`DeserializeConfig`].
///
/// See [`deserialize_with_config`] for a version with explicit limits.
pub fn deserialize<R: Read>(reader: &mut R) -> Result<Vec<CacheRecord>, SerializeError> {
    deserialize_with_config(reader, &DeserializeConfig::default())
}

/// Read [`CacheRecord`]s from `reader` with an explicit [`DeserializeConfig`].
///
/// # Errors
///
/// * [`SerializeError::InvalidMagic`] — magic header mismatch.
/// * [`SerializeError::UnsupportedVersion`] — version field is not 1.
/// * [`SerializeError::InvalidKeyUtf8`] — key bytes are not valid UTF-8.
/// * [`SerializeError::ValueTooLarge`] — value exceeds `config.max_value_bytes`.
/// * [`SerializeError::Io`] — any underlying I/O failure.
pub fn deserialize_with_config<R: Read>(
    reader: &mut R,
    config: &DeserializeConfig,
) -> Result<Vec<CacheRecord>, SerializeError> {
    // Magic.
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(SerializeError::InvalidMagic(magic));
    }

    // Version.
    let mut ver_buf = [0u8; 2];
    reader.read_exact(&mut ver_buf)?;
    let version = u16::from_le_bytes(ver_buf);
    if version != FORMAT_VERSION {
        return Err(SerializeError::UnsupportedVersion(version));
    }

    // Flags (ignored for now, reserved for future use).
    let mut flags_buf = [0u8; 2];
    reader.read_exact(&mut flags_buf)?;
    // Flags consumed but not acted upon.

    // Entry count.
    let mut n_buf = [0u8; 4];
    reader.read_exact(&mut n_buf)?;
    let n = u32::from_le_bytes(n_buf);

    let to_read = n.min(config.max_records);
    let mut records = Vec::with_capacity(to_read as usize);

    for idx in 0..n as usize {
        // key_len + key
        let mut klen_buf = [0u8; 4];
        reader.read_exact(&mut klen_buf)?;
        let key_len = u32::from_le_bytes(klen_buf) as usize;
        let mut key_bytes = vec![0u8; key_len];
        reader.read_exact(&mut key_bytes)?;
        let key =
            String::from_utf8(key_bytes).map_err(|e| SerializeError::InvalidKeyUtf8(idx, e))?;

        // val_len + value
        let mut vlen_buf = [0u8; 4];
        reader.read_exact(&mut vlen_buf)?;
        let val_len = u32::from_le_bytes(vlen_buf);
        if val_len > config.max_value_bytes {
            return Err(SerializeError::ValueTooLarge {
                index: idx,
                actual: val_len,
                limit: config.max_value_bytes,
            });
        }
        let mut value = vec![0u8; val_len as usize];
        reader.read_exact(&mut value)?;

        // priority
        let mut prio_buf = [0u8; 4];
        reader.read_exact(&mut prio_buf)?;
        let priority = u32::from_le_bytes(prio_buf);

        // ttl_secs
        let mut ttl_buf = [0u8; 8];
        reader.read_exact(&mut ttl_buf)?;
        let ttl_secs = u64::from_le_bytes(ttl_buf);

        if (idx as u32) < config.max_records {
            records.push(CacheRecord {
                key,
                value,
                ttl_secs,
                priority,
            });
        }
        // Records beyond max_records: we already read the bytes above; they
        // are discarded here.
    }

    Ok(records)
}

// ── Convenience: file-based helpers ───────────────────────────────────────────

/// Persist `records` to the file at `path`, creating or truncating it.
///
/// Equivalent to opening the file and calling [`serialize`].
pub fn save_to_file(path: &std::path::Path, records: &[CacheRecord]) -> Result<(), SerializeError> {
    let mut file = std::fs::File::create(path)?;
    serialize(&mut file, records)
}

/// Restore records from the file at `path`.
///
/// Equivalent to opening the file and calling [`deserialize`].
pub fn load_from_file(path: &std::path::Path) -> Result<Vec<CacheRecord>, SerializeError> {
    let mut file = std::fs::File::open(path)?;
    deserialize(&mut file)
}

/// Restore records from the file at `path` with explicit limits.
pub fn load_from_file_with_config(
    path: &std::path::Path,
    config: &DeserializeConfig,
) -> Result<Vec<CacheRecord>, SerializeError> {
    let mut file = std::fs::File::open(path)?;
    deserialize_with_config(&mut file, config)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn roundtrip(records: &[CacheRecord]) -> Vec<CacheRecord> {
        let mut buf = Vec::new();
        serialize(&mut buf, records).expect("serialize should succeed");
        let mut cursor = Cursor::new(&buf);
        deserialize(&mut cursor).expect("deserialize should succeed")
    }

    // 1. Empty snapshot round-trips cleanly
    #[test]
    fn test_empty_roundtrip() {
        let records: Vec<CacheRecord> = Vec::new();
        let restored = roundtrip(&records);
        assert!(restored.is_empty());
    }

    // 2. Single record round-trips correctly
    #[test]
    fn test_single_record_roundtrip() {
        let records = vec![CacheRecord::new("key-001", b"hello world".to_vec())];
        let restored = roundtrip(&records);
        assert_eq!(restored.len(), 1);
        assert_eq!(restored[0].key, "key-001");
        assert_eq!(restored[0].value, b"hello world");
        assert_eq!(restored[0].ttl_secs, 0);
        assert_eq!(restored[0].priority, 0);
    }

    // 3. Multiple records preserve order and content
    #[test]
    fn test_multiple_records_roundtrip() {
        let records: Vec<CacheRecord> = (0..20u32)
            .map(|i| {
                CacheRecord::new(format!("seg-{i:04}"), vec![i as u8; 128])
                    .with_ttl(300)
                    .with_priority(i % 5)
            })
            .collect();
        let restored = roundtrip(&records);
        assert_eq!(restored.len(), records.len());
        for (orig, rest) in records.iter().zip(restored.iter()) {
            assert_eq!(orig, rest);
        }
    }

    // 4. TTL and priority survive round-trip
    #[test]
    fn test_ttl_and_priority_roundtrip() {
        let rec = CacheRecord::new("manifest.m3u8", b"#EXTM3U".to_vec())
            .with_ttl(30)
            .with_priority(10);
        let restored = roundtrip(std::slice::from_ref(&rec));
        assert_eq!(restored[0].ttl_secs, 30);
        assert_eq!(restored[0].priority, 10);
    }

    // 5. Binary values (non-UTF8 payload) round-trip correctly
    #[test]
    fn test_binary_value_roundtrip() {
        let value: Vec<u8> = (0u8..=255).collect();
        let records = vec![CacheRecord::new("binary", value.clone())];
        let restored = roundtrip(&records);
        assert_eq!(restored[0].value, value);
    }

    // 6. Unicode key round-trips correctly
    #[test]
    fn test_unicode_key_roundtrip() {
        let records = vec![CacheRecord::new("媒体-segment-001", vec![1, 2, 3])];
        let restored = roundtrip(&records);
        assert_eq!(restored[0].key, "媒体-segment-001");
    }

    // 7. Invalid magic header returns error
    #[test]
    fn test_invalid_magic() {
        let garbage = b"GARBAGE_HEADER_DATA";
        let mut cursor = Cursor::new(garbage);
        let result = deserialize(&mut cursor);
        assert!(
            matches!(result, Err(SerializeError::InvalidMagic(_))),
            "expected InvalidMagic, got {result:?}"
        );
    }

    // 8. Wrong version returns error
    #[test]
    fn test_wrong_version() {
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&9999u16.to_le_bytes()); // bad version
        buf.extend_from_slice(&0u16.to_le_bytes()); // flags
        buf.extend_from_slice(&0u32.to_le_bytes()); // 0 records
        let mut cursor = Cursor::new(&buf);
        let result = deserialize(&mut cursor);
        assert!(
            matches!(result, Err(SerializeError::UnsupportedVersion(9999))),
            "expected UnsupportedVersion"
        );
    }

    // 9. max_records limit in DeserializeConfig
    #[test]
    fn test_max_records_limit() {
        let records: Vec<CacheRecord> = (0..10u32)
            .map(|i| CacheRecord::new(format!("k{i}"), vec![i as u8]))
            .collect();
        let mut buf = Vec::new();
        serialize(&mut buf, &records).expect("ok");
        let config = DeserializeConfig {
            max_records: 3,
            ..Default::default()
        };
        let mut cursor = Cursor::new(&buf);
        let restored = deserialize_with_config(&mut cursor, &config).expect("ok");
        assert_eq!(restored.len(), 3, "only 3 records should be restored");
    }

    // 10. max_value_bytes limit rejects oversized records
    #[test]
    fn test_max_value_bytes_rejected() {
        let records = vec![CacheRecord::new("big", vec![0u8; 1024])];
        let mut buf = Vec::new();
        serialize(&mut buf, &records).expect("ok");
        let config = DeserializeConfig {
            max_value_bytes: 128, // smaller than 1024
            ..Default::default()
        };
        let mut cursor = Cursor::new(&buf);
        let result = deserialize_with_config(&mut cursor, &config);
        assert!(
            matches!(result, Err(SerializeError::ValueTooLarge { .. })),
            "expected ValueTooLarge"
        );
    }

    // 11. File-based save/load round-trip
    #[test]
    fn test_file_save_load_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join("oximedia_cache_test_serialization.bin");
        let records = vec![
            CacheRecord::new("segment-1", b"data1".to_vec()).with_ttl(60),
            CacheRecord::new("segment-2", b"data2".to_vec()).with_priority(5),
        ];
        save_to_file(&path, &records).expect("save should succeed");
        let restored = load_from_file(&path).expect("load should succeed");
        assert_eq!(restored.len(), 2);
        assert_eq!(restored[0].key, "segment-1");
        assert_eq!(restored[1].priority, 5);
        // Clean up.
        let _ = std::fs::remove_file(&path);
    }

    // 12. Empty key is valid
    #[test]
    fn test_empty_key_roundtrip() {
        let records = vec![CacheRecord::new("", b"value".to_vec())];
        let restored = roundtrip(&records);
        assert_eq!(restored[0].key, "");
    }

    // 13. Empty value is valid
    #[test]
    fn test_empty_value_roundtrip() {
        let records = vec![CacheRecord::new("empty-val", Vec::new())];
        let restored = roundtrip(&records);
        assert!(restored[0].value.is_empty());
    }

    // 14. Serialized bytes start with magic
    #[test]
    fn test_serialized_magic_prefix() {
        let mut buf = Vec::new();
        serialize(&mut buf, &[]).expect("ok");
        assert_eq!(&buf[..8], MAGIC);
    }

    // 15. CacheRecord builder API
    #[test]
    fn test_cache_record_builder() {
        let rec = CacheRecord::new("k", vec![1, 2])
            .with_ttl(120)
            .with_priority(7);
        assert_eq!(rec.key, "k");
        assert_eq!(rec.ttl_secs, 120);
        assert_eq!(rec.priority, 7);
    }
}
