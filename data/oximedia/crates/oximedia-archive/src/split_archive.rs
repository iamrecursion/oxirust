//! Multi-part archive splitting and reassembly.
//!
//! Splits a collection of (path, data) entries across multiple part files,
//! each bounded by a configurable maximum size. Parts use a simple binary
//! container format with per-part checksums.

use crate::archive_verify::sha256_hex;
use crate::ArchiveError;
use std::path::Path;

// ---------------------------------------------------------------------------
// Configuration & public types
// ---------------------------------------------------------------------------

/// Configuration for splitting an archive into parts.
#[derive(Debug, Clone)]
pub struct SplitArchiveConfig {
    /// Maximum bytes per part file (e.g. 1_073_741_824 for 1 GiB).
    pub part_size_bytes: u64,
    /// Base name used to derive part filenames, e.g. `"backup"` →
    /// `"backup.part001"`, `"backup.part002"`, etc.
    pub base_name: String,
    /// Whether to compress entry data when packing (uses the streaming LZ77
    /// compressor from this crate; currently stored as raw for simplicity
    /// to keep the container format self-contained; set to false for now).
    pub compression: bool,
}

/// Metadata for a single part file.
#[derive(Debug, Clone)]
pub struct ArchivePart {
    /// Zero-based part index.
    pub index: u32,
    /// File path string.
    pub path: String,
    /// Actual byte size of the part file on disk.
    pub size_bytes: u64,
    /// SHA-256 hex checksum of the part file content.
    pub checksum: String,
    /// Inclusive range `(first_entry_index, last_entry_index)` into the
    /// original entry slice that this part contains.
    pub entry_range: (usize, usize),
}

/// Manifest returned after splitting, and required for reassembly.
#[derive(Debug, Clone)]
pub struct SplitArchiveManifest {
    pub total_parts: u32,
    pub total_entries: usize,
    pub total_original_bytes: u64,
    pub total_compressed_bytes: u64,
    pub parts: Vec<ArchivePart>,
}

// ---------------------------------------------------------------------------
// Part container format
// ---------------------------------------------------------------------------
//
// Each part file layout:
//
//   [magic:        4 bytes]   0x4F 0x41 0x52 0x43  ("OARC")
//   [part_index:   4 bytes]   u32 LE
//   [part_count:   4 bytes]   u32 LE
//   [entry_count:  4 bytes]   u32 LE
//   [entries...]
//
// Each entry:
//   [path_len:   2 bytes]   u16 LE
//   [path:       path_len bytes]
//   [data_len:   4 bytes]   u32 LE
//   [data:       data_len bytes]

const MAGIC: [u8; 4] = [0x4F, 0x41, 0x52, 0x43];
const HEADER_SIZE: usize = 4 + 4 + 4 + 4; // 16 bytes

/// Encode a single part into its binary representation.
fn encode_part(
    part_index: u32,
    part_count: u32,
    entries: &[(&str, &[u8])],
) -> Result<Vec<u8>, ArchiveError> {
    let entry_count = entries.len() as u32;
    let mut out = Vec::with_capacity(HEADER_SIZE + 1024);

    // Header
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&part_index.to_le_bytes());
    out.extend_from_slice(&part_count.to_le_bytes());
    out.extend_from_slice(&entry_count.to_le_bytes());

    // Entries
    for (path, data) in entries {
        let path_bytes = path.as_bytes();
        let path_len = path_bytes.len();
        if path_len > u16::MAX as usize {
            return Err(ArchiveError::Validation(format!(
                "path too long ({path_len} bytes): {path}"
            )));
        }
        let data_len = data.len();
        if data_len > u32::MAX as usize {
            return Err(ArchiveError::Validation(format!(
                "entry data too large ({data_len} bytes) for part format"
            )));
        }
        out.extend_from_slice(&(path_len as u16).to_le_bytes());
        out.extend_from_slice(path_bytes);
        out.extend_from_slice(&(data_len as u32).to_le_bytes());
        out.extend_from_slice(data);
    }

    Ok(out)
}

/// Decode a part from its binary representation.
/// Returns `(part_index, part_count, Vec<(path, data)>)`.
fn decode_part(data: &[u8]) -> Result<(u32, u32, Vec<(String, Vec<u8>)>), ArchiveError> {
    if data.len() < HEADER_SIZE {
        return Err(ArchiveError::Corruption(format!(
            "part too small: {} bytes",
            data.len()
        )));
    }
    if data[..4] != MAGIC {
        return Err(ArchiveError::Corruption("part magic mismatch".to_string()));
    }

    let part_index = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    let part_count = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
    let entry_count = u32::from_le_bytes([data[12], data[13], data[14], data[15]]);

    let mut cursor = HEADER_SIZE;
    let mut entries = Vec::with_capacity(entry_count as usize);

    for i in 0..entry_count {
        if cursor + 2 > data.len() {
            return Err(ArchiveError::Corruption(format!(
                "truncated path_len for entry {i}"
            )));
        }
        let path_len = u16::from_le_bytes([data[cursor], data[cursor + 1]]) as usize;
        cursor += 2;

        if cursor + path_len > data.len() {
            return Err(ArchiveError::Corruption(format!(
                "truncated path for entry {i}"
            )));
        }
        let path = std::str::from_utf8(&data[cursor..cursor + path_len])
            .map_err(|e| ArchiveError::Corruption(format!("invalid UTF-8 in path: {e}")))?
            .to_string();
        cursor += path_len;

        if cursor + 4 > data.len() {
            return Err(ArchiveError::Corruption(format!(
                "truncated data_len for entry {i}"
            )));
        }
        let data_len = u32::from_le_bytes([
            data[cursor],
            data[cursor + 1],
            data[cursor + 2],
            data[cursor + 3],
        ]) as usize;
        cursor += 4;

        if cursor + data_len > data.len() {
            return Err(ArchiveError::Corruption(format!(
                "truncated data for entry {i}"
            )));
        }
        let entry_data = data[cursor..cursor + data_len].to_vec();
        cursor += data_len;

        entries.push((path, entry_data));
    }

    Ok((part_index, part_count, entries))
}

/// Calculate the serialized size of a single entry (for pre-flight size checks).
fn entry_wire_size(path: &str, data: &[u8]) -> usize {
    2 + path.len() + 4 + data.len()
}

// ---------------------------------------------------------------------------
// SplitArchiver
// ---------------------------------------------------------------------------

/// Splits entries into multi-part archives and reassembles them.
pub struct SplitArchiver {
    config: SplitArchiveConfig,
}

impl SplitArchiver {
    /// Create a new archiver with the given configuration.
    pub fn new(config: SplitArchiveConfig) -> Self {
        Self { config }
    }

    /// Split `entries` into part files written to `output_dir`.
    ///
    /// Returns a [`SplitArchiveManifest`] describing the parts.
    pub fn split(
        &self,
        entries: &[(String, Vec<u8>)],
        output_dir: &Path,
    ) -> Result<SplitArchiveManifest, ArchiveError> {
        if entries.is_empty() {
            return Ok(SplitArchiveManifest {
                total_parts: 0,
                total_entries: 0,
                total_original_bytes: 0,
                total_compressed_bytes: 0,
                parts: vec![],
            });
        }

        // Plan parts: group entries greedily by size.
        let max_entry_bytes = self.config.part_size_bytes.max(1) as usize;

        // Each part holds a list of indices into `entries`.
        let mut part_groups: Vec<(usize, usize)> = Vec::new(); // (first_idx, last_idx)
        let mut current_first = 0usize;
        let mut current_size = HEADER_SIZE;

        for (i, (path, data)) in entries.iter().enumerate() {
            let wire = entry_wire_size(path, data);
            // If adding this entry would exceed the limit and we already have
            // at least one entry in the current part, flush the current part.
            if current_size + wire > max_entry_bytes && i > current_first {
                part_groups.push((current_first, i - 1));
                current_first = i;
                current_size = HEADER_SIZE;
            }
            current_size += wire;
        }
        // Always flush the last group.
        part_groups.push((current_first, entries.len() - 1));

        let total_parts = part_groups.len() as u32;
        let mut parts_meta: Vec<ArchivePart> = Vec::with_capacity(part_groups.len());
        let mut total_original_bytes = 0u64;

        for (part_idx, (first, last)) in part_groups.iter().enumerate() {
            let part_entries: Vec<(&str, &[u8])> = entries[*first..=*last]
                .iter()
                .map(|(p, d)| (p.as_str(), d.as_slice()))
                .collect();

            for (_, d) in &part_entries {
                total_original_bytes += d.len() as u64;
            }

            let part_data = encode_part(part_idx as u32, total_parts, &part_entries)?;
            let part_checksum = sha256_hex(&part_data);
            let part_filename = format!("{}.part{:03}", self.config.base_name, part_idx + 1);
            let part_path = output_dir.join(&part_filename);

            std::fs::write(&part_path, &part_data).map_err(ArchiveError::Io)?;

            parts_meta.push(ArchivePart {
                index: part_idx as u32,
                path: part_path.display().to_string(),
                size_bytes: part_data.len() as u64,
                checksum: part_checksum,
                entry_range: (*first, *last),
            });
        }

        Ok(SplitArchiveManifest {
            total_parts,
            total_entries: entries.len(),
            total_original_bytes,
            // No compression in this impl — compressed == original.
            total_compressed_bytes: total_original_bytes,
            parts: parts_meta,
        })
    }

    /// Reassemble entries from a set of part files described by `manifest`.
    ///
    /// Reads parts from `parts_dir` (using the filenames stored in each
    /// `ArchivePart.path`), verifies checksums, and returns entries in order.
    pub fn reassemble(
        manifest: &SplitArchiveManifest,
        parts_dir: &Path,
    ) -> Result<Vec<(String, Vec<u8>)>, ArchiveError> {
        if manifest.total_parts == 0 {
            return Ok(Vec::new());
        }

        let mut all_entries: Vec<(String, Vec<u8>)> = Vec::with_capacity(manifest.total_entries);

        for part_meta in &manifest.parts {
            // Resolve path: use just the filename component and join with parts_dir,
            // so the manifest can be moved to a different directory.
            let part_filename = Path::new(&part_meta.path).file_name().ok_or_else(|| {
                ArchiveError::Validation(format!("invalid part path: {}", part_meta.path))
            })?;
            let part_path = parts_dir.join(part_filename);

            let part_data = std::fs::read(&part_path).map_err(ArchiveError::Io)?;

            // Verify checksum.
            let actual_checksum = sha256_hex(&part_data);
            if actual_checksum != part_meta.checksum {
                return Err(ArchiveError::ChecksumMismatch {
                    expected: part_meta.checksum.clone(),
                    actual: actual_checksum,
                });
            }

            let (decoded_index, decoded_total, entries) = decode_part(&part_data)?;
            if decoded_index != part_meta.index {
                return Err(ArchiveError::Corruption(format!(
                    "part index mismatch: expected {}, got {decoded_index}",
                    part_meta.index
                )));
            }
            if decoded_total != manifest.total_parts {
                return Err(ArchiveError::Corruption(format!(
                    "part_count mismatch: expected {}, got {decoded_total}",
                    manifest.total_parts
                )));
            }

            all_entries.extend(entries);
        }

        Ok(all_entries)
    }
}

// ---------------------------------------------------------------------------
// Configurable split strategies
// ---------------------------------------------------------------------------

/// Strategy for splitting entries into parts.
#[derive(Debug, Clone)]
pub enum SplitStrategy {
    /// Split by maximum byte size per part (default behavior).
    BySize {
        /// Maximum bytes per part.
        max_bytes: u64,
    },
    /// Split by date (epoch seconds), grouping entries whose timestamps fall
    /// within the same time bucket.
    ByDate {
        /// Duration of each time bucket in seconds.
        bucket_duration_secs: u64,
    },
    /// Split by collection / grouping key. Each distinct key gets its own
    /// part (or set of parts if a single key exceeds `max_bytes_per_part`).
    ByCollection {
        /// Maximum bytes per part within a collection.
        max_bytes_per_part: u64,
    },
    /// Fixed number of entries per part.
    ByCount {
        /// Maximum entries per part.
        max_entries: usize,
    },
}

/// An entry with metadata used for strategy-based splitting.
#[derive(Debug, Clone)]
pub struct SplitEntry {
    /// Entry path/name.
    pub path: String,
    /// Entry data.
    pub data: Vec<u8>,
    /// Creation/modification timestamp (epoch seconds).
    pub timestamp_secs: u64,
    /// Collection or grouping key (for `ByCollection` strategy).
    pub collection_key: String,
}

impl SplitEntry {
    /// Create a new split entry.
    #[must_use]
    pub fn new(
        path: impl Into<String>,
        data: Vec<u8>,
        timestamp_secs: u64,
        collection_key: impl Into<String>,
    ) -> Self {
        Self {
            path: path.into(),
            data,
            timestamp_secs,
            collection_key: collection_key.into(),
        }
    }

    /// Wire size of this entry in the archive format.
    #[allow(dead_code)]
    pub(crate) fn wire_size(&self) -> usize {
        entry_wire_size(&self.path, &self.data)
    }
}

/// Result of strategy-based splitting, with group labels.
#[derive(Debug, Clone)]
pub struct StrategyManifest {
    /// The base split manifest.
    pub manifest: SplitArchiveManifest,
    /// Label for each part (e.g., date range, collection name).
    pub part_labels: Vec<String>,
    /// The strategy that was used.
    pub strategy_name: String,
}

/// Split entries using a configurable strategy.
pub struct StrategySplitter;

impl StrategySplitter {
    /// Split entries into groups according to the strategy, then encode each
    /// group as one or more parts.
    pub fn split(
        entries: &[SplitEntry],
        strategy: &SplitStrategy,
        base_name: &str,
        output_dir: &Path,
    ) -> Result<StrategyManifest, ArchiveError> {
        match strategy {
            SplitStrategy::BySize { max_bytes } => {
                Self::split_by_size(entries, *max_bytes, base_name, output_dir)
            }
            SplitStrategy::ByDate {
                bucket_duration_secs,
            } => Self::split_by_date(entries, *bucket_duration_secs, base_name, output_dir),
            SplitStrategy::ByCollection { max_bytes_per_part } => {
                Self::split_by_collection(entries, *max_bytes_per_part, base_name, output_dir)
            }
            SplitStrategy::ByCount { max_entries } => {
                Self::split_by_count(entries, *max_entries, base_name, output_dir)
            }
        }
    }

    /// Split by size (delegates to existing SplitArchiver logic).
    fn split_by_size(
        entries: &[SplitEntry],
        max_bytes: u64,
        base_name: &str,
        output_dir: &Path,
    ) -> Result<StrategyManifest, ArchiveError> {
        let plain_entries: Vec<(String, Vec<u8>)> = entries
            .iter()
            .map(|e| (e.path.clone(), e.data.clone()))
            .collect();

        let config = SplitArchiveConfig {
            part_size_bytes: max_bytes,
            base_name: base_name.to_string(),
            compression: false,
        };
        let archiver = SplitArchiver::new(config);
        let manifest = archiver.split(&plain_entries, output_dir)?;

        let part_labels = manifest
            .parts
            .iter()
            .map(|p| format!("size-part-{:03}", p.index + 1))
            .collect();

        Ok(StrategyManifest {
            manifest,
            part_labels,
            strategy_name: "by_size".to_string(),
        })
    }

    /// Split by date bucket.
    fn split_by_date(
        entries: &[SplitEntry],
        bucket_secs: u64,
        base_name: &str,
        output_dir: &Path,
    ) -> Result<StrategyManifest, ArchiveError> {
        if entries.is_empty() || bucket_secs == 0 {
            return Ok(StrategyManifest {
                manifest: SplitArchiveManifest {
                    total_parts: 0,
                    total_entries: 0,
                    total_original_bytes: 0,
                    total_compressed_bytes: 0,
                    parts: vec![],
                },
                part_labels: vec![],
                strategy_name: "by_date".to_string(),
            });
        }

        // Group entries by date bucket
        let mut buckets: std::collections::BTreeMap<u64, Vec<&SplitEntry>> =
            std::collections::BTreeMap::new();
        for entry in entries {
            let bucket_key = entry.timestamp_secs / bucket_secs;
            buckets.entry(bucket_key).or_default().push(entry);
        }

        let total_parts = buckets.len() as u32;
        let mut parts_meta = Vec::new();
        let mut part_labels = Vec::new();
        let mut total_original_bytes = 0u64;
        let mut global_entry_idx = 0usize;

        for (part_idx, (bucket_key, bucket_entries)) in buckets.iter().enumerate() {
            let start_secs = bucket_key * bucket_secs;
            let end_secs = start_secs + bucket_secs;
            part_labels.push(format!("{start_secs}-{end_secs}s"));

            let part_entries: Vec<(&str, &[u8])> = bucket_entries
                .iter()
                .map(|e| (e.path.as_str(), e.data.as_slice()))
                .collect();

            for (_, d) in &part_entries {
                total_original_bytes += d.len() as u64;
            }

            let first_idx = global_entry_idx;
            let last_idx = global_entry_idx + bucket_entries.len() - 1;
            global_entry_idx += bucket_entries.len();

            let part_data = encode_part(part_idx as u32, total_parts, &part_entries)?;
            let part_checksum = sha256_hex(&part_data);
            let part_filename = format!("{base_name}.part{:03}", part_idx + 1);
            let part_path = output_dir.join(&part_filename);

            std::fs::write(&part_path, &part_data).map_err(ArchiveError::Io)?;

            parts_meta.push(ArchivePart {
                index: part_idx as u32,
                path: part_path.display().to_string(),
                size_bytes: part_data.len() as u64,
                checksum: part_checksum,
                entry_range: (first_idx, last_idx),
            });
        }

        Ok(StrategyManifest {
            manifest: SplitArchiveManifest {
                total_parts,
                total_entries: entries.len(),
                total_original_bytes,
                total_compressed_bytes: total_original_bytes,
                parts: parts_meta,
            },
            part_labels,
            strategy_name: "by_date".to_string(),
        })
    }

    /// Split by collection key.
    fn split_by_collection(
        entries: &[SplitEntry],
        max_bytes_per_part: u64,
        base_name: &str,
        output_dir: &Path,
    ) -> Result<StrategyManifest, ArchiveError> {
        if entries.is_empty() {
            return Ok(StrategyManifest {
                manifest: SplitArchiveManifest {
                    total_parts: 0,
                    total_entries: 0,
                    total_original_bytes: 0,
                    total_compressed_bytes: 0,
                    parts: vec![],
                },
                part_labels: vec![],
                strategy_name: "by_collection".to_string(),
            });
        }

        // Group by collection key
        let mut groups: std::collections::BTreeMap<&str, Vec<&SplitEntry>> =
            std::collections::BTreeMap::new();
        for entry in entries {
            groups.entry(&entry.collection_key).or_default().push(entry);
        }

        // For each collection, create parts respecting max_bytes
        let mut all_parts = Vec::new();
        let mut part_labels = Vec::new();
        let mut total_original_bytes = 0u64;
        let mut global_entry_idx = 0usize;
        let mut part_index = 0u32;

        for (collection_key, collection_entries) in &groups {
            let plain_entries: Vec<(String, Vec<u8>)> = collection_entries
                .iter()
                .map(|e| (e.path.clone(), e.data.clone()))
                .collect();

            let config = SplitArchiveConfig {
                part_size_bytes: max_bytes_per_part,
                base_name: format!("{base_name}_{collection_key}"),
                compression: false,
            };
            let archiver = SplitArchiver::new(config);
            let sub_manifest = archiver.split(&plain_entries, output_dir)?;

            for part in &sub_manifest.parts {
                let first = global_entry_idx + part.entry_range.0;
                let last = global_entry_idx + part.entry_range.1;

                all_parts.push(ArchivePart {
                    index: part_index,
                    path: part.path.clone(),
                    size_bytes: part.size_bytes,
                    checksum: part.checksum.clone(),
                    entry_range: (first, last),
                });
                part_labels.push(format!("{collection_key}-part-{:03}", part.index + 1));
                part_index += 1;
            }

            total_original_bytes += sub_manifest.total_original_bytes;
            global_entry_idx += collection_entries.len();
        }

        Ok(StrategyManifest {
            manifest: SplitArchiveManifest {
                total_parts: part_index,
                total_entries: entries.len(),
                total_original_bytes,
                total_compressed_bytes: total_original_bytes,
                parts: all_parts,
            },
            part_labels,
            strategy_name: "by_collection".to_string(),
        })
    }

    /// Split by fixed entry count.
    fn split_by_count(
        entries: &[SplitEntry],
        max_entries: usize,
        base_name: &str,
        output_dir: &Path,
    ) -> Result<StrategyManifest, ArchiveError> {
        if entries.is_empty() || max_entries == 0 {
            return Ok(StrategyManifest {
                manifest: SplitArchiveManifest {
                    total_parts: 0,
                    total_entries: 0,
                    total_original_bytes: 0,
                    total_compressed_bytes: 0,
                    parts: vec![],
                },
                part_labels: vec![],
                strategy_name: "by_count".to_string(),
            });
        }

        let chunks: Vec<&[SplitEntry]> = entries.chunks(max_entries).collect();
        let total_parts = chunks.len() as u32;
        let mut parts_meta = Vec::new();
        let mut part_labels = Vec::new();
        let mut total_original_bytes = 0u64;
        let mut global_idx = 0usize;

        for (part_idx, chunk) in chunks.iter().enumerate() {
            let part_entries: Vec<(&str, &[u8])> = chunk
                .iter()
                .map(|e| (e.path.as_str(), e.data.as_slice()))
                .collect();

            for (_, d) in &part_entries {
                total_original_bytes += d.len() as u64;
            }

            let first = global_idx;
            let last = global_idx + chunk.len() - 1;
            global_idx += chunk.len();

            let part_data = encode_part(part_idx as u32, total_parts, &part_entries)?;
            let checksum = sha256_hex(&part_data);
            let filename = format!("{base_name}.part{:03}", part_idx + 1);
            let path = output_dir.join(&filename);

            std::fs::write(&path, &part_data).map_err(ArchiveError::Io)?;

            parts_meta.push(ArchivePart {
                index: part_idx as u32,
                path: path.display().to_string(),
                size_bytes: part_data.len() as u64,
                checksum,
                entry_range: (first, last),
            });
            part_labels.push(format!("count-part-{:03}", part_idx + 1));
        }

        Ok(StrategyManifest {
            manifest: SplitArchiveManifest {
                total_parts,
                total_entries: entries.len(),
                total_original_bytes,
                total_compressed_bytes: total_original_bytes,
                parts: parts_meta,
            },
            part_labels,
            strategy_name: "by_count".to_string(),
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entries(n: usize, data_size: usize) -> Vec<(String, Vec<u8>)> {
        (0..n)
            .map(|i| {
                let data: Vec<u8> = (0u8..=255).cycle().take(data_size).collect();
                (format!("entry_{i:04}.bin"), data)
            })
            .collect()
    }

    fn temp_dir(suffix: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("oximedia_split_test_{suffix}"));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    #[test]
    fn test_split_single_part() {
        let dir = temp_dir("single");
        let entries = make_entries(3, 100);
        let config = SplitArchiveConfig {
            part_size_bytes: 1_000_000,
            base_name: "archive".to_string(),
            compression: false,
        };
        let archiver = SplitArchiver::new(config);
        let manifest = archiver.split(&entries, &dir).expect("split failed");
        assert_eq!(manifest.total_parts, 1);
        assert_eq!(manifest.total_entries, 3);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_split_multiple_parts() {
        let dir = temp_dir("multi");
        let entries = make_entries(10, 200);
        // Each entry ~200 bytes + overhead; set part size to ~500 bytes → multiple parts.
        let config = SplitArchiveConfig {
            part_size_bytes: 700,
            base_name: "chunk".to_string(),
            compression: false,
        };
        let archiver = SplitArchiver::new(config);
        let manifest = archiver.split(&entries, &dir).expect("split failed");
        assert!(manifest.total_parts > 1, "expected multiple parts");
        assert_eq!(manifest.total_entries, 10);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_split_and_reassemble_roundtrip() {
        let dir = temp_dir("roundtrip");
        let entries = make_entries(5, 512);
        let config = SplitArchiveConfig {
            part_size_bytes: 1500,
            base_name: "rt".to_string(),
            compression: false,
        };
        let archiver = SplitArchiver::new(config);
        let manifest = archiver.split(&entries, &dir).expect("split failed");
        let recovered = SplitArchiver::reassemble(&manifest, &dir).expect("reassemble failed");
        assert_eq!(recovered.len(), entries.len());
        for (original, recovered) in entries.iter().zip(recovered.iter()) {
            assert_eq!(original.0, recovered.0, "path mismatch");
            assert_eq!(original.1, recovered.1, "data mismatch");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_split_empty_entries() {
        let dir = temp_dir("empty");
        let entries: Vec<(String, Vec<u8>)> = vec![];
        let config = SplitArchiveConfig {
            part_size_bytes: 1_000_000,
            base_name: "empty".to_string(),
            compression: false,
        };
        let archiver = SplitArchiver::new(config);
        let manifest = archiver.split(&entries, &dir).expect("split failed");
        assert_eq!(manifest.total_parts, 0);
        assert_eq!(manifest.total_entries, 0);
        let recovered = SplitArchiver::reassemble(&manifest, &dir).expect("reassemble failed");
        assert!(recovered.is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_split_single_large_entry() {
        let dir = temp_dir("large");
        // One entry larger than part_size_bytes (should still be emitted in one part).
        let entries = vec![("big.bin".to_string(), vec![0xABu8; 2048])];
        let config = SplitArchiveConfig {
            part_size_bytes: 512,
            base_name: "large".to_string(),
            compression: false,
        };
        let archiver = SplitArchiver::new(config);
        let manifest = archiver.split(&entries, &dir).expect("split failed");
        // Single entry can't be split below part granularity; it goes into one part.
        assert_eq!(manifest.total_entries, 1);
        let recovered = SplitArchiver::reassemble(&manifest, &dir).expect("reassemble failed");
        assert_eq!(recovered[0].1, entries[0].1);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_reassemble_checksum_verification() {
        let dir = temp_dir("cksum_verify");
        let entries = make_entries(2, 100);
        let config = SplitArchiveConfig {
            part_size_bytes: 1_000_000,
            base_name: "ckv".to_string(),
            compression: false,
        };
        let archiver = SplitArchiver::new(config);
        let manifest = archiver.split(&entries, &dir).expect("split failed");

        // Corrupt the part file.
        let part_path = manifest.parts[0].path.clone();
        let p = Path::new(&part_path);
        let mut data = std::fs::read(p).expect("read part");
        if !data.is_empty() {
            let last = data.len() - 1;
            data[last] ^= 0xFF;
        }
        std::fs::write(p, &data).expect("write corrupted part");

        let result = SplitArchiver::reassemble(&manifest, &dir);
        assert!(result.is_err(), "should detect corruption via checksum");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_split_part_filenames() {
        let dir = temp_dir("filenames");
        let entries = make_entries(4, 50);
        let config = SplitArchiveConfig {
            part_size_bytes: 300,
            base_name: "backup".to_string(),
            compression: false,
        };
        let archiver = SplitArchiver::new(config);
        let manifest = archiver.split(&entries, &dir).expect("split failed");
        for (i, part) in manifest.parts.iter().enumerate() {
            let expected_name = format!("backup.part{:03}", i + 1);
            assert!(
                part.path.ends_with(&expected_name),
                "unexpected part filename: {}",
                part.path
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_split_total_bytes_accounting() {
        let dir = temp_dir("bytes");
        let entries = make_entries(6, 256);
        let expected_bytes: u64 = entries.iter().map(|(_, d)| d.len() as u64).sum();
        let config = SplitArchiveConfig {
            part_size_bytes: 1000,
            base_name: "bytes".to_string(),
            compression: false,
        };
        let archiver = SplitArchiver::new(config);
        let manifest = archiver.split(&entries, &dir).expect("split failed");
        assert_eq!(manifest.total_original_bytes, expected_bytes);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_split_many_small_entries() {
        let dir = temp_dir("many_small");
        let entries: Vec<(String, Vec<u8>)> = (0..50)
            .map(|i| (format!("f{i}.txt"), b"hi".to_vec()))
            .collect();
        let config = SplitArchiveConfig {
            part_size_bytes: 200,
            base_name: "sm".to_string(),
            compression: false,
        };
        let archiver = SplitArchiver::new(config);
        let manifest = archiver.split(&entries, &dir).expect("split failed");
        let recovered = SplitArchiver::reassemble(&manifest, &dir).expect("reassemble failed");
        assert_eq!(recovered.len(), 50);
        for (i, (path, data)) in recovered.iter().enumerate() {
            assert_eq!(path, &format!("f{i}.txt"));
            assert_eq!(data, b"hi");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    // --- Strategy-based split tests ---

    fn make_split_entries(n: usize, data_size: usize) -> Vec<SplitEntry> {
        (0..n)
            .map(|i| {
                let data: Vec<u8> = (0u8..=255).cycle().take(data_size).collect();
                SplitEntry::new(
                    format!("entry_{i:04}.bin"),
                    data,
                    (i as u64) * 3600, // each 1 hour apart
                    if i % 2 == 0 { "collA" } else { "collB" },
                )
            })
            .collect()
    }

    #[test]
    fn test_strategy_by_size() {
        let dir = temp_dir("strat_size");
        let entries = make_split_entries(5, 100);
        let result = StrategySplitter::split(
            &entries,
            &SplitStrategy::BySize { max_bytes: 500 },
            "sz",
            &dir,
        )
        .expect("split by size failed");

        assert!(result.manifest.total_parts >= 1);
        assert_eq!(result.manifest.total_entries, 5);
        assert_eq!(result.strategy_name, "by_size");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_strategy_by_date() {
        let dir = temp_dir("strat_date");
        let entries = make_split_entries(6, 50);
        // bucket of 2 hours = 7200s. Entries are 0, 3600, 7200, 10800, 14400, 18000
        let result = StrategySplitter::split(
            &entries,
            &SplitStrategy::ByDate {
                bucket_duration_secs: 7200,
            },
            "dt",
            &dir,
        )
        .expect("split by date failed");

        assert_eq!(result.manifest.total_entries, 6);
        // Expect 3 buckets: [0,3600], [7200,10800], [14400,18000]
        assert_eq!(result.manifest.total_parts, 3);
        assert_eq!(result.part_labels.len(), 3);
        assert_eq!(result.strategy_name, "by_date");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_strategy_by_collection() {
        let dir = temp_dir("strat_coll");
        let entries = make_split_entries(6, 50);
        let result = StrategySplitter::split(
            &entries,
            &SplitStrategy::ByCollection {
                max_bytes_per_part: 1_000_000,
            },
            "col",
            &dir,
        )
        .expect("split by collection failed");

        assert_eq!(result.manifest.total_entries, 6);
        // 2 collections: collA (3 entries), collB (3 entries)
        assert!(result.manifest.total_parts >= 2);
        assert_eq!(result.strategy_name, "by_collection");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_strategy_by_count() {
        let dir = temp_dir("strat_count");
        let entries = make_split_entries(7, 50);
        let result = StrategySplitter::split(
            &entries,
            &SplitStrategy::ByCount { max_entries: 3 },
            "cnt",
            &dir,
        )
        .expect("split by count failed");

        assert_eq!(result.manifest.total_entries, 7);
        assert_eq!(result.manifest.total_parts, 3); // ceil(7/3) = 3
        assert_eq!(result.strategy_name, "by_count");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_strategy_by_count_exact_fit() {
        let dir = temp_dir("strat_count_exact");
        let entries = make_split_entries(6, 50);
        let result = StrategySplitter::split(
            &entries,
            &SplitStrategy::ByCount { max_entries: 3 },
            "cnt",
            &dir,
        )
        .expect("split failed");

        assert_eq!(result.manifest.total_parts, 2); // 6/3 = 2
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_strategy_empty_entries() {
        let dir = temp_dir("strat_empty");
        let entries: Vec<SplitEntry> = vec![];

        for strategy in [
            SplitStrategy::BySize { max_bytes: 1000 },
            SplitStrategy::ByDate {
                bucket_duration_secs: 3600,
            },
            SplitStrategy::ByCollection {
                max_bytes_per_part: 1000,
            },
            SplitStrategy::ByCount { max_entries: 10 },
        ] {
            let result =
                StrategySplitter::split(&entries, &strategy, "empty", &dir).expect("split failed");
            assert_eq!(result.manifest.total_parts, 0);
            assert_eq!(result.manifest.total_entries, 0);
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_strategy_by_date_single_bucket() {
        let dir = temp_dir("strat_date_single");
        let entries = make_split_entries(3, 50);
        // Huge bucket: all entries in one
        let result = StrategySplitter::split(
            &entries,
            &SplitStrategy::ByDate {
                bucket_duration_secs: 1_000_000,
            },
            "dt",
            &dir,
        )
        .expect("split failed");

        assert_eq!(result.manifest.total_parts, 1);
        assert_eq!(result.manifest.total_entries, 3);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_split_entry_wire_size() {
        let entry = SplitEntry::new("test.bin", vec![0u8; 100], 0, "default");
        assert_eq!(entry.wire_size(), 2 + 8 + 4 + 100); // path_len(2) + path(8) + data_len(4) + data(100)
    }

    #[test]
    fn test_entry_range_tracking() {
        let dir = temp_dir("ranges");
        let entries = make_entries(6, 100);
        let config = SplitArchiveConfig {
            part_size_bytes: 400,
            base_name: "rng".to_string(),
            compression: false,
        };
        let archiver = SplitArchiver::new(config);
        let manifest = archiver.split(&entries, &dir).expect("split failed");
        // Verify all entry indices are covered without gaps or overlaps.
        let mut covered = vec![false; entries.len()];
        for part in &manifest.parts {
            let (first, last) = part.entry_range;
            for idx in first..=last {
                assert!(!covered[idx], "entry {idx} in multiple parts");
                covered[idx] = true;
            }
        }
        assert!(
            covered.iter().all(|&c| c),
            "some entries not covered by any part"
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
