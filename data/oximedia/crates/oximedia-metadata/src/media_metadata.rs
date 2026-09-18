//! High-level `MediaMetadata` struct, format-specific parsers, and the
//! `MetadataStore` / `InMemoryMetadataStore` abstractions.
//!
//! # Parallel Extraction
//!
//! `ParallelMetadataExtractor` provides multi-format, rayon-parallel metadata
//! probing.  Given a slice of `ProbeTarget` descriptors it spawns one task per
//! target and merges the results.

#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::RwLock;

use rayon::prelude::*;

// ─────────────────────────────────────────────────────────────────────────────
// MediaMetadata
// ─────────────────────────────────────────────────────────────────────────────

/// High-level, format-agnostic media metadata container.
#[derive(Debug, Clone, Default)]
pub struct MediaMetadata {
    /// Human-readable title
    pub title: Option<String>,
    /// Description or comment
    pub description: Option<String>,
    /// Searchable tags
    pub tags: Vec<String>,
    /// Creator / artist / author
    pub creator: Option<String>,
    /// Creation date (ISO 8601)
    pub created_at: Option<String>,
    /// Duration in seconds
    pub duration: Option<f64>,
    /// Additional arbitrary fields
    pub extra: HashMap<String, String>,
}

impl MediaMetadata {
    /// Create an empty `MediaMetadata`
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: set title
    #[must_use]
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Builder: set description
    #[must_use]
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Builder: set creator
    #[must_use]
    pub fn with_creator(mut self, creator: impl Into<String>) -> Self {
        self.creator = Some(creator.into());
        self
    }

    /// Builder: set created_at
    #[must_use]
    pub fn with_created_at(mut self, ts: impl Into<String>) -> Self {
        self.created_at = Some(ts.into());
        self
    }

    /// Builder: set duration in seconds
    #[must_use]
    pub fn with_duration(mut self, secs: f64) -> Self {
        self.duration = Some(secs);
        self
    }

    /// Builder: add a tag
    #[must_use]
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Builder: add an extra key-value field
    #[must_use]
    pub fn with_extra(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra.insert(key.into(), value.into());
        self
    }

    /// Merge another `MediaMetadata` into self (self fields take precedence)
    pub fn merge_from(&mut self, other: &MediaMetadata) {
        if self.title.is_none() {
            self.title = other.title.clone();
        }
        if self.description.is_none() {
            self.description = other.description.clone();
        }
        if self.creator.is_none() {
            self.creator = other.creator.clone();
        }
        if self.created_at.is_none() {
            self.created_at = other.created_at.clone();
        }
        if self.duration.is_none() {
            self.duration = other.duration;
        }
        for tag in &other.tags {
            if !self.tags.contains(tag) {
                self.tags.push(tag.clone());
            }
        }
        for (k, v) in &other.extra {
            self.extra.entry(k.clone()).or_insert_with(|| v.clone());
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// XmpParser
// ─────────────────────────────────────────────────────────────────────────────

/// Basic XMP metadata extractor from RDF/XML.
pub struct XmpParser;

impl XmpParser {
    /// Parse an XMP XML string and return a `MediaMetadata`.
    ///
    /// This is a lightweight, dependency-free text scanner that looks for the
    /// most common Dublin Core and XMP Basic elements.
    #[must_use]
    pub fn parse(xml: &str) -> MediaMetadata {
        let mut meta = MediaMetadata::new();

        // Helper closure: extract text content of the first occurrence of a tag
        let extract = |tag: &str, src: &str| -> Option<String> {
            let open = format!("<{}", tag);
            let close = format!("</{}>", tag);
            if let Some(start) = src.find(&open) {
                // Find the end of the opening tag
                if let Some(gt) = src[start..].find('>') {
                    let content_start = start + gt + 1;
                    if let Some(end) = src[content_start..].find(&close) {
                        let text = src[content_start..content_start + end].trim().to_string();
                        if !text.is_empty() {
                            return Some(text);
                        }
                    }
                }
            }
            None
        };

        // dc:title
        if let Some(v) = extract("dc:title", xml).or_else(|| extract("title", xml)) {
            meta.title = Some(v);
        }

        // dc:description
        if let Some(v) = extract("dc:description", xml).or_else(|| extract("description", xml)) {
            meta.description = Some(v);
        }

        // dc:creator
        if let Some(v) = extract("dc:creator", xml).or_else(|| extract("creator", xml)) {
            meta.creator = Some(v);
        }

        // xmp:CreateDate / dc:date
        if let Some(v) = extract("xmp:CreateDate", xml)
            .or_else(|| extract("dc:date", xml))
            .or_else(|| extract("CreateDate", xml))
        {
            meta.created_at = Some(v);
        }

        // dc:subject (tags)
        let subject_open = "<dc:subject>";
        let subject_close = "</dc:subject>";
        if let Some(start) = xml.find(subject_open) {
            let inner_start = start + subject_open.len();
            if let Some(end) = xml[inner_start..].find(subject_close) {
                let inner = &xml[inner_start..inner_start + end];
                // Tags may be comma-separated or in <rdf:li> elements
                for part in inner.split(',') {
                    let tag = part
                        .trim()
                        .trim_start_matches("<rdf:li>")
                        .trim_end_matches("</rdf:li>")
                        .trim()
                        .to_string();
                    if !tag.is_empty() {
                        meta.tags.push(tag);
                    }
                }
            }
        }

        meta
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ID3Parser
// ─────────────────────────────────────────────────────────────────────────────

/// Basic ID3 tag reader for MP3 files.
pub struct ID3Parser;

impl ID3Parser {
    /// Read ID3 tags from raw MP3 `data` and return a `MediaMetadata`.
    ///
    /// Supports ID3v1 (fixed 128-byte trailer) and ID3v2.3/v2.4 headers.
    #[must_use]
    pub fn read(data: &[u8]) -> MediaMetadata {
        let mut meta = MediaMetadata::new();

        // Try ID3v2 first (header at byte 0: "ID3")
        if data.len() >= 10 && &data[..3] == b"ID3" {
            Self::parse_id3v2(data, &mut meta);
        }

        // Try ID3v1 (last 128 bytes: "TAG")
        if data.len() >= 128 {
            let offset = data.len() - 128;
            if &data[offset..offset + 3] == b"TAG" {
                Self::parse_id3v1(&data[offset..], &mut meta);
            }
        }

        meta
    }

    fn parse_id3v1(tag: &[u8], meta: &mut MediaMetadata) {
        // ID3v1 layout: TAG(3) + title(30) + artist(30) + album(30) + year(4) +
        //               comment(30) + genre(1)
        if tag.len() < 128 {
            return;
        }

        let read_str = |slice: &[u8]| -> Option<String> {
            let s: String = slice
                .iter()
                .take_while(|&&b| b != 0)
                .map(|&b| b as char)
                .collect();
            let s = s.trim().to_string();
            if s.is_empty() {
                None
            } else {
                Some(s)
            }
        };

        if meta.title.is_none() {
            meta.title = read_str(&tag[3..33]);
        }
        if meta.creator.is_none() {
            meta.creator = read_str(&tag[33..63]);
        }
        if meta.created_at.is_none() {
            meta.created_at = read_str(&tag[93..97]);
        }
    }

    fn parse_id3v2(data: &[u8], meta: &mut MediaMetadata) {
        if data.len() < 10 {
            return;
        }
        // version byte at data[3]
        let _version = data[3];
        // Size is a syncsafe integer (4 bytes at offset 6)
        let size = Self::syncsafe_to_u32(&data[6..10]) as usize;
        let end = (10 + size).min(data.len());
        let body = &data[10..end];

        let mut pos = 0;
        while pos + 10 <= body.len() {
            let frame_id = &body[pos..pos + 4];
            if frame_id == [0, 0, 0, 0] {
                break;
            }
            let frame_size =
                u32::from_be_bytes([body[pos + 4], body[pos + 5], body[pos + 6], body[pos + 7]])
                    as usize;
            pos += 10;

            if frame_size == 0 || pos + frame_size > body.len() {
                break;
            }

            let frame_data = &body[pos..pos + frame_size];
            pos += frame_size;

            // Text frames start with an encoding byte
            if frame_data.is_empty() {
                continue;
            }

            let text = Self::decode_text(frame_data);

            match frame_id {
                b"TIT2" => {
                    if meta.title.is_none() {
                        meta.title = Some(text);
                    }
                }
                b"TPE1" => {
                    if meta.creator.is_none() {
                        meta.creator = Some(text);
                    }
                }
                b"TYER" | b"TDRC" => {
                    if meta.created_at.is_none() {
                        meta.created_at = Some(text);
                    }
                }
                b"COMM" => {
                    if meta.description.is_none() && frame_data.len() > 4 {
                        // COMM: encoding(1) + lang(3) + short desc + 0x00 + text
                        let after_lang = &frame_data[4..];
                        if let Some(null_pos) = after_lang.iter().position(|&b| b == 0) {
                            let comment = Self::decode_text(&after_lang[null_pos + 1..]);
                            if !comment.is_empty() {
                                meta.description = Some(comment);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    fn decode_text(data: &[u8]) -> String {
        if data.is_empty() {
            return String::new();
        }
        let enc = data[0];
        let raw = &data[1..];
        match enc {
            0 => {
                // Latin-1
                raw.iter()
                    .take_while(|&&b| b != 0)
                    .map(|&b| b as char)
                    .collect::<String>()
                    .trim()
                    .to_string()
            }
            1 => {
                // UTF-16 with BOM
                if raw.len() < 2 {
                    return String::new();
                }
                let bom = u16::from_be_bytes([raw[0], raw[1]]);
                let (words, swap): (Vec<u16>, bool) = if bom == 0xFFFE {
                    // little-endian
                    (
                        raw[2..]
                            .chunks_exact(2)
                            .map(|c| u16::from_le_bytes([c[0], c[1]]))
                            .collect(),
                        false,
                    )
                } else {
                    // big-endian (0xFEFF or no BOM)
                    (
                        raw[2..]
                            .chunks_exact(2)
                            .map(|c| u16::from_be_bytes([c[0], c[1]]))
                            .collect(),
                        false,
                    )
                };
                let _ = swap;
                let s = String::from_utf16_lossy(
                    &words
                        .into_iter()
                        .take_while(|&w| w != 0)
                        .collect::<Vec<_>>(),
                );
                s.trim().to_string()
            }
            3 => {
                // UTF-8
                std::str::from_utf8(raw)
                    .unwrap_or("")
                    .trim_end_matches('\0')
                    .trim()
                    .to_string()
            }
            _ => String::new(),
        }
    }

    fn syncsafe_to_u32(bytes: &[u8]) -> u32 {
        ((bytes[0] as u32) << 21)
            | ((bytes[1] as u32) << 14)
            | ((bytes[2] as u32) << 7)
            | (bytes[3] as u32)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ExifReader
// ─────────────────────────────────────────────────────────────────────────────

/// EXIF data extractor for JPEG/TIFF images.
pub struct ExifReader;

impl ExifReader {
    /// Extract EXIF key-value pairs from `jpeg_data`.
    ///
    /// Returns an empty map if no EXIF APP1 marker is found.
    #[must_use]
    pub fn extract(jpeg_data: &[u8]) -> HashMap<String, String> {
        let mut result = HashMap::new();

        // JPEG SOI marker: 0xFF 0xD8
        if jpeg_data.len() < 4 || jpeg_data[0] != 0xFF || jpeg_data[1] != 0xD8 {
            return result;
        }

        // Walk JPEG markers to find APP1 (0xFF 0xE1) with Exif header
        let mut pos = 2;
        while pos + 4 <= jpeg_data.len() {
            if jpeg_data[pos] != 0xFF {
                break;
            }
            let marker = jpeg_data[pos + 1];
            let seg_len = u16::from_be_bytes([jpeg_data[pos + 2], jpeg_data[pos + 3]]) as usize;

            if marker == 0xE1 && pos + 2 + seg_len <= jpeg_data.len() {
                let seg = &jpeg_data[pos + 4..pos + 2 + seg_len];
                // APP1 with EXIF header: "Exif\0\0"
                if seg.len() >= 6 && &seg[..6] == b"Exif\0\0" {
                    Self::parse_tiff(&seg[6..], &mut result);
                    break;
                }
            }

            pos += 2 + seg_len;
        }

        result
    }

    fn parse_tiff(data: &[u8], out: &mut HashMap<String, String>) {
        if data.len() < 8 {
            return;
        }

        let little_endian = match &data[..2] {
            b"II" => true,
            b"MM" => false,
            _ => return,
        };

        let read_u16 = |buf: &[u8], off: usize| -> Option<u16> {
            if off + 2 > buf.len() {
                return None;
            }
            Some(if little_endian {
                u16::from_le_bytes([buf[off], buf[off + 1]])
            } else {
                u16::from_be_bytes([buf[off], buf[off + 1]])
            })
        };

        let read_u32 = |buf: &[u8], off: usize| -> Option<u32> {
            if off + 4 > buf.len() {
                return None;
            }
            Some(if little_endian {
                u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
            } else {
                u32::from_be_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
            })
        };

        // TIFF magic
        if read_u16(data, 2) != Some(42) {
            return;
        }

        let ifd_offset = read_u32(data, 4).unwrap_or(0) as usize;
        Self::parse_ifd(data, ifd_offset, &read_u16, &read_u32, out);
    }

    fn parse_ifd(
        data: &[u8],
        offset: usize,
        read_u16: &impl Fn(&[u8], usize) -> Option<u16>,
        read_u32: &impl Fn(&[u8], usize) -> Option<u32>,
        out: &mut HashMap<String, String>,
    ) {
        if offset + 2 > data.len() {
            return;
        }
        let count = read_u16(data, offset).unwrap_or(0) as usize;
        let entry_start = offset + 2;

        for i in 0..count {
            let entry_off = entry_start + i * 12;
            if entry_off + 12 > data.len() {
                break;
            }

            let tag = read_u16(data, entry_off).unwrap_or(0);
            let type_id = read_u16(data, entry_off + 2).unwrap_or(0);
            let n_components = read_u32(data, entry_off + 4).unwrap_or(0) as usize;

            // type 2 = ASCII string
            if type_id == 2 && n_components > 0 {
                let val_off = if n_components <= 4 {
                    entry_off + 8
                } else {
                    read_u32(data, entry_off + 8).unwrap_or(0) as usize
                };

                if val_off + n_components <= data.len() {
                    let raw = &data[val_off..val_off + n_components];
                    let s: String = raw
                        .iter()
                        .take_while(|&&b| b != 0)
                        .map(|&b| b as char)
                        .collect();
                    let s = s.trim().to_string();
                    if !s.is_empty() {
                        let name = Self::tag_name(tag);
                        out.insert(name, s);
                    }
                }
            }
        }
    }

    fn tag_name(tag: u16) -> String {
        match tag {
            0x010F => "Make".to_string(),
            0x0110 => "Model".to_string(),
            0x0132 => "DateTime".to_string(),
            0x013B => "Artist".to_string(),
            0x8298 => "Copyright".to_string(),
            0x9003 => "DateTimeOriginal".to_string(),
            0x9004 => "DateTimeDigitized".to_string(),
            0x9C9B => "XPTitle".to_string(),
            0x9C9C => "XPComment".to_string(),
            0x9C9D => "XPAuthor".to_string(),
            _ => format!("0x{:04X}", tag),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MetadataStore trait
// ─────────────────────────────────────────────────────────────────────────────

/// Error type for metadata store operations
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    /// Key not found
    #[error("Key not found: {0}")]
    NotFound(String),
    /// Internal storage error
    #[error("Store error: {0}")]
    Internal(String),
}

/// Trait for persistent or in-memory metadata stores
pub trait MetadataStore: Send + Sync {
    /// Get metadata by key
    ///
    /// # Errors
    ///
    /// Returns `StoreError::NotFound` if the key does not exist.
    fn get(&self, key: &str) -> Result<MediaMetadata, StoreError>;

    /// Set metadata for a key
    ///
    /// # Errors
    ///
    /// Returns an error if the store cannot persist the value.
    fn set(&self, key: &str, value: MediaMetadata) -> Result<(), StoreError>;

    /// Search for keys whose title, description, tags, or creator contains `query`
    ///
    /// # Errors
    ///
    /// Returns an error if the search fails.
    fn search(&self, query: &str) -> Result<Vec<String>, StoreError>;

    /// Delete metadata by key
    ///
    /// # Errors
    ///
    /// Returns `StoreError::NotFound` if the key does not exist.
    fn delete(&self, key: &str) -> Result<(), StoreError>;

    /// List all keys in the store
    ///
    /// # Errors
    ///
    /// Returns an error if the store cannot be listed.
    fn list_keys(&self) -> Result<Vec<String>, StoreError>;
}

// ─────────────────────────────────────────────────────────────────────────────
// InMemoryMetadataStore
// ─────────────────────────────────────────────────────────────────────────────

/// Thread-safe in-memory implementation of `MetadataStore`
pub struct InMemoryMetadataStore {
    data: RwLock<HashMap<String, MediaMetadata>>,
}

impl InMemoryMetadataStore {
    /// Create a new empty store
    #[must_use]
    pub fn new() -> Self {
        Self {
            data: RwLock::new(HashMap::new()),
        }
    }
}

impl Default for InMemoryMetadataStore {
    fn default() -> Self {
        Self::new()
    }
}

impl MetadataStore for InMemoryMetadataStore {
    fn get(&self, key: &str) -> Result<MediaMetadata, StoreError> {
        let guard = self
            .data
            .read()
            .map_err(|e| StoreError::Internal(e.to_string()))?;
        guard
            .get(key)
            .cloned()
            .ok_or_else(|| StoreError::NotFound(key.to_string()))
    }

    fn set(&self, key: &str, value: MediaMetadata) -> Result<(), StoreError> {
        let mut guard = self
            .data
            .write()
            .map_err(|e| StoreError::Internal(e.to_string()))?;
        guard.insert(key.to_string(), value);
        Ok(())
    }

    fn search(&self, query: &str) -> Result<Vec<String>, StoreError> {
        let query_lower = query.to_lowercase();
        let guard = self
            .data
            .read()
            .map_err(|e| StoreError::Internal(e.to_string()))?;

        let matches: Vec<String> = guard
            .iter()
            .filter_map(|(k, v)| {
                let hit = v
                    .title
                    .as_deref()
                    .unwrap_or("")
                    .to_lowercase()
                    .contains(&query_lower)
                    || v.description
                        .as_deref()
                        .unwrap_or("")
                        .to_lowercase()
                        .contains(&query_lower)
                    || v.creator
                        .as_deref()
                        .unwrap_or("")
                        .to_lowercase()
                        .contains(&query_lower)
                    || v.tags
                        .iter()
                        .any(|t| t.to_lowercase().contains(&query_lower));
                if hit {
                    Some(k.clone())
                } else {
                    None
                }
            })
            .collect();

        Ok(matches)
    }

    fn delete(&self, key: &str) -> Result<(), StoreError> {
        let mut guard = self
            .data
            .write()
            .map_err(|e| StoreError::Internal(e.to_string()))?;
        if guard.remove(key).is_some() {
            Ok(())
        } else {
            Err(StoreError::NotFound(key.to_string()))
        }
    }

    fn list_keys(&self) -> Result<Vec<String>, StoreError> {
        let guard = self
            .data
            .read()
            .map_err(|e| StoreError::Internal(e.to_string()))?;
        Ok(guard.keys().cloned().collect())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Parallel metadata extraction
// ─────────────────────────────────────────────────────────────────────────────

/// The format a `ProbeTarget` should be parsed as.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeFormat {
    /// Parse as ID3v2 (MP3)
    Id3v2,
    /// Parse as XMP (XML)
    Xmp,
    /// Extract EXIF from JPEG/TIFF bytes
    Exif,
    /// Try all known formats and return the first successful parse
    Auto,
}

/// A single probe target: a label plus the raw bytes to inspect.
#[derive(Debug, Clone)]
pub struct ProbeTarget {
    /// Human-readable label for this data source (e.g. a file path).
    pub label: String,
    /// The raw bytes of the media or sidecar to probe.
    pub data: Vec<u8>,
    /// Which format to attempt.
    pub format: ProbeFormat,
}

impl ProbeTarget {
    /// Create a new `ProbeTarget`.
    #[must_use]
    pub fn new(label: impl Into<String>, data: impl Into<Vec<u8>>, format: ProbeFormat) -> Self {
        Self {
            label: label.into(),
            data: data.into(),
            format,
        }
    }
}

/// The result of probing a single `ProbeTarget`.
#[derive(Debug, Clone)]
pub struct ProbeResult {
    /// The label passed in with the `ProbeTarget`.
    pub label: String,
    /// Extracted metadata, or an error message on failure.
    pub outcome: Result<MediaMetadata, String>,
}

impl ProbeResult {
    /// Return `true` if extraction succeeded.
    #[must_use]
    pub fn is_ok(&self) -> bool {
        self.outcome.is_ok()
    }

    /// Return a reference to the extracted `MediaMetadata`, if successful.
    #[must_use]
    pub fn metadata(&self) -> Option<&MediaMetadata> {
        self.outcome.as_ref().ok()
    }
}

/// Parallel metadata extractor.
///
/// Probes multiple sources concurrently using rayon.
///
/// # Example
///
/// ```rust
/// use oximedia_metadata::media_metadata::{ParallelMetadataExtractor, ProbeFormat, ProbeTarget};
///
/// let targets = vec![
///     ProbeTarget::new("sample1", b"ID3\x04\x00\x00\x00\x00\x00\x00".to_vec(), ProbeFormat::Id3v2),
/// ];
/// let results = ParallelMetadataExtractor::extract_all(&targets);
/// assert_eq!(results.len(), 1);
/// ```
pub struct ParallelMetadataExtractor;

impl ParallelMetadataExtractor {
    /// Extract metadata from all `targets` in parallel (rayon).
    ///
    /// Results are returned in the same order as `targets`.
    #[must_use]
    pub fn extract_all(targets: &[ProbeTarget]) -> Vec<ProbeResult> {
        targets
            .par_iter()
            .map(|target| {
                let outcome = Self::probe_one(target);
                ProbeResult {
                    label: target.label.clone(),
                    outcome,
                }
            })
            .collect()
    }

    /// Probe a single target and return `MediaMetadata` or an error string.
    fn probe_one(target: &ProbeTarget) -> Result<MediaMetadata, String> {
        match target.format {
            ProbeFormat::Id3v2 => {
                let meta = ID3Parser::read(&target.data);
                Ok(meta)
            }
            ProbeFormat::Xmp => {
                let text = std::str::from_utf8(&target.data)
                    .map_err(|e| format!("XMP data is not valid UTF-8: {e}"))?;
                Ok(XmpParser::parse(text))
            }
            ProbeFormat::Exif => {
                let exif_map = ExifReader::extract(&target.data);
                let mut meta = MediaMetadata::new();
                if let Some(title) = exif_map.get("ImageDescription") {
                    meta.title = Some(title.clone());
                }
                if let Some(artist) = exif_map.get("Artist") {
                    meta.creator = Some(artist.clone());
                }
                if let Some(dt) = exif_map
                    .get("DateTimeOriginal")
                    .or_else(|| exif_map.get("DateTime"))
                {
                    meta.created_at = Some(dt.clone());
                }
                for (k, v) in &exif_map {
                    meta.extra.insert(k.clone(), v.clone());
                }
                Ok(meta)
            }
            ProbeFormat::Auto => {
                // Try ID3v2, then XMP, then EXIF; return first success
                if target.data.len() >= 3 && &target.data[..3] == b"ID3" {
                    return Ok(ID3Parser::read(&target.data));
                }
                if target.data.len() >= 5
                    && (target.data.starts_with(b"<?xml")
                        || target.data.starts_with(b"<x:xm")
                        || target.data.starts_with(b"<rdf:"))
                {
                    if let Ok(text) = std::str::from_utf8(&target.data) {
                        return Ok(XmpParser::parse(text));
                    }
                }
                // Fallback: try EXIF from JPEG
                let exif_map = ExifReader::extract(&target.data);
                let mut meta = MediaMetadata::new();
                for (k, v) in exif_map {
                    meta.extra.insert(k, v);
                }
                Ok(meta)
            }
        }
    }

    /// Extract and merge all results into a single `MediaMetadata`.
    ///
    /// Results are merged in order, with earlier targets taking precedence
    /// for scalar fields (title, creator, etc.).
    #[must_use]
    pub fn extract_and_merge(targets: &[ProbeTarget]) -> MediaMetadata {
        let results = Self::extract_all(targets);
        let mut merged = MediaMetadata::new();
        for result in results {
            if let Ok(meta) = result.outcome {
                merged.merge_from(&meta);
            }
        }
        merged
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // --- MediaMetadata ---

    #[test]
    fn test_media_metadata_builder() {
        let m = MediaMetadata::new()
            .with_title("Test Video")
            .with_creator("Alice")
            .with_duration(120.5)
            .with_tag("sports")
            .with_tag("outdoor")
            .with_extra("resolution", "1920x1080");

        assert_eq!(m.title.as_deref(), Some("Test Video"));
        assert_eq!(m.creator.as_deref(), Some("Alice"));
        assert!((m.duration.expect("should succeed in test") - 120.5).abs() < f64::EPSILON);
        assert_eq!(m.tags.len(), 2);
        assert_eq!(
            m.extra.get("resolution").map(String::as_str),
            Some("1920x1080")
        );
    }

    #[test]
    fn test_media_metadata_default() {
        let m = MediaMetadata::default();
        assert!(m.title.is_none());
        assert!(m.tags.is_empty());
        assert!(m.extra.is_empty());
    }

    #[test]
    fn test_media_metadata_merge() {
        let mut base = MediaMetadata::new().with_title("Base");
        let other = MediaMetadata::new()
            .with_title("Other")
            .with_creator("Bob")
            .with_tag("music");

        base.merge_from(&other);

        // base title takes precedence
        assert_eq!(base.title.as_deref(), Some("Base"));
        // creator comes from other
        assert_eq!(base.creator.as_deref(), Some("Bob"));
        // tags merged
        assert!(base.tags.contains(&"music".to_string()));
    }

    #[test]
    fn test_media_metadata_merge_no_duplicate_tags() {
        let mut base = MediaMetadata::new().with_tag("rock");
        let other = MediaMetadata::new().with_tag("rock").with_tag("jazz");
        base.merge_from(&other);
        assert_eq!(base.tags.len(), 2);
    }

    // --- XmpParser ---

    #[test]
    fn test_xmp_parser_basic() {
        let xml = r#"<?xpacket?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF>
    <rdf:Description xmlns:dc="http://purl.org/dc/elements/1.1/">
      <dc:title>My Great Film</dc:title>
      <dc:creator>John Doe</dc:creator>
      <dc:description>A short film about adventure</dc:description>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>"#;

        let meta = XmpParser::parse(xml);
        assert_eq!(meta.title.as_deref(), Some("My Great Film"));
        assert_eq!(meta.creator.as_deref(), Some("John Doe"));
        assert!(meta.description.is_some());
    }

    #[test]
    fn test_xmp_parser_create_date() {
        let xml = r#"<rdf:Description>
  <xmp:CreateDate>2024-03-15T10:30:00Z</xmp:CreateDate>
</rdf:Description>"#;

        let meta = XmpParser::parse(xml);
        assert_eq!(meta.created_at.as_deref(), Some("2024-03-15T10:30:00Z"));
    }

    #[test]
    fn test_xmp_parser_empty() {
        let meta = XmpParser::parse("<empty/>");
        assert!(meta.title.is_none());
        assert!(meta.creator.is_none());
    }

    #[test]
    fn test_xmp_parser_subjects() {
        let xml = r#"<rdf:Description>
  <dc:title>Tagged</dc:title>
  <dc:subject>nature, landscape, travel</dc:subject>
</rdf:Description>"#;

        let meta = XmpParser::parse(xml);
        assert!(!meta.tags.is_empty());
    }

    // --- ID3Parser ---

    #[test]
    fn test_id3_parser_empty_returns_empty() {
        let meta = ID3Parser::read(b"not an mp3 file");
        assert!(meta.title.is_none());
    }

    #[test]
    fn test_id3v1_parsing() {
        // Build a minimal ID3v1 tag (128 bytes)
        let mut tag = vec![0u8; 128];
        tag[0] = b'T';
        tag[1] = b'A';
        tag[2] = b'G';
        // Title at bytes 3..33
        let title = b"TestSong";
        tag[3..3 + title.len()].copy_from_slice(title);
        // Artist at bytes 33..63
        let artist = b"TestArtist";
        tag[33..33 + artist.len()].copy_from_slice(artist);

        let meta = ID3Parser::read(&tag);
        assert_eq!(meta.title.as_deref(), Some("TestSong"));
        assert_eq!(meta.creator.as_deref(), Some("TestArtist"));
    }

    #[test]
    fn test_id3v2_minimal_header() {
        // Minimal ID3v2 frame: just the header, no frames
        let mut data = vec![0u8; 10];
        data[0] = b'I';
        data[1] = b'D';
        data[2] = b'3';
        data[3] = 3; // version 2.3
        data[4] = 0; // revision
        data[5] = 0; // flags
                     // syncsafe size = 0 (no frames)
        let meta = ID3Parser::read(&data);
        // No frames, but no crash
        assert!(meta.title.is_none());
    }

    // --- ExifReader ---

    #[test]
    fn test_exif_reader_non_jpeg_returns_empty() {
        let result = ExifReader::extract(b"not a jpeg");
        assert!(result.is_empty());
    }

    #[test]
    fn test_exif_reader_jpeg_without_exif() {
        // JPEG SOI marker only
        let data = vec![0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10];
        let result = ExifReader::extract(&data);
        assert!(result.is_empty());
    }

    #[test]
    fn test_exif_reader_tag_name_known() {
        assert_eq!(ExifReader::tag_name(0x010F), "Make");
        assert_eq!(ExifReader::tag_name(0x0110), "Model");
        assert_eq!(ExifReader::tag_name(0x0132), "DateTime");
    }

    #[test]
    fn test_exif_reader_tag_name_unknown() {
        let name = ExifReader::tag_name(0xABCD);
        assert!(name.starts_with("0x"));
    }

    // --- InMemoryMetadataStore ---

    #[test]
    fn test_store_set_get() {
        let store = InMemoryMetadataStore::new();
        let meta = MediaMetadata::new().with_title("Clip A");
        store.set("clip_a", meta).expect("should succeed in test");

        let retrieved = store.get("clip_a").expect("should succeed in test");
        assert_eq!(retrieved.title.as_deref(), Some("Clip A"));
    }

    #[test]
    fn test_store_not_found() {
        let store = InMemoryMetadataStore::new();
        let result = store.get("nonexistent");
        assert!(matches!(result, Err(StoreError::NotFound(_))));
    }

    #[test]
    fn test_store_search() {
        let store = InMemoryMetadataStore::new();
        store
            .set("a", MediaMetadata::new().with_title("Adventure Film"))
            .expect("should succeed in test");
        store
            .set("b", MediaMetadata::new().with_title("Documentary"))
            .expect("should succeed in test");
        store
            .set("c", MediaMetadata::new().with_creator("Adventure Bob"))
            .expect("should succeed in test");

        let results = store.search("adventure").expect("should succeed in test");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_store_delete() {
        let store = InMemoryMetadataStore::new();
        store
            .set("del_key", MediaMetadata::new())
            .expect("should succeed in test");
        store.delete("del_key").expect("should succeed in test");

        assert!(matches!(store.get("del_key"), Err(StoreError::NotFound(_))));
    }

    #[test]
    fn test_store_list_keys() {
        let store = InMemoryMetadataStore::new();
        store
            .set("k1", MediaMetadata::new())
            .expect("should succeed in test");
        store
            .set("k2", MediaMetadata::new())
            .expect("should succeed in test");
        store
            .set("k3", MediaMetadata::new())
            .expect("should succeed in test");

        let keys = store.list_keys().expect("should succeed in test");
        assert_eq!(keys.len(), 3);
    }

    #[test]
    fn test_store_delete_nonexistent() {
        let store = InMemoryMetadataStore::new();
        let result = store.delete("missing");
        assert!(matches!(result, Err(StoreError::NotFound(_))));
    }

    #[test]
    fn test_store_overwrite() {
        let store = InMemoryMetadataStore::new();
        store
            .set("key", MediaMetadata::new().with_title("v1"))
            .expect("should succeed in test");
        store
            .set("key", MediaMetadata::new().with_title("v2"))
            .expect("should succeed in test");
        let m = store.get("key").expect("should succeed in test");
        assert_eq!(m.title.as_deref(), Some("v2"));
    }

    #[test]
    fn test_store_search_by_tag() {
        let store = InMemoryMetadataStore::new();
        store
            .set(
                "x",
                MediaMetadata::new()
                    .with_tag("orchestral")
                    .with_tag("classical"),
            )
            .expect("should succeed in test");
        store
            .set("y", MediaMetadata::new().with_tag("jazz"))
            .expect("should succeed in test");

        let results = store.search("orchestral").expect("should succeed in test");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], "x");
    }

    // ─── ParallelMetadataExtractor ─────────────────────────────────────────

    fn make_minimal_id3v2() -> Vec<u8> {
        // Minimal valid ID3v2.4 tag: header only, no frames
        let mut data = Vec::new();
        data.extend_from_slice(b"ID3");
        data.push(4); // v2.4
        data.push(0);
        data.push(0);
        // synchsafe size = 0
        data.extend_from_slice(&[0u8; 4]);
        data
    }

    #[test]
    fn test_parallel_extractor_empty_targets() {
        let results = ParallelMetadataExtractor::extract_all(&[]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_parallel_extractor_id3v2_target() {
        let targets = vec![ProbeTarget::new(
            "test.mp3",
            make_minimal_id3v2(),
            ProbeFormat::Id3v2,
        )];
        let results = ParallelMetadataExtractor::extract_all(&targets);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].label, "test.mp3");
        assert!(results[0].is_ok());
    }

    #[test]
    fn test_parallel_extractor_xmp_target() {
        let xml = r#"<rdf:Description xmlns:dc="http://purl.org/dc/elements/1.1/">
  <dc:title>XMP Title</dc:title>
  <dc:creator>XMP Author</dc:creator>
</rdf:Description>"#;
        let targets = vec![ProbeTarget::new(
            "sample.xmp",
            xml.as_bytes(),
            ProbeFormat::Xmp,
        )];
        let results = ParallelMetadataExtractor::extract_all(&targets);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_ok());
        let meta = results[0].metadata().expect("Should have metadata");
        assert_eq!(meta.title.as_deref(), Some("XMP Title"));
    }

    #[test]
    fn test_parallel_extractor_exif_target_non_jpeg() {
        // Non-JPEG data: EXIF reader returns empty map
        let targets = vec![ProbeTarget::new(
            "not_a_jpeg.bin",
            b"not jpeg data",
            ProbeFormat::Exif,
        )];
        let results = ParallelMetadataExtractor::extract_all(&targets);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_ok());
    }

    #[test]
    fn test_parallel_extractor_auto_id3v2() {
        let targets = vec![ProbeTarget::new(
            "auto.mp3",
            make_minimal_id3v2(),
            ProbeFormat::Auto,
        )];
        let results = ParallelMetadataExtractor::extract_all(&targets);
        assert!(results[0].is_ok());
    }

    #[test]
    fn test_parallel_extractor_auto_xmp() {
        let xml = b"<?xml version=\"1.0\"?><root><dc:title>Auto XMP</dc:title></root>";
        let targets = vec![ProbeTarget::new(
            "auto.xmp",
            xml.to_vec(),
            ProbeFormat::Auto,
        )];
        let results = ParallelMetadataExtractor::extract_all(&targets);
        assert!(results[0].is_ok());
    }

    #[test]
    fn test_parallel_extractor_multiple_targets_in_order() {
        let targets = vec![
            ProbeTarget::new("first", make_minimal_id3v2(), ProbeFormat::Id3v2),
            ProbeTarget::new("second", make_minimal_id3v2(), ProbeFormat::Id3v2),
            ProbeTarget::new("third", make_minimal_id3v2(), ProbeFormat::Id3v2),
        ];
        let results = ParallelMetadataExtractor::extract_all(&targets);
        // Results must be in the same order as targets
        assert_eq!(results[0].label, "first");
        assert_eq!(results[1].label, "second");
        assert_eq!(results[2].label, "third");
    }

    #[test]
    fn test_parallel_extractor_probe_result_is_ok() {
        let targets = vec![ProbeTarget::new(
            "ok",
            make_minimal_id3v2(),
            ProbeFormat::Id3v2,
        )];
        let results = ParallelMetadataExtractor::extract_all(&targets);
        assert!(results[0].is_ok());
        assert!(results[0].metadata().is_some());
    }

    #[test]
    fn test_parallel_extractor_xmp_invalid_utf8() {
        // Non-UTF-8 bytes for XMP should return an error
        let bad = vec![0xFF, 0xFE, 0x00, 0x01];
        let targets = vec![ProbeTarget::new("bad.xmp", bad, ProbeFormat::Xmp)];
        let results = ParallelMetadataExtractor::extract_all(&targets);
        assert_eq!(results.len(), 1);
        // Invalid UTF-8 should be an error
        assert!(!results[0].is_ok());
    }

    #[test]
    fn test_parallel_extractor_extract_and_merge() {
        let xmp1 = r#"<rdf:Description xmlns:dc="http://purl.org/dc/elements/1.1/">
  <dc:title>Merged Title</dc:title>
</rdf:Description>"#;
        let xmp2 = r#"<rdf:Description xmlns:dc="http://purl.org/dc/elements/1.1/">
  <dc:creator>Merged Author</dc:creator>
</rdf:Description>"#;

        let targets = vec![
            ProbeTarget::new("src1.xmp", xmp1.as_bytes(), ProbeFormat::Xmp),
            ProbeTarget::new("src2.xmp", xmp2.as_bytes(), ProbeFormat::Xmp),
        ];

        let merged = ParallelMetadataExtractor::extract_and_merge(&targets);
        // Title from src1, creator from src2
        assert_eq!(merged.title.as_deref(), Some("Merged Title"));
        assert_eq!(merged.creator.as_deref(), Some("Merged Author"));
    }

    #[test]
    fn test_probe_target_new() {
        let t = ProbeTarget::new("label", vec![1u8, 2, 3], ProbeFormat::Auto);
        assert_eq!(t.label, "label");
        assert_eq!(t.data, vec![1u8, 2, 3]);
        assert_eq!(t.format, ProbeFormat::Auto);
    }
}
