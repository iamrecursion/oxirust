// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! CMAF byte-range addressing for single-file segment storage.
//!
//! Instead of writing one file per segment, byte-range addressing packs all
//! CMAF chunks of a single track into one contiguous container file.  Clients
//! retrieve each segment by issuing an HTTP `Range: bytes=<start>-<end>`
//! request, which dramatically reduces the number of files on a CDN origin
//! while remaining fully compatible with HLS and DASH manifests.
//!
//! # Layout
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────┐
//! │  init segment  (ftyp + moov)                                │
//! ├──────────────────────────────────────────────────────────────┤
//! │  segment_0  (styp + moof + mdat)   offset=init_len          │
//! ├──────────────────────────────────────────────────────────────┤
//! │  segment_1  (styp + moof + mdat)   offset=init_len+seg0_len │
//! ├──────────────────────────────────────────────────────────────┤
//! │  …                                                           │
//! └──────────────────────────────────────────────────────────────┘
//! ```
//!
//! # References
//!
//! - ISO/IEC 23000-19 (CMAF), section 7 — byte-range CMAF addressing
//! - HLS RFC 8216, section 4.3.2.2 — `EXT-X-BYTERANGE`
//! - DASH-IF IOP v4.3, section 4.5 — `SegmentBase` byte ranges

use crate::error::{PackagerError, PackagerResult};
use crate::isobmff_writer::{BoxWriter, MediaSample};
use std::time::Duration;

// ---------------------------------------------------------------------------
// SegmentByteRange
// ---------------------------------------------------------------------------

/// The byte range of a single CMAF segment within a container file.
///
/// This is the core addressing primitive: every segment is identified by its
/// `offset` (distance from the beginning of the container file) and `length`
/// (total size including `styp`, `moof`, and `mdat` boxes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentByteRange {
    /// Byte offset of this segment's first byte in the container file.
    pub offset: u64,
    /// Total byte length of this segment.
    pub length: u64,
}

impl SegmentByteRange {
    /// Create a new segment byte range.
    #[must_use]
    pub fn new(offset: u64, length: u64) -> Self {
        Self { offset, length }
    }

    /// Exclusive end byte (`offset + length`).
    #[must_use]
    pub fn end_byte(&self) -> u64 {
        self.offset + self.length
    }

    /// Format as an HLS `EXT-X-BYTERANGE` attribute value: `<length>@<offset>`.
    #[must_use]
    pub fn hls_byterange(&self) -> String {
        format!("{}@{}", self.length, self.offset)
    }

    /// Format as a DASH `@mediaRange` attribute: `<start>-<end_inclusive>`.
    #[must_use]
    pub fn dash_media_range(&self) -> String {
        format!("{}-{}", self.offset, self.end_byte().saturating_sub(1))
    }
}

// ---------------------------------------------------------------------------
// CmafByteRangeEntry
// ---------------------------------------------------------------------------

/// An indexed entry in the byte-range index, combining the raw range with
/// presentation metadata.
#[derive(Debug, Clone)]
pub struct CmafByteRangeEntry {
    /// Zero-based segment index.
    pub index: u32,
    /// Byte range within the container file.
    pub range: SegmentByteRange,
    /// Segment presentation duration.
    pub duration: Duration,
    /// Decode timestamp in timescale ticks.
    pub decode_time: u64,
    /// Whether this segment starts with a keyframe.
    pub starts_with_keyframe: bool,
}

impl CmafByteRangeEntry {
    /// Create a new entry.
    #[must_use]
    pub fn new(
        index: u32,
        offset: u64,
        length: u64,
        duration: Duration,
        decode_time: u64,
        starts_with_keyframe: bool,
    ) -> Self {
        Self {
            index,
            range: SegmentByteRange::new(offset, length),
            duration,
            decode_time,
            starts_with_keyframe,
        }
    }

    /// HLS byte-range attribute string.
    #[must_use]
    pub fn hls_byterange(&self) -> String {
        self.range.hls_byterange()
    }

    /// DASH media range attribute string.
    #[must_use]
    pub fn dash_media_range(&self) -> String {
        self.range.dash_media_range()
    }
}

// ---------------------------------------------------------------------------
// CmafByteRangeIndex
// ---------------------------------------------------------------------------

/// A complete byte-range index for a single-file CMAF container.
///
/// Tracks the init segment, all media segment ranges, and provides helpers
/// for generating HLS and DASH manifest attributes.
#[derive(Debug, Clone)]
pub struct CmafByteRangeIndex {
    /// Byte length of the init segment (ftyp + moov).
    init_length: u64,
    /// Media segment entries in segment order.
    entries: Vec<CmafByteRangeEntry>,
    /// Running write cursor (bytes appended so far).
    cursor: u64,
    /// Container file URI (used in manifest generation).
    container_uri: String,
    /// Timescale (ticks per second) for the track.
    timescale: u32,
}

impl CmafByteRangeIndex {
    /// Create a new byte-range index.
    ///
    /// # Arguments
    ///
    /// * `init_length` - Size of the init segment in bytes.
    /// * `container_uri` - URI of the single container file (used in manifests).
    /// * `timescale` - Media timescale (ticks per second).
    #[must_use]
    pub fn new(init_length: u64, container_uri: impl Into<String>, timescale: u32) -> Self {
        Self {
            init_length,
            entries: Vec::new(),
            cursor: init_length,
            container_uri: container_uri.into(),
            timescale,
        }
    }

    /// Append a media segment and record its byte range.
    ///
    /// Returns the recorded entry.
    pub fn append_segment(
        &mut self,
        segment_length: u64,
        duration: Duration,
        decode_time: u64,
        starts_with_keyframe: bool,
    ) -> CmafByteRangeEntry {
        let index = self.entries.len() as u32;
        let entry = CmafByteRangeEntry::new(
            index,
            self.cursor,
            segment_length,
            duration,
            decode_time,
            starts_with_keyframe,
        );
        self.cursor += segment_length;
        self.entries.push(entry.clone());
        entry
    }

    /// Return the init segment length.
    #[must_use]
    pub fn init_length(&self) -> u64 {
        self.init_length
    }

    /// Return the container URI.
    #[must_use]
    pub fn container_uri(&self) -> &str {
        &self.container_uri
    }

    /// Return all media segment entries.
    #[must_use]
    pub fn entries(&self) -> &[CmafByteRangeEntry] {
        &self.entries
    }

    /// Return the total file size (init + all segments).
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.cursor
    }

    /// Return the number of media segments.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.entries.len()
    }

    /// Look up an entry by segment index.
    #[must_use]
    pub fn get(&self, index: u32) -> Option<&CmafByteRangeEntry> {
        self.entries.get(index as usize)
    }

    /// Total presentation duration of all segments.
    #[must_use]
    pub fn total_duration(&self) -> Duration {
        self.entries.iter().map(|e| e.duration).sum()
    }

    /// Return only the entries that start with a keyframe (for I-frame playlists).
    #[must_use]
    pub fn keyframe_entries(&self) -> Vec<&CmafByteRangeEntry> {
        self.entries
            .iter()
            .filter(|e| e.starts_with_keyframe)
            .collect()
    }

    /// Validate the index: verify all entries are contiguous.
    ///
    /// # Errors
    ///
    /// Returns [`PackagerError::PackagingError`] if any invariant is violated.
    pub fn validate(&self) -> PackagerResult<()> {
        let mut expected_offset = self.init_length;
        for (i, entry) in self.entries.iter().enumerate() {
            if entry.range.offset != expected_offset {
                return Err(PackagerError::PackagingError(format!(
                    "Segment {i}: expected offset {expected_offset}, found {}",
                    entry.range.offset
                )));
            }
            if entry.range.length == 0 {
                return Err(PackagerError::PackagingError(format!(
                    "Segment {i}: length must not be zero"
                )));
            }
            expected_offset += entry.range.length;
        }
        if expected_offset != self.cursor {
            return Err(PackagerError::PackagingError(format!(
                "Cursor mismatch: expected {expected_offset}, found {}",
                self.cursor
            )));
        }
        Ok(())
    }

    /// Generate HLS `#EXT-X-BYTERANGE` segment lines for a media playlist.
    ///
    /// Each entry produces:
    /// ```text
    /// #EXTINF:<duration>,
    /// #EXT-X-BYTERANGE:<length>@<offset>
    /// <container_uri>
    /// ```
    #[must_use]
    pub fn to_hls_segments(&self) -> String {
        let mut out = String::new();
        for entry in &self.entries {
            let secs = entry.duration.as_secs_f64();
            out.push_str(&format!("#EXTINF:{secs:.6},\n"));
            out.push_str(&format!("#EXT-X-BYTERANGE:{}\n", entry.hls_byterange()));
            out.push_str(&self.container_uri);
            out.push('\n');
        }
        out
    }

    /// Generate HLS `EXT-X-MAP` tag for the init segment.
    #[must_use]
    pub fn to_hls_map_tag(&self) -> String {
        format!(
            "#EXT-X-MAP:URI=\"{}\",BYTERANGE=\"{}@0\"",
            self.container_uri, self.init_length
        )
    }

    /// Generate a DASH `SegmentBase` XML fragment.
    #[must_use]
    pub fn to_dash_segment_base(&self) -> String {
        let init_end = self.init_length.saturating_sub(1);
        format!(
            r#"<SegmentBase indexRange="0-{init_end}" timescale="{}"><Initialization range="0-{init_end}"/></SegmentBase>"#,
            self.timescale
        )
    }

    /// Generate DASH `SegmentList` XML fragment with byte ranges.
    #[must_use]
    pub fn to_dash_segment_list(&self) -> String {
        let init_end = self.init_length.saturating_sub(1);
        let mut out = format!(
            r#"<SegmentList timescale="{}"><Initialization range="0-{init_end}"/>"#,
            self.timescale
        );
        for entry in &self.entries {
            let dur_ticks =
                (entry.duration.as_secs_f64() * f64::from(self.timescale)).round() as u64;
            out.push_str(&format!(
                r#"<SegmentURL mediaRange="{}" duration="{}"/>"#,
                entry.dash_media_range(),
                dur_ticks
            ));
        }
        out.push_str("</SegmentList>");
        out
    }
}

// ---------------------------------------------------------------------------
// CmafByteRangeWriter
// ---------------------------------------------------------------------------

/// An in-memory writer for building a single-file CMAF container.
///
/// Appends init + media segments contiguously and maintains a
/// [`CmafByteRangeIndex`] automatically.
///
/// For large streams, write directly to a file and maintain the index
/// separately.
#[derive(Debug)]
pub struct CmafByteRangeWriter {
    data: Vec<u8>,
    index: Option<CmafByteRangeIndex>,
}

impl CmafByteRangeWriter {
    /// Create a new writer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            index: None,
        }
    }

    /// Write the init segment bytes and start the index.
    ///
    /// Must be called exactly once, before any `append_segment` calls.
    ///
    /// # Errors
    ///
    /// Returns an error if the init segment has already been written.
    pub fn write_init(
        &mut self,
        init_bytes: &[u8],
        container_uri: &str,
        timescale: u32,
    ) -> PackagerResult<()> {
        if self.index.is_some() {
            return Err(PackagerError::PackagingError(
                "init segment already written".into(),
            ));
        }
        self.data.extend_from_slice(init_bytes);
        self.index = Some(CmafByteRangeIndex::new(
            init_bytes.len() as u64,
            container_uri,
            timescale,
        ));
        Ok(())
    }

    /// Append a media segment and record its byte range.
    ///
    /// # Errors
    ///
    /// Returns an error if `write_init` has not been called yet.
    pub fn append_segment(
        &mut self,
        segment_bytes: &[u8],
        duration: Duration,
        decode_time: u64,
        starts_with_keyframe: bool,
    ) -> PackagerResult<CmafByteRangeEntry> {
        let idx = self.index.as_mut().ok_or_else(|| {
            PackagerError::PackagingError("init segment must be written first".into())
        })?;
        let entry = idx.append_segment(
            segment_bytes.len() as u64,
            duration,
            decode_time,
            starts_with_keyframe,
        );
        self.data.extend_from_slice(segment_bytes);
        Ok(entry)
    }

    /// Append a media segment built from [`MediaSample`] objects.
    ///
    /// Builds the moof+mdat automatically using the isobmff writer.
    ///
    /// # Errors
    ///
    /// Returns an error if init has not been written.
    pub fn append_media_samples(
        &mut self,
        samples: &[MediaSample],
        sequence_number: u32,
        base_media_decode_time: u64,
        duration: Duration,
    ) -> PackagerResult<CmafByteRangeEntry> {
        let segment_bytes = crate::isobmff_writer::write_media_segment(
            sequence_number,
            base_media_decode_time,
            samples,
        );

        let starts_with_keyframe = samples.first().is_some_and(|s| s.is_sync);

        self.append_segment(
            &segment_bytes,
            duration,
            base_media_decode_time,
            starts_with_keyframe,
        )
    }

    /// Return the accumulated bytes.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Return the byte-range index, if initialised.
    #[must_use]
    pub fn index(&self) -> Option<&CmafByteRangeIndex> {
        self.index.as_ref()
    }

    /// Consume the writer and return `(data, index)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the init segment was never written.
    pub fn finish(self) -> PackagerResult<(Vec<u8>, CmafByteRangeIndex)> {
        let index = self.index.ok_or_else(|| {
            PackagerError::PackagingError("cannot finish: init segment was never written".into())
        })?;
        index.validate()?;
        Ok((self.data, index))
    }
}

impl Default for CmafByteRangeWriter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// write_styp — segment type box
// ---------------------------------------------------------------------------

/// Write a `styp` (Segment Type) box for CMAF media segments.
///
/// This is analogous to `ftyp` but used at the start of each media segment
/// in single-file CMAF containers.
#[must_use]
pub fn write_styp() -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();
    BoxWriter::write_box(&mut out, b"styp", |w| {
        w.write_fourcc(b"msdh"); // major brand
        w.write_u32(0); // minor version
        w.write_fourcc(b"msdh"); // compatible brands
        w.write_fourcc(b"msix");
        w.write_fourcc(b"cmfc");
    });
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn dur(s: u64) -> Duration {
        Duration::from_secs(s)
    }

    // --- SegmentByteRange ---------------------------------------------------

    #[test]
    fn test_segment_byte_range_new() {
        let r = SegmentByteRange::new(100, 200);
        assert_eq!(r.offset, 100);
        assert_eq!(r.length, 200);
    }

    #[test]
    fn test_segment_byte_range_end_byte() {
        let r = SegmentByteRange::new(100, 200);
        assert_eq!(r.end_byte(), 300);
    }

    #[test]
    fn test_segment_byte_range_hls() {
        let r = SegmentByteRange::new(256, 1024);
        assert_eq!(r.hls_byterange(), "1024@256");
    }

    #[test]
    fn test_segment_byte_range_dash() {
        let r = SegmentByteRange::new(0, 1024);
        assert_eq!(r.dash_media_range(), "0-1023");
    }

    #[test]
    fn test_segment_byte_range_dash_nonzero() {
        let r = SegmentByteRange::new(512, 256);
        assert_eq!(r.dash_media_range(), "512-767");
    }

    // --- CmafByteRangeEntry -------------------------------------------------

    #[test]
    fn test_entry_new() {
        let e = CmafByteRangeEntry::new(0, 100, 500, dur(6), 0, true);
        assert_eq!(e.index, 0);
        assert_eq!(e.range.offset, 100);
        assert_eq!(e.range.length, 500);
        assert!(e.starts_with_keyframe);
    }

    #[test]
    fn test_entry_hls_byterange() {
        let e = CmafByteRangeEntry::new(1, 600, 400, dur(6), 540_000, false);
        assert_eq!(e.hls_byterange(), "400@600");
    }

    #[test]
    fn test_entry_dash_media_range() {
        let e = CmafByteRangeEntry::new(0, 100, 500, dur(6), 0, true);
        assert_eq!(e.dash_media_range(), "100-599");
    }

    // --- CmafByteRangeIndex -------------------------------------------------

    #[test]
    fn test_index_new() {
        let idx = CmafByteRangeIndex::new(256, "media.mp4", 90_000);
        assert_eq!(idx.init_length(), 256);
        assert_eq!(idx.container_uri(), "media.mp4");
        assert_eq!(idx.total_bytes(), 256);
        assert_eq!(idx.segment_count(), 0);
    }

    #[test]
    fn test_index_append_segment() {
        let mut idx = CmafByteRangeIndex::new(100, "v.mp4", 90_000);
        let e0 = idx.append_segment(500, dur(6), 0, true);
        let e1 = idx.append_segment(400, dur(6), 540_000, true);

        assert_eq!(e0.range.offset, 100);
        assert_eq!(e0.range.length, 500);
        assert_eq!(e1.range.offset, 600);
        assert_eq!(e1.range.length, 400);
        assert_eq!(idx.total_bytes(), 1000);
        assert_eq!(idx.segment_count(), 2);
    }

    #[test]
    fn test_index_get() {
        let mut idx = CmafByteRangeIndex::new(100, "v.mp4", 90_000);
        idx.append_segment(500, dur(6), 0, true);
        let entry = idx.get(0);
        assert!(entry.is_some());
        assert_eq!(entry.map(|e| e.range.offset), Some(100));
    }

    #[test]
    fn test_index_get_none() {
        let idx = CmafByteRangeIndex::new(100, "v.mp4", 90_000);
        assert!(idx.get(0).is_none());
    }

    #[test]
    fn test_index_total_duration() {
        let mut idx = CmafByteRangeIndex::new(100, "v.mp4", 90_000);
        idx.append_segment(500, dur(6), 0, true);
        idx.append_segment(400, dur(4), 540_000, false);
        assert_eq!(idx.total_duration(), dur(10));
    }

    #[test]
    fn test_index_keyframe_entries() {
        let mut idx = CmafByteRangeIndex::new(100, "v.mp4", 90_000);
        idx.append_segment(500, dur(6), 0, true);
        idx.append_segment(400, dur(6), 540_000, false);
        idx.append_segment(450, dur(6), 1_080_000, true);
        let kf = idx.keyframe_entries();
        assert_eq!(kf.len(), 2);
        assert_eq!(kf[0].index, 0);
        assert_eq!(kf[1].index, 2);
    }

    #[test]
    fn test_index_validate_ok() {
        let mut idx = CmafByteRangeIndex::new(100, "v.mp4", 90_000);
        idx.append_segment(500, dur(6), 0, true);
        idx.append_segment(400, dur(6), 540_000, true);
        assert!(idx.validate().is_ok());
    }

    #[test]
    fn test_index_to_hls_segments() {
        let mut idx = CmafByteRangeIndex::new(256, "media.mp4", 90_000);
        idx.append_segment(1000, dur(6), 0, true);
        idx.append_segment(900, dur(6), 540_000, true);
        let hls = idx.to_hls_segments();

        assert!(hls.contains("#EXTINF:6"));
        assert!(hls.contains("#EXT-X-BYTERANGE:1000@256"));
        assert!(hls.contains("#EXT-X-BYTERANGE:900@1256"));
        assert_eq!(hls.matches("media.mp4").count(), 2);
    }

    #[test]
    fn test_index_to_hls_map_tag() {
        let idx = CmafByteRangeIndex::new(512, "video.mp4", 90_000);
        let tag = idx.to_hls_map_tag();
        assert!(tag.contains("EXT-X-MAP"));
        assert!(tag.contains("video.mp4"));
        assert!(tag.contains("512@0"));
    }

    #[test]
    fn test_index_to_dash_segment_base() {
        let idx = CmafByteRangeIndex::new(512, "v.mp4", 90_000);
        let xml = idx.to_dash_segment_base();
        assert!(xml.contains("indexRange=\"0-511\""));
        assert!(xml.contains("timescale=\"90000\""));
    }

    #[test]
    fn test_index_to_dash_segment_list() {
        let mut idx = CmafByteRangeIndex::new(256, "v.mp4", 90_000);
        idx.append_segment(1000, dur(6), 0, true);
        let xml = idx.to_dash_segment_list();
        assert!(xml.contains("<SegmentList"));
        assert!(xml.contains("mediaRange=\"256-1255\""));
        assert!(xml.contains("</SegmentList>"));
    }

    // --- CmafByteRangeWriter ------------------------------------------------

    #[test]
    fn test_writer_new_empty() {
        let w = CmafByteRangeWriter::new();
        assert!(w.index().is_none());
        assert!(w.data().is_empty());
    }

    #[test]
    fn test_writer_write_init() {
        let mut w = CmafByteRangeWriter::new();
        w.write_init(b"init", "v.mp4", 90_000)
            .expect("write_init should succeed");
        assert!(w.index().is_some());
        assert_eq!(w.data(), b"init");
    }

    #[test]
    fn test_writer_double_init_fails() {
        let mut w = CmafByteRangeWriter::new();
        w.write_init(b"init", "v.mp4", 90_000)
            .expect("first init should succeed");
        assert!(w.write_init(b"init2", "v.mp4", 90_000).is_err());
    }

    #[test]
    fn test_writer_append_without_init_fails() {
        let mut w = CmafByteRangeWriter::new();
        assert!(w.append_segment(b"seg", dur(6), 0, true).is_err());
    }

    #[test]
    fn test_writer_append_segment() {
        let mut w = CmafByteRangeWriter::new();
        w.write_init(&[0u8; 64], "v.mp4", 90_000)
            .expect("write_init should succeed");
        let entry = w
            .append_segment(&[1u8; 128], dur(6), 0, true)
            .expect("append should succeed");
        assert_eq!(entry.range.offset, 64);
        assert_eq!(entry.range.length, 128);
        assert_eq!(w.data().len(), 192);
    }

    #[test]
    fn test_writer_finish() {
        let mut w = CmafByteRangeWriter::new();
        w.write_init(&[0u8; 64], "v.mp4", 90_000)
            .expect("init should succeed");
        w.append_segment(&[1u8; 128], dur(6), 0, true)
            .expect("seg0 should succeed");
        w.append_segment(&[2u8; 96], dur(4), 540_000, true)
            .expect("seg1 should succeed");

        let (data, index) = w.finish().expect("finish should succeed");
        assert_eq!(data.len(), 64 + 128 + 96);
        assert_eq!(index.segment_count(), 2);
        assert!(index.validate().is_ok());
    }

    #[test]
    fn test_writer_finish_without_init_fails() {
        let w = CmafByteRangeWriter::new();
        assert!(w.finish().is_err());
    }

    #[test]
    fn test_writer_append_media_samples() {
        let mut w = CmafByteRangeWriter::new();
        let init = crate::isobmff_writer::write_init_segment(
            &crate::isobmff_writer::InitConfig::new(1920, 1080, 90_000, *b"av01"),
        );
        w.write_init(&init, "video.mp4", 90_000)
            .expect("init should succeed");

        let samples = vec![
            MediaSample::new(vec![0xAA; 100], 3_000, true),
            MediaSample::new(vec![0xBB; 80], 3_000, false),
        ];
        let entry = w
            .append_media_samples(&samples, 1, 0, dur(6))
            .expect("append samples should succeed");
        assert!(entry.starts_with_keyframe);
        assert!(entry.range.length > 0);
    }

    // --- write_styp ---------------------------------------------------------

    #[test]
    fn test_write_styp_fourcc() {
        let styp = write_styp();
        assert_eq!(&styp[4..8], b"styp");
    }

    #[test]
    fn test_write_styp_major_brand() {
        let styp = write_styp();
        assert_eq!(&styp[8..12], b"msdh");
    }

    #[test]
    fn test_write_styp_size_correct() {
        let styp = write_styp();
        let size = u32::from_be_bytes(styp[0..4].try_into().expect("4 bytes")) as usize;
        assert_eq!(size, styp.len());
    }

    #[test]
    fn test_write_styp_contains_cmfc() {
        let styp = write_styp();
        let found = styp.windows(4).any(|w| w == b"cmfc");
        assert!(found, "styp should contain cmfc compatible brand");
    }

    // --- Integration: HLS playlist from writer ------------------------------

    #[test]
    fn test_writer_hls_integration() {
        let mut w = CmafByteRangeWriter::new();
        let init = vec![0u8; 100];
        let seg0 = vec![1u8; 500];
        let seg1 = vec![2u8; 400];

        w.write_init(&init, "video.mp4", 90_000)
            .expect("init should succeed");
        w.append_segment(&seg0, dur(6), 0, true)
            .expect("seg0 should succeed");
        w.append_segment(&seg1, dur(6), 540_000, true)
            .expect("seg1 should succeed");

        let (_, index) = w.finish().expect("finish should succeed");
        let hls = index.to_hls_segments();

        assert_eq!(hls.matches("video.mp4").count(), 2);
        assert!(hls.contains("500@100"));
        assert!(hls.contains("400@600"));
    }
}
