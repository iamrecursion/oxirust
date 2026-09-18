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
//! │  init.mp4  (ftyp + moov)                                    │
//! ├──────────────────────────────────────────────────────────────┤
//! │  segment_0  (styp + moof + mdat)   offset=0                │
//! ├──────────────────────────────────────────────────────────────┤
//! │  segment_1  (styp + moof + mdat)   offset=len(seg0)        │
//! ├──────────────────────────────────────────────────────────────┤
//! │  …                                                           │
//! └──────────────────────────────────────────────────────────────┘
//! ```
//!
//! Each [`ByteRangeEntry`] records the byte offset and length of one segment
//! within the container file.  The [`ByteRangeIndex`] owns the complete
//! collection and provides serialisation helpers for M3U8 `EXT-X-BYTERANGE`
//! and DASH `SegmentBase@indexRange` attributes.
//!
//! # References
//!
//! - ISO/IEC 23000-19 (CMAF), section 7 — byte-range CMAF addressing
//! - HLS RFC 8216, section 4.3.2.2 — `EXT-X-BYTERANGE`
//! - DASH-IF IOP v4.3, section 4.5 — `SegmentBase` byte ranges

use crate::error::{PackagerError, PackagerResult};
use std::time::Duration;

// ---------------------------------------------------------------------------
// ByteRangeEntry
// ---------------------------------------------------------------------------

/// The byte range of a single CMAF segment within a container file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteRangeEntry {
    /// Zero-based segment index within the container file.
    pub index: u32,
    /// Byte offset of the segment's first byte in the container file.
    pub offset: u64,
    /// Total byte length of this segment (all boxes: `styp` + `moof` + `mdat`).
    pub length: u64,
    /// Segment presentation duration.
    pub duration: Duration,
}

impl ByteRangeEntry {
    /// Create a new byte-range entry.
    #[must_use]
    pub fn new(index: u32, offset: u64, length: u64, duration: Duration) -> Self {
        Self {
            index,
            offset,
            length,
            duration,
        }
    }

    /// Exclusive end byte (`offset + length`).
    ///
    /// This is the value used in an HTTP `Range` header's upper bound (inclusive),
    /// so the actual end byte for range requests is `end_byte() - 1`.
    #[must_use]
    pub fn end_byte(&self) -> u64 {
        self.offset + self.length
    }

    /// Format as an HLS `EXT-X-BYTERANGE` attribute value: `<length>[@<offset>]`.
    ///
    /// The `@<offset>` part is emitted only when `offset > 0` or when
    /// `force_offset` is `true`, per RFC 8216 §4.3.2.2.
    #[must_use]
    pub fn hls_byterange_attr(&self, force_offset: bool) -> String {
        if self.offset == 0 && !force_offset {
            format!("{}", self.length)
        } else {
            format!("{}@{}", self.length, self.offset)
        }
    }

    /// Format as a DASH `@mediaRange` attribute value: `<start>-<end_inclusive>`.
    #[must_use]
    pub fn dash_media_range(&self) -> String {
        format!("{}-{}", self.offset, self.end_byte().saturating_sub(1))
    }
}

// ---------------------------------------------------------------------------
// InitSegmentRange
// ---------------------------------------------------------------------------

/// Byte range of the init segment (`ftyp` + `moov`) at the start of the
/// container file.
///
/// The first media segment immediately follows the init segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InitSegmentRange {
    /// Byte length of the init segment.
    pub length: u64,
}

impl InitSegmentRange {
    /// Create a new init segment range.
    #[must_use]
    pub fn new(length: u64) -> Self {
        Self { length }
    }

    /// Format as a DASH `@indexRange` attribute value.
    #[must_use]
    pub fn dash_index_range(&self) -> String {
        format!("0-{}", self.length.saturating_sub(1))
    }
}

// ---------------------------------------------------------------------------
// ByteRangeIndex
// ---------------------------------------------------------------------------

/// A complete byte-range index for a single-file CMAF container.
///
/// Tracks the init segment, all media segment ranges, and provides helpers
/// for generating manifest attributes.
#[derive(Debug, Clone)]
pub struct ByteRangeIndex {
    /// Init segment byte range.
    init: InitSegmentRange,
    /// Media segment entries in segment order.
    segments: Vec<ByteRangeEntry>,
    /// Running write cursor (bytes appended so far after init).
    cursor: u64,
}

impl ByteRangeIndex {
    /// Create a new byte-range index.
    ///
    /// `init_length` is the size of the init segment (`ftyp` + `moov`) in bytes.
    #[must_use]
    pub fn new(init_length: u64) -> Self {
        Self {
            init: InitSegmentRange::new(init_length),
            segments: Vec::new(),
            cursor: init_length,
        }
    }

    /// Append a media segment of `segment_length` bytes with the given duration.
    ///
    /// Returns the [`ByteRangeEntry`] that was recorded.
    pub fn append_segment(&mut self, segment_length: u64, duration: Duration) -> ByteRangeEntry {
        let index = self.segments.len() as u32;
        let entry = ByteRangeEntry::new(index, self.cursor, segment_length, duration);
        self.cursor += segment_length;
        self.segments.push(entry.clone());
        entry
    }

    /// Return the init segment range.
    #[must_use]
    pub fn init(&self) -> &InitSegmentRange {
        &self.init
    }

    /// Return all media segment entries.
    #[must_use]
    pub fn segments(&self) -> &[ByteRangeEntry] {
        &self.segments
    }

    /// Return the total file size (init + all segments).
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.cursor
    }

    /// Return the number of media segments.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Look up an entry by segment index.
    #[must_use]
    pub fn get(&self, index: u32) -> Option<&ByteRangeEntry> {
        self.segments.get(index as usize)
    }

    /// Total presentation duration of all segments.
    #[must_use]
    pub fn total_duration(&self) -> Duration {
        self.segments.iter().map(|s| s.duration).sum()
    }

    /// Validate the index: verify that all entries are contiguous and that
    /// no entry extends beyond the current cursor.
    ///
    /// # Errors
    ///
    /// Returns [`PackagerError::PackagingError`] if any invariant is violated.
    pub fn validate(&self) -> PackagerResult<()> {
        let mut expected_offset = self.init.length;
        for (i, entry) in self.segments.iter().enumerate() {
            if entry.offset != expected_offset {
                return Err(PackagerError::PackagingError(format!(
                    "Segment {i}: expected offset {expected_offset}, found {}",
                    entry.offset
                )));
            }
            if entry.length == 0 {
                return Err(PackagerError::PackagingError(format!(
                    "Segment {i}: length must not be zero"
                )));
            }
            expected_offset += entry.length;
        }
        if expected_offset != self.cursor {
            return Err(PackagerError::PackagingError(format!(
                "Cursor mismatch: expected {expected_offset}, found {}",
                self.cursor
            )));
        }
        Ok(())
    }

    /// Serialise all segment entries as HLS `#EXT-X-BYTERANGE` + URI lines,
    /// appended to a playlist string.
    ///
    /// Each entry produces:
    /// ```text
    /// #EXTINF:<duration_secs>,
    /// #EXT-X-BYTERANGE:<length>@<offset>
    /// <container_uri>
    /// ```
    ///
    /// # Arguments
    ///
    /// * `container_uri` – the URI of the single container file.
    #[must_use]
    pub fn to_hls_segments(&self, container_uri: &str) -> String {
        let mut out = String::new();
        for entry in &self.segments {
            let secs = entry.duration.as_secs_f64();
            out.push_str(&format!("#EXTINF:{secs:.6},\n"));
            out.push_str(&format!(
                "#EXT-X-BYTERANGE:{}\n",
                entry.hls_byterange_attr(true)
            ));
            out.push_str(container_uri);
            out.push('\n');
        }
        out
    }

    /// Serialise a DASH `SegmentBase` XML fragment for this index.
    ///
    /// ```xml
    /// <SegmentBase indexRange="0-1023" timescale="90000">
    ///   <Initialization range="0-511"/>
    /// </SegmentBase>
    /// ```
    ///
    /// The `timescale` argument should match the track's media timescale.
    #[must_use]
    pub fn to_dash_segment_base(&self, timescale: u32) -> String {
        let init_end = self.init.length.saturating_sub(1);
        format!(
            r#"<SegmentBase indexRange="0-{init_end}" timescale="{timescale}"><Initialization range="0-{init_end}"/></SegmentBase>"#
        )
    }
}

// ---------------------------------------------------------------------------
// ByteRangeWriter
// ---------------------------------------------------------------------------

/// A write-once accumulator for building a single-file CMAF container in
/// memory (suitable for tests and small streams).
///
/// For large streams, write directly to a file or stream and maintain a
/// [`ByteRangeIndex`] manually.
#[derive(Debug, Default)]
pub struct ByteRangeWriter {
    data: Vec<u8>,
    index: Option<ByteRangeIndex>,
}

impl ByteRangeWriter {
    /// Create a new writer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Write the init segment bytes.
    ///
    /// Must be called exactly once, before any `append_segment` calls.
    ///
    /// # Errors
    ///
    /// Returns an error if the init segment has already been written.
    pub fn write_init(&mut self, init_bytes: &[u8]) -> PackagerResult<()> {
        if self.index.is_some() {
            return Err(PackagerError::PackagingError(
                "init segment already written".into(),
            ));
        }
        self.data.extend_from_slice(init_bytes);
        self.index = Some(ByteRangeIndex::new(init_bytes.len() as u64));
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
    ) -> PackagerResult<ByteRangeEntry> {
        let idx = self.index.as_mut().ok_or_else(|| {
            PackagerError::PackagingError("init segment must be written first".into())
        })?;
        let entry = idx.append_segment(segment_bytes.len() as u64, duration);
        self.data.extend_from_slice(segment_bytes);
        Ok(entry)
    }

    /// Return the accumulated bytes.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Return the byte-range index, if initialised.
    #[must_use]
    pub fn index(&self) -> Option<&ByteRangeIndex> {
        self.index.as_ref()
    }

    /// Consume the writer and return `(data, index)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the init segment was never written.
    pub fn finish(self) -> PackagerResult<(Vec<u8>, ByteRangeIndex)> {
        let index = self.index.ok_or_else(|| {
            PackagerError::PackagingError("cannot finish: init segment was never written".into())
        })?;
        index.validate()?;
        Ok((self.data, index))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn dur_secs(s: u64) -> Duration {
        Duration::from_secs(s)
    }

    // --- ByteRangeEntry ------------------------------------------------------

    #[test]
    fn test_entry_end_byte() {
        let e = ByteRangeEntry::new(0, 100, 200, dur_secs(6));
        assert_eq!(e.end_byte(), 300);
    }

    #[test]
    fn test_entry_hls_byterange_attr_with_offset() {
        let e = ByteRangeEntry::new(1, 1024, 512, dur_secs(6));
        assert_eq!(e.hls_byterange_attr(false), "512@1024");
    }

    #[test]
    fn test_entry_hls_byterange_attr_no_offset_at_zero() {
        let e = ByteRangeEntry::new(0, 0, 512, dur_secs(6));
        assert_eq!(e.hls_byterange_attr(false), "512");
    }

    #[test]
    fn test_entry_hls_byterange_attr_force_offset_at_zero() {
        let e = ByteRangeEntry::new(0, 0, 512, dur_secs(6));
        assert_eq!(e.hls_byterange_attr(true), "512@0");
    }

    #[test]
    fn test_entry_dash_media_range() {
        let e = ByteRangeEntry::new(0, 0, 1024, dur_secs(6));
        assert_eq!(e.dash_media_range(), "0-1023");
    }

    #[test]
    fn test_entry_dash_media_range_nonzero_offset() {
        let e = ByteRangeEntry::new(1, 1024, 512, dur_secs(6));
        assert_eq!(e.dash_media_range(), "1024-1535");
    }

    // --- InitSegmentRange ----------------------------------------------------

    #[test]
    fn test_init_range_dash_index_range() {
        let r = InitSegmentRange::new(512);
        assert_eq!(r.dash_index_range(), "0-511");
    }

    #[test]
    fn test_init_range_length_one() {
        let r = InitSegmentRange::new(1);
        assert_eq!(r.dash_index_range(), "0-0");
    }

    // --- ByteRangeIndex ------------------------------------------------------

    #[test]
    fn test_index_new() {
        let idx = ByteRangeIndex::new(512);
        assert_eq!(idx.total_bytes(), 512);
        assert_eq!(idx.segment_count(), 0);
        assert!(idx.validate().is_ok());
    }

    #[test]
    fn test_index_append_segment_offsets() {
        let mut idx = ByteRangeIndex::new(256);
        let e0 = idx.append_segment(1000, dur_secs(6));
        let e1 = idx.append_segment(900, dur_secs(6));

        assert_eq!(e0.offset, 256);
        assert_eq!(e0.length, 1000);
        assert_eq!(e1.offset, 256 + 1000);
        assert_eq!(e1.length, 900);
        assert_eq!(idx.total_bytes(), 256 + 1000 + 900);
    }

    #[test]
    fn test_index_segment_count() {
        let mut idx = ByteRangeIndex::new(100);
        idx.append_segment(50, dur_secs(2));
        idx.append_segment(60, dur_secs(2));
        assert_eq!(idx.segment_count(), 2);
    }

    #[test]
    fn test_index_get() {
        let mut idx = ByteRangeIndex::new(100);
        idx.append_segment(400, dur_secs(6));
        let entry = idx.get(0).expect("segment 0 should exist");
        assert_eq!(entry.offset, 100);
        assert_eq!(entry.length, 400);
    }

    #[test]
    fn test_index_get_out_of_bounds() {
        let idx = ByteRangeIndex::new(100);
        assert!(idx.get(0).is_none());
    }

    #[test]
    fn test_index_total_duration() {
        let mut idx = ByteRangeIndex::new(100);
        idx.append_segment(400, dur_secs(6));
        idx.append_segment(380, dur_secs(4));
        assert_eq!(idx.total_duration(), dur_secs(10));
    }

    #[test]
    fn test_index_validate_ok() {
        let mut idx = ByteRangeIndex::new(256);
        idx.append_segment(1000, dur_secs(6));
        idx.append_segment(900, dur_secs(6));
        assert!(idx.validate().is_ok());
    }

    #[test]
    fn test_index_to_hls_segments() {
        let mut idx = ByteRangeIndex::new(256);
        idx.append_segment(1000, dur_secs(6));
        idx.append_segment(900, dur_secs(6));
        let output = idx.to_hls_segments("media.mp4");

        assert!(output.contains("#EXTINF:6"));
        assert!(output.contains("#EXT-X-BYTERANGE:1000@256"));
        assert!(output.contains("#EXT-X-BYTERANGE:900@1256"));
        assert_eq!(output.matches("media.mp4").count(), 2);
    }

    #[test]
    fn test_index_to_dash_segment_base() {
        let idx = ByteRangeIndex::new(512);
        let xml = idx.to_dash_segment_base(90_000);
        assert!(xml.contains("indexRange=\"0-511\""));
        assert!(xml.contains("timescale=\"90000\""));
        assert!(xml.contains("<Initialization"));
    }

    // --- ByteRangeWriter -----------------------------------------------------

    #[test]
    fn test_writer_no_init_no_index() {
        let writer = ByteRangeWriter::new();
        assert!(writer.index().is_none());
        assert!(writer.data().is_empty());
    }

    #[test]
    fn test_writer_write_init() {
        let mut writer = ByteRangeWriter::new();
        writer
            .write_init(b"init_data")
            .expect("write_init should succeed");
        assert!(writer.index().is_some());
        assert_eq!(writer.data(), b"init_data");
    }

    #[test]
    fn test_writer_double_init_fails() {
        let mut writer = ByteRangeWriter::new();
        writer
            .write_init(b"init")
            .expect("first write should succeed");
        assert!(writer.write_init(b"init2").is_err());
    }

    #[test]
    fn test_writer_append_without_init_fails() {
        let mut writer = ByteRangeWriter::new();
        assert!(writer.append_segment(b"seg", dur_secs(6)).is_err());
    }

    #[test]
    fn test_writer_append_segment() {
        let mut writer = ByteRangeWriter::new();
        writer
            .write_init(&[0u8; 64])
            .expect("write_init should succeed");
        let entry = writer
            .append_segment(&[1u8; 128], dur_secs(6))
            .expect("append should succeed");
        assert_eq!(entry.offset, 64);
        assert_eq!(entry.length, 128);
        assert_eq!(writer.data().len(), 64 + 128);
    }

    #[test]
    fn test_writer_finish() {
        let mut writer = ByteRangeWriter::new();
        writer
            .write_init(&[0u8; 64])
            .expect("write_init should succeed");
        writer
            .append_segment(&[1u8; 128], dur_secs(6))
            .expect("append should succeed");
        writer
            .append_segment(&[2u8; 96], dur_secs(4))
            .expect("append should succeed");

        let (data, index) = writer.finish().expect("finish should succeed");
        assert_eq!(data.len(), 64 + 128 + 96);
        assert_eq!(index.segment_count(), 2);
        assert!(index.validate().is_ok());
    }

    #[test]
    fn test_writer_finish_without_init_fails() {
        let writer = ByteRangeWriter::new();
        assert!(writer.finish().is_err());
    }

    #[test]
    fn test_writer_hls_playlist_integration() {
        let mut writer = ByteRangeWriter::new();
        let init = vec![0u8; 100];
        let seg0 = vec![1u8; 500];
        let seg1 = vec![2u8; 400];

        writer.write_init(&init).expect("write_init should succeed");
        writer
            .append_segment(&seg0, dur_secs(6))
            .expect("append should succeed");
        writer
            .append_segment(&seg1, dur_secs(6))
            .expect("append should succeed");

        let (_, index) = writer.finish().expect("finish should succeed");
        let hls = index.to_hls_segments("video.mp4");

        // Both segments reference the same container file
        assert_eq!(hls.matches("video.mp4").count(), 2);
        // Correct byte ranges
        assert!(hls.contains("500@100"));
        assert!(hls.contains("400@600"));
    }
}
