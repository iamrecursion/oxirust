#![allow(dead_code)]
//! Live manifest update engine for adaptive streaming.
//!
//! Manages incremental updates to HLS and DASH manifests during live
//! packaging, including sliding-window segment lists, sequence numbering,
//! and discontinuity tracking.

use std::collections::VecDeque;
use std::fmt;
use std::time::Duration;

/// The type of streaming manifest being managed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManifestType {
    /// HLS M3U8 media playlist.
    HlsMedia,
    /// HLS M3U8 master / multivariant playlist.
    HlsMaster,
    /// DASH MPD manifest.
    DashMpd,
}

impl fmt::Display for ManifestType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::HlsMedia => "HLS Media",
            Self::HlsMaster => "HLS Master",
            Self::DashMpd => "DASH MPD",
        };
        write!(f, "{label}")
    }
}

/// An entry representing a single segment in a manifest.
#[derive(Debug, Clone, PartialEq)]
pub struct ManifestSegmentEntry {
    /// Media sequence number.
    pub sequence: u64,
    /// Segment duration.
    pub duration: Duration,
    /// Segment URI (relative path or full URL).
    pub uri: String,
    /// Whether this segment introduces a discontinuity.
    pub discontinuity: bool,
    /// Optional byte range (offset, length).
    pub byte_range: Option<(u64, u64)>,
    /// Optional program date-time tag value (ISO-8601 string).
    pub program_date_time: Option<String>,
}

impl ManifestSegmentEntry {
    /// Create a basic segment entry.
    #[must_use]
    pub fn new(sequence: u64, duration: Duration, uri: impl Into<String>) -> Self {
        Self {
            sequence,
            duration,
            uri: uri.into(),
            discontinuity: false,
            byte_range: None,
            program_date_time: None,
        }
    }

    /// Mark as discontinuity.
    #[must_use]
    pub fn with_discontinuity(mut self) -> Self {
        self.discontinuity = true;
        self
    }

    /// Set byte range.
    #[must_use]
    pub fn with_byte_range(mut self, offset: u64, length: u64) -> Self {
        self.byte_range = Some((offset, length));
        self
    }

    /// Set program date-time.
    #[must_use]
    pub fn with_program_date_time(mut self, pdt: impl Into<String>) -> Self {
        self.program_date_time = Some(pdt.into());
        self
    }

    /// Format the `#EXTINF` line for HLS.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn extinf_line(&self) -> String {
        let secs = self.duration.as_secs_f64();
        format!("#EXTINF:{secs:.6},")
    }
}

/// Represents the delta produced by an incremental manifest update.
///
/// Contains the newly added segment, any removed segment (from sliding window
/// trimming), and updated sequence/version counters. This allows consumers
/// to patch an existing manifest file on disk rather than regenerating it
/// from scratch.
#[derive(Debug, Clone)]
pub struct ManifestDelta {
    /// The segment that was added.
    pub added: ManifestSegmentEntry,
    /// The segment that was removed from the window (if any).
    pub removed: Option<ManifestSegmentEntry>,
    /// Updated media sequence number after this delta.
    pub new_media_sequence: u64,
    /// Updated manifest version after this delta.
    pub new_version: u64,
    /// Current target duration in seconds (may have increased).
    pub target_duration_secs: u64,
    /// Whether this segment introduces a discontinuity.
    pub is_discontinuity: bool,
}

/// A live manifest update tracker.
#[derive(Debug, Clone)]
pub struct ManifestUpdater {
    /// Manifest type.
    manifest_type: ManifestType,
    /// Current media sequence number (HLS).
    media_sequence: u64,
    /// Target duration (maximum segment duration seen).
    target_duration: Duration,
    /// Sliding window of segment entries.
    window: VecDeque<ManifestSegmentEntry>,
    /// Maximum number of segments in the window (0 = unlimited / VOD).
    max_window_size: usize,
    /// Discontinuity sequence counter.
    discontinuity_sequence: u64,
    /// Version counter for the manifest (incremented on each update).
    version: u64,
    /// Whether the stream has ended (EXT-X-ENDLIST).
    ended: bool,
}

impl ManifestUpdater {
    /// Create a new manifest updater.
    #[must_use]
    pub fn new(manifest_type: ManifestType, max_window_size: usize) -> Self {
        Self {
            manifest_type,
            media_sequence: 0,
            target_duration: Duration::ZERO,
            window: VecDeque::new(),
            max_window_size,
            discontinuity_sequence: 0,
            version: 0,
            ended: false,
        }
    }

    /// Add a new segment to the manifest.
    pub fn add_segment(&mut self, entry: ManifestSegmentEntry) {
        if entry.duration > self.target_duration {
            self.target_duration = entry.duration;
        }
        if entry.discontinuity {
            self.discontinuity_sequence += 1;
        }

        self.window.push_back(entry);

        // Trim oldest segments if we exceed the window
        if self.max_window_size > 0 {
            while self.window.len() > self.max_window_size {
                self.window.pop_front();
                self.media_sequence += 1;
            }
        }

        self.version += 1;
    }

    /// Signal end of stream.
    pub fn end_stream(&mut self) {
        self.ended = true;
        self.version += 1;
    }

    /// Whether the manifest is for a live stream (has a sliding window).
    #[must_use]
    pub fn is_live(&self) -> bool {
        self.max_window_size > 0 && !self.ended
    }

    /// Current media sequence number.
    #[must_use]
    pub fn media_sequence(&self) -> u64 {
        self.media_sequence
    }

    /// Target duration (rounded up to nearest second for HLS compliance).
    #[must_use]
    pub fn target_duration_secs(&self) -> u64 {
        let ms = self.target_duration.as_millis();
        let secs = ms / 1000;
        if ms % 1000 > 0 {
            secs as u64 + 1
        } else {
            secs as u64
        }
    }

    /// Number of segments currently in the window.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.window.len()
    }

    /// Manifest version.
    #[must_use]
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Iterate over current segments.
    #[must_use]
    pub fn segments(&self) -> &VecDeque<ManifestSegmentEntry> {
        &self.window
    }

    /// Return whether the stream has ended.
    #[must_use]
    pub fn is_ended(&self) -> bool {
        self.ended
    }

    /// Discontinuity sequence count.
    #[must_use]
    pub fn discontinuity_sequence(&self) -> u64 {
        self.discontinuity_sequence
    }

    /// Apply an incremental update to the manifest without full regeneration.
    ///
    /// This adds a new segment and optionally trims the oldest, returning
    /// only the delta (new lines to append) for efficient live streaming.
    /// This avoids re-rendering the entire playlist on each segment arrival.
    pub fn incremental_update(&mut self, entry: ManifestSegmentEntry) -> ManifestDelta {
        let trimmed_sequence =
            if self.max_window_size > 0 && self.window.len() >= self.max_window_size {
                let old = self.window.front().cloned();
                self.window.pop_front();
                self.media_sequence += 1;
                old
            } else {
                None
            };

        if entry.duration > self.target_duration {
            self.target_duration = entry.duration;
        }
        let is_discontinuity = entry.discontinuity;
        if is_discontinuity {
            self.discontinuity_sequence += 1;
        }

        let new_entry = entry.clone();
        self.window.push_back(entry);
        self.version += 1;

        ManifestDelta {
            added: new_entry,
            removed: trimmed_sequence,
            new_media_sequence: self.media_sequence,
            new_version: self.version,
            target_duration_secs: self.target_duration_secs(),
            is_discontinuity,
        }
    }

    /// Apply a batch of segments as an incremental update.
    /// Returns all deltas produced.
    pub fn incremental_batch_update(
        &mut self,
        entries: Vec<ManifestSegmentEntry>,
    ) -> Vec<ManifestDelta> {
        let mut deltas = Vec::with_capacity(entries.len());
        for entry in entries {
            deltas.push(self.incremental_update(entry));
        }
        deltas
    }

    /// Render just the new segment lines for appending to an existing playlist file.
    /// This is useful for live HLS where we patch the file rather than rewriting.
    #[must_use]
    pub fn render_incremental_hls(&self, delta: &ManifestDelta) -> String {
        let mut out = String::new();
        if delta.is_discontinuity {
            out.push_str("#EXT-X-DISCONTINUITY\n");
        }
        if let Some(ref pdt) = delta.added.program_date_time {
            out.push_str(&format!("#EXT-X-PROGRAM-DATE-TIME:{pdt}\n"));
        }
        out.push_str(&delta.added.extinf_line());
        out.push('\n');
        if let Some((offset, length)) = delta.added.byte_range {
            out.push_str(&format!("#EXT-X-BYTERANGE:{length}@{offset}\n"));
        }
        out.push_str(&delta.added.uri);
        out.push('\n');
        out
    }

    /// Render a minimal DASH MPD period string for the current segments.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn render_dash_mpd(&self) -> String {
        let mut out = String::new();
        out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        out.push_str("<MPD xmlns=\"urn:mpeg:dash:schema:mpd:2011\" ");
        if self.is_live() {
            out.push_str("type=\"dynamic\" ");
            out.push_str(&format!(
                "minimumUpdatePeriod=\"PT{}S\" ",
                self.target_duration_secs()
            ));
        } else {
            out.push_str("type=\"static\" ");
        }

        // Calculate total duration
        let total_ms: u128 = self.window.iter().map(|e| e.duration.as_millis()).sum();
        let total_secs = total_ms as f64 / 1000.0;
        out.push_str(&format!(
            "mediaPresentationDuration=\"PT{total_secs:.3}S\">\n"
        ));

        out.push_str("  <Period>\n");
        out.push_str("    <AdaptationSet mimeType=\"video/mp4\">\n");
        out.push_str("      <SegmentList>\n");

        for entry in &self.window {
            let dur_secs = entry.duration.as_secs_f64();
            out.push_str(&format!(
                "        <SegmentURL media=\"{}\" duration=\"{dur_secs:.6}\"/>\n",
                entry.uri
            ));
        }

        out.push_str("      </SegmentList>\n");
        out.push_str("    </AdaptationSet>\n");
        out.push_str("  </Period>\n");
        out.push_str("</MPD>\n");
        out
    }

    /// Render a minimal HLS media playlist string.
    #[must_use]
    pub fn render_hls_media_playlist(&self) -> String {
        let mut out = String::new();
        out.push_str("#EXTM3U\n");
        out.push_str("#EXT-X-VERSION:6\n");
        out.push_str(&format!(
            "#EXT-X-TARGETDURATION:{}\n",
            self.target_duration_secs()
        ));
        out.push_str(&format!("#EXT-X-MEDIA-SEQUENCE:{}\n", self.media_sequence));

        if self.discontinuity_sequence > 0 {
            out.push_str(&format!(
                "#EXT-X-DISCONTINUITY-SEQUENCE:{}\n",
                self.discontinuity_sequence
            ));
        }

        for entry in &self.window {
            if entry.discontinuity {
                out.push_str("#EXT-X-DISCONTINUITY\n");
            }
            if let Some(ref pdt) = entry.program_date_time {
                out.push_str(&format!("#EXT-X-PROGRAM-DATE-TIME:{pdt}\n"));
            }
            out.push_str(&entry.extinf_line());
            out.push('\n');
            if let Some((offset, length)) = entry.byte_range {
                out.push_str(&format!("#EXT-X-BYTERANGE:{length}@{offset}\n"));
            }
            out.push_str(&entry.uri);
            out.push('\n');
        }

        if self.ended {
            out.push_str("#EXT-X-ENDLIST\n");
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_manifest_type_display() {
        assert_eq!(ManifestType::HlsMedia.to_string(), "HLS Media");
        assert_eq!(ManifestType::DashMpd.to_string(), "DASH MPD");
    }

    #[test]
    fn test_segment_entry_creation() {
        let e = ManifestSegmentEntry::new(0, Duration::from_secs(6), "seg0.m4s");
        assert_eq!(e.sequence, 0);
        assert_eq!(e.uri, "seg0.m4s");
        assert!(!e.discontinuity);
    }

    #[test]
    fn test_segment_entry_discontinuity() {
        let e =
            ManifestSegmentEntry::new(1, Duration::from_secs(6), "seg1.m4s").with_discontinuity();
        assert!(e.discontinuity);
    }

    #[test]
    fn test_segment_entry_byte_range() {
        let e = ManifestSegmentEntry::new(0, Duration::from_secs(6), "seg0.m4s")
            .with_byte_range(100, 5000);
        assert_eq!(e.byte_range, Some((100, 5000)));
    }

    #[test]
    fn test_extinf_line() {
        let e = ManifestSegmentEntry::new(0, Duration::from_millis(6006), "seg0.m4s");
        let line = e.extinf_line();
        assert!(line.starts_with("#EXTINF:6.006"));
    }

    #[test]
    fn test_updater_add_segment() {
        let mut u = ManifestUpdater::new(ManifestType::HlsMedia, 5);
        u.add_segment(ManifestSegmentEntry::new(
            0,
            Duration::from_secs(6),
            "s0.m4s",
        ));
        assert_eq!(u.segment_count(), 1);
        assert_eq!(u.version(), 1);
    }

    #[test]
    fn test_sliding_window() {
        let mut u = ManifestUpdater::new(ManifestType::HlsMedia, 3);
        for i in 0..5_u64 {
            u.add_segment(ManifestSegmentEntry::new(
                i,
                Duration::from_secs(6),
                format!("s{i}.m4s"),
            ));
        }
        assert_eq!(u.segment_count(), 3);
        assert_eq!(u.media_sequence(), 2); // 5-3 = 2 trimmed
    }

    #[test]
    fn test_target_duration() {
        let mut u = ManifestUpdater::new(ManifestType::HlsMedia, 10);
        u.add_segment(ManifestSegmentEntry::new(
            0,
            Duration::from_millis(5500),
            "s0.m4s",
        ));
        u.add_segment(ManifestSegmentEntry::new(
            1,
            Duration::from_millis(6200),
            "s1.m4s",
        ));
        // 6.2s rounds up to 7
        assert_eq!(u.target_duration_secs(), 7);
    }

    #[test]
    fn test_end_stream() {
        let mut u = ManifestUpdater::new(ManifestType::HlsMedia, 5);
        u.add_segment(ManifestSegmentEntry::new(
            0,
            Duration::from_secs(6),
            "s0.m4s",
        ));
        u.end_stream();
        assert!(u.is_ended());
        assert!(!u.is_live());
    }

    #[test]
    fn test_is_live() {
        let u = ManifestUpdater::new(ManifestType::HlsMedia, 5);
        assert!(u.is_live());
        let u2 = ManifestUpdater::new(ManifestType::HlsMedia, 0);
        assert!(!u2.is_live());
    }

    #[test]
    fn test_render_hls_basic() {
        let mut u = ManifestUpdater::new(ManifestType::HlsMedia, 10);
        u.add_segment(ManifestSegmentEntry::new(
            0,
            Duration::from_secs(6),
            "s0.m4s",
        ));
        u.add_segment(ManifestSegmentEntry::new(
            1,
            Duration::from_secs(6),
            "s1.m4s",
        ));
        let playlist = u.render_hls_media_playlist();
        assert!(playlist.contains("#EXTM3U"));
        assert!(playlist.contains("#EXT-X-TARGETDURATION:6"));
        assert!(playlist.contains("s0.m4s"));
        assert!(playlist.contains("s1.m4s"));
    }

    #[test]
    fn test_render_hls_endlist() {
        let mut u = ManifestUpdater::new(ManifestType::HlsMedia, 10);
        u.add_segment(ManifestSegmentEntry::new(
            0,
            Duration::from_secs(6),
            "s0.m4s",
        ));
        u.end_stream();
        let playlist = u.render_hls_media_playlist();
        assert!(playlist.contains("#EXT-X-ENDLIST"));
    }

    #[test]
    fn test_discontinuity_tracking() {
        let mut u = ManifestUpdater::new(ManifestType::HlsMedia, 10);
        u.add_segment(ManifestSegmentEntry::new(
            0,
            Duration::from_secs(6),
            "s0.m4s",
        ));
        u.add_segment(
            ManifestSegmentEntry::new(1, Duration::from_secs(6), "s1.m4s").with_discontinuity(),
        );
        assert_eq!(u.discontinuity_sequence(), 1);
        let playlist = u.render_hls_media_playlist();
        assert!(playlist.contains("#EXT-X-DISCONTINUITY"));
    }

    #[test]
    fn test_program_date_time() {
        let e = ManifestSegmentEntry::new(0, Duration::from_secs(6), "s0.m4s")
            .with_program_date_time("2026-03-02T12:00:00Z");
        assert_eq!(e.program_date_time.as_deref(), Some("2026-03-02T12:00:00Z"));
    }

    #[test]
    fn test_vod_mode() {
        let mut u = ManifestUpdater::new(ManifestType::HlsMedia, 0);
        for i in 0..10_u64 {
            u.add_segment(ManifestSegmentEntry::new(
                i,
                Duration::from_secs(6),
                format!("s{i}.m4s"),
            ));
        }
        // unlimited window, no trimming
        assert_eq!(u.segment_count(), 10);
        assert_eq!(u.media_sequence(), 0);
    }

    // --- Incremental update tests ---

    #[test]
    fn test_incremental_update_adds_segment() {
        let mut u = ManifestUpdater::new(ManifestType::HlsMedia, 5);
        let delta = u.incremental_update(ManifestSegmentEntry::new(
            0,
            Duration::from_secs(6),
            "s0.m4s",
        ));
        assert_eq!(delta.added.uri, "s0.m4s");
        assert!(delta.removed.is_none());
        assert_eq!(delta.new_version, 1);
        assert_eq!(u.segment_count(), 1);
    }

    #[test]
    fn test_incremental_update_sliding_window() {
        let mut u = ManifestUpdater::new(ManifestType::HlsMedia, 3);
        for i in 0..3_u64 {
            u.incremental_update(ManifestSegmentEntry::new(
                i,
                Duration::from_secs(6),
                format!("s{i}.m4s"),
            ));
        }
        // Window is now full, next add should trim
        let delta = u.incremental_update(ManifestSegmentEntry::new(
            3,
            Duration::from_secs(6),
            "s3.m4s",
        ));
        assert!(delta.removed.is_some());
        assert_eq!(
            delta.removed.as_ref().map(|r| r.uri.as_str()),
            Some("s0.m4s")
        );
        assert_eq!(delta.new_media_sequence, 1);
        assert_eq!(u.segment_count(), 3);
    }

    #[test]
    fn test_incremental_update_discontinuity() {
        let mut u = ManifestUpdater::new(ManifestType::HlsMedia, 10);
        let delta = u.incremental_update(
            ManifestSegmentEntry::new(0, Duration::from_secs(6), "s0.m4s").with_discontinuity(),
        );
        assert!(delta.is_discontinuity);
        assert_eq!(u.discontinuity_sequence(), 1);
    }

    #[test]
    fn test_incremental_batch_update() {
        let mut u = ManifestUpdater::new(ManifestType::HlsMedia, 5);
        let entries: Vec<ManifestSegmentEntry> = (0..4)
            .map(|i| ManifestSegmentEntry::new(i, Duration::from_secs(6), format!("s{i}.m4s")))
            .collect();
        let deltas = u.incremental_batch_update(entries);
        assert_eq!(deltas.len(), 4);
        assert_eq!(u.segment_count(), 4);
        assert_eq!(u.version(), 4);
    }

    #[test]
    fn test_incremental_batch_with_trimming() {
        let mut u = ManifestUpdater::new(ManifestType::HlsMedia, 2);
        let entries: Vec<ManifestSegmentEntry> = (0..5)
            .map(|i| ManifestSegmentEntry::new(i, Duration::from_secs(6), format!("s{i}.m4s")))
            .collect();
        let deltas = u.incremental_batch_update(entries);
        assert_eq!(deltas.len(), 5);
        // Window size is 2, so 3 segments trimmed
        let removed_count = deltas.iter().filter(|d| d.removed.is_some()).count();
        assert_eq!(removed_count, 3);
        assert_eq!(u.segment_count(), 2);
        assert_eq!(u.media_sequence(), 3);
    }

    #[test]
    fn test_render_incremental_hls_basic() {
        let mut u = ManifestUpdater::new(ManifestType::HlsMedia, 10);
        let delta = u.incremental_update(ManifestSegmentEntry::new(
            0,
            Duration::from_secs(6),
            "s0.m4s",
        ));
        let lines = u.render_incremental_hls(&delta);
        assert!(lines.contains("#EXTINF:6"));
        assert!(lines.contains("s0.m4s"));
    }

    #[test]
    fn test_render_incremental_hls_with_discontinuity() {
        let mut u = ManifestUpdater::new(ManifestType::HlsMedia, 10);
        let delta = u.incremental_update(
            ManifestSegmentEntry::new(0, Duration::from_secs(6), "s0.m4s").with_discontinuity(),
        );
        let lines = u.render_incremental_hls(&delta);
        assert!(lines.contains("#EXT-X-DISCONTINUITY"));
    }

    #[test]
    fn test_render_incremental_hls_with_byte_range() {
        let mut u = ManifestUpdater::new(ManifestType::HlsMedia, 10);
        let delta = u.incremental_update(
            ManifestSegmentEntry::new(0, Duration::from_secs(6), "s0.m4s")
                .with_byte_range(1000, 5000),
        );
        let lines = u.render_incremental_hls(&delta);
        assert!(lines.contains("#EXT-X-BYTERANGE:5000@1000"));
    }

    #[test]
    fn test_render_dash_mpd_static() {
        let mut u = ManifestUpdater::new(ManifestType::DashMpd, 0);
        u.add_segment(ManifestSegmentEntry::new(
            0,
            Duration::from_secs(6),
            "seg0.m4s",
        ));
        let mpd = u.render_dash_mpd();
        assert!(mpd.contains("type=\"static\""));
        assert!(mpd.contains("seg0.m4s"));
        assert!(mpd.contains("</MPD>"));
    }

    #[test]
    fn test_render_dash_mpd_dynamic() {
        let mut u = ManifestUpdater::new(ManifestType::DashMpd, 5);
        u.add_segment(ManifestSegmentEntry::new(
            0,
            Duration::from_secs(6),
            "seg0.m4s",
        ));
        let mpd = u.render_dash_mpd();
        assert!(mpd.contains("type=\"dynamic\""));
        assert!(mpd.contains("minimumUpdatePeriod"));
    }

    #[test]
    fn test_render_dash_mpd_multiple_segments() {
        let mut u = ManifestUpdater::new(ManifestType::DashMpd, 10);
        for i in 0..3_u64 {
            u.add_segment(ManifestSegmentEntry::new(
                i,
                Duration::from_secs(6),
                format!("seg{i}.m4s"),
            ));
        }
        let mpd = u.render_dash_mpd();
        assert!(mpd.contains("seg0.m4s"));
        assert!(mpd.contains("seg1.m4s"));
        assert!(mpd.contains("seg2.m4s"));
    }

    #[test]
    fn test_manifest_delta_target_duration_update() {
        let mut u = ManifestUpdater::new(ManifestType::HlsMedia, 10);
        u.incremental_update(ManifestSegmentEntry::new(
            0,
            Duration::from_millis(5500),
            "s0.m4s",
        ));
        let delta = u.incremental_update(ManifestSegmentEntry::new(
            1,
            Duration::from_millis(7200),
            "s1.m4s",
        ));
        // 7.2s rounds up to 8
        assert_eq!(delta.target_duration_secs, 8);
    }

    #[test]
    fn test_incremental_update_version_monotonic() {
        let mut u = ManifestUpdater::new(ManifestType::HlsMedia, 10);
        let d1 = u.incremental_update(ManifestSegmentEntry::new(
            0,
            Duration::from_secs(6),
            "s0.m4s",
        ));
        let d2 = u.incremental_update(ManifestSegmentEntry::new(
            1,
            Duration::from_secs(6),
            "s1.m4s",
        ));
        assert!(d2.new_version > d1.new_version);
    }
}
