// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Intelligent packaging optimisation.
//!
//! Provides profile-driven configuration generation for adaptive streaming
//! workflows.  Each [`PackagingProfile`] encodes domain-specific knowledge
//! about segment durations, chunk sizes, GOP alignment, CMAF conformance,
//! and trick-play rendition generation.
//!
//! # Profiles
//!
//! | Profile          | Segment | Chunk  | CMAF | Trick-play |
//! |------------------|---------|--------|------|------------|
//! | Low Latency      | 2 s     | 200 ms | yes  | no         |
//! | Broadcast Ingest | 6 s     | 1 s    | yes  | no         |
//! | OTT VOD          | 6 s     | —      | yes  | yes        |
//! | UHD HDR          | 4 s     | 500 ms | yes  | yes        |
//! | Preview          | 2 s     | —      | no   | no         |
//!
//! # Example
//!
//! ```
//! use oximedia_packager::packaging_optimizer::{PackagingProfile, optimize};
//!
//! let settings = optimize(&PackagingProfile::OttVod);
//! assert_eq!(settings.segment_duration_ms, 6_000);
//! assert!(settings.use_cmaf);
//! assert!(settings.enable_trick_play);
//! ```

use std::fmt;

// ---------------------------------------------------------------------------
// PackagingProfile
// ---------------------------------------------------------------------------

/// A predefined packaging profile that encapsulates best-practice defaults
/// for a given delivery scenario.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PackagingProfile {
    /// Ultra-low-latency streaming (LL-HLS / LL-DASH).
    ///
    /// Optimised for sub-3-second glass-to-glass latency with small chunks
    /// and CMAF byte-range addressing.
    LowLatency,

    /// Contribution / broadcast ingest (e.g. RIST, SRT to origin).
    ///
    /// Larger segments for resilience; GOP-aligned for frame-accurate
    /// switching at the origin.
    BroadcastIngest,

    /// Standard over-the-top video-on-demand packaging.
    ///
    /// 6-second segments, CMAF compliance, trick-play rendition for
    /// thumbnail seeking.
    OttVod,

    /// Ultra-HD / HDR content (4K+ with HDR10 / HLG / Dolby Vision metadata).
    ///
    /// Shorter segments to limit decode buffer requirements; trick-play
    /// I-frame track for fast seeking.
    UhdHdr,

    /// Lightweight preview / trailer packaging.
    ///
    /// Short segments, no encryption, no trick-play — designed for quick
    /// preview generation.
    Preview,
}

impl PackagingProfile {
    /// Return all defined profiles.
    #[must_use]
    pub fn all() -> &'static [Self] {
        &[
            Self::LowLatency,
            Self::BroadcastIngest,
            Self::OttVod,
            Self::UhdHdr,
            Self::Preview,
        ]
    }

    /// A short human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::LowLatency => "low-latency",
            Self::BroadcastIngest => "broadcast-ingest",
            Self::OttVod => "ott-vod",
            Self::UhdHdr => "uhd-hdr",
            Self::Preview => "preview",
        }
    }

    /// A longer description of the profile.
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            Self::LowLatency => "Ultra-low-latency streaming (LL-HLS / LL-DASH)",
            Self::BroadcastIngest => "Broadcast contribution / ingest",
            Self::OttVod => "Standard OTT video-on-demand",
            Self::UhdHdr => "Ultra-HD / HDR content delivery",
            Self::Preview => "Lightweight preview / trailer",
        }
    }

    /// Whether this profile targets live delivery.
    #[must_use]
    pub fn is_live(&self) -> bool {
        matches!(self, Self::LowLatency | Self::BroadcastIngest)
    }

    /// Whether this profile targets on-demand delivery.
    #[must_use]
    pub fn is_vod(&self) -> bool {
        !self.is_live()
    }
}

impl fmt::Display for PackagingProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.label())
    }
}

// ---------------------------------------------------------------------------
// OptimizedSettings
// ---------------------------------------------------------------------------

/// Optimised packaging parameters derived from a [`PackagingProfile`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptimizedSettings {
    /// Target segment duration in milliseconds.
    pub segment_duration_ms: u32,
    /// Target chunk (partial segment) duration in milliseconds.
    /// Zero means chunked transfer is disabled.
    pub chunk_duration_ms: u32,
    /// Whether segments must be GOP-aligned (start with a keyframe).
    pub gop_align: bool,
    /// Whether output must conform to CMAF (Common Media Application Format).
    pub use_cmaf: bool,
    /// Whether to generate a trick-play (I-frame only) rendition.
    pub enable_trick_play: bool,
    /// Optional trick-play configuration.
    pub trick_play: Option<TrickPlayConfig>,
    /// Maximum recommended bitrate for the top rung (bps), 0 = unlimited.
    pub max_bitrate_bps: u64,
    /// Whether to use byte-range addressing (single file) instead of
    /// discrete segment files.
    pub byte_range_addressing: bool,
    /// Recommended GOP size in frames (0 = auto).
    pub gop_size_frames: u32,
    /// Whether to enable low-latency signalling in manifests.
    pub low_latency_signalling: bool,
    /// Target start-up latency in milliseconds (for live profiles).
    pub target_latency_ms: u32,
    /// Number of segments to keep in the live sliding window (0 = VOD / unlimited).
    pub live_window_segments: u32,
}

impl Default for OptimizedSettings {
    fn default() -> Self {
        Self {
            segment_duration_ms: 6_000,
            chunk_duration_ms: 0,
            gop_align: true,
            use_cmaf: true,
            enable_trick_play: false,
            trick_play: None,
            max_bitrate_bps: 0,
            byte_range_addressing: false,
            gop_size_frames: 0,
            low_latency_signalling: false,
            target_latency_ms: 0,
            live_window_segments: 0,
        }
    }
}

impl OptimizedSettings {
    /// Compute the recommended GOP duration in milliseconds.
    ///
    /// This is equal to the segment duration for GOP-aligned profiles,
    /// or zero if GOP alignment is disabled.
    #[must_use]
    pub fn gop_duration_ms(&self) -> u32 {
        if self.gop_align {
            self.segment_duration_ms
        } else {
            0
        }
    }

    /// Compute the number of chunks per segment.
    ///
    /// Returns 1 if chunked transfer is disabled.
    #[must_use]
    pub fn chunks_per_segment(&self) -> u32 {
        (self.segment_duration_ms + self.chunk_duration_ms.saturating_sub(1))
            .checked_div(self.chunk_duration_ms)
            .unwrap_or(1)
    }

    /// Whether this configuration uses chunked transfer encoding.
    #[must_use]
    pub fn is_chunked(&self) -> bool {
        self.chunk_duration_ms > 0
    }

    /// Estimate the segment size in bytes for a given average bitrate (bps).
    #[must_use]
    pub fn estimated_segment_bytes(&self, avg_bitrate_bps: u64) -> u64 {
        let duration_s = self.segment_duration_ms as u64;
        (avg_bitrate_bps * duration_s) / 8_000
    }
}

// ---------------------------------------------------------------------------
// TrickPlayConfig
// ---------------------------------------------------------------------------

/// Configuration for a trick-play (I-frame only) rendition.
///
/// Trick-play renditions are used for thumbnail scrubbing and fast-forward
/// preview in player UIs.  They contain only key-frames at a reduced
/// resolution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrickPlayConfig {
    /// Temporal sub-sampling factor.
    ///
    /// A factor of 10 means one trick-play frame for every 10 source frames.
    pub factor: u8,
    /// Trick-play rendition width in pixels.
    pub width: u32,
    /// Trick-play rendition height in pixels.
    pub height: u32,
    /// Target bitrate for the trick-play rendition (bps).
    pub bitrate_bps: u32,
    /// Codec fourcc for the trick-play rendition.
    pub codec: TrickPlayCodec,
}

impl TrickPlayConfig {
    /// Create a new trick-play configuration.
    #[must_use]
    pub fn new(factor: u8, width: u32, height: u32) -> Self {
        Self {
            factor,
            width,
            height,
            bitrate_bps: 200_000,
            codec: TrickPlayCodec::Av1,
        }
    }

    /// Set the target bitrate.
    #[must_use]
    pub fn with_bitrate(mut self, bps: u32) -> Self {
        self.bitrate_bps = bps;
        self
    }

    /// Set the codec.
    #[must_use]
    pub fn with_codec(mut self, codec: TrickPlayCodec) -> Self {
        self.codec = codec;
        self
    }

    /// Compute the output frame rate given a source frame rate.
    #[must_use]
    pub fn output_fps(&self, source_fps: f64) -> f64 {
        if self.factor == 0 {
            return source_fps;
        }
        source_fps / f64::from(self.factor)
    }
}

/// Codec choices for trick-play renditions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrickPlayCodec {
    /// AV1 (patent-free, excellent compression for stills).
    Av1,
    /// VP9 (patent-free, wide decoder support).
    Vp9,
    /// JPEG (for HLS I-frame playlists with JPEG thumbnails).
    Jpeg,
}

impl fmt::Display for TrickPlayCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Av1 => write!(f, "av1"),
            Self::Vp9 => write!(f, "vp9"),
            Self::Jpeg => write!(f, "jpeg"),
        }
    }
}

// ---------------------------------------------------------------------------
// PackagingReport
// ---------------------------------------------------------------------------

/// A summary report of a packaging job.
#[derive(Debug, Clone, PartialEq)]
pub struct PackagingReport {
    /// Total number of segments produced.
    pub total_segments: u64,
    /// Average segment size in bytes.
    pub avg_segment_bytes: u64,
    /// Total output size in bytes.
    pub total_bytes: u64,
    /// Estimated CDN egress cost in USD (at `cost_per_gb` rate).
    pub cost_estimate_usd: f64,
    /// Number of representations (bitrate rungs) packaged.
    pub representation_count: u32,
    /// Duration of the source content in seconds.
    pub source_duration_s: f64,
    /// Whether trick-play was generated.
    pub has_trick_play: bool,
}

impl PackagingReport {
    /// Create a new report.
    #[must_use]
    pub fn new(total_segments: u64, avg_segment_bytes: u64, total_bytes: u64) -> Self {
        Self {
            total_segments,
            avg_segment_bytes,
            total_bytes,
            cost_estimate_usd: 0.0,
            representation_count: 1,
            source_duration_s: 0.0,
            has_trick_play: false,
        }
    }

    /// Compute the CDN cost estimate at the given rate per gigabyte.
    #[must_use]
    pub fn with_cost_per_gb(mut self, cost_per_gb: f64) -> Self {
        let gb = self.total_bytes as f64 / 1_073_741_824.0;
        self.cost_estimate_usd = gb * cost_per_gb;
        self
    }

    /// Set the representation count.
    #[must_use]
    pub fn with_representations(mut self, count: u32) -> Self {
        self.representation_count = count;
        self
    }

    /// Set the source duration.
    #[must_use]
    pub fn with_source_duration(mut self, seconds: f64) -> Self {
        self.source_duration_s = seconds;
        self
    }

    /// Set whether trick-play was generated.
    #[must_use]
    pub fn with_trick_play(mut self, enabled: bool) -> Self {
        self.has_trick_play = enabled;
        self
    }

    /// Compute the average bitrate in bits per second.
    ///
    /// Returns 0 if source duration is zero.
    #[must_use]
    pub fn average_bitrate_bps(&self) -> u64 {
        if self.source_duration_s <= 0.0 {
            return 0;
        }
        ((self.total_bytes as f64 * 8.0) / self.source_duration_s) as u64
    }

    /// Compute the storage efficiency ratio (bytes per second of content).
    #[must_use]
    pub fn bytes_per_second(&self) -> f64 {
        if self.source_duration_s <= 0.0 {
            return 0.0;
        }
        self.total_bytes as f64 / self.source_duration_s
    }

    /// Generate a human-readable summary.
    #[must_use]
    pub fn summary(&self) -> String {
        let total_mb = self.total_bytes as f64 / 1_048_576.0;
        let avg_kb = self.avg_segment_bytes as f64 / 1_024.0;
        format!(
            "Segments: {}, Avg: {avg_kb:.1} KB, Total: {total_mb:.2} MB, \
             Representations: {}, Cost: ${:.4}",
            self.total_segments, self.representation_count, self.cost_estimate_usd,
        )
    }
}

// ---------------------------------------------------------------------------
// optimize()
// ---------------------------------------------------------------------------

/// Generate optimised packaging settings for the given profile.
///
/// Each profile encodes industry best-practice defaults for its target
/// delivery scenario.
#[must_use]
pub fn optimize(profile: &PackagingProfile) -> OptimizedSettings {
    match profile {
        PackagingProfile::LowLatency => OptimizedSettings {
            segment_duration_ms: 2_000,
            chunk_duration_ms: 200,
            gop_align: true,
            use_cmaf: true,
            enable_trick_play: false,
            trick_play: None,
            max_bitrate_bps: 8_000_000,
            byte_range_addressing: true,
            gop_size_frames: 48,
            low_latency_signalling: true,
            target_latency_ms: 3_000,
            live_window_segments: 30,
        },

        PackagingProfile::BroadcastIngest => OptimizedSettings {
            segment_duration_ms: 6_000,
            chunk_duration_ms: 1_000,
            gop_align: true,
            use_cmaf: true,
            enable_trick_play: false,
            trick_play: None,
            max_bitrate_bps: 25_000_000,
            byte_range_addressing: false,
            gop_size_frames: 150,
            low_latency_signalling: false,
            target_latency_ms: 0,
            live_window_segments: 60,
        },

        PackagingProfile::OttVod => OptimizedSettings {
            segment_duration_ms: 6_000,
            chunk_duration_ms: 0,
            gop_align: true,
            use_cmaf: true,
            enable_trick_play: true,
            trick_play: Some(TrickPlayConfig::new(10, 320, 180)),
            max_bitrate_bps: 12_000_000,
            byte_range_addressing: false,
            gop_size_frames: 144,
            low_latency_signalling: false,
            target_latency_ms: 0,
            live_window_segments: 0,
        },

        PackagingProfile::UhdHdr => OptimizedSettings {
            segment_duration_ms: 4_000,
            chunk_duration_ms: 500,
            gop_align: true,
            use_cmaf: true,
            enable_trick_play: true,
            trick_play: Some(TrickPlayConfig::new(20, 640, 360)),
            max_bitrate_bps: 40_000_000,
            byte_range_addressing: false,
            gop_size_frames: 96,
            low_latency_signalling: false,
            target_latency_ms: 0,
            live_window_segments: 0,
        },

        PackagingProfile::Preview => OptimizedSettings {
            segment_duration_ms: 2_000,
            chunk_duration_ms: 0,
            gop_align: false,
            use_cmaf: false,
            enable_trick_play: false,
            trick_play: None,
            max_bitrate_bps: 2_000_000,
            byte_range_addressing: false,
            gop_size_frames: 48,
            low_latency_signalling: false,
            target_latency_ms: 0,
            live_window_segments: 0,
        },
    }
}

// ---------------------------------------------------------------------------
// estimate_packaging
// ---------------------------------------------------------------------------

/// Estimate packaging output metrics without actually running the packager.
///
/// Given a profile, source duration, and average bitrate, this produces
/// an approximate [`PackagingReport`].
#[must_use]
pub fn estimate_packaging(
    profile: &PackagingProfile,
    source_duration_s: f64,
    avg_bitrate_bps: u64,
    representation_count: u32,
    cost_per_gb: f64,
) -> PackagingReport {
    let settings = optimize(profile);
    let segment_bytes = settings.estimated_segment_bytes(avg_bitrate_bps);
    let seg_dur_s = settings.segment_duration_ms as f64 / 1_000.0;
    let total_segments_per_rep = if seg_dur_s > 0.0 {
        (source_duration_s / seg_dur_s).ceil() as u64
    } else {
        0
    };
    let total_segments = total_segments_per_rep * representation_count as u64;
    let total_bytes = segment_bytes * total_segments;
    let avg_segment_bytes = total_bytes.checked_div(total_segments).unwrap_or(0);

    PackagingReport::new(total_segments, avg_segment_bytes, total_bytes)
        .with_cost_per_gb(cost_per_gb)
        .with_representations(representation_count)
        .with_source_duration(source_duration_s)
        .with_trick_play(settings.enable_trick_play)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- PackagingProfile ---------------------------------------------------

    #[test]
    fn test_profile_all_count() {
        assert_eq!(PackagingProfile::all().len(), 5);
    }

    #[test]
    fn test_profile_labels_unique() {
        let labels: Vec<&str> = PackagingProfile::all().iter().map(|p| p.label()).collect();
        let mut unique = labels.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(labels.len(), unique.len());
    }

    #[test]
    fn test_profile_display() {
        assert_eq!(PackagingProfile::LowLatency.to_string(), "low-latency");
        assert_eq!(PackagingProfile::OttVod.to_string(), "ott-vod");
    }

    #[test]
    fn test_profile_is_live() {
        assert!(PackagingProfile::LowLatency.is_live());
        assert!(PackagingProfile::BroadcastIngest.is_live());
        assert!(!PackagingProfile::OttVod.is_live());
    }

    #[test]
    fn test_profile_is_vod() {
        assert!(PackagingProfile::OttVod.is_vod());
        assert!(PackagingProfile::Preview.is_vod());
        assert!(!PackagingProfile::LowLatency.is_vod());
    }

    #[test]
    fn test_profile_description_non_empty() {
        for profile in PackagingProfile::all() {
            assert!(!profile.description().is_empty());
        }
    }

    // --- optimize() ---------------------------------------------------------

    #[test]
    fn test_optimize_low_latency() {
        let s = optimize(&PackagingProfile::LowLatency);
        assert_eq!(s.segment_duration_ms, 2_000);
        assert_eq!(s.chunk_duration_ms, 200);
        assert!(s.use_cmaf);
        assert!(s.low_latency_signalling);
        assert!(!s.enable_trick_play);
    }

    #[test]
    fn test_optimize_broadcast_ingest() {
        let s = optimize(&PackagingProfile::BroadcastIngest);
        assert_eq!(s.segment_duration_ms, 6_000);
        assert_eq!(s.chunk_duration_ms, 1_000);
        assert!(s.gop_align);
    }

    #[test]
    fn test_optimize_ott_vod() {
        let s = optimize(&PackagingProfile::OttVod);
        assert_eq!(s.segment_duration_ms, 6_000);
        assert!(s.use_cmaf);
        assert!(s.enable_trick_play);
        assert!(s.trick_play.is_some());
    }

    #[test]
    fn test_optimize_uhd_hdr() {
        let s = optimize(&PackagingProfile::UhdHdr);
        assert_eq!(s.segment_duration_ms, 4_000);
        assert!(s.enable_trick_play);
        let tp = s.trick_play.as_ref().expect("trick_play should be set");
        assert_eq!(tp.width, 640);
        assert_eq!(tp.height, 360);
    }

    #[test]
    fn test_optimize_preview() {
        let s = optimize(&PackagingProfile::Preview);
        assert_eq!(s.segment_duration_ms, 2_000);
        assert!(!s.use_cmaf);
        assert!(!s.enable_trick_play);
        assert!(!s.gop_align);
    }

    // --- OptimizedSettings --------------------------------------------------

    #[test]
    fn test_settings_gop_duration_ms() {
        let s = optimize(&PackagingProfile::OttVod);
        assert_eq!(s.gop_duration_ms(), 6_000);
    }

    #[test]
    fn test_settings_gop_duration_no_align() {
        let s = optimize(&PackagingProfile::Preview);
        assert_eq!(s.gop_duration_ms(), 0);
    }

    #[test]
    fn test_settings_chunks_per_segment_chunked() {
        let s = optimize(&PackagingProfile::LowLatency);
        // 2000 / 200 = 10
        assert_eq!(s.chunks_per_segment(), 10);
    }

    #[test]
    fn test_settings_chunks_per_segment_not_chunked() {
        let s = optimize(&PackagingProfile::OttVod);
        assert_eq!(s.chunks_per_segment(), 1);
    }

    #[test]
    fn test_settings_is_chunked() {
        assert!(optimize(&PackagingProfile::LowLatency).is_chunked());
        assert!(!optimize(&PackagingProfile::OttVod).is_chunked());
    }

    #[test]
    fn test_settings_estimated_segment_bytes() {
        let s = optimize(&PackagingProfile::OttVod);
        // 6_000_000 bps * 6_000 ms / 8_000 = 4_500_000 bytes
        let est = s.estimated_segment_bytes(6_000_000);
        assert_eq!(est, 4_500_000);
    }

    #[test]
    fn test_settings_default() {
        let s = OptimizedSettings::default();
        assert_eq!(s.segment_duration_ms, 6_000);
        assert!(!s.enable_trick_play);
    }

    // --- TrickPlayConfig ----------------------------------------------------

    #[test]
    fn test_trick_play_new() {
        let tp = TrickPlayConfig::new(10, 320, 180);
        assert_eq!(tp.factor, 10);
        assert_eq!(tp.width, 320);
        assert_eq!(tp.height, 180);
        assert_eq!(tp.bitrate_bps, 200_000);
    }

    #[test]
    fn test_trick_play_with_bitrate() {
        let tp = TrickPlayConfig::new(10, 320, 180).with_bitrate(500_000);
        assert_eq!(tp.bitrate_bps, 500_000);
    }

    #[test]
    fn test_trick_play_with_codec() {
        let tp = TrickPlayConfig::new(10, 320, 180).with_codec(TrickPlayCodec::Vp9);
        assert_eq!(tp.codec, TrickPlayCodec::Vp9);
    }

    #[test]
    fn test_trick_play_output_fps() {
        let tp = TrickPlayConfig::new(10, 320, 180);
        let fps = tp.output_fps(30.0);
        assert!((fps - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_trick_play_output_fps_zero_factor() {
        let tp = TrickPlayConfig::new(0, 320, 180);
        let fps = tp.output_fps(24.0);
        assert!((fps - 24.0).abs() < 0.001);
    }

    #[test]
    fn test_trick_play_codec_display() {
        assert_eq!(TrickPlayCodec::Av1.to_string(), "av1");
        assert_eq!(TrickPlayCodec::Vp9.to_string(), "vp9");
        assert_eq!(TrickPlayCodec::Jpeg.to_string(), "jpeg");
    }

    // --- PackagingReport ----------------------------------------------------

    #[test]
    fn test_report_new() {
        let r = PackagingReport::new(100, 750_000, 75_000_000);
        assert_eq!(r.total_segments, 100);
        assert_eq!(r.avg_segment_bytes, 750_000);
        assert_eq!(r.total_bytes, 75_000_000);
    }

    #[test]
    fn test_report_cost_estimate() {
        let r = PackagingReport::new(100, 1_073_741, 1_073_741_824) // 1 GB
            .with_cost_per_gb(0.08);
        assert!((r.cost_estimate_usd - 0.08).abs() < 0.001);
    }

    #[test]
    fn test_report_average_bitrate() {
        // 10 MB over 10 seconds = 8_000_000 bps
        let r = PackagingReport::new(10, 1_000_000, 10_000_000).with_source_duration(10.0);
        assert_eq!(r.average_bitrate_bps(), 8_000_000);
    }

    #[test]
    fn test_report_average_bitrate_zero_duration() {
        let r = PackagingReport::new(10, 1_000_000, 10_000_000);
        assert_eq!(r.average_bitrate_bps(), 0);
    }

    #[test]
    fn test_report_bytes_per_second() {
        let r = PackagingReport::new(10, 1_000_000, 10_000_000).with_source_duration(10.0);
        assert!((r.bytes_per_second() - 1_000_000.0).abs() < 0.1);
    }

    #[test]
    fn test_report_summary_non_empty() {
        let r = PackagingReport::new(50, 500_000, 25_000_000);
        let s = r.summary();
        assert!(!s.is_empty());
        assert!(s.contains("Segments: 50"));
    }

    // --- estimate_packaging -------------------------------------------------

    #[test]
    fn test_estimate_packaging_ott_vod() {
        let r = estimate_packaging(
            &PackagingProfile::OttVod,
            60.0,      // 1 minute
            6_000_000, // 6 Mbps
            3,         // 3 representations
            0.08,      // $0.08/GB
        );
        assert!(r.total_segments > 0);
        assert!(r.total_bytes > 0);
        assert!(r.has_trick_play);
        assert_eq!(r.representation_count, 3);
    }

    #[test]
    fn test_estimate_packaging_preview() {
        let r = estimate_packaging(&PackagingProfile::Preview, 30.0, 2_000_000, 1, 0.0);
        assert!(!r.has_trick_play);
        assert_eq!(r.representation_count, 1);
    }

    #[test]
    fn test_estimate_packaging_zero_duration() {
        let r = estimate_packaging(&PackagingProfile::OttVod, 0.0, 6_000_000, 1, 0.08);
        assert_eq!(r.total_segments, 0);
        assert_eq!(r.total_bytes, 0);
    }
}
