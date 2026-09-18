// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Parallel multi-variant packaging using rayon.
//!
//! This module provides [`ParallelPackager`] which packages multiple bitrate
//! ladder rungs concurrently using rayon's work-stealing thread pool.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────┐
//! │ ParallelPackager  │
//! │                   │
//! │ .package_all()    │──▶ rayon par_iter
//! │                   │      ├── rung 0 (1080p) → segments + manifest
//! │                   │      ├── rung 1 (720p)  → segments + manifest
//! │                   │      ├── rung 2 (480p)  → segments + manifest
//! │                   │      └── rung 3 (360p)  → segments + manifest
//! │                   │
//! └──────────────────┘──▶ merge manifests
//! ```
//!
//! Each rung produces a [`VariantResult`] containing the packaged segments
//! and per-variant manifest. The final step merges these into a multivariant
//! manifest (HLS master / DASH MPD).

use crate::config::{BitrateEntry, SegmentFormat};
use crate::error::{PackagerError, PackagerResult};
use crate::isobmff_writer::{InitConfig, MediaSample};
use rayon::prelude::*;
use std::time::Duration;

// ---------------------------------------------------------------------------
// VariantSpec
// ---------------------------------------------------------------------------

/// Specification for a single bitrate ladder rung.
#[derive(Debug, Clone)]
pub struct VariantSpec {
    /// Unique identifier for this variant.
    pub id: String,
    /// Video width in pixels.
    pub width: u32,
    /// Video height in pixels.
    pub height: u32,
    /// Target video bitrate in bits per second.
    pub bitrate: u64,
    /// Codec fourcc (e.g. `"av01"`, `"vp09"`).
    pub codec: String,
    /// Segment duration.
    pub segment_duration: Duration,
    /// Segment format.
    pub segment_format: SegmentFormat,
}

impl VariantSpec {
    /// Create a new variant spec.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        width: u32,
        height: u32,
        bitrate: u64,
        codec: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            width,
            height,
            bitrate,
            codec: codec.into(),
            segment_duration: Duration::from_secs(6),
            segment_format: SegmentFormat::Fmp4,
        }
    }

    /// Set the segment duration.
    #[must_use]
    pub fn with_segment_duration(mut self, dur: Duration) -> Self {
        self.segment_duration = dur;
        self
    }

    /// Set the segment format.
    #[must_use]
    pub fn with_segment_format(mut self, format: SegmentFormat) -> Self {
        self.segment_format = format;
        self
    }

    /// Create from a [`BitrateEntry`].
    #[must_use]
    pub fn from_bitrate_entry(entry: &BitrateEntry, index: usize) -> Self {
        Self::new(
            format!("v{index}_{height}p", height = entry.height),
            entry.width,
            entry.height,
            u64::from(entry.bitrate),
            entry.codec.clone(),
        )
    }

    /// Resolution string (e.g. `"1920x1080"`).
    #[must_use]
    pub fn resolution_string(&self) -> String {
        format!("{}x{}", self.width, self.height)
    }

    /// Validate the spec.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are zero.
    pub fn validate(&self) -> PackagerResult<()> {
        if self.width == 0 || self.height == 0 {
            return Err(PackagerError::InvalidConfig(
                "Variant dimensions must be greater than zero".into(),
            ));
        }
        if self.bitrate == 0 {
            return Err(PackagerError::InvalidConfig(
                "Variant bitrate must not be zero".into(),
            ));
        }
        if self.id.is_empty() {
            return Err(PackagerError::InvalidConfig(
                "Variant ID must not be empty".into(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// VariantResult
// ---------------------------------------------------------------------------

/// Result of packaging a single variant (bitrate ladder rung).
#[derive(Debug, Clone)]
pub struct VariantResult {
    /// Variant spec that produced this result.
    pub spec: VariantSpec,
    /// Init segment bytes (ftyp + moov).
    pub init_segment: Vec<u8>,
    /// Media segments (moof + mdat) in order.
    pub media_segments: Vec<Vec<u8>>,
    /// Total number of samples processed.
    pub total_samples: usize,
    /// Total duration of all segments.
    pub total_duration: Duration,
}

impl VariantResult {
    /// Return the number of media segments.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.media_segments.len()
    }

    /// Return the total output size in bytes.
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.init_segment.len() + self.media_segments.iter().map(Vec::len).sum::<usize>()
    }

    /// Generate an HLS media playlist for this variant.
    #[must_use]
    pub fn to_hls_playlist(&self, init_uri: &str, segment_uri_prefix: &str) -> String {
        let mut out = String::new();
        out.push_str("#EXTM3U\n");
        out.push_str("#EXT-X-VERSION:7\n");
        let target = self.spec.segment_duration.as_secs().max(1);
        out.push_str(&format!("#EXT-X-TARGETDURATION:{target}\n"));
        out.push_str("#EXT-X-MEDIA-SEQUENCE:0\n");
        out.push_str(&format!("#EXT-X-MAP:URI=\"{init_uri}\"\n"));

        for (i, _seg) in self.media_segments.iter().enumerate() {
            let secs = self.spec.segment_duration.as_secs_f64();
            out.push_str(&format!("#EXTINF:{secs:.6},\n"));
            out.push_str(&format!("{segment_uri_prefix}segment_{i:05}.m4s\n"));
        }

        out.push_str("#EXT-X-ENDLIST\n");
        out
    }
}

// ---------------------------------------------------------------------------
// PackagingTask
// ---------------------------------------------------------------------------

/// A packaging task for a single variant, ready for parallel execution.
///
/// The task holds the variant spec and a closure-like data source that
/// provides samples for packaging.
#[derive(Debug, Clone)]
pub struct PackagingTask {
    /// Variant specification.
    pub spec: VariantSpec,
    /// Samples grouped by segment.
    ///
    /// Each inner `Vec` is one segment's worth of samples.
    pub segments: Vec<Vec<MediaSample>>,
    /// Timescale for the output.
    pub timescale: u32,
}

impl PackagingTask {
    /// Create a new packaging task.
    #[must_use]
    pub fn new(spec: VariantSpec, timescale: u32) -> Self {
        Self {
            spec,
            segments: Vec::new(),
            timescale,
        }
    }

    /// Add a segment's samples.
    pub fn add_segment(&mut self, samples: Vec<MediaSample>) {
        self.segments.push(samples);
    }

    /// Execute the packaging task: produce init + media segments.
    ///
    /// # Errors
    ///
    /// Returns an error if the variant spec is invalid.
    pub fn execute(&self) -> PackagerResult<VariantResult> {
        self.spec.validate()?;

        // Build fourcc from codec string
        let codec_fourcc = codec_to_fourcc(&self.spec.codec)?;

        // Generate init segment
        let init_config = InitConfig::new(
            self.spec.width,
            self.spec.height,
            self.timescale,
            codec_fourcc,
        );
        let init_segment = crate::isobmff_writer::write_init_segment(&init_config);

        // Generate media segments
        let mut media_segments = Vec::with_capacity(self.segments.len());
        let mut total_samples = 0;
        let mut decode_time: u64 = 0;

        for (seq, segment_samples) in self.segments.iter().enumerate() {
            let segment_data = crate::isobmff_writer::write_media_segment(
                (seq + 1) as u32,
                decode_time,
                segment_samples,
            );
            media_segments.push(segment_data);

            total_samples += segment_samples.len();
            let segment_ticks: u32 = segment_samples.iter().map(|s| s.duration).sum();
            decode_time += u64::from(segment_ticks);
        }

        let total_duration = self.spec.segment_duration * self.segments.len() as u32;

        Ok(VariantResult {
            spec: self.spec.clone(),
            init_segment,
            media_segments,
            total_samples,
            total_duration,
        })
    }
}

// ---------------------------------------------------------------------------
// ParallelPackager
// ---------------------------------------------------------------------------

/// Packages multiple bitrate ladder rungs concurrently using rayon.
///
/// # Example
///
/// ```
/// use oximedia_packager::parallel_packager::{ParallelPackager, PackagingTask, VariantSpec};
/// use oximedia_packager::isobmff_writer::MediaSample;
///
/// let mut packager = ParallelPackager::new();
///
/// // Add tasks for each rung of the bitrate ladder
/// let mut task_1080 = PackagingTask::new(
///     VariantSpec::new("v0_1080p", 1920, 1080, 5_000_000, "av01"),
///     90_000,
/// );
/// task_1080.add_segment(vec![MediaSample::new(vec![0u8; 100], 270_000, true)]);
/// packager.add_task(task_1080);
///
/// let mut task_720 = PackagingTask::new(
///     VariantSpec::new("v1_720p", 1280, 720, 3_000_000, "av01"),
///     90_000,
/// );
/// task_720.add_segment(vec![MediaSample::new(vec![0u8; 60], 270_000, true)]);
/// packager.add_task(task_720);
///
/// // Package all variants in parallel
/// let results = packager.package_all().expect("packaging should succeed");
/// assert_eq!(results.len(), 2);
/// ```
#[derive(Debug, Default)]
pub struct ParallelPackager {
    tasks: Vec<PackagingTask>,
}

impl ParallelPackager {
    /// Create a new parallel packager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a packaging task.
    pub fn add_task(&mut self, task: PackagingTask) {
        self.tasks.push(task);
    }

    /// Return the number of tasks.
    #[must_use]
    pub fn task_count(&self) -> usize {
        self.tasks.len()
    }

    /// Execute all tasks in parallel using rayon.
    ///
    /// Returns one [`VariantResult`] per task, in the same order.
    ///
    /// # Errors
    ///
    /// Returns the first error encountered across all tasks.
    pub fn package_all(&self) -> PackagerResult<Vec<VariantResult>> {
        let results: Vec<PackagerResult<VariantResult>> =
            self.tasks.par_iter().map(|task| task.execute()).collect();

        // Collect results, propagating the first error
        let mut successes = Vec::with_capacity(results.len());
        for result in results {
            successes.push(result?);
        }
        Ok(successes)
    }

    /// Generate a multivariant HLS playlist from the results.
    #[must_use]
    pub fn to_hls_multivariant(results: &[VariantResult], audio_group: Option<&str>) -> String {
        let mut out = String::new();
        out.push_str("#EXTM3U\n");
        out.push_str("#EXT-X-VERSION:7\n");
        out.push_str("#EXT-X-INDEPENDENT-SEGMENTS\n\n");

        for result in results {
            let mut attrs = Vec::new();
            attrs.push(format!("BANDWIDTH={}", result.spec.bitrate));
            attrs.push(format!("RESOLUTION={}", result.spec.resolution_string()));
            attrs.push(format!("CODECS=\"{}\"", result.spec.codec));

            if let Some(group) = audio_group {
                attrs.push(format!("AUDIO=\"{group}\""));
            }

            out.push_str(&format!("#EXT-X-STREAM-INF:{}\n", attrs.join(",")));
            out.push_str(&format!("{}/index.m3u8\n", result.spec.id));
        }

        out
    }

    /// Generate a minimal DASH MPD period fragment from the results.
    #[must_use]
    pub fn to_dash_period(results: &[VariantResult], timescale: u32) -> String {
        let mut xml = String::new();
        xml.push_str(r#"<Period>"#);
        xml.push_str(r#"<AdaptationSet contentType="video" mimeType="video/mp4">"#);

        for result in results {
            xml.push_str(&format!(
                r#"<Representation id="{}" bandwidth="{}" width="{}" height="{}" codecs="{}">"#,
                result.spec.id,
                result.spec.bitrate,
                result.spec.width,
                result.spec.height,
                result.spec.codec
            ));
            xml.push_str(&format!(
                r#"<SegmentTemplate timescale="{timescale}" initialization="{}/init.mp4" media="{}/segment_$Number%05d$.m4s" startNumber="0"/>"#,
                result.spec.id,
                result.spec.id
            ));
            xml.push_str("</Representation>");
        }

        xml.push_str("</AdaptationSet>");
        xml.push_str("</Period>");
        xml
    }
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

/// Convert a codec string to a 4-byte fourcc.
fn codec_to_fourcc(codec: &str) -> PackagerResult<[u8; 4]> {
    let padded = format!("{:\0<4}", codec);
    let bytes = padded.as_bytes();
    if bytes.len() < 4 {
        return Err(PackagerError::UnsupportedCodec(format!(
            "codec string too short: {codec}"
        )));
    }
    let mut fourcc = [0u8; 4];
    fourcc.copy_from_slice(&bytes[..4]);
    Ok(fourcc)
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

    // --- VariantSpec --------------------------------------------------------

    #[test]
    fn test_variant_spec_new() {
        let v = VariantSpec::new("v0_1080p", 1920, 1080, 5_000_000, "av01");
        assert_eq!(v.id, "v0_1080p");
        assert_eq!(v.width, 1920);
        assert_eq!(v.height, 1080);
        assert_eq!(v.bitrate, 5_000_000);
    }

    #[test]
    fn test_variant_spec_from_bitrate_entry() {
        let entry = BitrateEntry::new(3_000_000, 1280, 720, "av01");
        let v = VariantSpec::from_bitrate_entry(&entry, 1);
        assert_eq!(v.id, "v1_720p");
        assert_eq!(v.width, 1280);
        assert_eq!(v.height, 720);
    }

    #[test]
    fn test_variant_spec_resolution_string() {
        let v = VariantSpec::new("v0", 1920, 1080, 5_000_000, "av01");
        assert_eq!(v.resolution_string(), "1920x1080");
    }

    #[test]
    fn test_variant_spec_with_segment_duration() {
        let v = VariantSpec::new("v0", 1920, 1080, 5_000_000, "av01").with_segment_duration(dur(4));
        assert_eq!(v.segment_duration, dur(4));
    }

    #[test]
    fn test_variant_spec_validate_ok() {
        let v = VariantSpec::new("v0", 1920, 1080, 5_000_000, "av01");
        assert!(v.validate().is_ok());
    }

    #[test]
    fn test_variant_spec_validate_zero_width() {
        let v = VariantSpec::new("v0", 0, 1080, 5_000_000, "av01");
        assert!(v.validate().is_err());
    }

    #[test]
    fn test_variant_spec_validate_zero_bitrate() {
        let v = VariantSpec::new("v0", 1920, 1080, 0, "av01");
        assert!(v.validate().is_err());
    }

    #[test]
    fn test_variant_spec_validate_empty_id() {
        let v = VariantSpec::new("", 1920, 1080, 5_000_000, "av01");
        assert!(v.validate().is_err());
    }

    // --- codec_to_fourcc ----------------------------------------------------

    #[test]
    fn test_codec_to_fourcc_av01() {
        let f = codec_to_fourcc("av01").expect("should succeed");
        assert_eq!(&f, b"av01");
    }

    #[test]
    fn test_codec_to_fourcc_vp09() {
        let f = codec_to_fourcc("vp09").expect("should succeed");
        assert_eq!(&f, b"vp09");
    }

    #[test]
    fn test_codec_to_fourcc_short_padded() {
        let f = codec_to_fourcc("vp8").expect("should succeed");
        assert_eq!(&f, b"vp8\0");
    }

    // --- PackagingTask ------------------------------------------------------

    #[test]
    fn test_packaging_task_new() {
        let spec = VariantSpec::new("v0", 1920, 1080, 5_000_000, "av01");
        let task = PackagingTask::new(spec, 90_000);
        assert_eq!(task.timescale, 90_000);
        assert!(task.segments.is_empty());
    }

    #[test]
    fn test_packaging_task_add_segment() {
        let spec = VariantSpec::new("v0", 1920, 1080, 5_000_000, "av01");
        let mut task = PackagingTask::new(spec, 90_000);
        task.add_segment(vec![MediaSample::new(vec![0u8; 100], 270_000, true)]);
        assert_eq!(task.segments.len(), 1);
    }

    #[test]
    fn test_packaging_task_execute() {
        let spec = VariantSpec::new("v0", 1920, 1080, 5_000_000, "av01");
        let mut task = PackagingTask::new(spec, 90_000);
        task.add_segment(vec![
            MediaSample::new(vec![0u8; 100], 45_000, true),
            MediaSample::new(vec![0u8; 50], 45_000, false),
        ]);
        task.add_segment(vec![MediaSample::new(vec![0u8; 80], 90_000, true)]);

        let result = task.execute().expect("should succeed");
        assert_eq!(result.segment_count(), 2);
        assert_eq!(result.total_samples, 3);
        assert!(!result.init_segment.is_empty());
        assert!(result.total_bytes() > 0);
    }

    // --- VariantResult ------------------------------------------------------

    #[test]
    fn test_variant_result_segment_count() {
        let result = VariantResult {
            spec: VariantSpec::new("v0", 1920, 1080, 5_000_000, "av01"),
            init_segment: vec![0u8; 256],
            media_segments: vec![vec![0u8; 1000], vec![0u8; 900]],
            total_samples: 10,
            total_duration: dur(12),
        };
        assert_eq!(result.segment_count(), 2);
        assert_eq!(result.total_bytes(), 256 + 1000 + 900);
    }

    #[test]
    fn test_variant_result_to_hls_playlist() {
        let result = VariantResult {
            spec: VariantSpec::new("v0", 1920, 1080, 5_000_000, "av01"),
            init_segment: vec![0u8; 256],
            media_segments: vec![vec![0u8; 1000], vec![0u8; 900]],
            total_samples: 10,
            total_duration: dur(12),
        };
        let playlist = result.to_hls_playlist("init.mp4", "segments/");
        assert!(playlist.contains("#EXTM3U"));
        assert!(playlist.contains("#EXT-X-MAP:URI=\"init.mp4\""));
        assert!(playlist.contains("segments/segment_00000.m4s"));
        assert!(playlist.contains("segments/segment_00001.m4s"));
        assert!(playlist.contains("#EXT-X-ENDLIST"));
    }

    // --- ParallelPackager ---------------------------------------------------

    #[test]
    fn test_parallel_packager_new() {
        let p = ParallelPackager::new();
        assert_eq!(p.task_count(), 0);
    }

    #[test]
    fn test_parallel_packager_add_task() {
        let mut p = ParallelPackager::new();
        let spec = VariantSpec::new("v0", 1920, 1080, 5_000_000, "av01");
        p.add_task(PackagingTask::new(spec, 90_000));
        assert_eq!(p.task_count(), 1);
    }

    #[test]
    fn test_parallel_packager_package_all() {
        let mut p = ParallelPackager::new();

        // Add 1080p task
        let mut task1 = PackagingTask::new(
            VariantSpec::new("v0_1080p", 1920, 1080, 5_000_000, "av01"),
            90_000,
        );
        task1.add_segment(vec![MediaSample::new(vec![0u8; 100], 270_000, true)]);
        p.add_task(task1);

        // Add 720p task
        let mut task2 = PackagingTask::new(
            VariantSpec::new("v1_720p", 1280, 720, 3_000_000, "av01"),
            90_000,
        );
        task2.add_segment(vec![MediaSample::new(vec![0u8; 60], 270_000, true)]);
        p.add_task(task2);

        // Add 480p task
        let mut task3 = PackagingTask::new(
            VariantSpec::new("v2_480p", 854, 480, 1_500_000, "vp09"),
            90_000,
        );
        task3.add_segment(vec![MediaSample::new(vec![0u8; 40], 270_000, true)]);
        p.add_task(task3);

        let results = p.package_all().expect("should succeed");
        assert_eq!(results.len(), 3);

        for result in &results {
            assert_eq!(result.segment_count(), 1);
            assert!(!result.init_segment.is_empty());
        }
    }

    #[test]
    fn test_parallel_packager_package_all_error_propagated() {
        let mut p = ParallelPackager::new();

        // Invalid task: zero width
        let task = PackagingTask::new(VariantSpec::new("bad", 0, 1080, 5_000_000, "av01"), 90_000);
        p.add_task(task);

        assert!(p.package_all().is_err());
    }

    // --- HLS multivariant ---------------------------------------------------

    #[test]
    fn test_hls_multivariant() {
        let results = vec![
            VariantResult {
                spec: VariantSpec::new("v0_1080p", 1920, 1080, 5_000_000, "av01"),
                init_segment: Vec::new(),
                media_segments: Vec::new(),
                total_samples: 0,
                total_duration: Duration::ZERO,
            },
            VariantResult {
                spec: VariantSpec::new("v1_720p", 1280, 720, 3_000_000, "av01"),
                init_segment: Vec::new(),
                media_segments: Vec::new(),
                total_samples: 0,
                total_duration: Duration::ZERO,
            },
        ];

        let manifest = ParallelPackager::to_hls_multivariant(&results, Some("audio-group"));
        assert!(manifest.contains("#EXTM3U"));
        assert!(manifest.contains("#EXT-X-INDEPENDENT-SEGMENTS"));
        assert!(manifest.contains("BANDWIDTH=5000000"));
        assert!(manifest.contains("BANDWIDTH=3000000"));
        assert!(manifest.contains("RESOLUTION=1920x1080"));
        assert!(manifest.contains("RESOLUTION=1280x720"));
        assert!(manifest.contains("AUDIO=\"audio-group\""));
        assert!(manifest.contains("v0_1080p/index.m3u8"));
        assert!(manifest.contains("v1_720p/index.m3u8"));
    }

    #[test]
    fn test_hls_multivariant_no_audio_group() {
        let results = vec![VariantResult {
            spec: VariantSpec::new("v0", 1920, 1080, 5_000_000, "av01"),
            init_segment: Vec::new(),
            media_segments: Vec::new(),
            total_samples: 0,
            total_duration: Duration::ZERO,
        }];

        let manifest = ParallelPackager::to_hls_multivariant(&results, None);
        assert!(!manifest.contains("AUDIO="));
    }

    // --- DASH period --------------------------------------------------------

    #[test]
    fn test_dash_period() {
        let results = vec![
            VariantResult {
                spec: VariantSpec::new("v0_1080p", 1920, 1080, 5_000_000, "av01"),
                init_segment: Vec::new(),
                media_segments: Vec::new(),
                total_samples: 0,
                total_duration: Duration::ZERO,
            },
            VariantResult {
                spec: VariantSpec::new("v1_720p", 1280, 720, 3_000_000, "av01"),
                init_segment: Vec::new(),
                media_segments: Vec::new(),
                total_samples: 0,
                total_duration: Duration::ZERO,
            },
        ];

        let mpd = ParallelPackager::to_dash_period(&results, 90_000);
        assert!(mpd.contains("<Period>"));
        assert!(mpd.contains("bandwidth=\"5000000\""));
        assert!(mpd.contains("bandwidth=\"3000000\""));
        assert!(mpd.contains("width=\"1920\""));
        assert!(mpd.contains("width=\"1280\""));
        assert!(mpd.contains("timescale=\"90000\""));
    }

    // --- Empty packager -----------------------------------------------------

    #[test]
    fn test_parallel_packager_empty() {
        let p = ParallelPackager::new();
        let results = p.package_all().expect("empty package should succeed");
        assert!(results.is_empty());
    }
}
