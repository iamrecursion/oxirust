//! Filmstrip-style thumbnail sprite generation for video scrubbing.
//!
//! Generates sprite sheets (filmstrips) from video frames at regular intervals,
//! along with a VTT-format metadata file mapping timecodes to sprite regions.

#![allow(dead_code)]

use std::collections::HashMap;
use std::time::Duration;

/// Configuration for thumbnail strip generation.
#[derive(Debug, Clone)]
pub struct ThumbnailStripConfig {
    /// Width of each individual thumbnail in pixels.
    pub thumb_width: u32,
    /// Height of each individual thumbnail in pixels.
    pub thumb_height: u32,
    /// Number of columns in the sprite sheet.
    pub columns: u32,
    /// Number of rows in the sprite sheet.
    pub rows: u32,
    /// Interval between thumbnails.
    pub interval: Duration,
    /// JPEG quality for the sprite (1-100).
    pub jpeg_quality: u8,
    /// Whether to generate WebP format instead of JPEG.
    pub use_webp: bool,
    /// Maximum number of sprite sheets to generate.
    pub max_sheets: u32,
}

impl Default for ThumbnailStripConfig {
    fn default() -> Self {
        Self {
            thumb_width: 160,
            thumb_height: 90,
            columns: 10,
            rows: 10,
            interval: Duration::from_secs(5),
            jpeg_quality: 75,
            use_webp: false,
            max_sheets: 100,
        }
    }
}

impl ThumbnailStripConfig {
    /// Total thumbnails per sprite sheet.
    pub fn thumbs_per_sheet(&self) -> u32 {
        self.columns * self.rows
    }

    /// Width of a single sprite sheet in pixels.
    pub fn sheet_width(&self) -> u32 {
        self.thumb_width * self.columns
    }

    /// Height of a single sprite sheet in pixels.
    pub fn sheet_height(&self) -> u32 {
        self.thumb_height * self.rows
    }

    /// Total pixels per sheet.
    pub fn sheet_pixels(&self) -> u64 {
        self.sheet_width() as u64 * self.sheet_height() as u64
    }

    /// File extension for the sprite format.
    pub fn file_extension(&self) -> &str {
        if self.use_webp {
            "webp"
        } else {
            "jpg"
        }
    }

    /// MIME type for the sprite format.
    pub fn mime_type(&self) -> &str {
        if self.use_webp {
            "image/webp"
        } else {
            "image/jpeg"
        }
    }
}

/// A rectangular region within a sprite sheet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpriteRegion {
    /// X offset in pixels from the left edge.
    pub x: u32,
    /// Y offset in pixels from the top edge.
    pub y: u32,
    /// Width of the thumbnail.
    pub width: u32,
    /// Height of the thumbnail.
    pub height: u32,
}

impl SpriteRegion {
    /// Formats the region as a CSS-style fragment identifier.
    /// e.g. `#xywh=320,180,160,90`
    pub fn to_fragment(&self) -> String {
        format!("#xywh={},{},{},{}", self.x, self.y, self.width, self.height)
    }

    /// Formats as a spatial media fragment (W3C).
    pub fn to_media_fragment(&self) -> String {
        format!(
            "xywh=pixel:{},{},{},{}",
            self.x, self.y, self.width, self.height
        )
    }
}

/// A single thumbnail entry in the strip.
#[derive(Debug, Clone)]
pub struct ThumbnailEntry {
    /// Start time of the segment this thumbnail represents.
    pub start_time: Duration,
    /// End time of the segment.
    pub end_time: Duration,
    /// Index of the sprite sheet (0-based).
    pub sheet_index: u32,
    /// Region within the sprite sheet.
    pub region: SpriteRegion,
    /// Sequential index (0-based).
    pub index: u32,
}

impl ThumbnailEntry {
    /// Formats the timecode range in WebVTT format.
    pub fn vtt_time_range(&self) -> String {
        format!(
            "{} --> {}",
            format_vtt_time(self.start_time),
            format_vtt_time(self.end_time)
        )
    }
}

/// Formats a Duration as a WebVTT timestamp (HH:MM:SS.mmm).
fn format_vtt_time(d: Duration) -> String {
    let total_secs = d.as_secs();
    let hours = total_secs / 3600;
    let minutes = (total_secs % 3600) / 60;
    let seconds = total_secs % 60;
    let millis = d.subsec_millis();
    format!("{:02}:{:02}:{:02}.{:03}", hours, minutes, seconds, millis)
}

/// Metadata for a generated sprite sheet.
#[derive(Debug, Clone)]
pub struct SpriteSheet {
    /// Sheet index (0-based).
    pub index: u32,
    /// Filename of the sprite image.
    pub filename: String,
    /// Total width in pixels.
    pub width: u32,
    /// Total height in pixels.
    pub height: u32,
    /// Number of thumbnails in this sheet.
    pub thumb_count: u32,
    /// Estimated file size in bytes.
    pub estimated_size: u64,
}

/// A thumbnail strip plan (layout + metadata, without actual image data).
pub struct ThumbnailStrip {
    /// Configuration used.
    config: ThumbnailStripConfig,
    /// Video duration.
    video_duration: Duration,
    /// Total number of thumbnails.
    total_thumbs: u32,
    /// Thumbnail entries.
    entries: Vec<ThumbnailEntry>,
    /// Sprite sheet metadata.
    sheets: Vec<SpriteSheet>,
    /// Custom metadata.
    metadata: HashMap<String, String>,
}

impl ThumbnailStrip {
    /// Plans a thumbnail strip for a video of the given duration.
    pub fn plan(config: ThumbnailStripConfig, video_duration: Duration) -> Self {
        let interval_secs = config.interval.as_secs_f64().max(0.1);
        let total_thumbs = (video_duration.as_secs_f64() / interval_secs).ceil() as u32;
        let total_thumbs = total_thumbs.max(1);

        let thumbs_per_sheet = config.thumbs_per_sheet();
        let num_sheets = (total_thumbs + thumbs_per_sheet - 1) / thumbs_per_sheet;
        let num_sheets = num_sheets.min(config.max_sheets);

        let max_thumbs = num_sheets * thumbs_per_sheet;
        let total_thumbs = total_thumbs.min(max_thumbs);

        let mut entries = Vec::with_capacity(total_thumbs as usize);
        let mut sheets = Vec::new();

        for sheet_idx in 0..num_sheets {
            let start = sheet_idx * thumbs_per_sheet;
            let end = (start + thumbs_per_sheet).min(total_thumbs);
            let count = end - start;

            sheets.push(SpriteSheet {
                index: sheet_idx,
                filename: format!("sprite_{}.{}", sheet_idx, config.file_extension()),
                width: config.sheet_width(),
                height: config.sheet_height(),
                thumb_count: count,
                estimated_size: estimate_sprite_size(&config, count),
            });

            for i in start..end {
                let col = (i - start) % config.columns;
                let row = (i - start) / config.columns;
                let start_time = Duration::from_secs_f64(i as f64 * interval_secs);
                let end_time =
                    Duration::from_secs_f64((i as f64 + 1.0) * interval_secs).min(video_duration);

                entries.push(ThumbnailEntry {
                    start_time,
                    end_time,
                    sheet_index: sheet_idx,
                    region: SpriteRegion {
                        x: col * config.thumb_width,
                        y: row * config.thumb_height,
                        width: config.thumb_width,
                        height: config.thumb_height,
                    },
                    index: i,
                });
            }
        }

        Self {
            config,
            video_duration,
            total_thumbs,
            entries,
            sheets,
            metadata: HashMap::new(),
        }
    }

    /// Returns the total number of thumbnails.
    pub fn total_thumbs(&self) -> u32 {
        self.total_thumbs
    }

    /// Returns the number of sprite sheets.
    pub fn sheet_count(&self) -> u32 {
        self.sheets.len() as u32
    }

    /// Returns the sprite sheet metadata.
    pub fn sheets(&self) -> &[SpriteSheet] {
        &self.sheets
    }

    /// Returns all thumbnail entries.
    pub fn entries(&self) -> &[ThumbnailEntry] {
        &self.entries
    }

    /// Gets the thumbnail entry for a specific time.
    pub fn entry_at_time(&self, time: Duration) -> Option<&ThumbnailEntry> {
        self.entries
            .iter()
            .find(|e| time >= e.start_time && time < e.end_time)
    }

    /// Generates WebVTT content for the thumbnail strip.
    pub fn generate_vtt(&self, base_url: &str) -> String {
        let mut vtt = String::from("WEBVTT\n\n");

        for entry in &self.entries {
            let sheet = &self.sheets[entry.sheet_index as usize];
            vtt.push_str(&format!(
                "{}\n{}/{}{}\n\n",
                entry.vtt_time_range(),
                base_url,
                sheet.filename,
                entry.region.to_fragment(),
            ));
        }

        vtt
    }

    /// Estimated total file size across all sprite sheets.
    pub fn estimated_total_size(&self) -> u64 {
        self.sheets.iter().map(|s| s.estimated_size).sum()
    }

    /// Returns the configuration.
    pub fn config(&self) -> &ThumbnailStripConfig {
        &self.config
    }

    /// Returns the video duration.
    pub fn video_duration(&self) -> Duration {
        self.video_duration
    }

    /// Sets custom metadata.
    pub fn set_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
    }

    /// Gets custom metadata.
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(String::as_str)
    }
}

/// Estimates the file size of a sprite sheet.
fn estimate_sprite_size(config: &ThumbnailStripConfig, thumb_count: u32) -> u64 {
    // Rough estimate: width * height * bytes_per_pixel * compression_ratio
    let pixels_per_thumb = config.thumb_width as u64 * config.thumb_height as u64;
    let total_pixels = pixels_per_thumb * thumb_count as u64;
    // JPEG: ~0.2 bytes/pixel at quality 75, WebP: ~0.15
    let bytes_per_pixel = if config.use_webp { 0.15 } else { 0.2 };
    (total_pixels as f64 * bytes_per_pixel) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    // ThumbnailStripConfig

    #[test]
    fn test_default_config() {
        let cfg = ThumbnailStripConfig::default();
        assert_eq!(cfg.thumb_width, 160);
        assert_eq!(cfg.thumb_height, 90);
        assert_eq!(cfg.columns, 10);
        assert_eq!(cfg.rows, 10);
        assert_eq!(cfg.interval, Duration::from_secs(5));
    }

    #[test]
    fn test_config_thumbs_per_sheet() {
        let cfg = ThumbnailStripConfig::default();
        assert_eq!(cfg.thumbs_per_sheet(), 100);
    }

    #[test]
    fn test_config_sheet_dimensions() {
        let cfg = ThumbnailStripConfig::default();
        assert_eq!(cfg.sheet_width(), 1600);
        assert_eq!(cfg.sheet_height(), 900);
    }

    #[test]
    fn test_config_sheet_pixels() {
        let cfg = ThumbnailStripConfig::default();
        assert_eq!(cfg.sheet_pixels(), 1_440_000);
    }

    #[test]
    fn test_config_file_extension() {
        let cfg = ThumbnailStripConfig::default();
        assert_eq!(cfg.file_extension(), "jpg");
        let webp_cfg = ThumbnailStripConfig {
            use_webp: true,
            ..Default::default()
        };
        assert_eq!(webp_cfg.file_extension(), "webp");
    }

    #[test]
    fn test_config_mime_type() {
        let cfg = ThumbnailStripConfig::default();
        assert_eq!(cfg.mime_type(), "image/jpeg");
        let webp_cfg = ThumbnailStripConfig {
            use_webp: true,
            ..Default::default()
        };
        assert_eq!(webp_cfg.mime_type(), "image/webp");
    }

    // SpriteRegion

    #[test]
    fn test_sprite_region_fragment() {
        let r = SpriteRegion {
            x: 320,
            y: 180,
            width: 160,
            height: 90,
        };
        assert_eq!(r.to_fragment(), "#xywh=320,180,160,90");
    }

    #[test]
    fn test_sprite_region_media_fragment() {
        let r = SpriteRegion {
            x: 0,
            y: 0,
            width: 160,
            height: 90,
        };
        assert_eq!(r.to_media_fragment(), "xywh=pixel:0,0,160,90");
    }

    // format_vtt_time

    #[test]
    fn test_format_vtt_time_zero() {
        assert_eq!(format_vtt_time(Duration::ZERO), "00:00:00.000");
    }

    #[test]
    fn test_format_vtt_time_hours() {
        assert_eq!(format_vtt_time(Duration::from_secs(3661)), "01:01:01.000");
    }

    #[test]
    fn test_format_vtt_time_millis() {
        assert_eq!(format_vtt_time(Duration::from_millis(5500)), "00:00:05.500");
    }

    // ThumbnailEntry

    #[test]
    fn test_entry_vtt_time_range() {
        let entry = ThumbnailEntry {
            start_time: Duration::from_secs(10),
            end_time: Duration::from_secs(15),
            sheet_index: 0,
            region: SpriteRegion {
                x: 0,
                y: 0,
                width: 160,
                height: 90,
            },
            index: 0,
        };
        assert_eq!(entry.vtt_time_range(), "00:00:10.000 --> 00:00:15.000");
    }

    // ThumbnailStrip

    #[test]
    fn test_strip_plan_short_video() {
        let config = ThumbnailStripConfig::default();
        let strip = ThumbnailStrip::plan(config, Duration::from_secs(30));
        // 30s / 5s = 6 thumbnails, 1 sheet
        assert_eq!(strip.total_thumbs(), 6);
        assert_eq!(strip.sheet_count(), 1);
        assert_eq!(strip.entries().len(), 6);
    }

    #[test]
    fn test_strip_plan_long_video() {
        let config = ThumbnailStripConfig::default();
        let strip = ThumbnailStrip::plan(config, Duration::from_secs(600));
        // 600s / 5s = 120 thumbnails, 2 sheets (100 per sheet)
        assert_eq!(strip.total_thumbs(), 120);
        assert_eq!(strip.sheet_count(), 2);
    }

    #[test]
    fn test_strip_plan_very_short_video() {
        let config = ThumbnailStripConfig::default();
        let strip = ThumbnailStrip::plan(config, Duration::from_secs(1));
        assert!(strip.total_thumbs() >= 1);
    }

    #[test]
    fn test_strip_entry_regions() {
        let config = ThumbnailStripConfig {
            columns: 3,
            rows: 2,
            interval: Duration::from_secs(10),
            ..Default::default()
        };
        let strip = ThumbnailStrip::plan(config, Duration::from_secs(60));
        // 6 thumbnails, 1 sheet
        let entries = strip.entries();

        // First entry: top-left
        assert_eq!(entries[0].region.x, 0);
        assert_eq!(entries[0].region.y, 0);

        // Second entry: next column
        assert_eq!(entries[1].region.x, 160);
        assert_eq!(entries[1].region.y, 0);

        // Fourth entry: first column, second row
        assert_eq!(entries[3].region.x, 0);
        assert_eq!(entries[3].region.y, 90);
    }

    #[test]
    fn test_strip_entry_at_time() {
        let config = ThumbnailStripConfig::default();
        let strip = ThumbnailStrip::plan(config, Duration::from_secs(60));

        let entry = strip.entry_at_time(Duration::from_secs(7));
        assert!(entry.is_some());
        let e = entry.expect("should find entry");
        assert_eq!(e.index, 1); // 7s falls in the 5-10s range (index 1)
    }

    #[test]
    fn test_strip_entry_at_time_exact_boundary() {
        let config = ThumbnailStripConfig::default();
        let strip = ThumbnailStrip::plan(config, Duration::from_secs(60));

        let entry = strip.entry_at_time(Duration::from_secs(5));
        assert!(entry.is_some());
        assert_eq!(entry.expect("should find").index, 1);
    }

    #[test]
    fn test_strip_entry_at_time_not_found() {
        let config = ThumbnailStripConfig::default();
        let strip = ThumbnailStrip::plan(config, Duration::from_secs(10));

        // Beyond video duration
        let entry = strip.entry_at_time(Duration::from_secs(999));
        assert!(entry.is_none());
    }

    #[test]
    fn test_generate_vtt() {
        let config = ThumbnailStripConfig {
            columns: 2,
            rows: 2,
            interval: Duration::from_secs(10),
            ..Default::default()
        };
        let strip = ThumbnailStrip::plan(config, Duration::from_secs(20));
        let vtt = strip.generate_vtt("https://cdn.example.com/thumbs");

        assert!(vtt.starts_with("WEBVTT"));
        assert!(vtt.contains("00:00:00.000 --> 00:00:10.000"));
        assert!(vtt.contains("sprite_0.jpg"));
        assert!(vtt.contains("#xywh="));
    }

    #[test]
    fn test_estimated_total_size() {
        let config = ThumbnailStripConfig::default();
        let strip = ThumbnailStrip::plan(config, Duration::from_secs(60));
        assert!(strip.estimated_total_size() > 0);
    }

    #[test]
    fn test_strip_metadata() {
        let config = ThumbnailStripConfig::default();
        let mut strip = ThumbnailStrip::plan(config, Duration::from_secs(30));
        strip.set_metadata("media_id", "m-123");
        assert_eq!(strip.get_metadata("media_id"), Some("m-123"));
        assert_eq!(strip.get_metadata("missing"), None);
    }

    #[test]
    fn test_strip_video_duration() {
        let config = ThumbnailStripConfig::default();
        let strip = ThumbnailStrip::plan(config, Duration::from_secs(120));
        assert_eq!(strip.video_duration(), Duration::from_secs(120));
    }

    #[test]
    fn test_webp_sprite_estimation() {
        let config = ThumbnailStripConfig {
            use_webp: true,
            ..Default::default()
        };
        let strip_webp = ThumbnailStrip::plan(config, Duration::from_secs(30));

        let config_jpg = ThumbnailStripConfig::default();
        let strip_jpg = ThumbnailStrip::plan(config_jpg, Duration::from_secs(30));

        // WebP should have smaller estimated size
        assert!(strip_webp.estimated_total_size() < strip_jpg.estimated_total_size());
    }

    #[test]
    fn test_max_sheets_limit() {
        let config = ThumbnailStripConfig {
            interval: Duration::from_millis(10), // very short interval
            max_sheets: 2,
            ..Default::default()
        };
        let strip = ThumbnailStrip::plan(config, Duration::from_secs(3600));
        assert!(strip.sheet_count() <= 2);
    }
}
