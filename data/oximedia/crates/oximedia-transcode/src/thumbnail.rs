//! Thumbnail and preview image generation.
//!
//! This module provides configuration structures and utilities for generating
//! thumbnail images or sprite sheets from video content. Actual pixel decoding
//! is handled by the caller; this module focuses on timestamp selection, sizing,
//! and nearest-neighbour scaling.

#![allow(dead_code)]

/// Output format for generated thumbnails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThumbnailFormat {
    /// JPEG (lossy, small file size).
    Jpeg,
    /// PNG (lossless).
    Png,
    /// WebP (modern, good compression).
    Webp,
}

impl ThumbnailFormat {
    /// Returns the conventional file extension for this format.
    #[must_use]
    pub fn extension(&self) -> &'static str {
        match self {
            ThumbnailFormat::Jpeg => "jpg",
            ThumbnailFormat::Png => "png",
            ThumbnailFormat::Webp => "webp",
        }
    }

    /// Returns the MIME type for this format.
    #[must_use]
    pub fn mime_type(&self) -> &'static str {
        match self {
            ThumbnailFormat::Jpeg => "image/jpeg",
            ThumbnailFormat::Png => "image/png",
            ThumbnailFormat::Webp => "image/webp",
        }
    }
}

/// Strategy for selecting thumbnail timestamps.
#[derive(Debug, Clone)]
pub enum ThumbnailStrategy {
    /// Thumbnails at a fixed interval in milliseconds.
    FixedInterval,
    /// Thumbnails at detected scene-change points (caller must supply timestamps).
    SceneChange,
    /// Thumbnails evenly distributed across the duration.
    Uniform,
    /// Thumbnails at specific caller-supplied timestamps (in milliseconds).
    AtTimestamps(Vec<u64>),
}

/// Configuration for thumbnail generation.
#[derive(Debug, Clone)]
pub struct ThumbnailConfig {
    /// Width of each thumbnail in pixels.
    pub width: u32,
    /// Height of each thumbnail in pixels.
    pub height: u32,
    /// Output image format.
    pub format: ThumbnailFormat,
    /// Quality hint (0–100). Interpretation depends on the format.
    pub quality: u8,
    /// Number of thumbnails to generate (ignored for `AtTimestamps`).
    pub count: usize,
    /// Strategy for selecting frame timestamps.
    pub interval_strategy: ThumbnailStrategy,
}

impl ThumbnailConfig {
    /// Creates a sensible default config suitable for web use (320×180 JPEG).
    #[must_use]
    pub fn default_web() -> Self {
        Self {
            width: 320,
            height: 180,
            format: ThumbnailFormat::Jpeg,
            quality: 80,
            count: 10,
            interval_strategy: ThumbnailStrategy::Uniform,
        }
    }

    /// Creates a config appropriate for building a sprite sheet with the given thumbnail count.
    #[must_use]
    pub fn sprite_sheet(count: usize) -> Self {
        Self {
            width: 160,
            height: 90,
            format: ThumbnailFormat::Jpeg,
            quality: 70,
            count,
            interval_strategy: ThumbnailStrategy::Uniform,
        }
    }

    /// Returns `true` if the configured resolution is non-zero.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.width > 0 && self.height > 0 && self.count > 0
    }
}

/// A single generated thumbnail.
#[derive(Debug, Clone)]
pub struct Thumbnail {
    /// Timestamp in the source video (milliseconds).
    pub timestamp_ms: u64,
    /// Width of this thumbnail in pixels.
    pub width: u32,
    /// Height of this thumbnail in pixels.
    pub height: u32,
    /// Raw pixel data (RGBA, row-major).
    pub data: Vec<u8>,
}

impl Thumbnail {
    /// Creates a new thumbnail with the given parameters.
    #[must_use]
    pub fn new(timestamp_ms: u64, width: u32, height: u32, data: Vec<u8>) -> Self {
        Self {
            timestamp_ms,
            width,
            height,
            data,
        }
    }

    /// Returns the number of pixels in this thumbnail.
    #[must_use]
    pub fn pixel_count(&self) -> usize {
        (self.width * self.height) as usize
    }

    /// Returns the expected byte length for RGBA data.
    #[must_use]
    pub fn expected_byte_len(&self) -> usize {
        self.pixel_count() * 4
    }
}

/// Computes a list of timestamps (in milliseconds) at which to capture thumbnails.
///
/// # Arguments
///
/// * `duration_ms` - Total content duration in milliseconds.
/// * `strategy` - The selection strategy.
/// * `fps` - Frame rate of the source (used to snap timestamps to frame boundaries).
///   Pass `0.0` to skip frame-snapping.
#[must_use]
pub fn compute_thumbnail_timestamps(
    duration_ms: u64,
    strategy: &ThumbnailStrategy,
    fps: f64,
) -> Vec<u64> {
    if duration_ms == 0 {
        return Vec::new();
    }

    let snap = |ts: f64| -> u64 {
        if fps > 0.0 {
            let frame_ms = 1000.0 / fps;
            ((ts / frame_ms).round() * frame_ms) as u64
        } else {
            ts as u64
        }
    };

    match strategy {
        ThumbnailStrategy::AtTimestamps(ts) => {
            ts.iter().filter(|&&t| t <= duration_ms).copied().collect()
        }

        ThumbnailStrategy::Uniform => {
            // Will return config.count timestamps; here we use duration_ms to infer count
            // We default to 10 when called from the generic form without count.
            // Callers that know the count should use compute_uniform_timestamps.
            compute_uniform_timestamps(duration_ms, 10, fps)
        }

        ThumbnailStrategy::FixedInterval => {
            // Default: one thumbnail every 10 seconds
            let interval_ms = 10_000u64;
            let mut ts = Vec::new();
            let mut t = 0u64;
            while t <= duration_ms {
                ts.push(snap(t as f64));
                t += interval_ms;
            }
            ts
        }

        ThumbnailStrategy::SceneChange => {
            // Scene-change timestamps must be provided by the caller.
            // In the generic form, return empty (caller supplies via AtTimestamps).
            Vec::new()
        }
    }
}

/// Computes `count` uniformly spaced timestamps across `duration_ms`.
#[must_use]
pub fn compute_uniform_timestamps(duration_ms: u64, count: usize, fps: f64) -> Vec<u64> {
    if count == 0 || duration_ms == 0 {
        return Vec::new();
    }

    let snap = |ts: f64| -> u64 {
        if fps > 0.0 {
            let frame_ms = 1000.0 / fps;
            ((ts / frame_ms).round() * frame_ms) as u64
        } else {
            ts as u64
        }
    };

    if count == 1 {
        return vec![snap(duration_ms as f64 / 2.0)];
    }

    (0..count)
        .map(|i| {
            let t = (duration_ms as f64 * i as f64) / (count - 1) as f64;
            snap(t).min(duration_ms)
        })
        .collect()
}

/// Scales a source image buffer to the destination dimensions using nearest-neighbour sampling.
///
/// The buffers are expected to be RGBA (4 bytes per pixel), stored row-major.
///
/// Returns the scaled pixel data, or an empty `Vec` if any dimension is zero.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn scale_thumbnail(src: &[u8], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Vec<u8> {
    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
        return Vec::new();
    }

    let expected_len = (src_w * src_h * 4) as usize;
    if src.len() < expected_len {
        return Vec::new();
    }

    let mut dst = vec![0u8; (dst_w * dst_h * 4) as usize];

    for dy in 0..dst_h {
        for dx in 0..dst_w {
            // Nearest-neighbour mapping
            let sx = (f64::from(dx) * f64::from(src_w) / f64::from(dst_w)) as u32;
            let sy = (f64::from(dy) * f64::from(src_h) / f64::from(dst_h)) as u32;

            let src_idx = ((sy * src_w + sx) * 4) as usize;
            let dst_idx = ((dy * dst_w + dx) * 4) as usize;

            if src_idx + 3 < src.len() && dst_idx + 3 < dst.len() {
                dst[dst_idx] = src[src_idx];
                dst[dst_idx + 1] = src[src_idx + 1];
                dst[dst_idx + 2] = src[src_idx + 2];
                dst[dst_idx + 3] = src[src_idx + 3];
            }
        }
    }

    dst
}

// ─── Sprite sheet ─────────────────────────────────────────────────────────────

/// A sprite sheet: a single wide RGBA image that tiles multiple thumbnails
/// left-to-right, top-to-bottom.
///
/// Each cell in the sprite sheet corresponds to one thumbnail at a known
/// timestamp.  The sheet is intended for use with an accompanying VTT file
/// so that video players can display scrubber previews.
#[derive(Debug, Clone)]
pub struct SpriteSheet {
    /// Width of each individual thumbnail cell in pixels.
    pub cell_width: u32,
    /// Height of each individual thumbnail cell in pixels.
    pub cell_height: u32,
    /// Number of columns in the sheet.
    pub cols: u32,
    /// Number of rows in the sheet.
    pub rows: u32,
    /// Total width of the assembled image (`cols * cell_width`).
    pub sheet_width: u32,
    /// Total height of the assembled image (`rows * cell_height`).
    pub sheet_height: u32,
    /// Raw RGBA pixel data (row-major, top-to-bottom).
    pub data: Vec<u8>,
    /// Timestamps (ms) for each cell, in row-major order.
    pub timestamps_ms: Vec<u64>,
}

impl SpriteSheet {
    /// Assembles a sprite sheet from a collection of [`Thumbnail`]s.
    ///
    /// Thumbnails are laid out left-to-right, top-to-bottom.  All thumbnails
    /// must have the same dimensions; if they differ they are silently scaled
    /// to the first thumbnail's dimensions using nearest-neighbour sampling.
    ///
    /// Returns `None` when `thumbnails` is empty.
    #[must_use]
    pub fn from_thumbnails(thumbnails: &[Thumbnail], cols: u32) -> Option<Self> {
        if thumbnails.is_empty() || cols == 0 {
            return None;
        }

        let cell_w = thumbnails[0].width;
        let cell_h = thumbnails[0].height;
        if cell_w == 0 || cell_h == 0 {
            return None;
        }

        let count = thumbnails.len() as u32;
        let rows = (count + cols - 1) / cols; // ceiling division
        let sheet_w = cols * cell_w;
        let sheet_h = rows * cell_h;

        let mut sheet = vec![0u8; (sheet_w * sheet_h * 4) as usize];
        let mut timestamps = Vec::with_capacity(thumbnails.len());

        for (idx, thumb) in thumbnails.iter().enumerate() {
            timestamps.push(thumb.timestamp_ms);

            let col = idx as u32 % cols;
            let row = idx as u32 / cols;

            // Scale thumb to cell dimensions if needed.
            let cell_data = if thumb.width == cell_w && thumb.height == cell_h {
                thumb.data.clone()
            } else {
                scale_thumbnail(&thumb.data, thumb.width, thumb.height, cell_w, cell_h)
            };

            if cell_data.len() < (cell_w * cell_h * 4) as usize {
                continue; // skip malformed thumbnail
            }

            // Copy cell_data into the correct position in the sheet.
            let dest_x = col * cell_w;
            let dest_y = row * cell_h;

            for cy in 0..cell_h {
                let src_row_start = (cy * cell_w * 4) as usize;
                let src_row_end = src_row_start + (cell_w * 4) as usize;
                let dest_row_start = ((dest_y + cy) * sheet_w * 4 + dest_x * 4) as usize;
                let dest_row_end = dest_row_start + (cell_w * 4) as usize;

                if src_row_end <= cell_data.len() && dest_row_end <= sheet.len() {
                    sheet[dest_row_start..dest_row_end]
                        .copy_from_slice(&cell_data[src_row_start..src_row_end]);
                }
            }
        }

        Some(SpriteSheet {
            cell_width: cell_w,
            cell_height: cell_h,
            cols,
            rows,
            sheet_width: sheet_w,
            sheet_height: sheet_h,
            data: sheet,
            timestamps_ms: timestamps,
        })
    }

    /// Returns the pixel coordinate `(x, y)` of the top-left corner of the
    /// cell at `idx` within the sprite sheet.
    #[must_use]
    pub fn cell_origin(&self, idx: usize) -> (u32, u32) {
        let col = idx as u32 % self.cols;
        let row = idx as u32 / self.cols;
        (col * self.cell_width, row * self.cell_height)
    }

    /// Generates a WebVTT cue file (`.vtt`) for this sprite sheet.
    ///
    /// # Arguments
    ///
    /// * `sprite_url` – URL of the sprite sheet image referenced from the VTT.
    ///
    /// The returned string is a complete WebVTT document that can be written
    /// to disk and served alongside the sprite sheet image.
    #[must_use]
    pub fn to_vtt(&self, sprite_url: &str) -> String {
        let mut vtt = String::from("WEBVTT\n\n");

        for (idx, &ts_ms) in self.timestamps_ms.iter().enumerate() {
            let next_ts_ms = self
                .timestamps_ms
                .get(idx + 1)
                .copied()
                .unwrap_or(ts_ms + 1_000); // 1-second window for the last frame

            let (x, y) = self.cell_origin(idx);

            let start = format_vtt_time(ts_ms);
            let end = format_vtt_time(next_ts_ms);

            vtt.push_str(&format!(
                "{start} --> {end}\n{sprite_url}#xywh={x},{y},{w},{h}\n\n",
                w = self.cell_width,
                h = self.cell_height,
            ));
        }

        vtt
    }

    /// Returns the total number of cells in the sprite sheet.
    #[must_use]
    pub fn cell_count(&self) -> usize {
        self.timestamps_ms.len()
    }

    /// Returns the total byte length of the raw RGBA data.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.data.len()
    }
}

/// Formats a millisecond timestamp as a WebVTT time string (`HH:MM:SS.mmm`).
#[must_use]
pub fn format_vtt_time(ms: u64) -> String {
    let total_secs = ms / 1_000;
    let millis = ms % 1_000;
    let secs = total_secs % 60;
    let mins = (total_secs / 60) % 60;
    let hours = total_secs / 3_600;
    format!("{hours:02}:{mins:02}:{secs:02}.{millis:03}")
}

/// Configuration for generating a sprite sheet from a video.
#[derive(Debug, Clone)]
pub struct SpriteSheetConfig {
    /// Width of each thumbnail cell.
    pub cell_width: u32,
    /// Height of each thumbnail cell.
    pub cell_height: u32,
    /// Number of columns in the sprite sheet.
    pub cols: u32,
    /// Total number of thumbnails to generate.
    pub count: usize,
    /// Strategy for selecting timestamps.
    pub strategy: ThumbnailStrategy,
    /// JPEG quality (0-100, used when encoding the sprite sheet to JPEG).
    pub quality: u8,
}

impl SpriteSheetConfig {
    /// Returns a sensible default for web video players (160×90, 5 cols, 100 frames).
    #[must_use]
    pub fn default_web() -> Self {
        Self {
            cell_width: 160,
            cell_height: 90,
            cols: 5,
            count: 100,
            strategy: ThumbnailStrategy::Uniform,
            quality: 70,
        }
    }

    /// Returns a high-density config suitable for long-form content.
    #[must_use]
    pub fn high_density(count: usize) -> Self {
        Self {
            cell_width: 120,
            cell_height: 68,
            cols: 10,
            count,
            strategy: ThumbnailStrategy::Uniform,
            quality: 65,
        }
    }

    /// Returns `true` if the configuration is valid (all dimensions > 0).
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.cell_width > 0 && self.cell_height > 0 && self.cols > 0 && self.count > 0
    }

    /// Computes the timestamps for this sprite sheet given a content duration.
    #[must_use]
    pub fn compute_timestamps(&self, duration_ms: u64, fps: f64) -> Vec<u64> {
        match &self.strategy {
            ThumbnailStrategy::Uniform => compute_uniform_timestamps(duration_ms, self.count, fps),
            other => compute_thumbnail_timestamps(duration_ms, other, fps),
        }
    }
}

// ─── Smart thumbnail selection ────────────────────────────────────────────────

/// Computes the spatial variance of an RGBA thumbnail's luminance.
///
/// Higher variance indicates a more "interesting" or visually complex frame.
/// Used by [`select_smart_thumbnails`] to pick representative frames.
#[must_use]
pub fn thumbnail_variance(thumb: &Thumbnail) -> f64 {
    let pixel_count = thumb.pixel_count();
    if pixel_count == 0 || thumb.data.len() < pixel_count * 4 {
        return 0.0;
    }

    // Compute luminance using BT.709 coefficients
    let mut sum = 0.0;
    let mut sum_sq = 0.0;

    for i in 0..pixel_count {
        let offset = i * 4;
        let r = f64::from(thumb.data[offset]);
        let g = f64::from(thumb.data[offset + 1]);
        let b = f64::from(thumb.data[offset + 2]);
        let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        sum += lum;
        sum_sq += lum * lum;
    }

    let n = pixel_count as f64;
    let mean = sum / n;
    (sum_sq / n) - (mean * mean)
}

/// Selects the `count` most visually interesting thumbnails from a set,
/// ranked by spatial luminance variance.
///
/// Returns indices into the input slice, sorted by decreasing variance.
#[must_use]
pub fn select_smart_thumbnails(thumbnails: &[Thumbnail], count: usize) -> Vec<usize> {
    if thumbnails.is_empty() || count == 0 {
        return Vec::new();
    }

    let mut scored: Vec<(usize, f64)> = thumbnails
        .iter()
        .enumerate()
        .map(|(i, t)| (i, thumbnail_variance(t)))
        .collect();

    // Sort by variance descending
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    scored.iter().take(count).map(|&(i, _)| i).collect()
}

// ─── Animated thumbnail ───────────────────────────────────────────────────────

/// An animated thumbnail consisting of multiple frames with timing.
///
/// Represents a short looping preview (e.g. 2–5 seconds) from key moments
/// in a video. The frames are raw RGBA with per-frame duration.
#[derive(Debug, Clone)]
pub struct AnimatedThumbnail {
    /// Width of each frame in pixels.
    pub width: u32,
    /// Height of each frame in pixels.
    pub height: u32,
    /// Frames with their display durations.
    pub frames: Vec<AnimatedFrame>,
    /// Total duration of the animation in milliseconds.
    pub total_duration_ms: u64,
    /// Number of times to loop (0 = infinite).
    pub loop_count: u32,
}

/// A single frame in an animated thumbnail.
#[derive(Debug, Clone)]
pub struct AnimatedFrame {
    /// Raw RGBA pixel data (row-major).
    pub data: Vec<u8>,
    /// Display duration for this frame in milliseconds.
    pub duration_ms: u64,
    /// Source timestamp in the original video (milliseconds).
    pub source_timestamp_ms: u64,
}

impl AnimatedThumbnail {
    /// Creates a new animated thumbnail from a list of source thumbnails.
    ///
    /// Each thumbnail is assigned a uniform frame duration.
    /// Returns `None` if `thumbnails` is empty or dimensions are zero.
    #[must_use]
    pub fn from_thumbnails(
        thumbnails: &[Thumbnail],
        frame_duration_ms: u64,
        loop_count: u32,
    ) -> Option<Self> {
        if thumbnails.is_empty() || frame_duration_ms == 0 {
            return None;
        }

        let width = thumbnails[0].width;
        let height = thumbnails[0].height;
        if width == 0 || height == 0 {
            return None;
        }

        let mut frames = Vec::with_capacity(thumbnails.len());
        let mut total_ms = 0u64;

        for thumb in thumbnails {
            let data = if thumb.width == width && thumb.height == height {
                thumb.data.clone()
            } else {
                scale_thumbnail(&thumb.data, thumb.width, thumb.height, width, height)
            };

            if data.len() < (width * height * 4) as usize {
                continue;
            }

            frames.push(AnimatedFrame {
                data,
                duration_ms: frame_duration_ms,
                source_timestamp_ms: thumb.timestamp_ms,
            });
            total_ms += frame_duration_ms;
        }

        if frames.is_empty() {
            return None;
        }

        Some(Self {
            width,
            height,
            frames,
            total_duration_ms: total_ms,
            loop_count,
        })
    }

    /// Returns the number of frames in the animation.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Returns the total byte size of all frame data.
    #[must_use]
    pub fn total_byte_size(&self) -> usize {
        self.frames.iter().map(|f| f.data.len()).sum()
    }

    /// Creates an animated thumbnail from the most interesting frames,
    /// selected by luminance variance.
    #[must_use]
    pub fn from_smart_selection(
        thumbnails: &[Thumbnail],
        max_frames: usize,
        frame_duration_ms: u64,
        loop_count: u32,
    ) -> Option<Self> {
        let indices = select_smart_thumbnails(thumbnails, max_frames);
        if indices.is_empty() {
            return None;
        }

        // Sort selected indices by timestamp for temporal coherence
        let mut sorted_indices = indices;
        sorted_indices.sort_by_key(|&i| thumbnails[i].timestamp_ms);

        let selected: Vec<Thumbnail> = sorted_indices
            .iter()
            .map(|&i| thumbnails[i].clone())
            .collect();

        Self::from_thumbnails(&selected, frame_duration_ms, loop_count)
    }
}

// ─── Configurable thumbnail quality ──────────────────────────────────────────

/// Quality profile for thumbnail generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThumbnailQualityProfile {
    /// Low quality: smaller file size, faster generation.
    Low,
    /// Medium quality: balanced quality and size.
    Medium,
    /// High quality: larger files, better visual fidelity.
    High,
}

impl ThumbnailQualityProfile {
    /// Returns the JPEG quality value for this profile.
    #[must_use]
    pub fn jpeg_quality(self) -> u8 {
        match self {
            Self::Low => 50,
            Self::Medium => 75,
            Self::High => 92,
        }
    }

    /// Returns the recommended thumbnail dimensions `(width, height)`.
    #[must_use]
    pub fn dimensions(self) -> (u32, u32) {
        match self {
            Self::Low => (120, 68),
            Self::Medium => (240, 135),
            Self::High => (480, 270),
        }
    }

    /// Returns the recommended sprite sheet columns.
    #[must_use]
    pub fn sprite_cols(self) -> u32 {
        match self {
            Self::Low => 10,
            Self::Medium => 5,
            Self::High => 4,
        }
    }
}

/// Extended thumbnail configuration with quality profile support.
#[derive(Debug, Clone)]
pub struct ThumbnailExtConfig {
    /// Base configuration.
    pub base: ThumbnailConfig,
    /// Quality profile.
    pub quality_profile: ThumbnailQualityProfile,
    /// Whether to generate a sprite sheet.
    pub generate_sprite_sheet: bool,
    /// Whether to generate a WebVTT file for the sprite sheet.
    pub generate_vtt: bool,
    /// Whether to generate an animated preview.
    pub generate_animated: bool,
    /// Frame duration for animated thumbnails (milliseconds).
    pub animated_frame_duration_ms: u64,
    /// Maximum number of frames in the animated thumbnail.
    pub animated_max_frames: usize,
}

impl ThumbnailExtConfig {
    /// Creates a new extended config from a quality profile.
    #[must_use]
    pub fn from_profile(profile: ThumbnailQualityProfile, count: usize) -> Self {
        let (w, h) = profile.dimensions();
        Self {
            base: ThumbnailConfig {
                width: w,
                height: h,
                format: ThumbnailFormat::Jpeg,
                quality: profile.jpeg_quality(),
                count,
                interval_strategy: ThumbnailStrategy::Uniform,
            },
            quality_profile: profile,
            generate_sprite_sheet: true,
            generate_vtt: true,
            generate_animated: false,
            animated_frame_duration_ms: 200,
            animated_max_frames: 15,
        }
    }

    /// Enables animated thumbnail generation.
    #[must_use]
    pub fn with_animated(mut self, max_frames: usize, frame_duration_ms: u64) -> Self {
        self.generate_animated = true;
        self.animated_max_frames = max_frames;
        self.animated_frame_duration_ms = frame_duration_ms;
        self
    }

    /// Disables sprite sheet generation.
    #[must_use]
    pub fn without_sprite_sheet(mut self) -> Self {
        self.generate_sprite_sheet = false;
        self.generate_vtt = false;
        self
    }
}

/// Generates a WebVTT thumbnail track file from a sprite sheet and its URL.
///
/// This is a convenience wrapper around `SpriteSheet::to_vtt` that also
/// handles duration-based gap filling for the last cue.
#[must_use]
pub fn generate_vtt_track(
    sprite_sheet: &SpriteSheet,
    sprite_url: &str,
    total_duration_ms: u64,
) -> String {
    let mut vtt = String::from("WEBVTT\n\n");

    for (idx, &ts_ms) in sprite_sheet.timestamps_ms.iter().enumerate() {
        let next_ts_ms = sprite_sheet
            .timestamps_ms
            .get(idx + 1)
            .copied()
            .unwrap_or(total_duration_ms);

        let (x, y) = sprite_sheet.cell_origin(idx);
        let start = format_vtt_time(ts_ms);
        let end = format_vtt_time(next_ts_ms);

        vtt.push_str(&format!(
            "{start} --> {end}\n{sprite_url}#xywh={x},{y},{w},{h}\n\n",
            w = sprite_sheet.cell_width,
            h = sprite_sheet.cell_height,
        ));
    }

    vtt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thumbnail_format_extension() {
        assert_eq!(ThumbnailFormat::Jpeg.extension(), "jpg");
        assert_eq!(ThumbnailFormat::Png.extension(), "png");
        assert_eq!(ThumbnailFormat::Webp.extension(), "webp");
    }

    #[test]
    fn test_thumbnail_format_mime_type() {
        assert_eq!(ThumbnailFormat::Jpeg.mime_type(), "image/jpeg");
        assert_eq!(ThumbnailFormat::Png.mime_type(), "image/png");
        assert_eq!(ThumbnailFormat::Webp.mime_type(), "image/webp");
    }

    #[test]
    fn test_thumbnail_config_default_web() {
        let cfg = ThumbnailConfig::default_web();
        assert_eq!(cfg.width, 320);
        assert_eq!(cfg.height, 180);
        assert_eq!(cfg.format, ThumbnailFormat::Jpeg);
        assert_eq!(cfg.count, 10);
        assert!(cfg.is_valid());
    }

    #[test]
    fn test_thumbnail_config_sprite_sheet() {
        let cfg = ThumbnailConfig::sprite_sheet(20);
        assert_eq!(cfg.width, 160);
        assert_eq!(cfg.height, 90);
        assert_eq!(cfg.count, 20);
        assert!(cfg.is_valid());
    }

    #[test]
    fn test_thumbnail_pixel_count() {
        let thumb = Thumbnail::new(0, 160, 90, vec![0; 160 * 90 * 4]);
        assert_eq!(thumb.pixel_count(), 14400);
        assert_eq!(thumb.expected_byte_len(), 57600);
    }

    #[test]
    fn test_compute_timestamps_at_timestamps() {
        let strategy = ThumbnailStrategy::AtTimestamps(vec![1000, 2000, 3000]);
        let ts = compute_thumbnail_timestamps(5000, &strategy, 0.0);
        assert_eq!(ts, vec![1000, 2000, 3000]);
    }

    #[test]
    fn test_compute_timestamps_at_timestamps_filters_out_of_range() {
        let strategy = ThumbnailStrategy::AtTimestamps(vec![1000, 2000, 9999]);
        let ts = compute_thumbnail_timestamps(5000, &strategy, 0.0);
        assert_eq!(ts, vec![1000, 2000]);
    }

    #[test]
    fn test_compute_timestamps_zero_duration() {
        let ts = compute_thumbnail_timestamps(0, &ThumbnailStrategy::Uniform, 24.0);
        assert!(ts.is_empty());
    }

    #[test]
    fn test_compute_uniform_timestamps_count() {
        let ts = compute_uniform_timestamps(60_000, 5, 0.0);
        assert_eq!(ts.len(), 5);
        // First should be 0, last should be 60000
        assert_eq!(ts[0], 0);
        assert_eq!(ts[4], 60_000);
    }

    #[test]
    fn test_compute_uniform_timestamps_single() {
        let ts = compute_uniform_timestamps(10_000, 1, 0.0);
        assert_eq!(ts.len(), 1);
        assert_eq!(ts[0], 5000);
    }

    #[test]
    fn test_compute_fixed_interval_timestamps() {
        // 30 seconds → timestamps at 0, 10000, 20000, 30000
        let ts = compute_thumbnail_timestamps(30_000, &ThumbnailStrategy::FixedInterval, 0.0);
        assert_eq!(ts, vec![0, 10_000, 20_000, 30_000]);
    }

    #[test]
    fn test_scale_thumbnail_identity() {
        // 2x2 RGBA image (identity scale)
        let src = vec![
            255, 0, 0, 255, // pixel (0,0): red
            0, 255, 0, 255, // pixel (1,0): green
            0, 0, 255, 255, // pixel (0,1): blue
            255, 255, 0, 255, // pixel (1,1): yellow
        ];
        let dst = scale_thumbnail(&src, 2, 2, 2, 2);
        assert_eq!(dst, src);
    }

    #[test]
    fn test_scale_thumbnail_upscale() {
        // 1x1 → 2x2: all pixels should be the same
        let src = vec![100u8, 150, 200, 255];
        let dst = scale_thumbnail(&src, 1, 1, 2, 2);
        assert_eq!(dst.len(), 16);
        // All four pixels should replicate the source
        assert_eq!(&dst[0..4], &[100, 150, 200, 255]);
        assert_eq!(&dst[4..8], &[100, 150, 200, 255]);
    }

    #[test]
    fn test_scale_thumbnail_zero_dimensions() {
        let src = vec![255u8; 16];
        assert!(scale_thumbnail(&src, 0, 2, 4, 4).is_empty());
        assert!(scale_thumbnail(&src, 2, 2, 0, 4).is_empty());
    }

    #[test]
    fn test_scale_thumbnail_undersized_src() {
        // Supply less data than expected → empty result
        let src = vec![255u8; 4]; // only 1 pixel, but claiming 4x4
        let dst = scale_thumbnail(&src, 4, 4, 2, 2);
        assert!(dst.is_empty());
    }

    // ── SpriteSheet tests ─────────────────────────────────────────────────────

    #[test]
    fn test_sprite_sheet_from_thumbnails_empty() {
        assert!(SpriteSheet::from_thumbnails(&[], 5).is_none());
    }

    #[test]
    fn test_sprite_sheet_from_thumbnails_zero_cols() {
        let thumb = Thumbnail::new(0, 160, 90, vec![0u8; 160 * 90 * 4]);
        assert!(SpriteSheet::from_thumbnails(&[thumb], 0).is_none());
    }

    #[test]
    fn test_sprite_sheet_single_thumbnail() {
        let data = vec![200u8; 4 * 4 * 4]; // 4×4 RGBA
        let thumb = Thumbnail::new(1000, 4, 4, data.clone());
        let sheet = SpriteSheet::from_thumbnails(&[thumb], 1).expect("sheet ok");

        assert_eq!(sheet.cols, 1);
        assert_eq!(sheet.rows, 1);
        assert_eq!(sheet.sheet_width, 4);
        assert_eq!(sheet.sheet_height, 4);
        assert_eq!(sheet.cell_count(), 1);
        assert_eq!(sheet.timestamps_ms[0], 1000);
    }

    #[test]
    fn test_sprite_sheet_four_thumbnails_two_cols() {
        let data = vec![128u8; 2 * 2 * 4]; // 2×2 RGBA cells
        let thumbs: Vec<Thumbnail> = (0..4)
            .map(|i| Thumbnail::new(i * 1000, 2, 2, data.clone()))
            .collect();

        let sheet = SpriteSheet::from_thumbnails(&thumbs, 2).expect("sheet ok");

        assert_eq!(sheet.cols, 2);
        assert_eq!(sheet.rows, 2);
        assert_eq!(sheet.sheet_width, 4); // 2 cols × 2 pixels
        assert_eq!(sheet.sheet_height, 4); // 2 rows × 2 pixels
        assert_eq!(sheet.cell_count(), 4);
    }

    #[test]
    fn test_sprite_sheet_cell_origin() {
        let data = vec![0u8; 10 * 10 * 4];
        let thumbs: Vec<Thumbnail> = (0..6)
            .map(|i| Thumbnail::new(i * 1000, 10, 10, data.clone()))
            .collect();
        let sheet = SpriteSheet::from_thumbnails(&thumbs, 3).expect("sheet ok");

        // (col=0, row=0): (0,0)
        assert_eq!(sheet.cell_origin(0), (0, 0));
        // (col=1, row=0): (10, 0)
        assert_eq!(sheet.cell_origin(1), (10, 0));
        // (col=0, row=1): (0, 10)
        assert_eq!(sheet.cell_origin(3), (0, 10));
        // (col=2, row=1): (20, 10)
        assert_eq!(sheet.cell_origin(5), (20, 10));
    }

    #[test]
    fn test_sprite_sheet_vtt_basic() {
        let data = vec![0u8; 4 * 4 * 4];
        let thumbs = vec![
            Thumbnail::new(0, 4, 4, data.clone()),
            Thumbnail::new(10_000, 4, 4, data.clone()),
        ];
        let sheet = SpriteSheet::from_thumbnails(&thumbs, 2).expect("sheet ok");
        let vtt = sheet.to_vtt("https://cdn.example.com/sprites.jpg");

        assert!(vtt.starts_with("WEBVTT\n\n"));
        assert!(vtt.contains("xywh=0,0,4,4")); // first cell at (0,0)
        assert!(vtt.contains("xywh=4,0,4,4")); // second cell at (4,0)
        assert!(vtt.contains("00:00:00.000 --> 00:00:10.000"));
        assert!(vtt.contains("00:00:10.000 --> 00:00:11.000"));
    }

    #[test]
    fn test_format_vtt_time_basic() {
        assert_eq!(format_vtt_time(0), "00:00:00.000");
        assert_eq!(format_vtt_time(1_000), "00:00:01.000");
        assert_eq!(format_vtt_time(61_500), "00:01:01.500");
        assert_eq!(format_vtt_time(3_600_000), "01:00:00.000");
    }

    #[test]
    fn test_format_vtt_time_millis() {
        assert_eq!(format_vtt_time(123), "00:00:00.123");
        assert_eq!(format_vtt_time(1_234), "00:00:01.234");
    }

    #[test]
    fn test_sprite_sheet_config_default_web() {
        let cfg = SpriteSheetConfig::default_web();
        assert!(cfg.is_valid());
        assert_eq!(cfg.cell_width, 160);
        assert_eq!(cfg.cell_height, 90);
        assert_eq!(cfg.cols, 5);
        assert_eq!(cfg.count, 100);
    }

    #[test]
    fn test_sprite_sheet_config_high_density() {
        let cfg = SpriteSheetConfig::high_density(200);
        assert!(cfg.is_valid());
        assert_eq!(cfg.count, 200);
        assert_eq!(cfg.cols, 10);
    }

    #[test]
    fn test_sprite_sheet_config_timestamps() {
        let cfg = SpriteSheetConfig::default_web();
        // 100 seconds of content at 24fps
        let ts = cfg.compute_timestamps(100_000, 24.0);
        assert_eq!(ts.len(), cfg.count);
        // First timestamp should be at or near 0.
        assert!(ts[0] < 1000);
    }

    #[test]
    fn test_sprite_sheet_byte_len() {
        let data = vec![255u8; 4 * 4 * 4];
        let thumbs = vec![Thumbnail::new(0, 4, 4, data)];
        let sheet = SpriteSheet::from_thumbnails(&thumbs, 1).expect("sheet ok");
        // 4×4 RGBA = 64 bytes
        assert_eq!(sheet.byte_len(), 4 * 4 * 4);
    }

    #[test]
    fn test_sprite_sheet_pixel_composition() {
        // Red cell at (0,0), Blue cell at (1,0).
        let red = vec![
            255u8, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255,
        ]; // 2×2 red RGBA
        let blue = vec![
            0u8, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255,
        ]; // 2×2 blue RGBA

        let thumbs = vec![
            Thumbnail::new(0, 2, 2, red.clone()),
            Thumbnail::new(1000, 2, 2, blue.clone()),
        ];
        let sheet = SpriteSheet::from_thumbnails(&thumbs, 2).expect("sheet ok");
        assert_eq!(sheet.sheet_width, 4);
        assert_eq!(sheet.sheet_height, 2);

        // First pixel (0,0) should be red
        assert_eq!(sheet.data[0], 255, "R channel at (0,0)");
        assert_eq!(sheet.data[1], 0, "G channel at (0,0)");
        assert_eq!(sheet.data[2], 0, "B channel at (0,0)");

        // Pixel at (2,0) = first pixel of blue cell
        let blue_offset = 2 * 4; // x=2, y=0, 4 bytes per pixel, stride=sheet_width*4=16
        assert_eq!(sheet.data[blue_offset], 0, "R channel at (2,0)");
        assert_eq!(sheet.data[blue_offset + 2], 255, "B channel at (2,0)");
    }

    // ── Smart thumbnail selection tests ──────────────────────────────────────

    #[test]
    fn test_thumbnail_variance_flat_image() {
        // All same colour → variance near zero
        let data = vec![128u8, 128, 128, 255].repeat(4); // 2×2
        let thumb = Thumbnail::new(0, 2, 2, data);
        let v = thumbnail_variance(&thumb);
        assert!(v.abs() < 1.0);
    }

    #[test]
    fn test_thumbnail_variance_high_contrast() {
        // Black and white checkerboard → high variance
        let mut data = Vec::with_capacity(4 * 4);
        data.extend_from_slice(&[0, 0, 0, 255]); // black
        data.extend_from_slice(&[255, 255, 255, 255]); // white
        data.extend_from_slice(&[255, 255, 255, 255]); // white
        data.extend_from_slice(&[0, 0, 0, 255]); // black
        let thumb = Thumbnail::new(0, 2, 2, data);
        let v = thumbnail_variance(&thumb);
        assert!(v > 1000.0);
    }

    #[test]
    fn test_thumbnail_variance_empty() {
        let thumb = Thumbnail::new(0, 0, 0, Vec::new());
        assert!((thumbnail_variance(&thumb)).abs() < 1e-6);
    }

    #[test]
    fn test_select_smart_thumbnails_empty() {
        assert!(select_smart_thumbnails(&[], 5).is_empty());
    }

    #[test]
    fn test_select_smart_thumbnails_picks_interesting() {
        let flat = vec![128u8, 128, 128, 255].repeat(4); // low variance
        let high_contrast = {
            let mut d = Vec::with_capacity(16);
            d.extend_from_slice(&[0, 0, 0, 255]);
            d.extend_from_slice(&[255, 255, 255, 255]);
            d.extend_from_slice(&[255, 255, 255, 255]);
            d.extend_from_slice(&[0, 0, 0, 255]);
            d
        };

        let thumbs = vec![
            Thumbnail::new(0, 2, 2, flat.clone()),
            Thumbnail::new(1000, 2, 2, high_contrast),
            Thumbnail::new(2000, 2, 2, flat),
        ];

        let selected = select_smart_thumbnails(&thumbs, 1);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0], 1); // The high-contrast one
    }

    #[test]
    fn test_select_smart_thumbnails_count_capped() {
        let data = vec![128u8, 128, 128, 255].repeat(4);
        let thumbs: Vec<Thumbnail> = (0..3)
            .map(|i| Thumbnail::new(i * 1000, 2, 2, data.clone()))
            .collect();

        let selected = select_smart_thumbnails(&thumbs, 10);
        assert_eq!(selected.len(), 3); // Can't pick more than available
    }

    // ── Animated thumbnail tests ─────────────────────────────────────────────

    #[test]
    fn test_animated_thumbnail_empty() {
        assert!(AnimatedThumbnail::from_thumbnails(&[], 100, 0).is_none());
    }

    #[test]
    fn test_animated_thumbnail_zero_duration() {
        let thumb = Thumbnail::new(0, 2, 2, vec![0u8; 16]);
        assert!(AnimatedThumbnail::from_thumbnails(&[thumb], 0, 0).is_none());
    }

    #[test]
    fn test_animated_thumbnail_basic() {
        let data = vec![128u8; 4 * 4 * 4]; // 4×4 RGBA
        let thumbs: Vec<Thumbnail> = (0..3)
            .map(|i| Thumbnail::new(i * 1000, 4, 4, data.clone()))
            .collect();

        let anim = AnimatedThumbnail::from_thumbnails(&thumbs, 200, 0)
            .expect("should create animated thumbnail");
        assert_eq!(anim.frame_count(), 3);
        assert_eq!(anim.total_duration_ms, 600);
        assert_eq!(anim.width, 4);
        assert_eq!(anim.height, 4);
        assert!(anim.total_byte_size() > 0);
    }

    #[test]
    fn test_animated_thumbnail_smart_selection() {
        let flat = vec![128u8, 128, 128, 255].repeat(4);
        let interesting = {
            let mut d = Vec::with_capacity(16);
            d.extend_from_slice(&[0, 0, 0, 255]);
            d.extend_from_slice(&[255, 255, 255, 255]);
            d.extend_from_slice(&[200, 100, 50, 255]);
            d.extend_from_slice(&[50, 100, 200, 255]);
            d
        };

        let thumbs = vec![
            Thumbnail::new(0, 2, 2, flat.clone()),
            Thumbnail::new(1000, 2, 2, interesting.clone()),
            Thumbnail::new(2000, 2, 2, flat.clone()),
            Thumbnail::new(3000, 2, 2, interesting),
            Thumbnail::new(4000, 2, 2, flat),
        ];

        let anim =
            AnimatedThumbnail::from_smart_selection(&thumbs, 2, 300, 0).expect("should select");
        assert_eq!(anim.frame_count(), 2);
        assert_eq!(anim.total_duration_ms, 600);
    }

    // ── Quality profile tests ────────────────────────────────────────────────

    #[test]
    fn test_quality_profile_jpeg_quality() {
        assert!(
            ThumbnailQualityProfile::Low.jpeg_quality()
                < ThumbnailQualityProfile::Medium.jpeg_quality()
        );
        assert!(
            ThumbnailQualityProfile::Medium.jpeg_quality()
                < ThumbnailQualityProfile::High.jpeg_quality()
        );
    }

    #[test]
    fn test_quality_profile_dimensions() {
        let (lw, lh) = ThumbnailQualityProfile::Low.dimensions();
        let (mw, mh) = ThumbnailQualityProfile::Medium.dimensions();
        let (hw, hh) = ThumbnailQualityProfile::High.dimensions();
        assert!(lw < mw);
        assert!(mw < hw);
        assert!(lh < mh);
        assert!(mh < hh);
    }

    #[test]
    fn test_thumbnail_ext_config_from_profile() {
        let cfg = ThumbnailExtConfig::from_profile(ThumbnailQualityProfile::Medium, 50);
        assert_eq!(cfg.base.count, 50);
        assert_eq!(cfg.base.width, 240);
        assert!(cfg.generate_sprite_sheet);
        assert!(cfg.generate_vtt);
        assert!(!cfg.generate_animated);
    }

    #[test]
    fn test_thumbnail_ext_config_with_animated() {
        let cfg = ThumbnailExtConfig::from_profile(ThumbnailQualityProfile::High, 100)
            .with_animated(10, 150);
        assert!(cfg.generate_animated);
        assert_eq!(cfg.animated_max_frames, 10);
        assert_eq!(cfg.animated_frame_duration_ms, 150);
    }

    #[test]
    fn test_thumbnail_ext_config_without_sprite() {
        let cfg = ThumbnailExtConfig::from_profile(ThumbnailQualityProfile::Low, 20)
            .without_sprite_sheet();
        assert!(!cfg.generate_sprite_sheet);
        assert!(!cfg.generate_vtt);
    }

    // ── generate_vtt_track tests ─────────────────────────────────────────────

    #[test]
    fn test_generate_vtt_track_basic() {
        let data = vec![0u8; 4 * 4 * 4];
        let thumbs = vec![
            Thumbnail::new(0, 4, 4, data.clone()),
            Thumbnail::new(5_000, 4, 4, data),
        ];
        let sheet = SpriteSheet::from_thumbnails(&thumbs, 2).expect("sheet ok");
        let vtt = generate_vtt_track(&sheet, "sprites.jpg", 10_000);

        assert!(vtt.starts_with("WEBVTT"));
        assert!(vtt.contains("00:00:00.000 --> 00:00:05.000"));
        assert!(vtt.contains("00:00:05.000 --> 00:00:10.000"));
        assert!(vtt.contains("xywh=0,0,4,4"));
        assert!(vtt.contains("xywh=4,0,4,4"));
    }

    #[test]
    fn test_generate_vtt_track_single_thumb() {
        let data = vec![0u8; 4 * 4 * 4];
        let thumbs = vec![Thumbnail::new(0, 4, 4, data)];
        let sheet = SpriteSheet::from_thumbnails(&thumbs, 1).expect("sheet ok");
        let vtt = generate_vtt_track(&sheet, "s.jpg", 60_000);

        assert!(vtt.contains("00:00:00.000 --> 00:01:00.000"));
    }
}
