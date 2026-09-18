//! Video thumbnail sprite sheet generator.
//!
//! Generates sprite sheets (grid layouts of video thumbnails) for video preview,
//! seeking, and navigation. Supports:
//!
//! - Multiple sampling strategies (uniform, scene-based, keyframe-only)
//! - Configurable grid layouts (columns × rows)
//! - Various output formats (PNG, JPEG, WebP)
//! - WebVTT file generation for seeking metadata
//! - JSON manifest with thumbnail positions
//! - Timestamp overlays on thumbnails
//! - Quality control and compression settings
//! - Aspect ratio preservation
//! - Custom padding and margins

pub mod generate;
pub mod output;
pub mod timestamps;
pub mod utils;

pub use generate::generate_sprite_sheet;
#[allow(unused_imports)]
pub use utils::{adjust_grid_for_count, calculate_optimal_grid, parse_timestamp};
pub use utils::{parse_duration, validate_and_adjust_config};

pub use crate::progress::TranscodeProgress;
pub use anyhow::{anyhow, Context, Result};
pub use colored::Colorize;
pub use serde::{Deserialize, Serialize};
pub use std::fmt;
pub use std::path::{Path, PathBuf};
pub use tracing::{debug, info};

/// Maximum reasonable dimensions for sprite sheets.
const MAX_SPRITE_DIMENSION: u32 = 16384;

/// Default thumbnail dimensions.
const DEFAULT_THUMB_WIDTH: u32 = 160;
const DEFAULT_THUMB_HEIGHT: u32 = 90;

/// Default grid size.
const DEFAULT_GRID_COLS: usize = 5;
const DEFAULT_GRID_ROWS: usize = 5;

/// Default spacing between thumbnails.
const DEFAULT_SPACING: u32 = 2;

/// Default margin around the sprite sheet.
const DEFAULT_MARGIN: u32 = 0;

/// Options for sprite sheet generation.
#[derive(Debug, Clone)]
pub struct SpriteSheetOptions {
    /// Input video file path.
    pub input: PathBuf,

    /// Output sprite sheet file path.
    pub output: PathBuf,

    /// Sprite sheet configuration.
    pub config: SpriteSheetConfig,

    /// Whether to generate WebVTT file.
    pub generate_vtt: bool,

    /// WebVTT output file path (if generate_vtt is true).
    pub vtt_output: Option<PathBuf>,

    /// Whether to generate JSON manifest.
    pub generate_manifest: bool,

    /// JSON manifest output path (if generate_manifest is true).
    pub manifest_output: Option<PathBuf>,

    /// Whether to overlay timestamps on thumbnails.
    pub show_timestamps: bool,

    /// Whether to output results as JSON.
    pub json_output: bool,
}

/// Configuration for sprite sheet generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpriteSheetConfig {
    /// Interval between thumbnails in seconds (for uniform sampling).
    pub interval: Option<f64>,

    /// Total number of thumbnails to extract (alternative to interval).
    pub count: Option<usize>,

    /// Thumbnail width in pixels.
    pub thumbnail_width: u32,

    /// Thumbnail height in pixels.
    pub thumbnail_height: u32,

    /// Number of columns in the grid.
    pub columns: usize,

    /// Number of rows in the grid.
    pub rows: usize,

    /// Image output format.
    pub format: ImageFormat,

    /// Quality setting (0-100 for JPEG/WebP).
    pub quality: u8,

    /// Frame sampling strategy.
    pub strategy: SamplingStrategy,

    /// Layout mode for the sprite sheet.
    pub layout: LayoutMode,

    /// Spacing between thumbnails in pixels.
    pub spacing: u32,

    /// Margin around the sprite sheet in pixels.
    pub margin: u32,

    /// Whether to maintain aspect ratio when scaling.
    pub maintain_aspect_ratio: bool,

    /// Compression level (0-9, higher is smaller file size).
    pub compression: u8,
}

impl Default for SpriteSheetConfig {
    fn default() -> Self {
        Self {
            interval: None,
            count: None,
            thumbnail_width: DEFAULT_THUMB_WIDTH,
            thumbnail_height: DEFAULT_THUMB_HEIGHT,
            columns: DEFAULT_GRID_COLS,
            rows: DEFAULT_GRID_ROWS,
            format: ImageFormat::Png,
            quality: 90,
            strategy: SamplingStrategy::Uniform,
            layout: LayoutMode::Grid,
            spacing: DEFAULT_SPACING,
            margin: DEFAULT_MARGIN,
            maintain_aspect_ratio: true,
            compression: 6,
        }
    }
}

impl SpriteSheetConfig {
    /// Calculate total number of thumbnails to generate.
    pub fn total_thumbnails(&self) -> usize {
        if let Some(count) = self.count {
            count
        } else {
            self.columns * self.rows
        }
    }

    /// Calculate sprite sheet dimensions.
    pub fn sprite_dimensions(&self) -> (u32, u32) {
        let cols = self.columns as u32;
        let rows = self.rows as u32;

        let width =
            self.margin * 2 + self.thumbnail_width * cols + self.spacing * (cols.saturating_sub(1));

        let height = self.margin * 2
            + self.thumbnail_height * rows
            + self.spacing * (rows.saturating_sub(1));

        (width, height)
    }

    /// Validate configuration.
    pub fn validate(&self) -> Result<()> {
        // Validate dimensions
        if self.thumbnail_width == 0 || self.thumbnail_height == 0 {
            return Err(anyhow!("Thumbnail dimensions must be greater than zero"));
        }

        if self.columns == 0 || self.rows == 0 {
            return Err(anyhow!("Grid columns and rows must be greater than zero"));
        }

        // Check for excessively large sprite sheets
        let (width, height) = self.sprite_dimensions();
        if width > MAX_SPRITE_DIMENSION || height > MAX_SPRITE_DIMENSION {
            return Err(anyhow!(
                "Sprite sheet dimensions ({}x{}) exceed maximum ({}x{})",
                width,
                height,
                MAX_SPRITE_DIMENSION,
                MAX_SPRITE_DIMENSION
            ));
        }

        // Validate quality
        if self.quality > 100 {
            return Err(anyhow!("Quality must be between 0 and 100"));
        }

        // Validate compression
        if self.compression > 9 {
            return Err(anyhow!("Compression level must be between 0 and 9"));
        }

        // Validate count or interval is specified
        if self.count.is_none() && self.interval.is_none() {
            return Err(anyhow!("Either count or interval must be specified"));
        }

        Ok(())
    }
}

/// Image output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageFormat {
    /// PNG format (lossless).
    Png,

    /// JPEG format (lossy).
    Jpeg,

    /// WebP format (lossy or lossless).
    Webp,
}

impl ImageFormat {
    /// Parse format from string.
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "png" => Ok(Self::Png),
            "jpg" | "jpeg" => Ok(Self::Jpeg),
            "webp" => Ok(Self::Webp),
            _ => Err(anyhow!("Unsupported image format: {}", s)),
        }
    }

    /// Get format name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Png => "PNG",
            Self::Jpeg => "JPEG",
            Self::Webp => "WebP",
        }
    }

    /// Get file extension.
    #[allow(dead_code)]
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg => "jpg",
            Self::Webp => "webp",
        }
    }

    /// Check if format is lossless.
    pub fn is_lossless(&self) -> bool {
        matches!(self, Self::Png)
    }
}

impl fmt::Display for ImageFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Frame sampling strategy for thumbnail extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SamplingStrategy {
    /// Uniform sampling at fixed intervals.
    Uniform,

    /// Scene-based sampling (extracts representative frames from each scene).
    SceneBased,

    /// Keyframe-only sampling (extracts only I-frames).
    KeyframeOnly,

    /// Smart sampling (combination of scene detection and quality analysis).
    Smart,
}

impl SamplingStrategy {
    /// Parse strategy from string.
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "uniform" => Ok(Self::Uniform),
            "scene" | "scene-based" => Ok(Self::SceneBased),
            "keyframe" | "keyframe-only" => Ok(Self::KeyframeOnly),
            "smart" => Ok(Self::Smart),
            _ => Err(anyhow!("Unsupported sampling strategy: {}", s)),
        }
    }

    /// Get strategy name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Uniform => "Uniform",
            Self::SceneBased => "Scene-based",
            Self::KeyframeOnly => "Keyframe-only",
            Self::Smart => "Smart",
        }
    }

    /// Get strategy description.
    #[allow(dead_code)]
    pub fn description(&self) -> &'static str {
        match self {
            Self::Uniform => "Extract frames at regular intervals",
            Self::SceneBased => "Extract representative frames from each scene",
            Self::KeyframeOnly => "Extract only keyframes (I-frames)",
            Self::Smart => "Intelligent frame selection based on content",
        }
    }
}

impl fmt::Display for SamplingStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Layout mode for sprite sheet arrangement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LayoutMode {
    /// Grid layout (columns × rows).
    Grid,

    /// Vertical stack (single column).
    Vertical,

    /// Horizontal filmstrip (single row).
    Horizontal,

    /// Auto-calculated grid based on aspect ratio.
    Auto,
}

impl LayoutMode {
    /// Parse layout from string.
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "grid" => Ok(Self::Grid),
            "vertical" | "vert" | "column" => Ok(Self::Vertical),
            "horizontal" | "horiz" | "row" | "filmstrip" => Ok(Self::Horizontal),
            "auto" => Ok(Self::Auto),
            _ => Err(anyhow!("Unsupported layout mode: {}", s)),
        }
    }

    /// Get layout name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Grid => "Grid",
            Self::Vertical => "Vertical",
            Self::Horizontal => "Horizontal",
            Self::Auto => "Auto",
        }
    }
}

impl fmt::Display for LayoutMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Thumbnail metadata for sprite sheet positioning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThumbnailMetadata {
    /// Thumbnail index (0-based).
    pub index: usize,

    /// Timestamp in seconds.
    pub timestamp: f64,

    /// X position in sprite sheet (pixels).
    pub x: u32,

    /// Y position in sprite sheet (pixels).
    pub y: u32,

    /// Thumbnail width (pixels).
    pub width: u32,

    /// Thumbnail height (pixels).
    pub height: u32,
}

/// Sprite sheet generation result.
#[derive(Debug, Serialize)]
pub struct SpriteSheetResult {
    /// Whether generation was successful.
    pub success: bool,

    /// Output sprite sheet file path.
    pub sprite_path: String,

    /// Sprite sheet dimensions.
    pub sprite_width: u32,

    /// Sprite sheet dimensions.
    pub sprite_height: u32,

    /// Number of thumbnails generated.
    pub thumbnail_count: usize,

    /// Thumbnail dimensions.
    pub thumbnail_width: u32,

    /// Thumbnail dimensions.
    pub thumbnail_height: u32,

    /// Grid dimensions.
    pub columns: usize,

    /// Grid dimensions.
    pub rows: usize,

    /// Output format.
    pub format: String,

    /// WebVTT file path (if generated).
    pub vtt_path: Option<String>,

    /// JSON manifest path (if generated).
    pub manifest_path: Option<String>,

    /// Total processing time in seconds.
    pub processing_time: f64,
}

/// Sprite sheet manifest (JSON output).
#[derive(Debug, Serialize, Deserialize)]
pub struct SpriteSheetManifest {
    /// Sprite sheet file path.
    pub sprite_file: String,

    /// Video source file path.
    pub video_file: String,

    /// Sprite sheet dimensions.
    pub sprite_width: u32,

    /// Sprite sheet dimensions.
    pub sprite_height: u32,

    /// Thumbnail metadata for each thumbnail.
    pub thumbnails: Vec<ThumbnailMetadata>,

    /// Configuration used for generation.
    pub config: SpriteSheetConfig,

    /// Video duration in seconds.
    pub video_duration: f64,

    /// Video frame rate.
    pub video_fps: f64,

    /// Generation timestamp.
    pub generated_at: String,
}

#[allow(dead_code)]
pub mod presets {
    use super::*;

    /// YouTube video player sprite sheet preset.
    pub fn youtube_preview() -> SpriteSheetConfig {
        SpriteSheetConfig {
            interval: None,
            count: Some(100),
            thumbnail_width: 160,
            thumbnail_height: 90,
            columns: 10,
            rows: 10,
            format: ImageFormat::Jpeg,
            quality: 85,
            strategy: SamplingStrategy::Uniform,
            layout: LayoutMode::Grid,
            spacing: 0,
            margin: 0,
            maintain_aspect_ratio: true,
            compression: 6,
        }
    }

    /// High-quality preview sprite sheet.
    pub fn high_quality() -> SpriteSheetConfig {
        SpriteSheetConfig {
            interval: None,
            count: Some(100),
            thumbnail_width: 320,
            thumbnail_height: 180,
            columns: 10,
            rows: 10,
            format: ImageFormat::Png,
            quality: 95,
            strategy: SamplingStrategy::Smart,
            layout: LayoutMode::Grid,
            spacing: 4,
            margin: 2,
            maintain_aspect_ratio: true,
            compression: 6,
        }
    }

    /// Fast loading sprite sheet for web.
    pub fn web_optimized() -> SpriteSheetConfig {
        SpriteSheetConfig {
            interval: Some(5.0),
            count: None,
            thumbnail_width: 128,
            thumbnail_height: 72,
            columns: 8,
            rows: 8,
            format: ImageFormat::Webp,
            quality: 80,
            strategy: SamplingStrategy::Uniform,
            layout: LayoutMode::Grid,
            spacing: 0,
            margin: 0,
            maintain_aspect_ratio: true,
            compression: 9,
        }
    }

    /// Filmstrip layout preset.
    pub fn filmstrip() -> SpriteSheetConfig {
        SpriteSheetConfig {
            interval: Some(10.0),
            count: None,
            thumbnail_width: 200,
            thumbnail_height: 112,
            columns: 10,
            rows: 1,
            format: ImageFormat::Png,
            quality: 90,
            strategy: SamplingStrategy::KeyframeOnly,
            layout: LayoutMode::Horizontal,
            spacing: 4,
            margin: 0,
            maintain_aspect_ratio: true,
            compression: 6,
        }
    }

    /// Scene detection sprite sheet.
    pub fn scene_preview() -> SpriteSheetConfig {
        SpriteSheetConfig {
            interval: None,
            count: Some(50),
            thumbnail_width: 240,
            thumbnail_height: 135,
            columns: 10,
            rows: 5,
            format: ImageFormat::Jpeg,
            quality: 90,
            strategy: SamplingStrategy::SceneBased,
            layout: LayoutMode::Grid,
            spacing: 2,
            margin: 0,
            maintain_aspect_ratio: true,
            compression: 6,
        }
    }

    /// Compact mobile-friendly sprite sheet.
    pub fn mobile() -> SpriteSheetConfig {
        SpriteSheetConfig {
            interval: Some(15.0),
            count: None,
            thumbnail_width: 96,
            thumbnail_height: 54,
            columns: 6,
            rows: 6,
            format: ImageFormat::Webp,
            quality: 75,
            strategy: SamplingStrategy::Uniform,
            layout: LayoutMode::Grid,
            spacing: 1,
            margin: 0,
            maintain_aspect_ratio: true,
            compression: 9,
        }
    }

    /// Get preset by name.
    pub fn get_preset(name: &str) -> Option<SpriteSheetConfig> {
        match name.to_lowercase().as_str() {
            "youtube" | "youtube-preview" => Some(youtube_preview()),
            "high-quality" | "hq" => Some(high_quality()),
            "web" | "web-optimized" => Some(web_optimized()),
            "filmstrip" | "strip" => Some(filmstrip()),
            "scene" | "scene-preview" => Some(scene_preview()),
            "mobile" => Some(mobile()),
            _ => None,
        }
    }

    /// List all available preset names.
    pub fn list_presets() -> Vec<&'static str> {
        vec![
            "youtube-preview",
            "high-quality",
            "web-optimized",
            "filmstrip",
            "scene-preview",
            "mobile",
        ]
    }
}

/// Color utilities for timestamp overlays and decorations.
#[allow(dead_code)]
pub mod color {
    /// RGBA color representation.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Color {
        pub r: u8,
        pub g: u8,
        pub b: u8,
        pub a: u8,
    }

    impl Color {
        /// Create a new color from RGBA components.
        pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
            Self { r, g, b, a }
        }

        /// Create a new color from RGB components (fully opaque).
        pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
            Self::new(r, g, b, 255)
        }

        /// Create a new color from hex string (e.g., "#FF0000" or "FF0000").
        pub fn from_hex(hex: &str) -> Option<Self> {
            let hex = hex.trim_start_matches('#');

            if hex.len() != 6 && hex.len() != 8 {
                return None;
            }

            let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
            let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
            let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
            let a = if hex.len() == 8 {
                u8::from_str_radix(&hex[6..8], 16).ok()?
            } else {
                255
            };

            Some(Self::new(r, g, b, a))
        }

        /// Convert to hex string.
        pub fn to_hex(&self) -> String {
            if self.a == 255 {
                format!("#{:02X}{:02X}{:02X}", self.r, self.g, self.b)
            } else {
                format!("#{:02X}{:02X}{:02X}{:02X}", self.r, self.g, self.b, self.a)
            }
        }

        /// Blend this color with another using alpha compositing.
        pub fn blend(&self, other: &Self) -> Self {
            let alpha = other.a as f32 / 255.0;
            let inv_alpha = 1.0 - alpha;

            let r = (self.r as f32 * inv_alpha + other.r as f32 * alpha) as u8;
            let g = (self.g as f32 * inv_alpha + other.g as f32 * alpha) as u8;
            let b = (self.b as f32 * inv_alpha + other.b as f32 * alpha) as u8;
            let a = 255; // Result is fully opaque

            Self::new(r, g, b, a)
        }

        /// Create a semi-transparent version of this color.
        pub fn with_alpha(&self, alpha: u8) -> Self {
            Self::new(self.r, self.g, self.b, alpha)
        }

        /// Convert to grayscale.
        pub fn to_grayscale(&self) -> Self {
            // Use standard luminance formula
            let gray =
                (0.299 * self.r as f32 + 0.587 * self.g as f32 + 0.114 * self.b as f32) as u8;
            Self::new(gray, gray, gray, self.a)
        }
    }

    /// Common color constants.
    impl Color {
        pub const BLACK: Self = Self::rgb(0, 0, 0);
        pub const WHITE: Self = Self::rgb(255, 255, 255);
        pub const RED: Self = Self::rgb(255, 0, 0);
        pub const GREEN: Self = Self::rgb(0, 255, 0);
        pub const BLUE: Self = Self::rgb(0, 0, 255);
        pub const YELLOW: Self = Self::rgb(255, 255, 0);
        pub const CYAN: Self = Self::rgb(0, 255, 255);
        pub const MAGENTA: Self = Self::rgb(255, 0, 255);
        pub const TRANSPARENT: Self = Self::new(0, 0, 0, 0);
    }
}

/// Timestamp overlay configuration and rendering.
#[allow(dead_code)]
pub mod overlay {
    use super::*;

    /// Timestamp overlay style.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "kebab-case")]
    pub enum OverlayStyle {
        /// Simple text overlay.
        Simple,

        /// Text with background box.
        Box,

        /// Text with shadow.
        Shadow,

        /// Text with outline.
        Outline,

        /// Pill-shaped background.
        Pill,
    }

    /// Overlay position on thumbnail.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "kebab-case")]
    pub enum OverlayPosition {
        TopLeft,
        TopCenter,
        TopRight,
        MiddleLeft,
        MiddleCenter,
        MiddleRight,
        BottomLeft,
        BottomCenter,
        BottomRight,
    }

    /// Timestamp overlay configuration.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct OverlayConfig {
        /// Overlay style.
        pub style: OverlayStyle,

        /// Position on thumbnail.
        pub position: OverlayPosition,

        /// Text color.
        pub text_color: String,

        /// Background color (for box/pill styles).
        pub background_color: String,

        /// Font size in pixels.
        pub font_size: u32,

        /// Padding in pixels.
        pub padding: u32,

        /// Shadow offset (for shadow style).
        pub shadow_offset: (i32, i32),

        /// Outline width (for outline style).
        pub outline_width: u32,
    }

    impl Default for OverlayConfig {
        fn default() -> Self {
            Self {
                style: OverlayStyle::Box,
                position: OverlayPosition::BottomRight,
                text_color: "#FFFFFF".to_string(),
                background_color: "#000000CC".to_string(),
                font_size: 12,
                padding: 4,
                shadow_offset: (1, 1),
                outline_width: 1,
            }
        }
    }

    /// Calculate overlay dimensions for a given timestamp string.
    pub fn calculate_overlay_size(timestamp: &str, config: &OverlayConfig) -> (u32, u32) {
        // Rough estimation: each character is ~0.6 * font_size wide
        let char_width = (config.font_size as f32 * 0.6) as u32;
        let text_width = timestamp.len() as u32 * char_width;
        let text_height = config.font_size;

        let width = text_width + 2 * config.padding;
        let height = text_height + 2 * config.padding;

        (width, height)
    }

    /// Calculate overlay position coordinates.
    pub fn calculate_overlay_position(
        thumb_width: u32,
        thumb_height: u32,
        overlay_width: u32,
        overlay_height: u32,
        position: OverlayPosition,
    ) -> (u32, u32) {
        let margin = 4u32; // Small margin from edges

        match position {
            OverlayPosition::TopLeft => (margin, margin),
            OverlayPosition::TopCenter => ((thumb_width.saturating_sub(overlay_width)) / 2, margin),
            OverlayPosition::TopRight => {
                (thumb_width.saturating_sub(overlay_width + margin), margin)
            }
            OverlayPosition::MiddleLeft => {
                (margin, (thumb_height.saturating_sub(overlay_height)) / 2)
            }
            OverlayPosition::MiddleCenter => (
                (thumb_width.saturating_sub(overlay_width)) / 2,
                (thumb_height.saturating_sub(overlay_height)) / 2,
            ),
            OverlayPosition::MiddleRight => (
                thumb_width.saturating_sub(overlay_width + margin),
                (thumb_height.saturating_sub(overlay_height)) / 2,
            ),
            OverlayPosition::BottomLeft => {
                (margin, thumb_height.saturating_sub(overlay_height + margin))
            }
            OverlayPosition::BottomCenter => (
                (thumb_width.saturating_sub(overlay_width)) / 2,
                thumb_height.saturating_sub(overlay_height + margin),
            ),
            OverlayPosition::BottomRight => (
                thumb_width.saturating_sub(overlay_width + margin),
                thumb_height.saturating_sub(overlay_height + margin),
            ),
        }
    }
}

/// Image processing utilities for sprite sheet generation.
#[allow(dead_code)]
pub mod image_utils {

    /// Image buffer representation (simple placeholder).
    pub struct ImageBuffer {
        pub width: u32,
        pub height: u32,
        pub data: Vec<u8>,
        pub channels: u8,
    }

    impl ImageBuffer {
        /// Create a new image buffer.
        pub fn new(width: u32, height: u32, channels: u8) -> Self {
            let size = (width * height * channels as u32) as usize;
            Self {
                width,
                height,
                data: vec![0; size],
                channels,
            }
        }

        /// Create from existing data.
        pub fn from_raw(width: u32, height: u32, channels: u8, data: Vec<u8>) -> Self {
            Self {
                width,
                height,
                data,
                channels,
            }
        }

        /// Get pixel at position (returns RGBA).
        pub fn get_pixel(&self, x: u32, y: u32) -> Option<[u8; 4]> {
            if x >= self.width || y >= self.height {
                return None;
            }

            let offset = ((y * self.width + x) * self.channels as u32) as usize;

            match self.channels {
                1 => {
                    let gray = self.data[offset];
                    Some([gray, gray, gray, 255])
                }
                3 => Some([
                    self.data[offset],
                    self.data[offset + 1],
                    self.data[offset + 2],
                    255,
                ]),
                4 => Some([
                    self.data[offset],
                    self.data[offset + 1],
                    self.data[offset + 2],
                    self.data[offset + 3],
                ]),
                _ => None,
            }
        }

        /// Set pixel at position (from RGBA).
        pub fn set_pixel(&mut self, x: u32, y: u32, pixel: [u8; 4]) -> bool {
            if x >= self.width || y >= self.height {
                return false;
            }

            let offset = ((y * self.width + x) * self.channels as u32) as usize;

            match self.channels {
                1 => {
                    // Convert to grayscale
                    let gray = (0.299 * pixel[0] as f32
                        + 0.587 * pixel[1] as f32
                        + 0.114 * pixel[2] as f32) as u8;
                    self.data[offset] = gray;
                }
                3 => {
                    self.data[offset] = pixel[0];
                    self.data[offset + 1] = pixel[1];
                    self.data[offset + 2] = pixel[2];
                }
                4 => {
                    self.data[offset] = pixel[0];
                    self.data[offset + 1] = pixel[1];
                    self.data[offset + 2] = pixel[2];
                    self.data[offset + 3] = pixel[3];
                }
                _ => return false,
            }

            true
        }

        /// Resize image using nearest-neighbor sampling.
        pub fn resize_nearest(&self, new_width: u32, new_height: u32) -> Self {
            let mut result = Self::new(new_width, new_height, self.channels);

            let x_ratio = self.width as f32 / new_width as f32;
            let y_ratio = self.height as f32 / new_height as f32;

            for y in 0..new_height {
                for x in 0..new_width {
                    let src_x = (x as f32 * x_ratio) as u32;
                    let src_y = (y as f32 * y_ratio) as u32;

                    if let Some(pixel) = self.get_pixel(src_x, src_y) {
                        result.set_pixel(x, y, pixel);
                    }
                }
            }

            result
        }

        /// Resize image using bilinear interpolation.
        pub fn resize_bilinear(&self, new_width: u32, new_height: u32) -> Self {
            let mut result = Self::new(new_width, new_height, self.channels);

            let x_ratio = (self.width - 1) as f32 / new_width as f32;
            let y_ratio = (self.height - 1) as f32 / new_height as f32;

            for y in 0..new_height {
                for x in 0..new_width {
                    let gx = x as f32 * x_ratio;
                    let gy = y as f32 * y_ratio;

                    let gxi = gx as u32;
                    let gyi = gy as u32;

                    let gxi_next = (gxi + 1).min(self.width - 1);
                    let gyi_next = (gyi + 1).min(self.height - 1);

                    // Get four surrounding pixels
                    let c00 = self.get_pixel(gxi, gyi).unwrap_or([0; 4]);
                    let c10 = self.get_pixel(gxi_next, gyi).unwrap_or([0; 4]);
                    let c01 = self.get_pixel(gxi, gyi_next).unwrap_or([0; 4]);
                    let c11 = self.get_pixel(gxi_next, gyi_next).unwrap_or([0; 4]);

                    // Bilinear interpolation
                    let x_weight = gx - gxi as f32;
                    let y_weight = gy - gyi as f32;

                    let mut pixel = [0u8; 4];
                    for i in 0..4 {
                        let top = c00[i] as f32 * (1.0 - x_weight) + c10[i] as f32 * x_weight;
                        let bottom = c01[i] as f32 * (1.0 - x_weight) + c11[i] as f32 * x_weight;
                        pixel[i] = (top * (1.0 - y_weight) + bottom * y_weight) as u8;
                    }

                    result.set_pixel(x, y, pixel);
                }
            }

            result
        }

        /// Fill a rectangular region with a solid color.
        pub fn fill_rect(&mut self, x: u32, y: u32, width: u32, height: u32, color: [u8; 4]) {
            for dy in 0..height {
                for dx in 0..width {
                    self.set_pixel(x + dx, y + dy, color);
                }
            }
        }

        /// Composite another image onto this one at the specified position.
        pub fn composite(&mut self, other: &ImageBuffer, x: u32, y: u32, alpha_blend: bool) {
            for dy in 0..other.height {
                for dx in 0..other.width {
                    if let Some(src_pixel) = other.get_pixel(dx, dy) {
                        let dest_x = x + dx;
                        let dest_y = y + dy;

                        if dest_x < self.width && dest_y < self.height {
                            if alpha_blend && src_pixel[3] < 255 {
                                // Alpha blending
                                if let Some(dest_pixel) = self.get_pixel(dest_x, dest_y) {
                                    let alpha = src_pixel[3] as f32 / 255.0;
                                    let inv_alpha = 1.0 - alpha;

                                    let blended = [
                                        (dest_pixel[0] as f32 * inv_alpha
                                            + src_pixel[0] as f32 * alpha)
                                            as u8,
                                        (dest_pixel[1] as f32 * inv_alpha
                                            + src_pixel[1] as f32 * alpha)
                                            as u8,
                                        (dest_pixel[2] as f32 * inv_alpha
                                            + src_pixel[2] as f32 * alpha)
                                            as u8,
                                        255,
                                    ];

                                    self.set_pixel(dest_x, dest_y, blended);
                                }
                            } else {
                                self.set_pixel(dest_x, dest_y, src_pixel);
                            }
                        }
                    }
                }
            }
        }

        /// Calculate perceptual hash for similarity detection.
        pub fn perceptual_hash(&self) -> u64 {
            // Resize to 8x8 for hash calculation
            let small = self.resize_nearest(8, 8);

            // Convert to grayscale if not already
            let mut gray_values = Vec::new();
            for y in 0..8 {
                for x in 0..8 {
                    if let Some(pixel) = small.get_pixel(x, y) {
                        let gray = (0.299 * pixel[0] as f32
                            + 0.587 * pixel[1] as f32
                            + 0.114 * pixel[2] as f32) as u8;
                        gray_values.push(gray);
                    }
                }
            }

            // Calculate average
            let avg: u32 = gray_values.iter().map(|&v| v as u32).sum::<u32>() / 64;

            // Create hash
            let mut hash = 0u64;
            for (i, &value) in gray_values.iter().enumerate() {
                if value as u32 > avg {
                    hash |= 1 << i;
                }
            }

            hash
        }

        /// Calculate Hamming distance between two perceptual hashes.
        pub fn hash_distance(hash1: u64, hash2: u64) -> u32 {
            (hash1 ^ hash2).count_ones()
        }
    }
}

/// Video analysis utilities for intelligent frame selection.
#[allow(dead_code)]
pub mod analysis {

    /// Frame quality score (higher is better).
    pub struct FrameQuality {
        /// Sharpness score (0.0 - 1.0).
        pub sharpness: f64,

        /// Contrast score (0.0 - 1.0).
        pub contrast: f64,

        /// Brightness score (0.0 - 1.0).
        pub brightness: f64,

        /// Motion blur score (0.0 = no blur, 1.0 = heavy blur).
        pub motion_blur: f64,

        /// Overall quality score (weighted combination).
        pub overall: f64,
    }

    impl FrameQuality {
        /// Calculate overall quality score from individual metrics.
        pub fn calculate_overall(
            sharpness: f64,
            contrast: f64,
            brightness: f64,
            motion_blur: f64,
        ) -> f64 {
            // Weighted combination (sharpness and low motion blur are most important)
            let sharpness_weight = 0.4;
            let contrast_weight = 0.2;
            let brightness_weight = 0.1;
            let motion_blur_weight = 0.3;

            sharpness * sharpness_weight
                + contrast * contrast_weight
                + brightness * brightness_weight
                + (1.0 - motion_blur) * motion_blur_weight
        }

        /// Create a new frame quality assessment.
        pub fn new(sharpness: f64, contrast: f64, brightness: f64, motion_blur: f64) -> Self {
            let overall = Self::calculate_overall(sharpness, contrast, brightness, motion_blur);

            Self {
                sharpness,
                contrast,
                brightness,
                motion_blur,
                overall,
            }
        }

        /// Check if frame quality is acceptable for thumbnails.
        pub fn is_acceptable(&self) -> bool {
            self.overall >= 0.5 && self.motion_blur < 0.7
        }
    }

    /// Scene change detection result.
    pub struct SceneChange {
        /// Frame index where scene change occurs.
        pub frame_index: usize,

        /// Timestamp of scene change.
        pub timestamp: f64,

        /// Confidence score (0.0 - 1.0).
        pub confidence: f64,
    }

    /// Calculate histogram difference between two frames (for scene detection).
    pub fn histogram_difference(hist1: &[u32; 256], hist2: &[u32; 256]) -> f64 {
        let mut diff = 0.0;

        for i in 0..256 {
            let h1 = hist1[i] as f64;
            let h2 = hist2[i] as f64;
            diff += (h1 - h2).abs();
        }

        // Normalize to 0.0 - 1.0
        diff / (256.0 * 1000.0) // Assuming max pixel count of ~1000
    }

    /// Detect if two consecutive frames represent a scene change.
    pub fn is_scene_change(difference: f64, threshold: f64) -> bool {
        difference > threshold
    }
}
