// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Sprite sheet generation for video thumbnails.

use crate::Result;
use std::path::Path;

/// Generator for thumbnail sprite sheets.
#[derive(Debug, Clone)]
pub struct SpriteSheetGenerator {
    thumbnail_width: u32,
    thumbnail_height: u32,
    columns: usize,
    rows: usize,
    interval_seconds: f64,
}

impl SpriteSheetGenerator {
    /// Create a new sprite sheet generator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            thumbnail_width: 160,
            thumbnail_height: 90,
            columns: 10,
            rows: 10,
            interval_seconds: 1.0,
        }
    }

    /// Set the size of individual thumbnails.
    #[must_use]
    pub fn with_thumbnail_size(mut self, width: u32, height: u32) -> Self {
        self.thumbnail_width = width;
        self.thumbnail_height = height;
        self
    }

    /// Set the grid layout (columns x rows).
    #[must_use]
    pub fn with_grid(mut self, columns: usize, rows: usize) -> Self {
        self.columns = columns;
        self.rows = rows;
        self
    }

    /// Set the time interval between thumbnails.
    #[must_use]
    pub fn with_interval(mut self, seconds: f64) -> Self {
        self.interval_seconds = seconds;
        self
    }

    /// Generate a sprite sheet from a video.
    pub async fn generate<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input: P,
        output: Q,
    ) -> Result<SpriteSheetInfo> {
        let _input = input.as_ref();
        let _output = output.as_ref();

        // Placeholder for actual generation
        Ok(SpriteSheetInfo {
            width: self.thumbnail_width * self.columns as u32,
            height: self.thumbnail_height * self.rows as u32,
            thumbnail_width: self.thumbnail_width,
            thumbnail_height: self.thumbnail_height,
            columns: self.columns,
            rows: self.rows,
            count: self.columns * self.rows,
            interval_seconds: self.interval_seconds,
        })
    }

    /// Generate multiple sprite sheets if needed.
    pub async fn generate_multiple<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input: P,
        output_dir: Q,
    ) -> Result<Vec<SpriteSheetInfo>> {
        let _input = input.as_ref();
        let _output_dir = output_dir.as_ref();

        // Placeholder
        Ok(Vec::new())
    }

    /// Calculate the number of sprite sheets needed for a video.
    #[must_use]
    pub fn calculate_sheet_count(&self, duration_seconds: f64) -> usize {
        let thumbnails_per_sheet = self.columns * self.rows;
        let total_thumbnails = (duration_seconds / self.interval_seconds).ceil() as usize;
        total_thumbnails.div_ceil(thumbnails_per_sheet)
    }

    /// Create a preset for video seeking (10x10 grid, 1 second intervals).
    #[must_use]
    pub fn for_seeking() -> Self {
        Self::new().with_grid(10, 10).with_interval(1.0)
    }

    /// Create a preset for video preview (5x5 grid, 5 second intervals).
    #[must_use]
    pub fn for_preview() -> Self {
        Self::new()
            .with_grid(5, 5)
            .with_interval(5.0)
            .with_thumbnail_size(320, 180)
    }
}

impl Default for SpriteSheetGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Information about a generated sprite sheet.
#[derive(Debug, Clone)]
pub struct SpriteSheetInfo {
    /// Total width of the sprite sheet
    pub width: u32,
    /// Total height of the sprite sheet
    pub height: u32,
    /// Width of each thumbnail
    pub thumbnail_width: u32,
    /// Height of each thumbnail
    pub thumbnail_height: u32,
    /// Number of columns
    pub columns: usize,
    /// Number of rows
    pub rows: usize,
    /// Total number of thumbnails
    pub count: usize,
    /// Time interval between thumbnails
    pub interval_seconds: f64,
}

impl SpriteSheetInfo {
    /// Get the position (x, y) of a thumbnail by index.
    #[must_use]
    pub fn thumbnail_position(&self, index: usize) -> Option<(u32, u32)> {
        if index >= self.count {
            return None;
        }

        let row = index / self.columns;
        let col = index % self.columns;

        Some((
            col as u32 * self.thumbnail_width,
            row as u32 * self.thumbnail_height,
        ))
    }

    /// Get the time in seconds for a thumbnail by index.
    #[must_use]
    pub fn thumbnail_time(&self, index: usize) -> f64 {
        index as f64 * self.interval_seconds
    }

    /// Find the thumbnail index for a given time.
    #[must_use]
    pub fn index_for_time(&self, time_seconds: f64) -> usize {
        (time_seconds / self.interval_seconds).floor() as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generator_creation() {
        let gen = SpriteSheetGenerator::new();
        assert_eq!(gen.columns, 10);
        assert_eq!(gen.rows, 10);
    }

    #[test]
    fn test_generator_settings() {
        let gen = SpriteSheetGenerator::new()
            .with_thumbnail_size(200, 150)
            .with_grid(5, 5)
            .with_interval(2.0);

        assert_eq!(gen.thumbnail_width, 200);
        assert_eq!(gen.thumbnail_height, 150);
        assert_eq!(gen.columns, 5);
        assert_eq!(gen.rows, 5);
        assert_eq!(gen.interval_seconds, 2.0);
    }

    #[test]
    fn test_calculate_sheet_count() {
        let gen = SpriteSheetGenerator::new()
            .with_grid(10, 10)
            .with_interval(1.0);

        // 100 thumbnails per sheet, 150 seconds = 2 sheets
        assert_eq!(gen.calculate_sheet_count(150.0), 2);

        // 100 thumbnails per sheet, 100 seconds = 1 sheet
        assert_eq!(gen.calculate_sheet_count(100.0), 1);
    }

    #[test]
    fn test_presets() {
        let gen = SpriteSheetGenerator::for_seeking();
        assert_eq!(gen.columns, 10);
        assert_eq!(gen.interval_seconds, 1.0);

        let gen = SpriteSheetGenerator::for_preview();
        assert_eq!(gen.columns, 5);
        assert_eq!(gen.interval_seconds, 5.0);
    }

    #[test]
    fn test_sprite_info_position() {
        let info = SpriteSheetInfo {
            width: 1000,
            height: 1000,
            thumbnail_width: 100,
            thumbnail_height: 100,
            columns: 10,
            rows: 10,
            count: 100,
            interval_seconds: 1.0,
        };

        assert_eq!(info.thumbnail_position(0), Some((0, 0)));
        assert_eq!(info.thumbnail_position(5), Some((500, 0)));
        assert_eq!(info.thumbnail_position(15), Some((500, 100)));
        assert_eq!(info.thumbnail_position(100), None);
    }

    #[test]
    fn test_sprite_info_time() {
        let info = SpriteSheetInfo {
            width: 1000,
            height: 1000,
            thumbnail_width: 100,
            thumbnail_height: 100,
            columns: 10,
            rows: 10,
            count: 100,
            interval_seconds: 2.0,
        };

        assert_eq!(info.thumbnail_time(0), 0.0);
        assert_eq!(info.thumbnail_time(5), 10.0);
        assert_eq!(info.index_for_time(10.0), 5);
    }
}
