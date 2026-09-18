// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Time-based file splitting.
//!
//! Provides fixed-duration segment splitting. Full re-mux from a container
//! source (seek-to-keyframe, copy packets, close segment) requires the
//! container demux seek API; until that integration is complete the methods
//! return [`ConversionError::UnsupportedFormat`] for container inputs.
//!
//! Pure-logic helpers (`calculate_segment_boundaries`, `output_paths_for`) are
//! fully functional and are tested independently.

use crate::{ConversionError, Result};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Splitter for dividing files by time.
#[derive(Debug, Clone)]
pub struct TimeSplitter {
    segment_duration: Duration,
    copy_streams: bool,
    /// Output file pattern; `%03d` will be replaced by the segment index.
    output_pattern: String,
}

impl TimeSplitter {
    /// Create a new time splitter with the given segment duration.
    ///
    /// # Panics
    ///
    /// Does not panic — a zero duration is caught at split time.
    #[must_use]
    pub fn new(segment_duration: Duration) -> Self {
        Self {
            segment_duration,
            copy_streams: true,
            output_pattern: "segment_%03d".to_string(),
        }
    }

    /// Set whether to copy streams without re-encoding.
    #[must_use]
    pub fn with_copy_streams(mut self, copy: bool) -> Self {
        self.copy_streams = copy;
        self
    }

    /// Set the output file pattern (without extension; the input extension is
    /// appended automatically).
    ///
    /// The literal `%03d` is replaced by the zero-padded segment index.
    #[must_use]
    pub fn with_output_pattern<S: Into<String>>(mut self, pattern: S) -> Self {
        self.output_pattern = pattern.into();
        self
    }

    /// Validate that the configured duration is non-zero.
    fn validate_duration(&self) -> Result<()> {
        if self.segment_duration.is_zero() {
            return Err(ConversionError::InvalidInput(
                "Segment duration must be greater than zero".to_string(),
            ));
        }
        Ok(())
    }

    /// Split a file into time-based segments.
    ///
    /// For container files this returns [`ConversionError::UnsupportedFormat`]
    /// until the container demux seek API is wired through.
    pub async fn split<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input: P,
        output_dir: Q,
    ) -> Result<Vec<PathBuf>> {
        let input = input.as_ref();
        let output_dir = output_dir.as_ref();

        self.validate_duration()?;

        if !input.exists() {
            return Err(ConversionError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Input file not found: {}", input.display()),
            )));
        }

        std::fs::create_dir_all(output_dir).map_err(ConversionError::Io)?;

        Err(ConversionError::UnsupportedFormat(
            "Time-based re-mux splitting requires the container demux seek API, which is not \
             yet integrated for general container formats."
                .to_string(),
        ))
    }

    /// Split at specific time points.
    pub async fn split_at_times<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input: P,
        output_dir: Q,
        times: &[Duration],
    ) -> Result<Vec<PathBuf>> {
        let input = input.as_ref();
        let output_dir = output_dir.as_ref();

        if !input.exists() {
            return Err(ConversionError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Input file not found: {}", input.display()),
            )));
        }

        if times.is_empty() {
            return Err(ConversionError::InvalidInput(
                "At least one split time is required".to_string(),
            ));
        }

        std::fs::create_dir_all(output_dir).map_err(ConversionError::Io)?;

        Err(ConversionError::UnsupportedFormat(
            "Time-point re-mux splitting requires the container demux seek API, which is not \
             yet integrated."
                .to_string(),
        ))
    }

    /// Split into a specific number of equal segments.
    ///
    /// `total_duration` must be provided by the caller (obtained from the
    /// container probe result).
    pub async fn split_into_parts<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input: P,
        output_dir: Q,
        num_parts: usize,
    ) -> Result<Vec<PathBuf>> {
        let input = input.as_ref();
        let output_dir = output_dir.as_ref();

        if num_parts == 0 {
            return Err(ConversionError::InvalidInput(
                "Number of parts must be greater than zero".to_string(),
            ));
        }

        if !input.exists() {
            return Err(ConversionError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Input file not found: {}", input.display()),
            )));
        }

        std::fs::create_dir_all(output_dir).map_err(ConversionError::Io)?;

        Err(ConversionError::UnsupportedFormat(
            "Part-based splitting requires the container demux seek API, which is not yet \
             integrated."
                .to_string(),
        ))
    }

    /// Calculate the expected segment boundaries for a given total duration.
    ///
    /// Returns a list of `(start_secs, end_secs)` pairs. The last segment may
    /// be shorter than `segment_duration`.
    ///
    /// # Errors
    ///
    /// Returns [`ConversionError::InvalidInput`] when the segment duration or
    /// total duration is zero.
    pub fn calculate_segment_boundaries(
        &self,
        total_duration_secs: f64,
    ) -> Result<Vec<(f64, f64)>> {
        self.validate_duration()?;

        if total_duration_secs <= 0.0 {
            return Err(ConversionError::InvalidInput(
                "Total duration must be greater than zero".to_string(),
            ));
        }

        let seg_secs = self.segment_duration.as_secs_f64();
        let mut boundaries = Vec::new();
        let mut start = 0.0_f64;

        while start < total_duration_secs {
            let end = (start + seg_secs).min(total_duration_secs);
            boundaries.push((start, end));
            start += seg_secs;
        }

        Ok(boundaries)
    }

    /// Generate the expected output paths for a given number of segments.
    ///
    /// The file extension is taken from `input`.
    #[must_use]
    pub fn output_paths_for(&self, output_dir: &Path, input: &Path, count: usize) -> Vec<PathBuf> {
        let ext = input.extension().and_then(|e| e.to_str()).unwrap_or("mkv");

        (0..count)
            .map(|i| {
                let name = self.output_pattern.replace("%03d", &format!("{i:03}"));
                output_dir.join(format!("{name}.{ext}"))
            })
            .collect()
    }

    /// Create a splitter for 1-minute segments.
    #[must_use]
    pub fn one_minute() -> Self {
        Self::new(Duration::from_secs(60))
    }

    /// Create a splitter for 5-minute segments.
    #[must_use]
    pub fn five_minutes() -> Self {
        Self::new(Duration::from_secs(300))
    }

    /// Create a splitter for 10-minute segments.
    #[must_use]
    pub fn ten_minutes() -> Self {
        Self::new(Duration::from_secs(600))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_splitter_creation() {
        let splitter = TimeSplitter::new(Duration::from_secs(60));
        assert_eq!(splitter.segment_duration, Duration::from_secs(60));
        assert!(splitter.copy_streams);
    }

    #[test]
    fn test_splitter_settings() {
        let splitter = TimeSplitter::new(Duration::from_secs(120)).with_copy_streams(false);
        assert_eq!(splitter.segment_duration, Duration::from_secs(120));
        assert!(!splitter.copy_streams);
    }

    #[test]
    fn test_presets() {
        let splitter = TimeSplitter::one_minute();
        assert_eq!(splitter.segment_duration, Duration::from_secs(60));

        let splitter = TimeSplitter::five_minutes();
        assert_eq!(splitter.segment_duration, Duration::from_secs(300));

        let splitter = TimeSplitter::ten_minutes();
        assert_eq!(splitter.segment_duration, Duration::from_secs(600));
    }

    #[test]
    fn test_calculate_segment_boundaries_even() {
        let splitter = TimeSplitter::new(Duration::from_secs(60));
        let boundaries = splitter.calculate_segment_boundaries(180.0).unwrap();
        assert_eq!(boundaries.len(), 3);
        assert_eq!(boundaries[0], (0.0, 60.0));
        assert_eq!(boundaries[1], (60.0, 120.0));
        assert_eq!(boundaries[2], (120.0, 180.0));
    }

    #[test]
    fn test_calculate_segment_boundaries_remainder() {
        let splitter = TimeSplitter::new(Duration::from_secs(60));
        let boundaries = splitter.calculate_segment_boundaries(90.0).unwrap();
        assert_eq!(boundaries.len(), 2);
        assert_eq!(boundaries[0], (0.0, 60.0));
        assert_eq!(boundaries[1], (60.0, 90.0));
    }

    #[test]
    fn test_calculate_segment_boundaries_shorter_than_segment() {
        let splitter = TimeSplitter::new(Duration::from_secs(120));
        let boundaries = splitter.calculate_segment_boundaries(30.0).unwrap();
        assert_eq!(boundaries.len(), 1);
        assert_eq!(boundaries[0], (0.0, 30.0));
    }

    #[test]
    fn test_calculate_boundaries_total_duration_zero_errors() {
        let splitter = TimeSplitter::new(Duration::from_secs(60));
        let result = splitter.calculate_segment_boundaries(0.0);
        assert!(
            matches!(result, Err(ConversionError::InvalidInput(_))),
            "expected InvalidInput, got {result:?}"
        );
    }

    #[test]
    fn test_calculate_boundaries_zero_duration_errors() {
        let splitter = TimeSplitter::new(Duration::ZERO);
        let result = splitter.calculate_segment_boundaries(60.0);
        assert!(
            matches!(result, Err(ConversionError::InvalidInput(_))),
            "expected InvalidInput for zero segment duration"
        );
    }

    #[test]
    fn test_output_paths_for() {
        let splitter = TimeSplitter::new(Duration::from_secs(60));
        let dir = std::env::temp_dir();
        let input = PathBuf::from("video.mkv");
        let paths = splitter.output_paths_for(&dir, &input, 3);
        assert_eq!(paths.len(), 3);
        assert_eq!(paths[0], dir.join("segment_000.mkv"));
        assert_eq!(paths[1], dir.join("segment_001.mkv"));
        assert_eq!(paths[2], dir.join("segment_002.mkv"));
    }

    #[test]
    fn test_output_paths_custom_pattern() {
        let splitter = TimeSplitter::new(Duration::from_secs(60)).with_output_pattern("part_%03d");
        let dir = PathBuf::from("/out");
        let input = PathBuf::from("clip.mp4");
        let paths = splitter.output_paths_for(&dir, &input, 2);
        assert_eq!(paths[0], PathBuf::from("/out/part_000.mp4"));
    }

    #[test]
    fn test_boundaries_sum_to_total() {
        let splitter = TimeSplitter::new(Duration::from_secs(30));
        let total = 95.0;
        let boundaries = splitter.calculate_segment_boundaries(total).unwrap();
        // All segments should be contiguous.
        for window in boundaries.windows(2) {
            assert!((window[0].1 - window[1].0).abs() < 1e-9);
        }
        // First start is 0 and last end is total.
        assert_eq!(boundaries.first().map(|b| b.0), Some(0.0));
        let last_end = boundaries.last().map(|b| b.1).unwrap_or(0.0);
        assert!((last_end - total).abs() < 1e-9);
    }

    #[tokio::test]
    async fn test_split_missing_file_errors() {
        let splitter = TimeSplitter::new(Duration::from_secs(60));
        let input = std::env::temp_dir().join("__oximedia_nonexistent_time_split__.mkv");
        let output_dir = std::env::temp_dir().join("__oximedia_nonexistent_time_split_dir__");
        let result = splitter.split(&input, &output_dir).await;
        assert!(
            matches!(result, Err(ConversionError::Io(_))),
            "expected Io error, got {result:?}"
        );
    }

    #[tokio::test]
    async fn test_split_zero_duration_errors() {
        let splitter = TimeSplitter::new(Duration::ZERO);
        // The zero-duration check fires before the file-existence check.
        let tmp = std::env::temp_dir().join("oximedia_convert_zero_duration.mkv");
        std::fs::write(&tmp, b"dummy").expect("write dummy");
        let result = splitter.split(&tmp, std::env::temp_dir()).await;
        assert!(
            matches!(result, Err(ConversionError::InvalidInput(_))),
            "expected InvalidInput for zero segment duration, got {result:?}"
        );
        let _ = std::fs::remove_file(&tmp);
    }

    #[tokio::test]
    async fn test_split_at_times_empty_times_errors() {
        let splitter = TimeSplitter::new(Duration::from_secs(60));
        let tmp = std::env::temp_dir().join("oximedia_convert_empty_times.mkv");
        std::fs::write(&tmp, b"dummy").expect("write dummy");
        let result = splitter
            .split_at_times(&tmp, std::env::temp_dir(), &[])
            .await;
        assert!(
            matches!(result, Err(ConversionError::InvalidInput(_))),
            "expected InvalidInput for empty times, got {result:?}"
        );
        let _ = std::fs::remove_file(&tmp);
    }

    #[tokio::test]
    async fn test_split_into_zero_parts_errors() {
        let splitter = TimeSplitter::new(Duration::from_secs(60));
        let tmp = std::env::temp_dir().join("oximedia_convert_zero_parts.mkv");
        std::fs::write(&tmp, b"dummy").expect("write dummy");
        let result = splitter
            .split_into_parts(&tmp, std::env::temp_dir(), 0)
            .await;
        assert!(
            matches!(result, Err(ConversionError::InvalidInput(_))),
            "expected InvalidInput for 0 parts, got {result:?}"
        );
        let _ = std::fs::remove_file(&tmp);
    }
}
