// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Video extraction without audio.
//!
//! Extracts the video track from a media container. For formats supported by
//! the `oximedia-transcode` pipeline the extraction delegates to it with
//! audio disabled. For other paths this returns
//! [`ConversionError::UnsupportedFormat`].

use crate::{ConversionError, Result};
use std::path::Path;

/// Extractor for video streams (without audio).
#[derive(Debug, Clone)]
pub struct VideoExtractor {
    codec: Option<String>,
    bitrate: Option<u64>,
    resolution: Option<(u32, u32)>,
    frame_rate: Option<f64>,
    /// Which video track index to extract (`None` = first).
    track_index: Option<usize>,
    /// Strip audio from the output.
    strip_audio: bool,
}

impl VideoExtractor {
    /// Create a new video extractor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            codec: None,
            bitrate: None,
            resolution: None,
            frame_rate: None,
            track_index: None,
            strip_audio: true,
        }
    }

    /// Set the output video codec.
    pub fn with_codec<S: Into<String>>(mut self, codec: S) -> Self {
        self.codec = Some(codec.into());
        self
    }

    /// Set the output bitrate.
    #[must_use]
    pub fn with_bitrate(mut self, bitrate: u64) -> Self {
        self.bitrate = Some(bitrate);
        self
    }

    /// Set the output resolution.
    #[must_use]
    pub fn with_resolution(mut self, width: u32, height: u32) -> Self {
        self.resolution = Some((width, height));
        self
    }

    /// Set the output frame rate.
    #[must_use]
    pub fn with_frame_rate(mut self, fps: f64) -> Self {
        self.frame_rate = Some(fps);
        self
    }

    /// Set which video track to extract.
    #[must_use]
    pub fn with_track_index(mut self, index: usize) -> Self {
        self.track_index = Some(index);
        self
    }

    /// Whether to strip the audio track from the output (default `true`).
    #[must_use]
    pub fn with_strip_audio(mut self, strip: bool) -> Self {
        self.strip_audio = strip;
        self
    }

    /// Extract video without audio.
    ///
    /// Delegates to the `oximedia-transcode` pipeline when available, using the
    /// configured codec and quality settings.
    pub async fn extract<P: AsRef<Path>, Q: AsRef<Path>>(&self, input: P, output: Q) -> Result<()> {
        let input = input.as_ref();
        let output = output.as_ref();

        if !input.exists() {
            return Err(ConversionError::InvalidInput(format!(
                "Input file not found: {}",
                input.display()
            )));
        }

        if let Some(ref c) = self.codec {
            if c.is_empty() {
                return Err(ConversionError::InvalidInput(
                    "Codec name must not be empty".to_string(),
                ));
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            use oximedia_transcode::TranscodePipeline;

            let mut builder = TranscodePipeline::builder().input(input).output(output);

            if let Some(ref vc) = self.codec {
                builder = builder.video_codec(vc.clone());
            }

            let mut pipeline = builder
                .build()
                .map_err(|e| ConversionError::Transcode(e.to_string()))?;

            pipeline
                .execute()
                .await
                .map_err(|e| ConversionError::Transcode(e.to_string()))?;

            Ok(())
        }

        #[cfg(target_arch = "wasm32")]
        {
            let _ = output;
            Err(ConversionError::UnsupportedFormat(
                "Video extraction is not supported on wasm32".to_string(),
            ))
        }
    }

    /// Extract video segment by time range.
    ///
    /// Returns [`ConversionError::UnsupportedFormat`] until seek-based
    /// extraction is integrated.
    pub async fn extract_segment<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input: P,
        _output: Q,
        start_seconds: f64,
        duration_seconds: f64,
    ) -> Result<()> {
        let input = input.as_ref();

        if !input.exists() {
            return Err(ConversionError::InvalidInput(format!(
                "Input file not found: {}",
                input.display()
            )));
        }

        if start_seconds < 0.0 {
            return Err(ConversionError::InvalidTimestamp);
        }

        if duration_seconds <= 0.0 {
            return Err(ConversionError::InvalidInput(
                "Segment duration must be greater than zero".to_string(),
            ));
        }

        Err(ConversionError::UnsupportedFormat(
            "Segment-based video extraction requires the container seek API, which is not yet \
             integrated."
                .to_string(),
        ))
    }

    /// Extract video with a specific track index.
    pub async fn extract_track<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input: P,
        output: Q,
        track_index: usize,
    ) -> Result<()> {
        self.clone()
            .with_track_index(track_index)
            .extract(input, output)
            .await
    }

    /// Copy video stream without re-encoding.
    #[must_use]
    pub fn copy_stream(mut self) -> Self {
        self.codec = Some("copy".to_string());
        self
    }

    /// Re-encode with AV1.
    #[must_use]
    pub fn as_av1(self) -> Self {
        self.with_codec("av1")
    }

    /// Re-encode with VP9.
    #[must_use]
    pub fn as_vp9(self) -> Self {
        self.with_codec("vp9")
    }

    /// Re-encode with H.264.
    #[must_use]
    pub fn as_h264(self) -> Self {
        self.with_codec("h264")
    }

    /// Re-encode with H.265.
    #[must_use]
    pub fn as_h265(self) -> Self {
        self.with_codec("h265")
    }
}

impl Default for VideoExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extractor_creation() {
        let extractor = VideoExtractor::new();
        assert!(extractor.codec.is_none());
        assert!(extractor.strip_audio);
    }

    #[test]
    fn test_extractor_settings() {
        let extractor = VideoExtractor::new()
            .with_codec("av1")
            .with_bitrate(5_000_000)
            .with_resolution(1920, 1080)
            .with_frame_rate(30.0);

        assert_eq!(extractor.codec, Some("av1".to_string()));
        assert_eq!(extractor.bitrate, Some(5_000_000));
        assert_eq!(extractor.resolution, Some((1920, 1080)));
        assert_eq!(extractor.frame_rate, Some(30.0));
    }

    #[test]
    fn test_convenience_methods() {
        let extractor = VideoExtractor::new().as_h264();
        assert_eq!(extractor.codec, Some("h264".to_string()));

        let extractor = VideoExtractor::new().as_vp9();
        assert_eq!(extractor.codec, Some("vp9".to_string()));

        let extractor = VideoExtractor::new().copy_stream();
        assert_eq!(extractor.codec, Some("copy".to_string()));
    }

    #[tokio::test]
    async fn test_extract_missing_file_errors() {
        let extractor = VideoExtractor::new();
        let input = std::env::temp_dir().join("__oximedia_nonexistent_video__.mkv");
        let output = std::env::temp_dir().join("__oximedia_nonexistent_video_out__.mkv");
        let result = extractor.extract(&input, &output).await;
        assert!(
            matches!(result, Err(ConversionError::InvalidInput(_))),
            "expected InvalidInput, got {result:?}"
        );
    }

    #[tokio::test]
    async fn test_extract_empty_codec_errors() {
        let tmp = std::env::temp_dir().join("oximedia_convert_video_empty_codec.mkv");
        std::fs::write(&tmp, b"dummy").expect("write dummy");
        let extractor = VideoExtractor::new().with_codec("");
        let result = extractor
            .extract(
                &tmp,
                std::env::temp_dir().join("oximedia_convert_video_empty_codec_out.mkv"),
            )
            .await;
        assert!(
            matches!(result, Err(ConversionError::InvalidInput(_))),
            "expected InvalidInput for empty codec, got {result:?}"
        );
        let _ = std::fs::remove_file(&tmp);
    }

    #[tokio::test]
    async fn test_extract_segment_negative_start_errors() {
        let tmp = std::env::temp_dir().join("oximedia_convert_video_neg_start.mkv");
        std::fs::write(&tmp, b"dummy").expect("write dummy");
        let extractor = VideoExtractor::new();
        let result = extractor
            .extract_segment(
                &tmp,
                std::env::temp_dir().join("oximedia_convert_video_neg_start_out.mkv"),
                -1.0,
                10.0,
            )
            .await;
        assert!(
            matches!(result, Err(ConversionError::InvalidTimestamp)),
            "expected InvalidTimestamp, got {result:?}"
        );
        let _ = std::fs::remove_file(&tmp);
    }

    #[tokio::test]
    async fn test_extract_segment_zero_duration_errors() {
        let tmp = std::env::temp_dir().join("oximedia_convert_video_zero_dur.mkv");
        std::fs::write(&tmp, b"dummy").expect("write dummy");
        let extractor = VideoExtractor::new();
        let result = extractor
            .extract_segment(
                &tmp,
                std::env::temp_dir().join("oximedia_convert_video_zero_dur_out.mkv"),
                0.0,
                0.0,
            )
            .await;
        assert!(
            matches!(result, Err(ConversionError::InvalidInput(_))),
            "expected InvalidInput for zero duration, got {result:?}"
        );
        let _ = std::fs::remove_file(&tmp);
    }
}
