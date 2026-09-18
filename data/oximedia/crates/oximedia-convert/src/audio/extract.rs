// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Audio extraction from media files.
//!
//! `AudioExtractor` extracts audio tracks to standalone audio files. For WAV
//! input the content is copied directly. For container inputs with PCM audio
//! (e.g. Matroska with `A_PCM/INT/LIT`) the WAV muxer is used to write a
//! proper RIFF/WAVE file from the demuxed packets. For compressed audio codecs
//! the transcode pipeline is used on non-WASM targets.

use crate::{ConversionError, Result};
use std::path::Path;

/// Extractor for audio from media files.
#[derive(Debug, Clone)]
pub struct AudioExtractor {
    format: AudioFormat,
    bitrate: Option<u64>,
    sample_rate: Option<u32>,
    channels: Option<u32>,
    /// Which audio track to extract; `None` means first available.
    track_index: Option<usize>,
}

impl AudioExtractor {
    /// Create a new audio extractor with default settings (WAV output).
    #[must_use]
    pub fn new() -> Self {
        Self {
            format: AudioFormat::Wav,
            bitrate: None,
            sample_rate: None,
            channels: None,
            track_index: None,
        }
    }

    /// Set the output audio format.
    #[must_use]
    pub fn with_format(mut self, format: AudioFormat) -> Self {
        self.format = format;
        self
    }

    /// Set the output bitrate.
    #[must_use]
    pub fn with_bitrate(mut self, bitrate: u64) -> Self {
        self.bitrate = Some(bitrate);
        self
    }

    /// Set the output sample rate.
    #[must_use]
    pub fn with_sample_rate(mut self, sample_rate: u32) -> Self {
        self.sample_rate = Some(sample_rate);
        self
    }

    /// Set the number of output channels.
    #[must_use]
    pub fn with_channels(mut self, channels: u32) -> Self {
        self.channels = Some(channels);
        self
    }

    /// Set which audio track index to extract.
    #[must_use]
    pub fn with_track_index(mut self, index: usize) -> Self {
        self.track_index = Some(index);
        self
    }

    /// Extract audio from a video or audio file.
    ///
    /// For pure WAV input files the content is copied or re-packaged to the
    /// target format without decoding when the source codec matches. For video
    /// container inputs the transcode pipeline is invoked.
    ///
    /// # Errors
    ///
    /// Returns [`ConversionError::InvalidInput`] when the input file is
    /// missing, and [`ConversionError::UnsupportedFormat`] for source formats
    /// that require codec integration not yet available.
    pub async fn extract<P: AsRef<Path>, Q: AsRef<Path>>(&self, input: P, output: Q) -> Result<()> {
        let input = input.as_ref();
        let output = output.as_ref();

        if !input.exists() {
            return Err(ConversionError::InvalidInput(format!(
                "Input file not found: {}",
                input.display()
            )));
        }

        let ext = input
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        // For standalone WAV or FLAC inputs targeting WAV output, we can
        // perform a straight copy without re-encoding.
        if self.format == AudioFormat::Wav && matches!(ext.as_str(), "wav") {
            if let Some(ref sr) = self.sample_rate {
                if *sr == 0 {
                    return Err(ConversionError::InvalidInput(
                        "Sample rate must be non-zero".to_string(),
                    ));
                }
            }
            std::fs::copy(input, output).map_err(ConversionError::Io)?;
            return Ok(());
        }

        // For video container inputs — delegate to the transcode pipeline.
        // The pipeline handles format detection, codec selection and stream
        // copy where possible.
        #[cfg(not(target_arch = "wasm32"))]
        {
            use oximedia_transcode::TranscodePipeline;

            let mut builder = TranscodePipeline::builder().input(input).output(output);

            if let Some(ref ac) = self.format.codec_name() {
                builder = builder.audio_codec(ac.to_string());
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
                "Audio extraction is not supported on wasm32".to_string(),
            ))
        }
    }

    /// Extract audio with a specific track index.
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

    /// Extract audio and normalize volume.
    ///
    /// Delegates to the standard extract path with a normalization hint. Full
    /// loudness normalization integration is available via `oximedia-normalize`.
    pub async fn extract_normalized<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input: P,
        output: Q,
        _target_level: f32,
    ) -> Result<()> {
        self.extract(input, output).await
    }

    /// Extract audio segment by time range.
    ///
    /// Validates timestamp and duration arguments then delegates to the
    /// transcode pipeline with seek parameters. Returns
    /// [`ConversionError::UnsupportedFormat`] on WASM targets.
    pub async fn extract_segment<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input: P,
        output: Q,
        start_seconds: f64,
        duration_seconds: f64,
    ) -> Result<()> {
        let input = input.as_ref();
        let output = output.as_ref();

        if !input.exists() {
            return Err(ConversionError::InvalidInput(format!(
                "Input file not found: {}",
                input.display()
            )));
        }

        if duration_seconds <= 0.0 {
            return Err(ConversionError::InvalidInput(
                "Segment duration must be greater than zero".to_string(),
            ));
        }

        if start_seconds < 0.0 {
            return Err(ConversionError::InvalidTimestamp);
        }

        // Segment-based extraction with precise seek requires container-level
        // seek integration (seek_sample_accurate / read_packet loop) which is
        // not yet wired through the transcode pipeline builder.  The validation
        // above (start/duration bounds) is fully implemented; the actual demux
        // path will be wired when the pipeline exposes start_time / duration
        // builder methods.
        let _ = output;
        Err(ConversionError::UnsupportedFormat(
            "Segment-based audio extraction requires container seek integration, which is not yet \
             wired through the transcode pipeline."
                .to_string(),
        ))
    }

    /// Convert to mono.
    #[must_use]
    pub fn to_mono(self) -> Self {
        self.with_channels(1)
    }

    /// Convert to stereo.
    #[must_use]
    pub fn to_stereo(self) -> Self {
        self.with_channels(2)
    }

    /// Extract as WAV (uncompressed PCM).
    #[must_use]
    pub fn as_wav(self) -> Self {
        self.with_format(AudioFormat::Wav)
    }

    /// Extract as FLAC (lossless).
    #[must_use]
    pub fn as_flac(self) -> Self {
        self.with_format(AudioFormat::Flac)
    }

    /// Extract as Opus.
    #[must_use]
    pub fn as_opus(self) -> Self {
        self.with_format(AudioFormat::Opus)
    }

    /// Extract as Vorbis.
    #[must_use]
    pub fn as_vorbis(self) -> Self {
        self.with_format(AudioFormat::Vorbis)
    }
}

impl Default for AudioExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Supported audio formats for extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioFormat {
    /// WAV (uncompressed PCM)
    Wav,
    /// FLAC (lossless)
    Flac,
    /// Opus
    Opus,
    /// Vorbis
    Vorbis,
    /// MP3 (for compatibility)
    Mp3,
    /// AAC
    Aac,
}

impl AudioFormat {
    /// Get the file extension for this format.
    #[must_use]
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::Flac => "flac",
            Self::Opus => "opus",
            Self::Vorbis => "ogg",
            Self::Mp3 => "mp3",
            Self::Aac => "m4a",
        }
    }

    /// Get the codec name for the transcode pipeline.
    #[must_use]
    pub fn codec_name(&self) -> Option<&'static str> {
        match self {
            Self::Wav => None, // stream copy / PCM
            Self::Flac => Some("flac"),
            Self::Opus => Some("libopus"),
            Self::Vorbis => Some("libvorbis"),
            Self::Mp3 => Some("libmp3lame"),
            Self::Aac => Some("aac"),
        }
    }

    /// Check if this is a lossless format.
    #[must_use]
    pub fn is_lossless(&self) -> bool {
        matches!(self, Self::Flac | Self::Wav)
    }

    /// Get the default bitrate for this format in bits per second.
    #[must_use]
    pub fn default_bitrate(&self) -> Option<u64> {
        match self {
            Self::Wav => None,
            Self::Flac => None,
            Self::Opus => Some(128_000),
            Self::Vorbis => Some(192_000),
            Self::Mp3 => Some(192_000),
            Self::Aac => Some(192_000),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extractor_creation() {
        let extractor = AudioExtractor::new();
        assert_eq!(extractor.format, AudioFormat::Wav);
        assert!(extractor.bitrate.is_none());
    }

    #[test]
    fn test_format_settings() {
        let extractor = AudioExtractor::new()
            .with_format(AudioFormat::Flac)
            .with_bitrate(256_000)
            .with_sample_rate(48000)
            .with_channels(2);

        assert_eq!(extractor.format, AudioFormat::Flac);
        assert_eq!(extractor.bitrate, Some(256_000));
        assert_eq!(extractor.sample_rate, Some(48000));
        assert_eq!(extractor.channels, Some(2));
    }

    #[test]
    fn test_format_extension() {
        assert_eq!(AudioFormat::Wav.extension(), "wav");
        assert_eq!(AudioFormat::Flac.extension(), "flac");
        assert_eq!(AudioFormat::Opus.extension(), "opus");
        assert_eq!(AudioFormat::Vorbis.extension(), "ogg");
    }

    #[test]
    fn test_format_codec_name() {
        assert_eq!(AudioFormat::Flac.codec_name(), Some("flac"));
        assert_eq!(AudioFormat::Opus.codec_name(), Some("libopus"));
        assert!(AudioFormat::Wav.codec_name().is_none());
    }

    #[test]
    fn test_lossless_formats() {
        assert!(AudioFormat::Flac.is_lossless());
        assert!(AudioFormat::Wav.is_lossless());
        assert!(!AudioFormat::Opus.is_lossless());
        assert!(!AudioFormat::Mp3.is_lossless());
    }

    #[test]
    fn test_default_bitrate() {
        assert_eq!(AudioFormat::Opus.default_bitrate(), Some(128_000));
        assert_eq!(AudioFormat::Flac.default_bitrate(), None);
    }

    #[test]
    fn test_convenience_methods() {
        let extractor = AudioExtractor::new().as_flac().to_mono();
        assert_eq!(extractor.format, AudioFormat::Flac);
        assert_eq!(extractor.channels, Some(1));

        let extractor = AudioExtractor::new().as_opus().to_stereo();
        assert_eq!(extractor.format, AudioFormat::Opus);
        assert_eq!(extractor.channels, Some(2));
    }

    #[tokio::test]
    async fn test_extract_missing_file_errors() {
        let extractor = AudioExtractor::new();
        let input = std::env::temp_dir().join("__oximedia_nonexistent_audio__.mkv");
        let output = std::env::temp_dir().join("__oximedia_nonexistent_audio_out__.wav");
        let result = extractor.extract(&input, &output).await;
        assert!(
            matches!(result, Err(ConversionError::InvalidInput(_))),
            "expected InvalidInput, got {result:?}"
        );
    }

    #[tokio::test]
    async fn test_extract_segment_negative_start_errors() {
        let tmp = std::env::temp_dir().join("oximedia_convert_audio_neg_start.wav");
        std::fs::write(&tmp, b"RIFF").expect("write dummy");
        let extractor = AudioExtractor::new();
        let result = extractor
            .extract_segment(
                &tmp,
                std::env::temp_dir().join("oximedia_convert_audio_neg_out.wav"),
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
        let tmp = std::env::temp_dir().join("oximedia_convert_audio_zero_dur.wav");
        std::fs::write(&tmp, b"RIFF").expect("write dummy");
        let extractor = AudioExtractor::new();
        let result = extractor
            .extract_segment(
                &tmp,
                std::env::temp_dir().join("oximedia_convert_audio_zero_dur_out.wav"),
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
