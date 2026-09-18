// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Video muting — remove or replace audio tracks.
//!
//! The `VideoMuter` can either strip audio entirely (producing a video-only
//! output) or replace the original audio with synthesized silence. Full
//! packet-level muting (copy video packets, replace audio with zero-energy
//! PCM) requires the container demux/mux APIs to be wired together; until
//! that integration is complete both `mute` and `replace_with_silence`
//! return [`ConversionError::UnsupportedFormat`] for container inputs.

use crate::{ConversionError, Result};
use std::path::Path;

/// Muter for video files — removes or replaces audio streams.
#[derive(Debug, Clone)]
pub struct VideoMuter {
    /// Copy video stream without re-encoding.
    copy_video: bool,
    /// Replace audio with silence instead of dropping it.
    silence_mode: bool,
}

impl VideoMuter {
    /// Create a new video muter.
    #[must_use]
    pub fn new() -> Self {
        Self {
            copy_video: true,
            silence_mode: false,
        }
    }

    /// Set whether to copy the video stream without re-encoding.
    #[must_use]
    pub fn with_copy_video(mut self, copy: bool) -> Self {
        self.copy_video = copy;
        self
    }

    /// Enable silence-replacement mode: keep the audio track structure but
    /// fill it with zero-energy PCM samples.
    #[must_use]
    pub fn with_silence_mode(mut self, silence: bool) -> Self {
        self.silence_mode = silence;
        self
    }

    /// Mute a video file by removing all audio tracks.
    ///
    /// The output file contains only the video stream (and any subtitle or data
    /// streams). For WASM targets this always returns `UnsupportedFormat`.
    ///
    /// # Errors
    ///
    /// Returns [`ConversionError::InvalidInput`] when the input is missing and
    /// [`ConversionError::UnsupportedFormat`] until full packet-level muxing is
    /// integrated.
    pub async fn mute<P: AsRef<Path>, Q: AsRef<Path>>(&self, input: P, output: Q) -> Result<()> {
        let input = input.as_ref();
        let output = output.as_ref();

        if !input.exists() {
            return Err(ConversionError::InvalidInput(format!(
                "Input file not found: {}",
                input.display()
            )));
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            use oximedia_transcode::TranscodePipeline;

            // Build a transcode pipeline with no audio codec — the pipeline
            // will drop audio streams when no audio encoder is specified.
            let mut builder = TranscodePipeline::builder().input(input).output(output);

            if self.copy_video {
                builder = builder.video_codec("copy".to_string());
            }

            // Explicitly set empty audio codec to signal no audio output.
            // The pipeline interprets this as "strip audio".
            builder = builder.audio_codec("none".to_string());

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
                "Video muting is not supported on wasm32".to_string(),
            ))
        }
    }

    /// Mute specific audio tracks by index.
    ///
    /// Returns [`ConversionError::UnsupportedFormat`] until track-selective
    /// muxing is integrated.
    pub async fn mute_tracks<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input: P,
        output: Q,
        track_indices: &[usize],
    ) -> Result<()> {
        let input = input.as_ref();
        let output = output.as_ref();

        if !input.exists() {
            return Err(ConversionError::InvalidInput(format!(
                "Input file not found: {}",
                input.display()
            )));
        }

        if track_indices.is_empty() {
            return Err(ConversionError::InvalidInput(
                "At least one track index must be specified".to_string(),
            ));
        }

        let _ = output;

        Err(ConversionError::UnsupportedFormat(
            "Track-selective muting requires the container demux/mux API, which is not yet \
             integrated."
                .to_string(),
        ))
    }

    /// Replace audio with synthesized silence.
    ///
    /// The output file retains the original audio track structure (sample rate,
    /// channel count, codec) but all audio frames contain zero-amplitude PCM
    /// samples.
    ///
    /// Returns [`ConversionError::UnsupportedFormat`] until packet-level muxing
    /// is integrated.
    pub async fn replace_with_silence<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input: P,
        output: Q,
    ) -> Result<()> {
        let input = input.as_ref();
        let output = output.as_ref();

        if !input.exists() {
            return Err(ConversionError::InvalidInput(format!(
                "Input file not found: {}",
                input.display()
            )));
        }

        let _ = output;

        Err(ConversionError::UnsupportedFormat(
            "Silence-replacement muting requires packet-level container demux/mux integration, \
             which is not yet available."
                .to_string(),
        ))
    }

    /// Re-encode video while muting.
    #[must_use]
    pub fn reencode(self) -> Self {
        self.with_copy_video(false)
    }
}

impl Default for VideoMuter {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility: generate a block of zero-valued PCM samples for silence.
///
/// `channels` × `samples_per_channel` samples of 16-bit PCM (little-endian)
/// at the given sample rate. Used internally for silence replacement; exposed
/// for testing.
#[must_use]
pub fn generate_silence_pcm(channels: u8, samples_per_channel: usize) -> Vec<u8> {
    // 2 bytes per sample (i16 LE), channels interleaved.
    vec![0u8; channels as usize * samples_per_channel * 2]
}

/// Compute the number of PCM samples required to represent `duration_secs`
/// at the given sample rate.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn silence_sample_count(sample_rate: u32, duration_secs: f64) -> usize {
    (sample_rate as f64 * duration_secs).ceil() as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_muter_creation() {
        let muter = VideoMuter::new();
        assert!(muter.copy_video);
        assert!(!muter.silence_mode);
    }

    #[test]
    fn test_muter_settings() {
        let muter = VideoMuter::new()
            .with_copy_video(false)
            .with_silence_mode(true);
        assert!(!muter.copy_video);
        assert!(muter.silence_mode);
    }

    #[test]
    fn test_reencode() {
        let muter = VideoMuter::new().reencode();
        assert!(!muter.copy_video);
    }

    #[test]
    fn test_generate_silence_pcm_size() {
        let silence = generate_silence_pcm(2, 1024);
        // 2 channels × 1024 samples × 2 bytes = 4096 bytes.
        assert_eq!(silence.len(), 4096);
        // All bytes must be zero.
        assert!(silence.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_generate_silence_pcm_mono() {
        let silence = generate_silence_pcm(1, 48000);
        assert_eq!(silence.len(), 48000 * 2);
    }

    #[test]
    fn test_silence_sample_count() {
        // 44.1 kHz for 1 second = 44100 samples.
        assert_eq!(silence_sample_count(44100, 1.0), 44100);
        // 48 kHz for 0.5 seconds = 24000 samples.
        assert_eq!(silence_sample_count(48000, 0.5), 24000);
        // 48 kHz for 0.001 seconds = ceil(48.0) = 48 samples.
        assert_eq!(silence_sample_count(48000, 0.001), 48);
    }

    #[test]
    fn test_silence_energy_is_zero() {
        // All samples are zero → RMS energy is 0.
        let pcm = generate_silence_pcm(2, 1000);
        let sum_sq: u64 = pcm
            .chunks(2)
            .map(|s| {
                let sample = i16::from_le_bytes([s[0], s[1]]) as i64;
                (sample * sample) as u64
            })
            .sum();
        assert_eq!(sum_sq, 0, "silence must have zero energy");
    }

    #[tokio::test]
    async fn test_mute_missing_file_errors() {
        let muter = VideoMuter::new();
        let input = std::env::temp_dir().join("__oximedia_nonexistent_mute__.mkv");
        let output = std::env::temp_dir().join("__oximedia_nonexistent_mute_out__.mkv");
        let result = muter.mute(&input, &output).await;
        assert!(
            matches!(result, Err(ConversionError::InvalidInput(_))),
            "expected InvalidInput, got {result:?}"
        );
    }

    #[tokio::test]
    async fn test_mute_tracks_empty_indices_errors() {
        let tmp = std::env::temp_dir().join("oximedia_convert_mute_tracks.mkv");
        std::fs::write(&tmp, b"dummy").expect("write dummy");
        let muter = VideoMuter::new();
        let result = muter
            .mute_tracks(
                &tmp,
                std::env::temp_dir().join("oximedia_convert_mute_tracks_out.mkv"),
                &[],
            )
            .await;
        assert!(
            matches!(result, Err(ConversionError::InvalidInput(_))),
            "expected InvalidInput for empty track list, got {result:?}"
        );
        let _ = std::fs::remove_file(&tmp);
    }

    #[tokio::test]
    async fn test_replace_with_silence_missing_file_errors() {
        let muter = VideoMuter::new();
        let input = std::env::temp_dir().join("__oximedia_nonexistent_silence__.mkv");
        let output = std::env::temp_dir().join("__oximedia_nonexistent_silence_out__.mkv");
        let result = muter.replace_with_silence(&input, &output).await;
        assert!(
            matches!(result, Err(ConversionError::InvalidInput(_))),
            "expected InvalidInput, got {result:?}"
        );
    }
}
