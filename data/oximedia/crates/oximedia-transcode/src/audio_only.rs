//! Audio-only transcoding mode.
//!
//! Provides configuration and transcoding logic for audio-only pipelines,
//! including codec selection, sample-rate conversion stub, and channel mapping.

use crate::{Result, TranscodeError};

/// Identifies a specific audio codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioCodecId {
    /// Opus – modern, low-latency, patent-free.
    Opus,
    /// FLAC – Free Lossless Audio Codec.
    Flac,
    /// Vorbis – open source lossy codec.
    Vorbis,
    /// MP3 – MPEG Layer III (patents expired 2017).
    Mp3,
    /// AAC – Advanced Audio Coding.
    Aac,
    /// PCM – uncompressed linear PCM.
    Pcm,
}

impl AudioCodecId {
    /// Returns the common name string for this codec.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Opus => "opus",
            Self::Flac => "flac",
            Self::Vorbis => "vorbis",
            Self::Mp3 => "mp3",
            Self::Aac => "aac",
            Self::Pcm => "pcm",
        }
    }

    /// Returns `true` if this codec is lossless.
    #[must_use]
    pub fn is_lossless(self) -> bool {
        matches!(self, Self::Flac | Self::Pcm)
    }

    /// Returns the typical default bitrate (bits per second) for the codec at stereo 48 kHz.
    #[must_use]
    pub fn default_bitrate(self) -> u32 {
        match self {
            Self::Opus => 128_000,
            Self::Flac => 0, // lossless – no fixed bitrate
            Self::Vorbis => 128_000,
            Self::Mp3 => 192_000,
            Self::Aac => 192_000,
            Self::Pcm => 0, // uncompressed – no fixed bitrate
        }
    }
}

/// Configuration for an audio-only transcode operation.
#[derive(Debug, Clone)]
pub struct AudioOnlyConfig {
    /// Codec of the input audio stream.
    pub input_codec: AudioCodecId,
    /// Codec to encode the output audio to.
    pub output_codec: AudioCodecId,
    /// Target sample rate in Hz (e.g. 48000).
    pub sample_rate: u32,
    /// Number of output channels (1 = mono, 2 = stereo, etc.).
    pub channels: u8,
    /// Target bitrate in bits per second.
    /// Ignored for lossless codecs; use 0 to select the codec default.
    pub bitrate: u32,
}

impl AudioOnlyConfig {
    /// Creates a new `AudioOnlyConfig` and validates the parameters.
    ///
    /// # Errors
    ///
    /// Returns [`TranscodeError::InvalidInput`] when:
    /// - `channels` is 0 or greater than 8
    /// - `sample_rate` is below 8 000 Hz or above 192 000 Hz
    pub fn new(
        input_codec: AudioCodecId,
        output_codec: AudioCodecId,
        sample_rate: u32,
        channels: u8,
        bitrate: u32,
    ) -> Result<Self> {
        if channels == 0 || channels > 8 {
            return Err(TranscodeError::InvalidInput(format!(
                "channels must be 1–8, got {channels}"
            )));
        }
        if sample_rate < 8_000 || sample_rate > 192_000 {
            return Err(TranscodeError::InvalidInput(format!(
                "sample_rate must be 8000–192000 Hz, got {sample_rate}"
            )));
        }
        Ok(Self {
            input_codec,
            output_codec,
            sample_rate,
            channels,
            bitrate,
        })
    }

    /// Shortcut: stereo Opus at 128 kbps / 48 kHz.
    ///
    /// # Panics
    ///
    /// Never panics – the hard-coded values always pass validation.
    #[must_use]
    pub fn opus_stereo() -> Self {
        Self::new(AudioCodecId::Pcm, AudioCodecId::Opus, 48_000, 2, 128_000)
            .expect("hard-coded opus_stereo config is always valid")
    }

    /// Shortcut: stereo FLAC lossless at 48 kHz.
    ///
    /// # Panics
    ///
    /// Never panics – the hard-coded values always pass validation.
    #[must_use]
    pub fn flac_stereo() -> Self {
        Self::new(AudioCodecId::Pcm, AudioCodecId::Flac, 48_000, 2, 0)
            .expect("hard-coded flac_stereo config is always valid")
    }
}

/// Audio-only transcoder.
///
/// Accepts raw PCM samples (f32, -1.0..=1.0), applies channel mapping and
/// level adjustments, and returns the processed output samples.
///
/// In a full pipeline this struct would wrap actual codec encode/decode calls;
/// here it provides the configuration layer and frame-level processing hooks
/// that the codec pipeline can call.
pub struct AudioOnlyTranscoder {
    config: AudioOnlyConfig,
    /// Frame count processed so far (used for metrics).
    frames_processed: u64,
}

impl AudioOnlyTranscoder {
    /// Creates a new `AudioOnlyTranscoder` from a validated config.
    #[must_use]
    pub fn new(config: AudioOnlyConfig) -> Self {
        Self {
            config,
            frames_processed: 0,
        }
    }

    /// Returns the active configuration.
    #[must_use]
    pub fn config(&self) -> &AudioOnlyConfig {
        &self.config
    }

    /// Returns the output codec name string.
    #[must_use]
    pub fn codec_name(&self) -> &str {
        self.config.output_codec.name()
    }

    /// Returns the effective output bitrate in bits per second.
    ///
    /// If `config.bitrate` is 0 the codec default is returned.
    #[must_use]
    pub fn estimated_bitrate(&self) -> u32 {
        if self.config.bitrate == 0 {
            self.config.output_codec.default_bitrate()
        } else {
            self.config.bitrate
        }
    }

    /// Number of audio frames processed since creation (or last reset).
    #[must_use]
    pub fn frames_processed(&self) -> u64 {
        self.frames_processed
    }

    /// Process a block of interleaved PCM samples.
    ///
    /// Input samples must be interleaved with the number of channels declared in
    /// the config, in f32 format.  Output length equals input length after
    /// channel mapping (no sample-rate conversion is performed in this stub –
    /// that would be delegated to `oximedia-audio::resample`).
    ///
    /// # Errors
    ///
    /// Returns [`TranscodeError::InvalidInput`] when:
    /// - `input` length is not a multiple of `channels`
    /// - any individual sample is NaN or infinite
    pub fn transcode_samples(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        let ch = self.config.channels as usize;
        if ch == 0 {
            return Err(TranscodeError::InvalidInput(
                "channel count must not be zero".to_string(),
            ));
        }
        if input.len() % ch != 0 {
            return Err(TranscodeError::InvalidInput(format!(
                "input length {} is not a multiple of channel count {}",
                input.len(),
                ch
            )));
        }
        // Validate samples
        for (idx, &s) in input.iter().enumerate() {
            if s.is_nan() || s.is_infinite() {
                return Err(TranscodeError::InvalidInput(format!(
                    "sample at index {idx} is non-finite: {s}"
                )));
            }
        }

        let num_frames = input.len() / ch;
        self.frames_processed += num_frames as u64;

        // Apply a simple gain stage to simulate codec processing.
        // Real implementation would encode → decode via the relevant codec.
        let gain = self.codec_gain_factor();
        let output: Vec<f32> = input.iter().map(|&s| s * gain).collect();

        Ok(output)
    }

    /// Reset internal counters (e.g. between programs).
    pub fn reset(&mut self) {
        self.frames_processed = 0;
    }

    /// Update the config on the fly (e.g. for adaptive bitrate pipelines).
    ///
    /// # Errors
    ///
    /// Returns an error if the new config fails validation.
    pub fn update_config(
        &mut self,
        input_codec: AudioCodecId,
        output_codec: AudioCodecId,
        sample_rate: u32,
        channels: u8,
        bitrate: u32,
    ) -> Result<()> {
        let new_cfg =
            AudioOnlyConfig::new(input_codec, output_codec, sample_rate, channels, bitrate)?;
        self.config = new_cfg;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Returns a synthetic gain factor used by `transcode_samples` to simulate
    /// the slight level differences introduced by lossy codecs.
    fn codec_gain_factor(&self) -> f32 {
        match self.config.output_codec {
            AudioCodecId::Flac | AudioCodecId::Pcm => 1.0, // lossless
            AudioCodecId::Opus => 0.9999,
            AudioCodecId::Vorbis => 0.9998,
            AudioCodecId::Mp3 => 0.9997,
            AudioCodecId::Aac => 0.9996,
        }
    }
}

// ============================================================
// Unit tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ----------------------------------------------------------
    // AudioCodecId tests
    // ----------------------------------------------------------

    #[test]
    fn test_codec_id_names() {
        assert_eq!(AudioCodecId::Opus.name(), "opus");
        assert_eq!(AudioCodecId::Flac.name(), "flac");
        assert_eq!(AudioCodecId::Vorbis.name(), "vorbis");
        assert_eq!(AudioCodecId::Mp3.name(), "mp3");
        assert_eq!(AudioCodecId::Aac.name(), "aac");
        assert_eq!(AudioCodecId::Pcm.name(), "pcm");
    }

    #[test]
    fn test_codec_lossless_flag() {
        assert!(AudioCodecId::Flac.is_lossless());
        assert!(AudioCodecId::Pcm.is_lossless());
        assert!(!AudioCodecId::Opus.is_lossless());
        assert!(!AudioCodecId::Vorbis.is_lossless());
        assert!(!AudioCodecId::Mp3.is_lossless());
        assert!(!AudioCodecId::Aac.is_lossless());
    }

    #[test]
    fn test_codec_default_bitrate() {
        assert_eq!(AudioCodecId::Flac.default_bitrate(), 0);
        assert_eq!(AudioCodecId::Pcm.default_bitrate(), 0);
        assert!(AudioCodecId::Opus.default_bitrate() > 0);
        assert!(AudioCodecId::Mp3.default_bitrate() > 0);
    }

    // ----------------------------------------------------------
    // AudioOnlyConfig creation
    // ----------------------------------------------------------

    #[test]
    fn test_config_valid_creation() {
        let cfg = AudioOnlyConfig::new(AudioCodecId::Pcm, AudioCodecId::Opus, 48_000, 2, 128_000);
        assert!(cfg.is_ok(), "Valid config should succeed");
        let cfg = cfg.expect("already checked");
        assert_eq!(cfg.sample_rate, 48_000);
        assert_eq!(cfg.channels, 2);
        assert_eq!(cfg.bitrate, 128_000);
    }

    #[test]
    fn test_config_invalid_channels_zero() {
        let result =
            AudioOnlyConfig::new(AudioCodecId::Pcm, AudioCodecId::Opus, 48_000, 0, 128_000);
        assert!(result.is_err(), "channels=0 must fail");
        let msg = result.expect_err("expected error").to_string();
        assert!(
            msg.contains("channels"),
            "Error should mention 'channels': {msg}"
        );
    }

    #[test]
    fn test_config_invalid_channels_too_many() {
        let result =
            AudioOnlyConfig::new(AudioCodecId::Pcm, AudioCodecId::Opus, 48_000, 9, 128_000);
        assert!(result.is_err(), "channels=9 must fail");
    }

    #[test]
    fn test_config_invalid_sample_rate_too_low() {
        let result = AudioOnlyConfig::new(AudioCodecId::Pcm, AudioCodecId::Opus, 7_999, 2, 128_000);
        assert!(result.is_err(), "sample_rate=7999 must fail");
        let msg = result.expect_err("expected error").to_string();
        assert!(
            msg.contains("sample_rate"),
            "Error should mention 'sample_rate': {msg}"
        );
    }

    #[test]
    fn test_config_invalid_sample_rate_too_high() {
        let result =
            AudioOnlyConfig::new(AudioCodecId::Pcm, AudioCodecId::Opus, 192_001, 2, 128_000);
        assert!(result.is_err(), "sample_rate=192001 must fail");
    }

    #[test]
    fn test_config_boundary_sample_rates() {
        // Minimum valid sample rate
        let low = AudioOnlyConfig::new(AudioCodecId::Pcm, AudioCodecId::Pcm, 8_000, 1, 0);
        assert!(low.is_ok(), "sample_rate=8000 should be valid");

        // Maximum valid sample rate
        let high = AudioOnlyConfig::new(AudioCodecId::Pcm, AudioCodecId::Pcm, 192_000, 1, 0);
        assert!(high.is_ok(), "sample_rate=192000 should be valid");
    }

    #[test]
    fn test_config_shortcuts() {
        let opus = AudioOnlyConfig::opus_stereo();
        assert_eq!(opus.output_codec, AudioCodecId::Opus);
        assert_eq!(opus.channels, 2);
        assert_eq!(opus.sample_rate, 48_000);

        let flac = AudioOnlyConfig::flac_stereo();
        assert_eq!(flac.output_codec, AudioCodecId::Flac);
        assert_eq!(flac.channels, 2);
        assert!(flac.output_codec.is_lossless());
    }

    // ----------------------------------------------------------
    // AudioOnlyTranscoder tests
    // ----------------------------------------------------------

    #[test]
    fn test_transcoder_creation() {
        let cfg = AudioOnlyConfig::opus_stereo();
        let t = AudioOnlyTranscoder::new(cfg);
        assert_eq!(t.codec_name(), "opus");
        assert_eq!(t.frames_processed(), 0);
    }

    #[test]
    fn test_transcoder_estimated_bitrate_from_config() {
        let cfg = AudioOnlyConfig::new(AudioCodecId::Pcm, AudioCodecId::Opus, 48_000, 2, 256_000)
            .expect("valid");
        let t = AudioOnlyTranscoder::new(cfg);
        assert_eq!(t.estimated_bitrate(), 256_000);
    }

    #[test]
    fn test_transcoder_estimated_bitrate_uses_default_when_zero() {
        let cfg = AudioOnlyConfig::new(AudioCodecId::Pcm, AudioCodecId::Opus, 48_000, 2, 0)
            .expect("valid");
        let t = AudioOnlyTranscoder::new(cfg);
        assert_eq!(t.estimated_bitrate(), AudioCodecId::Opus.default_bitrate());
    }

    #[test]
    fn test_transcode_samples_sine_wave() {
        let cfg = AudioOnlyConfig::opus_stereo();
        let mut t = AudioOnlyTranscoder::new(cfg);

        // Generate a 1 kHz stereo sine wave at 100 samples
        let sample_rate = 48_000.0f32;
        let freq = 1_000.0f32;
        let num_frames = 100_usize;
        let mut input = Vec::with_capacity(num_frames * 2);
        for i in 0..num_frames {
            let s = 0.5 * (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate).sin();
            input.push(s); // L
            input.push(s); // R
        }

        let result = t.transcode_samples(&input);
        assert!(
            result.is_ok(),
            "transcode_samples must succeed: {:?}",
            result.err()
        );
        let output = result.expect("already checked");
        assert_eq!(output.len(), input.len(), "Output length must match input");
        assert_eq!(t.frames_processed(), num_frames as u64);
    }

    #[test]
    fn test_transcode_samples_rejects_nan() {
        let cfg = AudioOnlyConfig::opus_stereo();
        let mut t = AudioOnlyTranscoder::new(cfg);
        let input = vec![0.1f32, f32::NAN, 0.3, 0.4];
        let result = t.transcode_samples(&input);
        assert!(result.is_err(), "NaN sample must be rejected");
    }

    #[test]
    fn test_transcode_samples_rejects_infinite() {
        let cfg = AudioOnlyConfig::opus_stereo();
        let mut t = AudioOnlyTranscoder::new(cfg);
        let input = vec![0.1f32, f32::INFINITY, 0.3, 0.4];
        let result = t.transcode_samples(&input);
        assert!(result.is_err(), "Infinite sample must be rejected");
    }

    #[test]
    fn test_transcode_samples_rejects_misaligned_input() {
        let cfg = AudioOnlyConfig::opus_stereo(); // 2 channels
        let mut t = AudioOnlyTranscoder::new(cfg);
        // 3 samples is not a multiple of 2
        let input = vec![0.1f32, 0.2, 0.3];
        let result = t.transcode_samples(&input);
        assert!(result.is_err(), "Misaligned input must be rejected");
    }

    #[test]
    fn test_transcode_samples_empty_input_succeeds() {
        let cfg = AudioOnlyConfig::opus_stereo();
        let mut t = AudioOnlyTranscoder::new(cfg);
        let result = t.transcode_samples(&[]);
        assert!(result.is_ok());
        assert_eq!(result.expect("ok").len(), 0);
    }

    #[test]
    fn test_reset_clears_frame_count() {
        let cfg = AudioOnlyConfig::opus_stereo();
        let mut t = AudioOnlyTranscoder::new(cfg);
        let input = vec![0.1f32, 0.2]; // 1 stereo frame
        t.transcode_samples(&input).expect("ok");
        assert_eq!(t.frames_processed(), 1);
        t.reset();
        assert_eq!(t.frames_processed(), 0);
    }

    #[test]
    fn test_update_config_valid() {
        let cfg = AudioOnlyConfig::opus_stereo();
        let mut t = AudioOnlyTranscoder::new(cfg);
        let result = t.update_config(AudioCodecId::Pcm, AudioCodecId::Flac, 44_100, 2, 0);
        assert!(
            result.is_ok(),
            "update_config should succeed: {:?}",
            result.err()
        );
        assert_eq!(t.codec_name(), "flac");
    }

    #[test]
    fn test_update_config_invalid_channels() {
        let cfg = AudioOnlyConfig::opus_stereo();
        let mut t = AudioOnlyTranscoder::new(cfg);
        let result = t.update_config(AudioCodecId::Pcm, AudioCodecId::Opus, 48_000, 0, 128_000);
        assert!(result.is_err(), "Invalid channels in update should fail");
    }

    #[test]
    fn test_lossless_output_gain_is_unity() {
        let cfg = AudioOnlyConfig::new(AudioCodecId::Pcm, AudioCodecId::Flac, 48_000, 2, 0)
            .expect("valid");
        let mut t = AudioOnlyTranscoder::new(cfg);
        let input = vec![0.5f32, 0.5f32]; // one stereo frame
        let output = t.transcode_samples(&input).expect("ok");
        // FLAC gain factor is 1.0 → output must equal input exactly
        assert!(
            (output[0] - input[0]).abs() < f32::EPSILON,
            "FLAC should be lossless (gain=1.0)"
        );
    }
}
