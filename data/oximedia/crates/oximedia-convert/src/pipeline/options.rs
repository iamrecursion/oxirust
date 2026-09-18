// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Conversion options for video and audio.

use crate::formats::{AudioCodec, ChannelLayout, VideoCodec};
use serde::{Deserialize, Serialize};

/// Bitrate mode for encoding.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum BitrateMode {
    /// Constant bitrate (CBR)
    Cbr(u64),
    /// Variable bitrate (VBR) with target
    Vbr(u64),
    /// Constant quality (CRF)
    Crf(u32),
}

impl BitrateMode {
    /// Get target bitrate if applicable.
    #[must_use]
    pub const fn target_bitrate(&self) -> Option<u64> {
        match self {
            Self::Cbr(bitrate) | Self::Vbr(bitrate) => Some(*bitrate),
            Self::Crf(_) => None,
        }
    }

    /// Get quality value if applicable.
    #[must_use]
    pub const fn quality(&self) -> Option<u32> {
        match self {
            Self::Crf(quality) => Some(*quality),
            Self::Cbr(_) | Self::Vbr(_) => None,
        }
    }
}

/// Video conversion options.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoOptions {
    /// Video codec
    pub codec: VideoCodec,
    /// Target width (None to preserve)
    pub width: Option<u32>,
    /// Target height (None to preserve)
    pub height: Option<u32>,
    /// Target frame rate (None to preserve)
    pub frame_rate: Option<f64>,
    /// Bitrate mode
    pub bitrate: BitrateMode,
    /// Two-pass encoding
    pub two_pass: bool,
    /// Encoding speed (0=fastest, 4=slowest)
    pub speed: u8,
    /// Enable HDR to SDR tone mapping
    pub tone_map_hdr: bool,
    /// Preserve aspect ratio when scaling
    pub preserve_aspect_ratio: bool,
    /// Key frame interval (in frames)
    pub keyint: Option<u32>,
}

impl VideoOptions {
    /// Create default options for a codec.
    #[must_use]
    pub fn default_for_codec(codec: VideoCodec) -> Self {
        Self {
            codec,
            width: None,
            height: None,
            frame_rate: None,
            bitrate: BitrateMode::Crf(codec.default_quality()),
            two_pass: false,
            speed: 2, // Medium
            tone_map_hdr: false,
            preserve_aspect_ratio: true,
            keyint: None,
        }
    }

    /// Set resolution with aspect ratio preservation.
    #[must_use]
    pub fn with_resolution(mut self, width: u32, height: u32) -> Self {
        self.width = Some(width);
        self.height = Some(height);
        self
    }

    /// Set frame rate.
    #[must_use]
    pub fn with_frame_rate(mut self, fps: f64) -> Self {
        self.frame_rate = Some(fps);
        self
    }

    /// Set bitrate mode.
    #[must_use]
    pub fn with_bitrate(mut self, mode: BitrateMode) -> Self {
        self.bitrate = mode;
        self
    }

    /// Enable two-pass encoding.
    #[must_use]
    pub fn with_two_pass(mut self, enabled: bool) -> Self {
        self.two_pass = enabled;
        self
    }
}

/// Audio conversion options.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioOptions {
    /// Audio codec
    pub codec: AudioCodec,
    /// Sample rate in Hz (None to preserve)
    pub sample_rate: Option<u32>,
    /// Channel layout (None to preserve)
    pub channels: Option<ChannelLayout>,
    /// Bitrate in bits/second (for lossy codecs)
    pub bitrate: Option<u64>,
    /// Enable volume normalization
    pub normalize: bool,
    /// Normalization target in dB or LUFS
    pub normalization_target: f64,
    /// Apply dynamic range compression
    pub compress_dynamic_range: bool,
}

impl AudioOptions {
    /// Create default options for a codec.
    #[must_use]
    pub fn default_for_codec(codec: AudioCodec) -> Self {
        Self {
            codec,
            sample_rate: None,
            channels: None,
            bitrate: codec.default_bitrate().map(|kb| u64::from(kb) * 1000),
            normalize: false,
            normalization_target: -23.0, // EBU R128 target
            compress_dynamic_range: false,
        }
    }

    /// Set sample rate.
    #[must_use]
    pub fn with_sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = Some(rate);
        self
    }

    /// Set channel layout.
    #[must_use]
    pub fn with_channels(mut self, layout: ChannelLayout) -> Self {
        self.channels = Some(layout);
        self
    }

    /// Set bitrate (in kbps).
    #[must_use]
    pub fn with_bitrate_kbps(mut self, kbps: u32) -> Self {
        self.bitrate = Some(u64::from(kbps) * 1000);
        self
    }

    /// Enable normalization.
    #[must_use]
    pub fn with_normalization(mut self, target: f64) -> Self {
        self.normalize = true;
        self.normalization_target = target;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitrate_mode() {
        let cbr = BitrateMode::Cbr(5_000_000);
        assert_eq!(cbr.target_bitrate(), Some(5_000_000));
        assert_eq!(cbr.quality(), None);

        let vbr = BitrateMode::Vbr(5_000_000);
        assert_eq!(vbr.target_bitrate(), Some(5_000_000));
        assert_eq!(vbr.quality(), None);

        let crf = BitrateMode::Crf(30);
        assert_eq!(crf.target_bitrate(), None);
        assert_eq!(crf.quality(), Some(30));
    }

    #[test]
    fn test_video_options_builder() {
        let opts = VideoOptions::default_for_codec(VideoCodec::Vp9)
            .with_resolution(1920, 1080)
            .with_frame_rate(30.0)
            .with_bitrate(BitrateMode::Cbr(5_000_000))
            .with_two_pass(true);

        assert_eq!(opts.codec, VideoCodec::Vp9);
        assert_eq!(opts.width, Some(1920));
        assert_eq!(opts.height, Some(1080));
        assert_eq!(opts.frame_rate, Some(30.0));
        assert_eq!(opts.bitrate, BitrateMode::Cbr(5_000_000));
        assert!(opts.two_pass);
    }

    #[test]
    fn test_audio_options_builder() {
        let opts = AudioOptions::default_for_codec(AudioCodec::Opus)
            .with_sample_rate(48000)
            .with_channels(ChannelLayout::Stereo)
            .with_bitrate_kbps(128)
            .with_normalization(-23.0);

        assert_eq!(opts.codec, AudioCodec::Opus);
        assert_eq!(opts.sample_rate, Some(48000));
        assert_eq!(opts.channels, Some(ChannelLayout::Stereo));
        assert_eq!(opts.bitrate, Some(128_000));
        assert!(opts.normalize);
        assert_eq!(opts.normalization_target, -23.0);
    }
}
