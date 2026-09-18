// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Video and audio filters for media conversion.
//!
//! This module provides various filters that can be applied during conversion,
//! including deinterlacing, denoising, sharpening, color correction, and more.

pub mod audio;
pub mod chain;
pub mod video;

use serde::{Deserialize, Serialize};

pub use audio::{AudioFilter, EqualizerBand};
pub use chain::FilterChain;
pub use video::{DeinterlaceMode, RotateAngle, VideoFilter};

/// A filter that can be applied during conversion.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Filter {
    /// Video filter
    Video(VideoFilter),
    /// Audio filter
    Audio(AudioFilter),
}

impl Filter {
    /// Check if filter is a video filter.
    #[must_use]
    pub const fn is_video(&self) -> bool {
        matches!(self, Self::Video(_))
    }

    /// Check if filter is an audio filter.
    #[must_use]
    pub const fn is_audio(&self) -> bool {
        matches!(self, Self::Audio(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_is_video() {
        let video_filter = Filter::Video(VideoFilter::Deinterlace(DeinterlaceMode::Bob));
        assert!(video_filter.is_video());
        assert!(!video_filter.is_audio());
    }

    #[test]
    fn test_filter_is_audio() {
        let audio_filter = Filter::Audio(AudioFilter::VolumeAdjust(0.5));
        assert!(audio_filter.is_audio());
        assert!(!audio_filter.is_video());
    }
}
