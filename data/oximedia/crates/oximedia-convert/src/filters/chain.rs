// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Filter chain for sequential filter application.

use super::{AudioFilter, Filter, VideoFilter};
use serde::{Deserialize, Serialize};

/// A chain of filters to be applied in sequence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FilterChain {
    filters: Vec<Filter>,
}

impl FilterChain {
    /// Create a new empty filter chain.
    #[must_use]
    pub fn new() -> Self {
        Self {
            filters: Vec::new(),
        }
    }

    /// Add a filter to the chain.
    #[must_use]
    pub fn add(mut self, filter: Filter) -> Self {
        self.filters.push(filter);
        self
    }

    /// Add a video filter.
    #[must_use]
    pub fn add_video(self, filter: VideoFilter) -> Self {
        self.add(Filter::Video(filter))
    }

    /// Add an audio filter.
    #[must_use]
    pub fn add_audio(self, filter: AudioFilter) -> Self {
        self.add(Filter::Audio(filter))
    }

    /// Get all filters in the chain.
    #[must_use]
    pub fn filters(&self) -> &[Filter] {
        &self.filters
    }

    /// Get video filters only.
    #[must_use]
    pub fn video_filters(&self) -> Vec<&VideoFilter> {
        self.filters
            .iter()
            .filter_map(|f| match f {
                Filter::Video(vf) => Some(vf),
                _ => None,
            })
            .collect()
    }

    /// Get audio filters only.
    #[must_use]
    pub fn audio_filters(&self) -> Vec<&AudioFilter> {
        self.filters
            .iter()
            .filter_map(|f| match f {
                Filter::Audio(af) => Some(af),
                _ => None,
            })
            .collect()
    }

    /// Check if chain is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.filters.is_empty()
    }

    /// Get number of filters in chain.
    #[must_use]
    pub fn len(&self) -> usize {
        self.filters.len()
    }
}

impl Default for FilterChain {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filters::video::{DeinterlaceMode, RotateAngle};

    #[test]
    fn test_filter_chain_creation() {
        let chain = FilterChain::new();
        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);
    }

    #[test]
    fn test_filter_chain_add() {
        let chain = FilterChain::new()
            .add_video(VideoFilter::Deinterlace(DeinterlaceMode::Yadif))
            .add_video(VideoFilter::Rotate(RotateAngle::Rotate90))
            .add_audio(AudioFilter::VolumeAdjust(1.5));

        assert_eq!(chain.len(), 3);
        assert!(!chain.is_empty());
    }

    #[test]
    fn test_filter_chain_video_filters() {
        let chain = FilterChain::new()
            .add_video(VideoFilter::Deinterlace(DeinterlaceMode::Yadif))
            .add_audio(AudioFilter::VolumeAdjust(1.5))
            .add_video(VideoFilter::FlipHorizontal);

        let video_filters = chain.video_filters();
        assert_eq!(video_filters.len(), 2);
    }

    #[test]
    fn test_filter_chain_audio_filters() {
        let chain = FilterChain::new()
            .add_video(VideoFilter::Deinterlace(DeinterlaceMode::Yadif))
            .add_audio(AudioFilter::VolumeAdjust(1.5))
            .add_audio(AudioFilter::DcRemove);

        let audio_filters = chain.audio_filters();
        assert_eq!(audio_filters.len(), 2);
    }
}
