//! Channel extraction for selecting specific channels from multi-channel streams.

use serde::{Deserialize, Serialize};

/// Channel extraction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelExtractor {
    /// Total number of input channels
    pub input_channels: u8,
    /// Channels to extract
    pub extract_channels: Vec<u8>,
    /// Output channel ordering (maps extract index to output index)
    pub output_mapping: Vec<u8>,
}

impl ChannelExtractor {
    /// Create a new channel extractor
    #[must_use]
    pub fn new(input_channels: u8) -> Self {
        Self {
            input_channels,
            extract_channels: Vec::new(),
            output_mapping: Vec::new(),
        }
    }

    /// Add a channel to extract
    pub fn add_channel(&mut self, channel: u8) -> Result<(), ExtractError> {
        if channel >= self.input_channels {
            return Err(ExtractError::InvalidChannel(channel));
        }
        if self.extract_channels.contains(&channel) {
            return Err(ExtractError::DuplicateChannel(channel));
        }

        self.extract_channels.push(channel);
        self.output_mapping
            .push(self.extract_channels.len() as u8 - 1);
        Ok(())
    }

    /// Add multiple channels to extract
    pub fn add_channels(&mut self, channels: &[u8]) -> Result<(), ExtractError> {
        for &channel in channels {
            self.add_channel(channel)?;
        }
        Ok(())
    }

    /// Get the number of output channels
    #[must_use]
    pub fn output_count(&self) -> usize {
        self.extract_channels.len()
    }

    /// Check if a channel is being extracted
    #[must_use]
    pub fn is_extracting(&self, channel: u8) -> bool {
        self.extract_channels.contains(&channel)
    }

    /// Get the output index for an input channel
    #[must_use]
    pub fn get_output_index(&self, input_channel: u8) -> Option<usize> {
        self.extract_channels
            .iter()
            .position(|&ch| ch == input_channel)
    }

    /// Create an extractor for left channel only
    #[must_use]
    pub fn left_only(input_channels: u8) -> Self {
        let mut extractor = Self::new(input_channels);
        let _ = extractor.add_channel(0);
        extractor
    }

    /// Create an extractor for right channel only
    #[must_use]
    pub fn right_only(input_channels: u8) -> Self {
        let mut extractor = Self::new(input_channels);
        if input_channels >= 2 {
            let _ = extractor.add_channel(1);
        }
        extractor
    }

    /// Create an extractor for stereo pair
    #[must_use]
    pub fn stereo_pair(input_channels: u8) -> Self {
        let mut extractor = Self::new(input_channels);
        let _ = extractor.add_channels(&[0, 1]);
        extractor
    }

    /// Create an extractor for 5.1 surround
    #[must_use]
    pub fn surround_51(input_channels: u8) -> Self {
        let mut extractor = Self::new(input_channels);
        if input_channels >= 6 {
            let _ = extractor.add_channels(&[0, 1, 2, 3, 4, 5]);
        }
        extractor
    }
}

/// Channel selector for dynamic channel selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelSelector {
    /// Available input channels
    pub available_channels: Vec<ChannelInfo>,
    /// Currently selected channels
    pub selected_channels: Vec<u8>,
}

/// Information about a channel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelInfo {
    /// Channel index
    pub index: u8,
    /// Channel label
    pub label: String,
    /// Whether this channel is active
    pub active: bool,
}

impl ChannelInfo {
    /// Create new channel info
    #[must_use]
    pub fn new(index: u8, label: String) -> Self {
        Self {
            index,
            label,
            active: true,
        }
    }
}

impl ChannelSelector {
    /// Create a new channel selector
    #[must_use]
    pub fn new(channel_count: u8) -> Self {
        let available_channels = (0..channel_count)
            .map(|i| ChannelInfo::new(i, format!("Channel {}", i + 1)))
            .collect();

        Self {
            available_channels,
            selected_channels: Vec::new(),
        }
    }

    /// Select a channel
    pub fn select_channel(&mut self, channel: u8) -> Result<(), ExtractError> {
        if (channel as usize) >= self.available_channels.len() {
            return Err(ExtractError::InvalidChannel(channel));
        }
        if self.selected_channels.contains(&channel) {
            return Err(ExtractError::DuplicateChannel(channel));
        }

        self.selected_channels.push(channel);
        Ok(())
    }

    /// Deselect a channel
    pub fn deselect_channel(&mut self, channel: u8) {
        self.selected_channels.retain(|&ch| ch != channel);
    }

    /// Select all channels
    pub fn select_all(&mut self) {
        self.selected_channels = (0..self.available_channels.len() as u8).collect();
    }

    /// Deselect all channels
    pub fn deselect_all(&mut self) {
        self.selected_channels.clear();
    }

    /// Get selected channel count
    #[must_use]
    pub fn selected_count(&self) -> usize {
        self.selected_channels.len()
    }

    /// Check if a channel is selected
    #[must_use]
    pub fn is_selected(&self, channel: u8) -> bool {
        self.selected_channels.contains(&channel)
    }

    /// Set label for a channel
    pub fn set_channel_label(&mut self, channel: u8, label: String) -> Result<(), ExtractError> {
        if let Some(info) = self.available_channels.get_mut(channel as usize) {
            info.label = label;
            Ok(())
        } else {
            Err(ExtractError::InvalidChannel(channel))
        }
    }

    /// Get channel info
    #[must_use]
    pub fn get_channel_info(&self, channel: u8) -> Option<&ChannelInfo> {
        self.available_channels.get(channel as usize)
    }
}

/// Errors that can occur in channel extraction
#[derive(Debug, Clone, thiserror::Error)]
pub enum ExtractError {
    /// Invalid channel index
    #[error("Invalid channel: {0}")]
    InvalidChannel(u8),
    /// Duplicate channel
    #[error("Duplicate channel: {0}")]
    DuplicateChannel(u8),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extractor_creation() {
        let extractor = ChannelExtractor::new(8);
        assert_eq!(extractor.input_channels, 8);
        assert_eq!(extractor.output_count(), 0);
    }

    #[test]
    fn test_add_channel() {
        let mut extractor = ChannelExtractor::new(8);
        extractor.add_channel(0).expect("should succeed in test");
        extractor.add_channel(2).expect("should succeed in test");
        extractor.add_channel(5).expect("should succeed in test");

        assert_eq!(extractor.output_count(), 3);
        assert!(extractor.is_extracting(0));
        assert!(extractor.is_extracting(2));
        assert!(extractor.is_extracting(5));
        assert!(!extractor.is_extracting(1));
    }

    #[test]
    fn test_invalid_channel() {
        let mut extractor = ChannelExtractor::new(4);
        assert!(matches!(
            extractor.add_channel(10),
            Err(ExtractError::InvalidChannel(10))
        ));
    }

    #[test]
    fn test_duplicate_channel() {
        let mut extractor = ChannelExtractor::new(4);
        extractor.add_channel(0).expect("should succeed in test");
        assert!(matches!(
            extractor.add_channel(0),
            Err(ExtractError::DuplicateChannel(0))
        ));
    }

    #[test]
    fn test_output_index() {
        let mut extractor = ChannelExtractor::new(8);
        extractor.add_channel(2).expect("should succeed in test");
        extractor.add_channel(5).expect("should succeed in test");
        extractor.add_channel(7).expect("should succeed in test");

        assert_eq!(extractor.get_output_index(2), Some(0));
        assert_eq!(extractor.get_output_index(5), Some(1));
        assert_eq!(extractor.get_output_index(7), Some(2));
        assert_eq!(extractor.get_output_index(0), None);
    }

    #[test]
    fn test_left_only() {
        let extractor = ChannelExtractor::left_only(8);
        assert_eq!(extractor.output_count(), 1);
        assert!(extractor.is_extracting(0));
    }

    #[test]
    fn test_stereo_pair() {
        let extractor = ChannelExtractor::stereo_pair(8);
        assert_eq!(extractor.output_count(), 2);
        assert!(extractor.is_extracting(0));
        assert!(extractor.is_extracting(1));
    }

    #[test]
    fn test_surround_51() {
        let extractor = ChannelExtractor::surround_51(8);
        assert_eq!(extractor.output_count(), 6);
        for i in 0..6 {
            assert!(extractor.is_extracting(i));
        }
    }

    #[test]
    fn test_channel_selector() {
        let mut selector = ChannelSelector::new(4);
        assert_eq!(selector.selected_count(), 0);

        selector.select_channel(0).expect("should succeed in test");
        selector.select_channel(2).expect("should succeed in test");

        assert_eq!(selector.selected_count(), 2);
        assert!(selector.is_selected(0));
        assert!(selector.is_selected(2));
        assert!(!selector.is_selected(1));
    }

    #[test]
    fn test_selector_deselect() {
        let mut selector = ChannelSelector::new(4);
        selector.select_channel(0).expect("should succeed in test");
        selector.select_channel(1).expect("should succeed in test");

        selector.deselect_channel(0);
        assert!(!selector.is_selected(0));
        assert!(selector.is_selected(1));
    }

    #[test]
    fn test_select_deselect_all() {
        let mut selector = ChannelSelector::new(4);

        selector.select_all();
        assert_eq!(selector.selected_count(), 4);

        selector.deselect_all();
        assert_eq!(selector.selected_count(), 0);
    }

    #[test]
    fn test_channel_labels() {
        let mut selector = ChannelSelector::new(2);
        selector
            .set_channel_label(0, "Left".to_string())
            .expect("should succeed in test");
        selector
            .set_channel_label(1, "Right".to_string())
            .expect("should succeed in test");

        assert_eq!(
            selector
                .get_channel_info(0)
                .expect("should succeed in test")
                .label,
            "Left"
        );
        assert_eq!(
            selector
                .get_channel_info(1)
                .expect("should succeed in test")
                .label,
            "Right"
        );
    }
}
