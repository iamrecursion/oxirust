//! Channel splitting and combining operations.

use serde::{Deserialize, Serialize};

/// Configuration for splitting channels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelSplitter {
    /// Number of input channels
    pub input_channels: u8,
    /// Split configurations
    pub splits: Vec<Split>,
}

/// Represents a single split operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Split {
    /// Input channel to split from
    pub input_channel: u8,
    /// Output channels this input feeds to
    pub output_channels: Vec<u8>,
    /// Gain for each output (in dB)
    pub gains_db: Vec<f32>,
}

impl Split {
    /// Create a new split
    #[must_use]
    pub fn new(input_channel: u8) -> Self {
        Self {
            input_channel,
            output_channels: Vec::new(),
            gains_db: Vec::new(),
        }
    }

    /// Add an output channel
    pub fn add_output(&mut self, output_channel: u8, gain_db: f32) {
        self.output_channels.push(output_channel);
        self.gains_db.push(gain_db);
    }

    /// Get the number of outputs
    #[must_use]
    pub fn output_count(&self) -> usize {
        self.output_channels.len()
    }
}

impl ChannelSplitter {
    /// Create a new channel splitter
    #[must_use]
    pub fn new(input_channels: u8) -> Self {
        Self {
            input_channels,
            splits: Vec::new(),
        }
    }

    /// Add a split configuration
    pub fn add_split(&mut self, split: Split) -> Result<(), SplitError> {
        if split.input_channel >= self.input_channels {
            return Err(SplitError::InvalidInputChannel(split.input_channel));
        }
        self.splits.push(split);
        Ok(())
    }

    /// Get splits for a specific input channel
    #[must_use]
    pub fn get_splits_for_input(&self, input_channel: u8) -> Vec<&Split> {
        self.splits
            .iter()
            .filter(|s| s.input_channel == input_channel)
            .collect()
    }

    /// Calculate total number of output channels needed
    #[must_use]
    pub fn total_output_channels(&self) -> usize {
        self.splits
            .iter()
            .flat_map(|s| &s.output_channels)
            .max()
            .map_or(0, |&max| (max + 1) as usize)
    }
}

/// Configuration for combining channels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelCombiner {
    /// Number of output channels
    pub output_channels: u8,
    /// Combine configurations
    pub combines: Vec<Combine>,
}

/// Represents a single combine operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Combine {
    /// Output channel to combine into
    pub output_channel: u8,
    /// Input channels to combine
    pub input_channels: Vec<u8>,
    /// Gain for each input (in dB)
    pub gains_db: Vec<f32>,
}

impl Combine {
    /// Create a new combine
    #[must_use]
    pub fn new(output_channel: u8) -> Self {
        Self {
            output_channel,
            input_channels: Vec::new(),
            gains_db: Vec::new(),
        }
    }

    /// Add an input channel
    pub fn add_input(&mut self, input_channel: u8, gain_db: f32) {
        self.input_channels.push(input_channel);
        self.gains_db.push(gain_db);
    }

    /// Get the number of inputs
    #[must_use]
    pub fn input_count(&self) -> usize {
        self.input_channels.len()
    }
}

impl ChannelCombiner {
    /// Create a new channel combiner
    #[must_use]
    pub fn new(output_channels: u8) -> Self {
        Self {
            output_channels,
            combines: Vec::new(),
        }
    }

    /// Add a combine configuration
    pub fn add_combine(&mut self, combine: Combine) -> Result<(), SplitError> {
        if combine.output_channel >= self.output_channels {
            return Err(SplitError::InvalidOutputChannel(combine.output_channel));
        }
        self.combines.push(combine);
        Ok(())
    }

    /// Get combine for a specific output channel
    #[must_use]
    pub fn get_combine_for_output(&self, output_channel: u8) -> Option<&Combine> {
        self.combines
            .iter()
            .find(|c| c.output_channel == output_channel)
    }

    /// Calculate total number of input channels needed
    #[must_use]
    pub fn total_input_channels(&self) -> usize {
        self.combines
            .iter()
            .flat_map(|c| &c.input_channels)
            .max()
            .map_or(0, |&max| (max + 1) as usize)
    }
}

/// Errors that can occur in split/combine operations
#[derive(Debug, Clone, thiserror::Error)]
pub enum SplitError {
    /// Invalid input channel
    #[error("Invalid input channel: {0}")]
    InvalidInputChannel(u8),
    /// Invalid output channel
    #[error("Invalid output channel: {0}")]
    InvalidOutputChannel(u8),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_creation() {
        let mut split = Split::new(0);
        split.add_output(0, 0.0);
        split.add_output(1, -6.0);

        assert_eq!(split.input_channel, 0);
        assert_eq!(split.output_count(), 2);
        assert_eq!(split.output_channels, vec![0, 1]);
    }

    #[test]
    fn test_splitter() {
        let mut splitter = ChannelSplitter::new(2);

        let mut split = Split::new(0);
        split.add_output(0, 0.0);
        split.add_output(1, 0.0);
        split.add_output(2, 0.0);

        splitter.add_split(split).expect("should succeed in test");

        assert_eq!(splitter.total_output_channels(), 3);

        let splits = splitter.get_splits_for_input(0);
        assert_eq!(splits.len(), 1);
    }

    #[test]
    fn test_invalid_split() {
        let mut splitter = ChannelSplitter::new(2);

        let split = Split::new(5); // Invalid channel
        assert!(matches!(
            splitter.add_split(split),
            Err(SplitError::InvalidInputChannel(5))
        ));
    }

    #[test]
    fn test_combine_creation() {
        let mut combine = Combine::new(0);
        combine.add_input(0, 0.0);
        combine.add_input(1, -3.0);

        assert_eq!(combine.output_channel, 0);
        assert_eq!(combine.input_count(), 2);
    }

    #[test]
    fn test_combiner() {
        let mut combiner = ChannelCombiner::new(1);

        let mut combine = Combine::new(0);
        combine.add_input(0, -3.0);
        combine.add_input(1, -3.0);

        combiner
            .add_combine(combine)
            .expect("should succeed in test");

        assert_eq!(combiner.total_input_channels(), 2);

        let comb = combiner.get_combine_for_output(0);
        assert!(comb.is_some());
    }

    #[test]
    fn test_invalid_combine() {
        let mut combiner = ChannelCombiner::new(2);

        let combine = Combine::new(5); // Invalid channel
        assert!(matches!(
            combiner.add_combine(combine),
            Err(SplitError::InvalidOutputChannel(5))
        ));
    }

    #[test]
    fn test_multi_split() {
        let mut splitter = ChannelSplitter::new(1);

        // Split one input to multiple outputs
        let mut split = Split::new(0);
        split.add_output(0, 0.0);
        split.add_output(1, -6.0);
        split.add_output(2, -12.0);

        splitter.add_split(split).expect("should succeed in test");

        let splits = splitter.get_splits_for_input(0);
        assert_eq!(splits.len(), 1);
        assert_eq!(splits[0].output_count(), 3);
    }
}
