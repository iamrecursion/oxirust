//! Channel label and layout types for multi-channel audio routing.
//!
//! Provides speaker label enumeration, canonical channel layouts (stereo,
//! 5.1, 7.1, …), and a converter for remapping channels between layouts.

#![allow(dead_code)]

/// Identifies a single speaker or audio channel by its role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChannelLabel {
    /// Front-left speaker.
    FrontLeft,
    /// Front-right speaker.
    FrontRight,
    /// Front-center speaker.
    Center,
    /// Low-frequency effects channel (subwoofer).
    Lfe,
    /// Rear/back-left speaker (surround left).
    RearLeft,
    /// Rear/back-right speaker (surround right).
    RearRight,
    /// Side-left speaker (used in 7.1 layouts).
    SideLeft,
    /// Side-right speaker (used in 7.1 layouts).
    SideRight,
    /// Top-center overhead speaker.
    TopCenter,
    /// Mono (single-channel) signal.
    Mono,
    /// Unknown or unspecified channel role.
    Unknown,
}

impl ChannelLabel {
    /// Returns `true` if this label is the Low-Frequency Effects (LFE/subwoofer) channel.
    pub fn is_lfe(self) -> bool {
        self == ChannelLabel::Lfe
    }

    /// Returns a human-readable short name.
    pub fn short_name(self) -> &'static str {
        match self {
            ChannelLabel::FrontLeft => "FL",
            ChannelLabel::FrontRight => "FR",
            ChannelLabel::Center => "C",
            ChannelLabel::Lfe => "LFE",
            ChannelLabel::RearLeft => "RL",
            ChannelLabel::RearRight => "RR",
            ChannelLabel::SideLeft => "SL",
            ChannelLabel::SideRight => "SR",
            ChannelLabel::TopCenter => "TC",
            ChannelLabel::Mono => "M",
            ChannelLabel::Unknown => "?",
        }
    }
}

/// A canonical multi-channel layout described as an ordered list of [`ChannelLabel`]s.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioChannelLayout {
    labels: Vec<ChannelLabel>,
}

impl AudioChannelLayout {
    /// Create a layout from an explicit list of channel labels.
    pub fn new(labels: Vec<ChannelLabel>) -> Self {
        Self { labels }
    }

    /// Standard mono layout.
    pub fn mono() -> Self {
        Self::new(vec![ChannelLabel::Mono])
    }

    /// Standard stereo layout (FL, FR).
    pub fn stereo() -> Self {
        Self::new(vec![ChannelLabel::FrontLeft, ChannelLabel::FrontRight])
    }

    /// ITU 5.1 layout (FL, FR, C, LFE, RL, RR).
    pub fn surround_5_1() -> Self {
        Self::new(vec![
            ChannelLabel::FrontLeft,
            ChannelLabel::FrontRight,
            ChannelLabel::Center,
            ChannelLabel::Lfe,
            ChannelLabel::RearLeft,
            ChannelLabel::RearRight,
        ])
    }

    /// 7.1 layout (FL, FR, C, LFE, RL, RR, SL, SR).
    pub fn surround_7_1() -> Self {
        Self::new(vec![
            ChannelLabel::FrontLeft,
            ChannelLabel::FrontRight,
            ChannelLabel::Center,
            ChannelLabel::Lfe,
            ChannelLabel::RearLeft,
            ChannelLabel::RearRight,
            ChannelLabel::SideLeft,
            ChannelLabel::SideRight,
        ])
    }

    /// Returns the total number of channels in this layout.
    pub fn channel_count(&self) -> usize {
        self.labels.len()
    }

    /// Returns `true` if this layout contains an LFE channel.
    pub fn has_lfe(&self) -> bool {
        self.labels.iter().any(|l| l.is_lfe())
    }

    /// Returns the channel labels as a slice.
    pub fn labels(&self) -> &[ChannelLabel] {
        &self.labels
    }

    /// Returns the index of a label within this layout, if present.
    pub fn index_of(&self, label: ChannelLabel) -> Option<usize> {
        self.labels.iter().position(|&l| l == label)
    }
}

/// Remaps channels from one [`AudioChannelLayout`] to another.
///
/// Channels that exist in the output layout but not in the input are silenced.
pub struct ChannelConverter {
    src: AudioChannelLayout,
    dst: AudioChannelLayout,
}

impl ChannelConverter {
    /// Create a new converter from `src` layout to `dst` layout.
    pub fn new(src: AudioChannelLayout, dst: AudioChannelLayout) -> Self {
        Self { src, dst }
    }

    /// Remap a slice of interleaved `f32` samples from the source layout to the
    /// destination layout.
    ///
    /// # Parameters
    /// - `input`: interleaved samples in source layout, length must be a multiple of
    ///   `src.channel_count()`.
    ///
    /// Returns a new `Vec<f32>` in the destination layout.
    pub fn remap(&self, input: &[f32]) -> Vec<f32> {
        let src_ch = self.src.channel_count();
        if src_ch == 0 {
            return Vec::new();
        }
        let frames = input.len() / src_ch;
        let dst_ch = self.dst.channel_count();
        let mut output = vec![0.0_f32; frames * dst_ch];

        for (dst_idx, dst_label) in self.dst.labels().iter().enumerate() {
            if let Some(src_idx) = self.src.index_of(*dst_label) {
                for fr in 0..frames {
                    output[fr * dst_ch + dst_idx] = input[fr * src_ch + src_idx];
                }
            }
            // else: silence (already zero)
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_label_is_lfe_true() {
        assert!(ChannelLabel::Lfe.is_lfe());
    }

    #[test]
    fn test_channel_label_is_lfe_false() {
        assert!(!ChannelLabel::FrontLeft.is_lfe());
        assert!(!ChannelLabel::Center.is_lfe());
    }

    #[test]
    fn test_channel_label_short_name() {
        assert_eq!(ChannelLabel::FrontLeft.short_name(), "FL");
        assert_eq!(ChannelLabel::Lfe.short_name(), "LFE");
        assert_eq!(ChannelLabel::Center.short_name(), "C");
    }

    #[test]
    fn test_mono_channel_count() {
        assert_eq!(AudioChannelLayout::mono().channel_count(), 1);
    }

    #[test]
    fn test_stereo_channel_count() {
        assert_eq!(AudioChannelLayout::stereo().channel_count(), 2);
    }

    #[test]
    fn test_5_1_channel_count() {
        assert_eq!(AudioChannelLayout::surround_5_1().channel_count(), 6);
    }

    #[test]
    fn test_7_1_channel_count() {
        assert_eq!(AudioChannelLayout::surround_7_1().channel_count(), 8);
    }

    #[test]
    fn test_stereo_has_no_lfe() {
        assert!(!AudioChannelLayout::stereo().has_lfe());
    }

    #[test]
    fn test_5_1_has_lfe() {
        assert!(AudioChannelLayout::surround_5_1().has_lfe());
    }

    #[test]
    fn test_index_of_existing_label() {
        let layout = AudioChannelLayout::stereo();
        assert_eq!(layout.index_of(ChannelLabel::FrontLeft), Some(0));
        assert_eq!(layout.index_of(ChannelLabel::FrontRight), Some(1));
    }

    #[test]
    fn test_index_of_missing_label() {
        let layout = AudioChannelLayout::stereo();
        assert_eq!(layout.index_of(ChannelLabel::Lfe), None);
    }

    #[test]
    fn test_converter_stereo_passthrough() {
        let src = AudioChannelLayout::stereo();
        let dst = AudioChannelLayout::stereo();
        let conv = ChannelConverter::new(src, dst);
        let input = vec![1.0_f32, 2.0, 3.0, 4.0]; // 2 frames × 2 ch
        let output = conv.remap(&input);
        assert_eq!(output, input);
    }

    #[test]
    fn test_converter_stereo_to_mono_silences_center() {
        // Mono layout has only Mono label; stereo has FL/FR — no overlap, all silence.
        let src = AudioChannelLayout::stereo();
        let dst = AudioChannelLayout::mono();
        let conv = ChannelConverter::new(src, dst);
        let input = vec![0.5_f32, 0.5, 0.5, 0.5];
        let output = conv.remap(&input);
        assert_eq!(output.len(), 2); // 2 frames × 1 ch
        for s in &output {
            assert!((*s).abs() < 1e-6);
        }
    }

    #[test]
    fn test_converter_5_1_extracts_center() {
        let src = AudioChannelLayout::surround_5_1();
        let dst = AudioChannelLayout::new(vec![ChannelLabel::Center]);
        let conv = ChannelConverter::new(src, dst);
        // One frame: FL=0, FR=0, C=0.9, LFE=0, RL=0, RR=0
        let input = vec![0.0_f32, 0.0, 0.9, 0.0, 0.0, 0.0];
        let output = conv.remap(&input);
        assert_eq!(output.len(), 1);
        assert!((output[0] - 0.9).abs() < 1e-6);
    }
}
