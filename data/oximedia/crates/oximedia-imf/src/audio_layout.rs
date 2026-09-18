#![allow(dead_code)]
//! Audio channel layout definitions for IMF packages.
//!
//! Implements SMPTE ST 2067-2 audio channel layout handling for IMF,
//! including support for:
//!
//! - **Standard layouts** - Mono, stereo, 5.1, 7.1, Atmos bed layouts
//! - **Channel mapping** - Map between different layout conventions
//! - **Soundfield groups** - SMPTE MCA (Multi-Channel Audio) label support
//! - **Layout validation** - Ensure audio tracks match expected configurations

use std::collections::HashMap;
use std::fmt;

/// Standard audio channel identifiers per SMPTE ST 377-4.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AudioChannel {
    /// Left channel.
    Left,
    /// Right channel.
    Right,
    /// Center channel.
    Center,
    /// Low Frequency Effects (subwoofer).
    Lfe,
    /// Left Surround.
    LeftSurround,
    /// Right Surround.
    RightSurround,
    /// Left Surround Rear (7.1).
    LeftSurroundRear,
    /// Right Surround Rear (7.1).
    RightSurroundRear,
    /// Left Top Front (height channel).
    LeftTopFront,
    /// Right Top Front (height channel).
    RightTopFront,
    /// Left Top Rear (height channel).
    LeftTopRear,
    /// Right Top Rear (height channel).
    RightTopRear,
    /// Hearing/Visually Impaired narration.
    HearingImpaired,
    /// Visually Impaired narration.
    VisuallyImpaired,
    /// Mono channel (single channel).
    Mono,
}

impl AudioChannel {
    /// Return the SMPTE MCA tag symbol.
    pub fn mca_symbol(&self) -> &'static str {
        match self {
            Self::Left => "L",
            Self::Right => "R",
            Self::Center => "C",
            Self::Lfe => "LFE",
            Self::LeftSurround => "Ls",
            Self::RightSurround => "Rs",
            Self::LeftSurroundRear => "Lrs",
            Self::RightSurroundRear => "Rrs",
            Self::LeftTopFront => "Ltf",
            Self::RightTopFront => "Rtf",
            Self::LeftTopRear => "Ltr",
            Self::RightTopRear => "Rtr",
            Self::HearingImpaired => "HI",
            Self::VisuallyImpaired => "VIN",
            Self::Mono => "M",
        }
    }

    /// Return the human-readable name.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Left => "Left",
            Self::Right => "Right",
            Self::Center => "Center",
            Self::Lfe => "LFE",
            Self::LeftSurround => "Left Surround",
            Self::RightSurround => "Right Surround",
            Self::LeftSurroundRear => "Left Surround Rear",
            Self::RightSurroundRear => "Right Surround Rear",
            Self::LeftTopFront => "Left Top Front",
            Self::RightTopFront => "Right Top Front",
            Self::LeftTopRear => "Left Top Rear",
            Self::RightTopRear => "Right Top Rear",
            Self::HearingImpaired => "Hearing Impaired",
            Self::VisuallyImpaired => "Visually Impaired",
            Self::Mono => "Mono",
        }
    }

    /// Try to parse from an MCA symbol string.
    pub fn from_mca_symbol(s: &str) -> Option<Self> {
        match s {
            "L" => Some(Self::Left),
            "R" => Some(Self::Right),
            "C" => Some(Self::Center),
            "LFE" => Some(Self::Lfe),
            "Ls" => Some(Self::LeftSurround),
            "Rs" => Some(Self::RightSurround),
            "Lrs" => Some(Self::LeftSurroundRear),
            "Rrs" => Some(Self::RightSurroundRear),
            "Ltf" => Some(Self::LeftTopFront),
            "Rtf" => Some(Self::RightTopFront),
            "Ltr" => Some(Self::LeftTopRear),
            "Rtr" => Some(Self::RightTopRear),
            "HI" => Some(Self::HearingImpaired),
            "VIN" => Some(Self::VisuallyImpaired),
            "M" => Some(Self::Mono),
            _ => None,
        }
    }
}

impl fmt::Display for AudioChannel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.mca_symbol())
    }
}

/// Predefined audio channel layouts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LayoutPreset {
    /// Mono (1 channel).
    Mono,
    /// Stereo (2 channels: L, R).
    Stereo,
    /// 5.1 Surround (6 channels: L, R, C, LFE, Ls, Rs).
    Surround51,
    /// 7.1 Surround (8 channels: L, R, C, LFE, Ls, Rs, Lrs, Rrs).
    Surround71,
    /// 7.1.4 Immersive (12 channels: 7.1 + 4 height).
    Immersive714,
    /// Stereo + HI + VI (4 channels).
    StereoAccessibility,
    /// 5.1 + HI + VI (8 channels).
    Surround51Accessibility,
}

impl LayoutPreset {
    /// Return the channel count for this preset.
    pub fn channel_count(&self) -> usize {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
            Self::Surround51 => 6,
            Self::Surround71 => 8,
            Self::Immersive714 => 12,
            Self::StereoAccessibility => 4,
            Self::Surround51Accessibility => 8,
        }
    }

    /// Return the channels in order for this preset.
    pub fn channels(&self) -> Vec<AudioChannel> {
        match self {
            Self::Mono => vec![AudioChannel::Mono],
            Self::Stereo => vec![AudioChannel::Left, AudioChannel::Right],
            Self::Surround51 => vec![
                AudioChannel::Left,
                AudioChannel::Right,
                AudioChannel::Center,
                AudioChannel::Lfe,
                AudioChannel::LeftSurround,
                AudioChannel::RightSurround,
            ],
            Self::Surround71 => vec![
                AudioChannel::Left,
                AudioChannel::Right,
                AudioChannel::Center,
                AudioChannel::Lfe,
                AudioChannel::LeftSurround,
                AudioChannel::RightSurround,
                AudioChannel::LeftSurroundRear,
                AudioChannel::RightSurroundRear,
            ],
            Self::Immersive714 => vec![
                AudioChannel::Left,
                AudioChannel::Right,
                AudioChannel::Center,
                AudioChannel::Lfe,
                AudioChannel::LeftSurround,
                AudioChannel::RightSurround,
                AudioChannel::LeftSurroundRear,
                AudioChannel::RightSurroundRear,
                AudioChannel::LeftTopFront,
                AudioChannel::RightTopFront,
                AudioChannel::LeftTopRear,
                AudioChannel::RightTopRear,
            ],
            Self::StereoAccessibility => vec![
                AudioChannel::Left,
                AudioChannel::Right,
                AudioChannel::HearingImpaired,
                AudioChannel::VisuallyImpaired,
            ],
            Self::Surround51Accessibility => vec![
                AudioChannel::Left,
                AudioChannel::Right,
                AudioChannel::Center,
                AudioChannel::Lfe,
                AudioChannel::LeftSurround,
                AudioChannel::RightSurround,
                AudioChannel::HearingImpaired,
                AudioChannel::VisuallyImpaired,
            ],
        }
    }

    /// Return the SMPTE label for this layout.
    pub fn smpte_label(&self) -> &'static str {
        match self {
            Self::Mono => "MCA:Mono",
            Self::Stereo => "MCA:ST",
            Self::Surround51 => "MCA:51",
            Self::Surround71 => "MCA:71",
            Self::Immersive714 => "MCA:714",
            Self::StereoAccessibility => "MCA:ST+HIVIN",
            Self::Surround51Accessibility => "MCA:51+HIVIN",
        }
    }
}

impl fmt::Display for LayoutPreset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.smpte_label())
    }
}

/// A soundfield group as defined in SMPTE ST 377-4 / ST 2067-2.
#[derive(Clone, Debug)]
pub struct SoundfieldGroup {
    /// Group identifier (UUID or label).
    pub id: String,
    /// The MCA tag symbol for the soundfield group.
    pub mca_tag_symbol: String,
    /// Language tag (RFC 5646).
    pub language: Option<String>,
    /// Channels in this group.
    pub channels: Vec<AudioChannel>,
}

impl SoundfieldGroup {
    /// Create a new soundfield group.
    pub fn new(
        id: impl Into<String>,
        mca_tag: impl Into<String>,
        channels: Vec<AudioChannel>,
    ) -> Self {
        Self {
            id: id.into(),
            mca_tag_symbol: mca_tag.into(),
            language: None,
            channels,
        }
    }

    /// Set the language tag.
    pub fn with_language(mut self, lang: impl Into<String>) -> Self {
        self.language = Some(lang.into());
        self
    }

    /// Get the number of channels.
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    /// Check if this group matches a known layout preset.
    pub fn matches_preset(&self, preset: LayoutPreset) -> bool {
        self.channels == preset.channels()
    }
}

/// An audio layout for an IMF track file.
#[derive(Clone, Debug)]
pub struct AudioLayout {
    /// Track file identifier.
    pub track_id: String,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Bit depth.
    pub bit_depth: u16,
    /// Soundfield groups.
    pub soundfield_groups: Vec<SoundfieldGroup>,
    /// Total channel count.
    pub total_channels: usize,
}

impl AudioLayout {
    /// Create a new audio layout.
    pub fn new(track_id: impl Into<String>, sample_rate: u32, bit_depth: u16) -> Self {
        Self {
            track_id: track_id.into(),
            sample_rate,
            bit_depth,
            soundfield_groups: Vec::new(),
            total_channels: 0,
        }
    }

    /// Create from a layout preset.
    pub fn from_preset(
        track_id: impl Into<String>,
        sample_rate: u32,
        bit_depth: u16,
        preset: LayoutPreset,
    ) -> Self {
        let channels = preset.channels();
        let count = channels.len();
        let group = SoundfieldGroup::new("sg-main", preset.smpte_label(), channels);
        Self {
            track_id: track_id.into(),
            sample_rate,
            bit_depth,
            soundfield_groups: vec![group],
            total_channels: count,
        }
    }

    /// Add a soundfield group.
    pub fn add_soundfield_group(&mut self, group: SoundfieldGroup) {
        self.total_channels += group.channel_count();
        self.soundfield_groups.push(group);
    }

    /// Get a flat list of all channels across all soundfield groups.
    pub fn all_channels(&self) -> Vec<AudioChannel> {
        self.soundfield_groups
            .iter()
            .flat_map(|g| g.channels.iter().copied())
            .collect()
    }

    /// Validate the audio layout.
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();

        if self.sample_rate == 0 {
            errors.push("Sample rate is zero".to_string());
        }

        if self.bit_depth == 0 {
            errors.push("Bit depth is zero".to_string());
        }

        // Common valid sample rates
        let valid_rates = [44100, 48000, 96000, 192000];
        if !valid_rates.contains(&self.sample_rate) && self.sample_rate != 0 {
            errors.push(format!(
                "Unusual sample rate: {} Hz (expected one of {:?})",
                self.sample_rate, valid_rates
            ));
        }

        // Validate bit depth
        let valid_depths = [16, 24, 32];
        if !valid_depths.contains(&self.bit_depth) && self.bit_depth != 0 {
            errors.push(format!(
                "Unusual bit depth: {} (expected one of {:?})",
                self.bit_depth, valid_depths
            ));
        }

        // Check channel count consistency
        let actual_channels: usize = self
            .soundfield_groups
            .iter()
            .map(|g| g.channel_count())
            .sum();
        if actual_channels != self.total_channels {
            errors.push(format!(
                "Channel count mismatch: total_channels={} but groups contain {} channels",
                self.total_channels, actual_channels
            ));
        }

        if self.soundfield_groups.is_empty() {
            errors.push("No soundfield groups defined".to_string());
        }

        errors
    }
}

/// Build a channel mapping between two audio layouts.
///
/// Returns a map from source channel index to destination channel index.
#[allow(clippy::cast_precision_loss)]
pub fn build_channel_mapping(source: &AudioLayout, dest: &AudioLayout) -> HashMap<usize, usize> {
    let src_channels = source.all_channels();
    let dst_channels = dest.all_channels();
    let mut mapping = HashMap::new();

    for (si, src_ch) in src_channels.iter().enumerate() {
        for (di, dst_ch) in dst_channels.iter().enumerate() {
            if src_ch == dst_ch {
                mapping.insert(si, di);
                break;
            }
        }
    }

    mapping
}

/// Compute the total bitrate for an audio layout in bits per second.
#[allow(clippy::cast_precision_loss)]
pub fn compute_bitrate(layout: &AudioLayout) -> u64 {
    u64::from(layout.sample_rate) * u64::from(layout.bit_depth) * layout.total_channels as u64
}

/// Check if a downmix from source to destination is possible without losing main channels.
pub fn can_downmix(source: LayoutPreset, dest: LayoutPreset) -> bool {
    source.channel_count() >= dest.channel_count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_channel_mca_symbol() {
        assert_eq!(AudioChannel::Left.mca_symbol(), "L");
        assert_eq!(AudioChannel::Right.mca_symbol(), "R");
        assert_eq!(AudioChannel::Lfe.mca_symbol(), "LFE");
        assert_eq!(AudioChannel::LeftSurroundRear.mca_symbol(), "Lrs");
    }

    #[test]
    fn test_audio_channel_from_mca_symbol() {
        assert_eq!(AudioChannel::from_mca_symbol("L"), Some(AudioChannel::Left));
        assert_eq!(
            AudioChannel::from_mca_symbol("LFE"),
            Some(AudioChannel::Lfe)
        );
        assert!(AudioChannel::from_mca_symbol("INVALID").is_none());
    }

    #[test]
    fn test_layout_preset_channel_count() {
        assert_eq!(LayoutPreset::Mono.channel_count(), 1);
        assert_eq!(LayoutPreset::Stereo.channel_count(), 2);
        assert_eq!(LayoutPreset::Surround51.channel_count(), 6);
        assert_eq!(LayoutPreset::Surround71.channel_count(), 8);
        assert_eq!(LayoutPreset::Immersive714.channel_count(), 12);
    }

    #[test]
    fn test_layout_preset_channels() {
        let stereo = LayoutPreset::Stereo.channels();
        assert_eq!(stereo.len(), 2);
        assert_eq!(stereo[0], AudioChannel::Left);
        assert_eq!(stereo[1], AudioChannel::Right);

        let surround = LayoutPreset::Surround51.channels();
        assert_eq!(surround.len(), 6);
        assert_eq!(surround[3], AudioChannel::Lfe);
    }

    #[test]
    fn test_layout_preset_smpte_label() {
        assert_eq!(LayoutPreset::Stereo.smpte_label(), "MCA:ST");
        assert_eq!(LayoutPreset::Surround51.smpte_label(), "MCA:51");
    }

    #[test]
    fn test_soundfield_group_creation() {
        let group = SoundfieldGroup::new(
            "sg-001",
            "ST",
            vec![AudioChannel::Left, AudioChannel::Right],
        )
        .with_language("en");

        assert_eq!(group.channel_count(), 2);
        assert_eq!(group.language, Some("en".to_string()));
    }

    #[test]
    fn test_soundfield_matches_preset() {
        let group = SoundfieldGroup::new("sg-001", "ST", LayoutPreset::Stereo.channels());
        assert!(group.matches_preset(LayoutPreset::Stereo));
        assert!(!group.matches_preset(LayoutPreset::Surround51));
    }

    #[test]
    fn test_audio_layout_from_preset() {
        let layout = AudioLayout::from_preset("track-001", 48000, 24, LayoutPreset::Surround51);
        assert_eq!(layout.total_channels, 6);
        assert_eq!(layout.sample_rate, 48000);
        assert_eq!(layout.bit_depth, 24);
        assert_eq!(layout.soundfield_groups.len(), 1);
    }

    #[test]
    fn test_audio_layout_all_channels() {
        let layout = AudioLayout::from_preset("t1", 48000, 24, LayoutPreset::Surround71);
        let channels = layout.all_channels();
        assert_eq!(channels.len(), 8);
    }

    #[test]
    fn test_audio_layout_validate_valid() {
        let layout = AudioLayout::from_preset("t1", 48000, 24, LayoutPreset::Stereo);
        let errors = layout.validate();
        assert!(errors.is_empty(), "Valid layout had errors: {:?}", errors);
    }

    #[test]
    fn test_audio_layout_validate_zero_sample_rate() {
        let layout = AudioLayout::from_preset("t1", 0, 24, LayoutPreset::Stereo);
        let errors = layout.validate();
        assert!(!errors.is_empty());
        assert!(errors.iter().any(|e| e.contains("Sample rate")));
    }

    #[test]
    fn test_build_channel_mapping() {
        let src = AudioLayout::from_preset("s", 48000, 24, LayoutPreset::Surround51);
        let dst = AudioLayout::from_preset("d", 48000, 24, LayoutPreset::Stereo);
        let mapping = build_channel_mapping(&src, &dst);
        // L->L and R->R should map
        assert_eq!(mapping.get(&0), Some(&0)); // L -> L
        assert_eq!(mapping.get(&1), Some(&1)); // R -> R
                                               // C, LFE, Ls, Rs should not map
        assert!(mapping.get(&2).is_none());
    }

    #[test]
    fn test_compute_bitrate() {
        let layout = AudioLayout::from_preset("t1", 48000, 24, LayoutPreset::Stereo);
        let bitrate = compute_bitrate(&layout);
        // 48000 * 24 * 2 = 2_304_000
        assert_eq!(bitrate, 2_304_000);
    }

    #[test]
    fn test_can_downmix() {
        assert!(can_downmix(LayoutPreset::Surround51, LayoutPreset::Stereo));
        assert!(can_downmix(
            LayoutPreset::Surround71,
            LayoutPreset::Surround51
        ));
        assert!(!can_downmix(LayoutPreset::Stereo, LayoutPreset::Surround51));
    }

    #[test]
    fn test_layout_preset_display() {
        assert_eq!(format!("{}", LayoutPreset::Surround51), "MCA:51");
    }

    #[test]
    fn test_audio_channel_display() {
        assert_eq!(format!("{}", AudioChannel::Center), "C");
    }

    #[test]
    fn test_accessibility_layout() {
        let layout =
            AudioLayout::from_preset("t1", 48000, 24, LayoutPreset::Surround51Accessibility);
        assert_eq!(layout.total_channels, 8);
        let channels = layout.all_channels();
        assert!(channels.contains(&AudioChannel::HearingImpaired));
        assert!(channels.contains(&AudioChannel::VisuallyImpaired));
    }
}
