//! Channel mapping and remapping for complex audio routing.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Standard channel layouts
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChannelLayout {
    /// Mono (1.0)
    Mono,
    /// Stereo (2.0)
    Stereo,
    /// 2.1 (Stereo with LFE)
    Stereo21,
    /// 3.0 (L, R, C)
    Surround30,
    /// 5.0 (L, R, C, Ls, Rs)
    Surround50,
    /// 5.1 (L, R, C, LFE, Ls, Rs)
    Surround51,
    /// 7.1 (L, R, C, LFE, Ls, Rs, Lrs, Rrs)
    Surround71,
    /// 7.1.4 Dolby Atmos (adds Ltf, Rtf, Ltr, Rtr)
    Atmos714,
    /// Custom layout with specified channel count
    Custom(u8),
}

impl ChannelLayout {
    /// Get the number of channels for this layout
    #[must_use]
    pub const fn channel_count(&self) -> u8 {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
            Self::Stereo21 => 3,
            Self::Surround30 => 3,
            Self::Surround50 => 5,
            Self::Surround51 => 6,
            Self::Surround71 => 8,
            Self::Atmos714 => 12,
            Self::Custom(count) => *count,
        }
    }

    /// Check if this layout includes an LFE channel
    #[must_use]
    pub const fn has_lfe(&self) -> bool {
        matches!(
            self,
            Self::Stereo21 | Self::Surround51 | Self::Surround71 | Self::Atmos714
        )
    }
}

/// Individual channel position/role
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChannelPosition {
    /// Left
    Left,
    /// Right
    Right,
    /// Center
    Center,
    /// Low Frequency Effects
    Lfe,
    /// Left Surround
    LeftSurround,
    /// Right Surround
    RightSurround,
    /// Left Rear Surround
    LeftRearSurround,
    /// Right Rear Surround
    RightRearSurround,
    /// Left Top Front
    LeftTopFront,
    /// Right Top Front
    RightTopFront,
    /// Left Top Rear
    LeftTopRear,
    /// Right Top Rear
    RightTopRear,
    /// Custom channel
    Custom(u8),
}

/// Maps a single output channel from multiple input channels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelMap {
    /// Output channel index
    pub output_channel: u8,
    /// Input channels and their mix coefficients
    pub inputs: Vec<(u8, f32)>,
}

impl ChannelMap {
    /// Create a new channel map
    #[must_use]
    pub const fn new(output_channel: u8) -> Self {
        Self {
            output_channel,
            inputs: Vec::new(),
        }
    }

    /// Add an input channel with coefficient
    pub fn add_input(&mut self, input_channel: u8, coefficient: f32) {
        self.inputs.push((input_channel, coefficient));
    }

    /// Create a direct 1:1 mapping
    #[must_use]
    pub fn direct(input_channel: u8, output_channel: u8) -> Self {
        let mut map = Self::new(output_channel);
        map.add_input(input_channel, 1.0);
        map
    }

    /// Create a mix from multiple channels
    #[must_use]
    pub fn mix(output_channel: u8, inputs: Vec<(u8, f32)>) -> Self {
        Self {
            output_channel,
            inputs,
        }
    }
}

/// Complete channel remapping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelRemapper {
    /// Input layout
    pub input_layout: ChannelLayout,
    /// Output layout
    pub output_layout: ChannelLayout,
    /// Channel mappings
    pub maps: Vec<ChannelMap>,
}

impl ChannelRemapper {
    /// Create a new channel remapper
    #[must_use]
    pub fn new(input_layout: ChannelLayout, output_layout: ChannelLayout) -> Self {
        Self {
            input_layout,
            output_layout,
            maps: Vec::new(),
        }
    }

    /// Add a channel map
    pub fn add_map(&mut self, map: ChannelMap) {
        self.maps.push(map);
    }

    /// Create identity mapping (1:1 for matching channels)
    #[must_use]
    pub fn identity(layout: ChannelLayout) -> Self {
        let mut remapper = Self::new(layout, layout);
        let channel_count = layout.channel_count();

        for i in 0..channel_count {
            remapper.add_map(ChannelMap::direct(i, i));
        }

        remapper
    }

    /// Create stereo downmix from 5.1
    #[must_use]
    pub fn downmix_51_to_stereo() -> Self {
        let mut remapper = Self::new(ChannelLayout::Surround51, ChannelLayout::Stereo);

        // Left: L + 0.707*C + 0.707*Ls
        remapper.add_map(ChannelMap::mix(0, vec![(0, 1.0), (2, 0.707), (4, 0.707)]));

        // Right: R + 0.707*C + 0.707*Rs
        remapper.add_map(ChannelMap::mix(1, vec![(1, 1.0), (2, 0.707), (5, 0.707)]));

        remapper
    }

    /// Create stereo to mono downmix
    #[must_use]
    pub fn downmix_stereo_to_mono() -> Self {
        let mut remapper = Self::new(ChannelLayout::Stereo, ChannelLayout::Mono);

        // Mono: 0.5*L + 0.5*R
        remapper.add_map(ChannelMap::mix(0, vec![(0, 0.5), (1, 0.5)]));

        remapper
    }

    /// Create mono to stereo upmix
    #[must_use]
    pub fn upmix_mono_to_stereo() -> Self {
        let mut remapper = Self::new(ChannelLayout::Mono, ChannelLayout::Stereo);

        // Left: Mono
        remapper.add_map(ChannelMap::direct(0, 0));

        // Right: Mono
        remapper.add_map(ChannelMap::direct(0, 1));

        remapper
    }

    /// Create 7.1.4 Atmos bed to 7.1 downmix.
    ///
    /// Atmos 7.1.4 channel order (12 ch):
    ///   0=L, 1=R, 2=C, 3=LFE, 4=Ls, 5=Rs, 6=Lrs, 7=Rrs,
    ///   8=Ltf, 9=Rtf, 10=Ltr, 11=Rtr
    ///
    /// 7.1 channel order (8 ch):
    ///   0=L, 1=R, 2=C, 3=LFE, 4=Ls, 5=Rs, 6=Lrs, 7=Rrs
    ///
    /// Height channels are folded into the corresponding ear-level channels
    /// using the ITU-R BS.2051 recommended coefficient of 0.707 (-3 dB).
    #[must_use]
    pub fn downmix_714_to_71() -> Self {
        let mut remapper = Self::new(ChannelLayout::Atmos714, ChannelLayout::Surround71);
        let h: f32 = 0.707; // height fold-down coefficient

        // L  = L + h*Ltf
        remapper.add_map(ChannelMap::mix(0, vec![(0, 1.0), (8, h)]));
        // R  = R + h*Rtf
        remapper.add_map(ChannelMap::mix(1, vec![(1, 1.0), (9, h)]));
        // C  = C (no height center)
        remapper.add_map(ChannelMap::direct(2, 2));
        // LFE = LFE
        remapper.add_map(ChannelMap::direct(3, 3));
        // Ls  = Ls + h*Ltr
        remapper.add_map(ChannelMap::mix(4, vec![(4, 1.0), (10, h)]));
        // Rs  = Rs + h*Rtr
        remapper.add_map(ChannelMap::mix(5, vec![(5, 1.0), (11, h)]));
        // Lrs = Lrs
        remapper.add_map(ChannelMap::direct(6, 6));
        // Rrs = Rrs
        remapper.add_map(ChannelMap::direct(7, 7));

        remapper
    }

    /// Create 7.1.4 Atmos bed to 5.1 downmix.
    ///
    /// First folds height into ear-level (like `downmix_714_to_71`), then
    /// folds rear surrounds into side surrounds with 0.707 coefficient.
    #[must_use]
    pub fn downmix_714_to_51() -> Self {
        let mut remapper = Self::new(ChannelLayout::Atmos714, ChannelLayout::Surround51);
        let h: f32 = 0.707;
        let r: f32 = 0.707; // rear fold-down coefficient

        // L  = L + h*Ltf
        remapper.add_map(ChannelMap::mix(0, vec![(0, 1.0), (8, h)]));
        // R  = R + h*Rtf
        remapper.add_map(ChannelMap::mix(1, vec![(1, 1.0), (9, h)]));
        // C  = C
        remapper.add_map(ChannelMap::direct(2, 2));
        // LFE = LFE
        remapper.add_map(ChannelMap::direct(3, 3));
        // Ls  = Ls + r*Lrs + h*Ltr
        remapper.add_map(ChannelMap::mix(4, vec![(4, 1.0), (6, r), (10, h)]));
        // Rs  = Rs + r*Rrs + h*Rtr
        remapper.add_map(ChannelMap::mix(5, vec![(5, 1.0), (7, r), (11, h)]));

        remapper
    }

    /// Create 7.1.4 Atmos bed to stereo downmix.
    ///
    /// Full fold-down from 12 channels to 2-channel stereo using
    /// ITU-R BS.775 compatible coefficients.
    #[must_use]
    pub fn downmix_714_to_stereo() -> Self {
        let mut remapper = Self::new(ChannelLayout::Atmos714, ChannelLayout::Stereo);
        let c: f32 = 0.707; // center mix coefficient
        let s: f32 = 0.707; // surround mix coefficient
        let h: f32 = 0.5; // height mix coefficient (reduced for stereo)

        // L = L + c*C + s*Ls + s*Lrs + h*Ltf + h*Ltr
        remapper.add_map(ChannelMap::mix(
            0,
            vec![(0, 1.0), (2, c), (4, s), (6, s), (8, h), (10, h)],
        ));
        // R = R + c*C + s*Rs + s*Rrs + h*Rtf + h*Rtr
        remapper.add_map(ChannelMap::mix(
            1,
            vec![(1, 1.0), (2, c), (5, s), (7, s), (9, h), (11, h)],
        ));

        remapper
    }

    /// Create 5.1 to 7.1.4 Atmos bed upmix.
    ///
    /// 5.1 channel order: 0=L, 1=R, 2=C, 3=LFE, 4=Ls, 5=Rs
    ///
    /// The ear-level channels are copied directly. Rear surrounds (Lrs/Rrs)
    /// are derived from side surrounds. Height channels are derived from the
    /// corresponding ear-level channel attenuated by 0.5 (-6 dB) to create
    /// a subtle height impression.
    #[must_use]
    pub fn upmix_51_to_714() -> Self {
        let mut remapper = Self::new(ChannelLayout::Surround51, ChannelLayout::Atmos714);
        let s: f32 = 0.707; // surround → rear coefficient
        let h: f32 = 0.5; // ear-level → height coefficient

        // L = L
        remapper.add_map(ChannelMap::direct(0, 0));
        // R = R
        remapper.add_map(ChannelMap::direct(1, 1));
        // C = C
        remapper.add_map(ChannelMap::direct(2, 2));
        // LFE = LFE
        remapper.add_map(ChannelMap::direct(3, 3));
        // Ls = Ls
        remapper.add_map(ChannelMap::direct(4, 4));
        // Rs = Rs
        remapper.add_map(ChannelMap::direct(5, 5));
        // Lrs = s * Ls
        remapper.add_map(ChannelMap::mix(6, vec![(4, s)]));
        // Rrs = s * Rs
        remapper.add_map(ChannelMap::mix(7, vec![(5, s)]));
        // Ltf = h * L
        remapper.add_map(ChannelMap::mix(8, vec![(0, h)]));
        // Rtf = h * R
        remapper.add_map(ChannelMap::mix(9, vec![(1, h)]));
        // Ltr = h * Ls
        remapper.add_map(ChannelMap::mix(10, vec![(4, h)]));
        // Rtr = h * Rs
        remapper.add_map(ChannelMap::mix(11, vec![(5, h)]));

        remapper
    }

    /// Create 7.1 to 7.1.4 Atmos bed upmix.
    ///
    /// 7.1 channel order: 0=L, 1=R, 2=C, 3=LFE, 4=Ls, 5=Rs, 6=Lrs, 7=Rrs
    ///
    /// Height channels are derived from corresponding ear-level channels at
    /// 0.5 (-6 dB).
    #[must_use]
    pub fn upmix_71_to_714() -> Self {
        let mut remapper = Self::new(ChannelLayout::Surround71, ChannelLayout::Atmos714);
        let h: f32 = 0.5;

        // Direct copy of all 8 ear-level channels
        for i in 0..8u8 {
            remapper.add_map(ChannelMap::direct(i, i));
        }
        // Ltf = h * L
        remapper.add_map(ChannelMap::mix(8, vec![(0, h)]));
        // Rtf = h * R
        remapper.add_map(ChannelMap::mix(9, vec![(1, h)]));
        // Ltr = h * Lrs
        remapper.add_map(ChannelMap::mix(10, vec![(6, h)]));
        // Rtr = h * Rrs
        remapper.add_map(ChannelMap::mix(11, vec![(7, h)]));

        remapper
    }

    /// Get mapping for a specific output channel
    #[must_use]
    pub fn get_map_for_output(&self, output_channel: u8) -> Option<&ChannelMap> {
        self.maps
            .iter()
            .find(|map| map.output_channel == output_channel)
    }

    /// Validate the remapper configuration
    pub fn validate(&self) -> Result<(), ChannelMapError> {
        let input_count = self.input_layout.channel_count();
        let output_count = self.output_layout.channel_count();

        // Check all input references are valid
        for map in &self.maps {
            if map.output_channel >= output_count {
                return Err(ChannelMapError::InvalidOutputChannel(map.output_channel));
            }

            for &(input_ch, _) in &map.inputs {
                if input_ch >= input_count {
                    return Err(ChannelMapError::InvalidInputChannel(input_ch));
                }
            }
        }

        Ok(())
    }
}

/// Channel mapping manager for complex routing scenarios
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelMapManager {
    /// Named remapper configurations
    remappers: HashMap<String, ChannelRemapper>,
}

impl Default for ChannelMapManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ChannelMapManager {
    /// Create a new channel map manager
    #[must_use]
    pub fn new() -> Self {
        let mut manager = Self {
            remappers: HashMap::new(),
        };

        // Add standard remappers
        manager.add_remapper(
            "51_to_stereo".to_string(),
            ChannelRemapper::downmix_51_to_stereo(),
        );
        manager.add_remapper(
            "stereo_to_mono".to_string(),
            ChannelRemapper::downmix_stereo_to_mono(),
        );
        manager.add_remapper(
            "mono_to_stereo".to_string(),
            ChannelRemapper::upmix_mono_to_stereo(),
        );

        manager
    }

    /// Add a named remapper
    pub fn add_remapper(&mut self, name: String, remapper: ChannelRemapper) {
        self.remappers.insert(name, remapper);
    }

    /// Get a remapper by name
    #[must_use]
    pub fn get_remapper(&self, name: &str) -> Option<&ChannelRemapper> {
        self.remappers.get(name)
    }

    /// Remove a remapper
    pub fn remove_remapper(&mut self, name: &str) -> Option<ChannelRemapper> {
        self.remappers.remove(name)
    }

    /// List all remapper names
    #[must_use]
    pub fn list_remappers(&self) -> Vec<&str> {
        self.remappers.keys().map(String::as_str).collect()
    }
}

/// Errors that can occur in channel mapping
#[derive(Debug, Clone, thiserror::Error)]
pub enum ChannelMapError {
    /// Invalid input channel reference
    #[error("Invalid input channel: {0}")]
    InvalidInputChannel(u8),
    /// Invalid output channel reference
    #[error("Invalid output channel: {0}")]
    InvalidOutputChannel(u8),
    /// Remapper not found
    #[error("Remapper not found: {0}")]
    RemapperNotFound(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_layout() {
        assert_eq!(ChannelLayout::Mono.channel_count(), 1);
        assert_eq!(ChannelLayout::Stereo.channel_count(), 2);
        assert_eq!(ChannelLayout::Surround51.channel_count(), 6);
        assert!(ChannelLayout::Surround51.has_lfe());
        assert!(!ChannelLayout::Stereo.has_lfe());
    }

    #[test]
    fn test_channel_map() {
        let mut map = ChannelMap::new(0);
        map.add_input(0, 1.0);
        map.add_input(1, 0.5);

        assert_eq!(map.output_channel, 0);
        assert_eq!(map.inputs.len(), 2);
    }

    #[test]
    fn test_direct_mapping() {
        let map = ChannelMap::direct(2, 3);
        assert_eq!(map.output_channel, 3);
        assert_eq!(map.inputs.len(), 1);
        assert_eq!(map.inputs[0], (2, 1.0));
    }

    #[test]
    fn test_identity_remapper() {
        let remapper = ChannelRemapper::identity(ChannelLayout::Stereo);
        assert_eq!(remapper.input_layout, ChannelLayout::Stereo);
        assert_eq!(remapper.output_layout, ChannelLayout::Stereo);
        assert_eq!(remapper.maps.len(), 2);
    }

    #[test]
    fn test_51_to_stereo_downmix() {
        let remapper = ChannelRemapper::downmix_51_to_stereo();
        assert_eq!(remapper.input_layout, ChannelLayout::Surround51);
        assert_eq!(remapper.output_layout, ChannelLayout::Stereo);
        assert_eq!(remapper.maps.len(), 2);

        // Left channel should mix from L, C, and Ls
        if let Some(left_map) = remapper.get_map_for_output(0) {
            assert_eq!(left_map.inputs.len(), 3);
        } else {
            panic!("Left channel map not found");
        }
    }

    #[test]
    fn test_stereo_to_mono_downmix() {
        let remapper = ChannelRemapper::downmix_stereo_to_mono();
        assert_eq!(remapper.maps.len(), 1);

        if let Some(mono_map) = remapper.get_map_for_output(0) {
            assert_eq!(mono_map.inputs.len(), 2);
            assert!((mono_map.inputs[0].1 - 0.5).abs() < f32::EPSILON);
            assert!((mono_map.inputs[1].1 - 0.5).abs() < f32::EPSILON);
        } else {
            panic!("Mono map not found");
        }
    }

    #[test]
    fn test_mono_to_stereo_upmix() {
        let remapper = ChannelRemapper::upmix_mono_to_stereo();
        assert_eq!(remapper.maps.len(), 2);

        // Both outputs should come from same input
        for i in 0..2 {
            if let Some(map) = remapper.get_map_for_output(i) {
                assert_eq!(map.inputs.len(), 1);
                assert_eq!(map.inputs[0].0, 0); // Input channel 0
            } else {
                panic!("Output map not found");
            }
        }
    }

    #[test]
    fn test_remapper_validation() {
        let remapper = ChannelRemapper::identity(ChannelLayout::Stereo);
        assert!(remapper.validate().is_ok());

        // Create invalid remapper
        let mut invalid = ChannelRemapper::new(ChannelLayout::Mono, ChannelLayout::Stereo);
        invalid.add_map(ChannelMap::direct(5, 0)); // Invalid input channel

        assert!(matches!(
            invalid.validate(),
            Err(ChannelMapError::InvalidInputChannel(5))
        ));
    }

    #[test]
    fn test_channel_map_manager() {
        let manager = ChannelMapManager::new();

        // Should have default remappers
        assert!(manager.get_remapper("51_to_stereo").is_some());
        assert!(manager.get_remapper("stereo_to_mono").is_some());
        assert!(manager.get_remapper("mono_to_stereo").is_some());

        let remapper_names = manager.list_remappers();
        assert!(remapper_names.len() >= 3);
    }

    #[test]
    fn test_add_custom_remapper() {
        let mut manager = ChannelMapManager::new();

        let custom = ChannelRemapper::identity(ChannelLayout::Surround51);
        manager.add_remapper("my_51_passthrough".to_string(), custom);

        assert!(manager.get_remapper("my_51_passthrough").is_some());
    }

    // -----------------------------------------------------------------------
    // 7.1.4 Atmos downmix / upmix tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_downmix_714_to_71_layout() {
        let remapper = ChannelRemapper::downmix_714_to_71();
        assert_eq!(remapper.input_layout, ChannelLayout::Atmos714);
        assert_eq!(remapper.output_layout, ChannelLayout::Surround71);
        assert_eq!(remapper.maps.len(), 8);
        assert!(remapper.validate().is_ok());
    }

    #[test]
    fn test_downmix_714_to_71_height_fold() {
        let remapper = ChannelRemapper::downmix_714_to_71();
        // L channel (output 0) should mix from input 0 (L) and input 8 (Ltf)
        let left = remapper.get_map_for_output(0).expect("left map");
        assert_eq!(left.inputs.len(), 2);
        assert_eq!(left.inputs[0].0, 0); // L
        assert_eq!(left.inputs[1].0, 8); // Ltf
        assert!((left.inputs[1].1 - 0.707).abs() < 0.001);
    }

    #[test]
    fn test_downmix_714_to_71_direct_channels() {
        let remapper = ChannelRemapper::downmix_714_to_71();
        // C (output 2) is direct from input 2
        let center = remapper.get_map_for_output(2).expect("center map");
        assert_eq!(center.inputs.len(), 1);
        assert_eq!(center.inputs[0].0, 2);
        assert!((center.inputs[0].1 - 1.0).abs() < f32::EPSILON);

        // LFE (output 3) is direct from input 3
        let lfe = remapper.get_map_for_output(3).expect("lfe map");
        assert_eq!(lfe.inputs.len(), 1);
        assert_eq!(lfe.inputs[0].0, 3);
    }

    #[test]
    fn test_downmix_714_to_51_layout() {
        let remapper = ChannelRemapper::downmix_714_to_51();
        assert_eq!(remapper.input_layout, ChannelLayout::Atmos714);
        assert_eq!(remapper.output_layout, ChannelLayout::Surround51);
        assert_eq!(remapper.maps.len(), 6);
        assert!(remapper.validate().is_ok());
    }

    #[test]
    fn test_downmix_714_to_51_surround_fold() {
        let remapper = ChannelRemapper::downmix_714_to_51();
        // Ls (output 4) = Ls + 0.707*Lrs + 0.707*Ltr
        let ls = remapper.get_map_for_output(4).expect("ls map");
        assert_eq!(ls.inputs.len(), 3);
        assert_eq!(ls.inputs[0].0, 4); // Ls
        assert_eq!(ls.inputs[1].0, 6); // Lrs
        assert_eq!(ls.inputs[2].0, 10); // Ltr
    }

    #[test]
    fn test_downmix_714_to_stereo_layout() {
        let remapper = ChannelRemapper::downmix_714_to_stereo();
        assert_eq!(remapper.input_layout, ChannelLayout::Atmos714);
        assert_eq!(remapper.output_layout, ChannelLayout::Stereo);
        assert_eq!(remapper.maps.len(), 2);
        assert!(remapper.validate().is_ok());
    }

    #[test]
    fn test_downmix_714_to_stereo_left_channel_sources() {
        let remapper = ChannelRemapper::downmix_714_to_stereo();
        let left = remapper.get_map_for_output(0).expect("left map");
        // L = L + c*C + s*Ls + s*Lrs + h*Ltf + h*Ltr
        assert_eq!(left.inputs.len(), 6);
        // Check that L is at 1.0
        assert!((left.inputs[0].1 - 1.0).abs() < f32::EPSILON);
        // Center at 0.707
        assert!((left.inputs[1].1 - 0.707).abs() < 0.001);
    }

    #[test]
    fn test_upmix_51_to_714_layout() {
        let remapper = ChannelRemapper::upmix_51_to_714();
        assert_eq!(remapper.input_layout, ChannelLayout::Surround51);
        assert_eq!(remapper.output_layout, ChannelLayout::Atmos714);
        assert_eq!(remapper.maps.len(), 12);
        assert!(remapper.validate().is_ok());
    }

    #[test]
    fn test_upmix_51_to_714_direct_channels() {
        let remapper = ChannelRemapper::upmix_51_to_714();
        // First 6 channels should be direct copies
        for i in 0..6u8 {
            let map = remapper.get_map_for_output(i).expect("map exists");
            assert_eq!(map.inputs.len(), 1);
            assert_eq!(map.inputs[0].0, i);
            assert!((map.inputs[0].1 - 1.0).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_upmix_51_to_714_height_derived() {
        let remapper = ChannelRemapper::upmix_51_to_714();
        // Ltf (output 8) = 0.5 * L (input 0)
        let ltf = remapper.get_map_for_output(8).expect("ltf map");
        assert_eq!(ltf.inputs.len(), 1);
        assert_eq!(ltf.inputs[0].0, 0);
        assert!((ltf.inputs[0].1 - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_upmix_51_to_714_rear_derived() {
        let remapper = ChannelRemapper::upmix_51_to_714();
        // Lrs (output 6) = 0.707 * Ls (input 4)
        let lrs = remapper.get_map_for_output(6).expect("lrs map");
        assert_eq!(lrs.inputs.len(), 1);
        assert_eq!(lrs.inputs[0].0, 4);
        assert!((lrs.inputs[0].1 - 0.707).abs() < 0.001);
    }

    #[test]
    fn test_upmix_71_to_714_layout() {
        let remapper = ChannelRemapper::upmix_71_to_714();
        assert_eq!(remapper.input_layout, ChannelLayout::Surround71);
        assert_eq!(remapper.output_layout, ChannelLayout::Atmos714);
        assert_eq!(remapper.maps.len(), 12);
        assert!(remapper.validate().is_ok());
    }

    #[test]
    fn test_upmix_71_to_714_direct_ear_level() {
        let remapper = ChannelRemapper::upmix_71_to_714();
        // All 8 ear-level channels are direct 1:1
        for i in 0..8u8 {
            let map = remapper.get_map_for_output(i).expect("map exists");
            assert_eq!(map.inputs.len(), 1);
            assert_eq!(map.inputs[0].0, i);
            assert!((map.inputs[0].1 - 1.0).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_upmix_71_to_714_height_from_rears() {
        let remapper = ChannelRemapper::upmix_71_to_714();
        // Ltr (output 10) = 0.5 * Lrs (input 6)
        let ltr = remapper.get_map_for_output(10).expect("ltr map");
        assert_eq!(ltr.inputs.len(), 1);
        assert_eq!(ltr.inputs[0].0, 6);
        assert!((ltr.inputs[0].1 - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_atmos_714_channel_count() {
        assert_eq!(ChannelLayout::Atmos714.channel_count(), 12);
        assert!(ChannelLayout::Atmos714.has_lfe());
    }
}
