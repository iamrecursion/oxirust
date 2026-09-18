//! AFD and WSS handling for broadcast conform.
//!
//! Active Format Description (AFD) codes and Wide Screen Signaling (WSS)
//! are used in broadcast media to signal the aspect ratio and active picture area.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use serde::{Deserialize, Serialize};

/// Active Format Description (AFD) code as defined in SMPTE 2016-1.
///
/// The 4-bit AFD value describes the active picture area within the coded frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum AfdCode {
    /// AFD 0000 – Undefined
    Undefined = 0,
    /// AFD 1000 – Full frame, same as coded frame (16:9 or 4:3)
    FullFrame = 8,
    /// AFD 1001 – 4:3 content centered in 16:9 coded frame
    Box43In169 = 9,
    /// AFD 1010 – 16:9 content with alternative 14:9 center crop
    Box169Alt149 = 10,
    /// AFD 1011 – 14:9 content top-aligned in 16:9 coded frame
    Box149Top = 11,
    /// AFD 1101 – 4:3 content, top-aligned in 16:9 coded frame
    Box43Top = 13,
    /// AFD 1110 – 16:9 letterbox content in 4:3 coded frame
    Box169In43 = 14,
    /// AFD 1111 – 16:9 content, same as coded 16:9 frame
    FullFrame169 = 15,
}

impl AfdCode {
    /// Parse an AFD code from a raw 4-bit value (0–15).
    #[must_use]
    pub fn from_raw(value: u8) -> Option<Self> {
        match value & 0x0F {
            0 => Some(Self::Undefined),
            8 => Some(Self::FullFrame),
            9 => Some(Self::Box43In169),
            10 => Some(Self::Box169Alt149),
            11 => Some(Self::Box149Top),
            13 => Some(Self::Box43Top),
            14 => Some(Self::Box169In43),
            15 => Some(Self::FullFrame169),
            _ => None,
        }
    }

    /// Returns the raw 4-bit value.
    #[must_use]
    pub const fn raw_value(self) -> u8 {
        self as u8
    }

    /// Returns a human-readable description.
    #[must_use]
    pub const fn description(self) -> &'static str {
        match self {
            Self::Undefined => "Undefined",
            Self::FullFrame => "Full frame (same as coded frame)",
            Self::Box43In169 => "4:3 content centered in 16:9 coded frame",
            Self::Box169Alt149 => "16:9 content with 14:9 alternative center crop",
            Self::Box149Top => "14:9 content top-aligned in 16:9 coded frame",
            Self::Box43Top => "4:3 content top-aligned in 16:9 coded frame",
            Self::Box169In43 => "16:9 letterbox in 4:3 coded frame",
            Self::FullFrame169 => "Full 16:9 frame",
        }
    }

    /// Returns the nominal display aspect ratio of the active picture.
    ///
    /// Returns `(width_ratio, height_ratio)` — these are not pixel values but
    /// the display aspect ratio of the active image content.
    #[must_use]
    pub const fn display_aspect_ratio(self) -> (u32, u32) {
        match self {
            Self::Undefined | Self::FullFrame | Self::FullFrame169 => (16, 9),
            Self::Box43In169 | Self::Box43Top => (4, 3),
            Self::Box169Alt149 | Self::Box149Top => (14, 9),
            Self::Box169In43 => (16, 9),
        }
    }
}

/// WSS (Wide Screen Signaling) mode as used in analog PAL broadcasts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WssMode {
    /// Full-format 4:3 (625 lines)
    FullFormat43,
    /// Letterbox 14:9 top
    Letterbox149Top,
    /// Letterbox 14:9 center
    Letterbox149Center,
    /// Letterbox 16:9 top
    Letterbox169Top,
    /// Letterbox 16:9 center
    Letterbox169Center,
    /// Letterbox >16:9 (anamorphic) center
    LetterboxWideCenter,
    /// Full-format 16:9 (anamorphic)
    FullFormat169,
    /// Full-format 16:9 (with alternative 14:9 center)
    FullFormat169Alt149,
}

impl WssMode {
    /// Parse WSS mode from a 4-bit WSS group-1 bitfield.
    #[must_use]
    pub fn from_bits(bits: u8) -> Option<Self> {
        match bits & 0x0F {
            0b0000 => Some(Self::FullFormat43),
            0b0001 => Some(Self::Letterbox149Top),
            0b0010 => Some(Self::Letterbox149Center),
            0b0011 => Some(Self::Letterbox169Top),
            0b0100 => Some(Self::Letterbox169Center),
            0b0101 => Some(Self::LetterboxWideCenter),
            0b0110 => Some(Self::FullFormat169),
            0b0111 => Some(Self::FullFormat169Alt149),
            _ => None,
        }
    }

    /// Returns the 4-bit bitfield for this WSS mode.
    #[must_use]
    pub const fn to_bits(self) -> u8 {
        match self {
            Self::FullFormat43 => 0b0000,
            Self::Letterbox149Top => 0b0001,
            Self::Letterbox149Center => 0b0010,
            Self::Letterbox169Top => 0b0011,
            Self::Letterbox169Center => 0b0100,
            Self::LetterboxWideCenter => 0b0101,
            Self::FullFormat169 => 0b0110,
            Self::FullFormat169Alt149 => 0b0111,
        }
    }

    /// Returns the nominal aspect ratio signaled by this WSS mode.
    #[must_use]
    pub const fn aspect_ratio(self) -> (u32, u32) {
        match self {
            Self::FullFormat43 => (4, 3),
            Self::Letterbox149Top | Self::Letterbox149Center => (14, 9),
            Self::Letterbox169Top
            | Self::Letterbox169Center
            | Self::FullFormat169
            | Self::FullFormat169Alt149 => (16, 9),
            Self::LetterboxWideCenter => (21, 9),
        }
    }

    /// Returns true if this mode is letterboxed (has black bars top/bottom).
    #[must_use]
    pub const fn is_letterboxed(self) -> bool {
        matches!(
            self,
            Self::Letterbox149Top
                | Self::Letterbox149Center
                | Self::Letterbox169Top
                | Self::Letterbox169Center
                | Self::LetterboxWideCenter
        )
    }
}

/// Holds both AFD and WSS signaling for a video frame or stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AfdWssInfo {
    /// AFD code (if present)
    pub afd: Option<AfdCode>,
    /// WSS mode (if present)
    pub wss: Option<WssMode>,
    /// Bar data: number of black bars (top, bottom) in lines
    pub bar_data: Option<BarData>,
}

/// Vertical bar data indicating position of black bars.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BarData {
    /// Lines of black bar at top (0 = none)
    pub top_bar_lines: u16,
    /// Lines of black bar at bottom (0 = none)
    pub bottom_bar_lines: u16,
    /// Total frame height (for validation)
    pub frame_height: u16,
}

impl BarData {
    /// Create new bar data.
    #[must_use]
    pub const fn new(top: u16, bottom: u16, height: u16) -> Self {
        Self {
            top_bar_lines: top,
            bottom_bar_lines: bottom,
            frame_height: height,
        }
    }

    /// Returns the active picture height (frame height minus bars).
    #[must_use]
    pub const fn active_height(&self) -> u16 {
        self.frame_height
            .saturating_sub(self.top_bar_lines)
            .saturating_sub(self.bottom_bar_lines)
    }

    /// Returns the fraction of the frame that is active picture.
    #[must_use]
    pub fn active_fraction(&self) -> f32 {
        if self.frame_height == 0 {
            return 0.0;
        }
        f32::from(self.active_height()) / f32::from(self.frame_height)
    }
}

impl AfdWssInfo {
    /// Creates an `AfdWssInfo` with only an AFD code set.
    #[must_use]
    pub const fn from_afd(afd: AfdCode) -> Self {
        Self {
            afd: Some(afd),
            wss: None,
            bar_data: None,
        }
    }

    /// Creates an `AfdWssInfo` with only a WSS mode set.
    #[must_use]
    pub const fn from_wss(wss: WssMode) -> Self {
        Self {
            afd: None,
            wss: Some(wss),
            bar_data: None,
        }
    }

    /// Returns true if no AFD or WSS signaling is present.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.afd.is_none() && self.wss.is_none()
    }

    /// Derives the best-guess display aspect ratio from available signaling.
    ///
    /// AFD takes precedence over WSS when both are present.
    #[must_use]
    pub fn display_aspect_ratio(&self) -> Option<(u32, u32)> {
        if let Some(afd) = self.afd {
            return Some(afd.display_aspect_ratio());
        }
        self.wss.map(WssMode::aspect_ratio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_afd_from_raw_valid() {
        assert_eq!(AfdCode::from_raw(8), Some(AfdCode::FullFrame));
        assert_eq!(AfdCode::from_raw(15), Some(AfdCode::FullFrame169));
        assert_eq!(AfdCode::from_raw(9), Some(AfdCode::Box43In169));
    }

    #[test]
    fn test_afd_from_raw_invalid() {
        assert_eq!(AfdCode::from_raw(1), None);
        assert_eq!(AfdCode::from_raw(5), None);
        assert_eq!(AfdCode::from_raw(7), None);
    }

    #[test]
    fn test_afd_raw_value_roundtrip() {
        let code = AfdCode::FullFrame169;
        let raw = code.raw_value();
        assert_eq!(AfdCode::from_raw(raw), Some(code));
    }

    #[test]
    fn test_afd_display_aspect_ratio() {
        assert_eq!(AfdCode::Box43In169.display_aspect_ratio(), (4, 3));
        assert_eq!(AfdCode::FullFrame169.display_aspect_ratio(), (16, 9));
        assert_eq!(AfdCode::Box169In43.display_aspect_ratio(), (16, 9));
    }

    #[test]
    fn test_afd_description_nonempty() {
        for code in [
            AfdCode::Undefined,
            AfdCode::FullFrame,
            AfdCode::Box43In169,
            AfdCode::Box169Alt149,
            AfdCode::FullFrame169,
        ] {
            assert!(!code.description().is_empty());
        }
    }

    #[test]
    fn test_wss_from_bits_valid() {
        assert_eq!(WssMode::from_bits(0b0000), Some(WssMode::FullFormat43));
        assert_eq!(WssMode::from_bits(0b0110), Some(WssMode::FullFormat169));
        assert_eq!(
            WssMode::from_bits(0b0100),
            Some(WssMode::Letterbox169Center)
        );
    }

    #[test]
    fn test_wss_bits_roundtrip() {
        let mode = WssMode::Letterbox169Center;
        let bits = mode.to_bits();
        assert_eq!(WssMode::from_bits(bits), Some(mode));
    }

    #[test]
    fn test_wss_is_letterboxed() {
        assert!(WssMode::Letterbox169Center.is_letterboxed());
        assert!(WssMode::Letterbox149Top.is_letterboxed());
        assert!(!WssMode::FullFormat169.is_letterboxed());
        assert!(!WssMode::FullFormat43.is_letterboxed());
    }

    #[test]
    fn test_wss_aspect_ratio() {
        assert_eq!(WssMode::FullFormat43.aspect_ratio(), (4, 3));
        assert_eq!(WssMode::FullFormat169.aspect_ratio(), (16, 9));
        assert_eq!(WssMode::LetterboxWideCenter.aspect_ratio(), (21, 9));
    }

    #[test]
    fn test_bar_data_active_height() {
        let bar = BarData::new(100, 100, 1080);
        assert_eq!(bar.active_height(), 880);
    }

    #[test]
    fn test_bar_data_active_fraction() {
        let bar = BarData::new(0, 0, 1080);
        assert!((bar.active_fraction() - 1.0).abs() < 1e-5);

        let bar2 = BarData::new(140, 140, 1080);
        let frac = bar2.active_fraction();
        assert!(frac > 0.0 && frac < 1.0);
    }

    #[test]
    fn test_afd_wss_info_display_aspect_ratio_afd_takes_precedence() {
        let info = AfdWssInfo {
            afd: Some(AfdCode::Box43In169),
            wss: Some(WssMode::FullFormat169),
            bar_data: None,
        };
        // AFD should win
        assert_eq!(info.display_aspect_ratio(), Some((4, 3)));
    }

    #[test]
    fn test_afd_wss_info_display_aspect_ratio_wss_fallback() {
        let info = AfdWssInfo {
            afd: None,
            wss: Some(WssMode::FullFormat169),
            bar_data: None,
        };
        assert_eq!(info.display_aspect_ratio(), Some((16, 9)));
    }

    #[test]
    fn test_afd_wss_info_empty() {
        let info = AfdWssInfo {
            afd: None,
            wss: None,
            bar_data: None,
        };
        assert!(info.is_empty());
        assert_eq!(info.display_aspect_ratio(), None);
    }
}
