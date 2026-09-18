//! Dolby Vision backward compatibility signaling.
//!
//! Provides types and utilities for checking and describing backward
//! compatibility of Dolby Vision streams with SDR, HDR10, and HLG displays.

#![allow(dead_code)]

/// A compatibility layer that a Dolby Vision stream can fall back to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompatLayer {
    /// Standard Dynamic Range (BT.709/BT.2020, SDR).
    Sdr,
    /// HDR10 static metadata (BT.2020, PQ transfer function).
    Hdr10,
    /// HLG broadcast (BT.2020, Hybrid Log-Gamma transfer function).
    HlgBroadcast,
    /// Full Dolby Vision rendering (no backward compat downgrade).
    DolbyVisionFull,
}

impl CompatLayer {
    /// Bit depth associated with this compatibility layer.
    #[must_use]
    pub const fn bit_depth(&self) -> u8 {
        match self {
            Self::Sdr => 8,
            Self::Hdr10 => 10,
            Self::HlgBroadcast => 10,
            Self::DolbyVisionFull => 12,
        }
    }

    /// OETF/EOTF transfer function identifier string.
    #[must_use]
    pub const fn transfer_function(&self) -> &str {
        match self {
            Self::Sdr => "bt709",
            Self::Hdr10 => "smpte2084",
            Self::HlgBroadcast => "arib-std-b67",
            Self::DolbyVisionFull => "dolbyvision",
        }
    }

    /// Color primaries identifier string.
    #[must_use]
    pub const fn color_primaries(&self) -> &str {
        match self {
            Self::Sdr => "bt709",
            Self::Hdr10 => "bt2020",
            Self::HlgBroadcast => "bt2020",
            Self::DolbyVisionFull => "bt2020",
        }
    }
}

/// Describes how a Dolby Vision stream signals backward compatibility.
#[derive(Debug, Clone)]
pub struct CompatSignaling {
    /// The base layer compatibility target.
    pub base_layer: CompatLayer,
    /// Optional enhancement layer compatibility target.
    pub enhancement_layer: Option<CompatLayer>,
    /// Whether the stream is delivered as a single combined track.
    pub single_track: bool,
}

impl CompatSignaling {
    /// Profile 5: single-track, IPT-PQ, backward compatible with HDR10.
    #[must_use]
    pub const fn profile5_single_track() -> Self {
        Self {
            base_layer: CompatLayer::Hdr10,
            enhancement_layer: None,
            single_track: true,
        }
    }

    /// Profile 7: dual-track MEL+BL, backward compatible with HDR10.
    #[must_use]
    pub const fn profile7_dual_track() -> Self {
        Self {
            base_layer: CompatLayer::Hdr10,
            enhancement_layer: Some(CompatLayer::DolbyVisionFull),
            single_track: false,
        }
    }

    /// Profile 8 mezzanine: single-track BL, backward compatible with HDR10.
    #[must_use]
    pub const fn profile8_mezzanine() -> Self {
        Self {
            base_layer: CompatLayer::Hdr10,
            enhancement_layer: None,
            single_track: true,
        }
    }
}

/// Check whether a `CompatSignaling` configuration can be rendered on a
/// display that only supports `target` compatibility level.
///
/// Returns `true` if the base layer satisfies the target or if either layer
/// exactly matches the target.
#[must_use]
pub fn check_backward_compat(signaling: &CompatSignaling, target: CompatLayer) -> bool {
    // SDR displays can always get a picture from any stream (decoder strips DV)
    if target == CompatLayer::Sdr {
        return true;
    }

    // If the base layer matches or exceeds the target
    if layers_satisfy(signaling.base_layer, target) {
        return true;
    }

    // Check enhancement layer
    if let Some(el) = signaling.enhancement_layer {
        if layers_satisfy(el, target) {
            return true;
        }
    }

    false
}

/// Check if `layer` satisfies the requirements of `target`.
///
/// The ordering is: `Sdr` ≤ `Hdr10` = `HlgBroadcast` ≤ `DolbyVisionFull`.
fn layers_satisfy(layer: CompatLayer, target: CompatLayer) -> bool {
    match (layer, target) {
        // Exact match always satisfies
        (a, b) if a == b => true,
        // Full DV satisfies any target
        (CompatLayer::DolbyVisionFull, _) => true,
        // HDR10 satisfies SDR (downconvert is possible)
        (CompatLayer::Hdr10, CompatLayer::Sdr) => true,
        // HLG satisfies SDR
        (CompatLayer::HlgBroadcast, CompatLayer::Sdr) => true,
        // SDR cannot satisfy HDR targets
        _ => false,
    }
}

/// Return the number of distinct media tracks required for the signaling.
///
/// - Single-track signaling: 1 track
/// - Dual-track (MEL+BL) signaling: 2 tracks
#[must_use]
pub fn required_tracks(signaling: &CompatSignaling) -> usize {
    if signaling.single_track {
        1
    } else {
        2
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compat_layer_bit_depth() {
        assert_eq!(CompatLayer::Sdr.bit_depth(), 8);
        assert_eq!(CompatLayer::Hdr10.bit_depth(), 10);
        assert_eq!(CompatLayer::HlgBroadcast.bit_depth(), 10);
        assert_eq!(CompatLayer::DolbyVisionFull.bit_depth(), 12);
    }

    #[test]
    fn test_compat_layer_transfer_function() {
        assert_eq!(CompatLayer::Sdr.transfer_function(), "bt709");
        assert_eq!(CompatLayer::Hdr10.transfer_function(), "smpte2084");
        assert_eq!(
            CompatLayer::HlgBroadcast.transfer_function(),
            "arib-std-b67"
        );
        assert_eq!(
            CompatLayer::DolbyVisionFull.transfer_function(),
            "dolbyvision"
        );
    }

    #[test]
    fn test_compat_layer_color_primaries() {
        assert_eq!(CompatLayer::Sdr.color_primaries(), "bt709");
        assert_eq!(CompatLayer::Hdr10.color_primaries(), "bt2020");
        assert_eq!(CompatLayer::HlgBroadcast.color_primaries(), "bt2020");
        assert_eq!(CompatLayer::DolbyVisionFull.color_primaries(), "bt2020");
    }

    #[test]
    fn test_profile5_single_track() {
        let sig = CompatSignaling::profile5_single_track();
        assert!(sig.single_track);
        assert_eq!(sig.base_layer, CompatLayer::Hdr10);
        assert!(sig.enhancement_layer.is_none());
    }

    #[test]
    fn test_profile7_dual_track() {
        let sig = CompatSignaling::profile7_dual_track();
        assert!(!sig.single_track);
        assert_eq!(sig.base_layer, CompatLayer::Hdr10);
        assert!(sig.enhancement_layer.is_some());
    }

    #[test]
    fn test_profile8_mezzanine() {
        let sig = CompatSignaling::profile8_mezzanine();
        assert!(sig.single_track);
        assert_eq!(sig.base_layer, CompatLayer::Hdr10);
    }

    #[test]
    fn test_check_backward_compat_sdr_always_true() {
        let sig = CompatSignaling::profile5_single_track();
        assert!(check_backward_compat(&sig, CompatLayer::Sdr));
    }

    #[test]
    fn test_check_backward_compat_hdr10_base_satisfies_hdr10() {
        let sig = CompatSignaling::profile5_single_track();
        assert!(check_backward_compat(&sig, CompatLayer::Hdr10));
    }

    #[test]
    fn test_check_backward_compat_hlg_not_satisfied_by_hdr10_base() {
        let sig = CompatSignaling::profile5_single_track();
        // HLG is a different format; HDR10 base cannot satisfy HLG target
        assert!(!check_backward_compat(&sig, CompatLayer::HlgBroadcast));
    }

    #[test]
    fn test_check_backward_compat_full_dv_satisfies_all() {
        let sig = CompatSignaling {
            base_layer: CompatLayer::DolbyVisionFull,
            enhancement_layer: None,
            single_track: true,
        };
        assert!(check_backward_compat(&sig, CompatLayer::Hdr10));
        assert!(check_backward_compat(&sig, CompatLayer::HlgBroadcast));
        assert!(check_backward_compat(&sig, CompatLayer::DolbyVisionFull));
    }

    #[test]
    fn test_required_tracks_single() {
        let sig = CompatSignaling::profile5_single_track();
        assert_eq!(required_tracks(&sig), 1);
    }

    #[test]
    fn test_required_tracks_dual() {
        let sig = CompatSignaling::profile7_dual_track();
        assert_eq!(required_tracks(&sig), 2);
    }
}
