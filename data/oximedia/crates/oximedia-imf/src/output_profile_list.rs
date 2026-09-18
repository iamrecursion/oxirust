//! IMF Output Profile List (OPL) - high-level public API
//!
//! This module provides a clean, high-level API for working with IMF Output
//! Profile Lists as defined by SMPTE ST 2067-8.  It is independent of the
//! lower-level private `opl` module.

#![allow(dead_code)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::cast_precision_loss)]

/// A single output profile describing the technical requirements for one
/// delivery variant.
#[derive(Debug, Clone)]
pub struct ImfOutputProfile {
    /// Unique identifier for this profile (UUID string).
    pub id: String,
    /// Human-readable annotation / label.
    pub annotation: String,
    /// SMPTE signal standard string, e.g. `"SMPTE ST 2084"` for PQ HDR.
    pub signal_standard: String,
    /// Frame-rate numerator.
    pub frame_rate_num: u32,
    /// Frame-rate denominator (must be > 0).
    pub frame_rate_den: u32,
    /// Colour-space identifier, e.g. `"BT.2020"` or `"BT.709"`.
    pub color_space: String,
}

impl ImfOutputProfile {
    /// Create a new `ImfOutputProfile`.
    pub fn new(
        id: impl Into<String>,
        annotation: impl Into<String>,
        signal_standard: impl Into<String>,
        frame_rate_num: u32,
        frame_rate_den: u32,
        color_space: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            annotation: annotation.into(),
            signal_standard: signal_standard.into(),
            frame_rate_num,
            frame_rate_den,
            color_space: color_space.into(),
        }
    }

    /// Return the frame rate as `f32`.  Panics only if `frame_rate_den` is 0.
    pub fn frame_rate(&self) -> f32 {
        self.frame_rate_num as f32 / self.frame_rate_den as f32
    }

    /// Returns `true` when the profile's signal standard or colour space
    /// indicates an HDR format (PQ / HLG / BT.2020).
    pub fn is_hdr(&self) -> bool {
        let sig = self.signal_standard.to_uppercase();
        let cs = self.color_space.to_uppercase();
        sig.contains("2084")
            || sig.contains("2100")
            || sig.contains("HLG")
            || sig.contains("PQ")
            || cs.contains("2020")
            || cs.contains("2100")
    }
}

/// A collection of output profiles for an IMF package.
#[derive(Debug, Clone)]
pub struct ImfOutputProfileList {
    /// All output profiles.
    pub profiles: Vec<ImfOutputProfile>,
    /// Creator string.
    pub creator: String,
    /// Issue date (ISO-8601 string, e.g. `"2024-01-15"`).
    pub issue_date: String,
}

impl ImfOutputProfileList {
    /// Create an empty `ImfOutputProfileList`.
    pub fn new(creator: impl Into<String>, issue_date: impl Into<String>) -> Self {
        Self {
            profiles: Vec::new(),
            creator: creator.into(),
            issue_date: issue_date.into(),
        }
    }

    /// Append a profile to the list.
    pub fn add(&mut self, profile: ImfOutputProfile) {
        self.profiles.push(profile);
    }

    /// Find a profile by its UUID / ID string.
    pub fn find_by_id(&self, id: &str) -> Option<&ImfOutputProfile> {
        self.profiles.iter().find(|p| p.id == id)
    }

    /// Return all HDR profiles.
    pub fn hdr_profiles(&self) -> Vec<&ImfOutputProfile> {
        self.profiles.iter().filter(|p| p.is_hdr()).collect()
    }

    /// Return the number of profiles.
    pub fn profile_count(&self) -> usize {
        self.profiles.len()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sdr_profile(id: &str) -> ImfOutputProfile {
        ImfOutputProfile::new(id, "SDR Profile", "SMPTE ST 274", 24, 1, "BT.709")
    }

    fn hdr_pq_profile(id: &str) -> ImfOutputProfile {
        ImfOutputProfile::new(id, "HDR PQ Profile", "SMPTE ST 2084", 24, 1, "BT.2020")
    }

    fn hdr_hlg_profile(id: &str) -> ImfOutputProfile {
        ImfOutputProfile::new(id, "HLG Profile", "ITU-R BT.2100 HLG", 25, 1, "BT.2020")
    }

    // --- ImfOutputProfile ---

    #[test]
    fn test_frame_rate_24() {
        let p = sdr_profile("p1");
        assert!((p.frame_rate() - 24.0_f32).abs() < 0.001);
    }

    #[test]
    fn test_frame_rate_fractional() {
        let p = ImfOutputProfile::new("p", "label", "ST 274", 24000, 1001, "BT.709");
        let expected = 24000_f32 / 1001_f32;
        assert!((p.frame_rate() - expected).abs() < 0.001);
    }

    #[test]
    fn test_is_hdr_sdr_false() {
        assert!(!sdr_profile("p1").is_hdr());
    }

    #[test]
    fn test_is_hdr_pq_true() {
        assert!(hdr_pq_profile("p2").is_hdr());
    }

    #[test]
    fn test_is_hdr_hlg_true() {
        assert!(hdr_hlg_profile("p3").is_hdr());
    }

    #[test]
    fn test_is_hdr_bt2020_colorspace() {
        let p = ImfOutputProfile::new("p", "label", "SMPTE ST 274", 25, 1, "BT.2020");
        assert!(p.is_hdr());
    }

    // --- ImfOutputProfileList ---

    #[test]
    fn test_profile_count_empty() {
        let opl = ImfOutputProfileList::new("OxiMedia", "2024-01-01");
        assert_eq!(opl.profile_count(), 0);
    }

    #[test]
    fn test_profile_count_after_add() {
        let mut opl = ImfOutputProfileList::new("OxiMedia", "2024-01-01");
        opl.add(sdr_profile("p1"));
        opl.add(hdr_pq_profile("p2"));
        assert_eq!(opl.profile_count(), 2);
    }

    #[test]
    fn test_find_by_id_found() {
        let mut opl = ImfOutputProfileList::new("OxiMedia", "2024-01-01");
        opl.add(sdr_profile("my-id-123"));
        assert!(opl.find_by_id("my-id-123").is_some());
    }

    #[test]
    fn test_find_by_id_not_found() {
        let opl = ImfOutputProfileList::new("OxiMedia", "2024-01-01");
        assert!(opl.find_by_id("ghost").is_none());
    }

    #[test]
    fn test_hdr_profiles_none() {
        let mut opl = ImfOutputProfileList::new("OxiMedia", "2024-01-01");
        opl.add(sdr_profile("p1"));
        assert!(opl.hdr_profiles().is_empty());
    }

    #[test]
    fn test_hdr_profiles_some() {
        let mut opl = ImfOutputProfileList::new("OxiMedia", "2024-01-01");
        opl.add(sdr_profile("p1"));
        opl.add(hdr_pq_profile("p2"));
        opl.add(hdr_hlg_profile("p3"));
        assert_eq!(opl.hdr_profiles().len(), 2);
    }

    #[test]
    fn test_creator_stored() {
        let opl = ImfOutputProfileList::new("MyOrg", "2024-01-01");
        assert_eq!(opl.creator, "MyOrg");
    }

    #[test]
    fn test_issue_date_stored() {
        let opl = ImfOutputProfileList::new("OxiMedia", "2025-06-01");
        assert_eq!(opl.issue_date, "2025-06-01");
    }

    #[test]
    fn test_annotation_stored() {
        let p = ImfOutputProfile::new("id", "My Annotation", "ST 274", 30, 1, "BT.709");
        assert_eq!(p.annotation, "My Annotation");
    }

    #[test]
    fn test_signal_standard_stored() {
        let p = ImfOutputProfile::new("id", "label", "SMPTE ST 2084", 24, 1, "BT.2020");
        assert_eq!(p.signal_standard, "SMPTE ST 2084");
    }
}
