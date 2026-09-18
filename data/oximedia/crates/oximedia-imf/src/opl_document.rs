//! Output Profile List (OPL) document model – SMPTE ST 2067-8.
//!
//! An OPL describes the required output characteristics for each composition
//! in an IMF package (channel count, frame rate, HDR mode, etc.).

#![allow(dead_code)]

/// Defines the audio and video output characteristics for a specific delivery target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputProfile {
    /// Human-readable name for this profile.
    pub name: String,
    /// Number of audio channels (2 = stereo, 6 = 5.1, 8 = 7.1, …).
    pub audio_channels: u8,
    /// Target frame rate numerator.
    pub frame_rate_num: u32,
    /// Target frame rate denominator.
    pub frame_rate_den: u32,
    /// Whether HDR output is required.
    pub hdr: bool,
}

impl OutputProfile {
    /// Create a new output profile.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        audio_channels: u8,
        frame_rate_num: u32,
        frame_rate_den: u32,
        hdr: bool,
    ) -> Self {
        Self {
            name: name.into(),
            audio_channels,
            frame_rate_num,
            frame_rate_den,
            hdr,
        }
    }

    /// Returns `true` when this profile targets stereo (2-channel) audio.
    #[must_use]
    pub fn is_stereo(&self) -> bool {
        self.audio_channels == 2
    }

    /// Returns `true` when this profile requires surround audio (> 2 channels).
    #[must_use]
    pub fn is_surround(&self) -> bool {
        self.audio_channels > 2
    }

    /// Frame rate as a floating-point value (approximate).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn frame_rate_f64(&self) -> f64 {
        if self.frame_rate_den == 0 {
            0.0
        } else {
            self.frame_rate_num as f64 / self.frame_rate_den as f64
        }
    }
}

// ---------------------------------------------------------------------------

/// One entry in an OPL document, linking a composition to a profile.
#[derive(Debug, Clone)]
pub struct OplEntry {
    /// UUID of the CPL this entry applies to.
    pub cpl_id: String,
    /// The output profile required for this composition.
    pub profile: OutputProfile,
    /// Optional annotation / label.
    pub annotation: Option<String>,
}

impl OplEntry {
    /// Create a new entry.
    #[must_use]
    pub fn new(cpl_id: impl Into<String>, profile: OutputProfile) -> Self {
        Self {
            cpl_id: cpl_id.into(),
            profile,
            annotation: None,
        }
    }

    /// Returns `true` when this entry's profile matches `other` (by name equality).
    #[must_use]
    pub fn matches_profile(&self, other: &OutputProfile) -> bool {
        self.profile.name == other.name
    }

    /// Returns `true` if the entry is well-formed (non-empty CPL UUID).
    #[must_use]
    pub fn is_valid(&self) -> bool {
        !self.cpl_id.is_empty()
    }
}

// ---------------------------------------------------------------------------

/// In-memory representation of an OPL document.
#[derive(Debug, Clone, Default)]
pub struct OplDocument {
    /// UUID of this OPL.
    pub opl_id: String,
    entries: Vec<OplEntry>,
}

impl OplDocument {
    /// Create a new OPL document.
    #[must_use]
    pub fn new(opl_id: impl Into<String>) -> Self {
        Self {
            opl_id: opl_id.into(),
            entries: Vec::new(),
        }
    }

    /// Append an entry to this document.
    pub fn add_entry(&mut self, entry: OplEntry) {
        self.entries.push(entry);
    }

    /// Return all entries that match a given output profile by name.
    #[must_use]
    pub fn find_for_profile(&self, profile: &OutputProfile) -> Vec<&OplEntry> {
        self.entries
            .iter()
            .filter(|e| e.matches_profile(profile))
            .collect()
    }

    /// Number of entries in this document.
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Iterate over all entries.
    pub fn entries(&self) -> impl Iterator<Item = &OplEntry> {
        self.entries.iter()
    }

    /// Returns `true` if all entries are valid.
    #[must_use]
    pub fn all_valid(&self) -> bool {
        self.entries.iter().all(|e| e.is_valid())
    }

    /// Find entry by CPL UUID.
    #[must_use]
    pub fn find_by_cpl_id(&self, cpl_id: &str) -> Option<&OplEntry> {
        self.entries.iter().find(|e| e.cpl_id == cpl_id)
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn stereo_profile() -> OutputProfile {
        OutputProfile::new("stereo-sdr", 2, 24, 1, false)
    }

    fn surround_profile() -> OutputProfile {
        OutputProfile::new("surround-hdr", 6, 24, 1, true)
    }

    #[test]
    fn test_profile_is_stereo() {
        assert!(stereo_profile().is_stereo());
    }

    #[test]
    fn test_profile_not_surround_when_stereo() {
        assert!(!stereo_profile().is_surround());
    }

    #[test]
    fn test_profile_is_surround() {
        assert!(surround_profile().is_surround());
    }

    #[test]
    fn test_profile_not_stereo_when_surround() {
        assert!(!surround_profile().is_stereo());
    }

    #[test]
    fn test_profile_frame_rate_f64() {
        let p = OutputProfile::new("p", 2, 30000, 1001, false);
        let fps = p.frame_rate_f64();
        assert!((fps - 29.97).abs() < 0.01);
    }

    #[test]
    fn test_profile_frame_rate_zero_den() {
        let p = OutputProfile::new("p", 2, 24, 0, false);
        assert_eq!(p.frame_rate_f64(), 0.0);
    }

    #[test]
    fn test_entry_matches_profile_same_name() {
        let e = OplEntry::new("cpl-001", stereo_profile());
        assert!(e.matches_profile(&stereo_profile()));
    }

    #[test]
    fn test_entry_does_not_match_different_profile() {
        let e = OplEntry::new("cpl-001", stereo_profile());
        assert!(!e.matches_profile(&surround_profile()));
    }

    #[test]
    fn test_entry_is_valid_with_id() {
        let e = OplEntry::new("cpl-001", stereo_profile());
        assert!(e.is_valid());
    }

    #[test]
    fn test_entry_invalid_when_empty_cpl_id() {
        let e = OplEntry::new("", stereo_profile());
        assert!(!e.is_valid());
    }

    #[test]
    fn test_document_add_and_count() {
        let mut doc = OplDocument::new("opl-001");
        doc.add_entry(OplEntry::new("cpl-001", stereo_profile()));
        doc.add_entry(OplEntry::new("cpl-002", surround_profile()));
        assert_eq!(doc.entry_count(), 2);
    }

    #[test]
    fn test_document_find_for_profile() {
        let mut doc = OplDocument::new("opl-001");
        doc.add_entry(OplEntry::new("cpl-001", stereo_profile()));
        doc.add_entry(OplEntry::new("cpl-002", stereo_profile()));
        doc.add_entry(OplEntry::new("cpl-003", surround_profile()));
        let hits = doc.find_for_profile(&stereo_profile());
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn test_document_find_for_profile_no_match() {
        let doc = OplDocument::new("opl-001");
        let hits = doc.find_for_profile(&stereo_profile());
        assert!(hits.is_empty());
    }

    #[test]
    fn test_document_find_by_cpl_id() {
        let mut doc = OplDocument::new("opl-001");
        doc.add_entry(OplEntry::new("cpl-abc", stereo_profile()));
        assert!(doc.find_by_cpl_id("cpl-abc").is_some());
        assert!(doc.find_by_cpl_id("cpl-xyz").is_none());
    }

    #[test]
    fn test_document_all_valid() {
        let mut doc = OplDocument::new("opl-001");
        doc.add_entry(OplEntry::new("cpl-001", stereo_profile()));
        assert!(doc.all_valid());
    }

    #[test]
    fn test_document_not_all_valid_with_empty_id() {
        let mut doc = OplDocument::new("opl-001");
        doc.add_entry(OplEntry::new("", stereo_profile()));
        assert!(!doc.all_valid());
    }
}
