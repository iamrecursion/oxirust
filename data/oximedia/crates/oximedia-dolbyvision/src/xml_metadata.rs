//! Dolby Vision XML metadata structures (DolbyVision_RPU.xml / metadata sidecars).

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// DvXmlVersion
// ---------------------------------------------------------------------------

/// Version number embedded in a Dolby Vision XML metadata document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DvXmlVersion {
    /// Major version component.
    pub major: u32,
    /// Minor version component.
    pub minor: u32,
}

impl DvXmlVersion {
    /// Create a new version.
    #[must_use]
    pub fn new(major: u32, minor: u32) -> Self {
        Self { major, minor }
    }

    /// Returns `true` when `other` has the same major version and a minor
    /// version ≤ this version's minor (forward-compatible check).
    #[must_use]
    pub fn is_compatible(&self, other: &DvXmlVersion) -> bool {
        self.major == other.major && other.minor <= self.minor
    }
}

// ---------------------------------------------------------------------------
// DvXmlGlobalSettings
// ---------------------------------------------------------------------------

/// Global settings section of a Dolby Vision XML metadata document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DvXmlGlobalSettings {
    /// Whether Level 8 (target display) metadata is enabled.
    pub level_8_enabled: bool,
    /// Whether Level 254 (application-specific) metadata is enabled.
    pub level_254_enabled: bool,
    /// Canvas width in pixels.
    pub canvas_width: u32,
    /// Canvas height in pixels.
    pub canvas_height: u32,
}

impl DvXmlGlobalSettings {
    /// Create a new global settings block.
    #[must_use]
    pub fn new(
        level_8_enabled: bool,
        level_254_enabled: bool,
        canvas_width: u32,
        canvas_height: u32,
    ) -> Self {
        Self {
            level_8_enabled,
            level_254_enabled,
            canvas_width,
            canvas_height,
        }
    }

    /// Returns `true` when the canvas resolution is at least 1280×720 (HD).
    #[must_use]
    pub fn is_hd(&self) -> bool {
        self.canvas_width >= 1280 && self.canvas_height >= 720
    }
}

// ---------------------------------------------------------------------------
// DvXmlShot
// ---------------------------------------------------------------------------

/// A single shot / scene entry in a Dolby Vision XML metadata document.
#[derive(Debug, Clone, PartialEq)]
pub struct DvXmlShot {
    /// Frame offset (start frame index within the sequence).
    pub offset: u64,
    /// Number of frames in this shot.
    pub duration: u64,
    /// Optional Level 1 (min, avg, max) luminance tuple (all in nits × 10000).
    pub level1: Option<(u16, u16, u16)>,
}

impl DvXmlShot {
    /// Create a new shot entry.
    #[must_use]
    pub fn new(offset: u64, duration: u64, level1: Option<(u16, u16, u16)>) -> Self {
        Self {
            offset,
            duration,
            level1,
        }
    }

    /// Index of the last frame in this shot (exclusive end frame).
    #[must_use]
    pub fn end_frame(&self) -> u64 {
        self.offset + self.duration
    }
}

// ---------------------------------------------------------------------------
// DvXmlDoc
// ---------------------------------------------------------------------------

/// A complete Dolby Vision XML metadata document.
#[derive(Debug, Clone)]
pub struct DvXmlDoc {
    /// Document format version.
    pub version: DvXmlVersion,
    /// Global settings.
    pub global: DvXmlGlobalSettings,
    /// Ordered list of shots.
    pub shots: Vec<DvXmlShot>,
}

impl DvXmlDoc {
    /// Create a new empty document.
    #[must_use]
    pub fn new(version: DvXmlVersion, global: DvXmlGlobalSettings) -> Self {
        Self {
            version,
            global,
            shots: Vec::new(),
        }
    }

    /// Number of shots in this document.
    #[must_use]
    pub fn shot_count(&self) -> usize {
        self.shots.len()
    }

    /// Sum of all shot durations (total frame count covered by this document).
    #[must_use]
    pub fn total_frames(&self) -> u64 {
        self.shots.iter().map(|s| s.duration).sum()
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- DvXmlVersion ---

    #[test]
    fn test_version_compatible_same() {
        let v = DvXmlVersion::new(2, 5);
        let other = DvXmlVersion::new(2, 3);
        assert!(v.is_compatible(&other));
    }

    #[test]
    fn test_version_compatible_equal() {
        let v = DvXmlVersion::new(2, 5);
        assert!(v.is_compatible(&DvXmlVersion::new(2, 5)));
    }

    #[test]
    fn test_version_incompatible_higher_minor() {
        let v = DvXmlVersion::new(2, 3);
        assert!(!v.is_compatible(&DvXmlVersion::new(2, 5)));
    }

    #[test]
    fn test_version_incompatible_different_major() {
        let v = DvXmlVersion::new(2, 5);
        assert!(!v.is_compatible(&DvXmlVersion::new(3, 0)));
    }

    #[test]
    fn test_version_fields() {
        let v = DvXmlVersion::new(1, 7);
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 7);
    }

    // --- DvXmlGlobalSettings ---

    #[test]
    fn test_global_settings_is_hd_uhd() {
        let g = DvXmlGlobalSettings::new(true, false, 3840, 2160);
        assert!(g.is_hd());
    }

    #[test]
    fn test_global_settings_is_hd_1080() {
        let g = DvXmlGlobalSettings::new(false, false, 1920, 1080);
        assert!(g.is_hd());
    }

    #[test]
    fn test_global_settings_is_hd_720() {
        let g = DvXmlGlobalSettings::new(false, false, 1280, 720);
        assert!(g.is_hd());
    }

    #[test]
    fn test_global_settings_not_hd_sd() {
        let g = DvXmlGlobalSettings::new(false, false, 720, 576);
        assert!(!g.is_hd());
    }

    #[test]
    fn test_global_settings_fields() {
        let g = DvXmlGlobalSettings::new(true, true, 1920, 1080);
        assert!(g.level_8_enabled);
        assert!(g.level_254_enabled);
    }

    // --- DvXmlShot ---

    #[test]
    fn test_shot_end_frame() {
        let shot = DvXmlShot::new(0, 100, None);
        assert_eq!(shot.end_frame(), 100);
    }

    #[test]
    fn test_shot_end_frame_offset() {
        let shot = DvXmlShot::new(200, 50, None);
        assert_eq!(shot.end_frame(), 250);
    }

    #[test]
    fn test_shot_with_level1() {
        let shot = DvXmlShot::new(0, 24, Some((10, 500, 4000)));
        assert_eq!(shot.level1, Some((10, 500, 4000)));
    }

    #[test]
    fn test_shot_zero_duration() {
        let shot = DvXmlShot::new(10, 0, None);
        assert_eq!(shot.end_frame(), 10);
    }

    // --- DvXmlDoc ---

    #[test]
    fn test_doc_shot_count_empty() {
        let doc = DvXmlDoc::new(
            DvXmlVersion::new(2, 0),
            DvXmlGlobalSettings::new(true, false, 1920, 1080),
        );
        assert_eq!(doc.shot_count(), 0);
    }

    #[test]
    fn test_doc_shot_count_after_push() {
        let mut doc = DvXmlDoc::new(
            DvXmlVersion::new(2, 0),
            DvXmlGlobalSettings::new(true, false, 1920, 1080),
        );
        doc.shots.push(DvXmlShot::new(0, 100, None));
        doc.shots.push(DvXmlShot::new(100, 200, None));
        assert_eq!(doc.shot_count(), 2);
    }

    #[test]
    fn test_doc_total_frames() {
        let mut doc = DvXmlDoc::new(
            DvXmlVersion::new(2, 0),
            DvXmlGlobalSettings::new(true, false, 1920, 1080),
        );
        doc.shots.push(DvXmlShot::new(0, 100, None));
        doc.shots.push(DvXmlShot::new(100, 200, None));
        assert_eq!(doc.total_frames(), 300);
    }

    #[test]
    fn test_doc_total_frames_empty() {
        let doc = DvXmlDoc::new(
            DvXmlVersion::new(2, 0),
            DvXmlGlobalSettings::new(true, false, 1920, 1080),
        );
        assert_eq!(doc.total_frames(), 0);
    }
}
