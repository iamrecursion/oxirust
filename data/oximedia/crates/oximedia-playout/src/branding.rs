//! Channel branding / ID management for playout.
//!
//! Manages the set of branding assets (logos, watermarks, end-boards, etc.)
//! used during playout along with an active logo selection.

#![allow(dead_code)]

/// Classification of a branding element.
#[derive(Debug, Clone, PartialEq)]
pub enum BrandingElement {
    /// Station / channel logo overlaid during playout
    ChannelLogo,
    /// Semi-transparent watermark
    Watermark,
    /// Station bug (on-screen identifier)
    BugBug,
    /// End-board shown at end of programmes
    Endboard,
    /// Branded transition between items
    Transition,
}

impl BrandingElement {
    /// Returns true if this element is displayed persistently during playout
    /// (as opposed to being shown only at specific moments).
    pub fn is_persistent(&self) -> bool {
        matches!(self, Self::ChannelLogo | Self::Watermark | Self::BugBug)
    }
}

/// A single branding asset on disk.
#[derive(Debug, Clone)]
pub struct BrandingAsset {
    /// Unique identifier
    pub id: u32,
    /// Human-readable name
    pub name: String,
    /// The type of branding element this asset represents
    pub element: BrandingElement,
    /// Path to the asset file
    pub file_path: String,
    /// Fixed display duration in frames (None = displayed until removed)
    pub duration_frames: Option<u32>,
}

impl BrandingAsset {
    /// Create a new branding asset.
    pub fn new(
        id: u32,
        name: impl Into<String>,
        element: BrandingElement,
        file_path: impl Into<String>,
        duration_frames: Option<u32>,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            element,
            file_path: file_path.into(),
            duration_frames,
        }
    }

    /// Returns true if this asset has an animated (frame-count) duration.
    pub fn is_animated(&self) -> bool {
        self.duration_frames.is_some()
    }

    /// Returns true if this asset has a fixed (non-zero) display duration.
    pub fn has_fixed_duration(&self) -> bool {
        matches!(self.duration_frames, Some(d) if d > 0)
    }
}

/// A schedule of branding assets and the currently active logo.
#[derive(Debug, Clone, Default)]
pub struct BrandingSchedule {
    /// All registered branding assets
    pub assets: Vec<BrandingAsset>,
    /// ID of the asset currently used as the active logo
    pub active_logo_id: Option<u32>,
}

impl BrandingSchedule {
    /// Create a new empty branding schedule.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new asset.
    pub fn add_asset(&mut self, asset: BrandingAsset) {
        self.assets.push(asset);
    }

    /// Set the active logo by asset ID.
    ///
    /// The ID does not have to correspond to a registered asset (allows
    /// pre-selecting a logo before the asset is registered).
    pub fn set_active_logo(&mut self, id: u32) {
        self.active_logo_id = Some(id);
    }

    /// Return a reference to the currently active logo asset, if any.
    pub fn active_logo(&self) -> Option<&BrandingAsset> {
        let target = self.active_logo_id?;
        self.assets.iter().find(|a| a.id == target)
    }

    /// Return references to all assets of a given element type.
    pub fn assets_of_type(&self, t: &BrandingElement) -> Vec<&BrandingAsset> {
        self.assets.iter().filter(|a| &a.element == t).collect()
    }

    /// Return the total number of registered assets.
    pub fn asset_count(&self) -> usize {
        self.assets.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logo(id: u32) -> BrandingAsset {
        BrandingAsset::new(
            id,
            format!("Logo {id}"),
            BrandingElement::ChannelLogo,
            "/assets/logo.png",
            None,
        )
    }

    fn animated_endboard(id: u32, frames: u32) -> BrandingAsset {
        BrandingAsset::new(
            id,
            "Endboard",
            BrandingElement::Endboard,
            "/assets/endboard.mov",
            Some(frames),
        )
    }

    // --- BrandingElement tests ---

    #[test]
    fn test_channel_logo_is_persistent() {
        assert!(BrandingElement::ChannelLogo.is_persistent());
    }

    #[test]
    fn test_watermark_is_persistent() {
        assert!(BrandingElement::Watermark.is_persistent());
    }

    #[test]
    fn test_bug_is_persistent() {
        assert!(BrandingElement::BugBug.is_persistent());
    }

    #[test]
    fn test_endboard_is_not_persistent() {
        assert!(!BrandingElement::Endboard.is_persistent());
    }

    #[test]
    fn test_transition_is_not_persistent() {
        assert!(!BrandingElement::Transition.is_persistent());
    }

    // --- BrandingAsset tests ---

    #[test]
    fn test_asset_not_animated_when_no_duration() {
        let asset = logo(1);
        assert!(!asset.is_animated());
    }

    #[test]
    fn test_asset_is_animated_with_duration() {
        let asset = animated_endboard(2, 50);
        assert!(asset.is_animated());
    }

    #[test]
    fn test_asset_has_fixed_duration_nonzero() {
        let asset = animated_endboard(3, 25);
        assert!(asset.has_fixed_duration());
    }

    #[test]
    fn test_asset_no_fixed_duration_when_none() {
        let asset = logo(4);
        assert!(!asset.has_fixed_duration());
    }

    #[test]
    fn test_asset_no_fixed_duration_when_zero() {
        let asset = BrandingAsset::new(5, "X", BrandingElement::Endboard, "/x", Some(0));
        assert!(!asset.has_fixed_duration());
    }

    // --- BrandingSchedule tests ---

    #[test]
    fn test_schedule_empty_asset_count() {
        let sched = BrandingSchedule::new();
        assert_eq!(sched.asset_count(), 0);
    }

    #[test]
    fn test_schedule_add_asset_increases_count() {
        let mut sched = BrandingSchedule::new();
        sched.add_asset(logo(1));
        sched.add_asset(logo(2));
        assert_eq!(sched.asset_count(), 2);
    }

    #[test]
    fn test_active_logo_none_initially() {
        let sched = BrandingSchedule::new();
        assert!(sched.active_logo().is_none());
    }

    #[test]
    fn test_set_and_get_active_logo() {
        let mut sched = BrandingSchedule::new();
        sched.add_asset(logo(7));
        sched.set_active_logo(7);
        let active = sched.active_logo();
        assert!(active.is_some());
        assert_eq!(active.expect("should succeed in test").id, 7);
    }

    #[test]
    fn test_active_logo_missing_asset_returns_none() {
        let mut sched = BrandingSchedule::new();
        sched.set_active_logo(99); // ID 99 not registered
        assert!(sched.active_logo().is_none());
    }

    #[test]
    fn test_assets_of_type_filtered() {
        let mut sched = BrandingSchedule::new();
        sched.add_asset(logo(1));
        sched.add_asset(logo(2));
        sched.add_asset(animated_endboard(3, 50));
        let logos = sched.assets_of_type(&BrandingElement::ChannelLogo);
        assert_eq!(logos.len(), 2);
        let endboards = sched.assets_of_type(&BrandingElement::Endboard);
        assert_eq!(endboards.len(), 1);
    }
}
