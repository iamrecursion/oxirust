//! VR headset metadata and projection configuration helpers.
//!
//! Provides per-headset hardware specifications (FOV, resolution, refresh rate,
//! IPD) and derives optimal [`ProjectionConfig`] values so that
//! equirectangular-to-viewport renders are pre-tuned for each device.

#![allow(dead_code)]

use crate::viewport::ViewportParams;

// ─── HeadsetType ─────────────────────────────────────────────────────────────

/// VR headset model family.
///
/// Used to look up hardware-accurate display specifications and to derive
/// an optimised projection configuration for equirectangular content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HeadsetType {
    /// Meta / Oculus Quest 2 / Quest 3 class hardware.
    OculusQuest,
    /// Valve Index.
    ValveIndex,
    /// HTC Vive / Vive Pro class hardware.
    HTCVive,
    /// Pico Neo 3 / Pico 4 class hardware.
    PicoNeo,
    /// Catch-all for unspecified or custom headsets.
    Generic,
}

impl HeadsetType {
    /// Human-readable name for this headset.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::OculusQuest => "Oculus Quest",
            Self::ValveIndex => "Valve Index",
            Self::HTCVive => "HTC Vive",
            Self::PicoNeo => "Pico Neo",
            Self::Generic => "Generic",
        }
    }

    /// Return hardware metadata for this headset type.
    #[must_use]
    pub fn metadata(self) -> HeadsetMetadata {
        HeadsetMetadata::for_headset(self)
    }
}

// ─── HeadsetMetadata ─────────────────────────────────────────────────────────

/// Physical display characteristics of a VR headset.
#[derive(Debug, Clone, PartialEq)]
pub struct HeadsetMetadata {
    /// Headset model family.
    pub headset_type: HeadsetType,
    /// Horizontal field of view in degrees.
    pub fov_h: f32,
    /// Vertical field of view in degrees.
    pub fov_v: f32,
    /// Per-eye display resolution in pixels (width × height).
    pub resolution: (u32, u32),
    /// Native display refresh rate in Hz.
    pub refresh_rate: f32,
    /// Default / recommended inter-pupillary distance in millimetres.
    pub ipd_mm: f32,
}

impl HeadsetMetadata {
    /// Construct a [`HeadsetMetadata`] with explicit values.
    #[must_use]
    pub const fn new(
        headset_type: HeadsetType,
        fov_h: f32,
        fov_v: f32,
        resolution: (u32, u32),
        refresh_rate: f32,
        ipd_mm: f32,
    ) -> Self {
        Self {
            headset_type,
            fov_h,
            fov_v,
            resolution,
            refresh_rate,
            ipd_mm,
        }
    }

    /// Return pre-defined hardware specs for a known headset type.
    ///
    /// Values are sourced from publicly available manufacturer specifications.
    #[must_use]
    pub fn for_headset(headset_type: HeadsetType) -> Self {
        match headset_type {
            HeadsetType::OculusQuest => Self::new(
                HeadsetType::OculusQuest,
                96.0,         // ~96° horizontal FOV (Quest 2)
                96.0,         // ~96° vertical FOV
                (1832, 1920), // per-eye resolution, Quest 2
                120.0,        // max refresh rate Hz
                63.5,         // default IPD mm
            ),
            HeadsetType::ValveIndex => Self::new(
                HeadsetType::ValveIndex,
                108.0,        // ~108° horizontal FOV
                96.0,         // ~96° vertical FOV
                (1440, 1600), // per-eye resolution
                144.0,        // max refresh rate Hz
                63.0,         // default IPD mm
            ),
            HeadsetType::HTCVive => Self::new(
                HeadsetType::HTCVive,
                110.0,        // ~110° horizontal FOV
                113.0,        // ~113° vertical FOV
                (1080, 1200), // per-eye resolution (Vive Pro)
                90.0,         // refresh rate Hz
                63.0,         // default IPD mm
            ),
            HeadsetType::PicoNeo => Self::new(
                HeadsetType::PicoNeo,
                98.0,         // ~98° horizontal FOV (Pico 4)
                98.0,         // ~98° vertical FOV
                (2160, 2160), // per-eye resolution, Pico 4
                90.0,         // refresh rate Hz
                62.0,         // default IPD mm
            ),
            HeadsetType::Generic => Self::new(
                HeadsetType::Generic,
                90.0, // conservative default
                90.0,
                (1920, 1080),
                60.0,
                63.0,
            ),
        }
    }

    /// Horizontal FOV in radians.
    #[must_use]
    pub fn fov_h_rad(&self) -> f32 {
        self.fov_h.to_radians()
    }

    /// Vertical FOV in radians.
    #[must_use]
    pub fn fov_v_rad(&self) -> f32 {
        self.fov_v.to_radians()
    }

    /// Aspect ratio of the per-eye display (width / height).
    #[must_use]
    pub fn aspect_ratio(&self) -> f32 {
        let (w, h) = self.resolution;
        w as f32 / h.max(1) as f32
    }

    /// Return `true` if this headset runs at ≥ 120 Hz.
    #[must_use]
    pub fn is_high_refresh(&self) -> bool {
        self.refresh_rate >= 120.0
    }

    /// Return `true` if this headset's per-eye resolution is 4K or above
    /// (i.e. either dimension ≥ 2048 px).
    #[must_use]
    pub fn is_high_resolution(&self) -> bool {
        let (w, h) = self.resolution;
        w >= 2048 || h >= 2048
    }
}

// ─── ProjectionConfig ────────────────────────────────────────────────────────

/// Optimal projection configuration derived from a headset's hardware specs.
///
/// This wraps [`ViewportParams`] and adds headset-specific tuning hints that
/// callers can apply when rendering equirectangular panoramas.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectionConfig {
    /// Viewport parameters ready to pass to [`crate::viewport::render_viewport`].
    pub viewport: ViewportParams,
    /// Recommended output width for the render (equals per-eye display width).
    pub render_width: u32,
    /// Recommended output height for the render (equals per-eye display height).
    pub render_height: u32,
    /// IPD in millimetres, for stereo eye-offset calculations.
    pub ipd_mm: f32,
    /// Headset refresh rate — callers should target this frame rate.
    pub refresh_rate: f32,
    /// Headset model family this config was derived from.
    pub headset_type: HeadsetType,
}

impl ProjectionConfig {
    /// Build a [`ProjectionConfig`] from raw hardware values.
    #[must_use]
    pub fn new(meta: &HeadsetMetadata) -> Self {
        let (render_width, render_height) = meta.resolution;
        let viewport = ViewportParams::new(render_width, render_height).with_fov_deg(meta.fov_h);

        Self {
            viewport,
            render_width,
            render_height,
            ipd_mm: meta.ipd_mm,
            refresh_rate: meta.refresh_rate,
            headset_type: meta.headset_type,
        }
    }
}

// ─── HeadsetOptimizedConfig ──────────────────────────────────────────────────

/// Factory that returns a [`ProjectionConfig`] tuned for a given headset.
pub struct HeadsetOptimizedConfig;

impl HeadsetOptimizedConfig {
    /// Return an optimised [`ProjectionConfig`] for the specified headset.
    ///
    /// # Example
    ///
    /// ```rust
    /// use oximedia_360::headset_metadata::{HeadsetOptimizedConfig, HeadsetType};
    ///
    /// let cfg = HeadsetOptimizedConfig::for_headset(HeadsetType::ValveIndex);
    /// assert_eq!(cfg.render_width, 1440);
    /// ```
    #[must_use]
    pub fn for_headset(headset_type: HeadsetType) -> ProjectionConfig {
        let meta = HeadsetMetadata::for_headset(headset_type);
        ProjectionConfig::new(&meta)
    }

    /// Return optimised configs for all known headset types.
    #[must_use]
    pub fn all_headsets() -> Vec<(HeadsetType, ProjectionConfig)> {
        let types = [
            HeadsetType::OculusQuest,
            HeadsetType::ValveIndex,
            HeadsetType::HTCVive,
            HeadsetType::PicoNeo,
            HeadsetType::Generic,
        ];
        types
            .iter()
            .map(|&ht| (ht, Self::for_headset(ht)))
            .collect()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    // ── HeadsetType ──────────────────────────────────────────────────────────

    #[test]
    fn headset_type_name_oculus() {
        assert_eq!(HeadsetType::OculusQuest.name(), "Oculus Quest");
    }

    #[test]
    fn headset_type_name_valve_index() {
        assert_eq!(HeadsetType::ValveIndex.name(), "Valve Index");
    }

    #[test]
    fn headset_type_name_htc() {
        assert_eq!(HeadsetType::HTCVive.name(), "HTC Vive");
    }

    #[test]
    fn headset_type_name_pico() {
        assert_eq!(HeadsetType::PicoNeo.name(), "Pico Neo");
    }

    #[test]
    fn headset_type_name_generic() {
        assert_eq!(HeadsetType::Generic.name(), "Generic");
    }

    // ── HeadsetMetadata — construction ───────────────────────────────────────

    #[test]
    fn metadata_for_oculus_quest() {
        let meta = HeadsetMetadata::for_headset(HeadsetType::OculusQuest);
        assert_eq!(meta.headset_type, HeadsetType::OculusQuest);
        assert!(meta.fov_h > 90.0 && meta.fov_h <= 120.0);
        assert!(meta.fov_v > 80.0 && meta.fov_v <= 120.0);
        assert_eq!(meta.resolution, (1832, 1920));
        assert_eq!(meta.refresh_rate, 120.0);
        assert!(meta.ipd_mm > 50.0 && meta.ipd_mm < 80.0);
    }

    #[test]
    fn metadata_for_valve_index() {
        let meta = HeadsetMetadata::for_headset(HeadsetType::ValveIndex);
        assert_eq!(meta.headset_type, HeadsetType::ValveIndex);
        assert_eq!(meta.resolution, (1440, 1600));
        assert_eq!(meta.refresh_rate, 144.0);
    }

    #[test]
    fn metadata_for_htc_vive() {
        let meta = HeadsetMetadata::for_headset(HeadsetType::HTCVive);
        assert_eq!(meta.headset_type, HeadsetType::HTCVive);
        assert_eq!(meta.resolution, (1080, 1200));
        assert_eq!(meta.refresh_rate, 90.0);
    }

    #[test]
    fn metadata_for_pico_neo() {
        let meta = HeadsetMetadata::for_headset(HeadsetType::PicoNeo);
        assert_eq!(meta.headset_type, HeadsetType::PicoNeo);
        assert_eq!(meta.resolution, (2160, 2160));
    }

    #[test]
    fn metadata_for_generic() {
        let meta = HeadsetMetadata::for_headset(HeadsetType::Generic);
        assert_eq!(meta.headset_type, HeadsetType::Generic);
        assert_eq!(meta.resolution, (1920, 1080));
        assert_eq!(meta.refresh_rate, 60.0);
    }

    // ── HeadsetMetadata — derived properties ─────────────────────────────────

    #[test]
    fn fov_radians_conversion() {
        let meta = HeadsetMetadata::for_headset(HeadsetType::Generic);
        let expected = 90_f32.to_radians();
        assert!((meta.fov_h_rad() - expected).abs() < 1e-5);
        assert!((meta.fov_v_rad() - expected).abs() < 1e-5);
    }

    #[test]
    fn aspect_ratio_generic() {
        let meta = HeadsetMetadata::for_headset(HeadsetType::Generic);
        let ar = meta.aspect_ratio();
        // 1920 / 1080 ≈ 1.777…
        assert!((ar - (1920.0_f32 / 1080.0)).abs() < 1e-4);
    }

    #[test]
    fn is_high_refresh_valve_index() {
        let meta = HeadsetMetadata::for_headset(HeadsetType::ValveIndex);
        assert!(meta.is_high_refresh());
    }

    #[test]
    fn is_not_high_refresh_generic() {
        let meta = HeadsetMetadata::for_headset(HeadsetType::Generic);
        assert!(!meta.is_high_refresh());
    }

    #[test]
    fn is_high_resolution_pico() {
        let meta = HeadsetMetadata::for_headset(HeadsetType::PicoNeo);
        assert!(meta.is_high_resolution());
    }

    #[test]
    fn is_not_high_resolution_htc_vive() {
        let meta = HeadsetMetadata::for_headset(HeadsetType::HTCVive);
        assert!(!meta.is_high_resolution());
    }

    // ── ProjectionConfig ─────────────────────────────────────────────────────

    #[test]
    fn projection_config_render_dims_match_resolution() {
        let meta = HeadsetMetadata::for_headset(HeadsetType::OculusQuest);
        let cfg = ProjectionConfig::new(&meta);
        assert_eq!(cfg.render_width, meta.resolution.0);
        assert_eq!(cfg.render_height, meta.resolution.1);
    }

    #[test]
    fn projection_config_ipd_propagated() {
        let meta = HeadsetMetadata::for_headset(HeadsetType::ValveIndex);
        let cfg = ProjectionConfig::new(&meta);
        assert!((cfg.ipd_mm - meta.ipd_mm).abs() < 1e-6);
    }

    #[test]
    fn projection_config_refresh_rate_propagated() {
        let meta = HeadsetMetadata::for_headset(HeadsetType::ValveIndex);
        let cfg = ProjectionConfig::new(&meta);
        assert_eq!(cfg.refresh_rate, 144.0);
    }

    #[test]
    fn projection_config_fov_stored_in_viewport() {
        let meta = HeadsetMetadata::for_headset(HeadsetType::HTCVive);
        let cfg = ProjectionConfig::new(&meta);
        let expected_rad = meta.fov_h.to_radians();
        assert!((cfg.viewport.hfov_rad - expected_rad).abs() < 1e-4);
    }

    // ── HeadsetOptimizedConfig ───────────────────────────────────────────────

    #[test]
    fn optimized_config_for_all_headsets() {
        let all = HeadsetOptimizedConfig::all_headsets();
        assert_eq!(all.len(), 5);
    }

    #[test]
    fn optimized_config_valve_index_resolution() {
        let cfg = HeadsetOptimizedConfig::for_headset(HeadsetType::ValveIndex);
        assert_eq!(cfg.render_width, 1440);
        assert_eq!(cfg.render_height, 1600);
    }

    #[test]
    fn optimized_config_headset_type_preserved() {
        let cfg = HeadsetOptimizedConfig::for_headset(HeadsetType::PicoNeo);
        assert_eq!(cfg.headset_type, HeadsetType::PicoNeo);
    }

    #[test]
    fn fov_h_within_plausible_range() {
        for (_, cfg) in HeadsetOptimizedConfig::all_headsets() {
            // All headsets should have hfov between 45° and 140°
            let fov_deg = cfg.viewport.hfov_rad * 180.0 / PI;
            assert!(
                fov_deg >= 45.0 && fov_deg <= 140.0,
                "FOV {fov_deg}° is out of plausible range for {:?}",
                cfg.headset_type
            );
        }
    }
}
