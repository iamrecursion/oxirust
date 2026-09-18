//! LED volume and wall configuration for virtual production.
//!
//! Models LED panel specifications, wall segments, and full LED volumes used
//! in in-camera VFX (ICVFX) productions.  Also provides a simple moiré risk
//! assessor that estimates interference risk between LED pixel pitch and camera
//! sensor characteristics.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

// ---------------------------------------------------------------------------
// LedPanelSpec
// ---------------------------------------------------------------------------

/// Physical and electrical specification of a single LED panel tile.
#[derive(Debug, Clone, PartialEq)]
pub struct LedPanelSpec {
    /// Horizontal resolution of the panel in pixels.
    pub width_px: u32,
    /// Vertical resolution of the panel in pixels.
    pub height_px: u32,
    /// Pixel pitch (centre-to-centre distance) in millimetres.
    pub pitch_mm: f32,
    /// Peak brightness in nits (cd/m²).
    pub nits_max: u32,
    /// Panel refresh rate in Hz.
    pub refresh_hz: f32,
}

impl LedPanelSpec {
    /// Create a new LED panel specification.
    #[must_use]
    pub fn new(
        width_px: u32,
        height_px: u32,
        pitch_mm: f32,
        nits_max: u32,
        refresh_hz: f32,
    ) -> Self {
        Self {
            width_px,
            height_px,
            pitch_mm,
            nits_max,
            refresh_hz,
        }
    }

    /// Physical width of the panel in millimetres.
    ///
    /// `width_mm = width_px * pitch_mm`
    #[must_use]
    pub fn physical_width_mm(&self) -> f32 {
        self.width_px as f32 * self.pitch_mm
    }

    /// Physical height of the panel in millimetres.
    ///
    /// `height_mm = height_px * pitch_mm`
    #[must_use]
    pub fn physical_height_mm(&self) -> f32 {
        self.height_px as f32 * self.pitch_mm
    }

    /// Total number of pixels in this panel.
    #[must_use]
    pub fn pixel_count(&self) -> u64 {
        u64::from(self.width_px) * u64::from(self.height_px)
    }
}

// ---------------------------------------------------------------------------
// LedWallSegment
// ---------------------------------------------------------------------------

/// A rectangular array of [`LedPanelSpec`] tiles forming a wall segment.
#[derive(Debug, Clone, PartialEq)]
pub struct LedWallSegment {
    /// Unique identifier for this segment.
    pub id: u32,
    /// Number of panels arranged horizontally.
    pub panels_wide: u32,
    /// Number of panels arranged vertically.
    pub panels_high: u32,
    /// Specification of each individual panel tile.
    pub spec: LedPanelSpec,
    /// Whether the segment is curved.
    pub curved: bool,
    /// Curvature angle of the full segment in degrees (only meaningful when `curved` is true).
    pub curvature_deg: f32,
}

impl LedWallSegment {
    /// Create a flat (non-curved) wall segment.
    #[must_use]
    pub fn flat(id: u32, panels_wide: u32, panels_high: u32, spec: LedPanelSpec) -> Self {
        Self {
            id,
            panels_wide,
            panels_high,
            spec,
            curved: false,
            curvature_deg: 0.0,
        }
    }

    /// Create a curved wall segment.
    #[must_use]
    pub fn curved(
        id: u32,
        panels_wide: u32,
        panels_high: u32,
        spec: LedPanelSpec,
        curvature_deg: f32,
    ) -> Self {
        Self {
            id,
            panels_wide,
            panels_high,
            spec,
            curved: true,
            curvature_deg,
        }
    }

    /// Total physical width of the segment in millimetres.
    #[must_use]
    pub fn total_width_mm(&self) -> f32 {
        self.panels_wide as f32 * self.spec.physical_width_mm()
    }

    /// Total physical height of the segment in millimetres.
    #[must_use]
    pub fn total_height_mm(&self) -> f32 {
        self.panels_high as f32 * self.spec.physical_height_mm()
    }

    /// Total number of pixels in this segment.
    #[must_use]
    pub fn total_pixels(&self) -> u64 {
        u64::from(self.panels_wide) * u64::from(self.panels_high) * self.spec.pixel_count()
    }
}

// ---------------------------------------------------------------------------
// LedVolume
// ---------------------------------------------------------------------------

/// A complete LED volume composed of wall segments plus optional ceiling/floor.
#[derive(Debug, Clone)]
pub struct LedVolume {
    /// Side-wall segments (in display order).
    pub segments: Vec<LedWallSegment>,
    /// Optional ceiling LED panel.
    pub ceiling: Option<LedWallSegment>,
    /// Optional floor LED panel.
    pub floor: Option<LedWallSegment>,
}

impl LedVolume {
    /// Create an empty LED volume.
    #[must_use]
    pub fn new() -> Self {
        Self {
            segments: Vec::new(),
            ceiling: None,
            floor: None,
        }
    }

    /// Add a wall segment to the volume.
    pub fn add_segment(&mut self, seg: LedWallSegment) {
        self.segments.push(seg);
    }

    /// Maximum nits across all segments (and ceiling/floor if present).
    ///
    /// Returns 0 when the volume is empty.
    #[must_use]
    pub fn total_nits_max(&self) -> u32 {
        let mut max_nits = 0u32;
        for seg in &self.segments {
            if seg.spec.nits_max > max_nits {
                max_nits = seg.spec.nits_max;
            }
        }
        if let Some(c) = &self.ceiling {
            if c.spec.nits_max > max_nits {
                max_nits = c.spec.nits_max;
            }
        }
        if let Some(f) = &self.floor {
            if f.spec.nits_max > max_nits {
                max_nits = f.spec.nits_max;
            }
        }
        max_nits
    }

    /// Total pixel count across all segments, ceiling, and floor.
    #[must_use]
    pub fn total_pixels(&self) -> u64 {
        let mut total: u64 = self.segments.iter().map(LedWallSegment::total_pixels).sum();
        if let Some(c) = &self.ceiling {
            total += c.total_pixels();
        }
        if let Some(f) = &self.floor {
            total += f.total_pixels();
        }
        total
    }

    /// Whether a ceiling panel is installed.
    #[must_use]
    pub fn has_ceiling(&self) -> bool {
        self.ceiling.is_some()
    }
}

impl Default for LedVolume {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// MoireRiskAssessor
// ---------------------------------------------------------------------------

/// Estimates moiré interference risk when shooting an LED wall with a camera.
///
/// Returns a risk score in `[0.0, 1.0]` where 1.0 indicates maximum risk.
pub struct MoireRiskAssessor;

impl MoireRiskAssessor {
    /// Assess moiré risk for a given panel, shooting distance, and lens.
    ///
    /// The heuristic computes the apparent pixel pitch in millimetres as seen
    /// by the sensor, then compares it to a typical sensor pixel pitch.
    ///
    /// # Arguments
    /// * `panel` – LED panel specification.
    /// * `camera_distance_m` – Distance from camera to LED wall in metres.
    /// * `lens_focal_mm` – Effective focal length in millimetres.
    ///
    /// Returns a score in `[0.0, 1.0]`.
    #[must_use]
    pub fn assess(panel: &LedPanelSpec, camera_distance_m: f64, lens_focal_mm: f32) -> f32 {
        if camera_distance_m <= 0.0 || lens_focal_mm <= 0.0 {
            return 1.0; // degenerate case → maximum risk flag
        }
        // Apparent size of one LED pixel on the sensor (mm).
        let pitch_m = f64::from(panel.pitch_mm) / 1_000.0;
        let apparent_mm = (pitch_m / camera_distance_m) * f64::from(lens_focal_mm);

        // Typical full-frame sensor pixel pitch ≈ 0.006 mm.
        let sensor_pixel_pitch_mm = 0.006_f64;

        // Risk is high when apparent_mm approaches an integer multiple of sensor pitch.
        let ratio = apparent_mm / sensor_pixel_pitch_mm;
        let frac = ratio - ratio.floor();
        // Map fractional distance to 0.5 (worst = integer multiple) → risk in [0,1].
        let risk = 1.0 - (2.0 * (frac - 0.5).abs());
        risk.clamp(0.0, 1.0) as f32
    }
}

// ---------------------------------------------------------------------------
// LedPanel (new-style, richer metadata)
// ---------------------------------------------------------------------------

/// Color gamut of an LED panel.
#[derive(Debug, Clone, PartialEq)]
pub enum PanelGamut {
    Rec709,
    DciP3,
    Rec2020,
}

/// Face of the LED volume where a panel is mounted.
#[derive(Debug, Clone, PartialEq)]
pub enum WallFace {
    Front,
    Left,
    Right,
    Ceiling,
    Floor,
}

/// Physical position and orientation of an LED panel inside the volume.
#[derive(Debug, Clone, PartialEq)]
pub struct PanelPosition {
    /// Horizontal offset from centre in mm.
    pub x_mm: f32,
    /// Vertical offset from centre in mm.
    pub y_mm: f32,
    /// Depth offset in mm (positive = further from camera).
    pub z_mm: f32,
    /// Panel rotation angle in degrees.
    pub rotation_deg: f32,
    /// Which face of the volume this panel belongs to.
    pub face: WallFace,
}

/// A single LED panel tile with rich physical and electrical metadata.
#[derive(Debug, Clone)]
pub struct LedPanel {
    /// Unique string identifier for the panel.
    pub id: String,
    /// Horizontal resolution in pixels.
    pub width_pixels: u32,
    /// Vertical resolution in pixels.
    pub height_pixels: u32,
    /// Centre-to-centre pixel pitch in millimetres.
    pub pixel_pitch_mm: f32,
    /// Peak brightness in nits (cd/m²).
    pub nits_peak: f32,
    /// Scan / refresh rate in Hz.
    pub refresh_rate_hz: f32,
    /// Color gamut of the panel.
    pub color_gamut: PanelGamut,
    /// Physical position within the LED volume.
    pub position: PanelPosition,
}

impl LedPanel {
    /// Create a new panel with default position (front face, zero offset) and
    /// Rec709 gamut at 60 Hz.
    #[must_use]
    pub fn new(id: &str, width: u32, height: u32, pitch_mm: f32, nits: f32) -> Self {
        Self {
            id: id.to_owned(),
            width_pixels: width,
            height_pixels: height,
            pixel_pitch_mm: pitch_mm,
            nits_peak: nits,
            refresh_rate_hz: 60.0,
            color_gamut: PanelGamut::Rec709,
            position: PanelPosition {
                x_mm: 0.0,
                y_mm: 0.0,
                z_mm: 0.0,
                rotation_deg: 0.0,
                face: WallFace::Front,
            },
        }
    }

    /// Physical width of the panel in millimetres (`width_pixels × pixel_pitch_mm`).
    #[must_use]
    pub fn physical_width_mm(&self) -> f32 {
        self.width_pixels as f32 * self.pixel_pitch_mm
    }

    /// Physical height of the panel in millimetres (`height_pixels × pixel_pitch_mm`).
    #[must_use]
    pub fn physical_height_mm(&self) -> f32 {
        self.height_pixels as f32 * self.pixel_pitch_mm
    }

    /// Resolution in megapixels.
    #[must_use]
    pub fn resolution_mpx(&self) -> f32 {
        (self.width_pixels as f32 * self.height_pixels as f32) / 1_000_000.0
    }
}

// ---------------------------------------------------------------------------
// LedVolumeV2
// ---------------------------------------------------------------------------

/// Color processing mode for an LED volume.
#[derive(Debug, Clone, PartialEq)]
pub enum ColorProcessingMode {
    /// No gamma – suitable for in-camera capture.
    Linear,
    /// Standard monitor gamma.
    DisplayGamma,
    /// SMPTE ST 2084 HDR (PQ) curve.
    PqHdr,
}

/// A complete LED volume composed of [`LedPanel`] tiles.
///
/// This is a richer replacement for the older [`LedVolume`] type; both coexist
/// in this module for backward compatibility.
#[derive(Debug, Clone)]
pub struct LedVolumeV2 {
    /// Unique identifier.
    pub id: String,
    /// Human-readable display name.
    pub name: String,
    /// Constituent panels.
    pub panels: Vec<LedPanel>,
    /// Computed total horizontal pixel span (call [`LedVolumeV2::compute_total_resolution`]).
    pub total_width_pixels: u32,
    /// Computed total vertical pixel span.
    pub total_height_pixels: u32,
    /// Driving frame rate in Hz.
    pub driving_fps: f32,
    /// Color processing mode applied to content driven to the volume.
    pub color_processing: ColorProcessingMode,
}

impl LedVolumeV2 {
    /// Create an empty LED volume.
    #[must_use]
    pub fn new(id: &str, name: &str) -> Self {
        Self {
            id: id.to_owned(),
            name: name.to_owned(),
            panels: Vec::new(),
            total_width_pixels: 0,
            total_height_pixels: 0,
            driving_fps: 24.0,
            color_processing: ColorProcessingMode::Linear,
        }
    }

    /// Append a panel to the volume.
    pub fn add_panel(&mut self, panel: LedPanel) {
        self.panels.push(panel);
    }

    /// Remove the panel with the given `id`.  Returns `true` if a panel was removed.
    pub fn remove_panel(&mut self, id: &str) -> bool {
        let before = self.panels.len();
        self.panels.retain(|p| p.id != id);
        self.panels.len() < before
    }

    /// Recompute `total_width_pixels` and `total_height_pixels` from the panel list.
    ///
    /// Uses the max horizontal extent (sum of widths of front-face panels) and the
    /// maximum height found on any single panel as a practical approximation.
    pub fn compute_total_resolution(&mut self) {
        let front_width: u32 = self
            .panels
            .iter()
            .filter(|p| p.position.face == WallFace::Front)
            .map(|p| p.width_pixels)
            .sum();
        let max_height: u32 = self
            .panels
            .iter()
            .map(|p| p.height_pixels)
            .max()
            .unwrap_or(0);
        self.total_width_pixels = front_width;
        self.total_height_pixels = max_height;
    }

    /// Return references to all panels on the given face.
    #[must_use]
    pub fn panels_by_face(&self, face: &WallFace) -> Vec<&LedPanel> {
        self.panels
            .iter()
            .filter(|p| &p.position.face == face)
            .collect()
    }

    /// Minimum peak brightness across all panels (weakest-link metric).
    ///
    /// Returns `0.0` for an empty volume.
    #[must_use]
    pub fn peak_nits(&self) -> f32 {
        self.panels
            .iter()
            .map(|p| p.nits_peak)
            .reduce(f32::min)
            .unwrap_or(0.0)
    }

    /// Returns `true` when at least one panel exceeds 1000 nits **and** the
    /// color processing mode is [`ColorProcessingMode::PqHdr`].
    #[must_use]
    pub fn requires_hdr(&self) -> bool {
        self.color_processing == ColorProcessingMode::PqHdr
            && self.panels.iter().any(|p| p.nits_peak > 1000.0)
    }

    /// Validate the volume configuration.
    ///
    /// Returns a list of human-readable error strings (empty = valid).
    #[must_use]
    pub fn validate(&self) -> Vec<String> {
        let mut errors: Vec<String> = Vec::new();

        if self.panels.is_empty() {
            errors.push("LED volume has no panels".to_owned());
            return errors;
        }

        // Check refresh rate consistency.
        let first_hz = self.panels[0].refresh_rate_hz;
        let mismatched_refresh = self
            .panels
            .iter()
            .any(|p| (p.refresh_rate_hz - first_hz).abs() > 0.1);
        if mismatched_refresh {
            errors.push("Panels have mismatched refresh rates".to_owned());
        }

        // Check pixel pitch consistency.
        let first_pitch = self.panels[0].pixel_pitch_mm;
        let mismatched_pitch = self
            .panels
            .iter()
            .any(|p| (p.pixel_pitch_mm - first_pitch).abs() > 0.01);
        if mismatched_pitch {
            errors.push("Panels have inconsistent pixel pitch values".to_owned());
        }

        errors
    }
}

// ---------------------------------------------------------------------------
// MoireChecker (new-style, per CameraTransform context)
// ---------------------------------------------------------------------------

/// Estimates moiré risk between an LED panel and a camera sensor given a
/// shooting distance.
pub struct MoireChecker {
    /// Camera sensor resolution (width × height) in pixels.
    pub camera_sensor_pixels: (u32, u32),
    /// Lens focal length in millimetres.
    pub lens_focal_length_mm: f32,
}

impl MoireChecker {
    /// Compute a moiré risk score in `[0.0, 1.0]` for the given panel at
    /// `camera_distance_m` metres.
    ///
    /// Apparent pixel density at the sensor is estimated as:
    /// `panel_ppi × focal_length / (distance_mm − focal_length)`.
    ///
    /// Risk = `|sensor_ppi − apparent_ppi| / sensor_ppi`, clamped to `[0, 1]`.
    /// A score near 0 indicates similar densities (high interference risk); a
    /// score near 1 indicates very different densities (low risk).
    ///
    /// Returns `1.0` (maximum risk flag) for degenerate inputs.
    #[must_use]
    pub fn risk_score(&self, panel: &LedPanel, camera_distance_m: f32) -> f32 {
        if camera_distance_m <= 0.0
            || self.lens_focal_length_mm <= 0.0
            || panel.pixel_pitch_mm <= 0.0
        {
            return 1.0;
        }

        // Sensor PPI: pixels per mm on sensor (assuming square sensor of width
        // equal to sensor_pixels.0 at a canonical 36 mm full-frame width).
        let sensor_width_mm = 36.0_f32;
        let sensor_ppi = self.camera_sensor_pixels.0 as f32 / sensor_width_mm;

        // Panel pixels per mm in world space.
        let panel_ppi = 1.0_f32 / panel.pixel_pitch_mm;

        let distance_mm = camera_distance_m * 1_000.0_f32;
        let focal = self.lens_focal_length_mm;

        // Guard against division by zero when focal length ≥ distance.
        let denom = distance_mm - focal;
        if denom.abs() < 1e-6 {
            return 1.0;
        }

        // Apparent pixel density projected onto the sensor plane.
        let apparent_ppi = panel_ppi * (focal / denom).abs();

        if sensor_ppi <= 0.0 {
            return 1.0;
        }

        // Risk: similarity between sensor density and apparent LED density.
        let diff = (sensor_ppi - apparent_ppi).abs();
        (diff / sensor_ppi).clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_panel() -> LedPanelSpec {
        LedPanelSpec::new(256, 128, 2.8, 1500, 3840.0)
    }

    #[test]
    fn test_panel_physical_width() {
        let p = LedPanelSpec::new(256, 128, 2.8, 1500, 3840.0);
        let expected = 256.0_f32 * 2.8_f32;
        assert!((p.physical_width_mm() - expected).abs() < 1e-3);
    }

    #[test]
    fn test_panel_physical_height() {
        let p = LedPanelSpec::new(256, 128, 2.8, 1500, 3840.0);
        let expected = 128.0_f32 * 2.8_f32;
        assert!((p.physical_height_mm() - expected).abs() < 1e-3);
    }

    #[test]
    fn test_panel_pixel_count() {
        let p = LedPanelSpec::new(256, 128, 2.8, 1500, 3840.0);
        assert_eq!(p.pixel_count(), 256 * 128);
    }

    #[test]
    fn test_segment_flat_creation() {
        let seg = LedWallSegment::flat(1, 4, 2, sample_panel());
        assert!(!seg.curved);
        assert_eq!(seg.curvature_deg, 0.0);
    }

    #[test]
    fn test_segment_curved_creation() {
        let seg = LedWallSegment::curved(2, 6, 3, sample_panel(), 180.0);
        assert!(seg.curved);
        assert!((seg.curvature_deg - 180.0).abs() < 1e-5);
    }

    #[test]
    fn test_segment_total_width() {
        let spec = LedPanelSpec::new(256, 128, 2.8, 1500, 3840.0);
        let seg = LedWallSegment::flat(1, 4, 2, spec);
        // 4 panels × 256 px × 2.8 mm/px
        let expected = 4.0_f32 * 256.0_f32 * 2.8_f32;
        assert!((seg.total_width_mm() - expected).abs() < 1e-2);
    }

    #[test]
    fn test_segment_total_height() {
        let spec = LedPanelSpec::new(256, 128, 2.8, 1500, 3840.0);
        let seg = LedWallSegment::flat(1, 4, 2, spec);
        let expected = 2.0_f32 * 128.0_f32 * 2.8_f32;
        assert!((seg.total_height_mm() - expected).abs() < 1e-2);
    }

    #[test]
    fn test_segment_total_pixels() {
        let spec = LedPanelSpec::new(256, 128, 2.8, 1500, 3840.0);
        let seg = LedWallSegment::flat(1, 4, 2, spec);
        assert_eq!(seg.total_pixels(), 4 * 2 * 256 * 128);
    }

    #[test]
    fn test_volume_add_segment() {
        let mut vol = LedVolume::new();
        vol.add_segment(LedWallSegment::flat(1, 4, 2, sample_panel()));
        assert_eq!(vol.segments.len(), 1);
    }

    #[test]
    fn test_volume_total_nits_max_empty() {
        let vol = LedVolume::new();
        assert_eq!(vol.total_nits_max(), 0);
    }

    #[test]
    fn test_volume_total_nits_max_segments() {
        let mut vol = LedVolume::new();
        let spec_hi = LedPanelSpec::new(256, 128, 2.8, 2000, 3840.0);
        vol.add_segment(LedWallSegment::flat(1, 4, 2, sample_panel())); // 1500 nits
        vol.add_segment(LedWallSegment::flat(2, 4, 2, spec_hi)); // 2000 nits
        assert_eq!(vol.total_nits_max(), 2000);
    }

    #[test]
    fn test_volume_has_ceiling_false() {
        let vol = LedVolume::new();
        assert!(!vol.has_ceiling());
    }

    #[test]
    fn test_volume_has_ceiling_true() {
        let mut vol = LedVolume::new();
        vol.ceiling = Some(LedWallSegment::flat(99, 4, 2, sample_panel()));
        assert!(vol.has_ceiling());
    }

    #[test]
    fn test_volume_total_pixels_with_ceiling() {
        let mut vol = LedVolume::new();
        let seg = LedWallSegment::flat(1, 1, 1, LedPanelSpec::new(10, 10, 1.0, 1000, 60.0));
        let ceil = LedWallSegment::flat(2, 1, 1, LedPanelSpec::new(10, 10, 1.0, 1000, 60.0));
        vol.add_segment(seg);
        vol.ceiling = Some(ceil);
        // 10*10 + 10*10 = 200
        assert_eq!(vol.total_pixels(), 200);
    }

    #[test]
    fn test_moire_risk_assessor_range() {
        let panel = sample_panel();
        let risk = MoireRiskAssessor::assess(&panel, 5.0, 50.0);
        assert!((0.0..=1.0).contains(&risk));
    }

    #[test]
    fn test_moire_risk_assessor_degenerate() {
        let panel = sample_panel();
        let risk = MoireRiskAssessor::assess(&panel, 0.0, 50.0);
        assert_eq!(risk, 1.0);
    }

    // -----------------------------------------------------------------------
    // New-style LedPanel tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_led_panel_new_defaults() {
        let p = LedPanel::new("P1", 320, 160, 2.5, 1800.0);
        assert_eq!(p.id, "P1");
        assert_eq!(p.width_pixels, 320);
        assert_eq!(p.height_pixels, 160);
        assert!((p.pixel_pitch_mm - 2.5).abs() < 1e-5);
        assert!((p.nits_peak - 1800.0).abs() < 1e-5);
        assert_eq!(p.color_gamut, PanelGamut::Rec709);
        assert_eq!(p.position.face, WallFace::Front);
    }

    #[test]
    fn test_led_panel_physical_width() {
        let p = LedPanel::new("P2", 256, 128, 2.8, 1500.0);
        let expected = 256.0_f32 * 2.8_f32;
        assert!((p.physical_width_mm() - expected).abs() < 1e-3);
    }

    #[test]
    fn test_led_panel_physical_height() {
        let p = LedPanel::new("P3", 256, 128, 2.8, 1500.0);
        let expected = 128.0_f32 * 2.8_f32;
        assert!((p.physical_height_mm() - expected).abs() < 1e-3);
    }

    #[test]
    fn test_led_panel_resolution_mpx() {
        let p = LedPanel::new("P4", 1000, 1000, 1.0, 1000.0);
        assert!((p.resolution_mpx() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_led_volume_v2_add_remove_panel() {
        let mut vol = LedVolumeV2::new("V1", "Stage A");
        let p = LedPanel::new("PA", 256, 128, 2.5, 1200.0);
        vol.add_panel(p);
        assert_eq!(vol.panels.len(), 1);
        let removed = vol.remove_panel("PA");
        assert!(removed);
        assert!(vol.panels.is_empty());
    }

    #[test]
    fn test_led_volume_v2_remove_nonexistent() {
        let mut vol = LedVolumeV2::new("V2", "Stage B");
        let removed = vol.remove_panel("ghost");
        assert!(!removed);
    }

    #[test]
    fn test_led_volume_v2_compute_total_resolution() {
        let mut vol = LedVolumeV2::new("V3", "Stage C");
        let mut p1 = LedPanel::new("F1", 256, 128, 2.5, 1200.0);
        p1.position.face = WallFace::Front;
        let mut p2 = LedPanel::new("F2", 256, 256, 2.5, 1200.0);
        p2.position.face = WallFace::Front;
        vol.add_panel(p1);
        vol.add_panel(p2);
        vol.compute_total_resolution();
        // Width = 256 + 256 = 512 (front panels)
        assert_eq!(vol.total_width_pixels, 512);
        // Height = max(128, 256) = 256
        assert_eq!(vol.total_height_pixels, 256);
    }

    #[test]
    fn test_led_volume_v2_panels_by_face() {
        let mut vol = LedVolumeV2::new("V4", "Stage D");
        let mut lp = LedPanel::new("L1", 128, 128, 2.5, 1000.0);
        lp.position.face = WallFace::Left;
        vol.add_panel(LedPanel::new("F1", 256, 128, 2.5, 1000.0));
        vol.add_panel(lp);
        let left = vol.panels_by_face(&WallFace::Left);
        assert_eq!(left.len(), 1);
        assert_eq!(left[0].id, "L1");
    }

    #[test]
    fn test_led_volume_v2_peak_nits_weakest_link() {
        let mut vol = LedVolumeV2::new("V5", "Stage E");
        vol.add_panel(LedPanel::new("A", 100, 100, 2.0, 2000.0));
        vol.add_panel(LedPanel::new("B", 100, 100, 2.0, 800.0));
        // Weakest link = 800
        assert!((vol.peak_nits() - 800.0).abs() < 1e-5);
    }

    #[test]
    fn test_led_volume_v2_requires_hdr() {
        let mut vol = LedVolumeV2::new("V6", "Stage F");
        vol.color_processing = ColorProcessingMode::PqHdr;
        vol.add_panel(LedPanel::new("H1", 100, 100, 2.0, 1500.0));
        assert!(vol.requires_hdr());

        vol.color_processing = ColorProcessingMode::Linear;
        assert!(!vol.requires_hdr());
    }

    #[test]
    fn test_led_volume_v2_validate_empty() {
        let vol = LedVolumeV2::new("V7", "Stage G");
        let errors = vol.validate();
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_led_volume_v2_validate_mismatched_refresh() {
        let mut vol = LedVolumeV2::new("V8", "Stage H");
        let mut p1 = LedPanel::new("R1", 128, 128, 2.5, 1000.0);
        p1.refresh_rate_hz = 60.0;
        let mut p2 = LedPanel::new("R2", 128, 128, 2.5, 1000.0);
        p2.refresh_rate_hz = 120.0;
        vol.add_panel(p1);
        vol.add_panel(p2);
        let errors = vol.validate();
        assert!(errors.iter().any(|e| e.contains("refresh")));
    }

    #[test]
    fn test_moire_checker_risk_score_range() {
        let checker = MoireChecker {
            camera_sensor_pixels: (6000, 4000),
            lens_focal_length_mm: 50.0,
        };
        let panel = LedPanel::new("MC1", 256, 128, 2.8, 1500.0);
        let score = checker.risk_score(&panel, 5.0);
        assert!((0.0..=1.0).contains(&score));
    }

    #[test]
    fn test_moire_checker_degenerate_distance() {
        let checker = MoireChecker {
            camera_sensor_pixels: (6000, 4000),
            lens_focal_length_mm: 50.0,
        };
        let panel = LedPanel::new("MC2", 256, 128, 2.8, 1500.0);
        assert_eq!(checker.risk_score(&panel, 0.0), 1.0);
    }
}
