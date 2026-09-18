//! LED volume configuration and brightness management for virtual production.
//!
//! Models an LED volume as an arrangement of LED wall segments with brightness,
//! colour temperature, and refresh-rate parameters.

#![allow(dead_code)]

/// A segment of an LED volume (one wall, floor, or ceiling section).
#[derive(Debug, Clone)]
pub struct LedSegment {
    /// Segment identifier.
    pub id: u32,
    /// Human-readable label.
    pub name: String,
    /// Physical width of the segment in metres.
    pub width_m: f32,
    /// Physical height of the segment in metres.
    pub height_m: f32,
    /// Peak luminance in nits (cd/m²).
    pub peak_nits: f32,
    /// Colour temperature in Kelvin.
    pub colour_temp_k: u32,
    /// Refresh rate in Hz.
    pub refresh_hz: u32,
    /// Whether this segment is currently powered on.
    pub powered: bool,
}

impl LedSegment {
    /// Physical area of the segment in square metres.
    #[must_use]
    pub fn area_sqm(&self) -> f32 {
        self.width_m * self.height_m
    }

    /// Effective luminous flux output estimate in lumens (peak nits × area × π).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn luminous_flux_lm(&self) -> f32 {
        use std::f32::consts::PI;
        // Simplified Lambertian model: L × A × π
        self.peak_nits * self.area_sqm() * PI
    }

    /// Returns `true` if the refresh rate is high enough to avoid on-camera flicker
    /// for the given camera shutter angle (degrees) and frame rate.
    ///
    /// Rule of thumb: refresh must be ≥ 2× frame rate to guarantee no banding.
    #[must_use]
    pub fn flicker_free(&self, fps: f32) -> bool {
        self.refresh_hz as f32 >= fps * 2.0
    }
}

/// Orientation of an LED segment relative to the stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentOrientation {
    /// Rear wall facing camera.
    RearWall,
    /// Left side wall.
    LeftWall,
    /// Right side wall.
    RightWall,
    /// Ceiling LED panel.
    Ceiling,
    /// Floor LED panel.
    Floor,
}

impl SegmentOrientation {
    /// Whether this surface provides key lighting for actors.
    #[must_use]
    pub fn provides_key_light(&self) -> bool {
        matches!(
            self,
            SegmentOrientation::RearWall | SegmentOrientation::Ceiling
        )
    }
}

/// An LED volume composed of multiple segments.
#[derive(Debug, Clone, Default)]
pub struct LedVolume {
    /// All segments that make up the volume.
    pub segments: Vec<LedSegment>,
    /// Overall brightness scale applied to all segments (0.0–1.0).
    pub master_brightness: f32,
    /// Camera frame rate the volume is calibrated for.
    pub calibrated_fps: f32,
}

impl LedVolume {
    /// Create a new empty LED volume calibrated for a given frame rate.
    #[must_use]
    pub fn new(fps: f32) -> Self {
        Self {
            segments: Vec::new(),
            master_brightness: 1.0,
            calibrated_fps: fps,
        }
    }

    /// Add a segment to the volume. Returns the assigned segment ID.
    pub fn add_segment(&mut self, mut segment: LedSegment) -> u32 {
        let id = self.segments.len() as u32;
        segment.id = id;
        self.segments.push(segment);
        id
    }

    /// Total physical area of all powered segments in square metres.
    #[must_use]
    pub fn total_powered_area_sqm(&self) -> f32 {
        self.segments
            .iter()
            .filter(|s| s.powered)
            .map(LedSegment::area_sqm)
            .sum()
    }

    /// Count of segments that are flicker-free at the calibrated fps.
    #[must_use]
    pub fn flicker_free_count(&self) -> usize {
        self.segments
            .iter()
            .filter(|s| s.flicker_free(self.calibrated_fps))
            .count()
    }

    /// Total estimated luminous flux from all powered segments in lumens.
    #[must_use]
    pub fn total_flux_lm(&self) -> f32 {
        self.segments
            .iter()
            .filter(|s| s.powered)
            .map(|s| s.luminous_flux_lm() * self.master_brightness)
            .sum()
    }

    /// Set master brightness, clamped to [0.0, 1.0].
    pub fn set_master_brightness(&mut self, brightness: f32) {
        self.master_brightness = brightness.clamp(0.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_segment(powered: bool, refresh_hz: u32) -> LedSegment {
        LedSegment {
            id: 0,
            name: "test".to_string(),
            width_m: 4.0,
            height_m: 2.0,
            peak_nits: 1000.0,
            colour_temp_k: 5600,
            refresh_hz,
            powered,
        }
    }

    #[test]
    fn test_segment_area() {
        let seg = sample_segment(true, 3840);
        assert!((seg.area_sqm() - 8.0).abs() < 1e-5);
    }

    #[test]
    fn test_segment_luminous_flux() {
        let seg = sample_segment(true, 3840);
        // 1000 nits × 8 m² × π ≈ 25132
        let expected = 1000.0 * 8.0 * std::f32::consts::PI;
        assert!((seg.luminous_flux_lm() - expected).abs() < 1.0);
    }

    #[test]
    fn test_flicker_free_at_fps() {
        let seg = sample_segment(true, 3840);
        assert!(seg.flicker_free(60.0));
        assert!(seg.flicker_free(120.0));
    }

    #[test]
    fn test_not_flicker_free() {
        let seg = sample_segment(true, 50);
        assert!(!seg.flicker_free(60.0));
    }

    #[test]
    fn test_orientation_key_light_rear() {
        assert!(SegmentOrientation::RearWall.provides_key_light());
    }

    #[test]
    fn test_orientation_key_light_ceiling() {
        assert!(SegmentOrientation::Ceiling.provides_key_light());
    }

    #[test]
    fn test_orientation_no_key_light_floor() {
        assert!(!SegmentOrientation::Floor.provides_key_light());
    }

    #[test]
    fn test_orientation_no_key_light_side() {
        assert!(!SegmentOrientation::LeftWall.provides_key_light());
        assert!(!SegmentOrientation::RightWall.provides_key_light());
    }

    #[test]
    fn test_led_volume_add_segment() {
        let mut vol = LedVolume::new(60.0);
        let id = vol.add_segment(sample_segment(true, 3840));
        assert_eq!(id, 0);
        assert_eq!(vol.segments.len(), 1);
    }

    #[test]
    fn test_led_volume_powered_area() {
        let mut vol = LedVolume::new(60.0);
        vol.add_segment(sample_segment(true, 3840));
        vol.add_segment(sample_segment(false, 3840));
        // only the powered one (8 m²)
        assert!((vol.total_powered_area_sqm() - 8.0).abs() < 1e-5);
    }

    #[test]
    fn test_led_volume_flicker_free_count() {
        let mut vol = LedVolume::new(60.0);
        vol.add_segment(sample_segment(true, 3840));
        vol.add_segment(sample_segment(true, 50)); // not flicker-free at 60 fps
        assert_eq!(vol.flicker_free_count(), 1);
    }

    #[test]
    fn test_master_brightness_clamp() {
        let mut vol = LedVolume::new(60.0);
        vol.set_master_brightness(2.0);
        assert_eq!(vol.master_brightness, 1.0);
        vol.set_master_brightness(-0.5);
        assert_eq!(vol.master_brightness, 0.0);
    }

    #[test]
    fn test_total_flux_scales_with_brightness() {
        let mut vol = LedVolume::new(60.0);
        vol.add_segment(sample_segment(true, 3840));
        vol.set_master_brightness(1.0);
        let full = vol.total_flux_lm();
        vol.set_master_brightness(0.5);
        let half = vol.total_flux_lm();
        assert!((half - full * 0.5).abs() < 1.0);
    }
}
