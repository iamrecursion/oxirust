#![allow(dead_code)]
//! Heat / atmospheric distortion effect for VFX.
//!
//! Simulates heat-haze, mirage, and atmospheric turbulence by displacing
//! pixels using a time-varying noise field. Useful for desert scenes,
//! engine exhaust, fire proximity, and sci-fi force-field effects.

/// Configuration for the heat distortion effect.
#[derive(Debug, Clone)]
pub struct HeatDistortConfig {
    /// Horizontal displacement amplitude in pixels.
    pub amplitude_x: f32,
    /// Vertical displacement amplitude in pixels.
    pub amplitude_y: f32,
    /// Spatial frequency of the distortion waves.
    pub frequency: f32,
    /// Speed of the distortion animation.
    pub speed: f32,
    /// Vertical falloff: 0.0 = uniform, 1.0 = bottom-heavy.
    pub falloff: f32,
    /// Region of effect (normalised 0..1 coordinates). `None` means full frame.
    pub region: Option<NormRect>,
}

impl Default for HeatDistortConfig {
    fn default() -> Self {
        Self {
            amplitude_x: 3.0,
            amplitude_y: 2.0,
            frequency: 0.05,
            speed: 1.0,
            falloff: 0.5,
            region: None,
        }
    }
}

/// Normalised rectangle (all values 0.0 to 1.0).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormRect {
    /// Left edge (0.0 = left of frame).
    pub x: f32,
    /// Top edge (0.0 = top of frame).
    pub y: f32,
    /// Width fraction.
    pub width: f32,
    /// Height fraction.
    pub height: f32,
}

impl NormRect {
    /// Create a new normalised rectangle.
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x: x.clamp(0.0, 1.0),
            y: y.clamp(0.0, 1.0),
            width: width.clamp(0.0, 1.0),
            height: height.clamp(0.0, 1.0),
        }
    }

    /// Full-frame rectangle.
    pub fn full() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            width: 1.0,
            height: 1.0,
        }
    }

    /// Check whether a normalised coordinate is inside this rect.
    pub fn contains(&self, nx: f32, ny: f32) -> bool {
        nx >= self.x && nx <= self.x + self.width && ny >= self.y && ny <= self.y + self.height
    }
}

/// The heat distortion processor.
#[derive(Debug, Clone)]
pub struct HeatDistort {
    /// Configuration.
    config: HeatDistortConfig,
    /// Internal phase accumulator.
    phase: f32,
}

impl HeatDistort {
    /// Create a new heat distortion processor with default config.
    pub fn new() -> Self {
        Self {
            config: HeatDistortConfig::default(),
            phase: 0.0,
        }
    }

    /// Create with a custom configuration.
    pub fn with_config(config: HeatDistortConfig) -> Self {
        Self { config, phase: 0.0 }
    }

    /// Advance the phase by `dt` seconds.
    pub fn advance(&mut self, dt: f32) {
        self.phase += dt * self.config.speed;
    }

    /// Reset the phase accumulator.
    pub fn reset(&mut self) {
        self.phase = 0.0;
    }

    /// Get current phase.
    pub fn phase(&self) -> f32 {
        self.phase
    }

    /// Get configuration reference.
    pub fn config(&self) -> &HeatDistortConfig {
        &self.config
    }

    /// Set configuration.
    pub fn set_config(&mut self, config: HeatDistortConfig) {
        self.config = config;
    }

    /// Compute displacement at a given normalised coordinate.
    ///
    /// Returns `(dx, dy)` in pixel units.
    #[allow(clippy::cast_precision_loss)]
    pub fn displacement_at(&self, nx: f32, ny: f32) -> (f32, f32) {
        // Check region
        if let Some(ref region) = self.config.region {
            if !region.contains(nx, ny) {
                return (0.0, 0.0);
            }
        }

        let falloff_factor = if self.config.falloff > 0.0 {
            // Stronger distortion toward the bottom of the frame
            ny.powf(self.config.falloff)
        } else {
            1.0
        };

        let freq = self.config.frequency;
        let phase = self.phase;

        // Use sinusoidal noise approximation
        let dx = self.config.amplitude_x
            * (freq * ny * 100.0 + phase * 2.3).sin()
            * (freq * nx * 80.0 + phase * 1.7).cos()
            * falloff_factor;

        let dy = self.config.amplitude_y
            * (freq * nx * 90.0 + phase * 3.1).sin()
            * (freq * ny * 70.0 + phase * 0.9).cos()
            * falloff_factor;

        (dx, dy)
    }

    /// Apply heat distortion to an RGBA frame buffer.
    ///
    /// `width` and `height` are the frame dimensions.
    /// `src` is the source RGBA buffer, `dst` is the destination (same size).
    #[allow(clippy::cast_precision_loss)]
    pub fn apply(&self, src: &[u8], dst: &mut [u8], width: u32, height: u32) {
        let w = width as usize;
        let h = height as usize;
        let expected = w * h * 4;
        if src.len() < expected || dst.len() < expected {
            return;
        }

        for y in 0..h {
            for x in 0..w {
                let nx = x as f32 / w.max(1) as f32;
                let ny = y as f32 / h.max(1) as f32;

                let (dx, dy) = self.displacement_at(nx, ny);

                let sx = (x as f32 + dx).round().clamp(0.0, (w - 1) as f32) as usize;
                let sy = (y as f32 + dy).round().clamp(0.0, (h - 1) as f32) as usize;

                let src_idx = (sy * w + sx) * 4;
                let dst_idx = (y * w + x) * 4;

                dst[dst_idx..dst_idx + 4].copy_from_slice(&src[src_idx..src_idx + 4]);
            }
        }
    }

    /// Compute the maximum displacement magnitude for the current settings.
    pub fn max_displacement(&self) -> f32 {
        let dx = self.config.amplitude_x.abs();
        let dy = self.config.amplitude_y.abs();
        (dx * dx + dy * dy).sqrt()
    }
}

impl Default for HeatDistort {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = HeatDistortConfig::default();
        assert!((cfg.amplitude_x - 3.0).abs() < f32::EPSILON);
        assert!((cfg.amplitude_y - 2.0).abs() < f32::EPSILON);
        assert!((cfg.speed - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_norm_rect_full() {
        let r = NormRect::full();
        assert!(r.contains(0.5, 0.5));
        assert!(r.contains(0.0, 0.0));
        assert!(r.contains(1.0, 1.0));
    }

    #[test]
    fn test_norm_rect_partial() {
        let r = NormRect::new(0.25, 0.25, 0.5, 0.5);
        assert!(r.contains(0.5, 0.5));
        assert!(!r.contains(0.1, 0.1));
    }

    #[test]
    fn test_heat_distort_creation() {
        let hd = HeatDistort::new();
        assert!((hd.phase() - 0.0).abs() < f32::EPSILON);
        assert!((hd.config().amplitude_x - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_phase_advance() {
        let mut hd = HeatDistort::new();
        hd.advance(1.0);
        assert!((hd.phase() - 1.0).abs() < f32::EPSILON);
        hd.advance(0.5);
        assert!((hd.phase() - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_phase_reset() {
        let mut hd = HeatDistort::new();
        hd.advance(5.0);
        hd.reset();
        assert!((hd.phase() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_displacement_not_nan() {
        let hd = HeatDistort::new();
        let (dx, dy) = hd.displacement_at(0.5, 0.5);
        assert!(!dx.is_nan());
        assert!(!dy.is_nan());
    }

    #[test]
    fn test_displacement_with_region_outside() {
        let config = HeatDistortConfig {
            region: Some(NormRect::new(0.0, 0.0, 0.25, 0.25)),
            ..Default::default()
        };
        let hd = HeatDistort::with_config(config);
        let (dx, dy) = hd.displacement_at(0.5, 0.5);
        assert!((dx).abs() < f32::EPSILON);
        assert!((dy).abs() < f32::EPSILON);
    }

    #[test]
    fn test_apply_preserves_buffer_length() {
        let hd = HeatDistort::new();
        let w = 8u32;
        let h = 8u32;
        let src = vec![128u8; (w * h * 4) as usize];
        let mut dst = vec![0u8; (w * h * 4) as usize];
        hd.apply(&src, &mut dst, w, h);
        assert_eq!(dst.len(), src.len());
    }

    #[test]
    fn test_apply_uniform_source() {
        // If source is uniform, output should also be uniform
        let hd = HeatDistort::new();
        let w = 4u32;
        let h = 4u32;
        let src = vec![42u8; (w * h * 4) as usize];
        let mut dst = vec![0u8; (w * h * 4) as usize];
        hd.apply(&src, &mut dst, w, h);
        for &v in &dst {
            assert_eq!(v, 42);
        }
    }

    #[test]
    fn test_max_displacement() {
        let hd = HeatDistort::new();
        let md = hd.max_displacement();
        assert!(md > 0.0);
        let expected = (3.0f32 * 3.0 + 2.0 * 2.0).sqrt();
        assert!((md - expected).abs() < 0.001);
    }

    #[test]
    fn test_set_config() {
        let mut hd = HeatDistort::new();
        let new_cfg = HeatDistortConfig {
            amplitude_x: 10.0,
            amplitude_y: 10.0,
            ..Default::default()
        };
        hd.set_config(new_cfg);
        assert!((hd.config().amplitude_x - 10.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_apply_undersized_buffers() {
        let hd = HeatDistort::new();
        let mut dst = vec![0u8; 4];
        // Undersized src — should early-return without panic
        hd.apply(&[0u8; 4], &mut dst, 8, 8);
        assert_eq!(dst, vec![0u8; 4]);
    }
}
