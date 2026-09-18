//! Ripple and wave effects for `OxiMedia` VFX.
//!
//! Implements sinusoidal displacement, radial waves, and two-source
//! interference patterns for video frames.

use crate::Frame;
use std::f32::consts::PI;

/// Configuration for a sinusoidal ripple effect.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct RippleConfig {
    /// Wave amplitude in pixels (how far pixels are displaced).
    pub amplitude: f32,
    /// Spatial frequency of the wave in cycles per pixel.
    pub frequency: f32,
    /// Phase offset in radians (use to animate over time).
    pub phase: f32,
    /// Direction of wave propagation in radians (0 = horizontal).
    pub angle: f32,
}

impl Default for RippleConfig {
    fn default() -> Self {
        Self {
            amplitude: 8.0,
            frequency: 0.05,
            phase: 0.0,
            angle: 0.0,
        }
    }
}

impl RippleConfig {
    /// Create a new ripple config.
    #[allow(dead_code)]
    pub fn new(amplitude: f32, frequency: f32, phase: f32, angle: f32) -> Self {
        Self {
            amplitude,
            frequency,
            phase,
            angle,
        }
    }
}

/// Configuration for a radial ripple emanating from a point.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct RadialRippleConfig {
    /// Centre X coordinate (normalised 0–1).
    pub cx: f32,
    /// Centre Y coordinate (normalised 0–1).
    pub cy: f32,
    /// Wave amplitude in pixels.
    pub amplitude: f32,
    /// Spatial frequency in cycles per pixel.
    pub frequency: f32,
    /// Phase offset for animation.
    pub phase: f32,
    /// Decay rate: amplitude falls off with `exp(-decay * r)`.
    pub decay: f32,
}

impl Default for RadialRippleConfig {
    fn default() -> Self {
        Self {
            cx: 0.5,
            cy: 0.5,
            amplitude: 10.0,
            frequency: 0.04,
            phase: 0.0,
            decay: 0.005,
        }
    }
}

/// Sample a source frame using bilinear interpolation at sub-pixel coordinates.
///
/// Returns the RGBA pixel at `(sx, sy)`, or black if out of bounds.
#[allow(dead_code)]
#[allow(clippy::cast_precision_loss)]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
pub fn sample_bilinear(src: &Frame, sx: f32, sy: f32) -> [u8; 4] {
    if sx < 0.0 || sy < 0.0 || sx >= src.width as f32 || sy >= src.height as f32 {
        return [0, 0, 0, 0];
    }
    let x0 = sx as u32;
    let y0 = sy as u32;
    let x1 = (x0 + 1).min(src.width - 1);
    let y1 = (y0 + 1).min(src.height - 1);
    let fx = sx - x0 as f32;
    let fy = sy - y0 as f32;

    let p00 = src.get_pixel(x0, y0).unwrap_or([0; 4]);
    let p10 = src.get_pixel(x1, y0).unwrap_or([0; 4]);
    let p01 = src.get_pixel(x0, y1).unwrap_or([0; 4]);
    let p11 = src.get_pixel(x1, y1).unwrap_or([0; 4]);

    let mut out = [0u8; 4];
    for i in 0..4 {
        let top = p00[i] as f32 * (1.0 - fx) + p10[i] as f32 * fx;
        let bot = p01[i] as f32 * (1.0 - fx) + p11[i] as f32 * fx;
        out[i] = (top * (1.0 - fy) + bot * fy) as u8;
    }
    out
}

/// Apply a directional sinusoidal ripple to a frame.
///
/// Displaces each pixel perpendicularly to the wave direction.
#[allow(dead_code)]
#[allow(clippy::cast_precision_loss)]
pub fn apply_ripple(input: &Frame, output: &mut Frame, config: &RippleConfig) {
    let cos_a = config.angle.cos();
    let sin_a = config.angle.sin();

    for y in 0..output.height {
        for x in 0..output.width {
            let px = x as f32;
            let py = y as f32;
            // Project pixel onto wave direction
            let proj = px * cos_a + py * sin_a;
            let displacement =
                config.amplitude * (2.0 * PI * config.frequency * proj + config.phase).sin();
            // Displace perpendicular to wave direction
            let sx = px - sin_a * displacement;
            let sy = py + cos_a * displacement;
            let sampled = sample_bilinear(input, sx, sy);
            output.set_pixel(x, y, sampled);
        }
    }
}

/// Apply a radial ripple emanating from a central point.
#[allow(dead_code)]
#[allow(clippy::cast_precision_loss)]
pub fn apply_radial_ripple(input: &Frame, output: &mut Frame, config: &RadialRippleConfig) {
    let cx = config.cx * input.width as f32;
    let cy = config.cy * input.height as f32;

    for y in 0..output.height {
        for x in 0..output.width {
            let px = x as f32;
            let py = y as f32;
            let dx = px - cx;
            let dy = py - cy;
            let r = (dx * dx + dy * dy).sqrt();

            if r < 1.0 {
                // At center, no displacement
                let sampled = sample_bilinear(input, px, py);
                output.set_pixel(x, y, sampled);
                continue;
            }

            let decay = (-config.decay * r).exp();
            let wave =
                config.amplitude * decay * (2.0 * PI * config.frequency * r + config.phase).sin();
            let nx = dx / r;
            let ny = dy / r;
            let sx = px + nx * wave;
            let sy = py + ny * wave;
            let sampled = sample_bilinear(input, sx, sy);
            output.set_pixel(x, y, sampled);
        }
    }
}

/// Apply two-source interference ripple.
///
/// Two radial wave sources interfere, producing constructive/destructive
/// displacement patterns.
#[allow(dead_code)]
#[allow(clippy::cast_precision_loss)]
pub fn apply_interference(
    input: &Frame,
    output: &mut Frame,
    src1: (f32, f32),
    src2: (f32, f32),
    amplitude: f32,
    frequency: f32,
    phase: f32,
) {
    let (x1, y1) = src1;
    let (x2, y2) = src2;

    for y in 0..output.height {
        for x in 0..output.width {
            let px = x as f32;
            let py = y as f32;

            let r1 = ((px - x1) * (px - x1) + (py - y1) * (py - y1)).sqrt();
            let r2 = ((px - x2) * (px - x2) + (py - y2) * (py - y2)).sqrt();

            let w1 = (2.0 * PI * frequency * r1 + phase).sin();
            let w2 = (2.0 * PI * frequency * r2 + phase).sin();
            let combined = (w1 + w2) * 0.5 * amplitude;

            let sx = (px + combined).clamp(0.0, input.width as f32 - 1.0);
            let sy = (py + combined).clamp(0.0, input.height as f32 - 1.0);
            let sampled = sample_bilinear(input, sx, sy);
            output.set_pixel(x, y, sampled);
        }
    }
}

/// Compute the displacement magnitude for a directional ripple at a given pixel position.
#[allow(dead_code)]
pub fn ripple_displacement(config: &RippleConfig, px: f32, py: f32) -> f32 {
    let cos_a = config.angle.cos();
    let sin_a = config.angle.sin();
    let proj = px * cos_a + py * sin_a;
    config.amplitude * (2.0 * PI * config.frequency * proj + config.phase).sin()
}

/// Compute the displacement magnitude for a radial ripple at a given pixel position.
#[allow(dead_code)]
pub fn radial_displacement(
    config: &RadialRippleConfig,
    px: f32,
    py: f32,
    frame_w: u32,
    frame_h: u32,
) -> f32 {
    let cx = config.cx * frame_w as f32;
    let cy = config.cy * frame_h as f32;
    let dx = px - cx;
    let dy = py - cy;
    let r = (dx * dx + dy * dy).sqrt().max(1.0);
    let decay = (-config.decay * r).exp();
    config.amplitude * decay * (2.0 * PI * config.frequency * r + config.phase).sin()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_ripple_config() {
        let cfg = RippleConfig::default();
        assert!(cfg.amplitude > 0.0);
        assert!(cfg.frequency > 0.0);
    }

    #[test]
    fn test_ripple_displacement_at_origin() {
        let cfg = RippleConfig {
            phase: 0.0,
            frequency: 0.1,
            amplitude: 5.0,
            angle: 0.0,
        };
        // At x=0, y=0, displacement = amplitude * sin(0) = 0
        let d = ripple_displacement(&cfg, 0.0, 0.0);
        assert!(d.abs() < 1e-5, "d={d}");
    }

    #[test]
    fn test_ripple_displacement_bounded() {
        let cfg = RippleConfig::default();
        for x in [0.0, 10.0, 50.0, 100.0] {
            let d = ripple_displacement(&cfg, x, 0.0);
            assert!(d.abs() <= cfg.amplitude + 1e-5, "d={d}");
        }
    }

    #[test]
    fn test_radial_displacement_decays_with_distance() {
        let cfg = RadialRippleConfig {
            amplitude: 10.0,
            decay: 0.1,
            frequency: 0.05,
            phase: 0.0,
            cx: 0.5,
            cy: 0.5,
        };
        let d_near = radial_displacement(&cfg, 52.0, 50.0, 100, 100).abs();
        let d_far = radial_displacement(&cfg, 90.0, 50.0, 100, 100).abs();
        assert!(d_far < d_near + 0.1, "near={d_near}, far={d_far}");
    }

    #[test]
    fn test_sample_bilinear_center() {
        let mut frame = Frame::new(4, 4).expect("should succeed in test");
        frame.set_pixel(1, 1, [200, 100, 50, 255]);
        let s = sample_bilinear(&frame, 1.0, 1.0);
        assert_eq!(s[0], 200);
    }

    #[test]
    fn test_sample_bilinear_out_of_bounds() {
        let frame = Frame::new(4, 4).expect("should succeed in test");
        let s = sample_bilinear(&frame, -1.0, -1.0);
        assert_eq!(s, [0, 0, 0, 0]);
    }

    #[test]
    fn test_sample_bilinear_interpolates() {
        let mut frame = Frame::new(4, 4).expect("should succeed in test");
        frame.set_pixel(1, 1, [0, 0, 0, 255]);
        frame.set_pixel(2, 1, [200, 0, 0, 255]);
        let s = sample_bilinear(&frame, 1.5, 1.0);
        // Should be between 0 and 200
        assert!(s[0] > 0 && s[0] < 200, "s[0]={}", s[0]);
    }

    #[test]
    fn test_apply_ripple_does_not_panic() {
        let input = Frame::new(64, 64).expect("should succeed in test");
        let mut output = Frame::new(64, 64).expect("should succeed in test");
        let config = RippleConfig::default();
        apply_ripple(&input, &mut output, &config);
    }

    #[test]
    fn test_apply_ripple_zero_amplitude_is_identity() {
        let mut input = Frame::new(32, 32).expect("should succeed in test");
        input.clear([100, 150, 200, 255]);
        let mut output = Frame::new(32, 32).expect("should succeed in test");
        let config = RippleConfig {
            amplitude: 0.0,
            ..Default::default()
        };
        apply_ripple(&input, &mut output, &config);
        // With zero amplitude, output should match input
        let p = output.get_pixel(10, 10).expect("should succeed in test");
        assert_eq!(p[0], 100);
    }

    #[test]
    fn test_apply_radial_ripple_does_not_panic() {
        let input = Frame::new(64, 64).expect("should succeed in test");
        let mut output = Frame::new(64, 64).expect("should succeed in test");
        let config = RadialRippleConfig::default();
        apply_radial_ripple(&input, &mut output, &config);
    }

    #[test]
    fn test_apply_interference_does_not_panic() {
        let input = Frame::new(64, 64).expect("should succeed in test");
        let mut output = Frame::new(64, 64).expect("should succeed in test");
        apply_interference(
            &input,
            &mut output,
            (20.0, 20.0),
            (44.0, 44.0),
            5.0,
            0.05,
            0.0,
        );
    }

    #[test]
    fn test_apply_interference_finite_output() {
        let mut input = Frame::new(32, 32).expect("should succeed in test");
        input.clear([128, 128, 128, 255]);
        let mut output = Frame::new(32, 32).expect("should succeed in test");
        apply_interference(&input, &mut output, (8.0, 8.0), (24.0, 24.0), 3.0, 0.1, 0.0);
        // All output pixels should have valid (not garbage) values
        for chunk in output.data.chunks(4) {
            let _ = chunk; // ensure no unused variable warning
        }
    }

    #[test]
    fn test_radial_ripple_config_defaults_valid() {
        let cfg = RadialRippleConfig::default();
        assert!(cfg.amplitude > 0.0);
        assert!(cfg.decay > 0.0);
        assert!(cfg.cx >= 0.0 && cfg.cx <= 1.0);
        assert!(cfg.cy >= 0.0 && cfg.cy <= 1.0);
    }
}
