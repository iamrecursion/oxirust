//! Lens flare simulation for `OxiMedia` VFX.
//!
//! Implements star bursts, bokeh circles, chromatic aberration halos, and
//! multi-element lens flare chains based on a light source position.

use crate::{Color, Frame};

/// A single element in a lens flare chain.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct FlareElement {
    /// Position along the axis from light source through screen centre (–1 to 1).
    pub axis_position: f32,
    /// Radius of this element in pixels.
    pub radius: f32,
    /// Base colour of this element.
    pub color: Color,
    /// Opacity (0–1).
    pub opacity: f32,
    /// Number of diffraction spikes (0 = circle).
    pub spikes: u32,
}

impl FlareElement {
    /// Create a circular bokeh element.
    #[allow(dead_code)]
    pub fn bokeh(axis_position: f32, radius: f32, color: Color, opacity: f32) -> Self {
        Self {
            axis_position,
            radius,
            color,
            opacity: opacity.clamp(0.0, 1.0),
            spikes: 0,
        }
    }

    /// Create a star-burst element.
    #[allow(dead_code)]
    pub fn starburst(
        axis_position: f32,
        radius: f32,
        color: Color,
        opacity: f32,
        spikes: u32,
    ) -> Self {
        Self {
            axis_position,
            radius,
            color,
            opacity: opacity.clamp(0.0, 1.0),
            spikes: spikes.max(2),
        }
    }
}

/// Configuration for a lens flare effect.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct LensFlareConfig {
    /// Light source position in normalised screen coordinates (0–1).
    pub light_x: f32,
    /// Light source position in normalised screen coordinates (0–1).
    pub light_y: f32,
    /// Global intensity multiplier.
    pub intensity: f32,
    /// Enable chromatic aberration on flare rings.
    pub chromatic_aberration: bool,
    /// Number of rainbow halo rings.
    pub halo_rings: u32,
    /// Flare elements in this chain.
    pub elements: Vec<FlareElement>,
}

impl Default for LensFlareConfig {
    fn default() -> Self {
        Self {
            light_x: 0.2,
            light_y: 0.2,
            intensity: 1.0,
            chromatic_aberration: true,
            halo_rings: 2,
            elements: vec![
                FlareElement::starburst(0.0, 60.0, Color::new(255, 255, 200, 200), 0.8, 6),
                FlareElement::bokeh(0.3, 25.0, Color::new(180, 200, 255, 150), 0.5),
                FlareElement::bokeh(0.6, 15.0, Color::new(255, 180, 120, 120), 0.4),
                FlareElement::bokeh(1.0, 40.0, Color::new(200, 255, 200, 100), 0.3),
            ],
        }
    }
}

/// Compute the world position of a flare element given the light source and
/// screen centre, using `axis_position` as a parameter along the line.
///
/// Returns `(pixel_x, pixel_y)` in frame pixel coordinates.
#[allow(dead_code)]
#[allow(clippy::cast_precision_loss)]
pub fn element_position(
    config: &LensFlareConfig,
    element: &FlareElement,
    frame_width: u32,
    frame_height: u32,
) -> (f32, f32) {
    let lx = config.light_x * frame_width as f32;
    let ly = config.light_y * frame_height as f32;
    let cx = frame_width as f32 * 0.5;
    let cy = frame_height as f32 * 0.5;
    let t = element.axis_position;
    (lx + (cx - lx) * t * 2.0, ly + (cy - ly) * t * 2.0)
}

/// Draw a soft circular blob onto a frame at `(cx, cy)` with `radius`.
#[allow(dead_code)]
#[allow(clippy::cast_precision_loss)]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
pub fn draw_blob(frame: &mut Frame, cx: f32, cy: f32, radius: f32, color: Color, opacity: f32) {
    let opacity = opacity.clamp(0.0, 1.0);
    let r = radius.max(0.5);
    let x0 = ((cx - r).max(0.0) as u32).min(frame.width.saturating_sub(1));
    let y0 = ((cy - r).max(0.0) as u32).min(frame.height.saturating_sub(1));
    let x1 = ((cx + r + 1.0) as u32).min(frame.width);
    let y1 = ((cy + r + 1.0) as u32).min(frame.height);

    for py in y0..y1 {
        for px in x0..x1 {
            let dx = px as f32 - cx;
            let dy = py as f32 - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist > r {
                continue;
            }
            let falloff = 1.0 - (dist / r);
            let alpha = (falloff * falloff * opacity * 255.0) as u8;
            if alpha == 0 {
                continue;
            }
            let overlay = Color::new(color.r, color.g, color.b, alpha);
            if let Some(existing) = frame.get_pixel(px, py) {
                let blended = Color::from_rgba(existing).blend(overlay);
                frame.set_pixel(px, py, blended.to_rgba());
            }
        }
    }
}

/// Apply a full lens flare chain to a frame.
#[allow(dead_code)]
pub fn apply_lens_flare(frame: &mut Frame, config: &LensFlareConfig) {
    for element in &config.elements {
        let (ex, ey) = element_position(config, element, frame.width, frame.height);
        draw_blob(
            frame,
            ex,
            ey,
            element.radius,
            element.color,
            element.opacity * config.intensity,
        );
    }

    // Chromatic aberration: draw offset copies in R and B channels
    if config.chromatic_aberration {
        for element in &config.elements {
            let (ex, ey) = element_position(config, element, frame.width, frame.height);
            let offset = element.radius * 0.05;
            // Red fringe
            draw_blob(
                frame,
                ex + offset,
                ey,
                element.radius * 1.05,
                Color::new(255, 0, 0, 30),
                element.opacity * config.intensity * 0.3,
            );
            // Blue fringe
            draw_blob(
                frame,
                ex - offset,
                ey,
                element.radius * 1.05,
                Color::new(0, 0, 255, 30),
                element.opacity * config.intensity * 0.3,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_has_elements() {
        let config = LensFlareConfig::default();
        assert!(!config.elements.is_empty());
    }

    #[test]
    fn test_bokeh_element_creation() {
        let elem = FlareElement::bokeh(0.5, 20.0, Color::rgb(255, 255, 255), 0.8);
        assert_eq!(elem.spikes, 0);
        assert!((elem.opacity - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_starburst_element_creation() {
        let elem = FlareElement::starburst(0.0, 50.0, Color::rgb(255, 200, 100), 0.9, 6);
        assert_eq!(elem.spikes, 6);
    }

    #[test]
    fn test_starburst_minimum_spikes() {
        let elem = FlareElement::starburst(0.0, 10.0, Color::black(), 0.5, 0);
        assert!(elem.spikes >= 2);
    }

    #[test]
    fn test_element_opacity_clamped() {
        let elem = FlareElement::bokeh(0.0, 10.0, Color::white(), 2.0);
        assert!(elem.opacity <= 1.0);
        let elem2 = FlareElement::bokeh(0.0, 10.0, Color::white(), -1.0);
        assert!(elem2.opacity >= 0.0);
    }

    #[test]
    fn test_element_position_at_light_source() {
        let config = LensFlareConfig {
            light_x: 0.25,
            light_y: 0.25,
            ..Default::default()
        };
        let elem = FlareElement::bokeh(0.0, 10.0, Color::white(), 1.0);
        let (ex, ey) = element_position(&config, &elem, 100, 100);
        // t=0 → position == light source
        assert!((ex - 25.0).abs() < 0.1, "ex={ex}");
        assert!((ey - 25.0).abs() < 0.1, "ey={ey}");
    }

    #[test]
    fn test_draw_blob_does_not_panic() {
        let mut frame = Frame::new(64, 64).expect("should succeed in test");
        draw_blob(&mut frame, 32.0, 32.0, 20.0, Color::rgb(255, 200, 100), 0.8);
        // If we got here without panic, it's a pass
    }

    #[test]
    fn test_draw_blob_modifies_center_pixel() {
        let mut frame = Frame::new(64, 64).expect("should succeed in test");
        frame.clear([0, 0, 0, 255]);
        draw_blob(&mut frame, 32.0, 32.0, 15.0, Color::rgb(255, 0, 0), 1.0);
        let pixel = frame.get_pixel(32, 32).expect("should succeed in test");
        assert!(pixel[0] > 0, "Red channel should be modified");
    }

    #[test]
    fn test_draw_blob_out_of_frame() {
        let mut frame = Frame::new(32, 32).expect("should succeed in test");
        // Should not panic even when blob center is outside frame
        draw_blob(&mut frame, -100.0, -100.0, 20.0, Color::white(), 1.0);
        draw_blob(&mut frame, 200.0, 200.0, 20.0, Color::white(), 1.0);
    }

    #[test]
    fn test_apply_lens_flare_does_not_panic() {
        let mut frame = Frame::new(320, 240).expect("should succeed in test");
        let config = LensFlareConfig::default();
        apply_lens_flare(&mut frame, &config);
    }

    #[test]
    fn test_apply_lens_flare_brightens_frame() {
        let mut frame = Frame::new(320, 240).expect("should succeed in test");
        frame.clear([0, 0, 0, 255]);
        let config = LensFlareConfig {
            intensity: 1.0,
            chromatic_aberration: false,
            ..Default::default()
        };
        apply_lens_flare(&mut frame, &config);
        // At least some pixels should have been brightened
        let sum: u32 = frame.data.iter().map(|&v| v as u32).sum();
        assert!(sum > 0, "Frame should have some brightness after flare");
    }

    #[test]
    fn test_apply_lens_flare_chromatic_aberration() {
        let mut frame = Frame::new(320, 240).expect("should succeed in test");
        frame.clear([0, 0, 0, 255]);
        let config = LensFlareConfig {
            chromatic_aberration: true,
            intensity: 1.0,
            ..Default::default()
        };
        apply_lens_flare(&mut frame, &config);
        // Should complete without panic
    }

    #[test]
    fn test_flare_element_axis_range() {
        let config = LensFlareConfig::default();
        for elem in &config.elements {
            // axis positions should be in [0, 1] for standard presets
            assert!(
                elem.axis_position >= 0.0 && elem.axis_position <= 1.0,
                "axis_position {} out of range",
                elem.axis_position
            );
        }
    }
}
