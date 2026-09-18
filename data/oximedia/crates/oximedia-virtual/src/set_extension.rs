//! Set extension for virtual production: extending physical sets with virtual elements.
//!
//! Provides tools for seamlessly blending physical set pieces with CG environment
//! extensions, including edge matting, perspective matching, and sky/floor replacement.

use crate::{Result, VirtualProductionError};
use serde::{Deserialize, Serialize};

/// The type of set extension element.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExtensionType {
    /// Replace sky region above the horizon.
    SkyReplacement,
    /// Extend floor/ground plane with a virtual surface.
    FloorExtension,
    /// Extend walls and architecture.
    ArchitectureExtension,
    /// Add foreground elements (props, foliage) in front of set.
    ForegroundElement,
    /// Completely replace background behind the physical set.
    BackgroundPlate,
}

/// Edge blending mode for physical/virtual boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeBlendMode {
    /// Hard cut at the boundary.
    Hard,
    /// Soft feathered blend across a transition width.
    Feather,
    /// Smart edge detection + gradient blend.
    SmartEdge,
    /// Alpha gradient from a pre-computed matte.
    MatteBased,
}

/// A single set extension element.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SetExtensionElement {
    /// Unique identifier for this element.
    pub id: String,
    /// Type of extension.
    pub extension_type: ExtensionType,
    /// Blend mode at the physical/virtual boundary.
    pub blend_mode: EdgeBlendMode,
    /// Feather width in pixels (for `Feather` and `SmartEdge` modes).
    pub feather_px: f32,
    /// Vertical split line Y position [0.0, 1.0] normalized.
    /// For SkyReplacement: pixels above this are virtual.
    pub split_y: f32,
    /// Horizontal split line X position [0.0, 1.0] normalized.
    pub split_x: f32,
    /// Whether this element is active.
    pub active: bool,
    /// Overall opacity [0.0, 1.0].
    pub opacity: f32,
}

impl SetExtensionElement {
    /// Create a new sky replacement element.
    #[must_use]
    pub fn sky(id: &str, horizon_y: f32) -> Self {
        Self {
            id: id.to_string(),
            extension_type: ExtensionType::SkyReplacement,
            blend_mode: EdgeBlendMode::Feather,
            feather_px: 20.0,
            split_y: horizon_y.clamp(0.0, 1.0),
            split_x: 0.5,
            active: true,
            opacity: 1.0,
        }
    }

    /// Create a floor extension element.
    #[must_use]
    pub fn floor(id: &str, horizon_y: f32) -> Self {
        Self {
            id: id.to_string(),
            extension_type: ExtensionType::FloorExtension,
            blend_mode: EdgeBlendMode::SmartEdge,
            feather_px: 15.0,
            split_y: horizon_y.clamp(0.0, 1.0),
            split_x: 0.5,
            active: true,
            opacity: 1.0,
        }
    }

    /// Set opacity.
    #[must_use]
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity.clamp(0.0, 1.0);
        self
    }

    /// Set blend mode.
    #[must_use]
    pub fn with_blend_mode(mut self, mode: EdgeBlendMode) -> Self {
        self.blend_mode = mode;
        self
    }
}

/// Perspective matching parameters for aligning virtual extensions to the physical camera.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerspectiveParams {
    /// Camera field of view in degrees.
    pub fov_deg: f64,
    /// Camera tilt angle in degrees (positive = tilting up).
    pub tilt_deg: f64,
    /// Camera pan angle in degrees.
    pub pan_deg: f64,
    /// Camera roll angle in degrees.
    pub roll_deg: f64,
    /// Camera height in meters above the ground.
    pub height_m: f64,
}

impl Default for PerspectiveParams {
    fn default() -> Self {
        Self {
            fov_deg: 50.0,
            tilt_deg: 0.0,
            pan_deg: 0.0,
            roll_deg: 0.0,
            height_m: 1.5,
        }
    }
}

impl PerspectiveParams {
    /// Compute the image Y coordinate of the vanishing horizon line.
    #[must_use]
    pub fn horizon_y_normalized(&self, image_height: usize) -> f32 {
        // Horizon appears at atan(camera_tilt) within the vertical FOV
        let vfov_half_rad = (self.fov_deg / 2.0).to_radians();
        let tilt_rad = self.tilt_deg.to_radians();
        // Fraction from center: tilt / (vfov/2), clamped
        let frac = (tilt_rad / vfov_half_rad).clamp(-1.0, 1.0);
        // Convert to normalized [0,1] where 0 = top, 1 = bottom.
        // Tilting up (positive) makes the horizon appear lower in the frame (larger Y).
        let _ = image_height; // not needed for normalized output
        (0.5 + frac as f32 / 2.0).clamp(0.0, 1.0)
    }
}

/// Compositing result from a set extension operation.
#[derive(Debug, Clone)]
pub struct ExtendedFrame {
    /// RGB pixel data.
    pub pixels: Vec<u8>,
    /// Frame width.
    pub width: usize,
    /// Frame height.
    pub height: usize,
    /// Number of active extensions applied.
    pub extensions_applied: usize,
}

impl ExtendedFrame {
    #[must_use]
    pub fn get_pixel(&self, x: usize, y: usize) -> Option<[u8; 3]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let idx = (y * self.width + x) * 3;
        Some([self.pixels[idx], self.pixels[idx + 1], self.pixels[idx + 2]])
    }
}

/// Set extension compositor.
pub struct SetExtensionCompositor {
    elements: Vec<SetExtensionElement>,
    perspective: PerspectiveParams,
}

impl SetExtensionCompositor {
    /// Create a new compositor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            elements: Vec::new(),
            perspective: PerspectiveParams::default(),
        }
    }

    /// Add an extension element.
    pub fn add_element(&mut self, element: SetExtensionElement) {
        self.elements.push(element);
    }

    /// Remove element by id.
    pub fn remove_element(&mut self, id: &str) {
        self.elements.retain(|e| e.id != id);
    }

    /// Set perspective matching parameters.
    pub fn set_perspective(&mut self, params: PerspectiveParams) {
        self.perspective = params;
    }

    /// Number of registered elements.
    #[must_use]
    pub fn element_count(&self) -> usize {
        self.elements.len()
    }

    /// Active element count.
    #[must_use]
    pub fn active_element_count(&self) -> usize {
        self.elements.iter().filter(|e| e.active).count()
    }

    /// Apply set extensions: blend physical frame with virtual content.
    ///
    /// `physical` is the camera feed (RGB, row-major).
    /// `virtual_bg` is the CG environment render (same dimensions, RGB).
    ///
    /// Returns a composited frame.
    pub fn apply(
        &self,
        physical: &[u8],
        virtual_bg: &[u8],
        width: usize,
        height: usize,
    ) -> Result<ExtendedFrame> {
        let expected = width * height * 3;
        if physical.len() != expected {
            return Err(VirtualProductionError::Compositing(format!(
                "Physical frame size mismatch: expected {expected}, got {}",
                physical.len()
            )));
        }
        if virtual_bg.len() != expected {
            return Err(VirtualProductionError::Compositing(format!(
                "Virtual frame size mismatch: expected {expected}, got {}",
                virtual_bg.len()
            )));
        }

        // Build alpha mask from active elements
        let mut alpha = vec![0.0f32; width * height];
        let mut applied = 0usize;

        for element in self.elements.iter().filter(|e| e.active) {
            self.apply_element_mask(&mut alpha, element, width, height);
            applied += 1;
        }

        // Composite: out = physical * (1-alpha) + virtual * alpha
        let mut out = Vec::with_capacity(expected);
        for i in 0..(width * height) {
            let a = alpha[i].clamp(0.0, 1.0);
            let p_r = physical[i * 3] as f32;
            let p_g = physical[i * 3 + 1] as f32;
            let p_b = physical[i * 3 + 2] as f32;
            let v_r = virtual_bg[i * 3] as f32;
            let v_g = virtual_bg[i * 3 + 1] as f32;
            let v_b = virtual_bg[i * 3 + 2] as f32;

            out.push((p_r * (1.0 - a) + v_r * a) as u8);
            out.push((p_g * (1.0 - a) + v_g * a) as u8);
            out.push((p_b * (1.0 - a) + v_b * a) as u8);
        }

        Ok(ExtendedFrame {
            pixels: out,
            width,
            height,
            extensions_applied: applied,
        })
    }

    /// Compute per-pixel alpha contribution from a single element.
    fn apply_element_mask(
        &self,
        alpha: &mut [f32],
        element: &SetExtensionElement,
        width: usize,
        height: usize,
    ) {
        let opacity = element.opacity;
        let split_y_px = (element.split_y * height as f32) as i32;
        let feather = element.feather_px.max(1.0);

        match element.extension_type {
            ExtensionType::SkyReplacement => {
                // Virtual sky = above split_y
                for y in 0..height {
                    let dist = split_y_px - y as i32;
                    let a = match element.blend_mode {
                        EdgeBlendMode::Hard => {
                            if dist > 0 {
                                1.0f32
                            } else {
                                0.0
                            }
                        }
                        EdgeBlendMode::Feather
                        | EdgeBlendMode::SmartEdge
                        | EdgeBlendMode::MatteBased => {
                            // Smooth transition around split
                            let t = (dist as f32 / feather).clamp(-1.0, 1.0);
                            (t * 0.5 + 0.5).clamp(0.0, 1.0)
                        }
                    } * opacity;

                    for x in 0..width {
                        let i = y * width + x;
                        alpha[i] = alpha[i].max(a);
                    }
                }
            }
            ExtensionType::FloorExtension | ExtensionType::BackgroundPlate => {
                // Virtual extension = below split_y (inclusive of the split line)
                for y in 0..height {
                    let dist = y as i32 - split_y_px;
                    let a = match element.blend_mode {
                        EdgeBlendMode::Hard => {
                            if dist >= 0 {
                                1.0f32
                            } else {
                                0.0
                            }
                        }
                        _ => {
                            let t = (dist as f32 / feather).clamp(-1.0, 1.0);
                            (t * 0.5 + 0.5).clamp(0.0, 1.0)
                        }
                    } * opacity;

                    for x in 0..width {
                        let i = y * width + x;
                        alpha[i] = alpha[i].max(a);
                    }
                }
            }
            ExtensionType::ArchitectureExtension | ExtensionType::ForegroundElement => {
                // Apply uniform opacity everywhere (architecture / foreground props)
                for i in 0..(width * height) {
                    alpha[i] = alpha[i].max(opacity);
                }
            }
        }
    }

    /// Get perspective parameters.
    #[must_use]
    pub fn perspective(&self) -> &PerspectiveParams {
        &self.perspective
    }

    /// Get registered elements.
    #[must_use]
    pub fn elements(&self) -> &[SetExtensionElement] {
        &self.elements
    }
}

impl Default for SetExtensionCompositor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_rgb(w: usize, h: usize, r: u8, g: u8, b: u8) -> Vec<u8> {
        let mut buf = Vec::with_capacity(w * h * 3);
        for _ in 0..(w * h) {
            buf.push(r);
            buf.push(g);
            buf.push(b);
        }
        buf
    }

    #[test]
    fn test_compositor_creation() {
        let c = SetExtensionCompositor::new();
        assert_eq!(c.element_count(), 0);
    }

    #[test]
    fn test_add_remove_element() {
        let mut c = SetExtensionCompositor::new();
        c.add_element(SetExtensionElement::sky("sky1", 0.4));
        c.add_element(SetExtensionElement::floor("floor1", 0.6));
        assert_eq!(c.element_count(), 2);
        c.remove_element("sky1");
        assert_eq!(c.element_count(), 1);
    }

    #[test]
    fn test_active_element_count() {
        let mut c = SetExtensionCompositor::new();
        let mut el = SetExtensionElement::sky("sky", 0.5);
        el.active = false;
        c.add_element(el);
        c.add_element(SetExtensionElement::floor("floor", 0.6));
        assert_eq!(c.active_element_count(), 1);
    }

    #[test]
    fn test_apply_no_elements_returns_physical() {
        let c = SetExtensionCompositor::new();
        let physical = solid_rgb(8, 8, 255, 0, 0);
        let virtual_bg = solid_rgb(8, 8, 0, 0, 255);
        let result = c.apply(&physical, &virtual_bg, 8, 8);
        assert!(result.is_ok());
        let frame = result.expect("ok");
        // No alpha → output = physical
        assert_eq!(frame.get_pixel(0, 0), Some([255, 0, 0]));
    }

    #[test]
    fn test_apply_sky_replacement_top_region() {
        let mut c = SetExtensionCompositor::new();
        // Horizon at 50% → top half is sky
        c.add_element(SetExtensionElement::sky("sky", 0.5).with_blend_mode(EdgeBlendMode::Hard));

        let physical = solid_rgb(8, 8, 255, 0, 0);
        let virtual_bg = solid_rgb(8, 8, 0, 255, 0);

        let result = c.apply(&physical, &virtual_bg, 8, 8).expect("ok");

        // Top pixel (y=0) should be all virtual (green)
        let top = result.get_pixel(4, 0).expect("ok");
        assert_eq!(top, [0, 255, 0], "top should be virtual sky");

        // Bottom pixel (y=7) should be physical (red)
        let bot = result.get_pixel(4, 7).expect("ok");
        assert_eq!(bot, [255, 0, 0], "bottom should be physical");
    }

    #[test]
    fn test_apply_floor_extension_bottom_region() {
        let mut c = SetExtensionCompositor::new();
        // Horizon at 50% → bottom half is floor extension
        c.add_element(SetExtensionElement::floor("fl", 0.5).with_blend_mode(EdgeBlendMode::Hard));

        let physical = solid_rgb(8, 8, 255, 0, 0);
        let virtual_bg = solid_rgb(8, 8, 0, 0, 255);

        let result = c.apply(&physical, &virtual_bg, 8, 8).expect("ok");

        // Bottom pixel should be virtual (blue)
        let bot = result.get_pixel(4, 7).expect("ok");
        assert_eq!(bot, [0, 0, 255], "bottom should be virtual floor");
    }

    #[test]
    fn test_apply_wrong_size() {
        let c = SetExtensionCompositor::new();
        let physical = vec![0u8; 10];
        let virtual_bg = solid_rgb(8, 8, 0, 0, 0);
        let result = c.apply(&physical, &virtual_bg, 8, 8);
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_extensions_applied_count() {
        let mut c = SetExtensionCompositor::new();
        c.add_element(SetExtensionElement::sky("s1", 0.4));
        c.add_element(SetExtensionElement::floor("f1", 0.6));

        let physical = solid_rgb(8, 8, 100, 100, 100);
        let virtual_bg = solid_rgb(8, 8, 200, 200, 200);

        let result = c.apply(&physical, &virtual_bg, 8, 8).expect("ok");
        assert_eq!(result.extensions_applied, 2);
    }

    #[test]
    fn test_apply_opacity_zero_no_blend() {
        let mut c = SetExtensionCompositor::new();
        c.add_element(SetExtensionElement::sky("sky", 0.5).with_opacity(0.0));

        let physical = solid_rgb(8, 8, 255, 0, 0);
        let virtual_bg = solid_rgb(8, 8, 0, 255, 0);

        let result = c.apply(&physical, &virtual_bg, 8, 8).expect("ok");
        // Zero opacity → all physical
        let px = result.get_pixel(4, 0).expect("ok");
        assert_eq!(px, [255, 0, 0], "zero opacity: should stay physical");
    }

    #[test]
    fn test_perspective_horizon_level_camera() {
        let params = PerspectiveParams::default(); // tilt = 0°
        let hy = params.horizon_y_normalized(1080);
        assert!(
            (hy - 0.5).abs() < 0.05,
            "level camera horizon at midpoint: {hy}"
        );
    }

    #[test]
    fn test_perspective_horizon_tilted_up() {
        let params = PerspectiveParams {
            tilt_deg: 10.0, // tilted up
            ..PerspectiveParams::default()
        };
        let hy = params.horizon_y_normalized(1080);
        assert!(hy > 0.5, "tilt up should move horizon below center: {hy}");
    }

    #[test]
    fn test_perspective_horizon_tilted_down() {
        let params = PerspectiveParams {
            tilt_deg: -10.0,
            ..PerspectiveParams::default()
        };
        let hy = params.horizon_y_normalized(1080);
        assert!(hy < 0.5, "tilt down should move horizon above center: {hy}");
    }

    #[test]
    fn test_set_extension_element_builder() {
        let el = SetExtensionElement::sky("test", 0.4)
            .with_opacity(0.8)
            .with_blend_mode(EdgeBlendMode::Hard);
        assert_eq!(el.id, "test");
        assert!((el.opacity - 0.8).abs() < 1e-5);
        assert_eq!(el.blend_mode, EdgeBlendMode::Hard);
    }

    #[test]
    fn test_background_plate_covers_all() {
        let mut c = SetExtensionCompositor::new();
        c.add_element(SetExtensionElement {
            id: "bg".to_string(),
            extension_type: ExtensionType::BackgroundPlate,
            blend_mode: EdgeBlendMode::Hard,
            feather_px: 1.0,
            split_y: 0.0,
            split_x: 0.5,
            active: true,
            opacity: 1.0,
        });

        let physical = solid_rgb(4, 4, 255, 0, 0);
        let virtual_bg = solid_rgb(4, 4, 0, 255, 0);

        let result = c.apply(&physical, &virtual_bg, 4, 4).expect("ok");
        // split_y=0 → all below → all virtual (green)
        for y in 0..4 {
            for x in 0..4 {
                let px = result.get_pixel(x, y).expect("ok");
                assert_eq!(px, [0, 255, 0], "all should be virtual at ({x},{y})");
            }
        }
    }
}
