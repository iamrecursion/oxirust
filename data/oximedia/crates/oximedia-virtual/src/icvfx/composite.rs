//! In-camera VFX compositor
//!
//! Real-time compositor for blending foreground elements, virtual backgrounds,
//! and LED wall content with depth-based compositing. Supports multi-layer
//! compositing with per-layer opacity, blend modes, and alpha masks.

use super::{
    background::BackgroundRenderer, depth::DepthProcessor, foreground::ForegroundProcessor,
    BlendMode, CompositeLayer,
};
use crate::{Result, VirtualProductionError};
use serde::{Deserialize, Serialize};

/// Compositor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositorConfig {
    /// Output resolution (width, height)
    pub resolution: (usize, usize),
    /// Enable depth compositing
    pub depth_compositing: bool,
    /// Enable motion blur
    pub motion_blur: bool,
    /// Quality level (0.0 - 1.0)
    pub quality: f32,
}

impl Default for CompositorConfig {
    fn default() -> Self {
        Self {
            resolution: (1920, 1080),
            depth_compositing: true,
            motion_blur: false,
            quality: 1.0,
        }
    }
}

/// Composite frame data
#[derive(Debug, Clone)]
pub struct CompositeFrame {
    /// RGB pixel data
    pub pixels: Vec<u8>,
    /// Width in pixels
    pub width: usize,
    /// Height in pixels
    pub height: usize,
    /// Frame timestamp in nanoseconds
    pub timestamp_ns: u64,
}

impl CompositeFrame {
    /// Create new composite frame
    #[must_use]
    pub fn new(width: usize, height: usize, timestamp_ns: u64) -> Self {
        Self {
            pixels: vec![0; width * height * 3],
            width,
            height,
            timestamp_ns,
        }
    }

    /// Get pixel at position
    #[must_use]
    pub fn get_pixel(&self, x: usize, y: usize) -> Option<[u8; 3]> {
        if x >= self.width || y >= self.height {
            return None;
        }

        let idx = (y * self.width + x) * 3;
        Some([self.pixels[idx], self.pixels[idx + 1], self.pixels[idx + 2]])
    }

    /// Set pixel at position
    pub fn set_pixel(&mut self, x: usize, y: usize, rgb: [u8; 3]) {
        if x >= self.width || y >= self.height {
            return;
        }

        let idx = (y * self.width + x) * 3;
        self.pixels[idx] = rgb[0];
        self.pixels[idx + 1] = rgb[1];
        self.pixels[idx + 2] = rgb[2];
    }
}

/// Layer type for multi-layer compositing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerType {
    /// Background plate / LED wall content
    Background,
    /// Foreground talent (typically keyed)
    ForegroundTalent,
    /// CG overlay (text, graphics, particle effects)
    CgOverlay,
    /// Light wrap / reflection pass
    LightWrap,
    /// Matte / holdout layer
    Matte,
}

/// RGBA layer data for multi-layer compositing
#[derive(Debug, Clone)]
pub struct LayerData {
    /// Layer type identifier
    pub layer_type: LayerType,
    /// Layer name
    pub name: String,
    /// RGBA pixel data (4 channels, each 0.0-1.0)
    pub pixels_rgba: Vec<f32>,
    /// Width
    pub width: usize,
    /// Height
    pub height: usize,
    /// Per-layer opacity multiplier
    pub opacity: f32,
    /// Blend mode for this layer
    pub blend_mode: BlendMode,
    /// Whether this layer is enabled
    pub enabled: bool,
    /// Z-order (lower = further back)
    pub z_order: i32,
}

impl LayerData {
    /// Create a new layer from RGBA f32 data.
    pub fn new(
        name: &str,
        layer_type: LayerType,
        pixels_rgba: Vec<f32>,
        width: usize,
        height: usize,
    ) -> Result<Self> {
        let expected = width * height * 4;
        if pixels_rgba.len() != expected {
            return Err(VirtualProductionError::Compositing(format!(
                "Layer '{}' RGBA size mismatch: expected {expected}, got {}",
                name,
                pixels_rgba.len()
            )));
        }
        Ok(Self {
            layer_type,
            name: name.to_string(),
            pixels_rgba,
            width,
            height,
            opacity: 1.0,
            blend_mode: BlendMode::Normal,
            enabled: true,
            z_order: 0,
        })
    }

    /// Create a layer from RGB u8 data with a uniform alpha.
    pub fn from_rgb_u8(
        name: &str,
        layer_type: LayerType,
        rgb: &[u8],
        width: usize,
        height: usize,
        alpha: f32,
    ) -> Result<Self> {
        let expected_rgb = width * height * 3;
        if rgb.len() != expected_rgb {
            return Err(VirtualProductionError::Compositing(format!(
                "Layer '{}' RGB size mismatch: expected {expected_rgb}, got {}",
                name,
                rgb.len()
            )));
        }
        let mut rgba = Vec::with_capacity(width * height * 4);
        for chunk in rgb.chunks_exact(3) {
            rgba.push(f32::from(chunk[0]) / 255.0);
            rgba.push(f32::from(chunk[1]) / 255.0);
            rgba.push(f32::from(chunk[2]) / 255.0);
            rgba.push(alpha);
        }
        Self::new(name, layer_type, rgba, width, height)
    }

    /// Create a layer from RGB u8 data + separate alpha channel.
    pub fn from_rgb_u8_with_alpha(
        name: &str,
        layer_type: LayerType,
        rgb: &[u8],
        alpha_channel: &[f32],
        width: usize,
        height: usize,
    ) -> Result<Self> {
        let expected_rgb = width * height * 3;
        let expected_alpha = width * height;
        if rgb.len() != expected_rgb {
            return Err(VirtualProductionError::Compositing(format!(
                "Layer '{}' RGB size mismatch: expected {expected_rgb}, got {}",
                name,
                rgb.len()
            )));
        }
        if alpha_channel.len() != expected_alpha {
            return Err(VirtualProductionError::Compositing(format!(
                "Layer '{}' alpha size mismatch: expected {expected_alpha}, got {}",
                name,
                alpha_channel.len()
            )));
        }
        let mut rgba = Vec::with_capacity(width * height * 4);
        for (i, chunk) in rgb.chunks_exact(3).enumerate() {
            rgba.push(f32::from(chunk[0]) / 255.0);
            rgba.push(f32::from(chunk[1]) / 255.0);
            rgba.push(f32::from(chunk[2]) / 255.0);
            rgba.push(alpha_channel[i]);
        }
        Self::new(name, layer_type, rgba, width, height)
    }

    /// Set the blend mode for this layer.
    #[must_use]
    pub fn with_blend_mode(mut self, mode: BlendMode) -> Self {
        self.blend_mode = mode;
        self
    }

    /// Set the opacity for this layer.
    #[must_use]
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity.clamp(0.0, 1.0);
        self
    }

    /// Set the z-order for this layer.
    #[must_use]
    pub fn with_z_order(mut self, z_order: i32) -> Self {
        self.z_order = z_order;
        self
    }

    /// Get RGBA pixel at position as [r, g, b, a].
    #[must_use]
    pub fn get_rgba(&self, x: usize, y: usize) -> Option<[f32; 4]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let idx = (y * self.width + x) * 4;
        Some([
            self.pixels_rgba[idx],
            self.pixels_rgba[idx + 1],
            self.pixels_rgba[idx + 2],
            self.pixels_rgba[idx + 3],
        ])
    }
}

/// ICVFX compositor
pub struct IcvfxCompositor {
    config: CompositorConfig,
    #[allow(dead_code)]
    foreground_processor: ForegroundProcessor,
    #[allow(dead_code)]
    background_renderer: BackgroundRenderer,
    depth_processor: Option<DepthProcessor>,
    layers: Vec<CompositeLayer>,
}

impl IcvfxCompositor {
    /// Create new compositor
    pub fn new(config: CompositorConfig) -> Result<Self> {
        let foreground_processor = ForegroundProcessor::new()?;
        let background_renderer = BackgroundRenderer::new()?;
        let depth_processor = if config.depth_compositing {
            Some(DepthProcessor::new()?)
        } else {
            None
        };

        Ok(Self {
            config,
            foreground_processor,
            background_renderer,
            depth_processor,
            layers: Vec::new(),
        })
    }

    /// Add composite layer
    pub fn add_layer(&mut self, layer: CompositeLayer) {
        self.layers.push(layer);
    }

    /// Composite frame (legacy 2-layer interface)
    pub fn composite(
        &mut self,
        foreground: &[u8],
        background: &[u8],
        depth: Option<&[f32]>,
        timestamp_ns: u64,
    ) -> Result<CompositeFrame> {
        let (width, height) = self.config.resolution;

        // Validate input sizes
        let expected_size = width * height * 3;
        if foreground.len() != expected_size || background.len() != expected_size {
            return Err(VirtualProductionError::Compositing(format!(
                "Invalid input size: expected {}, got foreground: {}, background: {}",
                expected_size,
                foreground.len(),
                background.len()
            )));
        }

        let mut output = CompositeFrame::new(width, height, timestamp_ns);

        // Process depth if available
        let depth_map =
            if let (Some(depth_data), Some(processor)) = (depth, &mut self.depth_processor) {
                Some(processor.process(depth_data, width, height)?)
            } else {
                None
            };

        // Composite pixel by pixel
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 3;

                // Get foreground and background pixels
                let fg = [
                    f32::from(foreground[idx]) / 255.0,
                    f32::from(foreground[idx + 1]) / 255.0,
                    f32::from(foreground[idx + 2]) / 255.0,
                ];

                let bg = [
                    f32::from(background[idx]) / 255.0,
                    f32::from(background[idx + 1]) / 255.0,
                    f32::from(background[idx + 2]) / 255.0,
                ];

                // Determine blend alpha from depth
                let alpha = if let Some(ref depth_map) = depth_map {
                    depth_map[y * width + x]
                } else {
                    0.5 // Default blend
                };

                // Blend foreground and background
                let blended = BlendMode::Normal.blend(bg, fg, alpha);

                // Convert back to u8
                let result = [
                    (blended[0] * 255.0) as u8,
                    (blended[1] * 255.0) as u8,
                    (blended[2] * 255.0) as u8,
                ];

                output.set_pixel(x, y, result);
            }
        }

        Ok(output)
    }

    /// Multi-layer compositing: composites an arbitrary stack of `LayerData`
    /// in z-order, applying per-layer blend modes, opacity, and alpha.
    ///
    /// Layers are sorted by z_order (lowest first = furthest back).
    /// Disabled layers are skipped. All layers must match the compositor
    /// resolution.
    pub fn composite_multi_layer(
        &self,
        layers: &[LayerData],
        timestamp_ns: u64,
    ) -> Result<CompositeFrame> {
        let (width, height) = self.config.resolution;

        // Filter and sort layers
        let mut active_layers: Vec<&LayerData> = layers
            .iter()
            .filter(|l| l.enabled && l.opacity > 0.0)
            .collect();
        active_layers.sort_by_key(|l| l.z_order);

        // Validate dimensions
        for layer in &active_layers {
            if layer.width != width || layer.height != height {
                return Err(VirtualProductionError::Compositing(format!(
                    "Layer '{}' resolution {}x{} doesn't match compositor {}x{}",
                    layer.name, layer.width, layer.height, width, height
                )));
            }
        }

        let pixel_count = width * height;
        // Accumulate in f32 RGBA
        let mut canvas_r = vec![0.0_f32; pixel_count];
        let mut canvas_g = vec![0.0_f32; pixel_count];
        let mut canvas_b = vec![0.0_f32; pixel_count];
        let mut canvas_a = vec![0.0_f32; pixel_count];

        for layer in &active_layers {
            let layer_opacity = layer.opacity;

            for i in 0..pixel_count {
                let idx = i * 4;
                let lr = layer.pixels_rgba[idx];
                let lg = layer.pixels_rgba[idx + 1];
                let lb = layer.pixels_rgba[idx + 2];
                let la = layer.pixels_rgba[idx + 3] * layer_opacity;

                if la < 1e-6 {
                    continue;
                }

                let base = [canvas_r[i], canvas_g[i], canvas_b[i]];
                let blend_color = [lr, lg, lb];

                let blended = layer.blend_mode.blend(base, blend_color, la);

                // Porter-Duff "over" compositing for alpha
                let out_a = la + canvas_a[i] * (1.0 - la);

                if out_a > 1e-6 {
                    canvas_r[i] =
                        (blended[0] * la + canvas_r[i] * canvas_a[i] * (1.0 - la)) / out_a;
                    canvas_g[i] =
                        (blended[1] * la + canvas_g[i] * canvas_a[i] * (1.0 - la)) / out_a;
                    canvas_b[i] =
                        (blended[2] * la + canvas_b[i] * canvas_a[i] * (1.0 - la)) / out_a;
                }
                canvas_a[i] = out_a.min(1.0);
            }
        }

        // Convert to u8 output
        let mut output = CompositeFrame::new(width, height, timestamp_ns);
        for i in 0..pixel_count {
            let idx = i * 3;
            output.pixels[idx] = (canvas_r[i].clamp(0.0, 1.0) * 255.0) as u8;
            output.pixels[idx + 1] = (canvas_g[i].clamp(0.0, 1.0) * 255.0) as u8;
            output.pixels[idx + 2] = (canvas_b[i].clamp(0.0, 1.0) * 255.0) as u8;
        }

        Ok(output)
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> &CompositorConfig {
        &self.config
    }

    /// Get number of layers
    #[must_use]
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Clear all layers
    pub fn clear_layers(&mut self) {
        self.layers.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_composite_frame() {
        let frame = CompositeFrame::new(100, 100, 0);
        assert_eq!(frame.width, 100);
        assert_eq!(frame.height, 100);
        assert_eq!(frame.pixels.len(), 100 * 100 * 3);
    }

    #[test]
    fn test_composite_frame_pixel() {
        let mut frame = CompositeFrame::new(100, 100, 0);
        frame.set_pixel(50, 50, [255, 128, 64]);

        let pixel = frame.get_pixel(50, 50);
        assert_eq!(pixel, Some([255, 128, 64]));
    }

    #[test]
    fn test_compositor_creation() {
        let config = CompositorConfig::default();
        let compositor = IcvfxCompositor::new(config);
        assert!(compositor.is_ok());
    }

    #[test]
    fn test_compositor_layers() {
        let config = CompositorConfig::default();
        let mut compositor = IcvfxCompositor::new(config).expect("should succeed in test");

        compositor.add_layer(CompositeLayer::new("Layer 1".to_string()));
        compositor.add_layer(CompositeLayer::new("Layer 2".to_string()));

        assert_eq!(compositor.layer_count(), 2);

        compositor.clear_layers();
        assert_eq!(compositor.layer_count(), 0);
    }

    #[test]
    fn test_composite_simple() {
        let config = CompositorConfig {
            resolution: (10, 10),
            depth_compositing: false,
            motion_blur: false,
            quality: 1.0,
        };

        let mut compositor = IcvfxCompositor::new(config).expect("should succeed in test");

        let foreground = vec![255u8; 10 * 10 * 3];
        let background = vec![0u8; 10 * 10 * 3];

        let result = compositor.composite(&foreground, &background, None, 0);
        assert!(result.is_ok());
    }

    // --- Multi-layer compositing tests ---

    fn make_solid_layer(
        name: &str,
        layer_type: LayerType,
        r: f32,
        g: f32,
        b: f32,
        a: f32,
        w: usize,
        h: usize,
        z_order: i32,
    ) -> LayerData {
        let pixel_count = w * h;
        let mut rgba = Vec::with_capacity(pixel_count * 4);
        for _ in 0..pixel_count {
            rgba.push(r);
            rgba.push(g);
            rgba.push(b);
            rgba.push(a);
        }
        LayerData::new(name, layer_type, rgba, w, h)
            .expect("should succeed in test")
            .with_z_order(z_order)
    }

    #[test]
    fn test_multi_layer_single_opaque() {
        let config = CompositorConfig {
            resolution: (4, 4),
            depth_compositing: false,
            motion_blur: false,
            quality: 1.0,
        };
        let compositor = IcvfxCompositor::new(config).expect("should succeed in test");

        let bg = make_solid_layer("bg", LayerType::Background, 1.0, 0.0, 0.0, 1.0, 4, 4, 0);
        let result = compositor
            .composite_multi_layer(&[bg], 0)
            .expect("should succeed in test");

        let pixel = result.get_pixel(0, 0).expect("should succeed in test");
        assert_eq!(pixel, [255, 0, 0], "solid red background");
    }

    #[test]
    fn test_multi_layer_two_layers_over() {
        let config = CompositorConfig {
            resolution: (4, 4),
            depth_compositing: false,
            motion_blur: false,
            quality: 1.0,
        };
        let compositor = IcvfxCompositor::new(config).expect("should succeed in test");

        // Red background (z=0, fully opaque)
        let bg = make_solid_layer("bg", LayerType::Background, 1.0, 0.0, 0.0, 1.0, 4, 4, 0);
        // Green foreground at 50% alpha (z=1)
        let fg = make_solid_layer(
            "fg",
            LayerType::ForegroundTalent,
            0.0,
            1.0,
            0.0,
            0.5,
            4,
            4,
            1,
        );

        let result = compositor
            .composite_multi_layer(&[fg, bg], 0)
            .expect("should succeed in test");

        let pixel = result.get_pixel(2, 2).expect("should succeed in test");
        // Porter-Duff "over": green (50% alpha) composited over red (100% alpha).
        // Both channels should be present in the output.
        assert!(pixel[0] > 0, "red should bleed through: {pixel:?}");
        assert!(pixel[1] > 0, "green should contribute: {pixel:?}");
    }

    #[test]
    fn test_multi_layer_three_layers() {
        let config = CompositorConfig {
            resolution: (4, 4),
            depth_compositing: false,
            motion_blur: false,
            quality: 1.0,
        };
        let compositor = IcvfxCompositor::new(config).expect("should succeed in test");

        let bg = make_solid_layer("bg", LayerType::Background, 0.0, 0.0, 1.0, 1.0, 4, 4, 0);
        let fg = make_solid_layer(
            "talent",
            LayerType::ForegroundTalent,
            0.0,
            1.0,
            0.0,
            0.5,
            4,
            4,
            1,
        );
        let overlay = make_solid_layer("cg", LayerType::CgOverlay, 1.0, 0.0, 0.0, 0.3, 4, 4, 2);

        let result = compositor
            .composite_multi_layer(&[bg, fg, overlay], 0)
            .expect("should succeed in test");

        let pixel = result.get_pixel(0, 0).expect("should succeed in test");
        // All three channels should have some presence
        assert!(pixel[0] > 0, "red overlay should contribute");
        assert!(pixel[1] > 0, "green talent should contribute");
        assert!(pixel[2] > 0, "blue background should contribute");
    }

    #[test]
    fn test_multi_layer_disabled_layer_skipped() {
        let config = CompositorConfig {
            resolution: (4, 4),
            depth_compositing: false,
            motion_blur: false,
            quality: 1.0,
        };
        let compositor = IcvfxCompositor::new(config).expect("should succeed in test");

        let bg = make_solid_layer("bg", LayerType::Background, 1.0, 0.0, 0.0, 1.0, 4, 4, 0);
        let mut fg = make_solid_layer(
            "fg",
            LayerType::ForegroundTalent,
            0.0,
            1.0,
            0.0,
            1.0,
            4,
            4,
            1,
        );
        fg.enabled = false; // disabled

        let result = compositor
            .composite_multi_layer(&[bg, fg], 0)
            .expect("should succeed in test");

        let pixel = result.get_pixel(0, 0).expect("should succeed in test");
        assert_eq!(
            pixel,
            [255, 0, 0],
            "disabled green layer should not contribute"
        );
    }

    #[test]
    fn test_multi_layer_z_order_sorting() {
        let config = CompositorConfig {
            resolution: (4, 4),
            depth_compositing: false,
            motion_blur: false,
            quality: 1.0,
        };
        let compositor = IcvfxCompositor::new(config).expect("should succeed in test");

        // Provide layers in wrong order -- compositor should sort by z_order
        let top = make_solid_layer("top", LayerType::CgOverlay, 0.0, 0.0, 1.0, 1.0, 4, 4, 10);
        let bottom = make_solid_layer("bottom", LayerType::Background, 1.0, 0.0, 0.0, 1.0, 4, 4, 0);

        let result = compositor
            .composite_multi_layer(&[top, bottom], 0)
            .expect("should succeed in test");

        let pixel = result.get_pixel(0, 0).expect("should succeed in test");
        // Blue top layer (z=10) should completely cover red bottom (z=0)
        assert_eq!(pixel, [0, 0, 255], "top layer should dominate");
    }

    #[test]
    fn test_multi_layer_opacity_modifier() {
        let config = CompositorConfig {
            resolution: (4, 4),
            depth_compositing: false,
            motion_blur: false,
            quality: 1.0,
        };
        let compositor = IcvfxCompositor::new(config).expect("should succeed in test");

        let bg = make_solid_layer("bg", LayerType::Background, 1.0, 0.0, 0.0, 1.0, 4, 4, 0);
        let fg = make_solid_layer(
            "fg",
            LayerType::ForegroundTalent,
            0.0,
            1.0,
            0.0,
            1.0,
            4,
            4,
            1,
        )
        .with_opacity(0.0); // zero opacity => invisible

        let result = compositor
            .composite_multi_layer(&[bg, fg], 0)
            .expect("should succeed in test");

        let pixel = result.get_pixel(0, 0).expect("should succeed in test");
        assert_eq!(pixel, [255, 0, 0], "zero-opacity layer should be invisible");
    }

    #[test]
    fn test_multi_layer_resolution_mismatch_error() {
        let config = CompositorConfig {
            resolution: (4, 4),
            depth_compositing: false,
            motion_blur: false,
            quality: 1.0,
        };
        let compositor = IcvfxCompositor::new(config).expect("should succeed in test");

        let wrong = make_solid_layer("wrong", LayerType::Background, 1.0, 0.0, 0.0, 1.0, 8, 8, 0);

        let result = compositor.composite_multi_layer(&[wrong], 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_multi_layer_additive_blend() {
        let config = CompositorConfig {
            resolution: (4, 4),
            depth_compositing: false,
            motion_blur: false,
            quality: 1.0,
        };
        let compositor = IcvfxCompositor::new(config).expect("should succeed in test");

        let bg = make_solid_layer("bg", LayerType::Background, 0.5, 0.0, 0.0, 1.0, 4, 4, 0);
        let glow = make_solid_layer("glow", LayerType::CgOverlay, 0.3, 0.3, 0.0, 1.0, 4, 4, 1)
            .with_blend_mode(BlendMode::Add);

        let result = compositor
            .composite_multi_layer(&[bg, glow], 0)
            .expect("should succeed in test");

        let pixel = result.get_pixel(0, 0).expect("should succeed in test");
        // Additive: 0.5+0.3 = 0.8 for red, 0.0+0.3 = 0.3 for green
        assert!(pixel[0] > 180, "additive red: {}", pixel[0]);
        assert!(pixel[1] > 50, "additive green: {}", pixel[1]);
    }

    #[test]
    fn test_multi_layer_empty_layers() {
        let config = CompositorConfig {
            resolution: (4, 4),
            depth_compositing: false,
            motion_blur: false,
            quality: 1.0,
        };
        let compositor = IcvfxCompositor::new(config).expect("should succeed in test");

        let result = compositor
            .composite_multi_layer(&[], 0)
            .expect("should succeed in test");

        let pixel = result.get_pixel(0, 0).expect("should succeed in test");
        assert_eq!(pixel, [0, 0, 0], "empty layers should produce black");
    }

    #[test]
    fn test_layer_data_from_rgb_u8() {
        let rgb = vec![255u8, 0, 0, 0, 255, 0]; // 2 pixels
        let layer = LayerData::from_rgb_u8("test", LayerType::Background, &rgb, 2, 1, 0.8)
            .expect("should succeed in test");
        let px = layer.get_rgba(0, 0).expect("should succeed in test");
        assert!((px[0] - 1.0).abs() < 1e-3);
        assert!((px[3] - 0.8).abs() < 1e-3);
    }

    #[test]
    fn test_layer_data_from_rgb_u8_with_alpha() {
        let rgb = vec![128u8, 128, 128];
        let alpha = vec![0.5_f32];
        let layer = LayerData::from_rgb_u8_with_alpha("test", LayerType::Matte, &rgb, &alpha, 1, 1)
            .expect("should succeed in test");
        let px = layer.get_rgba(0, 0).expect("should succeed in test");
        assert!((px[3] - 0.5).abs() < 1e-3);
    }

    #[test]
    fn test_multi_layer_multiply_blend() {
        let config = CompositorConfig {
            resolution: (4, 4),
            depth_compositing: false,
            motion_blur: false,
            quality: 1.0,
        };
        let compositor = IcvfxCompositor::new(config).expect("should succeed in test");

        let bg = make_solid_layer("bg", LayerType::Background, 0.8, 0.6, 0.4, 1.0, 4, 4, 0);
        let mul = make_solid_layer("mul", LayerType::CgOverlay, 0.5, 0.5, 0.5, 1.0, 4, 4, 1)
            .with_blend_mode(BlendMode::Multiply);

        let result = compositor
            .composite_multi_layer(&[bg, mul], 0)
            .expect("should succeed in test");

        let pixel = result.get_pixel(0, 0).expect("should succeed in test");
        // Multiply: 0.8*0.5=0.4 → 102, 0.6*0.5=0.3 → 76, 0.4*0.5=0.2 → 51
        assert!(
            (pixel[0] as i32 - 102).abs() < 5,
            "multiply red: {}",
            pixel[0]
        );
    }
}
