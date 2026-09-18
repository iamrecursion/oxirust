//! Render layer management for virtual production compositing.
//!
//! Provides `LayerType`, `RenderLayer`, and `LayerStack` for ordering
//! visual elements in a virtual production pipeline, plus `LodConfig` for
//! level-of-detail rendering at reduced resolutions.

/// Category of a compositing layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerType {
    /// Furthest-back environment plate or generated background.
    Background,
    /// Subject or prop placed in front of the background.
    Foreground,
    /// Graphic or HUD element placed on top of everything else.
    Overlay,
    /// Alpha or luminance matte used to cut out areas.
    Matte,
}

impl LayerType {
    /// Returns `true` if this layer type typically contains transparency
    /// (i.e., it is not expected to fill the full frame).
    #[must_use]
    pub fn is_transparent(&self) -> bool {
        matches!(self, Self::Overlay | Self::Matte)
    }
}

/// Level-of-detail configuration for distance-based resolution scaling.
///
/// Divides the view into bands by camera distance.  Each band maps to a scale
/// factor applied to the render resolution.  The number of `scale_factors`
/// must equal `distance_thresholds.len() + 1`.
#[derive(Debug, Clone)]
pub struct LodConfig {
    /// Distance thresholds in metres, sorted ascending.
    /// Example: `[100.0, 500.0, 2000.0]`
    pub distance_thresholds: Vec<f32>,
    /// Scale factors per band — one more entry than `distance_thresholds`.
    /// Example: `[1.0, 0.5, 0.25, 0.125]`
    pub scale_factors: Vec<f32>,
}

impl LodConfig {
    /// Standard 4-band LOD: full → half → quarter → eighth resolution.
    #[must_use]
    pub fn default_4band() -> Self {
        Self {
            distance_thresholds: vec![100.0, 500.0, 2000.0],
            scale_factors: vec![1.0, 0.5, 0.25, 0.125],
        }
    }

    /// Select the scale factor for a given camera-to-object distance.
    #[must_use]
    pub fn scale_for_distance(&self, distance: f32) -> f32 {
        for (i, &threshold) in self.distance_thresholds.iter().enumerate() {
            if distance < threshold {
                return self.scale_factors.get(i).copied().unwrap_or(1.0);
            }
        }
        // Beyond the last threshold → lowest resolution band.
        self.scale_factors.last().copied().unwrap_or(0.125)
    }
}

impl Default for LodConfig {
    fn default() -> Self {
        Self::default_4band()
    }
}

/// Bilinear upscale a pixel buffer from `(src_w, src_h)` to `(dst_w, dst_h)`.
///
/// Input and output are RGB-interleaved `Vec<u8>`.
fn bilinear_upsample(src: &[u8], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Vec<u8> {
    let mut out = vec![0u8; dst_w as usize * dst_h as usize * 3];

    let scale_x = src_w as f32 / dst_w as f32;
    let scale_y = src_h as f32 / dst_h as f32;

    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let sx = (dx as f32 + 0.5) * scale_x - 0.5;
            let sy = (dy as f32 + 0.5) * scale_y - 0.5;

            let x0 = (sx.floor() as i32).max(0) as u32;
            let y0 = (sy.floor() as i32).max(0) as u32;
            let x1 = (x0 + 1).min(src_w - 1);
            let y1 = (y0 + 1).min(src_h - 1);

            let fx = sx - sx.floor();
            let fy = sy - sy.floor();

            let sample = |px: u32, py: u32, c: usize| -> f32 {
                let i = (py as usize * src_w as usize + px as usize) * 3 + c;
                src.get(i).copied().unwrap_or(0) as f32
            };

            let dst_i = (dy as usize * dst_w as usize + dx as usize) * 3;
            for c in 0..3 {
                let tl = sample(x0, y0, c);
                let tr = sample(x1, y0, c);
                let bl = sample(x0, y1, c);
                let br = sample(x1, y1, c);
                let v = tl * (1.0 - fx) * (1.0 - fy)
                    + tr * fx * (1.0 - fy)
                    + bl * (1.0 - fx) * fy
                    + br * fx * fy;
                out[dst_i + c] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    out
}

/// A single compositing layer with a z-order depth value.
#[derive(Debug, Clone)]
pub struct RenderLayer {
    /// Human-readable name for the layer.
    pub name: String,
    /// Layer category.
    pub layer_type: LayerType,
    /// Z-order depth; lower values render first (behind higher values).
    depth: i32,
    /// Opacity in \[0.0, 1.0\].
    pub opacity: f32,
    /// Optional LOD configuration.  When `Some`, `render_at_lod` will
    /// downscale before rendering and upscale back to target resolution.
    lod: Option<LodConfig>,
}

impl RenderLayer {
    /// Create a new `RenderLayer`.
    #[must_use]
    pub fn new(name: impl Into<String>, layer_type: LayerType, depth: i32) -> Self {
        Self {
            name: name.into(),
            layer_type,
            depth,
            opacity: 1.0,
            lod: None,
        }
    }

    /// Returns the z-order depth of this layer.
    #[must_use]
    pub fn z_order(&self) -> i32 {
        self.depth
    }

    /// Set the opacity.
    #[must_use]
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity.clamp(0.0, 1.0);
        self
    }

    /// Attach a LOD configuration to this layer.
    #[must_use]
    pub fn with_lod(mut self, config: LodConfig) -> Self {
        self.lod = Some(config);
        self
    }

    /// Select the LOD scale factor for the given camera-to-object distance.
    ///
    /// Returns `1.0` (full resolution) when no LOD config is attached.
    #[must_use]
    pub fn select_lod_scale(&self, camera_distance: f32) -> f32 {
        self.lod
            .as_ref()
            .map(|c| c.scale_for_distance(camera_distance))
            .unwrap_or(1.0)
    }

    /// Render this layer at the LOD resolution for the given camera distance,
    /// then bilinear-upsample to the requested `(w, h)` target.
    ///
    /// The returned `Vec<u8>` is an RGB-interleaved pixel buffer of size
    /// `w * h * 3`.  The reduced-resolution render is filled with a
    /// solid mid-grey placeholder; real content would come from the scene
    /// rasteriser passed to a higher-level compositor.
    #[must_use]
    pub fn render_at_lod(&self, camera_distance: f32, w: u32, h: u32) -> Vec<u8> {
        let scale = self.select_lod_scale(camera_distance).clamp(1e-3, 1.0);

        let reduced_w = ((w as f32 * scale).round() as u32).max(1);
        let reduced_h = ((h as f32 * scale).round() as u32).max(1);

        // Placeholder render at reduced resolution (solid mid-grey).
        let reduced_pixels = vec![128u8; reduced_w as usize * reduced_h as usize * 3];

        if reduced_w == w && reduced_h == h {
            return reduced_pixels;
        }

        // Bilinear upsample back to (w, h).
        bilinear_upsample(&reduced_pixels, reduced_w, reduced_h, w, h)
    }
}

/// An ordered stack of `RenderLayer` values.
#[derive(Debug, Default)]
pub struct LayerStack {
    layers: Vec<RenderLayer>,
}

impl LayerStack {
    /// Create an empty stack.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a layer onto the stack and re-sort by ascending z-order.
    pub fn push(&mut self, layer: RenderLayer) {
        self.layers.push(layer);
        self.layers.sort_by_key(RenderLayer::z_order);
    }

    /// Remove the topmost layer (highest z-order) and return it.
    pub fn pop(&mut self) -> Option<RenderLayer> {
        self.layers.pop()
    }

    /// Return the first layer with the given `LayerType`, if any.
    #[must_use]
    pub fn find_by_type(&self, layer_type: LayerType) -> Option<&RenderLayer> {
        self.layers.iter().find(|l| l.layer_type == layer_type)
    }

    /// Number of layers in the stack.
    #[must_use]
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Returns `true` if the stack is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Return all layers ordered by ascending z-order.
    #[must_use]
    pub fn layers(&self) -> &[RenderLayer] {
        &self.layers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_background_not_transparent() {
        assert!(!LayerType::Background.is_transparent());
    }

    #[test]
    fn test_foreground_not_transparent() {
        assert!(!LayerType::Foreground.is_transparent());
    }

    #[test]
    fn test_overlay_is_transparent() {
        assert!(LayerType::Overlay.is_transparent());
    }

    #[test]
    fn test_matte_is_transparent() {
        assert!(LayerType::Matte.is_transparent());
    }

    #[test]
    fn test_render_layer_z_order() {
        let layer = RenderLayer::new("bg", LayerType::Background, 0);
        assert_eq!(layer.z_order(), 0);
        let layer2 = RenderLayer::new("fg", LayerType::Foreground, 10);
        assert_eq!(layer2.z_order(), 10);
    }

    #[test]
    fn test_render_layer_opacity_clamped() {
        let layer = RenderLayer::new("hud", LayerType::Overlay, 100).with_opacity(1.5);
        assert!((layer.opacity - 1.0).abs() < 1e-6);
        let layer2 = RenderLayer::new("hud2", LayerType::Overlay, 100).with_opacity(-0.5);
        assert!((layer2.opacity).abs() < 1e-6);
    }

    #[test]
    fn test_stack_push_orders_by_z() {
        let mut stack = LayerStack::new();
        stack.push(RenderLayer::new("overlay", LayerType::Overlay, 50));
        stack.push(RenderLayer::new("bg", LayerType::Background, 0));
        stack.push(RenderLayer::new("fg", LayerType::Foreground, 20));
        let orders: Vec<i32> = stack
            .layers()
            .iter()
            .map(super::RenderLayer::z_order)
            .collect();
        assert_eq!(orders, vec![0, 20, 50]);
    }

    #[test]
    fn test_stack_pop_returns_topmost() {
        let mut stack = LayerStack::new();
        stack.push(RenderLayer::new("bg", LayerType::Background, 0));
        stack.push(RenderLayer::new("overlay", LayerType::Overlay, 99));
        let popped = stack.pop().expect("should succeed in test");
        assert_eq!(popped.z_order(), 99);
    }

    #[test]
    fn test_stack_find_by_type() {
        let mut stack = LayerStack::new();
        stack.push(RenderLayer::new("bg", LayerType::Background, 0));
        stack.push(RenderLayer::new("matte", LayerType::Matte, 5));
        let found = stack.find_by_type(LayerType::Matte);
        assert!(found.is_some());
        assert_eq!(found.expect("should succeed in test").name, "matte");
    }

    #[test]
    fn test_stack_find_missing_type() {
        let mut stack = LayerStack::new();
        stack.push(RenderLayer::new("bg", LayerType::Background, 0));
        assert!(stack.find_by_type(LayerType::Overlay).is_none());
    }

    #[test]
    fn test_stack_len_and_is_empty() {
        let mut stack = LayerStack::new();
        assert!(stack.is_empty());
        stack.push(RenderLayer::new("bg", LayerType::Background, 0));
        assert_eq!(stack.len(), 1);
        assert!(!stack.is_empty());
    }

    #[test]
    fn test_stack_pop_empty() {
        let mut stack = LayerStack::new();
        assert!(stack.pop().is_none());
    }
}
