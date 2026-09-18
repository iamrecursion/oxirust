//! Virtual set management for LED volume productions.
//!
//! Provides data structures and utilities for managing virtual sets,
//! including layered scene composition, camera frustum projection,
//! and placeholder rendering.

#![allow(dead_code)]

/// Type of a virtual set layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayerType {
    /// Background scene (e.g. sky, exterior environment).
    Background,
    /// Floor plane / ground.
    Floor,
    /// Ceiling effects and overhead elements.
    CeilingEffect,
    /// Prop or object within the scene.
    Prop,
    /// Lighting rig or light source element.
    LightingRig,
    /// Foreground element (closer to camera than subject).
    ForegroundElement,
}

impl LayerType {
    /// Returns a human-readable label for this layer type.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            LayerType::Background => "Background",
            LayerType::Floor => "Floor",
            LayerType::CeilingEffect => "Ceiling Effect",
            LayerType::Prop => "Prop",
            LayerType::LightingRig => "Lighting Rig",
            LayerType::ForegroundElement => "Foreground Element",
        }
    }
}

/// 3D transform (position, rotation, scale) in world space.
#[derive(Debug, Clone, PartialEq)]
pub struct SetTransform {
    /// Translation (x, y, z) in meters.
    pub position: (f32, f32, f32),
    /// Rotation (pitch, yaw, roll) in degrees.
    pub rotation: (f32, f32, f32),
    /// Scale (x, y, z).
    pub scale: (f32, f32, f32),
}

impl SetTransform {
    /// Returns the identity transform (no translation, rotation, or scale change).
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            position: (0.0, 0.0, 0.0),
            rotation: (0.0, 0.0, 0.0),
            scale: (1.0, 1.0, 1.0),
        }
    }

    /// Returns a transform with only position set.
    #[must_use]
    pub const fn from_position(x: f32, y: f32, z: f32) -> Self {
        Self {
            position: (x, y, z),
            rotation: (0.0, 0.0, 0.0),
            scale: (1.0, 1.0, 1.0),
        }
    }

    /// Returns a transform with uniform scale.
    #[must_use]
    pub const fn with_uniform_scale(mut self, s: f32) -> Self {
        self.scale = (s, s, s);
        self
    }
}

impl Default for SetTransform {
    fn default() -> Self {
        Self::identity()
    }
}

/// A single layer within a virtual set.
#[derive(Debug, Clone)]
pub struct VirtualSetLayer {
    /// Unique identifier for this layer.
    pub id: String,
    /// The type of content this layer represents.
    pub layer_type: LayerType,
    /// Rendering order (higher values render on top).
    pub z_order: i32,
    /// World-space transform for this layer.
    pub transform: SetTransform,
    /// Whether this layer is visible.
    pub visible: bool,
}

impl VirtualSetLayer {
    /// Creates a new virtual set layer.
    #[must_use]
    pub fn new(id: impl Into<String>, layer_type: LayerType, z_order: i32) -> Self {
        Self {
            id: id.into(),
            layer_type,
            z_order,
            transform: SetTransform::identity(),
            visible: true,
        }
    }

    /// Sets the transform for this layer and returns `self`.
    #[must_use]
    pub fn with_transform(mut self, transform: SetTransform) -> Self {
        self.transform = transform;
        self
    }

    /// Sets visibility and returns `self`.
    #[must_use]
    pub const fn with_visible(mut self, visible: bool) -> Self {
        self.visible = visible;
        self
    }
}

/// Camera frustum for computing perspective projections.
#[derive(Debug, Clone)]
pub struct CameraFrustum {
    /// Horizontal field-of-view in degrees.
    pub fov_deg: f32,
    /// Near clip plane distance in meters.
    pub near: f32,
    /// Far clip plane distance in meters.
    pub far: f32,
    /// Aspect ratio (width / height).
    pub aspect: f32,
}

impl CameraFrustum {
    /// Creates a new camera frustum.
    #[must_use]
    pub const fn new(fov_deg: f32, near: f32, far: f32, aspect: f32) -> Self {
        Self {
            fov_deg,
            near,
            far,
            aspect,
        }
    }

    /// Creates a standard 16:9 frustum with 60° FOV.
    #[must_use]
    pub fn standard_hd() -> Self {
        Self::new(60.0, 0.1, 1000.0, 16.0 / 9.0)
    }

    /// Projects a world-space point into normalized screen coordinates (-1..1, -1..1).
    ///
    /// `world` is the point position in world space.
    /// `cam_pos` is the camera position in world space (simplified: no rotation applied).
    ///
    /// Returns `None` if the point is behind the near plane or beyond the far plane,
    /// or if it falls outside the visible frustum.
    #[must_use]
    pub fn project_point(
        &self,
        world: (f32, f32, f32),
        cam_pos: (f32, f32, f32),
    ) -> Option<(f32, f32)> {
        // Compute vector from camera to point in camera space (no rotation: axis-aligned)
        let rel_x = world.0 - cam_pos.0;
        let rel_y = world.1 - cam_pos.1;
        let rel_z = world.2 - cam_pos.2;

        // In camera space, +Z is forward (depth)
        let depth = rel_z;
        if depth < self.near || depth > self.far {
            return None;
        }

        // Perspective divide
        let fov_rad = self.fov_deg.to_radians();
        let tan_half_fov = (fov_rad / 2.0).tan();

        let ndc_x = rel_x / (depth * tan_half_fov * self.aspect);
        let ndc_y = rel_y / (depth * tan_half_fov);

        if !(-1.0..=1.0).contains(&ndc_x) || !(-1.0..=1.0).contains(&ndc_y) {
            return None;
        }

        Some((ndc_x, ndc_y))
    }
}

impl Default for CameraFrustum {
    fn default() -> Self {
        Self::standard_hd()
    }
}

/// A complete virtual set composed of multiple layers.
#[derive(Debug, Clone)]
pub struct VirtualSet {
    /// Name of the virtual set.
    pub name: String,
    /// All layers in this set.
    pub layers: Vec<VirtualSetLayer>,
    /// Ambient light color [R, G, B] in linear 0..1 range.
    pub ambient_light: [f32; 3],
}

impl VirtualSet {
    /// Creates an empty virtual set.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            layers: Vec::new(),
            ambient_light: [0.1, 0.1, 0.1],
        }
    }

    /// Adds a layer to the set.
    pub fn add_layer(&mut self, layer: VirtualSetLayer) {
        self.layers.push(layer);
    }

    /// Removes the layer with the given ID.
    ///
    /// Returns `true` if a layer was removed, `false` if no layer with that ID existed.
    pub fn remove_layer(&mut self, id: &str) -> bool {
        let before = self.layers.len();
        self.layers.retain(|l| l.id != id);
        self.layers.len() < before
    }

    /// Returns layers sorted by `z_order` (ascending, lowest renders first).
    #[must_use]
    pub fn layers_sorted(&self) -> Vec<&VirtualSetLayer> {
        let mut sorted: Vec<&VirtualSetLayer> = self.layers.iter().collect();
        sorted.sort_by_key(|l| l.z_order);
        sorted
    }

    /// Returns only visible layers, sorted by `z_order`.
    #[must_use]
    pub fn visible_layers_sorted(&self) -> Vec<&VirtualSetLayer> {
        let mut sorted: Vec<&VirtualSetLayer> = self.layers.iter().filter(|l| l.visible).collect();
        sorted.sort_by_key(|l| l.z_order);
        sorted
    }

    /// Sets the ambient light color.
    pub fn set_ambient_light(&mut self, r: f32, g: f32, b: f32) {
        self.ambient_light = [r, g, b];
    }
}

/// A simple placeholder preview render of a virtual set.
#[derive(Debug, Clone)]
pub struct SetPreview {
    /// Preview width in pixels.
    pub width: u32,
    /// Preview height in pixels.
    pub height: u32,
    /// RGBA pixel data (one [u8; 4] per pixel).
    pub pixels: Vec<[u8; 4]>,
}

impl SetPreview {
    /// Creates a new blank preview.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            pixels: vec![[0, 0, 0, 255]; (width * height) as usize],
        }
    }

    /// Renders a placeholder preview by tinting each layer type with a distinct color.
    ///
    /// Layers are rendered in z-order. Visible layers paint horizontal bands.
    #[must_use]
    pub fn render_placeholder(set: &VirtualSet) -> Self {
        let width = 320u32;
        let height = 180u32;
        let mut preview = Self::new(width, height);

        // Assign a color per layer type
        let layer_color = |lt: &LayerType| -> [u8; 4] {
            match lt {
                LayerType::Background => [30, 30, 100, 255],
                LayerType::Floor => [60, 40, 20, 255],
                LayerType::CeilingEffect => [80, 80, 120, 255],
                LayerType::Prop => [100, 60, 60, 255],
                LayerType::LightingRig => [200, 200, 100, 255],
                LayerType::ForegroundElement => [60, 120, 60, 255],
            }
        };

        let sorted = set.visible_layers_sorted();
        if sorted.is_empty() {
            return preview;
        }

        // Paint each layer as a horizontal band
        let band_height = (height as usize).max(1) / sorted.len().max(1);
        for (i, layer) in sorted.iter().enumerate() {
            let color = layer_color(&layer.layer_type);
            let y_start = (i * band_height) as u32;
            let y_end = ((i + 1) * band_height).min(height as usize) as u32;
            for y in y_start..y_end {
                for x in 0..width {
                    let idx = (y * width + x) as usize;
                    if idx < preview.pixels.len() {
                        preview.pixels[idx] = color;
                    }
                }
            }
        }

        preview
    }

    /// Returns the pixel at (x, y), or `None` if out of bounds.
    #[must_use]
    pub fn get_pixel(&self, x: u32, y: u32) -> Option<[u8; 4]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        Some(self.pixels[(y * self.width + x) as usize])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_transform_identity() {
        let t = SetTransform::identity();
        assert_eq!(t.position, (0.0, 0.0, 0.0));
        assert_eq!(t.rotation, (0.0, 0.0, 0.0));
        assert_eq!(t.scale, (1.0, 1.0, 1.0));
    }

    #[test]
    fn test_set_transform_from_position() {
        let t = SetTransform::from_position(1.0, 2.0, 3.0);
        assert_eq!(t.position, (1.0, 2.0, 3.0));
        assert_eq!(t.scale, (1.0, 1.0, 1.0));
    }

    #[test]
    fn test_set_transform_with_uniform_scale() {
        let t = SetTransform::identity().with_uniform_scale(2.0);
        assert_eq!(t.scale, (2.0, 2.0, 2.0));
    }

    #[test]
    fn test_virtual_set_layer_creation() {
        let layer = VirtualSetLayer::new("bg", LayerType::Background, 0);
        assert_eq!(layer.id, "bg");
        assert_eq!(layer.layer_type, LayerType::Background);
        assert_eq!(layer.z_order, 0);
        assert!(layer.visible);
    }

    #[test]
    fn test_virtual_set_add_remove_layer() {
        let mut set = VirtualSet::new("Test Set");
        set.add_layer(VirtualSetLayer::new("bg", LayerType::Background, 0));
        set.add_layer(VirtualSetLayer::new("fg", LayerType::ForegroundElement, 10));
        assert_eq!(set.layers.len(), 2);

        let removed = set.remove_layer("bg");
        assert!(removed);
        assert_eq!(set.layers.len(), 1);

        let not_found = set.remove_layer("nonexistent");
        assert!(!not_found);
    }

    #[test]
    fn test_virtual_set_layers_sorted() {
        let mut set = VirtualSet::new("Test");
        set.add_layer(VirtualSetLayer::new("c", LayerType::CeilingEffect, 5));
        set.add_layer(VirtualSetLayer::new("a", LayerType::Background, -1));
        set.add_layer(VirtualSetLayer::new("b", LayerType::Floor, 2));

        let sorted = set.layers_sorted();
        assert_eq!(sorted[0].z_order, -1);
        assert_eq!(sorted[1].z_order, 2);
        assert_eq!(sorted[2].z_order, 5);
    }

    #[test]
    fn test_virtual_set_visible_layers() {
        let mut set = VirtualSet::new("Test");
        set.add_layer(VirtualSetLayer::new("a", LayerType::Background, 0).with_visible(false));
        set.add_layer(VirtualSetLayer::new("b", LayerType::Prop, 1));

        let visible = set.visible_layers_sorted();
        assert_eq!(visible.len(), 1);
        assert_eq!(visible[0].id, "b");
    }

    #[test]
    fn test_camera_frustum_project_point_in_view() {
        let frustum = CameraFrustum::new(90.0, 0.1, 1000.0, 1.0);
        let cam = (0.0, 0.0, 0.0);
        let point = (0.0, 0.0, 10.0); // directly ahead

        let result = frustum.project_point(point, cam);
        assert!(result.is_some());
        let (x, y) = result.expect("should succeed in test");
        assert!((x).abs() < 0.001);
        assert!((y).abs() < 0.001);
    }

    #[test]
    fn test_camera_frustum_project_point_behind() {
        let frustum = CameraFrustum::default();
        let cam = (0.0, 0.0, 10.0);
        let point = (0.0, 0.0, 0.0); // behind camera

        let result = frustum.project_point(point, cam);
        assert!(result.is_none()); // depth = -10, behind near plane
    }

    #[test]
    fn test_camera_frustum_project_point_outside() {
        let frustum = CameraFrustum::new(60.0, 0.1, 1000.0, 1.0);
        let cam = (0.0, 0.0, 0.0);
        let point = (1000.0, 0.0, 1.0); // far off to the side

        let result = frustum.project_point(point, cam);
        assert!(result.is_none());
    }

    #[test]
    fn test_set_preview_render_placeholder() {
        let mut set = VirtualSet::new("Preview Test");
        set.add_layer(VirtualSetLayer::new("bg", LayerType::Background, 0));
        set.add_layer(VirtualSetLayer::new("fg", LayerType::ForegroundElement, 10));

        let preview = SetPreview::render_placeholder(&set);
        assert_eq!(preview.width, 320);
        assert_eq!(preview.height, 180);
        assert_eq!(preview.pixels.len(), 320 * 180);
    }

    #[test]
    fn test_set_preview_empty_set() {
        let set = VirtualSet::new("Empty");
        let preview = SetPreview::render_placeholder(&set);
        assert_eq!(preview.pixels.len(), 320 * 180);
    }

    #[test]
    fn test_set_preview_get_pixel() {
        let preview = SetPreview::new(10, 10);
        assert!(preview.get_pixel(5, 5).is_some());
        assert!(preview.get_pixel(10, 0).is_none()); // out of bounds
    }

    #[test]
    fn test_layer_type_label() {
        assert_eq!(LayerType::Background.label(), "Background");
        assert_eq!(LayerType::Floor.label(), "Floor");
        assert_eq!(LayerType::LightingRig.label(), "Lighting Rig");
    }
}
