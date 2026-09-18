//! Virtual set and chroma-key background graphics.
//!
//! Provides virtual set management for broadcast productions including
//! background plates, perspective-correct element placement, and
//! depth-of-field simulation for realistic compositing.

#![allow(dead_code)]

/// A 2D point in virtual set space
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2D {
    /// X coordinate
    pub x: f64,
    /// Y coordinate
    pub y: f64,
}

impl Point2D {
    /// Creates a new 2D point
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Returns the origin point (0, 0)
    pub fn origin() -> Self {
        Self { x: 0.0, y: 0.0 }
    }

    /// Computes the distance to another point
    pub fn distance_to(&self, other: &Point2D) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

/// A 3D point in virtual set space
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point3D {
    /// X coordinate
    pub x: f64,
    /// Y coordinate
    pub y: f64,
    /// Z coordinate (depth)
    pub z: f64,
}

impl Point3D {
    /// Creates a new 3D point
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Returns the origin point (0, 0, 0)
    pub fn origin() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }
}

/// Virtual set layer type
#[derive(Debug, Clone, PartialEq)]
pub enum LayerType {
    /// Static background image
    Background,
    /// Foreground element (in front of talent)
    Foreground,
    /// Midground element
    Midground,
    /// Desk or surface element
    Desk,
    /// Screen element (for virtual monitors/displays)
    Screen,
    /// Lighting element
    Lighting,
    /// Shadow element
    Shadow,
}

impl LayerType {
    /// Returns the z-order priority (lower = further back)
    pub fn z_order(&self) -> i32 {
        match self {
            LayerType::Background => 0,
            LayerType::Lighting => 1,
            LayerType::Shadow => 2,
            LayerType::Midground => 3,
            LayerType::Desk => 4,
            LayerType::Screen => 5,
            LayerType::Foreground => 6,
        }
    }
}

/// A layer in the virtual set
#[derive(Debug, Clone)]
pub struct VirtualSetLayer {
    /// Unique identifier
    pub id: String,
    /// Layer type
    pub layer_type: LayerType,
    /// Source image path or identifier
    pub source: String,
    /// Position in 3D space
    pub position: Point3D,
    /// Scale factor (1.0 = original size)
    pub scale: f64,
    /// Opacity (0.0 to 1.0)
    pub opacity: f64,
    /// Whether this layer is visible
    pub visible: bool,
    /// Blend mode name
    pub blend_mode: String,
}

impl VirtualSetLayer {
    /// Creates a new virtual set layer
    pub fn new(id: impl Into<String>, layer_type: LayerType, source: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            layer_type,
            source: source.into(),
            position: Point3D::origin(),
            scale: 1.0,
            opacity: 1.0,
            visible: true,
            blend_mode: "normal".into(),
        }
    }

    /// Sets the position of the layer
    pub fn with_position(mut self, position: Point3D) -> Self {
        self.position = position;
        self
    }

    /// Sets the scale of the layer
    pub fn with_scale(mut self, scale: f64) -> Self {
        self.scale = scale;
        self
    }

    /// Sets the opacity of the layer
    pub fn with_opacity(mut self, opacity: f64) -> Self {
        self.opacity = opacity.clamp(0.0, 1.0);
        self
    }
}

/// Camera perspective settings
#[derive(Debug, Clone)]
pub struct CameraPerspective {
    /// Field of view in degrees
    pub fov: f64,
    /// Camera position
    pub position: Point3D,
    /// Camera target (look-at point)
    pub target: Point3D,
    /// Near clipping distance
    pub near_clip: f64,
    /// Far clipping distance
    pub far_clip: f64,
    /// Aperture for depth-of-field simulation
    pub aperture: f64,
    /// Focus distance
    pub focus_distance: f64,
}

impl Default for CameraPerspective {
    fn default() -> Self {
        Self {
            fov: 45.0,
            position: Point3D::new(0.0, 0.0, -5.0),
            target: Point3D::origin(),
            near_clip: 0.1,
            far_clip: 1000.0,
            aperture: 2.8,
            focus_distance: 5.0,
        }
    }
}

impl CameraPerspective {
    /// Creates a new camera perspective with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns whether depth-of-field is enabled (aperture < 22)
    pub fn dof_enabled(&self) -> bool {
        self.aperture < 22.0
    }

    /// Returns the depth-of-field blur amount at a given distance
    pub fn blur_at_distance(&self, distance: f64) -> f64 {
        let diff = (distance - self.focus_distance).abs();
        diff / (self.aperture * 0.1)
    }
}

/// Chroma key settings for background removal
#[derive(Debug, Clone)]
pub struct ChromaKeySettings {
    /// Key color (RGBA)
    pub key_color: [u8; 4],
    /// Similarity threshold (0.0 to 1.0)
    pub similarity: f64,
    /// Smoothness of the key edge
    pub smoothness: f64,
    /// Spill suppression amount
    pub spill_suppression: f64,
    /// Whether to use advanced edge refinement
    pub edge_refinement: bool,
}

impl Default for ChromaKeySettings {
    fn default() -> Self {
        Self {
            key_color: [0, 255, 0, 255], // Green screen
            similarity: 0.4,
            smoothness: 0.1,
            spill_suppression: 0.3,
            edge_refinement: true,
        }
    }
}

impl ChromaKeySettings {
    /// Creates green screen key settings
    pub fn green_screen() -> Self {
        Self::default()
    }

    /// Creates blue screen key settings
    pub fn blue_screen() -> Self {
        Self {
            key_color: [0, 0, 255, 255],
            ..Self::default()
        }
    }

    /// Returns whether the given color matches the key color within similarity threshold
    pub fn matches(&self, color: [u8; 4]) -> bool {
        let dr = (color[0] as f64 - self.key_color[0] as f64) / 255.0;
        let dg = (color[1] as f64 - self.key_color[1] as f64) / 255.0;
        let db = (color[2] as f64 - self.key_color[2] as f64) / 255.0;
        let distance = (dr * dr + dg * dg + db * db).sqrt();
        distance < self.similarity
    }
}

/// Virtual set management
#[derive(Debug)]
pub struct VirtualSet {
    /// Set name/identifier
    pub name: String,
    /// Layers sorted by z-order
    layers: Vec<VirtualSetLayer>,
    /// Camera perspective
    pub camera: CameraPerspective,
    /// Chroma key settings
    pub chroma_key: ChromaKeySettings,
    /// Overall output width
    pub width: u32,
    /// Overall output height
    pub height: u32,
    /// Whether tracking data is applied
    pub tracking_enabled: bool,
}

impl VirtualSet {
    /// Creates a new virtual set
    pub fn new(name: impl Into<String>, width: u32, height: u32) -> Self {
        Self {
            name: name.into(),
            layers: Vec::new(),
            camera: CameraPerspective::default(),
            chroma_key: ChromaKeySettings::green_screen(),
            width,
            height,
            tracking_enabled: false,
        }
    }

    /// Adds a layer to the virtual set
    pub fn add_layer(&mut self, layer: VirtualSetLayer) {
        self.layers.push(layer);
        self.layers.sort_by_key(|l| l.layer_type.z_order());
    }

    /// Removes a layer by ID, returns true if found and removed
    pub fn remove_layer(&mut self, id: &str) -> bool {
        let before = self.layers.len();
        self.layers.retain(|l| l.id != id);
        self.layers.len() < before
    }

    /// Returns the number of layers
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Returns visible layers in render order
    pub fn visible_layers(&self) -> Vec<&VirtualSetLayer> {
        self.layers.iter().filter(|l| l.visible).collect()
    }

    /// Finds a layer by ID
    pub fn find_layer(&self, id: &str) -> Option<&VirtualSetLayer> {
        self.layers.iter().find(|l| l.id == id)
    }

    /// Returns the aspect ratio
    pub fn aspect_ratio(&self) -> f64 {
        self.width as f64 / self.height as f64
    }

    /// Enables camera tracking
    pub fn enable_tracking(&mut self) {
        self.tracking_enabled = true;
    }

    /// Updates camera position from tracking data
    pub fn apply_tracking(&mut self, position: Point3D, target: Point3D) {
        if self.tracking_enabled {
            self.camera.position = position;
            self.camera.target = target;
        }
    }
}

/// Virtual screen element for displaying content within a virtual set
#[derive(Debug, Clone)]
pub struct VirtualScreen {
    /// Screen identifier
    pub id: String,
    /// 2D position on the canvas
    pub position: Point2D,
    /// Screen width in pixels
    pub width: u32,
    /// Screen height in pixels
    pub height: u32,
    /// Corner points for perspective correction
    pub corners: [Point2D; 4],
    /// Current content source
    pub content_source: Option<String>,
    /// Screen brightness multiplier
    pub brightness: f64,
}

impl VirtualScreen {
    /// Creates a new virtual screen
    pub fn new(id: impl Into<String>, position: Point2D, width: u32, height: u32) -> Self {
        let x = position.x;
        let y = position.y;
        let w = width as f64;
        let h = height as f64;
        Self {
            id: id.into(),
            position,
            width,
            height,
            corners: [
                Point2D::new(x, y),
                Point2D::new(x + w, y),
                Point2D::new(x + w, y + h),
                Point2D::new(x, y + h),
            ],
            content_source: None,
            brightness: 1.0,
        }
    }

    /// Sets the content source
    pub fn set_content(&mut self, source: impl Into<String>) {
        self.content_source = Some(source.into());
    }

    /// Clears the content source
    pub fn clear_content(&mut self) {
        self.content_source = None;
    }

    /// Returns whether the screen has content
    pub fn has_content(&self) -> bool {
        self.content_source.is_some()
    }

    /// Applies a perspective warp to the corner points
    pub fn apply_perspective_warp(&mut self, offsets: [Point2D; 4]) {
        for (corner, offset) in self.corners.iter_mut().zip(offsets.iter()) {
            corner.x += offset.x;
            corner.y += offset.y;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point2d_new() {
        let p = Point2D::new(3.0, 4.0);
        assert!((p.x - 3.0).abs() < 1e-10);
        assert!((p.y - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_point2d_distance() {
        let a = Point2D::new(0.0, 0.0);
        let b = Point2D::new(3.0, 4.0);
        assert!((a.distance_to(&b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_point2d_origin() {
        let p = Point2D::origin();
        assert!((p.x).abs() < 1e-10);
        assert!((p.y).abs() < 1e-10);
    }

    #[test]
    fn test_point3d_new() {
        let p = Point3D::new(1.0, 2.0, 3.0);
        assert!((p.x - 1.0).abs() < 1e-10);
        assert!((p.y - 2.0).abs() < 1e-10);
        assert!((p.z - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_layer_type_z_order() {
        assert!(LayerType::Background.z_order() < LayerType::Midground.z_order());
        assert!(LayerType::Midground.z_order() < LayerType::Foreground.z_order());
        assert!(LayerType::Screen.z_order() > LayerType::Desk.z_order());
    }

    #[test]
    fn test_virtual_set_layer_new() {
        let layer = VirtualSetLayer::new("bg", LayerType::Background, "bg.png");
        assert_eq!(layer.id, "bg");
        assert!(layer.visible);
        assert!((layer.opacity - 1.0).abs() < 1e-10);
        assert!((layer.scale - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_virtual_set_layer_builder() {
        let layer = VirtualSetLayer::new("fg", LayerType::Foreground, "fg.png")
            .with_scale(2.0)
            .with_opacity(0.5);
        assert!((layer.scale - 2.0).abs() < 1e-10);
        assert!((layer.opacity - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_virtual_set_layer_opacity_clamp() {
        let layer =
            VirtualSetLayer::new("test", LayerType::Midground, "test.png").with_opacity(1.5);
        assert!((layer.opacity - 1.0).abs() < 1e-10);
        let layer2 =
            VirtualSetLayer::new("test2", LayerType::Midground, "test2.png").with_opacity(-0.5);
        assert!((layer2.opacity).abs() < 1e-10);
    }

    #[test]
    fn test_camera_perspective_default() {
        let cam = CameraPerspective::new();
        assert!((cam.fov - 45.0).abs() < 1e-10);
        assert!(cam.dof_enabled());
    }

    #[test]
    fn test_camera_perspective_dof() {
        let mut cam = CameraPerspective::new();
        cam.aperture = 22.0;
        assert!(!cam.dof_enabled());
        cam.aperture = 1.4;
        assert!(cam.dof_enabled());
    }

    #[test]
    fn test_camera_blur_at_distance() {
        let cam = CameraPerspective::new();
        // At focus distance, blur should be 0
        let blur = cam.blur_at_distance(cam.focus_distance);
        assert!((blur).abs() < 1e-10);
        // Further away, blur increases
        let blur_far = cam.blur_at_distance(cam.focus_distance + 2.0);
        assert!(blur_far > 0.0);
    }

    #[test]
    fn test_chroma_key_green_screen() {
        let key = ChromaKeySettings::green_screen();
        assert_eq!(key.key_color, [0, 255, 0, 255]);
    }

    #[test]
    fn test_chroma_key_blue_screen() {
        let key = ChromaKeySettings::blue_screen();
        assert_eq!(key.key_color, [0, 0, 255, 255]);
    }

    #[test]
    fn test_chroma_key_matches() {
        let key = ChromaKeySettings::green_screen();
        // Pure green should match
        assert!(key.matches([0, 255, 0, 255]));
        // Pure red should not match
        assert!(!key.matches([255, 0, 0, 255]));
    }

    #[test]
    fn test_virtual_set_new() {
        let vs = VirtualSet::new("Studio A", 1920, 1080);
        assert_eq!(vs.name, "Studio A");
        assert_eq!(vs.layer_count(), 0);
        assert_eq!(vs.width, 1920);
        assert_eq!(vs.height, 1080);
    }

    #[test]
    fn test_virtual_set_aspect_ratio() {
        let vs = VirtualSet::new("Test", 1920, 1080);
        let ar = vs.aspect_ratio();
        assert!((ar - 16.0 / 9.0).abs() < 0.001);
    }

    #[test]
    fn test_virtual_set_add_remove_layer() {
        let mut vs = VirtualSet::new("Test", 1920, 1080);
        let layer = VirtualSetLayer::new("bg", LayerType::Background, "bg.png");
        vs.add_layer(layer);
        assert_eq!(vs.layer_count(), 1);
        let removed = vs.remove_layer("bg");
        assert!(removed);
        assert_eq!(vs.layer_count(), 0);
        let not_removed = vs.remove_layer("nonexistent");
        assert!(!not_removed);
    }

    #[test]
    fn test_virtual_set_visible_layers() {
        let mut vs = VirtualSet::new("Test", 1920, 1080);
        let mut layer1 = VirtualSetLayer::new("bg", LayerType::Background, "bg.png");
        let mut layer2 = VirtualSetLayer::new("fg", LayerType::Foreground, "fg.png");
        layer2.visible = false;
        layer1.visible = true;
        vs.add_layer(layer1);
        vs.add_layer(layer2);
        assert_eq!(vs.visible_layers().len(), 1);
    }

    #[test]
    fn test_virtual_set_layer_z_ordering() {
        let mut vs = VirtualSet::new("Test", 1920, 1080);
        vs.add_layer(VirtualSetLayer::new("fg", LayerType::Foreground, "fg.png"));
        vs.add_layer(VirtualSetLayer::new("bg", LayerType::Background, "bg.png"));
        // After sorting, background should come first
        assert_eq!(vs.layers[0].id, "bg");
        assert_eq!(vs.layers[1].id, "fg");
    }

    #[test]
    fn test_virtual_set_tracking() {
        let mut vs = VirtualSet::new("Test", 1920, 1080);
        assert!(!vs.tracking_enabled);
        vs.enable_tracking();
        assert!(vs.tracking_enabled);
        vs.apply_tracking(Point3D::new(1.0, 0.0, -5.0), Point3D::origin());
        assert!((vs.camera.position.x - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_virtual_screen_new() {
        let pos = Point2D::new(100.0, 100.0);
        let screen = VirtualScreen::new("screen1", pos, 800, 600);
        assert_eq!(screen.id, "screen1");
        assert!(!screen.has_content());
        assert_eq!(screen.width, 800);
    }

    #[test]
    fn test_virtual_screen_content() {
        let pos = Point2D::new(0.0, 0.0);
        let mut screen = VirtualScreen::new("s1", pos, 1920, 1080);
        screen.set_content("stream://live");
        assert!(screen.has_content());
        screen.clear_content();
        assert!(!screen.has_content());
    }
}
