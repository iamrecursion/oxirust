#![allow(dead_code)]
//! Background plate management for virtual production compositing.

/// Type of background plate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlateType {
    /// Real-world footage captured on set.
    LiveAction,
    /// Fully computer-generated imagery.
    Cgi,
    /// A still image used as background.
    StillImage,
    /// An HDRI environment map.
    Hdri,
    /// A procedurally generated background.
    Procedural,
}

impl PlateType {
    /// Return `true` if this plate is CGI-derived (not real footage).
    #[must_use]
    pub fn is_cgi(&self) -> bool {
        matches!(self, PlateType::Cgi | PlateType::Procedural)
    }

    /// Return a human-readable name for the plate type.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            PlateType::LiveAction => "Live Action",
            PlateType::Cgi => "CGI",
            PlateType::StillImage => "Still Image",
            PlateType::Hdri => "HDRI",
            PlateType::Procedural => "Procedural",
        }
    }
}

/// A background plate with metadata.
#[derive(Debug, Clone)]
pub struct BackgroundPlate {
    /// Plate identifier.
    pub id: String,
    /// Type of plate.
    pub plate_type: PlateType,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Frame rate (0.0 means still image).
    pub frame_rate: f64,
}

impl BackgroundPlate {
    /// Create a new background plate.
    pub fn new(
        id: impl Into<String>,
        plate_type: PlateType,
        width: u32,
        height: u32,
        frame_rate: f64,
    ) -> Self {
        Self {
            id: id.into(),
            plate_type,
            width,
            height,
            frame_rate,
        }
    }

    /// Return `true` if the plate has non-zero dimensions.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.width > 0 && self.height > 0
    }

    /// Aspect ratio (width / height).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn aspect_ratio(&self) -> f64 {
        f64::from(self.width) / f64::from(self.height)
    }
}

/// 2-D affine transform applied to a plate (scale, rotation, translation).
#[derive(Debug, Clone)]
pub struct PlateTransform {
    /// Uniform scale factor.
    pub scale: f64,
    /// Rotation in degrees.
    pub rotation_deg: f64,
    /// Horizontal translation in normalised units (0..1).
    pub tx: f64,
    /// Vertical translation in normalised units (0..1).
    pub ty: f64,
}

impl Default for PlateTransform {
    fn default() -> Self {
        Self {
            scale: 1.0,
            rotation_deg: 0.0,
            tx: 0.0,
            ty: 0.0,
        }
    }
}

impl PlateTransform {
    /// Create a new transform.
    #[must_use]
    pub fn new(scale: f64, rotation_deg: f64, tx: f64, ty: f64) -> Self {
        Self {
            scale,
            rotation_deg,
            tx,
            ty,
        }
    }

    /// Apply the transform to a point `(x, y)` and return the transformed point.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn apply_transform(&self, x: f64, y: f64) -> (f64, f64) {
        use std::f64::consts::PI;
        let angle = self.rotation_deg * PI / 180.0;
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        let sx = x * self.scale;
        let sy = y * self.scale;
        let rx = sx * cos_a - sy * sin_a;
        let ry = sx * sin_a + sy * cos_a;
        (rx + self.tx, ry + self.ty)
    }

    /// Return `true` if this is an identity transform.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        (self.scale - 1.0).abs() < 1e-9
            && self.rotation_deg.abs() < 1e-9
            && self.tx.abs() < 1e-9
            && self.ty.abs() < 1e-9
    }
}

/// Result of compositing a plate onto the output frame.
#[derive(Debug, Clone)]
pub struct CompositedFrame {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Plate ID that was composited.
    pub plate_id: String,
    /// Whether the plate was successfully applied.
    pub applied: bool,
}

/// Compositor that layers background plates.
#[derive(Debug, Default)]
pub struct PlateCompositor {
    plates: Vec<(BackgroundPlate, PlateTransform)>,
}

impl PlateCompositor {
    /// Create a new compositor.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a plate with a transform.
    pub fn add_plate(&mut self, plate: BackgroundPlate, transform: PlateTransform) {
        self.plates.push((plate, transform));
    }

    /// Composite all plates for a given output resolution.
    /// Returns one `CompositedFrame` per plate, in layer order.
    #[must_use]
    pub fn composite(&self, out_width: u32, out_height: u32) -> Vec<CompositedFrame> {
        self.plates
            .iter()
            .map(|(plate, _transform)| CompositedFrame {
                width: out_width,
                height: out_height,
                plate_id: plate.id.clone(),
                applied: plate.is_valid(),
            })
            .collect()
    }

    /// Return number of registered plates.
    #[must_use]
    pub fn plate_count(&self) -> usize {
        self.plates.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plate_type_is_cgi_true() {
        assert!(PlateType::Cgi.is_cgi());
        assert!(PlateType::Procedural.is_cgi());
    }

    #[test]
    fn test_plate_type_is_cgi_false() {
        assert!(!PlateType::LiveAction.is_cgi());
        assert!(!PlateType::StillImage.is_cgi());
        assert!(!PlateType::Hdri.is_cgi());
    }

    #[test]
    fn test_plate_type_labels() {
        assert_eq!(PlateType::LiveAction.label(), "Live Action");
        assert_eq!(PlateType::Cgi.label(), "CGI");
        assert_eq!(PlateType::Hdri.label(), "HDRI");
    }

    #[test]
    fn test_background_plate_valid() {
        let plate = BackgroundPlate::new("BG1", PlateType::LiveAction, 1920, 1080, 25.0);
        assert!(plate.is_valid());
    }

    #[test]
    fn test_background_plate_invalid_zero_width() {
        let plate = BackgroundPlate::new("BG0", PlateType::Cgi, 0, 1080, 25.0);
        assert!(!plate.is_valid());
    }

    #[test]
    fn test_background_plate_invalid_zero_height() {
        let plate = BackgroundPlate::new("BG0", PlateType::Cgi, 1920, 0, 25.0);
        assert!(!plate.is_valid());
    }

    #[test]
    fn test_background_plate_aspect_ratio() {
        let plate = BackgroundPlate::new("BG", PlateType::StillImage, 1920, 1080, 0.0);
        assert!((plate.aspect_ratio() - 16.0 / 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_plate_transform_default_identity() {
        let t = PlateTransform::default();
        assert!(t.is_identity());
    }

    #[test]
    fn test_plate_transform_non_identity() {
        let t = PlateTransform::new(2.0, 0.0, 0.0, 0.0);
        assert!(!t.is_identity());
    }

    #[test]
    fn test_plate_transform_apply_identity() {
        let t = PlateTransform::default();
        let (x, y) = t.apply_transform(3.0, 4.0);
        assert!((x - 3.0).abs() < 1e-9);
        assert!((y - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_plate_transform_apply_scale() {
        let t = PlateTransform::new(2.0, 0.0, 0.0, 0.0);
        let (x, y) = t.apply_transform(1.0, 1.0);
        assert!((x - 2.0).abs() < 1e-9);
        assert!((y - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_plate_transform_apply_translation() {
        let t = PlateTransform::new(1.0, 0.0, 0.5, 0.25);
        let (x, y) = t.apply_transform(0.0, 0.0);
        assert!((x - 0.5).abs() < 1e-9);
        assert!((y - 0.25).abs() < 1e-9);
    }

    #[test]
    fn test_compositor_empty() {
        let compositor = PlateCompositor::new();
        let frames = compositor.composite(1920, 1080);
        assert!(frames.is_empty());
    }

    #[test]
    fn test_compositor_composite() {
        let mut compositor = PlateCompositor::new();
        let plate = BackgroundPlate::new("P1", PlateType::Cgi, 1920, 1080, 24.0);
        compositor.add_plate(plate, PlateTransform::default());
        let frames = compositor.composite(1920, 1080);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].plate_id, "P1");
        assert!(frames[0].applied);
    }

    #[test]
    fn test_compositor_invalid_plate_not_applied() {
        let mut compositor = PlateCompositor::new();
        let plate = BackgroundPlate::new("Bad", PlateType::Cgi, 0, 0, 0.0);
        compositor.add_plate(plate, PlateTransform::default());
        let frames = compositor.composite(1920, 1080);
        assert!(!frames[0].applied);
    }

    #[test]
    fn test_compositor_plate_count() {
        let mut compositor = PlateCompositor::new();
        compositor.add_plate(
            BackgroundPlate::new("A", PlateType::Hdri, 4096, 2048, 0.0),
            PlateTransform::default(),
        );
        compositor.add_plate(
            BackgroundPlate::new("B", PlateType::Cgi, 1920, 1080, 30.0),
            PlateTransform::default(),
        );
        assert_eq!(compositor.plate_count(), 2);
    }
}
