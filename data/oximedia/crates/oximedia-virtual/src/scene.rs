//! Virtual production scene management.
//!
//! Provides 3D scene graph management for virtual production environments,
//! including transforms, scene objects, and visibility management.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// Vec3
// ---------------------------------------------------------------------------

/// A 3D vector with f64 components.
#[derive(Debug, Clone, PartialEq)]
pub struct Vec3 {
    /// X component.
    pub x: f64,
    /// Y component.
    pub y: f64,
    /// Z component.
    pub z: f64,
}

impl Vec3 {
    /// Create a new vector.
    #[must_use]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Zero vector.
    #[must_use]
    pub fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    /// Euclidean length of the vector.
    #[must_use]
    pub fn length(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Return the unit-length vector (or zero if length is zero).
    #[must_use]
    pub fn normalize(&self) -> Self {
        let len = self.length();
        if len == 0.0 {
            return Self::zero();
        }
        Self::new(self.x / len, self.y / len, self.z / len)
    }

    /// Dot product with another vector.
    #[must_use]
    pub fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Cross product with another vector.
    #[must_use]
    pub fn cross(&self, other: &Self) -> Self {
        Self::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    /// Component-wise addition.
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }

    /// Scalar multiplication.
    #[must_use]
    pub fn scale(&self, s: f64) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }
}

// ---------------------------------------------------------------------------
// Transform3D
// ---------------------------------------------------------------------------

/// A 3D transformation with position, rotation (Euler degrees), and scale.
#[derive(Debug, Clone, PartialEq)]
pub struct Transform3D {
    /// World-space position.
    pub position: Vec3,
    /// Rotation in Euler angles (degrees), XYZ order.
    pub rotation: Vec3,
    /// Per-axis scale factors.
    pub scale_vec: Vec3,
}

impl Transform3D {
    /// Identity transform: zero position and rotation, unit scale.
    #[must_use]
    pub fn identity() -> Self {
        Self {
            position: Vec3::zero(),
            rotation: Vec3::zero(),
            scale_vec: Vec3::new(1.0, 1.0, 1.0),
        }
    }

    /// Return a copy of this transform shifted by (dx, dy, dz).
    #[must_use]
    pub fn translate(&self, dx: f64, dy: f64, dz: f64) -> Self {
        Self {
            position: Vec3::new(
                self.position.x + dx,
                self.position.y + dy,
                self.position.z + dz,
            ),
            rotation: self.rotation.clone(),
            scale_vec: self.scale_vec.clone(),
        }
    }

    /// Return a copy of this transform with an additional Y-axis rotation (degrees).
    #[must_use]
    pub fn rotate_y(&self, degrees: f64) -> Self {
        Self {
            position: self.position.clone(),
            rotation: Vec3::new(self.rotation.x, self.rotation.y + degrees, self.rotation.z),
            scale_vec: self.scale_vec.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// SceneObject
// ---------------------------------------------------------------------------

/// A named object in the virtual scene.
#[derive(Debug, Clone)]
pub struct SceneObject {
    /// Unique identifier for this object.
    pub id: u32,
    /// Human-readable name.
    pub name: String,
    /// World-space transform.
    pub transform: Transform3D,
    /// Whether the object is visible.
    pub visible: bool,
}

impl SceneObject {
    /// Create a new scene object.
    #[must_use]
    pub fn new(id: u32, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            transform: Transform3D::identity(),
            visible: true,
        }
    }

    /// Distance of this object's position from the world origin.
    #[must_use]
    pub fn distance_from_origin(&self) -> f64 {
        self.transform.position.length()
    }
}

// ---------------------------------------------------------------------------
// VirtualScene
// ---------------------------------------------------------------------------

/// A collection of [`SceneObject`]s forming a virtual production scene.
#[derive(Debug, Clone)]
pub struct VirtualScene {
    /// All objects in the scene.
    pub objects: Vec<SceneObject>,
    /// Background colour as RGB bytes.
    pub background_color: [u8; 3],
}

impl VirtualScene {
    /// Create an empty scene with a black background.
    #[must_use]
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
            background_color: [0, 0, 0],
        }
    }

    /// Add an object to the scene.
    pub fn add_object(&mut self, obj: SceneObject) {
        self.objects.push(obj);
    }

    /// Remove an object by ID. Returns `true` if an object was removed.
    pub fn remove_object(&mut self, id: u32) -> bool {
        let before = self.objects.len();
        self.objects.retain(|o| o.id != id);
        self.objects.len() < before
    }

    /// Return references to all visible objects.
    #[must_use]
    pub fn visible_objects(&self) -> Vec<&SceneObject> {
        self.objects.iter().filter(|o| o.visible).collect()
    }

    /// Find an object by ID, returning `None` if not present.
    #[must_use]
    pub fn find_object(&self, id: u32) -> Option<&SceneObject> {
        self.objects.iter().find(|o| o.id == id)
    }

    /// Total number of objects in the scene (visible and hidden).
    #[must_use]
    pub fn object_count(&self) -> usize {
        self.objects.len()
    }
}

impl Default for VirtualScene {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec3_length_zero() {
        let v = Vec3::zero();
        assert_eq!(v.length(), 0.0);
    }

    #[test]
    fn test_vec3_length_unit() {
        let v = Vec3::new(1.0, 0.0, 0.0);
        assert!((v.length() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vec3_length_pythagorean() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        assert!((v.length() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_vec3_normalize_unit() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        let n = v.normalize();
        assert!((n.length() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vec3_normalize_zero() {
        let v = Vec3::zero();
        let n = v.normalize();
        assert_eq!(n, Vec3::zero());
    }

    #[test]
    fn test_vec3_dot() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert!((a.dot(&b) - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_vec3_cross() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        let c = a.cross(&b);
        assert!((c.x).abs() < 1e-10);
        assert!((c.y).abs() < 1e-10);
        assert!((c.z - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vec3_add() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        let c = a.add(&b);
        assert_eq!(c, Vec3::new(5.0, 7.0, 9.0));
    }

    #[test]
    fn test_vec3_scale() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let s = v.scale(2.0);
        assert_eq!(s, Vec3::new(2.0, 4.0, 6.0));
    }

    #[test]
    fn test_transform_identity() {
        let t = Transform3D::identity();
        assert_eq!(t.position, Vec3::zero());
        assert_eq!(t.rotation, Vec3::zero());
        assert_eq!(t.scale_vec, Vec3::new(1.0, 1.0, 1.0));
    }

    #[test]
    fn test_transform_translate() {
        let t = Transform3D::identity().translate(1.0, 2.0, 3.0);
        assert_eq!(t.position, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_transform_rotate_y() {
        let t = Transform3D::identity().rotate_y(90.0);
        assert!((t.rotation.y - 90.0).abs() < 1e-10);
    }

    #[test]
    fn test_scene_object_distance_from_origin() {
        let mut obj = SceneObject::new(1, "cube");
        obj.transform = Transform3D::identity().translate(3.0, 4.0, 0.0);
        assert!((obj.distance_from_origin() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_virtual_scene_add_remove() {
        let mut scene = VirtualScene::new();
        scene.add_object(SceneObject::new(1, "a"));
        scene.add_object(SceneObject::new(2, "b"));
        assert_eq!(scene.object_count(), 2);

        let removed = scene.remove_object(1);
        assert!(removed);
        assert_eq!(scene.object_count(), 1);
    }

    #[test]
    fn test_virtual_scene_remove_missing() {
        let mut scene = VirtualScene::new();
        assert!(!scene.remove_object(99));
    }

    #[test]
    fn test_virtual_scene_visible_objects() {
        let mut scene = VirtualScene::new();
        let mut obj = SceneObject::new(1, "hidden");
        obj.visible = false;
        scene.add_object(obj);
        scene.add_object(SceneObject::new(2, "visible"));

        let visible = scene.visible_objects();
        assert_eq!(visible.len(), 1);
        assert_eq!(visible[0].id, 2);
    }

    #[test]
    fn test_virtual_scene_find_object() {
        let mut scene = VirtualScene::new();
        scene.add_object(SceneObject::new(42, "target"));

        assert!(scene.find_object(42).is_some());
        assert!(scene.find_object(99).is_none());
    }
}
