#![allow(dead_code)]
//! Scene setup for virtual production: cameras, lights, and scene composition.

use std::f64::consts::PI;

/// A virtual camera in the scene.
#[derive(Debug, Clone)]
pub struct VirtualCamera {
    /// Horizontal field of view in degrees.
    pub fov_deg: f64,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Camera label.
    pub label: String,
}

impl VirtualCamera {
    /// Create a new virtual camera.
    pub fn new(label: impl Into<String>, fov_deg: f64, width: u32, height: u32) -> Self {
        Self {
            label: label.into(),
            fov_deg,
            width,
            height,
        }
    }

    /// Return field of view in radians.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn fov_radians(&self) -> f64 {
        self.fov_deg * PI / 180.0
    }

    /// Return the aspect ratio (width / height).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn aspect_ratio(&self) -> f64 {
        f64::from(self.width) / f64::from(self.height)
    }

    /// Vertical FOV derived from horizontal FOV and aspect ratio.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn vfov_radians(&self) -> f64 {
        let aspect = self.aspect_ratio();
        2.0 * ((self.fov_radians() / 2.0).tan() / aspect).atan()
    }
}

/// A virtual light source.
#[derive(Debug, Clone)]
pub struct VirtualLight {
    /// Light label.
    pub label: String,
    /// Luminous intensity in candela.
    pub intensity_cd: f64,
    /// Colour temperature in kelvin.
    pub color_temp_k: f64,
}

impl VirtualLight {
    /// Create a new virtual light.
    pub fn new(label: impl Into<String>, intensity_cd: f64, color_temp_k: f64) -> Self {
        Self {
            label: label.into(),
            intensity_cd,
            color_temp_k,
        }
    }

    /// Illuminance (lux) at a given distance in metres, using inverse-square law.
    #[must_use]
    pub fn intensity_at_distance(&self, distance_m: f64) -> f64 {
        if distance_m <= 0.0 {
            return 0.0;
        }
        self.intensity_cd / (distance_m * distance_m)
    }
}

/// Represents the full virtual scene composed of cameras and lights.
#[derive(Debug, Default)]
pub struct SceneSetup {
    cameras: Vec<VirtualCamera>,
    lights: Vec<VirtualLight>,
}

impl SceneSetup {
    /// Create a new, empty scene.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a camera to the scene.
    pub fn add_camera(&mut self, camera: VirtualCamera) {
        self.cameras.push(camera);
    }

    /// Add a light to the scene.
    pub fn add_light(&mut self, light: VirtualLight) {
        self.lights.push(light);
    }

    /// Return the number of cameras in the scene.
    #[must_use]
    pub fn camera_count(&self) -> usize {
        self.cameras.len()
    }

    /// Return the number of lights in the scene.
    #[must_use]
    pub fn light_count(&self) -> usize {
        self.lights.len()
    }

    /// Access all cameras.
    #[must_use]
    pub fn cameras(&self) -> &[VirtualCamera] {
        &self.cameras
    }

    /// Access all lights.
    #[must_use]
    pub fn lights(&self) -> &[VirtualLight] {
        &self.lights
    }

    /// Find a camera by label.
    #[must_use]
    pub fn camera_by_label(&self, label: &str) -> Option<&VirtualCamera> {
        self.cameras.iter().find(|c| c.label == label)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_fov_radians_zero() {
        let cam = VirtualCamera::new("A", 0.0, 1920, 1080);
        assert!((cam.fov_radians() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_fov_radians_90() {
        let cam = VirtualCamera::new("A", 90.0, 1920, 1080);
        assert!((cam.fov_radians() - PI / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_fov_radians_180() {
        let cam = VirtualCamera::new("A", 180.0, 1920, 1080);
        assert!((cam.fov_radians() - PI).abs() < 1e-10);
    }

    #[test]
    fn test_aspect_ratio_16_9() {
        let cam = VirtualCamera::new("A", 90.0, 1920, 1080);
        let ratio = cam.aspect_ratio();
        assert!((ratio - 16.0 / 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_aspect_ratio_1_1() {
        let cam = VirtualCamera::new("A", 90.0, 1024, 1024);
        assert!((cam.aspect_ratio() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vfov_radians_square() {
        let cam = VirtualCamera::new("A", 90.0, 1024, 1024);
        // For square sensor, VFOV == HFOV
        assert!((cam.vfov_radians() - cam.fov_radians()).abs() < 1e-10);
    }

    #[test]
    fn test_light_intensity_at_distance() {
        let light = VirtualLight::new("Key", 1000.0, 5600.0);
        let lux = light.intensity_at_distance(10.0);
        assert!((lux - 10.0).abs() < 1e-6); // 1000 / 100
    }

    #[test]
    fn test_light_intensity_at_zero_distance() {
        let light = VirtualLight::new("Key", 1000.0, 5600.0);
        assert_eq!(light.intensity_at_distance(0.0), 0.0);
    }

    #[test]
    fn test_light_intensity_negative_distance() {
        let light = VirtualLight::new("Key", 500.0, 4000.0);
        assert_eq!(light.intensity_at_distance(-5.0), 0.0);
    }

    #[test]
    fn test_scene_setup_empty() {
        let scene = SceneSetup::new();
        assert_eq!(scene.camera_count(), 0);
        assert_eq!(scene.light_count(), 0);
    }

    #[test]
    fn test_scene_add_camera() {
        let mut scene = SceneSetup::new();
        scene.add_camera(VirtualCamera::new("Cam1", 60.0, 1920, 1080));
        scene.add_camera(VirtualCamera::new("Cam2", 45.0, 3840, 2160));
        assert_eq!(scene.camera_count(), 2);
    }

    #[test]
    fn test_scene_add_light() {
        let mut scene = SceneSetup::new();
        scene.add_light(VirtualLight::new("Fill", 500.0, 4500.0));
        assert_eq!(scene.light_count(), 1);
    }

    #[test]
    fn test_scene_camera_by_label() {
        let mut scene = SceneSetup::new();
        scene.add_camera(VirtualCamera::new("Main", 55.0, 1920, 1080));
        let found = scene.camera_by_label("Main");
        assert!(found.is_some());
        assert_eq!(found.expect("should succeed in test").label, "Main");
    }

    #[test]
    fn test_scene_camera_by_label_missing() {
        let scene = SceneSetup::new();
        assert!(scene.camera_by_label("Ghost").is_none());
    }

    #[test]
    fn test_light_color_temp() {
        let light = VirtualLight::new("HMI", 2000.0, 5600.0);
        assert!((light.color_temp_k - 5600.0).abs() < 1e-6);
    }
}
