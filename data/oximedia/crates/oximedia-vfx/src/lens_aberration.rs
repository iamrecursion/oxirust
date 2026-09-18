//! Lens aberration and optical flare element simulation.
//!
//! Models the individual optical artefacts produced by imperfect lenses:
//! circular flare elements (ghosts, halos, streaks) and their visibility
//! gating.  Complements `lens_flare` with a composable element model.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

/// Type of optical flare artefact produced by a lens element.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FlareType {
    /// Circular ghost reflected between lens elements.
    CircularGhost,
    /// Hexagonal aperture ghost.
    HexagonalGhost,
    /// Star-burst streak radiating from a bright light source.
    Starburst,
    /// Soft halo around a bright source.
    Halo,
    /// Anamorphic horizontal streak (used in wide-angle lenses).
    AnamorphicStreak,
    /// Bright central disk directly over the light source.
    CentralDisk,
}

impl FlareType {
    /// Relative intensity scale factor for this flare type (0.0 – 1.0).
    pub fn intensity_scale(self) -> f32 {
        match self {
            Self::CircularGhost => 0.35,
            Self::HexagonalGhost => 0.25,
            Self::Starburst => 0.60,
            Self::Halo => 0.45,
            Self::AnamorphicStreak => 0.80,
            Self::CentralDisk => 1.00,
        }
    }

    /// Whether this type is a secondary reflection (ghost) rather than a primary artefact.
    pub fn is_ghost(self) -> bool {
        matches!(self, Self::CircularGhost | Self::HexagonalGhost)
    }

    /// Human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            Self::CircularGhost => "circular_ghost",
            Self::HexagonalGhost => "hexagonal_ghost",
            Self::Starburst => "starburst",
            Self::Halo => "halo",
            Self::AnamorphicStreak => "anamorphic_streak",
            Self::CentralDisk => "central_disk",
        }
    }
}

/// A single optical flare element with position, size, colour, and opacity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlareElement {
    /// Type of this optical artefact.
    pub flare_type: FlareType,
    /// Position along the axis from the light source to the image centre (0.0 = source, 1.0 = centre).
    pub axis_position: f32,
    /// Element radius in normalised image coordinates (0.0 – 1.0 range).
    pub radius: f32,
    /// RGBA colour (each channel 0.0 – 1.0).
    pub color: [f32; 4],
    /// Overall opacity multiplier (0.0 = invisible, 1.0 = fully visible).
    pub opacity: f32,
}

impl FlareElement {
    /// Create a new flare element with default neutral white colour.
    pub fn new(flare_type: FlareType, axis_position: f32, radius: f32) -> Self {
        Self {
            flare_type,
            axis_position: axis_position.clamp(-2.0, 2.0),
            radius: radius.max(0.0),
            color: [1.0, 1.0, 1.0, 1.0],
            opacity: 1.0,
        }
    }

    /// Set the element colour.
    pub fn with_color(mut self, r: f32, g: f32, b: f32, a: f32) -> Self {
        self.color = [r, g, b, a];
        self
    }

    /// Set the element opacity.
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity.clamp(0.0, 1.0);
        self
    }

    /// Returns `true` if this element would contribute visibly to the image.
    pub fn is_visible(&self) -> bool {
        self.opacity > 0.0 && self.radius > 0.0 && self.color[3] > 0.0
    }

    /// Effective intensity combining the flare-type scale and opacity.
    pub fn effective_intensity(&self) -> f32 {
        self.flare_type.intensity_scale() * self.opacity
    }
}

/// Configuration for a multi-element lens flare effect.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LensFlareConfig {
    /// Global brightness of the entire flare group (0.0 – 1.0).
    pub brightness: f32,
    /// Whether physically-based (energy-conserving) rendering should be used.
    pub realistic_mode: bool,
    /// Threshold luminance above which the flare activates (0.0 – 1.0).
    pub activation_threshold: f32,
    /// Number of aperture blades (affects ghost shape).
    pub aperture_blades: u8,
}

impl Default for LensFlareConfig {
    fn default() -> Self {
        Self {
            brightness: 0.8,
            realistic_mode: true,
            activation_threshold: 0.9,
            aperture_blades: 6,
        }
    }
}

impl LensFlareConfig {
    /// Returns `true` when realistic (physically-based) rendering is active.
    pub fn is_realistic(&self) -> bool {
        self.realistic_mode
    }
}

/// A lens flare composed of multiple [`FlareElement`]s.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LensFlare {
    elements: Vec<FlareElement>,
    /// Configuration for the overall effect.
    pub config: LensFlareConfig,
}

impl Default for LensFlare {
    fn default() -> Self {
        Self::new()
    }
}

impl LensFlare {
    /// Create an empty lens flare with default configuration.
    pub fn new() -> Self {
        Self {
            elements: Vec::new(),
            config: LensFlareConfig::default(),
        }
    }

    /// Add a [`FlareElement`] to this flare.
    pub fn add_element(&mut self, element: FlareElement) {
        self.elements.push(element);
    }

    /// Total number of elements (visible or not).
    pub fn element_count(&self) -> usize {
        self.elements.len()
    }

    /// Number of currently visible elements.
    pub fn visible_count(&self) -> usize {
        self.elements.iter().filter(|e| e.is_visible()).count()
    }

    /// Returns a reference to all elements.
    pub fn elements(&self) -> &[FlareElement] {
        &self.elements
    }

    /// Build a simple preset flare with a central disk, halo, and two ghosts.
    pub fn preset_simple() -> Self {
        let mut flare = Self::new();
        flare.add_element(FlareElement::new(FlareType::CentralDisk, 0.0, 0.05));
        flare.add_element(FlareElement::new(FlareType::Halo, 0.0, 0.25));
        flare.add_element(
            FlareElement::new(FlareType::CircularGhost, 0.4, 0.08).with_color(0.8, 0.9, 1.0, 1.0),
        );
        flare.add_element(
            FlareElement::new(FlareType::CircularGhost, 0.8, 0.05).with_color(1.0, 0.8, 0.7, 1.0),
        );
        flare
    }

    /// Total effective intensity across all visible elements.
    pub fn total_intensity(&self) -> f32 {
        self.elements
            .iter()
            .filter(|e| e.is_visible())
            .map(FlareElement::effective_intensity)
            .sum::<f32>()
            * self.config.brightness
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flare_type_intensity_scale_range() {
        let types = [
            FlareType::CircularGhost,
            FlareType::HexagonalGhost,
            FlareType::Starburst,
            FlareType::Halo,
            FlareType::AnamorphicStreak,
            FlareType::CentralDisk,
        ];
        for t in types {
            let s = t.intensity_scale();
            assert!(s > 0.0 && s <= 1.0, "{} out of range: {s}", t.label());
        }
    }

    #[test]
    fn test_flare_type_is_ghost() {
        assert!(FlareType::CircularGhost.is_ghost());
        assert!(FlareType::HexagonalGhost.is_ghost());
        assert!(!FlareType::Starburst.is_ghost());
        assert!(!FlareType::CentralDisk.is_ghost());
    }

    #[test]
    fn test_flare_type_central_disk_max_intensity() {
        assert_eq!(FlareType::CentralDisk.intensity_scale(), 1.0);
    }

    #[test]
    fn test_flare_element_is_visible_default() {
        let e = FlareElement::new(FlareType::Halo, 0.0, 0.2);
        assert!(e.is_visible());
    }

    #[test]
    fn test_flare_element_invisible_zero_opacity() {
        let e = FlareElement::new(FlareType::Halo, 0.0, 0.2).with_opacity(0.0);
        assert!(!e.is_visible());
    }

    #[test]
    fn test_flare_element_invisible_zero_radius() {
        let e = FlareElement::new(FlareType::Halo, 0.0, 0.0);
        assert!(!e.is_visible());
    }

    #[test]
    fn test_flare_element_effective_intensity() {
        let e = FlareElement::new(FlareType::CentralDisk, 0.0, 0.05).with_opacity(0.5);
        assert!((e.effective_intensity() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_lens_flare_add_element() {
        let mut flare = LensFlare::new();
        assert_eq!(flare.element_count(), 0);
        flare.add_element(FlareElement::new(FlareType::Halo, 0.0, 0.1));
        assert_eq!(flare.element_count(), 1);
    }

    #[test]
    fn test_lens_flare_visible_count() {
        let mut flare = LensFlare::new();
        flare.add_element(FlareElement::new(FlareType::Halo, 0.0, 0.1));
        flare.add_element(FlareElement::new(FlareType::Halo, 0.5, 0.0)); // zero radius = invisible
        assert_eq!(flare.visible_count(), 1);
    }

    #[test]
    fn test_lens_flare_preset_simple_visible() {
        let flare = LensFlare::preset_simple();
        assert_eq!(flare.element_count(), 4);
        assert_eq!(flare.visible_count(), 4);
    }

    #[test]
    fn test_lens_flare_config_is_realistic() {
        let cfg = LensFlareConfig::default();
        assert!(cfg.is_realistic());
        let cfg2 = LensFlareConfig {
            realistic_mode: false,
            ..Default::default()
        };
        assert!(!cfg2.is_realistic());
    }

    #[test]
    fn test_lens_flare_total_intensity_positive() {
        let flare = LensFlare::preset_simple();
        assert!(flare.total_intensity() > 0.0);
    }

    #[test]
    fn test_flare_element_color_stored() {
        let e = FlareElement::new(FlareType::Starburst, 0.0, 0.1).with_color(1.0, 0.0, 0.5, 0.8);
        assert!((e.color[0] - 1.0).abs() < 1e-6);
        assert!((e.color[1] - 0.0).abs() < 1e-6);
        assert!((e.color[3] - 0.8).abs() < 1e-6);
    }
}
