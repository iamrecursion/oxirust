#![allow(dead_code)]
//! Virtual-production lighting rig management.
//!
//! Models light fixtures on an LED-wall stage, including colour temperature,
//! intensity, position, and DMX-style grouping.  Provides helpers for
//! matching practical lights to the virtual scene and computing aggregate
//! exposure across the stage.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Colour
// ---------------------------------------------------------------------------

/// Linear RGB colour (0.0 – 1.0 per channel).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LightColor {
    /// Red channel.
    pub r: f64,
    /// Green channel.
    pub g: f64,
    /// Blue channel.
    pub b: f64,
}

impl LightColor {
    /// Create a new light colour.
    #[must_use]
    pub fn new(r: f64, g: f64, b: f64) -> Self {
        Self { r, g, b }
    }

    /// Pure white (D65-ish).
    #[must_use]
    pub fn white() -> Self {
        Self::new(1.0, 1.0, 1.0)
    }

    /// Scale (dim) by a factor.
    #[must_use]
    pub fn scaled(&self, factor: f64) -> Self {
        Self {
            r: (self.r * factor).clamp(0.0, 1.0),
            g: (self.g * factor).clamp(0.0, 1.0),
            b: (self.b * factor).clamp(0.0, 1.0),
        }
    }

    /// Approximate luminance (BT.709 coefficients).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn luminance(&self) -> f64 {
        0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b
    }

    /// Linear blend between two colours.
    #[must_use]
    pub fn blend(&self, other: &Self, t: f64) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self {
            r: self.r + (other.r - self.r) * t,
            g: self.g + (other.g - self.g) * t,
            b: self.b + (other.b - self.b) * t,
        }
    }

    /// Convert correlated colour temperature (Kelvin) to approximate RGB.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn from_cct(kelvin: u32) -> Self {
        // Tanner Helland approximation
        let temp = (kelvin as f64) / 100.0;
        let r = if temp <= 66.0 {
            1.0
        } else {
            let x = temp - 60.0;
            (329.698_727_446 * x.powf(-0.133_204_759_2) / 255.0).clamp(0.0, 1.0)
        };
        let g = if temp <= 66.0 {
            let x = temp;
            (99.470_802_586_1 * x.ln() - 161.119_568_166_1).clamp(0.0, 255.0) / 255.0
        } else {
            let x = temp - 60.0;
            (288.122_169_528_3 * x.powf(-0.075_514_849_2) / 255.0).clamp(0.0, 1.0)
        };
        let b = if temp >= 66.0 {
            1.0
        } else if temp <= 19.0 {
            0.0
        } else {
            let x = temp - 10.0;
            (138.517_731_223_1 * x.ln() - 305.044_792_730_7).clamp(0.0, 255.0) / 255.0
        };
        Self { r, g, b }
    }
}

impl Default for LightColor {
    fn default() -> Self {
        Self::white()
    }
}

// ---------------------------------------------------------------------------
// Light fixture
// ---------------------------------------------------------------------------

/// Type of light fixture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FixtureKind {
    /// Fresnel spot.
    Fresnel,
    /// LED panel / soft light.
    LedPanel,
    /// Par can.
    Par,
    /// Ellipsoidal / profile.
    Profile,
    /// Moving head.
    MovingHead,
    /// Practical (on-set prop light).
    Practical,
}

/// A single light fixture.
#[derive(Debug, Clone, PartialEq)]
pub struct LightFixture {
    /// Unique fixture identifier.
    pub id: String,
    /// Human name.
    pub name: String,
    /// Type of fixture.
    pub kind: FixtureKind,
    /// Current colour output.
    pub color: LightColor,
    /// Intensity 0.0 – 1.0.
    pub intensity: f64,
    /// Colour temperature in Kelvin (if white-light mode).
    pub cct_kelvin: u32,
    /// DMX universe (0-based).
    pub dmx_universe: u16,
    /// DMX start address (1-based).
    pub dmx_address: u16,
    /// Whether the fixture is currently active.
    pub active: bool,
}

impl LightFixture {
    /// Create a new fixture with sensible defaults.
    #[must_use]
    pub fn new(id: impl Into<String>, name: impl Into<String>, kind: FixtureKind) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            kind,
            color: LightColor::white(),
            intensity: 1.0,
            cct_kelvin: 5600,
            dmx_universe: 0,
            dmx_address: 1,
            active: true,
        }
    }

    /// Builder: set intensity.
    #[must_use]
    pub fn with_intensity(mut self, i: f64) -> Self {
        self.intensity = i.clamp(0.0, 1.0);
        self
    }

    /// Builder: set CCT.
    #[must_use]
    pub fn with_cct(mut self, kelvin: u32) -> Self {
        self.cct_kelvin = kelvin;
        self.color = LightColor::from_cct(kelvin);
        self
    }

    /// Builder: set DMX address.
    #[must_use]
    pub fn with_dmx(mut self, universe: u16, address: u16) -> Self {
        self.dmx_universe = universe;
        self.dmx_address = address;
        self
    }

    /// Effective colour (colour * intensity).
    #[must_use]
    pub fn effective_color(&self) -> LightColor {
        if self.active {
            self.color.scaled(self.intensity)
        } else {
            LightColor::new(0.0, 0.0, 0.0)
        }
    }

    /// Effective luminance output.
    #[must_use]
    pub fn effective_luminance(&self) -> f64 {
        self.effective_color().luminance()
    }
}

// ---------------------------------------------------------------------------
// Group / LightRig
// ---------------------------------------------------------------------------

/// A named group of fixtures.
#[derive(Debug, Clone)]
pub struct FixtureGroup {
    /// Group name.
    pub name: String,
    /// IDs of fixtures belonging to this group.
    pub fixture_ids: Vec<String>,
}

impl FixtureGroup {
    /// Create a new group.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            fixture_ids: Vec::new(),
        }
    }

    /// Add a fixture ID.
    pub fn add(&mut self, id: impl Into<String>) {
        self.fixture_ids.push(id.into());
    }

    /// Number of fixtures.
    #[must_use]
    pub fn len(&self) -> usize {
        self.fixture_ids.len()
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.fixture_ids.is_empty()
    }
}

/// The full lighting rig for a virtual-production stage.
#[derive(Debug, Clone)]
pub struct LightRig {
    /// All fixtures keyed by ID.
    fixtures: HashMap<String, LightFixture>,
    /// Named groups.
    groups: HashMap<String, FixtureGroup>,
}

impl LightRig {
    /// Create an empty rig.
    #[must_use]
    pub fn new() -> Self {
        Self {
            fixtures: HashMap::new(),
            groups: HashMap::new(),
        }
    }

    /// Add a fixture. Returns `false` if the ID is a duplicate.
    pub fn add_fixture(&mut self, fixture: LightFixture) -> bool {
        if self.fixtures.contains_key(&fixture.id) {
            return false;
        }
        self.fixtures.insert(fixture.id.clone(), fixture);
        true
    }

    /// Remove a fixture by ID.
    pub fn remove_fixture(&mut self, id: &str) -> Option<LightFixture> {
        self.fixtures.remove(id)
    }

    /// Get a fixture by ID.
    #[must_use]
    pub fn fixture(&self, id: &str) -> Option<&LightFixture> {
        self.fixtures.get(id)
    }

    /// Mutable access to a fixture.
    pub fn fixture_mut(&mut self, id: &str) -> Option<&mut LightFixture> {
        self.fixtures.get_mut(id)
    }

    /// Total number of fixtures.
    #[must_use]
    pub fn fixture_count(&self) -> usize {
        self.fixtures.len()
    }

    /// Add a named group.
    pub fn add_group(&mut self, group: FixtureGroup) {
        self.groups.insert(group.name.clone(), group);
    }

    /// Set intensity for an entire group.
    pub fn set_group_intensity(&mut self, group_name: &str, intensity: f64) {
        if let Some(group) = self.groups.get(group_name) {
            let ids: Vec<String> = group.fixture_ids.clone();
            for id in &ids {
                if let Some(f) = self.fixtures.get_mut(id) {
                    f.intensity = intensity.clamp(0.0, 1.0);
                }
            }
        }
    }

    /// Average luminance of all active fixtures.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn average_luminance(&self) -> f64 {
        let active: Vec<_> = self.fixtures.values().filter(|f| f.active).collect();
        if active.is_empty() {
            return 0.0;
        }
        let sum: f64 = active.iter().map(|f| f.effective_luminance()).sum();
        sum / active.len() as f64
    }

    /// Count of active fixtures.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.fixtures.values().filter(|f| f.active).count()
    }

    /// Blackout: set all intensities to 0.
    pub fn blackout(&mut self) {
        for f in self.fixtures.values_mut() {
            f.intensity = 0.0;
        }
    }

    /// Restore all fixtures to full intensity.
    pub fn full_on(&mut self) {
        for f in self.fixtures.values_mut() {
            f.intensity = 1.0;
            f.active = true;
        }
    }
}

impl Default for LightRig {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_fixture(id: &str) -> LightFixture {
        LightFixture::new(id, id, FixtureKind::LedPanel)
    }

    // -- LightColor --

    #[test]
    fn test_color_luminance_white() {
        let c = LightColor::white();
        assert!((c.luminance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_color_scaled() {
        let c = LightColor::white().scaled(0.5);
        assert!((c.r - 0.5).abs() < 1e-9);
        assert!((c.g - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_color_blend() {
        let a = LightColor::new(0.0, 0.0, 0.0);
        let b = LightColor::new(1.0, 1.0, 1.0);
        let mid = a.blend(&b, 0.5);
        assert!((mid.r - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_color_from_cct_daylight() {
        let c = LightColor::from_cct(5600);
        // Daylight should be close to white-ish
        assert!(c.r > 0.8);
        assert!(c.g > 0.8);
        assert!(c.b > 0.5);
    }

    #[test]
    fn test_color_from_cct_tungsten() {
        let c = LightColor::from_cct(3200);
        // Tungsten is warm → more red, less blue
        assert!(c.r > c.b);
    }

    // -- LightFixture --

    #[test]
    fn test_fixture_effective_color() {
        let f = make_fixture("a").with_intensity(0.5);
        let ec = f.effective_color();
        assert!((ec.r - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_fixture_inactive() {
        let mut f = make_fixture("a");
        f.active = false;
        assert!(f.effective_luminance() < 1e-9);
    }

    #[test]
    fn test_fixture_cct_builder() {
        let f = make_fixture("a").with_cct(3200);
        assert_eq!(f.cct_kelvin, 3200);
        assert!(f.color.r > 0.5);
    }

    // -- FixtureGroup --

    #[test]
    fn test_group_add() {
        let mut g = FixtureGroup::new("key");
        assert!(g.is_empty());
        g.add("f1");
        g.add("f2");
        assert_eq!(g.len(), 2);
    }

    // -- LightRig --

    #[test]
    fn test_rig_add_remove() {
        let mut rig = LightRig::new();
        assert!(rig.add_fixture(make_fixture("a")));
        assert!(!rig.add_fixture(make_fixture("a"))); // duplicate
        assert_eq!(rig.fixture_count(), 1);
        assert!(rig.remove_fixture("a").is_some());
        assert_eq!(rig.fixture_count(), 0);
    }

    #[test]
    fn test_rig_blackout_and_full() {
        let mut rig = LightRig::new();
        rig.add_fixture(make_fixture("a"));
        rig.add_fixture(make_fixture("b"));
        rig.blackout();
        assert!(rig.average_luminance() < 1e-9);
        rig.full_on();
        assert!(rig.average_luminance() > 0.9);
    }

    #[test]
    fn test_rig_group_intensity() {
        let mut rig = LightRig::new();
        rig.add_fixture(make_fixture("a"));
        rig.add_fixture(make_fixture("b"));
        let mut g = FixtureGroup::new("all");
        g.add("a");
        g.add("b");
        rig.add_group(g);
        rig.set_group_intensity("all", 0.25);
        let f = rig.fixture("a").expect("should succeed in test");
        assert!((f.intensity - 0.25).abs() < 1e-9);
    }

    #[test]
    fn test_rig_active_count() {
        let mut rig = LightRig::new();
        rig.add_fixture(make_fixture("a"));
        let mut f = make_fixture("b");
        f.active = false;
        rig.add_fixture(f);
        assert_eq!(rig.active_count(), 1);
    }
}
