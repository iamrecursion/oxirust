#![allow(dead_code)]

//! VFX preset management: named parameter bundles for effects.
//!
//! Presets allow users to save, load, and interpolate between effect
//! configurations. They are serialisable and can be shared across projects.

use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// Preset value model
// ---------------------------------------------------------------------------

/// A single parameter value stored inside a preset.
#[derive(Debug, Clone, PartialEq)]
pub enum PresetValue {
    /// Floating-point scalar.
    Float(f64),
    /// Integer scalar.
    Int(i64),
    /// Boolean flag.
    Bool(bool),
    /// Text string.
    Text(String),
    /// 2-D vector (x, y).
    Vec2(f64, f64),
    /// RGBA colour (0.0 – 1.0 each).
    Color(f64, f64, f64, f64),
}

impl PresetValue {
    /// Try to extract an `f64` from the value.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Self::Float(v) => Some(*v),
            Self::Int(v) => Some(*v as f64),
            _ => None,
        }
    }

    /// Try to extract an `i64`.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to extract a bool.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to extract a string slice.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Linearly interpolate between two `PresetValue`s of the same variant.
    ///
    /// Returns `None` when the variants differ or interpolation isn't meaningful.
    #[allow(clippy::cast_precision_loss)]
    pub fn lerp(&self, other: &Self, t: f64) -> Option<Self> {
        let t = t.clamp(0.0, 1.0);
        match (self, other) {
            (Self::Float(a), Self::Float(b)) => Some(Self::Float(a + (b - a) * t)),
            (Self::Int(a), Self::Int(b)) => {
                let v = *a as f64 + (*b as f64 - *a as f64) * t;
                Some(Self::Int(v.round() as i64))
            }
            (Self::Vec2(ax, ay), Self::Vec2(bx, by)) => {
                Some(Self::Vec2(ax + (bx - ax) * t, ay + (by - ay) * t))
            }
            (Self::Color(ar, ag, ab, aa), Self::Color(br, bg, bb, ba)) => Some(Self::Color(
                ar + (br - ar) * t,
                ag + (bg - ag) * t,
                ab + (bb - ab) * t,
                aa + (ba - aa) * t,
            )),
            _ => None,
        }
    }
}

/// Category tag for organising presets in a library.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PresetCategory {
    /// Colour grading presets.
    ColorGrade,
    /// Blur / defocus presets.
    Blur,
    /// Distortion presets.
    Distortion,
    /// Transition presets.
    Transition,
    /// Particle / generator presets.
    Particle,
    /// Keying presets.
    Keying,
    /// Lighting presets.
    Lighting,
    /// Miscellaneous / user-created.
    Custom,
}

impl PresetCategory {
    /// Return a human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            Self::ColorGrade => "Color Grade",
            Self::Blur => "Blur",
            Self::Distortion => "Distortion",
            Self::Transition => "Transition",
            Self::Particle => "Particle",
            Self::Keying => "Keying",
            Self::Lighting => "Lighting",
            Self::Custom => "Custom",
        }
    }
}

// ---------------------------------------------------------------------------
// Preset
// ---------------------------------------------------------------------------

/// A named, categorised collection of parameter values.
#[derive(Debug, Clone)]
pub struct VfxPreset {
    /// Unique name.
    pub name: String,
    /// Category tag.
    pub category: PresetCategory,
    /// Optional description.
    pub description: String,
    /// Parameter map (key -> value).
    params: BTreeMap<String, PresetValue>,
}

impl VfxPreset {
    /// Create a new empty preset.
    pub fn new(name: impl Into<String>, category: PresetCategory) -> Self {
        Self {
            name: name.into(),
            category,
            description: String::new(),
            params: BTreeMap::new(),
        }
    }

    /// Attach a description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Set a parameter value.
    pub fn set(&mut self, key: impl Into<String>, value: PresetValue) {
        self.params.insert(key.into(), value);
    }

    /// Get a parameter value.
    pub fn get(&self, key: &str) -> Option<&PresetValue> {
        self.params.get(key)
    }

    /// Remove a parameter. Returns the old value if present.
    pub fn remove(&mut self, key: &str) -> Option<PresetValue> {
        self.params.remove(key)
    }

    /// Number of parameters.
    pub fn param_count(&self) -> usize {
        self.params.len()
    }

    /// Return sorted parameter keys.
    pub fn keys(&self) -> Vec<String> {
        self.params.keys().cloned().collect()
    }

    /// Check if a parameter exists.
    pub fn contains(&self, key: &str) -> bool {
        self.params.contains_key(key)
    }

    /// Interpolate between `self` and `other`.
    ///
    /// Only parameters present in **both** presets and of the same variant
    /// will be interpolated. The result inherits the name of `self`.
    pub fn lerp(&self, other: &Self, t: f64) -> Self {
        let mut result = Self::new(self.name.clone(), self.category);
        result.description = self.description.clone();

        for (key, a_val) in &self.params {
            if let Some(b_val) = other.params.get(key) {
                if let Some(interp) = a_val.lerp(b_val, t) {
                    result.params.insert(key.clone(), interp);
                } else {
                    // not interpolatable — keep A's value
                    result.params.insert(key.clone(), a_val.clone());
                }
            } else {
                result.params.insert(key.clone(), a_val.clone());
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Preset library
// ---------------------------------------------------------------------------

/// In-memory library of presets.
#[derive(Debug, Default)]
pub struct PresetLibrary {
    presets: Vec<VfxPreset>,
}

impl PresetLibrary {
    /// Create an empty library.
    pub fn new() -> Self {
        Self {
            presets: Vec::new(),
        }
    }

    /// Add a preset. Overwrites any existing preset with the same name.
    pub fn add(&mut self, preset: VfxPreset) {
        if let Some(pos) = self.presets.iter().position(|p| p.name == preset.name) {
            self.presets[pos] = preset;
        } else {
            self.presets.push(preset);
        }
    }

    /// Remove a preset by name. Returns `true` if found.
    pub fn remove(&mut self, name: &str) -> bool {
        let before = self.presets.len();
        self.presets.retain(|p| p.name != name);
        self.presets.len() < before
    }

    /// Find a preset by name.
    pub fn find(&self, name: &str) -> Option<&VfxPreset> {
        self.presets.iter().find(|p| p.name == name)
    }

    /// List all presets in a given category.
    pub fn by_category(&self, cat: PresetCategory) -> Vec<&VfxPreset> {
        self.presets.iter().filter(|p| p.category == cat).collect()
    }

    /// Total number of presets.
    pub fn len(&self) -> usize {
        self.presets.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.presets.is_empty()
    }

    /// Return all preset names.
    pub fn names(&self) -> Vec<String> {
        self.presets.iter().map(|p| p.name.clone()).collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_preset(name: &str) -> VfxPreset {
        let mut p = VfxPreset::new(name, PresetCategory::ColorGrade);
        p.set("brightness", PresetValue::Float(0.5));
        p.set("contrast", PresetValue::Float(1.2));
        p.set("enabled", PresetValue::Bool(true));
        p.set("label", PresetValue::Text("default".into()));
        p
    }

    #[test]
    fn test_preset_value_as_float() {
        assert_eq!(PresetValue::Float(1.5).as_float(), Some(1.5));
        assert_eq!(PresetValue::Int(3).as_float(), Some(3.0));
        assert!(PresetValue::Bool(true).as_float().is_none());
    }

    #[test]
    fn test_preset_value_as_int() {
        assert_eq!(PresetValue::Int(42).as_int(), Some(42));
        assert!(PresetValue::Float(1.0).as_int().is_none());
    }

    #[test]
    fn test_preset_value_as_bool() {
        assert_eq!(PresetValue::Bool(false).as_bool(), Some(false));
        assert!(PresetValue::Float(1.0).as_bool().is_none());
    }

    #[test]
    fn test_preset_value_as_text() {
        assert_eq!(PresetValue::Text("hello".into()).as_text(), Some("hello"));
        assert!(PresetValue::Int(0).as_text().is_none());
    }

    #[test]
    fn test_preset_value_lerp_float() {
        let a = PresetValue::Float(0.0);
        let b = PresetValue::Float(10.0);
        let mid = a.lerp(&b, 0.5).expect("should succeed in test");
        assert_eq!(mid, PresetValue::Float(5.0));
    }

    #[test]
    fn test_preset_value_lerp_color() {
        let a = PresetValue::Color(0.0, 0.0, 0.0, 1.0);
        let b = PresetValue::Color(1.0, 1.0, 1.0, 1.0);
        if let Some(PresetValue::Color(r, g, _, _)) = a.lerp(&b, 0.5) {
            assert!((r - 0.5).abs() < 1e-9);
            assert!((g - 0.5).abs() < 1e-9);
        } else {
            panic!("expected Color");
        }
    }

    #[test]
    fn test_preset_value_lerp_mismatch() {
        let a = PresetValue::Float(1.0);
        let b = PresetValue::Bool(true);
        assert!(a.lerp(&b, 0.5).is_none());
    }

    #[test]
    fn test_preset_set_get() {
        let p = make_preset("test");
        assert_eq!(p.param_count(), 4);
        assert!(p.contains("brightness"));
        assert!(!p.contains("missing"));
    }

    #[test]
    fn test_preset_remove() {
        let mut p = make_preset("test");
        let old = p.remove("enabled");
        assert!(old.is_some());
        assert_eq!(p.param_count(), 3);
    }

    #[test]
    fn test_preset_lerp() {
        let mut a = VfxPreset::new("a", PresetCategory::Blur);
        a.set("radius", PresetValue::Float(0.0));
        let mut b = VfxPreset::new("b", PresetCategory::Blur);
        b.set("radius", PresetValue::Float(10.0));

        let mid = a.lerp(&b, 0.5);
        assert_eq!(mid.get("radius").and_then(PresetValue::as_float), Some(5.0));
    }

    #[test]
    fn test_preset_category_label() {
        assert_eq!(PresetCategory::ColorGrade.label(), "Color Grade");
        assert_eq!(PresetCategory::Custom.label(), "Custom");
    }

    #[test]
    fn test_library_add_find() {
        let mut lib = PresetLibrary::new();
        lib.add(make_preset("warm_grade"));
        assert_eq!(lib.len(), 1);
        assert!(lib.find("warm_grade").is_some());
        assert!(lib.find("nonexistent").is_none());
    }

    #[test]
    fn test_library_overwrite() {
        let mut lib = PresetLibrary::new();
        let mut p1 = make_preset("grade");
        p1.set("brightness", PresetValue::Float(0.5));
        lib.add(p1);

        let mut p2 = make_preset("grade");
        p2.set("brightness", PresetValue::Float(0.9));
        lib.add(p2);

        assert_eq!(lib.len(), 1);
        let found = lib.find("grade").expect("should succeed in test");
        assert_eq!(
            found.get("brightness").and_then(PresetValue::as_float),
            Some(0.9)
        );
    }

    #[test]
    fn test_library_remove() {
        let mut lib = PresetLibrary::new();
        lib.add(make_preset("x"));
        assert!(lib.remove("x"));
        assert!(!lib.remove("x"));
        assert!(lib.is_empty());
    }

    #[test]
    fn test_library_by_category() {
        let mut lib = PresetLibrary::new();
        lib.add(VfxPreset::new("a", PresetCategory::Blur));
        lib.add(VfxPreset::new("b", PresetCategory::Blur));
        lib.add(VfxPreset::new("c", PresetCategory::Keying));
        let blurs = lib.by_category(PresetCategory::Blur);
        assert_eq!(blurs.len(), 2);
    }

    #[test]
    fn test_library_names() {
        let mut lib = PresetLibrary::new();
        lib.add(VfxPreset::new("alpha", PresetCategory::Custom));
        lib.add(VfxPreset::new("beta", PresetCategory::Custom));
        let names = lib.names();
        assert!(names.contains(&"alpha".to_string()));
        assert!(names.contains(&"beta".to_string()));
    }
}
