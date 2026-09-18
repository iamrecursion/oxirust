#![allow(dead_code)]
//! Transition definitions for AAF compositions.
//!
//! Transitions describe how one segment blends into the next.  The AAF
//! specification supports a variety of transition types (dissolves, wipes,
//! SMPTE wipe patterns, etc.).  This module models those types with
//! [`TransitionType`], wraps each instance in a [`TransitionDef`], and
//! provides [`TransitionCatalog`] for managing collections of definitions.

use std::collections::HashMap;
use uuid::Uuid;

/// Enumeration of supported AAF transition types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransitionType {
    /// Cross-dissolve (linear blend from outgoing to incoming).
    Dissolve,
    /// Additive dissolve (additive blend during overlap).
    AddDissolve,
    /// SMPTE wipe identified by a numeric pattern code.
    SmpteWipe(u16),
    /// Dip to colour (e.g. dip-to-black, dip-to-white).
    DipToColor,
    /// Custom / plug-in transition identified by a UUID.
    Custom(Uuid),
}

impl TransitionType {
    /// Human-readable label.
    #[must_use]
    pub fn label(&self) -> String {
        match self {
            Self::Dissolve => "Dissolve".to_string(),
            Self::AddDissolve => "Additive Dissolve".to_string(),
            Self::SmpteWipe(code) => format!("SMPTE Wipe {code}"),
            Self::DipToColor => "Dip to Color".to_string(),
            Self::Custom(id) => format!("Custom ({id})"),
        }
    }

    /// Returns `true` for dissolve variants.
    #[must_use]
    pub const fn is_dissolve(&self) -> bool {
        matches!(self, Self::Dissolve | Self::AddDissolve)
    }

    /// Returns `true` for wipe variants.
    #[must_use]
    pub const fn is_wipe(&self) -> bool {
        matches!(self, Self::SmpteWipe(_))
    }
}

impl std::fmt::Display for TransitionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.label())
    }
}

/// A transition definition that pairs a [`TransitionType`] with duration
/// and optional parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct TransitionDef {
    /// Unique id for this definition.
    id: Uuid,
    /// The kind of transition.
    transition_type: TransitionType,
    /// Duration of the transition in edit units.
    duration: i64,
    /// Edit rate numerator.
    edit_rate_num: u32,
    /// Edit rate denominator.
    edit_rate_den: u32,
    /// Optional human-readable name.
    name: Option<String>,
    /// Is the transition reversed (outgoing becomes incoming)?
    reversed: bool,
}

impl TransitionDef {
    /// Create a new transition definition.
    #[must_use]
    pub fn new(
        transition_type: TransitionType,
        duration: i64,
        edit_rate_num: u32,
        edit_rate_den: u32,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            transition_type,
            duration,
            edit_rate_num,
            edit_rate_den,
            name: None,
            reversed: false,
        }
    }

    /// Builder: set a fixed id.
    #[must_use]
    pub fn with_id(mut self, id: Uuid) -> Self {
        self.id = id;
        self
    }

    /// Builder: set a name.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Builder: mark the transition as reversed.
    #[must_use]
    pub fn reversed(mut self) -> Self {
        self.reversed = true;
        self
    }

    /// Unique identifier.
    #[must_use]
    pub fn id(&self) -> Uuid {
        self.id
    }

    /// Transition type.
    #[must_use]
    pub fn transition_type(&self) -> &TransitionType {
        &self.transition_type
    }

    /// Duration in edit units.
    #[must_use]
    pub fn duration(&self) -> i64 {
        self.duration
    }

    /// Edit rate as `(num, den)`.
    #[must_use]
    pub fn edit_rate(&self) -> (u32, u32) {
        (self.edit_rate_num, self.edit_rate_den)
    }

    /// Optional name.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Whether the transition is reversed.
    #[must_use]
    pub fn is_reversed(&self) -> bool {
        self.reversed
    }

    /// Duration in seconds.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn duration_seconds(&self) -> f64 {
        if self.edit_rate_den == 0 || self.edit_rate_num == 0 {
            return 0.0;
        }
        let rate = self.edit_rate_num as f64 / self.edit_rate_den as f64;
        self.duration as f64 / rate
    }
}

/// A catalog that stores [`TransitionDef`] instances keyed by their id.
#[derive(Debug, Clone)]
pub struct TransitionCatalog {
    entries: HashMap<Uuid, TransitionDef>,
}

impl TransitionCatalog {
    /// Create an empty catalog.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Insert a transition definition, returning any previous definition with the same id.
    pub fn insert(&mut self, def: TransitionDef) -> Option<TransitionDef> {
        self.entries.insert(def.id(), def)
    }

    /// Remove a definition by id.
    pub fn remove(&mut self, id: &Uuid) -> Option<TransitionDef> {
        self.entries.remove(id)
    }

    /// Look up a definition by id.
    #[must_use]
    pub fn get(&self, id: &Uuid) -> Option<&TransitionDef> {
        self.entries.get(id)
    }

    /// Number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the catalog is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return all definitions of a given transition type.
    #[must_use]
    pub fn find_by_type(&self, tt: &TransitionType) -> Vec<&TransitionDef> {
        self.entries
            .values()
            .filter(|d| &d.transition_type == tt)
            .collect()
    }

    /// Return all dissolve definitions.
    #[must_use]
    pub fn dissolves(&self) -> Vec<&TransitionDef> {
        self.entries
            .values()
            .filter(|d| d.transition_type.is_dissolve())
            .collect()
    }

    /// Return all wipe definitions.
    #[must_use]
    pub fn wipes(&self) -> Vec<&TransitionDef> {
        self.entries
            .values()
            .filter(|d| d.transition_type.is_wipe())
            .collect()
    }

    /// Iterator over all definitions.
    pub fn iter(&self) -> impl Iterator<Item = &TransitionDef> {
        self.entries.values()
    }
}

impl Default for TransitionCatalog {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transition_type_label() {
        assert_eq!(TransitionType::Dissolve.label(), "Dissolve");
        assert_eq!(TransitionType::AddDissolve.label(), "Additive Dissolve");
        assert_eq!(TransitionType::SmpteWipe(1).label(), "SMPTE Wipe 1");
        assert_eq!(TransitionType::DipToColor.label(), "Dip to Color");
    }

    #[test]
    fn test_transition_type_is_dissolve() {
        assert!(TransitionType::Dissolve.is_dissolve());
        assert!(TransitionType::AddDissolve.is_dissolve());
        assert!(!TransitionType::SmpteWipe(4).is_dissolve());
        assert!(!TransitionType::DipToColor.is_dissolve());
    }

    #[test]
    fn test_transition_type_is_wipe() {
        assert!(TransitionType::SmpteWipe(1).is_wipe());
        assert!(!TransitionType::Dissolve.is_wipe());
    }

    #[test]
    fn test_transition_type_display() {
        let t = TransitionType::SmpteWipe(42);
        assert_eq!(format!("{t}"), "SMPTE Wipe 42");
    }

    #[test]
    fn test_transition_def_creation() {
        let def = TransitionDef::new(TransitionType::Dissolve, 30, 30, 1);
        assert_eq!(*def.transition_type(), TransitionType::Dissolve);
        assert_eq!(def.duration(), 30);
        assert_eq!(def.edit_rate(), (30, 1));
        assert!(!def.is_reversed());
        assert!(def.name().is_none());
    }

    #[test]
    fn test_transition_def_builders() {
        let id = Uuid::new_v4();
        let def = TransitionDef::new(TransitionType::SmpteWipe(1), 15, 25, 1)
            .with_id(id)
            .with_name("wipe_lr")
            .reversed();
        assert_eq!(def.id(), id);
        assert_eq!(def.name(), Some("wipe_lr"));
        assert!(def.is_reversed());
    }

    #[test]
    fn test_transition_def_duration_seconds() {
        let def = TransitionDef::new(TransitionType::Dissolve, 50, 25, 1);
        let dur = def.duration_seconds();
        assert!((dur - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_transition_def_duration_zero_rate() {
        let def = TransitionDef::new(TransitionType::Dissolve, 50, 0, 0);
        assert_eq!(def.duration_seconds(), 0.0);
    }

    #[test]
    fn test_catalog_insert_get_remove() {
        let mut cat = TransitionCatalog::new();
        let def = TransitionDef::new(TransitionType::Dissolve, 30, 25, 1);
        let id = def.id();
        cat.insert(def);
        assert_eq!(cat.len(), 1);
        assert!(cat.get(&id).is_some());
        assert!(cat.remove(&id).is_some());
        assert!(cat.is_empty());
    }

    #[test]
    fn test_catalog_find_by_type() {
        let mut cat = TransitionCatalog::new();
        cat.insert(TransitionDef::new(TransitionType::Dissolve, 30, 25, 1));
        cat.insert(TransitionDef::new(TransitionType::SmpteWipe(1), 15, 25, 1));
        cat.insert(TransitionDef::new(TransitionType::Dissolve, 20, 25, 1));

        let dissolves = cat.find_by_type(&TransitionType::Dissolve);
        assert_eq!(dissolves.len(), 2);
    }

    #[test]
    fn test_catalog_dissolves_and_wipes() {
        let mut cat = TransitionCatalog::new();
        cat.insert(TransitionDef::new(TransitionType::Dissolve, 30, 25, 1));
        cat.insert(TransitionDef::new(TransitionType::AddDissolve, 20, 25, 1));
        cat.insert(TransitionDef::new(TransitionType::SmpteWipe(1), 15, 25, 1));
        cat.insert(TransitionDef::new(TransitionType::SmpteWipe(4), 10, 25, 1));
        cat.insert(TransitionDef::new(TransitionType::DipToColor, 25, 25, 1));

        assert_eq!(cat.dissolves().len(), 2);
        assert_eq!(cat.wipes().len(), 2);
    }

    #[test]
    fn test_catalog_iter() {
        let mut cat = TransitionCatalog::new();
        cat.insert(TransitionDef::new(TransitionType::Dissolve, 30, 25, 1));
        cat.insert(TransitionDef::new(TransitionType::SmpteWipe(1), 15, 25, 1));
        assert_eq!(cat.iter().count(), 2);
    }

    #[test]
    fn test_catalog_default() {
        let cat = TransitionCatalog::default();
        assert!(cat.is_empty());
    }

    #[test]
    fn test_transition_type_custom() {
        let custom_id = Uuid::new_v4();
        let t = TransitionType::Custom(custom_id);
        assert!(!t.is_dissolve());
        assert!(!t.is_wipe());
        assert!(t.label().contains("Custom"));
    }
}
