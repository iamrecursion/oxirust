#![allow(dead_code)]
//! Named routing presets for rapid recall of complete routing configurations.
//!
//! A [`RoutePreset`] stores a snapshot of source-to-destination mappings,
//! gain values, and optional metadata. Presets can be grouped into banks,
//! recalled, compared, and merged.

use std::collections::HashMap;
use std::fmt;

/// A single source-to-destination mapping entry.
#[derive(Debug, Clone, PartialEq)]
pub struct RouteMapping {
    /// Source identifier.
    pub source: u32,
    /// Destination identifier.
    pub destination: u32,
    /// Gain in dB applied to this route (0.0 = unity).
    pub gain_db: f32,
    /// Whether this mapping is currently enabled.
    pub enabled: bool,
}

impl RouteMapping {
    /// Create a new enabled mapping at unity gain.
    pub fn new(source: u32, destination: u32) -> Self {
        Self {
            source,
            destination,
            gain_db: 0.0,
            enabled: true,
        }
    }

    /// Create a mapping with a specified gain.
    pub fn with_gain(source: u32, destination: u32, gain_db: f32) -> Self {
        Self {
            source,
            destination,
            gain_db,
            enabled: true,
        }
    }
}

impl fmt::Display for RouteMapping {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = if self.enabled { "ON" } else { "OFF" };
        write!(
            f,
            "S{} -> D{} ({:+.1}dB) [{}]",
            self.source, self.destination, self.gain_db, state
        )
    }
}

/// A named routing preset.
#[derive(Debug, Clone)]
pub struct RoutePreset {
    /// Preset name.
    pub name: String,
    /// Description.
    pub description: String,
    /// Route mappings stored in this preset.
    mappings: Vec<RouteMapping>,
    /// Unix timestamp of creation (seconds since epoch).
    pub created_at: u64,
    /// Unix timestamp of last modification.
    pub modified_at: u64,
    /// Arbitrary key-value tags.
    tags: HashMap<String, String>,
}

impl RoutePreset {
    /// Create a new empty preset with the given name.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            description: String::new(),
            mappings: Vec::new(),
            created_at: 0,
            modified_at: 0,
            tags: HashMap::new(),
        }
    }

    /// Create with name and description.
    pub fn with_description(name: &str, description: &str) -> Self {
        Self {
            name: name.to_owned(),
            description: description.to_owned(),
            mappings: Vec::new(),
            created_at: 0,
            modified_at: 0,
            tags: HashMap::new(),
        }
    }

    /// Add a mapping to the preset.
    pub fn add_mapping(&mut self, mapping: RouteMapping) {
        self.mappings.push(mapping);
    }

    /// Remove mappings for a given destination.
    pub fn remove_destination(&mut self, dest: u32) {
        self.mappings.retain(|m| m.destination != dest);
    }

    /// Number of mappings.
    pub fn mapping_count(&self) -> usize {
        self.mappings.len()
    }

    /// Read-only access to mappings.
    pub fn mappings(&self) -> &[RouteMapping] {
        &self.mappings
    }

    /// Find mappings for a given source.
    pub fn mappings_for_source(&self, source: u32) -> Vec<&RouteMapping> {
        self.mappings
            .iter()
            .filter(|m| m.source == source)
            .collect()
    }

    /// Find mappings for a given destination.
    pub fn mappings_for_destination(&self, dest: u32) -> Vec<&RouteMapping> {
        self.mappings
            .iter()
            .filter(|m| m.destination == dest)
            .collect()
    }

    /// Set a tag.
    pub fn set_tag(&mut self, key: &str, value: &str) {
        self.tags.insert(key.to_owned(), value.to_owned());
    }

    /// Get a tag.
    pub fn tag(&self, key: &str) -> Option<&str> {
        self.tags.get(key).map(String::as_str)
    }

    /// Number of tags.
    pub fn tag_count(&self) -> usize {
        self.tags.len()
    }

    /// Unique source ids referenced.
    pub fn source_ids(&self) -> Vec<u32> {
        let mut ids: Vec<u32> = self.mappings.iter().map(|m| m.source).collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// Unique destination ids referenced.
    pub fn destination_ids(&self) -> Vec<u32> {
        let mut ids: Vec<u32> = self.mappings.iter().map(|m| m.destination).collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// Compare this preset to another and return the differences.
    pub fn diff(&self, other: &RoutePreset) -> Vec<String> {
        let mut diffs = Vec::new();

        // Build lookup from (src, dst) -> mapping for each
        let self_map: HashMap<(u32, u32), &RouteMapping> = self
            .mappings
            .iter()
            .map(|m| ((m.source, m.destination), m))
            .collect();
        let other_map: HashMap<(u32, u32), &RouteMapping> = other
            .mappings
            .iter()
            .map(|m| ((m.source, m.destination), m))
            .collect();

        for (key, sm) in &self_map {
            match other_map.get(key) {
                None => diffs.push(format!("Removed: {sm}")),
                Some(om) => {
                    if (sm.gain_db - om.gain_db).abs() > f32::EPSILON {
                        diffs.push(format!(
                            "Gain changed S{}->D{}: {:+.1}dB -> {:+.1}dB",
                            key.0, key.1, sm.gain_db, om.gain_db,
                        ));
                    }
                    if sm.enabled != om.enabled {
                        diffs.push(format!(
                            "Enable changed S{}->D{}: {} -> {}",
                            key.0, key.1, sm.enabled, om.enabled,
                        ));
                    }
                }
            }
        }
        for (key, om) in &other_map {
            if !self_map.contains_key(key) {
                diffs.push(format!("Added: {om}"));
            }
        }

        diffs.sort();
        diffs
    }

    /// Merge another preset's mappings into this one (other wins on conflict).
    pub fn merge_from(&mut self, other: &RoutePreset) {
        for om in &other.mappings {
            if let Some(existing) = self
                .mappings
                .iter_mut()
                .find(|m| m.source == om.source && m.destination == om.destination)
            {
                existing.gain_db = om.gain_db;
                existing.enabled = om.enabled;
            } else {
                self.mappings.push(om.clone());
            }
        }
    }
}

impl fmt::Display for RoutePreset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Preset '{}' ({} mappings)",
            self.name,
            self.mappings.len()
        )
    }
}

/// A bank of presets for organized recall.
#[derive(Debug, Clone)]
pub struct PresetBank {
    /// Bank name.
    pub name: String,
    /// Ordered list of presets in the bank.
    presets: Vec<RoutePreset>,
}

impl PresetBank {
    /// Create a new empty bank.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            presets: Vec::new(),
        }
    }

    /// Add a preset to the bank. Returns its index.
    pub fn add(&mut self, preset: RoutePreset) -> usize {
        let idx = self.presets.len();
        self.presets.push(preset);
        idx
    }

    /// Recall a preset by index.
    pub fn recall(&self, idx: usize) -> Option<&RoutePreset> {
        self.presets.get(idx)
    }

    /// Number of presets in the bank.
    pub fn len(&self) -> usize {
        self.presets.len()
    }

    /// Whether the bank is empty.
    pub fn is_empty(&self) -> bool {
        self.presets.is_empty()
    }

    /// Find presets by name (case-insensitive substring match).
    pub fn search(&self, query: &str) -> Vec<(usize, &RoutePreset)> {
        let q = query.to_lowercase();
        self.presets
            .iter()
            .enumerate()
            .filter(|(_, p)| p.name.to_lowercase().contains(&q))
            .collect()
    }

    /// Remove a preset by index. Returns it if found.
    pub fn remove(&mut self, idx: usize) -> Option<RoutePreset> {
        if idx < self.presets.len() {
            Some(self.presets.remove(idx))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mapping_new() {
        let m = RouteMapping::new(0, 1);
        assert_eq!(m.source, 0);
        assert_eq!(m.destination, 1);
        assert!((m.gain_db).abs() < f32::EPSILON);
        assert!(m.enabled);
    }

    #[test]
    fn test_mapping_with_gain() {
        let m = RouteMapping::with_gain(2, 3, -6.0);
        assert!((m.gain_db - (-6.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_mapping_display() {
        let m = RouteMapping::new(0, 1);
        let s = format!("{m}");
        assert!(s.contains("S0"));
        assert!(s.contains("D1"));
        assert!(s.contains("ON"));
    }

    #[test]
    fn test_preset_new() {
        let p = RoutePreset::new("Show A");
        assert_eq!(p.name, "Show A");
        assert_eq!(p.mapping_count(), 0);
    }

    #[test]
    fn test_preset_add_mapping() {
        let mut p = RoutePreset::new("P");
        p.add_mapping(RouteMapping::new(0, 0));
        p.add_mapping(RouteMapping::new(1, 1));
        assert_eq!(p.mapping_count(), 2);
    }

    #[test]
    fn test_preset_remove_destination() {
        let mut p = RoutePreset::new("P");
        p.add_mapping(RouteMapping::new(0, 0));
        p.add_mapping(RouteMapping::new(1, 0));
        p.add_mapping(RouteMapping::new(2, 1));
        p.remove_destination(0);
        assert_eq!(p.mapping_count(), 1);
    }

    #[test]
    fn test_preset_source_ids() {
        let mut p = RoutePreset::new("P");
        p.add_mapping(RouteMapping::new(3, 0));
        p.add_mapping(RouteMapping::new(1, 1));
        p.add_mapping(RouteMapping::new(3, 2));
        assert_eq!(p.source_ids(), vec![1, 3]);
    }

    #[test]
    fn test_preset_tags() {
        let mut p = RoutePreset::new("P");
        p.set_tag("venue", "Studio A");
        assert_eq!(p.tag("venue"), Some("Studio A"));
        assert_eq!(p.tag_count(), 1);
    }

    #[test]
    fn test_preset_diff_added() {
        let a = RoutePreset::new("A");
        let mut b = RoutePreset::new("B");
        b.add_mapping(RouteMapping::new(0, 0));
        let diffs = a.diff(&b);
        assert_eq!(diffs.len(), 1);
        assert!(diffs[0].contains("Added"));
    }

    #[test]
    fn test_preset_diff_removed() {
        let mut a = RoutePreset::new("A");
        a.add_mapping(RouteMapping::new(0, 0));
        let b = RoutePreset::new("B");
        let diffs = a.diff(&b);
        assert_eq!(diffs.len(), 1);
        assert!(diffs[0].contains("Removed"));
    }

    #[test]
    fn test_preset_diff_gain_change() {
        let mut a = RoutePreset::new("A");
        a.add_mapping(RouteMapping::with_gain(0, 0, 0.0));
        let mut b = RoutePreset::new("B");
        b.add_mapping(RouteMapping::with_gain(0, 0, -6.0));
        let diffs = a.diff(&b);
        assert!(diffs.iter().any(|d| d.contains("Gain changed")));
    }

    #[test]
    fn test_preset_merge() {
        let mut a = RoutePreset::new("A");
        a.add_mapping(RouteMapping::with_gain(0, 0, 0.0));
        let mut b = RoutePreset::new("B");
        b.add_mapping(RouteMapping::with_gain(0, 0, -3.0));
        b.add_mapping(RouteMapping::new(1, 1));
        a.merge_from(&b);
        assert_eq!(a.mapping_count(), 2);
        // gain should be updated
        let m = a
            .mappings()
            .iter()
            .find(|m| m.source == 0)
            .expect("should succeed in test");
        assert!((m.gain_db - (-3.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_preset_display() {
        let p = RoutePreset::new("MyPreset");
        let s = format!("{p}");
        assert!(s.contains("MyPreset"));
        assert!(s.contains("0 mappings"));
    }

    #[test]
    fn test_bank_add_recall() {
        let mut bank = PresetBank::new("Main");
        let idx = bank.add(RoutePreset::new("P1"));
        assert_eq!(idx, 0);
        assert_eq!(bank.len(), 1);
        assert!(!bank.is_empty());
        let p = bank.recall(idx).expect("should succeed in test");
        assert_eq!(p.name, "P1");
    }

    #[test]
    fn test_bank_search() {
        let mut bank = PresetBank::new("Main");
        bank.add(RoutePreset::new("Show Alpha"));
        bank.add(RoutePreset::new("Show Beta"));
        bank.add(RoutePreset::new("Rehearsal"));
        let results = bank.search("show");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_bank_remove() {
        let mut bank = PresetBank::new("Main");
        bank.add(RoutePreset::new("P1"));
        bank.add(RoutePreset::new("P2"));
        let removed = bank.remove(0);
        assert!(removed.is_some());
        assert_eq!(bank.len(), 1);
    }
}
