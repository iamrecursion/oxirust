#![allow(dead_code)]
//! Preset conflict resolution, inheritance, and merging.
//!
//! Provides tools for resolving conflicts when multiple presets overlap,
//! building preset inheritance chains, and merging preset configurations
//! with well-defined priority rules.

use std::collections::HashMap;

/// Priority level used to resolve conflicts between presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Priority {
    /// Lowest priority — used as a fallback default.
    Default,
    /// User-defined preset.
    User,
    /// Project-level override.
    Project,
    /// Platform-mandated constraint.
    Platform,
    /// Highest priority — system-enforced requirement.
    System,
}

impl Priority {
    /// Return a human-readable name for the priority level.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Priority::Default => "default",
            Priority::User => "user",
            Priority::Project => "project",
            Priority::Platform => "platform",
            Priority::System => "system",
        }
    }

    /// Check if this priority overrides another.
    #[must_use]
    pub fn overrides(&self, other: &Self) -> bool {
        self > other
    }
}

/// A single preset field value with its origin and priority.
#[derive(Debug, Clone)]
pub struct ResolvedField {
    /// Name of the field.
    pub name: String,
    /// Resolved string value.
    pub value: String,
    /// Priority of the source that provided this value.
    pub priority: Priority,
    /// Identifier of the source preset.
    pub source: String,
}

impl ResolvedField {
    /// Create a new resolved field.
    #[must_use]
    pub fn new(name: &str, value: &str, priority: Priority, source: &str) -> Self {
        Self {
            name: name.to_string(),
            value: value.to_string(),
            priority,
            source: source.to_string(),
        }
    }
}

/// Result of resolving conflicts across multiple preset sources.
#[derive(Debug, Clone)]
pub struct ResolvedPreset {
    /// Resolved fields keyed by field name.
    fields: HashMap<String, ResolvedField>,
    /// Conflicts detected during resolution.
    conflicts: Vec<ConflictRecord>,
}

impl ResolvedPreset {
    /// Create a new empty resolved preset.
    #[must_use]
    pub fn new() -> Self {
        Self {
            fields: HashMap::new(),
            conflicts: Vec::new(),
        }
    }

    /// Insert or update a field. If a conflict arises, the higher-priority value wins.
    pub fn insert(&mut self, field: ResolvedField) {
        if let Some(existing) = self.fields.get(&field.name) {
            if field.priority > existing.priority {
                self.conflicts.push(ConflictRecord {
                    field_name: field.name.clone(),
                    winner_source: field.source.clone(),
                    winner_value: field.value.clone(),
                    loser_source: existing.source.clone(),
                    loser_value: existing.value.clone(),
                });
                self.fields.insert(field.name.clone(), field);
            } else if field.priority < existing.priority {
                self.conflicts.push(ConflictRecord {
                    field_name: field.name.clone(),
                    winner_source: existing.source.clone(),
                    winner_value: existing.value.clone(),
                    loser_source: field.source.clone(),
                    loser_value: field.value.clone(),
                });
            }
            // Equal priority, first-writer wins (no-op)
        } else {
            self.fields.insert(field.name.clone(), field);
        }
    }

    /// Get a resolved field value.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&ResolvedField> {
        self.fields.get(name)
    }

    /// Get the resolved value as a string.
    #[must_use]
    pub fn get_value(&self, name: &str) -> Option<&str> {
        self.fields.get(name).map(|f| f.value.as_str())
    }

    /// Get all resolved field names.
    #[must_use]
    pub fn field_names(&self) -> Vec<&str> {
        self.fields.keys().map(String::as_str).collect()
    }

    /// Get the number of resolved fields.
    #[must_use]
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }

    /// Get the detected conflicts.
    #[must_use]
    pub fn conflicts(&self) -> &[ConflictRecord] {
        &self.conflicts
    }

    /// Check if there were any conflicts during resolution.
    #[must_use]
    pub fn has_conflicts(&self) -> bool {
        !self.conflicts.is_empty()
    }
}

impl Default for ResolvedPreset {
    fn default() -> Self {
        Self::new()
    }
}

/// Record of a conflict that was resolved during merging.
#[derive(Debug, Clone)]
pub struct ConflictRecord {
    /// Name of the conflicting field.
    pub field_name: String,
    /// Source that won the conflict.
    pub winner_source: String,
    /// Value that won.
    pub winner_value: String,
    /// Source that lost the conflict.
    pub loser_source: String,
    /// Value that lost.
    pub loser_value: String,
}

/// A preset source that provides field values at a given priority.
#[derive(Debug, Clone)]
pub struct PresetSource {
    /// Identifier for this source.
    pub id: String,
    /// Priority level.
    pub priority: Priority,
    /// Fields provided by this source.
    pub fields: HashMap<String, String>,
}

impl PresetSource {
    /// Create a new preset source.
    #[must_use]
    pub fn new(id: &str, priority: Priority) -> Self {
        Self {
            id: id.to_string(),
            priority,
            fields: HashMap::new(),
        }
    }

    /// Add a field to this source.
    #[must_use]
    pub fn with_field(mut self, name: &str, value: &str) -> Self {
        self.fields.insert(name.to_string(), value.to_string());
        self
    }

    /// Get the number of fields in this source.
    #[must_use]
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }
}

/// Resolver that merges multiple preset sources into a single resolved preset.
pub struct PresetResolver {
    sources: Vec<PresetSource>,
}

impl PresetResolver {
    /// Create a new resolver.
    #[must_use]
    pub fn new() -> Self {
        Self {
            sources: Vec::new(),
        }
    }

    /// Add a preset source to be resolved.
    pub fn add_source(&mut self, source: PresetSource) {
        self.sources.push(source);
    }

    /// Resolve all sources into a single preset with priority-based conflict resolution.
    #[must_use]
    pub fn resolve(&self) -> ResolvedPreset {
        let mut result = ResolvedPreset::new();

        // Process sources sorted by priority (lowest first so higher overrides)
        let mut sorted: Vec<&PresetSource> = self.sources.iter().collect();
        sorted.sort_by_key(|s| s.priority);

        for source in sorted {
            for (name, value) in &source.fields {
                result.insert(ResolvedField::new(name, value, source.priority, &source.id));
            }
        }

        result
    }

    /// Get the number of sources.
    #[must_use]
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    /// Clear all sources.
    pub fn clear(&mut self) {
        self.sources.clear();
    }
}

impl Default for PresetResolver {
    fn default() -> Self {
        Self::new()
    }
}

/// Inheritance chain link.
#[derive(Debug, Clone)]
pub struct InheritanceLink {
    /// Child preset identifier.
    pub child_id: String,
    /// Parent preset identifier.
    pub parent_id: String,
}

/// Manages preset inheritance relationships.
pub struct InheritanceChain {
    /// Links from child to parent.
    links: Vec<InheritanceLink>,
}

impl InheritanceChain {
    /// Create a new empty inheritance chain.
    #[must_use]
    pub fn new() -> Self {
        Self { links: Vec::new() }
    }

    /// Add a parent-child relationship.
    pub fn add_link(&mut self, child_id: &str, parent_id: &str) {
        self.links.push(InheritanceLink {
            child_id: child_id.to_string(),
            parent_id: parent_id.to_string(),
        });
    }

    /// Get the parent of a given preset.
    #[must_use]
    pub fn parent_of(&self, child_id: &str) -> Option<&str> {
        self.links
            .iter()
            .find(|l| l.child_id == child_id)
            .map(|l| l.parent_id.as_str())
    }

    /// Get all children of a given parent.
    #[must_use]
    pub fn children_of(&self, parent_id: &str) -> Vec<&str> {
        self.links
            .iter()
            .filter(|l| l.parent_id == parent_id)
            .map(|l| l.child_id.as_str())
            .collect()
    }

    /// Build the full ancestor chain for a preset (child first, root last).
    #[must_use]
    pub fn ancestor_chain(&self, child_id: &str) -> Vec<&str> {
        let mut chain = Vec::new();
        let mut current = child_id;
        // Guard against cycles: max depth
        for _ in 0..100 {
            if let Some(parent) = self.parent_of(current) {
                chain.push(parent);
                current = parent;
            } else {
                break;
            }
        }
        chain
    }

    /// Check if a preset has a specific ancestor.
    #[must_use]
    pub fn has_ancestor(&self, child_id: &str, ancestor_id: &str) -> bool {
        self.ancestor_chain(child_id)
            .iter()
            .any(|a| *a == ancestor_id)
    }

    /// Get the depth of a preset in the inheritance tree.
    #[must_use]
    pub fn depth(&self, child_id: &str) -> usize {
        self.ancestor_chain(child_id).len()
    }

    /// Get the number of links.
    #[must_use]
    pub fn link_count(&self) -> usize {
        self.links.len()
    }
}

impl Default for InheritanceChain {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::System > Priority::Platform);
        assert!(Priority::Platform > Priority::Project);
        assert!(Priority::Project > Priority::User);
        assert!(Priority::User > Priority::Default);
    }

    #[test]
    fn test_priority_overrides() {
        assert!(Priority::System.overrides(&Priority::User));
        assert!(!Priority::Default.overrides(&Priority::User));
    }

    #[test]
    fn test_priority_name() {
        assert_eq!(Priority::Default.name(), "default");
        assert_eq!(Priority::System.name(), "system");
    }

    #[test]
    fn test_resolved_preset_insert_no_conflict() {
        let mut rp = ResolvedPreset::new();
        rp.insert(ResolvedField::new(
            "bitrate",
            "5000000",
            Priority::User,
            "user-preset",
        ));
        assert_eq!(rp.get_value("bitrate"), Some("5000000"));
        assert!(!rp.has_conflicts());
    }

    #[test]
    fn test_resolved_preset_higher_priority_wins() {
        let mut rp = ResolvedPreset::new();
        rp.insert(ResolvedField::new(
            "codec",
            "h264",
            Priority::Default,
            "default",
        ));
        rp.insert(ResolvedField::new(
            "codec",
            "h265",
            Priority::Platform,
            "platform",
        ));
        assert_eq!(rp.get_value("codec"), Some("h265"));
        assert!(rp.has_conflicts());
        assert_eq!(rp.conflicts().len(), 1);
    }

    #[test]
    fn test_resolved_preset_lower_priority_loses() {
        let mut rp = ResolvedPreset::new();
        rp.insert(ResolvedField::new(
            "codec",
            "h265",
            Priority::System,
            "system",
        ));
        rp.insert(ResolvedField::new("codec", "h264", Priority::User, "user"));
        assert_eq!(rp.get_value("codec"), Some("h265"));
        assert!(rp.has_conflicts());
    }

    #[test]
    fn test_preset_source_builder() {
        let source = PresetSource::new("my-source", Priority::User)
            .with_field("codec", "h264")
            .with_field("bitrate", "5000000");
        assert_eq!(source.field_count(), 2);
    }

    #[test]
    fn test_resolver_basic() {
        let mut resolver = PresetResolver::new();
        resolver.add_source(
            PresetSource::new("defaults", Priority::Default)
                .with_field("codec", "h264")
                .with_field("bitrate", "3000000"),
        );
        resolver.add_source(PresetSource::new("user", Priority::User).with_field("codec", "h265"));

        let resolved = resolver.resolve();
        assert_eq!(resolved.get_value("codec"), Some("h265"));
        assert_eq!(resolved.get_value("bitrate"), Some("3000000"));
        assert_eq!(resolved.field_count(), 2);
    }

    #[test]
    fn test_resolver_three_layers() {
        let mut resolver = PresetResolver::new();
        resolver
            .add_source(PresetSource::new("default", Priority::Default).with_field("fps", "30"));
        resolver.add_source(PresetSource::new("user", Priority::User).with_field("fps", "60"));
        resolver
            .add_source(PresetSource::new("platform", Priority::Platform).with_field("fps", "24"));

        let resolved = resolver.resolve();
        assert_eq!(resolved.get_value("fps"), Some("24"));
    }

    #[test]
    fn test_inheritance_chain_basic() {
        let mut chain = InheritanceChain::new();
        chain.add_link("child", "parent");
        chain.add_link("parent", "grandparent");

        assert_eq!(chain.parent_of("child"), Some("parent"));
        assert_eq!(chain.parent_of("parent"), Some("grandparent"));
        assert_eq!(chain.parent_of("grandparent"), None);
    }

    #[test]
    fn test_inheritance_chain_ancestors() {
        let mut chain = InheritanceChain::new();
        chain.add_link("c", "b");
        chain.add_link("b", "a");

        let ancestors = chain.ancestor_chain("c");
        assert_eq!(ancestors, vec!["b", "a"]);
    }

    #[test]
    fn test_inheritance_has_ancestor() {
        let mut chain = InheritanceChain::new();
        chain.add_link("c", "b");
        chain.add_link("b", "a");

        assert!(chain.has_ancestor("c", "a"));
        assert!(chain.has_ancestor("c", "b"));
        assert!(!chain.has_ancestor("a", "c"));
    }

    #[test]
    fn test_inheritance_depth() {
        let mut chain = InheritanceChain::new();
        chain.add_link("level3", "level2");
        chain.add_link("level2", "level1");
        chain.add_link("level1", "root");

        assert_eq!(chain.depth("level3"), 3);
        assert_eq!(chain.depth("root"), 0);
    }

    #[test]
    fn test_inheritance_children_of() {
        let mut chain = InheritanceChain::new();
        chain.add_link("child1", "parent");
        chain.add_link("child2", "parent");
        chain.add_link("other", "different");

        let children = chain.children_of("parent");
        assert_eq!(children.len(), 2);
        assert!(children.contains(&"child1"));
        assert!(children.contains(&"child2"));
    }

    #[test]
    fn test_resolver_clear() {
        let mut resolver = PresetResolver::new();
        resolver.add_source(PresetSource::new("s1", Priority::Default));
        assert_eq!(resolver.source_count(), 1);
        resolver.clear();
        assert_eq!(resolver.source_count(), 0);
    }
}
