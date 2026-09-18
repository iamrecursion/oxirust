//! Subclip inheritance: propagating metadata from parent clips to subclips.

#![allow(dead_code)]

use std::collections::HashMap;

/// Metadata that can be inherited from a parent clip.
#[derive(Debug, Clone, PartialEq)]
pub struct InheritableMetadata {
    /// Keywords / tags associated with the clip.
    pub keywords: Vec<String>,
    /// Free-text description.
    pub description: Option<String>,
    /// Camera roll / reel name.
    pub reel: Option<String>,
    /// Scene identifier.
    pub scene: Option<String>,
    /// Take number.
    pub take: Option<u32>,
}

impl InheritableMetadata {
    /// Create a blank metadata record.
    #[must_use]
    pub fn new() -> Self {
        Self {
            keywords: Vec::new(),
            description: None,
            reel: None,
            scene: None,
            take: None,
        }
    }

    /// Merge another metadata record into this one.
    ///
    /// Fields in `other` override fields in `self`.  Keywords are unioned.
    pub fn merge_from(&mut self, other: &Self) {
        for kw in &other.keywords {
            if !self.keywords.contains(kw) {
                self.keywords.push(kw.clone());
            }
        }
        if other.description.is_some() {
            self.description = other.description.clone();
        }
        if other.reel.is_some() {
            self.reel = other.reel.clone();
        }
        if other.scene.is_some() {
            self.scene = other.scene.clone();
        }
        if other.take.is_some() {
            self.take = other.take;
        }
    }
}

impl Default for InheritableMetadata {
    fn default() -> Self {
        Self::new()
    }
}

/// Policy controlling which fields are inherited by subclips.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InheritPolicy {
    /// Inherit all fields from parent.
    All,
    /// Inherit no fields (subclip stands alone).
    None,
    /// Inherit keywords only.
    KeywordsOnly,
    /// Inherit scene/reel/take only.
    ProductionOnly,
}

/// Resolve the effective metadata for a subclip by applying the policy.
#[must_use]
pub fn resolve_inherited(
    parent: &InheritableMetadata,
    subclip_override: &InheritableMetadata,
    policy: InheritPolicy,
) -> InheritableMetadata {
    let mut result = subclip_override.clone();
    match policy {
        InheritPolicy::None => {}
        InheritPolicy::All => {
            result.merge_from(parent);
        }
        InheritPolicy::KeywordsOnly => {
            for kw in &parent.keywords {
                if !result.keywords.contains(kw) {
                    result.keywords.push(kw.clone());
                }
            }
        }
        InheritPolicy::ProductionOnly => {
            if result.reel.is_none() {
                result.reel = parent.reel.clone();
            }
            if result.scene.is_none() {
                result.scene = parent.scene.clone();
            }
            if result.take.is_none() {
                result.take = parent.take;
            }
        }
    }
    result
}

/// Registry mapping parent clip IDs to their inheritable metadata.
pub struct InheritanceRegistry {
    parents: HashMap<u64, InheritableMetadata>,
}

impl InheritanceRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            parents: HashMap::new(),
        }
    }

    /// Register a parent clip's metadata.
    pub fn register_parent(&mut self, clip_id: u64, meta: InheritableMetadata) {
        self.parents.insert(clip_id, meta);
    }

    /// Retrieve inherited metadata for a subclip.
    ///
    /// Returns `None` if the parent clip is not registered.
    #[must_use]
    pub fn get_parent(&self, parent_id: u64) -> Option<&InheritableMetadata> {
        self.parents.get(&parent_id)
    }

    /// Resolve a subclip's metadata given its parent and a policy.
    #[must_use]
    pub fn resolve(
        &self,
        parent_id: u64,
        subclip_override: &InheritableMetadata,
        policy: InheritPolicy,
    ) -> InheritableMetadata {
        match self.parents.get(&parent_id) {
            Some(parent) => resolve_inherited(parent, subclip_override, policy),
            None => subclip_override.clone(),
        }
    }

    /// Number of registered parents.
    #[must_use]
    pub fn len(&self) -> usize {
        self.parents.len()
    }

    /// Returns true when no parents are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.parents.is_empty()
    }
}

impl Default for InheritanceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_parent() -> InheritableMetadata {
        let mut m = InheritableMetadata::new();
        m.keywords = vec!["interview".into(), "outdoors".into()];
        m.reel = Some("A001".into());
        m.scene = Some("5".into());
        m.take = Some(3);
        m.description = Some("Parent description".into());
        m
    }

    fn make_override() -> InheritableMetadata {
        let mut m = InheritableMetadata::new();
        m.keywords = vec!["closeup".into()];
        m
    }

    #[test]
    fn test_inheritable_metadata_new() {
        let m = InheritableMetadata::new();
        assert!(m.keywords.is_empty());
        assert!(m.description.is_none());
        assert!(m.reel.is_none());
        assert!(m.scene.is_none());
        assert!(m.take.is_none());
    }

    #[test]
    fn test_merge_from_keywords_union() {
        let mut base = make_override();
        let parent = make_parent();
        base.merge_from(&parent);
        // Must contain both "closeup" and parent keywords
        assert!(base.keywords.contains(&"interview".to_string()));
        assert!(base.keywords.contains(&"closeup".to_string()));
        // No duplicate "outdoors"
        let count = base.keywords.iter().filter(|k| *k == "outdoors").count();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_merge_from_overrides_description() {
        let mut base = InheritableMetadata::new();
        base.description = Some("Sub description".into());
        let parent = make_parent();
        base.merge_from(&parent);
        // Parent's description overrides (per policy: parent wins on merge_from)
        assert_eq!(base.description.as_deref(), Some("Parent description"));
    }

    #[test]
    fn test_resolve_inherit_none() {
        let parent = make_parent();
        let subclip = make_override();
        let result = resolve_inherited(&parent, &subclip, InheritPolicy::None);
        // Should equal the subclip override only
        assert_eq!(result.keywords, vec!["closeup".to_string()]);
        assert!(result.reel.is_none());
    }

    #[test]
    fn test_resolve_inherit_all() {
        let parent = make_parent();
        let subclip = make_override();
        let result = resolve_inherited(&parent, &subclip, InheritPolicy::All);
        assert!(result.keywords.contains(&"interview".to_string()));
        assert!(result.keywords.contains(&"closeup".to_string()));
        assert_eq!(result.reel.as_deref(), Some("A001"));
    }

    #[test]
    fn test_resolve_keywords_only() {
        let parent = make_parent();
        let subclip = make_override();
        let result = resolve_inherited(&parent, &subclip, InheritPolicy::KeywordsOnly);
        assert!(result.keywords.contains(&"interview".to_string()));
        assert!(result.keywords.contains(&"closeup".to_string()));
        // Reel should NOT be inherited
        assert!(result.reel.is_none());
    }

    #[test]
    fn test_resolve_production_only() {
        let parent = make_parent();
        let subclip = make_override();
        let result = resolve_inherited(&parent, &subclip, InheritPolicy::ProductionOnly);
        // Reel, scene, take come from parent
        assert_eq!(result.reel.as_deref(), Some("A001"));
        assert_eq!(result.scene.as_deref(), Some("5"));
        assert_eq!(result.take, Some(3));
        // Keywords are only the subclip's own
        assert_eq!(result.keywords, vec!["closeup".to_string()]);
    }

    #[test]
    fn test_resolve_production_only_subclip_overrides() {
        let parent = make_parent();
        let mut subclip = make_override();
        subclip.reel = Some("B002".into());
        let result = resolve_inherited(&parent, &subclip, InheritPolicy::ProductionOnly);
        // Subclip reel takes priority
        assert_eq!(result.reel.as_deref(), Some("B002"));
    }

    #[test]
    fn test_registry_register_and_resolve() {
        let mut reg = InheritanceRegistry::new();
        reg.register_parent(42, make_parent());
        assert!(!reg.is_empty());
        assert_eq!(reg.len(), 1);

        let sub = make_override();
        let result = reg.resolve(42, &sub, InheritPolicy::All);
        assert!(result.keywords.contains(&"interview".to_string()));
    }

    #[test]
    fn test_registry_unknown_parent() {
        let reg = InheritanceRegistry::new();
        let sub = make_override();
        let result = reg.resolve(999, &sub, InheritPolicy::All);
        // No parent: result == subclip override
        assert_eq!(result.keywords, vec!["closeup".to_string()]);
    }

    #[test]
    fn test_registry_get_parent() {
        let mut reg = InheritanceRegistry::new();
        reg.register_parent(1, make_parent());
        let p = reg.get_parent(1);
        assert!(p.is_some());
        let p2 = reg.get_parent(99);
        assert!(p2.is_none());
    }

    #[test]
    fn test_default_trait() {
        let m: InheritableMetadata = Default::default();
        assert!(m.keywords.is_empty());
        let r: InheritanceRegistry = Default::default();
        assert!(r.is_empty());
    }
}
