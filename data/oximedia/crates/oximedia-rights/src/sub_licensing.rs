//! Sub-licensing module for managing sub-license chains with parent-child
//! relationship tracking.
//!
//! This module models a tree of licenses where a parent license holder may
//! grant sub-licenses to downstream parties, subject to scope, territory,
//! and time constraints inherited from the parent.  The engine validates that
//! no sub-license exceeds the scope of its ancestor chain.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

// ── SubLicenseScope ─────────────────────────────────────────────────────────

/// The permitted scope of a (sub-)license.
#[derive(Debug, Clone)]
pub struct SubLicenseScope {
    /// Usage types permitted (e.g. "streaming", "broadcast").
    /// Empty means all types.
    pub usage_types: Vec<String>,
    /// ISO 3166-1 territories covered.
    /// Empty means worldwide.
    pub territories: Vec<String>,
    /// Start of validity (Unix epoch ms).
    pub valid_from_ms: u64,
    /// End of validity (Unix epoch ms). `None` = perpetual.
    pub valid_until_ms: Option<u64>,
    /// Whether this license grants the right to sub-license further.
    pub can_sublicense: bool,
    /// Maximum depth of sub-licensing allowed below this node.
    /// `None` = unlimited, `Some(0)` = cannot sub-license.
    pub max_sublicense_depth: Option<u32>,
}

impl SubLicenseScope {
    /// Create a worldwide, all-types scope.
    pub fn worldwide(valid_from_ms: u64, valid_until_ms: Option<u64>) -> Self {
        Self {
            usage_types: Vec::new(),
            territories: Vec::new(),
            valid_from_ms,
            valid_until_ms,
            can_sublicense: true,
            max_sublicense_depth: None,
        }
    }

    /// Returns `true` if this scope covers the given territory.
    pub fn covers_territory(&self, territory: &str) -> bool {
        self.territories.is_empty() || self.territories.iter().any(|t| t == territory)
    }

    /// Returns `true` if this scope covers the given usage type.
    pub fn covers_usage(&self, usage_type: &str) -> bool {
        self.usage_types.is_empty() || self.usage_types.iter().any(|u| u == usage_type)
    }

    /// Returns `true` if the given timestamp is within the validity period.
    pub fn is_valid_at(&self, ts_ms: u64) -> bool {
        if ts_ms < self.valid_from_ms {
            return false;
        }
        match self.valid_until_ms {
            Some(until) => ts_ms < until,
            None => true,
        }
    }

    /// Check whether `child_scope` is fully contained within `self`.
    ///
    /// A child scope is valid only if every territory, usage type, and time
    /// range it claims is also covered by the parent.
    pub fn contains(&self, child: &SubLicenseScope) -> Result<(), String> {
        // Time check
        if child.valid_from_ms < self.valid_from_ms {
            return Err(format!(
                "Child starts at {} before parent at {}",
                child.valid_from_ms, self.valid_from_ms,
            ));
        }
        if let (Some(parent_until), Some(child_until)) = (self.valid_until_ms, child.valid_until_ms)
        {
            if child_until > parent_until {
                return Err(format!(
                    "Child ends at {} after parent at {}",
                    child_until, parent_until,
                ));
            }
        }
        if self.valid_until_ms.is_some() && child.valid_until_ms.is_none() {
            return Err(
                "Child has no expiry but parent does; child would outlive parent".to_string(),
            );
        }

        // Territory check
        if !self.territories.is_empty() {
            for t in &child.territories {
                if !self.territories.contains(t) {
                    return Err(format!("Child territory '{}' not in parent territories", t));
                }
            }
            // Child with empty (worldwide) is not valid under a restricted parent
            if child.territories.is_empty() {
                return Err(
                    "Child claims worldwide but parent has restricted territories".to_string(),
                );
            }
        }

        // Usage type check
        if !self.usage_types.is_empty() {
            for u in &child.usage_types {
                if !self.usage_types.contains(u) {
                    return Err(format!(
                        "Child usage type '{}' not in parent usage types",
                        u,
                    ));
                }
            }
            if child.usage_types.is_empty() {
                return Err(
                    "Child claims all usage types but parent has restricted types".to_string(),
                );
            }
        }

        Ok(())
    }
}

// ── SubLicense ──────────────────────────────────────────────────────────────

/// A single license node in the sub-license tree.
#[derive(Debug, Clone)]
pub struct SubLicense {
    /// Unique license identifier.
    pub id: String,
    /// ID of the parent license (`None` for root licenses).
    pub parent_id: Option<String>,
    /// The asset this license covers.
    pub asset_id: String,
    /// The entity granting this license.
    pub licensor: String,
    /// The entity receiving this license.
    pub licensee: String,
    /// Scope of the license.
    pub scope: SubLicenseScope,
    /// Depth in the sub-license tree (root = 0).
    pub depth: u32,
    /// Whether this license is currently active.
    pub active: bool,
    /// Royalty share that the licensee must pass up to the licensor (0.0-1.0).
    pub royalty_passthrough: f64,
}

impl SubLicense {
    /// Create a root license (no parent).
    pub fn root(
        id: &str,
        asset_id: &str,
        licensor: &str,
        licensee: &str,
        scope: SubLicenseScope,
    ) -> Self {
        Self {
            id: id.to_string(),
            parent_id: None,
            asset_id: asset_id.to_string(),
            licensor: licensor.to_string(),
            licensee: licensee.to_string(),
            scope,
            depth: 0,
            active: true,
            royalty_passthrough: 0.0,
        }
    }

    /// Returns `true` if this is a root license.
    pub fn is_root(&self) -> bool {
        self.parent_id.is_none()
    }
}

// ── SubLicenseChain ─────────────────────────────────────────────────────────

/// The chain of licenses from a leaf sub-license up to the root.
#[derive(Debug)]
pub struct SubLicenseChain {
    /// Ordered from leaf (index 0) to root (last index).
    pub chain: Vec<SubLicense>,
}

impl SubLicenseChain {
    /// The depth of the chain (number of sub-licensing hops).
    pub fn depth(&self) -> u32 {
        if self.chain.is_empty() {
            return 0;
        }
        self.chain.len() as u32 - 1
    }

    /// Composite royalty passthrough (product of all passthroughs).
    pub fn composite_royalty_passthrough(&self) -> f64 {
        if self.chain.is_empty() {
            return 0.0;
        }
        self.chain
            .iter()
            .map(|l| 1.0 - l.royalty_passthrough)
            .product::<f64>()
    }

    /// Returns `true` if all licenses in the chain are active.
    pub fn all_active(&self) -> bool {
        self.chain.iter().all(|l| l.active)
    }

    /// Returns `true` if all licenses are valid at `ts_ms`.
    pub fn all_valid_at(&self, ts_ms: u64) -> bool {
        self.chain.iter().all(|l| l.scope.is_valid_at(ts_ms))
    }
}

// ── SubLicenseRegistry ──────────────────────────────────────────────────────

/// Registry that stores and validates sub-license trees.
#[derive(Debug, Default)]
pub struct SubLicenseRegistry {
    licenses: HashMap<String, SubLicense>,
}

impl SubLicenseRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a root license.
    pub fn register_root(&mut self, license: SubLicense) -> Result<(), String> {
        if license.parent_id.is_some() {
            return Err("Expected a root license (no parent_id)".to_string());
        }
        if self.licenses.contains_key(&license.id) {
            return Err(format!("License '{}' already registered", license.id));
        }
        self.licenses.insert(license.id.clone(), license);
        Ok(())
    }

    /// Grant a sub-license under an existing license.
    ///
    /// Validates that:
    /// 1. The parent exists and is active.
    /// 2. The parent's scope permits sub-licensing.
    /// 3. The child's scope is contained within the parent's scope.
    /// 4. The sub-license depth does not exceed the parent's limit.
    pub fn grant_sublicense(&mut self, parent_id: &str, child: SubLicense) -> Result<(), String> {
        // Validate parent exists
        let parent = self
            .licenses
            .get(parent_id)
            .ok_or_else(|| format!("Parent license '{}' not found", parent_id))?
            .clone();

        if !parent.active {
            return Err(format!("Parent license '{}' is not active", parent_id));
        }

        if !parent.scope.can_sublicense {
            return Err(format!(
                "Parent license '{}' does not permit sub-licensing",
                parent_id,
            ));
        }

        // Check depth
        let child_depth = parent.depth + 1;
        if let Some(max_depth) = parent.scope.max_sublicense_depth {
            if child_depth > parent.depth + max_depth {
                return Err(format!(
                    "Sub-license depth {} exceeds max allowed depth {} from parent '{}'",
                    child_depth,
                    parent.depth + max_depth,
                    parent_id,
                ));
            }
        }

        // Validate scope containment
        parent.scope.contains(&child.scope)?;

        // Validate child ID uniqueness
        if self.licenses.contains_key(&child.id) {
            return Err(format!("License '{}' already registered", child.id));
        }

        // Create the child with correct metadata
        let mut registered_child = child;
        registered_child.parent_id = Some(parent_id.to_string());
        registered_child.depth = child_depth;

        self.licenses
            .insert(registered_child.id.clone(), registered_child);
        Ok(())
    }

    /// Look up a license by ID.
    pub fn get(&self, id: &str) -> Option<&SubLicense> {
        self.licenses.get(id)
    }

    /// Revoke a license and all its sub-licenses (cascade).
    pub fn revoke(&mut self, id: &str) -> usize {
        let mut revoked = 0;
        let mut to_revoke = vec![id.to_string()];

        while let Some(current_id) = to_revoke.pop() {
            if let Some(license) = self.licenses.get_mut(&current_id) {
                if license.active {
                    license.active = false;
                    revoked += 1;
                }
            }
            // Find children
            let children: Vec<String> = self
                .licenses
                .values()
                .filter(|l| l.parent_id.as_deref() == Some(&current_id))
                .map(|l| l.id.clone())
                .collect();
            to_revoke.extend(children);
        }

        revoked
    }

    /// Build the chain from a license up to the root.
    pub fn chain(&self, id: &str) -> Option<SubLicenseChain> {
        let mut chain = Vec::new();
        let mut current_id = Some(id.to_string());

        while let Some(cid) = current_id {
            let license = self.licenses.get(&cid)?;
            chain.push(license.clone());
            current_id = license.parent_id.clone();
        }

        Some(SubLicenseChain { chain })
    }

    /// All direct children of a license.
    pub fn children(&self, parent_id: &str) -> Vec<&SubLicense> {
        self.licenses
            .values()
            .filter(|l| l.parent_id.as_deref() == Some(parent_id))
            .collect()
    }

    /// All descendants (children, grandchildren, ...) of a license.
    pub fn descendants(&self, id: &str) -> Vec<&SubLicense> {
        let mut result = Vec::new();
        let mut queue = vec![id.to_string()];
        while let Some(current) = queue.pop() {
            for child in self.children(&current) {
                result.push(child);
                queue.push(child.id.clone());
            }
        }
        result
    }

    /// All root licenses.
    pub fn roots(&self) -> Vec<&SubLicense> {
        self.licenses.values().filter(|l| l.is_root()).collect()
    }

    /// Total number of licenses in the registry.
    pub fn len(&self) -> usize {
        self.licenses.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.licenses.is_empty()
    }

    /// All active licenses for a given asset.
    pub fn active_for_asset(&self, asset_id: &str) -> Vec<&SubLicense> {
        self.licenses
            .values()
            .filter(|l| l.asset_id == asset_id && l.active)
            .collect()
    }

    /// Validate that a usage is covered by the license chain.
    pub fn is_usage_covered(
        &self,
        license_id: &str,
        usage_type: &str,
        territory: &str,
        ts_ms: u64,
    ) -> bool {
        let chain = match self.chain(license_id) {
            Some(c) => c,
            None => return false,
        };

        if !chain.all_active() {
            return false;
        }

        chain.chain.iter().all(|l| {
            l.scope.covers_usage(usage_type)
                && l.scope.covers_territory(territory)
                && l.scope.is_valid_at(ts_ms)
        })
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn worldwide_scope() -> SubLicenseScope {
        SubLicenseScope::worldwide(0, None)
    }

    fn restricted_scope() -> SubLicenseScope {
        SubLicenseScope {
            usage_types: vec!["streaming".to_string()],
            territories: vec!["US".to_string(), "CA".to_string()],
            valid_from_ms: 1000,
            valid_until_ms: Some(5000),
            can_sublicense: true,
            max_sublicense_depth: None,
        }
    }

    fn root_license(id: &str) -> SubLicense {
        SubLicense::root(id, "asset-1", "owner", "licensee-a", worldwide_scope())
    }

    fn child_license(id: &str, scope: SubLicenseScope) -> SubLicense {
        SubLicense {
            id: id.to_string(),
            parent_id: None, // will be set by grant_sublicense
            asset_id: "asset-1".to_string(),
            licensor: "licensee-a".to_string(),
            licensee: "sub-licensee-b".to_string(),
            scope,
            depth: 0,
            active: true,
            royalty_passthrough: 0.1,
        }
    }

    // ── SubLicenseScope ─────────────────────────────────────────────────────

    #[test]
    fn test_scope_worldwide_covers_all() {
        let s = worldwide_scope();
        assert!(s.covers_territory("US"));
        assert!(s.covers_territory("JP"));
        assert!(s.covers_usage("streaming"));
        assert!(s.covers_usage("broadcast"));
    }

    #[test]
    fn test_scope_restricted_covers_only_listed() {
        let s = restricted_scope();
        assert!(s.covers_territory("US"));
        assert!(!s.covers_territory("JP"));
        assert!(s.covers_usage("streaming"));
        assert!(!s.covers_usage("broadcast"));
    }

    #[test]
    fn test_scope_is_valid_at() {
        let s = restricted_scope();
        assert!(!s.is_valid_at(500));
        assert!(s.is_valid_at(1000));
        assert!(s.is_valid_at(3000));
        assert!(!s.is_valid_at(5000)); // exclusive
    }

    #[test]
    fn test_scope_contains_valid_child() {
        let parent = worldwide_scope();
        let child = restricted_scope();
        assert!(parent.contains(&child).is_ok());
    }

    #[test]
    fn test_scope_contains_child_exceeds_time() {
        let parent = SubLicenseScope {
            valid_from_ms: 1000,
            valid_until_ms: Some(3000),
            ..worldwide_scope()
        };
        let child = SubLicenseScope {
            valid_from_ms: 1000,
            valid_until_ms: Some(5000),
            ..worldwide_scope()
        };
        assert!(parent.contains(&child).is_err());
    }

    #[test]
    fn test_scope_contains_child_starts_before_parent() {
        let parent = SubLicenseScope {
            valid_from_ms: 1000,
            ..worldwide_scope()
        };
        let child = SubLicenseScope {
            valid_from_ms: 500,
            ..worldwide_scope()
        };
        assert!(parent.contains(&child).is_err());
    }

    #[test]
    fn test_scope_contains_child_exceeds_territory() {
        let parent = SubLicenseScope {
            territories: vec!["US".to_string()],
            ..worldwide_scope()
        };
        let child = SubLicenseScope {
            territories: vec!["US".to_string(), "JP".to_string()],
            ..worldwide_scope()
        };
        assert!(parent.contains(&child).is_err());
    }

    #[test]
    fn test_scope_contains_child_worldwide_under_restricted() {
        let parent = SubLicenseScope {
            territories: vec!["US".to_string()],
            ..worldwide_scope()
        };
        let child = SubLicenseScope {
            territories: vec![],
            ..worldwide_scope()
        };
        assert!(parent.contains(&child).is_err());
    }

    #[test]
    fn test_scope_contains_child_exceeds_usage() {
        let parent = SubLicenseScope {
            usage_types: vec!["streaming".to_string()],
            ..worldwide_scope()
        };
        let child = SubLicenseScope {
            usage_types: vec!["streaming".to_string(), "broadcast".to_string()],
            ..worldwide_scope()
        };
        assert!(parent.contains(&child).is_err());
    }

    #[test]
    fn test_scope_contains_child_perpetual_under_bounded() {
        let parent = SubLicenseScope {
            valid_from_ms: 0,
            valid_until_ms: Some(5000),
            ..worldwide_scope()
        };
        let child = SubLicenseScope {
            valid_from_ms: 0,
            valid_until_ms: None,
            ..worldwide_scope()
        };
        assert!(parent.contains(&child).is_err());
    }

    // ── SubLicense ──────────────────────────────────────────────────────────

    #[test]
    fn test_root_license_is_root() {
        let l = root_license("L1");
        assert!(l.is_root());
        assert_eq!(l.depth, 0);
    }

    // ── SubLicenseRegistry ──────────────────────────────────────────────────

    #[test]
    fn test_registry_register_root() {
        let mut reg = SubLicenseRegistry::new();
        assert!(reg.register_root(root_license("L1")).is_ok());
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn test_registry_register_root_duplicate() {
        let mut reg = SubLicenseRegistry::new();
        reg.register_root(root_license("L1"))
            .expect("should succeed");
        assert!(reg.register_root(root_license("L1")).is_err());
    }

    #[test]
    fn test_registry_register_non_root_as_root() {
        let mut reg = SubLicenseRegistry::new();
        let mut l = root_license("L1");
        l.parent_id = Some("P1".to_string());
        assert!(reg.register_root(l).is_err());
    }

    #[test]
    fn test_registry_grant_sublicense_success() {
        let mut reg = SubLicenseRegistry::new();
        reg.register_root(root_license("L1"))
            .expect("should succeed");
        let child = child_license("L2", restricted_scope());
        assert!(reg.grant_sublicense("L1", child).is_ok());
        assert_eq!(reg.len(), 2);
        let l2 = reg.get("L2").expect("should exist");
        assert_eq!(l2.depth, 1);
        assert_eq!(l2.parent_id.as_deref(), Some("L1"));
    }

    #[test]
    fn test_registry_grant_sublicense_parent_not_found() {
        let mut reg = SubLicenseRegistry::new();
        let child = child_license("L2", restricted_scope());
        assert!(reg.grant_sublicense("NOPE", child).is_err());
    }

    #[test]
    fn test_registry_grant_sublicense_parent_inactive() {
        let mut reg = SubLicenseRegistry::new();
        let mut root = root_license("L1");
        root.active = false;
        reg.licenses.insert("L1".to_string(), root);
        let child = child_license("L2", restricted_scope());
        assert!(reg.grant_sublicense("L1", child).is_err());
    }

    #[test]
    fn test_registry_grant_sublicense_no_permission() {
        let mut reg = SubLicenseRegistry::new();
        let mut root = root_license("L1");
        root.scope.can_sublicense = false;
        reg.licenses.insert("L1".to_string(), root);
        let child = child_license("L2", restricted_scope());
        assert!(reg.grant_sublicense("L1", child).is_err());
    }

    #[test]
    fn test_registry_grant_sublicense_exceeds_scope() {
        let mut reg = SubLicenseRegistry::new();
        let mut root = root_license("L1");
        root.scope.territories = vec!["US".to_string()];
        reg.licenses.insert("L1".to_string(), root);
        let child_scope = SubLicenseScope {
            territories: vec!["JP".to_string()],
            ..worldwide_scope()
        };
        let child = child_license("L2", child_scope);
        assert!(reg.grant_sublicense("L1", child).is_err());
    }

    #[test]
    fn test_registry_grant_sublicense_exceeds_depth() {
        let mut reg = SubLicenseRegistry::new();
        let mut root = root_license("L1");
        root.scope.max_sublicense_depth = Some(0);
        reg.licenses.insert("L1".to_string(), root);
        let child = child_license("L2", restricted_scope());
        assert!(reg.grant_sublicense("L1", child).is_err());
    }

    #[test]
    fn test_registry_revoke_cascades() {
        let mut reg = SubLicenseRegistry::new();
        reg.register_root(root_license("L1")).expect("ok");
        reg.grant_sublicense("L1", child_license("L2", restricted_scope()))
            .expect("ok");

        let child_of_l2 = SubLicense {
            id: "L3".to_string(),
            parent_id: None,
            asset_id: "asset-1".to_string(),
            licensor: "sub-licensee-b".to_string(),
            licensee: "sub-sub-c".to_string(),
            scope: SubLicenseScope {
                usage_types: vec!["streaming".to_string()],
                territories: vec!["US".to_string()],
                valid_from_ms: 1000,
                valid_until_ms: Some(5000),
                can_sublicense: false,
                max_sublicense_depth: Some(0),
            },
            depth: 0,
            active: true,
            royalty_passthrough: 0.05,
        };
        reg.grant_sublicense("L2", child_of_l2).expect("ok");

        let revoked = reg.revoke("L1");
        assert_eq!(revoked, 3); // L1, L2, L3
        assert!(!reg.get("L1").expect("exists").active);
        assert!(!reg.get("L2").expect("exists").active);
        assert!(!reg.get("L3").expect("exists").active);
    }

    #[test]
    fn test_registry_revoke_partial() {
        let mut reg = SubLicenseRegistry::new();
        reg.register_root(root_license("L1")).expect("ok");
        reg.grant_sublicense("L1", child_license("L2", restricted_scope()))
            .expect("ok");

        let revoked = reg.revoke("L2");
        assert_eq!(revoked, 1);
        assert!(reg.get("L1").expect("exists").active); // root untouched
        assert!(!reg.get("L2").expect("exists").active);
    }

    #[test]
    fn test_registry_chain() {
        let mut reg = SubLicenseRegistry::new();
        reg.register_root(root_license("L1")).expect("ok");
        reg.grant_sublicense("L1", child_license("L2", restricted_scope()))
            .expect("ok");

        let chain = reg.chain("L2").expect("chain should exist");
        assert_eq!(chain.chain.len(), 2);
        assert_eq!(chain.chain[0].id, "L2");
        assert_eq!(chain.chain[1].id, "L1");
        assert_eq!(chain.depth(), 1);
    }

    #[test]
    fn test_registry_chain_root() {
        let mut reg = SubLicenseRegistry::new();
        reg.register_root(root_license("L1")).expect("ok");

        let chain = reg.chain("L1").expect("chain should exist");
        assert_eq!(chain.chain.len(), 1);
        assert_eq!(chain.depth(), 0);
    }

    #[test]
    fn test_registry_chain_nonexistent() {
        let reg = SubLicenseRegistry::new();
        assert!(reg.chain("NOPE").is_none());
    }

    #[test]
    fn test_chain_all_active() {
        let mut reg = SubLicenseRegistry::new();
        reg.register_root(root_license("L1")).expect("ok");
        reg.grant_sublicense("L1", child_license("L2", restricted_scope()))
            .expect("ok");

        let chain = reg.chain("L2").expect("chain");
        assert!(chain.all_active());
    }

    #[test]
    fn test_chain_not_all_active_after_revoke() {
        let mut reg = SubLicenseRegistry::new();
        reg.register_root(root_license("L1")).expect("ok");
        reg.grant_sublicense("L1", child_license("L2", restricted_scope()))
            .expect("ok");
        reg.revoke("L1");

        let chain = reg.chain("L2").expect("chain");
        assert!(!chain.all_active());
    }

    #[test]
    fn test_chain_composite_royalty() {
        let mut reg = SubLicenseRegistry::new();
        let mut root = root_license("L1");
        root.royalty_passthrough = 0.1; // 10% passes through
        reg.licenses.insert("L1".to_string(), root);

        let mut child = child_license("L2", restricted_scope());
        child.parent_id = Some("L1".to_string());
        child.depth = 1;
        child.royalty_passthrough = 0.2; // 20% passes through
        reg.licenses.insert("L2".to_string(), child);

        let chain = reg.chain("L2").expect("chain");
        // (1 - 0.2) * (1 - 0.1) = 0.8 * 0.9 = 0.72
        assert!((chain.composite_royalty_passthrough() - 0.72).abs() < 1e-9);
    }

    #[test]
    fn test_registry_children() {
        let mut reg = SubLicenseRegistry::new();
        reg.register_root(root_license("L1")).expect("ok");
        reg.grant_sublicense("L1", child_license("L2", restricted_scope()))
            .expect("ok");

        let mut child3_scope = restricted_scope();
        child3_scope.territories = vec!["CA".to_string()];
        let mut child3 = child_license("L3", child3_scope);
        child3.licensee = "sub-licensee-c".to_string();
        reg.grant_sublicense("L1", child3).expect("ok");

        let children = reg.children("L1");
        assert_eq!(children.len(), 2);
    }

    #[test]
    fn test_registry_descendants() {
        let mut reg = SubLicenseRegistry::new();
        reg.register_root(root_license("L1")).expect("ok");
        reg.grant_sublicense("L1", child_license("L2", restricted_scope()))
            .expect("ok");

        let grandchild = SubLicense {
            id: "L3".to_string(),
            parent_id: None,
            asset_id: "asset-1".to_string(),
            licensor: "sub-licensee-b".to_string(),
            licensee: "grand-child".to_string(),
            scope: SubLicenseScope {
                usage_types: vec!["streaming".to_string()],
                territories: vec!["US".to_string()],
                valid_from_ms: 1000,
                valid_until_ms: Some(5000),
                can_sublicense: false,
                max_sublicense_depth: Some(0),
            },
            depth: 0,
            active: true,
            royalty_passthrough: 0.0,
        };
        reg.grant_sublicense("L2", grandchild).expect("ok");

        let desc = reg.descendants("L1");
        assert_eq!(desc.len(), 2); // L2 and L3
    }

    #[test]
    fn test_registry_roots() {
        let mut reg = SubLicenseRegistry::new();
        reg.register_root(root_license("L1")).expect("ok");
        reg.register_root(SubLicense::root(
            "L10",
            "asset-2",
            "owner-2",
            "lic-2",
            worldwide_scope(),
        ))
        .expect("ok");

        let roots = reg.roots();
        assert_eq!(roots.len(), 2);
    }

    #[test]
    fn test_registry_active_for_asset() {
        let mut reg = SubLicenseRegistry::new();
        reg.register_root(root_license("L1")).expect("ok");
        reg.grant_sublicense("L1", child_license("L2", restricted_scope()))
            .expect("ok");
        reg.revoke("L2");

        let active = reg.active_for_asset("asset-1");
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].id, "L1");
    }

    #[test]
    fn test_registry_is_usage_covered() {
        let mut reg = SubLicenseRegistry::new();
        reg.register_root(root_license("L1")).expect("ok");
        reg.grant_sublicense("L1", child_license("L2", restricted_scope()))
            .expect("ok");

        assert!(reg.is_usage_covered("L2", "streaming", "US", 2000));
        assert!(!reg.is_usage_covered("L2", "broadcast", "US", 2000));
        assert!(!reg.is_usage_covered("L2", "streaming", "JP", 2000));
        assert!(!reg.is_usage_covered("L2", "streaming", "US", 6000)); // expired
    }

    #[test]
    fn test_registry_is_usage_covered_revoked_chain() {
        let mut reg = SubLicenseRegistry::new();
        reg.register_root(root_license("L1")).expect("ok");
        reg.grant_sublicense("L1", child_license("L2", restricted_scope()))
            .expect("ok");
        reg.revoke("L1");

        assert!(!reg.is_usage_covered("L2", "streaming", "US", 2000));
    }

    #[test]
    fn test_registry_empty() {
        let reg = SubLicenseRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn test_chain_all_valid_at() {
        let mut reg = SubLicenseRegistry::new();
        reg.register_root(root_license("L1")).expect("ok");
        reg.grant_sublicense("L1", child_license("L2", restricted_scope()))
            .expect("ok");

        let chain = reg.chain("L2").expect("chain");
        assert!(chain.all_valid_at(2000));
        assert!(!chain.all_valid_at(6000));
    }
}
