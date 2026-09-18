//! Permission management for media access control.
//!
//! Provides `Permission`, `PermissionSet`, and `PermissionSetBuilder`.

#![allow(dead_code)]

use std::collections::HashSet;

/// An individual permission that can be granted or revoked.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Permission {
    /// Read/view media content.
    Read,
    /// Write or upload new content.
    Write,
    /// Delete existing content.
    Delete,
    /// Full administrative control.
    Admin,
    /// Publish or distribute content publicly.
    Publish,
}

impl Permission {
    /// Returns `true` for permissions that modify state on the server.
    #[must_use]
    pub fn is_write_action(&self) -> bool {
        matches!(
            self,
            Permission::Write | Permission::Delete | Permission::Admin | Permission::Publish
        )
    }

    /// Human-readable name for this permission.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Permission::Read => "Read",
            Permission::Write => "Write",
            Permission::Delete => "Delete",
            Permission::Admin => "Admin",
            Permission::Publish => "Publish",
        }
    }
}

/// An immutable snapshot of the permissions held by a principal.
#[derive(Debug, Clone, Default)]
pub struct PermissionSet {
    granted: HashSet<Permission>,
}

impl PermissionSet {
    /// Create an empty permission set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Grant a permission.
    pub fn grant(&mut self, permission: Permission) {
        self.granted.insert(permission);
    }

    /// Revoke a permission.
    pub fn revoke(&mut self, permission: &Permission) {
        self.granted.remove(permission);
    }

    /// Check whether a specific permission is currently granted.
    #[must_use]
    pub fn has(&self, permission: &Permission) -> bool {
        self.granted.contains(permission)
    }

    /// Number of permissions currently granted.
    #[must_use]
    pub fn count(&self) -> usize {
        self.granted.len()
    }

    /// Returns `true` if no permissions are granted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.granted.is_empty()
    }

    /// Returns an iterator over all granted permissions.
    pub fn iter(&self) -> impl Iterator<Item = &Permission> {
        self.granted.iter()
    }

    /// Merge another `PermissionSet` into this one (union).
    pub fn merge(&mut self, other: &PermissionSet) {
        for p in &other.granted {
            self.granted.insert(p.clone());
        }
    }

    /// Returns a new `PermissionSet` containing only the intersection.
    #[must_use]
    pub fn intersection(&self, other: &PermissionSet) -> PermissionSet {
        PermissionSet {
            granted: self.granted.intersection(&other.granted).cloned().collect(),
        }
    }
}

/// Fluent builder for constructing `PermissionSet` instances.
#[derive(Debug, Default)]
pub struct PermissionSetBuilder {
    set: PermissionSet,
}

impl PermissionSetBuilder {
    /// Start building a new set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Grant `Read` and `Write` permissions.
    #[must_use]
    pub fn allow_read_write(mut self) -> Self {
        self.set.grant(Permission::Read);
        self.set.grant(Permission::Write);
        self
    }

    /// Grant `Read` only.
    #[must_use]
    pub fn read_only(mut self) -> Self {
        self.set.grant(Permission::Read);
        self
    }

    /// Grant full admin access (all permissions).
    #[must_use]
    pub fn full_admin(mut self) -> Self {
        self.set.grant(Permission::Read);
        self.set.grant(Permission::Write);
        self.set.grant(Permission::Delete);
        self.set.grant(Permission::Admin);
        self.set.grant(Permission::Publish);
        self
    }

    /// Grant an individual permission.
    #[must_use]
    pub fn with(mut self, permission: Permission) -> Self {
        self.set.grant(permission);
        self
    }

    /// Consume the builder and return the constructed `PermissionSet`.
    #[must_use]
    pub fn build(self) -> PermissionSet {
        self.set
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permission_is_write_action_read_false() {
        assert!(!Permission::Read.is_write_action());
    }

    #[test]
    fn permission_is_write_action_write_true() {
        assert!(Permission::Write.is_write_action());
    }

    #[test]
    fn permission_is_write_action_delete_true() {
        assert!(Permission::Delete.is_write_action());
    }

    #[test]
    fn permission_is_write_action_admin_true() {
        assert!(Permission::Admin.is_write_action());
    }

    #[test]
    fn permission_name() {
        assert_eq!(Permission::Publish.name(), "Publish");
        assert_eq!(Permission::Read.name(), "Read");
    }

    #[test]
    fn permission_set_grant_and_has() {
        let mut set = PermissionSet::new();
        set.grant(Permission::Read);
        assert!(set.has(&Permission::Read));
        assert!(!set.has(&Permission::Write));
    }

    #[test]
    fn permission_set_revoke() {
        let mut set = PermissionSet::new();
        set.grant(Permission::Write);
        set.revoke(&Permission::Write);
        assert!(!set.has(&Permission::Write));
    }

    #[test]
    fn permission_set_count() {
        let mut set = PermissionSet::new();
        assert_eq!(set.count(), 0);
        set.grant(Permission::Read);
        set.grant(Permission::Write);
        assert_eq!(set.count(), 2);
    }

    #[test]
    fn permission_set_is_empty() {
        let set = PermissionSet::new();
        assert!(set.is_empty());
    }

    #[test]
    fn permission_set_merge() {
        let mut a = PermissionSet::new();
        a.grant(Permission::Read);
        let mut b = PermissionSet::new();
        b.grant(Permission::Write);
        a.merge(&b);
        assert!(a.has(&Permission::Read));
        assert!(a.has(&Permission::Write));
    }

    #[test]
    fn permission_set_intersection() {
        let mut a = PermissionSet::new();
        a.grant(Permission::Read);
        a.grant(Permission::Write);
        let mut b = PermissionSet::new();
        b.grant(Permission::Write);
        b.grant(Permission::Delete);
        let inter = a.intersection(&b);
        assert!(inter.has(&Permission::Write));
        assert!(!inter.has(&Permission::Read));
        assert!(!inter.has(&Permission::Delete));
    }

    #[test]
    fn builder_allow_read_write() {
        let set = PermissionSetBuilder::new().allow_read_write().build();
        assert!(set.has(&Permission::Read));
        assert!(set.has(&Permission::Write));
        assert!(!set.has(&Permission::Delete));
    }

    #[test]
    fn builder_read_only() {
        let set = PermissionSetBuilder::new().read_only().build();
        assert!(set.has(&Permission::Read));
        assert!(!set.has(&Permission::Write));
    }

    #[test]
    fn builder_full_admin() {
        let set = PermissionSetBuilder::new().full_admin().build();
        assert_eq!(set.count(), 5);
    }
}
