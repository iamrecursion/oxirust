//! Collaborative permission management with hierarchical access levels,
//! role-based granularity, and time-bounded access grants.
//!
//! Provides five permission levels (`View` < `Comment` < `Edit` < `Manage` < `Own`)
//! and six roles (`Viewer`, `Commenter`, `Reviewer`, `Editor`, `Manager`, `Owner`)
//! with fine-grained capability checking and resource-scoped grants.

#![allow(dead_code)]

use std::collections::HashMap;

/// A permission level that can be granted to a user on a resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Permission {
    /// Can view the resource.
    View,
    /// Can leave comments on the resource.
    Comment,
    /// Can make edits to the resource.
    Edit,
    /// Can manage collaborators and settings.
    Manage,
    /// Full ownership of the resource.
    Own,
}

impl Permission {
    /// Return `true` if this permission level allows write operations.
    #[must_use]
    pub fn allows_write(&self) -> bool {
        matches!(self, Self::Edit | Self::Manage | Self::Own)
    }

    /// Numeric level of this permission (higher = more privileged).
    #[must_use]
    pub fn level(&self) -> u8 {
        match self {
            Self::View => 1,
            Self::Comment => 2,
            Self::Edit => 3,
            Self::Manage => 4,
            Self::Own => 5,
        }
    }

    /// Return `true` if this permission level subsumes `other`.
    ///
    /// `Own` implies all others; `View` implies only itself.
    #[must_use]
    pub fn implies(&self, other: &Permission) -> bool {
        self.level() >= other.level()
    }

    /// All permission levels in ascending order.
    #[must_use]
    pub fn all_levels() -> &'static [Permission] {
        &[
            Permission::View,
            Permission::Comment,
            Permission::Edit,
            Permission::Manage,
            Permission::Own,
        ]
    }
}

impl std::fmt::Display for Permission {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::View => write!(f, "view"),
            Self::Comment => write!(f, "comment"),
            Self::Edit => write!(f, "edit"),
            Self::Manage => write!(f, "manage"),
            Self::Own => write!(f, "own"),
        }
    }
}

// ---------------------------------------------------------------------------
// Fine-grained capabilities
// ---------------------------------------------------------------------------

/// A fine-grained capability that can be checked against a role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Capability {
    /// Can view resources and timeline.
    ViewContent,
    /// Can add comments and annotations.
    AddComments,
    /// Can resolve/unresolve comments.
    ResolveComments,
    /// Can approve or request changes on review items.
    ReviewApprove,
    /// Can edit timeline, clips, and effects.
    EditTimeline,
    /// Can lock/unlock regions.
    ManageLocks,
    /// Can invite or remove collaborators.
    ManageCollaborators,
    /// Can change project settings.
    ChangeSettings,
    /// Can export the project.
    Export,
    /// Can delete the project.
    DeleteProject,
    /// Can transfer ownership.
    TransferOwnership,
}

// ---------------------------------------------------------------------------
// Role-based access control
// ---------------------------------------------------------------------------

/// A named role with an associated set of capabilities.
///
/// Six built-in roles span from read-only `Viewer` to full `Owner`.
/// Custom roles can be created via [`CollabRole::custom`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollabRole {
    /// Machine-readable role identifier.
    pub name: String,
    /// Human-readable label.
    pub label: String,
    /// The base permission level implied by this role.
    pub base_permission: Permission,
    /// Explicit set of capabilities this role grants.
    pub capabilities: Vec<Capability>,
    /// Priority for conflict resolution (higher = takes precedence).
    pub priority: u8,
}

impl CollabRole {
    /// Build a custom role.
    pub fn custom(
        name: impl Into<String>,
        label: impl Into<String>,
        base_permission: Permission,
        capabilities: Vec<Capability>,
        priority: u8,
    ) -> Self {
        Self {
            name: name.into(),
            label: label.into(),
            base_permission,
            capabilities,
            priority,
        }
    }

    /// Check whether this role grants a specific capability.
    #[must_use]
    pub fn has_capability(&self, cap: Capability) -> bool {
        self.capabilities.contains(&cap)
    }

    // ── Built-in roles ──────────────────────────────────────────────────

    /// Viewer — read-only access.
    #[must_use]
    pub fn viewer() -> Self {
        Self {
            name: "viewer".into(),
            label: "Viewer".into(),
            base_permission: Permission::View,
            capabilities: vec![Capability::ViewContent],
            priority: 10,
        }
    }

    /// Commenter — can view and add comments.
    #[must_use]
    pub fn commenter() -> Self {
        Self {
            name: "commenter".into(),
            label: "Commenter".into(),
            base_permission: Permission::Comment,
            capabilities: vec![Capability::ViewContent, Capability::AddComments],
            priority: 20,
        }
    }

    /// Reviewer — can view, comment, resolve comments, and approve.
    #[must_use]
    pub fn reviewer() -> Self {
        Self {
            name: "reviewer".into(),
            label: "Reviewer".into(),
            base_permission: Permission::Comment,
            capabilities: vec![
                Capability::ViewContent,
                Capability::AddComments,
                Capability::ResolveComments,
                Capability::ReviewApprove,
                Capability::Export,
            ],
            priority: 30,
        }
    }

    /// Editor — full editing rights (timeline, locks, export).
    #[must_use]
    pub fn editor() -> Self {
        Self {
            name: "editor".into(),
            label: "Editor".into(),
            base_permission: Permission::Edit,
            capabilities: vec![
                Capability::ViewContent,
                Capability::AddComments,
                Capability::ResolveComments,
                Capability::ReviewApprove,
                Capability::EditTimeline,
                Capability::ManageLocks,
                Capability::Export,
            ],
            priority: 40,
        }
    }

    /// Manager — can manage collaborators and settings.
    #[must_use]
    pub fn manager() -> Self {
        Self {
            name: "manager".into(),
            label: "Manager".into(),
            base_permission: Permission::Manage,
            capabilities: vec![
                Capability::ViewContent,
                Capability::AddComments,
                Capability::ResolveComments,
                Capability::ReviewApprove,
                Capability::EditTimeline,
                Capability::ManageLocks,
                Capability::ManageCollaborators,
                Capability::ChangeSettings,
                Capability::Export,
            ],
            priority: 50,
        }
    }

    /// Owner — full ownership with delete and transfer.
    #[must_use]
    pub fn owner() -> Self {
        Self {
            name: "owner".into(),
            label: "Owner".into(),
            base_permission: Permission::Own,
            capabilities: vec![
                Capability::ViewContent,
                Capability::AddComments,
                Capability::ResolveComments,
                Capability::ReviewApprove,
                Capability::EditTimeline,
                Capability::ManageLocks,
                Capability::ManageCollaborators,
                Capability::ChangeSettings,
                Capability::Export,
                Capability::DeleteProject,
                Capability::TransferOwnership,
            ],
            priority: 60,
        }
    }
}

impl std::fmt::Display for CollabRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label)
    }
}

// ---------------------------------------------------------------------------
// Role registry
// ---------------------------------------------------------------------------

/// Registry of available roles, supporting built-in and custom definitions.
#[derive(Debug, Default)]
pub struct RoleRegistry {
    roles: HashMap<String, CollabRole>,
}

impl RoleRegistry {
    /// Create a registry pre-loaded with the six built-in roles.
    #[must_use]
    pub fn with_defaults() -> Self {
        let mut reg = Self::default();
        for role in [
            CollabRole::viewer(),
            CollabRole::commenter(),
            CollabRole::reviewer(),
            CollabRole::editor(),
            CollabRole::manager(),
            CollabRole::owner(),
        ] {
            reg.register(role);
        }
        reg
    }

    /// Register (or overwrite) a role.
    pub fn register(&mut self, role: CollabRole) {
        self.roles.insert(role.name.clone(), role);
    }

    /// Look up a role by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&CollabRole> {
        self.roles.get(name)
    }

    /// Remove a role. Returns `true` if it existed.
    pub fn remove(&mut self, name: &str) -> bool {
        self.roles.remove(name).is_some()
    }

    /// List all registered role names.
    #[must_use]
    pub fn role_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.roles.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }

    /// Number of registered roles.
    #[must_use]
    pub fn len(&self) -> usize {
        self.roles.len()
    }

    /// Whether the registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.roles.is_empty()
    }
}

// ---------------------------------------------------------------------------
// User role assignment
// ---------------------------------------------------------------------------

/// Tracks which role each user holds on a per-resource basis.
#[derive(Debug, Default)]
pub struct RoleAssignment {
    /// (user_id, resource_id) → role name
    assignments: HashMap<(String, String), String>,
}

impl RoleAssignment {
    /// Create an empty assignment table.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Assign a role to a user on a resource.
    pub fn assign(
        &mut self,
        user_id: impl Into<String>,
        resource_id: impl Into<String>,
        role_name: impl Into<String>,
    ) {
        self.assignments
            .insert((user_id.into(), resource_id.into()), role_name.into());
    }

    /// Remove a user's role on a resource. Returns `true` if existed.
    pub fn unassign(&mut self, user_id: &str, resource_id: &str) -> bool {
        self.assignments
            .remove(&(user_id.to_string(), resource_id.to_string()))
            .is_some()
    }

    /// Get the role name assigned to a user on a resource.
    #[must_use]
    pub fn get_role(&self, user_id: &str, resource_id: &str) -> Option<&str> {
        self.assignments
            .get(&(user_id.to_string(), resource_id.to_string()))
            .map(String::as_str)
    }

    /// Check whether a user has a specific capability on a resource,
    /// resolving through the registry.
    #[must_use]
    pub fn has_capability(
        &self,
        user_id: &str,
        resource_id: &str,
        cap: Capability,
        registry: &RoleRegistry,
    ) -> bool {
        self.get_role(user_id, resource_id)
            .and_then(|rn| registry.get(rn))
            .map_or(false, |role| role.has_capability(cap))
    }

    /// List all resources a user has any role on.
    #[must_use]
    pub fn user_resources(&self, user_id: &str) -> Vec<&str> {
        let mut resources: Vec<&str> = self
            .assignments
            .iter()
            .filter(|((uid, _), _)| uid == user_id)
            .map(|((_, rid), _)| rid.as_str())
            .collect();
        resources.sort_unstable();
        resources.dedup();
        resources
    }

    /// List all users that have a role on a resource.
    #[must_use]
    pub fn resource_users(&self, resource_id: &str) -> Vec<(&str, &str)> {
        self.assignments
            .iter()
            .filter(|((_, rid), _)| rid == resource_id)
            .map(|((uid, _), rn)| (uid.as_str(), rn.as_str()))
            .collect()
    }
}

/// A single access grant associating a user with a permission on a resource.
#[derive(Debug, Clone)]
pub struct AccessGrant {
    /// Identifier of the user receiving the grant.
    pub user_id: String,
    /// Identifier of the resource the grant applies to.
    pub resource_id: String,
    /// Permission level granted.
    pub permission: Permission,
    /// Wall-clock time the grant was issued, in milliseconds since the Unix epoch.
    pub granted_ms: u64,
    /// Optional expiry time in milliseconds; `None` means the grant never expires.
    pub expires_ms: Option<u64>,
}

impl AccessGrant {
    /// Return `true` when this grant has an expiry and `now_ms` is past it.
    #[must_use]
    pub fn is_expired(&self, now_ms: u64) -> bool {
        self.expires_ms.map_or(false, |exp| now_ms > exp)
    }

    /// Return `true` when this grant exists and is not yet expired.
    #[must_use]
    pub fn is_valid(&self, now_ms: u64) -> bool {
        !self.is_expired(now_ms)
    }
}

/// A collection of `AccessGrant`s with query and mutation helpers.
#[derive(Debug, Default)]
pub struct AccessControl {
    /// All grants stored in insertion order.
    pub grants: Vec<AccessGrant>,
}

impl AccessControl {
    /// Create an empty `AccessControl`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a new grant for `user_id` on `resource_id` with the given `permission`.
    pub fn grant(
        &mut self,
        user_id: impl Into<String>,
        resource_id: impl Into<String>,
        perm: Permission,
        now_ms: u64,
        expires_ms: Option<u64>,
    ) {
        self.grants.push(AccessGrant {
            user_id: user_id.into(),
            resource_id: resource_id.into(),
            permission: perm,
            granted_ms: now_ms,
            expires_ms,
        });
    }

    /// Remove all grants for `user_id` on `resource_id`.
    ///
    /// Returns `true` if at least one grant was removed.
    pub fn revoke(&mut self, user_id: &str, resource_id: &str) -> bool {
        let before = self.grants.len();
        self.grants
            .retain(|g| !(g.user_id == user_id && g.resource_id == resource_id));
        self.grants.len() < before
    }

    /// Return `true` if `user_id` has at least `needed` access on `resource_id`
    /// via any currently valid grant.
    #[must_use]
    pub fn check(
        &self,
        user_id: &str,
        resource_id: &str,
        needed: &Permission,
        now_ms: u64,
    ) -> bool {
        self.grants.iter().any(|g| {
            g.user_id == user_id
                && g.resource_id == resource_id
                && g.is_valid(now_ms)
                && g.permission.implies(needed)
        })
    }

    /// Return the resource ids of all resources `user_id` has any valid grant on.
    #[must_use]
    pub fn user_resources(&self, user_id: &str, now_ms: u64) -> Vec<&str> {
        let mut resources: Vec<&str> = self
            .grants
            .iter()
            .filter(|g| g.user_id == user_id && g.is_valid(now_ms))
            .map(|g| g.resource_id.as_str())
            .collect();
        resources.sort_unstable();
        resources.dedup();
        resources
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Permission ----

    #[test]
    fn test_permission_levels_ordered() {
        assert!(Permission::Own.level() > Permission::Manage.level());
        assert!(Permission::Manage.level() > Permission::Edit.level());
        assert!(Permission::Edit.level() > Permission::Comment.level());
        assert!(Permission::Comment.level() > Permission::View.level());
    }

    #[test]
    fn test_allows_write_true_for_edit_manage_own() {
        assert!(Permission::Edit.allows_write());
        assert!(Permission::Manage.allows_write());
        assert!(Permission::Own.allows_write());
    }

    #[test]
    fn test_allows_write_false_for_view_comment() {
        assert!(!Permission::View.allows_write());
        assert!(!Permission::Comment.allows_write());
    }

    #[test]
    fn test_own_implies_all() {
        for p in [
            Permission::View,
            Permission::Comment,
            Permission::Edit,
            Permission::Manage,
            Permission::Own,
        ] {
            assert!(Permission::Own.implies(&p));
        }
    }

    #[test]
    fn test_view_implies_only_itself() {
        assert!(Permission::View.implies(&Permission::View));
        assert!(!Permission::View.implies(&Permission::Comment));
    }

    #[test]
    fn test_implies_is_transitive() {
        // Edit implies Comment, Comment implies View → Edit implies View
        assert!(Permission::Edit.implies(&Permission::Comment));
        assert!(Permission::Edit.implies(&Permission::View));
    }

    #[test]
    fn test_permission_display() {
        assert_eq!(Permission::View.to_string(), "view");
        assert_eq!(Permission::Own.to_string(), "own");
        assert_eq!(Permission::Comment.to_string(), "comment");
    }

    #[test]
    fn test_all_levels_returns_five() {
        assert_eq!(Permission::all_levels().len(), 5);
        assert_eq!(Permission::all_levels()[0], Permission::View);
        assert_eq!(Permission::all_levels()[4], Permission::Own);
    }

    // ---- CollabRole built-in roles ----

    #[test]
    fn test_viewer_role_capabilities() {
        let v = CollabRole::viewer();
        assert!(v.has_capability(Capability::ViewContent));
        assert!(!v.has_capability(Capability::AddComments));
        assert!(!v.has_capability(Capability::EditTimeline));
        assert_eq!(v.base_permission, Permission::View);
        assert_eq!(v.name, "viewer");
    }

    #[test]
    fn test_commenter_role_capabilities() {
        let c = CollabRole::commenter();
        assert!(c.has_capability(Capability::ViewContent));
        assert!(c.has_capability(Capability::AddComments));
        assert!(!c.has_capability(Capability::ResolveComments));
        assert!(!c.has_capability(Capability::EditTimeline));
        assert_eq!(c.base_permission, Permission::Comment);
    }

    #[test]
    fn test_reviewer_role_capabilities() {
        let r = CollabRole::reviewer();
        assert!(r.has_capability(Capability::ViewContent));
        assert!(r.has_capability(Capability::AddComments));
        assert!(r.has_capability(Capability::ResolveComments));
        assert!(r.has_capability(Capability::ReviewApprove));
        assert!(r.has_capability(Capability::Export));
        assert!(!r.has_capability(Capability::EditTimeline));
        assert!(!r.has_capability(Capability::ManageLocks));
    }

    #[test]
    fn test_editor_role_capabilities() {
        let e = CollabRole::editor();
        assert!(e.has_capability(Capability::EditTimeline));
        assert!(e.has_capability(Capability::ManageLocks));
        assert!(e.has_capability(Capability::Export));
        assert!(!e.has_capability(Capability::ManageCollaborators));
        assert_eq!(e.base_permission, Permission::Edit);
    }

    #[test]
    fn test_manager_role_capabilities() {
        let m = CollabRole::manager();
        assert!(m.has_capability(Capability::ManageCollaborators));
        assert!(m.has_capability(Capability::ChangeSettings));
        assert!(!m.has_capability(Capability::DeleteProject));
        assert_eq!(m.base_permission, Permission::Manage);
    }

    #[test]
    fn test_owner_role_capabilities() {
        let o = CollabRole::owner();
        assert!(o.has_capability(Capability::DeleteProject));
        assert!(o.has_capability(Capability::TransferOwnership));
        assert_eq!(o.base_permission, Permission::Own);
    }

    #[test]
    fn test_role_priority_ordering() {
        assert!(CollabRole::owner().priority > CollabRole::manager().priority);
        assert!(CollabRole::manager().priority > CollabRole::editor().priority);
        assert!(CollabRole::editor().priority > CollabRole::reviewer().priority);
        assert!(CollabRole::reviewer().priority > CollabRole::commenter().priority);
        assert!(CollabRole::commenter().priority > CollabRole::viewer().priority);
    }

    #[test]
    fn test_custom_role() {
        let custom = CollabRole::custom(
            "qa_lead",
            "QA Lead",
            Permission::Comment,
            vec![
                Capability::ViewContent,
                Capability::ReviewApprove,
                Capability::Export,
            ],
            35,
        );
        assert_eq!(custom.name, "qa_lead");
        assert!(custom.has_capability(Capability::ReviewApprove));
        assert!(!custom.has_capability(Capability::EditTimeline));
    }

    #[test]
    fn test_role_display() {
        assert_eq!(CollabRole::editor().to_string(), "Editor");
        assert_eq!(CollabRole::reviewer().to_string(), "Reviewer");
    }

    // ---- RoleRegistry ----

    #[test]
    fn test_registry_defaults_has_six_roles() {
        let reg = RoleRegistry::with_defaults();
        assert_eq!(reg.len(), 6);
        assert!(!reg.is_empty());
    }

    #[test]
    fn test_registry_lookup() {
        let reg = RoleRegistry::with_defaults();
        let reviewer = reg.get("reviewer");
        assert!(reviewer.is_some());
        let reviewer = reviewer.expect("role should exist");
        assert!(reviewer.has_capability(Capability::ReviewApprove));
    }

    #[test]
    fn test_registry_custom_role() {
        let mut reg = RoleRegistry::with_defaults();
        let custom = CollabRole::custom(
            "intern",
            "Intern",
            Permission::View,
            vec![Capability::ViewContent],
            5,
        );
        reg.register(custom);
        assert_eq!(reg.len(), 7);
        assert!(reg.get("intern").is_some());
    }

    #[test]
    fn test_registry_remove() {
        let mut reg = RoleRegistry::with_defaults();
        assert!(reg.remove("viewer"));
        assert!(!reg.remove("nonexistent"));
        assert_eq!(reg.len(), 5);
    }

    #[test]
    fn test_registry_role_names_sorted() {
        let reg = RoleRegistry::with_defaults();
        let names = reg.role_names();
        assert_eq!(names.len(), 6);
        // Should be alphabetically sorted
        for i in 1..names.len() {
            assert!(names[i - 1] <= names[i]);
        }
    }

    // ---- RoleAssignment ----

    #[test]
    fn test_assign_and_get_role() {
        let mut ra = RoleAssignment::new();
        ra.assign("alice", "proj-1", "editor");
        assert_eq!(ra.get_role("alice", "proj-1"), Some("editor"));
        assert_eq!(ra.get_role("alice", "proj-2"), None);
    }

    #[test]
    fn test_unassign() {
        let mut ra = RoleAssignment::new();
        ra.assign("bob", "proj-1", "viewer");
        assert!(ra.unassign("bob", "proj-1"));
        assert!(!ra.unassign("bob", "proj-1")); // already removed
        assert_eq!(ra.get_role("bob", "proj-1"), None);
    }

    #[test]
    fn test_has_capability_through_registry() {
        let reg = RoleRegistry::with_defaults();
        let mut ra = RoleAssignment::new();
        ra.assign("alice", "proj-1", "reviewer");
        ra.assign("bob", "proj-1", "viewer");

        assert!(ra.has_capability("alice", "proj-1", Capability::ReviewApprove, &reg));
        assert!(!ra.has_capability("alice", "proj-1", Capability::EditTimeline, &reg));
        assert!(ra.has_capability("bob", "proj-1", Capability::ViewContent, &reg));
        assert!(!ra.has_capability("bob", "proj-1", Capability::AddComments, &reg));
    }

    #[test]
    fn test_user_resources() {
        let mut ra = RoleAssignment::new();
        ra.assign("alice", "proj-1", "editor");
        ra.assign("alice", "proj-2", "reviewer");
        ra.assign("bob", "proj-1", "viewer");
        let res = ra.user_resources("alice");
        assert_eq!(res.len(), 2);
    }

    #[test]
    fn test_resource_users() {
        let mut ra = RoleAssignment::new();
        ra.assign("alice", "proj-1", "editor");
        ra.assign("bob", "proj-1", "reviewer");
        ra.assign("carol", "proj-2", "viewer");
        let users = ra.resource_users("proj-1");
        assert_eq!(users.len(), 2);
    }

    #[test]
    fn test_reassign_overwrites() {
        let mut ra = RoleAssignment::new();
        ra.assign("alice", "proj-1", "viewer");
        ra.assign("alice", "proj-1", "editor");
        assert_eq!(ra.get_role("alice", "proj-1"), Some("editor"));
    }

    // ---- AccessGrant ----

    #[test]
    fn test_grant_not_expired_when_no_expiry() {
        let g = AccessGrant {
            user_id: "u1".into(),
            resource_id: "r1".into(),
            permission: Permission::View,
            granted_ms: 0,
            expires_ms: None,
        };
        assert!(!g.is_expired(u64::MAX));
        assert!(g.is_valid(u64::MAX));
    }

    #[test]
    fn test_grant_expired_past_deadline() {
        let g = AccessGrant {
            user_id: "u1".into(),
            resource_id: "r1".into(),
            permission: Permission::View,
            granted_ms: 0,
            expires_ms: Some(1000),
        };
        assert!(g.is_expired(2000));
        assert!(!g.is_valid(2000));
    }

    #[test]
    fn test_grant_not_expired_before_deadline() {
        let g = AccessGrant {
            user_id: "u1".into(),
            resource_id: "r1".into(),
            permission: Permission::Edit,
            granted_ms: 0,
            expires_ms: Some(5000),
        };
        assert!(!g.is_expired(4999));
        assert!(g.is_valid(4999));
    }

    // ---- AccessControl ----

    #[test]
    fn test_access_control_check_granted() {
        let mut ac = AccessControl::new();
        ac.grant("alice", "res-1", Permission::Edit, 0, None);
        assert!(ac.check("alice", "res-1", &Permission::Edit, 0));
        assert!(ac.check("alice", "res-1", &Permission::View, 0)); // Edit implies View
    }

    #[test]
    fn test_access_control_check_not_enough_permission() {
        let mut ac = AccessControl::new();
        ac.grant("alice", "res-1", Permission::View, 0, None);
        assert!(!ac.check("alice", "res-1", &Permission::Edit, 0));
    }

    #[test]
    fn test_access_control_check_expired_grant() {
        let mut ac = AccessControl::new();
        ac.grant("alice", "res-1", Permission::Own, 0, Some(500));
        assert!(!ac.check("alice", "res-1", &Permission::View, 1000)); // expired
    }

    #[test]
    fn test_access_control_revoke_returns_true() {
        let mut ac = AccessControl::new();
        ac.grant("alice", "res-1", Permission::Edit, 0, None);
        assert!(ac.revoke("alice", "res-1"));
        assert!(!ac.check("alice", "res-1", &Permission::Edit, 0));
    }

    #[test]
    fn test_access_control_revoke_missing_returns_false() {
        let mut ac = AccessControl::new();
        assert!(!ac.revoke("nobody", "res-x"));
    }

    #[test]
    fn test_access_control_user_resources() {
        let mut ac = AccessControl::new();
        ac.grant("alice", "res-1", Permission::View, 0, None);
        ac.grant("alice", "res-2", Permission::Edit, 0, None);
        ac.grant("bob", "res-1", Permission::View, 0, None);
        let res = ac.user_resources("alice", 0);
        assert_eq!(res.len(), 2);
        assert!(res.contains(&"res-1"));
        assert!(res.contains(&"res-2"));
    }

    #[test]
    fn test_access_control_user_resources_excludes_expired() {
        let mut ac = AccessControl::new();
        ac.grant("alice", "res-1", Permission::View, 0, Some(100));
        let res = ac.user_resources("alice", 500);
        assert!(res.is_empty());
    }
}
