//! Fine-grained review permissions and role-based permission sets.
//!
//! Permissions control what actions a participant may take inside a review
//! session.  Each action maps to a boolean flag in `ReviewPermission`.  A
//! `PermissionSet` maps every `ReviewRole` to a default `ReviewPermission`
//! profile and allows callers to grant or revoke individual capabilities.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// ReviewPermission
// ---------------------------------------------------------------------------

/// Fine-grained capability flags for a review participant.
///
/// Every field defaults to `false`; use [`ReviewPermission::none`] for the
/// most-restrictive baseline or one of the role-specific constructors for
/// typical profiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ReviewPermission {
    /// May add annotations (drawings, markers, shapes).
    pub can_annotate: bool,
    /// May submit an approve/reject decision on the review.
    pub can_approve: bool,
    /// May export the review (PDF, CSV, EDL, etc.).
    pub can_export: bool,
    /// May invite additional participants to the session.
    pub can_invite: bool,
    /// May delete comments posted by other users.
    pub can_delete_comments: bool,
}

impl ReviewPermission {
    /// No capabilities — the most restrictive permission set.
    #[must_use]
    pub const fn none() -> Self {
        Self {
            can_annotate: false,
            can_approve: false,
            can_export: false,
            can_invite: false,
            can_delete_comments: false,
        }
    }

    /// All capabilities — the most permissive permission set.
    #[must_use]
    pub const fn all() -> Self {
        Self {
            can_annotate: true,
            can_approve: true,
            can_export: true,
            can_invite: true,
            can_delete_comments: true,
        }
    }

    /// Viewer profile: read-only access, no annotations or approvals.
    #[must_use]
    pub const fn viewer() -> Self {
        Self {
            can_annotate: false,
            can_approve: false,
            can_export: false,
            can_invite: false,
            can_delete_comments: false,
        }
    }

    /// Annotator profile: can add annotations but cannot approve, export, or invite.
    #[must_use]
    pub const fn annotator() -> Self {
        Self {
            can_annotate: true,
            can_approve: false,
            can_export: false,
            can_invite: false,
            can_delete_comments: false,
        }
    }

    /// Reviewer profile: can annotate and export but cannot make final approval decisions.
    #[must_use]
    pub const fn reviewer() -> Self {
        Self {
            can_annotate: true,
            can_approve: false,
            can_export: true,
            can_invite: false,
            can_delete_comments: false,
        }
    }

    /// Approver profile: can annotate, export, invite, and make approval decisions.
    #[must_use]
    pub const fn approver() -> Self {
        Self {
            can_annotate: true,
            can_approve: true,
            can_export: true,
            can_invite: true,
            can_delete_comments: false,
        }
    }

    /// Admin profile: all capabilities including deleting other users' comments.
    #[must_use]
    pub const fn admin() -> Self {
        Self::all()
    }

    /// Merge `other` into `self`, granting any additional permissions from `other`.
    ///
    /// This is a set-union: if either side has a flag set, the result has it set.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self {
            can_annotate: self.can_annotate || other.can_annotate,
            can_approve: self.can_approve || other.can_approve,
            can_export: self.can_export || other.can_export,
            can_invite: self.can_invite || other.can_invite,
            can_delete_comments: self.can_delete_comments || other.can_delete_comments,
        }
    }

    /// Intersect `self` with `other`, keeping only permissions both share.
    #[must_use]
    pub const fn intersect(self, other: Self) -> Self {
        Self {
            can_annotate: self.can_annotate && other.can_annotate,
            can_approve: self.can_approve && other.can_approve,
            can_export: self.can_export && other.can_export,
            can_invite: self.can_invite && other.can_invite,
            can_delete_comments: self.can_delete_comments && other.can_delete_comments,
        }
    }

    /// Grant `capability` from `other` to `self`.
    #[must_use]
    pub const fn with_annotate(mut self, value: bool) -> Self {
        self.can_annotate = value;
        self
    }

    /// Set the `can_approve` flag.
    #[must_use]
    pub const fn with_approve(mut self, value: bool) -> Self {
        self.can_approve = value;
        self
    }

    /// Set the `can_export` flag.
    #[must_use]
    pub const fn with_export(mut self, value: bool) -> Self {
        self.can_export = value;
        self
    }

    /// Set the `can_invite` flag.
    #[must_use]
    pub const fn with_invite(mut self, value: bool) -> Self {
        self.can_invite = value;
        self
    }

    /// Set the `can_delete_comments` flag.
    #[must_use]
    pub const fn with_delete_comments(mut self, value: bool) -> Self {
        self.can_delete_comments = value;
        self
    }
}

// ---------------------------------------------------------------------------
// ReviewRole (standalone, separate from the crate-level UserRole)
// ---------------------------------------------------------------------------

/// Role assigned to a participant in a review session, determining their
/// default `ReviewPermission` profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReviewRole {
    /// Read-only observer.
    Viewer,
    /// Can add annotations but cannot approve.
    Annotator,
    /// Can annotate and export; cannot approve or invite.
    Reviewer,
    /// Full reviewer with approval rights and invite capability.
    Approver,
    /// System administrator with unrestricted access.
    Admin,
}

impl ReviewRole {
    /// Return the default `ReviewPermission` for this role.
    #[must_use]
    pub fn default_permissions(self) -> ReviewPermission {
        match self {
            Self::Viewer => ReviewPermission::viewer(),
            Self::Annotator => ReviewPermission::annotator(),
            Self::Reviewer => ReviewPermission::reviewer(),
            Self::Approver => ReviewPermission::approver(),
            Self::Admin => ReviewPermission::admin(),
        }
    }

    /// Human-readable display name.
    #[must_use]
    pub fn display_name(self) -> &'static str {
        match self {
            Self::Viewer => "Viewer",
            Self::Annotator => "Annotator",
            Self::Reviewer => "Reviewer",
            Self::Approver => "Approver",
            Self::Admin => "Admin",
        }
    }
}

// ---------------------------------------------------------------------------
// PermissionSet
// ---------------------------------------------------------------------------

/// A mutable mapping from `ReviewRole` to `ReviewPermission` for a session.
///
/// The set is initialised with default profiles for every role.  Individual
/// roles can be overridden via [`PermissionSet::set`]; single flags can be
/// tweaked with [`PermissionSet::grant`] and [`PermissionSet::revoke`].
#[derive(Debug, Clone)]
pub struct PermissionSet {
    permissions: std::collections::HashMap<ReviewRole, ReviewPermission>,
}

impl PermissionSet {
    /// Create a new set with default role profiles.
    #[must_use]
    pub fn new() -> Self {
        let mut map = std::collections::HashMap::new();
        for role in [
            ReviewRole::Viewer,
            ReviewRole::Annotator,
            ReviewRole::Reviewer,
            ReviewRole::Approver,
            ReviewRole::Admin,
        ] {
            map.insert(role, role.default_permissions());
        }
        Self { permissions: map }
    }

    /// Retrieve the current `ReviewPermission` for `role`.
    #[must_use]
    pub fn get(&self, role: ReviewRole) -> ReviewPermission {
        self.permissions
            .get(&role)
            .copied()
            .unwrap_or(ReviewPermission::none())
    }

    /// Replace the entire permission profile for `role`.
    pub fn set(&mut self, role: ReviewRole, perm: ReviewPermission) {
        self.permissions.insert(role, perm);
    }

    /// Grant a specific capability to `role`.
    ///
    /// This is equivalent to `set(role, get(role).with_<flag>(true))` without
    /// requiring the caller to know which field to modify.
    pub fn grant(&mut self, role: ReviewRole, capability: Capability) {
        let perm = self.get(role);
        self.permissions
            .insert(role, apply_capability(perm, capability, true));
    }

    /// Revoke a specific capability from `role`.
    pub fn revoke(&mut self, role: ReviewRole, capability: Capability) {
        let perm = self.get(role);
        self.permissions
            .insert(role, apply_capability(perm, capability, false));
    }

    /// Returns `true` if `role` has the given `capability`.
    #[must_use]
    pub fn can(&self, role: ReviewRole, capability: Capability) -> bool {
        let perm = self.get(role);
        match capability {
            Capability::Annotate => perm.can_annotate,
            Capability::Approve => perm.can_approve,
            Capability::Export => perm.can_export,
            Capability::Invite => perm.can_invite,
            Capability::DeleteComments => perm.can_delete_comments,
        }
    }
}

impl Default for PermissionSet {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Capability enum  (used with grant/revoke/can)
// ---------------------------------------------------------------------------

/// Named capabilities that can be individually granted or revoked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Capability {
    /// Add annotations and drawings.
    Annotate,
    /// Submit approve/reject decisions.
    Approve,
    /// Export review artefacts.
    Export,
    /// Invite new participants.
    Invite,
    /// Delete other users' comments.
    DeleteComments,
}

fn apply_capability(mut perm: ReviewPermission, cap: Capability, value: bool) -> ReviewPermission {
    match cap {
        Capability::Annotate => perm.can_annotate = value,
        Capability::Approve => perm.can_approve = value,
        Capability::Export => perm.can_export = value,
        Capability::Invite => perm.can_invite = value,
        Capability::DeleteComments => perm.can_delete_comments = value,
    }
    perm
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // 1 — default profiles match role expectations
    #[test]
    fn test_default_viewer_cannot_annotate() {
        let perm = ReviewPermission::viewer();
        assert!(!perm.can_annotate);
        assert!(!perm.can_approve);
    }

    #[test]
    fn test_default_annotator_can_annotate_but_not_approve() {
        let perm = ReviewPermission::annotator();
        assert!(perm.can_annotate);
        assert!(!perm.can_approve);
    }

    #[test]
    fn test_default_reviewer_can_annotate_and_export() {
        let perm = ReviewPermission::reviewer();
        assert!(perm.can_annotate);
        assert!(!perm.can_approve);
        assert!(perm.can_export);
    }

    #[test]
    fn test_default_approver_can_annotate_approve_export_invite() {
        let perm = ReviewPermission::approver();
        assert!(perm.can_annotate);
        assert!(perm.can_approve);
        assert!(perm.can_export);
        assert!(perm.can_invite);
        assert!(!perm.can_delete_comments);
    }

    #[test]
    fn test_default_admin_has_all_permissions() {
        let perm = ReviewPermission::admin();
        assert!(perm.can_annotate);
        assert!(perm.can_approve);
        assert!(perm.can_export);
        assert!(perm.can_invite);
        assert!(perm.can_delete_comments);
    }

    // 2 — PermissionSet initialises with defaults
    #[test]
    fn test_permission_set_defaults() {
        let set = PermissionSet::new();
        assert!(!set.get(ReviewRole::Viewer).can_annotate);
        assert!(set.get(ReviewRole::Admin).can_delete_comments);
    }

    // 3 — grant adds capability
    #[test]
    fn test_permission_set_grant() {
        let mut set = PermissionSet::new();
        assert!(!set.can(ReviewRole::Viewer, Capability::Annotate));
        set.grant(ReviewRole::Viewer, Capability::Annotate);
        assert!(set.can(ReviewRole::Viewer, Capability::Annotate));
    }

    // 4 — revoke removes capability
    #[test]
    fn test_permission_set_revoke() {
        let mut set = PermissionSet::new();
        assert!(set.can(ReviewRole::Admin, Capability::DeleteComments));
        set.revoke(ReviewRole::Admin, Capability::DeleteComments);
        assert!(!set.can(ReviewRole::Admin, Capability::DeleteComments));
    }

    // 5 — set replaces entire profile
    #[test]
    fn test_permission_set_set_replaces_profile() {
        let mut set = PermissionSet::new();
        set.set(ReviewRole::Reviewer, ReviewPermission::none());
        assert!(!set.can(ReviewRole::Reviewer, Capability::Annotate));
        assert!(!set.can(ReviewRole::Reviewer, Capability::Export));
    }

    // 6 — union of permissions is correct
    #[test]
    fn test_permission_union() {
        let a = ReviewPermission::viewer(); // no capabilities
        let b = ReviewPermission::annotator(); // can_annotate
        let merged = a.union(b);
        assert!(merged.can_annotate);
        assert!(!merged.can_approve);
    }

    // 7 — intersect of permissions is correct
    #[test]
    fn test_permission_intersect() {
        let a = ReviewPermission::approver(); // can_annotate, can_approve, can_export, can_invite
        let b = ReviewPermission::reviewer(); // can_annotate, can_export (no approve/invite)
        let intersected = a.intersect(b);
        assert!(intersected.can_annotate);
        assert!(!intersected.can_approve);
        assert!(intersected.can_export);
        assert!(!intersected.can_invite);
    }

    // 8 — ReviewRole::display_name returns expected string
    #[test]
    fn test_review_role_display_name() {
        assert_eq!(ReviewRole::Viewer.display_name(), "Viewer");
        assert_eq!(ReviewRole::Admin.display_name(), "Admin");
    }
}
