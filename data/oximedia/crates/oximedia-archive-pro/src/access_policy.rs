//! Access policy management for archived content.
//!
//! Defines access levels (Open, Restricted, Embargoed, Confidential) and a
//! policy engine that evaluates whether access is permitted at a given time.

#![allow(dead_code)]

/// The level of access restriction applied to an archived item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AccessLevel {
    /// No restrictions — publicly accessible without authentication.
    Open,
    /// Access is limited to authorised users or groups.
    Restricted,
    /// Access is blocked until an embargo date has passed.
    Embargoed,
    /// Access requires explicit clearance at the highest level.
    Confidential,
}

impl AccessLevel {
    /// Returns `true` when authentication is required for this access level.
    #[must_use]
    pub const fn requires_auth(&self) -> bool {
        !matches!(self, Self::Open)
    }

    /// Returns a human-readable description of the access level.
    #[must_use]
    pub const fn description(&self) -> &'static str {
        match self {
            Self::Open => "Publicly accessible without authentication",
            Self::Restricted => "Accessible to authorised users only",
            Self::Embargoed => "Access blocked until embargo expires",
            Self::Confidential => "Requires explicit high-level clearance",
        }
    }

    /// Numeric severity: higher means more restrictive.
    #[must_use]
    pub const fn severity(&self) -> u8 {
        match self {
            Self::Open => 0,
            Self::Restricted => 1,
            Self::Embargoed => 2,
            Self::Confidential => 3,
        }
    }
}

/// An access policy for a single archived item.
///
/// An optional embargo timestamp (Unix seconds) can be set; the effective
/// access level degrades to [`AccessLevel::Embargoed`] until that time passes.
#[derive(Debug, Clone)]
pub struct AccessPolicy {
    /// The item this policy applies to.
    pub item_id: String,
    /// Base access level (ignoring embargo).
    pub base_level: AccessLevel,
    /// Optional Unix timestamp (seconds) after which the embargo lifts.
    pub embargo_until: Option<u64>,
}

impl AccessPolicy {
    /// Creates a new policy with the given base level and no embargo.
    #[must_use]
    pub fn new(item_id: impl Into<String>, base_level: AccessLevel) -> Self {
        Self {
            item_id: item_id.into(),
            base_level,
            embargo_until: None,
        }
    }

    /// Adds an embargo that lifts at `unix_ts` seconds.
    #[must_use]
    pub fn with_embargo(mut self, unix_ts: u64) -> Self {
        self.embargo_until = Some(unix_ts);
        self
    }

    /// Returns the effective access level at the given Unix timestamp.
    ///
    /// If the current time is before the embargo expires the level is at least
    /// [`AccessLevel::Embargoed`] (or more restrictive if the base level is).
    #[must_use]
    pub fn effective_level(&self, now_unix: u64) -> AccessLevel {
        if let Some(until) = self.embargo_until {
            if now_unix < until {
                // Return whichever is more restrictive.
                return if self.base_level > AccessLevel::Embargoed {
                    self.base_level
                } else {
                    AccessLevel::Embargoed
                };
            }
        }
        self.base_level
    }

    /// Returns `true` when access is allowed at the given time.
    ///
    /// Access is denied only when the effective level is `Confidential`.
    /// For all other levels the caller is expected to verify authentication
    /// separately; this method only checks the embargo / confidentiality gate.
    #[must_use]
    pub fn allows_at(&self, now_unix: u64) -> bool {
        self.effective_level(now_unix) != AccessLevel::Confidential
    }
}

/// A set of access policies, evaluated together.
#[derive(Debug, Default)]
pub struct AccessPolicySet {
    policies: Vec<AccessPolicy>,
}

impl AccessPolicySet {
    /// Creates an empty policy set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a policy to the set.
    pub fn add(&mut self, policy: AccessPolicy) {
        self.policies.push(policy);
    }

    /// Returns the number of policies in this set.
    #[must_use]
    pub fn len(&self) -> usize {
        self.policies.len()
    }

    /// Returns `true` if the set contains no policies.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.policies.is_empty()
    }

    /// Evaluates all policies at the given time, returning `true` only when
    /// **every** policy permits access.
    #[must_use]
    pub fn evaluate(&self, now_unix: u64) -> bool {
        self.policies.iter().all(|p| p.allows_at(now_unix))
    }

    /// Returns the most restrictive access level found across all policies at
    /// the given time.  Returns `None` when the set is empty.
    #[must_use]
    pub fn most_restrictive(&self, now_unix: u64) -> Option<AccessLevel> {
        self.policies
            .iter()
            .map(|p| p.effective_level(now_unix))
            .max()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_open_does_not_require_auth() {
        assert!(!AccessLevel::Open.requires_auth());
    }

    #[test]
    fn test_restricted_requires_auth() {
        assert!(AccessLevel::Restricted.requires_auth());
        assert!(AccessLevel::Embargoed.requires_auth());
        assert!(AccessLevel::Confidential.requires_auth());
    }

    #[test]
    fn test_severity_ordering() {
        assert!(AccessLevel::Open.severity() < AccessLevel::Restricted.severity());
        assert!(AccessLevel::Restricted.severity() < AccessLevel::Embargoed.severity());
        assert!(AccessLevel::Embargoed.severity() < AccessLevel::Confidential.severity());
    }

    #[test]
    fn test_access_level_ord() {
        assert!(AccessLevel::Open < AccessLevel::Restricted);
        assert!(AccessLevel::Restricted < AccessLevel::Embargoed);
        assert!(AccessLevel::Embargoed < AccessLevel::Confidential);
    }

    #[test]
    fn test_description_nonempty() {
        let levels = [
            AccessLevel::Open,
            AccessLevel::Restricted,
            AccessLevel::Embargoed,
            AccessLevel::Confidential,
        ];
        for l in levels {
            assert!(!l.description().is_empty());
        }
    }

    #[test]
    fn test_policy_no_embargo_effective_level() {
        let policy = AccessPolicy::new("item-1", AccessLevel::Restricted);
        assert_eq!(policy.effective_level(9_999_999), AccessLevel::Restricted);
    }

    #[test]
    fn test_policy_embargo_active() {
        let policy = AccessPolicy::new("item-2", AccessLevel::Open).with_embargo(2_000_000_000);
        assert_eq!(
            policy.effective_level(1_000_000_000),
            AccessLevel::Embargoed
        );
    }

    #[test]
    fn test_policy_embargo_lifted() {
        let policy = AccessPolicy::new("item-3", AccessLevel::Open).with_embargo(1_000_000_000);
        // Time is after embargo
        assert_eq!(policy.effective_level(2_000_000_000), AccessLevel::Open);
    }

    #[test]
    fn test_policy_confidential_overrides_embargo() {
        let policy =
            AccessPolicy::new("item-4", AccessLevel::Confidential).with_embargo(2_000_000_000);
        // Base level is more restrictive than Embargoed
        assert_eq!(
            policy.effective_level(1_000_000_000),
            AccessLevel::Confidential
        );
    }

    #[test]
    fn test_policy_allows_at_open() {
        let policy = AccessPolicy::new("item-5", AccessLevel::Open);
        assert!(policy.allows_at(0));
    }

    #[test]
    fn test_policy_allows_at_confidential_denied() {
        let policy = AccessPolicy::new("item-6", AccessLevel::Confidential);
        assert!(!policy.allows_at(9_999_999));
    }

    #[test]
    fn test_policy_set_evaluate_all_open() {
        let mut set = AccessPolicySet::new();
        set.add(AccessPolicy::new("a", AccessLevel::Open));
        set.add(AccessPolicy::new("b", AccessLevel::Open));
        assert!(set.evaluate(0));
    }

    #[test]
    fn test_policy_set_evaluate_one_confidential() {
        let mut set = AccessPolicySet::new();
        set.add(AccessPolicy::new("a", AccessLevel::Open));
        set.add(AccessPolicy::new("b", AccessLevel::Confidential));
        assert!(!set.evaluate(0));
    }

    #[test]
    fn test_policy_set_most_restrictive() {
        let mut set = AccessPolicySet::new();
        set.add(AccessPolicy::new("a", AccessLevel::Open));
        set.add(AccessPolicy::new("b", AccessLevel::Restricted));
        set.add(AccessPolicy::new("c", AccessLevel::Embargoed));
        assert_eq!(
            set.most_restrictive(9_999_999_999),
            Some(AccessLevel::Embargoed)
        );
    }

    #[test]
    fn test_policy_set_most_restrictive_empty() {
        let set = AccessPolicySet::new();
        assert_eq!(set.most_restrictive(0), None);
    }

    #[test]
    fn test_policy_set_len_and_empty() {
        let mut set = AccessPolicySet::new();
        assert!(set.is_empty());
        set.add(AccessPolicy::new("x", AccessLevel::Open));
        assert_eq!(set.len(), 1);
        assert!(!set.is_empty());
    }
}
