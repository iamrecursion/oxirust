#![allow(dead_code)]
//! Usage rights definitions and management for media assets.

/// The type of usage permitted for a media asset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UsageType {
    /// Broadcast on television or radio.
    Broadcast,
    /// Streaming over the internet.
    Streaming,
    /// Theatrical or cinema release.
    Theatrical,
    /// Distribution on physical media (DVD, Blu-ray, etc.).
    PhysicalMedia,
    /// Use for advertising or marketing material.
    Advertising,
    /// Editorial use (news, documentary, review).
    Editorial,
    /// Internal or corporate use only.
    Corporate,
    /// Educational use in academic contexts.
    Educational,
}

impl UsageType {
    /// Return `true` if this usage type requires an explicit clearance before use.
    pub fn requires_clearance(&self) -> bool {
        matches!(
            self,
            UsageType::Broadcast | UsageType::Theatrical | UsageType::Advertising
        )
    }
}

/// A single usage right granted for a media asset.
#[derive(Debug, Clone)]
pub struct UsageRight {
    /// The kind of usage this right permits.
    pub usage_type: UsageType,
    /// Human-readable identifier for the asset this right applies to.
    pub asset_id: String,
    /// Unix timestamp (seconds) at which this right expires, if any.
    pub expires_at: Option<u64>,
}

impl UsageRight {
    /// Create a new `UsageRight`.
    pub fn new(
        usage_type: UsageType,
        asset_id: impl Into<String>,
        expires_at: Option<u64>,
    ) -> Self {
        Self {
            usage_type,
            asset_id: asset_id.into(),
            expires_at,
        }
    }

    /// Return `true` if this right has expired at or before the given Unix timestamp.
    pub fn is_expired_at(&self, ts: u64) -> bool {
        match self.expires_at {
            Some(exp) => ts >= exp,
            None => false,
        }
    }
}

/// A collection of usage rights associated with a media asset.
#[derive(Debug, Default, Clone)]
pub struct UsageRights {
    rights: Vec<UsageRight>,
}

impl UsageRights {
    /// Create an empty `UsageRights` collection.
    pub fn new() -> Self {
        Self { rights: Vec::new() }
    }

    /// Add a usage right to this collection.
    pub fn add(&mut self, right: UsageRight) {
        self.rights.push(right);
    }

    /// Return `true` if there is at least one non-expired right of the given type
    /// at the specified Unix timestamp.
    pub fn can_use(&self, usage_type: &UsageType, at_ts: u64) -> bool {
        self.rights
            .iter()
            .any(|r| &r.usage_type == usage_type && !r.is_expired_at(at_ts))
    }

    /// Return all rights that have expired at or before the given timestamp.
    pub fn expired_rights(&self, at_ts: u64) -> Vec<&UsageRight> {
        self.rights
            .iter()
            .filter(|r| r.is_expired_at(at_ts))
            .collect()
    }

    /// Return the total number of rights in this collection.
    pub fn len(&self) -> usize {
        self.rights.len()
    }

    /// Return `true` if this collection contains no rights.
    pub fn is_empty(&self) -> bool {
        self.rights.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_requires_clearance() {
        assert!(UsageType::Broadcast.requires_clearance());
    }

    #[test]
    fn test_theatrical_requires_clearance() {
        assert!(UsageType::Theatrical.requires_clearance());
    }

    #[test]
    fn test_advertising_requires_clearance() {
        assert!(UsageType::Advertising.requires_clearance());
    }

    #[test]
    fn test_streaming_no_clearance() {
        assert!(!UsageType::Streaming.requires_clearance());
    }

    #[test]
    fn test_educational_no_clearance() {
        assert!(!UsageType::Educational.requires_clearance());
    }

    #[test]
    fn test_usage_right_not_expired() {
        let right = UsageRight::new(UsageType::Streaming, "asset-1", Some(2000));
        assert!(!right.is_expired_at(1999));
    }

    #[test]
    fn test_usage_right_expired_at_boundary() {
        let right = UsageRight::new(UsageType::Streaming, "asset-1", Some(2000));
        assert!(right.is_expired_at(2000));
    }

    #[test]
    fn test_usage_right_expired_past_expiry() {
        let right = UsageRight::new(UsageType::Broadcast, "asset-2", Some(1000));
        assert!(right.is_expired_at(5000));
    }

    #[test]
    fn test_usage_right_no_expiry_never_expires() {
        let right = UsageRight::new(UsageType::Editorial, "asset-3", None);
        assert!(!right.is_expired_at(u64::MAX));
    }

    #[test]
    fn test_usage_rights_empty() {
        let ur = UsageRights::new();
        assert!(ur.is_empty());
        assert_eq!(ur.len(), 0);
    }

    #[test]
    fn test_usage_rights_can_use_valid() {
        let mut ur = UsageRights::new();
        ur.add(UsageRight::new(UsageType::Streaming, "a", Some(9999)));
        assert!(ur.can_use(&UsageType::Streaming, 100));
    }

    #[test]
    fn test_usage_rights_cannot_use_expired() {
        let mut ur = UsageRights::new();
        ur.add(UsageRight::new(UsageType::Streaming, "a", Some(100)));
        assert!(!ur.can_use(&UsageType::Streaming, 200));
    }

    #[test]
    fn test_usage_rights_cannot_use_wrong_type() {
        let mut ur = UsageRights::new();
        ur.add(UsageRight::new(UsageType::Streaming, "a", None));
        assert!(!ur.can_use(&UsageType::Broadcast, 0));
    }

    #[test]
    fn test_expired_rights_count() {
        let mut ur = UsageRights::new();
        ur.add(UsageRight::new(UsageType::Streaming, "a", Some(100)));
        ur.add(UsageRight::new(UsageType::Broadcast, "b", Some(200)));
        ur.add(UsageRight::new(UsageType::Editorial, "c", None));
        let expired = ur.expired_rights(150);
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0].asset_id, "a");
    }

    #[test]
    fn test_usage_rights_add_increases_len() {
        let mut ur = UsageRights::new();
        ur.add(UsageRight::new(UsageType::Corporate, "x", None));
        ur.add(UsageRight::new(UsageType::Theatrical, "y", Some(500)));
        assert_eq!(ur.len(), 2);
    }
}
