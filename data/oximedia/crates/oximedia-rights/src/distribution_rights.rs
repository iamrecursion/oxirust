//! Distribution rights management for media assets.
//!
//! Models the rights that govern *how* and *where* content may be distributed,
//! covering channels (streaming, broadcast, theatrical, …), territories, and
//! time windows.  A [`DistributionPolicy`] groups one or more
//! [`DistributionRight`]s and can evaluate whether a proposed distribution
//! action is permitted.

#![allow(dead_code)]

/// The channel through which content is distributed.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DistributionChannel {
    /// Online streaming services (SVOD, AVOD, TVOD).
    Streaming,
    /// Terrestrial, satellite, or cable broadcast.
    Broadcast,
    /// Theatrical release in cinemas.
    Theatrical,
    /// Physical media (DVD, Blu-ray, UHD Blu-ray).
    PhysicalMedia,
    /// Electronic sell-through / permanent digital download.
    Est,
    /// Short-form video platforms (YouTube, TikTok, Reels, …).
    SocialMedia,
    /// In-flight entertainment systems.
    InFlight,
    /// Hotel or hospitality on-demand systems.
    Hospitality,
    /// Educational institutions and libraries.
    Educational,
    /// Any channel not enumerated above.
    Other(String),
}

impl DistributionChannel {
    /// Return a short identifier string for this channel.
    #[must_use]
    pub fn identifier(&self) -> &str {
        match self {
            Self::Streaming => "streaming",
            Self::Broadcast => "broadcast",
            Self::Theatrical => "theatrical",
            Self::PhysicalMedia => "physical",
            Self::Est => "est",
            Self::SocialMedia => "social",
            Self::InFlight => "inflight",
            Self::Hospitality => "hospitality",
            Self::Educational => "educational",
            Self::Other(s) => s.as_str(),
        }
    }

    /// Returns `true` if this channel requires a network connection to deliver
    /// content to the end user.
    #[must_use]
    pub fn is_digital(&self) -> bool {
        matches!(
            self,
            Self::Streaming | Self::Est | Self::SocialMedia | Self::Hospitality | Self::InFlight
        )
    }
}

/// A single distribution right, granting permission to distribute content
/// through a specific channel within a defined territory and time window.
///
/// # Example
///
/// ```
/// use oximedia_rights::distribution_rights::{DistributionChannel, DistributionRight};
///
/// let right = DistributionRight::new("dr-001", "asset-007", DistributionChannel::Streaming)
///     .with_territory("US")
///     .with_window(1_700_000_000, 1_800_000_000);
///
/// assert!(right.is_active(1_750_000_000));
/// assert!(!right.is_active(1_900_000_000));
/// ```
#[derive(Debug, Clone)]
pub struct DistributionRight {
    /// Unique identifier for this right.
    pub id: String,
    /// Identifier of the asset this right applies to.
    pub asset_id: String,
    /// Distribution channel this right covers.
    pub channel: DistributionChannel,
    /// ISO 3166-1 alpha-2 territory code, or `"WW"` for worldwide.
    pub territory: Option<String>,
    /// Unix timestamp (seconds) from which this right is valid.
    pub valid_from: Option<i64>,
    /// Unix timestamp (seconds) at which this right expires.
    pub valid_until: Option<i64>,
    /// Whether sublicensing is permitted.
    pub sublicensable: bool,
    /// Whether simultaneous distribution on multiple platforms is permitted.
    pub exclusive: bool,
}

impl DistributionRight {
    /// Create a new `DistributionRight` with the given id, asset, and channel.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        asset_id: impl Into<String>,
        channel: DistributionChannel,
    ) -> Self {
        Self {
            id: id.into(),
            asset_id: asset_id.into(),
            channel,
            territory: None,
            valid_from: None,
            valid_until: None,
            sublicensable: false,
            exclusive: false,
        }
    }

    /// Restrict to a specific territory.
    #[must_use]
    pub fn with_territory(mut self, territory: impl Into<String>) -> Self {
        self.territory = Some(territory.into());
        self
    }

    /// Set the validity window (Unix seconds).
    #[must_use]
    pub fn with_window(mut self, from: i64, until: i64) -> Self {
        self.valid_from = Some(from);
        self.valid_until = Some(until);
        self
    }

    /// Mark the right as sublicensable.
    #[must_use]
    pub fn sublicensable(mut self) -> Self {
        self.sublicensable = true;
        self
    }

    /// Mark the right as exclusive.
    #[must_use]
    pub fn exclusive(mut self) -> Self {
        self.exclusive = true;
        self
    }

    /// Returns `true` if this right is active at the given Unix timestamp.
    ///
    /// A right with no window set is considered perpetually active.
    #[must_use]
    pub fn is_active(&self, timestamp: i64) -> bool {
        let after_start = self.valid_from.is_none_or(|f| timestamp >= f);
        let before_end = self.valid_until.is_none_or(|u| timestamp < u);
        after_start && before_end
    }

    /// Returns `true` if this right covers the given territory.
    ///
    /// A right with no territory set is considered worldwide.
    #[must_use]
    pub fn covers_territory(&self, territory: &str) -> bool {
        match &self.territory {
            None => true,
            Some(t) => t == "WW" || t.eq_ignore_ascii_case(territory),
        }
    }
}

/// A policy that groups multiple distribution rights for a single asset and
/// can evaluate permission queries.
///
/// # Example
///
/// ```
/// use oximedia_rights::distribution_rights::{
///     DistributionChannel, DistributionPolicy, DistributionRight,
/// };
///
/// let mut policy = DistributionPolicy::new("asset-007");
/// policy.add_right(
///     DistributionRight::new("r1", "asset-007", DistributionChannel::Streaming)
///         .with_territory("US"),
/// );
/// policy.add_right(
///     DistributionRight::new("r2", "asset-007", DistributionChannel::Broadcast)
///         .with_territory("GB"),
/// );
///
/// assert!(policy.is_permitted(&DistributionChannel::Streaming, "US", 0));
/// assert!(!policy.is_permitted(&DistributionChannel::Streaming, "GB", 0));
/// ```
#[derive(Debug, Default)]
pub struct DistributionPolicy {
    /// Asset this policy governs.
    pub asset_id: String,
    rights: Vec<DistributionRight>,
}

impl DistributionPolicy {
    /// Create an empty policy for the given asset.
    #[must_use]
    pub fn new(asset_id: impl Into<String>) -> Self {
        Self {
            asset_id: asset_id.into(),
            rights: Vec::new(),
        }
    }

    /// Add a distribution right to the policy.
    pub fn add_right(&mut self, right: DistributionRight) {
        self.rights.push(right);
    }

    /// Return the number of rights in the policy.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rights.len()
    }

    /// Returns `true` if the policy has no rights.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rights.is_empty()
    }

    /// Returns `true` if at least one right in the policy permits distribution
    /// through `channel` in `territory` at `timestamp`.
    #[must_use]
    pub fn is_permitted(
        &self,
        channel: &DistributionChannel,
        territory: &str,
        timestamp: i64,
    ) -> bool {
        self.rights.iter().any(|r| {
            &r.channel == channel && r.covers_territory(territory) && r.is_active(timestamp)
        })
    }

    /// Return all rights that are currently active at the given timestamp.
    #[must_use]
    pub fn active_rights(&self, timestamp: i64) -> Vec<&DistributionRight> {
        self.rights
            .iter()
            .filter(|r| r.is_active(timestamp))
            .collect()
    }

    /// Return all rights covering the specified territory (active or not).
    #[must_use]
    pub fn rights_for_territory(&self, territory: &str) -> Vec<&DistributionRight> {
        self.rights
            .iter()
            .filter(|r| r.covers_territory(territory))
            .collect()
    }

    /// Return all rights for the specified channel (active or not).
    #[must_use]
    pub fn rights_for_channel(&self, channel: &DistributionChannel) -> Vec<&DistributionRight> {
        self.rights
            .iter()
            .filter(|r| &r.channel == channel)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_policy() -> DistributionPolicy {
        let mut policy = DistributionPolicy::new("asset-007");
        policy.add_right(
            DistributionRight::new("r1", "asset-007", DistributionChannel::Streaming)
                .with_territory("US")
                .with_window(1_000, 2_000),
        );
        policy.add_right(
            DistributionRight::new("r2", "asset-007", DistributionChannel::Broadcast)
                .with_territory("GB"),
        );
        policy.add_right(
            DistributionRight::new("r3", "asset-007", DistributionChannel::Streaming)
                .with_territory("WW")
                .with_window(3_000, 5_000),
        );
        policy
    }

    #[test]
    fn test_policy_len() {
        assert_eq!(make_policy().len(), 3);
    }

    #[test]
    fn test_permitted_streaming_us_in_window() {
        assert!(make_policy().is_permitted(&DistributionChannel::Streaming, "US", 1_500));
    }

    #[test]
    fn test_not_permitted_streaming_us_outside_window() {
        // Right r1 expired at 2000; r3 starts at 3000.
        assert!(!make_policy().is_permitted(&DistributionChannel::Streaming, "US", 2_500));
    }

    #[test]
    fn test_permitted_broadcast_gb_no_window() {
        // r2 has no window so always active.
        assert!(make_policy().is_permitted(&DistributionChannel::Broadcast, "GB", 999_999));
    }

    #[test]
    fn test_not_permitted_broadcast_us() {
        // No broadcast right for US.
        assert!(!make_policy().is_permitted(&DistributionChannel::Broadcast, "US", 0));
    }

    #[test]
    fn test_worldwide_right_covers_any_territory() {
        // r3 covers WW at timestamp 4_000.
        assert!(make_policy().is_permitted(&DistributionChannel::Streaming, "JP", 4_000));
        assert!(make_policy().is_permitted(&DistributionChannel::Streaming, "DE", 4_000));
    }

    #[test]
    fn test_active_rights_count() {
        // At timestamp 1_500: r1 (streaming US) and r2 (broadcast GB, no window) are active.
        let policy = make_policy();
        let active = policy.active_rights(1_500);
        assert_eq!(active.len(), 2);
    }

    #[test]
    fn test_active_rights_at_gap() {
        // At timestamp 2_500: r1 expired, r3 not started yet, only r2.
        let policy = make_policy();
        let active = policy.active_rights(2_500);
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].id, "r2");
    }

    #[test]
    fn test_rights_for_territory_us() {
        // r1 (US) and r3 (WW) cover US.
        let policy = make_policy();
        let rights = policy.rights_for_territory("US");
        assert_eq!(rights.len(), 2);
    }

    #[test]
    fn test_rights_for_channel_streaming() {
        assert_eq!(
            make_policy()
                .rights_for_channel(&DistributionChannel::Streaming)
                .len(),
            2
        );
    }

    #[test]
    fn test_distribution_channel_identifier() {
        assert_eq!(DistributionChannel::Streaming.identifier(), "streaming");
        assert_eq!(DistributionChannel::Broadcast.identifier(), "broadcast");
        assert_eq!(DistributionChannel::Educational.identifier(), "educational");
    }

    #[test]
    fn test_distribution_channel_is_digital() {
        assert!(DistributionChannel::Streaming.is_digital());
        assert!(DistributionChannel::Est.is_digital());
        assert!(!DistributionChannel::Theatrical.is_digital());
        assert!(!DistributionChannel::PhysicalMedia.is_digital());
    }

    #[test]
    fn test_right_is_active_no_window() {
        let right = DistributionRight::new("x", "a", DistributionChannel::Theatrical);
        assert!(right.is_active(0));
        assert!(right.is_active(i64::MAX));
    }

    #[test]
    fn test_right_covers_territory_none_means_worldwide() {
        let right = DistributionRight::new("x", "a", DistributionChannel::Est);
        assert!(right.covers_territory("JP"));
        assert!(right.covers_territory("US"));
    }

    #[test]
    fn test_empty_policy() {
        let policy = DistributionPolicy::new("asset-x");
        assert!(policy.is_empty());
        assert!(!policy.is_permitted(&DistributionChannel::Streaming, "US", 0));
    }
}
