//! DRM playback policy definitions.
//!
//! Provides:
//! - [`OutputControl`]: controls whether output to external displays is allowed
//! - [`PlaybackConstraint`]: resolution and HDCP constraints for playback
//! - [`RentalPolicy`]: time- and play-count-based rental rules
//! - [`DrmPolicy`]: top-level policy combining constraints and rental terms

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

// ---------------------------------------------------------------------------
// OutputControl
// ---------------------------------------------------------------------------

/// Controls whether output to an external display is permitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputControl {
    /// Output is fully allowed.
    Allow,
    /// Output is allowed but with restrictions (e.g., downscaled).
    Restrict,
    /// Output is denied.
    Deny,
}

impl OutputControl {
    /// Returns `true` if output is allowed (either `Allow` or `Restrict`).
    #[must_use]
    pub fn is_allowed(self) -> bool {
        matches!(self, OutputControl::Allow | OutputControl::Restrict)
    }

    /// Returns `true` if output is completely denied.
    #[must_use]
    pub fn is_denied(self) -> bool {
        matches!(self, OutputControl::Deny)
    }

    /// Returns `true` if output is restricted (but not denied).
    #[must_use]
    pub fn is_restricted(self) -> bool {
        matches!(self, OutputControl::Restrict)
    }
}

// ---------------------------------------------------------------------------
// PlaybackConstraint
// ---------------------------------------------------------------------------

/// Resolution and copy-protection constraints for playback.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlaybackConstraint {
    /// Maximum allowed horizontal resolution in pixels.
    pub max_resolution_width: u32,
    /// Maximum allowed vertical resolution in pixels.
    pub max_resolution_height: u32,
    /// Whether HDCP is required for external output.
    pub hdcp_required: bool,
    /// Output control level.
    pub output_control: OutputControl,
}

impl PlaybackConstraint {
    /// Default constraint for streaming (HD allowed, HDCP required).
    #[must_use]
    pub fn default_streaming() -> Self {
        Self {
            max_resolution_width: 1920,
            max_resolution_height: 1080,
            hdcp_required: true,
            output_control: OutputControl::Allow,
        }
    }

    /// Default constraint for downloaded content (4K allowed, HDCP required).
    #[must_use]
    pub fn default_download() -> Self {
        Self {
            max_resolution_width: 3840,
            max_resolution_height: 2160,
            hdcp_required: true,
            output_control: OutputControl::Allow,
        }
    }

    /// Returns `true` if the constraints allow 4K (≥ 3840×2160) playback.
    #[must_use]
    pub fn allows_4k(&self) -> bool {
        self.max_resolution_width >= 3840 && self.max_resolution_height >= 2160
    }

    /// Returns `true` if the constraint allows HD (≥ 1280×720).
    #[must_use]
    pub fn allows_hd(&self) -> bool {
        self.max_resolution_width >= 1280 && self.max_resolution_height >= 720
    }

    /// Check whether a given resolution falls within the allowed limits.
    #[must_use]
    pub fn allows_resolution(&self, width: u32, height: u32) -> bool {
        width <= self.max_resolution_width && height <= self.max_resolution_height
    }
}

// ---------------------------------------------------------------------------
// RentalPolicy
// ---------------------------------------------------------------------------

/// Rules governing rental (time- and play-count-limited) content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RentalPolicy {
    /// How long the rental is valid after the first play, in hours.
    pub duration_hours: u32,
    /// Maximum number of times the content may be played.
    pub max_plays: u32,
    /// Whether offline (downloaded) playback is permitted during the rental.
    pub offline_allowed: bool,
}

impl RentalPolicy {
    /// Create a standard 48-hour, 3-play rental.
    #[must_use]
    pub fn standard() -> Self {
        Self {
            duration_hours: 48,
            max_plays: 3,
            offline_allowed: false,
        }
    }

    /// Returns `true` if the rental has expired based on the start epoch and current epoch.
    ///
    /// `start_epoch` and `now_epoch` are Unix timestamps in **seconds**.
    #[must_use]
    pub fn is_expired(&self, start_epoch: u64, now_epoch: u64) -> bool {
        let duration_secs = self.duration_hours as u64 * 3600;
        now_epoch >= start_epoch.saturating_add(duration_secs)
    }

    /// Returns how many plays remain given `used` plays so far.
    ///
    /// Returns `0` if the limit has been reached or exceeded.
    #[must_use]
    pub fn plays_remaining(&self, used: u32) -> u32 {
        self.max_plays.saturating_sub(used)
    }

    /// Returns `true` if the rental can still be played (`used < max_plays`).
    #[must_use]
    pub fn can_play(&self, used: u32) -> bool {
        used < self.max_plays
    }
}

// ---------------------------------------------------------------------------
// DrmPolicy
// ---------------------------------------------------------------------------

/// Top-level DRM policy for a piece of content.
#[derive(Debug, Clone)]
pub struct DrmPolicy {
    /// Unique content identifier.
    pub content_id: String,
    /// Playback constraints.
    pub constraint: PlaybackConstraint,
    /// Optional rental terms.  `None` means permanent purchase.
    pub rental: Option<RentalPolicy>,
}

impl DrmPolicy {
    /// Create a new policy.
    #[must_use]
    pub fn new(
        content_id: impl Into<String>,
        constraint: PlaybackConstraint,
        rental: Option<RentalPolicy>,
    ) -> Self {
        Self {
            content_id: content_id.into(),
            constraint,
            rental,
        }
    }

    /// Returns `true` if this policy is for a rental (has a `RentalPolicy`).
    #[must_use]
    pub fn is_rental(&self) -> bool {
        self.rental.is_some()
    }

    /// Returns `true` if external output is allowed under this policy.
    #[must_use]
    pub fn allows_output(&self) -> bool {
        self.constraint.output_control.is_allowed()
    }

    /// Check whether the policy allows 4K playback.
    #[must_use]
    pub fn allows_4k(&self) -> bool {
        self.constraint.allows_4k()
    }

    /// Check rental expiry (returns `false` if not a rental or not expired).
    #[must_use]
    pub fn is_rental_expired(&self, start_epoch: u64, now_epoch: u64) -> bool {
        self.rental
            .as_ref()
            .map(|r| r.is_expired(start_epoch, now_epoch))
            .unwrap_or(false)
    }

    /// Remaining plays (returns `None` if not a rental).
    #[must_use]
    pub fn plays_remaining(&self, used: u32) -> Option<u32> {
        self.rental.as_ref().map(|r| r.plays_remaining(used))
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- OutputControl ----

    #[test]
    fn test_output_control_allow_is_allowed() {
        assert!(OutputControl::Allow.is_allowed());
        assert!(!OutputControl::Allow.is_denied());
        assert!(!OutputControl::Allow.is_restricted());
    }

    #[test]
    fn test_output_control_restrict_is_allowed() {
        assert!(OutputControl::Restrict.is_allowed());
        assert!(!OutputControl::Restrict.is_denied());
        assert!(OutputControl::Restrict.is_restricted());
    }

    #[test]
    fn test_output_control_deny() {
        assert!(!OutputControl::Deny.is_allowed());
        assert!(OutputControl::Deny.is_denied());
        assert!(!OutputControl::Deny.is_restricted());
    }

    // ---- PlaybackConstraint ----

    #[test]
    fn test_default_streaming_hd_not_4k() {
        let c = PlaybackConstraint::default_streaming();
        assert!(c.allows_hd());
        assert!(!c.allows_4k());
        assert!(c.hdcp_required);
    }

    #[test]
    fn test_default_download_allows_4k() {
        let c = PlaybackConstraint::default_download();
        assert!(c.allows_4k());
        assert!(c.allows_hd());
    }

    #[test]
    fn test_allows_resolution_within_limits() {
        let c = PlaybackConstraint::default_streaming();
        assert!(c.allows_resolution(1280, 720));
        assert!(c.allows_resolution(1920, 1080));
        assert!(!c.allows_resolution(3840, 2160));
    }

    #[test]
    fn test_constraint_output_allow() {
        let c = PlaybackConstraint::default_streaming();
        assert_eq!(c.output_control, OutputControl::Allow);
    }

    #[test]
    fn test_constraint_deny_output() {
        let c = PlaybackConstraint {
            max_resolution_width: 1920,
            max_resolution_height: 1080,
            hdcp_required: true,
            output_control: OutputControl::Deny,
        };
        assert!(!c.output_control.is_allowed());
    }

    // ---- RentalPolicy ----

    #[test]
    fn test_rental_not_expired_immediately() {
        let r = RentalPolicy::standard();
        // start and now are the same
        assert!(!r.is_expired(1_000_000, 1_000_000));
    }

    #[test]
    fn test_rental_expired_after_duration() {
        let r = RentalPolicy::standard(); // 48 hours
        let start = 0u64;
        let now = 48 * 3600 + 1;
        assert!(r.is_expired(start, now));
    }

    #[test]
    fn test_rental_not_expired_before_duration() {
        let r = RentalPolicy::standard();
        let start = 0u64;
        let now = 47 * 3600;
        assert!(!r.is_expired(start, now));
    }

    #[test]
    fn test_plays_remaining_zero_used() {
        let r = RentalPolicy::standard(); // max_plays = 3
        assert_eq!(r.plays_remaining(0), 3);
    }

    #[test]
    fn test_plays_remaining_saturates() {
        let r = RentalPolicy::standard();
        assert_eq!(r.plays_remaining(5), 0);
    }

    #[test]
    fn test_can_play() {
        let r = RentalPolicy::standard();
        assert!(r.can_play(2));
        assert!(!r.can_play(3));
    }

    // ---- DrmPolicy ----

    #[test]
    fn test_drm_policy_is_rental() {
        let p = DrmPolicy::new(
            "content-1",
            PlaybackConstraint::default_streaming(),
            Some(RentalPolicy::standard()),
        );
        assert!(p.is_rental());
    }

    #[test]
    fn test_drm_policy_not_rental() {
        let p = DrmPolicy::new("content-2", PlaybackConstraint::default_download(), None);
        assert!(!p.is_rental());
    }

    #[test]
    fn test_drm_policy_allows_output() {
        let p = DrmPolicy::new("c-3", PlaybackConstraint::default_streaming(), None);
        assert!(p.allows_output());
    }

    #[test]
    fn test_drm_policy_rental_expiry_propagation() {
        let p = DrmPolicy::new(
            "c-4",
            PlaybackConstraint::default_streaming(),
            Some(RentalPolicy::standard()),
        );
        assert!(!p.is_rental_expired(0, 1000));
        assert!(p.is_rental_expired(0, 48 * 3600 + 10));
    }
}
