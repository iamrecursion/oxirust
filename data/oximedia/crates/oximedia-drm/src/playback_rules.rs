//! Runtime playback restriction rules for DRM-protected content.
//!
//! Provides a composable set of [`PlaybackRestriction`] values and a
//! [`PlaybackRulesEngine`] that enforces them against a [`PlaybackContext`].

#![allow(dead_code)]

use std::collections::HashSet;

/// A single playback restriction that can be applied to content.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PlaybackRestriction {
    /// Content may only be played on devices with a certified secure display.
    RequireHdcp,
    /// Content may only be played in offline mode (no network needed).
    OfflineOnly,
    /// Content may only be played while connected to a network.
    OnlineOnly,
    /// Maximum playback resolution is constrained to the given pixel height.
    MaxResolution(u32),
    /// Playback is only permitted in the listed country codes (ISO 3166-1 alpha-2).
    AllowedRegions(Vec<String>),
    /// The content may not be screen-captured or recorded.
    NoScreenCapture,
    /// Content expires after this many seconds from first play.
    RentalExpirySeconds(u64),
}

/// Runtime context checked against [`PlaybackRules`].
#[derive(Debug, Clone, Default)]
pub struct PlaybackContext {
    /// Whether the current device supports HDCP.
    pub has_hdcp: bool,
    /// Whether the device is currently connected to a network.
    pub is_online: bool,
    /// Requested playback resolution (pixel height, e.g. 1080).
    pub requested_resolution: u32,
    /// Two-letter country code of the viewer (e.g. `"US"`).
    pub country_code: String,
    /// Whether the playback environment can capture the screen.
    pub can_capture_screen: bool,
    /// Seconds elapsed since first play for rental content.
    pub rental_elapsed_secs: u64,
}

/// A set of restrictions that apply to a piece of content.
#[derive(Debug, Clone, Default)]
pub struct PlaybackRules {
    restrictions: Vec<PlaybackRestriction>,
}

impl PlaybackRules {
    /// Create an empty [`PlaybackRules`] set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a restriction.
    pub fn add(mut self, restriction: PlaybackRestriction) -> Self {
        self.restrictions.push(restriction);
        self
    }

    /// Return the list of restrictions.
    pub fn restrictions(&self) -> &[PlaybackRestriction] {
        &self.restrictions
    }

    /// Return `true` if there are no restrictions.
    pub fn is_empty(&self) -> bool {
        self.restrictions.is_empty()
    }
}

/// The reason a playback check failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlaybackDeniedReason {
    /// HDCP is required but unavailable on the device.
    HdcpUnavailable,
    /// Content is offline-only but the device is online.
    MustBeOffline,
    /// Content is online-only but the device is offline.
    MustBeOnline,
    /// Requested resolution exceeds the allowed maximum.
    ResolutionTooHigh {
        /// Requested pixel height.
        requested: u32,
        /// Maximum allowed pixel height.
        max_allowed: u32,
    },
    /// Viewer's region is not in the allowed list.
    RegionNotAllowed { country: String },
    /// Screen capture is not permitted.
    ScreenCaptureNotAllowed,
    /// Rental period has expired.
    RentalExpired {
        /// Elapsed seconds since first play.
        elapsed: u64,
        /// Configured expiry in seconds.
        limit: u64,
    },
}

/// Result of a playback rules check.
#[derive(Debug, Clone)]
pub struct PlaybackCheckResult {
    /// `true` if all restrictions passed.
    pub allowed: bool,
    /// All reasons playback was denied (empty when `allowed = true`).
    pub denied_reasons: Vec<PlaybackDeniedReason>,
}

impl PlaybackCheckResult {
    fn permit() -> Self {
        Self {
            allowed: true,
            denied_reasons: vec![],
        }
    }
}

/// Enforces [`PlaybackRules`] against a [`PlaybackContext`].
pub struct PlaybackRulesEngine;

impl PlaybackRulesEngine {
    /// Create a new engine.
    pub fn new() -> Self {
        Self
    }

    /// Evaluate every restriction in `rules` against `ctx`.
    ///
    /// Returns a [`PlaybackCheckResult`] with `allowed = true` only when
    /// every restriction passes.
    pub fn check(&self, rules: &PlaybackRules, ctx: &PlaybackContext) -> PlaybackCheckResult {
        let mut reasons = Vec::new();

        for r in rules.restrictions() {
            match r {
                PlaybackRestriction::RequireHdcp => {
                    if !ctx.has_hdcp {
                        reasons.push(PlaybackDeniedReason::HdcpUnavailable);
                    }
                }
                PlaybackRestriction::OfflineOnly => {
                    if ctx.is_online {
                        reasons.push(PlaybackDeniedReason::MustBeOffline);
                    }
                }
                PlaybackRestriction::OnlineOnly => {
                    if !ctx.is_online {
                        reasons.push(PlaybackDeniedReason::MustBeOnline);
                    }
                }
                PlaybackRestriction::MaxResolution(max) => {
                    if ctx.requested_resolution > *max {
                        reasons.push(PlaybackDeniedReason::ResolutionTooHigh {
                            requested: ctx.requested_resolution,
                            max_allowed: *max,
                        });
                    }
                }
                PlaybackRestriction::AllowedRegions(regions) => {
                    let allowed: HashSet<&String> = regions.iter().collect();
                    if !allowed.contains(&ctx.country_code) {
                        reasons.push(PlaybackDeniedReason::RegionNotAllowed {
                            country: ctx.country_code.clone(),
                        });
                    }
                }
                PlaybackRestriction::NoScreenCapture => {
                    if ctx.can_capture_screen {
                        reasons.push(PlaybackDeniedReason::ScreenCaptureNotAllowed);
                    }
                }
                PlaybackRestriction::RentalExpirySeconds(limit) => {
                    if ctx.rental_elapsed_secs > *limit {
                        reasons.push(PlaybackDeniedReason::RentalExpired {
                            elapsed: ctx.rental_elapsed_secs,
                            limit: *limit,
                        });
                    }
                }
            }
        }

        PlaybackCheckResult {
            allowed: reasons.is_empty(),
            denied_reasons: reasons,
        }
    }
}

impl Default for PlaybackRulesEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> PlaybackRulesEngine {
        PlaybackRulesEngine::new()
    }

    fn base_ctx() -> PlaybackContext {
        PlaybackContext {
            has_hdcp: true,
            is_online: true,
            requested_resolution: 1080,
            country_code: "US".to_string(),
            can_capture_screen: false,
            rental_elapsed_secs: 0,
        }
    }

    #[test]
    fn test_no_restrictions_allows() {
        let rules = PlaybackRules::new();
        let result = engine().check(&rules, &base_ctx());
        assert!(result.allowed);
        assert!(result.denied_reasons.is_empty());
    }

    #[test]
    fn test_hdcp_required_passes_when_present() {
        let rules = PlaybackRules::new().add(PlaybackRestriction::RequireHdcp);
        let result = engine().check(&rules, &base_ctx());
        assert!(result.allowed);
    }

    #[test]
    fn test_hdcp_required_fails_when_absent() {
        let rules = PlaybackRules::new().add(PlaybackRestriction::RequireHdcp);
        let ctx = PlaybackContext {
            has_hdcp: false,
            ..base_ctx()
        };
        let result = engine().check(&rules, &ctx);
        assert!(!result.allowed);
        assert!(result
            .denied_reasons
            .contains(&PlaybackDeniedReason::HdcpUnavailable));
    }

    #[test]
    fn test_online_only_passes_when_online() {
        let rules = PlaybackRules::new().add(PlaybackRestriction::OnlineOnly);
        let result = engine().check(&rules, &base_ctx());
        assert!(result.allowed);
    }

    #[test]
    fn test_online_only_fails_when_offline() {
        let rules = PlaybackRules::new().add(PlaybackRestriction::OnlineOnly);
        let ctx = PlaybackContext {
            is_online: false,
            ..base_ctx()
        };
        let result = engine().check(&rules, &ctx);
        assert!(!result.allowed);
        assert!(result
            .denied_reasons
            .contains(&PlaybackDeniedReason::MustBeOnline));
    }

    #[test]
    fn test_offline_only_fails_when_online() {
        let rules = PlaybackRules::new().add(PlaybackRestriction::OfflineOnly);
        let result = engine().check(&rules, &base_ctx());
        assert!(!result.allowed);
    }

    #[test]
    fn test_max_resolution_passes() {
        let rules = PlaybackRules::new().add(PlaybackRestriction::MaxResolution(1080));
        let result = engine().check(&rules, &base_ctx());
        assert!(result.allowed);
    }

    #[test]
    fn test_max_resolution_fails_when_too_high() {
        let rules = PlaybackRules::new().add(PlaybackRestriction::MaxResolution(720));
        let result = engine().check(&rules, &base_ctx());
        assert!(!result.allowed);
        assert!(matches!(
            result.denied_reasons[0],
            PlaybackDeniedReason::ResolutionTooHigh {
                requested: 1080,
                max_allowed: 720
            }
        ));
    }

    #[test]
    fn test_region_allowed_passes() {
        let rules = PlaybackRules::new().add(PlaybackRestriction::AllowedRegions(vec![
            "US".to_string(),
            "GB".to_string(),
        ]));
        let result = engine().check(&rules, &base_ctx());
        assert!(result.allowed);
    }

    #[test]
    fn test_region_denied_for_unlisted_country() {
        let rules =
            PlaybackRules::new().add(PlaybackRestriction::AllowedRegions(vec!["GB".to_string()]));
        let result = engine().check(&rules, &base_ctx()); // ctx has "US"
        assert!(!result.allowed);
    }

    #[test]
    fn test_no_screen_capture_passes_when_no_capture() {
        let rules = PlaybackRules::new().add(PlaybackRestriction::NoScreenCapture);
        let result = engine().check(&rules, &base_ctx());
        assert!(result.allowed);
    }

    #[test]
    fn test_no_screen_capture_fails_when_capture_present() {
        let rules = PlaybackRules::new().add(PlaybackRestriction::NoScreenCapture);
        let ctx = PlaybackContext {
            can_capture_screen: true,
            ..base_ctx()
        };
        let result = engine().check(&rules, &ctx);
        assert!(!result.allowed);
    }

    #[test]
    fn test_rental_expiry_passes_within_limit() {
        let rules = PlaybackRules::new().add(PlaybackRestriction::RentalExpirySeconds(3600));
        let ctx = PlaybackContext {
            rental_elapsed_secs: 1000,
            ..base_ctx()
        };
        let result = engine().check(&rules, &ctx);
        assert!(result.allowed);
    }

    #[test]
    fn test_rental_expiry_fails_when_exceeded() {
        let rules = PlaybackRules::new().add(PlaybackRestriction::RentalExpirySeconds(3600));
        let ctx = PlaybackContext {
            rental_elapsed_secs: 7200,
            ..base_ctx()
        };
        let result = engine().check(&rules, &ctx);
        assert!(!result.allowed);
    }

    #[test]
    fn test_multiple_violations_all_reported() {
        let rules = PlaybackRules::new()
            .add(PlaybackRestriction::RequireHdcp)
            .add(PlaybackRestriction::MaxResolution(480));
        let ctx = PlaybackContext {
            has_hdcp: false,
            requested_resolution: 1080,
            ..base_ctx()
        };
        let result = engine().check(&rules, &ctx);
        assert!(!result.allowed);
        assert_eq!(result.denied_reasons.len(), 2);
    }

    #[test]
    fn test_rules_is_empty() {
        let rules = PlaybackRules::new();
        assert!(rules.is_empty());
        let rules2 = rules.add(PlaybackRestriction::RequireHdcp);
        assert!(!rules2.is_empty());
    }
}
