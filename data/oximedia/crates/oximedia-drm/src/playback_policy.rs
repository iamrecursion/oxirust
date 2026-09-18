#![allow(dead_code)]
//! Playback policy evaluation for DRM content.
//!
//! Defines a composable `PlaybackPolicy` describing what is and is not
//! permitted for a piece of content, and a `PlaybackPolicyEvaluator` that
//! tests a requested set of restrictions against such a policy.

/// Enumeration of individual playback restrictions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PlaybackRestriction {
    /// Content cannot be played back offline.
    NoOffline,
    /// Content cannot be downloaded to a device.
    NoDownload,
    /// Screen capture / casting is prohibited.
    NoScreenCapture,
    /// Number of simultaneous streams is capped.
    MaxStreams(u32),
    /// Playback is restricted to a geographic region.
    GeoRestricted { region_code: String },
    /// Playback requires HDCP (High-bandwidth Digital Content Protection).
    RequiresHdcp,
    /// Content has a rental window expiry (seconds from first play).
    RentalWindowSecs(u64),
}

impl PlaybackRestriction {
    /// Short human-readable description of this restriction.
    pub fn description(&self) -> String {
        match self {
            PlaybackRestriction::NoOffline => "No offline playback".to_string(),
            PlaybackRestriction::NoDownload => "No download permitted".to_string(),
            PlaybackRestriction::NoScreenCapture => "Screen capture prohibited".to_string(),
            PlaybackRestriction::MaxStreams(n) => format!("Max {} concurrent stream(s)", n),
            PlaybackRestriction::GeoRestricted { region_code } => {
                format!("Restricted to region '{}'", region_code)
            }
            PlaybackRestriction::RequiresHdcp => "HDCP required".to_string(),
            PlaybackRestriction::RentalWindowSecs(secs) => {
                format!("Rental window: {} seconds", secs)
            }
        }
    }
}

/// A policy that aggregates zero or more `PlaybackRestriction`s for a piece of content.
#[derive(Debug, Clone, Default)]
pub struct PlaybackPolicy {
    restrictions: Vec<PlaybackRestriction>,
}

impl PlaybackPolicy {
    /// Create a policy with no restrictions (everything allowed).
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a restriction to this policy.
    pub fn add_restriction(&mut self, restriction: PlaybackRestriction) {
        self.restrictions.push(restriction);
    }

    /// Builder-style helper to chain restrictions.
    pub fn with_restriction(mut self, restriction: PlaybackRestriction) -> Self {
        self.restrictions.push(restriction);
        self
    }

    /// Returns `true` if offline playback is permitted (i.e. `NoOffline` is absent).
    pub fn allows_offline(&self) -> bool {
        !self.restrictions.contains(&PlaybackRestriction::NoOffline)
    }

    /// Returns `true` if download is permitted (i.e. `NoDownload` is absent).
    pub fn allows_download(&self) -> bool {
        !self.restrictions.contains(&PlaybackRestriction::NoDownload)
    }

    /// Returns `true` if screen capture is permitted.
    pub fn allows_screen_capture(&self) -> bool {
        !self
            .restrictions
            .contains(&PlaybackRestriction::NoScreenCapture)
    }

    /// Maximum number of concurrent streams permitted, or `None` if unrestricted.
    pub fn max_streams(&self) -> Option<u32> {
        self.restrictions.iter().find_map(|r| {
            if let PlaybackRestriction::MaxStreams(n) = r {
                Some(*n)
            } else {
                None
            }
        })
    }

    /// All restrictions currently registered.
    pub fn restrictions(&self) -> &[PlaybackRestriction] {
        &self.restrictions
    }
}

/// Result of evaluating a playback request against a policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvaluationResult {
    /// Whether playback is permitted.
    pub permitted: bool,
    /// Reasons for any denial (empty when `permitted == true`).
    pub denied_reasons: Vec<String>,
}

impl EvaluationResult {
    fn allow() -> Self {
        Self {
            permitted: true,
            denied_reasons: vec![],
        }
    }

    fn deny(reasons: Vec<String>) -> Self {
        Self {
            permitted: false,
            denied_reasons: reasons,
        }
    }
}

/// Context describing the current playback request being evaluated.
#[derive(Debug, Clone, Default)]
pub struct PlaybackContext {
    /// Whether the user is attempting offline playback.
    pub is_offline: bool,
    /// Whether the user is attempting to download the asset.
    pub is_download: bool,
    /// Whether a screen-capture tool is active.
    pub screen_capture_active: bool,
    /// Number of streams already open for this user.
    pub current_stream_count: u32,
    /// ISO 3166-1 alpha-2 region code of the viewer.
    pub region_code: Option<String>,
    /// Whether HDCP is available on the output path.
    pub hdcp_available: bool,
}

/// Evaluates a `PlaybackContext` against a `PlaybackPolicy`.
#[derive(Debug, Default)]
pub struct PlaybackPolicyEvaluator;

impl PlaybackPolicyEvaluator {
    /// Create a new evaluator.
    pub fn new() -> Self {
        Self
    }

    /// Evaluate `ctx` against `policy` and return an `EvaluationResult`.
    pub fn evaluate(&self, policy: &PlaybackPolicy, ctx: &PlaybackContext) -> EvaluationResult {
        let mut reasons = Vec::new();

        for restriction in policy.restrictions() {
            match restriction {
                PlaybackRestriction::NoOffline if ctx.is_offline => {
                    reasons.push(restriction.description());
                }
                PlaybackRestriction::NoDownload if ctx.is_download => {
                    reasons.push(restriction.description());
                }
                PlaybackRestriction::NoScreenCapture if ctx.screen_capture_active => {
                    reasons.push(restriction.description());
                }
                PlaybackRestriction::MaxStreams(n) if ctx.current_stream_count >= *n => {
                    reasons.push(restriction.description());
                }
                PlaybackRestriction::GeoRestricted { region_code } => {
                    let viewer_region = ctx.region_code.as_deref().unwrap_or("");
                    if viewer_region != region_code {
                        reasons.push(restriction.description());
                    }
                }
                PlaybackRestriction::RequiresHdcp if !ctx.hdcp_available => {
                    reasons.push(restriction.description());
                }
                _ => {}
            }
        }

        if reasons.is_empty() {
            EvaluationResult::allow()
        } else {
            EvaluationResult::deny(reasons)
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn evaluator() -> PlaybackPolicyEvaluator {
        PlaybackPolicyEvaluator::new()
    }

    // PlaybackRestriction description tests

    #[test]
    fn test_description_no_offline() {
        assert!(!PlaybackRestriction::NoOffline.description().is_empty());
    }

    #[test]
    fn test_description_no_download() {
        let d = PlaybackRestriction::NoDownload.description();
        assert!(d.contains("download") || d.contains("Download"));
    }

    #[test]
    fn test_description_max_streams() {
        let d = PlaybackRestriction::MaxStreams(3).description();
        assert!(d.contains('3'));
    }

    #[test]
    fn test_description_geo_restricted() {
        let d = PlaybackRestriction::GeoRestricted {
            region_code: "US".to_string(),
        }
        .description();
        assert!(d.contains("US"));
    }

    // PlaybackPolicy allows_* tests

    #[test]
    fn test_empty_policy_allows_all() {
        let policy = PlaybackPolicy::new();
        assert!(policy.allows_offline());
        assert!(policy.allows_download());
        assert!(policy.allows_screen_capture());
        assert!(policy.max_streams().is_none());
    }

    #[test]
    fn test_policy_no_offline() {
        let policy = PlaybackPolicy::new().with_restriction(PlaybackRestriction::NoOffline);
        assert!(!policy.allows_offline());
        assert!(policy.allows_download());
    }

    #[test]
    fn test_policy_no_download() {
        let policy = PlaybackPolicy::new().with_restriction(PlaybackRestriction::NoDownload);
        assert!(!policy.allows_download());
    }

    #[test]
    fn test_policy_max_streams() {
        let policy = PlaybackPolicy::new().with_restriction(PlaybackRestriction::MaxStreams(2));
        assert_eq!(policy.max_streams(), Some(2));
    }

    // PlaybackPolicyEvaluator tests

    #[test]
    fn test_evaluate_clean_context_no_restrictions() {
        let policy = PlaybackPolicy::new();
        let ctx = PlaybackContext::default();
        let result = evaluator().evaluate(&policy, &ctx);
        assert!(result.permitted);
        assert!(result.denied_reasons.is_empty());
    }

    #[test]
    fn test_evaluate_offline_denied() {
        let policy = PlaybackPolicy::new().with_restriction(PlaybackRestriction::NoOffline);
        let ctx = PlaybackContext {
            is_offline: true,
            ..Default::default()
        };
        let result = evaluator().evaluate(&policy, &ctx);
        assert!(!result.permitted);
        assert_eq!(result.denied_reasons.len(), 1);
    }

    #[test]
    fn test_evaluate_download_denied() {
        let policy = PlaybackPolicy::new().with_restriction(PlaybackRestriction::NoDownload);
        let ctx = PlaybackContext {
            is_download: true,
            ..Default::default()
        };
        let result = evaluator().evaluate(&policy, &ctx);
        assert!(!result.permitted);
    }

    #[test]
    fn test_evaluate_stream_limit_reached() {
        let policy = PlaybackPolicy::new().with_restriction(PlaybackRestriction::MaxStreams(2));
        let ctx = PlaybackContext {
            current_stream_count: 2,
            ..Default::default()
        };
        let result = evaluator().evaluate(&policy, &ctx);
        assert!(!result.permitted);
    }

    #[test]
    fn test_evaluate_stream_limit_not_reached() {
        let policy = PlaybackPolicy::new().with_restriction(PlaybackRestriction::MaxStreams(3));
        let ctx = PlaybackContext {
            current_stream_count: 2,
            ..Default::default()
        };
        let result = evaluator().evaluate(&policy, &ctx);
        assert!(result.permitted);
    }

    #[test]
    fn test_evaluate_geo_restriction_wrong_region() {
        let policy = PlaybackPolicy::new().with_restriction(PlaybackRestriction::GeoRestricted {
            region_code: "GB".to_string(),
        });
        let ctx = PlaybackContext {
            region_code: Some("US".to_string()),
            ..Default::default()
        };
        let result = evaluator().evaluate(&policy, &ctx);
        assert!(!result.permitted);
    }

    #[test]
    fn test_evaluate_geo_restriction_correct_region() {
        let policy = PlaybackPolicy::new().with_restriction(PlaybackRestriction::GeoRestricted {
            region_code: "US".to_string(),
        });
        let ctx = PlaybackContext {
            region_code: Some("US".to_string()),
            ..Default::default()
        };
        let result = evaluator().evaluate(&policy, &ctx);
        assert!(result.permitted);
    }

    #[test]
    fn test_evaluate_hdcp_required_unavailable() {
        let policy = PlaybackPolicy::new().with_restriction(PlaybackRestriction::RequiresHdcp);
        let ctx = PlaybackContext {
            hdcp_available: false,
            ..Default::default()
        };
        let result = evaluator().evaluate(&policy, &ctx);
        assert!(!result.permitted);
    }

    #[test]
    fn test_evaluate_multiple_violations() {
        let policy = PlaybackPolicy::new()
            .with_restriction(PlaybackRestriction::NoOffline)
            .with_restriction(PlaybackRestriction::NoDownload);
        let ctx = PlaybackContext {
            is_offline: true,
            is_download: true,
            ..Default::default()
        };
        let result = evaluator().evaluate(&policy, &ctx);
        assert!(!result.permitted);
        assert_eq!(result.denied_reasons.len(), 2);
    }
}
