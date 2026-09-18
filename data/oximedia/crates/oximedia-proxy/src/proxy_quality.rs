//! Proxy quality tier definitions and bandwidth-adaptive selection.
//!
//! Provides `ProxyQualityTier`, `ProxyQualityConfig`, and `ProxyQualitySelector`
//! for selecting the appropriate proxy tier for a given network or storage context.

#![allow(dead_code)]

/// Proxy quality tier for different stages of the offline/online workflow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ProxyQualityTier {
    /// Rough-cut editing — smallest files, lowest resolution.
    Draft,
    /// Client review — medium files, visually acceptable quality.
    Review,
    /// Delivery / conform — highest quality proxy allowed.
    Delivery,
}

impl ProxyQualityTier {
    /// Maximum resolution cap for this tier (width × height).
    pub fn resolution_cap(self) -> (u32, u32) {
        match self {
            Self::Draft => (640, 360),
            Self::Review => (1280, 720),
            Self::Delivery => (1920, 1080),
        }
    }

    /// Return a short human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Draft => "Draft",
            Self::Review => "Review",
            Self::Delivery => "Delivery",
        }
    }

    /// Return the next higher tier, or `None` if already at `Delivery`.
    pub fn upgrade(self) -> Option<Self> {
        match self {
            Self::Draft => Some(Self::Review),
            Self::Review => Some(Self::Delivery),
            Self::Delivery => None,
        }
    }

    /// Return the next lower tier, or `None` if already at `Draft`.
    pub fn downgrade(self) -> Option<Self> {
        match self {
            Self::Draft => None,
            Self::Review => Some(Self::Draft),
            Self::Delivery => Some(Self::Review),
        }
    }
}

/// Configuration for a specific proxy quality tier.
#[derive(Debug, Clone)]
pub struct ProxyQualityConfig {
    /// The tier this config describes.
    pub tier: ProxyQualityTier,
    /// Target video bitrate in kbps.
    pub video_bitrate_kbps: u32,
    /// Target audio bitrate in kbps.
    pub audio_bitrate_kbps: u32,
    /// Container/codec hint (e.g. "h264", "vp9").
    pub codec_hint: String,
    /// Frames per second (0 = match source).
    pub fps_cap: f32,
}

impl ProxyQualityConfig {
    /// Create a new config for the given tier with default parameters.
    pub fn new(tier: ProxyQualityTier) -> Self {
        let (video_bitrate_kbps, audio_bitrate_kbps, codec_hint) = match tier {
            ProxyQualityTier::Draft => (500, 64, "h264"),
            ProxyQualityTier::Review => (2000, 128, "h264"),
            ProxyQualityTier::Delivery => (8000, 320, "h264"),
        };
        Self {
            tier,
            video_bitrate_kbps,
            audio_bitrate_kbps,
            codec_hint: codec_hint.to_string(),
            fps_cap: 0.0,
        }
    }

    /// Return total bitrate in kbps (video + audio).
    pub fn bitrate_kbps(&self) -> u32 {
        self.video_bitrate_kbps + self.audio_bitrate_kbps
    }

    /// Return `true` if total bitrate fits within `budget_kbps`.
    pub fn fits_budget(&self, budget_kbps: u32) -> bool {
        self.bitrate_kbps() <= budget_kbps
    }

    /// Effective resolution cap for this config.
    pub fn resolution_cap(&self) -> (u32, u32) {
        self.tier.resolution_cap()
    }
}

/// Selects the best `ProxyQualityTier` for a given available bandwidth.
pub struct ProxyQualitySelector {
    configs: Vec<ProxyQualityConfig>,
}

impl ProxyQualitySelector {
    /// Create a selector populated with default configs for all three tiers.
    pub fn new() -> Self {
        Self {
            configs: vec![
                ProxyQualityConfig::new(ProxyQualityTier::Draft),
                ProxyQualityConfig::new(ProxyQualityTier::Review),
                ProxyQualityConfig::new(ProxyQualityTier::Delivery),
            ],
        }
    }

    /// Create a selector with custom configs.
    pub fn with_configs(configs: Vec<ProxyQualityConfig>) -> Self {
        Self { configs }
    }

    /// Select the highest-quality tier that fits `available_bandwidth_kbps`.
    /// Returns `None` if even the lowest tier exceeds the budget.
    pub fn select_for_bandwidth(
        &self,
        available_bandwidth_kbps: u32,
    ) -> Option<&ProxyQualityConfig> {
        // Sort descending by quality (tier), pick first that fits
        let mut sorted: Vec<&ProxyQualityConfig> = self.configs.iter().collect();
        sorted.sort_by(|a, b| b.tier.cmp(&a.tier));
        sorted
            .into_iter()
            .find(|c| c.fits_budget(available_bandwidth_kbps))
    }

    /// Return all configs sorted from lowest to highest quality.
    pub fn all_tiers(&self) -> Vec<&ProxyQualityConfig> {
        let mut v: Vec<&ProxyQualityConfig> = self.configs.iter().collect();
        v.sort_by_key(|c| c.tier);
        v
    }
}

impl Default for ProxyQualitySelector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_draft_resolution_cap() {
        assert_eq!(ProxyQualityTier::Draft.resolution_cap(), (640, 360));
    }

    #[test]
    fn test_review_resolution_cap() {
        assert_eq!(ProxyQualityTier::Review.resolution_cap(), (1280, 720));
    }

    #[test]
    fn test_delivery_resolution_cap() {
        assert_eq!(ProxyQualityTier::Delivery.resolution_cap(), (1920, 1080));
    }

    #[test]
    fn test_tier_upgrade() {
        assert_eq!(
            ProxyQualityTier::Draft.upgrade(),
            Some(ProxyQualityTier::Review)
        );
        assert_eq!(ProxyQualityTier::Delivery.upgrade(), None);
    }

    #[test]
    fn test_tier_downgrade() {
        assert_eq!(
            ProxyQualityTier::Delivery.downgrade(),
            Some(ProxyQualityTier::Review)
        );
        assert_eq!(ProxyQualityTier::Draft.downgrade(), None);
    }

    #[test]
    fn test_tier_ordering() {
        assert!(ProxyQualityTier::Draft < ProxyQualityTier::Review);
        assert!(ProxyQualityTier::Review < ProxyQualityTier::Delivery);
    }

    #[test]
    fn test_config_bitrate_kbps() {
        let cfg = ProxyQualityConfig::new(ProxyQualityTier::Draft);
        assert_eq!(cfg.bitrate_kbps(), 564); // 500 + 64
    }

    #[test]
    fn test_config_fits_budget_true() {
        let cfg = ProxyQualityConfig::new(ProxyQualityTier::Draft);
        assert!(cfg.fits_budget(1000));
    }

    #[test]
    fn test_config_fits_budget_false() {
        let cfg = ProxyQualityConfig::new(ProxyQualityTier::Delivery);
        assert!(!cfg.fits_budget(100));
    }

    #[test]
    fn test_selector_selects_draft_for_low_bandwidth() {
        let sel = ProxyQualitySelector::new();
        let result = sel.select_for_bandwidth(600);
        assert!(result.is_some());
        assert_eq!(
            result.expect("should succeed in test").tier,
            ProxyQualityTier::Draft
        );
    }

    #[test]
    fn test_selector_selects_delivery_for_high_bandwidth() {
        let sel = ProxyQualitySelector::new();
        let result = sel.select_for_bandwidth(50_000);
        assert!(result.is_some());
        assert_eq!(
            result.expect("should succeed in test").tier,
            ProxyQualityTier::Delivery
        );
    }

    #[test]
    fn test_selector_returns_none_if_too_low() {
        let sel = ProxyQualitySelector::new();
        let result = sel.select_for_bandwidth(10); // below all tiers
        assert!(result.is_none());
    }

    #[test]
    fn test_selector_all_tiers_sorted() {
        let sel = ProxyQualitySelector::new();
        let tiers: Vec<ProxyQualityTier> = sel.all_tiers().iter().map(|c| c.tier).collect();
        assert_eq!(
            tiers,
            vec![
                ProxyQualityTier::Draft,
                ProxyQualityTier::Review,
                ProxyQualityTier::Delivery
            ]
        );
    }

    #[test]
    fn test_tier_label() {
        assert_eq!(ProxyQualityTier::Draft.label(), "Draft");
        assert_eq!(ProxyQualityTier::Review.label(), "Review");
        assert_eq!(ProxyQualityTier::Delivery.label(), "Delivery");
    }
}
