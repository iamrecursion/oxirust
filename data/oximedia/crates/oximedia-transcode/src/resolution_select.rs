//! Resolution tier selection for adaptive transcoding.
//!
//! Maps bandwidth budgets and quality targets to standard resolution tiers,
//! supporting adaptive bitrate ladder construction and device capability matching.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

/// Standard resolution tiers used in adaptive streaming and archiving.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ResolutionTier {
    /// 480p SD (854×480).
    Sd,
    /// 720p HD (1280×720).
    Hd,
    /// 1080p Full HD (1920×1080).
    FullHd,
    /// 2160p Ultra HD 4K (3840×2160).
    Uhd4k,
}

impl ResolutionTier {
    /// Total pixel count for this tier.
    #[must_use]
    pub fn pixel_count(self) -> u64 {
        let (w, h) = self.dimensions();
        u64::from(w) * u64::from(h)
    }

    /// Canonical width and height (pixels) for this tier.
    #[must_use]
    pub fn dimensions(self) -> (u32, u32) {
        match self {
            Self::Sd => (854, 480),
            Self::Hd => (1280, 720),
            Self::FullHd => (1920, 1080),
            Self::Uhd4k => (3840, 2160),
        }
    }

    /// Commonly recommended video bitrate (bits per second) for H.264/AVC.
    #[must_use]
    pub fn recommended_bitrate_bps(self) -> u64 {
        match self {
            Self::Sd => 1_500_000,
            Self::Hd => 4_000_000,
            Self::FullHd => 8_000_000,
            Self::Uhd4k => 40_000_000,
        }
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Sd => "480p",
            Self::Hd => "720p",
            Self::FullHd => "1080p",
            Self::Uhd4k => "2160p",
        }
    }

    /// Returns an ordered list of all tiers from lowest to highest.
    #[must_use]
    pub fn all_tiers() -> [Self; 4] {
        [Self::Sd, Self::Hd, Self::FullHd, Self::Uhd4k]
    }
}

/// Strategy to use when selecting among resolution tiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SelectionStrategy {
    /// Choose the highest tier that fits within the bandwidth budget.
    BandwidthFit,
    /// Choose the highest tier regardless of bandwidth.
    MaxQuality,
    /// Choose the lowest tier (minimise bandwidth).
    MinBandwidth,
    /// Choose an exact tier.
    Exact(ResolutionTier),
}

/// Selects a resolution tier based on bandwidth or quality constraints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionSelector {
    /// Maximum available bandwidth in bits per second.
    pub max_bandwidth_bps: u64,
    /// Minimum acceptable tier.
    pub min_tier: ResolutionTier,
    /// Maximum acceptable tier.
    pub max_tier: ResolutionTier,
    /// Headroom factor applied to the recommended bitrate when fitting.
    /// A value of 1.2 means 20 % safety margin is required.
    pub headroom_factor: f64,
}

impl Default for ResolutionSelector {
    fn default() -> Self {
        Self {
            max_bandwidth_bps: 10_000_000,
            min_tier: ResolutionTier::Sd,
            max_tier: ResolutionTier::Uhd4k,
            headroom_factor: 1.2,
        }
    }
}

impl ResolutionSelector {
    /// Create a selector with a specific bandwidth ceiling.
    #[must_use]
    pub fn new(max_bandwidth_bps: u64) -> Self {
        Self {
            max_bandwidth_bps,
            ..Self::default()
        }
    }

    /// Set the minimum tier floor.
    #[must_use]
    pub fn with_min_tier(mut self, tier: ResolutionTier) -> Self {
        self.min_tier = tier;
        self
    }

    /// Set the maximum tier ceiling.
    #[must_use]
    pub fn with_max_tier(mut self, tier: ResolutionTier) -> Self {
        self.max_tier = tier;
        self
    }

    /// Set the headroom factor (must be >= 1.0).
    #[must_use]
    pub fn with_headroom(mut self, factor: f64) -> Self {
        self.headroom_factor = factor.max(1.0);
        self
    }

    /// Select the highest tier that fits within the configured bandwidth budget.
    ///
    /// Returns `None` if no tier (not even the minimum) fits.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn select_for_bandwidth(&self) -> Option<ResolutionTier> {
        let budget = self.max_bandwidth_bps as f64 / self.headroom_factor;
        let mut best: Option<ResolutionTier> = None;
        for tier in ResolutionTier::all_tiers() {
            if tier < self.min_tier || tier > self.max_tier {
                continue;
            }
            if tier.recommended_bitrate_bps() as f64 <= budget {
                best = Some(tier);
            }
        }
        best
    }

    /// Select the maximum tier within the configured min/max bounds (ignores bandwidth).
    #[must_use]
    pub fn select_for_quality(&self) -> ResolutionTier {
        ResolutionTier::all_tiers()
            .iter()
            .filter(|&&t| t >= self.min_tier && t <= self.max_tier)
            .copied()
            .last()
            .unwrap_or(self.min_tier)
    }

    /// Return all tiers that fall within the configured min/max bounds.
    #[must_use]
    pub fn all_tiers(&self) -> Vec<ResolutionTier> {
        ResolutionTier::all_tiers()
            .iter()
            .filter(|&&t| t >= self.min_tier && t <= self.max_tier)
            .copied()
            .collect()
    }

    /// Build an ABR ladder: return all tiers that fit within the bandwidth budget.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn abr_ladder(&self) -> Vec<ResolutionTier> {
        let budget = self.max_bandwidth_bps as f64 / self.headroom_factor;
        ResolutionTier::all_tiers()
            .iter()
            .filter(|&&t| {
                t >= self.min_tier
                    && t <= self.max_tier
                    && t.recommended_bitrate_bps() as f64 <= budget
            })
            .copied()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pixel_count_ordering() {
        assert!(ResolutionTier::Sd.pixel_count() < ResolutionTier::Hd.pixel_count());
        assert!(ResolutionTier::Hd.pixel_count() < ResolutionTier::FullHd.pixel_count());
        assert!(ResolutionTier::FullHd.pixel_count() < ResolutionTier::Uhd4k.pixel_count());
    }

    #[test]
    fn test_sd_dimensions() {
        assert_eq!(ResolutionTier::Sd.dimensions(), (854, 480));
    }

    #[test]
    fn test_uhd4k_dimensions() {
        assert_eq!(ResolutionTier::Uhd4k.dimensions(), (3840, 2160));
    }

    #[test]
    fn test_recommended_bitrate_increases_with_tier() {
        let tiers = ResolutionTier::all_tiers();
        for window in tiers.windows(2) {
            assert!(window[0].recommended_bitrate_bps() < window[1].recommended_bitrate_bps());
        }
    }

    #[test]
    fn test_labels_non_empty() {
        for t in ResolutionTier::all_tiers() {
            assert!(!t.label().is_empty());
        }
    }

    #[test]
    fn test_all_tiers_count() {
        assert_eq!(ResolutionTier::all_tiers().len(), 4);
    }

    #[test]
    fn test_select_for_bandwidth_high_budget() {
        let sel = ResolutionSelector::new(100_000_000);
        assert_eq!(sel.select_for_bandwidth(), Some(ResolutionTier::Uhd4k));
    }

    #[test]
    fn test_select_for_bandwidth_low_budget() {
        // 1 Mbps is below even SD recommended bitrate with 1.2x headroom
        let sel = ResolutionSelector::new(1_000_000).with_headroom(1.0);
        assert!(sel.select_for_bandwidth().is_none());
    }

    #[test]
    fn test_select_for_bandwidth_mid_budget() {
        // 5 Mbps should fit HD but not FullHD (8 Mbps)
        let sel = ResolutionSelector::new(5_000_000).with_headroom(1.0);
        assert_eq!(sel.select_for_bandwidth(), Some(ResolutionTier::Hd));
    }

    #[test]
    fn test_select_for_quality_returns_max_tier() {
        let sel = ResolutionSelector::default().with_max_tier(ResolutionTier::FullHd);
        assert_eq!(sel.select_for_quality(), ResolutionTier::FullHd);
    }

    #[test]
    fn test_all_tiers_bounded() {
        let sel = ResolutionSelector::default()
            .with_min_tier(ResolutionTier::Hd)
            .with_max_tier(ResolutionTier::FullHd);
        let tiers = sel.all_tiers();
        assert_eq!(tiers, vec![ResolutionTier::Hd, ResolutionTier::FullHd]);
    }

    #[test]
    fn test_abr_ladder_large_budget() {
        let sel = ResolutionSelector::new(100_000_000).with_headroom(1.0);
        assert_eq!(sel.abr_ladder().len(), 4);
    }

    #[test]
    fn test_abr_ladder_small_budget() {
        // Budget just covers HD (4 Mbps) but not FullHD (8 Mbps)
        let sel = ResolutionSelector::new(4_000_000).with_headroom(1.0);
        let ladder = sel.abr_ladder();
        assert!(ladder.contains(&ResolutionTier::Hd));
        assert!(!ladder.contains(&ResolutionTier::FullHd));
    }

    #[test]
    fn test_tier_ordering() {
        assert!(ResolutionTier::Sd < ResolutionTier::Hd);
        assert!(ResolutionTier::Hd < ResolutionTier::FullHd);
        assert!(ResolutionTier::FullHd < ResolutionTier::Uhd4k);
    }
}
