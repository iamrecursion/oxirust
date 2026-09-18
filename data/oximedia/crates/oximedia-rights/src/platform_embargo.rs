//! Platform-specific embargo windows with sequential release strategy.
//!
//! Models the standard media release waterfall:
//! theatrical -> home video / PVOD -> streaming (SVOD) -> broadcast (free-to-air)
//!
//! Each platform has its own embargo window, and the engine enforces that a
//! content item can only be released on a platform when that platform's window
//! is open and all prerequisite (earlier) windows have been honoured.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

// ── DistributionPlatform ────────────────────────────────────────────────────

/// A distribution platform in the release waterfall.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DistributionPlatform {
    /// Theatrical / cinema exhibition.
    Theatrical,
    /// Premium video-on-demand (PVOD) / early digital rental.
    Pvod,
    /// Physical media and electronic sell-through (EST).
    HomeVideo,
    /// Subscription video-on-demand (Netflix, Disney+, etc.).
    Svod,
    /// Ad-supported video-on-demand (Tubi, Pluto, etc.).
    Avod,
    /// Traditional broadcast (linear TV / free-to-air).
    Broadcast,
    /// Custom platform identified by name.
    Custom(String),
}

impl DistributionPlatform {
    /// Short identifier for the platform.
    pub fn code(&self) -> &str {
        match self {
            Self::Theatrical => "THEATRICAL",
            Self::Pvod => "PVOD",
            Self::HomeVideo => "HOME_VIDEO",
            Self::Svod => "SVOD",
            Self::Avod => "AVOD",
            Self::Broadcast => "BROADCAST",
            Self::Custom(c) => c.as_str(),
        }
    }

    /// Default position in the standard release waterfall (lower = earlier).
    ///
    /// Custom platforms return a high number so they appear at the end by
    /// default.
    pub fn default_order(&self) -> u32 {
        match self {
            Self::Theatrical => 0,
            Self::Pvod => 1,
            Self::HomeVideo => 2,
            Self::Svod => 3,
            Self::Avod => 4,
            Self::Broadcast => 5,
            Self::Custom(_) => 100,
        }
    }

    /// Typical industry-standard embargo duration in days after theatrical
    /// release.
    pub fn typical_offset_days(&self) -> u32 {
        match self {
            Self::Theatrical => 0,
            Self::Pvod => 17,       // ~2.5 weeks
            Self::HomeVideo => 45,  // ~6 weeks
            Self::Svod => 90,       // ~3 months
            Self::Avod => 120,      // ~4 months
            Self::Broadcast => 180, // ~6 months
            Self::Custom(_) => 0,
        }
    }
}

// ── PlatformWindow ──────────────────────────────────────────────────────────

/// An embargo window for a specific platform and content item.
#[derive(Debug, Clone)]
pub struct PlatformWindow {
    /// The content item this window applies to.
    pub content_id: String,
    /// The platform.
    pub platform: DistributionPlatform,
    /// Unix timestamp (seconds) when the window opens.
    pub opens_at: u64,
    /// Unix timestamp (seconds) when the window closes.
    /// `None` means it remains open indefinitely.
    pub closes_at: Option<u64>,
    /// Optional territory restriction (ISO 3166-1).
    /// Empty means worldwide.
    pub territories: Vec<String>,
    /// Whether this window is exclusive (no other platform may be open
    /// simultaneously).
    pub exclusive: bool,
}

impl PlatformWindow {
    /// Create a new platform window.
    pub fn new(
        content_id: &str,
        platform: DistributionPlatform,
        opens_at: u64,
        closes_at: Option<u64>,
    ) -> Self {
        Self {
            content_id: content_id.to_string(),
            platform,
            opens_at,
            closes_at,
            territories: Vec::new(),
            exclusive: false,
        }
    }

    /// Builder: set territories.
    pub fn with_territories(mut self, territories: Vec<String>) -> Self {
        self.territories = territories;
        self
    }

    /// Builder: mark as exclusive.
    pub fn with_exclusive(mut self, exclusive: bool) -> Self {
        self.exclusive = exclusive;
        self
    }

    /// Returns `true` if the window is open at `now`.
    pub fn is_open_at(&self, now: u64) -> bool {
        if now < self.opens_at {
            return false;
        }
        match self.closes_at {
            Some(close) => now <= close,
            None => true,
        }
    }

    /// Returns `true` if the window has not yet opened at `now`.
    pub fn is_pending(&self, now: u64) -> bool {
        now < self.opens_at
    }

    /// Returns `true` if the window has closed at `now`.
    pub fn is_closed(&self, now: u64) -> bool {
        match self.closes_at {
            Some(close) => now > close,
            None => false,
        }
    }

    /// Duration of the window in seconds, if bounded.
    pub fn duration_seconds(&self) -> Option<u64> {
        self.closes_at.map(|c| c.saturating_sub(self.opens_at))
    }

    /// Returns `true` if this window covers the given territory.
    /// Empty territories = worldwide.
    pub fn covers_territory(&self, territory: &str) -> bool {
        self.territories.is_empty() || self.territories.iter().any(|t| t == territory)
    }
}

// ── ReleaseWaterfall ────────────────────────────────────────────────────────

/// Generates a standard release waterfall from a theatrical release date.
pub struct ReleaseWaterfall;

impl ReleaseWaterfall {
    /// Create a standard release waterfall starting from a theatrical release
    /// epoch.  Each platform's window opens at the standard offset and stays
    /// open until the next window opens (except the last, which is perpetual).
    pub fn standard(content_id: &str, theatrical_epoch: u64) -> Vec<PlatformWindow> {
        let platforms = [
            DistributionPlatform::Theatrical,
            DistributionPlatform::Pvod,
            DistributionPlatform::HomeVideo,
            DistributionPlatform::Svod,
            DistributionPlatform::Avod,
            DistributionPlatform::Broadcast,
        ];

        let day_secs: u64 = 86_400;
        let mut windows = Vec::with_capacity(platforms.len());

        for (i, platform) in platforms.iter().enumerate() {
            let opens_at = theatrical_epoch + u64::from(platform.typical_offset_days()) * day_secs;
            let closes_at = if i + 1 < platforms.len() {
                let next_opens =
                    theatrical_epoch + u64::from(platforms[i + 1].typical_offset_days()) * day_secs;
                Some(next_opens.saturating_sub(1)) // close 1 second before next opens
            } else {
                None // last window is perpetual
            };

            let mut w = PlatformWindow::new(content_id, platform.clone(), opens_at, closes_at);
            // Theatrical is typically exclusive
            if matches!(platform, DistributionPlatform::Theatrical) {
                w.exclusive = true;
            }
            windows.push(w);
        }

        windows
    }

    /// Create a day-and-date release where all platforms open simultaneously.
    pub fn day_and_date(content_id: &str, release_epoch: u64) -> Vec<PlatformWindow> {
        let platforms = [
            DistributionPlatform::Theatrical,
            DistributionPlatform::Pvod,
            DistributionPlatform::HomeVideo,
            DistributionPlatform::Svod,
            DistributionPlatform::Avod,
            DistributionPlatform::Broadcast,
        ];

        platforms
            .into_iter()
            .map(|p| PlatformWindow::new(content_id, p, release_epoch, None))
            .collect()
    }
}

// ── PlatformEmbargoManager ──────────────────────────────────────────────────

/// Manager that tracks platform windows for multiple content items and
/// enforces embargo rules.
#[derive(Debug, Default)]
pub struct PlatformEmbargoManager {
    /// Windows keyed by content ID.
    windows: HashMap<String, Vec<PlatformWindow>>,
}

impl PlatformEmbargoManager {
    /// Create a new empty manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a platform window.
    pub fn add_window(&mut self, window: PlatformWindow) {
        self.windows
            .entry(window.content_id.clone())
            .or_default()
            .push(window);
    }

    /// Register an entire waterfall at once.
    pub fn add_waterfall(&mut self, windows: Vec<PlatformWindow>) {
        for w in windows {
            self.add_window(w);
        }
    }

    /// Check if a content item may be distributed on a given platform at
    /// `now` in the specified territory.
    pub fn can_distribute(
        &self,
        content_id: &str,
        platform: &DistributionPlatform,
        territory: &str,
        now: u64,
    ) -> bool {
        let windows = match self.windows.get(content_id) {
            Some(w) => w,
            None => return false,
        };

        // Check if the requested platform has an open window
        let platform_open = windows
            .iter()
            .any(|w| w.platform == *platform && w.is_open_at(now) && w.covers_territory(territory));

        if !platform_open {
            return false;
        }

        // Check that no *exclusive* window for a different platform is active
        let exclusive_conflict = windows.iter().any(|w| {
            w.platform != *platform
                && w.exclusive
                && w.is_open_at(now)
                && w.covers_territory(territory)
        });

        !exclusive_conflict
    }

    /// All platforms that are currently available for a content item.
    pub fn available_platforms(
        &self,
        content_id: &str,
        territory: &str,
        now: u64,
    ) -> Vec<&DistributionPlatform> {
        let windows = match self.windows.get(content_id) {
            Some(w) => w,
            None => return vec![],
        };

        // Check for exclusive windows
        let has_exclusive = windows
            .iter()
            .any(|w| w.exclusive && w.is_open_at(now) && w.covers_territory(territory));

        if has_exclusive {
            // Only the exclusive platform(s)
            return windows
                .iter()
                .filter(|w| w.exclusive && w.is_open_at(now) && w.covers_territory(territory))
                .map(|w| &w.platform)
                .collect();
        }

        windows
            .iter()
            .filter(|w| w.is_open_at(now) && w.covers_territory(territory))
            .map(|w| &w.platform)
            .collect()
    }

    /// Upcoming windows that have not yet opened for a content item.
    pub fn upcoming_windows(&self, content_id: &str, now: u64) -> Vec<&PlatformWindow> {
        match self.windows.get(content_id) {
            Some(ws) => ws.iter().filter(|w| w.is_pending(now)).collect(),
            None => vec![],
        }
    }

    /// All windows for a content item.
    pub fn windows_for(&self, content_id: &str) -> Vec<&PlatformWindow> {
        match self.windows.get(content_id) {
            Some(ws) => ws.iter().collect(),
            None => vec![],
        }
    }

    /// Number of content items tracked.
    pub fn content_count(&self) -> usize {
        self.windows.len()
    }

    /// Total number of windows across all content.
    pub fn window_count(&self) -> usize {
        self.windows.values().map(|v| v.len()).sum()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const DAY: u64 = 86_400;
    const THEATRICAL_DATE: u64 = 1_700_000_000;

    // ── DistributionPlatform ────────────────────────────────────────────────

    #[test]
    fn test_platform_code() {
        assert_eq!(DistributionPlatform::Theatrical.code(), "THEATRICAL");
        assert_eq!(DistributionPlatform::Svod.code(), "SVOD");
        assert_eq!(
            DistributionPlatform::Custom("IMAX".to_string()).code(),
            "IMAX"
        );
    }

    #[test]
    fn test_platform_default_order() {
        assert!(
            DistributionPlatform::Theatrical.default_order()
                < DistributionPlatform::Pvod.default_order()
        );
        assert!(
            DistributionPlatform::Svod.default_order()
                < DistributionPlatform::Broadcast.default_order()
        );
    }

    #[test]
    fn test_platform_typical_offset_theatrical_zero() {
        assert_eq!(DistributionPlatform::Theatrical.typical_offset_days(), 0);
    }

    #[test]
    fn test_platform_typical_offset_svod() {
        assert_eq!(DistributionPlatform::Svod.typical_offset_days(), 90);
    }

    // ── PlatformWindow ──────────────────────────────────────────────────────

    #[test]
    fn test_window_is_open() {
        let w = PlatformWindow::new("m1", DistributionPlatform::Theatrical, 1000, Some(2000));
        assert!(!w.is_open_at(999));
        assert!(w.is_open_at(1000));
        assert!(w.is_open_at(1500));
        assert!(w.is_open_at(2000));
        assert!(!w.is_open_at(2001));
    }

    #[test]
    fn test_window_is_pending() {
        let w = PlatformWindow::new("m1", DistributionPlatform::Svod, 5000, None);
        assert!(w.is_pending(4999));
        assert!(!w.is_pending(5000));
    }

    #[test]
    fn test_window_is_closed() {
        let w = PlatformWindow::new("m1", DistributionPlatform::Pvod, 1000, Some(2000));
        assert!(!w.is_closed(2000));
        assert!(w.is_closed(2001));
    }

    #[test]
    fn test_window_perpetual_never_closed() {
        let w = PlatformWindow::new("m1", DistributionPlatform::Broadcast, 1000, None);
        assert!(!w.is_closed(999_999_999));
    }

    #[test]
    fn test_window_duration() {
        let w = PlatformWindow::new("m1", DistributionPlatform::Pvod, 1000, Some(5000));
        assert_eq!(w.duration_seconds(), Some(4000));
    }

    #[test]
    fn test_window_duration_perpetual() {
        let w = PlatformWindow::new("m1", DistributionPlatform::Broadcast, 1000, None);
        assert!(w.duration_seconds().is_none());
    }

    #[test]
    fn test_window_covers_territory_worldwide() {
        let w = PlatformWindow::new("m1", DistributionPlatform::Svod, 0, None);
        assert!(w.covers_territory("US"));
        assert!(w.covers_territory("JP"));
    }

    #[test]
    fn test_window_covers_territory_specific() {
        let w = PlatformWindow::new("m1", DistributionPlatform::Svod, 0, None)
            .with_territories(vec!["US".to_string(), "CA".to_string()]);
        assert!(w.covers_territory("US"));
        assert!(!w.covers_territory("JP"));
    }

    #[test]
    fn test_window_exclusive_flag() {
        let w = PlatformWindow::new("m1", DistributionPlatform::Theatrical, 0, Some(1000))
            .with_exclusive(true);
        assert!(w.exclusive);
    }

    // ── ReleaseWaterfall ────────────────────────────────────────────────────

    #[test]
    fn test_standard_waterfall_count() {
        let windows = ReleaseWaterfall::standard("m1", THEATRICAL_DATE);
        assert_eq!(windows.len(), 6);
    }

    #[test]
    fn test_standard_waterfall_order() {
        let windows = ReleaseWaterfall::standard("m1", THEATRICAL_DATE);
        // Each opens_at should be >= previous
        for w in windows.windows(2) {
            assert!(w[1].opens_at >= w[0].opens_at);
        }
    }

    #[test]
    fn test_standard_waterfall_theatrical_opens_first() {
        let windows = ReleaseWaterfall::standard("m1", THEATRICAL_DATE);
        assert_eq!(windows[0].platform, DistributionPlatform::Theatrical);
        assert_eq!(windows[0].opens_at, THEATRICAL_DATE);
    }

    #[test]
    fn test_standard_waterfall_theatrical_is_exclusive() {
        let windows = ReleaseWaterfall::standard("m1", THEATRICAL_DATE);
        assert!(windows[0].exclusive);
    }

    #[test]
    fn test_standard_waterfall_last_is_perpetual() {
        let windows = ReleaseWaterfall::standard("m1", THEATRICAL_DATE);
        assert!(windows
            .last()
            .map(|w| w.closes_at.is_none())
            .unwrap_or(false));
    }

    #[test]
    fn test_standard_waterfall_svod_offset() {
        let windows = ReleaseWaterfall::standard("m1", THEATRICAL_DATE);
        let svod = windows
            .iter()
            .find(|w| w.platform == DistributionPlatform::Svod);
        assert!(svod.is_some());
        assert_eq!(
            svod.expect("svod should exist").opens_at,
            THEATRICAL_DATE + 90 * DAY
        );
    }

    #[test]
    fn test_day_and_date_all_same_open() {
        let windows = ReleaseWaterfall::day_and_date("m1", THEATRICAL_DATE);
        assert_eq!(windows.len(), 6);
        for w in &windows {
            assert_eq!(w.opens_at, THEATRICAL_DATE);
            assert!(w.closes_at.is_none());
        }
    }

    // ── PlatformEmbargoManager ──────────────────────────────────────────────

    #[test]
    fn test_manager_add_and_count() {
        let mut mgr = PlatformEmbargoManager::new();
        let waterfall = ReleaseWaterfall::standard("m1", THEATRICAL_DATE);
        mgr.add_waterfall(waterfall);
        assert_eq!(mgr.content_count(), 1);
        assert_eq!(mgr.window_count(), 6);
    }

    #[test]
    fn test_manager_can_distribute_theatrical_during_exclusive() {
        let mut mgr = PlatformEmbargoManager::new();
        mgr.add_waterfall(ReleaseWaterfall::standard("m1", THEATRICAL_DATE));
        // During theatrical exclusive window
        let during_theatrical = THEATRICAL_DATE + 5 * DAY;
        assert!(mgr.can_distribute(
            "m1",
            &DistributionPlatform::Theatrical,
            "US",
            during_theatrical,
        ));
    }

    #[test]
    fn test_manager_cannot_distribute_svod_during_theatrical_exclusive() {
        let mut mgr = PlatformEmbargoManager::new();
        mgr.add_waterfall(ReleaseWaterfall::standard("m1", THEATRICAL_DATE));
        let during_theatrical = THEATRICAL_DATE + 5 * DAY;
        assert!(!mgr.can_distribute("m1", &DistributionPlatform::Svod, "US", during_theatrical,));
    }

    #[test]
    fn test_manager_can_distribute_svod_after_window_opens() {
        let mut mgr = PlatformEmbargoManager::new();
        mgr.add_waterfall(ReleaseWaterfall::standard("m1", THEATRICAL_DATE));
        let after_svod_opens = THEATRICAL_DATE + 91 * DAY;
        assert!(mgr.can_distribute("m1", &DistributionPlatform::Svod, "US", after_svod_opens,));
    }

    #[test]
    fn test_manager_unknown_content_returns_false() {
        let mgr = PlatformEmbargoManager::new();
        assert!(!mgr.can_distribute(
            "unknown",
            &DistributionPlatform::Svod,
            "US",
            THEATRICAL_DATE,
        ));
    }

    #[test]
    fn test_manager_available_platforms_during_exclusive() {
        let mut mgr = PlatformEmbargoManager::new();
        mgr.add_waterfall(ReleaseWaterfall::standard("m1", THEATRICAL_DATE));
        let during_theatrical = THEATRICAL_DATE + 5 * DAY;
        let available = mgr.available_platforms("m1", "US", during_theatrical);
        assert_eq!(available.len(), 1);
        assert_eq!(*available[0], DistributionPlatform::Theatrical);
    }

    #[test]
    fn test_manager_available_platforms_after_all_open() {
        let mut mgr = PlatformEmbargoManager::new();
        mgr.add_waterfall(ReleaseWaterfall::standard("m1", THEATRICAL_DATE));
        let far_future = THEATRICAL_DATE + 365 * DAY;
        let available = mgr.available_platforms("m1", "US", far_future);
        // Only broadcast (last perpetual window) should still be open
        // because earlier windows have closed
        assert!(!available.is_empty());
    }

    #[test]
    fn test_manager_upcoming_windows() {
        let mut mgr = PlatformEmbargoManager::new();
        mgr.add_waterfall(ReleaseWaterfall::standard("m1", THEATRICAL_DATE));
        let before_release = THEATRICAL_DATE - 1;
        let upcoming = mgr.upcoming_windows("m1", before_release);
        assert_eq!(upcoming.len(), 6); // all are upcoming
    }

    #[test]
    fn test_manager_upcoming_windows_some_open() {
        let mut mgr = PlatformEmbargoManager::new();
        mgr.add_waterfall(ReleaseWaterfall::standard("m1", THEATRICAL_DATE));
        let after_pvod = THEATRICAL_DATE + 18 * DAY;
        let upcoming = mgr.upcoming_windows("m1", after_pvod);
        // Theatrical and PVOD have opened, rest are upcoming
        assert_eq!(upcoming.len(), 4);
    }

    #[test]
    fn test_manager_windows_for() {
        let mut mgr = PlatformEmbargoManager::new();
        mgr.add_waterfall(ReleaseWaterfall::standard("m1", THEATRICAL_DATE));
        assert_eq!(mgr.windows_for("m1").len(), 6);
        assert!(mgr.windows_for("unknown").is_empty());
    }

    #[test]
    fn test_manager_day_and_date_all_available() {
        let mut mgr = PlatformEmbargoManager::new();
        mgr.add_waterfall(ReleaseWaterfall::day_and_date("m1", THEATRICAL_DATE));
        let available = mgr.available_platforms("m1", "US", THEATRICAL_DATE);
        assert_eq!(available.len(), 6);
    }

    #[test]
    fn test_manager_territory_restriction() {
        let mut mgr = PlatformEmbargoManager::new();
        let w = PlatformWindow::new("m1", DistributionPlatform::Svod, 0, None)
            .with_territories(vec!["US".to_string()]);
        mgr.add_window(w);
        assert!(mgr.can_distribute("m1", &DistributionPlatform::Svod, "US", 1000));
        assert!(!mgr.can_distribute("m1", &DistributionPlatform::Svod, "JP", 1000));
    }

    #[test]
    fn test_manager_multiple_content_items() {
        let mut mgr = PlatformEmbargoManager::new();
        mgr.add_waterfall(ReleaseWaterfall::standard("m1", THEATRICAL_DATE));
        mgr.add_waterfall(ReleaseWaterfall::standard("m2", THEATRICAL_DATE + 30 * DAY));
        assert_eq!(mgr.content_count(), 2);
        assert_eq!(mgr.window_count(), 12);
    }
}
