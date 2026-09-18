//! Review delivery management.
//!
//! Manages delivery packages sent to reviewers, including format selection,
//! optional watermarking, expiry tracking, and download counting.

/// Output format for a review delivery package.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum DeliveryFormat {
    /// H.264 MP4 screener for general distribution.
    ScreenerMp4,
    /// MXF container, typically used in broadcast pipelines.
    Mxf,
    /// Apple ProRes codec for editing-grade delivery.
    ProRes,
    /// H.264 at HD resolution.
    H264Hd,
    /// Low-bitrate web preview intended for quick review.
    WebPreview,
}

impl DeliveryFormat {
    /// Returns the typical bitrate for this format in Mbit/s.
    #[must_use]
    pub fn typical_bitrate_mbps(&self) -> f32 {
        match self {
            Self::ScreenerMp4 => 8.0,
            Self::Mxf => 50.0,
            Self::ProRes => 145.0,
            Self::H264Hd => 15.0,
            Self::WebPreview => 2.0,
        }
    }

    /// Returns `true` if this format is considered broadcast quality.
    #[must_use]
    pub fn is_broadcast_quality(&self) -> bool {
        matches!(self, Self::Mxf | Self::ProRes)
    }
}

/// Position of a watermark overlay within the frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum WatermarkPos {
    /// Top-left corner.
    TopLeft,
    /// Top-right corner.
    TopRight,
    /// Bottom-left corner.
    BottomLeft,
    /// Bottom-right corner.
    BottomRight,
    /// Centred in the frame.
    Center,
}

/// Configuration for the text watermark burned into a delivery package.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct WatermarkConfig {
    /// Text to render as the watermark.
    pub text: String,
    /// Opacity from 0.0 (invisible) to 1.0 (fully opaque).
    pub opacity: f32,
    /// Where to position the watermark within the frame.
    pub position: WatermarkPos,
}

impl WatermarkConfig {
    /// Create a watermark configuration suitable for screener distribution.
    ///
    /// Uses semi-transparent text centred in the frame.
    #[must_use]
    pub fn default_screener() -> Self {
        Self {
            text: "SCREENER – NOT FOR DISTRIBUTION".to_string(),
            opacity: 0.35,
            position: WatermarkPos::Center,
        }
    }
}

/// A single delivery package dispatched to a reviewer.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct DeliveryPackage {
    /// Unique identifier.
    pub id: u64,
    /// Identifier of the media asset being delivered.
    pub media_id: String,
    /// Output format.
    pub format: DeliveryFormat,
    /// Optional watermark configuration.
    pub watermark: Option<WatermarkConfig>,
    /// Absolute expiry time in milliseconds since epoch.
    pub expires_ms: u64,
    /// Number of times this package has been downloaded.
    pub download_count: u32,
}

impl DeliveryPackage {
    /// Returns `true` if the package has passed its expiry time.
    #[must_use]
    pub fn is_expired(&self, now_ms: u64) -> bool {
        now_ms >= self.expires_ms
    }

    /// Increment the download counter by one.
    pub fn increment_downloads(&mut self) {
        self.download_count = self.download_count.saturating_add(1);
    }

    /// Returns `true` if a watermark is configured for this package.
    #[must_use]
    pub fn has_watermark(&self) -> bool {
        self.watermark.is_some()
    }
}

/// Manages the lifecycle of delivery packages.
#[derive(Debug, Default)]
#[allow(dead_code)]
pub struct DeliveryManager {
    /// All packages, both active and expired.
    pub packages: Vec<DeliveryPackage>,
    /// Next ID to assign when creating a package.
    pub next_id: u64,
}

impl DeliveryManager {
    /// Create an empty `DeliveryManager`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            packages: Vec::new(),
            next_id: 1,
        }
    }

    /// Create and register a new delivery package.
    ///
    /// # Arguments
    ///
    /// * `media_id` - Identifier of the media asset.
    /// * `format`   - Output format.
    /// * `ttl_ms`   - Time-to-live in milliseconds relative to `now_ms`.
    /// * `now_ms`   - Current time in milliseconds since epoch.
    ///
    /// Returns the ID of the newly created package.
    pub fn create(
        &mut self,
        media_id: &str,
        format: DeliveryFormat,
        ttl_ms: u64,
        now_ms: u64,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.packages.push(DeliveryPackage {
            id,
            media_id: media_id.to_string(),
            format,
            watermark: None,
            expires_ms: now_ms.saturating_add(ttl_ms),
            download_count: 0,
        });
        id
    }

    /// Look up a package by its ID.
    #[must_use]
    pub fn get(&self, id: u64) -> Option<&DeliveryPackage> {
        self.packages.iter().find(|p| p.id == id)
    }

    /// Record a download for the package with the given ID.
    ///
    /// Returns `true` if the package was found.
    pub fn record_download(&mut self, id: u64) -> bool {
        if let Some(pkg) = self.packages.iter_mut().find(|p| p.id == id) {
            pkg.increment_downloads();
            true
        } else {
            false
        }
    }

    /// Return all packages that have not yet expired as of `now_ms`.
    #[must_use]
    pub fn active_packages(&self, now_ms: u64) -> Vec<&DeliveryPackage> {
        self.packages
            .iter()
            .filter(|p| !p.is_expired(now_ms))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- DeliveryFormat ---

    #[test]
    fn test_prores_is_broadcast_quality() {
        assert!(DeliveryFormat::ProRes.is_broadcast_quality());
    }

    #[test]
    fn test_mxf_is_broadcast_quality() {
        assert!(DeliveryFormat::Mxf.is_broadcast_quality());
    }

    #[test]
    fn test_web_preview_not_broadcast_quality() {
        assert!(!DeliveryFormat::WebPreview.is_broadcast_quality());
    }

    #[test]
    fn test_screener_mp4_not_broadcast_quality() {
        assert!(!DeliveryFormat::ScreenerMp4.is_broadcast_quality());
    }

    #[test]
    fn test_typical_bitrate_ordering() {
        assert!(
            DeliveryFormat::ProRes.typical_bitrate_mbps()
                > DeliveryFormat::H264Hd.typical_bitrate_mbps()
        );
        assert!(
            DeliveryFormat::WebPreview.typical_bitrate_mbps()
                < DeliveryFormat::ScreenerMp4.typical_bitrate_mbps()
        );
    }

    // --- WatermarkConfig ---

    #[test]
    fn test_default_screener_watermark_center() {
        let wm = WatermarkConfig::default_screener();
        assert_eq!(wm.position, WatermarkPos::Center);
    }

    #[test]
    fn test_default_screener_watermark_semi_transparent() {
        let wm = WatermarkConfig::default_screener();
        assert!(wm.opacity > 0.0 && wm.opacity < 1.0);
    }

    #[test]
    fn test_default_screener_watermark_text_non_empty() {
        let wm = WatermarkConfig::default_screener();
        assert!(!wm.text.is_empty());
    }

    // --- DeliveryPackage ---

    #[test]
    fn test_package_expired_when_now_ge_expires() {
        let pkg = DeliveryPackage {
            id: 1,
            media_id: "m1".to_string(),
            format: DeliveryFormat::ScreenerMp4,
            watermark: None,
            expires_ms: 1_000,
            download_count: 0,
        };
        assert!(pkg.is_expired(1_000));
        assert!(pkg.is_expired(2_000));
    }

    #[test]
    fn test_package_not_expired_before_expiry() {
        let pkg = DeliveryPackage {
            id: 2,
            media_id: "m2".to_string(),
            format: DeliveryFormat::H264Hd,
            watermark: None,
            expires_ms: 5_000,
            download_count: 0,
        };
        assert!(!pkg.is_expired(4_999));
    }

    #[test]
    fn test_package_increment_downloads() {
        let mut pkg = DeliveryPackage {
            id: 3,
            media_id: "m3".to_string(),
            format: DeliveryFormat::ProRes,
            watermark: None,
            expires_ms: 99_999,
            download_count: 0,
        };
        pkg.increment_downloads();
        pkg.increment_downloads();
        assert_eq!(pkg.download_count, 2);
    }

    #[test]
    fn test_package_has_watermark_false_when_none() {
        let pkg = DeliveryPackage {
            id: 4,
            media_id: "m4".to_string(),
            format: DeliveryFormat::WebPreview,
            watermark: None,
            expires_ms: 99_999,
            download_count: 0,
        };
        assert!(!pkg.has_watermark());
    }

    #[test]
    fn test_package_has_watermark_true_when_some() {
        let pkg = DeliveryPackage {
            id: 5,
            media_id: "m5".to_string(),
            format: DeliveryFormat::ScreenerMp4,
            watermark: Some(WatermarkConfig::default_screener()),
            expires_ms: 99_999,
            download_count: 0,
        };
        assert!(pkg.has_watermark());
    }

    // --- DeliveryManager ---

    #[test]
    fn test_manager_create_returns_sequential_ids() {
        let mut mgr = DeliveryManager::new();
        let id1 = mgr.create("m1", DeliveryFormat::ScreenerMp4, 3_600_000, 0);
        let id2 = mgr.create("m2", DeliveryFormat::H264Hd, 3_600_000, 0);
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
    }

    #[test]
    fn test_manager_get_returns_package() {
        let mut mgr = DeliveryManager::new();
        mgr.create("media-abc", DeliveryFormat::Mxf, 7_200_000, 1_000);
        let pkg = mgr.get(1).expect("should succeed in test");
        assert_eq!(pkg.media_id, "media-abc");
    }

    #[test]
    fn test_manager_get_nonexistent_returns_none() {
        let mgr = DeliveryManager::new();
        assert!(mgr.get(999).is_none());
    }

    #[test]
    fn test_manager_record_download() {
        let mut mgr = DeliveryManager::new();
        mgr.create("m1", DeliveryFormat::WebPreview, 3_600_000, 0);
        assert!(mgr.record_download(1));
        assert_eq!(
            mgr.get(1).expect("should succeed in test").download_count,
            1
        );
    }

    #[test]
    fn test_manager_record_download_nonexistent() {
        let mut mgr = DeliveryManager::new();
        assert!(!mgr.record_download(42));
    }

    #[test]
    fn test_manager_active_packages_excludes_expired() {
        let mut mgr = DeliveryManager::new();
        // expires at 1_000
        mgr.create("m1", DeliveryFormat::ScreenerMp4, 1_000, 0);
        // expires at 10_000
        mgr.create("m2", DeliveryFormat::H264Hd, 10_000, 0);
        let now = 2_000;
        let active = mgr.active_packages(now);
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].media_id, "m2");
    }
}
