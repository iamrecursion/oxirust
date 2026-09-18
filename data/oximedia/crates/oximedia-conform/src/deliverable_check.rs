//! Deliverable specification checking for media conform in `OxiMedia`.
//!
//! [`DeliverableChecker`] evaluates a list of [`DeliverableItem`]s against a
//! [`DeliverableSpec`] and reports pass/fail counts.

#![allow(dead_code)]

/// Video resolution preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Resolution {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl Resolution {
    /// Create a new resolution.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// 1920×1080.
    #[must_use]
    pub fn fhd() -> Self {
        Self::new(1920, 1080)
    }

    /// 3840×2160.
    #[must_use]
    pub fn uhd() -> Self {
        Self::new(3840, 2160)
    }

    /// 1280×720.
    #[must_use]
    pub fn hd720() -> Self {
        Self::new(1280, 720)
    }

    /// Total pixel count.
    #[must_use]
    pub fn pixel_count(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }
}

/// Specification that a deliverable must meet.
#[derive(Debug, Clone)]
pub struct DeliverableSpec {
    /// Required resolution.
    pub resolution: Resolution,
    /// Required frame rate (fps).
    pub frame_rate: f64,
    /// Required video codec name (e.g., `"h264"`).
    pub video_codec: String,
    /// Required audio codec name (e.g., `"aac"`).
    pub audio_codec: String,
    /// Maximum allowed file size in bytes (0 = no limit).
    pub max_file_size_bytes: u64,
    /// Required audio sample rate in Hz.
    pub audio_sample_rate: u32,
}

impl DeliverableSpec {
    /// Create a new deliverable spec.
    #[must_use]
    pub fn new(
        resolution: Resolution,
        frame_rate: f64,
        video_codec: &str,
        audio_codec: &str,
        max_file_size_bytes: u64,
        audio_sample_rate: u32,
    ) -> Self {
        Self {
            resolution,
            frame_rate,
            video_codec: video_codec.to_owned(),
            audio_codec: audio_codec.to_owned(),
            max_file_size_bytes,
            audio_sample_rate,
        }
    }

    /// Return `true` when the spec requires HD (1920×1080) or better.
    #[must_use]
    pub fn is_hd(&self) -> bool {
        self.resolution.width >= 1920 && self.resolution.height >= 1080
    }

    /// Return `true` when the spec requires UHD (3840×2160) or better.
    #[must_use]
    pub fn is_uhd(&self) -> bool {
        self.resolution.width >= 3840 && self.resolution.height >= 2160
    }
}

/// A single deliverable item with measured properties.
#[derive(Debug, Clone)]
pub struct DeliverableItem {
    /// Human-readable name for the item.
    pub name: String,
    /// Measured resolution.
    pub resolution: Resolution,
    /// Measured frame rate (fps).
    pub frame_rate: f64,
    /// Measured video codec name.
    pub video_codec: String,
    /// Measured audio codec name.
    pub audio_codec: String,
    /// Actual file size in bytes.
    pub file_size_bytes: u64,
    /// Measured audio sample rate in Hz.
    pub audio_sample_rate: u32,
}

impl DeliverableItem {
    /// Create a new deliverable item.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        name: &str,
        resolution: Resolution,
        frame_rate: f64,
        video_codec: &str,
        audio_codec: &str,
        file_size_bytes: u64,
        audio_sample_rate: u32,
    ) -> Self {
        Self {
            name: name.to_owned(),
            resolution,
            frame_rate,
            video_codec: video_codec.to_owned(),
            audio_codec: audio_codec.to_owned(),
            file_size_bytes,
            audio_sample_rate,
        }
    }

    /// Return `true` when this item's properties all meet the given spec.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn passes_spec(&self, spec: &DeliverableSpec) -> bool {
        self.resolution == spec.resolution
            && (self.frame_rate - spec.frame_rate).abs() < 0.01
            && self.video_codec == spec.video_codec
            && self.audio_codec == spec.audio_codec
            && self.audio_sample_rate == spec.audio_sample_rate
            && (spec.max_file_size_bytes == 0 || self.file_size_bytes <= spec.max_file_size_bytes)
    }
}

/// Outcome for a single item check.
#[derive(Debug, Clone)]
pub struct ItemCheckResult {
    /// Name of the deliverable item.
    pub name: String,
    /// Whether the item passed the spec.
    pub passed: bool,
    /// Brief reason for failure, if any.
    pub failure_reason: Option<String>,
}

impl ItemCheckResult {
    /// Return `true` when the item passed.
    #[must_use]
    pub fn is_pass(&self) -> bool {
        self.passed
    }
}

/// Checks a collection of deliverable items against a spec.
#[derive(Debug)]
pub struct DeliverableChecker {
    spec: DeliverableSpec,
}

impl DeliverableChecker {
    /// Create a checker for the given spec.
    #[must_use]
    pub fn new(spec: DeliverableSpec) -> Self {
        Self { spec }
    }

    /// Check all items and return per-item results.
    #[must_use]
    pub fn check(&self, items: &[DeliverableItem]) -> Vec<ItemCheckResult> {
        items
            .iter()
            .map(|item| {
                let passed = item.passes_spec(&self.spec);
                let failure_reason = if passed {
                    None
                } else {
                    Some(self.build_reason(item))
                };
                ItemCheckResult {
                    name: item.name.clone(),
                    passed,
                    failure_reason,
                }
            })
            .collect()
    }

    /// Count items that pass the spec.
    #[must_use]
    pub fn passing_count(&self, items: &[DeliverableItem]) -> usize {
        items.iter().filter(|i| i.passes_spec(&self.spec)).count()
    }

    /// Count items that fail the spec.
    #[must_use]
    pub fn failing_count(&self, items: &[DeliverableItem]) -> usize {
        items.iter().filter(|i| !i.passes_spec(&self.spec)).count()
    }

    /// Build a short human-readable failure reason.
    #[allow(clippy::cast_precision_loss)]
    fn build_reason(&self, item: &DeliverableItem) -> String {
        let mut reasons = Vec::new();
        if item.resolution != self.spec.resolution {
            reasons.push(format!(
                "resolution {}x{} != {}x{}",
                item.resolution.width,
                item.resolution.height,
                self.spec.resolution.width,
                self.spec.resolution.height
            ));
        }
        if (item.frame_rate - self.spec.frame_rate).abs() >= 0.01 {
            reasons.push(format!(
                "fps {:.3} != {:.3}",
                item.frame_rate, self.spec.frame_rate
            ));
        }
        if item.video_codec != self.spec.video_codec {
            reasons.push(format!(
                "vcodec {} != {}",
                item.video_codec, self.spec.video_codec
            ));
        }
        if item.audio_codec != self.spec.audio_codec {
            reasons.push(format!(
                "acodec {} != {}",
                item.audio_codec, self.spec.audio_codec
            ));
        }
        if item.audio_sample_rate != self.spec.audio_sample_rate {
            reasons.push(format!(
                "sample_rate {} != {}",
                item.audio_sample_rate, self.spec.audio_sample_rate
            ));
        }
        if self.spec.max_file_size_bytes > 0 && item.file_size_bytes > self.spec.max_file_size_bytes
        {
            reasons.push(format!(
                "size {} > {}",
                item.file_size_bytes, self.spec.max_file_size_bytes
            ));
        }
        if reasons.is_empty() {
            "unknown".to_owned()
        } else {
            reasons.join("; ")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hd_spec() -> DeliverableSpec {
        DeliverableSpec::new(Resolution::fhd(), 25.0, "h264", "aac", 500_000_000, 48000)
    }

    fn good_item(name: &str) -> DeliverableItem {
        DeliverableItem::new(
            name,
            Resolution::fhd(),
            25.0,
            "h264",
            "aac",
            100_000_000,
            48000,
        )
    }

    // ── Resolution ───────────────────────────────────────────────────────────

    #[test]
    fn test_resolution_pixel_count() {
        let r = Resolution::fhd();
        assert_eq!(r.pixel_count(), 1920 * 1080);
    }

    #[test]
    fn test_resolution_equality() {
        assert_eq!(Resolution::fhd(), Resolution::new(1920, 1080));
        assert_ne!(Resolution::fhd(), Resolution::uhd());
    }

    // ── DeliverableSpec ──────────────────────────────────────────────────────

    #[test]
    fn test_spec_is_hd_true() {
        assert!(hd_spec().is_hd());
    }

    #[test]
    fn test_spec_is_hd_false() {
        let spec = DeliverableSpec::new(Resolution::hd720(), 25.0, "h264", "aac", 0, 48000);
        assert!(!spec.is_hd());
    }

    #[test]
    fn test_spec_is_uhd() {
        let spec = DeliverableSpec::new(Resolution::uhd(), 25.0, "h264", "aac", 0, 48000);
        assert!(spec.is_uhd());
        assert!(!hd_spec().is_uhd());
    }

    // ── DeliverableItem ──────────────────────────────────────────────────────

    #[test]
    fn test_item_passes_matching_spec() {
        let item = good_item("ep01");
        assert!(item.passes_spec(&hd_spec()));
    }

    #[test]
    fn test_item_fails_wrong_codec() {
        let item = DeliverableItem::new(
            "ep01",
            Resolution::fhd(),
            25.0,
            "hevc",
            "aac",
            100_000_000,
            48000,
        );
        assert!(!item.passes_spec(&hd_spec()));
    }

    #[test]
    fn test_item_fails_wrong_resolution() {
        let item = DeliverableItem::new(
            "ep01",
            Resolution::hd720(),
            25.0,
            "h264",
            "aac",
            100_000_000,
            48000,
        );
        assert!(!item.passes_spec(&hd_spec()));
    }

    #[test]
    fn test_item_fails_exceeds_size_limit() {
        let item = DeliverableItem::new(
            "ep01",
            Resolution::fhd(),
            25.0,
            "h264",
            "aac",
            999_999_999,
            48000,
        );
        assert!(!item.passes_spec(&hd_spec()));
    }

    #[test]
    fn test_item_passes_no_size_limit() {
        let spec = DeliverableSpec::new(Resolution::fhd(), 25.0, "h264", "aac", 0, 48000);
        let item = DeliverableItem::new(
            "ep01",
            Resolution::fhd(),
            25.0,
            "h264",
            "aac",
            u64::MAX,
            48000,
        );
        assert!(item.passes_spec(&spec));
    }

    // ── DeliverableChecker ───────────────────────────────────────────────────

    #[test]
    fn test_checker_all_pass() {
        let checker = DeliverableChecker::new(hd_spec());
        let items = vec![good_item("ep01"), good_item("ep02")];
        let results = checker.check(&items);
        assert!(results.iter().all(super::ItemCheckResult::is_pass));
        assert_eq!(checker.passing_count(&items), 2);
        assert_eq!(checker.failing_count(&items), 0);
    }

    #[test]
    fn test_checker_partial_fail() {
        let checker = DeliverableChecker::new(hd_spec());
        let bad = DeliverableItem::new("ep02", Resolution::hd720(), 25.0, "h264", "aac", 0, 48000);
        let items = vec![good_item("ep01"), bad];
        assert_eq!(checker.passing_count(&items), 1);
        assert_eq!(checker.failing_count(&items), 1);
    }

    #[test]
    fn test_checker_failure_reason_present() {
        let checker = DeliverableChecker::new(hd_spec());
        let bad = DeliverableItem::new("ep01", Resolution::hd720(), 25.0, "h264", "aac", 0, 48000);
        let results = checker.check(&[bad]);
        assert!(!results[0].is_pass());
        assert!(results[0].failure_reason.is_some());
    }

    #[test]
    fn test_checker_pass_no_reason() {
        let checker = DeliverableChecker::new(hd_spec());
        let results = checker.check(&[good_item("ep01")]);
        assert!(results[0].is_pass());
        assert!(results[0].failure_reason.is_none());
    }

    #[test]
    fn test_checker_empty_list() {
        let checker = DeliverableChecker::new(hd_spec());
        let results = checker.check(&[]);
        assert!(results.is_empty());
        assert_eq!(checker.passing_count(&[]), 0);
    }

    #[test]
    fn test_checker_fps_tolerance() {
        let checker = DeliverableChecker::new(hd_spec());
        // 25.005 is within 0.01 tolerance
        let item = DeliverableItem::new("ep01", Resolution::fhd(), 25.005, "h264", "aac", 0, 48000);
        assert!(item.passes_spec(&hd_spec()));
        // 25.02 is outside tolerance
        let item2 = DeliverableItem::new("ep01", Resolution::fhd(), 25.02, "h264", "aac", 0, 48000);
        let results = checker.check(&[item2]);
        assert!(!results[0].is_pass());
    }
}
