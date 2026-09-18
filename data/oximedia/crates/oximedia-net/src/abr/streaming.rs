//! Streaming bandwidth measurement and quality adaptation utilities.

/// A single bandwidth measurement sample.
#[derive(Debug, Clone)]
pub struct BandwidthSample {
    /// Bytes downloaded in this measurement.
    pub bytes: u64,
    /// Wall-clock download time in milliseconds.
    pub duration_ms: u64,
    /// Instant when the sample was recorded.
    pub timestamp: std::time::Instant,
}

/// Exponentially-weighted moving average (EWMA) bandwidth estimator.
///
/// Maintains a sliding window of [`BandwidthSample`]s and computes an EWMA
/// estimate as well as order-statistic (percentile) estimates.
#[derive(Debug)]
pub struct AbrBandwidthEstimator {
    window: std::collections::VecDeque<BandwidthSample>,
    window_size: usize,
    smoothing_factor: f64,
    current_estimate_bps: f64,
}

impl AbrBandwidthEstimator {
    /// Creates a new estimator.
    ///
    /// * `window_size`      — maximum number of samples retained (default 10).
    /// * `smoothing_factor` — EWMA alpha ∈ `[0, 1]` (default 0.3, lower is smoother).
    #[must_use]
    pub fn new(window_size: usize) -> Self {
        Self {
            window: std::collections::VecDeque::with_capacity(window_size.max(1)),
            window_size: window_size.max(1),
            smoothing_factor: 0.3,
            current_estimate_bps: 0.0,
        }
    }

    /// Returns the current EWMA bandwidth estimate in bits per second.
    #[must_use]
    pub fn estimate_bps(&self) -> f64 {
        self.current_estimate_bps
    }

    /// Returns the current estimate in kilobits per second.
    #[must_use]
    pub fn estimate_kbps(&self) -> f64 {
        self.current_estimate_bps / 1_000.0
    }

    /// Returns the current estimate in megabits per second.
    #[must_use]
    pub fn estimate_mbps(&self) -> f64 {
        self.current_estimate_bps / 1_000_000.0
    }

    /// Adds a new download measurement and updates the EWMA estimate.
    pub fn add_sample(&mut self, bytes: u64, duration_ms: u64) {
        // Compute instantaneous rate in bits per second.
        let sample_bps = if duration_ms == 0 {
            0.0
        } else {
            (bytes as f64 * 8.0 * 1_000.0) / duration_ms as f64
        };

        if self.current_estimate_bps <= 0.0 {
            self.current_estimate_bps = sample_bps;
        } else {
            self.current_estimate_bps = self.smoothing_factor * sample_bps
                + (1.0 - self.smoothing_factor) * self.current_estimate_bps;
        }

        let sample = BandwidthSample {
            bytes,
            duration_ms,
            timestamp: std::time::Instant::now(),
        };

        if self.window.len() >= self.window_size {
            self.window.pop_front();
        }
        self.window.push_back(sample);
    }

    /// Returns the `percentile`-th order statistic of the sample rates,
    /// where `percentile` ∈ `[0.0, 1.0]`.
    ///
    /// For example `percentile_bps(0.15)` gives the 15th-percentile rate —
    /// a conservative estimate suitable for cautious ABR logic.
    #[must_use]
    pub fn percentile_bps(&self, percentile: f64) -> f64 {
        if self.window.is_empty() {
            return 0.0;
        }

        let mut rates: Vec<f64> = self
            .window
            .iter()
            .map(|s| {
                if s.duration_ms == 0 {
                    0.0
                } else {
                    (s.bytes as f64 * 8.0 * 1_000.0) / s.duration_ms as f64
                }
            })
            .collect();
        rates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let p = percentile.clamp(0.0, 1.0);
        let idx = ((rates.len() as f64 - 1.0) * p) as usize;
        rates[idx.min(rates.len() - 1)]
    }

    /// Returns the number of samples currently retained.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.window.len()
    }
}

/// One rendition of an adaptive stream (a single quality level).
#[derive(Debug, Clone)]
pub struct AbrVariant {
    /// Peak bandwidth in bits per second.
    pub bandwidth: u64,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Codec string (e.g. `"avc1.42c01e,mp4a.40.2"`).
    pub codecs: String,
    /// Playlist / segment URI for this rendition.
    pub uri: String,
    /// Human-readable name such as `"1080p"`.
    pub name: String,
    /// Frame rate, if known.
    pub frame_rate: Option<f64>,
    /// HDCP level, if specified.
    pub hdcp_level: Option<String>,
}

/// Reason why a quality switch was made.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbrSwitchReason {
    /// Estimated bandwidth increased enough to warrant a higher rendition.
    BandwidthIncrease,
    /// Estimated bandwidth decreased — switching down to stay sustainable.
    BandwidthDecrease,
    /// Buffer fell below the panic threshold.
    BufferStarvation,
    /// The application requested a specific rendition.
    UserRequested,
}

/// Result returned by [`AbrController::select_variant`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionResult {
    /// Staying on the current variant.
    Stay {
        /// Index of the current variant.
        variant: usize,
    },
    /// Switching to a higher-quality variant.
    SwitchUp {
        /// Index of the old variant.
        from: usize,
        /// Index of the new variant.
        to: usize,
        /// Reason for the switch.
        reason: AbrSwitchReason,
    },
    /// Switching to a lower-quality variant.
    SwitchDown {
        /// Index of the old variant.
        from: usize,
        /// Index of the new variant.
        to: usize,
        /// Reason for the switch.
        reason: AbrSwitchReason,
    },
    /// Emergency switch to the lowest available variant.
    EmergencySwitch {
        /// Index of the old variant.
        from: usize,
        /// Index of the new (lowest) variant.
        to: usize,
    },
}

impl SelectionResult {
    /// Returns the variant index that should be used after this decision.
    #[must_use]
    pub const fn variant_index(&self) -> usize {
        match self {
            Self::Stay { variant } => *variant,
            Self::SwitchUp { to, .. } => *to,
            Self::SwitchDown { to, .. } => *to,
            Self::EmergencySwitch { to, .. } => *to,
        }
    }

    /// Returns `true` if this decision involves changing the active variant.
    #[must_use]
    pub const fn is_switch(&self) -> bool {
        !matches!(self, Self::Stay { .. })
    }

    /// Returns `true` if this is an emergency switch.
    #[must_use]
    pub const fn is_emergency(&self) -> bool {
        matches!(self, Self::EmergencySwitch { .. })
    }
}

/// Adaptive bitrate controller for segment-based streaming.
///
/// Maintains a sorted list of [`AbrVariant`]s and selects the most appropriate
/// one at each segment boundary based on measured bandwidth and buffer level.
#[derive(Debug)]
pub struct AbrController {
    /// All renditions, sorted by bandwidth ascending.
    variants: Vec<AbrVariant>,
    /// Index into `variants` of the currently active rendition.
    current_index: usize,
    /// Bandwidth estimator.
    bandwidth_estimator: AbrBandwidthEstimator,
    /// Current buffer level in seconds.
    buffer_duration_s: f64,
    /// Minimum buffer required before attempting a switch-up (seconds).
    min_buffer_s: f64,
    /// Buffer level below which an emergency switch-down is triggered (seconds).
    panic_buffer_s: f64,
    /// Fraction of estimated bandwidth to use for selection (headroom).
    safety_factor: f64,
    /// Number of segments to wait between quality switches.
    switch_cooldown_segments: u32,
    /// Segments downloaded since the last quality switch.
    segments_since_switch: u32,
}

impl AbrController {
    /// Creates a new controller from a list of variants.
    ///
    /// Variants are sorted by bandwidth ascending so that index 0 is always
    /// the lowest quality.  Returns `Err` if `variants` is empty.
    pub fn new(mut variants: Vec<AbrVariant>) -> Result<Self, String> {
        if variants.is_empty() {
            return Err("AbrController requires at least one variant".into());
        }
        variants.sort_by_key(|v| v.bandwidth);
        Ok(Self {
            bandwidth_estimator: AbrBandwidthEstimator::new(10),
            variants,
            current_index: 0,
            buffer_duration_s: 0.0,
            min_buffer_s: 15.0,
            panic_buffer_s: 5.0,
            safety_factor: 0.8,
            switch_cooldown_segments: 3,
            segments_since_switch: 0,
        })
    }

    /// Returns a reference to the currently active variant.
    #[must_use]
    pub fn current_variant(&self) -> &AbrVariant {
        &self.variants[self.current_index]
    }

    /// Returns the total number of variants.
    #[must_use]
    pub fn variant_count(&self) -> usize {
        self.variants.len()
    }

    /// Feeds a new segment download measurement into the bandwidth estimator.
    pub fn update_bandwidth(&mut self, bytes: u64, duration_ms: u64) {
        self.bandwidth_estimator.add_sample(bytes, duration_ms);
    }

    /// Updates the current buffer level.
    pub fn update_buffer(&mut self, buffer_duration_s: f64) {
        self.buffer_duration_s = buffer_duration_s;
    }

    /// Runs the core ABR logic and returns a [`SelectionResult`].
    ///
    /// Decision rules (in priority order):
    /// 1. Buffer below panic threshold → emergency switch to index 0.
    /// 2. Cooldown active → stay.
    /// 3. Compute `safe_bw = estimate * safety_factor`.
    /// 4. Find highest variant whose bandwidth ≤ safe_bw.
    /// 5. If buffer ≥ min_buffer_s allow switching up one step; otherwise
    ///    only allow switching down.
    pub fn select_variant(&mut self) -> SelectionResult {
        let old = self.current_index;

        // Rule 1: emergency.
        if self.buffer_duration_s < self.panic_buffer_s && old > 0 {
            self.current_index = 0;
            self.segments_since_switch = 0;
            return SelectionResult::EmergencySwitch { from: old, to: 0 };
        }

        // Rule 2: cooldown.
        if self.segments_since_switch < self.switch_cooldown_segments {
            self.segments_since_switch += 1;
            return SelectionResult::Stay { variant: old };
        }

        // Rule 3-4: find best variant by bandwidth.
        let safe_bw = self.bandwidth_estimator.estimate_bps() * self.safety_factor;
        let mut target = 0usize;
        for (i, v) in self.variants.iter().enumerate() {
            if v.bandwidth as f64 <= safe_bw {
                target = i;
            }
        }

        // Rule 5: buffer-gated upswitch.
        let result = if target > old {
            if self.buffer_duration_s >= self.min_buffer_s {
                // Allow at most one step up.
                let next = (old + 1).min(target);
                self.current_index = next;
                self.segments_since_switch = 0;
                SelectionResult::SwitchUp {
                    from: old,
                    to: next,
                    reason: AbrSwitchReason::BandwidthIncrease,
                }
            } else {
                // Buffer too low to switch up → stay.
                self.segments_since_switch += 1;
                SelectionResult::Stay { variant: old }
            }
        } else if target < old {
            self.current_index = target;
            self.segments_since_switch = 0;
            SelectionResult::SwitchDown {
                from: old,
                to: target,
                reason: AbrSwitchReason::BandwidthDecrease,
            }
        } else {
            self.segments_since_switch += 1;
            SelectionResult::Stay { variant: old }
        };

        result
    }

    /// Forces a specific variant index, bypassing ABR logic.
    pub fn force_variant(&mut self, index: usize) -> Result<(), String> {
        if index >= self.variants.len() {
            return Err(format!(
                "Variant index {index} out of range (max {})",
                self.variants.len() - 1
            ));
        }
        self.current_index = index;
        self.segments_since_switch = 0;
        Ok(())
    }
}

/// A segment that has been downloaded and placed in the playback buffer.
#[derive(Debug, Clone)]
pub struct BufferedSegment {
    /// Sequence number of this segment.
    pub sequence: u64,
    /// Variant index this segment was downloaded at.
    pub variant_index: usize,
    /// Raw segment bytes.
    pub data: Vec<u8>,
    /// Playback duration of this segment in seconds.
    pub duration_s: f64,
    /// Download time in milliseconds.
    pub download_time_ms: u64,
}

/// Drives an [`AbrController`] and maintains an in-memory segment buffer.
#[derive(Debug)]
pub struct SegmentFetcher {
    /// Underlying ABR controller.
    controller: AbrController,
    /// Default segment playback duration in seconds.
    segment_duration_s: f64,
    /// Maximum number of segments to keep buffered.
    max_buffer_segments: usize,
    /// The buffered segment queue (oldest at front).
    buffered_segments: std::collections::VecDeque<BufferedSegment>,
}

impl SegmentFetcher {
    /// Creates a new fetcher wrapping the given controller.
    #[must_use]
    pub fn new(controller: AbrController, segment_duration_s: f64) -> Self {
        Self {
            controller,
            segment_duration_s,
            max_buffer_segments: 30,
            buffered_segments: std::collections::VecDeque::new(),
        }
    }

    /// Returns the total playback duration currently buffered, in seconds.
    #[must_use]
    pub fn buffer_level_s(&self) -> f64 {
        self.buffered_segments.iter().map(|s| s.duration_s).sum()
    }

    /// Selects the variant to use for the next segment download.
    ///
    /// Calls [`AbrController::select_variant`] then updates the buffer level
    /// inside the controller so the next call has accurate state.
    pub fn next_variant(&mut self) -> &AbrVariant {
        let _result = self.controller.select_variant();
        let buf = self.buffer_level_s();
        self.controller.update_buffer(buf);
        self.controller.current_variant()
    }

    /// Records a completed segment download.
    ///
    /// Updates bandwidth estimation and appends a [`BufferedSegment`] to the
    /// buffer.  If the buffer exceeds `max_buffer_segments`, the oldest entry
    /// is silently dropped.
    pub fn record_download(
        &mut self,
        sequence: u64,
        bytes: u64,
        duration_ms: u64,
        segment_duration_s: f64,
    ) {
        self.controller.update_bandwidth(bytes, duration_ms);
        let variant_index = self.controller.current_index;
        let seg = BufferedSegment {
            sequence,
            variant_index,
            data: Vec::new(), // caller fills in real data separately if needed
            duration_s: segment_duration_s,
            download_time_ms: duration_ms,
        };
        if self.buffered_segments.len() >= self.max_buffer_segments {
            self.buffered_segments.pop_front();
        }
        self.buffered_segments.push_back(seg);
    }

    /// Removes and returns the oldest buffered segment (simulating playback).
    pub fn pop_segment(&mut self) -> Option<BufferedSegment> {
        self.buffered_segments.pop_front()
    }

    /// Returns the number of segments currently buffered.
    #[must_use]
    pub fn buffered_count(&self) -> usize {
        self.buffered_segments.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests for streaming ABR types
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod streaming_abr_tests {
    use super::*;

    fn make_variants() -> Vec<AbrVariant> {
        vec![
            AbrVariant {
                bandwidth: 500_000,
                width: 640,
                height: 360,
                codecs: "avc1.42c01e,mp4a.40.2".into(),
                uri: "low.m3u8".into(),
                name: "360p".into(),
                frame_rate: Some(30.0),
                hdcp_level: None,
            },
            AbrVariant {
                bandwidth: 1_500_000,
                width: 1280,
                height: 720,
                codecs: "avc1.42c01e,mp4a.40.2".into(),
                uri: "mid.m3u8".into(),
                name: "720p".into(),
                frame_rate: Some(30.0),
                hdcp_level: None,
            },
            AbrVariant {
                bandwidth: 4_000_000,
                width: 1920,
                height: 1080,
                codecs: "avc1.640028,mp4a.40.2".into(),
                uri: "high.m3u8".into(),
                name: "1080p".into(),
                frame_rate: Some(60.0),
                hdcp_level: None,
            },
        ]
    }

    // ── BandwidthEstimator tests ──────────────────────────────────────────

    #[test]
    fn test_bandwidth_estimator_basic() {
        let mut est = AbrBandwidthEstimator::new(10);
        est.add_sample(1_000_000, 1_000); // 8 Mbps
        est.add_sample(2_000_000, 1_000); // 16 Mbps
        est.add_sample(1_500_000, 1_000); // 12 Mbps
        assert!(est.estimate_bps() > 0.0, "estimate should be positive");
        assert_eq!(est.sample_count(), 3);
    }

    #[test]
    fn test_bandwidth_estimator_percentile() {
        let mut est = AbrBandwidthEstimator::new(20);
        // 5 slow samples at 1 Mbps then 5 fast samples at 10 Mbps
        for _ in 0..5 {
            est.add_sample(125_000, 1_000); // 1 Mbps
        }
        for _ in 0..5 {
            est.add_sample(1_250_000, 1_000); // 10 Mbps
        }
        let p15 = est.percentile_bps(0.15);
        let p85 = est.percentile_bps(0.85);
        assert!(p15 < p85, "15th percentile should be lower than 85th");
        assert!(p15 > 0.0, "percentile should be positive");
    }

    // ── AbrController tests ───────────────────────────────────────────────

    #[test]
    fn test_abr_controller_creation() {
        // Provide variants out of bandwidth order; controller must sort them.
        let mut variants = make_variants();
        variants.reverse(); // highest bandwidth first
        let ctrl = AbrController::new(variants).expect("should create controller");
        assert_eq!(ctrl.variant_count(), 3);
        // After sorting, index 0 must be the lowest bandwidth.
        assert_eq!(ctrl.current_variant().bandwidth, 500_000);
    }

    #[test]
    fn test_abr_stay_on_low_buffer() {
        let mut ctrl = AbrController::new(make_variants()).expect("should succeed in test");
        // Force to highest variant.
        ctrl.force_variant(2).expect("should succeed in test");
        // Feed some bandwidth samples so estimate is non-zero.
        ctrl.update_bandwidth(500_000, 1_000); // 4 Mbps
                                               // Panic-level buffer.
        ctrl.update_buffer(2.0);
        // Reset cooldown so decision runs.
        ctrl.segments_since_switch = ctrl.switch_cooldown_segments;

        let result = ctrl.select_variant();
        assert!(
            result.is_emergency(),
            "expected emergency switch, got {result:?}"
        );
        assert_eq!(
            result.variant_index(),
            0,
            "emergency switch must go to index 0"
        );
    }

    #[test]
    fn test_abr_switch_up_good_bandwidth() {
        let mut ctrl = AbrController::new(make_variants()).expect("should succeed in test");
        // Start at index 0, simulate 40 Mbps link.
        // 5_000_000 bytes in 1000 ms = 40 Mbps
        ctrl.update_bandwidth(5_000_000, 1_000);
        // Healthy buffer.
        ctrl.update_buffer(20.0);
        // Ensure cooldown has expired.
        ctrl.segments_since_switch = ctrl.switch_cooldown_segments;

        let result = ctrl.select_variant();
        assert!(
            result.is_switch(),
            "expected a switch with excellent bandwidth"
        );
        assert!(
            result.variant_index() > 0,
            "should switch up from index 0, got {}",
            result.variant_index()
        );
    }

    #[test]
    fn test_abr_cooldown() {
        let mut ctrl = AbrController::new(make_variants()).expect("should succeed in test");
        // Feed strong bandwidth.
        ctrl.update_bandwidth(5_000_000, 1_000);
        ctrl.update_buffer(20.0);
        // Expire cooldown for the first call.
        ctrl.segments_since_switch = ctrl.switch_cooldown_segments;

        let first = ctrl.select_variant();
        // First call may switch up.
        let _ = first;

        // Immediately call again — cooldown should prevent another switch.
        let second = ctrl.select_variant();
        assert!(
            matches!(second, SelectionResult::Stay { .. }),
            "cooldown should prevent immediate second switch, got {second:?}"
        );
    }

    // ── SelectionResult tests ─────────────────────────────────────────────

    #[test]
    fn test_selection_result_accessors() {
        let stay = SelectionResult::Stay { variant: 1 };
        assert_eq!(stay.variant_index(), 1);
        assert!(!stay.is_switch());
        assert!(!stay.is_emergency());

        let up = SelectionResult::SwitchUp {
            from: 0,
            to: 1,
            reason: AbrSwitchReason::BandwidthIncrease,
        };
        assert_eq!(up.variant_index(), 1);
        assert!(up.is_switch());
        assert!(!up.is_emergency());

        let down = SelectionResult::SwitchDown {
            from: 2,
            to: 1,
            reason: AbrSwitchReason::BandwidthDecrease,
        };
        assert_eq!(down.variant_index(), 1);
        assert!(down.is_switch());
        assert!(!down.is_emergency());

        let emergency = SelectionResult::EmergencySwitch { from: 2, to: 0 };
        assert_eq!(emergency.variant_index(), 0);
        assert!(emergency.is_switch());
        assert!(emergency.is_emergency());
    }

    // ── SegmentFetcher tests ──────────────────────────────────────────────

    #[test]
    fn test_segment_fetcher_buffer_level() {
        let ctrl = AbrController::new(make_variants()).expect("should succeed in test");
        let mut fetcher = SegmentFetcher::new(ctrl, 4.0);

        fetcher.record_download(0, 500_000, 1_000, 4.0);
        fetcher.record_download(1, 500_000, 1_000, 4.0);
        fetcher.record_download(2, 500_000, 1_000, 4.0);

        let level = fetcher.buffer_level_s();
        assert!(
            (level - 12.0).abs() < f64::EPSILON,
            "3 × 4 s segments = 12 s, got {level}"
        );
        assert_eq!(fetcher.buffered_count(), 3);
    }

    #[test]
    fn test_segment_fetcher_pop() {
        let ctrl = AbrController::new(make_variants()).expect("should succeed in test");
        let mut fetcher = SegmentFetcher::new(ctrl, 6.0);

        fetcher.record_download(0, 750_000, 800, 6.0);
        fetcher.record_download(1, 750_000, 800, 6.0);

        assert_eq!(fetcher.buffered_count(), 2);

        let seg = fetcher.pop_segment().expect("should return a segment");
        assert_eq!(seg.sequence, 0);
        assert_eq!(fetcher.buffered_count(), 1);

        let level = fetcher.buffer_level_s();
        assert!(
            (level - 6.0).abs() < f64::EPSILON,
            "after pop, 1 × 6 s segment remains, got {level}"
        );
    }
}
