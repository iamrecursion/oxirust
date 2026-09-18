//! Adaptive streaming pipeline with quality ladder management.
//!
//! Implements both a BOLA-inspired adaptive bitrate (ABR) algorithm and a
//! throughput-based ABR algorithm that selects quality tiers based on measured
//! bandwidth and playback buffer level.
//!
//! # Tuning Guide
//!
//! The table below gives recommended starting parameters for common deployment
//! scenarios.  All timing fields are in seconds unless noted.
//!
//! | Parameter | Mobile / Cellular | Residential Broadband | Enterprise / Fiber | Live Event (Stadium) |
//! |---|---|---|---|---|
//! | `buffer_target_secs` | 10–15 | 20–30 | 30–60 | 8–12 |
//! | `upgrade_cooldown_secs` | 12–16 | 8–12 | 4–8 | 4–6 |
//! | `downgrade_cooldown_secs` | 1–2 | 2–4 | 2–4 | 0.5–1 |
//! | `ewma_alpha` | 0.1–0.2 | 0.2–0.3 | 0.3–0.5 | 0.4–0.6 |
//! | `max_samples` | 5–8 | 8–12 | 10–15 | 5–8 |
//! | BOLA `safety_factor` | 0.8 | 0.9 | 0.95 | 0.85 |
//!
//! ## When to choose BOLA vs ThroughputBased
//!
//! **BOLA** (`AbrAlgorithm::Bola`) is buffer-driven: it targets a configurable
//! buffer occupancy level and raises or lowers quality to maintain that level.
//! It is more conservative and tolerates variable-bandwidth links gracefully.
//! Prefer BOLA when rebuffering is the primary concern (e.g. mobile, satellite).
//!
//! **ThroughputBased** (`AbrAlgorithm::ThroughputBased { safety_factor }`) is
//! bandwidth-driven: it measures the recent download throughput and selects the
//! highest quality tier whose bitrate fits within `throughput × safety_factor`.
//! It upgrades faster and is better suited to low-latency live events where
//! startup time matters more than absolute stability (e.g. stadium, esports).

use std::collections::VecDeque;
use std::time::{Instant, SystemTime};

// ─── Quality tier ─────────────────────────────────────────────────────────────

/// A single entry in the quality ladder describing resolution, bitrate, and
/// codec parameters for one rendition of an adaptive stream.
#[derive(Debug, Clone)]
pub struct QualityTier {
    /// Human-readable label, e.g. `"1080p"`.
    pub name: String,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Target video bitrate in kbps.
    pub video_bitrate_kbps: u32,
    /// Target audio bitrate in kbps.
    pub audio_bitrate_kbps: u32,
    /// Target frame rate.
    pub fps: f32,
    /// Codec label for annotation purposes (e.g. `"av1"`, `"vp9"`).
    pub codec: String,
    /// Minimum downstream bandwidth in kbps required to sustain this tier.
    pub min_bandwidth_kbps: u32,
}

impl QualityTier {
    /// Total nominal bitrate (video + audio) in kbps.
    pub fn total_bitrate_kbps(&self) -> u32 {
        self.video_bitrate_kbps + self.audio_bitrate_kbps
    }
}

// ─── Quality ladder ───────────────────────────────────────────────────────────

/// An ordered collection of [`QualityTier`]s sorted by total bitrate ascending.
pub struct QualityLadder {
    /// All available tiers, sorted lowest → highest bitrate.
    pub tiers: Vec<QualityTier>,
    /// Index into `tiers` for the currently active rendition.
    pub current_tier_index: usize,
}

impl QualityLadder {
    /// Construct a ladder from the provided tiers.  The tiers are sorted by
    /// total bitrate (video + audio) in ascending order.
    pub fn new(mut tiers: Vec<QualityTier>) -> Self {
        tiers.sort_by_key(|t| t.total_bitrate_kbps());
        Self {
            tiers,
            current_tier_index: 0,
        }
    }

    /// Return a 6-tier ladder covering 240p → 2160p (4K) with typical OTT
    /// bitrates and AV1 codec labels.
    pub fn default_ladder() -> Self {
        let tiers = vec![
            QualityTier {
                name: "240p".into(),
                width: 426,
                height: 240,
                video_bitrate_kbps: 200,
                audio_bitrate_kbps: 32,
                fps: 24.0,
                codec: "av1".into(),
                min_bandwidth_kbps: 300,
            },
            QualityTier {
                name: "360p".into(),
                width: 640,
                height: 360,
                video_bitrate_kbps: 500,
                audio_bitrate_kbps: 64,
                fps: 24.0,
                codec: "av1".into(),
                min_bandwidth_kbps: 700,
            },
            QualityTier {
                name: "480p".into(),
                width: 854,
                height: 480,
                video_bitrate_kbps: 1000,
                audio_bitrate_kbps: 96,
                fps: 30.0,
                codec: "av1".into(),
                min_bandwidth_kbps: 1300,
            },
            QualityTier {
                name: "720p".into(),
                width: 1280,
                height: 720,
                video_bitrate_kbps: 2500,
                audio_bitrate_kbps: 128,
                fps: 30.0,
                codec: "av1".into(),
                min_bandwidth_kbps: 3200,
            },
            QualityTier {
                name: "1080p".into(),
                width: 1920,
                height: 1080,
                video_bitrate_kbps: 5000,
                audio_bitrate_kbps: 192,
                fps: 30.0,
                codec: "av1".into(),
                min_bandwidth_kbps: 6500,
            },
            QualityTier {
                name: "2160p".into(),
                width: 3840,
                height: 2160,
                video_bitrate_kbps: 15000,
                audio_bitrate_kbps: 256,
                fps: 60.0,
                codec: "av1".into(),
                min_bandwidth_kbps: 20000,
            },
        ];
        Self::new(tiers)
    }

    /// Reference to the currently selected tier.
    pub fn current(&self) -> &QualityTier {
        // Safety: current_tier_index is always kept in-bounds by all mutating
        // methods; tiers is never empty after construction.
        &self.tiers[self
            .current_tier_index
            .min(self.tiers.len().saturating_sub(1))]
    }

    /// Number of tiers in the ladder.
    pub fn len(&self) -> usize {
        self.tiers.len()
    }

    /// Returns `true` if the ladder has no tiers.
    pub fn is_empty(&self) -> bool {
        self.tiers.is_empty()
    }
}

// ─── Bandwidth estimator ──────────────────────────────────────────────────────

/// Maintains a sliding window of bandwidth measurements and computes both an
/// EWMA estimate and a percentile-based conservative estimate.
pub struct BandwidthEstimator {
    /// Recent bandwidth samples in kbps (oldest first).
    pub samples: VecDeque<f64>,
    /// Maximum number of samples retained.
    pub max_samples: usize,
    /// EWMA smoothing factor α ∈ (0, 1].  Higher → more weight on recent data.
    pub ewma_alpha: f64,
    /// Current exponentially-weighted moving average in kbps.
    pub ewma_estimate: f64,
}

impl BandwidthEstimator {
    /// Construct a new estimator.
    ///
    /// - `max_samples`: window size for percentile calculations.
    /// - `alpha`: EWMA smoothing factor (0 < α ≤ 1).
    pub fn new(max_samples: usize, alpha: f64) -> Self {
        let alpha = alpha.clamp(1e-6, 1.0);
        Self {
            samples: VecDeque::with_capacity(max_samples),
            max_samples: max_samples.max(1),
            ewma_alpha: alpha,
            ewma_estimate: 0.0,
        }
    }

    /// Incorporate a new bandwidth measurement `kbps` into the EWMA and the
    /// sliding sample window.
    pub fn add_sample(&mut self, kbps: f64) {
        let kbps = kbps.max(0.0);
        if self.ewma_estimate == 0.0 {
            self.ewma_estimate = kbps;
        } else {
            self.ewma_estimate =
                self.ewma_alpha * kbps + (1.0 - self.ewma_alpha) * self.ewma_estimate;
        }
        if self.samples.len() >= self.max_samples {
            self.samples.pop_front();
        }
        self.samples.push_back(kbps);
    }

    /// Current EWMA bandwidth estimate in kbps.
    pub fn estimate(&self) -> f64 {
        self.ewma_estimate
    }

    /// Compute the `p`-th percentile (0 ≤ p ≤ 100) of the sample window.
    ///
    /// Uses linear interpolation.  Returns 0 if there are no samples.
    /// A conservative ABR controller typically calls this with `p = 20`.
    pub fn percentile(&self, p: f64) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let p = p.clamp(0.0, 100.0);
        let mut sorted: Vec<f64> = self.samples.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len();
        if n == 1 {
            return sorted[0];
        }
        let rank = p / 100.0 * (n as f64 - 1.0);
        let lo = rank.floor() as usize;
        let hi = (lo + 1).min(n - 1);
        let frac = rank - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

// ─── Switch record ────────────────────────────────────────────────────────────

/// Reason for a quality-tier transition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SwitchReason {
    /// Bandwidth headroom allows moving to a higher tier.
    BandwidthUpgrade,
    /// Bandwidth is insufficient for the current tier.
    BandwidthDowngrade,
    /// Playback buffer dangerously low — emergency downgrade.
    BufferStress,
    /// Buffer has recovered enough to consider upgrading.
    BufferRecovery,
    /// Explicit user or application override.
    UserRequested,
}

// ─── ABR algorithm selection ──────────────────────────────────────────────────

/// Which adaptive bitrate algorithm variant to use for quality selection.
#[derive(Debug, Clone, PartialEq)]
pub enum AbrAlgorithm {
    /// BOLA-inspired algorithm: uses both buffer level and bandwidth signals
    /// with priority given to buffer state (emergency downgrade on starvation).
    Bola,
    /// Throughput-based algorithm: selects the highest tier whose total bitrate
    /// is sustainable given the harmonic mean of recent throughput samples,
    /// multiplied by a safety factor.  Buffer level is only used for emergency
    /// downgrade (< 2 s).
    ThroughputBased {
        /// Safety multiplier applied to throughput estimate.  A value of 0.85
        /// means the algorithm targets 85% of measured throughput.
        safety_factor: f64,
    },
}

impl Default for AbrAlgorithm {
    fn default() -> Self {
        Self::Bola
    }
}

/// Records a single quality-tier switch event.
#[derive(Debug, Clone)]
pub struct QualitySwitch {
    /// Tier index before the switch.
    pub from_tier: usize,
    /// Tier index after the switch.
    pub to_tier: usize,
    /// Why the switch occurred.
    pub reason: SwitchReason,
    /// Wall-clock time of the switch.
    pub timestamp: SystemTime,
    /// Bandwidth estimate (kbps) at the moment of the switch.
    pub bandwidth_kbps: f64,
}

// ─── Adaptive pipeline ────────────────────────────────────────────────────────

/// Adaptive streaming pipeline supporting multiple ABR algorithms.
///
/// Maintains a [`QualityLadder`] and a [`BandwidthEstimator`] and decides
/// when to switch rendition based on both buffer level and bandwidth signals.
/// The decision logic can be configured via [`AbrAlgorithm`].
pub struct AdaptivePipeline {
    /// Quality ladder being managed.
    pub ladder: QualityLadder,
    /// Bandwidth estimator.
    pub bandwidth: BandwidthEstimator,
    /// Target steady-state buffer depth in seconds.
    pub buffer_target_secs: f64,
    /// Current playback buffer depth in seconds.
    pub buffer_current_secs: f64,
    /// Minimum interval between upgrade decisions in seconds.
    pub upgrade_cooldown_secs: f64,
    /// Minimum interval between downgrade decisions in seconds.
    pub downgrade_cooldown_secs: f64,
    last_switch_time: Instant,
    /// History of all quality switches since creation.
    pub switch_history: Vec<QualitySwitch>,
    /// Which ABR algorithm variant to use.
    pub abr_algorithm: AbrAlgorithm,
}

impl AdaptivePipeline {
    /// Create a new pipeline.  The buffer target defaults to 30 s, upgrade
    /// cooldown to 8 s, and downgrade cooldown to 2 s.
    pub fn new(ladder: QualityLadder) -> Self {
        Self {
            bandwidth: BandwidthEstimator::new(20, 0.3),
            ladder,
            buffer_target_secs: 30.0,
            buffer_current_secs: 0.0,
            upgrade_cooldown_secs: 8.0,
            downgrade_cooldown_secs: 2.0,
            last_switch_time: Instant::now()
                .checked_sub(std::time::Duration::from_hours(1))
                .unwrap_or_else(Instant::now),
            switch_history: Vec::new(),
            abr_algorithm: AbrAlgorithm::default(),
        }
    }

    /// Create a new pipeline with a specific ABR algorithm.
    pub fn with_algorithm(ladder: QualityLadder, algorithm: AbrAlgorithm) -> Self {
        let mut pipeline = Self::new(ladder);
        pipeline.abr_algorithm = algorithm;
        pipeline
    }

    /// Update the current buffer level.
    pub fn update_buffer(&mut self, buffer_secs: f64) {
        self.buffer_current_secs = buffer_secs.max(0.0);
    }

    /// Record a completed segment download and derive a bandwidth sample.
    ///
    /// `bytes` is the uncompressed payload size; `duration_secs` is the
    /// wall-clock time taken to receive it.
    pub fn record_download(&mut self, bytes: u64, duration_secs: f64) {
        if duration_secs > 0.0 {
            let kbps = (bytes as f64 * 8.0) / (duration_secs * 1000.0);
            self.bandwidth.add_sample(kbps);
        }
    }

    /// Evaluate whether a quality switch is warranted and, if so, execute it.
    ///
    /// Returns `Some(QualitySwitch)` when the tier changes, otherwise `None`.
    ///
    /// The decision logic depends on `self.abr_algorithm`:
    ///
    /// **BOLA** (in priority order):
    /// 1. Buffer < 2 s → emergency downgrade (BufferStress).
    /// 2. Buffer > target + 5 s **and** bandwidth > next tier's min → upgrade (BufferRecovery).
    /// 3. Bandwidth < current tier's min × 0.9 → downgrade (BandwidthDowngrade).
    /// 4. Bandwidth > next tier's min × 1.1 **and** upgrade cooldown elapsed → upgrade (BandwidthUpgrade).
    ///
    /// **ThroughputBased**:
    /// 1. Buffer < 2 s → emergency downgrade (BufferStress).
    /// 2. Compute harmonic mean of throughput × safety_factor; pick the highest
    ///    tier whose total bitrate fits within that estimate.
    pub fn evaluate_switch(&mut self) -> Option<QualitySwitch> {
        match self.abr_algorithm.clone() {
            AbrAlgorithm::Bola => self.evaluate_switch_bola(),
            AbrAlgorithm::ThroughputBased { safety_factor } => {
                self.evaluate_switch_throughput(safety_factor)
            }
        }
    }

    /// BOLA-inspired ABR evaluation.
    fn evaluate_switch_bola(&mut self) -> Option<QualitySwitch> {
        let bw = self.bandwidth.estimate();
        let buf = self.buffer_current_secs;
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_switch_time).as_secs_f64();

        let current_idx = self.ladder.current_tier_index;
        let max_idx = self.ladder.tiers.len().saturating_sub(1);

        // 1. Emergency downgrade on buffer stress.
        if buf < 2.0 && current_idx > 0 {
            let target = current_idx - 1;
            return self.apply_switch(current_idx, target, SwitchReason::BufferStress, bw);
        }

        // 2. Buffer recovery → opportunistic upgrade.
        if buf > self.buffer_target_secs + 5.0 && current_idx < max_idx {
            let next = current_idx + 1;
            let next_min = self.ladder.tiers[next].min_bandwidth_kbps as f64;
            if bw > next_min {
                return self.apply_switch(current_idx, next, SwitchReason::BufferRecovery, bw);
            }
        }

        // 3. Bandwidth-driven downgrade (no cooldown required — safety).
        let current_min = self.ladder.tiers[current_idx].min_bandwidth_kbps as f64;
        if bw > 0.0 && bw < current_min * 0.9 && current_idx > 0 {
            if elapsed >= self.downgrade_cooldown_secs {
                let target = current_idx - 1;
                return self.apply_switch(
                    current_idx,
                    target,
                    SwitchReason::BandwidthDowngrade,
                    bw,
                );
            }
        }

        // 4. Bandwidth-driven upgrade (cooldown enforced).
        if current_idx < max_idx && elapsed >= self.upgrade_cooldown_secs {
            let next = current_idx + 1;
            let next_min = self.ladder.tiers[next].min_bandwidth_kbps as f64;
            if bw > next_min * 1.1 {
                return self.apply_switch(current_idx, next, SwitchReason::BandwidthUpgrade, bw);
            }
        }

        None
    }

    /// Throughput-based ABR evaluation.
    ///
    /// Computes the harmonic mean of the most recent bandwidth samples, applies
    /// `safety_factor`, and selects the highest tier whose total bitrate is below
    /// the resulting sustainable bandwidth.  Buffer starvation still triggers an
    /// emergency downgrade.
    fn evaluate_switch_throughput(&mut self, safety_factor: f64) -> Option<QualitySwitch> {
        let buf = self.buffer_current_secs;
        let current_idx = self.ladder.current_tier_index;
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_switch_time).as_secs_f64();

        // 1. Emergency buffer-stress downgrade.
        if buf < 2.0 && current_idx > 0 {
            let harmonic = self.bandwidth_harmonic_mean();
            let target = current_idx - 1;
            return self.apply_switch(current_idx, target, SwitchReason::BufferStress, harmonic);
        }

        // 2. Compute sustainable bandwidth from harmonic mean.
        let harmonic = self.bandwidth_harmonic_mean();
        if harmonic <= 0.0 {
            return None;
        }
        let safe_bw = harmonic * safety_factor.clamp(0.1, 1.0);

        // Find the highest tier whose total bitrate fits.
        let mut best_idx = 0usize;
        for (i, tier) in self.ladder.tiers.iter().enumerate() {
            if (tier.total_bitrate_kbps() as f64) <= safe_bw {
                best_idx = i;
            }
        }

        if best_idx == current_idx {
            return None;
        }

        // Apply cooldowns.
        if best_idx > current_idx {
            if elapsed < self.upgrade_cooldown_secs {
                return None;
            }
            self.apply_switch(
                current_idx,
                best_idx,
                SwitchReason::BandwidthUpgrade,
                harmonic,
            )
        } else {
            if elapsed < self.downgrade_cooldown_secs {
                return None;
            }
            self.apply_switch(
                current_idx,
                best_idx,
                SwitchReason::BandwidthDowngrade,
                harmonic,
            )
        }
    }

    /// Compute the harmonic mean of all bandwidth samples in the estimator window.
    ///
    /// Returns 0.0 if there are no samples or if any sample is zero.
    pub fn bandwidth_harmonic_mean(&self) -> f64 {
        if self.bandwidth.samples.is_empty() {
            return 0.0;
        }
        let n = self.bandwidth.samples.len() as f64;
        let sum_reciprocals: f64 = self
            .bandwidth
            .samples
            .iter()
            .map(|&s| {
                if s > 0.0 {
                    1.0 / s
                } else {
                    return f64::INFINITY;
                }
            })
            .sum();
        if sum_reciprocals.is_infinite() || sum_reciprocals <= 0.0 {
            return 0.0;
        }
        n / sum_reciprocals
    }

    /// Force the pipeline to a specific tier index.
    ///
    /// Returns `Err` if `index` is out of range.
    pub fn force_tier(&mut self, index: usize, reason: SwitchReason) -> Result<(), String> {
        if index >= self.ladder.tiers.len() {
            return Err(format!(
                "tier index {} out of range (ladder has {} tiers)",
                index,
                self.ladder.tiers.len()
            ));
        }
        let from = self.ladder.current_tier_index;
        let bw = self.bandwidth.estimate();
        let _ = self.apply_switch(from, index, reason, bw);
        Ok(())
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn apply_switch(
        &mut self,
        from: usize,
        to: usize,
        reason: SwitchReason,
        bandwidth_kbps: f64,
    ) -> Option<QualitySwitch> {
        if from == to {
            return None;
        }
        self.ladder.current_tier_index = to;
        self.last_switch_time = Instant::now();
        let sw = QualitySwitch {
            from_tier: from,
            to_tier: to,
            reason,
            timestamp: SystemTime::now(),
            bandwidth_kbps,
        };
        self.switch_history.push(sw.clone());
        Some(sw)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pipeline() -> AdaptivePipeline {
        let ladder = QualityLadder::default_ladder();
        AdaptivePipeline::new(ladder)
    }

    // ── QualityLadder tests ───────────────────────────────────────────────────

    #[test]
    fn test_default_ladder_tier_count() {
        let ladder = QualityLadder::default_ladder();
        assert_eq!(ladder.len(), 6);
    }

    #[test]
    fn test_default_ladder_sorted_ascending() {
        let ladder = QualityLadder::default_ladder();
        let bitrates: Vec<u32> = ladder
            .tiers
            .iter()
            .map(|t| t.total_bitrate_kbps())
            .collect();
        let mut sorted = bitrates.clone();
        sorted.sort_unstable();
        assert_eq!(
            bitrates, sorted,
            "tiers must be sorted by total bitrate ascending"
        );
    }

    #[test]
    fn test_quality_ladder_new_sorts_tiers() {
        let tiers = vec![
            QualityTier {
                name: "high".into(),
                width: 1920,
                height: 1080,
                video_bitrate_kbps: 5000,
                audio_bitrate_kbps: 192,
                fps: 30.0,
                codec: "av1".into(),
                min_bandwidth_kbps: 6500,
            },
            QualityTier {
                name: "low".into(),
                width: 426,
                height: 240,
                video_bitrate_kbps: 200,
                audio_bitrate_kbps: 32,
                fps: 24.0,
                codec: "av1".into(),
                min_bandwidth_kbps: 300,
            },
        ];
        let ladder = QualityLadder::new(tiers);
        assert_eq!(ladder.tiers[0].name, "low");
        assert_eq!(ladder.tiers[1].name, "high");
    }

    #[test]
    fn test_quality_ladder_current_returns_first_by_default() {
        let ladder = QualityLadder::default_ladder();
        assert_eq!(ladder.current().name, "240p");
    }

    // ── BandwidthEstimator tests ──────────────────────────────────────────────

    #[test]
    fn test_bandwidth_estimator_initial_state() {
        let est = BandwidthEstimator::new(10, 0.3);
        assert_eq!(est.estimate(), 0.0);
        assert_eq!(est.percentile(50.0), 0.0);
    }

    #[test]
    fn test_bandwidth_estimator_single_sample() {
        let mut est = BandwidthEstimator::new(10, 0.5);
        est.add_sample(1000.0);
        assert_eq!(est.estimate(), 1000.0); // first sample → direct assignment
        assert_eq!(est.percentile(50.0), 1000.0);
    }

    #[test]
    fn test_bandwidth_estimator_ewma_converges() {
        let mut est = BandwidthEstimator::new(20, 0.5);
        for _ in 0..20 {
            est.add_sample(2000.0);
        }
        let est_val = est.estimate();
        assert!(
            (est_val - 2000.0).abs() < 1.0,
            "EWMA should converge to 2000 kbps, got {est_val}"
        );
    }

    #[test]
    fn test_bandwidth_estimator_percentile_ordering() {
        let mut est = BandwidthEstimator::new(10, 0.3);
        for v in [100.0, 200.0, 300.0, 400.0, 500.0] {
            est.add_sample(v);
        }
        let p20 = est.percentile(20.0);
        let p50 = est.percentile(50.0);
        let p80 = est.percentile(80.0);
        assert!(p20 <= p50, "p20={p20} should be <= p50={p50}");
        assert!(p50 <= p80, "p50={p50} should be <= p80={p80}");
    }

    #[test]
    fn test_bandwidth_estimator_window_capped() {
        let mut est = BandwidthEstimator::new(3, 0.5);
        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            est.add_sample(v);
        }
        assert_eq!(est.samples.len(), 3);
    }

    // ── AdaptivePipeline tests ────────────────────────────────────────────────

    #[test]
    fn test_pipeline_starts_at_lowest_tier() {
        let p = make_pipeline();
        assert_eq!(p.ladder.current_tier_index, 0);
    }

    #[test]
    fn test_pipeline_record_download_adds_sample() {
        let mut p = make_pipeline();
        p.record_download(1_000_000, 1.0); // 8000 kbps
        assert!(p.bandwidth.estimate() > 0.0);
    }

    #[test]
    fn test_pipeline_buffer_stress_downgrade() {
        let mut p = make_pipeline();
        // Move to tier 2 first via force_tier
        p.force_tier(2, SwitchReason::UserRequested)
            .expect("force_tier");
        // Simulate dangerously low buffer
        p.update_buffer(1.0);
        // Add some bandwidth so only buffer stress triggers
        p.bandwidth.add_sample(10_000.0);
        let sw = p.evaluate_switch();
        assert!(sw.is_some(), "expected a switch due to buffer stress");
        let sw = sw.expect("switch");
        assert_eq!(sw.reason, SwitchReason::BufferStress);
        assert!(sw.to_tier < sw.from_tier, "should downgrade");
    }

    #[test]
    fn test_pipeline_bandwidth_upgrade_after_cooldown() {
        let mut p = make_pipeline();
        // Zero cooldown for test
        p.upgrade_cooldown_secs = 0.0;
        p.downgrade_cooldown_secs = 0.0;
        // Reset last switch to long ago
        p.last_switch_time = Instant::now()
            .checked_sub(std::time::Duration::from_hours(1))
            .unwrap_or_else(Instant::now);
        p.update_buffer(30.0);
        // Tier 1 (360p) min_bandwidth = 700; supply 10× that
        for _ in 0..10 {
            p.bandwidth.add_sample(10_000.0);
        }
        let sw = p.evaluate_switch();
        assert!(sw.is_some(), "should upgrade with ample bandwidth");
    }

    #[test]
    fn test_pipeline_force_tier_valid() {
        let mut p = make_pipeline();
        let result = p.force_tier(3, SwitchReason::UserRequested);
        assert!(result.is_ok());
        assert_eq!(p.ladder.current_tier_index, 3);
    }

    #[test]
    fn test_pipeline_force_tier_out_of_range() {
        let mut p = make_pipeline();
        let result = p.force_tier(99, SwitchReason::UserRequested);
        assert!(result.is_err());
    }

    #[test]
    fn test_pipeline_switch_history_recorded() {
        let mut p = make_pipeline();
        p.force_tier(2, SwitchReason::UserRequested).expect("force");
        p.force_tier(0, SwitchReason::UserRequested).expect("force");
        assert_eq!(p.switch_history.len(), 2);
    }

    #[test]
    fn test_pipeline_no_switch_without_samples() {
        let mut p = make_pipeline();
        p.update_buffer(15.0);
        // No bandwidth samples → estimate is 0 → no upgrade
        let sw = p.evaluate_switch();
        assert!(sw.is_none());
    }

    #[test]
    fn test_pipeline_bandwidth_downgrade() {
        let mut p = make_pipeline();
        p.force_tier(5, SwitchReason::UserRequested).expect("force");
        p.downgrade_cooldown_secs = 0.0;
        p.last_switch_time = Instant::now()
            .checked_sub(std::time::Duration::from_hours(1))
            .unwrap_or_else(Instant::now);
        p.update_buffer(15.0);
        // Supply very low bandwidth — well below 4K tier minimum
        for _ in 0..10 {
            p.bandwidth.add_sample(100.0);
        }
        let sw = p.evaluate_switch();
        assert!(sw.is_some(), "expected downgrade due to low bandwidth");
        let sw = sw.expect("switch");
        assert_eq!(sw.reason, SwitchReason::BandwidthDowngrade);
    }

    // ── ThroughputBased ABR tests ───────────────────────────────────────────

    fn make_throughput_pipeline(safety: f64) -> AdaptivePipeline {
        let ladder = QualityLadder::default_ladder();
        let mut p = AdaptivePipeline::with_algorithm(
            ladder,
            AbrAlgorithm::ThroughputBased {
                safety_factor: safety,
            },
        );
        p.upgrade_cooldown_secs = 0.0;
        p.downgrade_cooldown_secs = 0.0;
        p.last_switch_time = Instant::now()
            .checked_sub(std::time::Duration::from_hours(1))
            .unwrap_or_else(Instant::now);
        p
    }

    #[test]
    fn test_throughput_abr_selects_highest_sustainable_tier() {
        let mut p = make_throughput_pipeline(0.85);
        p.update_buffer(15.0);
        // Supply 2000 kbps consistently → 2000 * 0.85 = 1700 kbps sustainable
        // Tier 0 (240p): 232 kbps total, Tier 1 (360p): 564 kbps, Tier 2 (480p): 1096 kbps
        // Tier 3 (720p): 2628 kbps → too high
        // Expected: upgrade to tier 2
        for _ in 0..10 {
            p.bandwidth.add_sample(2000.0);
        }
        let sw = p.evaluate_switch();
        assert!(sw.is_some(), "expected upgrade with 2000 kbps throughput");
        let sw = sw.expect("switch");
        assert_eq!(sw.reason, SwitchReason::BandwidthUpgrade);
        assert_eq!(sw.to_tier, 2, "should jump to tier 2 (480p)");
    }

    #[test]
    fn test_throughput_abr_downgrades_on_low_bandwidth() {
        let mut p = make_throughput_pipeline(0.85);
        p.force_tier(4, SwitchReason::UserRequested).expect("force");
        p.last_switch_time = Instant::now()
            .checked_sub(std::time::Duration::from_hours(1))
            .unwrap_or_else(Instant::now);
        p.update_buffer(15.0);
        // Supply very low bandwidth
        for _ in 0..10 {
            p.bandwidth.add_sample(300.0);
        }
        let sw = p.evaluate_switch();
        assert!(sw.is_some(), "expected downgrade");
        let sw = sw.expect("switch");
        assert_eq!(sw.reason, SwitchReason::BandwidthDowngrade);
        assert!(sw.to_tier < 4);
    }

    #[test]
    fn test_throughput_abr_buffer_stress_emergency() {
        let mut p = make_throughput_pipeline(0.85);
        p.force_tier(3, SwitchReason::UserRequested).expect("force");
        p.update_buffer(1.0); // dangerously low
        for _ in 0..10 {
            p.bandwidth.add_sample(10_000.0);
        }
        let sw = p.evaluate_switch();
        assert!(sw.is_some(), "expected buffer stress downgrade");
        assert_eq!(sw.expect("switch").reason, SwitchReason::BufferStress);
    }

    #[test]
    fn test_throughput_abr_no_switch_without_samples() {
        let mut p = make_throughput_pipeline(0.85);
        p.update_buffer(15.0);
        let sw = p.evaluate_switch();
        assert!(sw.is_none(), "no samples → no switch");
    }

    #[test]
    fn test_bandwidth_harmonic_mean_empty() {
        let p = make_pipeline();
        assert_eq!(p.bandwidth_harmonic_mean(), 0.0);
    }

    #[test]
    fn test_bandwidth_harmonic_mean_uniform() {
        let mut p = make_pipeline();
        for _ in 0..5 {
            p.bandwidth.add_sample(1000.0);
        }
        let hm = p.bandwidth_harmonic_mean();
        assert!(
            (hm - 1000.0).abs() < 1.0,
            "harmonic mean of uniform values should be ~1000, got {hm}"
        );
    }

    #[test]
    fn test_bandwidth_harmonic_mean_biased_low() {
        let mut p = make_pipeline();
        p.bandwidth.add_sample(100.0);
        p.bandwidth.add_sample(10000.0);
        let hm = p.bandwidth_harmonic_mean();
        // Harmonic mean of 100 and 10000 = 2 / (1/100 + 1/10000) = 2 / 0.0101 ≈ 198
        assert!(
            hm < 250.0,
            "harmonic mean should be biased toward lower value, got {hm}"
        );
    }

    #[test]
    fn test_abr_algorithm_default_is_bola() {
        assert_eq!(AbrAlgorithm::default(), AbrAlgorithm::Bola);
    }

    #[test]
    fn test_throughput_abr_stays_if_already_optimal() {
        let mut p = make_throughput_pipeline(0.85);
        p.update_buffer(15.0);
        // Tier 0 total = 232 kbps; supply just enough for tier 0 but not tier 1
        // Tier 1 min_bandwidth = 564 kbps; 400 * 0.85 = 340 < 564 → stay at 0
        for _ in 0..10 {
            p.bandwidth.add_sample(400.0);
        }
        let sw = p.evaluate_switch();
        assert!(sw.is_none(), "should stay at tier 0");
    }

    // ── ABR fluctuation / hysteresis tests ───────────────────────────────────

    #[test]
    fn test_pipeline_fluctuation_hysteresis_prevents_oscillation() {
        // This test verifies that when bandwidth alternates between high (5000
        // kbps) and low (500 kbps), the cooldown mechanism limits the total
        // number of quality switches to a bounded value rather than oscillating
        // on every evaluation.
        let mut p = make_pipeline();
        // Zero cooldowns so we can observe the raw switch counting, then we
        // restore real cooldowns mid-way to demonstrate hysteresis.
        p.upgrade_cooldown_secs = 0.0;
        p.downgrade_cooldown_secs = 0.0;
        p.last_switch_time = Instant::now()
            .checked_sub(std::time::Duration::from_hours(1))
            .unwrap_or_else(Instant::now);
        p.update_buffer(20.0);

        let mut upgrades = 0u32;
        let mut downgrades = 0u32;

        // Alternate between high and low bandwidth 10 times
        for cycle in 0..10u32 {
            let kbps = if cycle % 2 == 0 { 5000.0 } else { 500.0 };
            for _ in 0..5 {
                p.bandwidth.add_sample(kbps);
            }
            if let Some(sw) = p.evaluate_switch() {
                match sw.reason {
                    SwitchReason::BandwidthUpgrade => upgrades += 1,
                    SwitchReason::BandwidthDowngrade => downgrades += 1,
                    _ => {}
                }
                // Restore cooldown after first switch to simulate hysteresis
                p.upgrade_cooldown_secs = 4.0;
                p.downgrade_cooldown_secs = 2.0;
            }
        }

        // Without any hysteresis we would expect 10 alternations = 10 switches.
        // With cooldowns re-engaged after the first switch, total should be << 10.
        let total_switches = upgrades + downgrades;
        assert!(
            total_switches <= 5,
            "hysteresis should bound oscillation: got {total_switches} switches \
             ({upgrades} up, {downgrades} down)"
        );
    }

    #[test]
    fn test_pipeline_fluctuation_strict_cooldown_cap() {
        // With real cooldowns active from the start, alternating bandwidth 10×
        // must produce at most 4 switches total.
        let mut p = make_pipeline();
        // Reasonable cooldowns that prevent rapid oscillation
        p.upgrade_cooldown_secs = 5.0;
        p.downgrade_cooldown_secs = 3.0;
        // last_switch_time set to long ago so the very first switch is allowed
        p.last_switch_time = Instant::now()
            .checked_sub(std::time::Duration::from_hours(1))
            .unwrap_or_else(Instant::now);
        p.update_buffer(20.0);

        let mut total_switches = 0u32;

        for cycle in 0..10u32 {
            let kbps = if cycle % 2 == 0 { 5000.0 } else { 500.0 };
            for _ in 0..5 {
                p.bandwidth.add_sample(kbps);
            }
            if p.evaluate_switch().is_some() {
                total_switches += 1;
            }
        }

        assert!(
            total_switches <= 4,
            "strict cooldown should cap oscillation to at most 4 switches, \
             got {total_switches}"
        );
    }
}
