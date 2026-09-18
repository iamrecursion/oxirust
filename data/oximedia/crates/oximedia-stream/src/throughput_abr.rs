//! Standalone throughput-based ABR (Adaptive Bitrate) algorithm.
//!
//! This module provides [`ThroughputAbr`], a self-contained bitrate selector that
//! maintains a sliding window of segment download measurements and selects the
//! highest sustainable rendition bitrate using the harmonic mean of measured
//! throughput multiplied by a safety factor.
//!
//! Unlike the [`crate::adaptive_pipeline::AdaptivePipeline`] which orchestrates a
//! full ABR state machine with buffer management and cooldowns, [`ThroughputAbr`]
//! is a *pure selector* — it answers the question "given what we have measured,
//! which bitrate should we pick?" without internal cooldown logic or buffer state.
//! This makes it composable and easy to test independently.
//!
//! # Example
//!
//! ```
//! use oximedia_stream::throughput_abr::{ThroughputAbr, ThroughputMeasurement};
//!
//! let mut abr = ThroughputAbr::new(5, 0.9);
//! abr.add_measurement(ThroughputMeasurement::new(1_000_000, 1000)); // 8 000 kbps
//! abr.add_measurement(ThroughputMeasurement::new(500_000, 500));    // 8 000 kbps
//!
//! let bitrates = [500_u32, 1_000, 2_500, 5_000, 8_000];
//! let selected = abr.select_bitrate(&bitrates);
//! assert!(selected >= 5_000, "should select 5000 or 8000 kbps tier");
//! ```

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// ThroughputMeasurement
// ---------------------------------------------------------------------------

/// Records the outcome of a single segment download.
#[derive(Debug, Clone)]
pub struct ThroughputMeasurement {
    /// Number of bytes transferred (payload only, excluding protocol overhead).
    pub bytes: u64,
    /// Download duration in milliseconds.  Must be > 0 for a meaningful result.
    pub duration_ms: u64,
}

impl ThroughputMeasurement {
    /// Create a new measurement.
    #[must_use]
    pub fn new(bytes: u64, duration_ms: u64) -> Self {
        Self { bytes, duration_ms }
    }

    /// Compute the download throughput in **kbps** (kilobits per second).
    ///
    /// Returns `0.0` when `duration_ms` is zero to avoid division by zero.
    #[must_use]
    pub fn throughput_kbps(&self) -> f64 {
        if self.duration_ms == 0 {
            return 0.0;
        }
        // bytes → bits → kilobits; ms → seconds
        (self.bytes as f64 * 8.0) / (self.duration_ms as f64)
        // = kbps because (bits / ms) == (kbits / s)
    }
}

// ---------------------------------------------------------------------------
// ThroughputAbr
// ---------------------------------------------------------------------------

/// Throughput-based adaptive bitrate selector.
///
/// Maintains a fixed-size sliding window of [`ThroughputMeasurement`]s and
/// selects the highest rendition bitrate that is sustainable given the harmonic
/// mean of measured throughput multiplied by `self.safety_factor`.
#[derive(Debug, Clone)]
pub struct ThroughputAbr {
    /// Maximum number of measurements kept in the sliding window.
    pub window_size: usize,
    /// Safety multiplier applied to the harmonic mean before bitrate selection.
    ///
    /// Clamped to `[0.01, 1.0]` at construction time.
    pub safety_factor: f32,
    /// Sliding window of recent measurements, oldest first.
    measurements: VecDeque<ThroughputMeasurement>,
}

impl Default for ThroughputAbr {
    /// Create an ABR selector with `window_size = 10` and `safety_factor = 0.9`.
    fn default() -> Self {
        Self::new(10, 0.9)
    }
}

impl ThroughputAbr {
    /// Create a new [`ThroughputAbr`].
    ///
    /// - `window_size`: number of measurements retained (clamped to ≥ 1).
    /// - `safety_factor`: fraction of measured throughput used for selection (clamped to `[0.01, 1.0]`).
    #[must_use]
    pub fn new(window_size: usize, safety_factor: f32) -> Self {
        Self {
            window_size: window_size.max(1),
            safety_factor: safety_factor.clamp(0.01, 1.0),
            measurements: VecDeque::new(),
        }
    }

    /// Add a new measurement.  If the window is full, the oldest entry is evicted.
    pub fn add_measurement(&mut self, m: ThroughputMeasurement) {
        if self.measurements.len() >= self.window_size {
            self.measurements.pop_front();
        }
        self.measurements.push_back(m);
    }

    /// Compute the harmonic mean of throughput (in kbps) across all measurements
    /// currently in the window.
    ///
    /// Returns `0.0` when:
    /// - the window is empty, or
    /// - any measurement has `duration_ms == 0` (throughput = 0.0 makes the
    ///   harmonic mean undefined / 0).
    #[must_use]
    pub fn harmonic_mean_kbps(&self) -> f64 {
        if self.measurements.is_empty() {
            return 0.0;
        }
        let n = self.measurements.len() as f64;
        let sum_reciprocals: f64 = self
            .measurements
            .iter()
            .map(|m| {
                let kbps = m.throughput_kbps();
                if kbps > 0.0 {
                    1.0 / kbps
                } else {
                    f64::INFINITY
                }
            })
            .sum();

        if sum_reciprocals.is_infinite() || sum_reciprocals <= 0.0 {
            return 0.0;
        }
        n / sum_reciprocals
    }

    /// Compute the sustainable throughput estimate: `harmonic_mean * safety_factor`.
    ///
    /// Returns `0.0` when the window is empty.
    #[must_use]
    pub fn sustainable_kbps(&self) -> f64 {
        self.harmonic_mean_kbps() * (self.safety_factor as f64)
    }

    /// Select the highest bitrate from `available_bitrates` that is at or below
    /// the sustainable throughput estimate.
    ///
    /// # Algorithm
    ///
    /// 1. Compute `sustainable = harmonic_mean_kbps() * safety_factor`.
    /// 2. If `sustainable == 0.0` (empty window), return the lowest available
    ///    bitrate (or `0` if the slice is empty).
    /// 3. Otherwise pick the highest `b ∈ available_bitrates` where
    ///    `b as f64 <= sustainable`.
    /// 4. If *no* bitrate fits, return the lowest available bitrate.
    ///
    /// `available_bitrates` does not need to be sorted.
    #[must_use]
    pub fn select_bitrate(&self, available_bitrates: &[u32]) -> u32 {
        if available_bitrates.is_empty() {
            return 0;
        }

        let min_bitrate = available_bitrates.iter().copied().min().unwrap_or(0);

        let sustainable = self.sustainable_kbps();
        if sustainable <= 0.0 {
            return min_bitrate;
        }

        // Find the highest bitrate that fits within sustainable bandwidth.
        let best = available_bitrates
            .iter()
            .copied()
            .filter(|&b| (b as f64) <= sustainable)
            .max();

        best.unwrap_or(min_bitrate)
    }

    /// Number of measurements currently in the window.
    #[must_use]
    pub fn measurement_count(&self) -> usize {
        self.measurements.len()
    }

    /// Clear all measurements from the window.
    pub fn clear(&mut self) {
        self.measurements.clear();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: fill ABR with `count` uniform measurements.
    fn fill_uniform(abr: &mut ThroughputAbr, kbps: f64, count: usize) {
        // kbps = bytes*8/ms → bytes = kbps*ms/8; use ms=1000
        let bytes = (kbps * 1000.0 / 8.0) as u64;
        for _ in 0..count {
            abr.add_measurement(ThroughputMeasurement::new(bytes, 1000));
        }
    }

    // 1. ThroughputMeasurement::throughput_kbps basic computation.
    //    1 MB (1_048_576 bytes) transferred in 1000 ms ≈ 8388.6 kbps.
    #[test]
    fn test_measurement_throughput_kbps_basic() {
        let m = ThroughputMeasurement::new(1_000_000, 1000);
        let expected = 8000.0; // 1_000_000 * 8 / 1000 = 8000 kbps
        assert!(
            (m.throughput_kbps() - expected).abs() < 1.0,
            "got {}",
            m.throughput_kbps()
        );
    }

    // 2. throughput_kbps returns 0.0 when duration_ms = 0.
    #[test]
    fn test_measurement_zero_duration_returns_zero() {
        let m = ThroughputMeasurement::new(1_000_000, 0);
        assert_eq!(m.throughput_kbps(), 0.0);
    }

    // 3. ThroughputAbr::new clamps window_size to >= 1.
    #[test]
    fn test_new_clamps_window_size_to_1() {
        let abr = ThroughputAbr::new(0, 0.9);
        assert_eq!(abr.window_size, 1);
    }

    // 4. Default has window_size=10 and safety_factor=0.9.
    #[test]
    fn test_default_values() {
        let abr = ThroughputAbr::default();
        assert_eq!(abr.window_size, 10);
        assert!((abr.safety_factor - 0.9).abs() < 1e-6);
    }

    // 5. add_measurement stores measurement.
    #[test]
    fn test_add_measurement_stores() {
        let mut abr = ThroughputAbr::new(5, 0.9);
        abr.add_measurement(ThroughputMeasurement::new(125_000, 1000));
        assert_eq!(abr.measurement_count(), 1);
    }

    // 6. add_measurement evicts oldest when window full.
    #[test]
    fn test_add_measurement_evicts_oldest() {
        let mut abr = ThroughputAbr::new(3, 0.9);
        for i in 1..=5u64 {
            abr.add_measurement(ThroughputMeasurement::new(i * 1000, 1000));
        }
        assert_eq!(abr.measurement_count(), 3);
    }

    // 7. measurement_count increases correctly up to window_size.
    #[test]
    fn test_measurement_count_increases() {
        let mut abr = ThroughputAbr::new(5, 0.9);
        for i in 0..5 {
            abr.add_measurement(ThroughputMeasurement::new(100_000, 1000));
            assert_eq!(abr.measurement_count(), i + 1);
        }
        // Should not exceed window_size.
        abr.add_measurement(ThroughputMeasurement::new(100_000, 1000));
        assert_eq!(abr.measurement_count(), 5);
    }

    // 8. harmonic_mean_kbps returns 0 when empty.
    #[test]
    fn test_harmonic_mean_empty() {
        let abr = ThroughputAbr::new(5, 0.9);
        assert_eq!(abr.harmonic_mean_kbps(), 0.0);
    }

    // 9. harmonic_mean_kbps of uniform measurements equals that value.
    #[test]
    fn test_harmonic_mean_uniform() {
        let mut abr = ThroughputAbr::new(10, 1.0);
        fill_uniform(&mut abr, 4000.0, 5);
        let hm = abr.harmonic_mean_kbps();
        assert!(
            (hm - 4000.0).abs() < 1.0,
            "uniform harmonic mean should be ~4000, got {hm}"
        );
    }

    // 10. harmonic_mean_kbps is biased toward lower values.
    //     HM(100, 10000) = 2 / (1/100 + 1/10000) ≈ 198.
    #[test]
    fn test_harmonic_mean_biased_low() {
        let mut abr = ThroughputAbr::new(5, 1.0);
        // 100 kbps: 100*1000/8 = 12500 bytes in 1000ms
        abr.add_measurement(ThroughputMeasurement::new(12_500, 1000));
        // 10000 kbps: 10000*1000/8 = 1_250_000 bytes in 1000ms
        abr.add_measurement(ThroughputMeasurement::new(1_250_000, 1000));
        let hm = abr.harmonic_mean_kbps();
        assert!(
            hm < 250.0,
            "harmonic mean of 100 and 10000 should be ~198, got {hm}"
        );
        assert!(hm > 150.0, "harmonic mean should be > 150, got {hm}");
    }

    // 11. select_bitrate returns 0 for empty available_bitrates.
    #[test]
    fn test_select_bitrate_empty_list() {
        let mut abr = ThroughputAbr::new(5, 0.9);
        fill_uniform(&mut abr, 5000.0, 3);
        assert_eq!(abr.select_bitrate(&[]), 0);
    }

    // 12. select_bitrate returns lowest bitrate when throughput too low for anything.
    #[test]
    fn test_select_bitrate_returns_lowest_when_too_low() {
        let mut abr = ThroughputAbr::new(5, 0.9);
        fill_uniform(&mut abr, 100.0, 5); // only 100 kbps
        let bitrates = [500_u32, 1000, 2500, 5000];
        // 100 * 0.9 = 90 kbps — below the lowest bitrate (500).
        let selected = abr.select_bitrate(&bitrates);
        assert_eq!(
            selected, 500,
            "should fall back to lowest when nothing fits"
        );
    }

    // 13. select_bitrate picks the highest sustainable bitrate.
    #[test]
    fn test_select_bitrate_picks_highest_sustainable() {
        let mut abr = ThroughputAbr::new(5, 1.0); // safety_factor=1.0 for simplicity
        fill_uniform(&mut abr, 2000.0, 5); // 2000 kbps sustainable
                                           // Tiers: 500, 1000, 1500, 2000, 3000 — should pick 2000.
        let bitrates = [500_u32, 1000, 1500, 2000, 3000];
        let selected = abr.select_bitrate(&bitrates);
        assert_eq!(selected, 2000);
    }

    // 14. safety_factor=0.9 correctly reduces the sustainable estimate.
    #[test]
    fn test_select_bitrate_respects_safety_factor() {
        let mut abr = ThroughputAbr::new(5, 0.9);
        fill_uniform(&mut abr, 2000.0, 5); // 2000 kbps measured
                                           // 2000 * 0.9 = 1800 kbps sustainable → should pick 1500, not 2000.
        let bitrates = [500_u32, 1000, 1500, 2000, 3000];
        let selected = abr.select_bitrate(&bitrates);
        assert_eq!(
            selected, 1500,
            "safety factor should prevent selecting 2000 kbps"
        );
    }

    // 15. clear empties the measurement window.
    #[test]
    fn test_clear_empties_window() {
        let mut abr = ThroughputAbr::new(5, 0.9);
        fill_uniform(&mut abr, 1000.0, 3);
        abr.clear();
        assert_eq!(abr.measurement_count(), 0);
        assert_eq!(abr.harmonic_mean_kbps(), 0.0);
    }

    // 16. sustainable_kbps returns 0.0 when window is empty.
    #[test]
    fn test_sustainable_kbps_empty() {
        let abr = ThroughputAbr::new(5, 0.9);
        assert_eq!(abr.sustainable_kbps(), 0.0);
    }

    // 17. sustainable_kbps = harmonic_mean * safety_factor.
    #[test]
    fn test_sustainable_kbps_equals_hm_times_factor() {
        let mut abr = ThroughputAbr::new(10, 0.85);
        fill_uniform(&mut abr, 3000.0, 5);
        let hm = abr.harmonic_mean_kbps();
        let sus = abr.sustainable_kbps();
        assert!((sus - hm * 0.85).abs() < 1.0, "sustainable={sus} hm={hm}");
    }

    // 18. select_bitrate handles unsorted available_bitrates correctly.
    #[test]
    fn test_select_bitrate_unsorted_input() {
        let mut abr = ThroughputAbr::new(5, 1.0);
        fill_uniform(&mut abr, 2000.0, 5);
        // Unsorted input — highest below 2000 is 1500.
        let bitrates = [3000_u32, 1500, 500, 2000, 1000];
        let selected = abr.select_bitrate(&bitrates);
        assert_eq!(selected, 2000, "should find 2000 even in unsorted slice");
    }

    // 19. Sliding window: old measurements evicted, new ones affect selection.
    #[test]
    fn test_sliding_window_eviction() {
        let mut abr = ThroughputAbr::new(3, 1.0);
        // Fill with low throughput.
        fill_uniform(&mut abr, 100.0, 3);
        let low = abr.harmonic_mean_kbps();
        assert!(low < 200.0, "low throughput window: {low}");

        // Add high-throughput measurements to push out the old ones.
        fill_uniform(&mut abr, 10_000.0, 3);
        let high = abr.harmonic_mean_kbps();
        assert!(
            high > 5_000.0,
            "after eviction, high throughput should dominate: {high}"
        );
    }
}
