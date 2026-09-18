//! Metering bridge for inserting level meters at arbitrary signal-path points.
//!
//! A [`MeteringBridge`] is a zero-latency tap that observes samples flowing
//! through a signal path and computes peak, RMS, and crest-factor readings.
//! Multiple bridges can be chained at different points for comprehensive
//! monitoring without affecting the audio.

use std::collections::HashMap;

/// A single metering snapshot taken from a bridge point.
#[derive(Debug, Clone, Copy)]
pub struct MeterReading {
    /// Peak level in dBFS since last reset.
    pub peak_dbfs: f32,
    /// RMS level in dBFS over the measurement window.
    pub rms_dbfs: f32,
    /// Crest factor (peak / RMS) in dB.
    pub crest_factor_db: f32,
    /// Number of samples that exceeded 0 dBFS (clipped).
    pub clip_count: u64,
    /// Total samples metered since last reset.
    pub sample_count: u64,
}

impl Default for MeterReading {
    fn default() -> Self {
        Self {
            peak_dbfs: f32::NEG_INFINITY,
            rms_dbfs: f32::NEG_INFINITY,
            crest_factor_db: 0.0,
            clip_count: 0,
            sample_count: 0,
        }
    }
}

impl MeterReading {
    /// Returns `true` if any clipping has been detected.
    pub fn has_clipped(&self) -> bool {
        self.clip_count > 0
    }

    /// Returns `true` if this reading contains no data yet.
    pub fn is_empty(&self) -> bool {
        self.sample_count == 0
    }
}

/// Internal state for one metering point.
#[derive(Debug, Clone)]
struct MeterState {
    #[allow(dead_code)]
    name: String,
    peak_linear: f32,
    sum_sq: f64,
    sample_count: u64,
    clip_count: u64,
    clip_threshold: f32,
}

impl MeterState {
    fn new(name: impl Into<String>, clip_threshold: f32) -> Self {
        Self {
            name: name.into(),
            peak_linear: 0.0,
            sum_sq: 0.0,
            sample_count: 0,
            clip_count: 0,
            clip_threshold,
        }
    }

    fn feed(&mut self, samples: &[f32]) {
        for &s in samples {
            let abs = s.abs();
            if abs > self.peak_linear {
                self.peak_linear = abs;
            }
            self.sum_sq += (s as f64) * (s as f64);
            self.sample_count += 1;
            if abs >= self.clip_threshold {
                self.clip_count += 1;
            }
        }
    }

    fn reading(&self) -> MeterReading {
        if self.sample_count == 0 {
            return MeterReading::default();
        }

        let peak_dbfs = linear_to_dbfs(self.peak_linear);
        let rms_linear = (self.sum_sq / self.sample_count as f64).sqrt() as f32;
        let rms_dbfs = linear_to_dbfs(rms_linear);
        let crest_factor_db = peak_dbfs - rms_dbfs;

        MeterReading {
            peak_dbfs,
            rms_dbfs,
            crest_factor_db,
            clip_count: self.clip_count,
            sample_count: self.sample_count,
        }
    }

    fn reset(&mut self) {
        self.peak_linear = 0.0;
        self.sum_sq = 0.0;
        self.sample_count = 0;
        self.clip_count = 0;
    }
}

/// Converts a linear amplitude to dBFS.
fn linear_to_dbfs(linear: f32) -> f32 {
    if linear <= 0.0 {
        return f32::NEG_INFINITY;
    }
    20.0 * linear.log10()
}

/// Manages multiple metering points across a signal path.
#[derive(Debug, Clone, Default)]
pub struct MeteringBridge {
    meters: HashMap<String, MeterState>,
    /// Default clip threshold (linear amplitude).
    clip_threshold: f32,
}

impl MeteringBridge {
    /// Creates a new metering bridge with a default clip threshold of 1.0.
    pub fn new() -> Self {
        Self {
            meters: HashMap::new(),
            clip_threshold: 1.0,
        }
    }

    /// Creates a bridge with a custom clip threshold (in linear amplitude).
    pub fn with_clip_threshold(clip_threshold: f32) -> Self {
        Self {
            meters: HashMap::new(),
            clip_threshold,
        }
    }

    /// Inserts a metering point at the named location.
    pub fn insert_meter(&mut self, name: impl Into<String>) {
        let name = name.into();
        self.meters
            .insert(name.clone(), MeterState::new(name, self.clip_threshold));
    }

    /// Removes a metering point.
    pub fn remove_meter(&mut self, name: &str) -> bool {
        self.meters.remove(name).is_some()
    }

    /// Feeds audio samples to the named meter point.
    ///
    /// Returns `false` if the meter point does not exist.
    pub fn feed(&mut self, name: &str, samples: &[f32]) -> bool {
        if let Some(state) = self.meters.get_mut(name) {
            state.feed(samples);
            true
        } else {
            false
        }
    }

    /// Gets the current reading for a meter point.
    pub fn reading(&self, name: &str) -> Option<MeterReading> {
        self.meters.get(name).map(|s| s.reading())
    }

    /// Resets a single meter point.
    pub fn reset_meter(&mut self, name: &str) {
        if let Some(state) = self.meters.get_mut(name) {
            state.reset();
        }
    }

    /// Resets all meter points.
    pub fn reset_all(&mut self) {
        for state in self.meters.values_mut() {
            state.reset();
        }
    }

    /// Returns the number of metering points.
    pub fn meter_count(&self) -> usize {
        self.meters.len()
    }

    /// Returns a snapshot of all readings keyed by meter name.
    pub fn all_readings(&self) -> HashMap<String, MeterReading> {
        self.meters
            .iter()
            .map(|(name, state)| (name.clone(), state.reading()))
            .collect()
    }

    /// Returns names of all meters that have clipped.
    pub fn clipped_meters(&self) -> Vec<String> {
        self.meters
            .iter()
            .filter(|(_, state)| state.clip_count > 0)
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Returns `true` if the named meter exists.
    pub fn has_meter(&self, name: &str) -> bool {
        self.meters.contains_key(name)
    }

    /// Returns all meter names.
    pub fn meter_names(&self) -> Vec<String> {
        self.meters.keys().cloned().collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_bridge() {
        let bridge = MeteringBridge::new();
        assert_eq!(bridge.meter_count(), 0);
    }

    #[test]
    fn test_insert_and_remove_meter() {
        let mut bridge = MeteringBridge::new();
        bridge.insert_meter("pre-eq");
        assert!(bridge.has_meter("pre-eq"));
        assert_eq!(bridge.meter_count(), 1);

        assert!(bridge.remove_meter("pre-eq"));
        assert!(!bridge.has_meter("pre-eq"));
        assert_eq!(bridge.meter_count(), 0);
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut bridge = MeteringBridge::new();
        assert!(!bridge.remove_meter("ghost"));
    }

    #[test]
    fn test_feed_unregistered() {
        let mut bridge = MeteringBridge::new();
        assert!(!bridge.feed("ghost", &[0.5, 0.5]));
    }

    #[test]
    fn test_reading_unregistered() {
        let bridge = MeteringBridge::new();
        assert!(bridge.reading("ghost").is_none());
    }

    #[test]
    fn test_silence_reading() {
        let mut bridge = MeteringBridge::new();
        bridge.insert_meter("m1");
        bridge.feed("m1", &[0.0, 0.0, 0.0]);
        let r = bridge.reading("m1").expect("meter exists");
        assert_eq!(r.sample_count, 3);
        assert!(r.peak_dbfs == f32::NEG_INFINITY);
        assert!(!r.has_clipped());
    }

    #[test]
    fn test_peak_detection() {
        let mut bridge = MeteringBridge::new();
        bridge.insert_meter("m1");
        bridge.feed("m1", &[0.1, 0.5, -0.8, 0.3]);
        let r = bridge.reading("m1").expect("meter exists");
        // Peak should be 0.8
        let expected_peak = 20.0 * 0.8_f32.log10();
        assert!((r.peak_dbfs - expected_peak).abs() < 0.01);
    }

    #[test]
    fn test_rms_calculation() {
        let mut bridge = MeteringBridge::new();
        bridge.insert_meter("m1");
        // Use a constant signal of 0.5 — RMS should also be 0.5
        let samples: Vec<f32> = vec![0.5; 100];
        bridge.feed("m1", &samples);
        let r = bridge.reading("m1").expect("meter exists");
        let expected_rms = 20.0 * 0.5_f32.log10();
        assert!((r.rms_dbfs - expected_rms).abs() < 0.1);
    }

    #[test]
    fn test_clip_detection() {
        let mut bridge = MeteringBridge::new();
        bridge.insert_meter("m1");
        bridge.feed("m1", &[0.5, 1.0, 1.5, -1.2]);
        let r = bridge.reading("m1").expect("meter exists");
        // Samples >= 1.0 in abs: 1.0, 1.5, 1.2 = 3 clips
        assert_eq!(r.clip_count, 3);
        assert!(r.has_clipped());
    }

    #[test]
    fn test_custom_clip_threshold() {
        let mut bridge = MeteringBridge::with_clip_threshold(0.9);
        bridge.insert_meter("m1");
        bridge.feed("m1", &[0.5, 0.95, 1.0]);
        let r = bridge.reading("m1").expect("meter exists");
        assert_eq!(r.clip_count, 2); // 0.95 and 1.0 both >= 0.9
    }

    #[test]
    fn test_crest_factor() {
        let mut bridge = MeteringBridge::new();
        bridge.insert_meter("m1");
        // Square wave has crest factor of 0 dB (peak == RMS)
        let samples: Vec<f32> = vec![0.5; 1000];
        bridge.feed("m1", &samples);
        let r = bridge.reading("m1").expect("meter exists");
        assert!(r.crest_factor_db.abs() < 0.01);
    }

    #[test]
    fn test_reset_meter() {
        let mut bridge = MeteringBridge::new();
        bridge.insert_meter("m1");
        bridge.feed("m1", &[0.5, 0.8]);
        bridge.reset_meter("m1");
        let r = bridge.reading("m1").expect("meter exists");
        assert!(r.is_empty());
        assert_eq!(r.sample_count, 0);
    }

    #[test]
    fn test_reset_all() {
        let mut bridge = MeteringBridge::new();
        bridge.insert_meter("a");
        bridge.insert_meter("b");
        bridge.feed("a", &[0.5]);
        bridge.feed("b", &[0.3]);
        bridge.reset_all();
        assert!(bridge.reading("a").expect("exists").is_empty());
        assert!(bridge.reading("b").expect("exists").is_empty());
    }

    #[test]
    fn test_all_readings() {
        let mut bridge = MeteringBridge::new();
        bridge.insert_meter("x");
        bridge.insert_meter("y");
        bridge.feed("x", &[0.5]);
        let readings = bridge.all_readings();
        assert_eq!(readings.len(), 2);
        assert!(readings.contains_key("x"));
        assert!(readings.contains_key("y"));
    }

    #[test]
    fn test_clipped_meters() {
        let mut bridge = MeteringBridge::new();
        bridge.insert_meter("clean");
        bridge.insert_meter("hot");
        bridge.feed("clean", &[0.5]);
        bridge.feed("hot", &[1.0, 1.5]);
        let clipped = bridge.clipped_meters();
        assert_eq!(clipped.len(), 1);
        assert_eq!(clipped[0], "hot");
    }

    #[test]
    fn test_meter_names() {
        let mut bridge = MeteringBridge::new();
        bridge.insert_meter("alpha");
        bridge.insert_meter("beta");
        let names = bridge.meter_names();
        assert_eq!(names.len(), 2);
    }

    #[test]
    fn test_empty_reading() {
        let r = MeterReading::default();
        assert!(r.is_empty());
        assert!(!r.has_clipped());
    }

    #[test]
    fn test_incremental_feed() {
        let mut bridge = MeteringBridge::new();
        bridge.insert_meter("m1");
        bridge.feed("m1", &[0.5, 0.3]);
        bridge.feed("m1", &[0.8]);
        let r = bridge.reading("m1").expect("exists");
        assert_eq!(r.sample_count, 3);
        let expected_peak = 20.0 * 0.8_f32.log10();
        assert!((r.peak_dbfs - expected_peak).abs() < 0.01);
    }
}
