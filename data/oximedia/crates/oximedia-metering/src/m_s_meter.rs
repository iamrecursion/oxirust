//! Mid/Side (M/S) metering for stereo audio.
//!
//! M/S processing encodes a stereo pair (L, R) into:
//! - **Mid** (M = (L + R) / 2): the mono-compatible centre content.
//! - **Side** (S = (L − R) / 2): the stereo difference (out-of-phase content).
//!
//! The M/S ratio and balance are useful for assessing stereo width, identifying
//! phase issues, and verifying mono compatibility before broadcast delivery.

#![allow(dead_code)]

/// A derived M/S balance measurement.
#[derive(Clone, Copy, Debug)]
pub struct MsBalance {
    /// Mid channel RMS level (linear).
    pub mid_rms: f64,
    /// Side channel RMS level (linear).
    pub side_rms: f64,
}

impl MsBalance {
    /// Create from mid and side RMS levels.
    pub fn new(mid_rms: f64, side_rms: f64) -> Self {
        Self { mid_rms, side_rms }
    }

    /// M/S ratio: side / mid.  A value > 1 means more side than mid.
    pub fn ms_ratio(&self) -> f64 {
        if self.mid_rms < 1e-15 {
            return f64::INFINITY;
        }
        self.side_rms / self.mid_rms
    }

    /// Returns `true` when the stereo field is "wide" (side energy dominates mid).
    pub fn is_wide(&self) -> bool {
        self.side_rms > self.mid_rms
    }

    /// Stereo width as a percentage in [0, 100].
    ///
    /// 0 % = pure mono (no side), 100 % = maximum width.
    pub fn width_percent(&self) -> f64 {
        let total = self.mid_rms + self.side_rms;
        if total < 1e-15 {
            return 0.0;
        }
        (self.side_rms / total * 100.0).clamp(0.0, 100.0)
    }
}

/// A channel in M/S space.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MsChannel {
    /// Centre / mono-compatible content.
    Mid,
    /// Stereo difference signal.
    Side,
}

impl MsChannel {
    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Mid => "Mid",
            Self::Side => "Side",
        }
    }
}

/// Running M/S meter that accumulates sum-of-squares for M and S channels.
#[derive(Clone, Debug, Default)]
pub struct MsMeter {
    mid_sum_sq: f64,
    side_sum_sq: f64,
    sample_count: usize,
    /// Peak M level seen (linear).
    mid_peak: f64,
    /// Peak S level seen (linear).
    side_peak: f64,
}

impl MsMeter {
    /// Create a new, empty M/S meter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Process a single stereo pair (left, right) sample.
    pub fn process_stereo_pair(&mut self, left: f64, right: f64) {
        let m = (left + right) * 0.5;
        let s = (left - right) * 0.5;

        self.mid_sum_sq += m * m;
        self.side_sum_sq += s * s;
        self.sample_count += 1;

        let m_abs = m.abs();
        let s_abs = s.abs();
        if m_abs > self.mid_peak {
            self.mid_peak = m_abs;
        }
        if s_abs > self.side_peak {
            self.side_peak = s_abs;
        }
    }

    /// Process an interleaved stereo buffer (L, R, L, R, …).
    pub fn process_interleaved(&mut self, samples: &[f64]) {
        assert!(
            samples.len() % 2 == 0,
            "interleaved buffer must have even length"
        );
        for pair in samples.chunks_exact(2) {
            self.process_stereo_pair(pair[0], pair[1]);
        }
    }

    /// Process a planar stereo buffer (all left samples then all right samples).
    pub fn process_planar(&mut self, left: &[f64], right: &[f64]) {
        assert_eq!(
            left.len(),
            right.len(),
            "left and right channels must have equal length"
        );
        for (&l, &r) in left.iter().zip(right.iter()) {
            self.process_stereo_pair(l, r);
        }
    }

    /// RMS level of the Mid channel (linear).
    pub fn mid_level(&self) -> f64 {
        if self.sample_count == 0 {
            return 0.0;
        }
        (self.mid_sum_sq / self.sample_count as f64).sqrt()
    }

    /// RMS level of the Side channel (linear).
    pub fn side_level(&self) -> f64 {
        if self.sample_count == 0 {
            return 0.0;
        }
        (self.side_sum_sq / self.sample_count as f64).sqrt()
    }

    /// Mid channel RMS in dBFS.
    pub fn mid_level_dbfs(&self) -> f64 {
        let rms = self.mid_level();
        if rms < 1e-15 {
            f64::NEG_INFINITY
        } else {
            20.0 * rms.log10()
        }
    }

    /// Side channel RMS in dBFS.
    pub fn side_level_dbfs(&self) -> f64 {
        let rms = self.side_level();
        if rms < 1e-15 {
            f64::NEG_INFINITY
        } else {
            20.0 * rms.log10()
        }
    }

    /// Peak Mid level (linear).
    pub fn mid_peak(&self) -> f64 {
        self.mid_peak
    }

    /// Peak Side level (linear).
    pub fn side_peak(&self) -> f64 {
        self.side_peak
    }

    /// M/S ratio (side RMS / mid RMS).
    pub fn ms_ratio(&self) -> f64 {
        let mid = self.mid_level();
        if mid < 1e-15 {
            return f64::INFINITY;
        }
        self.side_level() / mid
    }

    /// Compute a [`MsBalance`] snapshot from current accumulated data.
    pub fn balance(&self) -> MsBalance {
        MsBalance::new(self.mid_level(), self.side_level())
    }

    /// Number of stereo sample pairs processed.
    pub fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// Reset all accumulated state.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ms_channel_label_mid() {
        assert_eq!(MsChannel::Mid.label(), "Mid");
    }

    #[test]
    fn test_ms_channel_label_side() {
        assert_eq!(MsChannel::Side.label(), "Side");
    }

    #[test]
    fn test_ms_balance_is_wide_true() {
        let b = MsBalance::new(0.2, 0.8);
        assert!(b.is_wide());
    }

    #[test]
    fn test_ms_balance_is_wide_false() {
        let b = MsBalance::new(0.8, 0.2);
        assert!(!b.is_wide());
    }

    #[test]
    fn test_ms_balance_width_percent_mono() {
        let b = MsBalance::new(1.0, 0.0);
        assert!((b.width_percent() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_ms_balance_width_percent_full() {
        let b = MsBalance::new(0.0, 1.0);
        assert!((b.width_percent() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_ms_balance_ms_ratio() {
        let b = MsBalance::new(0.5, 1.0);
        assert!((b.ms_ratio() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_ms_meter_pure_mono_no_side() {
        // L == R => S = 0 exactly.
        let mut m = MsMeter::new();
        for _ in 0..100 {
            m.process_stereo_pair(0.5, 0.5);
        }
        assert!(m.side_level() < 1e-12);
        assert!(m.mid_level() > 0.0);
    }

    #[test]
    fn test_ms_meter_pure_side_no_mid() {
        // L == -R => M = 0 exactly.
        let mut m = MsMeter::new();
        for _ in 0..100 {
            m.process_stereo_pair(0.5, -0.5);
        }
        assert!(m.mid_level() < 1e-12);
        assert!(m.side_level() > 0.0);
    }

    #[test]
    fn test_ms_meter_sample_count() {
        let mut m = MsMeter::new();
        m.process_stereo_pair(0.1, -0.1);
        m.process_stereo_pair(0.2, 0.2);
        assert_eq!(m.sample_count(), 2);
    }

    #[test]
    fn test_ms_meter_interleaved() {
        let mut m = MsMeter::new();
        let samples = vec![0.5_f64, 0.5, 0.3, -0.3]; // 2 pairs
        m.process_interleaved(&samples);
        assert_eq!(m.sample_count(), 2);
    }

    #[test]
    fn test_ms_meter_planar() {
        let mut m = MsMeter::new();
        let left = vec![0.4_f64, 0.6];
        let right = vec![0.4_f64, -0.6];
        m.process_planar(&left, &right);
        assert_eq!(m.sample_count(), 2);
    }

    #[test]
    fn test_ms_meter_reset() {
        let mut m = MsMeter::new();
        m.process_stereo_pair(0.5, 0.3);
        m.reset();
        assert_eq!(m.sample_count(), 0);
        assert_eq!(m.mid_level(), 0.0);
        assert_eq!(m.side_level(), 0.0);
    }

    #[test]
    fn test_mid_level_dbfs_is_negative() {
        let mut m = MsMeter::new();
        m.process_stereo_pair(0.5, 0.5);
        let db = m.mid_level_dbfs();
        assert!(db < 0.0, "RMS dBFS should be negative for sub-unity signal");
    }

    #[test]
    fn test_side_level_dbfs_neg_inf_for_zero() {
        let m = MsMeter::new();
        assert!(m.side_level_dbfs().is_infinite());
    }

    #[test]
    fn test_peak_tracking() {
        let mut m = MsMeter::new();
        m.process_stereo_pair(0.8, -0.8); // M=0, S=0.8
        m.process_stereo_pair(0.2, 0.2); // M=0.2, S=0
        assert!((m.side_peak() - 0.8).abs() < 1e-9);
        assert!((m.mid_peak() - 0.2).abs() < 1e-9);
    }
}
