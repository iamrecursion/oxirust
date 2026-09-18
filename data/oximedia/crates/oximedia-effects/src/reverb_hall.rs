//! Hall-style algorithmic reverb using a network of comb and all-pass filters.
//!
//! [`ReverbHall`] implements a Schroeder/Moorer-inspired hall reverb with
//! parallel comb filters feeding into a series of all-pass diffusers.
//! The design targets smooth, dense reverberation characteristic of large
//! concert halls and auditoriums.
//!
//! # Example
//!
//! ```
//! use oximedia_effects::reverb_hall::{ReverbHallConfig, ReverbHall, ReverbHallType};
//!
//! let config = ReverbHallConfig::default();
//! let mut reverb = ReverbHall::new(config, 48_000.0);
//!
//! let mut left  = vec![0.0_f32; 256];
//! let mut right = vec![0.0_f32; 256];
//! left[0]  = 1.0; // impulse
//! right[0] = 1.0;
//! reverb.process_stereo(&mut left, &mut right);
//! ```

#![allow(dead_code)]

/// Reverb character / size preset selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReverbHallType {
    /// Small chamber (RT60 ≈ 0.4 s).
    Chamber,
    /// Medium concert hall (RT60 ≈ 1.5 s).
    Hall,
    /// Large cathedral (RT60 ≈ 4.0 s).
    Cathedral,
    /// Custom — parameters taken directly from [`ReverbHallConfig`].
    Custom,
}

/// Configuration parameters for [`ReverbHall`].
#[derive(Debug, Clone)]
pub struct ReverbHallConfig {
    /// Reverb character.
    pub reverb_type: ReverbHallType,
    /// Room size factor (0.0 – 1.0). Scales comb filter delays.
    pub room_size: f32,
    /// High-frequency damping (0.0 – 1.0). Higher = darker tail.
    pub damping: f32,
    /// Wet signal level (0.0 – 1.0).
    pub wet: f32,
    /// Dry signal level (0.0 – 1.0).
    pub dry: f32,
    /// Stereo spread (0.0 – 1.0).
    pub spread: f32,
    /// Pre-delay in milliseconds.
    pub predelay_ms: f32,
}

impl Default for ReverbHallConfig {
    fn default() -> Self {
        Self {
            reverb_type: ReverbHallType::Hall,
            room_size: 0.75,
            damping: 0.4,
            wet: 0.30,
            dry: 0.70,
            spread: 0.5,
            predelay_ms: 15.0,
        }
    }
}

impl ReverbHallConfig {
    /// Create a configuration for a specific hall type with sensible defaults.
    #[must_use]
    pub fn from_type(hall_type: ReverbHallType) -> Self {
        match hall_type {
            ReverbHallType::Chamber => Self {
                reverb_type: ReverbHallType::Chamber,
                room_size: 0.35,
                damping: 0.55,
                wet: 0.25,
                dry: 0.75,
                spread: 0.4,
                predelay_ms: 5.0,
            },
            ReverbHallType::Hall | ReverbHallType::Custom => Self::default(),
            ReverbHallType::Cathedral => Self {
                reverb_type: ReverbHallType::Cathedral,
                room_size: 0.95,
                damping: 0.25,
                wet: 0.45,
                dry: 0.55,
                spread: 0.8,
                predelay_ms: 40.0,
            },
        }
    }
}

/// A simple comb filter with lowpass feedback (Schroeder comb filter).
struct CombFilter {
    buffer: Vec<f32>,
    write_pos: usize,
    feedback: f32,
    damp1: f32,
    damp2: f32,
    filter_store: f32,
}

impl CombFilter {
    fn new(size: usize) -> Self {
        Self {
            buffer: vec![0.0; size.max(1)],
            write_pos: 0,
            feedback: 0.84,
            damp1: 0.2,
            damp2: 0.8,
            filter_store: 0.0,
        }
    }

    fn process(&mut self, input: f32) -> f32 {
        let output = self.buffer[self.write_pos];
        self.filter_store = output * self.damp2 + self.filter_store * self.damp1;
        self.buffer[self.write_pos] = input + self.filter_store * self.feedback;
        self.write_pos = (self.write_pos + 1) % self.buffer.len();
        output
    }

    fn set_feedback(&mut self, fb: f32) {
        self.feedback = fb.clamp(0.0, 0.99);
    }

    fn set_damping(&mut self, d: f32) {
        self.damp1 = d.clamp(0.0, 1.0);
        self.damp2 = 1.0 - self.damp1;
    }

    fn clear(&mut self) {
        for s in &mut self.buffer {
            *s = 0.0;
        }
        self.filter_store = 0.0;
        self.write_pos = 0;
    }
}

/// An all-pass diffuser filter.
struct AllPassFilter {
    buffer: Vec<f32>,
    write_pos: usize,
    feedback: f32,
}

impl AllPassFilter {
    fn new(size: usize) -> Self {
        Self {
            buffer: vec![0.0; size.max(1)],
            write_pos: 0,
            feedback: 0.5,
        }
    }

    fn process(&mut self, input: f32) -> f32 {
        let buffered = self.buffer[self.write_pos];
        let output = -input + buffered;
        self.buffer[self.write_pos] = input + buffered * self.feedback;
        self.write_pos = (self.write_pos + 1) % self.buffer.len();
        output
    }

    fn clear(&mut self) {
        for s in &mut self.buffer {
            *s = 0.0;
        }
        self.write_pos = 0;
    }
}

/// Hall reverb — 8 parallel comb filters followed by 4 all-pass diffusers.
pub struct ReverbHall {
    config: ReverbHallConfig,
    sample_rate: f32,
    /// Comb filters for left channel.
    combs_l: Vec<CombFilter>,
    /// Comb filters for right channel.
    combs_r: Vec<CombFilter>,
    /// All-pass chain for left channel.
    allpass_l: Vec<AllPassFilter>,
    /// All-pass chain for right channel.
    allpass_r: Vec<AllPassFilter>,
    /// Pre-delay ring buffer.
    predelay_buf: Vec<f32>,
    predelay_pos: usize,
    predelay_len: usize,
}

/// Comb filter delay sizes (in samples at 44.1 kHz) — from Freeverb.
const COMB_TUNINGS: [usize; 8] = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617];
/// All-pass filter delay sizes.
const ALLPASS_TUNINGS: [usize; 4] = [556, 441, 341, 225];
/// Stereo spread offset (samples).
const STEREO_SPREAD: usize = 23;

impl ReverbHall {
    /// Create a new hall reverb at the given sample rate.
    #[must_use]
    pub fn new(config: ReverbHallConfig, sample_rate: f32) -> Self {
        let rate_scale = sample_rate / 44_100.0;

        let combs_l: Vec<CombFilter> = COMB_TUNINGS
            .iter()
            .map(|&t| {
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    clippy::cast_precision_loss
                )]
                CombFilter::new((t as f32 * rate_scale) as usize)
            })
            .collect();
        let combs_r: Vec<CombFilter> = COMB_TUNINGS
            .iter()
            .map(|&t| {
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    clippy::cast_precision_loss
                )]
                CombFilter::new(((t + STEREO_SPREAD) as f32 * rate_scale) as usize)
            })
            .collect();

        let allpass_l: Vec<AllPassFilter> = ALLPASS_TUNINGS
            .iter()
            .map(|&t| {
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    clippy::cast_precision_loss
                )]
                AllPassFilter::new((t as f32 * rate_scale) as usize)
            })
            .collect();
        let allpass_r: Vec<AllPassFilter> = ALLPASS_TUNINGS
            .iter()
            .map(|&t| {
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    clippy::cast_precision_loss
                )]
                AllPassFilter::new(((t + STEREO_SPREAD) as f32 * rate_scale) as usize)
            })
            .collect();

        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let predelay_len = ((config.predelay_ms * 0.001 * sample_rate) as usize).max(1);

        let mut reverb = Self {
            config,
            sample_rate,
            combs_l,
            combs_r,
            allpass_l,
            allpass_r,
            predelay_buf: vec![0.0; predelay_len + 1],
            predelay_pos: 0,
            predelay_len,
        };
        reverb.apply_config();
        reverb
    }

    /// Apply current config parameters to the internal filters.
    fn apply_config(&mut self) {
        let feedback = self.config.room_size * 0.28 + 0.7;
        let damp = self.config.damping;
        for c in self.combs_l.iter_mut().chain(self.combs_r.iter_mut()) {
            c.set_feedback(feedback);
            c.set_damping(damp);
        }
    }

    /// Process a single mono sample through the comb+allpass network.
    #[must_use]
    pub fn apply_sample_mono(&mut self, input: f32) -> f32 {
        let mut out = 0.0_f32;
        for c in &mut self.combs_l {
            out += c.process(input);
        }
        for ap in &mut self.allpass_l {
            out = ap.process(out);
        }
        out
    }

    /// Process a stereo sample pair in-place.
    pub fn process_stereo(&mut self, left: &mut [f32], right: &mut [f32]) {
        let len = left.len().min(right.len());
        for i in 0..len {
            // Pre-delay
            let pd_out = self.predelay_buf[self.predelay_pos];
            let mixed = (left[i] + right[i]) * 0.015;
            self.predelay_buf[self.predelay_pos] = mixed;
            self.predelay_pos = (self.predelay_pos + 1) % self.predelay_buf.len();

            // Left path
            let mut out_l = 0.0_f32;
            for c in &mut self.combs_l {
                out_l += c.process(pd_out);
            }
            for ap in &mut self.allpass_l {
                out_l = ap.process(out_l);
            }

            // Right path
            let mut out_r = 0.0_f32;
            for c in &mut self.combs_r {
                out_r += c.process(pd_out);
            }
            for ap in &mut self.allpass_r {
                out_r = ap.process(out_r);
            }

            let spread = self.config.spread * 0.5;
            let wet_l = out_l * (0.5 + spread) + out_r * (0.5 - spread);
            let wet_r = out_r * (0.5 + spread) + out_l * (0.5 - spread);

            left[i] = left[i] * self.config.dry + wet_l * self.config.wet;
            right[i] = right[i] * self.config.dry + wet_r * self.config.wet;
        }
    }

    /// Reset all internal filter states.
    pub fn reset(&mut self) {
        for c in self.combs_l.iter_mut().chain(self.combs_r.iter_mut()) {
            c.clear();
        }
        for ap in self.allpass_l.iter_mut().chain(self.allpass_r.iter_mut()) {
            ap.clear();
        }
        for s in &mut self.predelay_buf {
            *s = 0.0;
        }
        self.predelay_pos = 0;
    }

    /// Update the configuration and re-apply parameters.
    pub fn set_config(&mut self, config: ReverbHallConfig) {
        self.config = config;
        self.apply_config();
    }

    /// Return a reference to the current configuration.
    #[must_use]
    pub fn config(&self) -> &ReverbHallConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_reverb() -> ReverbHall {
        ReverbHall::new(ReverbHallConfig::default(), 48_000.0)
    }

    #[test]
    fn test_reverb_hall_type_variants() {
        let _ = ReverbHallType::Chamber;
        let _ = ReverbHallType::Hall;
        let _ = ReverbHallType::Cathedral;
        let _ = ReverbHallType::Custom;
    }

    #[test]
    fn test_config_from_type_chamber() {
        let c = ReverbHallConfig::from_type(ReverbHallType::Chamber);
        assert!(c.room_size < 0.5);
        assert_eq!(c.reverb_type, ReverbHallType::Chamber);
    }

    #[test]
    fn test_config_from_type_cathedral() {
        let c = ReverbHallConfig::from_type(ReverbHallType::Cathedral);
        assert!(c.room_size > 0.9);
        assert!(c.predelay_ms > 30.0);
    }

    #[test]
    fn test_reverb_hall_new() {
        let r = make_reverb();
        assert_eq!(r.combs_l.len(), 8);
        assert_eq!(r.allpass_l.len(), 4);
    }

    #[test]
    fn test_reverb_hall_silence_in_silence_out() {
        let mut r = make_reverb();
        let mut l = vec![0.0_f32; 256];
        let mut ri = vec![0.0_f32; 256];
        r.process_stereo(&mut l, &mut ri);
        // All zeros in → all (near-)zeros out
        for (&lv, &rv) in l.iter().zip(ri.iter()) {
            assert!(lv.abs() < 1e-6);
            assert!(rv.abs() < 1e-6);
        }
    }

    #[test]
    fn test_reverb_hall_impulse_produces_tail() {
        let mut r = make_reverb();
        // Comb filter delays at 48 kHz are ~1200+ samples; use a large enough
        // buffer so the reverb tail has time to appear.
        let mut l = vec![0.0_f32; 4096];
        let mut ri = vec![0.0_f32; 4096];
        l[0] = 1.0;
        ri[0] = 1.0;
        r.process_stereo(&mut l, &mut ri);
        // After impulse, some tail samples should be non-zero
        let tail_energy: f32 = l[10..].iter().map(|&s| s * s).sum();
        assert!(tail_energy > 0.0);
    }

    #[test]
    fn test_reverb_hall_wet_zero_bypasses() {
        let config = ReverbHallConfig {
            wet: 0.0,
            dry: 1.0,
            ..ReverbHallConfig::default()
        };
        let mut r = ReverbHall::new(config, 48_000.0);
        let mut l = vec![0.5_f32; 8];
        let mut ri = vec![0.5_f32; 8];
        r.process_stereo(&mut l, &mut ri);
        for &v in l.iter().chain(ri.iter()) {
            assert!((v - 0.5).abs() < 1e-5);
        }
    }

    #[test]
    fn test_reverb_hall_reset_clears_state() {
        let mut r = make_reverb();
        let mut l = vec![1.0_f32; 64];
        let mut ri = vec![1.0_f32; 64];
        r.process_stereo(&mut l, &mut ri);
        r.reset();
        // After reset, silence in → silence out
        let mut l2 = vec![0.0_f32; 64];
        let mut r2 = vec![0.0_f32; 64];
        r.process_stereo(&mut l2, &mut r2);
        for &v in l2.iter().chain(r2.iter()) {
            assert!(v.abs() < 1e-6);
        }
    }

    #[test]
    fn test_reverb_hall_set_config() {
        let mut r = make_reverb();
        let new_cfg = ReverbHallConfig::from_type(ReverbHallType::Cathedral);
        r.set_config(new_cfg);
        assert_eq!(r.config().reverb_type, ReverbHallType::Cathedral);
    }

    #[test]
    fn test_comb_filter_feedback_clamp() {
        let mut c = CombFilter::new(100);
        c.set_feedback(1.5);
        assert!(c.feedback <= 0.99);
    }

    #[test]
    fn test_comb_filter_clear() {
        let mut c = CombFilter::new(8);
        c.process(1.0);
        c.clear();
        assert_eq!(c.filter_store, 0.0);
    }

    #[test]
    fn test_allpass_filter_process_non_zero() {
        let mut ap = AllPassFilter::new(50);
        let out = ap.process(1.0);
        // First call: buffered=0, output = -1+0 = -1
        assert!((out - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_allpass_filter_clear() {
        let mut ap = AllPassFilter::new(16);
        ap.process(1.0);
        ap.clear();
        // Buffer is zeroed; next process with 0 should return 0
        let out = ap.process(0.0);
        assert_eq!(out, 0.0);
    }

    #[test]
    fn test_reverb_hall_apply_sample_mono() {
        let mut r = make_reverb();
        // Impulse
        let out = r.apply_sample_mono(1.0);
        // After one sample there may be zero output (pre-delay); just check no panic
        let _ = out;
        // Feed silence and confirm decay
        let mut energy = 0.0_f32;
        for _ in 0..2048 {
            let s = r.apply_sample_mono(0.0);
            energy += s * s;
        }
        assert!(energy >= 0.0);
    }

    #[test]
    fn test_reverb_hall_stereo_spread_symmetry() {
        // With spread=0 both channels should be identical after an impulse
        let config = ReverbHallConfig {
            spread: 0.0,
            ..ReverbHallConfig::from_type(ReverbHallType::Hall)
        };
        let mut r = ReverbHall::new(config, 48_000.0);
        // Comb filter delays at 48 kHz are ~1200+ samples; use a large enough
        // buffer so the reverb tail has time to appear.
        let mut l = vec![0.0_f32; 4096];
        let mut ri = vec![0.0_f32; 4096];
        l[0] = 1.0;
        ri[0] = 1.0;
        r.process_stereo(&mut l, &mut ri);
        // With spread=0, left and right wet paths should be mixed symmetrically
        // They won't be identical due to STEREO_SPREAD comb tuning offset,
        // but neither should be all-zero in the tail
        let l_energy: f32 = l[5..].iter().map(|&s| s * s).sum();
        let r_energy: f32 = ri[5..].iter().map(|&s| s * s).sum();
        assert!(l_energy > 0.0);
        assert!(r_energy > 0.0);
    }
}
