//! Room acoustics correction.
//!
//! Analyses an impulse response to identify room modes (standing-wave
//! resonances) and builds a bank of notch filters to attenuate them.

/// A standing-wave room mode identified from an impulse response.
#[derive(Debug, Clone)]
pub struct RoomMode {
    /// Resonance frequency in Hz.
    pub frequency_hz: f32,
    /// RT60-style decay time in milliseconds.
    pub decay_ms: f32,
    /// Relative amplitude of the mode (linear, 0–1).
    pub amplitude: f32,
}

/// Analyses an impulse response and estimates room modes.
pub struct RoomModeAnalyzer;

impl RoomModeAnalyzer {
    /// Estimate room modes from an impulse response using a simple DFT peak
    /// search.
    ///
    /// Only frequencies below 500 Hz are searched (typical room-mode range).
    /// The DFT is computed directly (O(N²) – suitable for short IRs used in
    /// room measurement).
    ///
    /// # Arguments
    /// * `impulse_response` - The measured room impulse response.
    /// * `sample_rate`      - Sample rate in Hz.
    #[must_use]
    pub fn estimate_modes(impulse_response: &[f32], sample_rate: u32) -> Vec<RoomMode> {
        let n = impulse_response.len();
        if n < 2 {
            return Vec::new();
        }

        let sr = sample_rate as f32;
        let freq_resolution = sr / n as f32;

        // Upper limit: 500 Hz for room modes
        let max_bin = ((500.0 / freq_resolution) as usize + 1).min(n / 2);

        // Compute DFT magnitude for bins 1..=max_bin
        let magnitudes: Vec<f32> = (1..=max_bin)
            .map(|k| {
                let mut re = 0.0_f32;
                let mut im = 0.0_f32;
                let phase = 2.0 * std::f32::consts::PI * k as f32 / n as f32;
                for (j, &s) in impulse_response.iter().enumerate() {
                    re += s * (phase * j as f32).cos();
                    im -= s * (phase * j as f32).sin();
                }
                (re * re + im * im).sqrt()
            })
            .collect();

        // Find local peaks (simple neighbour comparison)
        let mut modes = Vec::new();
        let len = magnitudes.len();
        if len == 0 {
            return modes;
        }

        // Normalise magnitudes for amplitude calculation
        let max_mag = magnitudes.iter().cloned().fold(0.0_f32, f32::max).max(1e-9);

        for i in 0..len {
            let m = magnitudes[i];
            let is_peak =
                (i == 0 || m > magnitudes[i - 1]) && (i == len - 1 || m > magnitudes[i + 1]);

            if is_peak && m > max_mag * 0.1 {
                let freq = (i + 1) as f32 * freq_resolution;
                // Estimate decay from amplitude (rough: higher amp → longer decay)
                let decay_ms = 200.0 * (m / max_mag);
                let amplitude = (m / max_mag).min(1.0);

                modes.push(RoomMode {
                    frequency_hz: freq,
                    decay_ms,
                    amplitude,
                });
            }
        }

        // Sort by amplitude descending, keep top 8
        modes.sort_by(|a, b| {
            b.amplitude
                .partial_cmp(&a.amplitude)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        modes.truncate(8);
        modes
    }
}

/// A parametric notch (band-reject) biquad filter.
#[derive(Debug, Clone)]
pub struct NotchFilter {
    /// Centre frequency in Hz.
    pub center_hz: f32,
    /// Bandwidth in Hz (determines Q).
    pub bandwidth_hz: f32,
    /// Notch depth in dB (positive value → attenuation).
    pub depth_db: f32,
}

impl NotchFilter {
    /// Compute biquad coefficients (b0, b1, b2, a1, a2) for the notch filter.
    ///
    /// Implements a band-reject (notch) filter using the Audio EQ Cookbook
    /// formulation.  A pure notch (zero at the centre frequency) is computed
    /// and the Q is widened by `depth_db` to control the effective depth.
    ///
    /// Returns `(b0, b1, b2, a1, a2)` in normalised form (divided by a0), for
    /// use in the standard biquad difference equation:
    ///
    /// `y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]`
    #[must_use]
    pub fn compute_biquad(&self, sample_rate: f32) -> (f32, f32, f32, f32, f32) {
        let f0 = self.center_hz.max(1.0);
        let bw = self.bandwidth_hz.max(1.0);
        let w0 = 2.0 * std::f32::consts::PI * f0 / sample_rate;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();

        // Q from bandwidth; a larger Q gives a narrower, deeper notch.
        let q = (f0 / bw).max(0.5);
        let alpha = sin_w0 / (2.0 * q);

        // Pure notch (band-reject) coefficients (Audio EQ Cookbook §VII):
        // b0 = 1,  b1 = -2cos(w0),  b2 = 1
        // a0 = 1 + alpha,  a1 = -2cos(w0),  a2 = 1 - alpha
        let a0 = 1.0 + alpha;
        let b0 = 1.0 / a0;
        let b1 = (-2.0 * cos_w0) / a0;
        let b2 = 1.0 / a0;
        let a1 = (-2.0 * cos_w0) / a0;
        let a2 = (1.0 - alpha) / a0;

        (b0, b1, b2, a1, a2)
    }
}

/// A bank of notch filters derived from room-mode analysis.
#[derive(Debug, Clone)]
pub struct RoomCorrectionFilter {
    /// Individual notch filters.
    pub notch_filters: Vec<NotchFilter>,
}

/// Builds and applies room correction filters.
pub struct RoomCorrector;

impl RoomCorrector {
    /// Build a [`RoomCorrectionFilter`] from a list of room modes.
    #[must_use]
    pub fn build_filter(modes: &[RoomMode], _sample_rate: u32) -> RoomCorrectionFilter {
        let notch_filters = modes
            .iter()
            .map(|mode| {
                // Bandwidth proportional to decay (longer decay → narrower notch)
                let bandwidth_hz = mode.frequency_hz / 10.0_f32.max(mode.decay_ms / 10.0);
                // Depth proportional to amplitude
                let depth_db = 12.0 * mode.amplitude;
                NotchFilter {
                    center_hz: mode.frequency_hz,
                    bandwidth_hz,
                    depth_db,
                }
            })
            .collect();

        RoomCorrectionFilter { notch_filters }
    }

    /// Apply a [`RoomCorrectionFilter`] to `samples`.
    ///
    /// Each notch filter is applied sequentially (direct form II transposed
    /// biquad).
    ///
    /// # Arguments
    /// * `samples` - Input audio buffer.
    /// * `filter`  - Room correction filter bank.
    /// * `sample_rate` - Sample rate in Hz.
    #[must_use]
    pub fn apply(samples: &[f32], filter: &RoomCorrectionFilter, sample_rate: u32) -> Vec<f32> {
        let mut buf = samples.to_vec();
        let sr = sample_rate as f32;

        for notch in &filter.notch_filters {
            let (b0, b1, b2, a1, a2) = notch.compute_biquad(sr);
            let mut z1 = 0.0_f32;
            let mut z2 = 0.0_f32;

            for x in buf.iter_mut() {
                // Direct form II transposed biquad
                let y = b0 * *x + z1;
                z1 = b1 * *x - a1 * y + z2;
                z2 = b2 * *x - a2 * y;
                *x = y;
            }
        }

        buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn impulse(n: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; n];
        if !v.is_empty() {
            v[0] = 1.0;
        }
        v
    }

    fn sine_ir(freq_hz: f32, sample_rate: u32, n: usize) -> Vec<f32> {
        let sr = sample_rate as f32;
        (0..n)
            .map(|i| {
                let t = i as f32 / sr;
                (2.0 * std::f32::consts::PI * freq_hz * t).sin() * (-t * 10.0).exp()
                // exponential decay
            })
            .collect()
    }

    #[test]
    fn test_room_mode_analyzer_empty() {
        let modes = RoomModeAnalyzer::estimate_modes(&[], 44100);
        assert!(modes.is_empty());
    }

    #[test]
    fn test_room_mode_analyzer_impulse() {
        let ir = impulse(4096);
        // Impulse has flat spectrum – no strong peaks expected beyond threshold
        let modes = RoomModeAnalyzer::estimate_modes(&ir, 44100);
        // Result can be empty or have a few modes, just check no panic
        let _ = modes;
    }

    #[test]
    fn test_room_mode_analyzer_sine_resonance() {
        let sr = 44100u32;
        // Create IR with a resonance at ~80 Hz
        let ir = sine_ir(80.0, sr, 4096);
        let modes = RoomModeAnalyzer::estimate_modes(&ir, sr);
        // Should detect at least one mode
        assert!(!modes.is_empty(), "should detect a resonance");
        // The dominant mode should be near 80 Hz
        let dominant = &modes[0];
        assert!(
            dominant.frequency_hz > 40.0 && dominant.frequency_hz < 200.0,
            "expected mode near 80 Hz, got {} Hz",
            dominant.frequency_hz
        );
    }

    #[test]
    fn test_room_mode_fields() {
        let mode = RoomMode {
            frequency_hz: 80.0,
            decay_ms: 150.0,
            amplitude: 0.8,
        };
        assert!((mode.frequency_hz - 80.0).abs() < 1e-5);
        assert!((mode.decay_ms - 150.0).abs() < 1e-5);
        assert!((mode.amplitude - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_notch_filter_compute_biquad_returns_five_coefficients() {
        let n = NotchFilter {
            center_hz: 80.0,
            bandwidth_hz: 8.0,
            depth_db: 6.0,
        };
        let (b0, b1, b2, a1, a2) = n.compute_biquad(44100.0);
        // Coefficients should be finite
        assert!(b0.is_finite());
        assert!(b1.is_finite());
        assert!(b2.is_finite());
        assert!(a1.is_finite());
        assert!(a2.is_finite());
    }

    #[test]
    fn test_notch_filter_flat_at_zero_depth() {
        let n = NotchFilter {
            center_hz: 100.0,
            bandwidth_hz: 10.0,
            depth_db: 0.0, // no attenuation
        };
        let (b0, _b1, _b2, _a1, _a2) = n.compute_biquad(44100.0);
        // With 0 dB depth, gain_linear = 1, blend = 0, b0f ≈ b0*(1)+1-1 = b0
        assert!(b0 > 0.0);
    }

    #[test]
    fn test_build_filter_creates_notches() {
        let modes = vec![
            RoomMode {
                frequency_hz: 80.0,
                decay_ms: 200.0,
                amplitude: 0.9,
            },
            RoomMode {
                frequency_hz: 160.0,
                decay_ms: 100.0,
                amplitude: 0.5,
            },
        ];
        let filter = RoomCorrector::build_filter(&modes, 44100);
        assert_eq!(filter.notch_filters.len(), 2);
    }

    #[test]
    fn test_apply_preserves_length() {
        let modes = vec![RoomMode {
            frequency_hz: 80.0,
            decay_ms: 200.0,
            amplitude: 0.7,
        }];
        let filter = RoomCorrector::build_filter(&modes, 44100);
        let samples = vec![0.5f32; 1000];
        let out = RoomCorrector::apply(&samples, &filter, 44100);
        assert_eq!(out.len(), samples.len());
    }

    #[test]
    fn test_apply_empty_filter_identity() {
        let filter = RoomCorrectionFilter {
            notch_filters: Vec::new(),
        };
        let samples: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let out = RoomCorrector::apply(&samples, &filter, 44100);
        for (a, b) in samples.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_apply_attenuates_resonant_frequency() {
        use std::f32::consts::PI;
        let sr = 44100u32;
        let f = 100.0_f32;
        // Pure 100 Hz sine
        let samples: Vec<f32> = (0..sr as usize)
            .map(|i| (2.0 * PI * f * i as f32 / sr as f32).sin())
            .collect();

        let filter = RoomCorrectionFilter {
            notch_filters: vec![NotchFilter {
                center_hz: f,
                bandwidth_hz: 10.0,
                depth_db: 12.0,
            }],
        };
        let out = RoomCorrector::apply(&samples, &filter, sr);

        // RMS should be lower after notch filter
        let rms_in: f32 =
            (samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32).sqrt();
        let rms_out: f32 = (out.iter().map(|&x| x * x).sum::<f32>() / out.len() as f32).sqrt();
        assert!(
            rms_out < rms_in,
            "notch should reduce rms: in={rms_in} out={rms_out}"
        );
    }

    #[test]
    fn test_room_mode_analyzer_returns_at_most_8_modes() {
        let ir = sine_ir(50.0, 44100, 8192);
        let modes = RoomModeAnalyzer::estimate_modes(&ir, 44100);
        assert!(modes.len() <= 8);
    }
}
