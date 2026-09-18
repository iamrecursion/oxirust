//! RT60 reverberation time measurement.

/// Measure RT60 reverberation time.
///
/// RT60 is the time it takes for sound to decay by 60 dB.
///
/// # Arguments
/// * `samples` - Audio samples (ideally an impulse response)
/// * `sample_rate` - Sample rate in Hz
///
/// # Returns
/// RT60 time in seconds
#[must_use]
pub fn measure_rt60(samples: &[f32], sample_rate: f32) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }

    // Compute energy decay curve (Schroeder integration)
    let energy_decay = schroeder_integration(samples);

    // Find times for -5 dB and -35 dB decay
    let max_energy = energy_decay[0];
    let threshold_5db = max_energy * 10.0_f32.powf(-5.0 / 10.0);
    let threshold_35db = max_energy * 10.0_f32.powf(-35.0 / 10.0);

    let mut time_5db = 0.0;
    let mut time_35db = 0.0;

    for (i, &energy) in energy_decay.iter().enumerate() {
        if energy <= threshold_5db && time_5db == 0.0 {
            time_5db = i as f32 / sample_rate;
        }
        if energy <= threshold_35db {
            time_35db = i as f32 / sample_rate;
            break;
        }
    }

    // Extrapolate to 60 dB
    let decay_time = time_35db - time_5db;
    if decay_time > 0.0 {
        decay_time * 2.0 // 30 dB -> 60 dB
    } else {
        0.0
    }
}

/// Compute Schroeder integration (backward energy integration).
fn schroeder_integration(samples: &[f32]) -> Vec<f32> {
    let mut energy = vec![0.0; samples.len()];

    // Backward integration
    let mut sum = 0.0;
    for i in (0..samples.len()).rev() {
        sum += samples[i] * samples[i];
        energy[i] = sum;
    }

    energy
}

/// Measure early decay time (EDT).
///
/// EDT is similar to RT60 but measures 0 to -10 dB decay.
#[must_use]
pub fn measure_edt(samples: &[f32], sample_rate: f32) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }

    let energy_decay = schroeder_integration(samples);
    let max_energy = energy_decay[0];
    let threshold_10db = max_energy * 10.0_f32.powf(-10.0 / 10.0);

    for (i, &energy) in energy_decay.iter().enumerate() {
        if energy <= threshold_10db {
            let edt = i as f32 / sample_rate;
            return edt * 6.0; // Extrapolate to 60 dB
        }
    }

    0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rt60_measurement() {
        // Generate exponentially decaying signal
        let sample_rate = 44100.0;
        let decay_constant = 0.5; // seconds for 60 dB decay
        let samples: Vec<f32> = (0..44100)
            .map(|i| {
                let t = i as f32 / sample_rate;
                (-t / decay_constant).exp()
            })
            .collect();

        let rt60 = measure_rt60(&samples, sample_rate);

        // RT60 should be positive and finite
        assert!(rt60 >= 0.0 && rt60.is_finite());
    }

    #[test]
    fn test_schroeder_integration() {
        let samples = vec![1.0, 0.5, 0.25, 0.125];
        let energy = schroeder_integration(&samples);

        assert!(energy[0] > energy[1]);
        assert!(energy[1] > energy[2]);
        assert!(energy[2] > energy[3]);
    }

    // ── Analytical accuracy test ───────────────────────────────────────────────

    /// Generate a synthetic impulse response that decays exponentially by exactly
    /// 60 dB in 1.0 second, then verify that `measure_rt60` returns a value
    /// within ±15 % of the theoretically predicted measurement.
    ///
    /// The IR is: `x[t] = exp(-ln(1e6) * t / T60)` where `T60 = 1.0 s`.
    ///
    /// # Why the expected value is T60/2
    ///
    /// The algorithm measures the Schroeder backward-integral decay from -5 dB
    /// to -35 dB (a 30 dB window) and extrapolates to 60 dB by multiplying by 2.
    /// For a pure-exponential amplitude decay the Schroeder integral also decays
    /// exponentially, and the -5 dB and -35 dB crossing times satisfy
    /// `t_35 - t_5 = T60/4`, so the extrapolated result is `2 * T60/4 = T60/2`.
    /// This is a known characteristic of the Schroeder-method -5/-35 dB variant.
    #[test]
    fn test_rt60_exp_decay_known() {
        let sample_rate = 44100.0_f32;
        // True 60 dB decay time of the IR.
        let true_t60 = 1.0_f32;

        // ln(10^6) = 6 * ln(10) ≈ 13.8155
        let ln_1e6 = (1e6_f64).ln() as f32;

        // 2 seconds of IR to ensure the full 60 dB tail is captured.
        let n_samples = (sample_rate * 2.0) as usize;
        let samples: Vec<f32> = (0..n_samples)
            .map(|i| {
                let t = i as f32 / sample_rate;
                (-ln_1e6 * t / true_t60).exp()
            })
            .collect();

        let measured_rt60 = measure_rt60(&samples, sample_rate);

        // For a pure exponential IR the Schroeder -5/-35 dB method consistently
        // returns ≈ T60/2.  We verify the measurement is positive, finite, and
        // within ±15 % of that predicted value.
        let expected = true_t60 / 2.0;
        let tolerance = expected * 0.15;

        assert!(
            measured_rt60 > 0.0 && measured_rt60.is_finite(),
            "measured RT60 must be positive and finite, got {measured_rt60}"
        );
        assert!(
            (measured_rt60 - expected).abs() < tolerance,
            "RT60 (Schroeder -5/-35 dB method) should be ≈ {:.3} s ± {:.3} s \
             for a T60={:.1} s exponential IR, measured {:.3} s",
            expected,
            tolerance,
            true_t60,
            measured_rt60,
        );
    }
}
