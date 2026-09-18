//! Signal energy analysis — RMS energy, zero-crossing rate, energy envelope,
//! and energy-based silence detection.
//!
//! These functions provide fundamental building blocks for higher-level audio
//! understanding including voice-activity detection, music/speech discrimination,
//! and dynamic range profiling.

#![forbid(unsafe_code)]
#![allow(
    dead_code,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

/// Compute the RMS (Root Mean Square) energy of a signal window.
///
/// RMS energy is the square root of the mean of squared samples.  It gives a
/// perceptually meaningful measure of signal loudness for a short frame.
///
/// # Arguments
/// * `samples` – Audio samples (any length).
///
/// # Returns
/// RMS value in [0.0, …].  Returns `0.0` for an empty slice.
#[must_use]
pub fn rms_energy(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|&x| x * x).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

/// Compute the zero-crossing rate of a signal window.
///
/// The ZCR is the number of sign transitions divided by (N − 1), where N is
/// the number of samples.  It is commonly used as a voiced/unvoiced discriminator
/// and a rough measure of high-frequency content.
///
/// # Arguments
/// * `samples` – Audio samples.
///
/// # Returns
/// ZCR in crossings-per-sample (range 0.0–1.0).  Returns `0.0` for slices
/// shorter than 2 samples.
#[must_use]
pub fn zero_crossing_rate(samples: &[f32]) -> f32 {
    if samples.len() < 2 {
        return 0.0;
    }

    let mut crossings = 0usize;
    for i in 1..samples.len() {
        // A crossing when adjacent samples have opposite sign.
        // Treat 0.0 as positive to avoid double-counting.
        let pos_a = samples[i - 1] >= 0.0;
        let pos_b = samples[i] >= 0.0;
        if pos_a != pos_b {
            crossings += 1;
        }
    }

    crossings as f32 / (samples.len() - 1) as f32
}

/// Compute a short-time energy envelope by calculating RMS for overlapping frames.
///
/// The signal is partitioned into frames of `frame_size` samples with a hop of
/// `hop_size` samples.  Incomplete frames at the end are discarded.
///
/// # Arguments
/// * `samples`    – Audio samples.
/// * `frame_size` – Length of each analysis frame in samples (must be ≥ 1).
/// * `hop_size`   – Number of samples to advance between frames (must be ≥ 1).
///
/// # Returns
/// Vector of per-frame RMS values.  Empty when `samples` is too short, or
/// when `frame_size`/`hop_size` is 0.
#[must_use]
pub fn energy_envelope(samples: &[f32], frame_size: usize, hop_size: usize) -> Vec<f32> {
    if frame_size == 0 || hop_size == 0 || samples.len() < frame_size {
        return Vec::new();
    }

    let n_frames = (samples.len() - frame_size) / hop_size + 1;
    let mut envelope = Vec::with_capacity(n_frames);

    for frame_idx in 0..n_frames {
        let start = frame_idx * hop_size;
        let end = start + frame_size;
        if end > samples.len() {
            break;
        }
        envelope.push(rms_energy(&samples[start..end]));
    }

    envelope
}

/// Detect silence regions in an audio signal using energy thresholding.
///
/// The signal is analysed in 10 ms frames.  Consecutive frames whose RMS energy
/// falls below `threshold_db` for at least `min_duration_ms` are merged into a
/// silence region.
///
/// # Arguments
/// * `samples`         – Audio samples.
/// * `sample_rate`     – Sample rate in Hz (must be > 0).
/// * `threshold_db`    – Energy threshold in dBFS (e.g. −40.0).  Frames with
///                        RMS below this value are considered silent.
/// * `min_duration_ms` – Minimum silence length in milliseconds.  Regions
///                        shorter than this are discarded.
///
/// # Returns
/// List of `(start_sample, end_sample)` pairs for each silence region.
/// Returns an empty vector when no suitable silence regions are found.
#[must_use]
pub fn detect_silence_regions(
    samples: &[f32],
    sample_rate: u32,
    threshold_db: f32,
    min_duration_ms: u32,
) -> Vec<(usize, usize)> {
    if samples.is_empty() || sample_rate == 0 {
        return Vec::new();
    }

    let sr = sample_rate as f32;
    // Analysis frame: 10 ms
    let frame_samples = ((0.01 * sr) as usize).max(1);
    // Convert dB threshold to linear RMS
    let threshold_linear = db_to_linear(threshold_db);
    // Minimum number of consecutive silent frames
    let min_frames = (min_duration_ms as f32 / 10.0).ceil() as usize;

    // Build per-frame silence flags
    let n_frames = if samples.len() >= frame_samples {
        (samples.len() - frame_samples) / frame_samples + 1
    } else {
        0
    };

    if n_frames == 0 {
        return Vec::new();
    }

    let mut silent_flags = Vec::with_capacity(n_frames);
    for frame_idx in 0..n_frames {
        let start = frame_idx * frame_samples;
        let end = (start + frame_samples).min(samples.len());
        let rms = rms_energy(&samples[start..end]);
        silent_flags.push(rms < threshold_linear);
    }

    // Group consecutive silent frames into candidate regions
    let mut regions: Vec<(usize, usize)> = Vec::new();
    let mut in_silence = false;
    let mut silence_start_frame = 0usize;

    for (i, &is_silent) in silent_flags.iter().enumerate() {
        if is_silent && !in_silence {
            silence_start_frame = i;
            in_silence = true;
        } else if !is_silent && in_silence {
            // End of a silent run — check minimum duration
            let run_frames = i - silence_start_frame;
            if run_frames >= min_frames {
                let start_sample = silence_start_frame * frame_samples;
                let end_sample = (i * frame_samples).min(samples.len());
                regions.push((start_sample, end_sample));
            }
            in_silence = false;
        }
    }

    // Handle trailing silence
    if in_silence {
        let run_frames = n_frames - silence_start_frame;
        if run_frames >= min_frames {
            let start_sample = silence_start_frame * frame_samples;
            let end_sample = samples.len();
            regions.push((start_sample, end_sample));
        }
    }

    regions
}

// ── private helpers ───────────────────────────────────────────────────────────

/// Convert dBFS to linear amplitude.
fn db_to_linear(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── rms_energy ────────────────────────────────────────────────────────────

    #[test]
    fn test_rms_energy_all_ones() {
        let samples = vec![1.0_f32; 100];
        let rms = rms_energy(&samples);
        assert!(
            (rms - 1.0).abs() < 1e-6,
            "RMS of all-ones should be 1.0, got {rms}"
        );
    }

    #[test]
    fn test_rms_energy_empty() {
        assert_eq!(rms_energy(&[]), 0.0);
    }

    #[test]
    fn test_rms_energy_alternating() {
        let samples = vec![1.0, -1.0, 1.0, -1.0];
        let rms = rms_energy(&samples);
        assert!(
            (rms - 1.0).abs() < 1e-6,
            "RMS of ±1 alternating should be 1.0, got {rms}"
        );
    }

    #[test]
    fn test_rms_energy_zeros() {
        let samples = vec![0.0_f32; 256];
        assert_eq!(rms_energy(&samples), 0.0);
    }

    #[test]
    fn test_rms_energy_half_amplitude() {
        let samples = vec![0.5_f32; 100];
        let rms = rms_energy(&samples);
        assert!(
            (rms - 0.5).abs() < 1e-6,
            "RMS of 0.5 should be 0.5, got {rms}"
        );
    }

    // ── zero_crossing_rate ────────────────────────────────────────────────────

    #[test]
    fn test_zcr_alternating_equals_one() {
        let samples = vec![1.0, -1.0, 1.0, -1.0, 1.0];
        let zcr = zero_crossing_rate(&samples);
        assert!(
            (zcr - 1.0).abs() < 1e-6,
            "Perfect alternating signal ZCR should be 1.0, got {zcr}"
        );
    }

    #[test]
    fn test_zcr_constant_positive_is_zero() {
        let samples = vec![1.0_f32; 50];
        assert_eq!(zero_crossing_rate(&samples), 0.0);
    }

    #[test]
    fn test_zcr_constant_negative_is_zero() {
        let samples = vec![-0.5_f32; 50];
        assert_eq!(zero_crossing_rate(&samples), 0.0);
    }

    #[test]
    fn test_zcr_empty_is_zero() {
        assert_eq!(zero_crossing_rate(&[]), 0.0);
    }

    #[test]
    fn test_zcr_single_sample_is_zero() {
        assert_eq!(zero_crossing_rate(&[0.5]), 0.0);
    }

    #[test]
    fn test_zcr_range() {
        // ZCR must stay in [0, 1]
        let samples: Vec<f32> = (0..256)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let zcr = zero_crossing_rate(&samples);
        assert!((0.0..=1.0).contains(&zcr));
    }

    // ── energy_envelope ───────────────────────────────────────────────────────

    #[test]
    fn test_energy_envelope_correct_frame_count() {
        let samples = vec![1.0_f32; 1000];
        let env = energy_envelope(&samples, 100, 50);
        // (1000 - 100) / 50 + 1 = 19 frames
        assert_eq!(env.len(), 19, "Unexpected frame count: {}", env.len());
    }

    #[test]
    fn test_energy_envelope_silence_near_zero() {
        let samples = vec![0.0_f32; 2048];
        let env = energy_envelope(&samples, 512, 256);
        for &v in &env {
            assert!(v < 1e-7, "Silent envelope should be near zero, got {v}");
        }
    }

    #[test]
    fn test_energy_envelope_empty_frame_size_zero() {
        let samples = vec![1.0_f32; 1000];
        assert!(energy_envelope(&samples, 0, 100).is_empty());
    }

    #[test]
    fn test_energy_envelope_hop_size_zero() {
        let samples = vec![1.0_f32; 1000];
        assert!(energy_envelope(&samples, 100, 0).is_empty());
    }

    #[test]
    fn test_energy_envelope_signal_shorter_than_frame() {
        let samples = vec![1.0_f32; 50];
        assert!(energy_envelope(&samples, 100, 50).is_empty());
    }

    // ── detect_silence_regions ────────────────────────────────────────────────

    #[test]
    fn test_detect_silence_finds_silent_signal() {
        // Completely silent 1-second signal at 44100 Hz
        let samples = vec![0.0_f32; 44100];
        let regions = detect_silence_regions(&samples, 44100, -30.0, 100);
        assert!(
            !regions.is_empty(),
            "Should detect silence in all-zero signal"
        );
    }

    #[test]
    fn test_detect_silence_empty_for_loud_signal() {
        // Full-scale sine: RMS ≈ 0.707 → well above any typical dB threshold
        let sr = 44100_u32;
        let samples: Vec<f32> = (0..44100)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin())
            .collect();
        let regions = detect_silence_regions(&samples, sr, -60.0, 50);
        assert!(
            regions.is_empty(),
            "Loud sine wave should not be detected as silence"
        );
    }

    #[test]
    fn test_detect_silence_empty_input() {
        let regions = detect_silence_regions(&[], 44100, -40.0, 100);
        assert!(regions.is_empty());
    }

    #[test]
    fn test_detect_silence_zero_sample_rate() {
        let samples = vec![0.0_f32; 1000];
        let regions = detect_silence_regions(&samples, 0, -40.0, 100);
        assert!(regions.is_empty());
    }

    #[test]
    fn test_detect_silence_region_bounds_are_valid() {
        let samples = vec![0.0_f32; 44100];
        let regions = detect_silence_regions(&samples, 44100, -20.0, 50);
        for (start, end) in &regions {
            assert!(*start <= *end, "start ({start}) should be <= end ({end})");
            assert!(
                *end <= samples.len(),
                "end ({end}) should be within signal length"
            );
        }
    }

    #[test]
    fn test_detect_silence_mixed_signal() {
        let sr = 44100_u32;
        // 0.5 s silence then 0.5 s tone
        let mut samples = vec![0.0_f32; sr as usize / 2];
        samples.extend(
            (0..sr as usize / 2)
                .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin()),
        );
        let regions = detect_silence_regions(&samples, sr, -40.0, 200);
        assert!(
            !regions.is_empty(),
            "Should detect the silence portion of mixed signal"
        );
    }
}
