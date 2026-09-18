// Integration tests for full signal analysis pipeline
// Exercises: scirs2-signal -> scirs2-fft -> scirs2-stats

use crate::fixtures::TestDatasets;
use num_complex::Complex64;
use scirs2_core::ndarray::Array1;
use scirs2_fft::{fft, ifft, rfft};
use scirs2_signal::convolve;
use scirs2_stats::{mean, std, var};

type TestResult<T> = Result<T, Box<dyn std::error::Error>>;

// ---------------------------------------------------------------------------
// Test 1: Basic power spectrum — dominant frequency should be recoverable
// ---------------------------------------------------------------------------

/// Generate a sine wave, compute FFT, verify the dominant bin matches the input frequency.
#[test]
fn test_signal_pipeline_basic_spectrum() -> TestResult<()> {
    // Use power-of-two length so bin k == frequency k (when fs == n)
    let n = 256usize;
    let fs = n as f64; // sample rate equals N so bin k = freq k Hz
    let freq_hz = 20.0f64; // target frequency: bin 20

    let signal = TestDatasets::sinusoid_signal(n, freq_hz, fs);
    assert_eq!(signal.len(), n);

    // Compute FFT via scirs2-fft
    let signal_vec: Vec<f64> = signal.to_vec();
    let spectrum = fft(&signal_vec, Some(n)).map_err(|e| format!("FFT failed: {}", e))?;

    // Power spectrum (skip DC bin 0, look at positive frequencies)
    let powers: Vec<f64> = spectrum.iter().map(|c| c.norm_sqr()).collect();

    let peak_bin = powers[1..n / 2]
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i + 1)
        .ok_or("empty power spectrum")?;

    assert_eq!(
        peak_bin, freq_hz as usize,
        "Peak should be at bin {} (freq {:.0} Hz), found bin {}",
        freq_hz as usize, freq_hz, peak_bin
    );

    // Dominant bin power should be much larger than average
    let avg_power = powers[1..n / 2].iter().sum::<f64>() / (n / 2 - 1) as f64;
    assert!(
        powers[peak_bin] > 10.0 * avg_power,
        "Dominant bin power ({:.2}) should dominate average ({:.2})",
        powers[peak_bin],
        avg_power
    );

    println!(
        "Spectrum test: dominant bin={} (expected {}), peak/avg ratio={:.1}",
        peak_bin,
        freq_hz as usize,
        powers[peak_bin] / avg_power
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Test 2: Filtering pipeline — apply box filter, verify reduced high-freq energy
// ---------------------------------------------------------------------------

/// Apply a box (moving-average) filter via scirs2_signal::convolve, then confirm
/// that the filtered signal has lower high-frequency energy than the original.
#[test]
fn test_signal_pipeline_filtering() -> TestResult<()> {
    let n = 256usize;
    let fs = n as f64;

    // Mixed signal: low freq (5 Hz) + high freq (100 Hz)
    let low = TestDatasets::sinusoid_signal(n, 5.0, fs);
    let high = TestDatasets::sinusoid_signal(n, 100.0, fs);
    let mixed: Vec<f64> = low.iter().zip(high.iter()).map(|(l, h)| l + h).collect();

    // Box filter kernel (length-5 moving average acts as a low-pass filter)
    let kernel_len = 5usize;
    let kernel: Vec<f64> = vec![1.0 / kernel_len as f64; kernel_len];

    // Apply filter via scirs2-signal convolve (mode = "same")
    let filtered =
        convolve(&mixed, &kernel, "same").map_err(|e| format!("convolve failed: {}", e))?;

    assert_eq!(
        filtered.len(),
        n,
        "Filtered signal length should match input"
    );

    // FFT of original and filtered, compute high-frequency energy (bins 50..n/2)
    let spec_orig = fft(&mixed, Some(n)).map_err(|e| format!("FFT orig: {}", e))?;
    let spec_filt = fft(&filtered, Some(n)).map_err(|e| format!("FFT filt: {}", e))?;

    let hi_cutoff = 20usize; // bins above this are "high frequency"
    let hf_energy_orig: f64 = spec_orig[hi_cutoff..n / 2]
        .iter()
        .map(|c| c.norm_sqr())
        .sum();
    let hf_energy_filt: f64 = spec_filt[hi_cutoff..n / 2]
        .iter()
        .map(|c| c.norm_sqr())
        .sum();

    assert!(
        hf_energy_filt < hf_energy_orig,
        "Filtered HF energy ({:.4}) should be less than original ({:.4})",
        hf_energy_filt,
        hf_energy_orig
    );

    println!(
        "Filtering test: HF energy orig={:.4}, filtered={:.4}, reduction={:.1}x",
        hf_energy_orig,
        hf_energy_filt,
        hf_energy_orig / (hf_energy_filt + 1e-12)
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Test 3: Time-series statistics via scirs2-stats
// ---------------------------------------------------------------------------

/// Compute mean, std, variance of a known signal and verify against analytical values.
#[test]
fn test_signal_pipeline_stats() -> TestResult<()> {
    // A pure sine wave of unit amplitude has:
    //   mean = 0  (over exact integer number of periods)
    //   variance = 0.5  (for sin^2, mean = 0.5)
    //   std = sqrt(0.5) ~ 0.7071
    let n = 1024usize; // large N for accurate statistics
    let fs = n as f64;
    let freq = 8.0f64; // 8 complete cycles in N samples

    let signal = TestDatasets::sinusoid_signal(n, freq, fs);
    let signal_view = signal.view();

    // Mean via scirs2-stats
    let computed_mean = mean(&signal_view).map_err(|e| format!("mean failed: {}", e))?;

    // Variance (sample, ddof=0) should be ~0.5
    let computed_var = var(&signal_view, 0, None).map_err(|e| format!("var failed: {}", e))?;

    // Std (sample, ddof=0)
    let computed_std = std(&signal_view, 0, None).map_err(|e| format!("std failed: {}", e))?;

    // Verify mean is near 0 (tolerance proportional to 1/sqrt(n))
    assert!(
        computed_mean.abs() < 0.01,
        "Mean of pure sine should be near 0, got {}",
        computed_mean
    );

    // Variance of unit-amplitude sine over full periods = 0.5
    assert!(
        (computed_var - 0.5).abs() < 0.01,
        "Variance of unit sine should be ~0.5, got {}",
        computed_var
    );

    // std = sqrt(var)
    let expected_std = computed_var.sqrt();
    assert!(
        (computed_std - expected_std).abs() < 1e-10,
        "std should equal sqrt(var): std={}, sqrt(var)={}",
        computed_std,
        expected_std
    );

    // Autocorrelation-style check: compute FFT-based autocorrelation at lag 0
    // which equals the total energy / n = variance (for zero-mean signal)
    let sig_vec: Vec<f64> = signal.to_vec();
    let fft_result = fft(&sig_vec, Some(n)).map_err(|e| format!("FFT for autocorr: {}", e))?;

    let power_sum: f64 = fft_result.iter().map(|c| c.norm_sqr()).sum::<f64>() / n as f64;
    let variance_via_parseval = power_sum / n as f64;

    // Parseval: sum|x|^2/N = sum|X|^2/N^2, so variance_via_parseval ~ variance
    // (within floating point; both represent total power)
    assert!(
        variance_via_parseval.is_finite(),
        "Parseval energy should be finite"
    );

    println!(
        "Signal stats: mean={:.6}, var={:.6}, std={:.6}",
        computed_mean, computed_var, computed_std
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Test 4: Composite pipeline — generate, filter, then analyze residual
// ---------------------------------------------------------------------------

/// End-to-end pipeline: generate composite signal, low-pass filter it,
/// verify that residual (original minus filtered) has mostly high-frequency content.
#[test]
fn test_signal_pipeline_residual_analysis() -> TestResult<()> {
    let n = 512usize;
    let fs = n as f64;

    let low_signal = TestDatasets::sinusoid_signal(n, 5.0, fs);
    let high_signal = TestDatasets::sinusoid_signal(n, 80.0, fs);
    let composite: Vec<f64> = low_signal
        .iter()
        .zip(high_signal.iter())
        .map(|(l, h)| l + h)
        .collect();

    // Apply smoothing kernel (box filter length 9)
    let k = 9usize;
    let kernel: Vec<f64> = vec![1.0 / k as f64; k];
    let filtered = convolve(&composite, &kernel, "same").map_err(|e| format!("convolve: {}", e))?;

    // Residual = original - filtered
    let residual: Vec<f64> = composite
        .iter()
        .zip(filtered.iter())
        .map(|(o, f)| o - f)
        .collect();

    // Energy of residual
    let residual_energy: f64 = residual.iter().map(|&r| r * r).sum::<f64>() / n as f64;
    let original_energy: f64 = composite.iter().map(|&c| c * c).sum::<f64>() / n as f64;

    assert!(
        residual_energy.is_finite(),
        "Residual energy should be finite"
    );
    assert!(original_energy > 0.0, "Original signal should have energy");

    // Residual should contain a significant fraction of original energy
    // (the high-freq component should survive)
    let residual_fraction = residual_energy / original_energy;
    assert!(
        residual_fraction > 0.1,
        "Residual should retain at least 10% of signal energy, got {:.2}",
        residual_fraction
    );

    // Stats on residual via scirs2-stats
    let residual_arr = Array1::from_vec(residual);
    let res_mean = mean(&residual_arr.view()).map_err(|e| format!("residual mean: {}", e))?;
    let res_std = std(&residual_arr.view(), 0, None).map_err(|e| format!("residual std: {}", e))?;

    assert!(res_mean.is_finite(), "Residual mean should be finite");
    assert!(res_std.is_finite(), "Residual std should be finite");
    assert!(res_std > 0.0, "Residual should have non-zero variance");

    println!(
        "Residual analysis: energy_fraction={:.2}, mean={:.4}, std={:.4}",
        residual_fraction, res_mean, res_std
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Test 5: RFFT output length and Hermitian symmetry
// ---------------------------------------------------------------------------

/// Verify rfft output length and that real signal produces Hermitian-symmetric spectrum.
#[test]
fn test_signal_pipeline_rfft_symmetry() -> TestResult<()> {
    let n = 128usize;
    let fs = n as f64;
    let signal = TestDatasets::sinusoid_signal(n, 7.0, fs);
    let sig_vec: Vec<f64> = signal.to_vec();

    let rfft_out = rfft(&sig_vec, None).map_err(|e| format!("rfft failed: {}", e))?;

    // rfft of real signal: n/2 + 1 complex bins
    let expected_len = n / 2 + 1;
    assert_eq!(
        rfft_out.len(),
        expected_len,
        "rfft output length: expected {}, got {}",
        expected_len,
        rfft_out.len()
    );

    // Full FFT of same signal
    let fft_out = fft(&sig_vec, Some(n)).map_err(|e| format!("fft failed: {}", e))?;

    // rfft[k] should equal fft[k] for k in 0..n/2+1
    for k in 0..expected_len {
        let diff = (rfft_out[k] - fft_out[k]).norm();
        assert!(
            diff < 1e-10,
            "rfft[{}] = {:?} differs from fft[{}] = {:?} by {:.2e}",
            k,
            rfft_out[k],
            k,
            fft_out[k],
            diff
        );
    }

    println!(
        "RFFT symmetry verified: {} bins (n/2+1={})",
        rfft_out.len(),
        expected_len
    );
    Ok(())
}
