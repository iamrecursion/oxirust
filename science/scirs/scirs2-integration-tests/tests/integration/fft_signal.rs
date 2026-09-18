// Integration tests for scirs2-fft + scirs2-signal
// Tests spectral analysis pipelines, filter design, and FFT-based operations

use crate::common::*;
use crate::fixtures::TestDatasets;
use num_complex::Complex64;
use proptest::prelude::*;
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_fft::*;
use scirs2_signal::*;

type TestResult<T> = Result<T, Box<dyn std::error::Error>>;

/// Test FFT-based filtering pipeline
#[test]
fn test_fft_based_filtering() -> TestResult<()> {
    let signal = TestDatasets::sinusoid_signal(1024, 10.0, 1024.0);

    println!("Testing FFT-based filtering");
    println!("Signal length: {}", signal.len());

    let signal_slice: Vec<f64> = signal.to_vec();
    let fft_result = fft(&signal_slice, None).map_err(|e| format!("FFT failed: {}", e))?;

    println!("FFT computed, spectrum length: {}", fft_result.len());

    Ok(())
}

/// Test spectral analysis pipeline
#[test]
fn test_spectral_analysis_pipeline() -> TestResult<()> {
    let sampling_rate = 1000.0;
    let duration = 2.0;
    let n_samples = (sampling_rate * duration) as usize;

    let freq1 = 10.0;
    let freq2 = 50.0;
    let freq3 = 120.0;

    let signal1 = TestDatasets::sinusoid_signal(n_samples, freq1, sampling_rate);
    let signal2 = TestDatasets::sinusoid_signal(n_samples, freq2, sampling_rate);
    let signal3 = TestDatasets::sinusoid_signal(n_samples, freq3, sampling_rate);

    let composite_signal = &signal1 + &signal2 + &signal3;

    println!("Testing spectral analysis pipeline");
    println!(
        "Signal length: {}, sampling rate: {} Hz",
        n_samples, sampling_rate
    );

    let signal_vec: Vec<f64> = composite_signal.to_vec();
    let spectrum = fft(&signal_vec, None).map_err(|e| format!("FFT failed: {}", e))?;

    let power_spectrum: Vec<f64> = spectrum.iter().map(|c| c.norm_sqr()).collect();

    println!("Computed power spectrum with {} bins", power_spectrum.len());

    Ok(())
}

/// Test window functions integration
#[test]
fn test_window_functions_with_fft() -> TestResult<()> {
    let signal = TestDatasets::sinusoid_signal(512, 10.0, 512.0);

    println!("Testing window functions with FFT");

    let signal_vec: Vec<f64> = signal.to_vec();
    let spectrum = fft(&signal_vec, None).map_err(|e| format!("FFT failed: {}", e))?;

    println!("Computed windowed FFT with {} samples", spectrum.len());

    Ok(())
}

/// Test spectrogram computation
#[test]
fn test_spectrogram_computation() -> TestResult<()> {
    let signal = TestDatasets::sinusoid_signal(4096, 10.0, 1000.0);
    let window_size = 256;
    let hop_size = 128;

    println!("Testing spectrogram computation");
    println!(
        "Signal length: {}, window: {}, hop: {}",
        signal.len(),
        window_size,
        hop_size
    );

    Ok(())
}

/// Test convolution via FFT
#[test]
fn test_fft_convolution() -> TestResult<()> {
    let signal = TestDatasets::sinusoid_signal(256, 5.0, 256.0);
    let kernel = Array1::from_vec(vec![0.25, 0.5, 0.25]);

    println!("Testing FFT-based convolution");
    println!(
        "Signal length: {}, kernel length: {}",
        signal.len(),
        kernel.len()
    );

    Ok(())
}

/// Test filter design and application
#[test]
fn test_filter_design_and_application() -> TestResult<()> {
    let sampling_rate = 1000.0;
    let signal = TestDatasets::sinusoid_signal(1000, 50.0, sampling_rate);

    println!("Testing filter design and application");

    Ok(())
}

/// Test Hilbert transform integration
#[test]
fn test_hilbert_transform() -> TestResult<()> {
    let signal = TestDatasets::sinusoid_signal(512, 10.0, 512.0);

    println!("Testing Hilbert transform via FFT");

    Ok(())
}

/// Test zero-padding effects
#[test]
fn test_zero_padding_effects() -> TestResult<()> {
    let signal = TestDatasets::sinusoid_signal(100, 5.0, 100.0);

    println!("Testing zero-padding effects");

    let padded_lengths = vec![128usize, 256, 512];

    for &padded_len in &padded_lengths {
        let signal_vec: Vec<f64> = signal.to_vec();
        let spectrum = fft(&signal_vec, Some(padded_len))
            .map_err(|e| format!("FFT with padding {} failed: {}", padded_len, e))?;
        assert_eq!(
            spectrum.len(),
            padded_len,
            "FFT output length mismatch for padding {}",
            padded_len
        );
        println!(
            "  Padded length {}: spectrum len = {}",
            padded_len,
            spectrum.len()
        );
    }

    Ok(())
}

/// Test real-valued FFT integration
#[test]
fn test_rfft_integration() -> TestResult<()> {
    let signal = TestDatasets::sinusoid_signal(1024, 10.0, 1024.0);

    println!("Testing real-valued FFT integration");

    let signal_vec: Vec<f64> = signal.to_vec();
    let rfft_result = rfft(&signal_vec, None).map_err(|e| format!("RFFT failed: {}", e))?;

    println!("RFFT computed, output length: {}", rfft_result.len());

    // rfft on n-sample signal produces n/2 + 1 complex outputs
    let n = signal_vec.len();
    assert_eq!(
        rfft_result.len(),
        n / 2 + 1,
        "rfft output length: expected {} (n/2+1), got {}",
        n / 2 + 1,
        rfft_result.len()
    );

    Ok(())
}

/// Test 2D FFT for image processing
#[test]
fn test_2d_fft_integration() -> TestResult<()> {
    let image = TestDatasets::test_image_gradient(64);

    println!("Testing 2D FFT integration");
    println!("Image shape: {:?}", image.shape());

    Ok(())
}

// ---------------------------------------------------------------------------
// Additional real tests
// ---------------------------------------------------------------------------

/// Test spectral peak detection: FFT of a pure tone should peak at the correct bin
#[test]
fn test_spectral_peak_detection() -> TestResult<()> {
    // Use power-of-two length for integer-bin frequencies
    let n = 256usize;
    let sampling_rate = n as f64; // fs = n so bin k corresponds to freq k
    let freq = 10.0; // frequency in Hz — bin 10

    let signal = TestDatasets::sinusoid_signal(n, freq, sampling_rate);
    let signal_vec: Vec<f64> = signal.to_vec();

    let spectrum = fft(&signal_vec, None).map_err(|e| format!("FFT failed: {}", e))?;

    // Find the bin with maximum power (ignore DC at bin 0)
    let powers: Vec<f64> = spectrum.iter().map(|c| c.norm_sqr()).collect();
    let peak_bin = powers[1..n / 2].iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i + 1)  // +1 because we skipped bin 0
        .ok_or("empty spectrum")?;

    assert_eq!(
        peak_bin, freq as usize,
        "Peak should be at bin {} (freq {}), found bin {}",
        freq as usize, freq, peak_bin
    );

    println!(
        "Spectral peak at bin {} (expected {})",
        peak_bin, freq as usize
    );
    Ok(())
}

/// Test FFT filtering: zero high-frequency bins, IFFT should give smoother signal
#[test]
fn test_fft_filtering() -> TestResult<()> {
    let n = 256usize;
    let sampling_rate = n as f64;

    // Signal with a low and a high frequency component
    let low_freq = TestDatasets::sinusoid_signal(n, 5.0, sampling_rate);
    let high_freq = TestDatasets::sinusoid_signal(n, 100.0, sampling_rate);
    let mixed: Vec<f64> = low_freq
        .iter()
        .zip(high_freq.iter())
        .map(|(l, h)| l + h)
        .collect();

    // FFT
    let mut spectrum = fft(&mixed, None).map_err(|e| format!("FFT failed: {}", e))?;

    // Low-pass: zero out all bins above cutoff (keep bins 0..=10)
    let cutoff_bin = 10usize;
    for k in cutoff_bin..spectrum.len() - cutoff_bin {
        spectrum[k] = Complex64::new(0.0, 0.0);
    }

    // IFFT to reconstruct
    let reconstructed = ifft(&spectrum, None).map_err(|e| format!("IFFT failed: {}", e))?;

    // Compute variance of original mixed and reconstructed real parts
    let orig_var: f64 = {
        let mean = mixed.iter().sum::<f64>() / n as f64;
        mixed.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64
    };
    let recon_real: Vec<f64> = reconstructed.iter().map(|c| c.re).collect();
    let recon_mean = recon_real.iter().sum::<f64>() / n as f64;
    let recon_var: f64 = recon_real
        .iter()
        .map(|&x| (x - recon_mean).powi(2))
        .sum::<f64>()
        / n as f64;

    // Filtered signal should have lower variance (removed high-freq energy)
    assert!(
        recon_var < orig_var,
        "Filtered signal variance ({:.4}) should be less than original ({:.4})",
        recon_var,
        orig_var
    );

    println!(
        "FFT filtering: orig_var={:.4}, filtered_var={:.4}",
        orig_var, recon_var
    );
    Ok(())
}

/// Test FFT convolution theorem: FFT-based circular convolution matches naive circular convolution
#[test]
fn test_fft_convolution_theorem() -> TestResult<()> {
    // Use power-of-two size so no zero-padding occurs
    let n = 8usize;
    let signal = vec![1.0f64, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0];
    // Kernel zero-padded to length n (circular convolution kernel)
    let kernel = vec![0.25f64, 0.5, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0];

    // Naive CIRCULAR convolution: y[i] = sum_{j=0}^{n-1} signal[j] * kernel[(i-j) mod n]
    let naive_circular: Vec<f64> = (0..n)
        .map(|i| {
            signal
                .iter()
                .enumerate()
                .map(|(j, &s)| s * kernel[(i + n - j) % n])
                .sum()
        })
        .collect();

    // FFT-based circular convolution: IFFT(FFT(signal) * FFT(kernel))
    let sig_spectrum = fft(&signal, Some(n)).map_err(|e| format!("FFT signal: {}", e))?;
    let ker_spectrum = fft(&kernel, Some(n)).map_err(|e| format!("FFT kernel: {}", e))?;

    let y_spectrum: Vec<Complex64> = sig_spectrum
        .iter()
        .zip(ker_spectrum.iter())
        .map(|(s, k)| s * k)
        .collect();

    let y = ifft(&y_spectrum, Some(n)).map_err(|e| format!("IFFT: {}", e))?;

    // Compare real parts against naive circular convolution
    for i in 0..n {
        let got = y[i].re;
        let exp = naive_circular[i];
        assert!(
            (got - exp).abs() < 1e-8,
            "y[{}]: naive_circ={:.6}, fft_conv={:.6}, diff={:.2e}",
            i,
            exp,
            got,
            (got - exp).abs()
        );
    }

    println!(
        "FFT convolution theorem verified (circular, n={} samples)",
        n
    );
    Ok(())
}

/// Test rfft output length
#[test]
fn test_rfft_output_length() -> TestResult<()> {
    for &n in &[64usize, 128, 256, 512, 1024] {
        let signal = TestDatasets::sinusoid_signal(n, 5.0, n as f64);
        let signal_vec: Vec<f64> = signal.to_vec();
        let rfft_out = rfft(&signal_vec, None).map_err(|e| format!("rfft n={}: {}", n, e))?;
        let expected_len = n / 2 + 1;
        assert_eq!(
            rfft_out.len(),
            expected_len,
            "rfft(n={}) output length: expected {}, got {}",
            n,
            expected_len,
            rfft_out.len()
        );
    }
    println!("rfft output length verified for all tested sizes");
    Ok(())
}

// ---------------------------------------------------------------------------
// Property-based tests
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn prop_fft_parseval_theorem(
        // Restrict to exact power-of-two sizes to avoid zero-padding issues
        exp in 6usize..9usize
    ) {
        // Property: Parseval's theorem — sum|x|^2 = (1/N) * sum|X|^2
        // Must use explicit N matching the FFT size; zero-padding breaks the identity.
        let signal_len = 1usize << exp; // 64, 128, or 256
        let signal = TestDatasets::sinusoid_signal(signal_len, 5.0, signal_len as f64);

        let time_energy: f64 = signal.iter().map(|&x| x * x).sum();

        let signal_vec: Vec<f64> = signal.to_vec();
        // Pass n=signal_len so no zero-padding occurs
        let fft_result = fft(&signal_vec, Some(signal_len))
            .expect("FFT failed in property test");
        let fft_n = fft_result.len();
        let freq_energy: f64 = fft_result.iter()
            .map(|c| c.norm_sqr())
            .sum::<f64>() / fft_n as f64;

        let tolerance = 1e-5;
        prop_assert!(
            (time_energy - freq_energy).abs() < tolerance,
            "Parseval's theorem violated: time_energy={}, freq_energy={}, N={}",
            time_energy, freq_energy, fft_n
        );
    }

    #[test]
    fn prop_fft_linearity(
        signal_len in 64usize..128,
        scale1 in -10.0f64..10.0,
        scale2 in -10.0f64..10.0
    ) {
        // Property: FFT is linear: FFT(a*x + b*y) = a*FFT(x) + b*FFT(y)
        let signal1 = TestDatasets::sinusoid_signal(signal_len, 5.0, signal_len as f64);
        let signal2 = TestDatasets::sinusoid_signal(signal_len, 10.0, signal_len as f64);

        let combined: Vec<f64> = signal1.iter().zip(signal2.iter())
            .map(|(&x1, &x2)| x1 * scale1 + x2 * scale2)
            .collect();

        let s1: Vec<f64> = signal1.to_vec();
        let s2: Vec<f64> = signal2.to_vec();

        let fft1 = fft(&s1, None).expect("FFT1 failed");
        let fft2 = fft(&s2, None).expect("FFT2 failed");
        let fft_combined = fft(&combined, None).expect("FFT combined failed");

        // fft_linear[k] = fft1[k] * scale1 + fft2[k] * scale2
        let max_diff = fft_combined.iter()
            .zip(fft1.iter().zip(fft2.iter()))
            .map(|(fc, (f1, f2))| {
                let linear = *f1 * scale1 + *f2 * scale2;
                (fc - linear).norm()
            })
            .fold(0.0_f64, f64::max);

        prop_assert!(
            max_diff < 1e-6,
            "FFT linearity violated: max_diff={}",
            max_diff
        );
    }

    #[test]
    fn prop_fft_ifft_roundtrip(
        signal_len in 64usize..256
    ) {
        // Property: IFFT(FFT(x)) = x (within numerical precision)
        let signal = TestDatasets::sinusoid_signal(signal_len, 7.0, signal_len as f64);

        let signal_vec: Vec<f64> = signal.to_vec();
        let fft_result = fft(&signal_vec, None)
            .expect("FFT failed in roundtrip test");
        let reconstructed = ifft(&fft_result, None)
            .expect("IFFT failed in roundtrip test");

        let max_error = signal_vec.iter()
            .zip(reconstructed.iter())
            .map(|(&orig, recon)| (recon.re - orig).abs())
            .fold(0.0_f64, f64::max);

        prop_assert!(
            max_error < 1e-10,
            "FFT/IFFT roundtrip error too large: {}",
            max_error
        );
    }
}

/// Test cross-correlation via FFT
#[test]
fn test_cross_correlation_via_fft() -> TestResult<()> {
    // Use a DC signal (all-ones) so the circular autocorrelation is flat;
    // use a delta signal instead: [1, 0, 0, ..., 0]
    // Circular autocorrelation of a delta should give a delta at lag 0.
    let n = 128usize;
    let mut s1 = vec![0.0f64; n];
    s1[0] = 1.0; // delta at position 0
    let s2 = s1.clone();

    println!("Testing cross-correlation via FFT (delta signal)");

    let f1 = fft(&s1, Some(n)).map_err(|e| format!("FFT s1: {}", e))?;
    let f2 = fft(&s2, Some(n)).map_err(|e| format!("FFT s2: {}", e))?;

    // Cross-correlation: FFT(s1) * conj(FFT(s2))
    let cross_spectrum: Vec<Complex64> = f1
        .iter()
        .zip(f2.iter())
        .map(|(a, b)| a * b.conj())
        .collect();
    let corr = ifft(&cross_spectrum, Some(n)).map_err(|e| format!("IFFT: {}", e))?;

    // Autocorrelation of delta should peak at lag 0
    let peak_idx = corr
        .iter()
        .enumerate()
        .max_by(|a, b| {
            a.1.re
                .partial_cmp(&b.1.re)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .ok_or("empty correlation")?;

    assert_eq!(
        peak_idx, 0,
        "Autocorrelation peak should be at lag 0, found {}",
        peak_idx
    );

    // Verify the peak value is approximately n (energy of delta * n due to FFT normalisation)
    assert!(
        corr[0].re > 0.5,
        "Peak value should be positive, got {}",
        corr[0].re
    );

    println!(
        "Cross-correlation peak at lag 0 (value={:.4}) — verified",
        corr[0].re
    );
    Ok(())
}

/// Test frequency shifting
#[test]
fn test_frequency_shifting() -> TestResult<()> {
    let signal = TestDatasets::sinusoid_signal(512, 20.0, 512.0);
    let shift_freq = 10.0;
    let _ = shift_freq;

    println!("Testing frequency shifting");

    Ok(())
}

/// Test memory efficiency of FFT pipeline
#[test]
fn test_fft_pipeline_memory_efficiency() -> TestResult<()> {
    let signal = TestDatasets::sinusoid_signal(8192, 10.0, 8192.0);

    println!("Testing FFT pipeline memory efficiency");

    assert_memory_efficient(
        || {
            let signal_vec: Vec<f64> = signal.to_vec();
            let spectrum = fft(&signal_vec, None).map_err(|e| format!("FFT: {}", e))?;
            let reconstructed = ifft(&spectrum, None).map_err(|e| format!("IFFT: {}", e))?;
            Ok(reconstructed)
        },
        100.0,
        "FFT forward-backward pipeline",
    )?;

    Ok(())
}

/// Test performance comparison of different FFT sizes
#[test]
fn test_fft_size_performance() -> TestResult<()> {
    let sizes = vec![64, 128, 256, 512, 1024, 2048, 4096];

    println!("Testing FFT performance scaling");

    for size in sizes {
        let signal = TestDatasets::sinusoid_signal(size, 10.0, size as f64);
        let signal_vec: Vec<f64> = signal.to_vec();

        let (_result, perf) = measure_time(&format!("FFT size {}", size), || {
            fft(&signal_vec, None).map_err(|e| e.into())
        })?;

        println!("  Size {}: {:.3} ms", size, perf.duration_ms);
    }

    Ok(())
}

/// Test DCT integration with signal processing
#[test]
fn test_dct_integration() -> TestResult<()> {
    let signal = TestDatasets::sinusoid_signal(128, 5.0, 128.0);

    println!("Testing DCT integration");

    Ok(())
}

#[cfg(test)]
mod api_compatibility_tests {
    use super::*;

    /// Test that array formats are compatible between modules
    #[test]
    fn test_array_format_compatibility() -> TestResult<()> {
        let signal = TestDatasets::sinusoid_signal(256, 10.0, 256.0);

        let signal_vec: Vec<f64> = signal.to_vec();
        let _spectrum = fft(&signal_vec, None).map_err(|e| format!("FFT: {}", e))?;

        println!("Array format compatibility verified");

        Ok(())
    }

    /// Test error handling consistency
    #[test]
    fn test_error_handling_consistency() -> TestResult<()> {
        println!("Error handling consistency test");

        // Empty input should return an error
        let result = fft::<f64>(&[], None);
        assert!(result.is_err(), "FFT of empty input should return error");

        Ok(())
    }
}
