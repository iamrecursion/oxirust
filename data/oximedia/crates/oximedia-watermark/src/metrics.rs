//! Audio quality metrics for watermark imperceptibility.
//!
//! This module provides objective quality metrics to assess
//! watermark imperceptibility and distortion.

use oxifft::Complex;

/// Quality metrics for watermarked audio.
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Signal-to-Noise Ratio in dB.
    pub snr_db: f32,
    /// Peak Signal-to-Noise Ratio in dB.
    pub psnr_db: f32,
    /// Segmental SNR in dB.
    pub seg_snr_db: f32,
    /// Perceptual Evaluation of Audio Quality (PEAQ) - Objective Difference Grade.
    pub odg: f32,
    /// Log Spectral Distance.
    pub lsd: f32,
    /// Weighted SNR based on psychoacoustic model.
    pub wsnr_db: f32,
}

/// Calculate comprehensive quality metrics.
#[must_use]
pub fn calculate_metrics(original: &[f32], watermarked: &[f32]) -> QualityMetrics {
    let snr_db = calculate_snr(original, watermarked);
    let psnr_db = calculate_psnr(original, watermarked);
    let seg_snr_db = calculate_segmental_snr(original, watermarked, 256);
    let odg = estimate_odg(original, watermarked);
    let lsd = calculate_lsd(original, watermarked, 2048);
    let wsnr_db = calculate_wsnr(original, watermarked, 2048);

    QualityMetrics {
        snr_db,
        psnr_db,
        seg_snr_db,
        odg,
        lsd,
        wsnr_db,
    }
}

/// Calculate Signal-to-Noise Ratio (SNR).
#[must_use]
pub fn calculate_snr(original: &[f32], watermarked: &[f32]) -> f32 {
    let n = original.len().min(watermarked.len());

    let signal_power: f32 = original.iter().take(n).map(|&s| s * s).sum();
    let noise_power: f32 = original
        .iter()
        .zip(watermarked.iter())
        .take(n)
        .map(|(&o, &w)| {
            let diff = o - w;
            diff * diff
        })
        .sum();

    if noise_power > 1e-10 {
        10.0 * (signal_power / noise_power).log10()
    } else {
        100.0
    }
}

/// Calculate Peak Signal-to-Noise Ratio (PSNR).
#[must_use]
pub fn calculate_psnr(original: &[f32], watermarked: &[f32]) -> f32 {
    let n = original.len().min(watermarked.len());

    let mse: f32 = original
        .iter()
        .zip(watermarked.iter())
        .take(n)
        .map(|(&o, &w)| {
            let diff = o - w;
            diff * diff
        })
        .sum::<f32>()
        / n as f32;

    if mse > 1e-10 {
        10.0 * (1.0 / mse).log10()
    } else {
        100.0
    }
}

/// Calculate Segmental SNR.
#[must_use]
pub fn calculate_segmental_snr(original: &[f32], watermarked: &[f32], frame_size: usize) -> f32 {
    let n = original.len().min(watermarked.len());
    let num_frames = n / frame_size;

    if num_frames == 0 {
        return calculate_snr(original, watermarked);
    }

    let mut sum_snr = 0.0f32;

    for i in 0..num_frames {
        let start = i * frame_size;
        let end = (start + frame_size).min(n);

        let signal_power: f32 = original[start..end].iter().map(|&s| s * s).sum();
        let noise_power: f32 = original[start..end]
            .iter()
            .zip(watermarked[start..end].iter())
            .map(|(&o, &w)| {
                let diff = o - w;
                diff * diff
            })
            .sum();

        if noise_power > 1e-10 {
            sum_snr += 10.0 * (signal_power / noise_power).log10();
        } else {
            sum_snr += 100.0;
        }
    }

    #[allow(clippy::cast_precision_loss)]
    let result = sum_snr / num_frames as f32;
    result
}

/// Estimate Objective Difference Grade (ODG) - simplified PEAQ.
#[must_use]
pub fn estimate_odg(original: &[f32], watermarked: &[f32]) -> f32 {
    // Simplified ODG estimation based on SNR
    // Real PEAQ implementation is much more complex
    let snr = calculate_snr(original, watermarked);

    // Map SNR to ODG scale (-4 to 0)
    // ODG: 0 = imperceptible, -1 = perceptible but not annoying,
    //      -2 = slightly annoying, -3 = annoying, -4 = very annoying
    if snr >= 60.0 {
        0.0
    } else if snr >= 45.0 {
        -0.5
    } else if snr >= 35.0 {
        -1.0
    } else if snr >= 25.0 {
        -2.0
    } else if snr >= 15.0 {
        -3.0
    } else {
        -4.0
    }
}

/// Calculate Log Spectral Distance.
#[must_use]
pub fn calculate_lsd(original: &[f32], watermarked: &[f32], frame_size: usize) -> f32 {
    let n = original.len().min(watermarked.len());
    if n < frame_size {
        return 0.0;
    }

    let num_frames = n / frame_size;

    let mut total_lsd = 0.0f32;

    for i in 0..num_frames {
        let start = i * frame_size;
        let end = start + frame_size;

        if end > n {
            break;
        }

        // FFT of original
        let orig_input: Vec<Complex<f32>> = original[start..end]
            .iter()
            .map(|&s| Complex::new(s, 0.0))
            .collect();
        let orig_fft = oxifft::fft(&orig_input);

        // FFT of watermarked
        let wm_input: Vec<Complex<f32>> = watermarked[start..end]
            .iter()
            .map(|&s| Complex::new(s, 0.0))
            .collect();
        let wm_fft = oxifft::fft(&wm_input);

        // Calculate LSD for this frame
        let mut frame_lsd = 0.0f32;
        let bins = frame_size / 2;

        for bin in 1..bins {
            let orig_mag = orig_fft[bin].norm().max(1e-10);
            let wm_mag = wm_fft[bin].norm().max(1e-10);
            let ratio = orig_mag / wm_mag;
            frame_lsd += ratio.log10().powi(2);
        }

        #[allow(clippy::cast_precision_loss)]
        let divisor = (bins - 1) as f32;
        total_lsd += (frame_lsd / divisor).sqrt();
    }

    #[allow(clippy::cast_precision_loss)]
    let result = total_lsd / num_frames as f32;
    result
}

/// Calculate Weighted SNR using psychoacoustic model.
#[must_use]
pub fn calculate_wsnr(original: &[f32], watermarked: &[f32], frame_size: usize) -> f32 {
    use crate::psychoacoustic::PsychoacousticModel;

    let n = original.len().min(watermarked.len());
    if n < frame_size {
        return calculate_snr(original, watermarked);
    }

    let model = PsychoacousticModel::new(44100, frame_size);
    let num_frames = n / frame_size;

    let mut weighted_signal_power = 0.0f32;
    let mut weighted_noise_power = 0.0f32;

    for i in 0..num_frames {
        let start = i * frame_size;
        let end = start + frame_size;

        if end > n {
            break;
        }

        // Calculate masking threshold
        let threshold = model.calculate_masking_threshold(&original[start..end]);

        // Weight signal and noise by inverse of masking threshold
        for (j, &thresh_db) in threshold.iter().enumerate() {
            if start + j >= n {
                break;
            }

            let weight = 10.0f32.powf(-thresh_db / 20.0);
            let signal = original[start + j];
            let noise = watermarked[start + j] - original[start + j];

            weighted_signal_power += signal * signal * weight;
            weighted_noise_power += noise * noise * weight;
        }
    }

    if weighted_noise_power > 1e-10 {
        10.0 * (weighted_signal_power / weighted_noise_power).log10()
    } else {
        100.0
    }
}

/// Calculate correlation coefficient.
#[must_use]
pub fn calculate_correlation(signal1: &[f32], signal2: &[f32]) -> f32 {
    let n = signal1.len().min(signal2.len());
    if n == 0 {
        return 0.0;
    }

    #[allow(clippy::cast_precision_loss)]
    let mean1: f32 = signal1.iter().take(n).sum::<f32>() / n as f32;
    #[allow(clippy::cast_precision_loss)]
    let mean2: f32 = signal2.iter().take(n).sum::<f32>() / n as f32;

    let mut sum = 0.0f32;
    let mut sum1_sq = 0.0f32;
    let mut sum2_sq = 0.0f32;

    for i in 0..n {
        let diff1 = signal1[i] - mean1;
        let diff2 = signal2[i] - mean2;

        sum += diff1 * diff2;
        sum1_sq += diff1 * diff1;
        sum2_sq += diff2 * diff2;
    }

    if sum1_sq > 1e-10 && sum2_sq > 1e-10 {
        sum / (sum1_sq.sqrt() * sum2_sq.sqrt())
    } else {
        0.0
    }
}

/// Calculate Mean Opinion Score (MOS) estimate.
#[must_use]
pub fn estimate_mos(metrics: &QualityMetrics) -> f32 {
    // Simplified MOS estimation based on ODG
    // Real MOS requires subjective listening tests
    match metrics.odg {
        odg if odg >= -0.5 => 5.0, // Excellent
        odg if odg >= -1.5 => 4.0, // Good
        odg if odg >= -2.5 => 3.0, // Fair
        odg if odg >= -3.5 => 2.0, // Poor
        _ => 1.0,                  // Bad
    }
}

/// Calculate perceptual loudness difference.
#[must_use]
pub fn calculate_loudness_difference(original: &[f32], watermarked: &[f32]) -> f32 {
    // Simplified loudness calculation
    let orig_rms = calculate_rms(original);
    let wm_rms = calculate_rms(watermarked);

    if orig_rms > 1e-10 {
        20.0 * (wm_rms / orig_rms).log10()
    } else {
        0.0
    }
}

/// Calculate RMS level.
fn calculate_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }

    #[allow(clippy::cast_precision_loss)]
    let mean_sq = samples.iter().map(|&s| s * s).sum::<f32>() / samples.len() as f32;
    mean_sq.sqrt()
}

/// Calculate spectral centroid.
#[must_use]
pub fn calculate_spectral_centroid(samples: &[f32], sample_rate: u32) -> f32 {
    let frame_size = samples.len().next_power_of_two();

    let freq_input: Vec<Complex<f32>> = samples
        .iter()
        .map(|&s| Complex::new(s, 0.0))
        .chain(std::iter::repeat(Complex::new(0.0, 0.0)))
        .take(frame_size)
        .collect();

    let freq_data = oxifft::fft(&freq_input);

    let mut weighted_sum = 0.0f32;
    let mut magnitude_sum = 0.0f32;

    for (i, c) in freq_data.iter().take(frame_size / 2).enumerate() {
        let mag = c.norm();
        #[allow(clippy::cast_precision_loss)]
        let freq = i as f32 * sample_rate as f32 / frame_size as f32;

        weighted_sum += freq * mag;
        magnitude_sum += mag;
    }

    if magnitude_sum > 1e-10 {
        weighted_sum / magnitude_sum
    } else {
        0.0
    }
}

/// Calculate spectral flatness measure.
#[must_use]
pub fn calculate_spectral_flatness(samples: &[f32]) -> f32 {
    let frame_size = samples.len().next_power_of_two();

    let freq_input: Vec<Complex<f32>> = samples
        .iter()
        .map(|&s| Complex::new(s, 0.0))
        .chain(std::iter::repeat(Complex::new(0.0, 0.0)))
        .take(frame_size)
        .collect();

    let freq_data = oxifft::fft(&freq_input);

    let n = frame_size / 2;
    let mut geometric_mean = 1.0f32;
    let mut arithmetic_mean = 0.0f32;

    #[allow(clippy::cast_precision_loss)]
    let divisor = 1.0 / (n - 1) as f32;
    for c in freq_data.iter().take(n).skip(1) {
        let mag = c.norm().max(1e-10);
        geometric_mean *= mag.powf(divisor);
        arithmetic_mean += mag * divisor;
    }

    if arithmetic_mean > 1e-10 {
        geometric_mean / arithmetic_mean
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snr_calculation() {
        let original: Vec<f32> = vec![1.0; 1000];
        let watermarked: Vec<f32> = original.iter().map(|&s| s * 1.01).collect();

        let snr = calculate_snr(&original, &watermarked);
        assert!(snr > 0.0);
    }

    #[test]
    fn test_psnr_calculation() {
        let original: Vec<f32> = vec![0.5; 1000];
        let watermarked: Vec<f32> = original.clone();

        let psnr = calculate_psnr(&original, &watermarked);
        assert!(psnr > 50.0);
    }

    #[test]
    fn test_segmental_snr() {
        let original: Vec<f32> = vec![0.5; 1024];
        let watermarked: Vec<f32> = original.iter().map(|&s| s + 0.01).collect();

        let seg_snr = calculate_segmental_snr(&original, &watermarked, 256);
        assert!(seg_snr > 0.0);
    }

    #[test]
    fn test_odg_estimation() {
        let original: Vec<f32> = vec![0.5; 1000];
        let watermarked: Vec<f32> = original.clone();

        let odg = estimate_odg(&original, &watermarked);
        assert!(odg >= -4.0 && odg <= 0.0);
    }

    #[test]
    fn test_comprehensive_metrics() {
        let original: Vec<f32> = vec![0.5; 2048];
        let watermarked: Vec<f32> = original.iter().map(|&s| s + 0.001).collect();

        let metrics = calculate_metrics(&original, &watermarked);
        assert!(metrics.snr_db > 0.0);
        assert!(metrics.odg >= -4.0);
    }

    #[test]
    fn test_correlation() {
        let signal1: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let signal2: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

        let corr = calculate_correlation(&signal1, &signal2);
        assert!((corr - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_mos_estimation() {
        let metrics = QualityMetrics {
            snr_db: 45.0,
            psnr_db: 50.0,
            seg_snr_db: 43.0,
            odg: -0.5,
            lsd: 0.1,
            wsnr_db: 46.0,
        };

        let mos = estimate_mos(&metrics);
        assert!(mos >= 1.0 && mos <= 5.0);
    }

    #[test]
    fn test_spectral_centroid() {
        let samples: Vec<f32> = (0..1000)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                (i as f32 * 0.1).sin()
            })
            .collect();

        let centroid = calculate_spectral_centroid(&samples, 44100);
        assert!(centroid > 0.0);
    }

    #[test]
    fn test_spectral_flatness() {
        let samples: Vec<f32> = vec![0.5; 1024];
        let flatness = calculate_spectral_flatness(&samples);
        assert!(flatness >= 0.0 && flatness <= 1.0);
    }
}
