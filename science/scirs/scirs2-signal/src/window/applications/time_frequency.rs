//! Window Functions for Time-Frequency Analysis
//!
//! This module provides window functions optimized for time-frequency analysis
//! including spectrograms, short-time Fourier transforms, and wavelet analysis.

use super::super::families::{cosine, exponential, specialized, triangular};
use crate::error::{SignalError, SignalResult};
use std::f64::consts::PI;

/// Time-frequency analysis parameters
#[derive(Debug, Clone)]
pub struct TimeFrequencyParams {
    /// Window length in samples
    pub window_length: usize,
    /// Overlap between windows (0.0 to 1.0)
    pub overlap: f64,
    /// Frequency resolution requirement (Hz)
    pub freq_resolution: Option<f64>,
    /// Time resolution requirement (seconds)
    pub time_resolution: Option<f64>,
    /// Sampling rate (Hz)
    pub sample_rate: f64,
    /// Analysis type
    pub analysis_type: TimeFrequencyAnalysis,
}

/// Types of time-frequency analysis
#[derive(Debug, Clone, PartialEq)]
pub enum TimeFrequencyAnalysis {
    /// Short-Time Fourier Transform
    STFT,
    /// Spectrogram computation
    Spectrogram,
    /// Continuous Wavelet Transform
    CWT,
    /// Gabor Transform
    Gabor,
    /// Wigner-Ville Distribution
    WignerVille,
    /// Cohen's class distributions
    CohenClass,
}

/// Window recommendation for time-frequency analysis
#[derive(Debug, Clone)]
pub struct TimeFrequencyWindow {
    /// Window type
    pub window_type: TimeFrequencyWindowType,
    /// Window coefficients
    pub coefficients: Vec<f64>,
    /// Time-frequency resolution properties
    pub resolution_properties: ResolutionProperties,
    /// Uncertainty principle characteristics
    pub uncertainty_product: f64,
    /// Recommendation rationale
    pub rationale: String,
}

/// Window types for time-frequency analysis
#[derive(Debug, Clone, PartialEq)]
pub enum TimeFrequencyWindowType {
    /// Gaussian window (optimal uncertainty)
    Gaussian { std: f64 },
    /// Hann window (good general purpose)
    Hann,
    /// Hamming window
    Hamming,
    /// Blackman window
    Blackman,
    /// Kaiser window
    Kaiser { beta: f64 },
    /// Tukey window (adjustable taper)
    Tukey { alpha: f64 },
    /// Dpss window (multitaper)
    Dpss { nw: f64 },
    /// Morlet-like window
    MorletLike { sigma: f64 },
}

/// Time-frequency resolution properties
#[derive(Debug, Clone)]
pub struct ResolutionProperties {
    /// Time resolution (effective duration)
    pub time_resolution: f64,
    /// Frequency resolution (effective bandwidth)
    pub frequency_resolution: f64,
    /// Time centroid
    pub time_centroid: f64,
    /// Frequency centroid
    pub frequency_centroid: f64,
    /// Time variance
    pub time_variance: f64,
    /// Frequency variance
    pub frequency_variance: f64,
}

/// Recommend window for time-frequency analysis
///
/// Selects optimal window based on time-frequency trade-offs
///
/// # Arguments
/// * `params` - Analysis parameters and requirements
/// * `optimization` - Primary optimization criterion
///
/// # Returns
/// Recommended window with properties
pub fn recommend_timefreq_window(
    params: &TimeFrequencyParams,
    optimization: TimeFrequencyOptimization,
) -> SignalResult<TimeFrequencyWindow> {
    // Validate parameters
    validate_timefreq_params(params)?;

    // Select window type based on analysis and optimization
    let window_type = select_window_for_analysis(&params.analysis_type, optimization)?;

    // Generate window coefficients
    let coefficients = generate_timefreq_window(&window_type, params.window_length)?;

    // Analyze resolution properties
    let resolution_properties = analyze_timefreq_resolution(&coefficients, params.sample_rate);

    // Calculate uncertainty product
    let uncertainty_product = resolution_properties.time_variance.sqrt()
        * resolution_properties.frequency_variance.sqrt();

    let rationale = generate_timefreq_rationale(
        &window_type,
        &params.analysis_type,
        &resolution_properties,
        uncertainty_product,
    );

    Ok(TimeFrequencyWindow {
        window_type,
        coefficients,
        resolution_properties,
        uncertainty_product,
        rationale,
    })
}

/// Time-frequency optimization criteria
#[derive(Debug, Clone, PartialEq)]
pub enum TimeFrequencyOptimization {
    /// Minimize time-frequency uncertainty product
    MinimizeUncertainty,
    /// Optimize for time resolution
    OptimizeTimeResolution,
    /// Optimize for frequency resolution
    OptimizeFrequencyResolution,
    /// Balance time and frequency resolution
    BalancedResolution,
    /// Minimize spectral leakage
    MinimizeSpectralLeakage,
}

/// Validate time-frequency parameters
fn validate_timefreq_params(params: &TimeFrequencyParams) -> SignalResult<()> {
    if params.window_length == 0 {
        return Err(SignalError::ValueError(
            "Window length must be positive".to_string(),
        ));
    }

    if params.overlap < 0.0 || params.overlap >= 1.0 {
        return Err(SignalError::ValueError(
            "Overlap must be between 0 and 1".to_string(),
        ));
    }

    if params.sample_rate <= 0.0 {
        return Err(SignalError::ValueError(
            "Sample rate must be positive".to_string(),
        ));
    }

    Ok(())
}

/// Select window for specific analysis type
fn select_window_for_analysis(
    analysis_type: &TimeFrequencyAnalysis,
    optimization: TimeFrequencyOptimization,
) -> SignalResult<TimeFrequencyWindowType> {
    match (analysis_type, optimization.clone()) {
        (TimeFrequencyAnalysis::STFT, TimeFrequencyOptimization::MinimizeUncertainty) => {
            Ok(TimeFrequencyWindowType::Gaussian { std: 1.0 })
        }

        (TimeFrequencyAnalysis::STFT, TimeFrequencyOptimization::BalancedResolution) => {
            Ok(TimeFrequencyWindowType::Hann)
        }

        (
            TimeFrequencyAnalysis::Spectrogram,
            TimeFrequencyOptimization::MinimizeSpectralLeakage,
        ) => Ok(TimeFrequencyWindowType::Blackman),

        (TimeFrequencyAnalysis::Gabor, _) => Ok(TimeFrequencyWindowType::Gaussian { std: 1.0 }),

        (TimeFrequencyAnalysis::CWT, _) => Ok(TimeFrequencyWindowType::MorletLike { sigma: 1.0 }),

        (TimeFrequencyAnalysis::WignerVille, _) => {
            Ok(TimeFrequencyWindowType::Gaussian { std: 0.8 })
        }

        (TimeFrequencyAnalysis::CohenClass, _) => Ok(TimeFrequencyWindowType::Kaiser { beta: 6.0 }),

        _ => {
            // Default selections based on optimization
            match optimization {
                TimeFrequencyOptimization::MinimizeUncertainty => {
                    Ok(TimeFrequencyWindowType::Gaussian { std: 1.0 })
                }
                TimeFrequencyOptimization::OptimizeTimeResolution => {
                    Ok(TimeFrequencyWindowType::Tukey { alpha: 0.25 })
                }
                TimeFrequencyOptimization::OptimizeFrequencyResolution => {
                    Ok(TimeFrequencyWindowType::Kaiser { beta: 2.0 })
                }
                TimeFrequencyOptimization::BalancedResolution => Ok(TimeFrequencyWindowType::Hann),
                TimeFrequencyOptimization::MinimizeSpectralLeakage => {
                    Ok(TimeFrequencyWindowType::Blackman)
                }
            }
        }
    }
}

/// Generate time-frequency window
fn generate_timefreq_window(
    window_type: &TimeFrequencyWindowType,
    length: usize,
) -> SignalResult<Vec<f64>> {
    match window_type {
        TimeFrequencyWindowType::Gaussian { std } => exponential::gaussian(length, *std, true),
        TimeFrequencyWindowType::Hann => cosine::hann(length, true),
        TimeFrequencyWindowType::Hamming => cosine::hamming(length, true),
        TimeFrequencyWindowType::Blackman => cosine::blackman(length, true),
        TimeFrequencyWindowType::Kaiser { beta } => exponential::kaiser(length, *beta, true),
        TimeFrequencyWindowType::Tukey { alpha } => exponential::tukey(length, *alpha, true),
        TimeFrequencyWindowType::Dpss { nw } => specialized::dpss_approximation(length, *nw, true),
        TimeFrequencyWindowType::MorletLike { sigma } => {
            generate_morlet_like_window(length, *sigma)
        }
    }
}

/// Generate Morlet-like window for wavelet analysis
fn generate_morlet_like_window(length: usize, sigma: f64) -> SignalResult<Vec<f64>> {
    if sigma <= 0.0 {
        return Err(SignalError::ValueError(
            "Sigma must be positive".to_string(),
        ));
    }

    let center = (length - 1) as f64 / 2.0;
    let mut window = Vec::with_capacity(length);

    for i in 0..length {
        let t = (i as f64 - center) / sigma;
        // Morlet-like: Gaussian envelope * complex exponential (real part)
        let envelope = (-t * t / 2.0).exp();
        let oscillation = (2.0 * PI * t).cos(); // Central frequency = 1
        let w_val = envelope * oscillation;
        window.push(w_val);
    }

    // Normalize to unit energy
    let energy: f64 = window.iter().map(|&w| w * w).sum();
    let norm_factor = energy.sqrt();

    if norm_factor > 0.0 {
        for w in &mut window {
            *w /= norm_factor;
        }
    }

    Ok(window)
}

/// Analyze time-frequency resolution properties
fn analyze_timefreq_resolution(window: &[f64], sample_rate: f64) -> ResolutionProperties {
    let n = window.len();
    let dt = 1.0 / sample_rate;

    // Time domain analysis
    let total_energy: f64 = window.iter().map(|&w| w * w).sum();

    // Time centroid
    let time_centroid = window
        .iter()
        .enumerate()
        .map(|(i, &w)| i as f64 * w * w)
        .sum::<f64>()
        / total_energy;

    // Time variance
    let time_variance = window
        .iter()
        .enumerate()
        .map(|(i, &w)| {
            let t_diff = i as f64 - time_centroid;
            t_diff * t_diff * w * w
        })
        .sum::<f64>()
        / total_energy;

    // Time resolution (effective duration)
    let time_resolution = (time_variance.sqrt() * dt) * 2.0; // ±1σ width

    // Frequency domain analysis (approximated)
    let frequency_variance = estimate_frequency_variance(window);
    let frequency_resolution = (frequency_variance.sqrt() * sample_rate / n as f64) * 2.0;

    ResolutionProperties {
        time_resolution,
        frequency_resolution,
        time_centroid: time_centroid * dt,
        frequency_centroid: 0.0, // Assuming real window
        time_variance: time_variance * dt * dt,
        frequency_variance,
    }
}

/// Estimate frequency variance from window shape
fn estimate_frequency_variance(window: &[f64]) -> f64 {
    let n = window.len();
    if n < 3 {
        return 1.0;
    }

    // Approximate frequency variance based on window curvature
    let mut curvature_sum = 0.0;
    let total_energy: f64 = window.iter().map(|&w| w * w).sum();

    for i in 1..(n - 1) {
        let second_derivative = window[i - 1] - 2.0 * window[i] + window[i + 1];
        curvature_sum += second_derivative.abs() * window[i] * window[i];
    }

    // Scale to frequency domain
    (curvature_sum / total_energy) * (n as f64).powi(2) / (4.0 * PI * PI)
}

/// Generate rationale for window selection
fn generate_timefreq_rationale(
    window_type: &TimeFrequencyWindowType,
    analysis_type: &TimeFrequencyAnalysis,
    properties: &ResolutionProperties,
    uncertainty_product: f64,
) -> String {
    let mut rationale = Vec::new();

    // Analysis-specific rationale
    match analysis_type {
        TimeFrequencyAnalysis::STFT => {
            rationale.push("STFT analysis - balance time/frequency resolution".to_string());
        }
        TimeFrequencyAnalysis::Spectrogram => {
            rationale.push("Spectrogram - optimize visual clarity and resolution".to_string());
        }
        TimeFrequencyAnalysis::Gabor => {
            rationale.push("Gabor analysis - optimal uncertainty principle".to_string());
        }
        TimeFrequencyAnalysis::CWT => {
            rationale.push("Wavelet analysis - scale-adaptive resolution".to_string());
        }
        _ => {
            rationale.push("Time-frequency analysis".to_string());
        }
    }

    // Window-specific properties
    match window_type {
        TimeFrequencyWindowType::Gaussian { .. } => {
            rationale.push("Gaussian window achieves minimum uncertainty product".to_string());
            rationale.push("Optimal for Gabor and wavelet analysis".to_string());
        }
        TimeFrequencyWindowType::Hann => {
            rationale.push("Hann window provides good balance and zero endpoints".to_string());
        }
        TimeFrequencyWindowType::Blackman => {
            rationale.push("Blackman window minimizes spectral leakage".to_string());
        }
        TimeFrequencyWindowType::Kaiser { beta } => {
            rationale.push(format!(
                "Kaiser window (β={:.1}) adjustable characteristics",
                beta
            ));
        }
        TimeFrequencyWindowType::MorletLike { .. } => {
            rationale.push("Morlet-like window optimal for wavelet analysis".to_string());
        }
        _ => {}
    }

    // Performance metrics
    rationale.push(format!(
        "Time resolution: {:.3} ms",
        properties.time_resolution * 1000.0
    ));
    rationale.push(format!(
        "Frequency resolution: {:.2} Hz",
        properties.frequency_resolution
    ));
    rationale.push(format!("Uncertainty product: {:.3}", uncertainty_product));

    // Uncertainty principle assessment
    let gaussian_limit = 0.5 / (2.0 * PI); // Minimum for Gaussian
    if uncertainty_product <= gaussian_limit * 1.1 {
        rationale.push("Near-optimal uncertainty product".to_string());
    } else if uncertainty_product <= gaussian_limit * 2.0 {
        rationale.push("Good uncertainty product".to_string());
    } else {
        rationale.push("Higher uncertainty product - consider Gaussian window".to_string());
    }

    rationale.join("; ")
}

/// Compute optimal window parameters for STFT
///
/// Determines optimal window length and overlap for STFT analysis
/// based on signal characteristics and analysis requirements.
///
/// # Arguments
/// * `signal_duration` - Total signal duration (seconds)
/// * `freq_content` - Expected frequency content range (Hz)
/// * `time_resolution_req` - Required time resolution (seconds)
/// * `freq_resolution_req` - Required frequency resolution (Hz)
/// * `sample_rate` - Sampling rate (Hz)
///
/// # Returns
/// Optimal STFT parameters
pub fn compute_optimal_stft_params(
    signal_duration: f64,
    freq_content: (f64, f64),
    time_resolution_req: f64,
    freq_resolution_req: f64,
    sample_rate: f64,
) -> SignalResult<TimeFrequencyParams> {
    // Window length based on frequency resolution
    let window_length_freq = (sample_rate / freq_resolution_req) as usize;

    // Window length based on time resolution
    let window_length_time = (time_resolution_req * sample_rate) as usize;

    // Choose compromise or user preference
    let window_length = window_length_freq.min(window_length_time * 4).max(32);

    // Optimal overlap for good time resolution (typically 50-75%)
    let overlap = if time_resolution_req < 0.01 {
        0.75
    } else {
        0.5
    };

    // Analysis type
    let analysis_type = TimeFrequencyAnalysis::STFT;

    Ok(TimeFrequencyParams {
        window_length,
        overlap,
        freq_resolution: Some(freq_resolution_req),
        time_resolution: Some(time_resolution_req),
        sample_rate,
        analysis_type,
    })
}

/// Generate multitaper windows for spectral analysis
///
/// Creates multiple orthogonal windows for multitaper spectral estimation
/// which reduces variance compared to single-window methods.
///
/// # Arguments
/// * `length` - Window length
/// * `nw` - Time-bandwidth product
/// * `num_tapers` - Number of tapers to generate
///
/// # Returns
/// Vector of orthogonal window sequences
pub fn generate_multitaper_windows(
    length: usize,
    nw: f64,
    num_tapers: usize,
) -> SignalResult<Vec<Vec<f64>>> {
    if num_tapers == 0 {
        return Err(SignalError::ValueError(
            "Number of tapers must be positive".to_string(),
        ));
    }

    if nw <= 0.0 {
        return Err(SignalError::ValueError(
            "Time-bandwidth product must be positive".to_string(),
        ));
    }

    let mut tapers = Vec::with_capacity(num_tapers);

    // Generate DPSS approximations with different parameters
    for k in 0..num_tapers {
        let adjusted_nw = nw + k as f64 * 0.1; // Slight variation for orthogonality
        let taper = specialized::dpss_approximation(length, adjusted_nw, true)?;
        tapers.push(taper);
    }

    // Apply Gram-Schmidt orthogonalization
    orthogonalize_tapers(&mut tapers);

    Ok(tapers)
}

/// Orthogonalize taper sequences using Gram-Schmidt process
fn orthogonalize_tapers(tapers: &mut [Vec<f64>]) {
    if tapers.is_empty() {
        return;
    }

    // Normalize the first taper
    let norm0: f64 = tapers[0].iter().map(|&x| x * x).sum::<f64>().sqrt();
    if norm0 > 1e-12 {
        for x in &mut tapers[0] {
            *x /= norm0;
        }
    }

    // Orthogonalize and normalize remaining tapers
    for i in 1..tapers.len() {
        for j in 0..i {
            // Compute projection coefficient
            let dot_product: f64 = tapers[i]
                .iter()
                .zip(tapers[j].iter())
                .map(|(a, b)| a * b)
                .sum();

            let norm_j_squared: f64 = tapers[j].iter().map(|&x| x * x).sum();

            if norm_j_squared > 1e-12 {
                let projection_coeff = dot_product / norm_j_squared;

                // Subtract projection
                for k in 0..tapers[i].len() {
                    tapers[i][k] -= projection_coeff * tapers[j][k];
                }
            }
        }

        // Normalize
        let norm: f64 = tapers[i].iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm > 1e-12 {
            for x in &mut tapers[i] {
                *x /= norm;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timefreq_window_recommendation() {
        let params = TimeFrequencyParams {
            window_length: 256,
            overlap: 0.5,
            freq_resolution: Some(10.0),
            time_resolution: Some(0.01),
            sample_rate: 1000.0,
            analysis_type: TimeFrequencyAnalysis::STFT,
        };

        let result =
            recommend_timefreq_window(&params, TimeFrequencyOptimization::BalancedResolution);
        assert!(result.is_ok());

        let window = result.expect("Operation failed");
        assert_eq!(window.coefficients.len(), 256);
        assert!(window.uncertainty_product > 0.0);
    }

    #[test]
    fn test_morlet_like_window() {
        let window = generate_morlet_like_window(64, 1.0).expect("Operation failed");
        assert_eq!(window.len(), 64);

        // Should be approximately normalized
        let energy: f64 = window.iter().map(|&w| w * w).sum();
        assert!((energy - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_stft_params_computation() {
        let params = compute_optimal_stft_params(
            1.0,           // 1 second signal
            (50.0, 500.0), // 50-500 Hz content
            0.01,          // 10ms time resolution
            5.0,           // 5 Hz frequency resolution
            1000.0,        // 1000 Hz sample rate
        )
        .expect("Operation failed");

        assert!(params.window_length >= 32);
        assert!(params.overlap > 0.0 && params.overlap < 1.0);
        assert_eq!(params.analysis_type, TimeFrequencyAnalysis::STFT);
    }

    #[test]
    fn test_multitaper_generation() {
        let tapers = generate_multitaper_windows(64, 2.5, 3).expect("Operation failed");
        assert_eq!(tapers.len(), 3);

        for taper in &tapers {
            assert_eq!(taper.len(), 64);

            // Check normalization
            let energy: f64 = taper.iter().map(|&x| x * x).sum();
            assert!((energy - 1.0).abs() < 0.3); // Relaxed tolerance for multitaper normalization
        }

        // Check approximate orthogonality
        let dot_product: f64 = tapers[0]
            .iter()
            .zip(tapers[1].iter())
            .map(|(a, b)| a * b)
            .sum();
        assert!(dot_product.abs() < 0.1); // Should be small for orthogonal
    }

    #[test]
    fn test_resolution_analysis() {
        let window = cosine::hann(128, true).expect("Operation failed");
        let properties = analyze_timefreq_resolution(&window, 1000.0);

        assert!(properties.time_resolution > 0.0);
        assert!(properties.frequency_resolution > 0.0);
        assert!(properties.time_variance > 0.0);
        assert!(properties.frequency_variance > 0.0);
    }

    #[test]
    fn test_invalid_params() {
        let invalid_params = TimeFrequencyParams {
            window_length: 0, // Invalid
            overlap: 0.5,
            freq_resolution: None,
            time_resolution: None,
            sample_rate: 1000.0,
            analysis_type: TimeFrequencyAnalysis::STFT,
        };

        assert!(validate_timefreq_params(&invalid_params).is_err());
    }
}
