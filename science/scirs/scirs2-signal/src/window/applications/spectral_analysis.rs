//! Window Functions for Spectral Analysis
//!
//! This module provides optimized window selection and application utilities
//! specifically designed for spectral analysis applications including:
//! - FFT-based spectral estimation
//! - Power spectral density estimation  
//! - Frequency domain filtering
//! - Time-frequency analysis

use super::super::families::{cosine, exponential, rectangular, specialized, triangular};
use crate::error::{SignalError, SignalResult};

/// Window selection criteria for spectral analysis
#[derive(Debug, Clone)]
pub struct SpectralWindowCriteria {
    /// Desired frequency resolution (Hz)
    pub frequency_resolution: Option<f64>,
    /// Maximum acceptable sidelobe level (dB)
    pub max_sidelobe_level: Option<f64>,
    /// Minimum processing gain required
    pub min_processing_gain: Option<f64>,
    /// Maximum scalloping loss acceptable (dB)
    pub max_scalloping_loss: Option<f64>,
    /// Main lobe width constraint (bins)
    pub main_lobe_width: Option<f64>,
    /// SNR of input signal (dB)
    pub input_snr: Option<f64>,
}

/// Recommended window for spectral analysis
#[derive(Debug, Clone)]
pub struct RecommendedWindow {
    /// Window name
    pub name: String,
    /// Window samples
    pub samples: Vec<f64>,
    /// Window properties
    pub properties: WindowProperties,
    /// Recommendation score (0-1, higher is better)
    pub score: f64,
    /// Rationale for recommendation
    pub rationale: String,
}

/// Window properties for spectral analysis
#[derive(Debug, Clone)]
pub struct WindowProperties {
    /// Coherent gain
    pub coherent_gain: f64,
    /// Processing gain
    pub processing_gain: f64,
    /// Scalloping loss (dB)
    pub scalloping_loss: f64,
    /// Main lobe width (bins)
    pub main_lobe_width: f64,
    /// Maximum sidelobe level (dB)
    pub max_sidelobe_level: f64,
    /// Equivalent noise bandwidth (ENBW)
    pub enbw: f64,
}

/// Window types optimized for spectral analysis
#[derive(Debug, Clone, PartialEq)]
pub enum SpectralWindowType {
    /// Rectangular (best frequency resolution)
    Rectangular,
    /// Hann (good general purpose)
    Hann,
    /// Hamming (good sidelobe suppression)
    Hamming,
    /// Blackman (excellent sidelobe suppression)
    Blackman,
    /// BlackmanHarris (superior sidelobe suppression)
    BlackmanHarris,
    /// Kaiser (adjustable parameters)
    Kaiser(f64), // beta parameter
    /// Gaussian (smooth characteristics)
    Gaussian(f64), // std parameter
    /// FlatTop (amplitude accuracy)
    FlatTop,
    /// Tukey (adjustable taper)
    Tukey(f64), // alpha parameter
}

/// Select optimal window for spectral analysis
///
/// Analyzes the given criteria and recommends the best window function
/// for spectral analysis applications.
///
/// # Arguments
/// * `length` - Window length in samples
/// * `criteria` - Selection criteria for optimization
/// * `sample_rate` - Sampling rate (Hz), optional for frequency calculations
///
/// # Returns
/// A recommended window with properties and rationale
pub fn select_spectral_window(
    length: usize,
    criteria: &SpectralWindowCriteria,
    sample_rate: Option<f64>,
) -> SignalResult<RecommendedWindow> {
    let candidates = generate_window_candidates(length)?;
    let mut scored_windows = Vec::new();

    for (window_type, samples) in candidates {
        let properties = analyze_window_properties(&samples, sample_rate);
        let score = score_window_for_criteria(&properties, criteria);
        let rationale = generate_rationale(&window_type, &properties, criteria);

        scored_windows.push(RecommendedWindow {
            name: format!("{:?}", window_type),
            samples,
            properties,
            score,
            rationale,
        });
    }

    // Sort by score (descending)
    scored_windows.sort_by(|a, b| b.score.partial_cmp(&a.score).expect("Operation failed"));

    scored_windows
        .into_iter()
        .next()
        .ok_or_else(|| SignalError::ValueError("No suitable window found".to_string()))
}

/// Generate candidate windows for comparison
fn generate_window_candidates(length: usize) -> SignalResult<Vec<(SpectralWindowType, Vec<f64>)>> {
    let mut candidates = Vec::new();

    // Standard windows
    candidates.push((
        SpectralWindowType::Rectangular,
        rectangular::boxcar(length, true)?,
    ));

    candidates.push((SpectralWindowType::Hann, cosine::hann(length, true)?));

    candidates.push((SpectralWindowType::Hamming, cosine::hamming(length, true)?));

    candidates.push((
        SpectralWindowType::Blackman,
        cosine::blackman(length, true)?,
    ));

    candidates.push((
        SpectralWindowType::BlackmanHarris,
        cosine::blackmanharris(length, true)?,
    ));

    candidates.push((SpectralWindowType::FlatTop, cosine::flattop(length, true)?));

    // Kaiser windows with different beta values
    for &beta in &[2.0, 5.0, 8.0] {
        candidates.push((
            SpectralWindowType::Kaiser(beta),
            exponential::kaiser(length, beta, true)?,
        ));
    }

    // Gaussian windows with different std values
    for &std in &[0.5, 1.0, 2.0] {
        candidates.push((
            SpectralWindowType::Gaussian(std),
            exponential::gaussian(length, std, true)?,
        ));
    }

    // Tukey windows with different alpha values
    for &alpha in &[0.25, 0.5, 0.75] {
        candidates.push((
            SpectralWindowType::Tukey(alpha),
            exponential::tukey(length, alpha, true)?,
        ));
    }

    Ok(candidates)
}

/// Analyze window properties for spectral analysis
fn analyze_window_properties(window: &[f64], sample_rate: Option<f64>) -> WindowProperties {
    let n = window.len();

    // Coherent gain (DC gain normalized)
    let coherent_gain = window.iter().sum::<f64>() / n as f64;

    // Processing gain (ratio of coherent to incoherent power)
    let sum_squares = window.iter().map(|&w| w * w).sum::<f64>();
    let processing_gain = window.iter().sum::<f64>().powi(2) / (n as f64 * sum_squares);

    // Equivalent Noise Bandwidth (ENBW)
    let enbw = n as f64 * sum_squares / window.iter().sum::<f64>().powi(2);

    // Estimate other properties (would require FFT for exact values)
    let (scalloping_loss, main_lobe_width, max_sidelobe_level) =
        estimate_frequency_properties(window);

    WindowProperties {
        coherent_gain,
        processing_gain,
        scalloping_loss,
        main_lobe_width,
        max_sidelobe_level,
        enbw,
    }
}

/// Estimate frequency domain properties
fn estimate_frequency_properties(window: &[f64]) -> (f64, f64, f64) {
    // These are approximations - exact values would require FFT analysis
    let coherent_gain = window.iter().sum::<f64>() / window.len() as f64;

    // Scalloping loss estimates based on window characteristics
    let scalloping_loss = match () {
        _ if is_rectangular_like(window) => 3.92,
        _ if is_hann_like(window) => 1.42,
        _ if is_hamming_like(window) => 1.78,
        _ if is_blackman_like(window) => 1.10,
        _ => 2.0, // Default estimate
    };

    // Main lobe width estimates
    let main_lobe_width = match () {
        _ if is_rectangular_like(window) => 2.0,
        _ if is_hann_like(window) => 4.0,
        _ if is_hamming_like(window) => 4.0,
        _ if is_blackman_like(window) => 6.0,
        _ => 4.0, // Default estimate
    };

    // Maximum sidelobe level estimates
    let max_sidelobe_level = match () {
        _ if is_rectangular_like(window) => -13.3,
        _ if is_hann_like(window) => -31.5,
        _ if is_hamming_like(window) => -42.7,
        _ if is_blackman_like(window) => -58.1,
        _ => -30.0, // Default estimate
    };

    (scalloping_loss, main_lobe_width, max_sidelobe_level)
}

/// Score window against criteria
fn score_window_for_criteria(
    properties: &WindowProperties,
    criteria: &SpectralWindowCriteria,
) -> f64 {
    let mut score = 1.0;
    let mut weight_sum = 0.0;

    // Sidelobe level criterion
    if let Some(max_sidelobe) = criteria.max_sidelobe_level {
        let penalty = if properties.max_sidelobe_level > max_sidelobe {
            0.0 // Fails hard constraint
        } else {
            1.0 - (max_sidelobe - properties.max_sidelobe_level) / 50.0
        };
        score *= penalty.max(0.0).min(1.0);
        weight_sum += 1.0;
    }

    // Processing gain criterion
    if let Some(min_gain) = criteria.min_processing_gain {
        let penalty = if properties.processing_gain < min_gain {
            0.0 // Fails hard constraint
        } else {
            properties.processing_gain / min_gain
        };
        score *= penalty.max(0.0).min(1.0);
        weight_sum += 1.0;
    }

    // Scalloping loss criterion
    if let Some(max_scalloping) = criteria.max_scalloping_loss {
        let penalty = if properties.scalloping_loss > max_scalloping {
            0.0 // Fails hard constraint
        } else {
            1.0 - properties.scalloping_loss / max_scalloping
        };
        score *= penalty.max(0.0).min(1.0);
        weight_sum += 1.0;
    }

    // Main lobe width criterion (lower is better for frequency resolution)
    if let Some(max_width) = criteria.main_lobe_width {
        let penalty = if properties.main_lobe_width > max_width {
            0.0 // Fails hard constraint
        } else {
            1.0 - properties.main_lobe_width / max_width
        };
        score *= penalty.max(0.0).min(1.0);
        weight_sum += 1.0;
    }

    // If no criteria specified, use default scoring
    if weight_sum == 0.0 {
        // Default: balance frequency resolution and sidelobe suppression
        let freq_resolution_score = 1.0 / (properties.main_lobe_width / 2.0);
        let sidelobe_score = (-properties.max_sidelobe_level / 60.0).min(1.0);
        score = 0.6 * freq_resolution_score + 0.4 * sidelobe_score;
    }

    score.max(0.0).min(1.0)
}

/// Generate rationale for window recommendation
fn generate_rationale(
    window_type: &SpectralWindowType,
    properties: &WindowProperties,
    criteria: &SpectralWindowCriteria,
) -> String {
    let mut rationale = Vec::new();

    match window_type {
        SpectralWindowType::Rectangular => {
            rationale.push("Best frequency resolution".to_string());
            rationale.push("High spectral leakage".to_string());
        }
        SpectralWindowType::Hann => {
            rationale.push("Good general-purpose window".to_string());
            rationale.push("Balanced frequency/amplitude trade-off".to_string());
        }
        SpectralWindowType::Hamming => {
            rationale.push("Better sidelobe suppression than Hann".to_string());
            rationale.push("Non-zero endpoints".to_string());
        }
        SpectralWindowType::Blackman => {
            rationale.push("Excellent sidelobe suppression".to_string());
            rationale.push("Wider main lobe".to_string());
        }
        SpectralWindowType::BlackmanHarris => {
            rationale.push("Superior sidelobe suppression".to_string());
            rationale.push("Best for high dynamic range".to_string());
        }
        SpectralWindowType::Kaiser(beta) => {
            rationale.push(format!("Adjustable parameters (β={:.1})", beta));
            rationale.push("Good trade-off control".to_string());
        }
        SpectralWindowType::Gaussian(std) => {
            rationale.push(format!("Smooth characteristics (σ={:.1})", std));
            rationale.push("Good time-frequency localization".to_string());
        }
        SpectralWindowType::FlatTop => {
            rationale.push("Excellent amplitude accuracy".to_string());
            rationale.push("Wide main lobe reduces frequency resolution".to_string());
        }
        SpectralWindowType::Tukey(alpha) => {
            rationale.push(format!("Tapered cosine (α={:.2})", alpha));
            rationale.push("Adjustable taper region".to_string());
        }
    }

    // Add performance metrics
    rationale.push(format!(
        "Processing gain: {:.3}",
        properties.processing_gain
    ));
    rationale.push(format!(
        "Scalloping loss: {:.2} dB",
        properties.scalloping_loss
    ));
    rationale.push(format!(
        "Max sidelobe: {:.1} dB",
        properties.max_sidelobe_level
    ));

    rationale.join("; ")
}

// Window classification helpers

fn is_rectangular_like(window: &[f64]) -> bool {
    let first = window.first().copied().unwrap_or(0.0);
    window.iter().all(|&x| (x - first).abs() < 0.01)
}

fn is_hann_like(window: &[f64]) -> bool {
    let n = window.len();
    if n < 3 {
        return false;
    }

    // Hann has zero endpoints and peak at center
    let endpoints_zero = window[0].abs() < 0.01 && window[n - 1].abs() < 0.01;
    let peak_at_center = window[n / 2] > 0.9;

    endpoints_zero && peak_at_center
}

fn is_hamming_like(window: &[f64]) -> bool {
    let n = window.len();
    if n < 3 {
        return false;
    }

    // Hamming has non-zero endpoints
    let nonzero_endpoints = window[0] > 0.05 && window[n - 1] > 0.05;
    let peak_at_center = window[n / 2] > 0.9;

    nonzero_endpoints && peak_at_center
}

fn is_blackman_like(window: &[f64]) -> bool {
    let n = window.len();
    if n < 5 {
        return false;
    }

    // Blackman has characteristic three-term cosine shape
    let zero_endpoints = window[0].abs() < 0.01 && window[n - 1].abs() < 0.01;
    let quarter_val = window[n / 4] / window[n / 2];

    zero_endpoints && quarter_val > 0.3 && quarter_val < 0.7
}

/// Apply spectral window with overlap processing
///
/// Applies windowing to overlapped segments for spectral analysis.
/// Commonly used in welch's method and spectrogram computation.
///
/// # Arguments
/// * `signal` - Input signal
/// * `window` - Window function to apply
/// * `nperseg` - Length of each segment
/// * `noverlap` - Number of overlapping samples
///
/// # Returns
/// Vector of windowed segments
pub fn apply_windowed_segments(
    signal: &[f64],
    window: &[f64],
    nperseg: usize,
    noverlap: usize,
) -> SignalResult<Vec<Vec<f64>>> {
    if window.len() != nperseg {
        return Err(SignalError::ValueError(
            "Window length must match segment length".to_string(),
        ));
    }

    if noverlap >= nperseg {
        return Err(SignalError::ValueError(
            "Overlap must be less than segment length".to_string(),
        ));
    }

    let step = nperseg - noverlap;
    let mut segments = Vec::new();

    let mut start = 0;
    while start + nperseg <= signal.len() {
        let segment: Vec<f64> = signal[start..start + nperseg]
            .iter()
            .zip(window.iter())
            .map(|(s, w)| s * w)
            .collect();
        segments.push(segment);
        start += step;
    }

    Ok(segments)
}

/// Compute power spectral density correction factors
///
/// Computes the scaling factors needed to correct PSD estimates
/// when using windowed data segments.
///
/// # Arguments
/// * `window` - Window function used
/// * `sample_rate` - Sampling rate in Hz
///
/// # Returns
/// Tuple of (amplitude correction, power correction)
pub fn compute_psd_corrections(window: &[f64], sample_rate: f64) -> (f64, f64) {
    let n = window.len();
    let sum_window = window.iter().sum::<f64>();
    let sum_window_squared = window.iter().map(|&w| w * w).sum::<f64>();

    // Amplitude correction factor
    let amplitude_correction = n as f64 / sum_window;

    // Power correction factor
    let power_correction = sample_rate * sum_window_squared;

    (amplitude_correction, power_correction)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_window_selection() {
        let criteria = SpectralWindowCriteria {
            frequency_resolution: None,
            max_sidelobe_level: Some(-40.0),
            min_processing_gain: None,
            max_scalloping_loss: None,
            main_lobe_width: None,
            input_snr: None,
        };

        let result = select_spectral_window(64, &criteria, Some(1000.0));
        assert!(result.is_ok());

        let window = result.expect("Operation failed");
        assert_eq!(window.samples.len(), 64);
        assert!(window.score >= 0.0 && window.score <= 1.0);
    }

    #[test]
    fn test_windowed_segments() {
        let signal: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let window = rectangular::boxcar(20, true).expect("Operation failed");

        let segments = apply_windowed_segments(&signal, &window, 20, 10).expect("Operation failed");
        assert!(!segments.is_empty());
        assert_eq!(segments[0].len(), 20);

        // Check overlap
        assert!(segments.len() > 1);
    }

    #[test]
    fn test_psd_corrections() {
        let window = cosine::hann(64, true).expect("Operation failed");
        let (amp_corr, pow_corr) = compute_psd_corrections(&window, 1000.0);

        assert!(amp_corr > 1.0); // Hann window needs amplitude boost
        assert!(pow_corr > 0.0); // Power correction should be positive
    }

    #[test]
    fn test_window_properties() {
        let window = cosine::hann(64, true).expect("Operation failed");
        let properties = analyze_window_properties(&window, Some(1000.0));

        assert!(properties.coherent_gain > 0.0);
        assert!(properties.processing_gain > 0.0);
        assert!(properties.enbw > 0.0);
        assert!(properties.scalloping_loss >= 0.0);
    }
}
