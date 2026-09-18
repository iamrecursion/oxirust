//! Specialized Window Family
//!
//! This module implements specialized window functions including:
//! - Slepian windows (DPSS - Discrete Prolate Spheroidal Sequences)
//! - Chebwin (Dolph-Chebyshev window)
//! - Ultraspherical windows
//! - Planck-taper windows
//! - Custom optimized windows

use super::super::{_extend, _len_guards, _truncate};
use crate::error::{SignalError, SignalResult};
use std::f64::consts::PI;

/// Bohman window
///
/// The Bohman window is defined as the convolution of two half-duration
/// cosine lobes. It has good time localization properties.
///
/// # Arguments
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
/// A `Vec<f64>` of window values
///
/// # Examples
/// ```
/// use scirs2_signal::window::families::specialized::bohman;
/// let window = bohman(10, true).expect("Operation failed");
/// assert_eq!(window.len(), 10);
/// ```
pub fn bohman(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let frac = (i as f64 - (n - 1) as f64 / 2.0).abs() / ((n - 1) as f64 / 2.0);

        let w_val = if frac <= 1.0 {
            (1.0 - frac) * (PI * frac).cos() + (1.0 / PI) * (PI * frac).sin()
        } else {
            0.0
        };
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Poisson window
///
/// The Poisson window is an exponential window with parameter alpha.
/// It provides good frequency resolution with exponential decay.
///
/// # Arguments
/// * `m` - Number of points in the output window
/// * `alpha` - Exponential decay parameter (larger = faster decay)
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
/// A `Vec<f64>` of window values
pub fn poisson(m: usize, alpha: f64, sym: bool) -> SignalResult<Vec<f64>> {
    if alpha < 0.0 {
        return Err(SignalError::ValueError(
            "Alpha parameter must be non-negative".to_string(),
        ));
    }

    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);
    let center = (n - 1) as f64 / 2.0;

    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let distance = (i as f64 - center).abs();
        let w_val = (-alpha * distance / center).exp();
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Planck-taper window
///
/// The Planck-taper window uses the Planck function for tapering.
/// It provides smooth transitions with adjustable taper length.
///
/// # Arguments
/// * `m` - Number of points in the output window
/// * `epsilon` - Taper parameter (0 < epsilon < 0.5)
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
/// A `Vec<f64>` of window values
pub fn planck_taper(m: usize, epsilon: f64, sym: bool) -> SignalResult<Vec<f64>> {
    if epsilon <= 0.0 || epsilon >= 0.5 {
        return Err(SignalError::ValueError(
            "Epsilon must be between 0 and 0.5".to_string(),
        ));
    }

    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);
    let taper_length = (epsilon * n as f64) as usize;

    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let w_val = if i < taper_length {
            // Left taper using Planck function
            let x = (i as f64 / taper_length as f64 - 1.0) / (i as f64 / taper_length as f64);
            if x.is_finite() && x != 0.0 {
                1.0 / ((-1.0 / x).exp() + 1.0)
            } else {
                0.0
            }
        } else if i >= n - taper_length {
            // Right taper using Planck function
            let idx = n - 1 - i;
            let x = (idx as f64 / taper_length as f64 - 1.0) / (idx as f64 / taper_length as f64);
            if x.is_finite() && x != 0.0 {
                1.0 / ((-1.0 / x).exp() + 1.0)
            } else {
                0.0
            }
        } else {
            // Constant middle section
            1.0
        };
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// General cosine window with arbitrary number of terms
///
/// Creates a general cosine window with user-specified coefficients.
/// This is a generalization that can create many standard windows.
///
/// # Arguments
/// * `m` - Number of points in the output window
/// * `coeffs` - Cosine coefficients [a0, a1, a2, ...]
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
/// A `Vec<f64>` of window values
///
/// # Examples
/// ```
/// // Create a Hamming window using general cosine
/// use scirs2_signal::window::families::specialized::general_cosine;
/// let coeffs = vec![0.54, -0.46]; // Hamming coefficients
/// let window = general_cosine(10, &coeffs, true).expect("Operation failed");
/// ```
pub fn general_cosine(m: usize, coeffs: &[f64], sym: bool) -> SignalResult<Vec<f64>> {
    if coeffs.is_empty() {
        return Err(SignalError::ValueError(
            "At least one coefficient must be provided".to_string(),
        ));
    }

    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let mut w_val = 0.0;
        // scipy uses linspace(-π, π, M), so angle goes from -π to π
        // This is equivalent to 2π(i/(n-1) - 0.5) = 2πi/(n-1) - π
        let angle = 2.0 * PI * i as f64 / (n - 1) as f64 - PI;

        // No sign alternation - coefficients contain the signs
        // w[n] = sum_k a_k * cos(k * angle)
        for (k, &coeff) in coeffs.iter().enumerate() {
            w_val += coeff * (k as f64 * angle).cos();
        }
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// DPSS (Slepian) window approximation
///
/// Discrete Prolate Spheroidal Sequence window, optimized for concentration
/// in frequency domain. This is a simplified approximation.
///
/// # Arguments
/// * `m` - Number of points in the output window
/// * `nw` - Time-bandwidth product (typically 2.5 to 4)
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
/// A `Vec<f64>` of window values
pub fn dpss_approximation(m: usize, nw: f64, sym: bool) -> SignalResult<Vec<f64>> {
    if nw <= 0.0 {
        return Err(SignalError::ValueError(
            "Time-bandwidth product NW must be positive".to_string(),
        ));
    }

    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    // Simplified DPSS approximation using Kaiser window
    let beta = PI * (2.0 * nw - 1.0).sqrt();
    let mut w = Vec::with_capacity(n);
    let i0_beta = modified_bessel_i0(beta);
    let alpha = (n - 1) as f64 / 2.0;

    for i in 0..n {
        let x = (i as f64 - alpha) / alpha;
        let arg = beta * (1.0 - x * x).sqrt();
        let w_val = modified_bessel_i0(arg) / i0_beta;
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Ultraspherical window
///
/// A family of windows based on ultraspherical polynomials.
/// Provides good control over sidelobe behavior.
///
/// # Arguments
/// * `m` - Number of points in the output window
/// * `alpha` - Ultraspherical parameter (>= 0)
/// * `beta` - Window shaping parameter
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
/// A `Vec<f64>` of window values
pub fn ultraspherical(m: usize, alpha: f64, beta: f64, sym: bool) -> SignalResult<Vec<f64>> {
    if alpha < 0.0 {
        return Err(SignalError::ValueError(
            "Alpha parameter must be non-negative".to_string(),
        ));
    }

    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let x = 2.0 * i as f64 / (n - 1) as f64 - 1.0; // Map to [-1, 1]

        // Simplified ultraspherical polynomial approximation
        let w_val = if alpha == 0.0 {
            // Chebyshev case (alpha = 0)
            (beta * x.acos()).cos()
        } else {
            // General ultraspherical case (simplified)
            let term1 = (1.0 - x * x).powf(alpha);
            let term2 = (beta * x).cos();
            term1 * term2
        };

        w.push(w_val.abs()); // Ensure non-negative
    }

    // Normalize to peak of 1.0
    if let Some(&max_val) = w
        .iter()
        .max_by(|a, b| a.partial_cmp(b).expect("Operation failed"))
    {
        if max_val > 0.0 {
            for val in &mut w {
                *val /= max_val;
            }
        }
    }

    Ok(_truncate(w, needs_trunc))
}

/// Riesz window
///
/// The Riesz window is a family of windows with adjustable decay rate.
/// It provides a balance between time and frequency resolution.
///
/// # Arguments
/// * `m` - Number of points in the output window
/// * `alpha` - Decay parameter (>= 0)
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
/// A `Vec<f64>` of window values
pub fn riesz(m: usize, alpha: f64, sym: bool) -> SignalResult<Vec<f64>> {
    if alpha < 0.0 {
        return Err(SignalError::ValueError(
            "Alpha parameter must be non-negative".to_string(),
        ));
    }

    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);
    let center = (n - 1) as f64 / 2.0;

    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let x = (i as f64 - center) / center; // Normalize to [-1, 1]
        let w_val = (1.0 - x * x).powf(alpha);
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// De la Vallée Poussin window
///
/// A smooth window function that provides good frequency characteristics
/// with smooth transitions at the boundaries.
///
/// # Arguments
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
/// A `Vec<f64>` of window values
pub fn de_la_vallee_poussin(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let x = 2.0 * i as f64 / (n - 1) as f64 - 1.0; // Map to [-1, 1]

        let w_val = if x.abs() <= 0.5 {
            1.0
        } else if x.abs() <= 1.0 {
            2.0 * (1.0 - x.abs()).powi(2)
        } else {
            0.0
        };
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Analyze specialized window properties
///
/// Computes important properties of specialized windows for analysis
pub fn analyze_specialized_window(window: &[f64]) -> SpecializedWindowAnalysis {
    let n = window.len();

    // Basic properties
    let peak_value = window.iter().fold(0.0_f64, |a, &b| a.max(b));
    let min_value = window.iter().fold(1.0_f64, |a, &b| a.min(b));

    // Coherent gain
    let coherent_gain = window.iter().sum::<f64>() / n as f64;

    // Processing gain
    let sum_squares = window.iter().map(|&w| w * w).sum::<f64>();
    let processing_gain = window.iter().sum::<f64>().powi(2) / (n as f64 * sum_squares);

    // Dynamic range
    let dynamic_range = if min_value > 0.0 {
        20.0 * (peak_value / min_value).log10()
    } else {
        f64::INFINITY
    };

    // Effective bandwidth (3dB width approximation)
    let effective_bandwidth = estimate_effective_bandwidth(window);

    // Classify window type
    let window_type = classify_specialized_window(window);

    SpecializedWindowAnalysis {
        window_type,
        peak_value,
        min_value,
        coherent_gain,
        processing_gain,
        dynamic_range,
        effective_bandwidth,
    }
}

/// Specialized window analysis results
#[derive(Debug, Clone)]
pub struct SpecializedWindowAnalysis {
    /// Detected window type
    pub window_type: SpecializedWindowType,
    /// Peak value of the window
    pub peak_value: f64,
    /// Minimum value of the window
    pub min_value: f64,
    /// Coherent gain
    pub coherent_gain: f64,
    /// Processing gain
    pub processing_gain: f64,
    /// Dynamic range in dB
    pub dynamic_range: f64,
    /// Effective bandwidth estimate
    pub effective_bandwidth: f64,
}

/// Types of specialized windows
#[derive(Debug, Clone, PartialEq)]
pub enum SpecializedWindowType {
    /// Bohman window
    Bohman,
    /// Poisson window
    Poisson,
    /// Planck-taper window
    PlanckTaper,
    /// DPSS (Slepian) window
    Dpss,
    /// Ultraspherical window
    Ultraspherical,
    /// Riesz window
    Riesz,
    /// De la Vallée Poussin window
    DelaValleePoussin,
    /// Other/unknown specialized window
    Other,
}

// Helper functions

/// Modified Bessel function of the first kind, order 0
fn modified_bessel_i0(x: f64) -> f64 {
    let t = x / 3.75;
    if x.abs() < 3.75 {
        let y = t * t;
        1.0 + y
            * (3.515623
                + y * (3.089943 + y * (1.20675 + y * (0.265973 + y * (0.0360768 + y * 0.0045813)))))
    } else {
        let y = 1.0 / t;
        (x.abs().exp() / x.abs().sqrt())
            * (0.39894228
                + y * (0.01328592
                    + y * (0.00225319
                        + y * (-0.00157565
                            + y * (0.00916281
                                + y * (-0.02057706
                                    + y * (0.02635537 + y * (-0.01647633 + y * 0.00392377))))))))
    }
}

/// Estimate effective bandwidth of window
fn estimate_effective_bandwidth(window: &[f64]) -> f64 {
    let n = window.len();
    let peak_value = window.iter().fold(0.0_f64, |a, &b| a.max(b));
    let threshold = peak_value * 0.707; // -3dB point

    let mut left_edge = 0;
    let mut right_edge = n - 1;

    // Find -3dB points
    for i in 0..(n / 2) {
        if window[i] >= threshold {
            left_edge = i;
            break;
        }
    }

    for i in ((n / 2)..n).rev() {
        if window[i] >= threshold {
            right_edge = i;
            break;
        }
    }

    (right_edge - left_edge + 1) as f64 / n as f64
}

/// Classify the type of specialized window
fn classify_specialized_window(window: &[f64]) -> SpecializedWindowType {
    let n = window.len();
    if n < 3 {
        return SpecializedWindowType::Other;
    }

    // Check for flat regions (suggests Planck-taper or DPSS)
    let has_flat_region = has_approximately_flat_region(window);

    if has_flat_region {
        // Could be Planck-taper or DPSS
        if has_sharp_edges(window) {
            SpecializedWindowType::PlanckTaper
        } else {
            SpecializedWindowType::Dpss
        }
    } else {
        // Check for other characteristics
        let endpoint_ratio = window[0] / window[n / 2].max(1e-10);

        if endpoint_ratio < 0.1 {
            // Sharp falloff - could be Poisson or exponential-like
            SpecializedWindowType::Poisson
        } else if is_polynomial_like(window) {
            SpecializedWindowType::Riesz
        } else {
            SpecializedWindowType::Other
        }
    }
}

/// Check if window has approximately flat region
fn has_approximately_flat_region(window: &[f64]) -> bool {
    let n = window.len();
    let center_start = n / 3;
    let center_end = 2 * n / 3;

    if center_end <= center_start {
        return false;
    }

    let center_region = &window[center_start..center_end];
    let max_variation = center_region
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(0.0_f64, |a, b| a.max(b));

    max_variation < 0.05 // Less than 5% variation
}

/// Check if window has sharp edges
fn has_sharp_edges(window: &[f64]) -> bool {
    let n = window.len();
    if n < 4 {
        return false;
    }

    // Check left edge
    let left_gradient = window[2] - window[0];
    let right_gradient = window[n - 1] - window[n - 3];

    left_gradient.abs() > 0.5 || right_gradient.abs() > 0.5
}

/// Check if window follows polynomial pattern
fn is_polynomial_like(window: &[f64]) -> bool {
    let n = window.len();
    if n < 5 {
        return false;
    }

    // Check if it follows (1-x^2)^alpha pattern
    let center = n / 2;
    let quarter = n / 4;

    if quarter < center {
        let ratio1 = window[center - quarter] / window[center];
        let ratio2 = window[quarter] / window[center];

        // Should be symmetric and follow power law
        (ratio1 - ratio2).abs() < 0.1 && ratio1 > 0.5 && ratio1 < 0.95
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bohman_window() {
        let window = bohman(11, true).expect("Operation failed");
        assert_eq!(window.len(), 11);

        // Should be symmetric
        assert!((window[0] - window[10]).abs() < 1e-10);
        assert!((window[1] - window[9]).abs() < 1e-10);

        // Endpoints should be zero
        assert!(window[0].abs() < 1e-10);
        assert!(window[10].abs() < 1e-10);
    }

    #[test]
    fn test_poisson_window() {
        let window = poisson(10, 2.0, true).expect("Operation failed");
        assert_eq!(window.len(), 10);

        // Peak should be at center
        let center = window.len() / 2;
        let peak_idx = window
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("Operation failed"))
            .expect("Operation failed")
            .0;
        assert_eq!(peak_idx, center);
    }

    #[test]
    fn test_planck_taper() {
        let window = planck_taper(20, 0.1, true).expect("Operation failed");
        assert_eq!(window.len(), 20);

        // Should have middle region close to 1.0
        let middle_val = window[10];
        assert!((middle_val - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_general_cosine() {
        // Create Hamming window using general cosine
        // Note: scipy's general_cosine uses angle from -π to π, so the coefficient
        // should be positive [0.54, 0.46], not [0.54, -0.46]
        // hamming = 0.54 - 0.46*cos(2πn/(N-1)) but with fac from -π to π,
        // cos(-π) = -1, so we need w = 0.54 + 0.46*cos(fac) = 0.54 + 0.46*(-1) = 0.08 at endpoints
        let coeffs = vec![0.54, 0.46];
        let window = general_cosine(10, &coeffs, true).expect("Operation failed");
        assert_eq!(window.len(), 10);

        // Compare with direct Hamming calculation
        let expected =
            crate::window::families::cosine::hamming(10, true).expect("Operation failed");
        for (a, b) in window.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6); // More relaxed tolerance for floating-point comparison
        }
    }

    #[test]
    fn test_invalid_parameters() {
        assert!(poisson(10, -1.0, true).is_err());
        assert!(planck_taper(10, -0.1, true).is_err());
        assert!(planck_taper(10, 0.6, true).is_err());
        assert!(general_cosine(10, &[], true).is_err());
    }

    #[test]
    fn test_dpss_approximation() {
        let window = dpss_approximation(10, 3.0, true).expect("Operation failed");
        assert_eq!(window.len(), 10);

        // Should be roughly symmetric
        let n = window.len();
        for i in 0..(n / 2) {
            let diff = (window[i] - window[n - 1 - i]).abs();
            assert!(diff < 0.01, "Window not symmetric at index {}", i);
        }
    }
}
