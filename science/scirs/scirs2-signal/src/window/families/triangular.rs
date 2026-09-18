//! Triangular Window Family
//!
//! This module implements triangular window functions including:
//! - Bartlett window (triangular window with zero endpoints)
//! - Triangular window (triangular window with non-zero endpoints)
//! - Parzen window (triangular-based window with improved sidelobes)

use super::super::{_extend, _len_guards, _truncate};
use crate::error::{SignalError, SignalResult};

/// Bartlett window (triangular window with zero endpoints)
///
/// The Bartlett window is a triangular window with zero values at both endpoints.
/// It provides better sidelobe performance than rectangular but worse than most other windows.
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
/// use scirs2_signal::window::families::triangular::bartlett;
/// let window = bartlett(10, true).expect("Operation failed");
/// assert_eq!(window.len(), 10);
/// assert!((window[0] - 0.0).abs() < 1e-10); // Zero at endpoints
/// assert!((window[window.len()-1] - 0.0).abs() < 1e-10);
/// ```
pub fn bartlett(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let w_val = if i <= n / 2 {
            2.0 * i as f64 / (n - 1) as f64
        } else {
            2.0 - 2.0 * i as f64 / (n - 1) as f64
        };
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Triangular window (triangular window with non-zero endpoints)
///
/// The triangular window is similar to Bartlett but with non-zero endpoints.
/// The endpoints are at 1/N instead of 0.
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
/// use scirs2_signal::window::families::triangular::triang;
/// let window = triang(10, true).expect("Operation failed");
/// assert_eq!(window.len(), 10);
/// assert!(window[0] > 0.0); // Non-zero at endpoints
/// ```
pub fn triang(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let w_val = if i < n / 2 {
            2.0 * (i + 1) as f64 / (n + 1) as f64
        } else {
            2.0 - 2.0 * (i + 1) as f64 / (n + 1) as f64
        };
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Parzen window (triangular-based window with improved sidelobes)
///
/// The Parzen window is a 4th-order B-spline window that provides
/// better sidelobe suppression than simple triangular windows.
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
/// use scirs2_signal::window::families::triangular::parzen;
/// let window = parzen(10, true).expect("Operation failed");
/// assert_eq!(window.len(), 10);
/// ```
pub fn parzen(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    let center = (n - 1) as f64 / 2.0;

    for i in 0..n {
        let x = (i as f64 - center).abs() / center;

        let w_val = if x <= 0.5 {
            1.0 - 6.0 * x.powi(2) * (1.0 - x)
        } else if x <= 1.0 {
            2.0 * (1.0 - x).powi(3)
        } else {
            0.0
        };

        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Welch window (parabolic window)
///
/// The Welch window is a parabolic window that can be considered
/// a member of the triangular family with quadratic shape.
///
/// # Arguments
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
/// A `Vec<f64>` of window values
pub fn welch(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let x = (2.0 * i as f64) / (n - 1) as f64 - 1.0; // Normalize to [-1, 1]
        let w_val = 1.0 - x.powi(2);
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Generalized triangular window with custom slope
///
/// Creates a triangular window with user-specified slope characteristics.
///
/// # Arguments
/// * `m` - Number of points in the output window
/// * `slope_factor` - Controls the slope of the triangle (1.0 = standard, >1.0 = steeper)
/// * `zero_endpoints` - If true, endpoints are zero (Bartlett-style), otherwise non-zero
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
/// A `Vec<f64>` of window values
pub fn generalized_triangular(
    m: usize,
    slope_factor: f64,
    zero_endpoints: bool,
    sym: bool,
) -> SignalResult<Vec<f64>> {
    if slope_factor <= 0.0 {
        return Err(SignalError::ValueError(
            "Slope factor must be positive".to_string(),
        ));
    }

    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    let center = (n - 1) as f64 / 2.0;

    for i in 0..n {
        let normalized_pos = if zero_endpoints {
            i as f64 / (n - 1) as f64
        } else {
            (i + 1) as f64 / (n + 1) as f64
        };

        let distance_from_center = ((i as f64 - center) / center).abs();
        let triangular_value = (1.0 - distance_from_center.powf(slope_factor)).max(0.0);

        w.push(triangular_value);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Analyze triangular window properties
///
/// Computes important properties of triangular windows for analysis
pub fn analyze_triangular_window(window: &[f64]) -> TriangularWindowAnalysis {
    let n = window.len();

    // Find peak value and position
    let (peak_idx, peak_value) = window
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("Operation failed"))
        .unwrap_or((n / 2, &1.0));

    // Check if endpoints are zero (Bartlett-style)
    let zero_endpoints = window.first().copied().unwrap_or(0.0).abs() < 1e-10
        && window.last().copied().unwrap_or(0.0).abs() < 1e-10;

    // Coherent gain
    let coherent_gain = window.iter().sum::<f64>() / n as f64;

    // Processing gain
    let sum_squares = window.iter().map(|&w| w * w).sum::<f64>();
    let processing_gain = window.iter().sum::<f64>().powi(2) / (n as f64 * sum_squares);

    // Estimate window type
    let window_type = classify_triangular_window(window);

    // Scalloping loss (depends on window type)
    let scalloping_loss = match window_type {
        TriangularWindowType::Bartlett => 1.82,
        TriangularWindowType::Triangle => 1.82,
        TriangularWindowType::Parzen => 1.75,
        TriangularWindowType::Welch => 1.42,
        TriangularWindowType::Other => 2.0,
    };

    // Main lobe width
    let main_lobe_width = match window_type {
        TriangularWindowType::Bartlett | TriangularWindowType::Triangle => 4.0,
        TriangularWindowType::Parzen => 4.0,
        TriangularWindowType::Welch => 3.0,
        TriangularWindowType::Other => 4.0,
    };

    // Maximum sidelobe level
    let max_sidelobe_level = match window_type {
        TriangularWindowType::Bartlett | TriangularWindowType::Triangle => -26.5,
        TriangularWindowType::Parzen => -53.0,
        TriangularWindowType::Welch => -21.3,
        TriangularWindowType::Other => -25.0,
    };

    TriangularWindowAnalysis {
        window_type,
        peak_value: *peak_value,
        peak_position: peak_idx,
        zero_endpoints,
        coherent_gain,
        processing_gain,
        scalloping_loss,
        main_lobe_width,
        max_sidelobe_level,
    }
}

/// Triangular window analysis results
#[derive(Debug, Clone)]
pub struct TriangularWindowAnalysis {
    /// Detected window type
    pub window_type: TriangularWindowType,
    /// Peak value of the window
    pub peak_value: f64,
    /// Position of the peak (sample index)
    pub peak_position: usize,
    /// Whether endpoints are zero
    pub zero_endpoints: bool,
    /// Coherent gain
    pub coherent_gain: f64,
    /// Processing gain
    pub processing_gain: f64,
    /// Scalloping loss in dB
    pub scalloping_loss: f64,
    /// Main lobe width in bins
    pub main_lobe_width: f64,
    /// Maximum sidelobe level in dB
    pub max_sidelobe_level: f64,
}

/// Types of triangular windows
#[derive(Debug, Clone, PartialEq)]
pub enum TriangularWindowType {
    /// Bartlett window (zero endpoints)
    Bartlett,
    /// Triangular window (non-zero endpoints)
    Triangle,
    /// Parzen window (4th-order B-spline)
    Parzen,
    /// Welch window (parabolic)
    Welch,
    /// Other/unknown triangular variant
    Other,
}

/// Classify the type of triangular window
fn classify_triangular_window(window: &[f64]) -> TriangularWindowType {
    let n = window.len();
    if n < 3 {
        return TriangularWindowType::Other;
    }

    let first = window[0];
    let last = window[n - 1];
    let middle = window[n / 2];

    // Check if endpoints are zero (Bartlett)
    if first.abs() < 1e-10 && last.abs() < 1e-10 {
        // Could be Bartlett or Parzen
        // Parzen has a more curved shape
        if is_parzen_like(window) {
            TriangularWindowType::Parzen
        } else {
            TriangularWindowType::Bartlett
        }
    } else if is_parabolic_like(window) {
        TriangularWindowType::Welch
    } else if first > 0.0 && last > 0.0 {
        TriangularWindowType::Triangle
    } else {
        TriangularWindowType::Other
    }
}

/// Check if window has Parzen-like characteristics
fn is_parzen_like(window: &[f64]) -> bool {
    let n = window.len();
    if n < 5 {
        return false;
    }

    // Check for the characteristic B-spline curvature
    let quarter = n / 4;
    let half = n / 2;

    // Parzen window has a distinctive ratio pattern due to cubic falloff
    // Parzen ratio ~0.19 (low due to cubic polynomial), Bartlett ratio ~0.4 (linear)
    let ratio = window[quarter] / window[half];
    if ratio >= 0.3 {
        return false; // Bartlett and other linear windows have higher ratio
    }

    // Also check for non-linearity (curvature)
    // Parzen has polynomial curvature, Bartlett is linear
    // Check second differences to detect curvature
    let eighth = n / 8;
    if eighth > 0 && quarter < n && half < n {
        // For linear (Bartlett), second difference should be ~0
        // For polynomial (Parzen), second difference should be significant
        let w0 = window[eighth];
        let w1 = window[quarter];
        let w2 = window[quarter + eighth.max(1)];
        let second_diff = (w2 - w1) - (w1 - w0);
        // Parzen has noticeable curvature (positive second derivative in outer region)
        return second_diff.abs() > 0.01;
    }

    // Fallback to ratio-only check for small windows
    ratio < 0.3
}

/// Check if window has parabolic (Welch-like) characteristics
fn is_parabolic_like(window: &[f64]) -> bool {
    let n = window.len();
    if n < 3 {
        return false;
    }

    // Check if it follows parabolic shape: w[i] = 1 - x^2
    let center = (n - 1) as f64 / 2.0;
    let mut parabolic_error = 0.0;

    for (i, &w_val) in window.iter().enumerate() {
        let x = (i as f64 - center) / center;
        let expected = 1.0 - x * x;
        parabolic_error += (w_val - expected).abs();
    }

    parabolic_error / (n as f64) < 0.1 // Small error indicates parabolic shape
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bartlett_zero_endpoints() {
        let window = bartlett(5, true).expect("Operation failed");
        assert_eq!(window.len(), 5);
        assert!((window[0] - 0.0).abs() < 1e-10);
        assert!((window[4] - 0.0).abs() < 1e-10);
        assert!(window[2] > 0.8); // Peak in middle
    }

    #[test]
    fn test_triang_nonzero_endpoints() {
        let window = triang(5, true).expect("Operation failed");
        assert_eq!(window.len(), 5);
        assert!(window[0] > 0.0);
        assert!(window[4] > 0.0);
    }

    #[test]
    fn test_welch_parabolic() {
        let window = welch(5, true).expect("Operation failed");
        assert_eq!(window.len(), 5);

        // Check parabolic shape
        assert!((window[2] - 1.0).abs() < 1e-10); // Peak at center
        assert!(window[0] < window[1]); // Increasing towards center
        assert!(window[1] < window[2]); // Increasing towards center
    }

    #[test]
    fn test_window_classification() {
        let bartlett_win = bartlett(10, true).expect("Operation failed");
        let analysis = analyze_triangular_window(&bartlett_win);
        assert_eq!(analysis.window_type, TriangularWindowType::Bartlett);
        assert!(analysis.zero_endpoints);

        let triang_win = triang(10, true).expect("Operation failed");
        let analysis = analyze_triangular_window(&triang_win);
        assert_eq!(analysis.window_type, TriangularWindowType::Triangle);
        assert!(!analysis.zero_endpoints);
    }
}
