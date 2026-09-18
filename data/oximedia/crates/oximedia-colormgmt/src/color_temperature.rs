//! Color temperature utilities for converting between Kelvin and chromaticity.
//!
//! Provides correlated color temperature (CCT) estimation, Planckian locus
//! approximation, and tint (Duv) calculations for white-balance workflows.

/// Minimum supported color temperature in Kelvin.
const MIN_TEMP_K: f64 = 1000.0;

/// Maximum supported color temperature in Kelvin.
const MAX_TEMP_K: f64 = 25000.0;

/// CIE 1931 xy chromaticity coordinate pair.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Chromaticity {
    /// CIE x coordinate.
    pub x: f64,
    /// CIE y coordinate.
    pub y: f64,
}

impl Chromaticity {
    /// Creates a new chromaticity coordinate pair.
    ///
    /// # Arguments
    /// * `x` - CIE x coordinate (0..1 typical)
    /// * `y` - CIE y coordinate (0..1 typical)
    #[must_use]
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Returns the CIE u' v' coordinates (CIE 1976 UCS).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn to_upvp(&self) -> (f64, f64) {
        let denom = -2.0 * self.x + 12.0 * self.y + 3.0;
        if denom.abs() < 1e-12 {
            return (0.0, 0.0);
        }
        let up = 4.0 * self.x / denom;
        let vp = 9.0 * self.y / denom;
        (up, vp)
    }

    /// Euclidean distance to another chromaticity.
    #[must_use]
    pub fn distance(&self, other: &Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

/// Result of a CCT estimation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CctResult {
    /// Correlated color temperature in Kelvin.
    pub cct: f64,
    /// Distance from the Planckian locus (Duv). Positive = above (green tint),
    /// negative = below (magenta tint).
    pub duv: f64,
}

/// Approximates Planckian locus chromaticity from a color temperature.
///
/// Uses Kim et al. (2002) piecewise polynomial approximation for CIE 1931 xy.
///
/// # Arguments
/// * `kelvin` - Color temperature in Kelvin (clamped to 1000..25000)
///
/// # Returns
/// CIE 1931 xy chromaticity on the Planckian locus.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn kelvin_to_chromaticity(kelvin: f64) -> Chromaticity {
    let t = kelvin.clamp(MIN_TEMP_K, MAX_TEMP_K);
    let t2 = t * t;
    let t3 = t2 * t;

    // Piecewise polynomial for x
    let x = if t <= 4000.0 {
        -0.2661239e9 / t3 - 0.2343589e6 / t2 + 0.8776956e3 / t + 0.179910
    } else {
        -3.0258469e9 / t3 + 2.1070379e6 / t2 + 0.2226347e3 / t + 0.24039
    };

    let x2 = x * x;
    let x3 = x2 * x;

    // Piecewise polynomial for y
    let y = if t <= 2222.0 {
        -1.1063814 * x3 - 1.34811020 * x2 + 2.18555832 * x - 0.20219683
    } else if t <= 4000.0 {
        -0.9549476 * x3 - 1.37418593 * x2 + 2.09137015 * x - 0.16748867
    } else {
        3.0817580 * x3 - 5.87338670 * x2 + 3.75112997 * x - 0.37001483
    };

    Chromaticity::new(x, y)
}

/// Estimates the correlated color temperature (CCT) from CIE 1931 xy chromaticity.
///
/// Uses McCamy's approximation, accurate within about +/- 2 K for the range
/// 2000 K to 12500 K.
///
/// # Arguments
/// * `chroma` - CIE 1931 xy chromaticity
///
/// # Returns
/// Estimated CCT in Kelvin, or `None` if the chromaticity is too far from the
/// Planckian locus.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn chromaticity_to_cct(chroma: &Chromaticity) -> Option<f64> {
    // McCamy's epicentre
    let n = (chroma.x - 0.3320) / (chroma.y - 0.1858);
    let cct = -449.0 * n * n * n + 3525.0 * n * n - 6823.3 * n + 5520.33;
    if cct < MIN_TEMP_K || cct > MAX_TEMP_K {
        None
    } else {
        Some(cct)
    }
}

/// Full CCT + Duv estimation from CIE 1931 xy chromaticity.
///
/// Computes the correlated color temperature via McCamy and the distance
/// from the Planckian locus (Duv) in CIE 1976 UCS.
///
/// # Arguments
/// * `chroma` - CIE 1931 xy chromaticity
///
/// # Returns
/// `CctResult` with CCT and Duv, or `None` if out of range.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn estimate_cct_duv(chroma: &Chromaticity) -> Option<CctResult> {
    let cct = chromaticity_to_cct(chroma)?;
    let planckian = kelvin_to_chromaticity(cct);

    let (up_src, vp_src) = chroma.to_upvp();
    let (up_pl, vp_pl) = planckian.to_upvp();

    let du = up_src - up_pl;
    let dv = vp_src - vp_pl;
    let duv = (du * du + dv * dv).sqrt();

    // Sign: positive if source is above the Planckian locus (higher v')
    let sign = if vp_src >= vp_pl { 1.0 } else { -1.0 };

    Some(CctResult {
        cct,
        duv: sign * duv,
    })
}

/// Converts a Kelvin white-balance adjustment to an RGB multiplier triple.
///
/// A simple model for camera-style white-balance shifts. The reference
/// temperature is D65 (~6504 K).
///
/// # Arguments
/// * `kelvin` - Target color temperature in Kelvin
///
/// # Returns
/// `[r, g, b]` gain multipliers normalised so the green channel is 1.0.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn kelvin_to_rgb_gains(kelvin: f64) -> [f64; 3] {
    let t = kelvin.clamp(MIN_TEMP_K, MAX_TEMP_K);
    let target = kelvin_to_chromaticity(t);
    let d65 = kelvin_to_chromaticity(6504.0);

    // Ratio of target / D65 chromaticity as a simple gain model
    let rx = if target.x > 1e-12 {
        d65.x / target.x
    } else {
        1.0
    };
    let ry = if target.y > 1e-12 {
        d65.y / target.y
    } else {
        1.0
    };

    // Map x-shift to R/B balance, y-shift to overall scale
    let r_gain = rx;
    let b_gain = ry;
    let g_gain = 1.0;

    // Normalise so green = 1.0
    let max_g = g_gain;
    [r_gain / max_g, g_gain / max_g, b_gain / max_g]
}

/// Linearly interpolates between two color temperatures.
///
/// # Arguments
/// * `kelvin_a` - Start temperature in Kelvin
/// * `kelvin_b` - End temperature in Kelvin
/// * `t` - Interpolation factor (0.0 = a, 1.0 = b)
///
/// # Returns
/// Interpolated chromaticity on the Planckian locus.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn lerp_color_temperature(kelvin_a: f64, kelvin_b: f64, t: f64) -> Chromaticity {
    let t_clamped = t.clamp(0.0, 1.0);
    // Interpolate in reciprocal (MRD) space for perceptual uniformity
    let inv_a = 1.0 / kelvin_a.max(1.0);
    let inv_b = 1.0 / kelvin_b.max(1.0);
    let inv_lerp = inv_a + (inv_b - inv_a) * t_clamped;
    let kelvin_interp = 1.0 / inv_lerp;
    kelvin_to_chromaticity(kelvin_interp)
}

/// A color temperature ramp — an ordered sequence of temperature stops.
#[derive(Debug, Clone)]
pub struct TemperatureRamp {
    /// Ordered list of (Kelvin, weight) pairs.
    stops: Vec<(f64, f64)>,
}

impl TemperatureRamp {
    /// Creates a new empty temperature ramp.
    #[must_use]
    pub fn new() -> Self {
        Self { stops: Vec::new() }
    }

    /// Adds a temperature stop.
    ///
    /// # Arguments
    /// * `kelvin` - Temperature in Kelvin
    /// * `weight` - Blend weight (0..1)
    pub fn add_stop(&mut self, kelvin: f64, weight: f64) {
        self.stops.push((kelvin, weight.clamp(0.0, 1.0)));
    }

    /// Returns the number of stops.
    #[must_use]
    pub fn len(&self) -> usize {
        self.stops.len()
    }

    /// Returns `true` if the ramp has no stops.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.stops.is_empty()
    }

    /// Evaluates the ramp as a weighted average chromaticity.
    ///
    /// Returns `None` if the ramp is empty.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn evaluate(&self) -> Option<Chromaticity> {
        if self.stops.is_empty() {
            return None;
        }
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_w = 0.0;
        for &(kelvin, weight) in &self.stops {
            let c = kelvin_to_chromaticity(kelvin);
            sum_x += c.x * weight;
            sum_y += c.y * weight;
            sum_w += weight;
        }
        if sum_w < 1e-12 {
            return None;
        }
        Some(Chromaticity::new(sum_x / sum_w, sum_y / sum_w))
    }
}

impl Default for TemperatureRamp {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chromaticity_new() {
        let c = Chromaticity::new(0.31, 0.33);
        assert!((c.x - 0.31).abs() < 1e-9);
        assert!((c.y - 0.33).abs() < 1e-9);
    }

    #[test]
    fn test_chromaticity_distance() {
        let a = Chromaticity::new(0.0, 0.0);
        let b = Chromaticity::new(3.0, 4.0);
        assert!((a.distance(&b) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_chromaticity_to_upvp() {
        let d65 = Chromaticity::new(0.3127, 0.3290);
        let (up, vp) = d65.to_upvp();
        // D65 u' ~ 0.1978, v' ~ 0.4684
        assert!((up - 0.1978).abs() < 0.01);
        assert!((vp - 0.4684).abs() < 0.01);
    }

    #[test]
    fn test_kelvin_to_chromaticity_d65() {
        // D65 is approximately 6504 K
        let c = kelvin_to_chromaticity(6504.0);
        // Should be near (0.3127, 0.3290)
        assert!((c.x - 0.3127).abs() < 0.02);
        assert!((c.y - 0.3290).abs() < 0.02);
    }

    #[test]
    fn test_kelvin_to_chromaticity_clamping() {
        // Below minimum should clamp
        let c_low = kelvin_to_chromaticity(500.0);
        let c_min = kelvin_to_chromaticity(MIN_TEMP_K);
        assert!((c_low.x - c_min.x).abs() < 1e-9);

        // Above maximum should clamp
        let c_high = kelvin_to_chromaticity(50000.0);
        let c_max = kelvin_to_chromaticity(MAX_TEMP_K);
        assert!((c_high.x - c_max.x).abs() < 1e-9);
    }

    #[test]
    fn test_chromaticity_to_cct_roundtrip() {
        // Roundtrip: kelvin -> chromaticity -> cct
        for &k in &[2700.0, 4000.0, 5500.0, 6500.0, 8000.0] {
            let c = kelvin_to_chromaticity(k);
            let cct = chromaticity_to_cct(&c).expect("CCT should be in range");
            // McCamy is approximate; allow ~100 K error
            assert!(
                (cct - k).abs() < 150.0,
                "roundtrip failed for {k}: got {cct}"
            );
        }
    }

    #[test]
    fn test_chromaticity_to_cct_out_of_range() {
        // A very far-off chromaticity should return None
        let c = Chromaticity::new(0.7, 0.1);
        assert!(chromaticity_to_cct(&c).is_none());
    }

    #[test]
    fn test_estimate_cct_duv() {
        let c = kelvin_to_chromaticity(5000.0);
        let result = estimate_cct_duv(&c).expect("should be in range");
        // Duv should be near zero since we are on the Planckian locus
        assert!(result.duv.abs() < 0.01, "Duv too large: {}", result.duv);
        assert!(
            (result.cct - 5000.0).abs() < 150.0,
            "CCT off: {}",
            result.cct
        );
    }

    #[test]
    fn test_kelvin_to_rgb_gains_d65() {
        let gains = kelvin_to_rgb_gains(6504.0);
        // At D65, gains should be close to [1, 1, 1]
        for &g in &gains {
            assert!((g - 1.0).abs() < 0.05, "D65 gain expected ~1.0, got {g}");
        }
    }

    #[test]
    fn test_kelvin_to_rgb_gains_warm() {
        let gains = kelvin_to_rgb_gains(2700.0);
        // Warm light: blue channel should be boosted relative to reference
        assert!(
            gains[0] != gains[2],
            "R and B gains should differ for warm light"
        );
    }

    #[test]
    fn test_lerp_color_temperature() {
        let a = 3000.0;
        let b = 7000.0;
        let mid = lerp_color_temperature(a, b, 0.5);
        let ca = kelvin_to_chromaticity(a);
        let cb = kelvin_to_chromaticity(b);
        // Mid should be between the two endpoints in x
        let min_x = ca.x.min(cb.x);
        let max_x = ca.x.max(cb.x);
        assert!(mid.x >= min_x && mid.x <= max_x);
    }

    #[test]
    fn test_temperature_ramp_empty() {
        let ramp = TemperatureRamp::new();
        assert!(ramp.is_empty());
        assert_eq!(ramp.len(), 0);
        assert!(ramp.evaluate().is_none());
    }

    #[test]
    fn test_temperature_ramp_single_stop() {
        let mut ramp = TemperatureRamp::new();
        ramp.add_stop(5000.0, 1.0);
        assert_eq!(ramp.len(), 1);
        let c = ramp.evaluate().expect("should have result");
        let expected = kelvin_to_chromaticity(5000.0);
        assert!((c.x - expected.x).abs() < 1e-9);
        assert!((c.y - expected.y).abs() < 1e-9);
    }

    #[test]
    fn test_temperature_ramp_weighted_average() {
        let mut ramp = TemperatureRamp::new();
        ramp.add_stop(3000.0, 0.5);
        ramp.add_stop(7000.0, 0.5);
        let c = ramp.evaluate().expect("should have result");
        let c_a = kelvin_to_chromaticity(3000.0);
        let c_b = kelvin_to_chromaticity(7000.0);
        let expected_x = (c_a.x + c_b.x) / 2.0;
        let expected_y = (c_a.y + c_b.y) / 2.0;
        assert!((c.x - expected_x).abs() < 1e-9);
        assert!((c.y - expected_y).abs() < 1e-9);
    }
}
