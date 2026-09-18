//! Parametric color transforms (gamma curves, power functions, etc.).

/// Applies a power function (gamma) to each channel.
///
/// # Arguments
///
/// * `rgb` - Input RGB values [0, 1]
/// * `gamma` - Gamma value (e.g., 2.2, 2.4)
#[must_use]
pub fn apply_gamma(rgb: [f64; 3], gamma: f64) -> [f64; 3] {
    [rgb[0].powf(gamma), rgb[1].powf(gamma), rgb[2].powf(gamma)]
}

/// Applies exposure adjustment.
///
/// # Arguments
///
/// * `rgb` - Input linear RGB values
/// * `exposure` - Exposure adjustment in stops (positive = brighter, negative = darker)
#[must_use]
pub fn apply_exposure(rgb: [f64; 3], exposure: f64) -> [f64; 3] {
    let scale = 2.0_f64.powf(exposure);
    [rgb[0] * scale, rgb[1] * scale, rgb[2] * scale]
}

/// Applies brightness adjustment.
///
/// # Arguments
///
/// * `rgb` - Input RGB values [0, 1]
/// * `brightness` - Brightness adjustment [-1, 1]
#[must_use]
pub fn apply_brightness(rgb: [f64; 3], brightness: f64) -> [f64; 3] {
    [
        (rgb[0] + brightness).clamp(0.0, 1.0),
        (rgb[1] + brightness).clamp(0.0, 1.0),
        (rgb[2] + brightness).clamp(0.0, 1.0),
    ]
}

/// Applies contrast adjustment.
///
/// # Arguments
///
/// * `rgb` - Input RGB values [0, 1]
/// * `contrast` - Contrast factor (1.0 = no change, >1.0 = more contrast, <1.0 = less contrast)
#[must_use]
pub fn apply_contrast(rgb: [f64; 3], contrast: f64) -> [f64; 3] {
    let apply = |v: f64| ((v - 0.5) * contrast + 0.5).clamp(0.0, 1.0);
    [apply(rgb[0]), apply(rgb[1]), apply(rgb[2])]
}

/// Applies saturation adjustment in HSV space.
///
/// # Arguments
///
/// * `rgb` - Input RGB values [0, 1]
/// * `saturation` - Saturation factor (1.0 = no change, 0.0 = grayscale, >1.0 = more saturated)
#[must_use]
pub fn apply_saturation(rgb: [f64; 3], saturation: f64) -> [f64; 3] {
    // Calculate luminance (Rec.709)
    let luma = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];

    // Interpolate between grayscale and original
    [
        (luma + (rgb[0] - luma) * saturation).clamp(0.0, 1.0),
        (luma + (rgb[1] - luma) * saturation).clamp(0.0, 1.0),
        (luma + (rgb[2] - luma) * saturation).clamp(0.0, 1.0),
    ]
}

/// Applies temperature adjustment (simplified version).
///
/// # Arguments
///
/// * `rgb` - Input RGB values [0, 1]
/// * `temperature` - Temperature adjustment [-1, 1] (negative = cooler, positive = warmer)
#[must_use]
pub fn apply_temperature(rgb: [f64; 3], temperature: f64) -> [f64; 3] {
    if temperature > 0.0 {
        // Warmer: increase red, decrease blue
        [
            (rgb[0] * (1.0 + temperature * 0.3)).clamp(0.0, 1.0),
            rgb[1],
            (rgb[2] * (1.0 - temperature * 0.3)).clamp(0.0, 1.0),
        ]
    } else {
        // Cooler: decrease red, increase blue
        let temp_abs = temperature.abs();
        [
            (rgb[0] * (1.0 - temp_abs * 0.3)).clamp(0.0, 1.0),
            rgb[1],
            (rgb[2] * (1.0 + temp_abs * 0.3)).clamp(0.0, 1.0),
        ]
    }
}

/// Applies lift-gamma-gain color grading.
///
/// # Arguments
///
/// * `rgb` - Input linear RGB values
/// * `lift` - Lift adjustment (affects shadows)
/// * `gamma` - Gamma adjustment (affects midtones)
/// * `gain` - Gain adjustment (affects highlights)
#[must_use]
pub fn apply_lift_gamma_gain(rgb: [f64; 3], lift: f64, gamma: f64, gain: f64) -> [f64; 3] {
    let apply_channel = |v: f64| {
        // Apply lift
        let v = v + lift;
        // Apply gamma
        let v = if v > 0.0 { v.powf(1.0 / gamma) } else { 0.0 };
        // Apply gain
        (v * gain).clamp(0.0, 1.0)
    };

    [
        apply_channel(rgb[0]),
        apply_channel(rgb[1]),
        apply_channel(rgb[2]),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_gamma() {
        let rgb = [0.5, 0.5, 0.5];
        let result = apply_gamma(rgb, 2.0);

        assert!((result[0] - 0.25).abs() < 1e-10);
        assert!((result[1] - 0.25).abs() < 1e-10);
        assert!((result[2] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_apply_exposure() {
        let rgb = [0.5, 0.5, 0.5];
        let result = apply_exposure(rgb, 1.0); // +1 stop = 2x brighter

        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 1.0).abs() < 1e-10);
        assert!((result[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_apply_brightness() {
        let rgb = [0.5, 0.5, 0.5];
        let result = apply_brightness(rgb, 0.2);

        assert!((result[0] - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_apply_contrast() {
        let rgb = [0.5, 0.5, 0.5];
        let result = apply_contrast(rgb, 2.0);

        // Middle gray should stay at 0.5
        assert!((result[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_apply_saturation_grayscale() {
        let rgb = [1.0, 0.0, 0.0]; // Pure red
        let result = apply_saturation(rgb, 0.0); // Complete desaturation

        // All channels should be equal (grayscale)
        assert!((result[0] - result[1]).abs() < 1e-6);
        assert!((result[1] - result[2]).abs() < 1e-6);
    }

    #[test]
    fn test_apply_temperature() {
        let rgb = [0.5, 0.5, 0.5];

        let warmer = apply_temperature(rgb, 0.5);
        assert!(warmer[0] > rgb[0]); // More red
        assert!(warmer[2] < rgb[2]); // Less blue

        let cooler = apply_temperature(rgb, -0.5);
        assert!(cooler[0] < rgb[0]); // Less red
        assert!(cooler[2] > rgb[2]); // More blue
    }

    #[test]
    fn test_lift_gamma_gain_identity() {
        let rgb = [0.5, 0.3, 0.7];
        let result = apply_lift_gamma_gain(rgb, 0.0, 1.0, 1.0);

        assert!((result[0] - rgb[0]).abs() < 1e-10);
        assert!((result[1] - rgb[1]).abs() < 1e-10);
        assert!((result[2] - rgb[2]).abs() < 1e-10);
    }
}
