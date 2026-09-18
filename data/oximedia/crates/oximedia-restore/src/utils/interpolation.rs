//! Interpolation methods for audio restoration.

use crate::error::{RestoreError, RestoreResult};

/// Interpolation method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMethod {
    /// Linear interpolation.
    Linear,
    /// Cubic spline interpolation.
    Cubic,
    /// Hermite spline interpolation.
    Hermite,
    /// Lagrange polynomial interpolation.
    Lagrange,
}

/// Perform linear interpolation.
///
/// # Arguments
///
/// * `samples` - Input samples
/// * `start` - Start index of region to interpolate
/// * `end` - End index of region to interpolate
///
/// # Returns
///
/// Interpolated samples for the region.
pub fn linear(samples: &[f32], start: usize, end: usize) -> RestoreResult<Vec<f32>> {
    if start >= end || end > samples.len() {
        return Err(RestoreError::InvalidParameter(
            "Invalid interpolation range".to_string(),
        ));
    }

    let len = end - start;
    if len == 0 {
        return Ok(Vec::new());
    }

    let y0 = if start > 0 {
        samples[start - 1]
    } else {
        samples[start]
    };
    let y1 = if end < samples.len() {
        samples[end]
    } else {
        samples[end - 1]
    };

    let mut result = Vec::with_capacity(len);
    #[allow(clippy::cast_precision_loss)]
    let step = (y1 - y0) / len as f32;

    for i in 0..len {
        #[allow(clippy::cast_precision_loss)]
        let value = y0 + step * i as f32;
        result.push(value);
    }

    Ok(result)
}

/// Perform cubic spline interpolation.
///
/// # Arguments
///
/// * `samples` - Input samples
/// * `start` - Start index of region to interpolate
/// * `end` - End index of region to interpolate
///
/// # Returns
///
/// Interpolated samples for the region.
pub fn cubic(samples: &[f32], start: usize, end: usize) -> RestoreResult<Vec<f32>> {
    if start >= end || end > samples.len() {
        return Err(RestoreError::InvalidParameter(
            "Invalid interpolation range".to_string(),
        ));
    }

    let len = end - start;
    if len == 0 {
        return Ok(Vec::new());
    }

    // Get control points
    let p0 = if start > 1 {
        samples[start - 2]
    } else if start > 0 {
        samples[start - 1]
    } else {
        samples[start]
    };

    let p1 = if start > 0 {
        samples[start - 1]
    } else {
        samples[start]
    };

    let p2 = if end < samples.len() {
        samples[end]
    } else {
        samples[end - 1]
    };

    let p3 = if end + 1 < samples.len() {
        samples[end + 1]
    } else if end < samples.len() {
        samples[end]
    } else {
        samples[end - 1]
    };

    let mut result = Vec::with_capacity(len);

    for i in 0..len {
        #[allow(clippy::cast_precision_loss)]
        let t = i as f32 / len as f32;
        let t2 = t * t;
        let t3 = t2 * t;

        // Catmull-Rom spline coefficients
        let a0 = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3;
        let a1 = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3;
        let a2 = -0.5 * p0 + 0.5 * p2;
        let a3 = p1;

        let value = a0 * t3 + a1 * t2 + a2 * t + a3;
        result.push(value);
    }

    Ok(result)
}

/// Perform Hermite spline interpolation.
///
/// # Arguments
///
/// * `samples` - Input samples
/// * `start` - Start index of region to interpolate
/// * `end` - End index of region to interpolate
/// * `tension` - Tension parameter (0.0 = Catmull-Rom, 1.0 = linear)
/// * `bias` - Bias parameter (-1.0 to 1.0)
///
/// # Returns
///
/// Interpolated samples for the region.
#[allow(clippy::too_many_arguments)]
pub fn hermite(
    samples: &[f32],
    start: usize,
    end: usize,
    tension: f32,
    bias: f32,
) -> RestoreResult<Vec<f32>> {
    if start >= end || end > samples.len() {
        return Err(RestoreError::InvalidParameter(
            "Invalid interpolation range".to_string(),
        ));
    }

    let len = end - start;
    if len == 0 {
        return Ok(Vec::new());
    }

    // Get control points
    let y0 = if start > 0 {
        samples[start - 1]
    } else {
        samples[start]
    };

    let y1 = if start > 0 {
        samples[start - 1]
    } else {
        samples[start]
    };

    let y2 = if end < samples.len() {
        samples[end]
    } else {
        samples[end - 1]
    };

    let y3 = if end + 1 < samples.len() {
        samples[end + 1]
    } else if end < samples.len() {
        samples[end]
    } else {
        samples[end - 1]
    };

    let mut result = Vec::with_capacity(len);

    for i in 0..len {
        #[allow(clippy::cast_precision_loss)]
        let t = i as f32 / len as f32;
        let t2 = t * t;
        let t3 = t2 * t;

        // Compute tangents
        let m0 = (1.0 - tension) * (1.0 + bias) * (y1 - y0) / 2.0
            + (1.0 - tension) * (1.0 - bias) * (y2 - y1) / 2.0;
        let m1 = (1.0 - tension) * (1.0 + bias) * (y2 - y1) / 2.0
            + (1.0 - tension) * (1.0 - bias) * (y3 - y2) / 2.0;

        // Hermite basis functions
        let a0 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let a1 = t3 - 2.0 * t2 + t;
        let a2 = t3 - t2;
        let a3 = -2.0 * t3 + 3.0 * t2;

        let value = a0 * y1 + a1 * m0 + a2 * m1 + a3 * y2;
        result.push(value);
    }

    Ok(result)
}

/// Perform Lagrange polynomial interpolation.
///
/// # Arguments
///
/// * `samples` - Input samples
/// * `start` - Start index of region to interpolate
/// * `end` - End index of region to interpolate
/// * `order` - Polynomial order (3 recommended)
///
/// # Returns
///
/// Interpolated samples for the region.
pub fn lagrange(
    samples: &[f32],
    start: usize,
    end: usize,
    order: usize,
) -> RestoreResult<Vec<f32>> {
    if start >= end || end > samples.len() {
        return Err(RestoreError::InvalidParameter(
            "Invalid interpolation range".to_string(),
        ));
    }

    if order == 0 {
        return Err(RestoreError::InvalidParameter(
            "Order must be positive".to_string(),
        ));
    }

    let len = end - start;
    if len == 0 {
        return Ok(Vec::new());
    }

    // Collect control points before and after the gap
    let half_order = order / 2;
    let mut x_points = Vec::new();
    let mut y_points = Vec::new();

    // Points before the gap
    for i in 0..=half_order {
        if start > i {
            x_points.push(start - i - 1);
            y_points.push(samples[start - i - 1]);
        }
    }

    // Points after the gap
    for i in 0..=half_order {
        if end + i < samples.len() {
            x_points.push(end + i);
            y_points.push(samples[end + i]);
        }
    }

    if x_points.len() < 2 {
        // Fall back to linear interpolation
        return linear(samples, start, end);
    }

    let mut result = Vec::with_capacity(len);

    for i in 0..len {
        let x = start + i;
        let mut y = 0.0;

        // Lagrange interpolation formula
        for j in 0..x_points.len() {
            let mut term = y_points[j];
            for k in 0..x_points.len() {
                if j != k {
                    #[allow(clippy::cast_precision_loss)]
                    let numerator = (x as isize - x_points[k] as isize) as f32;
                    #[allow(clippy::cast_precision_loss)]
                    let denominator = (x_points[j] as isize - x_points[k] as isize) as f32;
                    if denominator.abs() > f32::EPSILON {
                        term *= numerator / denominator;
                    }
                }
            }
            y += term;
        }

        result.push(y);
    }

    Ok(result)
}

/// Interpolate a region using the specified method.
pub fn interpolate(
    samples: &[f32],
    start: usize,
    end: usize,
    method: InterpolationMethod,
) -> RestoreResult<Vec<f32>> {
    match method {
        InterpolationMethod::Linear => linear(samples, start, end),
        InterpolationMethod::Cubic => cubic(samples, start, end),
        InterpolationMethod::Hermite => hermite(samples, start, end, 0.0, 0.0),
        InterpolationMethod::Lagrange => lagrange(samples, start, end, 3),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_interpolation() {
        let samples = vec![0.0, 0.5, 0.0, 0.0, 0.0, 1.0];
        let result = linear(&samples, 2, 5).expect("should succeed in test");
        assert_eq!(result.len(), 3);
        // Interpolates from samples[1]=0.5 to samples[5]=1.0
        assert!((result[0] - 0.5).abs() < 0.1);
        assert!((result[2] - 0.833).abs() < 0.1);
    }

    #[test]
    fn test_cubic_interpolation() {
        let samples = vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let result = cubic(&samples, 2, 5).expect("should succeed in test");
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_hermite_interpolation() {
        let samples = vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let result = hermite(&samples, 2, 5, 0.0, 0.0).expect("should succeed in test");
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_lagrange_interpolation() {
        let samples = vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let result = lagrange(&samples, 2, 5, 3).expect("should succeed in test");
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_invalid_range() {
        let samples = vec![0.0, 1.0, 2.0];
        assert!(linear(&samples, 2, 1).is_err());
        assert!(linear(&samples, 0, 10).is_err());
    }
}
