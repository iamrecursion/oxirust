//! LOCO-I edge-detecting predictor (ISO 14495-1 §6.1).
//!
//! The predictor selects between clamping (for edges) and the gradient-based
//! estimate `a + b - c`, giving good prediction near both flat and edge regions.

/// Compute the LOCO-I predicted value.
///
/// `a` = left neighbour, `b` = above neighbour, `c` = above-left corner.
/// All values are in the reconstructed domain `[0, MaxVal]`.
///
/// The rule encodes a 2-context edge detector:
/// - If `c` is the local maximum (above or equal to both), clamp low.
/// - If `c` is the local minimum (below or equal to both), clamp high.
/// - Otherwise, use the linear estimate `a + b - c`.
#[inline]
pub fn predict(a: i32, b: i32, c: i32) -> i32 {
    if c >= a.max(b) {
        a.min(b)
    } else if c <= a.min(b) {
        a.max(b)
    } else {
        a + b - c
    }
}

/// Quantise a gradient difference into the range `[-4, 4]` using thresholds.
///
/// ISO 14495-1 §6.2 defines context quantisation to map an unbounded gradient
/// difference `d` into one of 9 levels, allowing the 3-gradient triple to be
/// mapped into 365 regular contexts.
#[inline]
pub fn quantize_gradient(d: i32, t1: i32, t2: i32, t3: i32) -> i8 {
    if d <= -t3 {
        -4
    } else if d <= -t2 {
        -3
    } else if d <= -t1 {
        -2
    } else if d < 0 {
        -1
    } else if d == 0 {
        0
    } else if d < t1 {
        1
    } else if d < t2 {
        2
    } else if d < t3 {
        3
    } else {
        4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predict_clamp_low_when_c_is_max() {
        // c >= max(a,b) → predict = min(a,b)
        assert_eq!(predict(10, 20, 25), 10);
    }

    #[test]
    fn predict_clamp_high_when_c_is_min() {
        // c <= min(a,b) → predict = max(a,b)
        assert_eq!(predict(10, 20, 5), 20);
    }

    #[test]
    fn predict_linear_in_middle() {
        // c is strictly between a and b → a + b - c
        assert_eq!(predict(10, 20, 15), 15);
    }

    #[test]
    fn predict_equal_neighbors() {
        assert_eq!(predict(8, 8, 8), 8);
    }

    #[test]
    fn quantize_gradient_boundaries() {
        // With t1=3, t2=7, t3=21
        assert_eq!(quantize_gradient(-25, 3, 7, 21), -4);
        assert_eq!(quantize_gradient(-10, 3, 7, 21), -3);
        assert_eq!(quantize_gradient(-5, 3, 7, 21), -2);
        assert_eq!(quantize_gradient(-1, 3, 7, 21), -1);
        assert_eq!(quantize_gradient(0, 3, 7, 21), 0);
        assert_eq!(quantize_gradient(1, 3, 7, 21), 1);
        assert_eq!(quantize_gradient(4, 3, 7, 21), 2);
        assert_eq!(quantize_gradient(10, 3, 7, 21), 3);
        assert_eq!(quantize_gradient(30, 3, 7, 21), 4);
    }
}
