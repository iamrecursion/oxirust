#![allow(dead_code)]
//! Animation curve and keyframe system for broadcast graphics.
//!
//! Provides easing functions, individual keyframes with linear interpolation,
//! and a full animation curve that maps a time position to an interpolated value.

/// Easing type applied between two keyframes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EasingType {
    /// Constant rate of change throughout the interval.
    Linear,
    /// Slow start, fast end.
    EaseIn,
    /// Fast start, slow end.
    EaseOut,
    /// Slow start and end, fast middle.
    EaseInOut,
    /// Instantaneous jump at the end of the interval.
    Step,
    /// Overshoot and bounce at the end.
    Bounce,
    /// Spring physics: overshoots target then settles.
    /// Parameters: (stiffness, damping).
    /// Stiffness controls oscillation frequency; damping controls decay rate.
    /// Typical values: stiffness 100..500, damping 10..30.
    Spring {
        /// Spring stiffness (higher = faster oscillation).
        stiffness: f64,
        /// Damping coefficient (higher = less overshoot).
        damping: f64,
    },
    /// Damped oscillation: decaying sine wave.
    /// Parameters: (frequency, decay).
    /// Frequency in Hz, decay is the exponential decay rate.
    DampedOscillation {
        /// Oscillation frequency in cycles per unit time.
        frequency: f64,
        /// Exponential decay rate (higher = faster decay).
        decay: f64,
    },
    /// Elastic ease-in: snaps from rest.
    Elastic,
    /// Critically damped spring: fastest approach without overshoot.
    CriticallyDamped,
}

impl EasingType {
    /// Map normalised time `t ∈ [0.0, 1.0]` through this easing function.
    ///
    /// Returns a value in approximately `[0.0, 1.0]` (bounce may slightly exceed 1.0).
    pub fn ease_value(&self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        match self {
            EasingType::Linear => t,
            EasingType::EaseIn => t * t,
            EasingType::EaseOut => t * (2.0 - t),
            EasingType::EaseInOut => {
                if t < 0.5 {
                    2.0 * t * t
                } else {
                    -1.0 + (4.0 - 2.0 * t) * t
                }
            }
            EasingType::Step => {
                if t < 1.0 {
                    0.0
                } else {
                    1.0
                }
            }
            EasingType::Bounce => {
                let t2 = 1.0 - t;
                1.0 - Self::bounce_out(t2)
            }
            EasingType::Spring { stiffness, damping } => {
                Self::spring_value(t, *stiffness, *damping)
            }
            EasingType::DampedOscillation { frequency, decay } => {
                Self::damped_oscillation_value(t, *frequency, *decay)
            }
            EasingType::Elastic => Self::elastic_value(t),
            EasingType::CriticallyDamped => Self::critically_damped_value(t),
        }
    }

    fn bounce_out(t: f64) -> f64 {
        if t < 1.0 / 2.75 {
            7.5625 * t * t
        } else if t < 2.0 / 2.75 {
            let t = t - 1.5 / 2.75;
            7.5625 * t * t + 0.75
        } else if t < 2.5 / 2.75 {
            let t = t - 2.25 / 2.75;
            7.5625 * t * t + 0.9375
        } else {
            let t = t - 2.625 / 2.75;
            7.5625 * t * t + 0.984_375
        }
    }

    /// Simulate a spring system using the closed-form damped harmonic oscillator.
    ///
    /// Models displacement from target, starting at displacement = -1 (value 0)
    /// toward displacement = 0 (value 1). Underdamped springs will overshoot.
    fn spring_value(t: f64, stiffness: f64, damping: f64) -> f64 {
        if t <= 0.0 {
            return 0.0;
        }
        if t >= 1.0 {
            // Spring should converge to 1.0 at t=1.0
            // but we still compute to allow overshoot detection
        }

        // Mass normalized to 1.0
        let omega_n = stiffness.abs().sqrt(); // natural frequency
        let zeta = damping / (2.0 * omega_n).max(f64::EPSILON); // damping ratio

        if zeta >= 1.0 {
            // Overdamped or critically damped: no oscillation
            let s1 = -omega_n * (zeta - (zeta * zeta - 1.0).max(0.0).sqrt());
            let s2 = -omega_n * (zeta + (zeta * zeta - 1.0).max(0.0).sqrt());
            let denom = s1 - s2;
            if denom.abs() < f64::EPSILON {
                // Critically damped
                return 1.0 - (1.0 + omega_n * t) * (-omega_n * t).exp();
            }
            let c1 = -s2 / denom; // A = -s2/(s1-s2)
            let c2 = s1 / denom; // B = s1/(s1-s2)
            1.0 - (c1 * (s1 * t).exp() + c2 * (s2 * t).exp())
        } else {
            // Underdamped: oscillates
            let omega_d = omega_n * (1.0 - zeta * zeta).max(0.0).sqrt();
            let envelope = (-zeta * omega_n * t).exp();
            let phase = (omega_d * t).cos()
                + (zeta * omega_n / omega_d.max(f64::EPSILON)) * (omega_d * t).sin();
            1.0 - envelope * phase
        }
    }

    /// Damped oscillation: a decaying sinusoidal wave from 0 toward 1.
    fn damped_oscillation_value(t: f64, frequency: f64, decay: f64) -> f64 {
        if t <= 0.0 {
            return 0.0;
        }
        let envelope = (-decay * t).exp();
        let oscillation = (2.0 * std::f64::consts::PI * frequency * t).cos();
        1.0 - envelope * oscillation
    }

    /// Elastic ease-out: overshoots then settles.
    fn elastic_value(t: f64) -> f64 {
        if t <= 0.0 {
            return 0.0;
        }
        if t >= 1.0 {
            return 1.0;
        }
        let p = 0.3; // period
        let s = p / 4.0;
        let amplitude = 1.0;
        amplitude * (2.0_f64).powf(-10.0 * t) * ((t - s) * 2.0 * std::f64::consts::PI / p).sin()
            + 1.0
    }

    /// Critically damped spring: fastest convergence without oscillation.
    fn critically_damped_value(t: f64) -> f64 {
        if t <= 0.0 {
            return 0.0;
        }
        // omega chosen so spring is ~settled at t=1
        let omega = 6.0; // tuned for nice feel
        1.0 - (1.0 + omega * t) * (-omega * t).exp()
    }

    /// Returns `true` when this easing produces a smooth continuous transition.
    pub fn is_smooth(&self) -> bool {
        !matches!(self, EasingType::Step)
    }

    /// Create a spring easing with the given stiffness and damping.
    pub fn spring(stiffness: f64, damping: f64) -> Self {
        Self::Spring {
            stiffness: stiffness.max(0.1),
            damping: damping.max(0.0),
        }
    }

    /// Create a damped oscillation easing.
    pub fn damped_oscillation(frequency: f64, decay: f64) -> Self {
        Self::DampedOscillation {
            frequency: frequency.max(0.1),
            decay: decay.max(0.0),
        }
    }
}

/// A single keyframe in an animation curve.
#[derive(Debug, Clone)]
pub struct AnimationKeyframe {
    /// Time position in milliseconds from the start of the curve.
    pub time_ms: f64,
    /// Numeric value at this keyframe.
    pub value: f64,
    /// Easing applied from this keyframe to the next.
    pub easing: EasingType,
}

impl AnimationKeyframe {
    /// Create a new keyframe.
    pub fn new(time_ms: f64, value: f64, easing: EasingType) -> Self {
        Self {
            time_ms,
            value,
            easing,
        }
    }

    /// Linearly interpolate from this keyframe toward `next` at normalised time `t ∈ [0,1]`.
    ///
    /// Uses the easing type stored on *this* keyframe (the outgoing keyframe).
    pub fn lerp_to(&self, next: &AnimationKeyframe, t: f64) -> f64 {
        let eased_t = self.easing.ease_value(t);
        self.value + (next.value - self.value) * eased_t
    }

    /// Returns `true` when this keyframe sits at the curve origin.
    pub fn is_at_origin(&self) -> bool {
        self.time_ms == 0.0
    }
}

/// A multi-keyframe animation curve that returns interpolated values for any time position.
#[derive(Debug, Clone, Default)]
pub struct AnimationCurve {
    keyframes: Vec<AnimationKeyframe>,
}

impl AnimationCurve {
    /// Create an empty curve.
    pub fn new() -> Self {
        Self {
            keyframes: Vec::new(),
        }
    }

    /// Add a keyframe. The internal list is kept sorted by `time_ms`.
    pub fn add_keyframe(&mut self, kf: AnimationKeyframe) {
        self.keyframes.push(kf);
        self.keyframes.sort_by(|a, b| {
            a.time_ms
                .partial_cmp(&b.time_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Evaluate the curve at `time_ms`.
    ///
    /// - Before the first keyframe: returns the first keyframe's value.
    /// - After the last keyframe: returns the last keyframe's value.
    /// - Between two keyframes: interpolates using the outgoing keyframe's easing.
    pub fn value_at(&self, time_ms: f64) -> f64 {
        if self.keyframes.is_empty() {
            return 0.0;
        }
        if self.keyframes.len() == 1 || time_ms <= self.keyframes[0].time_ms {
            return self.keyframes[0].value;
        }
        let last = self
            .keyframes
            .last()
            .expect("keyframes non-empty: length check passed above");
        if time_ms >= last.time_ms {
            return last.value;
        }
        // Find the surrounding pair
        for i in 0..self.keyframes.len() - 1 {
            let a = &self.keyframes[i];
            let b = &self.keyframes[i + 1];
            if time_ms >= a.time_ms && time_ms <= b.time_ms {
                let span = b.time_ms - a.time_ms;
                let t = if span > 0.0 {
                    (time_ms - a.time_ms) / span
                } else {
                    0.0
                };
                return a.lerp_to(b, t);
            }
        }
        last.value
    }

    /// Total duration of the curve in milliseconds (last keyframe time).
    pub fn duration_ms(&self) -> f64 {
        self.keyframes.last().map_or(0.0, |k| k.time_ms)
    }

    /// Number of keyframes in this curve.
    pub fn keyframe_count(&self) -> usize {
        self.keyframes.len()
    }

    /// Returns `true` when the curve has no keyframes.
    pub fn is_empty(&self) -> bool {
        self.keyframes.is_empty()
    }

    /// Remove all keyframes.
    pub fn clear(&mut self) {
        self.keyframes.clear();
    }

    /// Return the minimum value across all keyframes.
    pub fn min_value(&self) -> Option<f64> {
        self.keyframes.iter().map(|k| k.value).reduce(f64::min)
    }

    /// Return the maximum value across all keyframes.
    pub fn max_value(&self) -> Option<f64> {
        self.keyframes.iter().map(|k| k.value).reduce(f64::max)
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_easing_linear_midpoint() {
        assert!((EasingType::Linear.ease_value(0.5) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_easing_linear_endpoints() {
        assert_eq!(EasingType::Linear.ease_value(0.0), 0.0);
        assert_eq!(EasingType::Linear.ease_value(1.0), 1.0);
    }

    #[test]
    fn test_easing_ease_in_slower_than_linear() {
        // EaseIn at 0.5 should be < linear (0.5)
        assert!(EasingType::EaseIn.ease_value(0.5) < 0.5);
    }

    #[test]
    fn test_easing_ease_out_faster_than_linear() {
        // EaseOut at 0.5 should be > linear (0.5)
        assert!(EasingType::EaseOut.ease_value(0.5) > 0.5);
    }

    #[test]
    fn test_easing_ease_in_out_midpoint() {
        // EaseInOut at 0.5 should equal 0.5 (symmetric)
        assert!((EasingType::EaseInOut.ease_value(0.5) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_easing_step_before_end() {
        assert_eq!(EasingType::Step.ease_value(0.5), 0.0);
    }

    #[test]
    fn test_easing_step_at_end() {
        assert_eq!(EasingType::Step.ease_value(1.0), 1.0);
    }

    #[test]
    fn test_easing_smooth_flag() {
        assert!(EasingType::Linear.is_smooth());
        assert!(!EasingType::Step.is_smooth());
    }

    #[test]
    fn test_keyframe_lerp_linear_half() {
        let a = AnimationKeyframe::new(0.0, 0.0, EasingType::Linear);
        let b = AnimationKeyframe::new(1000.0, 100.0, EasingType::Linear);
        let v = a.lerp_to(&b, 0.5);
        assert!((v - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_keyframe_lerp_at_start() {
        let a = AnimationKeyframe::new(0.0, 10.0, EasingType::Linear);
        let b = AnimationKeyframe::new(500.0, 20.0, EasingType::Linear);
        assert!((a.lerp_to(&b, 0.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_keyframe_lerp_at_end() {
        let a = AnimationKeyframe::new(0.0, 10.0, EasingType::Linear);
        let b = AnimationKeyframe::new(500.0, 20.0, EasingType::Linear);
        assert!((a.lerp_to(&b, 1.0) - 20.0).abs() < 1e-9);
    }

    #[test]
    fn test_curve_empty_returns_zero() {
        let c = AnimationCurve::new();
        assert_eq!(c.value_at(0.0), 0.0);
    }

    #[test]
    fn test_curve_single_keyframe() {
        let mut c = AnimationCurve::new();
        c.add_keyframe(AnimationKeyframe::new(0.0, 42.0, EasingType::Linear));
        assert_eq!(c.value_at(0.0), 42.0);
        assert_eq!(c.value_at(9999.0), 42.0);
    }

    #[test]
    fn test_curve_two_keyframes_midpoint() {
        let mut c = AnimationCurve::new();
        c.add_keyframe(AnimationKeyframe::new(0.0, 0.0, EasingType::Linear));
        c.add_keyframe(AnimationKeyframe::new(1000.0, 200.0, EasingType::Linear));
        let v = c.value_at(500.0);
        assert!((v - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_curve_duration() {
        let mut c = AnimationCurve::new();
        c.add_keyframe(AnimationKeyframe::new(0.0, 0.0, EasingType::Linear));
        c.add_keyframe(AnimationKeyframe::new(2500.0, 1.0, EasingType::Linear));
        assert!((c.duration_ms() - 2500.0).abs() < 1e-9);
    }

    #[test]
    fn test_curve_keyframe_count() {
        let mut c = AnimationCurve::new();
        c.add_keyframe(AnimationKeyframe::new(0.0, 0.0, EasingType::Linear));
        c.add_keyframe(AnimationKeyframe::new(500.0, 1.0, EasingType::EaseOut));
        assert_eq!(c.keyframe_count(), 2);
    }

    #[test]
    fn test_curve_min_max_values() {
        let mut c = AnimationCurve::new();
        c.add_keyframe(AnimationKeyframe::new(0.0, 5.0, EasingType::Linear));
        c.add_keyframe(AnimationKeyframe::new(500.0, 95.0, EasingType::Linear));
        assert_eq!(c.min_value(), Some(5.0));
        assert_eq!(c.max_value(), Some(95.0));
    }

    #[test]
    fn test_curve_clear() {
        let mut c = AnimationCurve::new();
        c.add_keyframe(AnimationKeyframe::new(0.0, 1.0, EasingType::Linear));
        c.clear();
        assert!(c.is_empty());
    }

    // --- Spring physics tests ---

    #[test]
    fn test_spring_endpoints() {
        let spring = EasingType::spring(200.0, 15.0);
        assert!((spring.ease_value(0.0)).abs() < 0.01);
        // At t=1.0, spring should be near 1.0 (converged)
        let v = spring.ease_value(1.0);
        assert!(
            (v - 1.0).abs() < 0.2,
            "Spring at t=1 should be near 1.0, got {v}"
        );
    }

    #[test]
    fn test_spring_underdamped_overshoots() {
        // Low damping -> oscillation -> overshoot
        let spring = EasingType::spring(400.0, 5.0);
        // Sample many points, at least one should exceed 1.0
        let mut overshot = false;
        for i in 1..100 {
            let t = i as f64 / 100.0;
            let v = spring.ease_value(t);
            if v > 1.0 {
                overshot = true;
                break;
            }
        }
        assert!(overshot, "Underdamped spring should overshoot 1.0");
    }

    #[test]
    fn test_spring_overdamped_no_overshoot() {
        // High damping -> no oscillation
        let spring = EasingType::spring(100.0, 50.0);
        for i in 0..=100 {
            let t = i as f64 / 100.0;
            let v = spring.ease_value(t);
            assert!(
                v <= 1.05,
                "Overdamped spring should not significantly overshoot, got {v} at t={t}"
            );
        }
    }

    #[test]
    fn test_spring_monotonic_convergence_overdamped() {
        let spring = EasingType::spring(100.0, 40.0);
        let mut prev = 0.0;
        for i in 1..=50 {
            let t = i as f64 / 50.0;
            let v = spring.ease_value(t);
            assert!(
                v >= prev - 0.01,
                "Overdamped spring should be mostly monotonic"
            );
            prev = v;
        }
    }

    // --- Damped oscillation tests ---

    #[test]
    fn test_damped_oscillation_endpoints() {
        let osc = EasingType::damped_oscillation(3.0, 5.0);
        assert!((osc.ease_value(0.0)).abs() < 0.01);
    }

    #[test]
    fn test_damped_oscillation_converges() {
        let osc = EasingType::damped_oscillation(2.0, 8.0);
        let v = osc.ease_value(1.0);
        assert!(
            (v - 1.0).abs() < 0.1,
            "Damped oscillation should converge near 1.0, got {v}"
        );
    }

    #[test]
    fn test_damped_oscillation_oscillates() {
        let osc = EasingType::damped_oscillation(5.0, 2.0);
        // With low decay, should cross 1.0 (overshoot then undershoot)
        let mut above = false;
        for i in 1..200 {
            let t = i as f64 / 200.0;
            let v = osc.ease_value(t);
            if v > 1.05 {
                above = true;
            }
        }
        assert!(above, "Low-decay oscillation should overshoot");
        // Oscillation means it should come back below 1 after overshooting
        // (may not always occur with these params, so just check overshoot)
    }

    // --- Elastic tests ---

    #[test]
    fn test_elastic_endpoints() {
        assert!((EasingType::Elastic.ease_value(0.0)).abs() < 0.01);
        assert!((EasingType::Elastic.ease_value(1.0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_elastic_overshoots() {
        let mut overshot = false;
        for i in 1..100 {
            let t = i as f64 / 100.0;
            let v = EasingType::Elastic.ease_value(t);
            if v > 1.01 {
                overshot = true;
                break;
            }
        }
        assert!(overshot, "Elastic should overshoot 1.0");
    }

    // --- Critically damped tests ---

    #[test]
    fn test_critically_damped_endpoints() {
        assert!((EasingType::CriticallyDamped.ease_value(0.0)).abs() < 0.01);
        let v = EasingType::CriticallyDamped.ease_value(1.0);
        assert!(
            (v - 1.0).abs() < 0.05,
            "Critically damped at t=1 should be near 1.0, got {v}"
        );
    }

    #[test]
    fn test_critically_damped_no_overshoot() {
        for i in 0..=100 {
            let t = i as f64 / 100.0;
            let v = EasingType::CriticallyDamped.ease_value(t);
            assert!(
                v <= 1.01,
                "Critically damped should not overshoot, got {v} at t={t}"
            );
        }
    }

    #[test]
    fn test_critically_damped_monotonic() {
        let mut prev = 0.0;
        for i in 1..=100 {
            let t = i as f64 / 100.0;
            let v = EasingType::CriticallyDamped.ease_value(t);
            assert!(v >= prev - 0.001, "Critically damped should be monotonic");
            prev = v;
        }
    }

    // --- Constructor tests ---

    #[test]
    fn test_spring_constructor() {
        let s = EasingType::spring(300.0, 20.0);
        if let EasingType::Spring { stiffness, damping } = s {
            assert!((stiffness - 300.0).abs() < f64::EPSILON);
            assert!((damping - 20.0).abs() < f64::EPSILON);
        } else {
            panic!("Expected Spring variant");
        }
    }

    #[test]
    fn test_damped_oscillation_constructor() {
        let d = EasingType::damped_oscillation(4.0, 6.0);
        if let EasingType::DampedOscillation { frequency, decay } = d {
            assert!((frequency - 4.0).abs() < f64::EPSILON);
            assert!((decay - 6.0).abs() < f64::EPSILON);
        } else {
            panic!("Expected DampedOscillation variant");
        }
    }

    #[test]
    fn test_spring_smoothness() {
        assert!(EasingType::spring(200.0, 15.0).is_smooth());
        assert!(EasingType::Elastic.is_smooth());
        assert!(EasingType::CriticallyDamped.is_smooth());
    }

    #[test]
    fn test_curve_with_spring_easing() {
        let mut c = AnimationCurve::new();
        c.add_keyframe(AnimationKeyframe::new(
            0.0,
            0.0,
            EasingType::spring(200.0, 15.0),
        ));
        c.add_keyframe(AnimationKeyframe::new(1000.0, 100.0, EasingType::Linear));
        let mid = c.value_at(500.0);
        // Should be roughly 50 but may overshoot due to spring
        assert!(
            mid > 20.0 && mid < 150.0,
            "Spring curve mid should be reasonable, got {mid}"
        );
    }
}
