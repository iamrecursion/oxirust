//! Domain range clamping and validation for LUT inputs.
//!
//! Ensures that input values stay within the defined domain of a LUT,
//! with configurable clamp, wrap, and mirror strategies.

#![allow(dead_code)]

/// A half-open range `[min, max]` for a single channel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DomainRange {
    /// Minimum allowed value (inclusive).
    pub min: f64,
    /// Maximum allowed value (inclusive).
    pub max: f64,
}

impl Default for DomainRange {
    fn default() -> Self {
        Self { min: 0.0, max: 1.0 }
    }
}

impl DomainRange {
    /// Create a new domain range.
    ///
    /// If `min >= max` the range is normalised so that min < max.
    #[must_use]
    pub fn new(a: f64, b: f64) -> Self {
        if a < b {
            Self { min: a, max: b }
        } else if b < a {
            Self { min: b, max: a }
        } else {
            // a == b: expand by epsilon to avoid zero-width range.
            Self {
                min: a - 0.5,
                max: a + 0.5,
            }
        }
    }

    /// Width of the range.
    #[must_use]
    pub fn span(&self) -> f64 {
        self.max - self.min
    }

    /// Returns `true` if `v` is within `[min, max]`.
    #[must_use]
    pub fn contains(&self, v: f64) -> bool {
        v >= self.min && v <= self.max
    }

    /// Map `v` from this range into [0, 1].
    #[must_use]
    pub fn normalise(&self, v: f64) -> f64 {
        let s = self.span();
        if s <= 0.0 {
            return 0.0;
        }
        (v - self.min) / s
    }

    /// Map `t` from [0, 1] into this range.
    #[must_use]
    pub fn denormalise(&self, t: f64) -> f64 {
        self.min + t * self.span()
    }

    /// Clamp `v` to `[min, max]`.
    #[must_use]
    pub fn clamp(&self, v: f64) -> f64 {
        v.clamp(self.min, self.max)
    }

    /// Mirror `v` if it exceeds the range (ping-pong).
    #[must_use]
    pub fn mirror(&self, v: f64) -> f64 {
        let s = self.span();
        if s <= 0.0 {
            return self.min;
        }
        let mut t = (v - self.min) / s;
        // Fold into [0, 2] then mirror.
        t = t.rem_euclid(2.0);
        if t > 1.0 {
            t = 2.0 - t;
        }
        self.min + t * s
    }

    /// Wrap `v` modularly within the range.
    #[must_use]
    pub fn wrap(&self, v: f64) -> f64 {
        let s = self.span();
        if s <= 0.0 {
            return self.min;
        }
        self.min + (v - self.min).rem_euclid(s)
    }
}

/// Tri-channel domain clamp for RGB LUT inputs.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DomainClamp {
    /// Red channel domain.
    pub r: DomainRange,
    /// Green channel domain.
    pub g: DomainRange,
    /// Blue channel domain.
    pub b: DomainRange,
}

impl DomainClamp {
    /// Create a uniform domain clamp `[min, max]` for all three channels.
    #[must_use]
    pub fn uniform(min: f64, max: f64) -> Self {
        let range = DomainRange::new(min, max);
        Self {
            r: range,
            g: range,
            b: range,
        }
    }

    /// Create a per-channel domain clamp from min/max triplets.
    #[must_use]
    pub fn per_channel(min: [f64; 3], max: [f64; 3]) -> Self {
        Self {
            r: DomainRange::new(min[0], max[0]),
            g: DomainRange::new(min[1], max[1]),
            b: DomainRange::new(min[2], max[2]),
        }
    }

    /// Clamp an `[R, G, B]` input to the domain.
    #[must_use]
    pub fn clamp_input(&self, rgb: [f64; 3]) -> [f64; 3] {
        [
            self.r.clamp(rgb[0]),
            self.g.clamp(rgb[1]),
            self.b.clamp(rgb[2]),
        ]
    }

    /// Normalise an `[R, G, B]` input to [0, 1] within the domain.
    #[must_use]
    pub fn normalise_input(&self, rgb: [f64; 3]) -> [f64; 3] {
        [
            self.r.normalise(rgb[0]),
            self.g.normalise(rgb[1]),
            self.b.normalise(rgb[2]),
        ]
    }

    /// Denormalise an `[R, G, B]` value from [0, 1] back to domain coordinates.
    #[must_use]
    pub fn denormalise_input(&self, rgb: [f64; 3]) -> [f64; 3] {
        [
            self.r.denormalise(rgb[0]),
            self.g.denormalise(rgb[1]),
            self.b.denormalise(rgb[2]),
        ]
    }
}

/// Validates that LUT domain parameters are well-formed.
#[derive(Debug)]
pub struct DomainValidator;

impl DomainValidator {
    /// Returns `true` if all three ranges have positive span.
    #[must_use]
    pub fn is_valid(clamp: &DomainClamp) -> bool {
        clamp.r.span() > 0.0 && clamp.g.span() > 0.0 && clamp.b.span() > 0.0
    }

    /// Returns `true` if the domain covers the standard [0, 1] range in every channel.
    #[must_use]
    pub fn covers_unit_cube(clamp: &DomainClamp) -> bool {
        clamp.r.min <= 0.0
            && clamp.r.max >= 1.0
            && clamp.g.min <= 0.0
            && clamp.g.max >= 1.0
            && clamp.b.min <= 0.0
            && clamp.b.max >= 1.0
    }

    /// Returns a list of channels whose domains are degenerate (span <= 0).
    #[must_use]
    pub fn degenerate_channels(clamp: &DomainClamp) -> Vec<&'static str> {
        let mut bad = Vec::new();
        if clamp.r.span() <= 0.0 {
            bad.push("R");
        }
        if clamp.g.span() <= 0.0 {
            bad.push("G");
        }
        if clamp.b.span() <= 0.0 {
            bad.push("B");
        }
        bad
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- DomainRange ---

    #[test]
    fn test_domain_range_default() {
        let r = DomainRange::default();
        assert!((r.min).abs() < 1e-12);
        assert!((r.max - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_domain_range_swapped() {
        let r = DomainRange::new(1.0, 0.0);
        assert!((r.min).abs() < 1e-12);
        assert!((r.max - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_domain_range_equal_expanded() {
        let r = DomainRange::new(5.0, 5.0);
        assert!(r.span() > 0.0);
    }

    #[test]
    fn test_span() {
        let r = DomainRange::new(0.0, 10.0);
        assert!((r.span() - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_contains() {
        let r = DomainRange::new(0.0, 1.0);
        assert!(r.contains(0.5));
        assert!(r.contains(0.0));
        assert!(r.contains(1.0));
        assert!(!r.contains(-0.1));
        assert!(!r.contains(1.1));
    }

    #[test]
    fn test_normalise_denormalise_roundtrip() {
        let r = DomainRange::new(10.0, 20.0);
        let n = r.normalise(15.0);
        assert!((n - 0.5).abs() < 1e-12);
        let d = r.denormalise(n);
        assert!((d - 15.0).abs() < 1e-12);
    }

    #[test]
    fn test_clamp() {
        let r = DomainRange::new(0.0, 1.0);
        assert!((r.clamp(-0.5)).abs() < 1e-12);
        assert!((r.clamp(1.5) - 1.0).abs() < 1e-12);
        assert!((r.clamp(0.5) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_mirror() {
        let r = DomainRange::new(0.0, 1.0);
        assert!((r.mirror(0.5) - 0.5).abs() < 1e-6);
        // 1.3 should mirror to 0.7.
        assert!((r.mirror(1.3) - 0.7).abs() < 1e-6);
        // -0.2 should mirror to 0.2.
        assert!((r.mirror(-0.2) - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_wrap() {
        let r = DomainRange::new(0.0, 1.0);
        assert!((r.wrap(0.5) - 0.5).abs() < 1e-12);
        assert!((r.wrap(1.3) - 0.3).abs() < 1e-9);
        assert!((r.wrap(-0.2) - 0.8).abs() < 1e-9);
    }

    // --- DomainClamp ---

    #[test]
    fn test_domain_clamp_uniform() {
        let dc = DomainClamp::uniform(0.0, 1.0);
        let clamped = dc.clamp_input([1.5, -0.5, 0.5]);
        assert!((clamped[0] - 1.0).abs() < 1e-12);
        assert!((clamped[1]).abs() < 1e-12);
        assert!((clamped[2] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_domain_clamp_normalise() {
        let dc = DomainClamp::uniform(0.0, 10.0);
        let n = dc.normalise_input([5.0, 0.0, 10.0]);
        assert!((n[0] - 0.5).abs() < 1e-12);
        assert!((n[1]).abs() < 1e-12);
        assert!((n[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_domain_clamp_denormalise() {
        let dc = DomainClamp::uniform(0.0, 10.0);
        let d = dc.denormalise_input([0.5, 0.0, 1.0]);
        assert!((d[0] - 5.0).abs() < 1e-12);
        assert!((d[1]).abs() < 1e-12);
        assert!((d[2] - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_domain_clamp_per_channel() {
        let dc = DomainClamp::per_channel([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]);
        let n = dc.normalise_input([0.5, 1.0, 1.5]);
        assert!((n[0] - 0.5).abs() < 1e-12);
        assert!((n[1] - 0.5).abs() < 1e-12);
        assert!((n[2] - 0.5).abs() < 1e-12);
    }

    // --- DomainValidator ---

    #[test]
    fn test_validator_default_valid() {
        assert!(DomainValidator::is_valid(&DomainClamp::default()));
    }

    #[test]
    fn test_validator_covers_unit_cube() {
        assert!(DomainValidator::covers_unit_cube(&DomainClamp::default()));
        let wide = DomainClamp::uniform(-1.0, 2.0);
        assert!(DomainValidator::covers_unit_cube(&wide));
    }

    #[test]
    fn test_validator_not_covering_unit_cube() {
        let narrow = DomainClamp::uniform(0.2, 0.8);
        assert!(!DomainValidator::covers_unit_cube(&narrow));
    }

    #[test]
    fn test_degenerate_channels_none() {
        let dc = DomainClamp::default();
        assert!(DomainValidator::degenerate_channels(&dc).is_empty());
    }
}
