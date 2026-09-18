//! Phase/goniometer scope: Lissajous display data, phase angle, and correlation coefficient.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

#[cfg(test)]
use std::f64::consts::PI;

/// A single point on the Lissajous (goniometer) display.
/// Coordinates are in the range [-1.0, 1.0].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LissajousPoint {
    /// X-axis coordinate (mid channel: L+R, scaled).
    pub x: f64,
    /// Y-axis coordinate (side channel: L-R, scaled).
    pub y: f64,
}

impl LissajousPoint {
    /// Create from left and right sample values.
    /// Uses M/S (mid/side) encoding: X = (L+R)/√2, Y = (L-R)/√2.
    #[must_use]
    pub fn from_lr(left: f64, right: f64) -> Self {
        let sqrt2 = std::f64::consts::SQRT_2;
        Self {
            x: (left + right) / sqrt2,
            y: (left - right) / sqrt2,
        }
    }

    /// Return the Euclidean distance from the origin.
    #[must_use]
    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    /// Return the angle in radians from the positive X axis.
    #[must_use]
    pub fn angle_radians(&self) -> f64 {
        self.y.atan2(self.x)
    }

    /// Return the angle in degrees from the positive X axis.
    #[must_use]
    pub fn angle_degrees(&self) -> f64 {
        self.angle_radians().to_degrees()
    }
}

/// Configuration for the phase scope.
#[derive(Debug, Clone)]
pub struct PhaseScopeConfig {
    /// Number of Lissajous points to retain in the display buffer.
    pub buffer_size: usize,
    /// Low-pass decay factor for the correlation coefficient (0.0 = no smoothing, 1.0 = hold).
    /// Typical value: 0.9.
    pub correlation_decay: f64,
    /// Low-pass decay factor for the phase angle display (0.0 = instant, 1.0 = hold).
    pub angle_decay: f64,
}

impl PhaseScopeConfig {
    /// Create a default config suitable for broadcast monitoring.
    #[must_use]
    pub fn default_broadcast() -> Self {
        Self {
            buffer_size: 4096,
            correlation_decay: 0.9,
            angle_decay: 0.85,
        }
    }
}

impl Default for PhaseScopeConfig {
    fn default() -> Self {
        Self::default_broadcast()
    }
}

/// Phase scope / goniometer meter.
///
/// Accepts stereo audio frames and exposes:
/// - A ring buffer of Lissajous display points.
/// - A smoothed phase correlation coefficient in [-1, +1].
/// - A smoothed phase angle in radians.
pub struct PhaseScope {
    config: PhaseScopeConfig,
    /// Ring buffer of recent Lissajous points.
    points: Vec<LissajousPoint>,
    /// Write head index into the ring buffer.
    write_pos: usize,
    /// Total samples processed.
    total_samples: usize,
    /// Smoothed correlation coefficient.
    correlation: f64,
    /// Smoothed phase angle (radians).
    phase_angle: f64,
    /// Accumulated numerator for correlation (L·R sum).
    acc_lr: f64,
    /// Accumulated L² sum.
    acc_l2: f64,
    /// Accumulated R² sum.
    acc_r2: f64,
    /// Accumulation window size.
    acc_count: usize,
}

impl PhaseScope {
    /// Create a new phase scope with the given configuration.
    #[must_use]
    pub fn new(config: PhaseScopeConfig) -> Self {
        let buffer_size = config.buffer_size.max(1);
        Self {
            config,
            points: vec![LissajousPoint { x: 0.0, y: 0.0 }; buffer_size],
            write_pos: 0,
            total_samples: 0,
            correlation: 0.0,
            phase_angle: 0.0,
            acc_lr: 0.0,
            acc_l2: 0.0,
            acc_r2: 0.0,
            acc_count: 0,
        }
    }

    /// Create a phase scope with default broadcast configuration.
    #[must_use]
    pub fn default_broadcast() -> Self {
        Self::new(PhaseScopeConfig::default_broadcast())
    }

    /// Process a block of interleaved stereo samples (L, R, L, R, ...).
    pub fn process_interleaved(&mut self, samples: &[f64]) {
        let frames = samples.len() / 2;
        for i in 0..frames {
            let left = samples[i * 2];
            let right = samples[i * 2 + 1];
            self.process_frame(left, right);
        }
    }

    /// Process a single stereo frame.
    pub fn process_frame(&mut self, left: f64, right: f64) {
        // Write Lissajous point
        let pt = LissajousPoint::from_lr(left, right);
        self.points[self.write_pos] = pt;
        self.write_pos = (self.write_pos + 1) % self.points.len();
        self.total_samples += 1;

        // Accumulate for correlation calculation
        self.acc_lr += left * right;
        self.acc_l2 += left * left;
        self.acc_r2 += right * right;
        self.acc_count += 1;

        // Update smoothed correlation every acc_window samples
        let acc_window = (self.config.buffer_size / 16).max(1);
        if self.acc_count >= acc_window {
            let denom = (self.acc_l2 * self.acc_r2).sqrt();
            let instant_corr = if denom > 1e-12 {
                (self.acc_lr / denom).clamp(-1.0, 1.0)
            } else {
                0.0
            };

            let decay = self.config.correlation_decay;
            self.correlation = decay * self.correlation + (1.0 - decay) * instant_corr;

            // Update phase angle from correlation
            // phase angle θ where cos(θ) ≈ correlation
            let instant_angle = instant_corr.clamp(-1.0, 1.0).acos();
            let angle_decay = self.config.angle_decay;
            self.phase_angle = angle_decay * self.phase_angle + (1.0 - angle_decay) * instant_angle;

            // Reset accumulators
            self.acc_lr = 0.0;
            self.acc_l2 = 0.0;
            self.acc_r2 = 0.0;
            self.acc_count = 0;
        }
    }

    /// Return the smoothed phase correlation coefficient in [-1, +1].
    /// +1 = perfect mono-compatible, -1 = phase-inverted.
    #[must_use]
    pub fn correlation(&self) -> f64 {
        self.correlation
    }

    /// Return the smoothed phase angle in radians [0, π].
    #[must_use]
    pub fn phase_angle_radians(&self) -> f64 {
        self.phase_angle
    }

    /// Return the smoothed phase angle in degrees [0, 180].
    #[must_use]
    pub fn phase_angle_degrees(&self) -> f64 {
        self.phase_angle.to_degrees()
    }

    /// Return `true` if correlation is negative (potential phase cancellation).
    #[must_use]
    pub fn has_phase_issues(&self) -> bool {
        self.correlation < -0.1
    }

    /// Return the current Lissajous display buffer as a slice (ordered from oldest to newest).
    #[must_use]
    pub fn lissajous_points(&self) -> Vec<LissajousPoint> {
        let len = self.points.len();
        let mut result = Vec::with_capacity(len);
        for i in 0..len {
            result.push(self.points[(self.write_pos + i) % len]);
        }
        result
    }

    /// Return total samples processed.
    #[must_use]
    pub fn total_samples(&self) -> usize {
        self.total_samples
    }

    /// Reset the scope to initial state.
    pub fn reset(&mut self) {
        let len = self.points.len();
        for pt in &mut self.points {
            *pt = LissajousPoint { x: 0.0, y: 0.0 };
        }
        self.write_pos = 0;
        self.total_samples = 0;
        self.correlation = 0.0;
        self.phase_angle = 0.0;
        self.acc_lr = 0.0;
        self.acc_l2 = 0.0;
        self.acc_r2 = 0.0;
        self.acc_count = 0;
        let _ = len; // suppress unused warning
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_scope() -> PhaseScope {
        let config = PhaseScopeConfig {
            buffer_size: 64,
            correlation_decay: 0.0, // instant update for testing
            angle_decay: 0.0,
        };
        PhaseScope::new(config)
    }

    #[test]
    fn test_lissajous_point_from_lr_mono() {
        // Equal L and R → Y = 0
        let pt = LissajousPoint::from_lr(0.5, 0.5);
        assert!(pt.y.abs() < 1e-10);
    }

    #[test]
    fn test_lissajous_point_from_lr_side() {
        // Opposite L and R → X = 0
        let pt = LissajousPoint::from_lr(0.5, -0.5);
        assert!(pt.x.abs() < 1e-10);
        assert!(pt.y > 0.0);
    }

    #[test]
    fn test_lissajous_point_magnitude() {
        let pt = LissajousPoint { x: 3.0, y: 4.0 };
        assert!((pt.magnitude() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_lissajous_point_angle() {
        let pt = LissajousPoint { x: 0.0, y: 1.0 };
        let angle_deg = pt.angle_degrees();
        assert!((angle_deg - 90.0).abs() < 1e-6);
    }

    #[test]
    fn test_phase_scope_default_broadcast() {
        let scope = PhaseScope::default_broadcast();
        assert_eq!(scope.config.buffer_size, 4096);
    }

    #[test]
    fn test_phase_scope_initial_correlation_zero() {
        let scope = make_scope();
        assert_eq!(scope.correlation(), 0.0);
    }

    #[test]
    fn test_phase_scope_perfect_correlation() {
        let mut scope = make_scope();
        // Perfectly correlated (mono) signal
        let mut samples = Vec::new();
        for _ in 0..64 {
            samples.push(0.5_f64);
            samples.push(0.5_f64);
        }
        scope.process_interleaved(&samples);
        // Correlation should be close to +1
        assert!(
            scope.correlation() > 0.9,
            "correlation={}",
            scope.correlation()
        );
    }

    #[test]
    fn test_phase_scope_negative_correlation() {
        let mut scope = make_scope();
        // Phase-inverted signal
        let mut samples = Vec::new();
        for _ in 0..64 {
            samples.push(0.5_f64);
            samples.push(-0.5_f64);
        }
        scope.process_interleaved(&samples);
        assert!(
            scope.correlation() < -0.9,
            "correlation={}",
            scope.correlation()
        );
    }

    #[test]
    fn test_phase_scope_has_phase_issues() {
        let mut scope = make_scope();
        let mut samples = Vec::new();
        for _ in 0..64 {
            samples.push(0.5_f64);
            samples.push(-0.5_f64);
        }
        scope.process_interleaved(&samples);
        assert!(scope.has_phase_issues());
    }

    #[test]
    fn test_phase_scope_no_phase_issues_mono() {
        let mut scope = make_scope();
        let mut samples = Vec::new();
        for _ in 0..64 {
            samples.push(0.5_f64);
            samples.push(0.5_f64);
        }
        scope.process_interleaved(&samples);
        assert!(!scope.has_phase_issues());
    }

    #[test]
    fn test_phase_scope_total_samples() {
        let mut scope = make_scope();
        let samples: Vec<f64> = vec![0.1; 20]; // 10 frames
        scope.process_interleaved(&samples);
        assert_eq!(scope.total_samples(), 10);
    }

    #[test]
    fn test_phase_scope_lissajous_points_length() {
        let scope = make_scope();
        let pts = scope.lissajous_points();
        assert_eq!(pts.len(), scope.config.buffer_size);
    }

    #[test]
    fn test_phase_scope_reset() {
        let mut scope = make_scope();
        let samples: Vec<f64> = vec![0.5; 64];
        scope.process_interleaved(&samples);
        scope.reset();
        assert_eq!(scope.total_samples(), 0);
        assert_eq!(scope.correlation(), 0.0);
    }

    #[test]
    fn test_phase_scope_phase_angle_range() {
        let mut scope = make_scope();
        let mut samples = Vec::new();
        for _ in 0..64 {
            samples.push(0.7_f64);
            samples.push(0.3_f64);
        }
        scope.process_interleaved(&samples);
        let angle = scope.phase_angle_radians();
        assert!(angle >= 0.0 && angle <= PI, "angle={}", angle);
    }

    #[test]
    fn test_phase_scope_process_frame() {
        let mut scope = make_scope();
        scope.process_frame(0.5, 0.5);
        assert_eq!(scope.total_samples(), 1);
    }

    #[test]
    fn test_phase_scope_angle_degrees_range() {
        let mut scope = make_scope();
        let mut samples = Vec::new();
        for _ in 0..64 {
            samples.push(0.5_f64);
            samples.push(0.5_f64);
        }
        scope.process_interleaved(&samples);
        let deg = scope.phase_angle_degrees();
        assert!(deg >= 0.0 && deg <= 180.0, "deg={}", deg);
    }
}
