#![allow(dead_code)]
//! Tripod mode: detect a static camera and apply strong lock-on stabilization.
//!
//! When a camera is mounted on a tripod (or otherwise nearly static), the
//! motion between frames is dominated by sensor vibration, wind, and electronic
//! noise rather than intentional camera movement.  Standard stabilization
//! smoothing windows are optimized for hand-held motion and can be too
//! permissive for truly static scenes, leaving residual jitter.
//!
//! This module provides:
//!
//! - [`TripodDetector`] — classifies a sequence of motion vectors as either
//!   `Tripod`, `HandHeld`, or `Panning` based on velocity and variance statistics.
//! - [`TripodStabilizer`] — applies near-lock-on correction when tripod mode
//!   is confirmed, with configurable fallback to standard smoothing otherwise.
//! - [`TripodReport`] — per-frame classification and correction summary.
//!
//! # Detection Algorithm
//!
//! The detector computes a rolling window of:
//! - Mean motion magnitude (px/frame)
//! - Standard deviation of motion magnitudes
//! - Fraction of frames with magnitude below the `jitter_threshold`
//!
//! A static camera is declared when all three metrics fall below their
//! respective thresholds (`max_mean_motion`, `max_stddev`, `min_static_fraction`).
//! Panning is detected when the mean motion is high but the *direction* is
//! consistent (low angular variance).
//!
//! # Stabilization Strategy
//!
//! In tripod mode the stabilizer uses an extremely small EMA alpha (near 0),
//! effectively locking the camera to the global mean position computed over
//! the first `lock_window` frames.  Outside tripod mode it delegates to the
//! standard EMA smoother.
//!
//! # Example
//!
//! ```
//! use oximedia_stabilize::tripod_mode::{
//!     TripodConfig, TripodStabilizer, MotionSample,
//! };
//!
//! let config = TripodConfig::default();
//! let mut stabilizer = TripodStabilizer::new(config);
//!
//! // Feed per-frame motion samples (synthetic tripod: tiny random noise)
//! let samples: Vec<MotionSample> = (0..60).map(|_| {
//!     MotionSample { dx: 0.2, dy: -0.1, angle: 0.0 }
//! }).collect();
//!
//! let report = stabilizer.process(&samples);
//! assert!(report.is_tripod_detected);
//! ```

use std::f64::consts::PI;

use crate::error::{StabilizeError, StabilizeResult};

// ─────────────────────────────────────────────────────────────────
//  Data types
// ─────────────────────────────────────────────────────────────────

/// Raw per-frame camera motion sample used by the tripod detector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionSample {
    /// Horizontal translation (pixels/frame).
    pub dx: f64,
    /// Vertical translation (pixels/frame).
    pub dy: f64,
    /// Rotation (radians/frame, positive = clockwise).
    pub angle: f64,
}

impl MotionSample {
    /// Create a translation-only sample.
    #[must_use]
    pub const fn translation(dx: f64, dy: f64) -> Self {
        Self { dx, dy, angle: 0.0 }
    }

    /// Translational magnitude (pixels/frame).
    #[must_use]
    pub fn magnitude(&self) -> f64 {
        (self.dx * self.dx + self.dy * self.dy).sqrt()
    }

    /// Motion direction in radians (atan2 of dy, dx).
    #[must_use]
    pub fn direction(&self) -> f64 {
        self.dy.atan2(self.dx)
    }
}

impl Default for MotionSample {
    fn default() -> Self {
        Self {
            dx: 0.0,
            dy: 0.0,
            angle: 0.0,
        }
    }
}

/// Camera motion classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotionClass {
    /// Camera is essentially static (tripod / fixed mount).
    Tripod,
    /// Camera is hand-held with random jitter.
    HandHeld,
    /// Camera is performing an intentional pan or tilt.
    Panning,
    /// Not enough data to classify yet.
    Unknown,
}

impl MotionClass {
    /// Human-readable label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Tripod => "Tripod",
            Self::HandHeld => "HandHeld",
            Self::Panning => "Panning",
            Self::Unknown => "Unknown",
        }
    }

    /// Whether tripod-mode lock-on stabilization should be applied.
    #[must_use]
    pub const fn should_lock(self) -> bool {
        matches!(self, Self::Tripod)
    }
}

// ─────────────────────────────────────────────────────────────────
//  TripodConfig
// ─────────────────────────────────────────────────────────────────

/// Configuration for [`TripodDetector`] and [`TripodStabilizer`].
#[derive(Debug, Clone)]
pub struct TripodConfig {
    /// Maximum mean motion magnitude (px/frame) to be considered static.
    pub max_mean_motion: f64,
    /// Maximum standard deviation of motion magnitudes to be considered static.
    pub max_stddev: f64,
    /// Fraction of frames (0–1) that must be below `jitter_threshold` for
    /// a static classification.
    pub min_static_fraction: f64,
    /// Per-frame threshold for classifying a frame as "static" (px/frame).
    pub jitter_threshold: f64,
    /// Minimum mean motion (px/frame) to consider intentional panning.
    pub pan_mean_threshold: f64,
    /// Maximum direction variance (radians²) for panning detection.
    pub pan_direction_variance: f64,
    /// Window size (frames) for rolling classification.
    pub window_size: usize,
    /// EMA alpha for standard (non-tripod) smoothing.
    pub standard_alpha: f64,
    /// EMA alpha for tripod lock-on (very small → near lock).
    pub tripod_alpha: f64,
    /// Number of frames to average for the initial lock-on reference position.
    pub lock_window: usize,
    /// Hysteresis: number of consecutive `HandHeld` frames required to exit
    /// tripod mode once it has been engaged.
    pub exit_hysteresis: usize,
}

impl Default for TripodConfig {
    fn default() -> Self {
        Self {
            max_mean_motion: 1.5,
            max_stddev: 1.0,
            min_static_fraction: 0.85,
            jitter_threshold: 2.0,
            pan_mean_threshold: 8.0,
            pan_direction_variance: 0.3,
            window_size: 30,
            standard_alpha: 0.15,
            tripod_alpha: 0.02,
            lock_window: 10,
            exit_hysteresis: 15,
        }
    }
}

impl TripodConfig {
    /// Create a configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the max mean motion threshold.
    #[must_use]
    pub fn with_max_mean_motion(mut self, px: f64) -> Self {
        self.max_mean_motion = px.max(0.0);
        self
    }

    /// Set the jitter threshold.
    #[must_use]
    pub fn with_jitter_threshold(mut self, px: f64) -> Self {
        self.jitter_threshold = px.max(0.0);
        self
    }

    /// Set the analysis window size.
    #[must_use]
    pub fn with_window_size(mut self, n: usize) -> Self {
        self.window_size = n.max(5);
        self
    }

    /// Set the tripod lock-on EMA alpha.
    #[must_use]
    pub fn with_tripod_alpha(mut self, alpha: f64) -> Self {
        self.tripod_alpha = alpha.clamp(1e-4, 0.5);
        self
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`StabilizeError::InvalidParameter`] for out-of-range values.
    pub fn validate(&self) -> StabilizeResult<()> {
        if self.standard_alpha <= 0.0 || self.standard_alpha > 1.0 {
            return Err(StabilizeError::invalid_parameter(
                "standard_alpha",
                format!("{}", self.standard_alpha),
            ));
        }
        if self.window_size < 3 {
            return Err(StabilizeError::invalid_parameter(
                "window_size",
                format!("{}", self.window_size),
            ));
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────
//  TripodDetector
// ─────────────────────────────────────────────────────────────────

/// Offline classifier for entire motion sequences.
///
/// Analyses a slice of [`MotionSample`]s and classifies each frame using a
/// rolling window of statistics.
#[derive(Debug)]
pub struct TripodDetector {
    config: TripodConfig,
}

impl TripodDetector {
    /// Create a new detector with the given configuration.
    #[must_use]
    pub fn new(config: TripodConfig) -> Self {
        Self { config }
    }

    /// Classify each frame in `samples` and return a per-frame class vector.
    ///
    /// Each entry in the output corresponds to the same index in `samples`.
    #[must_use]
    pub fn classify(&self, samples: &[MotionSample]) -> Vec<MotionClass> {
        let n = samples.len();
        if n == 0 {
            return Vec::new();
        }

        let mut classes = vec![MotionClass::Unknown; n];
        let hw = self.config.window_size / 2;

        for i in 0..n {
            let start = i.saturating_sub(hw);
            let end = (i + hw + 1).min(n);
            let window = &samples[start..end];
            classes[i] = self.classify_window(window);
        }

        classes
    }

    /// Classify a single window of samples.
    fn classify_window(&self, window: &[MotionSample]) -> MotionClass {
        if window.is_empty() {
            return MotionClass::Unknown;
        }

        let mags: Vec<f64> = window.iter().map(|s| s.magnitude()).collect();
        let n = mags.len() as f64;

        let mean = mags.iter().sum::<f64>() / n;
        let variance = mags.iter().map(|m| (m - mean).powi(2)).sum::<f64>() / n;
        let stddev = variance.sqrt();
        let static_frac = mags
            .iter()
            .filter(|&&m| m < self.config.jitter_threshold)
            .count() as f64
            / n;

        // Check for panning: high mean motion with consistent direction
        if mean >= self.config.pan_mean_threshold {
            let dir_mean = circular_mean(window.iter().map(|s| s.direction()));
            let dir_var = window
                .iter()
                .map(|s| {
                    let d = angle_diff(s.direction(), dir_mean);
                    d * d
                })
                .sum::<f64>()
                / n;

            if dir_var < self.config.pan_direction_variance {
                return MotionClass::Panning;
            }
        }

        // Check for static / tripod
        if mean <= self.config.max_mean_motion
            && stddev <= self.config.max_stddev
            && static_frac >= self.config.min_static_fraction
        {
            return MotionClass::Tripod;
        }

        MotionClass::HandHeld
    }

    /// Compute a global classification for the entire sequence.
    ///
    /// Returns the most common class across all frames.
    #[must_use]
    pub fn global_class(&self, samples: &[MotionSample]) -> MotionClass {
        let classes = self.classify(samples);
        let tripod = classes
            .iter()
            .filter(|&&c| c == MotionClass::Tripod)
            .count();
        let panning = classes
            .iter()
            .filter(|&&c| c == MotionClass::Panning)
            .count();
        let handheld = classes
            .iter()
            .filter(|&&c| c == MotionClass::HandHeld)
            .count();

        if tripod >= handheld && tripod >= panning {
            MotionClass::Tripod
        } else if panning >= handheld {
            MotionClass::Panning
        } else {
            MotionClass::HandHeld
        }
    }
}

// ─────────────────────────────────────────────────────────────────
//  Per-frame stabilization output
// ─────────────────────────────────────────────────────────────────

/// Stabilization correction for one frame, produced by [`TripodStabilizer`].
#[derive(Debug, Clone, Copy)]
pub struct TripodCorrection {
    /// Frame index.
    pub index: usize,
    /// Horizontal correction (pixels).
    pub correction_dx: f64,
    /// Vertical correction (pixels).
    pub correction_dy: f64,
    /// Angular correction (radians).
    pub correction_angle: f64,
    /// Motion class assigned to this frame.
    pub motion_class: MotionClass,
    /// True if tripod lock-on was applied (vs. standard smoothing).
    pub tripod_locked: bool,
}

/// Summary report from [`TripodStabilizer::process`].
#[derive(Debug, Clone)]
pub struct TripodReport {
    /// Per-frame correction records.
    pub corrections: Vec<TripodCorrection>,
    /// Global motion classification.
    pub global_class: MotionClass,
    /// True if tripod mode was detected for the majority of the sequence.
    pub is_tripod_detected: bool,
    /// Fraction of frames in tripod mode.
    pub tripod_fraction: f64,
    /// Reference (lock-on) position used when in tripod mode.
    pub lock_position: (f64, f64),
    /// Average correction magnitude applied (pixels).
    pub avg_correction_px: f64,
}

// ─────────────────────────────────────────────────────────────────
//  TripodStabilizer
// ─────────────────────────────────────────────────────────────────

/// Offline tripod-aware stabilizer.
///
/// First classifies the entire sequence; then applies lock-on correction for
/// static segments and standard EMA smoothing for non-static segments.
#[derive(Debug)]
pub struct TripodStabilizer {
    config: TripodConfig,
    detector: TripodDetector,
}

impl TripodStabilizer {
    /// Create a new `TripodStabilizer` with the given configuration.
    #[must_use]
    pub fn new(config: TripodConfig) -> Self {
        let detector = TripodDetector::new(config.clone());
        Self { config, detector }
    }

    /// Process a full sequence of motion samples and return a [`TripodReport`].
    ///
    /// # Panics
    ///
    /// Does not panic.
    #[must_use]
    pub fn process(&mut self, samples: &[MotionSample]) -> TripodReport {
        let n = samples.len();
        if n == 0 {
            return TripodReport {
                corrections: Vec::new(),
                global_class: MotionClass::Unknown,
                is_tripod_detected: false,
                tripod_fraction: 0.0,
                lock_position: (0.0, 0.0),
                avg_correction_px: 0.0,
            };
        }

        // Step 1: classify each frame
        let classes = self.detector.classify(samples);
        let global = self.detector.global_class(samples);

        // Step 2: compute cumulative trajectory (absolute position)
        let mut traj_x = 0.0f64;
        let mut traj_y = 0.0f64;
        let mut traj_angle = 0.0f64;
        let mut trajectory: Vec<(f64, f64, f64)> = Vec::with_capacity(n);
        for s in samples {
            traj_x += s.dx;
            traj_y += s.dy;
            traj_angle += s.angle;
            trajectory.push((traj_x, traj_y, traj_angle));
        }

        // Step 3: compute lock-on reference position from the first `lock_window` frames
        let lock_n = self.config.lock_window.min(n);
        let (lock_x, lock_y, lock_angle) = if lock_n > 0 {
            let sum_x = trajectory[..lock_n].iter().map(|p| p.0).sum::<f64>() / lock_n as f64;
            let sum_y = trajectory[..lock_n].iter().map(|p| p.1).sum::<f64>() / lock_n as f64;
            let sum_a = trajectory[..lock_n].iter().map(|p| p.2).sum::<f64>() / lock_n as f64;
            (sum_x, sum_y, sum_a)
        } else {
            (trajectory[0].0, trajectory[0].1, trajectory[0].2)
        };

        // Step 4: EMA smoother for non-tripod frames
        let mut smooth_x = trajectory[0].0;
        let mut smooth_y = trajectory[0].1;
        let mut smooth_a = trajectory[0].2;

        // Step 5: hysteresis state for exiting tripod mode
        let mut in_tripod = classes[0] == MotionClass::Tripod;
        let mut handheld_streak = 0usize;

        let mut corrections: Vec<TripodCorrection> = Vec::with_capacity(n);
        let mut total_corr = 0.0;
        let mut tripod_frame_count = 0usize;

        for i in 0..n {
            let (raw_x, raw_y, raw_a) = trajectory[i];
            let class = classes[i];

            // Update hysteresis
            if class == MotionClass::Tripod {
                in_tripod = true;
                handheld_streak = 0;
            } else if in_tripod {
                handheld_streak += 1;
                if handheld_streak >= self.config.exit_hysteresis {
                    in_tripod = false;
                    handheld_streak = 0;
                }
            }

            let (cdx, cdy, ca, locked) = if in_tripod {
                tripod_frame_count += 1;
                // Lock-on: drive raw trajectory back to lock reference
                let alpha = self.config.tripod_alpha;
                smooth_x = alpha * lock_x + (1.0 - alpha) * smooth_x;
                smooth_y = alpha * lock_y + (1.0 - alpha) * smooth_y;
                smooth_a = alpha * lock_angle + (1.0 - alpha) * smooth_a;
                (smooth_x - raw_x, smooth_y - raw_y, smooth_a - raw_a, true)
            } else {
                // Standard EMA smoothing
                let alpha = self.config.standard_alpha;
                smooth_x = alpha * raw_x + (1.0 - alpha) * smooth_x;
                smooth_y = alpha * raw_y + (1.0 - alpha) * smooth_y;
                smooth_a = alpha * raw_a + (1.0 - alpha) * smooth_a;
                (smooth_x - raw_x, smooth_y - raw_y, smooth_a - raw_a, false)
            };

            let corr_mag = (cdx * cdx + cdy * cdy).sqrt();
            total_corr += corr_mag;

            corrections.push(TripodCorrection {
                index: i,
                correction_dx: cdx,
                correction_dy: cdy,
                correction_angle: ca,
                motion_class: class,
                tripod_locked: locked,
            });
        }

        let tripod_fraction = tripod_frame_count as f64 / n as f64;
        let avg_correction_px = total_corr / n as f64;

        TripodReport {
            corrections,
            global_class: global,
            is_tripod_detected: tripod_fraction >= 0.5,
            tripod_fraction,
            lock_position: (lock_x, lock_y),
            avg_correction_px,
        }
    }
}

// ─────────────────────────────────────────────────────────────────
//  Real-time online tripod stabilizer
// ─────────────────────────────────────────────────────────────────

/// Online (streaming) tripod-aware stabilizer.
///
/// Classifies each incoming frame in real time using a causal rolling window
/// and applies lock-on or standard smoothing accordingly.
#[derive(Debug)]
pub struct OnlineTripodStabilizer {
    config: TripodConfig,
    /// Rolling window of recent motion magnitudes.
    motion_window: std::collections::VecDeque<f64>,
    /// Rolling window of recent motion directions.
    direction_window: std::collections::VecDeque<f64>,
    /// Current absolute trajectory position.
    traj_x: f64,
    traj_y: f64,
    traj_angle: f64,
    /// Current EMA smoothed position.
    smooth_x: f64,
    smooth_y: f64,
    smooth_angle: f64,
    /// Lock-on reference.
    lock_x: f64,
    lock_y: f64,
    lock_angle: f64,
    lock_initialized: bool,
    lock_sample_count: usize,
    lock_sum_x: f64,
    lock_sum_y: f64,
    lock_sum_angle: f64,
    /// Current state.
    in_tripod: bool,
    handheld_streak: usize,
    frame_count: usize,
    ema_initialized: bool,
}

impl OnlineTripodStabilizer {
    /// Create a new online stabilizer.
    #[must_use]
    pub fn new(config: TripodConfig) -> Self {
        let cap = config.window_size;
        Self {
            config,
            motion_window: std::collections::VecDeque::with_capacity(cap),
            direction_window: std::collections::VecDeque::with_capacity(cap),
            traj_x: 0.0,
            traj_y: 0.0,
            traj_angle: 0.0,
            smooth_x: 0.0,
            smooth_y: 0.0,
            smooth_angle: 0.0,
            lock_x: 0.0,
            lock_y: 0.0,
            lock_angle: 0.0,
            lock_initialized: false,
            lock_sample_count: 0,
            lock_sum_x: 0.0,
            lock_sum_y: 0.0,
            lock_sum_angle: 0.0,
            in_tripod: false,
            handheld_streak: 0,
            frame_count: 0,
            ema_initialized: false,
        }
    }

    /// Push a motion sample and receive the stabilization correction.
    pub fn push(&mut self, sample: MotionSample) -> TripodCorrection {
        let idx = self.frame_count;
        self.frame_count += 1;

        // Integrate trajectory
        self.traj_x += sample.dx;
        self.traj_y += sample.dy;
        self.traj_angle += sample.angle;

        // Initialize EMA with first position
        if !self.ema_initialized {
            self.smooth_x = self.traj_x;
            self.smooth_y = self.traj_y;
            self.smooth_angle = self.traj_angle;
            self.ema_initialized = true;
        }

        // Accumulate lock-on reference from first `lock_window` frames
        if !self.lock_initialized {
            self.lock_sum_x += self.traj_x;
            self.lock_sum_y += self.traj_y;
            self.lock_sum_angle += self.traj_angle;
            self.lock_sample_count += 1;
            if self.lock_sample_count >= self.config.lock_window {
                let k = self.lock_sample_count as f64;
                self.lock_x = self.lock_sum_x / k;
                self.lock_y = self.lock_sum_y / k;
                self.lock_angle = self.lock_sum_angle / k;
                self.lock_initialized = true;
            }
        }

        // Update rolling motion window
        let mag = sample.magnitude();
        let dir = sample.direction();
        self.motion_window.push_back(mag);
        self.direction_window.push_back(dir);
        while self.motion_window.len() > self.config.window_size {
            self.motion_window.pop_front();
            self.direction_window.pop_front();
        }

        // Classify current window
        let class = self.classify_current_window();

        // Update hysteresis
        if class == MotionClass::Tripod {
            self.in_tripod = true;
            self.handheld_streak = 0;
        } else if self.in_tripod {
            self.handheld_streak += 1;
            if self.handheld_streak >= self.config.exit_hysteresis {
                self.in_tripod = false;
                self.handheld_streak = 0;
            }
        }

        // Compute correction
        let (cdx, cdy, ca, locked) = if self.in_tripod && self.lock_initialized {
            let alpha = self.config.tripod_alpha;
            self.smooth_x = alpha * self.lock_x + (1.0 - alpha) * self.smooth_x;
            self.smooth_y = alpha * self.lock_y + (1.0 - alpha) * self.smooth_y;
            self.smooth_angle = alpha * self.lock_angle + (1.0 - alpha) * self.smooth_angle;
            (
                self.smooth_x - self.traj_x,
                self.smooth_y - self.traj_y,
                self.smooth_angle - self.traj_angle,
                true,
            )
        } else {
            let alpha = self.config.standard_alpha;
            self.smooth_x = alpha * self.traj_x + (1.0 - alpha) * self.smooth_x;
            self.smooth_y = alpha * self.traj_y + (1.0 - alpha) * self.smooth_y;
            self.smooth_angle = alpha * self.traj_angle + (1.0 - alpha) * self.smooth_angle;
            (
                self.smooth_x - self.traj_x,
                self.smooth_y - self.traj_y,
                self.smooth_angle - self.traj_angle,
                false,
            )
        };

        TripodCorrection {
            index: idx,
            correction_dx: cdx,
            correction_dy: cdy,
            correction_angle: ca,
            motion_class: class,
            tripod_locked: locked,
        }
    }

    /// Reset the stabilizer to its initial state.
    pub fn reset(&mut self) {
        self.motion_window.clear();
        self.direction_window.clear();
        self.traj_x = 0.0;
        self.traj_y = 0.0;
        self.traj_angle = 0.0;
        self.smooth_x = 0.0;
        self.smooth_y = 0.0;
        self.smooth_angle = 0.0;
        self.lock_initialized = false;
        self.lock_sample_count = 0;
        self.lock_sum_x = 0.0;
        self.lock_sum_y = 0.0;
        self.lock_sum_angle = 0.0;
        self.in_tripod = false;
        self.handheld_streak = 0;
        self.frame_count = 0;
        self.ema_initialized = false;
    }

    /// Whether the stabilizer is currently in tripod lock-on mode.
    #[must_use]
    pub const fn is_in_tripod_mode(&self) -> bool {
        self.in_tripod
    }

    fn classify_current_window(&self) -> MotionClass {
        let mags: Vec<f64> = self.motion_window.iter().copied().collect();
        if mags.is_empty() {
            return MotionClass::Unknown;
        }
        let n = mags.len() as f64;
        let mean = mags.iter().sum::<f64>() / n;
        let variance = mags.iter().map(|m| (m - mean).powi(2)).sum::<f64>() / n;
        let stddev = variance.sqrt();
        let static_frac = mags
            .iter()
            .filter(|&&m| m < self.config.jitter_threshold)
            .count() as f64
            / n;

        if mean >= self.config.pan_mean_threshold {
            let dirs: Vec<f64> = self.direction_window.iter().copied().collect();
            let dir_mean = circular_mean(dirs.iter().copied());
            let dir_var = dirs
                .iter()
                .map(|d| {
                    let diff = angle_diff(*d, dir_mean);
                    diff * diff
                })
                .sum::<f64>()
                / n;
            if dir_var < self.config.pan_direction_variance {
                return MotionClass::Panning;
            }
        }

        if mean <= self.config.max_mean_motion
            && stddev <= self.config.max_stddev
            && static_frac >= self.config.min_static_fraction
        {
            return MotionClass::Tripod;
        }

        MotionClass::HandHeld
    }
}

// ─────────────────────────────────────────────────────────────────
//  Helper math
// ─────────────────────────────────────────────────────────────────

/// Compute the circular mean of an iterator of angles (radians).
fn circular_mean(angles: impl Iterator<Item = f64>) -> f64 {
    let mut sin_sum = 0.0;
    let mut cos_sum = 0.0;
    for a in angles {
        sin_sum += a.sin();
        cos_sum += a.cos();
    }
    sin_sum.atan2(cos_sum)
}

/// Compute the signed angular difference `a - b` wrapped to `(-π, π]`.
fn angle_diff(a: f64, b: f64) -> f64 {
    let diff = (a - b).rem_euclid(2.0 * PI);
    if diff > PI {
        diff - 2.0 * PI
    } else {
        diff
    }
}

// ─────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn static_samples(n: usize) -> Vec<MotionSample> {
        (0..n)
            .map(|i| {
                // Tiny alternating noise — clearly static
                let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                MotionSample::translation(sign * 0.3, sign * 0.2)
            })
            .collect()
    }

    fn handheld_samples(n: usize) -> Vec<MotionSample> {
        (0..n)
            .map(|i| {
                let t = i as f64 * 0.7;
                MotionSample::translation(t.sin() * 8.0, t.cos() * 6.0)
            })
            .collect()
    }

    fn panning_samples(n: usize) -> Vec<MotionSample> {
        // Consistent rightward pan
        (0..n)
            .map(|_| MotionSample::translation(15.0, 0.2))
            .collect()
    }

    // ── MotionSample ────────────────────────────────────────────────

    #[test]
    fn test_motion_sample_magnitude() {
        let s = MotionSample::translation(3.0, 4.0);
        assert!((s.magnitude() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_motion_sample_direction_zero() {
        let s = MotionSample::translation(0.0, 0.0);
        // atan2(0,0) is 0 by convention in most platforms
        let _ = s.direction(); // just ensure no panic
    }

    // ── TripodConfig ────────────────────────────────────────────────

    #[test]
    fn test_config_validate_ok() {
        assert!(TripodConfig::default().validate().is_ok());
    }

    #[test]
    fn test_config_validate_bad_alpha() {
        let cfg = TripodConfig {
            standard_alpha: 0.0,
            ..TripodConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    // ── TripodDetector ──────────────────────────────────────────────

    #[test]
    fn test_detect_static_as_tripod() {
        let cfg = TripodConfig::default();
        let detector = TripodDetector::new(cfg);
        let samples = static_samples(60);
        let global = detector.global_class(&samples);
        assert_eq!(
            global,
            MotionClass::Tripod,
            "static motion should be Tripod"
        );
    }

    #[test]
    fn test_detect_handheld() {
        let cfg = TripodConfig::default();
        let detector = TripodDetector::new(cfg);
        let samples = handheld_samples(60);
        let global = detector.global_class(&samples);
        // High random motion → HandHeld or Panning (not Tripod)
        assert_ne!(
            global,
            MotionClass::Tripod,
            "random large motion should not be Tripod"
        );
    }

    #[test]
    fn test_detect_panning() {
        let cfg = TripodConfig::default();
        let detector = TripodDetector::new(cfg);
        let samples = panning_samples(60);
        let global = detector.global_class(&samples);
        assert_eq!(
            global,
            MotionClass::Panning,
            "consistent pan should be Panning"
        );
    }

    #[test]
    fn test_classify_per_frame_length() {
        let cfg = TripodConfig::default();
        let detector = TripodDetector::new(cfg);
        let samples = static_samples(40);
        let classes = detector.classify(&samples);
        assert_eq!(classes.len(), 40);
    }

    #[test]
    fn test_classify_empty() {
        let cfg = TripodConfig::default();
        let detector = TripodDetector::new(cfg);
        assert!(detector.classify(&[]).is_empty());
    }

    // ── TripodStabilizer ────────────────────────────────────────────

    #[test]
    fn test_stabilizer_tripod_detection() {
        let cfg = TripodConfig::default();
        let mut stab = TripodStabilizer::new(cfg);
        let samples = static_samples(60);
        let report = stab.process(&samples);
        assert!(
            report.is_tripod_detected,
            "static sequence should trigger tripod mode"
        );
    }

    #[test]
    fn test_stabilizer_tripod_fraction_high() {
        let cfg = TripodConfig::default();
        let mut stab = TripodStabilizer::new(cfg);
        let samples = static_samples(60);
        let report = stab.process(&samples);
        assert!(report.tripod_fraction > 0.5);
    }

    #[test]
    fn test_stabilizer_corrections_length() {
        let cfg = TripodConfig::default();
        let mut stab = TripodStabilizer::new(cfg);
        let samples = static_samples(30);
        let report = stab.process(&samples);
        assert_eq!(report.corrections.len(), 30);
    }

    #[test]
    fn test_stabilizer_locked_frames_in_tripod_mode() {
        let cfg = TripodConfig::default();
        let mut stab = TripodStabilizer::new(cfg);
        let samples = static_samples(60);
        let report = stab.process(&samples);
        let locked = report
            .corrections
            .iter()
            .filter(|c| c.tripod_locked)
            .count();
        assert!(locked > 0, "at least some frames should be tripod-locked");
    }

    #[test]
    fn test_stabilizer_empty_sequence() {
        let cfg = TripodConfig::default();
        let mut stab = TripodStabilizer::new(cfg);
        let report = stab.process(&[]);
        assert!(!report.is_tripod_detected);
        assert!(report.corrections.is_empty());
    }

    #[test]
    fn test_stabilizer_corrections_indices_sequential() {
        let cfg = TripodConfig::default();
        let mut stab = TripodStabilizer::new(cfg);
        let samples = handheld_samples(20);
        let report = stab.process(&samples);
        for (i, c) in report.corrections.iter().enumerate() {
            assert_eq!(c.index, i);
        }
    }

    #[test]
    fn test_stabilizer_global_class_panning() {
        let cfg = TripodConfig::default();
        let mut stab = TripodStabilizer::new(cfg);
        let samples = panning_samples(60);
        let report = stab.process(&samples);
        assert_eq!(report.global_class, MotionClass::Panning);
    }

    // ── OnlineTripodStabilizer ──────────────────────────────────────

    #[test]
    fn test_online_stabilizer_no_panic() {
        let cfg = TripodConfig::default();
        let mut stab = OnlineTripodStabilizer::new(cfg);
        for s in static_samples(30) {
            let _ = stab.push(s);
        }
    }

    #[test]
    fn test_online_stabilizer_enters_tripod_mode() {
        let cfg = TripodConfig::default();
        let mut stab = OnlineTripodStabilizer::new(cfg);
        for s in static_samples(40) {
            stab.push(s);
        }
        assert!(
            stab.is_in_tripod_mode(),
            "should enter tripod mode for static input"
        );
    }

    #[test]
    fn test_online_stabilizer_exits_tripod_on_motion() {
        let mut cfg = TripodConfig::default();
        cfg.exit_hysteresis = 3;
        let mut stab = OnlineTripodStabilizer::new(cfg);

        // Enter tripod mode
        for s in static_samples(40) {
            stab.push(s);
        }
        assert!(stab.is_in_tripod_mode());

        // Large motion for exit_hysteresis+1 frames
        for s in handheld_samples(5) {
            stab.push(s);
        }
        // After hysteresis frames of large motion, should exit tripod
        assert!(
            !stab.is_in_tripod_mode(),
            "should exit tripod mode after large motion"
        );
    }

    #[test]
    fn test_online_stabilizer_reset() {
        let cfg = TripodConfig::default();
        let mut stab = OnlineTripodStabilizer::new(cfg);
        for s in static_samples(20) {
            stab.push(s);
        }
        stab.reset();
        assert_eq!(stab.frame_count, 0);
        assert!(!stab.is_in_tripod_mode());
    }

    #[test]
    fn test_online_output_indices_sequential() {
        let cfg = TripodConfig::default();
        let mut stab = OnlineTripodStabilizer::new(cfg);
        let samples: Vec<MotionSample> = static_samples(15)
            .into_iter()
            .chain(handheld_samples(15))
            .collect();

        for (expected_idx, s) in samples.iter().enumerate() {
            let c = stab.push(*s);
            assert_eq!(c.index, expected_idx);
        }
    }

    // ── Angle helpers ───────────────────────────────────────────────

    #[test]
    fn test_angle_diff_wrap() {
        let d = angle_diff(PI * 0.1, PI * 1.9);
        // Should be ~0.2π not ~-1.8π
        assert!(d.abs() < PI);
    }

    #[test]
    fn test_circular_mean_all_same() {
        let angles = vec![1.0f64; 5];
        let mean = circular_mean(angles.into_iter());
        assert!((mean - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_motion_class_labels() {
        assert_eq!(MotionClass::Tripod.label(), "Tripod");
        assert_eq!(MotionClass::Panning.label(), "Panning");
        assert!(MotionClass::Tripod.should_lock());
        assert!(!MotionClass::HandHeld.should_lock());
    }
}
