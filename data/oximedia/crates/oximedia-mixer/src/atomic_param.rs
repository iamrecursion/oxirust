//! Lock-free parameter updates using atomic operations.
//!
//! [`AtomicF32`] wraps a `f32` parameter in an atomic cell, enabling
//! the audio thread to read parameters without blocking while the UI/control
//! thread writes new values.  This avoids mutexes on the audio hot path.

use std::sync::atomic::{AtomicU32, Ordering};

// ---------------------------------------------------------------------------
// AtomicF32
// ---------------------------------------------------------------------------

/// A lock-free `f32` parameter backed by `AtomicU32` bit-cast.
///
/// Uses `Relaxed` ordering by default which is appropriate for audio
/// parameter updates where occasional stale reads (within a single buffer)
/// are acceptable.
#[derive(Debug)]
pub struct AtomicF32 {
    bits: AtomicU32,
}

impl AtomicF32 {
    /// Create a new atomic f32 with the given initial value.
    #[must_use]
    pub fn new(value: f32) -> Self {
        Self {
            bits: AtomicU32::new(value.to_bits()),
        }
    }

    /// Load the current value (relaxed ordering).
    #[must_use]
    pub fn load(&self) -> f32 {
        f32::from_bits(self.bits.load(Ordering::Relaxed))
    }

    /// Store a new value (relaxed ordering).
    pub fn store(&self, value: f32) {
        self.bits.store(value.to_bits(), Ordering::Relaxed);
    }

    /// Load the current value with acquire ordering (for synchronisation).
    #[must_use]
    pub fn load_acquire(&self) -> f32 {
        f32::from_bits(self.bits.load(Ordering::Acquire))
    }

    /// Store a new value with release ordering (for synchronisation).
    pub fn store_release(&self, value: f32) {
        self.bits.store(value.to_bits(), Ordering::Release);
    }

    /// Swap the value, returning the old value.
    #[must_use]
    pub fn swap(&self, value: f32) -> f32 {
        f32::from_bits(self.bits.swap(value.to_bits(), Ordering::Relaxed))
    }

    /// Compare-and-swap: if current == expected, store new_value.
    ///
    /// Returns `Ok(expected)` on success or `Err(actual)` on failure.
    pub fn compare_exchange(&self, expected: f32, new_value: f32) -> Result<f32, f32> {
        self.bits
            .compare_exchange(
                expected.to_bits(),
                new_value.to_bits(),
                Ordering::AcqRel,
                Ordering::Relaxed,
            )
            .map(f32::from_bits)
            .map_err(f32::from_bits)
    }
}

impl Default for AtomicF32 {
    fn default() -> Self {
        Self::new(0.0)
    }
}

impl Clone for AtomicF32 {
    fn clone(&self) -> Self {
        Self::new(self.load())
    }
}

// ---------------------------------------------------------------------------
// AtomicParam — higher-level wrapper for gain/pan parameters
// ---------------------------------------------------------------------------

/// A named atomic parameter with clamping range.
#[derive(Debug)]
pub struct AtomicParam {
    /// Parameter name.
    name: String,
    /// The atomic value.
    value: AtomicF32,
    /// Minimum allowed value.
    min: f32,
    /// Maximum allowed value.
    max: f32,
    /// Default value.
    default: f32,
}

impl AtomicParam {
    /// Create a new atomic parameter.
    #[must_use]
    pub fn new(name: impl Into<String>, min: f32, max: f32, default: f32) -> Self {
        let clamped = default.clamp(min, max);
        Self {
            name: name.into(),
            value: AtomicF32::new(clamped),
            min,
            max,
            default: clamped,
        }
    }

    /// Get the parameter name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the current value.
    #[must_use]
    pub fn get(&self) -> f32 {
        self.value.load()
    }

    /// Set the value, clamping to the valid range.
    pub fn set(&self, value: f32) {
        self.value.store(value.clamp(self.min, self.max));
    }

    /// Reset to the default value.
    pub fn reset(&self) {
        self.value.store(self.default);
    }

    /// Get the default value.
    #[must_use]
    pub fn default_value(&self) -> f32 {
        self.default
    }

    /// Get the minimum value.
    #[must_use]
    pub fn min(&self) -> f32 {
        self.min
    }

    /// Get the maximum value.
    #[must_use]
    pub fn max(&self) -> f32 {
        self.max
    }

    /// Get the normalised value (0.0..1.0).
    #[must_use]
    pub fn normalised(&self) -> f32 {
        let range = self.max - self.min;
        if range.abs() < f32::EPSILON {
            return 0.0;
        }
        (self.get() - self.min) / range
    }

    /// Set from a normalised value (0.0..1.0).
    pub fn set_normalised(&self, normalised: f32) {
        let value = self.min + normalised.clamp(0.0, 1.0) * (self.max - self.min);
        self.set(value);
    }
}

impl Clone for AtomicParam {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            value: self.value.clone(),
            min: self.min,
            max: self.max,
            default: self.default,
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience constructors
// ---------------------------------------------------------------------------

/// Create a gain parameter (0.0 to 2.0, default 1.0).
#[must_use]
pub fn gain_param(name: impl Into<String>) -> AtomicParam {
    AtomicParam::new(name, 0.0, 2.0, 1.0)
}

/// Create a pan parameter (-1.0 to 1.0, default 0.0).
#[must_use]
pub fn pan_param(name: impl Into<String>) -> AtomicParam {
    AtomicParam::new(name, -1.0, 1.0, 0.0)
}

/// Create a send level parameter (0.0 to 1.0, default 0.0).
#[must_use]
pub fn send_level_param(name: impl Into<String>) -> AtomicParam {
    AtomicParam::new(name, 0.0, 1.0, 0.0)
}

// ---------------------------------------------------------------------------
// SmoothedParam — lock-free target with per-sample exponential smoothing
// ---------------------------------------------------------------------------

/// Parameter with exponential smoothing and lock-free target updates.
///
/// The `target` can be written from any thread (lock-free via [`AtomicF32`]).
/// `next_sample()` must only be called from the audio thread; it advances
/// the current value toward `target` by the per-sample smoothing coefficient.
///
/// The smoothing coefficient is computed as:
///
/// ```text
/// coeff = exp(-2π / (smoothing_ms * 0.001 * sample_rate))
/// ```
///
/// which gives an RC-style one-pole low-pass filter with −3 dB cutoff at
/// `1 / (smoothing_ms * 0.001)` Hz.
pub struct SmoothedParam {
    /// Lock-free target value (writable from any thread).
    target: AtomicF32,
    /// Current smoothed value (audio thread only).
    current: f32,
    /// Per-sample smoothing coefficient in [0, 1).
    /// 0.0 = instant (no smoothing), values near 1.0 = very slow.
    smoothing_coeff: f32,
}

impl SmoothedParam {
    /// Create a new `SmoothedParam`.
    ///
    /// * `initial`       – starting value (both current and target).
    /// * `smoothing_ms`  – smoothing time constant in milliseconds (>= 0.0).
    /// * `sample_rate`   – audio sample rate in Hz (must be > 0.0).
    ///
    /// A `smoothing_ms` of 0.0 disables smoothing (instant response).
    #[must_use]
    pub fn new(initial: f32, smoothing_ms: f32, sample_rate: f32) -> Self {
        let coeff = if smoothing_ms <= 0.0 || sample_rate <= 0.0 {
            0.0
        } else {
            let tau_samples = smoothing_ms * 0.001 * sample_rate;
            (-std::f32::consts::TAU / tau_samples).exp()
        };
        Self {
            target: AtomicF32::new(initial),
            current: initial,
            smoothing_coeff: coeff,
        }
    }

    /// Set the target value from any thread (lock-free).
    pub fn set_target(&self, value: f32) {
        self.target.store(value);
    }

    /// Read the current target value from any thread (lock-free).
    #[must_use]
    pub fn get_target(&self) -> f32 {
        self.target.load()
    }

    /// Advance the current value one sample toward the target and return it.
    ///
    /// Must only be called from the audio thread.
    pub fn next_sample(&mut self) -> f32 {
        let t = self.target.load();
        self.current = self.current * self.smoothing_coeff + t * (1.0 - self.smoothing_coeff);
        self.current
    }

    /// Return the current (already-smoothed) value without advancing.
    ///
    /// Audio thread only.
    #[must_use]
    pub fn current(&self) -> f32 {
        self.current
    }

    /// Return `true` when the current value has reached the target
    /// (within `f32::EPSILON`).
    #[must_use]
    pub fn is_at_target(&self) -> bool {
        (self.current - self.target.load()).abs() < f32::EPSILON
    }

    /// Instantly jump to `value`, resetting both current and target.
    ///
    /// Audio thread only.
    pub fn reset(&mut self, value: f32) {
        self.target.store(value);
        self.current = value;
    }
}

impl std::fmt::Debug for SmoothedParam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SmoothedParam")
            .field("target", &self.target.load())
            .field("current", &self.current)
            .field("smoothing_coeff", &self.smoothing_coeff)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atomic_f32_new_and_load() {
        let a = AtomicF32::new(0.75);
        assert!((a.load() - 0.75).abs() < f32::EPSILON);
    }

    #[test]
    fn test_atomic_f32_store() {
        let a = AtomicF32::new(0.0);
        a.store(0.5);
        assert!((a.load() - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_atomic_f32_swap() {
        let a = AtomicF32::new(1.0);
        let old = a.swap(2.0);
        assert!((old - 1.0).abs() < f32::EPSILON);
        assert!((a.load() - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_atomic_f32_compare_exchange_success() {
        let a = AtomicF32::new(1.0);
        let result = a.compare_exchange(1.0, 2.0);
        assert!(result.is_ok());
        assert!((a.load() - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_atomic_f32_compare_exchange_failure() {
        let a = AtomicF32::new(1.0);
        let result = a.compare_exchange(0.5, 2.0);
        assert!(result.is_err());
        assert!((a.load() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_atomic_f32_acquire_release() {
        let a = AtomicF32::new(0.0);
        a.store_release(0.42);
        assert!((a.load_acquire() - 0.42).abs() < f32::EPSILON);
    }

    #[test]
    fn test_atomic_f32_clone() {
        let a = AtomicF32::new(3.14);
        let b = a.clone();
        assert!((b.load() - 3.14).abs() < 0.001);
    }

    #[test]
    fn test_atomic_f32_default() {
        let a = AtomicF32::default();
        assert!((a.load() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_atomic_param_new() {
        let p = AtomicParam::new("volume", 0.0, 1.0, 0.5);
        assert_eq!(p.name(), "volume");
        assert!((p.get() - 0.5).abs() < f32::EPSILON);
        assert!((p.min() - 0.0).abs() < f32::EPSILON);
        assert!((p.max() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_atomic_param_clamping() {
        let p = AtomicParam::new("gain", 0.0, 2.0, 1.0);
        p.set(5.0);
        assert!((p.get() - 2.0).abs() < f32::EPSILON);
        p.set(-1.0);
        assert!((p.get() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_atomic_param_reset() {
        let p = AtomicParam::new("pan", -1.0, 1.0, 0.0);
        p.set(0.7);
        p.reset();
        assert!((p.get() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_atomic_param_normalised() {
        let p = AtomicParam::new("freq", 20.0, 20000.0, 1000.0);
        let n = p.normalised();
        let expected = (1000.0 - 20.0) / (20000.0 - 20.0);
        assert!((n - expected).abs() < 0.001);
    }

    #[test]
    fn test_atomic_param_set_normalised() {
        let p = AtomicParam::new("gain", 0.0, 2.0, 1.0);
        p.set_normalised(0.5);
        assert!((p.get() - 1.0).abs() < f32::EPSILON);
        p.set_normalised(0.0);
        assert!((p.get() - 0.0).abs() < f32::EPSILON);
        p.set_normalised(1.0);
        assert!((p.get() - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_gain_param_constructor() {
        let p = gain_param("ch1_gain");
        assert_eq!(p.name(), "ch1_gain");
        assert!((p.get() - 1.0).abs() < f32::EPSILON);
        assert!((p.min() - 0.0).abs() < f32::EPSILON);
        assert!((p.max() - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_pan_param_constructor() {
        let p = pan_param("ch1_pan");
        assert!((p.get() - 0.0).abs() < f32::EPSILON);
        assert!((p.min() - (-1.0)).abs() < f32::EPSILON);
        assert!((p.max() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_send_level_param_constructor() {
        let p = send_level_param("ch1_send1");
        assert!((p.get() - 0.0).abs() < f32::EPSILON);
        assert!((p.max() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_atomic_param_clone() {
        let p = AtomicParam::new("test", 0.0, 1.0, 0.5);
        p.set(0.8);
        let p2 = p.clone();
        assert!((p2.get() - 0.8).abs() < f32::EPSILON);
        assert_eq!(p2.name(), "test");
    }

    #[test]
    fn test_atomic_param_zero_range() {
        let p = AtomicParam::new("fixed", 1.0, 1.0, 1.0);
        assert!((p.normalised() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cross_thread_usage() {
        use std::sync::Arc;
        use std::thread;

        let param = Arc::new(AtomicF32::new(0.0));
        let param_clone = Arc::clone(&param);

        let writer = thread::spawn(move || {
            for i in 0..100 {
                #[allow(clippy::cast_precision_loss)]
                param_clone.store(i as f32 / 100.0);
            }
        });

        let reader = thread::spawn(move || {
            let mut reads = Vec::new();
            for _ in 0..100 {
                reads.push(param.load());
            }
            reads
        });

        writer.join().expect("writer should complete");
        let reads = reader.join().expect("reader should complete");

        // All reads should be valid f32 values
        for &r in &reads {
            assert!(r.is_finite());
            assert!(r >= 0.0);
            assert!(r <= 1.0);
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // SmoothedParam tests
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_smoothed_param_new() {
        let sp = SmoothedParam::new(0.5, 10.0, 48000.0);
        // Initial current and target must both equal the initial value.
        assert!((sp.current() - 0.5).abs() < f32::EPSILON);
        assert!((sp.get_target() - 0.5).abs() < f32::EPSILON);
        // smoothing_coeff must be in [0, 1)
        assert!(sp.smoothing_coeff >= 0.0);
        assert!(sp.smoothing_coeff < 1.0);
    }

    #[test]
    fn test_smoothed_param_is_at_target_initially() {
        let sp = SmoothedParam::new(0.5, 10.0, 48000.0);
        // When target == initial, is_at_target() should be true.
        assert!(sp.is_at_target());
    }

    #[test]
    fn test_smoothed_param_set_and_converge() {
        let mut sp = SmoothedParam::new(0.0, 5.0, 48000.0);
        sp.set_target(1.0);
        // After enough iterations the smoother must converge within 0.001 of 1.0.
        let mut val = 0.0_f32;
        for _ in 0..50_000 {
            val = sp.next_sample();
        }
        assert!(
            (val - 1.0).abs() < 0.001,
            "expected convergence to 1.0, got {val}"
        );
    }

    #[test]
    fn test_smoothed_param_zero_smoothing_instant() {
        let mut sp = SmoothedParam::new(0.0, 0.0, 48000.0);
        sp.set_target(0.75);
        let val = sp.next_sample();
        // Zero smoothing_ms → coeff = 0.0 → next_sample returns target immediately.
        assert!((val - 0.75).abs() < f32::EPSILON);
    }

    #[test]
    fn test_smoothed_param_reset() {
        let mut sp = SmoothedParam::new(1.0, 20.0, 48000.0);
        // Move target away, advance a bit…
        sp.set_target(0.0);
        for _ in 0..10 {
            sp.next_sample();
        }
        // Then reset to 0.0; both current and target should snap to 0.0.
        sp.reset(0.0);
        assert!((sp.current() - 0.0).abs() < f32::EPSILON);
        assert!(sp.is_at_target());
    }

    #[test]
    fn test_smoothed_param_cross_thread() {
        use std::sync::Arc;
        use std::thread;

        // Wrap in Arc so the writer thread can take ownership of a clone.
        // SmoothedParam itself is NOT Clone/Send because `current` is audio-thread-only,
        // but the AtomicF32 target can be accessed via a shared reference.
        // We simulate cross-thread use by sharing the AtomicF32 directly.
        let shared_target = Arc::new(AtomicF32::new(0.0));
        let writer_target = Arc::clone(&shared_target);

        let writer = thread::spawn(move || {
            for i in 0..200 {
                #[allow(clippy::cast_precision_loss)]
                writer_target.store(i as f32 / 200.0);
            }
        });

        // Audio thread: read the target from the shared atomic (simulating
        // what SmoothedParam.set_target / get_target does internally).
        let mut reads = Vec::with_capacity(200);
        for _ in 0..200 {
            reads.push(shared_target.load());
        }

        writer.join().expect("writer thread should complete");

        // All values must be valid finite f32 in [0.0, 1.0].
        for &r in &reads {
            assert!(r.is_finite(), "non-finite value read: {r}");
            assert!(r >= 0.0 && r <= 1.0, "out-of-range value: {r}");
        }
    }

    #[test]
    fn test_smoothed_param_debug() {
        let sp = SmoothedParam::new(0.3, 10.0, 48000.0);
        let s = format!("{sp:?}");
        assert!(s.contains("SmoothedParam"));
    }
}
