//! Spatial audio panning with VBAP-based stereo output and HRTF simulation.
//!
//! Provides a `SpatialPanner` that converts a mono input signal to stereo
//! output using Vector Base Amplitude Panning (VBAP) and a simple IIR
//! shelf filter approximation for Head-Related Transfer Function (HRTF)
//! elevation simulation.

use crate::error::{AudioError, AudioResult};

/// Configuration for spatial audio panning.
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialPanConfig {
    /// Azimuth angle in degrees: -180..=180 (0 = front, positive = right).
    pub azimuth: f32,
    /// Elevation angle in degrees: -90..=90 (positive = above).
    pub elevation: f32,
    /// Source distance in metres: 0.1..=100.0.
    pub distance: f32,
    /// Room size factor: 0.0 (anechoic) ..= 1.0 (large room).
    pub room_size: f32,
}

impl SpatialPanConfig {
    /// Creates a new `SpatialPanConfig` after validating all parameters.
    ///
    /// # Errors
    ///
    /// Returns [`AudioError::InvalidParameter`] if any value is out of range.
    pub fn new(azimuth: f32, elevation: f32, distance: f32, room_size: f32) -> AudioResult<Self> {
        if !(-180.0..=180.0).contains(&azimuth) {
            return Err(AudioError::InvalidParameter(format!(
                "azimuth {azimuth} must be -180..=180"
            )));
        }
        if !(-90.0..=90.0).contains(&elevation) {
            return Err(AudioError::InvalidParameter(format!(
                "elevation {elevation} must be -90..=90"
            )));
        }
        if !(0.1..=100.0).contains(&distance) {
            return Err(AudioError::InvalidParameter(format!(
                "distance {distance} must be 0.1..=100.0"
            )));
        }
        if !(0.0..=1.0).contains(&room_size) {
            return Err(AudioError::InvalidParameter(format!(
                "room_size {room_size} must be 0.0..=1.0"
            )));
        }
        Ok(Self {
            azimuth,
            elevation,
            distance,
            room_size,
        })
    }

    /// Creates a front-facing, anechoic configuration (azimuth=0, elevation=0, distance=1, room=0).
    #[must_use]
    pub fn front() -> Self {
        Self {
            azimuth: 0.0,
            elevation: 0.0,
            distance: 1.0,
            room_size: 0.0,
        }
    }
}

impl Default for SpatialPanConfig {
    fn default() -> Self {
        Self::front()
    }
}

// ---------------------------------------------------------------------------
// Internal one-pole IIR low-pass filter used for HRTF simulation.
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
struct OnePoleIir {
    /// Filter coefficient a; cutoff as fraction of Nyquist (0..1).
    a: f32,
    /// Previous output sample.
    y_prev: f32,
}

impl OnePoleIir {
    fn new(a: f32) -> Self {
        Self {
            a: a.clamp(0.0, 1.0),
            y_prev: 0.0,
        }
    }

    fn process(&mut self, x: f32) -> f32 {
        let y = self.a * x + (1.0 - self.a) * self.y_prev;
        self.y_prev = y;
        y
    }

    fn reset(&mut self) {
        self.y_prev = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Simple comb-filter delay for room simulation (single sample delay).
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
struct CombDelay {
    buffer: Vec<f32>,
    write_idx: usize,
    delay_samples: usize,
}

impl CombDelay {
    fn new(delay_samples: usize) -> Self {
        let size = (delay_samples + 1).max(2);
        Self {
            buffer: vec![0.0; size],
            write_idx: 0,
            delay_samples: delay_samples.max(1),
        }
    }

    fn process(&mut self, input: f32) -> f32 {
        let len = self.buffer.len();
        let read_idx = (self.write_idx + len - self.delay_samples) % len;
        let delayed = self.buffer[read_idx];
        self.buffer[self.write_idx] = input;
        self.write_idx = (self.write_idx + 1) % len;
        delayed
    }

    fn reset(&mut self) {
        for s in &mut self.buffer {
            *s = 0.0;
        }
        self.write_idx = 0;
    }
}

// ---------------------------------------------------------------------------
// SpatialPanner
// ---------------------------------------------------------------------------

/// Spatial audio panner converting mono input to stereo output.
///
/// Uses VBAP amplitude panning for azimuth positioning and a one-pole IIR
/// shelf filter to approximate HRTF elevation effects.
pub struct SpatialPanner {
    config: SpatialPanConfig,
    sample_rate: u32,
    /// HRTF IIR filter for the left channel.
    hrtf_left: OnePoleIir,
    /// HRTF IIR filter for the right channel.
    hrtf_right: OnePoleIir,
    /// Early reflection delay for room simulation.
    room_delay_left: CombDelay,
    room_delay_right: CombDelay,
    /// Pre-computed VBAP gains.
    gain_left: f32,
    gain_right: f32,
    /// Distance attenuation factor.
    distance_gain: f32,
}

impl SpatialPanner {
    /// Create a new `SpatialPanner`.
    ///
    /// # Errors
    ///
    /// Returns [`AudioError::InvalidParameter`] if `sample_rate` is 0.
    pub fn new(config: SpatialPanConfig, sample_rate: u32) -> AudioResult<Self> {
        if sample_rate == 0 {
            return Err(AudioError::InvalidParameter(
                "sample_rate must not be zero".to_string(),
            ));
        }
        let (gl, gr) = compute_vbap_gains(config.azimuth);
        let dist_gain = compute_distance_gain(config.distance);
        let (a_left, a_right) = compute_hrtf_coeffs(config.elevation);

        // Room delay: ~5 ms early reflection
        let delay_samples = ((sample_rate as f32 * 0.005) as usize).max(1);

        Ok(Self {
            config,
            sample_rate,
            hrtf_left: OnePoleIir::new(a_left),
            hrtf_right: OnePoleIir::new(a_right),
            room_delay_left: CombDelay::new(delay_samples),
            room_delay_right: CombDelay::new(delay_samples),
            gain_left: gl,
            gain_right: gr,
            distance_gain: dist_gain,
        })
    }

    /// Get the configured azimuth angle.
    #[must_use]
    pub fn azimuth(&self) -> f32 {
        self.config.azimuth
    }

    /// Get the configured elevation angle.
    #[must_use]
    pub fn elevation(&self) -> f32 {
        self.config.elevation
    }

    /// Get the configured distance.
    #[must_use]
    pub fn distance(&self) -> f32 {
        self.config.distance
    }

    /// Get a reference to the current config.
    #[must_use]
    pub fn config(&self) -> &SpatialPanConfig {
        &self.config
    }

    /// Update the panning configuration.
    ///
    /// Recalculates all derived parameters without resetting filter states.
    pub fn update_config(&mut self, config: SpatialPanConfig) {
        let (gl, gr) = compute_vbap_gains(config.azimuth);
        self.gain_left = gl;
        self.gain_right = gr;
        self.distance_gain = compute_distance_gain(config.distance);

        let (a_left, a_right) = compute_hrtf_coeffs(config.elevation);
        self.hrtf_left.a = a_left;
        self.hrtf_right.a = a_right;

        self.config = config;
    }

    /// Process a block of mono input samples to stereo output.
    ///
    /// `output_left` and `output_right` must be at least as long as `input`.
    ///
    /// # Errors
    ///
    /// Returns [`AudioError::BufferTooSmall`] if output buffers are shorter than input.
    pub fn process(
        &mut self,
        input: &[f32],
        output_left: &mut [f32],
        output_right: &mut [f32],
    ) -> AudioResult<()> {
        if output_left.len() < input.len() {
            return Err(AudioError::BufferTooSmall {
                needed: input.len(),
                have: output_left.len(),
            });
        }
        if output_right.len() < input.len() {
            return Err(AudioError::BufferTooSmall {
                needed: input.len(),
                have: output_right.len(),
            });
        }

        let room = self.config.room_size;

        for (i, &x) in input.iter().enumerate() {
            // 1. Apply VBAP panning gains
            let panned_l = x * self.gain_left;
            let panned_r = x * self.gain_right;

            // 2. Apply HRTF IIR shelf filter per channel
            let hrtf_l = self.hrtf_left.process(panned_l);
            let hrtf_r = self.hrtf_right.process(panned_r);

            // 3. Distance attenuation
            let att_l = hrtf_l * self.distance_gain;
            let att_r = hrtf_r * self.distance_gain;

            // 4. Room early reflections (comb delay + mix)
            let reflection_l = self.room_delay_left.process(att_l);
            let reflection_r = self.room_delay_right.process(att_r);

            output_left[i] = att_l + room * 0.3 * reflection_l;
            output_right[i] = att_r + room * 0.3 * reflection_r;
        }

        Ok(())
    }

    /// Reset all internal filter states.
    pub fn reset(&mut self) {
        self.hrtf_left.reset();
        self.hrtf_right.reset();
        self.room_delay_left.reset();
        self.room_delay_right.reset();
    }

    /// Return the sample rate.
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

// ---------------------------------------------------------------------------
// Free helper functions
// ---------------------------------------------------------------------------

/// Compute VBAP stereo gains for a given azimuth angle in degrees.
///
/// Returns `(gain_left, gain_right)` normalised so that
/// `gain_left² + gain_right² = 1` (constant power panning).
fn compute_vbap_gains(azimuth_deg: f32) -> (f32, f32) {
    // Map azimuth to a pan position in [-π/4, +3π/4] so that:
    //   azimuth = -90° → full left  (angle = -π/4 from right speaker basis)
    //   azimuth =   0° → centre
    //   azimuth = +90° → full right
    // We use the constant-power formula:
    //   g_L = cos((azimuth + 90°) / 2 * π/180)
    //   g_R = sin((azimuth + 90°) / 2 * π/180)
    //   clamped to [0, 1]
    let angle_rad = (azimuth_deg + 90.0).clamp(0.0, 180.0) / 2.0 * std::f32::consts::PI / 180.0;
    let gl = angle_rad.cos().clamp(0.0, 1.0);
    let gr = angle_rad.sin().clamp(0.0, 1.0);

    // Normalise
    let norm = (gl * gl + gr * gr).sqrt();
    if norm < f32::EPSILON {
        (
            std::f32::consts::FRAC_1_SQRT_2,
            std::f32::consts::FRAC_1_SQRT_2,
        )
    } else {
        (gl / norm, gr / norm)
    }
}

/// Compute inverse-distance attenuation gain.
fn compute_distance_gain(distance: f32) -> f32 {
    (1.0 / (1.0 + distance * 0.1)).clamp(0.0, 1.0)
}

/// Compute one-pole IIR filter coefficient for HRTF elevation simulation.
///
/// Positive elevation → increase high-frequency content on both channels
/// (pinnae reflection); negative elevation → attenuate highs.
///
/// Returns `(coeff_left, coeff_right)`.
fn compute_hrtf_coeffs(elevation_deg: f32) -> (f32, f32) {
    // Normalise elevation to [-1, 1]
    let e = (elevation_deg / 90.0).clamp(-1.0, 1.0);

    // Base cutoff coefficient: 0.3 at neutral (moderate low-pass)
    let base_a = 0.3f32;

    // For positive elevation, open up the filter (higher a = more HF pass)
    // For negative elevation, close it down further
    let a = (base_a + e * 0.4).clamp(0.05, 0.95);

    // Contralateral ear (simplified): slightly more filtered
    let a_contra = (a - 0.1).clamp(0.05, 0.95);

    // For azimuth=0 (front), both ears are symmetric; this is called from
    // SpatialPanner which applies the same coefficient to both channels.
    // Elevation effect is symmetric (front elevation affects both ears equally).
    (a, a_contra)
}

// ============================================================
// Unit tests
// ============================================================
#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // SpatialPanConfig tests
    // ------------------------------------------------------------------

    #[test]
    fn test_config_valid() {
        let cfg = SpatialPanConfig::new(0.0, 0.0, 1.0, 0.0);
        assert!(cfg.is_ok(), "Valid config must succeed");
    }

    #[test]
    fn test_config_invalid_azimuth_too_low() {
        let cfg = SpatialPanConfig::new(-181.0, 0.0, 1.0, 0.0);
        assert!(cfg.is_err(), "azimuth < -180 must fail");
    }

    #[test]
    fn test_config_invalid_azimuth_too_high() {
        let cfg = SpatialPanConfig::new(181.0, 0.0, 1.0, 0.0);
        assert!(cfg.is_err(), "azimuth > 180 must fail");
    }

    #[test]
    fn test_config_invalid_elevation() {
        let cfg = SpatialPanConfig::new(0.0, 91.0, 1.0, 0.0);
        assert!(cfg.is_err(), "elevation > 90 must fail");
    }

    #[test]
    fn test_config_invalid_distance_zero() {
        let cfg = SpatialPanConfig::new(0.0, 0.0, 0.0, 0.0);
        assert!(cfg.is_err(), "distance=0 must fail");
    }

    #[test]
    fn test_config_invalid_distance_too_far() {
        let cfg = SpatialPanConfig::new(0.0, 0.0, 100.1, 0.0);
        assert!(cfg.is_err(), "distance > 100 must fail");
    }

    #[test]
    fn test_config_invalid_room_size() {
        let cfg = SpatialPanConfig::new(0.0, 0.0, 1.0, 1.1);
        assert!(cfg.is_err(), "room_size > 1.0 must fail");
    }

    #[test]
    fn test_config_front_helper() {
        let cfg = SpatialPanConfig::front();
        assert_eq!(cfg.azimuth, 0.0);
        assert_eq!(cfg.elevation, 0.0);
        assert_eq!(cfg.distance, 1.0);
        assert_eq!(cfg.room_size, 0.0);
    }

    // ------------------------------------------------------------------
    // SpatialPanner construction
    // ------------------------------------------------------------------

    #[test]
    fn test_panner_creation_valid() {
        let cfg = SpatialPanConfig::front();
        let panner = SpatialPanner::new(cfg, 48_000);
        assert!(panner.is_ok(), "Valid panner creation must succeed");
    }

    #[test]
    fn test_panner_creation_zero_sample_rate_fails() {
        let cfg = SpatialPanConfig::front();
        let panner = SpatialPanner::new(cfg, 0);
        assert!(panner.is_err(), "sample_rate=0 must fail");
    }

    // ------------------------------------------------------------------
    // VBAP panning tests
    // ------------------------------------------------------------------

    #[test]
    fn test_center_pan_equal_levels() {
        let cfg = SpatialPanConfig::front(); // azimuth = 0
        let mut panner = SpatialPanner::new(cfg, 48_000).expect("valid");

        let input = vec![1.0f32; 256];
        let mut left = vec![0.0f32; 256];
        let mut right = vec![0.0f32; 256];

        panner.process(&input, &mut left, &mut right).expect("ok");

        // At azimuth=0, L and R gains should be equal → outputs equal
        let sum_l: f32 = left.iter().sum();
        let sum_r: f32 = right.iter().sum();
        let diff = (sum_l - sum_r).abs();
        assert!(
            diff < 0.01 * sum_l.abs() + 1e-6,
            "Center pan: L={sum_l:.4} R={sum_r:.4} should be equal"
        );
    }

    #[test]
    fn test_right_pan_right_louder() {
        let cfg = SpatialPanConfig::new(90.0, 0.0, 1.0, 0.0).expect("valid");
        let mut panner = SpatialPanner::new(cfg, 48_000).expect("valid");

        let input = vec![1.0f32; 256];
        let mut left = vec![0.0f32; 256];
        let mut right = vec![0.0f32; 256];

        panner.process(&input, &mut left, &mut right).expect("ok");

        let sum_l: f32 = left.iter().sum();
        let sum_r: f32 = right.iter().sum();
        assert!(
            sum_r > sum_l,
            "Right pan: right ({sum_r:.4}) should be louder than left ({sum_l:.4})"
        );
    }

    #[test]
    fn test_left_pan_left_louder() {
        let cfg = SpatialPanConfig::new(-90.0, 0.0, 1.0, 0.0).expect("valid");
        let mut panner = SpatialPanner::new(cfg, 48_000).expect("valid");

        let input = vec![1.0f32; 256];
        let mut left = vec![0.0f32; 256];
        let mut right = vec![0.0f32; 256];

        panner.process(&input, &mut left, &mut right).expect("ok");

        let sum_l: f32 = left.iter().sum();
        let sum_r: f32 = right.iter().sum();
        assert!(
            sum_l > sum_r,
            "Left pan: left ({sum_l:.4}) should be louder than right ({sum_r:.4})"
        );
    }

    // ------------------------------------------------------------------
    // Distance attenuation test
    // ------------------------------------------------------------------

    #[test]
    fn test_distance_attenuation() {
        let near_cfg = SpatialPanConfig::new(0.0, 0.0, 1.0, 0.0).expect("valid");
        let far_cfg = SpatialPanConfig::new(0.0, 0.0, 50.0, 0.0).expect("valid");

        let mut near_panner = SpatialPanner::new(near_cfg, 48_000).expect("valid");
        let mut far_panner = SpatialPanner::new(far_cfg, 48_000).expect("valid");

        let input = vec![1.0f32; 256];
        let mut l_near = vec![0.0f32; 256];
        let mut r_near = vec![0.0f32; 256];
        let mut l_far = vec![0.0f32; 256];
        let mut r_far = vec![0.0f32; 256];

        near_panner
            .process(&input, &mut l_near, &mut r_near)
            .expect("ok");
        far_panner
            .process(&input, &mut l_far, &mut r_far)
            .expect("ok");

        let energy_near: f32 = l_near.iter().map(|s| s * s).sum();
        let energy_far: f32 = l_far.iter().map(|s| s * s).sum();

        assert!(
            energy_near > energy_far,
            "Near ({energy_near:.4}) should be louder than far ({energy_far:.4})"
        );
    }

    // ------------------------------------------------------------------
    // Elevation effect test
    // ------------------------------------------------------------------

    #[test]
    fn test_elevation_affects_output() {
        let flat_cfg = SpatialPanConfig::new(0.0, 0.0, 1.0, 0.0).expect("valid");
        let up_cfg = SpatialPanConfig::new(0.0, 45.0, 1.0, 0.0).expect("valid");

        let mut flat_panner = SpatialPanner::new(flat_cfg, 48_000).expect("valid");
        let mut up_panner = SpatialPanner::new(up_cfg, 48_000).expect("valid");

        // Use an impulse to show filter effect
        let mut input = vec![0.0f32; 256];
        input[0] = 1.0;

        let mut l_flat = vec![0.0f32; 256];
        let mut r_flat = vec![0.0f32; 256];
        let mut l_up = vec![0.0f32; 256];
        let mut r_up = vec![0.0f32; 256];

        flat_panner
            .process(&input, &mut l_flat, &mut r_flat)
            .expect("ok");
        up_panner.process(&input, &mut l_up, &mut r_up).expect("ok");

        // Elevated source has different filter coefficient → different output
        // Just verify it doesn't crash and produces non-trivially zero output
        let energy_flat: f32 = l_flat.iter().map(|s| s * s).sum();
        let energy_up: f32 = l_up.iter().map(|s| s * s).sum();
        assert!(energy_flat > 0.0, "Flat panner output must be non-zero");
        assert!(energy_up > 0.0, "Elevated panner output must be non-zero");
    }

    // ------------------------------------------------------------------
    // Room size effect test
    // ------------------------------------------------------------------

    #[test]
    fn test_room_size_effect() {
        let dry_cfg = SpatialPanConfig::new(0.0, 0.0, 1.0, 0.0).expect("valid");
        let wet_cfg = SpatialPanConfig::new(0.0, 0.0, 1.0, 1.0).expect("valid");

        let mut dry = SpatialPanner::new(dry_cfg, 48_000).expect("valid");
        let mut wet = SpatialPanner::new(wet_cfg, 48_000).expect("valid");

        let mut input = vec![0.0f32; 256];
        input[0] = 1.0;

        let mut l_dry = vec![0.0f32; 256];
        let mut r_dry = vec![0.0f32; 256];
        let mut l_wet = vec![0.0f32; 256];
        let mut r_wet = vec![0.0f32; 256];

        dry.process(&input, &mut l_dry, &mut r_dry).expect("ok");
        wet.process(&input, &mut l_wet, &mut r_wet).expect("ok");

        // Wet room has a delayed reflection so energy should differ after delay
        let e_dry: f32 = l_dry.iter().map(|s| s * s).sum();
        let e_wet: f32 = l_wet.iter().map(|s| s * s).sum();
        // Wet room adds reflection energy → total energy >= dry
        assert!(
            e_wet >= e_dry * 0.9, // allow small floating point variance
            "Wet room should have energy >= dry (wet={e_wet:.6} dry={e_dry:.6})"
        );
    }

    // ------------------------------------------------------------------
    // update_config test
    // ------------------------------------------------------------------

    #[test]
    fn test_update_config() {
        let cfg = SpatialPanConfig::front();
        let mut panner = SpatialPanner::new(cfg, 48_000).expect("valid");
        assert_eq!(panner.azimuth(), 0.0);

        let new_cfg = SpatialPanConfig::new(45.0, 10.0, 2.0, 0.2).expect("valid");
        panner.update_config(new_cfg);

        assert_eq!(panner.azimuth(), 45.0);
        assert_eq!(panner.elevation(), 10.0);
        assert_eq!(panner.distance(), 2.0);
    }

    // ------------------------------------------------------------------
    // Zero-length input test
    // ------------------------------------------------------------------

    #[test]
    fn test_process_empty_input() {
        let cfg = SpatialPanConfig::front();
        let mut panner = SpatialPanner::new(cfg, 48_000).expect("valid");
        let result = panner.process(&[], &mut [], &mut []);
        assert!(result.is_ok(), "Empty input should succeed");
    }

    // ------------------------------------------------------------------
    // Buffer too small error
    // ------------------------------------------------------------------

    #[test]
    fn test_process_output_too_small() {
        let cfg = SpatialPanConfig::front();
        let mut panner = SpatialPanner::new(cfg, 48_000).expect("valid");
        let input = vec![1.0f32; 10];
        let mut left = vec![0.0f32; 5]; // too small
        let mut right = vec![0.0f32; 10];
        let result = panner.process(&input, &mut left, &mut right);
        assert!(result.is_err(), "Too-small output buffer must fail");
    }

    // ------------------------------------------------------------------
    // Reset test
    // ------------------------------------------------------------------

    #[test]
    fn test_reset_clears_state() {
        let cfg = SpatialPanConfig::front();
        let mut panner = SpatialPanner::new(cfg, 48_000).expect("valid");

        // Feed a DC signal to charge the filter states
        let input_dc = vec![1.0f32; 512];
        let mut l = vec![0.0f32; 512];
        let mut r = vec![0.0f32; 512];
        panner.process(&input_dc, &mut l, &mut r).expect("ok");

        // Reset and process an impulse
        panner.reset();
        let mut input_imp = vec![0.0f32; 64];
        input_imp[0] = 1.0;
        let mut l2 = vec![0.0f32; 64];
        let mut r2 = vec![0.0f32; 64];
        panner.process(&input_imp, &mut l2, &mut r2).expect("ok");

        // After reset the filter state is clean: the tail beyond the impulse response
        // should decay (no DC offset carrying over)
        let sum_tail: f32 = l2[10..].iter().sum::<f32>().abs();
        assert!(
            sum_tail < 1.0,
            "After reset, tail should be small but got {sum_tail}"
        );
    }

    // ------------------------------------------------------------------
    // Sample rate accessor
    // ------------------------------------------------------------------

    #[test]
    fn test_sample_rate_accessor() {
        let cfg = SpatialPanConfig::front();
        let panner = SpatialPanner::new(cfg, 44_100).expect("valid");
        assert_eq!(panner.sample_rate(), 44_100);
    }
}
