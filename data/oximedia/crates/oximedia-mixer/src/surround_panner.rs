//! Surround panning (5.1 / 7.1) with VBAP-style gain computation and LFE crossover.
//!
//! The [`SurroundPanner`] computes per-speaker gain coefficients for a virtual
//! source placed at an arbitrary azimuth and elevation.  The algorithm is based
//! on Vector Base Amplitude Panning (VBAP) adapted for fixed ITU-R BS.775
//! speaker layouts.
//!
//! ## LFE Crossover
//!
//! An optional 2nd-order Butterworth low-pass filter routes bass content to the
//! LFE channel while removing it from the main speakers.  The crossover
//! frequency defaults to 80 Hz (THX standard).

use std::f32::consts::{FRAC_PI_2, PI};

// ---------------------------------------------------------------------------
// Task-spec API types
// ---------------------------------------------------------------------------

/// High-level surround format selector for the task-spec `SurroundPanner` API.
///
/// Unlike [`SurroundLayout`] (which drives the internal VBAP engine), this enum
/// is intended as the public-facing channel-format descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurroundFormat {
    /// Two-channel stereo: L, R.
    Stereo,
    /// 5.1 surround: L, R, C, LFE, Ls, Rs.
    Surround51,
    /// 7.1 surround: L, R, C, LFE, Ls, Rs, Lss, Rss.
    Surround71,
}

impl SurroundFormat {
    /// Number of output channels for this format.
    #[must_use]
    pub fn channel_count(self) -> usize {
        match self {
            Self::Stereo => 2,
            Self::Surround51 => 6,
            Self::Surround71 => 8,
        }
    }
}

/// Normalised virtual-source position for VBAP-style surround panning.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurroundPanPosition {
    /// Horizontal angle: −1.0 = hard left, 0.0 = centre, +1.0 = hard right.
    pub azimuth: f32,
    /// Vertical angle: −1.0 = below, 0.0 = ear level, +1.0 = above.
    pub elevation: f32,
    /// Source distance: 0.0 = origin (centre), 1.0 = far field.
    /// A higher distance attenuates the overall level slightly.
    pub distance: f32,
    /// Spatial divergence: 0.0 = point source (maximum localisation),
    /// 1.0 = omnidirectional (energy spread equally across all speakers).
    pub divergence: f32,
}

impl SurroundPanPosition {
    /// Construct a position in the horizontal plane at origin with no divergence.
    #[must_use]
    pub fn new(azimuth: f32) -> Self {
        Self {
            azimuth: azimuth.clamp(-1.0, 1.0),
            elevation: 0.0,
            distance: 0.0,
            divergence: 0.0,
        }
    }

    /// Front-centre shorthand.
    #[must_use]
    pub fn center() -> Self {
        Self::new(0.0)
    }
}

impl Default for SurroundPanPosition {
    fn default() -> Self {
        Self::center()
    }
}

// ---------------------------------------------------------------------------
// Speaker layout definitions
// ---------------------------------------------------------------------------

/// Speaker indices for a 5.1 layout (ITU-R BS.775).
///
/// `[L, R, C, LFE, Ls, Rs]`
pub const SURROUND_51_SPEAKERS: usize = 6;

/// Speaker indices for a 7.1 layout.
///
/// `[L, R, C, LFE, Ls, Rs, Lb, Rb]`
pub const SURROUND_71_SPEAKERS: usize = 8;

/// Fixed azimuth angles (radians) for ITU-R BS.775 5.1 speakers.
/// 0 = front centre, positive = clockwise.
///
/// L = -30, R = +30, C = 0, Ls = -110, Rs = +110.
/// (LFE has no spatial position; index 3 is handled separately.)
const SPEAKERS_51_AZ: [f32; 5] = [
    -30.0 * PI / 180.0,
    30.0 * PI / 180.0,
    0.0,
    -110.0 * PI / 180.0,
    110.0 * PI / 180.0,
];

/// Additional back speakers for 7.1 (Lb = -150, Rb = +150).
const SPEAKERS_71_BACK_AZ: [f32; 2] = [-150.0 * PI / 180.0, 150.0 * PI / 180.0];

// ---------------------------------------------------------------------------
// Surround layout enum
// ---------------------------------------------------------------------------

/// Target surround layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurroundLayout {
    /// 5.1 layout: L R C LFE Ls Rs
    Layout51,
    /// 7.1 layout: L R C LFE Ls Rs Lb Rb
    Layout71,
}

impl SurroundLayout {
    /// Number of output channels.
    #[must_use]
    pub fn channel_count(self) -> usize {
        match self {
            Self::Layout51 => SURROUND_51_SPEAKERS,
            Self::Layout71 => SURROUND_71_SPEAKERS,
        }
    }
}

// ---------------------------------------------------------------------------
// Biquad LPF for LFE crossover
// ---------------------------------------------------------------------------

/// 2nd-order Butterworth low-pass filter (biquad) for LFE crossover.
#[derive(Debug, Clone)]
struct BiquadLpf {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl BiquadLpf {
    /// Create a 2nd-order Butterworth LPF.
    fn new(cutoff_hz: f32, sample_rate: f32) -> Self {
        let w0 = 2.0 * PI * cutoff_hz / sample_rate;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        // Q = 1/sqrt(2) for Butterworth
        let alpha = sin_w0 / (2.0 * std::f32::consts::SQRT_2);

        let b0 = (1.0 - cos_w0) / 2.0;
        let b1 = 1.0 - cos_w0;
        let b2 = (1.0 - cos_w0) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    /// Process a single sample through the biquad (transposed direct form II).
    fn tick(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }

    /// Reset internal state.
    fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }
}

// ---------------------------------------------------------------------------
// SurroundPanner
// ---------------------------------------------------------------------------

/// Surround panner supporting 5.1 and 7.1 speaker layouts with VBAP-style
/// gain computation and optional LFE crossover.
#[derive(Debug, Clone)]
pub struct SurroundPanner {
    /// Target layout.
    layout: SurroundLayout,
    /// Source azimuth in radians (-PI..PI, 0 = front centre, positive = right).
    azimuth: f32,
    /// Source elevation in radians (-PI/2..PI/2, 0 = ear level, positive = up).
    elevation: f32,
    /// LFE crossover enabled.
    lfe_crossover_enabled: bool,
    /// LFE crossover filter (Butterworth LPF).
    lfe_lpf: BiquadLpf,
    /// LFE crossover frequency in Hz.
    lfe_crossover_freq: f32,
    /// Sample rate.
    sample_rate: f32,
    /// LFE send level (0.0..1.0).
    lfe_level: f32,
    /// Optional high-level format override (set via [`Self::from_format`]).
    /// Used by [`Self::pan`] to decide output channel count for Stereo fold-down.
    format_override: Option<SurroundFormat>,
}

impl SurroundPanner {
    /// Create a new surround panner.
    #[must_use]
    pub fn new(layout: SurroundLayout, sample_rate: f32) -> Self {
        let crossover_freq = 80.0;
        Self {
            layout,
            azimuth: 0.0,
            elevation: 0.0,
            lfe_crossover_enabled: true,
            lfe_lpf: BiquadLpf::new(crossover_freq, sample_rate),
            lfe_crossover_freq: crossover_freq,
            sample_rate,
            lfe_level: 0.707, // -3 dB default LFE level
            format_override: None,
        }
    }

    /// Set the source azimuth (radians).
    pub fn set_azimuth(&mut self, azimuth: f32) {
        self.azimuth = azimuth.clamp(-PI, PI);
    }

    /// Get the current azimuth (radians).
    #[must_use]
    pub fn azimuth(&self) -> f32 {
        self.azimuth
    }

    /// Set the source elevation (radians).
    pub fn set_elevation(&mut self, elevation: f32) {
        self.elevation = elevation.clamp(-FRAC_PI_2, FRAC_PI_2);
    }

    /// Get the current elevation (radians).
    #[must_use]
    pub fn elevation(&self) -> f32 {
        self.elevation
    }

    /// Set the LFE crossover frequency (Hz).
    pub fn set_lfe_crossover_freq(&mut self, freq: f32) {
        self.lfe_crossover_freq = freq.clamp(20.0, 200.0);
        self.lfe_lpf = BiquadLpf::new(self.lfe_crossover_freq, self.sample_rate);
    }

    /// Get the LFE crossover frequency (Hz).
    #[must_use]
    pub fn lfe_crossover_freq(&self) -> f32 {
        self.lfe_crossover_freq
    }

    /// Enable or disable LFE crossover.
    pub fn set_lfe_crossover_enabled(&mut self, enabled: bool) {
        self.lfe_crossover_enabled = enabled;
        if !enabled {
            self.lfe_lpf.reset();
        }
    }

    /// Set the LFE send level (0.0..1.0).
    pub fn set_lfe_level(&mut self, level: f32) {
        self.lfe_level = level.clamp(0.0, 1.0);
    }

    /// Get the layout.
    #[must_use]
    pub fn layout(&self) -> SurroundLayout {
        self.layout
    }

    /// Compute VBAP-style gains for the main speakers (excluding LFE).
    ///
    /// Returns a vector of gains indexed by speaker (LFE slot is 0.0).
    #[must_use]
    pub fn compute_gains(&self) -> Vec<f32> {
        let n_out = self.layout.channel_count();
        let mut gains = vec![0.0_f32; n_out];

        // Collect speaker azimuths (excluding LFE)
        let speaker_azimuths: Vec<f32> = match self.layout {
            SurroundLayout::Layout51 => SPEAKERS_51_AZ.to_vec(),
            SurroundLayout::Layout71 => {
                let mut az = SPEAKERS_51_AZ.to_vec();
                az.extend_from_slice(&SPEAKERS_71_BACK_AZ);
                az
            }
        };

        // Map speaker index in azimuths array to output channel index.
        // 5.1: azimuths[0..5] -> gains indices [0,1,2,4,5] (skip 3 = LFE)
        // 7.1: azimuths[0..7] -> gains indices [0,1,2,4,5,6,7]
        let spk_to_out: Vec<usize> = match self.layout {
            SurroundLayout::Layout51 => vec![0, 1, 2, 4, 5],
            SurroundLayout::Layout71 => vec![0, 1, 2, 4, 5, 6, 7],
        };

        // VBAP-style: compute angular weight for each speaker.
        let mut weights = Vec::with_capacity(speaker_azimuths.len());
        let mut total_weight = 0.0_f32;

        for &spk_az in &speaker_azimuths {
            let angle_diff = angular_distance(self.azimuth, spk_az);
            // Cosine weighting: speakers within ~90 degrees contribute.
            let w = (1.0 - angle_diff / PI).max(0.0);
            // Square for sharper localisation.
            let w = w * w;
            weights.push(w);
            total_weight += w;
        }

        // Normalize and assign to output channels.
        if total_weight > 1e-12 {
            for (i, &w) in weights.iter().enumerate() {
                if let Some(&out_idx) = spk_to_out.get(i) {
                    gains[out_idx] = w / total_weight;
                }
            }
        }

        // Apply elevation-based attenuation to surround speakers.
        // High elevation sources should come more from the front.
        let elev_factor = (1.0 - self.elevation.abs() / FRAC_PI_2).max(0.0);
        match self.layout {
            SurroundLayout::Layout51 => {
                gains[4] *= elev_factor; // Ls
                gains[5] *= elev_factor; // Rs
            }
            SurroundLayout::Layout71 => {
                gains[4] *= elev_factor; // Ls
                gains[5] *= elev_factor; // Rs
                gains[6] *= elev_factor; // Lb
                gains[7] *= elev_factor; // Rb
            }
        }

        // LFE gets a fixed level based on elevation (more LFE for low sounds).
        gains[3] = (1.0 - self.elevation.abs() / FRAC_PI_2).max(0.0) * self.lfe_level;

        gains
    }

    /// Pan a mono input buffer to surround output channels.
    ///
    /// Returns a vector of per-speaker output buffers.
    #[must_use]
    pub fn pan_buffer(&mut self, input: &[f32]) -> Vec<Vec<f32>> {
        let n_out = self.layout.channel_count();
        let n_samples = input.len();
        let gains = self.compute_gains();

        let mut outputs = vec![vec![0.0_f32; n_samples]; n_out];

        for (i, &sample) in input.iter().enumerate() {
            // LFE crossover: extract bass content for LFE channel
            let lfe_sample = if self.lfe_crossover_enabled {
                self.lfe_lpf.tick(sample)
            } else {
                0.0
            };

            for (ch, gain) in gains.iter().enumerate() {
                if ch == 3 {
                    // LFE channel: use crossover-filtered bass + gain
                    outputs[3][i] = if self.lfe_crossover_enabled {
                        lfe_sample * *gain
                    } else {
                        sample * *gain
                    };
                } else {
                    // Main speakers get the full signal * their gain
                    outputs[ch][i] = sample * *gain;
                }
            }
        }

        outputs
    }

    /// Reset the LFE crossover filter state.
    pub fn reset(&mut self) {
        self.lfe_lpf.reset();
    }

    // -----------------------------------------------------------------------
    // Task-spec API (SurroundFormat / SurroundPanPosition)
    // -----------------------------------------------------------------------

    /// Create a `SurroundPanner` from a high-level [`SurroundFormat`] descriptor.
    ///
    /// Uses a default sample rate of 48 000 Hz and the matching ITU layout.
    #[must_use]
    pub fn from_format(format: SurroundFormat) -> Self {
        let layout = match format {
            SurroundFormat::Stereo | SurroundFormat::Surround51 => SurroundLayout::Layout51,
            SurroundFormat::Surround71 => SurroundLayout::Layout71,
        };
        // For Stereo we use the 5.1 engine but will only expose 2 output channels
        // through the `pan()` method.
        let mut panner = Self::new(layout, 48_000.0);
        // Store format for channel-count decisions.
        panner.format_override = Some(format);
        panner
    }

    /// Pan a mono source (scaled by `mono_gain`) to per-speaker gains.
    ///
    /// `position` uses normalised coordinates (see [`SurroundPanPosition`]).
    ///
    /// Returns a `Vec<f32>` whose length equals the channel count of the format
    /// supplied to [`Self::from_format`], or [`SurroundLayout::channel_count`]
    /// for panners created with [`Self::new`].
    #[must_use]
    pub fn pan(&mut self, mono_gain: f32, position: &SurroundPanPosition) -> Vec<f32> {
        // Map normalised azimuth [-1, 1] → radians [-PI, PI]
        let az_rad = position.azimuth.clamp(-1.0, 1.0) * PI;
        // Map normalised elevation [-1, 1] → [-PI/2, PI/2]
        let el_rad = position.elevation.clamp(-1.0, 1.0) * FRAC_PI_2;

        self.set_azimuth(az_rad);
        self.set_elevation(el_rad);

        let gains = self.compute_gains();
        let n_out = gains.len();

        // Distance attenuation: 0.0 distance = full gain, 1.0 = −6 dB
        let dist_atten = 1.0 - position.distance.clamp(0.0, 1.0) * 0.5;

        // Divergence: lerp between localised gains and uniform energy spread.
        let div = position.divergence.clamp(0.0, 1.0);
        let uniform = if n_out > 0 {
            let sum: f32 = gains.iter().sum();
            let u = sum / n_out as f32;
            u
        } else {
            0.0
        };

        let panned: Vec<f32> = gains
            .iter()
            .map(|&g| {
                let localised = g * mono_gain * dist_atten;
                let spread = uniform * mono_gain * dist_atten;
                localised * (1.0 - div) + spread * div
            })
            .collect();

        // If created from SurroundFormat::Stereo, collapse to 2 channels: L + C*0.707, R + C*0.707
        match self.format_override {
            Some(SurroundFormat::Stereo) => {
                // panned layout is 5.1: [L, R, C, LFE, Ls, Rs]
                let l = panned.first().copied().unwrap_or(0.0)
                    + panned.get(2).copied().unwrap_or(0.0) * 0.707;
                let r = panned.get(1).copied().unwrap_or(0.0)
                    + panned.get(2).copied().unwrap_or(0.0) * 0.707;
                vec![l, r]
            }
            _ => panned,
        }
    }

    /// Returns (azimuth_degrees, elevation_degrees) for each speaker in the layout.
    ///
    /// For [`SurroundFormat::Stereo`] (or the equivalent 5.1 layout): returns 6 speaker positions.
    /// For 7.1: returns 8 positions.  LFE is placed at (0°, 0°).
    #[must_use]
    pub fn speaker_layout_degrees(&self) -> Vec<(f32, f32)> {
        match self.layout {
            SurroundLayout::Layout51 => vec![
                (-30.0, 0.0),  // L
                (30.0, 0.0),   // R
                (0.0, 0.0),    // C
                (0.0, 0.0),    // LFE
                (-110.0, 0.0), // Ls
                (110.0, 0.0),  // Rs
            ],
            SurroundLayout::Layout71 => vec![
                (-30.0, 0.0),  // L
                (30.0, 0.0),   // R
                (0.0, 0.0),    // C
                (0.0, 0.0),    // LFE
                (-110.0, 0.0), // Ls
                (110.0, 0.0),  // Rs
                (-60.0, 0.0),  // Lss (wide side surround)
                (60.0, 0.0),   // Rss
            ],
        }
    }
}

/// Compute the shortest angular distance between two angles in radians.
fn angular_distance(a: f32, b: f32) -> f32 {
    let diff = (a - b).abs();
    if diff > PI {
        2.0 * PI - diff
    } else {
        diff
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layout_channel_count() {
        assert_eq!(SurroundLayout::Layout51.channel_count(), 6);
        assert_eq!(SurroundLayout::Layout71.channel_count(), 8);
    }

    #[test]
    fn test_surround_panner_center_51() {
        let panner = SurroundPanner::new(SurroundLayout::Layout51, 48000.0);
        let gains = panner.compute_gains();
        assert_eq!(gains.len(), 6);
        // Centre speaker should have the highest gain for front-centre source.
        let max_idx = gains
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != 3) // skip LFE
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        assert_eq!(
            max_idx, 2,
            "centre speaker should be loudest for front source"
        );
    }

    #[test]
    fn test_surround_panner_hard_left() {
        let mut panner = SurroundPanner::new(SurroundLayout::Layout51, 48000.0);
        panner.set_azimuth(-30.0_f32.to_radians()); // Exact L position
        let gains = panner.compute_gains();
        // Left speaker should be dominant.
        assert!(
            gains[0] > gains[1],
            "L should be louder than R for hard-left source"
        );
    }

    #[test]
    fn test_surround_panner_71() {
        let panner = SurroundPanner::new(SurroundLayout::Layout71, 48000.0);
        let gains = panner.compute_gains();
        assert_eq!(gains.len(), 8);
        // All main gains should be non-negative.
        for (i, &g) in gains.iter().enumerate() {
            assert!(g >= 0.0, "gain[{i}] should be >= 0, got {g}");
        }
    }

    #[test]
    fn test_surround_panner_lfe_nonzero() {
        let panner = SurroundPanner::new(SurroundLayout::Layout51, 48000.0);
        let gains = panner.compute_gains();
        // At zero elevation, LFE should get some signal.
        assert!(gains[3] > 0.0, "LFE should have gain at zero elevation");
    }

    #[test]
    fn test_surround_panner_lfe_zero_at_top() {
        let mut panner = SurroundPanner::new(SurroundLayout::Layout51, 48000.0);
        panner.set_elevation(FRAC_PI_2); // Straight up
        let gains = panner.compute_gains();
        assert!(gains[3].abs() < 1e-6, "LFE should be zero at max elevation");
    }

    #[test]
    fn test_pan_buffer_51() {
        let mut panner = SurroundPanner::new(SurroundLayout::Layout51, 48000.0);
        let input = vec![1.0_f32; 64];
        let output = panner.pan_buffer(&input);
        assert_eq!(output.len(), 6);
        for ch in &output {
            assert_eq!(ch.len(), 64);
        }
    }

    #[test]
    fn test_pan_buffer_71() {
        let mut panner = SurroundPanner::new(SurroundLayout::Layout71, 48000.0);
        let input = vec![0.5_f32; 128];
        let output = panner.pan_buffer(&input);
        assert_eq!(output.len(), 8);
    }

    #[test]
    fn test_lfe_crossover_filter_produces_output() {
        let mut panner = SurroundPanner::new(SurroundLayout::Layout51, 48000.0);
        panner.set_lfe_crossover_enabled(true);
        // Generate a low-frequency signal (50 Hz sine at 48 kHz)
        let input: Vec<f32> = (0..512)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let t = i as f32 / 48000.0;
                (2.0 * PI * 50.0 * t).sin()
            })
            .collect();
        let output = panner.pan_buffer(&input);
        // LFE should have meaningful signal for a 50 Hz tone
        let lfe_energy: f32 = output[3].iter().map(|s| s * s).sum();
        assert!(lfe_energy > 0.0, "LFE should have signal for 50 Hz tone");
    }

    #[test]
    fn test_angular_distance_same() {
        assert!(angular_distance(0.0, 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_angular_distance_opposite() {
        let d = angular_distance(0.0, PI);
        assert!((d - PI).abs() < 1e-5);
    }

    #[test]
    fn test_angular_distance_wrap_around() {
        // -170 deg and +170 deg should be 20 degrees apart, not 340
        let a = -170.0_f32.to_radians();
        let b = 170.0_f32.to_radians();
        let d = angular_distance(a, b);
        let expected = 20.0_f32.to_radians();
        assert!(
            (d - expected).abs() < 0.01,
            "wrap-around distance should be ~20 deg, got {}",
            d.to_degrees()
        );
    }

    #[test]
    fn test_set_lfe_crossover_freq() {
        let mut panner = SurroundPanner::new(SurroundLayout::Layout51, 48000.0);
        panner.set_lfe_crossover_freq(120.0);
        assert!((panner.lfe_crossover_freq() - 120.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_set_lfe_crossover_freq_clamped() {
        let mut panner = SurroundPanner::new(SurroundLayout::Layout51, 48000.0);
        panner.set_lfe_crossover_freq(500.0);
        assert!(panner.lfe_crossover_freq() <= 200.0);
    }

    #[test]
    fn test_gains_sum_reasonable() {
        // For any position, the sum of main speaker gains (excl LFE) should be ~1.0
        let mut panner = SurroundPanner::new(SurroundLayout::Layout51, 48000.0);
        for az_deg in [-180, -90, -30, 0, 30, 90, 180] {
            #[allow(clippy::cast_precision_loss)]
            let az = (az_deg as f32).to_radians();
            panner.set_azimuth(az);
            let gains = panner.compute_gains();
            let main_sum: f32 = gains
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != 3)
                .map(|(_, g)| g)
                .sum();
            assert!(
                main_sum > 0.0 && main_sum < 2.0,
                "main gain sum should be reasonable at az={az_deg}, got {main_sum}"
            );
        }
    }

    #[test]
    fn test_rear_source_71_uses_back_speakers() {
        let mut panner = SurroundPanner::new(SurroundLayout::Layout71, 48000.0);
        panner.set_azimuth(150.0_f32.to_radians()); // Near Rb position
        let gains = panner.compute_gains();
        // Back right (index 7) should have significant gain
        assert!(
            gains[7] > 0.01,
            "Rb should have gain for rear-right source, got {}",
            gains[7]
        );
    }

    #[test]
    fn test_panner_reset_clears_filter() {
        let mut panner = SurroundPanner::new(SurroundLayout::Layout51, 48000.0);
        // Process some samples to populate filter state
        let input = vec![1.0_f32; 64];
        let _ = panner.pan_buffer(&input);
        panner.reset();
        // After reset, processing silence should produce silence
        let silence = vec![0.0_f32; 64];
        let output = panner.pan_buffer(&silence);
        for ch in &output {
            for &s in ch {
                assert!(s.abs() < 0.01, "should be near-silence after reset");
            }
        }
    }

    // -----------------------------------------------------------------------
    // Task-spec API tests (SurroundFormat, SurroundPanPosition, pan(), speaker_layout_degrees())
    // -----------------------------------------------------------------------

    #[test]
    fn test_surround_format_channel_count() {
        assert_eq!(SurroundFormat::Stereo.channel_count(), 2);
        assert_eq!(SurroundFormat::Surround51.channel_count(), 6);
        assert_eq!(SurroundFormat::Surround71.channel_count(), 8);
    }

    #[test]
    fn test_surround_pan_position_default() {
        let pos = SurroundPanPosition::default();
        assert!((pos.azimuth).abs() < f32::EPSILON);
        assert!((pos.elevation).abs() < f32::EPSILON);
        assert!((pos.distance).abs() < f32::EPSILON);
        assert!((pos.divergence).abs() < f32::EPSILON);
    }

    #[test]
    fn test_surround_pan_position_new_clamps() {
        let pos = SurroundPanPosition::new(2.0);
        assert!(
            (pos.azimuth - 1.0).abs() < f32::EPSILON,
            "should clamp to 1.0"
        );
    }

    #[test]
    fn test_from_format_51_pan_returns_6_channels() {
        let mut panner = SurroundPanner::from_format(SurroundFormat::Surround51);
        let pos = SurroundPanPosition::center();
        let gains = panner.pan(1.0, &pos);
        assert_eq!(gains.len(), 6, "5.1 pan should return 6 channel gains");
    }

    #[test]
    fn test_from_format_71_pan_returns_8_channels() {
        let mut panner = SurroundPanner::from_format(SurroundFormat::Surround71);
        let pos = SurroundPanPosition::center();
        let gains = panner.pan(1.0, &pos);
        assert_eq!(gains.len(), 8, "7.1 pan should return 8 channel gains");
    }

    #[test]
    fn test_from_format_stereo_pan_returns_2_channels() {
        let mut panner = SurroundPanner::from_format(SurroundFormat::Stereo);
        let pos = SurroundPanPosition::center();
        let gains = panner.pan(1.0, &pos);
        assert_eq!(gains.len(), 2, "Stereo pan should return 2 channel gains");
    }

    #[test]
    fn test_pan_gains_non_negative() {
        let mut panner = SurroundPanner::from_format(SurroundFormat::Surround51);
        let pos = SurroundPanPosition {
            azimuth: 0.5,
            elevation: 0.1,
            distance: 0.3,
            divergence: 0.0,
        };
        let gains = panner.pan(1.0, &pos);
        for (i, &g) in gains.iter().enumerate() {
            assert!(g >= 0.0, "gain[{i}] should be >= 0, got {g}");
        }
    }

    #[test]
    fn test_pan_mono_gain_scales_output() {
        let mut panner = SurroundPanner::from_format(SurroundFormat::Surround51);
        let pos = SurroundPanPosition::center();
        let gains_full = panner.pan(1.0, &pos);
        let gains_half = panner.pan(0.5, &pos);
        for (g_full, g_half) in gains_full.iter().zip(gains_half.iter()) {
            assert!(
                (g_full * 0.5 - g_half).abs() < 1e-5,
                "half gain should be half of full gain"
            );
        }
    }

    #[test]
    fn test_pan_distance_attenuates() {
        let mut panner = SurroundPanner::from_format(SurroundFormat::Surround51);
        let pos_near = SurroundPanPosition {
            azimuth: 0.0,
            elevation: 0.0,
            distance: 0.0,
            divergence: 0.0,
        };
        let pos_far = SurroundPanPosition {
            azimuth: 0.0,
            elevation: 0.0,
            distance: 1.0,
            divergence: 0.0,
        };
        let gains_near: f32 = panner.pan(1.0, &pos_near).iter().sum();
        let gains_far: f32 = panner.pan(1.0, &pos_far).iter().sum();
        assert!(
            gains_near > gains_far,
            "near source should be louder than far source"
        );
    }

    #[test]
    fn test_pan_divergence_spreads_energy() {
        let mut panner = SurroundPanner::from_format(SurroundFormat::Surround51);
        // Hard left — divergence=0 should concentrate in L/Ls
        let pos_focused = SurroundPanPosition {
            azimuth: -1.0,
            elevation: 0.0,
            distance: 0.0,
            divergence: 0.0,
        };
        let pos_spread = SurroundPanPosition {
            azimuth: -1.0,
            elevation: 0.0,
            distance: 0.0,
            divergence: 1.0,
        };
        let focused = panner.pan(1.0, &pos_focused);
        let spread = panner.pan(1.0, &pos_spread);
        // With full divergence, all channels should have more equal levels
        let focused_std = std_dev(&focused);
        let spread_std = std_dev(&spread);
        assert!(
            spread_std <= focused_std + 0.01,
            "divergence should reduce channel spread: focused_std={focused_std:.4}, spread_std={spread_std:.4}"
        );
    }

    #[test]
    fn test_speaker_layout_degrees_51() {
        let panner = SurroundPanner::from_format(SurroundFormat::Surround51);
        let layout = panner.speaker_layout_degrees();
        assert_eq!(layout.len(), 6);
        // L at -30, R at +30
        assert!((layout[0].0 - (-30.0)).abs() < f32::EPSILON);
        assert!((layout[1].0 - 30.0).abs() < f32::EPSILON);
        // C at 0
        assert!((layout[2].0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_speaker_layout_degrees_71() {
        let panner = SurroundPanner::from_format(SurroundFormat::Surround71);
        let layout = panner.speaker_layout_degrees();
        assert_eq!(layout.len(), 8);
        // Wide surround Lss at -60, Rss at +60
        assert!(
            (layout[6].0 - (-60.0)).abs() < f32::EPSILON,
            "Lss should be at -60°, got {}",
            layout[6].0
        );
        assert!(
            (layout[7].0 - 60.0).abs() < f32::EPSILON,
            "Rss should be at +60°, got {}",
            layout[7].0
        );
    }

    #[test]
    fn test_stereo_pan_center_is_equal_l_r() {
        let mut panner = SurroundPanner::from_format(SurroundFormat::Stereo);
        let pos = SurroundPanPosition::center();
        let gains = panner.pan(1.0, &pos);
        assert_eq!(gains.len(), 2);
        assert!(
            (gains[0] - gains[1]).abs() < 0.01,
            "centre pan stereo should be equal L/R: L={}, R={}",
            gains[0],
            gains[1]
        );
    }

    /// Compute standard deviation of a slice.
    fn std_dev(values: &[f32]) -> f32 {
        if values.is_empty() {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        #[allow(clippy::cast_precision_loss)]
        let variance =
            values.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / values.len() as f32;
        variance.sqrt()
    }
}
