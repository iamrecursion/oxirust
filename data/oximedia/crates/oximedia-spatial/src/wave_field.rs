//! Wave Field Synthesis (WFS) — large-scale spatial audio for loudspeaker arrays.
//!
//! WFS is a physical approach to spatial audio that attempts to recreate a correct
//! wave field across an extended listening area (rather than a single "sweet spot").
//! It works by driving a dense array of loudspeakers with individually delayed and
//! gain-weighted copies of the source signal so that, by Huygens' principle, the
//! superposition of all secondary wavelets reconstructs the desired wave front.
//!
//! This implementation provides:
//! - Speaker array definitions (linear, circular, planar)
//! - Virtual point-source and plane-wave objects
//! - 2.5D driving function computation (simplified, frequency-independent amplitude)
//! - Per-speaker delay-and-gain computation for real-time block processing
//!
//! # Reference
//! Berkhout, A. J. (1988). "A holographic approach to acoustic control."
//! JAES 36(12).

use std::f32::consts::PI;

// ─── Complex number ───────────────────────────────────────────────────────────

/// Minimal complex number type (f32).
///
/// No external crate is used; only the operations needed for the WFS driving
/// function are implemented here.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    pub re: f32,
    pub im: f32,
}

impl Complex {
    /// Create a new complex number.
    pub fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    /// Complex number from polar form: `r * e^(jθ)`.
    pub fn from_polar(r: f32, theta: f32) -> Self {
        Self {
            re: r * theta.cos(),
            im: r * theta.sin(),
        }
    }

    /// Complex conjugate.
    pub fn conj(&self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// Absolute value (modulus).
    pub fn abs(&self) -> f32 {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    /// Multiply two complex numbers.
    pub fn mul(&self, rhs: &Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }

    /// Add two complex numbers.
    pub fn add(&self, rhs: &Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }

    /// Scale by a real value.
    pub fn scale(&self, s: f32) -> Self {
        Self {
            re: self.re * s,
            im: self.im * s,
        }
    }

    /// Square root (principal branch).
    pub fn sqrt(&self) -> Self {
        let r = self.abs().sqrt();
        if r < 1e-15 {
            return Self::new(0.0, 0.0);
        }
        let theta = self.im.atan2(self.re) * 0.5;
        Self::from_polar(r, theta)
    }
}

// ─── Types ────────────────────────────────────────────────────────────────────

/// Geometry classification of the loudspeaker array.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrayType {
    /// All speakers along a straight line.
    Linear,
    /// Speakers arranged on a circle.
    Circular,
    /// Speakers on a flat rectangular surface.
    Planar,
}

/// A single loudspeaker in the WFS array.
#[derive(Debug, Clone)]
pub struct WfsSpeaker {
    /// X position in metres.
    pub x: f32,
    /// Y position in metres.
    pub y: f32,
    /// Z position in metres.
    pub z: f32,
    /// Output channel index.
    pub channel_index: usize,
    /// Normal vector X component (speaker facing direction).
    pub normal_x: f32,
    /// Normal vector Y component.
    pub normal_y: f32,
}

/// Loudspeaker array for WFS.
#[derive(Debug, Clone)]
pub struct WfsArray {
    /// All speakers in the array.
    pub speakers: Vec<WfsSpeaker>,
    /// Geometric type of the array.
    pub array_type: ArrayType,
}

/// A virtual audio source to be synthesised by the WFS array.
#[derive(Debug, Clone)]
pub struct VirtualSource {
    /// X position of the virtual source in metres (for point sources).
    pub x: f32,
    /// Y position of the virtual source in metres.
    pub y: f32,
    /// If `true`, the source is a plane wave (infinite distance).
    pub is_plane_wave: bool,
    /// Propagation direction of the plane wave (azimuth in degrees, only used when
    /// `is_plane_wave = true`).
    pub plane_wave_azimuth_deg: f32,
}

/// WFS renderer — computes per-speaker driving functions.
#[derive(Debug, Clone)]
pub struct WfsRenderer {
    /// The loudspeaker array.
    pub array: WfsArray,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Reference distance for amplitude normalisation (metres).  Default 1.0 m.
    pub reference_distance: f32,
}

// ─── Speed of sound ──────────────────────────────────────────────────────────

/// Speed of sound in air at 20 °C (m/s).
const SPEED_OF_SOUND: f32 = 343.0;

// ─── WfsArray factory methods ─────────────────────────────────────────────────

impl WfsArray {
    /// Build a linear array of `n` speakers spaced `spacing_m` metres apart,
    /// centred at the origin, facing in the +Y direction (normal = [0, 1]).
    pub fn linear(n: usize, spacing_m: f32) -> Self {
        let total_len = (n as f32 - 1.0) * spacing_m;
        let speakers = (0..n)
            .map(|i| WfsSpeaker {
                x: i as f32 * spacing_m - total_len * 0.5,
                y: 0.0,
                z: 0.0,
                channel_index: i,
                normal_x: 0.0,
                normal_y: 1.0,
            })
            .collect();
        Self {
            speakers,
            array_type: ArrayType::Linear,
        }
    }

    /// Build a circular array of `n` speakers on a circle of radius `radius_m`,
    /// each facing inward (toward the centre).
    pub fn circular(n: usize, radius_m: f32) -> Self {
        let speakers = (0..n)
            .map(|i| {
                let angle = 2.0 * PI * i as f32 / n as f32;
                let x = radius_m * angle.cos();
                let y = radius_m * angle.sin();
                WfsSpeaker {
                    x,
                    y,
                    z: 0.0,
                    channel_index: i,
                    normal_x: -angle.cos(), // inward normal
                    normal_y: -angle.sin(),
                }
            })
            .collect();
        Self {
            speakers,
            array_type: ArrayType::Circular,
        }
    }
}

// ─── VirtualSource ───────────────────────────────────────────────────────────

impl VirtualSource {
    /// Create a point source at position (x, y).
    pub fn point(x: f32, y: f32) -> Self {
        Self {
            x,
            y,
            is_plane_wave: false,
            plane_wave_azimuth_deg: 0.0,
        }
    }

    /// Create a plane wave arriving from `azimuth_deg`.
    pub fn plane_wave(azimuth_deg: f32) -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            is_plane_wave: true,
            plane_wave_azimuth_deg: azimuth_deg,
        }
    }
}

// ─── WfsRenderer ─────────────────────────────────────────────────────────────

impl WfsRenderer {
    /// Create a WFS renderer.
    pub fn new(array: WfsArray, sample_rate: u32) -> Self {
        Self {
            array,
            sample_rate,
            reference_distance: 1.0,
        }
    }

    /// Create a WFS renderer with an explicit reference distance.
    pub fn with_reference_distance(mut self, reference_distance: f32) -> Self {
        self.reference_distance = reference_distance.max(0.01);
        self
    }

    /// Compute the complex driving function for a single speaker–source pair at
    /// angular frequency `omega` (rad/s).
    ///
    /// # 2.5D WFS formula
    ///
    /// For a monopole point source at **xs** = (xs, ys), and a secondary source
    /// (speaker) at **x0** = (x0, y0) with outward normal **n**:
    ///
    /// ```text
    /// r = |xs − x0|
    /// cos_angle = dot(n, xs − x0) / r
    /// k = omega / c
    /// D(x0) = −jk * cos_angle * sqrt(r_ref / r) * e^(jkr) / sqrt(2πjkr)  [simplified]
    /// ```
    ///
    /// For a plane wave with unit direction **d**, the formula reduces to:
    ///
    /// ```text
    /// D(x0) = 2 * dot(n, d) * e^(jk * dot(d, x0))
    /// ```
    pub fn compute_driving_function(
        &self,
        speaker: &WfsSpeaker,
        source: &VirtualSource,
        omega: f32,
    ) -> Complex {
        let k = omega / SPEED_OF_SOUND;

        if source.is_plane_wave {
            // Plane wave direction unit vector.
            let az = source.plane_wave_azimuth_deg.to_radians();
            let dir_x = az.cos();
            let dir_y = az.sin();

            // Dot product of normal with direction.
            let dot_nd = speaker.normal_x * dir_x + speaker.normal_y * dir_y;
            if dot_nd <= 0.0 {
                // Speaker faces away from the wave — no contribution.
                return Complex::new(0.0, 0.0);
            }

            // Phase delay: e^(jk * dot(d, x0))
            let phase = k * (dir_x * speaker.x + dir_y * speaker.y);
            Complex::from_polar(2.0 * dot_nd, phase)
        } else {
            // Point source.
            let dx = source.x - speaker.x;
            let dy = source.y - speaker.y;
            let r = (dx * dx + dy * dy).sqrt();

            if r < 1e-6 {
                return Complex::new(0.0, 0.0);
            }

            // Normalised vector from speaker to source.
            let nx_to_src = dx / r;
            let ny_to_src = dy / r;

            // cos of angle between speaker normal and direction to source.
            let cos_angle = speaker.normal_x * nx_to_src + speaker.normal_y * ny_to_src;
            if cos_angle <= 0.0 {
                // Speaker faces away from the source — inactive.
                return Complex::new(0.0, 0.0);
            }

            // 2.5D amplitude term: sqrt(r_ref / r) for cylindrical-to-spherical correction.
            let amp = (self.reference_distance / r).sqrt();

            // Phase: e^(jkr)
            let phase = k * r;

            // Simplified 2.5D driving function: D = k * cos_angle * amp * e^(jkr)
            // The (sqrt(2π/jk) / sqrt(r)) prefactor is absorbed into `amp` and a real
            // frequency-dependent scale which is left as 1 here for the simplified model.
            let magnitude = k * cos_angle * amp;

            Complex::from_polar(magnitude, phase)
        }
    }

    /// Compute the delay (in samples) and real-valued gain for each speaker in the array.
    ///
    /// The delay accounts for the propagation time from the virtual source to each speaker.
    /// For a point source the gain additionally incorporates the 2.5D inverse-square rolloff.
    ///
    /// Returns a `Vec` of `(delay_samples, gain)` — one entry per speaker.
    pub fn compute_delays_and_gains(&self, source: &VirtualSource) -> Vec<(f32, f32)> {
        let n = self.array.speakers.len();
        let mut result = Vec::with_capacity(n);

        for speaker in &self.array.speakers {
            if source.is_plane_wave {
                let az = source.plane_wave_azimuth_deg.to_radians();
                let dir_x = az.cos();
                let dir_y = az.sin();

                // Plane wave: delay is the projection of the speaker position
                // onto the plane-wave propagation direction.
                let proj = dir_x * speaker.x + dir_y * speaker.y;
                // For a plane wave arriving from azimuth θ, the delay at x0 is
                // -dot(d, x0)/c (negative because the wave has already passed
                // speakers further along d).
                let delay_m = (-proj).max(0.0);
                let delay_s = delay_m / SPEED_OF_SOUND;
                let delay_samples = delay_s * self.sample_rate as f32;

                let dot_nd = speaker.normal_x * dir_x + speaker.normal_y * dir_y;
                let gain = dot_nd.max(0.0);

                result.push((delay_samples, gain));
            } else {
                // Point source.
                let dx = source.x - speaker.x;
                let dy = source.y - speaker.y;
                let r = (dx * dx + dy * dy).sqrt().max(1e-6);

                let delay_samples = delay_in_samples(r, self.sample_rate);

                let nx = dx / r;
                let ny = dy / r;
                let cos_angle = (speaker.normal_x * nx + speaker.normal_y * ny).max(0.0);

                // 2.5D gain: cos(angle) × sqrt(r_ref / r).
                let gain = cos_angle * (self.reference_distance / r).sqrt();

                result.push((delay_samples, gain));
            }
        }

        result
    }

    /// Return the number of speakers in the array.
    pub fn num_speakers(&self) -> usize {
        self.array.speakers.len()
    }
}

/// Convert a distance in metres to a fractional sample count.
///
/// `delay_samples = distance / SPEED_OF_SOUND × sample_rate`
pub fn delay_in_samples(distance_m: f32, sample_rate: u32) -> f32 {
    distance_m / SPEED_OF_SOUND * sample_rate as f32
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Complex ─────────────────────────────────────────────────────────────

    #[test]
    fn test_complex_new() {
        let c = Complex::new(3.0, 4.0);
        assert_eq!(c.re, 3.0);
        assert_eq!(c.im, 4.0);
    }

    #[test]
    fn test_complex_abs() {
        let c = Complex::new(3.0, 4.0);
        assert!((c.abs() - 5.0).abs() < 1e-5, "abs = {}", c.abs());
    }

    #[test]
    fn test_complex_mul_identity() {
        let c = Complex::new(3.0, 4.0);
        let one = Complex::new(1.0, 0.0);
        let r = c.mul(&one);
        assert!((r.re - 3.0).abs() < 1e-5 && (r.im - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_complex_mul_j_squared() {
        // j² = -1
        let j = Complex::new(0.0, 1.0);
        let r = j.mul(&j);
        assert!((r.re + 1.0).abs() < 1e-5, "j² should be -1");
        assert!(r.im.abs() < 1e-5);
    }

    #[test]
    fn test_complex_sqrt_real_positive() {
        let c = Complex::new(4.0, 0.0);
        let r = c.sqrt();
        assert!((r.re - 2.0).abs() < 1e-4, "sqrt(4+0j) = 2, got {}", r.re);
    }

    #[test]
    fn test_complex_from_polar_roundtrip() {
        let r = 2.5_f32;
        let theta = 1.2_f32;
        let c = Complex::from_polar(r, theta);
        let abs = c.abs();
        assert!((abs - r).abs() < 1e-4, "polar abs = {abs}");
    }

    #[test]
    fn test_complex_conj() {
        let c = Complex::new(2.0, 3.0);
        let conj = c.conj();
        assert_eq!(conj.re, 2.0);
        assert_eq!(conj.im, -3.0);
    }

    // ── delay_in_samples ────────────────────────────────────────────────────

    #[test]
    fn test_delay_in_samples_343m_at_48k() {
        // 343 m at 48 000 Hz → exactly 48 000 samples.
        let d = delay_in_samples(343.0, 48_000);
        assert!((d - 48_000.0).abs() < 1.0, "delay = {d}");
    }

    #[test]
    fn test_delay_in_samples_zero_distance() {
        let d = delay_in_samples(0.0, 48_000);
        assert_eq!(d, 0.0);
    }

    #[test]
    fn test_delay_in_samples_one_metre() {
        let d = delay_in_samples(1.0, 48_000);
        let expected = 48_000.0 / 343.0;
        assert!(
            (d - expected).abs() < 0.1,
            "1 m delay = {d}, expected ~{expected}"
        );
    }

    // ── WfsArray ────────────────────────────────────────────────────────────

    #[test]
    fn test_linear_array_count() {
        let arr = WfsArray::linear(8, 0.25);
        assert_eq!(arr.speakers.len(), 8);
    }

    #[test]
    fn test_linear_array_centred() {
        let arr = WfsArray::linear(5, 1.0);
        let x_vals: Vec<f32> = arr.speakers.iter().map(|s| s.x).collect();
        let centre: f32 = x_vals.iter().sum::<f32>() / x_vals.len() as f32;
        assert!(
            centre.abs() < 1e-4,
            "Array should be centred, centre={centre}"
        );
    }

    #[test]
    fn test_linear_array_normals() {
        let arr = WfsArray::linear(4, 0.5);
        for s in &arr.speakers {
            assert!(
                (s.normal_y - 1.0).abs() < 1e-5,
                "Linear normals should point in +Y"
            );
        }
    }

    #[test]
    fn test_circular_array_count() {
        let arr = WfsArray::circular(12, 1.5);
        assert_eq!(arr.speakers.len(), 12);
    }

    #[test]
    fn test_circular_array_radii() {
        let radius = 1.5_f32;
        let arr = WfsArray::circular(8, radius);
        for s in &arr.speakers {
            let r = (s.x * s.x + s.y * s.y).sqrt();
            assert!((r - radius).abs() < 1e-4, "Speaker not on circle: r={r}");
        }
    }

    // ── WfsRenderer ─────────────────────────────────────────────────────────

    #[test]
    fn test_renderer_num_speakers() {
        let arr = WfsArray::linear(16, 0.1);
        let renderer = WfsRenderer::new(arr, 48_000);
        assert_eq!(renderer.num_speakers(), 16);
    }

    #[test]
    fn test_compute_delays_and_gains_length() {
        let arr = WfsArray::linear(8, 0.25);
        let renderer = WfsRenderer::new(arr, 48_000);
        let source = VirtualSource::point(0.0, 2.0);
        let dg = renderer.compute_delays_and_gains(&source);
        assert_eq!(dg.len(), 8);
    }

    #[test]
    fn test_compute_delays_and_gains_all_finite() {
        let arr = WfsArray::linear(8, 0.25);
        let renderer = WfsRenderer::new(arr, 48_000);
        let source = VirtualSource::point(0.0, 2.0);
        let dg = renderer.compute_delays_and_gains(&source);
        for (i, &(delay, gain)) in dg.iter().enumerate() {
            assert!(delay.is_finite(), "delay[{i}] = {delay}");
            assert!(gain.is_finite(), "gain[{i}] = {gain}");
        }
    }

    #[test]
    fn test_compute_delays_gains_centre_speaker_minimum_delay() {
        // For a source directly in front of the centre of a linear array, the
        // centre speaker should have the shortest delay.
        let arr = WfsArray::linear(5, 1.0);
        let renderer = WfsRenderer::new(arr, 48_000);
        let source = VirtualSource::point(0.0, 2.0);
        let dg = renderer.compute_delays_and_gains(&source);
        let min_delay = dg.iter().map(|&(d, _)| d).fold(f32::INFINITY, f32::min);
        let centre_delay = dg[2].0; // index 2 = centre speaker
        assert!(
            (centre_delay - min_delay).abs() < 1.0,
            "Centre speaker should have near-minimum delay: centre={centre_delay}, min={min_delay}"
        );
    }

    #[test]
    fn test_plane_wave_gain_facing_speakers() {
        // A plane wave from 270° (−Y direction) on a linear array facing +Y should
        // produce zero gain (speakers face away).
        let arr = WfsArray::linear(4, 0.5);
        let renderer = WfsRenderer::new(arr, 48_000);
        let source = VirtualSource::plane_wave(270.0);
        let dg = renderer.compute_delays_and_gains(&source);
        for (_, gain) in &dg {
            assert_eq!(*gain, 0.0, "Backward plane wave should give 0 gain");
        }
    }

    #[test]
    fn test_driving_function_point_source_nonzero() {
        let arr = WfsArray::linear(4, 0.5);
        let renderer = WfsRenderer::new(arr, 48_000);
        let speaker = &renderer.array.speakers[2];
        let source = VirtualSource::point(0.0, 2.0);
        let omega = 2.0 * PI * 1_000.0; // 1 kHz
        let d = renderer.compute_driving_function(speaker, &source, omega);
        assert!(
            d.abs() > 0.0,
            "Driving function should be non-zero for active speaker"
        );
    }

    #[test]
    fn test_driving_function_plane_wave_direction() {
        // A plane wave from 90° (front, +Y) on a +Y-normal speaker should give positive gain.
        let arr = WfsArray::linear(1, 0.1);
        let renderer = WfsRenderer::new(arr, 48_000);
        let speaker = &renderer.array.speakers[0];
        let source = VirtualSource::plane_wave(90.0);
        let omega = 2.0 * PI * 500.0;
        let d = renderer.compute_driving_function(speaker, &source, omega);
        assert!(
            d.abs() > 0.0,
            "Plane wave toward speaker should give non-zero driving fn"
        );
    }
}
