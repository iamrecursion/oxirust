//! Simple room acoustics simulation using the image source method.
//!
//! The `ImageSourceModel::compute_reflections` method iterates over all
//! independent image-source triplets (nx, ny, nz) in parallel using Rayon,
//! giving near-linear speedup on multi-core systems for large reflection orders.
//!
//! Features:
//! - 1st-order early reflections (6 walls) via image source placement
//! - Late reverberation tail using exponentially decaying noise
//! - Room impulse response (RIR) convolution for reverb application
//!
//! # Example
//!
//! ```rust
//! use oximedia_spatial::room_simulation::{RoomConfig, RoomSimulator};
//!
//! let cfg = RoomConfig::small_room();
//! let sim = RoomSimulator::new(cfg, 48_000);
//! let rir = sim.generate_rir();
//! assert!(!rir.is_empty());
//! ```

// ─── Types ────────────────────────────────────────────────────────────────────

/// Physical configuration of the room.
#[derive(Debug, Clone)]
pub struct RoomConfig {
    /// Room width in meters (X axis).
    pub width_m: f32,
    /// Room height in meters (Y axis).
    pub height_m: f32,
    /// Room depth in meters (Z axis).
    pub depth_m: f32,
    /// Wall absorption coefficient [0.0 = mirror, 1.0 = anechoic].
    pub absorption_coefficient: f32,
    /// Reverberation time at -60 dB in seconds.
    pub rt60: f32,
}

/// A single early reflection.
#[derive(Debug, Clone)]
pub struct EarlyReflection {
    /// Delay from direct sound in samples.
    pub delay_samples: usize,
    /// Linear gain of the reflection.
    pub gain: f32,
    /// Azimuth of incoming reflection (degrees).
    pub azimuth: f32,
    /// Elevation of incoming reflection (degrees).
    pub elevation: f32,
}

/// Room acoustic simulator.
#[derive(Debug, Clone)]
pub struct RoomSimulator {
    /// Room configuration.
    pub config: RoomConfig,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Listener position (X, Y, Z) in metres.
    pub listener_pos: (f32, f32, f32),
    /// Sound source position (X, Y, Z) in metres.
    pub source_pos: (f32, f32, f32),
}

// ─── RoomConfig ──────────────────────────────────────────────────────────────

impl RoomConfig {
    /// Small domestic room: 4 × 3 × 2.5 m, absorption = 0.4, RT60 = 0.3 s.
    pub fn small_room() -> Self {
        Self {
            width_m: 4.0,
            height_m: 3.0,
            depth_m: 2.5,
            absorption_coefficient: 0.4,
            rt60: 0.3,
        }
    }

    /// Concert hall: 50 × 20 × 15 m, absorption = 0.1, RT60 = 2.5 s.
    pub fn concert_hall() -> Self {
        Self {
            width_m: 50.0,
            height_m: 20.0,
            depth_m: 15.0,
            absorption_coefficient: 0.1,
            rt60: 2.5,
        }
    }

    /// Recording studio: 8 × 6 × 3 m, absorption = 0.8, RT60 = 0.15 s.
    pub fn recording_studio() -> Self {
        Self {
            width_m: 8.0,
            height_m: 6.0,
            depth_m: 3.0,
            absorption_coefficient: 0.8,
            rt60: 0.15,
        }
    }
}

// ─── RoomSimulator ───────────────────────────────────────────────────────────

/// Speed of sound (m/s).
const SPEED_OF_SOUND: f32 = 343.0;

/// A deterministic, lightweight pseudo-random number generator (xorshift32).
struct XorShift32 {
    state: u32,
}

impl XorShift32 {
    fn new(seed: u32) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    /// Return a pseudo-random f32 in [-1, 1].
    fn next_f32(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 17;
        self.state ^= self.state << 5;
        // Map u32 to [-1, 1].
        (self.state as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

impl RoomSimulator {
    /// Create a new simulator.
    ///
    /// The listener is placed at the room centre and the source 1 m in front of the listener.
    pub fn new(config: RoomConfig, sample_rate: u32) -> Self {
        let lx = config.width_m / 2.0;
        let ly = config.height_m / 2.0;
        let lz = config.depth_m / 2.0;
        // Source is 1 m in front (along -Z) of the listener.
        let sx = lx;
        let sy = ly;
        let sz = (lz - 1.0).max(0.1);

        Self {
            config,
            sample_rate,
            listener_pos: (lx, ly, lz),
            source_pos: (sx, sy, sz),
        }
    }

    /// Distance between two 3-D points.
    fn dist3(a: (f32, f32, f32), b: (f32, f32, f32)) -> f32 {
        let dx = a.0 - b.0;
        let dy = a.1 - b.1;
        let dz = a.2 - b.2;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Azimuth and elevation (degrees) of `target` as seen from `observer`.
    fn direction_to(observer: (f32, f32, f32), target: (f32, f32, f32)) -> (f32, f32) {
        let dx = target.0 - observer.0;
        let dy = target.1 - observer.1;
        let dz = target.2 - observer.2;
        let horiz_dist = (dx * dx + dz * dz).sqrt();
        let az = dy.atan2(dx).to_degrees(); // azimuth in XY plane
        let el = dz.atan2(horiz_dist).to_degrees();
        (az, el)
    }

    /// Compute 1st-order early reflections using the image source method.
    ///
    /// Six image sources are created by mirroring the source through each of the six walls.
    pub fn compute_early_reflections(&self) -> Vec<EarlyReflection> {
        let RoomConfig {
            width_m,
            height_m,
            depth_m,
            absorption_coefficient,
            ..
        } = self.config;

        let (sx, sy, sz) = self.source_pos;
        let listener = self.listener_pos;
        let alpha = absorption_coefficient;

        // Image sources for each wall (mirror through the wall plane).
        // Wall pairs: X=0, X=W; Y=0, Y=H; Z=0, Z=D.
        let image_sources: [(f32, f32, f32); 6] = [
            (-sx, sy, sz),                 // X=0 wall
            (2.0 * width_m - sx, sy, sz),  // X=W wall
            (sx, -sy, sz),                 // Y=0 wall
            (sx, 2.0 * height_m - sy, sz), // Y=H wall
            (sx, sy, -sz),                 // Z=0 wall
            (sx, sy, 2.0 * depth_m - sz),  // Z=D wall
        ];

        let mut reflections = Vec::with_capacity(6);

        for &img_src in &image_sources {
            let dist = Self::dist3(img_src, listener);
            if dist < 1e-6 {
                continue;
            }

            let delay_s = dist / SPEED_OF_SOUND;
            let delay_samples = (delay_s * self.sample_rate as f32).round() as usize;

            // Gain = (1 - alpha) / distance (inverse-square not included for simplicity)
            let gain = (1.0 - alpha) / dist.max(0.01);

            let (azimuth, elevation) = Self::direction_to(listener, img_src);

            reflections.push(EarlyReflection {
                delay_samples,
                gain,
                azimuth,
                elevation,
            });
        }

        reflections
    }

    /// Generate the room impulse response (RIR).
    ///
    /// The RIR contains:
    /// 1. Direct sound at sample 0
    /// 2. Early reflections at computed delays
    /// 3. Exponentially decaying noise tail up to `rt60 × 3` seconds
    pub fn generate_rir(&self) -> Vec<f32> {
        let rt60 = self.config.rt60;
        let sr = self.sample_rate as f32;

        let direct_dist = Self::dist3(self.source_pos, self.listener_pos);
        let direct_delay = (direct_dist / SPEED_OF_SOUND * sr).round() as usize;
        let direct_gain = 1.0 / direct_dist.max(0.01);

        let reflections = self.compute_early_reflections();

        // RIR length = rt60 × 3 seconds (late tail).
        let tail_len = (rt60 * 3.0 * sr) as usize;
        let max_delay = reflections
            .iter()
            .map(|r| r.delay_samples)
            .max()
            .unwrap_or(0);
        let rir_len = (direct_delay + 1).max(max_delay + 1).max(tail_len);

        let mut rir = vec![0.0_f32; rir_len];

        // Direct sound.
        if direct_delay < rir_len {
            rir[direct_delay] += direct_gain;
        }

        // Early reflections.
        for r in &reflections {
            if r.delay_samples < rir_len {
                rir[r.delay_samples] += r.gain;
            }
        }

        // Late reverberation tail: pseudo-random noise × exp(-6.91 * t / rt60).
        // -6.91 ≈ ln(10^(-60/20)) = ln(0.001).
        let tail_start = (reflections
            .iter()
            .map(|r| r.delay_samples)
            .max()
            .unwrap_or(direct_delay) as f32
            * 1.5) as usize;

        let mut rng = XorShift32::new(0xDEAD_BEEF);
        for i in tail_start..rir_len {
            let t = i as f32 / sr;
            let env = (-6.91 * t / rt60.max(0.001)).exp();
            rir[i] += rng.next_f32() * env * 0.1;
        }

        rir
    }

    /// Convolve dry audio with the room impulse response and normalise the output.
    pub fn apply_reverb(&self, dry: &[f32]) -> Vec<f32> {
        let rir = self.generate_rir();
        let mut wet = crate::binaural::convolve(dry, &rir);

        // Normalise to prevent clipping.
        let peak = wet.iter().fold(0.0_f32, |m, &x| m.max(x.abs()));
        if peak > 1e-10 {
            for s in &mut wet {
                *s /= peak;
            }
        }
        wet
    }
}

// ─── Frequency-Dependent Absorption ─────────────────────────────────────────

/// Standard octave-band centre frequencies (Hz) used for absorption specification.
pub const OCTAVE_BANDS: &[f32] = &[125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0];

/// Per-frequency wall absorption coefficients for all six walls.
///
/// Each wall has absorption coefficients at the standard octave bands
/// (125 Hz, 250 Hz, 500 Hz, 1 kHz, 2 kHz, 4 kHz, 8 kHz).
#[derive(Debug, Clone)]
pub struct FreqDependentAbsorption {
    /// Absorption coefficients per wall per frequency band.
    /// Outer index: wall (0..6), inner: frequency band (7 octave bands).
    /// Wall order: X=0, X=W, Y=0, Y=H, Z=0, Z=D.
    pub wall_coefficients: [[f32; 7]; 6],
}

/// A single frequency-dependent early reflection.
#[derive(Debug, Clone)]
pub struct FreqDependentReflection {
    /// Delay from direct sound in samples.
    pub delay_samples: usize,
    /// Per-band gains (one per octave band).
    pub band_gains: [f32; 7],
    /// Azimuth of incoming reflection (degrees).
    pub azimuth: f32,
    /// Elevation of incoming reflection (degrees).
    pub elevation: f32,
}

/// Room simulator with frequency-dependent wall absorption.
#[derive(Debug, Clone)]
pub struct FreqDependentRoomSimulator {
    /// Base room configuration (dimensions, RT60).
    pub config: RoomConfig,
    /// Per-wall frequency-dependent absorption.
    pub absorption: FreqDependentAbsorption,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Listener position (X, Y, Z).
    pub listener_pos: (f32, f32, f32),
    /// Source position (X, Y, Z).
    pub source_pos: (f32, f32, f32),
}

impl FreqDependentAbsorption {
    /// Create uniform absorption across all walls and frequencies.
    pub fn uniform(alpha: f32) -> Self {
        let clamped = alpha.clamp(0.0, 1.0);
        Self {
            wall_coefficients: [[clamped; 7]; 6],
        }
    }

    /// Typical plaster/concrete room: low absorption at low frequencies,
    /// moderate at mid/high.
    pub fn plaster_room() -> Self {
        // Realistic absorption coefficients for painted plaster walls
        let plaster = [0.01_f32, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07];
        Self {
            wall_coefficients: [plaster; 6],
        }
    }

    /// Acoustically treated studio: higher absorption at mid and high frequencies.
    pub fn treated_studio() -> Self {
        // Floor: hardwood (less absorptive)
        let floor = [0.15_f32, 0.11, 0.10, 0.07, 0.06, 0.07, 0.07];
        // Ceiling: acoustic tiles
        let ceiling = [0.25_f32, 0.45, 0.65, 0.75, 0.80, 0.75, 0.65];
        // Side walls: broadband absorbers
        let side_wall = [0.30_f32, 0.55, 0.80, 0.90, 0.85, 0.80, 0.70];
        // Front/back: combination of absorbers and diffusers
        let front_back = [0.20_f32, 0.40, 0.60, 0.70, 0.75, 0.70, 0.60];

        Self {
            wall_coefficients: [
                side_wall,  // X=0
                side_wall,  // X=W
                front_back, // Y=0
                front_back, // Y=H
                floor,      // Z=0
                ceiling,    // Z=D
            ],
        }
    }

    /// Carpeted room with moderate absorption.
    pub fn carpeted_room() -> Self {
        let carpet_floor = [0.08_f32, 0.24, 0.57, 0.69, 0.71, 0.73, 0.73];
        let drywall = [0.08_f32, 0.11, 0.05, 0.03, 0.02, 0.03, 0.04];
        let ceiling_tile = [0.14_f32, 0.35, 0.55, 0.72, 0.70, 0.65, 0.60];
        Self {
            wall_coefficients: [
                drywall,
                drywall,
                drywall,
                drywall,
                carpet_floor,
                ceiling_tile,
            ],
        }
    }
}

impl FreqDependentRoomSimulator {
    /// Create a new frequency-dependent simulator.
    pub fn new(config: RoomConfig, absorption: FreqDependentAbsorption, sample_rate: u32) -> Self {
        let lx = config.width_m / 2.0;
        let ly = config.height_m / 2.0;
        let lz = config.depth_m / 2.0;
        let sx = lx;
        let sy = ly;
        let sz = (lz - 1.0).max(0.1);

        Self {
            config,
            absorption,
            sample_rate,
            listener_pos: (lx, ly, lz),
            source_pos: (sx, sy, sz),
        }
    }

    /// Compute 1st-order early reflections with per-frequency gains.
    pub fn compute_freq_dependent_reflections(&self) -> Vec<FreqDependentReflection> {
        let (sx, sy, sz) = self.source_pos;
        let listener = self.listener_pos;
        let w = self.config.width_m;
        let h = self.config.height_m;
        let d = self.config.depth_m;

        // Image sources for each wall
        let image_sources: [(f32, f32, f32); 6] = [
            (-sx, sy, sz),
            (2.0 * w - sx, sy, sz),
            (sx, -sy, sz),
            (sx, 2.0 * h - sy, sz),
            (sx, sy, -sz),
            (sx, sy, 2.0 * d - sz),
        ];

        let mut reflections = Vec::with_capacity(6);

        for (wall_idx, &img_src) in image_sources.iter().enumerate() {
            let dist = RoomSimulator::dist3(img_src, listener);
            if dist < 1e-6 {
                continue;
            }

            let delay_s = dist / SPEED_OF_SOUND;
            let delay_samples = (delay_s * self.sample_rate as f32).round() as usize;

            // Per-frequency gains: (1 - alpha_band) / distance
            let mut band_gains = [0.0_f32; 7];
            for (band, gain) in band_gains.iter_mut().enumerate() {
                let alpha = self.absorption.wall_coefficients[wall_idx][band];
                *gain = (1.0 - alpha) / dist.max(0.01);
            }

            let (azimuth, elevation) = RoomSimulator::direction_to(listener, img_src);

            reflections.push(FreqDependentReflection {
                delay_samples,
                band_gains,
                azimuth,
                elevation,
            });
        }

        reflections
    }

    /// Generate a frequency-dependent room impulse response.
    ///
    /// The RIR is generated per octave band and then summed.
    /// Each band uses the frequency-dependent reflection gains.
    pub fn generate_rir(&self) -> Vec<f32> {
        let rt60 = self.config.rt60;
        let sr = self.sample_rate as f32;

        let direct_dist = RoomSimulator::dist3(self.source_pos, self.listener_pos);
        let direct_delay = (direct_dist / SPEED_OF_SOUND * sr).round() as usize;
        let direct_gain = 1.0 / direct_dist.max(0.01);

        let reflections = self.compute_freq_dependent_reflections();

        let tail_len = (rt60 * 3.0 * sr) as usize;
        let max_delay = reflections
            .iter()
            .map(|r| r.delay_samples)
            .max()
            .unwrap_or(0);
        let rir_len = (direct_delay + 1).max(max_delay + 1).max(tail_len);

        let mut rir = vec![0.0_f32; rir_len];

        // Direct sound
        if direct_delay < rir_len {
            rir[direct_delay] += direct_gain;
        }

        // Early reflections: average over frequency bands for the broadband RIR
        for r in &reflections {
            if r.delay_samples < rir_len {
                let avg_gain: f32 = r.band_gains.iter().sum::<f32>() / 7.0;
                rir[r.delay_samples] += avg_gain;
            }
        }

        // Late reverberation with frequency-dependent decay
        // Use the average absorption across all walls and bands to estimate decay
        let avg_absorption: f32 = self
            .absorption
            .wall_coefficients
            .iter()
            .flat_map(|w| w.iter())
            .sum::<f32>()
            / 42.0; // 6 walls * 7 bands
        let effective_rt60 = if avg_absorption > 1e-6 {
            rt60 * (1.0 - avg_absorption).max(0.01)
        } else {
            rt60
        };

        let tail_start = (reflections
            .iter()
            .map(|r| r.delay_samples)
            .max()
            .unwrap_or(direct_delay) as f32
            * 1.5) as usize;

        let mut rng = XorShift32::new(0xDEAD_BEEF);
        for i in tail_start..rir_len {
            let t = i as f32 / sr;
            let env = (-6.91 * t / effective_rt60.max(0.001)).exp();
            rir[i] += rng.next_f32() * env * 0.1;
        }

        rir
    }

    /// Compute the per-band RT60 estimates using the Sabine equation.
    ///
    /// RT60 = 0.161 * V / A, where V = room volume, A = total absorption area per band.
    pub fn estimate_per_band_rt60(&self) -> [f32; 7] {
        let w = self.config.width_m;
        let h = self.config.height_m;
        let d = self.config.depth_m;
        let volume = w * h * d;

        // Surface areas for each wall pair
        let areas = [
            h * d,
            h * d, // X walls
            w * d,
            w * d, // Y walls
            w * h,
            w * h, // Z walls
        ];

        let mut rt60_bands = [0.0_f32; 7];

        for band in 0..7 {
            let total_absorption: f32 = (0..6)
                .map(|wall| areas[wall] * self.absorption.wall_coefficients[wall][band])
                .sum();

            rt60_bands[band] = if total_absorption > 1e-6 {
                0.161 * volume / total_absorption
            } else {
                10.0 // Cap for nearly reflective rooms
            };
        }

        rt60_bands
    }
}

// ─── Image Source Method (N-th order, per-wall absorption) ───────────────────

/// Physical room configuration with per-wall absorption coefficients.
///
/// Models a shoebox (rectangular parallelepiped) room with six independent
/// wall surfaces, each with its own broadband absorption coefficient.
#[derive(Debug, Clone)]
pub struct RoomAcousticsParams {
    /// Room width in meters (X axis).
    pub room_width_m: f32,
    /// Room length in meters (Y axis).
    pub room_length_m: f32,
    /// Room height in meters (Z axis).
    pub room_height_m: f32,
    /// Absorption coefficients per wall: [left, right, front, back, floor, ceiling].
    /// Each value in [0.0, 1.0]: 0.0 = fully reflective, 1.0 = fully absorptive.
    pub absorption_coefficients: [f32; 6],
    /// Approximate high-frequency air absorption rolloff (Hz).
    pub air_absorption_hz: f32,
}

impl RoomAcousticsParams {
    /// Standard listening room (6 × 4 × 3 m) with moderate absorption.
    pub fn listening_room() -> Self {
        Self {
            room_width_m: 6.0,
            room_length_m: 4.0,
            room_height_m: 3.0,
            absorption_coefficients: [0.3, 0.3, 0.25, 0.25, 0.4, 0.2],
            air_absorption_hz: 20_000.0,
        }
    }

    /// Anechoic chamber: all walls fully absorptive (no reflections).
    pub fn anechoic() -> Self {
        Self {
            room_width_m: 6.0,
            room_length_m: 6.0,
            room_height_m: 3.0,
            absorption_coefficients: [1.0; 6],
            air_absorption_hz: 20_000.0,
        }
    }
}

/// A single early reflection produced by the image source method.
#[derive(Debug, Clone)]
pub struct ImageReflection {
    /// Delay from direct sound in fractional samples (seconds when returned by
    /// `compute_reflections`; converted to samples by `apply_to_signal`).
    pub delay_samples: f32,
    /// Amplitude gain (inverse-distance attenuation × wall absorption product).
    pub gain: f32,
    /// Number of wall bounces (reflection order).
    pub wall_bounces: u32,
}

/// Image source method for early reflections up to N-th order.
///
/// Generates mirror-image sources by reflecting the source position across
/// each wall surface in a shoebox room.  For each image source up to
/// `max_order` total bounces, the delay and gain are computed from the
/// image-to-listener distance and the product of the reflection factors of
/// all walls involved.
///
/// **Parallel execution:** `compute_reflections` uses Rayon to process all
/// independent image-source triplets in parallel, giving near-linear speedup
/// for high reflection orders.
///
/// # Example
///
/// ```rust
/// use oximedia_spatial::room_simulation::{RoomAcousticsParams, ImageSourceModel};
///
/// let model = ImageSourceModel::new(1);
/// let room  = RoomAcousticsParams::listening_room();
/// let refs  = model.compute_reflections(&room, (1.5, 1.0, 1.0), (3.0, 2.0, 1.5));
/// assert_eq!(refs.len(), 6); // six first-order reflections
/// ```
#[derive(Debug, Clone)]
pub struct ImageSourceModel {
    /// Maximum reflection order (1 = first-order only, 2 = second-order, etc.).
    pub max_order: u32,
    /// Speed of sound in m/s (default 343.0).
    pub speed_of_sound: f32,
}

impl ImageSourceModel {
    /// Create a new model with the given reflection order and the standard
    /// speed of sound (343.0 m/s at 20 °C).
    pub fn new(max_order: u32) -> Self {
        Self {
            max_order,
            speed_of_sound: 343.0,
        }
    }

    /// Compute early reflections for a given room, source, and listener.
    ///
    /// Uses the image source method: the source is reflected across each wall
    /// boundary, iterating over all integer triplets (nx, ny, nz) whose
    /// L1-norm equals the reflection order, up to `max_order`.
    ///
    /// # Arguments
    /// - `room`: room geometry and per-wall absorption
    /// - `source_pos`: sound source position (x, y, z) in metres
    /// - `listener_pos`: listener position (x, y, z) in metres
    ///
    /// Returns reflections sorted by ascending delay (stored as seconds in the
    /// `delay_samples` field — pass the list to `apply_to_signal` to convert).
    pub fn compute_reflections(
        &self,
        room: &RoomAcousticsParams,
        source_pos: (f32, f32, f32),
        listener_pos: (f32, f32, f32),
    ) -> Vec<ImageReflection> {
        use rayon::prelude::*;

        let order = self.max_order as i32;
        let speed = self.speed_of_sound;
        let max_order = self.max_order;

        // Build the full list of triplets (nx, ny, nz) whose total order is in
        // [1, max_order].  This is the same set as the serial version but
        // stored as a flat Vec so Rayon can split it across threads.
        let triplets: Vec<(i32, i32, i32)> = (-order..=order)
            .flat_map(|nx| {
                (-order..=order).flat_map(move |ny| (-order..=order).map(move |nz| (nx, ny, nz)))
            })
            .filter(|&(nx, ny, nz)| {
                let total = nx.unsigned_abs() + ny.unsigned_abs() + nz.unsigned_abs();
                total >= 1 && total <= max_order
            })
            .collect();

        // Process each triplet independently in parallel.
        let mut reflections: Vec<ImageReflection> = triplets
            .par_iter()
            .filter_map(|&(nx, ny, nz)| {
                let total_order = nx.unsigned_abs() + ny.unsigned_abs() + nz.unsigned_abs();

                // Image source position in the extended (tiled) space.
                let (ix, iy, iz) = image_source_pos(source_pos, room, nx, ny, nz);

                // Distance from image source to listener.
                let dx = ix - listener_pos.0;
                let dy = iy - listener_pos.1;
                let dz = iz - listener_pos.2;
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();

                if dist < 1e-6 {
                    return None;
                }

                // Delay expressed in seconds (converted to samples later).
                let delay_s = dist / speed;

                // Gain: 1/r inverse-distance × product of wall reflection factors.
                let wall_gain = compute_image_wall_gain(&room.absorption_coefficients, nx, ny, nz);
                let dist_gain = 1.0 / dist.max(0.01);
                let gain = wall_gain * dist_gain;

                Some(ImageReflection {
                    delay_samples: delay_s,
                    gain,
                    wall_bounces: total_order,
                })
            })
            .collect();

        // Sort by ascending delay (deterministic output order regardless of thread scheduling).
        reflections.sort_by(|a, b| {
            a.delay_samples
                .partial_cmp(&b.delay_samples)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        reflections
    }

    /// Apply a set of early reflections to an input mono signal.
    ///
    /// Each reflection is a delayed, attenuated copy of the input.  All copies
    /// are summed into an output buffer.  The output is truncated to the input
    /// length (reflections that extend beyond the window are discarded).
    ///
    /// # Arguments
    /// - `signal`: input audio samples
    /// - `reflections`: list returned by `compute_reflections` (delay stored in seconds)
    /// - `sample_rate`: sample rate in Hz — used to convert delay seconds → samples
    ///
    /// Returns a buffer with the same length as `signal`.
    pub fn apply_to_signal(
        &self,
        signal: &[f32],
        reflections: &[ImageReflection],
        sample_rate: f32,
    ) -> Vec<f32> {
        if signal.is_empty() || reflections.is_empty() {
            return signal.to_vec();
        }

        // Find maximum delay to allocate a sufficient buffer.
        let max_delay_s = reflections
            .iter()
            .map(|r| r.delay_samples)
            .fold(0.0_f32, f32::max);
        let max_delay_samps = (max_delay_s * sample_rate).ceil() as usize;
        let out_len = signal.len() + max_delay_samps;

        let mut output = vec![0.0_f32; out_len];

        for refl in reflections {
            let delay_samp = (refl.delay_samples * sample_rate).round() as usize;
            let gain = refl.gain;
            for (i, &s) in signal.iter().enumerate() {
                let idx = i + delay_samp;
                if idx < out_len {
                    output[idx] += s * gain;
                }
            }
        }

        // Truncate to input length (reflections beyond the window are dropped).
        output.truncate(signal.len());
        output
    }
}

// ─── Image source helpers ─────────────────────────────────────────────────────

/// Compute the 3-D position of an image source given integer reflection indices.
///
/// For a shoebox room with dimensions (W, L, H), the image source for
/// indices (nx, ny, nz) is placed at the appropriate mirror position in the
/// periodically tiled space:
/// - Even n: the source sits at `n × dim + src_coord`
/// - Odd  n: the source is mirrored to `n × dim + (dim - src_coord)`
fn image_source_pos(
    src: (f32, f32, f32),
    room: &RoomAcousticsParams,
    nx: i32,
    ny: i32,
    nz: i32,
) -> (f32, f32, f32) {
    (
        image_coord_1d(src.0, room.room_width_m, nx),
        image_coord_1d(src.1, room.room_length_m, ny),
        image_coord_1d(src.2, room.room_height_m, nz),
    )
}

/// Compute a single 1-D image source coordinate.
#[inline]
fn image_coord_1d(src: f32, dim: f32, n: i32) -> f32 {
    if n % 2 == 0 {
        n as f32 * dim + src
    } else {
        n as f32 * dim + (dim - src)
    }
}

/// Compute the combined wall-reflection gain for a path (nx, ny, nz).
///
/// `absorption_coefficients` layout: [left, right, front, back, floor, ceiling].
///
/// The reflection factor for a wall with absorption α is `√(1 − α)`.
/// Each axis contributes alternating bounces off its two walls:
/// - X axis: left (idx 0), right (idx 1)
/// - Y axis: front (idx 2), back (idx 3)
/// - Z axis: floor (idx 4), ceiling (idx 5)
fn compute_image_wall_gain(absorption: &[f32; 6], nx: i32, ny: i32, nz: i32) -> f32 {
    /// Gain for `|n|` bounces alternating between two walls with given absorptions.
    #[inline]
    fn axis_gain(n: i32, a0: f32, a1: f32) -> f32 {
        let abs_n = n.unsigned_abs();
        let bounces_0 = (abs_n + 1) / 2;
        let bounces_1 = abs_n / 2;
        let r0 = (1.0 - a0.clamp(0.0, 1.0)).sqrt();
        let r1 = (1.0 - a1.clamp(0.0, 1.0)).sqrt();
        r0.powi(bounces_0 as i32) * r1.powi(bounces_1 as i32)
    }

    axis_gain(nx, absorption[0], absorption[1])
        * axis_gain(ny, absorption[2], absorption[3])
        * axis_gain(nz, absorption[4], absorption[5])
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn rms(buf: &[f32]) -> f32 {
        let sum: f32 = buf.iter().map(|x| x * x).sum();
        (sum / buf.len().max(1) as f32).sqrt()
    }

    // ── RoomConfig ───────────────────────────────────────────────────────────

    #[test]
    fn test_small_room_config() {
        let cfg = RoomConfig::small_room();
        assert_eq!(cfg.width_m, 4.0);
        assert_eq!(cfg.height_m, 3.0);
        assert_eq!(cfg.depth_m, 2.5);
        assert_eq!(cfg.absorption_coefficient, 0.4);
        assert_eq!(cfg.rt60, 0.3);
    }

    #[test]
    fn test_concert_hall_config() {
        let cfg = RoomConfig::concert_hall();
        assert!(cfg.width_m > 40.0);
        assert!(cfg.rt60 > 2.0);
    }

    #[test]
    fn test_recording_studio_config() {
        let cfg = RoomConfig::recording_studio();
        assert!(cfg.absorption_coefficient > 0.5);
        assert!(cfg.rt60 < 0.2);
    }

    // ── RoomSimulator ────────────────────────────────────────────────────────

    #[test]
    fn test_new_places_listener_at_centre() {
        let cfg = RoomConfig::small_room();
        let sim = RoomSimulator::new(cfg.clone(), 48_000);
        assert!((sim.listener_pos.0 - cfg.width_m / 2.0).abs() < 0.01);
        assert!((sim.listener_pos.1 - cfg.height_m / 2.0).abs() < 0.01);
        assert!((sim.listener_pos.2 - cfg.depth_m / 2.0).abs() < 0.01);
    }

    #[test]
    fn test_early_reflections_count() {
        let sim = RoomSimulator::new(RoomConfig::small_room(), 48_000);
        let refs = sim.compute_early_reflections();
        // 6 walls → up to 6 reflections (all should be valid for a room with positive dims).
        assert_eq!(
            refs.len(),
            6,
            "Expected 6 first-order reflections, got {}",
            refs.len()
        );
    }

    #[test]
    fn test_early_reflections_positive_gain() {
        let sim = RoomSimulator::new(RoomConfig::small_room(), 48_000);
        let refs = sim.compute_early_reflections();
        for r in &refs {
            assert!(r.gain > 0.0, "All reflections should have positive gain");
        }
    }

    #[test]
    fn test_early_reflections_positive_delay() {
        let sim = RoomSimulator::new(RoomConfig::small_room(), 48_000);
        let refs = sim.compute_early_reflections();
        for r in &refs {
            assert!(r.delay_samples > 0, "Reflections must arrive after t=0");
        }
    }

    #[test]
    fn test_generate_rir_not_empty() {
        let sim = RoomSimulator::new(RoomConfig::small_room(), 48_000);
        let rir = sim.generate_rir();
        assert!(!rir.is_empty());
    }

    #[test]
    fn test_generate_rir_has_direct_sound() {
        let sim = RoomSimulator::new(RoomConfig::small_room(), 48_000);
        let rir = sim.generate_rir();
        // Direct sound should be present as the largest non-tail peak.
        let peak = rir.iter().fold(0.0_f32, |m, &x| m.max(x.abs()));
        assert!(peak > 0.0, "RIR should have non-zero direct sound");
    }

    #[test]
    fn test_rir_length_covers_rt60() {
        let cfg = RoomConfig::small_room();
        let sr = 48_000_u32;
        let sim = RoomSimulator::new(cfg.clone(), sr);
        let rir = sim.generate_rir();
        let expected_min_len = (cfg.rt60 * 3.0 * sr as f32) as usize;
        assert!(
            rir.len() >= expected_min_len,
            "RIR length {} should be >= rt60*3 = {} samples",
            rir.len(),
            expected_min_len
        );
    }

    #[test]
    fn test_apply_reverb_output_length() {
        let sim = RoomSimulator::new(RoomConfig::small_room(), 48_000);
        let dry = vec![1.0_f32; 512];
        let wet = sim.apply_reverb(&dry);
        // Wet output must be at least as long as dry.
        assert!(wet.len() >= dry.len());
    }

    #[test]
    fn test_apply_reverb_normalised() {
        let sim = RoomSimulator::new(RoomConfig::small_room(), 48_000);
        let dry: Vec<f32> = (0..512).map(|i| (i as f32 * 0.1).sin()).collect();
        let wet = sim.apply_reverb(&dry);
        let peak = wet.iter().fold(0.0_f32, |m, &x| m.max(x.abs()));
        // After normalisation, peak should not exceed 1.0.
        assert!(
            peak <= 1.0 + 1e-5,
            "Normalised output peak should be ≤ 1.0, got {peak}"
        );
    }

    #[test]
    fn test_concert_hall_rir_longer_than_small_room() {
        let sim_small = RoomSimulator::new(RoomConfig::small_room(), 48_000);
        let sim_hall = RoomSimulator::new(RoomConfig::concert_hall(), 48_000);
        let rir_small = sim_small.generate_rir();
        let rir_hall = sim_hall.generate_rir();
        assert!(
            rir_hall.len() > rir_small.len(),
            "Concert hall RIR ({}) should be longer than small room RIR ({})",
            rir_hall.len(),
            rir_small.len()
        );
    }

    #[test]
    fn test_highly_absorptive_room_lower_reflection_gain() {
        let mut cfg_dead = RoomConfig::recording_studio();
        cfg_dead.absorption_coefficient = 0.99;
        let mut cfg_live = RoomConfig::recording_studio();
        cfg_live.absorption_coefficient = 0.01;

        let sim_dead = RoomSimulator::new(cfg_dead, 48_000);
        let sim_live = RoomSimulator::new(cfg_live, 48_000);

        let refs_dead = sim_dead.compute_early_reflections();
        let refs_live = sim_live.compute_early_reflections();

        let sum_dead: f32 = refs_dead.iter().map(|r| r.gain).sum();
        let sum_live: f32 = refs_live.iter().map(|r| r.gain).sum();

        assert!(
            sum_dead < sum_live,
            "High absorption should produce lower reflection gains: dead={sum_dead}, live={sum_live}"
        );
    }

    #[test]
    fn test_apply_reverb_produces_audio_from_silence() {
        // Dry silence in → wet silence out (no self-noise from RIR applied to zeros).
        let sim = RoomSimulator::new(RoomConfig::small_room(), 48_000);
        let dry = vec![0.0_f32; 256];
        let wet = sim.apply_reverb(&dry);
        assert!(
            wet.iter().all(|&x| x == 0.0),
            "Silence convolved with any IR should remain silence"
        );
    }

    // ── FreqDependentAbsorption ──────────────────────────────────────────────

    #[test]
    fn test_uniform_absorption_all_same() {
        let abs = FreqDependentAbsorption::uniform(0.5);
        for wall in &abs.wall_coefficients {
            for &coeff in wall {
                assert!((coeff - 0.5).abs() < 1e-6, "All coefficients should be 0.5");
            }
        }
    }

    #[test]
    fn test_uniform_absorption_clamped() {
        let abs = FreqDependentAbsorption::uniform(1.5);
        for wall in &abs.wall_coefficients {
            for &coeff in wall {
                assert!(coeff <= 1.0, "Absorption should be clamped to 1.0");
            }
        }
        let abs2 = FreqDependentAbsorption::uniform(-0.5);
        for wall in &abs2.wall_coefficients {
            for &coeff in wall {
                assert!(coeff >= 0.0, "Absorption should be clamped to 0.0");
            }
        }
    }

    #[test]
    fn test_plaster_room_low_absorption() {
        let abs = FreqDependentAbsorption::plaster_room();
        for wall in &abs.wall_coefficients {
            for &coeff in wall {
                assert!(
                    coeff < 0.2,
                    "Plaster should have low absorption, got {coeff}"
                );
                assert!(coeff >= 0.0, "Absorption must be non-negative");
            }
        }
    }

    #[test]
    fn test_treated_studio_higher_mid_high() {
        let abs = FreqDependentAbsorption::treated_studio();
        // Ceiling absorption at 1 kHz (index 3) should be higher than at 125 Hz (index 0).
        let ceiling = abs.wall_coefficients[5];
        assert!(
            ceiling[3] > ceiling[0],
            "Ceiling should absorb more at 1kHz than 125Hz: 1kHz={}, 125Hz={}",
            ceiling[3],
            ceiling[0]
        );
    }

    #[test]
    fn test_carpeted_room_floor_vs_walls() {
        let abs = FreqDependentAbsorption::carpeted_room();
        // Floor (index 4) at 500 Hz should be much more absorptive than drywall walls.
        let floor_500 = abs.wall_coefficients[4][2]; // 500 Hz
        let wall_500 = abs.wall_coefficients[0][2]; // X=0 wall, 500 Hz
        assert!(
            floor_500 > wall_500,
            "Carpet floor should absorb more than drywall at 500Hz: floor={floor_500}, wall={wall_500}"
        );
    }

    // ── FreqDependentRoomSimulator ───────────────────────────────────────────

    #[test]
    fn test_freq_dep_sim_constructs() {
        let cfg = RoomConfig::small_room();
        let abs = FreqDependentAbsorption::treated_studio();
        let sim = FreqDependentRoomSimulator::new(cfg, abs, 48_000);
        assert!(sim.sample_rate == 48_000);
    }

    #[test]
    fn test_freq_dep_reflections_count() {
        let cfg = RoomConfig::small_room();
        let abs = FreqDependentAbsorption::uniform(0.3);
        let sim = FreqDependentRoomSimulator::new(cfg, abs, 48_000);
        let refs = sim.compute_freq_dependent_reflections();
        assert_eq!(refs.len(), 6, "Should have 6 first-order reflections");
    }

    #[test]
    fn test_freq_dep_reflections_band_gains_positive() {
        let cfg = RoomConfig::small_room();
        let abs = FreqDependentAbsorption::uniform(0.3);
        let sim = FreqDependentRoomSimulator::new(cfg, abs, 48_000);
        let refs = sim.compute_freq_dependent_reflections();
        for r in &refs {
            for (band, &gain) in r.band_gains.iter().enumerate() {
                assert!(
                    gain > 0.0,
                    "Band gain should be positive: band={band}, gain={gain}"
                );
            }
        }
    }

    #[test]
    fn test_freq_dep_reflections_vary_with_absorption() {
        let cfg = RoomConfig::small_room();
        let abs = FreqDependentAbsorption::treated_studio();
        let sim = FreqDependentRoomSimulator::new(cfg, abs, 48_000);
        let refs = sim.compute_freq_dependent_reflections();
        // For treated studio, different bands should have different gains.
        for r in &refs {
            let min_gain = r.band_gains.iter().cloned().fold(f32::MAX, f32::min);
            let max_gain = r.band_gains.iter().cloned().fold(f32::MIN, f32::max);
            // At least some variation expected for non-uniform absorption.
            assert!(
                (max_gain - min_gain).abs() > 1e-6 || min_gain < 0.01,
                "Band gains should vary for treated studio"
            );
        }
    }

    #[test]
    fn test_freq_dep_rir_not_empty() {
        let cfg = RoomConfig::small_room();
        let abs = FreqDependentAbsorption::plaster_room();
        let sim = FreqDependentRoomSimulator::new(cfg, abs, 48_000);
        let rir = sim.generate_rir();
        assert!(!rir.is_empty(), "RIR should not be empty");
    }

    #[test]
    fn test_freq_dep_rir_has_direct_sound() {
        let cfg = RoomConfig::small_room();
        let abs = FreqDependentAbsorption::uniform(0.5);
        let sim = FreqDependentRoomSimulator::new(cfg, abs, 48_000);
        let rir = sim.generate_rir();
        let peak = rir.iter().fold(0.0_f32, |m, &x| m.max(x.abs()));
        assert!(peak > 0.0, "RIR should have direct sound");
    }

    #[test]
    fn test_per_band_rt60_varies_with_absorption() {
        let cfg = RoomConfig::small_room();
        let abs = FreqDependentAbsorption::treated_studio();
        let sim = FreqDependentRoomSimulator::new(cfg, abs, 48_000);
        let rt60s = sim.estimate_per_band_rt60();

        // All values should be positive and finite.
        for (band, &rt60) in rt60s.iter().enumerate() {
            assert!(rt60 > 0.0, "RT60 should be positive at band {band}");
            assert!(rt60.is_finite(), "RT60 should be finite at band {band}");
        }

        // Low frequencies (less absorption) should have longer RT60 than high.
        assert!(
            rt60s[0] > rt60s[4],
            "125 Hz RT60 ({}) should be > 2 kHz RT60 ({})",
            rt60s[0],
            rt60s[4]
        );
    }

    #[test]
    fn test_per_band_rt60_uniform_gives_same_across_bands() {
        let cfg = RoomConfig::small_room();
        let abs = FreqDependentAbsorption::uniform(0.3);
        let sim = FreqDependentRoomSimulator::new(cfg, abs, 48_000);
        let rt60s = sim.estimate_per_band_rt60();

        // With uniform absorption, all bands should have the same RT60.
        let first = rt60s[0];
        for (band, &rt60) in rt60s.iter().enumerate() {
            assert!(
                (rt60 - first).abs() < 1e-3,
                "Uniform absorption should give equal RT60: band {band} = {rt60}, band 0 = {first}"
            );
        }
    }

    #[test]
    fn test_freq_dep_more_absorptive_shorter_rir() {
        let cfg_live = RoomConfig::small_room();
        let cfg_dead = RoomConfig::small_room();
        let abs_live = FreqDependentAbsorption::plaster_room();
        let abs_dead = FreqDependentAbsorption::treated_studio();
        let sim_live = FreqDependentRoomSimulator::new(cfg_live, abs_live, 48_000);
        let sim_dead = FreqDependentRoomSimulator::new(cfg_dead, abs_dead, 48_000);

        let rir_live = sim_live.generate_rir();
        let rir_dead = sim_dead.generate_rir();

        let rms_live = rms(&rir_live);
        let rms_dead = rms(&rir_dead);

        // Dead room should have lower overall RMS than live room.
        assert!(
            rms_dead <= rms_live + 0.01,
            "Treated room should have <= RMS: dead={rms_dead}, live={rms_live}"
        );
    }

    #[test]
    fn test_octave_bands_correct_count() {
        assert_eq!(OCTAVE_BANDS.len(), 7, "Should have 7 octave bands");
        assert!((OCTAVE_BANDS[0] - 125.0).abs() < 0.1);
        assert!((OCTAVE_BANDS[6] - 8000.0).abs() < 0.1);
    }

    // ── ImageSourceModel tests ──────────────────────────────────────────────

    #[test]
    fn test_room_acoustics_params_listening_room() {
        let room = RoomAcousticsParams::listening_room();
        assert!(room.room_width_m > 0.0);
        assert_eq!(room.absorption_coefficients.len(), 6);
    }

    #[test]
    fn test_image_source_model_first_order_reflections() {
        let model = ImageSourceModel::new(1);
        let room = RoomAcousticsParams::listening_room();
        let source = (3.0_f32, 2.0, 1.5);
        let listener = (1.0_f32, 2.0, 1.5);
        let refs = model.compute_reflections(&room, source, listener);
        // First-order: triplets with |nx|+|ny|+|nz| == 1 — exactly 6 combinations.
        assert_eq!(
            refs.len(),
            6,
            "first-order should produce 6 reflections, got {}",
            refs.len()
        );
    }

    #[test]
    fn test_image_source_reflections_sorted_by_delay() {
        let model = ImageSourceModel::new(2);
        let room = RoomAcousticsParams::listening_room();
        let refs = model.compute_reflections(&room, (1.5, 1.0, 1.0), (3.0, 2.0, 1.5));
        for w in refs.windows(2) {
            assert!(
                w[0].delay_samples <= w[1].delay_samples,
                "reflections should be sorted ascending by delay"
            );
        }
    }

    #[test]
    fn test_image_source_reflections_positive_gain() {
        let model = ImageSourceModel::new(2);
        let room = RoomAcousticsParams::listening_room();
        let refs = model.compute_reflections(&room, (1.5, 1.0, 1.0), (3.0, 2.0, 1.5));
        for r in &refs {
            assert!(
                r.gain > 0.0,
                "all reflections should have positive gain, got {}",
                r.gain
            );
        }
    }

    #[test]
    fn test_image_source_anechoic_zero_gain() {
        // Anechoic room: absorption = 1.0 for all walls → reflection factor = 0 → gain = 0.
        let model = ImageSourceModel::new(1);
        let room = RoomAcousticsParams::anechoic();
        let refs = model.compute_reflections(&room, (1.5, 1.0, 1.0), (3.0, 2.0, 1.5));
        for r in &refs {
            assert!(
                r.gain.abs() < 1e-6,
                "anechoic room should have zero reflection gain, got {}",
                r.gain
            );
        }
    }

    #[test]
    fn test_apply_to_signal_no_change_on_empty_reflections() {
        let model = ImageSourceModel::new(1);
        let signal = vec![1.0_f32, 0.5, -0.5, 0.0];
        let out = model.apply_to_signal(&signal, &[], 48_000.0);
        assert_eq!(out, signal);
    }

    #[test]
    fn test_apply_to_signal_output_length() {
        let model = ImageSourceModel::new(1);
        let room = RoomAcousticsParams::listening_room();
        let refs = model.compute_reflections(&room, (1.5, 1.0, 1.0), (3.0, 2.0, 1.5));
        let signal = vec![1.0_f32; 1024];
        let out = model.apply_to_signal(&signal, &refs, 48_000.0);
        assert_eq!(
            out.len(),
            signal.len(),
            "output length should equal input length"
        );
    }

    #[test]
    fn test_apply_to_signal_produces_nonzero_output() {
        let model = ImageSourceModel::new(1);
        let room = RoomAcousticsParams::listening_room();
        let refs = model.compute_reflections(&room, (1.5, 1.0, 1.0), (3.0, 2.0, 1.5));
        let signal: Vec<f32> = (0..4096).map(|i| (i as f32 * 0.01).sin()).collect();
        let out = model.apply_to_signal(&signal, &refs, 48_000.0);
        let energy: f32 = out.iter().map(|x| x * x).sum();
        assert!(energy > 0.0, "reflected signal should have non-zero energy");
    }

    #[test]
    fn test_wall_bounces_correct_order() {
        let model = ImageSourceModel::new(2);
        let room = RoomAcousticsParams::listening_room();
        let refs = model.compute_reflections(&room, (1.5, 1.0, 1.0), (3.0, 2.0, 1.5));
        for r in &refs {
            assert!(
                r.wall_bounces >= 1 && r.wall_bounces <= 2,
                "wall_bounces should be in [1,2] for max_order=2, got {}",
                r.wall_bounces
            );
        }
    }
}
