//! Audio object metadata and renderer for immersive audio formats.
//!
//! This module models the *object audio* paradigm used in formats such as
//! Dolby Atmos, Auro-3D, DTS:X and Immerse Audio.  Each audio object carries
//! a position (x, y, z), a gain, and spatial spread parameters (width, height,
//! depth).  The renderer maps objects to speaker-bed channels using simple
//! panning and spread algorithms.
//!
//! ADM (Audio Definition Model) metadata parsing is also included — a minimal
//! XML attribute parser handles `audioObjectID`, `X`, `Y`, `Z` without pulling
//! in an external XML crate.
//!
//! # Coordinate convention
//! - All Cartesian coordinates in [-1, 1] (ADM standard).
//! - x = 1 → right, x = -1 → left
//! - y = 1 → front, y = -1 → back
//! - z = 1 → top,   z = -1 → bottom
//!
//! # Speaker layout channel indices
//! The `SpeakerLayout` enum documents the channel assignment for each layout.
//! Channel numbers follow the ITU-R BS.2051 / ADM convention.

use std::collections::HashMap;

use crate::SpatialError;

// ─── Types ────────────────────────────────────────────────────────────────────

/// Spatial spread and position parameters for a single audio object.
#[derive(Debug, Clone)]
pub struct AudioObject {
    /// Unique object identifier.
    pub id: u32,
    /// X position in [-1, 1]: negative = left, positive = right.
    pub x: f32,
    /// Y position in [-1, 1]: negative = back, positive = front.
    pub y: f32,
    /// Z position in [-1, 1]: negative = below, positive = above.
    pub z: f32,
    /// Linear gain applied to the object signal before panning.
    pub gain: f32,
    /// Spatial width spread in X [0, 2] (0 = point source).
    pub width: f32,
    /// Spatial height spread in Z [0, 2].
    pub height: f32,
    /// Spatial depth spread in Y [0, 2].
    pub depth: f32,
    /// Divergence: 0 = fully directional, 1 = fully omnidirectional.
    pub divergence: f32,
}

/// Object-audio container format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioObjectFormat {
    /// Dolby Atmos (Dolby ED2 / IMF).
    Atmos,
    /// Auro-3D (Galaxy Studios).
    Auro3D,
    /// DTS:X.
    DtsX,
    /// Immerse Audio (Waves / Xperi).
    ImmerseAudio,
}

/// Speaker layout for bed rendering.
///
/// Channel indices follow ITU-R BS.2051-3 where applicable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeakerLayout {
    /// 2-channel stereo: 0=L, 1=R.
    Stereo,
    /// 5.1 surround: 0=L, 1=R, 2=C, 3=LFE, 4=Ls, 5=Rs.
    FiveOnOne,
    /// 7.1 surround: 0=L, 1=R, 2=C, 3=LFE, 4=Ls, 5=Rs, 6=Lss, 7=Rss.
    SevenOnOne,
    /// Dolby Atmos 7.1.4: same as 7.1 plus 8=Ltf, 9=Rtf, 10=Ltr, 11=Rtr.
    SevenOnFourOnFour,
    /// 9.1.6: 0=L,1=R,2=C,3=LFE,4=Ls,5=Rs,6=Lss,7=Rss,8=Lts,9=Rts +6 heights.
    NineOnOnePointSix,
}

/// Object audio renderer — maps `AudioObject`s to speaker-bed channels.
#[derive(Debug, Clone)]
pub struct ObjectRenderer {
    /// Objects managed by this renderer, keyed by their `id`.
    pub objects: HashMap<u32, AudioObject>,
    /// Target speaker layout.
    pub output_layout: SpeakerLayout,
}

/// ADM Cartesian position (all values in [-1, 1]).
#[derive(Debug, Clone, Copy)]
pub struct Cartesian3D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// A single ADM audioObject element.
#[derive(Debug, Clone)]
pub struct ADMObject {
    /// `audioObjectID` attribute (e.g. `"AO_1001"`).
    pub audio_object_id: String,
    /// `audioObjectName` attribute.
    pub audio_object_name: String,
    /// Start time in milliseconds.
    pub start_time_ms: f32,
    /// Duration in milliseconds.
    pub duration_ms: f32,
    /// Cartesian position.
    pub position: Cartesian3D,
}

// ─── SpeakerLayout helpers ───────────────────────────────────────────────────

impl SpeakerLayout {
    /// Total number of channels in this layout.
    pub fn num_channels(&self) -> usize {
        match self {
            Self::Stereo => 2,
            Self::FiveOnOne => 6,
            Self::SevenOnOne => 8,
            Self::SevenOnFourOnFour => 12,
            Self::NineOnOnePointSix => 16,
        }
    }

    /// Return a slice of (azimuth_deg, elevation_deg) for each channel.
    ///
    /// Azimuths follow the OxiMedia convention: 0=front, 90=left, 180=back, 270=right.
    /// Elevations: 0=horizontal, positive=above.
    fn channel_directions(&self) -> Vec<(f32, f32)> {
        match self {
            Self::Stereo => vec![
                (30.0, 0.0),  // L
                (330.0, 0.0), // R
            ],
            Self::FiveOnOne => vec![
                (30.0, 0.0),  // L
                (330.0, 0.0), // R
                (0.0, 0.0),   // C
                (0.0, -10.0), // LFE (not a real direction, kept low)
                (110.0, 0.0), // Ls
                (250.0, 0.0), // Rs
            ],
            Self::SevenOnOne => vec![
                (30.0, 0.0),  // L
                (330.0, 0.0), // R
                (0.0, 0.0),   // C
                (0.0, -10.0), // LFE
                (110.0, 0.0), // Ls
                (250.0, 0.0), // Rs
                (60.0, 0.0),  // Lss
                (300.0, 0.0), // Rss
            ],
            Self::SevenOnFourOnFour => {
                let mut v = Self::SevenOnOne.channel_directions();
                v.push((45.0, 35.0)); // Ltf
                v.push((315.0, 35.0)); // Rtf
                v.push((135.0, 35.0)); // Ltr
                v.push((225.0, 35.0)); // Rtr
                v
            }
            Self::NineOnOnePointSix => {
                let mut v = Self::SevenOnOne.channel_directions();
                v.push((60.0, 20.0)); // Lts
                v.push((300.0, 20.0)); // Rts
                                       // 6 height channels
                v.push((30.0, 45.0));
                v.push((330.0, 45.0));
                v.push((0.0, 45.0));
                v.push((110.0, 45.0));
                v.push((250.0, 45.0));
                v.push((180.0, 45.0));
                v
            }
        }
    }
}

// ─── Panning math ─────────────────────────────────────────────────────────────

/// Convert ADM Cartesian (x, y, z) to azimuth (deg, OxiMedia convention) and elevation (deg).
///
/// ADM: x = lateral (right = +1), y = depth (front = +1), z = vertical (up = +1).
/// OxiMedia az: 0 = front (+y), 90 = left (+x negative = -x), 270 = right.
fn adm_to_oximedia_angles(x: f32, y: f32, z: f32) -> (f32, f32) {
    // azimuth: atan2(-x, y) so that front (y=1,x=0) → 0°, left (y=0,x=-1) → 90°.
    let az_rad = (-x).atan2(y);
    let az_deg = az_rad.to_degrees().rem_euclid(360.0);

    // elevation: atan2(z, sqrt(x²+y²))
    let horiz = (x * x + y * y).sqrt();
    let el_deg = z.atan2(horiz).to_degrees();

    (az_deg, el_deg)
}

/// Compute the angular distance between two directions (degrees).
fn angular_distance(az1: f32, el1: f32, az2: f32, el2: f32) -> f32 {
    let az1r = az1.to_radians();
    let az2r = az2.to_radians();
    let el1r = el1.to_radians();
    let el2r = el2.to_radians();

    let d_az = (az1r - az2r) / 2.0;
    let d_el = (el1r - el2r) / 2.0;
    let a = d_el.sin().powi(2) + el1r.cos() * el2r.cos() * d_az.sin().powi(2);
    2.0 * a.sqrt().asin().to_degrees()
}

// ─── AudioObject ─────────────────────────────────────────────────────────────

impl AudioObject {
    /// Create a point-source object at position (x, y, z) with unit gain and zero spread.
    pub fn new(id: u32, x: f32, y: f32, z: f32) -> Self {
        Self {
            id,
            x: x.clamp(-1.0, 1.0),
            y: y.clamp(-1.0, 1.0),
            z: z.clamp(-1.0, 1.0),
            gain: 1.0,
            width: 0.0,
            height: 0.0,
            depth: 0.0,
            divergence: 0.0,
        }
    }

    /// Azimuth and elevation (degrees, OxiMedia convention) for this object's position.
    pub fn to_angles(&self) -> (f32, f32) {
        adm_to_oximedia_angles(self.x, self.y, self.z)
    }
}

// ─── ObjectRenderer ──────────────────────────────────────────────────────────

impl ObjectRenderer {
    /// Create an empty renderer targeting the given speaker layout.
    pub fn new(layout: SpeakerLayout) -> Self {
        Self {
            objects: HashMap::new(),
            output_layout: layout,
        }
    }

    /// Insert or replace an object.
    pub fn upsert(&mut self, obj: AudioObject) {
        self.objects.insert(obj.id, obj);
    }

    /// Remove an object by ID.
    pub fn remove(&mut self, id: u32) {
        self.objects.remove(&id);
    }

    /// Render all objects in the renderer to a bed of channel gains.
    ///
    /// Returns a `HashMap<u32, f32>` mapping channel index to the sum of
    /// contributions from all objects.
    pub fn render_all(&self) -> HashMap<u32, f32> {
        let objects: Vec<&AudioObject> = self.objects.values().collect();
        render_to_beds(&objects, &self.output_layout)
    }
}

/// Render a slice of objects to speaker-bed channels.
///
/// Each object is panned to the nearest speaker using inverse-distance-weighted
/// panning, with optional spread applied via [`spread_gain`].
///
/// Returns `HashMap<channel_index → accumulated_gain>`.
pub fn render_to_beds(objects: &[&AudioObject], layout: &SpeakerLayout) -> HashMap<u32, f32> {
    let directions = layout.channel_directions();
    let mut gains: HashMap<u32, f32> = (0..directions.len()).map(|i| (i as u32, 0.0)).collect();

    for obj in objects {
        let spread_map = spread_gain(obj, layout);
        for (ch, &g) in &spread_map {
            *gains.entry(*ch).or_insert(0.0) += g * obj.gain;
        }
    }

    gains
}

/// Compute the per-channel gain contributions for a single object.
///
/// # Algorithm
///
/// 1. Convert the object's (x, y, z) to azimuth + elevation.
/// 2. For each output channel, compute the angular distance to the object.
/// 3. Compute a panning weight using an inverse-square law relative to the
///    minimum angular distance, scaled by the spatial spread radius.
///    The effective Gaussian sigma is `max(spread_deg, min_dist + 1.0)` so that
///    a zero-spread point source is always routed to at least the nearest speaker.
/// 4. Blend with the divergence factor toward equal distribution across all channels.
/// 5. Normalise so the sum of squares equals 1.
///
/// Returns `HashMap<channel_index → gain>`.
pub fn spread_gain(object: &AudioObject, layout: &SpeakerLayout) -> HashMap<u32, f32> {
    let directions = layout.channel_directions();
    let n = directions.len();
    let (obj_az, obj_el) = object.to_angles();

    // Compute angular distances to all speakers.
    let distances: Vec<f32> = directions
        .iter()
        .map(|&(ch_az, ch_el)| angular_distance(obj_az, obj_el, ch_az, ch_el))
        .collect();

    // Spread radius: combination of object width and height, in degrees.
    // For a point source (width=height=0) use a sigma that gives full gain only
    // to the nearest speaker and decays gracefully to the second-nearest.
    let spread_deg = (object.width * 90.0 + object.height * 90.0) / 2.0;
    let min_dist = distances.iter().cloned().fold(f32::INFINITY, f32::min);
    // Sigma floor: at least (min_dist + 5°) so the nearest speaker always gets gain.
    let sigma = spread_deg.max(min_dist + 5.0).max(1.0);

    let mut raw: Vec<f32> = distances
        .iter()
        .map(|&dist| {
            // Directional component: Gaussian decay with effective sigma.
            let directional = (-0.5 * (dist / sigma).powi(2)).exp();
            // Divergence blends toward omnidirectional (uniform = 1/sqrt(n)).
            let omni = 1.0 / (n as f32).sqrt();
            let blended = (1.0 - object.divergence) * directional + object.divergence * omni;
            blended.max(0.0)
        })
        .collect();

    // Energy-normalise.
    let energy: f32 = raw.iter().map(|&g| g * g).sum();
    if energy > 1e-10 {
        let norm = energy.sqrt();
        for g in &mut raw {
            *g /= norm;
        }
    }

    raw.into_iter()
        .enumerate()
        .map(|(i, g)| (i as u32, g))
        .collect()
}

// ─── Distance attenuation models ─────────────────────────────────────────────

/// A piecewise linear segment for a custom distance attenuation curve.
///
/// Distance values must be provided in ascending order when building a
/// [`CustomAttenuationCurve`].
#[derive(Debug, Clone, Copy)]
pub struct AttenuationPoint {
    /// Distance in metres (must be > 0).
    pub distance_m: f32,
    /// Attenuation gain at this distance (linear, ≥ 0).
    pub gain: f32,
}

/// Distance attenuation model.
///
/// All models are applied *after* the per-object linear gain, returning a
/// non-negative multiplier that decreases with distance.
#[derive(Debug, Clone)]
pub enum DistanceAttenuationModel {
    /// Physically correct inverse-square law: `gain = ref_distance / d`.
    ///
    /// The signal amplitude falls by 6 dB for every doubling of distance.
    /// `ref_distance` is the distance at which gain = 1.0 (default 1.0 m).
    InverseSquare {
        /// Distance at which the gain is defined as 1.0 (metres).
        ref_distance: f32,
        /// Minimum distance floor to avoid division by zero / infinite gain.
        min_distance: f32,
        /// Maximum distance beyond which gain is clamped to zero.
        max_distance: f32,
    },
    /// Logarithmic (perceptual) rolloff: `gain = 1 - rolloff * log10(d / ref_distance)`.
    ///
    /// Gain is clamped to [0, 1] at the boundaries.  Provides a gentler fade
    /// than inverse-square, matching the perceived loudness of many indoor spaces.
    Logarithmic {
        /// Distance at which gain = 1.0.
        ref_distance: f32,
        /// Rolloff factor (positive); higher = steeper decay.
        rolloff: f32,
        /// Maximum distance at which gain reaches 0.
        max_distance: f32,
    },
    /// None: no distance attenuation (gain always 1.0).
    None,
    /// User-supplied piecewise-linear curve.
    Custom(CustomAttenuationCurve),
}

/// Piecewise-linear distance attenuation curve.
///
/// The curve is defined by a sorted list of `(distance_m, gain)` control
/// points.  Gain is linearly interpolated between adjacent points; it is
/// clamped to the first/last gain values outside the defined range.
#[derive(Debug, Clone)]
pub struct CustomAttenuationCurve {
    /// Control points, sorted by ascending distance.
    pub points: Vec<AttenuationPoint>,
}

impl CustomAttenuationCurve {
    /// Build a new custom curve from control points.
    ///
    /// # Panics
    /// Does not panic; returns an empty curve if `points` is empty.
    ///
    /// # Sorting
    /// The points are sorted by `distance_m` in ascending order.
    pub fn new(mut points: Vec<AttenuationPoint>) -> Self {
        points.sort_by(|a, b| {
            a.distance_m
                .partial_cmp(&b.distance_m)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Self { points }
    }

    /// Evaluate the gain at the given distance.
    pub fn gain_at(&self, distance_m: f32) -> f32 {
        if self.points.is_empty() {
            return 1.0;
        }
        if distance_m <= self.points[0].distance_m {
            return self.points[0].gain;
        }
        if distance_m >= self.points[self.points.len() - 1].distance_m {
            return self.points[self.points.len() - 1].gain;
        }
        // Binary search for the enclosing segment.
        let mut lo = 0_usize;
        let mut hi = self.points.len() - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if self.points[mid].distance_m <= distance_m {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let p0 = &self.points[lo];
        let p1 = &self.points[hi];
        let span = p1.distance_m - p0.distance_m;
        if span < 1e-10 {
            return p0.gain;
        }
        let t = (distance_m - p0.distance_m) / span;
        p0.gain + t * (p1.gain - p0.gain)
    }
}

impl DistanceAttenuationModel {
    /// Compute the attenuation gain for the given distance (metres).
    ///
    /// Returns a non-negative linear gain multiplier in [0, ∞).
    pub fn gain_at(&self, distance_m: f32) -> f32 {
        match self {
            Self::None => 1.0,
            Self::InverseSquare {
                ref_distance,
                min_distance,
                max_distance,
            } => {
                let d = distance_m.max(*min_distance);
                if d >= *max_distance {
                    return 0.0;
                }
                (ref_distance / d).max(0.0)
            }
            Self::Logarithmic {
                ref_distance,
                rolloff,
                max_distance,
            } => {
                if distance_m >= *max_distance {
                    return 0.0;
                }
                let ratio = (distance_m / ref_distance.max(1e-6)).max(1e-10);
                let gain = 1.0 - rolloff * ratio.log10();
                gain.clamp(0.0, 1.0)
            }
            Self::Custom(curve) => curve.gain_at(distance_m),
        }
    }
}

// ─── Doppler effect ───────────────────────────────────────────────────────────

/// Simulates the Doppler effect for a moving sound source.
///
/// The Doppler effect causes a frequency shift when the relative velocity
/// between source and listener is non-zero.  The observed frequency is:
///
/// ```text
/// f_observed = f_source × (c + v_listener) / (c + v_source)
/// ```
///
/// where `c` is the speed of sound, `v_listener` is the component of the
/// listener's velocity *toward* the source, and `v_source` is the component
/// of the source's velocity *away from* the listener.
///
/// In practice this is implemented as a time-varying sample-rate ratio applied
/// to a fixed-size delay buffer.  The output buffer is resampled to match the
/// nominal sample rate using linear interpolation.
#[derive(Debug, Clone)]
pub struct DopplerSimulator {
    /// Speed of sound in metres per second.
    pub speed_of_sound: f32,
    /// Audio sample rate in Hz.
    pub sample_rate: u32,
    /// Circular delay buffer.
    delay_buf: Vec<f32>,
    /// Current write position in the delay buffer.
    write_pos: usize,
    /// Current fractional read position (sub-sample precision).
    read_pos: f64,
    /// Previous playback rate (samples per output sample).
    prev_rate: f64,
}

impl DopplerSimulator {
    /// Create a new Doppler simulator.
    ///
    /// # Parameters
    /// - `sample_rate`: audio sample rate (Hz).
    /// - `speed_of_sound`: speed of sound (m/s; default 343.0).
    /// - `max_delay_m`: maximum source-listener distance (metres); determines
    ///   the delay buffer size.
    pub fn new(sample_rate: u32, speed_of_sound: f32, max_delay_m: f32) -> Self {
        let sr = sample_rate as f32;
        let max_delay = ((max_delay_m / speed_of_sound) * sr).ceil() as usize + 2;
        let buf_size = max_delay.max(4);
        Self {
            speed_of_sound,
            sample_rate,
            delay_buf: vec![0.0_f32; buf_size],
            write_pos: 0,
            read_pos: 0.0,
            prev_rate: 1.0,
        }
    }

    /// Process one block of input samples, applying Doppler pitch shift.
    ///
    /// # Parameters
    /// - `input`: input audio samples.
    /// - `source_velocity_ms`: radial velocity of the source **toward** the
    ///   listener (positive = approaching, negative = receding).
    /// - `listener_velocity_ms`: radial velocity of the listener **toward** the
    ///   source (positive = approaching).
    ///
    /// # Returns
    /// Output buffer of the same length as `input`.  The pitch is shifted
    /// according to the Doppler formula, with the read position updated
    /// continuously to model smooth frequency glides.
    pub fn process(
        &mut self,
        input: &[f32],
        source_velocity_ms: f32,
        listener_velocity_ms: f32,
    ) -> Vec<f32> {
        let c = self.speed_of_sound;
        // Clamp velocities below the speed of sound.
        let vs = source_velocity_ms.clamp(-(c - 1.0), c - 1.0);
        let vl = listener_velocity_ms.clamp(-(c - 1.0), c - 1.0);

        // Doppler ratio: playback speed relative to the nominal sample rate.
        // f_obs / f_src = (c + v_listener) / (c - v_source)
        // where v_source is the component *away* from the listener.
        let denom = (c - vs).max(1.0); // avoid div-by-zero
        let rate = (c + vl) as f64 / denom as f64;
        let rate = rate.clamp(0.1, 10.0); // safety bounds

        let n = self.delay_buf.len();
        let mut output = Vec::with_capacity(input.len());

        for &sample in input {
            // Write input into the delay buffer.
            self.delay_buf[self.write_pos] = sample;
            self.write_pos = (self.write_pos + 1) % n;

            // Read from the buffer at the Doppler-shifted position.
            let read_int = self.read_pos as usize % n;
            let frac = self.read_pos - self.read_pos.floor();

            let s0 = self.delay_buf[read_int];
            let s1 = self.delay_buf[(read_int + 1) % n];
            let interp = s0 + frac as f32 * (s1 - s0);

            output.push(interp);

            // Advance read position by the Doppler rate.
            self.read_pos = (self.read_pos + rate) % n as f64;
        }

        self.prev_rate = rate;
        output
    }

    /// Compute the frequency ratio (observed / emitted) for the given velocities.
    ///
    /// This is the classic Doppler formula — useful for computing the pitch
    /// shift in cents for display or further processing.
    ///
    /// # Parameters
    /// - `source_velocity_ms`: radial velocity of source toward listener (m/s).
    /// - `listener_velocity_ms`: radial velocity of listener toward source (m/s).
    ///
    /// # Returns
    /// Frequency ratio f_obs / f_source.
    pub fn frequency_ratio(&self, source_velocity_ms: f32, listener_velocity_ms: f32) -> f32 {
        let c = self.speed_of_sound;
        let vs = source_velocity_ms.clamp(-(c - 1.0), c - 1.0);
        let vl = listener_velocity_ms.clamp(-(c - 1.0), c - 1.0);
        let denom = (c - vs).max(1.0);
        (c + vl) / denom
    }

    /// Reset the delay buffer and playback position.
    pub fn reset(&mut self) {
        self.delay_buf.fill(0.0);
        self.write_pos = 0;
        self.read_pos = 0.0;
        self.prev_rate = 1.0;
    }

    /// Return the last computed Doppler rate (samples per output sample).
    pub fn last_rate(&self) -> f64 {
        self.prev_rate
    }
}

// ─── ADM XML attribute parser ─────────────────────────────────────────────────

/// Minimal XML attribute value extractor.
///
/// Scans `xml_str` for `key="value"` or `key='value'` and returns the first
/// match, or `None` if not found.
fn find_xml_attr<'a>(xml_str: &'a str, key: &str) -> Option<&'a str> {
    // Build the search pattern: `key="` or `key='`.
    let pattern_double = format!("{}=\"", key);
    let pattern_single = format!("{}='", key);

    if let Some(start) = xml_str.find(&pattern_double) {
        let after = &xml_str[start + pattern_double.len()..];
        let end = after.find('"')?;
        return Some(&after[..end]);
    }
    if let Some(start) = xml_str.find(&pattern_single) {
        let after = &xml_str[start + pattern_single.len()..];
        let end = after.find('\'')?;
        return Some(&after[..end]);
    }
    None
}

/// Parse an `ADMObject` from a minimal XML fragment.
///
/// The parser looks for the following attributes (case-sensitive, in any order):
/// - `audioObjectID`
/// - `audioObjectName`
/// - `startTime` (milliseconds, parsed as float)
/// - `duration` (milliseconds, parsed as float)
/// - `X`, `Y`, `Z` (ADM Cartesian, clamped to [-1, 1])
///
/// Returns an error if `audioObjectID` or any of `X`, `Y`, `Z` are missing or
/// cannot be parsed.
pub fn parse_adm_object(xml_str: &str) -> Result<ADMObject, SpatialError> {
    let audio_object_id = find_xml_attr(xml_str, "audioObjectID")
        .ok_or_else(|| SpatialError::ParseError("missing audioObjectID attribute".into()))?
        .to_owned();

    let audio_object_name = find_xml_attr(xml_str, "audioObjectName")
        .unwrap_or("")
        .to_owned();

    let start_time_ms = find_xml_attr(xml_str, "startTime")
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.0);

    let duration_ms = find_xml_attr(xml_str, "duration")
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.0);

    let x_str = find_xml_attr(xml_str, "X")
        .ok_or_else(|| SpatialError::ParseError("missing X attribute".into()))?;
    let y_str = find_xml_attr(xml_str, "Y")
        .ok_or_else(|| SpatialError::ParseError("missing Y attribute".into()))?;
    let z_str = find_xml_attr(xml_str, "Z")
        .ok_or_else(|| SpatialError::ParseError("missing Z attribute".into()))?;

    let x = x_str
        .parse::<f32>()
        .map_err(|e| SpatialError::ParseError(format!("X parse error: {e}")))?
        .clamp(-1.0, 1.0);
    let y = y_str
        .parse::<f32>()
        .map_err(|e| SpatialError::ParseError(format!("Y parse error: {e}")))?
        .clamp(-1.0, 1.0);
    let z = z_str
        .parse::<f32>()
        .map_err(|e| SpatialError::ParseError(format!("Z parse error: {e}")))?
        .clamp(-1.0, 1.0);

    Ok(ADMObject {
        audio_object_id,
        audio_object_name,
        start_time_ms,
        duration_ms,
        position: Cartesian3D { x, y, z },
    })
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── SpeakerLayout ────────────────────────────────────────────────────────

    #[test]
    fn test_stereo_num_channels() {
        assert_eq!(SpeakerLayout::Stereo.num_channels(), 2);
    }

    #[test]
    fn test_five_one_num_channels() {
        assert_eq!(SpeakerLayout::FiveOnOne.num_channels(), 6);
    }

    #[test]
    fn test_seven_one_num_channels() {
        assert_eq!(SpeakerLayout::SevenOnOne.num_channels(), 8);
    }

    #[test]
    fn test_seven_four_four_num_channels() {
        assert_eq!(SpeakerLayout::SevenOnFourOnFour.num_channels(), 12);
    }

    #[test]
    fn test_nine_one_six_num_channels() {
        assert_eq!(SpeakerLayout::NineOnOnePointSix.num_channels(), 16);
    }

    #[test]
    fn test_channel_directions_length_matches_num_channels() {
        for layout in [
            SpeakerLayout::Stereo,
            SpeakerLayout::FiveOnOne,
            SpeakerLayout::SevenOnOne,
            SpeakerLayout::SevenOnFourOnFour,
            SpeakerLayout::NineOnOnePointSix,
        ] {
            let dirs = layout.channel_directions();
            assert_eq!(
                dirs.len(),
                layout.num_channels(),
                "Layout {:?} direction count mismatch",
                layout
            );
        }
    }

    // ── AudioObject ──────────────────────────────────────────────────────────

    #[test]
    fn test_audio_object_new_clamps() {
        let obj = AudioObject::new(1, 2.0, -3.0, 0.5);
        assert_eq!(obj.x, 1.0);
        assert_eq!(obj.y, -1.0);
        assert_eq!(obj.z, 0.5);
    }

    #[test]
    fn test_audio_object_front_angles() {
        let obj = AudioObject::new(1, 0.0, 1.0, 0.0); // directly front
        let (az, el) = obj.to_angles();
        assert!(
            az.abs() < 5.0 || (az - 360.0).abs() < 5.0,
            "front az should be ~0°, got {az}"
        );
        assert!(el.abs() < 5.0, "front el should be ~0°, got {el}");
    }

    #[test]
    fn test_audio_object_left_angles() {
        let obj = AudioObject::new(1, -1.0, 0.0, 0.0); // hard left
        let (az, _el) = obj.to_angles();
        // In OxiMedia convention, left = 90°.
        assert!(
            (az - 90.0).abs() < 5.0,
            "Hard-left source should be ~90°, got {az}"
        );
    }

    // ── spread_gain ──────────────────────────────────────────────────────────

    #[test]
    fn test_spread_gain_returns_correct_channel_count() {
        let obj = AudioObject::new(1, 0.0, 1.0, 0.0);
        let gains = spread_gain(&obj, &SpeakerLayout::Stereo);
        assert_eq!(gains.len(), 2);
    }

    #[test]
    fn test_spread_gain_normalised() {
        let obj = AudioObject::new(1, 0.5, 0.5, 0.0);
        let gains = spread_gain(&obj, &SpeakerLayout::FiveOnOne);
        let energy: f32 = gains.values().map(|&g| g * g).sum();
        assert!(
            (energy - 1.0).abs() < 0.05,
            "Spread gains should be energy-normalised, energy={energy}"
        );
    }

    #[test]
    fn test_spread_gain_non_negative() {
        let obj = AudioObject::new(1, 0.2, 0.8, 0.1);
        let gains = spread_gain(&obj, &SpeakerLayout::SevenOnOne);
        for (&ch, &g) in &gains {
            assert!(
                g >= 0.0,
                "Channel {ch} gain should be non-negative, got {g}"
            );
        }
    }

    #[test]
    fn test_spread_gain_divergence_one_is_omnidirectional() {
        let mut obj = AudioObject::new(1, 1.0, 0.0, 0.0);
        obj.divergence = 1.0;
        let gains = spread_gain(&obj, &SpeakerLayout::Stereo);
        let g0 = gains[&0];
        let g1 = gains[&1];
        // With full divergence, both channels should be nearly equal.
        assert!(
            (g0 - g1).abs() < 0.15,
            "Full divergence should give equal gains: g0={g0}, g1={g1}"
        );
    }

    #[test]
    fn test_spread_gain_with_width() {
        let mut obj_narrow = AudioObject::new(1, 0.0, 1.0, 0.0);
        obj_narrow.width = 0.0;
        let mut obj_wide = AudioObject::new(2, 0.0, 1.0, 0.0);
        obj_wide.width = 1.0;

        let narrow = spread_gain(&obj_narrow, &SpeakerLayout::SevenOnOne);
        let wide = spread_gain(&obj_wide, &SpeakerLayout::SevenOnOne);

        // Wide source should have more channels receiving significant gain.
        let narrow_active: usize = narrow.values().filter(|&&g| g > 0.1).count();
        let wide_active: usize = wide.values().filter(|&&g| g > 0.1).count();
        assert!(
            wide_active >= narrow_active,
            "Wider source should activate more channels: narrow={narrow_active}, wide={wide_active}"
        );
    }

    // ── render_to_beds ───────────────────────────────────────────────────────

    #[test]
    fn test_render_to_beds_returns_all_channels() {
        let obj = AudioObject::new(1, 0.0, 1.0, 0.0);
        let result = render_to_beds(&[&obj], &SpeakerLayout::FiveOnOne);
        assert_eq!(result.len(), 6);
    }

    #[test]
    fn test_render_to_beds_multiple_objects() {
        let obj1 = AudioObject::new(1, -1.0, 0.0, 0.0);
        let obj2 = AudioObject::new(2, 1.0, 0.0, 0.0);
        let result = render_to_beds(&[&obj1, &obj2], &SpeakerLayout::Stereo);
        // Both channels should receive energy from at least one object.
        let total: f32 = result.values().sum();
        assert!(total > 0.0, "Total gain should be positive");
    }

    #[test]
    fn test_render_to_beds_gain_applied() {
        let mut obj = AudioObject::new(1, 0.0, 1.0, 0.0);
        obj.gain = 2.0;
        let result = render_to_beds(&[&obj], &SpeakerLayout::Stereo);
        let max_gain = result.values().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(max_gain > 0.0, "Gain should be applied, max={max_gain}");
    }

    // ── ObjectRenderer ───────────────────────────────────────────────────────

    #[test]
    fn test_object_renderer_upsert_remove() {
        let mut renderer = ObjectRenderer::new(SpeakerLayout::Stereo);
        renderer.upsert(AudioObject::new(42, 0.0, 1.0, 0.0));
        assert_eq!(renderer.objects.len(), 1);
        renderer.remove(42);
        assert_eq!(renderer.objects.len(), 0);
    }

    #[test]
    fn test_object_renderer_render_all() {
        let mut renderer = ObjectRenderer::new(SpeakerLayout::FiveOnOne);
        renderer.upsert(AudioObject::new(1, 0.0, 1.0, 0.0));
        renderer.upsert(AudioObject::new(2, -0.5, 0.5, 0.0));
        let result = renderer.render_all();
        assert_eq!(result.len(), 6);
    }

    // ── parse_adm_object ─────────────────────────────────────────────────────

    #[test]
    fn test_parse_adm_object_basic() {
        let xml = r#"<audioObject audioObjectID="AO_1001" audioObjectName="Violin" X="0.5" Y="1.0" Z="0.0" />"#;
        let adm = parse_adm_object(xml).expect("should parse");
        assert_eq!(adm.audio_object_id, "AO_1001");
        assert_eq!(adm.audio_object_name, "Violin");
        assert!((adm.position.x - 0.5).abs() < 1e-4);
        assert!((adm.position.y - 1.0).abs() < 1e-4);
        assert!((adm.position.z - 0.0).abs() < 1e-4);
    }

    #[test]
    fn test_parse_adm_object_single_quotes() {
        let xml = r#"<audioObject audioObjectID='AO_2' X='0.0' Y='0.5' Z='0.2' />"#;
        let adm = parse_adm_object(xml).expect("single-quoted attrs should parse");
        assert_eq!(adm.audio_object_id, "AO_2");
    }

    #[test]
    fn test_parse_adm_object_missing_id_returns_error() {
        let xml = r#"<audioObject X="0.0" Y="0.0" Z="0.0" />"#;
        assert!(
            parse_adm_object(xml).is_err(),
            "Missing ID should be an error"
        );
    }

    #[test]
    fn test_parse_adm_object_missing_x_returns_error() {
        let xml = r#"<audioObject audioObjectID="AO_1" Y="0.0" Z="0.0" />"#;
        assert!(parse_adm_object(xml).is_err());
    }

    #[test]
    fn test_parse_adm_object_clamps_to_unit_cube() {
        let xml = r#"<audioObject audioObjectID="AO_3" X="5.0" Y="-3.0" Z="2.0" />"#;
        let adm = parse_adm_object(xml).expect("should parse with clamped values");
        assert_eq!(adm.position.x, 1.0);
        assert_eq!(adm.position.y, -1.0);
        assert_eq!(adm.position.z, 1.0);
    }

    #[test]
    fn test_parse_adm_object_with_timing() {
        let xml = r#"<audioObject audioObjectID="AO_10" X="0.0" Y="1.0" Z="0.0" startTime="500" duration="3000" />"#;
        let adm = parse_adm_object(xml).expect("should parse");
        assert!((adm.start_time_ms - 500.0).abs() < 0.1);
        assert!((adm.duration_ms - 3000.0).abs() < 0.1);
    }

    #[test]
    fn test_parse_adm_object_bad_number_returns_error() {
        let xml = r#"<audioObject audioObjectID="AO_4" X="abc" Y="0.0" Z="0.0" />"#;
        assert!(
            parse_adm_object(xml).is_err(),
            "Non-numeric X should be an error"
        );
    }

    #[test]
    fn test_find_xml_attr_present() {
        let xml = r#"key="value""#;
        assert_eq!(find_xml_attr(xml, "key"), Some("value"));
    }

    #[test]
    fn test_find_xml_attr_absent() {
        let xml = r#"other="val""#;
        assert!(find_xml_attr(xml, "key").is_none());
    }

    // ── DistanceAttenuationModel ──────────────────────────────────────────────

    fn assert_near(a: f32, b: f32, tol: f32, label: &str) {
        assert!(
            (a - b).abs() < tol,
            "{label}: expected ≈ {b}, got {a} (tol {tol})"
        );
    }

    #[test]
    fn test_distance_attenuation_none_always_one() {
        let model = DistanceAttenuationModel::None;
        for d in [0.1_f32, 1.0, 5.0, 100.0] {
            assert_near(model.gain_at(d), 1.0, 1e-6, "None model should return 1.0");
        }
    }

    #[test]
    fn test_distance_attenuation_inverse_square_at_ref_distance() {
        let model = DistanceAttenuationModel::InverseSquare {
            ref_distance: 1.0,
            min_distance: 0.1,
            max_distance: 1000.0,
        };
        // At ref_distance, gain should be 1.0.
        assert_near(model.gain_at(1.0), 1.0, 1e-5, "gain at ref distance");
    }

    #[test]
    fn test_distance_attenuation_inverse_square_halves_at_double_distance() {
        let model = DistanceAttenuationModel::InverseSquare {
            ref_distance: 1.0,
            min_distance: 0.01,
            max_distance: 1000.0,
        };
        let g1 = model.gain_at(2.0);
        let g2 = model.gain_at(4.0);
        // Gain should halve every time distance doubles.
        assert_near(
            g1 / g2,
            2.0,
            0.01,
            "inverse square: doubling distance halves gain",
        );
    }

    #[test]
    fn test_distance_attenuation_inverse_square_beyond_max_is_zero() {
        let model = DistanceAttenuationModel::InverseSquare {
            ref_distance: 1.0,
            min_distance: 0.1,
            max_distance: 10.0,
        };
        assert_near(
            model.gain_at(10.0),
            0.0,
            1e-6,
            "gain beyond max_distance should be 0",
        );
        assert_near(
            model.gain_at(50.0),
            0.0,
            1e-6,
            "gain way beyond max should be 0",
        );
    }

    #[test]
    fn test_distance_attenuation_logarithmic_at_ref_is_one() {
        let model = DistanceAttenuationModel::Logarithmic {
            ref_distance: 1.0,
            rolloff: 1.0,
            max_distance: 1000.0,
        };
        // At ref_distance, log10(1) = 0, so gain = 1.0.
        assert_near(model.gain_at(1.0), 1.0, 1e-5, "log gain at ref distance");
    }

    #[test]
    fn test_distance_attenuation_logarithmic_decreases_with_distance() {
        let model = DistanceAttenuationModel::Logarithmic {
            ref_distance: 1.0,
            rolloff: 1.0,
            max_distance: 1000.0,
        };
        let g1 = model.gain_at(2.0);
        let g2 = model.gain_at(10.0);
        assert!(
            g1 > g2,
            "Logarithmic gain should decrease with distance: g(2m)={g1}, g(10m)={g2}"
        );
    }

    #[test]
    fn test_distance_attenuation_logarithmic_clamped_to_zero_at_max() {
        let model = DistanceAttenuationModel::Logarithmic {
            ref_distance: 1.0,
            rolloff: 1.0,
            max_distance: 100.0,
        };
        assert_near(model.gain_at(100.0), 0.0, 1e-5, "log gain at max_distance");
    }

    // ── CustomAttenuationCurve ────────────────────────────────────────────────

    #[test]
    fn test_custom_curve_interpolation() {
        let curve = CustomAttenuationCurve::new(vec![
            AttenuationPoint {
                distance_m: 0.0,
                gain: 1.0,
            },
            AttenuationPoint {
                distance_m: 10.0,
                gain: 0.0,
            },
        ]);
        // At midpoint, gain should be ~0.5.
        let g = DistanceAttenuationModel::Custom(curve).gain_at(5.0);
        assert_near(g, 0.5, 0.01, "custom curve midpoint gain");
    }

    #[test]
    fn test_custom_curve_clamped_below_first_point() {
        let curve = CustomAttenuationCurve::new(vec![
            AttenuationPoint {
                distance_m: 1.0,
                gain: 0.8,
            },
            AttenuationPoint {
                distance_m: 5.0,
                gain: 0.2,
            },
        ]);
        // Below the first point, should use first gain.
        let g = DistanceAttenuationModel::Custom(curve).gain_at(0.0);
        assert_near(g, 0.8, 1e-5, "gain below first point should be first gain");
    }

    #[test]
    fn test_custom_curve_clamped_beyond_last_point() {
        let curve = CustomAttenuationCurve::new(vec![
            AttenuationPoint {
                distance_m: 1.0,
                gain: 1.0,
            },
            AttenuationPoint {
                distance_m: 10.0,
                gain: 0.1,
            },
        ]);
        let g = DistanceAttenuationModel::Custom(curve).gain_at(1000.0);
        assert_near(g, 0.1, 1e-5, "gain beyond last point should be last gain");
    }

    #[test]
    fn test_custom_curve_unsorted_points_get_sorted() {
        // Points provided in reverse order; should still sort correctly.
        let curve = CustomAttenuationCurve::new(vec![
            AttenuationPoint {
                distance_m: 10.0,
                gain: 0.0,
            },
            AttenuationPoint {
                distance_m: 0.0,
                gain: 1.0,
            },
        ]);
        let g_mid = DistanceAttenuationModel::Custom(curve).gain_at(5.0);
        assert_near(g_mid, 0.5, 0.01, "auto-sorted curve midpoint gain");
    }

    // ── DopplerSimulator ──────────────────────────────────────────────────────

    #[test]
    fn test_doppler_frequency_ratio_stationary() {
        let sim = DopplerSimulator::new(48_000, 343.0, 100.0);
        let ratio = sim.frequency_ratio(0.0, 0.0);
        assert_near(ratio, 1.0, 1e-5, "stationary source should have ratio 1.0");
    }

    #[test]
    fn test_doppler_frequency_ratio_approaching_source() {
        let sim = DopplerSimulator::new(48_000, 343.0, 100.0);
        let ratio = sim.frequency_ratio(30.0, 0.0); // source approaching at 30 m/s
        assert!(
            ratio > 1.0,
            "Approaching source should raise frequency, ratio={ratio}"
        );
    }

    #[test]
    fn test_doppler_frequency_ratio_receding_source() {
        let sim = DopplerSimulator::new(48_000, 343.0, 100.0);
        let ratio = sim.frequency_ratio(-30.0, 0.0); // source receding at 30 m/s
        assert!(
            ratio < 1.0,
            "Receding source should lower frequency, ratio={ratio}"
        );
    }

    #[test]
    fn test_doppler_process_output_length_matches_input() {
        let mut sim = DopplerSimulator::new(48_000, 343.0, 100.0);
        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01).sin()).collect();
        let output = sim.process(&input, 0.0, 0.0);
        assert_eq!(
            output.len(),
            input.len(),
            "Output length should match input"
        );
    }

    #[test]
    fn test_doppler_process_output_finite() {
        let mut sim = DopplerSimulator::new(48_000, 343.0, 100.0);
        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();
        let output = sim.process(&input, 10.0, 5.0);
        for (i, &s) in output.iter().enumerate() {
            assert!(
                s.is_finite(),
                "Doppler output[{i}] should be finite, got {s}"
            );
        }
    }

    #[test]
    fn test_doppler_reset_silences_delay_buffer() {
        let mut sim = DopplerSimulator::new(48_000, 343.0, 100.0);
        let impulse: Vec<f32> = (0..512).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
        let _ = sim.process(&impulse, 0.0, 0.0);
        sim.reset();
        let silence: Vec<f32> = vec![0.0; 64];
        let out = sim.process(&silence, 0.0, 0.0);
        let energy: f32 = out.iter().map(|x| x * x).sum();
        assert_near(energy, 0.0, 1e-5, "Reset Doppler should be silent");
    }

    #[test]
    fn test_doppler_last_rate_stationary_is_one() {
        let mut sim = DopplerSimulator::new(48_000, 343.0, 100.0);
        let _ = sim.process(&[0.0_f32; 64], 0.0, 0.0);
        assert_near(
            sim.last_rate() as f32,
            1.0,
            1e-5,
            "stationary last_rate should be 1.0",
        );
    }
}
