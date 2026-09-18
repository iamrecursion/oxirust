//! Spatial audio rendering for `OxiMedia` effects.
//!
//! Provides 3-D sound positioning, HRTF configuration, and a stereo renderer
//! that pans and attenuates sources based on their spatial position.

#![allow(dead_code)]

use std::f32::consts::PI;

/// A 3-D position in Cartesian space (metres).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpatialPosition {
    /// Forward-backward axis (+X is forward).
    pub x: f32,
    /// Left-right axis (+Y is left).
    pub y: f32,
    /// Vertical axis (+Z is up).
    pub z: f32,
}

impl SpatialPosition {
    /// Create a new position.
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Origin (listener position).
    #[must_use]
    pub const fn origin() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    /// Euclidean distance from the origin.
    #[must_use]
    pub fn distance(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Azimuth angle in degrees: 0° = front, 90° = left, −90° = right.
    #[must_use]
    pub fn azimuth_deg(&self) -> f32 {
        self.y.atan2(self.x).to_degrees()
    }

    /// Elevation angle in degrees: 0° = horizontal, 90° = directly above.
    #[must_use]
    pub fn elevation_deg(&self) -> f32 {
        let horiz = (self.x * self.x + self.y * self.y).sqrt();
        self.z.atan2(horiz).to_degrees()
    }

    /// Normalise to unit sphere.
    #[must_use]
    pub fn normalised(&self) -> Self {
        let d = self.distance();
        if d < 1e-6 {
            *self
        } else {
            Self::new(self.x / d, self.y / d, self.z / d)
        }
    }
}

impl Default for SpatialPosition {
    fn default() -> Self {
        Self::origin()
    }
}

/// HRTF (Head-Related Transfer Function) configuration.
#[derive(Debug, Clone)]
pub struct HrtfConfig {
    /// Use binaural HRTF (true) or simple panning (false).
    pub binaural: bool,
    /// Head radius in metres (used for ITD computation).
    pub head_radius_m: f32,
    /// Speed of sound in m/s.
    pub speed_of_sound: f32,
    /// Reference distance for 0 dB (inverse-square law).
    pub reference_distance_m: f32,
    /// Rolloff exponent (1 = linear, 2 = inverse-square).
    pub rolloff_exponent: f32,
}

impl Default for HrtfConfig {
    fn default() -> Self {
        Self {
            binaural: true,
            head_radius_m: 0.0875,
            speed_of_sound: 343.0,
            reference_distance_m: 1.0,
            rolloff_exponent: 1.0,
        }
    }
}

impl HrtfConfig {
    /// Returns `true` when all parameters are physically plausible.
    #[must_use]
    pub fn is_binaural(&self) -> bool {
        self.binaural
    }

    /// Returns `true` when the configuration is valid.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.head_radius_m > 0.0
            && self.speed_of_sound > 0.0
            && self.reference_distance_m > 0.0
            && self.rolloff_exponent > 0.0
    }

    /// Interaural time difference (ITD) in seconds for a given azimuth.
    ///
    /// Uses the Woodworth formula: `ITD = r/c * (sin(θ) + θ)` for `|θ| ≤ π/2`.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn itd_seconds(&self, azimuth_deg: f32) -> f32 {
        let az = azimuth_deg.to_radians().clamp(-PI / 2.0, PI / 2.0);
        self.head_radius_m / self.speed_of_sound * (az.sin() + az)
    }

    /// Compute the gain attenuation for a source at a given distance.
    #[must_use]
    pub fn distance_gain(&self, distance_m: f32) -> f32 {
        let d = distance_m.max(self.reference_distance_m);
        (self.reference_distance_m / d).powf(self.rolloff_exponent)
    }
}

/// An audio source registered with the spatial renderer.
#[derive(Debug, Clone)]
pub struct SpatialSource {
    /// Source identifier.
    pub id: u32,
    /// 3-D position.
    pub position: SpatialPosition,
    /// Source gain (0.0–1.0+).
    pub gain: f32,
}

/// Stereo spatial audio renderer.
pub struct SpatialAudioRenderer {
    hrtf: HrtfConfig,
    sources: Vec<SpatialSource>,
    sample_rate: f32,
    next_id: u32,
}

impl SpatialAudioRenderer {
    /// Create a new renderer.
    #[must_use]
    pub fn new(hrtf: HrtfConfig, sample_rate: f32) -> Self {
        Self {
            hrtf,
            sources: Vec::new(),
            sample_rate: sample_rate.max(1.0),
            next_id: 0,
        }
    }

    /// Register a source at `position` and return its ID.
    pub fn position_source(&mut self, position: SpatialPosition, gain: f32) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.sources.push(SpatialSource {
            id,
            position,
            gain: gain.max(0.0),
        });
        id
    }

    /// Update the position of an existing source.  Returns `false` if not found.
    pub fn update_position(&mut self, id: u32, position: SpatialPosition) -> bool {
        if let Some(src) = self.sources.iter_mut().find(|s| s.id == id) {
            src.position = position;
            true
        } else {
            false
        }
    }

    /// Remove a source by ID.
    pub fn remove_source(&mut self, id: u32) {
        self.sources.retain(|s| s.id != id);
    }

    /// Number of registered sources.
    #[must_use]
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    /// Compute the stereo pan gain pair (left, right) for an azimuth in degrees.
    ///
    /// Uses sine-law panning.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn pan_gains(azimuth_deg: f32) -> (f32, f32) {
        // Map 90° left → (1,0), 0° centre → (√2/2, √2/2), -90° right → (0,1)
        let az = azimuth_deg.clamp(-90.0, 90.0).to_radians();
        let l = ((PI / 4.0) + az / 2.0).cos();
        let r = ((PI / 4.0) - az / 2.0).cos();
        (l, r)
    }

    /// Render all sources into a stereo output buffer pair.
    ///
    /// `mono_inputs` maps source ID to a mono sample buffer.
    /// `left` and `right` must have the same length.
    ///
    /// # Panics
    ///
    /// Panics if `left` and `right` do not have the same length.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn render(&self, mono_inputs: &[(u32, &[f32])], left: &mut [f32], right: &mut [f32]) {
        assert_eq!(left.len(), right.len(), "left/right length mismatch");

        // Zero output
        for (l, r) in left.iter_mut().zip(right.iter_mut()) {
            *l = 0.0;
            *r = 0.0;
        }

        for (id, input) in mono_inputs {
            let Some(src) = self.sources.iter().find(|s| s.id == *id) else {
                continue;
            };

            let dist = src.position.distance().max(1e-3);
            let dist_gain = self.hrtf.distance_gain(dist);
            let az = src.position.azimuth_deg();
            let (pan_l, pan_r) = Self::pan_gains(az);

            let itd_samples = (self.hrtf.itd_seconds(az).abs() * self.sample_rate) as usize;

            let len = left.len().min(input.len());
            for i in 0..len {
                let s = input[i] * src.gain * dist_gain;
                // Simplified: apply ITD only to the delayed channel
                if az >= 0.0 {
                    // Source is left: left arrives first
                    left[i] += s * pan_l;
                    let ri = i + itd_samples;
                    if ri < right.len() {
                        right[ri] += s * pan_r;
                    }
                } else {
                    // Source is right: right arrives first
                    right[i] += s * pan_r;
                    let li = i + itd_samples;
                    if li < left.len() {
                        left[li] += s * pan_l;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_origin_distance() {
        let p = SpatialPosition::origin();
        assert_eq!(p.distance(), 0.0);
    }

    #[test]
    fn test_position_distance() {
        let p = SpatialPosition::new(3.0, 4.0, 0.0);
        assert!((p.distance() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_azimuth_front() {
        let p = SpatialPosition::new(1.0, 0.0, 0.0);
        assert!(p.azimuth_deg().abs() < 1e-3);
    }

    #[test]
    fn test_azimuth_left() {
        let p = SpatialPosition::new(0.0, 1.0, 0.0);
        assert!((p.azimuth_deg() - 90.0).abs() < 1e-3);
    }

    #[test]
    fn test_elevation_horizontal() {
        let p = SpatialPosition::new(1.0, 0.0, 0.0);
        assert!(p.elevation_deg().abs() < 1e-3);
    }

    #[test]
    fn test_elevation_above() {
        let p = SpatialPosition::new(0.0, 0.0, 1.0);
        assert!((p.elevation_deg() - 90.0).abs() < 1e-3);
    }

    #[test]
    fn test_hrtf_config_is_valid() {
        assert!(HrtfConfig::default().is_valid());
    }

    #[test]
    fn test_hrtf_config_is_binaural() {
        let cfg = HrtfConfig::default();
        assert!(cfg.is_binaural());
    }

    #[test]
    fn test_hrtf_distance_gain_reference() {
        let cfg = HrtfConfig::default();
        let gain = cfg.distance_gain(cfg.reference_distance_m);
        assert!((gain - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_hrtf_distance_gain_attenuates() {
        let cfg = HrtfConfig::default();
        let gain_far = cfg.distance_gain(10.0);
        assert!(gain_far < 1.0);
    }

    #[test]
    fn test_renderer_position_source() {
        let mut renderer = SpatialAudioRenderer::new(HrtfConfig::default(), 48000.0);
        let id = renderer.position_source(SpatialPosition::new(1.0, 0.0, 0.0), 1.0);
        assert_eq!(id, 0);
        assert_eq!(renderer.source_count(), 1);
    }

    #[test]
    fn test_renderer_render_silent() {
        let mut renderer = SpatialAudioRenderer::new(HrtfConfig::default(), 48000.0);
        let id = renderer.position_source(SpatialPosition::new(1.0, 0.0, 0.0), 1.0);
        let input = vec![0.0_f32; 512];
        let mut left = vec![0.0_f32; 512];
        let mut right = vec![0.0_f32; 512];
        renderer.render(&[(id, &input)], &mut left, &mut right);
        for s in &left {
            assert!(s.abs() < 1e-6);
        }
    }

    #[test]
    fn test_renderer_update_position() {
        let mut renderer = SpatialAudioRenderer::new(HrtfConfig::default(), 48000.0);
        let id = renderer.position_source(SpatialPosition::new(1.0, 0.0, 0.0), 1.0);
        let ok = renderer.update_position(id, SpatialPosition::new(2.0, 1.0, 0.0));
        assert!(ok);
    }

    #[test]
    fn test_renderer_remove_source() {
        let mut renderer = SpatialAudioRenderer::new(HrtfConfig::default(), 48000.0);
        let id = renderer.position_source(SpatialPosition::new(1.0, 0.0, 0.0), 1.0);
        renderer.remove_source(id);
        assert_eq!(renderer.source_count(), 0);
    }
}
