#![allow(dead_code)]
//! Room acoustics simulation and analysis for audio post-production.
//!
//! Provides room impulse response modelling, RT60 estimation, early reflection
//! placement, and reverb tail design for mix environments and virtual spaces.

use std::f64::consts::PI;

/// Physical room size category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RoomSize {
    /// Small booth or vocal booth (< 20 m^3).
    Booth,
    /// Small room such as a bedroom studio (20-60 m^3).
    Small,
    /// Medium room such as a control room (60-200 m^3).
    Medium,
    /// Large room such as a scoring stage (200-800 m^3).
    Large,
    /// Very large space such as a cathedral (> 800 m^3).
    Hall,
}

impl RoomSize {
    /// Return a representative volume in cubic metres.
    #[must_use]
    pub fn typical_volume_m3(&self) -> f64 {
        match self {
            Self::Booth => 10.0,
            Self::Small => 40.0,
            Self::Medium => 120.0,
            Self::Large => 500.0,
            Self::Hall => 2000.0,
        }
    }
}

/// Surface material with its absorption coefficient at a given frequency band.
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceMaterial {
    /// Human-readable name.
    pub name: String,
    /// Absorption coefficient at 125 Hz.
    pub alpha_125: f64,
    /// Absorption coefficient at 500 Hz.
    pub alpha_500: f64,
    /// Absorption coefficient at 2000 Hz.
    pub alpha_2k: f64,
    /// Absorption coefficient at 4000 Hz.
    pub alpha_4k: f64,
}

impl SurfaceMaterial {
    /// Create a new surface material.
    #[must_use]
    pub fn new(name: &str, alpha_125: f64, alpha_500: f64, alpha_2k: f64, alpha_4k: f64) -> Self {
        Self {
            name: name.to_string(),
            alpha_125,
            alpha_500,
            alpha_2k,
            alpha_4k,
        }
    }

    /// Concrete surface preset.
    #[must_use]
    pub fn concrete() -> Self {
        Self::new("Concrete", 0.01, 0.02, 0.02, 0.03)
    }

    /// Carpet surface preset.
    #[must_use]
    pub fn carpet() -> Self {
        Self::new("Carpet", 0.08, 0.24, 0.57, 0.69)
    }

    /// Acoustic panel preset.
    #[must_use]
    pub fn acoustic_panel() -> Self {
        Self::new("Acoustic Panel", 0.25, 0.80, 0.95, 0.90)
    }

    /// Glass surface preset.
    #[must_use]
    pub fn glass() -> Self {
        Self::new("Glass", 0.18, 0.06, 0.04, 0.03)
    }

    /// Average absorption across all stored bands.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn average_alpha(&self) -> f64 {
        (self.alpha_125 + self.alpha_500 + self.alpha_2k + self.alpha_4k) / 4.0
    }
}

/// An early reflection event.
#[derive(Debug, Clone, PartialEq)]
pub struct EarlyReflection {
    /// Delay relative to direct sound in seconds.
    pub delay_s: f64,
    /// Amplitude relative to direct sound (0.0..1.0).
    pub amplitude: f64,
    /// Wall index that generated the reflection.
    pub surface_index: usize,
}

/// Parameters describing the late reverb tail.
#[derive(Debug, Clone, PartialEq)]
pub struct ReverbTail {
    /// RT60 in seconds.
    pub rt60_s: f64,
    /// Pre-delay in seconds.
    pub pre_delay_s: f64,
    /// High-frequency damping factor (0.0 = none, 1.0 = full).
    pub hf_damping: f64,
    /// Diffusion factor (0.0 = sparse, 1.0 = dense).
    pub diffusion: f64,
}

impl ReverbTail {
    /// Create a new reverb tail.
    #[must_use]
    pub fn new(rt60_s: f64, pre_delay_s: f64, hf_damping: f64, diffusion: f64) -> Self {
        Self {
            rt60_s,
            pre_delay_s,
            hf_damping: hf_damping.clamp(0.0, 1.0),
            diffusion: diffusion.clamp(0.0, 1.0),
        }
    }

    /// Energy remaining at a given time (exponential decay model).
    #[must_use]
    pub fn energy_at(&self, time_s: f64) -> f64 {
        if self.rt60_s <= 0.0 || time_s < self.pre_delay_s {
            return if time_s < self.pre_delay_s { 0.0 } else { 1.0 };
        }
        let t = time_s - self.pre_delay_s;
        // RT60 means -60 dB at rt60_s => factor = 10^(-3) at rt60_s
        let decay_rate = -6.908 / self.rt60_s; // ln(0.001)
        (decay_rate * t).exp()
    }

    /// Time for the tail to reach -60 dB (same as `rt60_s` + `pre_delay_s`).
    #[must_use]
    pub fn total_decay_time_s(&self) -> f64 {
        self.pre_delay_s + self.rt60_s
    }
}

/// Complete room acoustics model.
#[derive(Debug, Clone)]
pub struct RoomAcoustics {
    /// Room size category.
    pub size: RoomSize,
    /// Room volume in cubic metres.
    pub volume_m3: f64,
    /// Total surface area in square metres.
    pub surface_area_m2: f64,
    /// Materials applied to the six surfaces (floor, ceiling, 4 walls).
    pub surfaces: Vec<SurfaceMaterial>,
    /// Speed of sound in m/s.
    pub speed_of_sound: f64,
}

impl RoomAcoustics {
    /// Create a new room model from dimensions.
    #[must_use]
    pub fn from_dimensions(length: f64, width: f64, height: f64) -> Self {
        let volume = length * width * height;
        let area = 2.0 * (length * width + length * height + width * height);
        let size = match volume {
            v if v < 20.0 => RoomSize::Booth,
            v if v < 60.0 => RoomSize::Small,
            v if v < 200.0 => RoomSize::Medium,
            v if v < 800.0 => RoomSize::Large,
            _ => RoomSize::Hall,
        };
        Self {
            size,
            volume_m3: volume,
            surface_area_m2: area,
            surfaces: Vec::new(),
            speed_of_sound: 343.0,
        }
    }

    /// Create a room from a preset size with default concrete surfaces.
    #[must_use]
    pub fn from_preset(size: RoomSize) -> Self {
        let (l, w, h) = match size {
            RoomSize::Booth => (2.0, 2.0, 2.5),
            RoomSize::Small => (4.0, 3.5, 2.8),
            RoomSize::Medium => (8.0, 6.0, 3.0),
            RoomSize::Large => (20.0, 12.0, 5.0),
            RoomSize::Hall => (40.0, 25.0, 12.0),
        };
        let mut room = Self::from_dimensions(l, w, h);
        room.surfaces = vec![SurfaceMaterial::concrete(); 6];
        room
    }

    /// Add or replace a surface material.
    pub fn set_surface(&mut self, index: usize, material: SurfaceMaterial) {
        if index < self.surfaces.len() {
            self.surfaces[index] = material;
        } else {
            self.surfaces.push(material);
        }
    }

    /// Compute average absorption coefficient across all surfaces.
    #[must_use]
    pub fn average_absorption(&self) -> f64 {
        if self.surfaces.is_empty() {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        let n = self.surfaces.len() as f64;
        self.surfaces
            .iter()
            .map(SurfaceMaterial::average_alpha)
            .sum::<f64>()
            / n
    }

    /// Estimate RT60 using the Sabine equation: RT60 = 0.161 * V / A.
    ///
    /// Where A = sum of (surface_area_i * alpha_i). When individual surface areas
    /// are unknown we distribute total area equally across surfaces.
    #[must_use]
    pub fn rt60(&self) -> f64 {
        let alpha = self.average_absorption();
        if alpha <= 0.0 {
            return f64::INFINITY;
        }
        0.161 * self.volume_m3 / (self.surface_area_m2 * alpha)
    }

    /// Estimate RT60 using the Eyring equation (better for well-damped rooms).
    #[must_use]
    pub fn rt60_eyring(&self) -> f64 {
        let alpha = self.average_absorption();
        if alpha <= 0.0 || alpha >= 1.0 {
            return if alpha >= 1.0 { 0.0 } else { f64::INFINITY };
        }
        let ln_term = -(1.0 - alpha).ln();
        0.161 * self.volume_m3 / (self.surface_area_m2 * ln_term)
    }

    /// Build a reverb tail descriptor from the current room model.
    #[must_use]
    pub fn reverb_tail(&self) -> ReverbTail {
        let rt60 = self.rt60();
        // Pre-delay: time for first reflection from nearest wall approximation
        let min_dim = (self.volume_m3 / self.surface_area_m2).sqrt();
        let pre_delay = min_dim / self.speed_of_sound;
        // HF damping proportional to average absorption
        let hf_damping = self.average_absorption().clamp(0.0, 1.0);
        ReverbTail::new(rt60, pre_delay, hf_damping, 0.8)
    }

    /// Generate early reflections for image-source method (simplified: 6 first-order reflections).
    #[must_use]
    pub fn early_reflections(&self) -> Vec<EarlyReflection> {
        // Approximate room as a rectangular box; compute first-order image sources.
        // We use volume / area as a proxy for characteristic dimension.
        let dim = (self.volume_m3).cbrt();
        let half = dim / 2.0;
        let alpha_avg = self.average_absorption();
        let refl_coeff = (1.0 - alpha_avg).sqrt();

        (0..6)
            .map(|i| {
                // Stagger distances slightly per surface
                #[allow(clippy::cast_precision_loss)]
                let dist = half * (1.0 + 0.1 * (i as f64));
                let delay = 2.0 * dist / self.speed_of_sound;
                EarlyReflection {
                    delay_s: delay,
                    amplitude: refl_coeff / (1.0 + delay),
                    surface_index: i,
                }
            })
            .collect()
    }

    /// Estimate critical distance (distance where direct = reverberant energy).
    #[must_use]
    pub fn critical_distance(&self) -> f64 {
        let alpha = self.average_absorption();
        if alpha <= 0.0 {
            return 0.0;
        }
        let a_total = self.surface_area_m2 * alpha;
        0.057 * (self.volume_m3 / self.rt60()).sqrt() * (a_total / (16.0 * PI)).sqrt().max(0.001)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_room_size_typical_volume() {
        assert!((RoomSize::Booth.typical_volume_m3() - 10.0).abs() < f64::EPSILON);
        assert!(RoomSize::Hall.typical_volume_m3() > 1000.0);
    }

    #[test]
    fn test_surface_material_presets() {
        let concrete = SurfaceMaterial::concrete();
        assert!(concrete.average_alpha() < 0.1);

        let panel = SurfaceMaterial::acoustic_panel();
        assert!(panel.average_alpha() > 0.5);
    }

    #[test]
    fn test_surface_material_custom() {
        let mat = SurfaceMaterial::new("Custom", 0.5, 0.5, 0.5, 0.5);
        assert!((mat.average_alpha() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_reverb_tail_creation() {
        let tail = ReverbTail::new(1.5, 0.01, 0.3, 0.7);
        assert!((tail.rt60_s - 1.5).abs() < f64::EPSILON);
        assert!((tail.pre_delay_s - 0.01).abs() < f64::EPSILON);
    }

    #[test]
    fn test_reverb_tail_energy_decay() {
        let tail = ReverbTail::new(1.0, 0.0, 0.0, 0.5);
        let e_start = tail.energy_at(0.0);
        let e_end = tail.energy_at(1.0);
        assert!((e_start - 1.0).abs() < 0.01);
        // At RT60 the energy should be ~0.001
        assert!(e_end < 0.01);
    }

    #[test]
    fn test_reverb_tail_total_decay() {
        let tail = ReverbTail::new(2.0, 0.05, 0.5, 0.8);
        assert!((tail.total_decay_time_s() - 2.05).abs() < f64::EPSILON);
    }

    #[test]
    fn test_room_from_dimensions() {
        let room = RoomAcoustics::from_dimensions(5.0, 4.0, 3.0);
        assert!((room.volume_m3 - 60.0).abs() < f64::EPSILON);
        // 2*(20+15+12) = 94
        assert!((room.surface_area_m2 - 94.0).abs() < f64::EPSILON);
        assert_eq!(room.size, RoomSize::Medium);
    }

    #[test]
    fn test_room_from_preset() {
        let room = RoomAcoustics::from_preset(RoomSize::Large);
        assert!(room.volume_m3 > 100.0);
        assert_eq!(room.surfaces.len(), 6);
    }

    #[test]
    fn test_rt60_sabine() {
        let mut room = RoomAcoustics::from_dimensions(8.0, 6.0, 3.0);
        room.surfaces = vec![SurfaceMaterial::concrete(); 6];
        let rt60 = room.rt60();
        // Concrete has low absorption => long RT60
        assert!(rt60 > 1.0);
    }

    #[test]
    fn test_rt60_eyring() {
        let mut room = RoomAcoustics::from_dimensions(8.0, 6.0, 3.0);
        room.surfaces = vec![SurfaceMaterial::acoustic_panel(); 6];
        let eyring = room.rt60_eyring();
        // Panels have high absorption => short RT60
        assert!(eyring < 1.0);
        assert!(eyring > 0.0);
    }

    #[test]
    fn test_room_reverb_tail() {
        let room = RoomAcoustics::from_preset(RoomSize::Medium);
        let tail = room.reverb_tail();
        assert!(tail.rt60_s > 0.0);
        assert!(tail.pre_delay_s > 0.0);
    }

    #[test]
    fn test_early_reflections() {
        let room = RoomAcoustics::from_preset(RoomSize::Medium);
        let refls = room.early_reflections();
        assert_eq!(refls.len(), 6);
        for r in &refls {
            assert!(r.delay_s > 0.0);
            assert!(r.amplitude > 0.0);
        }
    }

    #[test]
    fn test_set_surface() {
        let mut room = RoomAcoustics::from_preset(RoomSize::Small);
        room.set_surface(0, SurfaceMaterial::carpet());
        assert_eq!(room.surfaces[0].name, "Carpet");
    }

    #[test]
    fn test_average_absorption_empty() {
        let room = RoomAcoustics::from_dimensions(5.0, 4.0, 3.0);
        assert!((room.average_absorption() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_critical_distance() {
        let room = RoomAcoustics::from_preset(RoomSize::Large);
        let cd = room.critical_distance();
        assert!(cd > 0.0);
    }

    #[test]
    fn test_glass_material() {
        let glass = SurfaceMaterial::glass();
        // Glass has moderate absorption at low freq but low at high
        assert!(glass.alpha_125 > glass.alpha_4k);
    }
}
