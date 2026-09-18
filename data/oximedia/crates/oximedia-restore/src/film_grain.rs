//! Film grain simulation and analysis for vintage restoration and cinematic effects.

#![allow(dead_code)]

/// A description of a grain layer's characteristics.
#[derive(Debug, Clone)]
pub struct GrainProfile {
    /// Mean grain size in pixels (smaller = finer).
    pub grain_size: f32,
    /// Intensity scale factor in the range `[0.0, 1.0]`.
    pub intensity: f32,
    /// Whether the grain pattern is monochromatic (luminance only).
    pub monochromatic: bool,
    /// Grain correlation radius in pixels (spatial extent of each grain clump).
    pub correlation_radius: f32,
    /// Film stock name or description (informational).
    pub stock_name: String,
}

impl GrainProfile {
    /// Create a new grain profile.
    #[must_use]
    pub fn new(grain_size: f32, intensity: f32) -> Self {
        Self {
            grain_size,
            intensity: intensity.clamp(0.0, 1.0),
            monochromatic: false,
            correlation_radius: grain_size * 1.5,
            stock_name: String::new(),
        }
    }

    /// Return `true` if the grain is classified as fine (grain size ≤ 2.0 px).
    #[must_use]
    pub fn is_fine_grain(&self) -> bool {
        self.grain_size <= 2.0
    }

    /// Return `true` if the intensity is high (> 0.7).
    #[must_use]
    pub fn is_high_intensity(&self) -> bool {
        self.intensity > 0.7
    }

    /// Return the effective intensity after clamping.
    #[must_use]
    pub fn effective_intensity(&self) -> f32 {
        self.intensity.clamp(0.0, 1.0)
    }
}

impl Default for GrainProfile {
    fn default() -> Self {
        Self::new(1.5, 0.25)
    }
}

/// A single grain layer that can be composited over an image/frame.
#[derive(Debug, Clone)]
pub struct GrainLayer {
    /// Profile driving this layer's appearance.
    pub profile: GrainProfile,
    /// Pre-generated grain samples (luminance offsets).
    grain_samples: Vec<f32>,
}

impl GrainLayer {
    /// Create a grain layer with the given profile and a pre-seeded sample buffer.
    ///
    /// `seed` controls the pseudo-random pattern so results are reproducible.
    #[must_use]
    pub fn new(profile: GrainProfile, seed: u64) -> Self {
        let samples = Self::generate_samples(1024, &profile, seed);
        Self {
            profile,
            grain_samples: samples,
        }
    }

    /// Generate pseudo-random grain samples using a simple LCG.
    fn generate_samples(count: usize, profile: &GrainProfile, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_add(1);
        (0..count)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let norm = (state >> 33) as f32 / (u32::MAX as f32); // [0, 1]
                (norm * 2.0 - 1.0) * profile.intensity // [-intensity, +intensity]
            })
            .collect()
    }

    /// Apply this layer's grain to a slice of pixels at the given `intensity` scale.
    ///
    /// Pixel values are expected in `[-1.0, 1.0]` and are clamped to that range after adding grain.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn apply_intensity(&self, pixels: &[f32], intensity: f32) -> Vec<f32> {
        let scale = intensity.clamp(0.0, 1.0);
        pixels
            .iter()
            .enumerate()
            .map(|(i, &px)| {
                let grain = self.grain_samples[i % self.grain_samples.len()] * scale;
                (px + grain).clamp(-1.0, 1.0)
            })
            .collect()
    }

    /// Return the number of pre-generated grain samples.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.grain_samples.len()
    }
}

/// Adds film grain to frames/images according to a `GrainProfile`.
#[derive(Debug)]
pub struct FilmGrainSimulator {
    profile: GrainProfile,
    layers: Vec<GrainLayer>,
}

impl FilmGrainSimulator {
    /// Create a simulator with the given profile and a single grain layer.
    #[must_use]
    pub fn new(profile: GrainProfile) -> Self {
        let layer = GrainLayer::new(profile.clone(), 42);
        Self {
            profile,
            layers: vec![layer],
        }
    }

    /// Add grain to `pixels`, returning a new pixel buffer.
    ///
    /// Multiple layers are composited additively.
    #[must_use]
    pub fn add_grain(&self, pixels: &[f32]) -> Vec<f32> {
        let mut result = pixels.to_vec();
        for layer in &self.layers {
            result = layer.apply_intensity(&result, layer.profile.effective_intensity());
        }
        result
    }

    /// Add an extra grain layer with the given seed for layered effects.
    pub fn add_layer(&mut self, seed: u64) {
        self.layers
            .push(GrainLayer::new(self.profile.clone(), seed));
    }

    /// Return the number of active grain layers.
    #[must_use]
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }
}

/// Analyses a pixel buffer to estimate its grain level.
#[derive(Debug, Default)]
pub struct FilmGrainAnalyzer;

impl FilmGrainAnalyzer {
    /// Create a new analyzer.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Estimate the grain level of `pixels` as a value in `[0.0, 1.0]`.
    ///
    /// The estimate uses the mean absolute difference between adjacent pixels
    /// as a proxy for high-frequency noise caused by grain.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn estimate_grain_level(&self, pixels: &[f32]) -> f32 {
        if pixels.len() < 2 {
            return 0.0;
        }
        let sum: f32 = pixels.windows(2).map(|w| (w[1] - w[0]).abs()).sum();
        let mean_diff = sum / (pixels.len() - 1) as f32;
        // Heuristic: mean diff of 0.1 corresponds to moderate grain (0.5 level)
        (mean_diff / 0.2).clamp(0.0, 1.0)
    }

    /// Classify the estimated grain level as a description string.
    #[must_use]
    pub fn classify(&self, level: f32) -> &'static str {
        match level {
            l if l < 0.15 => "clean",
            l if l < 0.40 => "fine",
            l if l < 0.65 => "moderate",
            l if l < 0.85 => "heavy",
            _ => "extreme",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- GrainProfile ---

    #[test]
    fn test_grain_profile_is_fine_grain_true() {
        let p = GrainProfile::new(1.5, 0.2);
        assert!(p.is_fine_grain());
    }

    #[test]
    fn test_grain_profile_is_fine_grain_false() {
        let p = GrainProfile::new(3.0, 0.2);
        assert!(!p.is_fine_grain());
    }

    #[test]
    fn test_grain_profile_is_fine_grain_boundary() {
        let p = GrainProfile::new(2.0, 0.2);
        assert!(p.is_fine_grain()); // exactly 2.0 is fine
    }

    #[test]
    fn test_grain_profile_high_intensity() {
        let p = GrainProfile::new(1.0, 0.8);
        assert!(p.is_high_intensity());
    }

    #[test]
    fn test_grain_profile_intensity_clamped() {
        let p = GrainProfile::new(1.0, 1.5); // over-range
        assert_eq!(p.effective_intensity(), 1.0);
    }

    #[test]
    fn test_grain_profile_default() {
        let p = GrainProfile::default();
        assert!(p.is_fine_grain());
        assert!(!p.is_high_intensity());
    }

    // --- GrainLayer ---

    #[test]
    fn test_grain_layer_sample_count() {
        let layer = GrainLayer::new(GrainProfile::default(), 7);
        assert_eq!(layer.sample_count(), 1024);
    }

    #[test]
    fn test_grain_layer_apply_intensity_length() {
        let layer = GrainLayer::new(GrainProfile::default(), 13);
        let pixels = vec![0.0_f32; 100];
        let out = layer.apply_intensity(&pixels, 0.5);
        assert_eq!(out.len(), 100);
    }

    #[test]
    fn test_grain_layer_apply_intensity_zero() {
        let layer = GrainLayer::new(GrainProfile::default(), 1);
        let pixels = vec![0.5_f32; 50];
        let out = layer.apply_intensity(&pixels, 0.0);
        // zero intensity → no grain change
        assert!(out.iter().all(|&v| (v - 0.5).abs() < 1e-6));
    }

    #[test]
    fn test_grain_layer_output_clamped() {
        let profile = GrainProfile::new(1.0, 1.0);
        let layer = GrainLayer::new(profile, 99);
        let pixels = vec![1.0_f32; 200];
        let out = layer.apply_intensity(&pixels, 1.0);
        assert!(out.iter().all(|&v| v >= -1.0 && v <= 1.0));
    }

    // --- FilmGrainSimulator ---

    #[test]
    fn test_simulator_add_grain_length() {
        let sim = FilmGrainSimulator::new(GrainProfile::default());
        let pixels = vec![0.0_f32; 256];
        let out = sim.add_grain(&pixels);
        assert_eq!(out.len(), 256);
    }

    #[test]
    fn test_simulator_layer_count_default() {
        let sim = FilmGrainSimulator::new(GrainProfile::default());
        assert_eq!(sim.layer_count(), 1);
    }

    #[test]
    fn test_simulator_add_layer() {
        let mut sim = FilmGrainSimulator::new(GrainProfile::default());
        sim.add_layer(42);
        sim.add_layer(99);
        assert_eq!(sim.layer_count(), 3);
    }

    // --- FilmGrainAnalyzer ---

    #[test]
    fn test_analyzer_clean_signal() {
        let analyzer = FilmGrainAnalyzer::new();
        let pixels = vec![0.5_f32; 100];
        let level = analyzer.estimate_grain_level(&pixels);
        assert!(
            level < 0.1,
            "flat signal should have near-zero grain: {level}"
        );
    }

    #[test]
    fn test_analyzer_empty_slice() {
        let analyzer = FilmGrainAnalyzer::new();
        assert_eq!(analyzer.estimate_grain_level(&[]), 0.0);
    }

    #[test]
    fn test_analyzer_classify_clean() {
        let analyzer = FilmGrainAnalyzer::new();
        assert_eq!(analyzer.classify(0.05), "clean");
    }

    #[test]
    fn test_analyzer_classify_heavy() {
        let analyzer = FilmGrainAnalyzer::new();
        assert_eq!(analyzer.classify(0.75), "heavy");
    }

    #[test]
    fn test_analyzer_grain_level_clamped() {
        let analyzer = FilmGrainAnalyzer::new();
        // Alternating extreme values produce very high diffs
        let pixels: Vec<f32> = (0..100)
            .map(|i| if i % 2 == 0 { -1.0 } else { 1.0 })
            .collect();
        let level = analyzer.estimate_grain_level(&pixels);
        assert!(level <= 1.0);
    }
}
