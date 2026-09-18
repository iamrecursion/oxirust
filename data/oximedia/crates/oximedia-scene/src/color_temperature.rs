//! Scene color temperature estimation and analysis.
//!
//! Provides tools for classifying and analysing the color temperature of
//! video and image scenes for white-balance and grading workflows.

#![allow(dead_code)]

/// Common scene lighting conditions mapped to approximate color temperatures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SceneColorTemp {
    /// Clear blue sky or open shade (~10 000 K).
    ClearSky,
    /// Overcast / cloudy sky (~6500–7500 K).
    Cloudy,
    /// Direct shade outdoors (~7000 K).
    Shade,
    /// Midday daylight (~5500–6000 K).
    Daylight,
    /// Electronic flash (~5500 K).
    Flash,
    /// Fluorescent tube lighting (~4000–4500 K).
    Fluorescent,
    /// Tungsten / incandescent lamp (~2800–3200 K).
    Tungsten,
    /// Candlelight or very warm artificial light (~1800–2000 K).
    Candlelight,
}

impl SceneColorTemp {
    /// Returns the representative color temperature in Kelvin.
    #[must_use]
    pub fn kelvin(self) -> u32 {
        match self {
            Self::ClearSky => 10_000,
            Self::Cloudy => 6_500,
            Self::Shade => 7_000,
            Self::Daylight => 5_500,
            Self::Flash => 5_500,
            Self::Fluorescent => 4_200,
            Self::Tungsten => 3_000,
            Self::Candlelight => 1_900,
        }
    }

    /// Returns a human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::ClearSky => "Clear Sky",
            Self::Cloudy => "Cloudy",
            Self::Shade => "Shade",
            Self::Daylight => "Daylight",
            Self::Flash => "Flash",
            Self::Fluorescent => "Fluorescent",
            Self::Tungsten => "Tungsten",
            Self::Candlelight => "Candlelight",
        }
    }

    /// Returns the corresponding Rec.709 white balance tint as `(R_gain, G_gain, B_gain)`.
    ///
    /// Gains are normalised so that G = 1.0.
    #[must_use]
    pub fn wb_gains(self) -> (f32, f32, f32) {
        match self {
            Self::ClearSky => (0.75, 1.0, 1.45),
            Self::Cloudy => (0.90, 1.0, 1.20),
            Self::Shade => (0.88, 1.0, 1.22),
            Self::Daylight => (1.00, 1.0, 1.00),
            Self::Flash => (1.00, 1.0, 1.00),
            Self::Fluorescent => (1.05, 1.0, 0.85),
            Self::Tungsten => (1.35, 1.0, 0.60),
            Self::Candlelight => (1.60, 1.0, 0.45),
        }
    }

    /// Returns `true` if this is a daylight-spectrum source (≥ 5000 K).
    #[must_use]
    pub fn is_daylight(self) -> bool {
        self.kelvin() >= 5000
    }

    /// Returns the closest `SceneColorTemp` for a given Kelvin value.
    #[must_use]
    pub fn from_kelvin(k: u32) -> Self {
        let candidates = [
            Self::ClearSky,
            Self::Shade,
            Self::Cloudy,
            Self::Daylight,
            Self::Flash,
            Self::Fluorescent,
            Self::Tungsten,
            Self::Candlelight,
        ];
        candidates
            .iter()
            .min_by_key(|c| {
                let diff = c.kelvin() as i64 - k as i64;
                diff.unsigned_abs()
            })
            .copied()
            .unwrap_or(Self::Daylight)
    }
}

/// Analyses a scene to estimate its color temperature.
#[derive(Debug, Clone)]
pub struct ColorTempAnalyzer {
    /// Tolerance in Kelvin for mixed-lighting detection.
    pub mixed_lighting_tolerance_k: u32,
}

impl Default for ColorTempAnalyzer {
    fn default() -> Self {
        Self {
            mixed_lighting_tolerance_k: 1500,
        }
    }
}

impl ColorTempAnalyzer {
    /// Creates a new `ColorTempAnalyzer` with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Estimates the dominant color temperature from a normalised average RGB color.
    ///
    /// `avg_rgb` should be in the range `[0.0, 1.0]`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn estimate(&self, avg_rgb: (f32, f32, f32)) -> SceneColorTemp {
        let (r, _g, b) = avg_rgb;
        // Rough heuristic: high blue relative to red → cool; high red → warm
        let rb_ratio = if b > 0.001 { r / b } else { 10.0 };
        let estimated_k = if rb_ratio > 2.5 {
            1_900_u32
        } else if rb_ratio > 1.8 {
            3_000
        } else if rb_ratio > 1.2 {
            4_200
        } else if rb_ratio > 0.9 {
            5_500
        } else if rb_ratio > 0.75 {
            6_500
        } else {
            10_000
        };
        SceneColorTemp::from_kelvin(estimated_k)
    }

    /// Returns `true` when the two dominant color temperatures in a scene
    /// differ by more than the `mixed_lighting_tolerance_k`.
    #[must_use]
    pub fn is_mixed_lighting(&self, temps: &[u32]) -> bool {
        if temps.len() < 2 {
            return false;
        }
        let min_k = temps.iter().copied().min().unwrap_or(0);
        let max_k = temps.iter().copied().max().unwrap_or(0);
        (max_k - min_k) > self.mixed_lighting_tolerance_k
    }

    /// Returns the average Kelvin value across the supplied temperature samples.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn average_kelvin(temps: &[u32]) -> f64 {
        if temps.is_empty() {
            return 0.0;
        }
        temps.iter().copied().map(|k| k as f64).sum::<f64>() / temps.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_daylight_kelvin() {
        assert_eq!(SceneColorTemp::Daylight.kelvin(), 5500);
    }

    #[test]
    fn test_tungsten_kelvin() {
        assert_eq!(SceneColorTemp::Tungsten.kelvin(), 3000);
    }

    #[test]
    fn test_cool_sources_are_daylight() {
        assert!(SceneColorTemp::ClearSky.is_daylight());
        assert!(SceneColorTemp::Daylight.is_daylight());
        assert!(SceneColorTemp::Flash.is_daylight());
        assert!(SceneColorTemp::Cloudy.is_daylight());
    }

    #[test]
    fn test_warm_sources_not_daylight() {
        assert!(!SceneColorTemp::Tungsten.is_daylight());
        assert!(!SceneColorTemp::Candlelight.is_daylight());
        assert!(!SceneColorTemp::Fluorescent.is_daylight());
    }

    #[test]
    fn test_from_kelvin_tungsten() {
        let ct = SceneColorTemp::from_kelvin(3000);
        assert_eq!(ct, SceneColorTemp::Tungsten);
    }

    #[test]
    fn test_from_kelvin_daylight() {
        let ct = SceneColorTemp::from_kelvin(5500);
        assert!(ct == SceneColorTemp::Daylight || ct == SceneColorTemp::Flash);
    }

    #[test]
    fn test_labels_non_empty() {
        for ct in [
            SceneColorTemp::Daylight,
            SceneColorTemp::Tungsten,
            SceneColorTemp::Cloudy,
            SceneColorTemp::Fluorescent,
        ] {
            assert!(!ct.label().is_empty());
        }
    }

    #[test]
    fn test_wb_gains_g_is_one() {
        for ct in [
            SceneColorTemp::Daylight,
            SceneColorTemp::Tungsten,
            SceneColorTemp::ClearSky,
            SceneColorTemp::Fluorescent,
        ] {
            let (_, g, _) = ct.wb_gains();
            assert!((g - 1.0).abs() < 1e-6, "{} G gain != 1", ct.label());
        }
    }

    #[test]
    fn test_wb_gains_warm_has_high_r() {
        let (r, _, b) = SceneColorTemp::Tungsten.wb_gains();
        assert!(r > b, "Tungsten should have R > B");
    }

    #[test]
    fn test_wb_gains_cool_has_high_b() {
        let (r, _, b) = SceneColorTemp::ClearSky.wb_gains();
        assert!(b > r, "Clear sky should have B > R");
    }

    #[test]
    fn test_analyzer_estimate_warm() {
        let analyzer = ColorTempAnalyzer::new();
        // High red, low blue → warm
        let ct = analyzer.estimate((0.9, 0.6, 0.3));
        assert!(!ct.is_daylight());
    }

    #[test]
    fn test_analyzer_estimate_cool() {
        let analyzer = ColorTempAnalyzer::new();
        // Low red, high blue → cool
        let ct = analyzer.estimate((0.3, 0.6, 0.9));
        assert!(ct.is_daylight());
    }

    #[test]
    fn test_is_mixed_lighting_true() {
        let analyzer = ColorTempAnalyzer::new();
        assert!(analyzer.is_mixed_lighting(&[3000, 6500]));
    }

    #[test]
    fn test_is_mixed_lighting_false() {
        let analyzer = ColorTempAnalyzer::new();
        assert!(!analyzer.is_mixed_lighting(&[5000, 5500]));
    }

    #[test]
    fn test_is_mixed_lighting_single() {
        let analyzer = ColorTempAnalyzer::new();
        assert!(!analyzer.is_mixed_lighting(&[5500]));
    }

    #[test]
    fn test_average_kelvin() {
        let avg = ColorTempAnalyzer::average_kelvin(&[3000, 5000, 7000]);
        assert!((avg - 5000.0).abs() < 1e-6);
    }

    #[test]
    fn test_average_kelvin_empty() {
        assert_eq!(ColorTempAnalyzer::average_kelvin(&[]), 0.0);
    }
}
