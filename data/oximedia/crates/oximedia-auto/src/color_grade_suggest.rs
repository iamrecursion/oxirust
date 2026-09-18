//! Automatic color grade suggestions based on content mood analysis.
//!
//! Detects the visual mood of footage from histogram data and suggests
//! appropriate LUT/grade presets.

#![allow(dead_code)]

/// A named look preset describing a color grade.
#[derive(Debug, Clone)]
pub struct LookPreset {
    /// Human-readable name of the preset.
    pub name: String,
    /// Filename of the associated LUT.
    pub lut_name: String,
    /// Saturation adjustment multiplier (1.0 = unchanged).
    pub saturation: f32,
    /// Contrast adjustment multiplier (1.0 = unchanged).
    pub contrast: f32,
    /// Warmth adjustment: positive = warmer, negative = cooler.
    pub warmth: f32,
}

impl LookPreset {
    /// Create a new look preset.
    pub fn new(
        name: impl Into<String>,
        lut_name: impl Into<String>,
        saturation: f32,
        contrast: f32,
        warmth: f32,
    ) -> Self {
        Self {
            name: name.into(),
            lut_name: lut_name.into(),
            saturation,
            contrast,
            warmth,
        }
    }

    /// Cinematic look: slightly desaturated, higher contrast, neutral-cool.
    pub fn cinematic() -> Self {
        Self::new("Cinematic", "cinematic.cube", 0.85, 1.2, -5.0)
    }

    /// Documentary look: natural, moderate contrast.
    pub fn documentary() -> Self {
        Self::new("Documentary", "documentary.cube", 1.0, 1.05, 2.0)
    }

    /// News look: clean, neutral, slightly cool.
    pub fn news() -> Self {
        Self::new("News", "news.cube", 0.95, 1.0, -3.0)
    }

    /// Vintage look: faded, warm, desaturated.
    pub fn vintage() -> Self {
        Self::new("Vintage", "vintage.cube", 0.7, 0.9, 15.0)
    }

    /// Vivid look: saturated, punchy.
    pub fn vivid() -> Self {
        Self::new("Vivid", "vivid.cube", 1.4, 1.15, 5.0)
    }

    /// Desaturated look: almost monochrome.
    pub fn desaturated() -> Self {
        Self::new("Desaturated", "desaturated.cube", 0.4, 1.1, 0.0)
    }
}

/// Detected mood of the content based on colour analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentMood {
    /// Dominant warm tones (red/orange/yellow).
    Warm,
    /// Dominant cool tones (blue/cyan/teal).
    Cool,
    /// Balanced colour temperature.
    Neutral,
    /// Low-key, dark, shadowy footage.
    Dark,
    /// High-key, bright, overexposed-style footage.
    Bright,
    /// High contrast between shadows and highlights.
    Dramatic,
}

impl ContentMood {
    /// Return the name of the recommended look preset for this mood.
    pub fn suggested_look(&self) -> &str {
        match self {
            ContentMood::Warm => "Vintage",
            ContentMood::Cool => "Cinematic",
            ContentMood::Neutral => "Documentary",
            ContentMood::Dark => "Cinematic",
            ContentMood::Bright => "Vivid",
            ContentMood::Dramatic => "Desaturated",
        }
    }
}

/// Analyses histograms to detect visual mood.
pub struct MoodAnalyzer;

impl MoodAnalyzer {
    /// Detect mood from per-channel histograms.
    ///
    /// Decision logic:
    /// - Warm if `r_mean > b_mean + 20`
    /// - Cool if `b_mean > r_mean + 20`
    /// - Dark if the overall mean is below 64
    /// - Bright if the overall mean is above 192
    /// - Dramatic if the standard deviation of luminance is high (>60)
    /// - Otherwise Neutral
    pub fn detect_from_histogram(
        r_hist: &[u32; 256],
        g_hist: &[u32; 256],
        b_hist: &[u32; 256],
    ) -> ContentMood {
        let r_mean = channel_mean(r_hist);
        let g_mean = channel_mean(g_hist);
        let b_mean = channel_mean(b_hist);

        let overall_mean = (r_mean + g_mean + b_mean) / 3.0;
        let luma_std = channel_std(g_hist, g_mean); // approximate with green channel

        if luma_std > 60.0 {
            return ContentMood::Dramatic;
        }

        if overall_mean < 64.0 {
            return ContentMood::Dark;
        }

        if overall_mean > 192.0 {
            return ContentMood::Bright;
        }

        if r_mean > b_mean + 20.0 {
            return ContentMood::Warm;
        }

        if b_mean > r_mean + 20.0 {
            return ContentMood::Cool;
        }

        ContentMood::Neutral
    }
}

/// Compute the mean pixel value from a histogram.
fn channel_mean(hist: &[u32; 256]) -> f64 {
    let total: u64 = hist.iter().map(|&v| v as u64).sum();
    if total == 0 {
        return 0.0;
    }
    let weighted: u64 = hist
        .iter()
        .enumerate()
        .map(|(i, &v)| i as u64 * v as u64)
        .sum();
    weighted as f64 / total as f64
}

/// Compute the standard deviation of pixel values from a histogram.
fn channel_std(hist: &[u32; 256], mean: f64) -> f64 {
    let total: u64 = hist.iter().map(|&v| v as u64).sum();
    if total == 0 {
        return 0.0;
    }
    let variance: f64 = hist
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let diff = i as f64 - mean;
            diff * diff * v as f64
        })
        .sum::<f64>()
        / total as f64;
    variance.sqrt()
}

/// Suggests color grades based on mood and content type.
pub struct AutoColorGrader;

impl AutoColorGrader {
    /// Return the top 3 suggested `LookPreset`s for the given mood and content type.
    pub fn suggest(mood: ContentMood, content_type: &str) -> Vec<LookPreset> {
        let all_presets = [
            LookPreset::cinematic(),
            LookPreset::documentary(),
            LookPreset::news(),
            LookPreset::vintage(),
            LookPreset::vivid(),
            LookPreset::desaturated(),
        ];

        // Score each preset based on mood and content type
        let content_lower = content_type.to_lowercase();

        let mut scored: Vec<(f32, LookPreset)> = all_presets
            .into_iter()
            .map(|preset| {
                let mut score = 0.0_f32;

                // Mood-based scoring
                score += match (mood, preset.name.as_str()) {
                    (ContentMood::Warm, "Vintage") => 1.0,
                    (ContentMood::Warm, "Vivid") => 0.5,
                    (ContentMood::Cool, "Cinematic") => 1.0,
                    (ContentMood::Cool, "News") => 0.5,
                    (ContentMood::Neutral, "Documentary") => 1.0,
                    (ContentMood::Neutral, "News") => 0.6,
                    (ContentMood::Dark, "Cinematic") => 1.0,
                    (ContentMood::Dark, "Desaturated") => 0.7,
                    (ContentMood::Bright, "Vivid") => 1.0,
                    (ContentMood::Bright, "Documentary") => 0.5,
                    (ContentMood::Dramatic, "Desaturated") => 1.0,
                    (ContentMood::Dramatic, "Cinematic") => 0.8,
                    _ => 0.1,
                };

                // Content type bonus
                score += match (content_lower.as_str(), preset.name.as_str()) {
                    ("news", "News") => 0.5,
                    ("documentary", "Documentary") => 0.5,
                    ("sport" | "sports", "Vivid") => 0.5,
                    ("social" | "social_media", "Vivid") => 0.4,
                    ("film" | "cinematic", "Cinematic") => 0.5,
                    _ => 0.0,
                };

                (score, preset)
            })
            .collect();

        // Sort descending by score
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        scored.into_iter().take(3).map(|(_, p)| p).collect()
    }
}

/// Checks the visual consistency of a list of applied grades.
pub struct GradeConsistencyChecker;

impl GradeConsistencyChecker {
    /// Compute a consistency score (0.0–1.0) across a set of grade presets.
    ///
    /// Score = 1.0 means all grades are identical; 0.0 means maximum divergence.
    /// Compares saturation, contrast, and warmth across all pairs.
    pub fn check_consistency(grades: &[LookPreset]) -> f32 {
        if grades.len() < 2 {
            return 1.0;
        }

        let n = grades.len() as f32;

        // Compute mean of each parameter
        let sat_mean: f32 = grades.iter().map(|g| g.saturation).sum::<f32>() / n;
        let con_mean: f32 = grades.iter().map(|g| g.contrast).sum::<f32>() / n;
        let wrm_mean: f32 = grades.iter().map(|g| g.warmth).sum::<f32>() / n;

        // Compute normalised standard deviations
        let sat_std = std_dev(grades.iter().map(|g| g.saturation), sat_mean);
        let con_std = std_dev(grades.iter().map(|g| g.contrast), con_mean);
        let wrm_std = std_dev(grades.iter().map(|g| g.warmth), wrm_mean);

        // Map divergence to a 0-1 consistency score (lower std = higher consistency)
        // Normalise: saturation/contrast range ~0-2, warmth range ~-30 to +30
        let sat_score = 1.0 - (sat_std / 1.0).min(1.0);
        let con_score = 1.0 - (con_std / 1.0).min(1.0);
        let wrm_score = 1.0 - (wrm_std / 30.0).min(1.0);

        (sat_score + con_score + wrm_score) / 3.0
    }
}

fn std_dev(values: impl Iterator<Item = f32>, mean: f32) -> f32 {
    let collected: Vec<f32> = values.collect();
    let n = collected.len() as f32;
    if n == 0.0 {
        return 0.0;
    }
    let variance = collected.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    variance.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_hist(value: u8, count: u32) -> [u32; 256] {
        let mut h = [0u32; 256];
        h[value as usize] = count;
        h
    }

    fn uniform_hist() -> [u32; 256] {
        [1u32; 256]
    }

    fn bright_hist() -> [u32; 256] {
        // Concentrates near pixel value 220
        let mut h = [0u32; 256];
        h[220] = 1000;
        h
    }

    fn dark_hist() -> [u32; 256] {
        let mut h = [0u32; 256];
        h[30] = 1000;
        h
    }

    #[test]
    fn test_look_preset_cinematic() {
        let p = LookPreset::cinematic();
        assert_eq!(p.name, "Cinematic");
        assert!(p.saturation < 1.0);
        assert!(p.contrast > 1.0);
    }

    #[test]
    fn test_look_preset_vivid() {
        let p = LookPreset::vivid();
        assert!(p.saturation > 1.0);
    }

    #[test]
    fn test_mood_suggested_look() {
        assert_eq!(ContentMood::Warm.suggested_look(), "Vintage");
        assert_eq!(ContentMood::Cool.suggested_look(), "Cinematic");
        assert_eq!(ContentMood::Dramatic.suggested_look(), "Desaturated");
    }

    #[test]
    fn test_detect_mood_warm() {
        let r = flat_hist(180, 1000);
        let g = flat_hist(130, 1000);
        let b = flat_hist(100, 1000); // r significantly > b
        let mood = MoodAnalyzer::detect_from_histogram(&r, &g, &b);
        assert_eq!(mood, ContentMood::Warm);
    }

    #[test]
    fn test_detect_mood_cool() {
        let r = flat_hist(100, 1000);
        let g = flat_hist(130, 1000);
        let b = flat_hist(180, 1000); // b significantly > r
        let mood = MoodAnalyzer::detect_from_histogram(&r, &g, &b);
        assert_eq!(mood, ContentMood::Cool);
    }

    #[test]
    fn test_detect_mood_bright() {
        let h = bright_hist();
        let mood = MoodAnalyzer::detect_from_histogram(&h, &h, &h);
        assert_eq!(mood, ContentMood::Bright);
    }

    #[test]
    fn test_detect_mood_dark() {
        let h = dark_hist();
        let mood = MoodAnalyzer::detect_from_histogram(&h, &h, &h);
        assert_eq!(mood, ContentMood::Dark);
    }

    #[test]
    fn test_detect_mood_neutral() {
        let _h = uniform_hist(); // standard deviation is non-trivial but mean is ~128
                                 // uniform has mean ~127.5, std ~73, so it will be Dramatic
                                 // Use a narrow mid-range hist for neutral
        let mut h2 = [0u32; 256];
        for i in 115..140 {
            h2[i] = 100;
        }
        let mood = MoodAnalyzer::detect_from_histogram(&h2, &h2, &h2);
        assert_eq!(mood, ContentMood::Neutral);
    }

    #[test]
    fn test_auto_color_grader_returns_three() {
        let presets = AutoColorGrader::suggest(ContentMood::Cool, "film");
        assert_eq!(presets.len(), 3);
    }

    #[test]
    fn test_auto_color_grader_warm_top_is_vintage() {
        let presets = AutoColorGrader::suggest(ContentMood::Warm, "generic");
        assert_eq!(presets[0].name, "Vintage");
    }

    #[test]
    fn test_consistency_identical_grades() {
        let grades = vec![LookPreset::cinematic(), LookPreset::cinematic()];
        let score = GradeConsistencyChecker::check_consistency(&grades);
        assert!(
            (score - 1.0).abs() < 1e-5,
            "Identical grades should be consistent"
        );
    }

    #[test]
    fn test_consistency_single_grade() {
        let grades = vec![LookPreset::vivid()];
        let score = GradeConsistencyChecker::check_consistency(&grades);
        assert!((score - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_consistency_diverse_grades() {
        let grades = vec![
            LookPreset::cinematic(),
            LookPreset::vivid(),
            LookPreset::vintage(),
        ];
        let score = GradeConsistencyChecker::check_consistency(&grades);
        // Diverse grades should have lower consistency
        assert!(score < 1.0);
        assert!(score >= 0.0);
    }
}
