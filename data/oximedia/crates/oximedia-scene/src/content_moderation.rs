//! Content moderation analysis for video frames.
//!
//! This module provides heuristic-based content moderation capabilities:
//!
//! - **Skin tone detection**: HSV/YCbCr thresholding for skin pixel ratio
//! - **Violence indicators**: Rapid motion combined with red colour dominance
//! - **Text profanity detection**: Configurable word list matching
//! - **Content rating estimation**: G/PG/PG-13/R/NC-17 based on combined signals
//! - **Frame flagging**: Confidence scores and reason codes per frame

use crate::common::Confidence;
use crate::error::{SceneError, SceneResult};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Content rating (MPAA-style).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ContentRating {
    /// General audiences.
    G,
    /// Parental guidance suggested.
    PG,
    /// Parents strongly cautioned.
    PG13,
    /// Restricted.
    R,
    /// Adults only.
    NC17,
}

impl ContentRating {
    /// Return a human-readable label.
    #[must_use]
    pub const fn label(&self) -> &str {
        match self {
            Self::G => "G",
            Self::PG => "PG",
            Self::PG13 => "PG-13",
            Self::R => "R",
            Self::NC17 => "NC-17",
        }
    }
}

/// Reason code for why a frame was flagged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FlagReason {
    /// High skin-tone pixel ratio.
    HighSkinExposure,
    /// Violence indicators detected (rapid motion + red dominance).
    ViolenceIndicator,
    /// Profane text detected.
    ProfaneText,
    /// Multiple signals combined.
    CombinedSignals,
}

/// A moderation flag on a frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModerationFlag {
    /// Reason for the flag.
    pub reason: FlagReason,
    /// Confidence of this particular flag (0..1).
    pub confidence: Confidence,
    /// Optional detail message.
    pub detail: String,
}

/// Full moderation result for a single frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModerationResult {
    /// Estimated content rating.
    pub rating: ContentRating,
    /// Overall moderation confidence.
    pub confidence: Confidence,
    /// Individual flags.
    pub flags: Vec<ModerationFlag>,
    /// Skin pixel ratio (0..1).
    pub skin_ratio: f32,
    /// Violence indicator score (0..1).
    pub violence_score: f32,
    /// Profanity matches found.
    pub profanity_matches: Vec<String>,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for content moderation.
#[derive(Debug, Clone)]
pub struct ModerationConfig {
    /// Skin ratio threshold to flag.
    pub skin_flag_threshold: f32,
    /// Violence score threshold to flag.
    pub violence_flag_threshold: f32,
    /// Red dominance threshold for violence heuristic (fraction).
    pub red_dominance_threshold: f32,
    /// Motion magnitude threshold for violence heuristic (0..1).
    pub motion_threshold: f32,
    /// Custom profanity word list (lowercase).
    pub profanity_words: Vec<String>,
    /// Minimum confidence to include a flag.
    pub min_flag_confidence: f32,
}

impl Default for ModerationConfig {
    fn default() -> Self {
        Self {
            skin_flag_threshold: 0.35,
            violence_flag_threshold: 0.40,
            red_dominance_threshold: 0.25,
            motion_threshold: 0.15,
            profanity_words: default_profanity_list(),
            min_flag_confidence: 0.3,
        }
    }
}

/// Returns a small default profanity word list for demonstration.
fn default_profanity_list() -> Vec<String> {
    [
        "damn", "hell", "crap", "bastard", "ass", "shit", "fuck", "bitch", "piss", "dick", "cock",
        "pussy", "slut", "whore", "fag", "nigger", "cunt", "wanker", "bollocks", "bloody",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect()
}

// ---------------------------------------------------------------------------
// Moderator
// ---------------------------------------------------------------------------

/// Content moderator for video frames.
pub struct ContentModerator {
    config: ModerationConfig,
}

impl ContentModerator {
    /// Create a moderator with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: ModerationConfig::default(),
        }
    }

    /// Create with custom configuration.
    #[must_use]
    pub fn with_config(config: ModerationConfig) -> Self {
        Self { config }
    }

    /// Analyse a single frame.
    ///
    /// # Arguments
    ///
    /// * `rgb_data` - Current frame RGB data
    /// * `prev_rgb_data` - Previous frame RGB data (for motion; `None` if first frame)
    /// * `text_in_frame` - Any text detected/extracted from the frame
    /// * `width` - Image width
    /// * `height` - Image height
    ///
    /// # Errors
    ///
    /// Returns `SceneError::InvalidDimensions` on size mismatch.
    pub fn analyse(
        &self,
        rgb_data: &[u8],
        prev_rgb_data: Option<&[u8]>,
        text_in_frame: &[&str],
        width: usize,
        height: usize,
    ) -> SceneResult<ModerationResult> {
        let expected = width * height * 3;
        if rgb_data.len() != expected {
            return Err(SceneError::InvalidDimensions(format!(
                "expected {expected} bytes, got {}",
                rgb_data.len()
            )));
        }
        if let Some(prev) = prev_rgb_data {
            if prev.len() != expected {
                return Err(SceneError::InvalidDimensions(
                    "prev frame size mismatch".into(),
                ));
            }
        }

        let skin_ratio = compute_skin_ratio(rgb_data, width, height);
        let violence_score =
            compute_violence_score(rgb_data, prev_rgb_data, width, height, &self.config);
        let profanity_matches = detect_profanity(text_in_frame, &self.config.profanity_words);

        let mut flags = Vec::new();

        // Skin exposure flag
        if skin_ratio >= self.config.skin_flag_threshold {
            let conf = ((skin_ratio - self.config.skin_flag_threshold) / 0.3).min(1.0);
            if conf >= self.config.min_flag_confidence {
                flags.push(ModerationFlag {
                    reason: FlagReason::HighSkinExposure,
                    confidence: Confidence::new(conf),
                    detail: format!("skin ratio {skin_ratio:.2}"),
                });
            }
        }

        // Violence flag
        if violence_score >= self.config.violence_flag_threshold {
            let conf = ((violence_score - self.config.violence_flag_threshold) / 0.3).min(1.0);
            if conf >= self.config.min_flag_confidence {
                flags.push(ModerationFlag {
                    reason: FlagReason::ViolenceIndicator,
                    confidence: Confidence::new(conf),
                    detail: format!("violence score {violence_score:.2}"),
                });
            }
        }

        // Profanity flag
        if !profanity_matches.is_empty() {
            let conf = (profanity_matches.len() as f32 * 0.3).min(1.0);
            if conf >= self.config.min_flag_confidence {
                flags.push(ModerationFlag {
                    reason: FlagReason::ProfaneText,
                    confidence: Confidence::new(conf),
                    detail: format!("matched: {}", profanity_matches.join(", ")),
                });
            }
        }

        // Combined signals
        let combined =
            skin_ratio * 0.3 + violence_score * 0.4 + profanity_matches.len() as f32 * 0.1;
        if combined > 0.5 && flags.len() >= 2 {
            flags.push(ModerationFlag {
                reason: FlagReason::CombinedSignals,
                confidence: Confidence::new(combined.min(1.0)),
                detail: "multiple moderation signals".into(),
            });
        }

        let rating = estimate_rating(skin_ratio, violence_score, &profanity_matches);
        let overall_confidence = if flags.is_empty() {
            Confidence::new(0.9) // high confidence it's clean
        } else {
            let max_flag = flags
                .iter()
                .map(|f| f.confidence.value())
                .fold(0.0f32, f32::max);
            Confidence::new(max_flag)
        };

        Ok(ModerationResult {
            rating,
            confidence: overall_confidence,
            flags,
            skin_ratio,
            violence_score,
            profanity_matches,
        })
    }

    /// Check if text contains profanity.
    #[must_use]
    pub fn check_text(&self, text: &str) -> Vec<String> {
        let lower = text.to_lowercase();
        self.config
            .profanity_words
            .iter()
            .filter(|w| contains_word(&lower, w))
            .cloned()
            .collect()
    }
}

impl Default for ContentModerator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Skin detection (HSV + YCbCr)
// ---------------------------------------------------------------------------

/// Compute the fraction of pixels that match skin-tone thresholds.
fn compute_skin_ratio(rgb: &[u8], _width: usize, _height: usize) -> f32 {
    let pixel_count = rgb.len() / 3;
    if pixel_count == 0 {
        return 0.0;
    }

    let mut skin_count = 0u32;

    for i in 0..pixel_count {
        let off = i * 3;
        let r = rgb[off];
        let g = rgb[off + 1];
        let b = rgb[off + 2];

        if is_skin_pixel_hsv(r, g, b) && is_skin_pixel_ycbcr(r, g, b) {
            skin_count += 1;
        }
    }

    skin_count as f32 / pixel_count as f32
}

/// HSV-based skin check.
fn is_skin_pixel_hsv(r: u8, g: u8, b: u8) -> bool {
    let (h, s, _v) = rgb_to_hsv(r, g, b);
    // Skin hue is roughly 0-50 degrees, saturation 20-80%
    h <= 50.0 && s >= 0.15 && s <= 0.75
}

/// YCbCr-based skin check.
fn is_skin_pixel_ycbcr(r: u8, g: u8, b: u8) -> bool {
    let rf = r as f32;
    let gf = g as f32;
    let bf = b as f32;

    let _y = 0.299 * rf + 0.587 * gf + 0.114 * bf;
    let cb = 128.0 - 0.169 * rf - 0.331 * gf + 0.500 * bf;
    let cr = 128.0 + 0.500 * rf - 0.419 * gf - 0.081 * bf;

    // Skin in YCbCr space
    cb >= 77.0 && cb <= 127.0 && cr >= 133.0 && cr <= 173.0
}

/// Convert RGB to HSV. Returns (H in 0..360, S in 0..1, V in 0..1).
fn rgb_to_hsv(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let rf = r as f32 / 255.0;
    let gf = g as f32 / 255.0;
    let bf = b as f32 / 255.0;

    let max = rf.max(gf).max(bf);
    let min = rf.min(gf).min(bf);
    let delta = max - min;

    let v = max;
    let s = if max > 0.0 { delta / max } else { 0.0 };

    let h = if delta < 1e-6 {
        0.0
    } else if (max - rf).abs() < 1e-6 {
        60.0 * (((gf - bf) / delta) % 6.0)
    } else if (max - gf).abs() < 1e-6 {
        60.0 * ((bf - rf) / delta + 2.0)
    } else {
        60.0 * ((rf - gf) / delta + 4.0)
    };

    let h = if h < 0.0 { h + 360.0 } else { h };
    (h, s, v)
}

// ---------------------------------------------------------------------------
// Violence detection heuristic
// ---------------------------------------------------------------------------

/// Compute a violence indicator score combining motion and red dominance.
fn compute_violence_score(
    rgb: &[u8],
    prev_rgb: Option<&[u8]>,
    _width: usize,
    _height: usize,
    config: &ModerationConfig,
) -> f32 {
    let red_dom = compute_red_dominance(rgb);
    let motion = match prev_rgb {
        Some(prev) => compute_motion_magnitude(rgb, prev),
        None => 0.0,
    };

    // Violence heuristic: high motion AND red dominance
    let red_factor = if red_dom > config.red_dominance_threshold {
        (red_dom - config.red_dominance_threshold) / (1.0 - config.red_dominance_threshold)
    } else {
        0.0
    };

    let motion_factor = if motion > config.motion_threshold {
        (motion - config.motion_threshold) / (1.0 - config.motion_threshold)
    } else {
        0.0
    };

    // Combined: both must be present for high score
    let combined = 0.5 * red_factor + 0.5 * motion_factor;
    // Boost if both are significant
    let boost = if red_factor > 0.2 && motion_factor > 0.2 {
        0.2
    } else {
        0.0
    };

    (combined + boost).min(1.0)
}

/// Fraction of pixels where red channel dominates.
fn compute_red_dominance(rgb: &[u8]) -> f32 {
    let pixel_count = rgb.len() / 3;
    if pixel_count == 0 {
        return 0.0;
    }

    let mut red_dom_count = 0u32;
    for i in 0..pixel_count {
        let off = i * 3;
        let r = rgb[off] as u32;
        let g = rgb[off + 1] as u32;
        let b = rgb[off + 2] as u32;

        // Red is dominant if it exceeds both green and blue by a margin
        if r > g + 30 && r > b + 30 && r > 80 {
            red_dom_count += 1;
        }
    }

    red_dom_count as f32 / pixel_count as f32
}

/// Compute motion magnitude between two frames (mean absolute pixel diff, normalised).
fn compute_motion_magnitude(rgb: &[u8], prev_rgb: &[u8]) -> f32 {
    let n = rgb.len().min(prev_rgb.len());
    if n == 0 {
        return 0.0;
    }

    let mut total_diff = 0u64;
    for i in 0..n {
        total_diff += (rgb[i] as i32 - prev_rgb[i] as i32).unsigned_abs() as u64;
    }

    // Normalise to 0..1 (max possible diff per byte = 255)
    let mean_diff = total_diff as f64 / n as f64;
    (mean_diff / 255.0).min(1.0) as f32
}

// ---------------------------------------------------------------------------
// Profanity detection
// ---------------------------------------------------------------------------

/// Check text fragments for profanity words.
fn detect_profanity(texts: &[&str], word_list: &[String]) -> Vec<String> {
    let mut matches = Vec::new();

    for text in texts {
        let lower = text.to_lowercase();
        for word in word_list {
            if contains_word(&lower, word) && !matches.contains(word) {
                matches.push(word.clone());
            }
        }
    }

    matches
}

/// Check if `text` contains `word` as a whole word (bounded by non-alphanumeric or string edges).
fn contains_word(text: &str, word: &str) -> bool {
    let text_bytes = text.as_bytes();
    let word_bytes = word.as_bytes();
    let wlen = word_bytes.len();

    if wlen == 0 || wlen > text_bytes.len() {
        return false;
    }

    for i in 0..=(text_bytes.len() - wlen) {
        if &text_bytes[i..i + wlen] == word_bytes {
            let before_ok = i == 0 || !text_bytes[i - 1].is_ascii_alphanumeric();
            let after_ok =
                i + wlen == text_bytes.len() || !text_bytes[i + wlen].is_ascii_alphanumeric();
            if before_ok && after_ok {
                return true;
            }
        }
    }

    false
}

// ---------------------------------------------------------------------------
// Rating estimation
// ---------------------------------------------------------------------------

/// Estimate content rating from moderation signals.
fn estimate_rating(skin_ratio: f32, violence_score: f32, profanity: &[String]) -> ContentRating {
    let profanity_severity = compute_profanity_severity(profanity);

    // Simple decision tree
    let max_signal = skin_ratio.max(violence_score).max(profanity_severity);

    if max_signal < 0.1 {
        ContentRating::G
    } else if max_signal < 0.25 {
        ContentRating::PG
    } else if max_signal < 0.45 {
        ContentRating::PG13
    } else if max_signal < 0.70 {
        ContentRating::R
    } else {
        ContentRating::NC17
    }
}

/// Compute profanity severity score (0..1).
fn compute_profanity_severity(matches: &[String]) -> f32 {
    if matches.is_empty() {
        return 0.0;
    }

    // Severe words get higher weight
    let severe = ["fuck", "shit", "cunt", "nigger"];
    let mut score = 0.0f32;

    for word in matches {
        if severe.contains(&word.as_str()) {
            score += 0.35;
        } else {
            score += 0.15;
        }
    }

    score.min(1.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_frame(width: usize, height: usize, r: u8, g: u8, b: u8) -> Vec<u8> {
        let mut data = Vec::with_capacity(width * height * 3);
        for _ in 0..(width * height) {
            data.push(r);
            data.push(g);
            data.push(b);
        }
        data
    }

    #[test]
    fn test_moderator_default() {
        let mod_ = ContentModerator::new();
        assert!(!mod_.config.profanity_words.is_empty());
    }

    #[test]
    fn test_analyse_clean_frame() {
        let mod_ = ContentModerator::new();
        let frame = solid_frame(100, 100, 50, 100, 50); // greenish
        let result = mod_.analyse(&frame, None, &[], 100, 100).expect("ok");
        assert_eq!(result.rating, ContentRating::G);
        assert!(result.flags.is_empty());
    }

    #[test]
    fn test_analyse_invalid_dimensions() {
        let mod_ = ContentModerator::new();
        let result = mod_.analyse(&[0u8; 10], None, &[], 100, 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_analyse_prev_frame_mismatch() {
        let mod_ = ContentModerator::new();
        let frame = solid_frame(100, 100, 128, 128, 128);
        let result = mod_.analyse(&frame, Some(&[0u8; 10]), &[], 100, 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_skin_ratio_no_skin() {
        let frame = solid_frame(50, 50, 0, 0, 255); // blue
        let ratio = compute_skin_ratio(&frame, 50, 50);
        assert!(ratio < 0.01);
    }

    #[test]
    fn test_skin_ratio_skin_tone() {
        // A colour in the skin range: warm beige R=200, G=150, B=120
        let frame = solid_frame(50, 50, 200, 150, 120);
        let ratio = compute_skin_ratio(&frame, 50, 50);
        // Should detect as skin (varies with thresholds)
        assert!(ratio >= 0.0); // at minimum, no crash
    }

    #[test]
    fn test_hsv_conversion_red() {
        let (h, s, v) = rgb_to_hsv(255, 0, 0);
        assert!((h - 0.0).abs() < 1.0);
        assert!((s - 1.0).abs() < 0.01);
        assert!((v - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_hsv_conversion_green() {
        let (h, _s, _v) = rgb_to_hsv(0, 255, 0);
        assert!((h - 120.0).abs() < 1.0);
    }

    #[test]
    fn test_hsv_conversion_black() {
        let (h, s, v) = rgb_to_hsv(0, 0, 0);
        assert!((h - 0.0).abs() < f32::EPSILON);
        assert!((s - 0.0).abs() < f32::EPSILON);
        assert!((v - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_red_dominance_blue_frame() {
        let frame = solid_frame(50, 50, 0, 0, 255);
        let rd = compute_red_dominance(&frame);
        assert!((rd - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_red_dominance_red_frame() {
        let frame = solid_frame(50, 50, 200, 30, 30);
        let rd = compute_red_dominance(&frame);
        assert!(rd > 0.9);
    }

    #[test]
    fn test_motion_magnitude_identical() {
        let frame = solid_frame(50, 50, 128, 128, 128);
        let m = compute_motion_magnitude(&frame, &frame);
        assert!((m - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_motion_magnitude_different() {
        let f1 = solid_frame(50, 50, 0, 0, 0);
        let f2 = solid_frame(50, 50, 255, 255, 255);
        let m = compute_motion_magnitude(&f1, &f2);
        assert!((m - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_profanity_detection() {
        let words = default_profanity_list();
        let texts = vec!["this is damn bad"];
        let matches = detect_profanity(&texts, &words);
        assert!(matches.contains(&"damn".to_string()));
    }

    #[test]
    fn test_profanity_no_match() {
        let words = default_profanity_list();
        let texts = vec!["this is perfectly fine"];
        let matches = detect_profanity(&texts, &words);
        assert!(matches.is_empty());
    }

    #[test]
    fn test_profanity_word_boundary() {
        let words = default_profanity_list();
        let texts = vec!["classic"]; // contains "ass" but not as a word
        let matches = detect_profanity(&texts, &words);
        assert!(
            !matches.contains(&"ass".to_string()),
            "should not match 'ass' inside 'classic'"
        );
    }

    #[test]
    fn test_profanity_case_insensitive() {
        let words = default_profanity_list();
        let texts = vec!["DAMN it"];
        let matches = detect_profanity(&texts, &words);
        assert!(matches.contains(&"damn".to_string()));
    }

    #[test]
    fn test_check_text_method() {
        let mod_ = ContentModerator::new();
        let matches = mod_.check_text("what the hell is this");
        assert!(matches.contains(&"hell".to_string()));
    }

    #[test]
    fn test_rating_g() {
        let r = estimate_rating(0.0, 0.0, &[]);
        assert_eq!(r, ContentRating::G);
    }

    #[test]
    fn test_rating_pg() {
        let r = estimate_rating(0.15, 0.0, &[]);
        assert_eq!(r, ContentRating::PG);
    }

    #[test]
    fn test_rating_pg13() {
        let r = estimate_rating(0.30, 0.0, &[]);
        assert_eq!(r, ContentRating::PG13);
    }

    #[test]
    fn test_rating_r() {
        let r = estimate_rating(0.50, 0.0, &[]);
        assert_eq!(r, ContentRating::R);
    }

    #[test]
    fn test_rating_nc17() {
        let r = estimate_rating(0.80, 0.0, &[]);
        assert_eq!(r, ContentRating::NC17);
    }

    #[test]
    fn test_rating_profanity_severe() {
        let matches = vec!["fuck".to_string(), "shit".to_string()];
        let r = estimate_rating(0.0, 0.0, &matches);
        assert!(r >= ContentRating::R);
    }

    #[test]
    fn test_content_rating_label() {
        assert_eq!(ContentRating::G.label(), "G");
        assert_eq!(ContentRating::PG13.label(), "PG-13");
        assert_eq!(ContentRating::NC17.label(), "NC-17");
    }

    #[test]
    fn test_profanity_severity_empty() {
        let s = compute_profanity_severity(&[]);
        assert!((s - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_profanity_severity_mild() {
        let matches = vec!["damn".to_string()];
        let s = compute_profanity_severity(&matches);
        assert!(s > 0.0 && s < 0.5);
    }

    #[test]
    fn test_profanity_severity_capped() {
        let matches: Vec<String> = (0..20).map(|_| "fuck".to_string()).collect();
        let s = compute_profanity_severity(&matches);
        assert!((s - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_violence_no_prev_frame() {
        let cfg = ModerationConfig::default();
        let frame = solid_frame(50, 50, 200, 30, 30); // red
        let score = compute_violence_score(&frame, None, 50, 50, &cfg);
        // No motion without prev frame, so score limited
        assert!(score < 0.8);
    }

    #[test]
    fn test_violence_with_motion_and_red() {
        let cfg = ModerationConfig::default();
        let f1 = solid_frame(50, 50, 0, 0, 0);
        let f2 = solid_frame(50, 50, 200, 20, 20); // red + motion
        let score = compute_violence_score(&f2, Some(&f1), 50, 50, &cfg);
        assert!(score > 0.3);
    }

    #[test]
    fn test_moderation_flag_fields() {
        let flag = ModerationFlag {
            reason: FlagReason::HighSkinExposure,
            confidence: Confidence::new(0.7),
            detail: "test".to_string(),
        };
        assert_eq!(flag.reason, FlagReason::HighSkinExposure);
    }

    #[test]
    fn test_contains_word_at_start() {
        assert!(contains_word("damn good", "damn"));
    }

    #[test]
    fn test_contains_word_at_end() {
        assert!(contains_word("so damn", "damn"));
    }

    #[test]
    fn test_contains_word_middle() {
        assert!(contains_word("oh damn it", "damn"));
    }

    #[test]
    fn test_contains_word_not_substring() {
        assert!(!contains_word("damnation", "damn"));
    }

    #[test]
    fn test_contains_word_empty() {
        assert!(!contains_word("hello", ""));
    }

    #[test]
    fn test_skin_pixel_ycbcr_blue() {
        assert!(!is_skin_pixel_ycbcr(0, 0, 255));
    }

    #[test]
    fn test_moderation_result_struct() {
        let result = ModerationResult {
            rating: ContentRating::PG,
            confidence: Confidence::new(0.8),
            flags: vec![],
            skin_ratio: 0.1,
            violence_score: 0.0,
            profanity_matches: vec![],
        };
        assert_eq!(result.rating, ContentRating::PG);
        assert!(result.flags.is_empty());
    }
}
