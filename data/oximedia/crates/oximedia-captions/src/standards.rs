//! Broadcasting and compliance standards

use crate::types::Caption;

/// FCC (Federal Communications Commission) standards for US
pub mod fcc {
    use super::Caption;

    /// Maximum characters per line for CEA-608
    pub const CEA608_MAX_CHARS: usize = 32;

    /// Maximum lines for CEA-608
    pub const CEA608_MAX_LINES: usize = 4;

    /// Maximum characters per line for CEA-708
    pub const CEA708_MAX_CHARS: usize = 42;

    /// Maximum lines for CEA-708
    pub const CEA708_MAX_LINES: usize = 15;

    /// Maximum reading speed (words per minute)
    pub const MAX_READING_SPEED_WPM: f64 = 180.0;

    /// Recommended reading speed (words per minute)
    pub const RECOMMENDED_READING_SPEED_WPM: f64 = 160.0;

    /// Minimum caption duration (milliseconds)
    pub const MIN_DURATION_MS: i64 = 1500;

    /// Check if caption meets FCC CEA-608 standards
    #[must_use]
    pub fn check_cea608_compliance(caption: &Caption) -> Vec<String> {
        let mut issues = Vec::new();

        if caption.max_chars_per_line() > CEA608_MAX_CHARS {
            issues.push(format!(
                "Exceeds maximum characters per line: {} > {}",
                caption.max_chars_per_line(),
                CEA608_MAX_CHARS
            ));
        }

        if caption.line_count() > CEA608_MAX_LINES {
            issues.push(format!(
                "Exceeds maximum lines: {} > {}",
                caption.line_count(),
                CEA608_MAX_LINES
            ));
        }

        if caption.reading_speed_wpm() > MAX_READING_SPEED_WPM {
            issues.push(format!(
                "Exceeds maximum reading speed: {:.1} > {} WPM",
                caption.reading_speed_wpm(),
                MAX_READING_SPEED_WPM
            ));
        }

        if caption.duration().as_millis() < MIN_DURATION_MS {
            issues.push(format!(
                "Duration too short: {}ms < {}ms",
                caption.duration().as_millis(),
                MIN_DURATION_MS
            ));
        }

        issues
    }
}

/// WCAG (Web Content Accessibility Guidelines) standards
pub mod wcag {
    use crate::types::Color;

    /// WCAG AA minimum contrast ratio
    pub const AA_MIN_CONTRAST: f64 = 4.5;

    /// WCAG AAA minimum contrast ratio
    pub const AAA_MIN_CONTRAST: f64 = 7.0;

    /// WCAG AA minimum contrast ratio for large text
    pub const AA_LARGE_TEXT_CONTRAST: f64 = 3.0;

    /// WCAG AAA minimum contrast ratio for large text
    pub const AAA_LARGE_TEXT_CONTRAST: f64 = 4.5;

    /// Large text threshold (18pt or 14pt bold)
    pub const LARGE_TEXT_SIZE: u32 = 18;

    /// Check contrast ratio compliance
    #[must_use]
    pub fn check_contrast_aa(text_color: &Color, bg_color: &Color) -> bool {
        text_color.contrast_ratio(bg_color) >= AA_MIN_CONTRAST
    }

    /// Check contrast ratio compliance (AAA)
    #[must_use]
    pub fn check_contrast_aaa(text_color: &Color, bg_color: &Color) -> bool {
        text_color.contrast_ratio(bg_color) >= AAA_MIN_CONTRAST
    }
}

/// EBU (European Broadcasting Union) standards
pub mod ebu {
    use super::Caption;

    /// EBU R128 loudness standard (referenced for caption timing)
    pub const R128_TARGET_LUFS: f64 = -23.0;

    /// Maximum characters per line for EBU-STL
    pub const MAX_CHARS_PER_LINE: usize = 40;

    /// Maximum lines for EBU-STL
    pub const MAX_LINES: usize = 2;

    /// Minimum gap between captions (frames at 25fps)
    pub const MIN_GAP_FRAMES: u32 = 2;

    /// Maximum reading speed (characters per second)
    pub const MAX_CPS: f64 = 20.0;

    /// Calculate characters per second
    #[must_use]
    pub fn calculate_cps(caption: &Caption) -> f64 {
        let chars = caption.character_count() as f64;
        let duration_secs = caption.duration().as_secs() as f64;
        if duration_secs == 0.0 {
            0.0
        } else {
            chars / duration_secs
        }
    }
}

/// BBC (British Broadcasting Corporation) standards
pub mod bbc {

    /// Maximum characters per line
    pub const MAX_CHARS_PER_LINE: usize = 37;

    /// Maximum lines
    pub const MAX_LINES: usize = 2;

    /// Maximum reading speed (words per minute)
    pub const MAX_READING_SPEED_WPM: f64 = 180.0;

    /// Minimum caption duration (milliseconds)
    pub const MIN_DURATION_MS: i64 = 1200;

    /// Minimum gap between captions (milliseconds)
    pub const MIN_GAP_MS: i64 = 160;
}

/// Netflix subtitle standards
pub mod netflix {

    /// Maximum characters per line
    pub const MAX_CHARS_PER_LINE: usize = 42;

    /// Maximum lines
    pub const MAX_LINES: usize = 2;

    /// Maximum reading speed (characters per second)
    pub const MAX_CPS: f64 = 20.0;

    /// Minimum caption duration (milliseconds)
    pub const MIN_DURATION_MS: i64 = 833; // 20 frames at 24fps

    /// Maximum caption duration (seconds)
    pub const MAX_DURATION_S: i64 = 7;

    /// Minimum gap between captions (frames at 24fps)
    pub const MIN_GAP_FRAMES: u32 = 2;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Color, Timestamp};

    #[test]
    fn test_fcc_compliance() {
        let caption = Caption::new(
            Timestamp::from_secs(1),
            Timestamp::from_secs(3),
            "Short text".to_string(),
        );

        let issues = fcc::check_cea608_compliance(&caption);
        assert!(issues.is_empty());
    }

    #[test]
    fn test_wcag_contrast() {
        let white = Color::white();
        let black = Color::black();

        assert!(wcag::check_contrast_aa(&white, &black));
        assert!(wcag::check_contrast_aaa(&white, &black));
    }

    #[test]
    fn test_ebu_cps() {
        let caption = Caption::new(
            Timestamp::from_secs(0),
            Timestamp::from_secs(2),
            "Test caption text".to_string(),
        );

        let cps = ebu::calculate_cps(&caption);
        assert!(cps > 0.0);
        assert!(cps < ebu::MAX_CPS);
    }
}
