//! AI-powered clip tagging via rule-based heuristics.
//!
//! Analyses clip metadata (file name, duration, resolution, audio) and
//! returns a ranked list of [`TagSuggestion`]s, each backed by a confidence
//! score and a provenance [`TagSource`].

/// Origin of a generated tag suggestion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TagSource {
    /// Derived from the clip file name.
    FileName,
    /// Derived from the clip duration.
    Duration,
    /// Derived from the video resolution.
    Resolution,
    /// Derived from audio properties (channel count / level).
    AudioLevel,
    /// Produced by a configured manual rule.
    ManualRule,
}

/// A single AI-generated tag suggestion.
#[derive(Debug, Clone)]
pub struct TagSuggestion {
    /// The proposed tag string.
    pub tag: String,
    /// Confidence score in the range \[0.0, 1.0\].
    pub confidence: f32,
    /// Which analyser produced this suggestion.
    pub source: TagSource,
}

/// Metadata about a clip used as input to [`AiTagger`].
#[derive(Debug, Clone)]
pub struct ClipInfo {
    /// File name (base name only, e.g. `"interview_take1.mp4"`).
    pub file_name: String,
    /// Duration in seconds.
    pub duration_secs: f64,
    /// Video width in pixels (0 if unknown / audio-only).
    pub width: u32,
    /// Video height in pixels (0 if unknown / audio-only).
    pub height: u32,
    /// Number of audio channels (0 if no audio stream).
    pub audio_channels: u8,
    /// Integrated audio level in dBFS (0.0 if no audio).
    pub audio_level_db: f32,
}

/// Configuration for [`AiTagger`].
#[derive(Debug, Clone)]
pub struct AiTaggerConfig {
    /// Only emit suggestions with confidence >= this value.
    pub min_confidence: f32,
    /// Which sources are active. An empty vec enables all sources.
    pub enabled_sources: Vec<TagSource>,
}

impl Default for AiTaggerConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.0,
            enabled_sources: Vec::new(),
        }
    }
}

impl AiTaggerConfig {
    /// Create a config that emits suggestions from all sources above `threshold`.
    #[must_use]
    pub fn with_threshold(min_confidence: f32) -> Self {
        Self {
            min_confidence,
            enabled_sources: Vec::new(),
        }
    }
}

/// Rule-based AI tagger that analyses [`ClipInfo`] and produces [`TagSuggestion`]s.
#[derive(Debug, Clone)]
pub struct AiTagger {
    config: AiTaggerConfig,
}

impl AiTagger {
    /// Create a new tagger with the given configuration.
    #[must_use]
    pub fn new(config: AiTaggerConfig) -> Self {
        Self { config }
    }

    /// Returns true when `source` is enabled by the configuration.
    fn source_enabled(&self, source: TagSource) -> bool {
        if self.config.enabled_sources.is_empty() {
            return true;
        }
        self.config.enabled_sources.contains(&source)
    }

    /// Analyse `info` and return all tag suggestions above the configured threshold.
    #[must_use]
    pub fn suggest_tags(&self, info: &ClipInfo) -> Vec<TagSuggestion> {
        let mut out: Vec<TagSuggestion> = Vec::new();

        // --- Duration-based rules ---
        if self.source_enabled(TagSource::Duration) {
            if info.duration_secs > 0.0 && info.duration_secs < 30.0 {
                out.push(TagSuggestion {
                    tag: "short_clip".to_string(),
                    confidence: 0.90,
                    source: TagSource::Duration,
                });
            }
            if info.duration_secs > 300.0 {
                out.push(TagSuggestion {
                    tag: "long_form".to_string(),
                    confidence: 0.85,
                    source: TagSource::Duration,
                });
            }
        }

        // --- Resolution-based rules ---
        if self.source_enabled(TagSource::Resolution) && (info.width > 0 || info.height > 0) {
            if info.width >= 3840 || info.height >= 2160 {
                out.push(TagSuggestion {
                    tag: "uhd".to_string(),
                    confidence: 0.95,
                    source: TagSource::Resolution,
                });
            } else if info.width >= 1920 || info.height >= 1080 {
                out.push(TagSuggestion {
                    tag: "hd".to_string(),
                    confidence: 0.90,
                    source: TagSource::Resolution,
                });
            } else if info.width > 0 && info.width < 1280 {
                out.push(TagSuggestion {
                    tag: "sd".to_string(),
                    confidence: 0.85,
                    source: TagSource::Resolution,
                });
            }
        }

        // --- Audio channel / level rules ---
        if self.source_enabled(TagSource::AudioLevel) && info.audio_channels > 0 {
            match info.audio_channels {
                1 => out.push(TagSuggestion {
                    tag: "mono".to_string(),
                    confidence: 0.90,
                    source: TagSource::AudioLevel,
                }),
                2 => out.push(TagSuggestion {
                    tag: "stereo".to_string(),
                    confidence: 0.90,
                    source: TagSource::AudioLevel,
                }),
                _ => out.push(TagSuggestion {
                    tag: "surround".to_string(),
                    confidence: 0.88,
                    source: TagSource::AudioLevel,
                }),
            }

            if info.audio_level_db > -6.0 {
                out.push(TagSuggestion {
                    tag: "loud".to_string(),
                    confidence: 0.75,
                    source: TagSource::AudioLevel,
                });
            } else if info.audio_level_db < -30.0 {
                out.push(TagSuggestion {
                    tag: "quiet".to_string(),
                    confidence: 0.75,
                    source: TagSource::AudioLevel,
                });
            }
        }

        // --- File-name rules ---
        if self.source_enabled(TagSource::FileName) {
            let lower = info.file_name.to_lowercase();

            if lower.contains("interview") {
                out.push(TagSuggestion {
                    tag: "interview".to_string(),
                    confidence: 0.80,
                    source: TagSource::FileName,
                });
            }
            if lower.contains("broll") || lower.contains("b_roll") {
                out.push(TagSuggestion {
                    tag: "broll".to_string(),
                    confidence: 0.80,
                    source: TagSource::FileName,
                });
            }
            if lower.contains("timelapse") {
                out.push(TagSuggestion {
                    tag: "timelapse".to_string(),
                    confidence: 0.82,
                    source: TagSource::FileName,
                });
            }
            if lower.ends_with(".mp4") {
                out.push(TagSuggestion {
                    tag: "mp4".to_string(),
                    confidence: 0.95,
                    source: TagSource::FileName,
                });
            }
        }

        // --- Manual / compound rules ---
        if self.source_enabled(TagSource::ManualRule) {
            // Social-media sweet-spot: 25–35 seconds
            if info.duration_secs >= 25.0 && info.duration_secs <= 35.0 {
                out.push(TagSuggestion {
                    tag: "social_clip".to_string(),
                    confidence: 0.70,
                    source: TagSource::ManualRule,
                });
            }
        }

        // Apply confidence threshold
        out.retain(|s| s.confidence >= self.config.min_confidence);

        // Sort descending by confidence for convenience
        out.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        out
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_info() -> ClipInfo {
        ClipInfo {
            file_name: "clip.mp4".to_string(),
            duration_secs: 60.0,
            width: 1920,
            height: 1080,
            audio_channels: 2,
            audio_level_db: -18.0,
        }
    }

    fn tagger() -> AiTagger {
        AiTagger::new(AiTaggerConfig::default())
    }

    // --- Duration rules ---

    #[test]
    fn test_short_clip_tag() {
        let mut info = default_info();
        info.duration_secs = 10.0;
        let tags = tagger().suggest_tags(&info);
        assert!(tags.iter().any(|t| t.tag == "short_clip"));
    }

    #[test]
    fn test_long_form_tag() {
        let mut info = default_info();
        info.duration_secs = 600.0;
        let tags = tagger().suggest_tags(&info);
        assert!(tags.iter().any(|t| t.tag == "long_form"));
    }

    #[test]
    fn test_no_duration_tag_for_medium_clip() {
        let mut info = default_info();
        info.duration_secs = 60.0;
        let tags = tagger().suggest_tags(&info);
        assert!(!tags.iter().any(|t| t.tag == "short_clip"));
        assert!(!tags.iter().any(|t| t.tag == "long_form"));
    }

    #[test]
    fn test_short_clip_boundary_exactly_30s_not_tagged() {
        let mut info = default_info();
        info.duration_secs = 30.0; // NOT < 30
        let tags = tagger().suggest_tags(&info);
        assert!(!tags.iter().any(|t| t.tag == "short_clip"));
    }

    // --- Resolution rules ---

    #[test]
    fn test_uhd_tag_4k_width() {
        let mut info = default_info();
        info.width = 3840;
        info.height = 2160;
        let tags = tagger().suggest_tags(&info);
        assert!(tags.iter().any(|t| t.tag == "uhd"));
    }

    #[test]
    fn test_uhd_tag_height_only() {
        let mut info = default_info();
        info.width = 3840;
        info.height = 2160;
        let tags = tagger().suggest_tags(&info);
        assert!(tags.iter().any(|t| t.tag == "uhd"));
    }

    #[test]
    fn test_hd_tag() {
        let tags = tagger().suggest_tags(&default_info());
        assert!(tags.iter().any(|t| t.tag == "hd"));
    }

    #[test]
    fn test_sd_tag() {
        let mut info = default_info();
        info.width = 640;
        info.height = 480;
        let tags = tagger().suggest_tags(&info);
        assert!(tags.iter().any(|t| t.tag == "sd"));
    }

    // --- Audio rules ---

    #[test]
    fn test_stereo_tag() {
        let tags = tagger().suggest_tags(&default_info());
        assert!(tags.iter().any(|t| t.tag == "stereo"));
    }

    #[test]
    fn test_mono_tag() {
        let mut info = default_info();
        info.audio_channels = 1;
        let tags = tagger().suggest_tags(&info);
        assert!(tags.iter().any(|t| t.tag == "mono"));
    }

    #[test]
    fn test_surround_tag() {
        let mut info = default_info();
        info.audio_channels = 6;
        let tags = tagger().suggest_tags(&info);
        assert!(tags.iter().any(|t| t.tag == "surround"));
    }

    #[test]
    fn test_loud_tag() {
        let mut info = default_info();
        info.audio_level_db = -3.0;
        let tags = tagger().suggest_tags(&info);
        assert!(tags.iter().any(|t| t.tag == "loud"));
    }

    #[test]
    fn test_quiet_tag() {
        let mut info = default_info();
        info.audio_level_db = -40.0;
        let tags = tagger().suggest_tags(&info);
        assert!(tags.iter().any(|t| t.tag == "quiet"));
    }

    // --- File-name rules ---

    #[test]
    fn test_interview_tag() {
        let mut info = default_info();
        info.file_name = "Interview_Take1.mp4".to_string();
        let tags = tagger().suggest_tags(&info);
        assert!(tags.iter().any(|t| t.tag == "interview"));
    }

    #[test]
    fn test_broll_tag() {
        let mut info = default_info();
        info.file_name = "broll_forest.mp4".to_string();
        let tags = tagger().suggest_tags(&info);
        assert!(tags.iter().any(|t| t.tag == "broll"));
    }

    #[test]
    fn test_b_roll_underscore_tag() {
        let mut info = default_info();
        info.file_name = "b_roll_city.mp4".to_string();
        let tags = tagger().suggest_tags(&info);
        assert!(tags.iter().any(|t| t.tag == "broll"));
    }

    #[test]
    fn test_timelapse_tag() {
        let mut info = default_info();
        info.file_name = "timelapse_sunset.mp4".to_string();
        let tags = tagger().suggest_tags(&info);
        assert!(tags.iter().any(|t| t.tag == "timelapse"));
    }

    #[test]
    fn test_mp4_tag() {
        let tags = tagger().suggest_tags(&default_info());
        assert!(tags.iter().any(|t| t.tag == "mp4"));
    }

    // --- Manual rule ---

    #[test]
    fn test_social_clip_tag() {
        let mut info = default_info();
        info.duration_secs = 30.0;
        let tags = tagger().suggest_tags(&info);
        assert!(tags.iter().any(|t| t.tag == "social_clip"));
    }

    // --- Threshold filtering ---

    #[test]
    fn test_threshold_filters_low_confidence() {
        let config = AiTaggerConfig::with_threshold(0.80);
        let tagger = AiTagger::new(config);
        let mut info = default_info();
        info.audio_level_db = -3.0; // would produce "loud" at 0.75
        let tags = tagger.suggest_tags(&info);
        // "loud" (0.75) must be filtered out
        assert!(!tags.iter().any(|t| t.tag == "loud"));
        // "hd" (0.90) must survive
        assert!(tags.iter().any(|t| t.tag == "hd"));
    }

    #[test]
    fn test_threshold_zero_keeps_everything() {
        let config = AiTaggerConfig::with_threshold(0.0);
        let tagger = AiTagger::new(config);
        let mut info = default_info();
        info.audio_level_db = -3.0;
        let tags = tagger.suggest_tags(&info);
        assert!(tags.iter().any(|t| t.tag == "loud"));
    }

    // --- Source filter ---

    #[test]
    fn test_enabled_sources_filter() {
        let config = AiTaggerConfig {
            min_confidence: 0.0,
            enabled_sources: vec![TagSource::Resolution],
        };
        let tagger = AiTagger::new(config);
        let tags = tagger.suggest_tags(&default_info());
        // Only resolution tags should appear
        for tag in &tags {
            assert_eq!(tag.source, TagSource::Resolution);
        }
    }

    // --- Sorted output ---

    #[test]
    fn test_output_sorted_descending() {
        let mut info = default_info();
        info.width = 3840;
        info.height = 2160;
        info.audio_level_db = -3.0;
        let tags = tagger().suggest_tags(&info);
        for window in tags.windows(2) {
            assert!(window[0].confidence >= window[1].confidence);
        }
    }

    // --- No audio ---

    #[test]
    fn test_no_audio_produces_no_audio_tags() {
        let mut info = default_info();
        info.audio_channels = 0;
        let tags = tagger().suggest_tags(&info);
        assert!(!tags.iter().any(|t| t.source == TagSource::AudioLevel));
    }

    // --- Zero duration ---

    #[test]
    fn test_zero_duration_no_duration_tags() {
        let mut info = default_info();
        info.duration_secs = 0.0;
        let tags = tagger().suggest_tags(&info);
        assert!(!tags.iter().any(|t| t.tag == "short_clip"));
        assert!(!tags.iter().any(|t| t.tag == "long_form"));
    }
}
