//! Filename-based matching strategies.

use crate::config::ConformConfig;
use crate::types::{ClipMatch, ClipReference, MatchMethod, MediaFile};
use regex::Regex;
use strsim::levenshtein;

/// Match media files by exact filename.
#[must_use]
pub fn exact_filename_match(clip: &ClipReference, candidates: &[MediaFile]) -> Vec<ClipMatch> {
    let mut matches = Vec::new();

    if let Some(source_file) = &clip.source_file {
        for media in candidates {
            if media.filename == *source_file {
                matches.push(ClipMatch {
                    clip: clip.clone(),
                    media: media.clone(),
                    score: 1.0,
                    method: MatchMethod::ExactFilename,
                    details: format!("Exact filename match: {source_file}"),
                });
            }
        }
    }

    matches
}

/// Match media files by fuzzy filename.
#[must_use]
pub fn fuzzy_filename_match(
    clip: &ClipReference,
    candidates: &[MediaFile],
    config: &ConformConfig,
) -> Vec<ClipMatch> {
    let mut matches = Vec::new();

    if let Some(source_file) = &clip.source_file {
        for media in candidates {
            let distance = levenshtein(source_file, &media.filename);
            if distance <= config.fuzzy_max_distance {
                let max_len = source_file.len().max(media.filename.len());
                let score = if max_len == 0 {
                    0.0
                } else {
                    1.0 - (distance as f64 / max_len as f64)
                };

                // For fuzzy matching, fuzzy_max_distance is the binding constraint;
                // score is informational (not a pass/fail gate).
                matches.push(ClipMatch {
                    clip: clip.clone(),
                    media: media.clone(),
                    score,
                    method: MatchMethod::FuzzyFilename,
                    details: format!(
                        "Fuzzy filename match: {} -> {} (distance: {distance})",
                        source_file, media.filename
                    ),
                });
            }
        }
    }

    matches
}

/// Match media files by filename pattern.
#[must_use]
pub fn pattern_filename_match(
    clip: &ClipReference,
    candidates: &[MediaFile],
    pattern: &str,
) -> Vec<ClipMatch> {
    let mut matches = Vec::new();

    if let Ok(regex) = Regex::new(pattern) {
        for media in candidates {
            if regex.is_match(&media.filename) {
                matches.push(ClipMatch {
                    clip: clip.clone(),
                    media: media.clone(),
                    score: 0.9,
                    method: MatchMethod::ExactFilename,
                    details: format!("Pattern match: {} matches {pattern}", media.filename),
                });
            }
        }
    }

    matches
}

/// Normalize filename for matching (remove extension, convert to lowercase).
#[must_use]
pub fn normalize_filename(filename: &str) -> String {
    let without_ext = filename.rsplit_once('.').map_or(filename, |(base, _)| base);
    without_ext.to_lowercase()
}

/// Match with proxy-to-high-res relinking.
#[must_use]
pub fn relink_proxy_match(
    clip: &ClipReference,
    candidates: &[MediaFile],
    config: &ConformConfig,
) -> Vec<ClipMatch> {
    let mut matches = Vec::new();

    if let Some(source_file) = &clip.source_file {
        for (proxy_pattern, highres_pattern) in &config.proxy_patterns {
            if source_file.contains(proxy_pattern) {
                let highres_name = source_file.replace(proxy_pattern, highres_pattern);

                for media in candidates {
                    if media.filename == highres_name {
                        matches.push(ClipMatch {
                            clip: clip.clone(),
                            media: media.clone(),
                            score: 0.95,
                            method: MatchMethod::ExactFilename,
                            details: format!("Proxy relink: {source_file} -> {highres_name}"),
                        });
                    }
                }
            }
        }
    }

    matches
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{FrameRate, Timecode, TrackType};
    use std::path::PathBuf;

    fn create_test_clip(source_file: &str) -> ClipReference {
        ClipReference {
            id: "test".to_string(),
            source_file: Some(source_file.to_string()),
            source_in: Timecode::new(1, 0, 0, 0),
            source_out: Timecode::new(1, 0, 10, 0),
            record_in: Timecode::new(1, 0, 0, 0),
            record_out: Timecode::new(1, 0, 10, 0),
            track: TrackType::Video,
            fps: FrameRate::Fps25,
            metadata: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn test_exact_filename_match() {
        let clip = create_test_clip("test.mov");
        let media = MediaFile::new(PathBuf::from("/path/test.mov"));
        let matches = exact_filename_match(&clip, &[media]);
        assert_eq!(matches.len(), 1);
        assert!((matches[0].score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_fuzzy_filename_match() {
        let clip = create_test_clip("test.mov");
        let mut media = MediaFile::new(PathBuf::from("/path/tset.mov"));
        media.filename = "tset.mov".to_string();

        let config = ConformConfig::default();
        let matches = fuzzy_filename_match(&clip, &[media], &config);
        assert!(!matches.is_empty());
    }

    #[test]
    fn test_normalize_filename() {
        assert_eq!(normalize_filename("Test.MOV"), "test");
        assert_eq!(normalize_filename("my_file.mp4"), "my_file");
    }

    #[test]
    fn test_pattern_match() {
        let clip = create_test_clip("shot_001.mov");
        let media = MediaFile::new(PathBuf::from("/path/shot_001.mov"));
        let matches = pattern_filename_match(&clip, &[media], r"shot_\d+\.mov");
        assert_eq!(matches.len(), 1);
    }
}
