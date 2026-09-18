#![allow(dead_code)]

//! Clip resolver for matching clip references to actual media paths.
//!
//! Implements multiple resolution strategies including exact path matching,
//! filename-based search, reel name mapping, and fuzzy matching to locate
//! media files referenced in EDL/XML/AAF timelines.

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};

/// Strategy used to resolve a clip reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ResolveStrategy {
    /// Exact path match.
    ExactPath,
    /// Match by filename only (ignoring directory).
    FileName,
    /// Match by reel name mapping.
    ReelName,
    /// Match by stem (filename without extension).
    Stem,
    /// Fuzzy match by string similarity.
    Fuzzy,
    /// Manual override by user.
    Manual,
}

impl fmt::Display for ResolveStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExactPath => write!(f, "exact_path"),
            Self::FileName => write!(f, "filename"),
            Self::ReelName => write!(f, "reel_name"),
            Self::Stem => write!(f, "stem"),
            Self::Fuzzy => write!(f, "fuzzy"),
            Self::Manual => write!(f, "manual"),
        }
    }
}

/// A reference to a clip that needs resolution.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ClipRef {
    /// Unique identifier for the clip.
    pub id: String,
    /// Source reel name from EDL/XML.
    pub reel_name: String,
    /// Original file path from the timeline (may not exist).
    pub original_path: Option<String>,
    /// Source in timecode.
    pub source_in: String,
    /// Source out timecode.
    pub source_out: String,
}

impl ClipRef {
    /// Create a new clip reference.
    #[must_use]
    pub fn new(id: String, reel_name: String, source_in: String, source_out: String) -> Self {
        Self {
            id,
            reel_name,
            original_path: None,
            source_in,
            source_out,
        }
    }

    /// Set the original path.
    pub fn with_original_path(mut self, path: String) -> Self {
        self.original_path = Some(path);
        self
    }

    /// Extract filename from the original path, if available.
    #[must_use]
    pub fn original_filename(&self) -> Option<&str> {
        self.original_path
            .as_ref()
            .and_then(|p| Path::new(p).file_name())
            .and_then(|n| n.to_str())
    }

    /// Extract stem from the original path.
    #[must_use]
    pub fn original_stem(&self) -> Option<&str> {
        self.original_path
            .as_ref()
            .and_then(|p| Path::new(p).file_stem())
            .and_then(|n| n.to_str())
    }
}

/// Result of resolving a single clip.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ResolveResult {
    /// The clip ID.
    pub clip_id: String,
    /// Whether resolution was successful.
    pub resolved: bool,
    /// The resolved media path, if found.
    pub media_path: Option<PathBuf>,
    /// The strategy that succeeded.
    pub strategy: Option<ResolveStrategy>,
    /// Confidence score from 0.0 to 1.0.
    pub confidence: f64,
    /// Alternative candidates found.
    pub alternatives: Vec<PathBuf>,
}

impl ResolveResult {
    /// Create an unresolved result.
    #[must_use]
    pub fn unresolved(clip_id: String) -> Self {
        Self {
            clip_id,
            resolved: false,
            media_path: None,
            strategy: None,
            confidence: 0.0,
            alternatives: Vec::new(),
        }
    }

    /// Create a resolved result.
    #[must_use]
    pub fn resolved(
        clip_id: String,
        media_path: PathBuf,
        strategy: ResolveStrategy,
        confidence: f64,
    ) -> Self {
        Self {
            clip_id,
            resolved: true,
            media_path: Some(media_path),
            strategy: Some(strategy),
            confidence,
            alternatives: Vec::new(),
        }
    }
}

/// Configuration for the clip resolver.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ResolverConfig {
    /// Strategies to try in order of preference.
    pub strategies: Vec<ResolveStrategy>,
    /// Minimum confidence threshold for accepting a fuzzy match.
    pub fuzzy_threshold: f64,
    /// Maximum number of alternative candidates to keep.
    pub max_alternatives: usize,
    /// Known file extensions to consider as media files.
    pub media_extensions: Vec<String>,
}

impl Default for ResolverConfig {
    fn default() -> Self {
        Self {
            strategies: vec![
                ResolveStrategy::ExactPath,
                ResolveStrategy::FileName,
                ResolveStrategy::Stem,
                ResolveStrategy::ReelName,
                ResolveStrategy::Fuzzy,
            ],
            fuzzy_threshold: 0.7,
            max_alternatives: 5,
            media_extensions: vec![
                "mxf".to_string(),
                "mov".to_string(),
                "mp4".to_string(),
                "avi".to_string(),
                "wav".to_string(),
                "aif".to_string(),
                "aiff".to_string(),
                "dpx".to_string(),
                "exr".to_string(),
            ],
        }
    }
}

/// A media file entry in the search index.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MediaEntry {
    /// Full path to the media file.
    pub path: PathBuf,
    /// File name (without directory).
    pub filename: String,
    /// File stem (without extension).
    pub stem: String,
    /// File extension.
    pub extension: String,
}

impl MediaEntry {
    /// Create from a path.
    #[must_use]
    pub fn from_path(path: PathBuf) -> Self {
        let filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();
        let stem = path
            .file_stem()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();
        let extension = path
            .extension()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string()
            .to_lowercase();
        Self {
            path,
            filename,
            stem,
            extension,
        }
    }
}

/// Simple string similarity (Dice coefficient on bigrams).
#[allow(clippy::cast_precision_loss)]
fn string_similarity(a: &str, b: &str) -> f64 {
    if a == b {
        return 1.0;
    }
    if a.len() < 2 || b.len() < 2 {
        return 0.0;
    }
    let a_lower = a.to_lowercase();
    let b_lower = b.to_lowercase();
    let a_bigrams: Vec<(char, char)> = a_lower.chars().zip(a_lower.chars().skip(1)).collect();
    let b_bigrams: Vec<(char, char)> = b_lower.chars().zip(b_lower.chars().skip(1)).collect();

    let matches = a_bigrams.iter().filter(|bg| b_bigrams.contains(bg)).count();
    (2 * matches) as f64 / (a_bigrams.len() + b_bigrams.len()) as f64
}

/// The clip resolver engine.
#[derive(Debug, Clone)]
pub struct ClipResolver {
    /// Configuration.
    config: ResolverConfig,
    /// Indexed media files.
    media_index: Vec<MediaEntry>,
    /// Reel name to path mapping.
    reel_map: HashMap<String, PathBuf>,
}

impl ClipResolver {
    /// Create a new resolver with the given configuration.
    #[must_use]
    pub fn new(config: ResolverConfig) -> Self {
        Self {
            config,
            media_index: Vec::new(),
            reel_map: HashMap::new(),
        }
    }

    /// Add a media file to the search index.
    pub fn add_media(&mut self, path: PathBuf) {
        let entry = MediaEntry::from_path(path);
        if self.config.media_extensions.contains(&entry.extension) {
            self.media_index.push(entry);
        }
    }

    /// Add a reel name mapping.
    pub fn add_reel_mapping(&mut self, reel_name: String, path: PathBuf) {
        self.reel_map.insert(reel_name, path);
    }

    /// Number of indexed media files.
    #[must_use]
    pub fn media_count(&self) -> usize {
        self.media_index.len()
    }

    /// Resolve a single clip reference.
    #[must_use]
    pub fn resolve(&self, clip: &ClipRef) -> ResolveResult {
        let mut alternatives = Vec::new();

        for strategy in &self.config.strategies {
            match strategy {
                ResolveStrategy::ExactPath => {
                    if let Some(ref orig) = clip.original_path {
                        let path = PathBuf::from(orig);
                        if self.media_index.iter().any(|e| e.path == path) {
                            return ResolveResult::resolved(
                                clip.id.clone(),
                                path,
                                ResolveStrategy::ExactPath,
                                1.0,
                            );
                        }
                    }
                }
                ResolveStrategy::FileName => {
                    if let Some(filename) = clip.original_filename() {
                        let matches: Vec<&MediaEntry> = self
                            .media_index
                            .iter()
                            .filter(|e| e.filename == filename)
                            .collect();
                        if matches.len() == 1 {
                            return ResolveResult::resolved(
                                clip.id.clone(),
                                matches[0].path.clone(),
                                ResolveStrategy::FileName,
                                0.95,
                            );
                        }
                        for m in &matches {
                            alternatives.push(m.path.clone());
                        }
                    }
                }
                ResolveStrategy::Stem => {
                    if let Some(stem) = clip.original_stem() {
                        let matches: Vec<&MediaEntry> =
                            self.media_index.iter().filter(|e| e.stem == stem).collect();
                        if matches.len() == 1 {
                            return ResolveResult::resolved(
                                clip.id.clone(),
                                matches[0].path.clone(),
                                ResolveStrategy::Stem,
                                0.9,
                            );
                        }
                        for m in &matches {
                            if !alternatives.contains(&m.path) {
                                alternatives.push(m.path.clone());
                            }
                        }
                    }
                }
                ResolveStrategy::ReelName => {
                    if let Some(path) = self.reel_map.get(&clip.reel_name) {
                        return ResolveResult::resolved(
                            clip.id.clone(),
                            path.clone(),
                            ResolveStrategy::ReelName,
                            0.85,
                        );
                    }
                }
                ResolveStrategy::Fuzzy => {
                    if let Some(stem) = clip.original_stem() {
                        let mut best_score = 0.0_f64;
                        let mut best_entry: Option<&MediaEntry> = None;
                        for entry in &self.media_index {
                            let score = string_similarity(stem, &entry.stem);
                            if score > best_score {
                                best_score = score;
                                best_entry = Some(entry);
                            }
                        }
                        if best_score >= self.config.fuzzy_threshold {
                            if let Some(entry) = best_entry {
                                let mut result = ResolveResult::resolved(
                                    clip.id.clone(),
                                    entry.path.clone(),
                                    ResolveStrategy::Fuzzy,
                                    best_score,
                                );
                                alternatives.truncate(self.config.max_alternatives);
                                result.alternatives = alternatives;
                                return result;
                            }
                        }
                    }
                }
                ResolveStrategy::Manual => {
                    // Manual resolution is handled externally
                }
            }
        }

        let mut result = ResolveResult::unresolved(clip.id.clone());
        alternatives.truncate(self.config.max_alternatives);
        result.alternatives = alternatives;
        result
    }

    /// Resolve a batch of clip references.
    #[must_use]
    pub fn resolve_batch(&self, clips: &[ClipRef]) -> Vec<ResolveResult> {
        clips.iter().map(|c| self.resolve(c)).collect()
    }

    /// Summary statistics for batch resolution.
    #[must_use]
    pub fn batch_summary(results: &[ResolveResult]) -> ResolveSummary {
        let total = results.len();
        let resolved = results.iter().filter(|r| r.resolved).count();
        let by_strategy = {
            let mut map: HashMap<ResolveStrategy, usize> = HashMap::new();
            for r in results.iter().filter(|r| r.resolved) {
                if let Some(s) = r.strategy {
                    *map.entry(s).or_insert(0) += 1;
                }
            }
            map
        };
        let avg_confidence = if resolved > 0 {
            #[allow(clippy::cast_precision_loss)]
            let sum: f64 = results
                .iter()
                .filter(|r| r.resolved)
                .map(|r| r.confidence)
                .sum();
            #[allow(clippy::cast_precision_loss)]
            let avg = sum / resolved as f64;
            avg
        } else {
            0.0
        };
        ResolveSummary {
            total,
            resolved,
            unresolved: total - resolved,
            by_strategy,
            avg_confidence,
        }
    }
}

/// Summary of batch resolution.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ResolveSummary {
    /// Total clips processed.
    pub total: usize,
    /// Number resolved.
    pub resolved: usize,
    /// Number unresolved.
    pub unresolved: usize,
    /// Breakdown by strategy.
    pub by_strategy: HashMap<ResolveStrategy, usize>,
    /// Average confidence of resolved clips.
    pub avg_confidence: f64,
}

impl ResolveSummary {
    /// Resolution rate as percentage.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn resolve_rate(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        (self.resolved as f64 / self.total as f64) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_resolver() -> ClipResolver {
        let mut resolver = ClipResolver::new(ResolverConfig::default());
        resolver.add_media(PathBuf::from("/media/A001_C001.mxf"));
        resolver.add_media(PathBuf::from("/media/A001_C002.mxf"));
        resolver.add_media(PathBuf::from("/media/B001_C001.mov"));
        resolver
    }

    #[test]
    fn test_resolve_exact_path() {
        let resolver = make_resolver();
        let clip = ClipRef::new(
            "1".to_string(),
            "REEL01".to_string(),
            "01:00:00:00".to_string(),
            "01:00:10:00".to_string(),
        )
        .with_original_path("/media/A001_C001.mxf".to_string());
        let result = resolver.resolve(&clip);
        assert!(result.resolved);
        assert_eq!(result.strategy, Some(ResolveStrategy::ExactPath));
    }

    #[test]
    fn test_resolve_filename() {
        let resolver = make_resolver();
        let clip = ClipRef::new(
            "1".to_string(),
            "REEL01".to_string(),
            "01:00:00:00".to_string(),
            "01:00:10:00".to_string(),
        )
        .with_original_path("/other/path/A001_C001.mxf".to_string());
        let result = resolver.resolve(&clip);
        assert!(result.resolved);
        assert_eq!(result.strategy, Some(ResolveStrategy::FileName));
    }

    #[test]
    fn test_resolve_stem() {
        let resolver = make_resolver();
        let clip = ClipRef::new(
            "1".to_string(),
            "REEL01".to_string(),
            "01:00:00:00".to_string(),
            "01:00:10:00".to_string(),
        )
        .with_original_path("/other/B001_C001.mp4".to_string());
        let result = resolver.resolve(&clip);
        assert!(result.resolved);
        assert_eq!(result.strategy, Some(ResolveStrategy::Stem));
    }

    #[test]
    fn test_resolve_reel_name() {
        let mut resolver = make_resolver();
        resolver.add_reel_mapping("REEL01".to_string(), PathBuf::from("/media/A001_C001.mxf"));
        let clip = ClipRef::new(
            "1".to_string(),
            "REEL01".to_string(),
            "01:00:00:00".to_string(),
            "01:00:10:00".to_string(),
        );
        let result = resolver.resolve(&clip);
        assert!(result.resolved);
        assert_eq!(result.strategy, Some(ResolveStrategy::ReelName));
    }

    #[test]
    fn test_resolve_unresolved() {
        let resolver = make_resolver();
        let clip = ClipRef::new(
            "1".to_string(),
            "UNKNOWN".to_string(),
            "01:00:00:00".to_string(),
            "01:00:10:00".to_string(),
        )
        .with_original_path("/nonexistent/ZZZZ.dpx".to_string());
        let result = resolver.resolve(&clip);
        assert!(!result.resolved);
    }

    #[test]
    fn test_string_similarity_identical() {
        assert!((string_similarity("hello", "hello") - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_string_similarity_different() {
        let sim = string_similarity("hello", "world");
        assert!(sim < 0.5);
    }

    #[test]
    fn test_string_similarity_short() {
        assert!(string_similarity("a", "b").abs() < f64::EPSILON);
    }

    #[test]
    fn test_media_entry_from_path() {
        let entry = MediaEntry::from_path(PathBuf::from("/media/clip001.mxf"));
        assert_eq!(entry.filename, "clip001.mxf");
        assert_eq!(entry.stem, "clip001");
        assert_eq!(entry.extension, "mxf");
    }

    #[test]
    fn test_resolve_batch() {
        let resolver = make_resolver();
        let clips = vec![
            ClipRef::new(
                "1".to_string(),
                "R1".to_string(),
                "01:00:00:00".to_string(),
                "01:00:10:00".to_string(),
            )
            .with_original_path("/media/A001_C001.mxf".to_string()),
            ClipRef::new(
                "2".to_string(),
                "R2".to_string(),
                "01:00:10:00".to_string(),
                "01:00:20:00".to_string(),
            )
            .with_original_path("/nonexistent/ZZZ.dpx".to_string()),
        ];
        let results = resolver.resolve_batch(&clips);
        assert_eq!(results.len(), 2);
        assert!(results[0].resolved);
        assert!(!results[1].resolved);
    }

    #[test]
    fn test_batch_summary() {
        let resolver = make_resolver();
        let clips = vec![
            ClipRef::new(
                "1".to_string(),
                "R1".to_string(),
                "01:00:00:00".to_string(),
                "01:00:10:00".to_string(),
            )
            .with_original_path("/media/A001_C001.mxf".to_string()),
            ClipRef::new(
                "2".to_string(),
                "R2".to_string(),
                "01:00:10:00".to_string(),
                "01:00:20:00".to_string(),
            ),
        ];
        let results = resolver.resolve_batch(&clips);
        let summary = ClipResolver::batch_summary(&results);
        assert_eq!(summary.total, 2);
        assert_eq!(summary.resolved, 1);
        assert!(summary.resolve_rate() > 40.0);
    }

    #[test]
    fn test_resolve_strategy_display() {
        assert_eq!(ResolveStrategy::ExactPath.to_string(), "exact_path");
        assert_eq!(ResolveStrategy::Fuzzy.to_string(), "fuzzy");
    }

    #[test]
    fn test_clip_ref_original_filename() {
        let clip = ClipRef::new(
            "1".to_string(),
            "R1".to_string(),
            "00:00:00:00".to_string(),
            "00:00:10:00".to_string(),
        )
        .with_original_path("/path/to/clip.mxf".to_string());
        assert_eq!(clip.original_filename(), Some("clip.mxf"));
        assert_eq!(clip.original_stem(), Some("clip"));
    }
}
