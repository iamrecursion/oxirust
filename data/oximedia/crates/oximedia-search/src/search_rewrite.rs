#![allow(dead_code)]
//! Query rewriting and expansion engine for search optimization.
//!
//! This module provides automatic query rewriting capabilities including
//! synonym expansion, stemming-based expansion, typo correction integration,
//! and boolean query normalization to improve search recall and precision.

use std::collections::HashMap;

/// Strategy for rewriting a query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RewriteStrategy {
    /// Expand query with synonyms from a dictionary.
    SynonymExpansion,
    /// Apply stemming to broaden matches.
    StemExpansion,
    /// Remove duplicate or redundant terms.
    Deduplication,
    /// Normalize boolean operators (AND/OR/NOT).
    BooleanNormalize,
    /// Boost exact phrases over individual terms.
    PhraseBoost,
    /// Rewrite wildcard patterns for efficiency.
    WildcardOptimize,
}

/// A single rewrite rule mapping a source term to replacements.
#[derive(Debug, Clone)]
pub struct RewriteRule {
    /// The original term to match.
    pub source: String,
    /// Replacement terms (may include the original).
    pub replacements: Vec<String>,
    /// Weight applied to replacement terms relative to original.
    pub boost: f64,
}

impl RewriteRule {
    /// Create a new rewrite rule.
    #[must_use]
    pub fn new(source: &str, replacements: Vec<String>, boost: f64) -> Self {
        Self {
            source: source.to_lowercase(),
            replacements,
            boost,
        }
    }
}

/// A rewritten query token with optional boost.
#[derive(Debug, Clone, PartialEq)]
pub struct RewrittenToken {
    /// The token text.
    pub text: String,
    /// Boost factor for scoring.
    pub boost: f64,
    /// Whether this token was added by rewriting (not in original query).
    pub is_expansion: bool,
}

impl RewrittenToken {
    /// Create a new rewritten token.
    #[must_use]
    pub fn new(text: &str, boost: f64, is_expansion: bool) -> Self {
        Self {
            text: text.to_string(),
            boost,
            is_expansion,
        }
    }
}

/// Result of a query rewrite operation.
#[derive(Debug, Clone)]
pub struct RewriteResult {
    /// Original query string.
    pub original: String,
    /// Rewritten tokens.
    pub tokens: Vec<RewrittenToken>,
    /// Strategies that were applied.
    pub strategies_applied: Vec<RewriteStrategy>,
    /// Number of expansions performed.
    pub expansion_count: usize,
}

/// Synonym dictionary for query expansion.
#[derive(Debug, Clone)]
pub struct SynonymDictionary {
    /// Map from term to list of synonyms.
    entries: HashMap<String, Vec<String>>,
}

impl SynonymDictionary {
    /// Create an empty synonym dictionary.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Add a synonym group where all terms are interchangeable.
    pub fn add_group(&mut self, terms: &[&str]) {
        for &term in terms {
            let key = term.to_lowercase();
            let synonyms: Vec<String> = terms
                .iter()
                .filter(|&&t| t.to_lowercase() != key)
                .map(|t| t.to_lowercase())
                .collect();
            self.entries.insert(key, synonyms);
        }
    }

    /// Look up synonyms for a term.
    #[must_use]
    pub fn lookup(&self, term: &str) -> Option<&[String]> {
        self.entries
            .get(&term.to_lowercase())
            .map(std::vec::Vec::as_slice)
    }

    /// Return total number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if dictionary is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for SynonymDictionary {
    fn default() -> Self {
        Self::new()
    }
}

impl SynonymDictionary {
    /// Create a pre-populated dictionary with common media production synonyms.
    ///
    /// Covers audio, video, image, codec, and workflow terminology commonly
    /// used in media asset management searches.
    #[must_use]
    pub fn media_defaults() -> Self {
        let mut dict = Self::new();

        // Audio synonyms
        dict.add_group(&["audio", "sound", "music", "soundtrack"]);
        dict.add_group(&["voice", "speech", "narration", "dialogue"]);
        dict.add_group(&["noise", "static", "hiss", "hum"]);
        dict.add_group(&["loud", "noisy", "amplified"]);
        dict.add_group(&["quiet", "silent", "muted", "mute"]);
        dict.add_group(&["bass", "low-frequency", "subwoofer"]);
        dict.add_group(&["treble", "high-frequency", "highs"]);

        // Video synonyms
        dict.add_group(&["video", "clip", "footage", "recording"]);
        dict.add_group(&["frame", "still", "snapshot", "capture"]);
        dict.add_group(&["slow-motion", "slowmo", "slow-mo", "timelapse"]);
        dict.add_group(&["cut", "edit", "trim", "splice"]);
        dict.add_group(&["transition", "dissolve", "fade", "wipe"]);

        // Image synonyms
        dict.add_group(&["image", "photo", "picture", "photograph"]);
        dict.add_group(&["thumbnail", "preview", "icon"]);
        dict.add_group(&["portrait", "headshot", "closeup"]);
        dict.add_group(&["landscape", "panorama", "wide-angle"]);

        // Resolution / quality synonyms
        dict.add_group(&["high-definition", "hd", "1080p", "fullhd"]);
        dict.add_group(&["ultra-hd", "uhd", "4k", "2160p"]);
        dict.add_group(&["standard-definition", "sd", "480p"]);
        dict.add_group(&["high-quality", "hq", "lossless"]);
        dict.add_group(&["low-quality", "lq", "lossy", "compressed"]);

        // Codec synonyms
        dict.add_group(&["encode", "compress", "transcode"]);
        dict.add_group(&["decode", "decompress", "playback"]);
        dict.add_group(&["codec", "encoder", "decoder", "compressor"]);
        dict.add_group(&["bitrate", "datarate", "bandwidth"]);

        // Color synonyms
        dict.add_group(&["color", "colour", "hue", "tint"]);
        dict.add_group(&["brightness", "luminance", "exposure"]);
        dict.add_group(&["contrast", "dynamic-range"]);
        dict.add_group(&["saturation", "chroma", "vibrance"]);
        dict.add_group(&["grayscale", "monochrome", "black-and-white", "bw"]);

        // Production workflow synonyms
        dict.add_group(&["render", "export", "output", "publish"]);
        dict.add_group(&["import", "ingest", "upload"]);
        dict.add_group(&["project", "timeline", "sequence"]);
        dict.add_group(&["effect", "filter", "plugin"]);
        dict.add_group(&["subtitle", "caption", "closed-caption"]);
        dict.add_group(&["watermark", "overlay", "logo"]);

        dict
    }
}

/// A query rewriter pre-configured with media industry synonyms.
///
/// Convenience constructor that sets up the [`QueryRewriter`] with the
/// [`SynonymDictionary::media_defaults`] dictionary and all standard
/// rewrite strategies enabled.
impl QueryRewriter {
    /// Create a query rewriter pre-loaded with media production synonyms.
    #[must_use]
    pub fn with_media_synonyms() -> Self {
        Self::new().with_synonyms(SynonymDictionary::media_defaults())
    }

    /// Rewrite a query and return the expanded query as a single string.
    ///
    /// Tokens are joined with spaces. Expansion tokens are wrapped in
    /// parenthesised OR groups for boolean-query engines, e.g.:
    /// `"audio" -> "audio OR sound OR music OR soundtrack"`
    #[must_use]
    pub fn rewrite_to_string(&self, query: &str) -> String {
        let result = self.rewrite(query);
        if result.tokens.is_empty() {
            return String::new();
        }

        let mut groups: Vec<Vec<&RewrittenToken>> = Vec::new();
        let mut current_group: Vec<&RewrittenToken> = Vec::new();

        for token in &result.tokens {
            if token.is_expansion {
                current_group.push(token);
            } else {
                if !current_group.is_empty() {
                    groups.push(current_group);
                    current_group = Vec::new();
                }
                current_group.push(token);
            }
        }
        if !current_group.is_empty() {
            groups.push(current_group);
        }

        let mut parts = Vec::new();
        for group in &groups {
            if group.len() == 1 {
                parts.push(group[0].text.clone());
            } else {
                // Group the original term with its expansions using OR
                let or_terms: Vec<&str> = group.iter().map(|t| t.text.as_str()).collect();
                parts.push(format!("({})", or_terms.join(" OR ")));
            }
        }
        parts.join(" ")
    }
}

/// Query rewriter that applies various strategies to improve search quality.
#[derive(Debug, Clone)]
pub struct QueryRewriter {
    /// Synonym dictionary for expansion.
    synonyms: SynonymDictionary,
    /// Active strategies to apply in order.
    strategies: Vec<RewriteStrategy>,
    /// Maximum number of expansions per term.
    max_expansions_per_term: usize,
    /// Boost factor for synonym expansions.
    synonym_boost: f64,
    /// Stop words to remove during normalization.
    stop_words: Vec<String>,
}

impl QueryRewriter {
    /// Create a new query rewriter with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            synonyms: SynonymDictionary::new(),
            strategies: vec![
                RewriteStrategy::Deduplication,
                RewriteStrategy::BooleanNormalize,
                RewriteStrategy::SynonymExpansion,
            ],
            max_expansions_per_term: 3,
            synonym_boost: 0.8,
            stop_words: vec![
                "the".into(),
                "a".into(),
                "an".into(),
                "is".into(),
                "of".into(),
                "in".into(),
            ],
        }
    }

    /// Set the synonym dictionary.
    #[must_use]
    pub fn with_synonyms(mut self, synonyms: SynonymDictionary) -> Self {
        self.synonyms = synonyms;
        self
    }

    /// Set the active rewrite strategies.
    #[must_use]
    pub fn with_strategies(mut self, strategies: Vec<RewriteStrategy>) -> Self {
        self.strategies = strategies;
        self
    }

    /// Set maximum expansions per term.
    #[must_use]
    pub fn with_max_expansions(mut self, max: usize) -> Self {
        self.max_expansions_per_term = max;
        self
    }

    /// Rewrite a query string, returning expanded tokens.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn rewrite(&self, query: &str) -> RewriteResult {
        let raw_tokens: Vec<&str> = query.split_whitespace().collect();
        let mut tokens: Vec<RewrittenToken> = raw_tokens
            .iter()
            .map(|t| RewrittenToken::new(&t.to_lowercase(), 1.0, false))
            .collect();

        let mut strategies_applied = Vec::new();
        let mut expansion_count = 0_usize;

        for &strategy in &self.strategies {
            match strategy {
                RewriteStrategy::Deduplication => {
                    let before = tokens.len();
                    tokens = self.deduplicate(tokens);
                    if tokens.len() != before {
                        strategies_applied.push(strategy);
                    }
                }
                RewriteStrategy::BooleanNormalize => {
                    let changed = self.normalize_booleans(&mut tokens);
                    if changed {
                        strategies_applied.push(strategy);
                    }
                }
                RewriteStrategy::SynonymExpansion => {
                    let (expanded, count) = self.expand_synonyms(tokens);
                    tokens = expanded;
                    if count > 0 {
                        expansion_count += count;
                        strategies_applied.push(strategy);
                    }
                }
                RewriteStrategy::StemExpansion => {
                    let (expanded, count) = self.expand_stems(tokens);
                    tokens = expanded;
                    if count > 0 {
                        expansion_count += count;
                        strategies_applied.push(strategy);
                    }
                }
                RewriteStrategy::PhraseBoost => {
                    self.apply_phrase_boost(&mut tokens);
                    strategies_applied.push(strategy);
                }
                RewriteStrategy::WildcardOptimize => {
                    let changed = self.optimize_wildcards(&mut tokens);
                    if changed {
                        strategies_applied.push(strategy);
                    }
                }
            }
        }

        RewriteResult {
            original: query.to_string(),
            tokens,
            strategies_applied,
            expansion_count,
        }
    }

    /// Remove duplicate tokens while preserving order.
    fn deduplicate(&self, tokens: Vec<RewrittenToken>) -> Vec<RewrittenToken> {
        let mut seen = std::collections::HashSet::new();
        tokens
            .into_iter()
            .filter(|t| seen.insert(t.text.clone()))
            .collect()
    }

    /// Normalize boolean operators to uppercase.
    fn normalize_booleans(&self, tokens: &mut [RewrittenToken]) -> bool {
        let mut changed = false;
        for token in tokens.iter_mut() {
            let lower = token.text.to_lowercase();
            if lower == "and" || lower == "or" || lower == "not" {
                let upper = lower.to_uppercase();
                if token.text != upper {
                    token.text = upper;
                    changed = true;
                }
            }
        }
        changed
    }

    /// Expand tokens with synonyms.
    fn expand_synonyms(&self, tokens: Vec<RewrittenToken>) -> (Vec<RewrittenToken>, usize) {
        let mut result = Vec::new();
        let mut count = 0_usize;
        for token in tokens {
            result.push(token.clone());
            if !token.is_expansion {
                if let Some(syns) = self.synonyms.lookup(&token.text) {
                    let take = syns.len().min(self.max_expansions_per_term);
                    for syn in &syns[..take] {
                        result.push(RewrittenToken::new(syn, self.synonym_boost, true));
                        count += 1;
                    }
                }
            }
        }
        (result, count)
    }

    /// Expand tokens by simple stem suffix stripping (lightweight).
    fn expand_stems(&self, tokens: Vec<RewrittenToken>) -> (Vec<RewrittenToken>, usize) {
        let mut result = Vec::new();
        let mut count = 0_usize;
        for token in tokens {
            result.push(token.clone());
            if !token.is_expansion && token.text.len() > 4 {
                // Simple suffix stripping for common English endings
                for suffix in &["ing", "tion", "ed", "ly", "er", "est"] {
                    if token.text.ends_with(suffix) {
                        let stem_len = token.text.len() - suffix.len();
                        if stem_len >= 3 {
                            let stem = &token.text[..stem_len];
                            result.push(RewrittenToken::new(stem, 0.6, true));
                            count += 1;
                            break;
                        }
                    }
                }
            }
        }
        (result, count)
    }

    /// Boost consecutive non-boolean tokens as a potential phrase.
    fn apply_phrase_boost(&self, tokens: &mut [RewrittenToken]) {
        let booleans = ["AND", "OR", "NOT"];
        for token in tokens.iter_mut() {
            if !booleans.contains(&token.text.as_str()) && !token.is_expansion {
                token.boost *= 1.2;
            }
        }
    }

    /// Optimize wildcard patterns: remove leading wildcards.
    fn optimize_wildcards(&self, tokens: &mut [RewrittenToken]) -> bool {
        let mut changed = false;
        for token in tokens.iter_mut() {
            if token.text.starts_with('*') && token.text.len() > 1 {
                token.text = token.text.trim_start_matches('*').to_string();
                changed = true;
            }
        }
        changed
    }

    /// Remove stop words from the query tokens.
    #[must_use]
    pub fn remove_stop_words(&self, tokens: Vec<RewrittenToken>) -> Vec<RewrittenToken> {
        tokens
            .into_iter()
            .filter(|t| !self.stop_words.contains(&t.text))
            .collect()
    }

    /// Check if a term is a stop word.
    #[must_use]
    pub fn is_stop_word(&self, term: &str) -> bool {
        self.stop_words.contains(&term.to_lowercase())
    }
}

impl Default for QueryRewriter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synonym_dictionary_add_group() {
        let mut dict = SynonymDictionary::new();
        dict.add_group(&["fast", "quick", "rapid"]);
        assert_eq!(dict.len(), 3);
        let syns = dict.lookup("fast").expect("should succeed in test");
        assert!(syns.contains(&"quick".to_string()));
        assert!(syns.contains(&"rapid".to_string()));
    }

    #[test]
    fn test_synonym_dictionary_empty() {
        let dict = SynonymDictionary::new();
        assert!(dict.is_empty());
        assert_eq!(dict.len(), 0);
        assert!(dict.lookup("hello").is_none());
    }

    #[test]
    fn test_rewrite_basic_query() {
        let rewriter = QueryRewriter::new();
        let result = rewriter.rewrite("hello world");
        assert_eq!(result.original, "hello world");
        assert_eq!(result.tokens.len(), 2);
        assert_eq!(result.tokens[0].text, "hello");
        assert_eq!(result.tokens[1].text, "world");
    }

    #[test]
    fn test_deduplication() {
        let rewriter = QueryRewriter::new().with_strategies(vec![RewriteStrategy::Deduplication]);
        let result = rewriter.rewrite("cat dog cat bird dog");
        assert_eq!(result.tokens.len(), 3);
        assert_eq!(result.tokens[0].text, "cat");
        assert_eq!(result.tokens[1].text, "dog");
        assert_eq!(result.tokens[2].text, "bird");
    }

    #[test]
    fn test_boolean_normalization() {
        let rewriter =
            QueryRewriter::new().with_strategies(vec![RewriteStrategy::BooleanNormalize]);
        let result = rewriter.rewrite("cat and dog or bird not fish");
        let texts: Vec<&str> = result.tokens.iter().map(|t| t.text.as_str()).collect();
        assert!(texts.contains(&"AND"));
        assert!(texts.contains(&"OR"));
        assert!(texts.contains(&"NOT"));
    }

    #[test]
    fn test_synonym_expansion() {
        let mut dict = SynonymDictionary::new();
        dict.add_group(&["video", "clip", "footage"]);
        let rewriter = QueryRewriter::new()
            .with_synonyms(dict)
            .with_strategies(vec![RewriteStrategy::SynonymExpansion]);
        let result = rewriter.rewrite("video editing");
        assert!(result.expansion_count > 0);
        let texts: Vec<&str> = result.tokens.iter().map(|t| t.text.as_str()).collect();
        assert!(texts.contains(&"video"));
        assert!(texts.contains(&"clip") || texts.contains(&"footage"));
    }

    #[test]
    fn test_stem_expansion() {
        let rewriter = QueryRewriter::new().with_strategies(vec![RewriteStrategy::StemExpansion]);
        let result = rewriter.rewrite("encoding editing");
        assert!(result.expansion_count > 0);
        let texts: Vec<&str> = result.tokens.iter().map(|t| t.text.as_str()).collect();
        assert!(texts.contains(&"encod") || texts.contains(&"edit"));
    }

    #[test]
    fn test_phrase_boost() {
        let rewriter = QueryRewriter::new().with_strategies(vec![RewriteStrategy::PhraseBoost]);
        let result = rewriter.rewrite("color correction");
        for token in &result.tokens {
            assert!(token.boost > 1.0);
        }
    }

    #[test]
    fn test_wildcard_optimize() {
        let rewriter =
            QueryRewriter::new().with_strategies(vec![RewriteStrategy::WildcardOptimize]);
        let result = rewriter.rewrite("*codec *format");
        assert_eq!(result.tokens[0].text, "codec");
        assert_eq!(result.tokens[1].text, "format");
    }

    #[test]
    fn test_stop_word_removal() {
        let rewriter = QueryRewriter::new();
        let tokens = vec![
            RewrittenToken::new("the", 1.0, false),
            RewrittenToken::new("video", 1.0, false),
            RewrittenToken::new("is", 1.0, false),
            RewrittenToken::new("good", 1.0, false),
        ];
        let filtered = rewriter.remove_stop_words(tokens);
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].text, "video");
        assert_eq!(filtered[1].text, "good");
    }

    #[test]
    fn test_is_stop_word() {
        let rewriter = QueryRewriter::new();
        assert!(rewriter.is_stop_word("the"));
        assert!(rewriter.is_stop_word("a"));
        assert!(!rewriter.is_stop_word("video"));
    }

    #[test]
    fn test_max_expansions_limit() {
        let mut dict = SynonymDictionary::new();
        dict.add_group(&["a", "b", "c", "d", "e"]);
        let rewriter = QueryRewriter::new()
            .with_synonyms(dict)
            .with_max_expansions(2)
            .with_strategies(vec![RewriteStrategy::SynonymExpansion]);
        let result = rewriter.rewrite("a");
        // Original + at most 2 expansions
        let expansion_tokens: Vec<_> = result.tokens.iter().filter(|t| t.is_expansion).collect();
        assert!(expansion_tokens.len() <= 2);
    }

    #[test]
    fn test_rewrite_result_fields() {
        let rewriter = QueryRewriter::new().with_strategies(vec![RewriteStrategy::Deduplication]);
        let result = rewriter.rewrite("test test");
        assert_eq!(result.original, "test test");
        assert_eq!(result.tokens.len(), 1);
        assert!(result
            .strategies_applied
            .contains(&RewriteStrategy::Deduplication));
    }

    #[test]
    fn test_rewrite_rule_creation() {
        let rule = RewriteRule::new("Video", vec!["clip".into(), "footage".into()], 0.9);
        assert_eq!(rule.source, "video");
        assert_eq!(rule.replacements.len(), 2);
        assert!((rule.boost - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_empty_query_rewrite() {
        let rewriter = QueryRewriter::new();
        let result = rewriter.rewrite("");
        assert!(result.tokens.is_empty());
        assert_eq!(result.expansion_count, 0);
    }

    // ── Media synonym dictionary tests ──

    #[test]
    fn test_media_defaults_not_empty() {
        let dict = SynonymDictionary::media_defaults();
        assert!(!dict.is_empty());
        assert!(dict.len() > 30); // We defined many groups
    }

    #[test]
    fn test_media_defaults_audio_synonyms() {
        let dict = SynonymDictionary::media_defaults();
        let syns = dict.lookup("audio").expect("audio should have synonyms");
        assert!(syns.contains(&"sound".to_string()));
        assert!(syns.contains(&"music".to_string()));
        assert!(syns.contains(&"soundtrack".to_string()));
    }

    #[test]
    fn test_media_defaults_video_synonyms() {
        let dict = SynonymDictionary::media_defaults();
        let syns = dict.lookup("video").expect("video should have synonyms");
        assert!(syns.contains(&"clip".to_string()));
        assert!(syns.contains(&"footage".to_string()));
        assert!(syns.contains(&"recording".to_string()));
    }

    #[test]
    fn test_media_defaults_image_synonyms() {
        let dict = SynonymDictionary::media_defaults();
        let syns = dict.lookup("image").expect("image should have synonyms");
        assert!(syns.contains(&"photo".to_string()));
        assert!(syns.contains(&"picture".to_string()));
    }

    #[test]
    fn test_media_defaults_color_synonyms() {
        let dict = SynonymDictionary::media_defaults();
        let syns = dict.lookup("color").expect("color should have synonyms");
        assert!(syns.contains(&"colour".to_string()));
        assert!(syns.contains(&"hue".to_string()));
    }

    #[test]
    fn test_media_defaults_bidirectional() {
        let dict = SynonymDictionary::media_defaults();
        // Synonym groups are bidirectional
        let sound_syns = dict.lookup("sound").expect("sound should have synonyms");
        assert!(sound_syns.contains(&"audio".to_string()));
        let photo_syns = dict.lookup("photo").expect("photo should have synonyms");
        assert!(photo_syns.contains(&"image".to_string()));
    }

    #[test]
    fn test_media_defaults_codec_synonyms() {
        let dict = SynonymDictionary::media_defaults();
        let syns = dict.lookup("codec").expect("codec should have synonyms");
        assert!(syns.contains(&"encoder".to_string()));
        assert!(syns.contains(&"decoder".to_string()));
    }

    #[test]
    fn test_with_media_synonyms_rewriter() {
        let rewriter = QueryRewriter::with_media_synonyms();
        let result = rewriter.rewrite("audio editing");
        assert!(result.expansion_count > 0);
        let texts: Vec<&str> = result.tokens.iter().map(|t| t.text.as_str()).collect();
        // Should contain at least one synonym for "audio"
        assert!(
            texts.contains(&"sound") || texts.contains(&"music") || texts.contains(&"soundtrack")
        );
    }

    #[test]
    fn test_rewrite_to_string_basic() {
        let rewriter = QueryRewriter::new().with_strategies(vec![]);
        let output = rewriter.rewrite_to_string("hello world");
        assert_eq!(output, "hello world");
    }

    #[test]
    fn test_rewrite_to_string_with_expansions() {
        let mut dict = SynonymDictionary::new();
        dict.add_group(&["video", "clip", "footage"]);
        let rewriter = QueryRewriter::new()
            .with_synonyms(dict)
            .with_strategies(vec![RewriteStrategy::SynonymExpansion]);
        let output = rewriter.rewrite_to_string("video");
        assert!(output.contains("OR"));
        assert!(output.contains("video"));
        assert!(output.contains("clip") || output.contains("footage"));
    }

    #[test]
    fn test_rewrite_to_string_empty() {
        let rewriter = QueryRewriter::new();
        let output = rewriter.rewrite_to_string("");
        assert!(output.is_empty());
    }

    #[test]
    fn test_media_defaults_resolution_synonyms() {
        let dict = SynonymDictionary::media_defaults();
        let syns = dict.lookup("hd").expect("hd should have synonyms");
        assert!(syns.contains(&"high-definition".to_string()));
        assert!(syns.contains(&"1080p".to_string()));
    }

    #[test]
    fn test_media_defaults_workflow_synonyms() {
        let dict = SynonymDictionary::media_defaults();
        let render_syns = dict.lookup("render").expect("render should have synonyms");
        assert!(render_syns.contains(&"export".to_string()));
        assert!(render_syns.contains(&"output".to_string()));
        let sub_syns = dict
            .lookup("subtitle")
            .expect("subtitle should have synonyms");
        assert!(sub_syns.contains(&"caption".to_string()));
    }

    #[test]
    fn test_media_rewriter_no_expansion_for_unknown_term() {
        let rewriter = QueryRewriter::with_media_synonyms();
        let result = rewriter.rewrite("xyznonexistent");
        assert_eq!(result.expansion_count, 0);
    }

    #[test]
    fn test_media_defaults_grayscale_synonyms() {
        let dict = SynonymDictionary::media_defaults();
        let syns = dict
            .lookup("grayscale")
            .expect("grayscale should have synonyms");
        assert!(syns.contains(&"monochrome".to_string()));
        assert!(syns.contains(&"black-and-white".to_string()));
        assert!(syns.contains(&"bw".to_string()));
    }
}
