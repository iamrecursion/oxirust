#![allow(dead_code)]

//! Search pipeline with configurable phases.
//!
//! Implements a multi-phase search pipeline: parse, expand, execute, rank,
//! and format. Each phase can be customized or skipped independently.

use std::collections::HashMap;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Pipeline phases
// ---------------------------------------------------------------------------

/// Identifies a phase in the search pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelinePhase {
    /// Query parsing and validation.
    Parse,
    /// Query expansion (synonyms, stemming).
    Expand,
    /// Core search execution.
    Execute,
    /// Result ranking and re-ranking.
    Rank,
    /// Result formatting and output.
    Format,
}

impl PipelinePhase {
    /// Return all phases in execution order.
    #[must_use]
    pub fn all_phases() -> &'static [PipelinePhase] {
        &[
            PipelinePhase::Parse,
            PipelinePhase::Expand,
            PipelinePhase::Execute,
            PipelinePhase::Rank,
            PipelinePhase::Format,
        ]
    }

    /// Return a human-readable name for the phase.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            PipelinePhase::Parse => "parse",
            PipelinePhase::Expand => "expand",
            PipelinePhase::Execute => "execute",
            PipelinePhase::Rank => "rank",
            PipelinePhase::Format => "format",
        }
    }
}

// ---------------------------------------------------------------------------
// Parsed query
// ---------------------------------------------------------------------------

/// A parsed search query after the Parse phase.
#[derive(Debug, Clone)]
pub struct ParsedQuery {
    /// Original raw query string.
    pub raw: String,
    /// Extracted terms after tokenization.
    pub terms: Vec<String>,
    /// Negated terms (prefixed with `-`).
    pub excluded_terms: Vec<String>,
    /// Exact phrase matches (quoted strings).
    pub phrases: Vec<String>,
    /// Whether the query is valid.
    pub is_valid: bool,
}

/// Parse a raw query string into structured form.
///
/// Supports:
/// - Quoted phrases: `"exact match"`
/// - Excluded terms: `-unwanted`
/// - Plain terms
#[must_use]
pub fn parse_query(raw: &str) -> ParsedQuery {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return ParsedQuery {
            raw: raw.to_string(),
            terms: Vec::new(),
            excluded_terms: Vec::new(),
            phrases: Vec::new(),
            is_valid: false,
        };
    }

    let mut terms = Vec::new();
    let mut excluded = Vec::new();
    let mut phrases = Vec::new();

    let chars = trimmed.chars().peekable();
    let mut current = String::new();
    let mut in_quote = false;

    for ch in chars {
        if ch == '"' {
            if in_quote {
                // End of phrase
                if !current.is_empty() {
                    phrases.push(current.clone());
                    current.clear();
                }
                in_quote = false;
            } else {
                // Start of phrase
                if !current.is_empty() {
                    terms.push(current.clone());
                    current.clear();
                }
                in_quote = true;
            }
        } else if ch.is_whitespace() && !in_quote {
            if !current.is_empty() {
                if current.starts_with('-') && current.len() > 1 {
                    excluded.push(current[1..].to_string());
                } else {
                    terms.push(current.clone());
                }
                current.clear();
            }
        } else {
            current.push(ch);
        }
    }

    // Flush remaining
    if !current.is_empty() {
        if in_quote {
            // Unterminated quote treated as phrase
            phrases.push(current);
        } else if current.starts_with('-') && current.len() > 1 {
            excluded.push(current[1..].to_string());
        } else {
            terms.push(current);
        }
    }

    ParsedQuery {
        raw: raw.to_string(),
        terms,
        excluded_terms: excluded,
        phrases,
        is_valid: true,
    }
}

// ---------------------------------------------------------------------------
// Query expansion
// ---------------------------------------------------------------------------

/// A synonym mapping for query expansion.
#[derive(Debug, Clone)]
pub struct SynonymMap {
    /// Map from term to list of synonyms.
    entries: HashMap<String, Vec<String>>,
}

impl SynonymMap {
    /// Create an empty synonym map.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Add a synonym group (all terms map to each other).
    pub fn add_group(&mut self, group: &[&str]) {
        for &term in group {
            let synonyms: Vec<String> = group
                .iter()
                .filter(|&&t| t != term)
                .map(|&s| s.to_string())
                .collect();
            self.entries
                .entry(term.to_lowercase())
                .or_default()
                .extend(synonyms);
        }
    }

    /// Get synonyms for a term.
    pub fn synonyms_for(&self, term: &str) -> &[String] {
        self.entries
            .get(&term.to_lowercase())
            .map_or(&[], Vec::as_slice)
    }

    /// Expand a parsed query with synonyms, returning the additional terms.
    #[must_use]
    pub fn expand(&self, query: &ParsedQuery) -> Vec<String> {
        let mut extra = Vec::new();
        for term in &query.terms {
            for syn in self.synonyms_for(term) {
                if !query.terms.contains(syn) && !extra.contains(syn) {
                    extra.push(syn.clone());
                }
            }
        }
        extra
    }

    /// Return the total number of synonym entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the synonym map is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for SynonymMap {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Pipeline timing
// ---------------------------------------------------------------------------

/// Timing information for each pipeline phase.
#[derive(Debug, Clone)]
pub struct PipelineTiming {
    /// Duration of each phase in microseconds.
    pub phase_durations_us: HashMap<PipelinePhase, u64>,
    /// Total pipeline duration in microseconds.
    pub total_us: u64,
}

impl PipelineTiming {
    /// Create a new empty timing record.
    #[must_use]
    pub fn new() -> Self {
        Self {
            phase_durations_us: HashMap::new(),
            total_us: 0,
        }
    }

    /// Record the duration for a phase.
    pub fn record(&mut self, phase: PipelinePhase, duration_us: u64) {
        self.phase_durations_us.insert(phase, duration_us);
    }

    /// Finalize total duration.
    pub fn finalize(&mut self) {
        self.total_us = self.phase_durations_us.values().sum();
    }

    /// Get duration for a specific phase.
    #[must_use]
    pub fn phase_duration(&self, phase: PipelinePhase) -> u64 {
        self.phase_durations_us.get(&phase).copied().unwrap_or(0)
    }

    /// Return the slowest phase.
    #[must_use]
    pub fn slowest_phase(&self) -> Option<(PipelinePhase, u64)> {
        self.phase_durations_us
            .iter()
            .max_by_key(|&(_, &dur)| dur)
            .map(|(&phase, &dur)| (phase, dur))
    }
}

impl Default for PipelineTiming {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Pipeline configuration
// ---------------------------------------------------------------------------

/// Configuration for the search pipeline.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Whether to enable query expansion.
    pub enable_expansion: bool,
    /// Whether to enable re-ranking.
    pub enable_reranking: bool,
    /// Maximum number of results to return.
    pub max_results: usize,
    /// Minimum score threshold (results below this are discarded).
    pub min_score: f64,
    /// Enabled phases (all by default).
    pub enabled_phases: Vec<PipelinePhase>,
    /// Whether to collect timing information.
    pub collect_timing: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            enable_expansion: true,
            enable_reranking: true,
            max_results: 100,
            min_score: 0.0,
            enabled_phases: PipelinePhase::all_phases().to_vec(),
            collect_timing: true,
        }
    }
}

impl PipelineConfig {
    /// Check whether a phase is enabled.
    #[must_use]
    pub fn is_phase_enabled(&self, phase: PipelinePhase) -> bool {
        self.enabled_phases.contains(&phase)
    }
}

// ---------------------------------------------------------------------------
// Pipeline executor
// ---------------------------------------------------------------------------

/// A scored search result from the pipeline.
#[derive(Debug, Clone)]
pub struct ScoredResult {
    /// Unique identifier of the matched document.
    pub doc_id: String,
    /// Relevance score.
    pub score: f64,
    /// The terms that matched.
    pub matched_terms: Vec<String>,
}

/// Simulated search pipeline that processes a query through multiple phases.
#[derive(Debug)]
pub struct SearchPipeline {
    /// Pipeline configuration.
    config: PipelineConfig,
    /// Synonym map for expansion.
    synonyms: SynonymMap,
}

impl SearchPipeline {
    /// Create a new search pipeline with the given config.
    #[must_use]
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            config,
            synonyms: SynonymMap::new(),
        }
    }

    /// Create a search pipeline with default config.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self {
            config: PipelineConfig::default(),
            synonyms: SynonymMap::new(),
        }
    }

    /// Set the synonym map for query expansion.
    pub fn set_synonyms(&mut self, synonyms: SynonymMap) {
        self.synonyms = synonyms;
    }

    /// Execute the pipeline on a raw query string.
    ///
    /// Returns scored results and timing information.
    pub fn execute(&self, raw_query: &str) -> (Vec<ScoredResult>, PipelineTiming) {
        let mut timing = PipelineTiming::new();

        // Phase 1: Parse
        let start = Instant::now();
        let parsed = parse_query(raw_query);
        if self.config.collect_timing {
            timing.record(PipelinePhase::Parse, start.elapsed().as_micros() as u64);
        }

        if !parsed.is_valid {
            timing.finalize();
            return (Vec::new(), timing);
        }

        // Phase 2: Expand
        let start = Instant::now();
        let expanded_terms = if self.config.is_phase_enabled(PipelinePhase::Expand)
            && self.config.enable_expansion
        {
            self.synonyms.expand(&parsed)
        } else {
            Vec::new()
        };
        if self.config.collect_timing {
            timing.record(PipelinePhase::Expand, start.elapsed().as_micros() as u64);
        }

        // Phase 3: Execute (simulated — return results based on term count)
        let start = Instant::now();
        let mut results = Vec::new();
        let all_terms: Vec<&str> = parsed
            .terms
            .iter()
            .chain(expanded_terms.iter())
            .map(String::as_str)
            .collect();

        if self.config.is_phase_enabled(PipelinePhase::Execute) && !all_terms.is_empty() {
            // Simulate finding documents proportional to term count
            let n = all_terms.len().min(self.config.max_results);
            for i in 0..n {
                let score = 1.0 / (1.0 + i as f64);
                results.push(ScoredResult {
                    doc_id: format!("doc_{i}"),
                    score,
                    matched_terms: all_terms.iter().map(|s| (*s).to_string()).collect(),
                });
            }
        }
        if self.config.collect_timing {
            timing.record(PipelinePhase::Execute, start.elapsed().as_micros() as u64);
        }

        // Phase 4: Rank (filter by min_score)
        let start = Instant::now();
        if self.config.is_phase_enabled(PipelinePhase::Rank) {
            results.retain(|r| r.score >= self.config.min_score);
            results.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            results.truncate(self.config.max_results);
        }
        if self.config.collect_timing {
            timing.record(PipelinePhase::Rank, start.elapsed().as_micros() as u64);
        }

        // Phase 5: Format (no-op for now)
        let start = Instant::now();
        if self.config.collect_timing {
            timing.record(PipelinePhase::Format, start.elapsed().as_micros() as u64);
        }

        timing.finalize();
        (results, timing)
    }

    /// Return a reference to the pipeline config.
    #[must_use]
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- parse_query tests --

    #[test]
    fn test_parse_empty() {
        let q = parse_query("");
        assert!(!q.is_valid);
        assert!(q.terms.is_empty());
    }

    #[test]
    fn test_parse_single_term() {
        let q = parse_query("hello");
        assert!(q.is_valid);
        assert_eq!(q.terms, vec!["hello"]);
    }

    #[test]
    fn test_parse_multiple_terms() {
        let q = parse_query("quick brown fox");
        assert_eq!(q.terms, vec!["quick", "brown", "fox"]);
    }

    #[test]
    fn test_parse_excluded_terms() {
        let q = parse_query("fox -cat");
        assert_eq!(q.terms, vec!["fox"]);
        assert_eq!(q.excluded_terms, vec!["cat"]);
    }

    #[test]
    fn test_parse_phrases() {
        let q = parse_query("\"quick brown\" fox");
        assert_eq!(q.phrases, vec!["quick brown"]);
        assert_eq!(q.terms, vec!["fox"]);
    }

    #[test]
    fn test_parse_mixed() {
        let q = parse_query("\"hello world\" test -bad");
        assert_eq!(q.phrases, vec!["hello world"]);
        assert_eq!(q.terms, vec!["test"]);
        assert_eq!(q.excluded_terms, vec!["bad"]);
    }

    // -- SynonymMap tests --

    #[test]
    fn test_synonym_map_empty() {
        let map = SynonymMap::new();
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
    }

    #[test]
    fn test_synonym_add_group() {
        let mut map = SynonymMap::new();
        map.add_group(&["fast", "quick", "rapid"]);
        assert!(!map.is_empty());
        let syns = map.synonyms_for("fast");
        assert!(syns.contains(&"quick".to_string()));
        assert!(syns.contains(&"rapid".to_string()));
    }

    #[test]
    fn test_synonym_expand() {
        let mut map = SynonymMap::new();
        map.add_group(&["video", "clip", "footage"]);
        let q = parse_query("video");
        let expanded = map.expand(&q);
        assert!(expanded.contains(&"clip".to_string()));
        assert!(expanded.contains(&"footage".to_string()));
    }

    // -- PipelineTiming tests --

    #[test]
    fn test_timing_record() {
        let mut timing = PipelineTiming::new();
        timing.record(PipelinePhase::Parse, 100);
        timing.record(PipelinePhase::Execute, 500);
        timing.finalize();
        assert_eq!(timing.total_us, 600);
        assert_eq!(timing.phase_duration(PipelinePhase::Parse), 100);
    }

    #[test]
    fn test_timing_slowest() {
        let mut timing = PipelineTiming::new();
        timing.record(PipelinePhase::Parse, 10);
        timing.record(PipelinePhase::Execute, 200);
        timing.record(PipelinePhase::Rank, 50);
        let (phase, dur) = timing.slowest_phase().expect("should succeed in test");
        assert_eq!(phase, PipelinePhase::Execute);
        assert_eq!(dur, 200);
    }

    // -- PipelineConfig tests --

    #[test]
    fn test_config_default() {
        let config = PipelineConfig::default();
        assert!(config.enable_expansion);
        assert!(config.is_phase_enabled(PipelinePhase::Parse));
        assert_eq!(config.max_results, 100);
    }

    // -- SearchPipeline tests --

    #[test]
    fn test_pipeline_empty_query() {
        let pipeline = SearchPipeline::with_defaults();
        let (results, timing) = pipeline.execute("");
        assert!(results.is_empty());
        assert!(timing.total_us < 1_000_000);
    }

    #[test]
    fn test_pipeline_basic_query() {
        let pipeline = SearchPipeline::with_defaults();
        let (results, _timing) = pipeline.execute("hello world");
        assert!(!results.is_empty());
        assert!(results[0].score >= results.last().expect("should succeed in test").score);
    }

    #[test]
    fn test_pipeline_with_synonyms() {
        let mut pipeline = SearchPipeline::with_defaults();
        let mut syns = SynonymMap::new();
        syns.add_group(&["hello", "hi", "greetings"]);
        pipeline.set_synonyms(syns);
        let (results, _) = pipeline.execute("hello");
        assert!(!results.is_empty());
    }

    #[test]
    fn test_pipeline_min_score_filter() {
        let config = PipelineConfig {
            min_score: 0.9,
            ..Default::default()
        };
        let pipeline = SearchPipeline::new(config);
        let (results, _) = pipeline.execute("a b c d e");
        // Only the top-scoring results should survive
        for r in &results {
            assert!(r.score >= 0.9);
        }
    }

    #[test]
    fn test_phase_name() {
        assert_eq!(PipelinePhase::Parse.name(), "parse");
        assert_eq!(PipelinePhase::Expand.name(), "expand");
        assert_eq!(PipelinePhase::Execute.name(), "execute");
        assert_eq!(PipelinePhase::Rank.name(), "rank");
        assert_eq!(PipelinePhase::Format.name(), "format");
    }
}
