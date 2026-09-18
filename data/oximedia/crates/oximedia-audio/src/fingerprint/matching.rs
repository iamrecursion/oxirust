//! Fingerprint matching algorithm.

use super::hash::Hash;
use super::Fingerprint;
use std::collections::HashMap;

/// Fingerprint matcher.
pub struct FingerprintMatcher {
    /// Minimum number of matching hashes required.
    min_matches: usize,
    /// Time tolerance for offset alignment (seconds).
    time_tolerance: f64,
    /// Enable partial matching.
    allow_partial: bool,
}

impl FingerprintMatcher {
    /// Create a new fingerprint matcher.
    #[must_use]
    pub const fn new(min_matches: usize, time_tolerance: f64, allow_partial: bool) -> Self {
        Self {
            min_matches,
            time_tolerance,
            allow_partial,
        }
    }

    /// Match a query fingerprint against a reference.
    #[must_use]
    pub fn match_fingerprint(
        &self,
        query: &Fingerprint,
        reference: &Fingerprint,
    ) -> Option<MatchResult> {
        // Build hash lookup table for reference
        let mut hash_table: HashMap<Hash, Vec<f64>> = HashMap::new();
        for (hash, time) in &reference.hashes {
            hash_table.entry(*hash).or_default().push(*time);
        }

        // Find matching hashes and compute time offsets
        let mut offset_histogram: HashMap<i64, Vec<(Hash, f64, f64)>> = HashMap::new();

        for (query_hash, query_time) in &query.hashes {
            if let Some(ref_times) = hash_table.get(query_hash) {
                for &ref_time in ref_times {
                    // Compute time offset (quantized to 10ms bins)
                    let offset = ((ref_time - query_time) * 100.0).round() as i64;

                    offset_histogram.entry(offset).or_default().push((
                        *query_hash,
                        *query_time,
                        ref_time,
                    ));
                }
            }
        }

        // Find the offset with most matches
        let (best_offset, matches) = offset_histogram
            .iter()
            .max_by_key(|(_, matches)| matches.len())?;

        let match_count = matches.len();

        // Check minimum match threshold
        if match_count < self.min_matches {
            if !self.allow_partial {
                return None;
            }
        }

        // Calculate confidence
        let confidence = self.calculate_confidence(match_count, query, reference);

        // Convert offset back to seconds
        let time_offset = *best_offset as f64 / 100.0;

        Some(MatchResult {
            match_count,
            total_query_hashes: query.hashes.len(),
            total_reference_hashes: reference.hashes.len(),
            confidence,
            time_offset,
            matches: matches.clone(),
        })
    }

    /// Match against multiple reference fingerprints.
    #[must_use]
    pub fn match_multiple<'a>(
        &self,
        query: &Fingerprint,
        references: &'a [(String, Fingerprint)],
    ) -> Vec<(&'a str, MatchResult)> {
        let mut results = Vec::new();

        for (id, reference) in references {
            if let Some(result) = self.match_fingerprint(query, reference) {
                results.push((id.as_str(), result));
            }
        }

        // Sort by confidence
        results.sort_by(|a, b| {
            b.1.confidence
                .partial_cmp(&a.1.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results
    }

    /// Calculate match confidence (0-1).
    fn calculate_confidence(
        &self,
        match_count: usize,
        query: &Fingerprint,
        reference: &Fingerprint,
    ) -> f64 {
        if query.hashes.is_empty() || reference.hashes.is_empty() {
            return 0.0;
        }

        // Jaccard similarity
        let jaccard =
            match_count as f64 / (query.hashes.len() + reference.hashes.len() - match_count) as f64;

        // Match rate relative to query
        let query_coverage = match_count as f64 / query.hashes.len() as f64;

        // Combined confidence
        (jaccard * 0.5 + query_coverage * 0.5).min(1.0)
    }

    /// Verify match using temporal consistency.
    #[must_use]
    pub fn verify_match(&self, result: &MatchResult) -> bool {
        if result.matches.len() < self.min_matches {
            return false;
        }

        // Check temporal consistency of matches
        let mut time_diffs = Vec::new();
        for i in 1..result.matches.len() {
            let (_, q1, r1) = result.matches[i - 1];
            let (_, q2, r2) = result.matches[i];

            let query_diff = q2 - q1;
            let ref_diff = r2 - r1;
            let diff_error = (query_diff - ref_diff).abs();

            time_diffs.push(diff_error);
        }

        if time_diffs.is_empty() {
            return true;
        }

        // Check if most time differences are within tolerance
        let consistent_count = time_diffs
            .iter()
            .filter(|&&d| d <= self.time_tolerance)
            .count();

        let consistency_ratio = consistent_count as f64 / time_diffs.len() as f64;

        consistency_ratio >= 0.8
    }

    /// Find all matches above confidence threshold.
    #[must_use]
    pub fn find_matches<'a>(
        &self,
        query: &Fingerprint,
        references: &'a [(String, Fingerprint)],
        min_confidence: f64,
    ) -> Vec<(&'a str, MatchResult)> {
        self.match_multiple(query, references)
            .into_iter()
            .filter(|(_, result)| result.confidence >= min_confidence)
            .filter(|(_, result)| self.verify_match(result))
            .collect()
    }
}

impl Default for FingerprintMatcher {
    fn default() -> Self {
        Self::new(10, 0.1, false)
    }
}

/// Result of a fingerprint match.
#[derive(Clone, Debug)]
pub struct MatchResult {
    /// Number of matching hashes.
    pub match_count: usize,
    /// Total hashes in query fingerprint.
    pub total_query_hashes: usize,
    /// Total hashes in reference fingerprint.
    pub total_reference_hashes: usize,
    /// Confidence score (0-1).
    pub confidence: f64,
    /// Time offset (seconds, reference - query).
    pub time_offset: f64,
    /// Individual matches (hash, query_time, ref_time).
    pub matches: Vec<(Hash, f64, f64)>,
}

impl MatchResult {
    /// Get match coverage relative to query.
    #[must_use]
    pub fn query_coverage(&self) -> f64 {
        if self.total_query_hashes > 0 {
            self.match_count as f64 / self.total_query_hashes as f64
        } else {
            0.0
        }
    }

    /// Get match coverage relative to reference.
    #[must_use]
    pub fn reference_coverage(&self) -> f64 {
        if self.total_reference_hashes > 0 {
            self.match_count as f64 / self.total_reference_hashes as f64
        } else {
            0.0
        }
    }

    /// Check if match is strong (high confidence and coverage).
    #[must_use]
    pub fn is_strong_match(&self) -> bool {
        self.confidence >= 0.7 && self.query_coverage() >= 0.5
    }

    /// Check if match is likely a duplicate.
    #[must_use]
    pub fn is_likely_duplicate(&self) -> bool {
        self.confidence >= 0.9 && self.query_coverage() >= 0.8
    }

    /// Get temporal spread of matches.
    #[must_use]
    pub fn temporal_spread(&self) -> f64 {
        if self.matches.len() < 2 {
            return 0.0;
        }

        let query_times: Vec<f64> = self.matches.iter().map(|(_, q, _)| *q).collect();

        let min_time = query_times
            .iter()
            .copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        let max_time = query_times
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        max_time - min_time
    }

    /// Get match density (matches per second).
    #[must_use]
    pub fn match_density(&self) -> f64 {
        let spread = self.temporal_spread();
        if spread > 0.0 {
            self.match_count as f64 / spread
        } else {
            0.0
        }
    }
}

/// Advanced matching strategies.
pub struct AdvancedMatcher;

impl AdvancedMatcher {
    /// Sliding window matching for long audio.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::similar_names
    )]
    pub fn sliding_window_match(
        query: &Fingerprint,
        reference: &Fingerprint,
        window_size: f64,
        hop_size: f64,
    ) -> Vec<WindowMatch> {
        let mut matches = Vec::new();
        let matcher = FingerprintMatcher::default();

        let mut window_start = 0.0;
        while window_start + window_size <= query.duration {
            // Extract window from query
            let window_hashes: Vec<(Hash, f64)> = query
                .hashes
                .iter()
                .filter(|(_, t)| *t >= window_start && *t < window_start + window_size)
                .map(|(h, t)| (*h, *t - window_start))
                .collect();

            if !window_hashes.is_empty() {
                let window_fp = Fingerprint::new(
                    window_hashes,
                    query.sample_rate,
                    window_size,
                    query.config.clone(),
                );

                if let Some(result) = matcher.match_fingerprint(&window_fp, reference) {
                    matches.push(WindowMatch {
                        window_start,
                        window_end: window_start + window_size,
                        result,
                    });
                }
            }

            window_start += hop_size;
        }

        matches
    }

    /// Multi-scale matching (different time resolutions).
    #[must_use]
    #[allow(clippy::similar_names)]
    pub fn multiscale_match(
        query: &Fingerprint,
        reference: &Fingerprint,
        scales: &[f64],
    ) -> Vec<ScaleMatch> {
        let matcher = FingerprintMatcher::default();
        let mut matches = Vec::new();

        for &scale in scales {
            if let Some(result) = matcher.match_fingerprint(query, reference) {
                matches.push(ScaleMatch { scale, result });
            }
        }

        matches.sort_by(|a, b| {
            b.result
                .confidence
                .partial_cmp(&a.result.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        matches
    }

    /// Fuzzy matching with tolerance for variations.
    #[must_use]
    pub fn fuzzy_match(
        query: &Fingerprint,
        reference: &Fingerprint,
        hash_tolerance: u32,
    ) -> Option<MatchResult> {
        let mut offset_histogram: HashMap<i64, Vec<(Hash, f64, f64)>> = HashMap::new();

        // Build hash table with fuzzy matching
        let mut hash_table: HashMap<Hash, Vec<f64>> = HashMap::new();
        for (hash, time) in &reference.hashes {
            hash_table.entry(*hash).or_default().push(*time);
        }

        // Match with tolerance
        for (query_hash, query_time) in &query.hashes {
            // Exact match
            if let Some(ref_times) = hash_table.get(query_hash) {
                for &ref_time in ref_times {
                    let offset = ((ref_time - query_time) * 100.0).round() as i64;
                    offset_histogram.entry(offset).or_default().push((
                        *query_hash,
                        *query_time,
                        ref_time,
                    ));
                }
            }

            // Fuzzy match (check similar hashes)
            if hash_tolerance > 0 {
                for (ref_hash, ref_times) in &hash_table {
                    if super::hash::HashComparison::are_similar(
                        *query_hash,
                        *ref_hash,
                        hash_tolerance,
                    ) {
                        for &ref_time in ref_times {
                            let offset = ((ref_time - query_time) * 100.0).round() as i64;
                            offset_histogram.entry(offset).or_default().push((
                                *query_hash,
                                *query_time,
                                ref_time,
                            ));
                        }
                    }
                }
            }
        }

        // Find best offset
        let (best_offset, matches) = offset_histogram
            .iter()
            .max_by_key(|(_, matches)| matches.len())?;

        let match_count = matches.len();
        let confidence = if query.hashes.is_empty() || reference.hashes.is_empty() {
            0.0
        } else {
            let jaccard = match_count as f64
                / (query.hashes.len() + reference.hashes.len() - match_count) as f64;
            let coverage = match_count as f64 / query.hashes.len() as f64;
            (jaccard * 0.5 + coverage * 0.5).min(1.0)
        };

        Some(MatchResult {
            match_count,
            total_query_hashes: query.hashes.len(),
            total_reference_hashes: reference.hashes.len(),
            confidence,
            time_offset: *best_offset as f64 / 100.0,
            matches: matches.clone(),
        })
    }
}

/// Window match result.
#[derive(Clone, Debug)]
pub struct WindowMatch {
    /// Window start time (seconds).
    pub window_start: f64,
    /// Window end time (seconds).
    pub window_end: f64,
    /// Match result.
    pub result: MatchResult,
}

/// Scale match result.
#[derive(Clone, Debug)]
pub struct ScaleMatch {
    /// Time scale factor.
    pub scale: f64,
    /// Match result.
    pub result: MatchResult,
}
