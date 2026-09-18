//! Pattern matching for telecine detection.
//!
//! This module implements pattern matching algorithms for detecting various
//! telecine and pulldown patterns used in film-to-video transfer.

use std::collections::VecDeque;

/// Pulldown pattern types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PulldownPattern {
    /// 3:2 pulldown (24fps film → 29.97fps NTSC).
    ///
    /// Pattern: AA BB BC CD DD (5 frames from 4 film frames).
    Pulldown32,

    /// 2:2 pulldown (25fps PAL).
    ///
    /// Pattern: AA BB CC DD (each film frame shown twice).
    Pulldown22,

    /// 2:3:3:2 advanced pulldown.
    ///
    /// Pattern: AA BBB CCC DD (more complex pattern for 24fps → 30fps).
    Pulldown2332,

    /// Euro pulldown (24fps → 25fps speedup, no field pattern).
    EuroPulldown,

    /// No pulldown detected.
    None,
}

impl PulldownPattern {
    /// Returns the pattern length (number of frames in one cycle).
    #[must_use]
    pub const fn cycle_length(&self) -> usize {
        match self {
            Self::Pulldown32 => 5,
            Self::Pulldown22 => 4,
            Self::Pulldown2332 => 10,
            Self::EuroPulldown => 1,
            Self::None => 0,
        }
    }

    /// Returns the expected field pattern for this pulldown.
    ///
    /// Returns an array of field counts per frame in one cycle.
    #[must_use]
    pub const fn field_pattern(&self) -> &'static [u8] {
        match self {
            Self::Pulldown32 => &[3, 2, 3, 2],
            Self::Pulldown22 => &[2, 2, 2, 2],
            Self::Pulldown2332 => &[2, 3, 3, 2],
            Self::EuroPulldown => &[2],
            Self::None => &[],
        }
    }

    /// Returns all known pulldown patterns for detection.
    #[must_use]
    pub const fn all_patterns() -> &'static [Self] {
        &[
            Self::Pulldown32,
            Self::Pulldown22,
            Self::Pulldown2332,
            Self::EuroPulldown,
        ]
    }
}

/// Cadence pattern for telecine detection.
#[derive(Debug, Clone, PartialEq)]
pub struct CadencePattern {
    /// The detected pattern type.
    pub pattern_type: PulldownPattern,
    /// Confidence in this pattern (0.0-1.0).
    pub confidence: f64,
    /// Phase offset within the pattern cycle.
    pub phase: usize,
    /// Number of consecutive frames matching this pattern.
    pub match_count: usize,
    /// Stability score (how consistent the pattern is).
    pub stability: f64,
}

impl CadencePattern {
    /// Creates a new cadence pattern.
    #[must_use]
    pub const fn new(pattern_type: PulldownPattern) -> Self {
        Self {
            pattern_type,
            confidence: 0.0,
            phase: 0,
            match_count: 0,
            stability: 0.0,
        }
    }

    /// Returns true if this pattern is stable and confident.
    #[must_use]
    pub fn is_stable(&self, confidence_threshold: f64, stability_threshold: f64) -> bool {
        self.confidence >= confidence_threshold && self.stability >= stability_threshold
    }
}

/// Pattern matcher for telecine detection.
pub struct PatternMatcher {
    /// Window size for pattern analysis.
    window_size: usize,
    /// Minimum matches required to confirm pattern.
    min_matches: usize,
    /// Frame difference history.
    diff_history: VecDeque<FrameDifference>,
    /// Currently detected pattern.
    current_pattern: Option<CadencePattern>,
}

impl PatternMatcher {
    /// Creates a new pattern matcher.
    #[must_use]
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size: window_size.max(10),
            min_matches: 3,
            diff_history: VecDeque::with_capacity(window_size.max(10)),
            current_pattern: None,
        }
    }

    /// Creates a pattern matcher with default window size.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(30)
    }

    /// Adds a frame difference measurement to the history.
    pub fn add_frame_difference(&mut self, diff: FrameDifference) {
        if self.diff_history.len() >= self.window_size {
            self.diff_history.pop_front();
        }
        self.diff_history.push_back(diff);
    }

    /// Analyzes the frame difference history to detect patterns.
    pub fn detect_pattern(&mut self) -> Option<CadencePattern> {
        if self.diff_history.len() < 10 {
            return None;
        }

        let mut best_pattern = None;
        let mut best_confidence = 0.0;

        // Try to match each known pattern
        for pattern_type in PulldownPattern::all_patterns() {
            if let Some(detected) = self.match_pattern(*pattern_type) {
                if detected.confidence > best_confidence {
                    best_confidence = detected.confidence;
                    best_pattern = Some(detected);
                }
            }
        }

        best_pattern.clone_into(&mut self.current_pattern);
        best_pattern
    }

    /// Matches a specific pattern type against the history.
    fn match_pattern(&self, pattern_type: PulldownPattern) -> Option<CadencePattern> {
        let cycle_len = pattern_type.cycle_length();
        if cycle_len == 0 || self.diff_history.len() < cycle_len * 2 {
            return None;
        }

        let field_pattern = pattern_type.field_pattern();
        if field_pattern.is_empty() {
            return None;
        }

        let mut best_phase = 0;
        let mut best_match_score = 0.0;

        // Try all possible phase alignments
        for phase in 0..cycle_len {
            let score = self.score_pattern_match(field_pattern, phase);
            if score > best_match_score {
                best_match_score = score;
                best_phase = phase;
            }
        }

        // Convert match score to confidence
        let confidence = best_match_score;
        if confidence < 0.3 {
            return None;
        }

        // Calculate stability
        let stability = self.calculate_pattern_stability(field_pattern, best_phase);

        Some(CadencePattern {
            pattern_type,
            confidence,
            phase: best_phase,
            match_count: self.diff_history.len() / cycle_len,
            stability,
        })
    }

    /// Scores how well a pattern matches the frame difference history.
    fn score_pattern_match(&self, field_pattern: &[u8], phase: usize) -> f64 {
        let mut match_count = 0;
        let mut total_count = 0;

        for i in 0..self.diff_history.len() {
            let pattern_idx = (i + phase) % field_pattern.len();
            let expected_fields = field_pattern[pattern_idx];

            if let Some(diff) = self.diff_history.get(i) {
                // Check if this frame has a repeated field (low difference)
                let is_repeated = diff.temporal_diff < 0.1;
                let expects_repeat = expected_fields == 3; // Repeated field in 3:2 pulldown

                if is_repeated == expects_repeat {
                    match_count += 1;
                }
                total_count += 1;
            }
        }

        if total_count == 0 {
            return 0.0;
        }

        match_count as f64 / total_count as f64
    }

    /// Calculates the stability of a pattern match over time.
    fn calculate_pattern_stability(&self, field_pattern: &[u8], phase: usize) -> f64 {
        if self.diff_history.len() < field_pattern.len() * 3 {
            return 0.0;
        }

        let cycle_len = field_pattern.len();
        let num_cycles = self.diff_history.len() / cycle_len;

        let mut cycle_scores = Vec::with_capacity(num_cycles);

        for cycle in 0..num_cycles {
            let start_idx = cycle * cycle_len;
            let end_idx = start_idx + cycle_len;

            let mut cycle_match = 0;
            for i in start_idx..end_idx.min(self.diff_history.len()) {
                let pattern_idx = (i - start_idx + phase) % cycle_len;
                let expected_fields = field_pattern[pattern_idx];

                if let Some(diff) = self.diff_history.get(i) {
                    let is_repeated = diff.temporal_diff < 0.1;
                    let expects_repeat = expected_fields == 3;

                    if is_repeated == expects_repeat {
                        cycle_match += 1;
                    }
                }
            }

            let cycle_score = cycle_match as f64 / cycle_len as f64;
            cycle_scores.push(cycle_score);
        }

        if cycle_scores.is_empty() {
            return 0.0;
        }

        // Calculate variance of cycle scores (lower is more stable)
        let mean: f64 = cycle_scores.iter().sum::<f64>() / cycle_scores.len() as f64;
        let variance: f64 = cycle_scores
            .iter()
            .map(|&score| {
                let diff = score - mean;
                diff * diff
            })
            .sum::<f64>()
            / cycle_scores.len() as f64;

        // Convert variance to stability (0 variance = 1.0 stability)
        let stability = (-variance * 10.0).exp();
        stability.clamp(0.0, 1.0)
    }

    /// Gets the current detected pattern.
    #[must_use]
    pub const fn current_pattern(&self) -> Option<&CadencePattern> {
        self.current_pattern.as_ref()
    }

    /// Resets the pattern matcher state.
    pub fn reset(&mut self) {
        self.diff_history.clear();
        self.current_pattern = None;
    }

    /// Returns the pattern history for analysis.
    #[must_use]
    pub fn history(&self) -> &VecDeque<FrameDifference> {
        &self.diff_history
    }

    /// Generates a cadence map showing the pattern over time.
    #[must_use]
    pub fn generate_cadence_map(&self) -> Vec<CadenceMapEntry> {
        let mut map = Vec::with_capacity(self.diff_history.len());

        if let Some(pattern) = &self.current_pattern {
            let field_pattern = pattern.pattern_type.field_pattern();
            if field_pattern.is_empty() {
                return map;
            }

            for (i, diff) in self.diff_history.iter().enumerate() {
                let pattern_idx = (i + pattern.phase) % field_pattern.len();
                let expected_fields = field_pattern[pattern_idx];
                let is_repeated = diff.temporal_diff < 0.1;

                map.push(CadenceMapEntry {
                    frame_index: i,
                    expected_fields,
                    actual_diff: diff.temporal_diff,
                    is_repeated,
                    matches_pattern: (expected_fields == 3) == is_repeated,
                });
            }
        }

        map
    }

    /// Predicts the next expected frame difference based on the current pattern.
    #[must_use]
    pub fn predict_next_frame(&self) -> Option<FramePrediction> {
        let pattern = self.current_pattern.as_ref()?;
        if pattern.confidence < 0.5 {
            return None;
        }

        let field_pattern = pattern.pattern_type.field_pattern();
        if field_pattern.is_empty() {
            return None;
        }

        let next_idx = (self.diff_history.len() + pattern.phase) % field_pattern.len();
        let expected_fields = field_pattern[next_idx];

        Some(FramePrediction {
            expects_repeated_field: expected_fields == 3,
            pattern_position: next_idx,
            confidence: pattern.confidence,
        })
    }
}

impl Default for PatternMatcher {
    fn default() -> Self {
        Self::with_defaults()
    }
}

/// Frame difference measurement for pattern detection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameDifference {
    /// Temporal difference (frame-to-frame).
    pub temporal_diff: f64,
    /// Field difference (between fields in same frame).
    pub field_diff: f64,
    /// Frame index in sequence.
    pub frame_index: usize,
}

impl FrameDifference {
    /// Creates a new frame difference measurement.
    #[must_use]
    pub const fn new(temporal_diff: f64, field_diff: f64, frame_index: usize) -> Self {
        Self {
            temporal_diff,
            field_diff,
            frame_index,
        }
    }

    /// Returns true if this frame likely has a repeated field.
    #[must_use]
    pub fn is_repeated_field(&self, threshold: f64) -> bool {
        self.temporal_diff < threshold
    }
}

/// Entry in a cadence map visualization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CadenceMapEntry {
    /// Frame index in sequence.
    pub frame_index: usize,
    /// Expected number of fields for this frame in the pattern.
    pub expected_fields: u8,
    /// Actual temporal difference measured.
    pub actual_diff: f64,
    /// Whether this frame has a repeated field.
    pub is_repeated: bool,
    /// Whether this frame matches the expected pattern.
    pub matches_pattern: bool,
}

/// Prediction for the next frame based on detected pattern.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FramePrediction {
    /// Whether the next frame is expected to have a repeated field.
    pub expects_repeated_field: bool,
    /// Position within the pattern cycle.
    pub pattern_position: usize,
    /// Confidence in this prediction.
    pub confidence: f64,
}

/// Pattern validation result.
#[derive(Debug, Clone, PartialEq)]
pub struct PatternValidation {
    /// Whether the pattern is valid.
    pub is_valid: bool,
    /// Confidence in the validation.
    pub confidence: f64,
    /// Number of frames that matched the pattern.
    pub match_count: usize,
    /// Total number of frames analyzed.
    pub total_count: usize,
    /// Detected pattern type.
    pub pattern_type: PulldownPattern,
}

impl PatternValidation {
    /// Returns the match ratio (0.0-1.0).
    #[must_use]
    pub fn match_ratio(&self) -> f64 {
        if self.total_count == 0 {
            return 0.0;
        }
        self.match_count as f64 / self.total_count as f64
    }
}

/// Validates a detected pattern against a sequence of frames.
pub struct PatternValidator {
    /// Minimum match ratio to consider valid.
    min_match_ratio: f64,
    /// Minimum confidence threshold.
    min_confidence: f64,
}

impl PatternValidator {
    /// Creates a new pattern validator.
    #[must_use]
    pub const fn new(min_match_ratio: f64, min_confidence: f64) -> Self {
        Self {
            min_match_ratio,
            min_confidence,
        }
    }

    /// Creates a validator with default thresholds.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(0.8, 0.6)
    }

    /// Validates a cadence pattern against frame differences.
    pub fn validate(
        &self,
        pattern: &CadencePattern,
        frame_diffs: &[FrameDifference],
    ) -> PatternValidation {
        let field_pattern = pattern.pattern_type.field_pattern();
        if field_pattern.is_empty() {
            return PatternValidation {
                is_valid: false,
                confidence: 0.0,
                match_count: 0,
                total_count: 0,
                pattern_type: pattern.pattern_type,
            };
        }

        let mut match_count = 0;
        let total_count = frame_diffs.len();

        for (i, diff) in frame_diffs.iter().enumerate() {
            let pattern_idx = (i + pattern.phase) % field_pattern.len();
            let expected_fields = field_pattern[pattern_idx];
            let is_repeated = diff.temporal_diff < 0.1;
            let expects_repeat = expected_fields == 3;

            if is_repeated == expects_repeat {
                match_count += 1;
            }
        }

        let match_ratio = if total_count > 0 {
            match_count as f64 / total_count as f64
        } else {
            0.0
        };

        let confidence = (match_ratio + pattern.stability) / 2.0;
        let is_valid = match_ratio >= self.min_match_ratio && confidence >= self.min_confidence;

        PatternValidation {
            is_valid,
            confidence,
            match_count,
            total_count,
            pattern_type: pattern.pattern_type,
        }
    }
}

impl Default for PatternValidator {
    fn default() -> Self {
        Self::with_defaults()
    }
}
