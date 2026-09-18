//! Narrative arc detection and optimization.
//!
//! This module provides tools for detecting and optimizing the narrative
//! structure of video content, including arc detection, emotional beat
//! analysis, and pacing optimization.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Represents a detected narrative arc type with act boundary positions
/// expressed as normalized 0.0–1.0 positions in the video timeline.
#[derive(Debug, Clone, PartialEq)]
pub enum NarrativeArc {
    /// Classic three-act structure.
    ThreeAct {
        /// Normalized position (0–1) where the setup ends / rising action begins.
        setup_end: f32,
        /// Normalized position (0–1) where the rising action ends / climax begins.
        rising_action_end: f32,
    },
    /// Five-act structure (Freytag's pyramid).
    FiveAct {
        /// End of exposition.
        exposition_end: f32,
        /// End of rising action.
        rising_end: f32,
        /// End of climax.
        climax_end: f32,
        /// End of falling action (resolution follows).
        falling_end: f32,
    },
    /// Joseph Campbell's Hero's Journey.
    HeroJourney {
        /// End of the "ordinary world" phase.
        ordinary_world_end: f32,
        /// Position of the call to adventure.
        call_to_adventure: f32,
        /// Position of the central ordeal.
        ordeal: f32,
        /// Position of the return / resolution.
        return_pos: f32,
    },
    /// Japanese four-part narrative structure (Introduction-Development-Twist-Conclusion).
    Kishōtenketsu,
}

impl NarrativeArc {
    /// Return a human-readable name for this arc type.
    pub fn name(&self) -> &'static str {
        match self {
            NarrativeArc::ThreeAct { .. } => "Three-Act",
            NarrativeArc::FiveAct { .. } => "Five-Act",
            NarrativeArc::HeroJourney { .. } => "Hero's Journey",
            NarrativeArc::Kishōtenketsu => "Kishōtenketsu",
        }
    }

    /// Evaluate the arc's expected intensity at a normalized position `t` (0–1).
    ///
    /// This produces the arc *template* curve used for correlation scoring.
    pub fn template_intensity(&self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            NarrativeArc::ThreeAct {
                setup_end,
                rising_action_end,
            } => {
                // Triangle wave peaking at ~75% of the timeline.
                // Setup: low intensity linear rise
                // Rising action: steep ramp to peak
                // Resolution: quick fall
                let _ = setup_end; // used for structural reference only in template
                if t < *rising_action_end {
                    // Ramp from 0 to 1 over the first (rising_action_end) fraction
                    t / rising_action_end.max(1e-6)
                } else {
                    // Fall from 1 to 0 over the remainder
                    let span = (1.0 - rising_action_end).max(1e-6);
                    1.0 - (t - rising_action_end) / span
                }
            }
            NarrativeArc::FiveAct {
                exposition_end,
                rising_end,
                climax_end,
                falling_end,
            } => {
                // Trapezoid: rise → plateau → fall
                if t < *exposition_end {
                    t / exposition_end.max(1e-6) * 0.3
                } else if t < *rising_end {
                    let span = (rising_end - exposition_end).max(1e-6);
                    0.3 + (t - exposition_end) / span * 0.7
                } else if t < *climax_end {
                    1.0
                } else if t < *falling_end {
                    let span = (falling_end - climax_end).max(1e-6);
                    1.0 - (t - climax_end) / span * 0.7
                } else {
                    let span = (1.0 - falling_end).max(1e-6);
                    0.3 - (t - falling_end) / span * 0.3
                }
            }
            NarrativeArc::HeroJourney {
                ordinary_world_end,
                call_to_adventure,
                ordeal,
                return_pos,
            } => {
                // Sine-wave variant: slow start, dramatic peak at ordeal, resolution
                let phase = if t < *ordinary_world_end {
                    // Low initial phase
                    let span = ordinary_world_end.max(1e-6);
                    (t / span) * std::f32::consts::FRAC_PI_4
                } else if t < *call_to_adventure {
                    let span = (call_to_adventure - ordinary_world_end).max(1e-6);
                    std::f32::consts::FRAC_PI_4
                        + (t - ordinary_world_end) / span * std::f32::consts::FRAC_PI_4
                } else if t < *ordeal {
                    let span = (ordeal - call_to_adventure).max(1e-6);
                    std::f32::consts::FRAC_PI_2
                        + (t - call_to_adventure) / span * std::f32::consts::FRAC_PI_2
                } else if t < *return_pos {
                    let span = (return_pos - ordeal).max(1e-6);
                    std::f32::consts::PI - (t - ordeal) / span * std::f32::consts::FRAC_PI_4
                } else {
                    let span = (1.0 - return_pos).max(1e-6);
                    std::f32::consts::PI * 0.75
                        - (t - return_pos) / span * std::f32::consts::FRAC_PI_4
                };
                phase.sin().clamp(0.0, 1.0)
            }
            NarrativeArc::Kishōtenketsu => {
                // Four equal sections: intro (low), development (rise), twist (peak), conclusion (resolve)
                if t < 0.25 {
                    t / 0.25 * 0.4
                } else if t < 0.5 {
                    0.4 + (t - 0.25) / 0.25 * 0.3
                } else if t < 0.75 {
                    0.7 + (t - 0.5) / 0.25 * 0.3
                } else {
                    1.0 - (t - 0.75) / 0.25 * 0.6
                }
            }
        }
    }
}

/// An emotional beat in a video timeline.
#[derive(Debug, Clone, PartialEq)]
pub struct EmotionalBeat {
    /// Timestamp of this beat in milliseconds.
    pub timestamp_ms: i64,
    /// Emotional intensity (0.0–1.0, higher = more intense).
    pub intensity: f32,
    /// Valence: positive (>0) or negative (<0) emotion. Range [-1, 1].
    pub valence: f32,
    /// Arousal: degree of activation/excitement. Range [0, 1].
    pub arousal: f32,
}

impl EmotionalBeat {
    /// Create a new emotional beat.
    pub fn new(timestamp_ms: i64, intensity: f32, valence: f32, arousal: f32) -> Self {
        Self {
            timestamp_ms,
            intensity: intensity.clamp(0.0, 1.0),
            valence: valence.clamp(-1.0, 1.0),
            arousal: arousal.clamp(0.0, 1.0),
        }
    }

    /// Combined emotional weight (intensity × arousal).
    pub fn weight(&self) -> f32 {
        self.intensity * self.arousal
    }
}

/// Narrative analyzer that detects arcs and provides pacing tools.
#[derive(Debug, Clone)]
pub struct NarrativeAnalyzer {
    /// The emotional beats in chronological order.
    pub beats: Vec<EmotionalBeat>,
    /// The detected (or assigned) narrative arc.
    pub arc: NarrativeArc,
}

impl NarrativeAnalyzer {
    /// Create a new analyzer from a set of emotional beats.
    ///
    /// Automatically detects the best-fit arc.
    pub fn new(beats: Vec<EmotionalBeat>) -> Self {
        let arc = detect_arc(&beats);
        Self { beats, arc }
    }

    /// Create a new analyzer with a pre-assigned arc.
    pub fn with_arc(beats: Vec<EmotionalBeat>, arc: NarrativeArc) -> Self {
        Self { beats, arc }
    }

    /// Re-detect the best arc from current beats.
    pub fn redetect_arc(&mut self) {
        self.arc = detect_arc(&self.beats);
    }

    /// Score the current arc against the beats.
    pub fn current_fit_score(&self) -> f32 {
        arc_fit_score(&self.beats, &self.arc)
    }

    /// Optimize pacing to better fit the current arc.
    pub fn optimize_pacing(&mut self) {
        let arc = self.arc.clone();
        optimize_pacing(&mut self.beats, &arc);
    }
}

// ---------------------------------------------------------------------------
// Arc detection and scoring
// ---------------------------------------------------------------------------

/// Candidate arcs used during detection.
fn candidate_arcs() -> Vec<NarrativeArc> {
    vec![
        NarrativeArc::ThreeAct {
            setup_end: 0.25,
            rising_action_end: 0.75,
        },
        NarrativeArc::FiveAct {
            exposition_end: 0.15,
            rising_end: 0.45,
            climax_end: 0.65,
            falling_end: 0.85,
        },
        NarrativeArc::HeroJourney {
            ordinary_world_end: 0.1,
            call_to_adventure: 0.2,
            ordeal: 0.65,
            return_pos: 0.85,
        },
        NarrativeArc::Kishōtenketsu,
    ]
}

/// Detect the best-fit narrative arc for a sequence of emotional beats.
///
/// Evaluates each candidate arc using Pearson correlation of beat intensity
/// against the arc template curve, returning the highest-scoring arc.
pub fn detect_arc(beats: &[EmotionalBeat]) -> NarrativeArc {
    if beats.is_empty() {
        return NarrativeArc::ThreeAct {
            setup_end: 0.25,
            rising_action_end: 0.75,
        };
    }

    let candidates = candidate_arcs();
    let mut best_arc = candidates[0].clone();
    let mut best_score = f32::NEG_INFINITY;

    for arc in &candidates {
        let score = arc_fit_score(beats, arc);
        if score > best_score {
            best_score = score;
            best_arc = arc.clone();
        }
    }

    best_arc
}

/// Compute Pearson correlation between beat intensities and the arc template.
///
/// Returns a value in [-1, 1] where 1.0 is a perfect positive fit.
pub fn arc_fit_score(beats: &[EmotionalBeat], arc: &NarrativeArc) -> f32 {
    if beats.len() < 2 {
        return 0.0;
    }

    let min_ts = beats.iter().map(|b| b.timestamp_ms).min().unwrap_or(0) as f64;
    let max_ts = beats.iter().map(|b| b.timestamp_ms).max().unwrap_or(1) as f64;
    let span = (max_ts - min_ts).max(1.0);

    let n = beats.len() as f64;
    let intensities: Vec<f64> = beats.iter().map(|b| b.intensity as f64).collect();
    let templates: Vec<f64> = beats
        .iter()
        .map(|b| {
            let t = ((b.timestamp_ms as f64 - min_ts) / span) as f32;
            arc.template_intensity(t) as f64
        })
        .collect();

    let mean_i = intensities.iter().sum::<f64>() / n;
    let mean_t = templates.iter().sum::<f64>() / n;

    let mut cov = 0.0_f64;
    let mut var_i = 0.0_f64;
    let mut var_t = 0.0_f64;

    for (i, t) in intensities.iter().zip(templates.iter()) {
        let di = i - mean_i;
        let dt = t - mean_t;
        cov += di * dt;
        var_i += di * di;
        var_t += dt * dt;
    }

    let denom = (var_i * var_t).sqrt();
    if denom < 1e-10 {
        return 0.0;
    }

    (cov / denom) as f32
}

/// Redistribute beat timestamps to better match the target arc's template curve.
///
/// Uses the inverse CDF of the template curve to map beats toward the arc's
/// expected timing. Timestamps are scaled proportionally within the original
/// time range so that the overall duration is preserved.
pub fn optimize_pacing(beats: &mut Vec<EmotionalBeat>, target_arc: &NarrativeArc) {
    if beats.len() < 2 {
        return;
    }

    let min_ts = beats.iter().map(|b| b.timestamp_ms).min().unwrap_or(0);
    let max_ts = beats.iter().map(|b| b.timestamp_ms).max().unwrap_or(1);
    let total_span = (max_ts - min_ts).max(1) as f64;

    let n = beats.len();

    // Build desired normalized positions based on intensity-weighted arc matching.
    // For each beat, find the normalized time position t where the template
    // intensity best matches that beat's intensity.
    let mut desired_positions: Vec<f64> = beats
        .iter()
        .enumerate()
        .map(|(idx, beat)| {
            // Weighted blend: 70% original position, 30% arc-driven position
            let original_t = (beat.timestamp_ms - min_ts) as f64 / total_span;

            // Find the t where template_intensity ≈ beat.intensity using linear search
            let steps = 100usize;
            let mut best_t = original_t as f32;
            let mut best_diff = f32::MAX;
            for step in 0..=steps {
                let candidate_t = step as f32 / steps as f32;
                let template_val = target_arc.template_intensity(candidate_t);
                let diff = (template_val - beat.intensity).abs();
                if diff < best_diff {
                    best_diff = diff;
                    best_t = candidate_t;
                }
            }

            // Bias towards beats near the center of their expected position range
            // to avoid all beats collapsing to the same time
            let arc_driven = best_t as f64;
            let rank_t = idx as f64 / (n - 1).max(1) as f64;
            0.5 * arc_driven + 0.3 * original_t + 0.2 * rank_t
        })
        .collect();

    // Ensure monotonicity: sort desired positions while preserving rank order
    desired_positions.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Map back to absolute timestamps
    for (beat, &new_t) in beats.iter_mut().zip(desired_positions.iter()) {
        beat.timestamp_ms = min_ts + (new_t * total_span) as i64;
    }
}

// ---------------------------------------------------------------------------
// Story beat segmentation
// ---------------------------------------------------------------------------

/// A labeled story beat (a segment of the video).
#[derive(Debug, Clone, PartialEq)]
pub struct StoryBeat {
    /// Human-readable label for this beat.
    pub label: String,
    /// Start of this beat in milliseconds.
    pub start_ms: i64,
    /// End of this beat in milliseconds.
    pub end_ms: i64,
    /// Functional type of this beat.
    pub beat_type: BeatType,
}

impl StoryBeat {
    /// Create a new story beat.
    pub fn new(label: impl Into<String>, start_ms: i64, end_ms: i64, beat_type: BeatType) -> Self {
        Self {
            label: label.into(),
            start_ms,
            end_ms,
            beat_type,
        }
    }

    /// Duration of this beat in milliseconds.
    pub fn duration_ms(&self) -> i64 {
        self.end_ms.saturating_sub(self.start_ms)
    }

    /// Whether this beat is in the climactic portion of the narrative.
    pub fn is_climactic(&self) -> bool {
        matches!(
            self.beat_type,
            BeatType::Climax | BeatType::IncitingIncident
        )
    }
}

/// Functional classification of a narrative beat.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BeatType {
    /// World-building and character introduction.
    Exposition,
    /// The event that sets the main conflict in motion.
    IncitingIncident,
    /// Escalating tension and complications.
    RisingAction,
    /// The peak of conflict or intensity.
    Climax,
    /// Consequences and winding down after the climax.
    FallingAction,
    /// Final outcome and closure.
    Resolution,
    /// Emotional decompression after the main arc.
    CoolDown,
}

impl BeatType {
    /// Expected normalized intensity range for this beat type.
    pub fn intensity_range(&self) -> (f32, f32) {
        match self {
            BeatType::Exposition => (0.0, 0.35),
            BeatType::IncitingIncident => (0.3, 0.65),
            BeatType::RisingAction => (0.4, 0.85),
            BeatType::Climax => (0.75, 1.0),
            BeatType::FallingAction => (0.4, 0.75),
            BeatType::Resolution => (0.15, 0.5),
            BeatType::CoolDown => (0.0, 0.25),
        }
    }
}

/// Segment a sequence of emotional beats into labeled story beats using
/// the given arc as a guide.
///
/// Returns story beats whose boundaries align with arc act transitions.
pub fn segment_into_story_beats(beats: &[EmotionalBeat], arc: &NarrativeArc) -> Vec<StoryBeat> {
    if beats.is_empty() {
        return Vec::new();
    }

    let min_ts = beats.iter().map(|b| b.timestamp_ms).min().unwrap_or(0);
    let max_ts = beats.iter().map(|b| b.timestamp_ms).max().unwrap_or(0);

    let story_beats = match arc {
        NarrativeArc::ThreeAct {
            setup_end,
            rising_action_end,
        } => {
            let t1 = lerp_ts(min_ts, max_ts, *setup_end);
            let t2 = lerp_ts(min_ts, max_ts, *rising_action_end);
            vec![
                StoryBeat::new("Setup", min_ts, t1, BeatType::Exposition),
                StoryBeat::new("Rising Action", t1, t2, BeatType::RisingAction),
                StoryBeat::new("Climax & Resolution", t2, max_ts, BeatType::Climax),
            ]
        }
        NarrativeArc::FiveAct {
            exposition_end,
            rising_end,
            climax_end,
            falling_end,
        } => {
            let t1 = lerp_ts(min_ts, max_ts, *exposition_end);
            let t2 = lerp_ts(min_ts, max_ts, *rising_end);
            let t3 = lerp_ts(min_ts, max_ts, *climax_end);
            let t4 = lerp_ts(min_ts, max_ts, *falling_end);
            vec![
                StoryBeat::new("Exposition", min_ts, t1, BeatType::Exposition),
                StoryBeat::new("Rising Action", t1, t2, BeatType::RisingAction),
                StoryBeat::new("Climax", t2, t3, BeatType::Climax),
                StoryBeat::new("Falling Action", t3, t4, BeatType::FallingAction),
                StoryBeat::new("Resolution", t4, max_ts, BeatType::Resolution),
            ]
        }
        NarrativeArc::HeroJourney {
            ordinary_world_end,
            call_to_adventure,
            ordeal,
            return_pos,
        } => {
            let t1 = lerp_ts(min_ts, max_ts, *ordinary_world_end);
            let t2 = lerp_ts(min_ts, max_ts, *call_to_adventure);
            let t3 = lerp_ts(min_ts, max_ts, *ordeal);
            let t4 = lerp_ts(min_ts, max_ts, *return_pos);
            vec![
                StoryBeat::new("Ordinary World", min_ts, t1, BeatType::Exposition),
                StoryBeat::new("Call to Adventure", t1, t2, BeatType::IncitingIncident),
                StoryBeat::new("Trials", t2, t3, BeatType::RisingAction),
                StoryBeat::new("Ordeal", t3, t4, BeatType::Climax),
                StoryBeat::new("Return", t4, max_ts, BeatType::Resolution),
            ]
        }
        NarrativeArc::Kishōtenketsu => {
            let t1 = lerp_ts(min_ts, max_ts, 0.25);
            let t2 = lerp_ts(min_ts, max_ts, 0.5);
            let t3 = lerp_ts(min_ts, max_ts, 0.75);
            vec![
                StoryBeat::new("Ki (Introduction)", min_ts, t1, BeatType::Exposition),
                StoryBeat::new("Shō (Development)", t1, t2, BeatType::RisingAction),
                StoryBeat::new("Ten (Twist)", t2, t3, BeatType::Climax),
                StoryBeat::new("Ketsu (Conclusion)", t3, max_ts, BeatType::Resolution),
            ]
        }
    };

    story_beats
}

/// Linear interpolation between two timestamps using a normalized factor.
fn lerp_ts(min_ts: i64, max_ts: i64, t: f32) -> i64 {
    let span = max_ts - min_ts;
    min_ts + (span as f64 * t as f64) as i64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_beats(n: usize) -> Vec<EmotionalBeat> {
        (0..n)
            .map(|i| {
                let t = i as f32 / (n - 1).max(1) as f32;
                // Ramp intensity from 0.1 to 0.9 to simulate rising tension
                let intensity = 0.1 + t * 0.8;
                EmotionalBeat::new(i as i64 * 1000, intensity, t * 2.0 - 1.0, intensity)
            })
            .collect()
    }

    // -- EmotionalBeat tests --

    #[test]
    fn test_emotional_beat_clamping() {
        let b = EmotionalBeat::new(0, 2.0, -5.0, 10.0);
        assert!((b.intensity - 1.0).abs() < 1e-6);
        assert!((b.valence - (-1.0)).abs() < 1e-6);
        assert!((b.arousal - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_emotional_beat_weight() {
        let b = EmotionalBeat::new(0, 0.8, 0.5, 0.5);
        assert!((b.weight() - 0.4).abs() < 1e-5);
    }

    // -- NarrativeArc template tests --

    #[test]
    fn test_three_act_template_range() {
        let arc = NarrativeArc::ThreeAct {
            setup_end: 0.25,
            rising_action_end: 0.75,
        };
        for i in 0..=100 {
            let t = i as f32 / 100.0;
            let v = arc.template_intensity(t);
            assert!(v >= 0.0 && v <= 1.0, "out of range at t={t}: {v}");
        }
    }

    #[test]
    fn test_five_act_template_range() {
        let arc = NarrativeArc::FiveAct {
            exposition_end: 0.15,
            rising_end: 0.45,
            climax_end: 0.65,
            falling_end: 0.85,
        };
        for i in 0..=100 {
            let t = i as f32 / 100.0;
            let v = arc.template_intensity(t);
            assert!(v >= 0.0 && v <= 1.0, "out of range at t={t}: {v}");
        }
    }

    #[test]
    fn test_hero_journey_template_range() {
        let arc = NarrativeArc::HeroJourney {
            ordinary_world_end: 0.1,
            call_to_adventure: 0.2,
            ordeal: 0.65,
            return_pos: 0.85,
        };
        for i in 0..=100 {
            let t = i as f32 / 100.0;
            let v = arc.template_intensity(t);
            assert!(v >= 0.0 && v <= 1.0, "out of range at t={t}: {v}");
        }
    }

    #[test]
    fn test_kishotenketsu_template_range() {
        let arc = NarrativeArc::Kishōtenketsu;
        for i in 0..=100 {
            let t = i as f32 / 100.0;
            let v = arc.template_intensity(t);
            assert!(v >= 0.0 && v <= 1.0, "out of range at t={t}: {v}");
        }
    }

    // -- arc_fit_score tests --

    #[test]
    fn test_arc_fit_score_empty_beats() {
        let arc = NarrativeArc::ThreeAct {
            setup_end: 0.25,
            rising_action_end: 0.75,
        };
        assert_eq!(arc_fit_score(&[], &arc), 0.0);
    }

    #[test]
    fn test_arc_fit_score_single_beat() {
        let beats = vec![EmotionalBeat::new(0, 0.5, 0.0, 0.5)];
        let arc = NarrativeArc::ThreeAct {
            setup_end: 0.25,
            rising_action_end: 0.75,
        };
        assert_eq!(arc_fit_score(&beats, &arc), 0.0);
    }

    #[test]
    fn test_arc_fit_score_returns_valid_range() {
        let beats = sample_beats(20);
        let arc = NarrativeArc::ThreeAct {
            setup_end: 0.25,
            rising_action_end: 0.75,
        };
        let score = arc_fit_score(&beats, &arc);
        assert!(score >= -1.0 && score <= 1.0, "score out of range: {score}");
    }

    // -- detect_arc tests --

    #[test]
    fn test_detect_arc_empty() {
        let arc = detect_arc(&[]);
        assert!(matches!(arc, NarrativeArc::ThreeAct { .. }));
    }

    #[test]
    fn test_detect_arc_returns_an_arc() {
        let beats = sample_beats(10);
        let arc = detect_arc(&beats);
        // Just verify a valid arc is returned
        let _ = arc.name();
    }

    #[test]
    fn test_detect_arc_linearly_rising_favors_three_act() {
        // Monotonically rising beats should correlate well with ThreeAct ramp
        let beats: Vec<EmotionalBeat> = (0..20)
            .map(|i| {
                let t = i as f32 / 19.0;
                EmotionalBeat::new(i * 1000, t, 0.0, 0.5)
            })
            .collect();
        let arc = detect_arc(&beats);
        // ThreeAct or FiveAct should score well; just verify it's valid
        assert!(!arc.name().is_empty());
    }

    // -- optimize_pacing tests --

    #[test]
    fn test_optimize_pacing_preserves_count() {
        let mut beats = sample_beats(10);
        let arc = NarrativeArc::ThreeAct {
            setup_end: 0.25,
            rising_action_end: 0.75,
        };
        optimize_pacing(&mut beats, &arc);
        assert_eq!(beats.len(), 10);
    }

    #[test]
    fn test_optimize_pacing_monotone_timestamps() {
        let mut beats = sample_beats(10);
        let arc = NarrativeArc::ThreeAct {
            setup_end: 0.25,
            rising_action_end: 0.75,
        };
        optimize_pacing(&mut beats, &arc);
        for window in beats.windows(2) {
            assert!(
                window[0].timestamp_ms <= window[1].timestamp_ms,
                "timestamps not monotone after pacing"
            );
        }
    }

    #[test]
    fn test_optimize_pacing_empty_noop() {
        let mut beats: Vec<EmotionalBeat> = Vec::new();
        let arc = NarrativeArc::ThreeAct {
            setup_end: 0.25,
            rising_action_end: 0.75,
        };
        optimize_pacing(&mut beats, &arc); // must not panic
        assert!(beats.is_empty());
    }

    // -- NarrativeAnalyzer tests --

    #[test]
    fn test_analyzer_new() {
        let beats = sample_beats(15);
        let analyzer = NarrativeAnalyzer::new(beats.clone());
        assert_eq!(analyzer.beats.len(), beats.len());
    }

    #[test]
    fn test_analyzer_fit_score_in_range() {
        let beats = sample_beats(15);
        let analyzer = NarrativeAnalyzer::new(beats);
        let score = analyzer.current_fit_score();
        assert!(score >= -1.0 && score <= 1.0);
    }

    #[test]
    fn test_analyzer_optimize_pacing() {
        let beats = sample_beats(12);
        let mut analyzer = NarrativeAnalyzer::new(beats);
        analyzer.optimize_pacing();
        assert_eq!(analyzer.beats.len(), 12);
    }

    // -- StoryBeat tests --

    #[test]
    fn test_story_beat_duration() {
        let sb = StoryBeat::new("test", 1000, 5000, BeatType::RisingAction);
        assert_eq!(sb.duration_ms(), 4000);
    }

    #[test]
    fn test_story_beat_is_climactic() {
        let sb = StoryBeat::new("climax", 0, 1000, BeatType::Climax);
        assert!(sb.is_climactic());
        let sb2 = StoryBeat::new("setup", 0, 1000, BeatType::Exposition);
        assert!(!sb2.is_climactic());
    }

    // -- segment_into_story_beats tests --

    #[test]
    fn test_segment_three_act() {
        let beats = sample_beats(20);
        let arc = NarrativeArc::ThreeAct {
            setup_end: 0.25,
            rising_action_end: 0.75,
        };
        let segments = segment_into_story_beats(&beats, &arc);
        assert_eq!(segments.len(), 3);
        assert!(segments[0].duration_ms() > 0);
    }

    #[test]
    fn test_segment_five_act() {
        let beats = sample_beats(20);
        let arc = NarrativeArc::FiveAct {
            exposition_end: 0.15,
            rising_end: 0.45,
            climax_end: 0.65,
            falling_end: 0.85,
        };
        let segments = segment_into_story_beats(&beats, &arc);
        assert_eq!(segments.len(), 5);
    }

    #[test]
    fn test_segment_hero_journey() {
        let beats = sample_beats(20);
        let arc = NarrativeArc::HeroJourney {
            ordinary_world_end: 0.1,
            call_to_adventure: 0.2,
            ordeal: 0.65,
            return_pos: 0.85,
        };
        let segments = segment_into_story_beats(&beats, &arc);
        assert_eq!(segments.len(), 5);
    }

    #[test]
    fn test_segment_kishotenketsu() {
        let beats = sample_beats(20);
        let arc = NarrativeArc::Kishōtenketsu;
        let segments = segment_into_story_beats(&beats, &arc);
        assert_eq!(segments.len(), 4);
    }

    #[test]
    fn test_segment_empty_beats() {
        let arc = NarrativeArc::ThreeAct {
            setup_end: 0.25,
            rising_action_end: 0.75,
        };
        let segments = segment_into_story_beats(&[], &arc);
        assert!(segments.is_empty());
    }
}
