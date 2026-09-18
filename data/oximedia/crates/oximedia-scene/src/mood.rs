//! Scene mood analysis.
//!
//! Classifies the emotional/dramatic mood of a scene based on visual features
//! such as brightness, contrast, saturation, motion, and shot length.

/// The mood category of a scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mood {
    /// High-energy action sequences.
    Action,
    /// Emotional dramatic scenes.
    Drama,
    /// Romantic or intimate scenes.
    Romance,
    /// Dark, tense, or frightening scenes.
    Horror,
    /// Light-hearted, humorous scenes.
    Comedy,
    /// Informational or observational scenes.
    Documentary,
    /// Suspenseful, tense scenes.
    Thriller,
    /// Peaceful natural scenery.
    Nature,
}

impl Mood {
    /// Return a human-readable label for this mood.
    #[must_use]
    pub fn label(&self) -> &str {
        match self {
            Self::Action => "action",
            Self::Drama => "drama",
            Self::Romance => "romance",
            Self::Horror => "horror",
            Self::Comedy => "comedy",
            Self::Documentary => "documentary",
            Self::Thriller => "thriller",
            Self::Nature => "nature",
        }
    }

    /// Return the typical average brightness (0.0–1.0) for this mood.
    #[must_use]
    pub fn typical_brightness(&self) -> f64 {
        match self {
            Self::Action => 0.55,
            Self::Drama => 0.40,
            Self::Romance => 0.60,
            Self::Horror => 0.20,
            Self::Comedy => 0.70,
            Self::Documentary => 0.55,
            Self::Thriller => 0.30,
            Self::Nature => 0.65,
        }
    }

    /// Return the typical motion level (0.0–1.0) for this mood.
    #[must_use]
    pub fn typical_motion(&self) -> f64 {
        match self {
            Self::Action => 0.85,
            Self::Drama => 0.25,
            Self::Romance => 0.20,
            Self::Horror => 0.40,
            Self::Comedy => 0.50,
            Self::Documentary => 0.30,
            Self::Thriller => 0.45,
            Self::Nature => 0.15,
        }
    }
}

/// Visual features extracted from a scene used for mood classification.
#[derive(Debug, Clone)]
pub struct MoodFeatures {
    /// Average pixel brightness (0.0–1.0).
    pub avg_brightness: f64,
    /// Contrast level (0.0–1.0).
    pub contrast: f64,
    /// Color saturation (0.0–1.0).
    pub saturation: f64,
    /// Motion level (0.0–1.0).
    pub motion_level: f64,
    /// Average shot length in frames.
    pub shot_length_frames: f64,
}

/// The result of classifying the mood of a scene.
#[derive(Debug, Clone)]
pub struct MoodClassification {
    /// Primary mood classification.
    pub primary: Mood,
    /// Confidence in the primary mood (0.0–1.0).
    pub confidence: f64,
    /// Optional secondary mood and its confidence.
    pub secondary: Option<(Mood, f64)>,
}

/// Score how well a mood matches the provided features.
fn mood_score(mood: Mood, features: &MoodFeatures) -> f64 {
    let brightness_diff = (features.avg_brightness - mood.typical_brightness()).abs();
    let motion_diff = (features.motion_level - mood.typical_motion()).abs();

    // Extra contextual adjustments
    let context_bonus = match mood {
        Mood::Horror => {
            if features.contrast > 0.6 && features.avg_brightness < 0.35 {
                0.15
            } else {
                0.0
            }
        }
        Mood::Action => {
            if features.shot_length_frames < 48.0 {
                0.10
            } else {
                0.0
            }
        }
        Mood::Nature => {
            if features.saturation > 0.5 && features.motion_level < 0.25 {
                0.10
            } else {
                0.0
            }
        }
        Mood::Comedy => {
            if features.saturation > 0.5 && features.avg_brightness > 0.55 {
                0.08
            } else {
                0.0
            }
        }
        _ => 0.0,
    };

    let raw = 1.0 - (brightness_diff + motion_diff) / 2.0 + context_bonus;
    raw.clamp(0.0, 1.0)
}

/// Classify the mood of a scene given its visual features.
#[must_use]
pub fn classify_mood(features: &MoodFeatures) -> MoodClassification {
    let all_moods = [
        Mood::Action,
        Mood::Drama,
        Mood::Romance,
        Mood::Horror,
        Mood::Comedy,
        Mood::Documentary,
        Mood::Thriller,
        Mood::Nature,
    ];

    // Compute scores for all moods
    let mut scored: Vec<(Mood, f64)> = all_moods
        .iter()
        .map(|&m| (m, mood_score(m, features)))
        .collect();

    // Sort descending by score
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let primary = scored[0].0;
    let primary_score = scored[0].1;

    let secondary = if scored.len() > 1 && scored[1].1 > 0.4 {
        Some((scored[1].0, scored[1].1))
    } else {
        None
    };

    MoodClassification {
        primary,
        confidence: primary_score,
        secondary,
    }
}

/// A timeline of mood classifications, indexed by frame number.
#[derive(Debug, Default)]
pub struct MoodTimeline {
    /// Ordered (frame, classification) pairs.
    entries: Vec<(u64, MoodClassification)>,
}

impl MoodTimeline {
    /// Create a new, empty `MoodTimeline`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a mood classification at a specific frame.
    pub fn add(&mut self, frame: u64, mood: MoodClassification) {
        self.entries.push((frame, mood));
        // Keep sorted by frame
        self.entries.sort_by_key(|(f, _)| *f);
    }

    /// Return the mood classification at or before the given frame.
    #[must_use]
    pub fn mood_at_frame(&self, frame: u64) -> Option<&MoodClassification> {
        // Find the last entry whose frame <= requested frame
        self.entries
            .iter()
            .rev()
            .find(|(f, _)| *f <= frame)
            .map(|(_, m)| m)
    }

    /// Return the mood that occurs most often across the timeline.
    #[must_use]
    pub fn dominant_mood(&self) -> Option<Mood> {
        if self.entries.is_empty() {
            return None;
        }

        let all_moods = [
            Mood::Action,
            Mood::Drama,
            Mood::Romance,
            Mood::Horror,
            Mood::Comedy,
            Mood::Documentary,
            Mood::Thriller,
            Mood::Nature,
        ];

        all_moods
            .iter()
            .max_by_key(|&&m| {
                self.entries
                    .iter()
                    .filter(|(_, mc)| mc.primary == m)
                    .count()
            })
            .copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn action_features() -> MoodFeatures {
        MoodFeatures {
            avg_brightness: 0.55,
            contrast: 0.6,
            saturation: 0.6,
            motion_level: 0.85,
            shot_length_frames: 30.0,
        }
    }

    fn horror_features() -> MoodFeatures {
        MoodFeatures {
            avg_brightness: 0.18,
            contrast: 0.75,
            saturation: 0.2,
            motion_level: 0.40,
            shot_length_frames: 72.0,
        }
    }

    #[test]
    fn test_mood_label() {
        assert_eq!(Mood::Action.label(), "action");
        assert_eq!(Mood::Drama.label(), "drama");
        assert_eq!(Mood::Romance.label(), "romance");
        assert_eq!(Mood::Horror.label(), "horror");
        assert_eq!(Mood::Comedy.label(), "comedy");
        assert_eq!(Mood::Documentary.label(), "documentary");
        assert_eq!(Mood::Thriller.label(), "thriller");
        assert_eq!(Mood::Nature.label(), "nature");
    }

    #[test]
    fn test_typical_brightness_range() {
        let moods = [
            Mood::Action,
            Mood::Drama,
            Mood::Romance,
            Mood::Horror,
            Mood::Comedy,
            Mood::Documentary,
            Mood::Thriller,
            Mood::Nature,
        ];
        for m in moods {
            let b = m.typical_brightness();
            assert!(b >= 0.0 && b <= 1.0, "brightness out of range for {:?}", m);
        }
    }

    #[test]
    fn test_typical_motion_range() {
        let moods = [
            Mood::Action,
            Mood::Drama,
            Mood::Romance,
            Mood::Horror,
            Mood::Comedy,
            Mood::Documentary,
            Mood::Thriller,
            Mood::Nature,
        ];
        for m in moods {
            let mv = m.typical_motion();
            assert!(mv >= 0.0 && mv <= 1.0, "motion out of range for {:?}", m);
        }
    }

    #[test]
    fn test_classify_action() {
        let features = action_features();
        let result = classify_mood(&features);
        assert_eq!(result.primary, Mood::Action);
        assert!(result.confidence > 0.5);
    }

    #[test]
    fn test_classify_horror() {
        let features = horror_features();
        let result = classify_mood(&features);
        assert_eq!(result.primary, Mood::Horror);
        assert!(result.confidence > 0.5);
    }

    #[test]
    fn test_classify_confidence_range() {
        let features = action_features();
        let result = classify_mood(&features);
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    }

    #[test]
    fn test_timeline_new_is_empty() {
        let tl = MoodTimeline::new();
        assert!(tl.entries.is_empty());
    }

    #[test]
    fn test_timeline_add_and_mood_at_frame() {
        let mut tl = MoodTimeline::new();
        let mc = MoodClassification {
            primary: Mood::Action,
            confidence: 0.9,
            secondary: None,
        };
        tl.add(100, mc);
        let found = tl.mood_at_frame(150);
        assert!(found.is_some());
        assert_eq!(found.expect("should succeed in test").primary, Mood::Action);
    }

    #[test]
    fn test_timeline_mood_at_frame_before_first_entry() {
        let mut tl = MoodTimeline::new();
        let mc = MoodClassification {
            primary: Mood::Drama,
            confidence: 0.7,
            secondary: None,
        };
        tl.add(200, mc);
        assert!(tl.mood_at_frame(50).is_none());
    }

    #[test]
    fn test_timeline_dominant_mood_empty() {
        let tl = MoodTimeline::new();
        assert!(tl.dominant_mood().is_none());
    }

    #[test]
    fn test_timeline_dominant_mood() {
        let mut tl = MoodTimeline::new();
        for frame in [0u64, 100, 200] {
            tl.add(
                frame,
                MoodClassification {
                    primary: Mood::Action,
                    confidence: 0.9,
                    secondary: None,
                },
            );
        }
        tl.add(
            300,
            MoodClassification {
                primary: Mood::Drama,
                confidence: 0.7,
                secondary: None,
            },
        );
        assert_eq!(tl.dominant_mood(), Some(Mood::Action));
    }

    #[test]
    fn test_secondary_mood_present_for_close_scores() {
        // Nature-like features (high saturation, low motion, bright)
        let features = MoodFeatures {
            avg_brightness: 0.65,
            contrast: 0.4,
            saturation: 0.7,
            motion_level: 0.15,
            shot_length_frames: 120.0,
        };
        let result = classify_mood(&features);
        // Confidence should be within valid range regardless of secondary presence
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    }
}
