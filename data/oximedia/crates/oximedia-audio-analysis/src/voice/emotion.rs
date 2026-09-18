//! Emotion detection from voice characteristics.

/// Emotion classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Emotion {
    /// Neutral emotion
    Neutral,
    /// Happy/joyful
    Happy,
    /// Angry
    Angry,
    /// Sad
    Sad,
    /// Fearful
    Fearful,
    /// Unknown
    Unknown,
}

/// Detect emotion from voice characteristics.
///
/// Emotion affects voice prosody:
/// - Happy: Higher F0, high variability, fast speech
/// - Angry: Higher F0, high intensity, rapid changes
/// - Sad: Lower F0, low variability, slow speech
/// - Fear: Variable F0, high jitter
/// - Neutral: Moderate F0, low variability
///
/// # Arguments
/// * `f0` - Fundamental frequency in Hz
/// * `jitter` - Pitch variation (0-1)
/// * `shimmer` - Amplitude variation (0-1)
/// * `formants` - Formant frequencies
///
/// # Returns
/// Detected emotion
#[must_use]
pub fn detect_emotion(f0: f32, jitter: f32, shimmer: f32, formants: &[f32]) -> Emotion {
    // Compute prosodic features
    let f0_high = f0 > 200.0;
    let f0_low = f0 < 120.0;
    let high_variability = jitter > 0.015 || shimmer > 0.08;
    let low_variability = jitter < 0.008 && shimmer < 0.04;

    // Energy (approximated by shimmer)
    let high_energy = shimmer > 0.1;

    // Formant-based features
    let f1_high = formants.first().is_some_and(|&f1| f1 > 650.0);

    // Emotion classification rules
    if f0_high && high_variability && high_energy {
        return Emotion::Angry;
    }

    if f0_high && high_variability && !high_energy && f1_high {
        return Emotion::Happy;
    }

    if f0_low && low_variability {
        return Emotion::Sad;
    }

    if high_variability && !high_energy {
        return Emotion::Fearful;
    }

    if low_variability {
        return Emotion::Neutral;
    }

    Emotion::Unknown
}

/// Emotion detection with confidence scores for all emotions.
#[must_use]
pub fn detect_emotion_scores(
    f0: f32,
    jitter: f32,
    shimmer: f32,
    formants: &[f32],
) -> EmotionScores {
    let f0_normalized = (f0 - 150.0) / 100.0; // Normalize around average
    let jitter_normalized = jitter / 0.02;
    let shimmer_normalized = shimmer / 0.1;

    let f1 = formants.first().copied().unwrap_or(500.0);
    let f1_normalized = (f1 - 500.0) / 200.0;

    // Score each emotion based on features
    let happy_score = ((f0_normalized * 0.4).max(0.0)
        + (f1_normalized * 0.3).max(0.0)
        + (jitter_normalized * 0.3).max(0.0))
    .min(1.0);

    let angry_score =
        ((f0_normalized * 0.5).max(0.0) + (shimmer_normalized * 0.5).max(0.0)).min(1.0);

    let sad_score =
        (((-f0_normalized) * 0.5).max(0.0) + ((1.0 - jitter_normalized) * 0.5).max(0.0)).min(1.0);

    let fearful_score = (jitter_normalized * 0.7).min(1.0);

    let neutral_score = (1.0 - (jitter_normalized.abs() + shimmer_normalized.abs()) / 2.0).max(0.0);

    EmotionScores {
        neutral: neutral_score,
        happy: happy_score,
        angry: angry_score,
        sad: sad_score,
        fearful: fearful_score,
    }
}

/// Emotion scores for all categories.
#[derive(Debug, Clone)]
pub struct EmotionScores {
    /// Neutral emotion score (0-1)
    pub neutral: f32,
    /// Happy emotion score (0-1)
    pub happy: f32,
    /// Angry emotion score (0-1)
    pub angry: f32,
    /// Sad emotion score (0-1)
    pub sad: f32,
    /// Fearful emotion score (0-1)
    pub fearful: f32,
}

impl EmotionScores {
    /// Get the dominant emotion and its score.
    #[must_use]
    pub fn dominant(&self) -> (Emotion, f32) {
        let scores = [
            (Emotion::Neutral, self.neutral),
            (Emotion::Happy, self.happy),
            (Emotion::Angry, self.angry),
            (Emotion::Sad, self.sad),
            (Emotion::Fearful, self.fearful),
        ];

        scores
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .copied()
            .unwrap_or((Emotion::Unknown, 0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emotion_detection() {
        // Happy: high F0, high variability
        let happy = detect_emotion(230.0, 0.02, 0.06, &[700.0, 1800.0]);
        assert_eq!(happy, Emotion::Happy);

        // Sad: low F0, low variability
        let sad = detect_emotion(100.0, 0.005, 0.03, &[500.0, 1500.0]);
        assert_eq!(sad, Emotion::Sad);

        // Angry: high F0, high energy
        let angry = detect_emotion(250.0, 0.02, 0.15, &[600.0, 1600.0]);
        assert_eq!(angry, Emotion::Angry);
    }

    #[test]
    fn test_emotion_scores() {
        let scores = detect_emotion_scores(230.0, 0.02, 0.06, &[700.0, 1800.0]);
        let (dominant, score) = scores.dominant();
        assert!(score > 0.0);
        assert_ne!(dominant, Emotion::Unknown);
    }

    // ── Wave 27: ordinal / monotonic synthetic-signal pins ──────────────────────

    /// High-arousal prosody (high F0, high jitter/shimmer) must rank higher in the
    /// arousal-linked emotions (angry, happy) and lower in the calm emotions (sad,
    /// neutral) than a low-arousal voice — a robustness check that does NOT pin
    /// exact heuristic magnitudes, only their ORDER.
    ///
    /// The directions below were verified against the scoring math in
    /// `detect_emotion_scores`:
    /// - `angry`/`happy` rise with `f0_normalized` (and shimmer/jitter), so the
    ///   aroused voice scores higher.
    /// - `sad` rises with `-f0_normalized` and `(1 - jitter_normalized)`, so the
    ///   calm voice scores higher.
    /// - `neutral` is `1 - (|jitter_norm| + |shimmer_norm|)/2`, so low variability
    ///   (calm) scores higher.
    #[test]
    fn test_emotion_scores_high_arousal_vs_calm_ordinal() {
        let aroused = detect_emotion_scores(260.0, 0.025, 0.14, &[700.0, 1800.0]);
        let calm = detect_emotion_scores(95.0, 0.004, 0.02, &[480.0, 1500.0]);

        // Arousal-linked emotions: aroused > calm.
        assert!(
            aroused.angry > calm.angry,
            "aroused.angry ({}) should exceed calm.angry ({})",
            aroused.angry,
            calm.angry
        );
        assert!(
            aroused.happy > calm.happy,
            "aroused.happy ({}) should exceed calm.happy ({})",
            aroused.happy,
            calm.happy
        );

        // Calm-linked emotions: calm > aroused.
        assert!(
            calm.sad > aroused.sad,
            "calm.sad ({}) should exceed aroused.sad ({})",
            calm.sad,
            aroused.sad
        );
        assert!(
            calm.neutral > aroused.neutral,
            "calm.neutral ({}) should exceed aroused.neutral ({})",
            calm.neutral,
            aroused.neutral
        );

        // Dominant emotion direction (robust to the angry/happy tie — both are
        // arousal-linked, so either is acceptable for the aroused case).
        assert!(
            matches!(aroused.dominant().0, Emotion::Angry | Emotion::Happy),
            "aroused dominant should be Angry or Happy, got {:?}",
            aroused.dominant().0
        );
        assert!(
            matches!(calm.dominant().0, Emotion::Sad | Emotion::Neutral),
            "calm dominant should be Sad or Neutral, got {:?}",
            calm.dominant().0
        );
    }

    /// As F0 rises (holding jitter/shimmer/formants fixed) the `angry` score must
    /// be non-decreasing and the `sad` score non-increasing — the monotonic
    /// signature of the arousal/valence heuristic.
    #[test]
    fn test_emotion_scores_monotone_in_f0() {
        let jitter = 0.02_f32;
        let shimmer = 0.12_f32;
        let formants = [700.0_f32, 1800.0_f32];

        let scores: Vec<EmotionScores> = [100.0_f32, 150.0, 200.0, 260.0]
            .iter()
            .map(|&f0| detect_emotion_scores(f0, jitter, shimmer, &formants))
            .collect();

        assert!(
            scores.windows(2).all(|w| w[1].angry >= w[0].angry - 1e-6),
            "angry score should be non-decreasing in F0: {:?}",
            scores.iter().map(|s| s.angry).collect::<Vec<_>>()
        );
        assert!(
            scores.windows(2).all(|w| w[1].sad <= w[0].sad + 1e-6),
            "sad score should be non-increasing in F0: {:?}",
            scores.iter().map(|s| s.sad).collect::<Vec<_>>()
        );
    }
}
