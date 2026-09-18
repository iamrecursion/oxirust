//! Genre enum and rule-based genre classification from pre-extracted features.
//!
//! The `Genre` enum covers 15 categories commonly found in music catalogues.
//! The `classify_genre` function implements a multi-class scoring system that
//! maps low-level audio features to genre probabilities without requiring a
//! trained ML model, making it fully deterministic and patent-free.

#![allow(dead_code, clippy::cast_precision_loss)]

/// Broad genre categories for music classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Genre {
    /// Electronic / dance music (EDM, techno, house, synth-pop).
    Electronic,
    /// Rock (classic rock, indie, alternative).
    Rock,
    /// Pop (mainstream chart music).
    Pop,
    /// Classical / orchestral.
    Classical,
    /// Jazz (bebop, smooth jazz, fusion).
    Jazz,
    /// Hip-hop / rap.
    HipHop,
    /// Country / Americana.
    Country,
    /// R&B / soul.
    RnB,
    /// Metal (heavy metal, thrash, death metal).
    Metal,
    /// Folk / acoustic.
    Folk,
    /// Latin (salsa, reggaeton, bossa nova).
    Latin,
    /// World music (non-Western traditional styles).
    World,
    /// Ambient / new-age / drone.
    Ambient,
    /// Film/TV soundtrack / score.
    Soundtrack,
    /// Could not determine genre with sufficient confidence.
    Unknown,
}

impl Genre {
    /// Return a human-readable English name for this genre.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Electronic => "Electronic",
            Self::Rock => "Rock",
            Self::Pop => "Pop",
            Self::Classical => "Classical",
            Self::Jazz => "Jazz",
            Self::HipHop => "HipHop",
            Self::Country => "Country",
            Self::RnB => "RnB",
            Self::Metal => "Metal",
            Self::Folk => "Folk",
            Self::Latin => "Latin",
            Self::World => "World",
            Self::Ambient => "Ambient",
            Self::Soundtrack => "Soundtrack",
            Self::Unknown => "Unknown",
        }
    }

    /// All genre variants except `Unknown`.
    #[must_use]
    pub fn all_known() -> &'static [Genre] {
        &[
            Self::Electronic,
            Self::Rock,
            Self::Pop,
            Self::Classical,
            Self::Jazz,
            Self::HipHop,
            Self::Country,
            Self::RnB,
            Self::Metal,
            Self::Folk,
            Self::Latin,
            Self::World,
            Self::Ambient,
            Self::Soundtrack,
        ]
    }
}

impl std::fmt::Display for Genre {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

// ── Internal scoring helpers ──────────────────────────────────────────────────

/// Linearly score a value based on how close it is to an ideal range.
///
/// Returns 1.0 if `value` is inside `[ideal_min, ideal_max]`, tapering to 0.0
/// as it moves further away (with `tolerance` defining the half-width of the
/// linear ramp on each side).
fn range_score(value: f32, ideal_min: f32, ideal_max: f32, tolerance: f32) -> f32 {
    if value >= ideal_min && value <= ideal_max {
        return 1.0;
    }
    let tol = tolerance.max(1e-6);
    if value < ideal_min {
        ((value - (ideal_min - tol)) / tol).clamp(0.0, 1.0)
    } else {
        ((ideal_max + tol - value) / tol).clamp(0.0, 1.0)
    }
}

/// Score a feature that should be above a threshold.
fn above_score(value: f32, threshold: f32, tolerance: f32) -> f32 {
    range_score(value, threshold, f32::INFINITY, tolerance)
}

/// Score a feature that should be below a threshold.
fn below_score(value: f32, threshold: f32, tolerance: f32) -> f32 {
    range_score(value, f32::NEG_INFINITY, threshold, tolerance)
}

/// Compute variance of a 12-element chroma array.
fn chroma_variance(chroma: &[f32; 12]) -> f32 {
    let mean: f32 = chroma.iter().sum::<f32>() / 12.0;
    chroma.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / 12.0
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Classify the genre of an audio excerpt from pre-extracted features.
///
/// Uses a weighted rule-based scoring system: each genre accumulates a score
/// in `[0.0, 1.0]` based on how well the supplied features match its
/// characteristic profile.  The genre with the highest score is returned
/// together with a normalised confidence value.
///
/// # Arguments
/// * `spectral_centroid`  – Normalised spectral centroid (0.0–1.0, where 1.0
///                          equals the Nyquist frequency).
/// * `spectral_rolloff`   – Normalised spectral rolloff (0.0–1.0).
/// * `zero_crossing_rate` – Zero-crossing rate (0.0–1.0).
/// * `chroma`             – 12-bin chromagram (values need not sum to 1).
/// * `tempo_bpm`          – Estimated tempo in BPM (0 if unknown).
///
/// # Returns
/// `(Genre, f32)` — best-matching genre and confidence in `[0.0, 1.0]`.
#[must_use]
pub fn classify_genre(
    spectral_centroid: f32,
    spectral_rolloff: f32,
    zero_crossing_rate: f32,
    chroma: &[f32; 12],
    tempo_bpm: f32,
) -> (Genre, f32) {
    let chroma_var = chroma_variance(chroma);

    // ── Score each genre ───────────────────────────────────────────────────
    // Weights are chosen so that the dominant perceptual features contribute
    // most to the score, and secondary cues provide tie-breaking.

    let scores: [(Genre, f32); 14] = [
        // Electronic: bright spectrum, strong rhythmic regularity, ~120-145 BPM
        (
            Genre::Electronic,
            above_score(spectral_centroid, 0.25, 0.10) * 0.30
                + above_score(spectral_rolloff, 0.30, 0.10) * 0.20
                + range_score(tempo_bpm, 118.0, 148.0, 20.0) * 0.30
                + below_score(zero_crossing_rate, 0.12, 0.05) * 0.20,
        ),
        // Rock: high energy, high ZCR, moderate-fast tempo
        (
            Genre::Rock,
            above_score(spectral_rolloff, 0.35, 0.10) * 0.25
                + above_score(zero_crossing_rate, 0.08, 0.04) * 0.30
                + range_score(tempo_bpm, 108.0, 165.0, 20.0) * 0.30
                + above_score(spectral_centroid, 0.18, 0.08) * 0.15,
        ),
        // Metal: very high ZCR, bright/dense spectrum, fast tempo
        (
            Genre::Metal,
            above_score(zero_crossing_rate, 0.14, 0.04) * 0.35
                + above_score(spectral_centroid, 0.25, 0.08) * 0.25
                + above_score(spectral_rolloff, 0.40, 0.10) * 0.20
                + range_score(tempo_bpm, 130.0, 200.0, 25.0) * 0.20,
        ),
        // Classical: low ZCR, wide dynamic range (high rolloff variety), slow/varied tempo
        (
            Genre::Classical,
            below_score(zero_crossing_rate, 0.05, 0.03) * 0.35
                + above_score(spectral_rolloff, 0.20, 0.10) * 0.20
                + below_score(spectral_centroid, 0.25, 0.10) * 0.20
                + above_score(chroma_var, 0.01, 0.005) * 0.25,
        ),
        // Jazz: medium centroid, high harmonic complexity, moderate tempo
        (
            Genre::Jazz,
            range_score(spectral_centroid, 0.12, 0.30, 0.08) * 0.25
                + above_score(chroma_var, 0.015, 0.005) * 0.35
                + range_score(tempo_bpm, 80.0, 140.0, 20.0) * 0.25
                + range_score(zero_crossing_rate, 0.04, 0.10, 0.03) * 0.15,
        ),
        // Hip-hop: low-mid centroid (bass-heavy), low-mid tempo, moderate ZCR
        (
            Genre::HipHop,
            below_score(spectral_centroid, 0.20, 0.08) * 0.30
                + range_score(tempo_bpm, 70.0, 110.0, 15.0) * 0.35
                + range_score(zero_crossing_rate, 0.03, 0.09, 0.03) * 0.20
                + below_score(spectral_rolloff, 0.35, 0.10) * 0.15,
        ),
        // Pop: moderate everything, steady beat, broad appeal tempo range
        (
            Genre::Pop,
            range_score(spectral_centroid, 0.15, 0.30, 0.08) * 0.20
                + range_score(tempo_bpm, 100.0, 132.0, 15.0) * 0.40
                + range_score(spectral_rolloff, 0.25, 0.45, 0.10) * 0.25
                + range_score(zero_crossing_rate, 0.05, 0.12, 0.03) * 0.15,
        ),
        // Country: warm/mid-bright tone, moderate tempo, low ZCR (acoustic guitar)
        (
            Genre::Country,
            range_score(spectral_centroid, 0.12, 0.25, 0.06) * 0.25
                + below_score(zero_crossing_rate, 0.08, 0.04) * 0.25
                + range_score(tempo_bpm, 78.0, 132.0, 15.0) * 0.35
                + range_score(spectral_rolloff, 0.20, 0.40, 0.08) * 0.15,
        ),
        // R&B: warm low-mid spectrum, slow-mid tempo, smooth ZCR
        (
            Genre::RnB,
            below_score(spectral_centroid, 0.22, 0.08) * 0.30
                + range_score(tempo_bpm, 60.0, 100.0, 15.0) * 0.35
                + range_score(zero_crossing_rate, 0.03, 0.08, 0.03) * 0.25
                + range_score(spectral_rolloff, 0.15, 0.35, 0.08) * 0.10,
        ),
        // Folk: low centroid (acoustic, no bass boost), very low ZCR, slow tempo
        (
            Genre::Folk,
            below_score(spectral_centroid, 0.15, 0.06) * 0.35
                + below_score(zero_crossing_rate, 0.06, 0.03) * 0.30
                + range_score(tempo_bpm, 60.0, 130.0, 20.0) * 0.20
                + below_score(spectral_rolloff, 0.30, 0.10) * 0.15,
        ),
        // Latin: bright mid-high centroid, syncopated moderate-fast tempo
        (
            Genre::Latin,
            range_score(spectral_centroid, 0.18, 0.32, 0.08) * 0.25
                + range_score(tempo_bpm, 95.0, 145.0, 20.0) * 0.40
                + range_score(spectral_rolloff, 0.25, 0.45, 0.10) * 0.20
                + above_score(zero_crossing_rate, 0.06, 0.03) * 0.15,
        ),
        // Ambient: very low ZCR, very low centroid, slow/absent beat
        (
            Genre::Ambient,
            below_score(zero_crossing_rate, 0.04, 0.02) * 0.35
                + below_score(spectral_centroid, 0.12, 0.05) * 0.30
                + range_score(tempo_bpm, 0.0, 80.0, 20.0) * 0.20
                + below_score(spectral_rolloff, 0.20, 0.08) * 0.15,
        ),
        // Soundtrack: wide spectral range (high rolloff + high centroid), varied tempo
        (
            Genre::Soundtrack,
            above_score(spectral_rolloff, 0.40, 0.10) * 0.30
                + above_score(spectral_centroid, 0.20, 0.08) * 0.25
                + above_score(chroma_var, 0.012, 0.005) * 0.25
                + range_score(tempo_bpm, 60.0, 160.0, 30.0) * 0.20,
        ),
        // World: moderate centroid, high chroma variance (exotic scales/modes)
        (
            Genre::World,
            range_score(spectral_centroid, 0.10, 0.25, 0.08) * 0.25
                + above_score(chroma_var, 0.018, 0.005) * 0.40
                + range_score(tempo_bpm, 60.0, 160.0, 30.0) * 0.20
                + range_score(zero_crossing_rate, 0.04, 0.12, 0.04) * 0.15,
        ),
    ];

    // Find the highest-scoring genre
    let (best_genre, best_score) =
        scores
            .iter()
            .copied()
            .fold((Genre::Unknown, 0.0_f32), |acc, (g, s)| {
                if s > acc.1 {
                    (g, s)
                } else {
                    acc
                }
            });

    // Normalise confidence: find the sum of all scores and scale best
    let total: f32 = scores.iter().map(|(_, s)| s).sum();
    let confidence = if total > 0.0 {
        (best_score / total * scores.len() as f32).clamp(0.0, 1.0)
    } else {
        0.0
    };

    if best_score <= 0.0 {
        (Genre::Unknown, 0.0)
    } else {
        (best_genre, confidence)
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genre_electronic_name() {
        assert_eq!(Genre::Electronic.name(), "Electronic");
    }

    #[test]
    fn test_genre_unknown_name() {
        assert_eq!(Genre::Unknown.name(), "Unknown");
    }

    #[test]
    fn test_genre_display_trait() {
        let g = Genre::Rock;
        let s = format!("{g}");
        assert_eq!(s, "Rock");
    }

    #[test]
    fn test_all_known_genres_have_non_empty_names() {
        for g in Genre::all_known() {
            assert!(!g.name().is_empty(), "Genre {g:?} has an empty name");
        }
    }

    #[test]
    fn test_classify_genre_confidence_in_range() {
        // High centroid, high rolloff, moderate tempo → Electronic or similar
        let chroma = [1.0_f32 / 12.0; 12];
        let (_, conf) = classify_genre(0.40, 0.50, 0.05, &chroma, 130.0);
        assert!(
            (0.0..=1.0).contains(&conf),
            "Confidence must be in [0, 1], got {conf}"
        );
    }

    #[test]
    fn test_classify_genre_high_centroid_high_tempo_not_ambient() {
        // Bright spectrum at 130 BPM should not be classified as Ambient
        let chroma = [1.0_f32 / 12.0; 12];
        let (genre, _) = classify_genre(0.40, 0.55, 0.08, &chroma, 130.0);
        assert_ne!(
            genre,
            Genre::Ambient,
            "Bright, 130 BPM signal should not be Ambient"
        );
    }

    #[test]
    fn test_classify_genre_ambient_features() {
        // Very low ZCR, very low centroid, slow tempo → Ambient or Classical
        let chroma = [1.0_f32 / 12.0; 12];
        let (genre, conf) = classify_genre(0.05, 0.08, 0.01, &chroma, 40.0);
        assert!(
            matches!(genre, Genre::Ambient | Genre::Classical | Genre::Folk),
            "Low-energy slow signal should be Ambient/Classical/Folk, got {genre:?} ({conf})"
        );
    }

    #[test]
    fn test_classify_genre_high_zcr_high_centroid_tends_metal_or_rock() {
        // Very high ZCR, bright spectrum, fast tempo
        let chroma = [1.0_f32 / 12.0; 12];
        let (genre, _) = classify_genre(0.40, 0.60, 0.20, &chroma, 160.0);
        assert!(
            matches!(genre, Genre::Metal | Genre::Rock | Genre::Electronic),
            "High ZCR/centroid fast signal should be Metal/Rock/Electronic, got {genre:?}"
        );
    }

    #[test]
    fn test_classify_genre_not_unknown_for_valid_features() {
        // Any valid feature set should produce a non-Unknown result
        let chroma = [
            0.1_f32, 0.05, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05,
        ];
        let (genre, _) = classify_genre(0.20, 0.30, 0.07, &chroma, 120.0);
        assert_ne!(
            genre,
            Genre::Unknown,
            "Typical features should yield a known genre"
        );
    }

    #[test]
    fn test_genre_all_known_count() {
        assert_eq!(Genre::all_known().len(), 14);
    }

    #[test]
    fn test_classify_genre_returns_genre_and_confidence() {
        let chroma = [1.0_f32 / 12.0; 12];
        let result = classify_genre(0.25, 0.35, 0.06, &chroma, 120.0);
        // Destructuring should work and confidence should be finite
        let (_, conf) = result;
        assert!(conf.is_finite());
    }
}
