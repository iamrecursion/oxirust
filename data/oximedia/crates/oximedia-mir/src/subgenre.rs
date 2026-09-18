//! Sub-genre classification extending the top-level genre system.
//!
//! Each broad [`crate::genre::Genre`] is mapped to a set of sub-genres.
//! Sub-genre scoring uses the same feature vector as the top-level classifier
//! but applies genre-specific discriminant weights.
//!
//! # Sub-genre taxonomy
//!
//! | Top-level | Sub-genres |
//! |-----------|-----------|
//! | Electronic | House, Techno, Drum&Bass, Ambient Techno, Synthwave |
//! | Rock | Classic Rock, Indie, Punk, Progressive, Grunge |
//! | Pop | Dance Pop, Synth Pop, Indie Pop, K-Pop, Acoustic Pop |
//! | Classical | Baroque, Romantic, Contemporary, Minimalist, Chamber |
//! | Jazz | Bebop, Smooth Jazz, Fusion, Free Jazz, Swing |
//! | Hip-Hop | Trap, Old School, Lo-Fi, Boom Bap, Cloud Rap |
//! | Metal | Heavy Metal, Death Metal, Black Metal, Doom, Nu-Metal |
//! | Folk | Singer-Songwriter, Celtic, Bluegrass, Indie Folk, Americana |

// ---------------------------------------------------------------------------
// Sub-genre enum
// ---------------------------------------------------------------------------

/// Sub-genre labels, grouped under their parent genre.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SubGenre {
    // Electronic
    /// Four-to-the-floor beats, 120–130 BPM (Chicago house lineage).
    House,
    /// Repetitive, hypnotic kick patterns, 130–145 BPM (Detroit / Berlin).
    Techno,
    /// Fast syncopated breaks, 160–180 BPM, heavy bass.
    DrumAndBass,
    /// Electronic music with ambient / atmospheric textures.
    AmbientTechno,
    /// Retro-futurist synthesizer aesthetics, 80–110 BPM.
    Synthwave,

    // Rock
    /// Classic rock from the 1960s–80s, guitar-driven, 100–140 BPM.
    ClassicRock,
    /// Independent / alternative rock, jangly guitars.
    IndieRock,
    /// Short, fast, aggressive songs; high ZCR.
    Punk,
    /// Long compositions, complex arrangements.
    ProgRock,
    /// Distorted guitars, slow-to-moderate tempo, heavy.
    Grunge,

    // Pop
    /// Dance-oriented pop, four-on-the-floor, 115–128 BPM.
    DancePop,
    /// Synthesizer-dominated pop (Depeche Mode / 80s lineage).
    SynthPop,
    /// Indie-influenced pop with a lo-fi texture.
    IndiePop,
    /// Korean pop: bright spectral centroid, high energy, 100–135 BPM.
    KPop,
    /// Acoustic guitar / piano driven pop, lower energy.
    AcousticPop,

    // Classical
    /// Highly ornamented counterpoint, 1600–1750.
    Baroque,
    /// Expressive, dynamic, 1820–1900.
    Romantic,
    /// Post-tonal, experimental, 1900–present.
    Contemporary,
    /// Repetitive motifs, gradual variation.
    Minimalist,
    /// Small ensemble (string quartet etc.).
    Chamber,

    // Jazz
    /// Fast tempos, complex harmony, virtuosic improvisation.
    Bebop,
    /// Gentle, melodic, commercial jazz.
    SmoothJazz,
    /// Jazz + rock/funk elements.
    JazzFusion,
    /// Atonal or non-metric free improvisation.
    FreeJazz,
    /// Big-band / swing era, 1930s–40s.
    Swing,

    // Hip-Hop
    /// Heavy 808 bass, hi-hat rolls, 130–160 BPM, trap.
    Trap,
    /// Sampling-heavy, break-beat driven, 85–100 BPM.
    BoomBap,
    /// Lo-fi hip-hop: crate-digging aesthetics, warm tape saturation.
    LoFiHipHop,
    /// Classic East/West Coast hip-hop, 90s era.
    OldSchoolHipHop,
    /// Dreamy, ethereal, minimal lyrics.
    CloudRap,

    // Metal
    /// Down-tuned, power chords, 100–160 BPM.
    HeavyMetal,
    /// Blast-beats, guttural vocals, 160–220 BPM.
    DeathMetal,
    /// High-pitched shrieks, tremolo riffs, cold atmosphere.
    BlackMetal,
    /// Slow, ultra-heavy riffs, 50–80 BPM.
    DoomMetal,
    /// Cross-over with hip-hop / rap elements, 90–120 BPM.
    NuMetal,

    // Folk
    /// Solo/duo acoustic performance, lyric-centric.
    SingerSongwriter,
    /// Traditional Irish / Scottish music.
    Celtic,
    /// American roots music with banjo and mandolin.
    Bluegrass,
    /// Folk sensibility with indie-rock production.
    IndieFolk,
    /// American Americana / country-folk hybrid.
    Americana,

    /// Catch-all when no sub-genre is identified confidently.
    Unknown,
}

impl SubGenre {
    /// Human-readable name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::House => "House",
            Self::Techno => "Techno",
            Self::DrumAndBass => "Drum & Bass",
            Self::AmbientTechno => "Ambient Techno",
            Self::Synthwave => "Synthwave",
            Self::ClassicRock => "Classic Rock",
            Self::IndieRock => "Indie Rock",
            Self::Punk => "Punk",
            Self::ProgRock => "Progressive Rock",
            Self::Grunge => "Grunge",
            Self::DancePop => "Dance Pop",
            Self::SynthPop => "Synth Pop",
            Self::IndiePop => "Indie Pop",
            Self::KPop => "K-Pop",
            Self::AcousticPop => "Acoustic Pop",
            Self::Baroque => "Baroque",
            Self::Romantic => "Romantic",
            Self::Contemporary => "Contemporary",
            Self::Minimalist => "Minimalist",
            Self::Chamber => "Chamber Music",
            Self::Bebop => "Bebop",
            Self::SmoothJazz => "Smooth Jazz",
            Self::JazzFusion => "Jazz Fusion",
            Self::FreeJazz => "Free Jazz",
            Self::Swing => "Swing",
            Self::Trap => "Trap",
            Self::BoomBap => "Boom Bap",
            Self::LoFiHipHop => "Lo-Fi Hip-Hop",
            Self::OldSchoolHipHop => "Old School Hip-Hop",
            Self::CloudRap => "Cloud Rap",
            Self::HeavyMetal => "Heavy Metal",
            Self::DeathMetal => "Death Metal",
            Self::BlackMetal => "Black Metal",
            Self::DoomMetal => "Doom Metal",
            Self::NuMetal => "Nu-Metal",
            Self::SingerSongwriter => "Singer-Songwriter",
            Self::Celtic => "Celtic",
            Self::Bluegrass => "Bluegrass",
            Self::IndieFolk => "Indie Folk",
            Self::Americana => "Americana",
            Self::Unknown => "Unknown",
        }
    }
}

impl std::fmt::Display for SubGenre {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

// ---------------------------------------------------------------------------
// Sub-genre classification result
// ---------------------------------------------------------------------------

/// Result of sub-genre classification.
#[derive(Debug, Clone)]
pub struct SubGenreResult {
    /// All scored sub-genres, sorted descending by confidence.
    pub scores: Vec<(SubGenre, f32)>,
    /// Top sub-genre.
    pub top_subgenre: SubGenre,
    /// Confidence of the top sub-genre (0–1).
    pub confidence: f32,
}

impl SubGenreResult {
    /// Whether a confident sub-genre was detected.
    #[must_use]
    pub fn is_confident(&self) -> bool {
        self.top_subgenre != SubGenre::Unknown && self.confidence > 0.0
    }
}

// ---------------------------------------------------------------------------
// Sub-genre classifier
// ---------------------------------------------------------------------------

/// Sub-genre classifier.
///
/// Scores sub-genres within the top-level genre using a weighted rule-based
/// approach.  All scoring uses deterministic arithmetic — no ML model required.
pub struct SubGenreClassifier;

impl SubGenreClassifier {
    /// Classify sub-genre.
    ///
    /// # Arguments
    ///
    /// * `top_genre_name` — The name returned by the top-level classifier
    ///   (e.g. `"electronic"`, `"rock"`, `"pop"`, …).
    /// * `spectral_centroid` — Normalised spectral centroid (0–1).
    /// * `spectral_rolloff`  — Normalised spectral rolloff (0–1).
    /// * `zero_crossing_rate` — ZCR (0–1).
    /// * `tempo_bpm`         — Estimated BPM (0 if unknown).
    /// * `energy`            — Normalised RMS energy (0–1).
    /// * `energy_variance`   — Variance of frame energy (0–1).
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn classify(
        top_genre_name: &str,
        spectral_centroid: f32,
        spectral_rolloff: f32,
        zero_crossing_rate: f32,
        tempo_bpm: f32,
        energy: f32,
        energy_variance: f32,
    ) -> SubGenreResult {
        let candidates = Self::candidates_for_genre(top_genre_name);
        if candidates.is_empty() {
            return SubGenreResult {
                scores: Vec::new(),
                top_subgenre: SubGenre::Unknown,
                confidence: 0.0,
            };
        }

        let mut scored: Vec<(SubGenre, f32)> = candidates
            .into_iter()
            .map(|sg| {
                let score = Self::score(
                    sg,
                    spectral_centroid,
                    spectral_rolloff,
                    zero_crossing_rate,
                    tempo_bpm,
                    energy,
                    energy_variance,
                );
                (sg, score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let total: f32 = scored.iter().map(|(_, s)| *s).sum();
        let (top_sg, top_raw) = scored.first().copied().unwrap_or((SubGenre::Unknown, 0.0));
        let confidence = if total > 0.0 {
            (top_raw / total * scored.len() as f32).clamp(0.0, 1.0)
        } else {
            0.0
        };

        SubGenreResult {
            scores: scored,
            top_subgenre: top_sg,
            confidence,
        }
    }

    /// Return the sub-genre candidates for a given top-level genre name.
    fn candidates_for_genre(genre: &str) -> Vec<SubGenre> {
        match genre.to_lowercase().trim() {
            "electronic" => vec![
                SubGenre::House,
                SubGenre::Techno,
                SubGenre::DrumAndBass,
                SubGenre::AmbientTechno,
                SubGenre::Synthwave,
            ],
            "rock" => vec![
                SubGenre::ClassicRock,
                SubGenre::IndieRock,
                SubGenre::Punk,
                SubGenre::ProgRock,
                SubGenre::Grunge,
            ],
            "pop" => vec![
                SubGenre::DancePop,
                SubGenre::SynthPop,
                SubGenre::IndiePop,
                SubGenre::KPop,
                SubGenre::AcousticPop,
            ],
            "classical" => vec![
                SubGenre::Baroque,
                SubGenre::Romantic,
                SubGenre::Contemporary,
                SubGenre::Minimalist,
                SubGenre::Chamber,
            ],
            "jazz" => vec![
                SubGenre::Bebop,
                SubGenre::SmoothJazz,
                SubGenre::JazzFusion,
                SubGenre::FreeJazz,
                SubGenre::Swing,
            ],
            "hip-hop" => vec![
                SubGenre::Trap,
                SubGenre::BoomBap,
                SubGenre::LoFiHipHop,
                SubGenre::OldSchoolHipHop,
                SubGenre::CloudRap,
            ],
            "metal" => vec![
                SubGenre::HeavyMetal,
                SubGenre::DeathMetal,
                SubGenre::BlackMetal,
                SubGenre::DoomMetal,
                SubGenre::NuMetal,
            ],
            "folk" => vec![
                SubGenre::SingerSongwriter,
                SubGenre::Celtic,
                SubGenre::Bluegrass,
                SubGenre::IndieFolk,
                SubGenre::Americana,
            ],
            _ => Vec::new(),
        }
    }

    /// Score a single sub-genre candidate.
    #[allow(clippy::too_many_arguments)]
    fn score(
        sg: SubGenre,
        sc: f32,  // spectral centroid
        sr: f32,  // spectral rolloff
        zcr: f32, // zero crossing rate
        bpm: f32,
        energy: f32,
        energy_var: f32,
    ) -> f32 {
        // Helper: proximity to a value, within tolerance
        let near = |v: f32, target: f32, tol: f32| -> f32 {
            ((tol - (v - target).abs()) / tol).clamp(0.0, 1.0)
        };
        // Helper: above-threshold score
        let above = |v: f32, thr: f32, tol: f32| -> f32 { ((v - thr + tol) / tol).clamp(0.0, 1.0) };
        // Helper: below-threshold score
        let below = |v: f32, thr: f32, tol: f32| -> f32 { ((thr + tol - v) / tol).clamp(0.0, 1.0) };

        match sg {
            // ── Electronic ────────────────────────────────────────────────
            SubGenre::House => {
                near(bpm, 124.0, 8.0) * 0.40
                    + above(energy, 0.5, 0.2) * 0.30
                    + near(sc, 0.25, 0.08) * 0.30
            }
            SubGenre::Techno => {
                near(bpm, 138.0, 10.0) * 0.45
                    + above(sc, 0.25, 0.08) * 0.25
                    + below(energy_var, 0.3, 0.15) * 0.30
            }
            SubGenre::DrumAndBass => {
                near(bpm, 170.0, 15.0) * 0.50
                    + above(sc, 0.30, 0.10) * 0.25
                    + above(zcr, 0.08, 0.04) * 0.25
            }
            SubGenre::AmbientTechno => {
                below(bpm, 100.0, 40.0) * 0.30
                    + below(energy, 0.4, 0.2) * 0.35
                    + below(sc, 0.20, 0.08) * 0.35
            }
            SubGenre::Synthwave => {
                near(bpm, 95.0, 15.0) * 0.35
                    + near(sc, 0.20, 0.08) * 0.30
                    + below(zcr, 0.08, 0.04) * 0.35
            }

            // ── Rock ──────────────────────────────────────────────────────
            SubGenre::ClassicRock => {
                near(bpm, 130.0, 20.0) * 0.35
                    + near(sc, 0.22, 0.08) * 0.30
                    + above(energy, 0.4, 0.2) * 0.35
            }
            SubGenre::IndieRock => {
                near(bpm, 125.0, 20.0) * 0.30
                    + near(sc, 0.25, 0.08) * 0.35
                    + above(energy_var, 0.2, 0.1) * 0.35
            }
            SubGenre::Punk => {
                above(bpm, 160.0, 30.0) * 0.45
                    + above(zcr, 0.12, 0.05) * 0.35
                    + above(energy, 0.6, 0.2) * 0.20
            }
            SubGenre::ProgRock => {
                near(bpm, 110.0, 30.0) * 0.25
                    + above(energy_var, 0.3, 0.1) * 0.40
                    + above(sr, 0.40, 0.10) * 0.35
            }
            SubGenre::Grunge => {
                near(bpm, 110.0, 20.0) * 0.30
                    + above(zcr, 0.10, 0.04) * 0.35
                    + above(energy, 0.5, 0.2) * 0.35
            }

            // ── Pop ───────────────────────────────────────────────────────
            SubGenre::DancePop => {
                near(bpm, 122.0, 8.0) * 0.40
                    + above(energy, 0.55, 0.2) * 0.30
                    + above(sc, 0.25, 0.08) * 0.30
            }
            SubGenre::SynthPop => {
                near(bpm, 112.0, 12.0) * 0.30
                    + above(sc, 0.25, 0.08) * 0.35
                    + below(zcr, 0.10, 0.04) * 0.35
            }
            SubGenre::IndiePop => {
                near(bpm, 115.0, 15.0) * 0.25
                    + near(energy, 0.45, 0.15) * 0.35
                    + above(energy_var, 0.15, 0.08) * 0.40
            }
            SubGenre::KPop => {
                near(bpm, 120.0, 12.0) * 0.30
                    + above(sc, 0.30, 0.08) * 0.35
                    + above(energy, 0.60, 0.2) * 0.35
            }
            SubGenre::AcousticPop => {
                below(energy, 0.45, 0.20) * 0.35
                    + below(sc, 0.22, 0.08) * 0.30
                    + below(zcr, 0.08, 0.04) * 0.35
            }

            // ── Classical ─────────────────────────────────────────────────
            SubGenre::Baroque => {
                near(energy_var, 0.3, 0.1) * 0.40
                    + below(sc, 0.20, 0.08) * 0.35
                    + below(energy, 0.50, 0.2) * 0.25
            }
            SubGenre::Romantic => {
                above(energy_var, 0.35, 0.1) * 0.40
                    + above(sr, 0.30, 0.10) * 0.30
                    + near(energy, 0.5, 0.2) * 0.30
            }
            SubGenre::Contemporary => {
                above(zcr, 0.06, 0.03) * 0.30
                    + above(energy_var, 0.4, 0.1) * 0.40
                    + above(sr, 0.35, 0.10) * 0.30
            }
            SubGenre::Minimalist => {
                below(energy_var, 0.2, 0.1) * 0.45
                    + below(sc, 0.18, 0.07) * 0.30
                    + below(zcr, 0.06, 0.03) * 0.25
            }
            SubGenre::Chamber => {
                below(energy, 0.40, 0.2) * 0.35
                    + near(sc, 0.18, 0.08) * 0.35
                    + below(zcr, 0.07, 0.03) * 0.30
            }

            // ── Jazz ──────────────────────────────────────────────────────
            SubGenre::Bebop => {
                above(bpm, 150.0, 30.0) * 0.40
                    + above(energy_var, 0.3, 0.1) * 0.35
                    + above(zcr, 0.06, 0.03) * 0.25
            }
            SubGenre::SmoothJazz => {
                below(bpm, 110.0, 30.0) * 0.35
                    + below(energy, 0.45, 0.2) * 0.30
                    + near(sc, 0.18, 0.07) * 0.35
            }
            SubGenre::JazzFusion => {
                above(energy, 0.5, 0.2) * 0.30
                    + above(zcr, 0.07, 0.03) * 0.35
                    + near(bpm, 130.0, 25.0) * 0.35
            }
            SubGenre::FreeJazz => {
                above(energy_var, 0.45, 0.1) * 0.45
                    + above(zcr, 0.08, 0.03) * 0.30
                    + above(sr, 0.35, 0.10) * 0.25
            }
            SubGenre::Swing => {
                near(bpm, 140.0, 20.0) * 0.40
                    + near(energy_var, 0.2, 0.08) * 0.30
                    + near(sc, 0.20, 0.07) * 0.30
            }

            // ── Hip-Hop ───────────────────────────────────────────────────
            SubGenre::Trap => {
                near(bpm, 140.0, 15.0) * 0.40
                    + above(energy, 0.55, 0.2) * 0.30
                    + below(sc, 0.20, 0.08) * 0.30
            }
            SubGenre::BoomBap => {
                near(bpm, 92.0, 10.0) * 0.40
                    + above(sc, 0.15, 0.07) * 0.30
                    + near(energy, 0.5, 0.2) * 0.30
            }
            SubGenre::LoFiHipHop => {
                near(bpm, 80.0, 15.0) * 0.35
                    + below(energy, 0.45, 0.2) * 0.35
                    + below(sc, 0.18, 0.07) * 0.30
            }
            SubGenre::OldSchoolHipHop => {
                near(bpm, 95.0, 12.0) * 0.40
                    + near(sc, 0.20, 0.07) * 0.30
                    + near(energy, 0.5, 0.2) * 0.30
            }
            SubGenre::CloudRap => {
                near(bpm, 130.0, 15.0) * 0.30
                    + below(energy, 0.50, 0.2) * 0.35
                    + near(sc, 0.18, 0.07) * 0.35
            }

            // ── Metal ─────────────────────────────────────────────────────
            SubGenre::HeavyMetal => {
                near(bpm, 130.0, 20.0) * 0.35
                    + above(zcr, 0.12, 0.04) * 0.35
                    + above(energy, 0.6, 0.2) * 0.30
            }
            SubGenre::DeathMetal => {
                above(bpm, 170.0, 30.0) * 0.40
                    + above(zcr, 0.15, 0.04) * 0.35
                    + above(energy, 0.65, 0.2) * 0.25
            }
            SubGenre::BlackMetal => {
                above(bpm, 160.0, 30.0) * 0.35
                    + above(zcr, 0.14, 0.04) * 0.35
                    + above(sr, 0.45, 0.10) * 0.30
            }
            SubGenre::DoomMetal => {
                below(bpm, 80.0, 20.0) * 0.45
                    + above(energy, 0.55, 0.2) * 0.30
                    + below(zcr, 0.12, 0.05) * 0.25
            }
            SubGenre::NuMetal => {
                near(bpm, 105.0, 15.0) * 0.35
                    + above(zcr, 0.10, 0.04) * 0.30
                    + near(sc, 0.25, 0.08) * 0.35
            }

            // ── Folk ──────────────────────────────────────────────────────
            SubGenre::SingerSongwriter => {
                below(energy, 0.35, 0.15) * 0.40
                    + below(sc, 0.15, 0.06) * 0.30
                    + below(zcr, 0.06, 0.03) * 0.30
            }
            SubGenre::Celtic => {
                above(energy_var, 0.25, 0.1) * 0.35
                    + near(bpm, 120.0, 25.0) * 0.35
                    + near(sc, 0.18, 0.07) * 0.30
            }
            SubGenre::Bluegrass => {
                above(bpm, 130.0, 20.0) * 0.40
                    + above(zcr, 0.07, 0.03) * 0.30
                    + near(sc, 0.20, 0.07) * 0.30
            }
            SubGenre::IndieFolk => {
                near(energy, 0.40, 0.15) * 0.35
                    + near(sc, 0.18, 0.07) * 0.35
                    + above(energy_var, 0.2, 0.08) * 0.30
            }
            SubGenre::Americana => {
                near(bpm, 105.0, 20.0) * 0.35
                    + near(sc, 0.17, 0.07) * 0.30
                    + near(energy, 0.45, 0.15) * 0.35
            }

            SubGenre::Unknown => 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subgenre_names_nonempty() {
        let all = [
            SubGenre::House,
            SubGenre::Techno,
            SubGenre::Trap,
            SubGenre::Bebop,
            SubGenre::ClassicRock,
            SubGenre::DancePop,
            SubGenre::Baroque,
            SubGenre::HeavyMetal,
            SubGenre::Celtic,
            SubGenre::Unknown,
        ];
        for sg in &all {
            assert!(!sg.name().is_empty(), "{sg:?} has empty name");
        }
    }

    #[test]
    fn test_classify_electronic_house() {
        // 124 BPM, moderate centroid, high energy → House
        let result =
            SubGenreClassifier::classify("electronic", 0.25, 0.35, 0.06, 124.0, 0.65, 0.15);
        assert!(!result.scores.is_empty());
        assert!(result.is_confident());
        assert_eq!(result.top_subgenre, SubGenre::House);
    }

    #[test]
    fn test_classify_electronic_dnb() {
        // 170 BPM, bright centroid, high ZCR → Drum & Bass
        let result = SubGenreClassifier::classify("electronic", 0.35, 0.45, 0.12, 170.0, 0.7, 0.2);
        assert_eq!(result.top_subgenre, SubGenre::DrumAndBass);
    }

    #[test]
    fn test_classify_rock_punk() {
        // Very fast BPM, high ZCR, high energy → Punk
        let result = SubGenreClassifier::classify("rock", 0.28, 0.40, 0.18, 190.0, 0.75, 0.2);
        assert_eq!(result.top_subgenre, SubGenre::Punk);
    }

    #[test]
    fn test_classify_metal_doom() {
        // Very slow BPM, high energy, low ZCR → Doom Metal
        let result = SubGenreClassifier::classify("metal", 0.20, 0.30, 0.09, 60.0, 0.70, 0.15);
        assert_eq!(result.top_subgenre, SubGenre::DoomMetal);
    }

    #[test]
    fn test_classify_unknown_genre_returns_empty() {
        let result = SubGenreClassifier::classify("reggae", 0.20, 0.30, 0.07, 100.0, 0.5, 0.2);
        assert!(result.scores.is_empty());
        assert_eq!(result.top_subgenre, SubGenre::Unknown);
        assert!(!result.is_confident());
    }

    #[test]
    fn test_scores_sorted_descending() {
        let result = SubGenreClassifier::classify("jazz", 0.20, 0.30, 0.08, 120.0, 0.45, 0.25);
        for w in result.scores.windows(2) {
            assert!(
                w[0].1 >= w[1].1,
                "Scores not sorted descending: {:.4} < {:.4}",
                w[0].1,
                w[1].1
            );
        }
    }

    #[test]
    fn test_confidence_in_range() {
        let result = SubGenreClassifier::classify("pop", 0.28, 0.35, 0.08, 120.0, 0.6, 0.2);
        assert!(
            (0.0..=1.0).contains(&result.confidence),
            "Confidence out of range: {}",
            result.confidence
        );
    }

    #[test]
    fn test_hip_hop_trap() {
        // 140 BPM, high energy, dark spectrum → Trap
        let result = SubGenreClassifier::classify("hip-hop", 0.18, 0.25, 0.05, 140.0, 0.65, 0.1);
        assert_eq!(result.top_subgenre, SubGenre::Trap);
    }

    #[test]
    fn test_folk_singer_songwriter() {
        // Low energy, low centroid, low ZCR → Singer-Songwriter
        let result = SubGenreClassifier::classify("folk", 0.12, 0.20, 0.04, 80.0, 0.25, 0.1);
        assert_eq!(result.top_subgenre, SubGenre::SingerSongwriter);
    }
}
