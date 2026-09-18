//! Gaussian Naïve Bayes genre classifier operating on pre-extracted audio features.
//!
//! This module provides a **pure-Rust, patent-free** genre classifier that works
//! on a compact [`AudioFeatures`] vector rather than raw audio.  The classifier
//! uses class-conditional Gaussian distributions (Naïve Bayes assumption) with
//! equal priors over 10 genre classes.
//!
//! # Usage
//!
//! ```rust
//! use oximedia_mir::genre_classifier::{AudioFeatures, Genre, GenreClassifier};
//!
//! let features = AudioFeatures {
//!     mfcc_mean: vec![0.0; 13],
//!     mfcc_var:  vec![1.0; 13],
//!     spectral_centroid:   3000.0,
//!     zero_crossing_rate:  0.15,
//!     tempo:               140.0,
//!     spectral_rolloff:    6000.0,
//!     spectral_flatness:   0.1,
//!     energy:              0.6,
//! };
//!
//! let classifier = GenreClassifier::new();
//! let predictions = classifier.classify(&features);
//! // predictions[0] has the highest confidence
//! println!("{:?}: {:.3}", predictions[0].0, predictions[0].1);
//! ```

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::suboptimal_flops)]

use std::f32::consts::PI;

// ── Genre enum ────────────────────────────────────────────────────────────────

/// Music genre label (10 canonical categories).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Genre {
    /// Rock / Alternative rock.
    Rock,
    /// Pop (mainstream chart music).
    Pop,
    /// Jazz (bebop, fusion, smooth jazz).
    Jazz,
    /// Classical / orchestral.
    Classical,
    /// Electronic / EDM / dance music.
    Electronic,
    /// Hip-hop / rap.
    HipHop,
    /// Country / Americana.
    Country,
    /// R&B / soul.
    RnB,
    /// Folk / acoustic.
    Folk,
    /// Metal / heavy metal.
    Metal,
}

impl Genre {
    /// Return a human-readable name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Rock => "Rock",
            Self::Pop => "Pop",
            Self::Jazz => "Jazz",
            Self::Classical => "Classical",
            Self::Electronic => "Electronic",
            Self::HipHop => "Hip-Hop",
            Self::Country => "Country",
            Self::RnB => "R&B",
            Self::Folk => "Folk",
            Self::Metal => "Metal",
        }
    }

    /// All 10 genres in a fixed order (used internally by the classifier).
    fn all() -> [Self; 10] {
        [
            Self::Rock,
            Self::Pop,
            Self::Jazz,
            Self::Classical,
            Self::Electronic,
            Self::HipHop,
            Self::Country,
            Self::RnB,
            Self::Folk,
            Self::Metal,
        ]
    }
}

// ── AudioFeatures ─────────────────────────────────────────────────────────────

/// Compact feature vector consumed by [`GenreClassifier`].
#[derive(Debug, Clone)]
pub struct AudioFeatures {
    /// Per-coefficient mean of MFCC frames (typically 13 coefficients).
    pub mfcc_mean: Vec<f32>,
    /// Per-coefficient variance of MFCC frames.
    pub mfcc_var: Vec<f32>,
    /// Spectral centroid in Hz.
    pub spectral_centroid: f32,
    /// Zero-crossing rate (crossings per sample, in \[0, 1\]).
    pub zero_crossing_rate: f32,
    /// Estimated tempo in BPM.
    pub tempo: f32,
    /// Spectral rolloff (Hz below which 85 % of energy resides).
    pub spectral_rolloff: f32,
    /// Spectral flatness (Wiener entropy, in \[0, 1\]).
    pub spectral_flatness: f32,
    /// RMS energy of the signal (in \[0, 1\]).
    pub energy: f32,
}

impl Default for AudioFeatures {
    fn default() -> Self {
        Self {
            mfcc_mean: vec![0.0; 13],
            mfcc_var: vec![1.0; 13],
            spectral_centroid: 2000.0,
            zero_crossing_rate: 0.1,
            tempo: 120.0,
            spectral_rolloff: 4000.0,
            spectral_flatness: 0.3,
            energy: 0.5,
        }
    }
}

// ── Gaussian NB class prototypes ──────────────────────────────────────────────
//
// Each genre is described by 8 scalar features:
//   [spectral_centroid, zcr, tempo, energy, mfcc0_mean, mfcc1_mean, mfcc2_mean, spectral_flatness]
//
// Values are normalised to a rough [0,1] scale (centroid /10000, rolloff /20000, etc.)
// so that the Gaussian distances are comparable.
//
// These are hand-calibrated prototype means and standard deviations derived from
// informal analysis of standard genre datasets (GTZAN-like statistics).

/// Number of scalar features used by the classifier.
const N_FEATS: usize = 8;
/// Number of genre classes.
const N_GENRES: usize = 10;

/// Class-conditional means: shape [N_GENRES][N_FEATS].
///
/// Feature order: [centroid_norm, zcr, tempo_norm, energy, mfcc0, mfcc1, mfcc2, flatness]
/// where centroid_norm = centroid_hz / 10000, tempo_norm = tempo_bpm / 200.
static CLASS_MEANS: [[f32; N_FEATS]; N_GENRES] = [
    // Rock:      mid centroid, mod zcr, fast tempo, high energy, warm mfcc
    [0.25, 0.12, 0.70, 0.75, -5.0, 2.0, 1.0, 0.15],
    // Pop:       mid centroid, mod zcr, mod tempo, mod energy
    [0.22, 0.10, 0.60, 0.55, -3.0, 1.5, 0.5, 0.20],
    // Jazz:      low centroid, low zcr, slow tempo, low energy, complex mfcc
    [0.18, 0.06, 0.45, 0.40, -2.0, 3.0, 2.0, 0.35],
    // Classical: low centroid, very low zcr, varied tempo, low energy
    [0.15, 0.04, 0.40, 0.30, -1.0, 2.5, 1.5, 0.40],
    // Electronic: high centroid, high zcr, fast tempo, high energy, flat spectrum
    [0.55, 0.20, 0.80, 0.80, -8.0, 1.0, -0.5, 0.60],
    // HipHop:    mid centroid, mod zcr, slow-mod tempo, high energy
    [0.20, 0.15, 0.50, 0.70, -6.0, 0.5, 0.0, 0.25],
    // Country:   low centroid, low zcr, mid tempo, mid energy
    [0.17, 0.07, 0.55, 0.45, -2.5, 2.0, 1.2, 0.18],
    // RnB:       mid centroid, low zcr, mod tempo, mod-high energy
    [0.21, 0.08, 0.52, 0.60, -4.0, 1.8, 0.8, 0.22],
    // Folk:      low centroid, very low zcr, slow tempo, low energy, tonal
    [0.14, 0.05, 0.38, 0.28, -1.5, 2.2, 1.8, 0.12],
    // Metal:     high centroid, very high zcr, very fast tempo, max energy
    [0.38, 0.25, 0.90, 0.95, -10.0, 0.5, -1.0, 0.50],
];

/// Class-conditional standard deviations: shape [N_GENRES][N_FEATS].
static CLASS_STDS: [[f32; N_FEATS]; N_GENRES] = [
    // Rock
    [0.08, 0.05, 0.12, 0.12, 3.0, 2.0, 2.0, 0.08],
    // Pop
    [0.07, 0.04, 0.10, 0.12, 3.0, 2.0, 2.0, 0.10],
    // Jazz
    [0.07, 0.03, 0.12, 0.10, 2.5, 2.5, 2.5, 0.12],
    // Classical
    [0.08, 0.02, 0.15, 0.10, 2.5, 2.5, 2.5, 0.15],
    // Electronic
    [0.12, 0.06, 0.10, 0.10, 4.0, 2.0, 2.0, 0.15],
    // HipHop
    [0.07, 0.05, 0.10, 0.12, 3.0, 2.0, 2.0, 0.10],
    // Country
    [0.07, 0.04, 0.12, 0.12, 2.5, 2.0, 2.0, 0.08],
    // RnB
    [0.07, 0.04, 0.10, 0.12, 3.0, 2.0, 2.0, 0.10],
    // Folk
    [0.06, 0.03, 0.12, 0.10, 2.5, 2.0, 2.0, 0.06],
    // Metal
    [0.10, 0.06, 0.08, 0.06, 4.0, 2.0, 2.0, 0.12],
];

// ── GenreClassifier ───────────────────────────────────────────────────────────

/// Gaussian Naïve Bayes genre classifier.
///
/// The classifier converts an [`AudioFeatures`] vector into a feature slice of
/// length `N_FEATS`, computes the log-likelihood under each class-conditional
/// Gaussian model, applies uniform priors, and softmax-normalises the
/// log-posteriors into confidence scores in \[0, 1\].
pub struct GenreClassifier {
    // No mutable state; all parameters are compile-time constants.
    _priv: (),
}

impl GenreClassifier {
    /// Construct a new classifier.
    #[must_use]
    pub fn new() -> Self {
        Self { _priv: () }
    }

    /// Classify audio features into genre confidence scores.
    ///
    /// Returns a `Vec` of `(Genre, confidence)` pairs sorted by confidence
    /// in descending order.  The sum of all confidences is approximately 1.0.
    #[must_use]
    pub fn classify(&self, features: &AudioFeatures) -> Vec<(Genre, f32)> {
        let fv = build_feature_vec(features);
        let log_posts = compute_log_posteriors(&fv);
        let confidences = softmax(&log_posts);

        let mut result: Vec<(Genre, f32)> = Genre::all()
            .iter()
            .zip(confidences.iter())
            .map(|(&g, &c)| (g, c))
            .collect();

        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }
}

impl Default for GenreClassifier {
    fn default() -> Self {
        Self::new()
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Build the 8-element normalised feature vector from an [`AudioFeatures`].
fn build_feature_vec(f: &AudioFeatures) -> [f32; N_FEATS] {
    // Normalise to the approximate scales used in CLASS_MEANS
    let centroid_norm = (f.spectral_centroid / 10_000.0).clamp(0.0, 1.0);
    let zcr = f.zero_crossing_rate.clamp(0.0, 1.0);
    let tempo_norm = (f.tempo / 200.0).clamp(0.0, 1.5);
    let energy = f.energy.clamp(0.0, 1.0);

    // MFCC coefficients 0, 1, 2 (use 0 if not available)
    let mfcc0 = f.mfcc_mean.first().copied().unwrap_or(0.0);
    let mfcc1 = f.mfcc_mean.get(1).copied().unwrap_or(0.0);
    let mfcc2 = f.mfcc_mean.get(2).copied().unwrap_or(0.0);

    let flatness = f.spectral_flatness.clamp(0.0, 1.0);

    [
        centroid_norm,
        zcr,
        tempo_norm,
        energy,
        mfcc0,
        mfcc1,
        mfcc2,
        flatness,
    ]
}

/// Gaussian log-pdf: −½ ln(2π σ²) − (x−μ)²/(2σ²).
#[inline]
fn log_gaussian(x: f32, mean: f32, std: f32) -> f32 {
    let std = std.max(1e-6);
    let diff = x - mean;
    -0.5 * (2.0 * PI).ln() - std.ln() - 0.5 * (diff / std).powi(2)
}

/// Compute log-posterior (= log-likelihood under equal priors) for each genre.
fn compute_log_posteriors(fv: &[f32; N_FEATS]) -> [f32; N_GENRES] {
    let mut log_posts = [0.0_f32; N_GENRES];
    for (g, lp) in log_posts.iter_mut().enumerate() {
        for f in 0..N_FEATS {
            *lp += log_gaussian(fv[f], CLASS_MEANS[g][f], CLASS_STDS[g][f]);
        }
        // Equal prior: log(1/10) — cancels out in softmax but keep for correctness
        *lp += -(N_GENRES as f32).ln();
    }
    log_posts
}

/// Numerically-stable softmax.
fn softmax(logits: &[f32; N_GENRES]) -> [f32; N_GENRES] {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: [f32; N_GENRES] = [0.0; N_GENRES];
    let mut sum = 0.0_f32;
    for (i, &l) in logits.iter().enumerate() {
        exps[i] = (l - max).exp();
        sum += exps[i];
    }
    if sum < 1e-30 {
        // Degenerate: return uniform distribution
        return [1.0 / N_GENRES as f32; N_GENRES];
    }
    let mut out = [0.0_f32; N_GENRES];
    for (i, e) in exps.iter().enumerate() {
        out[i] = e / sum;
    }
    out
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn metal_features() -> AudioFeatures {
        AudioFeatures {
            mfcc_mean: vec![
                -10.0, 0.5, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            mfcc_var: vec![1.0; 13],
            spectral_centroid: 3800.0, // high centroid
            zero_crossing_rate: 0.25,  // very high zcr
            tempo: 180.0,              // very fast
            spectral_rolloff: 9000.0,
            spectral_flatness: 0.50,
            energy: 0.95, // max energy
        }
    }

    fn classical_features() -> AudioFeatures {
        AudioFeatures {
            mfcc_mean: vec![
                -1.0, 2.5, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            mfcc_var: vec![0.5; 13],
            spectral_centroid: 1500.0, // low centroid
            zero_crossing_rate: 0.04,  // very low zcr
            tempo: 80.0,               // slow
            spectral_rolloff: 3000.0,
            spectral_flatness: 0.40,
            energy: 0.30,
        }
    }

    fn electronic_features() -> AudioFeatures {
        AudioFeatures {
            mfcc_mean: vec![
                -8.0, 1.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            mfcc_var: vec![1.5; 13],
            spectral_centroid: 5500.0, // very high
            zero_crossing_rate: 0.20,
            tempo: 160.0,
            spectral_rolloff: 12000.0,
            spectral_flatness: 0.60, // flat spectrum
            energy: 0.80,
        }
    }

    fn rock_features() -> AudioFeatures {
        AudioFeatures {
            mfcc_mean: vec![
                -5.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            mfcc_var: vec![1.0; 13],
            spectral_centroid: 2500.0,
            zero_crossing_rate: 0.12,
            tempo: 140.0,
            spectral_rolloff: 5000.0,
            spectral_flatness: 0.15,
            energy: 0.75,
        }
    }

    // ── Basic structural tests ────────────────────────────────────────────────

    #[test]
    fn test_classify_returns_ten_entries() {
        let clf = GenreClassifier::new();
        let result = clf.classify(&AudioFeatures::default());
        assert_eq!(result.len(), 10, "should return one entry per genre");
    }

    #[test]
    fn test_classify_sorted_descending() {
        let clf = GenreClassifier::new();
        let result = clf.classify(&AudioFeatures::default());
        for i in 1..result.len() {
            assert!(
                result[i - 1].1 >= result[i].1,
                "not sorted at index {i}: {:.4} < {:.4}",
                result[i - 1].1,
                result[i].1
            );
        }
    }

    #[test]
    fn test_confidence_sum_approximately_one() {
        let clf = GenreClassifier::new();
        let result = clf.classify(&AudioFeatures::default());
        let sum: f32 = result.iter().map(|(_, c)| c).sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "confidence sum should ≈1, got {sum:.6}"
        );
    }

    #[test]
    fn test_each_confidence_in_zero_one() {
        let clf = GenreClassifier::new();
        let result = clf.classify(&AudioFeatures::default());
        for (g, c) in &result {
            assert!(
                *c >= 0.0 && *c <= 1.0,
                "confidence for {g:?} out of range: {c:.4}"
            );
        }
    }

    // ── Genre discrimination ──────────────────────────────────────────────────

    #[test]
    fn test_metal_features_top_genre_is_metal() {
        let clf = GenreClassifier::new();
        let result = clf.classify(&metal_features());
        assert_eq!(
            result[0].0,
            Genre::Metal,
            "metal features → top genre Metal"
        );
    }

    #[test]
    fn test_classical_features_top_genre_is_classical() {
        let clf = GenreClassifier::new();
        let result = clf.classify(&classical_features());
        assert_eq!(
            result[0].0,
            Genre::Classical,
            "classical features → top genre Classical"
        );
    }

    #[test]
    fn test_electronic_features_top_genre_is_electronic() {
        let clf = GenreClassifier::new();
        let result = clf.classify(&electronic_features());
        assert_eq!(
            result[0].0,
            Genre::Electronic,
            "electronic features → top genre Electronic"
        );
    }

    #[test]
    fn test_rock_features_top_genre_in_expected_set() {
        let clf = GenreClassifier::new();
        let result = clf.classify(&rock_features());
        // Rock or Metal are both reasonable for high-energy mid-centroid features
        let top = result[0].0;
        assert!(
            matches!(top, Genre::Rock | Genre::Metal | Genre::Pop),
            "rock features → top genre should be Rock/Metal/Pop, got {top:?}"
        );
    }

    // ── Edge cases ────────────────────────────────────────────────────────────

    #[test]
    fn test_empty_mfcc_still_classifies() {
        let clf = GenreClassifier::new();
        let features = AudioFeatures {
            mfcc_mean: vec![],
            mfcc_var: vec![],
            ..AudioFeatures::default()
        };
        let result = clf.classify(&features);
        assert_eq!(result.len(), 10);
    }

    #[test]
    fn test_zero_energy_classifies() {
        let clf = GenreClassifier::new();
        let features = AudioFeatures {
            energy: 0.0,
            ..AudioFeatures::default()
        };
        let result = clf.classify(&features);
        assert_eq!(result.len(), 10);
    }

    #[test]
    fn test_extreme_centroid_classifies() {
        let clf = GenreClassifier::new();
        let features = AudioFeatures {
            spectral_centroid: 20_000.0,
            ..AudioFeatures::default()
        };
        let result = clf.classify(&features);
        assert_eq!(result.len(), 10);
    }

    #[test]
    fn test_all_genres_appear_exactly_once() {
        let clf = GenreClassifier::new();
        let result = clf.classify(&AudioFeatures::default());
        for expected in Genre::all() {
            let count = result.iter().filter(|(g, _)| *g == expected).count();
            assert_eq!(count, 1, "genre {expected:?} should appear exactly once");
        }
    }

    #[test]
    fn test_default_classifier_equivalent_to_new() {
        let clf1 = GenreClassifier::new();
        let clf2 = GenreClassifier::default();
        let r1 = clf1.classify(&AudioFeatures::default());
        let r2 = clf2.classify(&AudioFeatures::default());
        for ((g1, c1), (g2, c2)) in r1.iter().zip(r2.iter()) {
            assert_eq!(g1, g2);
            assert!((c1 - c2).abs() < 1e-6);
        }
    }
}
