//! Music Information Retrieval (MIR) system for `OxiMedia`.
//!
//! This crate provides comprehensive music analysis capabilities for audio content:
//!
//! # Tempo and Beat Analysis
//!
//! - **Tempo Detection** - BPM detection using autocorrelation and comb filtering
//! - **Beat Tracking** - Beat and downbeat detection with dynamic programming
//! - **Onset Detection** - Transient detection using spectral flux and HFC
//!
//! # Tonal Analysis
//!
//! - **Key Detection** - Musical key detection (Krumhansl-Schmuckler algorithm)
//! - **Chord Recognition** - Chord progression analysis using chroma features
//! - **Melody Extraction** - Dominant melody line extraction
//! - **Harmonic Analysis** - Harmonic-percussive separation
//!
//! # Structure Analysis
//!
//! - **Structural Segmentation** - Section boundary detection
//! - **Self-Similarity Analysis** - Pattern and repetition detection
//! - **Section Labeling** - Intro, verse, chorus, bridge identification
//!
//! # High-Level Features
//!
//! - **Genre Classification** - Genre detection from audio features
//! - **Mood Detection** - Valence and arousal estimation
//! - **Loudness Analysis** - Integrated loudness and dynamics
//!
//! # Low-Level Features
//!
//! - **Spectral Features** - Centroid, rolloff, flux, contrast
//! - **Rhythm Features** - Rhythm patterns and complexity
//! - **Pitch Features** - Pitch class profiles and chromagrams
//!
//! # Usage
//!
//! ```no_run
//! use oximedia_mir::{MirAnalyzer, MirConfig, FeatureSet};
//!
//! // Create analyzer with default configuration
//! let config = MirConfig::default();
//! let analyzer = MirAnalyzer::new(config);
//!
//! // Analyze audio samples (f32, mono or stereo)
//! let samples = vec![0.0_f32; 44100]; // 1 second of silence
//! let sample_rate = 44100.0;
//!
//! // Perform analysis
//! let result = analyzer.analyze(&samples, sample_rate)?;
//!
//! // Access results
//! if let Some(ref tempo) = result.tempo {
//!     println!("Tempo: {:.1} BPM (confidence: {:.2})", tempo.bpm, tempo.confidence);
//! }
//! if let Some(ref key) = result.key {
//!     println!("Key: {} (confidence: {:.2})", key.key, key.confidence);
//! }
//! if let Some(ref genre) = result.genre {
//!     println!("Genre: {} (confidence: {:.2})", genre.top_genre().0, genre.top_genre().1);
//! }
//!
//! # Ok::<(), oximedia_mir::MirError>(())
//! ```
//!
//! # Patent-Free Implementation
//!
//! All algorithms are implemented using patent-free methods:
//! - Autocorrelation-based tempo detection
//! - Chroma-based chord recognition
//! - Spectral-based onset detection
//! - Krumhansl-Schmuckler key detection
//!
//! # Real-Time Capable
//!
//! Many features support frame-by-frame processing for real-time applications.

#![warn(missing_docs)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::similar_names)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::let_and_return)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::unused_self)]
#![allow(clippy::module_name_repetitions)]
#![allow(dead_code)]

pub mod audio_events;
pub mod audio_features;
pub mod beat;
pub mod beat_tracker;
pub mod beat_tracking;
pub mod chord;
pub mod chord_recognition;
pub mod chorus_detect;
pub mod cover_detect;
pub mod dynamic_range;
pub mod energy_contour;
pub mod fade_detect;
pub mod fingerprint;
pub mod genre;
pub mod genre_classifier;
pub mod genre_classify;
pub mod harmonic;
pub mod harmonic_analysis;
pub mod instrument;
pub mod instrument_detection;
pub mod key;
pub mod key_detection;
pub mod loudness;
pub mod melody;
pub mod midi;
pub mod mir_feature;
pub mod mood;
pub mod mood_detection;
pub mod music_summary;
pub mod onset_strength;
pub mod pitch_key;
pub mod pitch_track;
pub mod playlist;
pub mod playlist_gen;
pub mod rhythm;
pub mod rhythm_pattern;
pub mod segmentation;
pub mod similarity;
pub mod source_separation;
pub mod spectral;
pub mod spectral_contrast;
pub mod spectral_features;
pub mod streaming;
pub mod structure;
pub mod structure_analysis;
pub mod tempo;
pub mod tempo_map;
pub mod tuning_detect;
pub mod vocal_detect;

/// 12-bin chroma vector computation (pitch-class profiling, chord/key analysis).
pub mod chromagram;

/// Cached chromagram that amortises FFT cost across multiple consumers.
pub mod chroma_cache;

/// Audio watermark embedding (spread-spectrum LSB steganography).
pub mod watermark;

/// Audio watermark detection and extraction.
pub mod watermark_detect;

/// Cosine, Earth-mover, and multi-modal audio similarity measures.
pub mod audio_similarity;

/// Visual cover-art feature extraction (dominant colours, edge histograms).
pub mod cover_art_features;

/// DJ workflow features: Camelot wheel, beat-matching, compatibility scoring.
pub mod dj_features;

/// Neural-network-free rule-based genre classification (spectral features).
pub mod genre_classify_new;

/// Harmonic-percussive source separation (median-filter HPSS).
pub mod harmonic_percussive;

/// Inharmonicity, HNR, and THD from the harmonic series.
pub mod harmonic_spectral;

/// Instrument family and voice classification.
pub mod instrument_classifier;

/// Locality-sensitive hashing for fast approximate audio nearest-neighbour.
pub mod lsh_similarity;

/// Forced alignment of lyrics text to detected onset timestamps.
pub mod lyrics_align;

/// Energy/zero-crossing melody extraction and contour shape analysis.
pub mod melody_extract;

/// Harmonic-salience Viterbi melody extractor with vibrato detection.
pub mod melody_extractor;

/// Multi-stem analysis with per-stem feature extraction.
pub mod multitrack;

/// Onset peak picking with configurable threshold and minimum interval.
pub mod onset_peak;

/// Structural section segmentation via novelty-curve self-similarity.
pub mod section_segmenter;

/// Brute-force and LSH similarity search over fingerprint indices.
pub mod similarity_search;

/// Sub-genre tagging within broad genre families.
pub mod subgenre;

/// Tempo stability analysis (variance, drift, class).
pub mod tempo_stability;

/// Waveform thumbnail generation for audio browsers.
pub mod thumbnail;

#[cfg(feature = "onnx")]
pub mod ml;

mod error;
mod types;
mod utils;

pub use error::{MirError, MirResult};

pub use midi::{AudioToMidi, AudioToMidiConfig, MidiNote, MidiTempo, MidiTranscription};
#[cfg(feature = "onnx")]
pub use ml::{
    activate_and_rank, apply_activation, MusicTagger, MusicTags, TagActivation, TagActivationScore,
    DEFAULT_TOP_K,
};
pub use streaming::{
    StreamingAnalysisSummary, StreamingAnalyzer, StreamingConfig, StreamingFrameFeatures,
};
pub use types::{
    AnalysisResult, BeatResult, ChordResult, FeatureSet, GenreResult, HarmonicResult, KeyResult,
    LoudnessResult, MelodyResult, MoodResult, RhythmResult, SpectralResult, StructureResult,
    TempoResult,
};

use rayon::prelude::*;
use std::collections::HashMap;

/// Configuration for MIR analysis.
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct MirConfig {
    /// Window size for frame-based analysis (samples).
    pub window_size: usize,

    /// Hop size for frame-based analysis (samples).
    pub hop_size: usize,

    /// Minimum tempo to detect (BPM).
    pub min_tempo: f32,

    /// Maximum tempo to detect (BPM).
    pub max_tempo: f32,

    /// Enable beat tracking.
    pub enable_beat_tracking: bool,

    /// Enable key detection.
    pub enable_key_detection: bool,

    /// Enable chord recognition.
    pub enable_chord_recognition: bool,

    /// Enable melody extraction.
    pub enable_melody_extraction: bool,

    /// Enable structure analysis.
    pub enable_structure_analysis: bool,

    /// Enable genre classification.
    pub enable_genre_classification: bool,

    /// Enable mood detection.
    pub enable_mood_detection: bool,

    /// Enable spectral features.
    pub enable_spectral_features: bool,

    /// Enable rhythm features.
    pub enable_rhythm_features: bool,

    /// Enable harmonic analysis.
    pub enable_harmonic_analysis: bool,

    /// Confidence threshold for tempo detection (0.0 to 1.0).
    /// Results below this threshold are discarded (set to `None`).
    pub confidence_threshold_tempo: f32,

    /// Confidence threshold for key detection (0.0 to 1.0).
    pub confidence_threshold_key: f32,

    /// Confidence threshold for chord recognition (0.0 to 1.0).
    /// Chords below this threshold are filtered from the result.
    pub confidence_threshold_chord: f32,

    /// Confidence threshold for genre classification (0.0 to 1.0).
    pub confidence_threshold_genre: f32,

    /// Confidence threshold for mood detection (0.0 to 1.0).
    pub confidence_threshold_mood: f32,

    /// Number of stereo channels for mono conversion.
    /// Set to 2 to force stereo-to-mono conversion. 1 = mono input.
    pub num_channels: u8,
}

impl Default for MirConfig {
    fn default() -> Self {
        Self {
            window_size: 2048,
            hop_size: 512,
            min_tempo: 60.0,
            max_tempo: 200.0,
            enable_beat_tracking: true,
            enable_key_detection: true,
            enable_chord_recognition: true,
            enable_melody_extraction: true,
            enable_structure_analysis: true,
            enable_genre_classification: true,
            enable_mood_detection: true,
            enable_spectral_features: true,
            enable_rhythm_features: true,
            enable_harmonic_analysis: true,
            confidence_threshold_tempo: 0.0,
            confidence_threshold_key: 0.0,
            confidence_threshold_chord: 0.0,
            confidence_threshold_genre: 0.0,
            confidence_threshold_mood: 0.0,
            num_channels: 1,
        }
    }
}

/// Main MIR analyzer.
pub struct MirAnalyzer {
    config: MirConfig,
}

impl MirAnalyzer {
    /// Create a new MIR analyzer with the given configuration.
    #[must_use]
    pub fn new(config: MirConfig) -> Self {
        Self { config }
    }

    /// Analyze audio samples and extract all enabled features.
    ///
    /// Independent analysis branches (key, chord, melody, structure, genre,
    /// mood, spectral, rhythm, harmonic) are executed in parallel using rayon.
    /// Tempo detection is run first because beat tracking depends on its result.
    ///
    /// # Arguments
    ///
    /// * `samples` - Audio samples (f32, mono or interleaved stereo)
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    ///
    /// Complete analysis results including all enabled features.
    ///
    /// # Errors
    ///
    /// Returns error if analysis fails.
    #[allow(clippy::too_many_lines)]
    #[allow(clippy::cast_precision_loss)]
    pub fn analyze(&self, samples: &[f32], sample_rate: f32) -> MirResult<AnalysisResult> {
        // Convert to mono if stereo (force conversion when num_channels == 2)
        let mono = if self.config.num_channels == 2 {
            // Forced stereo-to-mono: average interleaved L/R pairs
            let half = samples.len() / 2;
            let mut out = Vec::with_capacity(half);
            for i in 0..half {
                out.push((samples[i * 2] + samples[i * 2 + 1]) * 0.5);
            }
            out
        } else {
            self.to_mono(samples)
        };

        // ── Tempo is needed for beat tracking so must run first ───────────
        let tempo = if self.config.enable_beat_tracking {
            let detector = tempo::TempoDetector::new(
                sample_rate,
                self.config.min_tempo,
                self.config.max_tempo,
            );
            Some(detector.detect(&mono)?)
        } else {
            None
        };

        // Beat tracking depends on tempo result
        let beat = if self.config.enable_beat_tracking {
            let tracker = beat::BeatTracker::new(sample_rate, self.config.hop_size);
            Some(tracker.track(&mono, tempo.as_ref())?)
        } else {
            None
        };

        // ── Parallel branch: 8 independent analyses ───────────────────────
        //
        // We represent each branch as a distinct index in a flat array so that
        // rayon can schedule them across the thread pool without needing to
        // unify heterogeneous closures into a single type.
        //
        //  0 = key          4 = genre
        //  1 = chord        5 = mood
        //  2 = melody       6 = spectral
        //  3 = structure    7 = rhythm
        //  8 = harmonic
        //
        // Each branch returns `Result<Option<BranchResult>, MirError>` encoded
        // as a `BranchOutput` enum to allow a single parallel collect pass.

        #[allow(clippy::large_enum_variant)]
        enum BranchOutput {
            Key(MirResult<Option<KeyResult>>),
            Chord(MirResult<Option<ChordResult>>),
            Melody(MirResult<Option<MelodyResult>>),
            Structure(MirResult<Option<StructureResult>>),
            Genre(MirResult<Option<GenreResult>>),
            Mood(MirResult<Option<MoodResult>>),
            Spectral(MirResult<Option<SpectralResult>>),
            Rhythm(MirResult<Option<RhythmResult>>),
            Harmonic(MirResult<Option<HarmonicResult>>),
        }

        let cfg = &self.config;
        let mono_ref: &[f32] = &mono;

        let results: Vec<BranchOutput> = (0_u8..9)
            .into_par_iter()
            .map(|branch| match branch {
                0 => BranchOutput::Key(if cfg.enable_key_detection {
                    let det = key::KeyDetector::new(sample_rate, cfg.window_size);
                    det.detect(mono_ref).map(Some)
                } else {
                    Ok(None)
                }),
                1 => BranchOutput::Chord(if cfg.enable_chord_recognition {
                    let rec =
                        chord::ChordRecognizer::new(sample_rate, cfg.window_size, cfg.hop_size);
                    rec.recognize(mono_ref).map(Some)
                } else {
                    Ok(None)
                }),
                2 => BranchOutput::Melody(if cfg.enable_melody_extraction {
                    let ext =
                        melody::MelodyExtractor::new(sample_rate, cfg.window_size, cfg.hop_size);
                    ext.extract(mono_ref).map(Some)
                } else {
                    Ok(None)
                }),
                3 => BranchOutput::Structure(if cfg.enable_structure_analysis {
                    let ana = structure::StructureAnalyzer::new(
                        sample_rate,
                        cfg.window_size,
                        cfg.hop_size,
                    );
                    ana.analyze(mono_ref).map(Some)
                } else {
                    Ok(None)
                }),
                4 => BranchOutput::Genre(if cfg.enable_genre_classification {
                    let cls = genre::GenreClassifier::new(sample_rate);
                    cls.classify(mono_ref).map(Some)
                } else {
                    Ok(None)
                }),
                5 => BranchOutput::Mood(if cfg.enable_mood_detection {
                    let det = mood::MoodDetector::new(sample_rate);
                    det.detect(mono_ref).map(Some)
                } else {
                    Ok(None)
                }),
                6 => BranchOutput::Spectral(if cfg.enable_spectral_features {
                    let ana =
                        spectral::SpectralAnalyzer::new(sample_rate, cfg.window_size, cfg.hop_size);
                    ana.analyze(mono_ref).map(Some)
                } else {
                    Ok(None)
                }),
                7 => BranchOutput::Rhythm(if cfg.enable_rhythm_features {
                    let ana = rhythm::RhythmAnalyzer::new(sample_rate, cfg.hop_size);
                    ana.analyze(mono_ref).map(Some)
                } else {
                    Ok(None)
                }),
                _ => BranchOutput::Harmonic(if cfg.enable_harmonic_analysis {
                    let ana =
                        harmonic::HarmonicAnalyzer::new(sample_rate, cfg.window_size, cfg.hop_size);
                    ana.analyze(mono_ref).map(Some)
                } else {
                    Ok(None)
                }),
            })
            .collect();

        // ── Unpack parallel results ────────────────────────────────────────
        let mut key_res: Option<KeyResult> = None;
        let mut chord_res: Option<ChordResult> = None;
        let mut melody_res: Option<MelodyResult> = None;
        let mut structure_res: Option<StructureResult> = None;
        let mut genre_res: Option<GenreResult> = None;
        let mut mood_res: Option<MoodResult> = None;
        let mut spectral_res: Option<SpectralResult> = None;
        let mut rhythm_res: Option<RhythmResult> = None;
        let mut harmonic_res: Option<HarmonicResult> = None;

        for output in results {
            match output {
                BranchOutput::Key(r) => key_res = r?,
                BranchOutput::Chord(r) => chord_res = r?,
                BranchOutput::Melody(r) => melody_res = r?,
                BranchOutput::Structure(r) => structure_res = r?,
                BranchOutput::Genre(r) => genre_res = r?,
                BranchOutput::Mood(r) => mood_res = r?,
                BranchOutput::Spectral(r) => spectral_res = r?,
                BranchOutput::Rhythm(r) => rhythm_res = r?,
                BranchOutput::Harmonic(r) => harmonic_res = r?,
            }
        }

        // ── Apply confidence thresholds -- discard low-confidence results ─
        let tempo = tempo.and_then(|t| {
            if t.confidence >= self.config.confidence_threshold_tempo {
                Some(t)
            } else {
                None
            }
        });

        let key = key_res.and_then(|k| {
            if k.confidence >= self.config.confidence_threshold_key {
                Some(k)
            } else {
                None
            }
        });

        let chord = chord_res.map(|mut c| {
            if self.config.confidence_threshold_chord > 0.0 {
                c.chords
                    .retain(|ch| ch.confidence >= self.config.confidence_threshold_chord);
            }
            c
        });

        let genre = genre_res.and_then(|g| {
            if g.top_genre_confidence >= self.config.confidence_threshold_genre {
                Some(g)
            } else {
                None
            }
        });

        let mood = mood_res.and_then(|m| {
            if m.intensity >= self.config.confidence_threshold_mood {
                Some(m)
            } else {
                None
            }
        });

        Ok(AnalysisResult {
            tempo,
            beat,
            key,
            chord,
            melody: melody_res,
            structure: structure_res,
            genre,
            mood,
            spectral: spectral_res,
            rhythm: rhythm_res,
            harmonic: harmonic_res,
            sample_rate,
            duration: mono.len() as f32 / sample_rate,
        })
    }

    /// Convert stereo to mono by averaging channels.
    ///
    /// Detects whether the input is stereo (interleaved L/R pairs) by checking
    /// if the sample count is even and the left/right channels show sufficient
    /// decorrelation. Falls back to treating the signal as mono if not stereo.
    fn to_mono(&self, samples: &[f32]) -> Vec<f32> {
        if samples.len() < 4 || samples.len() % 2 != 0 {
            return samples.to_vec();
        }

        // Heuristic: check if interleaved stereo by computing L/R correlation.
        // True stereo signals typically have decorrelated channels.
        let half = samples.len() / 2;
        let mut sum_l = 0.0_f64;
        let mut sum_r = 0.0_f64;
        let mut sum_ll = 0.0_f64;
        let mut sum_rr = 0.0_f64;
        let mut sum_lr = 0.0_f64;

        // Sample up to 4096 pairs to keep this fast
        let check_count = half.min(4096);
        for i in 0..check_count {
            let l = f64::from(samples[i * 2]);
            let r = f64::from(samples[i * 2 + 1]);
            sum_l += l;
            sum_r += r;
            sum_ll += l * l;
            sum_rr += r * r;
            sum_lr += l * r;
        }

        let n = check_count as f64;
        let var_l = (sum_ll / n) - (sum_l / n).powi(2);
        let var_r = (sum_rr / n) - (sum_r / n).powi(2);

        // If both channels have near-zero variance, treat as mono (silence or DC)
        if var_l < 1e-10 && var_r < 1e-10 {
            return samples.to_vec();
        }

        // Pearson correlation
        let denom = (var_l * var_r).sqrt();
        let correlation = if denom > 1e-12 {
            ((sum_lr / n) - (sum_l / n) * (sum_r / n)) / denom
        } else {
            1.0 // One channel is constant => treat as mono
        };

        // If correlation is very high (> 0.98), L and R are nearly identical:
        // the signal is likely mono data, not interleaved stereo.
        if correlation > 0.98 {
            return samples.to_vec();
        }

        // Down-mix interleaved stereo to mono by averaging L/R pairs.
        let mut mono = Vec::with_capacity(half);
        for i in 0..half {
            mono.push((samples[i * 2] + samples[i * 2 + 1]) * 0.5);
        }
        mono
    }

    /// Extract specific feature set.
    ///
    /// # Errors
    ///
    /// Returns error if feature extraction fails.
    pub fn extract_features(
        &self,
        samples: &[f32],
        sample_rate: f32,
        features: FeatureSet,
    ) -> MirResult<HashMap<String, Vec<f32>>> {
        let mono = self.to_mono(samples);
        let mut result = HashMap::new();

        if features.contains(FeatureSet::SPECTRAL) {
            let analyzer = spectral::SpectralAnalyzer::new(
                sample_rate,
                self.config.window_size,
                self.config.hop_size,
            );
            let spectral = analyzer.analyze(&mono)?;
            result.insert("spectral_centroid".to_string(), spectral.centroid);
            result.insert("spectral_rolloff".to_string(), spectral.rolloff);
            result.insert("spectral_flux".to_string(), spectral.flux);
        }

        if features.contains(FeatureSet::RHYTHM) {
            let analyzer = rhythm::RhythmAnalyzer::new(sample_rate, self.config.hop_size);
            let rhythm = analyzer.analyze(&mono)?;
            result.insert("onset_strength".to_string(), rhythm.onset_strength);
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    #[test]
    fn test_mir_config_default() {
        let config = MirConfig::default();
        assert_eq!(config.window_size, 2048);
        assert_eq!(config.hop_size, 512);
        assert!(config.enable_beat_tracking);
        assert!((config.confidence_threshold_tempo - 0.0).abs() < f32::EPSILON);
        assert!((config.confidence_threshold_key - 0.0).abs() < f32::EPSILON);
        assert_eq!(config.num_channels, 1);
    }

    #[test]
    fn test_mir_analyzer_creation() {
        let config = MirConfig::default();
        let _analyzer = MirAnalyzer::new(config);
    }

    #[test]
    fn test_analyze_silence() {
        let config = MirConfig {
            enable_beat_tracking: false, // Disable beat tracking for silence test
            enable_genre_classification: false,
            enable_structure_analysis: false,
            ..MirConfig::default()
        };
        let analyzer = MirAnalyzer::new(config);
        let samples = vec![0.0_f32; 44100]; // 1 second of silence
        let result = analyzer.analyze(&samples, 44100.0);
        assert!(result.is_ok());
    }

    // ── to_mono tests ──

    #[test]
    fn test_to_mono_mono_input() {
        let config = MirConfig::default();
        let analyzer = MirAnalyzer::new(config);
        let mono = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = analyzer.to_mono(&mono);
        // Odd-length input is always treated as mono
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_to_mono_stereo_detection() {
        let config = MirConfig::default();
        let analyzer = MirAnalyzer::new(config);
        let sr = 44100.0;
        let n = 8820; // ~200ms per channel

        // Create interleaved stereo with decorrelated channels
        let mut stereo = Vec::with_capacity(n * 2);
        for i in 0..n {
            let t = i as f32 / sr;
            let left = (TAU * 440.0 * t).sin(); // A4
            let right = (TAU * 554.37 * t).sin(); // C#5 -- different note
            stereo.push(left);
            stereo.push(right);
        }

        let result = analyzer.to_mono(&stereo);
        // Stereo detected => halved length
        assert_eq!(result.len(), n);
        // First sample should be average of L and R
        let expected = (stereo[0] + stereo[1]) * 0.5;
        assert!((result[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn test_to_mono_identical_channels_treated_as_mono() {
        let config = MirConfig::default();
        let analyzer = MirAnalyzer::new(config);

        // Interleaved but identical channels -> correlation ~1.0 -> treat as mono
        let mut data = Vec::with_capacity(8000);
        for i in 0..4000 {
            let v = (i as f32 / 100.0).sin();
            data.push(v);
            data.push(v); // Same value on both "channels"
        }

        let result = analyzer.to_mono(&data);
        // Should keep original length (treated as mono)
        assert_eq!(result.len(), 8000);
    }

    #[test]
    fn test_to_mono_short_signal() {
        let config = MirConfig::default();
        let analyzer = MirAnalyzer::new(config);
        let short = vec![1.0, 2.0];
        let result = analyzer.to_mono(&short);
        assert_eq!(result.len(), 2);
    }

    // ── Confidence threshold tests ──

    #[test]
    fn test_confidence_threshold_filters_tempo() {
        let config = MirConfig {
            enable_beat_tracking: true,
            enable_key_detection: false,
            enable_chord_recognition: false,
            enable_melody_extraction: false,
            enable_structure_analysis: false,
            enable_genre_classification: false,
            enable_mood_detection: false,
            enable_spectral_features: false,
            enable_rhythm_features: false,
            enable_harmonic_analysis: false,
            confidence_threshold_tempo: 0.999, // Very high threshold
            ..MirConfig::default()
        };
        let analyzer = MirAnalyzer::new(config);

        // Generate a signal with some periodic content
        let sr = 44100.0;
        let mut signal = Vec::new();
        for i in 0..(sr as usize * 3) {
            let t = i as f32 / sr;
            signal.push((TAU * 440.0 * t).sin());
        }

        let result = analyzer.analyze(&signal, sr);
        assert!(result.is_ok());
        // With threshold 0.999, tempo is likely filtered out
        // (detection of a pure tone rarely gives near-perfect confidence)
    }

    #[test]
    fn test_confidence_threshold_zero_keeps_all() {
        let config = MirConfig {
            enable_beat_tracking: false,
            enable_key_detection: true,
            enable_chord_recognition: false,
            enable_melody_extraction: false,
            enable_structure_analysis: false,
            enable_genre_classification: false,
            enable_mood_detection: false,
            enable_spectral_features: false,
            enable_rhythm_features: false,
            enable_harmonic_analysis: false,
            confidence_threshold_key: 0.0, // Accept everything
            ..MirConfig::default()
        };
        let analyzer = MirAnalyzer::new(config);
        let sr = 22050.0;
        let mut signal = Vec::new();
        for i in 0..(sr as usize * 2) {
            let t = i as f32 / sr;
            signal.push((TAU * 261.63 * t).sin()); // C note
        }

        let result = analyzer.analyze(&signal, sr);
        assert!(result.is_ok());
        let r = result.expect("should succeed");
        // With threshold 0.0, key should be present
        assert!(r.key.is_some());
    }

    // ── Forced stereo conversion via num_channels ──

    #[test]
    fn test_num_channels_forced_stereo() {
        let config = MirConfig {
            num_channels: 2,
            enable_beat_tracking: false,
            enable_key_detection: false,
            enable_chord_recognition: false,
            enable_melody_extraction: false,
            enable_structure_analysis: false,
            enable_genre_classification: false,
            enable_mood_detection: false,
            enable_spectral_features: true,
            enable_rhythm_features: false,
            enable_harmonic_analysis: false,
            ..MirConfig::default()
        };
        let analyzer = MirAnalyzer::new(config);

        // 4 samples interleaved = 2 mono samples
        let _stereo = vec![0.5, -0.5, 0.3, -0.3];
        // This is too short for spectral analysis but tests the conversion path
        let sr = 44100.0;

        // Create longer signal
        let mut stereo_long = Vec::new();
        for i in 0..44100 {
            let t = i as f32 / sr;
            stereo_long.push((TAU * 440.0 * t).sin());
            stereo_long.push((TAU * 550.0 * t).sin());
        }

        let result = analyzer.analyze(&stereo_long, sr);
        assert!(result.is_ok());
        let r = result.expect("should succeed");
        // Duration should be based on mono length (half the stereo = 44100 samples / 44100 Hz = 1.0s)
        assert!((r.duration - 1.0).abs() < 0.1);
    }
}
