//! Advanced audio analysis and forensics for `OxiMedia`.
//!
//! This crate provides comprehensive audio analysis capabilities for professional
//! audio applications including forensics, voice analysis, music analysis, and more.
//!
//! # Features
//!
//! ## Spectral Analysis
//! - Advanced frequency-domain analysis with multiple window functions
//! - Spectral centroid, flatness, crest factor, and bandwidth computation
//! - High-resolution spectral features for detailed audio characterization
//!
//! ## Voice Analysis
//! - Voice characteristic analysis (F0, formants, jitter, shimmer, HNR)
//! - Gender detection using formant analysis and F0 range
//! - Age estimation based on voice characteristics
//! - Emotion detection (anger, joy, sadness, neutral)
//! - Speaker identification and verification
//!
//! ## Music Analysis
//! - Harmonic analysis and chord progression detection
//! - Advanced rhythmic analysis extending MIR capabilities
//! - Timbral analysis for sound quality characterization
//! - Instrument identification using spectral and temporal features
//!
//! ## Source Separation
//! - Vocal/instrumental separation using harmonic-percussive decomposition
//! - Drum track isolation
//! - Bass line extraction
//! - Multi-source separation
//!
//! ## Echo and Reverb Analysis
//! - Echo and reverb detection
//! - Room acoustics analysis
//! - RT60 reverberation time measurement
//! - Early reflection pattern analysis
//!
//! ## Distortion Analysis
//! - Distortion detection and quantification
//! - Total Harmonic Distortion (THD) measurement
//! - Clipping detection with threshold analysis
//! - Non-linear distortion characterization
//!
//! ## Dynamic Range Analysis
//! - Detailed dynamic range computation
//! - Crest factor analysis
//! - RMS level tracking over time
//! - Loudness variation measurement
//!
//! ## Transient Detection
//! - Transient and attack detection
//! - Envelope analysis with ADSR characterization
//! - Onset strength function computation
//!
//! ## Pitch Analysis
//! - Pitch tracking using YIN algorithm (patent-free)
//! - Pitch contour analysis
//! - Vibrato detection and measurement
//! - F0 estimation with confidence scoring
//!
//! ## Formant Analysis
//! - Formant frequency analysis (F1-F4)
//! - Formant tracking over time
//! - Vowel detection and classification
//! - Linear Predictive Coding (LPC) for formant extraction
//!
//! ## Audio Forensics
//! - Audio authenticity verification
//! - Edit detection (cuts, splices, insertions)
//! - Compression history analysis
//! - Background noise consistency analysis
//! - ENF (Electrical Network Frequency) analysis
//!
//! ## Noise Analysis
//! - Noise profiling and characterization
//! - Noise type classification (white, pink, environmental)
//! - Signal-to-noise ratio (SNR) computation
//! - Noise floor estimation
//!
//! # Usage Example
//!
//! ```rust
//! use oximedia_audio_analysis::{
//!     AudioAnalyzer, AnalysisConfig,
//! };
//!
//! // Create analyzer with default configuration
//! let config = AnalysisConfig::default();
//! let analyzer = AudioAnalyzer::new(config);
//!
//! // Analyze audio samples
//! let samples = vec![0.0_f32; 44100]; // 1 second of audio
//! let sample_rate = 44100.0;
//!
//! let result = analyzer.analyze(&samples, sample_rate)?;
//!
//! // Access spectral features
//! println!("Spectral centroid: {:.1} Hz", result.spectral.centroid);
//! println!("Spectral flatness: {:.3}", result.spectral.flatness);
//!
//! // Access voice characteristics
//! if let Some(voice) = result.voice {
//!     println!("F0: {:.1} Hz", voice.f0);
//!     println!("Gender: {:?}", voice.gender);
//! }
//!
//! # Ok::<(), oximedia_audio_analysis::AnalysisError>(())
//! ```
//!
//! # Patent-Free Implementation
//!
//! All algorithms are implemented using patent-free methods:
//! - YIN algorithm for pitch detection
//! - LPC for formant analysis
//! - Harmonic-percussive separation for source separation
//! - Autocorrelation-based methods
//!
//! # Real-Time Capable
//!
//! Most analysis modules support frame-by-frame processing for real-time applications.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::similar_names)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::float_cmp)]
#![allow(clippy::struct_excessive_bools)]
#![allow(dead_code, clippy::missing_errors_doc, clippy::missing_panics_doc)]

pub mod beat;
pub mod cepstral;
pub mod chroma;
pub mod compression_analysis;
pub mod cqt;
pub mod distortion;
pub mod dynamics;
pub mod echo;
pub mod energy;
pub mod energy_contour;
pub mod event_detection;
pub mod forensics;
pub mod formant;
pub mod formant_track;
pub mod harmony;
pub mod loudness;
pub mod loudness_curve;
pub mod loudness_range;
/// Mel spectrogram computation for ML-oriented audio feature extraction.
pub mod mel_spectrogram;
pub mod music;
pub mod noise;
pub mod onset;
pub mod pitch;
pub mod pitch_detect;
pub mod pitch_tracker;
pub mod psychoacoustic;
pub mod quality_degradation;
pub mod rhythm;
/// Audio scene classification: Indoor, Outdoor, Quiet, Noisy, Speech, Music, Mixed.
pub mod scene_classify;
pub mod segmentation;
pub mod separate;
pub mod silence_detect;
pub mod singing;
pub mod spectral;
pub mod spectral_contrast;
pub mod spectral_features;
pub mod spectral_flux;
pub mod stereo_field;
pub mod tempo_analysis;
pub mod timbre;
pub mod transient;
pub mod voice;

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use thiserror::Error;

/// Errors that can occur during audio analysis.
#[derive(Error, Debug, Clone)]
pub enum AnalysisError {
    /// Invalid sample rate
    #[error("Invalid sample rate: {0}")]
    InvalidSampleRate(f32),

    /// Insufficient samples for analysis
    #[error("Insufficient samples: need at least {needed}, got {got}")]
    InsufficientSamples {
        /// Required number of samples
        needed: usize,
        /// Actual number of samples
        got: usize,
    },

    /// Invalid configuration parameter
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Analysis failed
    #[error("Analysis failed: {0}")]
    AnalysisFailed(String),

    /// FFT error
    #[error("FFT error: {0}")]
    FftError(String),

    /// Invalid input data
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Feature extraction failed
    #[error("Feature extraction failed: {0}")]
    FeatureExtractionFailed(String),
}

/// Result type for audio analysis operations.
pub type Result<T> = std::result::Result<T, AnalysisError>;

/// Configuration for audio analysis.
#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    /// FFT size for frequency analysis
    pub fft_size: usize,
    /// Hop size for frame-based analysis
    pub hop_size: usize,
    /// Window function type
    pub window_type: WindowType,
    /// Minimum frequency for analysis (Hz)
    pub min_frequency: f32,
    /// Maximum frequency for analysis (Hz)
    pub max_frequency: f32,
    /// Enable detailed analysis (slower but more accurate)
    pub detailed: bool,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            fft_size: 2048,
            hop_size: 512,
            window_type: WindowType::Hann,
            min_frequency: 20.0,
            max_frequency: 20000.0,
            detailed: false,
        }
    }
}

/// Window function types for spectral analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowType {
    /// Hann window (cosine-squared, good general purpose)
    Hann,
    /// Hamming window (modified cosine)
    Hamming,
    /// Blackman window (better frequency resolution)
    Blackman,
    /// Blackman-Harris window (very low sidelobes)
    BlackmanHarris,
    /// Rectangular window (no windowing)
    Rectangular,
}

/// Main audio analyzer that coordinates all analysis modules.
pub struct AudioAnalyzer {
    config: AnalysisConfig,
    spectral_analyzer: spectral::SpectralAnalyzer,
    voice_analyzer: voice::VoiceAnalyzer,
    pitch_tracker: pitch::PitchTracker,
    formant_analyzer: formant::FormantAnalyzer,
    dynamics_analyzer: dynamics::DynamicsAnalyzer,
    transient_detector: transient::TransientDetector,
}

impl AudioAnalyzer {
    /// Create a new audio analyzer with the given configuration.
    #[must_use]
    pub fn new(config: AnalysisConfig) -> Self {
        Self {
            spectral_analyzer: spectral::SpectralAnalyzer::new(config.clone()),
            voice_analyzer: voice::VoiceAnalyzer::new(config.clone()),
            pitch_tracker: pitch::PitchTracker::new(config.clone()),
            formant_analyzer: formant::FormantAnalyzer::new(config.clone()),
            dynamics_analyzer: dynamics::DynamicsAnalyzer::new(config.clone()),
            transient_detector: transient::TransientDetector::new(config.clone()),
            config,
        }
    }

    /// Perform comprehensive audio analysis on the given samples.
    ///
    /// # Arguments
    /// * `samples` - Audio samples (mono or interleaved stereo)
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    /// Complete analysis results including spectral, temporal, and high-level features.
    pub fn analyze(&self, samples: &[f32], sample_rate: f32) -> Result<AnalysisResult> {
        if !(8_000.0..=192_000.0).contains(&sample_rate) {
            return Err(AnalysisError::InvalidSampleRate(sample_rate));
        }

        if samples.len() < self.config.fft_size {
            return Err(AnalysisError::InsufficientSamples {
                needed: self.config.fft_size,
                got: samples.len(),
            });
        }

        // Perform all independent analyses concurrently via rayon.
        // The 5 sub-analyzers do not share mutable state, so they are safe to
        // run in parallel.  Voice analysis is kept sequential below because it
        // logically depends on the pitch result.
        let ((spectral_res, pitch_res), ((formants_res, dynamics_res), transients_res)) =
            rayon::join(
                || {
                    rayon::join(
                        || self.spectral_analyzer.analyze(samples, sample_rate),
                        || self.pitch_tracker.track(samples, sample_rate),
                    )
                },
                || {
                    rayon::join(
                        || {
                            rayon::join(
                                || self.formant_analyzer.analyze(samples, sample_rate),
                                || self.dynamics_analyzer.analyze(samples, sample_rate),
                            )
                        },
                        || self.transient_detector.detect(samples, sample_rate),
                    )
                },
            );

        let spectral = spectral_res?;
        let pitch_result = pitch_res?;
        let formants = formants_res?;
        let dynamics = dynamics_res?;
        let transients = transients_res?;

        // Voice analysis (optional, depends on pitch detection).
        // Must run AFTER pitch_result is available.
        let voice = if pitch_result.mean_f0 > 0.0 && pitch_result.voicing_rate > 0.5 {
            Some(self.voice_analyzer.analyze(samples, sample_rate)?)
        } else {
            None
        };

        Ok(AnalysisResult {
            spectral,
            pitch: pitch_result,
            formants,
            dynamics,
            transients,
            voice,
        })
    }

    /// Analyze audio in real-time, frame by frame.
    pub fn analyze_frame(&mut self, samples: &[f32], sample_rate: f32) -> Result<FrameAnalysis> {
        let spectral = self.spectral_analyzer.analyze_frame(samples, sample_rate)?;
        let pitch = self.pitch_tracker.track_frame(samples, sample_rate)?;
        let rms = compute_rms(samples);

        Ok(FrameAnalysis {
            spectral,
            pitch,
            rms,
        })
    }
}

/// Complete analysis result.
#[derive(Debug, Clone)]
pub struct AnalysisResult {
    /// Spectral analysis results
    pub spectral: spectral::SpectralFeatures,
    /// Pitch tracking results
    pub pitch: pitch::PitchResult,
    /// Formant analysis results
    pub formants: formant::FormantResult,
    /// Dynamic range analysis results
    pub dynamics: dynamics::DynamicsResult,
    /// Transient detection results
    pub transients: transient::TransientResult,
    /// Voice analysis results (optional)
    pub voice: Option<voice::VoiceCharacteristics>,
}

/// Frame-level analysis result for real-time processing.
#[derive(Debug, Clone)]
pub struct FrameAnalysis {
    /// Spectral features
    pub spectral: spectral::SpectralFeatures,
    /// Pitch estimate
    pub pitch: pitch::PitchEstimate,
    /// RMS level
    pub rms: f32,
}

/// Global window-coefficient cache keyed by `(WindowType discriminant, size)`.
///
/// Stores `Arc<Vec<f32>>` so that multiple `SpectralAnalyzer` instances with
/// the same `(window_type, size)` pair share the same backing allocation — no
/// trigonometric computation and no per-instance copy is needed after the first
/// construction.
static WINDOW_CACHE: OnceLock<Mutex<HashMap<(u8, usize), Arc<Vec<f32>>>>> = OnceLock::new();

/// Discriminant index used as the cache key for each `WindowType` variant.
fn window_type_index(window_type: WindowType) -> u8 {
    match window_type {
        WindowType::Hann => 0,
        WindowType::Hamming => 1,
        WindowType::Blackman => 2,
        WindowType::BlackmanHarris => 3,
        WindowType::Rectangular => 4,
    }
}

/// Return a shared `Arc` to the window coefficients for `(window_type, size)`.
///
/// On the first call for a given pair the coefficients are computed and
/// inserted into the global cache.  Subsequent calls clone only the `Arc`
/// handle — no trigonometric computation is repeated and no `Vec` is copied.
///
/// # Concurrency
/// The cache is protected by a `Mutex`.  The double-checked locking pattern
/// used here is safe: after releasing the read lock and before acquiring the
/// write lock another thread may have inserted the entry; the `entry()` API
/// avoids a redundant overwrite in that case.
#[must_use]
pub fn get_or_compute_window(window_type: WindowType, size: usize) -> Arc<Vec<f32>> {
    let cache = WINDOW_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (window_type_index(window_type), size);

    // Fast path: already in cache — only clone the Arc handle.
    {
        let guard = cache.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(arc) = guard.get(&key) {
            return Arc::clone(arc);
        }
    }

    // Slow path: compute the coefficients and store them.
    let coefficients = Arc::new(compute_window_uncached(window_type, size));
    {
        let mut guard = cache.lock().unwrap_or_else(|e| e.into_inner());
        // Another thread might have inserted the entry while we waited.
        let entry = guard
            .entry(key)
            .or_insert_with(|| Arc::clone(&coefficients));
        Arc::clone(entry)
    }
}

/// Generate (and cache) a window function of the specified type and size.
///
/// Returns an owned `Vec<f32>`.  If the same `(window_type, size)` pair has
/// been computed before the cached coefficients are cloned into a new `Vec`.
/// Prefer [`get_or_compute_window`] when you only need a shared reference, as
/// it avoids the `Vec` copy entirely.
#[must_use]
pub fn generate_window(window_type: WindowType, size: usize) -> Vec<f32> {
    // Delegate to the Arc-based cache so there is a single source of truth.
    (*get_or_compute_window(window_type, size)).clone()
}

/// Raw (uncached) window computation.
fn compute_window_uncached(window_type: WindowType, size: usize) -> Vec<f32> {
    match window_type {
        WindowType::Hann => hann_window(size),
        WindowType::Hamming => hamming_window(size),
        WindowType::Blackman => blackman_window(size),
        WindowType::BlackmanHarris => blackman_harris_window(size),
        WindowType::Rectangular => vec![1.0; size],
    }
}

fn hann_window(size: usize) -> Vec<f32> {
    // Symmetric Hann (raised-cosine) window: 0.5·(1 − cos(2π·i/(N−1))).
    //
    // NOTE: the cosine argument MUST span a full 2π over the window. An earlier
    // version used `PI` (not `2·PI`), which produced only a half-cosine ramp
    // rising from 0 to 1 — not a symmetric bell. That degenerate window has very
    // poor spectral concentration: a pure tone leaked across the entire spectrum,
    // pulling the spectral centroid far above the true pitch and making it swing
    // wildly with sub-bin phase (e.g. a 440 Hz tone reported a ~1800–3000 Hz
    // centroid instead of ~440 Hz), which in turn produced dozens of spurious
    // splice detections in the forensic edit detector.
    if size <= 1 {
        return vec![1.0; size];
    }
    (0..size)
        .map(|i| {
            let x = 2.0 * std::f32::consts::PI * i as f32 / (size - 1) as f32;
            0.5 * (1.0 - x.cos())
        })
        .collect()
}

fn hamming_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| {
            let x = 2.0 * std::f32::consts::PI * i as f32 / (size - 1) as f32;
            0.54 - 0.46 * x.cos()
        })
        .collect()
}

fn blackman_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| {
            let x = 2.0 * std::f32::consts::PI * i as f32 / (size - 1) as f32;
            0.42 - 0.5 * x.cos() + 0.08 * (2.0 * x).cos()
        })
        .collect()
}

fn blackman_harris_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| {
            let x = 2.0 * std::f32::consts::PI * i as f32 / (size - 1) as f32;
            0.35875 - 0.48829 * x.cos() + 0.14128 * (2.0 * x).cos() - 0.01168 * (3.0 * x).cos()
        })
        .collect()
}

/// Compute RMS (Root Mean Square) level of audio samples.
#[must_use]
pub fn compute_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }

    let sum_squares: f32 = samples.iter().map(|&x| x * x).sum();
    (sum_squares / samples.len() as f32).sqrt()
}

/// Compute zero-crossing rate.
#[must_use]
pub fn zero_crossing_rate(samples: &[f32]) -> f32 {
    if samples.len() < 2 {
        return 0.0;
    }

    let mut crossings = 0;
    for i in 1..samples.len() {
        if (samples[i] >= 0.0 && samples[i - 1] < 0.0)
            || (samples[i] < 0.0 && samples[i - 1] >= 0.0)
        {
            crossings += 1;
        }
    }

    crossings as f32 / (samples.len() - 1) as f32
}

/// Convert amplitude to decibels.
#[must_use]
pub fn amplitude_to_db(amplitude: f32) -> f32 {
    if amplitude <= 0.0 {
        -100.0 // Floor at -100 dB
    } else {
        20.0 * amplitude.log10()
    }
}

/// Convert decibels to amplitude.
#[must_use]
pub fn db_to_amplitude(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_window_generation() {
        let size = 1024;
        let hann = generate_window(WindowType::Hann, size);
        assert_eq!(hann.len(), size);
        assert!(hann[0] < 0.01); // Near zero at start
                                 // Hann window maximum should be near center
        let max_val = hann.iter().copied().fold(0.0_f32, f32::max);
        assert!(max_val > 0.9); // Maximum value should be close to 1

        // A correct *symmetric* Hann window must also taper back to ~0 at the END
        // and peak in the MIDDLE. (A degenerate half-cosine ramp — the previous
        // bug — ends at 1.0 and peaks at the last sample, failing both checks.)
        assert!(
            hann[size - 1] < 0.01,
            "Hann must taper to ~0 at the end, got {}",
            hann[size - 1]
        );
        assert!(
            hann[size / 2] > 0.99,
            "Hann must peak (~1.0) near the center, got {}",
            hann[size / 2]
        );
    }

    #[test]
    fn test_rms_computation() {
        let samples = vec![1.0, -1.0, 1.0, -1.0];
        let rms = compute_rms(&samples);
        assert!((rms - 1.0).abs() < 1e-6);

        let zeros = vec![0.0; 100];
        assert_eq!(compute_rms(&zeros), 0.0);
    }

    #[test]
    fn test_zero_crossing_rate() {
        let samples = vec![1.0, -1.0, 1.0, -1.0, 1.0];
        let zcr = zero_crossing_rate(&samples);
        assert!((zcr - 1.0).abs() < 1e-6); // All transitions are crossings

        let constant = vec![1.0; 10];
        assert_eq!(zero_crossing_rate(&constant), 0.0);
    }

    #[test]
    fn test_db_conversion() {
        let amp = 0.5;
        let db = amplitude_to_db(amp);
        let back = db_to_amplitude(db);
        assert!((amp - back).abs() < 1e-6);

        assert_eq!(amplitude_to_db(1.0), 0.0);
        assert_eq!(amplitude_to_db(0.0), -100.0);
    }

    #[test]
    fn test_analysis_config() {
        let config = AnalysisConfig::default();
        assert_eq!(config.fft_size, 2048);
        assert_eq!(config.hop_size, 512);
        assert_eq!(config.window_type, WindowType::Hann);
    }

    // ── Item 1: parallel analyze test ──────────────────────────────────────────

    /// Verify that the parallel `AudioAnalyzer::analyze` produces bit-identical
    /// output to a fresh analyzer run on the same input.  rayon's work-stealing
    /// scheduler is deterministic for pure functions so the results must match.
    #[test]
    fn test_parallel_analyze_matches_sequential() {
        let config = AnalysisConfig {
            fft_size: 2048,
            hop_size: 512,
            ..Default::default()
        };
        let analyzer = AudioAnalyzer::new(config.clone());

        // Generate 1 second of 440 Hz sine (pure signal — voicing_rate will be
        // low enough that voice analysis is skipped, keeping the test simple).
        let sample_rate = 44100.0_f32;
        let samples: Vec<f32> = (0..44100)
            .map(|i| {
                let t = i as f32 / sample_rate;
                (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5
            })
            .collect();

        // Run the parallel version.
        let result_a = analyzer
            .analyze(&samples, sample_rate)
            .expect("parallel analyze failed");

        // Run again with a second fresh analyzer (same config) — because all
        // sub-analyzers are pure/deterministic the results must be identical.
        let analyzer2 = AudioAnalyzer::new(config);
        let result_b = analyzer2
            .analyze(&samples, sample_rate)
            .expect("sequential-style analyze failed");

        // Compare all scalar fields from SpectralFeatures.
        assert!(
            (result_a.spectral.centroid - result_b.spectral.centroid).abs() < 1e-4,
            "centroid mismatch: {} vs {}",
            result_a.spectral.centroid,
            result_b.spectral.centroid,
        );
        assert!(
            (result_a.spectral.flatness - result_b.spectral.flatness).abs() < 1e-6,
            "flatness mismatch",
        );
        assert!(
            (result_a.pitch.mean_f0 - result_b.pitch.mean_f0).abs() < 1e-3,
            "mean_f0 mismatch: {} vs {}",
            result_a.pitch.mean_f0,
            result_b.pitch.mean_f0,
        );
        // Dynamics and transients must also be equal.
        assert_eq!(
            result_a.transients.transient_times.len(),
            result_b.transients.transient_times.len(),
            "transient_times length mismatch",
        );
    }

    // ── Item 2: window-coefficient cache tests ─────────────────────────────────

    /// Two calls with the same (type, size) must return bit-identical coefficients.
    #[test]
    fn test_window_cache_identical() {
        let a = get_or_compute_window(WindowType::Hann, 1024);
        let b = get_or_compute_window(WindowType::Hann, 1024);

        // Content must be bit-identical.
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.to_bits(), y.to_bits(), "window coefficients differ");
        }
        // The two Arcs must point to the same allocation (pointer equality).
        assert!(
            Arc::ptr_eq(&a, &b),
            "expected same Arc backing allocation for cache hit"
        );
    }

    /// Different sizes must produce separate cache entries with the correct lengths.
    #[test]
    fn test_window_cache_different_sizes() {
        let w512 = get_or_compute_window(WindowType::Hann, 512);
        let w1024 = get_or_compute_window(WindowType::Hann, 1024);

        assert_eq!(w512.len(), 512);
        assert_eq!(w1024.len(), 1024);
        assert!(
            !Arc::ptr_eq(&w512, &w1024),
            "distinct sizes must not share Arc"
        );
    }
}
