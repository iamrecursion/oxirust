//! Fine-grained instrument recognition from MFCC and spectral features.
//!
//! This module provides [`InstrumentRecognizer`], which maps a pair of
//! (MFCC vector, spectral features) into confidence scores for 12 instrument
//! classes.  The scoring is based on Mahalanobis-like distance to
//! hand-calibrated per-instrument prototype centroids — a technique related
//! to Linear Discriminant Analysis (LDA).
//!
//! Each instrument prototype is defined by:
//! - Expected ranges for spectral centroid, rolloff, and flatness.
//! - Typical MFCC\[0\] (log-energy proxy) and MFCC\[1..2\] shape.
//!
//! Confidence scores are softmax-normalised so they sum to approximately 1.
//!
//! # Example
//!
//! ```rust
//! use oximedia_mir::instrument::{Instrument, InstrumentRecognizer};
//! use oximedia_mir::spectral_features::SpectralFeatures;
//!
//! let spectral = SpectralFeatures {
//!     centroid:   880.0,
//!     spread:     400.0,
//!     skewness:   0.1,
//!     kurtosis:   3.0,
//!     rolloff_85: 2000.0,
//!     flatness:   0.05,
//! };
//! let mfcc = vec![-8.0_f32, 3.0, 1.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
//!
//! let recognizer = InstrumentRecognizer::new();
//! let results = recognizer.recognize(&mfcc, &spectral);
//! println!("Top instrument: {:?} ({:.1}%)", results[0].0, results[0].1 * 100.0);
//! ```

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::suboptimal_flops)]

use crate::spectral_features::SpectralFeatures;

// ── Instrument enum ───────────────────────────────────────────────────────────

/// Fine-grained instrument class (12 categories).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Instrument {
    /// Acoustic or grand piano.
    Piano,
    /// Acoustic or electric guitar.
    Guitar,
    /// Violin (bowed string).
    Violin,
    /// Drum kit / percussion.
    Drums,
    /// Electric or upright bass.
    Bass,
    /// Trumpet.
    Trumpet,
    /// Alto or tenor saxophone.
    Saxophone,
    /// Concert flute.
    Flute,
    /// Human voice / vocals.
    Vocals,
    /// Electronic synthesizer / pad.
    Synth,
    /// Pipe or electronic organ.
    Organ,
    /// Cello (bowed string).
    Cello,
}

impl Instrument {
    /// Human-readable English name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Piano => "Piano",
            Self::Guitar => "Guitar",
            Self::Violin => "Violin",
            Self::Drums => "Drums",
            Self::Bass => "Bass",
            Self::Trumpet => "Trumpet",
            Self::Saxophone => "Saxophone",
            Self::Flute => "Flute",
            Self::Vocals => "Vocals",
            Self::Synth => "Synth",
            Self::Organ => "Organ",
            Self::Cello => "Cello",
        }
    }

    /// All 12 instruments in a fixed canonical order.
    fn all() -> [Self; 12] {
        [
            Self::Piano,
            Self::Guitar,
            Self::Violin,
            Self::Drums,
            Self::Bass,
            Self::Trumpet,
            Self::Saxophone,
            Self::Flute,
            Self::Vocals,
            Self::Synth,
            Self::Organ,
            Self::Cello,
        ]
    }
}

// ── Prototype definitions ─────────────────────────────────────────────────────
//
// Feature vector (6 elements):
//   [centroid_norm, rolloff_norm, flatness, mfcc0_norm, mfcc1, mfcc2]
//
// Normalisations:
//   centroid_norm  = centroid_hz  / 10_000
//   rolloff_norm   = rolloff_85   / 20_000
//   flatness       = spectral flatness (already in [0,1])
//   mfcc0_norm     = mfcc[0]      / 20     (log-energy; range ≈ [-20, 0])
//   mfcc1, mfcc2   = raw mfcc[1], mfcc[2]

const N_FEATS: usize = 6;
const N_INST: usize = 12;

/// Class-conditional prototype means [N_INST][N_FEATS].
static PROTO_MEANS: [[f32; N_FEATS]; N_INST] = [
    // Piano:      mid centroid, mid rolloff, low flatness, moderate energy
    [0.25, 0.20, 0.08, -0.40, 2.5, 1.2],
    // Guitar:     low-mid centroid, low rolloff, low flatness, warm mfcc
    [0.18, 0.14, 0.06, -0.35, 2.0, 0.8],
    // Violin:     mid-high centroid, narrow rolloff, low flatness, bright
    [0.35, 0.28, 0.04, -0.30, 1.5, 0.5],
    // Drums:      high centroid, wide rolloff, HIGH flatness, loud
    [0.55, 0.55, 0.70, -0.20, 0.5, -0.5],
    // Bass:       very low centroid, very low rolloff, low flatness
    [0.06, 0.05, 0.07, -0.45, 3.0, 1.5],
    // Trumpet:    high centroid, high rolloff, low flatness, bright
    [0.40, 0.35, 0.05, -0.25, 1.0, 0.0],
    // Saxophone:  mid-high centroid, mid rolloff, low flatness
    [0.30, 0.25, 0.06, -0.32, 1.8, 0.6],
    // Flute:      very high centroid, high rolloff, medium flatness, airy
    [0.50, 0.45, 0.25, -0.50, 1.2, 0.3],
    // Vocals:     mid centroid, mid rolloff, very low flatness, tonal
    [0.22, 0.18, 0.03, -0.38, 2.2, 1.0],
    // Synth:      variable centroid, high rolloff, HIGH flatness, synthetic
    [0.45, 0.40, 0.65, -0.35, 0.8, -0.3],
    // Organ:      mid-high centroid, mid rolloff, medium flatness
    [0.28, 0.22, 0.15, -0.38, 2.0, 0.8],
    // Cello:      low centroid, low rolloff, very low flatness, warm
    [0.12, 0.09, 0.04, -0.32, 2.8, 1.4],
];

/// Class-conditional standard deviations [N_INST][N_FEATS].
static PROTO_STDS: [[f32; N_FEATS]; N_INST] = [
    // Piano
    [0.08, 0.07, 0.04, 0.15, 1.5, 1.2],
    // Guitar
    [0.07, 0.06, 0.03, 0.15, 1.5, 1.2],
    // Violin
    [0.09, 0.08, 0.02, 0.12, 1.2, 1.0],
    // Drums
    [0.12, 0.12, 0.15, 0.12, 1.0, 1.0],
    // Bass
    [0.03, 0.03, 0.03, 0.15, 1.5, 1.2],
    // Trumpet
    [0.09, 0.09, 0.02, 0.12, 1.2, 1.0],
    // Saxophone
    [0.09, 0.08, 0.03, 0.12, 1.2, 1.0],
    // Flute
    [0.08, 0.08, 0.08, 0.15, 1.2, 1.0],
    // Vocals
    [0.07, 0.07, 0.02, 0.12, 1.2, 1.0],
    // Synth
    [0.12, 0.10, 0.15, 0.15, 1.5, 1.2],
    // Organ
    [0.08, 0.07, 0.05, 0.12, 1.2, 1.0],
    // Cello
    [0.05, 0.05, 0.02, 0.12, 1.5, 1.2],
];

// ── InstrumentRecognizer ──────────────────────────────────────────────────────

/// Recognises instrument classes from MFCC and spectral features.
pub struct InstrumentRecognizer {
    _priv: (),
}

impl InstrumentRecognizer {
    /// Construct a new recogniser.
    #[must_use]
    pub fn new() -> Self {
        Self { _priv: () }
    }

    /// Recognise instrument classes from MFCC and spectral features.
    ///
    /// # Arguments
    ///
    /// * `mfcc`     – MFCC coefficient vector (at least 3 elements; shorter vectors are zero-padded).
    /// * `spectral` – Spectral features computed from the same frame.
    ///
    /// # Returns
    ///
    /// A `Vec<(Instrument, f32)>` with all 12 classes sorted by confidence
    /// in descending order.  Confidence values sum to approximately 1.0.
    #[must_use]
    pub fn recognize(&self, mfcc: &[f32], spectral: &SpectralFeatures) -> Vec<(Instrument, f32)> {
        let fv = build_feature_vec(mfcc, spectral);
        let log_posts = compute_log_posteriors(&fv);
        let confidences = softmax(&log_posts);

        let mut result: Vec<(Instrument, f32)> = Instrument::all()
            .iter()
            .zip(confidences.iter())
            .map(|(&inst, &c)| (inst, c))
            .collect();

        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }
}

impl Default for InstrumentRecognizer {
    fn default() -> Self {
        Self::new()
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Build the 6-element normalised feature vector.
fn build_feature_vec(mfcc: &[f32], s: &SpectralFeatures) -> [f32; N_FEATS] {
    let centroid_norm = (s.centroid / 10_000.0).clamp(0.0, 1.5);
    let rolloff_norm = (s.rolloff_85 / 20_000.0).clamp(0.0, 1.5);
    let flatness = s.flatness.clamp(0.0, 1.0);
    let mfcc0_norm = mfcc.first().copied().unwrap_or(0.0) / 20.0;
    let mfcc1 = mfcc.get(1).copied().unwrap_or(0.0);
    let mfcc2 = mfcc.get(2).copied().unwrap_or(0.0);

    [
        centroid_norm,
        rolloff_norm,
        flatness,
        mfcc0_norm,
        mfcc1,
        mfcc2,
    ]
}

/// Log-Gaussian density.
#[inline]
fn log_gaussian(x: f32, mean: f32, std: f32) -> f32 {
    use std::f32::consts::PI;
    let std = std.max(1e-6);
    let diff = x - mean;
    -0.5 * (2.0 * PI).ln() - std.ln() - 0.5 * (diff / std).powi(2)
}

/// Compute log-posteriors under equal priors.
fn compute_log_posteriors(fv: &[f32; N_FEATS]) -> [f32; N_INST] {
    let mut lp = [0.0_f32; N_INST];
    for (i, lpi) in lp.iter_mut().enumerate() {
        for f in 0..N_FEATS {
            *lpi += log_gaussian(fv[f], PROTO_MEANS[i][f], PROTO_STDS[i][f]);
        }
        *lpi += -(N_INST as f32).ln(); // uniform prior
    }
    lp
}

/// Numerically-stable softmax over N_INST logits.
fn softmax(logits: &[f32; N_INST]) -> [f32; N_INST] {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exps = [0.0_f32; N_INST];
    let mut sum = 0.0_f32;
    for (i, &l) in logits.iter().enumerate() {
        exps[i] = (l - max).exp();
        sum += exps[i];
    }
    if sum < 1e-30 {
        return [1.0 / N_INST as f32; N_INST];
    }
    let mut out = [0.0_f32; N_INST];
    for (i, e) in exps.iter().enumerate() {
        out[i] = e / sum;
    }
    out
}

// ── Instrument Family Grouping ───────────────────────────────────────────────

/// High-level instrument family classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InstrumentFamilyGroup {
    /// Bowed or plucked string instruments (violin, viola, cello, double bass, harp).
    Strings,
    /// Woodwind instruments (flute, clarinet, oboe, bassoon, saxophone).
    Woodwinds,
    /// Brass instruments (trumpet, trombone, French horn, tuba).
    Brass,
    /// Unpitched and pitched percussion (drums, timpani, xylophone, etc.).
    Percussion,
    /// Keyboard instruments (piano, organ, harpsichord, accordion).
    Keys,
    /// Electronic instruments (synthesizer, drum machine, sampler).
    Electronic,
    /// Human voice.
    Vocal,
}

impl InstrumentFamilyGroup {
    /// Human-readable name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Strings => "Strings",
            Self::Woodwinds => "Woodwinds",
            Self::Brass => "Brass",
            Self::Percussion => "Percussion",
            Self::Keys => "Keys",
            Self::Electronic => "Electronic",
            Self::Vocal => "Vocal",
        }
    }

    /// All families.
    #[must_use]
    pub fn all() -> &'static [Self] {
        &[
            Self::Strings,
            Self::Woodwinds,
            Self::Brass,
            Self::Percussion,
            Self::Keys,
            Self::Electronic,
            Self::Vocal,
        ]
    }
}

impl Instrument {
    /// Return the instrument family group for this instrument.
    #[must_use]
    pub fn family(self) -> InstrumentFamilyGroup {
        match self {
            Self::Violin | Self::Cello | Self::Guitar | Self::Bass => {
                InstrumentFamilyGroup::Strings
            }
            Self::Flute | Self::Saxophone => InstrumentFamilyGroup::Woodwinds,
            Self::Trumpet => InstrumentFamilyGroup::Brass,
            Self::Drums => InstrumentFamilyGroup::Percussion,
            Self::Piano | Self::Organ => InstrumentFamilyGroup::Keys,
            Self::Synth => InstrumentFamilyGroup::Electronic,
            Self::Vocals => InstrumentFamilyGroup::Vocal,
        }
    }

    /// Return all instruments in a given family.
    #[must_use]
    pub fn instruments_in_family(family: InstrumentFamilyGroup) -> Vec<Instrument> {
        Self::all()
            .iter()
            .copied()
            .filter(|i| i.family() == family)
            .collect()
    }
}

// ── General MIDI Program Number Mapping ─────────────────────────────────────

/// General MIDI program number (0-127) mapping for an instrument.
#[derive(Debug, Clone, Copy)]
pub struct MidiProgramMapping {
    /// The instrument.
    pub instrument: Instrument,
    /// Primary GM program number (0-indexed).
    pub program_number: u8,
    /// Human-readable GM program name.
    pub program_name: &'static str,
}

/// Return the General MIDI program number mapping for an instrument.
///
/// Maps each instrument to its most common GM program number.
#[must_use]
pub fn midi_program(instrument: Instrument) -> MidiProgramMapping {
    match instrument {
        Instrument::Piano => MidiProgramMapping {
            instrument,
            program_number: 0,
            program_name: "Acoustic Grand Piano",
        },
        Instrument::Guitar => MidiProgramMapping {
            instrument,
            program_number: 25,
            program_name: "Acoustic Guitar (steel)",
        },
        Instrument::Violin => MidiProgramMapping {
            instrument,
            program_number: 40,
            program_name: "Violin",
        },
        Instrument::Drums => MidiProgramMapping {
            instrument,
            program_number: 0,
            program_name: "Standard Drum Kit (Channel 10)",
        },
        Instrument::Bass => MidiProgramMapping {
            instrument,
            program_number: 33,
            program_name: "Electric Bass (finger)",
        },
        Instrument::Trumpet => MidiProgramMapping {
            instrument,
            program_number: 56,
            program_name: "Trumpet",
        },
        Instrument::Saxophone => MidiProgramMapping {
            instrument,
            program_number: 65,
            program_name: "Alto Sax",
        },
        Instrument::Flute => MidiProgramMapping {
            instrument,
            program_number: 73,
            program_name: "Flute",
        },
        Instrument::Vocals => MidiProgramMapping {
            instrument,
            program_number: 52,
            program_name: "Choir Aahs",
        },
        Instrument::Synth => MidiProgramMapping {
            instrument,
            program_number: 80,
            program_name: "Lead 1 (square)",
        },
        Instrument::Organ => MidiProgramMapping {
            instrument,
            program_number: 19,
            program_name: "Church Organ",
        },
        Instrument::Cello => MidiProgramMapping {
            instrument,
            program_number: 42,
            program_name: "Cello",
        },
    }
}

/// Look up an instrument by General MIDI program number.
///
/// Returns the closest matching instrument for the given GM program number,
/// or `None` if no close match exists.
#[must_use]
pub fn instrument_from_midi_program(program: u8) -> Option<Instrument> {
    // GM program ranges mapped to our instruments
    match program {
        0..=7 => Some(Instrument::Piano),       // Piano family
        8..=15 => Some(Instrument::Piano),      // Chromatic Percussion (close to keys)
        16..=23 => Some(Instrument::Organ),     // Organ family
        24..=31 => Some(Instrument::Guitar),    // Guitar family
        32..=39 => Some(Instrument::Bass),      // Bass family
        40 => Some(Instrument::Violin),         // Violin
        41 => Some(Instrument::Violin),         // Viola (mapped to violin)
        42 => Some(Instrument::Cello),          // Cello
        43..=47 => Some(Instrument::Cello),     // Strings
        48..=55 => Some(Instrument::Vocals),    // Ensemble / Choir
        56..=59 => Some(Instrument::Trumpet),   // Brass
        60..=63 => Some(Instrument::Trumpet),   // More brass
        64..=71 => Some(Instrument::Saxophone), // Reed instruments
        72..=79 => Some(Instrument::Flute),     // Pipe / Flute
        80..=103 => Some(Instrument::Synth),    // Synth Lead + Pad
        104..=111 => Some(Instrument::Guitar),  // Ethnic (sitar etc.)
        112..=119 => Some(Instrument::Drums),   // Percussive
        120..=127 => Some(Instrument::Synth),   // Sound Effects
        128..=u8::MAX => None,                  // Out of GM range
    }
}

// ── Instrument Range Detection ──────────────────────────────────────────────

/// Frequency range for an instrument in Hz.
#[derive(Debug, Clone, Copy)]
pub struct InstrumentRange {
    /// The instrument.
    pub instrument: Instrument,
    /// Lowest fundamental frequency (Hz).
    pub low_hz: f32,
    /// Highest fundamental frequency (Hz).
    pub high_hz: f32,
    /// Typical performing range centre frequency (Hz).
    pub center_hz: f32,
}

impl InstrumentRange {
    /// The bandwidth in Hz.
    #[must_use]
    pub fn bandwidth(&self) -> f32 {
        self.high_hz - self.low_hz
    }

    /// Number of octaves the instrument spans.
    #[must_use]
    pub fn octaves(&self) -> f32 {
        if self.low_hz <= 0.0 {
            return 0.0;
        }
        (self.high_hz / self.low_hz).log2()
    }

    /// Whether a given frequency falls within this instrument's range.
    #[must_use]
    pub fn contains(&self, freq_hz: f32) -> bool {
        freq_hz >= self.low_hz && freq_hz <= self.high_hz
    }
}

/// Return the typical frequency range for an instrument.
///
/// Values represent the fundamental frequency range (not harmonics).
#[must_use]
pub fn instrument_range(instrument: Instrument) -> InstrumentRange {
    match instrument {
        Instrument::Piano => InstrumentRange {
            instrument,
            low_hz: 27.5,
            high_hz: 4186.0,
            center_hz: 440.0,
        },
        Instrument::Guitar => InstrumentRange {
            instrument,
            low_hz: 82.0,
            high_hz: 1175.0,
            center_hz: 330.0,
        },
        Instrument::Violin => InstrumentRange {
            instrument,
            low_hz: 196.0,
            high_hz: 3520.0,
            center_hz: 660.0,
        },
        Instrument::Drums => InstrumentRange {
            instrument,
            low_hz: 40.0,
            high_hz: 15000.0,
            center_hz: 1000.0,
        },
        Instrument::Bass => InstrumentRange {
            instrument,
            low_hz: 41.0,
            high_hz: 400.0,
            center_hz: 100.0,
        },
        Instrument::Trumpet => InstrumentRange {
            instrument,
            low_hz: 165.0,
            high_hz: 1175.0,
            center_hz: 466.0,
        },
        Instrument::Saxophone => InstrumentRange {
            instrument,
            low_hz: 138.0,
            high_hz: 880.0,
            center_hz: 370.0,
        },
        Instrument::Flute => InstrumentRange {
            instrument,
            low_hz: 262.0,
            high_hz: 2093.0,
            center_hz: 880.0,
        },
        Instrument::Vocals => InstrumentRange {
            instrument,
            low_hz: 80.0,
            high_hz: 1100.0,
            center_hz: 300.0,
        },
        Instrument::Synth => InstrumentRange {
            instrument,
            low_hz: 20.0,
            high_hz: 20000.0,
            center_hz: 440.0,
        },
        Instrument::Organ => InstrumentRange {
            instrument,
            low_hz: 32.7,
            high_hz: 4186.0,
            center_hz: 440.0,
        },
        Instrument::Cello => InstrumentRange {
            instrument,
            low_hz: 65.0,
            high_hz: 988.0,
            center_hz: 220.0,
        },
    }
}

/// Find all instruments whose range includes a given frequency.
#[must_use]
pub fn instruments_at_frequency(freq_hz: f32) -> Vec<Instrument> {
    Instrument::all()
        .iter()
        .copied()
        .filter(|&inst| instrument_range(inst).contains(freq_hz))
        .collect()
}

// ── Ensemble Detection ──────────────────────────────────────────────────────

/// A common instrumental ensemble configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EnsembleType {
    /// String quartet: 2 violins, viola (mapped to violin), cello.
    StringQuartet,
    /// Rock band: guitar, bass, drums, vocals.
    RockBand,
    /// Jazz trio: piano, bass, drums.
    JazzTrio,
    /// Jazz quartet: piano, bass, drums, saxophone.
    JazzQuartet,
    /// Orchestra: strings, brass, woodwinds, percussion.
    Orchestra,
    /// Piano trio: piano, violin, cello.
    PianoTrio,
    /// Pop band: vocals, guitar, bass, drums, synth.
    PopBand,
    /// Brass quintet: trumpets and brass.
    BrassQuintet,
    /// Electronic duo: synth + drums.
    ElectronicDuo,
    /// Singer-songwriter: vocals + guitar.
    SingerSongwriter,
}

impl EnsembleType {
    /// Human-readable name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::StringQuartet => "String Quartet",
            Self::RockBand => "Rock Band",
            Self::JazzTrio => "Jazz Trio",
            Self::JazzQuartet => "Jazz Quartet",
            Self::Orchestra => "Orchestra",
            Self::PianoTrio => "Piano Trio",
            Self::PopBand => "Pop Band",
            Self::BrassQuintet => "Brass Quintet",
            Self::ElectronicDuo => "Electronic Duo",
            Self::SingerSongwriter => "Singer-Songwriter",
        }
    }

    /// Required instruments for this ensemble.
    #[must_use]
    pub fn required_instruments(self) -> &'static [Instrument] {
        match self {
            Self::StringQuartet => &[Instrument::Violin, Instrument::Cello],
            Self::RockBand => &[
                Instrument::Guitar,
                Instrument::Bass,
                Instrument::Drums,
                Instrument::Vocals,
            ],
            Self::JazzTrio => &[Instrument::Piano, Instrument::Bass, Instrument::Drums],
            Self::JazzQuartet => &[
                Instrument::Piano,
                Instrument::Bass,
                Instrument::Drums,
                Instrument::Saxophone,
            ],
            Self::Orchestra => &[
                Instrument::Violin,
                Instrument::Cello,
                Instrument::Trumpet,
                Instrument::Flute,
            ],
            Self::PianoTrio => &[Instrument::Piano, Instrument::Violin, Instrument::Cello],
            Self::PopBand => &[
                Instrument::Vocals,
                Instrument::Guitar,
                Instrument::Bass,
                Instrument::Drums,
            ],
            Self::BrassQuintet => &[Instrument::Trumpet],
            Self::ElectronicDuo => &[Instrument::Synth, Instrument::Drums],
            Self::SingerSongwriter => &[Instrument::Vocals, Instrument::Guitar],
        }
    }
}

/// Result of ensemble detection.
#[derive(Debug, Clone)]
pub struct EnsembleDetection {
    /// Detected ensemble type.
    pub ensemble: EnsembleType,
    /// Confidence (0.0 to 1.0) based on how many required instruments are present.
    pub confidence: f32,
}

/// Detect likely ensemble types from a set of detected instruments.
///
/// Each detected instrument should have a confidence score.
/// Returns all matching ensembles sorted by confidence (descending).
///
/// # Arguments
/// * `detected` - slice of (Instrument, confidence) pairs
/// * `min_confidence` - minimum instrument confidence to count as present
#[must_use]
pub fn detect_ensembles(
    detected: &[(Instrument, f32)],
    min_confidence: f32,
) -> Vec<EnsembleDetection> {
    let present: std::collections::HashSet<Instrument> = detected
        .iter()
        .filter(|(_, c)| *c >= min_confidence)
        .map(|(inst, _)| *inst)
        .collect();

    let all_ensembles = [
        EnsembleType::StringQuartet,
        EnsembleType::RockBand,
        EnsembleType::JazzTrio,
        EnsembleType::JazzQuartet,
        EnsembleType::Orchestra,
        EnsembleType::PianoTrio,
        EnsembleType::PopBand,
        EnsembleType::BrassQuintet,
        EnsembleType::ElectronicDuo,
        EnsembleType::SingerSongwriter,
    ];

    let mut results: Vec<EnsembleDetection> = all_ensembles
        .iter()
        .filter_map(|&ensemble| {
            let required = ensemble.required_instruments();
            if required.is_empty() {
                return None;
            }
            let matched = required.iter().filter(|i| present.contains(i)).count();
            #[allow(clippy::cast_precision_loss)]
            let confidence = matched as f32 / required.len() as f32;
            if confidence > 0.0 {
                Some(EnsembleDetection {
                    ensemble,
                    confidence,
                })
            } else {
                None
            }
        })
        .collect();

    results.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spectral_features::SpectralFeatures;

    // ── Helper feature sets ────────────────────────────────────────────────────

    fn piano_spectral() -> SpectralFeatures {
        SpectralFeatures {
            centroid: 2500.0,
            spread: 600.0,
            skewness: 0.2,
            kurtosis: 3.0,
            rolloff_85: 5000.0,
            flatness: 0.07,
        }
    }
    fn piano_mfcc() -> Vec<f32> {
        let mut v = vec![0.0_f32; 13];
        v[0] = -8.0;
        v[1] = 2.5;
        v[2] = 1.2;
        v
    }

    fn drums_spectral() -> SpectralFeatures {
        SpectralFeatures {
            centroid: 5500.0,
            spread: 2000.0,
            skewness: 0.5,
            kurtosis: 4.0,
            rolloff_85: 11000.0,
            flatness: 0.70,
        }
    }
    fn drums_mfcc() -> Vec<f32> {
        let mut v = vec![0.0_f32; 13];
        v[0] = -4.0;
        v[1] = 0.5;
        v[2] = -0.5;
        v
    }

    fn flute_spectral() -> SpectralFeatures {
        SpectralFeatures {
            centroid: 5000.0,
            spread: 800.0,
            skewness: 0.1,
            kurtosis: 3.0,
            rolloff_85: 9000.0,
            flatness: 0.25,
        }
    }
    fn flute_mfcc() -> Vec<f32> {
        let mut v = vec![0.0_f32; 13];
        v[0] = -10.0;
        v[1] = 1.2;
        v[2] = 0.3;
        v
    }

    fn bass_spectral() -> SpectralFeatures {
        SpectralFeatures {
            centroid: 600.0,
            spread: 150.0,
            skewness: 0.0,
            kurtosis: 2.5,
            rolloff_85: 1000.0,
            flatness: 0.06,
        }
    }
    fn bass_mfcc() -> Vec<f32> {
        let mut v = vec![0.0_f32; 13];
        v[0] = -9.0;
        v[1] = 3.0;
        v[2] = 1.5;
        v
    }

    fn default_spectral() -> SpectralFeatures {
        SpectralFeatures {
            centroid: 2000.0,
            spread: 500.0,
            skewness: 0.0,
            kurtosis: 3.0,
            rolloff_85: 4000.0,
            flatness: 0.10,
        }
    }

    // ── Structural correctness ────────────────────────────────────────────────

    #[test]
    fn test_recognize_returns_twelve_entries() {
        let rec = InstrumentRecognizer::new();
        let result = rec.recognize(&piano_mfcc(), &default_spectral());
        assert_eq!(result.len(), 12, "should return 12 entries");
    }

    #[test]
    fn test_recognize_sorted_descending() {
        let rec = InstrumentRecognizer::new();
        let result = rec.recognize(&piano_mfcc(), &default_spectral());
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
        let rec = InstrumentRecognizer::new();
        let result = rec.recognize(&piano_mfcc(), &piano_spectral());
        let sum: f32 = result.iter().map(|(_, c)| c).sum();
        assert!((sum - 1.0).abs() < 1e-4, "confidence sum ≈1, got {sum:.6}");
    }

    #[test]
    fn test_each_confidence_in_range() {
        let rec = InstrumentRecognizer::new();
        let result = rec.recognize(&piano_mfcc(), &piano_spectral());
        for (inst, c) in &result {
            assert!(
                *c >= 0.0 && *c <= 1.0,
                "{inst:?} confidence out of [0,1]: {c:.4}"
            );
        }
    }

    #[test]
    fn test_all_instruments_appear_exactly_once() {
        let rec = InstrumentRecognizer::new();
        let result = rec.recognize(&piano_mfcc(), &default_spectral());
        for expected in Instrument::all() {
            let count = result.iter().filter(|(i, _)| *i == expected).count();
            assert_eq!(count, 1, "{expected:?} should appear exactly once");
        }
    }

    // ── Instrument discrimination ─────────────────────────────────────────────

    #[test]
    fn test_piano_features_top_instrument_reasonable() {
        let rec = InstrumentRecognizer::new();
        let result = rec.recognize(&piano_mfcc(), &piano_spectral());
        let top = result[0].0;
        // Piano or Organ or Guitar are all reasonable for piano-like features
        assert!(
            matches!(
                top,
                Instrument::Piano | Instrument::Organ | Instrument::Guitar | Instrument::Saxophone
            ),
            "piano features → expected piano-family, got {top:?}"
        );
    }

    #[test]
    fn test_drums_features_top_instrument_is_drums() {
        let rec = InstrumentRecognizer::new();
        let result = rec.recognize(&drums_mfcc(), &drums_spectral());
        assert_eq!(result[0].0, Instrument::Drums, "drums features → Drums");
    }

    #[test]
    fn test_flute_features_top_instrument_is_flute_or_synth() {
        let rec = InstrumentRecognizer::new();
        let result = rec.recognize(&flute_mfcc(), &flute_spectral());
        let top = result[0].0;
        assert!(
            matches!(
                top,
                Instrument::Flute | Instrument::Synth | Instrument::Trumpet
            ),
            "flute features → Flute/Synth/Trumpet expected, got {top:?}"
        );
    }

    #[test]
    fn test_bass_features_top_instrument_is_bass() {
        let rec = InstrumentRecognizer::new();
        let result = rec.recognize(&bass_mfcc(), &bass_spectral());
        assert_eq!(result[0].0, Instrument::Bass, "bass features → Bass");
    }

    // ── Edge cases ────────────────────────────────────────────────────────────

    #[test]
    fn test_empty_mfcc_still_works() {
        let rec = InstrumentRecognizer::new();
        let result = rec.recognize(&[], &default_spectral());
        assert_eq!(result.len(), 12);
    }

    #[test]
    fn test_short_mfcc_still_works() {
        let rec = InstrumentRecognizer::new();
        let result = rec.recognize(&[0.0], &default_spectral());
        assert_eq!(result.len(), 12);
    }

    #[test]
    fn test_default_recognizer_same_as_new() {
        let r1 = InstrumentRecognizer::new();
        let r2 = InstrumentRecognizer::default();
        let res1 = r1.recognize(&piano_mfcc(), &piano_spectral());
        let res2 = r2.recognize(&piano_mfcc(), &piano_spectral());
        for ((i1, c1), (i2, c2)) in res1.iter().zip(res2.iter()) {
            assert_eq!(i1, i2);
            assert!((c1 - c2).abs() < 1e-6);
        }
    }

    // ── Instrument family tests ───────────────────────────────────────────────

    #[test]
    fn test_piano_family_is_keys() {
        assert_eq!(Instrument::Piano.family(), InstrumentFamilyGroup::Keys);
    }

    #[test]
    fn test_violin_family_is_strings() {
        assert_eq!(Instrument::Violin.family(), InstrumentFamilyGroup::Strings);
    }

    #[test]
    fn test_trumpet_family_is_brass() {
        assert_eq!(Instrument::Trumpet.family(), InstrumentFamilyGroup::Brass);
    }

    #[test]
    fn test_flute_family_is_woodwinds() {
        assert_eq!(Instrument::Flute.family(), InstrumentFamilyGroup::Woodwinds);
    }

    #[test]
    fn test_drums_family_is_percussion() {
        assert_eq!(
            Instrument::Drums.family(),
            InstrumentFamilyGroup::Percussion
        );
    }

    #[test]
    fn test_synth_family_is_electronic() {
        assert_eq!(
            Instrument::Synth.family(),
            InstrumentFamilyGroup::Electronic
        );
    }

    #[test]
    fn test_vocals_family_is_vocal() {
        assert_eq!(Instrument::Vocals.family(), InstrumentFamilyGroup::Vocal);
    }

    #[test]
    fn test_instruments_in_strings_family() {
        let strings = Instrument::instruments_in_family(InstrumentFamilyGroup::Strings);
        assert!(strings.contains(&Instrument::Violin));
        assert!(strings.contains(&Instrument::Cello));
        assert!(strings.contains(&Instrument::Guitar));
        assert!(strings.contains(&Instrument::Bass));
    }

    #[test]
    fn test_all_families_covered() {
        // Every instrument should belong to exactly one family
        for inst in Instrument::all() {
            let _family = inst.family(); // should not panic
        }
    }

    #[test]
    fn test_family_group_names() {
        for family in InstrumentFamilyGroup::all() {
            assert!(!family.name().is_empty());
        }
    }

    // ── MIDI program tests ────────────────────────────────────────────────────

    #[test]
    fn test_midi_program_piano() {
        let m = midi_program(Instrument::Piano);
        assert_eq!(m.program_number, 0);
        assert_eq!(m.program_name, "Acoustic Grand Piano");
    }

    #[test]
    fn test_midi_program_violin() {
        let m = midi_program(Instrument::Violin);
        assert_eq!(m.program_number, 40);
    }

    #[test]
    fn test_midi_program_roundtrip_piano() {
        let inst = instrument_from_midi_program(0);
        assert_eq!(inst, Some(Instrument::Piano));
    }

    #[test]
    fn test_midi_program_roundtrip_cello() {
        let inst = instrument_from_midi_program(42);
        assert_eq!(inst, Some(Instrument::Cello));
    }

    #[test]
    fn test_midi_all_programs_mapped() {
        // Every program 0-127 should map to some instrument
        for p in 0..=127_u8 {
            assert!(
                instrument_from_midi_program(p).is_some(),
                "program {p} not mapped"
            );
        }
    }

    // ── Instrument range tests ────────────────────────────────────────────────

    #[test]
    fn test_instrument_range_piano_spans_many_octaves() {
        let r = instrument_range(Instrument::Piano);
        assert!(
            r.octaves() > 6.0,
            "piano should span >6 octaves, got {}",
            r.octaves()
        );
    }

    #[test]
    fn test_instrument_range_bass_low() {
        let r = instrument_range(Instrument::Bass);
        assert!(r.low_hz < 50.0, "bass lowest should be < 50 Hz");
        assert!(r.high_hz < 500.0, "bass highest should be < 500 Hz");
    }

    #[test]
    fn test_instrument_range_contains() {
        let r = instrument_range(Instrument::Violin);
        assert!(r.contains(440.0), "violin should contain A4 (440 Hz)");
        assert!(!r.contains(50.0), "violin should not contain 50 Hz");
    }

    #[test]
    fn test_instruments_at_440_hz() {
        let insts = instruments_at_frequency(440.0);
        assert!(insts.contains(&Instrument::Piano));
        assert!(insts.contains(&Instrument::Violin));
        assert!(
            !insts.contains(&Instrument::Bass),
            "bass range does not reach 440 Hz"
        );
    }

    #[test]
    fn test_instrument_range_bandwidth_positive() {
        for inst in Instrument::all() {
            let r = instrument_range(inst);
            assert!(
                r.bandwidth() > 0.0,
                "{:?} bandwidth should be positive",
                inst
            );
        }
    }

    // ── Ensemble detection tests ──────────────────────────────────────────────

    #[test]
    fn test_detect_rock_band() {
        let detected = vec![
            (Instrument::Guitar, 0.9),
            (Instrument::Bass, 0.8),
            (Instrument::Drums, 0.85),
            (Instrument::Vocals, 0.7),
        ];
        let ensembles = detect_ensembles(&detected, 0.5);
        let rock_band = ensembles
            .iter()
            .find(|e| e.ensemble == EnsembleType::RockBand);
        assert!(rock_band.is_some(), "should detect rock band");
        assert!((rock_band.expect("rock band").confidence - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_detect_jazz_trio() {
        let detected = vec![
            (Instrument::Piano, 0.9),
            (Instrument::Bass, 0.8),
            (Instrument::Drums, 0.7),
        ];
        let ensembles = detect_ensembles(&detected, 0.5);
        let jazz = ensembles
            .iter()
            .find(|e| e.ensemble == EnsembleType::JazzTrio);
        assert!(jazz.is_some(), "should detect jazz trio");
    }

    #[test]
    fn test_detect_ensemble_empty_input() {
        let ensembles = detect_ensembles(&[], 0.5);
        assert!(ensembles.is_empty());
    }

    #[test]
    fn test_detect_ensemble_min_confidence_filter() {
        let detected = vec![
            (Instrument::Guitar, 0.3), // below threshold
            (Instrument::Bass, 0.8),
            (Instrument::Drums, 0.7),
            (Instrument::Vocals, 0.6),
        ];
        let ensembles = detect_ensembles(&detected, 0.5);
        let rock_band = ensembles
            .iter()
            .find(|e| e.ensemble == EnsembleType::RockBand);
        // Guitar is below threshold so rock band should not be 100%
        if let Some(rb) = rock_band {
            assert!(rb.confidence < 1.0);
        }
    }

    #[test]
    fn test_ensemble_type_names() {
        assert_eq!(EnsembleType::StringQuartet.name(), "String Quartet");
        assert_eq!(EnsembleType::RockBand.name(), "Rock Band");
        assert_eq!(EnsembleType::JazzTrio.name(), "Jazz Trio");
    }

    #[test]
    fn test_ensemble_required_instruments_not_empty() {
        let all = [
            EnsembleType::StringQuartet,
            EnsembleType::RockBand,
            EnsembleType::JazzTrio,
            EnsembleType::JazzQuartet,
            EnsembleType::Orchestra,
            EnsembleType::PianoTrio,
            EnsembleType::PopBand,
            EnsembleType::BrassQuintet,
            EnsembleType::ElectronicDuo,
            EnsembleType::SingerSongwriter,
        ];
        for e in all {
            assert!(
                !e.required_instruments().is_empty(),
                "{:?} should have required instruments",
                e
            );
        }
    }

    #[test]
    fn test_detect_string_quartet() {
        let detected = vec![(Instrument::Violin, 0.9), (Instrument::Cello, 0.8)];
        let ensembles = detect_ensembles(&detected, 0.5);
        let sq = ensembles
            .iter()
            .find(|e| e.ensemble == EnsembleType::StringQuartet);
        assert!(sq.is_some(), "should detect string quartet");
    }

    #[test]
    fn test_detect_electronic_duo() {
        let detected = vec![(Instrument::Synth, 0.9), (Instrument::Drums, 0.8)];
        let ensembles = detect_ensembles(&detected, 0.5);
        let ed = ensembles
            .iter()
            .find(|e| e.ensemble == EnsembleType::ElectronicDuo);
        assert!(ed.is_some(), "should detect electronic duo");
    }
}
