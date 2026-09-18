//! Pitch class tracking and pitch history analysis.
//!
//! Provides note-level pitch classification, a rolling pitch history buffer,
//! and basic melody-line extraction on top of raw F0 estimates.

#![allow(dead_code)]

use std::collections::VecDeque;

/// The twelve pitch classes of the chromatic scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PitchClass {
    /// C natural.
    C,
    /// C sharp / D flat.
    CSharp,
    /// D natural.
    D,
    /// D sharp / E flat.
    DSharp,
    /// E natural.
    E,
    /// F natural.
    F,
    /// F sharp / G flat.
    FSharp,
    /// G natural.
    G,
    /// G sharp / A flat.
    GSharp,
    /// A natural.
    A,
    /// A sharp / B flat.
    ASharp,
    /// B natural.
    B,
}

impl PitchClass {
    /// Return the pitch class for a given MIDI note number (0–127).
    #[must_use]
    pub fn from_midi(midi: u8) -> Self {
        match midi % 12 {
            0 => Self::C,
            1 => Self::CSharp,
            2 => Self::D,
            3 => Self::DSharp,
            4 => Self::E,
            5 => Self::F,
            6 => Self::FSharp,
            7 => Self::G,
            8 => Self::GSharp,
            9 => Self::A,
            10 => Self::ASharp,
            _ => Self::B,
        }
    }

    /// Return the pitch class closest to the given frequency in Hz, based on
    /// A4 = 440 Hz.
    #[must_use]
    pub fn from_frequency(freq_hz: f32) -> Option<Self> {
        if freq_hz <= 0.0 {
            return None;
        }
        // MIDI note number closest to freq_hz
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let midi = (69.0_f32 + 12.0 * (freq_hz / 440.0).log2()).round() as u8;
        Some(Self::from_midi(midi))
    }

    /// Short ASCII name for the pitch class (e.g. "C#").
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::C => "C",
            Self::CSharp => "C#",
            Self::D => "D",
            Self::DSharp => "D#",
            Self::E => "E",
            Self::F => "F",
            Self::FSharp => "F#",
            Self::G => "G",
            Self::GSharp => "G#",
            Self::A => "A",
            Self::ASharp => "A#",
            Self::B => "B",
        }
    }

    /// Chromatic index of the pitch class (0 = C, 11 = B).
    #[must_use]
    pub fn index(&self) -> u8 {
        match self {
            Self::C => 0,
            Self::CSharp => 1,
            Self::D => 2,
            Self::DSharp => 3,
            Self::E => 4,
            Self::F => 5,
            Self::FSharp => 6,
            Self::G => 7,
            Self::GSharp => 8,
            Self::A => 9,
            Self::ASharp => 10,
            Self::B => 11,
        }
    }
}

/// A single frame-level pitch observation.
#[derive(Debug, Clone)]
pub struct PitchObservation {
    /// Fundamental frequency estimate in Hz (0.0 = unvoiced).
    pub f0_hz: f32,
    /// Closest pitch class (None if unvoiced).
    pub pitch_class: Option<PitchClass>,
    /// MIDI octave of the note (4 = middle octave, A4 = 440 Hz).
    pub octave: Option<i32>,
    /// Confidence score \[0.0, 1.0\].
    pub confidence: f32,
}

impl PitchObservation {
    /// Create a new voiced observation.
    #[must_use]
    pub fn voiced(f0_hz: f32, confidence: f32) -> Self {
        let pitch_class = PitchClass::from_frequency(f0_hz);
        #[allow(clippy::cast_possible_truncation)]
        let octave = if f0_hz > 0.0 {
            Some((69.0_f32 + 12.0 * (f0_hz / 440.0).log2()).round() as i32 / 12 - 1)
        } else {
            None
        };
        Self {
            f0_hz,
            pitch_class,
            octave,
            confidence,
        }
    }

    /// Create an unvoiced (silence / noise) observation.
    #[must_use]
    pub fn unvoiced() -> Self {
        Self {
            f0_hz: 0.0,
            pitch_class: None,
            octave: None,
            confidence: 0.0,
        }
    }

    /// Returns `true` when a voiced pitch was detected.
    #[must_use]
    pub fn is_voiced(&self) -> bool {
        self.f0_hz > 0.0 && self.confidence > 0.3
    }
}

/// Rolling buffer of recent pitch observations.
#[derive(Debug, Clone)]
pub struct PitchHistory {
    capacity: usize,
    history: VecDeque<PitchObservation>,
}

impl PitchHistory {
    /// Create a new [`PitchHistory`] with the given buffer capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            history: VecDeque::with_capacity(capacity),
        }
    }

    /// Push a new observation into the history, evicting the oldest if full.
    pub fn push(&mut self, obs: PitchObservation) {
        if self.history.len() == self.capacity {
            self.history.pop_front();
        }
        self.history.push_back(obs);
    }

    /// Number of observations currently stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.history.len()
    }

    /// Returns `true` when the history buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    /// Mean F0 of all voiced frames in the history (returns 0.0 if none).
    #[must_use]
    pub fn mean_f0(&self) -> f32 {
        let voiced: Vec<f32> = self
            .history
            .iter()
            .filter(|o| o.is_voiced())
            .map(|o| o.f0_hz)
            .collect();
        if voiced.is_empty() {
            return 0.0;
        }
        voiced.iter().sum::<f32>() / voiced.len() as f32
    }

    /// Return the most-frequent pitch class in the history window.
    #[must_use]
    pub fn dominant_pitch_class(&self) -> Option<PitchClass> {
        let mut counts = [0u32; 12];
        for obs in self.history.iter().filter(|o| o.is_voiced()) {
            if let Some(pc) = obs.pitch_class {
                counts[pc.index() as usize] += 1;
            }
        }
        let max_idx = counts
            .iter()
            .enumerate()
            .max_by_key(|&(_, &v)| v)
            .map(|(i, _)| i)?;
        if counts[max_idx] == 0 {
            return None;
        }
        Some(PitchClass::from_midi(max_idx as u8))
    }

    /// Fraction of frames in the history that are voiced \[0.0, 1.0\].
    #[must_use]
    pub fn voicing_rate(&self) -> f32 {
        if self.history.is_empty() {
            return 0.0;
        }
        let voiced = self.history.iter().filter(|o| o.is_voiced()).count();
        voiced as f32 / self.history.len() as f32
    }

    /// Iterate over all stored observations.
    pub fn iter(&self) -> impl Iterator<Item = &PitchObservation> {
        self.history.iter()
    }
}

/// Tracks pitch frame-by-frame and maintains a rolling [`PitchHistory`].
pub struct PitchTracker {
    sample_rate: f32,
    frame_size: usize,
    history: PitchHistory,
    /// YIN threshold for voiced/unvoiced decision.
    yin_threshold: f32,
}

impl PitchTracker {
    /// Create a new [`PitchTracker`].
    ///
    /// # Arguments
    /// * `sample_rate`   – Sample rate in Hz.
    /// * `frame_size`    – Analysis frame length in samples.
    /// * `history_size`  – Number of frames retained in the pitch history.
    #[must_use]
    pub fn new(sample_rate: f32, frame_size: usize, history_size: usize) -> Self {
        Self {
            sample_rate,
            frame_size,
            history: PitchHistory::new(history_size),
            yin_threshold: 0.1,
        }
    }

    /// Process a single audio frame and return the resulting observation.
    ///
    /// Also pushes the observation into the internal history buffer.
    pub fn process_frame(&mut self, frame: &[f32]) -> PitchObservation {
        let obs = self.estimate_pitch(frame);
        self.history.push(obs.clone());
        obs
    }

    /// Immutable reference to the internal pitch history.
    #[must_use]
    pub fn history(&self) -> &PitchHistory {
        &self.history
    }

    /// Reset the pitch history.
    pub fn reset(&mut self) {
        self.history = PitchHistory::new(self.history.capacity);
    }

    // ── private helpers ────────────────────────────────────────────────────

    /// Lightweight YIN-inspired pitch estimator.
    #[allow(clippy::cast_precision_loss)]
    fn estimate_pitch(&self, frame: &[f32]) -> PitchObservation {
        let n = frame.len().min(self.frame_size);
        if n < 8 {
            return PitchObservation::unvoiced();
        }

        // Reject silence / near-silence before running YIN
        let energy: f32 = frame[..n].iter().map(|&s| s * s).sum::<f32>() / n as f32;
        if energy < 1e-10 {
            return PitchObservation::unvoiced();
        }

        // Difference function d(tau)
        let tau_max = n / 2;
        let mut diff = vec![0.0_f32; tau_max];
        for tau in 1..tau_max {
            diff[tau] = (0..tau_max - tau)
                .map(|j| {
                    let d = frame[j] - frame[j + tau];
                    d * d
                })
                .sum();
        }

        // Cumulative mean normalised difference
        diff[0] = 1.0;
        let mut running = 0.0_f32;
        #[allow(clippy::needless_range_loop)]
        for tau in 1..tau_max {
            running += diff[tau];
            if running > 0.0 {
                diff[tau] *= tau as f32 / running;
            }
        }

        // Find first dip below threshold
        let min_tau = (self.sample_rate / 800.0) as usize; // ~800 Hz max
        let max_tau = (self.sample_rate / 50.0) as usize; //  ~50 Hz min
        let max_tau = max_tau.min(tau_max - 1);

        let mut best_tau = 0usize;
        #[allow(clippy::needless_range_loop)]
        for tau in min_tau..=max_tau {
            if diff[tau] < self.yin_threshold {
                best_tau = tau;
                break;
            }
        }

        if best_tau == 0 {
            return PitchObservation::unvoiced();
        }

        let f0 = self.sample_rate / best_tau as f32;
        let confidence = (1.0 - diff[best_tau]).clamp(0.0, 1.0);
        PitchObservation::voiced(f0, confidence)
    }
}

impl Default for PitchTracker {
    fn default() -> Self {
        Self::new(44100.0, 2048, 128)
    }
}

// ── unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── PitchClass ────────────────────────────────────────────────────────

    #[test]
    fn test_pitch_class_from_midi_c() {
        assert_eq!(PitchClass::from_midi(0), PitchClass::C);
        assert_eq!(PitchClass::from_midi(12), PitchClass::C);
        assert_eq!(PitchClass::from_midi(60), PitchClass::C); // Middle C
    }

    #[test]
    fn test_pitch_class_from_midi_a() {
        assert_eq!(PitchClass::from_midi(69), PitchClass::A); // A4 = 440 Hz
    }

    #[test]
    fn test_pitch_class_from_frequency_a4() {
        let pc = PitchClass::from_frequency(440.0);
        assert_eq!(pc, Some(PitchClass::A));
    }

    #[test]
    fn test_pitch_class_from_frequency_zero() {
        assert!(PitchClass::from_frequency(0.0).is_none());
        assert!(PitchClass::from_frequency(-10.0).is_none());
    }

    #[test]
    fn test_pitch_class_names() {
        assert_eq!(PitchClass::C.name(), "C");
        assert_eq!(PitchClass::CSharp.name(), "C#");
        assert_eq!(PitchClass::B.name(), "B");
    }

    #[test]
    fn test_pitch_class_index_range() {
        for midi in 0u8..12 {
            let pc = PitchClass::from_midi(midi);
            assert_eq!(pc.index(), midi);
        }
    }

    // ── PitchObservation ─────────────────────────────────────────────────

    #[test]
    fn test_voiced_observation() {
        let obs = PitchObservation::voiced(440.0, 0.9);
        assert!(obs.is_voiced());
        assert_eq!(obs.pitch_class, Some(PitchClass::A));
        assert!(obs.octave.is_some());
    }

    #[test]
    fn test_unvoiced_observation() {
        let obs = PitchObservation::unvoiced();
        assert!(!obs.is_voiced());
        assert!(obs.pitch_class.is_none());
    }

    #[test]
    fn test_voiced_low_confidence_not_voiced() {
        let obs = PitchObservation::voiced(440.0, 0.1);
        assert!(!obs.is_voiced()); // confidence < 0.3
    }

    // ── PitchHistory ─────────────────────────────────────────────────────

    #[test]
    fn test_history_capacity_eviction() {
        let mut hist = PitchHistory::new(3);
        for _ in 0..5 {
            hist.push(PitchObservation::voiced(440.0, 0.9));
        }
        assert_eq!(hist.len(), 3);
    }

    #[test]
    fn test_history_mean_f0() {
        let mut hist = PitchHistory::new(10);
        hist.push(PitchObservation::voiced(400.0, 0.9));
        hist.push(PitchObservation::voiced(500.0, 0.9));
        let mean = hist.mean_f0();
        assert!((mean - 450.0).abs() < 1.0);
    }

    #[test]
    fn test_history_voicing_rate() {
        let mut hist = PitchHistory::new(10);
        hist.push(PitchObservation::voiced(440.0, 0.9));
        hist.push(PitchObservation::unvoiced());
        assert!((hist.voicing_rate() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_dominant_pitch_class() {
        let mut hist = PitchHistory::new(10);
        for _ in 0..5 {
            hist.push(PitchObservation::voiced(440.0, 0.9)); // A
        }
        hist.push(PitchObservation::voiced(261.63, 0.9)); // C
        assert_eq!(hist.dominant_pitch_class(), Some(PitchClass::A));
    }

    // ── PitchTracker ──────────────────────────────────────────────────────

    #[test]
    fn test_tracker_default_construction() {
        let tracker = PitchTracker::default();
        assert_eq!(tracker.sample_rate, 44100.0);
        assert_eq!(tracker.frame_size, 2048);
        assert!(tracker.history.is_empty());
    }

    #[test]
    fn test_tracker_silence_unvoiced() {
        let mut tracker = PitchTracker::default();
        let frame = vec![0.0_f32; 2048];
        let obs = tracker.process_frame(&frame);
        assert!(!obs.is_voiced());
    }

    #[test]
    fn test_tracker_reset_clears_history() {
        let mut tracker = PitchTracker::default();
        tracker.process_frame(&vec![0.0_f32; 2048]);
        assert_eq!(tracker.history().len(), 1);
        tracker.reset();
        assert!(tracker.history().is_empty());
    }
}
