#![allow(dead_code)]

//! Chromagram computation and analysis for pitch-class profiling.
//!
//! Computes 12-bin chroma vectors (C, C#, D, ..., B) from audio, useful for
//! chord recognition, key detection, and harmonic similarity analysis.

/// Number of pitch classes (semitones per octave).
const CHROMA_BINS: usize = 12;

/// Standard tuning reference frequency for A4.
const A4_FREQ: f64 = 440.0;

/// MIDI note number for A4.
const A4_MIDI: f64 = 69.0;

/// Pitch class names.
const PITCH_NAMES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

/// A single chroma vector (12 pitch class energies).
#[derive(Debug, Clone)]
pub struct ChromaVector {
    /// Energy per pitch class `[C, C#, D, D#, E, F, F#, G, G#, A, A#, B]`.
    pub bins: [f64; CHROMA_BINS],
}

impl ChromaVector {
    /// Create a zero chroma vector.
    #[must_use]
    pub fn zero() -> Self {
        Self {
            bins: [0.0; CHROMA_BINS],
        }
    }

    /// Create a chroma vector from a 12-element slice.
    ///
    /// # Panics
    ///
    /// Panics if the slice length is not 12.
    #[must_use]
    pub fn from_slice(data: &[f64]) -> Self {
        assert_eq!(data.len(), CHROMA_BINS, "Chroma vector must have 12 bins");
        let mut bins = [0.0; CHROMA_BINS];
        bins.copy_from_slice(data);
        Self { bins }
    }

    /// Return the index of the strongest pitch class.
    #[must_use]
    pub fn dominant_pitch_class(&self) -> usize {
        self.bins
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i)
    }

    /// Return the name of the dominant pitch class.
    #[must_use]
    pub fn dominant_pitch_name(&self) -> &'static str {
        PITCH_NAMES[self.dominant_pitch_class()]
    }

    /// Normalize the vector so it sums to 1.0 (or remains zero if all zero).
    #[must_use]
    pub fn normalized(&self) -> Self {
        let sum: f64 = self.bins.iter().sum();
        if sum < 1e-12 {
            return self.clone();
        }
        let mut bins = [0.0; CHROMA_BINS];
        for (i, &v) in self.bins.iter().enumerate() {
            bins[i] = v / sum;
        }
        Self { bins }
    }

    /// Compute cosine similarity with another chroma vector.
    #[must_use]
    pub fn cosine_similarity(&self, other: &Self) -> f64 {
        let dot: f64 = self
            .bins
            .iter()
            .zip(other.bins.iter())
            .map(|(a, b)| a * b)
            .sum();
        let mag_a: f64 = self.bins.iter().map(|v| v * v).sum::<f64>().sqrt();
        let mag_b: f64 = other.bins.iter().map(|v| v * v).sum::<f64>().sqrt();
        if mag_a < 1e-12 || mag_b < 1e-12 {
            return 0.0;
        }
        dot / (mag_a * mag_b)
    }

    /// Circular shift the chroma vector by `n` semitones (transpose).
    #[must_use]
    pub fn transposed(&self, semitones: i32) -> Self {
        let mut bins = [0.0; CHROMA_BINS];
        for i in 0..CHROMA_BINS {
            let target = ((i as i32 + semitones).rem_euclid(CHROMA_BINS as i32)) as usize;
            bins[target] = self.bins[i];
        }
        Self { bins }
    }
}

/// Configuration for chromagram computation.
#[derive(Debug, Clone)]
pub struct ChromagramConfig {
    /// Sample rate in Hz.
    pub sample_rate: f32,
    /// Window size in samples.
    pub window_size: usize,
    /// Hop size in samples.
    pub hop_size: usize,
    /// Tuning reference for A4 in Hz.
    pub tuning_ref: f64,
    /// Minimum frequency to consider.
    pub min_freq: f64,
    /// Maximum frequency to consider.
    pub max_freq: f64,
}

impl Default for ChromagramConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100.0,
            window_size: 4096,
            hop_size: 512,
            tuning_ref: A4_FREQ,
            min_freq: 65.0,   // C2
            max_freq: 2093.0, // C7
        }
    }
}

/// Chromagram analyzer.
#[derive(Debug)]
pub struct ChromagramAnalyzer {
    config: ChromagramConfig,
}

impl ChromagramAnalyzer {
    /// Create a new chromagram analyzer.
    #[must_use]
    pub fn new(config: ChromagramConfig) -> Self {
        Self { config }
    }

    /// Create an analyzer with default config and specified sample rate.
    #[must_use]
    pub fn with_sample_rate(sample_rate: f32) -> Self {
        Self::new(ChromagramConfig {
            sample_rate,
            ..ChromagramConfig::default()
        })
    }

    /// Convert a frequency in Hz to the nearest pitch class index (0-11).
    #[must_use]
    pub fn freq_to_chroma(&self, freq: f64) -> usize {
        if freq <= 0.0 {
            return 0;
        }
        let midi = 12.0 * (freq / self.config.tuning_ref).log2() + A4_MIDI;
        let chroma = midi.round() as i64 % 12;
        if chroma < 0 {
            (chroma + 12) as usize
        } else {
            chroma as usize
        }
    }

    /// Compute a chromagram from audio samples.
    ///
    /// Returns a sequence of chroma vectors, one per hop.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn compute(&self, samples: &[f32]) -> Vec<ChromaVector> {
        let win = self.config.window_size.max(1);
        let hop = self.config.hop_size.max(1);
        let sr = f64::from(self.config.sample_rate);

        if samples.len() < win {
            return vec![self.compute_frame(samples, sr)];
        }

        let n_frames = (samples.len() - win) / hop + 1;
        let mut chromagram = Vec::with_capacity(n_frames);

        for i in 0..n_frames {
            let start = i * hop;
            let end = start + win;
            let frame = &samples[start..end];
            chromagram.push(self.compute_frame(frame, sr));
        }
        chromagram
    }

    /// Compute a single chroma vector from a windowed frame.
    ///
    /// Uses a simplified energy-folding approach: compute per-bin energy from
    /// a DFT magnitude approximation, then fold into 12 chroma bins.
    #[allow(clippy::cast_precision_loss)]
    fn compute_frame(&self, frame: &[f32], sr: f64) -> ChromaVector {
        let n = frame.len();
        if n == 0 {
            return ChromaVector::zero();
        }

        // Compute magnitude spectrum (real-valued DFT approximation via Goertzel-like sums)
        let n_bins = n / 2 + 1;
        let mut chroma = [0.0f64; CHROMA_BINS];

        for k in 1..n_bins {
            let freq = k as f64 * sr / n as f64;
            if freq < self.config.min_freq || freq > self.config.max_freq {
                continue;
            }

            // Goertzel magnitude estimate for bin k.
            // s_cur and s_prev are the two-sample Goertzel state;
            // s_prev2 is the oldest value used in the final magnitude formula.
            let omega = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
            let coeff = 2.0 * omega.cos();
            let mut s_prev2 = 0.0f64;
            let mut s_prev = 0.0f64;
            for sample in frame {
                let s_cur = f64::from(*sample) + coeff * s_prev - s_prev2;
                s_prev2 = s_prev;
                s_prev = s_cur;
            }
            let magnitude = (s_prev * s_prev + s_prev2 * s_prev2 - coeff * s_prev * s_prev2)
                .abs()
                .sqrt();

            let pitch_class = self.freq_to_chroma(freq);
            chroma[pitch_class] += magnitude;
        }

        ChromaVector { bins: chroma }
    }

    /// Compute the mean chroma vector across all frames.
    #[must_use]
    pub fn mean_chroma(&self, samples: &[f32]) -> ChromaVector {
        let chromagram = self.compute(samples);
        if chromagram.is_empty() {
            return ChromaVector::zero();
        }
        let mut sum = [0.0f64; CHROMA_BINS];
        for cv in &chromagram {
            for (i, &v) in cv.bins.iter().enumerate() {
                sum[i] += v;
            }
        }
        #[allow(clippy::cast_precision_loss)]
        let n = chromagram.len() as f64;
        for v in &mut sum {
            *v /= n;
        }
        ChromaVector { bins: sum }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chroma_vector_zero() {
        let cv = ChromaVector::zero();
        for &v in &cv.bins {
            assert!((v - 0.0).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_chroma_vector_from_slice() {
        let data: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let cv = ChromaVector::from_slice(&data);
        assert_eq!(cv.bins[0], 0.0);
        assert_eq!(cv.bins[11], 11.0);
    }

    #[test]
    #[should_panic(expected = "Chroma vector must have 12 bins")]
    fn test_chroma_vector_from_slice_wrong_size() {
        let _ = ChromaVector::from_slice(&[1.0, 2.0]);
    }

    #[test]
    fn test_dominant_pitch_class() {
        let mut cv = ChromaVector::zero();
        cv.bins[9] = 10.0; // A is index 9
        assert_eq!(cv.dominant_pitch_class(), 9);
        assert_eq!(cv.dominant_pitch_name(), "A");
    }

    #[test]
    fn test_normalized() {
        let mut cv = ChromaVector::zero();
        cv.bins[0] = 4.0;
        cv.bins[1] = 6.0;
        let norm = cv.normalized();
        let sum: f64 = norm.bins.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_normalized_zero() {
        let cv = ChromaVector::zero();
        let norm = cv.normalized();
        let sum: f64 = norm.bins.iter().sum();
        assert!((sum - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let mut cv = ChromaVector::zero();
        cv.bins[0] = 1.0;
        cv.bins[4] = 1.0;
        let sim = cv.cosine_similarity(&cv);
        assert!((sim - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let mut a = ChromaVector::zero();
        a.bins[0] = 1.0;
        let mut b = ChromaVector::zero();
        b.bins[6] = 1.0;
        let sim = a.cosine_similarity(&b);
        assert!(sim.abs() < 1e-9);
    }

    #[test]
    fn test_transposed() {
        let mut cv = ChromaVector::zero();
        cv.bins[0] = 1.0; // C
        let transposed = cv.transposed(4); // C -> E
        assert!((transposed.bins[4] - 1.0).abs() < f64::EPSILON);
        assert!((transposed.bins[0] - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_transposed_negative() {
        let mut cv = ChromaVector::zero();
        cv.bins[2] = 1.0; // D
        let transposed = cv.transposed(-3); // D -> B
        assert!((transposed.bins[11] - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_freq_to_chroma_a440() {
        let analyzer = ChromagramAnalyzer::with_sample_rate(44100.0);
        let chroma = analyzer.freq_to_chroma(440.0);
        assert_eq!(chroma, 9); // A = index 9
    }

    #[test]
    fn test_freq_to_chroma_c261() {
        let analyzer = ChromagramAnalyzer::with_sample_rate(44100.0);
        let chroma = analyzer.freq_to_chroma(261.63); // Middle C
        assert_eq!(chroma, 0); // C = index 0
    }

    #[test]
    fn test_compute_silence() {
        let analyzer = ChromagramAnalyzer::with_sample_rate(44100.0);
        let silence = vec![0.0f32; 44100];
        let chromagram = analyzer.compute(&silence);
        assert!(!chromagram.is_empty());
        for cv in &chromagram {
            for &v in &cv.bins {
                assert!(v.abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_compute_short_signal() {
        let analyzer = ChromagramAnalyzer::with_sample_rate(8000.0);
        let signal = vec![0.5f32; 100];
        let chromagram = analyzer.compute(&signal);
        assert!(!chromagram.is_empty());
    }

    #[test]
    fn test_mean_chroma() {
        let analyzer = ChromagramAnalyzer::with_sample_rate(44100.0);
        let silence = vec![0.0f32; 44100];
        let mean = analyzer.mean_chroma(&silence);
        for &v in &mean.bins {
            assert!(v.abs() < 1e-6);
        }
    }

    #[test]
    fn test_config_default() {
        let cfg = ChromagramConfig::default();
        assert!((cfg.sample_rate - 44100.0).abs() < f32::EPSILON);
        assert_eq!(cfg.window_size, 4096);
        assert_eq!(cfg.hop_size, 512);
    }

    /// Synthesise one second of a 440 Hz sine wave, compute the chromagram,
    /// and verify that the mean chroma vector's dominant pitch class is A (index 9).
    ///
    /// Pitch class mapping (C = 0 convention):
    /// C=0, C#=1, D=2, D#=3, E=4, F=5, F#=6, G=7, G#=8, **A=9**, A#=10, B=11
    #[test]
    fn test_chromagram_a4_440hz_pitch_class_a() {
        let sample_rate = 44100_u32;
        let freq = 440.0_f32; // A4
        let samples: Vec<f32> = (0..sample_rate)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin())
            .collect();

        let analyzer = ChromagramAnalyzer::with_sample_rate(sample_rate as f32);
        let mean = analyzer.mean_chroma(&samples);
        let dominant = mean.dominant_pitch_class();
        assert_eq!(
            dominant, 9,
            "440 Hz sine wave should produce dominant pitch class A (9), got {dominant}"
        );
    }
}
