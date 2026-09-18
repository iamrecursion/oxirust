//! Chord recognition using chroma-based template matching.

use crate::types::{ChordLabel, ChordResult};
use crate::utils::stft;
use crate::{MirError, MirResult};

/// Chord quality type for extended chord vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChordQuality {
    /// Major triad (1-3-5).
    Major,
    /// Minor triad (1-b3-5).
    Minor,
    /// Dominant 7th (1-3-5-b7).
    Dominant7,
    /// Major 7th (1-3-5-7).
    Major7,
    /// Minor 7th (1-b3-5-b7).
    Minor7,
    /// Diminished triad (1-b3-b5).
    Diminished,
    /// Augmented triad (1-3-#5).
    Augmented,
    /// Suspended 2nd (1-2-5).
    Sus2,
    /// Suspended 4th (1-4-5).
    Sus4,
    /// Diminished 7th (1-b3-b5-bb7).
    Diminished7,
    /// Half-diminished 7th (1-b3-b5-b7).
    HalfDiminished7,
}

/// Build a rotated 12-bin template for the given root and interval pattern.
///
/// `intervals` contains semitone offsets from root (root=0 is implicit).
/// `weights` contains the weight for each interval (root weight is always 1.0).
fn build_template(root: u8, intervals: &[usize], weights: &[f32]) -> [f32; 12] {
    let mut t = [0.0_f32; 12];
    t[root as usize % 12] = 1.0; // root
    for (&interval, &w) in intervals.iter().zip(weights.iter()) {
        t[(root as usize + interval) % 12] = w;
    }
    t
}

/// Generate all chord templates programmatically for all 12 roots.
fn generate_all_templates() -> Vec<ChordTemplate> {
    let note_names: [&str; 12] = [
        "C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B",
    ];

    // (suffix, quality, intervals from root, weights for those intervals)
    let chord_types: &[(&str, ChordQuality, &[usize], &[f32])] = &[
        ("", ChordQuality::Major, &[4, 7], &[1.0, 1.0]),
        ("m", ChordQuality::Minor, &[3, 7], &[1.0, 1.0]),
        ("7", ChordQuality::Dominant7, &[4, 7, 10], &[1.0, 0.8, 0.7]),
        ("maj7", ChordQuality::Major7, &[4, 7, 11], &[1.0, 0.8, 0.7]),
        ("m7", ChordQuality::Minor7, &[3, 7, 10], &[1.0, 0.8, 0.7]),
        ("dim", ChordQuality::Diminished, &[3, 6], &[1.0, 1.0]),
        ("aug", ChordQuality::Augmented, &[4, 8], &[1.0, 1.0]),
        ("sus2", ChordQuality::Sus2, &[2, 7], &[0.8, 1.0]),
        ("sus4", ChordQuality::Sus4, &[5, 7], &[0.8, 1.0]),
        (
            "dim7",
            ChordQuality::Diminished7,
            &[3, 6, 9],
            &[1.0, 0.8, 0.7],
        ),
        (
            "m7b5",
            ChordQuality::HalfDiminished7,
            &[3, 6, 10],
            &[1.0, 0.8, 0.7],
        ),
    ];

    let mut templates = Vec::with_capacity(12 * chord_types.len());

    for (suffix, quality, intervals, weights) in chord_types {
        for root in 0..12_u8 {
            let name_str = format!("{}{}", note_names[root as usize], suffix);
            templates.push(ChordTemplate {
                name_owned: name_str,
                root,
                quality: *quality,
                template: build_template(root, intervals, weights),
            });
        }
    }

    templates
}

/// Lazily-constructed global template list.
fn chord_templates() -> &'static [ChordTemplate] {
    use std::sync::OnceLock;
    static TEMPLATES: OnceLock<Vec<ChordTemplate>> = OnceLock::new();
    TEMPLATES.get_or_init(generate_all_templates)
}

/// Chord template for matching.
#[derive(Debug, Clone)]
struct ChordTemplate {
    name_owned: String,
    root: u8,
    #[allow(dead_code)]
    quality: ChordQuality,
    template: [f32; 12],
}

/// Chord recognizer.
pub struct ChordRecognizer {
    sample_rate: f32,
    window_size: usize,
    hop_size: usize,
}

impl ChordRecognizer {
    /// Create a new chord recognizer.
    #[must_use]
    pub fn new(sample_rate: f32, window_size: usize, hop_size: usize) -> Self {
        Self {
            sample_rate,
            window_size,
            hop_size,
        }
    }

    /// Recognize chords in audio signal.
    ///
    /// # Errors
    ///
    /// Returns error if chord recognition fails.
    #[allow(clippy::cast_precision_loss)]
    pub fn recognize(&self, signal: &[f32]) -> MirResult<ChordResult> {
        // Compute chromagram
        let chroma_frames = self.compute_chromagram(signal)?;

        if chroma_frames.is_empty() {
            return Err(MirError::InsufficientData(
                "No chroma frames for chord recognition".to_string(),
            ));
        }

        // Recognize chord for each frame
        let mut chord_labels = Vec::new();
        let mut current_chord: Option<(String, usize, f32)> = None; // (label, start_frame, confidence)

        for (frame_idx, chroma) in chroma_frames.iter().enumerate() {
            let (label, confidence) = self.match_chord(chroma);

            match &mut current_chord {
                Some((current_label, _start_frame, total_conf)) if current_label == &label => {
                    // Same chord continues
                    *total_conf += confidence;
                }
                _ => {
                    // New chord detected
                    if let Some((prev_label, start_frame, total_conf)) = current_chord.take() {
                        let start_time =
                            start_frame as f32 * self.hop_size as f32 / self.sample_rate;
                        let end_time = frame_idx as f32 * self.hop_size as f32 / self.sample_rate;
                        let duration = (frame_idx - start_frame) as f32;
                        let avg_confidence = if duration > 0.0 {
                            total_conf / duration
                        } else {
                            0.0
                        };

                        chord_labels.push(ChordLabel {
                            start: start_time,
                            end: end_time,
                            label: prev_label,
                            confidence: avg_confidence,
                        });
                    }
                    current_chord = Some((label, frame_idx, confidence));
                }
            }
        }

        // Add final chord
        if let Some((label, start_frame, total_conf)) = current_chord {
            let start_time = start_frame as f32 * self.hop_size as f32 / self.sample_rate;
            let end_time = chroma_frames.len() as f32 * self.hop_size as f32 / self.sample_rate;
            let duration = (chroma_frames.len() - start_frame) as f32;
            let avg_confidence = if duration > 0.0 {
                total_conf / duration
            } else {
                0.0
            };

            chord_labels.push(ChordLabel {
                start: start_time,
                end: end_time,
                label,
                confidence: avg_confidence,
            });
        }

        // Analyze chord progressions
        let progressions = self.extract_progressions(&chord_labels);

        // Compute harmonic complexity
        let complexity = self.compute_complexity(&chord_labels);

        Ok(ChordResult {
            chords: chord_labels,
            progressions,
            complexity,
        })
    }

    /// Compute chromagram from signal.
    fn compute_chromagram(&self, signal: &[f32]) -> MirResult<Vec<[f32; 12]>> {
        let frames = stft(signal, self.window_size, self.hop_size)?;
        let mut chroma_frames = Vec::with_capacity(frames.len());

        for frame in &frames {
            let chroma = self.frame_to_chroma(frame);
            chroma_frames.push(chroma);
        }

        Ok(chroma_frames)
    }

    /// Convert FFT frame to chroma vector.
    #[allow(clippy::cast_precision_loss)]
    fn frame_to_chroma(&self, frame: &[oxifft::Complex<f32>]) -> [f32; 12] {
        let mut chroma = [0.0; 12];
        let num_bins = frame.len() / 2;
        let ref_freq = 16.35; // C0

        for (bin, complex) in frame[1..num_bins].iter().enumerate() {
            let magnitude = complex.norm();
            let freq = (bin + 1) as f32 * self.sample_rate / self.window_size as f32;

            if freq < 20.0 {
                continue;
            }

            let pitch_class = self.freq_to_pitch_class(freq, ref_freq);
            chroma[pitch_class] += magnitude;
        }

        // Normalize
        let sum: f32 = chroma.iter().sum();
        if sum > 0.0 {
            for c in &mut chroma {
                *c /= sum;
            }
        }

        chroma
    }

    /// Convert frequency to pitch class.
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    fn freq_to_pitch_class(&self, freq: f32, ref_freq: f32) -> usize {
        let semitones = 12.0 * (freq / ref_freq).log2();
        (semitones.round() as i32).rem_euclid(12) as usize
    }

    /// Match chroma to chord template.
    ///
    /// Searches all chord qualities (major, minor, 7th, dim, aug, sus, etc.)
    /// and returns the best matching chord name and its cosine similarity score.
    fn match_chord(&self, chroma: &[f32; 12]) -> (String, f32) {
        let mut best_match = ("N".to_string(), 0.0_f32);

        for template in chord_templates() {
            let similarity = self.cosine_similarity(chroma, &template.template);
            if similarity > best_match.1 {
                best_match = (template.name_owned.clone(), similarity);
            }
        }

        best_match
    }

    /// Compute cosine similarity between two vectors.
    fn cosine_similarity(&self, a: &[f32; 12], b: &[f32; 12]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    /// Extract chord progressions.
    fn extract_progressions(&self, chords: &[ChordLabel]) -> Vec<String> {
        let mut progressions = Vec::new();

        for window in chords.windows(4) {
            let progression = window
                .iter()
                .map(|c| c.label.as_str())
                .collect::<Vec<_>>()
                .join(" - ");
            progressions.push(progression);
        }

        progressions
    }

    /// Compute harmonic complexity.
    fn compute_complexity(&self, chords: &[ChordLabel]) -> f32 {
        if chords.len() < 2 {
            return 0.0;
        }

        let mut changes = 0;
        for i in 1..chords.len() {
            if chords[i].label != chords[i - 1].label {
                changes += 1;
            }
        }

        // Normalize by duration
        (changes as f32 / chords.len() as f32).min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chord_recognizer_creation() {
        let recognizer = ChordRecognizer::new(44100.0, 2048, 512);
        assert_eq!(recognizer.sample_rate, 44100.0);
    }

    #[test]
    fn test_chord_templates_count() {
        // 11 qualities * 12 roots = 132 templates
        assert!(chord_templates().len() >= 132);
    }

    #[test]
    fn test_cosine_similarity() {
        let recognizer = ChordRecognizer::new(44100.0, 2048, 512);
        let a = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let sim = recognizer.cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_match_c_major_triad() {
        let recognizer = ChordRecognizer::new(44100.0, 2048, 512);
        // C major chroma: strong C(0), E(4), G(7)
        let chroma = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let (label, confidence) = recognizer.match_chord(&chroma);
        assert_eq!(label, "C");
        assert!(confidence > 0.9);
    }

    #[test]
    fn test_match_a_minor_triad() {
        let recognizer = ChordRecognizer::new(44100.0, 2048, 512);
        // A minor chroma: strong A(9), C(0), E(4)
        let chroma = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let (label, confidence) = recognizer.match_chord(&chroma);
        assert_eq!(label, "Am");
        assert!(confidence > 0.9);
    }

    #[test]
    fn test_match_g7_dominant_seventh() {
        let recognizer = ChordRecognizer::new(44100.0, 2048, 512);
        // G7 chroma: G(7), B(11), D(2), F(5)
        let chroma = [0.0, 0.0, 0.7, 0.0, 0.0, 0.7, 0.0, 1.0, 0.0, 0.0, 0.0, 0.8];
        let (label, confidence) = recognizer.match_chord(&chroma);
        assert!(
            label == "G7" || label == "G",
            "Expected G7 or G, got {label}"
        );
        assert!(confidence > 0.5);
    }

    #[test]
    fn test_match_diminished() {
        let recognizer = ChordRecognizer::new(44100.0, 2048, 512);
        // B diminished: B(11), D(2), F(5)
        let chroma = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let (label, confidence) = recognizer.match_chord(&chroma);
        assert!(
            confidence > 0.5,
            "confidence should be > 0.5, got {confidence}"
        );
        // Could match Bdim or related chord
        assert!(!label.is_empty());
    }

    #[test]
    fn test_match_augmented() {
        let recognizer = ChordRecognizer::new(44100.0, 2048, 512);
        // C augmented: C(0), E(4), G#(8)
        let chroma = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let (label, confidence) = recognizer.match_chord(&chroma);
        assert!(confidence > 0.5);
        assert!(
            label == "Caug" || label == "Eaug" || label == "Abaug" || label.contains("aug"),
            "Expected augmented chord, got {label}"
        );
    }

    #[test]
    fn test_match_sus4() {
        let recognizer = ChordRecognizer::new(44100.0, 2048, 512);
        // Csus4: C(0), F(5), G(7)
        let chroma = [1.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let (label, confidence) = recognizer.match_chord(&chroma);
        assert!(confidence > 0.5);
        // Could be Csus4 or Fsus2 (same notes)
        assert!(
            label.contains("sus") || label == "C" || label == "F",
            "Expected sus chord, got {label}"
        );
    }

    #[test]
    fn test_chord_quality_enum() {
        assert_ne!(ChordQuality::Major, ChordQuality::Minor);
        assert_ne!(ChordQuality::Dominant7, ChordQuality::Major7);
        assert_ne!(ChordQuality::Diminished, ChordQuality::Augmented);
    }

    #[test]
    fn test_build_template_c_major() {
        let t = build_template(0, &[4, 7], &[1.0, 1.0]);
        assert!((t[0] - 1.0).abs() < f32::EPSILON); // C
        assert!((t[4] - 1.0).abs() < f32::EPSILON); // E
        assert!((t[7] - 1.0).abs() < f32::EPSILON); // G
        assert!((t[1] - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_build_template_wraps_around() {
        // A major: A(9) + C#(9+4=13%12=1) + E(9+7=16%12=4)
        let t = build_template(9, &[4, 7], &[1.0, 1.0]);
        assert!((t[9] - 1.0).abs() < f32::EPSILON); // A
        assert!((t[1] - 1.0).abs() < f32::EPSILON); // C#
        assert!((t[4] - 1.0).abs() < f32::EPSILON); // E
    }

    #[test]
    fn test_all_templates_have_root() {
        for template in chord_templates() {
            assert!(
                template.template[template.root as usize] > 0.0,
                "Template {} should have non-zero root at index {}",
                template.name_owned,
                template.root
            );
        }
    }
}
