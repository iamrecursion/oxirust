//! Audio segmentation: automatic speech/music/silence boundary detection.
//!
//! Uses a combination of features to classify each frame and detect
//! transitions between content types:
//! - Zero-crossing rate (high for speech, low for music, very low for silence)
//! - Spectral flux (high for music/speech, low for silence)
//! - RMS energy (near zero for silence)
//! - Spectral flatness (high for noise/speech, low for tonal music)
//!
//! Boundaries are found by sliding a novelty function over the frame-level
//! class labels and detecting peaks.

use crate::spectral::{SpectralAnalyzer, SpectralFeatures};
use crate::{compute_rms, zero_crossing_rate, AnalysisConfig, AnalysisError, Result};

/// Content type of an audio segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentType {
    /// Silence or very low-level audio
    Silence,
    /// Human speech (non-singing)
    Speech,
    /// Music (instrumental or singing)
    Music,
    /// Mixed content (speech over music, etc.)
    Mixed,
    /// Unknown / ambiguous
    Unknown,
}

impl ContentType {
    fn as_str(self) -> &'static str {
        match self {
            Self::Silence => "silence",
            Self::Speech => "speech",
            Self::Music => "music",
            Self::Mixed => "mixed",
            Self::Unknown => "unknown",
        }
    }
}

impl std::fmt::Display for ContentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A single detected audio segment.
#[derive(Debug, Clone)]
pub struct AudioSegment {
    /// Start time in seconds
    pub start_time: f32,
    /// End time in seconds
    pub end_time: f32,
    /// Content type of this segment
    pub content_type: ContentType,
    /// Confidence in the classification (0.0–1.0)
    pub confidence: f32,
}

impl AudioSegment {
    /// Duration of this segment in seconds.
    #[must_use]
    pub fn duration(&self) -> f32 {
        self.end_time - self.start_time
    }
}

/// Result of segmentation analysis.
#[derive(Debug, Clone)]
pub struct SegmentationResult {
    /// List of detected segments in chronological order
    pub segments: Vec<AudioSegment>,
    /// Frame-level content type labels
    pub frame_labels: Vec<ContentType>,
    /// Frame-level confidence scores
    pub frame_confidences: Vec<f32>,
    /// Novelty function values used to detect boundaries
    pub novelty: Vec<f32>,
}

/// Audio segmentation analyzer.
pub struct Segmenter {
    config: AnalysisConfig,
    spectral: SpectralAnalyzer,
    /// Minimum segment duration in seconds (avoids over-segmentation)
    min_segment_duration: f32,
    /// Novelty threshold for boundary detection (relative to max novelty)
    novelty_threshold: f32,
}

impl Segmenter {
    /// Create a new segmenter with default parameters.
    #[must_use]
    pub fn new(config: AnalysisConfig) -> Self {
        let spectral = SpectralAnalyzer::new(config.clone());
        Self {
            config,
            spectral,
            min_segment_duration: 0.5,
            novelty_threshold: 0.35,
        }
    }

    /// Set the minimum segment duration in seconds.
    #[must_use]
    pub fn with_min_segment_duration(mut self, duration: f32) -> Self {
        self.min_segment_duration = duration.max(0.0);
        self
    }

    /// Set novelty threshold (0.0–1.0) relative to maximum novelty.
    #[must_use]
    pub fn with_novelty_threshold(mut self, threshold: f32) -> Self {
        self.novelty_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Segment audio into speech/music/silence regions.
    pub fn segment(&self, samples: &[f32], sample_rate: f32) -> Result<SegmentationResult> {
        let fft_size = self.config.fft_size;
        let hop = self.config.hop_size;

        if samples.len() < fft_size {
            return Err(AnalysisError::InsufficientSamples {
                needed: fft_size,
                got: samples.len(),
            });
        }

        // ── Step 1: extract frame-level features ────────────────────────────
        let num_frames = (samples.len() - fft_size) / hop + 1;

        let mut frame_rms = Vec::with_capacity(num_frames);
        let mut frame_zcr = Vec::with_capacity(num_frames);
        let mut frame_flatness = Vec::with_capacity(num_frames);
        let mut frame_flux = Vec::with_capacity(num_frames);

        let mut prev_mag: Option<Vec<f32>> = None;

        for idx in 0..num_frames {
            let start = idx * hop;
            let end = (start + fft_size).min(samples.len());
            if end - start < fft_size {
                break;
            }
            let frame = &samples[start..end];

            let rms = compute_rms(frame);
            let zcr = zero_crossing_rate(frame);
            let feats: SpectralFeatures = self.spectral.analyze_frame(frame, sample_rate)?;
            let flux = match &prev_mag {
                Some(pm) => crate::spectral::spectral_flux(&feats.magnitude_spectrum, pm),
                None => 0.0,
            };

            frame_rms.push(rms);
            frame_zcr.push(zcr);
            frame_flatness.push(feats.flatness);
            frame_flux.push(flux);

            prev_mag = Some(feats.magnitude_spectrum);
        }

        let n = frame_rms.len();
        if n == 0 {
            return Ok(SegmentationResult {
                segments: Vec::new(),
                frame_labels: Vec::new(),
                frame_confidences: Vec::new(),
                novelty: Vec::new(),
            });
        }

        // ── Step 2: classify each frame ──────────────────────────────────────
        let silence_threshold = 0.005_f32;
        let (mut frame_labels, mut frame_confidences) =
            (Vec::with_capacity(n), Vec::with_capacity(n));

        for i in 0..n {
            let (label, conf) = classify_frame(
                frame_rms[i],
                frame_zcr[i],
                frame_flatness[i],
                silence_threshold,
            );
            frame_labels.push(label);
            frame_confidences.push(conf);
        }

        // ── Step 3: novelty function (label change + flux) ───────────────────
        let mut novelty = vec![0.0_f32; n];
        for i in 1..n {
            let label_change = if frame_labels[i] != frame_labels[i - 1] {
                1.0_f32
            } else {
                0.0
            };
            let flux_norm = (frame_flux[i]
                / (frame_flux.iter().cloned().fold(f32::EPSILON, f32::max)))
            .min(1.0);
            novelty[i] = label_change * 0.6 + flux_norm * 0.4;
        }

        // ── Step 4: boundary detection via peak picking ──────────────────────
        let max_novelty = novelty.iter().cloned().fold(f32::EPSILON, f32::max);
        let threshold = max_novelty * self.novelty_threshold;
        let min_frames = ((self.min_segment_duration * sample_rate) / hop as f32).max(1.0) as usize;

        let mut boundaries = vec![0usize];
        let mut last_boundary = 0;

        for i in 1..n {
            if novelty[i] >= threshold && i - last_boundary >= min_frames {
                boundaries.push(i);
                last_boundary = i;
            }
        }
        boundaries.push(n);

        // ── Step 5: build segment list ───────────────────────────────────────
        let hop_duration = hop as f32 / sample_rate;
        let mut segments = Vec::new();

        for w in boundaries.windows(2) {
            let seg_start = w[0];
            let seg_end = w[1];

            // Majority vote for content type in this segment
            let (content_type, confidence) = majority_vote(
                &frame_labels[seg_start..seg_end],
                &frame_confidences[seg_start..seg_end],
            );

            segments.push(AudioSegment {
                start_time: seg_start as f32 * hop_duration,
                end_time: seg_end as f32 * hop_duration,
                content_type,
                confidence,
            });
        }

        Ok(SegmentationResult {
            segments,
            frame_labels,
            frame_confidences,
            novelty,
        })
    }
}

// ── internal helpers ─────────────────────────────────────────────────────────

/// Classify a single frame into silence/speech/music/unknown.
fn classify_frame(rms: f32, zcr: f32, flatness: f32, silence_threshold: f32) -> (ContentType, f32) {
    if rms < silence_threshold {
        return (ContentType::Silence, 0.9);
    }

    // Speech features: moderate-to-high ZCR, moderate flatness
    let speech_score = {
        let zcr_score = if (0.05..=0.45).contains(&zcr) {
            1.0_f32
        } else {
            0.3
        };
        let flat_score = if (0.1..=0.6).contains(&flatness) {
            1.0
        } else {
            0.4
        };
        (zcr_score + flat_score) / 2.0
    };

    // Music features: low ZCR, low flatness (tonal), or high ZCR + high energy
    let music_score = {
        let zcr_score = if zcr < 0.1 { 1.0_f32 } else { 0.3 };
        let flat_score = if flatness < 0.25 { 1.0 } else { 0.4 };
        (zcr_score + flat_score) / 2.0
    };

    if speech_score > music_score + 0.1 {
        (ContentType::Speech, speech_score.min(1.0))
    } else if music_score > speech_score + 0.1 {
        (ContentType::Music, music_score.min(1.0))
    } else if (speech_score - music_score).abs() < 0.15 && rms > silence_threshold * 3.0 {
        (ContentType::Mixed, 0.5)
    } else {
        (ContentType::Unknown, 0.3)
    }
}

/// Majority-vote content type for a segment, returning type and mean confidence.
fn majority_vote(labels: &[ContentType], confidences: &[f32]) -> (ContentType, f32) {
    if labels.is_empty() {
        return (ContentType::Unknown, 0.0);
    }

    let mut counts = [0usize; 5]; // Silence, Speech, Music, Mixed, Unknown
    let mut conf_sums = [0.0_f32; 5];

    for (&label, &conf) in labels.iter().zip(confidences.iter()) {
        let idx = match label {
            ContentType::Silence => 0,
            ContentType::Speech => 1,
            ContentType::Music => 2,
            ContentType::Mixed => 3,
            ContentType::Unknown => 4,
        };
        counts[idx] += 1;
        conf_sums[idx] += conf;
    }

    let best_idx = counts
        .iter()
        .enumerate()
        .max_by_key(|(_, &c)| c)
        .map(|(i, _)| i)
        .unwrap_or(4);

    let best_type = match best_idx {
        0 => ContentType::Silence,
        1 => ContentType::Speech,
        2 => ContentType::Music,
        3 => ContentType::Mixed,
        _ => ContentType::Unknown,
    };

    let mean_conf = if counts[best_idx] > 0 {
        conf_sums[best_idx] / counts[best_idx] as f32
    } else {
        0.0
    };

    (best_type, mean_conf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn make_silence(n: usize) -> Vec<f32> {
        vec![0.0; n]
    }

    fn make_sine(freq: f32, n: usize, sr: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / sr).sin() * 0.6)
            .collect()
    }

    fn make_noise(n: usize) -> Vec<f32> {
        // Deterministic pseudo-noise via LCG
        let mut x: u32 = 0xdeadbeef;
        (0..n)
            .map(|_| {
                x = x.wrapping_mul(1664525).wrapping_add(1013904223);
                (x as i32 as f32) / i32::MAX as f32 * 0.5
            })
            .collect()
    }

    #[test]
    fn test_segmenter_basic() {
        let config = AnalysisConfig::default();
        let segmenter = Segmenter::new(config);

        let sr = 44100.0;
        let mut samples = Vec::new();
        samples.extend(make_silence(22050)); // 0.5 s silence
        samples.extend(make_sine(440.0, 44100, sr)); // 1 s music
        samples.extend(make_noise(22050)); // 0.5 s noise/speech

        let result = segmenter.segment(&samples, sr);
        assert!(result.is_ok(), "Segmentation should succeed");
        let r = result.expect("should succeed");
        assert!(!r.segments.is_empty());
        // All segments should have valid durations
        for seg in &r.segments {
            assert!(seg.duration() >= 0.0);
        }
    }

    #[test]
    fn test_classify_silence() {
        let (ct, conf) = classify_frame(0.001, 0.0, 0.0, 0.005);
        assert_eq!(ct, ContentType::Silence);
        assert!(conf > 0.5);
    }

    #[test]
    fn test_classify_music() {
        // Clearly low ZCR, clearly low flatness → music (flatness < 0.1, zcr < 0.05)
        let (ct, _) = classify_frame(0.5, 0.02, 0.05, 0.005);
        assert_eq!(ct, ContentType::Music);
    }

    #[test]
    fn test_classify_speech() {
        // Moderate ZCR, moderate flatness → speech
        let (ct, _) = classify_frame(0.3, 0.25, 0.4, 0.005);
        assert_eq!(ct, ContentType::Speech);
    }

    #[test]
    fn test_majority_vote() {
        let labels = vec![ContentType::Music, ContentType::Music, ContentType::Speech];
        let confs = vec![0.9, 0.8, 0.6];
        let (ct, _) = majority_vote(&labels, &confs);
        assert_eq!(ct, ContentType::Music);
    }

    #[test]
    fn test_segment_durations_positive() {
        let config = AnalysisConfig::default();
        let segmenter = Segmenter::new(config);
        let sr = 44100.0;
        let samples = make_sine(440.0, 44100 * 2, sr);
        let result = segmenter.segment(&samples, sr).expect("should succeed");
        for seg in &result.segments {
            assert!(
                seg.end_time >= seg.start_time,
                "Segment end must be >= start: {} >= {}",
                seg.end_time,
                seg.start_time
            );
        }
    }

    #[test]
    fn test_insufficient_samples() {
        let config = AnalysisConfig::default();
        let segmenter = Segmenter::new(config);
        let result = segmenter.segment(&[0.1; 100], 44100.0);
        assert!(result.is_err());
    }
}
