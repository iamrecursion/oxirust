//! Multi-speaker separation and diarization.
//!
//! Speaker diarization answers "who spoke when". This module implements a
//! feature-based approach:
//!
//! 1. Segment audio into short overlapping frames.
//! 2. Extract a speaker feature vector (F0, formants, spectral centroid, ZCR)
//!    per frame.
//! 3. Cluster frames using an online agglomerative approach into up to
//!    `max_speakers` clusters.
//! 4. Merge short segments (< `min_segment_duration`) into surrounding segments.
//! 5. Return a timeline of speaker segments.

use crate::{AnalysisConfig, AnalysisError, Result};

/// A single speaker segment in the diarization result.
#[derive(Debug, Clone)]
pub struct SpeakerSegment {
    /// Speaker label (0-indexed integer, stable within a recording)
    pub speaker_id: usize,
    /// Start time in seconds
    pub start_time: f32,
    /// End time in seconds
    pub end_time: f32,
    /// Confidence in speaker assignment (0.0–1.0)
    pub confidence: f32,
}

impl SpeakerSegment {
    /// Duration of this segment in seconds.
    #[must_use]
    pub fn duration(&self) -> f32 {
        self.end_time - self.start_time
    }
}

/// Full diarization result.
#[derive(Debug, Clone)]
pub struct DiarizationResult {
    /// Ordered list of speaker segments
    pub segments: Vec<SpeakerSegment>,
    /// Estimated number of distinct speakers found
    pub num_speakers: usize,
    /// Per-frame speaker assignments (frame index → speaker_id)
    pub frame_assignments: Vec<usize>,
}

impl DiarizationResult {
    /// Return all segments for a specific speaker.
    #[must_use]
    pub fn segments_for_speaker(&self, speaker_id: usize) -> Vec<&SpeakerSegment> {
        self.segments
            .iter()
            .filter(|s| s.speaker_id == speaker_id)
            .collect()
    }

    /// Total speaking time in seconds for a specific speaker.
    #[must_use]
    pub fn speaking_time(&self, speaker_id: usize) -> f32 {
        self.segments_for_speaker(speaker_id)
            .iter()
            .map(|s| s.duration())
            .sum()
    }
}

/// Speaker diarization engine.
pub struct SpeakerDiarizer {
    config: AnalysisConfig,
    /// Maximum number of distinct speakers to discover
    max_speakers: usize,
    /// Minimum segment duration in seconds
    min_segment_duration: f32,
    /// Feature distance threshold for merging two clusters
    merge_threshold: f32,
}

impl SpeakerDiarizer {
    /// Create a new diarizer with default settings.
    #[must_use]
    pub fn new(config: AnalysisConfig) -> Self {
        Self {
            config,
            max_speakers: 6,
            min_segment_duration: 0.3,
            merge_threshold: 0.25,
        }
    }

    /// Set maximum number of speakers to detect.
    #[must_use]
    pub fn with_max_speakers(mut self, n: usize) -> Self {
        self.max_speakers = n.max(1);
        self
    }

    /// Set minimum segment duration in seconds.
    #[must_use]
    pub fn with_min_segment_duration(mut self, duration: f32) -> Self {
        self.min_segment_duration = duration.max(0.0);
        self
    }

    /// Set distance threshold for cluster merging (lower = fewer speakers).
    #[must_use]
    pub fn with_merge_threshold(mut self, threshold: f32) -> Self {
        self.merge_threshold = threshold.clamp(0.05, 1.0);
        self
    }

    /// Perform diarization on audio samples.
    pub fn diarize(&self, samples: &[f32], sample_rate: f32) -> Result<DiarizationResult> {
        if samples.len() < self.config.fft_size {
            return Err(AnalysisError::InsufficientSamples {
                needed: self.config.fft_size,
                got: samples.len(),
            });
        }

        // ── Step 1: extract frame features ─────────────────────────────────
        let frame_features = self.extract_frame_features(samples, sample_rate)?;

        if frame_features.is_empty() {
            return Ok(DiarizationResult {
                segments: Vec::new(),
                num_speakers: 0,
                frame_assignments: Vec::new(),
            });
        }

        // ── Step 2: initial cluster per frame, then agglomeratively merge ──
        let n_frames = frame_features.len();
        let mut assignments: Vec<usize> = (0..n_frames).collect();
        let mut centroids: Vec<SpeakerFeature> = frame_features.clone();

        // Merge closest clusters until we reach max_speakers or no pairs are close enough
        loop {
            let n_clusters = {
                let mut ids: Vec<usize> = assignments.clone();
                ids.sort_unstable();
                ids.dedup();
                ids.len()
            };

            if n_clusters <= 1 {
                break;
            }

            // Find the two closest cluster centroids
            let cluster_ids: Vec<usize> = {
                let mut ids: Vec<usize> = assignments.clone();
                ids.sort_unstable();
                ids.dedup();
                ids
            };

            let mut min_dist = f32::INFINITY;
            let mut merge_a = 0;
            let mut merge_b = 0;

            for i in 0..cluster_ids.len() {
                for j in (i + 1)..cluster_ids.len() {
                    let ca = cluster_ids[i];
                    let cb = cluster_ids[j];
                    let dist = feature_distance(&centroids[ca], &centroids[cb]);
                    if dist < min_dist {
                        min_dist = dist;
                        merge_a = ca;
                        merge_b = cb;
                    }
                }
            }

            // Stop merging if minimum distance is above threshold and we're at or below max_speakers
            if min_dist > self.merge_threshold && n_clusters <= self.max_speakers {
                break;
            }

            // Merge cluster merge_b into merge_a
            let count_a = assignments.iter().filter(|&&x| x == merge_a).count() as f32;
            let count_b = assignments.iter().filter(|&&x| x == merge_b).count() as f32;
            let total = count_a + count_b;

            if total > 0.0 && merge_a < centroids.len() && merge_b < centroids.len() {
                let new_centroid = SpeakerFeature {
                    mean_f0: (centroids[merge_a].mean_f0 * count_a
                        + centroids[merge_b].mean_f0 * count_b)
                        / total,
                    spectral_centroid: (centroids[merge_a].spectral_centroid * count_a
                        + centroids[merge_b].spectral_centroid * count_b)
                        / total,
                    zcr: (centroids[merge_a].zcr * count_a + centroids[merge_b].zcr * count_b)
                        / total,
                    rms: (centroids[merge_a].rms * count_a + centroids[merge_b].rms * count_b)
                        / total,
                    f1: (centroids[merge_a].f1 * count_a + centroids[merge_b].f1 * count_b) / total,
                };
                centroids[merge_a] = new_centroid;
            }

            for a in &mut assignments {
                if *a == merge_b {
                    *a = merge_a;
                }
            }
        }

        // ── Step 3: remap cluster IDs to 0..N-1 ────────────────────────────
        let mut id_map: Vec<Option<usize>> = vec![None; n_frames + 1];
        let mut next_id = 0usize;
        let mut remapped = vec![0usize; n_frames];

        for (i, &raw_id) in assignments.iter().enumerate() {
            let mapped = if raw_id < id_map.len() {
                if let Some(m) = id_map[raw_id] {
                    m
                } else {
                    id_map[raw_id] = Some(next_id);
                    let m = next_id;
                    next_id += 1;
                    m
                }
            } else {
                // Fallback
                0
            };
            remapped[i] = mapped;
        }

        let num_speakers = next_id;

        // ── Step 4: build segments from consecutive same-speaker frames ─────
        let hop = self.config.hop_size;
        let hop_dur = hop as f32 / sample_rate;
        let min_frames = ((self.min_segment_duration / hop_dur).ceil() as usize).max(1);

        let mut segments: Vec<SpeakerSegment> = Vec::new();

        if !remapped.is_empty() {
            let mut seg_start = 0usize;
            let mut seg_speaker = remapped[0];

            for i in 1..=remapped.len() {
                let cur_speaker = if i < remapped.len() { remapped[i] } else { usize::MAX };

                if cur_speaker != seg_speaker {
                    let seg_len = i - seg_start;
                    if seg_len >= min_frames {
                        // Confidence: how homogeneous is this cluster
                        let conf = compute_cluster_confidence(
                            &frame_features[seg_start..i],
                            &centroids,
                            assignments[seg_start.min(assignments.len() - 1)],
                        );
                        segments.push(SpeakerSegment {
                            speaker_id: seg_speaker,
                            start_time: seg_start as f32 * hop_dur,
                            end_time: i as f32 * hop_dur,
                            confidence: conf,
                        });
                    } else if !segments.is_empty() {
                        // Absorb into previous segment
                        if let Some(last) = segments.last_mut() {
                            last.end_time = i as f32 * hop_dur;
                        }
                    }
                    seg_start = i;
                    seg_speaker = cur_speaker;
                }
            }
        }

        Ok(DiarizationResult {
            segments,
            num_speakers,
            frame_assignments: remapped,
        })
    }

    /// Extract a speaker feature vector for each hop-sized frame.
    fn extract_frame_features(
        &self,
        samples: &[f32],
        sample_rate: f32,
    ) -> Result<Vec<SpeakerFeature>> {
        let fft_size = self.config.fft_size;
        let hop = self.config.hop_size;
        let n_frames = (samples.len() - fft_size) / hop + 1;

        let mut features = Vec::with_capacity(n_frames);

        for idx in 0..n_frames {
            let start = idx * hop;
            let end = (start + fft_size).min(samples.len());
            if end - start < fft_size {
                break;
            }
            let frame = &samples[start..end];
            let feat = extract_speaker_feature(frame, sample_rate);
            features.push(feat);
        }

        Ok(features)
    }
}

// ── feature types and distance ──────────────────────────────────────────────

/// Compact speaker feature vector.
#[derive(Debug, Clone)]
struct SpeakerFeature {
    /// Mean fundamental frequency (or 0 if unvoiced)
    mean_f0: f32,
    /// Spectral centroid in Hz
    spectral_centroid: f32,
    /// Zero-crossing rate
    zcr: f32,
    /// RMS energy
    rms: f32,
    /// First formant estimate (from simple LPC approximation)
    f1: f32,
}

/// Compute Euclidean distance between two feature vectors, normalized.
fn feature_distance(a: &SpeakerFeature, b: &SpeakerFeature) -> f32 {
    let f0_d = ((a.mean_f0 - b.mean_f0) / 300.0_f32.max(1.0)).powi(2);
    let cent_d = ((a.spectral_centroid - b.spectral_centroid) / 4000.0_f32.max(1.0)).powi(2);
    let zcr_d = (a.zcr - b.zcr).powi(2);
    let f1_d = ((a.f1 - b.f1) / 2000.0_f32.max(1.0)).powi(2);

    (f0_d * 0.35 + cent_d * 0.30 + zcr_d * 0.15 + f1_d * 0.20).sqrt()
}

/// Extract speaker feature from a single frame.
fn extract_speaker_feature(frame: &[f32], sample_rate: f32) -> SpeakerFeature {
    let rms = {
        let sq: f32 = frame.iter().map(|&x| x * x).sum();
        (sq / frame.len() as f32).sqrt()
    };

    // Zero-crossing rate
    let zcr = crate::zero_crossing_rate(frame);

    // Spectral centroid via magnitude spectrum
    let n = frame.len().next_power_of_two();
    let window: Vec<f32> = (0..frame.len())
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (frame.len() - 1) as f32).cos()))
        .collect();

    let buffer: Vec<oxifft::Complex<f64>> = frame
        .iter()
        .zip(&window)
        .map(|(&s, &w)| oxifft::Complex::new(f64::from(s * w), 0.0))
        .chain(std::iter::repeat(oxifft::Complex::new(0.0, 0.0)))
        .take(n)
        .collect();

    let spectrum = oxifft::fft(&buffer);
    let mag: Vec<f32> = spectrum[..=n / 2]
        .iter()
        .map(|c| c.norm() as f32)
        .collect();

    let hz_per_bin = sample_rate / (2.0 * (mag.len() - 1) as f32);

    let total_energy: f32 = mag.iter().map(|&m| m * m).sum();
    let spectral_centroid = if total_energy > 1e-10 {
        let weighted: f32 = mag
            .iter()
            .enumerate()
            .map(|(i, &m)| i as f32 * hz_per_bin * m * m)
            .sum();
        weighted / total_energy
    } else {
        0.0
    };

    // Simple F0 estimate via autocorrelation peak
    let mean_f0 = simple_f0_estimate(frame, sample_rate);

    // F1 estimate: first prominent spectral peak between 250–900 Hz
    let f1_bin_lo = (250.0 / hz_per_bin) as usize;
    let f1_bin_hi = (900.0 / hz_per_bin) as usize;
    let f1 = if f1_bin_hi < mag.len() && f1_bin_lo < f1_bin_hi {
        let peak_bin = mag[f1_bin_lo..=f1_bin_hi]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i + f1_bin_lo)
            .unwrap_or(f1_bin_lo);
        peak_bin as f32 * hz_per_bin
    } else {
        500.0 // default
    };

    SpeakerFeature {
        mean_f0,
        spectral_centroid,
        zcr,
        rms,
        f1,
    }
}

/// Simple F0 estimate via normalized autocorrelation.
fn simple_f0_estimate(frame: &[f32], sample_rate: f32) -> f32 {
    let min_lag = (sample_rate / 800.0) as usize; // ~800 Hz max
    let max_lag = (sample_rate / 70.0) as usize; // ~70 Hz min

    if max_lag >= frame.len() || min_lag >= max_lag {
        return 0.0;
    }

    let energy: f32 = frame.iter().map(|&x| x * x).sum();
    if energy < 1e-10 {
        return 0.0;
    }

    let mut best_lag = 0usize;
    let mut best_corr = 0.0_f32;

    for lag in min_lag..=max_lag.min(frame.len() - 1) {
        let corr: f32 = frame[..frame.len() - lag]
            .iter()
            .zip(&frame[lag..])
            .map(|(&a, &b)| a * b)
            .sum::<f32>()
            / energy;

        if corr > best_corr {
            best_corr = corr;
            best_lag = lag;
        }
    }

    if best_corr > 0.35 && best_lag > 0 {
        sample_rate / best_lag as f32
    } else {
        0.0
    }
}

/// Confidence: average similarity of frames in this segment to the centroid.
fn compute_cluster_confidence(
    features: &[SpeakerFeature],
    centroids: &[SpeakerFeature],
    centroid_id: usize,
) -> f32 {
    if features.is_empty() || centroid_id >= centroids.len() {
        return 0.5;
    }
    let centroid = &centroids[centroid_id];
    let mean_dist = features
        .iter()
        .map(|f| feature_distance(f, centroid))
        .sum::<f32>()
        / features.len() as f32;
    (1.0 - mean_dist).max(0.0).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn make_sine(freq: f32, n: usize, sr: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / sr).sin() * 0.6)
            .collect()
    }

    #[test]
    fn test_diarizer_single_speaker() {
        let config = AnalysisConfig::default();
        let diarizer = SpeakerDiarizer::new(config);
        let samples = make_sine(200.0, 44100, 44100.0);
        let result = diarizer.diarize(&samples, 44100.0);
        assert!(result.is_ok(), "Diarization should succeed");
        let r = result.expect("should succeed");
        assert!(r.num_speakers <= 6);
    }

    #[test]
    fn test_diarizer_two_different_speakers() {
        let config = AnalysisConfig::default();
        let diarizer = SpeakerDiarizer::new(config).with_merge_threshold(0.15);

        // Two distinct "voices": different frequencies
        let mut samples = make_sine(150.0, 44100, 44100.0);  // male-like
        samples.extend(make_sine(300.0, 44100, 44100.0));    // female-like

        let result = diarizer.diarize(&samples, 44100.0).expect("should succeed");
        // We should detect at least 1 speaker
        assert!(result.num_speakers >= 1);
        // Segments should be non-empty
        for seg in &result.segments {
            assert!(seg.end_time >= seg.start_time);
            assert!(seg.confidence >= 0.0 && seg.confidence <= 1.0);
        }
    }

    #[test]
    fn test_speaking_time() {
        let result = DiarizationResult {
            segments: vec![
                SpeakerSegment { speaker_id: 0, start_time: 0.0, end_time: 1.5, confidence: 0.8 },
                SpeakerSegment { speaker_id: 1, start_time: 1.5, end_time: 3.0, confidence: 0.7 },
                SpeakerSegment { speaker_id: 0, start_time: 3.0, end_time: 4.0, confidence: 0.9 },
            ],
            num_speakers: 2,
            frame_assignments: Vec::new(),
        };
        let time_0 = result.speaking_time(0);
        assert!((time_0 - 2.5).abs() < 1e-5, "Speaker 0 speaking time should be 2.5: {time_0}");
    }

    #[test]
    fn test_insufficient_samples() {
        let config = AnalysisConfig::default();
        let diarizer = SpeakerDiarizer::new(config);
        let result = diarizer.diarize(&[0.0; 100], 44100.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_feature_distance_identical() {
        let f = SpeakerFeature {
            mean_f0: 150.0,
            spectral_centroid: 1000.0,
            zcr: 0.1,
            rms: 0.3,
            f1: 500.0,
        };
        let dist = feature_distance(&f, &f);
        assert!(dist < 1e-6, "Distance from self should be near zero: {dist}");
    }

    #[test]
    fn test_simple_f0_estimate_silence() {
        let frame = vec![0.0_f32; 512];
        let f0 = simple_f0_estimate(&frame, 44100.0);
        assert_eq!(f0, 0.0);
    }
}
