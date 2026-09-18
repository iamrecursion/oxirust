//! Speech detection and analysis module.
//!
//! Provides tools for detecting speech segments in audio data,
//! computing speech statistics, and analyzing silence patterns.

use serde::{Deserialize, Serialize};

/// A detected speech segment with timing and speaker information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechSegment {
    /// Start time in milliseconds
    pub start_ms: u64,
    /// End time in milliseconds
    pub end_ms: u64,
    /// Detection confidence in [0.0, 1.0]
    pub confidence: f64,
    /// Optional speaker identifier
    pub speaker_id: Option<u32>,
}

impl SpeechSegment {
    /// Create a new speech segment.
    #[must_use]
    pub fn new(start_ms: u64, end_ms: u64, confidence: f64) -> Self {
        Self {
            start_ms,
            end_ms,
            confidence,
            speaker_id: None,
        }
    }

    /// Create a speech segment with an assigned speaker.
    #[must_use]
    pub fn with_speaker(start_ms: u64, end_ms: u64, confidence: f64, speaker_id: u32) -> Self {
        Self {
            start_ms,
            end_ms,
            confidence,
            speaker_id: Some(speaker_id),
        }
    }

    /// Duration of this segment in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }

    /// Returns `true` if the segment is longer than 3 seconds.
    #[must_use]
    pub fn is_long(&self) -> bool {
        self.duration_ms() > 3_000
    }
}

/// Speech detector using energy-based voice activity detection.
#[derive(Debug, Clone)]
pub struct SpeechDetector {
    /// RMS energy threshold above which audio is considered speech
    pub energy_threshold: f64,
    /// Minimum duration in ms for a segment to count as speech
    pub min_duration_ms: u64,
    /// Audio sample rate in Hz
    pub sample_rate: u32,
}

impl SpeechDetector {
    /// Create a new `SpeechDetector` with default parameters for the given sample rate.
    #[must_use]
    pub fn new(sample_rate: u32) -> Self {
        Self {
            energy_threshold: 0.02,
            min_duration_ms: 200,
            sample_rate,
        }
    }

    /// Detect speech segments from a sequence of RMS energy values.
    ///
    /// Each element of `rms_values` corresponds to one audio frame of
    /// `frame_duration_ms` milliseconds.
    #[must_use]
    pub fn detect_segments(
        &self,
        rms_values: &[f64],
        frame_duration_ms: u64,
    ) -> Vec<SpeechSegment> {
        let mut segments = Vec::new();
        let mut in_speech = false;
        let mut seg_start_ms: u64 = 0;

        for (i, &rms) in rms_values.iter().enumerate() {
            let t = i as u64 * frame_duration_ms;
            let is_speech = rms >= self.energy_threshold;

            if is_speech && !in_speech {
                in_speech = true;
                seg_start_ms = t;
            } else if !is_speech && in_speech {
                in_speech = false;
                let duration = t.saturating_sub(seg_start_ms);
                if duration >= self.min_duration_ms {
                    let confidence =
                        Self::estimate_confidence(rms_values, seg_start_ms, t, frame_duration_ms);
                    segments.push(SpeechSegment::new(seg_start_ms, t, confidence));
                }
            }
        }

        // Close any open segment at the end
        if in_speech {
            let t_end = rms_values.len() as u64 * frame_duration_ms;
            let duration = t_end.saturating_sub(seg_start_ms);
            if duration >= self.min_duration_ms {
                let confidence =
                    Self::estimate_confidence(rms_values, seg_start_ms, t_end, frame_duration_ms);
                segments.push(SpeechSegment::new(seg_start_ms, t_end, confidence));
            }
        }

        segments
    }

    /// Estimate confidence as the mean RMS normalized to [0, 1] over the segment.
    fn estimate_confidence(
        rms_values: &[f64],
        start_ms: u64,
        end_ms: u64,
        frame_duration_ms: u64,
    ) -> f64 {
        if frame_duration_ms == 0 {
            return 0.5;
        }
        let start_idx = (start_ms / frame_duration_ms) as usize;
        let end_idx = ((end_ms / frame_duration_ms) as usize).min(rms_values.len());
        if start_idx >= end_idx {
            return 0.5;
        }
        let slice = &rms_values[start_idx..end_idx];
        let mean = slice.iter().sum::<f64>() / slice.len() as f64;
        // Clamp to [0, 1]
        mean.clamp(0.0, 1.0)
    }

    /// Compute the fraction of `total_ms` that is covered by speech segments.
    #[must_use]
    pub fn speech_ratio(segments: &[SpeechSegment], total_ms: u64) -> f64 {
        if total_ms == 0 {
            return 0.0;
        }
        let speech_ms: u64 = segments.iter().map(SpeechSegment::duration_ms).sum();
        (speech_ms as f64 / total_ms as f64).clamp(0.0, 1.0)
    }

    /// Find the longest contiguous silence gap between (and around) segments.
    #[must_use]
    pub fn longest_silence_ms(segments: &[SpeechSegment], total_ms: u64) -> u64 {
        if segments.is_empty() {
            return total_ms;
        }

        let mut sorted = segments.to_vec();
        sorted.sort_by_key(|s| s.start_ms);

        let mut max_silence: u64 = 0;

        // Silence before first segment
        max_silence = max_silence.max(sorted[0].start_ms);

        // Gaps between segments
        for pair in sorted.windows(2) {
            let gap = pair[1].start_ms.saturating_sub(pair[0].end_ms);
            max_silence = max_silence.max(gap);
        }

        // Silence after last segment
        if let Some(last) = sorted.last() {
            let trailing = total_ms.saturating_sub(last.end_ms);
            max_silence = max_silence.max(trailing);
        }

        max_silence
    }
}

/// Aggregate statistics for a collection of speech segments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechStats {
    /// Total duration of speech in ms
    pub total_speech_ms: u64,
    /// Total silence duration in ms
    pub total_silence_ms: u64,
    /// Number of speech segments detected
    pub segment_count: usize,
    /// Average segment duration in ms
    pub avg_segment_ms: f64,
    /// Ratio of speech to total audio (0.0–1.0)
    pub speech_ratio: f64,
}

/// Compute speech statistics from detected segments and total media duration.
#[must_use]
pub fn compute_speech_stats(segments: &[SpeechSegment], total_ms: u64) -> SpeechStats {
    let total_speech_ms: u64 = segments.iter().map(SpeechSegment::duration_ms).sum();
    let total_silence_ms = total_ms.saturating_sub(total_speech_ms);
    let segment_count = segments.len();
    let avg_segment_ms = if segment_count == 0 {
        0.0
    } else {
        total_speech_ms as f64 / segment_count as f64
    };
    let speech_ratio = SpeechDetector::speech_ratio(segments, total_ms);

    SpeechStats {
        total_speech_ms,
        total_silence_ms,
        segment_count,
        avg_segment_ms,
        speech_ratio,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_seg(start: u64, end: u64) -> SpeechSegment {
        SpeechSegment::new(start, end, 0.9)
    }

    #[test]
    fn test_duration_ms() {
        let seg = make_seg(1000, 4000);
        assert_eq!(seg.duration_ms(), 3000);
    }

    #[test]
    fn test_duration_ms_zero_when_reversed() {
        let seg = make_seg(5000, 3000);
        assert_eq!(seg.duration_ms(), 0); // saturating_sub
    }

    #[test]
    fn test_is_long_true() {
        let seg = make_seg(0, 5000);
        assert!(seg.is_long());
    }

    #[test]
    fn test_is_long_false() {
        let seg = make_seg(0, 2000);
        assert!(!seg.is_long());
    }

    #[test]
    fn test_is_long_boundary() {
        // Exactly 3000 ms is NOT long (> 3000 required)
        let seg = make_seg(0, 3000);
        assert!(!seg.is_long());
    }

    #[test]
    fn test_with_speaker() {
        let seg = SpeechSegment::with_speaker(0, 1000, 0.8, 42);
        assert_eq!(seg.speaker_id, Some(42));
    }

    #[test]
    fn test_detect_segments_basic() {
        let detector = SpeechDetector::new(44100);
        // 5 frames: silence, speech, speech, speech, silence
        let rms = vec![0.001, 0.1, 0.15, 0.12, 0.001];
        let segs = detector.detect_segments(&rms, 100);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].start_ms, 100);
        assert_eq!(segs[0].end_ms, 400);
    }

    #[test]
    fn test_detect_segments_empty() {
        let detector = SpeechDetector::new(44100);
        let segs = detector.detect_segments(&[], 100);
        assert!(segs.is_empty());
    }

    #[test]
    fn test_detect_segments_all_silence() {
        let detector = SpeechDetector::new(44100);
        let rms = vec![0.001, 0.002, 0.001, 0.001];
        let segs = detector.detect_segments(&rms, 100);
        assert!(segs.is_empty());
    }

    #[test]
    fn test_detect_segments_all_speech_closes_at_end() {
        let mut detector = SpeechDetector::new(44100);
        detector.min_duration_ms = 0;
        let rms = vec![0.5, 0.6, 0.7];
        let segs = detector.detect_segments(&rms, 100);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].start_ms, 0);
        assert_eq!(segs[0].end_ms, 300);
    }

    #[test]
    fn test_speech_ratio_basic() {
        let segs = vec![make_seg(0, 500), make_seg(1000, 2000)];
        let ratio = SpeechDetector::speech_ratio(&segs, 4000);
        assert!((ratio - 0.375).abs() < 1e-9); // 1500/4000
    }

    #[test]
    fn test_speech_ratio_zero_total() {
        let segs = vec![make_seg(0, 100)];
        assert_eq!(SpeechDetector::speech_ratio(&segs, 0), 0.0);
    }

    #[test]
    fn test_longest_silence_ms_no_segments() {
        assert_eq!(SpeechDetector::longest_silence_ms(&[], 5000), 5000);
    }

    #[test]
    fn test_longest_silence_ms_gap_between() {
        let segs = vec![make_seg(0, 1000), make_seg(3000, 4000)];
        // gap = 2000, trailing = 1000, leading = 0
        assert_eq!(SpeechDetector::longest_silence_ms(&segs, 5000), 2000);
    }

    #[test]
    fn test_longest_silence_ms_leading() {
        let segs = vec![make_seg(3000, 4000)];
        // leading = 3000, trailing = 1000
        assert_eq!(SpeechDetector::longest_silence_ms(&segs, 5000), 3000);
    }

    #[test]
    fn test_compute_speech_stats() {
        let segs = vec![make_seg(0, 2000), make_seg(3000, 5000)];
        let stats = compute_speech_stats(&segs, 10000);
        assert_eq!(stats.total_speech_ms, 4000);
        assert_eq!(stats.total_silence_ms, 6000);
        assert_eq!(stats.segment_count, 2);
        assert!((stats.avg_segment_ms - 2000.0).abs() < 1e-9);
        assert!((stats.speech_ratio - 0.4).abs() < 1e-9);
    }

    #[test]
    fn test_compute_speech_stats_empty() {
        let stats = compute_speech_stats(&[], 5000);
        assert_eq!(stats.segment_count, 0);
        assert_eq!(stats.total_speech_ms, 0);
        assert_eq!(stats.avg_segment_ms, 0.0);
        assert_eq!(stats.speech_ratio, 0.0);
    }
}
