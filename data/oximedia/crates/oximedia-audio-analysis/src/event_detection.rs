//! Sound event detection: applause, laughter, coughing, siren, and more.
//!
//! Uses a rule-based approach derived from spectral and temporal features:
//! - **Applause**: broadband energy bursts with high spectral flatness and
//!   strong transient density
//! - **Laughter**: rhythmic bursts of voiced energy, moderate pitch
//! - **Coughing**: sharp broadband burst, single transient, short duration
//! - **Siren**: frequency-modulated tonal component sweeping 500–1500 Hz
//! - **Alarm**: short repeated tonal bursts at fixed frequency
//! - **Footsteps**: low-frequency rhythmic transients
//! - **Door slam**: very short wideband impulse, strong low-frequency content
//! - **Gunshot**: extremely short (<20 ms) broadband transient, very high crest
//! - **Music**: tonal, low flatness, sustained energy
//! - **Speech**: moderate ZCR, variable pitch, short voiced segments

use crate::spectral::{SpectralAnalyzer, SpectralFeatures};
use crate::{compute_rms, zero_crossing_rate, AnalysisConfig, AnalysisError, Result};

/// Sound event type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SoundEvent {
    /// Applause (crowd clapping)
    Applause,
    /// Laughter
    Laughter,
    /// Coughing
    Coughing,
    /// Siren (emergency vehicle)
    Siren,
    /// Alarm (repeating beep / fire alarm)
    Alarm,
    /// Footsteps
    Footsteps,
    /// Door slam
    DoorSlam,
    /// Gunshot or explosion
    Gunshot,
    /// Music
    Music,
    /// Speech
    Speech,
    /// Unknown / no clear event
    Unknown,
}

impl SoundEvent {
    fn as_str(self) -> &'static str {
        match self {
            Self::Applause => "applause",
            Self::Laughter => "laughter",
            Self::Coughing => "coughing",
            Self::Siren => "siren",
            Self::Alarm => "alarm",
            Self::Footsteps => "footsteps",
            Self::DoorSlam => "door_slam",
            Self::Gunshot => "gunshot",
            Self::Music => "music",
            Self::Speech => "speech",
            Self::Unknown => "unknown",
        }
    }
}

impl std::fmt::Display for SoundEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A detected sound event occurrence.
#[derive(Debug, Clone)]
pub struct DetectedEvent {
    /// Type of sound event
    pub event: SoundEvent,
    /// Start time in seconds
    pub start_time: f32,
    /// End time in seconds
    pub end_time: f32,
    /// Confidence score (0.0–1.0)
    pub confidence: f32,
}

impl DetectedEvent {
    /// Duration of the event in seconds.
    #[must_use]
    pub fn duration(&self) -> f32 {
        self.end_time - self.start_time
    }
}

/// Probability scores for all event types in a single frame.
#[derive(Debug, Clone, Default)]
pub struct EventScores {
    /// Per-event probability (event, score) pairs, sorted by score descending
    pub scores: Vec<(SoundEvent, f32)>,
}

impl EventScores {
    /// Return the most likely event and its score.
    #[must_use]
    pub fn top(&self) -> Option<(SoundEvent, f32)> {
        self.scores.first().copied()
    }
}

/// Sound event detector.
pub struct EventDetector {
    config: AnalysisConfig,
    spectral: SpectralAnalyzer,
    /// Minimum confidence to emit an event
    min_confidence: f32,
    /// Minimum event duration in seconds
    min_duration: f32,
}

impl EventDetector {
    /// Create a new event detector.
    #[must_use]
    pub fn new(config: AnalysisConfig) -> Self {
        let spectral = SpectralAnalyzer::new(config.clone());
        Self {
            config,
            spectral,
            min_confidence: 0.4,
            min_duration: 0.05,
        }
    }

    /// Set the minimum confidence threshold.
    #[must_use]
    pub fn with_min_confidence(mut self, threshold: f32) -> Self {
        self.min_confidence = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set the minimum event duration in seconds.
    #[must_use]
    pub fn with_min_duration(mut self, duration: f32) -> Self {
        self.min_duration = duration.max(0.0);
        self
    }

    /// Detect sound events in audio and return list of detected occurrences.
    pub fn detect(&self, samples: &[f32], sample_rate: f32) -> Result<Vec<DetectedEvent>> {
        let frame_scores = self.score_frames(samples, sample_rate)?;
        let hop = self.config.hop_size;
        let hop_dur = hop as f32 / sample_rate;
        let min_frames = ((self.min_duration / hop_dur).ceil() as usize).max(1);

        // Merge consecutive frames with the same top event
        let mut events: Vec<DetectedEvent> = Vec::new();

        if frame_scores.is_empty() {
            return Ok(events);
        }

        let mut current_event = frame_scores[0].top().unwrap_or((SoundEvent::Unknown, 0.0));
        let mut run_start = 0usize;
        let mut run_scores: Vec<f32> = vec![current_event.1];

        for (i, scores) in frame_scores.iter().enumerate().skip(1) {
            let (top_event, top_score) = scores.top().unwrap_or((SoundEvent::Unknown, 0.0));

            if top_event == current_event.0 {
                run_scores.push(top_score);
            } else {
                // Emit current run
                let mean_conf = run_scores.iter().sum::<f32>() / run_scores.len() as f32;
                if i - run_start >= min_frames && mean_conf >= self.min_confidence {
                    events.push(DetectedEvent {
                        event: current_event.0,
                        start_time: run_start as f32 * hop_dur,
                        end_time: i as f32 * hop_dur,
                        confidence: mean_conf,
                    });
                }
                current_event = (top_event, top_score);
                run_start = i;
                run_scores = vec![top_score];
            }
        }

        // Final run
        let n = frame_scores.len();
        let mean_conf = run_scores.iter().sum::<f32>() / run_scores.len() as f32;
        if n - run_start >= min_frames && mean_conf >= self.min_confidence {
            events.push(DetectedEvent {
                event: current_event.0,
                start_time: run_start as f32 * hop_dur,
                end_time: n as f32 * hop_dur,
                confidence: mean_conf,
            });
        }

        Ok(events)
    }

    /// Compute per-frame event probability scores.
    pub fn score_frames(&self, samples: &[f32], sample_rate: f32) -> Result<Vec<EventScores>> {
        let fft_size = self.config.fft_size;
        let hop = self.config.hop_size;

        if samples.len() < fft_size {
            return Err(AnalysisError::InsufficientSamples {
                needed: fft_size,
                got: samples.len(),
            });
        }

        let num_frames = (samples.len() - fft_size) / hop + 1;
        let mut frame_scores = Vec::with_capacity(num_frames);

        let mut prev_mag: Option<Vec<f32>> = None;
        let mut prev_centroid = 0.0_f32;

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

            let centroid_delta = (feats.centroid - prev_centroid).abs();

            let scores = score_frame(
                rms,
                zcr,
                feats.flatness,
                feats.centroid,
                feats.crest,
                flux,
                centroid_delta,
                sample_rate,
            );

            prev_mag = Some(feats.magnitude_spectrum);
            prev_centroid = feats.centroid;

            frame_scores.push(scores);
        }

        Ok(frame_scores)
    }
}

// ── classification rules ─────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn score_frame(
    rms: f32,
    zcr: f32,
    flatness: f32,
    centroid: f32,
    crest: f32,
    flux: f32,
    centroid_delta: f32,
    sample_rate: f32,
) -> EventScores {
    let nyquist = sample_rate / 2.0;

    // ── applause ──────────────────────────────────────────────────────────
    // broadband, high flatness, high flux, strong transients
    let applause = score_applause(rms, flatness, flux, crest);

    // ── laughter ─────────────────────────────────────────────────────────
    // rhythmic bursts, moderate pitch (200-600 Hz centroid), moderate flatness
    let laughter = score_laughter(rms, centroid, flatness, flux);

    // ── coughing ─────────────────────────────────────────────────────────
    // sharp burst, short, wideband
    let coughing = score_coughing(rms, flatness, flux, crest);

    // ── siren ─────────────────────────────────────────────────────────────
    // tonal, sweeping centroid, mid-range frequency
    let siren = score_siren(rms, centroid, centroid_delta, flatness);

    // ── alarm ─────────────────────────────────────────────────────────────
    // tonal beeps, narrow band, low flatness
    let alarm = score_alarm(rms, centroid, flatness, nyquist);

    // ── footsteps ─────────────────────────────────────────────────────────
    // rhythmic low-frequency transients
    let footsteps = score_footsteps(rms, centroid, crest, zcr);

    // ── door slam ─────────────────────────────────────────────────────────
    // impulsive, wideband, very high crest
    let door_slam = score_door_slam(rms, flatness, crest, flux);

    // ── gunshot ──────────────────────────────────────────────────────────
    // extremely high crest, broadband, near-instantaneous
    let gunshot = score_gunshot(rms, flatness, crest);

    // ── music ─────────────────────────────────────────────────────────────
    let music = score_music(rms, flatness, zcr);

    // ── speech ────────────────────────────────────────────────────────────
    let speech = score_speech(rms, zcr, flatness, centroid);

    let mut scores = vec![
        (SoundEvent::Applause, applause),
        (SoundEvent::Laughter, laughter),
        (SoundEvent::Coughing, coughing),
        (SoundEvent::Siren, siren),
        (SoundEvent::Alarm, alarm),
        (SoundEvent::Footsteps, footsteps),
        (SoundEvent::DoorSlam, door_slam),
        (SoundEvent::Gunshot, gunshot),
        (SoundEvent::Music, music),
        (SoundEvent::Speech, speech),
        (SoundEvent::Unknown, 0.1),
    ];

    scores.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    EventScores { scores }
}

fn score_applause(rms: f32, flatness: f32, flux: f32, crest: f32) -> f32 {
    if rms < 0.01 {
        return 0.0;
    }
    let flat_score = if flatness > 0.6 {
        1.0_f32
    } else {
        flatness / 0.6
    };
    let flux_score = (flux / 10.0).min(1.0);
    let crest_score = if crest > 3.0 { 1.0_f32 } else { crest / 3.0 };
    (flat_score * 0.4 + flux_score * 0.35 + crest_score * 0.25).min(1.0)
}

fn score_laughter(rms: f32, centroid: f32, flatness: f32, flux: f32) -> f32 {
    if rms < 0.02 {
        return 0.0;
    }
    let cent_score = if (200.0..=800.0).contains(&centroid) {
        1.0_f32
    } else {
        0.3
    };
    let flat_score = if (0.2..=0.6).contains(&flatness) {
        1.0_f32
    } else {
        0.4
    };
    let flux_score = (flux / 5.0).min(1.0);
    (cent_score * 0.4 + flat_score * 0.3 + flux_score * 0.3).min(1.0)
}

fn score_coughing(rms: f32, flatness: f32, flux: f32, crest: f32) -> f32 {
    if rms < 0.05 {
        return 0.0;
    }
    let flat_score = if flatness > 0.5 {
        1.0_f32
    } else {
        flatness / 0.5
    };
    let crest_score = if crest > 5.0 { 1.0_f32 } else { crest / 5.0 };
    let flux_score = (flux / 20.0).min(1.0);
    (flat_score * 0.35 + crest_score * 0.4 + flux_score * 0.25).min(1.0)
}

fn score_siren(rms: f32, centroid: f32, centroid_delta: f32, flatness: f32) -> f32 {
    if rms < 0.01 {
        return 0.0;
    }
    // Siren: centroid sweeps 500–1500 Hz
    let cent_score = if (400.0..=1600.0).contains(&centroid) {
        1.0_f32
    } else {
        0.2
    };
    let sweep_score = (centroid_delta / 200.0).min(1.0);
    let tonal_score = if flatness < 0.35 { 1.0_f32 } else { 0.4 };
    (cent_score * 0.35 + sweep_score * 0.4 + tonal_score * 0.25).min(1.0)
}

fn score_alarm(rms: f32, centroid: f32, flatness: f32, nyquist: f32) -> f32 {
    if rms < 0.01 {
        return 0.0;
    }
    // Beeping alarm: narrow tonal around 1–4 kHz, very low flatness
    let freq_score = if (800.0..=4000.0_f32.min(nyquist)).contains(&centroid) {
        1.0_f32
    } else {
        0.2
    };
    let tonal_score = if flatness < 0.15 { 1.0_f32 } else { 0.3 };
    (freq_score * 0.5 + tonal_score * 0.5).min(1.0)
}

fn score_footsteps(rms: f32, centroid: f32, crest: f32, zcr: f32) -> f32 {
    if rms < 0.01 {
        return 0.0;
    }
    let cent_score = if centroid < 600.0 {
        1.0_f32
    } else {
        300.0 / centroid
    };
    let crest_score = if crest > 4.0 { 1.0_f32 } else { crest / 4.0 };
    let zcr_score = if zcr < 0.15 { 1.0_f32 } else { 0.3 };
    (cent_score * 0.4 + crest_score * 0.35 + zcr_score * 0.25).min(1.0)
}

fn score_door_slam(rms: f32, flatness: f32, crest: f32, flux: f32) -> f32 {
    if rms < 0.05 {
        return 0.0;
    }
    let flat_score = if flatness > 0.5 {
        1.0_f32
    } else {
        flatness / 0.5
    };
    let crest_score = if crest > 8.0 { 1.0_f32 } else { crest / 8.0 };
    let flux_score = (flux / 30.0).min(1.0);
    (flat_score * 0.3 + crest_score * 0.45 + flux_score * 0.25).min(1.0)
}

fn score_gunshot(rms: f32, flatness: f32, crest: f32) -> f32 {
    if rms < 0.1 {
        return 0.0;
    }
    let crest_score = if crest > 15.0 { 1.0_f32 } else { crest / 15.0 };
    let flat_score = if flatness > 0.65 {
        1.0_f32
    } else {
        flatness / 0.65
    };
    (crest_score * 0.6 + flat_score * 0.4).min(1.0)
}

fn score_music(rms: f32, flatness: f32, zcr: f32) -> f32 {
    if rms < 0.005 {
        return 0.0;
    }
    let tonal = if flatness < 0.3 { 1.0_f32 } else { 0.3 };
    let low_zcr = if zcr < 0.12 { 1.0_f32 } else { 0.4 };
    (tonal * 0.6 + low_zcr * 0.4).min(1.0)
}

fn score_speech(rms: f32, zcr: f32, flatness: f32, centroid: f32) -> f32 {
    if rms < 0.005 {
        return 0.0;
    }
    let zcr_score = if (0.05..=0.45).contains(&zcr) {
        1.0_f32
    } else {
        0.3
    };
    let flat_score = if (0.1..=0.6).contains(&flatness) {
        1.0
    } else {
        0.3
    };
    let cent_score = if (200.0..=3000.0).contains(&centroid) {
        1.0_f32
    } else {
        0.4
    };
    (zcr_score * 0.35 + flat_score * 0.35 + cent_score * 0.3).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn make_sine(freq: f32, n: usize, sr: f32, amp: f32) -> Vec<f32> {
        (0..n)
            .map(|i| amp * (2.0 * PI * freq * i as f32 / sr).sin())
            .collect()
    }

    fn make_noise(n: usize, amp: f32) -> Vec<f32> {
        let mut x: u64 = 0x123456789abcdef0;
        (0..n)
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (x as i64 as f64 / i64::MAX as f64 * amp as f64) as f32
            })
            .collect()
    }

    #[test]
    fn test_event_detector_sine_wave() {
        let config = AnalysisConfig::default();
        let detector = EventDetector::new(config);
        let samples = make_sine(1000.0, 44100, 44100.0, 0.5);
        let events = detector.detect(&samples, 44100.0);
        assert!(events.is_ok());
    }

    #[test]
    fn test_event_detector_noise() {
        let config = AnalysisConfig::default();
        let detector = EventDetector::new(config);
        let samples = make_noise(44100, 0.5);
        let events = detector.detect(&samples, 44100.0);
        assert!(events.is_ok());
    }

    #[test]
    fn test_event_scores_all_present() {
        let scores = score_frame(0.3, 0.15, 0.4, 1000.0, 4.0, 5.0, 50.0, 44100.0);
        assert!(!scores.scores.is_empty());
        for (_, s) in &scores.scores {
            assert!(*s >= 0.0 && *s <= 1.0, "Score out of range: {s}");
        }
    }

    #[test]
    fn test_high_crest_scores_impulsive_event() {
        // Very high crest + broadband → should score an impulsive event highly
        // (Gunshot, DoorSlam, Coughing, or Applause — all impulsive-transient classes).
        let scores = score_frame(0.9, 0.3, 0.7, 3000.0, 20.0, 40.0, 10.0, 44100.0);
        let top = scores.top().expect("should have top");
        assert!(
            top.0 == SoundEvent::Gunshot
                || top.0 == SoundEvent::DoorSlam
                || top.0 == SoundEvent::Coughing
                || top.0 == SoundEvent::Applause,
            "Very high crest/flatness/flux should score an impulsive event, got {:?}",
            top.0
        );
    }

    #[test]
    fn test_tonal_low_zcr_scores_tonal_event() {
        // Very low flatness + low ZCR → tonal, structured audio
        // Depending on centroid (800 Hz) it may score Music, Speech, or Alarm.
        let scores = score_frame(0.3, 0.05, 0.05, 800.0, 2.0, 0.5, 5.0, 44100.0);
        let top = scores.top().expect("should have top");
        // Verify score is meaningful (not Unknown)
        assert_ne!(
            top.0,
            SoundEvent::Unknown,
            "Structured audio should not score Unknown"
        );
        // Tonal events have high score
        assert!(
            top.1 > 0.1,
            "Tonal audio top score should be > 0.1, got {:.3}",
            top.1
        );
    }

    #[test]
    fn test_insufficient_samples() {
        let config = AnalysisConfig::default();
        let detector = EventDetector::new(config);
        let result = detector.score_frames(&[0.0; 100], 44100.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_detected_event_duration() {
        let e = DetectedEvent {
            event: SoundEvent::Applause,
            start_time: 1.0,
            end_time: 3.5,
            confidence: 0.8,
        };
        assert!((e.duration() - 2.5).abs() < 1e-6);
    }
}
