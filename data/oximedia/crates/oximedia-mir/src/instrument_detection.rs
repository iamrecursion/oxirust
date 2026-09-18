//! Musical instrument detection from spectral audio features.
//!
//! Provides heuristic-based instrument family detection using spectral centroid,
//! spectral rolloff, and zero-crossing rate.

#![allow(dead_code)]

// ── InstrumentFamily ─────────────────────────────────────────────────────────

/// High-level instrument family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstrumentFamily {
    /// Piano and keyboard instruments.
    Piano,
    /// Acoustic or electric guitar.
    Guitar,
    /// Orchestral strings (violin, viola, cello, double bass).
    Strings,
    /// Brass instruments (trumpet, trombone, French horn, tuba).
    Brass,
    /// Woodwind instruments (flute, clarinet, saxophone, oboe).
    Woodwind,
    /// Percussion (drums, cymbals, xylophone, etc.).
    Percussion,
    /// Human voice / vocal.
    Vocal,
    /// Electronic synthesizer.
    Synth,
    /// Bass instruments (electric bass, upright bass).
    Bass,
}

impl InstrumentFamily {
    /// Returns `true` for instruments that produce a definite pitch.
    #[must_use]
    pub fn is_pitched(&self) -> bool {
        !matches!(self, Self::Percussion)
    }
}

// ── InstrumentDetection ───────────────────────────────────────────────────────

/// Detection result for a single instrument family.
#[derive(Debug, Clone)]
pub struct InstrumentDetection {
    /// The detected instrument family.
    pub instrument: InstrumentFamily,
    /// Confidence score in \[0.0, 1.0\].
    pub confidence: f32,
    /// Whether this instrument is currently playing (active in the mix).
    pub active: bool,
}

impl InstrumentDetection {
    /// Returns `true` if the confidence exceeds `threshold` AND the instrument is active.
    #[must_use]
    pub fn is_present(&self, threshold: f32) -> bool {
        self.active && self.confidence > threshold
    }
}

// ── InstrumentMix ─────────────────────────────────────────────────────────────

/// A collection of instrument detections representing a full mix.
#[derive(Debug, Clone, Default)]
pub struct InstrumentMix {
    /// All instrument detections in the mix.
    pub detections: Vec<InstrumentDetection>,
}

impl InstrumentMix {
    /// Return references to all instruments present above `threshold`.
    #[must_use]
    pub fn active_instruments(&self, threshold: f32) -> Vec<&InstrumentDetection> {
        self.detections
            .iter()
            .filter(|d| d.is_present(threshold))
            .collect()
    }

    /// Return the instrument with the highest confidence, or `None` if the mix is empty.
    #[must_use]
    pub fn dominant(&self) -> Option<&InstrumentDetection> {
        self.detections.iter().max_by(|a, b| {
            a.confidence
                .partial_cmp(&b.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Return the number of instruments present above `threshold`.
    #[must_use]
    pub fn ensemble_size(&self, threshold: f32) -> usize {
        self.active_instruments(threshold).len()
    }
}

// ── SpectralInstrumentEstimator ────────────────────────────────────────────────

/// Heuristic instrument estimator based on spectral features.
pub struct SpectralInstrumentEstimator;

impl SpectralInstrumentEstimator {
    /// Estimate the most likely instrument family from three spectral features.
    ///
    /// # Arguments
    /// * `spectral_centroid` – mean frequency of the spectrum (Hz).
    /// * `rolloff_hz`        – frequency below which 85% of spectral energy resides (Hz).
    /// * `zcr`               – zero-crossing rate (crossings per second).
    ///
    /// # Heuristic rules
    ///
    /// | Centroid | Rolloff | ZCR  | → Instrument |
    /// |----------|---------|------|-------------|
    /// | very high | high    | high | Percussion  |
    /// | high      | high    | low  | Brass / Woodwind |
    /// | mid-high  | mid     | low  | Vocal       |
    /// | mid       | mid     | any  | Piano / Guitar |
    /// | low       | low     | any  | Bass / Strings |
    #[must_use]
    pub fn estimate(spectral_centroid: f32, rolloff_hz: f32, zcr: f32) -> InstrumentDetection {
        let (instrument, confidence) = if spectral_centroid > 6000.0 && zcr > 3000.0 {
            (InstrumentFamily::Percussion, 0.75)
        } else if spectral_centroid > 4000.0 && rolloff_hz > 8000.0 {
            (InstrumentFamily::Brass, 0.65)
        } else if spectral_centroid > 3000.0 && rolloff_hz > 6000.0 && zcr < 2000.0 {
            (InstrumentFamily::Woodwind, 0.60)
        } else if spectral_centroid > 2000.0 && spectral_centroid <= 4000.0 && zcr < 1500.0 {
            (InstrumentFamily::Vocal, 0.55)
        } else if spectral_centroid > 1000.0 && spectral_centroid <= 3000.0 {
            (InstrumentFamily::Piano, 0.50)
        } else if spectral_centroid > 500.0 && spectral_centroid <= 1500.0 {
            (InstrumentFamily::Guitar, 0.50)
        } else if spectral_centroid <= 500.0 && rolloff_hz <= 2000.0 {
            (InstrumentFamily::Bass, 0.60)
        } else {
            (InstrumentFamily::Strings, 0.40)
        };

        InstrumentDetection {
            instrument,
            confidence,
            active: confidence > 0.0,
        }
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_detection(
        instrument: InstrumentFamily,
        confidence: f32,
        active: bool,
    ) -> InstrumentDetection {
        InstrumentDetection {
            instrument,
            confidence,
            active,
        }
    }

    // ── InstrumentFamily ────────────────────────────────────────────────────────

    #[test]
    fn test_piano_is_pitched() {
        assert!(InstrumentFamily::Piano.is_pitched());
    }

    #[test]
    fn test_percussion_not_pitched() {
        assert!(!InstrumentFamily::Percussion.is_pitched());
    }

    #[test]
    fn test_guitar_is_pitched() {
        assert!(InstrumentFamily::Guitar.is_pitched());
    }

    #[test]
    fn test_bass_is_pitched() {
        assert!(InstrumentFamily::Bass.is_pitched());
    }

    #[test]
    fn test_vocal_is_pitched() {
        assert!(InstrumentFamily::Vocal.is_pitched());
    }

    // ── InstrumentDetection ─────────────────────────────────────────────────────

    #[test]
    fn test_is_present_active_above_threshold() {
        let det = make_detection(InstrumentFamily::Piano, 0.8, true);
        assert!(det.is_present(0.7));
    }

    #[test]
    fn test_is_present_inactive_false() {
        let det = make_detection(InstrumentFamily::Guitar, 0.9, false);
        assert!(!det.is_present(0.5));
    }

    #[test]
    fn test_is_present_below_threshold_false() {
        let det = make_detection(InstrumentFamily::Brass, 0.3, true);
        assert!(!det.is_present(0.5));
    }

    // ── InstrumentMix ───────────────────────────────────────────────────────────

    #[test]
    fn test_mix_active_instruments_count() {
        let mix = InstrumentMix {
            detections: vec![
                make_detection(InstrumentFamily::Piano, 0.9, true),
                make_detection(InstrumentFamily::Guitar, 0.4, true),
                make_detection(InstrumentFamily::Percussion, 0.8, true),
            ],
        };
        assert_eq!(mix.active_instruments(0.7).len(), 2);
    }

    #[test]
    fn test_mix_dominant_returns_highest_confidence() {
        let mix = InstrumentMix {
            detections: vec![
                make_detection(InstrumentFamily::Vocal, 0.6, true),
                make_detection(InstrumentFamily::Strings, 0.95, true),
                make_detection(InstrumentFamily::Bass, 0.3, true),
            ],
        };
        let dom = mix.dominant().expect("should succeed in test");
        assert_eq!(dom.instrument, InstrumentFamily::Strings);
    }

    #[test]
    fn test_mix_dominant_empty_is_none() {
        let mix = InstrumentMix { detections: vec![] };
        assert!(mix.dominant().is_none());
    }

    #[test]
    fn test_mix_ensemble_size() {
        let mix = InstrumentMix {
            detections: vec![
                make_detection(InstrumentFamily::Piano, 0.85, true),
                make_detection(InstrumentFamily::Strings, 0.75, true),
                make_detection(InstrumentFamily::Woodwind, 0.45, true),
            ],
        };
        assert_eq!(mix.ensemble_size(0.7), 2);
    }

    // ── SpectralInstrumentEstimator ─────────────────────────────────────────────

    #[test]
    fn test_estimator_high_centroid_high_zcr_is_percussion() {
        let det = SpectralInstrumentEstimator::estimate(7000.0, 10000.0, 5000.0);
        assert_eq!(det.instrument, InstrumentFamily::Percussion);
    }

    #[test]
    fn test_estimator_low_centroid_low_rolloff_is_bass() {
        let det = SpectralInstrumentEstimator::estimate(300.0, 1500.0, 500.0);
        assert_eq!(det.instrument, InstrumentFamily::Bass);
    }

    #[test]
    fn test_estimator_always_returns_active() {
        let det = SpectralInstrumentEstimator::estimate(3000.0, 6500.0, 1000.0);
        // active should be true as long as confidence > 0
        assert!(det.active);
    }

    #[test]
    fn test_estimator_confidence_in_range() {
        let det = SpectralInstrumentEstimator::estimate(2500.0, 5000.0, 800.0);
        assert!(det.confidence >= 0.0);
        assert!(det.confidence <= 1.0);
    }
}
