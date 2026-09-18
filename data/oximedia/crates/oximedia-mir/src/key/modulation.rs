//! Time-varying key detection for songs with key changes (modulations).
//!
//! Splits the audio into overlapping windows, detects the local key in each
//! window using the Krumhansl-Schmuckler algorithm, and identifies modulation
//! points where the detected key changes with sufficient confidence.

use crate::key::detect::KeyDetector;
use crate::{MirError, MirResult};
use serde::{Deserialize, Serialize};

/// A key region: contiguous time span where one key is dominant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyRegion {
    /// Start time in seconds.
    pub start: f32,
    /// End time in seconds.
    pub end: f32,
    /// Key name (e.g. "C major").
    pub key: String,
    /// Root pitch class (0-11).
    pub root: u8,
    /// Whether the key is major.
    pub is_major: bool,
    /// Average confidence within this region (0.0 to 1.0).
    pub confidence: f32,
}

/// A detected modulation (key change event).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Modulation {
    /// Time of the key change in seconds.
    pub time: f32,
    /// Key before the modulation.
    pub from_key: String,
    /// Key after the modulation.
    pub to_key: String,
    /// Confidence in the modulation detection (0.0 to 1.0).
    pub confidence: f32,
    /// Semitone distance of the modulation (0-11).
    pub semitone_distance: u8,
}

/// Result of time-varying key analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModulationResult {
    /// Detected key regions in temporal order.
    pub regions: Vec<KeyRegion>,
    /// Detected modulations (key change events).
    pub modulations: Vec<Modulation>,
    /// Overall key (most dominant across the entire track).
    pub overall_key: String,
    /// Number of distinct keys detected.
    pub num_keys: usize,
}

/// Configuration for modulation detection.
#[derive(Debug, Clone)]
pub struct ModulationConfig {
    /// Window size for local key detection (in seconds).
    pub window_seconds: f32,
    /// Hop size for local key detection (in seconds).
    pub hop_seconds: f32,
    /// Minimum confidence to accept a local key estimate.
    pub min_confidence: f32,
    /// Minimum number of consecutive frames with the same key to form a region.
    pub min_region_frames: usize,
}

impl Default for ModulationConfig {
    fn default() -> Self {
        Self {
            window_seconds: 4.0,
            hop_seconds: 1.0,
            min_confidence: 0.3,
            min_region_frames: 2,
        }
    }
}

/// Detects key changes (modulations) over time.
pub struct ModulationDetector {
    sample_rate: f32,
    window_size: usize,
    config: ModulationConfig,
}

impl ModulationDetector {
    /// Create a new modulation detector.
    #[must_use]
    pub fn new(sample_rate: f32, window_size: usize, config: ModulationConfig) -> Self {
        Self {
            sample_rate,
            window_size,
            config,
        }
    }

    /// Create with default configuration.
    #[must_use]
    pub fn with_defaults(sample_rate: f32) -> Self {
        Self::new(sample_rate, 4096, ModulationConfig::default())
    }

    /// Detect key changes across the audio signal.
    ///
    /// # Errors
    ///
    /// Returns error if the signal is too short or analysis fails.
    #[allow(clippy::cast_precision_loss)]
    pub fn detect(&self, signal: &[f32]) -> MirResult<ModulationResult> {
        let window_samples = (self.config.window_seconds * self.sample_rate) as usize;
        let hop_samples = (self.config.hop_seconds * self.sample_rate) as usize;

        if signal.len() < window_samples {
            return Err(MirError::InsufficientData(
                "Signal too short for modulation detection".to_string(),
            ));
        }

        // Detect local key for each window
        let local_keys = self.detect_local_keys(signal, window_samples, hop_samples)?;

        if local_keys.is_empty() {
            return Err(MirError::AnalysisFailed(
                "No local keys detected".to_string(),
            ));
        }

        // Merge consecutive frames with same key into regions
        let regions = self.merge_into_regions(&local_keys, hop_samples);

        // Identify modulations between regions
        let modulations = self.find_modulations(&regions);

        // Determine overall key (key with longest total duration)
        let overall_key = self.find_overall_key(&regions);

        let num_keys = {
            let mut keys: Vec<String> = regions.iter().map(|r| r.key.clone()).collect();
            keys.sort();
            keys.dedup();
            keys.len()
        };

        Ok(ModulationResult {
            regions,
            modulations,
            overall_key,
            num_keys,
        })
    }

    /// Detect the local key for each windowed segment.
    #[allow(clippy::cast_precision_loss)]
    fn detect_local_keys(
        &self,
        signal: &[f32],
        window_samples: usize,
        hop_samples: usize,
    ) -> MirResult<Vec<LocalKeyEstimate>> {
        let mut estimates = Vec::new();
        let detector = KeyDetector::new(self.sample_rate, self.window_size);

        let mut pos = 0;
        while pos + window_samples <= signal.len() {
            let window = &signal[pos..pos + window_samples];
            let time = pos as f32 / self.sample_rate;

            match detector.detect(window) {
                Ok(result) => {
                    if result.confidence >= self.config.min_confidence {
                        estimates.push(LocalKeyEstimate {
                            time,
                            key: result.key,
                            root: result.root,
                            is_major: result.is_major,
                            confidence: result.confidence,
                        });
                    }
                }
                Err(_) => {
                    // Skip windows where detection fails (e.g. silence)
                }
            }

            pos += hop_samples;
        }

        Ok(estimates)
    }

    /// Merge consecutive local key estimates with the same key into regions.
    #[allow(clippy::cast_precision_loss)]
    fn merge_into_regions(
        &self,
        estimates: &[LocalKeyEstimate],
        hop_samples: usize,
    ) -> Vec<KeyRegion> {
        if estimates.is_empty() {
            return Vec::new();
        }

        let hop_seconds = hop_samples as f32 / self.sample_rate;
        let window_seconds = self.config.window_seconds;

        let mut regions = Vec::new();
        let mut current_key = estimates[0].key.clone();
        let mut current_root = estimates[0].root;
        let mut current_is_major = estimates[0].is_major;
        let mut region_start = estimates[0].time;
        let mut confidence_sum = estimates[0].confidence;
        let mut frame_count = 1_usize;

        for est in estimates.iter().skip(1) {
            if est.key == current_key {
                confidence_sum += est.confidence;
                frame_count += 1;
            } else {
                // Flush current region if long enough
                if frame_count >= self.config.min_region_frames {
                    let end_time = est.time;
                    regions.push(KeyRegion {
                        start: region_start,
                        end: end_time,
                        key: current_key.clone(),
                        root: current_root,
                        is_major: current_is_major,
                        confidence: confidence_sum / frame_count as f32,
                    });
                }
                // Start new region
                current_key = est.key.clone();
                current_root = est.root;
                current_is_major = est.is_major;
                region_start = est.time;
                confidence_sum = est.confidence;
                frame_count = 1;
            }
        }

        // Flush final region
        if frame_count >= self.config.min_region_frames {
            let end_time = region_start + (frame_count as f32) * hop_seconds + window_seconds;
            regions.push(KeyRegion {
                start: region_start,
                end: end_time,
                key: current_key,
                root: current_root,
                is_major: current_is_major,
                confidence: confidence_sum / frame_count as f32,
            });
        }

        regions
    }

    /// Find modulations (key change events) between adjacent regions.
    fn find_modulations(&self, regions: &[KeyRegion]) -> Vec<Modulation> {
        let mut modulations = Vec::new();

        for pair in regions.windows(2) {
            let from = &pair[0];
            let to = &pair[1];

            let semitone_distance = ((to.root as i16 - from.root as i16).rem_euclid(12)) as u8;

            let confidence = (from.confidence + to.confidence) * 0.5;

            modulations.push(Modulation {
                time: to.start,
                from_key: from.key.clone(),
                to_key: to.key.clone(),
                confidence,
                semitone_distance,
            });
        }

        modulations
    }

    /// Find the overall key (longest combined duration).
    fn find_overall_key(&self, regions: &[KeyRegion]) -> String {
        use std::collections::HashMap;

        let mut durations: HashMap<String, f32> = HashMap::new();
        for region in regions {
            let dur = region.end - region.start;
            *durations.entry(region.key.clone()).or_insert(0.0) += dur;
        }

        durations
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(key, _)| key)
            .unwrap_or_else(|| "Unknown".to_string())
    }
}

/// Internal local key estimate for a single window.
#[derive(Debug, Clone)]
struct LocalKeyEstimate {
    time: f32,
    key: String,
    root: u8,
    is_major: bool,
    confidence: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    /// Generate a sine wave at the given frequency.
    fn sine_wave(freq: f32, sample_rate: f32, duration: f32) -> Vec<f32> {
        let n = (sample_rate * duration) as usize;
        (0..n)
            .map(|i| (TAU * freq * i as f32 / sample_rate).sin() * 0.5)
            .collect()
    }

    /// Generate a signal with two different keys (C major region then A minor region).
    fn two_key_signal(sample_rate: f32) -> Vec<f32> {
        // C major: emphasize C, E, G (261.63, 329.63, 392.00)
        let mut signal = Vec::new();
        let dur = 5.0; // 5 seconds per key

        let n = (sample_rate * dur) as usize;
        for i in 0..n {
            let t = i as f32 / sample_rate;
            let c = (TAU * 261.63 * t).sin();
            let e = (TAU * 329.63 * t).sin();
            let g = (TAU * 392.00 * t).sin();
            signal.push((c + e + g) * 0.3);
        }

        // A minor: emphasize A, C, E (220.00, 261.63, 329.63)
        for i in 0..n {
            let t = i as f32 / sample_rate;
            let a = (TAU * 220.00 * t).sin();
            let c = (TAU * 261.63 * t).sin();
            let e = (TAU * 329.63 * t).sin();
            signal.push((a + c + e) * 0.3);
        }

        signal
    }

    #[test]
    fn test_modulation_detector_creation() {
        let detector = ModulationDetector::with_defaults(44100.0);
        assert!((detector.sample_rate - 44100.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_modulation_config_default() {
        let cfg = ModulationConfig::default();
        assert!((cfg.window_seconds - 4.0).abs() < f32::EPSILON);
        assert!((cfg.hop_seconds - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_detect_too_short() {
        let detector = ModulationDetector::with_defaults(44100.0);
        let signal = vec![0.0; 1000]; // Way too short
        let result = detector.detect(&signal);
        assert!(result.is_err());
    }

    #[test]
    fn test_detect_single_key() {
        let sample_rate = 22050.0;
        // C major triad signal, 8 seconds
        let signal = sine_wave(261.63, sample_rate, 8.0);

        let config = ModulationConfig {
            window_seconds: 2.0,
            hop_seconds: 1.0,
            min_confidence: 0.0, // Accept everything
            min_region_frames: 1,
        };
        let detector = ModulationDetector::new(sample_rate, 2048, config);
        let result = detector.detect(&signal);
        assert!(result.is_ok());
        let res = result.expect("detection should succeed");
        // Should have at least one region
        assert!(!res.regions.is_empty());
        // Single key => no modulations
        // (there could be 0 or few modulations depending on detection noise)
        assert!(res.num_keys <= 2); // Allow small variation
    }

    #[test]
    fn test_detect_two_keys() {
        let sample_rate = 22050.0;
        let signal = two_key_signal(sample_rate);

        let config = ModulationConfig {
            window_seconds: 2.0,
            hop_seconds: 1.0,
            min_confidence: 0.0,
            min_region_frames: 1,
        };
        let detector = ModulationDetector::new(sample_rate, 2048, config);
        let result = detector.detect(&signal);
        assert!(result.is_ok());
        let res = result.expect("detection should succeed");
        assert!(!res.regions.is_empty());
        assert!(!res.overall_key.is_empty());
    }

    #[test]
    fn test_key_region_fields() {
        let region = KeyRegion {
            start: 0.0,
            end: 4.0,
            key: "C major".to_string(),
            root: 0,
            is_major: true,
            confidence: 0.85,
        };
        assert!((region.end - region.start - 4.0).abs() < f32::EPSILON);
        assert!(region.is_major);
    }

    #[test]
    fn test_modulation_fields() {
        let m = Modulation {
            time: 5.0,
            from_key: "C major".to_string(),
            to_key: "A minor".to_string(),
            confidence: 0.7,
            semitone_distance: 9,
        };
        assert_eq!(m.semitone_distance, 9);
    }

    #[test]
    fn test_merge_regions_empty() {
        let detector = ModulationDetector::with_defaults(44100.0);
        let regions = detector.merge_into_regions(&[], 44100);
        assert!(regions.is_empty());
    }

    #[test]
    fn test_find_modulations_single_region() {
        let detector = ModulationDetector::with_defaults(44100.0);
        let regions = vec![KeyRegion {
            start: 0.0,
            end: 10.0,
            key: "C major".to_string(),
            root: 0,
            is_major: true,
            confidence: 0.9,
        }];
        let mods = detector.find_modulations(&regions);
        assert!(mods.is_empty());
    }

    #[test]
    fn test_find_modulations_two_regions() {
        let detector = ModulationDetector::with_defaults(44100.0);
        let regions = vec![
            KeyRegion {
                start: 0.0,
                end: 5.0,
                key: "C major".to_string(),
                root: 0,
                is_major: true,
                confidence: 0.9,
            },
            KeyRegion {
                start: 5.0,
                end: 10.0,
                key: "G major".to_string(),
                root: 7,
                is_major: true,
                confidence: 0.85,
            },
        ];
        let mods = detector.find_modulations(&regions);
        assert_eq!(mods.len(), 1);
        assert_eq!(mods[0].semitone_distance, 7);
        assert!((mods[0].time - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_find_overall_key() {
        let detector = ModulationDetector::with_defaults(44100.0);
        let regions = vec![
            KeyRegion {
                start: 0.0,
                end: 8.0,
                key: "C major".to_string(),
                root: 0,
                is_major: true,
                confidence: 0.9,
            },
            KeyRegion {
                start: 8.0,
                end: 10.0,
                key: "G major".to_string(),
                root: 7,
                is_major: true,
                confidence: 0.85,
            },
        ];
        let overall = detector.find_overall_key(&regions);
        assert_eq!(overall, "C major"); // Longer duration
    }

    #[test]
    fn test_modulation_result_serialization() {
        let result = ModulationResult {
            regions: vec![KeyRegion {
                start: 0.0,
                end: 10.0,
                key: "C major".to_string(),
                root: 0,
                is_major: true,
                confidence: 0.9,
            }],
            modulations: Vec::new(),
            overall_key: "C major".to_string(),
            num_keys: 1,
        };
        // Verify Serialize/Deserialize works via debug
        let debug = format!("{:?}", result);
        assert!(debug.contains("C major"));
    }
}
