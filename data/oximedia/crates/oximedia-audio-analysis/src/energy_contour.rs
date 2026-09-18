#![allow(dead_code)]
//! Energy contour analysis for audio signals.
//!
//! This module computes and analyzes the energy envelope of an audio signal over
//! time. It provides short-time energy tracking, smoothing, segmentation by
//! energy level, and detection of energy transients (sudden rises or drops).

/// Configuration for energy contour computation.
#[derive(Debug, Clone)]
pub struct EnergyContourConfig {
    /// Frame length in samples.
    pub frame_length: usize,
    /// Hop size in samples.
    pub hop_size: usize,
    /// Smoothing window radius (0 = no smoothing).
    pub smoothing_radius: usize,
    /// Threshold in dB below peak for "active" detection.
    pub active_threshold_db: f64,
}

impl Default for EnergyContourConfig {
    fn default() -> Self {
        Self {
            frame_length: 1024,
            hop_size: 512,
            smoothing_radius: 3,
            active_threshold_db: -30.0,
        }
    }
}

/// A single point in the energy contour.
#[derive(Debug, Clone, Copy)]
pub struct EnergyPoint {
    /// Frame index.
    pub frame: usize,
    /// Time position in seconds.
    pub time_seconds: f64,
    /// RMS energy (linear).
    pub rms: f64,
    /// Energy in dB (relative to full scale).
    pub db: f64,
}

/// Result of energy contour analysis.
#[derive(Debug, Clone)]
pub struct EnergyContour {
    /// Per-frame energy points.
    pub points: Vec<EnergyPoint>,
    /// Peak energy in dB.
    pub peak_db: f64,
    /// Mean energy in dB.
    pub mean_db: f64,
    /// Minimum energy in dB.
    pub min_db: f64,
    /// Dynamic range in dB (peak - min among active frames).
    pub dynamic_range_db: f64,
}

/// A segment classified by energy level.
#[derive(Debug, Clone)]
pub struct EnergySegment {
    /// Start frame index.
    pub start_frame: usize,
    /// End frame index (exclusive).
    pub end_frame: usize,
    /// Start time in seconds.
    pub start_time: f64,
    /// End time in seconds.
    pub end_time: f64,
    /// Classification of the segment.
    pub classification: EnergyClass,
    /// Mean energy in dB over this segment.
    pub mean_db: f64,
}

/// Classification of an energy segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnergyClass {
    /// Very quiet / silence.
    Silence,
    /// Quiet but audible.
    Quiet,
    /// Moderate energy.
    Moderate,
    /// Loud.
    Loud,
}

/// An energy transient (sudden change in energy).
#[derive(Debug, Clone, Copy)]
pub struct EnergyTransient {
    /// Frame index where the transient occurs.
    pub frame: usize,
    /// Time in seconds.
    pub time_seconds: f64,
    /// Change in dB.
    pub delta_db: f64,
    /// Whether this is a rise (true) or drop (false).
    pub is_rise: bool,
}

/// Compute RMS energy for a single frame.
#[allow(clippy::cast_precision_loss)]
fn frame_rms(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f64 = samples.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
    (sum / samples.len() as f64).sqrt()
}

/// Convert linear amplitude to dB (with floor).
fn linear_to_db(val: f64) -> f64 {
    if val > 1e-15 {
        20.0 * val.log10()
    } else {
        -300.0
    }
}

/// Apply simple moving average smoothing.
#[allow(clippy::cast_precision_loss)]
fn smooth(values: &[f64], radius: usize) -> Vec<f64> {
    if radius == 0 {
        return values.to_vec();
    }
    let n = values.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let lo = i.saturating_sub(radius);
        let hi = (i + radius + 1).min(n);
        let count = (hi - lo) as f64;
        let sum: f64 = values[lo..hi].iter().sum();
        out.push(sum / count);
    }
    out
}

/// Compute the energy contour of an audio signal.
#[allow(clippy::cast_precision_loss)]
pub fn compute_contour(
    samples: &[f32],
    sample_rate: f64,
    config: &EnergyContourConfig,
) -> EnergyContour {
    let mut rms_values: Vec<f64> = Vec::new();
    let mut pos = 0_usize;
    while pos + config.frame_length <= samples.len() {
        let frame = &samples[pos..pos + config.frame_length];
        rms_values.push(frame_rms(frame));
        pos += config.hop_size;
    }

    // Smooth
    let smoothed = smooth(&rms_values, config.smoothing_radius);

    // Build points
    let points: Vec<EnergyPoint> = smoothed
        .iter()
        .enumerate()
        .map(|(i, &rms)| {
            let time = (i * config.hop_size) as f64 / sample_rate;
            let db = linear_to_db(rms);
            EnergyPoint {
                frame: i,
                time_seconds: time,
                rms,
                db,
            }
        })
        .collect();

    let peak_db = points
        .iter()
        .map(|p| p.db)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_db = points.iter().map(|p| p.db).fold(f64::INFINITY, f64::min);
    let mean_db = if points.is_empty() {
        -300.0
    } else {
        points.iter().map(|p| p.db).sum::<f64>() / points.len() as f64
    };

    // Dynamic range: among active frames (above threshold)
    let active_thresh = peak_db + config.active_threshold_db;
    let active_dbs: Vec<f64> = points
        .iter()
        .filter(|p| p.db >= active_thresh)
        .map(|p| p.db)
        .collect();
    let dynamic_range_db = if active_dbs.len() >= 2 {
        let amax = active_dbs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let amin = active_dbs.iter().copied().fold(f64::INFINITY, f64::min);
        amax - amin
    } else {
        0.0
    };

    EnergyContour {
        points,
        peak_db,
        mean_db,
        min_db,
        dynamic_range_db,
    }
}

/// Segment the energy contour into classified regions.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn segment_by_energy(
    contour: &EnergyContour,
    silence_threshold_db: f64,
    quiet_threshold_db: f64,
    loud_threshold_db: f64,
    hop_size: usize,
    sample_rate: f64,
) -> Vec<EnergySegment> {
    if contour.points.is_empty() {
        return Vec::new();
    }

    let classify = |db: f64| -> EnergyClass {
        if db < silence_threshold_db {
            EnergyClass::Silence
        } else if db < quiet_threshold_db {
            EnergyClass::Quiet
        } else if db < loud_threshold_db {
            EnergyClass::Moderate
        } else {
            EnergyClass::Loud
        }
    };

    let mut segments = Vec::new();
    let mut start = 0_usize;
    let mut current_class = classify(contour.points[0].db);
    let mut db_sum = contour.points[0].db;
    let mut count = 1_usize;

    for i in 1..contour.points.len() {
        let cls = classify(contour.points[i].db);
        if cls == current_class {
            db_sum += contour.points[i].db;
            count += 1;
        } else {
            segments.push(EnergySegment {
                start_frame: start,
                end_frame: i,
                start_time: (start * hop_size) as f64 / sample_rate,
                end_time: (i * hop_size) as f64 / sample_rate,
                classification: current_class,
                mean_db: db_sum / count as f64,
            });
            start = i;
            current_class = cls;
            db_sum = contour.points[i].db;
            count = 1;
        }
    }

    // Final segment
    let n = contour.points.len();
    segments.push(EnergySegment {
        start_frame: start,
        end_frame: n,
        start_time: (start * hop_size) as f64 / sample_rate,
        end_time: (n * hop_size) as f64 / sample_rate,
        classification: current_class,
        mean_db: db_sum / count as f64,
    });

    segments
}

/// Detect energy transients (sudden rises or drops).
#[must_use]
pub fn detect_transients(contour: &EnergyContour, threshold_db: f64) -> Vec<EnergyTransient> {
    if contour.points.len() < 2 {
        return Vec::new();
    }

    let mut transients = Vec::new();
    for i in 1..contour.points.len() {
        let delta = contour.points[i].db - contour.points[i - 1].db;
        if delta.abs() >= threshold_db {
            transients.push(EnergyTransient {
                frame: i,
                time_seconds: contour.points[i].time_seconds,
                delta_db: delta,
                is_rise: delta > 0.0,
            });
        }
    }
    transients
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_signal(freq: f64, sample_rate: f64, duration: f64, amplitude: f32) -> Vec<f32> {
        let n = (sample_rate * duration) as usize;
        (0..n)
            .map(|i| {
                let t = i as f64 / sample_rate;
                #[allow(clippy::cast_possible_truncation)]
                let sample =
                    (amplitude as f64 * (2.0 * std::f64::consts::PI * freq * t).sin()) as f32;
                sample
            })
            .collect()
    }

    #[test]
    fn test_frame_rms_silence() {
        let samples = vec![0.0_f32; 1024];
        assert!((frame_rms(&samples)).abs() < 1e-10);
    }

    #[test]
    fn test_frame_rms_unity() {
        let samples = vec![1.0_f32; 100];
        assert!((frame_rms(&samples) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_frame_rms_empty() {
        assert!((frame_rms(&[])).abs() < 1e-10);
    }

    #[test]
    fn test_linear_to_db() {
        assert!((linear_to_db(1.0)).abs() < 1e-10);
        assert!((linear_to_db(0.1) - (-20.0)).abs() < 0.01);
    }

    #[test]
    fn test_linear_to_db_floor() {
        assert!(linear_to_db(0.0) < -200.0);
    }

    #[test]
    fn test_smooth_no_radius() {
        let vals = vec![1.0, 2.0, 3.0];
        let out = smooth(&vals, 0);
        assert_eq!(out, vals);
    }

    #[test]
    fn test_smooth_with_radius() {
        let vals = vec![0.0, 0.0, 10.0, 0.0, 0.0];
        let out = smooth(&vals, 1);
        // Center value: (0 + 10 + 0) / 3 ≈ 3.33
        assert!((out[2] - 10.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_contour_basic() {
        let signal = sine_signal(440.0, 16000.0, 0.5, 0.5);
        let config = EnergyContourConfig {
            frame_length: 512,
            hop_size: 256,
            smoothing_radius: 0,
            active_threshold_db: -30.0,
        };
        let contour = compute_contour(&signal, 16000.0, &config);
        assert!(!contour.points.is_empty());
        assert!(contour.peak_db > -20.0);
    }

    #[test]
    fn test_contour_silence_has_low_energy() {
        let signal = vec![0.0_f32; 8000];
        let config = EnergyContourConfig::default();
        let contour = compute_contour(&signal, 16000.0, &config);
        for p in &contour.points {
            assert!(p.db < -100.0);
        }
    }

    #[test]
    fn test_segment_by_energy_single_class() {
        let signal = vec![0.5_f32; 16000];
        let config = EnergyContourConfig {
            frame_length: 512,
            hop_size: 256,
            smoothing_radius: 0,
            active_threshold_db: -30.0,
        };
        let contour = compute_contour(&signal, 16000.0, &config);
        let segments = segment_by_energy(&contour, -80.0, -40.0, -6.0, 256, 16000.0);
        assert!(!segments.is_empty());
        // All frames have same DC level, so should be one segment
        assert_eq!(segments.len(), 1);
    }

    #[test]
    fn test_detect_transients_sudden_onset() {
        // Silence then loud
        let mut signal = vec![0.0001_f32; 8000];
        signal.extend(vec![0.5_f32; 8000]);
        let config = EnergyContourConfig {
            frame_length: 512,
            hop_size: 256,
            smoothing_radius: 0,
            active_threshold_db: -30.0,
        };
        let contour = compute_contour(&signal, 16000.0, &config);
        let transients = detect_transients(&contour, 10.0);
        assert!(!transients.is_empty());
        // At least one rise
        assert!(transients.iter().any(|t| t.is_rise));
    }

    #[test]
    fn test_detect_transients_none_for_steady() {
        let signal = vec![0.3_f32; 16000];
        let config = EnergyContourConfig {
            frame_length: 512,
            hop_size: 256,
            smoothing_radius: 0,
            active_threshold_db: -30.0,
        };
        let contour = compute_contour(&signal, 16000.0, &config);
        let transients = detect_transients(&contour, 10.0);
        assert!(transients.is_empty());
    }

    #[test]
    fn test_energy_contour_default_config() {
        let config = EnergyContourConfig::default();
        assert_eq!(config.frame_length, 1024);
        assert_eq!(config.hop_size, 512);
        assert_eq!(config.smoothing_radius, 3);
    }
}
