//! Noise type classification.
//!
//! Classifies audio noise into categories: white, pink, brown, hum, hiss,
//! rumble, click/impulse, broadband, environmental, or unknown.
//! Uses spectral shape, slope, temporal characteristics, and peak analysis.

use crate::spectral::SpectralFeatures;

/// Noise type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoiseType {
    /// White noise (flat spectrum, equal energy per Hz)
    White,
    /// Pink noise (1/f spectrum, equal energy per octave)
    Pink,
    /// Brown/red noise (1/f² spectrum, heavily low-frequency)
    Brown,
    /// Environmental noise (irregular, broadband, moderate variation)
    Environmental,
    /// Hum (power line interference at 50/60 Hz and harmonics)
    Hum,
    /// Hiss (high-frequency dominant noise, e.g. tape hiss, preamp noise)
    Hiss,
    /// Rumble (very low frequency noise, e.g. mechanical vibration, wind)
    Rumble,
    /// Click/impulse noise (transient, sparse, high crest factor)
    Click,
    /// Broadband noise (wide bandwidth, moderate flatness, not fitting other categories)
    Broadband,
    /// Unknown/other
    Unknown,
}

impl NoiseType {
    /// Return a human-readable label for this noise type.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            NoiseType::White => "White noise",
            NoiseType::Pink => "Pink noise (1/f)",
            NoiseType::Brown => "Brown noise (1/f²)",
            NoiseType::Environmental => "Environmental noise",
            NoiseType::Hum => "Hum (power line)",
            NoiseType::Hiss => "Hiss (high-frequency)",
            NoiseType::Rumble => "Rumble (low-frequency)",
            NoiseType::Click => "Click/impulse",
            NoiseType::Broadband => "Broadband noise",
            NoiseType::Unknown => "Unknown",
        }
    }
}

/// Detailed noise classification result with per-category scores.
#[derive(Debug, Clone)]
pub struct NoiseClassification {
    /// Primary noise type (highest score).
    pub primary: NoiseType,
    /// Confidence in the primary classification (0.0 - 1.0).
    pub confidence: f32,
    /// Scores for each noise type.
    pub scores: NoiseScores,
}

/// Per-category scores for noise classification.
#[derive(Debug, Clone)]
pub struct NoiseScores {
    /// White noise score
    pub white: f32,
    /// Pink noise score
    pub pink: f32,
    /// Brown noise score
    pub brown: f32,
    /// Environmental noise score
    pub environmental: f32,
    /// Hum score
    pub hum: f32,
    /// Hiss score
    pub hiss: f32,
    /// Rumble score
    pub rumble: f32,
    /// Click/impulse score
    pub click: f32,
    /// Broadband score
    pub broadband: f32,
}

/// Classify noise type from spectral features (simple API, backward-compatible).
///
/// # Arguments
/// * `spectral` - Spectral features of the noise
///
/// # Returns
/// Classified noise type
#[must_use]
pub fn classify_noise(spectral: &SpectralFeatures) -> NoiseType {
    classify_noise_detailed(spectral).primary
}

/// Classify noise type with detailed scores and confidence.
///
/// Uses multiple heuristics:
/// - Spectral flatness for white/colored noise distinction
/// - Spectral slope for pink/brown classification
/// - Spectral centroid for frequency-dominant noise (hiss vs rumble)
/// - Crest factor for transient/click detection
/// - Peak analysis for hum detection (harmonic series at 50/60 Hz)
///
/// # Arguments
/// * `spectral` - Spectral features of the noise
///
/// # Returns
/// Detailed classification with scores for each noise category
#[must_use]
pub fn classify_noise_detailed(spectral: &SpectralFeatures) -> NoiseClassification {
    let mut scores = NoiseScores {
        white: 0.0,
        pink: 0.0,
        brown: 0.0,
        environmental: 0.0,
        hum: 0.0,
        hiss: 0.0,
        rumble: 0.0,
        click: 0.0,
        broadband: 0.0,
    };

    let slope = estimate_spectral_slope(&spectral.magnitude_spectrum);
    let high_freq_ratio = compute_high_frequency_ratio(&spectral.magnitude_spectrum);
    let low_freq_ratio = compute_low_frequency_ratio(&spectral.magnitude_spectrum);
    let peak_harmonicity = detect_harmonic_peaks(&spectral.magnitude_spectrum);

    // ── White noise: very flat spectrum ──
    if spectral.flatness > 0.85 {
        scores.white = spectral.flatness;
    } else if spectral.flatness > 0.7 {
        scores.white = (spectral.flatness - 0.7) / 0.3 * 0.5;
    }

    // ── Hiss: high-frequency dominated, moderate-to-high flatness ──
    if high_freq_ratio > 0.6 && spectral.centroid > 3000.0 {
        scores.hiss = high_freq_ratio * 0.8;
        if spectral.flatness > 0.4 {
            scores.hiss += 0.2;
        }
        scores.hiss = scores.hiss.min(1.0);
    } else if spectral.centroid > 5000.0 && spectral.flatness > 0.3 {
        scores.hiss = 0.6;
    }

    // ── Rumble: very low frequency dominated ──
    if low_freq_ratio > 0.7 && spectral.centroid < 200.0 {
        scores.rumble = low_freq_ratio * 0.9;
        if spectral.bandwidth < 300.0 {
            scores.rumble = (scores.rumble + 0.1).min(1.0);
        }
    } else if spectral.centroid < 80.0 && spectral.flatness < 0.4 && spectral.bandwidth < 200.0 {
        scores.rumble = 0.7;
    }

    // ── Hum: harmonic peaks at 50/60 Hz multiples ──
    if spectral.centroid < 150.0 && spectral.flatness < 0.3 && peak_harmonicity > 0.5 {
        scores.hum = peak_harmonicity;
    } else if spectral.centroid < 100.0 && spectral.flatness < 0.2 {
        scores.hum = 0.6;
    }

    // ── Click: high crest factor (impulsive), low flatness ──
    if spectral.crest > 8.0 {
        scores.click = ((spectral.crest - 5.0) / 15.0).clamp(0.0, 1.0);
    } else if spectral.crest > 5.0 && spectral.flatness < 0.4 {
        scores.click = 0.4;
    }

    // ── Pink noise: -3 dB/octave slope ──
    {
        let pink_deviation = (slope - (-3.0)).abs();
        if pink_deviation < 1.5 && spectral.flatness > 0.2 && spectral.flatness < 0.85 {
            scores.pink = (1.0 - pink_deviation / 3.0).max(0.0);
        }
    }

    // ── Brown noise: -6 dB/octave slope ──
    {
        let brown_deviation = (slope - (-6.0)).abs();
        if brown_deviation < 1.5 && spectral.flatness > 0.1 && spectral.flatness < 0.7 {
            scores.brown = (1.0 - brown_deviation / 3.0).max(0.0);
        }
    }

    // ── Environmental: moderate flatness, irregular ──
    if spectral.flatness > 0.25
        && spectral.flatness < 0.7
        && spectral.bandwidth > 500.0
        && spectral.crest < 6.0
    {
        scores.environmental = 0.5;
        // Boost if spectral features are "average" (not extreme)
        if (200.0..=2000.0).contains(&spectral.centroid) {
            scores.environmental += 0.2;
        }
    }

    // ── Broadband: wide bandwidth, moderate characteristics ──
    if spectral.bandwidth > 2000.0
        && spectral.flatness > 0.3
        && spectral.flatness < 0.85
        && spectral.crest < 5.0
    {
        scores.broadband = 0.5;
        if spectral.flatness > 0.5 {
            scores.broadband += 0.2;
        }
    }

    // Find primary type and confidence
    let type_scores = [
        (NoiseType::White, scores.white),
        (NoiseType::Pink, scores.pink),
        (NoiseType::Brown, scores.brown),
        (NoiseType::Hum, scores.hum),
        (NoiseType::Hiss, scores.hiss),
        (NoiseType::Rumble, scores.rumble),
        (NoiseType::Click, scores.click),
        (NoiseType::Environmental, scores.environmental),
        (NoiseType::Broadband, scores.broadband),
    ];

    let (primary, confidence) = type_scores.iter().fold(
        (NoiseType::Unknown, 0.0_f32),
        |(best_type, best_score), &(noise_type, score)| {
            if score > best_score {
                (noise_type, score)
            } else {
                (best_type, best_score)
            }
        },
    );

    // If no strong match, classify as Unknown
    let (primary, confidence) = if confidence < 0.2 {
        (NoiseType::Unknown, confidence)
    } else {
        (primary, confidence)
    };

    NoiseClassification {
        primary,
        confidence,
        scores,
    }
}

/// Classify noise from raw audio samples with a given sample rate.
///
/// Computes spectral features internally and classifies.
///
/// # Arguments
/// * `samples` - Audio samples (mono)
/// * `sample_rate` - Sample rate in Hz
///
/// # Returns
/// Detailed noise classification result, or error if analysis fails.
pub fn classify_noise_from_samples(
    samples: &[f32],
    sample_rate: f32,
) -> crate::Result<NoiseClassification> {
    let config = crate::AnalysisConfig::default();
    let analyzer = crate::spectral::SpectralAnalyzer::new(config);
    let features = analyzer.analyze(samples, sample_rate)?;
    Ok(classify_noise_detailed(&features))
}

/// Estimate spectral slope (in dB/octave).
fn estimate_spectral_slope(spectrum: &[f32]) -> f32 {
    if spectrum.len() < 10 {
        return 0.0;
    }

    // Divide spectrum into logarithmic bins (octaves)
    let num_bins = 6;
    let mut bin_energies = vec![0.0; num_bins];
    let mut bin_counts = vec![0; num_bins];

    for (i, &mag) in spectrum.iter().enumerate() {
        if i > 0 {
            let octave = (i as f32).log2() as usize;
            if octave < num_bins {
                bin_energies[octave] += mag * mag;
                bin_counts[octave] += 1;
            }
        }
    }

    // Average energy per bin
    for i in 0..num_bins {
        if bin_counts[i] > 0 {
            bin_energies[i] /= bin_counts[i] as f32;
        }
    }

    // Compute slope via linear regression in log space
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_xx = 0.0;
    let mut n = 0;

    for (i, &energy) in bin_energies.iter().enumerate() {
        if energy > 0.0 {
            let x = i as f32;
            let y = 10.0 * energy.log10();

            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
            n += 1;
        }
    }

    if n > 1 {
        let denom = n as f32 * sum_xx - sum_x * sum_x;
        if denom.abs() < f32::EPSILON {
            0.0
        } else {
            (n as f32 * sum_xy - sum_x * sum_y) / denom
        }
    } else {
        0.0
    }
}

/// Compute the ratio of energy in the upper half of the spectrum.
fn compute_high_frequency_ratio(spectrum: &[f32]) -> f32 {
    if spectrum.len() < 4 {
        return 0.0;
    }
    let mid = spectrum.len() / 2;
    let total_energy: f32 = spectrum.iter().map(|&m| m * m).sum();
    if total_energy < f32::EPSILON {
        return 0.0;
    }
    let high_energy: f32 = spectrum[mid..].iter().map(|&m| m * m).sum();
    (high_energy / total_energy).clamp(0.0, 1.0)
}

/// Compute the ratio of energy in the lower quarter of the spectrum.
fn compute_low_frequency_ratio(spectrum: &[f32]) -> f32 {
    if spectrum.len() < 4 {
        return 0.0;
    }
    let quarter = spectrum.len() / 4;
    let total_energy: f32 = spectrum.iter().map(|&m| m * m).sum();
    if total_energy < f32::EPSILON {
        return 0.0;
    }
    let low_energy: f32 = spectrum[..quarter].iter().map(|&m| m * m).sum();
    (low_energy / total_energy).clamp(0.0, 1.0)
}

/// Detect harmonic peak structure (characteristic of hum).
///
/// Looks for peaks at roughly evenly-spaced intervals in the spectrum,
/// which is characteristic of power line hum (50/60 Hz and harmonics).
fn detect_harmonic_peaks(spectrum: &[f32]) -> f32 {
    if spectrum.len() < 20 {
        return 0.0;
    }

    // Find the top peaks in the spectrum
    let mut peaks: Vec<(usize, f32)> = Vec::new();
    for i in 1..(spectrum.len() - 1) {
        if spectrum[i] > spectrum[i - 1] && spectrum[i] > spectrum[i + 1] && spectrum[i] > 0.01 {
            peaks.push((i, spectrum[i]));
        }
    }

    if peaks.len() < 3 {
        return 0.0;
    }

    // Sort by magnitude (descending) and take top peaks
    peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_peaks: Vec<usize> = peaks.iter().take(8).map(|&(i, _)| i).collect();

    if top_peaks.len() < 3 {
        return 0.0;
    }

    // Check if peaks form a harmonic series (roughly integer multiples of fundamental)
    let fundamental = *top_peaks.iter().filter(|&&p| p > 0).min().unwrap_or(&1);

    if fundamental == 0 {
        return 0.0;
    }

    let mut harmonic_count = 0;
    for &peak in &top_peaks {
        let ratio = peak as f32 / fundamental as f32;
        let nearest_int = ratio.round();
        let deviation = (ratio - nearest_int).abs();
        if deviation < 0.15 && nearest_int >= 1.0 {
            harmonic_count += 1;
        }
    }

    (harmonic_count as f32 / top_peaks.len() as f32).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_spectral(
        centroid: f32,
        flatness: f32,
        crest: f32,
        bandwidth: f32,
        spectrum: Vec<f32>,
    ) -> SpectralFeatures {
        SpectralFeatures {
            centroid,
            flatness,
            crest,
            bandwidth,
            rolloff: centroid * 2.0,
            flux: 0.0,
            magnitude_spectrum: spectrum,
        }
    }

    #[test]
    fn test_noise_classification_white() {
        let spectral = make_spectral(1000.0, 0.95, 1.5, 2000.0, vec![1.0; 100]);
        assert_eq!(classify_noise(&spectral), NoiseType::White);
    }

    #[test]
    fn test_noise_classification_hum() {
        // Create spectrum with harmonic peaks at 50 Hz multiples
        let mut spectrum = vec![0.01_f32; 200];
        spectrum[5] = 1.0; // ~50 Hz bin
        spectrum[10] = 0.7; // ~100 Hz
        spectrum[15] = 0.5; // ~150 Hz
        spectrum[20] = 0.3; // ~200 Hz
        let spectral = make_spectral(60.0, 0.1, 5.0, 50.0, spectrum);
        assert_eq!(classify_noise(&spectral), NoiseType::Hum);
    }

    #[test]
    fn test_noise_classification_hiss() {
        // High-frequency dominated spectrum
        let mut spectrum = vec![0.01_f32; 200];
        for i in 100..200 {
            spectrum[i] = 0.8;
        }
        let spectral = make_spectral(6000.0, 0.5, 2.0, 4000.0, spectrum);
        let result = classify_noise_detailed(&spectral);
        assert_eq!(result.primary, NoiseType::Hiss);
        assert!(result.confidence > 0.3);
    }

    #[test]
    fn test_noise_classification_rumble() {
        // Very low frequency dominated
        let mut spectrum = vec![0.01_f32; 200];
        for i in 0..20 {
            spectrum[i] = 1.0;
        }
        let spectral = make_spectral(50.0, 0.15, 3.0, 100.0, spectrum);
        let result = classify_noise_detailed(&spectral);
        assert_eq!(result.primary, NoiseType::Rumble);
        assert!(result.confidence > 0.3);
    }

    #[test]
    fn test_noise_classification_click() {
        let spectral = make_spectral(2000.0, 0.2, 12.0, 3000.0, vec![0.1; 100]);
        let result = classify_noise_detailed(&spectral);
        assert_eq!(result.primary, NoiseType::Click);
        assert!(result.confidence > 0.3);
    }

    #[test]
    fn test_noise_classification_broadband() {
        // High flatness, wide bandwidth, low crest — clearly broadband
        let spectral = make_spectral(3000.0, 0.75, 1.8, 5000.0, vec![0.5; 200]);
        let result = classify_noise_detailed(&spectral);
        assert_eq!(result.primary, NoiseType::Broadband);
        assert!(result.confidence > 0.3);
    }

    #[test]
    fn test_noise_type_labels() {
        assert_eq!(NoiseType::White.label(), "White noise");
        assert_eq!(NoiseType::Hiss.label(), "Hiss (high-frequency)");
        assert_eq!(NoiseType::Rumble.label(), "Rumble (low-frequency)");
        assert_eq!(NoiseType::Click.label(), "Click/impulse");
        assert_eq!(NoiseType::Broadband.label(), "Broadband noise");
        assert_eq!(NoiseType::Hum.label(), "Hum (power line)");
        assert_eq!(NoiseType::Pink.label(), "Pink noise (1/f)");
        assert_eq!(NoiseType::Brown.label(), "Brown noise (1/f²)");
        assert_eq!(NoiseType::Environmental.label(), "Environmental noise");
        assert_eq!(NoiseType::Unknown.label(), "Unknown");
    }

    #[test]
    fn test_detailed_classification_has_all_scores() {
        let spectral = make_spectral(1000.0, 0.95, 1.5, 2000.0, vec![1.0; 100]);
        let result = classify_noise_detailed(&spectral);
        // White noise should have highest score
        assert!(result.scores.white > result.scores.pink);
        assert!(result.scores.white > result.scores.hum);
    }

    #[test]
    fn test_high_frequency_ratio() {
        let mut spectrum = vec![0.0_f32; 100];
        for s in spectrum[50..].iter_mut() {
            *s = 1.0;
        }
        let ratio = compute_high_frequency_ratio(&spectrum);
        assert!(ratio > 0.9, "High freq ratio should be ~1.0, got {ratio}");
    }

    #[test]
    fn test_low_frequency_ratio() {
        let mut spectrum = vec![0.0_f32; 100];
        for s in spectrum[..25].iter_mut() {
            *s = 1.0;
        }
        let ratio = compute_low_frequency_ratio(&spectrum);
        assert!(ratio > 0.9, "Low freq ratio should be ~1.0, got {ratio}");
    }

    #[test]
    fn test_harmonic_peaks_detection() {
        // Create spectrum with harmonic peaks at bins 10, 20, 30, 40
        let mut spectrum = vec![0.01_f32; 100];
        spectrum[10] = 1.0;
        spectrum[20] = 0.8;
        spectrum[30] = 0.6;
        spectrum[40] = 0.4;
        let score = detect_harmonic_peaks(&spectrum);
        assert!(
            score > 0.5,
            "Harmonic peaks should be detected, score={score}"
        );
    }

    #[test]
    fn test_harmonic_peaks_no_harmonics() {
        // Random non-harmonic peaks
        let mut spectrum = vec![0.01_f32; 100];
        spectrum[7] = 1.0;
        spectrum[23] = 0.8;
        spectrum[53] = 0.6;
        spectrum[91] = 0.4;
        let score = detect_harmonic_peaks(&spectrum);
        assert!(
            score < 0.8,
            "Non-harmonic peaks should have lower score: {score}"
        );
    }

    #[test]
    fn test_classify_noise_from_samples() {
        // Generate white-ish noise (constant signal gives flat spectrum approximately)
        let samples: Vec<f32> = (0..4096)
            .map(|i| ((i as f32 * 0.1).sin() + (i as f32 * 0.37).sin()) * 0.5)
            .collect();
        let result = classify_noise_from_samples(&samples, 44100.0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_empty_spectrum_no_panic() {
        let spectral = make_spectral(0.0, 0.0, 0.0, 0.0, vec![]);
        let result = classify_noise_detailed(&spectral);
        // Should not panic, return Unknown
        assert!(result.confidence <= 1.0);
    }

    #[test]
    fn test_pink_noise_spectrum() {
        // Create 1/f spectrum (pink noise approximation)
        let spectrum: Vec<f32> = (1..=200)
            .map(|i| 1.0 / (i as f32).sqrt()) // 1/sqrt(f) power => -3dB/oct magnitude
            .collect();
        let spectral = make_spectral(500.0, 0.4, 2.0, 1500.0, spectrum);
        let result = classify_noise_detailed(&spectral);
        // Should have non-zero pink score
        assert!(
            result.scores.pink > 0.0 || result.scores.brown > 0.0,
            "1/f spectrum should score as pink or brown"
        );
    }
}
