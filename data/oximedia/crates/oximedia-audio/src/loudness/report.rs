//! Loudness measurement reporting and visualization.
//!
//! Provides detailed reports, statistics, and compliance checking
//! for loudness measurements.

#![forbid(unsafe_code)]
#![allow(clippy::cast_lossless)]

use super::normalize::NormalizationParams;
use super::r128::{ComplianceStatus, R128Compliance, R128Meter};
use std::fmt;

/// Comprehensive loudness measurement report.
#[derive(Clone, Debug)]
pub struct LoudnessReport {
    /// Integrated loudness in LUFS.
    pub integrated_lufs: f64,
    /// Loudness range in LU.
    pub loudness_range: f64,
    /// Maximum momentary loudness in LUFS.
    pub max_momentary: f64,
    /// Maximum short-term loudness in LUFS.
    pub max_short_term: f64,
    /// True peak in dBTP.
    pub true_peak_dbtp: f64,
    /// Per-channel true peaks in dBTP.
    pub channel_peaks_dbtp: Vec<f64>,
    /// Sample rate in Hz.
    pub sample_rate: f64,
    /// Number of channels.
    pub channels: usize,
    /// Duration in seconds.
    pub duration_seconds: f64,
    /// Compliance status for EBU R128.
    pub ebu_compliance: EbuR128Compliance,
    /// Compliance status for ATSC A/85.
    pub atsc_compliance: AtscA85Compliance,
}

impl LoudnessReport {
    /// Create a loudness report from an R128 meter.
    ///
    /// # Arguments
    ///
    /// * `meter` - R128 loudness meter
    /// * `duration_seconds` - Total duration in seconds
    #[must_use]
    pub fn from_meter(meter: &R128Meter, duration_seconds: f64) -> Self {
        let integrated_lufs = meter.integrated_loudness();
        let loudness_range = meter.loudness_range();
        let max_momentary = meter.max_momentary();
        let max_short_term = meter.max_short_term();
        let true_peak_dbtp = meter.true_peak_dbtp();

        let channel_peaks = meter.channel_peaks();
        let channel_peaks_dbtp: Vec<f64> = channel_peaks
            .iter()
            .map(|&p| super::peak::TruePeakDetector::linear_to_dbtp(p))
            .collect();

        let ebu_compliance = EbuR128Compliance {
            program_loudness: R128Compliance::check_program_loudness(integrated_lufs),
            true_peak_ok: R128Compliance::check_true_peak(true_peak_dbtp),
            loudness_range_ok: R128Compliance::check_loudness_range(loudness_range),
            recommended_gain: R128Compliance::recommended_gain_adjustment(integrated_lufs, -23.0),
        };

        let atsc_target = -24.0;
        let atsc_tolerance = 2.0;
        let atsc_status = if integrated_lufs.is_finite()
            && integrated_lufs >= atsc_target - atsc_tolerance
            && integrated_lufs <= atsc_target + atsc_tolerance
        {
            ComplianceStatus::Compliant
        } else if integrated_lufs > atsc_target + atsc_tolerance {
            ComplianceStatus::TooLoud(integrated_lufs - atsc_target)
        } else if integrated_lufs.is_finite() {
            ComplianceStatus::TooQuiet(atsc_target - integrated_lufs)
        } else {
            ComplianceStatus::Unknown
        };

        let atsc_compliance = AtscA85Compliance {
            program_loudness: atsc_status,
            recommended_gain: if integrated_lufs.is_finite() {
                atsc_target - integrated_lufs
            } else {
                0.0
            },
        };

        Self {
            integrated_lufs,
            loudness_range,
            max_momentary,
            max_short_term,
            true_peak_dbtp,
            channel_peaks_dbtp,
            sample_rate: meter.sample_rate(),
            channels: meter.channels(),
            duration_seconds,
            ebu_compliance,
            atsc_compliance,
        }
    }

    /// Generate a human-readable text report.
    #[must_use]
    pub fn to_text(&self) -> String {
        let mut report = String::new();

        report.push_str("=== Loudness Measurement Report ===\n\n");

        // Audio properties
        report.push_str("Audio Properties:\n");
        report.push_str(&format!("  Sample Rate: {} Hz\n", self.sample_rate));
        report.push_str(&format!("  Channels: {}\n", self.channels));
        report.push_str(&format!(
            "  Duration: {:.2} seconds\n\n",
            self.duration_seconds
        ));

        // Loudness measurements
        report.push_str("Loudness Measurements:\n");
        report.push_str(&format!("  Integrated: {:.1} LUFS\n", self.integrated_lufs));
        report.push_str(&format!(
            "  Loudness Range: {:.1} LU\n",
            self.loudness_range
        ));
        report.push_str(&format!(
            "  Maximum Momentary: {:.1} LUFS\n",
            self.max_momentary
        ));
        report.push_str(&format!(
            "  Maximum Short-term: {:.1} LUFS\n\n",
            self.max_short_term
        ));

        // Peak measurements
        report.push_str("Peak Measurements:\n");
        report.push_str(&format!("  True Peak: {:.1} dBTP\n", self.true_peak_dbtp));
        for (i, &peak) in self.channel_peaks_dbtp.iter().enumerate() {
            report.push_str(&format!("  Channel {}: {:.1} dBTP\n", i + 1, peak));
        }
        report.push('\n');

        // EBU R128 compliance
        report.push_str("EBU R128 Compliance:\n");
        report.push_str(&format!(
            "  Target: -23 LUFS ±1 LU\n  Status: {}\n",
            self.format_compliance_status(&self.ebu_compliance.program_loudness)
        ));
        report.push_str(&format!(
            "  True Peak: {} (requirement: ≤ -1.0 dBTP)\n",
            if self.ebu_compliance.true_peak_ok {
                "PASS"
            } else {
                "FAIL"
            }
        ));
        report.push_str(&format!(
            "  Recommended Gain: {:+.1} dB\n\n",
            self.ebu_compliance.recommended_gain
        ));

        // ATSC A/85 compliance
        report.push_str("ATSC A/85 Compliance:\n");
        report.push_str(&format!(
            "  Target: -24 LKFS ±2 dB\n  Status: {}\n",
            self.format_compliance_status(&self.atsc_compliance.program_loudness)
        ));
        report.push_str(&format!(
            "  Recommended Gain: {:+.1} dB\n\n",
            self.atsc_compliance.recommended_gain
        ));

        report
    }

    /// Generate a JSON report.
    #[must_use]
    pub fn to_json(&self) -> String {
        format!(
            r#"{{
  "integrated_lufs": {:.2},
  "loudness_range": {:.2},
  "max_momentary": {:.2},
  "max_short_term": {:.2},
  "true_peak_dbtp": {:.2},
  "channel_peaks_dbtp": {:?},
  "sample_rate": {},
  "channels": {},
  "duration_seconds": {:.2},
  "ebu_r128": {{
    "compliant": {},
    "recommended_gain_db": {:.2}
  }},
  "atsc_a85": {{
    "compliant": {},
    "recommended_gain_db": {:.2}
  }}
}}"#,
            self.integrated_lufs,
            self.loudness_range,
            self.max_momentary,
            self.max_short_term,
            self.true_peak_dbtp,
            self.channel_peaks_dbtp,
            self.sample_rate,
            self.channels,
            self.duration_seconds,
            self.ebu_compliance.program_loudness.is_compliant(),
            self.ebu_compliance.recommended_gain,
            self.atsc_compliance.program_loudness.is_compliant(),
            self.atsc_compliance.recommended_gain,
        )
    }

    /// Format compliance status as a string.
    fn format_compliance_status(&self, status: &ComplianceStatus) -> String {
        match status {
            ComplianceStatus::Compliant => "COMPLIANT".to_string(),
            ComplianceStatus::TooLoud(db) => format!("TOO LOUD by {:.1} dB", db),
            ComplianceStatus::TooQuiet(db) => format!("TOO QUIET by {:.1} dB", db),
            ComplianceStatus::Unknown => "UNKNOWN".to_string(),
        }
    }

    /// Check if compliant with both EBU R128 and ATSC A/85.
    #[must_use]
    pub fn is_broadcast_compliant(&self) -> bool {
        self.ebu_compliance.program_loudness.is_compliant()
            && self.ebu_compliance.true_peak_ok
            && self.atsc_compliance.program_loudness.is_compliant()
    }
}

impl fmt::Display for LoudnessReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_text())
    }
}

/// EBU R128 compliance information.
#[derive(Clone, Debug)]
pub struct EbuR128Compliance {
    /// Program loudness compliance status.
    pub program_loudness: ComplianceStatus,
    /// True peak compliance (≤ -1.0 dBTP).
    pub true_peak_ok: bool,
    /// Loudness range acceptable.
    pub loudness_range_ok: bool,
    /// Recommended gain adjustment in dB.
    pub recommended_gain: f64,
}

/// ATSC A/85 compliance information.
#[derive(Clone, Debug)]
pub struct AtscA85Compliance {
    /// Program loudness compliance status.
    pub program_loudness: ComplianceStatus,
    /// Recommended gain adjustment in dB.
    pub recommended_gain: f64,
}

/// Normalization report combining measurement and normalization parameters.
#[derive(Clone, Debug)]
pub struct NormalizationReport {
    /// Original loudness report.
    pub original: LoudnessReport,
    /// Normalization parameters applied.
    pub normalization: NormalizationParams,
    /// Target standard.
    pub target_standard: String,
}

impl NormalizationReport {
    /// Create a normalization report.
    ///
    /// # Arguments
    ///
    /// * `original` - Original loudness report
    /// * `normalization` - Normalization parameters
    /// * `target_standard` - Target standard name (e.g., "EBU R128")
    #[must_use]
    pub fn new(
        original: LoudnessReport,
        normalization: NormalizationParams,
        target_standard: String,
    ) -> Self {
        Self {
            original,
            normalization,
            target_standard,
        }
    }

    /// Generate a text report.
    #[must_use]
    pub fn to_text(&self) -> String {
        let mut report = String::new();

        report.push_str("=== Loudness Normalization Report ===\n\n");

        // Original measurements
        report.push_str("Original Measurements:\n");
        report.push_str(&format!(
            "  Integrated Loudness: {:.1} LUFS\n",
            self.original.integrated_lufs
        ));
        report.push_str(&format!(
            "  True Peak: {:.1} dBTP\n",
            self.original.true_peak_dbtp
        ));
        report.push_str(&format!(
            "  Loudness Range: {:.1} LU\n\n",
            self.original.loudness_range
        ));

        // Normalization parameters
        report.push_str(&format!("Target Standard: {}\n", self.target_standard));
        report.push_str(&format!(
            "Target Loudness: {:.1} LUFS\n\n",
            self.normalization.target_lufs
        ));

        report.push_str("Normalization Parameters:\n");
        report.push_str(&format!(
            "  Loudness Adjustment: {:+.1} dB\n",
            self.normalization.gain_db
        ));
        report.push_str(&format!(
            "  Limiting Adjustment: {:+.1} dB\n",
            self.normalization.limiting_gain_db
        ));
        report.push_str(&format!(
            "  Total Gain: {:+.1} dB\n\n",
            self.normalization.total_gain_db
        ));

        // Predicted results
        report.push_str("Predicted After Normalization:\n");
        report.push_str(&format!(
            "  Integrated Loudness: {:.1} LUFS\n",
            self.normalization.target_lufs
        ));
        report.push_str(&format!(
            "  True Peak: {:.1} dBTP\n",
            self.normalization.predicted_peak_dbtp
        ));

        if self.normalization.will_clip(-1.0) {
            report.push_str("\nWARNING: Predicted peak exceeds -1.0 dBTP threshold!\n");
        }

        report.push('\n');
        report
    }
}

impl fmt::Display for NormalizationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_text())
    }
}

/// Loudness history for visualization.
///
/// Tracks momentary and short-term loudness over time.
#[derive(Clone, Debug)]
pub struct LoudnessHistory {
    /// Momentary loudness samples (LUFS).
    pub momentary: Vec<f64>,
    /// Short-term loudness samples (LUFS).
    pub short_term: Vec<f64>,
    /// Time stamps in seconds.
    pub timestamps: Vec<f64>,
    /// Sample interval in seconds.
    pub sample_interval: f64,
}

impl LoudnessHistory {
    /// Create a new loudness history.
    ///
    /// # Arguments
    ///
    /// * `sample_interval` - Interval between samples in seconds
    #[must_use]
    pub fn new(sample_interval: f64) -> Self {
        Self {
            momentary: Vec::new(),
            short_term: Vec::new(),
            timestamps: Vec::new(),
            sample_interval,
        }
    }

    /// Add a sample to the history.
    ///
    /// # Arguments
    ///
    /// * `momentary` - Momentary loudness in LUFS
    /// * `short_term` - Short-term loudness in LUFS
    /// * `timestamp` - Time in seconds
    pub fn add_sample(&mut self, momentary: f64, short_term: f64, timestamp: f64) {
        self.momentary.push(momentary);
        self.short_term.push(short_term);
        self.timestamps.push(timestamp);
    }

    /// Get the number of samples.
    #[must_use]
    pub fn len(&self) -> usize {
        self.momentary.len()
    }

    /// Check if history is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.momentary.is_empty()
    }

    /// Generate ASCII art graph of loudness over time.
    ///
    /// # Arguments
    ///
    /// * `width` - Graph width in characters
    /// * `height` - Graph height in characters
    #[must_use]
    pub fn to_ascii_graph(&self, width: usize, height: usize) -> String {
        if self.is_empty() || width == 0 || height == 0 {
            return String::new();
        }

        let mut graph = vec![vec![' '; width]; height];

        // Find loudness range
        let min_lufs = self
            .short_term
            .iter()
            .copied()
            .filter(|x| x.is_finite())
            .fold(f64::INFINITY, f64::min)
            .min(-50.0);
        let max_lufs = self
            .short_term
            .iter()
            .copied()
            .filter(|x| x.is_finite())
            .fold(f64::NEG_INFINITY, f64::max)
            .max(-10.0);

        let range = max_lufs - min_lufs;
        if range <= 0.0 {
            return String::new();
        }

        // Plot short-term loudness
        for (i, &lufs) in self.short_term.iter().enumerate() {
            if lufs.is_finite() {
                let x = (i * width) / self.short_term.len();
                let y_norm = (lufs - min_lufs) / range;
                let y = height - 1 - ((y_norm * (height - 1) as f64) as usize).min(height - 1);

                if x < width && y < height {
                    graph[y][x] = '#';
                }
            }
        }

        // Convert to string
        let mut result = String::new();
        result.push_str(&format!(
            "Loudness Graph ({:.1} to {:.1} LUFS)\n",
            min_lufs, max_lufs
        ));
        for row in graph {
            result.push_str(&row.iter().collect::<String>());
            result.push('\n');
        }

        result
    }

    /// Export to CSV format.
    #[must_use]
    pub fn to_csv(&self) -> String {
        let mut csv = String::from("Time (s),Momentary (LUFS),Short-term (LUFS)\n");

        for i in 0..self.len() {
            csv.push_str(&format!(
                "{:.2},{:.2},{:.2}\n",
                self.timestamps[i], self.momentary[i], self.short_term[i]
            ));
        }

        csv
    }

    /// Calculate statistics.
    #[must_use]
    pub fn statistics(&self) -> LoudnessStatistics {
        let finite_momentary: Vec<f64> = self
            .momentary
            .iter()
            .copied()
            .filter(|x| x.is_finite())
            .collect();
        let finite_short_term: Vec<f64> = self
            .short_term
            .iter()
            .copied()
            .filter(|x| x.is_finite())
            .collect();

        LoudnessStatistics {
            momentary_min: finite_momentary
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min),
            momentary_max: finite_momentary
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max),
            momentary_mean: finite_momentary.iter().sum::<f64>() / finite_momentary.len() as f64,
            short_term_min: finite_short_term
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min),
            short_term_max: finite_short_term
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max),
            short_term_mean: finite_short_term.iter().sum::<f64>() / finite_short_term.len() as f64,
        }
    }
}

/// Loudness statistics over time.
#[derive(Clone, Debug)]
pub struct LoudnessStatistics {
    /// Minimum momentary loudness in LUFS.
    pub momentary_min: f64,
    /// Maximum momentary loudness in LUFS.
    pub momentary_max: f64,
    /// Mean momentary loudness in LUFS.
    pub momentary_mean: f64,
    /// Minimum short-term loudness in LUFS.
    pub short_term_min: f64,
    /// Maximum short-term loudness in LUFS.
    pub short_term_max: f64,
    /// Mean short-term loudness in LUFS.
    pub short_term_mean: f64,
}
