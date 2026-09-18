// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Quality comparison between original and converted files.

/// Quality comparison metrics.
#[derive(Debug, Clone)]
pub struct QualityComparison {
    /// Original file size in bytes
    pub original_size: u64,
    /// Converted file size in bytes
    pub converted_size: u64,
    /// Size reduction as a percentage
    pub size_reduction_percent: f64,
    /// Peak Signal-to-Noise Ratio (higher is better)
    pub psnr: Option<f64>,
    /// Structural Similarity Index (0-1, higher is better)
    pub ssim: Option<f64>,
    /// Video Multimethod Assessment Fusion (0-100, higher is better)
    pub vmaf: Option<f64>,
}

impl QualityComparison {
    /// Check if the conversion resulted in size reduction.
    #[must_use]
    pub fn has_size_reduction(&self) -> bool {
        self.size_reduction_percent > 0.0
    }

    /// Check if the size increased after conversion.
    #[must_use]
    pub fn has_size_increase(&self) -> bool {
        self.size_reduction_percent < 0.0
    }

    /// Get the compression ratio (original / converted).
    #[must_use]
    pub fn compression_ratio(&self) -> f64 {
        if self.converted_size == 0 {
            return 0.0;
        }
        self.original_size as f64 / self.converted_size as f64
    }

    /// Check if PSNR indicates good quality (> 30 dB).
    #[must_use]
    pub fn is_psnr_good(&self) -> Option<bool> {
        self.psnr.map(|psnr| psnr > 30.0)
    }

    /// Check if SSIM indicates good quality (> 0.95).
    #[must_use]
    pub fn is_ssim_good(&self) -> Option<bool> {
        self.ssim.map(|ssim| ssim > 0.95)
    }

    /// Check if VMAF indicates good quality (> 80).
    #[must_use]
    pub fn is_vmaf_good(&self) -> Option<bool> {
        self.vmaf.map(|vmaf| vmaf > 80.0)
    }

    /// Get overall quality assessment.
    #[must_use]
    pub fn overall_assessment(&self) -> QualityAssessment {
        let metrics = [
            self.is_psnr_good(),
            self.is_ssim_good(),
            self.is_vmaf_good(),
        ];

        let available_metrics: Vec<_> = metrics.iter().filter_map(|&m| m).collect();

        if available_metrics.is_empty() {
            return QualityAssessment::Unknown;
        }

        let good_count = available_metrics.iter().filter(|&&good| good).count();

        let ratio = good_count as f64 / available_metrics.len() as f64;

        if ratio >= 0.67 {
            QualityAssessment::Good
        } else if ratio >= 0.33 {
            QualityAssessment::Acceptable
        } else {
            QualityAssessment::Poor
        }
    }

    /// Format the comparison as a summary string.
    #[must_use]
    pub fn summary(&self) -> String {
        let mut parts = vec![format!(
            "Size: {} -> {} ({:.1}% reduction)",
            format_bytes(self.original_size),
            format_bytes(self.converted_size),
            self.size_reduction_percent
        )];

        if let Some(psnr) = self.psnr {
            parts.push(format!("PSNR: {psnr:.2} dB"));
        }

        if let Some(ssim) = self.ssim {
            parts.push(format!("SSIM: {ssim:.4}"));
        }

        if let Some(vmaf) = self.vmaf {
            parts.push(format!("VMAF: {vmaf:.2}"));
        }

        parts.join(", ")
    }
}

/// Overall quality assessment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityAssessment {
    /// Quality metrics indicate good quality
    Good,
    /// Quality metrics indicate acceptable quality
    Acceptable,
    /// Quality metrics indicate poor quality
    Poor,
    /// Not enough metrics to assess
    Unknown,
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    format!("{:.2} {}", size, UNITS[unit_index])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_comparison() -> QualityComparison {
        QualityComparison {
            original_size: 10_000_000,
            converted_size: 5_000_000,
            size_reduction_percent: 50.0,
            psnr: Some(35.0),
            ssim: Some(0.97),
            vmaf: Some(85.0),
        }
    }

    #[test]
    fn test_has_size_reduction() {
        let comp = create_test_comparison();
        assert!(comp.has_size_reduction());
        assert!(!comp.has_size_increase());
    }

    #[test]
    fn test_compression_ratio() {
        let comp = create_test_comparison();
        assert_eq!(comp.compression_ratio(), 2.0);
    }

    #[test]
    fn test_quality_checks() {
        let comp = create_test_comparison();
        assert_eq!(comp.is_psnr_good(), Some(true));
        assert_eq!(comp.is_ssim_good(), Some(true));
        assert_eq!(comp.is_vmaf_good(), Some(true));
    }

    #[test]
    fn test_overall_assessment() {
        let comp = create_test_comparison();
        assert_eq!(comp.overall_assessment(), QualityAssessment::Good);

        let poor_comp = QualityComparison {
            original_size: 10_000_000,
            converted_size: 5_000_000,
            size_reduction_percent: 50.0,
            psnr: Some(25.0),
            ssim: Some(0.85),
            vmaf: Some(60.0),
        };
        assert_eq!(poor_comp.overall_assessment(), QualityAssessment::Poor);

        let unknown_comp = QualityComparison {
            original_size: 10_000_000,
            converted_size: 5_000_000,
            size_reduction_percent: 50.0,
            psnr: None,
            ssim: None,
            vmaf: None,
        };
        assert_eq!(
            unknown_comp.overall_assessment(),
            QualityAssessment::Unknown
        );
    }

    #[test]
    fn test_summary() {
        let comp = create_test_comparison();
        let summary = comp.summary();
        assert!(summary.contains("50.0% reduction"));
        assert!(summary.contains("PSNR"));
        assert!(summary.contains("SSIM"));
        assert!(summary.contains("VMAF"));
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
    }
}
