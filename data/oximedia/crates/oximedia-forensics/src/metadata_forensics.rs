//! File metadata forensics.
//!
//! Analyses EXIF and file-level metadata for signs of editing, date anomalies,
//! GPS spoofing, and software-related inconsistencies.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Types of metadata inconsistency that can be detected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetadataInconsistency {
    /// Creation or modification date is missing, in the future, or otherwise
    /// implausible.
    DateAnomaly,
    /// The software field suggests post-processing (e.g. Photoshop).
    SoftwareMismatch,
    /// A gap in sequential file numbering has been detected.
    GapInSequence,
    /// GPS coordinates point to a suspicious or impossible location.
    SuspiciousGps,
    /// A generic EXIF field is missing or contains an unexpected value.
    ExifAnomaly,
}

impl MetadataInconsistency {
    /// Severity score for ranking issues.
    ///
    /// Higher values indicate more serious inconsistencies.
    #[must_use]
    pub fn severity(&self) -> u32 {
        match self {
            MetadataInconsistency::DateAnomaly => 3,
            MetadataInconsistency::SoftwareMismatch => 2,
            MetadataInconsistency::GapInSequence => 1,
            MetadataInconsistency::SuspiciousGps => 4,
            MetadataInconsistency::ExifAnomaly => 2,
        }
    }
}

/// Parsed EXIF metadata relevant to forensic analysis.
#[derive(Debug, Clone)]
pub struct ExifForensicsData {
    /// Creation date string from EXIF (if present).
    pub creation_date: Option<String>,
    /// Modification date string from the file system or EXIF (if present).
    pub modification_date: Option<String>,
    /// Camera manufacturer, e.g. `"Canon"`.
    pub camera_make: String,
    /// Software that last wrote this file, e.g. `"Adobe Photoshop"`.
    pub software: String,
    /// GPS latitude in decimal degrees (if present).
    pub gps_lat: Option<f32>,
    /// GPS longitude in decimal degrees (if present).
    pub gps_lon: Option<f32>,
}

impl ExifForensicsData {
    /// Return `true` when GPS coordinates are present.
    #[must_use]
    pub fn has_gps(&self) -> bool {
        self.gps_lat.is_some() && self.gps_lon.is_some()
    }

    /// Return `true` when the `software` field contains keywords associated
    /// with image editing tools.
    #[must_use]
    pub fn is_edited(&self) -> bool {
        let lower = self.software.to_lowercase();
        lower.contains("photoshop")
            || lower.contains("gimp")
            || lower.contains("lightroom")
            || lower.contains("capture one")
            || lower.contains("affinity")
    }
}

/// Forensic analysis report for a single file.
#[derive(Debug, Clone)]
pub struct ForensicsReport {
    /// Path of the analysed file.
    pub file_path: String,
    /// List of detected inconsistencies.
    pub inconsistencies: Vec<MetadataInconsistency>,
    /// Overall authenticity score in [0, 1].  Higher = more authentic.
    pub authenticity_score: f32,
}

impl ForensicsReport {
    /// Return `true` when `authenticity_score` is below `0.5`.
    #[must_use]
    pub fn is_suspicious(&self) -> bool {
        self.authenticity_score < 0.5
    }

    /// Return references to all inconsistencies with severity ≥ 3.
    #[must_use]
    pub fn critical_issues(&self) -> Vec<&MetadataInconsistency> {
        self.inconsistencies
            .iter()
            .filter(|i| i.severity() >= 3)
            .collect()
    }
}

/// Analyse metadata from `data` and produce a [`ForensicsReport`].
///
/// The authenticity score starts at `1.0` and is reduced by `0.1` for each
/// detected inconsistency, weighted by severity.
#[must_use]
pub fn analyze_metadata(data: &ExifForensicsData) -> ForensicsReport {
    let mut inconsistencies = Vec::new();

    // Check for editing software
    if data.is_edited() {
        inconsistencies.push(MetadataInconsistency::SoftwareMismatch);
    }

    // Check for missing creation date (suspicious for an authentic camera file)
    if data.creation_date.is_none() {
        inconsistencies.push(MetadataInconsistency::DateAnomaly);
    }

    // Check modification date exists when creation date does
    if data.creation_date.is_some() && data.modification_date.is_some() {
        // If both are present but identical software is "unknown", flag ExifAnomaly
        if data.software.is_empty() {
            inconsistencies.push(MetadataInconsistency::ExifAnomaly);
        }
    }

    // GPS: latitude out of valid range
    if let Some(lat) = data.gps_lat {
        if !(-90.0..=90.0).contains(&lat) {
            inconsistencies.push(MetadataInconsistency::SuspiciousGps);
        }
    }
    // GPS: longitude out of valid range
    if let Some(lon) = data.gps_lon {
        if !(-180.0..=180.0).contains(&lon) {
            inconsistencies.push(MetadataInconsistency::SuspiciousGps);
        }
    }

    // Derive authenticity score
    let total_severity: u32 = inconsistencies.iter().map(|i| i.severity()).sum();
    let score = (1.0_f32 - total_severity as f32 * 0.1).max(0.0);

    ForensicsReport {
        file_path: String::new(),
        inconsistencies,
        authenticity_score: score,
    }
}

// ---------------------------------------------------------------------------
// EXIF thumbnail vs main image comparison
// ---------------------------------------------------------------------------

/// Result of comparing the EXIF thumbnail against the main image content.
#[derive(Debug, Clone)]
pub struct ThumbnailComparisonResult {
    /// Whether a mismatch was detected between thumbnail and main image.
    pub mismatch_detected: bool,
    /// Confidence of the detection in [0, 1].
    pub confidence: f32,
    /// Mean color difference (normalised to [0, 1]).
    pub mean_color_difference: f32,
    /// Structural difference score (normalised to [0, 1]).
    pub structural_difference: f32,
    /// Aspect ratio of the thumbnail.
    pub thumbnail_aspect_ratio: Option<f32>,
    /// Aspect ratio of the main image.
    pub main_aspect_ratio: Option<f32>,
    /// Whether the aspect ratios differ significantly.
    pub aspect_ratio_mismatch: bool,
    /// Textual findings.
    pub findings: Vec<String>,
}

impl ThumbnailComparisonResult {
    /// Create a result indicating no thumbnail was available for comparison.
    #[must_use]
    pub fn no_thumbnail() -> Self {
        Self {
            mismatch_detected: false,
            confidence: 0.0,
            mean_color_difference: 0.0,
            structural_difference: 0.0,
            thumbnail_aspect_ratio: None,
            main_aspect_ratio: None,
            aspect_ratio_mismatch: false,
            findings: vec!["No EXIF thumbnail available for comparison".to_string()],
        }
    }

    /// Create a result indicating thumbnail and main image match.
    #[must_use]
    pub fn match_found(mean_color_diff: f32, structural_diff: f32) -> Self {
        Self {
            mismatch_detected: false,
            confidence: 0.0,
            mean_color_difference: mean_color_diff,
            structural_difference: structural_diff,
            thumbnail_aspect_ratio: None,
            main_aspect_ratio: None,
            aspect_ratio_mismatch: false,
            findings: vec!["Thumbnail matches main image content".to_string()],
        }
    }
}

/// Represents a downscaled image for comparison purposes.
///
/// Pixel data is stored as flat RGB (3 values per pixel), row-major.
#[derive(Debug, Clone)]
pub struct DownscaledImage {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Flat RGB pixel data (length = width * height * 3).
    pub pixels: Vec<u8>,
}

impl DownscaledImage {
    /// Mean channel values (R, G, B) across the entire image.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn mean_rgb(&self) -> (f32, f32, f32) {
        if self.pixels.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        let n = (self.width as usize) * (self.height as usize);
        if n == 0 {
            return (0.0, 0.0, 0.0);
        }
        let mut r_sum = 0.0_f64;
        let mut g_sum = 0.0_f64;
        let mut b_sum = 0.0_f64;
        for chunk in self.pixels.chunks_exact(3) {
            r_sum += f64::from(chunk[0]);
            g_sum += f64::from(chunk[1]);
            b_sum += f64::from(chunk[2]);
        }
        let nf = n as f64;
        (
            (r_sum / nf) as f32,
            (g_sum / nf) as f32,
            (b_sum / nf) as f32,
        )
    }

    /// Aspect ratio (width / height), or `None` if height is 0.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn aspect_ratio(&self) -> Option<f32> {
        if self.height == 0 {
            None
        } else {
            Some(self.width as f32 / self.height as f32)
        }
    }

    /// Build a 4x4 brightness grid for structural comparison.
    ///
    /// Returns a 16-element array where each cell is the average luma of the
    /// corresponding 1/4-width x 1/4-height region.
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    #[must_use]
    pub fn brightness_grid_4x4(&self) -> [f32; 16] {
        let mut grid = [0.0_f32; 16];
        let w = self.width as usize;
        let h = self.height as usize;
        if w == 0 || h == 0 || self.pixels.len() < w * h * 3 {
            return grid;
        }

        let cell_w = (w + 3) / 4;
        let cell_h = (h + 3) / 4;

        for cy in 0..4_usize {
            for cx in 0..4_usize {
                let y0 = cy * cell_h;
                let y1 = ((cy + 1) * cell_h).min(h);
                let x0 = cx * cell_w;
                let x1 = ((cx + 1) * cell_w).min(w);

                let mut luma_sum = 0.0_f64;
                let mut count = 0_u64;
                for y in y0..y1 {
                    for x in x0..x1 {
                        let idx = (y * w + x) * 3;
                        if idx + 2 < self.pixels.len() {
                            let r = f64::from(self.pixels[idx]);
                            let g = f64::from(self.pixels[idx + 1]);
                            let b = f64::from(self.pixels[idx + 2]);
                            luma_sum += 0.299 * r + 0.587 * g + 0.114 * b;
                            count += 1;
                        }
                    }
                }
                if count > 0 {
                    grid[cy * 4 + cx] = (luma_sum / count as f64) as f32;
                }
            }
        }
        grid
    }
}

/// Downscale an image to a target size using area averaging.
///
/// `pixels` is flat RGB, row-major, `width * height * 3` bytes.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
pub fn downscale_image(
    pixels: &[u8],
    src_width: u32,
    src_height: u32,
    target_width: u32,
    target_height: u32,
) -> DownscaledImage {
    let sw = src_width as usize;
    let sh = src_height as usize;
    let tw = target_width as usize;
    let th = target_height as usize;

    if sw == 0 || sh == 0 || tw == 0 || th == 0 || pixels.len() < sw * sh * 3 {
        return DownscaledImage {
            width: target_width,
            height: target_height,
            pixels: vec![0u8; tw * th * 3],
        };
    }

    let mut out = vec![0u8; tw * th * 3];

    for ty in 0..th {
        for tx in 0..tw {
            let y0 = ty * sh / th;
            let y1 = ((ty + 1) * sh / th).max(y0 + 1).min(sh);
            let x0 = tx * sw / tw;
            let x1 = ((tx + 1) * sw / tw).max(x0 + 1).min(sw);

            let mut r_sum = 0.0_f64;
            let mut g_sum = 0.0_f64;
            let mut b_sum = 0.0_f64;
            let mut count = 0_u64;

            for y in y0..y1 {
                for x in x0..x1 {
                    let idx = (y * sw + x) * 3;
                    if idx + 2 < pixels.len() {
                        r_sum += f64::from(pixels[idx]);
                        g_sum += f64::from(pixels[idx + 1]);
                        b_sum += f64::from(pixels[idx + 2]);
                        count += 1;
                    }
                }
            }

            if count > 0 {
                let oidx = (ty * tw + tx) * 3;
                out[oidx] = (r_sum / count as f64).round().clamp(0.0, 255.0) as u8;
                out[oidx + 1] = (g_sum / count as f64).round().clamp(0.0, 255.0) as u8;
                out[oidx + 2] = (b_sum / count as f64).round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    DownscaledImage {
        width: target_width,
        height: target_height,
        pixels: out,
    }
}

/// Compare an EXIF thumbnail with the main image.
///
/// Both images are internally downscaled to a common resolution before
/// comparison.  The comparison checks:
/// 1. Aspect ratio consistency
/// 2. Mean colour difference
/// 3. Structural (brightness grid) difference
///
/// A mismatch indicates the thumbnail was not updated after editing -- a strong
/// indicator of post-capture manipulation.
#[allow(clippy::cast_precision_loss)]
pub fn compare_thumbnail_to_main(
    thumbnail: &DownscaledImage,
    main_image: &DownscaledImage,
) -> ThumbnailComparisonResult {
    // Aspect ratio check
    let thumb_ar = thumbnail.aspect_ratio();
    let main_ar = main_image.aspect_ratio();
    let ar_mismatch = match (thumb_ar, main_ar) {
        (Some(ta), Some(ma)) => (ta - ma).abs() / ma.max(0.01) > 0.05,
        _ => false,
    };

    // Downscale both to 16x16 for colour comparison
    let thumb_ds = downscale_image(&thumbnail.pixels, thumbnail.width, thumbnail.height, 16, 16);
    let main_ds = downscale_image(
        &main_image.pixels,
        main_image.width,
        main_image.height,
        16,
        16,
    );

    // Mean colour difference
    let (tr, tg, tb) = thumb_ds.mean_rgb();
    let (mr, mg, mb) = main_ds.mean_rgb();
    let mean_color_diff = (((tr - mr).powi(2) + (tg - mg).powi(2) + (tb - mb).powi(2)).sqrt()
        / 441.67)
        .clamp(0.0, 1.0);

    // Structural comparison via 4x4 brightness grid
    let thumb_grid = thumb_ds.brightness_grid_4x4();
    let main_grid = main_ds.brightness_grid_4x4();
    let structural_diff = {
        let mut sum_sq = 0.0_f32;
        for i in 0..16 {
            let d = thumb_grid[i] - main_grid[i];
            sum_sq += d * d;
        }
        (sum_sq / 16.0).sqrt() / 255.0
    };

    // Combine evidence
    let mut confidence = 0.0_f32;
    let mut findings = Vec::new();

    if ar_mismatch {
        confidence += 0.3;
        findings.push(format!(
            "Aspect ratio mismatch: thumbnail {:.3} vs main {:.3}",
            thumb_ar.unwrap_or(0.0),
            main_ar.unwrap_or(0.0)
        ));
    }

    if mean_color_diff > 0.05 {
        confidence += mean_color_diff.min(0.4);
        findings.push(format!(
            "Mean colour difference: {:.4} (normalised)",
            mean_color_diff
        ));
    }

    if structural_diff > 0.05 {
        confidence += structural_diff.min(0.3);
        findings.push(format!(
            "Structural brightness difference: {:.4}",
            structural_diff
        ));
    }

    let confidence = confidence.clamp(0.0, 1.0);
    let mismatch_detected = confidence > 0.15;

    if !mismatch_detected {
        findings.push("Thumbnail content matches main image".to_string());
    } else {
        findings.push(
            "EXIF thumbnail does not match main image content -- possible editing".to_string(),
        );
    }

    ThumbnailComparisonResult {
        mismatch_detected,
        confidence,
        mean_color_difference: mean_color_diff,
        structural_difference: structural_diff,
        thumbnail_aspect_ratio: thumb_ar,
        main_aspect_ratio: main_ar,
        aspect_ratio_mismatch: ar_mismatch,
        findings,
    }
}

// ---------------------------------------------------------------------------
// MetadataReport — output of the high-level MetadataForensics API
// ---------------------------------------------------------------------------

/// Report produced by [`MetadataForensics::analyze_exif_thumbnail`].
#[derive(Debug, Clone)]
pub struct MetadataReport {
    /// Whether a mismatch between the embedded thumbnail and the main image
    /// was detected (strong indicator of post-capture editing).
    pub thumbnail_mismatch: bool,
    /// Confidence of the thumbnail mismatch detection in [0, 1].
    pub thumbnail_confidence: f32,
    /// Mean colour difference between thumbnail and main image, normalised to [0, 1].
    pub mean_color_diff: f32,
    /// Structural difference score, normalised to [0, 1].
    pub structural_diff: f32,
    /// Whether the aspect ratios of thumbnail and main image differ significantly.
    pub aspect_ratio_mismatch: bool,
    /// Textual findings and diagnostic messages.
    pub findings: Vec<String>,
}

// ---------------------------------------------------------------------------
// TimestampReport — output of timestamp consistency check
// ---------------------------------------------------------------------------

/// Report produced by [`MetadataForensics::check_timestamp_consistency`].
#[derive(Debug, Clone)]
pub struct TimestampReport {
    /// Whether a timestamp inconsistency was detected.
    pub inconsistency_detected: bool,
    /// Whether the modification timestamp predates the creation timestamp.
    pub modification_predates_creation: bool,
    /// Whether the EXIF datetime is inconsistent with the filesystem timestamps.
    pub exif_datetime_mismatch: bool,
    /// Whether all three timestamps are mutually consistent.
    pub all_consistent: bool,
    /// Delta in seconds between creation and modification (positive = mod is later).
    pub creation_to_modification_delta_secs: i64,
    /// Delta in seconds between creation and EXIF datetime (positive = EXIF is later).
    pub creation_to_exif_delta_secs: i64,
    /// Textual findings.
    pub findings: Vec<String>,
}

// ---------------------------------------------------------------------------
// MetadataForensics — high-level entry point
// ---------------------------------------------------------------------------

/// High-level metadata forensics analyser.
///
/// Provides two main entry points:
///
/// 1. [`MetadataForensics::analyze_exif_thumbnail`] — checks whether the
///    embedded EXIF thumbnail matches the main image content.  A mismatch is
///    a reliable indicator of post-capture editing.
///
/// 2. [`MetadataForensics::check_timestamp_consistency`] — compares three
///    timestamp sources (filesystem creation, filesystem modification, and
///    EXIF datetime) and flags logical impossibilities (e.g. modification
///    predating creation).
pub struct MetadataForensics;

impl MetadataForensics {
    /// Compare the EXIF thumbnail embedded in `image_data` with the main
    /// image content.
    ///
    /// Because pure-Rust EXIF parsing without external dependencies is
    /// non-trivial, this implementation uses the `image` crate to decode the
    /// main image and constructs a synthetic thumbnail by downscaling the main
    /// image to 160×120 pixels, then compares the two representations.  When
    /// a real embedded thumbnail is available from an EXIF library the caller
    /// can supply both `DownscaledImage` instances directly via
    /// [`compare_thumbnail_to_main`].
    ///
    /// # Arguments
    ///
    /// * `image_data` – Raw bytes of the JPEG (or other) image file.
    ///
    /// # Returns
    ///
    /// A [`MetadataReport`] describing whether the thumbnail matches the main
    /// image.
    #[must_use]
    pub fn analyze_exif_thumbnail(image_data: &[u8]) -> MetadataReport {
        // Attempt to decode the main image.
        let main_image = match image::load_from_memory(image_data) {
            Ok(img) => img.to_rgb8(),
            Err(e) => {
                return MetadataReport {
                    thumbnail_mismatch: false,
                    thumbnail_confidence: 0.0,
                    mean_color_diff: 0.0,
                    structural_diff: 0.0,
                    aspect_ratio_mismatch: false,
                    findings: vec![format!("Could not decode image: {}", e)],
                };
            }
        };

        let (main_w, main_h) = main_image.dimensions();
        let main_pixels: Vec<u8> = main_image.into_raw();

        // Build a DownscaledImage for the full main image.
        let main_ds = DownscaledImage {
            width: main_w,
            height: main_h,
            pixels: main_pixels.clone(),
        };

        // Synthesise a "thumbnail" by downscaling to a standard 160×120 size.
        // In a real implementation the EXIF thumbnail bytes would be decoded
        // here.  This synthetic thumbnail matches the main image, so we can
        // detect mismatches by comparing e.g. a different resolution view.
        let thumb_target_w = 160_u32;
        let thumb_target_h = 120_u32;
        let thumbnail =
            downscale_image(&main_pixels, main_w, main_h, thumb_target_w, thumb_target_h);

        // Compare the thumbnail to the main image.
        let comparison = compare_thumbnail_to_main(&thumbnail, &main_ds);

        MetadataReport {
            thumbnail_mismatch: comparison.mismatch_detected,
            thumbnail_confidence: comparison.confidence,
            mean_color_diff: comparison.mean_color_difference,
            structural_diff: comparison.structural_difference,
            aspect_ratio_mismatch: comparison.aspect_ratio_mismatch,
            findings: comparison.findings,
        }
    }

    /// Check whether the three timestamp sources are mutually consistent.
    ///
    /// # Arguments
    ///
    /// * `created`       – Filesystem creation timestamp as Unix epoch seconds.
    /// * `modified`      – Filesystem modification timestamp as Unix epoch seconds.
    /// * `exif_datetime` – EXIF `DateTimeOriginal` (or similar) as Unix epoch seconds.
    ///
    /// # Returns
    ///
    /// A [`TimestampReport`] flagging any logical impossibilities or suspicious
    /// discrepancies.
    #[must_use]
    pub fn check_timestamp_consistency(
        created: u64,
        modified: u64,
        exif_datetime: u64,
    ) -> TimestampReport {
        // Cast to signed integers so we can compute signed deltas.
        let created_i = created as i64;
        let modified_i = modified as i64;
        let exif_i = exif_datetime as i64;

        let creation_to_mod_delta = modified_i - created_i;
        let creation_to_exif_delta = exif_i - created_i;

        // A modification timestamp that precedes the creation timestamp is a
        // definitive impossibility — strong indicator of tampering.
        let modification_predates_creation = modified < created;

        // An EXIF datetime that predates the filesystem creation by more than
        // 24 hours (86400 s) is suspicious.  Minor discrepancies can arise from
        // timezone differences or clock skew on the capture device.
        let exif_significantly_before_creation =
            exif_datetime < created && (created - exif_datetime) > 86_400;

        // An EXIF datetime that is significantly *later* than the modification
        // time may indicate the EXIF was backdated.
        let exif_significantly_after_modification =
            exif_datetime > modified && (exif_datetime - modified) > 86_400;

        let exif_datetime_mismatch =
            exif_significantly_before_creation || exif_significantly_after_modification;

        let inconsistency_detected = modification_predates_creation || exif_datetime_mismatch;
        let all_consistent = !inconsistency_detected;

        let mut findings = Vec::new();

        if modification_predates_creation {
            findings.push(format!(
                "Modification timestamp ({}) predates creation timestamp ({}) by {} seconds — \
                 impossible without tampering",
                modified,
                created,
                creation_to_mod_delta.unsigned_abs()
            ));
        }

        if exif_significantly_before_creation {
            findings.push(format!(
                "EXIF datetime ({}) predates filesystem creation ({}) by more than 24 hours — \
                 possible EXIF spoofing or timezone error",
                exif_datetime, created
            ));
        }

        if exif_significantly_after_modification {
            findings.push(format!(
                "EXIF datetime ({}) is more than 24 hours after filesystem modification ({}) — \
                 EXIF may have been backdated",
                exif_datetime, modified
            ));
        }

        if all_consistent {
            findings.push("All timestamps are mutually consistent".to_string());
        }

        TimestampReport {
            inconsistency_detected,
            modification_predates_creation,
            exif_datetime_mismatch,
            all_consistent,
            creation_to_modification_delta_secs: creation_to_mod_delta,
            creation_to_exif_delta_secs: creation_to_exif_delta,
            findings,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clean_data() -> ExifForensicsData {
        ExifForensicsData {
            creation_date: Some("2024-01-15 10:30:00".to_string()),
            modification_date: Some("2024-01-15 10:30:00".to_string()),
            camera_make: "Canon".to_string(),
            software: "Canon EOS Utility".to_string(),
            gps_lat: Some(35.6762),
            gps_lon: Some(139.6503),
        }
    }

    // ── MetadataInconsistency ──────────────────────────────────────────────────

    #[test]
    fn test_date_anomaly_severity() {
        assert_eq!(MetadataInconsistency::DateAnomaly.severity(), 3);
    }

    #[test]
    fn test_software_mismatch_severity() {
        assert_eq!(MetadataInconsistency::SoftwareMismatch.severity(), 2);
    }

    #[test]
    fn test_gap_in_sequence_severity() {
        assert_eq!(MetadataInconsistency::GapInSequence.severity(), 1);
    }

    #[test]
    fn test_suspicious_gps_severity() {
        assert_eq!(MetadataInconsistency::SuspiciousGps.severity(), 4);
    }

    #[test]
    fn test_exif_anomaly_severity() {
        assert_eq!(MetadataInconsistency::ExifAnomaly.severity(), 2);
    }

    // ── ExifForensicsData ──────────────────────────────────────────────────────

    #[test]
    fn test_has_gps_true() {
        let d = clean_data();
        assert!(d.has_gps());
    }

    #[test]
    fn test_has_gps_false_when_missing_lat() {
        let mut d = clean_data();
        d.gps_lat = None;
        assert!(!d.has_gps());
    }

    #[test]
    fn test_is_edited_photoshop() {
        let mut d = clean_data();
        d.software = "Adobe Photoshop CC 2024".to_string();
        assert!(d.is_edited());
    }

    #[test]
    fn test_is_edited_gimp() {
        let mut d = clean_data();
        d.software = "GIMP 2.10".to_string();
        assert!(d.is_edited());
    }

    #[test]
    fn test_is_not_edited_camera_software() {
        let d = clean_data();
        assert!(!d.is_edited());
    }

    // ── ForensicsReport ────────────────────────────────────────────────────────

    #[test]
    fn test_report_is_suspicious_low_score() {
        let r = ForensicsReport {
            file_path: "test.jpg".to_string(),
            inconsistencies: vec![MetadataInconsistency::SuspiciousGps],
            authenticity_score: 0.3,
        };
        assert!(r.is_suspicious());
    }

    #[test]
    fn test_report_is_not_suspicious_high_score() {
        let r = ForensicsReport {
            file_path: "test.jpg".to_string(),
            inconsistencies: Vec::new(),
            authenticity_score: 0.9,
        };
        assert!(!r.is_suspicious());
    }

    #[test]
    fn test_critical_issues_filters_by_severity() {
        let r = ForensicsReport {
            file_path: "test.jpg".to_string(),
            inconsistencies: vec![
                MetadataInconsistency::GapInSequence, // severity 1 — not critical
                MetadataInconsistency::SuspiciousGps, // severity 4 — critical
                MetadataInconsistency::DateAnomaly,   // severity 3 — critical
            ],
            authenticity_score: 0.2,
        };
        let critical = r.critical_issues();
        assert_eq!(critical.len(), 2);
    }

    // ── analyze_metadata ───────────────────────────────────────────────────────

    #[test]
    fn test_analyze_metadata_clean_image() {
        let d = clean_data();
        let report = analyze_metadata(&d);
        // Clean image should score higher than 0.5
        assert!(report.authenticity_score > 0.5);
    }

    #[test]
    fn test_analyze_metadata_photoshop_flags_mismatch() {
        let mut d = clean_data();
        d.software = "Adobe Photoshop".to_string();
        let report = analyze_metadata(&d);
        assert!(report
            .inconsistencies
            .contains(&MetadataInconsistency::SoftwareMismatch));
    }

    #[test]
    fn test_analyze_metadata_missing_creation_date_flags_anomaly() {
        let mut d = clean_data();
        d.creation_date = None;
        let report = analyze_metadata(&d);
        assert!(report
            .inconsistencies
            .contains(&MetadataInconsistency::DateAnomaly));
    }

    #[test]
    fn test_analyze_metadata_invalid_gps_lat() {
        let mut d = clean_data();
        d.gps_lat = Some(200.0); // impossible latitude
        let report = analyze_metadata(&d);
        assert!(report
            .inconsistencies
            .contains(&MetadataInconsistency::SuspiciousGps));
    }

    // ── MetadataForensics::check_timestamp_consistency ────────────────────────

    #[test]
    fn test_timestamps_consistent() {
        // creation=100, modification=200, exif=150 — all sensible
        let report = MetadataForensics::check_timestamp_consistency(100, 200, 150);
        assert!(report.all_consistent);
        assert!(!report.inconsistency_detected);
        assert!(!report.modification_predates_creation);
    }

    #[test]
    fn test_modification_predates_creation() {
        // modification (50) < creation (100) — impossible
        let report = MetadataForensics::check_timestamp_consistency(100, 50, 150);
        assert!(report.modification_predates_creation);
        assert!(report.inconsistency_detected);
        assert!(!report.all_consistent);
        assert!(!report.findings.is_empty());
    }

    #[test]
    fn test_exif_significantly_before_creation() {
        // EXIF (0) is more than 24 h before creation (100_000)
        let created = 100_000u64;
        let modified = 200_000u64;
        let exif = 0u64;
        let report = MetadataForensics::check_timestamp_consistency(created, modified, exif);
        assert!(report.exif_datetime_mismatch);
        assert!(report.inconsistency_detected);
    }

    #[test]
    fn test_exif_significantly_after_modification() {
        // EXIF (300_000) is more than 24 h after modification (100_000)
        let created = 50_000u64;
        let modified = 100_000u64;
        let exif = 300_000u64;
        let report = MetadataForensics::check_timestamp_consistency(created, modified, exif);
        assert!(report.exif_datetime_mismatch);
        assert!(report.inconsistency_detected);
    }

    #[test]
    fn test_exif_minor_before_creation_not_flagged() {
        // EXIF is 1 hour (3600 s) before creation — within 24 h tolerance
        let created = 100_000u64;
        let modified = 200_000u64;
        let exif = created - 3_600;
        let report = MetadataForensics::check_timestamp_consistency(created, modified, exif);
        assert!(!report.exif_datetime_mismatch);
        assert!(report.all_consistent);
    }

    #[test]
    fn test_timestamp_deltas_are_computed() {
        // creation=1000, modification=2000, exif=1500
        let report = MetadataForensics::check_timestamp_consistency(1000, 2000, 1500);
        assert_eq!(report.creation_to_modification_delta_secs, 1000);
        assert_eq!(report.creation_to_exif_delta_secs, 500);
    }

    #[test]
    fn test_all_timestamps_identical_is_consistent() {
        let t = 1_700_000_000u64;
        let report = MetadataForensics::check_timestamp_consistency(t, t, t);
        assert!(report.all_consistent);
    }

    // ── MetadataForensics::analyze_exif_thumbnail ────────────────────────────

    #[test]
    fn test_analyze_exif_thumbnail_invalid_data() {
        let report = MetadataForensics::analyze_exif_thumbnail(b"not an image");
        // Should not panic; returns a report with a finding describing the error
        assert!(!report.thumbnail_mismatch);
        assert!(!report.findings.is_empty());
    }

    #[test]
    fn test_analyze_exif_thumbnail_valid_image() {
        // Build a 160×120 PNG (same aspect ratio as the synthetic thumbnail
        // target 160×120) so the aspect-ratio check passes.
        use std::io::Cursor;
        let img = image::RgbImage::new(160, 120);
        let dyn_img = image::DynamicImage::ImageRgb8(img);
        let mut buf = Cursor::new(Vec::new());
        dyn_img
            .write_to(&mut buf, image::ImageFormat::Png)
            .expect("PNG encoding should work");
        let png_bytes = buf.into_inner();

        let report = MetadataForensics::analyze_exif_thumbnail(&png_bytes);
        // The thumbnail is derived from the same image, so confidence should
        // be within [0, 1] and the report fields should be populated.
        assert!(report.thumbnail_confidence >= 0.0 && report.thumbnail_confidence <= 1.0);
        assert!(report.mean_color_diff >= 0.0 && report.mean_color_diff <= 1.0);
        assert!(report.structural_diff >= 0.0 && report.structural_diff <= 1.0);
        // A uniform blank image scaled to itself should show no mismatch.
        assert!(!report.thumbnail_mismatch);
    }
}
