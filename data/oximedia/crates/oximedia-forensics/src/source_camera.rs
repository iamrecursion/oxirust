#![allow(dead_code)]
//! Source camera identification via sensor fingerprinting.
//!
//! This module identifies the source camera of an image using sensor
//! noise fingerprints (PRNU), lens aberration patterns, color filter
//! array (CFA) interpolation artifacts, and other device-specific
//! characteristics.
//!
//! # Methodology
//!
//! [`build_prnu_fingerprint`] extracts a device's PRNU pattern from a set of
//! reference images by high-pass filtering ([`extract_noise_residual`]) and
//! averaging out scene content, leaving the sensor's fixed-pattern
//! signature. [`PrnuDatabase`] then matches an unknown image's residual
//! against enrolled [`PrnuCameraFingerprint`]s via normalized correlation
//! ([`PrnuMatchResult`]) to attribute a source device — the same underlying
//! technique used for tamper localization in [`crate::noise`] and
//! [`crate::noise_analysis`], but applied device-wise rather than
//! region-wise. [`CfaAnalysis`] and [`LensFingerprint`] provide
//! complementary, independent device signatures: the Color Filter Array
//! demosaicing pattern left by the camera's Bayer/X-Trans interpolation
//! algorithm, and the lens's characteristic geometric/chromatic distortion
//! profile, respectively. Combining multiple independent fingerprints in
//! [`SourceCameraResult`] / [`CameraMatch`] improves robustness over any
//! single cue.
//!
//! # References
//!
//! - J. Lukáš, J. Fridrich, M. Goljan, "Digital Camera Identification from
//!   Sensor Pattern Noise", IEEE Transactions on Information Forensics and
//!   Security, 1(2), 205–214 (2006) — the foundational PRNU
//!   device-fingerprinting and matching method.
//! - M. Kharrazi, H. T. Sencar, N. Memon, "Blind source camera
//!   identification", IEEE ICIP (2004) — surveys CFA-interpolation and
//!   other statistical camera-model fingerprints complementary to PRNU.
//! - S. Bayram, H. Sencar, N. Memon, I. Avcibas, "Source camera
//!   identification based on CFA interpolation", IEEE ICIP (2005) — the
//!   CFA demosaicing-artifact fingerprint underlying [`CfaAnalysis`].
//! - K. San Choi, E. Y. Lam, K. K. Y. Wong, "Source camera identification
//!   using footprints from lens aberration", SPIE Digital Photography
//!   (2006) — the lens-aberration fingerprint underlying
//!   [`LensFingerprint`].

use std::collections::HashMap;

/// Camera sensor type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SensorType {
    /// CCD sensor.
    Ccd,
    /// CMOS sensor (rolling shutter).
    CmosRolling,
    /// CMOS sensor (global shutter).
    CmosGlobal,
    /// Back-side illuminated CMOS.
    BsiCmos,
    /// Foveon (stacked RGB layers).
    Foveon,
    /// Unknown sensor type.
    Unknown,
}

impl SensorType {
    /// Return a human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Ccd => "CCD",
            Self::CmosRolling => "CMOS (Rolling)",
            Self::CmosGlobal => "CMOS (Global)",
            Self::BsiCmos => "BSI CMOS",
            Self::Foveon => "Foveon",
            Self::Unknown => "Unknown",
        }
    }
}

/// Color Filter Array pattern type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CfaPattern {
    /// Standard Bayer RGGB pattern.
    BayerRggb,
    /// Bayer BGGR pattern.
    BayerBggr,
    /// Bayer GRBG pattern.
    BayerGrbg,
    /// Bayer GBRG pattern.
    BayerGbrg,
    /// X-Trans pattern (Fujifilm).
    XTrans,
    /// No CFA (Foveon, monochrome).
    None,
}

impl CfaPattern {
    /// Return a human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::BayerRggb => "Bayer RGGB",
            Self::BayerBggr => "Bayer BGGR",
            Self::BayerGrbg => "Bayer GRBG",
            Self::BayerGbrg => "Bayer GBRG",
            Self::XTrans => "X-Trans",
            Self::None => "None",
        }
    }
}

/// CFA demosaicing artifact analysis result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CfaAnalysis {
    /// Detected CFA pattern.
    pub pattern: CfaPattern,
    /// Strength of CFA artifacts (higher = more visible).
    pub artifact_strength: f64,
    /// Confidence of the detection (0.0..1.0).
    pub confidence: f64,
}

impl CfaAnalysis {
    /// Create a new CFA analysis result.
    #[must_use]
    pub fn new(pattern: CfaPattern, artifact_strength: f64, confidence: f64) -> Self {
        Self {
            pattern,
            artifact_strength,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Check if CFA artifacts are detectable.
    #[must_use]
    pub fn artifacts_present(&self) -> bool {
        self.artifact_strength > 0.01 && self.confidence > 0.5
    }
}

/// Lens aberration fingerprint for camera identification.
#[derive(Debug, Clone, PartialEq)]
pub struct LensFingerprint {
    /// Radial distortion coefficients [k1, k2, k3].
    pub radial_coeffs: [f64; 3],
    /// Chromatic aberration magnitude (pixels).
    pub chromatic_aberration: f64,
    /// Vignetting falloff factor.
    pub vignetting_falloff: f64,
    /// Confidence of the lens identification.
    pub confidence: f64,
}

impl LensFingerprint {
    /// Create a new lens fingerprint.
    #[must_use]
    pub fn new(
        radial_coeffs: [f64; 3],
        chromatic_aberration: f64,
        vignetting_falloff: f64,
    ) -> Self {
        Self {
            radial_coeffs,
            chromatic_aberration,
            vignetting_falloff,
            confidence: 0.0,
        }
    }

    /// Compute the similarity to another lens fingerprint.
    #[must_use]
    pub fn similarity(&self, other: &Self) -> f64 {
        // Weighted distance in parameter space
        let mut dist = 0.0;
        for i in 0..3 {
            dist += (self.radial_coeffs[i] - other.radial_coeffs[i]).powi(2);
        }
        dist += (self.chromatic_aberration - other.chromatic_aberration).powi(2) * 0.1;
        dist += (self.vignetting_falloff - other.vignetting_falloff).powi(2) * 0.1;
        let distance = dist.sqrt();

        // Convert distance to similarity (0..1)
        1.0 / (1.0 + distance * 10.0)
    }

    /// Check if distortion is minimal (possibly a prime lens or corrected).
    #[must_use]
    pub fn is_low_distortion(&self) -> bool {
        self.radial_coeffs.iter().all(|c| c.abs() < 0.001)
    }
}

/// Camera identification match result.
#[derive(Debug, Clone)]
pub struct CameraMatch {
    /// Camera make/model string.
    pub camera_model: String,
    /// Match confidence (0.0..1.0).
    pub confidence: f64,
    /// PRNU correlation score.
    pub prnu_score: f64,
    /// CFA pattern match score.
    pub cfa_score: f64,
    /// Lens fingerprint similarity score.
    pub lens_score: f64,
}

impl CameraMatch {
    /// Create a new camera match result.
    #[must_use]
    pub fn new(camera_model: &str, confidence: f64) -> Self {
        Self {
            camera_model: camera_model.to_string(),
            confidence: confidence.clamp(0.0, 1.0),
            prnu_score: 0.0,
            cfa_score: 0.0,
            lens_score: 0.0,
        }
    }

    /// Compute the weighted overall score.
    #[must_use]
    pub fn weighted_score(&self, prnu_weight: f64, cfa_weight: f64, lens_weight: f64) -> f64 {
        let total_weight = prnu_weight + cfa_weight + lens_weight;
        if total_weight < 1e-10 {
            return 0.0;
        }
        (self.prnu_score * prnu_weight
            + self.cfa_score * cfa_weight
            + self.lens_score * lens_weight)
            / total_weight
    }
}

/// Database of known camera fingerprints.
#[derive(Debug, Clone)]
pub struct CameraDatabase {
    /// Known camera entries indexed by model name.
    entries: HashMap<String, CameraEntry>,
}

/// A single camera entry in the database.
#[derive(Debug, Clone)]
pub struct CameraEntry {
    /// Camera model name.
    pub model: String,
    /// Sensor type.
    pub sensor_type: SensorType,
    /// Expected CFA pattern.
    pub cfa_pattern: CfaPattern,
    /// Reference lens fingerprint.
    pub lens_fingerprint: Option<LensFingerprint>,
    /// Native resolution (width, height).
    pub native_resolution: (u32, u32),
}

impl CameraEntry {
    /// Create a new camera entry.
    #[must_use]
    pub fn new(
        model: &str,
        sensor_type: SensorType,
        cfa_pattern: CfaPattern,
        resolution: (u32, u32),
    ) -> Self {
        Self {
            model: model.to_string(),
            sensor_type,
            cfa_pattern,
            lens_fingerprint: None,
            native_resolution: resolution,
        }
    }
}

impl CameraDatabase {
    /// Create a new empty camera database.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Add a camera entry to the database.
    pub fn add_entry(&mut self, entry: CameraEntry) {
        self.entries.insert(entry.model.clone(), entry);
    }

    /// Look up a camera by model name.
    #[must_use]
    pub fn lookup(&self, model: &str) -> Option<&CameraEntry> {
        self.entries.get(model)
    }

    /// Return the number of entries in the database.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the database is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Find the best matching camera for a given CFA analysis and lens fingerprint.
    #[must_use]
    pub fn find_best_match(
        &self,
        cfa: &CfaAnalysis,
        lens: &LensFingerprint,
    ) -> Option<CameraMatch> {
        let mut best: Option<CameraMatch> = None;

        for entry in self.entries.values() {
            let cfa_score = if entry.cfa_pattern == cfa.pattern {
                cfa.confidence
            } else {
                0.0
            };

            let lens_score = entry
                .lens_fingerprint
                .as_ref()
                .map_or(0.0, |ref_lens| lens.similarity(ref_lens));

            let overall = cfa_score * 0.4 + lens_score * 0.6;

            let should_replace = match &best {
                Some(current) => overall > current.confidence,
                None => overall > 0.1,
            };

            if should_replace {
                let mut m = CameraMatch::new(&entry.model, overall);
                m.cfa_score = cfa_score;
                m.lens_score = lens_score;
                best = Some(m);
            }
        }

        best
    }
}

impl Default for CameraDatabase {
    fn default() -> Self {
        Self::new()
    }
}

/// Comprehensive source camera identification result.
#[derive(Debug, Clone)]
pub struct SourceCameraResult {
    /// Best camera match (if any).
    pub best_match: Option<CameraMatch>,
    /// All candidate matches sorted by confidence.
    pub candidates: Vec<CameraMatch>,
    /// CFA analysis result.
    pub cfa_analysis: Option<CfaAnalysis>,
    /// Lens fingerprint.
    pub lens_fingerprint: Option<LensFingerprint>,
    /// Detected sensor type.
    pub sensor_type: SensorType,
    /// Textual findings.
    pub findings: Vec<String>,
}

impl SourceCameraResult {
    /// Create a new empty result.
    #[must_use]
    pub fn new() -> Self {
        Self {
            best_match: None,
            candidates: Vec::new(),
            cfa_analysis: None,
            lens_fingerprint: None,
            sensor_type: SensorType::Unknown,
            findings: Vec::new(),
        }
    }

    /// Add a finding.
    pub fn add_finding(&mut self, finding: &str) {
        self.findings.push(finding.to_string());
    }

    /// Whether a camera was identified.
    #[must_use]
    pub fn is_identified(&self) -> bool {
        self.best_match
            .as_ref()
            .map_or(false, |m| m.confidence > 0.5)
    }
}

impl Default for SourceCameraResult {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// PRNU fingerprint extraction and matching
// ---------------------------------------------------------------------------

/// Configuration for PRNU fingerprint extraction.
#[derive(Debug, Clone)]
pub struct PrnuConfig {
    /// Denoising filter kernel radius (pixels).
    pub denoise_radius: usize,
    /// Minimum images required to build a reliable fingerprint.
    pub min_images: usize,
    /// Correlation threshold for positive camera match.
    pub match_threshold: f64,
}

impl Default for PrnuConfig {
    fn default() -> Self {
        Self {
            denoise_radius: 2,
            min_images: 1,
            match_threshold: 0.45,
        }
    }
}

/// A PRNU fingerprint with metadata and matching capabilities.
#[derive(Debug, Clone)]
pub struct PrnuCameraFingerprint {
    /// Width of the fingerprint.
    pub width: usize,
    /// Height of the fingerprint.
    pub height: usize,
    /// PRNU noise residual pattern (row-major, one value per pixel).
    pub pattern: Vec<f64>,
    /// Camera identifier.
    pub camera_id: String,
    /// Number of images used to build this fingerprint.
    pub num_images: usize,
}

impl PrnuCameraFingerprint {
    /// Create a new empty fingerprint for a given camera.
    #[must_use]
    pub fn new(width: usize, height: usize, camera_id: &str) -> Self {
        Self {
            width,
            height,
            pattern: vec![0.0; width * height],
            camera_id: camera_id.to_string(),
            num_images: 0,
        }
    }

    /// Normalized cross-correlation with another fingerprint.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn correlate(&self, other: &Self) -> f64 {
        if self.pattern.len() != other.pattern.len() || self.pattern.is_empty() {
            return 0.0;
        }
        let n = self.pattern.len() as f64;
        let mean_a: f64 = self.pattern.iter().sum::<f64>() / n;
        let mean_b: f64 = other.pattern.iter().sum::<f64>() / n;

        let mut cov = 0.0;
        let mut var_a = 0.0;
        let mut var_b = 0.0;
        for (a, b) in self.pattern.iter().zip(other.pattern.iter()) {
            let da = a - mean_a;
            let db = b - mean_b;
            cov += da * db;
            var_a += da * da;
            var_b += db * db;
        }
        let denom = (var_a * var_b).sqrt();
        if denom < 1e-15 {
            0.0
        } else {
            cov / denom
        }
    }

    /// Check whether this fingerprint matches another above a given threshold.
    #[must_use]
    pub fn matches(&self, other: &Self, threshold: f64) -> bool {
        self.correlate(other) >= threshold
    }
}

/// Extract a noise residual from a single image (grayscale pixel rows).
///
/// The residual is `original - denoised`. The denoising uses a simple
/// local mean filter with the given kernel `radius`.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn extract_noise_residual(pixel_rows: &[Vec<f64>], radius: usize) -> Vec<f64> {
    let height = pixel_rows.len();
    if height == 0 {
        return Vec::new();
    }
    let width = pixel_rows[0].len();
    let mut residual = Vec::with_capacity(height * width);

    for y in 0..height {
        for x in 0..width {
            let original = pixel_rows[y][x];
            // Local mean filter.
            let mut sum = 0.0;
            let mut count = 0u32;
            let y_start = y.saturating_sub(radius);
            let y_end = (y + radius + 1).min(height);
            let x_start = x.saturating_sub(radius);
            let x_end = (x + radius + 1).min(width);
            for ry in y_start..y_end {
                for rx in x_start..x_end {
                    if rx < pixel_rows[ry].len() {
                        sum += pixel_rows[ry][rx];
                        count += 1;
                    }
                }
            }
            let denoised = if count > 0 { sum / count as f64 } else { 0.0 };
            residual.push(original - denoised);
        }
    }

    residual
}

/// Build a PRNU camera fingerprint by averaging noise residuals from
/// multiple images taken by the same camera.
///
/// Each entry in `images` is a set of grayscale pixel rows.
/// All images must have the same dimensions.
#[allow(clippy::cast_precision_loss)]
pub fn build_prnu_fingerprint(
    images: &[Vec<Vec<f64>>],
    camera_id: &str,
    config: &PrnuConfig,
) -> Result<PrnuCameraFingerprint, &'static str> {
    if images.is_empty() {
        return Err("at least one image is required");
    }
    if images.len() < config.min_images {
        return Err("not enough images for reliable fingerprint");
    }

    let height = images[0].len();
    if height == 0 {
        return Err("image has zero height");
    }
    let width = images[0][0].len();

    let pixel_count = height * width;
    let mut accumulated = vec![0.0f64; pixel_count];

    for img in images {
        if img.len() != height {
            return Err("image dimensions do not match");
        }
        let residual = extract_noise_residual(img, config.denoise_radius);
        if residual.len() != pixel_count {
            return Err("image dimensions do not match");
        }
        for (acc, &r) in accumulated.iter_mut().zip(residual.iter()) {
            *acc += r;
        }
    }

    let n = images.len() as f64;
    for v in &mut accumulated {
        *v /= n;
    }

    Ok(PrnuCameraFingerprint {
        width,
        height,
        pattern: accumulated,
        camera_id: camera_id.to_string(),
        num_images: images.len(),
    })
}

/// A database of PRNU camera fingerprints for matching query images.
#[derive(Debug, Clone, Default)]
pub struct PrnuDatabase {
    fingerprints: Vec<PrnuCameraFingerprint>,
}

/// Result of a PRNU database query.
#[derive(Debug, Clone)]
pub struct PrnuMatchResult {
    /// Camera ID of the best match.
    pub camera_id: String,
    /// Correlation score of the best match.
    pub correlation: f64,
    /// Whether the correlation exceeds the threshold.
    pub is_match: bool,
}

impl PrnuDatabase {
    /// Create a new empty database.
    #[must_use]
    pub fn new() -> Self {
        Self {
            fingerprints: Vec::new(),
        }
    }

    /// Add a fingerprint to the database.
    pub fn add_fingerprint(&mut self, fp: PrnuCameraFingerprint) {
        self.fingerprints.push(fp);
    }

    /// Number of fingerprints in the database.
    #[must_use]
    pub fn len(&self) -> usize {
        self.fingerprints.len()
    }

    /// Whether the database is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.fingerprints.is_empty()
    }

    /// Query the database with a noise residual to find the best matching camera.
    #[must_use]
    pub fn query(&self, query_residual: &[f64], threshold: f64) -> Option<PrnuMatchResult> {
        let mut best: Option<PrnuMatchResult> = None;

        // Build a temporary fingerprint for correlation.
        for fp in &self.fingerprints {
            if fp.pattern.len() != query_residual.len() {
                continue;
            }
            // Compute NCC inline.
            let n = fp.pattern.len() as f64;
            if n < 1.0 {
                continue;
            }
            let mean_a: f64 = fp.pattern.iter().sum::<f64>() / n;
            let mean_b: f64 = query_residual.iter().sum::<f64>() / n;

            let mut cov = 0.0;
            let mut var_a = 0.0;
            let mut var_b = 0.0;
            for (a, b) in fp.pattern.iter().zip(query_residual.iter()) {
                let da = a - mean_a;
                let db = b - mean_b;
                cov += da * db;
                var_a += da * da;
                var_b += db * db;
            }
            let denom = (var_a * var_b).sqrt();
            let corr = if denom < 1e-15 { 0.0 } else { cov / denom };

            let should_replace = match &best {
                Some(current) => corr > current.correlation,
                None => true,
            };
            if should_replace {
                best = Some(PrnuMatchResult {
                    camera_id: fp.camera_id.clone(),
                    correlation: corr,
                    is_match: corr >= threshold,
                });
            }
        }

        best
    }

    /// Query with default threshold from `PrnuConfig::default().match_threshold`.
    #[must_use]
    pub fn query_default(&self, query_residual: &[f64]) -> Option<PrnuMatchResult> {
        self.query(query_residual, PrnuConfig::default().match_threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sensor_type_labels() {
        assert_eq!(SensorType::Ccd.label(), "CCD");
        assert_eq!(SensorType::CmosRolling.label(), "CMOS (Rolling)");
        assert_eq!(SensorType::BsiCmos.label(), "BSI CMOS");
        assert_eq!(SensorType::Unknown.label(), "Unknown");
    }

    #[test]
    fn test_cfa_pattern_labels() {
        assert_eq!(CfaPattern::BayerRggb.label(), "Bayer RGGB");
        assert_eq!(CfaPattern::XTrans.label(), "X-Trans");
        assert_eq!(CfaPattern::None.label(), "None");
    }

    #[test]
    fn test_cfa_analysis_artifacts_present() {
        let cfa = CfaAnalysis::new(CfaPattern::BayerRggb, 0.1, 0.8);
        assert!(cfa.artifacts_present());

        let cfa2 = CfaAnalysis::new(CfaPattern::BayerRggb, 0.001, 0.8);
        assert!(!cfa2.artifacts_present());
    }

    #[test]
    fn test_cfa_analysis_low_confidence() {
        let cfa = CfaAnalysis::new(CfaPattern::BayerRggb, 0.5, 0.3);
        assert!(!cfa.artifacts_present());
    }

    #[test]
    fn test_lens_fingerprint_self_similarity() {
        let lf = LensFingerprint::new([0.01, -0.02, 0.001], 0.5, 0.8);
        assert!((lf.similarity(&lf) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_lens_fingerprint_different() {
        let lf1 = LensFingerprint::new([0.01, -0.02, 0.001], 0.5, 0.8);
        let lf2 = LensFingerprint::new([0.1, -0.2, 0.05], 2.0, 0.3);
        let sim = lf1.similarity(&lf2);
        assert!(sim < 0.5);
        assert!(sim > 0.0);
    }

    #[test]
    fn test_lens_low_distortion() {
        let lf = LensFingerprint::new([0.0001, -0.0005, 0.0002], 0.5, 0.8);
        assert!(lf.is_low_distortion());

        let lf2 = LensFingerprint::new([0.01, -0.02, 0.001], 0.5, 0.8);
        assert!(!lf2.is_low_distortion());
    }

    #[test]
    fn test_camera_match_weighted_score() {
        let mut m = CameraMatch::new("TestCam", 0.8);
        m.prnu_score = 0.9;
        m.cfa_score = 0.7;
        m.lens_score = 0.8;
        let score = m.weighted_score(1.0, 1.0, 1.0);
        assert!((score - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_camera_match_zero_weights() {
        let m = CameraMatch::new("TestCam", 0.8);
        assert!((m.weighted_score(0.0, 0.0, 0.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_camera_database_add_lookup() {
        let mut db = CameraDatabase::new();
        db.add_entry(CameraEntry::new(
            "Canon EOS R5",
            SensorType::CmosRolling,
            CfaPattern::BayerRggb,
            (8192, 5464),
        ));
        assert_eq!(db.len(), 1);
        assert!(!db.is_empty());
        assert!(db.lookup("Canon EOS R5").is_some());
        assert!(db.lookup("Nikon Z9").is_none());
    }

    #[test]
    fn test_camera_database_empty() {
        let db = CameraDatabase::new();
        assert!(db.is_empty());
        assert_eq!(db.len(), 0);
    }

    #[test]
    fn test_camera_database_find_best_match() {
        let mut db = CameraDatabase::new();
        let mut entry = CameraEntry::new(
            "Canon EOS R5",
            SensorType::CmosRolling,
            CfaPattern::BayerRggb,
            (8192, 5464),
        );
        entry.lens_fingerprint = Some(LensFingerprint::new([0.01, -0.02, 0.001], 0.5, 0.8));
        db.add_entry(entry);

        let cfa = CfaAnalysis::new(CfaPattern::BayerRggb, 0.1, 0.9);
        let lens = LensFingerprint::new([0.01, -0.02, 0.001], 0.5, 0.8);
        let best = db.find_best_match(&cfa, &lens);
        assert!(best.is_some());
        assert_eq!(
            best.expect("test expectation failed").camera_model,
            "Canon EOS R5"
        );
    }

    #[test]
    fn test_source_camera_result_not_identified() {
        let result = SourceCameraResult::new();
        assert!(!result.is_identified());
    }

    #[test]
    fn test_source_camera_result_identified() {
        let mut result = SourceCameraResult::new();
        result.best_match = Some(CameraMatch::new("TestCam", 0.8));
        assert!(result.is_identified());
    }

    #[test]
    fn test_source_camera_result_findings() {
        let mut result = SourceCameraResult::new();
        result.add_finding("CFA pattern detected: Bayer RGGB");
        result.add_finding("Lens distortion matches Canon 24-70mm");
        assert_eq!(result.findings.len(), 2);
    }

    // ---- PRNU fingerprint tests ----

    #[test]
    fn test_extract_noise_residual_empty() {
        let residual = extract_noise_residual(&[], 2);
        assert!(residual.is_empty());
    }

    #[test]
    fn test_extract_noise_residual_uniform() {
        let rows: Vec<Vec<f64>> = (0..8).map(|_| vec![128.0; 8]).collect();
        let residual = extract_noise_residual(&rows, 2);
        assert_eq!(residual.len(), 64);
        // Uniform image: residuals should be very close to zero.
        for &v in &residual {
            assert!(v.abs() < 1e-10, "residual should be ~0 for uniform image");
        }
    }

    #[test]
    fn test_extract_noise_residual_gradient() {
        let rows: Vec<Vec<f64>> = (0..8)
            .map(|y| (0..8).map(|x| (x + y) as f64 * 10.0).collect())
            .collect();
        let residual = extract_noise_residual(&rows, 1);
        assert_eq!(residual.len(), 64);
    }

    #[test]
    fn test_build_prnu_fingerprint_single_image() {
        let img: Vec<Vec<f64>> = (0..4)
            .map(|y| (0..4).map(|x| ((x * 7 + y * 13) % 256) as f64).collect())
            .collect();
        let config = PrnuConfig {
            denoise_radius: 1,
            min_images: 1,
            match_threshold: 0.45,
        };
        let fp = build_prnu_fingerprint(&[img], "cam-A", &config);
        assert!(fp.is_ok());
        let fp = fp.expect("should succeed");
        assert_eq!(fp.camera_id, "cam-A");
        assert_eq!(fp.num_images, 1);
        assert_eq!(fp.pattern.len(), 16);
    }

    #[test]
    fn test_build_prnu_fingerprint_no_images() {
        let config = PrnuConfig::default();
        let result = build_prnu_fingerprint(&[], "cam-X", &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_prnu_fingerprint_self_correlation() {
        let img: Vec<Vec<f64>> = (0..8)
            .map(|y| (0..8).map(|x| ((x * 3 + y * 5 + 7) % 200) as f64).collect())
            .collect();
        let config = PrnuConfig {
            denoise_radius: 1,
            min_images: 1,
            match_threshold: 0.45,
        };
        let fp = build_prnu_fingerprint(&[img], "cam-B", &config).expect("should succeed");
        let corr = fp.correlate(&fp);
        assert!(
            (corr - 1.0).abs() < 1e-10,
            "self correlation should be 1.0, got {corr}"
        );
    }

    #[test]
    fn test_prnu_fingerprint_matches() {
        let img: Vec<Vec<f64>> = (0..4)
            .map(|y| (0..4).map(|x| ((x + y * 2) % 128) as f64 + 50.0).collect())
            .collect();
        let config = PrnuConfig {
            denoise_radius: 1,
            min_images: 1,
            match_threshold: 0.9,
        };
        let fp = build_prnu_fingerprint(&[img], "cam-C", &config).expect("should succeed");
        assert!(fp.matches(&fp, 0.9));
    }

    #[test]
    fn test_prnu_database_add_and_query() {
        let img1: Vec<Vec<f64>> = (0..4)
            .map(|y| (0..4).map(|x| ((x * 3 + y * 7) % 200) as f64).collect())
            .collect();
        let config = PrnuConfig {
            denoise_radius: 1,
            min_images: 1,
            match_threshold: 0.45,
        };
        let fp = build_prnu_fingerprint(std::slice::from_ref(&img1), "cam-D", &config)
            .expect("should succeed");

        let mut db = PrnuDatabase::new();
        assert!(db.is_empty());
        db.add_fingerprint(fp);
        assert_eq!(db.len(), 1);

        // Query with the same image residual.
        let residual = extract_noise_residual(&img1, 1);
        let result = db.query(&residual, 0.3);
        assert!(result.is_some());
        let m = result.expect("should match");
        assert_eq!(m.camera_id, "cam-D");
        assert!(m.correlation > 0.3);
    }

    #[test]
    fn test_prnu_database_query_no_match() {
        let db = PrnuDatabase::new();
        let residual = vec![1.0, 2.0, 3.0];
        assert!(db.query(&residual, 0.5).is_none());
    }

    #[test]
    fn test_prnu_database_size_mismatch() {
        let img: Vec<Vec<f64>> = (0..4)
            .map(|y| (0..4).map(|x| (x + y) as f64).collect())
            .collect();
        let config = PrnuConfig {
            denoise_radius: 1,
            min_images: 1,
            match_threshold: 0.45,
        };
        let fp = build_prnu_fingerprint(&[img], "cam-E", &config).expect("should succeed");
        let mut db = PrnuDatabase::new();
        db.add_fingerprint(fp);

        // Query with wrong size residual.
        let residual = vec![1.0, 2.0, 3.0]; // 3 != 16
        let result = db.query(&residual, 0.3);
        // Should return None since no fingerprint matches the size.
        assert!(result.is_none());
    }

    #[test]
    fn test_prnu_config_default() {
        let cfg = PrnuConfig::default();
        assert_eq!(cfg.denoise_radius, 2);
        assert_eq!(cfg.min_images, 1);
        assert!((cfg.match_threshold - 0.45).abs() < 1e-10);
    }

    #[test]
    fn test_prnu_multiple_images_averaging() {
        let img1: Vec<Vec<f64>> = (0..4)
            .map(|y| (0..4).map(|x| ((x + y) * 10) as f64).collect())
            .collect();
        let img2: Vec<Vec<f64>> = (0..4)
            .map(|y| (0..4).map(|x| ((x + y) * 10 + 5) as f64).collect())
            .collect();
        let config = PrnuConfig {
            denoise_radius: 1,
            min_images: 1,
            match_threshold: 0.45,
        };
        let fp = build_prnu_fingerprint(&[img1, img2], "cam-F", &config).expect("should succeed");
        assert_eq!(fp.num_images, 2);
        assert_eq!(fp.pattern.len(), 16);
    }
}
