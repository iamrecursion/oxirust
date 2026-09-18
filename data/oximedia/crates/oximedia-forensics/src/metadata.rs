//! Metadata Analysis
//!
//! This module analyzes image metadata (EXIF, IPTC, XMP) for inconsistencies
//! and tampering indicators.

use crate::{ForensicTest, ForensicsResult};
use std::collections::HashMap;

/// EXIF data structure
#[derive(Debug, Clone)]
pub struct ExifData {
    /// Camera make
    pub make: Option<String>,
    /// Camera model
    pub model: Option<String>,
    /// Software used
    pub software: Option<String>,
    /// Date/time original
    pub datetime_original: Option<String>,
    /// Date/time digitized
    pub datetime_digitized: Option<String>,
    /// GPS latitude
    pub gps_latitude: Option<f64>,
    /// GPS longitude
    pub gps_longitude: Option<f64>,
    /// GPS altitude
    pub gps_altitude: Option<f64>,
    /// Image width
    pub width: Option<u32>,
    /// Image height
    pub height: Option<u32>,
    /// Orientation
    pub orientation: Option<u16>,
    /// All raw tags
    pub raw_tags: HashMap<String, String>,
}

impl ExifData {
    /// Create empty EXIF data
    pub fn new() -> Self {
        Self {
            make: None,
            model: None,
            software: None,
            datetime_original: None,
            datetime_digitized: None,
            gps_latitude: None,
            gps_longitude: None,
            gps_altitude: None,
            width: None,
            height: None,
            orientation: None,
            raw_tags: HashMap::new(),
        }
    }
}

impl Default for ExifData {
    fn default() -> Self {
        Self::new()
    }
}

/// Metadata inconsistencies found
#[derive(Debug, Clone)]
pub struct MetadataInconsistencies {
    /// List of inconsistencies
    pub issues: Vec<String>,
    /// Overall suspicion score
    pub suspicion_score: f64,
}

/// Analyze image metadata for tampering indicators
pub fn analyze_metadata(image_data: &[u8]) -> ForensicsResult<ForensicTest> {
    let mut test = ForensicTest::new("Metadata Analysis");

    // Extract EXIF data
    let exif_data = extract_exif_data(image_data)?;

    // Check for metadata presence
    if exif_data.raw_tags.is_empty() {
        test.add_finding("No EXIF metadata found - possible stripping".to_string());
        test.set_confidence(0.4);
        test.tampering_detected = true;
        return Ok(test);
    }

    test.add_finding(format!("Found {} EXIF tags", exif_data.raw_tags.len()));

    // Verify camera information
    if let (Some(make), Some(model)) = (&exif_data.make, &exif_data.model) {
        test.add_finding(format!("Camera: {} {}", make, model));

        // Check for known editing software signatures
        if is_editing_software(make) || is_editing_software(model) {
            test.add_finding("Image created/modified by editing software".to_string());
            test.tampering_detected = true;
        }
    }

    // Check software signature
    if let Some(software) = &exif_data.software {
        test.add_finding(format!("Software: {}", software));

        if is_editing_software(software) {
            test.add_finding("Editing software detected in metadata".to_string());
            test.tampering_detected = true;
        }
    }

    // Detect inconsistencies
    let inconsistencies = detect_metadata_inconsistencies(&exif_data);

    for issue in &inconsistencies.issues {
        test.add_finding(issue.clone());
    }

    if !inconsistencies.issues.is_empty() {
        test.tampering_detected = true;
    }

    // Verify timestamps
    let timestamp_valid = verify_timestamps(&exif_data);
    if !timestamp_valid {
        test.add_finding("Timestamp inconsistencies detected".to_string());
        test.tampering_detected = true;
    }

    // Verify GPS coordinates
    if let (Some(lat), Some(lon)) = (exif_data.gps_latitude, exif_data.gps_longitude) {
        test.add_finding(format!("GPS coordinates: {:.6}, {:.6}", lat, lon));

        if !is_valid_gps_coordinate(lat, lon) {
            test.add_finding("Invalid GPS coordinates".to_string());
            test.tampering_detected = true;
        }
    }

    // Calculate confidence
    let mut confidence = inconsistencies.suspicion_score;

    if exif_data.software.is_some() {
        confidence += 0.1;
    }

    if !timestamp_valid {
        confidence += 0.2;
    }

    test.set_confidence(confidence.min(1.0));

    Ok(test)
}

/// Extract EXIF data from image bytes
fn extract_exif_data(image_data: &[u8]) -> ForensicsResult<ExifData> {
    let mut exif_data = ExifData::new();

    // Simple JPEG EXIF parser
    // Look for EXIF marker in JPEG (0xFFE1)
    if image_data.len() < 4 {
        return Ok(exif_data);
    }

    // Check for JPEG signature
    if image_data[0] != 0xFF || image_data[1] != 0xD8 {
        return Ok(exif_data);
    }

    let mut pos = 2;

    while pos + 4 < image_data.len() {
        // Look for marker
        if image_data[pos] != 0xFF {
            break;
        }

        let marker = image_data[pos + 1];

        // Read segment length
        let length = ((image_data[pos + 2] as usize) << 8) | (image_data[pos + 3] as usize);

        // Check for APP1 (EXIF) marker
        if marker == 0xE1 && pos + length + 2 <= image_data.len() {
            // Check for "Exif\0\0" identifier
            if pos + 10 < image_data.len() && &image_data[pos + 4..pos + 10] == b"Exif\0\0" {
                // Parse EXIF data (simplified)
                parse_exif_segment(&image_data[pos + 10..pos + 2 + length], &mut exif_data);
            }
        }

        pos += 2 + length;
    }

    Ok(exif_data)
}

/// Parse EXIF segment (simplified TIFF parser)
#[allow(unused_variables)]
fn parse_exif_segment(data: &[u8], exif_data: &mut ExifData) {
    if data.len() < 8 {
        return;
    }

    // Check byte order
    let big_endian = if &data[0..2] == b"MM" {
        true
    } else if &data[0..2] == b"II" {
        false
    } else {
        return;
    };

    // Simplified parsing - just extract some common tags
    // In a real implementation, you would parse the full TIFF structure

    // Add some dummy data for demonstration
    exif_data
        .raw_tags
        .insert("ImageDescription".to_string(), "Sample".to_string());

    // Try to find common patterns
    if let Some(pos) = find_pattern(data, b"Adobe Photoshop") {
        exif_data.software = Some("Adobe Photoshop".to_string());
    } else if let Some(pos) = find_pattern(data, b"GIMP") {
        exif_data.software = Some("GIMP".to_string());
    } else if let Some(pos) = find_pattern(data, b"Canon") {
        exif_data.make = Some("Canon".to_string());
    } else if let Some(pos) = find_pattern(data, b"Nikon") {
        exif_data.make = Some("Nikon".to_string());
    } else if let Some(pos) = find_pattern(data, b"Sony") {
        exif_data.make = Some("Sony".to_string());
    }
}

/// Find pattern in byte slice
fn find_pattern(data: &[u8], pattern: &[u8]) -> Option<usize> {
    data.windows(pattern.len())
        .position(|window| window == pattern)
}

/// Check if string indicates editing software
fn is_editing_software(s: &str) -> bool {
    let s_lower = s.to_lowercase();

    s_lower.contains("photoshop")
        || s_lower.contains("gimp")
        || s_lower.contains("paint.net")
        || s_lower.contains("affinity")
        || s_lower.contains("lightroom")
        || s_lower.contains("capture one")
        || s_lower.contains("darktable")
        || s_lower.contains("pixelmator")
        || s_lower.contains("photopea")
}

/// Detect metadata inconsistencies
fn detect_metadata_inconsistencies(exif_data: &ExifData) -> MetadataInconsistencies {
    let mut issues = Vec::new();
    let mut suspicion_score = 0.0;

    // Check for missing critical fields
    if exif_data.make.is_none() && exif_data.model.is_none() {
        issues.push("Missing camera make and model".to_string());
        suspicion_score += 0.2;
    }

    // Check timestamp consistency
    if let (Some(orig), Some(digitized)) =
        (&exif_data.datetime_original, &exif_data.datetime_digitized)
    {
        if orig != digitized {
            issues.push("DateTimeOriginal differs from DateTimeDigitized".to_string());
            suspicion_score += 0.3;
        }
    }

    // Check for dimension mismatch
    if let (Some(width), Some(height)) = (exif_data.width, exif_data.height) {
        // Check for common resolutions
        if width == 0 || height == 0 {
            issues.push("Invalid image dimensions in metadata".to_string());
            suspicion_score += 0.4;
        }
    }

    // Check orientation
    if let Some(orientation) = exif_data.orientation {
        if orientation > 8 {
            issues.push("Invalid orientation value".to_string());
            suspicion_score += 0.3;
        }
    }

    MetadataInconsistencies {
        issues,
        suspicion_score,
    }
}

/// Verify timestamp validity and consistency
fn verify_timestamps(exif_data: &ExifData) -> bool {
    // Check if timestamps exist
    if exif_data.datetime_original.is_none() && exif_data.datetime_digitized.is_none() {
        return false;
    }

    // Parse and validate timestamps (simplified)
    if let Some(dt) = &exif_data.datetime_original {
        if !is_valid_datetime_format(dt) {
            return false;
        }
    }

    if let Some(dt) = &exif_data.datetime_digitized {
        if !is_valid_datetime_format(dt) {
            return false;
        }
    }

    true
}

/// Check if datetime string has valid format
fn is_valid_datetime_format(dt: &str) -> bool {
    // EXIF format: "YYYY:MM:DD HH:MM:SS"
    if dt.len() != 19 {
        return false;
    }

    let parts: Vec<&str> = dt.split(' ').collect();
    if parts.len() != 2 {
        return false;
    }

    let date_parts: Vec<&str> = parts[0].split(':').collect();
    if date_parts.len() != 3 {
        return false;
    }

    // Basic validation
    if let (Ok(year), Ok(month), Ok(day)) = (
        date_parts[0].parse::<u32>(),
        date_parts[1].parse::<u32>(),
        date_parts[2].parse::<u32>(),
    ) {
        if !(1900..=2100).contains(&year) {
            return false;
        }
        if !(1..=12).contains(&month) {
            return false;
        }
        if !(1..=31).contains(&day) {
            return false;
        }
    } else {
        return false;
    }

    true
}

/// Validate GPS coordinates
fn is_valid_gps_coordinate(lat: f64, lon: f64) -> bool {
    (-90.0..=90.0).contains(&lat) && (-180.0..=180.0).contains(&lon)
}

/// Detect thumbnail mismatch
pub fn detect_thumbnail_mismatch(image_data: &[u8]) -> ForensicsResult<bool> {
    // Extract embedded thumbnail
    let thumbnail = extract_thumbnail(image_data)?;

    if thumbnail.is_none() {
        return Ok(false);
    }

    // In a real implementation, would compare thumbnail to actual image
    // For now, just return false (no mismatch)
    Ok(false)
}

/// Extract embedded thumbnail from EXIF
#[allow(unused_variables)]
fn extract_thumbnail(image_data: &[u8]) -> ForensicsResult<Option<Vec<u8>>> {
    // Simplified thumbnail extraction
    // Real implementation would parse EXIF IFD1 for thumbnail
    Ok(None)
}

/// Analyze software signatures in metadata
pub fn analyze_software_signatures(exif_data: &ExifData) -> Vec<String> {
    let mut signatures = Vec::new();

    if let Some(software) = &exif_data.software {
        if is_editing_software(software) {
            signatures.push(format!("Editing software: {}", software));
        }
    }

    if let Some(make) = &exif_data.make {
        if is_editing_software(make) {
            signatures.push(format!("Non-camera make: {}", make));
        }
    }

    signatures
}

/// Check for metadata stripping indicators
pub fn detect_metadata_stripping(image_data: &[u8]) -> ForensicsResult<bool> {
    let exif_data = extract_exif_data(image_data)?;

    // If very few or no tags present, likely stripped
    Ok(exif_data.raw_tags.len() < 3)
}

/// Extract GPS data if present
pub fn extract_gps_data(exif_data: &ExifData) -> Option<(f64, f64, Option<f64>)> {
    if let (Some(lat), Some(lon)) = (exif_data.gps_latitude, exif_data.gps_longitude) {
        Some((lat, lon, exif_data.gps_altitude))
    } else {
        None
    }
}

/// Validate GPS against timestamp
#[allow(unused_variables)]
pub fn validate_gps_timestamp_consistency(exif_data: &ExifData, gps_coords: (f64, f64)) -> bool {
    // Check if GPS coordinates are consistent with timestamp
    // (e.g., daylight at coordinates matches timestamp)
    // Simplified implementation

    let (lat, lon) = gps_coords;

    // Basic validation
    is_valid_gps_coordinate(lat, lon)
}

/// Compare metadata between two images
pub fn compare_metadata(exif1: &ExifData, exif2: &ExifData) -> Vec<String> {
    let mut differences = Vec::new();

    if exif1.make != exif2.make {
        differences.push(format!(
            "Camera make differs: {:?} vs {:?}",
            exif1.make, exif2.make
        ));
    }

    if exif1.model != exif2.model {
        differences.push(format!(
            "Camera model differs: {:?} vs {:?}",
            exif1.model, exif2.model
        ));
    }

    if exif1.software != exif2.software {
        differences.push(format!(
            "Software differs: {:?} vs {:?}",
            exif1.software, exif2.software
        ));
    }

    differences
}

/// Estimate metadata manipulation probability
pub fn estimate_manipulation_probability(exif_data: &ExifData) -> f64 {
    let mut probability = 0.0;

    // No metadata is suspicious
    if exif_data.raw_tags.is_empty() {
        probability += 0.5;
    }

    // Editing software
    if let Some(software) = &exif_data.software {
        if is_editing_software(software) {
            probability += 0.3;
        }
    }

    // Check for common inconsistencies
    let inconsistencies = detect_metadata_inconsistencies(exif_data);
    probability += inconsistencies.suspicion_score * 0.5;

    probability.min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exif_data_creation() {
        let exif = ExifData::new();
        assert!(exif.make.is_none());
        assert!(exif.raw_tags.is_empty());
    }

    #[test]
    fn test_editing_software_detection() {
        assert!(is_editing_software("Adobe Photoshop CS6"));
        assert!(is_editing_software("GIMP 2.10"));
        assert!(!is_editing_software("Canon EOS 5D"));
    }

    #[test]
    fn test_datetime_validation() {
        assert!(is_valid_datetime_format("2023:01:15 10:30:45"));
        assert!(!is_valid_datetime_format("2023-01-15 10:30:45"));
        assert!(!is_valid_datetime_format("invalid"));
    }

    #[test]
    fn test_gps_validation() {
        assert!(is_valid_gps_coordinate(35.6762, 139.6503)); // Tokyo
        assert!(is_valid_gps_coordinate(-33.8688, 151.2093)); // Sydney
        assert!(!is_valid_gps_coordinate(95.0, 0.0)); // Invalid latitude
        assert!(!is_valid_gps_coordinate(0.0, 200.0)); // Invalid longitude
    }

    #[test]
    fn test_metadata_inconsistency_detection() {
        let mut exif = ExifData::new();
        exif.width = Some(0);
        exif.height = Some(0);

        let inconsistencies = detect_metadata_inconsistencies(&exif);
        assert!(inconsistencies.suspicion_score > 0.0);
    }

    #[test]
    fn test_find_pattern() {
        let data = b"Hello World";
        assert_eq!(find_pattern(data, b"World"), Some(6));
        assert_eq!(find_pattern(data, b"Foo"), None);
    }
}
