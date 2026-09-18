//! Export stabilization data as motion vectors for external compositing applications.
//!
//! Stabilization tools (e.g. Adobe After Effects, Nuke, DaVinci Resolve) can
//! ingest per-frame motion vector data to apply stabilization transforms in the
//! compositor rather than baking the warp into the pixels.  This module encodes
//! the per-frame stabilization corrections as motion vectors in a set of
//! interchange formats.
//!
//! # Supported Formats
//!
//! | Format | Description |
//! |--------|-------------|
//! | [`MotionVectorFormat::Csv`] | Plain CSV: one row per frame. |
//! | [`MotionVectorFormat::AfterEffects`] | AE-style tab-delimited keyframe text. |
//! | [`MotionVectorFormat::Resolve`] | DaVinci Resolve tracker `.drx` style JSON. |
//! | [`MotionVectorFormat::OpenFx`] | Simplified OFX-compatible float map. |
//!
//! # Example
//!
//! ```rust
//! use oximedia_stabilize::motion_vector_export::{
//!     MotionVectors, PerFrameVector, MotionVectorFormat, MotionVectorExporter,
//! };
//!
//! let vectors = vec![
//!     PerFrameVector::new(0, 0.0, 0.0, 0.0, 1.0),
//!     PerFrameVector::new(1, -2.5, 1.0, 0.001, 1.02),
//! ];
//! let mvs = MotionVectors { vectors, fps: 30.0, frame_width: 1920, frame_height: 1080 };
//! let csv = MotionVectorExporter::export(&mvs, MotionVectorFormat::Csv).expect("ok");
//! assert!(csv.contains("frame"));
//! ```

use crate::error::{StabilizeError, StabilizeResult};

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// A single-frame stabilization motion vector.
#[derive(Debug, Clone, PartialEq)]
pub struct PerFrameVector {
    /// Zero-based frame index.
    pub frame: usize,
    /// Horizontal translation correction (pixels; positive = shift right).
    pub dx: f64,
    /// Vertical translation correction (pixels; positive = shift down).
    pub dy: f64,
    /// Rotation correction (radians; positive = counter-clockwise).
    pub rotation: f64,
    /// Scale correction (1.0 = no change).
    pub scale: f64,
}

impl PerFrameVector {
    /// Create a new motion vector.
    #[must_use]
    pub const fn new(frame: usize, dx: f64, dy: f64, rotation: f64, scale: f64) -> Self {
        Self {
            frame,
            dx,
            dy,
            rotation,
            scale,
        }
    }

    /// Identity vector (no correction).
    #[must_use]
    pub const fn identity(frame: usize) -> Self {
        Self::new(frame, 0.0, 0.0, 0.0, 1.0)
    }

    /// Total displacement magnitude (pixels).
    #[must_use]
    pub fn displacement(&self) -> f64 {
        (self.dx * self.dx + self.dy * self.dy).sqrt()
    }

    /// Return true if this vector represents effectively no correction.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.displacement() < 1e-6 && self.rotation.abs() < 1e-6 && (self.scale - 1.0).abs() < 1e-6
    }
}

/// A complete set of motion vectors for a video sequence.
#[derive(Debug, Clone)]
pub struct MotionVectors {
    /// Per-frame vectors, one per frame in display order.
    pub vectors: Vec<PerFrameVector>,
    /// Frame rate of the source video.
    pub fps: f64,
    /// Source frame width (pixels).
    pub frame_width: usize,
    /// Source frame height (pixels).
    pub frame_height: usize,
}

impl MotionVectors {
    /// Create from a vector list and source metadata.
    #[must_use]
    pub const fn new(
        vectors: Vec<PerFrameVector>,
        fps: f64,
        frame_width: usize,
        frame_height: usize,
    ) -> Self {
        Self {
            vectors,
            fps,
            frame_width,
            frame_height,
        }
    }

    /// Maximum translation magnitude across all frames.
    #[must_use]
    pub fn max_displacement(&self) -> f64 {
        self.vectors
            .iter()
            .map(PerFrameVector::displacement)
            .fold(0.0f64, f64::max)
    }

    /// Maximum absolute rotation across all frames (radians).
    #[must_use]
    pub fn max_rotation(&self) -> f64 {
        self.vectors
            .iter()
            .map(|v| v.rotation.abs())
            .fold(0.0f64, f64::max)
    }

    /// Total number of frames.
    #[must_use]
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// True when there are no vectors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Convert translation from pixels to normalised `[0, 1]` coordinates.
    #[must_use]
    pub fn normalised_translation(&self) -> Vec<(f64, f64)> {
        let w = self.frame_width.max(1) as f64;
        let h = self.frame_height.max(1) as f64;
        self.vectors.iter().map(|v| (v.dx / w, v.dy / h)).collect()
    }
}

// ---------------------------------------------------------------------------
// Output formats
// ---------------------------------------------------------------------------

/// Interchange format for motion vector export.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotionVectorFormat {
    /// Plain CSV text (frame,dx,dy,rotation_deg,scale).
    Csv,
    /// Adobe After Effects keyframe text (tab-delimited).
    AfterEffects,
    /// DaVinci Resolve tracker JSON.
    Resolve,
    /// Simplified OFX float map (one line per frame).
    OpenFx,
}

// ---------------------------------------------------------------------------
// Exporter
// ---------------------------------------------------------------------------

/// Exports [`MotionVectors`] to various interchange formats.
pub struct MotionVectorExporter;

impl MotionVectorExporter {
    /// Export motion vectors to the specified format.
    ///
    /// Returns a `String` containing the formatted data, ready to be written
    /// to a file or piped to an external tool.
    ///
    /// # Errors
    ///
    /// - [`StabilizeError::EmptyFrameSequence`] – no vectors to export.
    pub fn export(mvs: &MotionVectors, format: MotionVectorFormat) -> StabilizeResult<String> {
        if mvs.is_empty() {
            return Err(StabilizeError::EmptyFrameSequence);
        }
        match format {
            MotionVectorFormat::Csv => Ok(Self::to_csv(mvs)),
            MotionVectorFormat::AfterEffects => Ok(Self::to_after_effects(mvs)),
            MotionVectorFormat::Resolve => Ok(Self::to_resolve_json(mvs)),
            MotionVectorFormat::OpenFx => Ok(Self::to_ofx(mvs)),
        }
    }

    // ------------------------------------------------------------------
    // CSV
    // ------------------------------------------------------------------

    fn to_csv(mvs: &MotionVectors) -> String {
        let mut out = String::from("frame,dx,dy,rotation_deg,scale\n");
        for v in &mvs.vectors {
            out.push_str(&format!(
                "{},{:.6},{:.6},{:.6},{:.6}\n",
                v.frame,
                v.dx,
                v.dy,
                v.rotation.to_degrees(),
                v.scale
            ));
        }
        out
    }

    // ------------------------------------------------------------------
    // After Effects keyframe text
    // ------------------------------------------------------------------

    fn to_after_effects(mvs: &MotionVectors) -> String {
        let mut out = String::new();
        // Header block
        out.push_str("Adobe After Effects 8.0 Keyframe Data\r\n\r\n");
        out.push_str("\tUnits Per Second\t");
        out.push_str(&format!("{:.3}\r\n", mvs.fps));
        out.push_str("\tSource Width\t");
        out.push_str(&format!("{}\r\n", mvs.frame_width));
        out.push_str("\tSource Height\t");
        out.push_str(&format!("{}\r\n", mvs.frame_height));
        out.push_str("\tSource Pixel Aspect Ratio\t1\r\n");
        out.push_str("\tComp Pixel Aspect Ratio\t1\r\n\r\n");

        // Position channel
        out.push_str("Transform\tPosition\r\n");
        out.push_str("\tFrame\tX pixels\tY pixels\r\n");
        for v in &mvs.vectors {
            let cx = mvs.frame_width as f64 / 2.0 + v.dx;
            let cy = mvs.frame_height as f64 / 2.0 + v.dy;
            out.push_str(&format!("\t{}\t{:.3}\t{:.3}\r\n", v.frame, cx, cy));
        }
        out.push_str("\r\n");

        // Rotation channel
        out.push_str("Transform\tRotation\r\n");
        out.push_str("\tFrame\tDegrees\r\n");
        for v in &mvs.vectors {
            out.push_str(&format!(
                "\t{}\t{:.6}\r\n",
                v.frame,
                v.rotation.to_degrees()
            ));
        }
        out.push_str("\r\n");

        // Scale channel
        out.push_str("Transform\tScale\r\n");
        out.push_str("\tFrame\tX percent\tY percent\r\n");
        for v in &mvs.vectors {
            let pct = v.scale * 100.0;
            out.push_str(&format!("\t{}\t{:.3}\t{:.3}\r\n", v.frame, pct, pct));
        }
        out.push_str("\r\n");
        out.push_str("End of Keyframe Data\r\n");

        out
    }

    // ------------------------------------------------------------------
    // DaVinci Resolve JSON (.drx)
    // ------------------------------------------------------------------

    fn to_resolve_json(mvs: &MotionVectors) -> String {
        let mut frames_json = String::new();
        for (i, v) in mvs.vectors.iter().enumerate() {
            if i > 0 {
                frames_json.push_str(",\n");
            }
            frames_json.push_str(&format!(
                "    {{\"frame\":{},\"dx\":{:.6},\"dy\":{:.6},\"rotation\":{:.6},\"scale\":{:.6}}}",
                v.frame, v.dx, v.dy, v.rotation, v.scale
            ));
        }

        format!(
            "{{\n  \"version\": \"1.0\",\n  \"fps\": {fps:.3},\n  \"width\": {w},\n  \"height\": {h},\n  \"tracks\": [\n{frames}\n  ]\n}}\n",
            fps = mvs.fps,
            w = mvs.frame_width,
            h = mvs.frame_height,
            frames = frames_json
        )
    }

    // ------------------------------------------------------------------
    // OFX-style flat float map
    // ------------------------------------------------------------------

    fn to_ofx(mvs: &MotionVectors) -> String {
        let mut out = String::new();
        out.push_str("# OxiMedia OFX motion export v1\n");
        out.push_str(&format!(
            "# fps={:.3} width={} height={}\n",
            mvs.fps, mvs.frame_width, mvs.frame_height
        ));
        out.push_str("# frame dx_norm dy_norm rotation_rad scale\n");

        let w = mvs.frame_width.max(1) as f64;
        let h = mvs.frame_height.max(1) as f64;

        for v in &mvs.vectors {
            out.push_str(&format!(
                "{} {:.8} {:.8} {:.8} {:.8}\n",
                v.frame,
                v.dx / w,
                v.dy / h,
                v.rotation,
                v.scale
            ));
        }

        out
    }
}

// ---------------------------------------------------------------------------
// Import helpers
// ---------------------------------------------------------------------------

/// Parse a CSV string (as produced by [`MotionVectorExporter`]) back into
/// [`MotionVectors`].
///
/// Lines beginning with `#` or containing the header word `"frame"` are skipped.
///
/// # Errors
///
/// Returns [`StabilizeError::General`] on a parse failure.
pub fn import_csv(
    csv: &str,
    fps: f64,
    frame_width: usize,
    frame_height: usize,
) -> StabilizeResult<MotionVectors> {
    let mut vectors = Vec::new();

    for line in csv.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("frame") {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 5 {
            return Err(StabilizeError::General(format!(
                "malformed CSV line: {line}"
            )));
        }
        let parse = |s: &str| -> StabilizeResult<f64> {
            s.trim()
                .parse::<f64>()
                .map_err(|e| StabilizeError::General(format!("CSV parse error: {e}")))
        };
        let frame: usize = parts[0]
            .trim()
            .parse()
            .map_err(|e| StabilizeError::General(format!("CSV frame index: {e}")))?;
        let dx = parse(parts[1])?;
        let dy = parse(parts[2])?;
        let rotation_deg = parse(parts[3])?;
        let scale = parse(parts[4])?;

        vectors.push(PerFrameVector::new(
            frame,
            dx,
            dy,
            rotation_deg.to_radians(),
            scale,
        ));
    }

    if vectors.is_empty() {
        return Err(StabilizeError::EmptyFrameSequence);
    }

    Ok(MotionVectors::new(vectors, fps, frame_width, frame_height))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_mvs() -> MotionVectors {
        MotionVectors::new(
            vec![
                PerFrameVector::new(0, 0.0, 0.0, 0.0, 1.0),
                PerFrameVector::new(1, -2.0, 1.5, 0.01, 1.02),
                PerFrameVector::new(2, 3.0, -0.5, -0.005, 0.98),
            ],
            30.0,
            1920,
            1080,
        )
    }

    #[test]
    fn test_per_frame_vector_identity() {
        let v = PerFrameVector::identity(0);
        assert!(v.is_identity());
        assert!((v.displacement()).abs() < 1e-9);
    }

    #[test]
    fn test_per_frame_vector_displacement() {
        let v = PerFrameVector::new(0, 3.0, 4.0, 0.0, 1.0);
        assert!((v.displacement() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_motion_vectors_max_displacement() {
        let mvs = sample_mvs();
        let max = mvs.max_displacement();
        assert!(max > 0.0);
        // Frame 2 has displacement sqrt(9+0.25) ≈ 3.04.
        assert!(max > 3.0);
    }

    #[test]
    fn test_motion_vectors_len() {
        let mvs = sample_mvs();
        assert_eq!(mvs.len(), 3);
        assert!(!mvs.is_empty());
    }

    #[test]
    fn test_motion_vectors_normalised_translation() {
        let mvs = sample_mvs();
        let nt = mvs.normalised_translation();
        assert_eq!(nt.len(), 3);
        // Frame 0 should be (0,0).
        assert!((nt[0].0).abs() < 1e-9);
    }

    #[test]
    fn test_export_csv_header() {
        let mvs = sample_mvs();
        let csv = MotionVectorExporter::export(&mvs, MotionVectorFormat::Csv).expect("ok");
        assert!(csv.starts_with("frame,dx,dy"));
    }

    #[test]
    fn test_export_csv_row_count() {
        let mvs = sample_mvs();
        let csv = MotionVectorExporter::export(&mvs, MotionVectorFormat::Csv).expect("ok");
        // Header + 3 data rows.
        let lines: Vec<&str> = csv.lines().filter(|l| !l.is_empty()).collect();
        assert_eq!(lines.len(), 4);
    }

    #[test]
    fn test_export_after_effects_contains_header() {
        let mvs = sample_mvs();
        let ae = MotionVectorExporter::export(&mvs, MotionVectorFormat::AfterEffects).expect("ok");
        assert!(ae.contains("Adobe After Effects"));
        assert!(ae.contains("Transform"));
        assert!(ae.contains("Position"));
        assert!(ae.contains("Rotation"));
        assert!(ae.contains("Scale"));
    }

    #[test]
    fn test_export_resolve_json_valid_structure() {
        let mvs = sample_mvs();
        let json = MotionVectorExporter::export(&mvs, MotionVectorFormat::Resolve).expect("ok");
        assert!(json.contains("\"version\""));
        assert!(json.contains("\"fps\""));
        assert!(json.contains("\"tracks\""));
        assert!(json.contains("\"frame\""));
    }

    #[test]
    fn test_export_ofx_header_and_rows() {
        let mvs = sample_mvs();
        let ofx = MotionVectorExporter::export(&mvs, MotionVectorFormat::OpenFx).expect("ok");
        assert!(ofx.contains("OxiMedia OFX"));
        // 3 comment lines + 3 data lines.
        let data_lines: Vec<&str> = ofx
            .lines()
            .filter(|l| !l.starts_with('#') && !l.is_empty())
            .collect();
        assert_eq!(data_lines.len(), 3);
    }

    #[test]
    fn test_export_empty_returns_error() {
        let mvs = MotionVectors::new(vec![], 30.0, 1920, 1080);
        let result = MotionVectorExporter::export(&mvs, MotionVectorFormat::Csv);
        assert!(matches!(result, Err(StabilizeError::EmptyFrameSequence)));
    }

    #[test]
    fn test_csv_roundtrip() {
        let mvs = sample_mvs();
        let csv = MotionVectorExporter::export(&mvs, MotionVectorFormat::Csv).expect("ok");
        let restored = import_csv(&csv, 30.0, 1920, 1080).expect("import ok");
        assert_eq!(restored.len(), mvs.len());
        for (orig, rest) in mvs.vectors.iter().zip(restored.vectors.iter()) {
            assert_eq!(orig.frame, rest.frame);
            assert!((orig.dx - rest.dx).abs() < 1e-4);
            assert!((orig.dy - rest.dy).abs() < 1e-4);
            assert!((orig.scale - rest.scale).abs() < 1e-4);
        }
    }

    #[test]
    fn test_import_csv_empty_string_error() {
        let result = import_csv("", 30.0, 1920, 1080);
        assert!(result.is_err());
    }

    #[test]
    fn test_import_csv_malformed_line_error() {
        let csv = "frame,dx\nbad,data\n";
        let result = import_csv(csv, 30.0, 1920, 1080);
        assert!(result.is_err());
    }

    #[test]
    fn test_max_rotation() {
        let mvs = MotionVectors::new(
            vec![
                PerFrameVector::new(0, 0.0, 0.0, 0.1, 1.0),
                PerFrameVector::new(1, 0.0, 0.0, -0.3, 1.0),
            ],
            30.0,
            1920,
            1080,
        );
        assert!((mvs.max_rotation() - 0.3).abs() < 1e-9);
    }
}
