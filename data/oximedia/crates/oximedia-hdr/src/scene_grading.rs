//! Scene-by-scene HDR grading metadata.
//!
//! Provides `SceneGradingMetadata` to store per-scene colour grade and trim
//! passes, and `SceneGradingDatabase` to index grades by presentation timestamp.

use crate::{HdrError, Result};

// ─── ColorGrade ───────────────────────────────────────────────────────────────

/// A lift-gamma-gain colour grade with saturation control.
///
/// All three channels are ordered `[R, G, B]`.
#[derive(Debug, Clone)]
pub struct ColorGrade {
    /// Shadow offset added before gamma, typically −0.1..0.1.
    pub lift: [f32; 3],
    /// Midtone power (exponent reciprocal), typically 0.5..2.0.
    pub gamma: [f32; 3],
    /// Highlight scale applied after gamma, typically 0.5..2.0.
    pub gain: [f32; 3],
    /// Global saturation factor; 1.0 = neutral, 0.0 = greyscale.
    pub saturation: f32,
}

impl ColorGrade {
    /// Create a neutral identity grade (no colour change).
    pub fn identity() -> Self {
        Self {
            lift: [0.0, 0.0, 0.0],
            gamma: [1.0, 1.0, 1.0],
            gain: [1.0, 1.0, 1.0],
            saturation: 1.0,
        }
    }

    /// Apply lift-gamma-gain followed by saturation to a linear-light RGB pixel.
    ///
    /// Formula per channel `i`:
    /// ```text
    /// raised[i] = (rgb[i] + lift[i]).max(0.0)
    /// graded[i] = raised[i].powf(1.0 / gamma[i].max(1e-6)) * gain[i]
    /// ```
    /// Then mix toward luma (`Rec. 709`) based on `saturation`.
    pub fn apply_to_linear(&self, rgb: [f32; 3]) -> [f32; 3] {
        let mut out = [0.0f32; 3];
        for i in 0..3 {
            let raised = (rgb[i] + self.lift[i]).max(0.0);
            let g = self.gamma[i].max(1e-6_f32);
            out[i] = raised.powf(1.0 / g) * self.gain[i];
        }
        // Desaturate: mix toward Rec. 709 luma.
        let luma = 0.2126 * out[0] + 0.7152 * out[1] + 0.0722 * out[2];
        let s = self.saturation;
        [
            luma + s * (out[0] - luma),
            luma + s * (out[1] - luma),
            luma + s * (out[2] - luma),
        ]
    }
}

// ─── TrimPass ─────────────────────────────────────────────────────────────────

/// A target-display colour grade for a specific display peak luminance.
#[derive(Debug, Clone)]
pub struct TrimPass {
    /// Target display peak luminance in nits (e.g. 100.0 for SDR, 1000.0 for HDR).
    pub display_peak_nits: f32,
    /// The colour grade applied when targeting this display.
    pub grade: ColorGrade,
}

impl TrimPass {
    /// Create a new trim pass.
    pub fn new(display_peak_nits: f32, grade: ColorGrade) -> Self {
        Self {
            display_peak_nits,
            grade,
        }
    }
}

// ─── SceneGradingMetadata ─────────────────────────────────────────────────────

/// Per-scene HDR colour-grading metadata with optional per-display trim passes.
#[derive(Debug, Clone)]
pub struct SceneGradingMetadata {
    /// Unique scene identifier.
    pub scene_id: u64,
    /// Presentation timestamp (start) — unit is caller-defined (e.g. 90 kHz ticks).
    pub pts_start: u64,
    /// Presentation timestamp (exclusive end).
    pub pts_end: u64,
    /// Primary colour grade for this scene.
    pub grade: ColorGrade,
    /// Per-display trim passes (sorted by `display_peak_nits` for fast lookup).
    pub trim_passes: Vec<TrimPass>,
}

impl SceneGradingMetadata {
    /// Create a new scene grading entry.
    pub fn new(scene_id: u64, pts_start: u64, pts_end: u64, grade: ColorGrade) -> Self {
        Self {
            scene_id,
            pts_start,
            pts_end,
            grade,
            trim_passes: Vec::new(),
        }
    }

    /// Append a trim pass for a target display tier.
    pub fn add_trim_pass(&mut self, pass: TrimPass) {
        self.trim_passes.push(pass);
    }

    /// Return the trim pass whose `display_peak_nits` is closest to `peak_nits`.
    ///
    /// Returns `None` if no trim passes have been registered.
    pub fn trim_for_display(&self, peak_nits: f32) -> Option<&TrimPass> {
        self.trim_passes.iter().min_by(|a, b| {
            let da = (a.display_peak_nits - peak_nits).abs();
            let db = (b.display_peak_nits - peak_nits).abs();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Serialize to a compact binary representation (little-endian).
    ///
    /// Layout:
    /// ```text
    /// scene_id:   u64 LE
    /// pts_start:  u64 LE
    /// pts_end:    u64 LE
    /// lift:       3 × f32 LE
    /// gamma:      3 × f32 LE
    /// gain:       3 × f32 LE
    /// saturation: f32 LE
    /// num_passes: u32 LE
    /// per pass:
    ///   display_peak_nits: f32 LE
    ///   lift:              3 × f32 LE
    ///   gamma:             3 × f32 LE
    ///   gain:              3 × f32 LE
    ///   saturation:        f32 LE
    /// ```
    pub fn to_bytes(&self) -> Vec<u8> {
        let per_pass_size = 4 + 12 + 12 + 12 + 4; // 44 bytes each
        let capacity = 8 + 8 + 8 + 12 + 12 + 12 + 4 + 4 + self.trim_passes.len() * per_pass_size;
        let mut buf = Vec::with_capacity(capacity);

        buf.extend_from_slice(&self.scene_id.to_le_bytes());
        buf.extend_from_slice(&self.pts_start.to_le_bytes());
        buf.extend_from_slice(&self.pts_end.to_le_bytes());
        push_color_grade_bytes(&mut buf, &self.grade);

        buf.extend_from_slice(&(self.trim_passes.len() as u32).to_le_bytes());
        for pass in &self.trim_passes {
            buf.extend_from_slice(&pass.display_peak_nits.to_le_bytes());
            push_color_grade_bytes(&mut buf, &pass.grade);
        }
        buf
    }

    /// Deserialize from bytes produced by `to_bytes`.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        // Minimum: 8+8+8+12+12+12+4+4 = 68 bytes (header + grade + num_passes=0)
        const HEADER_SIZE: usize = 68;
        if data.len() < HEADER_SIZE {
            return Err(HdrError::MetadataParseError(format!(
                "too short for SceneGradingMetadata: {} < {HEADER_SIZE}",
                data.len()
            )));
        }

        let mut pos = 0usize;

        let scene_id = read_u64_le(data, &mut pos)?;
        let pts_start = read_u64_le(data, &mut pos)?;
        let pts_end = read_u64_le(data, &mut pos)?;
        let grade = read_color_grade(data, &mut pos)?;
        let num_passes = read_u32_le(data, &mut pos)? as usize;

        let per_pass_size = 44usize;
        let needed = pos + num_passes * per_pass_size;
        if data.len() < needed {
            return Err(HdrError::MetadataParseError(format!(
                "truncated trim passes: have {} bytes, need {needed}",
                data.len()
            )));
        }

        let mut trim_passes = Vec::with_capacity(num_passes);
        for _ in 0..num_passes {
            let peak_nits = read_f32_le(data, &mut pos)?;
            let grade = read_color_grade(data, &mut pos)?;
            trim_passes.push(TrimPass {
                display_peak_nits: peak_nits,
                grade,
            });
        }

        Ok(Self {
            scene_id,
            pts_start,
            pts_end,
            grade,
            trim_passes,
        })
    }
}

// ─── SceneGradingDatabase ─────────────────────────────────────────────────────

/// Indexed collection of per-scene grading metadata.
///
/// Scenes are stored in insertion order; `lookup_by_pts` does a linear scan
/// for an exact range hit and falls back to the nearest `pts_start`.
#[derive(Debug, Clone, Default)]
pub struct SceneGradingDatabase {
    scenes: Vec<SceneGradingMetadata>,
}

impl SceneGradingDatabase {
    /// Create an empty database.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a scene.  Duplicate `scene_id`s are allowed (last write wins on
    /// lookup if ranges overlap).
    pub fn insert(&mut self, scene: SceneGradingMetadata) {
        self.scenes.push(scene);
    }

    /// Find the scene whose `[pts_start, pts_end)` range contains `pts`.
    ///
    /// If no range contains `pts`, returns the scene with the closest
    /// `pts_start` (nearest by absolute distance).
    pub fn lookup_by_pts(&self, pts: u64) -> Option<&SceneGradingMetadata> {
        // First try exact range containment.
        if let Some(hit) = self
            .scenes
            .iter()
            .find(|s| s.pts_start <= pts && pts < s.pts_end)
        {
            return Some(hit);
        }
        // Fall back to closest pts_start.
        self.scenes.iter().min_by_key(|s| pts.abs_diff(s.pts_start))
    }

    /// Number of scenes in the database.
    pub fn len(&self) -> usize {
        self.scenes.len()
    }

    /// Return `true` if the database contains no scenes.
    pub fn is_empty(&self) -> bool {
        self.scenes.is_empty()
    }

    /// Remove all scenes with the given `scene_id`.
    ///
    /// Returns `true` if at least one scene was removed.
    pub fn remove_by_scene_id(&mut self, scene_id: u64) -> bool {
        let before = self.scenes.len();
        self.scenes.retain(|s| s.scene_id != scene_id);
        self.scenes.len() < before
    }

    /// Serialize all scenes to bytes.
    ///
    /// Layout: `[num_scenes: u32 LE]` followed by each scene prefixed by
    /// `[scene_len: u32 LE]`.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(self.scenes.len() as u32).to_le_bytes());
        for scene in &self.scenes {
            let scene_bytes = scene.to_bytes();
            buf.extend_from_slice(&(scene_bytes.len() as u32).to_le_bytes());
            buf.extend_from_slice(&scene_bytes);
        }
        buf
    }

    /// Deserialize from bytes produced by `to_bytes`.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < 4 {
            return Err(HdrError::MetadataParseError(
                "too short for SceneGradingDatabase header".to_string(),
            ));
        }
        let num_scenes = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let mut pos = 4usize;
        let mut scenes = Vec::with_capacity(num_scenes);
        for _ in 0..num_scenes {
            if data.len() < pos + 4 {
                return Err(HdrError::MetadataParseError(
                    "truncated scene length prefix".to_string(),
                ));
            }
            let scene_len =
                u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
                    as usize;
            pos += 4;
            let end = pos
                .checked_add(scene_len)
                .ok_or_else(|| HdrError::MetadataParseError("scene length overflow".to_string()))?;
            if data.len() < end {
                return Err(HdrError::MetadataParseError(format!(
                    "truncated scene data: have {} bytes at pos {pos}, need {scene_len}",
                    data.len()
                )));
            }
            let scene = SceneGradingMetadata::from_bytes(&data[pos..end])?;
            scenes.push(scene);
            pos = end;
        }
        Ok(Self { scenes })
    }
}

// ─── Binary serialization helpers ─────────────────────────────────────────────

fn push_color_grade_bytes(buf: &mut Vec<u8>, g: &ColorGrade) {
    for v in &g.lift {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    for v in &g.gamma {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    for v in &g.gain {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf.extend_from_slice(&g.saturation.to_le_bytes());
}

fn read_u64_le(data: &[u8], pos: &mut usize) -> Result<u64> {
    let end = pos
        .checked_add(8)
        .ok_or_else(|| HdrError::MetadataParseError("position overflow reading u64".to_string()))?;
    if data.len() < end {
        return Err(HdrError::MetadataParseError(format!(
            "too short reading u64 at pos {pos}: data.len()={}",
            data.len()
        )));
    }
    let val = u64::from_le_bytes(
        data[*pos..end]
            .try_into()
            .map_err(|_| HdrError::MetadataParseError("slice to u64 failed".to_string()))?,
    );
    *pos = end;
    Ok(val)
}

fn read_u32_le(data: &[u8], pos: &mut usize) -> Result<u32> {
    let end = pos
        .checked_add(4)
        .ok_or_else(|| HdrError::MetadataParseError("position overflow reading u32".to_string()))?;
    if data.len() < end {
        return Err(HdrError::MetadataParseError(format!(
            "too short reading u32 at pos {pos}: data.len()={}",
            data.len()
        )));
    }
    let val = u32::from_le_bytes(
        data[*pos..end]
            .try_into()
            .map_err(|_| HdrError::MetadataParseError("slice to u32 failed".to_string()))?,
    );
    *pos = end;
    Ok(val)
}

fn read_f32_le(data: &[u8], pos: &mut usize) -> Result<f32> {
    let end = pos
        .checked_add(4)
        .ok_or_else(|| HdrError::MetadataParseError("position overflow reading f32".to_string()))?;
    if data.len() < end {
        return Err(HdrError::MetadataParseError(format!(
            "too short reading f32 at pos {pos}: data.len()={}",
            data.len()
        )));
    }
    let val = f32::from_le_bytes(
        data[*pos..end]
            .try_into()
            .map_err(|_| HdrError::MetadataParseError("slice to f32 failed".to_string()))?,
    );
    *pos = end;
    Ok(val)
}

fn read_color_grade(data: &[u8], pos: &mut usize) -> Result<ColorGrade> {
    let mut lift = [0.0f32; 3];
    let mut gamma = [0.0f32; 3];
    let mut gain = [0.0f32; 3];
    for v in &mut lift {
        *v = read_f32_le(data, pos)?;
    }
    for v in &mut gamma {
        *v = read_f32_le(data, pos)?;
    }
    for v in &mut gain {
        *v = read_f32_le(data, pos)?;
    }
    let saturation = read_f32_le(data, pos)?;
    Ok(ColorGrade {
        lift,
        gamma,
        gain,
        saturation,
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_scene(id: u64, start: u64, end: u64) -> SceneGradingMetadata {
        SceneGradingMetadata::new(id, start, end, ColorGrade::identity())
    }

    // ── ColorGrade ────────────────────────────────────────────────────────────

    #[test]
    fn test_identity_grade_leaves_pixel_unchanged() {
        let g = ColorGrade::identity();
        let pixel = [0.5f32, 0.3, 0.8];
        let out = g.apply_to_linear(pixel);
        for i in 0..3 {
            assert!(
                (out[i] - pixel[i]).abs() < 1e-5,
                "channel {i}: {} != {}",
                out[i],
                pixel[i]
            );
        }
    }

    #[test]
    fn test_non_identity_grade_changes_values() {
        let g = ColorGrade {
            lift: [0.1, 0.0, 0.0],
            gamma: [1.0, 1.0, 1.0],
            gain: [2.0, 1.0, 1.0],
            saturation: 1.0,
        };
        let out = g.apply_to_linear([0.5, 0.5, 0.5]);
        // R channel: (0.5 + 0.1) ^ 1.0 * 2.0 = 1.2
        assert!((out[0] - 1.2).abs() < 1e-5, "R={}", out[0]);
    }

    #[test]
    fn test_zero_saturation_makes_grey() {
        let g = ColorGrade {
            lift: [0.0; 3],
            gamma: [1.0; 3],
            gain: [1.0; 3],
            saturation: 0.0,
        };
        let out = g.apply_to_linear([1.0, 0.0, 0.0]);
        // All channels should equal the Rec.709 luma of [1,0,0] = 0.2126.
        let expected_luma = 0.2126f32;
        for (i, &val) in out.iter().enumerate() {
            assert!((val - expected_luma).abs() < 1e-5, "ch{i}={val}");
        }
    }

    #[test]
    fn test_negative_lift_clamps_at_zero() {
        let g = ColorGrade {
            lift: [-2.0; 3],
            gamma: [1.0; 3],
            gain: [1.0; 3],
            saturation: 1.0,
        };
        let out = g.apply_to_linear([0.5, 0.5, 0.5]);
        // (0.5 - 2.0).max(0.0) = 0.0 → all channels 0.
        for (i, &val) in out.iter().enumerate() {
            assert!(val.abs() < 1e-5, "ch{i} should be 0 but got {val}",);
        }
    }

    // ── TrimPass ──────────────────────────────────────────────────────────────

    #[test]
    fn test_trim_pass_fields() {
        let tp = TrimPass::new(1000.0, ColorGrade::identity());
        assert!((tp.display_peak_nits - 1000.0).abs() < 1e-6);
    }

    // ── SceneGradingMetadata ──────────────────────────────────────────────────

    #[test]
    fn test_scene_round_trip_no_passes() {
        let original = identity_scene(42, 1000, 5000);
        let bytes = original.to_bytes();
        let decoded = SceneGradingMetadata::from_bytes(&bytes).expect("decode");
        assert_eq!(decoded.scene_id, 42);
        assert_eq!(decoded.pts_start, 1000);
        assert_eq!(decoded.pts_end, 5000);
        assert_eq!(decoded.trim_passes.len(), 0);
    }

    #[test]
    fn test_scene_round_trip_with_passes() {
        let mut scene = identity_scene(7, 0, 90_000);
        scene.add_trim_pass(TrimPass::new(100.0, ColorGrade::identity()));
        scene.add_trim_pass(TrimPass::new(1000.0, ColorGrade::identity()));
        let bytes = scene.to_bytes();
        let decoded = SceneGradingMetadata::from_bytes(&bytes).expect("decode");
        assert_eq!(decoded.trim_passes.len(), 2);
        assert!((decoded.trim_passes[0].display_peak_nits - 100.0).abs() < 1e-5);
        assert!((decoded.trim_passes[1].display_peak_nits - 1000.0).abs() < 1e-5);
    }

    #[test]
    fn test_scene_from_bytes_empty_returns_err() {
        let result = SceneGradingMetadata::from_bytes(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_scene_from_bytes_truncated_returns_err() {
        let result = SceneGradingMetadata::from_bytes(&[0u8; 10]);
        assert!(result.is_err());
    }

    #[test]
    fn test_trim_for_display_closest() {
        let mut scene = identity_scene(1, 0, 1000);
        scene.add_trim_pass(TrimPass::new(100.0, ColorGrade::identity()));
        scene.add_trim_pass(TrimPass::new(400.0, ColorGrade::identity()));
        scene.add_trim_pass(TrimPass::new(1000.0, ColorGrade::identity()));
        // 600 is closer to 400 (diff=200) than 1000 (diff=400).
        let tp = scene.trim_for_display(600.0).expect("trim pass found");
        assert!((tp.display_peak_nits - 400.0).abs() < 1e-5);
    }

    #[test]
    fn test_trim_for_display_no_passes_returns_none() {
        let scene = identity_scene(1, 0, 1000);
        assert!(scene.trim_for_display(100.0).is_none());
    }

    // ── SceneGradingDatabase ──────────────────────────────────────────────────

    #[test]
    fn test_database_insert_and_lookup_in_range() {
        let mut db = SceneGradingDatabase::new();
        db.insert(identity_scene(1, 0, 90_000));
        db.insert(identity_scene(2, 90_000, 180_000));
        let scene = db.lookup_by_pts(45_000).expect("scene found");
        assert_eq!(scene.scene_id, 1);
        let scene2 = db.lookup_by_pts(120_000).expect("scene2 found");
        assert_eq!(scene2.scene_id, 2);
    }

    #[test]
    fn test_database_lookup_out_of_range_returns_closest() {
        let mut db = SceneGradingDatabase::new();
        db.insert(identity_scene(10, 1000, 2000));
        db.insert(identity_scene(20, 5000, 6000));
        // pts=0 is before both; closest pts_start is 1000.
        let scene = db.lookup_by_pts(0).expect("closest scene");
        assert_eq!(scene.scene_id, 10);
    }

    #[test]
    fn test_database_remove_by_scene_id() {
        let mut db = SceneGradingDatabase::new();
        db.insert(identity_scene(1, 0, 1000));
        db.insert(identity_scene(2, 1000, 2000));
        assert!(db.remove_by_scene_id(1));
        assert_eq!(db.len(), 1);
        assert!(!db.remove_by_scene_id(99));
    }

    #[test]
    fn test_database_round_trip() {
        let mut db = SceneGradingDatabase::new();
        db.insert(identity_scene(1, 0, 100));
        db.insert(identity_scene(2, 100, 200));
        let bytes = db.to_bytes();
        let decoded = SceneGradingDatabase::from_bytes(&bytes).expect("decode");
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded.scenes[0].scene_id, 1);
        assert_eq!(decoded.scenes[1].scene_id, 2);
    }

    #[test]
    fn test_database_from_bytes_empty_slice_returns_err() {
        let result = SceneGradingDatabase::from_bytes(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_database_is_empty() {
        let db = SceneGradingDatabase::new();
        assert!(db.is_empty());
        assert_eq!(db.len(), 0);
    }
}
