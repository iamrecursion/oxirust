#![allow(dead_code)]
//! False color mapping for exposure, focus, and motion visualization.
//!
//! Provides flexible false color processing with configurable zones
//! and color mappings for broadcast camera operators.

/// Categorizes the purpose of a false color visualization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FalseColorMapping {
    /// Exposure false color (overexposed / underexposed zones).
    Exposure,
    /// Focus peaking false color (sharp-edge highlight).
    Focus,
    /// Motion vector false color (pixel displacement magnitude).
    Motion,
}

impl FalseColorMapping {
    /// Returns a short human-readable label for the mapping type.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Exposure => "Exposure",
            Self::Focus => "Focus",
            Self::Motion => "Motion",
        }
    }

    /// Returns whether this mapping type uses an overlay rather than a full
    /// frame replace.
    #[must_use]
    pub fn is_overlay(self) -> bool {
        matches!(self, Self::Focus | Self::Motion)
    }
}

/// An RGBA color expressed as four bytes (r, g, b, a).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rgba(pub u8, pub u8, pub u8, pub u8);

impl Rgba {
    /// Fully opaque white.
    pub const WHITE: Self = Self(255, 255, 255, 255);
    /// Fully opaque black.
    pub const BLACK: Self = Self(0, 0, 0, 255);
    /// Fully transparent.
    pub const TRANSPARENT: Self = Self(0, 0, 0, 0);
}

/// A single threshold entry in a false color scale.
///
/// When a luma value falls within `[lower, upper)` the corresponding
/// `color` is applied.
#[derive(Debug, Clone)]
pub struct FalseColorThreshold {
    /// Lower bound (inclusive), in IRE units `[0, 109]`.
    pub lower: f32,
    /// Upper bound (exclusive), in IRE units `[0, 109]`.
    pub upper: f32,
    /// Color to apply for luma values in this range.
    pub color: Rgba,
}

impl FalseColorThreshold {
    /// Creates a new threshold entry.
    #[must_use]
    pub fn new(lower: f32, upper: f32, color: Rgba) -> Self {
        Self {
            lower,
            upper,
            color,
        }
    }

    /// Returns the color this threshold maps to, or `None` when the luma
    /// value falls outside the range.
    #[must_use]
    pub fn maps_to_color(&self, luma_ire: f32) -> Option<Rgba> {
        if luma_ire >= self.lower && luma_ire < self.upper {
            Some(self.color)
        } else {
            None
        }
    }

    /// Returns the midpoint IRE value for this zone.
    #[must_use]
    pub fn midpoint(&self) -> f32 {
        (self.lower + self.upper) * 0.5
    }
}

/// A complete false color scale composed of ordered threshold entries.
#[derive(Debug, Clone, Default)]
pub struct FalseColorScale {
    thresholds: Vec<FalseColorThreshold>,
}

impl FalseColorScale {
    /// Creates an empty scale (all pixels pass through unchanged).
    #[must_use]
    pub fn new() -> Self {
        Self {
            thresholds: Vec::new(),
        }
    }

    /// Adds a threshold zone to the scale.
    pub fn add_threshold(&mut self, t: FalseColorThreshold) {
        self.thresholds.push(t);
    }

    /// Returns the number of defined threshold zones.
    #[must_use]
    pub fn zone_count(&self) -> usize {
        self.thresholds.len()
    }

    /// Looks up the color for a given luma IRE value.
    ///
    /// Returns `None` if no threshold matches.
    #[must_use]
    pub fn lookup(&self, luma_ire: f32) -> Option<Rgba> {
        for t in &self.thresholds {
            if let Some(c) = t.maps_to_color(luma_ire) {
                return Some(c);
            }
        }
        None
    }
}

/// Processes video frame data applying false color visualization.
#[derive(Debug, Clone)]
pub struct FalseColorProcessor {
    mapping: FalseColorMapping,
    scale: FalseColorScale,
}

impl FalseColorProcessor {
    /// Creates a new processor with a given mapping type and color scale.
    #[must_use]
    pub fn new(mapping: FalseColorMapping, scale: FalseColorScale) -> Self {
        Self { mapping, scale }
    }

    /// Returns the mapping type used by this processor.
    #[must_use]
    pub fn mapping(&self) -> FalseColorMapping {
        self.mapping
    }

    /// Applies false color to a single luma IRE value.
    ///
    /// Returns the replacement color, or `None` if the value is in a
    /// "neutral" zone (no threshold matches).
    #[must_use]
    pub fn apply(&self, luma_ire: f32) -> Option<Rgba> {
        self.scale.lookup(luma_ire)
    }

    /// Processes a full luma plane (values in `[0.0, 109.0]` IRE) and
    /// returns per-pixel replacement colors.
    ///
    /// Pixels for which no threshold matches are returned as `None`.
    #[must_use]
    pub fn apply_frame(&self, luma_plane: &[f32]) -> Vec<Option<Rgba>> {
        luma_plane.iter().map(|&v| self.apply(v)).collect()
    }

    /// Calculates the fraction of pixels that fall within any threshold zone.
    ///
    /// Returns a value in `[0.0, 1.0]`.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn zone_coverage_pct(&self, luma_plane: &[f32]) -> f32 {
        if luma_plane.is_empty() {
            return 0.0;
        }
        let hits = luma_plane
            .iter()
            .filter(|&&v| self.apply(v).is_some())
            .count();
        hits as f32 / luma_plane.len() as f32
    }
}

// =============================================================================
// Custom user-defined color LUT support for false color mapping
// =============================================================================

/// A user-defined color look-up table for false color mapping.
///
/// Maps 256 luma code values (0-255) to RGBA colors.  Users can provide
/// custom artistic or diagnostic palettes beyond the threshold-based approach.
#[derive(Debug, Clone)]
pub struct FalseColorLut {
    /// 256 RGBA entries, one per 8-bit code value.
    entries: [Rgba; 256],
    /// Optional human-readable name for the LUT.
    name: String,
}

impl FalseColorLut {
    /// Creates a new LUT with all entries set to transparent.
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self {
            entries: [Rgba::TRANSPARENT; 256],
            name: name.to_string(),
        }
    }

    /// Creates a LUT from a full 256-entry array.
    #[must_use]
    pub fn from_entries(name: &str, entries: [Rgba; 256]) -> Self {
        Self {
            entries,
            name: name.to_string(),
        }
    }

    /// Creates a LUT by linearly interpolating between anchor points.
    ///
    /// Anchor points are `(code_value, color)` pairs sorted by code value.
    /// Values between anchors are linearly interpolated in RGBA space.
    ///
    /// Returns `None` if anchors is empty.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn from_anchors(name: &str, anchors: &[(u8, Rgba)]) -> Option<Self> {
        if anchors.is_empty() {
            return None;
        }

        let mut entries = [Rgba::TRANSPARENT; 256];

        if anchors.len() == 1 {
            entries.fill(anchors[0].1);
            return Some(Self::from_entries(name, entries));
        }

        // Sort anchors by code value
        let mut sorted: Vec<(u8, Rgba)> = anchors.to_vec();
        sorted.sort_by_key(|(cv, _)| *cv);

        // Fill before first anchor
        let first_color = sorted[0].1;
        for entry in entries.iter_mut().take(sorted[0].0 as usize) {
            *entry = first_color;
        }

        // Fill after last anchor
        let last_idx = sorted.len() - 1;
        let last_color = sorted[last_idx].1;
        for entry in entries.iter_mut().skip(sorted[last_idx].0 as usize) {
            *entry = last_color;
        }

        // Interpolate between adjacent anchors
        for pair in sorted.windows(2) {
            let (cv0, c0) = pair[0];
            let (cv1, c1) = pair[1];
            let range = u16::from(cv1) - u16::from(cv0);
            if range == 0 {
                continue;
            }
            for cv in cv0..=cv1 {
                let t = f32::from(cv - cv0) / range as f32;
                let inv_t = 1.0 - t;
                entries[cv as usize] = Rgba(
                    (f32::from(c0.0) * inv_t + f32::from(c1.0) * t) as u8,
                    (f32::from(c0.1) * inv_t + f32::from(c1.1) * t) as u8,
                    (f32::from(c0.2) * inv_t + f32::from(c1.2) * t) as u8,
                    (f32::from(c0.3) * inv_t + f32::from(c1.3) * t) as u8,
                );
            }
        }

        Some(Self::from_entries(name, entries))
    }

    /// Creates a standard "heat map" LUT (black → blue → cyan → green → yellow → red → white).
    #[must_use]
    pub fn heat_map() -> Self {
        Self::from_anchors(
            "Heat Map",
            &[
                (0, Rgba(0, 0, 0, 255)),
                (42, Rgba(0, 0, 255, 255)),
                (85, Rgba(0, 255, 255, 255)),
                (127, Rgba(0, 255, 0, 255)),
                (170, Rgba(255, 255, 0, 255)),
                (212, Rgba(255, 0, 0, 255)),
                (255, Rgba(255, 255, 255, 255)),
            ],
        )
        .unwrap_or_else(|| Self::new("Heat Map"))
    }

    /// Creates a grayscale LUT where each code value maps to its gray equivalent.
    #[must_use]
    pub fn grayscale() -> Self {
        let mut entries = [Rgba::BLACK; 256];
        for (i, entry) in entries.iter_mut().enumerate() {
            let v = i as u8;
            *entry = Rgba(v, v, v, 255);
        }
        Self::from_entries("Grayscale", entries)
    }

    /// Creates a "traffic light" LUT for exposure: blue (dark) → green (good) → red (hot).
    #[must_use]
    pub fn traffic_light() -> Self {
        Self::from_anchors(
            "Traffic Light",
            &[
                (0, Rgba(0, 0, 128, 255)),     // dark blue
                (40, Rgba(0, 0, 255, 255)),    // blue
                (80, Rgba(0, 200, 0, 255)),    // green
                (128, Rgba(0, 255, 0, 255)),   // bright green
                (180, Rgba(255, 255, 0, 255)), // yellow
                (220, Rgba(255, 128, 0, 255)), // orange
                (255, Rgba(255, 0, 0, 255)),   // red
            ],
        )
        .unwrap_or_else(|| Self::new("Traffic Light"))
    }

    /// Looks up the color for a given 8-bit code value.
    #[must_use]
    pub fn lookup(&self, code_value: u8) -> Rgba {
        self.entries[code_value as usize]
    }

    /// Looks up the color for a normalized luma value [0.0, 1.0].
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn lookup_normalized(&self, luma: f32) -> Rgba {
        let cv = (luma.clamp(0.0, 1.0) * 255.0) as u8;
        self.lookup(cv)
    }

    /// Returns the name of this LUT.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Sets a single entry in the LUT.
    pub fn set_entry(&mut self, code_value: u8, color: Rgba) {
        self.entries[code_value as usize] = color;
    }

    /// Returns the raw entries array.
    #[must_use]
    pub fn entries(&self) -> &[Rgba; 256] {
        &self.entries
    }

    /// Returns the number of non-transparent entries.
    #[must_use]
    pub fn active_entries(&self) -> usize {
        self.entries.iter().filter(|e| e.3 > 0).count()
    }
}

/// Applies a user-defined LUT to a luma plane.
///
/// Input: normalized luma values [0.0, 1.0].
/// Output: per-pixel RGBA colors from the LUT.
#[must_use]
pub fn apply_lut_to_luma_plane(lut: &FalseColorLut, luma_plane: &[f32]) -> Vec<Rgba> {
    luma_plane
        .iter()
        .map(|&v| lut.lookup_normalized(v))
        .collect()
}

/// Applies a user-defined LUT to an RGB24 frame via luma conversion.
///
/// Returns a flat RGBA buffer (4 bytes per pixel).
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn apply_lut_to_rgb24_frame(
    lut: &FalseColorLut,
    frame: &[u8],
    width: u32,
    height: u32,
) -> Vec<u8> {
    let pixel_count = (width as usize) * (height as usize);
    let mut output = Vec::with_capacity(pixel_count * 4);

    for i in 0..pixel_count {
        let base = i * 3;
        if base + 2 >= frame.len() {
            output.extend_from_slice(&[0, 0, 0, 255]);
            continue;
        }
        let r = frame[base];
        let g = frame[base + 1];
        let b = frame[base + 2];
        // BT.709 luma
        let luma = (0.2126 * f32::from(r) + 0.7152 * f32::from(g) + 0.0722 * f32::from(b))
            .clamp(0.0, 255.0);
        let cv = luma as u8;
        let color = lut.lookup(cv);
        output.push(color.0);
        output.push(color.1);
        output.push(color.2);
        output.push(color.3);
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exposure_scale() -> FalseColorScale {
        let mut s = FalseColorScale::new();
        // underexposed zone: 0–20 IRE → blue
        s.add_threshold(FalseColorThreshold::new(0.0, 20.0, Rgba(0, 0, 255, 255)));
        // skin tone zone: 55–65 IRE → pink
        s.add_threshold(FalseColorThreshold::new(
            55.0,
            65.0,
            Rgba(255, 128, 128, 255),
        ));
        // overexposed zone: 90–109 IRE → red
        s.add_threshold(FalseColorThreshold::new(90.0, 109.0, Rgba(255, 0, 0, 255)));
        s
    }

    #[test]
    fn test_mapping_label_exposure() {
        assert_eq!(FalseColorMapping::Exposure.label(), "Exposure");
    }

    #[test]
    fn test_mapping_label_focus() {
        assert_eq!(FalseColorMapping::Focus.label(), "Focus");
    }

    #[test]
    fn test_mapping_label_motion() {
        assert_eq!(FalseColorMapping::Motion.label(), "Motion");
    }

    #[test]
    fn test_mapping_is_overlay_focus() {
        assert!(FalseColorMapping::Focus.is_overlay());
    }

    #[test]
    fn test_mapping_is_overlay_exposure_false() {
        assert!(!FalseColorMapping::Exposure.is_overlay());
    }

    #[test]
    fn test_threshold_maps_to_color_hit() {
        let t = FalseColorThreshold::new(0.0, 20.0, Rgba(0, 0, 255, 255));
        assert_eq!(t.maps_to_color(10.0), Some(Rgba(0, 0, 255, 255)));
    }

    #[test]
    fn test_threshold_maps_to_color_miss() {
        let t = FalseColorThreshold::new(0.0, 20.0, Rgba(0, 0, 255, 255));
        assert_eq!(t.maps_to_color(50.0), None);
    }

    #[test]
    fn test_threshold_lower_bound_inclusive() {
        let t = FalseColorThreshold::new(20.0, 40.0, Rgba(255, 255, 0, 255));
        assert!(t.maps_to_color(20.0).is_some());
    }

    #[test]
    fn test_threshold_upper_bound_exclusive() {
        let t = FalseColorThreshold::new(20.0, 40.0, Rgba(255, 255, 0, 255));
        assert!(t.maps_to_color(40.0).is_none());
    }

    #[test]
    fn test_threshold_midpoint() {
        let t = FalseColorThreshold::new(10.0, 30.0, Rgba::WHITE);
        assert!((t.midpoint() - 20.0).abs() < 1e-6);
    }

    #[test]
    fn test_scale_zone_count() {
        let s = exposure_scale();
        assert_eq!(s.zone_count(), 3);
    }

    #[test]
    fn test_processor_apply_underexposed() {
        let proc = FalseColorProcessor::new(FalseColorMapping::Exposure, exposure_scale());
        assert_eq!(proc.apply(10.0), Some(Rgba(0, 0, 255, 255)));
    }

    #[test]
    fn test_processor_apply_neutral_zone() {
        let proc = FalseColorProcessor::new(FalseColorMapping::Exposure, exposure_scale());
        // 70 IRE has no threshold
        assert_eq!(proc.apply(70.0), None);
    }

    #[test]
    fn test_processor_zone_coverage_pct() {
        let proc = FalseColorProcessor::new(FalseColorMapping::Exposure, exposure_scale());
        // 2 out of 4 values fall in a zone (10 and 95)
        let plane = vec![10.0, 50.0, 70.0, 95.0];
        let pct = proc.zone_coverage_pct(&plane);
        assert!((pct - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_processor_zone_coverage_empty() {
        let proc = FalseColorProcessor::new(FalseColorMapping::Exposure, exposure_scale());
        assert_eq!(proc.zone_coverage_pct(&[]), 0.0);
    }

    #[test]
    fn test_rgba_constants() {
        assert_eq!(Rgba::WHITE, Rgba(255, 255, 255, 255));
        assert_eq!(Rgba::BLACK, Rgba(0, 0, 0, 255));
        assert_eq!(Rgba::TRANSPARENT.3, 0);
    }

    // ── FalseColorLut tests ──────────────────────────────────────────

    #[test]
    fn test_lut_new_all_transparent() {
        let lut = FalseColorLut::new("test");
        assert_eq!(lut.name(), "test");
        assert_eq!(lut.active_entries(), 0);
        assert_eq!(lut.lookup(128), Rgba::TRANSPARENT);
    }

    #[test]
    fn test_lut_from_entries() {
        let mut entries = [Rgba::BLACK; 256];
        entries[100] = Rgba(255, 0, 0, 255);
        let lut = FalseColorLut::from_entries("custom", entries);
        assert_eq!(lut.lookup(100), Rgba(255, 0, 0, 255));
        assert_eq!(lut.lookup(0), Rgba::BLACK);
    }

    #[test]
    fn test_lut_set_entry() {
        let mut lut = FalseColorLut::new("test");
        lut.set_entry(50, Rgba(0, 255, 0, 255));
        assert_eq!(lut.lookup(50), Rgba(0, 255, 0, 255));
    }

    #[test]
    fn test_lut_from_anchors_single() {
        let lut = FalseColorLut::from_anchors("single", &[(128, Rgba(255, 0, 0, 255))]);
        assert!(lut.is_some());
        let lut = lut.expect("single anchor LUT");
        // All entries should be the single color
        assert_eq!(lut.lookup(0), Rgba(255, 0, 0, 255));
        assert_eq!(lut.lookup(255), Rgba(255, 0, 0, 255));
    }

    #[test]
    fn test_lut_from_anchors_empty() {
        assert!(FalseColorLut::from_anchors("empty", &[]).is_none());
    }

    #[test]
    fn test_lut_from_anchors_two_points() {
        let lut = FalseColorLut::from_anchors(
            "gradient",
            &[(0, Rgba(0, 0, 0, 255)), (255, Rgba(255, 255, 255, 255))],
        )
        .expect("two-point LUT");

        // Endpoints
        assert_eq!(lut.lookup(0), Rgba(0, 0, 0, 255));
        assert_eq!(lut.lookup(255), Rgba(255, 255, 255, 255));

        // Midpoint should be ~128
        let mid = lut.lookup(128);
        assert!((mid.0 as i16 - 128).abs() < 3);
    }

    #[test]
    fn test_lut_from_anchors_interpolation() {
        let lut = FalseColorLut::from_anchors(
            "interp",
            &[
                (0, Rgba(0, 0, 0, 255)),
                (100, Rgba(100, 0, 0, 255)),
                (200, Rgba(100, 200, 0, 255)),
                (255, Rgba(255, 255, 255, 255)),
            ],
        )
        .expect("multi-anchor LUT");

        // At anchor points, colors should match exactly
        assert_eq!(lut.lookup(0), Rgba(0, 0, 0, 255));
        assert_eq!(lut.lookup(100), Rgba(100, 0, 0, 255));
        assert_eq!(lut.lookup(200), Rgba(100, 200, 0, 255));
    }

    #[test]
    fn test_heat_map_lut() {
        let lut = FalseColorLut::heat_map();
        assert_eq!(lut.name(), "Heat Map");
        assert!(lut.active_entries() == 256);
        // Black at 0
        assert_eq!(lut.lookup(0), Rgba(0, 0, 0, 255));
    }

    #[test]
    fn test_grayscale_lut() {
        let lut = FalseColorLut::grayscale();
        assert_eq!(lut.name(), "Grayscale");
        assert_eq!(lut.lookup(0), Rgba(0, 0, 0, 255));
        assert_eq!(lut.lookup(128), Rgba(128, 128, 128, 255));
        assert_eq!(lut.lookup(255), Rgba(255, 255, 255, 255));
    }

    #[test]
    fn test_traffic_light_lut() {
        let lut = FalseColorLut::traffic_light();
        assert_eq!(lut.name(), "Traffic Light");
        assert!(lut.active_entries() == 256);
    }

    #[test]
    fn test_lookup_normalized() {
        let lut = FalseColorLut::grayscale();
        let c = lut.lookup_normalized(0.5);
        // 0.5 * 255 = 127
        assert_eq!(c.0, 127);
    }

    #[test]
    fn test_lookup_normalized_clamp() {
        let lut = FalseColorLut::grayscale();
        let c_low = lut.lookup_normalized(-1.0);
        assert_eq!(c_low.0, 0);
        let c_high = lut.lookup_normalized(2.0);
        assert_eq!(c_high.0, 255);
    }

    #[test]
    fn test_active_entries_count() {
        let mut lut = FalseColorLut::new("partial");
        lut.set_entry(10, Rgba(255, 0, 0, 255));
        lut.set_entry(20, Rgba(0, 255, 0, 255));
        assert_eq!(lut.active_entries(), 2);
    }

    #[test]
    fn test_entries_accessor() {
        let lut = FalseColorLut::grayscale();
        let entries = lut.entries();
        assert_eq!(entries.len(), 256);
        assert_eq!(entries[128], Rgba(128, 128, 128, 255));
    }

    // ── apply_lut tests ──────────────────────────────────────────────

    #[test]
    fn test_apply_lut_to_luma_plane() {
        let lut = FalseColorLut::grayscale();
        let plane = vec![0.0, 0.5, 1.0];
        let result = apply_lut_to_luma_plane(&lut, &plane);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].0, 0);
        assert_eq!(result[1].0, 127);
        assert_eq!(result[2].0, 255);
    }

    #[test]
    fn test_apply_lut_to_luma_plane_empty() {
        let lut = FalseColorLut::grayscale();
        let result = apply_lut_to_luma_plane(&lut, &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_apply_lut_to_rgb24_frame() {
        let lut = FalseColorLut::heat_map();
        // 4 pixels: black, mid-gray, near-white, white
        let frame = vec![
            0, 0, 0, // black
            128, 128, 128, // mid
            200, 200, 200, // bright
            255, 255, 255, // white
        ];
        let result = apply_lut_to_rgb24_frame(&lut, &frame, 4, 1);
        assert_eq!(result.len(), 4 * 4); // 4 pixels × 4 channels (RGBA)
    }

    #[test]
    fn test_apply_lut_to_rgb24_frame_short_data() {
        let lut = FalseColorLut::grayscale();
        // Frame too short: expect black fill for missing pixels
        let frame = vec![128, 128, 128]; // 1 pixel only
        let result = apply_lut_to_rgb24_frame(&lut, &frame, 2, 1);
        assert_eq!(result.len(), 2 * 4);
    }
}
