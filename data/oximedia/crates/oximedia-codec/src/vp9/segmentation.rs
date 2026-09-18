//! VP9 Segmentation handling.
//!
//! This module provides segmentation support for VP9 decoding.
//! Segmentation allows different encoding parameters for different
//! regions of the frame.

#![forbid(unsafe_code)]
#![allow(dead_code)]

use super::mv::MvRefType;

/// Maximum number of segments.
pub const MAX_SEGMENTS: usize = 8;

/// Number of segment features.
pub const SEG_FEATURES: usize = 4;

/// Number of prediction probabilities.
pub const SEG_PRED_PROBS: usize = 3;

/// Segment feature types.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SegmentFeature {
    /// Alternate quantizer.
    AltQ = 0,
    /// Alternate loop filter level.
    AltLf = 1,
    /// Reference frame override.
    RefFrame = 2,
    /// Skip residual coding.
    Skip = 3,
}

impl SegmentFeature {
    /// All segment features.
    pub const ALL: [SegmentFeature; SEG_FEATURES] = [
        SegmentFeature::AltQ,
        SegmentFeature::AltLf,
        SegmentFeature::RefFrame,
        SegmentFeature::Skip,
    ];

    /// Maximum data values for each feature.
    const MAX_DATA: [i16; SEG_FEATURES] = [255, 63, 3, 0];

    /// Feature data is signed for these features.
    const SIGNED: [bool; SEG_FEATURES] = [true, true, false, false];

    /// Number of bits for each feature data.
    const DATA_BITS: [u8; SEG_FEATURES] = [8, 6, 2, 0];

    /// Converts from u8 value to `SegmentFeature`.
    #[must_use]
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::AltQ),
            1 => Some(Self::AltLf),
            2 => Some(Self::RefFrame),
            3 => Some(Self::Skip),
            _ => None,
        }
    }

    /// Returns the maximum data value for this feature.
    #[must_use]
    pub const fn max_data(&self) -> i16 {
        Self::MAX_DATA[*self as usize]
    }

    /// Returns true if this feature uses signed data.
    #[must_use]
    pub const fn is_signed(&self) -> bool {
        Self::SIGNED[*self as usize]
    }

    /// Returns the number of bits for this feature's data.
    #[must_use]
    pub const fn data_bits(&self) -> u8 {
        Self::DATA_BITS[*self as usize]
    }

    /// Returns the feature index.
    #[must_use]
    pub const fn index(&self) -> usize {
        *self as usize
    }
}

impl From<SegmentFeature> for u8 {
    fn from(value: SegmentFeature) -> Self {
        value as u8
    }
}

/// Segment data for a single segment.
#[derive(Clone, Copy, Debug, Default)]
pub struct SegmentData {
    /// Feature enabled flags.
    pub feature_enabled: [bool; SEG_FEATURES],
    /// Feature data values.
    pub feature_data: [i16; SEG_FEATURES],
}

impl SegmentData {
    /// Creates a new segment data with defaults.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            feature_enabled: [false; SEG_FEATURES],
            feature_data: [0; SEG_FEATURES],
        }
    }

    /// Returns true if a feature is enabled.
    #[must_use]
    pub const fn is_feature_enabled(&self, feature: SegmentFeature) -> bool {
        self.feature_enabled[feature.index()]
    }

    /// Returns the data for a feature.
    #[must_use]
    pub const fn feature_data(&self, feature: SegmentFeature) -> i16 {
        self.feature_data[feature.index()]
    }

    /// Enables a feature with the given data.
    pub fn enable_feature(&mut self, feature: SegmentFeature, data: i16) {
        let idx = feature.index();
        self.feature_enabled[idx] = true;
        self.feature_data[idx] = data.clamp(-feature.max_data(), feature.max_data());
    }

    /// Disables a feature.
    pub fn disable_feature(&mut self, feature: SegmentFeature) {
        let idx = feature.index();
        self.feature_enabled[idx] = false;
        self.feature_data[idx] = 0;
    }

    /// Returns true if skip is forced for this segment.
    #[must_use]
    pub const fn is_skip_forced(&self) -> bool {
        self.feature_enabled[SegmentFeature::Skip as usize]
    }

    /// Returns the reference frame if overridden.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn reference_frame(&self) -> Option<MvRefType> {
        if self.feature_enabled[SegmentFeature::RefFrame as usize] {
            let ref_idx = self.feature_data[SegmentFeature::RefFrame as usize] as u8;
            MvRefType::from_u8(ref_idx)
        } else {
            None
        }
    }

    /// Returns the quantizer delta if set.
    #[must_use]
    pub const fn qindex_delta(&self) -> Option<i16> {
        if self.feature_enabled[SegmentFeature::AltQ as usize] {
            Some(self.feature_data[SegmentFeature::AltQ as usize])
        } else {
            None
        }
    }

    /// Returns the loop filter delta if set.
    #[must_use]
    pub const fn lf_delta(&self) -> Option<i16> {
        if self.feature_enabled[SegmentFeature::AltLf as usize] {
            Some(self.feature_data[SegmentFeature::AltLf as usize])
        } else {
            None
        }
    }

    /// Clears all features.
    pub fn clear(&mut self) {
        self.feature_enabled = [false; SEG_FEATURES];
        self.feature_data = [0; SEG_FEATURES];
    }
}

/// Segmentation information for the frame.
#[derive(Clone, Debug, Default)]
#[allow(clippy::struct_excessive_bools)]
pub struct Segmentation {
    /// Segmentation enabled.
    pub enabled: bool,
    /// Update map (segment IDs are transmitted).
    pub update_map: bool,
    /// Temporal update (use previous frame segment IDs).
    pub temporal_update: bool,
    /// Update data (segment features are transmitted).
    pub update_data: bool,
    /// Absolute or delta mode for features.
    pub abs_delta: bool,
    /// Segment data for each segment.
    pub segments: [SegmentData; MAX_SEGMENTS],
    /// Prediction probabilities for temporal update.
    pub pred_probs: [u8; SEG_PRED_PROBS],
    /// Tree probabilities for segment ID.
    pub tree_probs: [u8; MAX_SEGMENTS - 1],
}

impl Segmentation {
    /// Creates new segmentation with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            enabled: false,
            update_map: false,
            temporal_update: false,
            update_data: false,
            abs_delta: false,
            segments: [SegmentData::new(); MAX_SEGMENTS],
            pred_probs: [255; SEG_PRED_PROBS],
            tree_probs: [128; MAX_SEGMENTS - 1],
        }
    }

    /// Resets segmentation to disabled state.
    pub fn reset(&mut self) {
        self.enabled = false;
        self.update_map = false;
        self.temporal_update = false;
        self.update_data = false;
        self.abs_delta = false;
        for segment in &mut self.segments {
            segment.clear();
        }
        self.pred_probs = [255; SEG_PRED_PROBS];
        self.tree_probs = [128; MAX_SEGMENTS - 1];
    }

    /// Clears all segment features without disabling segmentation.
    pub fn clear_features(&mut self) {
        for segment in &mut self.segments {
            segment.clear();
        }
    }

    /// Returns true if segmentation is active.
    #[must_use]
    pub const fn is_active(&self) -> bool {
        self.enabled
    }

    /// Returns the segment data for a segment ID.
    #[must_use]
    pub const fn segment(&self, segment_id: u8) -> &SegmentData {
        &self.segments[(segment_id as usize) & (MAX_SEGMENTS - 1)]
    }

    /// Returns mutable segment data for a segment ID.
    #[must_use]
    pub fn segment_mut(&mut self, segment_id: u8) -> &mut SegmentData {
        &mut self.segments[(segment_id as usize) & (MAX_SEGMENTS - 1)]
    }

    /// Checks if any segment has a specific feature enabled.
    #[must_use]
    pub fn any_segment_has_feature(&self, feature: SegmentFeature) -> bool {
        self.segments.iter().any(|s| s.is_feature_enabled(feature))
    }

    /// Returns true if any segment forces skip.
    #[must_use]
    pub fn has_skip_segment(&self) -> bool {
        self.any_segment_has_feature(SegmentFeature::Skip)
    }

    /// Returns true if any segment overrides the reference frame.
    #[must_use]
    pub fn has_ref_frame_segment(&self) -> bool {
        self.any_segment_has_feature(SegmentFeature::RefFrame)
    }

    /// Returns the quantizer index for a segment.
    ///
    /// If the segment has an ALT_Q feature, returns the adjusted qindex.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn get_qindex(&self, segment_id: u8, base_qindex: u8) -> u8 {
        let segment = self.segment(segment_id);

        if let Some(delta) = segment.qindex_delta() {
            if self.abs_delta {
                (delta as i32).clamp(0, 255) as u8
            } else {
                (i32::from(base_qindex) + i32::from(delta)).clamp(0, 255) as u8
            }
        } else {
            base_qindex
        }
    }

    /// Returns the loop filter level for a segment.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn get_lf_level(&self, segment_id: u8, base_level: u8) -> u8 {
        let segment = self.segment(segment_id);

        if let Some(delta) = segment.lf_delta() {
            if self.abs_delta {
                (delta as i32).clamp(0, 63) as u8
            } else {
                (i32::from(base_level) + i32::from(delta)).clamp(0, 63) as u8
            }
        } else {
            base_level
        }
    }

    /// Sets a feature for a segment.
    pub fn set_feature(&mut self, segment_id: u8, feature: SegmentFeature, data: i16) {
        self.segment_mut(segment_id).enable_feature(feature, data);
    }

    /// Clears a feature for a segment.
    pub fn clear_feature(&mut self, segment_id: u8, feature: SegmentFeature) {
        self.segment_mut(segment_id).disable_feature(feature);
    }

    /// Returns the prediction probability for a given context.
    #[must_use]
    pub fn pred_prob(&self, context: usize) -> u8 {
        if context < SEG_PRED_PROBS {
            self.pred_probs[context]
        } else {
            128
        }
    }
}

/// Segment ID map for a frame.
#[derive(Clone, Debug)]
pub struct SegmentMap {
    /// Width in 4x4 blocks.
    mi_cols: usize,
    /// Height in 4x4 blocks.
    mi_rows: usize,
    /// Segment IDs for each 4x4 block.
    ids: Vec<u8>,
}

impl SegmentMap {
    /// Creates a new segment map.
    #[must_use]
    pub fn new(mi_cols: usize, mi_rows: usize) -> Self {
        Self {
            mi_cols,
            mi_rows,
            ids: vec![0; mi_cols * mi_rows],
        }
    }

    /// Returns the segment ID at the given position.
    #[must_use]
    pub fn get(&self, mi_col: usize, mi_row: usize) -> u8 {
        if mi_col < self.mi_cols && mi_row < self.mi_rows {
            self.ids[mi_row * self.mi_cols + mi_col]
        } else {
            0
        }
    }

    /// Sets the segment ID at the given position.
    pub fn set(&mut self, mi_col: usize, mi_row: usize, segment_id: u8) {
        if mi_col < self.mi_cols && mi_row < self.mi_rows {
            self.ids[mi_row * self.mi_cols + mi_col] = segment_id & 7;
        }
    }

    /// Fills a block region with a segment `ID`.
    pub fn fill_block(&mut self, mi_col: usize, mi_row: usize, w: usize, h: usize, segment_id: u8) {
        for row in mi_row..mi_row.saturating_add(h).min(self.mi_rows) {
            for col in mi_col..mi_col.saturating_add(w).min(self.mi_cols) {
                self.ids[row * self.mi_cols + col] = segment_id & 7;
            }
        }
    }

    /// Clears all segment IDs to zero.
    pub fn clear(&mut self) {
        self.ids.fill(0);
    }

    /// Copies from another segment map.
    pub fn copy_from(&mut self, other: &SegmentMap) {
        if self.mi_cols == other.mi_cols && self.mi_rows == other.mi_rows {
            self.ids.copy_from_slice(&other.ids);
        }
    }

    /// Returns the dimensions.
    #[must_use]
    pub const fn dimensions(&self) -> (usize, usize) {
        (self.mi_cols, self.mi_rows)
    }
}

impl Default for SegmentMap {
    fn default() -> Self {
        Self::new(0, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segment_feature() {
        assert_eq!(SegmentFeature::AltQ.index(), 0);
        assert_eq!(SegmentFeature::Skip.index(), 3);
        assert!(SegmentFeature::AltQ.is_signed());
        assert!(!SegmentFeature::RefFrame.is_signed());
    }

    #[test]
    fn test_segment_feature_from_u8() {
        assert_eq!(SegmentFeature::from_u8(0), Some(SegmentFeature::AltQ));
        assert_eq!(SegmentFeature::from_u8(3), Some(SegmentFeature::Skip));
        assert_eq!(SegmentFeature::from_u8(4), None);
    }

    #[test]
    fn test_segment_data() {
        let mut data = SegmentData::new();
        assert!(!data.is_feature_enabled(SegmentFeature::AltQ));

        data.enable_feature(SegmentFeature::AltQ, -10);
        assert!(data.is_feature_enabled(SegmentFeature::AltQ));
        assert_eq!(data.feature_data(SegmentFeature::AltQ), -10);
        assert_eq!(data.qindex_delta(), Some(-10));

        data.disable_feature(SegmentFeature::AltQ);
        assert!(!data.is_feature_enabled(SegmentFeature::AltQ));
    }

    #[test]
    fn test_segment_data_skip() {
        let mut data = SegmentData::new();
        assert!(!data.is_skip_forced());

        data.enable_feature(SegmentFeature::Skip, 0);
        assert!(data.is_skip_forced());
    }

    #[test]
    fn test_segment_data_ref_frame() {
        let mut data = SegmentData::new();
        assert!(data.reference_frame().is_none());

        data.enable_feature(SegmentFeature::RefFrame, 1);
        assert_eq!(data.reference_frame(), Some(MvRefType::Last));
    }

    #[test]
    fn test_segmentation_new() {
        let seg = Segmentation::new();
        assert!(!seg.enabled);
        assert!(!seg.is_active());
    }

    #[test]
    fn test_segmentation_qindex() {
        let mut seg = Segmentation::new();
        seg.enabled = true;
        seg.abs_delta = false;
        seg.set_feature(1, SegmentFeature::AltQ, -20);

        assert_eq!(seg.get_qindex(0, 100), 100); // No feature
        assert_eq!(seg.get_qindex(1, 100), 80); // 100 + (-20)
    }

    #[test]
    fn test_segmentation_qindex_absolute() {
        let mut seg = Segmentation::new();
        seg.enabled = true;
        seg.abs_delta = true;
        seg.set_feature(2, SegmentFeature::AltQ, 50);

        assert_eq!(seg.get_qindex(2, 100), 50); // Absolute value
    }

    #[test]
    fn test_segmentation_lf_level() {
        let mut seg = Segmentation::new();
        seg.enabled = true;
        seg.set_feature(3, SegmentFeature::AltLf, 10);

        assert_eq!(seg.get_lf_level(3, 30), 40); // 30 + 10
    }

    #[test]
    fn test_segmentation_reset() {
        let mut seg = Segmentation::new();
        seg.enabled = true;
        seg.set_feature(0, SegmentFeature::Skip, 0);

        seg.reset();

        assert!(!seg.enabled);
        assert!(!seg.segment(0).is_skip_forced());
    }

    #[test]
    fn test_segmentation_any_segment_has_feature() {
        let mut seg = Segmentation::new();
        assert!(!seg.has_skip_segment());

        seg.set_feature(5, SegmentFeature::Skip, 0);
        assert!(seg.has_skip_segment());
    }

    #[test]
    fn test_segment_map() {
        let mut map = SegmentMap::new(16, 16);
        assert_eq!(map.get(0, 0), 0);

        map.set(5, 5, 3);
        assert_eq!(map.get(5, 5), 3);
    }

    #[test]
    fn test_segment_map_fill_block() {
        let mut map = SegmentMap::new(16, 16);
        map.fill_block(2, 2, 4, 4, 5);

        assert_eq!(map.get(2, 2), 5);
        assert_eq!(map.get(5, 5), 5);
        assert_eq!(map.get(1, 1), 0);
        assert_eq!(map.get(6, 6), 0);
    }

    #[test]
    fn test_segment_map_clear() {
        let mut map = SegmentMap::new(16, 16);
        map.set(5, 5, 7);
        map.clear();
        assert_eq!(map.get(5, 5), 0);
    }

    #[test]
    fn test_segment_map_bounds() {
        let map = SegmentMap::new(16, 16);
        // Out of bounds should return 0
        assert_eq!(map.get(100, 100), 0);
    }
}
