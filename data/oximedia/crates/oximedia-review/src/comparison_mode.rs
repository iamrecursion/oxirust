//! High-level A/B comparison mode for media review.
//!
//! This module builds on the low-level [`crate::compare`] primitives to provide
//! a convenient [`ComparisonSession`] that manages a pair of media versions,
//! exposes A/B split and difference overlay controls, and produces composite
//! RGBA frames ready for display.
//!
//! # Example
//!
//! ```
//! use oximedia_review::comparison_mode::{ComparisonSession, AbSplitMode, DifferenceMode};
//! use oximedia_review::compare::CompareVersion;
//!
//! let va = CompareVersion::new("v1", "Version 1", 8, 8);
//! let vb = CompareVersion::new("v2", "Version 2", 8, 8);
//!
//! let mut session = ComparisonSession::new(va, vb);
//! session.set_ab_split(AbSplitMode::SideBySide);
//! let _frame = session.render_ab();
//! ```

#![allow(dead_code)]

use crate::compare::{
    apply_compare_filter, CompareFilter, CompareLayout, CompareVersion, DiffStats, WipeAngle,
};

// ── A/B split mode ────────────────────────────────────────────────────────────

/// How to display two versions side-by-side in an A/B comparison.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AbSplitMode {
    /// Version A on the left, B on the right (50/50 split).
    SideBySide,
    /// Version A on top, B on bottom.
    TopBottom,
    /// Horizontal wipe: everything left of `split_pos` shows A, right shows B.
    /// `split_pos` is in the range 0.0–1.0.
    HorizontalWipe {
        /// Wipe position (0.0 = all B, 1.0 = all A).
        split_pos: f32,
    },
    /// Vertical wipe: everything above `split_pos` shows A, below shows B.
    VerticalWipe {
        /// Wipe position (0.0 = all B, 1.0 = all A).
        split_pos: f32,
    },
    /// Interactive wipe with a draggable cursor.
    InteractiveSplit {
        /// Current cursor x position (0.0–1.0).
        cursor_x: f32,
    },
    /// Alpha-blended overlay (onion-skin).
    Overlay {
        /// Alpha for version B (0.0 = all A, 1.0 = all B).
        alpha: f32,
    },
}

impl AbSplitMode {
    /// Convert to the lower-level [`CompareLayout`].
    #[must_use]
    pub fn to_compare_layout(self) -> CompareLayout {
        match self {
            Self::SideBySide => CompareLayout::SideBySide,
            Self::TopBottom => CompareLayout::TopBottom,
            Self::HorizontalWipe { split_pos } => CompareLayout::Wipe {
                position: split_pos.clamp(0.0, 1.0),
                angle: WipeAngle::Horizontal,
            },
            Self::VerticalWipe { split_pos } => CompareLayout::Wipe {
                position: split_pos.clamp(0.0, 1.0),
                angle: WipeAngle::Vertical,
            },
            Self::InteractiveSplit { cursor_x } => CompareLayout::InteractiveSplit {
                x_position: cursor_x.clamp(0.0, 1.0),
            },
            Self::Overlay { alpha } => CompareLayout::Overlay {
                alpha: alpha.clamp(0.0, 1.0),
            },
        }
    }
}

// ── Difference overlay mode ────────────────────────────────────────────────────

/// Controls how the pixel-difference overlay is rendered.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DifferenceMode {
    /// Absolute |A - B| per channel.
    Absolute,
    /// Amplify differences by a gain factor.
    Amplified {
        /// Multiplier applied to each channel difference (>1 makes subtle diffs visible).
        gain: f32,
    },
    /// Threshold mask: pixels above the threshold shown as white, rest black.
    Threshold {
        /// Per-channel threshold value (0–255).
        threshold: u8,
    },
    /// Heat map: cold blue (no change) → hot red (large change).
    Heatmap,
    /// Isolate a single colour channel.
    ChannelIsolate {
        /// Channel index: 0=R, 1=G, 2=B.
        channel: u8,
    },
}

impl DifferenceMode {
    /// Convert to the lower-level [`CompareFilter`].
    #[must_use]
    pub fn to_compare_filter(self) -> CompareFilter {
        match self {
            Self::Absolute => CompareFilter::None, // caller applies |A-B| separately
            Self::Amplified { gain } => CompareFilter::DifferenceAmplify { gain },
            Self::Threshold { threshold } => CompareFilter::Threshold { threshold },
            Self::Heatmap => CompareFilter::Heatmap,
            Self::ChannelIsolate { channel } => CompareFilter::ChannelIsolate { channel },
        }
    }
}

// ── ComparisonSession ─────────────────────────────────────────────────────────

/// A stateful A/B comparison session managing two [`CompareVersion`]s.
///
/// Call [`Self::render_ab`] to get the A/B composite and [`Self::render_diff`]
/// to get the difference overlay.
#[derive(Debug)]
pub struct ComparisonSession {
    /// Version A (reference).
    pub version_a: CompareVersion,
    /// Version B (compare).
    pub version_b: CompareVersion,
    /// Current A/B split mode.
    pub ab_mode: AbSplitMode,
    /// Current difference overlay mode.
    pub diff_mode: DifferenceMode,
    /// Cached pixel-difference statistics (lazily computed on first `diff_stats()` call).
    cached_diff: Option<DiffStats>,
}

impl ComparisonSession {
    /// Create a new session with `SideBySide` A/B mode and `Absolute` difference mode.
    #[must_use]
    pub fn new(version_a: CompareVersion, version_b: CompareVersion) -> Self {
        Self {
            version_a,
            version_b,
            ab_mode: AbSplitMode::SideBySide,
            diff_mode: DifferenceMode::Absolute,
            cached_diff: None,
        }
    }

    /// Set the A/B split mode (clears cached diff stats).
    pub fn set_ab_split(&mut self, mode: AbSplitMode) {
        self.ab_mode = mode;
        self.cached_diff = None;
    }

    /// Set the difference overlay mode.
    pub fn set_diff_mode(&mut self, mode: DifferenceMode) {
        self.diff_mode = mode;
        self.cached_diff = None;
    }

    /// Output dimensions: width is max(a.width, b.width), height is max(a.height, b.height).
    #[must_use]
    pub fn output_size(&self) -> (u32, u32) {
        (
            self.version_a.width.max(self.version_b.width),
            self.version_a.height.max(self.version_b.height),
        )
    }

    /// Render the A/B composite according to the current split mode.
    ///
    /// Returns an RGBA byte buffer of size `output_width * output_height * 4`.
    #[must_use]
    pub fn render_ab(&self) -> Vec<u8> {
        use crate::compare::MediaComparator;
        let mut cmp = MediaComparator::new().with_layout(self.ab_mode.to_compare_layout());
        cmp.add_version(self.version_a.clone());
        cmp.add_version(self.version_b.clone());
        cmp.compare(&self.version_a.id, &self.version_b.id)
            .map(|r| r.output_data)
            .unwrap_or_default()
    }

    /// Render the difference overlay according to the current difference mode.
    ///
    /// Returns an RGBA byte buffer of size `output_width * output_height * 4`.
    #[must_use]
    pub fn render_diff(&self) -> Vec<u8> {
        let (w, h) = self.output_size();
        match self.diff_mode {
            DifferenceMode::Absolute => {
                // Produce |A-B| by using the Difference layout
                use crate::compare::MediaComparator;
                let mut cmp = MediaComparator::new().with_layout(CompareLayout::Difference);
                cmp.add_version(self.version_a.clone());
                cmp.add_version(self.version_b.clone());
                cmp.compare(&self.version_a.id, &self.version_b.id)
                    .map(|r| r.output_data)
                    .unwrap_or_default()
            }
            other => apply_compare_filter(
                &self.version_a,
                &self.version_b,
                other.to_compare_filter(),
                w,
                h,
            ),
        }
    }

    /// Compute and cache pixel-difference statistics between A and B.
    #[must_use]
    pub fn diff_stats(&mut self) -> &DiffStats {
        if self.cached_diff.is_none() {
            self.cached_diff = Some(DiffStats::compute(
                &self.version_a.frame_data,
                &self.version_b.frame_data,
            ));
        }
        // SAFETY: set above
        self.cached_diff.as_ref().expect("just computed")
    }

    /// Return cached diff stats without recomputing (returns `None` if not yet computed).
    #[must_use]
    pub fn cached_diff_stats(&self) -> Option<&DiffStats> {
        self.cached_diff.as_ref()
    }

    /// Swap A and B, clearing cached stats.
    pub fn swap(&mut self) {
        std::mem::swap(&mut self.version_a, &mut self.version_b);
        self.cached_diff = None;
    }

    /// Returns `true` when A and B have identical frame data (byte-level).
    #[must_use]
    pub fn are_identical(&self) -> bool {
        self.version_a.frame_data == self.version_b.frame_data
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_rgba(w: u32, h: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        let n = (w * h * 4) as usize;
        let mut v = Vec::with_capacity(n);
        for _ in 0..(w * h) {
            v.extend_from_slice(&[r, g, b, 255]);
        }
        v
    }

    fn make_version(id: &str, w: u32, h: u32, r: u8, g: u8, b: u8) -> CompareVersion {
        CompareVersion::new(id, id, w, h).with_frame_data(solid_rgba(w, h, r, g, b))
    }

    fn make_session() -> ComparisonSession {
        ComparisonSession::new(
            make_version("a", 8, 8, 255, 0, 0),
            make_version("b", 8, 8, 0, 255, 0),
        )
    }

    // 1 — output_size returns max of each dimension
    #[test]
    fn test_output_size() {
        let s = make_session();
        assert_eq!(s.output_size(), (8, 8));
    }

    // 2 — render_ab produces correct buffer length
    #[test]
    fn test_render_ab_length() {
        let s = make_session();
        let out = s.render_ab();
        assert_eq!(out.len(), (8 * 8 * 4) as usize);
    }

    // 3 — render_diff produces correct buffer length
    #[test]
    fn test_render_diff_length() {
        let s = make_session();
        let out = s.render_diff();
        assert_eq!(out.len(), (8 * 8 * 4) as usize);
    }

    // 4 — are_identical true for same data
    #[test]
    fn test_are_identical_same() {
        let va = make_version("a", 4, 4, 10, 20, 30);
        let vb = make_version("b", 4, 4, 10, 20, 30);
        let s = ComparisonSession::new(va, vb);
        assert!(s.are_identical());
    }

    // 5 — are_identical false for different data
    #[test]
    fn test_are_identical_different() {
        let s = make_session();
        assert!(!s.are_identical());
    }

    // 6 — diff_stats caching
    #[test]
    fn test_diff_stats_caching() {
        let mut s = make_session();
        assert!(s.cached_diff_stats().is_none());
        let stats = s.diff_stats().clone();
        assert!(s.cached_diff_stats().is_some());
        assert!(!stats.identical);
    }

    // 7 — swap exchanges A and B
    #[test]
    fn test_swap() {
        let mut s = make_session();
        let a_id = s.version_a.id.clone();
        let b_id = s.version_b.id.clone();
        s.swap();
        assert_eq!(s.version_a.id, b_id);
        assert_eq!(s.version_b.id, a_id);
    }

    // 8 — swap clears cached stats
    #[test]
    fn test_swap_clears_cache() {
        let mut s = make_session();
        let _ = s.diff_stats();
        assert!(s.cached_diff_stats().is_some());
        s.swap();
        assert!(s.cached_diff_stats().is_none());
    }

    // 9 — AbSplitMode::to_compare_layout for side-by-side
    #[test]
    fn test_ab_split_mode_side_by_side() {
        assert_eq!(
            AbSplitMode::SideBySide.to_compare_layout(),
            CompareLayout::SideBySide
        );
    }

    // 10 — AbSplitMode::to_compare_layout for horizontal wipe
    #[test]
    fn test_ab_split_mode_horizontal_wipe() {
        let layout = AbSplitMode::HorizontalWipe { split_pos: 0.3 }.to_compare_layout();
        assert!(matches!(
            layout,
            CompareLayout::Wipe {
                angle: WipeAngle::Horizontal,
                ..
            }
        ));
    }

    // 11 — AbSplitMode::to_compare_layout for overlay
    #[test]
    fn test_ab_split_mode_overlay() {
        let layout = AbSplitMode::Overlay { alpha: 0.5 }.to_compare_layout();
        assert!(matches!(layout, CompareLayout::Overlay { .. }));
    }

    // 12 — DifferenceMode::to_compare_filter for heatmap
    #[test]
    fn test_diff_mode_heatmap() {
        let filter = DifferenceMode::Heatmap.to_compare_filter();
        assert_eq!(filter, CompareFilter::Heatmap);
    }

    // 13 — DifferenceMode::to_compare_filter for threshold
    #[test]
    fn test_diff_mode_threshold() {
        let filter = DifferenceMode::Threshold { threshold: 20 }.to_compare_filter();
        assert!(matches!(filter, CompareFilter::Threshold { threshold: 20 }));
    }

    // 14 — set_ab_split clears cache
    #[test]
    fn test_set_ab_split_clears_cache() {
        let mut s = make_session();
        let _ = s.diff_stats();
        s.set_ab_split(AbSplitMode::TopBottom);
        assert!(s.cached_diff_stats().is_none());
    }

    // 15 — render_diff with Amplified mode
    #[test]
    fn test_render_diff_amplified() {
        let mut s = make_session();
        s.set_diff_mode(DifferenceMode::Amplified { gain: 2.0 });
        let out = s.render_diff();
        assert_eq!(out.len(), (8 * 8 * 4) as usize);
    }

    // 16 — render_diff with Heatmap mode
    #[test]
    fn test_render_diff_heatmap() {
        let mut s = make_session();
        s.set_diff_mode(DifferenceMode::Heatmap);
        let out = s.render_diff();
        assert!(!out.is_empty());
    }

    // 17 — render_ab with interactive split
    #[test]
    fn test_render_ab_interactive_split() {
        let mut s = make_session();
        s.set_ab_split(AbSplitMode::InteractiveSplit { cursor_x: 0.6 });
        let out = s.render_ab();
        assert_eq!(out.len(), (8 * 8 * 4) as usize);
    }
}
