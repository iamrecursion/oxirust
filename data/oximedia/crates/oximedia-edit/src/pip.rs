//! Picture-in-Picture (PiP) layout with position/scale keyframes.
//!
//! Composites a secondary video source over a primary frame. The secondary
//! clip's position and scale can be animated via keyframes for dynamic
//! transitions (e.g., swipe-in from off-screen, shrink to corner).

#![allow(dead_code)]

use crate::clip::ClipId;

// ─────────────────────────────────────────────────────────────────────────────
// Keyframe types
// ─────────────────────────────────────────────────────────────────────────────

/// A single keyframe that defines `(position_x, position_y, scale_x, scale_y,
/// opacity)` at a given timeline position.
///
/// All values are in *normalised* coordinates where the primary frame occupies
/// the unit square `[0.0, 1.0] × [0.0, 1.0]`:
/// - `pos_x / pos_y` — top-left anchor of the PiP window.
/// - `scale_x / scale_y` — fraction of the primary frame's width/height.
/// - `opacity` — 0.0 (invisible) … 1.0 (fully opaque).
#[derive(Debug, Clone, PartialEq)]
pub struct PipKeyframe {
    /// Timeline position (timebase units).
    pub time: i64,
    /// Horizontal position (normalised, 0.0 = left edge).
    pub pos_x: f64,
    /// Vertical position (normalised, 0.0 = top edge).
    pub pos_y: f64,
    /// Horizontal scale (fraction of primary width).
    pub scale_x: f64,
    /// Vertical scale (fraction of primary height).
    pub scale_y: f64,
    /// Opacity (0.0–1.0).
    pub opacity: f64,
}

impl PipKeyframe {
    /// Create a new keyframe with uniform scale.
    #[must_use]
    pub fn new(time: i64, pos_x: f64, pos_y: f64, scale: f64, opacity: f64) -> Self {
        Self {
            time,
            pos_x,
            pos_y,
            scale_x: scale,
            scale_y: scale,
            opacity: opacity.clamp(0.0, 1.0),
        }
    }

    /// Create a keyframe with independent horizontal and vertical scale.
    #[must_use]
    pub fn with_scale_xy(
        time: i64,
        pos_x: f64,
        pos_y: f64,
        scale_x: f64,
        scale_y: f64,
        opacity: f64,
    ) -> Self {
        Self {
            time,
            pos_x,
            pos_y,
            scale_x,
            scale_y,
            opacity: opacity.clamp(0.0, 1.0),
        }
    }
}

/// Interpolated state at a specific timeline position.
#[derive(Debug, Clone, PartialEq)]
pub struct PipState {
    /// Horizontal position (normalised).
    pub pos_x: f64,
    /// Vertical position (normalised).
    pub pos_y: f64,
    /// Horizontal scale.
    pub scale_x: f64,
    /// Vertical scale.
    pub scale_y: f64,
    /// Opacity.
    pub opacity: f64,
}

impl PipState {
    /// Compute pixel-space bounding box given primary frame dimensions.
    ///
    /// Returns `(left, top, width, height)` in pixels.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn pixel_rect(&self, frame_width: u32, frame_height: u32) -> (i32, i32, u32, u32) {
        let left = (self.pos_x * frame_width as f64).round() as i32;
        let top = (self.pos_y * frame_height as f64).round() as i32;
        let w = ((self.scale_x * frame_width as f64).round() as u32).max(1);
        let h = ((self.scale_y * frame_height as f64).round() as u32).max(1);
        (left, top, w, h)
    }

    /// Returns `true` if the PiP is effectively invisible (opacity < threshold).
    #[must_use]
    pub fn is_invisible(&self) -> bool {
        self.opacity < 1e-4
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PipLayout
// ─────────────────────────────────────────────────────────────────────────────

/// Identifier for a PiP layout.
pub type PipId = u64;

/// Preset corner positions (normalised) for common PiP placements.
pub struct PipPreset;

impl PipPreset {
    /// Bottom-right corner at 25 % size with 2 % margin.
    #[must_use]
    pub fn bottom_right() -> PipKeyframe {
        PipKeyframe::new(0, 0.73, 0.73, 0.25, 1.0)
    }

    /// Bottom-left corner at 25 % size with 2 % margin.
    #[must_use]
    pub fn bottom_left() -> PipKeyframe {
        PipKeyframe::new(0, 0.02, 0.73, 0.25, 1.0)
    }

    /// Top-right corner at 25 % size with 2 % margin.
    #[must_use]
    pub fn top_right() -> PipKeyframe {
        PipKeyframe::new(0, 0.73, 0.02, 0.25, 1.0)
    }

    /// Top-left corner at 25 % size with 2 % margin.
    #[must_use]
    pub fn top_left() -> PipKeyframe {
        PipKeyframe::new(0, 0.02, 0.02, 0.25, 1.0)
    }

    /// Full-screen (1:1) replacement.
    #[must_use]
    pub fn fullscreen() -> PipKeyframe {
        PipKeyframe::new(0, 0.0, 0.0, 1.0, 1.0)
    }
}

/// A picture-in-picture composition that places a secondary video clip
/// (`pip_clip`) over a primary clip (`base_clip`).
///
/// The secondary clip's geometry is driven by a set of keyframes.  If there
/// are no keyframes a static default state is used.
#[derive(Debug, Clone)]
pub struct PipLayout {
    /// Unique identifier.
    pub id: PipId,
    /// The primary (background) clip.
    pub base_clip: ClipId,
    /// The secondary (overlay) clip.
    pub pip_clip: ClipId,
    /// Timeline start (inclusive).
    pub start: i64,
    /// Timeline end (exclusive).
    pub end: i64,
    /// Keyframe list, sorted by `time`.
    pub keyframes: Vec<PipKeyframe>,
    /// Default / static state used when no keyframes are defined.
    pub default_state: PipState,
    /// Whether to maintain the PiP clip's aspect ratio when scaling.
    pub maintain_aspect: bool,
}

impl PipLayout {
    /// Create a new PiP layout with the bottom-right preset.
    #[must_use]
    pub fn new(id: PipId, base_clip: ClipId, pip_clip: ClipId, start: i64, end: i64) -> Self {
        let kf = PipPreset::bottom_right();
        Self {
            id,
            base_clip,
            pip_clip,
            start,
            end,
            keyframes: Vec::new(),
            default_state: PipState {
                pos_x: kf.pos_x,
                pos_y: kf.pos_y,
                scale_x: kf.scale_x,
                scale_y: kf.scale_y,
                opacity: kf.opacity,
            },
            maintain_aspect: true,
        }
    }

    /// Add a keyframe (sorted insertion).
    pub fn add_keyframe(&mut self, kf: PipKeyframe) {
        self.keyframes.push(kf);
        self.keyframes.sort_by_key(|k| k.time);
    }

    /// Remove all keyframes at the given `time`.
    pub fn remove_keyframes_at(&mut self, time: i64) {
        self.keyframes.retain(|k| k.time != time);
    }

    /// Evaluate the PiP state at `time` by linearly interpolating keyframes.
    ///
    /// If there are no keyframes the [`default_state`](Self::default_state) is
    /// returned.  If `time` is before the first keyframe the first keyframe's
    /// state is returned; if after the last, the last is returned.
    #[must_use]
    pub fn state_at(&self, time: i64) -> PipState {
        if self.keyframes.is_empty() {
            return self.default_state.clone();
        }
        let kf = &self.keyframes;
        if time <= kf[0].time {
            return kf_to_state(&kf[0]);
        }
        let last = &kf[kf.len() - 1];
        if time >= last.time {
            return kf_to_state(last);
        }
        // Linear interpolation.
        let idx = kf.partition_point(|k| k.time <= time) - 1;
        let k0 = &kf[idx];
        let k1 = &kf[idx + 1];
        let span = (k1.time - k0.time) as f64;
        let alpha = if span > 0.0 {
            (time - k0.time) as f64 / span
        } else {
            0.0
        };
        PipState {
            pos_x: lerp(k0.pos_x, k1.pos_x, alpha),
            pos_y: lerp(k0.pos_y, k1.pos_y, alpha),
            scale_x: lerp(k0.scale_x, k1.scale_x, alpha),
            scale_y: lerp(k0.scale_y, k1.scale_y, alpha),
            opacity: lerp(k0.opacity, k1.opacity, alpha).clamp(0.0, 1.0),
        }
    }

    /// Returns `true` if the layout is active at `time`.
    #[must_use]
    pub fn is_active_at(&self, time: i64) -> bool {
        time >= self.start && time < self.end
    }

    /// Number of keyframes.
    #[must_use]
    pub fn keyframe_count(&self) -> usize {
        self.keyframes.len()
    }
}

/// Linear interpolation helper.
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}

/// Convert a `PipKeyframe` reference to a `PipState`.
fn kf_to_state(k: &PipKeyframe) -> PipState {
    PipState {
        pos_x: k.pos_x,
        pos_y: k.pos_y,
        scale_x: k.scale_x,
        scale_y: k.scale_y,
        opacity: k.opacity,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PipManager
// ─────────────────────────────────────────────────────────────────────────────

/// Manages PiP layouts for a timeline.
#[derive(Debug, Default)]
pub struct PipManager {
    layouts: Vec<PipLayout>,
    next_id: PipId,
}

impl PipManager {
    /// Create an empty manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            layouts: Vec::new(),
            next_id: 1,
        }
    }

    /// Add a PiP layout and return its assigned ID.
    pub fn add(&mut self, mut layout: PipLayout) -> PipId {
        let id = self.next_id;
        self.next_id += 1;
        layout.id = id;
        self.layouts.push(layout);
        id
    }

    /// Remove a layout by ID.
    pub fn remove(&mut self, id: PipId) -> Option<PipLayout> {
        if let Some(pos) = self.layouts.iter().position(|l| l.id == id) {
            Some(self.layouts.remove(pos))
        } else {
            None
        }
    }

    /// Get a reference.
    #[must_use]
    pub fn get(&self, id: PipId) -> Option<&PipLayout> {
        self.layouts.iter().find(|l| l.id == id)
    }

    /// Get a mutable reference.
    pub fn get_mut(&mut self, id: PipId) -> Option<&mut PipLayout> {
        self.layouts.iter_mut().find(|l| l.id == id)
    }

    /// Return all layouts active at `time`.
    #[must_use]
    pub fn active_at(&self, time: i64) -> Vec<&PipLayout> {
        self.layouts
            .iter()
            .filter(|l| l.is_active_at(time))
            .collect()
    }

    /// Total layout count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.layouts.len()
    }

    /// Returns `true` if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.layouts.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_layout(start: i64, end: i64) -> PipLayout {
        PipLayout::new(0, 1, 2, start, end)
    }

    #[test]
    fn test_pip_layout_active_at() {
        let l = make_layout(1000, 5000);
        assert!(l.is_active_at(1000));
        assert!(l.is_active_at(4999));
        assert!(!l.is_active_at(5000));
        assert!(!l.is_active_at(999));
    }

    #[test]
    fn test_pip_state_pixel_rect() {
        let state = PipState {
            pos_x: 0.5,
            pos_y: 0.5,
            scale_x: 0.25,
            scale_y: 0.25,
            opacity: 1.0,
        };
        let (left, top, w, h) = state.pixel_rect(1920, 1080);
        assert_eq!(left, 960);
        assert_eq!(top, 540);
        assert_eq!(w, 480);
        assert_eq!(h, 270);
    }

    #[test]
    fn test_pip_layout_no_keyframes_returns_default() {
        let l = make_layout(0, 10000);
        let state = l.state_at(5000);
        // Default is bottom_right preset: pos=(0.73, 0.73), scale=0.25, opacity=1.0
        assert!((state.pos_x - 0.73).abs() < 1e-9);
        assert!((state.scale_x - 0.25).abs() < 1e-9);
    }

    #[test]
    fn test_pip_layout_keyframe_interpolation() {
        let mut l = make_layout(0, 10000);
        l.add_keyframe(PipKeyframe::new(0, 0.0, 0.0, 1.0, 0.0));
        l.add_keyframe(PipKeyframe::new(1000, 0.5, 0.5, 0.5, 1.0));
        let state = l.state_at(500);
        assert!((state.pos_x - 0.25).abs() < 1e-9);
        assert!((state.opacity - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_pip_layout_keyframe_before_first() {
        let mut l = make_layout(0, 10000);
        l.add_keyframe(PipKeyframe::new(1000, 0.1, 0.2, 0.5, 1.0));
        let state = l.state_at(0);
        assert!((state.pos_x - 0.1).abs() < 1e-9);
    }

    #[test]
    fn test_pip_layout_keyframe_after_last() {
        let mut l = make_layout(0, 10000);
        l.add_keyframe(PipKeyframe::new(0, 0.1, 0.2, 0.5, 1.0));
        let state = l.state_at(99999);
        assert!((state.pos_x - 0.1).abs() < 1e-9);
    }

    #[test]
    fn test_pip_layout_remove_keyframes_at() {
        let mut l = make_layout(0, 10000);
        l.add_keyframe(PipKeyframe::new(0, 0.0, 0.0, 1.0, 1.0));
        l.add_keyframe(PipKeyframe::new(500, 0.5, 0.5, 0.5, 0.5));
        assert_eq!(l.keyframe_count(), 2);
        l.remove_keyframes_at(0);
        assert_eq!(l.keyframe_count(), 1);
    }

    #[test]
    fn test_pip_manager_add_remove() {
        let mut mgr = PipManager::new();
        let layout = make_layout(0, 1000);
        let id = mgr.add(layout);
        assert_eq!(mgr.len(), 1);
        assert!(mgr.get(id).is_some());
        assert!(mgr.remove(id).is_some());
        assert!(mgr.is_empty());
    }

    #[test]
    fn test_pip_manager_active_at() {
        let mut mgr = PipManager::new();
        mgr.add(make_layout(0, 1000));
        mgr.add(make_layout(500, 2000));
        assert_eq!(mgr.active_at(600).len(), 2);
        assert_eq!(mgr.active_at(100).len(), 1);
        assert_eq!(mgr.active_at(2500).len(), 0);
    }

    #[test]
    fn test_pip_presets_normalised_range() {
        for kf in [
            PipPreset::bottom_right(),
            PipPreset::bottom_left(),
            PipPreset::top_right(),
            PipPreset::top_left(),
            PipPreset::fullscreen(),
        ] {
            assert!(kf.pos_x >= 0.0 && kf.pos_x <= 1.0);
            assert!(kf.pos_y >= 0.0 && kf.pos_y <= 1.0);
            assert!(kf.scale_x > 0.0 && kf.scale_x <= 1.0);
            assert!((kf.opacity - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn test_pip_state_invisible() {
        let invisible = PipState {
            opacity: 0.0,
            pos_x: 0.0,
            pos_y: 0.0,
            scale_x: 1.0,
            scale_y: 1.0,
        };
        assert!(invisible.is_invisible());
        let visible = PipState {
            opacity: 0.5,
            pos_x: 0.0,
            pos_y: 0.0,
            scale_x: 1.0,
            scale_y: 1.0,
        };
        assert!(!visible.is_invisible());
    }
}
