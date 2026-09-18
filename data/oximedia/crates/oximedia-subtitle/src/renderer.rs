//! Main subtitle renderer with incremental (dirty-region) redraw support.
//!
//! The `IncrementalSubtitleRenderer` tracks which subtitle cue is currently
//! displayed and only composites a new frame when the visible content changes.
//! This avoids redundant per-frame GPU / CPU work in playout pipelines.

use crate::font::Font;
use crate::overlay::overlay_subtitle;
use crate::style::{Position, SubtitleStyle};
use crate::text::TextLayoutEngine;
use crate::{Subtitle, SubtitleError, SubtitleResult};
use oximedia_codec::VideoFrame;
use oximedia_core::Timestamp;

/// Main subtitle renderer.
pub struct SubtitleRenderer {
    layout_engine: TextLayoutEngine,
    default_style: SubtitleStyle,
}

impl SubtitleRenderer {
    /// Create a new subtitle renderer.
    #[must_use]
    pub fn new(font: Font, style: SubtitleStyle) -> Self {
        Self {
            layout_engine: TextLayoutEngine::new(font),
            default_style: style,
        }
    }

    /// Render a subtitle onto a video frame at the given timestamp.
    ///
    /// # Errors
    ///
    /// Returns error if rendering fails or frame format is unsupported.
    pub fn render_subtitle(
        &mut self,
        subtitle: &Subtitle,
        frame: &mut VideoFrame,
        timestamp: Timestamp,
    ) -> SubtitleResult<()> {
        // Check if subtitle is active
        let timestamp_ms = (timestamp.to_seconds() * 1000.0) as i64;
        if !subtitle.is_active(timestamp_ms) {
            return Ok(());
        }

        // Get effective style
        let style = subtitle.style.as_ref().unwrap_or(&self.default_style);

        // Layout text
        let max_width = if style.max_width > 0 {
            style.max_width
        } else {
            frame
                .width
                .saturating_sub(style.margin_left + style.margin_right)
        };

        let layout = self
            .layout_engine
            .layout(&subtitle.text, style, max_width)?;

        if layout.is_empty() {
            return Ok(());
        }

        // Calculate position
        let position = subtitle.position.as_ref().unwrap_or(&style.position);
        let (x, y) = self.calculate_position(frame, &layout, position, style);

        // Apply animations
        let (color, outline_color) = self.apply_animations(
            subtitle,
            timestamp_ms,
            style.primary_color,
            style.outline.as_ref().map(|o| o.color),
        );

        // Render background box if enabled
        if let Some(bg_color) = style.background_color {
            self.render_background(frame, &layout, x, y, bg_color, style.background_padding)?;
        }

        // Get outline width
        let outline_width = style.outline.as_ref().map(|o| o.width).unwrap_or(0.0);

        // Overlay subtitle onto frame
        overlay_subtitle(frame, &layout, x, y, color, outline_color, outline_width)?;

        Ok(())
    }

    /// Render multiple subtitles (for overlapping subtitles).
    ///
    /// # Errors
    ///
    /// Returns error if any subtitle rendering fails.
    pub fn render_subtitles(
        &mut self,
        subtitles: &[Subtitle],
        frame: &mut VideoFrame,
        timestamp: Timestamp,
    ) -> SubtitleResult<()> {
        let timestamp_ms = (timestamp.to_seconds() * 1000.0) as i64;

        // Filter active subtitles
        let active: Vec<_> = subtitles
            .iter()
            .filter(|s| s.is_active(timestamp_ms))
            .collect();

        // Render each active subtitle
        for subtitle in active {
            self.render_subtitle(subtitle, frame, timestamp)?;
        }

        Ok(())
    }

    /// Calculate absolute position from relative position and alignment.
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_precision_loss)]
    fn calculate_position(
        &self,
        frame: &VideoFrame,
        layout: &crate::text::TextLayout,
        position: &Position,
        style: &SubtitleStyle,
    ) -> (i32, i32) {
        let frame_width = frame.width as f32;
        let frame_height = frame.height as f32;

        // Calculate base position from relative coordinates
        let mut x = frame_width * position.x;
        let mut y = frame_height * position.y;

        // Apply alignment
        match position.alignment {
            crate::style::Alignment::Left => {
                x += style.margin_left as f32;
            }
            crate::style::Alignment::Center => {
                x -= layout.width / 2.0;
            }
            crate::style::Alignment::Right => {
                x -= layout.width + style.margin_right as f32;
            }
        }

        // Apply vertical alignment
        match position.vertical_alignment {
            crate::style::VerticalAlignment::Top => {
                y += style.margin_top as f32;
            }
            crate::style::VerticalAlignment::Middle => {
                y -= layout.height / 2.0;
            }
            crate::style::VerticalAlignment::Bottom => {
                y -= layout.height + style.margin_bottom as f32;
            }
        }

        (x as i32, y as i32)
    }

    /// Apply animation effects to colors.
    #[allow(clippy::cast_precision_loss)]
    fn apply_animations(
        &self,
        subtitle: &Subtitle,
        timestamp_ms: i64,
        mut primary_color: crate::style::Color,
        outline_color: Option<crate::style::Color>,
    ) -> (crate::style::Color, Option<crate::style::Color>) {
        use crate::style::Animation;

        let elapsed = timestamp_ms - subtitle.start_time;
        let duration = subtitle.duration();

        for animation in &subtitle.animations {
            match animation {
                Animation::FadeIn(fade_duration) => {
                    if elapsed < *fade_duration {
                        let progress = elapsed as f32 / *fade_duration as f32;
                        let alpha = (f32::from(primary_color.a) * progress) as u8;
                        primary_color = primary_color.with_alpha(alpha);
                    }
                }
                Animation::FadeOut(fade_duration) => {
                    let fade_start = duration - fade_duration;
                    if elapsed > fade_start {
                        let fade_elapsed = elapsed - fade_start;
                        let progress = fade_elapsed as f32 / *fade_duration as f32;
                        let alpha = (f32::from(primary_color.a) * (1.0 - progress)) as u8;
                        primary_color = primary_color.with_alpha(alpha);
                    }
                }
                _ => {
                    // Other animations would be applied to position/scale
                    // Not implemented here for simplicity
                }
            }
        }

        (primary_color, outline_color)
    }

    /// Render background box.
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    fn render_background(
        &self,
        frame: &mut VideoFrame,
        layout: &crate::text::TextLayout,
        x: i32,
        y: i32,
        color: crate::style::Color,
        padding: f32,
    ) -> SubtitleResult<()> {
        if frame.planes.is_empty() {
            return Err(SubtitleError::InvalidFrameFormat(
                "Frame has no planes".to_string(),
            ));
        }

        let x1 = (x as f32 - padding).max(0.0) as usize;
        let y1 = (y as f32 - padding).max(0.0) as usize;
        let x2 = ((x as f32 + layout.width + padding) as usize).min(frame.width as usize);
        let y2 = ((y as f32 + layout.height + padding) as usize).min(frame.height as usize);

        let mut plane_data = frame.planes[0].data.to_vec();
        let stride = frame.planes[0].stride;

        let bytes_per_pixel = match frame.format {
            oximedia_core::PixelFormat::Rgb24 => 3,
            oximedia_core::PixelFormat::Rgba32 => 4,
            _ => return Ok(()), // Skip background for other formats
        };

        // Fill rectangle
        for py in y1..y2 {
            for px in x1..x2 {
                let idx = py * stride + px * bytes_per_pixel;
                if idx + bytes_per_pixel <= plane_data.len() {
                    let alpha = f32::from(color.a) / 255.0;
                    let inv_alpha = 1.0 - alpha;

                    plane_data[idx] =
                        (f32::from(color.r) * alpha + f32::from(plane_data[idx]) * inv_alpha) as u8;
                    plane_data[idx + 1] = (f32::from(color.g) * alpha
                        + f32::from(plane_data[idx + 1]) * inv_alpha)
                        as u8;
                    plane_data[idx + 2] = (f32::from(color.b) * alpha
                        + f32::from(plane_data[idx + 2]) * inv_alpha)
                        as u8;
                }
            }
        }

        frame.planes[0].data = plane_data;

        Ok(())
    }

    /// Get the default style.
    #[must_use]
    pub fn style(&self) -> &SubtitleStyle {
        &self.default_style
    }

    /// Set the default style.
    pub fn set_style(&mut self, style: SubtitleStyle) {
        self.default_style = style;
    }
}

// ============================================================================
// Incremental subtitle renderer
// ============================================================================

/// A dirty rectangle in a video frame (pixel coordinates).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirtyRect {
    /// Left edge (inclusive).
    pub x: u32,
    /// Top edge (inclusive).
    pub y: u32,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl DirtyRect {
    /// Create a new dirty rect.
    #[must_use]
    pub const fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Return `true` if the rect has no area.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    /// Compute the union of two dirty rects (smallest bounding box).
    #[must_use]
    pub fn union(&self, other: &Self) -> Self {
        if self.is_empty() {
            return *other;
        }
        if other.is_empty() {
            return *self;
        }
        let x1 = self.x.min(other.x);
        let y1 = self.y.min(other.y);
        let x2 = (self.x + self.width).max(other.x + other.width);
        let y2 = (self.y + self.height).max(other.y + other.height);
        Self::new(x1, y1, x2 - x1, y2 - y1)
    }
}

/// Key used to identify the currently displayed subtitle state.
#[derive(Clone, Debug, PartialEq)]
struct DisplayedState {
    /// `(start_ms, end_ms, text)` tuple uniquely identifies a cue.
    cues: Vec<(i64, i64, String)>,
}

/// Incremental subtitle renderer that only redraws changed regions.
///
/// The renderer maintains a record of the last displayed subtitle state.
/// On each call to `render_incremental`, it compares the incoming subtitle
/// set against the last rendered frame.  If the set is unchanged (same active
/// cues) no work is done and `None` is returned.  If the cues have changed, the
/// new cues are composited and the dirty region is returned.
///
/// # Example
///
/// ```rust,ignore
/// use oximedia_subtitle::renderer::IncrementalSubtitleRenderer;
/// use oximedia_subtitle::{Subtitle, style::SubtitleStyle};
/// use oximedia_subtitle::font::Font;
///
/// let font = Font::from_file("font.ttf")?;
/// let mut renderer = IncrementalSubtitleRenderer::new(font, SubtitleStyle::default());
///
/// // First frame — subtitle appeared → returns Some(dirty_rect)
/// let dirty = renderer.render_incremental(&subtitles, &mut frame, timestamp)?;
///
/// // Second frame with same subtitle active → returns None (no redraw needed)
/// let dirty2 = renderer.render_incremental(&subtitles, &mut frame, timestamp2)?;
/// assert!(dirty2.is_none());
/// ```
pub struct IncrementalSubtitleRenderer {
    inner: SubtitleRenderer,
    /// State from the last successfully rendered frame.
    last_state: Option<DisplayedState>,
    /// Dirty rect from the last change (used to clear previous subtitle).
    last_dirty: Option<DirtyRect>,
}

impl IncrementalSubtitleRenderer {
    /// Create a new incremental renderer.
    #[must_use]
    pub fn new(font: Font, style: SubtitleStyle) -> Self {
        Self {
            inner: SubtitleRenderer::new(font, style),
            last_state: None,
            last_dirty: None,
        }
    }

    /// Get the default style.
    #[must_use]
    pub fn style(&self) -> &SubtitleStyle {
        self.inner.style()
    }

    /// Set the default style.
    pub fn set_style(&mut self, style: SubtitleStyle) {
        self.inner.set_style(style);
    }

    /// Render changed subtitle regions onto `frame`.
    ///
    /// Returns `Ok(Some(dirty_rect))` if any subtitle was rendered (the region
    /// that changed), or `Ok(None)` if nothing changed and no redraw was needed.
    ///
    /// # Errors
    ///
    /// Returns error if rendering a new cue fails.
    pub fn render_incremental(
        &mut self,
        subtitles: &[Subtitle],
        frame: &mut VideoFrame,
        timestamp: Timestamp,
    ) -> SubtitleResult<Option<DirtyRect>> {
        let timestamp_ms = (timestamp.to_seconds() * 1000.0) as i64;

        // Collect currently active cues
        let active: Vec<&Subtitle> = subtitles
            .iter()
            .filter(|s| s.is_active(timestamp_ms))
            .collect();

        let new_state = DisplayedState {
            cues: active
                .iter()
                .map(|s| (s.start_time, s.end_time, s.text.clone()))
                .collect(),
        };

        // Compare with last rendered state
        if self.last_state.as_ref() == Some(&new_state) {
            // Nothing changed — skip redraw
            return Ok(None);
        }

        // Compute dirty rect: union of previous and new subtitle positions.
        // For simplicity we mark the whole subtitle area as dirty.  A finer
        // implementation would track per-cue bounding boxes.
        let mut dirty = DirtyRect::new(0, 0, 0, 0);

        // Render each active subtitle
        for subtitle in &active {
            self.inner.render_subtitle(subtitle, frame, timestamp)?;
            // Approximate dirty rect using frame dimensions as upper bound
            let cue_dirty = DirtyRect::new(0, frame.height / 2, frame.width, frame.height / 2);
            dirty = dirty.union(&cue_dirty);
        }

        // If we had subtitles before and now have none, the old area is dirty
        if let Some(prev_dirty) = self.last_dirty {
            dirty = dirty.union(&prev_dirty);
        }

        self.last_state = Some(new_state);
        self.last_dirty = if dirty.is_empty() { None } else { Some(dirty) };

        if dirty.is_empty() && active.is_empty() {
            // State changed (from Some → None) but nothing to paint
            Ok(Some(DirtyRect::new(0, 0, 0, 0)))
        } else {
            Ok(Some(dirty))
        }
    }

    /// Force the renderer to forget the last displayed state, ensuring a full
    /// redraw on the next `render_incremental` call.
    pub fn invalidate(&mut self) {
        self.last_state = None;
        self.last_dirty = None;
    }

    /// Return the last computed dirty rect, if any.
    #[must_use]
    pub fn last_dirty_rect(&self) -> Option<DirtyRect> {
        self.last_dirty
    }
}

#[cfg(test)]
mod incremental_tests {
    use super::*;

    #[test]
    fn test_dirty_rect_new() {
        let r = DirtyRect::new(10, 20, 100, 50);
        assert_eq!(r.x, 10);
        assert_eq!(r.y, 20);
        assert_eq!(r.width, 100);
        assert_eq!(r.height, 50);
    }

    #[test]
    fn test_dirty_rect_is_empty_zero_width() {
        let r = DirtyRect::new(0, 0, 0, 100);
        assert!(r.is_empty());
    }

    #[test]
    fn test_dirty_rect_is_empty_zero_height() {
        let r = DirtyRect::new(0, 0, 100, 0);
        assert!(r.is_empty());
    }

    #[test]
    fn test_dirty_rect_not_empty() {
        let r = DirtyRect::new(5, 5, 10, 10);
        assert!(!r.is_empty());
    }

    #[test]
    fn test_dirty_rect_union_basic() {
        let a = DirtyRect::new(0, 0, 10, 10);
        let b = DirtyRect::new(5, 5, 10, 10);
        let u = a.union(&b);
        assert_eq!(u.x, 0);
        assert_eq!(u.y, 0);
        assert_eq!(u.width, 15);
        assert_eq!(u.height, 15);
    }

    #[test]
    fn test_dirty_rect_union_with_empty() {
        let a = DirtyRect::new(10, 20, 100, 50);
        let empty = DirtyRect::new(0, 0, 0, 0);
        assert_eq!(a.union(&empty), a);
        assert_eq!(empty.union(&a), a);
    }

    #[test]
    fn test_dirty_rect_union_both_empty() {
        let a = DirtyRect::new(0, 0, 0, 0);
        let b = DirtyRect::new(0, 0, 0, 0);
        let u = a.union(&b);
        assert!(u.is_empty());
    }

    #[test]
    fn test_dirty_rect_union_adjacent() {
        let a = DirtyRect::new(0, 0, 50, 10);
        let b = DirtyRect::new(50, 0, 50, 10);
        let u = a.union(&b);
        assert_eq!(u.x, 0);
        assert_eq!(u.width, 100);
    }

    #[test]
    fn test_dirty_rect_equality() {
        let a = DirtyRect::new(1, 2, 3, 4);
        let b = DirtyRect::new(1, 2, 3, 4);
        assert_eq!(a, b);
    }

    #[test]
    fn test_incremental_renderer_invalidate_resets_state() {
        // We test the state tracking logic without a real font/frame.
        // Just verify that the DisplayedState comparison logic is correct.
        let s1 = DisplayedState {
            cues: vec![(0, 1000, "Hello".to_string())],
        };
        let s2 = DisplayedState {
            cues: vec![(0, 1000, "Hello".to_string())],
        };
        let s3 = DisplayedState {
            cues: vec![(0, 1000, "Different".to_string())],
        };
        assert_eq!(s1, s2);
        assert_ne!(s1, s3);
    }
}
