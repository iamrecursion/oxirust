//! Digital Video Effects (DVE) for video switchers.
//!
//! Implements picture-in-picture, squeeze, crop, and other spatial transformations.

use oximedia_codec::VideoFrame;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur with DVE operations.
#[derive(Error, Debug, Clone)]
pub enum DveError {
    #[error("Invalid position: {0}")]
    InvalidPosition(f32),

    #[error("Invalid scale: {0}")]
    InvalidScale(f32),

    #[error("Invalid rotation: {0}")]
    InvalidRotation(f32),

    #[error("Processing error: {0}")]
    ProcessingError(String),
}

/// DVE position in normalized coordinates (0.0 - 1.0).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DvePosition {
    /// X position (0.0 = left, 1.0 = right)
    pub x: f32,
    /// Y position (0.0 = top, 1.0 = bottom)
    pub y: f32,
}

impl DvePosition {
    /// Create a new position.
    pub fn new(x: f32, y: f32) -> Self {
        Self {
            x: x.clamp(0.0, 1.0),
            y: y.clamp(0.0, 1.0),
        }
    }

    /// Center position.
    pub fn center() -> Self {
        Self::new(0.5, 0.5)
    }

    /// Top-left corner.
    pub fn top_left() -> Self {
        Self::new(0.0, 0.0)
    }

    /// Top-right corner.
    pub fn top_right() -> Self {
        Self::new(1.0, 0.0)
    }

    /// Bottom-left corner.
    pub fn bottom_left() -> Self {
        Self::new(0.0, 1.0)
    }

    /// Bottom-right corner.
    pub fn bottom_right() -> Self {
        Self::new(1.0, 1.0)
    }
}

impl Default for DvePosition {
    fn default() -> Self {
        Self::center()
    }
}

/// DVE scale (size).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DveScale {
    /// X scale (width multiplier)
    pub x: f32,
    /// Y scale (height multiplier)
    pub y: f32,
}

impl DveScale {
    /// Create a new scale.
    pub fn new(x: f32, y: f32) -> Self {
        Self {
            x: x.max(0.0),
            y: y.max(0.0),
        }
    }

    /// Uniform scale.
    pub fn uniform(scale: f32) -> Self {
        Self::new(scale, scale)
    }

    /// Full size (1:1).
    pub fn full() -> Self {
        Self::uniform(1.0)
    }

    /// Half size (1:2).
    pub fn half() -> Self {
        Self::uniform(0.5)
    }

    /// Quarter size (1:4).
    pub fn quarter() -> Self {
        Self::uniform(0.25)
    }
}

impl Default for DveScale {
    fn default() -> Self {
        Self::full()
    }
}

/// DVE crop rectangle in normalized coordinates.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DveCrop {
    /// Top edge (0.0 - 1.0)
    pub top: f32,
    /// Bottom edge (0.0 - 1.0)
    pub bottom: f32,
    /// Left edge (0.0 - 1.0)
    pub left: f32,
    /// Right edge (0.0 - 1.0)
    pub right: f32,
}

impl DveCrop {
    /// Create a new crop.
    pub fn new(top: f32, bottom: f32, left: f32, right: f32) -> Self {
        Self {
            top: top.clamp(0.0, 1.0),
            bottom: bottom.clamp(0.0, 1.0),
            left: left.clamp(0.0, 1.0),
            right: right.clamp(0.0, 1.0),
        }
    }

    /// No crop (full frame).
    pub fn none() -> Self {
        Self::new(0.0, 1.0, 0.0, 1.0)
    }

    /// Center crop.
    pub fn center(amount: f32) -> Self {
        let half = amount / 2.0;
        Self::new(half, 1.0 - half, half, 1.0 - half)
    }
}

impl Default for DveCrop {
    fn default() -> Self {
        Self::none()
    }
}

/// DVE border configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DveBorder {
    /// Enable border
    pub enabled: bool,
    /// Border width in pixels
    pub width: f32,
    /// Border color (R, G, B)
    pub color: (u8, u8, u8),
    /// Border opacity (0.0 - 1.0)
    pub opacity: f32,
}

impl DveBorder {
    /// Create a new border.
    pub fn new() -> Self {
        Self {
            enabled: false,
            width: 2.0,
            color: (255, 255, 255),
            opacity: 1.0,
        }
    }

    /// Enable the border.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable the border.
    pub fn disable(&mut self) {
        self.enabled = false;
    }
}

impl Default for DveBorder {
    fn default() -> Self {
        Self::new()
    }
}

/// DVE shadow configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DveShadow {
    /// Enable shadow
    pub enabled: bool,
    /// Shadow offset X
    pub offset_x: f32,
    /// Shadow offset Y
    pub offset_y: f32,
    /// Shadow blur radius
    pub blur: f32,
    /// Shadow opacity (0.0 - 1.0)
    pub opacity: f32,
}

impl DveShadow {
    /// Create a new shadow.
    pub fn new() -> Self {
        Self {
            enabled: false,
            offset_x: 5.0,
            offset_y: 5.0,
            blur: 10.0,
            opacity: 0.5,
        }
    }

    /// Enable the shadow.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable the shadow.
    pub fn disable(&mut self) {
        self.enabled = false;
    }
}

impl Default for DveShadow {
    fn default() -> Self {
        Self::new()
    }
}

/// DVE parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DveParams {
    /// Position
    pub position: DvePosition,
    /// Scale
    pub scale: DveScale,
    /// Rotation in degrees (0.0 - 360.0)
    pub rotation: f32,
    /// Crop
    pub crop: DveCrop,
    /// Border
    pub border: DveBorder,
    /// Shadow
    pub shadow: DveShadow,
    /// Aspect ratio correction
    pub maintain_aspect: bool,
}

impl DveParams {
    /// Create new DVE parameters with defaults.
    pub fn new() -> Self {
        Self {
            position: DvePosition::center(),
            scale: DveScale::full(),
            rotation: 0.0,
            crop: DveCrop::none(),
            border: DveBorder::new(),
            shadow: DveShadow::new(),
            maintain_aspect: true,
        }
    }

    /// Create parameters for picture-in-picture (PIP).
    pub fn pip_bottom_right() -> Self {
        Self {
            position: DvePosition::new(0.75, 0.75),
            scale: DveScale::quarter(),
            rotation: 0.0,
            crop: DveCrop::none(),
            border: DveBorder::new(),
            shadow: DveShadow::new(),
            maintain_aspect: true,
        }
    }

    /// Create parameters for side-by-side.
    pub fn side_by_side_left() -> Self {
        Self {
            position: DvePosition::new(0.25, 0.5),
            scale: DveScale::new(0.5, 1.0),
            rotation: 0.0,
            crop: DveCrop::none(),
            border: DveBorder::new(),
            shadow: DveShadow::new(),
            maintain_aspect: false,
        }
    }

    /// Create parameters for side-by-side right.
    pub fn side_by_side_right() -> Self {
        Self {
            position: DvePosition::new(0.75, 0.5),
            scale: DveScale::new(0.5, 1.0),
            rotation: 0.0,
            crop: DveCrop::none(),
            border: DveBorder::new(),
            shadow: DveShadow::new(),
            maintain_aspect: false,
        }
    }

    /// Set position.
    pub fn set_position(&mut self, x: f32, y: f32) {
        self.position = DvePosition::new(x, y);
    }

    /// Set scale.
    pub fn set_scale(&mut self, x: f32, y: f32) {
        self.scale = DveScale::new(x, y);
    }

    /// Set rotation.
    pub fn set_rotation(&mut self, rotation: f32) -> Result<(), DveError> {
        if !(0.0..=360.0).contains(&rotation) {
            return Err(DveError::InvalidRotation(rotation));
        }
        self.rotation = rotation;
        Ok(())
    }
}

impl Default for DveParams {
    fn default() -> Self {
        Self::new()
    }
}

/// DVE processor.
pub struct DveProcessor {
    params: DveParams,
    enabled: bool,
}

impl DveProcessor {
    /// Create a new DVE processor.
    pub fn new() -> Self {
        Self {
            params: DveParams::new(),
            enabled: true,
        }
    }

    /// Create with specific parameters.
    pub fn with_params(params: DveParams) -> Self {
        Self {
            params,
            enabled: true,
        }
    }

    /// Get the parameters.
    pub fn params(&self) -> &DveParams {
        &self.params
    }

    /// Get mutable parameters.
    pub fn params_mut(&mut self) -> &mut DveParams {
        &mut self.params
    }

    /// Set parameters.
    pub fn set_params(&mut self, params: DveParams) {
        self.params = params;
    }

    /// Enable or disable the DVE.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if the DVE is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Process a video frame through the DVE.
    ///
    /// Applies crop, scale, position (and optionally border) to produce a
    /// composited output frame on a black canvas of the same dimensions as the input.
    pub fn process(&self, input: &VideoFrame) -> Result<VideoFrame, DveError> {
        if !self.enabled {
            // DVE disabled: return a clone of the input
            return Ok(input.clone());
        }

        if input.planes.is_empty() {
            return Err(DveError::ProcessingError(
                "Input frame has no planes".to_string(),
            ));
        }

        let canvas_w = input.width;
        let canvas_h = input.height;

        // Create output frame (black canvas)
        let mut output = VideoFrame::new(input.format, canvas_w, canvas_h);
        output.allocate();
        output.timestamp = input.timestamp;
        output.frame_type = input.frame_type;
        output.color_info = input.color_info;

        // Work on the luma (first) plane for simplicity. The same logic
        // is applied to all planes, adjusting for chroma sub-sampling.
        for plane_idx in 0..output.planes.len().min(input.planes.len()) {
            let src_plane = &input.planes[plane_idx];
            let src_w = src_plane.width as usize;
            let src_h = src_plane.height as usize;
            let src_stride = src_plane.stride;

            let dst_plane = &mut output.planes[plane_idx];
            let dst_w = dst_plane.width as usize;
            let dst_h = dst_plane.height as usize;
            let dst_stride = dst_plane.stride;

            // Calculate source crop region in pixels
            let crop_x0 = (self.params.crop.left * src_w as f32) as usize;
            let crop_y0 = (self.params.crop.top * src_h as f32) as usize;
            let crop_x1 = (self.params.crop.right * src_w as f32) as usize;
            let crop_y1 = (self.params.crop.bottom * src_h as f32) as usize;

            let crop_w = crop_x1.saturating_sub(crop_x0).max(1);
            let crop_h = crop_y1.saturating_sub(crop_y0).max(1);

            // Calculate destination rectangle
            let out_w = (dst_w as f32 * self.params.scale.x) as usize;
            let out_h = (dst_h as f32 * self.params.scale.y) as usize;

            if out_w == 0 || out_h == 0 {
                continue;
            }

            let out_x = ((self.params.position.x * dst_w as f32) - (out_w as f32 / 2.0)) as i32;
            let out_y = ((self.params.position.y * dst_h as f32) - (out_h as f32 / 2.0)) as i32;

            // Nearest-neighbour blit from cropped source into the destination rectangle
            for dy in 0..out_h {
                let dst_row = out_y + dy as i32;
                if dst_row < 0 || dst_row as usize >= dst_h {
                    continue;
                }
                let dst_row = dst_row as usize;

                for dx in 0..out_w {
                    let dst_col = out_x + dx as i32;
                    if dst_col < 0 || dst_col as usize >= dst_w {
                        continue;
                    }
                    let dst_col = dst_col as usize;

                    // Map destination pixel back to cropped source
                    let src_x = crop_x0 + (dx * crop_w / out_w).min(crop_w.saturating_sub(1));
                    let src_y = crop_y0 + (dy * crop_h / out_h).min(crop_h.saturating_sub(1));

                    if src_x < src_w && src_y < src_h {
                        let si = src_y * src_stride + src_x;
                        let di = dst_row * dst_stride + dst_col;
                        if si < src_plane.data.len() && di < dst_plane.data.len() {
                            dst_plane.data[di] = src_plane.data[si];
                        }
                    }
                }
            }

            // Draw border if enabled
            if self.params.border.enabled {
                let border_px = (self.params.border.width as usize).max(1);
                let border_val = match plane_idx {
                    0 => {
                        // Luma: use BT.709 approximate
                        let (r, g, b) = self.params.border.color;
                        (f32::from(r) * 0.2126 + f32::from(g) * 0.7152 + f32::from(b) * 0.0722)
                            as u8
                    }
                    _ => 128u8, // Chroma neutral
                };

                for dy in 0..out_h {
                    let dst_row = out_y + dy as i32;
                    if dst_row < 0 || dst_row as usize >= dst_h {
                        continue;
                    }
                    let dst_row = dst_row as usize;

                    for dx in 0..out_w {
                        let dst_col = out_x + dx as i32;
                        if dst_col < 0 || dst_col as usize >= dst_w {
                            continue;
                        }
                        let dst_col = dst_col as usize;

                        let on_border = dx < border_px
                            || dx >= out_w.saturating_sub(border_px)
                            || dy < border_px
                            || dy >= out_h.saturating_sub(border_px);

                        if on_border {
                            let di = dst_row * dst_stride + dst_col;
                            if di < dst_plane.data.len() {
                                dst_plane.data[di] = border_val;
                            }
                        }
                    }
                }
            }
        }

        Ok(output)
    }

    /// Calculate the output rectangle for the current parameters.
    pub fn output_rect(&self, canvas_width: u32, canvas_height: u32) -> (f32, f32, f32, f32) {
        let width = canvas_width as f32 * self.params.scale.x;
        let height = canvas_height as f32 * self.params.scale.y;

        let x = self.params.position.x * canvas_width as f32 - width / 2.0;
        let y = self.params.position.y * canvas_height as f32 - height / 2.0;

        (x, y, width, height)
    }
}

impl Default for DveProcessor {
    fn default() -> Self {
        Self::new()
    }
}

// ── Keyframe animation ────────────────────────────────────────────────────────

/// Easing function for DVE fly-key keyframe interpolation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DveEasing {
    /// Linear interpolation (t).
    Linear,
    /// Ease-in (t²).
    EaseIn,
    /// Ease-out (1-(1-t)²).
    EaseOut,
    /// Ease-in-out (smooth S-curve).
    EaseInOut,
    /// Hold start value until next keyframe.
    Hold,
}

impl DveEasing {
    /// Apply the easing function to normalized progress `t` in [0.0, 1.0].
    pub fn apply(self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Linear => t,
            Self::EaseIn => t * t,
            Self::EaseOut => 1.0 - (1.0 - t) * (1.0 - t),
            Self::EaseInOut => {
                if t < 0.5 {
                    2.0 * t * t
                } else {
                    1.0 - 2.0 * (1.0 - t) * (1.0 - t)
                }
            }
            Self::Hold => 0.0,
        }
    }
}

/// A single DVE keyframe in an animation sequence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DveKeyframe {
    /// Frame index (zero-based) at which this keyframe is defined.
    pub frame: u32,
    /// Position at this keyframe.
    pub position: DvePosition,
    /// Scale at this keyframe.
    pub scale: DveScale,
    /// Rotation at this keyframe (degrees, 0.0–360.0).
    pub rotation: f32,
    /// Easing applied from this keyframe to the next.
    pub easing: DveEasing,
}

impl DveKeyframe {
    /// Create a new keyframe.
    pub fn new(
        frame: u32,
        position: DvePosition,
        scale: DveScale,
        rotation: f32,
        easing: DveEasing,
    ) -> Self {
        Self {
            frame,
            position,
            scale,
            rotation: rotation.clamp(0.0, 360.0),
            easing,
        }
    }

    /// Create a keyframe using default DVE params.
    pub fn at_frame(frame: u32) -> Self {
        Self::new(
            frame,
            DvePosition::center(),
            DveScale::full(),
            0.0,
            DveEasing::Linear,
        )
    }
}

/// Fly-key animation that stores a sequence of `DveKeyframe` values and
/// interpolates between them to produce smooth DVE motion paths.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DveFlyKeyAnimation {
    keyframes: Vec<DveKeyframe>,
}

impl DveFlyKeyAnimation {
    /// Create a new empty animation.
    pub fn new() -> Self {
        Self {
            keyframes: Vec::new(),
        }
    }

    /// Add a keyframe to the animation.
    ///
    /// Keyframes must be added in non-decreasing frame order; adding one out
    /// of order returns `DveError::ProcessingError`.
    pub fn add_keyframe(&mut self, keyframe: DveKeyframe) -> Result<(), DveError> {
        if let Some(last) = self.keyframes.last() {
            if keyframe.frame < last.frame {
                return Err(DveError::ProcessingError(format!(
                    "Keyframe at frame {} would be before last keyframe at frame {}",
                    keyframe.frame, last.frame
                )));
            }
        }
        self.keyframes.push(keyframe);
        Ok(())
    }

    /// Number of keyframes stored.
    pub fn keyframe_count(&self) -> usize {
        self.keyframes.len()
    }

    /// Remove all keyframes.
    pub fn clear_keyframes(&mut self) {
        self.keyframes.clear();
    }

    /// Get a slice of all keyframes.
    pub fn keyframes(&self) -> &[DveKeyframe] {
        &self.keyframes
    }

    /// Evaluate the animation at the given frame, returning interpolated
    /// `DveParams`.  Returns `None` if the animation has no keyframes.
    pub fn evaluate(&self, frame: u32) -> Option<DveParams> {
        if self.keyframes.is_empty() {
            return None;
        }

        // Before or at first keyframe: clamp to first.
        if frame <= self.keyframes[0].frame {
            let kf = &self.keyframes[0];
            let mut params = DveParams::new();
            params.position = kf.position;
            params.scale = kf.scale;
            params.rotation = kf.rotation;
            return Some(params);
        }

        // After or at last keyframe: clamp to last.
        let last = self.keyframes.last()?;
        if frame >= last.frame {
            let mut params = DveParams::new();
            params.position = last.position;
            params.scale = last.scale;
            params.rotation = last.rotation;
            return Some(params);
        }

        // Find the surrounding pair [a, b] such that a.frame <= frame < b.frame.
        let b_idx = self.keyframes.partition_point(|kf| kf.frame <= frame);
        if b_idx == 0 || b_idx >= self.keyframes.len() {
            // Shouldn't happen given the clamps above, but be safe.
            return None;
        }
        let a = &self.keyframes[b_idx - 1];
        let b = &self.keyframes[b_idx];

        let span = (b.frame - a.frame) as f32;
        let raw_t = if span > 0.0 {
            (frame - a.frame) as f32 / span
        } else {
            1.0
        };
        let t = a.easing.apply(raw_t);

        let lerp = |va: f32, vb: f32| va + (vb - va) * t;

        let mut params = DveParams::new();
        params.position = DvePosition {
            x: lerp(a.position.x, b.position.x),
            y: lerp(a.position.y, b.position.y),
        };
        params.scale = DveScale {
            x: lerp(a.scale.x, b.scale.x),
            y: lerp(a.scale.y, b.scale.y),
        };
        // Shortest-path rotation interpolation.
        let rot_diff = {
            let d = b.rotation - a.rotation;
            // Normalise to [-180, 180]
            if d > 180.0 {
                d - 360.0
            } else if d < -180.0 {
                d + 360.0
            } else {
                d
            }
        };
        params.rotation = (a.rotation + rot_diff * t).rem_euclid(360.0);

        Some(params)
    }
}

/// DVE fly-key: combines a `DveProcessor` with a `DveFlyKeyAnimation`,
/// advancing frame-by-frame and applying interpolated parameters automatically.
pub struct DveFlyKey {
    processor: DveProcessor,
    animation: DveFlyKeyAnimation,
    current_frame: u32,
}

impl DveFlyKey {
    /// Create a new fly-key with default settings.
    pub fn new() -> Self {
        Self {
            processor: DveProcessor::new(),
            animation: DveFlyKeyAnimation::new(),
            current_frame: 0,
        }
    }

    /// Access the animation for keyframe editing.
    pub fn animation_mut(&mut self) -> &mut DveFlyKeyAnimation {
        &mut self.animation
    }

    /// Access the animation read-only.
    pub fn animation(&self) -> &DveFlyKeyAnimation {
        &self.animation
    }

    /// Access the underlying DVE processor (read-only).
    pub fn processor(&self) -> &DveProcessor {
        &self.processor
    }

    /// Access the underlying DVE processor (mutable).
    pub fn processor_mut(&mut self) -> &mut DveProcessor {
        &mut self.processor
    }

    /// Current animation frame index.
    pub fn current_frame(&self) -> u32 {
        self.current_frame
    }

    /// Advance the animation by one frame and update the processor's params.
    ///
    /// The frame counter is incremented first, then the animation is evaluated
    /// at the new frame index.  This means after N calls `current_frame()` is N
    /// and the processor reflects frame N's interpolated state.
    ///
    /// If the animation has keyframes, the interpolated params are applied;
    /// otherwise the processor's existing params remain unchanged.
    pub fn advance_frame(&mut self) {
        self.current_frame = self.current_frame.saturating_add(1);
        if let Some(params) = self.animation.evaluate(self.current_frame) {
            self.processor.set_params(params);
        }
    }

    /// Reset the frame counter to zero and re-apply the first keyframe (if any).
    pub fn reset(&mut self) {
        self.current_frame = 0;
        if let Some(params) = self.animation.evaluate(0) {
            self.processor.set_params(params);
        }
    }
}

impl Default for DveFlyKey {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dve_position_creation() {
        let pos = DvePosition::new(0.5, 0.5);
        assert_eq!(pos.x, 0.5);
        assert_eq!(pos.y, 0.5);

        let center = DvePosition::center();
        assert_eq!(center.x, 0.5);
        assert_eq!(center.y, 0.5);
    }

    #[test]
    fn test_dve_position_corners() {
        let tl = DvePosition::top_left();
        assert_eq!(tl.x, 0.0);
        assert_eq!(tl.y, 0.0);

        let br = DvePosition::bottom_right();
        assert_eq!(br.x, 1.0);
        assert_eq!(br.y, 1.0);
    }

    #[test]
    fn test_dve_position_clamping() {
        let pos = DvePosition::new(-0.5, 1.5);
        assert_eq!(pos.x, 0.0);
        assert_eq!(pos.y, 1.0);
    }

    #[test]
    fn test_dve_scale() {
        let scale = DveScale::uniform(0.5);
        assert_eq!(scale.x, 0.5);
        assert_eq!(scale.y, 0.5);

        let full = DveScale::full();
        assert_eq!(full.x, 1.0);
        assert_eq!(full.y, 1.0);

        let quarter = DveScale::quarter();
        assert_eq!(quarter.x, 0.25);
        assert_eq!(quarter.y, 0.25);
    }

    #[test]
    fn test_dve_crop() {
        let crop = DveCrop::none();
        assert_eq!(crop.top, 0.0);
        assert_eq!(crop.bottom, 1.0);
        assert_eq!(crop.left, 0.0);
        assert_eq!(crop.right, 1.0);

        let center_crop = DveCrop::center(0.2);
        assert_eq!(center_crop.top, 0.1);
        assert_eq!(center_crop.bottom, 0.9);
    }

    #[test]
    fn test_dve_border() {
        let mut border = DveBorder::new();
        assert!(!border.enabled);

        border.enable();
        assert!(border.enabled);

        border.disable();
        assert!(!border.enabled);
    }

    #[test]
    fn test_dve_shadow() {
        let mut shadow = DveShadow::new();
        assert!(!shadow.enabled);

        shadow.enable();
        assert!(shadow.enabled);

        shadow.offset_x = 10.0;
        shadow.offset_y = 10.0;
        assert_eq!(shadow.offset_x, 10.0);
        assert_eq!(shadow.offset_y, 10.0);
    }

    #[test]
    fn test_dve_params_default() {
        let params = DveParams::new();
        assert_eq!(params.position.x, 0.5);
        assert_eq!(params.position.y, 0.5);
        assert_eq!(params.scale.x, 1.0);
        assert_eq!(params.scale.y, 1.0);
        assert_eq!(params.rotation, 0.0);
        assert!(params.maintain_aspect);
    }

    #[test]
    fn test_dve_params_pip() {
        let params = DveParams::pip_bottom_right();
        assert_eq!(params.position.x, 0.75);
        assert_eq!(params.position.y, 0.75);
        assert_eq!(params.scale.x, 0.25);
        assert_eq!(params.scale.y, 0.25);
    }

    #[test]
    fn test_dve_params_side_by_side() {
        let left = DveParams::side_by_side_left();
        assert_eq!(left.position.x, 0.25);
        assert_eq!(left.scale.x, 0.5);
        assert!(!left.maintain_aspect);

        let right = DveParams::side_by_side_right();
        assert_eq!(right.position.x, 0.75);
        assert_eq!(right.scale.x, 0.5);
    }

    #[test]
    fn test_set_position() {
        let mut params = DveParams::new();
        params.set_position(0.3, 0.7);
        assert_eq!(params.position.x, 0.3);
        assert_eq!(params.position.y, 0.7);
    }

    #[test]
    fn test_set_scale() {
        let mut params = DveParams::new();
        params.set_scale(0.5, 0.8);
        assert_eq!(params.scale.x, 0.5);
        assert_eq!(params.scale.y, 0.8);
    }

    #[test]
    fn test_set_rotation() {
        let mut params = DveParams::new();

        assert!(params.set_rotation(45.0).is_ok());
        assert_eq!(params.rotation, 45.0);

        assert!(params.set_rotation(180.0).is_ok());
        assert_eq!(params.rotation, 180.0);

        assert!(params.set_rotation(361.0).is_err());
        assert!(params.set_rotation(-1.0).is_err());
    }

    #[test]
    fn test_dve_processor_creation() {
        let processor = DveProcessor::new();
        assert!(processor.is_enabled());
        assert_eq!(processor.params().position.x, 0.5);
    }

    #[test]
    fn test_dve_processor_enable_disable() {
        let mut processor = DveProcessor::new();
        assert!(processor.is_enabled());

        processor.set_enabled(false);
        assert!(!processor.is_enabled());
    }

    #[test]
    fn test_dve_processor_with_params() {
        let params = DveParams::pip_bottom_right();
        let processor = DveProcessor::with_params(params);

        assert_eq!(processor.params().position.x, 0.75);
        assert_eq!(processor.params().scale.x, 0.25);
    }

    #[test]
    fn test_output_rect() {
        let mut processor = DveProcessor::new();
        processor.params_mut().position = DvePosition::center();
        processor.params_mut().scale = DveScale::half();

        let (x, y, width, height) = processor.output_rect(1920, 1080);

        assert_eq!(width, 960.0);
        assert_eq!(height, 540.0);
        assert_eq!(x, 480.0); // Center - half width
        assert_eq!(y, 270.0); // Center - half height
    }

    #[test]
    fn test_output_rect_quarter() {
        let mut processor = DveProcessor::new();
        processor.params_mut().position = DvePosition::top_left();
        processor.params_mut().scale = DveScale::quarter();

        let (x, y, width, height) = processor.output_rect(1920, 1080);

        assert_eq!(width, 480.0);
        assert_eq!(height, 270.0);
        assert_eq!(x, -240.0); // Left edge - half width
        assert_eq!(y, -135.0); // Top edge - half height
    }

    // ── DveFlyKeyAnimation tests ──────────────────────────────────────────────

    #[test]
    fn test_easing_linear() {
        assert!((DveEasing::Linear.apply(0.5) - 0.5).abs() < 1e-6);
        assert!((DveEasing::Linear.apply(0.0) - 0.0).abs() < 1e-6);
        assert!((DveEasing::Linear.apply(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_easing_ease_in() {
        let t = 0.5_f32;
        let expected = t * t; // 0.25
        assert!((DveEasing::EaseIn.apply(t) - expected).abs() < 1e-6);
    }

    #[test]
    fn test_easing_ease_out() {
        let t = 0.5_f32;
        let expected = 1.0 - (1.0 - t) * (1.0 - t); // 0.75
        assert!((DveEasing::EaseOut.apply(t) - expected).abs() < 1e-6);
    }

    #[test]
    fn test_easing_ease_in_out_midpoint() {
        // At t=0.5, EaseInOut should return 0.5 (symmetric).
        assert!((DveEasing::EaseInOut.apply(0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_easing_hold_always_zero() {
        assert!((DveEasing::Hold.apply(0.0) - 0.0).abs() < 1e-6);
        assert!((DveEasing::Hold.apply(0.5) - 0.0).abs() < 1e-6);
        assert!((DveEasing::Hold.apply(1.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_animation_empty_returns_none() {
        let anim = DveFlyKeyAnimation::new();
        assert!(anim.evaluate(0).is_none());
        assert_eq!(anim.keyframe_count(), 0);
    }

    #[test]
    fn test_animation_single_keyframe() {
        let mut anim = DveFlyKeyAnimation::new();
        let kf = DveKeyframe::new(
            10,
            DvePosition::new(0.3, 0.7),
            DveScale::half(),
            45.0,
            DveEasing::Linear,
        );
        anim.add_keyframe(kf).expect("should succeed");

        // Any frame should return the single keyframe value.
        let params = anim.evaluate(0).expect("should return params");
        assert!((params.position.x - 0.3).abs() < 1e-5);
        assert!((params.scale.x - 0.5).abs() < 1e-5);

        let params_after = anim.evaluate(100).expect("should return params");
        assert!((params_after.position.x - 0.3).abs() < 1e-5);
    }

    #[test]
    fn test_animation_two_keyframes_midpoint() {
        let mut anim = DveFlyKeyAnimation::new();
        anim.add_keyframe(DveKeyframe::new(
            0,
            DvePosition::new(0.0, 0.0),
            DveScale::full(),
            0.0,
            DveEasing::Linear,
        ))
        .expect("ok");
        anim.add_keyframe(DveKeyframe::new(
            10,
            DvePosition::new(1.0, 1.0),
            DveScale::half(),
            180.0,
            DveEasing::Linear,
        ))
        .expect("ok");

        // At frame 5 (midpoint, linear), position should be 0.5, scale x should be 0.75.
        let params = anim.evaluate(5).expect("should return params");
        assert!((params.position.x - 0.5).abs() < 1e-5);
        assert!((params.position.y - 0.5).abs() < 1e-5);
        assert!((params.scale.x - 0.75).abs() < 1e-5);
        assert!((params.rotation - 90.0).abs() < 1e-4);
    }

    #[test]
    fn test_animation_out_of_order_keyframe_error() {
        let mut anim = DveFlyKeyAnimation::new();
        anim.add_keyframe(DveKeyframe::at_frame(10)).expect("ok");
        let result = anim.add_keyframe(DveKeyframe::at_frame(5));
        assert!(result.is_err(), "out-of-order keyframe should fail");
    }

    #[test]
    fn test_animation_clear_keyframes() {
        let mut anim = DveFlyKeyAnimation::new();
        anim.add_keyframe(DveKeyframe::at_frame(0)).expect("ok");
        assert_eq!(anim.keyframe_count(), 1);
        anim.clear_keyframes();
        assert_eq!(anim.keyframe_count(), 0);
        assert!(anim.evaluate(0).is_none());
    }

    #[test]
    fn test_fly_key_advance_frame() {
        let mut fly_key = DveFlyKey::new();

        // Add two keyframes spanning frames 0-10.
        fly_key
            .animation_mut()
            .add_keyframe(DveKeyframe::new(
                0,
                DvePosition::new(0.0, 0.0),
                DveScale::full(),
                0.0,
                DveEasing::Linear,
            ))
            .expect("ok");
        fly_key
            .animation_mut()
            .add_keyframe(DveKeyframe::new(
                10,
                DvePosition::new(1.0, 1.0),
                DveScale::half(),
                180.0,
                DveEasing::Linear,
            ))
            .expect("ok");

        // Advance 5 frames.
        for _ in 0..5 {
            fly_key.advance_frame();
        }
        assert_eq!(fly_key.current_frame(), 5);
        let pos = fly_key.processor().params().position;
        assert!((pos.x - 0.5).abs() < 1e-4, "expected ~0.5, got {}", pos.x);
    }

    #[test]
    fn test_fly_key_reset() {
        let mut fly_key = DveFlyKey::new();
        fly_key
            .animation_mut()
            .add_keyframe(DveKeyframe::at_frame(0))
            .expect("ok");
        for _ in 0..20 {
            fly_key.advance_frame();
        }
        assert_eq!(fly_key.current_frame(), 20);
        fly_key.reset();
        assert_eq!(fly_key.current_frame(), 0);
    }
}
