//! Adjustment layer module for applying effects to all tracks below.
//!
//! An adjustment layer is a special transparent clip that sits on a video track
//! and applies its effect stack to the composite of all tracks beneath it.
//! This allows non-destructive grading and filtering without modifying
//! individual clips.
//!
//! # Workflow
//!
//! 1. Create an `AdjustmentLayer` with a time range and effects.
//! 2. Place it on a video track above the content tracks.
//! 3. During rendering, the `AdjustmentLayerProcessor` detects adjustment
//!    layers and applies their effects to the composited output of all
//!    lower tracks.
//!
//! # Example
//!
//! ```
//! use oximedia_timeline::adjustment_layer::{AdjustmentLayer, AdjustmentLayerProcessor};
//! use oximedia_timeline::types::{Position, Duration};
//! use oximedia_timeline::effects::{Effect, EffectType, EffectStack};
//!
//! let mut layer = AdjustmentLayer::new(
//!     "Color Grade",
//!     Position::new(0),
//!     Duration::new(240),
//! );
//! layer.add_effect(Effect::new("Brightness +10".to_string(), EffectType::BrightnessContrast));
//! assert!(layer.set_opacity(0.8).is_ok());
//!
//! assert!(layer.contains_position(Position::new(100)));
//! assert!(!layer.contains_position(Position::new(300)));
//! ```

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::effects::{Effect, EffectStack, EffectType};
use crate::error::{TimelineError, TimelineResult};
use crate::renderer::PixelBuffer;
use crate::types::{Duration, Position};

/// Unique identifier for an adjustment layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AdjustmentLayerId(Uuid);

impl AdjustmentLayerId {
    /// Creates a new random adjustment layer ID.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for AdjustmentLayerId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for AdjustmentLayerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Blend mode for how the adjustment layer's effect result is composited.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdjustmentBlendMode {
    /// Normal: the processed result replaces the input entirely (modulated by
    /// opacity).
    Normal,
    /// Multiply: processed result is multiplied with the input.
    Multiply,
    /// Screen: inverse-multiply blend.
    Screen,
    /// Overlay: combination of multiply and screen.
    Overlay,
    /// SoftLight: gentle lightening/darkening.
    SoftLight,
}

impl Default for AdjustmentBlendMode {
    fn default() -> Self {
        Self::Normal
    }
}

impl AdjustmentBlendMode {
    /// Blend two pixel values (0-255 range) using this blend mode.
    ///
    /// `base` is the original pixel value, `processed` is the post-effect value.
    /// Returns the blended value.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn blend(self, base: u8, processed: u8) -> u8 {
        let b = f32::from(base) / 255.0;
        let p = f32::from(processed) / 255.0;
        let result = match self {
            Self::Normal => p,
            Self::Multiply => b * p,
            Self::Screen => 1.0 - (1.0 - b) * (1.0 - p),
            Self::Overlay => {
                if b < 0.5 {
                    2.0 * b * p
                } else {
                    1.0 - 2.0 * (1.0 - b) * (1.0 - p)
                }
            }
            Self::SoftLight => {
                if p < 0.5 {
                    b - (1.0 - 2.0 * p) * b * (1.0 - b)
                } else {
                    let d = if b <= 0.25 {
                        ((16.0 * b - 12.0) * b + 4.0) * b
                    } else {
                        b.sqrt()
                    };
                    b + (2.0 * p - 1.0) * (d - b)
                }
            }
        };
        (result.clamp(0.0, 1.0) * 255.0).round() as u8
    }
}

/// An adjustment layer that applies effects to all tracks below it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdjustmentLayer {
    /// Unique identifier.
    pub id: AdjustmentLayerId,
    /// Human-readable name.
    pub name: String,
    /// Start position on the timeline.
    pub timeline_in: Position,
    /// Duration on the timeline.
    pub duration: Duration,
    /// Effect stack applied by this adjustment layer.
    pub effects: EffectStack,
    /// Opacity of the adjustment (0.0 = no effect, 1.0 = full effect).
    pub opacity: f32,
    /// Whether this adjustment layer is enabled.
    pub enabled: bool,
    /// Blend mode for compositing the effect result.
    pub blend_mode: AdjustmentBlendMode,
    /// Whether to apply effects to the alpha channel.
    pub affect_alpha: bool,
}

impl AdjustmentLayer {
    /// Creates a new adjustment layer.
    #[must_use]
    pub fn new(name: impl Into<String>, timeline_in: Position, duration: Duration) -> Self {
        Self {
            id: AdjustmentLayerId::new(),
            name: name.into(),
            timeline_in,
            duration,
            effects: EffectStack::new(),
            opacity: 1.0,
            enabled: true,
            blend_mode: AdjustmentBlendMode::Normal,
            affect_alpha: false,
        }
    }

    /// Returns the timeline out position.
    #[must_use]
    pub fn timeline_out(&self) -> Position {
        self.timeline_in + self.duration
    }

    /// Returns `true` if the given position is within this adjustment layer.
    #[must_use]
    pub fn contains_position(&self, position: Position) -> bool {
        position >= self.timeline_in && position < self.timeline_out()
    }

    /// Returns `true` if this adjustment layer overlaps with the given range.
    #[must_use]
    pub fn overlaps(&self, start: Position, end: Position) -> bool {
        self.timeline_in < end && self.timeline_out() > start
    }

    /// Adds an effect to this adjustment layer's stack.
    pub fn add_effect(&mut self, effect: Effect) {
        self.effects.add_effect(effect);
    }

    /// Returns the number of effects.
    #[must_use]
    pub fn effect_count(&self) -> usize {
        self.effects.len()
    }

    /// Sets the opacity (0.0 to 1.0).
    ///
    /// # Errors
    ///
    /// Returns error if opacity is outside the valid range.
    pub fn set_opacity(&mut self, opacity: f32) -> TimelineResult<()> {
        if !(0.0..=1.0).contains(&opacity) {
            return Err(TimelineError::Other(format!(
                "Invalid opacity: {opacity} (must be 0.0-1.0)"
            )));
        }
        self.opacity = opacity;
        Ok(())
    }

    /// Sets the blend mode.
    pub fn set_blend_mode(&mut self, mode: AdjustmentBlendMode) {
        self.blend_mode = mode;
    }

    /// Enables or disables the adjustment layer.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Trims the start of the adjustment layer.
    pub fn trim_start(&mut self, new_in: Position) {
        let old_end = self.timeline_out();
        self.timeline_in = new_in;
        let new_duration = old_end.value() - new_in.value();
        self.duration = Duration::new(new_duration.max(0));
    }

    /// Trims the end of the adjustment layer.
    pub fn trim_end(&mut self, new_out: Position) {
        let new_duration = new_out.value() - self.timeline_in.value();
        self.duration = Duration::new(new_duration.max(0));
    }

    /// Moves the adjustment layer to a new position.
    pub fn move_to(&mut self, new_position: Position) {
        self.timeline_in = new_position;
    }

    /// Splits this adjustment layer at the given position.
    ///
    /// Returns `(left, right)` halves.
    ///
    /// # Errors
    ///
    /// Returns error if the position is not within this adjustment layer.
    pub fn split_at(&self, position: Position) -> TimelineResult<(Self, Self)> {
        if !self.contains_position(position) {
            return Err(TimelineError::InvalidPosition(format!(
                "Position {position} not in adjustment layer range"
            )));
        }

        let left_duration = Duration::new(position.value() - self.timeline_in.value());
        let right_duration = Duration::new(self.timeline_out().value() - position.value());

        let mut left = self.clone();
        left.id = AdjustmentLayerId::new();
        left.duration = left_duration;

        let mut right = self.clone();
        right.id = AdjustmentLayerId::new();
        right.timeline_in = position;
        right.duration = right_duration;

        Ok((left, right))
    }
}

/// Processes adjustment layers during rendering.
///
/// The processor holds a collection of adjustment layers and provides
/// methods to apply them to rendered frames at any position.
#[derive(Clone, Debug, Default)]
pub struct AdjustmentLayerProcessor {
    /// All adjustment layers, sorted by timeline position.
    layers: Vec<AdjustmentLayer>,
}

impl AdjustmentLayerProcessor {
    /// Creates a new empty processor.
    #[must_use]
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    /// Adds an adjustment layer.
    pub fn add_layer(&mut self, layer: AdjustmentLayer) {
        self.layers.push(layer);
        self.layers.sort_by_key(|l| l.timeline_in.value());
    }

    /// Removes an adjustment layer by ID.
    ///
    /// # Errors
    ///
    /// Returns error if the layer is not found.
    pub fn remove_layer(&mut self, id: AdjustmentLayerId) -> TimelineResult<AdjustmentLayer> {
        let index = self
            .layers
            .iter()
            .position(|l| l.id == id)
            .ok_or_else(|| TimelineError::Other(format!("Adjustment layer {id} not found")))?;
        Ok(self.layers.remove(index))
    }

    /// Gets an adjustment layer by ID.
    #[must_use]
    pub fn get_layer(&self, id: AdjustmentLayerId) -> Option<&AdjustmentLayer> {
        self.layers.iter().find(|l| l.id == id)
    }

    /// Gets a mutable reference to an adjustment layer.
    pub fn get_layer_mut(&mut self, id: AdjustmentLayerId) -> Option<&mut AdjustmentLayer> {
        self.layers.iter_mut().find(|l| l.id == id)
    }

    /// Returns all active (enabled) adjustment layers at the given position.
    #[must_use]
    pub fn active_layers_at(&self, position: Position) -> Vec<&AdjustmentLayer> {
        self.layers
            .iter()
            .filter(|l| l.enabled && l.contains_position(position))
            .collect()
    }

    /// Returns the total number of adjustment layers.
    #[must_use]
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Returns `true` if there are no adjustment layers.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Applies all active adjustment layers to a frame at the given position.
    ///
    /// Each active layer's effects are applied based on their `EffectType`.
    /// Returns a new buffer with all adjustments applied.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn apply_to_frame(&self, input: &PixelBuffer, position: Position) -> PixelBuffer {
        let active = self.active_layers_at(position);
        if active.is_empty() {
            return input.clone();
        }

        let mut result = input.clone();

        for layer in &active {
            if layer.effects.is_empty() || layer.opacity <= 0.0 {
                continue;
            }

            let processed = self.apply_effect_stack(&result, &layer.effects, layer.affect_alpha);

            self.composite_layer(
                &mut result,
                &processed,
                layer.blend_mode,
                layer.opacity,
                layer.affect_alpha,
            );
        }

        result
    }

    /// Applies an effect stack to a pixel buffer.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn apply_effect_stack(
        &self,
        input: &PixelBuffer,
        effects: &EffectStack,
        affect_alpha: bool,
    ) -> PixelBuffer {
        let mut buf = input.clone();

        for effect in effects.effects() {
            match effect.effect_type {
                EffectType::BrightnessContrast => {
                    let offset: f32 = parse_effect_value(&effect.name).unwrap_or(10.0);
                    for i in (0..buf.data.len()).step_by(4) {
                        let channels = if affect_alpha { 4 } else { 3 };
                        for c in 0..channels {
                            let v = f32::from(buf.data[i + c]) + offset;
                            buf.data[i + c] = v.clamp(0.0, 255.0).round() as u8;
                        }
                    }
                }
                EffectType::ColorCorrection => {
                    // Invert operation (used for testing)
                    for i in (0..buf.data.len()).step_by(4) {
                        let channels = if affect_alpha { 4 } else { 3 };
                        for c in 0..channels {
                            buf.data[i + c] = 255 - buf.data[i + c];
                        }
                    }
                }
                EffectType::HueSaturation => {
                    let factor: f32 = parse_effect_value(&effect.name).unwrap_or(1.2);
                    for i in (0..buf.data.len()).step_by(4) {
                        let channels = if affect_alpha { 4 } else { 3 };
                        for c in 0..channels {
                            let v = (f32::from(buf.data[i + c]) - 128.0) * factor + 128.0;
                            buf.data[i + c] = v.clamp(0.0, 255.0).round() as u8;
                        }
                    }
                }
                _ => {
                    // Other effect types: pass through unchanged.
                }
            }
        }

        buf
    }

    /// Composites the processed buffer over the result buffer.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn composite_layer(
        &self,
        result: &mut PixelBuffer,
        processed: &PixelBuffer,
        blend_mode: AdjustmentBlendMode,
        opacity: f32,
        affect_alpha: bool,
    ) {
        let opacity = opacity.clamp(0.0, 1.0);
        let inv_opacity = 1.0 - opacity;

        for i in (0..result.data.len()).step_by(4) {
            let channels = if affect_alpha { 4 } else { 3 };
            for c in 0..channels {
                let base = result.data[i + c];
                let proc = processed.data[i + c];
                let blended = blend_mode.blend(base, proc);
                let final_val = f32::from(base) * inv_opacity + f32::from(blended) * opacity;
                result.data[i + c] = final_val.clamp(0.0, 255.0).round() as u8;
            }
        }
    }
}

/// Parses a numeric value from an effect name string.
fn parse_effect_value(name: &str) -> Option<f32> {
    for part in name.split_whitespace() {
        let clean = part.trim_start_matches('+');
        if let Ok(v) = clean.parse::<f32>() {
            return Some(v);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_layer(start: i64, duration: i64) -> AdjustmentLayer {
        AdjustmentLayer::new("Test Layer", Position::new(start), Duration::new(duration))
    }

    fn brightness_effect(offset: f32) -> Effect {
        Effect::new(
            format!("Brightness +{offset}"),
            EffectType::BrightnessContrast,
        )
    }

    fn invert_effect() -> Effect {
        Effect::new("Invert Colors".to_string(), EffectType::ColorCorrection)
    }

    fn contrast_effect(factor: f32) -> Effect {
        Effect::new(format!("Contrast {factor}"), EffectType::HueSaturation)
    }

    // --- AdjustmentLayerId tests ---

    #[test]
    fn test_adjustment_layer_id_unique() {
        let id1 = AdjustmentLayerId::new();
        let id2 = AdjustmentLayerId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_adjustment_layer_id_default() {
        let id = AdjustmentLayerId::default();
        let id2 = AdjustmentLayerId::default();
        assert_ne!(id, id2);
    }

    #[test]
    fn test_adjustment_layer_id_display() {
        let id = AdjustmentLayerId::new();
        let s = format!("{id}");
        assert!(!s.is_empty());
    }

    // --- AdjustmentBlendMode tests ---

    #[test]
    fn test_blend_mode_default() {
        assert_eq!(AdjustmentBlendMode::default(), AdjustmentBlendMode::Normal);
    }

    #[test]
    fn test_blend_normal() {
        assert_eq!(AdjustmentBlendMode::Normal.blend(100, 200), 200);
    }

    #[test]
    fn test_blend_multiply() {
        let result = AdjustmentBlendMode::Multiply.blend(100, 200);
        assert!(
            (result as i32 - 78).abs() <= 1,
            "Expected ~78, got {result}"
        );
    }

    #[test]
    fn test_blend_screen() {
        let result = AdjustmentBlendMode::Screen.blend(100, 200);
        assert!(result > 200, "Screen should brighten: got {result}");
    }

    #[test]
    fn test_blend_overlay_dark() {
        let result = AdjustmentBlendMode::Overlay.blend(50, 100);
        let expected = (2.0_f32 * 50.0 / 255.0 * 100.0 / 255.0 * 255.0).round() as u8;
        assert!((result as i32 - expected as i32).abs() <= 1);
    }

    #[test]
    fn test_blend_overlay_light() {
        let result = AdjustmentBlendMode::Overlay.blend(200, 180);
        assert!(result > 150, "Overlay light should be bright: got {result}");
    }

    #[test]
    fn test_blend_soft_light() {
        let result = AdjustmentBlendMode::SoftLight.blend(128, 128);
        assert!(
            (result as i32 - 128).abs() <= 5,
            "SoftLight midpoint: got {result}"
        );
    }

    // --- AdjustmentLayer tests ---

    #[test]
    fn test_adjustment_layer_creation() {
        let layer = make_layer(0, 100);
        assert_eq!(layer.name, "Test Layer");
        assert_eq!(layer.timeline_in, Position::new(0));
        assert_eq!(layer.duration, Duration::new(100));
        assert!((layer.opacity - 1.0).abs() < f32::EPSILON);
        assert!(layer.enabled);
        assert!(!layer.affect_alpha);
    }

    #[test]
    fn test_adjustment_layer_timeline_out() {
        let layer = make_layer(50, 100);
        assert_eq!(layer.timeline_out(), Position::new(150));
    }

    #[test]
    fn test_adjustment_layer_contains_position() {
        let layer = make_layer(10, 100);
        assert!(layer.contains_position(Position::new(10)));
        assert!(layer.contains_position(Position::new(50)));
        assert!(layer.contains_position(Position::new(109)));
        assert!(!layer.contains_position(Position::new(9)));
        assert!(!layer.contains_position(Position::new(110)));
    }

    #[test]
    fn test_adjustment_layer_overlaps() {
        let layer = make_layer(10, 100);
        assert!(layer.overlaps(Position::new(0), Position::new(50)));
        assert!(layer.overlaps(Position::new(50), Position::new(200)));
        assert!(!layer.overlaps(Position::new(110), Position::new(200)));
        assert!(!layer.overlaps(Position::new(0), Position::new(10)));
    }

    #[test]
    fn test_add_effect() {
        let mut layer = make_layer(0, 100);
        layer.add_effect(brightness_effect(20.0));
        assert_eq!(layer.effect_count(), 1);
    }

    #[test]
    fn test_set_opacity_valid() {
        let mut layer = make_layer(0, 100);
        assert!(layer.set_opacity(0.5).is_ok());
        assert!((layer.opacity - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_set_opacity_invalid() {
        let mut layer = make_layer(0, 100);
        assert!(layer.set_opacity(-0.1).is_err());
        assert!(layer.set_opacity(1.1).is_err());
    }

    #[test]
    fn test_set_blend_mode() {
        let mut layer = make_layer(0, 100);
        layer.set_blend_mode(AdjustmentBlendMode::Multiply);
        assert_eq!(layer.blend_mode, AdjustmentBlendMode::Multiply);
    }

    #[test]
    fn test_set_enabled() {
        let mut layer = make_layer(0, 100);
        layer.set_enabled(false);
        assert!(!layer.enabled);
        layer.set_enabled(true);
        assert!(layer.enabled);
    }

    #[test]
    fn test_trim_start() {
        let mut layer = make_layer(10, 100);
        layer.trim_start(Position::new(30));
        assert_eq!(layer.timeline_in, Position::new(30));
        assert_eq!(layer.duration, Duration::new(80));
    }

    #[test]
    fn test_trim_end() {
        let mut layer = make_layer(10, 100);
        layer.trim_end(Position::new(60));
        assert_eq!(layer.duration, Duration::new(50));
    }

    #[test]
    fn test_move_to() {
        let mut layer = make_layer(0, 100);
        layer.move_to(Position::new(50));
        assert_eq!(layer.timeline_in, Position::new(50));
    }

    #[test]
    fn test_split_at() {
        let mut layer = make_layer(0, 100);
        layer.add_effect(brightness_effect(10.0));

        let (left, right) = layer
            .split_at(Position::new(40))
            .expect("should succeed in test");

        assert_eq!(left.timeline_in, Position::new(0));
        assert_eq!(left.duration, Duration::new(40));
        assert_eq!(right.timeline_in, Position::new(40));
        assert_eq!(right.duration, Duration::new(60));
        assert_eq!(left.effect_count(), 1);
        assert_eq!(right.effect_count(), 1);
        assert_ne!(left.id, right.id);
        assert_ne!(left.id, layer.id);
    }

    #[test]
    fn test_split_at_invalid_position() {
        let layer = make_layer(0, 100);
        assert!(layer.split_at(Position::new(200)).is_err());
    }

    // --- AdjustmentLayerProcessor tests ---

    #[test]
    fn test_processor_empty() {
        let proc = AdjustmentLayerProcessor::new();
        assert!(proc.is_empty());
        assert_eq!(proc.layer_count(), 0);
    }

    #[test]
    fn test_processor_add_layer() {
        let mut proc = AdjustmentLayerProcessor::new();
        proc.add_layer(make_layer(0, 100));
        assert_eq!(proc.layer_count(), 1);
        assert!(!proc.is_empty());
    }

    #[test]
    fn test_processor_remove_layer() {
        let mut proc = AdjustmentLayerProcessor::new();
        let layer = make_layer(0, 100);
        let id = layer.id;
        proc.add_layer(layer);
        assert!(proc.remove_layer(id).is_ok());
        assert!(proc.is_empty());
    }

    #[test]
    fn test_processor_remove_nonexistent() {
        let mut proc = AdjustmentLayerProcessor::new();
        assert!(proc.remove_layer(AdjustmentLayerId::new()).is_err());
    }

    #[test]
    fn test_processor_get_layer() {
        let mut proc = AdjustmentLayerProcessor::new();
        let layer = make_layer(0, 100);
        let id = layer.id;
        proc.add_layer(layer);
        assert!(proc.get_layer(id).is_some());
        assert!(proc.get_layer(AdjustmentLayerId::new()).is_none());
    }

    #[test]
    fn test_processor_active_layers_at() {
        let mut proc = AdjustmentLayerProcessor::new();
        proc.add_layer(make_layer(0, 100));
        proc.add_layer(make_layer(50, 100));

        let active = proc.active_layers_at(Position::new(75));
        assert_eq!(active.len(), 2);

        let active = proc.active_layers_at(Position::new(25));
        assert_eq!(active.len(), 1);

        let active = proc.active_layers_at(Position::new(200));
        assert_eq!(active.len(), 0);
    }

    #[test]
    fn test_processor_active_layers_skips_disabled() {
        let mut proc = AdjustmentLayerProcessor::new();
        let mut layer = make_layer(0, 100);
        layer.set_enabled(false);
        proc.add_layer(layer);

        let active = proc.active_layers_at(Position::new(50));
        assert_eq!(active.len(), 0);
    }

    #[test]
    fn test_processor_apply_no_layers() {
        let proc = AdjustmentLayerProcessor::new();
        let input = PixelBuffer::solid(4, 4, [128, 128, 128, 255]);
        let result = proc.apply_to_frame(&input, Position::new(50));
        assert_eq!(result.data[0], 128);
    }

    #[test]
    fn test_processor_apply_brightness() {
        let mut proc = AdjustmentLayerProcessor::new();
        let mut layer = make_layer(0, 100);
        layer.add_effect(brightness_effect(20.0));
        proc.add_layer(layer);

        let input = PixelBuffer::solid(4, 4, [100, 100, 100, 255]);
        let result = proc.apply_to_frame(&input, Position::new(50));
        assert_eq!(result.data[0], 120);
        assert_eq!(result.data[1], 120);
        assert_eq!(result.data[2], 120);
        assert_eq!(result.data[3], 255);
    }

    #[test]
    fn test_processor_apply_invert() {
        let mut proc = AdjustmentLayerProcessor::new();
        let mut layer = make_layer(0, 100);
        layer.add_effect(invert_effect());
        proc.add_layer(layer);

        let input = PixelBuffer::solid(2, 2, [200, 100, 50, 255]);
        let result = proc.apply_to_frame(&input, Position::new(50));
        assert_eq!(result.data[0], 55);
        assert_eq!(result.data[1], 155);
        assert_eq!(result.data[2], 205);
        assert_eq!(result.data[3], 255);
    }

    #[test]
    fn test_processor_apply_contrast() {
        let mut proc = AdjustmentLayerProcessor::new();
        let mut layer = make_layer(0, 100);
        layer.add_effect(contrast_effect(2.0));
        proc.add_layer(layer);

        let input = PixelBuffer::solid(2, 2, [128, 128, 128, 255]);
        let result = proc.apply_to_frame(&input, Position::new(50));
        assert_eq!(result.data[0], 128);
    }

    #[test]
    fn test_processor_apply_with_opacity() {
        let mut proc = AdjustmentLayerProcessor::new();
        let mut layer = make_layer(0, 100);
        layer.add_effect(brightness_effect(100.0));
        layer.set_opacity(0.5).expect("should succeed in test");
        proc.add_layer(layer);

        let input = PixelBuffer::solid(2, 2, [100, 100, 100, 255]);
        let result = proc.apply_to_frame(&input, Position::new(50));
        assert!(
            (result.data[0] as i32 - 150).abs() <= 1,
            "Expected ~150, got {}",
            result.data[0]
        );
    }

    #[test]
    fn test_processor_apply_outside_range() {
        let mut proc = AdjustmentLayerProcessor::new();
        let mut layer = make_layer(0, 100);
        layer.add_effect(brightness_effect(50.0));
        proc.add_layer(layer);

        let input = PixelBuffer::solid(2, 2, [100, 100, 100, 255]);
        let result = proc.apply_to_frame(&input, Position::new(200));
        assert_eq!(result.data[0], 100);
    }

    #[test]
    fn test_processor_apply_multiply_blend() {
        let mut proc = AdjustmentLayerProcessor::new();
        let mut layer = make_layer(0, 100);
        layer.add_effect(brightness_effect(0.0));
        layer.set_blend_mode(AdjustmentBlendMode::Multiply);
        proc.add_layer(layer);

        let input = PixelBuffer::solid(2, 2, [200, 200, 200, 255]);
        let result = proc.apply_to_frame(&input, Position::new(50));
        assert!(
            result.data[0] < 200,
            "Multiply should darken: got {}",
            result.data[0]
        );
    }

    #[test]
    fn test_processor_layers_sorted() {
        let mut proc = AdjustmentLayerProcessor::new();
        proc.add_layer(make_layer(100, 50));
        proc.add_layer(make_layer(0, 50));
        proc.add_layer(make_layer(50, 50));

        let all: Vec<i64> = proc.layers.iter().map(|l| l.timeline_in.value()).collect();
        assert_eq!(all, vec![0, 50, 100]);
    }

    #[test]
    fn test_parse_effect_value_positive() {
        assert!(
            (parse_effect_value("Brightness +20").expect("should parse") - 20.0).abs()
                < f32::EPSILON
        );
    }

    #[test]
    fn test_parse_effect_value_decimal() {
        assert!(
            (parse_effect_value("Contrast 1.5").expect("should parse") - 1.5).abs() < f32::EPSILON
        );
    }

    #[test]
    fn test_parse_effect_value_negative() {
        assert!(
            (parse_effect_value("Level -10").expect("should parse") - (-10.0)).abs() < f32::EPSILON
        );
    }

    #[test]
    fn test_parse_effect_value_none() {
        assert!(parse_effect_value("Invert Colors").is_none());
    }

    #[test]
    fn test_processor_get_layer_mut() {
        let mut proc = AdjustmentLayerProcessor::new();
        let layer = make_layer(0, 100);
        let id = layer.id;
        proc.add_layer(layer);

        let layer_mut = proc.get_layer_mut(id).expect("should find");
        layer_mut.name = "Modified".to_string();
        assert_eq!(proc.get_layer(id).expect("should find").name, "Modified");
    }

    #[test]
    fn test_stacked_effects() {
        let mut proc = AdjustmentLayerProcessor::new();
        let mut layer = make_layer(0, 100);
        layer.add_effect(brightness_effect(50.0));
        layer.add_effect(invert_effect());
        proc.add_layer(layer);

        let input = PixelBuffer::solid(2, 2, [100, 100, 100, 255]);
        let result = proc.apply_to_frame(&input, Position::new(50));
        // brightness +50: 100 -> 150
        // invert: 150 -> 105
        assert_eq!(result.data[0], 105);
    }
}
