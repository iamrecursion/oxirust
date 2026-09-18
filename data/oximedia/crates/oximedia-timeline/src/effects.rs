//! Effects system for clips and tracks.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::keyframe::Keyframe;

/// Unique identifier for an effect.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EffectId(Uuid);

impl EffectId {
    /// Creates a new random effect ID.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for EffectId {
    fn default() -> Self {
        Self::new()
    }
}

/// Type of effect.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EffectType {
    /// Color correction.
    ColorCorrection,
    /// Brightness/contrast.
    BrightnessContrast,
    /// Hue/saturation.
    HueSaturation,
    /// Blur.
    Blur,
    /// Sharpen.
    Sharpen,
    /// Scale/transform.
    Transform,
    /// Crop.
    Crop,
    /// Opacity.
    Opacity,
    /// Audio gain.
    AudioGain,
    /// Audio EQ.
    AudioEq,
    /// Audio compression.
    AudioCompressor,
    /// Audio reverb.
    AudioReverb,
    /// Audio delay.
    AudioDelay,
    /// Custom effect.
    Custom(String),
}

/// An effect applied to a clip or track.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Effect {
    /// Unique identifier.
    pub id: EffectId,
    /// Name of the effect.
    pub name: String,
    /// Type of effect.
    pub effect_type: EffectType,
    /// Whether effect is enabled.
    pub enabled: bool,
    /// Effect parameters.
    pub parameters: std::collections::HashMap<String, EffectParameter>,
    /// Keyframes for animated parameters.
    pub keyframes: Vec<Keyframe>,
}

impl Effect {
    /// Creates a new effect.
    #[must_use]
    pub fn new(name: String, effect_type: EffectType) -> Self {
        Self {
            id: EffectId::new(),
            name,
            effect_type,
            enabled: true,
            parameters: std::collections::HashMap::new(),
            keyframes: Vec::new(),
        }
    }

    /// Adds a parameter to the effect.
    pub fn add_parameter(&mut self, name: String, parameter: EffectParameter) {
        self.parameters.insert(name, parameter);
    }

    /// Gets a parameter value.
    #[must_use]
    pub fn get_parameter(&self, name: &str) -> Option<&EffectParameter> {
        self.parameters.get(name)
    }

    /// Sets a parameter value.
    pub fn set_parameter(&mut self, name: String, parameter: EffectParameter) {
        self.parameters.insert(name, parameter);
    }

    /// Adds a keyframe.
    pub fn add_keyframe(&mut self, keyframe: Keyframe) {
        self.keyframes.push(keyframe);
        self.keyframes.sort_by_key(|k| k.position.value());
    }

    /// Enables or disables the effect.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

/// Effect parameter value.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum EffectParameter {
    /// Float value.
    Float(f64),
    /// Integer value.
    Int(i64),
    /// Boolean value.
    Bool(bool),
    /// String value.
    String(String),
    /// Color (RGBA 0.0-1.0).
    Color([f32; 4]),
    /// 2D point.
    Point2D([f64; 2]),
}

impl EffectParameter {
    /// Creates a float parameter.
    #[must_use]
    pub const fn float(value: f64) -> Self {
        Self::Float(value)
    }

    /// Creates an integer parameter.
    #[must_use]
    pub const fn int(value: i64) -> Self {
        Self::Int(value)
    }

    /// Creates a boolean parameter.
    #[must_use]
    pub const fn bool(value: bool) -> Self {
        Self::Bool(value)
    }

    /// Creates a string parameter.
    #[must_use]
    pub fn string(value: String) -> Self {
        Self::String(value)
    }

    /// Creates a color parameter.
    #[must_use]
    pub const fn color(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self::Color([r, g, b, a])
    }

    /// Creates a 2D point parameter.
    #[must_use]
    pub const fn point2d(x: f64, y: f64) -> Self {
        Self::Point2D([x, y])
    }
}

/// Stack of effects applied in order.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EffectStack {
    /// Effects in the stack (applied in order).
    effects: Vec<Effect>,
}

impl EffectStack {
    /// Creates a new empty effect stack.
    #[must_use]
    pub fn new() -> Self {
        Self {
            effects: Vec::new(),
        }
    }

    /// Adds an effect to the stack.
    pub fn add_effect(&mut self, effect: Effect) {
        self.effects.push(effect);
    }

    /// Removes an effect by ID.
    ///
    /// # Errors
    ///
    /// Returns error if effect not found.
    pub fn remove_effect(&mut self, effect_id: EffectId) -> crate::error::TimelineResult<Effect> {
        let index = self
            .effects
            .iter()
            .position(|e| e.id == effect_id)
            .ok_or_else(|| crate::error::TimelineError::EffectNotFound(format!("{effect_id:?}")))?;
        Ok(self.effects.remove(index))
    }

    /// Gets an effect by ID.
    #[must_use]
    pub fn get_effect(&self, effect_id: EffectId) -> Option<&Effect> {
        self.effects.iter().find(|e| e.id == effect_id)
    }

    /// Gets a mutable reference to an effect by ID.
    pub fn get_effect_mut(&mut self, effect_id: EffectId) -> Option<&mut Effect> {
        self.effects.iter_mut().find(|e| e.id == effect_id)
    }

    /// Returns all effects in the stack.
    #[must_use]
    pub fn effects(&self) -> &[Effect] {
        &self.effects
    }

    /// Returns the number of effects.
    #[must_use]
    pub fn len(&self) -> usize {
        self.effects.len()
    }

    /// Checks if the stack is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.effects.is_empty()
    }

    /// Clears all effects.
    pub fn clear(&mut self) {
        self.effects.clear();
    }

    /// Moves an effect to a new position in the stack.
    ///
    /// # Errors
    ///
    /// Returns error if effect not found or index out of bounds.
    pub fn move_effect(
        &mut self,
        effect_id: EffectId,
        new_index: usize,
    ) -> crate::error::TimelineResult<()> {
        let old_index = self
            .effects
            .iter()
            .position(|e| e.id == effect_id)
            .ok_or_else(|| crate::error::TimelineError::EffectNotFound(format!("{effect_id:?}")))?;

        if new_index >= self.effects.len() {
            return Err(crate::error::TimelineError::Other(
                "New index out of bounds".to_string(),
            ));
        }

        let effect = self.effects.remove(old_index);
        self.effects.insert(new_index, effect);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effect_id_creation() {
        let id1 = EffectId::new();
        let id2 = EffectId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_effect_creation() {
        let effect = Effect::new("Blur".to_string(), EffectType::Blur);
        assert_eq!(effect.name, "Blur");
        assert_eq!(effect.effect_type, EffectType::Blur);
        assert!(effect.enabled);
    }

    #[test]
    fn test_effect_add_parameter() {
        let mut effect = Effect::new("Blur".to_string(), EffectType::Blur);
        effect.add_parameter("radius".to_string(), EffectParameter::float(5.0));
        assert!(effect.get_parameter("radius").is_some());
    }

    #[test]
    fn test_effect_set_parameter() {
        let mut effect = Effect::new("Blur".to_string(), EffectType::Blur);
        effect.set_parameter("radius".to_string(), EffectParameter::float(5.0));
        assert_eq!(
            effect.get_parameter("radius"),
            Some(&EffectParameter::float(5.0))
        );
    }

    #[test]
    fn test_effect_parameter_types() {
        let float = EffectParameter::float(1.5);
        let int = EffectParameter::int(42);
        let bool = EffectParameter::bool(true);
        let string = EffectParameter::string("test".to_string());
        let color = EffectParameter::color(1.0, 0.0, 0.0, 1.0);
        let point = EffectParameter::point2d(10.0, 20.0);

        assert!(matches!(float, EffectParameter::Float(_)));
        assert!(matches!(int, EffectParameter::Int(_)));
        assert!(matches!(bool, EffectParameter::Bool(_)));
        assert!(matches!(string, EffectParameter::String(_)));
        assert!(matches!(color, EffectParameter::Color(_)));
        assert!(matches!(point, EffectParameter::Point2D(_)));
    }

    #[test]
    fn test_effect_stack_add() {
        let mut stack = EffectStack::new();
        let effect = Effect::new("Blur".to_string(), EffectType::Blur);
        stack.add_effect(effect);
        assert_eq!(stack.len(), 1);
        assert!(!stack.is_empty());
    }

    #[test]
    fn test_effect_stack_remove() {
        let mut stack = EffectStack::new();
        let effect = Effect::new("Blur".to_string(), EffectType::Blur);
        let effect_id = effect.id;
        stack.add_effect(effect);
        assert!(stack.remove_effect(effect_id).is_ok());
        assert!(stack.is_empty());
    }

    #[test]
    fn test_effect_stack_get() {
        let mut stack = EffectStack::new();
        let effect = Effect::new("Blur".to_string(), EffectType::Blur);
        let effect_id = effect.id;
        stack.add_effect(effect);
        assert!(stack.get_effect(effect_id).is_some());
    }

    #[test]
    fn test_effect_stack_clear() {
        let mut stack = EffectStack::new();
        stack.add_effect(Effect::new("Blur".to_string(), EffectType::Blur));
        stack.add_effect(Effect::new("Sharpen".to_string(), EffectType::Sharpen));
        assert_eq!(stack.len(), 2);
        stack.clear();
        assert!(stack.is_empty());
    }

    #[test]
    fn test_effect_stack_move() {
        let mut stack = EffectStack::new();
        let effect1 = Effect::new("Effect 1".to_string(), EffectType::Blur);
        let effect2 = Effect::new("Effect 2".to_string(), EffectType::Sharpen);
        let effect1_id = effect1.id;

        stack.add_effect(effect1);
        stack.add_effect(effect2);

        assert!(stack.move_effect(effect1_id, 1).is_ok());
        assert_eq!(stack.effects()[1].id, effect1_id);
    }

    #[test]
    fn test_effect_enable_disable() {
        let mut effect = Effect::new("Blur".to_string(), EffectType::Blur);
        assert!(effect.enabled);
        effect.set_enabled(false);
        assert!(!effect.enabled);
    }
}
