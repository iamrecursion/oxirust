//! Effect system with keyframe animation.
//!
//! Effects can be applied to clips and animated using keyframes.

#![allow(missing_docs)]

use oximedia_core::Rational;
use oximedia_graph::FilterGraph;
use std::collections::BTreeMap;

use crate::error::{EditError, EditResult};

/// Stack of effects applied to a clip.
#[derive(Clone, Debug, Default)]
pub struct EffectStack {
    /// Effects in the stack (applied in order).
    pub effects: Vec<Effect>,
}

impl EffectStack {
    /// Create a new empty effect stack.
    #[must_use]
    pub fn new() -> Self {
        Self {
            effects: Vec::new(),
        }
    }

    /// Add an effect to the stack.
    pub fn add(&mut self, effect: Effect) {
        self.effects.push(effect);
    }

    /// Remove an effect by index.
    pub fn remove(&mut self, index: usize) -> Option<Effect> {
        if index < self.effects.len() {
            Some(self.effects.remove(index))
        } else {
            None
        }
    }

    /// Get an effect by index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&Effect> {
        self.effects.get(index)
    }

    /// Get mutable effect by index.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut Effect> {
        self.effects.get_mut(index)
    }

    /// Get number of effects.
    #[must_use]
    pub fn len(&self) -> usize {
        self.effects.len()
    }

    /// Check if stack is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.effects.is_empty()
    }

    /// Clear all effects.
    pub fn clear(&mut self) {
        self.effects.clear();
    }

    /// Evaluate all effects at a given time.
    pub fn evaluate_at(&self, time: i64, timebase: Rational) -> EditResult<Vec<EffectInstance>> {
        self.effects
            .iter()
            .map(|effect| effect.evaluate_at(time, timebase))
            .collect()
    }
}

/// An effect that can be applied to a clip.
#[derive(Clone, Debug)]
pub struct Effect {
    /// Effect type.
    pub effect_type: EffectType,
    /// Effect parameters with keyframe support.
    pub parameters: BTreeMap<String, Parameter>,
    /// Effect is enabled.
    pub enabled: bool,
    /// Effect name (user-defined).
    pub name: Option<String>,
}

impl Effect {
    /// Create a new effect.
    #[must_use]
    pub fn new(effect_type: EffectType) -> Self {
        Self {
            effect_type,
            parameters: BTreeMap::new(),
            enabled: true,
            name: None,
        }
    }

    /// Set a parameter value.
    pub fn set_parameter(&mut self, name: String, parameter: Parameter) {
        self.parameters.insert(name, parameter);
    }

    /// Get a parameter value.
    #[must_use]
    pub fn get_parameter(&self, name: &str) -> Option<&Parameter> {
        self.parameters.get(name)
    }

    /// Evaluate effect at a specific time.
    pub fn evaluate_at(&self, time: i64, timebase: Rational) -> EditResult<EffectInstance> {
        let mut values = BTreeMap::new();

        for (name, param) in &self.parameters {
            let value = param.evaluate_at(time, timebase)?;
            values.insert(name.clone(), value);
        }

        Ok(EffectInstance {
            effect_type: self.effect_type.clone(),
            values,
        })
    }
}

/// Effect type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EffectType {
    /// Brightness adjustment.
    Brightness,
    /// Contrast adjustment.
    Contrast,
    /// Saturation adjustment.
    Saturation,
    /// Hue rotation.
    Hue,
    /// Blur effect.
    Blur,
    /// Sharpen effect.
    Sharpen,
    /// Color correction (curves).
    ColorCurve,
    /// Crop.
    Crop,
    /// Scale/resize.
    Scale,
    /// Rotate.
    Rotate,
    /// Position/translate.
    Position,
    /// Opacity/transparency.
    Opacity,
    /// Audio gain.
    AudioGain,
    /// Audio pan.
    AudioPan,
    /// Audio equalizer.
    AudioEq,
    /// Audio reverb.
    AudioReverb,
    /// Chroma key (green screen).
    ChromaKey,
    /// Custom filter graph.
    Custom(String),
}

/// Effect parameter with keyframe support.
#[derive(Clone, Debug)]
pub struct Parameter {
    /// Parameter keyframes.
    pub keyframes: BTreeMap<i64, ParameterValue>,
    /// Interpolation mode.
    pub interpolation: InterpolationMode,
}

impl Parameter {
    /// Create a new parameter with a constant value.
    #[must_use]
    pub fn constant(value: ParameterValue) -> Self {
        let mut keyframes = BTreeMap::new();
        keyframes.insert(0, value);
        Self {
            keyframes,
            interpolation: InterpolationMode::Linear,
        }
    }

    /// Add a keyframe.
    pub fn add_keyframe(&mut self, time: i64, value: ParameterValue) {
        self.keyframes.insert(time, value);
    }

    /// Remove a keyframe.
    pub fn remove_keyframe(&mut self, time: i64) -> Option<ParameterValue> {
        self.keyframes.remove(&time)
    }

    /// Evaluate parameter at a specific time.
    pub fn evaluate_at(&self, time: i64, _timebase: Rational) -> EditResult<ParameterValue> {
        if self.keyframes.is_empty() {
            return Err(EditError::KeyframeError("No keyframes defined".to_string()));
        }

        // If there's an exact keyframe, return it
        if let Some(value) = self.keyframes.get(&time) {
            return Ok(value.clone());
        }

        // Find surrounding keyframes for interpolation
        let before = self
            .keyframes
            .range(..time)
            .next_back()
            .map(|(t, v)| (*t, v.clone()));
        let after = self
            .keyframes
            .range(time..)
            .next()
            .map(|(t, v)| (*t, v.clone()));

        match (before, after) {
            (Some((t1, v1)), Some((t2, v2))) => {
                // Interpolate between keyframes
                let factor = if t2 == t1 {
                    0.0
                } else {
                    #[allow(clippy::cast_precision_loss)]
                    let result = (time - t1) as f64 / (t2 - t1) as f64;
                    result
                };
                Ok(self.interpolation.interpolate(&v1, &v2, factor))
            }
            (Some((_, v)), None) | (None, Some((_, v))) => {
                // Use the closest keyframe
                Ok(v)
            }
            (None, None) => Err(EditError::KeyframeError("No keyframes found".to_string())),
        }
    }

    /// Get all keyframe times.
    #[must_use]
    pub fn keyframe_times(&self) -> Vec<i64> {
        self.keyframes.keys().copied().collect()
    }
}

/// Parameter value type.
#[derive(Clone, Debug)]
pub enum ParameterValue {
    /// Floating point value.
    Float(f64),
    /// Integer value.
    Int(i64),
    /// Boolean value.
    Bool(bool),
    /// String value.
    String(String),
    /// 2D point.
    Point2D { x: f64, y: f64 },
    /// 3D point.
    Point3D { x: f64, y: f64, z: f64 },
    /// Color (RGBA).
    Color { r: f32, g: f32, b: f32, a: f32 },
}

/// Keyframe interpolation mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterpolationMode {
    /// No interpolation (step/hold).
    Constant,
    /// Linear interpolation.
    Linear,
    /// Smooth interpolation (ease in/out).
    Smooth,
    /// Bezier curve interpolation.
    Bezier,
}

impl InterpolationMode {
    /// Interpolate between two values.
    #[must_use]
    #[allow(clippy::needless_pass_by_value)]
    pub fn interpolate(
        &self,
        v1: &ParameterValue,
        v2: &ParameterValue,
        factor: f64,
    ) -> ParameterValue {
        let t = match self {
            Self::Constant => 0.0,
            Self::Linear => factor,
            Self::Smooth => {
                // Smooth step (ease in/out)
                factor * factor * (3.0 - 2.0 * factor)
            }
            Self::Bezier => {
                // Simple bezier approximation
                let t = factor;
                t * t * (3.0 - 2.0 * t)
            }
        };

        match (v1, v2) {
            (ParameterValue::Float(a), ParameterValue::Float(b)) => {
                ParameterValue::Float(a + (b - a) * t)
            }
            (ParameterValue::Int(a), ParameterValue::Int(b)) => {
                #[allow(clippy::cast_possible_truncation)]
                let result = a + ((b - a) as f64 * t) as i64;
                ParameterValue::Int(result)
            }
            (ParameterValue::Bool(a), ParameterValue::Bool(_)) => {
                // No interpolation for booleans
                ParameterValue::Bool(*a)
            }
            (ParameterValue::String(s), _) => {
                // No interpolation for strings
                ParameterValue::String(s.clone())
            }
            (
                ParameterValue::Point2D { x: x1, y: y1 },
                ParameterValue::Point2D { x: x2, y: y2 },
            ) => ParameterValue::Point2D {
                x: x1 + (x2 - x1) * t,
                y: y1 + (y2 - y1) * t,
            },
            (
                ParameterValue::Point3D {
                    x: x1,
                    y: y1,
                    z: z1,
                },
                ParameterValue::Point3D {
                    x: x2,
                    y: y2,
                    z: z2,
                },
            ) => ParameterValue::Point3D {
                x: x1 + (x2 - x1) * t,
                y: y1 + (y2 - y1) * t,
                z: z1 + (z2 - z1) * t,
            },
            (
                ParameterValue::Color {
                    r: r1,
                    g: g1,
                    b: b1,
                    a: a1,
                },
                ParameterValue::Color {
                    r: r2,
                    g: g2,
                    b: b2,
                    a: a2,
                },
            ) => {
                #[allow(clippy::cast_possible_truncation)]
                let result = ParameterValue::Color {
                    r: r1 + (r2 - r1) * t as f32,
                    g: g1 + (g2 - g1) * t as f32,
                    b: b1 + (b2 - b1) * t as f32,
                    a: a1 + (a2 - a1) * t as f32,
                };
                result
            }
            _ => {
                // Type mismatch, return first value
                v1.clone()
            }
        }
    }
}

/// Evaluated effect instance at a specific time.
#[derive(Clone, Debug)]
pub struct EffectInstance {
    /// Effect type.
    pub effect_type: EffectType,
    /// Evaluated parameter values.
    pub values: BTreeMap<String, ParameterValue>,
}

impl EffectInstance {
    /// Get a float parameter value.
    #[must_use]
    pub fn get_float(&self, name: &str) -> Option<f64> {
        match self.values.get(name) {
            Some(ParameterValue::Float(v)) => Some(*v),
            _ => None,
        }
    }

    /// Get an integer parameter value.
    #[must_use]
    pub fn get_int(&self, name: &str) -> Option<i64> {
        match self.values.get(name) {
            Some(ParameterValue::Int(v)) => Some(*v),
            _ => None,
        }
    }

    /// Get a boolean parameter value.
    #[must_use]
    pub fn get_bool(&self, name: &str) -> Option<bool> {
        match self.values.get(name) {
            Some(ParameterValue::Bool(v)) => Some(*v),
            _ => None,
        }
    }

    /// Get a 2D point parameter value.
    #[must_use]
    pub fn get_point2d(&self, name: &str) -> Option<(f64, f64)> {
        match self.values.get(name) {
            Some(ParameterValue::Point2D { x, y }) => Some((*x, *y)),
            _ => None,
        }
    }
}

/// Effect preset for common effect configurations.
#[derive(Clone, Debug)]
pub struct EffectPreset {
    /// Preset name.
    pub name: String,
    /// Preset description.
    pub description: String,
    /// Effect configuration.
    pub effect: Effect,
}

impl EffectPreset {
    /// Create a new preset.
    #[must_use]
    pub fn new(name: String, description: String, effect: Effect) -> Self {
        Self {
            name,
            description,
            effect,
        }
    }

    /// Create a brightness preset.
    #[must_use]
    pub fn brightness(value: f64) -> Self {
        let mut effect = Effect::new(EffectType::Brightness);
        effect.set_parameter(
            "brightness".to_string(),
            Parameter::constant(ParameterValue::Float(value)),
        );
        Self::new(
            "Brightness".to_string(),
            format!("Adjust brightness to {value}"),
            effect,
        )
    }

    /// Create a blur preset.
    #[must_use]
    pub fn blur(radius: f64) -> Self {
        let mut effect = Effect::new(EffectType::Blur);
        effect.set_parameter(
            "radius".to_string(),
            Parameter::constant(ParameterValue::Float(radius)),
        );
        Self::new(
            "Blur".to_string(),
            format!("Apply blur with radius {radius}"),
            effect,
        )
    }

    /// Create a fade in preset.
    #[must_use]
    pub fn fade_in(duration: i64) -> Self {
        let mut effect = Effect::new(EffectType::Opacity);
        let mut parameter = Parameter::constant(ParameterValue::Float(0.0));
        parameter.add_keyframe(0, ParameterValue::Float(0.0));
        parameter.add_keyframe(duration, ParameterValue::Float(1.0));
        parameter.interpolation = InterpolationMode::Linear;
        effect.set_parameter("opacity".to_string(), parameter);
        Self::new(
            "Fade In".to_string(),
            format!("Fade in over {duration} frames"),
            effect,
        )
    }

    /// Create a fade out preset.
    #[must_use]
    pub fn fade_out(duration: i64) -> Self {
        let mut effect = Effect::new(EffectType::Opacity);
        let mut parameter = Parameter::constant(ParameterValue::Float(1.0));
        parameter.add_keyframe(0, ParameterValue::Float(1.0));
        parameter.add_keyframe(duration, ParameterValue::Float(0.0));
        parameter.interpolation = InterpolationMode::Linear;
        effect.set_parameter("opacity".to_string(), parameter);
        Self::new(
            "Fade Out".to_string(),
            format!("Fade out over {duration} frames"),
            effect,
        )
    }
}

/// Builder for creating filter graphs from effects.
#[derive(Debug, Default)]
pub struct EffectGraphBuilder;

impl EffectGraphBuilder {
    /// Create a new effect graph builder.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Build a filter graph from an effect stack.
    pub fn build(&self, _effects: &EffectStack) -> EditResult<FilterGraph> {
        // This would build an actual filter graph from the effect stack
        // For now, return a default graph
        Ok(FilterGraph::new())
    }
}
