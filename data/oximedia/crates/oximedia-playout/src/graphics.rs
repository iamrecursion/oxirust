//! Graphics overlay engine
//!
//! Provides logo/bug insertion, lower thirds, character generator,
//! ticker/crawler, alpha blending, and animation support.

use crate::{PlayoutError, Result};
use chrono::Utc;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use uuid::Uuid;

/// Graphics engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphicsConfig {
    /// Maximum number of simultaneous layers
    pub max_layers: usize,

    /// Enable hardware acceleration
    pub hardware_accel: bool,

    /// Default font
    pub default_font: String,

    /// Font size
    pub default_font_size: u32,

    /// Assets directory
    pub assets_dir: PathBuf,
}

impl Default for GraphicsConfig {
    fn default() -> Self {
        Self {
            max_layers: 10,
            hardware_accel: true,
            default_font: "Arial".to_string(),
            default_font_size: 48,
            assets_dir: PathBuf::from("/var/oximedia/assets"),
        }
    }
}

/// Graphics layer type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerType {
    /// Static logo/bug
    Logo,
    /// Lower third (name/title)
    LowerThird,
    /// Full-screen character generator
    CG,
    /// Ticker/crawler
    Ticker,
    /// Static image
    Image,
    /// Text overlay
    Text,
    /// Custom graphics
    Custom,
}

/// Position on screen
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Position {
    /// X coordinate (0.0 - 1.0, normalized)
    pub x: f32,
    /// Y coordinate (0.0 - 1.0, normalized)
    pub y: f32,
}

impl Position {
    /// Create new position
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Top-left corner
    pub fn top_left() -> Self {
        Self { x: 0.0, y: 0.0 }
    }

    /// Top-right corner
    pub fn top_right() -> Self {
        Self { x: 1.0, y: 0.0 }
    }

    /// Bottom-left corner
    pub fn bottom_left() -> Self {
        Self { x: 0.0, y: 1.0 }
    }

    /// Bottom-right corner
    pub fn bottom_right() -> Self {
        Self { x: 1.0, y: 1.0 }
    }

    /// Center
    pub fn center() -> Self {
        Self { x: 0.5, y: 0.5 }
    }
}

/// Size specification
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Size {
    /// Width (0.0 - 1.0, normalized, or pixels if > 1.0)
    pub width: f32,
    /// Height (0.0 - 1.0, normalized, or pixels if > 1.0)
    pub height: f32,
}

impl Size {
    /// Create new size
    pub fn new(width: f32, height: f32) -> Self {
        Self { width, height }
    }
}

/// Color in RGBA format
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    /// Create new color
    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// White color
    pub fn white() -> Self {
        Self::new(255, 255, 255, 255)
    }

    /// Black color
    pub fn black() -> Self {
        Self::new(0, 0, 0, 255)
    }

    /// Transparent
    pub fn transparent() -> Self {
        Self::new(0, 0, 0, 0)
    }

    /// Red
    pub fn red() -> Self {
        Self::new(255, 0, 0, 255)
    }

    /// Green
    pub fn green() -> Self {
        Self::new(0, 255, 0, 255)
    }

    /// Blue
    pub fn blue() -> Self {
        Self::new(0, 0, 255, 255)
    }
}

/// Animation curve
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnimationCurve {
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut,
    Bounce,
}

/// Animation type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Animation {
    /// Fade in/out
    Fade {
        from: f32,
        to: f32,
        duration_frames: u32,
        curve: AnimationCurve,
    },
    /// Move from one position to another
    Move {
        from: Position,
        to: Position,
        duration_frames: u32,
        curve: AnimationCurve,
    },
    /// Scale
    Scale {
        from: f32,
        to: f32,
        duration_frames: u32,
        curve: AnimationCurve,
    },
    /// Rotate
    Rotate {
        from: f32,
        to: f32,
        duration_frames: u32,
        curve: AnimationCurve,
    },
    /// Scroll (for ticker)
    Scroll {
        direction: ScrollDirection,
        speed: f32,
    },
}

/// Scroll direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScrollDirection {
    Left,
    Right,
    Up,
    Down,
}

/// Text alignment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TextAlign {
    Left,
    Center,
    Right,
    Justify,
}

/// Graphics layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphicsLayer {
    /// Unique layer ID
    pub id: Uuid,

    /// Layer name
    pub name: String,

    /// Layer type
    pub layer_type: LayerType,

    /// Position on screen
    pub position: Position,

    /// Size
    pub size: Option<Size>,

    /// Z-order (higher values are on top)
    pub z_order: i32,

    /// Opacity (0.0 - 1.0)
    pub opacity: f32,

    /// Visible flag
    pub visible: bool,

    /// Content data
    pub content: LayerContent,

    /// Active animations
    pub animations: Vec<Animation>,

    /// Current animation frame
    pub animation_frame: u32,
}

/// Layer content data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayerContent {
    /// Logo from file
    Logo { path: PathBuf },

    /// Lower third
    LowerThird {
        title: String,
        subtitle: String,
        background_color: Color,
        text_color: Color,
        font: String,
        font_size: u32,
    },

    /// Character generator
    CG {
        lines: Vec<String>,
        font: String,
        font_size: u32,
        text_color: Color,
        background_color: Color,
        align: TextAlign,
    },

    /// Ticker/crawler
    Ticker {
        text: String,
        font: String,
        font_size: u32,
        text_color: Color,
        background_color: Color,
        speed: f32,
    },

    /// Static image
    Image { path: PathBuf },

    /// Text overlay
    Text {
        text: String,
        font: String,
        font_size: u32,
        color: Color,
        align: TextAlign,
    },

    /// Custom content (placeholder)
    Custom { data: HashMap<String, String> },
}

impl GraphicsLayer {
    /// Create a logo layer
    pub fn logo(name: String, path: PathBuf, position: Position) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            layer_type: LayerType::Logo,
            position,
            size: None,
            z_order: 100,
            opacity: 1.0,
            visible: true,
            content: LayerContent::Logo { path },
            animations: Vec::new(),
            animation_frame: 0,
        }
    }

    /// Create a lower third layer
    #[allow(clippy::too_many_arguments)]
    pub fn lower_third(
        name: String,
        title: String,
        subtitle: String,
        position: Position,
        font: String,
        font_size: u32,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            layer_type: LayerType::LowerThird,
            position,
            size: Some(Size::new(0.8, 0.15)),
            z_order: 90,
            opacity: 1.0,
            visible: true,
            content: LayerContent::LowerThird {
                title,
                subtitle,
                background_color: Color::new(0, 0, 0, 200),
                text_color: Color::white(),
                font,
                font_size,
            },
            animations: Vec::new(),
            animation_frame: 0,
        }
    }

    /// Create a ticker layer
    #[allow(clippy::too_many_arguments)]
    pub fn ticker(
        name: String,
        text: String,
        position: Position,
        font: String,
        font_size: u32,
        speed: f32,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            layer_type: LayerType::Ticker,
            position,
            size: Some(Size::new(1.0, 0.1)),
            z_order: 95,
            opacity: 1.0,
            visible: true,
            content: LayerContent::Ticker {
                text,
                font,
                font_size,
                text_color: Color::white(),
                background_color: Color::new(0, 0, 0, 180),
                speed,
            },
            animations: vec![Animation::Scroll {
                direction: ScrollDirection::Left,
                speed,
            }],
            animation_frame: 0,
        }
    }

    /// Add an animation
    pub fn add_animation(&mut self, animation: Animation) {
        self.animations.push(animation);
    }

    /// Update animation state
    pub fn update_animation(&mut self) {
        self.animation_frame += 1;

        // Process active animations
        for animation in &self.animations {
            match animation {
                Animation::Fade {
                    from,
                    to,
                    duration_frames,
                    curve,
                } => {
                    if self.animation_frame < *duration_frames {
                        let progress = self.animation_frame as f32 / *duration_frames as f32;
                        let adjusted = apply_curve(progress, *curve);
                        self.opacity = from + (to - from) * adjusted;
                    }
                }
                Animation::Move {
                    from,
                    to,
                    duration_frames,
                    curve,
                } => {
                    if self.animation_frame < *duration_frames {
                        let progress = self.animation_frame as f32 / *duration_frames as f32;
                        let adjusted = apply_curve(progress, *curve);
                        self.position.x = from.x + (to.x - from.x) * adjusted;
                        self.position.y = from.y + (to.y - from.y) * adjusted;
                    }
                }
                Animation::Scroll { direction, speed } => {
                    // Continuous scroll
                    match direction {
                        ScrollDirection::Left => {
                            self.position.x -= speed;
                            if self.position.x < -1.0 {
                                self.position.x = 1.0;
                            }
                        }
                        ScrollDirection::Right => {
                            self.position.x += speed;
                            if self.position.x > 1.0 {
                                self.position.x = -1.0;
                            }
                        }
                        ScrollDirection::Up => {
                            self.position.y -= speed;
                            if self.position.y < -1.0 {
                                self.position.y = 1.0;
                            }
                        }
                        ScrollDirection::Down => {
                            self.position.y += speed;
                            if self.position.y > 1.0 {
                                self.position.y = -1.0;
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// Reset animations
    pub fn reset_animation(&mut self) {
        self.animation_frame = 0;
    }
}

/// Apply animation curve
fn apply_curve(t: f32, curve: AnimationCurve) -> f32 {
    match curve {
        AnimationCurve::Linear => t,
        AnimationCurve::EaseIn => t * t,
        AnimationCurve::EaseOut => t * (2.0 - t),
        AnimationCurve::EaseInOut => {
            if t < 0.5 {
                2.0 * t * t
            } else {
                -1.0 + (4.0 - 2.0 * t) * t
            }
        }
        AnimationCurve::Bounce => {
            let n1 = 7.5625;
            let d1 = 2.75;
            let mut t = t;

            if t < 1.0 / d1 {
                n1 * t * t
            } else if t < 2.0 / d1 {
                t -= 1.5 / d1;
                n1 * t * t + 0.75
            } else if t < 2.5 / d1 {
                t -= 2.25 / d1;
                n1 * t * t + 0.9375
            } else {
                t -= 2.625 / d1;
                n1 * t * t + 0.984375
            }
        }
    }
}

/// Graphics template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphicsTemplate {
    /// Template ID
    pub id: String,

    /// Template name
    pub name: String,

    /// Description
    pub description: String,

    /// Layers in this template
    pub layers: Vec<GraphicsLayer>,

    /// Template parameters (for customization)
    pub parameters: HashMap<String, String>,
}

/// Internal graphics state
struct GraphicsState {
    /// Active layers (sorted by z-order)
    layers: Vec<GraphicsLayer>,

    /// Templates
    templates: HashMap<String, GraphicsTemplate>,

    /// Frame counter
    frame_count: u64,
}

/// Graphics engine
pub struct GraphicsEngine {
    config: GraphicsConfig,
    state: Arc<RwLock<GraphicsState>>,
}

impl GraphicsEngine {
    /// Create a new graphics engine
    pub fn new(config: GraphicsConfig) -> Result<Self> {
        let state = GraphicsState {
            layers: Vec::new(),
            templates: HashMap::new(),
            frame_count: 0,
        };

        Ok(Self {
            config,
            state: Arc::new(RwLock::new(state)),
        })
    }

    /// Add a layer
    pub fn add_layer(&self, layer: GraphicsLayer) -> Result<Uuid> {
        let mut state = self.state.write();

        if state.layers.len() >= self.config.max_layers {
            return Err(PlayoutError::Graphics(
                "Maximum number of layers reached".to_string(),
            ));
        }

        let id = layer.id;
        state.layers.push(layer);
        self.sort_layers_internal(&mut state);

        Ok(id)
    }

    /// Remove a layer
    pub fn remove_layer(&self, layer_id: Uuid) -> Result<()> {
        let mut state = self.state.write();
        let original_len = state.layers.len();
        state.layers.retain(|layer| layer.id != layer_id);

        if state.layers.len() < original_len {
            Ok(())
        } else {
            Err(PlayoutError::Graphics(format!(
                "Layer not found: {layer_id}"
            )))
        }
    }

    /// Get a layer
    pub fn get_layer(&self, layer_id: Uuid) -> Option<GraphicsLayer> {
        self.state
            .read()
            .layers
            .iter()
            .find(|layer| layer.id == layer_id)
            .cloned()
    }

    /// Update a layer
    pub fn update_layer(&self, layer_id: Uuid, updated: GraphicsLayer) -> Result<()> {
        let mut state = self.state.write();
        if let Some(layer) = state.layers.iter_mut().find(|l| l.id == layer_id) {
            *layer = updated;
            self.sort_layers_internal(&mut state);
            Ok(())
        } else {
            Err(PlayoutError::Graphics(format!(
                "Layer not found: {layer_id}"
            )))
        }
    }

    /// Show layer
    pub fn show_layer(&self, layer_id: Uuid) -> Result<()> {
        let mut state = self.state.write();
        if let Some(layer) = state.layers.iter_mut().find(|l| l.id == layer_id) {
            layer.visible = true;
            Ok(())
        } else {
            Err(PlayoutError::Graphics(format!(
                "Layer not found: {layer_id}"
            )))
        }
    }

    /// Hide layer
    pub fn hide_layer(&self, layer_id: Uuid) -> Result<()> {
        let mut state = self.state.write();
        if let Some(layer) = state.layers.iter_mut().find(|l| l.id == layer_id) {
            layer.visible = false;
            Ok(())
        } else {
            Err(PlayoutError::Graphics(format!(
                "Layer not found: {layer_id}"
            )))
        }
    }

    /// Clear all layers
    pub fn clear_layers(&self) {
        self.state.write().layers.clear();
    }

    /// Get all visible layers
    pub fn get_visible_layers(&self) -> Vec<GraphicsLayer> {
        self.state
            .read()
            .layers
            .iter()
            .filter(|layer| layer.visible)
            .cloned()
            .collect()
    }

    /// Update all layer animations
    pub fn update_animations(&self) {
        let mut state = self.state.write();
        for layer in &mut state.layers {
            if layer.visible {
                layer.update_animation();
            }
        }
        state.frame_count += 1;
    }

    /// Sort layers by z-order
    fn sort_layers_internal(&self, state: &mut GraphicsState) {
        state.layers.sort_by_key(|layer| layer.z_order);
    }

    /// Add a template
    pub fn add_template(&self, template: GraphicsTemplate) {
        let mut state = self.state.write();
        state.templates.insert(template.id.clone(), template);
    }

    /// Remove a template
    pub fn remove_template(&self, template_id: &str) -> Result<()> {
        let mut state = self.state.write();
        state.templates.remove(template_id);
        Ok(())
    }

    /// Substitute template variables in a string.
    ///
    /// Recognised variable patterns (curly-brace syntax):
    ///   `{key}` is replaced by the value of `key` in `params`.
    ///
    /// Built-in variables that are always available:
    ///   `{time}`  – current UTC time as HH:MM:SS
    ///   `{date}`  – current UTC date as YYYY-MM-DD
    ///   `{frame}` – current graphics engine frame counter
    fn substitute_params(text: &str, params: &HashMap<String, String>, frame_count: u64) -> String {
        let now = Utc::now();
        let built_ins: &[(&str, String)] = &[
            ("time", now.format("%H:%M:%S").to_string()),
            ("date", now.format("%Y-%m-%d").to_string()),
            ("frame", frame_count.to_string()),
        ];

        let mut result = text.to_string();

        // Apply user-supplied parameters first (allow them to override built-ins)
        for (key, value) in params {
            let placeholder = format!("{{{key}}}");
            result = result.replace(&placeholder, value);
        }

        // Apply built-in variables for any remaining placeholders
        for (key, value) in built_ins {
            let placeholder = format!("{{{key}}}");
            result = result.replace(&placeholder, value);
        }

        result
    }

    /// Apply parameter substitution to all text fields within a layer.
    fn apply_params_to_layer(
        layer: &mut GraphicsLayer,
        params: &HashMap<String, String>,
        frame_count: u64,
    ) {
        let sub = |s: &str| Self::substitute_params(s, params, frame_count);

        layer.name = sub(&layer.name);

        match &mut layer.content {
            LayerContent::LowerThird {
                title, subtitle, ..
            } => {
                *title = sub(title);
                *subtitle = sub(subtitle);
            }
            LayerContent::CG { lines, .. } => {
                for line in lines.iter_mut() {
                    *line = sub(line);
                }
            }
            LayerContent::Ticker { text, .. } => {
                *text = sub(text);
            }
            LayerContent::Text { text, .. } => {
                *text = sub(text);
            }
            LayerContent::Custom { data } => {
                for value in data.values_mut() {
                    *value = sub(value);
                }
            }
            // Logo and Image layers have no text fields to substitute.
            LayerContent::Logo { .. } | LayerContent::Image { .. } => {}
        }
    }

    /// Instantiate a template
    pub fn instantiate_template(
        &self,
        template_id: &str,
        params: HashMap<String, String>,
    ) -> Result<Vec<Uuid>> {
        // Clone the layers and frame count first, then release the lock
        let (layers, frame_count) = {
            let state = self.state.read();
            let template = state.templates.get(template_id).ok_or_else(|| {
                PlayoutError::Graphics(format!("Template not found: {template_id}"))
            })?;
            // Merge template parameters with caller-supplied params
            let mut merged_params = template.parameters.clone();
            for (k, v) in &params {
                merged_params.insert(k.clone(), v.clone());
            }
            (template.layers.clone(), state.frame_count)
        };

        // Re-derive merged params (template params merged with caller params)
        let merged_params = {
            let state = self.state.read();
            let template = state.templates.get(template_id).ok_or_else(|| {
                PlayoutError::Graphics(format!("Template not found: {template_id}"))
            })?;
            let mut mp = template.parameters.clone();
            for (k, v) in &params {
                mp.insert(k.clone(), v.clone());
            }
            mp
        };

        let mut layer_ids = Vec::new();
        for layer in &layers {
            let mut new_layer = layer.clone();
            new_layer.id = Uuid::new_v4();

            // Apply parameter substitution to text fields
            Self::apply_params_to_layer(&mut new_layer, &merged_params, frame_count);

            let id = self.add_layer(new_layer)?;
            layer_ids.push(id);
        }

        Ok(layer_ids)
    }

    /// Get frame count
    pub fn frame_count(&self) -> u64 {
        self.state.read().frame_count
    }

    /// Reset frame count
    pub fn reset_frame_count(&self) {
        self.state.write().frame_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position() {
        let pos = Position::center();
        assert_eq!(pos.x, 0.5);
        assert_eq!(pos.y, 0.5);
    }

    #[test]
    fn test_color() {
        let white = Color::white();
        assert_eq!(white.r, 255);
        assert_eq!(white.a, 255);

        let transparent = Color::transparent();
        assert_eq!(transparent.a, 0);
    }

    #[test]
    fn test_graphics_layer_logo() {
        let layer = GraphicsLayer::logo(
            "Test Logo".to_string(),
            PathBuf::from("/logo.png"),
            Position::top_right(),
        );

        assert_eq!(layer.layer_type, LayerType::Logo);
        assert!(layer.visible);
    }

    #[test]
    fn test_graphics_layer_ticker() {
        let layer = GraphicsLayer::ticker(
            "Test Ticker".to_string(),
            "Breaking news".to_string(),
            Position::bottom_left(),
            "Arial".to_string(),
            32,
            0.01,
        );

        assert_eq!(layer.layer_type, LayerType::Ticker);
        assert_eq!(layer.animations.len(), 1);
    }

    #[test]
    fn test_animation_curve() {
        let linear = apply_curve(0.5, AnimationCurve::Linear);
        assert_eq!(linear, 0.5);

        let ease_in = apply_curve(0.5, AnimationCurve::EaseIn);
        assert_eq!(ease_in, 0.25);
    }

    #[test]
    fn test_graphics_engine() {
        let config = GraphicsConfig::default();
        let engine = GraphicsEngine::new(config).expect("should succeed in test");

        let layer = GraphicsLayer::logo(
            "Logo".to_string(),
            PathBuf::from("/logo.png"),
            Position::top_left(),
        );

        let id = engine.add_layer(layer).expect("should succeed in test");
        assert!(engine.get_layer(id).is_some());

        engine.remove_layer(id).expect("should succeed in test");
        assert!(engine.get_layer(id).is_none());
    }

    #[test]
    fn test_layer_visibility() {
        let config = GraphicsConfig::default();
        let engine = GraphicsEngine::new(config).expect("should succeed in test");

        let layer = GraphicsLayer::logo(
            "Logo".to_string(),
            PathBuf::from("/logo.png"),
            Position::top_left(),
        );
        let id = engine.add_layer(layer).expect("should succeed in test");

        engine.hide_layer(id).expect("should succeed in test");
        let hidden = engine.get_layer(id).expect("should succeed in test");
        assert!(!hidden.visible);

        engine.show_layer(id).expect("should succeed in test");
        let shown = engine.get_layer(id).expect("should succeed in test");
        assert!(shown.visible);
    }

    #[test]
    fn test_z_order() {
        let mut layer1 = GraphicsLayer::logo(
            "Layer 1".to_string(),
            PathBuf::from("/logo1.png"),
            Position::top_left(),
        );
        layer1.z_order = 10;

        let mut layer2 = GraphicsLayer::logo(
            "Layer 2".to_string(),
            PathBuf::from("/logo2.png"),
            Position::top_left(),
        );
        layer2.z_order = 20;

        assert!(layer2.z_order > layer1.z_order);
    }
}
