//! Layer management for compositing.

use super::blend::BlendMode;
use crate::{Frame, VfxResult};
use serde::{Deserialize, Serialize};

/// 2D transform for layer positioning.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Transform {
    /// Position X.
    pub position_x: f32,
    /// Position Y.
    pub position_y: f32,
    /// Rotation in degrees.
    pub rotation: f32,
    /// Scale X.
    pub scale_x: f32,
    /// Scale Y.
    pub scale_y: f32,
    /// Anchor point X.
    pub anchor_x: f32,
    /// Anchor point Y.
    pub anchor_y: f32,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position_x: 0.0,
            position_y: 0.0,
            rotation: 0.0,
            scale_x: 1.0,
            scale_y: 1.0,
            anchor_x: 0.0,
            anchor_y: 0.0,
        }
    }
}

impl Transform {
    /// Create a new identity transform.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set position.
    #[must_use]
    pub const fn with_position(mut self, x: f32, y: f32) -> Self {
        self.position_x = x;
        self.position_y = y;
        self
    }

    /// Set rotation in degrees.
    #[must_use]
    pub const fn with_rotation(mut self, degrees: f32) -> Self {
        self.rotation = degrees;
        self
    }

    /// Set scale.
    #[must_use]
    pub const fn with_scale(mut self, x: f32, y: f32) -> Self {
        self.scale_x = x;
        self.scale_y = y;
        self
    }

    /// Set anchor point.
    #[must_use]
    pub const fn with_anchor(mut self, x: f32, y: f32) -> Self {
        self.anchor_x = x;
        self.anchor_y = y;
        self
    }

    /// Transform a point from layer space to canvas space.
    #[must_use]
    pub fn transform_point(&self, x: f32, y: f32) -> (f32, f32) {
        // Translate to anchor
        let x = x - self.anchor_x;
        let y = y - self.anchor_y;

        // Scale
        let x = x * self.scale_x;
        let y = y * self.scale_y;

        // Rotate
        let rad = self.rotation.to_radians();
        let cos = rad.cos();
        let sin = rad.sin();
        let x_rot = x * cos - y * sin;
        let y_rot = x * sin + y * cos;

        // Translate to position
        (x_rot + self.position_x, y_rot + self.position_y)
    }

    /// Inverse transform a point from canvas space to layer space.
    #[must_use]
    pub fn inverse_transform_point(&self, x: f32, y: f32) -> (f32, f32) {
        // Translate from position
        let x = x - self.position_x;
        let y = y - self.position_y;

        // Inverse rotate
        let rad = -self.rotation.to_radians();
        let cos = rad.cos();
        let sin = rad.sin();
        let x_rot = x * cos - y * sin;
        let y_rot = x * sin + y * cos;

        // Inverse scale
        let x = if self.scale_x == 0.0 {
            0.0
        } else {
            x_rot / self.scale_x
        };
        let y = if self.scale_y == 0.0 {
            0.0
        } else {
            y_rot / self.scale_y
        };

        // Translate from anchor
        (x + self.anchor_x, y + self.anchor_y)
    }
}

/// A compositing layer.
#[derive(Debug, Clone)]
pub struct Layer {
    /// Layer name.
    pub name: String,
    /// Layer frame data.
    pub frame: Frame,
    /// Layer transform.
    pub transform: Transform,
    /// Layer opacity (0.0 - 1.0).
    pub opacity: f32,
    /// Layer blend mode.
    pub blend_mode: BlendMode,
    /// Layer visibility.
    pub visible: bool,
    /// Z-index for layer ordering.
    pub z_index: i32,
}

impl Layer {
    /// Create a new layer.
    #[must_use]
    pub fn new(name: String, frame: Frame) -> Self {
        Self {
            name,
            frame,
            transform: Transform::default(),
            opacity: 1.0,
            blend_mode: BlendMode::Normal,
            visible: true,
            z_index: 0,
        }
    }

    /// Set layer opacity.
    #[must_use]
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity.clamp(0.0, 1.0);
        self
    }

    /// Set layer blend mode.
    #[must_use]
    pub const fn with_blend_mode(mut self, blend_mode: BlendMode) -> Self {
        self.blend_mode = blend_mode;
        self
    }

    /// Set layer visibility.
    #[must_use]
    pub const fn with_visibility(mut self, visible: bool) -> Self {
        self.visible = visible;
        self
    }

    /// Set layer z-index.
    #[must_use]
    pub const fn with_z_index(mut self, z_index: i32) -> Self {
        self.z_index = z_index;
        self
    }

    /// Set layer transform.
    #[must_use]
    pub const fn with_transform(mut self, transform: Transform) -> Self {
        self.transform = transform;
        self
    }
}

/// Stack of compositing layers.
#[derive(Debug, Clone)]
pub struct LayerStack {
    layers: Vec<Layer>,
}

impl LayerStack {
    /// Create a new empty layer stack.
    #[must_use]
    pub const fn new() -> Self {
        Self { layers: Vec::new() }
    }

    /// Add a layer to the stack.
    pub fn add_layer(&mut self, layer: Layer) {
        self.layers.push(layer);
        self.sort_layers();
    }

    /// Remove a layer by index.
    pub fn remove_layer(&mut self, index: usize) -> Option<Layer> {
        if index < self.layers.len() {
            Some(self.layers.remove(index))
        } else {
            None
        }
    }

    /// Get a layer by index.
    #[must_use]
    pub fn get_layer(&self, index: usize) -> Option<&Layer> {
        self.layers.get(index)
    }

    /// Get a mutable layer by index.
    pub fn get_layer_mut(&mut self, index: usize) -> Option<&mut Layer> {
        self.layers.get_mut(index)
    }

    /// Get number of layers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Check if stack is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Sort layers by z-index.
    fn sort_layers(&mut self) {
        self.layers.sort_by_key(|layer| layer.z_index);
    }

    /// Composite all layers into output frame.
    pub fn composite(&self, output: &mut Frame) -> VfxResult<()> {
        // Clear output
        output.clear([0, 0, 0, 0]);

        // Composite each visible layer
        for layer in &self.layers {
            if !layer.visible {
                continue;
            }

            self.composite_layer(layer, output)?;
        }

        Ok(())
    }

    /// Composite a single layer onto output.
    fn composite_layer(&self, layer: &Layer, output: &mut Frame) -> VfxResult<()> {
        use super::blend::blend_pixels;

        for y in 0..output.height {
            for x in 0..output.width {
                // Transform canvas coordinates to layer coordinates
                let (layer_x, layer_y) =
                    layer.transform.inverse_transform_point(x as f32, y as f32);

                // Sample layer pixel
                let layer_pixel = if layer_x >= 0.0
                    && layer_x < layer.frame.width as f32
                    && layer_y >= 0.0
                    && layer_y < layer.frame.height as f32
                {
                    // Bilinear interpolation
                    self.sample_bilinear(&layer.frame, layer_x, layer_y)
                } else {
                    [0, 0, 0, 0]
                };

                // Apply layer opacity
                let mut layer_pixel_opacity = layer_pixel;
                layer_pixel_opacity[3] = (f32::from(layer_pixel[3]) * layer.opacity) as u8;

                // Blend with output
                let output_pixel = output.get_pixel(x, y).unwrap_or([0, 0, 0, 0]);
                let blended =
                    blend_pixels(output_pixel, layer_pixel_opacity, layer.blend_mode, 1.0);
                output.set_pixel(x, y, blended);
            }
        }

        Ok(())
    }

    /// Sample pixel with bilinear interpolation.
    fn sample_bilinear(&self, frame: &Frame, x: f32, y: f32) -> [u8; 4] {
        let x0 = x.floor() as u32;
        let y0 = y.floor() as u32;
        let x1 = (x0 + 1).min(frame.width - 1);
        let y1 = (y0 + 1).min(frame.height - 1);

        let fx = x - x0 as f32;
        let fy = y - y0 as f32;

        let p00 = frame.get_pixel(x0, y0).unwrap_or([0, 0, 0, 0]);
        let p10 = frame.get_pixel(x1, y0).unwrap_or([0, 0, 0, 0]);
        let p01 = frame.get_pixel(x0, y1).unwrap_or([0, 0, 0, 0]);
        let p11 = frame.get_pixel(x1, y1).unwrap_or([0, 0, 0, 0]);

        let mut result = [0u8; 4];
        for i in 0..4 {
            let v0 = f32::from(p00[i]) * (1.0 - fx) + f32::from(p10[i]) * fx;
            let v1 = f32::from(p01[i]) * (1.0 - fx) + f32::from(p11[i]) * fx;
            result[i] = (v0 * (1.0 - fy) + v1 * fy) as u8;
        }

        result
    }
}

impl Default for LayerStack {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transform_identity() {
        let transform = Transform::default();
        let (x, y) = transform.transform_point(10.0, 20.0);
        assert!((x - 10.0).abs() < 0.01);
        assert!((y - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_transform_translation() {
        let transform = Transform::default().with_position(5.0, 10.0);
        let (x, y) = transform.transform_point(0.0, 0.0);
        assert!((x - 5.0).abs() < 0.01);
        assert!((y - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_transform_scale() {
        let transform = Transform::default().with_scale(2.0, 2.0);
        let (x, y) = transform.transform_point(10.0, 10.0);
        assert!((x - 20.0).abs() < 0.01);
        assert!((y - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_transform_inverse() {
        let transform = Transform::default()
            .with_position(10.0, 20.0)
            .with_rotation(45.0)
            .with_scale(2.0, 2.0);
        let (x, y) = transform.transform_point(5.0, 5.0);
        let (x2, y2) = transform.inverse_transform_point(x, y);
        assert!((x2 - 5.0).abs() < 0.01);
        assert!((y2 - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_layer_stack() -> VfxResult<()> {
        let mut stack = LayerStack::new();
        let frame1 = Frame::new(100, 100)?;
        let frame2 = Frame::new(100, 100)?;

        stack.add_layer(Layer::new("Layer 1".to_string(), frame1).with_z_index(0));
        stack.add_layer(Layer::new("Layer 2".to_string(), frame2).with_z_index(1));

        assert_eq!(stack.len(), 2);
        assert_eq!(
            stack.get_layer(0).expect("should succeed in test").name,
            "Layer 1"
        );
        Ok(())
    }
}
