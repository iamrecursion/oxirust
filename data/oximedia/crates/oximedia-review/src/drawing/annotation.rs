//! Annotation management for drawings.

use crate::{
    drawing::{Drawing, Shape, StrokeStyle},
    error::ReviewResult,
    DrawingId, SessionId,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Annotation with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Annotation {
    /// Annotation ID.
    pub id: DrawingId,
    /// Associated drawing.
    pub drawing: Drawing,
    /// Annotation label.
    pub label: Option<String>,
    /// Visibility.
    pub visible: bool,
    /// Locked status (prevents editing).
    pub locked: bool,
    /// Layer index (for z-ordering).
    pub layer: usize,
    /// Creation timestamp.
    pub created_at: DateTime<Utc>,
    /// Last updated timestamp.
    pub updated_at: DateTime<Utc>,
}

impl Annotation {
    /// Create a new annotation.
    #[must_use]
    pub fn new(drawing: Drawing) -> Self {
        let now = Utc::now();
        Self {
            id: drawing.id,
            drawing,
            label: None,
            visible: true,
            locked: false,
            layer: 0,
            created_at: now,
            updated_at: now,
        }
    }

    /// Set the label.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set the layer.
    #[must_use]
    pub fn with_layer(mut self, layer: usize) -> Self {
        self.layer = layer;
        self
    }

    /// Hide the annotation.
    pub fn hide(&mut self) {
        self.visible = false;
    }

    /// Show the annotation.
    pub fn show(&mut self) {
        self.visible = true;
    }

    /// Lock the annotation.
    pub fn lock(&mut self) {
        self.locked = true;
    }

    /// Unlock the annotation.
    pub fn unlock(&mut self) {
        self.locked = false;
    }
}

/// Layer of annotations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotationLayer {
    /// Layer ID.
    pub id: String,
    /// Layer name.
    pub name: String,
    /// Annotations in this layer.
    pub annotations: Vec<Annotation>,
    /// Layer visibility.
    pub visible: bool,
    /// Layer locked status.
    pub locked: bool,
    /// Layer opacity (0.0-1.0).
    pub opacity: f32,
}

impl AnnotationLayer {
    /// Create a new layer.
    #[must_use]
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            annotations: Vec::new(),
            visible: true,
            locked: false,
            opacity: 1.0,
        }
    }

    /// Add an annotation to the layer.
    pub fn add_annotation(&mut self, annotation: Annotation) {
        self.annotations.push(annotation);
    }

    /// Remove an annotation from the layer.
    pub fn remove_annotation(&mut self, id: DrawingId) -> bool {
        if let Some(index) = self.annotations.iter().position(|a| a.id == id) {
            self.annotations.remove(index);
            true
        } else {
            false
        }
    }

    /// Get annotation by ID.
    #[must_use]
    pub fn get_annotation(&self, id: DrawingId) -> Option<&Annotation> {
        self.annotations.iter().find(|a| a.id == id)
    }

    /// Count annotations in the layer.
    #[must_use]
    pub fn count(&self) -> usize {
        self.annotations.len()
    }

    /// Hide all annotations in the layer.
    pub fn hide_all(&mut self) {
        for annotation in &mut self.annotations {
            annotation.hide();
        }
    }

    /// Show all annotations in the layer.
    pub fn show_all(&mut self) {
        for annotation in &mut self.annotations {
            annotation.show();
        }
    }
}

/// Annotation manager for organizing drawings.
pub struct AnnotationManager {
    layers: HashMap<String, AnnotationLayer>,
    session_id: SessionId,
}

impl AnnotationManager {
    /// Create a new annotation manager.
    #[must_use]
    pub fn new(session_id: SessionId) -> Self {
        let mut layers = HashMap::new();
        layers.insert(
            "default".to_string(),
            AnnotationLayer::new("default", "Default Layer"),
        );

        Self { layers, session_id }
    }

    /// Create a new layer.
    pub fn create_layer(&mut self, id: impl Into<String>, name: impl Into<String>) {
        let id = id.into();
        let layer = AnnotationLayer::new(id.clone(), name);
        self.layers.insert(id, layer);
    }

    /// Get a layer by ID.
    #[must_use]
    pub fn get_layer(&self, id: &str) -> Option<&AnnotationLayer> {
        self.layers.get(id)
    }

    /// Get a mutable layer by ID.
    pub fn get_layer_mut(&mut self, id: &str) -> Option<&mut AnnotationLayer> {
        self.layers.get_mut(id)
    }

    /// Delete a layer.
    pub fn delete_layer(&mut self, id: &str) -> bool {
        if id == "default" {
            return false; // Cannot delete default layer
        }
        self.layers.remove(id).is_some()
    }

    /// List all layers.
    #[must_use]
    pub fn list_layers(&self) -> Vec<&AnnotationLayer> {
        self.layers.values().collect()
    }

    /// Add annotation to a layer.
    ///
    /// # Errors
    ///
    /// Returns error if layer not found.
    pub fn add_annotation(&mut self, layer_id: &str, annotation: Annotation) -> ReviewResult<()> {
        let layer = self
            .get_layer_mut(layer_id)
            .ok_or_else(|| crate::error::ReviewError::Other("Layer not found".to_string()))?;

        layer.add_annotation(annotation);
        Ok(())
    }

    /// Get session ID.
    #[must_use]
    pub fn session_id(&self) -> SessionId {
        self.session_id
    }
}

/// Create a drawing annotation.
///
/// # Errors
///
/// Returns error if annotation cannot be created.
pub async fn create_annotation(
    session_id: SessionId,
    frame: i64,
    shape: Shape,
    style: StrokeStyle,
) -> ReviewResult<Annotation> {
    let drawing = Drawing {
        id: DrawingId::new(),
        session_id,
        frame,
        tool: crate::drawing::tools::DrawingTool::Pen,
        shape,
        style,
        author: "system".to_string(),
    };

    Ok(Annotation::new(drawing))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drawing::{Circle, Color, Point};

    #[test]
    fn test_annotation_creation() {
        let drawing = Drawing {
            id: DrawingId::new(),
            session_id: SessionId::new(),
            frame: 100,
            tool: crate::drawing::tools::DrawingTool::Circle,
            shape: Shape::Circle(Circle::new(Point::new(0.5, 0.5), 0.2)),
            style: StrokeStyle::solid(Color::red(), 2.0),
            author: "test".to_string(),
        };

        let annotation = Annotation::new(drawing);
        assert!(annotation.visible);
        assert!(!annotation.locked);
    }

    #[test]
    fn test_annotation_layer() {
        let mut layer = AnnotationLayer::new("layer1", "Test Layer");
        assert_eq!(layer.count(), 0);

        let drawing = Drawing {
            id: DrawingId::new(),
            session_id: SessionId::new(),
            frame: 100,
            tool: crate::drawing::tools::DrawingTool::Circle,
            shape: Shape::Circle(Circle::new(Point::new(0.5, 0.5), 0.2)),
            style: StrokeStyle::solid(Color::red(), 2.0),
            author: "test".to_string(),
        };

        let annotation = Annotation::new(drawing);
        let annotation_id = annotation.id;

        layer.add_annotation(annotation);
        assert_eq!(layer.count(), 1);

        assert!(layer.remove_annotation(annotation_id));
        assert_eq!(layer.count(), 0);
    }

    #[test]
    fn test_annotation_manager() {
        let session_id = SessionId::new();
        let mut manager = AnnotationManager::new(session_id);

        manager.create_layer("layer1", "Test Layer");
        assert!(manager.get_layer("layer1").is_some());

        let layers = manager.list_layers();
        assert_eq!(layers.len(), 2); // default + layer1

        assert!(!manager.delete_layer("default")); // Cannot delete default
        assert!(manager.delete_layer("layer1"));
    }
}
