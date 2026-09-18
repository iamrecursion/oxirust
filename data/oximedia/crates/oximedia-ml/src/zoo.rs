//! Lightweight model registry ("zoo").
//!
//! The zoo is a static, in-memory catalogue that maps stable model IDs
//! to descriptive metadata. It intentionally does **not** embed weights
//! or network fetch logic — users bring their own `.onnx` file. The zoo
//! is purely a discovery surface so pipelines can advertise their
//! expected input contract.
//!
//! ## Example
//!
//! ```
//! use oximedia_ml::ModelZoo;
//!
//! let zoo = ModelZoo::with_defaults();
//! let scene = zoo.get("places365/resnet18").expect("default entry");
//! assert_eq!(scene.input_size, Some((224, 224)));
//! assert_eq!(scene.num_classes, Some(365));
//! ```
//!
//! Register custom entries with [`ModelZoo::register`]. IDs are unique
//! — registering the same ID twice overwrites the previous record.

use std::collections::HashMap;

use crate::pipeline::PipelineTask;

/// Metadata entry describing a model that can be plugged into a pipeline.
///
/// Entries are static — each field is `&'static str` / `Option<…>` so
/// entries can live in a `const`-friendly table. Use
/// [`ModelZoo::register`] to add a [`ModelEntry`] to a zoo instance.
#[derive(Clone, Debug)]
pub struct ModelEntry {
    /// Stable unique ID, e.g. `"places365/resnet18"`.
    pub id: &'static str,
    /// Human-readable name.
    pub name: &'static str,
    /// Which pipeline task this model is intended for.
    pub task: PipelineTask,
    /// Expected `(width, height)` of the image input, if applicable.
    pub input_size: Option<(u32, u32)>,
    /// Number of output classes, if applicable.
    pub num_classes: Option<usize>,
    /// Short notes / citation for the user.
    pub notes: &'static str,
}

/// In-memory registry of known models.
#[derive(Clone, Debug, Default)]
pub struct ModelZoo {
    entries: HashMap<&'static str, ModelEntry>,
}

impl ModelZoo {
    /// Create an empty zoo.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create the default zoo with built-in entries.
    #[must_use]
    pub fn with_defaults() -> Self {
        let mut zoo = Self::new();
        zoo.register(ModelEntry {
            id: "places365/resnet18",
            name: "Places365 ResNet-18 scene classifier",
            task: PipelineTask::SceneClassification,
            input_size: Some((224, 224)),
            num_classes: Some(365),
            notes: "Bring your own ONNX export of the Places365 ResNet-18 model.",
        });
        zoo.register(ModelEntry {
            id: "transnet-v2",
            name: "TransNet V2 shot boundary detector",
            task: PipelineTask::ShotBoundary,
            input_size: Some((48, 27)),
            num_classes: Some(2),
            notes: "Sliding-window input of 100 frames at 48x27 RGB per window.",
        });
        zoo
    }

    /// Register a new entry (overwrites any existing entry with the same ID).
    pub fn register(&mut self, entry: ModelEntry) {
        self.entries.insert(entry.id, entry);
    }

    /// Look up an entry by ID.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&ModelEntry> {
        self.entries.get(id)
    }

    /// Return all registered entries.
    pub fn entries(&self) -> impl Iterator<Item = &ModelEntry> {
        self.entries.values()
    }

    /// Number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the zoo has no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_contain_scene_and_shot() {
        let zoo = ModelZoo::with_defaults();
        assert!(zoo.get("places365/resnet18").is_some());
        assert!(zoo.get("transnet-v2").is_some());
    }

    #[test]
    fn empty_zoo_reports_empty() {
        let zoo = ModelZoo::new();
        assert!(zoo.is_empty());
        assert_eq!(zoo.len(), 0);
    }

    #[test]
    fn register_adds_entry() {
        let mut zoo = ModelZoo::new();
        zoo.register(ModelEntry {
            id: "demo/x",
            name: "Demo",
            task: PipelineTask::Custom,
            input_size: None,
            num_classes: None,
            notes: "",
        });
        assert_eq!(zoo.len(), 1);
        assert!(zoo.get("demo/x").is_some());
    }
}
