//! Pipeline stage graph for transcoding workflows.
//!
//! Provides a directed graph model of transcode pipeline stages,
//! supporting stage classification, passthrough detection, and
//! transcode-stage presence checks.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

/// Classification of a pipeline stage by its transformation role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StageType {
    /// Decode compressed media into raw frames.
    Decode,
    /// Encode raw frames into compressed media.
    Encode,
    /// Apply a filter or transformation to raw frames.
    Filter,
    /// Scale or resize video frames.
    Scale,
    /// Normalize or adjust audio.
    AudioProcess,
    /// Demux a container into elementary streams.
    Demux,
    /// Mux elementary streams into a container.
    Mux,
    /// Pass data through unchanged.
    Passthrough,
    /// Analyse content without modifying it.
    Analyse,
}

impl StageType {
    /// Returns `true` if this stage type modifies media data.
    #[must_use]
    pub fn is_transform(self) -> bool {
        matches!(
            self,
            Self::Decode | Self::Encode | Self::Filter | Self::Scale | Self::AudioProcess
        )
    }

    /// Returns `true` if this stage type is a transcode-class operation
    /// (i.e. involves both decode and encode).
    #[must_use]
    pub fn is_transcode_class(self) -> bool {
        matches!(self, Self::Decode | Self::Encode)
    }

    /// Returns a human-readable label for the stage type.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Decode => "decode",
            Self::Encode => "encode",
            Self::Filter => "filter",
            Self::Scale => "scale",
            Self::AudioProcess => "audio_process",
            Self::Demux => "demux",
            Self::Mux => "mux",
            Self::Passthrough => "passthrough",
            Self::Analyse => "analyse",
        }
    }
}

/// A single stage within a transcode pipeline graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStage {
    /// Unique identifier for this stage.
    pub id: u32,
    /// Human-readable name.
    pub name: String,
    /// Functional classification of this stage.
    pub stage_type: StageType,
    /// Whether this stage is currently enabled.
    pub enabled: bool,
    /// Estimated processing cost (arbitrary units, higher = slower).
    pub cost_estimate: u32,
}

impl GraphStage {
    /// Create a new enabled stage.
    pub fn new(id: u32, name: impl Into<String>, stage_type: StageType) -> Self {
        Self {
            id,
            name: name.into(),
            stage_type,
            enabled: true,
            cost_estimate: 1,
        }
    }

    /// Returns `true` if this stage passes data through unchanged.
    #[must_use]
    pub fn is_passthrough(&self) -> bool {
        self.stage_type == StageType::Passthrough || !self.enabled
    }

    /// Returns `true` if this stage transforms the media.
    #[must_use]
    pub fn is_transform(&self) -> bool {
        self.enabled && self.stage_type.is_transform()
    }

    /// Set the cost estimate for this stage.
    #[must_use]
    pub fn with_cost(mut self, cost: u32) -> Self {
        self.cost_estimate = cost;
        self
    }

    /// Disable this stage (makes it behave as passthrough).
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Enable this stage.
    pub fn enable(&mut self) {
        self.enabled = true;
    }
}

/// A directed acyclic graph of pipeline stages for a transcode job.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TranscodeGraph {
    stages: Vec<GraphStage>,
    /// Edges stored as (`from_id`, `to_id`).
    edges: Vec<(u32, u32)>,
}

impl TranscodeGraph {
    /// Create a new empty pipeline graph.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a stage to the graph and return its assigned id.
    pub fn add_stage(&mut self, stage: GraphStage) -> u32 {
        let id = stage.id;
        self.stages.push(stage);
        id
    }

    /// Connect two stages by their ids.  Returns `false` if either id is unknown.
    pub fn connect(&mut self, from_id: u32, to_id: u32) -> bool {
        let has_from = self.stages.iter().any(|s| s.id == from_id);
        let has_to = self.stages.iter().any(|s| s.id == to_id);
        if has_from && has_to {
            self.edges.push((from_id, to_id));
            true
        } else {
            false
        }
    }

    /// Return the total number of stages in the graph.
    #[must_use]
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// Return the number of active (enabled) stages.
    #[must_use]
    pub fn active_stage_count(&self) -> usize {
        self.stages.iter().filter(|s| s.enabled).count()
    }

    /// Return `true` if the graph contains at least one transcode-class stage.
    #[must_use]
    pub fn has_transcode_stage(&self) -> bool {
        self.stages
            .iter()
            .any(|s| s.enabled && s.stage_type.is_transcode_class())
    }

    /// Return `true` if every enabled stage is a passthrough or analyse stage.
    #[must_use]
    pub fn is_fully_passthrough(&self) -> bool {
        self.stages
            .iter()
            .all(|s| s.is_passthrough() || s.stage_type == StageType::Analyse)
    }

    /// Estimate total processing cost by summing all enabled stage costs.
    #[must_use]
    pub fn total_cost(&self) -> u32 {
        self.stages
            .iter()
            .filter(|s| s.enabled)
            .map(|s| s.cost_estimate)
            .sum()
    }

    /// Return a list of stage labels in insertion order.
    #[must_use]
    pub fn stage_labels(&self) -> Vec<&str> {
        self.stages.iter().map(|s| s.stage_type.label()).collect()
    }

    /// Find a stage by id.
    #[must_use]
    pub fn find_stage(&self, id: u32) -> Option<&GraphStage> {
        self.stages.iter().find(|s| s.id == id)
    }

    /// Remove a stage by id.  Also removes associated edges.
    pub fn remove_stage(&mut self, id: u32) -> bool {
        if let Some(pos) = self.stages.iter().position(|s| s.id == id) {
            self.stages.remove(pos);
            self.edges.retain(|(f, t)| *f != id && *t != id);
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_stage(id: u32, t: StageType) -> GraphStage {
        GraphStage::new(id, format!("stage_{id}"), t)
    }

    #[test]
    fn test_stage_type_is_transform_decode() {
        assert!(StageType::Decode.is_transform());
    }

    #[test]
    fn test_stage_type_is_transform_passthrough() {
        assert!(!StageType::Passthrough.is_transform());
    }

    #[test]
    fn test_stage_type_transcode_class() {
        assert!(StageType::Encode.is_transcode_class());
        assert!(!StageType::Filter.is_transcode_class());
    }

    #[test]
    fn test_stage_type_labels_unique() {
        let all = [
            StageType::Decode,
            StageType::Encode,
            StageType::Filter,
            StageType::Scale,
            StageType::AudioProcess,
            StageType::Demux,
            StageType::Mux,
            StageType::Passthrough,
            StageType::Analyse,
        ];
        let labels: Vec<_> = all.iter().map(|t| t.label()).collect();
        let unique: std::collections::HashSet<_> = labels.iter().collect();
        assert_eq!(labels.len(), unique.len());
    }

    #[test]
    fn test_graph_stage_is_passthrough_when_disabled() {
        let mut s = make_stage(1, StageType::Encode);
        assert!(!s.is_passthrough());
        s.disable();
        assert!(s.is_passthrough());
        s.enable();
        assert!(!s.is_passthrough());
    }

    #[test]
    fn test_graph_stage_passthrough_type() {
        let s = make_stage(2, StageType::Passthrough);
        assert!(s.is_passthrough());
    }

    #[test]
    fn test_graph_add_stage_count() {
        let mut g = TranscodeGraph::new();
        g.add_stage(make_stage(1, StageType::Decode));
        g.add_stage(make_stage(2, StageType::Encode));
        assert_eq!(g.stage_count(), 2);
    }

    #[test]
    fn test_has_transcode_stage_true() {
        let mut g = TranscodeGraph::new();
        g.add_stage(make_stage(1, StageType::Decode));
        g.add_stage(make_stage(2, StageType::Encode));
        assert!(g.has_transcode_stage());
    }

    #[test]
    fn test_has_transcode_stage_false() {
        let mut g = TranscodeGraph::new();
        g.add_stage(make_stage(1, StageType::Passthrough));
        g.add_stage(make_stage(2, StageType::Filter));
        assert!(!g.has_transcode_stage());
    }

    #[test]
    fn test_is_fully_passthrough() {
        let mut g = TranscodeGraph::new();
        g.add_stage(make_stage(1, StageType::Passthrough));
        g.add_stage(make_stage(2, StageType::Analyse));
        assert!(g.is_fully_passthrough());
    }

    #[test]
    fn test_connect_valid() {
        let mut g = TranscodeGraph::new();
        g.add_stage(make_stage(1, StageType::Decode));
        g.add_stage(make_stage(2, StageType::Encode));
        assert!(g.connect(1, 2));
        assert_eq!(g.edges.len(), 1);
    }

    #[test]
    fn test_connect_invalid_id() {
        let mut g = TranscodeGraph::new();
        g.add_stage(make_stage(1, StageType::Decode));
        assert!(!g.connect(1, 99));
    }

    #[test]
    fn test_total_cost() {
        let mut g = TranscodeGraph::new();
        g.add_stage(make_stage(1, StageType::Decode).with_cost(10));
        g.add_stage(make_stage(2, StageType::Encode).with_cost(20));
        assert_eq!(g.total_cost(), 30);
    }

    #[test]
    fn test_remove_stage_removes_edges() {
        let mut g = TranscodeGraph::new();
        g.add_stage(make_stage(1, StageType::Decode));
        g.add_stage(make_stage(2, StageType::Encode));
        g.connect(1, 2);
        g.remove_stage(1);
        assert_eq!(g.stage_count(), 1);
        assert!(g.edges.is_empty());
    }

    #[test]
    fn test_find_stage() {
        let mut g = TranscodeGraph::new();
        g.add_stage(make_stage(42, StageType::Filter));
        let s = g.find_stage(42).expect("should succeed in test");
        assert_eq!(s.id, 42);
        assert!(g.find_stage(0).is_none());
    }
}
