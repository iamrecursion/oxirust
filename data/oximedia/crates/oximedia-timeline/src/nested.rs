//! Nested sequence support.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use uuid::Uuid;

use crate::error::{TimelineError, TimelineResult};
use crate::timeline::Timeline;

/// Sequence identifier.
pub type SequenceId = Uuid;

/// Nested sequence reference.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SequenceReference {
    /// ID of the referenced sequence.
    pub sequence_id: SequenceId,
    /// Name of the referenced sequence.
    pub name: String,
}

impl SequenceReference {
    /// Creates a new sequence reference.
    #[must_use]
    pub fn new(sequence_id: SequenceId, name: String) -> Self {
        Self { sequence_id, name }
    }
}

/// Sequence dependency graph.
pub struct SequenceDependencyGraph {
    /// Dependencies: `sequence_id` -> set of sequences it depends on.
    dependencies: HashMap<SequenceId, HashSet<SequenceId>>,
}

impl SequenceDependencyGraph {
    /// Creates a new dependency graph.
    #[must_use]
    pub fn new() -> Self {
        Self {
            dependencies: HashMap::new(),
        }
    }

    /// Adds a dependency.
    pub fn add_dependency(&mut self, sequence: SequenceId, depends_on: SequenceId) {
        self.dependencies
            .entry(sequence)
            .or_default()
            .insert(depends_on);
    }

    /// Removes a dependency.
    pub fn remove_dependency(&mut self, sequence: SequenceId, depends_on: SequenceId) {
        if let Some(deps) = self.dependencies.get_mut(&sequence) {
            deps.remove(&depends_on);
        }
    }

    /// Checks for circular dependencies.
    ///
    /// # Errors
    ///
    /// Returns error if circular dependency detected.
    pub fn check_circular(&self, sequence: SequenceId) -> TimelineResult<()> {
        let mut visited = HashSet::new();
        let mut stack = HashSet::new();
        self.check_circular_helper(sequence, &mut visited, &mut stack)
    }

    /// Helper for circular dependency check (DFS).
    fn check_circular_helper(
        &self,
        sequence: SequenceId,
        visited: &mut HashSet<SequenceId>,
        stack: &mut HashSet<SequenceId>,
    ) -> TimelineResult<()> {
        if stack.contains(&sequence) {
            return Err(TimelineError::CircularDependency);
        }

        if visited.contains(&sequence) {
            return Ok(());
        }

        visited.insert(sequence);
        stack.insert(sequence);

        if let Some(deps) = self.dependencies.get(&sequence) {
            for dep in deps {
                self.check_circular_helper(*dep, visited, stack)?;
            }
        }

        stack.remove(&sequence);
        Ok(())
    }

    /// Gets all dependencies for a sequence (recursive).
    #[must_use]
    pub fn get_all_dependencies(&self, sequence: SequenceId) -> HashSet<SequenceId> {
        let mut result = HashSet::new();
        self.collect_dependencies(sequence, &mut result);
        result
    }

    /// Helper to collect all dependencies recursively.
    fn collect_dependencies(&self, sequence: SequenceId, result: &mut HashSet<SequenceId>) {
        if let Some(deps) = self.dependencies.get(&sequence) {
            for dep in deps {
                if result.insert(*dep) {
                    self.collect_dependencies(*dep, result);
                }
            }
        }
    }

    /// Gets direct dependencies for a sequence.
    #[must_use]
    pub fn get_direct_dependencies(&self, sequence: SequenceId) -> HashSet<SequenceId> {
        self.dependencies
            .get(&sequence)
            .cloned()
            .unwrap_or_default()
    }

    /// Gets sequences that depend on this sequence.
    #[must_use]
    pub fn get_dependents(&self, sequence: SequenceId) -> HashSet<SequenceId> {
        let mut result = HashSet::new();
        for (seq, deps) in &self.dependencies {
            if deps.contains(&sequence) {
                result.insert(*seq);
            }
        }
        result
    }

    /// Removes all dependencies for a sequence.
    pub fn remove_sequence(&mut self, sequence: SequenceId) {
        self.dependencies.remove(&sequence);
        // Also remove from other sequences' dependencies
        for deps in self.dependencies.values_mut() {
            deps.remove(&sequence);
        }
    }
}

impl Default for SequenceDependencyGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Sequence manager for nested sequences.
pub struct SequenceManager {
    /// All sequences.
    sequences: HashMap<SequenceId, Arc<Timeline>>,
    /// Dependency graph.
    dependencies: SequenceDependencyGraph,
}

impl SequenceManager {
    /// Creates a new sequence manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            sequences: HashMap::new(),
            dependencies: SequenceDependencyGraph::new(),
        }
    }

    /// Adds a sequence.
    pub fn add_sequence(&mut self, timeline: Arc<Timeline>) {
        let sequence_id = timeline.id;
        self.sequences.insert(sequence_id, timeline);
    }

    /// Removes a sequence.
    ///
    /// # Errors
    ///
    /// Returns error if sequence is used by other sequences.
    pub fn remove_sequence(&mut self, sequence_id: SequenceId) -> TimelineResult<Arc<Timeline>> {
        // Check if any sequences depend on this one
        let dependents = self.dependencies.get_dependents(sequence_id);
        if !dependents.is_empty() {
            return Err(TimelineError::Other(format!(
                "Cannot remove sequence: {} other sequences depend on it",
                dependents.len()
            )));
        }

        self.dependencies.remove_sequence(sequence_id);
        self.sequences
            .remove(&sequence_id)
            .ok_or_else(|| TimelineError::Other("Sequence not found".to_string()))
    }

    /// Gets a sequence.
    #[must_use]
    pub fn get_sequence(&self, sequence_id: SequenceId) -> Option<Arc<Timeline>> {
        self.sequences.get(&sequence_id).cloned()
    }

    /// Adds a sequence reference.
    ///
    /// # Errors
    ///
    /// Returns error if circular dependency detected.
    pub fn add_reference(&mut self, parent: SequenceId, child: SequenceId) -> TimelineResult<()> {
        self.dependencies.add_dependency(parent, child);
        self.dependencies.check_circular(parent)?;
        Ok(())
    }

    /// Removes a sequence reference.
    pub fn remove_reference(&mut self, parent: SequenceId, child: SequenceId) {
        self.dependencies.remove_dependency(parent, child);
    }

    /// Gets all dependencies for a sequence.
    #[must_use]
    pub fn get_dependencies(&self, sequence_id: SequenceId) -> HashSet<SequenceId> {
        self.dependencies.get_all_dependencies(sequence_id)
    }

    /// Gets sequences that depend on this sequence.
    #[must_use]
    pub fn get_dependents(&self, sequence_id: SequenceId) -> HashSet<SequenceId> {
        self.dependencies.get_dependents(sequence_id)
    }

    /// Lists all sequences.
    #[must_use]
    pub fn list_sequences(&self) -> Vec<SequenceId> {
        self.sequences.keys().copied().collect()
    }

    /// Resolves a nested sequence hierarchy.
    ///
    /// # Errors
    ///
    /// Returns error if resolution fails.
    pub fn resolve_hierarchy(&self, sequence_id: SequenceId) -> TimelineResult<Vec<SequenceId>> {
        let mut result = vec![sequence_id];
        let deps = self.dependencies.get_all_dependencies(sequence_id);

        // Topological sort of dependencies
        let mut visited = HashSet::new();
        for dep in deps {
            if !visited.contains(&dep) {
                self.topological_sort(dep, &mut visited, &mut result)?;
            }
        }

        Ok(result)
    }

    /// Helper for topological sort.
    fn topological_sort(
        &self,
        sequence: SequenceId,
        visited: &mut HashSet<SequenceId>,
        result: &mut Vec<SequenceId>,
    ) -> TimelineResult<()> {
        visited.insert(sequence);

        let deps = self.dependencies.get_direct_dependencies(sequence);
        for dep in deps {
            if !visited.contains(&dep) {
                self.topological_sort(dep, visited, result)?;
            }
        }

        result.push(sequence);
        Ok(())
    }
}

impl Default for SequenceManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::Rational;

    fn create_test_timeline(name: &str) -> Arc<Timeline> {
        Arc::new(Timeline::new(name, Rational::new(24, 1), 48000).expect("should succeed in test"))
    }

    #[test]
    fn test_dependency_graph_add() {
        let mut graph = SequenceDependencyGraph::new();
        let seq1 = Uuid::new_v4();
        let seq2 = Uuid::new_v4();
        graph.add_dependency(seq1, seq2);
        let deps = graph.get_direct_dependencies(seq1);
        assert_eq!(deps.len(), 1);
        assert!(deps.contains(&seq2));
    }

    #[test]
    fn test_dependency_graph_remove() {
        let mut graph = SequenceDependencyGraph::new();
        let seq1 = Uuid::new_v4();
        let seq2 = Uuid::new_v4();
        graph.add_dependency(seq1, seq2);
        graph.remove_dependency(seq1, seq2);
        let deps = graph.get_direct_dependencies(seq1);
        assert!(deps.is_empty());
    }

    #[test]
    fn test_dependency_graph_circular() {
        let mut graph = SequenceDependencyGraph::new();
        let seq1 = Uuid::new_v4();
        let seq2 = Uuid::new_v4();
        graph.add_dependency(seq1, seq2);
        graph.add_dependency(seq2, seq1);
        assert!(graph.check_circular(seq1).is_err());
    }

    #[test]
    fn test_dependency_graph_no_circular() {
        let mut graph = SequenceDependencyGraph::new();
        let seq1 = Uuid::new_v4();
        let seq2 = Uuid::new_v4();
        let seq3 = Uuid::new_v4();
        graph.add_dependency(seq1, seq2);
        graph.add_dependency(seq2, seq3);
        assert!(graph.check_circular(seq1).is_ok());
    }

    #[test]
    fn test_dependency_graph_all_dependencies() {
        let mut graph = SequenceDependencyGraph::new();
        let seq1 = Uuid::new_v4();
        let seq2 = Uuid::new_v4();
        let seq3 = Uuid::new_v4();
        graph.add_dependency(seq1, seq2);
        graph.add_dependency(seq2, seq3);

        let all_deps = graph.get_all_dependencies(seq1);
        assert_eq!(all_deps.len(), 2);
        assert!(all_deps.contains(&seq2));
        assert!(all_deps.contains(&seq3));
    }

    #[test]
    fn test_dependency_graph_dependents() {
        let mut graph = SequenceDependencyGraph::new();
        let seq1 = Uuid::new_v4();
        let seq2 = Uuid::new_v4();
        let seq3 = Uuid::new_v4();
        graph.add_dependency(seq1, seq3);
        graph.add_dependency(seq2, seq3);

        let dependents = graph.get_dependents(seq3);
        assert_eq!(dependents.len(), 2);
        assert!(dependents.contains(&seq1));
        assert!(dependents.contains(&seq2));
    }

    #[test]
    fn test_sequence_manager_add() {
        let mut manager = SequenceManager::new();
        let timeline = create_test_timeline("Test");
        let id = timeline.id;
        manager.add_sequence(timeline);
        assert!(manager.get_sequence(id).is_some());
    }

    #[test]
    fn test_sequence_manager_remove() {
        let mut manager = SequenceManager::new();
        let timeline = create_test_timeline("Test");
        let id = timeline.id;
        manager.add_sequence(timeline);
        assert!(manager.remove_sequence(id).is_ok());
        assert!(manager.get_sequence(id).is_none());
    }

    #[test]
    fn test_sequence_manager_add_reference() {
        let mut manager = SequenceManager::new();
        let timeline1 = create_test_timeline("Timeline1");
        let timeline2 = create_test_timeline("Timeline2");
        let id1 = timeline1.id;
        let id2 = timeline2.id;
        manager.add_sequence(timeline1);
        manager.add_sequence(timeline2);
        assert!(manager.add_reference(id1, id2).is_ok());
    }

    #[test]
    fn test_sequence_manager_circular_reference() {
        let mut manager = SequenceManager::new();
        let timeline1 = create_test_timeline("Timeline1");
        let timeline2 = create_test_timeline("Timeline2");
        let id1 = timeline1.id;
        let id2 = timeline2.id;
        manager.add_sequence(timeline1);
        manager.add_sequence(timeline2);
        manager
            .add_reference(id1, id2)
            .expect("should succeed in test");
        assert!(manager.add_reference(id2, id1).is_err());
    }

    #[test]
    fn test_sequence_manager_remove_with_dependents() {
        let mut manager = SequenceManager::new();
        let timeline1 = create_test_timeline("Timeline1");
        let timeline2 = create_test_timeline("Timeline2");
        let id1 = timeline1.id;
        let id2 = timeline2.id;
        manager.add_sequence(timeline1);
        manager.add_sequence(timeline2);
        manager
            .add_reference(id1, id2)
            .expect("should succeed in test");
        // Cannot remove timeline2 because timeline1 depends on it
        assert!(manager.remove_sequence(id2).is_err());
    }

    #[test]
    fn test_sequence_manager_list() {
        let mut manager = SequenceManager::new();
        let timeline1 = create_test_timeline("Timeline1");
        let timeline2 = create_test_timeline("Timeline2");
        manager.add_sequence(timeline1);
        manager.add_sequence(timeline2);
        assert_eq!(manager.list_sequences().len(), 2);
    }
}
