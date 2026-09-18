//! Output routing and matrix configuration for broadcast video switchers.
//!
//! Models the output side of a video switcher: physical output connectors,
//! their types, and a matrix that maps switcher buses to output connectors.

#![allow(dead_code)]

/// The type (purpose) of a switcher output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputType {
    /// Program (on-air) output.
    Program,
    /// Preview (off-air) output.
    Preview,
    /// Auxiliary output independently routed.
    Aux,
    /// Clean feed (program minus keyers).
    CleanFeed,
    /// Multi-viewer composite output.
    Multiview,
    /// Record / tape output.
    Record,
}

impl OutputType {
    /// Returns `true` if this output carries the preview signal.
    pub fn is_preview(&self) -> bool {
        matches!(self, OutputType::Preview)
    }

    /// Returns `true` if this output is an on-air program output.
    pub fn is_program(&self) -> bool {
        matches!(self, OutputType::Program)
    }

    /// Returns `true` if this output can be freely re-routed.
    pub fn is_routable(&self) -> bool {
        matches!(self, OutputType::Aux | OutputType::Record)
    }

    /// Return a short descriptive label.
    pub fn label(&self) -> &'static str {
        match self {
            OutputType::Program => "PGM",
            OutputType::Preview => "PVW",
            OutputType::Aux => "AUX",
            OutputType::CleanFeed => "CLN",
            OutputType::Multiview => "MV",
            OutputType::Record => "REC",
        }
    }
}

/// Configuration for a single physical output connector.
#[derive(Debug, Clone)]
pub struct OutputConfig {
    /// 1-based output number.
    pub number: usize,
    /// Custom human-readable label.
    pub name: String,
    /// Signal type for this output.
    pub output_type: OutputType,
    /// Whether the output is currently active.
    pub active: bool,
}

impl OutputConfig {
    /// Create a new `OutputConfig`.
    pub fn new(number: usize, name: impl Into<String>, output_type: OutputType) -> Self {
        Self {
            number,
            name: name.into(),
            output_type,
            active: true,
        }
    }

    /// Return the label, falling back to the output type label when empty.
    pub fn label(&self) -> &str {
        if self.name.trim().is_empty() {
            self.output_type.label()
        } else {
            &self.name
        }
    }

    /// Returns `true` if the output configuration is valid.
    pub fn is_valid(&self) -> bool {
        self.number > 0
    }

    /// Deactivate this output.
    pub fn deactivate(&mut self) {
        self.active = false;
    }
}

/// A routing matrix that maps source bus IDs to output connectors.
///
/// Source IDs are arbitrary (e.g. input slot numbers, M/E bus IDs);
/// output IDs are 1-based output connector numbers.
#[derive(Debug, Default)]
pub struct OutputMatrix {
    /// Map from output number → source bus ID.
    assignments: std::collections::HashMap<usize, usize>,
}

impl OutputMatrix {
    /// Create a new empty `OutputMatrix`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Assign `source` to `output_number`.
    ///
    /// Returns the previously assigned source, if any.
    pub fn assign(&mut self, output_number: usize, source: usize) -> Option<usize> {
        self.assignments.insert(output_number, source)
    }

    /// Clear the assignment for `output_number`.
    ///
    /// Returns the removed source, if any.
    pub fn clear_assignment(&mut self, output_number: usize) -> Option<usize> {
        self.assignments.remove(&output_number)
    }

    /// Return the source currently assigned to `output_number`.
    pub fn get_assignment(&self, output_number: usize) -> Option<usize> {
        self.assignments.get(&output_number).copied()
    }

    /// Return the total number of active assignments.
    pub fn assignment_count(&self) -> usize {
        self.assignments.len()
    }

    /// Return all outputs that route the given source.
    pub fn outputs_for_source(&self, source: usize) -> Vec<usize> {
        self.assignments
            .iter()
            .filter_map(|(&out, &src)| if src == source { Some(out) } else { None })
            .collect()
    }

    /// Clear all assignments.
    pub fn clear_all(&mut self) {
        self.assignments.clear();
    }

    /// Apply a default mapping: output `n` → source `n`.
    pub fn apply_passthrough(&mut self, output_count: usize) {
        for n in 1..=output_count {
            self.assignments.insert(n, n);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_type_is_preview() {
        assert!(OutputType::Preview.is_preview());
        assert!(!OutputType::Program.is_preview());
    }

    #[test]
    fn test_output_type_is_program() {
        assert!(OutputType::Program.is_program());
        assert!(!OutputType::Aux.is_program());
    }

    #[test]
    fn test_output_type_is_routable() {
        assert!(OutputType::Aux.is_routable());
        assert!(OutputType::Record.is_routable());
        assert!(!OutputType::Program.is_routable());
        assert!(!OutputType::Preview.is_routable());
    }

    #[test]
    fn test_output_type_label() {
        assert_eq!(OutputType::Program.label(), "PGM");
        assert_eq!(OutputType::Preview.label(), "PVW");
        assert_eq!(OutputType::Aux.label(), "AUX");
        assert_eq!(OutputType::Multiview.label(), "MV");
    }

    #[test]
    fn test_output_config_label_custom() {
        let cfg = OutputConfig::new(1, "Monitor A", OutputType::Aux);
        assert_eq!(cfg.label(), "Monitor A");
    }

    #[test]
    fn test_output_config_label_fallback() {
        let cfg = OutputConfig::new(1, "   ", OutputType::Preview);
        assert_eq!(cfg.label(), "PVW");
    }

    #[test]
    fn test_output_config_is_valid() {
        let cfg = OutputConfig::new(1, "PGM", OutputType::Program);
        assert!(cfg.is_valid());
    }

    #[test]
    fn test_output_config_invalid_zero() {
        let cfg = OutputConfig::new(0, "Bad", OutputType::Program);
        assert!(!cfg.is_valid());
    }

    #[test]
    fn test_output_config_deactivate() {
        let mut cfg = OutputConfig::new(1, "Out", OutputType::Aux);
        assert!(cfg.active);
        cfg.deactivate();
        assert!(!cfg.active);
    }

    #[test]
    fn test_output_matrix_assign_and_get() {
        let mut matrix = OutputMatrix::new();
        matrix.assign(1, 5);
        assert_eq!(matrix.get_assignment(1), Some(5));
    }

    #[test]
    fn test_output_matrix_assignment_count() {
        let mut matrix = OutputMatrix::new();
        matrix.assign(1, 1);
        matrix.assign(2, 3);
        assert_eq!(matrix.assignment_count(), 2);
    }

    #[test]
    fn test_output_matrix_clear_assignment() {
        let mut matrix = OutputMatrix::new();
        matrix.assign(1, 5);
        let removed = matrix.clear_assignment(1);
        assert_eq!(removed, Some(5));
        assert_eq!(matrix.assignment_count(), 0);
    }

    #[test]
    fn test_output_matrix_outputs_for_source() {
        let mut matrix = OutputMatrix::new();
        matrix.assign(1, 3);
        matrix.assign(2, 3);
        matrix.assign(3, 5);
        let mut outs = matrix.outputs_for_source(3);
        outs.sort();
        assert_eq!(outs, vec![1, 2]);
    }

    #[test]
    fn test_output_matrix_passthrough() {
        let mut matrix = OutputMatrix::new();
        matrix.apply_passthrough(4);
        assert_eq!(matrix.assignment_count(), 4);
        for n in 1..=4 {
            assert_eq!(matrix.get_assignment(n), Some(n));
        }
    }

    #[test]
    fn test_output_matrix_clear_all() {
        let mut matrix = OutputMatrix::new();
        matrix.apply_passthrough(3);
        matrix.clear_all();
        assert_eq!(matrix.assignment_count(), 0);
    }
}
