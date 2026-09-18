//! Common Subexpression Elimination (CSE)
//!
//! This module implements CSE optimization to detect and merge duplicate computations
//! in the computation graph.

use crate::{error::AutogradError, Float, Result};
use std::collections::{HashMap, HashSet};

/// Expression fingerprint for CSE
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct ExprFingerprint {
    /// Operation name
    pub op_name: String,
    /// Input IDs (sorted for commutative operations)
    pub inputs: Vec<usize>,
    /// Operation attributes (e.g., axis, keepdims)
    pub attributes: Vec<String>,
}

impl ExprFingerprint {
    /// Create a new expression fingerprint
    pub fn new(op_name: String, inputs: Vec<usize>, attributes: Vec<String>) -> Self {
        Self {
            op_name,
            inputs,
            attributes,
        }
    }

    /// Create fingerprint for a commutative operation
    pub fn commutative(op_name: String, mut inputs: Vec<usize>, attributes: Vec<String>) -> Self {
        inputs.sort_unstable();
        Self {
            op_name,
            inputs,
            attributes,
        }
    }
}

/// CSE optimizer
pub struct CSEOptimizer {
    /// Map from fingerprint to existing computation ID
    fingerprint_map: HashMap<ExprFingerprint, usize>,
    /// Set of replaced computations
    replaced: HashSet<usize>,
    /// Statistics
    num_eliminations: usize,
}

impl CSEOptimizer {
    /// Create a new CSE optimizer
    pub fn new() -> Self {
        Self {
            fingerprint_map: HashMap::new(),
            replaced: HashSet::new(),
            num_eliminations: 0,
        }
    }

    /// Check if an expression can be eliminated
    pub fn find_duplicate(&self, fingerprint: &ExprFingerprint) -> Option<usize> {
        self.fingerprint_map.get(fingerprint).copied()
    }

    /// Record an expression
    pub fn record(&mut self, fingerprint: ExprFingerprint, id: usize) {
        self.fingerprint_map.insert(fingerprint, id);
    }

    /// Mark a computation as replaced
    pub fn mark_replaced(&mut self, id: usize) {
        self.replaced.insert(id);
        self.num_eliminations += 1;
    }

    /// Check if a computation has been replaced
    pub fn is_replaced(&self, id: usize) -> bool {
        self.replaced.contains(&id)
    }

    /// Get number of eliminations
    pub fn num_eliminations(&self) -> usize {
        self.num_eliminations
    }

    /// Clear the optimizer state
    pub fn clear(&mut self) {
        self.fingerprint_map.clear();
        self.replaced.clear();
        self.num_eliminations = 0;
    }
}

impl Default for CSEOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// CSE pass result
#[derive(Debug, Clone)]
pub struct CSEResult {
    /// Number of subexpressions eliminated
    pub eliminations: usize,
    /// Memory saved (bytes)
    pub memory_saved: usize,
    /// Computation cost saved (arbitrary units)
    pub cost_saved: usize,
}

impl CSEResult {
    /// Create a new CSE result
    pub fn new(eliminations: usize, memory_saved: usize, cost_saved: usize) -> Self {
        Self {
            eliminations,
            memory_saved,
            cost_saved,
        }
    }

    /// Check if any optimizations were applied
    pub fn has_changes(&self) -> bool {
        self.eliminations > 0
    }
}

/// Perform CSE optimization on a computation graph
pub fn eliminate_common_subexpressions<T: Float>(
    _graph: &mut Vec<ExprFingerprint>,
) -> Result<CSEResult> {
    let mut optimizer = CSEOptimizer::new();
    let mut replacements = 0;

    // This is a simplified implementation
    // In production, would traverse actual computation graph

    for (id, expr) in _graph.iter().enumerate() {
        if let Some(existing_id) = optimizer.find_duplicate(expr) {
            // Found duplicate - mark for replacement
            optimizer.mark_replaced(id);
            replacements += 1;

            // Would update graph to use existing_id instead
            let _ = existing_id; // Placeholder
        } else {
            // Record this expression
            optimizer.record(expr.clone(), id);
        }
    }

    Ok(CSEResult::new(
        replacements,
        replacements * 1024, // Estimated memory saved
        replacements * 100,  // Estimated computation saved
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expression_fingerprint() {
        let fp1 = ExprFingerprint::new("add".to_string(), vec![1, 2], vec![]);
        let fp2 = ExprFingerprint::new("add".to_string(), vec![1, 2], vec![]);

        assert_eq!(fp1, fp2);
    }

    #[test]
    fn test_commutative_fingerprint() {
        let fp1 = ExprFingerprint::commutative("add".to_string(), vec![2, 1], vec![]);
        let fp2 = ExprFingerprint::commutative("add".to_string(), vec![1, 2], vec![]);

        // Should be equal after sorting
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn test_cse_optimizer() {
        let mut optimizer = CSEOptimizer::new();

        let fp = ExprFingerprint::new("mul".to_string(), vec![1, 2], vec![]);

        // Record first occurrence
        optimizer.record(fp.clone(), 100);

        // Should find duplicate
        assert_eq!(optimizer.find_duplicate(&fp), Some(100));

        // Mark as replaced
        optimizer.mark_replaced(101);
        assert!(optimizer.is_replaced(101));
        assert_eq!(optimizer.num_eliminations(), 1);
    }

    #[test]
    fn test_cse_result() {
        let result = CSEResult::new(5, 5120, 500);

        assert!(result.has_changes());
        assert_eq!(result.eliminations, 5);
        assert_eq!(result.memory_saved, 5120);
    }
}
