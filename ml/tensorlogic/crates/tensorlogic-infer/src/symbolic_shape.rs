//! Symbolic shape support for TensorLogic.
//!
//! Enables shape inference for graphs with unknown or dynamic dimensions.
//! Uses a unification-based approach: symbolic names act as type variables
//! that get resolved when unified with concrete sizes.

use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;

/// A single tensor dimension — either a known size or a symbolic name.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SymbolicDim {
    /// A fixed, known size.
    Fixed(usize),
    /// A symbolic name (e.g., "batch", "seq_len", "N").
    Symbolic(Arc<str>),
    /// A product of two dimensions (e.g., batch * seq_len).
    Product(Box<SymbolicDim>, Box<SymbolicDim>),
}

impl SymbolicDim {
    pub fn fixed(n: usize) -> Self {
        SymbolicDim::Fixed(n)
    }

    pub fn symbolic(name: impl Into<Arc<str>>) -> Self {
        SymbolicDim::Symbolic(name.into())
    }

    pub fn product(a: SymbolicDim, b: SymbolicDim) -> Self {
        SymbolicDim::Product(Box::new(a), Box::new(b))
    }

    pub fn is_fixed(&self) -> bool {
        matches!(self, SymbolicDim::Fixed(_))
    }

    pub fn is_symbolic(&self) -> bool {
        matches!(self, SymbolicDim::Symbolic(_))
    }

    /// If this dimension is fully resolved, return its concrete value.
    pub fn concrete_value(&self) -> Option<usize> {
        match self {
            SymbolicDim::Fixed(n) => Some(*n),
            SymbolicDim::Symbolic(_) => None,
            SymbolicDim::Product(a, b) => {
                let va = a.concrete_value()?;
                let vb = b.concrete_value()?;
                Some(va * vb)
            }
        }
    }
}

impl std::fmt::Display for SymbolicDim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SymbolicDim::Fixed(n) => write!(f, "{}", n),
            SymbolicDim::Symbolic(s) => write!(f, "{}", s),
            SymbolicDim::Product(a, b) => write!(f, "({}*{})", a, b),
        }
    }
}

/// A tensor shape as a vector of symbolic dimensions.
pub type SymbolicShape = Vec<SymbolicDim>;

/// Constraints between symbolic dimensions for consistency checking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SymbolicShapeConstraint {
    /// Two dimensions must be equal.
    Equal(SymbolicDim, SymbolicDim),
    /// First dimension must be strictly greater than second.
    GreaterThan(SymbolicDim, SymbolicDim),
    /// First dimension must be a multiple of second.
    Multiple(SymbolicDim, SymbolicDim),
}

/// Error types for symbolic shape operations.
#[derive(Debug, Error)]
pub enum ShapeError {
    #[error("Dimension contradiction: cannot unify {0} with {1}")]
    Contradiction(String, String),
    #[error("Unresolved symbolic dimension: {0}")]
    Unresolved(String),
    #[error("Invalid einsum spec: {0}")]
    InvalidSpec(String),
    #[error("Arity mismatch: expected {expected} inputs, got {got}")]
    ArityMismatch { expected: usize, got: usize },
}

/// Unification environment for symbolic shapes.
///
/// Maintains a substitution map from symbolic names to resolved dimensions.
/// Uses union-find semantics: unifying a symbolic name binds it permanently.
#[derive(Debug, Default)]
pub struct SymbolicShapeEnv {
    /// Map from symbolic name → resolved SymbolicDim
    bindings: HashMap<Arc<str>, SymbolicDim>,
    /// Registered constraints
    constraints: Vec<SymbolicShapeConstraint>,
}

impl SymbolicShapeEnv {
    pub fn new() -> Self {
        Self::default()
    }

    /// Resolve a dimension through the substitution map.
    /// Returns the most-resolved form available given current bindings.
    pub fn resolve(&self, dim: &SymbolicDim) -> SymbolicDim {
        match dim {
            SymbolicDim::Symbolic(name) => {
                if let Some(bound) = self.bindings.get(name) {
                    self.resolve(bound)
                } else {
                    dim.clone()
                }
            }
            SymbolicDim::Product(a, b) => SymbolicDim::product(self.resolve(a), self.resolve(b)),
            SymbolicDim::Fixed(_) => dim.clone(),
        }
    }

    /// Try to get a concrete value for a dimension.
    pub fn concrete_value(&self, dim: &SymbolicDim) -> Option<usize> {
        self.resolve(dim).concrete_value()
    }

    /// Unify two dimensions. If both are Fixed, they must be equal.
    /// If one is Symbolic, it gets bound to the other.
    pub fn unify(&mut self, a: &SymbolicDim, b: &SymbolicDim) -> Result<SymbolicDim, ShapeError> {
        let ra = self.resolve(a);
        let rb = self.resolve(b);
        match (&ra, &rb) {
            (SymbolicDim::Fixed(x), SymbolicDim::Fixed(y)) => {
                if x == y {
                    Ok(ra)
                } else {
                    Err(ShapeError::Contradiction(
                        format!("{}", x),
                        format!("{}", y),
                    ))
                }
            }
            (SymbolicDim::Symbolic(name_a), SymbolicDim::Symbolic(name_b)) => {
                // Avoid self-binding: if both resolve to the same symbolic name, no-op
                if name_a == name_b {
                    Ok(ra)
                } else {
                    // Bind name_a → rb (pointing to name_b or its binding)
                    self.bindings.insert(name_a.clone(), rb.clone());
                    Ok(rb)
                }
            }
            (SymbolicDim::Symbolic(name), _) => {
                self.bindings.insert(name.clone(), rb.clone());
                Ok(rb)
            }
            (_, SymbolicDim::Symbolic(name)) => {
                self.bindings.insert(name.clone(), ra.clone());
                Ok(ra)
            }
            // Product × Fixed: try to resolve
            (SymbolicDim::Product(_, _), SymbolicDim::Fixed(_)) => {
                if let Some(va) = ra.concrete_value() {
                    if let Some(vb) = rb.concrete_value() {
                        if va == vb {
                            Ok(ra)
                        } else {
                            Err(ShapeError::Contradiction(
                                format!("{}", va),
                                format!("{}", vb),
                            ))
                        }
                    } else {
                        Ok(ra)
                    }
                } else {
                    // Cannot resolve product yet, store as constraint
                    self.add_constraint(SymbolicShapeConstraint::Equal(ra, rb));
                    Ok(SymbolicDim::symbolic("_unresolved"))
                }
            }
            (SymbolicDim::Fixed(_), SymbolicDim::Product(_, _)) => self.unify(b, a),
            _ => Ok(ra),
        }
    }

    /// Register a shape constraint for later consistency checking.
    pub fn add_constraint(&mut self, c: SymbolicShapeConstraint) {
        self.constraints.push(c);
    }

    /// Check that all registered constraints are satisfiable given current bindings.
    pub fn check_consistency(&self) -> bool {
        for c in &self.constraints {
            match c {
                SymbolicShapeConstraint::Equal(a, b) => {
                    if let (Some(va), Some(vb)) = (self.concrete_value(a), self.concrete_value(b)) {
                        if va != vb {
                            return false;
                        }
                    }
                }
                SymbolicShapeConstraint::GreaterThan(a, b) => {
                    if let (Some(va), Some(vb)) = (self.concrete_value(a), self.concrete_value(b)) {
                        if va <= vb {
                            return false;
                        }
                    }
                }
                SymbolicShapeConstraint::Multiple(a, b) => {
                    if let (Some(va), Some(vb)) = (self.concrete_value(a), self.concrete_value(b)) {
                        if vb == 0 || va % vb != 0 {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    /// Number of bindings currently in the environment.
    pub fn binding_count(&self) -> usize {
        self.bindings.len()
    }

    /// All symbolic names currently bound.
    pub fn bound_names(&self) -> impl Iterator<Item = &Arc<str>> {
        self.bindings.keys()
    }
}

/// Infer the output shape of an einsum operation given symbolic input shapes.
///
/// Uses the einsum spec notation: `"ij,jk->ik"` with inputs `[["M","K"],["K","N"]]`
/// produces `["M","N"]`.
///
/// # Rules
/// - Each index character maps to a SymbolicDim from its position in the corresponding input
/// - Shared indices that appear in multiple inputs are unified (must be equal)
/// - Output indices are collected in spec output order
pub fn propagate_einsum_shapes(
    spec: &str,
    input_shapes: &[SymbolicShape],
    env: &mut SymbolicShapeEnv,
) -> Result<SymbolicShape, ShapeError> {
    // Parse spec: "ij,jk->ik"
    let arrow_pos = spec
        .find("->")
        .ok_or_else(|| ShapeError::InvalidSpec(format!("missing '->' in einsum spec: {}", spec)))?;
    let inputs_part = &spec[..arrow_pos];
    let output_part = &spec[arrow_pos + 2..];

    let operand_specs: Vec<&str> = inputs_part.split(',').collect();
    if operand_specs.len() != input_shapes.len() {
        return Err(ShapeError::ArityMismatch {
            expected: operand_specs.len(),
            got: input_shapes.len(),
        });
    }

    // Build index → SymbolicDim map
    let mut index_map: HashMap<char, SymbolicDim> = HashMap::new();

    for (op_spec, shape) in operand_specs.iter().zip(input_shapes.iter()) {
        let chars: Vec<char> = op_spec.chars().filter(|c| c.is_alphabetic()).collect();
        if chars.len() != shape.len() {
            return Err(ShapeError::InvalidSpec(format!(
                "spec '{}' has {} indices but shape has {} dims",
                op_spec,
                chars.len(),
                shape.len()
            )));
        }
        for (ch, dim) in chars.iter().zip(shape.iter()) {
            if let Some(existing) = index_map.get(ch) {
                // Unify existing with new
                let unified = env.unify(existing, dim)?;
                index_map.insert(*ch, unified);
            } else {
                index_map.insert(*ch, env.resolve(dim));
            }
        }
    }

    // Collect output shape
    let output_chars: Vec<char> = output_part.chars().filter(|c| c.is_alphabetic()).collect();
    let mut out_shape = Vec::with_capacity(output_chars.len());
    for ch in output_chars {
        let dim = index_map
            .get(&ch)
            .cloned()
            .unwrap_or_else(|| SymbolicDim::symbolic(format!("_out_{}", ch)));
        out_shape.push(env.resolve(&dim));
    }

    Ok(out_shape)
}

/// Convenience: propagate shapes through a chain of einsum operations.
pub fn propagate_chain(
    specs: &[&str],
    initial_shapes: &[SymbolicShape],
    env: &mut SymbolicShapeEnv,
) -> Result<Vec<SymbolicShape>, ShapeError> {
    let mut results = Vec::new();
    let mut current_shapes: Vec<SymbolicShape> = initial_shapes.to_vec();
    for spec in specs {
        let out = propagate_einsum_shapes(spec, &current_shapes, env)?;
        results.push(out.clone());
        // For chains, the output becomes the first input of next (simplified)
        current_shapes = vec![out];
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    // SymbolicDim tests
    #[test]
    fn test_fixed_dim_equality() {
        assert_eq!(SymbolicDim::fixed(3), SymbolicDim::fixed(3));
        assert_ne!(SymbolicDim::fixed(3), SymbolicDim::fixed(4));
    }

    #[test]
    fn test_symbolic_dim_display() {
        let d = SymbolicDim::symbolic("batch");
        assert_eq!(format!("{}", d), "batch");
    }

    #[test]
    fn test_fixed_dim_concrete_value() {
        assert_eq!(SymbolicDim::fixed(42).concrete_value(), Some(42));
    }

    #[test]
    fn test_symbolic_dim_no_concrete_value() {
        assert_eq!(SymbolicDim::symbolic("N").concrete_value(), None);
    }

    #[test]
    fn test_product_dim_resolves_when_both_fixed() {
        let p = SymbolicDim::product(SymbolicDim::fixed(3), SymbolicDim::fixed(4));
        assert_eq!(p.concrete_value(), Some(12));
    }

    #[test]
    fn test_product_dim_unresolved_when_symbolic() {
        let p = SymbolicDim::product(SymbolicDim::symbolic("N"), SymbolicDim::fixed(4));
        assert_eq!(p.concrete_value(), None);
    }

    // SymbolicShapeEnv unification tests
    #[test]
    fn test_unify_fixed_same() -> Result<(), ShapeError> {
        let mut env = SymbolicShapeEnv::new();
        let result = env.unify(&SymbolicDim::fixed(5), &SymbolicDim::fixed(5))?;
        assert_eq!(result, SymbolicDim::fixed(5));
        Ok(())
    }

    #[test]
    fn test_unify_fixed_contradiction() {
        let mut env = SymbolicShapeEnv::new();
        let result = env.unify(&SymbolicDim::fixed(3), &SymbolicDim::fixed(4));
        assert!(result.is_err());
    }

    #[test]
    fn test_unify_symbolic_binds_to_fixed() -> Result<(), ShapeError> {
        let mut env = SymbolicShapeEnv::new();
        env.unify(&SymbolicDim::symbolic("N"), &SymbolicDim::fixed(7))?;
        assert_eq!(env.concrete_value(&SymbolicDim::symbolic("N")), Some(7));
        Ok(())
    }

    #[test]
    fn test_unify_fixed_binds_symbolic() -> Result<(), ShapeError> {
        let mut env = SymbolicShapeEnv::new();
        env.unify(&SymbolicDim::fixed(4), &SymbolicDim::symbolic("M"))?;
        assert_eq!(env.concrete_value(&SymbolicDim::symbolic("M")), Some(4));
        Ok(())
    }

    #[test]
    fn test_unify_two_symbolics() -> Result<(), ShapeError> {
        let mut env = SymbolicShapeEnv::new();
        env.unify(&SymbolicDim::symbolic("A"), &SymbolicDim::symbolic("B"))?;
        // Both should now resolve to the same thing
        let ra = env.resolve(&SymbolicDim::symbolic("A"));
        let rb = env.resolve(&SymbolicDim::symbolic("B"));
        // They should both resolve to the same concrete or both remain symbolic
        assert_eq!(ra, rb);
        Ok(())
    }

    #[test]
    fn test_resolve_chain() -> Result<(), ShapeError> {
        let mut env = SymbolicShapeEnv::new();
        env.unify(&SymbolicDim::symbolic("A"), &SymbolicDim::symbolic("B"))?;
        env.unify(&SymbolicDim::symbolic("B"), &SymbolicDim::fixed(10))?;
        assert_eq!(env.concrete_value(&SymbolicDim::symbolic("A")), Some(10));
        Ok(())
    }

    #[test]
    fn test_binding_count() -> Result<(), ShapeError> {
        let mut env = SymbolicShapeEnv::new();
        assert_eq!(env.binding_count(), 0);
        env.unify(&SymbolicDim::symbolic("N"), &SymbolicDim::fixed(5))?;
        assert_eq!(env.binding_count(), 1);
        Ok(())
    }

    // SymbolicShapeConstraint tests
    #[test]
    fn test_constraint_consistency_equal() {
        let mut env = SymbolicShapeEnv::new();
        env.add_constraint(SymbolicShapeConstraint::Equal(
            SymbolicDim::fixed(3),
            SymbolicDim::fixed(3),
        ));
        assert!(env.check_consistency());
    }

    #[test]
    fn test_constraint_inconsistency_equal() {
        let mut env = SymbolicShapeEnv::new();
        env.add_constraint(SymbolicShapeConstraint::Equal(
            SymbolicDim::fixed(3),
            SymbolicDim::fixed(5),
        ));
        assert!(!env.check_consistency());
    }

    #[test]
    fn test_constraint_greater_than() {
        let mut env = SymbolicShapeEnv::new();
        env.add_constraint(SymbolicShapeConstraint::GreaterThan(
            SymbolicDim::fixed(10),
            SymbolicDim::fixed(5),
        ));
        assert!(env.check_consistency());
    }

    #[test]
    fn test_constraint_multiple() {
        let mut env = SymbolicShapeEnv::new();
        env.add_constraint(SymbolicShapeConstraint::Multiple(
            SymbolicDim::fixed(12),
            SymbolicDim::fixed(4),
        ));
        assert!(env.check_consistency());
    }

    // Einsum propagation tests
    #[test]
    fn test_propagate_matmul_symbolic() -> Result<(), ShapeError> {
        let mut env = SymbolicShapeEnv::new();
        let shape_a = vec![SymbolicDim::symbolic("M"), SymbolicDim::symbolic("K")];
        let shape_b = vec![SymbolicDim::symbolic("K"), SymbolicDim::symbolic("N")];
        let out = propagate_einsum_shapes("ij,jk->ik", &[shape_a, shape_b], &mut env)?;
        assert_eq!(out.len(), 2);
        assert_eq!(format!("{}", out[0]), "M");
        assert_eq!(format!("{}", out[1]), "N");
        Ok(())
    }

    #[test]
    fn test_propagate_matmul_fixed() -> Result<(), ShapeError> {
        let mut env = SymbolicShapeEnv::new();
        let shape_a = vec![SymbolicDim::fixed(4), SymbolicDim::fixed(3)];
        let shape_b = vec![SymbolicDim::fixed(3), SymbolicDim::fixed(5)];
        let out = propagate_einsum_shapes("ij,jk->ik", &[shape_a, shape_b], &mut env)?;
        assert_eq!(out[0].concrete_value(), Some(4));
        assert_eq!(out[1].concrete_value(), Some(5));
        Ok(())
    }

    #[test]
    fn test_propagate_contraction_unifies_k() -> Result<(), ShapeError> {
        let mut env = SymbolicShapeEnv::new();
        let shape_a = vec![SymbolicDim::symbolic("M"), SymbolicDim::symbolic("K")];
        let shape_b = vec![SymbolicDim::symbolic("K"), SymbolicDim::fixed(5)];
        propagate_einsum_shapes("ij,jk->ik", &[shape_a, shape_b], &mut env)?;
        // K should be unbound (it's contracted, not in output)
        Ok(())
    }

    #[test]
    fn test_propagate_inner_product() -> Result<(), ShapeError> {
        let mut env = SymbolicShapeEnv::new();
        let shape_a = vec![SymbolicDim::symbolic("N")];
        let shape_b = vec![SymbolicDim::symbolic("N")];
        let out = propagate_einsum_shapes("i,i->", &[shape_a, shape_b], &mut env)?;
        assert_eq!(out.len(), 0); // scalar output
        Ok(())
    }

    #[test]
    fn test_propagate_batch_matmul() -> Result<(), ShapeError> {
        let mut env = SymbolicShapeEnv::new();
        let shape_a = vec![
            SymbolicDim::symbolic("B"),
            SymbolicDim::symbolic("M"),
            SymbolicDim::symbolic("K"),
        ];
        let shape_b = vec![
            SymbolicDim::symbolic("B"),
            SymbolicDim::symbolic("K"),
            SymbolicDim::symbolic("N"),
        ];
        let out = propagate_einsum_shapes("bij,bjk->bik", &[shape_a, shape_b], &mut env)?;
        assert_eq!(out.len(), 3);
        assert_eq!(format!("{}", out[0]), "B");
        Ok(())
    }

    #[test]
    fn test_propagate_arity_mismatch_error() {
        let mut env = SymbolicShapeEnv::new();
        let shape_a = vec![SymbolicDim::fixed(3), SymbolicDim::fixed(4)];
        // Spec expects 2 inputs but we provide only 1
        let result = propagate_einsum_shapes("ij,jk->ik", &[shape_a], &mut env);
        assert!(matches!(result, Err(ShapeError::ArityMismatch { .. })));
    }

    #[test]
    fn test_propagate_missing_arrow_error() {
        let mut env = SymbolicShapeEnv::new();
        let shape = vec![SymbolicDim::fixed(3)];
        let result = propagate_einsum_shapes("i,j", &[shape.clone(), shape], &mut env);
        assert!(matches!(result, Err(ShapeError::InvalidSpec(_))));
    }
}
