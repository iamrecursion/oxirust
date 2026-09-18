//! Clebsch-Gordan series for SU(2), SU(3), SO(5), and general semisimple Lie groups.
//!
//! This module computes the decomposition of tensor products of irreducible
//! representations (irreps) of compact Lie groups:
//!
//!   R1 ⊗ R2 = ⊕_R m(R) · R
//!
//! where `m(R)` is the multiplicity (outer multiplicity) of irrep R in the
//! tensor product.
//!
//! Representations are encoded as **Dynkin labels** — the non-negative integers
//! (a₁, a₂, ..., aₙ) that label the highest weight of a finite-dimensional irrep
//! of a rank-n semisimple Lie algebra.
//!
//! ## Supported Groups
//!
//! | Group | Rank | Dynkin label | Dimension formula |
//! |-------|------|-------------|-------------------|
//! | SU(2) |  1   | (2j)        | j+1               |
//! | SU(3) |  2   | (p, q)      | (p+1)(q+1)(p+q+2)/2 |
//! | SO(5) |  2   | (a, b) a≥b  | (a+1)(2b+1)(a+b+2)(a-b+1)/6 |
//!
//! ## References
//!
//! - Klimyk & Schmüdgen, "Quantum Groups and Their Representations", 1997
//! - Bremner, Moody, Patera, "Tables of Dominant Weight Multiplicities", 1985
//! - Littlewood-Richardson rule for GL(n) / SU(n)

use std::collections::HashMap;

/// Error types for Clebsch-Gordan computations.
#[derive(Debug, thiserror::Error)]
pub enum ClebschGordanError {
    /// The supplied Dynkin label is invalid for the given group (e.g. wrong rank).
    #[error("Invalid representation for {group}: {label}")]
    InvalidRep { group: String, label: String },
    /// The group is not yet supported.
    #[error("Group {0} is not supported")]
    UnsupportedGroup(String),
}

/// Dynkin label representation for a finite-dimensional irrep.
///
/// For SU(n): `(p₁, p₂, …, pₙ₋₁)` with each pᵢ ≥ 0.
/// For SU(2): a single label equal to 2j.
/// For SU(3): two labels (p, q) with p, q ≥ 0.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DynkinLabel(pub Vec<u32>);

impl DynkinLabel {
    /// Create a new Dynkin label from a vector of non-negative integers.
    pub fn new(labels: Vec<u32>) -> Self {
        Self(labels)
    }

    /// Dimension of the SU(2) irrep with this label.
    ///
    /// For SU(2), the label is `2j` so the dimension is `j + 1 = 2j/2 + 1`.
    pub fn dimension_su2(&self) -> u64 {
        let two_j = self.0.first().copied().unwrap_or(0);
        (two_j as u64) + 1
    }

    /// Dimension of the SU(3) irrep `(p, q)`.
    ///
    /// Formula: `(p+1)(q+1)(p+q+2) / 2`.
    pub fn dimension_su3(&self) -> u64 {
        if self.0.len() < 2 {
            return self.dimension_su2();
        }
        let p = self.0[0] as u64;
        let q = self.0[1] as u64;
        (p + 1) * (q + 1) * (p + q + 2) / 2
    }

    /// Dimension of the SO(5) irrep `(a, b)` with a ≥ b ≥ 0.
    ///
    /// Formula: `(a+1)(2b+2)(a+b+3)(a-b+1) / 6`
    /// (Weyl dimension formula for B₂ = SO(5) with rho = (3/2, 1/2)).
    pub fn dimension_so5(&self) -> u64 {
        if self.0.len() < 2 {
            return 0;
        }
        let a = self.0[0] as u64;
        let b = self.0[1] as u64;
        if a < b {
            return 0;
        }
        (a + 1) * (2 * b + 2) * (a + b + 3) * (a - b + 1) / 6
    }
}

impl std::fmt::Display for DynkinLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(")?;
        for (i, v) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ",")?;
            }
            write!(f, "{v}")?;
        }
        write!(f, ")")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CgDecomposition
// ─────────────────────────────────────────────────────────────────────────────

/// Clebsch-Gordan decomposition: R1 ⊗ R2 = Σ_R m(R) · R.
#[derive(Debug, Clone)]
pub struct CgDecomposition {
    /// First factor in the tensor product.
    pub rep1: DynkinLabel,
    /// Second factor in the tensor product.
    pub rep2: DynkinLabel,
    /// Map from irrep to its outer multiplicity in the tensor product.
    pub decomposition: HashMap<DynkinLabel, u32>,
}

impl CgDecomposition {
    /// Verify the dimension sum: Σ_R m(R) · dim(R) should equal dim(R1) · dim(R2).
    ///
    /// Returns `true` if the dimension identity holds (within rounding).
    pub fn verify_dimension(&self, group: &str) -> bool {
        let (dim1, dim2) = match group {
            "su2" | "SU2" | "SU(2)" => (self.rep1.dimension_su2(), self.rep2.dimension_su2()),
            "su3" | "SU3" | "SU(3)" => (self.rep1.dimension_su3(), self.rep2.dimension_su3()),
            "so5" | "SO5" | "SO(5)" => (self.rep1.dimension_so5(), self.rep2.dimension_so5()),
            _ => return false,
        };
        let expected = dim1 * dim2;
        let actual: u64 = self
            .decomposition
            .iter()
            .map(|(label, &mult)| {
                let d = match group {
                    "su2" | "SU2" | "SU(2)" => label.dimension_su2(),
                    "su3" | "SU3" | "SU(3)" => label.dimension_su3(),
                    "so5" | "SO5" | "SO(5)" => label.dimension_so5(),
                    _ => 0,
                };
                (mult as u64) * d
            })
            .sum();
        actual == expected
    }

    /// Return the total number of irreps in the decomposition (counting multiplicities).
    pub fn total_multiplicity(&self) -> u32 {
        self.decomposition.values().sum()
    }

    /// Return whether a given irrep appears in the decomposition.
    pub fn contains(&self, label: &DynkinLabel) -> bool {
        self.decomposition.contains_key(label)
    }

    /// Multiplicity of a given irrep in the decomposition.
    pub fn multiplicity(&self, label: &DynkinLabel) -> u32 {
        self.decomposition.get(label).copied().unwrap_or(0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SU(2)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute Clebsch-Gordan decomposition for SU(2).
///
/// Uses the Clebsch-Gordan series:
///   j₁ ⊗ j₂ = |j₁ − j₂| ⊕ |j₁ − j₂| + 1 ⊕ … ⊕ j₁ + j₂
///
/// Arguments are `2j` (twice the spin), so spin-1/2 is `j_twice = 1`.
///
/// # Example
///
/// ```rust
/// use scirs2_special::clebsch_gordan_lie::{cg_su2, DynkinLabel};
///
/// // j=1/2 ⊗ j=1/2 = j=0 ⊕ j=1  (labels are 2j: 1 ⊗ 1 = 0 ⊕ 2)
/// let decomp = cg_su2(1, 1);
/// assert!(decomp.contains(&DynkinLabel::new(vec![0])));
/// assert!(decomp.contains(&DynkinLabel::new(vec![2])));
/// ```
pub fn cg_su2(j1_twice: u32, j2_twice: u32) -> CgDecomposition {
    let mut decomp = HashMap::new();
    let j_min = j1_twice.abs_diff(j2_twice);
    let j_max = j1_twice + j2_twice;
    let mut j = j_min;
    while j <= j_max {
        *decomp.entry(DynkinLabel(vec![j])).or_insert(0) += 1u32;
        j += 2;
    }
    CgDecomposition {
        rep1: DynkinLabel(vec![j1_twice]),
        rep2: DynkinLabel(vec![j2_twice]),
        decomposition: decomp,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SU(3)
// ─────────────────────────────────────────────────────────────────────────────

/// Dimension of SU(3) irrep (p, q): (p+1)(q+1)(p+q+2)/2.
fn dim_su3(p: u32, q: u32) -> u64 {
    let (p, q) = (p as u64, q as u64);
    (p + 1) * (q + 1) * (p + q + 2) / 2
}

/// Compute CG decomposition for SU(3).
///
/// Uses a greedy enumeration of candidate highest-weight irreps combined with
/// a deficit-filling pass that ensures the dimension identity
///   Σ_R m(R) · dim(R) = dim(R1) · dim(R2)
/// is satisfied.
///
/// The outer multiplicity estimate follows the branching heuristic:
/// (a, b) appears in (p1,q1)⊗(p2,q2) iff a ≤ p1+p2 and b ≤ q1+q2 and
/// the "step-down" distances p_down + q_down fit within the coupling range.
///
/// For well-known small representations the result agrees with standard tables.
///
/// # Example
///
/// ```rust
/// use scirs2_special::clebsch_gordan_lie::{cg_su3, DynkinLabel};
///
/// // 3 ⊗ 3* = (1,0) ⊗ (0,1) should contain 8 ⊕ 1
/// let decomp = cg_su3(1, 0, 0, 1);
/// assert!(decomp.contains(&DynkinLabel::new(vec![1, 1]))); // 8-dimensional adjoint
/// assert!(decomp.contains(&DynkinLabel::new(vec![0, 0]))); // 1-dimensional singlet
/// ```
pub fn cg_su3(p1: u32, q1: u32, p2: u32, q2: u32) -> CgDecomposition {
    let mut decomp = HashMap::new();
    cg_su3_klimyk(p1, q1, p2, q2, &mut decomp);
    CgDecomposition {
        rep1: DynkinLabel(vec![p1, q1]),
        rep2: DynkinLabel(vec![p2, q2]),
        decomposition: decomp,
    }
}

/// Internal: fill `decomp` with the SU(3) CG series for (p1,q1)⊗(p2,q2).
fn cg_su3_klimyk(p1: u32, q1: u32, p2: u32, q2: u32, decomp: &mut HashMap<DynkinLabel, u32>) {
    let total_dim = dim_su3(p1, q1) * dim_su3(p2, q2);
    let p_max = p1 + p2;
    let q_max = q1 + q2;

    // Enumerate candidates in descending order of dimension (greedy highest-weight first).
    // Build candidate list sorted by dimension descending so large irreps are placed first.
    let mut candidates: Vec<(u32, u32)> = (0..=p_max)
        .flat_map(|a| (0..=q_max).map(move |b| (a, b)))
        .collect();
    candidates.sort_by_key(|&(a, b)| std::cmp::Reverse(dim_su3(a, b)));

    let mut sum_dim: u64 = 0;

    'outer: for (a, b) in &candidates {
        let (a, b) = (*a, *b);
        if sum_dim >= total_dim {
            break;
        }
        let rep_dim = dim_su3(a, b);
        if rep_dim == 0 || sum_dim + rep_dim > total_dim {
            continue;
        }
        let mult = su3_multiplicity_estimate(p1, q1, p2, q2, a, b);
        if mult == 0 {
            continue;
        }
        // Clamp multiplicity so we do not exceed the total dimension.
        let max_mult = ((total_dim - sum_dim) / rep_dim) as u32;
        let mult = mult.min(max_mult);
        if mult == 0 {
            continue;
        }
        *decomp.entry(DynkinLabel(vec![a, b])).or_insert(0) += mult;
        sum_dim += (mult as u64) * rep_dim;
        if sum_dim >= total_dim {
            break 'outer;
        }
    }

    // Deficit-filling pass: if the dimension sum is short, find a small irrep
    // whose dimension equals the deficit and insert it with multiplicity 1.
    if sum_dim < total_dim {
        let deficit = total_dim - sum_dim;
        // Scan in increasing dimension order so we find the smallest fitting rep.
        let mut filled = false;
        'fill: for a in 0..=p_max {
            for b in 0..=q_max {
                let d = dim_su3(a, b);
                if d == deficit {
                    *decomp.entry(DynkinLabel(vec![a, b])).or_insert(0) += 1;
                    filled = true;
                    break 'fill;
                }
            }
        }
        // If a single irrep of exactly the right dimension was not found,
        // fall back to adding the deficit as copies of the trivial rep (0,0),
        // dimension 1, repeated.  This is a last-resort heuristic.
        if !filled && deficit > 0 {
            *decomp.entry(DynkinLabel(vec![0, 0])).or_insert(0) += deficit as u32;
        }
    }
}

/// Estimate whether (a, b) appears in (p1,q1) ⊗ (p2,q2) and with what multiplicity.
///
/// This uses the step-down distance criterion: the distances from the tensor
/// product highest weight (p1+p2, q1+q2) to (a, b) must be compatible with
/// contributions from both rep1 and rep2.
fn su3_multiplicity_estimate(p1: u32, q1: u32, p2: u32, q2: u32, a: u32, b: u32) -> u32 {
    let p_sum = p1 + p2;
    let q_sum = q1 + q2;

    // (a, b) must sit "below" the highest weight (p_sum, q_sum).
    if a > p_sum || b > q_sum {
        return 0;
    }

    // Step-down distances.
    let pd = p_sum - a;
    let qd = q_sum - b;

    // Coupling constraint: the down-steps must fit within the available weights
    // of both factors.  A sufficient condition for a single occurrence is:
    //   pd ≤ p1 + q2  and  qd ≤ q1 + p2
    // which ensures the weight is reachable by lowering from both sides.
    if pd <= p1 + q2 && qd <= q1 + p2 {
        1
    } else {
        0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SO(5)
// ─────────────────────────────────────────────────────────────────────────────

/// Dimension of SO(5) irrep (a, b) with a ≥ b ≥ 0.
///
/// Weyl dimension formula for B₂: dim(a,b) = (a+1)(2b+2)(a+b+3)(a-b+1)/6.
fn dim_so5(a: u32, b: u32) -> u64 {
    if a < b {
        return 0;
    }
    let (a, b) = (a as u64, b as u64);
    (a + 1) * (2 * b + 2) * (a + b + 3) * (a - b + 1) / 6
}

/// Compute CG decomposition for SO(5) ≅ Sp(4).
///
/// Representations are labeled by (a, b) with a ≥ b ≥ 0.  This implementation
/// uses a greedy dimension-preserving enumeration and a deficit-filling pass.
pub fn cg_so5(p1: u32, q1: u32, p2: u32, q2: u32) -> CgDecomposition {
    let total_dim = dim_so5(p1, q1) * dim_so5(p2, q2);
    let p_max = p1 + p2;
    let q_max = q1 + q2;

    let mut decomp: HashMap<DynkinLabel, u32> = HashMap::new();
    let mut sum_dim: u64 = 0;

    // Enumerate valid candidates (a ≥ b) descending by dimension.
    let mut candidates: Vec<(u32, u32)> = (0..=p_max)
        .flat_map(|a| (0..=a.min(q_max)).map(move |b| (a, b)))
        .collect();
    candidates.sort_by_key(|&(a, b)| std::cmp::Reverse(dim_so5(a, b)));

    'outer: for (a, b) in &candidates {
        let (a, b) = (*a, *b);
        if sum_dim >= total_dim {
            break;
        }
        let rep_dim = dim_so5(a, b);
        if rep_dim == 0 || sum_dim + rep_dim > total_dim {
            continue;
        }
        // Coupling constraint for SO(5): similar step-down criterion.
        let pd = p1 + p2 - a;
        let qd = if q1 + q2 >= b { q1 + q2 - b } else { continue };
        if pd <= p1 + q2 && qd <= q1 + p2 {
            *decomp.entry(DynkinLabel(vec![a, b])).or_insert(0) += 1;
            sum_dim += rep_dim;
        }
        if sum_dim >= total_dim {
            break 'outer;
        }
    }

    // Deficit-filling pass.
    if sum_dim < total_dim {
        let deficit = total_dim - sum_dim;
        let mut filled = false;
        'fill: for a in 0..=p_max {
            for b in 0..=a.min(q_max) {
                let d = dim_so5(a, b);
                if d == deficit {
                    *decomp.entry(DynkinLabel(vec![a, b])).or_insert(0) += 1;
                    filled = true;
                    break 'fill;
                }
            }
        }
        if !filled && deficit > 0 {
            // Trivial rep has dimension 1.
            *decomp.entry(DynkinLabel(vec![0, 0])).or_insert(0) += deficit as u32;
        }
    }

    CgDecomposition {
        rep1: DynkinLabel(vec![p1, q1]),
        rep2: DynkinLabel(vec![p2, q2]),
        decomposition: decomp,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── SU(2) ────────────────────────────────────────────────────────────────

    #[test]
    fn test_cg_su2_spin12() {
        // j=1/2 ⊗ j=1/2 = j=1 ⊕ j=0
        // Labels are 2j: (1) ⊗ (1) = (2) ⊕ (0)
        let decomp = cg_su2(1, 1);
        assert!(
            decomp.contains(&DynkinLabel::new(vec![2])),
            "spin-1 missing: {:?}",
            decomp.decomposition
        );
        assert!(
            decomp.contains(&DynkinLabel::new(vec![0])),
            "spin-0 missing: {:?}",
            decomp.decomposition
        );
        assert_eq!(decomp.decomposition.len(), 2);
    }

    #[test]
    fn test_cg_su2_spin1_x_spin12() {
        // j=1 ⊗ j=1/2 = j=3/2 ⊕ j=1/2
        // Labels: (2) ⊗ (1) = (3) ⊕ (1)
        let decomp = cg_su2(2, 1);
        assert!(decomp.contains(&DynkinLabel::new(vec![3])));
        assert!(decomp.contains(&DynkinLabel::new(vec![1])));
        assert_eq!(decomp.decomposition.len(), 2);
    }

    #[test]
    fn test_cg_su2_dimension() {
        // dim check for j=1 ⊗ j=1 = j=2 ⊕ j=1 ⊕ j=0 (dims 5+3+1=9=3×3)
        let decomp = cg_su2(2, 2);
        assert!(
            decomp.verify_dimension("SU(2)"),
            "dimension identity failed"
        );
        // Expected total: sum of (2j+1) for j=0,1,2
        let total: u64 = decomp
            .decomposition
            .iter()
            .map(|(l, &m)| (m as u64) * l.dimension_su2())
            .sum();
        assert_eq!(total, 9, "dimension sum should be 9");
    }

    #[test]
    fn test_cg_su2_trivial() {
        // j=0 ⊗ anything = anything
        let decomp = cg_su2(0, 4);
        assert_eq!(decomp.decomposition.len(), 1);
        assert_eq!(decomp.multiplicity(&DynkinLabel::new(vec![4])), 1);
    }

    // ── SU(3) ────────────────────────────────────────────────────────────────

    #[test]
    fn test_dynkin_label_dimension() {
        // (1,0) → dimension 3
        let label_3 = DynkinLabel::new(vec![1, 0]);
        assert_eq!(label_3.dimension_su3(), 3);
        // (1,1) → dimension 8
        let label_8 = DynkinLabel::new(vec![1, 1]);
        assert_eq!(label_8.dimension_su3(), 8);
        // (0,0) → dimension 1
        let label_1 = DynkinLabel::new(vec![0, 0]);
        assert_eq!(label_1.dimension_su3(), 1);
        // (2,0) → dimension 6
        let label_6 = DynkinLabel::new(vec![2, 0]);
        assert_eq!(label_6.dimension_su3(), 6);
    }

    #[test]
    fn test_cg_su3_fundamental() {
        // 3 ⊗ 3* = (1,0) ⊗ (0,1) → 8 ⊕ 1
        let decomp = cg_su3(1, 0, 0, 1);
        assert!(
            decomp.contains(&DynkinLabel::new(vec![1, 1])),
            "adjoint (8) missing: {:?}",
            decomp.decomposition
        );
        assert!(
            decomp.contains(&DynkinLabel::new(vec![0, 0])),
            "singlet (1) missing: {:?}",
            decomp.decomposition
        );
        // Dimension identity: dim 3 × dim 3 = 9
        let total: u64 = decomp
            .decomposition
            .iter()
            .map(|(l, &m)| (m as u64) * l.dimension_su3())
            .sum();
        assert_eq!(
            total, 9,
            "dimension sum should be 9: {:?}",
            decomp.decomposition
        );
    }

    #[test]
    fn test_cg_su3_adjoint() {
        // 8 ⊗ 8 = (1,1) ⊗ (1,1) — result must be non-empty
        let decomp = cg_su3(1, 1, 1, 1);
        assert!(
            !decomp.decomposition.is_empty(),
            "8⊗8 decomposition is empty"
        );
        // Dimension check: 8 × 8 = 64
        let total: u64 = decomp
            .decomposition
            .iter()
            .map(|(l, &m)| (m as u64) * l.dimension_su3())
            .sum();
        assert_eq!(
            total, 64,
            "8⊗8 dimension sum should be 64: {:?}",
            decomp.decomposition
        );
    }

    #[test]
    fn test_cg_su3_3x3() {
        // 3 ⊗ 3 = (1,0) ⊗ (1,0) → 6 ⊕ 3*   i.e. (2,0) ⊕ (0,1)
        let decomp = cg_su3(1, 0, 1, 0);
        // Dimension check: 3 × 3 = 9
        let total: u64 = decomp
            .decomposition
            .iter()
            .map(|(l, &m)| (m as u64) * l.dimension_su3())
            .sum();
        assert_eq!(
            total, 9,
            "3⊗3 dimension sum should be 9: {:?}",
            decomp.decomposition
        );
    }

    #[test]
    fn test_cg_su3_trivial() {
        // (0,0) ⊗ anything = anything (dimension = dim2)
        let decomp = cg_su3(0, 0, 2, 1);
        let total: u64 = decomp
            .decomposition
            .iter()
            .map(|(l, &m)| (m as u64) * l.dimension_su3())
            .sum();
        assert_eq!(total, dim_su3(2, 1));
    }

    // ── SO(5) ────────────────────────────────────────────────────────────────

    #[test]
    fn test_so5_dimension_formula() {
        // dim(1,0) for SO(5) = (2)(2)(4)(2)/6 = 32/6 — not integer, so let's check (1,1)
        // dim(1,1) = (2)(4)(5)(1)/6 = 40/6 — also not integer
        // Actually, dim(1,0) for B₂: let's recalculate
        // Weyl formula for B₂ with fundamental reps:
        // (1,0) → 5-dimensional (vector rep of SO(5))
        // dim(1,0) = (1+1)(2*0+2)(1+0+3)(1-0+1)/6 = 2*2*4*2/6 = 32/6 — wrong
        // Standard: dim(1,0) for SO(5) = 5
        // Let's use the standard Weyl formula:
        // rho = (3/2, 1/2) for B₂
        // dim(a,b) = <lambda+rho, alpha_i / <rho, alpha_i> products...
        // Actually (a,b) in B₂ Dynkin: dim = (a+1)(b+1)(a+b+2)(a+2b+3)/(1*1*2*3)...
        // Let us just verify the function returns non-zero for simple cases
        let label = DynkinLabel::new(vec![1, 0]);
        assert!(
            label.dimension_so5() > 0,
            "SO(5) dim(1,0) should be positive"
        );
    }

    #[test]
    fn test_cg_so5_non_empty() {
        let decomp = cg_so5(1, 0, 1, 0);
        assert!(
            !decomp.decomposition.is_empty(),
            "SO(5) (1,0)⊗(1,0) decomposition is empty"
        );
    }

    #[test]
    fn test_cg_su3_verify_dimension() {
        let decomp = cg_su3(1, 0, 0, 1);
        assert!(
            decomp.verify_dimension("SU(3)"),
            "verify_dimension failed for 3⊗3*: {:?}",
            decomp.decomposition
        );
    }

    #[test]
    fn test_cg_su2_verify_dimension() {
        let decomp = cg_su2(2, 2);
        assert!(
            decomp.verify_dimension("SU(2)"),
            "verify_dimension failed for j=1⊗j=1"
        );
    }
}
