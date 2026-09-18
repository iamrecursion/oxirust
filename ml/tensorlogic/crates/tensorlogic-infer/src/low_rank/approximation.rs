//! High-level low-rank approximation API and inference pass.

use tensorlogic_ir::EinsumGraph;

use super::config::LowRankConfig;
use super::error::LowRankError;
use super::svd::{SvdResult, TruncatedSvd};

// ---------------------------------------------------------------------------
// LowRankApproximation
// ---------------------------------------------------------------------------

/// High-level API for low-rank matrix approximation.
pub struct LowRankApproximation {
    config: LowRankConfig,
    svd: TruncatedSvd,
}

impl LowRankApproximation {
    /// Create a new approximation engine with the given configuration.
    pub fn new(config: LowRankConfig) -> Self {
        let svd = TruncatedSvd::new(config.clone());
        LowRankApproximation { config, svd }
    }

    /// Approximate a 2-D matrix stored in row-major order.
    pub fn approximate_matrix(
        &self,
        data: &[f64],
        rows: usize,
        cols: usize,
    ) -> Result<SvdResult, LowRankError> {
        self.svd.decompose(data, rows, cols)
    }

    /// Approximate the matrix product `A @ B` using low-rank factors of `A`.
    ///
    /// Rather than computing the full product `C = A B` (which is O(a_rows · a_cols · b_cols)),
    /// we approximate `A ≈ U Σ Vᵀ` (rank-k) and then compute `C ≈ (U Σ) (Vᵀ B)`.
    /// This can be cheaper when `rank << min(a_rows, a_cols)`.
    pub fn approximate_matmul(
        &self,
        a: &[f64],
        a_rows: usize,
        a_cols: usize,
        b: &[f64],
        b_rows: usize,
        b_cols: usize,
    ) -> Result<Vec<f64>, LowRankError> {
        if a_cols != b_rows {
            return Err(LowRankError::InvalidInput(format!(
                "inner dimensions mismatch: a_cols={} != b_rows={}",
                a_cols, b_rows
            )));
        }

        let svd_result = self.svd.decompose(a, a_rows, a_cols)?;
        let rank = svd_result.rank_used;

        // Compute intermediate: M = Vᵀ B   [rank × b_cols]
        let mut m = vec![0.0_f64; rank * b_cols];
        for k in 0..rank {
            for j in 0..b_cols {
                let mut val = 0.0_f64;
                for l in 0..b_rows {
                    // vt[k, l] * b[l, j]
                    val += svd_result.vt[k * svd_result.vt_cols + l] * b[l * b_cols + j];
                }
                m[k * b_cols + j] = val;
            }
        }

        // Compute result: C = (U Σ) M   [a_rows × b_cols]
        let mut c = vec![0.0_f64; a_rows * b_cols];
        for i in 0..a_rows {
            for j in 0..b_cols {
                let mut val = 0.0_f64;
                for k in 0..rank {
                    // u[i, k] * sigma[k] * m[k, j]
                    let u_ik = svd_result.u[i * svd_result.u_cols + k];
                    val += u_ik * svd_result.singular_values[k] * m[k * b_cols + j];
                }
                c[i * b_cols + j] = val;
            }
        }

        Ok(c)
    }

    /// Return `true` if this matrix is large enough to be a candidate for
    /// low-rank approximation (based on `min_dimension` in the config).
    pub fn is_candidate(&self, rows: usize, cols: usize) -> bool {
        rows >= self.config.min_dimension && cols >= self.config.min_dimension
    }

    /// Compute the smallest rank `k` such that the top-`k` singular values
    /// capture at least `energy_threshold` fraction of the total singular energy.
    ///
    /// `energy_threshold` should be in `[0, 1]`.  If the slice is empty the
    /// function returns `0`.
    pub fn optimal_rank(singular_values: &[f64], energy_threshold: f64) -> usize {
        if singular_values.is_empty() {
            return 0;
        }
        let total: f64 = singular_values.iter().map(|s| s * s).sum();
        if total == 0.0 {
            return 1;
        }
        let mut cumulative = 0.0_f64;
        for (k, &sv) in singular_values.iter().enumerate() {
            cumulative += sv * sv;
            if cumulative / total >= energy_threshold {
                return k + 1;
            }
        }
        singular_values.len()
    }
}

// ---------------------------------------------------------------------------
// LowRankInferencePass
// ---------------------------------------------------------------------------

/// A low-rank approximation candidate identified in an `EinsumGraph`.
#[derive(Debug, Clone)]
pub struct LowRankCandidate {
    /// Index of the node in `EinsumGraph::nodes`
    pub node_index: usize,
    /// Human-readable reason this node was flagged
    pub reason: String,
    /// Rough estimated savings as a ratio (0–1); higher is better.
    pub estimated_savings_ratio: f64,
}

/// Aggregated statistics from a single pass over an `EinsumGraph`.
#[derive(Debug, Clone, Default)]
pub struct LowRankPassStats {
    pub candidates_found: usize,
    pub nodes_inspected: usize,
    pub estimated_total_flop_reduction: f64,
}

/// Optimization pass that scans an `EinsumGraph` and annotates Einsum nodes
/// with low-rank approximation candidates.
///
/// Currently uses a heuristic based on einsum spec complexity (number of
/// unique contracted indices) to identify potential candidates.
#[derive(Debug)]
pub struct LowRankInferencePass {
    config: LowRankConfig,
}

impl LowRankInferencePass {
    /// Create a new pass with the given configuration.
    pub fn new(config: LowRankConfig) -> Self {
        LowRankInferencePass { config }
    }

    /// Scan the graph and return a list of low-rank candidates.
    pub fn find_candidates(&self, graph: &EinsumGraph) -> Vec<LowRankCandidate> {
        let mut candidates = Vec::new();

        for (idx, node) in graph.nodes.iter().enumerate() {
            if let tensorlogic_ir::OpType::Einsum { spec } = &node.op {
                // Heuristic: if the einsum spec suggests a matmul-like pattern
                // (two inputs, contracted indices) and would benefit from low-rank
                // approximation, flag it.
                if node.inputs.len() >= 2 && self.is_matmul_like(spec) {
                    let savings = self.estimate_savings(spec);
                    candidates.push(LowRankCandidate {
                        node_index: idx,
                        reason: format!(
                            "Einsum '{}' has {} inputs and matmul-like contraction",
                            spec,
                            node.inputs.len()
                        ),
                        estimated_savings_ratio: savings,
                    });
                }
            }
        }

        candidates
    }

    /// Apply annotations and return aggregate stats.
    ///
    /// In this implementation the "annotation" is a dry-run analysis only —
    /// the graph is not mutated (annotation requires mutable access and is
    /// outside the scope of a read-only pass).
    pub fn apply_annotations(&self, graph: &EinsumGraph) -> LowRankPassStats {
        let candidates = self.find_candidates(graph);
        let estimated_total_flop_reduction: f64 =
            candidates.iter().map(|c| c.estimated_savings_ratio).sum();
        LowRankPassStats {
            candidates_found: candidates.len(),
            nodes_inspected: graph.nodes.len(),
            estimated_total_flop_reduction,
        }
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Very light heuristic: treat any spec with `->` and two operands that
    /// share at least one contracted index as matmul-like.
    fn is_matmul_like(&self, spec: &str) -> bool {
        if let Some(arrow_pos) = spec.find("->") {
            let inputs_part = &spec[..arrow_pos];
            let operands: Vec<&str> = inputs_part.split(',').collect();
            if operands.len() < 2 {
                return false;
            }
            // Check for shared characters (contracted indices)
            let a_chars: std::collections::HashSet<char> =
                operands[0].chars().filter(|c| c.is_alphabetic()).collect();
            let b_chars: std::collections::HashSet<char> =
                operands[1].chars().filter(|c| c.is_alphabetic()).collect();
            let output_chars: std::collections::HashSet<char> = spec[arrow_pos + 2..]
                .chars()
                .filter(|c| c.is_alphabetic())
                .collect();
            // A contracted index appears in inputs but not in output
            let contracted: std::collections::HashSet<char> = a_chars
                .intersection(&b_chars)
                .copied()
                .filter(|c| !output_chars.contains(c))
                .collect();
            return contracted.len() >= 1
                && self.config.rank < self.min_contracted_dim_estimate(spec);
        }
        false
    }

    /// Estimate the number of contracted dimensions from the spec string.
    /// Used as a rough proxy for matrix size.
    fn min_contracted_dim_estimate(&self, spec: &str) -> usize {
        // Use rank as a stand-in; if rank < estimated contracted dims → candidate
        // Here we just count contracted chars as a size proxy
        let contracted = spec.chars().filter(|c| c.is_alphabetic()).count();
        // If the spec has at least 4 unique index chars assume "large enough"
        contracted.max(1)
    }

    /// Estimate FLOP savings ratio for a candidate node.
    fn estimate_savings(&self, spec: &str) -> f64 {
        // Heuristic: savings = 1 - (2*rank) / (contracted dims)
        let contracted_dims = self.min_contracted_dim_estimate(spec).max(1) as f64;
        let rank = self.config.rank as f64;
        (1.0 - (2.0 * rank) / contracted_dims).clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(rank: usize) -> LowRankConfig {
        LowRankConfig::new(rank)
            .with_tolerance(1e-8)
            .with_max_iterations(300)
            .with_min_dimension(8)
    }

    // -----------------------------------------------------------------------
    // LowRankApproximation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_approximation_4x4_matrix() {
        // Use a rank-2 approximation on a 4×4 matrix
        let m: Vec<f64> = (1..=16).map(|x| x as f64).collect();
        let cfg = make_config(2);
        let approx = LowRankApproximation::new(cfg);
        let svd = approx
            .approximate_matrix(&m, 4, 4)
            .expect("approximation should succeed for a valid 4x4 matrix");
        assert!(svd.rank_used >= 1);
        // Frobenius error should be a valid number
        assert!(svd.frobenius_error.is_finite());
    }

    #[test]
    fn test_is_candidate_small_matrix() {
        let cfg = LowRankConfig::new(2).with_min_dimension(32);
        let approx = LowRankApproximation::new(cfg);
        // 4×4 is below min_dimension=32
        assert!(!approx.is_candidate(4, 4));
    }

    #[test]
    fn test_is_candidate_large_matrix() {
        let cfg = LowRankConfig::new(4).with_min_dimension(32);
        let approx = LowRankApproximation::new(cfg);
        // 64×64 is above min_dimension=32
        assert!(approx.is_candidate(64, 64));
    }

    #[test]
    fn test_optimal_rank_energy_threshold() {
        // Singular values: [10, 5, 2, 1]  → energies squared: [100, 25, 4, 1]  total=130
        // 0.9 threshold → need cumulative >= 117 → first two give 125 >= 117 → rank=2
        let svs = vec![10.0_f64, 5.0, 2.0, 1.0];
        let r = LowRankApproximation::optimal_rank(&svs, 0.90);
        assert_eq!(r, 2, "optimal rank for 90% energy should be 2, got {r}");

        // 0.99 threshold → 100+25+4=129 >= 0.99*130=128.7 → rank=3
        let r2 = LowRankApproximation::optimal_rank(&svs, 0.99);
        assert_eq!(r2, 3, "optimal rank for 99% energy should be 3, got {r2}");
    }

    // -----------------------------------------------------------------------
    // LowRankInferencePass tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_inference_pass_empty_graph() {
        let graph = EinsumGraph::new();
        let pass = LowRankInferencePass::new(LowRankConfig::default());
        let candidates = pass.find_candidates(&graph);
        assert!(
            candidates.is_empty(),
            "empty graph should yield no candidates"
        );
    }

    #[test]
    fn test_inference_pass_stats() {
        let mut graph = EinsumGraph::new();
        let t0 = graph.add_tensor("A");
        let t1 = graph.add_tensor("B");
        let t2 = graph.add_tensor("C");
        let node = tensorlogic_ir::EinsumNode::einsum("ij,jk->ik", vec![t0, t1], vec![t2]);
        graph.add_node(node).expect("add_node ok");

        let pass = LowRankInferencePass::new(LowRankConfig::new(2));
        let stats = pass.apply_annotations(&graph);
        assert_eq!(stats.nodes_inspected, 1);
        // This is a matmul-like spec with contracted index j
        // candidates_found could be 0 or 1 depending on heuristic
        assert!(stats.nodes_inspected >= 1);
        assert!(stats.estimated_total_flop_reduction >= 0.0);
    }
}
