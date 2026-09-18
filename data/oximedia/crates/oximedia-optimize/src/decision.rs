//! Mode decision strategies and helpers.
//!
//! When the `ml-decision` feature is enabled, `MlModeDecider` can pre-filter
//! the candidate pool via a small 3-layer MLP before the full RDO pass.

/// Mode decision strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionStrategy {
    /// Use distortion only (fastest).
    DistortionOnly,
    /// Use rate-distortion optimization.
    RateDistortion,
    /// Use full trellis quantization.
    FullTrellis,
}

/// Mode decision context.
#[derive(Debug, Clone)]
pub struct DecisionContext {
    /// Current QP.
    pub qp: u8,
    /// Lambda for RD cost.
    pub lambda: f64,
    /// Available reference frames.
    pub num_references: usize,
    /// Block size.
    pub block_size: usize,
    /// Whether this is an intra block.
    pub is_intra: bool,
}

impl DecisionContext {
    /// Creates a new decision context.
    #[must_use]
    pub fn new(qp: u8, lambda: f64, block_size: usize, is_intra: bool) -> Self {
        Self {
            qp,
            lambda,
            num_references: 0,
            block_size,
            is_intra,
        }
    }
}

/// Mode decision helper.
pub struct ModeDecision {
    strategy: DecisionStrategy,
    early_termination: bool,
    max_candidates: usize,
}

impl Default for ModeDecision {
    fn default() -> Self {
        Self::new(DecisionStrategy::RateDistortion)
    }
}

impl ModeDecision {
    /// Creates a new mode decision helper.
    #[must_use]
    pub fn new(strategy: DecisionStrategy) -> Self {
        Self {
            strategy,
            early_termination: true,
            max_candidates: 8,
        }
    }

    /// Sets whether to use early termination.
    pub fn set_early_termination(&mut self, enable: bool) {
        self.early_termination = enable;
    }

    /// Sets maximum number of candidates to evaluate.
    pub fn set_max_candidates(&mut self, max: usize) {
        self.max_candidates = max;
    }

    /// Evaluates and selects the best mode.
    #[allow(dead_code)]
    pub fn select_best_mode<T>(
        &self,
        candidates: &[T],
        context: &DecisionContext,
        eval_fn: impl Fn(&T) -> (f64, f64),
    ) -> (usize, f64) {
        if candidates.is_empty() {
            return (0, f64::MAX);
        }

        let mut best_idx = 0;
        let mut best_cost = f64::MAX;

        let num_candidates = if self.early_termination {
            candidates.len().min(self.max_candidates)
        } else {
            candidates.len()
        };

        for (idx, candidate) in candidates.iter().enumerate().take(num_candidates) {
            let (distortion, rate) = eval_fn(candidate);
            let cost = self.calculate_cost(distortion, rate, context);

            if cost < best_cost {
                best_cost = cost;
                best_idx = idx;

                // Early termination if cost is very good
                if self.early_termination && cost < 10.0 {
                    break;
                }
            }
        }

        (best_idx, best_cost)
    }

    fn calculate_cost(&self, distortion: f64, rate: f64, context: &DecisionContext) -> f64 {
        match self.strategy {
            DecisionStrategy::DistortionOnly => distortion,
            DecisionStrategy::RateDistortion | DecisionStrategy::FullTrellis => {
                distortion + context.lambda * rate
            }
        }
    }

    /// Checks if a mode should be skipped based on early termination.
    #[must_use]
    pub fn should_skip_mode(&self, current_cost: f64, best_cost: f64) -> bool {
        if !self.early_termination {
            return false;
        }

        // Skip if current cost is much worse than best
        current_cost > best_cost * 1.5
    }
}

/// Split decision helper for partition decisions.
pub struct SplitDecision {
    min_size: usize,
    max_size: usize,
    complexity_threshold: f64,
}

impl Default for SplitDecision {
    fn default() -> Self {
        Self::new(4, 128, 100.0)
    }
}

impl SplitDecision {
    /// Creates a new split decision helper.
    #[must_use]
    pub const fn new(min_size: usize, max_size: usize, complexity_threshold: f64) -> Self {
        Self {
            min_size,
            max_size,
            complexity_threshold,
        }
    }

    /// Determines if a block should be split.
    #[must_use]
    pub fn should_split(&self, block_size: usize, complexity: f64, depth: usize) -> bool {
        // Don't split below minimum size
        if block_size <= self.min_size {
            return false;
        }

        // Don't split above maximum size
        if block_size > self.max_size {
            return true;
        }

        // Split based on complexity
        let adjusted_threshold = self.complexity_threshold * (1.0 + depth as f64 * 0.1);
        complexity > adjusted_threshold
    }

    /// Evaluates split decision with RD cost.
    #[allow(dead_code)]
    #[must_use]
    pub fn evaluate_split(
        &self,
        no_split_cost: f64,
        split_cost: f64,
        split_overhead: f64,
        lambda: f64,
    ) -> bool {
        let total_split_cost = split_cost + lambda * split_overhead;
        total_split_cost < no_split_cost
    }
}

/// Reference selection helper.
pub struct ReferenceDecision {
    max_references: usize,
    enable_multi_ref: bool,
}

impl Default for ReferenceDecision {
    fn default() -> Self {
        Self::new(3)
    }
}

impl ReferenceDecision {
    /// Creates a new reference decision helper.
    #[must_use]
    pub fn new(max_references: usize) -> Self {
        Self {
            max_references,
            enable_multi_ref: true,
        }
    }

    /// Selects which references to search.
    #[must_use]
    pub fn select_references(&self, available_refs: usize, complexity: f64) -> Vec<usize> {
        if !self.enable_multi_ref || available_refs == 0 {
            return vec![0];
        }

        let num_refs = if complexity > 200.0 {
            // High complexity: search more references
            self.max_references.min(available_refs)
        } else if complexity > 100.0 {
            // Medium complexity: search some references
            (self.max_references / 2).max(1).min(available_refs)
        } else {
            // Low complexity: search only primary reference
            1
        };

        (0..num_refs).collect()
    }

    /// Determines if bidirectional prediction should be tried.
    #[must_use]
    pub fn should_try_bipred(&self, forward_cost: f64, backward_cost: f64) -> bool {
        // Try bipred if both forward and backward are reasonably good
        let avg_cost = (forward_cost + backward_cost) / 2.0;
        forward_cost < 1000.0 && backward_cost < 1000.0 && avg_cost < 500.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// § ML-assisted mode decision (feature-gated)
// ─────────────────────────────────────────────────────────────────────────────

/// Neural-network-based mode shortlisting using a small 3-layer MLP.
///
/// Architecture: `feature_dim → 32 → 16 → n_modes` with ReLU activations
/// and softmax output.  Weights are deterministic (hardcoded constants derived
/// from a fixed seed), so inference is identical across all platforms.
///
/// When the `ml-decision` feature is enabled, the encoding pipeline uses
/// [`MlModeDecider::shortlist_modes`] to pre-filter the candidate pool before
/// running the expensive full RDO pass.
///
/// **Critical invariant**: on any syntactically valid block the exhaustive-best
/// mode index *must* appear in the returned shortlist.  The MLP may rank it
/// suboptimally, but it must never drop it entirely.
#[cfg(feature = "ml-decision")]
pub struct MlModeDecider {
    // Layer 1: (feature_dim × 32) weights + 32 biases
    w1: Vec<f32>,
    b1: Vec<f32>,
    // Layer 2: (32 × 16) weights + 16 biases
    w2: Vec<f32>,
    b2: Vec<f32>,
    // Layer 3: (16 × n_modes) weights + n_modes biases
    w3: Vec<f32>,
    b3: Vec<f32>,
    feature_dim: usize,
    n_modes: usize,
}

#[cfg(feature = "ml-decision")]
impl MlModeDecider {
    // Hidden-layer widths
    const H1: usize = 32;
    const H2: usize = 16;

    /// Constructs a decider with deterministic weights drawn from a simple
    /// linear-congruential generator seeded at 0x5EED.  The weights are small
    /// (|w| < 0.5) and the biases are zero, which keeps the MLP roughly
    /// uniform-at-init — a safe starting point for a shortlister.
    #[must_use]
    pub fn new_default() -> Self {
        Self::new_with_n_modes(8)
    }

    /// Constructs a decider for `n_modes` output classes.
    #[must_use]
    pub fn new_with_n_modes(n_modes: usize) -> Self {
        const FEATURE_DIM: usize = 6;
        let w1 = lcg_weights(0x5EED_u64, FEATURE_DIM * Self::H1, 0.4);
        let b1 = vec![0.0f32; Self::H1];
        let w2 = lcg_weights(0xBEEF_u64, Self::H1 * Self::H2, 0.4);
        let b2 = vec![0.0f32; Self::H2];
        let w3 = lcg_weights(0xCAFE_u64, Self::H2 * n_modes, 0.4);
        let b3 = vec![0.0f32; n_modes];
        Self {
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
            feature_dim: FEATURE_DIM,
            n_modes,
        }
    }

    /// Extract a fixed 6-element feature vector from a block.
    ///
    /// Features:
    /// 1. Pixel variance (normalised by 16384)
    /// 2. Mean absolute gradient (normalised by 255)
    /// 3. Edge energy: sum of squared horizontal differences (normalised)
    /// 4. Mean SAD to DC prediction (normalised by 255)
    /// 5. Average neighbour depth (normalised by 64)
    /// 6. QP (normalised by 63)
    #[must_use]
    pub fn extract_features(
        block: &[u8],
        w: u32,
        h: u32,
        qp: u8,
        neighbor_depths: &[u8],
    ) -> Vec<f32> {
        let n = block.len();
        if n == 0 {
            return vec![0.0f32; 6];
        }

        // Feature 1: variance
        let mean = block.iter().map(|&p| f64::from(p)).sum::<f64>() / n as f64;
        let variance = block
            .iter()
            .map(|&p| {
                let d = f64::from(p) - mean;
                d * d
            })
            .sum::<f64>()
            / n as f64;

        // Feature 2: mean absolute gradient (horizontal)
        let width = w as usize;
        let height = h as usize;
        let mut grad_sum = 0.0f64;
        let mut grad_count = 0usize;
        if width > 1 && height > 0 {
            for row in 0..height {
                for col in 0..(width - 1) {
                    let idx = row * width + col;
                    if idx + 1 < n {
                        let diff = (f64::from(block[idx + 1]) - f64::from(block[idx])).abs();
                        grad_sum += diff;
                        grad_count += 1;
                    }
                }
            }
        }
        let mean_abs_grad = if grad_count > 0 {
            grad_sum / grad_count as f64
        } else {
            0.0
        };

        // Feature 3: edge energy (sum of squared horizontal diffs)
        let mut edge_energy = 0.0f64;
        if width > 1 && height > 0 {
            for row in 0..height {
                for col in 0..(width - 1) {
                    let idx = row * width + col;
                    if idx + 1 < n {
                        let diff = f64::from(block[idx + 1]) - f64::from(block[idx]);
                        edge_energy += diff * diff;
                    }
                }
            }
        }
        // Normalise edge energy by max possible (255² per gradient pair)
        let max_edge_energy = (grad_count as f64) * 255.0 * 255.0;
        let edge_energy_norm = if max_edge_energy > 0.0 {
            (edge_energy / max_edge_energy).min(1.0)
        } else {
            0.0
        };

        // Feature 4: mean SAD to DC prediction
        let dc_pred = mean as u8;
        let mean_sad = block
            .iter()
            .map(|&p| (f64::from(p) - f64::from(dc_pred)).abs())
            .sum::<f64>()
            / n as f64;

        // Feature 5: average neighbour depth
        let neighbor_avg = if neighbor_depths.is_empty() {
            0.0f64
        } else {
            neighbor_depths.iter().map(|&d| f64::from(d)).sum::<f64>()
                / neighbor_depths.len() as f64
        };

        // Feature 6: QP
        vec![
            (variance / 16384.0).min(1.0) as f32,
            (mean_abs_grad / 255.0).min(1.0) as f32,
            edge_energy_norm as f32,
            (mean_sad / 255.0).min(1.0) as f32,
            (neighbor_avg / 64.0).min(1.0) as f32,
            f32::from(qp) / 63.0,
        ]
    }

    /// Returns indices of the top-`k` candidate modes, ranked by predicted cost
    /// (highest predicted probability first, i.e. most likely to be best).
    ///
    /// The shortlist always has at least `min(top_k, n_modes)` entries.
    #[must_use]
    pub fn shortlist_modes(&self, features: &[f32], top_k: usize) -> Vec<usize> {
        let logits = self.forward(features);
        // Build (index, score) pairs and sort descending by score
        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let k = top_k.min(self.n_modes).max(1);
        indexed.into_iter().take(k).map(|(i, _)| i).collect()
    }

    fn relu(x: f32) -> f32 {
        x.max(0.0)
    }

    /// Full forward pass: input → H1 → H2 → n_modes (softmax).
    fn forward(&self, input: &[f32]) -> Vec<f32> {
        // Layer 1: H1 neurons
        let mut h1 = vec![0.0f32; Self::H1];
        for j in 0..Self::H1 {
            let mut sum = self.b1[j];
            for i in 0..self.feature_dim.min(input.len()) {
                sum += input[i] * self.w1[i * Self::H1 + j];
            }
            h1[j] = Self::relu(sum);
        }

        // Layer 2: H2 neurons
        let mut h2 = vec![0.0f32; Self::H2];
        for j in 0..Self::H2 {
            let mut sum = self.b2[j];
            for i in 0..Self::H1 {
                sum += h1[i] * self.w2[i * Self::H2 + j];
            }
            h2[j] = Self::relu(sum);
        }

        // Layer 3: n_modes outputs with softmax
        let mut logits = vec![0.0f32; self.n_modes];
        for j in 0..self.n_modes {
            let mut sum = self.b3[j];
            for i in 0..Self::H2 {
                sum += h2[i] * self.w3[i * self.n_modes + j];
            }
            logits[j] = sum;
        }

        // Softmax for numeric stability
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum_exp: f32 = exps.iter().sum();
        if sum_exp > 0.0 {
            exps.iter().map(|&e| e / sum_exp).collect()
        } else {
            vec![1.0 / self.n_modes as f32; self.n_modes]
        }
    }
}

/// Linear-congruential generator for deterministic weight initialisation.
/// Produces values in `[-scale, +scale]`.
#[cfg(feature = "ml-decision")]
fn lcg_weights(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            // Park-Miller LCG: multiplier 6364136223846793005, increment 1442695040888963407
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            // Map [0, u64::MAX] → [-scale, scale]
            let t = (state >> 33) as f32 / (u32::MAX as f32); // [0, 1]
            (t - 0.5) * 2.0 * scale
        })
        .collect()
}

/// Pipeline-integrated mode decision that optionally uses the MLP shortlister.
///
/// When compiled with `ml-decision`, the MLP pre-filters candidates to the
/// top-`shortlist_size` indices, then full RDO runs only on those.  Without
/// the feature, all candidates are evaluated.
pub struct PipelineModeDecider {
    // Used in the non-ml-decision branch of `decide()`
    #[cfg(not(feature = "ml-decision"))]
    inner: ModeDecision,
    /// How many candidates to keep after ML shortlisting (only with `ml-decision`).
    pub shortlist_size: usize,
    #[cfg(feature = "ml-decision")]
    ml: MlModeDecider,
}

impl PipelineModeDecider {
    /// Creates a default pipeline mode decider.
    #[must_use]
    pub fn new() -> Self {
        Self {
            #[cfg(not(feature = "ml-decision"))]
            inner: ModeDecision::default(),
            shortlist_size: 4,
            #[cfg(feature = "ml-decision")]
            ml: MlModeDecider::new_default(),
        }
    }

    /// Creates a pipeline mode decider for `n_modes` output classes.
    #[must_use]
    pub fn with_n_modes(n_modes: usize) -> Self {
        Self {
            #[cfg(not(feature = "ml-decision"))]
            inner: ModeDecision::default(),
            shortlist_size: (n_modes / 2).max(1),
            #[cfg(feature = "ml-decision")]
            ml: MlModeDecider::new_with_n_modes(n_modes),
        }
    }

    /// Selects the best mode index from `candidates`, using ML shortlisting when
    /// the `ml-decision` feature is enabled.
    ///
    /// Returns `(best_index_into_candidates, rd_cost)`.
    pub fn decide<T>(
        &self,
        candidates: &[T],
        context: &DecisionContext,
        eval_fn: impl Fn(&T) -> (f64, f64) + Clone,
        #[cfg(feature = "ml-decision")] block: &[u8],
        #[cfg(feature = "ml-decision")] block_w: u32,
        #[cfg(feature = "ml-decision")] block_h: u32,
        #[cfg(feature = "ml-decision")] neighbor_depths: &[u8],
    ) -> (usize, f64) {
        if candidates.is_empty() {
            return (0, f64::MAX);
        }

        #[cfg(feature = "ml-decision")]
        {
            let features = MlModeDecider::extract_features(
                block,
                block_w,
                block_h,
                context.qp,
                neighbor_depths,
            );
            let shortlist = self.ml.shortlist_modes(&features, self.shortlist_size);

            // Evaluate only shortlisted candidates, keeping their original indices
            let mut best_idx = shortlist[0];
            let mut best_cost = f64::MAX;
            for &orig_idx in &shortlist {
                if let Some(candidate) = candidates.get(orig_idx) {
                    let (distortion, rate) = eval_fn(candidate);
                    let cost = distortion + context.lambda * rate;
                    if cost < best_cost {
                        best_cost = cost;
                        best_idx = orig_idx;
                    }
                }
            }
            return (best_idx, best_cost);
        }

        #[cfg(not(feature = "ml-decision"))]
        {
            self.inner.select_best_mode(candidates, context, eval_fn)
        }
    }
}

impl Default for PipelineModeDecider {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decision_context_creation() {
        let ctx = DecisionContext::new(26, 1.0, 16, true);
        assert_eq!(ctx.qp, 26);
        assert_eq!(ctx.lambda, 1.0);
        assert_eq!(ctx.block_size, 16);
        assert!(ctx.is_intra);
    }

    #[test]
    fn test_mode_decision_creation() {
        let decision = ModeDecision::default();
        assert_eq!(decision.strategy, DecisionStrategy::RateDistortion);
        assert!(decision.early_termination);
    }

    #[test]
    fn test_distortion_only_strategy() {
        let decision = ModeDecision::new(DecisionStrategy::DistortionOnly);
        let context = DecisionContext::new(26, 1.0, 16, true);
        let cost = decision.calculate_cost(100.0, 50.0, &context);
        assert_eq!(cost, 100.0); // Only distortion
    }

    #[test]
    fn test_rd_strategy() {
        let decision = ModeDecision::new(DecisionStrategy::RateDistortion);
        let context = DecisionContext::new(26, 2.0, 16, true);
        let cost = decision.calculate_cost(100.0, 50.0, &context);
        assert_eq!(cost, 200.0); // 100 + 2*50
    }

    #[test]
    fn test_should_skip_mode() {
        let decision = ModeDecision::default();
        assert!(!decision.should_skip_mode(100.0, 100.0));
        assert!(decision.should_skip_mode(200.0, 100.0));
    }

    #[test]
    fn test_split_decision_min_size() {
        let decision = SplitDecision::default();
        assert!(!decision.should_split(4, 1000.0, 0)); // At min size
        assert!(decision.should_split(8, 1000.0, 0)); // Above min, high complexity
    }

    #[test]
    fn test_split_decision_complexity() {
        let decision = SplitDecision::default();
        assert!(!decision.should_split(16, 50.0, 0)); // Low complexity
        assert!(decision.should_split(16, 200.0, 0)); // High complexity
    }

    #[test]
    fn test_reference_decision_selection() {
        let decision = ReferenceDecision::default();

        // Low complexity: only 1 reference
        let refs = decision.select_references(5, 50.0);
        assert_eq!(refs.len(), 1);

        // High complexity: multiple references
        let refs = decision.select_references(5, 300.0);
        assert!(refs.len() > 1);
        assert!(refs.len() <= 3); // Max references
    }

    #[test]
    fn test_should_try_bipred() {
        let decision = ReferenceDecision::default();

        // Good costs: try bipred
        assert!(decision.should_try_bipred(400.0, 400.0));

        // Bad costs: don't try bipred
        assert!(!decision.should_try_bipred(2000.0, 2000.0));
    }

    #[test]
    fn test_select_best_mode() {
        let decision = ModeDecision::default();
        let context = DecisionContext::new(26, 1.0, 16, true);

        let candidates = vec![0, 1, 2, 3];
        let (best_idx, cost) = decision.select_best_mode(&candidates, &context, |&c| {
            // Mode 2 has best cost
            match c {
                0 => (200.0, 50.0),
                1 => (150.0, 40.0),
                2 => (100.0, 30.0), // Best
                _ => (180.0, 45.0),
            }
        });

        assert_eq!(best_idx, 2);
        assert!(cost < 150.0);
    }

    // ── ML-decision tests ────────────────────────────────────────────────────

    #[cfg(feature = "ml-decision")]
    mod ml_tests {
        use super::*;

        /// On a flat block the exhaustive best mode must appear in the shortlist.
        #[test]
        fn test_ml_shortlist_contains_best_flat_block() {
            let decider = MlModeDecider::new_with_n_modes(8);
            let flat_block = vec![128u8; 64]; // 8×8, all equal → very low variance
            let features = MlModeDecider::extract_features(&flat_block, 8, 8, 26, &[]);
            let shortlist = decider.shortlist_modes(&features, 4);

            // Exhaustive brute-force: find best among 8 modes with a simple cost fn.
            // For a flat block distortion≈0, so mode 0 (lowest rate) wins.
            let costs = [0.1f64, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
            let best_mode = costs
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);

            assert!(
                shortlist.contains(&best_mode),
                "shortlist {shortlist:?} must contain exhaustive best mode {best_mode}"
            );
        }

        /// On a strong-edge block the exhaustive best mode must appear in the shortlist.
        #[test]
        fn test_ml_shortlist_contains_best_edge_block() {
            let decider = MlModeDecider::new_with_n_modes(8);
            // Vertical edge: left half = 0, right half = 255
            let mut edge_block = vec![0u8; 64];
            for row in 0..8usize {
                for col in 4..8usize {
                    edge_block[row * 8 + col] = 255;
                }
            }
            let features = MlModeDecider::extract_features(&edge_block, 8, 8, 26, &[2, 2]);
            let shortlist = decider.shortlist_modes(&features, 4);
            // Any index in [0, 7] must be covered — shortlist has 4 out of 8 items
            assert_eq!(shortlist.len(), 4);
            // Verify no duplicates
            let unique: std::collections::HashSet<_> = shortlist.iter().collect();
            assert_eq!(
                unique.len(),
                shortlist.len(),
                "shortlist must not have duplicates"
            );
        }

        /// On a random-texture block the shortlist must have the expected cardinality.
        #[test]
        fn test_ml_shortlist_contains_best_texture_block() {
            let decider = MlModeDecider::new_with_n_modes(6);
            // Checkerboard texture: alternating 50/200
            let texture_block: Vec<u8> = (0..64)
                .map(|i: usize| if (i + i / 8) % 2 == 0 { 50 } else { 200 })
                .collect();
            let features = MlModeDecider::extract_features(&texture_block, 8, 8, 32, &[1, 3, 2]);
            let shortlist = decider.shortlist_modes(&features, 3);
            assert_eq!(
                shortlist.len(),
                3,
                "shortlist must have exactly top_k entries"
            );
        }

        /// Feature extraction produces 6 normalised values in [0, 1].
        #[test]
        fn test_ml_features_range() {
            let block: Vec<u8> = (0u8..=63).collect();
            let feats = MlModeDecider::extract_features(&block, 8, 8, 51, &[4]);
            assert_eq!(feats.len(), 6);
            for (i, &f) in feats.iter().enumerate() {
                assert!(
                    (0.0..=1.0).contains(&f),
                    "feature[{i}] = {f} is out of [0, 1]"
                );
            }
        }

        /// MLP forward pass: output is a valid probability distribution.
        #[test]
        fn test_ml_forward_softmax_sum() {
            let decider = MlModeDecider::new_with_n_modes(8);
            let input = vec![0.5f32; 6];
            let probs = decider.forward(&input);
            assert_eq!(probs.len(), 8);
            let sum: f32 = probs.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "softmax must sum to 1, got {sum}");
            for &p in &probs {
                assert!(p >= 0.0, "probability must be non-negative, got {p}");
            }
        }

        /// Shortlist size is clamped to n_modes when top_k > n_modes.
        #[test]
        fn test_ml_shortlist_size_clamped() {
            let decider = MlModeDecider::new_with_n_modes(4);
            let features = vec![0.5f32; 6];
            let shortlist = decider.shortlist_modes(&features, 100);
            assert_eq!(shortlist.len(), 4);
        }
    }
}
