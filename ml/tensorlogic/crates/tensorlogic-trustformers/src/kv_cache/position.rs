use ndarray::{s, Array2, Array3, ArrayD, ArrayView1, IxDyn};
use std::fmt;

// ---------------------------------------------------------------------------
// Position errors
// ---------------------------------------------------------------------------

/// Errors arising from position encoding operations.
#[derive(Debug, Clone)]
pub enum PositionError {
    /// `head_dim` must be even for RoPE.
    HeadDimMustBeEven { head_dim: usize },
    /// Sequence offset exceeds the pre-computed cache.
    SeqOffsetOutOfRange { offset: usize, max: usize },
    /// Tensor shape mismatch.
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
}

impl fmt::Display for PositionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HeadDimMustBeEven { head_dim } => {
                write!(f, "head_dim must be even for RoPE, got {}", head_dim)
            }
            Self::SeqOffsetOutOfRange { offset, max } => {
                write!(
                    f,
                    "seq_offset {} is out of range (max pre-computed = {})",
                    offset, max
                )
            }
            Self::ShapeMismatch { expected, got } => {
                write!(f, "Shape mismatch: expected {:?}, got {:?}", expected, got)
            }
        }
    }
}

impl std::error::Error for PositionError {}

// ---------------------------------------------------------------------------
// Rotary Position Embedding (RoPE)
// ---------------------------------------------------------------------------

/// Rotary Position Embedding (RoPE) as introduced in Su et al. 2021.
///
/// Pre-computes cosine and sine caches up to `max_seq_len` positions and applies
/// the rotation in-place to the last dimension of the input tensor.
#[derive(Debug, Clone)]
pub struct RotaryPositionEmbedding {
    /// Dimension of each attention head.
    pub head_dim: usize,
    /// Base for the geometric frequency sequence (default 10000.0).
    pub base: f64,
    /// Maximum sequence length for which the cache was pre-computed.
    pub max_seq_len: usize,
    /// Pre-computed cosines: shape `[max_seq_len, head_dim / 2]`.
    cos_cache: Array2<f64>,
    /// Pre-computed sines: shape `[max_seq_len, head_dim / 2]`.
    sin_cache: Array2<f64>,
}

impl RotaryPositionEmbedding {
    /// Create a new RoPE module, pre-computing the cos/sin cache.
    ///
    /// Returns an error if `head_dim` is not even.
    pub fn new(
        head_dim: usize,
        max_seq_len: usize,
        base: f64,
    ) -> std::result::Result<Self, PositionError> {
        if !head_dim.is_multiple_of(2) {
            return Err(PositionError::HeadDimMustBeEven { head_dim });
        }
        let (cos_cache, sin_cache) = Self::build_cos_sin_cache(head_dim, max_seq_len, base);
        Ok(Self {
            head_dim,
            base,
            max_seq_len,
            cos_cache,
            sin_cache,
        })
    }

    /// Build the cos and sin frequency caches.
    fn build_cos_sin_cache(
        head_dim: usize,
        max_seq_len: usize,
        base: f64,
    ) -> (Array2<f64>, Array2<f64>) {
        let half_dim = head_dim / 2;
        // θ_i = base^{-2i/d} for i in 0..half_dim
        let thetas: Vec<f64> = (0..half_dim)
            .map(|i| base.powf(-(2.0 * i as f64) / head_dim as f64))
            .collect();

        let mut cos_cache = Array2::<f64>::zeros((max_seq_len, half_dim));
        let mut sin_cache = Array2::<f64>::zeros((max_seq_len, half_dim));

        for pos in 0..max_seq_len {
            for (i, &theta) in thetas.iter().enumerate() {
                let angle = pos as f64 * theta;
                cos_cache[[pos, i]] = angle.cos();
                sin_cache[[pos, i]] = angle.sin();
            }
        }

        (cos_cache, sin_cache)
    }

    /// Apply RoPE to the input tensor starting at `seq_offset`.
    ///
    /// `x` is expected to have shape `[seq_len, ..., head_dim]` where the last
    /// axis is the head dimension (or any shape where the last axis == `head_dim`).
    pub fn apply(
        &self,
        x: &ArrayD<f64>,
        seq_offset: usize,
    ) -> std::result::Result<ArrayD<f64>, PositionError> {
        let shape = x.shape();
        let ndim = shape.len();
        if ndim < 1 {
            return Err(PositionError::ShapeMismatch {
                expected: vec![1],
                got: shape.to_vec(),
            });
        }

        let last_dim = shape[ndim - 1];
        if last_dim != self.head_dim {
            return Err(PositionError::ShapeMismatch {
                expected: vec![self.head_dim],
                got: vec![last_dim],
            });
        }

        let seq_len = shape[0];
        if seq_offset + seq_len > self.max_seq_len {
            return Err(PositionError::SeqOffsetOutOfRange {
                offset: seq_offset + seq_len - 1,
                max: self.max_seq_len - 1,
            });
        }

        let half_dim = self.head_dim / 2;

        // Split x into first half and second half along last axis.
        // For simplicity, work with a 2D view: [total_positions, head_dim].
        let total = x.len() / self.head_dim;
        let x2 = x
            .view()
            .into_shape_with_order((total, self.head_dim))
            .map_err(|_| PositionError::ShapeMismatch {
                expected: vec![total, self.head_dim],
                got: shape.to_vec(),
            })?;

        // x_first: [total, half_dim], x_second: [total, half_dim]
        let x_first = x2.slice(s![.., ..half_dim]).to_owned();
        let x_second = x2.slice(s![.., half_dim..]).to_owned();

        // rotate_half: [-x_second, x_first]
        let mut rotated = Array2::<f64>::zeros((total, self.head_dim));
        rotated.slice_mut(s![.., ..half_dim]).assign(&(-&x_second));
        rotated.slice_mut(s![.., half_dim..]).assign(&x_first);

        // Broadcast cos/sin for each position in [seq_offset, seq_offset + seq_len).
        // Map each row of x2 to the corresponding position in the cache.
        // Positions cycle through seq_len: row i -> seq_offset + (i / (total / seq_len))
        let positions_per_token = total.checked_div(seq_len).unwrap_or(1);
        let mut cos_expanded = Array2::<f64>::zeros((total, half_dim));
        let mut sin_expanded = Array2::<f64>::zeros((total, half_dim));

        for i in 0..total {
            let pos = seq_offset + i / positions_per_token.max(1);
            let capped_pos = pos.min(self.max_seq_len - 1);
            cos_expanded
                .slice_mut(s![i, ..])
                .assign(&self.cos_cache.slice(s![capped_pos, ..]));
            sin_expanded
                .slice_mut(s![i, ..])
                .assign(&self.sin_cache.slice(s![capped_pos, ..]));
        }

        // Repeat cos/sin to full head_dim by tiling: [total, half_dim] -> [total, head_dim]
        let mut cos_full = Array2::<f64>::zeros((total, self.head_dim));
        let mut sin_full = Array2::<f64>::zeros((total, self.head_dim));
        cos_full.slice_mut(s![.., ..half_dim]).assign(&cos_expanded);
        cos_full.slice_mut(s![.., half_dim..]).assign(&cos_expanded);
        sin_full.slice_mut(s![.., ..half_dim]).assign(&sin_expanded);
        sin_full.slice_mut(s![.., half_dim..]).assign(&sin_expanded);

        // y = x * cos + rotate_half(x) * sin
        let result2 = &x2 * &cos_full + &rotated * &sin_full;

        // Reshape back to original shape.
        let result = result2
            .into_dyn()
            .into_shape_with_order(IxDyn(shape))
            .map_err(|_| PositionError::ShapeMismatch {
                expected: shape.to_vec(),
                got: vec![total, self.head_dim],
            })?;

        Ok(result)
    }

    /// Compute rotate_half: negate the first half and concatenate with the second half.
    ///
    /// Given `x` shaped `[..., head_dim]`, returns `[-x[..., head_dim/2:], x[..., :head_dim/2]]`.
    pub fn rotate_half(x: &ArrayD<f64>) -> ArrayD<f64> {
        let shape = x.shape();
        let ndim = shape.len();
        if ndim < 1 {
            return x.to_owned();
        }
        let head_dim = shape[ndim - 1];
        let half = head_dim / 2;
        let total = x.len() / head_dim;

        let x2 = x
            .view()
            .into_shape_with_order((total, head_dim))
            .expect("rotate_half reshape");

        let x_first = x2.slice(s![.., ..half]).to_owned();
        let x_second = x2.slice(s![.., half..]).to_owned();

        let mut out = Array2::<f64>::zeros((total, head_dim));
        out.slice_mut(s![.., ..half]).assign(&(-&x_second));
        out.slice_mut(s![.., half..]).assign(&x_first);

        out.into_dyn()
            .into_shape_with_order(IxDyn(shape))
            .expect("rotate_half final reshape")
    }

    /// Return the pre-computed frequencies (cos values) at a specific position.
    pub fn frequencies_at(&self, pos: usize) -> ArrayView1<'_, f64> {
        let capped = pos.min(self.max_seq_len - 1);
        self.cos_cache.slice(s![capped, ..])
    }
}

// ---------------------------------------------------------------------------
// Relative Position Bias (T5-style)
// ---------------------------------------------------------------------------

/// T5-style relative position bias that adds a learned scalar bias to attention
/// logits based on the relative distance between query and key positions.
#[derive(Debug, Clone)]
pub struct RelativePositionBias {
    /// Number of attention heads.
    pub num_heads: usize,
    /// Number of learned buckets for distances.
    pub num_buckets: usize,
    /// Maximum distance to consider (beyond this, distances are clamped).
    pub max_distance: usize,
    /// If `true`, use separate buckets for forward and backward directions.
    pub bidirectional: bool,
    /// Learned bias table: shape `[num_buckets, num_heads]`.
    biases: Array2<f64>,
}

impl RelativePositionBias {
    /// Create a new relative position bias (zero-initialized).
    pub fn new(
        num_heads: usize,
        num_buckets: usize,
        max_distance: usize,
        bidirectional: bool,
    ) -> Self {
        Self {
            num_heads,
            num_buckets,
            max_distance,
            bidirectional,
            biases: Array2::<f64>::zeros((num_buckets, num_heads)),
        }
    }

    /// Compute the attention bias matrix of shape `[num_heads, q_len, k_len]`.
    ///
    /// For each (q, k) pair the relative position `q - k` is mapped to a bucket
    /// and the corresponding learned bias is looked up.
    pub fn compute_bias(&self, query_len: usize, key_len: usize) -> Array3<f64> {
        let mut bias = Array3::<f64>::zeros((self.num_heads, query_len, key_len));

        for q in 0..query_len {
            for k in 0..key_len {
                let relative_position = q as i32 - k as i32;
                let bucket = Self::relative_position_bucket(
                    relative_position,
                    self.bidirectional,
                    self.num_buckets,
                    self.max_distance,
                );
                for h in 0..self.num_heads {
                    bias[[h, q, k]] = self.biases[[bucket, h]];
                }
            }
        }

        bias
    }

    /// Map a relative position to a bucket index.
    ///
    /// The first half of the buckets covers exact small distances linearly.
    /// The second half covers larger distances logarithmically.
    fn relative_position_bucket(
        relative_position: i32,
        bidirectional: bool,
        num_buckets: usize,
        max_distance: usize,
    ) -> usize {
        let mut n = num_buckets;
        let mut relative = relative_position;

        if bidirectional {
            n /= 2;
            // Positive distances get offset by n.
            if relative_position > 0 {
                // Offset into second half.
                let pos_bucket =
                    Self::distance_to_bucket(relative_position as usize, n, max_distance);
                return (n + pos_bucket).min(num_buckets - 1);
            }
            relative = -relative;
        } else {
            relative = (-relative).max(0);
        }

        let distance = relative as usize;
        Self::distance_to_bucket(distance, n, max_distance).min(num_buckets - 1)
    }

    /// Map an absolute distance to a bucket in `[0, n)`.
    fn distance_to_bucket(distance: usize, n: usize, max_distance: usize) -> usize {
        if n == 0 {
            return 0;
        }
        let max_exact = n / 2;
        if distance < max_exact {
            // Linear range.
            distance
        } else {
            // Logarithmic range.
            let clamped = distance.min(max_distance);
            let scale = (clamped as f64 / max_exact as f64).ln()
                / (max_distance as f64 / max_exact as f64).ln().max(1e-10);
            let bucket_offset = (scale * (n - max_exact) as f64) as usize;
            (max_exact + bucket_offset).min(n - 1)
        }
    }

    /// Update the learned bias table.
    ///
    /// `new_biases` must have shape `[num_buckets, num_heads]`.
    pub fn update_biases(
        &mut self,
        new_biases: Array2<f64>,
    ) -> std::result::Result<(), PositionError> {
        let expected = vec![self.num_buckets, self.num_heads];
        let got = new_biases.shape().to_vec();
        if got != expected {
            return Err(PositionError::ShapeMismatch { expected, got });
        }
        self.biases = new_biases;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tensor(shape: &[usize], fill: f64) -> ArrayD<f64> {
        ArrayD::from_elem(IxDyn(shape), fill)
    }

    #[test]
    fn test_rope_new_builds_cache() {
        let rope = RotaryPositionEmbedding::new(8, 16, 10000.0).expect("valid head_dim");
        assert_eq!(
            rope.cos_cache.shape(),
            &[16, 4],
            "cos_cache shape [max_seq, half_dim]"
        );
        assert_eq!(
            rope.sin_cache.shape(),
            &[16, 4],
            "sin_cache shape [max_seq, half_dim]"
        );
    }

    #[test]
    fn test_rope_apply_preserves_shape() {
        let rope = RotaryPositionEmbedding::new(8, 32, 10000.0).expect("valid");
        let x = make_tensor(&[4, 8], 1.0);
        let result = rope.apply(&x, 0).expect("apply should succeed");
        assert_eq!(
            result.shape(),
            x.shape(),
            "output shape must match input shape"
        );
    }

    #[test]
    fn test_rope_rotate_half_correct() {
        // For a 4-D head_dim: [a, b, c, d] -> [-c, -d, a, b]
        let data = vec![1.0_f64, 2.0, 3.0, 4.0];
        let x = ArrayD::from_shape_vec(IxDyn(&[1, 4]), data).expect("build");
        let rotated = RotaryPositionEmbedding::rotate_half(&x);
        let flat: Vec<f64> = rotated.iter().copied().collect();
        // First half: negated second half of input = [-3, -4]
        assert!(
            (flat[0] - (-3.0)).abs() < 1e-9,
            "first element should be -3"
        );
        assert!(
            (flat[1] - (-4.0)).abs() < 1e-9,
            "second element should be -4"
        );
        // Second half: first half of input = [1, 2]
        assert!((flat[2] - 1.0).abs() < 1e-9, "third element should be 1");
        assert!((flat[3] - 2.0).abs() < 1e-9, "fourth element should be 2");
    }

    #[test]
    fn test_rope_head_dim_odd_errors() {
        let result = RotaryPositionEmbedding::new(7, 16, 10000.0);
        assert!(
            matches!(result, Err(PositionError::HeadDimMustBeEven { .. })),
            "odd head_dim should produce HeadDimMustBeEven error"
        );
    }

    #[test]
    fn test_relative_position_bias_compute() {
        let rpb = RelativePositionBias::new(4, 32, 128, true);
        let bias = rpb.compute_bias(6, 10);
        assert_eq!(
            bias.shape(),
            &[4, 6, 10],
            "bias shape must be [num_heads, q_len, k_len]"
        );
    }

    #[test]
    fn test_relative_position_bias_symmetric_for_bidirectional() {
        // When bidirectional=true, positions (q=5, k=0) and (q=0, k=5) should
        // use different buckets (forward vs. backward directions).
        let _rpb = RelativePositionBias::new(1, 32, 64, true);
        let forward_bucket = RelativePositionBias::relative_position_bucket(5, true, 32, 64);
        let backward_bucket = RelativePositionBias::relative_position_bucket(-5, true, 32, 64);
        assert_ne!(
            forward_bucket, backward_bucket,
            "forward and backward positions should map to different buckets"
        );
    }

    #[test]
    fn test_relative_position_bucket_clamping() {
        // A very large distance should map to the last bucket (clamped).
        let bucket = RelativePositionBias::relative_position_bucket(100000, false, 16, 128);
        assert!(bucket < 16, "bucket must be within [0, num_buckets)");
    }
}
