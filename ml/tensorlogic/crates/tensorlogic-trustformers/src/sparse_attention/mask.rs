//! Longformer-style attention mask generation.
//!
//! Builds a dense `seq_len x seq_len` boolean mask encoding which (query, key)
//! pairs are attended.  The mask combines three rules:
//!
//! 1. **Sliding window**: position `i` attends to `j` if `|i - j| <= window_size`.
//! 2. **Global tokens**: any position listed in `global_token_indices` attends to
//!    *all* positions, and *all* positions attend to it.
//! 3. **Causal constraint** (optional): `j > i` is never attended.

use super::config::SparseAttentionConfig;
use super::error::{SparseAttentionError, SparseAttentionResult};

/// A dense boolean attention mask for a single head.
///
/// `data[i][j] == true` means position `i` can attend to position `j`.
/// This is the research-preview representation; a sparse CSR layout is a
/// v0.2.0 optimisation target.
#[derive(Clone, Debug)]
pub struct AttentionMask {
    /// Dense `seq_len x seq_len` attendance matrix.
    pub data: Vec<Vec<bool>>,
    /// Sequence length this mask was built for.
    pub seq_len: usize,
}

impl AttentionMask {
    /// Query whether position `i` attends to position `j`.
    ///
    /// Out-of-bounds indices return `false` rather than panicking, so
    /// callers that iterate a fixed grid need not bounds-check.
    pub fn is_attended(&self, i: usize, j: usize) -> bool {
        self.data
            .get(i)
            .and_then(|row| row.get(j).copied())
            .unwrap_or(false)
    }

    /// Count the number of `true` entries (attended pairs).
    pub fn attended_count(&self) -> usize {
        self.data
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&b| b)
            .count()
    }

    /// Sparsity ratio: fraction of entries that are `false`.
    pub fn sparsity(&self) -> f64 {
        let total = self.seq_len * self.seq_len;
        if total == 0 {
            return 0.0;
        }
        1.0 - (self.attended_count() as f64 / total as f64)
    }
}

/// Build a [`AttentionMask`] from the given configuration and sequence length.
///
/// # Errors
///
/// Returns [`SparseAttentionError::InvalidSequenceLength`] if `seq_len == 0`,
/// or [`SparseAttentionError::InvalidGlobalIndices`] if any global index is
/// out of bounds.
pub fn build_mask(
    seq_len: usize,
    config: &SparseAttentionConfig,
) -> SparseAttentionResult<AttentionMask> {
    if seq_len == 0 {
        return Err(SparseAttentionError::InvalidSequenceLength(0));
    }
    config.validate()?;
    config.validate_globals_for_seq_len(seq_len)?;

    let global_set: std::collections::HashSet<usize> =
        config.global_token_indices.iter().copied().collect();

    let mut data = vec![vec![false; seq_len]; seq_len];

    for (i, row) in data.iter_mut().enumerate() {
        let i_is_global = global_set.contains(&i);

        for (j, cell) in row.iter_mut().enumerate() {
            if config.causal && j > i {
                continue;
            }

            let j_is_global = global_set.contains(&j);
            let in_window = i.abs_diff(j) <= config.window_size;

            if in_window || i_is_global || j_is_global {
                *cell = true;
            }
        }
    }

    Ok(AttentionMask { data, seq_len })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(window: usize, globals: Vec<usize>, causal: bool) -> SparseAttentionConfig {
        SparseAttentionConfig::new(window, 1, 4)
            .map(|c| c.with_global_tokens(globals).with_causal(causal))
            .expect("test config should be valid")
    }

    #[test]
    fn sliding_window_small() {
        let cfg = make_config(1, vec![], false);
        let mask = build_mask(5, &cfg).expect("mask should build");

        // Position 0 attends to [0,1], position 2 attends to [1,2,3], etc.
        assert!(mask.is_attended(0, 0));
        assert!(mask.is_attended(0, 1));
        assert!(!mask.is_attended(0, 2));
        assert!(mask.is_attended(2, 1));
        assert!(mask.is_attended(2, 2));
        assert!(mask.is_attended(2, 3));
        assert!(!mask.is_attended(2, 4));
    }

    #[test]
    fn global_tokens_attend_everywhere() {
        let cfg = make_config(1, vec![0], false);
        let mask = build_mask(5, &cfg).expect("mask should build");

        // Position 0 (global) attends to all
        for j in 0..5 {
            assert!(mask.is_attended(0, j), "global 0 should attend to {}", j);
        }
        // All positions attend to global 0
        for i in 0..5 {
            assert!(
                mask.is_attended(i, 0),
                "position {} should attend to global 0",
                i
            );
        }
    }

    #[test]
    fn causal_mask_blocks_future() {
        let cfg = make_config(10, vec![], true);
        let mask = build_mask(5, &cfg).expect("mask should build");

        // window >= seq_len so all non-causal pairs would be attended
        for i in 0..5 {
            for j in 0..5 {
                if j <= i {
                    assert!(mask.is_attended(i, j));
                } else {
                    assert!(!mask.is_attended(i, j));
                }
            }
        }
    }

    #[test]
    fn zero_seq_len_rejected() {
        let cfg = make_config(1, vec![], false);
        assert!(matches!(
            build_mask(0, &cfg),
            Err(SparseAttentionError::InvalidSequenceLength(0))
        ));
    }

    #[test]
    fn oob_global_rejected() {
        let cfg = make_config(1, vec![100], false);
        assert!(matches!(
            build_mask(8, &cfg),
            Err(SparseAttentionError::InvalidGlobalIndices { .. })
        ));
    }

    #[test]
    fn sparsity_metric() {
        let cfg = make_config(1, vec![], false);
        let mask = build_mask(8, &cfg).expect("mask should build");
        // Each row has at most 3 true entries (window of 3) for interior,
        // 2 for edges.  Sparsity should be > 0.
        assert!(mask.sparsity() > 0.0);
    }

    #[test]
    fn out_of_bounds_returns_false() {
        let cfg = make_config(1, vec![], false);
        let mask = build_mask(4, &cfg).expect("mask should build");
        assert!(!mask.is_attended(100, 0));
        assert!(!mask.is_attended(0, 100));
    }
}
