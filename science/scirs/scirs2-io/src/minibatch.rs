//! Efficient mini-batch sampler with shuffle and stratified splitting.
//!
//! Provides index-based batch sampling for machine learning pipelines:
//! - Configurable batch size with optional last-batch dropping
//! - Deterministic or random shuffling via seeded xorshift64 PRNG
//! - Stratified sampling that preserves class-label distributions
//! - Train/validation/test split with optional stratification
//!
//! # Example
//! ```rust
//! use scirs2_io::minibatch::{BatchSampler, BatchSamplerConfig};
//!
//! let config = BatchSamplerConfig {
//!     batch_size: 32,
//!     shuffle: true,
//!     seed: Some(42),
//!     drop_last: false,
//!     stratified: false,
//! };
//! let mut sampler = BatchSampler::new(100, config);
//! while let Some(indices) = sampler.next_batch() {
//!     // use `indices` to index into your dataset
//!     let _ = indices;
//! }
//! ```

use std::collections::HashMap;

// ---- Internal PRNG -------------------------------------------------------

/// Minimal xorshift64 PRNG — no external crate dependency.
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    /// Create a new PRNG.  Falls back to a fixed seed when `seed == 0`.
    fn new(seed: u64) -> Self {
        let state = if seed == 0 { 0x853c49e6748fea9b } else { seed };
        Self { state }
    }

    /// Return the next pseudo-random `u64`.
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Return a value in `[0, n)`.
    fn next_usize(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u64() as usize) % n
    }
}

/// In-place Fisher–Yates shuffle using the given PRNG.
fn fisher_yates_shuffle(indices: &mut [usize], rng: &mut Xorshift64) {
    let n = indices.len();
    for i in (1..n).rev() {
        let j = rng.next_usize(i + 1);
        indices.swap(i, j);
    }
}

// ---- Public API -----------------------------------------------------------

/// Configuration for [`BatchSampler`].
#[derive(Debug, Clone)]
pub struct BatchSamplerConfig {
    /// Number of samples per batch.
    pub batch_size: usize,
    /// Whether to shuffle the indices at the start of each epoch.
    pub shuffle: bool,
    /// Optional seed for the PRNG (reproducible shuffles).
    pub seed: Option<u64>,
    /// If `true`, the final incomplete batch is discarded.
    pub drop_last: bool,
    /// If `true`, `from_labels` builds a stratified index ordering
    /// that interleaves class examples to preserve label balance across batches.
    pub stratified: bool,
}

impl Default for BatchSamplerConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            shuffle: false,
            seed: None,
            drop_last: false,
            stratified: false,
        }
    }
}

/// Index-based mini-batch sampler.
///
/// Produces batches of dataset indices rather than the data itself,
/// so it can be used with any storage backend (arrays, files, databases).
pub struct BatchSampler {
    /// The current permutation of sample indices.
    indices: Vec<usize>,
    /// Configuration controlling sampling behaviour.
    config: BatchSamplerConfig,
    /// Position in `indices` for the next batch.
    current_pos: usize,
    /// Number of completed epochs.
    epoch: usize,
    /// PRNG state — kept so epoch seeds are derived deterministically from the
    /// initial seed rather than being random each epoch.
    rng: Xorshift64,
}

impl BatchSampler {
    /// Create a sampler for `n_samples` samples.
    pub fn new(n_samples: usize, config: BatchSamplerConfig) -> Self {
        let seed = config.seed.unwrap_or(0xa1b2c3d4e5f6_u64);
        let mut rng = Xorshift64::new(seed);
        let mut indices: Vec<usize> = (0..n_samples).collect();
        if config.shuffle {
            fisher_yates_shuffle(&mut indices, &mut rng);
        }
        Self {
            indices,
            config,
            current_pos: 0,
            epoch: 0,
            rng,
        }
    }

    /// Create a stratified sampler from a label array.
    ///
    /// The initial `indices` ordering places examples so that each batch
    /// contains a balanced representation of all classes.  Stratification
    /// is performed within each per-class group; class groups are then
    /// interleaved round-robin.
    pub fn from_labels(labels: &[usize], config: BatchSamplerConfig) -> Self {
        let n_samples = labels.len();
        let seed = config.seed.unwrap_or(0xa1b2c3d4e5f6_u64);
        let mut rng = Xorshift64::new(seed);

        let indices = if config.stratified {
            // Group indices by class label.
            let mut class_buckets: HashMap<usize, Vec<usize>> = HashMap::new();
            for (idx, &label) in labels.iter().enumerate() {
                class_buckets.entry(label).or_default().push(idx);
            }

            // Optionally shuffle within each bucket.
            let mut sorted_classes: Vec<usize> = class_buckets.keys().copied().collect();
            sorted_classes.sort_unstable();

            let mut buckets: Vec<Vec<usize>> = sorted_classes
                .into_iter()
                .map(|cls| {
                    let mut bucket = class_buckets.remove(&cls).unwrap_or_default();
                    if config.shuffle {
                        fisher_yates_shuffle(&mut bucket, &mut rng);
                    }
                    bucket
                })
                .collect();

            // Interleave round-robin across buckets.
            let mut interleaved = Vec::with_capacity(n_samples);
            loop {
                let mut any_pushed = false;
                for bucket in &mut buckets {
                    if let Some(idx) = bucket.first().copied() {
                        *bucket = bucket[1..].to_vec();
                        interleaved.push(idx);
                        any_pushed = true;
                    }
                }
                if !any_pushed {
                    break;
                }
            }
            interleaved
        } else {
            let mut idxs: Vec<usize> = (0..n_samples).collect();
            if config.shuffle {
                fisher_yates_shuffle(&mut idxs, &mut rng);
            }
            idxs
        };

        Self {
            indices,
            config,
            current_pos: 0,
            epoch: 0,
            rng,
        }
    }

    /// Return the next batch of indices, or `None` when the epoch is exhausted.
    pub fn next_batch(&mut self) -> Option<Vec<usize>> {
        let remaining = self.indices.len().saturating_sub(self.current_pos);
        if remaining == 0 {
            return None;
        }

        if self.config.drop_last && remaining < self.config.batch_size {
            return None;
        }

        let end = (self.current_pos + self.config.batch_size).min(self.indices.len());
        let batch = self.indices[self.current_pos..end].to_vec();
        self.current_pos = end;
        Some(batch)
    }

    /// Reset for the next epoch, re-shuffling if `config.shuffle` is `true`.
    ///
    /// This also increments the epoch counter.
    pub fn reset(&mut self) {
        self.epoch += 1;
        self.current_pos = 0;
        if self.config.shuffle {
            // Derive a fresh but deterministic seed for this epoch.
            // Mix the epoch number into the RNG state for reproducibility.
            let epoch_mix = self.epoch as u64 * 0x9e3779b97f4a7c15;
            let mut epoch_rng = Xorshift64::new(self.rng.state ^ epoch_mix);
            fisher_yates_shuffle(&mut self.indices, &mut epoch_rng);
            // Advance the main RNG state so subsequent resets differ.
            self.rng.state = epoch_rng.state;
        }
    }

    /// Total number of batches that will be produced per epoch.
    pub fn n_batches(&self) -> usize {
        let n = self.indices.len();
        let bs = self.config.batch_size;
        if bs == 0 {
            return 0;
        }
        if self.config.drop_last {
            n / bs
        } else {
            n.div_ceil(bs)
        }
    }

    /// Return the epoch counter (incremented by each call to [`reset`](Self::reset)).
    pub fn epoch(&self) -> usize {
        self.epoch
    }
}

// ---- Train/Val/Test split ------------------------------------------------

/// Partition `n_samples` into train, validation, and test index sets.
///
/// If `labels` is provided and both `train_frac` + `val_frac` leave room for
/// stratification, the split preserves the class distribution within each fold.
/// Otherwise, indices are shuffled uniformly.
///
/// `train_frac` and `val_frac` must be in `(0, 1)` and sum to less than 1.0.
/// The test set takes the remainder.
///
/// Returns `(train_indices, val_indices, test_indices)`.
pub fn train_val_test_split(
    n_samples: usize,
    labels: Option<&[usize]>,
    train_frac: f64,
    val_frac: f64,
    seed: Option<u64>,
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let rng_seed = seed.unwrap_or(0xdeadbeef_cafebabe);
    let mut rng = Xorshift64::new(rng_seed);

    let do_stratify = labels.is_some();

    if do_stratify {
        let labels = labels.expect("checked above");
        // Group by class.
        let mut class_buckets: HashMap<usize, Vec<usize>> = HashMap::new();
        for (idx, &label) in labels.iter().enumerate() {
            class_buckets.entry(label).or_default().push(idx);
        }

        let mut train_set = Vec::new();
        let mut val_set = Vec::new();
        let mut test_set = Vec::new();

        let mut sorted_classes: Vec<usize> = class_buckets.keys().copied().collect();
        sorted_classes.sort_unstable();

        for cls in sorted_classes {
            let mut bucket = class_buckets.remove(&cls).unwrap_or_default();
            fisher_yates_shuffle(&mut bucket, &mut rng);
            let n = bucket.len();
            let n_train = ((n as f64 * train_frac).round() as usize).min(n);
            let n_val = ((n as f64 * val_frac).round() as usize).min(n.saturating_sub(n_train));
            train_set.extend_from_slice(&bucket[..n_train]);
            val_set.extend_from_slice(&bucket[n_train..n_train + n_val]);
            test_set.extend_from_slice(&bucket[n_train + n_val..]);
        }

        (train_set, val_set, test_set)
    } else {
        let mut all: Vec<usize> = (0..n_samples).collect();
        fisher_yates_shuffle(&mut all, &mut rng);

        let n_train = ((n_samples as f64 * train_frac).round() as usize).min(n_samples);
        let n_val =
            ((n_samples as f64 * val_frac).round() as usize).min(n_samples.saturating_sub(n_train));

        let train_set = all[..n_train].to_vec();
        let val_set = all[n_train..n_train + n_val].to_vec();
        let test_set = all[n_train + n_val..].to_vec();

        (train_set, val_set, test_set)
    }
}

// ---- Tests ---------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_batch_sampler_basic() {
        let config = BatchSamplerConfig {
            batch_size: 10,
            shuffle: false,
            seed: None,
            drop_last: false,
            stratified: false,
        };
        let mut sampler = BatchSampler::new(100, config);
        let n = sampler.n_batches();
        assert_eq!(n, 10, "expected 10 batches for 100 samples / batch_size 10");

        let mut count = 0usize;
        let mut all_indices: Vec<usize> = Vec::new();
        while let Some(batch) = sampler.next_batch() {
            assert_eq!(batch.len(), 10, "each batch should have 10 elements");
            all_indices.extend_from_slice(&batch);
            count += 1;
        }
        assert_eq!(count, 10);
        // Verify all 100 indices appear exactly once.
        let unique: HashSet<usize> = all_indices.into_iter().collect();
        assert_eq!(unique.len(), 100);
    }

    #[test]
    fn test_batch_sampler_drop_last() {
        let config = BatchSamplerConfig {
            batch_size: 10,
            shuffle: false,
            seed: None,
            drop_last: true,
            stratified: false,
        };
        let mut sampler = BatchSampler::new(105, config);
        assert_eq!(sampler.n_batches(), 10, "drop_last should yield 10 batches");

        let mut count = 0usize;
        while let Some(_batch) = sampler.next_batch() {
            count += 1;
        }
        assert_eq!(
            count, 10,
            "should get exactly 10 batches with drop_last=true"
        );
    }

    #[test]
    fn test_batch_sampler_no_drop_last() {
        // 105 samples / 10 batch_size = 10 full + 1 partial = 11 total
        let config = BatchSamplerConfig {
            batch_size: 10,
            shuffle: false,
            seed: None,
            drop_last: false,
            stratified: false,
        };
        let mut sampler = BatchSampler::new(105, config);
        assert_eq!(sampler.n_batches(), 11);

        let mut count = 0usize;
        let mut total_items = 0usize;
        while let Some(batch) = sampler.next_batch() {
            total_items += batch.len();
            count += 1;
        }
        assert_eq!(count, 11);
        assert_eq!(total_items, 105);
    }

    #[test]
    fn test_batch_sampler_shuffle() {
        let config = BatchSamplerConfig {
            batch_size: 10,
            shuffle: true,
            seed: Some(12345),
            drop_last: false,
            stratified: false,
        };
        let mut sampler = BatchSampler::new(50, config);

        // Collect epoch 0 ordering.
        let mut epoch0: Vec<usize> = Vec::new();
        while let Some(batch) = sampler.next_batch() {
            epoch0.extend_from_slice(&batch);
        }

        // Reset and collect epoch 1 ordering.
        sampler.reset();
        let mut epoch1: Vec<usize> = Vec::new();
        while let Some(batch) = sampler.next_batch() {
            epoch1.extend_from_slice(&batch);
        }

        // Both orderings must contain exactly all 50 indices.
        let set0: HashSet<usize> = epoch0.iter().copied().collect();
        let set1: HashSet<usize> = epoch1.iter().copied().collect();
        assert_eq!(set0.len(), 50);
        assert_eq!(set1.len(), 50);

        // The orderings should differ (they could theoretically be equal by
        // chance but that's astronomically unlikely for 50 elements).
        assert_ne!(epoch0, epoch1, "two shuffled epochs should differ");
    }

    #[test]
    fn test_batch_sampler_epoch_counter() {
        let config = BatchSamplerConfig {
            batch_size: 5,
            shuffle: false,
            seed: None,
            drop_last: false,
            stratified: false,
        };
        let mut sampler = BatchSampler::new(20, config);
        assert_eq!(sampler.epoch(), 0);
        // Drain epoch 0.
        while sampler.next_batch().is_some() {}
        sampler.reset();
        assert_eq!(sampler.epoch(), 1);
        while sampler.next_batch().is_some() {}
        sampler.reset();
        assert_eq!(sampler.epoch(), 2);
    }

    #[test]
    fn test_batch_sampler_empty() {
        let config = BatchSamplerConfig {
            batch_size: 10,
            shuffle: false,
            seed: None,
            drop_last: false,
            stratified: false,
        };
        let mut sampler = BatchSampler::new(0, config);
        assert_eq!(sampler.n_batches(), 0);
        assert!(sampler.next_batch().is_none());
    }

    #[test]
    fn test_stratified_sampler_balance() {
        // 3 classes, 30 samples each = 90 total.
        let mut labels = Vec::new();
        for cls in 0..3usize {
            for _ in 0..30 {
                labels.push(cls);
            }
        }
        let config = BatchSamplerConfig {
            batch_size: 9,
            shuffle: false,
            seed: Some(99),
            drop_last: false,
            stratified: true,
        };
        let mut sampler = BatchSampler::from_labels(&labels, config);

        // Each batch of 9 should see 3 examples from each class.
        while let Some(batch) = sampler.next_batch() {
            if batch.len() < 9 {
                continue; // skip partial last batch
            }
            let mut counts = [0usize; 3];
            for &idx in &batch {
                counts[labels[idx]] += 1;
            }
            // With round-robin interleaving each class appears batch_size/3 times.
            assert_eq!(counts[0], 3, "class 0 count in batch");
            assert_eq!(counts[1], 3, "class 1 count in batch");
            assert_eq!(counts[2], 3, "class 2 count in batch");
        }
    }

    #[test]
    fn test_train_val_test_split() {
        let n = 1000;
        let (train, val, test) = train_val_test_split(n, None, 0.7, 0.15, Some(42));

        // No overlap.
        let train_set: HashSet<usize> = train.iter().copied().collect();
        let val_set: HashSet<usize> = val.iter().copied().collect();
        let test_set: HashSet<usize> = test.iter().copied().collect();

        assert!(
            train_set.is_disjoint(&val_set),
            "train and val should not overlap"
        );
        assert!(
            train_set.is_disjoint(&test_set),
            "train and test should not overlap"
        );
        assert!(
            val_set.is_disjoint(&test_set),
            "val and test should not overlap"
        );

        // All indices covered.
        let total = train.len() + val.len() + test.len();
        assert_eq!(total, n, "all samples must be assigned to a split");

        // Rough size checks (within 5% of requested fractions).
        let train_frac = train.len() as f64 / n as f64;
        let val_frac = val.len() as f64 / n as f64;
        assert!(
            (train_frac - 0.7).abs() < 0.05,
            "train fraction {train_frac:.3} too far from 0.7"
        );
        assert!(
            (val_frac - 0.15).abs() < 0.05,
            "val fraction {val_frac:.3} too far from 0.15"
        );
    }

    #[test]
    fn test_train_val_test_split_stratified() {
        let n = 600;
        // 3 balanced classes of 200 each.
        let labels: Vec<usize> = (0..n).map(|i| i % 3).collect();
        let (train, val, test) = train_val_test_split(n, Some(&labels), 0.6, 0.2, Some(7));

        // No overlap.
        let train_set: HashSet<usize> = train.iter().copied().collect();
        let val_set: HashSet<usize> = val.iter().copied().collect();
        let test_set: HashSet<usize> = test.iter().copied().collect();

        assert!(train_set.is_disjoint(&val_set));
        assert!(train_set.is_disjoint(&test_set));
        assert!(val_set.is_disjoint(&test_set));

        let total = train.len() + val.len() + test.len();
        assert_eq!(total, n);

        // Each split should be roughly balanced across classes.
        for split in [&train, &val, &test] {
            let mut counts = [0usize; 3];
            for &idx in split {
                counts[labels[idx]] += 1;
            }
            let min_count = *counts.iter().min().expect("non-empty split");
            let max_count = *counts.iter().max().expect("non-empty split");
            // Balanced if ratio >= 0.8 (i.e. not wildly skewed).
            if split.len() >= 10 {
                let balance = min_count as f64 / max_count as f64;
                assert!(
                    balance >= 0.8,
                    "split imbalanced: counts {counts:?}, balance {balance:.2}"
                );
            }
        }
    }

    #[test]
    fn test_xorshift64_coverage() {
        // Verify PRNG produces distinct values and the modulo helper is sound.
        let mut rng = Xorshift64::new(0xfedcba9876543210);
        let vals: HashSet<u64> = (0..1000).map(|_| rng.next_u64()).collect();
        // With a quality PRNG, 1000 draws out of 2^64 should all be distinct.
        assert_eq!(vals.len(), 1000, "PRNG produced duplicate values");

        // next_usize must be in [0, n).
        for n in [1usize, 2, 5, 100] {
            for _ in 0..200 {
                let v = rng.next_usize(n);
                assert!(v < n, "next_usize({n}) returned {v}");
            }
        }
    }
}
