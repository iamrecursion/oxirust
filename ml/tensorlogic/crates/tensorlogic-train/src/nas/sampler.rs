//! Random architecture sampler and mutation operators for NAS.
//!
//! [`ArchSampler`] draws uniformly random architectures from an [`ArchSearchSpace`]
//! and provides a neighbourhood mutation used by evolutionary search.

use scirs2_core::random::{SeedableRng, StdRng};

use crate::error::{TrainError, TrainResult};

use super::space::{ArchSearchSpace, Architecture, LayerSpec};

// ─── ArchSampler ────────────────────────────────────────────────────────────

/// Samples and mutates architectures within an [`ArchSearchSpace`].
pub struct ArchSampler {
    space: ArchSearchSpace,
    rng: StdRng,
}

impl ArchSampler {
    /// Create a new sampler for the given search space, seeded for reproducibility.
    pub fn new(space: ArchSearchSpace, seed: u64) -> Self {
        Self {
            space,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Sample a uniformly random architecture within the search space.
    ///
    /// Depth is drawn uniformly from `[min_depth, max_depth]`.  Each layer's
    /// op, width, and activation are drawn independently and uniformly from
    /// their respective option lists.
    pub fn random_architecture(&mut self) -> TrainResult<Architecture> {
        let depth_range = self.space.max_depth - self.space.min_depth + 1;
        let depth = self.space.min_depth + self.rng.gen_range(0..depth_range);

        let mut layers = Vec::with_capacity(depth);
        for _ in 0..depth {
            layers.push(self.sample_layer()?);
        }

        Ok(Architecture { layers })
    }

    /// Mutate one aspect of `arch` at random, returning a new valid architecture.
    ///
    /// Four mutations are selected uniformly at random:
    ///
    /// | Index | Mutation |
    /// |-------|----------|
    /// | 0 | Change a random layer's **op** |
    /// | 1 | Change a random layer's **width** |
    /// | 2 | Change a random layer's **activation** |
    /// | 3 | **Add** a layer (if depth < max_depth) or **remove** one (if depth > min_depth); if neither is possible, fall back to changing op |
    pub fn mutate(&mut self, arch: &Architecture) -> TrainResult<Architecture> {
        let mut new_arch = arch.clone();
        let mutation_type = self.rng.gen_range(0..4_usize);

        match mutation_type {
            0 => {
                // Change a random layer's op
                let layer_idx = self.pick_layer_index(&new_arch)?;
                let new_op = self.pick_option(&self.space.op_options.clone())?;
                new_arch.layers[layer_idx].op = new_op;
            }
            1 => {
                // Change a random layer's width
                let layer_idx = self.pick_layer_index(&new_arch)?;
                let new_width = self.pick_width()?;
                new_arch.layers[layer_idx].width = new_width;
            }
            2 => {
                // Change a random layer's activation
                let layer_idx = self.pick_layer_index(&new_arch)?;
                let new_act = self.pick_option(&self.space.activation_options.clone())?;
                new_arch.layers[layer_idx].activation = new_act;
            }
            3 => {
                // Add or remove a layer
                let can_add = new_arch.depth() < self.space.max_depth;
                let can_remove = new_arch.depth() > self.space.min_depth;

                if can_add && can_remove {
                    // Choose randomly
                    if self.rng.gen_range(0..2_usize) == 0 {
                        self.add_random_layer(&mut new_arch)?;
                    } else {
                        self.remove_random_layer(&mut new_arch)?;
                    }
                } else if can_add {
                    self.add_random_layer(&mut new_arch)?;
                } else if can_remove {
                    self.remove_random_layer(&mut new_arch)?;
                } else {
                    // Neither add nor remove possible — fall back to op change
                    let layer_idx = self.pick_layer_index(&new_arch)?;
                    let new_op = self.pick_option(&self.space.op_options.clone())?;
                    new_arch.layers[layer_idx].op = new_op;
                }
            }
            _ => unreachable!("gen_range(0..4) is always in 0..3"),
        }

        Ok(new_arch)
    }

    // ── private helpers ──────────────────────────────────────────────────

    /// Sample a single fresh layer from the search space.
    fn sample_layer(&mut self) -> TrainResult<LayerSpec> {
        let op = self.pick_option(&self.space.op_options.clone())?;
        let width = self.pick_width()?;
        let activation = self.pick_option(&self.space.activation_options.clone())?;
        Ok(LayerSpec {
            op,
            width,
            activation,
        })
    }

    /// Return a random element from `options` (assumed non-empty by space invariants).
    fn pick_option(&mut self, options: &[String]) -> TrainResult<String> {
        if options.is_empty() {
            return Err(TrainError::InvalidParameter(
                "option list must be non-empty".to_string(),
            ));
        }
        let idx = self.rng.gen_range(0..options.len());
        Ok(options[idx].clone())
    }

    /// Return a random width from `width_options`.
    fn pick_width(&mut self) -> TrainResult<usize> {
        if self.space.width_options.is_empty() {
            return Err(TrainError::InvalidParameter(
                "width_options must be non-empty".to_string(),
            ));
        }
        let idx = self.rng.gen_range(0..self.space.width_options.len());
        Ok(self.space.width_options[idx])
    }

    /// Return a valid random index into `arch.layers`.
    fn pick_layer_index(&mut self, arch: &Architecture) -> TrainResult<usize> {
        if arch.layers.is_empty() {
            return Err(TrainError::InvalidParameter(
                "architecture has no layers to mutate".to_string(),
            ));
        }
        Ok(self.rng.gen_range(0..arch.layers.len()))
    }

    /// Insert a new random layer at a random position.
    fn add_random_layer(&mut self, arch: &mut Architecture) -> TrainResult<()> {
        let new_layer = self.sample_layer()?;
        // Insert at a random position in [0, depth]
        let pos = self.rng.gen_range(0..=arch.layers.len());
        arch.layers.insert(pos, new_layer);
        Ok(())
    }

    /// Remove the layer at a random position.
    fn remove_random_layer(&mut self, arch: &mut Architecture) -> TrainResult<()> {
        let idx = self.pick_layer_index(arch)?;
        arch.layers.remove(idx);
        Ok(())
    }

    /// Generate a uniformly random `usize` in `[lower, upper)`.
    ///
    /// Used by [`super::RegularizedEvolution`] to share the sampler's RNG for
    /// tournament index shuffling.  Returns `lower` when `upper <= lower`.
    pub fn gen_range_usize(&mut self, lower: usize, upper: usize) -> usize {
        if upper <= lower {
            return lower;
        }
        lower + self.rng.gen_range(0..(upper - lower))
    }
}
