//! Mix-minus routing for broadcast IFB (Interruptible Foldback) feeds.
//!
//! In broadcast, each talent receives a mix of all sources *except* their own
//! microphone to avoid feedback. This module provides [`MixMinusBus`] and
//! [`MixMinusRouter`] for managing multiple IFB feeds from a shared source pool.

use std::collections::{HashMap, HashSet};

use super::crosspoint::MatrixError;

// ---------------------------------------------------------------------------
// MixMinusBus — one IFB feed for a single talent / destination
// ---------------------------------------------------------------------------

/// A single mix-minus bus that sums all connected inputs except the excluded
/// one(s), each with an independent gain coefficient.
#[derive(Debug, Clone)]
pub struct MixMinusBus {
    /// Human-readable name of this bus (e.g. "Anchor IFB").
    pub name: String,
    /// Index of the output (destination) this bus feeds.
    pub output_index: usize,
    /// Set of input indices excluded from this bus (typically the talent's own mic).
    excluded_inputs: HashSet<usize>,
    /// Per-input gain overrides in dB. Inputs not listed use 0 dB.
    input_gains: HashMap<usize, f32>,
    /// Master gain applied after summing, in dB.
    pub master_gain_db: f32,
    /// Whether the bus is active.
    pub active: bool,
}

impl MixMinusBus {
    /// Creates a new mix-minus bus for the given output, excluding one input.
    pub fn new(name: impl Into<String>, output_index: usize, excluded_input: usize) -> Self {
        let mut excluded = HashSet::new();
        excluded.insert(excluded_input);
        Self {
            name: name.into(),
            output_index,
            excluded_inputs: excluded,
            input_gains: HashMap::new(),
            master_gain_db: 0.0,
            active: true,
        }
    }

    /// Adds another input to the exclusion set.
    pub fn add_exclusion(&mut self, input_index: usize) {
        self.excluded_inputs.insert(input_index);
    }

    /// Removes an input from the exclusion set.
    pub fn remove_exclusion(&mut self, input_index: usize) {
        self.excluded_inputs.remove(&input_index);
    }

    /// Returns `true` if the given input is excluded from this bus.
    pub fn is_excluded(&self, input_index: usize) -> bool {
        self.excluded_inputs.contains(&input_index)
    }

    /// Returns the set of excluded input indices.
    pub fn excluded_inputs(&self) -> &HashSet<usize> {
        &self.excluded_inputs
    }

    /// Sets a per-input gain override in dB.
    pub fn set_input_gain(&mut self, input_index: usize, gain_db: f32) {
        self.input_gains.insert(input_index, gain_db);
    }

    /// Returns the gain (in dB) for a specific input, defaulting to 0.0.
    pub fn input_gain(&self, input_index: usize) -> f32 {
        self.input_gains.get(&input_index).copied().unwrap_or(0.0)
    }

    /// Converts a dB value to a linear gain coefficient.
    fn db_to_linear(db: f32) -> f32 {
        10.0_f32.powf(db / 20.0)
    }

    /// Computes the mix-minus output for this bus given a slice of input samples.
    ///
    /// Each input sample is multiplied by its per-input linear gain and included
    /// in the sum unless excluded. The result is scaled by `master_gain_db`.
    ///
    /// Returns `0.0` if the bus is inactive.
    pub fn compute(&self, input_samples: &[f32]) -> f32 {
        if !self.active {
            return 0.0;
        }

        let mut sum = 0.0_f32;
        for (idx, &sample) in input_samples.iter().enumerate() {
            if self.excluded_inputs.contains(&idx) {
                continue;
            }
            let gain_linear = Self::db_to_linear(self.input_gain(idx));
            sum += sample * gain_linear;
        }

        sum * Self::db_to_linear(self.master_gain_db)
    }

    /// Returns the number of excluded inputs.
    pub fn exclusion_count(&self) -> usize {
        self.excluded_inputs.len()
    }
}

// ---------------------------------------------------------------------------
// MixMinusRouter — manages a collection of mix-minus buses
// ---------------------------------------------------------------------------

/// Manages multiple [`MixMinusBus`] instances for an entire broadcast routing
/// scenario. Each bus is identified by its output index.
#[derive(Debug, Clone, Default)]
pub struct MixMinusRouter {
    /// Map from output index to mix-minus bus.
    buses: HashMap<usize, MixMinusBus>,
    /// Total number of inputs available to the router.
    input_count: usize,
}

impl MixMinusRouter {
    /// Creates a new router with the given input count.
    pub fn new(input_count: usize) -> Self {
        Self {
            buses: HashMap::new(),
            input_count,
        }
    }

    /// Adds a mix-minus bus for the given output, excluding one input.
    ///
    /// Returns an error if the output or excluded input index is out of range.
    pub fn add_bus(
        &mut self,
        name: impl Into<String>,
        output_index: usize,
        excluded_input: usize,
    ) -> Result<(), MatrixError> {
        if excluded_input >= self.input_count {
            return Err(MatrixError::InvalidInput(excluded_input));
        }
        let bus = MixMinusBus::new(name, output_index, excluded_input);
        self.buses.insert(output_index, bus);
        Ok(())
    }

    /// Returns a reference to the bus for the given output.
    pub fn get_bus(&self, output_index: usize) -> Option<&MixMinusBus> {
        self.buses.get(&output_index)
    }

    /// Returns a mutable reference to the bus for the given output.
    pub fn get_bus_mut(&mut self, output_index: usize) -> Option<&mut MixMinusBus> {
        self.buses.get_mut(&output_index)
    }

    /// Removes the bus for the given output.
    pub fn remove_bus(&mut self, output_index: usize) -> Option<MixMinusBus> {
        self.buses.remove(&output_index)
    }

    /// Returns the number of buses.
    pub fn bus_count(&self) -> usize {
        self.buses.len()
    }

    /// Returns the configured input count.
    pub fn input_count(&self) -> usize {
        self.input_count
    }

    /// Computes all mix-minus outputs from a shared set of input samples.
    ///
    /// Returns a map from output index to computed sample value.
    pub fn compute_all(&self, input_samples: &[f32]) -> HashMap<usize, f32> {
        self.buses
            .iter()
            .map(|(&out_idx, bus)| (out_idx, bus.compute(input_samples)))
            .collect()
    }

    /// Creates a standard IFB configuration where each of `n` talents gets a
    /// mix of all inputs except their own. Input `i` is excluded from output `i`.
    pub fn create_standard_ifb(&mut self, talent_count: usize) -> Result<(), MatrixError> {
        if talent_count > self.input_count {
            return Err(MatrixError::InvalidInput(talent_count));
        }
        for i in 0..talent_count {
            let name = format!("IFB {}", i + 1);
            self.add_bus(name, i, i)?;
        }
        Ok(())
    }

    /// Returns a list of all output indices that have a bus.
    pub fn active_outputs(&self) -> Vec<usize> {
        let mut outputs: Vec<usize> = self.buses.keys().copied().collect();
        outputs.sort();
        outputs
    }

    /// Returns `true` if the given output has a mix-minus bus.
    pub fn has_bus(&self, output_index: usize) -> bool {
        self.buses.contains_key(&output_index)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mix_minus_bus_basic() {
        // 4 inputs, output 0 excludes input 0
        let bus = MixMinusBus::new("Anchor IFB", 0, 0);
        assert!(bus.is_excluded(0));
        assert!(!bus.is_excluded(1));
        assert_eq!(bus.exclusion_count(), 1);
    }

    #[test]
    fn test_mix_minus_bus_compute_excludes_self() {
        let bus = MixMinusBus::new("Test", 0, 0);
        // Input 0 = 1.0, Input 1 = 0.5, Input 2 = 0.3
        let samples = [1.0_f32, 0.5, 0.3];
        let result = bus.compute(&samples);
        // Should sum inputs 1 + 2 = 0.5 + 0.3 = 0.8 (all gains 0 dB = linear 1.0)
        assert!((result - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_mix_minus_bus_compute_inactive() {
        let mut bus = MixMinusBus::new("Test", 0, 0);
        bus.active = false;
        let samples = [1.0_f32, 0.5];
        assert!((bus.compute(&samples)).abs() < 1e-10);
    }

    #[test]
    fn test_mix_minus_bus_multiple_exclusions() {
        let mut bus = MixMinusBus::new("Test", 0, 0);
        bus.add_exclusion(2);
        let samples = [1.0_f32, 0.5, 0.3, 0.2];
        let result = bus.compute(&samples);
        // Should sum inputs 1 + 3 = 0.5 + 0.2 = 0.7
        assert!((result - 0.7).abs() < 1e-5);
    }

    #[test]
    fn test_mix_minus_bus_with_input_gain() {
        let mut bus = MixMinusBus::new("Test", 0, 0);
        // Boost input 1 by +6 dB (~2x)
        bus.set_input_gain(1, 6.0);
        let samples = [1.0_f32, 0.5];
        let result = bus.compute(&samples);
        // Input 1 at 0.5 * ~1.995 ≈ 0.9976
        let expected = 0.5 * 10.0_f32.powf(6.0 / 20.0);
        assert!((result - expected).abs() < 1e-3);
    }

    #[test]
    fn test_mix_minus_bus_master_gain() {
        let mut bus = MixMinusBus::new("Test", 0, 0);
        bus.master_gain_db = -6.0;
        let samples = [1.0_f32, 1.0];
        let result = bus.compute(&samples);
        // Sum of input 1 = 1.0, then * 10^(-6/20) ≈ 0.5012
        let expected = 1.0 * 10.0_f32.powf(-6.0 / 20.0);
        assert!((result - expected).abs() < 1e-3);
    }

    #[test]
    fn test_mix_minus_bus_remove_exclusion() {
        let mut bus = MixMinusBus::new("Test", 0, 0);
        bus.add_exclusion(1);
        assert!(bus.is_excluded(1));
        bus.remove_exclusion(1);
        assert!(!bus.is_excluded(1));
    }

    #[test]
    fn test_mix_minus_bus_input_gain_default() {
        let bus = MixMinusBus::new("Test", 0, 0);
        assert!((bus.input_gain(5) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_router_creation() {
        let router = MixMinusRouter::new(8);
        assert_eq!(router.input_count(), 8);
        assert_eq!(router.bus_count(), 0);
    }

    #[test]
    fn test_router_add_bus() {
        let mut router = MixMinusRouter::new(4);
        router.add_bus("IFB 1", 0, 0).expect("valid indices");
        assert_eq!(router.bus_count(), 1);
        assert!(router.has_bus(0));
    }

    #[test]
    fn test_router_add_bus_invalid_input() {
        let mut router = MixMinusRouter::new(4);
        let result = router.add_bus("Bad", 0, 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_router_remove_bus() {
        let mut router = MixMinusRouter::new(4);
        router.add_bus("IFB 1", 0, 0).expect("valid");
        let removed = router.remove_bus(0);
        assert!(removed.is_some());
        assert_eq!(router.bus_count(), 0);
    }

    #[test]
    fn test_router_compute_all() {
        let mut router = MixMinusRouter::new(3);
        // Output 0 excludes input 0, Output 1 excludes input 1
        router.add_bus("IFB 0", 0, 0).expect("valid");
        router.add_bus("IFB 1", 1, 1).expect("valid");

        let samples = [1.0_f32, 0.5, 0.3];
        let results = router.compute_all(&samples);

        // Output 0 = 0.5 + 0.3 = 0.8
        assert!((results[&0] - 0.8).abs() < 1e-5);
        // Output 1 = 1.0 + 0.3 = 1.3
        assert!((results[&1] - 1.3).abs() < 1e-5);
    }

    #[test]
    fn test_router_standard_ifb() {
        let mut router = MixMinusRouter::new(4);
        router.create_standard_ifb(3).expect("valid");
        assert_eq!(router.bus_count(), 3);

        // Each bus excludes its own index
        for i in 0..3 {
            let bus = router.get_bus(i).expect("bus exists");
            assert!(bus.is_excluded(i));
        }
    }

    #[test]
    fn test_router_standard_ifb_exceeds_inputs() {
        let mut router = MixMinusRouter::new(2);
        let result = router.create_standard_ifb(5);
        assert!(result.is_err());
    }

    #[test]
    fn test_router_active_outputs() {
        let mut router = MixMinusRouter::new(4);
        router.add_bus("A", 2, 0).expect("valid");
        router.add_bus("B", 0, 1).expect("valid");
        let outputs = router.active_outputs();
        assert_eq!(outputs, vec![0, 2]);
    }

    #[test]
    fn test_router_get_bus_mut() {
        let mut router = MixMinusRouter::new(4);
        router.add_bus("IFB", 0, 0).expect("valid");
        if let Some(bus) = router.get_bus_mut(0) {
            bus.master_gain_db = -3.0;
        }
        let bus = router.get_bus(0).expect("bus exists");
        assert!((bus.master_gain_db - (-3.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_mix_minus_full_broadcast_scenario() {
        // 4 talent mics, each gets IFB of everyone else
        let mut router = MixMinusRouter::new(4);
        router.create_standard_ifb(4).expect("valid");

        let samples = [0.8_f32, 0.6, 0.4, 0.2];
        let results = router.compute_all(&samples);

        // Talent 0 hears: 0.6 + 0.4 + 0.2 = 1.2
        assert!((results[&0] - 1.2).abs() < 1e-5);
        // Talent 1 hears: 0.8 + 0.4 + 0.2 = 1.4
        assert!((results[&1] - 1.4).abs() < 1e-5);
        // Talent 2 hears: 0.8 + 0.6 + 0.2 = 1.6
        assert!((results[&2] - 1.6).abs() < 1e-5);
        // Talent 3 hears: 0.8 + 0.6 + 0.4 = 1.8
        assert!((results[&3] - 1.8).abs() < 1e-5);
    }

    #[test]
    fn test_mix_minus_empty_samples() {
        let bus = MixMinusBus::new("Test", 0, 0);
        let result = bus.compute(&[]);
        assert!((result).abs() < 1e-10);
    }

    #[test]
    fn test_mix_minus_all_excluded() {
        let mut bus = MixMinusBus::new("Test", 0, 0);
        bus.add_exclusion(1);
        let samples = [1.0_f32, 0.5];
        let result = bus.compute(&samples);
        assert!((result).abs() < 1e-10);
    }

    #[test]
    fn test_router_has_bus_false() {
        let router = MixMinusRouter::new(4);
        assert!(!router.has_bus(0));
    }

    #[test]
    fn test_router_get_bus_none() {
        let router = MixMinusRouter::new(4);
        assert!(router.get_bus(99).is_none());
    }
}
