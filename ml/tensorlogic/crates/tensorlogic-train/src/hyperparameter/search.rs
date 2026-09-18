//! Grid and random search strategies for hyperparameter optimization.

use scirs2_core::random::{SeedableRng, StdRng};
use std::collections::HashMap;

use super::space::HyperparamSpace;
use super::value::{HyperparamConfig, HyperparamResult, HyperparamValue};

/// Grid search strategy for hyperparameter optimization.
///
/// Exhaustively searches over a grid of hyperparameter values.
#[derive(Debug)]
pub struct GridSearch {
    /// Parameter space definition.
    param_space: HashMap<String, HyperparamSpace>,
    /// Number of grid points per continuous parameter.
    num_grid_points: usize,
    /// Results from all evaluations.
    results: Vec<HyperparamResult>,
}

impl GridSearch {
    /// Create a new grid search.
    ///
    /// # Arguments
    /// * `param_space` - Hyperparameter space definition
    /// * `num_grid_points` - Number of points for continuous parameters
    pub fn new(param_space: HashMap<String, HyperparamSpace>, num_grid_points: usize) -> Self {
        Self {
            param_space,
            num_grid_points,
            results: Vec::new(),
        }
    }

    /// Generate all parameter configurations for grid search.
    pub fn generate_configs(&self) -> Vec<HyperparamConfig> {
        if self.param_space.is_empty() {
            return vec![HashMap::new()];
        }
        let mut param_names: Vec<String> = self.param_space.keys().cloned().collect();
        param_names.sort();
        let mut all_values: Vec<Vec<HyperparamValue>> = Vec::new();
        for name in &param_names {
            let space = &self.param_space[name];
            all_values.push(space.grid_values(self.num_grid_points));
        }
        let mut configs = Vec::new();
        self.generate_cartesian_product(
            &param_names,
            &all_values,
            0,
            &mut HashMap::new(),
            &mut configs,
        );
        configs
    }

    /// Recursively generate Cartesian product of parameter values.
    #[allow(clippy::only_used_in_recursion)]
    fn generate_cartesian_product(
        &self,
        param_names: &[String],
        all_values: &[Vec<HyperparamValue>],
        depth: usize,
        current_config: &mut HyperparamConfig,
        configs: &mut Vec<HyperparamConfig>,
    ) {
        if depth == param_names.len() {
            configs.push(current_config.clone());
            return;
        }
        let param_name = &param_names[depth];
        let values = &all_values[depth];
        for value in values {
            current_config.insert(param_name.clone(), value.clone());
            self.generate_cartesian_product(
                param_names,
                all_values,
                depth + 1,
                current_config,
                configs,
            );
        }
        current_config.remove(param_name);
    }

    /// Add a result from evaluating a configuration.
    pub fn add_result(&mut self, result: HyperparamResult) {
        self.results.push(result);
    }

    /// Get the best result found so far.
    pub fn best_result(&self) -> Option<&HyperparamResult> {
        self.results.iter().max_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Get all results sorted by score (descending).
    pub fn sorted_results(&self) -> Vec<&HyperparamResult> {
        let mut results: Vec<&HyperparamResult> = self.results.iter().collect();
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    /// Get all results.
    pub fn results(&self) -> &[HyperparamResult] {
        &self.results
    }

    /// Get total number of configurations to evaluate.
    pub fn total_configs(&self) -> usize {
        self.generate_configs().len()
    }
}

/// Random search strategy for hyperparameter optimization.
///
/// Randomly samples from the hyperparameter space.
#[derive(Debug)]
pub struct RandomSearch {
    /// Parameter space definition.
    param_space: HashMap<String, HyperparamSpace>,
    /// Number of random samples to evaluate.
    num_samples: usize,
    /// Random number generator.
    rng: StdRng,
    /// Results from all evaluations.
    results: Vec<HyperparamResult>,
}

impl RandomSearch {
    /// Create a new random search.
    ///
    /// # Arguments
    /// * `param_space` - Hyperparameter space definition
    /// * `num_samples` - Number of random configurations to try
    /// * `seed` - Random seed for reproducibility
    pub fn new(
        param_space: HashMap<String, HyperparamSpace>,
        num_samples: usize,
        seed: u64,
    ) -> Self {
        Self {
            param_space,
            num_samples,
            rng: StdRng::seed_from_u64(seed),
            results: Vec::new(),
        }
    }

    /// Generate random parameter configurations.
    pub fn generate_configs(&mut self) -> Vec<HyperparamConfig> {
        let mut configs = Vec::with_capacity(self.num_samples);
        for _ in 0..self.num_samples {
            let mut config = HashMap::new();
            for (name, space) in &self.param_space {
                let value = space.sample(&mut self.rng);
                config.insert(name.clone(), value);
            }
            configs.push(config);
        }
        configs
    }

    /// Add a result from evaluating a configuration.
    pub fn add_result(&mut self, result: HyperparamResult) {
        self.results.push(result);
    }

    /// Get the best result found so far.
    pub fn best_result(&self) -> Option<&HyperparamResult> {
        self.results.iter().max_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Get all results sorted by score (descending).
    pub fn sorted_results(&self) -> Vec<&HyperparamResult> {
        let mut results: Vec<&HyperparamResult> = self.results.iter().collect();
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    /// Get all results.
    pub fn results(&self) -> &[HyperparamResult] {
        &self.results
    }
}
