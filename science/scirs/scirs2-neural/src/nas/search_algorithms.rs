//! Search algorithms for Neural Architecture Search

use crate::error::{NeuralError, Result};
use crate::nas::architecture_encoding::ArchitectureEncoding;
use crate::nas::SearchResult;
use std::sync::Arc;

/// Trait for search algorithms
pub trait SearchAlgorithm: Send + Sync {
    /// Propose architectures to evaluate
    fn propose_architectures(
        &self,
        history: &[SearchResult],
        n_proposals: usize,
    ) -> Result<Vec<Arc<dyn ArchitectureEncoding>>>;
    /// Update the algorithm with new results
    fn update(&mut self, results: &[SearchResult]) -> Result<()>;
    /// Get algorithm name
    fn name(&self) -> &str;
}

/// Random search algorithm
pub struct RandomSearch {
    seed: Option<u64>,
}

impl Default for RandomSearch {
    fn default() -> Self {
        Self::new()
    }
}

impl RandomSearch {
    /// Create a new random search algorithm
    pub fn new() -> Self {
        Self { seed: None }
    }

    /// Create with a specific seed
    pub fn with_seed(seed: u64) -> Self {
        Self { seed: Some(seed) }
    }
}

impl SearchAlgorithm for RandomSearch {
    fn propose_architectures(
        &self,
        _history: &[SearchResult],
        n_proposals: usize,
    ) -> Result<Vec<Arc<dyn ArchitectureEncoding>>> {
        use scirs2_core::random::prelude::*;
        let mut rng_inst = if let Some(seed) = self.seed {
            StdRng::seed_from_u64(seed)
        } else {
            let random_seed = scirs2_core::random::random::<u64>();
            StdRng::seed_from_u64(random_seed)
        };
        let mut proposals = Vec::with_capacity(n_proposals);
        for _ in 0..n_proposals {
            let encoding = crate::nas::architecture_encoding::GraphEncoding::random(&mut rng_inst)?;
            proposals.push(Arc::new(encoding) as Arc<dyn ArchitectureEncoding>);
        }
        Ok(proposals)
    }

    fn update(&mut self, _results: &[SearchResult]) -> Result<()> {
        Ok(())
    }

    fn name(&self) -> &str {
        "RandomSearch"
    }
}

/// Evolutionary search algorithm
pub struct EvolutionarySearch {
    population_size: usize,
    mutation_rate: f32,
    crossover_rate: f32,
    tournament_size: usize,
    elite_size: usize,
    population: Vec<Arc<dyn ArchitectureEncoding>>,
    fitness_scores: Vec<f64>,
}

impl EvolutionarySearch {
    /// Create a new evolutionary search algorithm
    pub fn new(population_size: usize) -> Self {
        Self {
            population_size,
            mutation_rate: 0.1,
            crossover_rate: 0.9,
            tournament_size: 3,
            elite_size: (population_size / 10).max(1),
            population: Vec::new(),
            fitness_scores: Vec::new(),
        }
    }

    /// Set mutation rate
    pub fn with_mutation_rate(mut self, rate: f32) -> Self {
        self.mutation_rate = rate;
        self
    }

    /// Set crossover rate
    pub fn with_crossover_rate(mut self, rate: f32) -> Self {
        self.crossover_rate = rate;
        self
    }

    /// Tournament selection — returns index of the best individual in a random tournament
    fn tournament_select(&self, rng: &mut impl scirs2_core::random::RngExt) -> usize {
        if self.population.is_empty() {
            return 0;
        }
        let mut best_idx = rng.random_range(0..self.population.len());
        let mut best_fitness = self.fitness_scores[best_idx];
        for _ in 1..self.tournament_size {
            let idx = rng.random_range(0..self.population.len());
            if self.fitness_scores[idx] > best_fitness {
                best_idx = idx;
                best_fitness = self.fitness_scores[idx];
            }
        }
        best_idx
    }
}

impl SearchAlgorithm for EvolutionarySearch {
    fn propose_architectures(
        &self,
        _history: &[SearchResult],
        n_proposals: usize,
    ) -> Result<Vec<Arc<dyn ArchitectureEncoding>>> {
        use scirs2_core::random::prelude::*;
        let mut rng_inst = thread_rng();
        if self.population.is_empty() {
            let mut proposals = Vec::with_capacity(n_proposals);
            for _ in 0..n_proposals {
                let encoding =
                    crate::nas::architecture_encoding::GraphEncoding::random(&mut rng_inst)?;
                proposals.push(Arc::new(encoding) as Arc<dyn ArchitectureEncoding>);
            }
            return Ok(proposals);
        }
        let mut proposals: Vec<Arc<dyn ArchitectureEncoding>> = Vec::with_capacity(n_proposals);
        let mut elite_indices: Vec<usize> = (0..self.population.len()).collect();
        elite_indices.sort_by(|&a, &b| {
            self.fitness_scores[b]
                .partial_cmp(&self.fitness_scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for &idx in elite_indices.iter().take(self.elite_size.min(n_proposals)) {
            proposals.push(self.population[idx].clone());
        }
        while proposals.len() < n_proposals {
            if rng_inst.random::<f32>() < self.crossover_rate && self.population.len() >= 2 {
                let parent1_idx = self.tournament_select(&mut rng_inst);
                let parent2_idx = self.tournament_select(&mut rng_inst);
                if parent1_idx != parent2_idx {
                    let offspring = self.population[parent1_idx]
                        .crossover(self.population[parent2_idx].as_ref())?;
                    proposals.push(Arc::from(offspring));
                } else {
                    let parent_idx = self.tournament_select(&mut rng_inst);
                    let offspring = self.population[parent_idx].mutate(self.mutation_rate)?;
                    proposals.push(Arc::from(offspring));
                }
            } else {
                let parent_idx = self.tournament_select(&mut rng_inst);
                let offspring = self.population[parent_idx].mutate(self.mutation_rate)?;
                proposals.push(Arc::from(offspring));
            }
        }
        Ok(proposals)
    }

    fn update(&mut self, results: &[SearchResult]) -> Result<()> {
        for result in results {
            self.population.push(result.architecture.clone());
            let fitness = if result.metrics.is_empty() {
                0.0
            } else {
                result.metrics.values().sum::<f64>() / result.metrics.len() as f64
            };
            self.fitness_scores.push(fitness);
        }
        if self.population.len() > self.population_size {
            let mut indices: Vec<usize> = (0..self.population.len()).collect();
            indices.sort_by(|&a, &b| {
                self.fitness_scores[b]
                    .partial_cmp(&self.fitness_scores[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let new_population: Vec<_> = indices
                .iter()
                .take(self.population_size)
                .map(|&idx| self.population[idx].clone())
                .collect();
            let new_scores: Vec<_> = indices
                .iter()
                .take(self.population_size)
                .map(|&idx| self.fitness_scores[idx])
                .collect();
            self.population = new_population;
            self.fitness_scores = new_scores;
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "EvolutionarySearch"
    }
}

/// Internal controller network for reinforcement learning search
struct ControllerNetwork {
    hidden_size: usize,
    embedding_dim: usize,
    vocab_size: usize,
}

impl ControllerNetwork {
    fn new(hidden_size: usize, embedding_dim: usize) -> Self {
        Self {
            hidden_size,
            embedding_dim,
            vocab_size: 50,
        }
    }
}

/// Reinforcement learning based search with REINFORCE controller
pub struct ReinforcementSearch {
    controller_hidden_size: usize,
    learning_rate: f32,
    entropy_weight: f32,
    baseline_decay: f32,
    pub baseline: Option<f64>,
    controller_network: Option<ControllerNetwork>,
}

impl Default for ReinforcementSearch {
    fn default() -> Self {
        Self::new()
    }
}

impl ReinforcementSearch {
    /// Create a new reinforcement learning search
    pub fn new() -> Self {
        Self {
            controller_hidden_size: 100,
            learning_rate: 3.5e-4,
            entropy_weight: 0.01,
            baseline_decay: 0.99,
            baseline: None,
            controller_network: None,
        }
    }

    /// Initialize the controller network
    fn initialize_controller(&mut self) {
        let embedding_dim = 32;
        self.controller_network = Some(ControllerNetwork::new(
            self.controller_hidden_size,
            embedding_dim,
        ));
    }

    /// Generate a random architecture sequence using the controller
    fn generate_architecture_sequence(
        &self,
        rng: &mut scirs2_core::random::prelude::ThreadRng,
    ) -> Vec<usize> {
        let vocab_size = self
            .controller_network
            .as_ref()
            .map(|c| c.vocab_size)
            .unwrap_or(50);
        let length = rng.random_range(5..20);
        (0..length)
            .map(|_| rng.random_range(1..vocab_size))
            .collect()
    }

    /// Convert sequence to architecture encoding
    fn sequence_to_encoding(&self, sequence: &[usize]) -> Result<Arc<dyn ArchitectureEncoding>> {
        use crate::nas::search_space::LayerType;
        let mut layers = Vec::new();
        for &token in sequence {
            let layer_type = match token % 7 {
                0 => continue,
                1 => LayerType::Dense(64 + (token % 4) * 64),
                2 => LayerType::Conv2D {
                    filters: 32 + (token % 4) * 32,
                    kernel_size: (3, 3),
                    stride: (1, 1),
                },
                3 => LayerType::Dropout(0.1 + (token % 4) as f32 * 0.1),
                4 => LayerType::BatchNorm,
                5 => LayerType::Activation("relu".to_string()),
                _ => LayerType::MaxPool2D {
                    pool_size: (2, 2),
                    stride: (2, 2),
                },
            };
            layers.push(layer_type);
            if layers.len() >= 15 {
                break;
            }
        }
        let encoding = crate::nas::architecture_encoding::SequentialEncoding::new(layers);
        Ok(Arc::new(encoding) as Arc<dyn ArchitectureEncoding>)
    }
}

impl SearchAlgorithm for ReinforcementSearch {
    fn propose_architectures(
        &self,
        _history: &[SearchResult],
        n_proposals: usize,
    ) -> Result<Vec<Arc<dyn ArchitectureEncoding>>> {
        use scirs2_core::random::prelude::*;
        let mut rng_inst = thread_rng();
        let mut proposals = Vec::with_capacity(n_proposals);
        for _ in 0..n_proposals {
            let sequence = self.generate_architecture_sequence(&mut rng_inst);
            let encoding = self.sequence_to_encoding(&sequence)?;
            proposals.push(encoding);
        }
        Ok(proposals)
    }

    fn update(&mut self, results: &[SearchResult]) -> Result<()> {
        let rewards: Vec<f64> = results
            .iter()
            .map(|r| {
                if r.metrics.is_empty() {
                    0.0
                } else {
                    r.metrics.values().sum::<f64>() / r.metrics.len() as f64
                }
            })
            .collect();
        if rewards.is_empty() {
            return Ok(());
        }
        let mean_reward = rewards.iter().copied().sum::<f64>() / rewards.len() as f64;
        self.baseline = Some(match self.baseline {
            Some(b) => {
                self.baseline_decay as f64 * b + (1.0 - self.baseline_decay as f64) * mean_reward
            }
            None => mean_reward,
        });
        Ok(())
    }

    fn name(&self) -> &str {
        "ReinforcementSearch"
    }
}

/// Differentiable architecture search (DARTS)
pub struct DifferentiableSearch {
    temperature: f64,
    arch_learning_rate: f32,
    arch_weight_decay: f32,
    alpha_normal: Option<Vec<Vec<f32>>>,
    mixed_ops: Vec<String>,
    num_intermediate_nodes: usize,
    current_epoch: usize,
}

impl Default for DifferentiableSearch {
    fn default() -> Self {
        Self::new()
    }
}

impl DifferentiableSearch {
    /// Create a new differentiable search
    pub fn new() -> Self {
        Self {
            temperature: 1.0,
            arch_learning_rate: 3e-4,
            arch_weight_decay: 1e-3,
            alpha_normal: None,
            mixed_ops: vec![
                "none".to_string(),
                "max_pool_3x3".to_string(),
                "avg_pool_3x3".to_string(),
                "skip_connect".to_string(),
                "sep_conv_3x3".to_string(),
                "sep_conv_5x5".to_string(),
                "dil_conv_3x3".to_string(),
                "dil_conv_5x5".to_string(),
            ],
            num_intermediate_nodes: 4,
            current_epoch: 0,
        }
    }

    /// Initialize architecture parameters
    fn initialize_alphas(&mut self) {
        let num_ops = self.mixed_ops.len();
        let num_edges = self.num_intermediate_nodes * (self.num_intermediate_nodes + 1) / 2;
        let alpha_normal = vec![vec![0.0f32; num_ops]; num_edges];
        self.alpha_normal = Some(alpha_normal);
    }

    /// Sample architecture from continuous distribution
    fn sample_architecture(&self) -> crate::nas::architecture_encoding::SequentialEncoding {
        use crate::nas::search_space::LayerType;
        use scirs2_core::random::prelude::*;
        let mut rng_inst = thread_rng();
        let alpha: Vec<Vec<f32>> = match &self.alpha_normal {
            Some(a) => a.clone(),
            None => {
                // Use default alpha values if not initialized
                let num_ops = self.mixed_ops.len().max(1);
                let num_edges = self.num_intermediate_nodes * (self.num_intermediate_nodes + 1) / 2;
                vec![vec![0.0f32; num_ops]; num_edges]
            }
        };
        let mut layers = Vec::new();
        for edge_logits in &alpha {
            let mut max_prob = f32::NEG_INFINITY;
            let mut selected_op = 0;
            let temp = self.temperature as f32;
            for (i, &logit) in edge_logits.iter().enumerate() {
                let u: f32 = rng_inst.random();
                let u_clamped = u.max(1e-40);
                let gumbel = -(-u_clamped.ln()).ln();
                let y = (logit + gumbel) / temp;
                if y > max_prob {
                    max_prob = y;
                    selected_op = i;
                }
            }
            if selected_op > 0 {
                if let Some(layer) = self.operation_to_layer_type(selected_op) {
                    layers.push(layer);
                }
            }
        }
        if layers.len() < 3 {
            layers.push(LayerType::Dense(128));
            layers.push(LayerType::Activation("relu".to_string()));
            layers.push(LayerType::Dense(64));
        }
        crate::nas::architecture_encoding::SequentialEncoding::new(layers)
    }

    /// Convert operation index to layer type
    fn operation_to_layer_type(
        &self,
        op_idx: usize,
    ) -> Option<crate::nas::search_space::LayerType> {
        use crate::nas::search_space::LayerType;
        let op = self.mixed_ops.get(op_idx)?;
        match op.as_str() {
            "none" => None,
            "max_pool_3x3" => Some(LayerType::MaxPool2D {
                pool_size: (3, 3),
                stride: (1, 1),
            }),
            "avg_pool_3x3" => Some(LayerType::AvgPool2D {
                pool_size: (3, 3),
                stride: (1, 1),
            }),
            "skip_connect" => Some(LayerType::Residual),
            "sep_conv_3x3" => Some(LayerType::Conv2D {
                filters: 64,
                kernel_size: (3, 3),
                stride: (1, 1),
            }),
            "sep_conv_5x5" => Some(LayerType::Conv2D {
                filters: 64,
                kernel_size: (5, 5),
                stride: (1, 1),
            }),
            "dil_conv_3x3" => Some(LayerType::Conv2D {
                filters: 64,
                kernel_size: (3, 3),
                stride: (1, 1),
            }),
            "dil_conv_5x5" => Some(LayerType::Conv2D {
                filters: 64,
                kernel_size: (5, 5),
                stride: (1, 1),
            }),
            _ => Some(LayerType::Dense(64)),
        }
    }

    /// Update architecture parameters
    fn update_alphas(&mut self, validation_loss: f64) {
        if let Some(ref mut alpha) = self.alpha_normal {
            let gradient_scale = self.arch_learning_rate * validation_loss as f32;
            for edge in alpha.iter_mut() {
                for param in edge.iter_mut() {
                    *param = *param * (1.0 - self.arch_weight_decay) - gradient_scale * 0.001;
                }
            }
        }
    }

    /// Progressive shrinking of temperature
    fn update_temperature(&mut self) {
        self.current_epoch += 1;
        self.temperature = (self.temperature * 0.98).max(0.1);
    }
}

impl SearchAlgorithm for DifferentiableSearch {
    fn propose_architectures(
        &self,
        _history: &[SearchResult],
        n_proposals: usize,
    ) -> Result<Vec<Arc<dyn ArchitectureEncoding>>> {
        let mut proposals = Vec::with_capacity(n_proposals);
        for _ in 0..n_proposals {
            let encoding = self.sample_architecture();
            proposals.push(Arc::new(encoding) as Arc<dyn ArchitectureEncoding>);
        }
        Ok(proposals)
    }

    fn update(&mut self, results: &[SearchResult]) -> Result<()> {
        if results.is_empty() {
            return Ok(());
        }
        let avg_loss = results
            .iter()
            .filter_map(|r| r.metrics.get("validation_loss"))
            .copied()
            .sum::<f64>()
            / results.len() as f64;
        self.update_alphas(avg_loss);
        self.update_temperature();
        Ok(())
    }

    fn name(&self) -> &str {
        "DifferentiableSearch"
    }
}

/// Bayesian optimization for architecture search
pub struct BayesianOptimization {
    surrogate_type: String,
    acquisition_function: String,
    n_initial_points: usize,
    xi: f64,
}

impl Default for BayesianOptimization {
    fn default() -> Self {
        Self::new()
    }
}

impl BayesianOptimization {
    /// Create a new Bayesian optimization search
    pub fn new() -> Self {
        Self {
            surrogate_type: "gaussian_process".to_string(),
            acquisition_function: "expected_improvement".to_string(),
            n_initial_points: 10,
            xi: 0.01,
        }
    }

    /// Set acquisition function
    pub fn with_acquisition(mut self, acquisition: &str) -> Self {
        self.acquisition_function = acquisition.to_string();
        self
    }
}

impl SearchAlgorithm for BayesianOptimization {
    fn propose_architectures(
        &self,
        history: &[SearchResult],
        n_proposals: usize,
    ) -> Result<Vec<Arc<dyn ArchitectureEncoding>>> {
        use scirs2_core::random::prelude::*;
        let mut rng_inst = thread_rng();
        if history.len() < self.n_initial_points {
            let mut proposals = Vec::with_capacity(n_proposals);
            for _ in 0..n_proposals {
                let encoding =
                    crate::nas::architecture_encoding::SequentialEncoding::random(&mut rng_inst)?;
                proposals.push(Arc::new(encoding) as Arc<dyn ArchitectureEncoding>);
            }
            return Ok(proposals);
        }
        let best_result = history.iter().max_by(|a, b| {
            let a_score = if a.metrics.is_empty() {
                0.0
            } else {
                a.metrics.values().sum::<f64>() / a.metrics.len() as f64
            };
            let b_score = if b.metrics.is_empty() {
                0.0
            } else {
                b.metrics.values().sum::<f64>() / b.metrics.len() as f64
            };
            a_score
                .partial_cmp(&b_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut proposals = Vec::with_capacity(n_proposals);
        if let Some(best) = best_result {
            while proposals.len() < n_proposals {
                let mutated = best.architecture.mutate(0.1)?;
                proposals.push(Arc::from(mutated));
            }
        } else {
            for _ in 0..n_proposals {
                let encoding =
                    crate::nas::architecture_encoding::SequentialEncoding::random(&mut rng_inst)?;
                proposals.push(Arc::new(encoding) as Arc<dyn ArchitectureEncoding>);
            }
        }
        Ok(proposals)
    }

    fn update(&mut self, _results: &[SearchResult]) -> Result<()> {
        Ok(())
    }

    fn name(&self) -> &str {
        "BayesianOptimization"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nas::EvaluationMetrics;

    fn create_dummy_result() -> SearchResult {
        use scirs2_core::random::prelude::*;
        let encoding =
            crate::nas::architecture_encoding::SequentialEncoding::random(&mut thread_rng())
                .expect("failed to create encoding");
        let mut metrics = EvaluationMetrics::new();
        metrics.insert("accuracy".to_string(), 0.95);
        SearchResult {
            architecture: Arc::new(encoding),
            metrics,
            training_time: 100.0,
            parameter_count: 1_000_000,
            flops: Some(1_000_000),
        }
    }

    #[test]
    fn test_random_search() {
        let search = RandomSearch::new();
        let proposals = search
            .propose_architectures(&[], 5)
            .expect("failed to propose");
        assert_eq!(proposals.len(), 5);
    }

    #[test]
    fn test_evolutionary_search() {
        let mut search = EvolutionarySearch::new(10);
        let results = vec![create_dummy_result(); 5];
        search.update(&results).expect("failed to update");
    }

    #[test]
    fn test_reinforcement_search() {
        let mut search = ReinforcementSearch::new();
        let proposals = search
            .propose_architectures(&[], 3)
            .expect("failed to propose");
        assert_eq!(proposals.len(), 3);
        let results = vec![create_dummy_result(); 3];
        search.update(&results).expect("failed to update");
        assert!(search.baseline.is_some());
    }

    #[test]
    fn test_bayesian_optimization() {
        let search = BayesianOptimization::new();
        let proposals = search
            .propose_architectures(&[], 3)
            .expect("failed to propose");
        assert_eq!(proposals.len(), 3);
    }
}
