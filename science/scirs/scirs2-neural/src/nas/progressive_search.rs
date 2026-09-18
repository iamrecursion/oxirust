//! Progressive Neural Architecture Search
//!
//! This module implements progressive search strategies that adaptively expand
//! the search space and improve search efficiency over time.

use crate::error::Result;
use crate::nas::{
    architecture_encoding::ArchitectureEncoding, search_space::LayerType, SearchResult,
    SearchSpace, SearchSpaceConfig,
};
use std::sync::Arc;

/// Configuration for progressive search
#[derive(Debug, Clone)]
pub struct ProgressiveConfig {
    pub initial_search_space: SearchSpaceConfig,
    pub num_stages: usize,
    pub architectures_per_stage: usize,
    pub expansion_strategy: ExpansionStrategy,
    pub advancement_threshold: f64,
    pub max_complexity_increase: f64,
    pub early_stopping_patience: usize,
}

impl Default for ProgressiveConfig {
    fn default() -> Self {
        Self {
            initial_search_space: SearchSpaceConfig::default(),
            num_stages: 5,
            architectures_per_stage: 50,
            expansion_strategy: ExpansionStrategy::AdaptiveComplexity,
            advancement_threshold: 0.02,
            max_complexity_increase: 0.5,
            early_stopping_patience: 2,
        }
    }
}

/// Strategies for expanding the search space
#[derive(Debug, Clone)]
pub enum ExpansionStrategy {
    AdaptiveComplexity,
    LayerTypeExpansion,
    ScaleExpansion,
    ConnectionExpansion,
    Composite(Vec<ExpansionStrategy>),
}

/// Progressive search implementation
pub struct ProgressiveSearch {
    config: ProgressiveConfig,
    current_stage: usize,
    search_spaces: Vec<SearchSpace>,
    stage_results: Vec<Vec<SearchResult>>,
    best_per_stage: Vec<Option<Arc<dyn ArchitectureEncoding>>>,
    complexity_history: Vec<f64>,
    performance_history: Vec<f64>,
    stagnation_counter: usize,
}

impl ProgressiveSearch {
    pub fn new(config: ProgressiveConfig) -> Result<Self> {
        let initial_space = SearchSpace::new(config.initial_search_space.clone())?;
        Ok(Self {
            config,
            current_stage: 0,
            search_spaces: vec![initial_space],
            stage_results: vec![Vec::new()],
            best_per_stage: vec![None],
            complexity_history: Vec::new(),
            performance_history: Vec::new(),
            stagnation_counter: 0,
        })
    }

    pub fn current_search_space(&self) -> &SearchSpace {
        &self.search_spaces[self.current_stage]
    }

    pub fn add_stage_results(&mut self, results: Vec<SearchResult>) -> Result<()> {
        while self.stage_results.len() <= self.current_stage {
            self.stage_results.push(Vec::new());
        }
        self.stage_results[self.current_stage].extend(results);
        if let Some(best_result) = self.get_best_result_in_stage(self.current_stage) {
            let arch = best_result.architecture.clone();
            let perf =
                best_result.metrics.values().sum::<f64>() / best_result.metrics.len().max(1) as f64;
            let complexity = self.estimate_architecture_complexity(&arch)?;
            while self.best_per_stage.len() <= self.current_stage {
                self.best_per_stage.push(None);
            }
            self.best_per_stage[self.current_stage] = Some(arch);
            self.performance_history.push(perf);
            self.complexity_history.push(complexity);
        }
        Ok(())
    }

    pub fn should_advance_stage(&self) -> bool {
        if self.current_stage >= self.config.num_stages - 1 {
            return false;
        }
        let current_evals = self
            .stage_results
            .get(self.current_stage)
            .map(|r| r.len())
            .unwrap_or(0);
        if current_evals < self.config.architectures_per_stage {
            return false;
        }
        if self.performance_history.len() >= 2 {
            let cur = self.performance_history.last().copied().unwrap_or(0.0);
            let prev = self.performance_history[self.performance_history.len() - 2];
            let improvement = if prev.abs() > 1e-12 {
                (cur - prev) / prev.abs()
            } else {
                0.0
            };
            if improvement >= self.config.advancement_threshold {
                return true;
            }
        }
        self.stagnation_counter >= self.config.early_stopping_patience
    }

    pub fn advance_stage(&mut self) -> Result<()> {
        if self.current_stage >= self.config.num_stages - 1 {
            return Ok(());
        }
        self.current_stage += 1;
        let expanded_space = self.expand_search_space()?;
        self.search_spaces.push(expanded_space);
        self.stage_results.push(Vec::new());
        self.best_per_stage.push(None);
        self.stagnation_counter = 0;
        Ok(())
    }

    fn expand_search_space(&self) -> Result<SearchSpace> {
        let mut expanded_config = self.config.initial_search_space.clone();
        let strategy = self.config.expansion_strategy.clone();
        self.apply_expansion_strategy(&strategy, &mut expanded_config)?;
        SearchSpace::new(expanded_config)
    }

    fn apply_expansion_strategy(
        &self,
        strategy: &ExpansionStrategy,
        config: &mut SearchSpaceConfig,
    ) -> Result<()> {
        match strategy {
            ExpansionStrategy::AdaptiveComplexity => self.expand_by_complexity(config)?,
            ExpansionStrategy::LayerTypeExpansion => self.expand_by_layer_types(config)?,
            ExpansionStrategy::ScaleExpansion => self.expand_by_scale(config)?,
            ExpansionStrategy::ConnectionExpansion => self.expand_by_connections(config)?,
            ExpansionStrategy::Composite(strategies) => {
                for s in strategies {
                    match s {
                        ExpansionStrategy::AdaptiveComplexity => {
                            self.expand_by_complexity(config)?
                        }
                        ExpansionStrategy::LayerTypeExpansion => {
                            self.expand_by_layer_types(config)?
                        }
                        ExpansionStrategy::ScaleExpansion => self.expand_by_scale(config)?,
                        ExpansionStrategy::ConnectionExpansion => {
                            self.expand_by_connections(config)?
                        }
                        _ => {} // Avoid infinite recursion for nested Composite
                    }
                }
            }
        }
        Ok(())
    }

    fn expand_by_complexity(&self, config: &mut SearchSpaceConfig) -> Result<()> {
        let factor = 1.0
            + (self.current_stage as f64 * self.config.max_complexity_increase
                / self.config.num_stages.max(1) as f64);
        let mut new_types = Vec::new();
        for lt in &config.layer_types {
            match lt {
                LayerType::Dense(units) => {
                    new_types.push(LayerType::Dense((*units as f64 * factor) as usize));
                }
                LayerType::Conv2D {
                    filters,
                    kernel_size,
                    stride,
                } => {
                    new_types.push(LayerType::Conv2D {
                        filters: (*filters as f64 * factor) as usize,
                        kernel_size: *kernel_size,
                        stride: *stride,
                    });
                }
                other => new_types.push(other.clone()),
            }
        }
        new_types.extend(config.layer_types.clone());
        config.layer_types = new_types;
        config.max_layers = (config.max_layers as f64 * factor) as usize;
        Ok(())
    }

    fn expand_by_layer_types(&self, config: &mut SearchSpaceConfig) -> Result<()> {
        match self.current_stage {
            1 => {
                config.layer_types.push(LayerType::Attention {
                    num_heads: 4,
                    key_dim: 64,
                });
                config.layer_types.push(LayerType::LayerNorm);
            }
            2 => {
                config.layer_types.push(LayerType::LSTM {
                    units: 128,
                    return_sequences: false,
                });
                config.layer_types.push(LayerType::GRU {
                    units: 128,
                    return_sequences: false,
                });
            }
            3 => {
                config.layer_types.push(LayerType::Conv2D {
                    filters: 128,
                    kernel_size: (5, 5),
                    stride: (1, 1),
                });
                config.layer_types.push(LayerType::Conv2D {
                    filters: 256,
                    kernel_size: (7, 7),
                    stride: (1, 1),
                });
            }
            _ => {
                config.layer_types.push(LayerType::Embedding {
                    vocab_size: 10000,
                    embedding_dim: 128,
                });
            }
        }
        Ok(())
    }

    fn expand_by_scale(&self, config: &mut SearchSpaceConfig) -> Result<()> {
        let scale_factor = 1.0 + (self.current_stage as f32 * 0.25);
        for &base in &[0.5f32, 0.75, 1.0, 1.25, 1.5] {
            let new_mult = base * scale_factor;
            if !config.width_multipliers.contains(&new_mult) {
                config.width_multipliers.push(new_mult);
            }
            if !config.depth_multipliers.contains(&new_mult) {
                config.depth_multipliers.push(new_mult);
            }
        }
        Ok(())
    }

    fn expand_by_connections(&self, config: &mut SearchSpaceConfig) -> Result<()> {
        config.allow_branches = true;
        config.skip_connection_prob = (config.skip_connection_prob + 0.1).min(0.8);
        config.max_branches = (config.max_branches + 1).min(5);
        Ok(())
    }

    fn get_best_result_in_stage(&self, stage: usize) -> Option<&SearchResult> {
        self.stage_results.get(stage)?.iter().max_by(|a, b| {
            let as_ = a.metrics.values().sum::<f64>() / a.metrics.len().max(1) as f64;
            let bs = b.metrics.values().sum::<f64>() / b.metrics.len().max(1) as f64;
            as_.partial_cmp(&bs).unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    fn estimate_architecture_complexity(
        &self,
        architecture: &Arc<dyn ArchitectureEncoding>,
    ) -> Result<f64> {
        let vec = architecture.to_vector();
        let complexity = vec.iter().map(|x| x.abs()).sum::<f64>();
        Ok(complexity / vec.len().max(1) as f64)
    }

    pub fn current_stage(&self) -> usize {
        self.current_stage
    }

    pub fn total_stages(&self) -> usize {
        self.config.num_stages
    }

    pub fn get_all_results(&self) -> &[Vec<SearchResult>] {
        &self.stage_results
    }

    pub fn get_global_best(&self) -> Option<&Arc<dyn ArchitectureEncoding>> {
        let mut best_perf = f64::NEG_INFINITY;
        let mut best_arch = None;
        for results in &self.stage_results {
            if let Some(best) = results.iter().max_by(|a, b| {
                let as_ = a.metrics.values().sum::<f64>() / a.metrics.len().max(1) as f64;
                let bs = b.metrics.values().sum::<f64>() / b.metrics.len().max(1) as f64;
                as_.partial_cmp(&bs).unwrap_or(std::cmp::Ordering::Equal)
            }) {
                let perf = best.metrics.values().sum::<f64>() / best.metrics.len().max(1) as f64;
                if perf > best_perf {
                    best_perf = perf;
                    best_arch = Some(&best.architecture);
                }
            }
        }
        best_arch
    }

    pub fn get_performance_trend(&self) -> &[f64] {
        &self.performance_history
    }

    pub fn get_complexity_trend(&self) -> &[f64] {
        &self.complexity_history
    }

    pub fn has_converged(&self) -> bool {
        if self.performance_history.len() < 3 {
            return false;
        }
        let recent: Vec<f64> = self
            .performance_history
            .windows(2)
            .take(3)
            .map(|w| {
                if w[0].abs() > 1e-12 {
                    (w[1] - w[0]) / w[0].abs()
                } else {
                    0.0
                }
            })
            .collect();
        let avg = recent.iter().sum::<f64>() / recent.len() as f64;
        avg < self.config.advancement_threshold / 2.0
    }

    pub fn generate_evolution_report(&self) -> String {
        let mut report = String::new();
        report.push_str("# Progressive Search Evolution Report\n\n");
        report.push_str(&format!(
            "Current Stage: {}/{}\n",
            self.current_stage + 1,
            self.config.num_stages
        ));
        let total_evals: usize = self.stage_results.iter().map(|r| r.len()).sum();
        report.push_str(&format!("Total Evaluations: {}\n", total_evals));
        report.push_str("\n## Performance Evolution\n");
        for (stage, &perf) in self.performance_history.iter().enumerate() {
            report.push_str(&format!("Stage {}: {:.4}\n", stage + 1, perf));
        }
        report.push_str("\n## Complexity Evolution\n");
        for (stage, &complexity) in self.complexity_history.iter().enumerate() {
            report.push_str(&format!("Stage {}: {:.4}\n", stage + 1, complexity));
        }
        if let Some(best) = self.get_global_best() {
            report.push_str("\n## Best Architecture Found\n");
            report.push_str(&format!("{}\n", best));
        }
        report.push_str("\n## Convergence Status\n");
        report.push_str(&format!("Converged: {}\n", self.has_converged()));
        report.push_str(&format!(
            "Stagnation Counter: {}\n",
            self.stagnation_counter
        ));
        report
    }
}

/// Progressive search builder for easier configuration
pub struct ProgressiveSearchBuilder {
    config: ProgressiveConfig,
}

impl ProgressiveSearchBuilder {
    pub fn new() -> Self {
        Self {
            config: ProgressiveConfig::default(),
        }
    }

    pub fn stages(mut self, num_stages: usize) -> Self {
        self.config.num_stages = num_stages;
        self
    }

    pub fn architectures_per_stage(mut self, count: usize) -> Self {
        self.config.architectures_per_stage = count;
        self
    }

    pub fn expansion_strategy(mut self, strategy: ExpansionStrategy) -> Self {
        self.config.expansion_strategy = strategy;
        self
    }

    pub fn advancement_threshold(mut self, threshold: f64) -> Self {
        self.config.advancement_threshold = threshold;
        self
    }

    pub fn build(self) -> Result<ProgressiveSearch> {
        ProgressiveSearch::new(self.config)
    }
}

impl Default for ProgressiveSearchBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progressive_config_default() {
        let config = ProgressiveConfig::default();
        assert_eq!(config.num_stages, 5);
        assert_eq!(config.architectures_per_stage, 50);
    }

    #[test]
    fn test_progressive_search_creation() {
        let config = ProgressiveConfig::default();
        let search = ProgressiveSearch::new(config).expect("failed to create");
        assert_eq!(search.current_stage(), 0);
        assert_eq!(search.total_stages(), 5);
    }

    #[test]
    fn test_builder_pattern() {
        let search = ProgressiveSearchBuilder::new()
            .stages(3)
            .architectures_per_stage(25)
            .advancement_threshold(0.05)
            .build()
            .expect("failed to build");
        assert_eq!(search.total_stages(), 3);
    }
}
