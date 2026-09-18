//! Category balancing for diversity.

use std::collections::HashMap;

/// Category balancer
pub struct CategoryBalancer {
    /// Target distribution (category -> proportion)
    target_distribution: HashMap<String, f32>,
}

impl CategoryBalancer {
    /// Create a new category balancer
    #[must_use]
    pub fn new() -> Self {
        Self {
            target_distribution: HashMap::new(),
        }
    }

    /// Set target distribution for a category
    pub fn set_target(&mut self, category: String, proportion: f32) {
        self.target_distribution
            .insert(category, proportion.clamp(0.0, 1.0));
    }

    /// Calculate category distribution from recommendations
    #[must_use]
    pub fn calculate_distribution(&self, categories: &[Vec<String>]) -> HashMap<String, f32> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        let mut total = 0;

        for cat_list in categories {
            for category in cat_list {
                *counts.entry(category.clone()).or_insert(0) += 1;
                total += 1;
            }
        }

        counts
            .into_iter()
            .map(|(cat, count)| (cat, count as f32 / total as f32))
            .collect()
    }

    /// Calculate deviation from target distribution
    #[must_use]
    pub fn calculate_deviation(&self, actual: &HashMap<String, f32>) -> f32 {
        let mut total_deviation = 0.0;

        for (category, &target_prop) in &self.target_distribution {
            let actual_prop = actual.get(category).unwrap_or(&0.0);
            total_deviation += (target_prop - actual_prop).abs();
        }

        total_deviation
    }
}

impl Default for CategoryBalancer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_category_balancer() {
        let mut balancer = CategoryBalancer::new();
        balancer.set_target(String::from("Action"), 0.3);
        balancer.set_target(String::from("Drama"), 0.3);

        assert_eq!(balancer.target_distribution.len(), 2);
    }

    #[test]
    fn test_calculate_distribution() {
        let balancer = CategoryBalancer::new();
        let categories = vec![
            vec![String::from("Action")],
            vec![String::from("Action")],
            vec![String::from("Drama")],
        ];

        let dist = balancer.calculate_distribution(&categories);
        assert!(dist.contains_key("Action"));
    }
}
