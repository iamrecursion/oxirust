//! Switching hybrid recommendation strategy.

/// Switching hybrid strategy selector
pub struct SwitchingStrategy {
    /// User history threshold for switching
    history_threshold: usize,
}

impl SwitchingStrategy {
    /// Create a new switching strategy
    #[must_use]
    pub fn new(history_threshold: usize) -> Self {
        Self { history_threshold }
    }

    /// Select the best method based on context
    #[must_use]
    pub fn select_method(&self, context: &SwitchContext) -> RecommendMethod {
        // If user is new (cold start), use content-based
        if context.user_history_length < self.history_threshold {
            return RecommendMethod::ContentBased;
        }

        // If user has enough history, use collaborative
        if context.user_history_length >= self.history_threshold * 2 {
            return RecommendMethod::Collaborative;
        }

        // During peak hours, consider trending
        if context.is_peak_hours && context.user_history_length > 0 {
            return RecommendMethod::Trending;
        }

        // Default to hybrid
        RecommendMethod::Hybrid
    }

    /// Determine if should switch methods
    #[must_use]
    pub fn should_switch(&self, current: &RecommendMethod, context: &SwitchContext) -> bool {
        let optimal = self.select_method(context);
        !matches!(
            (current, optimal),
            (RecommendMethod::ContentBased, RecommendMethod::ContentBased)
                | (
                    RecommendMethod::Collaborative,
                    RecommendMethod::Collaborative
                )
                | (RecommendMethod::Trending, RecommendMethod::Trending)
                | (RecommendMethod::Hybrid, RecommendMethod::Hybrid)
        )
    }
}

impl Default for SwitchingStrategy {
    fn default() -> Self {
        Self::new(5)
    }
}

/// Context for switching decisions
#[derive(Debug, Clone)]
pub struct SwitchContext {
    /// User history length
    pub user_history_length: usize,
    /// Is peak hours
    pub is_peak_hours: bool,
    /// User engagement score
    pub engagement_score: f32,
    /// Content availability
    pub content_availability: usize,
}

impl Default for SwitchContext {
    fn default() -> Self {
        Self {
            user_history_length: 0,
            is_peak_hours: false,
            engagement_score: 0.5,
            content_availability: 100,
        }
    }
}

/// Recommendation method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecommendMethod {
    /// Content-based filtering
    ContentBased,
    /// Collaborative filtering
    Collaborative,
    /// Trending content
    Trending,
    /// Hybrid approach
    Hybrid,
}

/// Adaptive switching strategy
pub struct AdaptiveSwitching {
    /// Base strategy
    base_strategy: SwitchingStrategy,
    /// Performance tracking
    method_performance: MethodPerformance,
}

impl AdaptiveSwitching {
    /// Create a new adaptive switching strategy
    #[must_use]
    pub fn new() -> Self {
        Self {
            base_strategy: SwitchingStrategy::default(),
            method_performance: MethodPerformance::new(),
        }
    }

    /// Select method with performance consideration
    #[must_use]
    pub fn select_method(&self, context: &SwitchContext) -> RecommendMethod {
        let base_method = self.base_strategy.select_method(context);

        // Check if we should override based on performance
        let best_performing = self.method_performance.get_best_method();

        if self.method_performance.get_confidence(best_performing) > 0.8 {
            best_performing
        } else {
            base_method
        }
    }

    /// Update performance metrics
    pub fn update_performance(&mut self, method: RecommendMethod, success: bool) {
        self.method_performance.record_outcome(method, success);
    }
}

impl Default for AdaptiveSwitching {
    fn default() -> Self {
        Self::new()
    }
}

/// Method performance tracker
struct MethodPerformance {
    /// Success counts
    successes: [u32; 4],
    /// Total counts
    totals: [u32; 4],
}

impl MethodPerformance {
    fn new() -> Self {
        Self {
            successes: [0; 4],
            totals: [0; 4],
        }
    }

    fn record_outcome(&mut self, method: RecommendMethod, success: bool) {
        let idx = method_to_index(method);
        self.totals[idx] += 1;
        if success {
            self.successes[idx] += 1;
        }
    }

    fn get_confidence(&self, method: RecommendMethod) -> f32 {
        let idx = method_to_index(method);
        if self.totals[idx] == 0 {
            0.0
        } else {
            self.successes[idx] as f32 / self.totals[idx] as f32
        }
    }

    fn get_best_method(&self) -> RecommendMethod {
        let mut best_idx = 0;
        let mut best_confidence = 0.0;

        for i in 0..4 {
            if self.totals[i] > 0 {
                let confidence = self.successes[i] as f32 / self.totals[i] as f32;
                if confidence > best_confidence {
                    best_confidence = confidence;
                    best_idx = i;
                }
            }
        }

        index_to_method(best_idx)
    }
}

fn method_to_index(method: RecommendMethod) -> usize {
    match method {
        RecommendMethod::ContentBased => 0,
        RecommendMethod::Collaborative => 1,
        RecommendMethod::Trending => 2,
        RecommendMethod::Hybrid => 3,
    }
}

fn index_to_method(idx: usize) -> RecommendMethod {
    match idx {
        0 => RecommendMethod::ContentBased,
        1 => RecommendMethod::Collaborative,
        2 => RecommendMethod::Trending,
        _ => RecommendMethod::Hybrid,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_switching_strategy() {
        let strategy = SwitchingStrategy::new(5);
        let context = SwitchContext {
            user_history_length: 3,
            ..Default::default()
        };

        let method = strategy.select_method(&context);
        assert_eq!(method, RecommendMethod::ContentBased);
    }

    #[test]
    fn test_switching_with_history() {
        let strategy = SwitchingStrategy::new(5);
        let context = SwitchContext {
            user_history_length: 15,
            ..Default::default()
        };

        let method = strategy.select_method(&context);
        assert_eq!(method, RecommendMethod::Collaborative);
    }

    #[test]
    fn test_should_switch() {
        let strategy = SwitchingStrategy::new(5);
        let current = RecommendMethod::ContentBased;
        let context = SwitchContext {
            user_history_length: 15,
            ..Default::default()
        };

        assert!(strategy.should_switch(&current, &context));
    }

    #[test]
    fn test_adaptive_switching() {
        let mut adaptive = AdaptiveSwitching::new();
        adaptive.update_performance(RecommendMethod::ContentBased, true);
        adaptive.update_performance(RecommendMethod::ContentBased, true);

        let context = SwitchContext::default();
        let _method = adaptive.select_method(&context);
    }
}
