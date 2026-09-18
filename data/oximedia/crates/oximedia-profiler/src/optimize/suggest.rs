//! Optimization suggestions.

use crate::bottleneck::detect::Bottleneck;
use crate::cpu::hotspot::Hotspot;
use serde::{Deserialize, Serialize};

/// Optimization suggestion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Suggestion {
    /// Target (function/module).
    pub target: String,

    /// Suggestion category.
    pub category: SuggestionCategory,

    /// Description.
    pub description: String,

    /// Expected impact.
    pub expected_impact: ImpactLevel,

    /// Implementation difficulty.
    pub difficulty: DifficultyLevel,

    /// Code example.
    pub example: Option<String>,
}

/// Suggestion category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SuggestionCategory {
    /// Algorithm optimization.
    Algorithm,

    /// Memory optimization.
    Memory,

    /// I/O optimization.
    IO,

    /// Parallelization.
    Parallel,

    /// Caching.
    Cache,

    /// Data structure.
    DataStructure,

    /// Other.
    Other,
}

/// Impact level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImpactLevel {
    /// High impact (>20% improvement).
    High,

    /// Medium impact (5-20% improvement).
    Medium,

    /// Low impact (<5% improvement).
    Low,
}

/// Difficulty level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DifficultyLevel {
    /// Easy to implement.
    Easy,

    /// Moderate difficulty.
    Moderate,

    /// Difficult to implement.
    Hard,
}

/// Optimization suggester.
#[derive(Debug)]
pub struct OptimizationSuggester {
    suggestions: Vec<Suggestion>,
}

impl OptimizationSuggester {
    /// Create a new optimization suggester.
    pub fn new() -> Self {
        Self {
            suggestions: Vec::new(),
        }
    }

    /// Generate suggestions from hotspots.
    pub fn suggest_from_hotspots(&mut self, hotspots: &[Hotspot]) {
        for hotspot in hotspots {
            if hotspot.is_critical() {
                let suggestion = Suggestion {
                    target: hotspot.function.clone(),
                    category: SuggestionCategory::Algorithm,
                    description: format!(
                        "Critical hotspot consuming {:.2}% of execution time",
                        hotspot.time_percentage
                    ),
                    expected_impact: ImpactLevel::High,
                    difficulty: DifficultyLevel::Moderate,
                    example: None,
                };
                self.suggestions.push(suggestion);
            }
        }
    }

    /// Generate suggestions from bottlenecks.
    pub fn suggest_from_bottlenecks(&mut self, bottlenecks: &[Bottleneck]) {
        for bottleneck in bottlenecks {
            if let Some(ref suggestion_text) = bottleneck.suggestion {
                let category = if bottleneck.location.contains("alloc") {
                    SuggestionCategory::Memory
                } else if bottleneck.location.contains("io") {
                    SuggestionCategory::IO
                } else {
                    SuggestionCategory::Other
                };

                let suggestion = Suggestion {
                    target: bottleneck.location.clone(),
                    category,
                    description: suggestion_text.clone(),
                    expected_impact: if bottleneck.is_critical() {
                        ImpactLevel::High
                    } else {
                        ImpactLevel::Medium
                    },
                    difficulty: DifficultyLevel::Moderate,
                    example: None,
                };
                self.suggestions.push(suggestion);
            }
        }
    }

    /// Get all suggestions.
    pub fn suggestions(&self) -> &[Suggestion] {
        &self.suggestions
    }

    /// Get suggestions by category.
    pub fn by_category(&self, category: SuggestionCategory) -> Vec<&Suggestion> {
        self.suggestions
            .iter()
            .filter(|s| s.category == category)
            .collect()
    }

    /// Get high-impact suggestions.
    pub fn high_impact(&self) -> Vec<&Suggestion> {
        self.suggestions
            .iter()
            .filter(|s| s.expected_impact == ImpactLevel::High)
            .collect()
    }

    /// Generate a report.
    pub fn report(&self) -> String {
        let mut report = String::new();

        report.push_str(&format!(
            "Optimization Suggestions: {}\n\n",
            self.suggestions.len()
        ));

        for (i, suggestion) in self.suggestions.iter().enumerate() {
            report.push_str(&format!("{}. {}\n", i + 1, suggestion.target));
            report.push_str(&format!("   Category: {:?}\n", suggestion.category));
            report.push_str(&format!("   {}\n", suggestion.description));
            report.push_str(&format!(
                "   Expected Impact: {:?}\n",
                suggestion.expected_impact
            ));
            report.push_str(&format!("   Difficulty: {:?}\n\n", suggestion.difficulty));
        }

        report
    }
}

impl Default for OptimizationSuggester {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_optimization_suggester() {
        let suggester = OptimizationSuggester::new();
        assert_eq!(suggester.suggestions().len(), 0);
    }

    #[test]
    fn test_suggest_from_hotspots() {
        let mut suggester = OptimizationSuggester::new();

        let hotspot = Hotspot::new("test_function".to_string(), Duration::from_secs(1), 100)
            .with_percentage(80.0);

        suggester.suggest_from_hotspots(&[hotspot]);
        assert_eq!(suggester.suggestions().len(), 1);
    }

    #[test]
    fn test_suggest_from_bottlenecks() {
        let mut suggester = OptimizationSuggester::new();

        let bottleneck = Bottleneck::new(
            "test".to_string(),
            "memory_alloc".to_string(),
            Duration::from_secs(1),
        )
        .with_suggestion("Reduce allocations".to_string());

        suggester.suggest_from_bottlenecks(&[bottleneck]);
        assert_eq!(suggester.suggestions().len(), 1);
    }

    #[test]
    fn test_high_impact_filter() {
        let mut suggester = OptimizationSuggester::new();

        let hotspot = Hotspot::new("critical_function".to_string(), Duration::from_secs(1), 100)
            .with_percentage(80.0);

        suggester.suggest_from_hotspots(&[hotspot]);

        let high_impact = suggester.high_impact();
        assert_eq!(high_impact.len(), 1);
    }
}
