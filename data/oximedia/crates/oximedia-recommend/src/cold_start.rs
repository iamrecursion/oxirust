#![allow(dead_code)]
//! Cold start strategies for recommendation systems.
//!
//! Handles the fundamental cold start problem in recommendations:
//! new users with no interaction history, and new items with no ratings.
//! Provides popularity-based fallbacks, demographic matching, content
//! attribute-based bootstrapping, and onboarding questionnaire scoring.

use std::collections::HashMap;

/// Strategy for handling cold start scenarios.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColdStartStrategy {
    /// Use globally popular items.
    Popularity,
    /// Match by demographic segment.
    DemographicMatch,
    /// Bootstrap from content attributes (genre, tags).
    ContentAttribute,
    /// Use onboarding questionnaire preferences.
    OnboardingPreference,
    /// Combine multiple strategies with weights.
    Hybrid,
}

impl std::fmt::Display for ColdStartStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Popularity => write!(f, "Popularity"),
            Self::DemographicMatch => write!(f, "DemographicMatch"),
            Self::ContentAttribute => write!(f, "ContentAttribute"),
            Self::OnboardingPreference => write!(f, "OnboardingPreference"),
            Self::Hybrid => write!(f, "Hybrid"),
        }
    }
}

/// Demographic segment for a user.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DemographicSegment {
    /// Age group (e.g., "18-24", "25-34").
    pub age_group: String,
    /// Region or country code.
    pub region: String,
    /// Language preference.
    pub language: String,
}

impl DemographicSegment {
    /// Create a new demographic segment.
    pub fn new(
        age_group: impl Into<String>,
        region: impl Into<String>,
        language: impl Into<String>,
    ) -> Self {
        Self {
            age_group: age_group.into(),
            region: region.into(),
            language: language.into(),
        }
    }

    /// Returns a matching key for looking up segment preferences.
    #[must_use]
    pub fn segment_key(&self) -> String {
        format!("{}:{}:{}", self.age_group, self.region, self.language)
    }
}

/// A scored item from cold start resolution.
#[derive(Debug, Clone)]
pub struct ColdStartItem {
    /// Item identifier.
    pub item_id: String,
    /// Score assigned by the cold start strategy.
    pub score: f64,
    /// Which strategy produced this item.
    pub strategy: ColdStartStrategy,
    /// Explanation of why this item was selected.
    pub reason: String,
}

/// Popularity data for a single item.
#[derive(Debug, Clone)]
pub struct PopularityEntry {
    /// Item identifier.
    pub item_id: String,
    /// View count.
    pub view_count: u64,
    /// Average rating (0-5).
    pub avg_rating: f64,
    /// Categories/genres.
    pub categories: Vec<String>,
    /// Language of the content.
    pub language: String,
}

/// Configuration for the cold start resolver.
#[derive(Debug, Clone)]
pub struct ColdStartConfig {
    /// Primary strategy to use.
    pub strategy: ColdStartStrategy,
    /// Weight for popularity in hybrid mode (0.0-1.0).
    pub popularity_weight: f64,
    /// Weight for demographic matching in hybrid mode (0.0-1.0).
    pub demographic_weight: f64,
    /// Weight for content attributes in hybrid mode (0.0-1.0).
    pub content_weight: f64,
    /// Minimum interactions before a user is considered "warm".
    pub warm_threshold: u32,
    /// Maximum items to return.
    pub max_results: usize,
}

impl Default for ColdStartConfig {
    fn default() -> Self {
        Self {
            strategy: ColdStartStrategy::Hybrid,
            popularity_weight: 0.4,
            demographic_weight: 0.3,
            content_weight: 0.3,
            warm_threshold: 5,
            max_results: 20,
        }
    }
}

/// The cold start resolver.
///
/// Resolves recommendations for new users and items using various
/// fallback strategies.
#[derive(Debug)]
pub struct ColdStartResolver {
    /// Configuration.
    config: ColdStartConfig,
    /// Global popularity chart.
    popularity: Vec<PopularityEntry>,
    /// Per-demographic preferences: `segment_key` -> list of preferred categories.
    demographic_preferences: HashMap<String, Vec<String>>,
    /// Onboarding preference data: user -> preferred categories.
    onboarding_data: HashMap<String, Vec<String>>,
    /// Total resolutions performed.
    resolution_count: u64,
}

impl ColdStartResolver {
    /// Create a new cold start resolver.
    #[must_use]
    pub fn new(config: ColdStartConfig) -> Self {
        Self {
            config,
            popularity: Vec::new(),
            demographic_preferences: HashMap::new(),
            onboarding_data: HashMap::new(),
            resolution_count: 0,
        }
    }

    /// Add a popularity entry.
    pub fn add_popularity_entry(&mut self, entry: PopularityEntry) {
        self.popularity.push(entry);
    }

    /// Set demographic preferences for a segment.
    pub fn set_demographic_preferences(
        &mut self,
        segment: &DemographicSegment,
        categories: Vec<String>,
    ) {
        self.demographic_preferences
            .insert(segment.segment_key(), categories);
    }

    /// Set onboarding preferences for a user.
    pub fn set_onboarding_preferences(
        &mut self,
        user_id: impl Into<String>,
        categories: Vec<String>,
    ) {
        self.onboarding_data.insert(user_id.into(), categories);
    }

    /// Check if a user is cold (below warm threshold).
    #[must_use]
    pub fn is_cold_user(&self, interaction_count: u32) -> bool {
        interaction_count < self.config.warm_threshold
    }

    /// Resolve cold start recommendations for a user.
    #[allow(clippy::cast_precision_loss)]
    pub fn resolve(
        &mut self,
        user_id: &str,
        segment: Option<&DemographicSegment>,
    ) -> Vec<ColdStartItem> {
        self.resolution_count += 1;

        match self.config.strategy {
            ColdStartStrategy::Popularity => self.resolve_popularity(),
            ColdStartStrategy::DemographicMatch => self.resolve_demographic(segment),
            ColdStartStrategy::ContentAttribute => self.resolve_content_attribute(user_id),
            ColdStartStrategy::OnboardingPreference => self.resolve_onboarding(user_id),
            ColdStartStrategy::Hybrid => self.resolve_hybrid(user_id, segment),
        }
    }

    /// Popularity-based resolution.
    #[allow(clippy::cast_precision_loss)]
    fn resolve_popularity(&self) -> Vec<ColdStartItem> {
        let mut sorted = self.popularity.clone();
        sorted.sort_by(|a, b| b.view_count.cmp(&a.view_count));
        sorted
            .into_iter()
            .take(self.config.max_results)
            .enumerate()
            .map(|(i, entry)| {
                let score = 1.0 - (i as f64 / self.config.max_results.max(1) as f64);
                ColdStartItem {
                    item_id: entry.item_id,
                    score,
                    strategy: ColdStartStrategy::Popularity,
                    reason: format!("Popular with {} views", entry.view_count),
                }
            })
            .collect()
    }

    /// Demographic-based resolution.
    #[allow(clippy::cast_precision_loss)]
    fn resolve_demographic(&self, segment: Option<&DemographicSegment>) -> Vec<ColdStartItem> {
        let Some(segment) = segment else {
            return self.resolve_popularity();
        };

        let key = segment.segment_key();
        let preferred = match self.demographic_preferences.get(&key) {
            Some(cats) => cats,
            None => return self.resolve_popularity(),
        };

        let mut results: Vec<ColdStartItem> = self
            .popularity
            .iter()
            .filter(|entry| entry.categories.iter().any(|c| preferred.contains(c)))
            .take(self.config.max_results)
            .enumerate()
            .map(|(i, entry)| {
                let score = 1.0 - (i as f64 / self.config.max_results.max(1) as f64);
                ColdStartItem {
                    item_id: entry.item_id.clone(),
                    score,
                    strategy: ColdStartStrategy::DemographicMatch,
                    reason: format!("Matches {key} segment preferences"),
                }
            })
            .collect();

        // Fill with popularity if not enough
        if results.len() < self.config.max_results {
            let existing_ids: Vec<_> = results.iter().map(|r| r.item_id.clone()).collect();
            let mut pop = self.resolve_popularity();
            pop.retain(|p| !existing_ids.contains(&p.item_id));
            results.extend(
                pop.into_iter()
                    .take(self.config.max_results - results.len()),
            );
        }

        results
    }

    /// Content attribute-based resolution using onboarding data.
    #[allow(clippy::cast_precision_loss)]
    fn resolve_content_attribute(&self, user_id: &str) -> Vec<ColdStartItem> {
        let preferred = match self.onboarding_data.get(user_id) {
            Some(cats) => cats,
            None => return self.resolve_popularity(),
        };

        self.popularity
            .iter()
            .filter(|entry| entry.categories.iter().any(|c| preferred.contains(c)))
            .take(self.config.max_results)
            .enumerate()
            .map(|(i, entry)| {
                let score = 1.0 - (i as f64 / self.config.max_results.max(1) as f64);
                ColdStartItem {
                    item_id: entry.item_id.clone(),
                    score,
                    strategy: ColdStartStrategy::ContentAttribute,
                    reason: "Matches preferred content attributes".to_string(),
                }
            })
            .collect()
    }

    /// Onboarding preference resolution.
    fn resolve_onboarding(&self, user_id: &str) -> Vec<ColdStartItem> {
        self.resolve_content_attribute(user_id)
    }

    /// Hybrid resolution combining multiple strategies.
    fn resolve_hybrid(
        &self,
        user_id: &str,
        segment: Option<&DemographicSegment>,
    ) -> Vec<ColdStartItem> {
        let mut score_map: HashMap<String, (f64, String)> = HashMap::new();

        // Popularity
        for item in self.resolve_popularity() {
            let weighted = item.score * self.config.popularity_weight;
            let entry = score_map
                .entry(item.item_id)
                .or_insert((0.0, String::new()));
            entry.0 += weighted;
            if entry.1.is_empty() {
                entry.1 = item.reason;
            }
        }

        // Demographic
        for item in self.resolve_demographic(segment) {
            let weighted = item.score * self.config.demographic_weight;
            let entry = score_map
                .entry(item.item_id)
                .or_insert((0.0, String::new()));
            entry.0 += weighted;
        }

        // Content
        for item in self.resolve_content_attribute(user_id) {
            let weighted = item.score * self.config.content_weight;
            let entry = score_map
                .entry(item.item_id)
                .or_insert((0.0, String::new()));
            entry.0 += weighted;
        }

        let mut items: Vec<ColdStartItem> = score_map
            .into_iter()
            .map(|(item_id, (score, reason))| ColdStartItem {
                item_id,
                score,
                strategy: ColdStartStrategy::Hybrid,
                reason,
            })
            .collect();

        items.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        items.truncate(self.config.max_results);
        items
    }

    /// Get total resolution count.
    #[must_use]
    pub fn resolution_count(&self) -> u64 {
        self.resolution_count
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &ColdStartConfig {
        &self.config
    }

    /// Number of popularity entries.
    #[must_use]
    pub fn popularity_count(&self) -> usize {
        self.popularity.len()
    }
}

impl Default for ColdStartResolver {
    fn default() -> Self {
        Self::new(ColdStartConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(id: &str, views: u64, categories: Vec<&str>) -> PopularityEntry {
        PopularityEntry {
            item_id: id.to_string(),
            view_count: views,
            avg_rating: 4.0,
            categories: categories.into_iter().map(String::from).collect(),
            language: "en".to_string(),
        }
    }

    fn setup_resolver() -> ColdStartResolver {
        let mut resolver = ColdStartResolver::default();
        resolver.add_popularity_entry(make_entry("item1", 1000, vec!["action", "thriller"]));
        resolver.add_popularity_entry(make_entry("item2", 800, vec!["comedy"]));
        resolver.add_popularity_entry(make_entry("item3", 600, vec!["drama", "romance"]));
        resolver.add_popularity_entry(make_entry("item4", 400, vec!["action"]));
        resolver.add_popularity_entry(make_entry("item5", 200, vec!["documentary"]));
        resolver
    }

    #[test]
    fn test_cold_start_strategy_display() {
        assert_eq!(ColdStartStrategy::Popularity.to_string(), "Popularity");
        assert_eq!(ColdStartStrategy::Hybrid.to_string(), "Hybrid");
        assert_eq!(
            ColdStartStrategy::DemographicMatch.to_string(),
            "DemographicMatch"
        );
    }

    #[test]
    fn test_demographic_segment_key() {
        let seg = DemographicSegment::new("18-24", "US", "en");
        assert_eq!(seg.segment_key(), "18-24:US:en");
    }

    #[test]
    fn test_cold_start_config_default() {
        let cfg = ColdStartConfig::default();
        assert_eq!(cfg.warm_threshold, 5);
        assert_eq!(cfg.max_results, 20);
        assert_eq!(cfg.strategy, ColdStartStrategy::Hybrid);
    }

    #[test]
    fn test_is_cold_user() {
        let resolver = ColdStartResolver::default();
        assert!(resolver.is_cold_user(0));
        assert!(resolver.is_cold_user(4));
        assert!(!resolver.is_cold_user(5));
        assert!(!resolver.is_cold_user(100));
    }

    #[test]
    fn test_popularity_resolution() {
        let config = ColdStartConfig {
            strategy: ColdStartStrategy::Popularity,
            max_results: 3,
            ..Default::default()
        };
        let mut resolver = ColdStartResolver::new(config);
        resolver.add_popularity_entry(make_entry("a", 500, vec!["action"]));
        resolver.add_popularity_entry(make_entry("b", 100, vec!["comedy"]));
        resolver.add_popularity_entry(make_entry("c", 900, vec!["drama"]));
        resolver.add_popularity_entry(make_entry("d", 300, vec!["horror"]));

        let results = resolver.resolve("user1", None);
        assert_eq!(results.len(), 3);
        // Highest view count first
        assert_eq!(results[0].item_id, "c");
        assert!(results[0].score > results[1].score);
    }

    #[test]
    fn test_demographic_resolution() {
        let config = ColdStartConfig {
            strategy: ColdStartStrategy::DemographicMatch,
            max_results: 5,
            ..Default::default()
        };
        let mut resolver = ColdStartResolver::new(config);
        resolver.add_popularity_entry(make_entry("a", 500, vec!["action"]));
        resolver.add_popularity_entry(make_entry("b", 100, vec!["comedy"]));

        let seg = DemographicSegment::new("25-34", "US", "en");
        resolver.set_demographic_preferences(&seg, vec!["comedy".to_string()]);

        let results = resolver.resolve("user1", Some(&seg));
        assert!(!results.is_empty());
        // First result should be the comedy item
        assert_eq!(results[0].item_id, "b");
    }

    #[test]
    fn test_onboarding_preferences() {
        let config = ColdStartConfig {
            strategy: ColdStartStrategy::OnboardingPreference,
            max_results: 5,
            ..Default::default()
        };
        let mut resolver = ColdStartResolver::new(config);
        resolver.add_popularity_entry(make_entry("a", 500, vec!["action"]));
        resolver.add_popularity_entry(make_entry("b", 100, vec!["drama"]));
        resolver.set_onboarding_preferences("user1", vec!["drama".to_string()]);

        let results = resolver.resolve("user1", None);
        assert!(!results.is_empty());
        assert_eq!(results[0].item_id, "b");
    }

    #[test]
    fn test_hybrid_resolution() {
        let mut resolver = setup_resolver();
        let seg = DemographicSegment::new("18-24", "US", "en");
        resolver.set_demographic_preferences(&seg, vec!["action".to_string()]);
        resolver.set_onboarding_preferences("user1", vec!["action".to_string()]);

        let results = resolver.resolve("user1", Some(&seg));
        assert!(!results.is_empty());
        // Action items should score highest in hybrid
        assert!(results[0].score > 0.0);
    }

    #[test]
    fn test_resolution_count_tracking() {
        let mut resolver = setup_resolver();
        assert_eq!(resolver.resolution_count(), 0);
        resolver.resolve("user1", None);
        resolver.resolve("user2", None);
        assert_eq!(resolver.resolution_count(), 2);
    }

    #[test]
    fn test_popularity_count() {
        let resolver = setup_resolver();
        assert_eq!(resolver.popularity_count(), 5);
    }

    #[test]
    fn test_fallback_to_popularity_no_segment() {
        let config = ColdStartConfig {
            strategy: ColdStartStrategy::DemographicMatch,
            max_results: 3,
            ..Default::default()
        };
        let mut resolver = ColdStartResolver::new(config);
        resolver.add_popularity_entry(make_entry("a", 500, vec!["action"]));
        // No segment provided, should fall back to popularity
        let results = resolver.resolve("user1", None);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_empty_resolver_returns_empty() {
        let config = ColdStartConfig {
            strategy: ColdStartStrategy::Popularity,
            max_results: 10,
            ..Default::default()
        };
        let mut resolver = ColdStartResolver::new(config);
        let results = resolver.resolve("user1", None);
        assert!(results.is_empty());
    }
}
