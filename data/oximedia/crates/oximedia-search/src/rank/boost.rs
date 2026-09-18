//! Field boosting for search relevance.

use crate::SearchResultItem;
use std::collections::HashMap;

/// Field booster
pub struct FieldBooster {
    boosts: HashMap<String, f32>,
}

impl FieldBooster {
    /// Create a new field booster
    #[must_use]
    pub fn new() -> Self {
        let mut boosts = HashMap::new();
        boosts.insert("title".to_string(), 2.0);
        boosts.insert("description".to_string(), 1.5);
        boosts.insert("keywords".to_string(), 1.3);
        boosts.insert("transcript".to_string(), 1.0);

        Self { boosts }
    }

    /// Set boost for a field
    pub fn set_boost(&mut self, field: &str, boost: f32) {
        self.boosts.insert(field.to_string(), boost);
    }

    /// Apply boosts to results
    pub fn apply(&self, results: &mut [SearchResultItem]) {
        for result in results {
            let mut boost = 1.0;

            for field in &result.matched_fields {
                if let Some(&field_boost) = self.boosts.get(field) {
                    boost *= field_boost;
                }
            }

            result.score *= boost;
        }
    }

    /// Get boost for a field
    #[must_use]
    pub fn get_boost(&self, field: &str) -> f32 {
        self.boosts.get(field).copied().unwrap_or(1.0)
    }
}

impl Default for FieldBooster {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn test_field_boost() {
        let mut booster = FieldBooster::new();
        assert_eq!(booster.get_boost("title"), 2.0);

        booster.set_boost("custom", 3.0);
        assert_eq!(booster.get_boost("custom"), 3.0);
    }

    #[test]
    fn test_apply_boost() {
        let booster = FieldBooster::new();
        let mut results = vec![SearchResultItem {
            asset_id: Uuid::new_v4(),
            score: 1.0,
            title: Some("Test".to_string()),
            description: None,
            file_path: String::new(),
            mime_type: None,
            duration_ms: None,
            created_at: 0,
            modified_at: None,
            file_size: None,
            matched_fields: vec!["title".to_string()],
            thumbnail_url: None,
        }];

        booster.apply(&mut results);
        assert_eq!(results[0].score, 2.0); // Title boost = 2.0
    }
}
