//! Trending content detection.

use crate::error::RecommendResult;
use crate::{ContentMetadata, Recommendation, RecommendationReason};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Trending content item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendingItem {
    /// Content ID
    pub content_id: Uuid,
    /// Trending score
    pub score: f32,
    /// View velocity (views per hour)
    pub view_velocity: f32,
    /// Engagement rate
    pub engagement_rate: f32,
    /// Time window analyzed
    pub time_window_hours: u32,
}

/// Trending detector
pub struct TrendingDetector {
    /// Trending items
    trending_items: HashMap<Uuid, TrendingItem>,
    /// Content metadata
    content_metadata: HashMap<Uuid, ContentMetadata>,
    /// View counts by time window
    view_counts: HashMap<Uuid, Vec<ViewCount>>,
}

/// View count for a time period
#[derive(Debug, Clone)]
struct ViewCount {
    timestamp: i64,
    count: u32,
}

impl TrendingDetector {
    /// Create a new trending detector
    #[must_use]
    pub fn new() -> Self {
        Self {
            trending_items: HashMap::new(),
            content_metadata: HashMap::new(),
            view_counts: HashMap::new(),
        }
    }

    /// Record a view for content
    pub fn record_view(&mut self, content_id: Uuid) {
        let now = chrono::Utc::now().timestamp();
        let counts = self.view_counts.entry(content_id).or_default();

        // Add new count or update existing
        if let Some(last) = counts.last_mut() {
            let time_diff = now - last.timestamp;
            if time_diff < 3600 {
                // Same hour
                last.count += 1;
                return;
            }
        }

        counts.push(ViewCount {
            timestamp: now,
            count: 1,
        });

        // Trim old counts (keep last 24 hours)
        counts.retain(|vc| now - vc.timestamp < 86400);
    }

    /// Update trending scores
    ///
    /// # Errors
    ///
    /// Returns an error if update fails
    pub fn update_scores(&mut self) -> RecommendResult<()> {
        let now = chrono::Utc::now().timestamp();

        for (content_id, counts) in &self.view_counts {
            // Calculate trending score
            let score = self.calculate_trending_score(counts, now);

            // Calculate view velocity
            let velocity = self.calculate_view_velocity(counts, now, 6);

            // Calculate engagement rate (simplified)
            let engagement_rate = self.calculate_engagement_rate(*content_id);

            let trending_item = TrendingItem {
                content_id: *content_id,
                score,
                view_velocity: velocity,
                engagement_rate,
                time_window_hours: 24,
            };

            self.trending_items.insert(*content_id, trending_item);
        }

        Ok(())
    }

    /// Calculate trending score
    fn calculate_trending_score(&self, counts: &[ViewCount], now: i64) -> f32 {
        if counts.is_empty() {
            return 0.0;
        }

        let mut score = 0.0;

        // Recent views weighted more heavily
        for count in counts {
            let age_hours = (now - count.timestamp) as f32 / 3600.0;
            let decay = super::decay::exponential_decay(age_hours, 6.0);
            score += count.count as f32 * decay;
        }

        score
    }

    /// Calculate view velocity (views per hour)
    fn calculate_view_velocity(&self, counts: &[ViewCount], now: i64, window_hours: i64) -> f32 {
        let window_start = now - (window_hours * 3600);

        let recent_views: u32 = counts
            .iter()
            .filter(|vc| vc.timestamp >= window_start)
            .map(|vc| vc.count)
            .sum();

        recent_views as f32 / window_hours as f32
    }

    /// Calculate engagement rate
    fn calculate_engagement_rate(&self, _content_id: Uuid) -> f32 {
        // Simplified - in real implementation would calculate from interactions
        0.5
    }

    /// Get trending content
    ///
    /// # Errors
    ///
    /// Returns an error if retrieval fails
    pub fn get_trending(&self, limit: usize) -> RecommendResult<Vec<Recommendation>> {
        let mut trending: Vec<&TrendingItem> = self.trending_items.values().collect();

        trending.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let recommendations: Vec<Recommendation> = trending
            .into_iter()
            .take(limit)
            .enumerate()
            .filter_map(|(idx, item)| {
                self.content_metadata
                    .get(&item.content_id)
                    .map(|metadata| Recommendation {
                        content_id: item.content_id,
                        score: item.score,
                        rank: idx + 1,
                        reasons: vec![RecommendationReason::Trending {
                            trending_score: item.score,
                        }],
                        metadata: metadata.clone(),
                        explanation: None,
                    })
            })
            .collect();

        Ok(recommendations)
    }

    /// Get trending items raw
    #[must_use]
    pub fn get_trending_items(&self, limit: usize) -> Vec<TrendingItem> {
        let mut items: Vec<TrendingItem> = self.trending_items.values().cloned().collect();
        items.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        items.truncate(limit);
        items
    }

    /// Check if content is trending
    #[must_use]
    pub fn is_trending(&self, content_id: Uuid, threshold: f32) -> bool {
        self.trending_items
            .get(&content_id)
            .is_some_and(|item| item.score >= threshold)
    }

    /// Add content metadata
    pub fn add_content(&mut self, content_id: Uuid, metadata: ContentMetadata) {
        self.content_metadata.insert(content_id, metadata);
    }
}

impl Default for TrendingDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trending_detector_creation() {
        let detector = TrendingDetector::new();
        assert_eq!(detector.trending_items.len(), 0);
    }

    #[test]
    fn test_record_view() {
        let mut detector = TrendingDetector::new();
        let content_id = Uuid::new_v4();

        detector.record_view(content_id);
        assert!(detector.view_counts.contains_key(&content_id));
    }

    #[test]
    fn test_update_scores() {
        let mut detector = TrendingDetector::new();
        let content_id = Uuid::new_v4();

        detector.record_view(content_id);
        detector.record_view(content_id);

        let result = detector.update_scores();
        assert!(result.is_ok());
        assert!(detector.trending_items.contains_key(&content_id));
    }

    #[test]
    fn test_is_trending() {
        let mut detector = TrendingDetector::new();
        let content_id = Uuid::new_v4();

        detector.record_view(content_id);
        detector.update_scores().expect("should succeed in test");

        // Low threshold should pass
        assert!(detector.is_trending(content_id, 0.0));
    }
}
