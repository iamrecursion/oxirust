//! User segment-based recommendations.
//!
//! Clusters users into behavioural segments based on profile features
//! (engagement, preferred categories, session patterns) and produces
//! segment-level aggregate preferences that drive recommendations for
//! segment members — especially useful for users with sparse histories.

use crate::error::{RecommendError, RecommendResult};
use crate::profile::user::UserProfile;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Segment definition
// ---------------------------------------------------------------------------

/// Identifier for a user segment.
pub type SegmentId = usize;

/// A user segment produced by clustering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserSegment {
    /// Unique segment identifier.
    pub id: SegmentId,
    /// Human-readable label (e.g. "power-bingers", "casual-morning").
    pub label: String,
    /// User IDs belonging to this segment.
    pub members: Vec<Uuid>,
    /// Centroid feature vector.
    pub centroid: Vec<f32>,
    /// Aggregate category affinity across members (category -> avg weight).
    pub category_affinities: HashMap<String, f32>,
    /// Average engagement level across segment members.
    pub avg_engagement: f32,
    /// Average session duration in minutes.
    pub avg_session_duration_min: u32,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the segment recommender.
#[derive(Debug, Clone)]
pub struct SegmentConfig {
    /// Number of segments (k for k-means).
    pub num_segments: usize,
    /// Maximum k-means iterations.
    pub max_iterations: usize,
    /// Weight of segment preferences vs individual preferences (0.0-1.0).
    /// 0.0 = pure individual, 1.0 = pure segment.
    pub segment_weight: f32,
    /// Minimum members required for a segment to be valid.
    pub min_segment_size: usize,
}

impl Default for SegmentConfig {
    fn default() -> Self {
        Self {
            num_segments: 5,
            max_iterations: 20,
            segment_weight: 0.4,
            min_segment_size: 1,
        }
    }
}

// ---------------------------------------------------------------------------
// Segment recommender
// ---------------------------------------------------------------------------

/// Builds user segments and provides segment-aware recommendation helpers.
pub struct SegmentRecommender {
    /// Configuration.
    config: SegmentConfig,
    /// Computed segments.
    segments: Vec<UserSegment>,
    /// User -> segment mapping for O(1) lookup.
    user_segment_map: HashMap<Uuid, SegmentId>,
}

impl SegmentRecommender {
    /// Create a new segment recommender.
    #[must_use]
    pub fn new(config: SegmentConfig) -> Self {
        Self {
            config,
            segments: Vec::new(),
            user_segment_map: HashMap::new(),
        }
    }

    /// Build segments from a set of user profiles.
    ///
    /// # Errors
    ///
    /// Returns an error when there are fewer profiles than segments.
    pub fn build_segments(&mut self, profiles: &HashMap<Uuid, UserProfile>) -> RecommendResult<()> {
        if profiles.len() < self.config.num_segments {
            return Err(RecommendError::insufficient_data(format!(
                "Need at least {} profiles, got {}",
                self.config.num_segments,
                profiles.len()
            )));
        }

        let (user_ids, vectors) = Self::profiles_to_vectors(profiles);
        let assignments = self.kmeans(
            &vectors,
            self.config.num_segments,
            self.config.max_iterations,
        );

        self.segments.clear();
        self.user_segment_map.clear();

        for seg_id in 0..self.config.num_segments {
            let mut members = Vec::new();
            let dim = if vectors.is_empty() {
                0
            } else {
                vectors[0].len()
            };
            let mut centroid = vec![0.0f32; dim];
            let mut cat_sums: HashMap<String, f32> = HashMap::new();
            let mut eng_sum = 0.0f32;
            let mut sess_sum = 0u64;
            let mut count = 0u32;

            for (idx, &cluster) in assignments.iter().enumerate() {
                if cluster != seg_id {
                    continue;
                }
                let uid = user_ids[idx];
                members.push(uid);
                self.user_segment_map.insert(uid, seg_id);

                for (j, val) in vectors[idx].iter().enumerate() {
                    centroid[j] += val;
                }

                if let Some(profile) = profiles.get(&uid) {
                    for (cat, &w) in &profile.preferred_categories {
                        *cat_sums.entry(cat.clone()).or_insert(0.0) += w;
                    }
                    eng_sum += profile.engagement_level;
                    sess_sum += u64::from(profile.viewing_patterns.avg_session_duration_min);
                }
                count += 1;
            }

            if count > 0 {
                for v in &mut centroid {
                    *v /= count as f32;
                }
                for w in cat_sums.values_mut() {
                    *w /= count as f32;
                }
            }

            let avg_engagement = if count > 0 {
                eng_sum / count as f32
            } else {
                0.0
            };
            let avg_session_duration_min = if count > 0 {
                (sess_sum / u64::from(count)) as u32
            } else {
                0
            };

            let label = format!("segment_{seg_id}");

            self.segments.push(UserSegment {
                id: seg_id,
                label,
                members,
                centroid,
                category_affinities: cat_sums,
                avg_engagement,
                avg_session_duration_min,
            });
        }

        Ok(())
    }

    /// Get the segment a user belongs to.
    #[must_use]
    pub fn user_segment(&self, user_id: Uuid) -> Option<&UserSegment> {
        let seg_id = self.user_segment_map.get(&user_id)?;
        self.segments.get(*seg_id)
    }

    /// Get segment-level category recommendations for a user.
    ///
    /// Returns a sorted list of (category, blended_weight) pairs combining
    /// the individual user's preferences with the segment aggregate.
    ///
    /// # Errors
    ///
    /// Returns an error if the user has no segment assignment.
    pub fn segment_category_recommendations(
        &self,
        user_id: Uuid,
        profile: &UserProfile,
        limit: usize,
    ) -> RecommendResult<Vec<(String, f32)>> {
        let segment = self.user_segment(user_id).ok_or_else(|| {
            RecommendError::PersonalizationError(format!(
                "User {user_id} not assigned to any segment"
            ))
        })?;

        let sw = self.config.segment_weight;
        let uw = 1.0 - sw;

        let mut blended: HashMap<String, f32> = HashMap::new();

        // Individual preferences
        for (cat, &w) in &profile.preferred_categories {
            *blended.entry(cat.clone()).or_insert(0.0) += uw * w;
        }

        // Segment preferences
        for (cat, &w) in &segment.category_affinities {
            *blended.entry(cat.clone()).or_insert(0.0) += sw * w;
        }

        let mut result: Vec<(String, f32)> = blended.into_iter().collect();
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result.truncate(limit);
        Ok(result)
    }

    /// Assign a new user to the nearest existing segment without rebuilding.
    ///
    /// # Errors
    ///
    /// Returns an error when no segments have been built yet.
    pub fn assign_user(
        &mut self,
        user_id: Uuid,
        profile: &UserProfile,
    ) -> RecommendResult<SegmentId> {
        if self.segments.is_empty() {
            return Err(RecommendError::insufficient_data("No segments built yet"));
        }

        let vec = Self::profile_to_vector(profile);
        let mut best_seg = 0;
        let mut best_dist = f32::INFINITY;

        for seg in &self.segments {
            let dist = Self::euclidean_distance(&vec, &seg.centroid);
            if dist < best_dist {
                best_dist = dist;
                best_seg = seg.id;
            }
        }

        self.user_segment_map.insert(user_id, best_seg);
        if let Some(seg) = self.segments.get_mut(best_seg) {
            if !seg.members.contains(&user_id) {
                seg.members.push(user_id);
            }
        }

        Ok(best_seg)
    }

    /// Get all computed segments.
    #[must_use]
    pub fn segments(&self) -> &[UserSegment] {
        &self.segments
    }

    /// Number of segments.
    #[must_use]
    pub fn num_segments(&self) -> usize {
        self.segments.len()
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    fn profiles_to_vectors(profiles: &HashMap<Uuid, UserProfile>) -> (Vec<Uuid>, Vec<Vec<f32>>) {
        let mut ids = Vec::with_capacity(profiles.len());
        let mut vecs = Vec::with_capacity(profiles.len());
        for (uid, prof) in profiles {
            ids.push(*uid);
            vecs.push(Self::profile_to_vector(prof));
        }
        (ids, vecs)
    }

    fn profile_to_vector(profile: &UserProfile) -> Vec<f32> {
        vec![
            profile.engagement_level,
            profile.avg_completion_rate,
            profile.avg_watch_duration_ms as f32 / 3_600_000.0,
            profile.viewing_patterns.binge_tendency,
            profile.viewing_patterns.avg_session_duration_min as f32 / 120.0,
        ]
    }

    fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    fn kmeans(&self, vectors: &[Vec<f32>], k: usize, max_iter: usize) -> Vec<usize> {
        if vectors.is_empty() {
            return Vec::new();
        }
        let n = vectors.len();
        let dim = vectors[0].len();

        // Initialize assignments round-robin
        let mut assignments: Vec<usize> = (0..n).map(|i| i % k).collect();
        let mut centroids = vec![vec![0.0f32; dim]; k];

        for _ in 0..max_iter {
            // Compute centroids
            let mut counts = vec![0u32; k];
            for c in &mut centroids {
                c.fill(0.0);
            }
            for (idx, cluster) in assignments.iter().enumerate() {
                for (j, &val) in vectors[idx].iter().enumerate() {
                    centroids[*cluster][j] += val;
                }
                counts[*cluster] += 1;
            }
            for (c, cnt) in centroids.iter_mut().zip(counts.iter()) {
                if *cnt > 0 {
                    for v in c.iter_mut() {
                        *v /= *cnt as f32;
                    }
                }
            }

            // Reassign
            let mut changed = false;
            for (idx, vec) in vectors.iter().enumerate() {
                let mut best = 0;
                let mut best_d = f32::INFINITY;
                for (c, centroid) in centroids.iter().enumerate() {
                    let d = Self::euclidean_distance(vec, centroid);
                    if d < best_d {
                        best_d = d;
                        best = c;
                    }
                }
                if best != assignments[idx] {
                    assignments[idx] = best;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        assignments
    }
}

impl Default for SegmentRecommender {
    fn default() -> Self {
        Self::new(SegmentConfig::default())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_profile(
        uid: Uuid,
        engagement: f32,
        binge: f32,
        categories: &[(&str, f32)],
    ) -> UserProfile {
        let mut profile = UserProfile::new(uid);
        profile.engagement_level = engagement;
        profile.avg_completion_rate = 0.7;
        profile.viewing_patterns.binge_tendency = binge;
        profile.viewing_patterns.avg_session_duration_min = 45;
        for &(cat, w) in categories {
            profile.preferred_categories.insert(cat.to_string(), w);
        }
        profile
    }

    fn build_test_profiles() -> HashMap<Uuid, UserProfile> {
        let mut profiles = HashMap::new();
        // Cluster A: high engagement action fans
        for _ in 0..5 {
            let uid = Uuid::new_v4();
            profiles.insert(
                uid,
                make_profile(uid, 0.9, 0.8, &[("Action", 5.0), ("Thriller", 3.0)]),
            );
        }
        // Cluster B: low engagement comedy fans
        for _ in 0..5 {
            let uid = Uuid::new_v4();
            profiles.insert(
                uid,
                make_profile(uid, 0.2, 0.1, &[("Comedy", 5.0), ("Drama", 2.0)]),
            );
        }
        profiles
    }

    #[test]
    fn test_segment_config_default() {
        let config = SegmentConfig::default();
        assert_eq!(config.num_segments, 5);
        assert!((config.segment_weight - 0.4).abs() < f32::EPSILON);
    }

    #[test]
    fn test_build_segments_insufficient_profiles() {
        let mut recommender = SegmentRecommender::new(SegmentConfig {
            num_segments: 10,
            ..Default::default()
        });
        let profiles: HashMap<Uuid, UserProfile> = HashMap::new();
        let result = recommender.build_segments(&profiles);
        assert!(result.is_err());
    }

    #[test]
    fn test_build_segments_success() {
        let profiles = build_test_profiles();
        let mut recommender = SegmentRecommender::new(SegmentConfig {
            num_segments: 2,
            max_iterations: 30,
            ..Default::default()
        });
        let result = recommender.build_segments(&profiles);
        assert!(result.is_ok());
        assert_eq!(recommender.num_segments(), 2);

        // Every user should be assigned
        for uid in profiles.keys() {
            assert!(recommender.user_segment(*uid).is_some());
        }
    }

    #[test]
    fn test_user_segment_lookup() {
        let profiles = build_test_profiles();
        let mut recommender = SegmentRecommender::new(SegmentConfig {
            num_segments: 2,
            ..Default::default()
        });
        recommender
            .build_segments(&profiles)
            .expect("build should succeed");

        let uid = *profiles.keys().next().expect("should have a user");
        let seg = recommender.user_segment(uid);
        assert!(seg.is_some());
        assert!(!seg.expect("segment exists").members.is_empty());
    }

    #[test]
    fn test_segment_category_recommendations() {
        let profiles = build_test_profiles();
        let mut recommender = SegmentRecommender::new(SegmentConfig {
            num_segments: 2,
            segment_weight: 0.5,
            ..Default::default()
        });
        recommender
            .build_segments(&profiles)
            .expect("build should succeed");

        let (&uid, profile) = profiles.iter().next().expect("should have a user");
        let recs = recommender
            .segment_category_recommendations(uid, profile, 5)
            .expect("should succeed");
        assert!(!recs.is_empty());
        // Weights should be positive
        for (_, w) in &recs {
            assert!(*w > 0.0);
        }
    }

    #[test]
    fn test_segment_category_recommendations_unknown_user() {
        let recommender = SegmentRecommender::default();
        let profile = UserProfile::new(Uuid::new_v4());
        let result = recommender.segment_category_recommendations(Uuid::new_v4(), &profile, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_assign_new_user() {
        let profiles = build_test_profiles();
        let mut recommender = SegmentRecommender::new(SegmentConfig {
            num_segments: 2,
            ..Default::default()
        });
        recommender
            .build_segments(&profiles)
            .expect("build should succeed");

        let new_uid = Uuid::new_v4();
        let new_profile = make_profile(new_uid, 0.85, 0.9, &[("Action", 4.0)]);
        let seg_id = recommender
            .assign_user(new_uid, &new_profile)
            .expect("should assign");
        assert!(seg_id < 2);

        // Should now be findable
        assert!(recommender.user_segment(new_uid).is_some());
    }

    #[test]
    fn test_assign_user_no_segments_errors() {
        let mut recommender = SegmentRecommender::default();
        let profile = UserProfile::new(Uuid::new_v4());
        let result = recommender.assign_user(Uuid::new_v4(), &profile);
        assert!(result.is_err());
    }

    #[test]
    fn test_segment_has_category_affinities() {
        let profiles = build_test_profiles();
        let mut recommender = SegmentRecommender::new(SegmentConfig {
            num_segments: 2,
            ..Default::default()
        });
        recommender
            .build_segments(&profiles)
            .expect("build should succeed");

        let total_cats: usize = recommender
            .segments()
            .iter()
            .map(|s| s.category_affinities.len())
            .sum();
        assert!(total_cats > 0, "segments should have category affinities");
    }

    #[test]
    fn test_segments_have_valid_engagement() {
        let profiles = build_test_profiles();
        let mut recommender = SegmentRecommender::new(SegmentConfig {
            num_segments: 2,
            ..Default::default()
        });
        recommender
            .build_segments(&profiles)
            .expect("build should succeed");

        for seg in recommender.segments() {
            if !seg.members.is_empty() {
                assert!(
                    seg.avg_engagement >= 0.0 && seg.avg_engagement <= 1.0,
                    "engagement should be in [0,1], got {}",
                    seg.avg_engagement
                );
            }
        }
    }
}
