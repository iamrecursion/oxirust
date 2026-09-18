//! User clustering for collaborative filtering.

use super::user::UserProfile;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// User cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserCluster {
    /// Cluster ID
    pub cluster_id: usize,
    /// Users in this cluster
    pub users: Vec<Uuid>,
    /// Cluster centroid (feature vector)
    pub centroid: Vec<f32>,
    /// Cluster characteristics
    pub characteristics: ClusterCharacteristics,
}

/// Cluster characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterCharacteristics {
    /// Average engagement level
    pub avg_engagement: f32,
    /// Common categories
    pub common_categories: Vec<String>,
    /// Average session duration
    pub avg_session_duration: u32,
}

impl Default for ClusterCharacteristics {
    fn default() -> Self {
        Self {
            avg_engagement: 0.5,
            common_categories: Vec::new(),
            avg_session_duration: 30,
        }
    }
}

/// K-means clustering for user profiles
pub struct UserClusterer {
    /// Number of clusters
    num_clusters: usize,
    /// Clusters
    clusters: Vec<UserCluster>,
}

impl UserClusterer {
    /// Create a new user clusterer
    #[must_use]
    pub fn new(num_clusters: usize) -> Self {
        Self {
            num_clusters,
            clusters: Vec::new(),
        }
    }

    /// Cluster users using k-means
    pub fn cluster_users(&mut self, profiles: &HashMap<Uuid, UserProfile>) {
        if profiles.is_empty() || self.num_clusters == 0 {
            return;
        }

        // Convert profiles to feature vectors
        let (user_ids, feature_vectors) = self.profiles_to_vectors(profiles);

        // Run k-means
        let assignments = self.kmeans(&feature_vectors, self.num_clusters, 10);

        // Build clusters
        self.build_clusters(&user_ids, &feature_vectors, &assignments, profiles);
    }

    /// Convert profiles to feature vectors
    fn profiles_to_vectors(
        &self,
        profiles: &HashMap<Uuid, UserProfile>,
    ) -> (Vec<Uuid>, Vec<Vec<f32>>) {
        let mut user_ids = Vec::new();
        let mut vectors = Vec::new();

        for (user_id, profile) in profiles {
            user_ids.push(*user_id);
            vectors.push(self.profile_to_vector(profile));
        }

        (user_ids, vectors)
    }

    /// Convert profile to feature vector
    fn profile_to_vector(&self, profile: &UserProfile) -> Vec<f32> {
        let mut features = Vec::new();

        // Add engagement level
        features.push(profile.engagement_level);

        // Add completion rate
        features.push(profile.avg_completion_rate);

        // Add normalized watch duration (in hours)
        features.push(profile.avg_watch_duration_ms as f32 / 3_600_000.0);

        // Add binge tendency
        features.push(profile.viewing_patterns.binge_tendency);

        features
    }

    /// K-means clustering algorithm
    fn kmeans(&self, vectors: &[Vec<f32>], k: usize, max_iterations: usize) -> Vec<usize> {
        if vectors.is_empty() {
            return Vec::new();
        }

        let n = vectors.len();
        let dim = vectors[0].len();

        // Initialize assignments randomly
        let mut assignments = vec![0; n];
        for (i, assignment) in assignments.iter_mut().enumerate() {
            *assignment = i % k;
        }

        // Initialize centroids
        let mut centroids = vec![vec![0.0; dim]; k];

        for _ in 0..max_iterations {
            // Update centroids
            self.update_centroids(vectors, &assignments, &mut centroids, k);

            // Assign points to nearest centroids
            let mut changed = false;
            for (i, vector) in vectors.iter().enumerate() {
                let new_cluster = self.find_nearest_centroid(vector, &centroids);
                if new_cluster != assignments[i] {
                    assignments[i] = new_cluster;
                    changed = true;
                }
            }

            if !changed {
                break;
            }
        }

        assignments
    }

    /// Update cluster centroids
    fn update_centroids(
        &self,
        vectors: &[Vec<f32>],
        assignments: &[usize],
        centroids: &mut [Vec<f32>],
        k: usize,
    ) {
        let _dim = vectors[0].len();
        let mut counts = vec![0; k];

        // Reset centroids
        for centroid in centroids.iter_mut() {
            centroid.fill(0.0);
        }

        // Sum vectors in each cluster
        for (vector, &cluster) in vectors.iter().zip(assignments.iter()) {
            for (i, &value) in vector.iter().enumerate() {
                centroids[cluster][i] += value;
            }
            counts[cluster] += 1;
        }

        // Average
        for (cluster, centroid) in centroids.iter_mut().enumerate() {
            if counts[cluster] > 0 {
                for value in centroid.iter_mut() {
                    *value /= counts[cluster] as f32;
                }
            }
        }
    }

    /// Find nearest centroid for a vector
    fn find_nearest_centroid(&self, vector: &[f32], centroids: &[Vec<f32>]) -> usize {
        let mut min_dist = f32::INFINITY;
        let mut nearest = 0;

        for (i, centroid) in centroids.iter().enumerate() {
            let dist = self.euclidean_distance(vector, centroid);
            if dist < min_dist {
                min_dist = dist;
                nearest = i;
            }
        }

        nearest
    }

    /// Calculate Euclidean distance
    fn euclidean_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Build cluster structures
    fn build_clusters(
        &mut self,
        user_ids: &[Uuid],
        vectors: &[Vec<f32>],
        assignments: &[usize],
        profiles: &HashMap<Uuid, UserProfile>,
    ) {
        self.clusters.clear();

        for cluster_id in 0..self.num_clusters {
            let mut cluster_users = Vec::new();
            let mut centroid_sum = vec![0.0; vectors[0].len()];
            let mut count = 0;

            for (i, &assignment) in assignments.iter().enumerate() {
                if assignment == cluster_id {
                    cluster_users.push(user_ids[i]);
                    for (j, &value) in vectors[i].iter().enumerate() {
                        centroid_sum[j] += value;
                    }
                    count += 1;
                }
            }

            if count > 0 {
                for value in &mut centroid_sum {
                    *value /= count as f32;
                }
            }

            let characteristics = self.calculate_characteristics(&cluster_users, profiles);

            self.clusters.push(UserCluster {
                cluster_id,
                users: cluster_users,
                centroid: centroid_sum,
                characteristics,
            });
        }
    }

    /// Calculate cluster characteristics
    fn calculate_characteristics(
        &self,
        users: &[Uuid],
        profiles: &HashMap<Uuid, UserProfile>,
    ) -> ClusterCharacteristics {
        if users.is_empty() {
            return ClusterCharacteristics::default();
        }

        let mut total_engagement = 0.0;
        let mut total_session = 0;
        let mut category_counts: HashMap<String, usize> = HashMap::new();

        for user_id in users {
            if let Some(profile) = profiles.get(user_id) {
                total_engagement += profile.engagement_level;
                total_session += profile.viewing_patterns.avg_session_duration_min;

                for category in profile.preferred_categories.keys() {
                    *category_counts.entry(category.clone()).or_insert(0) += 1;
                }
            }
        }

        let avg_engagement = total_engagement / users.len() as f32;
        let avg_session_duration = total_session / users.len() as u32;

        let mut common_categories: Vec<(String, usize)> = category_counts.into_iter().collect();
        common_categories.sort_by(|a, b| b.1.cmp(&a.1));
        let common_categories = common_categories
            .into_iter()
            .take(5)
            .map(|(cat, _)| cat)
            .collect();

        ClusterCharacteristics {
            avg_engagement,
            common_categories,
            avg_session_duration,
        }
    }

    /// Get cluster for a user
    #[must_use]
    pub fn get_user_cluster(&self, user_id: Uuid) -> Option<&UserCluster> {
        self.clusters
            .iter()
            .find(|cluster| cluster.users.contains(&user_id))
    }

    /// Get all clusters
    #[must_use]
    pub fn get_clusters(&self) -> &[UserCluster] {
        &self.clusters
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_clusterer_creation() {
        let clusterer = UserClusterer::new(5);
        assert_eq!(clusterer.num_clusters, 5);
        assert_eq!(clusterer.clusters.len(), 0);
    }

    #[test]
    fn test_euclidean_distance() {
        let clusterer = UserClusterer::new(3);
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let dist = clusterer.euclidean_distance(&a, &b);
        assert!((dist - 5.196_152).abs() < 0.001);
    }

    #[test]
    fn test_profile_to_vector() {
        let clusterer = UserClusterer::new(3);
        let profile = UserProfile::new(Uuid::new_v4());
        let vector = clusterer.profile_to_vector(&profile);
        assert!(!vector.is_empty());
    }
}
