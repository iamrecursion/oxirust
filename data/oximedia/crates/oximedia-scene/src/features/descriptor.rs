//! Feature descriptors and matching.

use crate::common::Point;
use serde::{Deserialize, Serialize};

/// Feature descriptor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureDescriptor {
    /// Location of the feature.
    pub location: Point,
    /// Descriptor vector.
    pub descriptor: Vec<f32>,
    /// Scale of the feature.
    pub scale: f32,
    /// Orientation of the feature (radians).
    pub orientation: f32,
}

/// Feature match between two descriptors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureMatch {
    /// Index of first descriptor.
    pub index1: usize,
    /// Index of second descriptor.
    pub index2: usize,
    /// Match distance (lower is better).
    pub distance: f32,
}

/// Feature matcher.
pub struct FeatureMatcher {
    distance_threshold: f32,
}

impl FeatureMatcher {
    /// Create a new feature matcher.
    #[must_use]
    pub fn new() -> Self {
        Self {
            distance_threshold: 0.7,
        }
    }

    /// Match features between two sets.
    #[must_use]
    pub fn match_features(
        &self,
        desc1: &[FeatureDescriptor],
        desc2: &[FeatureDescriptor],
    ) -> Vec<FeatureMatch> {
        let mut matches = Vec::new();

        for (i, d1) in desc1.iter().enumerate() {
            let mut best_dist = f32::MAX;
            let mut second_best_dist = f32::MAX;
            let mut best_idx = 0;

            for (j, d2) in desc2.iter().enumerate() {
                let dist = self.compute_distance(&d1.descriptor, &d2.descriptor);

                if dist < best_dist {
                    second_best_dist = best_dist;
                    best_dist = dist;
                    best_idx = j;
                } else if dist < second_best_dist {
                    second_best_dist = dist;
                }
            }

            // Lowe's ratio test
            if best_dist < self.distance_threshold * second_best_dist {
                matches.push(FeatureMatch {
                    index1: i,
                    index2: best_idx,
                    distance: best_dist,
                });
            }
        }

        matches
    }

    fn compute_distance(&self, desc1: &[f32], desc2: &[f32]) -> f32 {
        if desc1.len() != desc2.len() {
            return f32::MAX;
        }

        let mut sum = 0.0;
        for i in 0..desc1.len() {
            let diff = desc1[i] - desc2[i];
            sum += diff * diff;
        }

        sum.sqrt()
    }
}

impl Default for FeatureMatcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_matcher() {
        let matcher = FeatureMatcher::new();

        let desc1 = vec![FeatureDescriptor {
            location: Point::new(10.0, 10.0),
            descriptor: vec![0.5, 0.3, 0.2],
            scale: 1.0,
            orientation: 0.0,
        }];

        let desc2 = vec![FeatureDescriptor {
            location: Point::new(11.0, 11.0),
            descriptor: vec![0.5, 0.3, 0.2],
            scale: 1.0,
            orientation: 0.0,
        }];

        let matches = matcher.match_features(&desc1, &desc2);
        assert!(!matches.is_empty());
    }
}
