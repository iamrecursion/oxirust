//! Rating normalization.

/// Rating normalizer
pub struct RatingNormalizer {
    /// Global mean rating
    global_mean: f32,
}

impl RatingNormalizer {
    /// Create a new rating normalizer
    #[must_use]
    pub fn new(global_mean: f32) -> Self {
        Self { global_mean }
    }

    /// Normalize a rating
    #[must_use]
    pub fn normalize(&self, rating: f32, user_mean: f32, item_mean: f32) -> f32 {
        rating - user_mean - item_mean + self.global_mean
    }

    /// Denormalize a rating
    #[must_use]
    pub fn denormalize(&self, normalized: f32, user_mean: f32, item_mean: f32) -> f32 {
        (normalized + user_mean + item_mean - self.global_mean).clamp(0.0, 5.0)
    }

    /// Z-score normalization
    #[must_use]
    pub fn z_score(&self, rating: f32, mean: f32, std_dev: f32) -> f32 {
        if std_dev < f32::EPSILON {
            return 0.0;
        }
        (rating - mean) / std_dev
    }

    /// Min-max normalization
    #[must_use]
    pub fn min_max(&self, rating: f32, min: f32, max: f32) -> f32 {
        if (max - min).abs() < f32::EPSILON {
            return 0.5;
        }
        ((rating - min) / (max - min)).clamp(0.0, 1.0)
    }
}

impl Default for RatingNormalizer {
    fn default() -> Self {
        Self::new(3.0)
    }
}

/// Rating scale converter
pub struct RatingScaleConverter;

impl RatingScaleConverter {
    /// Convert from 5-star to 10-point scale
    #[must_use]
    pub fn five_to_ten(rating: f32) -> f32 {
        (rating * 2.0).clamp(0.0, 10.0)
    }

    /// Convert from 10-point to 5-star scale
    #[must_use]
    pub fn ten_to_five(rating: f32) -> f32 {
        (rating / 2.0).clamp(0.0, 5.0)
    }

    /// Convert from percentage to 5-star scale
    #[must_use]
    pub fn percentage_to_five(percentage: f32) -> f32 {
        (percentage / 20.0).clamp(0.0, 5.0)
    }

    /// Convert from 5-star to percentage
    #[must_use]
    pub fn five_to_percentage(rating: f32) -> f32 {
        (rating * 20.0).clamp(0.0, 100.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rating_normalizer() {
        let normalizer = RatingNormalizer::new(3.0);
        let normalized = normalizer.normalize(4.0, 3.5, 3.2);
        assert!(normalized < 4.0);
    }

    #[test]
    fn test_z_score() {
        let normalizer = RatingNormalizer::new(3.0);
        let z = normalizer.z_score(4.0, 3.0, 1.0);
        assert!((z - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_min_max() {
        let normalizer = RatingNormalizer::new(3.0);
        let normalized = normalizer.min_max(3.0, 1.0, 5.0);
        assert!((normalized - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_scale_conversion() {
        let ten_point = RatingScaleConverter::five_to_ten(4.0);
        assert!((ten_point - 8.0).abs() < f32::EPSILON);

        let five_star = RatingScaleConverter::ten_to_five(8.0);
        assert!((five_star - 4.0).abs() < f32::EPSILON);
    }
}
