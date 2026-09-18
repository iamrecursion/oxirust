//! Time decay functions for trending calculations.

/// Exponential decay function
#[must_use]
pub fn exponential_decay(age: f32, half_life: f32) -> f32 {
    let decay_constant = half_life.ln() / half_life;
    (-decay_constant * age).exp()
}

/// Linear decay function
#[must_use]
pub fn linear_decay(age: f32, max_age: f32) -> f32 {
    if age >= max_age {
        return 0.0;
    }
    (1.0 - age / max_age).max(0.0)
}

/// Logarithmic decay function
#[must_use]
pub fn logarithmic_decay(age: f32) -> f32 {
    1.0 / (1.0 + age).ln()
}

/// Polynomial decay function
#[must_use]
pub fn polynomial_decay(age: f32, degree: u32) -> f32 {
    1.0 / (1.0 + age).powi(degree as i32)
}

/// Gaussian decay function
#[must_use]
pub fn gaussian_decay(age: f32, sigma: f32) -> f32 {
    (-(age.powi(2)) / (2.0 * sigma.powi(2))).exp()
}

/// Step decay function
#[must_use]
pub fn step_decay(age: f32, thresholds: &[(f32, f32)]) -> f32 {
    for (threshold, weight) in thresholds {
        if age < *threshold {
            return *weight;
        }
    }
    0.0
}

/// Hackernews ranking algorithm
#[must_use]
pub fn hackernews_score(points: f32, age_hours: f32, gravity: f32) -> f32 {
    points / (age_hours + 2.0).powf(gravity)
}

/// Reddit hot ranking algorithm
#[must_use]
pub fn reddit_hot_score(upvotes: i32, downvotes: i32, timestamp: i64) -> f32 {
    let score = upvotes - downvotes;
    let order = (score.abs() as f32).log10().max(1.0);
    let sign = if score > 0 {
        1.0
    } else if score < 0 {
        -1.0
    } else {
        0.0
    };

    // Epoch: April 23, 2005
    let epoch = 1_114_293_600;
    let seconds = (timestamp - epoch).max(0) as f32;

    sign * order + seconds / 45000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exponential_decay() {
        let decay = exponential_decay(0.0, 6.0);
        assert!((decay - 1.0).abs() < f32::EPSILON);

        let decay_half = exponential_decay(6.0, 6.0);
        assert!(decay_half < 1.0 && decay_half > 0.0);
    }

    #[test]
    fn test_linear_decay() {
        let decay = linear_decay(5.0, 10.0);
        assert!((decay - 0.5).abs() < f32::EPSILON);

        let decay_zero = linear_decay(10.0, 10.0);
        assert!(decay_zero.abs() < f32::EPSILON);
    }

    #[test]
    fn test_logarithmic_decay() {
        let decay = logarithmic_decay(0.0);
        assert!(decay > 0.0);
    }

    #[test]
    fn test_polynomial_decay() {
        let decay = polynomial_decay(1.0, 2);
        assert!((decay - 0.25).abs() < f32::EPSILON);
    }

    #[test]
    fn test_gaussian_decay() {
        let decay = gaussian_decay(0.0, 1.0);
        assert!((decay - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_step_decay() {
        let thresholds = vec![(5.0, 1.0), (10.0, 0.5), (20.0, 0.1)];
        let decay = step_decay(3.0, &thresholds);
        assert!((decay - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_hackernews_score() {
        let score = hackernews_score(100.0, 1.0, 1.8);
        assert!(score > 0.0);
    }

    #[test]
    fn test_reddit_hot_score() {
        let now = chrono::Utc::now().timestamp();
        let score = reddit_hot_score(100, 10, now);
        assert!(score > 0.0);
    }
}
