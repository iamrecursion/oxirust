//! Timestamp validation.
//!
//! This module provides functions to validate timestamps in media files.

use super::fix::TimestampIssue;

/// Validation result.
#[derive(Debug)]
pub struct ValidationResult {
    /// Whether timestamps are valid.
    pub valid: bool,
    /// List of issues found.
    pub issues: Vec<(usize, TimestampIssue)>,
    /// Statistics about timestamps.
    pub stats: TimestampStats,
}

/// Timestamp statistics.
#[derive(Debug)]
pub struct TimestampStats {
    /// Minimum timestamp.
    pub min: i64,
    /// Maximum timestamp.
    pub max: i64,
    /// Average frame duration.
    pub avg_duration: i64,
    /// Standard deviation of frame duration.
    pub std_deviation: f64,
    /// Number of gaps detected.
    pub gaps: usize,
}

/// Validate a sequence of timestamps.
pub fn validate_timestamps(timestamps: &[i64]) -> ValidationResult {
    let mut issues = Vec::new();

    if timestamps.is_empty() {
        return ValidationResult {
            valid: false,
            issues,
            stats: TimestampStats {
                min: 0,
                max: 0,
                avg_duration: 0,
                std_deviation: 0.0,
                gaps: 0,
            },
        };
    }

    // Check for negative timestamps
    for (i, &ts) in timestamps.iter().enumerate() {
        if ts < 0 {
            issues.push((i, TimestampIssue::Negative));
        }
    }

    // Check for out-of-order timestamps
    for i in 1..timestamps.len() {
        if timestamps[i] < timestamps[i - 1] {
            issues.push((i, TimestampIssue::OutOfOrder));
        }
    }

    // Check for duplicates
    for i in 1..timestamps.len() {
        if timestamps[i] == timestamps[i - 1] {
            issues.push((i, TimestampIssue::Duplicate));
        }
    }

    // Calculate statistics
    let stats = calculate_stats(timestamps);

    // Check for gaps
    let avg_delta = stats.avg_duration;
    for i in 1..timestamps.len() {
        let delta = timestamps[i] - timestamps[i - 1];
        if delta > avg_delta * 5 {
            issues.push((i, TimestampIssue::Gap));
        }
    }

    ValidationResult {
        valid: issues.is_empty(),
        issues,
        stats,
    }
}

/// Calculate timestamp statistics.
fn calculate_stats(timestamps: &[i64]) -> TimestampStats {
    let min = *timestamps.iter().min().unwrap_or(&0);
    let max = *timestamps.iter().max().unwrap_or(&0);

    if timestamps.len() < 2 {
        return TimestampStats {
            min,
            max,
            avg_duration: 0,
            std_deviation: 0.0,
            gaps: 0,
        };
    }

    // Calculate deltas
    let mut deltas = Vec::new();
    for i in 1..timestamps.len() {
        let delta = timestamps[i] - timestamps[i - 1];
        if delta > 0 {
            deltas.push(delta);
        }
    }

    if deltas.is_empty() {
        return TimestampStats {
            min,
            max,
            avg_duration: 0,
            std_deviation: 0.0,
            gaps: 0,
        };
    }

    let avg_duration = deltas.iter().sum::<i64>() / deltas.len() as i64;

    // Calculate standard deviation
    let variance: f64 = deltas
        .iter()
        .map(|&d| {
            let diff = d - avg_duration;
            (diff * diff) as f64
        })
        .sum::<f64>()
        / deltas.len() as f64;

    let std_deviation = variance.sqrt();

    // Count gaps (deltas > 5x average)
    let gaps = deltas.iter().filter(|&&d| d > avg_duration * 5).count();

    TimestampStats {
        min,
        max,
        avg_duration,
        std_deviation,
        gaps,
    }
}

/// Check if timestamps are monotonically increasing.
pub fn is_monotonic(timestamps: &[i64]) -> bool {
    timestamps.windows(2).all(|w| w[0] < w[1])
}

/// Check if timestamps have consistent spacing.
pub fn has_consistent_spacing(timestamps: &[i64], tolerance: f64) -> bool {
    if timestamps.len() < 3 {
        return true;
    }

    let stats = calculate_stats(timestamps);
    let avg = stats.avg_duration as f64;

    for i in 1..timestamps.len() {
        let delta = (timestamps[i] - timestamps[i - 1]) as f64;
        let diff = (delta - avg).abs();

        if diff > avg * tolerance {
            return false;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_timestamps_valid() {
        let timestamps = vec![0, 40, 80, 120, 160];
        let result = validate_timestamps(&timestamps);
        assert!(result.valid);
        assert_eq!(result.issues.len(), 0);
    }

    #[test]
    fn test_validate_timestamps_negative() {
        let timestamps = vec![-100, 0, 100];
        let result = validate_timestamps(&timestamps);
        assert!(!result.valid);
        assert!(!result.issues.is_empty());
    }

    #[test]
    fn test_is_monotonic_true() {
        let timestamps = vec![0, 100, 200, 300];
        assert!(is_monotonic(&timestamps));
    }

    #[test]
    fn test_is_monotonic_false() {
        let timestamps = vec![0, 200, 100, 300];
        assert!(!is_monotonic(&timestamps));
    }

    #[test]
    fn test_has_consistent_spacing() {
        let timestamps = vec![0, 100, 200, 300];
        assert!(has_consistent_spacing(&timestamps, 0.1));
    }

    #[test]
    fn test_has_inconsistent_spacing() {
        let timestamps = vec![0, 100, 150, 400];
        assert!(!has_consistent_spacing(&timestamps, 0.1));
    }
}
