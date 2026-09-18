#![allow(dead_code)]

//! Shot statistics computation: duration distributions, type histograms,
//! and aggregate metrics for shot analysis.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Duration distribution
// ---------------------------------------------------------------------------

/// A histogram bucket for shot durations.
#[derive(Debug, Clone)]
pub struct DurationBucket {
    /// Lower bound of the bucket in seconds.
    pub lower_sec: f64,
    /// Upper bound of the bucket in seconds.
    pub upper_sec: f64,
    /// Number of shots in this bucket.
    pub count: usize,
}

/// Compute a duration histogram from a list of shot durations (in seconds).
///
/// `bucket_width` is the width of each bucket in seconds.
pub fn duration_histogram(durations: &[f64], bucket_width: f64) -> Vec<DurationBucket> {
    if durations.is_empty() || bucket_width <= 0.0 {
        return Vec::new();
    }

    let max_dur = durations.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let num_buckets = ((max_dur / bucket_width).ceil() as usize).max(1);

    let mut buckets: Vec<DurationBucket> = (0..num_buckets)
        .map(|i| {
            let lo = i as f64 * bucket_width;
            let hi = lo + bucket_width;
            DurationBucket {
                lower_sec: lo,
                upper_sec: hi,
                count: 0,
            }
        })
        .collect();

    for &dur in durations {
        let idx = ((dur / bucket_width).floor() as usize).min(num_buckets - 1);
        buckets[idx].count += 1;
    }

    buckets
}

// ---------------------------------------------------------------------------
// Shot type histogram
// ---------------------------------------------------------------------------

/// A named shot type and its count.
#[derive(Debug, Clone)]
pub struct TypeCount {
    /// The shot type label (e.g. "CU", "MS", "LS").
    pub label: String,
    /// Number of shots of this type.
    pub count: usize,
    /// Proportion of total shots (0.0 to 1.0).
    pub proportion: f64,
}

/// Build a histogram of shot types from a list of type labels.
pub fn shot_type_histogram(labels: &[&str]) -> Vec<TypeCount> {
    if labels.is_empty() {
        return Vec::new();
    }

    let mut counts: HashMap<&str, usize> = HashMap::new();
    for &label in labels {
        *counts.entry(label).or_insert(0) += 1;
    }

    let total = labels.len() as f64;
    let mut result: Vec<TypeCount> = counts
        .into_iter()
        .map(|(label, count)| TypeCount {
            label: label.to_string(),
            count,
            proportion: count as f64 / total,
        })
        .collect();

    result.sort_by(|a, b| b.count.cmp(&a.count));
    result
}

// ---------------------------------------------------------------------------
// Aggregate statistics
// ---------------------------------------------------------------------------

/// Aggregate statistics computed over a set of shot durations.
#[derive(Debug, Clone)]
pub struct AggregateStats {
    /// Number of shots.
    pub count: usize,
    /// Total combined duration in seconds.
    pub total_duration_sec: f64,
    /// Mean shot duration in seconds.
    pub mean_sec: f64,
    /// Median shot duration in seconds.
    pub median_sec: f64,
    /// Standard deviation of durations.
    pub std_dev_sec: f64,
    /// Minimum duration in seconds.
    pub min_sec: f64,
    /// Maximum duration in seconds.
    pub max_sec: f64,
    /// 25th percentile duration.
    pub p25_sec: f64,
    /// 75th percentile duration.
    pub p75_sec: f64,
}

/// Compute aggregate statistics from a list of durations (in seconds).
pub fn compute_aggregate_stats(durations: &[f64]) -> AggregateStats {
    if durations.is_empty() {
        return AggregateStats {
            count: 0,
            total_duration_sec: 0.0,
            mean_sec: 0.0,
            median_sec: 0.0,
            std_dev_sec: 0.0,
            min_sec: 0.0,
            max_sec: 0.0,
            p25_sec: 0.0,
            p75_sec: 0.0,
        };
    }

    let mut sorted = durations.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    let total: f64 = sorted.iter().sum();
    let mean = total / n as f64;
    let variance = sorted.iter().map(|&d| (d - mean).powi(2)).sum::<f64>() / n as f64;
    let std_dev = variance.sqrt();

    let median = percentile(&sorted, 50.0);
    let p25 = percentile(&sorted, 25.0);
    let p75 = percentile(&sorted, 75.0);

    AggregateStats {
        count: n,
        total_duration_sec: total,
        mean_sec: mean,
        median_sec: median,
        std_dev_sec: std_dev,
        min_sec: sorted[0],
        max_sec: sorted[n - 1],
        p25_sec: p25,
        p75_sec: p75,
    }
}

/// Compute a percentile value from sorted data using linear interpolation.
fn percentile(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }

    let idx = pct / 100.0 * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = idx - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

// ---------------------------------------------------------------------------
// Editing pace metrics
// ---------------------------------------------------------------------------

/// Metrics describing the editing pace of a sequence.
#[derive(Debug, Clone)]
pub struct PaceMetrics {
    /// Average shots per minute.
    pub shots_per_minute: f64,
    /// Average shot length in seconds.
    pub avg_shot_length_sec: f64,
    /// Coefficient of variation (std_dev / mean), measuring regularity.
    pub coefficient_of_variation: f64,
    /// Pace classification label.
    pub pace_label: PaceLabel,
}

/// Classification of editing pace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaceLabel {
    /// Slow pace (< 4 shots/min).
    Slow,
    /// Moderate pace (4-10 shots/min).
    Moderate,
    /// Fast pace (10-20 shots/min).
    Fast,
    /// Rapid pace (> 20 shots/min).
    Rapid,
}

/// Compute editing pace metrics from shot durations.
pub fn compute_pace_metrics(durations: &[f64]) -> PaceMetrics {
    if durations.is_empty() {
        return PaceMetrics {
            shots_per_minute: 0.0,
            avg_shot_length_sec: 0.0,
            coefficient_of_variation: 0.0,
            pace_label: PaceLabel::Slow,
        };
    }

    let stats = compute_aggregate_stats(durations);
    let total_minutes = stats.total_duration_sec / 60.0;
    let spm = if total_minutes > 0.0 {
        stats.count as f64 / total_minutes
    } else {
        0.0
    };

    let cv = if stats.mean_sec > 0.0 {
        stats.std_dev_sec / stats.mean_sec
    } else {
        0.0
    };

    let label = if spm < 4.0 {
        PaceLabel::Slow
    } else if spm < 10.0 {
        PaceLabel::Moderate
    } else if spm < 20.0 {
        PaceLabel::Fast
    } else {
        PaceLabel::Rapid
    };

    PaceMetrics {
        shots_per_minute: spm,
        avg_shot_length_sec: stats.mean_sec,
        coefficient_of_variation: cv,
        pace_label: label,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- duration_histogram tests --

    #[test]
    fn test_duration_histogram_empty() {
        let h = duration_histogram(&[], 1.0);
        assert!(h.is_empty());
    }

    #[test]
    fn test_duration_histogram_single() {
        let h = duration_histogram(&[2.5], 1.0);
        assert_eq!(h.len(), 3);
        assert_eq!(h[2].count, 1);
    }

    #[test]
    fn test_duration_histogram_multiple() {
        let durations = vec![0.5, 1.5, 2.5, 3.5, 1.2];
        let h = duration_histogram(&durations, 2.0);
        assert_eq!(h.len(), 2);
        let total: usize = h.iter().map(|b| b.count).sum();
        assert_eq!(total, 5);
    }

    #[test]
    fn test_duration_histogram_zero_width() {
        let h = duration_histogram(&[1.0], 0.0);
        assert!(h.is_empty());
    }

    // -- shot_type_histogram tests --

    #[test]
    fn test_type_histogram_empty() {
        let h = shot_type_histogram(&[]);
        assert!(h.is_empty());
    }

    #[test]
    fn test_type_histogram_basic() {
        let labels = vec!["CU", "MS", "CU", "LS", "CU"];
        let h = shot_type_histogram(&labels);
        let cu = h
            .iter()
            .find(|t| t.label == "CU")
            .expect("should succeed in test");
        assert_eq!(cu.count, 3);
        assert!((cu.proportion - 0.6).abs() < 1e-9);
    }

    #[test]
    fn test_type_histogram_single_type() {
        let labels = vec!["LS", "LS", "LS"];
        let h = shot_type_histogram(&labels);
        assert_eq!(h.len(), 1);
        assert!((h[0].proportion - 1.0).abs() < f64::EPSILON);
    }

    // -- aggregate stats tests --

    #[test]
    fn test_aggregate_empty() {
        let s = compute_aggregate_stats(&[]);
        assert_eq!(s.count, 0);
        assert!((s.mean_sec - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_aggregate_single() {
        let s = compute_aggregate_stats(&[5.0]);
        assert_eq!(s.count, 1);
        assert!((s.mean_sec - 5.0).abs() < f64::EPSILON);
        assert!((s.median_sec - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_aggregate_multiple() {
        let durations = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let s = compute_aggregate_stats(&durations);
        assert_eq!(s.count, 5);
        assert!((s.mean_sec - 3.0).abs() < 1e-9);
        assert!((s.median_sec - 3.0).abs() < 1e-9);
        assert!((s.min_sec - 1.0).abs() < f64::EPSILON);
        assert!((s.max_sec - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_aggregate_percentiles() {
        let durations = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let s = compute_aggregate_stats(&durations);
        // p25 should be around 3.25, p75 around 7.75
        assert!(s.p25_sec > 2.0 && s.p25_sec < 4.0);
        assert!(s.p75_sec > 7.0 && s.p75_sec < 9.0);
    }

    // -- pace metrics tests --

    #[test]
    fn test_pace_empty() {
        let p = compute_pace_metrics(&[]);
        assert!((p.shots_per_minute - 0.0).abs() < f64::EPSILON);
        assert_eq!(p.pace_label, PaceLabel::Slow);
    }

    #[test]
    fn test_pace_slow() {
        // 2 shots, each 30 seconds => 1 minute total => 2 shots/min
        let p = compute_pace_metrics(&[30.0, 30.0]);
        assert!(p.shots_per_minute < 4.0);
        assert_eq!(p.pace_label, PaceLabel::Slow);
    }

    #[test]
    fn test_pace_fast() {
        // 20 shots of 3 seconds => 1 minute total => 20 shots/min
        let durations: Vec<f64> = vec![3.0; 20];
        let p = compute_pace_metrics(&durations);
        assert!(p.shots_per_minute >= 10.0);
        assert!(p.pace_label == PaceLabel::Fast || p.pace_label == PaceLabel::Rapid);
    }

    #[test]
    fn test_pace_coefficient_of_variation() {
        // All same duration => cv = 0
        let durations = vec![5.0, 5.0, 5.0, 5.0];
        let p = compute_pace_metrics(&durations);
        assert!((p.coefficient_of_variation - 0.0).abs() < 1e-9);
    }
}
