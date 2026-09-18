//! Timestamp regeneration.
//!
//! This module provides functions to regenerate timestamps
//! when they are completely corrupted or missing.

use crate::Result;

/// Regenerate timestamps based on frame rate.
pub fn regenerate_timestamps(count: usize, fps: f64) -> Vec<i64> {
    let frame_duration = (1000.0 / fps) as i64; // milliseconds per frame

    (0..count).map(|i| i as i64 * frame_duration).collect()
}

/// Regenerate timestamps based on existing valid timestamps.
pub fn regenerate_from_existing(timestamps: &[i64]) -> Vec<i64> {
    if timestamps.len() < 2 {
        return timestamps.to_vec();
    }

    // Calculate average frame duration from valid timestamps
    let avg_duration = super::fix::calculate_average_delta(timestamps);

    // Generate new timestamps
    (0..timestamps.len())
        .map(|i| i as i64 * avg_duration)
        .collect()
}

/// Regenerate timestamps with variable frame rate.
pub fn regenerate_vfr(count: usize, durations: &[i64]) -> Result<Vec<i64>> {
    if durations.is_empty() {
        return Err(crate::RepairError::InvalidOptions(
            "No frame durations provided".to_string(),
        ));
    }

    let mut timestamps = Vec::with_capacity(count);
    let mut current = 0i64;

    for i in 0..count {
        timestamps.push(current);
        let duration_idx = i % durations.len();
        current += durations[duration_idx];
    }

    Ok(timestamps)
}

/// Interpolate missing timestamps.
pub fn interpolate_timestamps(timestamps: &mut [Option<i64>]) -> Vec<i64> {
    let mut result = Vec::with_capacity(timestamps.len());

    if timestamps.is_empty() {
        return result;
    }

    // Find first valid timestamp
    let mut last_valid_idx = 0;
    let mut last_valid_value = 0i64;

    for (i, ts) in timestamps.iter().enumerate() {
        if let Some(value) = ts {
            last_valid_idx = i;
            last_valid_value = *value;
            break;
        }
    }

    // Interpolate
    for (i, ts) in timestamps.iter().enumerate() {
        if let Some(value) = ts {
            result.push(*value);
            last_valid_idx = i;
            last_valid_value = *value;
        } else {
            // Find next valid timestamp
            if let Some((next_idx, next_value)) = find_next_valid(timestamps, i) {
                // Interpolate between last and next
                let delta = next_value - last_valid_value;
                let steps = (next_idx - last_valid_idx) as i64;
                let step = delta / steps;
                let value = last_valid_value + step * (i - last_valid_idx) as i64;
                result.push(value);
            } else {
                // No next valid, extrapolate
                let avg_delta = if last_valid_idx > 0 {
                    last_valid_value / last_valid_idx as i64
                } else {
                    40 // Default
                };
                result.push(last_valid_value + avg_delta * (i - last_valid_idx) as i64);
            }
        }
    }

    result
}

/// Find next valid timestamp.
fn find_next_valid(timestamps: &[Option<i64>], start: usize) -> Option<(usize, i64)> {
    for (i, ts) in timestamps.iter().enumerate().skip(start + 1) {
        if let Some(value) = ts {
            return Some((i, *value));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regenerate_timestamps() {
        let timestamps = regenerate_timestamps(5, 25.0); // 25 fps
        assert_eq!(timestamps.len(), 5);
        assert_eq!(timestamps[0], 0);
        assert_eq!(timestamps[1], 40); // 1000/25 = 40ms
        assert_eq!(timestamps[4], 160);
    }

    #[test]
    fn test_regenerate_timestamps_30fps() {
        let timestamps = regenerate_timestamps(3, 30.0);
        assert_eq!(timestamps.len(), 3);
        assert_eq!(timestamps[0], 0);
        assert_eq!(timestamps[1], 33); // 1000/30 ≈ 33ms
    }

    #[test]
    fn test_regenerate_from_existing() {
        let existing = vec![0, 50, 100, 150];
        let regenerated = regenerate_from_existing(&existing);
        assert_eq!(regenerated.len(), 4);
        assert_eq!(regenerated[0], 0);
        assert_eq!(regenerated[1], 50);
    }

    #[test]
    fn test_interpolate_timestamps() {
        let mut timestamps = vec![Some(0), None, None, Some(300), None];
        let interpolated = interpolate_timestamps(&mut timestamps);

        assert_eq!(interpolated.len(), 5);
        assert_eq!(interpolated[0], 0);
        assert_eq!(interpolated[1], 100); // Interpolated
        assert_eq!(interpolated[2], 200); // Interpolated
        assert_eq!(interpolated[3], 300);
    }
}
