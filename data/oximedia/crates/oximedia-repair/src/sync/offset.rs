//! Sync offset calculation.
//!
//! This module provides functions to calculate the sync offset between
//! audio and video streams.

/// Calculate optimal sync offset using cross-correlation.
pub fn calculate_optimal_offset(audio_timestamps: &[i64], video_timestamps: &[i64]) -> i64 {
    let min_len = audio_timestamps.len().min(video_timestamps.len());

    if min_len == 0 {
        return 0;
    }

    // Simple approach: use median offset
    let mut offsets = Vec::with_capacity(min_len);

    for i in 0..min_len {
        offsets.push(audio_timestamps[i] - video_timestamps[i]);
    }

    offsets.sort_unstable();

    offsets[offsets.len() / 2]
}

/// Calculate offset at specific time point.
pub fn calculate_offset_at_time(
    audio_timestamps: &[i64],
    video_timestamps: &[i64],
    time: i64,
) -> Option<i64> {
    // Find closest timestamps to the target time
    let audio_idx = find_closest_timestamp(audio_timestamps, time)?;
    let video_idx = find_closest_timestamp(video_timestamps, time)?;

    Some(audio_timestamps[audio_idx] - video_timestamps[video_idx])
}

/// Find index of timestamp closest to target time.
fn find_closest_timestamp(timestamps: &[i64], time: i64) -> Option<usize> {
    if timestamps.is_empty() {
        return None;
    }

    let mut closest_idx = 0;
    let mut closest_diff = (timestamps[0] - time).abs();

    for (i, &ts) in timestamps.iter().enumerate() {
        let diff = (ts - time).abs();
        if diff < closest_diff {
            closest_diff = diff;
            closest_idx = i;
        }
    }

    Some(closest_idx)
}

/// Detect variable offset (drift).
pub fn detect_variable_offset(
    audio_timestamps: &[i64],
    video_timestamps: &[i64],
) -> Option<(i64, i64)> {
    let min_len = audio_timestamps.len().min(video_timestamps.len());

    if min_len < 2 {
        return None;
    }

    let start_offset = audio_timestamps[0] - video_timestamps[0];
    let end_offset = audio_timestamps[min_len - 1] - video_timestamps[min_len - 1];

    if (end_offset - start_offset).abs() > 10 {
        Some((start_offset, end_offset))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_optimal_offset() {
        let audio = vec![100, 200, 300];
        let video = vec![0, 100, 200];

        let offset = calculate_optimal_offset(&audio, &video);
        assert_eq!(offset, 100);
    }

    #[test]
    fn test_find_closest_timestamp() {
        let timestamps = vec![0, 100, 200, 300];

        assert_eq!(find_closest_timestamp(&timestamps, 150), Some(1));
        assert_eq!(find_closest_timestamp(&timestamps, 250), Some(2));
    }

    #[test]
    fn test_detect_variable_offset() {
        let audio = vec![100, 200, 400];
        let video = vec![0, 100, 200];

        let result = detect_variable_offset(&audio, &video);
        assert!(result.is_some());
    }
}
