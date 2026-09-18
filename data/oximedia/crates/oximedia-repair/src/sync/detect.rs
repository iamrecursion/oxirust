//! A/V sync issue detection.
//!
//! This module provides functions to detect audio/video synchronization issues.

/// Sync issue type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncIssue {
    /// Audio is ahead of video.
    AudioAhead,
    /// Video is ahead of audio.
    VideoAhead,
    /// Drift detected (gradual desync).
    Drift,
}

/// Detect A/V sync issues.
pub fn detect_sync_issues(
    audio_timestamps: &[i64],
    video_timestamps: &[i64],
) -> Vec<(usize, SyncIssue, i64)> {
    let mut issues = Vec::new();

    let min_len = audio_timestamps.len().min(video_timestamps.len());

    for i in 0..min_len {
        let diff = audio_timestamps[i] - video_timestamps[i];

        // Threshold: 40ms (for 25fps video)
        if diff > 40 {
            issues.push((i, SyncIssue::AudioAhead, diff));
        } else if diff < -40 {
            issues.push((i, SyncIssue::VideoAhead, diff.abs()));
        }
    }

    // Detect drift
    if detect_drift(audio_timestamps, video_timestamps) {
        issues.push((0, SyncIssue::Drift, 0));
    }

    issues
}

/// Detect gradual drift in A/V sync.
fn detect_drift(audio_timestamps: &[i64], video_timestamps: &[i64]) -> bool {
    let min_len = audio_timestamps.len().min(video_timestamps.len());

    if min_len < 10 {
        return false;
    }

    // Calculate difference at start and end
    let start_diff = audio_timestamps[0] - video_timestamps[0];
    let end_diff = audio_timestamps[min_len - 1] - video_timestamps[min_len - 1];

    let total_drift = (end_diff - start_diff).abs();

    // If drift is more than 100ms over the sequence, it's significant
    total_drift > 100
}

/// Calculate average A/V offset.
pub fn calculate_av_offset(audio_timestamps: &[i64], video_timestamps: &[i64]) -> i64 {
    let min_len = audio_timestamps.len().min(video_timestamps.len());

    if min_len == 0 {
        return 0;
    }

    let mut sum = 0i64;
    for i in 0..min_len {
        sum += audio_timestamps[i] - video_timestamps[i];
    }

    sum / min_len as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_sync_issues_audio_ahead() {
        let audio = vec![100, 200, 300];
        let video = vec![0, 100, 200];

        let issues = detect_sync_issues(&audio, &video);
        assert!(!issues.is_empty());
        assert!(issues
            .iter()
            .any(|(_, issue, _)| *issue == SyncIssue::AudioAhead));
    }

    #[test]
    fn test_detect_sync_issues_in_sync() {
        let audio = vec![0, 100, 200];
        let video = vec![0, 100, 200];

        let issues = detect_sync_issues(&audio, &video);
        let non_drift_issues: Vec<_> = issues
            .iter()
            .filter(|(_, issue, _)| *issue != SyncIssue::Drift)
            .collect();
        assert!(non_drift_issues.is_empty());
    }

    #[test]
    fn test_calculate_av_offset() {
        let audio = vec![100, 200, 300];
        let video = vec![0, 100, 200];

        let offset = calculate_av_offset(&audio, &video);
        assert_eq!(offset, 100);
    }
}
