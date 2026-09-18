//! View history analysis.

use super::track::ViewEvent;
use chrono::{Datelike, Timelike};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// History analyzer
pub struct HistoryAnalyzer;

impl HistoryAnalyzer {
    /// Analyze user viewing patterns
    #[must_use]
    pub fn analyze_patterns(events: &[ViewEvent]) -> ViewingPatternAnalysis {
        if events.is_empty() {
            return ViewingPatternAnalysis::default();
        }

        let temporal_patterns = Self::analyze_temporal_patterns(events);
        let content_patterns = Self::analyze_content_patterns(events);
        let engagement_patterns = Self::analyze_engagement_patterns(events);

        ViewingPatternAnalysis {
            temporal_patterns,
            content_patterns,
            engagement_patterns,
            total_events: events.len(),
        }
    }

    /// Analyze temporal viewing patterns
    fn analyze_temporal_patterns(events: &[ViewEvent]) -> TemporalPatterns {
        let mut hourly_counts = vec![0u32; 24];
        let mut daily_counts = vec![0u32; 7];

        for event in events {
            let datetime = chrono::DateTime::from_timestamp(event.timestamp, 0);
            if let Some(dt) = datetime {
                let hour = dt.hour() as usize;
                let day = dt.weekday().num_days_from_monday() as usize;

                if hour < 24 {
                    hourly_counts[hour] += 1;
                }
                if day < 7 {
                    daily_counts[day] += 1;
                }
            }
        }

        let peak_hour = hourly_counts
            .iter()
            .enumerate()
            .max_by_key(|(_, &count)| count)
            .map_or(0, |(hour, _)| hour as u8);

        let peak_day = daily_counts
            .iter()
            .enumerate()
            .max_by_key(|(_, &count)| count)
            .map_or(0, |(day, _)| day as u8);

        TemporalPatterns {
            hourly_distribution: hourly_counts,
            daily_distribution: daily_counts,
            peak_hour,
            peak_day,
        }
    }

    /// Analyze content viewing patterns
    fn analyze_content_patterns(events: &[ViewEvent]) -> ContentPatterns {
        let mut content_counts: HashMap<Uuid, usize> = HashMap::new();
        let mut device_counts: HashMap<String, usize> = HashMap::new();
        let mut quality_counts: HashMap<String, usize> = HashMap::new();

        for event in events {
            *content_counts.entry(event.content_id).or_insert(0) += 1;

            if let Some(ref device) = event.device {
                *device_counts.entry(device.clone()).or_insert(0) += 1;
            }

            if let Some(ref quality) = event.quality {
                *quality_counts.entry(quality.clone()).or_insert(0) += 1;
            }
        }

        let most_watched = content_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(content_id, _)| content_id);

        let preferred_device = device_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(device, _)| device);

        let preferred_quality = quality_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(quality, _)| quality);

        ContentPatterns {
            most_watched,
            preferred_device,
            preferred_quality,
        }
    }

    /// Analyze engagement patterns
    fn analyze_engagement_patterns(events: &[ViewEvent]) -> EngagementPatterns {
        if events.is_empty() {
            return EngagementPatterns::default();
        }

        let total_watch_time: i64 = events.iter().map(|e| e.watch_time_ms).sum();
        let avg_watch_time = total_watch_time
            .checked_div(events.len() as i64)
            .unwrap_or(0);

        let completed_count = events.iter().filter(|e| e.completed).count();
        #[allow(clippy::manual_checked_ops)]
        let completion_rate = completed_count as f32 / events.len() as f32;

        // Calculate session consistency
        let mut session_gaps = Vec::new();
        let mut sorted_events = events.to_vec();
        sorted_events.sort_by_key(|e| e.timestamp);

        for window in sorted_events.windows(2) {
            let gap = window[1].timestamp - window[0].timestamp;
            session_gaps.push(gap);
        }

        let avg_session_gap = session_gaps
            .iter()
            .sum::<i64>()
            .checked_div(session_gaps.len() as i64)
            .unwrap_or(0);

        EngagementPatterns {
            avg_watch_time_ms: avg_watch_time,
            completion_rate,
            avg_session_gap_seconds: avg_session_gap,
            total_sessions: events.len(),
        }
    }

    /// Detect binge-watching behavior
    #[must_use]
    pub fn detect_binge_watching(events: &[ViewEvent]) -> BingeWatchingMetrics {
        if events.is_empty() {
            return BingeWatchingMetrics::default();
        }

        let mut sorted_events = events.to_vec();
        sorted_events.sort_by_key(|e| e.timestamp);

        let mut binge_sessions = 0;
        let mut current_session_count = 0;
        let binge_threshold_seconds = 1800; // 30 minutes

        for window in sorted_events.windows(2) {
            let gap = window[1].timestamp - window[0].timestamp;

            if gap <= binge_threshold_seconds {
                current_session_count += 1;
            } else {
                if current_session_count >= 2 {
                    binge_sessions += 1;
                }
                current_session_count = 0;
            }
        }

        #[allow(clippy::manual_checked_ops)]
        let binge_tendency = if events.is_empty() {
            0.0
        } else {
            binge_sessions as f32 / events.len() as f32
        };

        BingeWatchingMetrics {
            binge_sessions,
            binge_tendency,
            avg_consecutive_views: events.len().checked_div(binge_sessions).unwrap_or(0),
        }
    }

    /// Calculate content diversity score
    #[must_use]
    pub fn calculate_diversity_score(events: &[ViewEvent]) -> f32 {
        if events.is_empty() {
            return 0.0;
        }

        #[allow(clippy::manual_checked_ops)]
        {
            let unique_content: std::collections::HashSet<Uuid> =
                events.iter().map(|e| e.content_id).collect();

            unique_content.len() as f32 / events.len() as f32
        }
    }
}

/// Viewing pattern analysis results
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ViewingPatternAnalysis {
    /// Temporal patterns
    pub temporal_patterns: TemporalPatterns,
    /// Content patterns
    pub content_patterns: ContentPatterns,
    /// Engagement patterns
    pub engagement_patterns: EngagementPatterns,
    /// Total events analyzed
    pub total_events: usize,
}

/// Temporal viewing patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalPatterns {
    /// Hourly distribution (0-23)
    pub hourly_distribution: Vec<u32>,
    /// Daily distribution (0-6, Monday-Sunday)
    pub daily_distribution: Vec<u32>,
    /// Peak viewing hour
    pub peak_hour: u8,
    /// Peak viewing day
    pub peak_day: u8,
}

impl Default for TemporalPatterns {
    fn default() -> Self {
        Self {
            hourly_distribution: vec![0; 24],
            daily_distribution: vec![0; 7],
            peak_hour: 0,
            peak_day: 0,
        }
    }
}

/// Content viewing patterns
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ContentPatterns {
    /// Most watched content
    pub most_watched: Option<Uuid>,
    /// Preferred device
    pub preferred_device: Option<String>,
    /// Preferred quality
    pub preferred_quality: Option<String>,
}

/// Engagement patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngagementPatterns {
    /// Average watch time (milliseconds)
    pub avg_watch_time_ms: i64,
    /// Completion rate
    pub completion_rate: f32,
    /// Average gap between sessions (seconds)
    pub avg_session_gap_seconds: i64,
    /// Total sessions
    pub total_sessions: usize,
}

impl Default for EngagementPatterns {
    fn default() -> Self {
        Self {
            avg_watch_time_ms: 0,
            completion_rate: 0.0,
            avg_session_gap_seconds: 0,
            total_sessions: 0,
        }
    }
}

/// Binge-watching metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BingeWatchingMetrics {
    /// Number of binge sessions detected
    pub binge_sessions: usize,
    /// Binge tendency (0-1)
    pub binge_tendency: f32,
    /// Average consecutive views per binge
    pub avg_consecutive_views: usize,
}

impl Default for BingeWatchingMetrics {
    fn default() -> Self {
        Self {
            binge_sessions: 0,
            binge_tendency: 0.0,
            avg_consecutive_views: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyze_patterns_empty() {
        let events = vec![];
        let analysis = HistoryAnalyzer::analyze_patterns(&events);
        assert_eq!(analysis.total_events, 0);
    }

    #[test]
    fn test_detect_binge_watching() {
        let events = vec![ViewEvent::new(Uuid::new_v4(), Uuid::new_v4(), 60000, true)];
        let metrics = HistoryAnalyzer::detect_binge_watching(&events);
        assert_eq!(metrics.binge_sessions, 0);
    }

    #[test]
    fn test_calculate_diversity_score() {
        let user_id = Uuid::new_v4();
        let content1 = Uuid::new_v4();
        let content2 = Uuid::new_v4();

        let events = vec![
            ViewEvent::new(user_id, content1, 60000, true),
            ViewEvent::new(user_id, content2, 60000, true),
            ViewEvent::new(user_id, content1, 60000, true),
        ];

        let diversity = HistoryAnalyzer::calculate_diversity_score(&events);
        assert!((diversity - 0.666_666_7).abs() < 0.001);
    }
}
