//! DRM usage analytics.
//!
//! Tracks DRM lifecycle events and computes aggregate statistics.

/// The type of DRM event that occurred.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DrmEventType {
    LicenseRequest,
    LicenseGranted,
    LicenseDenied,
    PlaybackStart,
    PlaybackStop,
    OfflineSync,
    KeyRotation,
}

/// A single recorded DRM event.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct DrmEvent {
    pub event_id: u64,
    pub timestamp: u64,
    pub event_type: DrmEventType,
    pub content_id: String,
    pub user_id: String,
}

/// Accumulates DRM events and provides aggregate statistics.
#[allow(dead_code)]
#[derive(Debug, Default)]
pub struct DrmAnalytics {
    events: Vec<DrmEvent>,
    next_id: u64,
}

impl DrmAnalytics {
    /// Create a new analytics collector.
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            next_id: 1,
        }
    }

    /// Record a DRM event and return the assigned event ID.
    pub fn record(
        &mut self,
        ts: u64,
        event_type: DrmEventType,
        content_id: &str,
        user_id: &str,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.events.push(DrmEvent {
            event_id: id,
            timestamp: ts,
            event_type,
            content_id: content_id.to_string(),
            user_id: user_id.to_string(),
        });
        id
    }

    /// Return all events associated with the given content ID.
    pub fn events_for_content(&self, content_id: &str) -> Vec<&DrmEvent> {
        self.events
            .iter()
            .filter(|e| e.content_id == content_id)
            .collect()
    }

    /// Count `LicenseDenied` events.
    pub fn denied_count(&self) -> usize {
        self.events
            .iter()
            .filter(|e| e.event_type == DrmEventType::LicenseDenied)
            .count()
    }

    /// Count `LicenseGranted` events.
    pub fn granted_count(&self) -> usize {
        self.events
            .iter()
            .filter(|e| e.event_type == DrmEventType::LicenseGranted)
            .count()
    }

    /// Fraction of license requests that were granted.
    ///
    /// Returns `0.0` when there are no requests at all.
    pub fn success_rate(&self) -> f64 {
        let granted = self.granted_count();
        let denied = self.denied_count();
        let total = granted + denied;
        if total == 0 {
            0.0
        } else {
            granted as f64 / total as f64
        }
    }

    /// Return a slice of all recorded events.
    pub fn all_events(&self) -> &[DrmEvent] {
        &self.events
    }

    /// Return the total number of recorded events.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }
}

/// A single playback session event.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PlaybackEvent {
    /// Content identifier.
    pub content_id: String,
    /// User identifier.
    pub user_id: String,
    /// Timestamp when playback started (Unix milliseconds).
    pub timestamp_ms: u64,
    /// How long the user actually watched (milliseconds).
    pub duration_ms: u64,
    /// Requester IP address.
    pub ip: String,
    /// Device category (e.g. `"mobile"`, `"desktop"`, `"tv"`).
    pub device_type: String,
}

impl PlaybackEvent {
    /// Create a new playback event.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        content_id: impl Into<String>,
        user_id: impl Into<String>,
        timestamp_ms: u64,
        duration_ms: u64,
        ip: impl Into<String>,
        device_type: impl Into<String>,
    ) -> Self {
        Self {
            content_id: content_id.into(),
            user_id: user_id.into(),
            timestamp_ms,
            duration_ms,
            ip: ip.into(),
            device_type: device_type.into(),
        }
    }

    /// Returns `true` when the user watched at least 90 % of `content_duration_ms`.
    pub fn is_complete_view(&self, content_duration_ms: u64) -> bool {
        if content_duration_ms == 0 {
            return false;
        }
        // Use integer arithmetic to avoid floating-point: watched * 10 >= duration * 9
        self.duration_ms.saturating_mul(10) >= content_duration_ms.saturating_mul(9)
    }
}

/// Aggregates playback events for DRM usage reporting.
#[allow(dead_code)]
#[derive(Debug, Default)]
pub struct PlaybackAnalytics {
    events: Vec<PlaybackEvent>,
}

impl PlaybackAnalytics {
    /// Create a new analytics store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a playback event.
    pub fn record(&mut self, event: PlaybackEvent) {
        self.events.push(event);
    }

    /// Total number of recorded playback events.
    pub fn total_plays(&self) -> usize {
        self.events.len()
    }

    /// Number of distinct user IDs in the recorded events.
    pub fn unique_users(&self) -> usize {
        let mut seen: Vec<&str> = Vec::new();
        for ev in &self.events {
            if !seen.contains(&ev.user_id.as_str()) {
                seen.push(&ev.user_id);
            }
        }
        seen.len()
    }

    /// Number of plays for a specific content ID.
    pub fn plays_per_content(&self, content_id: &str) -> usize {
        self.events
            .iter()
            .filter(|e| e.content_id == content_id)
            .count()
    }

    /// Return the top-`n` content IDs by play count, sorted descending.
    pub fn top_content(&self, n: usize) -> Vec<(String, usize)> {
        let mut counts: Vec<(String, usize)> = Vec::new();
        for ev in &self.events {
            if let Some(entry) = counts.iter_mut().find(|(id, _)| id == &ev.content_id) {
                entry.1 += 1;
            } else {
                counts.push((ev.content_id.clone(), 1));
            }
        }
        counts.sort_by(|a, b| b.1.cmp(&a.1));
        counts.truncate(n);
        counts
    }
}

/// A summary report for a specific time window.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct UsageReport {
    /// Report window start (Unix milliseconds).
    pub period_start_ms: u64,
    /// Report window end (Unix milliseconds).
    pub period_end_ms: u64,
    /// Total plays within the window.
    pub total_plays: u64,
    /// Unique users within the window.
    pub unique_users: u64,
    /// Average playback duration within the window (milliseconds).
    pub avg_duration_ms: f64,
}

impl UsageReport {
    /// Generate a report for events in `[start_ms, end_ms)`.
    pub fn generate(analytics: &PlaybackAnalytics, start_ms: u64, end_ms: u64) -> UsageReport {
        let window: Vec<&PlaybackEvent> = analytics
            .events
            .iter()
            .filter(|e| e.timestamp_ms >= start_ms && e.timestamp_ms < end_ms)
            .collect();

        let total_plays = window.len() as u64;

        let mut seen_users: Vec<&str> = Vec::new();
        for ev in &window {
            if !seen_users.contains(&ev.user_id.as_str()) {
                seen_users.push(&ev.user_id);
            }
        }
        let unique_users = seen_users.len() as u64;

        #[allow(clippy::cast_precision_loss)]
        let avg_duration_ms = if total_plays == 0 {
            0.0_f64
        } else {
            let total_dur: u64 = window.iter().map(|e| e.duration_ms).sum();
            total_dur as f64 / total_plays as f64
        };

        UsageReport {
            period_start_ms: start_ms,
            period_end_ms: end_ms,
            total_plays,
            unique_users,
            avg_duration_ms,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_analytics_empty() {
        let a = DrmAnalytics::new();
        assert_eq!(a.event_count(), 0);
        assert_eq!(a.granted_count(), 0);
        assert_eq!(a.denied_count(), 0);
    }

    #[test]
    fn test_record_returns_sequential_ids() {
        let mut a = DrmAnalytics::new();
        let id1 = a.record(1000, DrmEventType::LicenseRequest, "c1", "u1");
        let id2 = a.record(1001, DrmEventType::LicenseGranted, "c1", "u1");
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
    }

    #[test]
    fn test_event_count_grows() {
        let mut a = DrmAnalytics::new();
        a.record(1, DrmEventType::LicenseRequest, "c", "u");
        a.record(2, DrmEventType::PlaybackStart, "c", "u");
        assert_eq!(a.event_count(), 2);
    }

    #[test]
    fn test_events_for_content_filters() {
        let mut a = DrmAnalytics::new();
        a.record(1, DrmEventType::LicenseGranted, "movie-1", "u");
        a.record(2, DrmEventType::LicenseGranted, "movie-2", "u");
        a.record(3, DrmEventType::LicenseDenied, "movie-1", "u");
        let m1 = a.events_for_content("movie-1");
        assert_eq!(m1.len(), 2);
        for ev in &m1 {
            assert_eq!(ev.content_id, "movie-1");
        }
    }

    #[test]
    fn test_events_for_content_none() {
        let a = DrmAnalytics::new();
        assert!(a.events_for_content("missing").is_empty());
    }

    #[test]
    fn test_denied_count() {
        let mut a = DrmAnalytics::new();
        a.record(1, DrmEventType::LicenseDenied, "c", "u");
        a.record(2, DrmEventType::LicenseDenied, "c", "u");
        a.record(3, DrmEventType::LicenseGranted, "c", "u");
        assert_eq!(a.denied_count(), 2);
    }

    #[test]
    fn test_granted_count() {
        let mut a = DrmAnalytics::new();
        a.record(1, DrmEventType::LicenseGranted, "c", "u");
        a.record(2, DrmEventType::LicenseGranted, "c", "u");
        assert_eq!(a.granted_count(), 2);
    }

    #[test]
    fn test_success_rate_all_granted() {
        let mut a = DrmAnalytics::new();
        a.record(1, DrmEventType::LicenseGranted, "c", "u");
        a.record(2, DrmEventType::LicenseGranted, "c", "u");
        assert!((a.success_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_success_rate_mixed() {
        let mut a = DrmAnalytics::new();
        a.record(1, DrmEventType::LicenseGranted, "c", "u");
        a.record(2, DrmEventType::LicenseDenied, "c", "u");
        assert!((a.success_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_success_rate_no_events() {
        let a = DrmAnalytics::new();
        assert_eq!(a.success_rate(), 0.0);
    }

    #[test]
    fn test_all_event_types_recordable() {
        let mut a = DrmAnalytics::new();
        let types = [
            DrmEventType::LicenseRequest,
            DrmEventType::LicenseGranted,
            DrmEventType::LicenseDenied,
            DrmEventType::PlaybackStart,
            DrmEventType::PlaybackStop,
            DrmEventType::OfflineSync,
            DrmEventType::KeyRotation,
        ];
        for (i, t) in types.into_iter().enumerate() {
            a.record(i as u64, t, "c", "u");
        }
        assert_eq!(a.event_count(), 7);
    }

    #[test]
    fn test_event_fields_stored_correctly() {
        let mut a = DrmAnalytics::new();
        a.record(42, DrmEventType::PlaybackStop, "film-99", "alice");
        let ev = &a.all_events()[0];
        assert_eq!(ev.timestamp, 42);
        assert_eq!(ev.content_id, "film-99");
        assert_eq!(ev.user_id, "alice");
        assert_eq!(ev.event_type, DrmEventType::PlaybackStop);
    }

    #[test]
    fn test_success_rate_all_denied() {
        let mut a = DrmAnalytics::new();
        a.record(1, DrmEventType::LicenseDenied, "c", "u");
        assert_eq!(a.success_rate(), 0.0);
    }

    // ----- PlaybackEvent tests -----

    #[test]
    fn test_playback_event_is_complete_view_true() {
        let ev = PlaybackEvent::new("c", "u", 0, 900, "1.2.3.4", "desktop");
        // 90% of 1000ms = 900ms -> complete
        assert!(ev.is_complete_view(1000));
    }

    #[test]
    fn test_playback_event_is_complete_view_false() {
        let ev = PlaybackEvent::new("c", "u", 0, 800, "1.2.3.4", "desktop");
        assert!(!ev.is_complete_view(1000));
    }

    #[test]
    fn test_playback_event_is_complete_view_zero_content() {
        let ev = PlaybackEvent::new("c", "u", 0, 100, "ip", "mobile");
        assert!(!ev.is_complete_view(0));
    }

    // ----- PlaybackAnalytics tests -----

    #[test]
    fn test_playback_analytics_total_plays() {
        let mut pa = PlaybackAnalytics::new();
        pa.record(PlaybackEvent::new("c1", "u1", 0, 100, "ip", "tv"));
        pa.record(PlaybackEvent::new("c1", "u2", 0, 100, "ip", "tv"));
        assert_eq!(pa.total_plays(), 2);
    }

    #[test]
    fn test_playback_analytics_unique_users() {
        let mut pa = PlaybackAnalytics::new();
        pa.record(PlaybackEvent::new("c", "alice", 0, 50, "ip", "mobile"));
        pa.record(PlaybackEvent::new("c", "alice", 10, 50, "ip", "mobile"));
        pa.record(PlaybackEvent::new("c", "bob", 20, 50, "ip", "desktop"));
        assert_eq!(pa.unique_users(), 2);
    }

    #[test]
    fn test_playback_analytics_plays_per_content() {
        let mut pa = PlaybackAnalytics::new();
        pa.record(PlaybackEvent::new("movie-a", "u", 0, 100, "ip", "tv"));
        pa.record(PlaybackEvent::new("movie-a", "v", 0, 100, "ip", "tv"));
        pa.record(PlaybackEvent::new("movie-b", "u", 0, 100, "ip", "tv"));
        assert_eq!(pa.plays_per_content("movie-a"), 2);
        assert_eq!(pa.plays_per_content("movie-b"), 1);
        assert_eq!(pa.plays_per_content("movie-c"), 0);
    }

    #[test]
    fn test_playback_analytics_top_content() {
        let mut pa = PlaybackAnalytics::new();
        for _ in 0..3 {
            pa.record(PlaybackEvent::new("popular", "u", 0, 100, "ip", "tv"));
        }
        pa.record(PlaybackEvent::new("rare", "u", 0, 100, "ip", "tv"));
        let top = pa.top_content(1);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].0, "popular");
        assert_eq!(top[0].1, 3);
    }

    // ----- UsageReport tests -----

    #[test]
    fn test_usage_report_empty_window() {
        let pa = PlaybackAnalytics::new();
        let report = UsageReport::generate(&pa, 0, 1000);
        assert_eq!(report.total_plays, 0);
        assert_eq!(report.unique_users, 0);
        assert_eq!(report.avg_duration_ms, 0.0);
    }

    #[test]
    fn test_usage_report_correct_aggregates() {
        let mut pa = PlaybackAnalytics::new();
        pa.record(PlaybackEvent::new("c", "u1", 100, 200, "ip", "tv"));
        pa.record(PlaybackEvent::new("c", "u2", 500, 400, "ip", "mobile"));
        // Event at ts=2000 is outside [0, 1000)
        pa.record(PlaybackEvent::new("c", "u3", 2000, 100, "ip", "desktop"));

        let report = UsageReport::generate(&pa, 0, 1000);
        assert_eq!(report.total_plays, 2);
        assert_eq!(report.unique_users, 2);
        assert!((report.avg_duration_ms - 300.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_usage_report_period_fields() {
        let pa = PlaybackAnalytics::new();
        let report = UsageReport::generate(&pa, 5000, 10_000);
        assert_eq!(report.period_start_ms, 5000);
        assert_eq!(report.period_end_ms, 10_000);
    }
}
