//! Session-based recommendation engine.
//!
//! Tracks in-session user interactions (views, clicks, plays, etc.) and uses
//! them to generate real-time recommendations without requiring a persistent
//! user profile.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Event types
// ---------------------------------------------------------------------------

/// The type of a session event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum SessionEventType {
    /// User viewed the content thumbnail / detail page
    View,
    /// User clicked on the content
    Click,
    /// User started playing the content
    Play,
    /// User skipped past the content
    Skip,
    /// User explicitly liked the content
    Like,
    /// User explicitly disliked the content
    Dislike,
    /// User shared the content
    Share,
    /// User purchased the content
    Purchase,
}

impl SessionEventType {
    /// Return the default affinity weight for this event type.
    ///
    /// Positive values indicate interest, negative values indicate disinterest.
    #[must_use]
    pub fn affinity_weight(self) -> f32 {
        match self {
            Self::Like => 1.0,
            Self::Purchase => 0.9,
            Self::Share => 0.8,
            Self::Play => 0.7,
            Self::Click => 0.4,
            Self::View => 0.2,
            Self::Skip => -0.3,
            Self::Dislike => -0.8,
        }
    }
}

// ---------------------------------------------------------------------------
// Session event
// ---------------------------------------------------------------------------

/// A single interaction event recorded within a session.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SessionEvent {
    /// ID of the media item involved
    pub media_id: u64,
    /// Type of interaction
    pub event_type: SessionEventType,
    /// Event timestamp in milliseconds since epoch
    pub timestamp_ms: u64,
    /// How long the user engaged (milliseconds), if applicable
    pub duration_ms: Option<u64>,
}

impl SessionEvent {
    /// Create a new session event.
    #[must_use]
    pub fn new(
        media_id: u64,
        event_type: SessionEventType,
        timestamp_ms: u64,
        duration_ms: Option<u64>,
    ) -> Self {
        Self {
            media_id,
            event_type,
            timestamp_ms,
            duration_ms,
        }
    }
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

/// A user session containing a sequence of interaction events.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Session {
    /// Unique session identifier
    pub id: String,
    /// Optional user ID (anonymous sessions have `None`)
    pub user_id: Option<u64>,
    /// Ordered list of events in this session
    pub events: Vec<SessionEvent>,
    /// Session creation time in milliseconds since epoch
    pub created_ms: u64,
}

impl Session {
    /// Create a new empty session.
    #[must_use]
    pub fn new(id: impl Into<String>, user_id: Option<u64>, created_ms: u64) -> Self {
        Self {
            id: id.into(),
            user_id,
            events: Vec::new(),
            created_ms,
        }
    }

    /// Append an event to this session.
    pub fn add_event(&mut self, event: SessionEvent) {
        self.events.push(event);
    }

    /// Compute the completion rate for a media item.
    ///
    /// Defined as the total play duration / total available duration from all
    /// `Play` events for that media item.  Returns 0.0 if no play events or
    /// if the total duration cannot be determined.
    #[must_use]
    pub fn completion_rate(&self, media_id: u64) -> f32 {
        let mut total_play_ms: u64 = 0;
        let mut total_available_ms: u64 = 0;

        for event in &self.events {
            if event.media_id != media_id || event.event_type != SessionEventType::Play {
                continue;
            }
            if let Some(dur) = event.duration_ms {
                total_play_ms += dur;
                // Heuristic: assume the first play event's duration is the
                // content length if we only have one data point.
                total_available_ms += dur;
            }
        }

        if total_available_ms == 0 {
            return 0.0;
        }

        (total_play_ms as f32 / total_available_ms as f32).min(1.0)
    }

    /// Compute an affinity score for a category.
    ///
    /// Looks up each interacted media item in `categories` and sums up the
    /// event weights for any media item that belongs to `category`.
    /// The result is clamped to `[0.0, 1.0]`.
    #[must_use]
    pub fn affinity_score(&self, category: &str, categories: &HashMap<u64, Vec<String>>) -> f32 {
        if self.events.is_empty() {
            return 0.0;
        }

        let mut score = 0.0_f32;
        let mut count = 0usize;

        for event in &self.events {
            if let Some(cats) = categories.get(&event.media_id) {
                if cats.iter().any(|c| c == category) {
                    score += event.event_type.affinity_weight();
                    count += 1;
                }
            }
        }

        if count == 0 {
            return 0.0;
        }

        (score / count as f32).clamp(0.0, 1.0)
    }

    /// Return the most recent timestamp in the session, or `created_ms` if empty.
    #[must_use]
    pub fn last_activity_ms(&self) -> u64 {
        self.events
            .iter()
            .map(|e| e.timestamp_ms)
            .max()
            .unwrap_or(self.created_ms)
    }
}

// ---------------------------------------------------------------------------
// Session store
// ---------------------------------------------------------------------------

/// In-memory store for active sessions.
#[derive(Debug, Default)]
#[allow(dead_code)]
pub struct SessionStore {
    sessions: HashMap<String, Session>,
    next_id: u64,
}

impl SessionStore {
    /// Create a new empty session store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new anonymous session and return its ID.
    pub fn create_session(&mut self) -> String {
        self.create_session_for_user(None)
    }

    /// Create a new session for an optional user and return its ID.
    pub fn create_session_for_user(&mut self, user_id: Option<u64>) -> String {
        self.next_id += 1;
        let id = format!("session-{}", self.next_id);
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let session = Session::new(id.clone(), user_id, now_ms);
        self.sessions.insert(id.clone(), session);
        id
    }

    /// Add an event to an existing session.
    ///
    /// Returns `true` if the session was found and the event was recorded.
    pub fn add_event(&mut self, session_id: &str, event: SessionEvent) -> bool {
        if let Some(session) = self.sessions.get_mut(session_id) {
            session.add_event(event);
            true
        } else {
            false
        }
    }

    /// Retrieve a session by ID.
    #[must_use]
    pub fn get_session(&self, session_id: &str) -> Option<&Session> {
        self.sessions.get(session_id)
    }

    /// Return the number of active sessions.
    #[must_use]
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }
}

// ---------------------------------------------------------------------------
// Session-based recommender
// ---------------------------------------------------------------------------

/// Recommends content based on in-session interactions.
///
/// Scores candidate items by summing recency-weighted event affinities for
/// items that share categories with already-interacted items.
#[derive(Debug, Default)]
#[allow(dead_code)]
pub struct SessionBasedRecommender {
    /// Category map: `media_id` → list of categories
    categories: HashMap<u64, Vec<String>>,
}

impl SessionBasedRecommender {
    /// Create a new recommender.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register category information for a media item.
    pub fn register_categories(&mut self, media_id: u64, cats: Vec<String>) {
        self.categories.insert(media_id, cats);
    }

    /// Recommend from `candidates`, returning up to `top_k` media IDs.
    ///
    /// Scoring logic:
    /// - For each event in the session, add the event's affinity weight
    ///   (scaled by a recency factor) to every candidate that shares at least
    ///   one category with the interacted item.
    /// - Already-interacted items are excluded from the results.
    #[must_use]
    pub fn recommend(&self, session: &Session, candidates: &[u64], top_k: usize) -> Vec<u64> {
        if candidates.is_empty() || session.events.is_empty() {
            return Vec::new();
        }

        let last_ts = session.last_activity_ms();
        let mut scores: HashMap<u64, f32> = HashMap::new();

        // Set of items the user already interacted with
        let interacted: std::collections::HashSet<u64> =
            session.events.iter().map(|e| e.media_id).collect();

        for event in &session.events {
            // Recency factor: events closer to the end of the session score higher
            let age_ms = last_ts.saturating_sub(event.timestamp_ms);
            let recency = 1.0 / (1.0 + age_ms as f32 / 60_000.0); // half-life ~1 min

            let weight = event.event_type.affinity_weight() * recency;

            // Find categories of the interacted item
            let event_cats = match self.categories.get(&event.media_id) {
                Some(c) => c,
                None => continue,
            };

            // Score candidates that share a category
            for &cand in candidates {
                if interacted.contains(&cand) {
                    continue;
                }
                if let Some(cand_cats) = self.categories.get(&cand) {
                    let shared = cand_cats.iter().any(|c| event_cats.contains(c));
                    if shared {
                        *scores.entry(cand).or_insert(0.0) += weight;
                    }
                }
            }
        }

        // Sort by score descending
        let mut ranked: Vec<(u64, f32)> = scores.into_iter().collect();
        ranked.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        ranked.into_iter().take(top_k).map(|(id, _)| id).collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(offset_s: u64) -> u64 {
        1_700_000_000_000 + offset_s * 1000
    }

    #[test]
    fn test_session_event_creation() {
        let e = SessionEvent::new(42, SessionEventType::Play, ts(0), Some(5000));
        assert_eq!(e.media_id, 42);
        assert_eq!(e.event_type, SessionEventType::Play);
        assert_eq!(e.duration_ms, Some(5000));
    }

    #[test]
    fn test_event_type_affinity_weights() {
        assert!(
            SessionEventType::Like.affinity_weight() > SessionEventType::Click.affinity_weight()
        );
        assert!(SessionEventType::Skip.affinity_weight() < 0.0);
        assert!(
            SessionEventType::Dislike.affinity_weight() < SessionEventType::Skip.affinity_weight()
        );
    }

    #[test]
    fn test_session_add_and_count() {
        let mut session = Session::new("s1", Some(1), ts(0));
        session.add_event(SessionEvent::new(
            10,
            SessionEventType::Play,
            ts(1),
            Some(3000),
        ));
        assert_eq!(session.events.len(), 1);
    }

    #[test]
    fn test_session_completion_rate_no_events() {
        let session = Session::new("s1", None, ts(0));
        assert_eq!(session.completion_rate(1), 0.0);
    }

    #[test]
    fn test_session_completion_rate_with_play() {
        let mut session = Session::new("s1", None, ts(0));
        session.add_event(SessionEvent::new(
            5,
            SessionEventType::Play,
            ts(1),
            Some(6000),
        ));
        // single play event: play / available = 1.0
        assert!((session.completion_rate(5) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_session_affinity_score_no_match() {
        let session = Session::new("s1", None, ts(0));
        let cats: HashMap<u64, Vec<String>> = HashMap::new();
        assert_eq!(session.affinity_score("drama", &cats), 0.0);
    }

    #[test]
    fn test_session_affinity_score_positive() {
        let mut session = Session::new("s1", None, ts(0));
        session.add_event(SessionEvent::new(10, SessionEventType::Like, ts(1), None));
        let mut cats: HashMap<u64, Vec<String>> = HashMap::new();
        cats.insert(10, vec!["comedy".to_string()]);
        let score = session.affinity_score("comedy", &cats);
        assert!(score > 0.0);
    }

    #[test]
    fn test_session_last_activity() {
        let mut session = Session::new("s1", None, ts(0));
        session.add_event(SessionEvent::new(1, SessionEventType::View, ts(5), None));
        session.add_event(SessionEvent::new(2, SessionEventType::Click, ts(10), None));
        assert_eq!(session.last_activity_ms(), ts(10));
    }

    #[test]
    fn test_session_store_create() {
        let mut store = SessionStore::new();
        let id = store.create_session();
        assert!(!id.is_empty());
        assert_eq!(store.session_count(), 1);
    }

    #[test]
    fn test_session_store_add_event() {
        let mut store = SessionStore::new();
        let id = store.create_session();
        let event = SessionEvent::new(99, SessionEventType::Click, ts(0), None);
        let ok = store.add_event(&id, event);
        assert!(ok);
        assert_eq!(
            store
                .get_session(&id)
                .expect("should succeed in test")
                .events
                .len(),
            1
        );
    }

    #[test]
    fn test_session_store_add_event_unknown() {
        let mut store = SessionStore::new();
        let event = SessionEvent::new(1, SessionEventType::View, ts(0), None);
        assert!(!store.add_event("no-such-session", event));
    }

    #[test]
    fn test_session_based_recommender() {
        let mut rec = SessionBasedRecommender::new();
        rec.register_categories(1, vec!["action".to_string()]);
        rec.register_categories(2, vec!["action".to_string()]);
        rec.register_categories(3, vec!["drama".to_string()]);

        let mut session = Session::new("s1", None, ts(0));
        session.add_event(SessionEvent::new(1, SessionEventType::Like, ts(1), None));

        let results = rec.recommend(&session, &[2, 3], 2);
        // Item 2 shares "action" with item 1 → should rank first
        assert!(!results.is_empty());
        assert_eq!(results[0], 2);
    }

    #[test]
    fn test_session_based_recommender_excludes_interacted() {
        let mut rec = SessionBasedRecommender::new();
        rec.register_categories(1, vec!["sci-fi".to_string()]);
        rec.register_categories(2, vec!["sci-fi".to_string()]);

        let mut session = Session::new("s1", None, ts(0));
        session.add_event(SessionEvent::new(
            1,
            SessionEventType::Play,
            ts(1),
            Some(5000),
        ));

        let results = rec.recommend(&session, &[1, 2], 5);
        assert!(!results.contains(&1)); // already interacted
    }
}
