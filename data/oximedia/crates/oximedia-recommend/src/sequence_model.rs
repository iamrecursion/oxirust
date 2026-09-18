#![allow(dead_code)]
//! Sequential recommendation modeling.
//!
//! Models user behavior as ordered sequences of interactions to predict the
//! next item a user is likely to engage with. Supports Markov chain transition
//! models and windowed sequence patterns.

use std::collections::HashMap;

/// Represents a single interaction event in a sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct InteractionEvent {
    /// Item identifier.
    pub item_id: String,
    /// Timestamp of the interaction (milliseconds since epoch).
    pub timestamp_ms: u64,
    /// Interaction type.
    pub interaction_type: InteractionType,
    /// Engagement duration in milliseconds (e.g., watch time).
    pub duration_ms: Option<u64>,
}

impl InteractionEvent {
    /// Create a new interaction event.
    #[must_use]
    pub fn new(item_id: &str, timestamp_ms: u64, interaction_type: InteractionType) -> Self {
        Self {
            item_id: item_id.to_string(),
            timestamp_ms,
            interaction_type,
            duration_ms: None,
        }
    }

    /// Set the engagement duration.
    #[must_use]
    pub fn with_duration(mut self, duration_ms: u64) -> Self {
        self.duration_ms = Some(duration_ms);
        self
    }
}

/// Type of user interaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InteractionType {
    /// User viewed/watched the item.
    View,
    /// User clicked the item.
    Click,
    /// User rated the item.
    Rate,
    /// User added to favorites/bookmarks.
    Favorite,
    /// User shared the item.
    Share,
    /// User skipped the item.
    Skip,
}

/// A user interaction sequence.
#[derive(Debug, Clone)]
pub struct UserSequence {
    /// User identifier.
    pub user_id: String,
    /// Ordered list of interaction events (chronological).
    pub events: Vec<InteractionEvent>,
}

impl UserSequence {
    /// Create a new empty user sequence.
    #[must_use]
    pub fn new(user_id: &str) -> Self {
        Self {
            user_id: user_id.to_string(),
            events: Vec::new(),
        }
    }

    /// Append an event to the sequence.
    pub fn push(&mut self, event: InteractionEvent) {
        self.events.push(event);
    }

    /// Return the number of events in the sequence.
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Return whether the sequence is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Get the last N events (or fewer if the sequence is shorter).
    #[must_use]
    pub fn last_n(&self, n: usize) -> &[InteractionEvent] {
        let start = self.events.len().saturating_sub(n);
        &self.events[start..]
    }

    /// Extract item IDs in order.
    #[must_use]
    pub fn item_ids(&self) -> Vec<&str> {
        self.events.iter().map(|e| e.item_id.as_str()).collect()
    }

    /// Get unique item IDs interacted with.
    #[must_use]
    pub fn unique_items(&self) -> Vec<&str> {
        let mut seen = std::collections::HashSet::new();
        self.events
            .iter()
            .filter(|e| seen.insert(e.item_id.as_str()))
            .map(|e| e.item_id.as_str())
            .collect()
    }
}

/// Transition probability from one item to another.
#[derive(Debug, Clone)]
pub struct Transition {
    /// Source item ID.
    pub from: String,
    /// Destination item ID.
    pub to: String,
    /// Transition probability (0.0 to 1.0).
    pub probability: f64,
    /// Number of observed transitions.
    pub count: u64,
}

/// First-order Markov chain transition model.
pub struct MarkovModel {
    /// Transition counts: `from_item` -> (`to_item` -> count).
    transitions: HashMap<String, HashMap<String, u64>>,
    /// Total outgoing transitions per item.
    totals: HashMap<String, u64>,
}

impl MarkovModel {
    /// Create a new empty Markov model.
    #[must_use]
    pub fn new() -> Self {
        Self {
            transitions: HashMap::new(),
            totals: HashMap::new(),
        }
    }

    /// Train the model on a user sequence.
    pub fn train(&mut self, sequence: &UserSequence) {
        let items = sequence.item_ids();
        for window in items.windows(2) {
            let from = window[0].to_string();
            let to = window[1].to_string();
            *self
                .transitions
                .entry(from.clone())
                .or_default()
                .entry(to)
                .or_insert(0) += 1;
            *self.totals.entry(from).or_insert(0) += 1;
        }
    }

    /// Predict the top-k next items given the current item.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn predict(&self, current_item: &str, k: usize) -> Vec<Transition> {
        let Some(next_map) = self.transitions.get(current_item) else {
            return Vec::new();
        };
        let total = *self.totals.get(current_item).unwrap_or(&1);

        let mut predictions: Vec<Transition> = next_map
            .iter()
            .map(|(to, &count)| Transition {
                from: current_item.to_string(),
                to: to.clone(),
                probability: count as f64 / total as f64,
                count,
            })
            .collect();
        predictions.sort_by(|a, b| {
            b.probability
                .partial_cmp(&a.probability)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        predictions.truncate(k);
        predictions
    }

    /// Return the number of unique source items in the model.
    #[must_use]
    pub fn source_count(&self) -> usize {
        self.transitions.len()
    }

    /// Get the transition probability from one item to another.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn transition_probability(&self, from: &str, to: &str) -> f64 {
        let Some(next_map) = self.transitions.get(from) else {
            return 0.0;
        };
        let count = *next_map.get(to).unwrap_or(&0);
        let total = *self.totals.get(from).unwrap_or(&1);
        if total == 0 {
            0.0
        } else {
            count as f64 / total as f64
        }
    }
}

impl Default for MarkovModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Windowed sequence pattern detector.
///
/// Finds common sub-sequences of a given window size and uses them to
/// predict the next item.
pub struct WindowedPredictor {
    /// Window size for n-gram extraction.
    window_size: usize,
    /// Pattern counts: (`item_sequence`) -> (`next_item` -> count).
    patterns: HashMap<Vec<String>, HashMap<String, u64>>,
    /// Total counts per pattern.
    pattern_totals: HashMap<Vec<String>, u64>,
}

impl WindowedPredictor {
    /// Create a new windowed predictor with the given window size.
    #[must_use]
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size: window_size.max(1),
            patterns: HashMap::new(),
            pattern_totals: HashMap::new(),
        }
    }

    /// Train on a user sequence.
    pub fn train(&mut self, sequence: &UserSequence) {
        let items = sequence.item_ids();
        if items.len() <= self.window_size {
            return;
        }
        for window in items.windows(self.window_size + 1) {
            let prefix: Vec<String> = window[..self.window_size]
                .iter()
                .map(|s| (*s).to_string())
                .collect();
            let next = window[self.window_size].to_string();
            *self
                .patterns
                .entry(prefix.clone())
                .or_default()
                .entry(next)
                .or_insert(0) += 1;
            *self.pattern_totals.entry(prefix).or_insert(0) += 1;
        }
    }

    /// Predict the next item given a context window.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn predict(&self, context: &[&str], k: usize) -> Vec<(String, f64)> {
        let key: Vec<String> = context.iter().map(|s| (*s).to_string()).collect();
        let Some(next_map) = self.patterns.get(&key) else {
            return Vec::new();
        };
        let total = *self.pattern_totals.get(&key).unwrap_or(&1);

        let mut preds: Vec<(String, f64)> = next_map
            .iter()
            .map(|(item, &count)| (item.clone(), count as f64 / total as f64))
            .collect();
        preds.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        preds.truncate(k);
        preds
    }

    /// Return the number of unique patterns learned.
    #[must_use]
    pub fn pattern_count(&self) -> usize {
        self.patterns.len()
    }

    /// Return the configured window size.
    #[must_use]
    pub fn window_size(&self) -> usize {
        self.window_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sequence(user: &str, items: &[&str]) -> UserSequence {
        let mut seq = UserSequence::new(user);
        for (i, item) in items.iter().enumerate() {
            seq.push(InteractionEvent::new(
                item,
                (i as u64) * 1000,
                InteractionType::View,
            ));
        }
        seq
    }

    #[test]
    fn test_user_sequence_basics() {
        let seq = make_sequence("user1", &["a", "b", "c"]);
        assert_eq!(seq.len(), 3);
        assert!(!seq.is_empty());
        assert_eq!(seq.item_ids(), vec!["a", "b", "c"]);
    }

    #[test]
    fn test_user_sequence_last_n() {
        let seq = make_sequence("user1", &["a", "b", "c", "d", "e"]);
        let last3 = seq.last_n(3);
        assert_eq!(last3.len(), 3);
        assert_eq!(last3[0].item_id, "c");
    }

    #[test]
    fn test_user_sequence_unique_items() {
        let seq = make_sequence("user1", &["a", "b", "a", "c", "b"]);
        let unique = seq.unique_items();
        assert_eq!(unique.len(), 3);
    }

    #[test]
    fn test_interaction_event_with_duration() {
        let evt = InteractionEvent::new("item1", 1000, InteractionType::View).with_duration(5000);
        assert_eq!(evt.duration_ms, Some(5000));
    }

    #[test]
    fn test_markov_model_train_and_predict() {
        let mut model = MarkovModel::new();
        let seq = make_sequence("u1", &["a", "b", "c", "a", "b", "d"]);
        model.train(&seq);

        let preds = model.predict("a", 5);
        assert!(!preds.is_empty());
        // "a" -> "b" should be the most probable
        assert_eq!(preds[0].to, "b");
    }

    #[test]
    fn test_markov_model_transition_probability() {
        let mut model = MarkovModel::new();
        let seq = make_sequence("u1", &["a", "b", "a", "b", "a", "c"]);
        model.train(&seq);

        let p_ab = model.transition_probability("a", "b");
        let p_ac = model.transition_probability("a", "c");
        assert!(p_ab > p_ac);
        assert!((p_ab + p_ac - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_markov_model_unknown_item() {
        let model = MarkovModel::new();
        let preds = model.predict("unknown", 5);
        assert!(preds.is_empty());
    }

    #[test]
    fn test_markov_model_source_count() {
        let mut model = MarkovModel::new();
        let seq = make_sequence("u1", &["x", "y", "z"]);
        model.train(&seq);
        assert_eq!(model.source_count(), 2); // x->y and y->z
    }

    #[test]
    fn test_windowed_predictor_train_and_predict() {
        let mut predictor = WindowedPredictor::new(2);
        let seq = make_sequence("u1", &["a", "b", "c", "a", "b", "d"]);
        predictor.train(&seq);

        let preds = predictor.predict(&["a", "b"], 5);
        assert!(!preds.is_empty());
        // "a","b" -> "c" and "a","b" -> "d" both observed
        assert_eq!(preds.len(), 2);
    }

    #[test]
    fn test_windowed_predictor_unknown_context() {
        let predictor = WindowedPredictor::new(2);
        let preds = predictor.predict(&["x", "y"], 5);
        assert!(preds.is_empty());
    }

    #[test]
    fn test_windowed_predictor_pattern_count() {
        let mut predictor = WindowedPredictor::new(2);
        let seq = make_sequence("u1", &["a", "b", "c", "d"]);
        predictor.train(&seq);
        assert_eq!(predictor.pattern_count(), 2); // [a,b]->c and [b,c]->d
    }

    #[test]
    fn test_windowed_predictor_window_size() {
        let predictor = WindowedPredictor::new(3);
        assert_eq!(predictor.window_size(), 3);
    }

    #[test]
    fn test_empty_sequence() {
        let seq = UserSequence::new("empty");
        assert!(seq.is_empty());
        assert_eq!(seq.len(), 0);
        assert!(seq.item_ids().is_empty());
        assert!(seq.unique_items().is_empty());
    }
}
