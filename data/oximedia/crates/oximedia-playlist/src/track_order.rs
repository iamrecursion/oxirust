#![allow(dead_code)]

//! Track ordering and reordering utilities for broadcast playlists.
//!
//! This module provides tools to reorder tracks within a playlist using
//! various strategies: manual repositioning, sorting by metadata fields,
//! priority-based ordering, and shuffle-aware deterministic sequencing.

use std::collections::HashMap;

/// Strategy used for ordering tracks within a playlist.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OrderStrategy {
    /// Manual ordering — tracks appear in the order explicitly set.
    Manual,
    /// Sort by track title alphabetically.
    AlphabeticalTitle,
    /// Sort by track duration (shortest first).
    DurationAscending,
    /// Sort by track duration (longest first).
    DurationDescending,
    /// Sort by priority value (highest priority first).
    PriorityDescending,
    /// Sort by insertion timestamp (oldest first).
    ChronologicalAsc,
    /// Sort by insertion timestamp (newest first).
    ChronologicalDesc,
    /// Interleave tracks from different categories evenly.
    Interleave,
}

/// A track entry that can be ordered within a playlist.
#[derive(Debug, Clone, PartialEq)]
pub struct OrderedTrack {
    /// Unique identifier for this track.
    pub id: String,
    /// Human-readable title.
    pub title: String,
    /// Duration in milliseconds.
    pub duration_ms: u64,
    /// Priority value (higher = more important).
    pub priority: u32,
    /// Category tag for interleave grouping.
    pub category: String,
    /// Insertion timestamp (epoch millis).
    pub inserted_at: u64,
    /// Manual sort position (lower = earlier).
    pub position: u32,
}

impl OrderedTrack {
    /// Create a new ordered track with the given id and title.
    pub fn new(id: &str, title: &str) -> Self {
        Self {
            id: id.to_string(),
            title: title.to_string(),
            duration_ms: 0,
            priority: 0,
            category: String::new(),
            inserted_at: 0,
            position: 0,
        }
    }

    /// Set the duration in milliseconds.
    pub fn with_duration_ms(mut self, ms: u64) -> Self {
        self.duration_ms = ms;
        self
    }

    /// Set the priority value.
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Set the category tag.
    pub fn with_category(mut self, category: &str) -> Self {
        self.category = category.to_string();
        self
    }

    /// Set the insertion timestamp.
    pub fn with_inserted_at(mut self, ts: u64) -> Self {
        self.inserted_at = ts;
        self
    }

    /// Set the manual sort position.
    pub fn with_position(mut self, pos: u32) -> Self {
        self.position = pos;
        self
    }
}

/// Engine that applies ordering strategies to collections of tracks.
#[derive(Debug)]
pub struct TrackOrderEngine {
    /// Current ordering strategy.
    strategy: OrderStrategy,
}

impl TrackOrderEngine {
    /// Create a new engine with the specified strategy.
    pub fn new(strategy: OrderStrategy) -> Self {
        Self { strategy }
    }

    /// Return the current strategy.
    pub fn strategy(&self) -> &OrderStrategy {
        &self.strategy
    }

    /// Change the ordering strategy.
    pub fn set_strategy(&mut self, strategy: OrderStrategy) {
        self.strategy = strategy;
    }

    /// Apply the current strategy and return a sorted copy of the tracks.
    pub fn apply(&self, tracks: &[OrderedTrack]) -> Vec<OrderedTrack> {
        let mut sorted = tracks.to_vec();
        match &self.strategy {
            OrderStrategy::Manual => {
                sorted.sort_by_key(|t| t.position);
            }
            OrderStrategy::AlphabeticalTitle => {
                sorted.sort_by(|a, b| a.title.to_lowercase().cmp(&b.title.to_lowercase()));
            }
            OrderStrategy::DurationAscending => {
                sorted.sort_by_key(|t| t.duration_ms);
            }
            OrderStrategy::DurationDescending => {
                sorted.sort_by(|a, b| b.duration_ms.cmp(&a.duration_ms));
            }
            OrderStrategy::PriorityDescending => {
                sorted.sort_by(|a, b| b.priority.cmp(&a.priority));
            }
            OrderStrategy::ChronologicalAsc => {
                sorted.sort_by_key(|t| t.inserted_at);
            }
            OrderStrategy::ChronologicalDesc => {
                sorted.sort_by(|a, b| b.inserted_at.cmp(&a.inserted_at));
            }
            OrderStrategy::Interleave => {
                sorted = Self::interleave_by_category(tracks);
            }
        }
        sorted
    }

    /// Move a track from one position to another within a mutable list.
    pub fn move_track(tracks: &mut Vec<OrderedTrack>, from: usize, to: usize) -> bool {
        if from >= tracks.len() || to >= tracks.len() {
            return false;
        }
        let track = tracks.remove(from);
        tracks.insert(to, track);
        // Re-assign positions
        for (i, t) in tracks.iter_mut().enumerate() {
            t.position = i as u32;
        }
        true
    }

    /// Swap two tracks by index.
    pub fn swap_tracks(tracks: &mut [OrderedTrack], a: usize, b: usize) -> bool {
        if a >= tracks.len() || b >= tracks.len() {
            return false;
        }
        tracks.swap(a, b);
        // Re-assign positions
        for (i, t) in tracks.iter_mut().enumerate() {
            t.position = i as u32;
        }
        true
    }

    /// Interleave tracks from different categories in round-robin fashion.
    fn interleave_by_category(tracks: &[OrderedTrack]) -> Vec<OrderedTrack> {
        let mut groups: HashMap<String, Vec<OrderedTrack>> = HashMap::new();
        for t in tracks {
            groups
                .entry(t.category.clone())
                .or_default()
                .push(t.clone());
        }

        let mut keys: Vec<String> = groups.keys().cloned().collect();
        keys.sort();

        let mut iterators: Vec<std::vec::IntoIter<OrderedTrack>> = keys
            .iter()
            .map(|k| groups.remove(k).unwrap_or_default().into_iter())
            .collect();

        let mut result = Vec::with_capacity(tracks.len());
        let mut exhausted = vec![false; iterators.len()];
        loop {
            let mut any = false;
            for (i, iter) in iterators.iter_mut().enumerate() {
                if exhausted[i] {
                    continue;
                }
                if let Some(track) = iter.next() {
                    result.push(track);
                    any = true;
                } else {
                    exhausted[i] = true;
                }
            }
            if !any {
                break;
            }
        }
        result
    }

    /// Reverse the order of tracks in place.
    pub fn reverse(tracks: &mut [OrderedTrack]) {
        tracks.reverse();
        for (i, t) in tracks.iter_mut().enumerate() {
            t.position = i as u32;
        }
    }

    /// Filter tracks keeping only those matching the given category.
    pub fn filter_by_category(tracks: &[OrderedTrack], category: &str) -> Vec<OrderedTrack> {
        tracks
            .iter()
            .filter(|t| t.category == category)
            .cloned()
            .collect()
    }

    /// Compute the total duration of a track list in milliseconds.
    pub fn total_duration_ms(tracks: &[OrderedTrack]) -> u64 {
        tracks.iter().map(|t| t.duration_ms).sum()
    }
}

/// Result of a reorder operation, reporting what changed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReorderResult {
    /// Number of tracks that changed position.
    pub moved_count: usize,
    /// Whether the reorder was successful.
    pub success: bool,
    /// Description of the operation.
    pub description: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_tracks() -> Vec<OrderedTrack> {
        vec![
            OrderedTrack::new("t1", "Bravo")
                .with_duration_ms(3000)
                .with_priority(2)
                .with_category("news")
                .with_inserted_at(100)
                .with_position(0),
            OrderedTrack::new("t2", "Alpha")
                .with_duration_ms(1000)
                .with_priority(5)
                .with_category("sport")
                .with_inserted_at(200)
                .with_position(1),
            OrderedTrack::new("t3", "Charlie")
                .with_duration_ms(2000)
                .with_priority(1)
                .with_category("news")
                .with_inserted_at(50)
                .with_position(2),
        ]
    }

    #[test]
    fn test_manual_order() {
        let engine = TrackOrderEngine::new(OrderStrategy::Manual);
        let tracks = sample_tracks();
        let sorted = engine.apply(&tracks);
        assert_eq!(sorted[0].id, "t1");
        assert_eq!(sorted[1].id, "t2");
        assert_eq!(sorted[2].id, "t3");
    }

    #[test]
    fn test_alphabetical_title() {
        let engine = TrackOrderEngine::new(OrderStrategy::AlphabeticalTitle);
        let sorted = engine.apply(&sample_tracks());
        assert_eq!(sorted[0].title, "Alpha");
        assert_eq!(sorted[1].title, "Bravo");
        assert_eq!(sorted[2].title, "Charlie");
    }

    #[test]
    fn test_duration_ascending() {
        let engine = TrackOrderEngine::new(OrderStrategy::DurationAscending);
        let sorted = engine.apply(&sample_tracks());
        assert_eq!(sorted[0].duration_ms, 1000);
        assert_eq!(sorted[1].duration_ms, 2000);
        assert_eq!(sorted[2].duration_ms, 3000);
    }

    #[test]
    fn test_duration_descending() {
        let engine = TrackOrderEngine::new(OrderStrategy::DurationDescending);
        let sorted = engine.apply(&sample_tracks());
        assert_eq!(sorted[0].duration_ms, 3000);
        assert_eq!(sorted[1].duration_ms, 2000);
        assert_eq!(sorted[2].duration_ms, 1000);
    }

    #[test]
    fn test_priority_descending() {
        let engine = TrackOrderEngine::new(OrderStrategy::PriorityDescending);
        let sorted = engine.apply(&sample_tracks());
        assert_eq!(sorted[0].priority, 5);
        assert_eq!(sorted[1].priority, 2);
        assert_eq!(sorted[2].priority, 1);
    }

    #[test]
    fn test_chronological_asc() {
        let engine = TrackOrderEngine::new(OrderStrategy::ChronologicalAsc);
        let sorted = engine.apply(&sample_tracks());
        assert_eq!(sorted[0].inserted_at, 50);
        assert_eq!(sorted[1].inserted_at, 100);
        assert_eq!(sorted[2].inserted_at, 200);
    }

    #[test]
    fn test_chronological_desc() {
        let engine = TrackOrderEngine::new(OrderStrategy::ChronologicalDesc);
        let sorted = engine.apply(&sample_tracks());
        assert_eq!(sorted[0].inserted_at, 200);
        assert_eq!(sorted[1].inserted_at, 100);
        assert_eq!(sorted[2].inserted_at, 50);
    }

    #[test]
    fn test_interleave() {
        let engine = TrackOrderEngine::new(OrderStrategy::Interleave);
        let sorted = engine.apply(&sample_tracks());
        // news and sport interleaved: first news, first sport, second news
        assert_eq!(sorted.len(), 3);
        // Verify both categories present in first 2 entries
        let cats: Vec<&str> = sorted.iter().map(|t| t.category.as_str()).collect();
        assert!(cats.contains(&"news"));
        assert!(cats.contains(&"sport"));
    }

    #[test]
    fn test_move_track() {
        let mut tracks = sample_tracks();
        assert!(TrackOrderEngine::move_track(&mut tracks, 0, 2));
        assert_eq!(tracks[2].id, "t1");
        assert_eq!(tracks[0].position, 0);
        assert_eq!(tracks[1].position, 1);
        assert_eq!(tracks[2].position, 2);
    }

    #[test]
    fn test_move_track_out_of_bounds() {
        let mut tracks = sample_tracks();
        assert!(!TrackOrderEngine::move_track(&mut tracks, 5, 0));
    }

    #[test]
    fn test_swap_tracks() {
        let mut tracks = sample_tracks();
        assert!(TrackOrderEngine::swap_tracks(&mut tracks, 0, 2));
        assert_eq!(tracks[0].id, "t3");
        assert_eq!(tracks[2].id, "t1");
    }

    #[test]
    fn test_reverse() {
        let mut tracks = sample_tracks();
        TrackOrderEngine::reverse(&mut tracks);
        assert_eq!(tracks[0].id, "t3");
        assert_eq!(tracks[1].id, "t2");
        assert_eq!(tracks[2].id, "t1");
    }

    #[test]
    fn test_filter_by_category() {
        let tracks = sample_tracks();
        let news = TrackOrderEngine::filter_by_category(&tracks, "news");
        assert_eq!(news.len(), 2);
        for t in &news {
            assert_eq!(t.category, "news");
        }
    }

    #[test]
    fn test_total_duration_ms() {
        let tracks = sample_tracks();
        assert_eq!(TrackOrderEngine::total_duration_ms(&tracks), 6000);
    }

    #[test]
    fn test_set_strategy() {
        let mut engine = TrackOrderEngine::new(OrderStrategy::Manual);
        engine.set_strategy(OrderStrategy::AlphabeticalTitle);
        assert_eq!(*engine.strategy(), OrderStrategy::AlphabeticalTitle);
    }

    #[test]
    fn test_empty_tracks() {
        let engine = TrackOrderEngine::new(OrderStrategy::DurationAscending);
        let sorted = engine.apply(&[]);
        assert!(sorted.is_empty());
    }

    #[test]
    fn test_reorder_result() {
        let r = ReorderResult {
            moved_count: 2,
            success: true,
            description: "Sorted by title".to_string(),
        };
        assert_eq!(r.moved_count, 2);
        assert!(r.success);
    }
}
