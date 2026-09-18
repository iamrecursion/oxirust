//! Smart playlist generation based on configurable rules and constraints.

#![allow(dead_code)]

/// A rule used to filter or match playlist items.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayRule {
    /// Match by musical genre tag.
    ByGenre,
    /// Match by mood descriptor.
    ByMood,
    /// Match by beats-per-minute range.
    ByBpm,
    /// Match by artist identifier.
    ByArtist,
    /// Match by release year.
    ByYear,
    /// Match by user rating score.
    ByRating,
    /// Match by historical play count.
    ByPlayCount,
}

impl PlayRule {
    /// Returns `true` for rules that reflect explicit user preferences.
    pub fn is_user_preference(&self) -> bool {
        matches!(
            self,
            PlayRule::ByMood | PlayRule::ByRating | PlayRule::ByPlayCount
        )
    }
}

/// A constraint pairing a rule with an acceptable value range.
#[derive(Debug, Clone)]
pub struct PlayConstraint {
    /// Rule to apply.
    pub rule: PlayRule,
    /// Minimum acceptable value (inclusive).
    pub min_value: f32,
    /// Maximum acceptable value (inclusive).
    pub max_value: f32,
}

impl PlayConstraint {
    /// Returns `true` when `value` falls within `[min_value, max_value]`.
    pub fn is_satisfied(&self, value: f32) -> bool {
        value >= self.min_value && value <= self.max_value
    }
}

/// A smart playlist that selects items satisfying all constraints.
#[derive(Debug, Clone)]
pub struct SmartPlaylist {
    /// Human-readable playlist name.
    pub name: String,
    /// Constraints that items must satisfy.
    pub constraints: Vec<PlayConstraint>,
    /// Maximum number of items to include.
    pub max_items: u32,
    /// Whether to shuffle the output.
    pub shuffle: bool,
}

impl SmartPlaylist {
    /// Creates a new smart playlist with no constraints.
    pub fn new(name: impl Into<String>, max_items: u32) -> Self {
        Self {
            name: name.into(),
            constraints: Vec::new(),
            max_items,
            shuffle: false,
        }
    }

    /// Adds a constraint to the playlist.
    pub fn add_constraint(&mut self, constraint: PlayConstraint) {
        self.constraints.push(constraint);
    }

    /// Filters `items` (each is `(id, value)`) returning ids that satisfy all
    /// constraints, limited to `max_items`.
    ///
    /// Note: shuffling is intentionally deterministic in this pure function;
    /// callers should shuffle the result if `self.shuffle` is true.
    pub fn filter_items(&self, items: &[(u64, f32)]) -> Vec<u64> {
        let filtered: Vec<u64> = items
            .iter()
            .filter(|(_, value)| self.constraints.iter().all(|c| c.is_satisfied(*value)))
            .map(|(id, _)| *id)
            .take(self.max_items as usize)
            .collect();
        filtered
    }

    /// Returns the number of constraints currently registered.
    pub fn item_count(&self) -> usize {
        self.constraints.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- PlayRule ---

    #[test]
    fn test_by_mood_is_user_preference() {
        assert!(PlayRule::ByMood.is_user_preference());
    }

    #[test]
    fn test_by_rating_is_user_preference() {
        assert!(PlayRule::ByRating.is_user_preference());
    }

    #[test]
    fn test_by_play_count_is_user_preference() {
        assert!(PlayRule::ByPlayCount.is_user_preference());
    }

    #[test]
    fn test_by_genre_not_user_preference() {
        assert!(!PlayRule::ByGenre.is_user_preference());
    }

    #[test]
    fn test_by_bpm_not_user_preference() {
        assert!(!PlayRule::ByBpm.is_user_preference());
    }

    #[test]
    fn test_by_artist_not_user_preference() {
        assert!(!PlayRule::ByArtist.is_user_preference());
    }

    #[test]
    fn test_by_year_not_user_preference() {
        assert!(!PlayRule::ByYear.is_user_preference());
    }

    // --- PlayConstraint ---

    #[test]
    fn test_constraint_satisfied_within_range() {
        let c = PlayConstraint {
            rule: PlayRule::ByBpm,
            min_value: 120.0,
            max_value: 140.0,
        };
        assert!(c.is_satisfied(130.0));
    }

    #[test]
    fn test_constraint_satisfied_at_min() {
        let c = PlayConstraint {
            rule: PlayRule::ByBpm,
            min_value: 120.0,
            max_value: 140.0,
        };
        assert!(c.is_satisfied(120.0));
    }

    #[test]
    fn test_constraint_satisfied_at_max() {
        let c = PlayConstraint {
            rule: PlayRule::ByBpm,
            min_value: 120.0,
            max_value: 140.0,
        };
        assert!(c.is_satisfied(140.0));
    }

    #[test]
    fn test_constraint_not_satisfied_below() {
        let c = PlayConstraint {
            rule: PlayRule::ByBpm,
            min_value: 120.0,
            max_value: 140.0,
        };
        assert!(!c.is_satisfied(100.0));
    }

    #[test]
    fn test_constraint_not_satisfied_above() {
        let c = PlayConstraint {
            rule: PlayRule::ByBpm,
            min_value: 120.0,
            max_value: 140.0,
        };
        assert!(!c.is_satisfied(160.0));
    }

    // --- SmartPlaylist ---

    fn make_playlist() -> SmartPlaylist {
        let mut pl = SmartPlaylist::new("Workout", 10);
        pl.add_constraint(PlayConstraint {
            rule: PlayRule::ByBpm,
            min_value: 120.0,
            max_value: 160.0,
        });
        pl
    }

    #[test]
    fn test_item_count_after_add() {
        let pl = make_playlist();
        assert_eq!(pl.item_count(), 1);
    }

    #[test]
    fn test_filter_items_basic() {
        let pl = make_playlist();
        let items = vec![(1u64, 130.0), (2u64, 100.0), (3u64, 150.0)];
        let result = pl.filter_items(&items);
        assert_eq!(result, vec![1, 3]);
    }

    #[test]
    fn test_filter_items_max_items_limit() {
        let mut pl = SmartPlaylist::new("Limited", 2);
        pl.add_constraint(PlayConstraint {
            rule: PlayRule::ByRating,
            min_value: 0.0,
            max_value: 10.0,
        });
        let items: Vec<(u64, f32)> = (1..=5).map(|i| (i, 5.0)).collect();
        let result = pl.filter_items(&items);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_filter_items_empty_constraints() {
        let pl = SmartPlaylist::new("All", 100);
        let items = vec![(10u64, 0.0), (20u64, 999.0)];
        let result = pl.filter_items(&items);
        assert_eq!(result.len(), 2);
    }
}
