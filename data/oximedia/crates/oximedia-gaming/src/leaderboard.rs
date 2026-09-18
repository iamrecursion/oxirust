//! Gaming leaderboard management for `oximedia-gaming`.
//!
//! Provides a ranked list of player scores with support for different scopes
//! (global, regional, friends, weekly) and helper utilities for rank lookup.

#![allow(dead_code)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::module_name_repetitions)]

// ---------------------------------------------------------------------------
// LeaderboardEntry
// ---------------------------------------------------------------------------

/// A single ranked entry on a leaderboard.
#[derive(Debug, Clone)]
pub struct LeaderboardEntry {
    /// Unique player identifier.
    pub player_id: String,
    /// Score value (higher is better).
    pub score: u64,
    /// Rank on the board (1 = best, updated by [`Leaderboard::assign_ranks`]).
    pub rank: u32,
    /// When the score was achieved, in milliseconds since the Unix epoch.
    pub timestamp_ms: u64,
}

impl LeaderboardEntry {
    /// Create a new entry.  `rank` is set to `0` until ranks are assigned.
    pub fn new(player_id: &str, score: u64, timestamp_ms: u64) -> Self {
        Self {
            player_id: player_id.to_string(),
            score,
            rank: 0,
            timestamp_ms,
        }
    }

    /// Returns `true` when this entry's score exceeds `prev_best`.
    pub fn is_personal_best(&self, prev_best: u64) -> bool {
        self.score > prev_best
    }
}

// ---------------------------------------------------------------------------
// LeaderboardScope
// ---------------------------------------------------------------------------

/// Determines which players are eligible to appear on the leaderboard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeaderboardScope {
    /// All players across every region.
    Global,
    /// Players in the same geographic region.
    Regional,
    /// Only friends of the querying player.
    Friends,
    /// Resets on a weekly cadence.
    Weekly,
}

impl LeaderboardScope {
    /// Returns `true` when this scope has a time-bounded reset cycle.
    pub fn is_time_limited(self) -> bool {
        matches!(self, Self::Weekly)
    }
}

// ---------------------------------------------------------------------------
// Leaderboard
// ---------------------------------------------------------------------------

/// A ranked collection of player scores for a specific scope.
#[derive(Debug, Clone)]
pub struct Leaderboard {
    /// Scope of this leaderboard.
    pub scope: LeaderboardScope,
    /// Sorted entries (highest score first after [`assign_ranks`]).
    ///
    /// [`assign_ranks`]: Leaderboard::assign_ranks
    pub entries: Vec<LeaderboardEntry>,
    /// Maximum number of entries retained.
    pub max_entries: usize,
}

impl Leaderboard {
    /// Create a new leaderboard for the given `scope` with a capacity of
    /// `max_entries`.
    pub fn new(scope: LeaderboardScope, max_entries: usize) -> Self {
        Self {
            scope,
            entries: Vec::new(),
            max_entries,
        }
    }

    /// Submit a score for `player_id`.
    ///
    /// If the player already has an entry, the score is updated only when the
    /// new value is higher.  After insertion the entries are sorted and ranks
    /// re-assigned, and any entries beyond `max_entries` are pruned.
    pub fn submit(&mut self, player_id: &str, score: u64, now_ms: u64) {
        if let Some(existing) = self.entries.iter_mut().find(|e| e.player_id == player_id) {
            if score > existing.score {
                existing.score = score;
                existing.timestamp_ms = now_ms;
            }
        } else {
            self.entries
                .push(LeaderboardEntry::new(player_id, score, now_ms));
        }
        self.entries.sort_by(|a, b| b.score.cmp(&a.score));
        self.entries.truncate(self.max_entries);
        self.assign_ranks();
    }

    /// Returns the rank (1-based) of `player_id`, or `None` if not present.
    pub fn rank_of(&self, player_id: &str) -> Option<u32> {
        self.entries
            .iter()
            .find(|e| e.player_id == player_id)
            .map(|e| e.rank)
    }

    /// Returns references to the top `n` entries (fewer if the board is smaller).
    pub fn top(&self, n: usize) -> Vec<&LeaderboardEntry> {
        self.entries.iter().take(n).collect()
    }

    /// Re-assigns sequential 1-based ranks to every entry in their current
    /// order (assumed to be sorted highest-score-first).
    pub fn assign_ranks(&mut self) {
        for (i, entry) in self.entries.iter_mut().enumerate() {
            #[allow(clippy::cast_possible_truncation)]
            {
                entry.rank = (i + 1) as u32;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_lb() -> Leaderboard {
        Leaderboard::new(LeaderboardScope::Global, 10)
    }

    #[test]
    fn test_scope_is_time_limited_weekly() {
        assert!(LeaderboardScope::Weekly.is_time_limited());
    }

    #[test]
    fn test_scope_is_time_limited_global_false() {
        assert!(!LeaderboardScope::Global.is_time_limited());
    }

    #[test]
    fn test_scope_is_time_limited_regional_false() {
        assert!(!LeaderboardScope::Regional.is_time_limited());
    }

    #[test]
    fn test_scope_is_time_limited_friends_false() {
        assert!(!LeaderboardScope::Friends.is_time_limited());
    }

    #[test]
    fn test_entry_is_personal_best_true() {
        let e = LeaderboardEntry::new("p1", 1000, 0);
        assert!(e.is_personal_best(999));
    }

    #[test]
    fn test_entry_is_personal_best_false_equal() {
        let e = LeaderboardEntry::new("p1", 1000, 0);
        assert!(!e.is_personal_best(1000));
    }

    #[test]
    fn test_entry_is_personal_best_false_lower() {
        let e = LeaderboardEntry::new("p1", 500, 0);
        assert!(!e.is_personal_best(1000));
    }

    #[test]
    fn test_submit_single_player() {
        let mut lb = make_lb();
        lb.submit("alice", 500, 1_000);
        assert_eq!(lb.rank_of("alice"), Some(1));
    }

    #[test]
    fn test_submit_two_players_ordering() {
        let mut lb = make_lb();
        lb.submit("alice", 300, 1_000);
        lb.submit("bob", 500, 2_000);
        assert_eq!(lb.rank_of("bob"), Some(1));
        assert_eq!(lb.rank_of("alice"), Some(2));
    }

    #[test]
    fn test_submit_updates_score_higher() {
        let mut lb = make_lb();
        lb.submit("alice", 100, 1_000);
        lb.submit("alice", 900, 2_000);
        assert_eq!(lb.entries[0].score, 900);
    }

    #[test]
    fn test_submit_does_not_lower_score() {
        let mut lb = make_lb();
        lb.submit("alice", 900, 1_000);
        lb.submit("alice", 100, 2_000);
        assert_eq!(lb.entries[0].score, 900);
    }

    #[test]
    fn test_rank_of_absent() {
        let lb = make_lb();
        assert_eq!(lb.rank_of("nobody"), None);
    }

    #[test]
    fn test_top_n() {
        let mut lb = make_lb();
        lb.submit("a", 100, 0);
        lb.submit("b", 200, 0);
        lb.submit("c", 300, 0);
        let top2 = lb.top(2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].player_id, "c");
    }

    #[test]
    fn test_max_entries_respected() {
        let mut lb = Leaderboard::new(LeaderboardScope::Global, 3);
        lb.submit("a", 100, 0);
        lb.submit("b", 200, 0);
        lb.submit("c", 300, 0);
        lb.submit("d", 400, 0); // should push "a" out
        assert_eq!(lb.entries.len(), 3);
        assert!(lb.rank_of("a").is_none());
    }

    #[test]
    fn test_assign_ranks_sequential() {
        let mut lb = make_lb();
        lb.submit("a", 10, 0);
        lb.submit("b", 20, 0);
        lb.submit("c", 30, 0);
        assert_eq!(lb.rank_of("c"), Some(1));
        assert_eq!(lb.rank_of("b"), Some(2));
        assert_eq!(lb.rank_of("a"), Some(3));
    }
}
