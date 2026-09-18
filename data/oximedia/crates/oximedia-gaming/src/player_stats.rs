//! Player statistics tracking for game streaming sessions.
//!
//! Provides types for collecting, storing, and analysing per-player
//! performance metrics (kills, deaths, score, etc.) during a gaming session.

#![allow(dead_code)]

use std::collections::VecDeque;

/// A single measurable player statistic category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlayerStat {
    /// Number of enemy eliminations.
    Kills,
    /// Number of times the player was eliminated.
    Deaths,
    /// Player score within the match.
    Score,
    /// Number of assists recorded.
    Assists,
    /// Number of headshots (where applicable).
    Headshots,
    /// Damage dealt to opponents.
    DamageDealt,
    /// Win count for the session.
    Wins,
    /// Loss count for the session.
    Losses,
}

impl PlayerStat {
    /// Human-readable label for the statistic.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Kills => "Kills",
            Self::Deaths => "Deaths",
            Self::Score => "Score",
            Self::Assists => "Assists",
            Self::Headshots => "Headshots",
            Self::DamageDealt => "Damage Dealt",
            Self::Wins => "Wins",
            Self::Losses => "Losses",
        }
    }
}

/// A snapshot of player statistics at a point in time.
#[derive(Debug, Clone)]
pub struct PlayerStats {
    /// Player display name.
    pub player_name: String,
    /// Kills recorded.
    pub kills: u32,
    /// Deaths recorded.
    pub deaths: u32,
    /// Assists recorded.
    pub assists: u32,
    /// Score accumulated.
    pub score: u32,
    /// Headshots recorded.
    pub headshots: u32,
    /// Total damage dealt.
    pub damage_dealt: u32,
    /// Wins in the session.
    pub wins: u32,
    /// Losses in the session.
    pub losses: u32,
}

impl PlayerStats {
    /// Create a blank stats record for a player.
    #[must_use]
    pub fn new(player_name: impl Into<String>) -> Self {
        Self {
            player_name: player_name.into(),
            kills: 0,
            deaths: 0,
            assists: 0,
            score: 0,
            headshots: 0,
            damage_dealt: 0,
            wins: 0,
            losses: 0,
        }
    }

    /// Kill/Death ratio.  Returns `f64::INFINITY` when deaths == 0.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn kd_ratio(&self) -> f64 {
        if self.deaths == 0 {
            return f64::INFINITY;
        }
        f64::from(self.kills) / f64::from(self.deaths)
    }

    /// Headshot percentage (0–100).  Returns 0 when there are no kills.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn headshot_pct(&self) -> f64 {
        if self.kills == 0 {
            return 0.0;
        }
        f64::from(self.headshots) / f64::from(self.kills) * 100.0
    }

    /// Win-rate percentage (0–100).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn win_rate(&self) -> f64 {
        let total = self.wins + self.losses;
        if total == 0 {
            return 0.0;
        }
        f64::from(self.wins) / f64::from(total) * 100.0
    }

    /// Retrieve the raw value for any [`PlayerStat`] category.
    #[must_use]
    pub fn get(&self, stat: PlayerStat) -> u32 {
        match stat {
            PlayerStat::Kills => self.kills,
            PlayerStat::Deaths => self.deaths,
            PlayerStat::Score => self.score,
            PlayerStat::Assists => self.assists,
            PlayerStat::Headshots => self.headshots,
            PlayerStat::DamageDealt => self.damage_dealt,
            PlayerStat::Wins => self.wins,
            PlayerStat::Losses => self.losses,
        }
    }
}

/// Tracks rolling statistics across multiple matches or rounds.
pub struct StatsTracker {
    player_name: String,
    /// Sliding window of recent kills (per match/round).
    kill_history: VecDeque<u32>,
    /// Sliding window of recent deaths.
    death_history: VecDeque<u32>,
    /// Sliding window of recent scores.
    score_history: VecDeque<u32>,
    /// Maximum number of entries retained in each window.
    window_size: usize,
    /// Cumulative totals across all tracked matches.
    totals: PlayerStats,
}

impl StatsTracker {
    /// Create a new tracker with the specified rolling-window size.
    ///
    /// # Panics
    ///
    /// Panics if `window_size` is zero.
    #[must_use]
    pub fn new(player_name: impl Into<String>, window_size: usize) -> Self {
        assert!(window_size > 0, "window_size must be at least 1");
        let name = player_name.into();
        Self {
            player_name: name.clone(),
            kill_history: VecDeque::with_capacity(window_size),
            death_history: VecDeque::with_capacity(window_size),
            score_history: VecDeque::with_capacity(window_size),
            window_size,
            totals: PlayerStats::new(name),
        }
    }

    /// Record a completed match result.
    pub fn record_match(&mut self, kills: u32, deaths: u32, score: u32, won: bool) {
        // Maintain the rolling window.
        if self.kill_history.len() == self.window_size {
            self.kill_history.pop_front();
            self.death_history.pop_front();
            self.score_history.pop_front();
        }
        self.kill_history.push_back(kills);
        self.death_history.push_back(deaths);
        self.score_history.push_back(score);

        // Accumulate totals.
        self.totals.kills = self.totals.kills.saturating_add(kills);
        self.totals.deaths = self.totals.deaths.saturating_add(deaths);
        self.totals.score = self.totals.score.saturating_add(score);
        if won {
            self.totals.wins = self.totals.wins.saturating_add(1);
        } else {
            self.totals.losses = self.totals.losses.saturating_add(1);
        }
    }

    /// Rolling average kills over the recent window.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn rolling_average(&self, stat: PlayerStat) -> f64 {
        let history: &VecDeque<u32> = match stat {
            PlayerStat::Kills => &self.kill_history,
            PlayerStat::Deaths => &self.death_history,
            PlayerStat::Score => &self.score_history,
            _ => return 0.0,
        };
        if history.is_empty() {
            return 0.0;
        }
        let sum: u64 = history.iter().map(|&v| u64::from(v)).sum();
        sum as f64 / history.len() as f64
    }

    /// Number of matches recorded in the current window.
    #[must_use]
    pub fn window_len(&self) -> usize {
        self.kill_history.len()
    }

    /// Cumulative statistics across all recorded matches.
    #[must_use]
    pub fn totals(&self) -> &PlayerStats {
        &self.totals
    }

    /// Player name associated with this tracker.
    #[must_use]
    pub fn player_name(&self) -> &str {
        &self.player_name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- PlayerStat ---

    #[test]
    fn test_player_stat_labels() {
        assert_eq!(PlayerStat::Kills.label(), "Kills");
        assert_eq!(PlayerStat::Deaths.label(), "Deaths");
        assert_eq!(PlayerStat::Score.label(), "Score");
        assert_eq!(PlayerStat::Assists.label(), "Assists");
        assert_eq!(PlayerStat::Headshots.label(), "Headshots");
        assert_eq!(PlayerStat::DamageDealt.label(), "Damage Dealt");
        assert_eq!(PlayerStat::Wins.label(), "Wins");
        assert_eq!(PlayerStat::Losses.label(), "Losses");
    }

    // --- PlayerStats ---

    #[test]
    fn test_new_stats_are_zeroed() {
        let s = PlayerStats::new("Alice");
        assert_eq!(s.kills, 0);
        assert_eq!(s.deaths, 0);
        assert_eq!(s.score, 0);
    }

    #[test]
    fn test_kd_ratio_normal() {
        let mut s = PlayerStats::new("Bob");
        s.kills = 10;
        s.deaths = 5;
        let ratio = s.kd_ratio();
        assert!((ratio - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_kd_ratio_zero_deaths() {
        let mut s = PlayerStats::new("Ace");
        s.kills = 5;
        s.deaths = 0;
        assert!(s.kd_ratio().is_infinite());
    }

    #[test]
    fn test_headshot_pct_normal() {
        let mut s = PlayerStats::new("Sniper");
        s.kills = 10;
        s.headshots = 7;
        let pct = s.headshot_pct();
        assert!((pct - 70.0).abs() < 1e-6);
    }

    #[test]
    fn test_headshot_pct_no_kills() {
        let s = PlayerStats::new("Rookie");
        assert!((s.headshot_pct() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_win_rate_normal() {
        let mut s = PlayerStats::new("Champion");
        s.wins = 3;
        s.losses = 1;
        assert!((s.win_rate() - 75.0).abs() < 1e-6);
    }

    #[test]
    fn test_win_rate_no_games() {
        let s = PlayerStats::new("Fresh");
        assert!((s.win_rate() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_get_retrieves_correct_field() {
        let mut s = PlayerStats::new("Getter");
        s.kills = 9;
        s.score = 1500;
        s.wins = 4;
        assert_eq!(s.get(PlayerStat::Kills), 9);
        assert_eq!(s.get(PlayerStat::Score), 1500);
        assert_eq!(s.get(PlayerStat::Wins), 4);
    }

    // --- StatsTracker ---

    #[test]
    fn test_tracker_empty_rolling_average() {
        let t = StatsTracker::new("Empty", 5);
        assert!((t.rolling_average(PlayerStat::Kills) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_tracker_record_and_average() {
        let mut t = StatsTracker::new("Pro", 5);
        t.record_match(10, 2, 500, true);
        t.record_match(8, 3, 400, false);
        let avg_kills = t.rolling_average(PlayerStat::Kills);
        assert!((avg_kills - 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_tracker_window_eviction() {
        let mut t = StatsTracker::new("Slider", 3);
        t.record_match(1, 0, 100, true);
        t.record_match(2, 0, 200, true);
        t.record_match(3, 0, 300, true);
        // Window is full; adding one more evicts the first entry (kills=1).
        t.record_match(4, 0, 400, true);
        assert_eq!(t.window_len(), 3);
        // Average over {2, 3, 4} = 3.0
        let avg = t.rolling_average(PlayerStat::Kills);
        assert!((avg - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_tracker_cumulative_totals() {
        let mut t = StatsTracker::new("Total", 10);
        t.record_match(5, 2, 300, true);
        t.record_match(3, 1, 150, false);
        let totals = t.totals();
        assert_eq!(totals.kills, 8);
        assert_eq!(totals.deaths, 3);
        assert_eq!(totals.score, 450);
        assert_eq!(totals.wins, 1);
        assert_eq!(totals.losses, 1);
    }

    #[test]
    fn test_tracker_player_name() {
        let t = StatsTracker::new("NamedPlayer", 5);
        assert_eq!(t.player_name(), "NamedPlayer");
    }

    #[test]
    fn test_rolling_average_unsupported_stat_returns_zero() {
        let mut t = StatsTracker::new("Misc", 5);
        t.record_match(5, 2, 300, true);
        // Assists is not tracked in history — returns 0.
        assert!((t.rolling_average(PlayerStat::Assists) - 0.0).abs() < 1e-9);
    }
}
