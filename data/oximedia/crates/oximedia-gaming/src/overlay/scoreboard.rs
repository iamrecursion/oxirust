//! Game scoreboard overlay.

/// Scoreboard overlay.
#[allow(dead_code)]
pub struct Scoreboard {
    config: ScoreboardConfig,
    scores: Vec<(String, i32)>,
}

/// Scoreboard configuration.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ScoreboardConfig {
    /// Position (x, y)
    pub position: (i32, i32),
    /// Show player names
    pub show_names: bool,
    /// Show scores
    pub show_scores: bool,
}

impl Scoreboard {
    /// Create a new scoreboard.
    #[must_use]
    pub fn new(config: ScoreboardConfig) -> Self {
        Self {
            config,
            scores: Vec::new(),
        }
    }

    /// Update player score.
    pub fn update_score(&mut self, player: String, score: i32) {
        if let Some(entry) = self.scores.iter_mut().find(|(p, _)| p == &player) {
            entry.1 = score;
        } else {
            self.scores.push((player, score));
        }
    }

    /// Get score count.
    #[must_use]
    pub fn score_count(&self) -> usize {
        self.scores.len()
    }
}

impl Default for ScoreboardConfig {
    fn default() -> Self {
        Self {
            position: (10, 10),
            show_names: true,
            show_scores: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scoreboard_creation() {
        let scoreboard = Scoreboard::new(ScoreboardConfig::default());
        assert_eq!(scoreboard.score_count(), 0);
    }

    #[test]
    fn test_update_score() {
        let mut scoreboard = Scoreboard::new(ScoreboardConfig::default());
        scoreboard.update_score("Player1".to_string(), 100);
        assert_eq!(scoreboard.score_count(), 1);
    }
}
