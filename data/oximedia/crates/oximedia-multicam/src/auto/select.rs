//! Automatic camera angle selection.

use super::{AngleScoreCache, RuleEngine, SelectionCriteria, SwitchingRule};
use crate::auto::score::{AngleScore, AngleScorer};
use crate::{AngleId, FrameNumber, Result};

/// Automatic camera switcher
#[derive(Debug)]
pub struct AutoSwitcher {
    /// Rule engine
    rule_engine: RuleEngine,
    /// Angle scorer
    scorer: AngleScorer,
    /// Score cache — avoids re-computing scores for unchanged frames.
    score_cache: AngleScoreCache,
    /// Currently selected angle
    current_angle: AngleId,
    /// Selection history
    history: Vec<SelectionRecord>,
    /// Minimum angle hold time (frames)
    min_hold_frames: u32,
    /// Last selection frame
    last_selection_frame: FrameNumber,
}

/// Selection record
#[derive(Debug, Clone)]
pub struct SelectionRecord {
    /// Frame number
    pub frame: FrameNumber,
    /// Selected angle
    pub angle: AngleId,
    /// Selection scores for all angles
    pub scores: Vec<AngleScore>,
    /// Final confidence
    pub confidence: f32,
}

impl AutoSwitcher {
    /// Default capacity for the angle score cache.
    const DEFAULT_CACHE_CAPACITY: usize = 1024;

    /// Create a new auto switcher
    #[must_use]
    pub fn new() -> Self {
        Self {
            rule_engine: RuleEngine::new(),
            scorer: AngleScorer::new(),
            score_cache: AngleScoreCache::new(Self::DEFAULT_CACHE_CAPACITY),
            current_angle: 0,
            history: Vec::new(),
            min_hold_frames: 50, // 2 seconds at 25fps
            last_selection_frame: 0,
        }
    }

    /// Add switching rule
    pub fn add_rule(&mut self, rule: SwitchingRule) {
        self.rule_engine.add_rule(rule);
    }

    /// Set minimum hold time
    pub fn set_min_hold_frames(&mut self, frames: u32) {
        self.min_hold_frames = frames;
    }

    /// Select best angle for current frame
    ///
    /// # Errors
    ///
    /// Returns an error if selection fails
    pub fn select_angle(
        &mut self,
        frame: FrameNumber,
        angle_count: usize,
        criteria: &SelectionCriteria,
    ) -> Result<AngleId> {
        // Check if we should switch (respect hold time)
        if frame < self.last_selection_frame + u64::from(self.min_hold_frames) {
            return Ok(self.current_angle);
        }

        // Score all angles, using the cache to skip recomputation when the
        // frame content has not changed.
        let mut scores = Vec::new();
        for angle in 0..angle_count {
            let total = if let Some(cached) = self.score_cache.get(angle, frame) {
                // Cache hit: build a score struct from the stored total.
                let mut s = AngleScore::new(angle);
                s.total_score = cached;
                s
            } else {
                // Cache miss: compute fresh and store the total for reuse.
                let s = self.scorer.score_angle(angle, criteria)?;
                self.score_cache.insert(angle, frame, s.total_score);
                s
            };
            scores.push(total);
        }

        // Apply rules
        let modified_scores = self
            .rule_engine
            .apply_rules(&scores, self.current_angle, frame);

        // Find best angle
        let best = modified_scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.total_score
                    .partial_cmp(&b.total_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map_or(self.current_angle, |(idx, _)| idx);

        let confidence = modified_scores[best].total_score;

        // Only switch if confidence is high enough
        if confidence >= criteria.min_confidence && best != self.current_angle {
            self.current_angle = best;
            self.last_selection_frame = frame;
        }

        // Record selection
        self.history.push(SelectionRecord {
            frame,
            angle: self.current_angle,
            scores: modified_scores,
            confidence,
        });

        Ok(self.current_angle)
    }

    /// Get current angle
    #[must_use]
    pub fn current_angle(&self) -> AngleId {
        self.current_angle
    }

    /// Get selection history
    #[must_use]
    pub fn history(&self) -> &[SelectionRecord] {
        &self.history
    }

    /// Clear history
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Get angle usage statistics
    #[must_use]
    pub fn angle_usage(&self) -> Vec<(AngleId, usize)> {
        let mut usage: std::collections::HashMap<AngleId, usize> = std::collections::HashMap::new();

        for record in &self.history {
            *usage.entry(record.angle).or_insert(0) += 1;
        }

        let mut result: Vec<_> = usage.into_iter().collect();
        result.sort_by_key(|&(_, count)| std::cmp::Reverse(count));
        result
    }

    /// Get average confidence
    #[must_use]
    pub fn average_confidence(&self) -> f32 {
        if self.history.is_empty() {
            return 0.0;
        }

        let sum: f32 = self.history.iter().map(|r| r.confidence).sum();
        sum / self.history.len() as f32
    }

    /// Get scorer
    #[must_use]
    pub fn scorer(&self) -> &AngleScorer {
        &self.scorer
    }

    /// Get mutable scorer
    pub fn scorer_mut(&mut self) -> &mut AngleScorer {
        &mut self.scorer
    }

    /// Get rule engine
    #[must_use]
    pub fn rule_engine(&self) -> &RuleEngine {
        &self.rule_engine
    }

    /// Get mutable rule engine
    pub fn rule_engine_mut(&mut self) -> &mut RuleEngine {
        &mut self.rule_engine
    }

    /// Reset to initial state
    pub fn reset(&mut self, initial_angle: AngleId) {
        self.current_angle = initial_angle;
        self.last_selection_frame = 0;
        self.history.clear();
    }

    /// Borrow the angle score cache (read-only).
    #[must_use]
    pub fn score_cache(&self) -> &AngleScoreCache {
        &self.score_cache
    }

    /// Borrow the angle score cache mutably.
    pub fn score_cache_mut(&mut self) -> &mut AngleScoreCache {
        &mut self.score_cache
    }
}

impl Default for AutoSwitcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Smart switcher with learning capability
#[derive(Debug)]
pub struct SmartSwitcher {
    /// Base auto switcher
    switcher: AutoSwitcher,
    /// Learning rate (0.0 to 1.0)
    learning_rate: f32,
    /// Angle preferences (learned)
    preferences: Vec<f32>,
}

impl SmartSwitcher {
    /// Create a new smart switcher
    #[must_use]
    pub fn new(angle_count: usize) -> Self {
        Self {
            switcher: AutoSwitcher::new(),
            learning_rate: 0.1,
            preferences: vec![0.5; angle_count],
        }
    }

    /// Set learning rate
    pub fn set_learning_rate(&mut self, rate: f32) {
        self.learning_rate = rate.clamp(0.0, 1.0);
    }

    /// Select angle with learning
    ///
    /// # Errors
    ///
    /// Returns an error if selection fails
    pub fn select_with_learning(
        &mut self,
        frame: FrameNumber,
        criteria: &SelectionCriteria,
    ) -> Result<AngleId> {
        let angle = self
            .switcher
            .select_angle(frame, self.preferences.len(), criteria)?;

        // Update preferences based on selection
        self.update_preferences(angle);

        Ok(angle)
    }

    /// Update angle preferences
    fn update_preferences(&mut self, selected_angle: AngleId) {
        if selected_angle < self.preferences.len() {
            // Increase preference for selected angle
            self.preferences[selected_angle] +=
                self.learning_rate * (1.0 - self.preferences[selected_angle]);

            // Decrease preference for other angles slightly
            for (i, pref) in self.preferences.iter_mut().enumerate() {
                if i != selected_angle {
                    *pref *= 1.0 - self.learning_rate * 0.1;
                }
            }
        }
    }

    /// Get angle preferences
    #[must_use]
    pub fn preferences(&self) -> &[f32] {
        &self.preferences
    }

    /// Reset preferences
    pub fn reset_preferences(&mut self) {
        for pref in &mut self.preferences {
            *pref = 0.5;
        }
    }

    /// Get inner switcher
    #[must_use]
    pub fn switcher(&self) -> &AutoSwitcher {
        &self.switcher
    }

    /// Get mutable inner switcher
    pub fn switcher_mut(&mut self) -> &mut AutoSwitcher {
        &mut self.switcher
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_switcher_creation() {
        let switcher = AutoSwitcher::new();
        assert_eq!(switcher.current_angle(), 0);
        assert_eq!(switcher.min_hold_frames, 50);
    }

    #[test]
    fn test_set_min_hold_frames() {
        let mut switcher = AutoSwitcher::new();
        switcher.set_min_hold_frames(100);
        assert_eq!(switcher.min_hold_frames, 100);
    }

    #[test]
    fn test_smart_switcher() {
        let switcher = SmartSwitcher::new(3);
        assert_eq!(switcher.preferences().len(), 3);
        assert_eq!(switcher.preferences()[0], 0.5);
    }

    #[test]
    fn test_learning() {
        let mut switcher = SmartSwitcher::new(3);
        switcher.set_learning_rate(0.2);

        switcher.update_preferences(1);
        assert!(switcher.preferences()[1] > 0.5);
        assert!(switcher.preferences()[0] < 0.5);
    }

    #[test]
    fn test_reset_preferences() {
        let mut switcher = SmartSwitcher::new(3);
        switcher.update_preferences(1);
        switcher.reset_preferences();

        for pref in switcher.preferences() {
            assert_eq!(*pref, 0.5);
        }
    }
}
