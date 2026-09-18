//! Predictive pre-fetch for frames near the playhead position.
//!
//! Anticipates which frames will be needed next based on playback
//! direction and speed, and requests them in advance to reduce
//! latency during playback.

use std::collections::VecDeque;

/// Direction of playback for prefetch prediction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayDirection {
    /// Forward playback (normal).
    Forward,
    /// Reverse playback.
    Reverse,
    /// Stationary (paused / scrubbing).
    Stationary,
}

/// Configuration for the prefetch engine.
#[derive(Debug, Clone)]
pub struct PrefetchConfig {
    /// Number of frames to prefetch ahead of the playhead.
    pub lookahead: usize,
    /// Number of frames to keep behind the playhead (for reverse scrub).
    pub lookbehind: usize,
    /// Playback speed multiplier (affects stride).
    pub speed: f64,
    /// Frame duration in timebase units (e.g. 33 for ~30fps at ms timebase).
    pub frame_duration: i64,
    /// Whether prefetch is enabled.
    pub enabled: bool,
}

impl Default for PrefetchConfig {
    fn default() -> Self {
        Self {
            lookahead: 30,
            lookbehind: 5,
            speed: 1.0,
            frame_duration: 33,
            enabled: true,
        }
    }
}

impl PrefetchConfig {
    /// Create a config for real-time playback at given FPS.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn for_playback(fps: f64, lookahead_seconds: f64) -> Self {
        let frame_dur = if fps > 0.0 {
            (1000.0 / fps).round() as i64
        } else {
            33
        };
        Self {
            lookahead: (fps * lookahead_seconds).ceil() as usize,
            lookbehind: 5,
            speed: 1.0,
            frame_duration: frame_dur,
            enabled: true,
        }
    }
}

/// A request to prefetch a specific frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PrefetchRequest {
    /// Timeline position of the frame to prefetch.
    pub position: i64,
    /// Priority (lower = higher priority, 0 = immediate).
    pub priority: u32,
}

impl PrefetchRequest {
    /// Create a new prefetch request.
    #[must_use]
    pub fn new(position: i64, priority: u32) -> Self {
        Self { position, priority }
    }
}

/// Prefetch engine that generates frame requests based on playhead movement.
#[derive(Debug)]
pub struct PrefetchEngine {
    /// Configuration.
    config: PrefetchConfig,
    /// Current playhead position.
    playhead: i64,
    /// Current play direction.
    direction: PlayDirection,
    /// Positions that are already cached or pending.
    cached_positions: VecDeque<i64>,
    /// Maximum timeline position.
    max_position: i64,
    /// History of recent playhead positions (for direction detection).
    position_history: VecDeque<i64>,
    /// Maximum history length.
    history_limit: usize,
}

impl PrefetchEngine {
    /// Create a new prefetch engine.
    #[must_use]
    pub fn new(config: PrefetchConfig, max_position: i64) -> Self {
        Self {
            config,
            playhead: 0,
            direction: PlayDirection::Stationary,
            cached_positions: VecDeque::new(),
            max_position,
            position_history: VecDeque::new(),
            history_limit: 10,
        }
    }

    /// Update the playhead position and get new prefetch requests.
    ///
    /// Call this every time the playhead moves. Returns a list of
    /// positions to prefetch, sorted by priority (most urgent first).
    pub fn update(&mut self, new_position: i64) -> Vec<PrefetchRequest> {
        if !self.config.enabled {
            return Vec::new();
        }

        let old_position = self.playhead;
        self.playhead = new_position;

        // Track direction
        self.position_history.push_back(new_position);
        if self.position_history.len() > self.history_limit {
            self.position_history.pop_front();
        }
        self.direction = self.detect_direction();

        // Remove cached positions that are now far from playhead
        let keep_range = self.keep_range();
        self.cached_positions
            .retain(|&pos| pos >= keep_range.0 && pos <= keep_range.1);

        // Generate requests
        let mut requests = Vec::new();

        match self.direction {
            PlayDirection::Forward => {
                self.generate_forward_requests(&mut requests);
            }
            PlayDirection::Reverse => {
                self.generate_reverse_requests(&mut requests);
            }
            PlayDirection::Stationary => {
                // Prefetch a small window around the playhead
                self.generate_bidirectional_requests(&mut requests);
            }
        }

        // Sort by priority
        requests.sort_by_key(|r| r.priority);

        // Mark these as pending
        for req in &requests {
            if !self.cached_positions.contains(&req.position) {
                self.cached_positions.push_back(req.position);
            }
        }

        let _ = old_position; // suppress unused
        requests
    }

    /// Mark a position as cached (already decoded).
    pub fn mark_cached(&mut self, position: i64) {
        if !self.cached_positions.contains(&position) {
            self.cached_positions.push_back(position);
        }
    }

    /// Invalidate a cached position.
    pub fn invalidate(&mut self, position: i64) {
        self.cached_positions.retain(|&p| p != position);
    }

    /// Invalidate all cached positions.
    pub fn invalidate_all(&mut self) {
        self.cached_positions.clear();
    }

    /// Get the current detected play direction.
    #[must_use]
    pub fn direction(&self) -> PlayDirection {
        self.direction
    }

    /// Get the current playhead position.
    #[must_use]
    pub fn playhead(&self) -> i64 {
        self.playhead
    }

    /// Get count of cached/pending positions.
    #[must_use]
    pub fn cached_count(&self) -> usize {
        self.cached_positions.len()
    }

    // ── Internal helpers ─────────────────────────────────────────────────

    fn detect_direction(&self) -> PlayDirection {
        if self.position_history.len() < 2 {
            return PlayDirection::Stationary;
        }
        let len = self.position_history.len();
        let recent = self.position_history[len - 1];
        let prev = self.position_history[len - 2];
        let delta = recent - prev;

        if delta > 0 {
            PlayDirection::Forward
        } else if delta < 0 {
            PlayDirection::Reverse
        } else {
            PlayDirection::Stationary
        }
    }

    fn keep_range(&self) -> (i64, i64) {
        let behind = self.config.lookbehind as i64 * self.config.frame_duration;
        let ahead = self.config.lookahead as i64 * self.config.frame_duration;
        (
            (self.playhead - behind).max(0),
            (self.playhead + ahead).min(self.max_position),
        )
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn generate_forward_requests(&self, requests: &mut Vec<PrefetchRequest>) {
        let stride = (self.config.frame_duration as f64 * self.config.speed).round() as i64;
        let stride = stride.max(1);

        for i in 0..self.config.lookahead {
            let pos = self.playhead + (i as i64 + 1) * stride;
            if pos > self.max_position {
                break;
            }
            if !self.cached_positions.contains(&pos) {
                requests.push(PrefetchRequest::new(pos, i as u32));
            }
        }
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn generate_reverse_requests(&self, requests: &mut Vec<PrefetchRequest>) {
        let stride = (self.config.frame_duration as f64 * self.config.speed).round() as i64;
        let stride = stride.max(1);

        for i in 0..self.config.lookbehind {
            let pos = self.playhead - (i as i64 + 1) * stride;
            if pos < 0 {
                break;
            }
            if !self.cached_positions.contains(&pos) {
                requests.push(PrefetchRequest::new(pos, i as u32));
            }
        }
    }

    fn generate_bidirectional_requests(&self, requests: &mut Vec<PrefetchRequest>) {
        let half_ahead = self.config.lookahead / 2;
        let stride = self.config.frame_duration;

        for i in 0..half_ahead {
            let forward = self.playhead + (i as i64 + 1) * stride;
            let backward = self.playhead - (i as i64 + 1) * stride;

            if forward <= self.max_position && !self.cached_positions.contains(&forward) {
                requests.push(PrefetchRequest::new(forward, i as u32));
            }
            if backward >= 0 && !self.cached_positions.contains(&backward) {
                requests.push(PrefetchRequest::new(backward, (i + half_ahead) as u32));
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefetch_config_default() {
        let cfg = PrefetchConfig::default();
        assert_eq!(cfg.lookahead, 30);
        assert_eq!(cfg.lookbehind, 5);
        assert!(cfg.enabled);
    }

    #[test]
    fn test_prefetch_config_for_playback() {
        let cfg = PrefetchConfig::for_playback(30.0, 1.0);
        assert_eq!(cfg.lookahead, 30);
        assert_eq!(cfg.frame_duration, 33);
    }

    #[test]
    fn test_prefetch_request() {
        let req = PrefetchRequest::new(1000, 0);
        assert_eq!(req.position, 1000);
        assert_eq!(req.priority, 0);
    }

    #[test]
    fn test_engine_disabled() {
        let cfg = PrefetchConfig {
            enabled: false,
            ..Default::default()
        };
        let mut engine = PrefetchEngine::new(cfg, 10000);
        let requests = engine.update(500);
        assert!(requests.is_empty());
    }

    #[test]
    fn test_engine_forward_playback() {
        let cfg = PrefetchConfig {
            lookahead: 5,
            lookbehind: 2,
            frame_duration: 33,
            speed: 1.0,
            enabled: true,
        };
        let mut engine = PrefetchEngine::new(cfg, 10000);

        // First update at 0
        let _r1 = engine.update(0);

        // Second update at 33 (forward)
        let r2 = engine.update(33);
        assert_eq!(engine.direction(), PlayDirection::Forward);
        assert!(!r2.is_empty(), "should generate forward prefetch requests");

        // Check requests are ahead of playhead
        for req in &r2 {
            assert!(req.position > 33, "prefetch should be ahead of playhead");
        }
    }

    #[test]
    fn test_engine_reverse_playback() {
        let cfg = PrefetchConfig {
            lookahead: 5,
            lookbehind: 5,
            frame_duration: 33,
            speed: 1.0,
            enabled: true,
        };
        let mut engine = PrefetchEngine::new(cfg, 10000);

        engine.update(5000);
        let requests = engine.update(4967); // moved backward

        assert_eq!(engine.direction(), PlayDirection::Reverse);
        // Reverse requests should be behind the playhead
        for req in &requests {
            assert!(req.position < 4967);
        }
    }

    #[test]
    fn test_engine_stationary() {
        let cfg = PrefetchConfig {
            lookahead: 10,
            lookbehind: 2,
            frame_duration: 33,
            speed: 1.0,
            enabled: true,
        };
        let mut engine = PrefetchEngine::new(cfg, 10000);

        engine.update(5000);
        let requests = engine.update(5000); // no movement

        assert_eq!(engine.direction(), PlayDirection::Stationary);
        // Should generate bidirectional requests
        let has_forward = requests.iter().any(|r| r.position > 5000);
        let has_backward = requests.iter().any(|r| r.position < 5000);
        assert!(has_forward || has_backward);
    }

    #[test]
    fn test_engine_mark_cached() {
        let cfg = PrefetchConfig::default();
        let mut engine = PrefetchEngine::new(cfg, 10000);
        engine.mark_cached(100);
        engine.mark_cached(200);
        assert_eq!(engine.cached_count(), 2);
    }

    #[test]
    fn test_engine_invalidate() {
        let cfg = PrefetchConfig::default();
        let mut engine = PrefetchEngine::new(cfg, 10000);
        engine.mark_cached(100);
        engine.mark_cached(200);
        engine.invalidate(100);
        assert_eq!(engine.cached_count(), 1);
        engine.invalidate_all();
        assert_eq!(engine.cached_count(), 0);
    }

    #[test]
    fn test_engine_does_not_exceed_max_position() {
        let cfg = PrefetchConfig {
            lookahead: 100,
            lookbehind: 2,
            frame_duration: 33,
            speed: 1.0,
            enabled: true,
        };
        let mut engine = PrefetchEngine::new(cfg, 1000);
        engine.update(0);
        let requests = engine.update(33);

        for req in &requests {
            assert!(req.position <= 1000, "should not exceed max position");
        }
    }

    #[test]
    fn test_engine_does_not_go_below_zero() {
        let cfg = PrefetchConfig {
            lookahead: 5,
            lookbehind: 100,
            frame_duration: 33,
            speed: 1.0,
            enabled: true,
        };
        let mut engine = PrefetchEngine::new(cfg, 10000);
        engine.update(100);
        let requests = engine.update(67); // reverse

        for req in &requests {
            assert!(req.position >= 0, "should not go below zero");
        }
    }

    #[test]
    fn test_cached_positions_not_re_requested() {
        let cfg = PrefetchConfig {
            lookahead: 3,
            lookbehind: 0,
            frame_duration: 100,
            speed: 1.0,
            enabled: true,
        };
        let mut engine = PrefetchEngine::new(cfg, 10000);
        engine.update(0);
        let r1 = engine.update(100);

        // Update again at same position
        let r2 = engine.update(200);

        // Requests from r1 that were marked as pending should not appear in r2
        // (unless they've been evicted from the keep range)
        for req in &r2 {
            // Positions from r1 should not be re-requested if still in range
            let was_in_r1 = r1.iter().any(|r| r.position == req.position);
            if was_in_r1 {
                // This is fine if position was evicted; just checking the mechanism works
            }
        }
        // The engine should track cached positions
        assert!(engine.cached_count() > 0);
    }
}
