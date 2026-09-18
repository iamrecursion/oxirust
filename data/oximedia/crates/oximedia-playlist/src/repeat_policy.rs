#![allow(dead_code)]

//! Repeat and looping policies for playlists.
//!
//! Controls how playlists loop, shuffle on repeat, and schedule
//! recurring playback windows for broadcast automation.

use std::time::Duration;

/// How a playlist should repeat.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepeatMode {
    /// Play once and stop.
    None,
    /// Loop the entire playlist indefinitely.
    LoopAll,
    /// Loop only the current item.
    LoopOne,
    /// Loop a fixed number of times.
    LoopCount(u32),
    /// Loop until a wall-clock deadline (seconds since epoch).
    LoopUntil(u64),
}

/// Shuffle behavior on each repeat cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShuffleOnRepeat {
    /// Keep the same order every cycle.
    KeepOrder,
    /// Shuffle items at the start of each new cycle.
    ShuffleEachCycle,
    /// Shuffle once and then keep that order.
    ShuffleOnce,
}

/// Policy governing playlist repetition.
#[derive(Debug, Clone)]
pub struct RepeatPolicy {
    /// The repeat mode.
    pub mode: RepeatMode,
    /// Shuffle behavior.
    pub shuffle: ShuffleOnRepeat,
    /// Minimum gap between repeats.
    pub inter_cycle_gap: Duration,
    /// Maximum total play time across all repeats (0 = unlimited).
    pub max_total_duration: Duration,
    /// Whether to fade out at end of final cycle.
    pub fade_out_on_end: bool,
}

impl Default for RepeatPolicy {
    fn default() -> Self {
        Self {
            mode: RepeatMode::None,
            shuffle: ShuffleOnRepeat::KeepOrder,
            inter_cycle_gap: Duration::ZERO,
            max_total_duration: Duration::ZERO,
            fade_out_on_end: false,
        }
    }
}

impl RepeatPolicy {
    /// Create a simple non-repeating policy.
    pub fn once() -> Self {
        Self::default()
    }

    /// Create an infinite loop policy.
    pub fn loop_forever() -> Self {
        Self {
            mode: RepeatMode::LoopAll,
            ..Self::default()
        }
    }

    /// Create a counted loop policy.
    pub fn loop_n(count: u32) -> Self {
        Self {
            mode: RepeatMode::LoopCount(count),
            ..Self::default()
        }
    }

    /// Whether the policy allows any repeat.
    pub fn will_repeat(&self) -> bool {
        !matches!(self.mode, RepeatMode::None)
    }

    /// Set the shuffle behavior.
    pub fn with_shuffle(mut self, shuffle: ShuffleOnRepeat) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Set the inter-cycle gap.
    pub fn with_gap(mut self, gap: Duration) -> Self {
        self.inter_cycle_gap = gap;
        self
    }

    /// Set the maximum total duration.
    pub fn with_max_duration(mut self, max: Duration) -> Self {
        self.max_total_duration = max;
        self
    }

    /// Enable fade-out at end of final cycle.
    pub fn with_fade_out(mut self) -> Self {
        self.fade_out_on_end = true;
        self
    }
}

/// Tracks the state of repeat scheduling.
#[derive(Debug, Clone)]
pub struct RepeatScheduler {
    policy: RepeatPolicy,
    current_cycle: u32,
    total_played: Duration,
    is_finished: bool,
}

impl RepeatScheduler {
    /// Create a new scheduler from a policy.
    pub fn new(policy: RepeatPolicy) -> Self {
        Self {
            policy,
            current_cycle: 0,
            total_played: Duration::ZERO,
            is_finished: false,
        }
    }

    /// Current cycle number (0-based).
    pub fn current_cycle(&self) -> u32 {
        self.current_cycle
    }

    /// Whether playback is finished.
    pub fn is_finished(&self) -> bool {
        self.is_finished
    }

    /// Total time played so far.
    pub fn total_played(&self) -> Duration {
        self.total_played
    }

    /// Record that one cycle of the given duration completed.
    /// Returns `true` if another cycle should start.
    pub fn complete_cycle(&mut self, cycle_duration: Duration) -> bool {
        if self.is_finished {
            return false;
        }

        self.total_played += cycle_duration;
        self.current_cycle += 1;

        // Check max total duration
        if self.policy.max_total_duration > Duration::ZERO
            && self.total_played >= self.policy.max_total_duration
        {
            self.is_finished = true;
            return false;
        }

        match self.policy.mode {
            RepeatMode::None => {
                self.is_finished = true;
                false
            }
            RepeatMode::LoopAll | RepeatMode::LoopOne => true,
            RepeatMode::LoopCount(n) => {
                if self.current_cycle >= n {
                    self.is_finished = true;
                    false
                } else {
                    true
                }
            }
            RepeatMode::LoopUntil(deadline) => {
                let now = self.total_played.as_secs();
                if now >= deadline {
                    self.is_finished = true;
                    false
                } else {
                    true
                }
            }
        }
    }

    /// Duration to wait before starting the next cycle.
    pub fn next_gap(&self) -> Duration {
        if self.is_finished {
            Duration::ZERO
        } else {
            self.policy.inter_cycle_gap
        }
    }

    /// Whether the scheduler wants shuffle for the upcoming cycle.
    pub fn should_shuffle_next(&self) -> bool {
        match self.policy.shuffle {
            ShuffleOnRepeat::KeepOrder => false,
            ShuffleOnRepeat::ShuffleEachCycle => true,
            ShuffleOnRepeat::ShuffleOnce => self.current_cycle == 0,
        }
    }

    /// Reset the scheduler back to initial state.
    pub fn reset(&mut self) {
        self.current_cycle = 0;
        self.total_played = Duration::ZERO;
        self.is_finished = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repeat_mode_none() {
        let policy = RepeatPolicy::once();
        assert!(!policy.will_repeat());
    }

    #[test]
    fn test_repeat_mode_loop_all() {
        let policy = RepeatPolicy::loop_forever();
        assert!(policy.will_repeat());
        assert_eq!(policy.mode, RepeatMode::LoopAll);
    }

    #[test]
    fn test_repeat_mode_loop_n() {
        let policy = RepeatPolicy::loop_n(3);
        assert!(policy.will_repeat());
        assert_eq!(policy.mode, RepeatMode::LoopCount(3));
    }

    #[test]
    fn test_builder_with_shuffle() {
        let policy = RepeatPolicy::loop_forever().with_shuffle(ShuffleOnRepeat::ShuffleEachCycle);
        assert_eq!(policy.shuffle, ShuffleOnRepeat::ShuffleEachCycle);
    }

    #[test]
    fn test_builder_with_gap() {
        let policy = RepeatPolicy::loop_forever().with_gap(Duration::from_secs(5));
        assert_eq!(policy.inter_cycle_gap, Duration::from_secs(5));
    }

    #[test]
    fn test_scheduler_once_stops_after_one() {
        let mut sched = RepeatScheduler::new(RepeatPolicy::once());
        assert!(!sched.is_finished());
        let again = sched.complete_cycle(Duration::from_secs(60));
        assert!(!again);
        assert!(sched.is_finished());
    }

    #[test]
    fn test_scheduler_loop_all_continues() {
        let mut sched = RepeatScheduler::new(RepeatPolicy::loop_forever());
        let again = sched.complete_cycle(Duration::from_secs(30));
        assert!(again);
        assert!(!sched.is_finished());
    }

    #[test]
    fn test_scheduler_loop_count() {
        let mut sched = RepeatScheduler::new(RepeatPolicy::loop_n(2));
        assert!(sched.complete_cycle(Duration::from_secs(10)));
        assert!(!sched.complete_cycle(Duration::from_secs(10)));
        assert!(sched.is_finished());
        assert_eq!(sched.current_cycle(), 2);
    }

    #[test]
    fn test_scheduler_max_duration_cap() {
        let policy = RepeatPolicy::loop_forever().with_max_duration(Duration::from_secs(100));
        let mut sched = RepeatScheduler::new(policy);
        assert!(sched.complete_cycle(Duration::from_secs(50)));
        assert!(!sched.complete_cycle(Duration::from_secs(60))); // total 110 >= 100
        assert!(sched.is_finished());
    }

    #[test]
    fn test_scheduler_total_played() {
        let mut sched = RepeatScheduler::new(RepeatPolicy::loop_forever());
        sched.complete_cycle(Duration::from_secs(10));
        sched.complete_cycle(Duration::from_secs(20));
        assert_eq!(sched.total_played(), Duration::from_secs(30));
    }

    #[test]
    fn test_scheduler_gap() {
        let policy = RepeatPolicy::loop_forever().with_gap(Duration::from_secs(3));
        let sched = RepeatScheduler::new(policy);
        assert_eq!(sched.next_gap(), Duration::from_secs(3));
    }

    #[test]
    fn test_shuffle_keep_order() {
        let policy = RepeatPolicy::loop_forever().with_shuffle(ShuffleOnRepeat::KeepOrder);
        let sched = RepeatScheduler::new(policy);
        assert!(!sched.should_shuffle_next());
    }

    #[test]
    fn test_shuffle_each_cycle() {
        let policy = RepeatPolicy::loop_forever().with_shuffle(ShuffleOnRepeat::ShuffleEachCycle);
        let mut sched = RepeatScheduler::new(policy);
        assert!(sched.should_shuffle_next());
        sched.complete_cycle(Duration::from_secs(10));
        assert!(sched.should_shuffle_next());
    }

    #[test]
    fn test_shuffle_once() {
        let policy = RepeatPolicy::loop_forever().with_shuffle(ShuffleOnRepeat::ShuffleOnce);
        let mut sched = RepeatScheduler::new(policy);
        assert!(sched.should_shuffle_next()); // cycle 0
        sched.complete_cycle(Duration::from_secs(10));
        assert!(!sched.should_shuffle_next()); // cycle 1
    }

    #[test]
    fn test_scheduler_reset() {
        let mut sched = RepeatScheduler::new(RepeatPolicy::loop_n(2));
        sched.complete_cycle(Duration::from_secs(10));
        sched.complete_cycle(Duration::from_secs(10));
        assert!(sched.is_finished());
        sched.reset();
        assert!(!sched.is_finished());
        assert_eq!(sched.current_cycle(), 0);
        assert_eq!(sched.total_played(), Duration::ZERO);
    }

    #[test]
    fn test_fade_out_builder() {
        let policy = RepeatPolicy::once().with_fade_out();
        assert!(policy.fade_out_on_end);
    }

    #[test]
    fn test_finished_scheduler_gap_zero() {
        let mut sched = RepeatScheduler::new(RepeatPolicy::once().with_gap(Duration::from_secs(5)));
        sched.complete_cycle(Duration::from_secs(10));
        assert!(sched.is_finished());
        assert_eq!(sched.next_gap(), Duration::ZERO);
    }
}
