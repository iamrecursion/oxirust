#![allow(dead_code)]
//! Automatic gap detection and filler content insertion for playout.
//!
//! When a broadcast schedule has gaps between programme items (or an item
//! under-runs), this module provides strategies for selecting and inserting
//! filler content so the output never goes to black or silence.

use std::collections::VecDeque;
use std::fmt;

// ---------------------------------------------------------------------------
// Filler strategy
// ---------------------------------------------------------------------------

/// Strategy for filling schedule gaps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FillerStrategy {
    /// Loop a single designated filler clip.
    LoopSingle,
    /// Play from a pool of short filler clips (promos, idents).
    PoolRotation,
    /// Hold the last frame of the previous item (freeze frame).
    FreezeFrame,
    /// Output colour bars and tone.
    ColourBars,
    /// Output black and silence.
    BlackSilence,
}

impl fmt::Display for FillerStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::LoopSingle => "Loop Single",
            Self::PoolRotation => "Pool Rotation",
            Self::FreezeFrame => "Freeze Frame",
            Self::ColourBars => "Colour Bars",
            Self::BlackSilence => "Black & Silence",
        };
        write!(f, "{s}")
    }
}

// ---------------------------------------------------------------------------
// Filler clip
// ---------------------------------------------------------------------------

/// A filler clip available for insertion.
#[derive(Debug, Clone)]
pub struct FillerClip {
    /// Unique identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Duration in seconds.
    pub duration_sec: f64,
    /// File path or URI.
    pub path: String,
    /// Priority (higher = preferred).
    pub priority: u32,
    /// Number of times this clip has been used in the current day.
    pub usage_count: u32,
    /// Maximum uses per day (0 = unlimited).
    pub max_uses_per_day: u32,
}

impl FillerClip {
    /// Whether the clip can still be used today.
    pub fn available(&self) -> bool {
        self.max_uses_per_day == 0 || self.usage_count < self.max_uses_per_day
    }

    /// Effective priority (adjusted for usage to avoid repetition).
    #[allow(clippy::cast_precision_loss)]
    pub fn effective_priority(&self) -> f64 {
        let base = self.priority as f64;
        let penalty = self.usage_count as f64 * 0.5;
        (base - penalty).max(0.0)
    }
}

// ---------------------------------------------------------------------------
// Gap
// ---------------------------------------------------------------------------

/// Represents a detected gap in the schedule.
#[derive(Debug, Clone)]
pub struct ScheduleGap {
    /// Start of the gap (seconds since midnight).
    pub start_sec: f64,
    /// End of the gap (seconds since midnight).
    pub end_sec: f64,
    /// Channel name.
    pub channel: String,
}

impl ScheduleGap {
    /// Duration of the gap in seconds.
    pub fn duration(&self) -> f64 {
        if self.end_sec >= self.start_sec {
            self.end_sec - self.start_sec
        } else {
            // Wraps past midnight
            86400.0 - self.start_sec + self.end_sec
        }
    }
}

impl fmt::Display for ScheduleGap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Gap[{}: {:.1}s-{:.1}s ({:.1}s)]",
            self.channel,
            self.start_sec,
            self.end_sec,
            self.duration()
        )
    }
}

// ---------------------------------------------------------------------------
// Fill plan
// ---------------------------------------------------------------------------

/// A planned filler insertion to cover a gap.
#[derive(Debug, Clone)]
pub struct FillPlanItem {
    /// Clip to play.
    pub clip_id: String,
    /// Start time (seconds since midnight).
    pub start_sec: f64,
    /// Duration of the clip to use (may be trimmed).
    pub play_duration_sec: f64,
    /// Whether the clip should loop.
    pub looping: bool,
}

/// A complete fill plan for a single gap.
#[derive(Debug, Clone)]
pub struct FillPlan {
    /// The gap being filled.
    pub gap: ScheduleGap,
    /// Ordered list of clips to play.
    pub items: Vec<FillPlanItem>,
    /// Strategy used.
    pub strategy: FillerStrategy,
    /// Whether the gap is fully covered.
    pub fully_covered: bool,
}

impl FillPlan {
    /// Total planned duration.
    pub fn total_duration(&self) -> f64 {
        self.items.iter().map(|i| i.play_duration_sec).sum()
    }

    /// Number of items in the plan.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether the plan has no items.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Filler pool
// ---------------------------------------------------------------------------

/// A pool of available filler clips.
#[derive(Debug, Clone)]
pub struct FillerPool {
    /// Available clips.
    pub clips: Vec<FillerClip>,
    /// Recently used clip ids (for round-robin / anti-repetition).
    pub recent: VecDeque<String>,
    /// Maximum recent history to track.
    pub recent_limit: usize,
}

impl FillerPool {
    /// Create a new empty filler pool.
    pub fn new(recent_limit: usize) -> Self {
        Self {
            clips: Vec::new(),
            recent: VecDeque::new(),
            recent_limit,
        }
    }

    /// Add a clip to the pool.
    pub fn add_clip(&mut self, clip: FillerClip) {
        self.clips.push(clip);
    }

    /// Select the best clip that fits within a given duration.
    pub fn select_best(&self, max_duration_sec: f64) -> Option<&FillerClip> {
        self.clips
            .iter()
            .filter(|c| c.available() && c.duration_sec <= max_duration_sec)
            .max_by(|a, b| {
                a.effective_priority()
                    .partial_cmp(&b.effective_priority())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Select the longest clip that fits within a given duration.
    pub fn select_longest(&self, max_duration_sec: f64) -> Option<&FillerClip> {
        self.clips
            .iter()
            .filter(|c| c.available() && c.duration_sec <= max_duration_sec)
            .max_by(|a, b| {
                a.duration_sec
                    .partial_cmp(&b.duration_sec)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Mark a clip as used.
    pub fn mark_used(&mut self, clip_id: &str) {
        if let Some(clip) = self.clips.iter_mut().find(|c| c.id == clip_id) {
            clip.usage_count += 1;
        }
        self.recent.push_back(clip_id.to_string());
        if self.recent.len() > self.recent_limit {
            self.recent.pop_front();
        }
    }

    /// Reset daily usage counts.
    pub fn reset_daily(&mut self) {
        for clip in &mut self.clips {
            clip.usage_count = 0;
        }
        self.recent.clear();
    }

    /// Number of clips in the pool.
    pub fn len(&self) -> usize {
        self.clips.len()
    }

    /// Whether the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.clips.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Gap filler engine
// ---------------------------------------------------------------------------

/// Configuration for the gap filler engine.
#[derive(Debug, Clone)]
pub struct GapFillerConfig {
    /// Primary fill strategy.
    pub strategy: FillerStrategy,
    /// Fallback strategy if primary fails.
    pub fallback: FillerStrategy,
    /// Minimum gap duration (seconds) to fill; shorter gaps are ignored.
    pub min_gap_sec: f64,
    /// Whether to trim filler to exact gap length.
    pub trim_to_fit: bool,
    /// Maximum clips to chain in one gap.
    pub max_chain_length: usize,
}

impl Default for GapFillerConfig {
    fn default() -> Self {
        Self {
            strategy: FillerStrategy::PoolRotation,
            fallback: FillerStrategy::ColourBars,
            min_gap_sec: 1.0,
            trim_to_fit: true,
            max_chain_length: 20,
        }
    }
}

/// Gap filler engine.
#[derive(Debug, Clone)]
pub struct GapFiller {
    /// Configuration.
    pub config: GapFillerConfig,
    /// Pool of filler clips.
    pub pool: FillerPool,
}

impl GapFiller {
    /// Create a new gap filler.
    pub fn new(config: GapFillerConfig, pool: FillerPool) -> Self {
        Self { config, pool }
    }

    /// Plan how to fill a single gap.
    pub fn plan_fill(&self, gap: &ScheduleGap) -> FillPlan {
        let gap_duration = gap.duration();

        if gap_duration < self.config.min_gap_sec {
            return FillPlan {
                gap: gap.clone(),
                items: Vec::new(),
                strategy: self.config.strategy,
                fully_covered: false,
            };
        }

        match self.config.strategy {
            FillerStrategy::LoopSingle => self.plan_loop_single(gap, gap_duration),
            FillerStrategy::PoolRotation => self.plan_pool_rotation(gap, gap_duration),
            FillerStrategy::FreezeFrame
            | FillerStrategy::ColourBars
            | FillerStrategy::BlackSilence => self.plan_technical_fill(gap, gap_duration),
        }
    }

    /// Plan: loop a single clip.
    fn plan_loop_single(&self, gap: &ScheduleGap, gap_duration: f64) -> FillPlan {
        if let Some(clip) = self.pool.select_best(f64::MAX) {
            let item = FillPlanItem {
                clip_id: clip.id.clone(),
                start_sec: gap.start_sec,
                play_duration_sec: gap_duration,
                looping: true,
            };
            FillPlan {
                gap: gap.clone(),
                items: vec![item],
                strategy: FillerStrategy::LoopSingle,
                fully_covered: true,
            }
        } else {
            self.plan_technical_fill(gap, gap_duration)
        }
    }

    /// Plan: rotate through pool clips.
    fn plan_pool_rotation(&self, gap: &ScheduleGap, gap_duration: f64) -> FillPlan {
        let mut items = Vec::new();
        let mut remaining = gap_duration;
        let mut cursor = gap.start_sec;

        for _ in 0..self.config.max_chain_length {
            if remaining <= 0.0 {
                break;
            }
            if let Some(clip) = self.pool.select_longest(remaining) {
                let play_dur = if self.config.trim_to_fit {
                    clip.duration_sec.min(remaining)
                } else {
                    clip.duration_sec
                };
                items.push(FillPlanItem {
                    clip_id: clip.id.clone(),
                    start_sec: cursor,
                    play_duration_sec: play_dur,
                    looping: false,
                });
                cursor += play_dur;
                remaining -= play_dur;
            } else {
                break;
            }
        }

        let covered = remaining <= 0.0;
        FillPlan {
            gap: gap.clone(),
            items,
            strategy: FillerStrategy::PoolRotation,
            fully_covered: covered,
        }
    }

    /// Plan: technical fill (colour bars, black, freeze).
    fn plan_technical_fill(&self, gap: &ScheduleGap, gap_duration: f64) -> FillPlan {
        let item = FillPlanItem {
            clip_id: format!("__technical_{:?}", self.config.strategy),
            start_sec: gap.start_sec,
            play_duration_sec: gap_duration,
            looping: true,
        };
        FillPlan {
            gap: gap.clone(),
            items: vec![item],
            strategy: self.config.strategy,
            fully_covered: true,
        }
    }

    /// Detect gaps in a list of scheduled item intervals.
    #[allow(clippy::cast_precision_loss)]
    pub fn detect_gaps(
        items: &[(f64, f64)],
        channel: &str,
        day_length_sec: f64,
    ) -> Vec<ScheduleGap> {
        if items.is_empty() {
            return vec![ScheduleGap {
                start_sec: 0.0,
                end_sec: day_length_sec,
                channel: channel.to_string(),
            }];
        }

        let mut sorted: Vec<(f64, f64)> = items.to_vec();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut gaps = Vec::new();
        let mut cursor = 0.0f64;

        for &(start, end) in &sorted {
            if start > cursor {
                gaps.push(ScheduleGap {
                    start_sec: cursor,
                    end_sec: start,
                    channel: channel.to_string(),
                });
            }
            if end > cursor {
                cursor = end;
            }
        }

        if cursor < day_length_sec {
            gaps.push(ScheduleGap {
                start_sec: cursor,
                end_sec: day_length_sec,
                channel: channel.to_string(),
            });
        }

        gaps
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_pool() -> FillerPool {
        let mut pool = FillerPool::new(10);
        pool.add_clip(FillerClip {
            id: "promo1".into(),
            name: "Promo A".into(),
            duration_sec: 15.0,
            path: "/content/promos/a.mxf".into(),
            priority: 10,
            usage_count: 0,
            max_uses_per_day: 5,
        });
        pool.add_clip(FillerClip {
            id: "ident1".into(),
            name: "Station Ident".into(),
            duration_sec: 5.0,
            path: "/content/idents/1.mxf".into(),
            priority: 8,
            usage_count: 0,
            max_uses_per_day: 0, // unlimited
        });
        pool.add_clip(FillerClip {
            id: "promo2".into(),
            name: "Promo B".into(),
            duration_sec: 30.0,
            path: "/content/promos/b.mxf".into(),
            priority: 7,
            usage_count: 0,
            max_uses_per_day: 3,
        });
        pool
    }

    #[test]
    fn test_filler_clip_available() {
        let clip = FillerClip {
            id: "c1".into(),
            name: "Clip".into(),
            duration_sec: 10.0,
            path: String::new(),
            priority: 5,
            usage_count: 3,
            max_uses_per_day: 5,
        };
        assert!(clip.available());

        let exhausted = FillerClip {
            usage_count: 5,
            ..clip
        };
        assert!(!exhausted.available());
    }

    #[test]
    fn test_effective_priority() {
        let clip = FillerClip {
            id: "c1".into(),
            name: "Clip".into(),
            duration_sec: 10.0,
            path: String::new(),
            priority: 10,
            usage_count: 0,
            max_uses_per_day: 0,
        };
        assert!((clip.effective_priority() - 10.0).abs() < 1e-9);

        let used = FillerClip {
            usage_count: 4,
            ..clip
        };
        assert!((used.effective_priority() - 8.0).abs() < 1e-9);
    }

    #[test]
    fn test_gap_duration() {
        let gap = ScheduleGap {
            start_sec: 100.0,
            end_sec: 130.0,
            channel: "CH1".into(),
        };
        assert!((gap.duration() - 30.0).abs() < 1e-9);
    }

    #[test]
    fn test_gap_display() {
        let gap = ScheduleGap {
            start_sec: 100.0,
            end_sec: 130.0,
            channel: "CH1".into(),
        };
        let s = format!("{gap}");
        assert!(s.contains("CH1"));
        assert!(s.contains("30.0"));
    }

    #[test]
    fn test_pool_select_best() {
        let pool = sample_pool();
        let best = pool.select_best(20.0).expect("should succeed in test");
        assert_eq!(best.id, "promo1"); // highest priority that fits in 20s
    }

    #[test]
    fn test_pool_select_longest() {
        let pool = sample_pool();
        let longest = pool.select_longest(100.0).expect("should succeed in test");
        assert_eq!(longest.id, "promo2"); // 30s is longest
    }

    #[test]
    fn test_pool_mark_used() {
        let mut pool = sample_pool();
        pool.mark_used("promo1");
        let clip = pool
            .clips
            .iter()
            .find(|c| c.id == "promo1")
            .expect("should succeed in test");
        assert_eq!(clip.usage_count, 1);
        assert_eq!(pool.recent.len(), 1);
    }

    #[test]
    fn test_pool_reset_daily() {
        let mut pool = sample_pool();
        pool.mark_used("promo1");
        pool.mark_used("promo1");
        pool.reset_daily();
        for clip in &pool.clips {
            assert_eq!(clip.usage_count, 0);
        }
        assert!(pool.recent.is_empty());
    }

    #[test]
    fn test_detect_gaps_no_items() {
        let gaps = GapFiller::detect_gaps(&[], "CH1", 86400.0);
        assert_eq!(gaps.len(), 1);
        assert!((gaps[0].duration() - 86400.0).abs() < 1e-6);
    }

    #[test]
    fn test_detect_gaps_with_items() {
        let items = vec![(100.0, 200.0), (300.0, 400.0)];
        let gaps = GapFiller::detect_gaps(&items, "CH1", 500.0);
        // 0-100, 200-300, 400-500
        assert_eq!(gaps.len(), 3);
    }

    #[test]
    fn test_plan_loop_single() {
        let pool = sample_pool();
        let config = GapFillerConfig {
            strategy: FillerStrategy::LoopSingle,
            ..Default::default()
        };
        let filler = GapFiller::new(config, pool);
        let gap = ScheduleGap {
            start_sec: 100.0,
            end_sec: 160.0,
            channel: "CH1".into(),
        };
        let plan = filler.plan_fill(&gap);
        assert!(plan.fully_covered);
        assert_eq!(plan.len(), 1);
        assert!(plan.items[0].looping);
    }

    #[test]
    fn test_plan_pool_rotation() {
        let pool = sample_pool();
        let config = GapFillerConfig {
            strategy: FillerStrategy::PoolRotation,
            trim_to_fit: true,
            ..Default::default()
        };
        let filler = GapFiller::new(config, pool);
        let gap = ScheduleGap {
            start_sec: 100.0,
            end_sec: 145.0,
            channel: "CH1".into(),
        };
        let plan = filler.plan_fill(&gap);
        assert!(!plan.is_empty());
        assert!(plan.total_duration() <= 45.0 + 1e-6);
    }

    #[test]
    fn test_plan_technical_fill() {
        let pool = FillerPool::new(5);
        let config = GapFillerConfig {
            strategy: FillerStrategy::ColourBars,
            ..Default::default()
        };
        let filler = GapFiller::new(config, pool);
        let gap = ScheduleGap {
            start_sec: 0.0,
            end_sec: 10.0,
            channel: "CH1".into(),
        };
        let plan = filler.plan_fill(&gap);
        assert!(plan.fully_covered);
        assert_eq!(plan.strategy, FillerStrategy::ColourBars);
    }

    #[test]
    fn test_plan_short_gap_ignored() {
        let pool = sample_pool();
        let config = GapFillerConfig {
            min_gap_sec: 5.0,
            ..Default::default()
        };
        let filler = GapFiller::new(config, pool);
        let gap = ScheduleGap {
            start_sec: 0.0,
            end_sec: 2.0,
            channel: "CH1".into(),
        };
        let plan = filler.plan_fill(&gap);
        assert!(!plan.fully_covered);
        assert!(plan.is_empty());
    }

    #[test]
    fn test_filler_strategy_display() {
        assert_eq!(format!("{}", FillerStrategy::PoolRotation), "Pool Rotation");
        assert_eq!(format!("{}", FillerStrategy::ColourBars), "Colour Bars");
    }
}
