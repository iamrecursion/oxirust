//! Caption timing adjustment: shifting, scaling, and gap analysis.

#![allow(dead_code)]

/// Strategy used when adjusting caption timing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimingStrategy {
    /// Shift all entries by a fixed offset (milliseconds).
    ConstantShift,
    /// Scale all durations by a ratio relative to total programme length.
    ProportionalScale,
    /// Snap entry/exit points to the nearest frame boundary.
    FrameSnap,
    /// Adjust timing to maintain a minimum gap between consecutive entries.
    GapEnforce,
}

impl TimingStrategy {
    /// Human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::ConstantShift => "Constant Shift",
            Self::ProportionalScale => "Proportional Scale",
            Self::FrameSnap => "Frame Snap",
            Self::GapEnforce => "Gap Enforce",
        }
    }

    /// Whether this strategy can be applied without knowing total duration.
    #[must_use]
    pub fn is_duration_independent(&self) -> bool {
        matches!(
            self,
            Self::ConstantShift | Self::GapEnforce | Self::FrameSnap
        )
    }
}

/// A timed caption entry with start and end timestamps in milliseconds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimedEntry {
    /// Unique sequential identifier (1-based index).
    pub index: usize,
    /// Start time in milliseconds.
    pub start_ms: u64,
    /// End time in milliseconds.
    pub end_ms: u64,
    /// Caption text.
    pub text: String,
}

impl TimedEntry {
    /// Create a new timed entry.
    ///
    /// # Panics
    ///
    /// Panics in debug if `end_ms <= start_ms`.
    #[must_use]
    pub fn new(index: usize, start_ms: u64, end_ms: u64, text: impl Into<String>) -> Self {
        debug_assert!(end_ms > start_ms, "end_ms must be greater than start_ms");
        Self {
            index,
            start_ms,
            end_ms,
            text: text.into(),
        }
    }

    /// Duration of this entry in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }
}

/// Adjusts the timing of a collection of caption entries.
#[derive(Debug, Clone)]
pub struct CaptionTimingAdjuster {
    entries: Vec<TimedEntry>,
}

impl CaptionTimingAdjuster {
    /// Create a new adjuster from an ordered list of entries.
    #[must_use]
    pub fn new(entries: Vec<TimedEntry>) -> Self {
        Self { entries }
    }

    /// Return a reference to the current entries.
    #[must_use]
    pub fn entries(&self) -> &[TimedEntry] {
        &self.entries
    }

    /// Shift all entry timestamps by `offset_ms`.
    ///
    /// A negative offset will clamp `start_ms` and `end_ms` to zero rather
    /// than underflowing.
    pub fn shift_all(&mut self, offset_ms: i64) {
        for entry in &mut self.entries {
            if offset_ms >= 0 {
                entry.start_ms = entry.start_ms.saturating_add(offset_ms as u64);
                entry.end_ms = entry.end_ms.saturating_add(offset_ms as u64);
            } else {
                let abs = (-offset_ms) as u64;
                entry.start_ms = entry.start_ms.saturating_sub(abs);
                entry.end_ms = entry.end_ms.saturating_sub(abs);
            }
        }
    }

    /// Scale all entry durations (and start times) by `factor`.
    ///
    /// This is useful when the programme has been conform-adjusted.
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub fn scale_duration(&mut self, factor: f64) {
        for entry in &mut self.entries {
            entry.start_ms = (entry.start_ms as f64 * factor).round() as u64;
            entry.end_ms = (entry.end_ms as f64 * factor).round() as u64;
        }
    }

    /// Enforce a minimum gap of `min_gap_ms` milliseconds between consecutive entries.
    ///
    /// If two entries overlap or are too close, the later one's start is pushed
    /// forward; its duration is preserved.
    pub fn enforce_min_gap(&mut self, min_gap_ms: u64) {
        for i in 1..self.entries.len() {
            let prev_end = self.entries[i - 1].end_ms;
            let cur_start = self.entries[i].start_ms;
            if prev_end + min_gap_ms > cur_start {
                let new_start = prev_end + min_gap_ms;
                let dur = self.entries[i].duration_ms();
                self.entries[i].start_ms = new_start;
                self.entries[i].end_ms = new_start + dur;
            }
        }
    }

    /// Count of entries in the adjuster.
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }
}

/// Summary statistics derived from a caption track's timing.
#[derive(Debug, Clone)]
pub struct CaptionTimingReport {
    /// Duration statistics per entry (milliseconds).
    durations_ms: Vec<u64>,
    /// Gaps between consecutive entries (milliseconds).
    gaps_ms: Vec<u64>,
}

impl CaptionTimingReport {
    /// Build a report from a slice of timed entries.
    ///
    /// Entries must be sorted by `start_ms`.
    #[must_use]
    pub fn from_entries(entries: &[TimedEntry]) -> Self {
        let durations_ms: Vec<u64> = entries.iter().map(TimedEntry::duration_ms).collect();

        let gaps_ms: Vec<u64> = entries
            .windows(2)
            .map(|w| w[1].start_ms.saturating_sub(w[0].end_ms))
            .collect();

        Self {
            durations_ms,
            gaps_ms,
        }
    }

    /// Maximum gap between consecutive entries in milliseconds.
    ///
    /// Returns `0` if there are fewer than two entries.
    #[must_use]
    pub fn max_gap_ms(&self) -> u64 {
        self.gaps_ms.iter().copied().max().unwrap_or(0)
    }

    /// Average duration of all entries in milliseconds.
    ///
    /// Returns `0.0` when there are no entries.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn avg_duration_ms(&self) -> f64 {
        if self.durations_ms.is_empty() {
            return 0.0;
        }
        let total: u64 = self.durations_ms.iter().sum();
        total as f64 / self.durations_ms.len() as f64
    }

    /// Minimum duration across all entries.
    #[must_use]
    pub fn min_duration_ms(&self) -> u64 {
        self.durations_ms.iter().copied().min().unwrap_or(0)
    }

    /// Maximum duration across all entries.
    #[must_use]
    pub fn max_duration_ms(&self) -> u64 {
        self.durations_ms.iter().copied().max().unwrap_or(0)
    }

    /// Number of entries that were analysed.
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.durations_ms.len()
    }

    /// Number of gaps between consecutive entries.
    #[must_use]
    pub fn gap_count(&self) -> usize {
        self.gaps_ms.len()
    }
}

// ─── unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_entries() -> Vec<TimedEntry> {
        vec![
            TimedEntry::new(1, 0, 3000, "Hello world"),
            TimedEntry::new(2, 5000, 8000, "This is a test"),
            TimedEntry::new(3, 10000, 12000, "Caption three"),
        ]
    }

    // 1 — TimingStrategy::label
    #[test]
    fn test_strategy_label() {
        assert_eq!(TimingStrategy::ConstantShift.label(), "Constant Shift");
        assert_eq!(
            TimingStrategy::ProportionalScale.label(),
            "Proportional Scale"
        );
        assert_eq!(TimingStrategy::FrameSnap.label(), "Frame Snap");
        assert_eq!(TimingStrategy::GapEnforce.label(), "Gap Enforce");
    }

    // 2 — TimingStrategy::is_duration_independent
    #[test]
    fn test_strategy_duration_independent() {
        assert!(TimingStrategy::ConstantShift.is_duration_independent());
        assert!(!TimingStrategy::ProportionalScale.is_duration_independent());
    }

    // 3 — TimedEntry::duration_ms
    #[test]
    fn test_timed_entry_duration() {
        let e = TimedEntry::new(1, 1000, 4000, "text");
        assert_eq!(e.duration_ms(), 3000);
    }

    // 4 — shift_all positive
    #[test]
    fn test_shift_all_positive() {
        let mut adj = CaptionTimingAdjuster::new(sample_entries());
        adj.shift_all(2000);
        assert_eq!(adj.entries()[0].start_ms, 2000);
        assert_eq!(adj.entries()[0].end_ms, 5000);
    }

    // 5 — shift_all negative clamps at zero
    #[test]
    fn test_shift_all_negative_clamp() {
        let entries = vec![TimedEntry::new(1, 500, 2000, "early")];
        let mut adj = CaptionTimingAdjuster::new(entries);
        adj.shift_all(-1000);
        // 500 - 1000 saturates to 0
        assert_eq!(adj.entries()[0].start_ms, 0);
        // 2000 - 1000 = 1000
        assert_eq!(adj.entries()[0].end_ms, 1000);
    }

    // 6 — scale_duration doubles all times
    #[test]
    fn test_scale_duration_double() {
        let mut adj = CaptionTimingAdjuster::new(sample_entries());
        adj.scale_duration(2.0);
        assert_eq!(adj.entries()[0].start_ms, 0);
        assert_eq!(adj.entries()[0].end_ms, 6000);
        assert_eq!(adj.entries()[1].start_ms, 10000);
    }

    // 7 — scale_duration halves all times
    #[test]
    fn test_scale_duration_half() {
        let mut adj = CaptionTimingAdjuster::new(sample_entries());
        adj.scale_duration(0.5);
        assert_eq!(adj.entries()[0].end_ms, 1500);
        assert_eq!(adj.entries()[1].start_ms, 2500);
    }

    // 8 — enforce_min_gap separates overlapping entries
    #[test]
    fn test_enforce_min_gap() {
        let entries = vec![
            TimedEntry::new(1, 0, 3000, "A"),
            TimedEntry::new(2, 3000, 5000, "B"), // starts exactly at prev end — gap = 0
        ];
        let mut adj = CaptionTimingAdjuster::new(entries);
        adj.enforce_min_gap(80);
        // Entry 2 should start at 3080
        assert_eq!(adj.entries()[1].start_ms, 3080);
        assert_eq!(adj.entries()[1].end_ms, 5080); // duration preserved
    }

    // 9 — entry_count
    #[test]
    fn test_entry_count() {
        let adj = CaptionTimingAdjuster::new(sample_entries());
        assert_eq!(adj.entry_count(), 3);
    }

    // 10 — CaptionTimingReport::max_gap_ms
    #[test]
    fn test_report_max_gap() {
        let entries = sample_entries();
        let report = CaptionTimingReport::from_entries(&entries);
        // gaps: 5000-3000=2000, 10000-8000=2000
        assert_eq!(report.max_gap_ms(), 2000);
    }

    // 11 — CaptionTimingReport::avg_duration_ms
    #[test]
    fn test_report_avg_duration() {
        let entries = sample_entries();
        let report = CaptionTimingReport::from_entries(&entries);
        // durations: 3000, 3000, 2000 → avg 2666.67
        let avg = report.avg_duration_ms();
        assert!((avg - 2666.666_666_666_666_5).abs() < 1.0);
    }

    // 12 — CaptionTimingReport with single entry — no gaps
    #[test]
    fn test_report_single_entry_no_gap() {
        let entries = vec![TimedEntry::new(1, 0, 5000, "only")];
        let report = CaptionTimingReport::from_entries(&entries);
        assert_eq!(report.max_gap_ms(), 0);
        assert_eq!(report.gap_count(), 0);
    }

    // 13 — CaptionTimingReport empty entries
    #[test]
    fn test_report_empty() {
        let report = CaptionTimingReport::from_entries(&[]);
        assert_eq!(report.avg_duration_ms(), 0.0);
        assert_eq!(report.max_gap_ms(), 0);
        assert_eq!(report.entry_count(), 0);
    }

    // 14 — min/max duration
    #[test]
    fn test_report_min_max_duration() {
        let entries = sample_entries();
        let report = CaptionTimingReport::from_entries(&entries);
        assert_eq!(report.min_duration_ms(), 2000);
        assert_eq!(report.max_duration_ms(), 3000);
    }

    // 15 — gap_count equals entries - 1
    #[test]
    fn test_report_gap_count() {
        let entries = sample_entries(); // 3 entries
        let report = CaptionTimingReport::from_entries(&entries);
        assert_eq!(report.gap_count(), 2); // 3 - 1 = 2 gaps
    }
}
