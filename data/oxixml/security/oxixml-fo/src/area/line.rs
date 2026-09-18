//! Line-breaking **primitives**.
//!
//! These are the building blocks a renderer uses to turn a stream of inline
//! material into [`LineArea`](crate::area::LineArea)s; nothing here reads a
//! formatting object tree, and no pagination happens. The vocabulary is the
//! classical box/glue/penalty model of Knuth and Plass, which is also how XSL
//! 1.1 §4.7.2 describes line building.
//!
//! Two strategies are provided:
//!
//! * [`LineBreakingStrategy::Greedy`] — first fit, the behaviour of a simple
//!   formatter and of `wrap-option="wrap"` at its cheapest;
//! * [`LineBreakingStrategy::TotalFit`] — a Knuth–Plass-lite total-fit pass that
//!   minimizes the sum of the demerits of all lines by dynamic programming over
//!   the feasible breakpoints.
//!
//! ```
//! use oxixml_fo::area::{Item, LineBreaker, LineBreakingStrategy};
//! use oxixml_fo::property::Length;
//!
//! let pt = Length::from_points;
//! // "aaa bbb ccc" with 30pt words and a 10pt stretchable space.
//! let items = vec![
//!     Item::boxed(pt(30.0)),
//!     Item::glue(pt(10.0), pt(5.0), pt(3.0)),
//!     Item::boxed(pt(30.0)),
//!     Item::glue(pt(10.0), pt(5.0), pt(3.0)),
//!     Item::boxed(pt(30.0)),
//!     Item::forced_break(),
//! ];
//! let breaker = LineBreaker::new(pt(75.0)).with_strategy(LineBreakingStrategy::Greedy);
//! let lines = breaker.break_lines(&items);
//! assert_eq!(lines.len(), 2);
//! ```

use crate::property::Length;

/// The penalty of a breakpoint.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Penalty {
    /// The cost of breaking here; negative values encourage a break.
    pub cost: f64,
    /// Whether breaking here duplicates material (a hyphen, for instance),
    /// which makes two consecutive breaks at flagged points undesirable.
    pub flagged: bool,
}

impl Penalty {
    /// A break that must be taken.
    pub const FORCED: Self = Self {
        cost: f64::NEG_INFINITY,
        flagged: false,
    };
    /// A break that must not be taken.
    pub const FORBIDDEN: Self = Self {
        cost: f64::INFINITY,
        flagged: false,
    };
}

/// One item of the inline stream presented to the line breaker.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Item {
    /// Unbreakable material of a fixed width, such as a word.
    Box {
        /// The inline extent.
        width: Length,
    },
    /// Breakable, elastic material, such as a word space.
    Glue {
        /// The natural inline extent.
        width: Length,
        /// How much the glue may grow.
        stretch: Length,
        /// How much the glue may shrink.
        shrink: Length,
    },
    /// A potential breakpoint with a cost, such as a hyphenation point or a
    /// forced line break.
    Penalty {
        /// The width contributed only when the break is taken (a hyphen).
        width: Length,
        /// The penalty itself.
        penalty: Penalty,
    },
}

impl Item {
    /// A box of `width`.
    #[must_use]
    pub const fn boxed(width: Length) -> Self {
        Self::Box { width }
    }

    /// Glue with the given natural width and elasticity.
    #[must_use]
    pub const fn glue(width: Length, stretch: Length, shrink: Length) -> Self {
        Self::Glue {
            width,
            stretch,
            shrink,
        }
    }

    /// A mandatory break, as produced by `linefeed-treatment="preserve"`.
    #[must_use]
    pub const fn forced_break() -> Self {
        Self::Penalty {
            width: Length::ZERO,
            penalty: Penalty::FORCED,
        }
    }

    /// A hyphenation opportunity costing `cost` and adding `width` when taken.
    #[must_use]
    pub const fn hyphen(width: Length, cost: f64) -> Self {
        Self::Penalty {
            width,
            penalty: Penalty {
                cost,
                flagged: true,
            },
        }
    }

    /// The width this item always contributes.
    #[must_use]
    pub fn width(self) -> Length {
        match self {
            Self::Box { width } | Self::Glue { width, .. } => width,
            Self::Penalty { .. } => Length::ZERO,
        }
    }

    /// Whether a line may end just before this item.
    #[must_use]
    pub fn is_breakpoint(self, previous: Option<Item>) -> bool {
        match self {
            // Glue is a legal breakpoint only when it follows a box.
            Self::Glue { .. } => matches!(previous, Some(Self::Box { .. })),
            Self::Penalty { penalty, .. } => penalty.cost < f64::INFINITY,
            Self::Box { .. } => false,
        }
    }
}

/// One line produced by the line breaker.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Breakpoint {
    /// Index of the first item on the line.
    pub start: usize,
    /// Index one past the last item on the line.
    pub end: usize,
    /// The natural width of the line's content, excluding trailing glue.
    pub natural_width: Length,
    /// The adjustment ratio the line needs: positive means the glue stretches,
    /// negative that it shrinks.
    pub adjustment_ratio: f64,
    /// Whether the line ended at a forced break or at the end of the stream.
    pub forced: bool,
}

/// One state of the total-fit dynamic program: the best way found so far of
/// reaching a given breakpoint.
#[derive(Debug, Clone, Copy)]
struct Node {
    /// Total demerits of every line up to and including this breakpoint.
    demerits: f64,
    /// The breakpoint the last line started after, or `None` for the first line.
    previous: Option<usize>,
    /// The adjustment ratio of the line ending here.
    adjustment_ratio: f64,
    /// The natural width of the line ending here.
    natural_width: Length,
    /// Whether this breakpoint is flagged (a hyphen).
    flagged: bool,
}

/// Which line-breaking algorithm [`LineBreaker`] runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LineBreakingStrategy {
    /// First fit: break as late as each line allows.
    #[default]
    Greedy,
    /// Knuth–Plass-lite total fit: minimize the total demerits of all lines.
    TotalFit,
}

/// Breaks a stream of [`Item`]s into lines of a fixed measure.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LineBreaker {
    /// The inline extent available to each line.
    pub measure: Length,
    /// Which algorithm to run.
    pub strategy: LineBreakingStrategy,
    /// Extra demerits charged for a line that ends at a flagged breakpoint when
    /// the previous one did too.
    pub double_hyphen_demerits: f64,
    /// Demerits charged for a line whose adjustment ratio exceeds
    /// [`Self::tolerance`] in the greedy pass, and the threshold beyond which
    /// the total-fit pass rejects a line.
    pub tolerance: f64,
}

impl LineBreaker {
    /// A breaker for lines of `measure` inline extent.
    #[must_use]
    pub const fn new(measure: Length) -> Self {
        Self {
            measure,
            strategy: LineBreakingStrategy::Greedy,
            double_hyphen_demerits: 3000.0,
            tolerance: 1.0,
        }
    }

    /// Selects the algorithm.
    #[must_use]
    pub const fn with_strategy(mut self, strategy: LineBreakingStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Sets the maximum adjustment ratio a line may need.
    #[must_use]
    pub const fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Breaks `items` into lines.
    #[must_use]
    pub fn break_lines(&self, items: &[Item]) -> Vec<Breakpoint> {
        match self.strategy {
            LineBreakingStrategy::Greedy => self.greedy(items),
            LineBreakingStrategy::TotalFit => self.total_fit(items),
        }
    }

    /// The measurements of the run `items[start..end]`, ignoring glue that
    /// would end the line.
    fn measure_run(items: &[Item], start: usize, end: usize) -> (Length, Length, Length) {
        let mut width = Length::ZERO;
        let mut stretch = Length::ZERO;
        let mut shrink = Length::ZERO;
        for (index, item) in items.iter().enumerate().take(end).skip(start) {
            match *item {
                Item::Box { width: w } => width = width + w,
                Item::Glue {
                    width: w,
                    stretch: st,
                    shrink: sh,
                } => {
                    // Glue at the very end of a line disappears.
                    if index + 1 == end {
                        continue;
                    }
                    width = width + w;
                    stretch = stretch + st;
                    shrink = shrink + sh;
                }
                // A penalty inside the line contributes nothing; the one the
                // line breaks at is handled below.
                Item::Penalty { .. } => {}
            }
        }
        // Breaking at a penalty materializes its width (a discretionary hyphen).
        if let Some(Item::Penalty { width: extra, .. }) = items.get(end) {
            width = width + *extra;
        }
        (width, stretch, shrink)
    }

    fn adjustment_ratio(&self, natural: Length, stretch: Length, shrink: Length) -> f64 {
        let slack = self.measure.points() - natural.points();
        if slack > 0.0 {
            if stretch.points() > 0.0 {
                slack / stretch.points()
            } else {
                f64::INFINITY
            }
        } else if slack < 0.0 {
            if shrink.points() > 0.0 {
                slack / shrink.points()
            } else {
                f64::NEG_INFINITY
            }
        } else {
            0.0
        }
    }

    /// Every index at which a line may end, plus the end of the stream.
    ///
    /// A stream that already ends with a forced break does not get the extra
    /// end-of-stream breakpoint: the two would be equivalent, and the duplicate
    /// would let the total-fit pass return a spurious empty last line.
    fn breakpoints(items: &[Item]) -> Vec<usize> {
        let mut points = Vec::new();
        for index in 0..items.len() {
            let previous = index.checked_sub(1).map(|i| items[i]);
            if items[index].is_breakpoint(previous) {
                points.push(index);
            }
        }
        let ends_forced = matches!(
            items.last(),
            Some(Item::Penalty { penalty, .. }) if penalty.cost == f64::NEG_INFINITY
        );
        if !ends_forced {
            points.push(items.len());
        }
        points
    }

    fn is_forced(items: &[Item], end: usize) -> bool {
        end >= items.len()
            || matches!(items[end], Item::Penalty { penalty, .. } if penalty.cost == f64::NEG_INFINITY)
    }

    /// Where the next line starts after a break at `end`: leading glue and the
    /// penalty item itself are discarded.
    fn next_start(items: &[Item], end: usize) -> usize {
        let mut start = end;
        if matches!(items.get(start), Some(Item::Penalty { .. })) {
            start += 1;
        }
        while matches!(items.get(start), Some(Item::Glue { .. })) {
            start += 1;
        }
        start
    }

    fn greedy(&self, items: &[Item]) -> Vec<Breakpoint> {
        let mut lines = Vec::new();
        if items.is_empty() {
            return lines;
        }
        let points = Self::breakpoints(items);
        let mut start = 0usize;
        while start < items.len() {
            let mut chosen = None;
            for point in points.iter().copied().filter(|point| *point > start) {
                let (width, stretch, shrink) = Self::measure_run(items, start, point);
                let ratio = self.adjustment_ratio(width, stretch, shrink);
                let fits = ratio >= -1.0;
                if fits {
                    chosen = Some((point, width, ratio));
                }
                if !fits || Self::is_forced(items, point) {
                    break;
                }
            }
            let (end, width, ratio) = match chosen {
                Some(found) => found,
                None => {
                    // Nothing fits: take the first breakpoint anyway so that
                    // the algorithm always makes progress (overfull line).
                    let point = points
                        .iter()
                        .copied()
                        .find(|point| *point > start)
                        .unwrap_or(items.len());
                    let (width, stretch, shrink) = Self::measure_run(items, start, point);
                    (point, width, self.adjustment_ratio(width, stretch, shrink))
                }
            };
            lines.push(Breakpoint {
                start,
                end,
                natural_width: width,
                adjustment_ratio: ratio,
                forced: Self::is_forced(items, end),
            });
            let next = Self::next_start(items, end);
            if next <= start {
                break;
            }
            start = next;
        }
        lines
    }

    /// The badness of a line, `100 * |ratio|^3`, as in Knuth–Plass.
    fn badness(ratio: f64) -> f64 {
        if !ratio.is_finite() {
            return f64::INFINITY;
        }
        100.0 * ratio.abs().powi(3)
    }

    fn demerits(&self, ratio: f64, penalty: f64, flagged: bool, previous_flagged: bool) -> f64 {
        let badness = Self::badness(ratio);
        if !badness.is_finite() {
            return f64::INFINITY;
        }
        let line_penalty = if penalty.is_finite() { penalty } else { 0.0 };
        let base = if line_penalty >= 0.0 {
            (1.0 + badness + line_penalty).powi(2)
        } else {
            (1.0 + badness).powi(2) - line_penalty * line_penalty
        };
        if flagged && previous_flagged {
            base + self.double_hyphen_demerits
        } else {
            base
        }
    }

    fn total_fit(&self, items: &[Item]) -> Vec<Breakpoint> {
        if items.is_empty() {
            return Vec::new();
        }
        let points = Self::breakpoints(items);
        let mut best: Vec<Option<Node>> = vec![None; points.len()];
        let starts: Vec<usize> = points
            .iter()
            .map(|point| Self::next_start(items, *point))
            .collect();

        for (index, end) in points.iter().copied().enumerate() {
            // A line may start at item 0 or just after an earlier breakpoint.
            let mut candidates: Vec<(usize, f64, Option<usize>, bool)> =
                vec![(0usize, 0.0, None, false)];
            for (previous, previous_end) in points.iter().copied().enumerate() {
                if previous_end >= end {
                    break;
                }
                if let Some(node) = best[previous] {
                    candidates.push((
                        starts[previous],
                        node.demerits,
                        Some(previous),
                        node.flagged,
                    ));
                }
            }
            for (start, so_far, previous, previous_flagged) in candidates {
                if start >= end {
                    continue;
                }
                let (width, stretch, shrink) = Self::measure_run(items, start, end);
                let ratio = self.adjustment_ratio(width, stretch, shrink);
                if ratio < -1.0 || ratio > self.tolerance {
                    // Infeasible, unless the break is forced.
                    if !Self::is_forced(items, end) {
                        continue;
                    }
                }
                let (penalty_cost, flagged) = match items.get(end) {
                    Some(Item::Penalty { penalty, .. }) => (penalty.cost, penalty.flagged),
                    _ => (0.0, false),
                };
                let step = self.demerits(
                    if ratio.is_finite() { ratio } else { 0.0 },
                    penalty_cost,
                    flagged,
                    previous_flagged,
                );
                if !step.is_finite() {
                    continue;
                }
                let total = so_far + step;
                if best[index].is_none_or(|current| total < current.demerits) {
                    best[index] = Some(Node {
                        demerits: total,
                        previous,
                        adjustment_ratio: ratio,
                        natural_width: width,
                        flagged,
                    });
                }
            }
        }

        let Some(last) = best.len().checked_sub(1) else {
            return Vec::new();
        };
        if best[last].is_none() {
            // No feasible total-fit solution; fall back to the greedy pass so
            // that a caller always gets usable lines.
            return self.greedy(items);
        }

        let mut chain = Vec::new();
        let mut cursor = Some(last);
        while let Some(index) = cursor {
            let Some(node) = best[index] else {
                break;
            };
            let end = points[index];
            let start = node.previous.map_or(0, |previous| starts[previous]);
            chain.push(Breakpoint {
                start,
                end,
                natural_width: node.natural_width,
                adjustment_ratio: node.adjustment_ratio,
                forced: Self::is_forced(items, end),
            });
            cursor = node.previous;
        }
        chain.reverse();
        chain
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pt(value: f64) -> Length {
        Length::from_points(value)
    }

    /// Three 30pt words separated by 10pt spaces.
    fn words(count: usize) -> Vec<Item> {
        let mut items = Vec::new();
        for index in 0..count {
            if index > 0 {
                items.push(Item::glue(pt(10.0), pt(5.0), pt(3.0)));
            }
            items.push(Item::boxed(pt(30.0)));
        }
        items.push(Item::forced_break());
        items
    }

    #[test]
    fn greedy_fills_each_line_before_breaking() {
        let items = words(4);
        let lines = LineBreaker::new(pt(75.0)).break_lines(&items);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].natural_width, pt(70.0));
        assert!(lines[0].adjustment_ratio > 0.0, "the line stretches");
        assert!(lines.last().is_some_and(|line| line.forced));
    }

    #[test]
    fn a_single_line_that_fits_is_not_broken() {
        let items = words(2);
        let lines = LineBreaker::new(pt(500.0)).break_lines(&items);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].start, 0);
        assert_eq!(lines[0].end, items.len() - 1);
    }

    #[test]
    fn forced_breaks_always_end_a_line() {
        let items = vec![
            Item::boxed(pt(10.0)),
            Item::forced_break(),
            Item::boxed(pt(10.0)),
            Item::forced_break(),
        ];
        let lines = LineBreaker::new(pt(500.0)).break_lines(&items);
        assert_eq!(lines.len(), 2);
        assert!(lines.iter().all(|line| line.forced));
    }

    #[test]
    fn total_fit_balances_the_lines() {
        let items = words(6);
        let greedy = LineBreaker::new(pt(120.0)).break_lines(&items);
        let total = LineBreaker::new(pt(120.0))
            .with_strategy(LineBreakingStrategy::TotalFit)
            .with_tolerance(4.0)
            .break_lines(&items);
        assert!(!total.is_empty());
        // Both strategies must cover the whole stream.
        assert_eq!(greedy.last().map(|line| line.end), Some(items.len() - 1));
        assert_eq!(total.last().map(|line| line.end), Some(items.len() - 1));
        // Total fit never produces a worse maximum badness than greedy.
        let worst = |lines: &[Breakpoint]| {
            lines
                .iter()
                .filter(|line| !line.forced)
                .map(|line| line.adjustment_ratio.abs())
                .fold(0.0_f64, f64::max)
        };
        assert!(worst(&total) <= worst(&greedy) + 1e-9);
    }

    #[test]
    fn hyphenation_points_are_breakable() {
        let items = vec![
            Item::boxed(pt(40.0)),
            Item::hyphen(pt(3.0), 50.0),
            Item::boxed(pt(40.0)),
            Item::forced_break(),
        ];
        let lines = LineBreaker::new(pt(45.0)).break_lines(&items);
        assert_eq!(lines.len(), 2);
        // The hyphen widens the first line to 43pt.
        assert_eq!(lines[0].natural_width, pt(43.0));
    }

    #[test]
    fn an_empty_stream_produces_no_lines() {
        assert!(LineBreaker::new(pt(100.0)).break_lines(&[]).is_empty());
    }

    #[test]
    fn overfull_lines_still_make_progress() {
        let items = vec![Item::boxed(pt(500.0)), Item::forced_break()];
        let lines = LineBreaker::new(pt(10.0)).break_lines(&items);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].adjustment_ratio < 0.0);
    }
}
