//! Walking a path while tracking the current point.
//!
//! The walk implements the point-tracking half of the path semantics: the
//! current point, the start point of the current subpath (which `Z` returns
//! to), and the rule that a path whose first command is a relative `m` treats
//! that first pair as absolute.
//!
//! Reflected control points for the smooth commands `S`/`s` and `T`/`t` are
//! deliberately *not* computed here: they are curve state rather than path
//! state, and belong to the geometry layer that consumes these steps.

use std::iter::FusedIterator;
use std::slice;

use super::segment::PathSegment;

/// One step of a path walk: a segment together with the points it runs
/// between.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SegmentStep {
    /// The segment exactly as it was authored, absolute or relative.
    pub segment: PathSegment,
    /// The same segment rewritten in absolute form.
    ///
    /// A leading relative `m` appears here as an `M`, per the specification's
    /// rule that a path's first moveto is absolute whichever letter was used.
    pub absolute: PathSegment,
    /// The current point before the segment is drawn.
    pub start: (f64, f64),
    /// The current point after the segment is drawn.
    pub end: (f64, f64),
    /// The start point of the subpath the segment belongs to, which `Z`
    /// returns the current point to. For a moveto this is the point the
    /// moveto itself establishes.
    pub subpath_start: (f64, f64),
}

/// The point-tracking state of a walk.
#[derive(Debug, Clone, Copy)]
struct State {
    /// The current point.
    current: (f64, f64),
    /// The start point of the current subpath.
    subpath_start: (f64, f64),
    /// True until the first segment has been consumed.
    first: bool,
}

impl State {
    /// A walk positioned at the origin, as every path starts.
    const fn new() -> Self {
        Self {
            current: (0.0, 0.0),
            subpath_start: (0.0, 0.0),
            first: true,
        }
    }

    /// Advances over one segment.
    fn step(&mut self, segment: PathSegment) -> SegmentStep {
        let start = self.current;
        // "If a relative moveto (m) appears as the first element of the path,
        // then it is treated as a pair of absolute coordinates." Only the
        // moveto itself is affected: the implicit linetos that may follow it
        // stay relative, and are relative to the point it established.
        let absolute = match segment {
            PathSegment::MoveTo { abs: false, x, y } if self.first => {
                PathSegment::MoveTo { abs: true, x, y }
            }
            other => other.absolute_from(start),
        };
        self.first = false;
        let end = absolute.end_point(start, self.subpath_start);
        if absolute.is_move_to() {
            self.subpath_start = end;
        }
        self.current = end;
        SegmentStep {
            segment,
            absolute,
            start,
            end,
            subpath_start: self.subpath_start,
        }
    }
}

/// An iterator over a path's segments that tracks the current point.
///
/// Created by [`PathData::walk`](super::PathData::walk), or directly from any
/// segment slice with [`SegmentWalker::new`].
///
/// ```
/// use oxixml_svg::values::path::{PathData, SegmentWalker};
///
/// let path = PathData::parse("M 10 10 l 5 0 Z").expect("valid path data");
/// let ends: Vec<(f64, f64)> = path.walk().map(|step| step.end).collect();
/// assert_eq!(ends, vec![(10.0, 10.0), (15.0, 10.0), (10.0, 10.0)]);
///
/// let same = SegmentWalker::new(path.segments()).count();
/// assert_eq!(same, 3);
/// ```
#[derive(Debug, Clone)]
pub struct SegmentWalker<'a> {
    segments: slice::Iter<'a, PathSegment>,
    state: State,
}

impl<'a> SegmentWalker<'a> {
    /// Starts a walk over `segments`, with the current point at the origin.
    #[must_use]
    pub fn new(segments: &'a [PathSegment]) -> Self {
        Self {
            segments: segments.iter(),
            state: State::new(),
        }
    }

    /// The current point: the origin before the first step, and the end point
    /// of the most recent step afterwards.
    #[must_use]
    pub const fn current_point(&self) -> (f64, f64) {
        self.state.current
    }

    /// The start point of the subpath the walk is currently inside.
    #[must_use]
    pub const fn subpath_start(&self) -> (f64, f64) {
        self.state.subpath_start
    }

    /// The segments that have not been walked yet.
    #[must_use]
    pub fn remainder(&self) -> &'a [PathSegment] {
        self.segments.as_slice()
    }
}

impl Iterator for SegmentWalker<'_> {
    type Item = SegmentStep;

    fn next(&mut self) -> Option<Self::Item> {
        let segment = *self.segments.next()?;
        Some(self.state.step(segment))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.segments.size_hint()
    }
}

impl ExactSizeIterator for SegmentWalker<'_> {}

impl FusedIterator for SegmentWalker<'_> {}

/// Rewrites every segment in absolute form.
pub(super) fn to_absolute(segments: &[PathSegment]) -> Vec<PathSegment> {
    SegmentWalker::new(segments)
        .map(|step| step.absolute)
        .collect()
}

/// Rewrites every segment in relative form.
pub(super) fn to_relative(segments: &[PathSegment]) -> Vec<PathSegment> {
    SegmentWalker::new(segments)
        .map(|step| step.absolute.relative_from(step.start))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::PathData;
    use super::*;

    fn parse(input: &str) -> PathData {
        PathData::parse(input).unwrap_or_else(|error| panic!("{input:?}: {error}"))
    }

    #[test]
    fn a_leading_relative_moveto_is_treated_as_absolute() {
        let path = parse("m 10 20 l 5 5");
        assert_eq!(path.to_absolute().to_string(), "M 10 20 L 15 25");
        assert_eq!(
            path.to_absolute().segments()[0],
            PathSegment::MoveTo {
                abs: true,
                x: 10.0,
                y: 20.0
            }
        );
    }

    #[test]
    fn implicit_relative_linetos_after_a_leading_moveto_stay_relative() {
        // `m 10 20 5 5` is a moveto plus one implicit relative lineto; only
        // the moveto is promoted to absolute, and the lineto is relative to
        // the point it established.
        let path = parse("m 10 20 5 5");
        assert!(path.segments()[1].is_relative());
        assert_eq!(path.to_absolute().to_string(), "M 10 20 L 15 25");
    }

    #[test]
    fn closepath_returns_the_current_point_to_the_subpath_start() {
        let path = parse("M0,0 L10,0 L10,10 Z L20,20");
        let steps: Vec<SegmentStep> = path.walk().collect();
        assert_eq!(steps[3].end, (0.0, 0.0));
        assert_eq!(steps[3].subpath_start, (0.0, 0.0));
        // The lineto after the closepath starts from the subpath start, and
        // does not itself open a new subpath.
        assert_eq!(steps[4].start, (0.0, 0.0));
        assert_eq!(steps[4].end, (20.0, 20.0));
        assert_eq!(steps[4].subpath_start, (0.0, 0.0));
    }

    #[test]
    fn a_relative_moveto_after_a_closepath_starts_from_the_subpath_start() {
        let path = parse("M10,10 L20,20 Z m 1 1");
        let steps: Vec<SegmentStep> = path.walk().collect();
        assert_eq!(steps[2].end, (10.0, 10.0));
        assert_eq!(steps[3].start, (10.0, 10.0));
        assert_eq!(steps[3].end, (11.0, 11.0));
        assert_eq!(steps[3].subpath_start, (11.0, 11.0));
        assert_eq!(
            steps[3].absolute,
            PathSegment::MoveTo {
                abs: true,
                x: 11.0,
                y: 11.0
            }
        );
    }

    #[test]
    fn horizontal_and_vertical_commands_keep_their_kind_when_absolutized() {
        let path = parse("M10 10 h5 v-5");
        assert_eq!(path.to_absolute().to_string(), "M 10 10 H 15 V 5");
        assert_eq!(
            path.to_absolute().segments()[1],
            PathSegment::HorizontalTo { abs: true, x: 15.0 }
        );
        assert_eq!(
            path.to_absolute().segments()[2],
            PathSegment::VerticalTo { abs: true, y: 5.0 }
        );
        let steps: Vec<SegmentStep> = path.walk().collect();
        assert_eq!(steps[1].end, (15.0, 10.0));
        assert_eq!(steps[2].end, (15.0, 5.0));
    }

    #[test]
    fn relative_curves_and_arcs_are_offset_by_the_current_point() {
        let path = parse("M10 10 c1 1 2 2 3 3 s1 1 2 2 q1 1 2 2 t1 1 a5 6 7 0 1 1 1");
        let absolute = path.to_absolute();
        assert_eq!(
            absolute.segments()[1],
            PathSegment::CurveTo {
                abs: true,
                x1: 11.0,
                y1: 11.0,
                x2: 12.0,
                y2: 12.0,
                x: 13.0,
                y: 13.0
            }
        );
        assert_eq!(
            absolute.segments()[2],
            PathSegment::SmoothCurveTo {
                abs: true,
                x2: 14.0,
                y2: 14.0,
                x: 15.0,
                y: 15.0
            }
        );
        assert_eq!(
            absolute.segments()[3],
            PathSegment::QuadTo {
                abs: true,
                x1: 16.0,
                y1: 16.0,
                x: 17.0,
                y: 17.0
            }
        );
        assert_eq!(
            absolute.segments()[4],
            PathSegment::SmoothQuadTo {
                abs: true,
                x: 18.0,
                y: 18.0
            }
        );
        assert_eq!(
            absolute.segments()[5],
            PathSegment::ArcTo {
                abs: true,
                rx: 5.0,
                ry: 6.0,
                x_axis_rotation: 7.0,
                large_arc: false,
                sweep: true,
                x: 19.0,
                y: 19.0
            }
        );
    }

    #[test]
    fn absolute_paths_are_left_alone() {
        let path = parse("M0 0 L10 10 H20 V30 Z");
        assert_eq!(path.to_absolute(), path);
    }

    #[test]
    fn walking_reports_the_start_and_end_of_every_segment() {
        let path = parse("M0 0 L3 4 l3 4");
        let steps: Vec<SegmentStep> = path.walk().collect();
        assert_eq!(steps.len(), 3);
        assert_eq!(steps[0].start, (0.0, 0.0));
        assert_eq!(steps[1].start, (0.0, 0.0));
        assert_eq!(steps[1].end, (3.0, 4.0));
        assert_eq!(steps[2].start, (3.0, 4.0));
        assert_eq!(steps[2].end, (6.0, 8.0));
        assert_eq!(steps[2].segment.command(), 'l');
        assert_eq!(steps[2].absolute.command(), 'L');
    }

    #[test]
    fn the_walker_exposes_its_own_progress() {
        let path = parse("M1 2 L3 4");
        let mut walker = path.walk();
        assert_eq!(walker.current_point(), (0.0, 0.0));
        assert_eq!(walker.len(), 2);
        let _ = walker.next();
        assert_eq!(walker.current_point(), (1.0, 2.0));
        assert_eq!(walker.subpath_start(), (1.0, 2.0));
        assert_eq!(walker.remainder().len(), 1);
        let _ = walker.next();
        assert_eq!(walker.current_point(), (3.0, 4.0));
        assert_eq!(walker.next(), None);
    }

    #[test]
    fn converting_to_relative_preserves_the_geometry() {
        let path = parse("M10 10 L20 20 H30 V40 C1 1 2 2 3 3 Z");
        let relative = path.to_relative();
        assert!(
            relative
                .segments()
                .iter()
                .all(|segment| segment.is_relative() || segment.is_close())
        );
        assert_eq!(relative.to_absolute(), path.to_absolute());
    }

    #[test]
    fn an_empty_path_walks_zero_steps() {
        let path = PathData::new();
        assert_eq!(path.walk().count(), 0);
        assert_eq!(path.to_absolute(), path);
        assert_eq!(path.to_relative(), path);
    }
}
