//! The individual commands of an SVG path.
//!
//! [`PathSegment`] models the ten path commands of the `d` microsyntax. Each
//! command except `closepath` exists in an absolute and a relative form, which
//! the `abs` field records: the form the author wrote is preserved exactly, so
//! a parsed path can be re-serialized without changing its command letters.

use std::fmt;

use super::write::{self, Style};

/// One command of an SVG path.
///
/// The variants are named after the path grammar's productions rather than
/// after their command letters; [`PathSegment::command`] returns the letter,
/// including its case. `Z` and `z` are identical in effect and share the single
/// [`PathSegment::Close`] variant.
///
/// Every numeric field stores the value exactly as the author wrote it: no
/// normalization is applied at this layer. In particular an elliptical arc may
/// carry negative radii and an out-of-range `x_axis_rotation`; the geometry
/// layer applies the spec's out-of-range handling (SVG 1.1 Appendix F.6.2).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PathSegment {
    /// `M`/`m` — start a new subpath at the given point.
    MoveTo {
        /// True for `M`, false for `m`.
        abs: bool,
        /// The x coordinate, or the x offset when `abs` is false.
        x: f64,
        /// The y coordinate, or the y offset when `abs` is false.
        y: f64,
    },
    /// `L`/`l` — draw a straight line to the given point.
    LineTo {
        /// True for `L`, false for `l`.
        abs: bool,
        /// The x coordinate, or the x offset when `abs` is false.
        x: f64,
        /// The y coordinate, or the y offset when `abs` is false.
        y: f64,
    },
    /// `H`/`h` — draw a horizontal line, keeping the current y coordinate.
    HorizontalTo {
        /// True for `H`, false for `h`.
        abs: bool,
        /// The x coordinate, or the x offset when `abs` is false.
        x: f64,
    },
    /// `V`/`v` — draw a vertical line, keeping the current x coordinate.
    VerticalTo {
        /// True for `V`, false for `v`.
        abs: bool,
        /// The y coordinate, or the y offset when `abs` is false.
        y: f64,
    },
    /// `C`/`c` — draw a cubic Bézier curve with two explicit control points.
    CurveTo {
        /// True for `C`, false for `c`.
        abs: bool,
        /// The x coordinate of the first control point.
        x1: f64,
        /// The y coordinate of the first control point.
        y1: f64,
        /// The x coordinate of the second control point.
        x2: f64,
        /// The y coordinate of the second control point.
        y2: f64,
        /// The x coordinate of the end point.
        x: f64,
        /// The y coordinate of the end point.
        y: f64,
    },
    /// `S`/`s` — draw a cubic Bézier curve whose first control point is the
    /// reflection of the previous curve's second control point.
    SmoothCurveTo {
        /// True for `S`, false for `s`.
        abs: bool,
        /// The x coordinate of the second control point.
        x2: f64,
        /// The y coordinate of the second control point.
        y2: f64,
        /// The x coordinate of the end point.
        x: f64,
        /// The y coordinate of the end point.
        y: f64,
    },
    /// `Q`/`q` — draw a quadratic Bézier curve with one explicit control point.
    QuadTo {
        /// True for `Q`, false for `q`.
        abs: bool,
        /// The x coordinate of the control point.
        x1: f64,
        /// The y coordinate of the control point.
        y1: f64,
        /// The x coordinate of the end point.
        x: f64,
        /// The y coordinate of the end point.
        y: f64,
    },
    /// `T`/`t` — draw a quadratic Bézier curve whose control point is the
    /// reflection of the previous quadratic curve's control point.
    SmoothQuadTo {
        /// True for `T`, false for `t`.
        abs: bool,
        /// The x coordinate of the end point.
        x: f64,
        /// The y coordinate of the end point.
        y: f64,
    },
    /// `A`/`a` — draw an elliptical arc to the given point.
    ArcTo {
        /// True for `A`, false for `a`.
        abs: bool,
        /// The x radius of the ellipse, as authored (may be negative).
        rx: f64,
        /// The y radius of the ellipse, as authored (may be negative).
        ry: f64,
        /// The rotation of the ellipse's x axis, in degrees, as authored (not
        /// reduced modulo 360).
        x_axis_rotation: f64,
        /// The large-arc flag: true selects the arc sweeping more than 180°.
        large_arc: bool,
        /// The sweep flag: true selects the arc drawn in the direction of
        /// increasing angle.
        sweep: bool,
        /// The x coordinate of the end point.
        x: f64,
        /// The y coordinate of the end point.
        y: f64,
    },
    /// `Z`/`z` — close the current subpath with a straight line back to its
    /// start point.
    Close,
}

impl PathSegment {
    /// The command letter this segment was authored with, including its case.
    ///
    /// [`PathSegment::Close`] always reports the upper-case `Z`, because `Z`
    /// and `z` are the same command.
    #[must_use]
    pub const fn command(&self) -> char {
        match self {
            Self::MoveTo { abs, .. } => {
                if *abs {
                    'M'
                } else {
                    'm'
                }
            }
            Self::LineTo { abs, .. } => {
                if *abs {
                    'L'
                } else {
                    'l'
                }
            }
            Self::HorizontalTo { abs, .. } => {
                if *abs {
                    'H'
                } else {
                    'h'
                }
            }
            Self::VerticalTo { abs, .. } => {
                if *abs {
                    'V'
                } else {
                    'v'
                }
            }
            Self::CurveTo { abs, .. } => {
                if *abs {
                    'C'
                } else {
                    'c'
                }
            }
            Self::SmoothCurveTo { abs, .. } => {
                if *abs {
                    'S'
                } else {
                    's'
                }
            }
            Self::QuadTo { abs, .. } => {
                if *abs {
                    'Q'
                } else {
                    'q'
                }
            }
            Self::SmoothQuadTo { abs, .. } => {
                if *abs {
                    'T'
                } else {
                    't'
                }
            }
            Self::ArcTo { abs, .. } => {
                if *abs {
                    'A'
                } else {
                    'a'
                }
            }
            Self::Close => 'Z',
        }
    }

    /// True when the segment's coordinates are absolute.
    ///
    /// [`PathSegment::Close`] reports `true`: it carries no coordinates, so the
    /// distinction does not apply to it.
    #[must_use]
    pub const fn is_absolute(&self) -> bool {
        match self {
            Self::MoveTo { abs, .. }
            | Self::LineTo { abs, .. }
            | Self::HorizontalTo { abs, .. }
            | Self::VerticalTo { abs, .. }
            | Self::CurveTo { abs, .. }
            | Self::SmoothCurveTo { abs, .. }
            | Self::QuadTo { abs, .. }
            | Self::SmoothQuadTo { abs, .. }
            | Self::ArcTo { abs, .. } => *abs,
            Self::Close => true,
        }
    }

    /// True when the segment's coordinates are relative to the current point.
    #[must_use]
    pub const fn is_relative(&self) -> bool {
        !self.is_absolute()
    }

    /// True for [`PathSegment::Close`].
    #[must_use]
    pub const fn is_close(&self) -> bool {
        matches!(self, Self::Close)
    }

    /// True for [`PathSegment::MoveTo`].
    #[must_use]
    pub const fn is_move_to(&self) -> bool {
        matches!(self, Self::MoveTo { .. })
    }

    /// The number of grammar arguments the command takes, counting each arc
    /// flag as one argument.
    #[must_use]
    pub const fn argument_count(&self) -> usize {
        match self {
            Self::Close => 0,
            Self::HorizontalTo { .. } | Self::VerticalTo { .. } => 1,
            Self::MoveTo { .. } | Self::LineTo { .. } | Self::SmoothQuadTo { .. } => 2,
            Self::QuadTo { .. } | Self::SmoothCurveTo { .. } => 4,
            Self::CurveTo { .. } => 6,
            Self::ArcTo { .. } => 7,
        }
    }

    /// True when every numeric field of the segment is finite.
    ///
    /// Path data may name magnitudes that overflow an `f64` (`1e999`), which
    /// saturate to an infinity while parsing. Such a segment cannot be
    /// serialized back into valid path data, so callers that round-trip
    /// untrusted input should check this first.
    #[must_use]
    pub fn is_finite(&self) -> bool {
        match *self {
            Self::Close => true,
            Self::HorizontalTo { x: value, .. } | Self::VerticalTo { y: value, .. } => {
                value.is_finite()
            }
            Self::MoveTo { x, y, .. }
            | Self::LineTo { x, y, .. }
            | Self::SmoothQuadTo { x, y, .. } => x.is_finite() && y.is_finite(),
            Self::QuadTo { x1, y1, x, y, .. } => {
                x1.is_finite() && y1.is_finite() && x.is_finite() && y.is_finite()
            }
            Self::SmoothCurveTo { x2, y2, x, y, .. } => {
                x2.is_finite() && y2.is_finite() && x.is_finite() && y.is_finite()
            }
            Self::CurveTo {
                x1,
                y1,
                x2,
                y2,
                x,
                y,
                ..
            } => {
                x1.is_finite()
                    && y1.is_finite()
                    && x2.is_finite()
                    && y2.is_finite()
                    && x.is_finite()
                    && y.is_finite()
            }
            Self::ArcTo {
                rx,
                ry,
                x_axis_rotation,
                x,
                y,
                ..
            } => {
                rx.is_finite()
                    && ry.is_finite()
                    && x_axis_rotation.is_finite()
                    && x.is_finite()
                    && y.is_finite()
            }
        }
    }

    /// The absolute form of this segment, given the current point.
    ///
    /// A segment that is already absolute is returned unchanged, so the
    /// operation is idempotent. `current` is ignored for
    /// [`PathSegment::Close`], which has no coordinates.
    ///
    /// This is the raw per-segment conversion; it does not know about the
    /// spec's rule that a path's leading `m` is treated as an `M`. Use
    /// [`PathData::to_absolute`](super::PathData::to_absolute) or
    /// [`SegmentWalker`](super::SegmentWalker) to convert a whole path.
    #[must_use]
    pub fn absolute_from(&self, current: (f64, f64)) -> Self {
        let (cx, cy) = current;
        match *self {
            Self::MoveTo { abs: false, x, y } => Self::MoveTo {
                abs: true,
                x: cx + x,
                y: cy + y,
            },
            Self::LineTo { abs: false, x, y } => Self::LineTo {
                abs: true,
                x: cx + x,
                y: cy + y,
            },
            Self::HorizontalTo { abs: false, x } => Self::HorizontalTo {
                abs: true,
                x: cx + x,
            },
            Self::VerticalTo { abs: false, y } => Self::VerticalTo {
                abs: true,
                y: cy + y,
            },
            Self::CurveTo {
                abs: false,
                x1,
                y1,
                x2,
                y2,
                x,
                y,
            } => Self::CurveTo {
                abs: true,
                x1: cx + x1,
                y1: cy + y1,
                x2: cx + x2,
                y2: cy + y2,
                x: cx + x,
                y: cy + y,
            },
            Self::SmoothCurveTo {
                abs: false,
                x2,
                y2,
                x,
                y,
            } => Self::SmoothCurveTo {
                abs: true,
                x2: cx + x2,
                y2: cy + y2,
                x: cx + x,
                y: cy + y,
            },
            Self::QuadTo {
                abs: false,
                x1,
                y1,
                x,
                y,
            } => Self::QuadTo {
                abs: true,
                x1: cx + x1,
                y1: cy + y1,
                x: cx + x,
                y: cy + y,
            },
            Self::SmoothQuadTo { abs: false, x, y } => Self::SmoothQuadTo {
                abs: true,
                x: cx + x,
                y: cy + y,
            },
            Self::ArcTo {
                abs: false,
                rx,
                ry,
                x_axis_rotation,
                large_arc,
                sweep,
                x,
                y,
            } => Self::ArcTo {
                abs: true,
                rx,
                ry,
                x_axis_rotation,
                large_arc,
                sweep,
                x: cx + x,
                y: cy + y,
            },
            already_absolute => already_absolute,
        }
    }

    /// The relative form of this segment, given the current point.
    ///
    /// A segment that is already relative is returned unchanged, so the
    /// operation is idempotent. Because the conversion subtracts the current
    /// point, converting an absolute path to relative and back is not
    /// guaranteed to be bit-identical in `f64` arithmetic.
    #[must_use]
    pub fn relative_from(&self, current: (f64, f64)) -> Self {
        let (cx, cy) = current;
        match *self {
            Self::MoveTo { abs: true, x, y } => Self::MoveTo {
                abs: false,
                x: x - cx,
                y: y - cy,
            },
            Self::LineTo { abs: true, x, y } => Self::LineTo {
                abs: false,
                x: x - cx,
                y: y - cy,
            },
            Self::HorizontalTo { abs: true, x } => Self::HorizontalTo {
                abs: false,
                x: x - cx,
            },
            Self::VerticalTo { abs: true, y } => Self::VerticalTo {
                abs: false,
                y: y - cy,
            },
            Self::CurveTo {
                abs: true,
                x1,
                y1,
                x2,
                y2,
                x,
                y,
            } => Self::CurveTo {
                abs: false,
                x1: x1 - cx,
                y1: y1 - cy,
                x2: x2 - cx,
                y2: y2 - cy,
                x: x - cx,
                y: y - cy,
            },
            Self::SmoothCurveTo {
                abs: true,
                x2,
                y2,
                x,
                y,
            } => Self::SmoothCurveTo {
                abs: false,
                x2: x2 - cx,
                y2: y2 - cy,
                x: x - cx,
                y: y - cy,
            },
            Self::QuadTo {
                abs: true,
                x1,
                y1,
                x,
                y,
            } => Self::QuadTo {
                abs: false,
                x1: x1 - cx,
                y1: y1 - cy,
                x: x - cx,
                y: y - cy,
            },
            Self::SmoothQuadTo { abs: true, x, y } => Self::SmoothQuadTo {
                abs: false,
                x: x - cx,
                y: y - cy,
            },
            Self::ArcTo {
                abs: true,
                rx,
                ry,
                x_axis_rotation,
                large_arc,
                sweep,
                x,
                y,
            } => Self::ArcTo {
                abs: false,
                rx,
                ry,
                x_axis_rotation,
                large_arc,
                sweep,
                x: x - cx,
                y: y - cy,
            },
            already_relative => already_relative,
        }
    }

    /// The current point after this segment has been drawn.
    ///
    /// `current` is the current point before the segment and `subpath_start`
    /// the start point of the subpath it belongs to, which
    /// [`PathSegment::Close`] returns to.
    #[must_use]
    pub fn end_point(&self, current: (f64, f64), subpath_start: (f64, f64)) -> (f64, f64) {
        let (cx, cy) = current;
        match *self {
            Self::Close => subpath_start,
            Self::HorizontalTo { abs, x } => (if abs { x } else { cx + x }, cy),
            Self::VerticalTo { abs, y } => (cx, if abs { y } else { cy + y }),
            Self::MoveTo { abs, x, y }
            | Self::LineTo { abs, x, y }
            | Self::SmoothQuadTo { abs, x, y }
            | Self::CurveTo { abs, x, y, .. }
            | Self::SmoothCurveTo { abs, x, y, .. }
            | Self::QuadTo { abs, x, y, .. }
            | Self::ArcTo { abs, x, y, .. } => {
                if abs {
                    (x, y)
                } else {
                    (cx + x, cy + y)
                }
            }
        }
    }
}

impl fmt::Display for PathSegment {
    /// Writes the segment in the canonical layout: the command letter, then
    /// its arguments, separated by single spaces.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&write::render(
            std::slice::from_ref(self),
            Style::Canonical,
            None,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_letters_record_the_authored_case() {
        assert_eq!(
            PathSegment::MoveTo {
                abs: true,
                x: 0.0,
                y: 0.0
            }
            .command(),
            'M'
        );
        assert_eq!(
            PathSegment::MoveTo {
                abs: false,
                x: 0.0,
                y: 0.0
            }
            .command(),
            'm'
        );
        assert_eq!(PathSegment::Close.command(), 'Z');
        assert_eq!(
            PathSegment::ArcTo {
                abs: false,
                rx: 1.0,
                ry: 1.0,
                x_axis_rotation: 0.0,
                large_arc: false,
                sweep: false,
                x: 0.0,
                y: 0.0
            }
            .command(),
            'a'
        );
    }

    #[test]
    fn closepath_reports_itself_as_absolute() {
        assert!(PathSegment::Close.is_absolute());
        assert!(!PathSegment::Close.is_relative());
        assert!(PathSegment::Close.is_close());
        assert_eq!(PathSegment::Close.argument_count(), 0);
    }

    #[test]
    fn argument_counts_match_the_grammar() {
        assert_eq!(
            PathSegment::HorizontalTo { abs: true, x: 1.0 }.argument_count(),
            1
        );
        assert_eq!(
            PathSegment::LineTo {
                abs: true,
                x: 1.0,
                y: 1.0
            }
            .argument_count(),
            2
        );
        assert_eq!(
            PathSegment::QuadTo {
                abs: true,
                x1: 1.0,
                y1: 1.0,
                x: 2.0,
                y: 2.0
            }
            .argument_count(),
            4
        );
        assert_eq!(
            PathSegment::CurveTo {
                abs: true,
                x1: 1.0,
                y1: 1.0,
                x2: 2.0,
                y2: 2.0,
                x: 3.0,
                y: 3.0
            }
            .argument_count(),
            6
        );
        assert_eq!(
            PathSegment::ArcTo {
                abs: true,
                rx: 1.0,
                ry: 1.0,
                x_axis_rotation: 0.0,
                large_arc: false,
                sweep: false,
                x: 1.0,
                y: 1.0
            }
            .argument_count(),
            7
        );
    }

    #[test]
    fn relative_segments_become_absolute_against_the_current_point() {
        let relative = PathSegment::CurveTo {
            abs: false,
            x1: 1.0,
            y1: 2.0,
            x2: 3.0,
            y2: 4.0,
            x: 5.0,
            y: 6.0,
        };
        assert_eq!(
            relative.absolute_from((10.0, 20.0)),
            PathSegment::CurveTo {
                abs: true,
                x1: 11.0,
                y1: 22.0,
                x2: 13.0,
                y2: 24.0,
                x: 15.0,
                y: 26.0,
            }
        );
    }

    #[test]
    fn absolute_conversion_is_idempotent() {
        let absolute = PathSegment::LineTo {
            abs: true,
            x: 1.0,
            y: 2.0,
        };
        assert_eq!(absolute.absolute_from((10.0, 10.0)), absolute);
        let relative = PathSegment::LineTo {
            abs: false,
            x: 1.0,
            y: 2.0,
        };
        assert_eq!(relative.relative_from((10.0, 10.0)), relative);
    }

    #[test]
    fn arc_conversion_only_moves_the_end_point() {
        let arc = PathSegment::ArcTo {
            abs: false,
            rx: -3.0,
            ry: 4.0,
            x_axis_rotation: 450.0,
            large_arc: true,
            sweep: false,
            x: 1.0,
            y: 2.0,
        };
        assert_eq!(
            arc.absolute_from((10.0, 10.0)),
            PathSegment::ArcTo {
                abs: true,
                rx: -3.0,
                ry: 4.0,
                x_axis_rotation: 450.0,
                large_arc: true,
                sweep: false,
                x: 11.0,
                y: 12.0,
            }
        );
    }

    #[test]
    fn end_points_follow_the_command_semantics() {
        let current = (10.0, 20.0);
        let subpath_start = (1.0, 2.0);
        assert_eq!(
            PathSegment::HorizontalTo { abs: true, x: 5.0 }.end_point(current, subpath_start),
            (5.0, 20.0)
        );
        assert_eq!(
            PathSegment::HorizontalTo { abs: false, x: 5.0 }.end_point(current, subpath_start),
            (15.0, 20.0)
        );
        assert_eq!(
            PathSegment::VerticalTo { abs: true, y: 5.0 }.end_point(current, subpath_start),
            (10.0, 5.0)
        );
        assert_eq!(
            PathSegment::VerticalTo { abs: false, y: 5.0 }.end_point(current, subpath_start),
            (10.0, 25.0)
        );
        assert_eq!(
            PathSegment::Close.end_point(current, subpath_start),
            subpath_start
        );
    }

    #[test]
    fn non_finite_coordinates_are_reported() {
        assert!(
            PathSegment::LineTo {
                abs: true,
                x: 1.0,
                y: 2.0
            }
            .is_finite()
        );
        assert!(
            !PathSegment::LineTo {
                abs: true,
                x: f64::INFINITY,
                y: 2.0
            }
            .is_finite()
        );
        assert!(PathSegment::Close.is_finite());
    }

    #[test]
    fn segments_display_in_the_canonical_layout() {
        assert_eq!(
            PathSegment::MoveTo {
                abs: true,
                x: 10.0,
                y: 20.0
            }
            .to_string(),
            "M 10 20"
        );
        assert_eq!(PathSegment::Close.to_string(), "Z");
        assert_eq!(
            PathSegment::ArcTo {
                abs: false,
                rx: 25.0,
                ry: 25.0,
                x_axis_rotation: 0.0,
                large_arc: true,
                sweep: false,
                x: -25.0,
                y: 25.0
            }
            .to_string(),
            "a 25 25 0 1 0 -25 25"
        );
    }
}
