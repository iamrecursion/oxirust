//! Rendering path data back into the `d` attribute microsyntax.
//!
//! Two layouts are produced from one token walker: a canonical layout that
//! separates every token with a single space and repeats every command letter,
//! and a compact layout that drops separators wherever the path grammar keeps
//! the result unambiguous and omits command letters that the grammar's
//! implicit-repetition rule can restore.
//!
//! Both layouts are strictly conformant to the `svg-path` grammar; the compact
//! layout never relies on parser leniency to be re-readable.

use super::segment::PathSegment;

/// The layout used when rendering a segment list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Style {
    /// Every token separated by one space, every command letter present.
    Canonical,
    /// Separators and command letters elided wherever the grammar allows.
    Compact,
}

/// One rendered argument of a path command.
#[derive(Debug, Clone, Copy)]
enum Arg {
    /// A `<number>` argument.
    Number(f64),
    /// An elliptical-arc `flag` argument, rendered as a single `0` or `1`.
    Flag(bool),
}

/// The arguments of one command, stored inline (the longest command,
/// `elliptical-arc`, takes seven).
#[derive(Debug, Clone, Copy)]
struct Args {
    items: [Arg; 7],
    len: usize,
}

impl Args {
    /// Copies up to seven arguments into the inline buffer.
    fn new(items: &[Arg]) -> Self {
        let mut buffer = [Arg::Number(0.0); 7];
        let len = items.len().min(buffer.len());
        buffer[..len].copy_from_slice(&items[..len]);
        Self { items: buffer, len }
    }

    /// The arguments that were stored.
    fn as_slice(&self) -> &[Arg] {
        &self.items[..self.len]
    }
}

/// The kind of token that was emitted last, which decides whether the next
/// token needs a separator in front of it.
#[derive(Debug, Clone, Copy)]
enum Prev {
    /// Nothing has been emitted yet, or the last thing emitted was a command
    /// letter — both are self-delimiting, so no separator is ever needed.
    None,
    /// A `<number>` was emitted. `fractional` records whether it already
    /// contains a `.` (or an exponent), which decides whether a following
    /// `.`-leading number would merge into it.
    Number {
        /// True when the token already contains a decimal point or exponent.
        fractional: bool,
    },
    /// An arc flag was emitted. Flags are exactly one character wide in the
    /// grammar, so any following token is self-delimiting against them.
    Flag,
}

/// Renders `segments` in the requested layout.
///
/// `precision` rounds every number to that many decimal places (round-half-to-
/// even, as `format!` defines) before trailing zeros are trimmed; `None` emits
/// the shortest representation that round-trips through `f64`.
pub(super) fn render(segments: &[PathSegment], style: Style, precision: Option<usize>) -> String {
    let compact = matches!(style, Style::Compact);
    let mut out = String::new();
    let mut pending: Option<char> = None;
    let mut prev = Prev::None;
    for segment in segments {
        let letter = segment.command();
        let implicit = compact && pending.is_some_and(|previous| implicit_ok(previous, letter));
        if implicit {
            pending = Some(letter);
        } else {
            if !compact && !out.is_empty() {
                out.push(' ');
            }
            out.push(letter);
            pending = Some(letter);
            prev = Prev::None;
        }
        for arg in args(*segment).as_slice() {
            match *arg {
                Arg::Number(value) => {
                    let token = number_token(value, precision, compact);
                    if !compact || needs_separator(prev, &token) {
                        out.push(' ');
                    }
                    prev = Prev::Number {
                        fractional: token.contains(['.', 'e', 'E']),
                    };
                    out.push_str(&token);
                }
                Arg::Flag(value) => {
                    let token = if value { "1" } else { "0" };
                    if !compact || needs_separator(prev, token) {
                        out.push(' ');
                    }
                    prev = Prev::Flag;
                    out.push_str(token);
                }
            }
        }
    }
    out
}

/// True when a segment starting with `letter` may drop its command letter
/// because `pending` — the letter currently in effect — already implies it.
///
/// The path grammar implies a repeated command letter for any argument group
/// that follows another group of the same command, and implies `L`/`l` for the
/// coordinate pairs that follow an `M`/`m`. Closepath takes no arguments, so it
/// can never be implied, and a second moveto is never implied (further pairs
/// after a moveto are linetos).
const fn implicit_ok(pending: char, letter: char) -> bool {
    match (pending, letter) {
        ('M', 'L') | ('m', 'l') => true,
        ('M' | 'm' | 'Z' | 'z', _) => false,
        _ => pending == letter,
    }
}

/// The rendered arguments of one command, in grammar order.
fn args(segment: PathSegment) -> Args {
    match segment {
        PathSegment::MoveTo { x, y, .. }
        | PathSegment::LineTo { x, y, .. }
        | PathSegment::SmoothQuadTo { x, y, .. } => Args::new(&[Arg::Number(x), Arg::Number(y)]),
        PathSegment::HorizontalTo { x, .. } => Args::new(&[Arg::Number(x)]),
        PathSegment::VerticalTo { y, .. } => Args::new(&[Arg::Number(y)]),
        PathSegment::CurveTo {
            x1,
            y1,
            x2,
            y2,
            x,
            y,
            ..
        } => Args::new(&[
            Arg::Number(x1),
            Arg::Number(y1),
            Arg::Number(x2),
            Arg::Number(y2),
            Arg::Number(x),
            Arg::Number(y),
        ]),
        PathSegment::SmoothCurveTo { x2, y2, x, y, .. } => Args::new(&[
            Arg::Number(x2),
            Arg::Number(y2),
            Arg::Number(x),
            Arg::Number(y),
        ]),
        PathSegment::QuadTo { x1, y1, x, y, .. } => Args::new(&[
            Arg::Number(x1),
            Arg::Number(y1),
            Arg::Number(x),
            Arg::Number(y),
        ]),
        PathSegment::ArcTo {
            rx,
            ry,
            x_axis_rotation,
            large_arc,
            sweep,
            x,
            y,
            ..
        } => Args::new(&[
            Arg::Number(rx),
            Arg::Number(ry),
            Arg::Number(x_axis_rotation),
            Arg::Flag(large_arc),
            Arg::Flag(sweep),
            Arg::Number(x),
            Arg::Number(y),
        ]),
        PathSegment::Close => Args::new(&[]),
    }
}

/// Formats one number.
fn number_token(value: f64, precision: Option<usize>, compact: bool) -> String {
    let mut text = match precision {
        Some(places) => {
            let mut rounded = format!("{value:.places$}");
            trim_trailing_zeros(&mut rounded);
            rounded
        }
        None => format!("{value}"),
    };
    if compact {
        strip_leading_zero(&mut text);
    }
    text
}

/// Removes the trailing zeros — and then a bare trailing point — of a fixed
/// precision rendering, so `1.500` becomes `1.5` and `2.000` becomes `2`.
fn trim_trailing_zeros(text: &mut String) {
    if !text.contains('.') {
        return;
    }
    while text.ends_with('0') {
        text.pop();
    }
    if text.ends_with('.') {
        text.pop();
    }
}

/// Drops the redundant integer-part zero of `0.5`/`-0.5`, which the path
/// grammar's `floating-point-constant` production allows to be omitted.
fn strip_leading_zero(text: &mut String) {
    if text.starts_with("0.") {
        text.remove(0);
    } else if text.starts_with("-0.") {
        text.remove(1);
    }
}

/// True when `token` must be separated from the previously emitted token to
/// keep the two from lexing as one.
fn needs_separator(prev: Prev, token: &str) -> bool {
    match prev {
        Prev::None | Prev::Flag => false,
        Prev::Number { fractional } => match token.as_bytes().first() {
            // A sign always starts a new number.
            Some(b'-' | b'+') => false,
            // A point continues the previous number unless it already had one.
            Some(b'.') => !fractional,
            // Digits (and anything else) would merge.
            _ => true,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn canonical(segments: &[PathSegment]) -> String {
        render(segments, Style::Canonical, None)
    }

    fn compact(segments: &[PathSegment]) -> String {
        render(segments, Style::Compact, None)
    }

    #[test]
    fn canonical_layout_spaces_every_token() {
        let segments = [
            PathSegment::MoveTo {
                abs: true,
                x: 10.0,
                y: 20.0,
            },
            PathSegment::LineTo {
                abs: true,
                x: 30.0,
                y: 40.0,
            },
            PathSegment::Close,
        ];
        assert_eq!(canonical(&segments), "M 10 20 L 30 40 Z");
    }

    #[test]
    fn an_empty_segment_list_renders_as_the_empty_string() {
        assert_eq!(canonical(&[]), "");
        assert_eq!(compact(&[]), "");
    }

    #[test]
    fn integers_render_without_a_decimal_point() {
        assert_eq!(number_token(10.0, None, false), "10");
        assert_eq!(number_token(-4.0, None, false), "-4");
        assert_eq!(number_token(0.5, None, false), "0.5");
        assert_eq!(number_token(0.5, None, true), ".5");
        assert_eq!(number_token(-0.5, None, true), "-.5");
        assert_eq!(number_token(1.25, None, true), "1.25");
    }

    #[test]
    fn precision_rounds_then_trims() {
        assert_eq!(number_token(1.0 / 3.0, Some(3), false), "0.333");
        assert_eq!(number_token(2.0, Some(6), false), "2");
        assert_eq!(number_token(1.5, Some(6), false), "1.5");
        assert_eq!(number_token(1.5, Some(0), false), "2");
        assert_eq!(number_token(0.5, Some(3), true), ".5");
    }

    #[test]
    fn compact_packs_arc_flags_but_keeps_the_rotation_separator() {
        let segments = [PathSegment::ArcTo {
            abs: false,
            rx: 1.0,
            ry: 1.0,
            x_axis_rotation: 0.0,
            large_arc: false,
            sweep: true,
            x: 1.0,
            y: 0.0,
        }];
        assert_eq!(compact(&segments), "a1 1 0 011 0");
    }

    #[test]
    fn compact_keeps_the_rotation_separator_for_non_zero_rotations() {
        let segments = [PathSegment::ArcTo {
            abs: false,
            rx: 1.0,
            ry: 1.0,
            x_axis_rotation: 45.0,
            large_arc: false,
            sweep: true,
            x: 1.0,
            y: 0.0,
        }];
        assert_eq!(compact(&segments), "a1 1 45 011 0");
    }

    #[test]
    fn compact_separates_a_point_leading_number_only_when_it_would_merge() {
        let merging = [PathSegment::MoveTo {
            abs: true,
            x: 5.0,
            y: 0.5,
        }];
        assert_eq!(compact(&merging), "M5 .5");
        let safe = [PathSegment::MoveTo {
            abs: true,
            x: 1.5,
            y: 0.5,
        }];
        assert_eq!(compact(&safe), "M1.5.5");
    }

    #[test]
    fn compact_uses_signs_as_separators() {
        let segments = [
            PathSegment::MoveTo {
                abs: true,
                x: 0.0,
                y: 0.0,
            },
            PathSegment::LineTo {
                abs: true,
                x: -1.0,
                y: -2.0,
            },
        ];
        assert_eq!(compact(&segments), "M0 0-1-2");
    }

    #[test]
    fn compact_drops_repeated_and_implied_command_letters() {
        let segments = [
            PathSegment::MoveTo {
                abs: true,
                x: 0.0,
                y: 0.0,
            },
            PathSegment::LineTo {
                abs: true,
                x: 1.0,
                y: 1.0,
            },
            PathSegment::LineTo {
                abs: true,
                x: 2.0,
                y: 2.0,
            },
        ];
        assert_eq!(compact(&segments), "M0 0 1 1 2 2");
    }

    #[test]
    fn compact_keeps_a_letter_whose_case_differs_from_the_pending_command() {
        let segments = [
            PathSegment::MoveTo {
                abs: true,
                x: 0.0,
                y: 0.0,
            },
            PathSegment::LineTo {
                abs: false,
                x: 1.0,
                y: 1.0,
            },
            PathSegment::MoveTo {
                abs: false,
                x: 2.0,
                y: 2.0,
            },
        ];
        assert_eq!(compact(&segments), "M0 0l1 1m2 2");
    }

    #[test]
    fn compact_never_implies_a_closepath() {
        let segments = [
            PathSegment::MoveTo {
                abs: true,
                x: 0.0,
                y: 0.0,
            },
            PathSegment::Close,
            PathSegment::Close,
        ];
        assert_eq!(compact(&segments), "M0 0ZZ");
    }

    #[test]
    fn implicit_repetition_table_matches_the_grammar() {
        assert!(implicit_ok('M', 'L'));
        assert!(implicit_ok('m', 'l'));
        assert!(!implicit_ok('M', 'l'));
        assert!(!implicit_ok('m', 'L'));
        assert!(!implicit_ok('M', 'M'));
        assert!(!implicit_ok('Z', 'Z'));
        assert!(implicit_ok('L', 'L'));
        assert!(implicit_ok('a', 'a'));
        assert!(!implicit_ok('A', 'a'));
    }

    #[test]
    fn separator_rules_cover_every_token_boundary() {
        assert!(!needs_separator(Prev::None, "1"));
        assert!(!needs_separator(Prev::Flag, "1"));
        assert!(!needs_separator(Prev::Flag, ".5"));
        assert!(needs_separator(Prev::Number { fractional: false }, "1"));
        assert!(needs_separator(Prev::Number { fractional: false }, ".5"));
        assert!(!needs_separator(Prev::Number { fractional: true }, ".5"));
        assert!(!needs_separator(Prev::Number { fractional: false }, "-1"));
        assert!(!needs_separator(Prev::Number { fractional: true }, "-.5"));
    }
}
