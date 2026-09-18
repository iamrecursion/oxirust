//! Path data — the `d` attribute microsyntax.
//!
//! [`PathData`] is a list of [`PathSegment`]s, one per argument group of the
//! `svg-path` grammar (SVG 1.1 §8.3, SVG 2 §9.3.9). The parser keeps the path
//! exactly as it was authored: absolute and relative commands stay apart,
//! elliptical-arc radii keep their sign, and implicit command repetition is
//! expanded into explicit segments that still carry the letter case of the
//! command that implied them.
//!
//! # Grammar
//!
//! ```text
//! svg-path ::= wsp* moveto-drawto-command-groups? wsp*
//! moveto-drawto-command-group ::= moveto wsp* drawto-commands?
//! moveto ::= ("M"|"m") wsp* moveto-argument-sequence
//! moveto-argument-sequence ::= coordinate-pair
//!     | coordinate-pair comma-wsp? lineto-argument-sequence
//! elliptical-arc-argument ::= nonnegative-number comma-wsp? nonnegative-number
//!     comma-wsp? number comma-wsp flag comma-wsp? flag comma-wsp?
//!     coordinate-pair
//! comma-wsp ::= (wsp+ comma? wsp*) | (comma wsp*)
//! ```
//!
//! Three consequences of that grammar are worth spelling out, because they are
//! where path data most often surprises:
//!
//! * **The first command must be a moveto.** A value whose first non-white-space
//!   token is anything else is entirely in error and renders nothing at all.
//! * **Command letters may be omitted.** Further argument groups repeat the
//!   previous command, and the coordinate pairs after a moveto are linetos of
//!   the same absoluteness — `M0 0 10 10` is a moveto plus an absolute lineto,
//!   `m0 0 10 10` a moveto plus a relative one.
//! * **Separators are optional wherever a token is self-delimiting.** A sign or
//!   a second decimal point starts a new number, so `M10-20L.5.5` is four
//!   numbers, and arc flags are single digits, so `a1 1 0 011 0` is
//!   `large-arc = 0`, `sweep = 1`, endpoint `(1, 0)`.
//!
//! # Errors and recovery
//!
//! SVG's error handling for path data (SVG 2 §9.5.4) renders everything up to
//! but not including the command containing the first error, and within that
//! command everything up to and including the last correctly formed segment.
//! [`PathData::parse_recover`] returns exactly that prefix alongside the error;
//! [`PathData::parse`] is the strict entry point that fails instead. Every
//! error carries the 0-based byte offset into the value where the grammar broke.
//!
//! ```
//! use oxixml_svg::values::path::PathData;
//!
//! let (path, error) = PathData::parse_recover("M10 10 L20 20 X30 30 L40 40");
//! assert_eq!(path.to_string(), "M 10 10 L 20 20");
//! assert_eq!(error.and_then(|error| error.offset()), Some(14));
//! ```
//!
//! # Deviations from the letter of the grammar
//!
//! * `nonnegative-number` forbids a sign on an elliptical arc's radii, yet the
//!   specification's own out-of-range handling (SVG 1.1 Appendix F.6.2) says a
//!   negative radius has its sign dropped, which presupposes that it parses.
//!   Signed radii are therefore accepted and stored **as authored**; taking the
//!   absolute value is left to the geometry layer, as is reducing
//!   `x_axis_rotation` modulo 360.
//! * The grammar makes `comma-wsp` mandatory between an arc's rotation and its
//!   first flag. No separate check enforces it: the shared number scanner is
//!   greedy, so `a1 1 0011 0` reads `0011` as the rotation and then fails on the
//!   flags, exactly as a strict reading requires.
//! * Only white space may stand between two commands, so a trailing comma such
//!   as `M0 0L1 1,L2 2` is reported at the comma rather than accepted.
//!
//! # Serialization
//!
//! [`Display`](core::fmt::Display) writes a canonical, readable form
//! (`M 10 20 L 30 40`), [`PathData::to_string_compact`] the shortest strictly
//! conformant form, and [`PathData::fmt_with_precision`] rounds every
//! coordinate first. Parsing any of them yields the same segments again.

mod parse;
mod segment;
mod walk;
mod write;

use std::fmt;
use std::str::FromStr;

use crate::error::ValueError;

pub use segment::PathSegment;
pub use walk::{SegmentStep, SegmentWalker};

use write::Style;

/// The parsed contents of a `d` attribute.
///
/// ```
/// use oxixml_svg::values::path::{PathData, PathSegment};
///
/// let path = PathData::parse("M0 0 10 10 z").expect("valid path data");
/// assert_eq!(path.len(), 3);
/// assert_eq!(
///     path.segments()[1],
///     PathSegment::LineTo { abs: true, x: 10.0, y: 10.0 }
/// );
/// assert_eq!(path.to_string_compact(), "M0 0 10 10Z");
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PathData {
    segments: Vec<PathSegment>,
}

impl PathData {
    /// Creates an empty path, which is a valid `d` value that draws nothing.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            segments: Vec::new(),
        }
    }

    /// Creates an empty path with room for `capacity` segments.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            segments: Vec::with_capacity(capacity),
        }
    }

    /// Creates a path from segments that were built elsewhere.
    #[must_use]
    pub const fn from_segments(segments: Vec<PathSegment>) -> Self {
        Self { segments }
    }

    /// Parses a whole `d` attribute value, failing on the first error.
    ///
    /// # Errors
    ///
    /// Returns the [`ValueError`] describing the first grammar violation, with
    /// the byte offset into `input` at which it was found. Use
    /// [`PathData::parse_recover`] to obtain the renderable prefix as well.
    pub fn parse(input: &str) -> Result<Self, ValueError> {
        let (segments, error) = parse::run(input);
        match error {
            Some(error) => Err(error),
            None => Ok(Self { segments }),
        }
    }

    /// Parses a whole `d` attribute value, keeping the part that is valid.
    ///
    /// The returned path is the prefix SVG's error handling says to render:
    /// every complete segment before the error, with any half-read argument
    /// group discarded. A path whose very first command is not a moveto is
    /// entirely in error, so the prefix is then empty.
    ///
    /// ```
    /// use oxixml_svg::values::path::PathData;
    ///
    /// let (path, error) = PathData::parse_recover("M10,10 L 20,20 30");
    /// assert_eq!(path.to_string(), "M 10 10 L 20 20");
    /// assert!(error.is_some());
    ///
    /// let (nothing, error) = PathData::parse_recover("L 10 10");
    /// assert!(nothing.is_empty());
    /// assert!(error.is_some());
    /// ```
    #[must_use]
    pub fn parse_recover(input: &str) -> (Self, Option<ValueError>) {
        let (segments, error) = parse::run(input);
        (Self { segments }, error)
    }

    /// The segments of the path, in document order.
    #[must_use]
    pub fn segments(&self) -> &[PathSegment] {
        &self.segments
    }

    /// The segments of the path, mutably.
    pub fn segments_mut(&mut self) -> &mut [PathSegment] {
        &mut self.segments
    }

    /// Consumes the path and returns its segments.
    #[must_use]
    pub fn into_segments(self) -> Vec<PathSegment> {
        self.segments
    }

    /// The number of segments.
    #[must_use]
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    /// True when the path has no segments, which is the state of a `d=""`
    /// attribute.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// Borrows the segments as an iterator.
    pub fn iter(&self) -> std::slice::Iter<'_, PathSegment> {
        self.segments.iter()
    }

    /// Appends one segment.
    pub fn push(&mut self, segment: PathSegment) {
        self.segments.push(segment);
    }

    /// Removes and returns the last segment.
    pub fn pop(&mut self) -> Option<PathSegment> {
        self.segments.pop()
    }

    /// Removes every segment, keeping the allocated capacity.
    pub fn clear(&mut self) {
        self.segments.clear();
    }

    /// True when every coordinate of every segment is finite.
    ///
    /// Path data may name magnitudes that overflow an `f64` (`M1e999 0`), which
    /// saturate to an infinity while parsing. Such a path cannot be serialized
    /// back into valid path data.
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.segments.iter().all(PathSegment::is_finite)
    }

    /// Walks the path, yielding each segment with the points it runs between.
    ///
    /// The walk applies the current-point rules: `Z` returns the current point
    /// to the start of the subpath, and a leading relative `m` is treated as an
    /// absolute `M`.
    #[must_use]
    pub fn walk(&self) -> SegmentWalker<'_> {
        SegmentWalker::new(&self.segments)
    }

    /// Rewrites every segment in absolute form.
    ///
    /// Commands keep their kind: a horizontal lineto stays a horizontal lineto
    /// with an absolute x coordinate rather than becoming a general lineto, and
    /// the smooth commands keep their implied control points. Flattening those
    /// distinctions is the geometry layer's normalization step, not this one.
    ///
    /// ```
    /// use oxixml_svg::values::path::PathData;
    ///
    /// let path = PathData::parse("m 10 20 l 5 5 h 5 z").expect("valid path data");
    /// assert_eq!(path.to_absolute().to_string(), "M 10 20 L 15 25 H 20 Z");
    /// ```
    #[must_use]
    pub fn to_absolute(&self) -> Self {
        Self {
            segments: walk::to_absolute(&self.segments),
        }
    }

    /// Rewrites every segment in relative form.
    ///
    /// The leading moveto becomes an `m`, which the specification reads back as
    /// absolute, so the geometry is preserved. Because the conversion subtracts
    /// the current point, it is not guaranteed to be bit-identical in `f64`
    /// arithmetic when converted back.
    #[must_use]
    pub fn to_relative(&self) -> Self {
        Self {
            segments: walk::to_relative(&self.segments),
        }
    }

    /// Renders the path in the shortest strictly conformant form.
    ///
    /// Separators are dropped wherever the next token is self-delimiting,
    /// repeated command letters are omitted, a redundant leading zero is
    /// dropped from fractions, and arc flags are packed together. The result
    /// re-parses to the same segments.
    ///
    /// ```
    /// use oxixml_svg::values::path::PathData;
    ///
    /// let path = PathData::parse("M 0 0 L 1 1 L -2.5 0.5 a 1 1 0 0 1 1 0")
    ///     .expect("valid path data");
    /// assert_eq!(path.to_string_compact(), "M0 0 1 1-2.5.5a1 1 0 011 0");
    /// assert_eq!(PathData::parse(&path.to_string_compact()), Ok(path));
    /// ```
    #[must_use]
    pub fn to_string_compact(&self) -> String {
        write::render(&self.segments, Style::Compact, None)
    }

    /// Renders the path in the canonical layout with every coordinate rounded
    /// to `precision` decimal places.
    ///
    /// Rounding is round-half-to-even, and trailing zeros — and a trailing
    /// decimal point — are trimmed afterwards, so the output is stable across
    /// runs and free of `1.500000`-style noise.
    ///
    /// ```
    /// use oxixml_svg::values::path::PathData;
    ///
    /// let path = PathData::parse("M 0.3333333333 2.0000001").expect("valid path data");
    /// assert_eq!(path.fmt_with_precision(3), "M 0.333 2");
    /// ```
    #[must_use]
    pub fn fmt_with_precision(&self, precision: usize) -> String {
        write::render(&self.segments, Style::Canonical, Some(precision))
    }

    /// Renders the path in the compact layout with every coordinate rounded to
    /// `precision` decimal places.
    #[must_use]
    pub fn to_string_compact_with_precision(&self, precision: usize) -> String {
        write::render(&self.segments, Style::Compact, Some(precision))
    }
}

impl fmt::Display for PathData {
    /// Writes the canonical layout: every command letter present, every token
    /// separated by a single space. An empty path writes nothing.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&write::render(&self.segments, Style::Canonical, None))
    }
}

impl FromStr for PathData {
    type Err = ValueError;

    /// Parses a whole `d` attribute value strictly; see [`PathData::parse`].
    fn from_str(input: &str) -> Result<Self, Self::Err> {
        Self::parse(input)
    }
}

impl From<Vec<PathSegment>> for PathData {
    fn from(segments: Vec<PathSegment>) -> Self {
        Self { segments }
    }
}

impl From<PathData> for Vec<PathSegment> {
    fn from(path: PathData) -> Self {
        path.segments
    }
}

impl AsRef<[PathSegment]> for PathData {
    fn as_ref(&self) -> &[PathSegment] {
        &self.segments
    }
}

impl FromIterator<PathSegment> for PathData {
    fn from_iter<I: IntoIterator<Item = PathSegment>>(iter: I) -> Self {
        Self {
            segments: iter.into_iter().collect(),
        }
    }
}

impl Extend<PathSegment> for PathData {
    fn extend<I: IntoIterator<Item = PathSegment>>(&mut self, iter: I) {
        self.segments.extend(iter);
    }
}

impl<'a> Extend<&'a PathSegment> for PathData {
    fn extend<I: IntoIterator<Item = &'a PathSegment>>(&mut self, iter: I) {
        self.segments.extend(iter.into_iter().copied());
    }
}

impl IntoIterator for PathData {
    type Item = PathSegment;
    type IntoIter = std::vec::IntoIter<PathSegment>;

    fn into_iter(self) -> Self::IntoIter {
        self.segments.into_iter()
    }
}

impl<'a> IntoIterator for &'a PathData {
    type Item = &'a PathSegment;
    type IntoIter = std::slice::Iter<'a, PathSegment>;

    fn into_iter(self) -> Self::IntoIter {
        self.segments.iter()
    }
}

/// Parses a whole `d` attribute value strictly.
///
/// This is the free-function spelling of [`PathData::parse`], for callers that
/// dispatch attribute values through a table of parser functions.
///
/// # Errors
///
/// Returns the [`ValueError`] describing the first grammar violation.
pub fn parse_path(input: &str) -> Result<PathData, ValueError> {
    PathData::parse(input)
}

/// Parses a whole `d` attribute value, keeping the renderable prefix.
///
/// This is the free-function spelling of [`PathData::parse_recover`].
#[must_use]
pub fn parse_path_recover(input: &str) -> (PathData, Option<ValueError>) {
    PathData::parse_recover(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Path data covering every command, both cases, and the awkward
    /// separator and number forms. Every entry is finite, so every entry can
    /// be round-tripped.
    const ROUND_TRIP_CORPUS: &[&str] = &[
        "",
        "M0 0",
        "M 15 20 Q 30 120 130 30 M 180 80 q -75 -100 -163 -60 z",
        "M372 130Q272 50 422 10zm70 0q50-150-80-90z",
        "M208 168Q258 268 308 168T258 118Q128 88 208 168z",
        "M240 296q25-100 47 0t47 0t47 0t47 0z",
        "M 25 70 A 40 40 0 1 0 25 69 Z",
        "m 150 100 a 50 40 0 1 0 25 -70 z",
        "M 100 100 C 100 20 200 20 200 100 S 300 180 300 100",
        "M 100 250 S 200 200 200 250 300 300 300 250",
        "M120,120 h25 a25,25 0 1,0 -25,25 z",
        "M10-20L.5.5",
        "M 240.5 56.25 H 270 V 86 H 300 V 116 Z",
        "m 240 190 h 30 v 30 h -30 v -30 z",
        "M0 0 10 10 20 20",
        "M0 0 Z Z",
        "M .5 -.5 L -.5 .5",
        "M1e-2 5e3 L 1.5E+2 0.001",
        "M0,0z m 1 1 L 2 2",
        "M0 0A-1 -2 400 0 1 2 0",
        "M0 0 H1 2 3 V4 5 6",
        "M0 0 T1 1 T2 2 t3 3",
    ];

    fn parse(input: &str) -> PathData {
        PathData::parse(input).unwrap_or_else(|error| panic!("{input:?}: {error}"))
    }

    #[test]
    fn canonical_output_round_trips_structurally() {
        for input in ROUND_TRIP_CORPUS {
            let path = parse(input);
            let rendered = path.to_string();
            assert_eq!(
                PathData::parse(&rendered),
                Ok(path.clone()),
                "canonical round trip of {input:?} via {rendered:?}"
            );
        }
    }

    #[test]
    fn compact_output_round_trips_structurally() {
        for input in ROUND_TRIP_CORPUS {
            let path = parse(input);
            let rendered = path.to_string_compact();
            let reparsed = PathData::parse(&rendered);
            assert_eq!(
                reparsed,
                Ok(path.clone()),
                "compact round trip of {input:?} via {rendered:?}"
            );
            assert_eq!(
                reparsed.map(|path| path.to_absolute()),
                Ok(path.to_absolute()),
                "compact geometry of {input:?} via {rendered:?}"
            );
        }
    }

    #[test]
    fn rounded_output_round_trips_structurally() {
        for input in ROUND_TRIP_CORPUS {
            let path = parse(input);
            for rendered in [
                path.fmt_with_precision(9),
                path.to_string_compact_with_precision(9),
            ] {
                assert_eq!(
                    PathData::parse(&rendered),
                    Ok(path.clone()),
                    "rounded round trip of {input:?} via {rendered:?}"
                );
            }
        }
    }

    #[test]
    fn compact_output_is_never_longer_than_canonical_output() {
        for input in ROUND_TRIP_CORPUS {
            let path = parse(input);
            assert!(
                path.to_string_compact().len() <= path.to_string().len(),
                "compacting {input:?} did not shrink it"
            );
        }
    }

    #[test]
    fn an_empty_path_is_valid_and_renders_as_nothing() {
        let path = parse("");
        assert!(path.is_empty());
        assert_eq!(path.len(), 0);
        assert_eq!(path.to_string(), "");
        assert_eq!(path.to_string_compact(), "");
        assert_eq!(PathData::new(), path);
        assert_eq!(PathData::default(), path);
        assert_eq!(parse("  \t\r\n "), path);
    }

    #[test]
    fn the_strict_and_recovering_entry_points_agree_on_valid_input() {
        let (path, error) = PathData::parse_recover("M0 0 L1 1");
        assert_eq!(error, None);
        assert_eq!(Ok(path), PathData::parse("M0 0 L1 1"));
    }

    #[test]
    fn the_strict_entry_point_rejects_what_the_recovering_one_truncates() {
        assert_eq!(
            PathData::parse("M 10 10 L"),
            Err(ValueError::UnexpectedEnd { offset: 9 })
        );
        let (path, error) = PathData::parse_recover("M 10 10 L");
        assert_eq!(path.to_string(), "M 10 10");
        assert_eq!(error, Some(ValueError::UnexpectedEnd { offset: 9 }));
    }

    #[test]
    fn free_functions_mirror_the_associated_ones() {
        assert_eq!(parse_path("M0 0"), PathData::parse("M0 0"));
        let (path, error) = parse_path_recover("M0 0 L");
        assert_eq!(path, parse("M0 0"));
        assert!(error.is_some());
    }

    #[test]
    fn paths_parse_through_the_from_str_trait() {
        let path: PathData = "M0 0 L1 1".parse().expect("valid path data");
        assert_eq!(path.len(), 2);
        assert!("L0 0".parse::<PathData>().is_err());
    }

    #[test]
    fn segments_can_be_pushed_extended_and_collected() {
        let mut path = PathData::with_capacity(4);
        path.push(PathSegment::MoveTo {
            abs: true,
            x: 0.0,
            y: 0.0,
        });
        path.extend([PathSegment::LineTo {
            abs: true,
            x: 1.0,
            y: 1.0,
        }]);
        let tail = [PathSegment::Close];
        path.extend(tail.iter());
        assert_eq!(path.to_string(), "M 0 0 L 1 1 Z");
        assert_eq!(path.pop(), Some(PathSegment::Close));

        let collected: PathData = path.iter().copied().collect();
        assert_eq!(collected, path);
        let from_vec = PathData::from(path.clone().into_segments());
        assert_eq!(from_vec, path);
        assert_eq!(PathData::from_segments(Vec::from(path.clone())), path);
        let borrowed: &[PathSegment] = path.as_ref();
        assert_eq!(borrowed, path.segments());

        path.clear();
        assert!(path.is_empty());
    }

    #[test]
    fn paths_iterate_by_value_and_by_reference() {
        let path = parse("M0 0 L1 1 Z");
        let by_reference: Vec<char> = (&path).into_iter().map(PathSegment::command).collect();
        assert_eq!(by_reference, vec!['M', 'L', 'Z']);
        let by_value: Vec<PathSegment> = path.clone().into_iter().collect();
        assert_eq!(by_value, path.segments());
        assert_eq!(path.iter().count(), 3);
    }

    #[test]
    fn segments_can_be_edited_in_place() {
        let mut path = parse("M0 0 L1 1");
        for segment in path.segments_mut() {
            *segment = segment.relative_from((0.0, 0.0));
        }
        assert_eq!(path.to_string(), "m 0 0 l 1 1");
    }

    #[test]
    fn saturated_coordinates_are_reported_as_non_finite() {
        let path = parse("M1e999 0");
        assert!(!path.is_finite());
        assert!(parse("M1e300 0").is_finite());
        assert!(PathData::new().is_finite());
    }

    #[test]
    fn documented_examples_hold() {
        let path = parse("M0 0 10 10 z");
        assert_eq!(path.to_string_compact(), "M0 0 10 10Z");
        assert_eq!(path.to_string(), "M 0 0 L 10 10 Z");
        let path = parse("M 0 0 L 1 1 L -2.5 0.5 a 1 1 0 0 1 1 0");
        assert_eq!(path.to_string_compact(), "M0 0 1 1-2.5.5a1 1 0 011 0");
        let path = parse("m 10 20 l 5 5 h 5 z");
        assert_eq!(path.to_absolute().to_string(), "M 10 20 L 15 25 H 20 Z");
        let path = parse("M 0.3333333333 2.0000001");
        assert_eq!(path.fmt_with_precision(3), "M 0.333 2");
    }
}
