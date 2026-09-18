//! `points` attribute values for `polyline` and `polygon`.
//!
//! The `points` attribute is a list of `(x, y)` coordinate pairs. Per
//! `SVG11/shapes.html` §9 (carried over unchanged into SVG2, which does not
//! redefine `points` itself), a value with an odd number of coordinates —
//! or any other grammar violation — is in error "with the same user agent
//! behavior as occurs with an incorrectly specified `path` element": render
//! the complete pairs already scanned and drop everything from the error
//! onward, rather than rejecting the whole attribute. [`parse_points`]
//! implements exactly that recovery rule; [`parse_points_strict`] is the
//! all-or-nothing counterpart for callers that want a hard failure instead.
//! [`write_points`] serializes a point list back to the attribute syntax.
//!
//! # Separators
//!
//! Between two coordinates — whether the two halves of one pair, or the
//! last coordinate of one pair and the first of the next — a `comma-wsp`
//! separator is generally required, but it may be omitted immediately
//! before a coordinate that begins with `+`, `-`, or `.`, since that
//! leading character already marks where the next number starts (the same
//! rule `crate::values::lex::parse_number_list` applies to plain number
//! lists; the SVG grammar's own `coordinate negative-coordinate`
//! alternative is the special case of it where the glued-on coordinate
//! happens to be negative). A **trailing** separator with nothing valid
//! after it, however, is an error here — unlike
//! `crate::values::length::parse_length_list`'s deliberate leniency for
//! plain lists, `points` has its own explicit spec-mandated recovery rule
//! to fall back on instead of silently accepting a dangling comma.
//!
//! # Examples
//!
//! ```
//! use oxixml_svg::values::points::{parse_points, parse_points_strict, write_points};
//!
//! let (points, error) = parse_points("0,0 10,0 10,10 0,10");
//! assert!(error.is_none());
//! assert_eq!(points, vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]);
//! assert_eq!(write_points(&points), "0,0 10,0 10,10 0,10");
//!
//! // An odd trailing coordinate keeps the valid pairs and reports the error.
//! let (points, error) = parse_points("0,0 1,1 2");
//! assert_eq!(points, vec![[0.0, 0.0], [1.0, 1.0]]);
//! assert!(error.is_some());
//!
//! assert!(parse_points_strict("0,0 1,1 2").is_err());
//! ```

use crate::error::ValueError;
use crate::values::lex::Cursor;

/// Parses a `points` attribute value, applying SVG's own error-recovery
/// rule: the complete leading run of coordinate pairs is always returned,
/// and if scanning stopped because of an odd trailing coordinate or any
/// other malformed content, the error describing why is returned alongside
/// it rather than discarding the valid prefix.
///
/// An empty (or white-space-only) value is not an error; it yields an empty
/// list and no error, disabling rendering of the shape it belongs to
/// without saying anything is wrong with the attribute itself.
///
/// See the [module documentation](self) for the separator rules, including
/// why a trailing separator with nothing after it *is* treated as an
/// error here, unlike the general `list-of-Ts` parsers in
/// [`crate::values::length`].
///
/// Use [`parse_points_strict`] for a `Result`-returning all-or-nothing
/// variant, and [`write_points`] to serialize the result back.
#[must_use]
pub fn parse_points(input: &str) -> (Vec<[f64; 2]>, Option<ValueError>) {
    let mut cursor = Cursor::new(input);
    let mut points = Vec::new();
    cursor.skip_wsp();
    loop {
        if cursor.is_at_end() {
            return (points, None);
        }
        match scan_pair(&mut cursor) {
            Ok(pair) => points.push(pair),
            Err(error) => return (points, Some(error)),
        }
        let (separated, had_comma) = skip_separator(&mut cursor);
        if cursor.is_at_end() {
            return if had_comma {
                // A comma promises another pair follows; running out of
                // input right after it is exactly the "value ended before
                // it was complete" situation, not unremarkable trailing
                // white space.
                (
                    points,
                    Some(ValueError::UnexpectedEnd {
                        offset: cursor.pos(),
                    }),
                )
            } else {
                (points, None)
            };
        }
        if !separated && !matches!(cursor.peek(), Some('+' | '-' | '.' | '0'..='9')) {
            return (
                points,
                Some(ValueError::TrailingData {
                    offset: cursor.pos(),
                }),
            );
        }
    }
}

/// Parses a `points` attribute value, failing on any grammar violation
/// instead of recovering a partial list.
///
/// Equivalent to [`parse_points`] with its `Option<ValueError>` and
/// `Vec<[f64; 2]>` collapsed into a single `Result`: `Ok` only when
/// [`parse_points`] reports no error at all.
///
/// # Errors
///
/// Returns whatever error [`parse_points`] would have reported alongside
/// its (possibly empty, possibly partial) point list.
pub fn parse_points_strict(input: &str) -> Result<Vec<[f64; 2]>, ValueError> {
    match parse_points(input) {
        (points, None) => Ok(points),
        (_, Some(error)) => Err(error),
    }
}

/// Serializes a point list as a `points` attribute value: `x,y` within a
/// pair, a space between pairs (`"x1,y1 x2,y2"`), and the empty string for
/// an empty slice.
#[must_use]
pub fn write_points(points: &[[f64; 2]]) -> String {
    let mut out = String::new();
    for (index, [x, y]) in points.iter().enumerate() {
        if index > 0 {
            out.push(' ');
        }
        out.push_str(&x.to_string());
        out.push(',');
        out.push_str(&y.to_string());
    }
    out
}

/// Scans one `coordinate-pair`: two numbers, the second either following a
/// `comma-wsp` separator or glued on directly when its own leading `+`,
/// `-`, or `.` already marks where it begins (see the
/// [module documentation](self)). Any failure to read either number —
/// including running out of input for the second — propagates as-is, with
/// [`Cursor::number`]'s own offset.
fn scan_pair(cursor: &mut Cursor<'_>) -> Result<[f64; 2], ValueError> {
    let x = cursor.number()?;
    cursor.skip_comma_wsp();
    let y = cursor.number()?;
    Ok([x, y])
}

/// Skips one `comma-wsp` separator like [`Cursor::skip_comma_wsp`], but
/// additionally reports whether the separator included a literal comma (as
/// opposed to bare white space). Returns `(separator_consumed,
/// comma_seen)`, so callers can tell a dangling comma promising more
/// content from unremarkable trailing white space.
fn skip_separator(cursor: &mut Cursor<'_>) -> (bool, bool) {
    let start = cursor.pos();
    cursor.skip_wsp();
    let had_comma = cursor.eat(',');
    if had_comma {
        cursor.skip_wsp();
    }
    (cursor.pos() != start, had_comma)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_simple_polygon() {
        let (points, error) = parse_points("0,0 10,0 10,10 0,10");
        assert!(error.is_none());
        assert_eq!(
            points,
            vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
        );
    }

    #[test]
    fn accepts_wsp_only_separators_and_surrounding_white_space() {
        let (points, error) = parse_points("  0 0   1 1  ");
        assert!(error.is_none());
        assert_eq!(points, vec![[0.0, 0.0], [1.0, 1.0]]);
    }

    #[test]
    fn empty_value_is_a_valid_empty_list() {
        assert_eq!(parse_points(""), (Vec::new(), None));
        assert_eq!(parse_points("   "), (Vec::new(), None));
    }

    #[test]
    fn odd_trailing_coordinate_keeps_the_valid_pair_prefix() {
        let (points, error) = parse_points("0,0 1,1 2");
        assert_eq!(points, vec![[0.0, 0.0], [1.0, 1.0]]);
        assert_eq!(error, Some(ValueError::UnexpectedEnd { offset: 9 }));
    }

    #[test]
    fn odd_leading_coordinate_yields_an_empty_prefix_and_an_error() {
        let (points, error) = parse_points("5");
        assert_eq!(points, Vec::<[f64; 2]>::new());
        assert_eq!(error, Some(ValueError::UnexpectedEnd { offset: 1 }));
    }

    #[test]
    fn separator_free_sign_and_point_glued_coordinates_recover_the_valid_prefix() {
        // "1-2.3.4" tokenizes as three numbers (1, -2.3, 0.4) with no
        // separators at all between them, via the same maximal-munch
        // number scanner every list parser in this crate shares: the pair
        // (1, -2.3) is valid, and the lone trailing 0.4 is the odd
        // coordinate the recovery rule reports.
        let (points, error) = parse_points("1-2.3.4");
        assert_eq!(points, vec![[1.0, -2.3]]);
        assert_eq!(error, Some(ValueError::UnexpectedEnd { offset: 7 }));
    }

    #[test]
    fn negative_coordinate_needs_no_separator_from_its_pair_partner() {
        let (points, error) = parse_points("0-1 2-3");
        assert!(error.is_none());
        assert_eq!(points, vec![[0.0, -1.0], [2.0, -3.0]]);
    }

    #[test]
    fn trailing_comma_is_an_error_per_grammar() {
        // `list-of-points ::= wsp* coordinate-pairs? wsp*` has no
        // trailing-comma alternative: a comma always promises another
        // `coordinate-pair`. Unlike the general number/length list
        // parsers, `points` does not extend its leniency to this case.
        let (points, error) = parse_points("0,0,");
        assert_eq!(points, vec![[0.0, 0.0]]);
        assert_eq!(error, Some(ValueError::UnexpectedEnd { offset: 4 }));
    }

    #[test]
    fn trailing_comma_followed_by_white_space_is_still_an_error() {
        let (points, error) = parse_points("0,0, ");
        assert_eq!(points, vec![[0.0, 0.0]]);
        assert_eq!(error, Some(ValueError::UnexpectedEnd { offset: 5 }));
    }

    #[test]
    fn plain_trailing_white_space_with_no_comma_is_not_an_error() {
        let (points, error) = parse_points("0,0 ");
        assert_eq!(points, vec![[0.0, 0.0]]);
        assert!(error.is_none());
    }

    #[test]
    fn garbage_after_a_separator_is_reported_at_its_own_offset() {
        let (points, error) = parse_points("0,0 x");
        assert_eq!(points, vec![[0.0, 0.0]]);
        assert_eq!(error, Some(ValueError::InvalidNumber { offset: 4 }));
    }

    #[test]
    fn garbage_at_the_very_start_yields_an_empty_prefix() {
        let (points, error) = parse_points("x,y");
        assert_eq!(points, Vec::<[f64; 2]>::new());
        assert_eq!(error, Some(ValueError::InvalidNumber { offset: 0 }));
    }

    #[test]
    fn garbage_between_the_two_coordinates_of_a_pair_is_reported() {
        let (points, error) = parse_points("0,0 1 y");
        assert_eq!(points, vec![[0.0, 0.0]]);
        assert_eq!(error, Some(ValueError::InvalidNumber { offset: 6 }));
    }

    #[test]
    fn parse_points_strict_succeeds_when_there_is_no_error() {
        assert_eq!(
            parse_points_strict("0,0 1,1").expect("well-formed points"),
            vec![[0.0, 0.0], [1.0, 1.0]]
        );
        assert_eq!(
            parse_points_strict("").expect("empty is valid"),
            Vec::<[f64; 2]>::new()
        );
    }

    #[test]
    fn parse_points_strict_fails_on_any_recoverable_error_too() {
        assert_eq!(
            parse_points_strict("0,0 1,1 2"),
            Err(ValueError::UnexpectedEnd { offset: 9 })
        );
        assert_eq!(
            parse_points_strict("0,0,"),
            Err(ValueError::UnexpectedEnd { offset: 4 })
        );
    }

    #[test]
    fn write_points_formats_comma_within_pair_space_between_pairs() {
        assert_eq!(
            write_points(&[[0.0, 0.0], [1.0, 1.0], [-2.5, 3.0]]),
            "0,0 1,1 -2.5,3"
        );
    }

    #[test]
    fn write_points_of_an_empty_slice_is_an_empty_string() {
        assert_eq!(write_points(&[]), "");
    }

    #[test]
    fn write_points_of_a_single_pair_has_no_space() {
        assert_eq!(write_points(&[[1.0, 2.0]]), "1,2");
    }

    #[test]
    fn round_trips_through_write_then_parse_strict() {
        let original = vec![[0.0, 0.0], [10.5, -3.25], [100.0, 0.0]];
        let text = write_points(&original);
        assert_eq!(
            parse_points_strict(&text).expect("write_points output re-parses"),
            original
        );
    }

    #[test]
    fn parses_a_realistic_polygon_points_value() {
        // A hexagon, in the shape typical hand-authored SVG uses.
        let (points, error) =
            parse_points("850,75 958,137.5 958,262.5 850,325 742,262.5 742,137.5");
        assert!(error.is_none());
        assert_eq!(
            points,
            vec![
                [850.0, 75.0],
                [958.0, 137.5],
                [958.0, 262.5],
                [850.0, 325.0],
                [742.0, 262.5],
                [742.0, 137.5],
            ]
        );
    }

    #[test]
    fn mixed_comma_and_white_space_separators_all_work_together() {
        let (points, error) = parse_points("0,0, 1 1 ,2,2");
        assert!(error.is_none());
        assert_eq!(points, vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]);
    }
}
