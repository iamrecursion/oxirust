//! Number formatting for serialized SVG attribute values.
//!
//! Every builder-facing setter that emits a bare number — as opposed to a
//! [`crate::values`] type with its own [`Display`](core::fmt::Display), such
//! as [`crate::values::length::Length`] or [`crate::values::color::Color`] —
//! goes through this module, so that every number written anywhere in a
//! document built with [`crate::writer::builder::SvgBuilder`] follows the
//! same rules:
//!
//! * an integral value is written without a trailing decimal point (`42`,
//!   not `42.0`);
//! * a non-integral value is written in the shortest form that reads back to
//!   the same `f64` ([`f64`]'s own [`Display`](core::fmt::Display) already
//!   guarantees this);
//! * `-0` never appears — a value that is (or rounds to) negative zero is
//!   normalized to plain `0`;
//! * scientific notation is never emitted: [`fmt_number`] delegates to
//!   `f64`'s `{}` `Display`, which this module's own tests confirm writes the
//!   full decimal expansion (never `1e20`-style exponent form) for
//!   magnitudes up to `f64::MAX` and down to `f64::MIN_POSITIVE`;
//! * rounding to a fixed number of decimal places ([`fmt_number_precision`])
//!   is round-half-to-even, because that is what `format!("{:.N}")`
//!   implements for `f64`, and the result has its trailing zeros — and a
//!   bare trailing point — trimmed away afterwards (`1.500000` -> `1.5`,
//!   `2.000000` -> `2`), mirroring the trimming rule
//!   [`crate::values::path`]'s own writer uses for the same reason.
//!
//! # `NaN` and infinity
//!
//! Nothing in this module rejects a non-finite value: [`fmt_number`] and
//! [`fmt_number_precision`] format it using `f64`'s own `Display` (`NaN`,
//! `inf`, `-inf`), exactly as they format any other value — deciding
//! *whether* a non-finite number belongs in a document at all is a policy
//! question for the caller, not something a number formatter should silently
//! paper over by substituting `0`. `is_attr_safe` (crate-private) is the
//! check every typed numeric setter on
//! [`SvgBuilder`](crate::writer::builder::SvgBuilder) runs first, since
//! `NaN`/`inf`/`-inf` are not valid SVG `<number>` tokens and a value
//! formatted that way will not survive being re-parsed.

use std::fmt::Write as _;

/// Formats `value` in the minimal spec-valid form: an integer with no
/// trailing decimal point, the shortest round-tripping decimal form
/// otherwise, and never a bare `-0`.
///
/// See the [module documentation](self) for the exact rules and the
/// deliberate non-handling of `NaN`/infinite values.
#[must_use]
pub fn fmt_number(value: f64) -> String {
    let mut out = String::new();
    write_number(&mut out, value);
    out
}

/// Appends [`fmt_number`]'s rendering of `value` to `out`, without an
/// intermediate allocation.
pub fn write_number(out: &mut String, value: f64) {
    let value = normalize_negative_zero(value);
    // `f64`'s `Display` is already the shortest round-tripping decimal form,
    // with no trailing `.0` for integers and no scientific notation at any
    // finite magnitude (see the module documentation).
    let _ = write!(out, "{value}");
}

/// Formats `value` rounded to exactly `precision` decimal places, then trims
/// trailing zeros and a bare trailing decimal point (`1.500000` -> `1.5`,
/// `2.000000` -> `2`).
///
/// Rounding is round-half-to-even, because that is what `format!("{:.N}")`
/// implements for `f64`; the result is therefore stable and reproducible
/// across runs, which matters for byte-identical re-serialization. A
/// non-finite `value` is formatted with [`fmt_number`] instead: rounding
/// `NaN`/`±inf` to a fixed number of places is not a meaningful operation,
/// and falling back keeps every caller from having to special-case it.
#[must_use]
pub fn fmt_number_precision(value: f64, precision: usize) -> String {
    let mut out = String::new();
    write_number_precision(&mut out, value, precision);
    out
}

/// Appends [`fmt_number_precision`]'s rendering of `value` to `out`, without
/// an intermediate allocation.
pub fn write_number_precision(out: &mut String, value: f64, precision: usize) {
    if !value.is_finite() {
        write_number(out, value);
        return;
    }
    let start = out.len();
    let _ = write!(out, "{value:.precision$}");
    trim_trailing_zeros(out, start);
    normalize_negative_zero_tail(out, start);
}

/// Formats `value` with [`fmt_number`] when `precision` is `None`, or
/// [`fmt_number_precision`] otherwise.
///
/// This is the single entry point every precision-aware setter on
/// [`SvgBuilder`](crate::writer::builder::SvgBuilder) — `attr_number`, path
/// data, and point lists — calls, so that a builder's `set_precision`
/// affects every one of them identically.
#[must_use]
pub fn fmt_number_with(value: f64, precision: Option<usize>) -> String {
    let mut out = String::new();
    write_number_with(&mut out, value, precision);
    out
}

/// Appends [`fmt_number_with`]'s rendering of `value` to `out`, without an
/// intermediate allocation.
pub fn write_number_with(out: &mut String, value: f64, precision: Option<usize>) {
    match precision {
        Some(precision) => write_number_precision(out, value, precision),
        None => write_number(out, value),
    }
}

/// Appends `sep`-separated renderings of `values` to `out` (via
/// [`fmt_number`]), with no leading or trailing separator. An empty slice
/// appends nothing.
pub fn push_number_list(out: &mut String, values: &[f64], sep: &str) {
    for (index, &value) in values.iter().enumerate() {
        if index > 0 {
            out.push_str(sep);
        }
        write_number(out, value);
    }
}

/// As [`push_number_list`], but every element is formatted with
/// [`fmt_number_with`], honoring a builder's precision setting.
pub fn push_number_list_with(
    out: &mut String,
    values: &[f64],
    sep: &str,
    precision: Option<usize>,
) {
    for (index, &value) in values.iter().enumerate() {
        if index > 0 {
            out.push_str(sep);
        }
        write_number_with(out, value, precision);
    }
}

/// Formats a list of `(x, y)` pairs the way the `points` attribute
/// microsyntax ([`crate::values::points`]) writes them: `x,y` within a pair,
/// a single space between pairs, honoring `precision` the same way
/// [`fmt_number_with`] does.
///
/// Unlike [`crate::values::points::write_points`], which always uses the
/// shortest round-tripping form, this lets
/// [`SvgBuilder::attr_points`](crate::writer::builder::SvgBuilder::attr_points)
/// respect a builder's `set_precision`.
#[must_use]
pub fn format_points(points: &[[f64; 2]], precision: Option<usize>) -> String {
    let mut out = String::new();
    for (index, [x, y]) in points.iter().enumerate() {
        if index > 0 {
            out.push(' ');
        }
        write_number_with(&mut out, *x, precision);
        out.push(',');
        write_number_with(&mut out, *y, precision);
    }
    out
}

/// True when `value` is safe to write into an SVG attribute: finite, i.e.
/// neither `NaN` nor infinite.
///
/// `NaN`/`inf`/`-inf` are not valid SVG `<number>` tokens; a document that
/// contains one will not re-parse. Every typed numeric setter on
/// [`SvgBuilder`](crate::writer::builder::SvgBuilder) checks this before
/// writing an attribute and records a
/// [`BuildFlaw`](crate::writer::builder::BuildFlaw) instead when it fails,
/// rather than serializing a non-conformant value.
#[must_use]
pub(crate) fn is_attr_safe(value: f64) -> bool {
    value.is_finite()
}

/// Collapses `-0.0` to plain `0.0`; `+0.0 == -0.0` under IEEE 754 equality,
/// so this comparison alone is enough to catch both.
fn normalize_negative_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

/// Removes the trailing zeros — and then a bare trailing point — of the
/// fixed-precision rendering appended to `out` at or after `start`.
fn trim_trailing_zeros(out: &mut String, start: usize) {
    if !out[start..].contains('.') {
        return;
    }
    while out.ends_with('0') {
        out.pop();
    }
    if out.ends_with('.') {
        out.pop();
    }
}

/// Collapses a trimmed rendering of exactly `-0` (reachable when a small
/// negative magnitude rounds all the way down to zero at the requested
/// precision, e.g. `-0.0000001` at 3 places) to plain `0`.
fn normalize_negative_zero_tail(out: &mut String, start: usize) {
    if &out[start..] == "-0" {
        out.truncate(start);
        out.push('0');
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integers_have_no_trailing_decimal_point() {
        assert_eq!(fmt_number(42.0), "42");
        assert_eq!(fmt_number(0.0), "0");
        assert_eq!(fmt_number(-7.0), "-7");
    }

    #[test]
    fn non_integers_use_the_shortest_round_tripping_form() {
        assert_eq!(fmt_number(1.5), "1.5");
        let text = fmt_number(0.1 + 0.2);
        assert_eq!(text.parse::<f64>(), Ok(0.1 + 0.2));
    }

    #[test]
    fn negative_zero_normalizes_to_plain_zero() {
        assert_eq!(fmt_number(-0.0), "0");
        assert_eq!(fmt_number(0.0), fmt_number(-0.0));
    }

    #[test]
    fn no_scientific_notation_from_the_smallest_to_the_largest_finite_magnitude() {
        // Empirically confirmed (see the module documentation): Rust's f64
        // `Display` never switches to exponent form, from the smallest
        // positive normal value up to `f64::MAX`.
        assert_eq!(
            fmt_number(1.0e20),
            "100000000000000000000",
            "no exponent at 1e20"
        );
        assert!(!fmt_number(f64::MAX).contains('e'));
        assert!(!fmt_number(f64::MIN_POSITIVE).contains('e'));
        assert_eq!(fmt_number(f64::MAX).parse::<f64>(), Ok(f64::MAX));
    }

    #[test]
    fn non_finite_values_format_via_display_unmodified() {
        assert_eq!(fmt_number(f64::NAN), "NaN");
        assert_eq!(fmt_number(f64::INFINITY), "inf");
        assert_eq!(fmt_number(f64::NEG_INFINITY), "-inf");
        assert!(!is_attr_safe(f64::NAN));
        assert!(!is_attr_safe(f64::INFINITY));
        assert!(!is_attr_safe(f64::NEG_INFINITY));
        assert!(is_attr_safe(0.0));
        assert!(is_attr_safe(-1.5));
    }

    #[test]
    fn precision_rounds_then_trims_trailing_zeros_and_point() {
        assert_eq!(fmt_number_precision(1.5, 6), "1.5");
        assert_eq!(fmt_number_precision(2.0, 6), "2");
        assert_eq!(fmt_number_precision(1.0 / 3.0, 3), "0.333");
        assert_eq!(fmt_number_precision(0.0, 3), "0");
    }

    #[test]
    fn precision_rounding_is_half_to_even() {
        // `format!("{:.0}")` on f64 rounds half-to-even: 0.5 and 2.5 are
        // both exactly representable in binary, so these are genuine ties.
        assert_eq!(fmt_number_precision(0.5, 0), "0");
        assert_eq!(fmt_number_precision(2.5, 0), "2");
        assert_eq!(fmt_number_precision(1.5, 0), "2");
    }

    #[test]
    fn precision_never_emits_a_bare_negative_zero() {
        assert_eq!(fmt_number_precision(-0.0000001, 3), "0");
        assert_eq!(fmt_number_precision(-0.0, 3), "0");
    }

    #[test]
    fn precision_of_a_non_finite_value_falls_back_to_the_plain_form() {
        assert_eq!(fmt_number_precision(f64::NAN, 2), "NaN");
        assert_eq!(fmt_number_precision(f64::INFINITY, 2), "inf");
    }

    #[test]
    fn with_dispatches_on_the_presence_of_a_precision() {
        assert_eq!(fmt_number_with(1.0 / 3.0, None), fmt_number(1.0 / 3.0));
        assert_eq!(
            fmt_number_with(1.0 / 3.0, Some(2)),
            fmt_number_precision(1.0 / 3.0, 2)
        );
    }

    #[test]
    fn write_variants_append_rather_than_replace() {
        let mut out = String::from("prefix:");
        write_number(&mut out, 1.0);
        assert_eq!(out, "prefix:1");

        let mut out = String::from("prefix:");
        write_number_precision(&mut out, 1.5, 0);
        assert_eq!(out, "prefix:2");

        let mut out = String::from("prefix:");
        write_number_with(&mut out, 1.5, Some(0));
        assert_eq!(out, "prefix:2");
    }

    #[test]
    fn push_number_list_joins_with_the_given_separator() {
        let mut out = String::new();
        push_number_list(&mut out, &[1.0, 2.5, -3.0], ",");
        assert_eq!(out, "1,2.5,-3");
    }

    #[test]
    fn push_number_list_of_an_empty_slice_appends_nothing() {
        let mut out = String::from("x");
        push_number_list(&mut out, &[], ",");
        assert_eq!(out, "x");
    }

    #[test]
    fn push_number_list_with_honors_precision() {
        let mut out = String::new();
        push_number_list_with(&mut out, &[1.0 / 3.0, 2.0], " ", Some(2));
        assert_eq!(out, "0.33 2");
    }

    #[test]
    fn format_points_uses_comma_within_pair_space_between_pairs() {
        assert_eq!(
            format_points(&[[0.0, 0.0], [1.0, 1.0], [-2.5, 3.0]], None),
            "0,0 1,1 -2.5,3"
        );
    }

    #[test]
    fn format_points_of_an_empty_slice_is_empty() {
        assert_eq!(format_points(&[], None), "");
    }

    #[test]
    fn format_points_honors_precision() {
        assert_eq!(
            format_points(&[[1.0 / 3.0, 2.0 / 3.0]], Some(2)),
            "0.33,0.67"
        );
    }
}
