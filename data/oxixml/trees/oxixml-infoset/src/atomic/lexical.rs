//! Extra lexical-space restrictions the date, time and duration types carry beyond what the
//! value-space parsers in `oxixml-xsd` enforce.
//!
//! `oxixml-xsd` maps a lexical form to a *value*, and is deliberately generous where two readings
//! agree on that value: it normalises `24:00:00` to the next midnight, accepts a leading decimal
//! point in a duration's seconds field, and reads year `0000` as the astronomical year zero. XML
//! Schema's lexical spaces are narrower than that, and XPath's cast operator is defined against
//! the *lexical* space, so a form outside it must raise `FORG0001` rather than quietly denote the
//! value a wider reading would give it.
//!
//! Everything here is a pure syntax check applied *before* the value parser runs, so a form that
//! is rejected never reaches — and never overflows — the value space. That ordering matters:
//! `"99999999999999999999999999999-XX" cast as xs:gYearMonth` is a lexical error (`FORG0001`),
//! not the overflow condition `FODT0001`, because `XX` is not a month.
//!
//! # Which schema version
//!
//! The restrictions below are those of **XML Schema 1.0**, the version this processor reports.
//! Where 1.0 and 1.1 differ the QT3 suite splits the case in two on an `xsd-version` dependency —
//! `cbcl-cast-gYear-002` (1.0, year `0000` is `FORG0001`) against `cbcl-cast-gYear-003a` (1.1,
//! the same form is legal) — so the 1.1 readings are out of profile rather than wrong.
//!
//! The one rule taken from neither version verbatim is the duration seconds field. XSD 1.0
//! §3.2.6 requires a digit on both sides of the decimal point; XSD 1.1's `duSecondFrag` appears
//! to allow `.5S`, which the QT3 suite records as a spec error (`CastAs688`: "XSD 1.1 apparently
//! doesn't, but this is believed to be an error in the spec"). The 1.0 reading is enforced.

use super::types::AtomicType;

/// Checks the restrictions of this module for `atomic_type`, whose primitive base decides which
/// grammar applies. Returns a human-readable reason when the form is outside the lexical space.
///
/// `lexical` must already have had its white-space facet applied.
pub(super) fn extra_lexical_restriction(
    atomic_type: AtomicType,
    lexical: &str,
) -> Result<(), &'static str> {
    match atomic_type.primitive() {
        AtomicType::Duration => check_duration(lexical),
        AtomicType::DateTime => {
            let (date, time) = lexical
                .split_once('T')
                .ok_or("a `T` separator is required")?;
            check_year(date)?;
            check_time(time)
        }
        AtomicType::Date | AtomicType::GYear => check_year(lexical),
        AtomicType::GYearMonth => {
            check_year(lexical)?;
            check_gyearmonth(lexical)
        }
        AtomicType::Time => check_time(lexical),
        AtomicType::GMonth => check_gmonth(lexical),
        _ => Ok(()),
    }
}

/// Splits a trailing timezone off a lexical form, returning the body and the timezone.
fn split_timezone(lexical: &str) -> (&str, &str) {
    if let Some(body) = lexical.strip_suffix('Z') {
        return (body, "Z");
    }
    // `+hh:mm` and `-hh:mm` are six characters; a leading `-` on the *year* is never at this
    // offset, because every lexical form that can carry one has at least four digits after it.
    let bytes = lexical.as_bytes();
    if lexical.len() >= 6 {
        let boundary = lexical.len() - 6;
        if matches!(bytes[boundary], b'+' | b'-') && bytes[boundary + 3] == b':' {
            return lexical.split_at(boundary);
        }
    }
    (lexical, "")
}

/// XSD 1.0 §3.2.7: the year `0000` is not in the lexical space of any date-bearing type — the
/// proleptic Gregorian calendar of XML Schema 1.0 has no year zero.
fn check_year(lexical: &str) -> Result<(), &'static str> {
    let (body, _) = split_timezone(lexical);
    let digits = body.strip_prefix('-').unwrap_or(body);
    let year: &str = digits.split('-').next().unwrap_or(digits);
    if year.len() >= 4 && year.bytes().all(|byte| byte == b'0') {
        return Err("the year 0000 is not in the XML Schema 1.0 lexical space");
    }
    Ok(())
}

/// `xs:gYearMonth` is `'-'? yyyy '-' mm` with a two-digit month. Checking the month here, before
/// the value parser reads the year, keeps a form whose *month* is malformed a lexical error even
/// when its year would also overflow (QT3 `cbcl-cast-gYearMonth-002`).
fn check_gyearmonth(lexical: &str) -> Result<(), &'static str> {
    let (body, _) = split_timezone(lexical);
    let digits = body.strip_prefix('-').unwrap_or(body);
    let month = match digits.split_once('-') {
        Some((_, month)) => month,
        None => return Err("xs:gYearMonth is written `yyyy-mm`"),
    };
    if month.len() == 2 && month.bytes().all(|byte| byte.is_ascii_digit()) {
        Ok(())
    } else {
        Err("the month of an xs:gYearMonth is two digits")
    }
}

/// XSD 1.0 §3.2.8: hour `24` stands for the end of the day, and is legal only in the exact form
/// `24:00:00`, optionally followed by a zero fraction.
fn check_time(lexical: &str) -> Result<(), &'static str> {
    let (body, _) = split_timezone(lexical);
    let Some(rest) = body.strip_prefix("24:") else {
        return Ok(());
    };
    let ok = match rest.split_once('.') {
        Some((seconds, fraction)) => {
            seconds == "00:00" && !fraction.is_empty() && fraction.bytes().all(|byte| byte == b'0')
        }
        None => rest == "00:00",
    };
    if ok {
        Ok(())
    } else {
        Err("hour 24 is legal only in the form `24:00:00`")
    }
}

/// XSD 1.0's second edition erratum settled `xs:gMonth` on `--MM`; the first edition's `--MM--`
/// is not in the lexical space (QT3 `K2-SeqExprCast-506`).
fn check_gmonth(lexical: &str) -> Result<(), &'static str> {
    let (body, _) = split_timezone(lexical);
    if body.len() == 4 {
        Ok(())
    } else {
        Err("xs:gMonth is written `--MM`, without the trailing `--` of XSD 1.0's first edition")
    }
}

/// The duration lexical space of XSD 1.0 §3.2.6:
///
/// ```text
/// '-'? 'P' (nnn 'Y')? (nnn 'M')? (nnn 'D')? ('T' (nnn 'H')? (nnn 'M')? (nnn('.'nnn)? 'S')?)?
/// ```
///
/// with at least one component overall, and — this is what the value parser does not check — at
/// least one component after a `T`, and a digit on each side of the seconds' decimal point.
fn check_duration(lexical: &str) -> Result<(), &'static str> {
    let body = lexical.strip_prefix('-').unwrap_or(lexical);
    let Some(mut rest) = body.strip_prefix('P') else {
        return Err("a duration begins with `P`");
    };
    // The date part, then an optional time part. `designators` is scanned in order, so a
    // component out of sequence or repeated simply fails to match and is left for the value
    // parser to name.
    let mut components = 0_usize;
    for designator in ['Y', 'M', 'D'] {
        if let Some(after) = take_component(rest, designator, false) {
            rest = after;
            components += 1;
        }
    }
    if let Some(after) = rest.strip_prefix('T') {
        rest = after;
        let mut time_components = 0_usize;
        for (designator, fractional) in [('H', false), ('M', false), ('S', true)] {
            if let Some(after) = take_component(rest, designator, fractional) {
                rest = after;
                time_components += 1;
            }
        }
        if time_components == 0 && rest.is_empty() {
            return Err("a `T` separator must be followed by at least one time component");
        }
        components += time_components;
    }
    if components == 0 {
        return Err("a duration needs at least one component");
    }
    if rest.is_empty() {
        Ok(())
    } else {
        Err(
            "a duration component must be digits followed by its designator, with a digit on \
             each side of the seconds' decimal point",
        )
    }
}

/// Matches `digits designator` at the front of `rest`, allowing `digits '.' digits` when
/// `fractional`. Returns what is left, or `None` when the component is not there.
fn take_component(rest: &str, designator: char, fractional: bool) -> Option<&str> {
    let mut chars = rest.char_indices();
    let mut digits = 0_usize;
    let mut seen_point = false;
    let mut fraction_digits = 0_usize;
    loop {
        let (offset, character) = chars.next()?;
        if character.is_ascii_digit() {
            if seen_point {
                fraction_digits += 1;
            } else {
                digits += 1;
            }
        } else if character == '.' && fractional && !seen_point && digits > 0 {
            seen_point = true;
        } else if character == designator && digits > 0 && (!seen_point || fraction_digits > 0) {
            return Some(&rest[offset + character.len_utf8()..]);
        } else {
            return None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn accepts(atomic_type: AtomicType, lexical: &str) -> bool {
        extra_lexical_restriction(atomic_type, lexical).is_ok()
    }

    #[test]
    fn a_t_separator_needs_a_time_component() {
        assert!(!accepts(AtomicType::Duration, "P1Y24MT"));
        assert!(!accepts(AtomicType::DayTimeDuration, "P1DT"));
        assert!(!accepts(AtomicType::YearMonthDuration, "P1Y24MT"));
        assert!(accepts(AtomicType::Duration, "P1Y24MT1S"));
    }

    #[test]
    fn the_seconds_decimal_point_needs_digits_on_both_sides() {
        assert!(!accepts(AtomicType::Duration, "PT.5S"));
        assert!(!accepts(AtomicType::Duration, "PT30.S"));
        assert!(!accepts(AtomicType::DayTimeDuration, "PT10M.5S"));
        assert!(!accepts(AtomicType::DayTimeDuration, "PT10M30.S"));
        assert!(accepts(AtomicType::Duration, "PT0.5S"));
        assert!(accepts(AtomicType::Duration, "PT30S"));
    }

    #[test]
    fn well_formed_durations_are_accepted() {
        for lexical in [
            "P1Y",
            "-P1Y",
            "P1Y2M3DT4H5M6S",
            "PT0S",
            "P1D",
            "-P1Y2M",
            "PT1.000S",
        ] {
            assert!(accepts(AtomicType::Duration, lexical), "{lexical}");
        }
    }

    #[test]
    fn hour_24_is_only_midnight() {
        assert!(!accepts(AtomicType::Time, "24:01:00"));
        assert!(!accepts(AtomicType::Time, "24:00:01"));
        assert!(!accepts(AtomicType::Time, "24:00:00.1"));
        assert!(!accepts(AtomicType::DateTime, "2002-02-15T24:00:00.001"));
        assert!(!accepts(AtomicType::DateTime, "2002-02-15T24:01:00"));
        assert!(accepts(AtomicType::Time, "24:00:00"));
        assert!(accepts(AtomicType::Time, "24:00:00.0"));
        assert!(accepts(AtomicType::Time, "24:00:00Z"));
        assert!(accepts(AtomicType::Time, "24:00:00-05:00"));
        assert!(accepts(AtomicType::Time, "23:59:59"));
        assert!(accepts(AtomicType::DateTime, "2002-02-15T24:00:00"));
    }

    #[test]
    fn year_zero_is_outside_xsd_10() {
        assert!(!accepts(AtomicType::GYear, "0000"));
        assert!(!accepts(AtomicType::GYear, "-0000"));
        assert!(!accepts(AtomicType::GYearMonth, "0000-05"));
        assert!(!accepts(AtomicType::Date, "0000-01-01"));
        assert!(!accepts(AtomicType::DateTime, "0000-01-01T00:00:00"));
        assert!(accepts(AtomicType::GYear, "0001"));
        assert!(accepts(AtomicType::GYear, "-0001"));
        assert!(accepts(AtomicType::Date, "2002-02-15"));
        assert!(accepts(AtomicType::Date, "-2002-02-15Z"));
        assert!(accepts(AtomicType::DateTime, "2002-02-15T12:00:00+05:30"));
    }

    #[test]
    fn gmonth_has_no_trailing_dashes() {
        assert!(!accepts(AtomicType::GMonth, "--02--"));
        assert!(accepts(AtomicType::GMonth, "--02"));
        assert!(accepts(AtomicType::GMonth, "--02Z"));
        assert!(accepts(AtomicType::GMonth, "--02+05:00"));
    }

    #[test]
    fn types_without_an_extra_restriction_are_untouched() {
        assert!(accepts(AtomicType::String, "anything at all"));
        assert!(accepts(AtomicType::GDay, "---05"));
        assert!(accepts(AtomicType::GMonthDay, "--02-03"));
    }

    #[test]
    fn a_malformed_month_beats_a_year_overflow() {
        assert!(!accepts(
            AtomicType::GYearMonth,
            "99999999999999999999999999999-XX"
        ));
        assert!(accepts(AtomicType::GYearMonth, "2002-05"));
        assert!(accepts(AtomicType::GYearMonth, "-2002-05Z"));
    }
}
