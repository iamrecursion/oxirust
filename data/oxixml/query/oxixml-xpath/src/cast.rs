//! `cast as` and `castable as`, plus the `xs:*` constructor functions that are the same
//! operation written differently.
//!
//! [F&O 3.1 §19](https://www.w3.org/TR/xpath-functions-31/#casting) defines casting as a matrix
//! over the nineteen primitive types. Most of the matrix is *lexical*: the source value's
//! canonical string is fed through the target type's lexical mapping, which is exactly what
//! [`AtomicValue::parse`] does, facets included. The entries that are **not** lexical are the
//! ones this module spells out:
//!
//! * numeric to numeric, where `xs:double` `3.7` casts to the `xs:integer` `3` by truncation
//!   towards zero rather than by re-reading `"3.7"`;
//! * numeric or `xs:boolean` in either direction, where `0` and `NaN` are `false`;
//! * the date/time family, where `xs:dateTime` to `xs:date` drops the time and keeps the
//!   timezone, and every `xs:gXxx` extraction keeps the timezone too;
//! * the duration family, where `xs:duration` truncates to one of its two halves, and where the
//!   two halves cast to each other as `P0M` and `PT0S`;
//! * `xs:hexBinary` and `xs:base64Binary`, which share a value space and differ only lexically;
//! * `xs:QName`, whose lexical form needs the in-scope namespaces to resolve a prefix.

use crate::context::StaticContext;
use crate::error::{XPathError, XPathErrorCode, XPathResult};
use oxixml_infoset::{AtomicData, AtomicType, AtomicValue, ErrorCode as XdmErrorCode};
use oxixml_xsd::{
    DayTimeDuration, Decimal, Double, Duration, Float, Integer, TimezoneOffset, YearMonthDuration,
};

/// Casts `value` to `target`.
///
/// # Errors
///
/// [`XPathErrorCode::Forg0001`] when the value is not in the target type's value space, and
/// [`XPathErrorCode::Xpty0004`] when no cast between the two types exists at all.
pub(crate) fn cast(
    value: &AtomicValue,
    target: AtomicType,
    context: &StaticContext,
) -> XPathResult<AtomicValue> {
    cast_with(value, target, context, StringSource::Computed)
}

/// Whether a textual operand of a cast to a namespace-sensitive type may have its prefix
/// resolved against the static context.
///
/// XPath 2.0 §3.12.3 allowed that only for a *string literal*, and made every other textual
/// operand `XPTY0004`, because the cast was evaluated during static analysis. XPath 3.0 dropped
/// the restriction (W3C bug 16059): the prefix is resolved against the statically known
/// namespaces whenever the cast runs, so `$var cast as xs:QName` is legal from 3.0 onwards. QT3
/// pins both readings — `K-SeqExprCast-71a` (`XP20`) expects `XPTY0004` where `K-SeqExprCast-71b`
/// (`XP30+`) expects the value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StringSource {
    /// A prefix may be resolved: the operand is a `StringLiteral`, or the version is 3.0+.
    Literal,
    /// XPath 2.0 with an operand that is not a string literal.
    Computed,
}

/// Casts `value` to `target`, knowing whether a textual operand was written as a literal.
///
/// # Errors
///
/// See [`cast`], plus [`XPathErrorCode::Fons0004`] when a literal `prefix:local` names a prefix
/// the static context does not bind.
pub(crate) fn cast_with(
    value: &AtomicValue,
    target: AtomicType,
    context: &StaticContext,
    source_kind: StringSource,
) -> XPathResult<AtomicValue> {
    if value.atomic_type() == target {
        return Ok(value.clone());
    }
    if matches!(target, AtomicType::AnyAtomicType) {
        return Err(XPathError::new(
            XPathErrorCode::Xpst0080,
            "`xs:anyAtomicType` is not a legal cast target",
        ));
    }

    let source = value.atomic_type();

    // Every type casts to the xs:string hierarchy and to xs:untypedAtomic through its string
    // value. xs:anyURI is *not* in that group: F&O admits only the text types as its source.
    if target.primitive() == AtomicType::AnyUri && !source.is_text() {
        return Err(XPathError::type_error(format!(
            "there is no cast from {source} to {target}: only the string types cast to xs:anyURI"
        )));
    }
    if target.is_text() {
        if matches!(value.data(), AtomicData::QName(_))
            && !matches!(target, AtomicType::String | AtomicType::UntypedAtomic)
        {
            return Err(forg0001(
                value,
                target,
                "an xs:QName casts only to xs:string",
            ));
        }
        return parse(target, &crate::value::atomic_string_value(value), value);
    }

    // `xs:anyURI` is a source only for the text types, which the branch above has already dealt
    // with: the casting table of F&O 3.1 §19.1 has no entry from `xs:anyURI` to a number, a
    // boolean or a date, so the cast does not exist and the error is a *type* error.
    if source.primitive() == AtomicType::AnyUri {
        return Err(XPathError::type_error(format!(
            "there is no cast from {source} to {target}: xs:anyURI casts only to the string types"
        )));
    }

    // The string family and xs:untypedAtomic cast to everything through the target's lexical
    // mapping — except xs:QName, whose prefix must be resolved against the static context.
    if source.is_text() {
        let lexical = value.as_str().unwrap_or_default();
        if matches!(target.primitive(), AtomicType::QName | AtomicType::Notation) {
            if source_kind != StringSource::Literal {
                return Err(XPathError::type_error(format!(
                    "only a string literal casts to {target}: the prefix of `{lexical}` can be \
                     resolved in the static context alone"
                )));
            }
            // A prefix that is not in scope is `FONS0004`, the "no namespace found for prefix"
            // condition, rather than the lexical `FORG0001` the data model would report.
            if let Some((prefix, _)) = lexical.trim().split_once(':') {
                if !prefix.is_empty() && context.namespace(prefix).is_none() {
                    return Err(XPathError::new(
                        XPathErrorCode::Fons0004,
                        format!("no namespace is bound to the prefix `{prefix}` of `{lexical}`"),
                    ));
                }
            }
            return AtomicValue::parse_qname(target, lexical, |prefix| match prefix {
                None => context.default_element_namespace().map(str::to_owned),
                Some(prefix) => context.namespace(prefix).map(str::to_owned),
            })
            .map_err(XPathError::from);
        }
        return parse(target, lexical, value);
    }

    if source.is_numeric() && target.is_numeric() {
        return numeric_to_numeric(value, target);
    }
    if source.is_numeric() && target == AtomicType::Boolean {
        let number = value.as_f64().unwrap_or(f64::NAN);
        return Ok(AtomicValue::boolean(number != 0.0 && !number.is_nan()));
    }
    if source == AtomicType::Boolean && target.is_numeric() {
        let truthy = value.as_bool().unwrap_or_default();
        return numeric_to_numeric(&AtomicValue::integer(i64::from(truthy)), target);
    }
    if source.is_date_time() && target.is_date_time() {
        return date_time_to_date_time(value, target);
    }
    if source.primitive() == AtomicType::Duration && target.primitive() == AtomicType::Duration {
        return duration_to_duration(value, target);
    }
    if matches!(
        (source.primitive(), target.primitive()),
        (
            AtomicType::HexBinary | AtomicType::Base64Binary,
            AtomicType::HexBinary | AtomicType::Base64Binary
        )
    ) {
        let bytes = value.as_bytes().unwrap_or_default().to_vec();
        let converted = if target.primitive() == AtomicType::Base64Binary {
            AtomicValue::base64_binary(bytes)
        } else {
            AtomicValue::hex_binary(bytes)
        };
        return parse(target, &converted.string_value(), value);
    }
    if source.primitive() == target.primitive() {
        // A restriction of the same primitive: re-validate the canonical form against the
        // target's facets.
        return parse(target, &value.string_value(), value);
    }

    Err(XPathError::type_error(format!(
        "there is no cast from {source} to {target}"
    )))
}

/// `castable as`: whether [`cast_with`] would succeed.
pub(crate) fn castable(
    value: &AtomicValue,
    target: AtomicType,
    context: &StaticContext,
    source_kind: StringSource,
) -> bool {
    cast_with(value, target, context, source_kind).is_ok()
}

fn parse(target: AtomicType, lexical: &str, source: &AtomicValue) -> XPathResult<AtomicValue> {
    AtomicValue::parse(target, lexical).map_err(|error| {
        // A *lexically invalid* value is `FORG0001`, but a well-formed one that is out of range
        // has its own code — `FODT0001` for the date/time types and `FODT0002` for the
        // durations (F&O 3.1 §10.1.1) — which the data model has already decided. Overwriting
        // it with `FORG0001` would report the wrong condition.
        let code = match error.code() {
            XdmErrorCode::Fodt0001 => XPathErrorCode::Fodt0001,
            XdmErrorCode::Fodt0002 => XPathErrorCode::Fodt0002,
            // F&O 3.1 §19.1 gives the numeric types the same treatment: a *syntactically
            // valid* number the target's value space cannot hold is a range error —
            // `FOCA0001` for `xs:decimal`, `FOCA0003` for `xs:integer` — not the lexical
            // `FORG0001`. The remap used to live only on the numeric-to-numeric path, so a
            // cast from an attribute or text value reported the wrong condition.
            // Only the two unrestricted types qualify: a derived type such as
            // `xs:nonNegativeInteger` rejects a well-formed value by *facet*, which stays
            // `FORG0001`.
            _ if target == AtomicType::Integer && is_integer_lexical(lexical.trim()) => {
                XPathErrorCode::Foca0003
            }
            _ if target == AtomicType::Decimal && is_decimal_lexical(lexical.trim()) => {
                XPathErrorCode::Foca0001
            }
            _ => XPathErrorCode::Forg0001,
        };
        XPathError::new(
            code,
            format!(
                "`{}` cannot be cast to {target}: {}",
                source.string_value(),
                error.message()
            ),
        )
    })
}

fn forg0001(value: &AtomicValue, target: AtomicType, reason: &str) -> XPathError {
    XPathError::new(
        XPathErrorCode::Forg0001,
        format!(
            "`{}` cannot be cast to {target}: {reason}",
            value.string_value()
        ),
    )
}

fn numeric_to_numeric(value: &AtomicValue, target: AtomicType) -> XPathResult<AtomicValue> {
    match target.primitive() {
        AtomicType::Double => {
            let number = value.as_f64().unwrap_or(f64::NAN);
            Ok(AtomicValue::double(Double::from(number)))
        }
        AtomicType::Float => {
            let number = value.as_f64().unwrap_or(f64::NAN);
            #[allow(clippy::cast_possible_truncation)]
            Ok(AtomicValue::float(Float::from(number as f32)))
        }
        _ => {
            // xs:decimal, xs:integer and every restriction of them. The conversion goes through
            // the *shortest round-tripping decimal* rather than through a binary-to-decimal
            // approximation, so `xs:float("1e5")` casts to exactly `100000` rather than to
            // `99999.999999999991611392`.
            let to_integer = target.is_derived_from(AtomicType::Integer);
            let lexical = match value.data() {
                AtomicData::Integer(integer) => integer.to_string(),
                AtomicData::Decimal(decimal) => {
                    if to_integer {
                        Integer::try_from(*decimal)
                            .map_err(|_| too_large(value, target))?
                            .to_string()
                    } else {
                        decimal.to_string()
                    }
                }
                AtomicData::Double(double) => {
                    let number = f64::from(*double);
                    exact_lexical(number, format!("{number}"), to_integer, value, target)?
                }
                AtomicData::Float(float) => {
                    // Render from the single-precision value, so that `xs:float("-1.75e-3")`
                    // becomes `-0.00175` rather than the widened `-0.0017500000540167093`.
                    let number = f32::from(*float);
                    exact_lexical(
                        f64::from(number),
                        format!("{number}"),
                        to_integer,
                        value,
                        target,
                    )?
                }
                other => {
                    // A value that overflowed the fixed width casts through its exact decimal
                    // string, exactly as the fixed forms above do; the re-parse then promotes the
                    // target if it, too, needs arbitrary precision.
                    if let Some(integer) = other.as_big_integer() {
                        integer.to_string()
                    } else if let Some(decimal) = other.as_big_decimal() {
                        if to_integer {
                            decimal
                                .truncate()
                                .ok_or_else(|| too_large(value, target))?
                                .to_string()
                        } else {
                            decimal.to_string()
                        }
                    } else {
                        return Err(XPathError::type_error("a numeric value is required"));
                    }
                }
            };
            parse(target, &lexical, value).map_err(|error| {
                // F&O 3.1 §19.1: a numeric value whose magnitude is outside the range of
                // `xs:decimal` raises `FOCA0001`, not the lexical error `FORG0001` — the form is
                // impeccable, it is the range that is not there. Only a *syntactically valid*
                // decimal can reach this condition, so that is the test.
                if error.code() == Some(XPathErrorCode::Forg0001)
                    && !to_integer
                    && is_decimal_lexical(&lexical)
                {
                    return XPathError::new(
                        XPathErrorCode::Foca0001,
                        format!(
                            "`{}` is outside the range of {target}",
                            crate::value::atomic_string_value(value)
                        ),
                    );
                }
                error
            })
        }
    }
}

/// Whether `lexical` matches the `xs:decimal` lexical grammar, ignoring range.
fn is_decimal_lexical(lexical: &str) -> bool {
    let body = lexical.strip_prefix(['+', '-']).unwrap_or(lexical);
    let (integer, fraction) = body.split_once('.').unwrap_or((body, ""));
    !(integer.is_empty() && fraction.is_empty())
        && integer.bytes().all(|byte| byte.is_ascii_digit())
        && fraction.bytes().all(|byte| byte.is_ascii_digit())
}

/// Whether `lexical` matches the `xs:integer` lexical grammar, ignoring range.
fn is_integer_lexical(lexical: &str) -> bool {
    let body = lexical.strip_prefix(['+', '-']).unwrap_or(lexical);
    !body.is_empty() && body.bytes().all(|byte| byte.is_ascii_digit())
}

/// The lexical form a floating-point value takes when cast to `xs:decimal` or `xs:integer`.
fn exact_lexical(
    value: f64,
    rendered: String,
    to_integer: bool,
    source: &AtomicValue,
    target: AtomicType,
) -> XPathResult<String> {
    if !value.is_finite() {
        return Err(XPathError::new(
            XPathErrorCode::Foca0002,
            format!(
                "`{}` is not finite and cannot be cast to {target}",
                crate::value::atomic_string_value(source)
            ),
        ));
    }
    if to_integer {
        let truncated = value.trunc();
        #[allow(clippy::cast_possible_truncation)]
        let bounded = truncated as i128;
        if i64::try_from(bounded).is_err() {
            return Err(too_large(source, target));
        }
        return Ok(bounded.to_string());
    }
    Ok(rendered)
}

fn too_large(value: &AtomicValue, target: AtomicType) -> XPathError {
    XPathError::new(
        XPathErrorCode::Foca0003,
        format!(
            "`{}` does not fit in {target}",
            crate::value::atomic_string_value(value)
        ),
    )
}

fn date_time_to_date_time(value: &AtomicValue, target: AtomicType) -> XPathResult<AtomicValue> {
    let lexical = match (value.data(), target.primitive()) {
        (AtomicData::DateTime(source), primitive) => {
            let timezone = source.timezone_offset();
            match primitive {
                AtomicType::Date => format!(
                    "{}{}",
                    ymd(source.year(), source.month(), source.day()),
                    zone(timezone)
                ),
                AtomicType::Time => format!(
                    "{}{}",
                    hms(source.hour(), source.minute(), source.second()),
                    zone(timezone)
                ),
                AtomicType::GYearMonth => {
                    format!("{}{}", ym(source.year(), source.month()), zone(timezone))
                }
                AtomicType::GYear => format!("{}{}", year(source.year()), zone(timezone)),
                AtomicType::GMonthDay => {
                    format!(
                        "--{:02}-{:02}{}",
                        source.month(),
                        source.day(),
                        zone(timezone)
                    )
                }
                AtomicType::GMonth => format!("--{:02}{}", source.month(), zone(timezone)),
                AtomicType::GDay => format!("---{:02}{}", source.day(), zone(timezone)),
                _ => return Err(no_cast(value, target)),
            }
        }
        (AtomicData::Date(source), primitive) => {
            let timezone = source.timezone_offset();
            match primitive {
                AtomicType::DateTime => format!(
                    "{}T00:00:00{}",
                    ymd(source.year(), source.month(), source.day()),
                    zone(timezone)
                ),
                AtomicType::GYearMonth => {
                    format!("{}{}", ym(source.year(), source.month()), zone(timezone))
                }
                AtomicType::GYear => format!("{}{}", year(source.year()), zone(timezone)),
                AtomicType::GMonthDay => {
                    format!(
                        "--{:02}-{:02}{}",
                        source.month(),
                        source.day(),
                        zone(timezone)
                    )
                }
                AtomicType::GMonth => format!("--{:02}{}", source.month(), zone(timezone)),
                AtomicType::GDay => format!("---{:02}{}", source.day(), zone(timezone)),
                _ => return Err(no_cast(value, target)),
            }
        }
        _ => return Err(no_cast(value, target)),
    };
    parse(target, &lexical, value)
}

fn no_cast(value: &AtomicValue, target: AtomicType) -> XPathError {
    XPathError::type_error(format!(
        "there is no cast from {} to {target}",
        value.atomic_type()
    ))
}

fn zone(timezone: Option<TimezoneOffset>) -> String {
    timezone
        .map(|offset| offset.to_string())
        .unwrap_or_default()
}

fn year(value: i64) -> String {
    if value < 0 {
        format!("-{:04}", value.unsigned_abs())
    } else {
        format!("{value:04}")
    }
}

fn ym(y: i64, m: u8) -> String {
    format!("{}-{m:02}", year(y))
}

fn ymd(y: i64, m: u8, d: u8) -> String {
    format!("{}-{m:02}-{d:02}", year(y))
}

fn hms(hour: u8, minute: u8, second: Decimal) -> String {
    let seconds = second.to_string();
    let padded = match seconds.split_once('.') {
        Some((whole, _)) if whole.len() == 1 => format!("0{seconds}"),
        None if seconds.len() == 1 => format!("0{seconds}"),
        _ => seconds,
    };
    format!("{hour:02}:{minute:02}:{padded}")
}

fn duration_to_duration(value: &AtomicValue, target: AtomicType) -> XPathResult<AtomicValue> {
    let converted = match (value.data(), target.primitive()) {
        (AtomicData::Duration(source), _) => match target {
            AtomicType::YearMonthDuration => {
                AtomicValue::year_month_duration(year_month_of(*source)?)
            }
            AtomicType::DayTimeDuration => AtomicValue::day_time_duration(day_time_of(*source)?),
            _ => return Err(no_cast(value, target)),
        },
        (AtomicData::YearMonthDuration(source), _) => match target {
            AtomicType::Duration => AtomicValue::duration(Duration::from(*source)),
            AtomicType::DayTimeDuration => {
                AtomicValue::day_time_duration(DayTimeDuration::new(0_i64))
            }
            _ => return Err(no_cast(value, target)),
        },
        (AtomicData::DayTimeDuration(source), _) => match target {
            AtomicType::Duration => AtomicValue::duration(Duration::from(*source)),
            AtomicType::YearMonthDuration => {
                AtomicValue::year_month_duration(YearMonthDuration::new(0_i64))
            }
            _ => return Err(no_cast(value, target)),
        },
        _ => return Err(no_cast(value, target)),
    };
    if converted.atomic_type() == target {
        return Ok(converted);
    }
    parse(target, &converted.string_value(), value)
}

fn year_month_of(source: Duration) -> XPathResult<YearMonthDuration> {
    let months = source
        .years()
        .checked_mul(12)
        .and_then(|years| years.checked_add(source.months()))
        .ok_or_else(|| XPathError::type_error("the duration overflows xs:yearMonthDuration"))?;
    Ok(YearMonthDuration::new(months))
}

fn day_time_of(source: Duration) -> XPathResult<DayTimeDuration> {
    let whole = source
        .days()
        .checked_mul(86_400)
        .and_then(|days| source.hours().checked_mul(3_600).map(|hours| days + hours))
        .and_then(|so_far| {
            source
                .minutes()
                .checked_mul(60)
                .and_then(|minutes| so_far.checked_add(minutes))
        })
        .ok_or_else(|| XPathError::type_error("the duration overflows xs:dayTimeDuration"))?;
    let seconds = Decimal::from(whole)
        .checked_add(source.seconds())
        .ok_or_else(|| XPathError::type_error("the duration overflows xs:dayTimeDuration"))?;
    Ok(DayTimeDuration::new(seconds))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn casted(lexical: &str, from: AtomicType, to: AtomicType) -> XPathResult<String> {
        let context = StaticContext::new();
        let value = AtomicValue::parse(from, lexical)?;
        Ok(crate::value::atomic_string_value(&cast(
            &value, to, &context,
        )?))
    }

    #[test]
    fn numeric_casts_truncate_towards_zero() -> XPathResult<()> {
        assert_eq!(casted("3.7", AtomicType::Double, AtomicType::Integer)?, "3");
        assert_eq!(
            casted("-3.7", AtomicType::Double, AtomicType::Integer)?,
            "-3"
        );
        assert_eq!(
            casted("3.7", AtomicType::Decimal, AtomicType::Integer)?,
            "3"
        );
        assert_eq!(casted("3", AtomicType::Integer, AtomicType::Double)?, "3");
        assert_eq!(casted("1", AtomicType::Integer, AtomicType::Decimal)?, "1");
        assert!(casted("INF", AtomicType::Double, AtomicType::Integer).is_err());
        assert!(casted("NaN", AtomicType::Double, AtomicType::Decimal).is_err());
        Ok(())
    }

    #[test]
    fn casts_between_the_arbitrary_precision_numerics() -> XPathResult<()> {
        let big = "99999999999999999999999999";
        // A big integer casts to decimal and back, and to double (lossily).
        assert_eq!(casted(big, AtomicType::Integer, AtomicType::Decimal)?, big);
        assert_eq!(
            casted(big, AtomicType::Integer, AtomicType::Double)?,
            "1.0E26"
        );
        // A big decimal casts to integer by truncating towards zero.
        assert_eq!(
            casted(
                "123456789012345678901234567890.9",
                AtomicType::Decimal,
                AtomicType::Integer
            )?,
            "123456789012345678901234567890"
        );
        Ok(())
    }

    #[test]
    fn boolean_casts() -> XPathResult<()> {
        assert_eq!(
            casted("0", AtomicType::Integer, AtomicType::Boolean)?,
            "false"
        );
        assert_eq!(
            casted("1", AtomicType::Integer, AtomicType::Boolean)?,
            "true"
        );
        assert_eq!(
            casted("NaN", AtomicType::Double, AtomicType::Boolean)?,
            "false"
        );
        assert_eq!(
            casted("true", AtomicType::Boolean, AtomicType::Integer)?,
            "1"
        );
        assert_eq!(
            casted("false", AtomicType::Boolean, AtomicType::Double)?,
            "0"
        );
        assert_eq!(
            casted("1", AtomicType::String, AtomicType::Boolean)?,
            "true"
        );
        Ok(())
    }

    #[test]
    fn string_casts_go_through_the_canonical_form() -> XPathResult<()> {
        assert_eq!(casted("1.0", AtomicType::Decimal, AtomicType::String)?, "1");
        assert_eq!(
            casted(" 42 ", AtomicType::String, AtomicType::Integer)?,
            "42"
        );
        assert!(casted("x", AtomicType::String, AtomicType::Integer).is_err());
        assert_eq!(
            casted("a b", AtomicType::String, AtomicType::UntypedAtomic)?,
            "a b"
        );
        Ok(())
    }

    #[test]
    fn date_time_casts_keep_the_timezone() -> XPathResult<()> {
        assert_eq!(
            casted(
                "2002-12-31T19:20:00-05:00",
                AtomicType::DateTime,
                AtomicType::Date
            )?,
            "2002-12-31-05:00"
        );
        assert_eq!(
            casted(
                "2002-12-31T19:20:05.5Z",
                AtomicType::DateTime,
                AtomicType::Time
            )?,
            "19:20:05.5Z"
        );
        assert_eq!(
            casted("2002-12-31Z", AtomicType::Date, AtomicType::DateTime)?,
            "2002-12-31T00:00:00Z"
        );
        assert_eq!(
            casted("2002-12-31", AtomicType::Date, AtomicType::GYearMonth)?,
            "2002-12"
        );
        assert_eq!(
            casted("2002-12-31", AtomicType::Date, AtomicType::GYear)?,
            "2002"
        );
        assert_eq!(
            casted("2002-12-31", AtomicType::Date, AtomicType::GMonthDay)?,
            "--12-31"
        );
        assert_eq!(
            casted("2002-12-31", AtomicType::Date, AtomicType::GDay)?,
            "---31"
        );
        Ok(())
    }

    #[test]
    fn duration_casts_truncate_to_one_half() -> XPathResult<()> {
        assert_eq!(
            casted(
                "P1Y2M3DT4H5M6S",
                AtomicType::Duration,
                AtomicType::YearMonthDuration
            )?,
            "P1Y2M"
        );
        assert_eq!(
            casted(
                "P1Y2M3DT4H5M6S",
                AtomicType::Duration,
                AtomicType::DayTimeDuration
            )?,
            "P3DT4H5M6S"
        );
        assert_eq!(
            casted(
                "P1Y",
                AtomicType::YearMonthDuration,
                AtomicType::DayTimeDuration
            )?,
            "PT0S"
        );
        assert_eq!(
            casted(
                "PT1S",
                AtomicType::DayTimeDuration,
                AtomicType::YearMonthDuration
            )?,
            "P0M"
        );
        Ok(())
    }

    #[test]
    fn binary_casts_change_only_the_lexical_form() -> XPathResult<()> {
        assert_eq!(
            casted("0FB7", AtomicType::HexBinary, AtomicType::Base64Binary)?,
            "D7c="
        );
        assert_eq!(
            casted("D7c=", AtomicType::Base64Binary, AtomicType::HexBinary)?,
            "0FB7"
        );
        Ok(())
    }

    #[test]
    fn qname_casts_resolve_against_the_static_context() -> XPathResult<()> {
        let context = StaticContext::new().with_namespace("p", "urn:p");
        let value = AtomicValue::string("p:local");
        let casted = cast_with(&value, AtomicType::QName, &context, StringSource::Literal)?;
        let qname = casted.as_qname().expect("an xs:QName");
        assert_eq!(qname.namespace(), Some("urn:p"));
        assert_eq!(qname.local_name(), "local");
        assert_eq!(
            cast(&casted, AtomicType::String, &context)?.string_value(),
            "p:local"
        );
        assert!(
            cast_with(
                &AtomicValue::string("q:x"),
                AtomicType::QName,
                &context,
                StringSource::Literal
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn only_a_string_literal_casts_to_a_namespace_sensitive_type() {
        let context = StaticContext::new().with_namespace("p", "urn:p");
        let computed = cast(&AtomicValue::string("p:local"), AtomicType::QName, &context)
            .expect_err("a computed string cannot be cast to xs:QName");
        assert_eq!(computed.code(), Some(XPathErrorCode::Xpty0004));
        let unbound = cast_with(
            &AtomicValue::string("q:local"),
            AtomicType::QName,
            &context,
            StringSource::Literal,
        )
        .expect_err("the prefix `q` is not bound");
        assert_eq!(unbound.code(), Some(XPathErrorCode::Fons0004));
    }

    #[test]
    fn impossible_casts_are_type_errors() {
        let context = StaticContext::new();
        let value = AtomicValue::integer(1_i64);
        let failure = cast(&value, AtomicType::Date, &context).expect_err("no such cast");
        assert_eq!(failure.code(), Some(XPathErrorCode::Xpty0004));
        assert!(!castable(
            &value,
            AtomicType::Date,
            &context,
            StringSource::Computed
        ));
        assert!(castable(
            &value,
            AtomicType::String,
            &context,
            StringSource::Computed
        ));
    }
}
