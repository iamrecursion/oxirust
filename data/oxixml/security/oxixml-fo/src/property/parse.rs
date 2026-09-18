//! Turning the text of a property specification into a computed [`Value`].
//!
//! This is the second half of *refinement*: the attribute text has already had
//! any shorthand expanded (see [`shorthand`](crate::property::shorthand)), so
//! what arrives here is the specification of a single property — or of a single
//! component of a compound one — which is evaluated as an expression and then
//! converted into the datatype the property expects.

use crate::error::{FoError, Result};
use crate::property::datatype::{
    Conditionality, Keep, KeepValue, Length, LengthBpIp, LengthConditional, LengthRange, Measure,
    Precedence, Shape, Space,
};
use crate::property::expr::{Eval, EvalContext, evaluate, parse as parse_expr};
use crate::property::id::PropertyId;
use crate::property::key::PropertyKey;
use crate::property::spec::{Compound, PropertySpec, forms};
use crate::property::value::Value;

/// The outcome of parsing one property specification.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Specified {
    /// The literal keyword `inherit`.
    Inherit,
    /// A computed value.
    Value(Value),
}

/// How the items of a list-valued property are separated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ListKind {
    /// Items are separated by commas, e.g. `font-family`.
    Comma,
    /// Items are separated by whitespace, e.g. `text-decoration`.
    Whitespace,
}

fn list_kind(id: PropertyId) -> Option<ListKind> {
    match id {
        PropertyId::FontFamily | PropertyId::VoiceFamily | PropertyId::TextShadow => {
            Some(ListKind::Comma)
        }
        PropertyId::SourceDocument
        | PropertyId::SwitchTo
        | PropertyId::AllowedHeightScale
        | PropertyId::AllowedWidthScale => Some(ListKind::Whitespace),
        _ => None,
    }
}

/// Border widths named by the `thin`, `medium` and `thick` keywords.
///
/// XSL 1.1 §7.8.9 leaves these to the implementation; the values below are the
/// ones Apache FOP uses, so that documents render identically.
#[must_use]
pub(crate) fn border_width_keyword(keyword: &str) -> Option<Length> {
    Some(Length::from_points(match keyword {
        "thin" => 0.5,
        "medium" => 1.0,
        "thick" => 2.0,
        _ => return None,
    }))
}

/// The `<absolute-size>` scale of CSS2, anchored at `medium = 12pt` with the
/// recommended 1.2 scaling factor.
#[must_use]
pub(crate) fn absolute_font_size(keyword: &str) -> Option<Length> {
    Some(Length::from_points(match keyword {
        "xx-small" => 12.0 / 1.2 / 1.2 / 1.2,
        "x-small" => 12.0 / 1.2 / 1.2,
        "small" => 12.0 / 1.2,
        "medium" => 12.0,
        "large" => 12.0 * 1.2,
        "x-large" => 12.0 * 1.2 * 1.2,
        "xx-large" => 12.0 * 1.2 * 1.2 * 1.2,
        _ => return None,
    }))
}

/// Splits `input` at top-level separators, ignoring anything inside parentheses
/// or quotes.
pub(crate) fn split_top_level(input: &str, kind_comma: bool) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut depth = 0usize;
    let mut quote: Option<char> = None;
    for character in input.chars() {
        if let Some(open) = quote {
            current.push(character);
            if character == open {
                quote = None;
            }
            continue;
        }
        match character {
            '"' | '\'' => {
                quote = Some(character);
                current.push(character);
            }
            '(' => {
                depth += 1;
                current.push(character);
            }
            ')' => {
                depth = depth.saturating_sub(1);
                current.push(character);
            }
            ',' if depth == 0 && kind_comma => {
                parts.push(core::mem::take(&mut current));
            }
            c if c.is_whitespace() && depth == 0 && !kind_comma => {
                if !current.trim().is_empty() {
                    parts.push(core::mem::take(&mut current));
                } else {
                    current.clear();
                }
            }
            other => current.push(other),
        }
    }
    if !current.trim().is_empty() {
        parts.push(current);
    }
    parts
        .into_iter()
        .map(|part| part.trim().to_owned())
        .filter(|part| !part.is_empty())
        .collect()
}

/// Extracts the IRI from a `url( … )` specification.
///
/// ```text
/// <uri-specification> ::= "url(" S? ("'" | '"')? IRI ("'" | '"')? S? ")"
/// ```
#[must_use]
pub(crate) fn parse_uri_specification(input: &str) -> Option<String> {
    let trimmed = input.trim();
    let rest = trimmed.strip_prefix("url(")?;
    let inner = rest.strip_suffix(')')?.trim();
    let unquoted = if let Some(stripped) = inner.strip_prefix('"') {
        stripped.strip_suffix('"')?
    } else if let Some(stripped) = inner.strip_prefix('\'') {
        stripped.strip_suffix('\'')?
    } else {
        inner
    };
    Some(unquoted.to_owned())
}

fn parse_measure(raw: &str, context: &dyn EvalContext) -> Result<Measure> {
    let trimmed = raw.trim();
    if trimmed == "auto" {
        return Ok(Measure::Auto);
    }
    let expr = parse_expr(context.property_name(), trimmed)?;
    match evaluate(&expr, context)? {
        Eval::Numeric { value, power: 1 } => Ok(Measure::Length(Length::from_points(value))),
        Eval::Numeric {
            value: 0.0,
            power: 0,
        } => Ok(Measure::ZERO),
        Eval::Percent(percent) => Ok(Measure::Percent(percent)),
        Eval::Token(token) if token == "auto" => Ok(Measure::Auto),
        Eval::Value(Value::Length(length)) => Ok(Measure::Length(length)),
        Eval::Value(Value::Percent(percent)) => Ok(Measure::Percent(percent)),
        Eval::Value(value) => value.as_measure().ok_or_else(|| {
            FoError::invalid_value(context.property_name(), trimmed, "expected a length")
        }),
        other => Err(FoError::invalid_value(
            context.property_name(),
            trimmed,
            format!("expected a length, got {other:?}"),
        )),
    }
}

fn parse_integer(raw: &str, context: &dyn EvalContext) -> Result<i64> {
    let expr = parse_expr(context.property_name(), raw.trim())?;
    match evaluate(&expr, context)? {
        Eval::Numeric { value, power: 0 } => Ok(value.round() as i64),
        other => Err(FoError::invalid_value(
            context.property_name(),
            raw,
            format!("expected an integer, got {other:?}"),
        )),
    }
}

/// Parses the value of one component of a compound property.
fn parse_component(
    spec: &PropertySpec,
    component: &str,
    raw: &str,
    context: &dyn EvalContext,
) -> Result<Value> {
    let trimmed = raw.trim();
    match (spec.compound, component) {
        (Compound::Space, "precedence")
        | (Compound::Space, "conditionality")
        | (Compound::LengthConditional, "conditionality") => {
            if let Some(conditionality) = Conditionality::parse(trimmed) {
                return Ok(Value::keyword(conditionality.name()));
            }
            if trimmed == "force" {
                return Ok(Value::keyword("force"));
            }
            Ok(Value::Integer(parse_integer(trimmed, context)?))
        }
        (Compound::Keep, _) => match KeepValue::parse(trimmed) {
            Some(KeepValue::Auto) => Ok(Value::keyword("auto")),
            Some(KeepValue::Always) => Ok(Value::keyword("always")),
            Some(KeepValue::Strength(strength)) => Ok(Value::Integer(i64::from(strength))),
            None => Ok(Value::Integer(parse_integer(trimmed, context)?)),
        },
        (Compound::None, _) => Err(FoError::UnknownComponent {
            property: spec.name.to_owned(),
            component: component.to_owned(),
        }),
        _ => {
            if let Some(length) = border_width_keyword(trimmed)
                && spec.name.ends_with("-width")
            {
                return Ok(Value::Length(length));
            }
            Ok(match parse_measure(trimmed, context)? {
                Measure::Length(length) => Value::Length(length),
                Measure::Percent(percent) => Value::Percent(percent),
                Measure::Auto => Value::keyword("auto"),
            })
        }
    }
}

fn measure_of(value: &Value) -> Measure {
    value.as_measure().unwrap_or(Measure::Auto)
}

/// Applies a component value onto an existing compound value.
pub(crate) fn set_component(base: &mut Value, compound: Compound, component: &str, value: &Value) {
    match (compound, base) {
        (Compound::Space, Value::Space(space)) => match component {
            "minimum" => space.minimum = measure_of(value),
            "optimum" => space.optimum = measure_of(value),
            "maximum" => space.maximum = measure_of(value),
            "precedence" => {
                space.precedence = match value {
                    Value::Keyword(keyword) if &**keyword == "force" => Precedence::Force,
                    other => Precedence::Value(other.as_integer().unwrap_or(0) as i32),
                }
            }
            "conditionality" => {
                space.conditionality = value
                    .as_str()
                    .and_then(Conditionality::parse)
                    .unwrap_or_default();
            }
            _ => {}
        },
        (Compound::LengthRange, Value::LengthRange(range)) => match component {
            "minimum" => range.minimum = measure_of(value),
            "optimum" => range.optimum = measure_of(value),
            "maximum" => range.maximum = measure_of(value),
            _ => {}
        },
        (Compound::LengthConditional, Value::LengthConditional(conditional)) => match component {
            "length" => conditional.length = measure_of(value),
            "conditionality" => {
                conditional.conditionality = value
                    .as_str()
                    .and_then(Conditionality::parse)
                    .unwrap_or_default();
            }
            _ => {}
        },
        (Compound::LengthBpIp, Value::LengthBpIp(bp_ip)) => match component {
            "block-progression-direction" => bp_ip.block_progression_direction = measure_of(value),
            "inline-progression-direction" => {
                bp_ip.inline_progression_direction = measure_of(value)
            }
            _ => {}
        },
        (Compound::Keep, Value::Keep(keep)) => {
            let keep_value = match value {
                Value::Keyword(keyword) if &**keyword == "always" => KeepValue::Always,
                Value::Keyword(_) => KeepValue::Auto,
                other => other.as_integer().map_or(KeepValue::Auto, |strength| {
                    KeepValue::Strength(strength as i32)
                }),
            };
            match component {
                "within-line" => keep.within_line = keep_value,
                "within-column" => keep.within_column = keep_value,
                "within-page" => keep.within_page = keep_value,
                _ => {}
            }
        }
        _ => {}
    }
}

fn parse_shape(spec: &PropertySpec, raw: &str, context: &dyn EvalContext) -> Result<Value> {
    let trimmed = raw.trim();
    let inner = trimmed
        .strip_prefix("rect")
        .map(str::trim_start)
        .and_then(|rest| rest.strip_prefix('('))
        .and_then(|rest| rest.strip_suffix(')'))
        .ok_or_else(|| {
            FoError::invalid_value(
                spec.name,
                trimmed,
                "expected `rect(<top> <right> <bottom> <left>)`",
            )
        })?;
    let parts: Vec<String> = split_top_level(&inner.replace(',', " "), false);
    if parts.len() != 4 {
        return Err(FoError::invalid_value(
            spec.name,
            trimmed,
            "`rect()` takes exactly four offsets",
        ));
    }
    Ok(Value::Shape(Shape {
        top: parse_measure(&parts[0], context)?,
        right: parse_measure(&parts[1], context)?,
        bottom: parse_measure(&parts[2], context)?,
        left: parse_measure(&parts[3], context)?,
    }))
}

fn convert(spec: &PropertySpec, raw: &str, value: Eval) -> Result<Value> {
    let accepts = |mask: u32| spec.forms & mask != 0;
    match value {
        Eval::Token(token) => {
            if spec.accepts_keyword(&token) {
                return Ok(Value::keyword(&token));
            }
            if accepts(forms::COLOR)
                && let Some(color) = crate::property::datatype::Color::from_keyword(&token)
            {
                return Ok(Value::Color(color));
            }
            if accepts(forms::CHARACTER) && token.chars().count() == 1 {
                let mut characters = token.chars();
                if let Some(character) = characters.next() {
                    return Ok(Value::Character(character));
                }
            }
            if accepts(forms::NAME) {
                return Ok(Value::Name(token.into()));
            }
            if accepts(forms::STRING) {
                return Ok(Value::Text(token.into()));
            }
            Err(FoError::invalid_value(
                spec.name,
                raw,
                format!("`{token}` is not one of {:?}", spec.keywords),
            ))
        }
        Eval::Literal(text) => {
            if accepts(forms::STRING) {
                Ok(Value::Text(text.into()))
            } else if accepts(forms::URI) {
                Ok(Value::Uri(text.into()))
            } else if accepts(forms::NAME) {
                Ok(Value::Name(text.into()))
            } else {
                Err(FoError::invalid_value(
                    spec.name,
                    raw,
                    "a string literal is not allowed here",
                ))
            }
        }
        Eval::Numeric { value, power: 0 } => {
            if accepts(forms::INTEGER) && value.fract() == 0.0 {
                Ok(Value::Integer(value as i64))
            } else if accepts(forms::NUMBER) {
                Ok(Value::Number(value))
            } else if accepts(forms::INTEGER) {
                Ok(Value::Integer(value.round() as i64))
            } else if accepts(forms::LENGTH) && value == 0.0 {
                // A bare `0` is universally written for a zero length.
                Ok(Value::Length(Length::ZERO))
            } else if accepts(forms::ANGLE) {
                Ok(Value::Angle(value.rem_euclid(360.0)))
            } else if accepts(forms::STRING) {
                Ok(Value::Text(raw.trim().into()))
            } else {
                Err(FoError::invalid_value(
                    spec.name,
                    raw,
                    "a unit-less number is not allowed here",
                ))
            }
        }
        Eval::Numeric { value, power: 1 } => {
            if accepts(forms::LENGTH) {
                Ok(Value::Length(Length::from_points(value)))
            } else {
                Err(FoError::invalid_value(
                    spec.name,
                    raw,
                    "a length is not allowed here",
                ))
            }
        }
        Eval::Numeric { power, .. } => Err(FoError::invalid_value(
            spec.name,
            raw,
            format!("the value of a property must have unit power 0 or 1, got {power}"),
        )),
        Eval::Percent(percent) => {
            if accepts(forms::PERCENT) {
                Ok(Value::Percent(percent))
            } else {
                Err(FoError::invalid_value(
                    spec.name,
                    raw,
                    "a percentage is not allowed here",
                ))
            }
        }
        Eval::Color(color) => {
            if accepts(forms::COLOR) {
                Ok(Value::Color(color))
            } else {
                Err(FoError::invalid_value(
                    spec.name,
                    raw,
                    "a colour is not allowed here",
                ))
            }
        }
        Eval::Value(value) => Ok(value),
    }
}

fn parse_atom(spec: &PropertySpec, raw: &str, context: &dyn EvalContext) -> Result<Value> {
    let trimmed = raw.trim();
    let accepts = |mask: u32| spec.forms & mask != 0;

    if accepts(forms::URI)
        && let Some(uri) = parse_uri_specification(trimmed)
    {
        return Ok(Value::Uri(uri.into()));
    }
    if accepts(forms::SHAPE) && trimmed.starts_with("rect") {
        return parse_shape(spec, trimmed, context);
    }
    if accepts(forms::CHARACTER) && raw.chars().count() == 1 {
        if let Some(character) = raw.chars().next() {
            return Ok(Value::Character(character));
        }
    }
    if accepts(forms::ANGLE)
        && let Some(angle) = parse_angle(trimmed)
    {
        return Ok(Value::Angle(angle));
    }
    if accepts(forms::TIME | forms::FREQUENCY) && !spec.accepts_keyword(trimmed) {
        // <time> and <frequency> have no arithmetic in the expression language
        // (XSL 1.1 §5.9 note); they are carried through as strings.
        if trimmed.ends_with("ms")
            || trimmed.ends_with('s')
            || trimmed.ends_with("Hz")
            || trimmed.ends_with("kHz")
        {
            return Ok(Value::Text(trimmed.into()));
        }
    }

    match parse_expr(spec.name, trimmed).and_then(|expr| evaluate(&expr, context)) {
        Ok(value) => convert(spec, trimmed, value),
        Err(error) => {
            if accepts(forms::STRING) {
                // §5.11 note: an implementation may recover from an
                // unevaluatable <string> by using the unevaluated text.
                Ok(Value::Text(trimmed.into()))
            } else {
                Err(error)
            }
        }
    }
}

fn parse_angle(input: &str) -> Option<f64> {
    for (suffix, factor) in [
        ("deg", 1.0),
        ("grad", 360.0 / 400.0),
        ("rad", 180.0 / core::f64::consts::PI),
    ] {
        if let Some(number) = input.strip_suffix(suffix)
            && let Ok(value) = number.trim().parse::<f64>()
        {
            return Some((value * factor).rem_euclid(360.0));
        }
    }
    None
}

/// Parses the whole (short-form) specification of a property.
fn parse_whole(spec: &PropertySpec, raw: &str, context: &dyn EvalContext) -> Result<Value> {
    let trimmed = raw.trim();
    if let Some(kind) = list_kind(spec.id) {
        let items = split_top_level(trimmed, kind == ListKind::Comma);
        if items.len() > 1 {
            let mut values = Vec::with_capacity(items.len());
            for item in &items {
                values.push(parse_atom(spec, item, context)?);
            }
            return Ok(Value::List(values));
        }
    }

    match spec.compound {
        Compound::None => parse_atom(spec, trimmed, context),
        Compound::Keep => {
            let keep = KeepValue::parse(trimmed).map_or_else(
                || parse_integer(trimmed, context).map(|value| KeepValue::Strength(value as i32)),
                Ok,
            )?;
            Ok(Value::Keep(Keep::uniform(keep)))
        }
        Compound::Space => {
            if spec.accepts_keyword(trimmed) {
                return Ok(Value::keyword(trimmed));
            }
            if let Ok(measure) = parse_measure(trimmed, context) {
                return Ok(Value::Space(Space::uniform(measure).normalized()));
            }
            parse_atom(spec, trimmed, context)
        }
        Compound::LengthRange => {
            if trimmed != "auto" && spec.accepts_keyword(trimmed) {
                return Ok(Value::keyword(trimmed));
            }
            let measure = parse_measure(trimmed, context)?;
            Ok(Value::LengthRange(
                LengthRange::uniform(measure).normalized(),
            ))
        }
        Compound::LengthConditional => {
            if let Some(length) = border_width_keyword(trimmed) {
                return Ok(Value::LengthConditional(LengthConditional::new(
                    Measure::Length(length),
                )));
            }
            let measure = parse_measure(trimmed, context)?;
            Ok(Value::LengthConditional(LengthConditional::new(measure)))
        }
        Compound::LengthBpIp => {
            let measure = parse_measure(trimmed, context)?;
            Ok(Value::LengthBpIp(LengthBpIp::uniform(measure)))
        }
    }
}

/// Parses one property specification into a computed value.
///
/// `key` may address a whole property or a single component of a compound one.
pub(crate) fn parse_specified(
    key: PropertyKey,
    raw: &str,
    context: &dyn EvalContext,
) -> Result<Specified> {
    let spec = key.id.spec();
    if raw.trim() == "inherit" {
        return Ok(Specified::Inherit);
    }
    let value = match key.component {
        Some(component) => parse_component(spec, component, raw, context)?,
        None => parse_whole(spec, raw, context)?,
    };
    Ok(Specified::Value(value))
}

/// Builds the initial value of a property from the text in the property table.
///
/// Returns [`Value::Absent`] for properties that have no initial value and for
/// the internal markers, which the refinement engine resolves itself.
pub(crate) fn initial_value(id: PropertyId, context: &dyn EvalContext) -> Value {
    let spec = id.spec();
    if spec.initial.is_empty() || spec.initial.starts_with('#') {
        return match spec.compound {
            Compound::Space => Value::Space(Space::zero()),
            Compound::Keep => Value::Keep(Keep::default()),
            Compound::LengthRange if spec.initial == "#leader-length" => {
                Value::LengthRange(LengthRange {
                    minimum: Measure::ZERO,
                    optimum: Measure::Length(Length::from_points(12.0)),
                    maximum: Measure::Percent(100.0),
                })
            }
            _ => Value::Absent,
        };
    }
    parse_whole(spec, spec.initial, context).unwrap_or(Value::Absent)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_level_splitting_respects_parentheses_and_quotes() {
        assert_eq!(
            split_top_level("rgb(1, 2, 3), serif", true),
            vec!["rgb(1, 2, 3)".to_owned(), "serif".to_owned()]
        );
        assert_eq!(
            split_top_level("underline  overline", false),
            vec!["underline".to_owned(), "overline".to_owned()]
        );
        assert_eq!(
            split_top_level("\"a b\", c", true),
            vec!["\"a b\"".to_owned(), "c".to_owned()]
        );
    }

    #[test]
    fn uri_specifications_are_unwrapped() {
        assert_eq!(
            parse_uri_specification("url('logo.png')").as_deref(),
            Some("logo.png")
        );
        assert_eq!(
            parse_uri_specification("url( \"http://example.org/a b\" )").as_deref(),
            Some("http://example.org/a b")
        );
        assert_eq!(parse_uri_specification("logo.png"), None);
    }

    #[test]
    fn angles_normalize_to_degrees() {
        assert_eq!(parse_angle("90deg"), Some(90.0));
        assert_eq!(parse_angle("400grad"), Some(0.0));
        assert_eq!(parse_angle("-90deg"), Some(270.0));
    }

    #[test]
    fn border_and_font_keyword_lengths() {
        assert_eq!(
            border_width_keyword("medium"),
            Some(Length::from_points(1.0))
        );
        assert_eq!(
            absolute_font_size("medium"),
            Some(Length::from_points(12.0))
        );
        assert!(absolute_font_size("large").is_some());
    }
}
