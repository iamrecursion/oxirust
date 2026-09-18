//! The `math:` namespace of [F&O 3.1 §4.8](https://www.w3.org/TR/xpath-functions-31/#trigonometry).
//!
//! Every function here takes `xs:double?` and returns `xs:double?`: an empty argument yields the
//! empty sequence, and the IEEE 754 special cases (`NaN`, `±INF`, signed zero) propagate exactly
//! as the specification's tables require, because they propagate exactly as Rust's `f64`
//! operations do.

use crate::error::{XPathError, XPathResult};
use crate::value::XpathValue;
use oxixml_infoset::AtomicValue;

/// Whether `math:local` exists at `arity`.
pub(crate) fn supports(local: &str, arity: usize) -> bool {
    matches!(
        (local, arity),
        ("pi", 0)
            | ("exp", 1)
            | ("exp10", 1)
            | ("log", 1)
            | ("log10", 1)
            | ("sqrt", 1)
            | ("sin", 1)
            | ("cos", 1)
            | ("tan", 1)
            | ("asin", 1)
            | ("acos", 1)
            | ("atan", 1)
            | ("pow", 2)
            | ("atan2", 2)
    )
}

/// Invokes `math:local`.
///
/// # Errors
///
/// [`XPathErrorCode::Xpty0004`](crate::XPathErrorCode::Xpty0004) when an argument is not at most
/// one atomic value.
pub(crate) fn call(local: &str, arguments: &[XpathValue]) -> XPathResult<XpathValue> {
    if local == "pi" {
        return Ok(double(std::f64::consts::PI));
    }
    let Some(first) = argument(arguments, 0)? else {
        return Ok(XpathValue::empty());
    };
    match local {
        "exp" => Ok(double(first.exp())),
        "exp10" => Ok(double(10.0_f64.powf(first))),
        "log" => Ok(double(first.ln())),
        "log10" => Ok(double(first.log10())),
        "sqrt" => Ok(double(first.sqrt())),
        "sin" => Ok(double(first.sin())),
        "cos" => Ok(double(first.cos())),
        "tan" => Ok(double(first.tan())),
        "asin" => Ok(double(first.asin())),
        "acos" => Ok(double(first.acos())),
        "atan" => Ok(double(first.atan())),
        "pow" => {
            let Some(exponent) = argument(arguments, 1)? else {
                return Ok(XpathValue::empty());
            };
            Ok(double(first.powf(exponent)))
        }
        "atan2" => {
            let Some(second) = argument(arguments, 1)? else {
                return Ok(XpathValue::empty());
            };
            Ok(double(first.atan2(second)))
        }
        other => Err(XPathError::type_error(format!(
            "`math:{other}` is not implemented"
        ))),
    }
}

fn argument(arguments: &[XpathValue], index: usize) -> XPathResult<Option<f64>> {
    let Some(value) = arguments.get(index) else {
        return Ok(None);
    };
    let Some(atomic) = value.atomize_at_most_one("a math: argument")? else {
        return Ok(None);
    };
    Ok(Some(oxixml_fn::numeric::cast_to_double(&atomic)))
}

fn double(value: f64) -> XpathValue {
    XpathValue::from(AtomicValue::double(value))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call_with(local: &str, arguments: &[f64]) -> XPathResult<String> {
        let values: Vec<XpathValue> = arguments
            .iter()
            .map(|value| XpathValue::from(AtomicValue::double(*value)))
            .collect();
        Ok(call(local, &values)?.to_string())
    }

    #[test]
    fn the_catalogue_matches_the_specification() {
        assert!(supports("pi", 0));
        assert!(supports("pow", 2));
        assert!(!supports("pow", 1));
        assert!(!supports("tau", 0));
    }

    #[test]
    fn values_match_the_specification_examples() -> XPathResult<()> {
        assert_eq!(call("pi", &[])?.to_string(), "3.141592653589793");
        assert_eq!(call_with("exp", &[0.0])?, "1");
        assert_eq!(call_with("exp10", &[3.0])?, "1000");
        assert_eq!(call_with("log", &[1.0])?, "0");
        assert_eq!(call_with("log10", &[1000.0])?, "3");
        assert_eq!(call_with("sqrt", &[4.0])?, "2");
        assert_eq!(call_with("pow", &[2.0, 10.0])?, "1024");
        assert_eq!(call_with("atan2", &[0.0, 1.0])?, "0");
        assert_eq!(call_with("sin", &[0.0])?, "0");
        assert_eq!(call_with("cos", &[0.0])?, "1");
        Ok(())
    }

    #[test]
    fn an_empty_argument_gives_the_empty_sequence() -> XPathResult<()> {
        assert!(call("sqrt", &[XpathValue::empty()])?.is_empty());
        assert!(call("pow", &[XpathValue::empty(), XpathValue::empty()])?.is_empty());
        Ok(())
    }

    #[test]
    fn ieee_specials_propagate() -> XPathResult<()> {
        assert_eq!(call_with("sqrt", &[-1.0])?, "NaN");
        assert_eq!(call_with("log", &[0.0])?, "-INF");
        assert_eq!(call_with("exp", &[f64::INFINITY])?, "INF");
        Ok(())
    }
}
