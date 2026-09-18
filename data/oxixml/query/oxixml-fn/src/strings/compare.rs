//! The collation-sensitive string functions of
//! [F&O 3.1 §5.3 and §5.5](https://www.w3.org/TR/xpath-functions-31/#string-compare):
//! `fn:compare`, `fn:codepoint-equal`, and the five substring-matching functions.
//!
//! Each of them takes a collation. Passing `None` selects the context's default collation;
//! passing a URI that names no known collation raises `FOCH0002`; and passing a collation that
//! cannot decompose a string into collation units makes the substring-matching functions raise
//! `FOCH0004`.

use crate::collation::{
    Collation, collation_contains, collation_ends_with, collation_starts_with,
    collation_substring_after, collation_substring_before,
};
use crate::context::FnContext;
use crate::error::{FnError, FnErrorCode, FnResult};

/// [`fn:compare`](https://www.w3.org/TR/xpath-functions-31/#func-compare): `-1`, `0` or `1`.
///
/// # Errors
///
/// [`FnErrorCode::Foch0002`] when `collation` is unknown.
///
/// # Examples
///
/// ```
/// use oxixml_fn::{strings::compare, FnContext, HTML_ASCII_CASE_INSENSITIVE_COLLATION};
///
/// let context = FnContext::new();
/// assert_eq!(compare(&context, "abc", "abc", None)?, 0);
/// assert_eq!(compare(&context, "Strasse", "Straße", None)?, -1);
/// assert_eq!(
///     compare(&context, "ABC", "abc", Some(HTML_ASCII_CASE_INSENSITIVE_COLLATION))?,
///     0
/// );
/// # Ok::<(), oxixml_fn::FnError>(())
/// ```
pub fn compare(
    context: &FnContext,
    left: &str,
    right: &str,
    collation: Option<&str>,
) -> FnResult<i64> {
    let collation = context.collation(collation)?;
    Ok(match collation.compare(left, right) {
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Greater => 1,
    })
}

/// [`fn:codepoint-equal`](https://www.w3.org/TR/xpath-functions-31/#func-codepoint-equal):
/// equality under the code point collation, whatever the default collation is.
///
/// # Examples
///
/// ```
/// use oxixml_fn::strings::codepoint_equal;
///
/// assert!(codepoint_equal("abcd", "abcd"));
/// assert!(!codepoint_equal("abcd", "abcd "));
/// assert!(codepoint_equal("", ""));
/// ```
#[must_use]
pub fn codepoint_equal(left: &str, right: &str) -> bool {
    left == right
}

/// [`fn:contains`](https://www.w3.org/TR/xpath-functions-31/#func-contains).
///
/// # Errors
///
/// [`FnErrorCode::Foch0002`] for an unknown collation,
/// [`FnErrorCode::Foch0004`] for one without collation units.
pub fn contains(
    context: &FnContext,
    haystack: &str,
    needle: &str,
    collation: Option<&str>,
) -> FnResult<bool> {
    with_collation(context, collation, |collation| {
        collation_contains(collation, haystack, needle)
    })
}

/// [`fn:starts-with`](https://www.w3.org/TR/xpath-functions-31/#func-starts-with).
///
/// # Errors
///
/// See [`contains`].
pub fn starts_with(
    context: &FnContext,
    haystack: &str,
    needle: &str,
    collation: Option<&str>,
) -> FnResult<bool> {
    with_collation(context, collation, |collation| {
        collation_starts_with(collation, haystack, needle)
    })
}

/// [`fn:ends-with`](https://www.w3.org/TR/xpath-functions-31/#func-ends-with).
///
/// # Errors
///
/// See [`contains`].
pub fn ends_with(
    context: &FnContext,
    haystack: &str,
    needle: &str,
    collation: Option<&str>,
) -> FnResult<bool> {
    with_collation(context, collation, |collation| {
        collation_ends_with(collation, haystack, needle)
    })
}

/// [`fn:substring-before`](https://www.w3.org/TR/xpath-functions-31/#func-substring-before).
///
/// # Errors
///
/// See [`contains`].
pub fn substring_before(
    context: &FnContext,
    haystack: &str,
    needle: &str,
    collation: Option<&str>,
) -> FnResult<String> {
    with_collation(context, collation, |collation| {
        collation_substring_before(collation, haystack, needle)
    })
}

/// [`fn:substring-after`](https://www.w3.org/TR/xpath-functions-31/#func-substring-after).
///
/// # Errors
///
/// See [`contains`].
pub fn substring_after(
    context: &FnContext,
    haystack: &str,
    needle: &str,
    collation: Option<&str>,
) -> FnResult<String> {
    with_collation(context, collation, |collation| {
        collation_substring_after(collation, haystack, needle)
    })
}

/// [`fn:contains-token`](https://www.w3.org/TR/xpath-functions-31/#func-contains-token):
/// whether any of the whitespace-separated tokens of `input` equals `token` under `collation`.
///
/// The token is trimmed of whitespace before the comparison, and a token that is all whitespace
/// never matches.
///
/// # Errors
///
/// [`FnErrorCode::Foch0002`] for an unknown collation.
///
/// # Examples
///
/// ```
/// use oxixml_fn::{strings::contains_token, FnContext, HTML_ASCII_CASE_INSENSITIVE_COLLATION};
///
/// let context = FnContext::new();
/// assert!(contains_token(&context, &["red green blue".to_owned()], " red ", None)?);
/// assert!(contains_token(&context, &["red".to_owned(), "green".to_owned()], "green", None)?);
/// assert!(!contains_token(&context, &["red green".to_owned()], "blue", None)?);
/// assert!(contains_token(
///     &context,
///     &["red green".to_owned()],
///     "RED",
///     Some(HTML_ASCII_CASE_INSENSITIVE_COLLATION)
/// )?);
/// # Ok::<(), oxixml_fn::FnError>(())
/// ```
pub fn contains_token(
    context: &FnContext,
    input: &[String],
    token: &str,
    collation: Option<&str>,
) -> FnResult<bool> {
    let collation = context.collation(collation)?;
    // F&O 3.1 §5.5.9 tokenizes at *XML* whitespace — `#x9`, `#xA`, `#xD` and `#x20` — not at
    // every Unicode space: `abc<NBSP>def` is one token, so it does not contain the token `abc`.
    let is_xml_space = |character: char| matches!(character, '\t' | '\n' | '\r' | ' ');
    let wanted = token.trim_matches(is_xml_space);
    if wanted.is_empty() {
        return Ok(false);
    }
    Ok(input
        .iter()
        .flat_map(|value| value.split(is_xml_space))
        .filter(|candidate| !candidate.is_empty())
        .any(|candidate| collation.equal(candidate, wanted)))
}

/// [`fn:collation-key`](https://www.w3.org/TR/xpath-functions-31/#func-collation-key): an
/// `xs:base64Binary` value that compares under the code point collation exactly as `value`
/// compares under `collation`.
///
/// The key is the UTF-8 encoding of the collation's own decomposition of the string, which is
/// why a collation with no collation units cannot produce one.
///
/// # Errors
///
/// [`FnErrorCode::Foch0002`] for an unknown collation and
/// [`FnErrorCode::Foch0004`] for one with no collation units.
pub fn collation_key(
    context: &FnContext,
    value: &str,
    collation: Option<&str>,
) -> FnResult<Vec<u8>> {
    let collation = context.collation(collation)?;
    collation.sort_key(value).ok_or_else(|| {
        FnError::new(
            FnErrorCode::Foch0004,
            format!(
                "the collation `{}` cannot produce a collation key",
                collation.uri()
            ),
        )
    })
}

/// [`fn:default-collation`](https://www.w3.org/TR/xpath-functions-31/#func-default-collation).
#[must_use]
pub fn default_collation(context: &FnContext) -> String {
    context.default_collation().uri().to_owned()
}

/// [`fn:static-base-uri`](https://www.w3.org/TR/xpath-functions-31/#func-static-base-uri), which
/// is the empty sequence when the context has no base URI.
#[must_use]
pub fn static_base_uri(context: &FnContext) -> Option<String> {
    context.static_base_uri().map(str::to_owned)
}

fn with_collation<T>(
    context: &FnContext,
    collation: Option<&str>,
    body: impl FnOnce(&dyn Collation) -> FnResult<T>,
) -> FnResult<T> {
    let collation = context.collation(collation)?;
    body(collation.as_ref())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collation::{CASELESS_COLLATION, HTML_ASCII_CASE_INSENSITIVE_COLLATION};
    use crate::error::FnErrorCode;

    fn context() -> FnContext {
        FnContext::new()
    }

    #[test]
    fn compare_examples_from_the_specification() -> FnResult<()> {
        let context = context();
        assert_eq!(compare(&context, "abc", "abc", None)?, 0);
        assert_eq!(compare(&context, "Strasse", "Straße", None)?, -1);
        assert_eq!(
            compare(&context, "Strasse", "Strasse", Some(CASELESS_COLLATION))?,
            0
        );
        Ok(())
    }

    #[test]
    fn an_unknown_collation_raises_foch0002() {
        let error =
            compare(&context(), "a", "b", Some("urn:example:nope")).expect_err("unknown collation");
        assert!(error.is(FnErrorCode::Foch0002));
    }

    #[test]
    fn substring_functions_examples_from_the_specification() -> FnResult<()> {
        let context = context();
        assert!(contains(&context, "tattoo", "t", None)?);
        assert!(!contains(&context, "tattoo", "ttt", None)?);
        assert!(contains(&context, "", "", None)?);
        assert!(starts_with(&context, "tattoo", "tat", None)?);
        assert!(!starts_with(&context, "tattoo", "att", None)?);
        assert!(ends_with(&context, "tattoo", "too", None)?);
        assert!(!ends_with(&context, "tattoo", "att", None)?);
        assert_eq!(substring_before(&context, "tattoo", "attoo", None)?, "t");
        assert_eq!(substring_before(&context, "tattoo", "tatto", None)?, "");
        assert_eq!(substring_after(&context, "tattoo", "tat", None)?, "too");
        assert_eq!(substring_after(&context, "tattoo", "tattoo", None)?, "");
        Ok(())
    }

    #[test]
    fn substring_matching_honours_a_caseless_collation() -> FnResult<()> {
        let context = context();
        assert!(contains(
            &context,
            "abcdefghi",
            "DEF",
            Some(HTML_ASCII_CASE_INSENSITIVE_COLLATION)
        )?);
        assert_eq!(
            substring_after(
                &context,
                "abcdefghi",
                "DEF",
                Some(HTML_ASCII_CASE_INSENSITIVE_COLLATION)
            )?,
            "ghi"
        );
        Ok(())
    }

    #[test]
    fn contains_token_examples_from_the_specification() -> FnResult<()> {
        let context = context();
        assert!(contains_token(
            &context,
            &["red green blue ".to_owned()],
            "red",
            None
        )?);
        assert!(contains_token(
            &context,
            &["red".to_owned(), "green".to_owned(), "blue".to_owned()],
            "red",
            None
        )?);
        assert!(!contains_token(
            &context,
            &["red green blue".to_owned()],
            "or",
            None
        )?);
        assert!(!contains_token(
            &context,
            &["red green blue".to_owned()],
            "  ",
            None
        )?);
        assert!(contains_token(
            &context,
            &["red green blue".to_owned()],
            "GREEN",
            Some(HTML_ASCII_CASE_INSENSITIVE_COLLATION)
        )?);
        Ok(())
    }

    #[test]
    fn collation_keys_order_like_their_collation() -> FnResult<()> {
        let context = context();
        let caseless = FnContext::new().with_default_collation(CASELESS_COLLATION)?;
        assert_eq!(
            collation_key(&caseless, "ABC", None)?,
            collation_key(&caseless, "abc", None)?
        );
        assert_ne!(
            collation_key(&context, "ABC", None)?,
            collation_key(&context, "abc", None)?
        );
        Ok(())
    }

    #[test]
    fn the_context_reports_its_defaults() -> FnResult<()> {
        let context = FnContext::new().with_static_base_uri("http://example.com/");
        assert_eq!(default_collation(&context), crate::CODEPOINT_COLLATION);
        assert_eq!(
            static_base_uri(&context),
            Some("http://example.com/".to_owned())
        );
        assert_eq!(static_base_uri(&FnContext::new()), None);
        Ok(())
    }

    #[test]
    fn codepoint_equal_ignores_the_default_collation() -> FnResult<()> {
        let context = FnContext::new().with_default_collation(CASELESS_COLLATION)?;
        assert_eq!(compare(&context, "ABC", "abc", None)?, 0);
        assert!(!codepoint_equal("ABC", "abc"));
        Ok(())
    }
}
