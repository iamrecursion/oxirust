//! The URI functions of
//! [F&O 3.1 §6](https://www.w3.org/TR/xpath-functions-31/#uri-functions): `fn:resolve-uri`,
//! `fn:encode-for-uri`, `fn:iri-to-uri` and `fn:escape-html-uri`.
//!
//! The three escaping functions differ only in which characters they leave alone, so they are
//! all one call to [`oxixml_text::percent::percent_encode_str`] with a different
//! [`AsciiSet`]. RFC 3986 asks producers to emit upper-case
//! hexadecimal, which that function does.

use crate::context::FnContext;
use crate::error::{FnError, FnErrorCode, FnResult};
use oxixml_iri::{Iri, IriRef};
use oxixml_text::percent::{AsciiSet, UNRESERVED, percent_encode_str};

/// The characters `fn:iri-to-uri` leaves unescaped: the printable ASCII range, less the
/// characters RFC 3987 declares invalid in an IRI (`<`, `>`, `"`, space, `{`, `}`, `|`, `\`,
/// `^` and `` ` ``).
const IRI_ALLOWED: AsciiSet = AsciiSet::empty()
    // `!`, then `#`..`;` (skipping `"` and stopping before `<`).
    .add(b'!')
    .add_range(b'#', b';')
    // `=`, skipping `>`.
    .add(b'=')
    // `?`..`[`, skipping `\`.
    .add_range(b'?', b'[')
    // `]`, skipping `^`; `_`, skipping the backtick.
    .add(b']')
    .add(b'_')
    // `a`..`z`, skipping `{`, `|` and `}`; then `~`.
    .add_range(b'a', b'z')
    .add(b'~');

/// The characters `fn:escape-html-uri` leaves unescaped: every printable US-ASCII character,
/// code points 32 to 126 inclusive.
const HTML_URI_ALLOWED: AsciiSet = AsciiSet::empty().add_range(b' ', b'~');

/// [`fn:encode-for-uri`](https://www.w3.org/TR/xpath-functions-31/#func-encode-for-uri): escapes
/// everything that RFC 3986 does not call *unreserved*.
///
/// # Examples
///
/// ```
/// use oxixml_fn::strings::encode_for_uri;
///
/// assert_eq!(
///     encode_for_uri("http://www.example.com/00/Weather/CA/Los%20Angeles#ocean"),
///     "http%3A%2F%2Fwww.example.com%2F00%2FWeather%2FCA%2FLos%2520Angeles%23ocean"
/// );
/// assert_eq!(encode_for_uri("~bébé"), "~b%C3%A9b%C3%A9");
/// assert_eq!(encode_for_uri("100% organic"), "100%25%20organic");
/// ```
#[must_use]
pub fn encode_for_uri(value: &str) -> String {
    percent_encode_str(value, &UNRESERVED)
}

/// [`fn:iri-to-uri`](https://www.w3.org/TR/xpath-functions-31/#func-iri-to-uri): escapes the
/// characters that are legal in an IRI but not in a URI, leaving `%` alone so that the function
/// is idempotent.
///
/// # Examples
///
/// ```
/// use oxixml_fn::strings::iri_to_uri;
///
/// assert_eq!(
///     iri_to_uri("http://www.example.com/00/Weather/CA/Los%20Angeles#ocean"),
///     "http://www.example.com/00/Weather/CA/Los%20Angeles#ocean"
/// );
/// assert_eq!(
///     iri_to_uri("http://www.example.com/~bébé"),
///     "http://www.example.com/~b%C3%A9b%C3%A9"
/// );
/// // Idempotent, but not invertible: a space and its escape converge.
/// assert_eq!(iri_to_uri("My Documents"), iri_to_uri("My%20Documents"));
/// ```
#[must_use]
pub fn iri_to_uri(value: &str) -> String {
    percent_encode_str(value, &IRI_ALLOWED)
}

/// [`fn:escape-html-uri`](https://www.w3.org/TR/xpath-functions-31/#func-escape-html-uri):
/// escapes everything outside printable US-ASCII, which is how an HTML user agent treats a URI
/// attribute value.
///
/// # Examples
///
/// ```
/// use oxixml_fn::strings::escape_html_uri;
///
/// assert_eq!(
///     escape_html_uri("http://www.example.com/00/Weather/CA/Los Angeles#ocean"),
///     "http://www.example.com/00/Weather/CA/Los Angeles#ocean"
/// );
/// assert_eq!(
///     escape_html_uri("http://www.example.com/~bébé"),
///     "http://www.example.com/~b%C3%A9b%C3%A9"
/// );
/// ```
#[must_use]
pub fn escape_html_uri(value: &str) -> String {
    percent_encode_str(value, &HTML_URI_ALLOWED)
}

/// [`fn:resolve-uri`](https://www.w3.org/TR/xpath-functions-31/#func-resolve-uri), the
/// two-argument form: resolves `relative` against `base`.
///
/// # Errors
///
/// [`FnErrorCode::Forg0002`] when either argument is not a valid URI reference, and when `base`
/// is not absolute.
///
/// # Examples
///
/// ```
/// use oxixml_fn::strings::resolve_uri_against;
///
/// assert_eq!(
///     resolve_uri_against("dir/file.txt", "http://example.com/base/page.html")?,
///     "http://example.com/base/dir/file.txt"
/// );
/// // An absolute reference is returned unchanged.
/// assert_eq!(
///     resolve_uri_against("http://other.example/", "http://example.com/")?,
///     "http://other.example/"
/// );
/// # Ok::<(), oxixml_fn::FnError>(())
/// ```
pub fn resolve_uri_against(relative: &str, base: &str) -> FnResult<String> {
    // F&O 3.1 §8.1 rule 3: an absolute `$href` is returned unchanged, whatever `$base` is —
    // including the zero-length string, which is not a base URI at all.
    if is_absolute_uri(relative) {
        return Ok(relative.to_owned());
    }
    // A base URI with a fragment identifier is not a valid absolute URI (rule 4).
    if base.contains('#') {
        return Err(FnError::new(
            FnErrorCode::Forg0002,
            format!("`{base}` is not a valid absolute URI: it has a fragment identifier"),
        ));
    }
    let base = Iri::parse(base).map_err(|error| {
        FnError::new(
            FnErrorCode::Forg0002,
            format!("`{base}` is not a valid absolute URI: {error}"),
        )
    })?;
    let reference = IriRef::parse(relative).map_err(|error| {
        FnError::new(
            FnErrorCode::Forg0002,
            format!("`{relative}` is not a valid URI reference: {error}"),
        )
    })?;
    base.resolve(&reference)
        .map(Iri::into_inner)
        .map_err(|error| {
            FnError::new(
                FnErrorCode::Forg0009,
                format!("`{relative}` could not be resolved: {error}"),
            )
        })
}

/// [`fn:resolve-uri`](https://www.w3.org/TR/xpath-functions-31/#func-resolve-uri), the
/// one-argument form: resolves against the context's static base URI.
///
/// # Errors
///
/// [`FnErrorCode::Fons0005`] when the context has no static base URI, plus whatever
/// [`resolve_uri_against`] raises.
pub fn resolve_uri(context: &FnContext, relative: &str) -> FnResult<String> {
    // An absolute `$href` needs no base at all, so the absence of a static base URI is only an
    // error for a genuinely relative reference (F&O 3.1 §8.1 rules 3 and 4).
    if is_absolute_uri(relative) {
        return Ok(relative.to_owned());
    }
    let base = context.require_static_base_uri()?;
    resolve_uri_against(relative, base)
}

/// Whether `value` is an absolute URI, which is what `fn:resolve-uri` checks before it decides
/// to short-circuit.
#[must_use]
pub fn is_absolute_uri(value: &str) -> bool {
    Iri::parse(value).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_for_uri_examples_from_the_specification() {
        assert_eq!(
            encode_for_uri("http://www.example.com/00/Weather/CA/Los%20Angeles#ocean"),
            "http%3A%2F%2Fwww.example.com%2F00%2FWeather%2FCA%2FLos%2520Angeles%23ocean"
        );
        assert_eq!(encode_for_uri("~bébé"), "~b%C3%A9b%C3%A9");
        assert_eq!(encode_for_uri("100% organic"), "100%25%20organic");
        assert_eq!(encode_for_uri(""), "");
    }

    #[test]
    fn iri_to_uri_examples_from_the_specification() {
        assert_eq!(
            iri_to_uri("http://www.example.com/00/Weather/CA/Los%20Angeles#ocean"),
            "http://www.example.com/00/Weather/CA/Los%20Angeles#ocean"
        );
        assert_eq!(
            iri_to_uri("http://www.example.com/~bébé"),
            "http://www.example.com/~b%C3%A9b%C3%A9"
        );
        assert_eq!(iri_to_uri("My Documents"), "My%20Documents");
        assert_eq!(
            iri_to_uri("a<b>c\"d{e}f|g\\h^i`j"),
            "a%3Cb%3Ec%22d%7Be%7Df%7Cg%5Ch%5Ei%60j"
        );
    }

    #[test]
    fn escape_html_uri_examples_from_the_specification() {
        assert_eq!(
            escape_html_uri("http://www.example.com/00/Weather/CA/Los Angeles#ocean"),
            "http://www.example.com/00/Weather/CA/Los Angeles#ocean"
        );
        assert_eq!(
            escape_html_uri(
                "javascript:if (navigator.browserLanguage == 'fr') window.open('http://www.example.com/~bébé');"
            ),
            "javascript:if (navigator.browserLanguage == 'fr') window.open('http://www.example.com/~b%C3%A9b%C3%A9');"
        );
    }

    #[test]
    fn resolve_uri_resolves_against_a_base() -> FnResult<()> {
        assert_eq!(
            resolve_uri_against("dir/file.txt", "http://example.com/base/page.html")?,
            "http://example.com/base/dir/file.txt"
        );
        assert_eq!(
            resolve_uri_against("../up.txt", "http://example.com/a/b/c.html")?,
            "http://example.com/a/up.txt"
        );
        assert_eq!(
            resolve_uri_against("http://other.example/", "http://example.com/")?,
            "http://other.example/"
        );
        Ok(())
    }

    #[test]
    fn a_relative_base_raises_forg0002() {
        let error = resolve_uri_against("a", "not a uri at all").expect_err("relative base");
        assert!(error.is(FnErrorCode::Forg0002));
    }

    #[test]
    fn the_one_argument_form_needs_a_static_base_uri() -> FnResult<()> {
        let bare = FnContext::new();
        assert!(
            resolve_uri(&bare, "a.txt")
                .expect_err("no base URI")
                .is(FnErrorCode::Fons0005)
        );

        let based = FnContext::new().with_static_base_uri("http://example.com/dir/");
        assert_eq!(
            resolve_uri(&based, "a.txt")?,
            "http://example.com/dir/a.txt"
        );
        Ok(())
    }

    #[test]
    fn absolute_uris_are_recognized() {
        assert!(is_absolute_uri("http://example.com/"));
        assert!(!is_absolute_uri("relative/path"));
    }
}
