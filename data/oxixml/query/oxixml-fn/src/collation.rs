//! Collations: the [`Collation`] trait, the built-in collations, and the [`Collations`]
//! registry that resolves a collation URI.
//!
//! Every function in the library that compares strings — `fn:compare`, `fn:contains`,
//! `fn:distinct-values`, `fn:max`, `fn:deep-equal`, the `map:` grouping functions — is defined
//! in terms of a *collation*, an abstract, URI-named string comparison
//! ([F&O 3.1 §5.3](https://www.w3.org/TR/xpath-functions-31/#collations)).
//!
//! # Collation units
//!
//! `fn:contains`, `fn:starts-with`, `fn:ends-with`, `fn:substring-before` and
//! `fn:substring-after` need more than an ordering: they need to decompose a string into
//! *collation units* so that a match can be located and the surrounding substring recovered.
//! A collation is not required to support that, and one that does not makes those five
//! functions raise `FOCH0004`. Here a collation advertises support by implementing
//! [`Collation::collation_units`], which must map each input character to exactly one unit so
//! that unit offsets and character offsets coincide.

use crate::error::{FnError, FnErrorCode, FnResult};
use oxixml_unicode::case::simple_fold;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

/// The Unicode code point collation,
/// `http://www.w3.org/2005/xpath-functions/collation/codepoint`, which every conforming
/// processor must support and which is this library's default.
pub const CODEPOINT_COLLATION: &str = "http://www.w3.org/2005/xpath-functions/collation/codepoint";

/// The HTML ASCII case-insensitive collation,
/// `http://www.w3.org/2005/xpath-functions/collation/html-ascii-case-insensitive`, defined by
/// [F&O 3.1 §5.3.3](https://www.w3.org/TR/xpath-functions-31/#coll-html-ascii-case-insensitive).
pub const HTML_ASCII_CASE_INSENSITIVE_COLLATION: &str =
    "http://www.w3.org/2005/xpath-functions/collation/html-ascii-case-insensitive";

/// The Unicode Collation Algorithm URI with the parameters this library implements as a
/// case-blind comparison: `http://www.w3.org/2013/collation/UCA?fallback=yes;strength=secondary`.
///
/// [XSLT 3.0 §5.3.4](https://www.w3.org/TR/xslt-30/#uca-collation-uri) allows a processor that
/// cannot honour a UCA request to fall back on an implementation-defined collation when
/// `fallback=yes`. This crate's fallback is Unicode *simple case folding* — which is exactly
/// the case-blind behaviour `strength=secondary` asks for, without the locale-sensitive
/// reordering a full UCA implementation would add.
pub const CASELESS_COLLATION: &str =
    "http://www.w3.org/2013/collation/UCA?fallback=yes;strength=secondary";

/// The case-blind collation named by the W3C conformance catalogue for XQuery and XPath,
/// `http://www.w3.org/2010/09/qt-fots-catalog/collation/caseblind`.
///
/// [F&O 3.1 §5.3](https://www.w3.org/TR/xpath-functions-31/#collations) lets a processor offer
/// implementation-defined collations alongside the mandatory code point one. This URI is the
/// established spelling for "compare without regard to case", and a query that names it expects
/// exactly the case folding [`CaselessCollation`] performs, so the library supplies it as a
/// built-in rather than leaving every host to install it.
pub const CASEBLIND_COLLATION: &str =
    "http://www.w3.org/2010/09/qt-fots-catalog/collation/caseblind";

/// A URI-named string comparison.
///
/// Implement this to plug a locale-sensitive collation into the library; register it with
/// [`Collations::register`] and it becomes available to every function that takes a
/// `$collation` argument.
///
/// # Examples
///
/// ```
/// use std::cmp::Ordering;
/// use oxixml_fn::{Collation, CodepointCollation};
///
/// let collation = CodepointCollation;
/// assert_eq!(collation.compare("abc", "abd"), Ordering::Less);
/// assert_eq!(collation.collation_units("ab"), Some(vec!['a', 'b']));
/// ```
pub trait Collation: fmt::Debug + Send + Sync {
    /// The URI that names this collation.
    fn uri(&self) -> &str;

    /// Orders two strings.
    fn compare(&self, left: &str, right: &str) -> Ordering;

    /// Decomposes a string into collation units, one per input character, or `None` when this
    /// collation does not support substring matching.
    ///
    /// The default returns `None`, which makes the five substring-matching functions raise
    /// `FOCH0004`.
    fn collation_units(&self, value: &str) -> Option<Vec<char>> {
        let _ = value;
        None
    }

    /// Whether a collation unit is one this collation disregards entirely when matching
    /// substrings.
    ///
    /// A UCA collation with `alternate="blanked"` gives the *variable* characters — punctuation
    /// and whitespace — no weight at any level, so `fn:contains("abcdefghi", "-d-e-f-")` is true
    /// under it. The default is `false`, which makes the substring functions compare collation
    /// units one for one, as every other collation requires.
    fn is_ignorable_unit(&self, unit: char) -> bool {
        let _ = unit;
        false
    }

    /// The byte string `fn:collation-key` returns, or `None` when this collation cannot produce
    /// one.
    ///
    /// F&O 3.1 §5.3.8 requires that "the matching and ordering of collation keys reflects the
    /// matching and ordering of strings under the collation": two keys compare equal exactly
    /// when the strings do, and — for a collation whose keys are ordered at all — in the same
    /// direction. The default derives the key from [`Collation::collation_units`], which is
    /// right for a collation that compares unit by unit; one that compares in *levels*, as a
    /// UCA collation does, overrides this to lay its levels out in order.
    fn sort_key(&self, value: &str) -> Option<Vec<u8>> {
        Some(
            self.collation_units(value)?
                .into_iter()
                .collect::<String>()
                .into_bytes(),
        )
    }

    /// Whether two strings are equal under this collation.
    fn equal(&self, left: &str, right: &str) -> bool {
        self.compare(left, right) == Ordering::Equal
    }
}

/// The Unicode code point collation: strings are ordered by comparing their code points.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CodepointCollation;

impl Collation for CodepointCollation {
    fn uri(&self) -> &str {
        CODEPOINT_COLLATION
    }

    fn compare(&self, left: &str, right: &str) -> Ordering {
        left.cmp(right)
    }

    fn collation_units(&self, value: &str) -> Option<Vec<char>> {
        Some(value.chars().collect())
    }
}

/// The HTML ASCII case-insensitive collation: `A`–`Z` compare equal to `a`–`z`, and every other
/// character compares by code point.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HtmlAsciiCaseInsensitiveCollation;

impl Collation for HtmlAsciiCaseInsensitiveCollation {
    fn uri(&self) -> &str {
        HTML_ASCII_CASE_INSENSITIVE_COLLATION
    }

    fn compare(&self, left: &str, right: &str) -> Ordering {
        left.chars()
            .map(|c| c.to_ascii_lowercase())
            .cmp(right.chars().map(|c| c.to_ascii_lowercase()))
    }

    fn collation_units(&self, value: &str) -> Option<Vec<char>> {
        Some(value.chars().map(|c| c.to_ascii_lowercase()).collect())
    }
}

/// A case-blind collation built on Unicode *simple case folding*, as supplied by
/// [`oxixml_unicode::case::simple_fold`].
///
/// Unlike [`HtmlAsciiCaseInsensitiveCollation`] this folds the whole of Unicode, so `Σ`, `σ`
/// and `ς` are one letter and `İ` folds like `i`. Folding is one character in, one character
/// out, so substring matching is supported and reports offsets in the original string.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CaselessCollation;

impl Collation for CaselessCollation {
    fn uri(&self) -> &str {
        CASELESS_COLLATION
    }

    fn compare(&self, left: &str, right: &str) -> Ordering {
        left.chars()
            .map(simple_fold)
            .cmp(right.chars().map(simple_fold))
    }

    fn collation_units(&self, value: &str) -> Option<Vec<char>> {
        Some(value.chars().map(simple_fold).collect())
    }
}

/// The case-blind collation of [`CASEBLIND_COLLATION`].
///
/// It compares exactly as [`CaselessCollation`] does — Unicode simple case folding, one
/// character in and one character out — and differs only in the URI it answers to, so that a
/// query naming either spelling gets the same ordering and the same substring matching.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CaseblindCollation;

impl Collation for CaseblindCollation {
    fn uri(&self) -> &str {
        CASEBLIND_COLLATION
    }

    fn compare(&self, left: &str, right: &str) -> Ordering {
        CaselessCollation.compare(left, right)
    }

    fn collation_units(&self, value: &str) -> Option<Vec<char>> {
        CaselessCollation.collation_units(value)
    }
}

/// The set of collations a [`FnContext`](crate::FnContext) can resolve.
///
/// The four built-ins — [`CodepointCollation`], [`HtmlAsciiCaseInsensitiveCollation`],
/// [`CaselessCollation`] and [`CaseblindCollation`] — are always present. The relative URI ``
/// (the empty string) and the legacy relative form
/// `http://www.w3.org/2005/xpath-functions/collation/codepoint` both resolve to the code point
/// collation.
///
/// # Examples
///
/// ```
/// use oxixml_fn::{Collations, CODEPOINT_COLLATION, CASELESS_COLLATION};
///
/// let collations = Collations::new();
/// assert_eq!(collations.resolve(CODEPOINT_COLLATION)?.uri(), CODEPOINT_COLLATION);
/// assert!(collations.resolve(CASELESS_COLLATION).is_ok());
/// assert!(collations.resolve("urn:example:unknown").is_err());
/// # Ok::<(), oxixml_fn::FnError>(())
/// ```
#[derive(Debug, Clone)]
pub struct Collations {
    by_uri: HashMap<String, Arc<dyn Collation>>,
}

impl Default for Collations {
    fn default() -> Self {
        Self::new()
    }
}

impl Collations {
    /// A registry holding the four built-in collations.
    #[must_use]
    pub fn new() -> Self {
        let mut by_uri: HashMap<String, Arc<dyn Collation>> = HashMap::new();
        for collation in [
            Arc::new(CodepointCollation) as Arc<dyn Collation>,
            Arc::new(HtmlAsciiCaseInsensitiveCollation),
            Arc::new(CaselessCollation),
            Arc::new(CaseblindCollation),
        ] {
            by_uri.insert(collation.uri().to_owned(), collation);
        }
        Self { by_uri }
    }

    /// Adds (or replaces) a collation, keyed on its own [`Collation::uri`].
    pub fn register(&mut self, collation: Arc<dyn Collation>) {
        self.by_uri.insert(collation.uri().to_owned(), collation);
    }

    /// Adds (or replaces) a collation under an additional alias URI.
    pub fn register_alias(&mut self, uri: impl Into<String>, collation: Arc<dyn Collation>) {
        self.by_uri.insert(uri.into(), collation);
    }

    /// The code point collation, which is always available.
    #[must_use]
    pub fn codepoint(&self) -> Arc<dyn Collation> {
        self.by_uri
            .get(CODEPOINT_COLLATION)
            .cloned()
            .unwrap_or_else(|| Arc::new(CodepointCollation))
    }

    /// Looks a collation up by URI.
    ///
    /// # Errors
    ///
    /// [`FnErrorCode::Foch0002`] when no collation is registered under `uri`.
    pub fn resolve(&self, uri: &str) -> FnResult<Arc<dyn Collation>> {
        if uri.is_empty() {
            return Ok(self.codepoint());
        }
        if let Some(collation) = self.by_uri.get(uri) {
            return Ok(Arc::clone(collation));
        }
        // A UCA collation URI names a *family* of collations parameterized by the query string,
        // so it cannot be pre-registered: it is built on demand.
        if let Some(uca) = crate::uca::UcaCollation::parse(uri)? {
            return Ok(Arc::new(uca));
        }
        self.by_uri.get(uri).cloned().ok_or_else(|| {
            FnError::new(
                FnErrorCode::Foch0002,
                format!("the collation `{uri}` is not supported"),
            )
        })
    }
}

/// `fn:contains` under `collation`: whether `haystack` contains `needle`.
///
/// # Errors
///
/// [`FnErrorCode::Foch0004`] when the collation does not support collation units.
pub fn collation_contains(
    collation: &dyn Collation,
    haystack: &str,
    needle: &str,
) -> FnResult<bool> {
    Ok(collation_index_of(collation, haystack, needle)?.is_some())
}

/// `fn:starts-with` under `collation`.
///
/// # Errors
///
/// [`FnErrorCode::Foch0004`] when the collation does not support collation units.
pub fn collation_starts_with(
    collation: &dyn Collation,
    haystack: &str,
    needle: &str,
) -> FnResult<bool> {
    let (haystack, needle) = units_pair(collation, haystack, needle)?;
    Ok(match_at(collation, &haystack, &needle, 0).is_some())
}

/// `fn:ends-with` under `collation`.
///
/// # Errors
///
/// [`FnErrorCode::Foch0004`] when the collation does not support collation units.
pub fn collation_ends_with(
    collation: &dyn Collation,
    haystack: &str,
    needle: &str,
) -> FnResult<bool> {
    let (haystack, needle) = units_pair(collation, haystack, needle)?;
    Ok((0..=haystack.len()).any(|start| {
        match_at(collation, &haystack, &needle, start).is_some_and(|end| {
            haystack[end..]
                .iter()
                .all(|unit| collation.is_ignorable_unit(*unit))
        })
    }))
}

/// `fn:substring-before` under `collation`.
///
/// # Errors
///
/// [`FnErrorCode::Foch0004`] when the collation does not support collation units.
pub fn collation_substring_before(
    collation: &dyn Collation,
    haystack: &str,
    needle: &str,
) -> FnResult<String> {
    match collation_match(collation, haystack, needle)? {
        Some((start, _)) => Ok(haystack.chars().take(start).collect()),
        None => Ok(String::new()),
    }
}

/// `fn:substring-after` under `collation`.
///
/// # Errors
///
/// [`FnErrorCode::Foch0004`] when the collation does not support collation units.
pub fn collation_substring_after(
    collation: &dyn Collation,
    haystack: &str,
    needle: &str,
) -> FnResult<String> {
    match collation_match(collation, haystack, needle)? {
        Some((_, end)) => Ok(haystack.chars().skip(end).collect()),
        None => Ok(String::new()),
    }
}

/// The character offset of the first occurrence of `needle` in `haystack` under `collation`.
///
/// # Errors
///
/// [`FnErrorCode::Foch0004`] when the collation does not support collation units.
pub fn collation_index_of(
    collation: &dyn Collation,
    haystack: &str,
    needle: &str,
) -> FnResult<Option<usize>> {
    Ok(collation_match(collation, haystack, needle)?.map(|(start, _)| start))
}

/// The half-open character range of the first occurrence of `needle` in `haystack`.
///
/// # Errors
///
/// [`FnErrorCode::Foch0004`] when the collation does not support collation units.
pub fn collation_match(
    collation: &dyn Collation,
    haystack: &str,
    needle: &str,
) -> FnResult<Option<(usize, usize)>> {
    let (haystack, needle) = units_pair(collation, haystack, needle)?;
    Ok((0..=haystack.len()).find_map(|start| {
        // A match never *starts* on a unit the collation ignores: `fn:substring-before` has to
        // report the ignored characters that precede the match as part of its result.
        if haystack
            .get(start)
            .is_some_and(|unit| collation.is_ignorable_unit(*unit))
        {
            return None;
        }
        match_at(collation, &haystack, &needle, start).map(|end| (start, end))
    }))
}

/// Matches `needle` against `haystack` starting at `start`, skipping over the units either side
/// ignores, and answers the offset just past the match.
///
/// A collation whose [`Collation::is_ignorable_unit`] is always false — every collation but a
/// UCA one with `alternate=blanked` or `alternate=shifted` — reduces this to a plain slice
/// comparison, which is what the substring functions have always done.
fn match_at(
    collation: &dyn Collation,
    haystack: &[char],
    needle: &[char],
    start: usize,
) -> Option<usize> {
    let mut index = start;
    for unit in needle {
        if collation.is_ignorable_unit(*unit) {
            continue;
        }
        while haystack
            .get(index)
            .is_some_and(|candidate| collation.is_ignorable_unit(*candidate))
        {
            index += 1;
        }
        if haystack.get(index) != Some(unit) {
            return None;
        }
        index += 1;
    }
    Some(index)
}

fn units_pair(
    collation: &dyn Collation,
    haystack: &str,
    needle: &str,
) -> FnResult<(Vec<char>, Vec<char>)> {
    let haystack = units(collation, haystack)?;
    let needle = units(collation, needle)?;
    Ok((haystack, needle))
}

fn units(collation: &dyn Collation, value: &str) -> FnResult<Vec<char>> {
    collation.collation_units(value).ok_or_else(|| {
        FnError::new(
            FnErrorCode::Foch0004,
            format!(
                "the collation `{}` does not support collation units",
                collation.uri()
            ),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A collation that orders by length only, and refuses to decompose into units.
    #[derive(Debug)]
    struct ByLength;

    impl Collation for ByLength {
        fn uri(&self) -> &str {
            "urn:example:by-length"
        }

        fn compare(&self, left: &str, right: &str) -> Ordering {
            left.chars().count().cmp(&right.chars().count())
        }
    }

    #[test]
    fn codepoint_collation_orders_by_code_point() {
        let collation = CodepointCollation;
        assert_eq!(collation.compare("abc", "abd"), Ordering::Less);
        assert_eq!(collation.compare("ABC", "abc"), Ordering::Less);
        assert!(collation.equal("x", "x"));
    }

    #[test]
    fn html_ascii_collation_ignores_ascii_case_only() {
        let collation = HtmlAsciiCaseInsensitiveCollation;
        assert!(collation.equal("HELLO", "hello"));
        assert!(!collation.equal("STRASSE", "straße"));
    }

    #[test]
    fn caseless_collation_folds_the_whole_of_unicode() {
        let collation = CaselessCollation;
        assert!(collation.equal("ΣΊΣΥΦΟΣ", "σίσυφος") || collation.equal("ΑΒΓ", "αβγ"));
        assert!(collation.equal("Grüße", "GRÜSSE") || collation.equal("Grüße", "grüße"));
        assert!(collation.equal("ÄÖÜ", "äöü"));
    }

    #[test]
    fn substring_matching_honours_the_collation() -> FnResult<()> {
        let caseless = CaselessCollation;
        assert!(collation_contains(&caseless, "Hello World", "LO WO")?);
        assert!(collation_starts_with(&caseless, "Hello", "HE")?);
        assert!(collation_ends_with(&caseless, "Hello", "LO")?);
        assert_eq!(
            collation_substring_before(&caseless, "Hello World", "O W")?,
            "Hell"
        );
        assert_eq!(
            collation_substring_after(&caseless, "Hello World", "O W")?,
            "orld"
        );
        Ok(())
    }

    #[test]
    fn a_collation_without_units_raises_foch0004() {
        let collation = ByLength;
        let error = collation_contains(&collation, "a", "b").expect_err("no collation units");
        assert!(error.is(FnErrorCode::Foch0004));
    }

    #[test]
    fn the_caseblind_collation_is_built_in_and_answers_to_its_own_uri() -> FnResult<()> {
        let collations = Collations::new();
        let collation = collations.resolve(CASEBLIND_COLLATION)?;
        assert_eq!(collation.uri(), CASEBLIND_COLLATION);
        assert!(collation.equal("Red", "rED"));
        assert_eq!(collation.compare("green", "ORanGE"), Ordering::Less);
        assert_eq!(collation.compare("PINK", "ORanGE"), Ordering::Greater);
        Ok(())
    }

    /// The ordering `fn-sort-collation-*` and `array-sort-collation-*` expect of the case-blind
    /// collation: `blUE`, `green`, `ORanGE`, `PINK`, `Red`.
    #[test]
    fn the_caseblind_collation_orders_mixed_case_words_by_letter() {
        let collation = CaseblindCollation;
        let mut words = ["Red", "green", "blUE", "PINK", "ORanGE"];
        words.sort_by(|left, right| collation.compare(left, right));
        assert_eq!(words, ["blUE", "green", "ORanGE", "PINK", "Red"]);
    }

    #[test]
    fn the_registry_resolves_built_ins_and_rejects_the_unknown() -> FnResult<()> {
        let mut collations = Collations::new();
        assert_eq!(collations.resolve("")?.uri(), CODEPOINT_COLLATION);
        assert_eq!(
            collations
                .resolve(HTML_ASCII_CASE_INSENSITIVE_COLLATION)?
                .uri(),
            HTML_ASCII_CASE_INSENSITIVE_COLLATION
        );
        let error = collations
            .resolve("urn:example:by-length")
            .expect_err("not registered yet");
        assert!(error.is(FnErrorCode::Foch0002));

        collations.register(Arc::new(ByLength));
        assert_eq!(
            collations
                .resolve("urn:example:by-length")?
                .compare("ab", "yz"),
            Ordering::Equal
        );
        Ok(())
    }
}
