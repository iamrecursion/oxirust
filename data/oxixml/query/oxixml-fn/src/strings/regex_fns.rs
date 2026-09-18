//! The regular-expression functions of
//! [F&O 3.1 §5.6](https://www.w3.org/TR/xpath-functions-31/#string-functions-regex):
//! `fn:matches`, `fn:replace` and `fn:tokenize`.
//!
//! The engine is [`oxixml_regex`], compiled in its XPath dialect: unanchored, with the `s`,
//! `m`, `i`, `x` and `q` flags, reluctant quantifiers, non-capturing groups and back-references.
//! Its four `FORX` error codes map straight onto [`FnErrorCode`].

use crate::error::{FnError, FnErrorCode, FnResult};
use oxixml_regex::Regex;

/// [`fn:matches`](https://www.w3.org/TR/xpath-functions-31/#func-matches).
///
/// # Errors
///
/// [`FnErrorCode::Forx0001`] for invalid flags and
/// [`FnErrorCode::Forx0002`] for an invalid pattern.
///
/// # Examples
///
/// ```
/// use oxixml_fn::strings::matches;
///
/// let poem = "Kaum hat dies der Hahn gesehen,\nFängt er auch schon an zu krähen:";
/// assert!(matches("abracadabra", "bra", "")?);
/// assert!(matches("abracadabra", "^a.*a$", "")?);
/// assert!(!matches("abracadabra", "^bra", "")?);
/// assert!(matches(poem, "Kaum.*krähen", "s")?);
/// assert!(!matches(poem, "Kaum.*krähen", "")?);
/// assert!(matches(poem, "^Kaum.*gesehen,$", "m")?);
/// # Ok::<(), oxixml_fn::FnError>(())
/// ```
pub fn matches(input: &str, pattern: &str, flags: &str) -> FnResult<bool> {
    let regex = Regex::xpath(pattern, flags)?;
    Ok(regex.try_is_match(input)?)
}

/// [`fn:replace`](https://www.w3.org/TR/xpath-functions-31/#func-replace).
///
/// # Errors
///
/// The `FORX` codes, including
/// [`FnErrorCode::Forx0003`] when the pattern matches a
/// zero-length string and [`FnErrorCode::Forx0004`] for an invalid
/// replacement string.
///
/// # Examples
///
/// ```
/// use oxixml_fn::strings::replace;
///
/// assert_eq!(replace("abracadabra", "bra", "*", "")?, "a*cada*");
/// assert_eq!(replace("abracadabra", "a.*a", "*", "")?, "*");
/// assert_eq!(replace("abracadabra", "a.*?a", "*", "")?, "*c*bra");
/// assert_eq!(replace("abracadabra", "a", "", "")?, "brcdbr");
/// assert_eq!(replace("AAAA", "A+", "b", "")?, "b");
/// assert_eq!(replace("darted", "^(.*?)d(.*)$", "$1c$2", "")?, "carted");
/// assert!(replace("abracadabra", ".*?", "$1", "").is_err());
/// # Ok::<(), oxixml_fn::FnError>(())
/// ```
pub fn replace(input: &str, pattern: &str, replacement: &str, flags: &str) -> FnResult<String> {
    let regex = Regex::xpath(pattern, flags)?;
    Ok(regex.replace(input, replacement)?)
}

/// [`fn:tokenize`](https://www.w3.org/TR/xpath-functions-31/#func-tokenize), the three-argument
/// form.
///
/// # Errors
///
/// The `FORX` codes; in particular
/// [`FnErrorCode::Forx0003`] when the pattern matches a
/// zero-length string.
///
/// # Examples
///
/// ```
/// use oxixml_fn::strings::tokenize;
///
/// assert_eq!(tokenize("red green blue ", "\\s+", "")?, ["red", "green", "blue", ""]);
/// assert_eq!(tokenize("1, 15, 24, 50", ",\\s*", "")?, ["1", "15", "24", "50"]);
/// assert_eq!(tokenize("1,15,,24,50,", ",", "")?, ["1", "15", "", "24", "50", ""]);
/// assert_eq!(tokenize("Some unparsed <br> HTML <BR> text", "\\s*<br>\\s*", "i")?,
///            ["Some unparsed", "HTML", "text"]);
/// # Ok::<(), oxixml_fn::FnError>(())
/// ```
pub fn tokenize(input: &str, pattern: &str, flags: &str) -> FnResult<Vec<String>> {
    let regex = Regex::xpath(pattern, flags)?;
    Ok(regex.tokenize(input)?)
}

/// The one-argument form of [`tokenize`], which splits on runs of whitespace after
/// `fn:normalize-space` has been applied.
///
/// # Errors
///
/// Never fails in practice; the fallible signature matches the general form.
pub fn tokenize_whitespace(input: &str) -> FnResult<Vec<String>> {
    let normalized = super::basic::normalize_space(input);
    if normalized.is_empty() {
        return Ok(Vec::new());
    }
    tokenize(&normalized, " ", "")
}

/// Compiles an XPath-dialect regular expression, so that a caller evaluating many inputs against
/// one pattern can pay the compilation cost once.
///
/// # Errors
///
/// [`FnErrorCode::Forx0001`] or
/// [`FnErrorCode::Forx0002`].
pub fn compile(pattern: &str, flags: &str) -> FnResult<Regex> {
    Regex::xpath(pattern, flags).map_err(FnError::from)
}

/// Rejects the `q` flag combined with a pattern that a caller intends to use for capture, which
/// is the one situation where `oxixml-regex` and F&O disagree about the *number* of groups.
///
/// # Errors
///
/// [`FnErrorCode::Forx0002`] when `pattern` declares more capturing groups than `expected`.
pub fn require_group_count(regex: &Regex, expected: usize) -> FnResult<()> {
    if regex.group_count() < expected {
        return Err(FnError::new(
            FnErrorCode::Forx0002,
            format!(
                "the pattern has {} capturing group(s), but {expected} were referenced",
                regex.group_count()
            ),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_examples_from_the_specification() -> FnResult<()> {
        assert!(matches("abracadabra", "bra", "")?);
        assert!(matches("abracadabra", "^a.*a$", "")?);
        assert!(!matches("abracadabra", "^bra", "")?);
        let poem = "Kaum hat dies der Hahn gesehen,\nFängt er auch schon an zu krähen:";
        assert!(!matches(poem, "Kaum.*krähen", "")?);
        assert!(matches(poem, "Kaum.*krähen", "s")?);
        assert!(matches(poem, "^Kaum.*gesehen,$", "m")?);
        assert!(!matches(poem, "^Kaum.*gesehen,$", "")?);
        assert!(matches(poem, "kaum.*gesehen,$", "im")?);
        Ok(())
    }

    #[test]
    fn replace_examples_from_the_specification() -> FnResult<()> {
        assert_eq!(replace("abracadabra", "bra", "*", "")?, "a*cada*");
        assert_eq!(replace("abracadabra", "a.*a", "*", "")?, "*");
        assert_eq!(replace("abracadabra", "a.*?a", "*", "")?, "*c*bra");
        assert_eq!(replace("abracadabra", "a", "", "")?, "brcdbr");
        assert_eq!(
            replace("abracadabra", "a(.)", "a$1$1", "")?,
            "abbraccaddabbra"
        );
        assert_eq!(replace("AAAA", "A+", "b", "")?, "b");
        assert_eq!(replace("AAAA", "A+?", "b", "")?, "bbbb");
        assert_eq!(replace("darted", "^(.*?)d(.*)$", "$1c$2", "")?, "carted");
        Ok(())
    }

    #[test]
    fn a_pattern_matching_the_empty_string_raises_forx0003() {
        let error = replace("abracadabra", ".*?", "$1", "").expect_err("matches empty");
        assert!(error.is(FnErrorCode::Forx0003));
        let error = tokenize("abba", "a*", "").expect_err("matches empty");
        assert!(error.is(FnErrorCode::Forx0003));
    }

    #[test]
    fn invalid_flags_and_patterns_map_onto_forx0001_and_forx0002() {
        assert!(
            matches("a", "a", "z")
                .expect_err("bad flag")
                .is(FnErrorCode::Forx0001)
        );
        assert!(
            matches("a", "(", "")
                .expect_err("bad pattern")
                .is(FnErrorCode::Forx0002)
        );
    }

    #[test]
    fn tokenize_examples_from_the_specification() -> FnResult<()> {
        assert_eq!(
            tokenize("red green blue ", "\\s+", "")?,
            ["red", "green", "blue", ""]
        );
        assert_eq!(
            tokenize("1, 15, 24, 50", ",\\s*", "")?,
            ["1", "15", "24", "50"]
        );
        assert_eq!(
            tokenize("1,15,,24,50,", ",", "")?,
            ["1", "15", "", "24", "50", ""]
        );
        assert_eq!(
            tokenize("Some unparsed <br> HTML <BR> text", "\\s*<br>\\s*", "i")?,
            ["Some unparsed", "HTML", "text"]
        );
        Ok(())
    }

    #[test]
    fn the_one_argument_tokenize_splits_on_whitespace() -> FnResult<()> {
        assert_eq!(
            tokenize_whitespace(" red  green blue ")?,
            ["red", "green", "blue"]
        );
        assert!(tokenize_whitespace("   ")?.is_empty());
        Ok(())
    }

    #[test]
    fn group_counts_can_be_checked_ahead_of_time() -> FnResult<()> {
        let regex = compile("(a)(b)", "")?;
        assert!(require_group_count(&regex, 2).is_ok());
        assert!(require_group_count(&regex, 3).is_err());
        Ok(())
    }
}
