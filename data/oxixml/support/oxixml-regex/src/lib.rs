//! `oxixml-regex` — the regular-expression engine shared across the
//! [OxiXML](https://github.com/cool-japan/oxixml) workspace.
//!
//! Part of the COOLJAPAN Pure-Rust foundation for all XML and RDF processing.
//! This crate is `std`-only and depends only on
//! [`oxixml-unicode`](https://docs.rs/oxixml-unicode) for Unicode property data;
//! it contains **no external regex dependency** — the whole engine is
//! implemented here from the two W3C specifications.
//!
//! # Two dialects, one engine
//!
//! A [`Regex`] can be built in either of two flavours that a conforming XML
//! stack must implement:
//!
//! * [`Regex::xsd`] — the [XML Schema `pattern` facet][xsd] flavour. The pattern
//!   is *implicitly anchored to the whole string*, has no flags, no anchors, and
//!   no back-references, but supports character-class **range subtraction**
//!   (`[a-z-[aeiou]]`, nestable), the XML-name-aware escapes `\i` `\c`, and
//!   `\p{...}` category and `\p{Is...}` block escapes. It reads the pattern
//!   against XSD 1.1; [`Regex::xsd_versioned`] selects the stricter XSD 1.0
//!   grammar instead (see [`XsdVersion`]).
//! * [`Regex::xpath`] — the [XPath / XQuery `fn:matches`][fo] flavour. It adds
//!   the anchors `^` and `$`, the flags `s` `m` `i` `x` `q`, reluctant
//!   quantifiers (`a*?`), non-capturing groups (`(?:…)`), and back-references
//!   (`\1`…`\9`), and it is **not** implicitly anchored. F&O defines its
//!   character-class grammar by reference to XML Schema's, so this dialect is
//!   read against an [`XsdVersion`] too — [`DEFAULT_XSD_VERSION`] unless
//!   [`Regex::xpath_versioned`] says otherwise.
//!
//! Patterns without back-references compile to a linear-time Pike VM, so they
//! cannot suffer catastrophic back-tracking. Patterns *with* back-references run
//! on a back-tracking interpreter bounded by a step budget, which returns
//! [`RegexError::BudgetExceeded`] rather than hanging.
//!
//! # Per-match, per-group positions
//!
//! [`Regex::is_match`], [`Regex::replace`], and [`Regex::tokenize`] answer
//! *whether* and *how* an input matches, but not *where*. [`Regex::find_iter`]
//! (and the single-match convenience [`Regex::find`]) fill that gap: they
//! yield a [`Match`] per non-overlapping match, each carrying the byte
//! position of the whole match and of every capturing group — the primitive
//! an implementation of [`fn:analyze-string`][as] needs.
//!
//! [as]: https://www.w3.org/TR/xpath-functions-31/#func-analyze-string
//!
//! # Errors
//!
//! Failures map onto the four W3C error codes through [`RegexError::forx_code`]:
//! `FORX0001` (flags), `FORX0002` (pattern), `FORX0003` (a `replace` or
//! `tokenize` pattern that matches the empty string), and `FORX0004` (an invalid
//! replacement string).
//!
//! [xsd]: https://www.w3.org/TR/xmlschema-2/#regexs
//! [fo]: https://www.w3.org/TR/xpath-functions-31/#regex-syntax
//!
//! # Example
//!
//! ```
//! use oxixml_regex::Regex;
//!
//! // XSD: implicitly anchored to the whole string.
//! let postcode = Regex::xsd("[0-9]{5}(-[0-9]{4})?")?;
//! assert!(postcode.is_match("12345"));
//! assert!(postcode.is_match("12345-6789"));
//! assert!(!postcode.is_match("12345-678"));   // must match the whole string
//!
//! // XSD range subtraction: consonants only.
//! let consonant = Regex::xsd("[a-z-[aeiou]]+")?;
//! assert!(consonant.is_match("rhythm"));
//! assert!(!consonant.is_match("aeiou"));
//!
//! // XPath: unanchored search with the case-insensitive flag.
//! let word = Regex::xpath("hello", "i")?;
//! assert!(word.is_match("Well, HELLO there"));
//!
//! // fn:replace with capture references.
//! let swap = Regex::xpath("(\\w+)\\s+(\\w+)", "")?;
//! assert_eq!(swap.replace("John Smith", "$2, $1")?, "Smith, John");
//!
//! // fn:tokenize.
//! let commas = Regex::xpath(",\\s*", "")?;
//! assert_eq!(commas.tokenize("1, 15, 24, 50")?, ["1", "15", "24", "50"]);
//! # Ok::<(), oxixml_regex::RegexError>(())
//! ```

#![deny(missing_docs)]
#![forbid(unsafe_code)]

mod ast;
mod backtrack;
mod charclass;
mod error;
mod matches;
mod parser;
mod pikevm;
mod program;
mod replace;

pub use error::{ForxCode, RegexError};
pub use matches::{Match, Matches};

use backtrack::BtProgram;
use parser::{Dialect, Flags};
use pikevm::Captures;
use program::Program;

/// Which version of [XML Schema][xsd] a pattern is read against.
///
/// This governs *both* dialects. A `pattern` facet is read against the version
/// of the schema that declares it; an XPath pattern is read against the version
/// the processor implements, because [F&O 3.1 §5.6.1][fo] defines the XPath
/// syntax by reference to the XML Schema grammar rather than restating it.
///
/// The two Appendices give the same grammar except in a character class, where
/// XSD 1.1 relaxed two XSD 1.0 rules:
///
/// * a raw `-` may stand as a member anywhere in a character group, not only as
///   its first or last member — so `[a-a-x-x]`, `[a-c-1-4x-z-7-9]`, and
///   `[^a-d-b-c]` compile under 1.1 and are rejected under 1.0. This one
///   applies in both dialects.
/// * an unrecognised `\p{Is…}` block name matches every character rather than
///   being an error — so `\p{IsaA0-a9}` compiles under 1.1 and is rejected
///   under 1.0. This one is the **`pattern`-facet dialect's only**: the
///   XPath dialect rejects an unknown block name whatever the version, which
///   is what QT3 `fn-matches.re` `re00216` requires of that very pattern with
///   no `<dependency>` on either side.
///
/// Everything else, including range subtraction and the rule that an unescaped
/// `-` is never a range *endpoint*, is common to both.
///
/// [xsd]: https://www.w3.org/TR/xmlschema-2/#regexs
/// [fo]: https://www.w3.org/TR/xpath-functions-31/#regex-syntax
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum XsdVersion {
    /// [XML Schema 1.0][spec], Appendix F.
    ///
    /// [spec]: https://www.w3.org/TR/xmlschema-2/#regexs
    V1_0,
    /// [XML Schema 1.1][spec], Appendix G.
    ///
    /// [spec]: https://www.w3.org/TR/xmlschema11-2/#regexs
    V1_1,
}

/// The XML Schema version [`Regex::xpath`] reads a pattern against.
///
/// An XPath processor answers for exactly one version — the suites write the
/// two readings as pairs over the same pattern and gate them on it — and this
/// workspace's processor reports `1.0`: `system-property('xsl:xsd-version')`
/// answers `1.0`, and the XSLT 3.0 driver declares the catalogue's `XSD_1.1`
/// feature unsupported. A host that implements XSD 1.1 compiles its patterns
/// with [`Regex::xpath_versioned`] instead.
///
/// This is deliberately *not* [`Regex::xsd`]'s default: a `pattern` facet is
/// read against the version of the schema document that carries it, which the
/// validator knows and passes to [`Regex::xsd_versioned`] explicitly.
pub const DEFAULT_XSD_VERSION: XsdVersion = XsdVersion::V1_0;

/// Which anchoring rule applies when deciding whether an input matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MatchMode {
    /// The whole string must match (XSD dialect).
    Full,
    /// Any substring may match (XPath dialect).
    Search,
}

/// The compiled form of the pattern.
#[derive(Debug, Clone)]
enum Engine {
    /// A linear-time Pike VM (patterns without back-references).
    Pike(Program),
    /// A budgeted back-tracking interpreter (patterns with back-references).
    Backtrack(BtProgram),
}

/// A single match with its capture slots, in the engine's native
/// [`char`]-index coordinate space (an index into the `Vec<char>` the engine
/// matches over, not a byte offset). This is the shared internal
/// representation `replace`, `tokenize`, and [`matches::Matches`] all build
/// on; the public, byte-indexed [`Match`] is derived from it by
/// [`matches::Matches::next`].
pub(crate) struct RawMatch {
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) caps: Captures,
}

impl RawMatch {
    fn from_captures(caps: Captures) -> Option<RawMatch> {
        let start = caps.first().copied().flatten()?;
        let end = caps.get(1).copied().flatten()?;
        Some(RawMatch { start, end, caps })
    }
}

/// A compiled regular expression in either the XSD or the XPath dialect.
///
/// Build one with [`Regex::xsd`] or [`Regex::xpath`], then test inputs with
/// [`Regex::is_match`], or transform them with [`Regex::replace`] and
/// [`Regex::tokenize`].
#[derive(Debug, Clone)]
pub struct Regex {
    engine: Engine,
    mode: MatchMode,
    flags: Flags,
    group_count: usize,
    /// The lower bound of a counted repetition the compiler declined to
    /// unroll, if the pattern has one.
    ///
    /// Such a repetition compiles to an instruction that always fails, which
    /// is the correct answer only while the input is shorter than the bound —
    /// so an input at least that long is refused with the `FORX0002` the
    /// compiler would otherwise have raised. In practice that means never:
    /// the bound is above 200 000 characters.
    unexpandable_min: Option<usize>,
}

impl Regex {
    /// Compiles `pattern` as an [XML Schema 1.1 regular expression][xsd].
    ///
    /// The pattern is matched against the **whole** input string (implicit
    /// anchoring), there are no flags, `^` and `$` are literal characters, and
    /// back-references are not permitted.
    ///
    /// This is [`Regex::xsd_versioned`] with [`XsdVersion::V1_1`]; a validator
    /// running in XSD 1.0 mode must call that instead.
    ///
    /// # Errors
    ///
    /// Returns [`RegexError::InvalidPattern`] (`FORX0002`) if the pattern is not
    /// syntactically valid.
    ///
    /// [xsd]: https://www.w3.org/TR/xmlschema11-2/#regexs
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_regex::Regex;
    ///
    /// let re = Regex::xsd("\\d{3}-\\d{4}")?;
    /// assert!(re.is_match("867-5309"));
    /// assert!(!re.is_match("call 867-5309"));   // whole-string match
    /// # Ok::<(), oxixml_regex::RegexError>(())
    /// ```
    pub fn xsd(pattern: &str) -> Result<Regex, RegexError> {
        Self::xsd_versioned(pattern, XsdVersion::V1_1)
    }

    /// Compiles `pattern` as an XML Schema regular expression, reading it
    /// against the grammar of the given [`XsdVersion`].
    ///
    /// Matching behaves exactly as for [`Regex::xsd`]; the version only decides
    /// which patterns are accepted at all.
    ///
    /// # Errors
    ///
    /// Returns [`RegexError::InvalidPattern`] (`FORX0002`) if the pattern is not
    /// syntactically valid under `version`.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_regex::{Regex, XsdVersion};
    ///
    /// // A '-' chained between two ranges: XSD 1.1 reads it as a literal
    /// // hyphen member, XSD 1.0 rejects the pattern outright.
    /// let lenient = Regex::xsd_versioned("[a-a-x-x]+", XsdVersion::V1_1)?;
    /// assert!(lenient.is_match("a-x"));
    /// assert!(Regex::xsd_versioned("[a-a-x-x]+", XsdVersion::V1_0).is_err());
    /// # Ok::<(), oxixml_regex::RegexError>(())
    /// ```
    pub fn xsd_versioned(pattern: &str, version: XsdVersion) -> Result<Regex, RegexError> {
        let flags = Flags::default();
        let parsed = parser::parse(pattern, Dialect::Xsd, version, flags)?;
        // XSD patterns can never contain a back-reference.
        let program = program::compile(&parsed.ast, parsed.group_count)?;
        Ok(Regex {
            engine: Engine::Pike(program),
            mode: MatchMode::Full,
            flags,
            group_count: parsed.group_count,
            unexpandable_min: parsed.ast.unexpandable_min(program::EXPANSION_LIMIT),
        })
    }

    /// Compiles `pattern` as an [XPath / XQuery regular expression][fo] with the
    /// given `flags` string (any combination of `s`, `m`, `i`, `x`, `q`).
    ///
    /// Matching is unanchored: [`is_match`](Regex::is_match) returns `true` if
    /// any substring matches.
    ///
    /// Character classes are read against [`DEFAULT_XSD_VERSION`], the version
    /// this workspace's processor reports; [`Regex::xpath_versioned`] names
    /// another.
    ///
    /// # Errors
    ///
    /// Returns [`RegexError::InvalidFlags`] (`FORX0001`) if `flags` contains an
    /// unsupported character, or [`RegexError::InvalidPattern`] (`FORX0002`) if
    /// the pattern is not syntactically valid.
    ///
    /// [fo]: https://www.w3.org/TR/xpath-functions-31/#regex-syntax
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_regex::Regex;
    ///
    /// let re = Regex::xpath("^a.*z$", "")?;
    /// assert!(re.is_match("abcz"));
    ///
    /// // Multi-line and dot-all flags.
    /// let ml = Regex::xpath("^b", "m")?;
    /// assert!(ml.is_match("a\nb"));
    /// # Ok::<(), oxixml_regex::RegexError>(())
    /// ```
    pub fn xpath(pattern: &str, flags: &str) -> Result<Regex, RegexError> {
        Self::xpath_versioned(pattern, flags, DEFAULT_XSD_VERSION)
    }

    /// Compiles `pattern` as an [XPath / XQuery regular expression][fo],
    /// reading its character classes against the grammar of the given
    /// [`XsdVersion`].
    ///
    /// [F&O 3.1 §5.6.1][fo] defines the XPath regular-expression syntax by
    /// reference to the XML Schema grammar, so which of the two Appendices
    /// applies is a property of the *processor* — the XSD version it
    /// implements and reports — rather than of the call. [`Regex::xpath`] names
    /// [`DEFAULT_XSD_VERSION`]; a processor that implements the other one calls
    /// this instead.
    ///
    /// Matching behaves exactly as for [`Regex::xpath`]; the version only
    /// decides which patterns are accepted at all, and it decides it for the
    /// two rules [`XsdVersion`] documents.
    ///
    /// # Errors
    ///
    /// [`RegexError::InvalidFlags`] (`FORX0001`) for an unsupported flag, or
    /// [`RegexError::InvalidPattern`] (`FORX0002`) if the pattern is not
    /// syntactically valid under `version`.
    ///
    /// [fo]: https://www.w3.org/TR/xpath-functions-31/#regex-syntax
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_regex::{Regex, XsdVersion};
    ///
    /// // QT3 `fn-matches.re` writes this pattern twice, once per XSD version.
    /// assert!(Regex::xpath_versioned("[^a-d-b-c]", "", XsdVersion::V1_0).is_err());
    /// assert!(Regex::xpath_versioned("[^a-d-b-c]", "", XsdVersion::V1_1)?.is_match("x"));
    /// # Ok::<(), oxixml_regex::RegexError>(())
    /// ```
    pub fn xpath_versioned(
        pattern: &str,
        flags: &str,
        version: XsdVersion,
    ) -> Result<Regex, RegexError> {
        let flags = Flags::parse(flags)?;
        let parsed = parser::parse(pattern, Dialect::Xpath, version, flags)?;
        let engine = if parsed.ast.has_backref() {
            Engine::Backtrack(backtrack::compile(&parsed.ast, parsed.group_count)?)
        } else {
            Engine::Pike(program::compile(&parsed.ast, parsed.group_count)?)
        };
        Ok(Regex {
            engine,
            mode: MatchMode::Search,
            flags,
            group_count: parsed.group_count,
            unexpandable_min: parsed.ast.unexpandable_min(program::EXPANSION_LIMIT),
        })
    }

    /// Returns the number of capturing groups in the pattern.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_regex::Regex;
    ///
    /// assert_eq!(Regex::xpath("(a)(b)(?:c)", "")?.group_count(), 2);
    /// # Ok::<(), oxixml_regex::RegexError>(())
    /// ```
    #[must_use]
    pub fn group_count(&self) -> usize {
        self.group_count
    }

    /// Returns `true` if the input matches.
    ///
    /// For an XSD regex the **whole** string must match; for an XPath regex any
    /// substring may match. If a back-reference pattern exhausts its evaluation
    /// budget this returns `false`; use [`try_is_match`](Regex::try_is_match) to
    /// observe [`RegexError::BudgetExceeded`] instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_regex::Regex;
    ///
    /// assert!(Regex::xpath("a|b", "")?.is_match("xbx"));
    /// assert!(!Regex::xsd("a|b")?.is_match("xbx"));
    /// # Ok::<(), oxixml_regex::RegexError>(())
    /// ```
    #[must_use]
    pub fn is_match(&self, input: &str) -> bool {
        self.try_is_match(input).unwrap_or(false)
    }

    /// Like [`is_match`](Regex::is_match) but surfaces
    /// [`RegexError::BudgetExceeded`] for back-reference patterns whose
    /// evaluation is abandoned.
    ///
    /// # Errors
    ///
    /// Returns [`RegexError::BudgetExceeded`] if a back-reference pattern
    /// consumes its whole step budget without deciding.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_regex::Regex;
    ///
    /// let re = Regex::xpath("(a|b)\\1", "")?;
    /// assert_eq!(re.try_is_match("aa"), Ok(true));
    /// assert_eq!(re.try_is_match("ab"), Ok(false));
    /// # Ok::<(), oxixml_regex::RegexError>(())
    /// ```
    pub fn try_is_match(&self, input: &str) -> Result<bool, RegexError> {
        let chars: Vec<char> = input.chars().collect();
        self.check_unexpandable(chars.len())?;
        let ci = self.flags.case_insensitive;
        let ml = self.flags.multiline;
        match &self.engine {
            Engine::Pike(program) => {
                let (anchored, full) = match self.mode {
                    MatchMode::Full => (true, true),
                    MatchMode::Search => (false, false),
                };
                Ok(pikevm::run(program, &chars, 0, anchored, full, ci, ml).is_some())
            }
            Engine::Backtrack(program) => {
                let mut budget = backtrack::DEFAULT_BUDGET;
                match self.mode {
                    MatchMode::Search => {
                        backtrack::is_match_search(program, &chars, ci, ml, &mut budget)
                    }
                    MatchMode::Full => {
                        backtrack::is_match_full(program, &chars, ci, ml, &mut budget)
                    }
                }
            }
        }
    }

    /// Returns an iterator over every non-overlapping match of the pattern in
    /// `input`, each carrying the byte position of the whole match and of
    /// every capturing group.
    ///
    /// This is the primitive an implementation of [`fn:analyze-string`][as]
    /// needs: for each match, [`Match::range`] gives the matched substring's
    /// position and [`Match::group`] gives the position of each capturing
    /// group (`None` when a group did not participate, for example an
    /// untaken alternation branch), from which the nested
    /// `fn:match`/`fn:group` result structure can be derived.
    ///
    /// The search is unanchored — it looks for a match starting at every
    /// position of `input`, the same way [`Regex::replace`] and
    /// [`Regex::tokenize`] do internally — regardless of whether the pattern
    /// was compiled with [`Regex::xsd`] or [`Regex::xpath`]; only
    /// [`is_match`](Regex::is_match) applies XSD's whole-string anchoring.
    ///
    /// See [`Match`] for the zero-length-match advancement rule.
    ///
    /// [as]: https://www.w3.org/TR/xpath-functions-31/#func-analyze-string
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_regex::Regex;
    ///
    /// let re = Regex::xpath("(a)(b)?", "")?;
    /// let matches: Vec<_> = re.find_iter("ab a").collect::<Result<_, _>>()?;
    /// assert_eq!(matches.len(), 2);
    /// assert_eq!(matches[0].as_str(), "ab");
    /// assert_eq!(matches[0].group_str(1), Some("a"));
    /// assert_eq!(matches[0].group_str(2), Some("b"));
    /// assert_eq!(matches[1].as_str(), "a");
    /// assert_eq!(matches[1].group_str(2), None); // group 2 did not participate
    /// # Ok::<(), oxixml_regex::RegexError>(())
    /// ```
    pub fn find_iter<'r, 't>(&'r self, input: &'t str) -> Matches<'r, 't> {
        matches::Matches::new(self, input)
    }

    /// Returns the left-most match of the pattern in `input`, or `None` if it
    /// does not match. A convenience for `re.find_iter(input).next().transpose()`.
    ///
    /// # Errors
    ///
    /// Returns [`RegexError::BudgetExceeded`] for an abandoned back-reference
    /// pattern; see [`Regex::find_iter`].
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_regex::Regex;
    ///
    /// let re = Regex::xpath("[0-9]+", "")?;
    /// let m = re.find("abc123def")?.expect("should match");
    /// assert_eq!(m.as_str(), "123");
    /// assert_eq!(re.find("no digits here")?, None);
    /// # Ok::<(), oxixml_regex::RegexError>(())
    /// ```
    pub fn find<'t>(&self, input: &'t str) -> Result<Option<Match<'t>>, RegexError> {
        self.find_iter(input).next().transpose()
    }

    /// Replaces every non-overlapping match with `replacement`, following the
    /// [`fn:replace`][fo] semantics.
    ///
    /// Within `replacement`, `$0` is the whole match and `$N` is the text of
    /// capturing group `N`; a literal `$` or `\` must be written `\$` or `\\`.
    /// If the pattern was compiled with the `q` flag the replacement is used
    /// verbatim.
    ///
    /// # Errors
    ///
    /// * [`RegexError::InvalidReplacement`] (`FORX0004`) if `replacement` has a
    ///   stray `$` or `\`.
    /// * [`RegexError::MatchesEmptyString`] (`FORX0003`) if the pattern can match
    ///   a zero-length string.
    /// * [`RegexError::BudgetExceeded`] for an abandoned back-reference pattern.
    ///
    /// [fo]: https://www.w3.org/TR/xpath-functions-31/#func-replace
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_regex::Regex;
    ///
    /// let re = Regex::xpath("[0-9]+", "")?;
    /// assert_eq!(re.replace("a1b22c333", "#")?, "a#b#c#");
    /// # Ok::<(), oxixml_regex::RegexError>(())
    /// ```
    pub fn replace(&self, input: &str, replacement: &str) -> Result<String, RegexError> {
        if !self.flags.literal {
            replace::validate_replacement(replacement)?;
        }
        let chars: Vec<char> = input.chars().collect();
        if self.matches_empty()? {
            return Err(RegexError::MatchesEmptyString);
        }
        let mut out = String::new();
        let mut last = 0;
        // One budget for the whole call: every match this loop looks for
        // shares it, rather than each starting fresh, so a back-reference
        // pattern cannot buy unbounded total work by matching (cheaply)
        // over and over across a long input. See `backtrack::find_from`.
        let mut budget = backtrack::DEFAULT_BUDGET;
        loop {
            match self.find_from(&chars, last, &mut budget)? {
                None => break,
                Some(m) => {
                    out.extend(chars[last..m.start].iter());
                    if self.flags.literal {
                        out.push_str(replacement);
                    } else {
                        replace::expand_into(
                            &mut out,
                            replacement,
                            &chars,
                            &m.caps,
                            self.group_count,
                        );
                    }
                    if m.end == m.start {
                        // Guarded above, but never advance by zero.
                        break;
                    }
                    last = m.end;
                }
            }
            if last >= chars.len() {
                break;
            }
        }
        out.extend(chars[last..].iter());
        Ok(out)
    }

    /// Splits `input` at every non-overlapping match, following the
    /// [`fn:tokenize`][fo] semantics.
    ///
    /// A match at the start or end of the input, or two adjacent matches, yields
    /// a zero-length token in the corresponding position. An empty input yields
    /// an empty vector.
    ///
    /// # Errors
    ///
    /// * [`RegexError::MatchesEmptyString`] (`FORX0003`) if the pattern can match
    ///   a zero-length string.
    /// * [`RegexError::BudgetExceeded`] for an abandoned back-reference pattern.
    ///
    /// [fo]: https://www.w3.org/TR/xpath-functions-31/#func-tokenize
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_regex::Regex;
    ///
    /// let ws = Regex::xpath("\\s+", "")?;
    /// assert_eq!(ws.tokenize(" red green blue ")?, ["", "red", "green", "blue", ""]);
    /// # Ok::<(), oxixml_regex::RegexError>(())
    /// ```
    pub fn tokenize(&self, input: &str) -> Result<Vec<String>, RegexError> {
        let chars: Vec<char> = input.chars().collect();
        if chars.is_empty() {
            return Ok(Vec::new());
        }
        if self.matches_empty()? {
            return Err(RegexError::MatchesEmptyString);
        }
        let mut result = Vec::new();
        let mut last = 0;
        // See the identical comment in `replace`: one budget for the whole
        // call, shared across every match found.
        let mut budget = backtrack::DEFAULT_BUDGET;
        loop {
            match self.find_from(&chars, last, &mut budget)? {
                None => break,
                Some(m) => {
                    result.push(chars[last..m.start].iter().collect());
                    last = m.end;
                    if m.end == m.start {
                        break;
                    }
                }
            }
            if last > chars.len() {
                break;
            }
        }
        result.push(chars[last..].iter().collect());
        Ok(result)
    }

    /// Finds the left-most match starting at or after `pos`, in `char`-index
    /// coordinates. Shared by `replace`, `tokenize`, and
    /// [`Matches`](matches::Matches), the engine underneath
    /// [`Regex::find_iter`].
    ///
    /// `budget` is only consulted for a back-reference pattern (the
    /// [`Engine::Pike`] branch below never touches it, since the linear-time
    /// Pike VM cannot suffer catastrophic back-tracking in the first place);
    /// it is shared across every starting position tried *within this one
    /// call*, so a search that is expensive at many positions is bounded,
    /// not only a search that is catastrophic at a single one — see
    /// `backtrack::find_from`. Whether that same budget also carries across
    /// more than one call to `find_from` is the caller's choice: `replace`
    /// and `tokenize` each keep one budget alive across every match found
    /// during that single call, since the whole call is one logical
    /// operation with one result, while [`Matches`](matches::Matches) hands
    /// each match its own fresh budget, since an iterator over a huge
    /// document is expected to keep finding matches for as long as the
    /// document has more to give — that is ordinary operation, not the
    /// runaway back-tracking this budget exists to catch.
    pub(crate) fn find_from(
        &self,
        chars: &[char],
        pos: usize,
        budget: &mut u64,
    ) -> Result<Option<RawMatch>, RegexError> {
        self.check_unexpandable(chars.len())?;
        let ci = self.flags.case_insensitive;
        let ml = self.flags.multiline;
        let caps = match &self.engine {
            Engine::Pike(program) => pikevm::run(program, chars, pos, false, false, ci, ml),
            Engine::Backtrack(program) => {
                backtrack::find_from(program, chars, pos, ci, ml, budget)?
            }
        };
        Ok(caps.and_then(RawMatch::from_captures))
    }

    /// Refuses an input long enough to reach a counted repetition the compiler
    /// declined to unroll.
    ///
    /// Below that length the un-unrolled repetition is correctly compiled as a
    /// failure, because its body consumes at least one character per
    /// iteration; at or above it the program would have to be the real
    /// expansion, which is what the compiler could not build.
    fn check_unexpandable(&self, length: usize) -> Result<(), RegexError> {
        match self.unexpandable_min {
            Some(min) if length >= min => Err(RegexError::InvalidPattern(
                "quantifier lower bound is too large".to_owned(),
            )),
            _ => Ok(()),
        }
    }

    /// Whether the pattern can match a zero-length string, per the `FORX0003`
    /// rule `fn:matches("", pattern)`.
    fn matches_empty(&self) -> Result<bool, RegexError> {
        let empty: [char; 0] = [];
        let ci = self.flags.case_insensitive;
        let ml = self.flags.multiline;
        match &self.engine {
            Engine::Pike(program) => {
                Ok(pikevm::run(program, &empty, 0, false, false, ci, ml).is_some())
            }
            Engine::Backtrack(program) => {
                let mut budget = backtrack::DEFAULT_BUDGET;
                backtrack::is_match_search(program, &empty, ci, ml, &mut budget)
            }
        }
    }
}
