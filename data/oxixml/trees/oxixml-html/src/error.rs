//! Parse errors reported by the tokenizer and the tree construction stage.
//!
//! An HTML parser never fails: [HTML § 13](https://html.spec.whatwg.org/multipage/parsing.html)
//! defines a recovery behaviour for every possible input, so a "parse error" is
//! a diagnostic rather than a failure. This crate therefore collects errors in a
//! list on the parse result instead of returning `Result`, and the caller can
//! ignore them entirely, surface them, or treat any of them as fatal.
//!
//! Every error carries the [spec error code](codes) that named it — the same
//! kebab-case identifiers the standard uses in its
//! [parse errors table](https://html.spec.whatwg.org/multipage/parsing.html#parse-errors)
//! — plus the position in the *preprocessed* input stream (that is, after
//! newline normalisation) at which the parser noticed it.

use core::fmt;

use oxixml_support::pos::TextPosition;

/// A single parse error: a spec error code plus where it was noticed.
///
/// # Examples
///
/// ```
/// use oxixml_html::parse_document;
///
/// // A missing DOCTYPE is the classic error of a fragmentary document.
/// let parsed = parse_document("<p>unterminated");
/// assert!(parsed.errors().iter().any(|error| error.code() == "expected-doctype"));
/// assert_eq!(parsed.errors()[0].position().line, 0); // 0-based line
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    position: TextPosition,
    code: &'static str,
}

impl ParseError {
    /// Creates an error with the given spec code at the given position.
    #[must_use]
    pub const fn new(position: TextPosition, code: &'static str) -> Self {
        Self { position, code }
    }

    /// The position in the preprocessed input stream at which the error was
    /// noticed.
    ///
    /// "Preprocessed" means after the newline normalisation of
    /// [HTML § 13.2.3.5](https://html.spec.whatwg.org/multipage/parsing.html#preprocessing-the-input-stream),
    /// so a `CRLF` pair counts as the single character it was folded into.
    ///
    /// Its `line` and `column` are **0-based** (see [`oxixml_support::pos::TextPosition`]);
    /// only its `Display` impl renders the human-facing 1-based numbering.
    #[must_use]
    pub const fn position(&self) -> TextPosition {
        self.position
    }

    /// The spec error code, for example `unexpected-null-character`.
    ///
    /// The constants in [`codes`] name every code this crate emits.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        self.code
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} at line {}, column {}",
            self.code,
            self.position.line + 1,
            self.position.column + 1
        )
    }
}

impl std::error::Error for ParseError {}

/// The spec error codes this crate emits.
///
/// Codes whose name appears in the
/// [HTML standard's parse error table](https://html.spec.whatwg.org/multipage/parsing.html#parse-errors)
/// are spelled exactly as the standard spells them. The handful of codes the
/// standard leaves unnamed — tree construction errors, and the two codes that
/// report content which cannot be represented in an XML DOM — are marked as
/// such in their documentation.
pub mod codes {
    /// `<!--` never closed before end of file.
    pub const EOF_IN_COMMENT: &str = "eof-in-comment";
    /// A `<!DOCTYPE` never closed before end of file.
    pub const EOF_IN_DOCTYPE: &str = "eof-in-doctype";
    /// A `<![CDATA[` never closed before end of file.
    pub const EOF_IN_CDATA: &str = "eof-in-cdata";
    /// End of file inside a tag.
    pub const EOF_IN_TAG: &str = "eof-in-tag";
    /// End of file inside `<script>` text that had been escaped with `<!--`.
    pub const EOF_IN_SCRIPT_HTML_COMMENT_LIKE_TEXT: &str = "eof-in-script-html-comment-like-text";
    /// End of file right after `<`.
    pub const EOF_BEFORE_TAG_NAME: &str = "eof-before-tag-name";
    /// `<` followed by something that cannot start a tag name.
    pub const INVALID_FIRST_CHARACTER_OF_TAG_NAME: &str = "invalid-first-character-of-tag-name";
    /// `</>`, an end tag with no name.
    pub const MISSING_END_TAG_NAME: &str = "missing-end-tag-name";
    /// `<?` where a tag name was expected, which the standard reconsumes in
    /// the bogus comment state.
    pub const UNEXPECTED_QUESTION_MARK_INSTEAD_OF_TAG_NAME: &str =
        "unexpected-question-mark-instead-of-tag-name";
    /// End of file inside a `<?…?>` processing instruction.
    pub const EOF_IN_PROCESSING_INSTRUCTION: &str = "eof-in-processing-instruction";
    /// `<?` followed by something that cannot start a target.
    pub const INVALID_FIRST_CHARACTER_OF_PROCESSING_INSTRUCTION_TARGET: &str =
        "invalid-first-character-of-processing-instruction-target";
    /// A processing instruction target containing a character other than an
    /// ASCII alphanumeric, `-` or `_`.
    pub const INVALID_PROCESSING_INSTRUCTION_TARGET: &str = "invalid-processing-instruction-target";
    /// A processing instruction whose target is `xml` or `xml-stylesheet`,
    /// which HTML reserves.
    pub const DISALLOWED_PROCESSING_INSTRUCTION_TARGET: &str =
        "disallowed-processing-instruction-target";
    /// An end tag that carried attributes, which are always ignored.
    pub const END_TAG_WITH_ATTRIBUTES: &str = "end-tag-with-attributes";
    /// `</foo/>`, an end tag marked self-closing.
    pub const END_TAG_WITH_TRAILING_SOLIDUS: &str = "end-tag-with-trailing-solidus";
    /// A U+0000 NULL in a context where it is replaced or dropped.
    pub const UNEXPECTED_NULL_CHARACTER: &str = "unexpected-null-character";
    /// `=` where an attribute name was expected.
    pub const UNEXPECTED_EQUALS_SIGN_BEFORE_ATTRIBUTE_NAME: &str =
        "unexpected-equals-sign-before-attribute-name";
    /// `"`, `'` or `<` inside an attribute name.
    pub const UNEXPECTED_CHARACTER_IN_ATTRIBUTE_NAME: &str =
        "unexpected-character-in-attribute-name";
    /// `` ` ``, `"` or `'` at the start of an unquoted attribute value.
    pub const UNEXPECTED_CHARACTER_IN_UNQUOTED_ATTRIBUTE_VALUE: &str =
        "unexpected-character-in-unquoted-attribute-value";
    /// `>` where an attribute value was expected.
    pub const MISSING_ATTRIBUTE_VALUE: &str = "missing-attribute-value";
    /// Two attributes separated by nothing.
    pub const MISSING_WHITESPACE_BETWEEN_ATTRIBUTES: &str = "missing-whitespace-between-attributes";
    /// An attribute name that already occurred on the same tag; it is dropped.
    pub const DUPLICATE_ATTRIBUTE: &str = "duplicate-attribute";
    /// A `/` that is not part of `/>`.
    pub const UNEXPECTED_SOLIDUS_IN_TAG: &str = "unexpected-solidus-in-tag";
    /// `<!` followed by something other than `--`, `DOCTYPE` or `[CDATA[`.
    pub const INCORRECTLY_OPENED_COMMENT: &str = "incorrectly-opened-comment";
    /// `<!-->` or `<!--->`.
    pub const ABRUPT_CLOSING_OF_EMPTY_COMMENT: &str = "abrupt-closing-of-empty-comment";
    /// `--!>` used to close a comment.
    pub const INCORRECTLY_CLOSED_COMMENT: &str = "incorrectly-closed-comment";
    /// A `<!-- -- -->` style nested comment start.
    pub const NESTED_COMMENT: &str = "nested-comment";
    /// `<![CDATA[` outside foreign content.
    pub const CDATA_IN_HTML_CONTENT: &str = "cdata-in-html-content";
    /// `<!DOCTYPEhtml`, with no space after the keyword.
    pub const MISSING_WHITESPACE_BEFORE_DOCTYPE_NAME: &str =
        "missing-whitespace-before-doctype-name";
    /// `<!DOCTYPE>`, with no name at all.
    pub const MISSING_DOCTYPE_NAME: &str = "missing-doctype-name";
    /// A DOCTYPE whose public identifier is not preceded by white space.
    pub const MISSING_WHITESPACE_AFTER_DOCTYPE_PUBLIC_KEYWORD: &str =
        "missing-whitespace-after-doctype-public-keyword";
    /// A DOCTYPE whose system identifier is not preceded by white space.
    pub const MISSING_WHITESPACE_AFTER_DOCTYPE_SYSTEM_KEYWORD: &str =
        "missing-whitespace-after-doctype-system-keyword";
    /// A DOCTYPE with `PUBLIC` but no identifier.
    pub const MISSING_DOCTYPE_PUBLIC_IDENTIFIER: &str = "missing-doctype-public-identifier";
    /// A DOCTYPE with `SYSTEM` but no identifier.
    pub const MISSING_DOCTYPE_SYSTEM_IDENTIFIER: &str = "missing-doctype-system-identifier";
    /// A DOCTYPE public identifier that is not quoted.
    pub const MISSING_QUOTE_BEFORE_DOCTYPE_PUBLIC_IDENTIFIER: &str =
        "missing-quote-before-doctype-public-identifier";
    /// A DOCTYPE system identifier that is not quoted.
    pub const MISSING_QUOTE_BEFORE_DOCTYPE_SYSTEM_IDENTIFIER: &str =
        "missing-quote-before-doctype-system-identifier";
    /// A DOCTYPE public identifier closed by `>` instead of its quote.
    pub const ABRUPT_DOCTYPE_PUBLIC_IDENTIFIER: &str = "abrupt-doctype-public-identifier";
    /// A DOCTYPE system identifier closed by `>` instead of its quote.
    pub const ABRUPT_DOCTYPE_SYSTEM_IDENTIFIER: &str = "abrupt-doctype-system-identifier";
    /// Two DOCTYPE identifiers with nothing between them.
    pub const MISSING_WHITESPACE_BETWEEN_DOCTYPE_PUBLIC_AND_SYSTEM_IDENTIFIERS: &str =
        "missing-whitespace-between-doctype-public-and-system-identifiers";
    /// Junk after a complete DOCTYPE.
    pub const UNEXPECTED_CHARACTER_AFTER_DOCTYPE_SYSTEM_IDENTIFIER: &str =
        "unexpected-character-after-doctype-system-identifier";
    /// A DOCTYPE keyword that is neither `PUBLIC` nor `SYSTEM`.
    pub const INVALID_CHARACTER_SEQUENCE_AFTER_DOCTYPE_NAME: &str =
        "invalid-character-sequence-after-doctype-name";
    /// `&#;` — a numeric reference with no digits.
    pub const ABSENCE_OF_DIGITS_IN_NUMERIC_CHARACTER_REFERENCE: &str =
        "absence-of-digits-in-numeric-character-reference";
    /// A numeric reference not terminated by `;`.
    pub const MISSING_SEMICOLON_AFTER_CHARACTER_REFERENCE: &str =
        "missing-semicolon-after-character-reference";
    /// `&#0;`.
    pub const NULL_CHARACTER_REFERENCE: &str = "null-character-reference";
    /// A numeric reference above U+10FFFF.
    pub const CHARACTER_REFERENCE_OUTSIDE_UNICODE_RANGE: &str =
        "character-reference-outside-unicode-range";
    /// A numeric reference naming a surrogate code point.
    pub const SURROGATE_CHARACTER_REFERENCE: &str = "surrogate-character-reference";
    /// A numeric reference naming a noncharacter.
    pub const NONCHARACTER_CHARACTER_REFERENCE: &str = "noncharacter-character-reference";
    /// A numeric reference naming a control character.
    pub const CONTROL_CHARACTER_REFERENCE: &str = "control-character-reference";
    /// `&notit;` — a name that looks like an entity but is not one.
    pub const UNKNOWN_NAMED_CHARACTER_REFERENCE: &str = "unknown-named-character-reference";
    /// `&foo=bar` inside an attribute value.
    pub const UNEXPECTED_CHARACTER_AFTER_AMPERSAND: &str = "unexpected-character-after-ampersand";

    // ---------------------------------------------------------------- tree
    /// Tree construction: a document that starts with something other than a
    /// DOCTYPE. Not named by the standard, which describes it in prose.
    pub const EXPECTED_DOCTYPE: &str = "expected-doctype";
    /// Tree construction: a DOCTYPE somewhere other than at the start.
    pub const UNEXPECTED_DOCTYPE: &str = "unexpected-doctype";
    /// Tree construction: a start tag that is not allowed here.
    pub const UNEXPECTED_START_TAG: &str = "unexpected-start-tag";
    /// Tree construction: an end tag that is not allowed here.
    pub const UNEXPECTED_END_TAG: &str = "unexpected-end-tag";
    /// Tree construction: a character that is not allowed here.
    pub const UNEXPECTED_CHARACTER: &str = "unexpected-character";
    /// Tree construction: end of file with elements still open.
    pub const UNEXPECTED_EOF: &str = "unexpected-eof";
    /// Tree construction: an end tag whose element is not in scope.
    pub const END_TAG_NOT_IN_SCOPE: &str = "end-tag-not-in-scope";
    /// Tree construction: the adoption agency algorithm ran, which always means
    /// misnested formatting elements.
    pub const MISNESTED_TAG: &str = "misnested-tag";
    /// Tree construction: content was foster parented out of a table.
    pub const FOSTER_PARENTING: &str = "foster-parenting";

    // ------------------------------------------------------------- XML DOM
    /// Not a standard code: a name that HTML permits but XML does not, which
    /// this crate rewrote so it could live in an XML DOM. See the crate-level
    /// documentation section on XML representability.
    pub const UNREPRESENTABLE_NAME: &str = "oxixml-unrepresentable-name";
    /// Not a standard code: character data that HTML permits but XML does not,
    /// which this crate rewrote so it could live in an XML DOM. See the
    /// crate-level documentation section on XML representability.
    pub const UNREPRESENTABLE_CHARACTER_DATA: &str = "oxixml-unrepresentable-character-data";

    // -------------------------------------------------------- resource limits
    /// Not a standard code: the stack of open elements reached
    /// [`crate::ParseOptions::max_depth`] (default
    /// [`crate::ParseOptions::DEFAULT_MAX_DEPTH`]; raise or lower it with
    /// [`crate::ParseOptions::with_max_depth`]).
    ///
    /// Several tree construction algorithms — matching an end tag against the
    /// stack, the six element scopes, the adoption agency algorithm — walk
    /// the stack of open elements from the top down, so its length bounds
    /// their cost. Once the limit is reached, the tree builder keeps
    /// consuming tokens rather than growing the stack any further: an
    /// element created from this point on is still inserted into the
    /// document, but is no longer tracked as open, so it becomes a child of
    /// whichever element was current when the limit was first reached rather
    /// than of the element that actually encloses it in the source. This
    /// code is recorded the first time the limit is reached; it is not
    /// repeated for every element past it.
    pub const MAX_DEPTH_EXCEEDED: &str = "oxixml-max-depth-exceeded";
}
