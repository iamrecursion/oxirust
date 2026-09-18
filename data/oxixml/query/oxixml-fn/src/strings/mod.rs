//! The functions on strings of
//! [F&O 3.1 §5 and §6](https://www.w3.org/TR/xpath-functions-31/#string-functions).
//!
//! * [`basic`] — construction, decomposition, normalization, case conversion and `fn:translate`.
//! * [`mod@compare`] — the collation-sensitive functions: `fn:compare` and the five substring
//!   matchers.
//! * [`analyze`] — `fn:analyze-string`, which reports a match-by-match split as an XML tree.
//! * [`regex_fns`] — `fn:matches`, `fn:replace` and `fn:tokenize`, over [`oxixml_regex`].
//! * [`uri`] — `fn:resolve-uri` and the three URI escaping functions.

pub mod analyze;
pub mod basic;
pub mod compare;
pub mod regex_fns;
pub mod uri;

pub use analyze::analyze_string;
pub use basic::{
    NormalizationForm, codepoints_to_string, codepoints_to_string_in, concat, lower_case,
    normalize_space, normalize_unicode, optional_string, string_join, string_length,
    string_to_codepoints, substring, translate, upper_case,
};
pub use compare::{
    codepoint_equal, collation_key, compare, contains, contains_token, default_collation,
    ends_with, starts_with, static_base_uri, substring_after, substring_before,
};
pub use regex_fns::{compile, matches, replace, tokenize, tokenize_whitespace};
pub use uri::{
    encode_for_uri, escape_html_uri, iri_to_uri, is_absolute_uri, resolve_uri, resolve_uri_against,
};
