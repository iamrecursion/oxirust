//! CURIE, term and IRI processing.
//!
//! [RDFa Core 1.1 §6](https://www.w3.org/TR/rdfa-core/#s_curies) defines a *CURIE* — a compact IRI
//! expression — and [§7.4](https://www.w3.org/TR/rdfa-core/#s_curieprocessing) defines the three
//! attribute datatypes built on it. This module implements both, plus the `term` production of
//! [§7.4.3](https://www.w3.org/TR/rdfa-core/#s_terms).
//!
//! ```text
//! prefix      ::= NCName
//! reference   ::= ( ipath-absolute / ipath-rootless / ipath-empty )
//!                 [ "?" iquery ] [ "#" ifragment ]
//! curie       ::= [ [ prefix ] ':' ] reference
//! safe_curie  ::= '[' [ [ prefix ] ':' ] reference ']'
//! term        ::= NCNameStartChar termChar*
//! termChar    ::= ( NameChar - ':' ) | '/'
//! ```

use std::collections::BTreeMap;

use oxixml_iri::{Iri, IriRef};
use oxixml_qname::is_ncname;

use crate::context::XHTML_VOCABULARY;

/// The result of expanding a CURIE or an IRI reference: either a full IRI, or a reference to a
/// blank node named in the document with the `_:` prefix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Reference {
    /// A full, absolute IRI.
    Iri(String),
    /// The label of a blank node written as `_:label` (or, for the special empty case, `_:`).
    Blank(String),
}

/// The in-scope prefix mappings of an element.
///
/// Prefixes are compared case-insensitively, so every key stored here is already lower-cased. The
/// empty key is the *default prefix* used by a CURIE such as `:license`; RDFa Core fixes it to the
/// XHTML vocabulary, and a document may override it with `@prefix=": …"`.
pub(crate) type Prefixes = BTreeMap<String, String>;

/// The in-scope term mappings of an element, keyed by the term exactly as the initial context
/// spells it.
pub(crate) type Terms = BTreeMap<String, String>;

/// Builds the prefix map that every document starts from: the initial contexts of the host
/// language, plus the fixed default-prefix mapping to the XHTML vocabulary.
pub(crate) fn initial_prefixes(contexts: &[&[(&str, &str)]]) -> Prefixes {
    let mut prefixes = Prefixes::new();
    prefixes.insert(String::new(), XHTML_VOCABULARY.to_owned());
    for context in contexts {
        for (prefix, iri) in *context {
            prefixes.insert((*prefix).to_ascii_lowercase(), (*iri).to_owned());
        }
    }
    prefixes
}

/// Builds the term map that every document starts from.
pub(crate) fn initial_terms(contexts: &[&[(&str, &str)]]) -> Terms {
    let mut terms = Terms::new();
    for context in contexts {
        for (term, iri) in *context {
            terms.insert((*term).to_owned(), (*iri).to_owned());
        }
    }
    terms
}

/// Returns `true` when `value` matches the RDFa `term` production: an XML `NCName` that may also
/// contain `/` anywhere but at the start.
///
/// `NCNameStartChar` excludes `-`, and `/` is legal in a term exactly where `-` is legal in an
/// `NCName`, so rewriting `/` to `-` turns the term production into the `NCName` production.
pub(crate) fn is_term(value: &str) -> bool {
    if value.contains('/') {
        is_ncname(&value.replace('/', "-"))
    } else {
        is_ncname(value)
    }
}

/// Expands a CURIE against the in-scope prefix mappings, or returns `None` when the value is not a
/// CURIE at all.
///
/// A value with no colon is never a CURIE: the production allows it, but RDFa leaves the "no
/// prefix" mapping undefined and forbids host languages from defining one.
pub(crate) fn expand_curie(value: &str, prefixes: &Prefixes) -> Option<Reference> {
    let (prefix, reference) = value.split_once(':')?;
    if reference.contains(char::is_whitespace) {
        return None;
    }
    if prefix == "_" {
        // `_:label`, and the special bare `_:`, name a blank node. Any document-supplied mapping
        // for the `_` prefix must be ignored.
        return Some(Reference::Blank(reference.to_owned()));
    }
    if !prefix.is_empty() && !is_ncname(prefix) {
        return None;
    }
    let mapping = prefixes.get(&prefix.to_ascii_lowercase())?;
    Some(Reference::Iri(format!("{mapping}{reference}")))
}

/// Resolves an IRI reference against `base`, returning `None` when either is unusable.
pub(crate) fn resolve_iri(value: &str, base: &Iri<String>) -> Option<String> {
    let reference = IriRef::parse(value).ok()?;
    Some(base.resolve(&reference).ok()?.into_inner())
}

/// Processes an attribute of datatype `SafeCURIEorCURIEorIRI` — `@about` and `@resource`.
///
/// * `[…]` is a *safe CURIE*: the content is expanded as a CURIE, and ignored when that fails.
/// * anything else is expanded as a CURIE if it is one, and otherwise resolved as an IRI
///   reference. An empty value is therefore the base IRI, and `[]` is nothing at all.
pub(crate) fn safe_curie_or_curie_or_iri(
    value: &str,
    prefixes: &Prefixes,
    base: &Iri<String>,
    curie_base: &Iri<String>,
) -> Option<Reference> {
    if let Some(inner) = value.strip_prefix('[').and_then(|v| v.strip_suffix(']')) {
        return expand_curie(inner, prefixes).and_then(|r| absolutize(r, curie_base));
    }
    if let Some(reference) = expand_curie(value, prefixes) {
        return absolutize(reference, curie_base);
    }
    resolve_iri(value, base).map(Reference::Iri)
}

/// Makes a CURIE expansion absolute.
///
/// [RDFa Core 1.1 §7.4](https://www.w3.org/TR/rdfa-core/#s_curieprocessing) forbids resolving the
/// IRI a prefix maps to against the *current* base — an `@xml:base` or a `<base href>` deeper in
/// the document must not change what a prefix means. An RDF IRI still has to be absolute, so a
/// relative expansion is resolved against the IRI of the document itself, which is what the
/// W3C RDFa Test Suite's `Relative @profile` case expects.
fn absolutize(reference: Reference, curie_base: &Iri<String>) -> Option<Reference> {
    match reference {
        Reference::Iri(iri) => resolve_iri(&iri, curie_base).map(Reference::Iri),
        blank => Some(blank),
    }
}

/// Processes an attribute of datatype `TERMorCURIEorAbsIRI` — one value of `@property`,
/// `@typeof`, `@rel`, `@rev` or `@datatype`.
///
/// The order is the one [RDFa Core 1.1
/// §7.4](https://www.w3.org/TR/rdfa-core/#s_curieprocessing) prescribes: term, then CURIE, then
/// absolute IRI, then nothing. Relative IRI references are *not* accepted, which is what
/// distinguishes this datatype from `SafeCURIEorCURIEorIRI`.
pub(crate) fn term_or_curie_or_abs_iri(
    value: &str,
    prefixes: &Prefixes,
    terms: &Terms,
    vocabulary: Option<&str>,
    curie_base: &Iri<String>,
) -> Option<Reference> {
    if let Some(inner) = value.strip_prefix('[').and_then(|v| v.strip_suffix(']')) {
        // Not part of the datatype, but harmless and widely authored; a safe CURIE that does not
        // expand is ignored rather than falling through to the IRI branch.
        return expand_curie(inner, prefixes).and_then(|r| absolutize(r, curie_base));
    }
    if is_term(value) {
        return expand_term(value, terms, vocabulary)
            .and_then(|iri| absolutize(Reference::Iri(iri), curie_base));
    }
    if let Some(reference) = expand_curie(value, prefixes) {
        return absolutize(reference, curie_base);
    }
    if Iri::parse(value).is_ok() {
        return Some(Reference::Iri(value.to_owned()));
    }
    None
}

/// Expands a value that matches the `term` production, per [RDFa Core 1.1
/// §7.4.3](https://www.w3.org/TR/rdfa-core/#s_terms): the default vocabulary wins outright, then a
/// case-sensitive term lookup, then a case-insensitive one.
pub(crate) fn expand_term(value: &str, terms: &Terms, vocabulary: Option<&str>) -> Option<String> {
    if let Some(vocabulary) = vocabulary {
        return Some(format!("{vocabulary}{value}"));
    }
    if let Some(iri) = terms.get(value) {
        return Some(iri.clone());
    }
    terms
        .iter()
        .find(|(term, _)| term.eq_ignore_ascii_case(value))
        .map(|(_, iri)| iri.clone())
}

/// Splits an attribute that holds a whitespace-separated list of values.
pub(crate) fn split_values(value: &str) -> impl Iterator<Item = &str> {
    value.split_ascii_whitespace()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> Iri<String> {
        Iri::parse("http://example.com/doc".to_owned()).expect("a valid base")
    }

    fn prefixes() -> Prefixes {
        let mut prefixes = Prefixes::new();
        prefixes.insert(String::new(), XHTML_VOCABULARY.to_owned());
        prefixes.insert("foaf".to_owned(), "http://xmlns.com/foaf/0.1/".to_owned());
        prefixes
    }

    #[test]
    fn terms_accept_slashes_but_not_colons() {
        assert!(is_term("license"));
        assert!(is_term("a/b/c"));
        assert!(!is_term("/leading"));
        assert!(!is_term("foaf:name"));
        assert!(!is_term(""));
    }

    #[test]
    fn a_value_without_a_colon_is_not_a_curie() {
        assert_eq!(expand_curie("license", &prefixes()), None);
    }

    #[test]
    fn the_empty_prefix_is_the_xhtml_vocabulary() {
        assert_eq!(
            expand_curie(":license", &prefixes()),
            Some(Reference::Iri(
                "http://www.w3.org/1999/xhtml/vocab#license".to_owned()
            ))
        );
    }

    #[test]
    fn an_underscore_prefix_names_a_blank_node() {
        assert_eq!(
            expand_curie("_:john", &prefixes()),
            Some(Reference::Blank("john".to_owned()))
        );
        assert_eq!(
            expand_curie("_:", &prefixes()),
            Some(Reference::Blank(String::new()))
        );
    }

    #[test]
    fn an_unmapped_prefix_falls_back_to_an_iri() {
        assert_eq!(
            safe_curie_or_curie_or_iri("http://example.org/x", &prefixes(), &base(), &base()),
            Some(Reference::Iri("http://example.org/x".to_owned()))
        );
    }

    #[test]
    fn an_empty_safe_curie_is_ignored_but_an_empty_value_is_the_base() {
        assert_eq!(
            safe_curie_or_curie_or_iri("[]", &prefixes(), &base(), &base()),
            None
        );
        assert_eq!(
            safe_curie_or_curie_or_iri("", &prefixes(), &base(), &base()),
            Some(Reference::Iri("http://example.com/doc".to_owned()))
        );
    }

    #[test]
    fn a_relative_reference_is_not_a_term_or_curie_or_abs_iri() {
        let terms = Terms::new();
        assert_eq!(
            term_or_curie_or_abs_iri("../up", &prefixes(), &terms, None, &base()),
            None
        );
    }
}
