//! `FourSectionDictionary` — the default HDT dictionary implementation.
//!
//! An HDT dictionary maps every distinct RDF term string appearing in a dataset to a compact
//! integer ID and back. `FourSectionDictionary` splits that mapping into four
//! `PlainFrontCoding` sections, per the specification:
//!
//! * **shared** — strings that occur at least once as a subject *and* at least once as an
//!   object.
//! * **subjects** — strings that occur only as a subject.
//! * **predicates** — every predicate string.
//! * **objects** — strings that occur only as an object.
//!
//! # ID space ("mapping 1")
//!
//! Subject and object IDs both re-use the shared section's numbering for shared terms, then
//! continue independently for their own exclusive strings ("mapping 1" in the specification's
//! terms — the alternative "mapping 2", where subject and object numbering both continue past
//! the shared *and* subjects sections, is not implemented):
//!
//! ```text
//! subject ID s:  1..=nshared          -> shared[s]
//!                nshared+1..=nsubj    -> subjects[s - nshared]
//!
//! object ID o:   1..=nshared          -> shared[o]
//!                nshared+1..=nobj     -> objects[o - nshared]
//!
//! predicate ID p: 1..=npred           -> predicates[p]   (independent ID space)
//! ```
//!
//! Every term's dictionary string is its canonical N-Triples serialization —
//! `<iri>`, `_:label`, or a quoted (optionally language- or datatype-tagged) literal — produced
//! and parsed by [`oxixml_model`]'s own [`Display`](std::fmt::Display)/[`FromStr`]
//! implementations, so quoting/escaping is exercised exactly once, in one well-tested place.

mod pfc;

use std::collections::BTreeSet;
use std::str::FromStr;

use oxixml_model::{
    NamedNode, NamedNodeRef, NamedOrBlankNode, NamedOrBlankNodeRef, Term, TermRef, Triple,
};

use crate::control_info::{ControlInfoType, ControlInformation};
use crate::error::{HdtError, Result};

use pfc::PlainFrontCoding;

/// The `FourSectionDictionary` format IRI, as written in the dictionary
/// [`ControlInformation`]'s `format` field.
const DICTIONARY_FORMAT: &str = "<http://purl.org/HDT/hdt#dictionaryFour>";

/// The dictionary section this crate exercises (`"dictionary"`, further qualified per-section
/// for more precise error messages).
const SECTION: &str = "dictionary";

/// A [`FourSectionDictionary`]: bidirectional string ↔ ID mapping split into shared / subjects
/// / predicates / objects sections.
#[derive(Debug, Clone, Default)]
pub struct FourSectionDictionary {
    shared: PlainFrontCoding,
    subjects: PlainFrontCoding,
    predicates: PlainFrontCoding,
    objects: PlainFrontCoding,
}

impl FourSectionDictionary {
    /// Builds a dictionary covering every subject, predicate, and object appearing in
    /// `triples`.
    #[must_use]
    pub fn build(triples: &[Triple]) -> Self {
        let mut subject_strings: BTreeSet<String> = BTreeSet::new();
        let mut predicate_strings: BTreeSet<String> = BTreeSet::new();
        let mut object_strings: BTreeSet<String> = BTreeSet::new();
        for triple in triples {
            subject_strings.insert(triple.subject.to_string());
            predicate_strings.insert(triple.predicate.to_string());
            object_strings.insert(triple.object.to_string());
        }

        // `BTreeSet::intersection` (and iteration in general) yields elements in ascending
        // order, so these are already valid `PlainFrontCoding::from_sorted_strings` inputs.
        let shared: Vec<String> = subject_strings
            .intersection(&object_strings)
            .cloned()
            .collect();
        let shared_set: BTreeSet<&str> = shared.iter().map(String::as_str).collect();
        let subjects: Vec<String> = subject_strings
            .iter()
            .filter(|s| !shared_set.contains(s.as_str()))
            .cloned()
            .collect();
        let objects: Vec<String> = object_strings
            .iter()
            .filter(|s| !shared_set.contains(s.as_str()))
            .cloned()
            .collect();
        let predicates: Vec<String> = predicate_strings.into_iter().collect();

        Self {
            shared: PlainFrontCoding::from_sorted_strings(&shared),
            subjects: PlainFrontCoding::from_sorted_strings(&subjects),
            predicates: PlainFrontCoding::from_sorted_strings(&predicates),
            objects: PlainFrontCoding::from_sorted_strings(&objects),
        }
    }

    /// The number of strings shared between the subject and object position.
    #[must_use]
    #[inline]
    pub fn shared_count(&self) -> u64 {
        self.shared.len() as u64
    }

    /// The total number of distinct subject IDs (`shared` + subject-exclusive strings).
    #[must_use]
    #[inline]
    pub fn subject_count(&self) -> u64 {
        self.shared_count() + self.subjects.len() as u64
    }

    /// The total number of distinct predicate IDs.
    #[must_use]
    #[inline]
    pub fn predicate_count(&self) -> u64 {
        self.predicates.len() as u64
    }

    /// The total number of distinct object IDs (`shared` + object-exclusive strings).
    #[must_use]
    #[inline]
    pub fn object_count(&self) -> u64 {
        self.shared_count() + self.objects.len() as u64
    }

    /// Returns `true` if this dictionary has no entries at all (equivalently, the dataset it
    /// was built from had no triples).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.shared.is_empty()
            && self.subjects.is_empty()
            && self.predicates.is_empty()
            && self.objects.is_empty()
    }

    /// Iterates over every distinct predicate IRI in the dictionary, in ascending
    /// lexicographic order of its dictionary string.
    ///
    /// A predicate string that fails to re-parse as a [`NamedNode`] (only possible for a
    /// corrupted or adversarially crafted file) is silently skipped.
    #[must_use = "this returns an iterator without enumerating it"]
    pub fn predicates(&self) -> impl Iterator<Item = NamedNode> + '_ {
        self.predicates
            .iter()
            .filter_map(|string| NamedNode::from_str(&string).ok())
    }

    /// Resolves a `1`-based subject ID to its term.
    ///
    /// # Errors
    ///
    /// Returns [`HdtError::IdOutOfRange`] if `id` is out of range, or
    /// [`HdtError::TermParse`] if the stored string is not a valid subject term.
    pub fn resolve_subject(&self, id: u32) -> Result<NamedOrBlankNode> {
        let shared_len = self.shared.len() as u32;
        let string = if id != 0 && id <= shared_len {
            self.shared.id_to_string(id)?
        } else {
            self.subjects.id_to_string(id.saturating_sub(shared_len))?
        };
        let term = Term::from_str(&string).map_err(|error| term_parse_error(&string, error))?;
        NamedOrBlankNode::try_from(term).map_err(|error| term_parse_error(&string, error))
    }

    /// Resolves a `1`-based predicate ID to its term.
    ///
    /// # Errors
    ///
    /// Returns [`HdtError::IdOutOfRange`] if `id` is out of range, or
    /// [`HdtError::TermParse`] if the stored string is not a valid IRI.
    pub fn resolve_predicate(&self, id: u32) -> Result<NamedNode> {
        let string = self.predicates.id_to_string(id)?;
        NamedNode::from_str(&string).map_err(|error| term_parse_error(&string, error))
    }

    /// Resolves a `1`-based object ID to its term.
    ///
    /// # Errors
    ///
    /// Returns [`HdtError::IdOutOfRange`] if `id` is out of range, or
    /// [`HdtError::TermParse`] if the stored string is not a valid term.
    pub fn resolve_object(&self, id: u32) -> Result<Term> {
        let shared_len = self.shared.len() as u32;
        let string = if id != 0 && id <= shared_len {
            self.shared.id_to_string(id)?
        } else {
            self.objects.id_to_string(id.saturating_sub(shared_len))?
        };
        Term::from_str(&string).map_err(|error| term_parse_error(&string, error))
    }

    /// Looks up the `1`-based subject ID for `term`, if it appears in the dictionary.
    #[must_use]
    pub fn subject_id(&self, term: NamedOrBlankNodeRef<'_>) -> Option<u32> {
        let string = term.to_string();
        self.shared.string_to_id(&string).or_else(|| {
            self.subjects
                .string_to_id(&string)
                .map(|id| id + self.shared.len() as u32)
        })
    }

    /// Looks up the `1`-based predicate ID for `term`, if it appears in the dictionary.
    #[must_use]
    pub fn predicate_id(&self, term: NamedNodeRef<'_>) -> Option<u32> {
        self.predicates.string_to_id(&term.to_string())
    }

    /// Looks up the `1`-based object ID for `term`, if it appears in the dictionary.
    #[must_use]
    pub fn object_id(&self, term: TermRef<'_>) -> Option<u32> {
        let string = term.to_string();
        self.shared.string_to_id(&string).or_else(|| {
            self.objects
                .string_to_id(&string)
                .map(|id| id + self.shared.len() as u32)
        })
    }

    /// Serializes this dictionary (its control information and all four sections), appending
    /// to `out`.
    pub fn write(&self, out: &mut Vec<u8>) {
        let elements =
            self.shared.len() + self.subjects.len() + self.predicates.len() + self.objects.len();
        let info = ControlInformation::new(ControlInfoType::Dictionary, DICTIONARY_FORMAT)
            .with_property("mapping", "1")
            .with_property("elements", elements.to_string());
        info.write(out);
        self.shared.write(out);
        self.subjects.write(out);
        self.predicates.write(out);
        self.objects.write(out);
    }

    /// Deserializes a dictionary previously written by [`write`](Self::write), starting at
    /// `*pos` and advancing it past the dictionary.
    ///
    /// # Errors
    ///
    /// Returns an [`HdtError`] if the control information is malformed, declares an unsupported
    /// `format` (e.g. the alternative `dictionaryPlain` encoding, which this crate does not
    /// implement) or mapping, or any of the four sections fail to parse.
    pub fn read(data: &[u8], pos: &mut usize) -> Result<Self> {
        let info = ControlInformation::read(data, pos, SECTION)?;
        if info.control_type != Some(ControlInfoType::Dictionary) {
            return Err(HdtError::UnexpectedControlType {
                section: SECTION,
                found: info.control_type.map_or(0, |type_| type_ as u8),
            });
        }
        if !info.format_matches(DICTIONARY_FORMAT) {
            return Err(HdtError::UnsupportedFormat {
                section: SECTION,
                found: info.format.clone(),
            });
        }
        let mapping = info.property_u64("mapping");
        if mapping != 0 && mapping != 1 {
            return Err(HdtError::UnsupportedFormat {
                section: SECTION,
                found: format!("mapping={mapping}"),
            });
        }

        let shared = PlainFrontCoding::read(data, pos, "dictionary/shared")?;
        let subjects = PlainFrontCoding::read(data, pos, "dictionary/subjects")?;
        let predicates = PlainFrontCoding::read(data, pos, "dictionary/predicates")?;
        let objects = PlainFrontCoding::read(data, pos, "dictionary/objects")?;
        Ok(Self {
            shared,
            subjects,
            predicates,
            objects,
        })
    }
}

/// Wraps a term-parsing failure with the offending dictionary string, for diagnostics.
fn term_parse_error(value: &str, error: impl std::fmt::Display) -> HdtError {
    HdtError::TermParse {
        value: value.to_owned(),
        message: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_model::{BlankNode, Literal, NamedNode};

    fn sample_triples() -> Vec<Triple> {
        let ex = |s: &str| NamedNode::new(format!("http://example.com/{s}")).expect("valid iri");
        vec![
            Triple::new(ex("a"), ex("knows"), ex("b")),
            Triple::new(ex("b"), ex("knows"), ex("a")),
            Triple::new(
                ex("a"),
                ex("name"),
                Literal::new_language_tagged_literal("Alice", "en").expect("valid tag"),
            ),
            Triple::new(BlankNode::new("x").expect("valid id"), ex("knows"), ex("a")),
        ]
    }

    #[test]
    fn shared_terms_are_deduplicated_across_sections() {
        let dictionary = FourSectionDictionary::build(&sample_triples());
        // "a" and "b" are both subjects and objects -> shared.
        assert_eq!(dictionary.shared_count(), 2);
        // subject-only: the blank node `_:x`.
        assert_eq!(dictionary.subject_count(), 3);
        // predicates: knows, name.
        assert_eq!(dictionary.predicate_count(), 2);
        // object-only: the literal "Alice"@en.
        assert_eq!(dictionary.object_count(), 3);
    }

    #[test]
    fn subject_and_object_ids_roundtrip() -> Result<()> {
        let triples = sample_triples();
        let dictionary = FourSectionDictionary::build(&triples);
        for triple in &triples {
            let s_id = dictionary
                .subject_id(triple.subject.as_ref())
                .expect("subject present");
            let p_id = dictionary
                .predicate_id(triple.predicate.as_ref())
                .expect("predicate present");
            let o_id = dictionary
                .object_id(triple.object.as_ref())
                .expect("object present");
            assert_eq!(dictionary.resolve_subject(s_id)?, triple.subject);
            assert_eq!(dictionary.resolve_predicate(p_id)?, triple.predicate);
            assert_eq!(dictionary.resolve_object(o_id)?, triple.object);
        }
        Ok(())
    }

    #[test]
    fn predicates_enumerates_distinct_predicates_sorted() {
        let dictionary = FourSectionDictionary::build(&sample_triples());
        let names: Vec<String> = dictionary
            .predicates()
            .map(|p| p.into_string().into())
            .collect();
        assert_eq!(
            names,
            vec![
                "http://example.com/knows".to_owned(),
                "http://example.com/name".to_owned(),
            ]
        );
    }

    #[test]
    fn is_empty_reflects_dataset() {
        assert!(FourSectionDictionary::build(&[]).is_empty());
        assert!(!FourSectionDictionary::build(&sample_triples()).is_empty());
    }

    #[test]
    fn unknown_terms_have_no_id() {
        let dictionary = FourSectionDictionary::build(&sample_triples());
        let unknown = NamedNode::new("http://example.com/nope").expect("valid iri");
        assert_eq!(dictionary.subject_id(unknown.as_ref().into()), None);
        assert_eq!(dictionary.predicate_id(unknown.as_ref()), None);
        assert_eq!(dictionary.object_id(unknown.as_ref().into()), None);
    }

    #[test]
    fn wire_roundtrip() -> Result<()> {
        let triples = sample_triples();
        let dictionary = FourSectionDictionary::build(&triples);
        let mut buffer = Vec::new();
        dictionary.write(&mut buffer);
        let mut pos = 0;
        let decoded = FourSectionDictionary::read(&buffer, &mut pos)?;
        assert_eq!(pos, buffer.len());
        assert_eq!(decoded.shared_count(), dictionary.shared_count());
        assert_eq!(decoded.subject_count(), dictionary.subject_count());
        assert_eq!(decoded.predicate_count(), dictionary.predicate_count());
        assert_eq!(decoded.object_count(), dictionary.object_count());
        for triple in &triples {
            let s_id = decoded
                .subject_id(triple.subject.as_ref())
                .expect("subject present");
            assert_eq!(decoded.resolve_subject(s_id)?, triple.subject);
        }
        Ok(())
    }

    #[test]
    fn empty_dictionary_roundtrip() -> Result<()> {
        let dictionary = FourSectionDictionary::build(&[]);
        let mut buffer = Vec::new();
        dictionary.write(&mut buffer);
        let mut pos = 0;
        let decoded = FourSectionDictionary::read(&buffer, &mut pos)?;
        assert_eq!(decoded.subject_count(), 0);
        assert_eq!(decoded.predicate_count(), 0);
        assert_eq!(decoded.object_count(), 0);
        Ok(())
    }

    /// A dictionary section whose `ControlInformation` declares a `format` this crate does not
    /// implement (e.g. the alternative `dictionaryPlain` encoding) must be rejected with a clear
    /// [`HdtError::UnsupportedFormat`], rather than being handed to the `PlainFrontCoding`
    /// decoder it does not actually describe.
    #[test]
    fn read_rejects_an_unrecognized_format() {
        let mut buffer = Vec::new();
        ControlInformation::new(
            ControlInfoType::Dictionary,
            "<http://example.com/dictionaryPlain>",
        )
        .with_property("mapping", "1")
        .write(&mut buffer);
        let mut pos = 0;
        let error = FourSectionDictionary::read(&buffer, &mut pos).expect_err("expected an error");
        assert!(matches!(
            error,
            HdtError::UnsupportedFormat {
                section: SECTION,
                ..
            }
        ));
    }
}
