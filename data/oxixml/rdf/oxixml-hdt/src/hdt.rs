//! [`Hdt`] — a fully-loaded, queryable HDT dataset.

use std::io::Read;

use oxixml_model::{NamedNodeRef, NamedOrBlankNodeRef, TermRef, Triple};

use crate::control_info::{ControlInfoType, ControlInformation};
use crate::dictionary::FourSectionDictionary;
use crate::error::{HdtError, Result};
use crate::header::Header;
use crate::triples::BitmapTriples;

/// The section name used in diagnostics for the file-level control information.
const SECTION: &str = "global";

/// The global `ControlInformation`'s `format` value: the HDT container format IRI.
pub(crate) const GLOBAL_FORMAT: &str = "<http://purl.org/HDT/hdt#HDTv1>";

/// A fully-loaded HDT dataset: header, dictionary, and triples, all held in memory.
///
/// Built with [`Hdt::from_reader`] or [`Hdt::from_bytes`]; queried with [`Hdt::triples`] and
/// [`Hdt::triples_matching`].
///
/// ```
/// use oxixml_hdt::{Hdt, HdtWriter};
/// use oxixml_model::{NamedNode, Triple};
///
/// let ex = |s: &str| NamedNode::new(format!("http://example.com/{s}")).expect("valid iri");
/// let triple = Triple::new(ex("alice"), ex("knows"), ex("bob"));
///
/// let mut bytes = Vec::new();
/// HdtWriter::new().write(&mut bytes, vec![triple.clone()])?;
///
/// let hdt = Hdt::from_bytes(&bytes)?;
/// assert_eq!(hdt.len(), 1);
/// let roundtripped: Vec<Triple> = hdt.triples().collect::<Result<_, _>>()?;
/// assert_eq!(roundtripped, vec![triple]);
/// # Result::<(), Box<dyn std::error::Error>>::Ok(())
/// ```
#[derive(Debug, Clone, Default)]
pub struct Hdt {
    header: Header,
    dictionary: FourSectionDictionary,
    triples: BitmapTriples,
}

impl Hdt {
    /// Reads a complete HDT file from `reader` into memory.
    ///
    /// This crate never memory-maps files: the entire byte stream is buffered (via
    /// [`Read::read_to_end`]) before parsing begins.
    ///
    /// # Errors
    ///
    /// See [`Hdt::from_bytes`].
    pub fn from_reader<R: Read>(mut reader: R) -> Result<Self> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        Self::from_bytes(&bytes)
    }

    /// Parses a complete HDT file already held in memory.
    ///
    /// # Errors
    ///
    /// Returns an [`HdtError`] if the file does not start with the `$HDT` cookie, any section's
    /// checksum fails to verify, the file is structurally malformed or truncated, or a section
    /// declares a `format` this crate does not implement (e.g. the alternative `triplesList` or
    /// `dictionaryPlain` encodings some other HDT tools emit).
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let mut pos = 0;
        let global = ControlInformation::read(data, &mut pos, SECTION)?;
        if global.control_type != Some(ControlInfoType::Global) {
            return Err(HdtError::UnexpectedControlType {
                section: SECTION,
                found: global.control_type.map_or(0, |type_| type_ as u8),
            });
        }
        if !global.format_matches(GLOBAL_FORMAT) {
            return Err(HdtError::UnsupportedFormat {
                section: SECTION,
                found: global.format.clone(),
            });
        }
        let header = Header::read(data, &mut pos)?;
        let dictionary = FourSectionDictionary::read(data, &mut pos)?;
        let triples = BitmapTriples::read(data, &mut pos)?;
        Ok(Self {
            header,
            dictionary,
            triples,
        })
    }

    /// The total number of triples in this dataset.
    #[must_use]
    #[inline]
    pub const fn len(&self) -> u64 {
        self.triples.len()
    }

    /// Returns `true` if this dataset has no triples.
    #[must_use]
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.triples.is_empty()
    }

    /// The raw bytes of the (`ntriples`-formatted) header payload.
    #[must_use]
    #[inline]
    pub fn header(&self) -> &[u8] {
        self.header.as_bytes()
    }

    /// The dictionary backing this dataset, for direct string↔ID lookups.
    #[must_use]
    #[inline]
    pub fn dictionary(&self) -> &FourSectionDictionary {
        &self.dictionary
    }

    /// Iterates over every triple in the dataset (SPO order).
    #[must_use]
    pub fn triples(&self) -> HdtTriplesIter<'_> {
        self.triples_matching(None, None, None)
    }

    /// Iterates over every triple matching the given (optionally bound) subject / predicate /
    /// object pattern.
    ///
    /// A term that does not appear in the dictionary at all makes the whole pattern match
    /// nothing (rather than erroring), matching a `graph.triples_for_pattern`-style query.
    #[must_use]
    pub fn triples_matching<'a>(
        &'a self,
        subject: Option<NamedOrBlankNodeRef<'a>>,
        predicate: Option<NamedNodeRef<'a>>,
        object: Option<TermRef<'a>>,
    ) -> HdtTriplesIter<'a> {
        let subject_id = subject.and_then(|term| self.dictionary.subject_id(term));
        let predicate_id = predicate.and_then(|term| self.dictionary.predicate_id(term));
        let object_id = object.and_then(|term| self.dictionary.object_id(term));

        let unresolvable = (subject.is_some() && subject_id.is_none())
            || (predicate.is_some() && predicate_id.is_none())
            || (object.is_some() && object_id.is_none());

        let inner: Box<dyn Iterator<Item = (u32, u32, u32)> + 'a> = if unresolvable {
            Box::new(std::iter::empty())
        } else {
            self.triples
                .matching_ids(subject_id, predicate_id, object_id)
        };

        HdtTriplesIter { hdt: self, inner }
    }
}

/// An iterator over resolved [`Triple`]s produced by [`Hdt::triples`] or
/// [`Hdt::triples_matching`].
///
/// Yields [`Result`] rather than bare [`Triple`]s because dictionary resolution is not proven
/// infallible at the type level: a structurally valid but adversarially crafted file could
/// still reference IDs that, while in range, decode to non-UTF-8 or otherwise malformed term
/// strings.
pub struct HdtTriplesIter<'a> {
    hdt: &'a Hdt,
    inner: Box<dyn Iterator<Item = (u32, u32, u32)> + 'a>,
}

impl Iterator for HdtTriplesIter<'_> {
    type Item = Result<Triple>;

    fn next(&mut self) -> Option<Self::Item> {
        let (subject_id, predicate_id, object_id) = self.inner.next()?;
        Some(self.resolve(subject_id, predicate_id, object_id))
    }
}

impl HdtTriplesIter<'_> {
    /// Resolves one ID triple into an RDF [`Triple`].
    fn resolve(&self, subject_id: u32, predicate_id: u32, object_id: u32) -> Result<Triple> {
        let subject = self.hdt.dictionary.resolve_subject(subject_id)?;
        let predicate = self.hdt.dictionary.resolve_predicate(predicate_id)?;
        let object = self.hdt.dictionary.resolve_object(object_id)?;
        Ok(Triple::new(subject, predicate, object))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::writer::HdtWriter;
    use oxixml_model::NamedNode;

    fn ex(name: &str) -> NamedNode {
        NamedNode::new(format!("http://example.com/{name}")).expect("valid iri")
    }

    fn sample_triples() -> Vec<Triple> {
        vec![
            Triple::new(ex("a"), ex("knows"), ex("b")),
            Triple::new(ex("a"), ex("knows"), ex("c")),
            Triple::new(ex("b"), ex("knows"), ex("a")),
        ]
    }

    #[test]
    fn triples_matching_by_subject() -> Result<()> {
        let mut bytes = Vec::new();
        HdtWriter::new().write(&mut bytes, sample_triples())?;
        let hdt = Hdt::from_bytes(&bytes)?;

        let a = ex("a");
        let matched: Vec<Triple> = hdt
            .triples_matching(Some(a.as_ref().into()), None, None)
            .collect::<Result<_>>()?;
        assert_eq!(matched.len(), 2);
        Ok(())
    }

    #[test]
    fn triples_matching_unknown_term_yields_nothing() -> Result<()> {
        let mut bytes = Vec::new();
        HdtWriter::new().write(&mut bytes, sample_triples())?;
        let hdt = Hdt::from_bytes(&bytes)?;

        let unknown = ex("does-not-exist");
        let matched: Vec<Triple> = hdt
            .triples_matching(Some(unknown.as_ref().into()), None, None)
            .collect::<Result<_>>()?;
        assert!(matched.is_empty());
        Ok(())
    }

    #[test]
    fn from_bytes_rejects_garbage() {
        let error = Hdt::from_bytes(b"not an hdt file at all").expect_err("expected an error");
        assert!(matches!(error, HdtError::InvalidCookie { .. }));
    }

    /// A file whose global `ControlInformation` declares a `format` this crate does not
    /// recognize must be rejected with a clear [`HdtError::UnsupportedFormat`] naming the
    /// offending value, rather than being accepted and later misread by the header/dictionary/
    /// triples decoders (which assume the declared format).
    #[test]
    fn from_bytes_rejects_an_unrecognized_global_format() {
        let mut buffer = Vec::new();
        ControlInformation::new(
            ControlInfoType::Global,
            "<http://example.com/totally-not-hdt-v1>",
        )
        .write(&mut buffer);
        let error = Hdt::from_bytes(&buffer).expect_err("expected an error");
        assert!(matches!(
            error,
            HdtError::UnsupportedFormat {
                section: SECTION,
                ..
            }
        ));
    }

    /// The bracket-vs-bare-IRI leniency (see [`ControlInformation::format_matches`]) applies at
    /// the global check too: a file is not rejected merely for spelling the format IRI without
    /// its surrounding angle brackets.
    #[test]
    fn from_bytes_accepts_the_bare_global_format() -> Result<()> {
        let mut bytes = Vec::new();
        HdtWriter::new().write(&mut bytes, sample_triples())?;
        // `GLOBAL_FORMAT` is written with angle brackets; replace it with the bare IRI (same
        // length reasoning does not apply here, so re-serialize the global block from scratch).
        let mut pos = 0;
        let mut global = ControlInformation::read(&bytes, &mut pos, SECTION)?;
        global.format = GLOBAL_FORMAT
            .trim_start_matches('<')
            .trim_end_matches('>')
            .to_owned();
        let mut rebuilt = Vec::new();
        global.write(&mut rebuilt);
        rebuilt.extend_from_slice(&bytes[pos..]);

        let hdt = Hdt::from_bytes(&rebuilt)?;
        assert_eq!(hdt.len(), sample_triples().len() as u64);
        Ok(())
    }
}
