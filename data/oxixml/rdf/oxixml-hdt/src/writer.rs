//! [`HdtWriter`] — serializes RDF triples into a complete, spec-compliant HDT file.

use std::io::Write;

use oxixml_model::Triple;

use crate::control_info::{ControlInfoType, ControlInformation};
use crate::dictionary::FourSectionDictionary;
use crate::error::Result;
use crate::hdt::GLOBAL_FORMAT;
use crate::header::Header;
use crate::triples::{BitmapTriples, resolve_id_triples};

/// Writes RDF triples out as a complete HDT file: global control information, header,
/// [`FourSectionDictionary`], and [`BitmapTriples`], in that order.
///
/// ```
/// use oxixml_hdt::HdtWriter;
/// use oxixml_model::{NamedNode, Triple};
///
/// let ex = NamedNode::new("http://example.com/s")?;
/// let triple = Triple::new(ex.clone(), ex.clone(), ex);
///
/// let mut bytes = Vec::new();
/// HdtWriter::new().write(&mut bytes, vec![triple])?;
/// assert!(bytes.starts_with(b"$HDT"));
/// # Result::<(), Box<dyn std::error::Error>>::Ok(())
/// ```
#[derive(Debug, Clone, Default)]
pub struct HdtWriter {
    header: Header,
}

impl HdtWriter {
    /// Creates a writer with an empty header payload.
    #[must_use]
    pub fn new() -> Self {
        Self {
            header: Header::empty(),
        }
    }

    /// Creates a writer that embeds `header` (conventionally, N-Triples-formatted dataset
    /// metadata) as the file's header payload.
    #[must_use]
    pub fn with_header(header: impl Into<Vec<u8>>) -> Self {
        Self {
            header: Header::new(header),
        }
    }

    /// Writes every triple in `triples` (a [`oxixml_model::Graph`]'s
    /// [`iter`](oxixml_model::Graph::iter), a plain [`Vec<Triple>`], or any other triple
    /// source) out to `writer` as a complete HDT file.
    ///
    /// Input order does not matter and duplicates are harmless: triples are deduplicated and
    /// sorted into SPO order internally, as the format requires.
    ///
    /// # Errors
    ///
    /// Returns an [`crate::error::HdtError`] if writing to `writer` fails.
    pub fn write<W: Write>(
        &self,
        mut writer: W,
        triples: impl IntoIterator<Item = Triple>,
    ) -> Result<()> {
        let triples: Vec<Triple> = triples.into_iter().collect();
        let dictionary = FourSectionDictionary::build(&triples);
        let id_triples = resolve_id_triples(&triples, &dictionary);
        let bitmap_triples = BitmapTriples::build(&id_triples);

        let mut out = Vec::new();
        ControlInformation::new(ControlInfoType::Global, GLOBAL_FORMAT).write(&mut out);
        self.header.write(&mut out);
        dictionary.write(&mut out);
        bitmap_triples.write(&mut out)?;

        writer.write_all(&out)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdt::Hdt;
    use oxixml_model::NamedNode;

    #[test]
    fn writes_a_recognizable_hdt_file() -> Result<()> {
        let ex = NamedNode::new("http://example.com/s").expect("valid iri");
        let triple = Triple::new(ex.clone(), ex.clone(), ex);
        let mut bytes = Vec::new();
        HdtWriter::new().write(&mut bytes, vec![triple])?;
        assert!(bytes.starts_with(b"$HDT"));
        Ok(())
    }

    #[test]
    fn round_trips_through_hdt_from_bytes() -> Result<()> {
        let ex = |s: &str| NamedNode::new(format!("http://example.com/{s}")).expect("valid iri");
        let triples = vec![
            Triple::new(ex("a"), ex("p"), ex("b")),
            Triple::new(ex("b"), ex("p"), ex("a")),
        ];
        let mut bytes = Vec::new();
        HdtWriter::with_header(&b"<urn:x> <urn:y> <urn:z> .\n"[..])
            .write(&mut bytes, triples.clone())?;

        let hdt = Hdt::from_bytes(&bytes)?;
        assert_eq!(hdt.header(), b"<urn:x> <urn:y> <urn:z> .\n");
        let mut roundtripped: Vec<Triple> = hdt.triples().collect::<Result<_>>()?;
        let mut expected = triples;
        roundtripped.sort_by_key(|t| t.to_string());
        expected.sort_by_key(|t| t.to_string());
        assert_eq!(roundtripped, expected);
        Ok(())
    }

    #[test]
    fn empty_dataset_round_trips() -> Result<()> {
        let mut bytes = Vec::new();
        HdtWriter::new().write(&mut bytes, Vec::<Triple>::new())?;
        let hdt = Hdt::from_bytes(&bytes)?;
        assert!(hdt.is_empty());
        assert_eq!(hdt.triples().count(), 0);
        Ok(())
    }

    #[test]
    fn duplicate_input_triples_are_deduplicated() -> Result<()> {
        let ex = NamedNode::new("http://example.com/s").expect("valid iri");
        let triple = Triple::new(ex.clone(), ex.clone(), ex);
        let mut bytes = Vec::new();
        HdtWriter::new().write(&mut bytes, vec![triple.clone(), triple.clone(), triple])?;
        let hdt = Hdt::from_bytes(&bytes)?;
        assert_eq!(hdt.len(), 1);
        Ok(())
    }
}
