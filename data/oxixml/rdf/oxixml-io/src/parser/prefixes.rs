//! [`PrefixesIter`]: the IRI prefixes declared so far by a parse in progress.

use oxixml_jsonld::JsonLdPrefixesIter;
use oxixml_rdfxml::RdfXmlPrefixesIter;
use oxixml_turtle::n3::N3PrefixesIter;
use oxixml_turtle::trig::TriGPrefixesIter;
use oxixml_turtle::turtle::TurtlePrefixesIter;

/// Iterator over the `(prefix name, prefix IRI)` pairs declared so far by a parse in progress.
///
/// See [`ReaderQuadParser::prefixes`](crate::ReaderQuadParser::prefixes) and
/// [`SliceQuadParser::prefixes`](crate::SliceQuadParser::prefixes).
///
/// Empty for N-Quads, N-Triples, and TriX: none of these formats has a notion of a namespace
/// prefix.
pub struct PrefixesIter<'a> {
    pub(super) inner: PrefixesIterKind<'a>,
}

pub(super) enum PrefixesIterKind<'a> {
    JsonLd(JsonLdPrefixesIter<'a>),
    N3(N3PrefixesIter<'a>),
    RdfXml(RdfXmlPrefixesIter<'a>),
    TriG(TriGPrefixesIter<'a>),
    Turtle(TurtlePrefixesIter<'a>),
    None,
}

impl<'a> Iterator for PrefixesIter<'a> {
    type Item = (&'a str, &'a str);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            PrefixesIterKind::JsonLd(iter) => iter.next(),
            PrefixesIterKind::N3(iter) => iter.next(),
            PrefixesIterKind::RdfXml(iter) => iter.next(),
            PrefixesIterKind::TriG(iter) => iter.next(),
            PrefixesIterKind::Turtle(iter) => iter.next(),
            PrefixesIterKind::None => None,
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.inner {
            PrefixesIterKind::JsonLd(iter) => iter.size_hint(),
            PrefixesIterKind::N3(iter) => iter.size_hint(),
            PrefixesIterKind::RdfXml(iter) => iter.size_hint(),
            PrefixesIterKind::TriG(iter) => iter.size_hint(),
            PrefixesIterKind::Turtle(iter) => iter.size_hint(),
            PrefixesIterKind::None => (0, Some(0)),
        }
    }
}
